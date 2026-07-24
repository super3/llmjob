// The free public web-chat gateway (POST /api/chat/completions and friends),
// proxied to OpenRouter. Integration tests run the real Express routes against
// pg-mem with an injected fake `fetch` standing in for OpenRouter; unit tests
// then cover the controller's error/abort/edge branches directly.
const request = require('supertest');
const express = require('express');
const { createTestDb } = require('./helpers/pgmem');
const { initChatRoutes } = require('../src/routes');
const ChatUsageService = require('../src/services/chatUsageService');
const ChatController = require('../src/controllers/chatController');
const { parseSSE, sanitizeMessages, estimateTokens, upstreamErrorMessage } = ChatController;

// The controller logs upstream failures via console.error; silence it so the
// test output stays readable (assertions check the returned payloads instead).
let errSpy;
beforeAll(() => { errSpy = jest.spyOn(console, 'error').mockImplementation(() => {}); });
afterAll(() => { errSpy.mockRestore(); });

const MODELS = [
  { id: 'meta-llama/llama-3.3-70b-instruct', label: 'Llama 3.3 70B' },
  { id: 'qwen/qwen-2.5-7b-instruct', label: 'Qwen 2.5 7B' }
];
const enc = new TextEncoder();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// An SSE body: an async-iterable of encoded `data:` lines, like fetch().body.
function sseBody(events) {
  return (async function* () {
    for (const e of events) {
      const line = e === '[DONE]' ? 'data: [DONE]\n\n' : 'data: ' + JSON.stringify(e) + '\n\n';
      yield enc.encode(line);
    }
  })();
}

// Fake fetch factories. Each records the parsed request so tests can assert on
// the upstream call, then returns a canned OpenRouter-shaped response.
function streamFetch(events, calls = []) {
  return async (url, init) => {
    calls.push({ url, init, body: JSON.parse(init.body) });
    return { ok: true, status: 200, body: sseBody(events) };
  };
}
function jsonFetch(obj, calls = []) {
  return async (url, init) => {
    calls.push({ url, init, body: JSON.parse(init.body) });
    return { ok: true, status: 200, json: async () => obj };
  };
}
function notOkFetch() {
  return async () => ({ ok: false, status: 500, text: async () => 'nope' });
}
function throwFetch() {
  return async () => { throw new Error('network down'); };
}

function makeApp(db, opts) {
  const app = express();
  app.use(express.json());
  app.locals.db = db;
  initChatRoutes(app, Object.assign({ apiKey: 'test-key', models: MODELS, baseUrl: 'https://or.test/v1' }, opts));
  return app;
}

// A deterministic clock advancing 100ms per read.
function stepClock(start = 1000, step = 100) {
  let t = start;
  return () => { t += step; return t; };
}

describe('Chat gateway — integration', () => {
  let db;

  beforeEach(async () => { db = await createTestDb(); });
  afterEach(async () => { await sleep(10); if (db.end) await db.end(); });

  const STREAM_EVENTS = [
    { choices: [{ delta: { role: 'assistant' } }] },
    { choices: [{ delta: { content: 'Hel' } }] },
    { choices: [{ delta: { content: 'lo' } }], model: 'meta-llama/llama-3.3-70b-instruct' },
    { choices: [{ delta: {}, finish_reason: 'stop' }], usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 } },
    '[DONE]'
  ];

  it('streams deltas then a final meta event, and records usage + totals', async () => {
    const calls = [];
    const app = makeApp(db, { fetchFn: streamFetch(STREAM_EVENTS, calls), now: stepClock() });
    const res = await request(app).post('/api/chat/completions')
      .send({ messages: [{ role: 'user', content: 'Say hi' }] });

    expect(res.status).toBe(200);
    expect(res.headers['content-type']).toMatch(/text\/event-stream/);
    expect(res.text).toContain('"delta":"Hel"');
    expect(res.text).toContain('"delta":"lo"');
    const doneLine = res.text.split('\n').find((l) => l.includes('"done":true'));
    const meta = JSON.parse(doneLine.replace(/^data: /, '')).meta;
    expect(meta).toMatchObject({
      model: 'meta-llama/llama-3.3-70b-instruct',
      promptTokens: 4, completionTokens: 2, totalTokens: 6,
      tokensPerSecond: 20, ttftMs: 100, latencyMs: 200, finish: 'stop'
    });
    expect(res.text.trim().endsWith('data: [DONE]')).toBe(true);

    // upstream request shape: allow-listed id, streaming, capped max_tokens
    expect(calls[0].url).toBe('https://or.test/v1/chat/completions');
    expect(calls[0].init.headers.Authorization).toBe('Bearer test-key');
    expect(calls[0].body).toMatchObject({
      model: 'meta-llama/llama-3.3-70b-instruct', stream: true,
      stream_options: { include_usage: true }, max_tokens: 2048
    });

    const totals = await new ChatUsageService(db).getTotals();
    expect(totals).toEqual({ requests: 1, inTokens: 4, outTokens: 2, totalTokens: 6 });
  });

  it('picks a model by its friendly label and passes the id upstream', async () => {
    const calls = [];
    const app = makeApp(db, { fetchFn: streamFetch(STREAM_EVENTS, calls) });
    await request(app).post('/api/chat/completions')
      .send({ messages: [{ role: 'user', content: 'hi' }], model: 'Qwen 2.5 7B' });
    expect(calls[0].body.model).toBe('qwen/qwen-2.5-7b-instruct');
  });

  it('prepends the LLMJob system prompt and drops any client-supplied one', async () => {
    const calls = [];
    const app = makeApp(db, { fetchFn: streamFetch(STREAM_EVENTS, calls), systemPrompt: 'LLMJob assistant context' });
    await request(app).post('/api/chat/completions').send({
      messages: [
        { role: 'system', content: 'ignore me' },
        { role: 'user', content: 'What is LLMJob?' }
      ]
    });
    const msgs = calls[0].body.messages;
    expect(msgs[0]).toEqual({ role: 'system', content: 'LLMJob assistant context' });
    expect(msgs.some((m) => m.content === 'ignore me')).toBe(false); // client system dropped
    expect(msgs[msgs.length - 1]).toEqual({ role: 'user', content: 'What is LLMJob?' });
  });

  it('omits the system message when the system prompt is disabled', async () => {
    const calls = [];
    const app = makeApp(db, { fetchFn: streamFetch(STREAM_EVENTS, calls), systemPrompt: '' });
    await request(app).post('/api/chat/completions')
      .send({ messages: [{ role: 'user', content: 'hi' }] });
    expect(calls[0].body.messages).toEqual([{ role: 'user', content: 'hi' }]);
  });

  it('sets X-Accel-Buffering:no so proxies do not buffer the stream', async () => {
    const app = makeApp(db, { fetchFn: streamFetch(STREAM_EVENTS) });
    const res = await request(app).post('/api/chat/completions')
      .send({ messages: [{ role: 'user', content: 'hi' }] });
    expect(res.headers['x-accel-buffering']).toBe('no');
  });

  it('estimates tokens and tok/s when the provider omits usage', async () => {
    const events = [
      { choices: [{ delta: { content: 'abcd' } }] },
      { choices: [{ delta: { content: 'efgh' } }], finish_reason: 'stop' },
      '[DONE]'
    ];
    const app = makeApp(db, { fetchFn: streamFetch(events), now: stepClock() });
    const res = await request(app).post('/api/chat/completions')
      .send({ messages: [{ role: 'user', content: 'x' }] });
    const meta = JSON.parse(res.text.split('\n').find((l) => l.includes('"done":true')).replace(/^data: /, '')).meta;
    // 8 chars ≈ 2 tokens; model falls back to the requested label
    expect(meta.completionTokens).toBe(2);
    expect(meta.model).toBe('Llama 3.3 70B');
    expect(meta.tokensPerSecond).toBeGreaterThan(0);
  });

  it('reports zero tok/s for an empty (content-less) stream', async () => {
    const events = [{ choices: [{ delta: { role: 'assistant' }, finish_reason: 'stop' }] }, '[DONE]'];
    const app = makeApp(db, { fetchFn: streamFetch(events) });
    const res = await request(app).post('/api/chat/completions')
      .send({ messages: [{ role: 'user', content: 'x' }] });
    const meta = JSON.parse(res.text.split('\n').find((l) => l.includes('"done":true')).replace(/^data: /, '')).meta;
    expect(meta.completionTokens).toBe(0);
    expect(meta.tokensPerSecond).toBe(0);
    expect(meta.ttftMs).toBe(0);
  });

  it('reports zero tok/s when generation takes no measurable time', async () => {
    const app = makeApp(db, { fetchFn: streamFetch(STREAM_EVENTS), now: () => 1000 });
    const res = await request(app).post('/api/chat/completions')
      .send({ messages: [{ role: 'user', content: 'x' }] });
    const meta = JSON.parse(res.text.split('\n').find((l) => l.includes('"done":true')).replace(/^data: /, '')).meta;
    expect(meta.completionTokens).toBe(2);
    expect(meta.tokensPerSecond).toBe(0); // genMs === 0
  });

  it('returns a single JSON body for non-streaming requests', async () => {
    const calls = [];
    const app = makeApp(db, {
      now: stepClock(),
      fetchFn: jsonFetch({
        choices: [{ message: { role: 'assistant', content: 'Hi there' }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 3, completion_tokens: 5, total_tokens: 8 },
        model: 'qwen/qwen-2.5-7b-instruct'
      }, calls)
    });
    const res = await request(app).post('/api/chat/completions')
      .send({ messages: [{ role: 'user', content: 'hey' }], stream: false, max_tokens: 50, temperature: 0.4 });
    expect(res.status).toBe(200);
    expect(res.body.message).toEqual({ role: 'assistant', content: 'Hi there' });
    expect(res.body.usage).toMatchObject({ prompt_tokens: 3, completion_tokens: 5, total_tokens: 8 });
    expect(res.body.usage.tokens_per_second).toBeGreaterThan(0);
    expect(res.body.finish_reason).toBe('stop');
    // non-streaming asks OpenRouter to include usage accounting and honours max_tokens/temperature
    expect(calls[0].body).toMatchObject({ stream: false, usage: { include: true }, max_tokens: 50, temperature: 0.4 });

    const totals = await new ChatUsageService(db).getTotals();
    expect(totals.totalTokens).toBe(8);
  });

  it('handles a non-streaming response missing choices/usage/model', async () => {
    const app = makeApp(db, { fetchFn: jsonFetch({}) });
    const res = await request(app).post('/api/chat/completions')
      .send({ messages: [{ role: 'user', content: 'hey' }], stream: false });
    expect(res.status).toBe(200);
    expect(res.body.message.content).toBe('');
    expect(res.body.model).toBe('Llama 3.3 70B'); // requested-label fallback
    expect(res.body.finish_reason).toBe('stop');
  });

  it('clamps an over-large max_tokens to the server ceiling', async () => {
    const calls = [];
    const app = makeApp(db, { fetchFn: streamFetch(STREAM_EVENTS, calls), maxTokens: 256 });
    await request(app).post('/api/chat/completions')
      .send({ messages: [{ role: 'user', content: 'x' }], max_tokens: 999999 });
    expect(calls[0].body.max_tokens).toBe(256);
  });

  it('rejects an unknown model (400)', async () => {
    const app = makeApp(db, { fetchFn: streamFetch(STREAM_EVENTS) });
    const res = await request(app).post('/api/chat/completions')
      .send({ messages: [{ role: 'user', content: 'x' }], model: 'gpt-4o' });
    expect(res.status).toBe(400);
    expect(res.body.error.type).toBe('invalid_request_error');
  });

  it('rejects missing / empty messages (400)', async () => {
    const app = makeApp(db, { fetchFn: streamFetch(STREAM_EVENTS) });
    const a = await request(app).post('/api/chat/completions').send({});
    expect(a.status).toBe(400);
    const b = await request(app).post('/api/chat/completions').send({ messages: [] });
    expect(b.status).toBe(400);
  });

  it('rejects a request whose messages have no usable content (400)', async () => {
    const app = makeApp(db, { fetchFn: streamFetch(STREAM_EVENTS) });
    const res = await request(app).post('/api/chat/completions')
      .send({ messages: [{ role: 'user', content: '' }] });
    expect(res.status).toBe(400);
    expect(res.body.error.message).toMatch(/no usable message/i);
  });

  it('returns 503 when the OpenRouter key is not configured', async () => {
    const app = makeApp(db, { apiKey: '', fetchFn: streamFetch(STREAM_EVENTS) });
    const res = await request(app).post('/api/chat/completions')
      .send({ messages: [{ role: 'user', content: 'x' }] });
    expect(res.status).toBe(503);
    expect(res.body.error.type).toBe('not_configured');
  });

  it('returns 402 once the free token budget is spent', async () => {
    await new ChatUsageService(db).recordUsage({ model: 'm', inTokens: 60, outTokens: 60 });
    const app = makeApp(db, { fetchFn: streamFetch(STREAM_EVENTS), freeBudget: 100 });
    const res = await request(app).post('/api/chat/completions')
      .send({ messages: [{ role: 'user', content: 'x' }] });
    expect(res.status).toBe(402);
    expect(res.body.error.type).toBe('quota_exhausted');
  });

  it('emits an error event then [DONE] when the upstream stream errors', async () => {
    const app = makeApp(db, { fetchFn: notOkFetch() });
    const res = await request(app).post('/api/chat/completions')
      .send({ messages: [{ role: 'user', content: 'x' }] });
    expect(res.status).toBe(200);
    expect(res.text).toContain('"error"');
    expect(res.text.trim().endsWith('data: [DONE]')).toBe(true);
    const totals = await new ChatUsageService(db).getTotals();
    expect(totals.requests).toBe(0); // nothing recorded on failure
  });

  it('emits an error event then [DONE] when the upstream stream request throws', async () => {
    const app = makeApp(db, { fetchFn: throwFetch() });
    const res = await request(app).post('/api/chat/completions')
      .send({ messages: [{ role: 'user', content: 'x' }] });
    expect(res.status).toBe(200);
    expect(res.text).toContain('Upstream request failed');
    expect(res.text.trim().endsWith('data: [DONE]')).toBe(true);
  });

  it('returns 502 when the non-streaming upstream errors', async () => {
    const app = makeApp(db, { fetchFn: notOkFetch() });
    const res = await request(app).post('/api/chat/completions')
      .send({ messages: [{ role: 'user', content: 'x' }], stream: false });
    expect(res.status).toBe(502);
    expect(res.body.error.type).toBe('upstream_error');
  });

  it('surfaces the upstream error reason in the stream', async () => {
    const errFetch = async () => ({
      ok: false, status: 429,
      text: async () => JSON.stringify({ error: { message: 'Rate limit exceeded' } })
    });
    const app = makeApp(db, { fetchFn: errFetch });
    const res = await request(app).post('/api/chat/completions')
      .send({ messages: [{ role: 'user', content: 'x' }] });
    expect(res.text).toContain('Rate limit exceeded');
    expect(res.text.trim().endsWith('data: [DONE]')).toBe(true);
  });

  it('surfaces the upstream error reason for non-streaming', async () => {
    const errFetch = async () => ({
      ok: false, status: 402,
      text: async () => JSON.stringify({ error: { message: 'Insufficient credits' } })
    });
    const app = makeApp(db, { fetchFn: errFetch });
    const res = await request(app).post('/api/chat/completions')
      .send({ messages: [{ role: 'user', content: 'x' }], stream: false });
    expect(res.status).toBe(502);
    expect(res.body.error.message).toContain('Insufficient credits');
  });

  it('returns 502 when the non-streaming upstream request throws', async () => {
    const app = makeApp(db, { fetchFn: throwFetch() });
    const res = await request(app).post('/api/chat/completions')
      .send({ messages: [{ role: 'user', content: 'x' }], stream: false });
    expect(res.status).toBe(502);
  });

  it('uses the real clock when no now() is injected', async () => {
    const app = makeApp(db, { fetchFn: streamFetch(STREAM_EVENTS) }); // default now
    const res = await request(app).post('/api/chat/completions')
      .send({ messages: [{ role: 'user', content: 'x' }] });
    expect(res.status).toBe(200);
    const meta = JSON.parse(res.text.split('\n').find((l) => l.includes('"done":true')).replace(/^data: /, '')).meta;
    expect(meta.totalTokens).toBe(6);
  });

  describe('GET /api/chat/models', () => {
    it('lists the allow-listed models plus the LLMJob-network model', async () => {
      const app = makeApp(db, {});
      const res = await request(app).get('/api/chat/models');
      expect(res.status).toBe(200);
      expect(res.body.models).toEqual([
        { id: 'meta-llama/llama-3.3-70b-instruct', label: 'Llama 3.3 70B' },
        { id: 'qwen/qwen-2.5-7b-instruct', label: 'Qwen 2.5 7B' },
        { id: 'llmjob-gemma-4-e4b', label: 'Gemma 4 E4B (LLMJob network)' }
      ]);
    });
  });

  describe('GET /api/chat/usage', () => {
    it('reports running totals and remaining free budget', async () => {
      await new ChatUsageService(db).recordUsage({ model: 'm', inTokens: 10, outTokens: 20 });
      const app = makeApp(db, { freeBudget: 100 });
      const res = await request(app).get('/api/chat/usage');
      expect(res.body.totals.totalTokens).toBe(30);
      expect(res.body.freeBudget).toBe(100);
      expect(res.body.remaining).toBe(70);
      expect(res.body.exhausted).toBe(false);
    });

    it('marks the budget exhausted once totals cross it', async () => {
      await new ChatUsageService(db).recordUsage({ model: 'm', inTokens: 60, outTokens: 60 });
      const app = makeApp(db, { freeBudget: 100 });
      const res = await request(app).get('/api/chat/usage');
      expect(res.body.exhausted).toBe(true);
      expect(res.body.remaining).toBe(0);
    });

    it('reports no cap when the free budget is disabled', async () => {
      const app = makeApp(db, { freeBudget: 0 });
      const res = await request(app).get('/api/chat/usage');
      expect(res.body.freeBudget).toBeNull();
      expect(res.body.remaining).toBeNull();
      expect(res.body.exhausted).toBe(false);
    });
  });
});

// ── Unit tests: abort/error branches with fake req/res ────────────────────────

function fakeReq(body) {
  return { app: { locals: { db: {} } }, body };
}
function fakeRes(withFlush = true) {
  const res = {
    statusCode: 200, headers: {}, headersSent: false, body: null, writes: [], ended: false,
    status(c) { this.statusCode = c; return this; },
    json(o) { this.body = o; this.headersSent = true; return this; },
    setHeader(k, v) { this.headers[k] = v; },
    write(s) { this.writes.push(s); this.headersSent = true; return true; },
    end() { this.ended = true; return this; }
  };
  if (withFlush) res.flushHeaders = function () { this.headersSent = true; };
  return res;
}
// A response whose 'close' listener can be fired to simulate a hang-up.
function fakeResClosable() {
  const res = fakeRes();
  const listeners = [];
  res.on = (ev, fn) => { if (ev === 'close') listeners.push(fn); };
  res.emitClose = () => listeners.forEach((fn) => fn());
  return res;
}
function usageSpy() {
  const recorded = [];
  return { chatUsage: { getTotals: async () => ({ totalTokens: 0 }), recordUsage: async (e) => { recorded.push(e); }, _recorded: recorded } };
}
// A stream body that fires `onYield(i)` right after emitting item i.
function hookedBody(events, onYield) {
  return (async function* () {
    for (let i = 0; i < events.length; i++) {
      yield enc.encode('data: ' + JSON.stringify(events[i]) + '\n\n');
      onYield(i);
    }
  })();
}

describe('Chat gateway — controller branches', () => {
  it('stops a stream mid-flight when the caller hangs up (no meta, no [DONE])', async () => {
    const services = usageSpy();
    const events = [{ choices: [{ delta: { content: 'a' } }] }, { choices: [{ delta: { content: 'b' } }] }];
    const res = fakeResClosable();
    const fetchFn = async () => ({ ok: true, status: 200, body: hookedBody(events, (i) => { if (i === 0) res.emitClose(); }) });
    const ctrl = new ChatController({ apiKey: 'k', models: MODELS, services, fetchFn });
    await ctrl.chatCompletions(fakeReq({ messages: [{ role: 'user', content: 'x' }] }), res);
    expect(res.writes.some((w) => w.includes('"delta":"a"'))).toBe(true);
    expect(res.writes.join('')).not.toContain('[DONE]');
    expect(res.writes.join('')).not.toContain('"done":true');
    expect(res.ended).toBe(false);
    expect(services.chatUsage._recorded).toHaveLength(0);
  });

  it('stops after the last chunk when the hang-up lands as the stream ends', async () => {
    const services = usageSpy();
    const events = [{ choices: [{ delta: { content: 'a' } }] }];
    const res = fakeResClosable();
    const fetchFn = async () => ({ ok: true, status: 200, body: hookedBody(events, () => res.emitClose()) });
    const ctrl = new ChatController({ apiKey: 'k', models: MODELS, services, fetchFn });
    await ctrl.chatCompletions(fakeReq({ messages: [{ role: 'user', content: 'x' }] }), res);
    expect(res.writes.join('')).not.toContain('"done":true');
    expect(services.chatUsage._recorded).toHaveLength(0);
  });

  it('ends the stream (headers already sent) when reading the body throws', async () => {
    const services = usageSpy();
    const badBody = (async function* () { yield enc.encode('data: {}\n\n'); throw new Error('mid-stream'); })();
    const ctrl = new ChatController({ apiKey: 'k', models: MODELS, services, fetchFn: async () => ({ ok: true, body: badBody }) });
    const res = fakeRes();
    await ctrl.chatCompletions(fakeReq({ messages: [{ role: 'user', content: 'x' }], stream: true }), res);
    expect(res.headersSent).toBe(true);
    expect(res.ended).toBe(true);
  });

  it('returns 500 (headers not yet sent) when the non-streamed body parse throws', async () => {
    const services = usageSpy();
    const ctrl = new ChatController({
      apiKey: 'k', models: MODELS, services,
      fetchFn: async () => ({ ok: true, json: async () => { throw new Error('bad json'); } })
    });
    const res = fakeRes();
    await ctrl.chatCompletions(fakeReq({ messages: [{ role: 'user', content: 'x' }], stream: false }), res);
    expect(res.statusCode).toBe(500);
    expect(res.body.error.type).toBe('api_error');
  });

  it('streams without flushHeaders available', async () => {
    const services = usageSpy();
    const events = [{ choices: [{ delta: { content: 'hi' }, finish_reason: 'stop' }] }];
    const ctrl = new ChatController({ apiKey: 'k', models: MODELS, services, fetchFn: async () => ({ ok: true, body: sseBody(events) }) });
    const res = fakeRes(false); // no flushHeaders
    await ctrl.chatCompletions(fakeReq({ messages: [{ role: 'user', content: 'x' }] }), res);
    expect(res.ended).toBe(true);
    expect(res.writes.join('')).toContain('"done":true');
    expect(services.chatUsage._recorded).toHaveLength(1);
  });

  it('swallows a failure while recording usage', async () => {
    const services = { chatUsage: { getTotals: async () => ({ totalTokens: 0 }), recordUsage: async () => { throw new Error('db down'); } } };
    const events = [{ choices: [{ delta: { content: 'hi' }, finish_reason: 'stop' }] }];
    const ctrl = new ChatController({ apiKey: 'k', models: MODELS, services, fetchFn: async () => ({ ok: true, body: sseBody(events) }) });
    const res = fakeRes();
    await ctrl.chatCompletions(fakeReq({ messages: [{ role: 'user', content: 'x' }] }), res);
    expect(res.writes.join('')).toContain('"done":true'); // response still completes
  });

  it('treats a bodyless request as an empty body (400)', async () => {
    const ctrl = new ChatController({ apiKey: 'k', models: MODELS, services: usageSpy() });
    const res = fakeRes();
    await ctrl.chatCompletions({ app: { locals: { db: {} } } }, res); // no `body`
    expect(res.statusCode).toBe(400);
  });

  it('builds a ChatUsageService from req.app.locals.db when none is injected', () => {
    const ctrl = new ChatController({ apiKey: 'k' });
    const svc = ctrl.services({ app: { locals: { db: {} } } });
    expect(svc.chatUsage).toBeInstanceOf(ChatUsageService);
  });

  it('initChatRoutes returns a controller with default options', () => {
    const app = express();
    const ctrl = initChatRoutes(app);
    expect(ctrl).toBeInstanceOf(ChatController);
  });
});

// ── LLMJob-network model (served by a live node via the jobs queue) ───────────

// Fake services: a chat-usage recorder plus a jobs service whose getJobResult
// walks a scripted sequence of job states (running → completed/failed).
function netServices(results) {
  let i = 0;
  const recorded = [];
  const created = [];
  return {
    chatUsage: { getTotals: async () => ({ totalTokens: 0 }), recordUsage: async (e) => { recorded.push(e); }, _recorded: recorded },
    jobs: {
      createJob: async (j) => { created.push(j); return { id: 'job-1' }; },
      getJobResult: async () => results[Math.min(i++, results.length - 1)]
    },
    _created: created
  };
}
const NET_ID = 'llmjob-gemma-4-e4b';
const NET_LABEL = 'Gemma 4 E4B (LLMJob network)';

describe('Chat gateway — LLMJob-network model', () => {
  it('serves the model as a job, streaming incremental deltas then a final meta', async () => {
    const services = netServices([
      { status: 'running', partial: 'Hel', chunks: [] },
      { status: 'running', partial: 'Hello', chunks: [] },
      { status: 'completed', result: 'Hello world', metrics: { totalTokens: 3, tokensPerSecond: 30, model: 'Gemma-4-E4B-it-Q4_K_M' } }
    ]);
    // No OpenRouter key needed — this path runs on the fleet, not paid API.
    const ctrl = new ChatController({ apiKey: '', models: MODELS, services, sleep: async () => {}, now: stepClock() });
    const res = fakeRes();
    await ctrl.chatCompletions(fakeReq({ model: NET_ID, temperature: 0.5, messages: [{ role: 'user', content: 'hi' }] }), res);

    const out = res.writes.join('');
    expect(out).toContain('"delta":"Hel"');
    expect(out).toContain('"delta":"lo"');       // only the new suffix each poll
    expect(out).toContain('"delta":" world"');
    expect(out).toContain('"done":true');
    expect(out).toContain('Gemma-4-E4B-it-Q4_K_M');
    expect(out).toContain('[DONE]');
    // The job is anonymous and carries the full (system-grounded) messages array.
    expect(services._created[0]).toMatchObject({ userId: null, temperature: 0.5 });
    expect(services._created[0].messages.some((m) => m.role === 'system')).toBe(true);
    expect(services._created[0].messages[services._created[0].messages.length - 1]).toEqual({ role: 'user', content: 'hi' });
    // Usage recorded from the node's real metrics.
    expect(services.chatUsage._recorded).toHaveLength(1);
    expect(services.chatUsage._recorded[0]).toMatchObject({ outTokens: 3, speed: 30 });
  });

  it('resolves the network model by its label too', async () => {
    const services = netServices([{ status: 'completed', result: 'ok', metrics: {} }]);
    const ctrl = new ChatController({ apiKey: '', models: MODELS, services, sleep: async () => {}, now: stepClock() });
    const res = fakeRes();
    await ctrl.chatCompletions(fakeReq({ model: NET_LABEL, messages: [{ role: 'user', content: 'hi' }] }), res);
    expect(res.writes.join('')).toContain('"done":true');
  });

  it('falls back to estimates and the requested label when the node reports no metrics', async () => {
    const services = netServices([{ status: 'completed', result: 'Hello there', metrics: null }]);
    const ctrl = new ChatController({ apiKey: '', models: MODELS, services, sleep: async () => {}, now: stepClock() });
    const res = fakeRes();
    await ctrl.chatCompletions(fakeReq({ model: NET_ID, messages: [{ role: 'user', content: 'hi' }] }), res);
    const meta = JSON.parse(res.writes.map((w) => w.replace(/^data: /, '')).find((w) => w.includes('"done"'))).meta;
    expect(meta.model).toBe(NET_LABEL);      // no node model → the requested label
    expect(meta.completionTokens).toBeGreaterThan(0); // estimated from the result text
  });

  it('reports a node failure to the stream (falling back when no reason is given)', async () => {
    const services = netServices([{ status: 'failed' }]); // no error field
    const ctrl = new ChatController({ apiKey: '', models: MODELS, services, sleep: async () => {}, now: stepClock() });
    const res = fakeRes();
    await ctrl.chatCompletions(fakeReq({ model: NET_ID, messages: [{ role: 'user', content: 'hi' }] }), res);
    expect(res.writes.join('')).toContain('unknown error');
    expect(res.writes.join('')).toContain('[DONE]');
    expect(services.chatUsage._recorded).toHaveLength(0);
  });

  it('streams an empty completion (no result, no metrics) without flushHeaders', async () => {
    const services = netServices([{ status: 'completed', result: '', metrics: null }]);
    const ctrl = new ChatController({ apiKey: '', models: MODELS, services, sleep: async () => {}, now: stepClock() });
    const res = fakeRes(false); // no flushHeaders available
    await ctrl.chatCompletions(fakeReq({ model: NET_ID, messages: [{ role: 'user', content: 'hi' }] }), res);
    const out = res.writes.join('');
    expect(out).toContain('"done":true');
    expect(out).not.toContain('"delta"'); // nothing to stream
    const meta = JSON.parse(res.writes.map((w) => w.replace(/^data: /, '')).find((w) => w.includes('"done"'))).meta;
    expect(meta.completionTokens).toBe(0);
    expect(meta.tokensPerSecond).toBe(0);
  });

  it('gives up with a helpful error when no node finishes in time', async () => {
    const services = netServices([{ status: 'pending' }]);
    const ctrl = new ChatController({ apiKey: '', models: MODELS, services, sleep: async () => {}, now: stepClock(), jobTimeoutMs: 0 });
    const res = fakeRes();
    await ctrl.chatCompletions(fakeReq({ model: NET_ID, messages: [{ role: 'user', content: 'hi' }] }), res);
    expect(res.writes.join('')).toContain('No node is available');
  });

  it('stops the network long-poll when the caller hangs up (streaming)', async () => {
    const res = fakeResClosable();
    const services = netServices([{ status: 'running', partial: '' }]);
    services.jobs.getJobResult = async () => { res.emitClose(); return { status: 'running', partial: '' }; };
    const ctrl = new ChatController({ apiKey: '', models: MODELS, services, sleep: async () => {}, now: stepClock() });
    await ctrl.chatCompletions(fakeReq({ model: NET_ID, messages: [{ role: 'user', content: 'hi' }] }), res);
    expect(res.writes.join('')).not.toContain('[DONE]');
    expect(res.ended).toBe(false);
  });

  it('serves the network model without streaming (JSON body)', async () => {
    const services = netServices([
      { status: 'running', partial: '' },
      { status: 'completed', result: 'Hi!', metrics: { totalTokens: 1, tokensPerSecond: 12, model: 'Gemma-4-E4B-it-Q4_K_M' } }
    ]);
    const ctrl = new ChatController({ apiKey: '', models: MODELS, services, sleep: async () => {}, now: stepClock() });
    const res = fakeRes();
    await ctrl.chatCompletions(fakeReq({ model: NET_ID, stream: false, messages: [{ role: 'user', content: 'hi' }] }), res);
    expect(res.statusCode).toBe(200);
    expect(res.body.message).toEqual({ role: 'assistant', content: 'Hi!' });
    expect(res.body.usage.completion_tokens).toBe(1);
    expect(res.body.usage.tokens_per_second).toBe(12);
    expect(services.chatUsage._recorded).toHaveLength(1);
  });

  it('returns 502 on a node failure (non-streaming)', async () => {
    const services = netServices([{ status: 'failed', error: 'boom' }]);
    const ctrl = new ChatController({ apiKey: '', models: MODELS, services, sleep: async () => {}, now: stepClock() });
    const res = fakeRes();
    await ctrl.chatCompletions(fakeReq({ model: NET_ID, stream: false, messages: [{ role: 'user', content: 'hi' }] }), res);
    expect(res.statusCode).toBe(502);
    expect(res.body.error.type).toBe('node_error');
    expect(res.body.error.message).toContain('boom');
  });

  it('returns an empty assistant message when the node produced no text (non-streaming)', async () => {
    const services = netServices([{ status: 'completed', result: '', metrics: {} }]);
    const ctrl = new ChatController({ apiKey: '', models: MODELS, services, sleep: async () => {}, now: stepClock() });
    const res = fakeRes();
    await ctrl.chatCompletions(fakeReq({ model: NET_ID, stream: false, messages: [{ role: 'user', content: 'hi' }] }), res);
    expect(res.statusCode).toBe(200);
    expect(res.body.message.content).toBe('');
  });

  it('returns 504 on timeout (non-streaming)', async () => {
    const services = netServices([{ status: 'pending' }]);
    const ctrl = new ChatController({ apiKey: '', models: MODELS, services, sleep: async () => {}, now: stepClock(), jobTimeoutMs: 0 });
    const res = fakeRes();
    await ctrl.chatCompletions(fakeReq({ model: NET_ID, stream: false, messages: [{ role: 'user', content: 'hi' }] }), res);
    expect(res.statusCode).toBe(504);
  });

  it('stops the non-streaming long-poll when the caller hangs up', async () => {
    const res = fakeResClosable();
    const services = netServices([{ status: 'running' }]);
    services.jobs.getJobResult = async () => { res.emitClose(); return { status: 'running' }; };
    const ctrl = new ChatController({ apiKey: '', models: MODELS, services, sleep: async () => {}, now: stepClock() });
    await ctrl.chatCompletions(fakeReq({ model: NET_ID, stream: false, messages: [{ role: 'user', content: 'hi' }] }), res);
    expect(res.body).toBeNull();
    expect(res.ended).toBe(false);
  });

  it('keeps the single-string prompt from the last user turn (or joins when none)', async () => {
    const services = netServices([{ status: 'completed', result: 'r', metrics: {} }]);
    const ctrl = new ChatController({ apiKey: '', models: MODELS, services, sleep: async () => {}, now: stepClock(), systemPrompt: '' });
    const res = fakeRes();
    // assistant-only conversation → no user turn → prompt falls back to joined content
    await ctrl.chatCompletions(fakeReq({ model: NET_ID, messages: [{ role: 'assistant', content: 'prior' }] }), res);
    expect(services._created[0].prompt).toBe('prior');
  });

  it('ends the stream when the network long-poll throws after headers are sent', async () => {
    const services = netServices([{ status: 'running', partial: '' }]);
    services.jobs.getJobResult = async () => { throw new Error('db gone'); };
    const ctrl = new ChatController({ apiKey: '', models: MODELS, services, sleep: async () => {}, now: stepClock() });
    const res = fakeRes(); // flushHeaders marks headers sent inside _streamNetwork
    await ctrl.chatCompletions(fakeReq({ model: NET_ID, messages: [{ role: 'user', content: 'hi' }] }), res);
    expect(res.headersSent).toBe(true);
    expect(res.ended).toBe(true);
  });

  it('returns 500 when the network path throws before responding (non-streaming)', async () => {
    const services = netServices([]);
    services.jobs.createJob = async () => { throw new Error('db gone'); };
    const ctrl = new ChatController({ apiKey: '', models: MODELS, services, sleep: async () => {}, now: stepClock() });
    const res = fakeRes();
    await ctrl.chatCompletions(fakeReq({ model: NET_ID, stream: false, messages: [{ role: 'user', content: 'hi' }] }), res);
    expect(res.statusCode).toBe(500);
    expect(res.body.error.type).toBe('api_error');
  });

  it('uses a real timer for its default poll sleep', async () => {
    const ctrl = new ChatController({ apiKey: '' }); // default sleep
    await ctrl.sleep(1); // resolves via setTimeout — exercises the default
  });

  it('rejects an unknown model with 400 (the network model is not a catch-all)', async () => {
    const ctrl = new ChatController({ apiKey: 'k', models: MODELS, services: usageSpy() });
    const res = fakeRes();
    await ctrl.chatCompletions(fakeReq({ model: 'nope/not-a-model', messages: [{ role: 'user', content: 'hi' }] }), res);
    expect(res.statusCode).toBe(400);
    expect(res.body.error.type).toBe('invalid_request_error');
  });
});

// ── Config / pure helpers ─────────────────────────────────────────────────────

describe('ChatController — config', () => {
  const ENV_KEYS = ['OPENROUTER_API_KEY', 'OPENROUTER_BASE_URL', 'OPENROUTER_MODELS',
    'OPENROUTER_FREE_TOKEN_BUDGET', 'OPENROUTER_MAX_TOKENS', 'OPENROUTER_REFERER',
    'OPENROUTER_SYSTEM_PROMPT'];
  let saved;
  beforeEach(() => { saved = {}; ENV_KEYS.forEach((k) => { saved[k] = process.env[k]; delete process.env[k]; }); });
  afterEach(() => { ENV_KEYS.forEach((k) => { if (saved[k] === undefined) delete process.env[k]; else process.env[k] = saved[k]; }); });

  it('falls back to built-in defaults when nothing is configured', () => {
    const ctrl = new ChatController();
    expect(ctrl.apiKey).toBeUndefined();
    expect(ctrl.baseUrl).toBe('https://openrouter.ai/api/v1');
    expect(ctrl.networkModel).toEqual({ id: 'llmjob-gemma-4-e4b', label: 'Gemma 4 E4B (LLMJob network)' });
    expect(ctrl.models).toBe(ChatController.DEFAULT_MODELS);
    expect(ctrl.freeBudget).toBe(1000000);
    expect(ctrl.maxTokens).toBe(2048);
    expect(ctrl.referer).toBe('https://llmjob.com');
    expect(ctrl.title).toBe('LLMJob');
    expect(ctrl.systemPrompt).toContain('LLMJob assistant');
  });

  it('reads configuration from the environment', () => {
    process.env.OPENROUTER_API_KEY = 'env-key';
    process.env.OPENROUTER_BASE_URL = 'https://env.test/v1';
    process.env.OPENROUTER_MODELS = JSON.stringify([{ id: 'x/y', label: 'XY' }]);
    process.env.OPENROUTER_FREE_TOKEN_BUDGET = '500';
    process.env.OPENROUTER_MAX_TOKENS = '64';
    process.env.OPENROUTER_REFERER = 'https://env.example';
    process.env.OPENROUTER_SYSTEM_PROMPT = 'env system prompt';
    const ctrl = new ChatController();
    expect(ctrl.apiKey).toBe('env-key');
    expect(ctrl.baseUrl).toBe('https://env.test/v1');
    expect(ctrl.models).toEqual([{ id: 'x/y', label: 'XY' }]);
    expect(ctrl.freeBudget).toBe(500);
    expect(ctrl.maxTokens).toBe(64);
    expect(ctrl.referer).toBe('https://env.example');
    expect(ctrl.systemPrompt).toBe('env system prompt');
  });

  it('honours explicit constructor overrides for referer, title, and system prompt', () => {
    const ctrl = new ChatController({ referer: 'https://o.test', title: 'Zed', systemPrompt: 'custom' });
    expect(ctrl.referer).toBe('https://o.test');
    expect(ctrl.title).toBe('Zed');
    expect(ctrl.systemPrompt).toBe('custom');
    // an explicit empty string disables injection (distinct from "not provided")
    expect(new ChatController({ systemPrompt: '' }).systemPrompt).toBe('');
  });

  it('falls back to defaults when the env budget/max are not numbers', () => {
    process.env.OPENROUTER_FREE_TOKEN_BUDGET = 'lots';
    process.env.OPENROUTER_MAX_TOKENS = 'plenty';
    const ctrl = new ChatController();
    expect(ctrl.freeBudget).toBe(1000000);
    expect(ctrl.maxTokens).toBe(2048);
  });

  describe('parseModels', () => {
    it('parses a valid JSON array of {id,label}', () => {
      expect(ChatController.parseModels('[{"id":"a/b","label":"AB"},{"id":"c/d"}]'))
        .toEqual([{ id: 'a/b', label: 'AB' }, { id: 'c/d', label: 'c/d' }]);
    });
    it('returns null for empty/invalid/non-array/entry-less input', () => {
      expect(ChatController.parseModels('')).toBeNull();
      expect(ChatController.parseModels('not json')).toBeNull();
      expect(ChatController.parseModels('{"id":"a"}')).toBeNull();
      expect(ChatController.parseModels('[{"label":"no id"}]')).toBeNull();
    });
  });

  it('_resolveMaxTokens clamps, defaults on invalid, and floors', () => {
    const ctrl = new ChatController({ apiKey: 'k', maxTokens: 100 });
    expect(ctrl._resolveMaxTokens(40.9)).toBe(40);
    expect(ctrl._resolveMaxTokens(500)).toBe(100);
    expect(ctrl._resolveMaxTokens(0)).toBe(100);
    expect(ctrl._resolveMaxTokens('abc')).toBe(100);
    expect(ctrl._resolveMaxTokens(undefined)).toBe(100);
  });

  it('_resolveModel matches by id or label, defaults, and rejects unknown', () => {
    const ctrl = new ChatController({ apiKey: 'k', models: MODELS });
    expect(ctrl._resolveModel()).toBe(MODELS[0]);
    expect(ctrl._resolveModel('qwen/qwen-2.5-7b-instruct')).toBe(MODELS[1]);
    expect(ctrl._resolveModel('Llama 3.3 70B')).toBe(MODELS[0]);
    expect(ctrl._resolveModel('nope')).toBeNull();
  });
});

describe('Chat gateway — pure helpers', () => {
  it('estimateTokens is ~4 chars/token and tolerates empty input', () => {
    expect(estimateTokens('abcd')).toBe(1);
    expect(estimateTokens('')).toBe(0);
    expect(estimateTokens(null)).toBe(0);
  });

  it('sanitizeMessages coerces roles, drops empties/non-objects, and caps length', () => {
    const out = sanitizeMessages([
      null,
      'not an object',
      { role: 'tool', content: 'demoted to user' },
      { role: 'user', content: null },
      { role: 'assistant', content: 'kept' }
    ]);
    expect(out).toEqual([
      { role: 'user', content: 'demoted to user' },
      { role: 'assistant', content: 'kept' }
    ]);
  });

  it('sanitizeMessages truncates at the character budget and stops', () => {
    const big = 'a'.repeat(30000);
    const out = sanitizeMessages([{ role: 'user', content: big }, { role: 'user', content: 'dropped' }]);
    expect(out).toHaveLength(1);
    expect(out[0].content.length).toBe(24000);
  });

  it('upstreamErrorMessage extracts a reason from JSON, raw text, or status', async () => {
    const j = (body) => ({ status: 500, text: async () => body });
    expect(await upstreamErrorMessage(j(JSON.stringify({ error: { message: 'bad model' } })))).toBe('bad model');
    expect(await upstreamErrorMessage(j('{"foo":1}'))).toBe('{"foo":1}'); // JSON without error.message → raw body
    expect(await upstreamErrorMessage(j('plain text boom'))).toBe('plain text boom'); // not JSON
    expect(await upstreamErrorMessage(j(''))).toBe('HTTP 500'); // empty body → bare status
    expect(await upstreamErrorMessage({ status: 503, text: async () => { throw new Error('no body'); } })).toBe('HTTP 503');
  });

  it('upstreamErrorMessage truncates a very long body', async () => {
    const long = 'z'.repeat(1000);
    const out = await upstreamErrorMessage({ status: 500, text: async () => long });
    expect(out.length).toBe(300);
  });

  it('parseSSE yields events, tolerates split chunks/comments/garbage, and stops at [DONE]', async () => {
    async function* body() {
      yield enc.encode(': a comment line\n');
      yield enc.encode('data: {"a":1}\n');
      yield enc.encode('data: {"b":');   // split across chunks
      yield enc.encode('2}\n');
      yield enc.encode('ignored non-data line\n');
      yield enc.encode('data: {bad json}\n');
      yield enc.encode('data: [DONE]\n');
      yield enc.encode('data: {"never":1}\n'); // after [DONE], never reached
    }
    const seen = [];
    for await (const obj of parseSSE(body())) seen.push(obj);
    expect(seen).toEqual([{ a: 1 }, { b: 2 }]);
  });
});
