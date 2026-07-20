// The OpenAI-compatible gateway (POST /v1/chat/completions).
//
// Integration tests run the real Express route against pg-mem while the test
// plays the node (claim → chunks → complete/fail) via JobService. supertest only
// dispatches a request when its promise is awaited, so the gateway call and the
// node simulation are driven together with Promise.all. Unit tests then cover the
// controller's error/edge branches with injected fakes.
const request = require('supertest');
const express = require('express');
const { createTestDb } = require('./helpers/pgmem');
const { initOpenAiRoutes } = require('../src/routes');
const JobService = require('../src/services/jobService');
const ApiKeyService = require('../src/services/apiKeyService');
const OpenAiController = require('../src/controllers/openaiController');
const { lastUserText, estimateTokens, modelName, completionTokens } = OpenAiController;

const NODE_ID = 'node-openai-test';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function makeApp(db, opts) {
  const app = express();
  app.use(express.json());
  app.locals.db = db;
  initOpenAiRoutes(app, Object.assign({ pollMs: 5, timeoutMs: 1500 }, opts));
  return app;
}

// Play the node: wait for the gateway's pending job, claim it, stream chunks,
// complete it. Returns the claimed job (so tests can inspect job.messages).
async function nodeServe(jobService, chunks, metrics) {
  let job;
  for (let i = 0; i < 300 && !job; i++) {
    job = (await jobService.assignJobsToNode(NODE_ID, 1))[0];
    if (!job) await sleep(5);
  }
  if (!job) throw new Error('gateway never created a job');
  await jobService.handleHeartbeat(job.id, NODE_ID); // → running
  for (let i = 0; i < chunks.length; i++) {
    const isFinal = i === chunks.length - 1;
    await jobService.storeChunk(job.id, NODE_ID, {
      chunkIndex: i, content: chunks[i], isFinal, metrics: isFinal ? metrics : undefined,
    });
  }
  await jobService.completeJob(job.id, NODE_ID);
  return job;
}

async function nodeFail(jobService, reason) {
  let job;
  for (let i = 0; i < 300 && !job; i++) {
    job = (await jobService.assignJobsToNode(NODE_ID, 1))[0];
    if (!job) await sleep(5);
  }
  await jobService.handleHeartbeat(job.id, NODE_ID);
  await jobService.failJob(job.id, NODE_ID, reason);
}

describe('OpenAI gateway — integration', () => {
  let db, app, jobService, rawKey, userId;

  beforeEach(async () => {
    db = await createTestDb();
    app = makeApp(db);
    jobService = new JobService(db);
    userId = 'user-openai';
    rawKey = (await new ApiKeyService(db).createKey(userId, 'gateway-test')).key;
  });

  afterEach(async () => {
    await sleep(25); // let best-effort usage writes land before the pool closes
    if (db.end) await db.end();
  });

  const auth = () => ['Authorization', 'Bearer ' + rawKey];

  it('returns an OpenAI chat.completion once a node serves the job', async () => {
    const [res, job] = await Promise.all([
      request(app).post('/v1/chat/completions').set(...auth())
        .send({ model: 'my-model', messages: [{ role: 'user', content: 'Say hi' }] }),
      nodeServe(jobService, ['Hello', ' there'], { totalTokens: 2, tokensPerSecond: 20, model: 'Gemma-4-E4B-it-Q4_K_M' }),
    ]);
    expect(res.status).toBe(200);
    expect(res.body.object).toBe('chat.completion');
    expect(res.body.id).toBe('chatcmpl-' + job.id);
    expect(res.body.choices[0]).toMatchObject({
      index: 0, message: { role: 'assistant', content: 'Hello there' }, finish_reason: 'stop',
    });
    expect(res.body.model).toBe('Gemma-4-E4B-it-Q4_K_M'); // what the node actually ran
    expect(res.body.usage.completion_tokens).toBe(2);
    expect(res.body.usage.prompt_tokens).toBeGreaterThan(0);
    expect(res.body.usage.total_tokens).toBe(res.body.usage.prompt_tokens + 2);
    expect(job.model).toBe('my-model'); // the requested model rode through to the job
  });

  it('carries a full multi-turn messages array through to the node', async () => {
    const messages = [
      { role: 'system', content: 'Be terse.' },
      { role: 'user', content: 'Hi' },
      { role: 'assistant', content: 'Hey.' },
      { role: 'user', content: 'Bye?' },
    ];
    const [, job] = await Promise.all([
      request(app).post('/v1/chat/completions').set(...auth()).send({ messages }),
      nodeServe(jobService, ['bye'], { totalTokens: 1 }),
    ]);
    expect(job.messages).toEqual(messages); // whole conversation reached the node
    expect(job.prompt).toBe('Bye?');        // last user turn kept as the display prompt
  });

  it('streams chat.completion.chunk SSE events ending with [DONE]', async () => {
    const [res] = await Promise.all([
      request(app).post('/v1/chat/completions').set(...auth())
        .send({ messages: [{ role: 'user', content: 'Hi' }], stream: true }),
      // includes an empty chunk, which must be skipped
      nodeServe(jobService, ['', 'Hel', 'lo'], { totalTokens: 2 }),
    ]);
    expect(res.status).toBe(200);
    expect(res.headers['content-type']).toMatch(/text\/event-stream/);
    const text = res.text;
    expect(text).toContain('"delta":{"role":"assistant"}');
    expect(text).toContain('"delta":{"content":"Hel"}');
    expect(text).toContain('"delta":{"content":"lo"}');
    expect(text).not.toContain('"content":""'); // empty chunk skipped
    expect(text).toContain('"finish_reason":"stop"');
    expect(text.trim().endsWith('data: [DONE]')).toBe(true);
  });

  it('records usage against the API key after completion', async () => {
    await Promise.all([
      request(app).post('/v1/chat/completions').set(...auth())
        .send({ messages: [{ role: 'user', content: 'count my tokens' }] }),
      nodeServe(jobService, ['ok'], { totalTokens: 5, tokensPerSecond: 12 }),
    ]);
    const keyService = new ApiKeyService(db);
    let usage = 0;
    for (let i = 0; i < 60 && usage === 0; i++) {
      usage = (await keyService.listKeys(userId))[0].usage;
      if (!usage) await sleep(10);
    }
    expect(usage).toBeGreaterThanOrEqual(5);
  });

  it('rejects a request with no API key (401)', async () => {
    const res = await request(app).post('/v1/chat/completions')
      .send({ messages: [{ role: 'user', content: 'hi' }] });
    expect(res.status).toBe(401);
  });

  it('rejects missing / empty / bodyless messages (400)', async () => {
    const a = await request(app).post('/v1/chat/completions').set(...auth()).send({});
    expect(a.status).toBe(400);
    expect(a.body.error.type).toBe('invalid_request_error');
    const b = await request(app).post('/v1/chat/completions').set(...auth()).send({ messages: [] });
    expect(b.status).toBe(400);
    const c = await request(app).post('/v1/chat/completions').set(...auth()); // no body at all
    expect(c.status).toBe(400);
  });

  it('returns 502 when the node fails the job (non-streaming)', async () => {
    const [res] = await Promise.all([
      request(app).post('/v1/chat/completions').set(...auth())
        .send({ messages: [{ role: 'user', content: 'boom' }] }),
      nodeFail(jobService, 'model crashed'),
    ]);
    expect(res.status).toBe(502);
    expect(res.body.error.type).toBe('node_error');
    expect(res.body.error.message).toContain('model crashed');
  });

  it('writes a node_error event then [DONE] when a streamed job fails', async () => {
    const [res] = await Promise.all([
      request(app).post('/v1/chat/completions').set(...auth())
        .send({ messages: [{ role: 'user', content: 'boom' }], stream: true }),
      nodeFail(jobService, 'kaboom'),
    ]);
    expect(res.status).toBe(200);
    expect(res.text).toContain('"type":"node_error"');
    expect(res.text.trim().endsWith('data: [DONE]')).toBe(true);
  });

  it('returns 504 when no node picks the job up before the timeout', async () => {
    const fast = makeApp(db, { timeoutMs: 120 });
    const res = await request(fast).post('/v1/chat/completions').set(...auth())
      .send({ messages: [{ role: 'user', content: 'nobody home' }] });
    expect(res.status).toBe(504);
    expect(res.body.error.type).toBe('timeout_error');
  });

  it('writes a timeout_error event then [DONE] when a streamed job times out', async () => {
    const fast = makeApp(db, { timeoutMs: 120 });
    const res = await request(fast).post('/v1/chat/completions').set(...auth())
      .send({ messages: [{ role: 'user', content: 'nobody' }], stream: true });
    expect(res.status).toBe(200);
    expect(res.text).toContain('"type":"timeout_error"');
    expect(res.text.trim().endsWith('data: [DONE]')).toBe(true);
  });
});

// ── Unit tests: controller error/edge branches with injected fakes ────────────

function fakeReq(body) {
  return { app: { locals: { db: {} } }, body, apiKey: { userId: 'u', name: 'k', hash: 'h' } };
}
function fakeRes(withFlush = true) {
  const res = {
    statusCode: 200, headers: {}, headersSent: false, body: null, writes: [], ended: false,
    status(c) { this.statusCode = c; return this; },
    json(o) { this.body = o; this.headersSent = true; return this; },
    setHeader(k, v) { this.headers[k] = v; },
    write(s) { this.writes.push(s); this.headersSent = true; return true; },
    end() { this.ended = true; return this; },
  };
  if (withFlush) res.flushHeaders = function () { this.headersSent = true; };
  return res;
}
function fakeServices(over = {}) {
  return {
    jobService: Object.assign({
      createJob: async (j) => ({ id: 'job-1', model: j.model || 'default', messages: j.messages }),
      getJobResult: async () => ({ status: 'completed', result: 'hi', metrics: { totalTokens: 1 }, assignedTo: 'n1' }),
    }, over.jobService),
    logService: Object.assign({ recordLog: async () => {} }, over.logService),
    apiKeyService: Object.assign({ recordUsage: async () => {} }, over.apiKeyService),
  };
}

describe('OpenAI gateway — controller branches', () => {
  it('returns 500 (headers not yet sent) when a non-streamed job lookup throws', async () => {
    const services = fakeServices({ jobService: { getJobResult: async () => { throw new Error('boom'); } } });
    const ctrl = new OpenAiController({ services, pollMs: 1, timeoutMs: 50 });
    const res = fakeRes();
    await ctrl.chatCompletions(fakeReq({ messages: [{ role: 'user', content: 'x' }] }), res);
    expect(res.statusCode).toBe(500);
    expect(res.body.error.type).toBe('api_error');
  });

  it('ends the stream (headers already sent) when a streamed job lookup throws', async () => {
    const services = fakeServices({ jobService: { getJobResult: async () => { throw new Error('mid-stream'); } } });
    const ctrl = new OpenAiController({ services, pollMs: 1, timeoutMs: 50 });
    const res = fakeRes();
    await ctrl.chatCompletions(fakeReq({ messages: [{ role: 'user', content: 'x' }], stream: true }), res);
    expect(res.headersSent).toBe(true);
    expect(res.ended).toBe(true);
    expect(res.writes.some((w) => w.includes('"role":"assistant"'))).toBe(true);
  });

  it('swallows a failure while recording usage', async () => {
    const services = fakeServices({ logService: { recordLog: async () => { throw new Error('log db down'); } } });
    const ctrl = new OpenAiController({ services, pollMs: 1 });
    const res = fakeRes();
    await ctrl.chatCompletions(fakeReq({ messages: [{ role: 'user', content: 'x' }] }), res);
    expect(res.statusCode).toBe(200); // response still succeeds despite the usage error
  });

  it('handles a completion missing model/metrics/assignedTo and works without flushHeaders', async () => {
    const services = fakeServices({
      jobService: { getJobResult: async () => ({ status: 'completed', result: 'hello world' }) }, // no metrics/assignedTo
    });
    const recorded = [];
    services.logService.recordLog = async (uid, entry) => { recorded.push(entry); };
    const ctrl = new OpenAiController({ services, pollMs: 1 });
    const res = fakeRes(false); // no flushHeaders
    await ctrl.chatCompletions(fakeReq({ messages: [{ role: 'user', content: 'x' }], stream: true }), res);
    expect(res.ended).toBe(true);
    // model falls back to the job's model; usage falls back to an estimate; node 'unknown'
    expect(recorded[0]).toMatchObject({ model: 'default', node: 'unknown', speed: 0 });
    expect(recorded[0].out).toBe(Math.ceil('hello world'.length / 4));
  });

  it('returns an empty assistant message when the completed result is empty', async () => {
    const services = fakeServices({
      jobService: { getJobResult: async () => ({ status: 'completed', result: '', metrics: { totalTokens: 0 }, assignedTo: 'n' }) },
    });
    const ctrl = new OpenAiController({ services, pollMs: 1 });
    const res = fakeRes();
    await ctrl.chatCompletions(fakeReq({ messages: [{ role: 'user', content: 'x' }] }), res);
    expect(res.statusCode).toBe(200);
    expect(res.body.choices[0].message.content).toBe('');
  });

  it('reports a generic reason when a failed job carries no error (json + stream)', async () => {
    const services = fakeServices({ jobService: { getJobResult: async () => ({ status: 'failed' }) } });
    const jsonRes = fakeRes();
    await new OpenAiController({ services, pollMs: 1 }).chatCompletions(fakeReq({ messages: [{ role: 'user', content: 'x' }] }), jsonRes);
    expect(jsonRes.statusCode).toBe(502);
    expect(jsonRes.body.error.message).toContain('unknown error');

    const streamRes = fakeRes();
    await new OpenAiController({ services, pollMs: 1 }).chatCompletions(fakeReq({ messages: [{ role: 'user', content: 'x' }], stream: true }), streamRes);
    expect(streamRes.writes.join('')).toContain('unknown error');
    expect(streamRes.ended).toBe(true);
  });

  it('builds services from req.app.locals.db when none are injected', () => {
    const ctrl = new OpenAiController();
    const svc = ctrl.services({ app: { locals: { db: {} } } });
    expect(svc.jobService).toBeTruthy();
    expect(svc.logService).toBeTruthy();
    expect(svc.apiKeyService).toBeTruthy();
  });

  it('initOpenAiRoutes registers the route with default options when none are given', () => {
    const app = express();
    const ctrl = initOpenAiRoutes(app);
    expect(ctrl).toBeInstanceOf(OpenAiController);
  });
});

describe('OpenAI gateway — pure helpers', () => {
  it('lastUserText finds the last user turn, else joins all content', () => {
    expect(lastUserText([{ role: 'user', content: 'a' }, { role: 'assistant', content: 'b' }])).toBe('a');
    expect(lastUserText([null, { role: 'user', content: null }, { role: 'user', content: 'z' }])).toBe('z');
    expect(lastUserText([{ role: 'assistant', content: 'x' }])).toBe('x'); // no user → join
    expect(lastUserText([{ role: 'system' }])).toBe('');                    // no content anywhere
  });

  it('modelName prefers the node-reported model, else the job model', () => {
    expect(modelName({ metrics: { model: 'X' } }, { model: 'Y' })).toBe('X');
    expect(modelName({ metrics: {} }, { model: 'Y' })).toBe('Y');
    expect(modelName(null, { model: 'Y' })).toBe('Y');
  });

  it('completionTokens uses reported tokens, else estimates the result', () => {
    expect(completionTokens({ metrics: { totalTokens: 7 } })).toBe(7);
    expect(completionTokens({ metrics: {}, result: 'abcd' })).toBe(1);
    expect(completionTokens({ result: '' })).toBe(0);
  });

  it('estimateTokens is ~4 chars/token and tolerates empty input', () => {
    expect(estimateTokens('abcd')).toBe(1);
    expect(estimateTokens('')).toBe(0);
    expect(estimateTokens(null)).toBe(0);
  });
});
