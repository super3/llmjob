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
const { lastUserText, estimateTokens, modelName, completionTokens, timeoutBody } = OpenAiController;
const { DEFAULT_MODEL } = JobService;

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
async function nodeServe(jobService, chunks, metrics, reasoning) {
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
      reasoning: isFinal ? reasoning : undefined, // thinking models attach it to the last chunk
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

// Put a node row in the table so the gateway's target-node liveness check can see
// it. `stale` pushes last_seen past the offline threshold (node exists but offline).
async function seedNode(db, nodeId, { stale = false } = {}) {
  const lastSeen = stale ? Date.now() - 20 * 60 * 1000 : Date.now();
  await db.query(
    `INSERT INTO nodes (node_id, public_key, name, user_id, status, is_public, last_seen, claimed_at)
     VALUES ($1, $1, $1, 'user-openai', 'online', false, $2, $2)`,
    [nodeId, lastSeen]
  );
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
    // The requested model must NOT ride through to the job: the node ignores it and
    // would otherwise echo it back via metrics.model as the model that ran.
    expect(job.model).toBe('Gemma-4-E4B-it-Q4_K_M'); // the fleet default, not 'my-model'
  });

  it('reports finish_reason "length" and the reasoning when a thinking model runs out of budget', async () => {
    // The silent-empty-answer bug: max_tokens was spent on the chain of thought,
    // so `content` is empty. That must read as a truncation, not a clean stop.
    const [res] = await Promise.all([
      request(app).post('/v1/chat/completions').set(...auth())
        .send({ messages: [{ role: 'user', content: 'Say hi' }], max_tokens: 40 }),
      nodeServe(jobService, [''], { totalTokens: 40, finishReason: 'length' }, 'thinking about it'),
    ]);
    expect(res.status).toBe(200);
    expect(res.body.choices[0].finish_reason).toBe('length');
    expect(res.body.choices[0].message.content).toBe('');
    expect(res.body.choices[0].message.reasoning_content).toBe('thinking about it');
    expect(res.body.usage.completion_tokens).toBe(40); // reasoning tokens still billed
  });

  it('omits reasoning_content for an ordinary model', async () => {
    const [res] = await Promise.all([
      request(app).post('/v1/chat/completions').set(...auth())
        .send({ messages: [{ role: 'user', content: 'Say hi' }] }),
      nodeServe(jobService, ['Hi'], { totalTokens: 1 }),
    ]);
    expect(res.body.choices[0]).not.toHaveProperty('message.reasoning_content');
    expect(res.body.choices[0].message.reasoning_content).toBeUndefined();
    expect(res.body.choices[0].finish_reason).toBe('stop'); // node reported none → default
  });

  it('streams reasoning_content and the real finish_reason', async () => {
    const [res] = await Promise.all([
      request(app).post('/v1/chat/completions').set(...auth())
        .send({ messages: [{ role: 'user', content: 'Hi' }], stream: true, max_tokens: 40 }),
      nodeServe(jobService, [''], { totalTokens: 40, finishReason: 'length' }, 'hmm'),
    ]);
    expect(res.text).toContain('"delta":{"reasoning_content":"hmm"}');
    expect(res.text).toContain('"finish_reason":"length"');
    expect(res.text.trim().endsWith('data: [DONE]')).toBe(true);
  });

  it('logs the real finish reason against the key', async () => {
    await Promise.all([
      request(app).post('/v1/chat/completions').set(...auth())
        .send({ messages: [{ role: 'user', content: 'Hi' }], max_tokens: 40 }),
      nodeServe(jobService, ['cut'], { totalTokens: 40, finishReason: 'length' }),
    ]);
    await sleep(30); // usage accounting is best-effort/after-response
    const LogService = require('../src/services/logService');
    const logs = await new LogService(db).getLogs(userId, 10);
    expect(logs[0].finish).toBe('length');
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

  it('never echoes the caller-requested model; reports what the fleet ran', async () => {
    // A caller can send any `model` string, but the node serves its own local
    // model regardless — so the response must report the real one, not "llmjob".
    const [res] = await Promise.all([
      request(app).post('/v1/chat/completions').set(...auth())
        .send({ model: 'llmjob', messages: [{ role: 'user', content: 'Hi' }] }),
      nodeServe(jobService, ['hi'], { totalTokens: 1, model: 'Gemma-4-E4B-it-Q4_K_M' }),
    ]);
    expect(res.body.model).toBe('Gemma-4-E4B-it-Q4_K_M');
    expect(res.body.model).not.toBe('llmjob');
  });

  it('reports the fleet default when the node tags no model in its metrics', async () => {
    // Older nodes omit metrics.model; the fallback is the fleet default, still
    // not the caller's requested string.
    const [res] = await Promise.all([
      request(app).post('/v1/chat/completions').set(...auth())
        .send({ model: 'llmjob', messages: [{ role: 'user', content: 'Hi' }] }),
      nodeServe(jobService, ['hi'], { totalTokens: 1 }), // no model in metrics
    ]);
    expect(res.body.model).toBe(DEFAULT_MODEL);
  });

  it('reports the serving node as a response header', async () => {
    const [res] = await Promise.all([
      request(app).post('/v1/chat/completions').set(...auth())
        .send({ messages: [{ role: 'user', content: 'Hi' }] }),
      nodeServe(jobService, ['hi'], { totalTokens: 3 }),
    ]);
    expect(res.headers['x-llmjob-served-by']).toBe(NODE_ID);
    expect(res.headers['x-llmjob-tokens-per-second']).toBeUndefined(); // throughput intentionally not reported
  });

  it('pins a request to a targeted online node and records it on the job', async () => {
    await seedNode(db, NODE_ID); // online
    const [res, job] = await Promise.all([
      request(app).post('/v1/chat/completions').set(...auth())
        .set('X-LLMJob-Node', NODE_ID)
        .send({ messages: [{ role: 'user', content: 'Hi' }] }),
      nodeServe(jobService, ['ok'], { totalTokens: 1 }),
    ]);
    expect(res.status).toBe(200);
    expect(job.targetNode).toBe(NODE_ID);        // pinned on the job record
    expect(res.headers['x-llmjob-served-by']).toBe(NODE_ID);
  });

  it('fast-fails with 404 when the targeted node is offline', async () => {
    await seedNode(db, 'sleepy-node', { stale: true });
    const res = await request(app).post('/v1/chat/completions').set(...auth())
      .set('X-LLMJob-Node', 'sleepy-node')
      .send({ messages: [{ role: 'user', content: 'Hi' }] });
    expect(res.status).toBe(404);
    expect(res.body.error.type).toBe('target_node_error');
    expect(res.body.error.message).toMatch(/is offline/);
  });

  it('fast-fails with 404 when the targeted node is unknown', async () => {
    const res = await request(app).post('/v1/chat/completions').set(...auth())
      .set('X-LLMJob-Node', 'ghost')
      .send({ messages: [{ role: 'user', content: 'Hi' }] });
    expect(res.status).toBe(404);
    expect(res.body.error.message).toMatch(/is not a known node/);
  });

  it('ignores a blank target header (whitespace) and serves normally', async () => {
    const [res] = await Promise.all([
      request(app).post('/v1/chat/completions').set(...auth())
        .set('X-LLMJob-Node', '   ')
        .send({ messages: [{ role: 'user', content: 'Hi' }] }),
      nodeServe(jobService, ['hi'], { totalTokens: 1 }),
    ]);
    expect(res.status).toBe(200); // no node row seeded, but blank target means no targeting
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

  // A non-empty array whose entries all clamp away must not become a job — the
  // gateway clamps prompts now, so this branch is reachable.
  it('rejects messages that survive the array check but hold no usable content (400)', async () => {
    const res = await request(app).post('/v1/chat/completions').set(...auth())
      .send({ messages: [{ role: 'user', content: '' }, null, 'nope'] });
    expect(res.status).toBe(400);
    expect(res.body.error.message).toBe('No usable message content.');
  });

  // max_tokens rode through unbounded before, so one key could ask for millions
  // of tokens and hold a node's GPU for as long as it took.
  it('clamps an oversized max_tokens onto the job', async () => {
    const jobs = [];
    const clampApp = makeApp(db, {
      services: {
        jobService: {
          createJob: async (j) => { jobs.push(j); return { id: 'job-clamp' }; },
          getJobResult: async () => ({ status: 'completed', result: 'ok', chunks: [] }),
        },
        logService: { recordLog: async () => {} },
        apiKeyService: { recordUsage: async () => {} },
        nodeService: { getNodeStatus: async () => ({ exists: true, online: true }) },
      },
    });
    await request(clampApp).post('/v1/chat/completions').set(...auth())
      .send({ messages: [{ role: 'user', content: 'hi' }], max_tokens: 999999 });

    expect(jobs).toHaveLength(1);
    expect(jobs[0].maxTokens).toBe(32768);
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
    // Same diagnostics a 504 carries — a failure is attributable either way.
    expect(res.body.error.message).toContain(`Node ${NODE_ID}`);
    expect(res.body.error.served_by).toBe(NODE_ID);
    expect(res.body.error.job_status).toBe('failed');
    expect(res.body.error.job_id).toEqual(expect.any(String));
    expect(res.headers['x-llmjob-served-by']).toBe(NODE_ID);
  });

  it('writes a node_error event then [DONE] when a streamed job fails', async () => {
    const [res] = await Promise.all([
      request(app).post('/v1/chat/completions').set(...auth())
        .send({ messages: [{ role: 'user', content: 'boom' }], stream: true }),
      nodeFail(jobService, 'kaboom'),
    ]);
    expect(res.status).toBe(200);
    expect(res.text).toContain('"type":"node_error"');
    expect(res.text).toContain(`"served_by":"${NODE_ID}"`); // headers are long gone
    expect(res.text).toContain('"job_status":"failed"');
    expect(res.text.trim().endsWith('data: [DONE]')).toBe(true);
  });

  it('returns 504 when no node picks the job up before the timeout', async () => {
    const fast = makeApp(db, { timeoutMs: 120 });
    const res = await request(fast).post('/v1/chat/completions').set(...auth())
      .send({ messages: [{ role: 'user', content: 'nobody home' }] });
    expect(res.status).toBe(504);
    expect(res.body.error.type).toBe('timeout_error');
    // Diagnosable: the job id to look it up by, and proof no node ever had it.
    expect(res.body.error.job_id).toEqual(expect.any(String));
    expect(res.body.error.served_by).toBeNull();
    expect(res.body.error.job_status).toBe('pending');
    expect(res.body.error.message).toContain('No node picked the job up');
    expect(res.headers['x-llmjob-served-by']).toBeUndefined(); // nobody to name
  });

  it('writes a timeout_error event then [DONE] when a streamed job times out', async () => {
    const fast = makeApp(db, { timeoutMs: 120 });
    const res = await request(fast).post('/v1/chat/completions').set(...auth())
      .send({ messages: [{ role: 'user', content: 'nobody' }], stream: true });
    expect(res.status).toBe(200);
    expect(res.text).toContain('"type":"timeout_error"');
    expect(res.text).toContain('"served_by":null');   // headers are long gone; it rides in the body
    expect(res.text).toContain('"job_status":"pending"');
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
// A fake response that can simulate the caller hanging up (a 'close' event).
function fakeResClosable() {
  const res = fakeRes();
  const listeners = [];
  res.on = (ev, fn) => { if (ev === 'close') listeners.push(fn); };
  res.emitClose = () => listeners.forEach((fn) => fn());
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
  it('_setServedByHeader no-ops without setHeader, and skips an absent node', () => {
    const ctrl = new OpenAiController();
    // A res that can't take headers (or a non-final poll result) must not throw.
    expect(() => ctrl._setServedByHeader({}, { assignedTo: 'n' })).not.toThrow();
    // A completed result missing assignedTo sets no header.
    const headers = {};
    ctrl._setServedByHeader({ setHeader: (k, v) => { headers[k] = v; } }, { status: 'completed' });
    expect(headers).toEqual({});
  });

  it('defaults to a timeout that leaves a reasoning model room to finish', () => {
    // The generation ceiling is this timeout times the node's tokens/sec, so
    // shortening it silently truncates long answers into 504s. 280s stays under
    // Railway's 5-minute cut for a connection with no bytes flowing.
    expect(new OpenAiController().timeoutMs).toBe(280000);
    expect(new OpenAiController({ timeoutMs: 5000 }).timeoutMs).toBe(5000); // still injectable
  });

  it('names the node that had the job when a non-streamed request times out', async () => {
    // The hard case to debug: a node claimed the job and went quiet. Without the
    // node id this looks identical to an empty fleet.
    const services = fakeServices({
      jobService: { getJobResult: async () => ({ status: 'assigned', assignedTo: 'n5' }) },
    });
    const ctrl = new OpenAiController({ services, pollMs: 1, timeoutMs: 20 });
    const res = fakeRes();
    await ctrl.chatCompletions(fakeReq({ messages: [{ role: 'user', content: 'x' }] }), res);
    expect(res.statusCode).toBe(504);
    expect(res.body.error.message).toContain('Node n5 took the job but produced no output');
    expect(res.body.error.served_by).toBe('n5');
    expect(res.body.error.job_status).toBe('assigned');
    expect(res.body.error.job_id).toBe('job-1');
    expect(res.headers['X-LLMJob-Served-By']).toBe('n5');
  });

  // checkTimeouts deletes an abandoned attempt's chunks when it requeues a job,
  // so the list a stream is walking can shrink. Holding the old high-water mark
  // would skip everything the retry produces until it grew past the dead
  // attempt's length — the caller would watch the answer stop mid-sentence.
  it('rewinds the stream cursor when a requeue drops the previous attempt\'s chunks', async () => {
    const pages = [
      { status: 'running', chunks: [{ content: 'A0 ' }, { content: 'A1 ' }, { content: 'A2 ' }] },
      { status: 'running', chunks: [] },                                  // requeued: chunks cleared
      { status: 'running', chunks: [{ content: 'B0 ' }] },                // retry rebuilds from idx 0
      { status: 'completed', chunks: [{ content: 'B0 ' }, { content: 'B1 ' }], result: 'B0 B1 ', assignedTo: 'n1' },
    ];
    let i = 0;
    const services = fakeServices({
      jobService: { getJobResult: async () => pages[Math.min(i++, pages.length - 1)] },
    });
    const ctrl = new OpenAiController({ services, pollMs: 1, timeoutMs: 5000 });
    const res = fakeRes();
    await ctrl.chatCompletions(fakeReq({ messages: [{ role: 'user', content: 'x' }], stream: true }), res);
    const out = res.writes.join('');
    // The retry's output reaches the caller instead of being swallowed.
    expect(out).toContain('B0 ');
    expect(out).toContain('B1 ');
    expect(out).toContain('[DONE]');
  });

  it('reports partial progress when a streamed request times out mid-generation', async () => {
    const services = fakeServices({
      jobService: { getJobResult: async () => ({ status: 'running', assignedTo: 'n7', chunks: [{ content: 'part' }] }) },
    });
    const ctrl = new OpenAiController({ services, pollMs: 1, timeoutMs: 20 });
    const res = fakeRes();
    await ctrl.chatCompletions(fakeReq({ messages: [{ role: 'user', content: 'x' }], stream: true }), res);
    const out = res.writes.join('');
    expect(out).toContain('Node n7 was still generating');
    expect(out).toContain('1 chunk(s) streamed');
    expect(out).toContain('"served_by":"n7"');
    expect(out).toContain('[DONE]');
  });

  it('timeoutBody defaults a result-less timeout to pending', () => {
    // Defensive: every caller passes a poll result, but a missing one must still
    // produce a well-formed error rather than throw inside the 504 path.
    const body = timeoutBody('job-9', null, 280000);
    expect(body.error).toEqual({
      message: 'No node picked the job up within 280s. Is a node online and serving?',
      type: 'timeout_error', code: null, job_id: 'job-9', served_by: null, job_status: 'pending',
    });
  });

  it('stops the non-streaming poll when the caller hangs up', async () => {
    const services = fakeServices({ jobService: { getJobResult: async () => ({ status: 'pending' }) } });
    const ctrl = new OpenAiController({ services, pollMs: 5, timeoutMs: 60000 });
    const res = fakeResClosable();
    const p = ctrl.chatCompletions(fakeReq({ messages: [{ role: 'user', content: 'x' }] }), res);
    await sleep(20);   // let it poll a couple of times against a job that never completes
    res.emitClose();   // client disconnects
    await p;           // resolves promptly instead of running to the 60s timeout
    expect(res.body).toBeNull();      // never sent a completion
    expect(res.statusCode).toBe(200); // untouched default
  });

  it('stops a stream when the caller hangs up (no [DONE], no end)', async () => {
    const services = fakeServices({ jobService: { getJobResult: async () => ({ status: 'running', chunks: [] }) } });
    const ctrl = new OpenAiController({ services, pollMs: 5, timeoutMs: 60000 });
    const res = fakeResClosable();
    const p = ctrl.chatCompletions(fakeReq({ messages: [{ role: 'user', content: 'x' }], stream: true }), res);
    await sleep(20);
    res.emitClose();
    await p;
    expect(res.writes.some((w) => w.includes('"role":"assistant"'))).toBe(true); // opened the stream
    expect(res.writes.join('')).not.toContain('[DONE]');                          // bailed before finishing
    expect(res.ended).toBe(false);
  });

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
    // model falls back to the fleet default (not the request); usage estimated; node 'unknown'
    expect(recorded[0]).toMatchObject({ model: DEFAULT_MODEL, node: 'unknown', speed: 0 });
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
    expect(jsonRes.body.error.message).toBe('The node failed to run the job: unknown error');
    expect(jsonRes.body.error.served_by).toBeNull(); // failed before anyone claimed it
    expect(jsonRes.headers['X-LLMJob-Served-By']).toBeUndefined();

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

  it('modelName reports the node-run model, else the fleet default — never the request', () => {
    // The caller's requested model never picks what runs, so it must never be
    // echoed: a node that tagged its metrics wins, otherwise the fleet default.
    expect(modelName({ metrics: { model: 'X' } })).toBe('X');
    expect(modelName({ metrics: {} })).toBe(DEFAULT_MODEL);
    expect(modelName(null)).toBe(DEFAULT_MODEL);
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
