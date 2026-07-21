const JobService = require('../services/jobService');
const LogService = require('../services/logService');
const ApiKeyService = require('../services/apiKeyService');

// OpenAI-compatible chat-completions gateway.
//
// POST /v1/chat/completions (authenticated with an `lj-` API key) turns a
// standard OpenAI request into an LLMJob inference job, waits for an online node
// in the fleet to serve it against its local model, and returns the result in
// OpenAI's shape — non-streaming JSON or an SSE `chat.completion.chunk` stream.
// This is the front door that makes the API key mean something: callers use it
// like `https://<host>/v1` with any OpenAI SDK, and it fans out to whatever node
// picks the job up. Usage (tokens + speed) is recorded against the key on finish.
//
// The node side (earn/src/main/jobWorker.js) polls, runs, and streams chunks
// back; this controller only creates the job and long-polls its result.
class OpenAiController {
  constructor(opts = {}) {
    this.pollMs = opts.pollMs || 250;          // how often to check the job for progress
    this.timeoutMs = opts.timeoutMs || 120000; // give up if no node finishes it in time
    this.now = opts.now || (() => Date.now());
    this.sleep = opts.sleep || ((ms) => new Promise((r) => setTimeout(r, ms)));
    // Services are built per-request from req.app.locals.db so one controller
    // instance can be registered before the DB pool is connected. Injectable for
    // tests.
    this._services = opts.services || null;
  }

  services(req) {
    if (this._services) return this._services;
    const db = req.app.locals.db;
    return { jobService: new JobService(db), logService: new LogService(db), apiKeyService: new ApiKeyService(db) };
  }

  // POST /v1/chat/completions
  async chatCompletions(req, res) {
    const body = req.body || {};
    const messages = body.messages;
    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json(errorBody('`messages` must be a non-empty array', 'invalid_request_error'));
    }

    const svc = this.services(req);
    const job = await svc.jobService.createJob({
      prompt: lastUserText(messages),   // display/fallback for nodes that read prompt
      messages,
      model: body.model || undefined,   // jobService fills the fleet default
      maxTokens: body.max_tokens,
      temperature: body.temperature,
      userId: req.apiKey.userId,
    });

    const ctx = { res, svc, job, key: req.apiKey, promptTokens: estimateTokens(joinContent(messages)), aborted: false };
    // If the caller hangs up mid-request, stop the long-poll instead of querying
    // the DB and writing to a dead socket until the job finishes or times out.
    if (res.on) res.on('close', () => { ctx.aborted = true; });
    try {
      if (body.stream === true) await this._streamResult(ctx);
      else await this._jsonResult(ctx);
    } catch (err) {
      if (!res.headersSent) res.status(500).json(errorBody('Gateway error: ' + err.message, 'api_error'));
      else res.end();
    } finally {
      // Best-effort usage accounting — never blocks or breaks the response.
      try { await this._recordUsage(ctx); } catch (e) { /* ignore */ }
    }
  }

  // Poll the job until it finishes, then return one OpenAI chat.completion.
  async _jsonResult(ctx) {
    const { res, svc, job } = ctx;
    const started = this.now();
    for (;;) {
      if (ctx.aborted) return; // caller hung up — stop polling, the socket is gone
      const r = await svc.jobService.getJobResult(job.id);
      if (r.status === 'completed') {
        const out = completionTokens(r);
        return res.status(200).json({
          id: 'chatcmpl-' + job.id,
          object: 'chat.completion',
          created: Math.floor(this.now() / 1000),
          model: modelName(r, job),
          choices: [{ index: 0, message: { role: 'assistant', content: r.result || '' }, finish_reason: 'stop' }],
          usage: { prompt_tokens: ctx.promptTokens, completion_tokens: out, total_tokens: ctx.promptTokens + out },
        });
      }
      if (r.status === 'failed') {
        return res.status(502).json(errorBody('The node failed to run the job: ' + (r.error || 'unknown error'), 'node_error'));
      }
      if (this.now() - started > this.timeoutMs) {
        return res.status(504).json(errorBody('No node produced a result before the timeout. Is a node online and serving?', 'timeout_error'));
      }
      await this.sleep(this.pollMs);
    }
  }

  // Stream the job's chunks as OpenAI chat.completion.chunk SSE events.
  async _streamResult(ctx) {
    const { res, svc, job } = ctx;
    res.status(200);
    res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache, no-transform');
    res.setHeader('Connection', 'keep-alive');
    if (res.flushHeaders) res.flushHeaders();

    const id = 'chatcmpl-' + job.id;
    const created = Math.floor(this.now() / 1000);
    const send = (delta, finish) => res.write('data: ' + JSON.stringify({
      id, object: 'chat.completion.chunk', created, model: modelName(null, job),
      choices: [{ index: 0, delta, finish_reason: finish || null }],
    }) + '\n\n');

    send({ role: 'assistant' }); // OpenAI opens with the role

    const started = this.now();
    let emitted = 0;
    for (;;) {
      if (ctx.aborted) return; // caller hung up — stop; skip the [DONE]/end writes
      const r = await svc.jobService.getJobResult(job.id);
      const chunks = r.chunks || [];
      for (; emitted < chunks.length; emitted++) {
        if (chunks[emitted].content) send({ content: chunks[emitted].content });
      }
      if (r.status === 'completed') {
        send({}, 'stop');
        break;
      }
      if (r.status === 'failed') {
        res.write('data: ' + JSON.stringify(errorBody('The node failed to run the job: ' + (r.error || 'unknown error'), 'node_error')) + '\n\n');
        break;
      }
      if (this.now() - started > this.timeoutMs) {
        res.write('data: ' + JSON.stringify(errorBody('No node produced a result before the timeout.', 'timeout_error')) + '\n\n');
        break;
      }
      await this.sleep(this.pollMs);
    }
    res.write('data: [DONE]\n\n');
    res.end();
  }

  // Record token usage + a request-log row against the API key, once the job is
  // done. Reads the final result fresh so it works after either response path.
  async _recordUsage(ctx) {
    const { svc, job, key } = ctx;
    const r = await svc.jobService.getJobResult(job.id);
    if (r.status !== 'completed') return;
    const out = completionTokens(r);
    await svc.logService.recordLog(key.userId, {
      model: modelName(r, job),
      node: r.assignedTo || 'unknown',
      app: 'api',
      in: ctx.promptTokens,
      out,
      speed: (r.metrics && r.metrics.tokensPerSecond) || 0,
      finish: 'stop',
      key: key.name,
    });
    await svc.apiKeyService.recordUsage(key.hash, ctx.promptTokens + out);
  }
}

// The last user message's text — the single-string prompt kept on the job record.
function lastUserText(messages) {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m && m.role === 'user' && m.content != null) return String(m.content);
  }
  return String(joinContent(messages)); // no user turn: fall back to all content
}

function joinContent(messages) {
  return messages.map((m) => (m && m.content) || '').join('\n');
}

// The model to report back: what the node actually ran (final metrics), else the
// job's model.
function modelName(result, job) {
  if (result && result.metrics && result.metrics.model) return result.metrics.model;
  return job.model;
}

// completion_tokens from the node's final metrics, falling back to an estimate of
// the assembled result when the node didn't report a count.
function completionTokens(result) {
  if (result.metrics && Number.isFinite(result.metrics.totalTokens)) return result.metrics.totalTokens;
  return estimateTokens(result.result || '');
}

// A rough token count (~4 chars/token) — good enough for usage display/billing of
// prompt tokens, which the node doesn't measure.
function estimateTokens(text) {
  return Math.ceil(String(text || '').length / 4);
}

function errorBody(message, type) {
  return { error: { message, type, code: null } };
}

module.exports = OpenAiController;
module.exports.lastUserText = lastUserText;
module.exports.estimateTokens = estimateTokens;
module.exports.modelName = modelName;
module.exports.completionTokens = completionTokens;
