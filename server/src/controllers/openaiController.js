const JobService = require('../services/jobService');
const LogService = require('../services/logService');
const ApiKeyService = require('../services/apiKeyService');
const NodeService = require('../services/nodeService');
const {
  estimateTokens, errorBody, joinContent, lastUserText,
  writeSsePreamble, pollJobResult, clampMessages, resolveMaxTokens, MAX_PROMPT_CHARS,
} = require('./gatewayShared');

// Per-request completion ceiling for this gateway. The web-chat gateway has
// always clamped max_tokens; here it rode through unbounded, so a single key
// could ask for millions of tokens and hold a node's GPU for as long as it took.
// Sized to the node's context window (earn/src/shared/config.js LLM.ctxSize).
const MAX_COMPLETION_TOKENS = 6400;

// Header a caller sets to pin a request to one specific node (health/perf testing).
// Lowercase — Express lowercases header names. Kept OpenAI-SDK friendly: passable
// via the stock client's default/extra headers, no body-schema changes.
const TARGET_NODE_HEADER = 'x-llmjob-node';

// The model the fleet actually serves, for reporting when the node didn't tag its
// metrics with a model name (older clients). Same source the job default uses.
const { DEFAULT_MODEL } = JobService;

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
    // Give up if no node finishes in time. 280s sits just under Railway's 5-minute
    // cut for a connection with no bytes flowing, which is what the non-streaming
    // path looks like while it long-polls. The old 120s was well inside that budget
    // and was the real cap on generation length: at a node's ~24 tok/s it stopped a
    // reasoning model at roughly 2,900 tokens, so hard prompts 504'd rather than
    // finishing. (Streaming keeps bytes moving and could run to Railway's 15-minute
    // ceiling, but one timeout for both paths keeps the contract simple.)
    this.timeoutMs = opts.timeoutMs || 280000;
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
    return {
      jobService: new JobService(db), logService: new LogService(db),
      apiKeyService: new ApiKeyService(db), nodeService: new NodeService(db),
    };
  }

  // The requested target node, from the X-LLMJob-Node header (Express lowercases
  // header keys). Absent → undefined → no targeting.
  _headerTarget(req) {
    const h = req && req.headers;
    return h ? h[TARGET_NODE_HEADER] : undefined;
  }

  // Report which node served the request as a response header — so a caller
  // testing a node confirms it actually served, without a second job-status
  // lookup. Non-streaming only: a stream flushes headers before any node has the
  // job. (Throughput is deliberately not reported: the node-side tok/s is measured
  // over the whole job including model load, so it isn't a consistent metric yet.)
  _setServedByHeader(res, result) {
    if (!res.setHeader) return;
    const node = result && result.assignedTo;
    if (node) res.setHeader('X-LLMJob-Served-By', String(node));
  }

  // POST /v1/chat/completions
  async chatCompletions(req, res) {
    const body = req.body || {};
    const messages = body.messages;
    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json(errorBody('`messages` must be a non-empty array', 'invalid_request_error'));
    }

    const svc = this.services(req);

    // Optional node targeting (X-LLMJob-Node): pin the request to one node so a
    // caller can test whether that node serves and how fast. Fast-fail if it's
    // offline/unknown rather than long-polling to the timeout — the point of the
    // feature is a quick verdict. It only narrows: a targeted job still passes the
    // normal visibility filter at assignment, so this can't reach a node the key
    // isn't already allowed to use.
    const targetNode = (this._headerTarget(req) || '').trim() || null;
    if (targetNode) {
      const status = await svc.nodeService.getNodeStatus(targetNode);
      if (!status.online) {
        const why = status.exists ? 'is offline' : 'is not a known node';
        return res.status(404).json(errorBody(`Target node ${targetNode} ${why}.`, 'target_node_error'));
      }
    }

    // Bound the request before it becomes a job. Prompt size and completion
    // budget are both caller-controlled and both cost a node's GPU time; the
    // web-chat gateway has always clamped them and this one didn't.
    const clean = clampMessages(messages, MAX_PROMPT_CHARS);
    if (clean.length === 0) {
      return res.status(400).json(errorBody('No usable message content.', 'invalid_request_error'));
    }

    const job = await svc.jobService.createJob({
      prompt: lastUserText(clean),      // display/fallback for nodes that read prompt
      messages: clean,
      // Intentionally NOT body.model: the node serves its own local model no matter
      // what the caller asks for, and a passed model rides through to the node's
      // reported metrics.model and back out as the "model that ran" — which is how
      // a request for "gpt-4"/"llmjob" got echoed as if the fleet ran it. Dropping
      // it here lets jobService fill the real fleet default end to end.
      maxTokens: resolveMaxTokens(body.max_tokens, MAX_COMPLETION_TOKENS),
      temperature: body.temperature,
      userId: req.apiKey.userId,
      visibility: req.apiKey.visibility, // 'private' → only this user's own nodes
      targetNode,                        // null unless X-LLMJob-Node was set
    });

    const ctx = { res, svc, job, key: req.apiKey, promptTokens: estimateTokens(joinContent(clean)), aborted: false };
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
    for await (const r of pollJobResult({
      jobService: svc.jobService, jobId: job.id,
      now: this.now, sleep: this.sleep, pollMs: this.pollMs, timeoutMs: this.timeoutMs,
      isAborted: () => ctx.aborted, // caller hung up — stop polling, the socket is gone
    })) {
      if (r.status === 'timeout') {
        // Same header as a success, so a caller reads "who served this" the same
        // way whether the job finished or ran out the clock.
        this._setServedByHeader(res, r.last);
        return res.status(504).json(timeoutBody(job.id, r.last, this.timeoutMs));
      }
      if (r.status === 'failed') {
        // Same diagnostics (node id, job id) a 504 carries, so a failed job is as
        // attributable as a timed-out one.
        this._setServedByHeader(res, r);
        return res.status(502).json(nodeErrorBody(job.id, r));
      }
      if (r.status === 'completed') {
        const out = completionTokens(r);
        const message = { role: 'assistant', content: r.result || '' };
        // Surfaced only when the node ran a thinking model. Clients that don't
        // know the field ignore it; the ones that do can show why `content` is
        // short (or empty) when max_tokens ran out mid-thought.
        const thoughts = reasoningText(r);
        if (thoughts) message.reasoning_content = thoughts;
        this._setServedByHeader(res, r);
        return res.status(200).json({
          id: 'chatcmpl-' + job.id,
          object: 'chat.completion',
          created: Math.floor(this.now() / 1000),
          model: modelName(r),
          choices: [{ index: 0, message, finish_reason: finishReason(r) }],
          usage: { prompt_tokens: ctx.promptTokens, completion_tokens: out, total_tokens: ctx.promptTokens + out },
        });
      }
      // pending/assigned/running: keep polling until terminal, timeout, or abort.
    }
  }

  // Stream the job's chunks as OpenAI chat.completion.chunk SSE events.
  async _streamResult(ctx) {
    const { res, svc, job } = ctx;
    writeSsePreamble(res);

    const id = 'chatcmpl-' + job.id;
    const created = Math.floor(this.now() / 1000);
    // `model` is resolved per write, not captured once: the node's real model only
    // becomes known when its first metrics arrive, and hardcoding modelName(null)
    // meant every streamed chunk reported the fleet default even after the node
    // had said what it actually ran.
    let served = null;
    const send = (delta, finish) => res.write('data: ' + JSON.stringify({
      id, object: 'chat.completion.chunk', created, model: modelName(served),
      choices: [{ index: 0, delta, finish_reason: finish || null }],
    }) + '\n\n');

    send({ role: 'assistant' }); // OpenAI opens with the role

    let emitted = 0;
    for await (const r of pollJobResult({
      jobService: svc.jobService, jobId: job.id,
      now: this.now, sleep: this.sleep, pollMs: this.pollMs, timeoutMs: this.timeoutMs,
      isAborted: () => ctx.aborted,
    })) {
      if (r.status === 'timeout') {
        // A stream flushed its headers long before this, so the diagnostics can
        // only ride in the event body.
        res.write('data: ' + JSON.stringify(timeoutBody(job.id, r.last, this.timeoutMs)) + '\n\n');
        break;
      }
      served = r; // once the node reports metrics, chunks name the real model
      const chunks = r.chunks || [];
      // A shrinking chunk list means the job was requeued and its abandoned
      // attempt's partials were dropped, so the retry is rebuilding from idx 0.
      //
      // Rewinding replays: the caller has already received the dead attempt's
      // text and now receives the retry's in full, so a requeue mid-stream shows
      // up as a restarted answer. That is the deliberate trade. SSE has no way to
      // retract what was already written, and the alternative — holding the old
      // high-water mark — skips everything the retry produces until it grows past
      // the dead attempt's length, so the caller watches the answer stop
      // mid-sentence and never resume. A visible restart is recoverable; silent
      // truncation is not, and it corrupts the non-streaming result too.
      if (chunks.length < emitted) emitted = 0;
      for (; emitted < chunks.length; emitted++) {
        if (chunks[emitted].content) send({ content: chunks[emitted].content });
      }
      if (r.status === 'completed') {
        const thoughts = reasoningText(r);
        if (thoughts) send({ reasoning_content: thoughts });
        send({}, finishReason(r));
        break;
      }
      if (r.status === 'failed') {
        res.write('data: ' + JSON.stringify(nodeErrorBody(job.id, r)) + '\n\n');
        break;
      }
    }
    if (ctx.aborted) return; // caller hung up — skip the [DONE]/end writes
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
      model: modelName(r),
      node: r.assignedTo || 'unknown',
      app: 'api',
      in: ctx.promptTokens,
      out,
      speed: (r.metrics && r.metrics.tokensPerSecond) || 0,
      finish: finishReason(r),
      key: key.name,
    });
    await svc.apiKeyService.recordUsage(key.hash, ctx.promptTokens + out);
  }
}

// The model to report back: what the node actually ran (from its final metrics),
// else the fleet's default. Deliberately NOT the caller's requested model — the
// node serves its own local model regardless of the `model` field, so echoing the
// request would report a lie. A caller who sent `model: "llmjob"` (or any alias)
// was still served Gemma; the log and the response should say so.
function modelName(result) {
  if (result && result.metrics && result.metrics.model) return result.metrics.model;
  return DEFAULT_MODEL;
}

// Why generation stopped, as reported by the node's final metrics. 'length' means
// max_tokens ran out — the signal a caller needs to tell a truncated answer from a
// finished one, and the only way an empty `content` from a thinking model reads as
// anything but a silent failure. Older nodes don't send it; they were always
// stopping normally, so 'stop' is the right default.
function finishReason(result) {
  const reason = result && result.metrics && result.metrics.finishReason;
  return typeof reason === 'string' && reason ? reason : 'stop';
}

// The chain of thought a thinking model produced, carried on the final chunk.
// Empty for ordinary models.
function reasoningText(result) {
  const chunks = (result && result.chunks) || [];
  return chunks.map((c) => (c && c.reasoning) || '').join('');
}

// completion_tokens from the node's final metrics, falling back to an estimate of
// the assembled result when the node didn't report a count.
function completionTokens(result) {
  if (result.metrics && Number.isFinite(result.metrics.totalTokens)) return result.metrics.totalTokens;
  return estimateTokens(result.result || '');
}

// A 504 that says what actually went wrong. "No node produced a result before
// the timeout" is the same sentence whether the fleet was empty, a node claimed
// the job and never spoke again, or a node was mid-generation when the clock ran
// out — three very different problems. The job's last known state tells them
// apart, so carry it: the id to look the job up by, the node that had it, and
// where it got to. Without these a timed-out request is unattributable — a
// benchmark run can report which node served every success and nothing at all
// about its failures.
//
// The extra fields hang off the standard OpenAI `error` object, so a strict SDK
// still parses it as an ordinary error and only clients that look find them.
// The 502 twin of timeoutBody. #165 gave timeouts a node id and left this path
// alone, which showed up on the very next benchmark run: six 504s named the node
// that had them while four 502s came back anonymous, so the failures we could
// investigate were decided by which branch happened to fire.
function nodeErrorBody(jobId, result) {
  const reason = (result && result.error) || 'unknown error';
  const node = (result && result.assignedTo) || null;
  const who = node ? `Node ${node}` : 'The node';
  const body = errorBody(`${who} failed to run the job: ${reason}`, 'node_error');
  body.error.job_id = jobId;
  body.error.served_by = node;
  body.error.job_status = 'failed';
  return body;
}

function timeoutBody(jobId, result, timeoutMs) {
  const secs = Math.round(timeoutMs / 1000);
  const node = (result && result.assignedTo) || null;
  const status = (result && result.status) || 'pending';
  const chunks = ((result && result.chunks) || []).length;

  let message;
  if (!node) {
    message = `No node picked the job up within ${secs}s. Is a node online and serving?`;
  } else if (chunks > 0) {
    message = `Node ${node} was still generating after ${secs}s (${chunks} chunk(s) streamed).`;
  } else {
    message = `Node ${node} took the job but produced no output within ${secs}s.`;
  }

  const body = errorBody(message, 'timeout_error');
  body.error.job_id = jobId;
  body.error.served_by = node;
  body.error.job_status = status;
  return body;
}

module.exports = OpenAiController;
module.exports.lastUserText = lastUserText;
module.exports.estimateTokens = estimateTokens;
module.exports.modelName = modelName;
module.exports.completionTokens = completionTokens;
module.exports.timeoutBody = timeoutBody;
module.exports.nodeErrorBody = nodeErrorBody;
