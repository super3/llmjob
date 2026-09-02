const crypto = require('crypto');
const JobService = require('../services/jobService');
const LogService = require('../services/logService');
const ApiKeyService = require('../services/apiKeyService');
const NodeService = require('../services/nodeService');
const ChatUsageService = require('../services/chatUsageService');
const OpenRouterService = require('../services/openRouterService');
const {
  parseSSE, deltaContent, deltaReasoning, upstreamErrorMessage, logUpstreamError, usageMeta,
} = OpenRouterService;
const {
  estimateTokens, errorBody, joinContent, lastUserText,
  writeSsePreamble, pollJobResult, clampMessages, resolveMaxTokens, MAX_PROMPT_CHARS,
} = require('./gatewayShared');

// Per-request completion ceiling for this gateway. The web-chat gateway has
// always clamped max_tokens; here it rode through unbounded, so a single key
// could ask for millions of tokens and hold a node's GPU for as long as it took.
// Sized to the node's context window (earn/src/shared/config.js LLM.ctxSize).
// Still a real bound, just a larger one: what a caller can actually be served is
// governed by the 280s timeout regardless, which at fleet speeds lands well
// under this — the ceiling stops the unbounded ask, the budget stops the rest.
const MAX_COMPLETION_TOKENS = 32768;

// Header a caller sets to pin a request to one specific node (health/perf testing).
// Lowercase — Express lowercases header names. Kept OpenAI-SDK friendly: passable
// via the stock client's default/extra headers, no body-schema changes.
const TARGET_NODE_HEADER = 'x-llmjob-node';

// The model the fleet actually serves, for reporting when the node didn't tag its
// metrics with a model name (older clients). Same source the job default uses.
const { DEFAULT_MODEL } = JobService;

// What we report as the "node" for a hosted-model request — in the served-by
// header, and in the dashboard's log rows. No node ran it, and saying so is more
// useful than an empty column.
const HOSTED_SERVED_BY = 'openrouter';

// OpenAI-compatible chat-completions gateway.
//
// POST /v1/chat/completions (authenticated with an `lj-` API key) takes a
// standard OpenAI request and returns an OpenAI-shaped result — non-streaming
// JSON or an SSE `chat.completion.chunk` stream. This is the front door that
// makes the API key mean something: callers point any OpenAI SDK at
// `https://<host>/v1`. Usage (tokens + speed) is recorded against the key on
// finish, either way.
//
// Two backends sit behind it, chosen by the request's `model`:
//
//   • The node network (the default, and anything unrecognised). The request
//     becomes an LLMJob inference job and we long-poll until an online node
//     serves it against its own local model. The node side
//     (earn/src/main/jobWorker.js) polls, runs, and streams chunks back; this
//     controller only creates the job and waits.
//   • A hosted model — one of the OpenRouter-served models the public Chat page
//     offers — when the caller names it exactly. Only `public` keys may ask for
//     one, and the spend comes out of the same free budget the web chat draws
//     on. This exists so real API traffic can reach those models now, ahead of
//     the network serving them itself; an unknown model still falls through to
//     the network, so no existing caller changes behaviour.
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
    // The hosted-model backend, sharing its configuration (allow-list, key,
    // ceilings, free budget) with the web-chat gateway so both front doors serve
    // the same models under the same limits. `opts.openRouter` overrides it for
    // tests; otherwise it reads the same OPENROUTER_* env the chat gateway does.
    this.openRouter = new OpenRouterService(opts.openRouter || {});
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
      chatUsage: new ChatUsageService(db),
    };
  }

  // GET /v1/models — the models this key may name, in OpenAI's list shape.
  //
  // The network entry is the model the fleet actually runs; any unrecognised
  // model id is served by it too, so this is a guide rather than a whitelist.
  // The hosted models are listed only when they are actually reachable: a
  // private key's requests never leave its owner's nodes, so advertising them
  // there would just produce a 403 on the next call.
  async listModels(req, res) {
    const created = Math.floor(this.now() / 1000);
    const data = [];
    if (this.openRouter.configured && req.apiKey.visibility !== 'private') {
      for (const m of this.openRouter.models) {
        data.push({ id: m.id, object: 'model', created, owned_by: 'llmjob-hosted' });
      }
    }
    // What the fleet is actually running, so a caller can NAME one and reach it.
    // Listing only a hardcoded default meant the one model anybody could discover
    // was also the only one they could not choose to avoid.
    //
    // Best-effort: this endpoint is a guide, and a database hiccup should degrade
    // it to the default rather than fail a request that was only asking what is
    // available.
    let live = [];
    try { live = await this.services(req).nodeService.listNetworkModels(); } catch (e) { live = []; }
    // Keyed case-insensitively, to match how a requested name is resolved against
    // what nodes report: the same model spelled two ways is one model, and
    // listing it twice would suggest a choice that does not exist.
    const seen = new Set();
    for (const m of live) {
      const key = String(m.id).toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      data.push({ id: m.id, object: 'model', created, owned_by: 'llmjob-network', nodes: m.nodes });
    }
    // Keep the default listed even when no node reports it: it is the model an
    // unpinned job is recorded against, so a caller seeing it in the list and
    // sending it back gets exactly what they already get by sending nothing.
    if (!seen.has(DEFAULT_MODEL.toLowerCase())) {
      data.push({ id: DEFAULT_MODEL, object: 'model', created, owned_by: 'llmjob-network' });
    }
    res.json({ object: 'list', data });
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

    // Bound the request before it goes anywhere. Prompt size and completion
    // budget are both caller-controlled and both cost real resources — a node's
    // GPU time, or OpenRouter credit; the web-chat gateway has always clamped
    // them and this one didn't.
    const clean = clampMessages(messages, MAX_PROMPT_CHARS);
    if (clean.length === 0) {
      return res.status(400).json(errorBody('No usable message content.', 'invalid_request_error'));
    }

    const targetNode = (this._headerTarget(req) || '').trim() || null;

    // A hosted model, named exactly? Then this request leaves for OpenRouter
    // rather than the fleet. Anything else — including no model at all, and every
    // alias callers have always sent — goes to the node network as before.
    const hosted = this.openRouter.resolveModel(body.model);
    if (hosted) return this._hostedCompletion(req, res, { svc, body, clean, hosted, targetNode });

    // Optional node targeting (X-LLMJob-Node): pin the request to one node so a
    // caller can test whether that node serves and how fast. Fast-fail if it's
    // offline/unknown rather than long-polling to the timeout — the point of the
    // feature is a quick verdict. It only narrows: a targeted job still passes the
    // normal visibility filter at assignment, so this can't reach a node the key
    // isn't already allowed to use.
    if (targetNode) {
      const status = await svc.nodeService.getNodeStatus(targetNode);
      if (!status.online) {
        const why = status.exists ? 'is offline' : 'is not a known node';
        return res.status(404).json(errorBody(`Target node ${targetNode} ${why}.`, 'target_node_error'));
      }
    }

    const job = await svc.jobService.createJob({
      prompt: lastUserText(clean),      // display/fallback for nodes that read prompt
      messages: clean,
      // Still intentionally NOT body.model: a passed model rides through to the
      // node's reported metrics.model and back out as the "model that ran", which
      // is how a request for "gpt-4"/"llmjob" got echoed as if the fleet ran it.
      // jobService fills the real fleet default end to end.
      //
      // requestedModel is the separate ROUTING channel. It never reaches
      // data.model, so the echo stays honest; it only decides WHICH node may take
      // the job, and only when the name matches a model the fleet is actually
      // running. An unknown name resolves to no pin, i.e. exactly today's
      // behaviour -- which is what keeps /v1/models a guide rather than a
      // whitelist.
      requestedModel: body.model,
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

  // ── Hosted models (OpenRouter) ──────────────────────────────────────────────

  // Gate a hosted-model request, then serve it. Every rejection here is a
  // deliberate one — the caller named a model we proxy, so silence would look
  // like the network had served it.
  async _hostedCompletion(req, res, { svc, body, clean, hosted, targetNode }) {
    const key = req.apiKey;

    // Private means "my own machines only". A hosted model is neither, so
    // serving one would quietly break the promise the toggle makes; say so
    // instead, and name the fix.
    if (key.visibility === 'private') {
      return res.status(403).json(errorBody(
        `Model ${hosted.id} runs on LLMJob's hosted backend, and this key is private — its requests only run on your own nodes. `
        + 'Switch the key to public in the dashboard, or omit `model` to use the node network.',
        'permission_error'));
    }
    // Pinning a node and asking for a model no node runs are contradictory. The
    // targeting header exists to give a fast verdict about one machine, so
    // ignoring it here would be the one answer that helps nobody.
    if (targetNode) {
      return res.status(400).json(errorBody(
        `Model ${hosted.id} is not served by the node network, so it cannot be pinned to node ${targetNode}.`,
        'invalid_request_error'));
    }
    if (!this.openRouter.configured) {
      return res.status(503).json(errorBody('Hosted models are not configured on this deployment.', 'not_configured'));
    }
    // The same pot of credit the free web chat draws on, and the same cap. An API
    // key that could spend past it would drain exactly what the cap protects.
    const totals = await svc.chatUsage.getTotals();
    if (this.openRouter.freeBudget > 0 && totals.totalTokens >= this.openRouter.freeBudget) {
      return res.status(402).json(errorBody(
        'Hosted models have reached their free usage cap for now — omit `model` to run this request on the node network.',
        'quota_exhausted'));
    }

    const ctx = {
      res, svc, key,
      id: 'chatcmpl-' + crypto.randomBytes(12).toString('hex'),
      created: Math.floor(this.now() / 1000),
      messages: clean,
      promptText: joinContent(clean),
      modelId: hosted.id,
      requestedLabel: hosted.id, // what to report if the provider names no model
      maxTokens: this.openRouter.resolveMaxTokens(body.max_tokens),
      temperature: typeof body.temperature === 'number' ? body.temperature : null,
      start: this.now(), firstTokenAt: 0,
      text: '', reasoning: '', usage: null, model: null, finish: 'stop', aborted: false,
    };
    ctx.controller = new (globalThis.AbortController)();
    // Caller hung up — abort the upstream generation rather than paying for
    // tokens nobody will read, and stop writing to a dead socket.
    if (res.on) {
      res.on('close', () => {
        ctx.aborted = true;
        try { ctx.controller.abort(); } catch (e) { /* ignore */ }
      });
    }

    try {
      if (body.stream === true) await this._streamHosted(ctx);
      else await this._jsonHosted(ctx);
    } catch (err) {
      if (!res.headersSent) res.status(500).json(errorBody('Gateway error: ' + err.message, 'api_error'));
      else { try { res.end(); } catch (e) { /* ignore */ } }
    }
  }

  _callHosted(ctx, stream) {
    return this.openRouter.send({
      model: ctx.modelId,
      messages: ctx.messages,
      maxTokens: ctx.maxTokens,
      temperature: ctx.temperature,
      stream,
      signal: ctx.controller.signal,
    });
  }

  // One upstream call, one OpenAI chat.completion back.
  async _jsonHosted(ctx) {
    const { res } = ctx;
    let upstream;
    try {
      upstream = await this._callHosted(ctx, false);
    } catch (err) {
      return res.status(502).json(errorBody('Upstream request failed.', 'upstream_error'));
    }
    if (!upstream.ok) {
      const detail = await upstreamErrorMessage(upstream);
      logUpstreamError('api', upstream.status, detail);
      return res.status(502).json(errorBody('The model backend returned an error: ' + detail, 'upstream_error'));
    }

    const data = await upstream.json();
    const choice = (data.choices && data.choices[0]) || null;
    const msg = (choice && choice.message) || null;
    ctx.text = (msg && msg.content) || '';
    ctx.reasoning = (msg && (msg.reasoning_content || msg.reasoning)) || '';
    ctx.usage = data.usage || null;
    ctx.model = data.model || null;
    ctx.finish = (choice && choice.finish_reason) || 'stop';
    ctx.firstTokenAt = ctx.start; // no TTFT signal — attribute speed to full latency

    const meta = usageMeta(ctx, this.now());
    await this._recordHostedUsage(ctx, meta);

    const message = { role: 'assistant', content: ctx.text };
    if (ctx.reasoning) message.reasoning_content = ctx.reasoning;
    this._setServedByHeader(res, { assignedTo: HOSTED_SERVED_BY });
    return res.status(200).json({
      id: ctx.id,
      object: 'chat.completion',
      created: ctx.created,
      model: meta.model,
      choices: [{ index: 0, message, finish_reason: meta.finish }],
      usage: {
        prompt_tokens: meta.promptTokens,
        completion_tokens: meta.completionTokens,
        total_tokens: meta.totalTokens,
      },
    });
  }

  // Re-emit the upstream generation as our own chat.completion.chunk stream.
  // Deliberately not a byte-for-byte passthrough: the ids, the `model` field and
  // the terminator then read the same whether a node or a hosted model served
  // the request, and we still see every token for the usage accounting.
  async _streamHosted(ctx) {
    const { res } = ctx;
    writeSsePreamble(res);
    const send = (delta, finish) => res.write('data: ' + JSON.stringify({
      id: ctx.id, object: 'chat.completion.chunk', created: ctx.created,
      model: ctx.model || ctx.requestedLabel,
      choices: [{ index: 0, delta, finish_reason: finish || null }],
    }) + '\n\n');
    const fail = (message, type) => {
      res.write('data: ' + JSON.stringify(errorBody(message, type)) + '\n\n');
      res.write('data: [DONE]\n\n');
      res.end();
    };

    let upstream;
    try {
      upstream = await this._callHosted(ctx, true);
    } catch (err) {
      return fail('Upstream request failed.', 'upstream_error');
    }
    if (!upstream.ok) {
      const detail = await upstreamErrorMessage(upstream);
      logUpstreamError('api', upstream.status, detail);
      return fail('The model backend returned an error: ' + detail, 'upstream_error');
    }

    send({ role: 'assistant' }); // OpenAI opens with the role
    try {
      for await (const obj of parseSSE(upstream.body)) {
        if (ctx.aborted) break;
        if (obj.model) ctx.model = obj.model; // chunks name the real model once it's known
        const content = deltaContent(obj);
        if (content) {
          if (!ctx.firstTokenAt) ctx.firstTokenAt = this.now();
          ctx.text += content;
          send({ content });
        }
        // Surfaced only for a thinking model. Clients that don't know the field
        // ignore it; the ones that do can show why `content` is short (or empty)
        // when the completion budget ran out mid-thought.
        const reasoning = deltaReasoning(obj);
        if (reasoning) {
          ctx.reasoning += reasoning;
          send({ reasoning_content: reasoning });
        }
        if (obj.usage) ctx.usage = obj.usage;
        const fr = obj.choices && obj.choices[0] && obj.choices[0].finish_reason;
        if (fr) ctx.finish = fr;
      }
    } finally {
      // Bill whatever OpenRouter already streamed, even if the caller hung up
      // mid-generation: it produced (and charged us for) those tokens whether or
      // not anyone was still listening, so the budget has to see them.
      await this._recordHostedUsage(ctx, usageMeta(ctx, this.now()));
    }
    if (ctx.aborted) return; // socket gone — skip the finish/[DONE] writes
    send({}, ctx.finish);
    res.write('data: [DONE]\n\n');
    res.end();
  }

  // Record a completed hosted generation in all three places it belongs, and
  // never let a bookkeeping failure break the response:
  //   • chat_usage_totals — this is OpenRouter spend, so it counts against the
  //     same free cap the web chat draws on (`viaApiKey` tags the slice so the
  //     public totals don't count it twice against the key's own usage);
  //   • the user's request log — so the dashboard shows the request like any
  //     other, attributed to `openrouter` rather than to a node;
  //   • the key's lifetime usage.
  async _recordHostedUsage(ctx, meta) {
    const { svc, key } = ctx;
    try {
      await svc.chatUsage.recordUsage({
        model: meta.model,
        inTokens: meta.promptTokens,
        outTokens: meta.completionTokens,
        speed: meta.tokensPerSecond,
        latencyMs: meta.latencyMs,
        ttftMs: meta.ttftMs,
        finish: meta.finish,
        viaApiKey: true,
      });
      await svc.logService.recordLog(key.userId, {
        model: meta.model,
        node: HOSTED_SERVED_BY,
        app: 'api',
        in: meta.promptTokens,
        out: meta.completionTokens,
        speed: meta.tokensPerSecond,
        finish: meta.finish,
        key: key.name,
      });
      await svc.apiKeyService.recordUsage(key.hash, meta.totalTokens);
    } catch (e) { /* ignore */ }
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
