const ChatUsageService = require('../services/chatUsageService');
const JobService = require('../services/jobService');
const ApiKeyService = require('../services/apiKeyService');
const OpenRouterService = require('../services/openRouterService');
const {
  parseSSE, deltaContent, upstreamErrorMessage, logUpstreamError, usageMeta,
} = OpenRouterService;
const {
  estimateTokens, round1, errorBody, lastUserText, nodeFailMessage,
  writeSsePreamble, pollJobResult,
} = require('./gatewayShared');

// Free public web-chat gateway, backed by OpenRouter.
//
// The website's Chat page (/chat) talks to this proxy instead of OpenRouter
// directly, so the OpenRouter API key never leaves the server and every request
// is gated by our own rules:
//
//   • Free-usage cap    — a global running token total (chat_usage_totals). Once
//                         it crosses OPENROUTER_FREE_TOKEN_BUDGET the endpoint
//                         returns 402 until we flip chat over to the LLMJob node
//                         network.
//   • No prompt logging — we record performance (latency, TTFT, tok/s) and token
//                         counts, but never the prompt or reply text.
//   • Bounded cost      — only an allow-listed set of models is reachable, and
//                         max_tokens / prompt length are clamped server-side.
//
// The OpenRouter half of that (key, base URL, allow-list, budget, ceilings, the
// upstream call itself) lives in services/openRouterService.js, because the /v1
// API gateway serves the same hosted models to public API keys and must do so
// under exactly the same limits.
//
// The client protocol is a small SSE stream of our own shape (not raw OpenAI):
//   data: {"delta":"..."}                 incremental text
//   data: {"done":true,"meta":{...}}      final token/perf summary
//   data: {"error":"..."}                 upstream/gateway failure
//   data: [DONE]                          terminator
// Non-streaming callers (stream:false) get a single JSON body instead.

// The one model served by the LLMJob node fleet itself (not OpenRouter): a public
// chat request for it becomes an inference job that a live node runs on its own
// GPU. Always offered alongside the OpenRouter models, but never the default —
// callers opt in by selecting it. Its served model id is the fleet default in
// jobService (JobService fills it in when the job omits `model`).
const NETWORK_MODEL = { id: 'llmjob-gemma-4-e4b', label: 'Gemma 4 E4B' };
const MAX_PROMPT_CHARS = 24000;      // total prompt characters kept per request

// Injected as the system message on every request so the model has context about
// LLMJob (the suggestion prompts — "What is LLMJob?", "What is PPLNS?" — are
// unanswerable without it). Deliberately factual and hedged: it avoids inventing
// specific payout numbers. Override with OPENROUTER_SYSTEM_PROMPT.
const DEFAULT_SYSTEM_PROMPT = [
  'You are the LLMJob assistant — a concise, friendly AI assistant embedded on llmjob.com.',
  '',
  'About LLMJob: LLMJob lets people build their own AI infrastructure from spare GPUs and devices. You pool the graphics power you are not using into one OpenAI-compatible network, so you get private, self-hosted AI without renting cloud servers — it can run entirely on your own hardware.',
  '',
  'LLMJob Earn is a desktop app (Windows/Linux) that turns idle GPU time into crypto today: it mines the Pearl (PRL) cryptocurrency via the AlphaPool miner — paste a payout address, hit Start, and earn, with no command line. It is the on-ramp that gets GPUs onto the network, ahead of LLM co-mining. Pools like this typically pay out with PPLNS (Pay Per Last N Shares), which splits each block reward across the last N shares miners submitted, rewarding sustained contribution rather than luck.',
  '',
  'This chat itself is free and served through the LLMJob network.',
  '',
  'Answer helpfully and concisely. If you are unsure of a specific LLMJob detail (exact payout amounts, schedules, or feature availability), say so rather than inventing specifics, and point people to the Discord or the site.'
].join('\n');

class ChatController {
  constructor(opts = {}) {
    // Everything OpenRouter — key, base URL, allow-list, budget, ceilings — lives
    // on the shared client so the /v1 API gateway serves the same models under
    // the same limits. The familiar `this.apiKey` / `this.models` / … accessors
    // below read straight through to it.
    this.openRouter = new OpenRouterService(opts);
    this.systemPrompt = opts.systemPrompt !== undefined
      ? opts.systemPrompt
      : (process.env.OPENROUTER_SYSTEM_PROMPT || DEFAULT_SYSTEM_PROMPT);
    this.now = opts.now || (() => Date.now());
    // The LLMJob-network model (a job served by a live node) is always available.
    this.networkModel = NETWORK_MODEL;
    this.jobPollMs = opts.jobPollMs || 250;          // how often to check the job for progress
    this.jobTimeoutMs = opts.jobTimeoutMs || 120000; // give up if no node finishes it in time
    this.sleep = opts.sleep || ((ms) => new Promise((r) => setTimeout(r, ms)));
    // Services are built per-request from req.app.locals.db so one controller can
    // be registered before the DB pool connects. Injectable for tests.
    this._services = opts.services || null;
  }

  // OpenRouter configuration, read through to the shared client. Kept as
  // properties of the controller because that is how the rest of the class (and
  // its tests) have always referred to them.
  get apiKey() { return this.openRouter.apiKey; }
  get baseUrl() { return this.openRouter.baseUrl; }
  get models() { return this.openRouter.models; }
  get freeBudget() { return this.openRouter.freeBudget; }
  get maxTokens() { return this.openRouter.maxTokens; }
  get referer() { return this.openRouter.referer; }
  get title() { return this.openRouter.title; }

  services(req) {
    if (this._services) return this._services;
    const db = req.app.locals.db;
    return { chatUsage: new ChatUsageService(db), jobs: new JobService(db), apiKeys: new ApiKeyService(db) };
  }

  // GET /api/chat/models — the models the Chat UI may offer: the OpenRouter
  // allow-list plus the LLMJob-network model at the end.
  listModels(req, res) {
    const models = this.models.map((m) => ({ id: m.id, label: m.label }));
    models.push({ id: this.networkModel.id, label: this.networkModel.label });
    res.json({ models });
  }

  // GET /api/chat/usage — running totals + how much free budget remains.
  async usage(req, res) {
    const svc = this.services(req);
    const totals = await svc.chatUsage.getTotals();
    // `totals` is every token we have bought from OpenRouter — the free web chat
    // plus the hosted models the /v1 gateway serves to public API keys — and is
    // what the free-usage cap is measured against. Node-served API traffic is
    // deliberately absent: it costs the fleet's GPU time, not our credits, and
    // folding it in would burn the free budget on somebody else's hardware.
    //
    // `network` is the headline "tokens served" figure on the network and chat
    // pages: everything that went through LLMJob, counted once. `apiTokens`
    // (billed per key) and `totals` overlap by exactly the hosted-model slice the
    // /v1 gateway records in both places, so subtract it back out.
    const apiTokens = svc.apiKeys ? await svc.apiKeys.getTotalUsage() : 0;
    const capped = this.freeBudget > 0;
    res.json({
      totals,
      network: { apiTokens, totalTokens: totals.totalTokens + apiTokens - (totals.apiTotalTokens || 0) },
      freeBudget: capped ? this.freeBudget : null,
      remaining: capped ? Math.max(0, this.freeBudget - totals.totalTokens) : null,
      exhausted: capped && totals.totalTokens >= this.freeBudget
    });
  }

  // POST /api/chat/completions
  async chatCompletions(req, res) {
    const body = req.body || {};
    const messages = body.messages;
    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json(errorBody('`messages` must be a non-empty array', 'invalid_request_error'));
    }
    const resolved = this._resolveModel(body.model);
    if (!resolved) {
      return res.status(400).json(errorBody('Unknown model.', 'invalid_request_error'));
    }

    const clean = sanitizeMessages(messages);
    if (clean.length === 0) {
      return res.status(400).json(errorBody('No usable message content.', 'invalid_request_error'));
    }
    // Prepend our own system prompt (dropping any client-supplied one) so the
    // model always has LLMJob context and callers can't override it.
    const outgoing = this.systemPrompt
      ? [{ role: 'system', content: this.systemPrompt }, ...clean.filter((m) => m.role !== 'system')]
      : clean;

    const svc = this.services(req);
    const ctx = {
      res, svc,
      messages: outgoing,
      promptText: outgoing.map((m) => m.content).join('\n'),
      modelId: resolved.id,
      requestedLabel: resolved.label,
      maxTokens: this._resolveMaxTokens(body.max_tokens),
      temperature: typeof body.temperature === 'number' ? body.temperature : null,
      start: this.now(),
      firstTokenAt: 0,
      text: '',
      usage: null,
      model: null,
      finish: 'stop',
      aborted: false
    };
    // Caller hung up — stop work and don't write to a dead socket.
    if (res.on) res.on('close', () => { ctx.aborted = true; if (ctx.controller) { try { ctx.controller.abort(); } catch (e) { /* ignore */ } } });

    // LLMJob-network model: the request becomes an inference job that a live node
    // serves on its own GPU. No OpenRouter key and no free-budget gate — it runs
    // on the fleet's hardware, not paid API credits.
    if (resolved.network) {
      try {
        if (body.stream === false) await this._jsonNetwork(ctx);
        else await this._streamNetwork(ctx);
      } catch (err) {
        if (!res.headersSent) res.status(500).json(errorBody('Gateway error: ' + err.message, 'api_error'));
        else { try { res.end(); } catch (e) { /* ignore */ } }
      }
      return;
    }

    // OpenRouter path — needs the API key and is gated by the free-usage cap.
    if (!this.apiKey) {
      return res.status(503).json(errorBody('Chat is not configured yet.', 'not_configured'));
    }
    const totals = await svc.chatUsage.getTotals();
    if (this.freeBudget > 0 && totals.totalTokens >= this.freeBudget) {
      return res.status(402).json(errorBody(
        'Free chat has reached its usage cap for now — switch to the LLMJob network model.',
        'quota_exhausted'));
    }

    ctx.controller = new (globalThis.AbortController)();
    try {
      if (body.stream === false) await this._jsonProxy(ctx);
      else await this._streamProxy(ctx);
    } catch (err) {
      if (!res.headersSent) res.status(500).json(errorBody('Gateway error: ' + err.message, 'api_error'));
      else { try { res.end(); } catch (e) { /* ignore */ } }
    }
  }

  // Create the inference job for a network-model request. Anonymous (no userId),
  // multi-turn `messages`, with the single-string prompt kept as a node fallback.
  _createNetworkJob(ctx) {
    return ctx.svc.jobs.createJob({
      prompt: lastUserText(ctx.messages),
      messages: ctx.messages,
      model: undefined,           // JobService fills the fleet default
      maxTokens: ctx.maxTokens,
      temperature: ctx.temperature != null ? ctx.temperature : undefined,
      userId: null,               // public chat has no account
      visibility: 'public',       // anyone's free chat runs on any node
    });
  }

  // Stream a network job's assembled text as our SSE protocol, long-polling the
  // job result until a node completes it (or it fails / times out).
  async _streamNetwork(ctx) {
    const { res, svc } = ctx;
    writeSsePreamble(res);
    const send = (obj) => res.write('data: ' + JSON.stringify(obj) + '\n\n');
    const done = () => { res.write('data: [DONE]\n\n'); res.end(); };

    const job = await this._createNetworkJob(ctx);
    let sent = 0; // chars already streamed to the client
    for await (const r of pollJobResult({
      jobService: svc.jobs, jobId: job.id,
      now: this.now, sleep: this.sleep, pollMs: this.jobPollMs, timeoutMs: this.jobTimeoutMs,
      isAborted: () => ctx.aborted, // socket gone — stop polling
    })) {
      if (r.status === 'timeout') {
        send({ error: 'No node is available to serve this model right now. Please try again shortly.' });
        return done();
      }
      const text = r.status === 'completed' ? (r.result || '') : (r.partial || '');
      // Shrinking partial = the job was requeued and the abandoned attempt's
      // chunks were dropped, so the retry is rebuilding from the start. Rewind
      // with it, or the old cursor would swallow the retry's output until it
      // passed the dead attempt's length and the reply would look truncated.
      if (text.length < sent) sent = 0;
      if (text.length > sent) {
        if (!ctx.firstTokenAt) ctx.firstTokenAt = this.now();
        send({ delta: text.slice(sent) });
        sent = text.length;
      }
      if (r.status === 'completed') {
        ctx.text = r.result || '';
        const meta = this._networkMeta(ctx, r);
        await this._record(ctx, meta);
        send({ done: true, meta: publicMeta(meta) });
        return done();
      }
      if (r.status === 'failed') {
        send({ error: nodeFailMessage(r) });
        return done();
      }
    }
  }

  // Non-streaming network path — poll to completion, return one JSON body.
  async _jsonNetwork(ctx) {
    const { res, svc } = ctx;
    const job = await this._createNetworkJob(ctx);
    for await (const r of pollJobResult({
      jobService: svc.jobs, jobId: job.id,
      now: this.now, sleep: this.sleep, pollMs: this.jobPollMs, timeoutMs: this.jobTimeoutMs,
      isAborted: () => ctx.aborted, // socket gone — stop polling
    })) {
      if (r.status === 'timeout') {
        return res.status(504).json(errorBody('No node is available to serve this model right now.', 'timeout_error'));
      }
      if (r.status === 'failed') {
        return res.status(502).json(errorBody(nodeFailMessage(r), 'node_error'));
      }
      if (r.status === 'completed') {
        ctx.text = r.result || '';
        const meta = this._networkMeta(ctx, r);
        await this._record(ctx, meta);
        return res.status(200).json({
          model: meta.model,
          message: { role: 'assistant', content: ctx.text },
          usage: {
            prompt_tokens: meta.promptTokens,
            completion_tokens: meta.completionTokens,
            total_tokens: meta.totalTokens,
            tokens_per_second: meta.tokensPerSecond
          },
          finish_reason: meta.finish
        });
      }
    }
  }

  // Token/perf summary for a completed network job — prefers the node's reported
  // metrics (real GPU tok/s and token count), estimating only what's missing.
  _networkMeta(ctx, r) {
    const end = this.now();
    const m = r.metrics || {};
    // The node's own count, but bounded by what it actually produced. `metrics`
    // comes straight off the node's chunk POST, and any anonymous machine can
    // enroll and serve public chat jobs — so an unbounded value here goes into
    // chat_usage_totals and can push the global free-token budget past its cap in
    // ONE request, 402-ing the public chat permanently.
    const completionTokens = boundedTokens(m.totalTokens, ctx.text);
    const promptTokens = estimateTokens(ctx.promptText);
    const genMs = Math.max(0, end - (ctx.firstTokenAt || ctx.start));
    const tokensPerSecond = (Number.isFinite(m.tokensPerSecond) && m.tokensPerSecond > 0)
      ? round1(m.tokensPerSecond)
      : (completionTokens > 0 && genMs > 0 ? round1(completionTokens / (genMs / 1000)) : 0);
    return {
      model: m.model || ctx.requestedLabel,
      promptTokens,
      completionTokens,
      totalTokens: promptTokens + completionTokens,
      tokensPerSecond,
      latencyMs: Math.max(0, end - ctx.start),
      ttftMs: ctx.firstTokenAt ? Math.max(0, ctx.firstTokenAt - ctx.start) : 0,
      finish: 'stop'
    };
  }

  // Stream the upstream generation to the client as our small SSE protocol.
  async _streamProxy(ctx) {
    const { res } = ctx;
    writeSsePreamble(res);
    const send = (obj) => res.write('data: ' + JSON.stringify(obj) + '\n\n');
    const done = () => { res.write('data: [DONE]\n\n'); res.end(); };

    let upstream;
    try {
      upstream = await this._callUpstream(ctx, true);
    } catch (err) {
      send({ error: 'Upstream request failed.' });
      return done();
    }
    if (!upstream.ok) {
      const detail = await upstreamErrorMessage(upstream);
      logUpstreamError('chat', upstream.status, detail);
      send({ error: 'The model backend returned an error: ' + detail });
      return done();
    }

    let meta = null;
    try {
      for await (const obj of parseSSE(upstream.body)) {
        if (ctx.aborted) return; // socket gone — skip meta/[DONE]
        const delta = deltaContent(obj);
        if (delta) {
          if (!ctx.firstTokenAt) ctx.firstTokenAt = this.now();
          ctx.text += delta;
          send({ delta });
        }
        if (obj.usage) ctx.usage = obj.usage;
        if (obj.model) ctx.model = obj.model;
        const fr = obj.choices && obj.choices[0] && obj.choices[0].finish_reason;
        if (fr) ctx.finish = fr;
      }
    } finally {
      // Bill whatever OpenRouter already streamed — even if the caller hung up or
      // the upstream aborted mid-stream. Recording only after a clean finish let an
      // anonymous caller read most of a stream, drop the socket before the final
      // event, and pay nothing: OpenRouter had already generated (and billed us
      // for) those tokens, but the free-usage cap that gates this endpoint never
      // saw them, so the budget never tripped. Computed once here (not again for
      // the meta below) so the perf clock is only sampled once, and `_record` is
      // best-effort and runs at most once per request.
      meta = this._computeUsage(ctx);
      await this._record(ctx, meta);
    }
    if (ctx.aborted) return;

    send({ done: true, meta: publicMeta(meta) });
    done();
  }

  // Non-streaming path — one upstream call, one JSON body back.
  async _jsonProxy(ctx) {
    const { res } = ctx;
    let upstream;
    try {
      upstream = await this._callUpstream(ctx, false);
    } catch (err) {
      return res.status(502).json(errorBody('Upstream request failed.', 'upstream_error'));
    }
    if (!upstream.ok) {
      const detail = await upstreamErrorMessage(upstream);
      logUpstreamError('chat', upstream.status, detail);
      return res.status(502).json(errorBody('The model backend returned an error: ' + detail, 'upstream_error'));
    }
    const data = await upstream.json();
    const choice = data.choices && data.choices[0];
    ctx.text = (choice && choice.message && choice.message.content) || '';
    ctx.usage = data.usage || null;
    ctx.model = data.model || null;
    ctx.finish = (choice && choice.finish_reason) || 'stop';
    ctx.firstTokenAt = ctx.start; // no TTFT signal — attribute speed to full latency

    const meta = this._computeUsage(ctx);
    await this._record(ctx, meta);
    return res.status(200).json({
      model: meta.model,
      message: { role: 'assistant', content: ctx.text },
      usage: {
        prompt_tokens: meta.promptTokens,
        completion_tokens: meta.completionTokens,
        total_tokens: meta.totalTokens,
        tokens_per_second: meta.tokensPerSecond
      },
      finish_reason: meta.finish
    });
  }

  async _callUpstream(ctx, stream) {
    return this.openRouter.send({
      model: ctx.modelId,
      messages: ctx.messages,
      maxTokens: ctx.maxTokens,
      temperature: ctx.temperature,
      stream,
      signal: ctx.controller.signal
    });
  }

  // Derive token counts + performance from upstream usage, falling back to a
  // rough estimate when the provider doesn't report counts.
  _computeUsage(ctx) {
    return usageMeta(ctx, this.now());
  }

  // Best-effort usage accounting — never blocks or breaks the response.
  async _record(ctx, meta) {
    try {
      await ctx.svc.chatUsage.recordUsage({
        model: meta.model,
        inTokens: meta.promptTokens,
        outTokens: meta.completionTokens,
        speed: meta.tokensPerSecond,
        latencyMs: meta.latencyMs,
        ttftMs: meta.ttftMs,
        finish: meta.finish
      });
    } catch (e) { /* ignore */ }
  }

  _resolveModel(requested) {
    const nm = this.networkModel;
    if (requested != null && (String(requested) === nm.id || String(requested) === nm.label)) {
      return { id: nm.id, label: nm.label, network: true };
    }
    if (!requested) return this.openRouter.defaultModel; // default is always an OpenRouter model
    return this.openRouter.resolveModel(requested);
  }

  _resolveMaxTokens(v) {
    return this.openRouter.resolveMaxTokens(v);
  }

  static parseModels(str) {
    return OpenRouterService.parseModels(str);
  }
}

// The subset of the usage/perf summary we expose to the browser.
function publicMeta(meta) {
  return {
    model: meta.model,
    promptTokens: meta.promptTokens,
    completionTokens: meta.completionTokens,
    totalTokens: meta.totalTokens,
    tokensPerSecond: meta.tokensPerSecond,
    latencyMs: meta.latencyMs,
    ttftMs: meta.ttftMs,
    finish: meta.finish // 'stop' | 'length' — the UI flags a 'length' cutoff
  };
}

// Clamp the conversation to allowed roles and a total character budget so a
// single request can't run up an unbounded prompt cost. Empty turns are dropped.
//
// The budget is spent newest-first: we walk from the last message backward and
// stop once it's exhausted, so the most recent turns — including the question the
// user just asked, which the chat page always sends LAST — are the ones kept.
// (Walking front-to-back instead spent the budget on the oldest turns and silently
// dropped the current question in a long conversation, leaving the model to
// "answer" stale context with a 200 and no error.) The kept turns are returned in
// their original chronological order.
function sanitizeMessages(messages) {
  const allowed = new Set(['system', 'user', 'assistant']);
  const out = [];
  let budget = MAX_PROMPT_CHARS;
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (!m || typeof m !== 'object') continue;
    const role = allowed.has(m.role) ? m.role : 'user';
    let content = m.content == null ? '' : String(m.content);
    if (!content) continue;
    if (content.length > budget) content = content.slice(0, budget);
    budget -= content.length;
    out.push({ role, content });
    if (budget <= 0) break;
  }
  return out.reverse();
}

// How far a node's self-reported completion count may exceed our estimate of the
// text it returned. Generous — tokenizers differ, and a thinking model's hidden
// reasoning tokens are billed but never delivered — while still bounding an
// arbitrary claim to something proportional to real work.
const TOKEN_REPORT_SLACK = 8;

// Reconcile a node-reported token count against the text actually delivered.
// Reported values are used when plausible (more accurate than a character
// estimate) and clamped when they aren't; non-finite or negative reports fall
// back to the estimate entirely.
function boundedTokens(reported, text) {
  const est = estimateTokens(text);
  const ceiling = Math.max(est * TOKEN_REPORT_SLACK, 1000);
  if (!Number.isFinite(reported) || reported < 0) return est;
  return Math.min(reported, ceiling);
}

module.exports = ChatController;
module.exports.sanitizeMessages = sanitizeMessages;
module.exports.estimateTokens = estimateTokens;
module.exports.boundedTokens = boundedTokens;
// Re-exported from the shared OpenRouter client, which owns them now: callers
// (and tests) that reach for them through the chat gateway still find them.
module.exports.parseSSE = parseSSE;
module.exports.upstreamErrorMessage = upstreamErrorMessage;
module.exports.DEFAULT_MODELS = OpenRouterService.DEFAULT_MODELS;
