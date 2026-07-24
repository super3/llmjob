const ChatUsageService = require('../services/chatUsageService');
const JobService = require('../services/jobService');

// Free public web-chat gateway, backed by OpenRouter.
//
// The website's Chat page (chat.html) talks to this proxy instead of OpenRouter
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
// The client protocol is a small SSE stream of our own shape (not raw OpenAI):
//   data: {"delta":"..."}                 incremental text
//   data: {"done":true,"meta":{...}}      final token/perf summary
//   data: {"error":"..."}                 upstream/gateway failure
//   data: [DONE]                          terminator
// Non-streaming callers (stream:false) get a single JSON body instead.

// Sensible defaults; every one is overridable via env or constructor opts so the
// founder can retune the free tier without a code change.
const DEFAULT_MODELS = [
  { id: 'qwen/qwen3.6-27b', label: 'Qwen3.6 27B' },
  { id: 'qwen/qwen3.6-35b-a3b', label: 'Qwen3.6 35B A3B' }
];
// The one model served by the LLMJob node fleet itself (not OpenRouter): a public
// chat request for it becomes an inference job that a live node runs on its own
// GPU. Always offered alongside the OpenRouter models, but never the default —
// callers opt in by selecting it. Its served model id is the fleet default in
// jobService (JobService fills it in when the job omits `model`).
const NETWORK_MODEL = { id: 'llmjob-gemma-4-e4b', label: 'Gemma 4 E4B (LLMJob network)' };
const DEFAULT_FREE_BUDGET = 1000000; // total tokens of free usage before the cap
const DEFAULT_MAX_TOKENS = 2048;     // per-request completion ceiling
const MAX_PROMPT_CHARS = 24000;      // total prompt characters kept per request
const OPENROUTER_BASE = 'https://openrouter.ai/api/v1';

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
    this.apiKey = opts.apiKey !== undefined ? opts.apiKey : process.env.OPENROUTER_API_KEY;
    this.baseUrl = opts.baseUrl || process.env.OPENROUTER_BASE_URL || OPENROUTER_BASE;
    this.models = opts.models || ChatController.parseModels(process.env.OPENROUTER_MODELS) || DEFAULT_MODELS;
    this.freeBudget = opts.freeBudget !== undefined
      ? opts.freeBudget
      : numberEnv(process.env.OPENROUTER_FREE_TOKEN_BUDGET, DEFAULT_FREE_BUDGET);
    this.maxTokens = opts.maxTokens !== undefined
      ? opts.maxTokens
      : numberEnv(process.env.OPENROUTER_MAX_TOKENS, DEFAULT_MAX_TOKENS);
    this.referer = opts.referer || process.env.OPENROUTER_REFERER || 'https://llmjob.com';
    this.title = opts.title || 'LLMJob';
    this.systemPrompt = opts.systemPrompt !== undefined
      ? opts.systemPrompt
      : (process.env.OPENROUTER_SYSTEM_PROMPT || DEFAULT_SYSTEM_PROMPT);
    this.fetchFn = opts.fetchFn || globalThis.fetch;
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

  services(req) {
    if (this._services) return this._services;
    const db = req.app.locals.db;
    return { chatUsage: new ChatUsageService(db), jobs: new JobService(db) };
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
    const totals = await this.services(req).chatUsage.getTotals();
    const capped = this.freeBudget > 0;
    res.json({
      totals,
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
      return res.status(400).json(errBody('`messages` must be a non-empty array', 'invalid_request_error'));
    }
    const resolved = this._resolveModel(body.model);
    if (!resolved) {
      return res.status(400).json(errBody('Unknown model.', 'invalid_request_error'));
    }

    const clean = sanitizeMessages(messages);
    if (clean.length === 0) {
      return res.status(400).json(errBody('No usable message content.', 'invalid_request_error'));
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
        if (!res.headersSent) res.status(500).json(errBody('Gateway error: ' + err.message, 'api_error'));
        else { try { res.end(); } catch (e) { /* ignore */ } }
      }
      return;
    }

    // OpenRouter path — needs the API key and is gated by the free-usage cap.
    if (!this.apiKey) {
      return res.status(503).json(errBody('Chat is not configured yet.', 'not_configured'));
    }
    const totals = await svc.chatUsage.getTotals();
    if (this.freeBudget > 0 && totals.totalTokens >= this.freeBudget) {
      return res.status(402).json(errBody(
        'Free chat has reached its usage cap for now — switch to the LLMJob network model.',
        'quota_exhausted'));
    }

    ctx.controller = new (globalThis.AbortController)();
    try {
      if (body.stream === false) await this._jsonProxy(ctx);
      else await this._streamProxy(ctx);
    } catch (err) {
      if (!res.headersSent) res.status(500).json(errBody('Gateway error: ' + err.message, 'api_error'));
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
    res.status(200);
    res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache, no-transform');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');
    if (res.flushHeaders) res.flushHeaders();
    const send = (obj) => res.write('data: ' + JSON.stringify(obj) + '\n\n');
    const done = () => { res.write('data: [DONE]\n\n'); res.end(); };

    const job = await this._createNetworkJob(ctx);
    const started = this.now();
    let sent = 0; // chars already streamed to the client
    for (;;) {
      if (ctx.aborted) return; // socket gone — stop polling
      const r = await svc.jobs.getJobResult(job.id);
      const text = r.status === 'completed' ? (r.result || '') : (r.partial || '');
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
      if (this.now() - started > this.jobTimeoutMs) {
        send({ error: 'No node is available to serve this model right now. Please try again shortly.' });
        return done();
      }
      await this.sleep(this.jobPollMs);
    }
  }

  // Non-streaming network path — poll to completion, return one JSON body.
  async _jsonNetwork(ctx) {
    const { res, svc } = ctx;
    const job = await this._createNetworkJob(ctx);
    const started = this.now();
    for (;;) {
      if (ctx.aborted) return; // socket gone — stop polling
      const r = await svc.jobs.getJobResult(job.id);
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
      if (r.status === 'failed') {
        return res.status(502).json(errBody(nodeFailMessage(r), 'node_error'));
      }
      if (this.now() - started > this.jobTimeoutMs) {
        return res.status(504).json(errBody('No node is available to serve this model right now.', 'timeout_error'));
      }
      await this.sleep(this.jobPollMs);
    }
  }

  // Token/perf summary for a completed network job — prefers the node's reported
  // metrics (real GPU tok/s and token count), estimating only what's missing.
  _networkMeta(ctx, r) {
    const end = this.now();
    const m = r.metrics || {};
    const completionTokens = Number.isFinite(m.totalTokens) ? m.totalTokens : estimateTokens(ctx.text);
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
    res.status(200);
    res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache, no-transform');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no'); // defeat proxy/edge SSE buffering (Railway/nginx)
    if (res.flushHeaders) res.flushHeaders();
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
      logUpstreamError(upstream.status, detail);
      send({ error: 'The model backend returned an error: ' + detail });
      return done();
    }

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
    if (ctx.aborted) return;

    const meta = this._computeUsage(ctx);
    await this._record(ctx, meta);
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
      return res.status(502).json(errBody('Upstream request failed.', 'upstream_error'));
    }
    if (!upstream.ok) {
      const detail = await upstreamErrorMessage(upstream);
      logUpstreamError(upstream.status, detail);
      return res.status(502).json(errBody('The model backend returned an error: ' + detail, 'upstream_error'));
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
    const payload = {
      model: ctx.modelId,
      messages: ctx.messages,
      stream,
      max_tokens: ctx.maxTokens
    };
    if (ctx.temperature != null) payload.temperature = ctx.temperature;
    if (stream) payload.stream_options = { include_usage: true };
    else payload.usage = { include: true };
    return this.fetchFn(this.baseUrl + '/chat/completions', {
      method: 'POST',
      headers: {
        Authorization: 'Bearer ' + this.apiKey,
        'Content-Type': 'application/json',
        'HTTP-Referer': this.referer,
        'X-Title': this.title
      },
      body: JSON.stringify(payload),
      signal: ctx.controller.signal
    });
  }

  // Derive token counts + performance from upstream usage, falling back to a
  // rough estimate when the provider doesn't report counts.
  _computeUsage(ctx) {
    const end = this.now();
    const u = ctx.usage || {};
    const promptTokens = int(u.prompt_tokens, estimateTokens(ctx.promptText));
    const completionTokens = int(u.completion_tokens, estimateTokens(ctx.text));
    const totalTokens = int(u.total_tokens, promptTokens + completionTokens);
    const genMs = Math.max(0, end - (ctx.firstTokenAt || ctx.start));
    const tokensPerSecond = (completionTokens > 0 && genMs > 0)
      ? round1(completionTokens / (genMs / 1000))
      : 0;
    const ttftMs = ctx.firstTokenAt ? Math.max(0, ctx.firstTokenAt - ctx.start) : 0;
    return {
      model: ctx.model || ctx.requestedLabel,
      promptTokens,
      completionTokens,
      totalTokens,
      tokensPerSecond,
      latencyMs: Math.max(0, end - ctx.start),
      ttftMs,
      finish: ctx.finish // always set: 'stop' by default, or the upstream reason
    };
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
    if (!requested) return this.models[0]; // default is always an OpenRouter model
    const r = String(requested);
    return this.models.find((m) => m.id === r || m.label === r) || null;
  }

  _resolveMaxTokens(v) {
    const n = Number(v);
    if (Number.isFinite(n) && n > 0) return Math.min(Math.floor(n), this.maxTokens);
    return this.maxTokens;
  }

  // Parse the OPENROUTER_MODELS env (a JSON array of {id,label}); returns null on
  // anything malformed so the caller falls back to the defaults.
  static parseModels(str) {
    if (!str) return null;
    try {
      const arr = JSON.parse(str);
      if (!Array.isArray(arr)) return null;
      const models = arr
        .filter((m) => m && m.id)
        .map((m) => ({ id: String(m.id), label: String(m.label || m.id) }));
      return models.length ? models : null;
    } catch (e) {
      return null;
    }
  }
}

// The assistant text delta on an OpenAI-style streaming chunk, if any.
function deltaContent(obj) {
  return obj.choices && obj.choices[0] && obj.choices[0].delta && obj.choices[0].delta.content;
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
function sanitizeMessages(messages) {
  const allowed = new Set(['system', 'user', 'assistant']);
  const out = [];
  let budget = MAX_PROMPT_CHARS;
  for (const m of messages) {
    if (!m || typeof m !== 'object') continue;
    const role = allowed.has(m.role) ? m.role : 'user';
    let content = m.content == null ? '' : String(m.content);
    if (!content) continue;
    if (content.length > budget) content = content.slice(0, budget);
    budget -= content.length;
    out.push({ role, content });
    if (budget <= 0) break;
  }
  return out;
}

// Iterate an OpenAI-style SSE body, yielding each parsed JSON event. Tolerates
// chunk boundaries splitting a line and skips comments / unparseable payloads.
async function* parseSSE(body) {
  const decoder = new (globalThis.TextDecoder)();
  let buf = '';
  for await (const chunk of body) {
    buf += decoder.decode(chunk, { stream: true });
    let idx;
    while ((idx = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 1);
      if (!line || line.startsWith(':')) continue;
      if (!line.startsWith('data:')) continue;
      const payload = line.slice(5).trim();
      if (payload === '[DONE]') return;
      try { yield JSON.parse(payload); } catch (e) { /* skip partial/garbage */ }
    }
  }
}

// Pull a human-readable reason out of a failed upstream response. OpenRouter
// returns `{ "error": { "message": "…" } }`; fall back to the raw body (trimmed)
// or the bare status. Never contains our API key, so it's safe to relay.
async function upstreamErrorMessage(resp) {
  let text = '';
  try { text = await resp.text(); } catch (e) { return 'HTTP ' + resp.status; }
  if (text) {
    try {
      const j = JSON.parse(text);
      if (j && j.error && j.error.message) return String(j.error.message);
    } catch (e) { /* not JSON — fall through to the raw body */ }
    return text.slice(0, 300);
  }
  return 'HTTP ' + resp.status;
}

// Surface upstream failures in the server logs so they're diagnosable from the
// deploy console (the browser only sees the sanitized message).
function logUpstreamError(status, detail) {
  console.error('[chat] OpenRouter error ' + status + ': ' + detail);
}

// The single-string prompt kept on the job record as a fallback for nodes that
// read `prompt` rather than the `messages` array: the last user turn, or the whole
// (already-sanitized) conversation joined when there's no user turn.
function lastUserText(messages) {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === 'user') return messages[i].content;
  }
  return messages.map((m) => m.content).join('\n');
}

// A node-failure message for either response path (one place, so the empty-reason
// fallback is covered once).
function nodeFailMessage(r) {
  return 'The node failed to run the job: ' + (r.error || 'unknown error');
}

// A rough token count (~4 chars/token) for when the provider omits usage.
function estimateTokens(text) {
  return Math.ceil(String(text || '').length / 4);
}

function int(v, fallback) {
  return Number.isFinite(v) ? Math.round(v) : fallback;
}

function round1(v) {
  return Math.round(v * 10) / 10;
}

function numberEnv(v, fallback) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function errBody(message, type) {
  return { error: { message, type, code: null } };
}

module.exports = ChatController;
module.exports.parseSSE = parseSSE;
module.exports.sanitizeMessages = sanitizeMessages;
module.exports.estimateTokens = estimateTokens;
module.exports.upstreamErrorMessage = upstreamErrorMessage;
module.exports.DEFAULT_MODELS = DEFAULT_MODELS;
