const ChatUsageService = require('../services/chatUsageService');

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
    // Services are built per-request from req.app.locals.db so one controller can
    // be registered before the DB pool connects. Injectable for tests.
    this._services = opts.services || null;
  }

  services(req) {
    if (this._services) return this._services;
    return { chatUsage: new ChatUsageService(req.app.locals.db) };
  }

  // GET /api/chat/models — the allow-listed models the Chat UI may offer.
  listModels(req, res) {
    res.json({ models: this.models.map((m) => ({ id: m.id, label: m.label })) });
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
    if (!this.apiKey) {
      return res.status(503).json(errBody('Chat is not configured yet.', 'not_configured'));
    }
    const resolved = this._resolveModel(body.model);
    if (!resolved) {
      return res.status(400).json(errBody('Unknown model.', 'invalid_request_error'));
    }

    const svc = this.services(req);
    const totals = await svc.chatUsage.getTotals();
    if (this.freeBudget > 0 && totals.totalTokens >= this.freeBudget) {
      return res.status(402).json(errBody(
        'Free chat has reached its usage cap for now — LLMJob network inference is coming soon.',
        'quota_exhausted'));
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

    const controller = new (globalThis.AbortController)();
    const ctx = {
      res, svc, controller,
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
    // Caller hung up — stop reading upstream and don't write to a dead socket.
    if (res.on) res.on('close', () => { ctx.aborted = true; try { controller.abort(); } catch (e) { /* ignore */ } });

    try {
      if (body.stream === false) await this._jsonProxy(ctx);
      else await this._streamProxy(ctx);
    } catch (err) {
      if (!res.headersSent) res.status(500).json(errBody('Gateway error: ' + err.message, 'api_error'));
      else { try { res.end(); } catch (e) { /* ignore */ } }
    }
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
    if (!requested) return this.models[0];
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
