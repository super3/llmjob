'use strict';

const { estimateTokens, int, round1 } = require('../controllers/gatewayShared');

// The OpenRouter client both gateways share.
//
// Two different front doors reach the same hosted models:
//
//   • the free public web chat (chatController) — anonymous, no key, gated by a
//     global free-token budget, and
//   • the OpenAI-compatible API gateway (openaiController) — where a `public`
//     API key may ask for a hosted model by id, so we can put real traffic on
//     them before the node network serves them itself.
//
// Everything the two have in common — which models are reachable, the allow-list
// lookup, the completion ceiling, the upstream HTTP call, reading its SSE body,
// and turning its `usage` block into our metrics — lives here so they can't
// drift the way the SSE preamble once did (see controllers/gatewayShared.js).
//
// The spend cap is carried here (`freeBudget`) but not enforced here: each
// gateway decides when to weigh it against the running chat_usage_totals,
// because only the caller knows whether this request is billable to that pot.

// Sensible defaults; every one is overridable via env or constructor opts so the
// founder can retune the free tier without a code change.
const DEFAULT_MODELS = [
  { id: 'qwen/qwen3.6-27b', label: 'Qwen3.6 27B' },
  { id: 'qwen/qwen3.6-35b-a3b', label: 'Qwen3.6 35B A3B' },
  { id: 'qwen/qwen3.8-27b', label: 'Qwen3.8 27B' }
];
const DEFAULT_FREE_BUDGET = 1000000; // total tokens of free usage before the cap
const DEFAULT_MAX_TOKENS = 2048;     // per-request completion ceiling
const OPENROUTER_BASE = 'https://openrouter.ai/api/v1';

function numberEnv(v, fallback) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

class OpenRouterService {
  constructor(opts = {}) {
    this.apiKey = opts.apiKey !== undefined ? opts.apiKey : process.env.OPENROUTER_API_KEY;
    this.baseUrl = opts.baseUrl || process.env.OPENROUTER_BASE_URL || OPENROUTER_BASE;
    this.models = opts.models || OpenRouterService.parseModels(process.env.OPENROUTER_MODELS) || DEFAULT_MODELS;
    this.freeBudget = opts.freeBudget !== undefined
      ? opts.freeBudget
      : numberEnv(process.env.OPENROUTER_FREE_TOKEN_BUDGET, DEFAULT_FREE_BUDGET);
    this.maxTokens = opts.maxTokens !== undefined
      ? opts.maxTokens
      : numberEnv(process.env.OPENROUTER_MAX_TOKENS, DEFAULT_MAX_TOKENS);
    this.referer = opts.referer || process.env.OPENROUTER_REFERER || 'https://llmjob.com';
    this.title = opts.title || 'LLMJob';
    this.fetchFn = opts.fetchFn || globalThis.fetch;
  }

  // Whether an upstream key is present. Without one the hosted models exist in
  // the catalogue but cannot be served, so callers report 503 rather than
  // failing at the fetch.
  get configured() {
    return !!this.apiKey;
  }

  // The model a caller gets when they don't name one (web chat only — the API
  // gateway sends an unnamed model to the node network instead).
  get defaultModel() {
    return this.models[0];
  }

  // Look a requested model up in the allow-list, by id or by friendly label.
  // Returns null for anything else: the allow-list is what bounds our spend, so
  // an unknown name must never reach OpenRouter.
  resolveModel(requested) {
    if (requested == null) return null;
    const r = String(requested);
    return this.models.find((m) => m.id === r || m.label === r) || null;
  }

  // Clamp a caller-supplied max_tokens to the configured ceiling. `0` and
  // negatives are not "unset" — a completion budget of zero produces nothing —
  // so they fall back to the ceiling.
  resolveMaxTokens(v) {
    const n = Number(v);
    if (Number.isFinite(n) && n > 0) return Math.min(Math.floor(n), this.maxTokens);
    return this.maxTokens;
  }

  // One upstream chat-completions call. Returns the raw fetch response so the
  // caller can stream the body or read it as JSON.
  send({ model, messages, maxTokens, temperature, stream, signal }) {
    const payload = {
      model,
      messages,
      stream: !!stream,
      max_tokens: maxTokens
    };
    if (temperature != null) payload.temperature = temperature;
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
      signal
    });
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

// The assistant text delta on an OpenAI-style streaming chunk, if any.
function deltaContent(obj) {
  return obj.choices && obj.choices[0] && obj.choices[0].delta && obj.choices[0].delta.content;
}

// The chain-of-thought delta a thinking model streams alongside its answer.
// OpenRouter normalises this to `reasoning`; some providers send the OpenAI-ish
// `reasoning_content`. Either way it is the only thing that explains an empty
// `content` when the completion budget ran out mid-thought.
function deltaReasoning(obj) {
  const d = obj.choices && obj.choices[0] && obj.choices[0].delta;
  return d ? (d.reasoning || d.reasoning_content) : undefined;
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
// deploy console (the caller only sees the sanitized message). `scope` names the
// gateway that hit it ('chat' | 'api').
function logUpstreamError(scope, status, detail) {
  console.error('[' + scope + '] OpenRouter error ' + status + ': ' + detail);
}

// Derive token counts + performance from an upstream `usage` block, falling back
// to a rough estimate when the provider doesn't report counts. `ctx` carries the
// request's clock marks (start, firstTokenAt), the text we assembled, and the
// model/finish reason observed upstream; `end` is the current time.
function usageMeta(ctx, end) {
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

module.exports = OpenRouterService;
module.exports.parseSSE = parseSSE;
module.exports.deltaContent = deltaContent;
module.exports.deltaReasoning = deltaReasoning;
module.exports.upstreamErrorMessage = upstreamErrorMessage;
module.exports.logUpstreamError = logUpstreamError;
module.exports.usageMeta = usageMeta;
module.exports.DEFAULT_MODELS = DEFAULT_MODELS;
