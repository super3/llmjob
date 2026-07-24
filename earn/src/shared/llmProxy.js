'use strict';

// Pure helpers for reaching a model "through LLMJob as a proxy": when a box can't
// run a model locally (its GPU is too small, or the LLM is off), the client
// forwards the chat to the LLMJob web gateway (server/src/controllers/
// chatController.js), which already serves an allow-listed set of models. The
// HTTP + streaming live in main/io.js (streamProxyChat); this builds the request
// body, parses the gateway's Server-Sent-Events, and resolves which models the
// gateway offers — so all of it is unit-testable without a running server.
//
// This is the complement to the on-device serving path (VRAM-tiered local model
// selection): local when the hardware can, proxy when it can't.
//
// The gateway speaks a small SSE shape of its own (not raw OpenAI):
//   data: {"delta":"..."}              incremental text
//   data: {"done":true,"meta":{...}}   final token/perf summary
//   data: {"error":"..."}              gateway/upstream failure
//   data: [DONE]                       terminator

// The models the LLMJob gateway serves, as { id, label }. Mirrors the server's
// chatController DEFAULT_MODELS so the client offers the same set offline; the
// live list (which the server can override via OPENROUTER_MODELS) is fetched from
// GET /api/chat/models and takes precedence when reachable (see parseModelsResponse).
const PROXY_MODELS = [
  { id: 'qwen/qwen3.6-27b', label: 'Qwen3.6 27B' },
  { id: 'qwen/qwen3.6-35b-a3b', label: 'Qwen3.6 35B A3B' },
];

// The gateway chat + model-list endpoints for a server base URL (slashes trimmed).
function proxyChatUrl(serverUrl) {
  return String(serverUrl == null ? '' : serverUrl).replace(/\/+$/, '') + '/api/chat/completions';
}

function proxyModelsUrl(serverUrl) {
  return String(serverUrl == null ? '' : serverUrl).replace(/\/+$/, '') + '/api/chat/models';
}

// Normalize a GET /api/chat/models body ({ models: [{ id, label }] }) into a
// clean [{ id, label }] list. Returns null on anything malformed so the caller
// falls back to the built-in PROXY_MODELS instead of showing an empty picker.
function parseModelsResponse(json) {
  const arr = json && json.models;
  if (!Array.isArray(arr)) return null;
  const models = arr
    .filter((m) => m && m.id)
    .map((m) => ({ id: String(m.id), label: String(m.label || m.id) }));
  return models.length ? models : null;
}

// Resolve a proxy model by id or label (case-insensitive), from `models`
// (defaults to the built-in list). Returns the entry or null.
function findProxyModel(idOrLabel, models = PROXY_MODELS) {
  if (!idOrLabel) return null;
  const key = String(idOrLabel).toLowerCase();
  return (models || []).find((m) => m
    && (String(m.id).toLowerCase() === key || String(m.label).toLowerCase() === key)) || null;
}

// Build the gateway /api/chat/completions request body. `opts.model` is the
// gateway model id (e.g. "qwen/qwen3.6-27b"). Roles are coerced to
// system/user/assistant and content to a string so a stray value can't break
// JSON.stringify. Streaming is on unless explicitly disabled.
function buildProxyChatBody(messages, opts = {}) {
  const body = {
    model: opts.model,
    messages: (Array.isArray(messages) ? messages : []).map((m) => ({
      role: m && m.role === 'user' ? 'user' : m && m.role === 'system' ? 'system' : 'assistant',
      content: String(m && m.content != null ? m.content : ''),
    })),
    stream: opts.stream !== false,
  };
  if (opts.temperature != null && Number.isFinite(Number(opts.temperature))) {
    body.temperature = Number(opts.temperature);
  }
  return body;
}

// Parse a rolling SSE buffer from the gateway's streamed response. Returns
// { deltas: string[], done: bool, error: string|null, rest }, where `rest` is
// the trailing partial line the caller keeps and prepends to the next chunk.
function parseProxyStream(buffer) {
  const deltas = [];
  let done = false;
  let error = null;
  const parts = String(buffer == null ? '' : buffer).split('\n');
  const rest = parts.pop(); // last (possibly incomplete) line stays buffered
  for (const raw of parts) {
    const line = raw.trim();
    if (line.indexOf('data:') !== 0) continue; // skip blanks / comment lines
    const payload = line.slice(5).trim();
    if (!payload) continue;
    if (payload === '[DONE]') { done = true; continue; }
    let obj;
    try { obj = JSON.parse(payload); } catch (e) { continue; } // ignore a torn frame
    if (obj && typeof obj.delta === 'string' && obj.delta) deltas.push(obj.delta);
    if (obj && obj.error) error = String(obj.error);
    if (obj && obj.done) done = true;
  }
  return { deltas, done, error, rest: rest || '' };
}

module.exports = {
  PROXY_MODELS,
  proxyChatUrl, proxyModelsUrl, parseModelsResponse, findProxyModel,
  buildProxyChatBody, parseProxyStream,
};
