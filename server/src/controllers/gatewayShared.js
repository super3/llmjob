'use strict';

// Shared machinery for the two chat gateways that turn a request into an LLMJob
// inference job and long-poll its result: the OpenAI-compatible API gateway
// (openaiController) and the free web-chat proxy's network-model path
// (chatController). These helpers used to be copy-pasted between the two files,
// which let them drift — the /v1 SSE stream was missing the X-Accel-Buffering
// header the web-chat path added to defeat Railway/nginx buffering. Keeping the
// one copy here means a fix lands in both gateways at once.

// A rough token count (~4 chars/token) for when a provider omits usage counts.
function estimateTokens(text) {
  return Math.ceil(String(text || '').length / 4);
}

// A reported integer, or `fallback` when the provider omitted it (or sent
// something non-numeric).
function int(v, fallback) {
  return Number.isFinite(v) ? Math.round(v) : fallback;
}

// One decimal place — the precision we report tok/s at.
function round1(v) {
  return Math.round(v * 10) / 10;
}

// The OpenAI-style error envelope both gateways return.
function errorBody(message, type) {
  return { error: { message, type, code: null } };
}

// Join every message's text — the whole-conversation fallback prompt.
function joinContent(messages) {
  return messages.map((m) => (m && m.content) || '').join('\n');
}

// The single-string prompt kept on the job record for nodes that read `prompt`
// rather than the `messages` array: the last user turn, or the whole
// conversation joined when there is no user turn.
function lastUserText(messages) {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m && m.role === 'user' && m.content != null) return String(m.content);
  }
  return String(joinContent(messages));
}

// A node-failure message for either response path (one place, so the empty-reason
// fallback is covered once).
function nodeFailMessage(r) {
  return 'The node failed to run the job: ' + ((r && r.error) || 'unknown error');
}

// Write the SSE response preamble both gateways share. `X-Accel-Buffering: no` is
// load-bearing: without it Railway/nginx buffer the whole "stream" and deliver it
// in one burst at the end (and a long silent buffer can trip the platform's
// no-bytes-flowing cutoff the gateway timeouts are sized around).
function writeSsePreamble(res) {
  res.status(200);
  res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  if (res.flushHeaders) res.flushHeaders();
}

// Long-poll a job to a terminal state, yielding the latest result on every tick
// so a streaming caller can emit partial output as it lands. Yields the
// job-result object each poll; yields a final `{ status: 'timeout', last }`
// sentinel (carrying the last real poll result so the caller can report which
// node had the job and how far it got) and stops when the deadline passes; stops
// immediately (yielding nothing more) when `isAborted()` turns true — the socket
// is gone, so there's no one to write to. Terminal 'completed'/'failed' results
// are yielded, then iteration ends.
async function* pollJobResult({ jobService, jobId, now, sleep, pollMs, timeoutMs, isAborted }) {
  const started = now();
  for (;;) {
    if (isAborted()) return;
    const r = await jobService.getJobResult(jobId);
    yield r;
    if (r.status === 'completed' || r.status === 'failed') return;
    if (now() - started > timeoutMs) {
      yield { status: 'timeout', last: r };
      return;
    }
    await sleep(pollMs);
  }
}

// Total prompt characters accepted per request, across all messages.
const MAX_PROMPT_CHARS = 24000;

// Trim a message list to `maxChars`, oldest-first, preserving order. Entries that
// aren't objects, or whose content is empty after coercion, are dropped;
// unexpected roles are mapped onto 'user'.
//
// This lives here for the same reason the SSE preamble does: only the web-chat
// gateway ever clamped its input. The /v1 gateway passed the caller's messages
// straight through, so a single API key could hand a node an unbounded prompt.
function clampMessages(messages, maxChars) {
  const allowed = new Set(['system', 'user', 'assistant']);
  const out = [];
  let budget = maxChars == null ? MAX_PROMPT_CHARS : maxChars;
  for (const m of messages || []) {
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

// Resolve a caller-supplied max_tokens against a ceiling. The other half of the
// same gap: /v1 forwarded `max_tokens` verbatim, so a key could ask for millions
// and hold a node's GPU for as long as it took to refuse.
//
// `0` and negatives are not "unset" here — a completion budget of zero produces
// nothing — so they fall back to the ceiling. Deliberately NOT the same rule as
// jobService's temperature handling, where 0 is meaningful and must survive.
function resolveMaxTokens(v, ceiling) {
  const n = Number(v);
  if (Number.isFinite(n) && n > 0) return Math.min(Math.floor(n), ceiling);
  return ceiling;
}

module.exports = {
  estimateTokens,
  int,
  round1,
  errorBody,
  joinContent,
  lastUserText,
  nodeFailMessage,
  writeSsePreamble,
  pollJobResult,
  clampMessages,
  resolveMaxTokens,
  MAX_PROMPT_CHARS,
};
