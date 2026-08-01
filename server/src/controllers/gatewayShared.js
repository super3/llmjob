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

module.exports = {
  estimateTokens,
  errorBody,
  joinContent,
  lastUserText,
  nodeFailMessage,
  writeSsePreamble,
  pollJobResult,
};
