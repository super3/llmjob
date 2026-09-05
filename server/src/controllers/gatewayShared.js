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
// The text of one message, whatever shape its content is in. Multimodal content
// is an array of parts; only the text parts have characters worth counting, and
// an image's data: URL must NOT be counted — billing a caller for a megabyte of
// base64 as if it were prompt text would be wrong, and joining it into a prompt
// string would be worse.
function contentText(content) {
  if (content == null) return '';
  if (Array.isArray(content)) {
    return content
      .filter((p) => p && p.type === 'text' && p.text != null)
      .map((p) => String(p.text))
      .join('\n');
  }
  return String(content);
}

function joinContent(messages) {
  return messages.map((m) => contentText(m && m.content)).join('\n');
}

// The single-string prompt kept on the job record for nodes that read `prompt`
// rather than the `messages` array: the last user turn, or the whole
// conversation joined when there is no user turn.
function lastUserText(messages) {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m && m.role === 'user' && m.content != null) {
      // Text only: this is the single-string `prompt` kept on the job record for
      // nodes that read it, and a base64 image belongs in `messages`, not here.
      const text = contentText(m.content);
      if (text) return text;
    }
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
  // Read only the chunk rows this loop has not seen yet. Watching a job used to
  // re-read every chunk on every 250ms tick, so the DB cost of one generation
  // grew with the square of its length — see jobService._chunksSince.
  const cursor = {};
  for (;;) {
    if (isAborted()) return;
    const r = await jobService.getJobResult(jobId, cursor);
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

// Images accepted per request, and the largest a single one may be.
//
// Deliberately budgeted apart from MAX_PROMPT_CHARS rather than counted against
// it. A base64 data: URL for even a small screenshot runs to hundreds of
// kilobytes, so charging it to a 24,000-character text budget would either
// reject every image request or, if the budget were raised to fit one, let a
// caller send 24,000 characters of prose they were never meant to have. The two
// are different resources and get different limits.
//
// Sized so the worst case is a request body a server should actually accept.
// These were 8 images of 8 MB, i.e. a 64 MB request — which nothing enforced,
// because express.json()'s own 100 KB default rejected the whole thing long
// before this code ran, so no image request had ever reached a node. Now that
// the body limit is real (MAX_BODY_BYTES, applied to the gateway routes) the two
// have to agree, and 4 x 4 MB is both a generous screenshot allowance — base64
// is 4/3 of the raw bytes, so 4 MB holds a ~3 MB image — and a bound worth
// having.
const MAX_IMAGES = 4;
const MAX_IMAGE_CHARS = 4 * 1024 * 1024;   // ~4 MB of base64 per image

// The request-body ceiling for a gateway route, covering the image budget above
// plus the text and JSON overhead around it.
//
// This is applied per route rather than globally: the anonymous web-chat
// endpoint keeps express.json()'s small default, because raising the body limit
// on an unauthenticated route is a cost with no matching benefit — the chat page
// sends no images.
const MAX_BODY_BYTES = MAX_IMAGES * MAX_IMAGE_CHARS + 4 * 1024 * 1024;

// One OpenAI content part, normalised, or null if it isn't one we pass on.
//
// OpenAI's vision requests put an ARRAY in `content`:
//   content: [{type:'text', text:'…'}, {type:'image_url', image_url:{url:'data:…'}}]
// The previous String(m.content) turned that array into the literal string
// "[object Object]" and handed it to a node — so a vision request was not merely
// unsupported, it was silently corrupted into nonsense that the model then
// answered. Anything we cannot recognise is dropped rather than coerced, for
// exactly that reason.
function normalisePart(part) {
  if (!part || typeof part !== 'object') return null;
  if (part.type === 'text') {
    const text = part.text == null ? '' : String(part.text);
    return text ? { type: 'text', text } : null;
  }
  if (part.type === 'image_url') {
    const raw = part.image_url && part.image_url.url;
    const url = raw == null ? '' : String(raw);
    if (!url || url.length > MAX_IMAGE_CHARS) return null;
    const detail = part.image_url.detail;
    const image_url = detail ? { url, detail: String(detail) } : { url };
    return { type: 'image_url', image_url };
  }
  return null;
}

// Trim a message list to `maxChars`, preserving chronological order. Entries
// that aren't objects, or whose content is empty after coercion, are dropped;
// unexpected roles are mapped onto 'user'.
//
// This lives here for the same reason the SSE preamble does: only the web-chat
// gateway ever clamped its input. The /v1 gateway passed the caller's messages
// straight through, so a single API key could hand a node an unbounded prompt.
//
// The budget is spent NEWEST-FIRST: we walk from the last message backward and
// stop once it is exhausted, so the most recent turns — including the question
// the caller just asked, which every chat client sends LAST — are the ones kept.
// The kept turns are returned in their original order.
//
// Walking oldest-first instead spent the budget on the oldest turns and silently
// dropped the current question in any conversation over the budget, leaving the
// model to "answer" stale context with a 200 and no error. That was fixed once,
// in the web-chat gateway's own private copy of this function, and the two
// copies then disagreed for as long as both existed: /v1 kept answering the
// wrong turn. This is the surviving copy, and both gateways now use it — which
// is the whole point of this module.
function clampMessages(messages, maxChars) {
  const allowed = new Set(['system', 'user', 'assistant']);
  const out = [];
  let budget = maxChars == null ? MAX_PROMPT_CHARS : maxChars;
  // Newest-first, so the image budget is spent on the newest images too: a
  // caller who attached a screenshot to THIS turn must not lose it to eight
  // stale ones earlier in the conversation.
  let images = 0;
  const list = messages || [];
  for (let i = list.length - 1; i >= 0; i--) {
    const m = list[i];
    if (!m || typeof m !== 'object') continue;
    const role = allowed.has(m.role) ? m.role : 'user';

    // Array content is OpenAI's multimodal shape. Text parts are charged to the
    // same character budget a plain string would be, so mixing the two cannot
    // buy a caller extra room; image parts are counted separately.
    if (Array.isArray(m.content)) {
      const parts = [];
      for (const raw of m.content) {
        const part = normalisePart(raw);
        if (!part) continue;
        if (part.type === 'image_url') {
          if (images >= MAX_IMAGES) continue;
          images++;
          parts.push(part);
          continue;
        }
        let text = part.text;
        if (text.length > budget) text = text.slice(0, budget);
        if (!text) continue;
        budget -= text.length;
        parts.push({ type: 'text', text });
      }
      if (parts.length) out.push({ role, content: parts });
      if (budget <= 0) break;
      continue;
    }

    let content = m.content == null ? '' : String(m.content);
    if (!content) continue;
    if (content.length > budget) content = content.slice(0, budget);
    budget -= content.length;
    out.push({ role, content });
    if (budget <= 0) break;
  }
  // Collected newest-first; hand back the conversation the right way round.
  return out.reverse();
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
  contentText,
  normalisePart,
  MAX_PROMPT_CHARS,
  MAX_IMAGES,
  MAX_IMAGE_CHARS,
  MAX_BODY_BYTES,
};
