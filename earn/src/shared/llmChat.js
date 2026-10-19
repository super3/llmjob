'use strict';

// Pure helpers for the local llama-server OpenAI-compatible chat API. The HTTP
// request + streaming live in main.js (the IO shell); this builds the request
// body and parses the Server-Sent-Events stream into text deltas, so both are
// unit-testable without a running server.

const { LLM } = require('./config');

// Build the /v1/chat/completions request body. `messages` is a list of
// { role, content }; anything else is coerced to a string so a stray value can't
// break JSON.stringify. Streaming is on unless explicitly disabled.
function buildChatBody(messages, opts = {}) {
  return {
    model: opts.model || LLM.model.name,
    messages: (Array.isArray(messages) ? messages : []).map((m) => ({
      role: m && m.role === 'user' ? 'user' : m && m.role === 'system' ? 'system' : 'assistant',
      content: String(m && m.content != null ? m.content : ''),
    })),
    stream: opts.stream !== false,
    temperature: opts.temperature != null ? opts.temperature : 0.7,
  };
}

// Parse a rolling SSE buffer from llama-server's streamed response. Frames are
// `data: {json}` lines (with `data: [DONE]` to finish), separated by newlines.
// Returns { deltas, reasoning, finishReason, error, done, rest }, where `rest` is
// the trailing partial line the caller must keep and prepend to the next chunk.
//
// A request that fails after streaming has begun can't change the HTTP status,
// so llama-server sends one `data: {"error":{"message":...}}` frame and closes
// the stream without [DONE] (b9902, tools/server/server-context.cpp). `error` is
// that message, so the caller can fail the request rather than take the closed
// stream for a finished answer.
//
// Thinking models split their output across two delta fields: `reasoning_content`
// (the chain of thought) and `content` (the answer). Both burn tokens against
// max_tokens, so a cap that runs out mid-thought yields an entirely empty
// `content` — collecting reasoning separately is what lets callers see where the
// budget went instead of getting a silent blank. `finish_reason` rides along for
// the same reason: it's the difference between "the model stopped" and "we cut
// it off".
function parseChatStream(buffer) {
  const deltas = [];
  const reasoning = [];
  let finishReason = null;
  let error = null;
  let done = false;
  const parts = String(buffer == null ? '' : buffer).split('\n');
  const rest = parts.pop(); // last (possibly incomplete) line stays buffered
  for (const raw of parts) {
    const line = raw.trim();
    if (line.indexOf('data:') !== 0) continue; // skip blanks / comment lines
    const payload = line.slice(5).trim();
    if (payload === '[DONE]') { done = true; continue; }
    let obj;
    try { obj = JSON.parse(payload); } catch (e) { continue; } // ignore a torn frame
    if (obj && obj.error) {
      const msg = obj.error.message;
      error = typeof msg === 'string' && msg ? msg : JSON.stringify(obj.error);
      continue;
    }
    const choice = obj && obj.choices && obj.choices[0];
    const delta = choice && choice.delta;
    const content = delta && delta.content;
    if (typeof content === 'string' && content) deltas.push(content);
    const thought = delta && delta.reasoning_content;
    if (typeof thought === 'string' && thought) reasoning.push(thought);
    if (choice && typeof choice.finish_reason === 'string' && choice.finish_reason) {
      finishReason = choice.finish_reason;
    }
  }
  return { deltas, reasoning, finishReason, error, done, rest: rest || '' };
}

module.exports = { buildChatBody, parseChatStream };
