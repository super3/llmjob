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
// Returns { deltas: string[], done: bool, rest }, where `rest` is the trailing
// partial line the caller must keep and prepend to the next chunk.
function parseChatStream(buffer) {
  const deltas = [];
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
    const choice = obj && obj.choices && obj.choices[0];
    const content = choice && choice.delta && choice.delta.content;
    if (typeof content === 'string' && content) deltas.push(content);
  }
  return { deltas, done, rest: rest || '' };
}

module.exports = { buildChatBody, parseChatStream };
