'use strict';

// Pure helpers for the node-side job worker — the piece that lets people proxy
// LLM requests through LLMJob without any inbound networking. A caller submits a
// job to the server (with an API key); the server hands it to an online node,
// which runs it against its local model and streams result chunks back. This
// module turns a server job into a local chat request; the polling/streaming IO
// lives in main/jobWorker.js.

const { LLM } = require('./config');

// Turn a server job ({ prompt, model, maxTokens, temperature }) into an
// OpenAI-compatible /v1/chat/completions body for the local llama-server. The
// single prompt becomes one user message; only set fields are included so the
// server's own defaults apply otherwise.
function jobToChatBody(job) {
  const j = job || {};
  const body = {
    model: j.model || LLM.model.name,
    messages: [{ role: 'user', content: String(j.prompt == null ? '' : j.prompt) }],
    stream: true,
  };
  if (j.temperature != null && Number.isFinite(Number(j.temperature))) body.temperature = Number(j.temperature);
  if (j.maxTokens != null && Number.isFinite(Number(j.maxTokens))) body.max_tokens = Number(j.maxTokens);
  return body;
}

module.exports = { jobToChatBody };
