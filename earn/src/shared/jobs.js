'use strict';

// Pure helpers for the node-side job worker — the piece that lets people proxy
// LLM requests through LLMJob without any inbound networking. A caller submits a
// job to the server (with an API key); the server hands it to an online node,
// which runs it against its local model and streams result chunks back. This
// module turns a server job into a local chat request; the polling/streaming IO
// lives in main/jobWorker.js.

const { LLM } = require('./config');

// Turn a server job into an OpenAI-compatible /v1/chat/completions body for the
// local llama-server. A job submitted through the OpenAI gateway carries a full
// `messages` array (multi-turn, system prompts preserved); older/simple jobs
// carry a single `prompt` that becomes one user message. Only set fields are
// included so the server's own defaults apply otherwise.
function jobToChatBody(job, model) {
  const j = job || {};
  const messages = Array.isArray(j.messages) && j.messages.length
    ? j.messages.map((m) => {
      const mm = m || {};
      return { role: mm.role || 'user', content: String(mm.content == null ? '' : mm.content) };
    })
    : [{ role: 'user', content: String(j.prompt == null ? '' : j.prompt) }];
  const body = {
    // Always the node's own loaded model, never the job's requested model: this
    // node serves one local model regardless of what was asked, and the request
    // string would otherwise reach the final metrics (metrics.model = chatBody.model)
    // and be reported back as the model that ran. A mismatched name can also make a
    // stricter llama-server reject the request outright.
    //
    // Which model that is became a question the moment nodes stopped all running
    // the same one. The caller passes what it actually loaded; the fleet default
    // is the fallback for a caller that has not been taught to, so an unwired path
    // reports the old answer rather than `undefined`.
    model: (model && model.name) || LLM.model.name,
    messages,
    stream: true,
  };
  if (j.temperature != null && Number.isFinite(Number(j.temperature))) body.temperature = Number(j.temperature);
  if (j.maxTokens != null && Number.isFinite(Number(j.maxTokens))) body.max_tokens = Number(j.maxTokens);
  return body;
}

module.exports = { jobToChatBody };
