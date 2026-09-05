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
// One message's content, preserving OpenAI's multimodal ARRAY shape.
//
// This used to be a bare String(), which turned
//   content: [{type:'text',…},{type:'image_url',…}]
// into the literal "[object Object]" and handed that to llama-server. So a
// vision request was not merely unsupported at this end — it was corrupted into
// nonsense that the model then answered, which is the same bug the server's
// normalisePart exists to prevent and which was only ever fixed on that side.
//
// The parts are passed through as the server sent them: gatewayShared has
// already normalised and bounded them (recognised types only, image count and
// size capped), so re-validating here would only risk the two disagreeing. A
// non-array content stays a plain string, which is what every text-only job is.
function contentFor(m) {
  if (Array.isArray(m.content)) return m.content;
  return String(m.content == null ? '' : m.content);
}

function jobToChatBody(job, model) {
  const j = job || {};
  const messages = Array.isArray(j.messages) && j.messages.length
    ? j.messages.map((m) => {
      const mm = m || {};
      return { role: mm.role || 'user', content: contentFor(mm) };
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
  const want = Number(j.maxTokens);
  if (j.maxTokens != null && Number.isFinite(want)) {
    // Floor the budget at what this model needs to think AND answer. On a
    // reasoning tier a small explicit max_tokens is spent entirely on the <think>
    // block and the completion comes back empty; see minCompletionTokens.
    //
    // POSITIVE budgets only. 0 is not "too small" -- it is a caller saying
    // "generate nothing", which jobService.clampMaxTokens preserves on purpose
    // ("a meaningful OpenAI value", pinned by its own test). Flooring it would
    // make this node the one place in the stack that throws that answer away.
    //
    // This does make max_tokens advisory on such a tier: a caller asking for 60
    // can be served up to the floor and is billed for what it generates. That is
    // the trade -- the alternative bills them for an empty string.
    const floor = Number((model && model.minCompletionTokens) || 0);
    body.max_tokens = want > 0 ? Math.max(want, floor) : want;
  }
  return body;
}

module.exports = { jobToChatBody };
