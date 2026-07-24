'use strict';

const { LLM } = require('./config');

// Pure helpers for the local llama.cpp `llama-server` (OpenAI-compatible). The
// process spawn/supervision lives in main/llmManager.js; this module just builds
// the command line, resolves paths/URLs, and parses the server's output — all
// unit-testable without a GPU or a running server.

// Resolve the llama-server binary: a configured path wins, else the per-platform
// name (callers pass the install dir; here we only decide the executable name).
function resolveServerBinary(binaryPath, platform) {
  if (binaryPath) return binaryPath;
  return LLM.serverBin[platform] || LLM.serverBin.linux;
}

function serverBaseUrl(opts = {}) {
  return 'http://' + (opts.host || LLM.host) + ':' + (opts.port || LLM.port);
}

// Build the llama-server argument vector. `modelPath` and `nGpuLayers` are the
// per-run bits (the VRAM budgeter picks nGpuLayers); host/port/ctx default from
// config. `--n-gpu-layers 0` runs on CPU.
function buildServerArgs(opts = {}) {
  const ngl = opts.nGpuLayers != null ? opts.nGpuLayers : LLM.model.layers;
  // Default to keeping the model on ONE GPU. On multi-GPU rigs the Vulkan
  // backend tries to split the graph across every device and can trip
  // GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS) in ggml-backend.cpp,
  // crash-looping the server before it ever serves (seen in the field on a
  // 3060 Ti + 5060 Ti box). A model that fits one card wants 'none'. A model too
  // big for any single card is instead SHARDED across cards — the caller passes
  // splitMode 'layer' (pipeline; forgiving of low interconnect bandwidth) or
  // 'row' (tensor-parallel) plus a `tensorSplit` proportion per physical GPU
  // (0 excludes a card), so only the chosen cards host the model.
  const splitMode = opts.splitMode || 'none';
  const args = [
    '--model', opts.modelPath || '',
    '--host', opts.host || LLM.host,
    '--port', String(opts.port || LLM.port),
    '--ctx-size', String(opts.ctxSize || LLM.ctxSize),
    '--n-gpu-layers', String(ngl),
    '--parallel', String(opts.parallel || LLM.parallel),
    '--split-mode', splitMode,
  ];
  // When sharding, distribute the model across GPUs by proportion. A 0 entry
  // excludes that physical card (so the rig's other cards stay free to mine).
  if (splitMode !== 'none' && Array.isArray(opts.tensorSplit) && opts.tensorSplit.length) {
    args.push('--tensor-split', opts.tensorSplit.join(','));
  }
  // Pin the primary GPU when the caller picked one. With --split-mode none the
  // model loads on a single device; without --main-gpu that's always device 0,
  // which on a mining rig is busy and may lack the headroom. When sharding it's
  // the card that holds the non-repeating tensors. A non-negative integer only;
  // anything else falls back to device 0.
  if (Number.isInteger(opts.mainGpu) && opts.mainGpu >= 0) {
    args.push('--main-gpu', String(opts.mainGpu));
  }
  if (opts.flashAttn) args.push('--flash-attn');
  return args;
}

// Flip the manager to "ready" only on lines llama-server prints AFTER the model
// has loaded ("model loaded", "server is listening on … - starting the main
// loop", "all slots are idle"). The earlier "main: HTTP server is listening"
// line appears BEFORE the multi-GB model loads — while /v1/chat/completions
// still returns 503 — so it must NOT count as ready.
function isServerReady(line) {
  return /model loaded|starting the main loop|all slots are idle/i.test(String(line == null ? '' : line));
}

// Best-effort tokens/sec from llama-server's timing lines
// (e.g. "eval time = 1234.5 ms / 200 tokens ... 162.02 tokens per second").
function parseTokensPerSec(line) {
  const m = String(line == null ? '' : line).match(/([\d.]+)\s*tokens per second/i);
  return m ? Number(m[1]) : null;
}

module.exports = {
  resolveServerBinary, serverBaseUrl, buildServerArgs, isServerReady, parseTokensPerSec,
};
