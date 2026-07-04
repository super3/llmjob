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
  const args = [
    '--model', opts.modelPath || '',
    '--host', opts.host || LLM.host,
    '--port', String(opts.port || LLM.port),
    '--ctx-size', String(opts.ctxSize || LLM.ctxSize),
    '--n-gpu-layers', String(ngl),
    '--parallel', String(opts.parallel || LLM.parallel),
  ];
  if (opts.flashAttn) args.push('--flash-attn');
  return args;
}

// llama-server prints a listening line once the HTTP server is up and the model
// is loaded. Used to flip the manager to "ready".
function isServerReady(line) {
  return /server (is )?listening|HTTP server is listening|main: server is listening|all slots are idle/i.test(String(line == null ? '' : line));
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
