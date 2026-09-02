'use strict';

const { LLM } = require('./config');
const { ALL_LAYERS } = require('./vram');

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

// Which llama-server archive this machine downloads. Most platforms publish a
// single build, but macOS ships separate Apple-silicon (arm64) and Intel (x64)
// archives, so an arch-qualified key ("darwin-x64") wins over the bare platform
// key when config has one. Handing an Intel Mac the arm64 build would install a
// binary the kernel refuses to exec, and the LLM is the whole point of the Mac
// build. Unknown platforms fall back to Linux, mirroring resolveServerBinary.
function resolveServerUrl(platform, arch) {
  return LLM.serverUrl[platform + '-' + arch] || LLM.serverUrl[platform] || LLM.serverUrl.linux;
}

function serverBaseUrl(opts = {}) {
  return 'http://' + (opts.host || LLM.host) + ':' + (opts.port || LLM.port);
}

// Build the llama-server argument vector. `modelPath` and `nGpuLayers` are the
// per-run bits (the VRAM budgeter picks nGpuLayers, which is all-or-nothing —
// see shared/vram.computeGpuLayers); host/port/ctx default from config.
//
// The default is ALL_LAYERS rather than a layer count of our own: llama.cpp
// clamps an over-large value to what the model really has, whereas a hardcoded
// guess that's too low silently leaves the remainder on the CPU, where the
// weights sit in host RAM and OOM a small-RAM rig.
function buildServerArgs(opts = {}) {
  const ngl = opts.nGpuLayers != null ? opts.nGpuLayers : ALL_LAYERS;
  const args = [
    '--model', opts.modelPath || '',
    '--host', opts.host || LLM.host,
    '--port', String(opts.port || LLM.port),
    '--ctx-size', String(opts.ctxSize || LLM.ctxSize),
    '--n-gpu-layers', String(ngl),
    '--parallel', String(opts.parallel || LLM.parallel),
    // Keep the model on ONE GPU. On multi-GPU rigs the Vulkan backend tries to
    // split the graph across every device and can trip
    // GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS) in ggml-backend.cpp,
    // crash-looping the server before it ever serves (seen in the field on a
    // 3060 Ti + 5060 Ti box). Our model is ~5 GB — a single card is plenty, and
    // single-GPU machines are unaffected by the flag.
    '--split-mode', 'none',
  ];
  // Pin the model to a specific GPU when the caller picked one. With
  // --split-mode none the model loads on a single device; without --main-gpu
  // that's always device 0, which on a mining rig is busy and may lack the
  // headroom the model needs. The caller (VRAM budgeter) chooses the card with
  // the most free VRAM and passes its index here so the server lands there
  // instead. A non-negative integer only; anything else falls back to device 0.
  if (Number.isInteger(opts.mainGpu) && opts.mainGpu >= 0) {
    args.push('--main-gpu', String(opts.mainGpu));
  }
  // The vision projector, for a multimodal model that ships one. The weights
  // alone load and serve happily as a text-only model, so a missing projector is
  // a silent capability loss rather than a startup failure — which is exactly why
  // the caller passes the path explicitly instead of us inferring it.
  // What the server calls itself. Without it llama-server falls back to the model
  // PATH, so /v1/models and the `model` field of every completion come back as
  // an absolute local filesystem path -- which leaks the node's directory layout
  // to anything consuming the endpoint and does not match the model name the
  // fleet reports elsewhere. Verified on a 5090: without this the id is
  // "/home/.../llm/Qwen3.8-27B-UD-Q4_K_XL.gguf".
  if (opts.alias) args.push('--alias', String(opts.alias));
  if (opts.mmprojPath) args.push('--mmproj', String(opts.mmprojPath));
  if (opts.flashAttn) args.push('--flash-attn');
  // Per-model flags, appended last so a model can override an earlier default.
  // A large model is not simply a bigger version of a small one: Qwen3.8 needs a
  // quantised KV cache to fit its context at all, its own chat template to
  // produce output matching what it was tuned for, and can self-speculate from a
  // head inside its own GGUF. None of that belongs in the shared arg list, and
  // none of it can be inferred — it comes from the model entry in config.
  if (Array.isArray(opts.extraArgs)) {
    for (const a of opts.extraArgs) if (a != null && a !== '') args.push(String(a));
  }
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

// llama-server prints TWO timing lines per request:
//
//   prompt eval time = ...  1840.00 tokens per second   <- prefill
//          eval time = ...   162.02 tokens per second   <- generation
//
// A single /tokens per second/ regex matched both, so whichever printed last
// won and the two were reported under one name. They measure different things
// and routinely differ by an order of magnitude, which made the number we
// showed -- and sent to the network board -- meaningless.
//
// Returns { kind: 'prompt'|'gen', tokensPerSec } or null.
function parseTiming(line) {
  const s = String(line == null ? '' : line);
  const m = s.match(/([\d.]+)\s*tokens per second/i);
  if (!m) return null;
  return { kind: /prompt eval/i.test(s) ? 'prompt' : 'gen', tokensPerSec: Number(m[1]) };
}

// Kept for callers that only want a number and do not care which phase it came
// from. Its old behaviour is preserved exactly: the last timing line wins.
function parseTokensPerSec(line) {
  const t = parseTiming(line);
  return t ? t.tokensPerSec : null;
}

module.exports = {
  resolveServerBinary, resolveServerUrl, serverBaseUrl, buildServerArgs, isServerReady,
  parseTokensPerSec, parseTiming,
};
