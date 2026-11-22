'use strict';

// The parts of starting the local LLM that both shells do the same way: the
// engine that fetches llama-server, the model and its projector; the fleet;
// and the options every instance starts with. main.js and earn-cli.js each
// had a copy. The CLI's took a cached llama-server without going through
// ensureServer, so a binary left without its execute bit failed with EACCES on
// every start, while the GUI's repaired it.
//
// Planning stays in each shell on purpose. The GUI leaves out models a card
// kept losing its GPU under and shows held and status lines; the CLI honours
// --llm-model and --llm-max-instances. No Electron here: esbuild bundles the CLI.

const fs = require('fs');
const path = require('path');
const http = require('http');
const { spawn } = require('child_process');
const { LLM } = require('../shared/config');
const { ctxLadder } = require('../shared/models');
const { resolveServerUrl, cudaServerUrl } = require('../shared/llama');
const { LlmEngineManager } = require('./llmEngineManager');
const { LlmManager } = require('./llmManager');
const { LlmFleet } = require('./llmFleet');
const { downloadFile } = require('./io');
const { findFreePort } = require('./probe');
const { resetMemoryClock } = require('./gpuClocks');

// What fetches llama-server, the model and its projector into `dir`, each only
// when it is missing. ensureServer also puts the execute bit back on a cached
// binary. `extract` unpacks the llama.cpp archive; each shell brings its own,
// since the GUI unzips with PowerShell on Windows.
function llmEngine(dir, extract) {
  return new LlmEngineManager({
    dir, platform: process.platform, serverUrl: resolveServerUrl(process.platform, process.arch),
    fs, download: downloadFile, extract, chmod: fs.chmodSync,
  });
}

// The same for the CUDA build (LLM.cudaServer), in its own folder under `dir`,
// from the release of `version` (llama.cudaServerUrl). Linux only.
function llmCudaEngine(dir, extract, version) {
  return new LlmEngineManager({
    dir: path.join(dir, LLM.cudaServer.dir), platform: process.platform, serverUrl: cudaServerUrl(version),
    fs, download: downloadFile, extract, chmod: fs.chmodSync,
  });
}

// A fleet of llama-servers, one per planned card: each walks to the first free
// port from LLM.port, and retries an early exit a few times for a port clash
// that clears. Each card's memory clock goes back to the driver before its
// model loads (LlmFleet._spawn). Both shells plan the fleet from nvidia-smi's
// cards, so a card's index is nvidia-smi's. `opts` adds makeWorker and anything
// else the shell needs.
function makeLlmFleet(opts) {
  return new LlmFleet(Object.assign({
    host: LLM.host,
    basePort: LLM.port,
    makeManager: () => new LlmManager({ spawn, startAttempts: LLM.startAttempts, retryDelayMs: LLM.startRetryMs }),
    findFreePort,
    releaseMemClock: (index) => resetMemoryClock(index),
  }, opts));
}

// What every instance starts with. llmFleet merges it into each
// llmManager.start(), so the model's context ladder, projector and tuned flags
// reach llama-server. Both context values come from ctxLadder(), which gives a
// model without its own ctxSize the global LLM.ctxSize: reading model.ctxSize
// here once handed llama-server `undefined` for the default model.
function llmRunOptions(model, { binaryPath, modelPath, mmprojPath }) {
  const ladder = ctxLadder(model);
  return {
    platform: process.platform, binaryPath, modelPath, mmprojPath,
    ctxSize: ladder[0], ctxLadder: ladder, extraArgs: model.extraArgs,
    alias: model.name,
  };
}

// GET <baseUrl>/health — resolves true when a llama-server is already listening
// and healthy on our port. Lets the app adopt an existing server instead of
// spawning a second one that would fail to bind 8080 (llama-server exits with
// "couldn't bind HTTP server socket" and the LLM silently never comes up). This
// happens right after an "Update & restart", when the outgoing server may still
// be releasing the port as the relaunched app tries to claim it.
function probeLlmHealth(baseUrl, timeoutMs) {
  return new Promise((resolve) => {
    let u;
    try { u = new URL(baseUrl + '/health'); } catch (e) { return resolve(false); }
    // Always http: the only URL it is given is our own llama-server's.
    const req = http.get(u, (res) => {
      if (res.statusCode !== 200) { res.resume(); return resolve(false); }
      // Require llama-server's own health body ({"status":"ok"}), not just any
      // 200 — otherwise unrelated software on the port (a dev server, NAS UI)
      // would be adopted as "the LLM" and chat would break against it.
      let raw = '';
      res.setEncoding('utf8');
      res.on('data', (c) => { raw += c; if (raw.length > 4096) res.destroy(); });
      res.on('end', () => {
        try { resolve(JSON.parse(raw).status === 'ok'); } catch (e) { resolve(false); }
      });
      res.on('error', () => resolve(false));
      // A destroyed response (the 4 KB cap above) never emits 'end', only
      // 'close'. After 'end' this resolve is a no-op.
      res.on('close', () => resolve(false));
    });
    req.on('error', () => resolve(false));
    req.setTimeout(timeoutMs || 2000, () => req.destroy());
  });
}

// The CLI probes its running llama-server with it too (shared/llmGuard's
// health check).
module.exports = { llmEngine, llmCudaEngine, makeLlmFleet, llmRunOptions, probeLlmHealth };
