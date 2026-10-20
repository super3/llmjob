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
const { spawn } = require('child_process');
const { LLM } = require('../shared/config');
const { ctxLadder } = require('../shared/models');
const { resolveServerUrl } = require('../shared/llama');
const { LlmEngineManager } = require('./llmEngineManager');
const { LlmManager } = require('./llmManager');
const { LlmFleet } = require('./llmFleet');
const { downloadFile } = require('./io');
const { findFreePort } = require('./probe');

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

// A fleet of llama-servers, one per planned card: each walks to the first free
// port from LLM.port, and retries an early exit a few times for a port clash
// that clears. `opts` adds makeWorker and anything else the shell needs.
function makeLlmFleet(opts) {
  return new LlmFleet(Object.assign({
    host: LLM.host,
    basePort: LLM.port,
    makeManager: () => new LlmManager({ spawn, startAttempts: LLM.startAttempts, retryDelayMs: LLM.startRetryMs }),
    findFreePort,
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

module.exports = { llmEngine, makeLlmFleet, llmRunOptions };
