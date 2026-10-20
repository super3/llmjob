'use strict';

// The LLM start-up pieces both shells share (src/main/llmLaunch.js). The engine,
// fleet and manager are the real classes: only what they would download or
// probe is mocked.

jest.mock('../src/main/io', () => ({ downloadFile: jest.fn() }));
jest.mock('../src/main/probe', () => ({ findFreePort: jest.fn() }));
jest.mock('child_process', () => ({ spawn: jest.fn() }));

const fs = require('fs');
const { spawn } = require('child_process');
const { downloadFile } = require('../src/main/io');
const { findFreePort } = require('../src/main/probe');
const { LLM } = require('../src/shared/config');
const { ctxLadder } = require('../src/shared/models');
const { resolveServerUrl } = require('../src/shared/llama');
const { LlmEngineManager } = require('../src/main/llmEngineManager');
const { LlmManager } = require('../src/main/llmManager');
const { LlmFleet } = require('../src/main/llmFleet');
const { llmEngine, makeLlmFleet, llmRunOptions } = require('../src/main/llmLaunch');

describe('llmEngine', () => {
  // Both shells fetch through this, so the CLI goes through ensureServer too,
  // which puts the execute bit back on a cached binary.
  it('builds the engine for this platform with the shell\'s extractor', () => {
    const extract = jest.fn();
    const engine = llmEngine('/llm', extract);
    expect(engine).toBeInstanceOf(LlmEngineManager);
    expect(engine).toMatchObject({
      dir: '/llm', platform: process.platform, serverUrl: resolveServerUrl(process.platform, process.arch),
      fs, download: downloadFile, extract, chmod: fs.chmodSync,
    });
  });
});

describe('makeLlmFleet', () => {
  it('spawns retrying managers on free ports from LLM.port, with the shell\'s extras', () => {
    const makeWorker = jest.fn();
    const probeHealth = jest.fn();
    const fleet = makeLlmFleet({ makeWorker, probeHealth });
    expect(fleet).toBeInstanceOf(LlmFleet);
    expect(fleet).toMatchObject({ host: LLM.host, basePort: LLM.port, findFreePort, makeWorker, probeHealth });
    const mgr = fleet.makeManager();
    expect(mgr).toBeInstanceOf(LlmManager);
    expect(mgr).toMatchObject({ spawn, startAttempts: LLM.startAttempts, retryDelayMs: LLM.startRetryMs });
  });
});

describe('llmRunOptions', () => {
  const files = { binaryPath: '/b', modelPath: '/m', mmprojPath: null };

  // The default model has no ctxSize of its own; reading model.ctxSize handed
  // llama-server `undefined` for it once.
  it('gives the default model the global context size', () => {
    expect(llmRunOptions(LLM.model, files)).toEqual({
      platform: process.platform, binaryPath: '/b', modelPath: '/m', mmprojPath: null,
      ctxSize: LLM.ctxSize, ctxLadder: [LLM.ctxSize], extraArgs: LLM.model.extraArgs,
      alias: LLM.model.name,
    });
  });

  it('carries a tier\'s own ladder, flags and projector', () => {
    const tier = LLM.tiers.find((t) => t.extraArgs && ctxLadder(t).length > 1);
    const opts = llmRunOptions(tier, { binaryPath: '/b', modelPath: '/m', mmprojPath: '/p' });
    expect(opts).toMatchObject({
      mmprojPath: '/p', ctxSize: ctxLadder(tier)[0], ctxLadder: ctxLadder(tier),
      extraArgs: tier.extraArgs, alias: tier.name,
    });
  });
});
