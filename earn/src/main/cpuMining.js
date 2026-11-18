'use strict';

// ── CPU mining (Monero) ─────────────────────────────────────────────────────
// The desktop app's CPU miner, moved out of main.js unchanged.
//
// Opt-in and off by default (Settings → CPU Mining). runPlan starts it for any
// START that asks for it, whether or not the GPU miner runs (the CLI does the
// same), and STOP and the idle pause stop it. The hashing runs in a worker
// process of its own (see xmrEngine), so a crash there, or an antivirus
// removing the worker, stops the CPU miner and nothing else. Its line on the
// Mine tab comes on 'cpu:stats', apart from the GPU miner's ticks, which an
// LLM-only START never sends.
//
// What it needs from main.js:
//   send(channel, payload)  to the window
//   xmrEngine(opts)         shared/minerDeps' Monero engine
//   onGaveUp()              the engine stopped on its own: end the START when
//                           nothing else runs
//   inSession()             whether a START is still on after that

const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const { findWorker, makeSpawnWorker } = require('./xmrWorker');
const { readCpuFacts } = require('./cpuFacts');
const { planCpuThreads } = require('../shared/xmr/cpuPlan');
const { checkXmrAddress } = require('../shared/xmr/address');
const { DEFAULTS, liveRegion, xmrEndpointFor } = require('../shared/config');
const format = require('../shared/format');

function createCpuMining({ send, xmrEngine, onGaveUp, inSession }) {
  let cpuMiner = null;
  let cpuError = null;     // why the CPU miner last stopped, for the Mine tab
  let cpuFacts = null;     // read once: CPUs, cgroup limit, memory, L3

  // Where this install's RandomX worker is, or null: a build without one (macOS
  // has none), or an antivirus removed it.
  function workerPath() {
    return findWorker({ resourcesPath: process.resourcesPath, env: process.env, exists: fs.existsSync });
  }

  // The thread plan for a requested count (0 for the default).
  function plan(threads) {
    if (!cpuFacts) cpuFacts = readCpuFacts({ fs, os });
    return planCpuThreads(cpuFacts, threads);
  }

  function running() {
    return !!(cpuMiner && cpuMiner.isRunning());
  }

  function sendCpu() {
    send('cpu:stats', view());
  }

  function start(settings) {
    if (cpuMiner || !settings.cpuMining) return;
    cpuError = null;
    const say = (level, line) => send('miner:log', { level, line: 'cpu: ' + line });
    const addr = checkXmrAddress(settings.xmrAddress);
    if (!addr.ok) {
      cpuError = 'Monero address: ' + addr.error;
      say('warn', 'not mining Monero: ' + cpuError);
      sendCpu();
      return;
    }
    const threads = plan(Number(settings.cpuThreads) || 0);
    if (!threads.threads) {
      cpuError = threads.reason;
      say('warn', 'not mining Monero: ' + threads.reason);
      sendCpu();
      return;
    }
    const engine = xmrEngine({ spawnWorker: makeSpawnWorker(spawn), workerPath: workerPath() });
    engine.on('log', (l) => send('miner:log', l));
    // Kept for the Mine tab. The engine has logged it.
    engine.on('error', (e) => { cpuError = e.message; });
    // Every status change and hashrate tick (every 2 s) updates the Mine tab.
    engine.on('event', () => { if (cpuMiner === engine) sendCpu(); });
    // Code 1: the engine gave up on its own. A stop we asked for is code 0, and
    // stop() lets go of the engine first anyway. With nothing else running the
    // START ends, or the window shows STOP over nothing. The line keeps saying
    // why.
    engine.on('stopped', (code) => {
      if (code !== 1 || cpuMiner !== engine) return;
      const why = cpuError;
      onGaveUp();
      if (!inSession()) {
        cpuError = why;
        sendCpu();
      }
    });
    cpuMiner = engine;
    if (threads.reason) say('info', threads.reason);
    engine.start({
      address: String(settings.xmrAddress).trim(),
      worker: settings.worker || DEFAULTS.worker,
      // The pool and server picked for Pearl.
      endpoint: xmrEndpointFor(liveRegion(settings.pool, settings.region), settings.pool),
      threads: threads.threads,
    });
    sendCpu();
  }

  function stop() {
    if (!cpuMiner && !cpuError) return;
    const m = cpuMiner;
    cpuMiner = null;
    cpuError = null;
    if (m) m.stop();
    sendCpu();
  }

  // The CPU line on the Mine tab, or null when CPU mining is not running.
  function view() {
    if (!cpuMiner && !cpuError) return null;
    const s = cpuMiner ? cpuMiner.snapshot() : { state: 'stopped', hashrate: 0, accepted: 0, rejected: 0, threads: 0 };
    const off = s.state === 'stopped' || s.state === 'failed';
    return {
      state: s.state,
      hashrate: format.formatKhs(s.hashrate),
      accepted: s.accepted,
      rejected: s.rejected,
      threads: s.threads,
      error: off ? cpuError : null,
      // Why the pool can't be used, while the miner runs without it.
      pool: off ? null : s.poolError || null,
    };
  }

  return {
    start, stop, running, view, plan, workerPath,
    // The engine while one runs, for the board's CPU hashrate (hostStats).
    engine: () => cpuMiner,
  };
}

module.exports = { createCpuMining };
