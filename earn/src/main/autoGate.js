'use strict';

// Wiring for demand-driven auto mode: owns the switch between the miner and the
// LLM, and the public endpoint in front of them.
//
// Split out of earn-cli because the interesting part is the lifecycle, not the
// shell around it. Stopping an engine here is deliberate and must NOT look like
// one dying -- the CLI's own 'stopped' handlers tear the process down -- so the
// switching flag is owned here and read by those handlers.

const { LlmGate } = require('../shared/llmGate');
const { LlmGateServer } = require('./llmGateServer');

function createAutoGate(opts) {
  const {
    miner, startLlm, stopLlm, isLlmReady, startMinerArgs,
    port, upstreamPort, modelName, idleMs, log = () => {},
    minerStopTimeoutMs = 15000, llmReadyTimeoutMs = 180000,
  } = opts;

  let switching = false;

  const stopMiner = () => new Promise((resolve) => {
    switching = true;
    // Wait for the engine to actually exit: it holds VRAM until it does, and
    // llama-server started before then fails to allocate.
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      miner.removeListener('stopped', finish);
      resolve();
    };
    miner.once('stopped', finish);
    try { miner.stop(); } catch { finish(); }
    // Never wedge the gate on an engine that will not exit.
    setTimeout(finish, minerStopTimeoutMs);
  });

  const startMiner = async () => {
    switching = true;
    try { miner.start(startMinerArgs()); } finally { switching = false; }
  };

  // Wait for the server to actually answer, not merely to have been spawned: the
  // gate is holding a caller's request open across this, so returning early would
  // forward it into a socket that is not listening yet.
  const wrappedStartLlm = async () => {
    const fleet = await startLlm();
    if (!fleet) throw new Error('llama-server did not start');
    if (!(fleet.readyCount && fleet.readyCount() > 0)) {
      await new Promise((resolve, reject) => {
        const t = setTimeout(() => reject(new Error('llama-server was not ready in time')),
          llmReadyTimeoutMs);
        fleet.once('ready', () => { clearTimeout(t); resolve(); });
      });
    }
    switching = false;
  };

  const wrappedStopLlm = async () => {
    switching = true;
    await stopLlm();
  };

  const gate = new LlmGate({
    idleMs, isLlmReady,
    startLlm: wrappedStartLlm, stopLlm: wrappedStopLlm,
    startMiner, stopMiner,
  });
  gate.on('state', (s) => log('auto:       ' + s));

  const server = new LlmGateServer({ port, upstreamPort, modelName, gate, log });
  return {
    gate,
    server,
    isSwitching: () => switching,
    llmReadyTimeoutMs,
    start() { server.start(); return this; },
    stop() { server.stop(); },
  };
}

module.exports = { createAutoGate };
