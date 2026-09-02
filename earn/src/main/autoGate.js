'use strict';

// Wiring for demand-driven auto mode: owns the switch between the miner and the
// LLM, and the public endpoint in front of them.
//
// Split out of earn-cli because the interesting part is the lifecycle, not the
// shell around it. Stopping an engine here is deliberate and must NOT look like
// one dying -- the CLI's own 'stopped' handlers tear the process down -- so the
// switching flag is owned here and read by those handlers.

const { LlmGate, SERVING } = require('../shared/llmGate');
const { LlmGateServer } = require('./llmGateServer');

function createAutoGate(opts) {
  const {
    miner, startLlm, stopLlm, isLlmReady, startMinerArgs,
    port, upstreamPort, modelName, quietMs, log = () => {},
    onMinerFailed = () => {},
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
    let ok;
    try { ok = miner.start(startMinerArgs()); } finally { switching = false; }
    // Same contract as the up-front start in the CLI: false means the core did
    // not construct, so no socket, no job and no 'stopped' event is coming.
    // Dropping it here left the node believing it had resumed mining when the
    // card was in fact doing nothing, and telemetry kept reporting the last
    // hashrate forever because only a 'status' event overwrites it.
    if (ok === false) onMinerFailed();
  };

  // Wait for the server to actually answer, not merely to have been spawned: the
  // gate is holding a caller's request open across this, so returning early would
  // forward it into a socket that is not listening yet.
  const wrappedStartLlm = async () => {
    // finally, not a trailing assignment: on a throw the flag stayed true, and a
    // stuck `switching` makes the CLI's miner-'stopped' handler early-return
    // forever, so the process could neither exit nor be restarted by systemd.
    try {
      const fleet = await startLlm();
      if (!fleet) throw new Error('llama-server did not start');
      if (!(fleet.readyCount && fleet.readyCount() > 0)) {
        await new Promise((resolve, reject) => {
          const t = setTimeout(() => reject(new Error('llama-server was not ready in time')),
            llmReadyTimeoutMs);
          fleet.once('ready', () => { clearTimeout(t); resolve(); });
        });
      }
    } finally { switching = false; }
  };

  const wrappedStopLlm = async () => {
    switching = true;
    await stopLlm();
  };

  const gate = new LlmGate({
    quietMs, isLlmReady,
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

// A gate with nothing to switch.
//
// In llm mode there is no miner, so there is no handoff to manage -- but the
// public endpoint still has to BE the public endpoint. Without this the gate
// existed only in auto, so choosing llm mode silently moved callers from the
// documented port to llama-server's own one: the endpoint moved out from under
// every client at exactly the moment the operator committed to serving.
//
// Deliberately the same shape as createAutoGate, so the CLI's teardown and its
// `auto && ...` guards need no special case for it.
function createServeGate(opts = {}) {
  const {
    port, upstreamPort, modelName, isLlmReady, log = () => {},
  } = opts;
  const gate = new LlmGate({
    isLlmReady,
    state: SERVING,
    // Never release. There is no miner to release TO, and firing would flip the
    // reported state to MINING while the model is loaded and answering.
    quietMs: Infinity,
  });
  const server = new LlmGateServer({ port, upstreamPort, modelName, gate, log });
  return {
    gate,
    server,
    // Nothing here ever stops an engine, so no engine death is ever ours.
    isSwitching: () => false,
    start() { server.start(); return this; },
    stop() { server.stop(); },
  };
}

module.exports = { createAutoGate, createServeGate };
