'use strict';

const EventEmitter = require('events');
const { createAutoGate } = require('../src/main/autoGate');

// A fleet-like object: the gate waits on it for readiness.
function mkFleet(readyNow) {
  const f = new EventEmitter();
  f.readyCount = () => (readyNow ? 1 : 0);
  return f;
}

function mkMiner() {
  const m = new EventEmitter();
  m.start = jest.fn();
  m.stop = jest.fn(() => { setImmediate(() => m.emit('stopped', 0)); });
  return m;
}
function mk(over = {}) {
  const miner = mkMiner();
  const calls = [];
  let ready = false;
  const auto = createAutoGate(Object.assign({
    miner,
    startMinerArgs: () => ({ endpoint: 'pool:1200' }),
    isLlmReady: () => ready,
    startLlm: async () => { calls.push('startLlm'); ready = true; return mkFleet(true); },
    stopLlm: async () => { calls.push('stopLlm'); ready = false; },
    port: 0, upstreamPort: 8080, modelName: 'M', quietMs: 50,
  }, over));
  return { auto, miner, calls, setReady: (v) => { ready = v; } };
}

describe('createAutoGate', () => {
  test('waking stops the miner and waits for it to actually exit', async () => {
    const { auto, miner, calls } = mk();
    await auto.gate.ensureServing();
    expect(miner.stop).toHaveBeenCalled();
    expect(calls).toEqual(['startLlm']);
  });

  test('isSwitching is true across a deliberate stop and clears once settled', async () => {
    const { auto, miner } = mk();
    const p = auto.gate.ensureServing();
    expect(auto.isSwitching()).toBe(true);   // so the CLI ignores the miner's 'stopped'
    await p;
    expect(auto.isSwitching()).toBe(false);
    expect(miner.stop).toHaveBeenCalledTimes(1);
  });

  test('releasing stops the LLM and restarts the miner with its endpoint', async () => {
    const { auto, miner, calls } = mk();
    await auto.gate.ensureServing();
    calls.length = 0;
    await auto.gate.ensureMining();
    expect(calls).toEqual(['stopLlm']);
    expect(miner.start).toHaveBeenCalledWith({ endpoint: 'pool:1200' });
    expect(auto.isSwitching()).toBe(false);
  });

  // A miner that never emits 'stopped' must not wedge the gate forever, or auto
  // mode would stop serving entirely after one bad transition.
  test('a miner that never exits is given up on after the timeout', async () => {
    jest.useFakeTimers();
    const miner = new EventEmitter();
    miner.start = jest.fn(); miner.stop = jest.fn();   // never emits 'stopped'
    let ready = false;
    const auto = createAutoGate({
      miner, startMinerArgs: () => ({}), isLlmReady: () => ready,
      startLlm: async () => { ready = true; return mkFleet(true); }, stopLlm: async () => { ready = false; },
      port: 0, upstreamPort: 8080, modelName: 'M', quietMs: 50, minerStopTimeoutMs: 100,
    });
    const p = auto.gate.ensureServing();
    jest.advanceTimersByTime(150);
    await p;
    expect(auto.gate.state).toBe('SERVING');
    jest.useRealTimers();
  });

  test('a miner whose stop() throws still completes the transition', async () => {
    const miner = new EventEmitter();
    miner.start = jest.fn();
    miner.stop = jest.fn(() => { throw new Error('already gone'); });
    let ready = false;
    const { auto } = mk({ miner, isLlmReady: () => ready,
      startLlm: async () => { ready = true; return mkFleet(true); } });
    await auto.gate.ensureServing();
    expect(auto.gate.state).toBe('SERVING');
  });

  test('start/stop bring the endpoint up and down', () => {
    const { auto } = mk();
    auto.start();
    expect(auto.server.server).toBeTruthy();
    auto.stop();
    expect(auto.server.server).toBeNull();
  });

  test('state changes are logged', async () => {
    const lines = [];
    const { auto } = mk({ log: (l) => lines.push(l) });
    await auto.gate.ensureServing();
    expect(lines.join('\n')).toContain('SERVING');
  });
});

test('the stop timeout firing after a real stop does not resolve twice', async () => {
  jest.useFakeTimers();
  const miner = new EventEmitter();
  miner.start = jest.fn();
  miner.stop = jest.fn(() => { miner.emit('stopped', 0); });   // synchronous exit
  let ready = false;
  const auto = createAutoGate({
    miner, startMinerArgs: () => ({}), isLlmReady: () => ready,
    startLlm: async () => { ready = true; return mkFleet(true); }, stopLlm: async () => { ready = false; },
    port: 0, upstreamPort: 8080, modelName: 'M', quietMs: 50, minerStopTimeoutMs: 10,
  });
  const p = auto.gate.ensureServing();
  jest.advanceTimersByTime(50);   // the guard timer fires after 'stopped' already did
  await p;
  expect(auto.gate.state).toBe('SERVING');
  jest.useRealTimers();
});

describe('waiting for the server to answer', () => {
  test('a fleet that is not ready yet is waited for, then resolves on its event', async () => {
    const fleet = mkFleet(false);
    const { auto } = mk({ startLlm: async () => fleet });
    const p = auto.gate.ensureServing();
    await new Promise((r) => setImmediate(r));
    fleet.emit('ready');
    await p;
    expect(auto.gate.state).toBe('SERVING');
  });

  test('a fleet that never answers fails rather than holding the caller forever', async () => {
    const { auto } = mk({ startLlm: async () => mkFleet(false), llmReadyTimeoutMs: 30 });
    await expect(auto.gate.ensureServing()).rejects.toThrow('was not ready in time');
  });

  test('a fleet that failed to start is reported, not treated as up', async () => {
    const { auto } = mk({ startLlm: async () => null });
    await expect(auto.gate.ensureServing()).rejects.toThrow('llama-server did not start');
  });
});
