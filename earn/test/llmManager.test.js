'use strict';

const { EventEmitter } = require('events');
const { LlmManager } = require('../src/main/llmManager');

function makeChild() {
  const proc = new EventEmitter();
  proc.stdout = new EventEmitter();
  proc.stderr = new EventEmitter();
  proc.kill = jest.fn();
  return proc;
}

describe('LlmManager', () => {
  test('constructs not running/ready; stop is a no-op', () => {
    const m = new LlmManager();
    expect(m.isRunning()).toBe(false);
    expect(m.isReady()).toBe(false);
    expect(m.stop()).toBe(false);
  });

  test('start spawns llama-server with built args and emits started with the base URL', () => {
    const child = makeChild();
    const spawn = jest.fn(() => child);
    const m = new LlmManager({ spawn });
    const started = jest.fn();
    m.on('started', started);

    const ok = m.start({ modelPath: '/m.gguf', nGpuLayers: 16, platform: 'win32' });

    expect(ok).toBe(true);
    expect(m.isRunning()).toBe(true);
    const [bin, args] = spawn.mock.calls[0];
    expect(bin).toBe('llama-server.exe');
    expect(args).toEqual(expect.arrayContaining(['--model', '/m.gguf', '--n-gpu-layers', '16']));
    expect(started).toHaveBeenCalledWith({ bin, args, baseUrl: 'http://127.0.0.1:8080' });
  });

  test('uses the bare binary off-Windows and is a no-op while already running', () => {
    const spawn = jest.fn(() => makeChild());
    const m = new LlmManager({ spawn });
    expect(m.start({})).toBe(true);
    expect(spawn.mock.calls[0][0]).toBe('llama-server');
    expect(m.start({})).toBe(false);
    expect(spawn).toHaveBeenCalledTimes(1);
  });

  test('flips to ready once on the listening line and emits stats for tokens/sec', () => {
    const child = makeChild();
    const m = new LlmManager({ spawn: () => child });
    const ready = jest.fn();
    const stats = jest.fn();
    const logs = [];
    m.on('ready', ready);
    m.on('stats', stats);
    m.on('log', (l) => logs.push(l.line));
    m.start({ modelPath: '/m.gguf' });

    // the pre-load listening line must NOT flip ready — the model is still loading
    child.stderr.emit('data', 'main: HTTP server is listening, hostname: 127.0.0.1, port: 8080\nloading model\n');
    expect(m.isReady()).toBe(false);
    expect(ready).not.toHaveBeenCalled();

    child.stderr.emit('data', 'main: server is listening on http://127.0.0.1:8080 - starting the main loop\n');
    expect(m.isReady()).toBe(true);
    expect(ready).toHaveBeenCalledTimes(1);
    expect(ready).toHaveBeenCalledWith({ baseUrl: 'http://127.0.0.1:8080' });

    // a later ready line must not re-emit ready
    child.stdout.emit('data', 'srv update_slots: all slots are idle\n');
    expect(ready).toHaveBeenCalledTimes(1);

    child.stdout.emit('data', 'eval time = 10 ms / 200 tokens ... 162.02 tokens per second\n');
    expect(stats).toHaveBeenCalledWith({ tokensPerSec: 162.02 });
    expect(logs).toContain('loading model');
  });

  test('exit resets state, errors are surfaced, and stop kills the process', () => {
    const child = makeChild();
    const m = new LlmManager({ spawn: () => child });
    const stopped = jest.fn();
    const onErr = jest.fn();
    m.on('stopped', stopped);
    m.on('error', onErr);
    m.start({ modelPath: '/m.gguf' });

    expect(m.stop()).toBe(true);
    expect(child.kill).toHaveBeenCalledTimes(1);

    child.emit('error', new Error('boom'));
    expect(onErr).toHaveBeenCalled();

    child.emit('exit', 0);
    expect(stopped).toHaveBeenCalledWith(0);
    expect(m.isRunning()).toBe(false);
    expect(m.isReady()).toBe(false);
  });

  test('tolerates a child with no stdout/stderr streams and start() with no opts', () => {
    const proc = new EventEmitter();
    proc.kill = jest.fn();
    const m = new LlmManager({ spawn: () => proc });
    expect(m.start()).toBe(true); // default opts
    proc.emit('exit', 1); // still wired without stream handlers
    expect(m.isRunning()).toBe(false);
  });

  // Self-heal on a port-bind clash (e.g. an update relaunch overlapping the
  // previous llama-server on port 8080): an exit before ready is retried.
  const flush = () => new Promise((r) => setImmediate(r));

  test('retries a spawn that exits before ready, then goes ready without emitting stopped', async () => {
    const children = [];
    const spawn = jest.fn(() => { const c = makeChild(); children.push(c); return c; });
    const m = new LlmManager({ spawn, sleep: () => Promise.resolve(), startAttempts: 3, retryDelayMs: 1 });
    const stopped = jest.fn();
    const ready = jest.fn();
    const logs = [];
    m.on('stopped', stopped);
    m.on('ready', ready);
    m.on('log', (l) => logs.push(l.line));

    m.start({ modelPath: '/m.gguf' });
    expect(spawn).toHaveBeenCalledTimes(1);

    // first attempt dies before ready → schedules a retry, stays "running"
    children[0].emit('exit', 1);
    await flush();
    expect(spawn).toHaveBeenCalledTimes(2);
    expect(stopped).not.toHaveBeenCalled();
    expect(m.isRunning()).toBe(true);
    expect(logs.some((l) => /retrying \(attempt 1\/3\)/.test(l))).toBe(true);

    // second attempt loads and goes ready
    children[1].stderr.emit('data', 'srv  llama_server: model loaded\n');
    expect(ready).toHaveBeenCalledTimes(1);
    expect(m.isReady()).toBe(true);

    // an exit AFTER it was ready is a real stop — no further retry
    children[1].emit('exit', 0);
    await flush();
    expect(spawn).toHaveBeenCalledTimes(2);
    expect(stopped).toHaveBeenCalledWith(0);
  });

  test('gives up after startAttempts spawns and emits stopped', async () => {
    const children = [];
    const spawn = jest.fn(() => { const c = makeChild(); children.push(c); return c; });
    const m = new LlmManager({ spawn, sleep: () => Promise.resolve(), startAttempts: 3, retryDelayMs: 1 });
    const stopped = jest.fn();
    m.on('stopped', stopped);

    m.start({});
    children[0].emit('exit', 1); await flush(); // → retry (spawn 2)
    children[1].emit('exit', 1); await flush(); // → retry (spawn 3)
    children[2].emit('exit', 1); await flush(); // exhausted → stopped

    expect(spawn).toHaveBeenCalledTimes(3);
    expect(stopped).toHaveBeenCalledWith(1);
    expect(m.isRunning()).toBe(false);
  });

  test('waits with a real timer between retries when no sleep is injected', async () => {
    const children = [];
    const spawn = jest.fn(() => { const c = makeChild(); children.push(c); return c; });
    const m = new LlmManager({ spawn, startAttempts: 2, retryDelayMs: 5 }); // default sleep

    m.start({});
    children[0].emit('exit', 1);
    await new Promise((r) => setTimeout(r, 25)); // let the real 5ms retry delay elapse
    expect(spawn).toHaveBeenCalledTimes(2);
  });

  test('stop() during the retry wait cancels the pending re-spawn', async () => {
    const children = [];
    let resolveSleep;
    const spawn = jest.fn(() => { const c = makeChild(); children.push(c); return c; });
    const m = new LlmManager({ spawn, sleep: () => new Promise((r) => { resolveSleep = r; }), startAttempts: 3, retryDelayMs: 5 });

    m.start({});
    children[0].emit('exit', 1); // schedules a retry, now waiting on sleep
    expect(m.stop()).toBe(true);  // proc already gone, but still "running" (retry pending)
    resolveSleep();
    await flush();

    expect(spawn).toHaveBeenCalledTimes(1); // retry was cancelled
    expect(m.isRunning()).toBe(false);
  });
});
