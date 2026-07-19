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
});
