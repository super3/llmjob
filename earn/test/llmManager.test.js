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
    expect(stats).toHaveBeenCalledWith({ tokensPerSec: 162.02, promptTokensPerSec: null });
    // The prefill line is tagged separately: one regex used to match both, so
    // whichever printed last was reported as though it were the other.
    child.stdout.emit('data', 'prompt eval time = 5 ms / 900 tokens ... 1840.00 tokens per second\n');
    expect(stats).toHaveBeenCalledWith({ tokensPerSec: null, promptTokensPerSec: 1840 });
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

    // An exit AFTER it was ready is a crash, not a stop: llama-server dying
    // mid-serve used to be permanent, which took nodes out of the cluster for
    // hours while the miner kept running and the machine still looked healthy.
    children[1].emit('exit', 0);
    await flush();
    expect(spawn).toHaveBeenCalledTimes(3);
    expect(stopped).not.toHaveBeenCalled();
    expect(m.isRunning()).toBe(true);
  });

  test('restarts a crash after it was serving, and says so', async () => {
    const children = [];
    const spawn = jest.fn(() => { const c = makeChild(); children.push(c); return c; });
    const m = new LlmManager({ spawn, sleep: () => Promise.resolve(), retryDelayMs: 1 });
    const stopped = jest.fn();
    const crashed = jest.fn();
    const logs = [];
    m.on('stopped', stopped);
    m.on('crashed', crashed);
    m.on('log', (l) => logs.push(l.line));

    m.start({ modelPath: '/m.gguf' });
    children[0].stderr.emit('data', 'srv  llama_server: model loaded\n');
    expect(m.isReady()).toBe(true);

    children[0].emit('exit', 139); // segfault mid-generation
    await flush();

    expect(spawn).toHaveBeenCalledTimes(2);
    expect(stopped).not.toHaveBeenCalled();
    // The fleet needs this to pull the instance out of service while it is down,
    // or it keeps advertising a model that isn't listening and fails every job.
    expect(crashed).toHaveBeenCalledWith(expect.objectContaining({ code: 139, attempt: 1 }));
    expect(m.isReady()).toBe(false);
    expect(logs.some((l) => /crashed \(code 139\) after serving/.test(l))).toBe(true);
  });

  test('gives up after crashRestarts consecutive crashes', async () => {
    const children = [];
    const spawn = jest.fn(() => { const c = makeChild(); children.push(c); return c; });
    const m = new LlmManager({ spawn, sleep: () => Promise.resolve(), retryDelayMs: 1, crashRestarts: 2 });
    const stopped = jest.fn();
    m.on('stopped', stopped);

    m.start({ modelPath: '/m.gguf' });
    // Ready then dies, three times over: a model that cannot stay up must end as
    // a stopped LLM the user can see, not an endless respawn on their GPU.
    for (let i = 0; i < 3; i++) {
      children[i].stderr.emit('data', 'srv  llama_server: model loaded\n');
      children[i].emit('exit', 1);
      await flush();
    }
    expect(spawn).toHaveBeenCalledTimes(3); // initial + 2 restarts, then no more
    expect(stopped).toHaveBeenCalledWith(1);
    expect(m.isRunning()).toBe(false);
  });

  test('forgives earlier crashes once an instance has stayed up', async () => {
    const children = [];
    const spawn = jest.fn(() => { const c = makeChild(); children.push(c); return c; });
    let now = 1000;
    const m = new LlmManager({
      spawn, sleep: () => Promise.resolve(), retryDelayMs: 1,
      crashRestarts: 1, crashSettleMs: 60000, now: () => now,
    });
    const stopped = jest.fn();
    m.on('stopped', stopped);

    m.start({ modelPath: '/m.gguf' });
    children[0].stderr.emit('data', 'srv  llama_server: model loaded\n');
    children[0].emit('exit', 1);          // crash 1 of a budget of 1
    await flush();
    expect(spawn).toHaveBeenCalledTimes(2);

    children[1].stderr.emit('data', 'srv  llama_server: model loaded\n');
    now += 10 * 60 * 1000;                 // served ten minutes — a recovery
    children[1].emit('exit', 1);
    await flush();
    // Without the reset a node that crashes once a day would exhaust its budget
    // and stay down for good.
    expect(spawn).toHaveBeenCalledTimes(3);
    expect(stopped).not.toHaveBeenCalled();
  });

  test('a user stop after ready does not restart', async () => {
    const children = [];
    const spawn = jest.fn(() => { const c = makeChild(); children.push(c); return c; });
    const m = new LlmManager({ spawn, sleep: () => Promise.resolve(), retryDelayMs: 1 });
    const stopped = jest.fn();
    m.on('stopped', stopped);

    m.start({ modelPath: '/m.gguf' });
    children[0].stderr.emit('data', 'srv  llama_server: model loaded\n');
    m.stop();                    // sets _stopping — this is intentional
    children[0].emit('exit', 0);
    await flush();

    expect(spawn).toHaveBeenCalledTimes(1);
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

// A large-context model is the one case where "won't start" is a sizing problem
// the client can fix itself, so it walks a ladder of context sizes down instead
// of restarting forever at a size the card cannot hold.
describe('LlmManager context ladder', () => {
  function ladderManager(extra = {}) {
    const children = [];
    const spawn = jest.fn(() => { const c = makeChild(); children.push(c); return c; });
    const m = new LlmManager({
      spawn, sleep: () => Promise.resolve(), startAttempts: 2, retryDelayMs: 0, ...extra,
    });
    return { m, spawn, children };
  }
  const ctxOf = (spawn, i) => {
    const args = spawn.mock.calls[i][1];
    return args[args.indexOf('--ctx-size') + 1];
  };
  const flush = () => new Promise((r) => setTimeout(r, 0));

  test('starts at the top rung', () => {
    const { m, spawn } = ladderManager();
    m.start({ modelPath: '/m.gguf', ctxLadder: [262144, 65536], ctxSize: 999 });
    // The ladder wins over a bare ctxSize — otherwise the two could disagree and
    // the server would be asked for a window the caller never chose.
    expect(ctxOf(spawn, 0)).toBe('262144');
  });

  test('exhausts the start retries at one rung BEFORE dropping to the next', async () => {
    // A port-bind clash also exits before ready. Shrinking the context because
    // the previous server had not released 8080 would apply the wrong fix, and
    // apply it permanently — so the retries at the current size come first.
    const { m, spawn, children } = ladderManager();
    const down = jest.fn();
    m.on('ctx-downgrade', down);
    m.start({ modelPath: '/m.gguf', ctxLadder: [262144, 65536] });

    children[0].emit('exit', 1);
    await flush();
    expect(spawn).toHaveBeenCalledTimes(2);
    expect(ctxOf(spawn, 1)).toBe('262144');   // still the same rung
    expect(down).not.toHaveBeenCalled();

    children[1].emit('exit', 1);
    await flush();
    expect(spawn).toHaveBeenCalledTimes(3);
    expect(ctxOf(spawn, 2)).toBe('65536');    // now it steps down
    expect(down).toHaveBeenCalledWith({ from: 262144, to: 65536 });
  });

  test('gives up once the ladder is spent, rather than looping forever', async () => {
    const { m, spawn, children } = ladderManager();
    const stopped = jest.fn();
    m.on('stopped', stopped);
    m.start({ modelPath: '/m.gguf', ctxLadder: [262144, 65536] });

    for (let i = 0; i < 4; i++) { children[i].emit('exit', 1); await flush(); }
    expect(spawn).toHaveBeenCalledTimes(4);   // 2 attempts x 2 rungs
    expect(stopped).toHaveBeenCalledWith(1);
    expect(m.isRunning()).toBe(false);
  });

  test('a server that becomes ready never downgrades', async () => {
    // Reaching ready proves the size fits. A later exit is a crash, and a crash
    // must not quietly shrink the context the node advertises.
    const { m, spawn, children } = ladderManager({ crashRestarts: 1 });
    const down = jest.fn();
    m.on('ctx-downgrade', down);
    m.start({ modelPath: '/m.gguf', ctxLadder: [262144, 65536] });

    children[0].stderr.emit('data', 'main: server is listening on http://127.0.0.1:8080 - starting the main loop\n');
    expect(m.isReady()).toBe(true);
    children[0].emit('exit', 1);
    await flush();

    expect(down).not.toHaveBeenCalled();
    expect(ctxOf(spawn, 1)).toBe('262144');
  });

  test('no ladder means the caller ctxSize is used unchanged', async () => {
    const { m, spawn, children } = ladderManager();
    m.start({ modelPath: '/m.gguf', ctxSize: 32768 });
    expect(ctxOf(spawn, 0)).toBe('32768');
    children[0].emit('exit', 1);
    await flush();
    expect(ctxOf(spawn, 1)).toBe('32768');
  });

  test('junk rungs are ignored rather than asked for', () => {
    const { m, spawn } = ladderManager();
    m.start({ modelPath: '/m.gguf', ctxLadder: [0, -5, 65536], ctxSize: 1 });
    expect(ctxOf(spawn, 0)).toBe('65536');
  });
});
