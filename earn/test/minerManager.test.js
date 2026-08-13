'use strict';

const { EventEmitter } = require('events');
const { MinerManager } = require('../src/main/minerManager');

function makeChild() {
  const proc = new EventEmitter();
  proc.stdout = new EventEmitter();
  proc.stderr = new EventEmitter();
  proc.kill = jest.fn();
  return proc;
}

describe('MinerManager', () => {
  test('constructs with no arguments and reports not running', () => {
    const mgr = new MinerManager();
    expect(mgr.isRunning()).toBe(false);
    expect(mgr.stop()).toBe(false); // nothing to stop
  });

  test('start spawns the binary with built args and emits started', () => {
    const child = makeChild();
    const spawn = jest.fn(() => child);
    const mgr = new MinerManager({ spawn });
    const started = jest.fn();
    mgr.on('started', started);

    const ok = mgr.start({ address: 'prl1pabc', platform: 'win32' });

    expect(ok).toBe(true);
    expect(mgr.isRunning()).toBe(true);
    expect(spawn).toHaveBeenCalledTimes(1);
    const [bin, args] = spawn.mock.calls[0];
    expect(bin).toBe('alpha-miner-windows.exe');
    expect(args).toEqual(expect.arrayContaining(['--address', 'prl1pabc', '--worker', 'rig01']));
    expect(started).toHaveBeenCalledWith({ bin, args });
  });

  test('start defaults settings when called with no arguments', () => {
    const child = makeChild();
    const spawn = jest.fn(() => child);
    const mgr = new MinerManager({ spawn });
    expect(mgr.start()).toBe(true);
    expect(spawn.mock.calls[0][0]).toBe('alpha-miner'); // non-Windows default binary
  });

  test('start is a no-op while already running', () => {
    const spawn = jest.fn(() => makeChild());
    const mgr = new MinerManager({ spawn });
    expect(mgr.start({ address: 'prl1pabc' })).toBe(true);
    expect(mgr.start({ address: 'prl1pabc' })).toBe(false);
    expect(spawn).toHaveBeenCalledTimes(1);
  });

  test('stdout is split into log lines and parsed events', () => {
    const child = makeChild();
    const mgr = new MinerManager({ spawn: () => child });
    const logs = [];
    const events = [];
    mgr.on('log', (l) => logs.push(l));
    mgr.on('event', (e) => events.push(e));
    mgr.start({ address: 'prl1pabc' });

    const connected = 'ts level=INFO gpu=0:NVIDIA GeForce RTX 4090 component=pool connected host=us2.alphapool.tech port=5566 tls=false';
    const status = 'ts level=INFO gpu=0:NVIDIA GeForce RTX 4090 component=miner status attempts=100 accepted=5 rejected=0 hashrate_th_s=286.86 power=449W';
    child.stdout.emit('data', connected + '\njust noise\n\n' + status + '\n');

    expect(logs.map((l) => l.line)).toEqual([connected, 'just noise', status]);
    expect(logs.every((l) => l.level === 'info')).toBe(true);
    expect(events).toEqual([
      { type: 'connected', gpuIndex: 0, endpoint: 'us2.alphapool.tech:5566', gpu: 'NVIDIA GeForce RTX 4090' },
      { type: 'status', gpuIndex: 0, hashrate: 286.86, accepted: 5, rejected: 0, power: 449, temp: null, gpu: 'NVIDIA GeForce RTX 4090' },
    ]);
  });

  // A 'data' chunk is a slice of the pipe, not a line. Splitting each chunk on
  // its own turned every straddling status line into two fragments in the log
  // (`…power=44` then `9W`) and parseLine saw neither as an event.
  test('a status line split across two chunks is emitted once, whole', () => {
    const child = makeChild();
    const mgr = new MinerManager({ spawn: () => child });
    const logs = [];
    const events = [];
    mgr.on('log', (l) => logs.push(l));
    mgr.on('event', (e) => events.push(e));
    mgr.start({ address: 'prl1pabc' });

    const status = 'ts level=INFO gpu=0:NVIDIA GeForce RTX 4090 component=miner status attempts=100 accepted=5 rejected=0 hashrate_th_s=286.86 power=449W';
    const cut = status.length - 2; // splits mid-"449W", as the real log did
    child.stdout.emit('data', status.slice(0, cut));
    expect(logs).toEqual([]); // nothing emitted until the line is complete
    child.stdout.emit('data', status.slice(cut) + '\n');

    expect(logs.map((l) => l.line)).toEqual([status]);
    expect(events).toEqual([
      { type: 'status', gpuIndex: 0, hashrate: 286.86, accepted: 5, rejected: 0, power: 449, temp: null, gpu: 'NVIDIA GeForce RTX 4090' },
    ]);
  });

  test('a trailing partial line is flushed when the child exits', () => {
    const child = makeChild();
    const mgr = new MinerManager({ spawn: () => child });
    const logs = [];
    mgr.on('log', (l) => logs.push(l));
    mgr.start({ address: 'prl1pabc' });

    child.stdout.emit('data', 'died mid-sentence');
    child.emit('exit', 1);
    expect(logs.map((l) => l.line)).toEqual(['died mid-sentence']);

    // A restart must not inherit the previous run's leftovers.
    mgr.start({ address: 'prl1pabc' });
    expect(mgr.stdoutBuf).toBe('');
  });

  test('a runaway line with no newline is flushed at the cap', () => {
    const child = makeChild();
    const mgr = new MinerManager({ spawn: () => child });
    const logs = [];
    mgr.on('log', (l) => logs.push(l));
    mgr.start({ address: 'prl1pabc' });

    child.stdout.emit('data', 'x'.repeat(64 * 1024 + 1));
    expect(logs).toHaveLength(1);
    expect(logs[0].line).toHaveLength(64 * 1024 + 1);
    expect(mgr.stdoutBuf).toBe(''); // buffer released, next chunk starts fresh
  });

  test('an exit with nothing buffered emits no stray log line', () => {
    const child = makeChild();
    const mgr = new MinerManager({ spawn: () => child });
    const logs = [];
    mgr.on('log', (l) => logs.push(l));
    mgr.start({ address: 'prl1pabc' });

    child.stdout.emit('data', 'complete\n');
    child.emit('exit', 0);
    expect(logs.map((l) => l.line)).toEqual(['complete']);
  });

  test('stderr is emitted as error-level log', () => {
    const child = makeChild();
    const mgr = new MinerManager({ spawn: () => child });
    const logs = [];
    mgr.on('log', (l) => logs.push(l));
    mgr.start({ address: 'prl1pabc' });

    child.stderr.emit('data', '  boom  ');
    expect(logs).toContainEqual({ level: 'error', line: 'boom' });
  });

  test('child error is surfaced', () => {
    const child = makeChild();
    const mgr = new MinerManager({ spawn: () => child });
    const onErr = jest.fn();
    mgr.on('error', onErr);
    mgr.start({ address: 'prl1pabc' });

    const err = new Error('spawn failed');
    child.emit('error', err);
    expect(onErr).toHaveBeenCalledWith(err);
  });

  test('exit resets state and emits stopped with the code', () => {
    const child = makeChild();
    const mgr = new MinerManager({ spawn: () => child });
    const stopped = jest.fn();
    mgr.on('stopped', stopped);
    mgr.start({ address: 'prl1pabc' });

    child.emit('exit', 0);
    expect(stopped).toHaveBeenCalledWith(0);
    expect(mgr.isRunning()).toBe(false);
    expect(mgr.stop()).toBe(false); // proc cleared
  });

  test('stop kills the running process', () => {
    const child = makeChild();
    const mgr = new MinerManager({ spawn: () => child });
    mgr.start({ address: 'prl1pabc' });
    expect(mgr.stop()).toBe(true);
    expect(child.kill).toHaveBeenCalledTimes(1);
  });
});

describe('MinerManager — injected launch plan (SRBMiner)', () => {
  // A second engine needs a different executable and a different argument
  // vector, but the same supervision: stderr routing, exit codes, the
  // partial-line buffer. Injecting the plan keeps all of that in one place
  // instead of forking the manager per engine.
  test('spawns what the plan says, not the alpha-miner resolution', () => {
    const child = makeChild();
    const spawn = jest.fn(() => child);
    const planFor = jest.fn(() => ({ bin: '/cache/SRBMiner-MULTI', args: ['--algorithm', 'pearlhash'] }));
    const mgr = new MinerManager({ spawn, planFor });
    const started = jest.fn();
    mgr.on('started', started);

    expect(mgr.start({ address: 'prl1abc', platform: 'linux' })).toBe(true);

    expect(planFor).toHaveBeenCalledWith({ address: 'prl1abc', platform: 'linux' });
    expect(spawn).toHaveBeenCalledWith('/cache/SRBMiner-MULTI', ['--algorithm', 'pearlhash']);
    expect(started).toHaveBeenCalledWith({ bin: '/cache/SRBMiner-MULTI', args: ['--algorithm', 'pearlhash'] });
    expect(mgr.isRunning()).toBe(true);
  });
});
