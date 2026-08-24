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
    expect(bin).toBe('peakminer.exe');
    expect(args).toEqual(expect.arrayContaining(['-c', 'pearl', '-u', 'prl1pabc', '-w', 'rig01']));
    expect(started).toHaveBeenCalledWith({ bin, args });
  });

  test('start defaults settings when called with no arguments', () => {
    const child = makeChild();
    const spawn = jest.fn(() => child);
    const mgr = new MinerManager({ spawn });
    expect(mgr.start()).toBe(true);
    expect(spawn.mock.calls[0][0]).toBe('peakminer'); // non-Windows default binary
  });

  test('start is a no-op while already running', () => {
    const spawn = jest.fn(() => makeChild());
    const mgr = new MinerManager({ spawn });
    expect(mgr.start({ address: 'prl1pabc' })).toBe(true);
    expect(mgr.start({ address: 'prl1pabc' })).toBe(false);
    expect(spawn).toHaveBeenCalledTimes(1);
  });

  // PeakMiner writes everything to stderr and leaves stdout empty, so the event
  // stream has to come off stderr. Both are wired to the same buffer.
  test('stderr is split into log lines and parsed events', () => {
    const child = makeChild();
    const mgr = new MinerManager({ spawn: () => child });
    const logs = [];
    const events = [];
    mgr.on('log', (l) => logs.push(l));
    mgr.on('event', (e) => events.push(e));
    mgr.start({ address: 'prl1pabc' });

    const connected = '2026-08-23 23:38:40  INFO connected us.pearl.herominers.com:1200  diff —  ping 1059ms';
    const status = '  0  RTX 4090  296.5 TH/s       3 / 0      78°C   48%   449W  660.4 GH/W  10251MHz   2340MHz';
    child.stderr.emit('data', connected + '\njust noise\n\n' + status + '\n');

    expect(logs.map((l) => l.line)).toEqual([connected, 'just noise', status]);
    expect(logs.every((l) => l.level === 'info')).toBe(true);
    expect(events).toEqual([
      { type: 'connected', gpuIndex: null, endpoint: 'us.pearl.herominers.com:1200', gpu: null },
      { type: 'status', gpuIndex: 0, hashrate: 296.5, accepted: 3, rejected: 0, power: 449, temp: 78, gpu: 'RTX 4090' },
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

    const status = '  0  RTX 4090  296.5 TH/s       3 / 0      78°C   48%   449W  660.4 GH/W  10251MHz   2340MHz';
    const cut = status.length - 6; // splits mid-clock, as the real log did
    child.stderr.emit('data', status.slice(0, cut));
    expect(logs).toEqual([]); // nothing emitted until the line is complete
    child.stderr.emit('data', status.slice(cut) + '\n');

    expect(logs.map((l) => l.line)).toEqual([status]);
    expect(events).toEqual([
      { type: 'status', gpuIndex: 0, hashrate: 296.5, accepted: 3, rejected: 0, power: 449, temp: 78, gpu: 'RTX 4090' },
    ]);
  });

  test('a trailing partial line is flushed when the child exits', () => {
    const child = makeChild();
    const mgr = new MinerManager({ spawn: () => child });
    const logs = [];
    mgr.on('log', (l) => logs.push(l));
    mgr.start({ address: 'prl1pabc' });

    child.stderr.emit('data', 'died mid-sentence');
    child.emit('exit', 1);
    expect(logs.map((l) => l.line)).toEqual(['died mid-sentence']);

    // A restart must not inherit the previous run's leftovers.
    mgr.start({ address: 'prl1pabc' });
    expect(mgr.lineBuf).toBe('');
  });

  test('a runaway line with no newline is flushed at the cap', () => {
    const child = makeChild();
    const mgr = new MinerManager({ spawn: () => child });
    const logs = [];
    mgr.on('log', (l) => logs.push(l));
    mgr.start({ address: 'prl1pabc' });

    child.stderr.emit('data', 'x'.repeat(64 * 1024 + 1));
    expect(logs).toHaveLength(1);
    expect(logs[0].line).toHaveLength(64 * 1024 + 1);
    expect(mgr.lineBuf).toBe(''); // buffer released, next chunk starts fresh
  });

  test('an exit with nothing buffered emits no stray log line', () => {
    const child = makeChild();
    const mgr = new MinerManager({ spawn: () => child });
    const logs = [];
    mgr.on('log', (l) => logs.push(l));
    mgr.start({ address: 'prl1pabc' });

    child.stderr.emit('data', 'complete\n');
    child.emit('exit', 0);
    expect(logs.map((l) => l.line)).toEqual(['complete']);
  });

  // Level comes from the line, not the pipe: everything arrives on stderr, so
  // marking the whole stream 'error' would paint the entire log red.
  test('severity is read from the line, not the stream', () => {
    const child = makeChild();
    const mgr = new MinerManager({ spawn: () => child });
    const logs = [];
    mgr.on('log', (l) => logs.push(l));
    mgr.start({ address: 'prl1pabc' });

    child.stderr.emit('data', '2026-08-23 23:39:42 ERROR failed to connect to h:1: boom\nroutine progress\n');
    expect(logs[0].level).toBe('error');
    expect(logs[1]).toEqual({ level: 'info', line: 'routine progress' });

    // stdout is still read, in case a future engine ever uses it.
    child.stdout.emit('data', 'from stdout\n');
    expect(logs[2]).toEqual({ level: 'info', line: 'from stdout' });
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
