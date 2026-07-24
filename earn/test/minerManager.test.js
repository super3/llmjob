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
      { type: 'status', gpuIndex: 0, hashrate: 286.86, accepted: 5, rejected: 0, power: 449, gpu: 'NVIDIA GeForce RTX 4090' },
    ]);
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

  test('pause/resume freeze and thaw the engine with SIGSTOP/SIGCONT (POSIX)', () => {
    const child = makeChild();
    const mgr = new MinerManager({ spawn: () => child });
    mgr.start({ address: 'prl1pabc', platform: 'linux' });

    expect(mgr.isPaused()).toBe(false);
    expect(mgr.pause()).toBe(true);
    expect(child.kill).toHaveBeenCalledWith('SIGSTOP');
    expect(mgr.isPaused()).toBe(true);
    expect(mgr.pause()).toBe(false); // already paused — no double signal
    expect(child.kill).toHaveBeenCalledTimes(1);

    expect(mgr.resume()).toBe(true);
    expect(child.kill).toHaveBeenCalledWith('SIGCONT');
    expect(mgr.isPaused()).toBe(false);
    expect(mgr.resume()).toBe(false); // already running
  });

  test('pause is a no-op with nothing running and on Windows (no SIGSTOP)', () => {
    const idle = new MinerManager({ spawn: () => makeChild() });
    expect(idle.pause()).toBe(false); // not started

    const child = makeChild();
    const win = new MinerManager({ spawn: () => child });
    win.start({ address: 'prl1pabc', platform: 'win32' });
    expect(win.pause()).toBe(false);
    expect(child.kill).not.toHaveBeenCalled();
    expect(win.isPaused()).toBe(false);
  });

  test('a failing SIGSTOP leaves the miner running; a failing SIGCONT still clears the flag', () => {
    const child = makeChild();
    child.kill.mockImplementationOnce(() => { throw new Error('ESRCH'); }); // first SIGSTOP throws
    const mgr = new MinerManager({ spawn: () => child });
    mgr.start({ address: 'prl1pabc', platform: 'linux' });
    expect(mgr.pause()).toBe(false);   // signal failed → not paused
    expect(mgr.isPaused()).toBe(false);

    expect(mgr.pause()).toBe(true);    // second attempt succeeds
    child.kill.mockImplementationOnce(() => { throw new Error('ESRCH'); }); // SIGCONT throws
    expect(mgr.resume()).toBe(true);   // still returns true and clears paused
    expect(mgr.isPaused()).toBe(false);
  });

  test('exit clears the paused flag', () => {
    const child = makeChild();
    const mgr = new MinerManager({ spawn: () => child });
    mgr.start({ address: 'prl1pabc', platform: 'linux' });
    mgr.pause();
    expect(mgr.isPaused()).toBe(true);
    child.emit('exit', 0);
    expect(mgr.isPaused()).toBe(false);
  });
});
