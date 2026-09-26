'use strict';

// gpuClocks sets real clocks on a real card, so nothing here may reach the real
// nvidia-smi. child_process is mocked for the whole file: the tests that inject
// their own exec never touch it, and the ones that check the defaults get this
// fake rather than the driver.
jest.mock('child_process', () => ({ execFileSync: jest.fn() }));

const { execFileSync } = require('child_process');
const { lockMemoryClock, resetMemoryClock, TIMEOUT_MS } = require('../src/main/gpuClocks');

const ROOT = () => 0;
const USER = () => 1000;

// An error shaped like the one execFileSync throws for a non-zero exit.
function failed(over) {
  return Object.assign(new Error('Command failed'), { status: 1, stdout: '', stderr: '' }, over);
}

beforeEach(() => execFileSync.mockReset());

describe('gpuClocks — the command', () => {
  test('root locks the card directly, min and max the same', () => {
    const exec = jest.fn();
    expect(lockMemoryClock(0, 7001, { exec, getuid: ROOT })).toEqual({ ok: true, error: null });
    expect(exec).toHaveBeenCalledWith('nvidia-smi', ['-i', '0', '-lmc', '7001,7001'],
      expect.objectContaining({ timeout: TIMEOUT_MS }));
  });

  test('root resets the card directly', () => {
    const exec = jest.fn();
    expect(resetMemoryClock(3, { exec, getuid: ROOT })).toEqual({ ok: true, error: null });
    expect(exec).toHaveBeenCalledWith('nvidia-smi', ['-i', '3', '-rmc'], expect.any(Object));
  });

  // -n: never prompt. A miner under systemd has no terminal to type a password
  // into, and a sudo waiting for one would hold up the start until the timeout.
  test('anyone else goes through sudo -n', () => {
    const exec = jest.fn();
    lockMemoryClock(1, 7001, { exec, getuid: USER });
    expect(exec).toHaveBeenCalledWith('sudo', ['-n', 'nvidia-smi', '-i', '1', '-lmc', '7001,7001'],
      expect.any(Object));
    resetMemoryClock(1, { exec, getuid: USER });
    expect(exec).toHaveBeenLastCalledWith('sudo', ['-n', 'nvidia-smi', '-i', '1', '-rmc'],
      expect.any(Object));
  });

  // argv, with output captured rather than spilled onto the miner's own log,
  // and a bound on how long a wedged driver can hold up a start or a stop.
  test('runs without a shell, captures output, and times out in seconds', () => {
    const exec = jest.fn();
    lockMemoryClock(0, 7001, { exec, getuid: ROOT });
    const opts = exec.mock.calls[0][2];
    expect(opts.shell).toBeUndefined();
    expect(opts.stdio).toEqual(['ignore', 'pipe', 'pipe']);
    expect(TIMEOUT_MS).toBeGreaterThanOrEqual(1000);
    expect(TIMEOUT_MS).toBeLessThanOrEqual(10000);
  });
});

describe('gpuClocks — failure never throws', () => {
  test('sudo with no NOPASSWD rule reports what sudo said', () => {
    const exec = jest.fn(() => { throw failed({ stderr: 'sudo: a password is required\n' }); });
    expect(lockMemoryClock(0, 7001, { exec, getuid: USER }))
      .toEqual({ ok: false, error: 'sudo: a password is required' });
  });

  // nvidia-smi says why on stdout, over more than one line; the first is the reason.
  test('an nvidia-smi refusal reports its first line', () => {
    const exec = jest.fn(() => {
      throw failed({ stdout: 'Setting locked memory clocks is not supported for GPU 00000000:01:00.0.\r\nTreating as warning and moving on.\n' });
    });
    expect(resetMemoryClock(0, { exec, getuid: ROOT }))
      .toEqual({ ok: false, error: 'Setting locked memory clocks is not supported for GPU 00000000:01:00.0.' });
  });

  test('a spawn failure with no output reports the error message', () => {
    const exec = jest.fn(() => { throw failed({ message: 'spawnSync sudo ENOENT', stdout: undefined, stderr: undefined }); });
    expect(lockMemoryClock(0, 7001, { exec, getuid: USER }))
      .toEqual({ ok: false, error: 'spawnSync sudo ENOENT' });
  });

  test('whatever is thrown still comes back as an answer', () => {
    const exec = jest.fn(() => { throw 'killed'; });
    expect(lockMemoryClock(0, 7001, { exec, getuid: ROOT })).toEqual({ ok: false, error: 'killed' });
    exec.mockImplementation(() => { throw undefined; });
    expect(lockMemoryClock(0, 7001, { exec, getuid: ROOT })).toEqual({ ok: false, error: 'undefined' });
  });
});

describe('gpuClocks — defaults', () => {
  // The real thing is execFileSync -- synchronous on purpose, so a reset has
  // finished before the caller's next line runs. Mocked above.
  test('uses execFileSync and the process uid', () => {
    // Assigned, not jest.spyOn'd: Windows has no process.getuid to spy on,
    // and spyOn throws on a missing method before the test even starts.
    const real = process.getuid;
    const uid = jest.fn(() => 0);
    process.getuid = uid;
    try {
      expect(lockMemoryClock(0, 7001)).toEqual({ ok: true, error: null });
      expect(execFileSync).toHaveBeenCalledWith('nvidia-smi', ['-i', '0', '-lmc', '7001,7001'], expect.any(Object));
      uid.mockReturnValue(1000);
      resetMemoryClock(0);
      expect(execFileSync).toHaveBeenLastCalledWith('sudo', ['-n', 'nvidia-smi', '-i', '0', '-rmc'], expect.any(Object));
    } finally {
      process.getuid = real;
    }
  });

  // No uids at all (Windows): no sudo to go through either.
  test('a platform with no uids runs nvidia-smi directly', () => {
    const real = process.getuid;
    process.getuid = undefined;
    try {
      resetMemoryClock(0);
      expect(execFileSync).toHaveBeenCalledWith('nvidia-smi', ['-i', '0', '-rmc'], expect.any(Object));
    } finally {
      process.getuid = real;
    }
  });
});
