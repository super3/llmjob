'use strict';

const fs = require('fs');
const path = require('path');
const { EventEmitter } = require('events');
const {
  POOL_THREADS, widenThreadPool, needsRelaunch, relaunchEnv, relaunchApp, relaunchCli,
} = require('../src/shared/threadPool');

const SRC = path.join(__dirname, '..', 'src');
const read = (...p) => fs.readFileSync(path.join(SRC, ...p), 'utf8');

describe('widenThreadPool', () => {
  test('sets a pool big enough for a scan a card on a large rig when nobody chose a size', () => {
    expect(POOL_THREADS).toBeGreaterThanOrEqual(16);
    const env = {};
    expect(widenThreadPool(env)).toBe(true);
    expect(env.UV_THREADPOOL_SIZE).toBe(String(POOL_THREADS));
    const blank = { UV_THREADPOOL_SIZE: ' ' };
    expect(widenThreadPool(blank)).toBe(true);
    expect(blank.UV_THREADPOOL_SIZE).toBe(String(POOL_THREADS));
  });

  test('keeps a size the user chose', () => {
    const env = { UV_THREADPOOL_SIZE: '8' };
    expect(widenThreadPool(env)).toBe(false);
    expect(env.UV_THREADPOOL_SIZE).toBe('8');
  });

  // libuv reads the variable when the pool is first used, so each shell calls this before its other requires.
  test('both shells call it first, before any module that could use the pool', () => {
    for (const [file, call] of [
      [['cli', 'earn-cli.js'], "require('../shared/threadPool');"],
      [['main', 'main.js'], "require('../shared/threadPool').relaunchApp(process, require('child_process').spawn);"],
    ]) {
      const text = read(...file);
      expect([file.join('/'), text.indexOf(call) > 0]).toEqual([file.join('/'), true]);
      expect([file.join('/'), text.indexOf('require(') === text.indexOf("require('../shared/threadPool')")]).toEqual([file.join('/'), true]);
    }
  });
});

// Set from inside the process, the variable does nothing on Windows
// (scripts/check-thread-pool.js): the shell has to start again with it.
describe('starting again on Windows', () => {
  const proc = (over) => Object.assign(new EventEmitter(), {
    platform: 'win32',
    env: { PATH: 'x' },
    versions: { electron: '33.2.0' },
    execPath: 'C:\\Earn\\LLMJob Earn.exe',
    execArgv: [],
    argv: ['C:\\Earn\\LLMJob Earn.exe', '--hidden'],
    exit: jest.fn(),
  }, over);

  test('needsRelaunch: on Windows, when nobody chose a size', () => {
    expect(needsRelaunch(proc())).toBe(true);
    expect(needsRelaunch(proc({ env: { UV_THREADPOOL_SIZE: '8' } }))).toBe(false);
    expect(needsRelaunch(proc({ platform: 'linux' }))).toBe(false);
  });

  test('relaunchEnv adds the pool to a copy of the environment', () => {
    const env = { PATH: 'x' };
    expect(relaunchEnv(env)).toEqual({ PATH: 'x', UV_THREADPOOL_SIZE: String(POOL_THREADS) });
    expect(env).toEqual({ PATH: 'x' });
  });

  test('the app starts itself again, detached, with its arguments and the pool, and exits', () => {
    const p = proc();
    const child = { unref: jest.fn() };
    const spawn = jest.fn(() => child);
    expect(relaunchApp(p, spawn)).toBe(true);
    expect(spawn).toHaveBeenCalledWith('C:\\Earn\\LLMJob Earn.exe', ['--hidden'],
      { env: { PATH: 'x', UV_THREADPOOL_SIZE: String(POOL_THREADS) }, detached: true, stdio: 'ignore' });
    expect(child.unref).toHaveBeenCalled();
    expect(p.exit).toHaveBeenCalledWith(0);
  });

  test('the app sets the pool in place off Windows, outside Electron, or when a size was chosen', () => {
    for (const over of [{ platform: 'linux' }, { versions: {} }, { env: { UV_THREADPOOL_SIZE: '8' } }]) {
      const p = proc(over);
      const spawn = jest.fn();
      expect(relaunchApp(p, spawn)).toBe(false);
      expect(spawn).not.toHaveBeenCalled();
      expect(p.exit).not.toHaveBeenCalled();
      expect(p.env.UV_THREADPOOL_SIZE).toBe(over.env ? '8' : String(POOL_THREADS));
    }
  });

  test('an app that cannot start again carries on in this process', () => {
    const p = proc();
    expect(relaunchApp(p, () => { throw new Error('EPERM'); })).toBe(false);
    expect(p.exit).not.toHaveBeenCalled();
    expect(p.env.UV_THREADPOOL_SIZE).toBe(String(POOL_THREADS));
  });

  const cliProc = () => proc({
    versions: {}, execPath: 'C:\\node\\node.exe', execArgv: ['--max-old-space-size=4096'],
    argv: ['C:\\node\\node.exe', 'earn-cli.js', '--coin', 'quantus'],
  });

  test('the CLI runs again in this console with the pool, ignores Ctrl+C itself, and ends with the child\'s code', async () => {
    const p = cliProc();
    const child = new EventEmitter();
    const spawn = jest.fn(() => child);
    const done = relaunchCli(p, spawn);
    expect(spawn).toHaveBeenCalledWith('C:\\node\\node.exe', ['--max-old-space-size=4096', 'earn-cli.js', '--coin', 'quantus'],
      { env: { PATH: 'x', UV_THREADPOOL_SIZE: String(POOL_THREADS) }, stdio: 'inherit' });
    expect(p.listenerCount('SIGINT')).toBe(1);
    expect(p.listenerCount('SIGTERM')).toBe(1);
    p.emit('SIGINT');
    child.emit('exit', 3);
    await expect(done).resolves.toBe(3);
  });

  test('a CLI child killed by a signal ends with 1', async () => {
    const child = new EventEmitter();
    const done = relaunchCli(cliProc(), () => child);
    child.emit('exit', null, 'SIGKILL');
    await expect(done).resolves.toBe(1);
  });

  test('a CLI child that cannot start resolves null, and the caller runs in this process', async () => {
    await expect(relaunchCli(cliProc(), () => { throw new Error('EPERM'); })).resolves.toBe(null);
    const child = new EventEmitter();
    const done = relaunchCli(cliProc(), () => child);
    child.emit('error', new Error('ENOENT'));
    await expect(done).resolves.toBe(null);
  });
});
