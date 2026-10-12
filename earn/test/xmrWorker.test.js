'use strict';

const path = require('path');
const { FILES, workerFile, findWorker, makeSpawnWorker } = require('../src/main/xmrWorker');

const EARN = path.join(__dirname, '..');

describe('workerFile', () => {
  test('one file per platform that mines, none for the rest', () => {
    expect(FILES).toEqual({ win32: 'llmjob-xmr-worker.exe', linux: 'llmjob-xmr-worker' });
    expect(workerFile('linux')).toBe('llmjob-xmr-worker');
    expect(workerFile('win32')).toBe('llmjob-xmr-worker.exe');
    expect(workerFile('darwin')).toBeNull();
  });
});

describe('findWorker', () => {
  const only = (file) => (p) => p === file;

  test('LLMJOB_XMR_WORKER first', () => {
    const exists = (p) => p === '/opt/w' || p.endsWith('llmjob-xmr-worker');
    expect(findWorker({ env: { LLMJOB_XMR_WORKER: '/opt/w' }, platform: 'linux', execPath: '/x/cli', exists })).toBe('/opt/w');
  });

  test('the installer\'s resources/native, then beside the executable, then native/ beside it', () => {
    const res = path.join('/app/resources', 'native', 'llmjob-xmr-worker.exe');
    expect(findWorker({ resourcesPath: '/app/resources', platform: 'win32', env: {}, execPath: '/app/x.exe', exists: only(res) })).toBe(res);
    const beside = path.join('/rig', 'llmjob-xmr-worker');
    expect(findWorker({ platform: 'linux', env: {}, execPath: '/rig/llmjob-earn-cli-linux', exists: only(beside) })).toBe(beside);
    const nested = path.join('/rig', 'native', 'llmjob-xmr-worker');
    expect(findWorker({ platform: 'linux', env: {}, execPath: '/rig/cli', exists: only(nested) })).toBe(nested);
  });

  test('a source checkout finds its own build, or what --stage put in vendor/native', () => {
    for (const rel of [['xmr-worker', 'build'], ['xmr-worker', 'build', 'Release'], ['vendor', 'native']]) {
      const file = path.join(EARN, ...rel, 'llmjob-xmr-worker');
      expect(findWorker({ platform: 'linux', env: {}, execPath: '/usr/bin/node', exists: only(file) })).toBe(file);
    }
  });

  test('null when there is none, or none for this platform', () => {
    expect(findWorker({ platform: 'linux', env: {}, execPath: '/x', exists: () => false })).toBeNull();
    expect(findWorker({ platform: 'darwin', env: {}, exists: () => true })).toBeNull();
  });

  test('defaults to this process\'s platform, environment and executable', () => {
    const seen = [];
    findWorker({ exists: (p) => { seen.push(p); return false; } });
    if (workerFile(process.platform)) {
      expect(seen).toContain(path.join(path.dirname(process.execPath), workerFile(process.platform)));
    } else {
      expect(seen).toEqual([]);
    }
  });
});

describe('makeSpawnWorker', () => {
  test('pipes all three streams, hides the window, and keeps Ctrl+C from the worker on Linux', () => {
    const spawn = jest.fn(() => 'child');
    const pipes = { stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true };
    expect(makeSpawnWorker(spawn, 'linux')('/w', ['--threads', '2'])).toBe('child');
    expect(spawn).toHaveBeenLastCalledWith('/w', ['--threads', '2'], Object.assign({ detached: true }, pipes));
    // On Windows detached would give the worker a console of its own.
    makeSpawnWorker(spawn, 'win32')('w.exe', []);
    expect(spawn).toHaveBeenLastCalledWith('w.exe', [], Object.assign({ detached: false }, pipes));
    // This platform when none is named.
    makeSpawnWorker(spawn)('/w', []);
    expect(spawn.mock.calls[2][2].detached).toBe(process.platform !== 'win32');
  });
});
