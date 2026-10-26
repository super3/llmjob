'use strict';

// The CLI's HiveOS IO (src/cli/hiveos.js) against real folders in a temp dir:
// the package swap, the trial and the dashboard posts. The download, tar, the
// new CLI and HiveOS's `message` are stand-ins.

jest.mock('child_process', () => ({ spawn: jest.fn(), spawnSync: jest.fn(), execFile: jest.fn() }));
jest.mock('../src/main/io', () => ({ downloadFile: jest.fn() }));

const fs = require('fs');
const os = require('os');
const path = require('path');
const { EventEmitter } = require('events');
const { spawn, spawnSync, execFile } = require('child_process');
const { downloadFile } = require('../src/main/io');
const hive = require('../src/cli/hiveos');

let tmp;
let dir;
let conf;
beforeEach(() => {
  jest.resetAllMocks();
  downloadFile.mockResolvedValue(undefined);
  tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'hivecli-'));
  dir = path.join(tmp, 'llmjob-earn');
  fs.mkdirSync(dir);
  conf = path.join(dir, 'llmjob-earn.conf');
  fs.writeFileSync(path.join(dir, 'h-manifest.conf'),
    'CUSTOM_NAME=llmjob-earn\nCUSTOM_VERSION=0.6.9\nCUSTOM_CONFIG_FILENAME=' + conf + '\n');
  fs.writeFileSync(conf, '--address prl1pabc --stats-file /run/hive/llmjob-earn-stats.json\n');
  fs.writeFileSync(path.join(dir, 'llmjob-earn-cli-linux'), 'old binary');
});
afterEach(() => {
  fs.rmSync(tmp, { recursive: true, force: true });
});

const read = (f) => fs.readFileSync(f, 'utf8');
const PLAN = { currentVersion: '0.6.9', latestVersion: '0.7.0', packageUrl: 'https://dl/llmjob-earn-0.7.0.tar.gz' };
const EXEC = () => path.join(dir, 'llmjob-earn-cli-linux');

// tar unpacks a package of `version` into the folder after -C; the new CLI says
// `said` to --version.
function stubPackage({ version = '0.7.0', said = '0.7.0\n', tar } = {}) {
  spawnSync.mockImplementation((cmd, args) => {
    if (cmd === 'tar') {
      if (tar) return tar;
      const into = path.join(args[args.indexOf('-C') + 1], 'llmjob-earn');
      fs.mkdirSync(into, { recursive: true });
      if (version) fs.writeFileSync(path.join(into, 'h-manifest.conf'), 'CUSTOM_VERSION=' + version + '\n');
      fs.writeFileSync(path.join(into, 'llmjob-earn-cli-linux'), 'new binary');
      return { status: 0, stdout: '', stderr: '' };
    }
    return typeof said === 'object' ? said : { status: 0, stdout: said, stderr: '' };
  });
}

describe('isHiveos and packageDir', () => {
  test('LLMJOB_HIVEOS=1 is HiveOS, anything else is not', () => {
    expect(hive.isHiveos({ LLMJOB_HIVEOS: '1' })).toBe(true);
    expect(hive.isHiveos({ LLMJOB_HIVEOS: '0' })).toBe(false);
    expect(hive.isHiveos({})).toBe(false);
    expect(hive.isHiveos()).toBe(process.env.LLMJOB_HIVEOS === '1');
  });

  test('the package is the folder the binary is in', () => {
    expect(hive.packageDir('/hive/miners/custom/llmjob-earn/llmjob-earn-cli-linux')).toBe(path.dirname('/hive/miners/custom/llmjob-earn/llmjob-earn-cli-linux'));
    expect(hive.packageDir()).toBe(path.dirname(process.execPath));
  });
});

describe('postMessage', () => {
  let bin;
  let state;
  let child;
  beforeEach(() => {
    bin = path.join(tmp, 'message');
    state = path.join(tmp, 'posted.json');
    fs.writeFileSync(bin, '');
    child = new EventEmitter();
    child.unref = jest.fn();
    spawn.mockReturnValue(child);
  });
  const post = (text, now) => hive.postMessage('warning', text, { messageBin: bin, statePath: state, now: () => now });

  test('runs HiveOS\'s message command, detached, and remembers the post', () => {
    expect(post('hello', 1000)).toBe(true);
    expect(spawn).toHaveBeenCalledWith(bin, ['warning', 'hello'], { detached: true, stdio: 'ignore' });
    expect(child.unref).toHaveBeenCalled();
    expect(JSON.parse(read(state))).toEqual({ hello: 1000 });
    // A failed message command is not this process's error.
    expect(() => child.emit('error', new Error('ENOENT'))).not.toThrow();
  });

  test('the same text waits an hour; another goes out at once', () => {
    post('hello', 1000);
    expect(post('hello', 1000 + 60 * 60 * 1000 - 1)).toBe(false);
    expect(post('other', 2000)).toBe(true);
    expect(post('hello', 1000 + 60 * 60 * 1000)).toBe(true);
    expect(spawn).toHaveBeenCalledTimes(3);
  });

  test('a record it cannot read or write does not stop the post', () => {
    fs.writeFileSync(state, 'not json');
    expect(post('hello', 1000)).toBe(true);
    const nowhere = path.join(tmp, 'missing', 'posted.json');
    expect(hive.postMessage('info', 'x', { messageBin: bin, statePath: nowhere, now: () => 1 })).toBe(true);
    fs.writeFileSync(state, 'null');
    expect(post('again', 1000)).toBe(true);
  });

  test('a message command that will not spawn is a false, not a throw', () => {
    spawn.mockImplementation(() => { throw new Error('EACCES'); });
    expect(post('hello', 1000)).toBe(false);
  });

  test('off HiveOS there is no message command, and nothing is posted', () => {
    expect(hive.postMessage('info', 'x', { messageBin: path.join(tmp, 'nope') })).toBe(false);
    // The real paths: no /hive/bin/message on a test machine.
    expect(hive.postMessage('info', 'x')).toBe(false);
    expect(spawn).not.toHaveBeenCalled();
  });
});

describe('trial and skip files', () => {
  test('readTrial reads the trial, or null without one', () => {
    expect(hive.readTrial(dir)).toBeNull();
    fs.writeFileSync(path.join(dir, '.update-trial'), 'UPDATE_FROM=0.6.9\nUPDATE_TO=0.7.0\n');
    expect(hive.readTrial(dir)).toEqual({ from: '0.6.9', to: '0.7.0' });
  });

  test('isSkipped reads the skip h-run.sh wrote', () => {
    expect(hive.isSkipped(dir, '0.7.0', 0)).toBe(false);
    fs.writeFileSync(path.join(dir, '.update-skip'), 'SKIP_VERSION=0.7.0\nSKIP_UNTIL=100\n');
    expect(hive.isSkipped(dir, '0.7.0', 99999)).toBe(true);
    expect(hive.isSkipped(dir, '0.7.0', 100000)).toBe(false);
  });

  test('confirmTrial drops the trial and the old folder', () => {
    fs.writeFileSync(path.join(dir, '.update-trial'), 'UPDATE_TO=0.7.0\n');
    fs.mkdirSync(dir + '.prev');
    fs.writeFileSync(path.join(dir + '.prev', 'llmjob-earn-cli-linux'), 'old');
    hive.confirmTrial(dir);
    expect(fs.existsSync(path.join(dir, '.update-trial'))).toBe(false);
    expect(fs.existsSync(dir + '.prev')).toBe(false);
    expect(fs.existsSync(dir)).toBe(true);
    // Twice is fine.
    expect(() => hive.confirmTrial(dir)).not.toThrow();
  });
});

describe('cleanStaging', () => {
  test('removes what an update killed half way left, and nothing else', () => {
    fs.mkdirSync(path.join(tmp, '.llmjob-earn-update-123'));
    fs.writeFileSync(path.join(tmp, '.llmjob-earn-update-123.tar.gz'), 'partial');
    fs.mkdirSync(path.join(tmp, 'other-miner'));
    hive.cleanStaging(dir);
    expect(fs.readdirSync(tmp).sort()).toEqual(['llmjob-earn', 'other-miner']);
  });

  test('a parent it cannot list is left alone', () => {
    expect(() => hive.cleanStaging(path.join(tmp, 'missing', 'llmjob-earn'))).not.toThrow();
  });
});

describe('stagePackage', () => {
  const stage = () => path.join(tmp, '.llmjob-earn-update-42');
  const opts = () => ({ pid: 42, execPath: EXEC(), now: () => 1700000000000 });

  test('unpacks beside the package, checks it, carries the config and starts a trial', async () => {
    stubPackage();
    const fresh = await hive.stagePackage(PLAN, dir, opts());
    expect(fresh).toBe(path.join(stage(), 'llmjob-earn'));
    expect(downloadFile).toHaveBeenCalledWith(PLAN.packageUrl, stage() + '.tar.gz');
    expect(spawnSync).toHaveBeenCalledWith('tar', ['-xzf', stage() + '.tar.gz', '-C', stage()], expect.any(Object));
    expect(spawnSync).toHaveBeenCalledWith(path.join(fresh, 'llmjob-earn-cli-linux'), ['--version'], expect.any(Object));
    expect(read(path.join(fresh, 'llmjob-earn.conf'))).toBe(read(conf));
    expect(read(path.join(fresh, '.update-trial'))).toBe('UPDATE_FROM=0.6.9\nUPDATE_TO=0.7.0\nUPDATE_AT=1700000000\n');
    // The download goes; the installed package is untouched.
    expect(fs.existsSync(stage() + '.tar.gz')).toBe(false);
    expect(read(EXEC())).toBe('old binary');
  });

  test('a config outside the package, or none, is not carried', async () => {
    stubPackage();
    const outside = path.join(tmp, 'elsewhere.conf');
    fs.writeFileSync(outside, 'x');
    fs.writeFileSync(path.join(dir, 'h-manifest.conf'), 'CUSTOM_CONFIG_FILENAME=' + outside + '\n');
    let fresh = await hive.stagePackage(PLAN, dir, opts());
    expect(fs.existsSync(path.join(fresh, 'elsewhere.conf'))).toBe(false);

    fs.rmSync(stage(), { recursive: true });
    fs.writeFileSync(path.join(dir, 'h-manifest.conf'), 'CUSTOM_CONFIG_FILENAME=' + path.join(dir, 'gone.conf') + '\n');
    fresh = await hive.stagePackage(PLAN, dir, opts());
    expect(fs.existsSync(path.join(fresh, 'gone.conf'))).toBe(false);

    fs.rmSync(stage(), { recursive: true });
    fs.writeFileSync(path.join(dir, 'h-manifest.conf'), 'CUSTOM_NAME=llmjob-earn\n');
    fresh = await hive.stagePackage(PLAN, dir, opts());
    expect(fs.readdirSync(fresh).sort()).toEqual(['.update-trial', 'h-manifest.conf', 'llmjob-earn-cli-linux']);
  });

  // Each failure throws and leaves nothing behind.
  async function fails(message) {
    await expect(hive.stagePackage(PLAN, dir, opts())).rejects.toThrow(message);
    expect(fs.readdirSync(tmp)).toEqual(['llmjob-earn']);
    expect(read(EXEC())).toBe('old binary');
  }

  test('a failed download', async () => {
    downloadFile.mockRejectedValue(new Error('HTTP 404'));
    await fails('HTTP 404');
  });

  test('a package tar cannot unpack', async () => {
    stubPackage({ tar: { status: 2, stdout: '', stderr: 'gzip: stdin: not in gzip format\n' } });
    await fails('could not unpack it (gzip: stdin: not in gzip format)');
    stubPackage({ tar: { status: 2, stdout: '', stderr: '' } });
    await fails('could not unpack it (tar exited 2)');
    stubPackage({ tar: { error: new Error('spawnSync tar ENOENT'), status: null } });
    await fails('could not unpack it (spawnSync tar ENOENT)');
  });

  test('a package of another version, or of none', async () => {
    stubPackage({ version: '0.6.0' });
    await fails('it holds version 0.6.0, not 0.7.0');
    stubPackage({ version: null });
    await fails('it holds version none, not 0.7.0');
  });

  test('a new CLI that does not run here', async () => {
    stubPackage({ said: { error: new Error('spawnSync EACCES'), status: null } });
    await fails('its CLI did not run here (spawnSync EACCES)');
    stubPackage({ said: { status: 1, stdout: '', stderr: "version `GLIBC_2.99' not found\n" } });
    await fails("its CLI did not run here (version `GLIBC_2.99' not found)");
    stubPackage({ said: '0.6.9\n' });
    await fails('its CLI did not run here (0.6.9)');
    stubPackage({ said: { status: 139, stdout: '', stderr: '' } });
    await fails('its CLI did not run here (exit 139)');
    stubPackage({ said: { status: 0, stdout: null, stderr: null } });
    await fails('its CLI did not run here (exit 0)');
  });

  test('without options it uses this process and the clock', async () => {
    stubPackage();
    const fresh = await hive.stagePackage(PLAN, dir);
    expect(fresh).toBe(path.join(tmp, '.llmjob-earn-update-' + process.pid, 'llmjob-earn'));
    expect(read(path.join(fresh, '.update-trial'))).toMatch(/^UPDATE_FROM=0\.6\.9\nUPDATE_TO=0\.7\.0\nUPDATE_AT=\d+\n$/);
  });
});

describe('swapIn and updatePackage', () => {
  test('the new folder takes the package\'s place, and the old one waits as .prev', () => {
    const fresh = path.join(tmp, 'fresh');
    fs.mkdirSync(fresh);
    fs.writeFileSync(path.join(fresh, 'llmjob-earn-cli-linux'), 'new binary');
    fs.mkdirSync(dir + '.prev');
    fs.writeFileSync(path.join(dir + '.prev', 'stale'), 'from an earlier update');
    hive.swapIn(fresh, dir);
    expect(read(EXEC())).toBe('new binary');
    expect(read(path.join(dir + '.prev', 'llmjob-earn-cli-linux'))).toBe('old binary');
    expect(fs.existsSync(path.join(dir + '.prev', 'stale'))).toBe(false);
    expect(fs.existsSync(fresh)).toBe(false);
  });

  test('a swap that fails half way puts the old folder back', () => {
    expect(() => hive.swapIn(path.join(tmp, 'missing'), dir)).toThrow();
    expect(read(EXEC())).toBe('old binary');
    expect(fs.existsSync(dir + '.prev')).toBe(false);
  });

  test('updatePackage stages, swaps and tidies up', async () => {
    stubPackage();
    await hive.updatePackage(PLAN, { pid: 7, execPath: EXEC() });
    expect(read(EXEC())).toBe('new binary');
    expect(read(path.join(dir, '.update-trial'))).toContain('UPDATE_TO=0.7.0');
    expect(read(path.join(dir, 'llmjob-earn.conf'))).toContain('--address prl1pabc');
    expect(read(path.join(dir + '.prev', 'llmjob-earn-cli-linux'))).toBe('old binary');
    expect(fs.readdirSync(tmp).sort()).toEqual(['llmjob-earn', 'llmjob-earn.prev']);
  });

  test('updatePackage tidies up after a failed swap too', async () => {
    stubPackage();
    const rename = fs.renameSync;
    let calls = 0;
    const spy = jest.spyOn(fs, 'renameSync').mockImplementation((a, b) => {
      calls++;
      if (calls === 2) throw new Error('EXDEV');
      return rename(a, b);
    });
    await expect(hive.updatePackage(PLAN, { pid: 8, execPath: EXEC() })).rejects.toThrow('EXDEV');
    spy.mockRestore();
    expect(read(EXEC())).toBe('old binary');
    expect(fs.readdirSync(tmp)).toEqual(['llmjob-earn']);
  });

  test('updatePackage with no options updates the folder this process runs from', async () => {
    stubPackage();
    // Stopped before it touches anything: this is the test runner's own folder.
    const rm = jest.spyOn(fs, 'rmSync').mockImplementation(() => {});
    const mkdir = jest.spyOn(fs, 'mkdirSync').mockImplementation(() => { throw new Error('stop here'); });
    await expect(hive.updatePackage(PLAN)).rejects.toThrow('stop here');
    expect(rm).toHaveBeenCalledWith(path.join(path.dirname(path.dirname(process.execPath)),
      '.' + path.basename(path.dirname(process.execPath)) + '-update-' + process.pid), { recursive: true, force: true });
    rm.mockRestore();
    mkdir.mockRestore();
    expect(downloadFile).not.toHaveBeenCalled();
  });
});

describe('checkDriver', () => {
  test('a Blackwell card on an old driver gets a message', async () => {
    const proc = path.join(tmp, 'version');
    fs.writeFileSync(proc, 'NVRM version: NVIDIA UNIX x86_64 Kernel Module  550.127.05  Tue\n');
    execFile.mockImplementation((cmd, args, opts, cb) => cb(null, '02:00.0 0300: 10de:2b85 (rev a1)\n'));
    await expect(hive.checkDriver({ versionFile: proc })).resolves.toContain('needs NVIDIA driver 570 or newer, and this rig has 550');
    expect(execFile).toHaveBeenCalledWith('lspci', ['-n'], { timeout: 10000 }, expect.any(Function));
  });

  test('no lspci, or no such card, is no message', async () => {
    execFile.mockImplementation((cmd, args, opts, cb) => cb(new Error('ENOENT')));
    await expect(hive.checkDriver()).resolves.toBeNull();
    execFile.mockImplementation((cmd, args, opts, cb) => cb(null, '01:00.0 0300: 10de:2684\n'));
    await expect(hive.checkDriver()).resolves.toBeNull();
  });
});
