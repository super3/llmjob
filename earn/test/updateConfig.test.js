'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { UPDATE_CONFIG, usableUpdateConfig, repairUpdateConfig } = require('../src/shared/updateConfig');

// The real reader, so these tests read the file the way the app does. Given an
// app of our own, electron-updater never touches Electron.
const { NsisUpdater } = require('electron-updater/out/NsisUpdater');
function updaterReading(file, dir) {
  const app = {
    version: '0.0.1', isPackaged: true, name: 'LLMJob Earn', userDataPath: dir, baseCachePath: dir,
    appUpdateConfigPath: file, whenReady: () => Promise.resolve(), onQuit: () => {},
  };
  const u = new NsisUpdater(null, app);
  u.logger = null;
  return u;
}

let dir;
beforeEach(() => { dir = fs.mkdtempSync(path.join(os.tmpdir(), 'update-config-')); });
afterEach(() => { fs.rmSync(dir, { recursive: true, force: true }); });

describe('the copy of app-update.yml', () => {
  it('says what electron-builder writes from package.json', async () => {
    const pkg = require('../package.json');
    const file = path.join(dir, 'app-update.yml');
    fs.writeFileSync(file, UPDATE_CONFIG);
    // updaterCacheDirName is the package name, lowercased, plus "-updater". The
    // installer keeps its own copy in that folder too, so it has to match.
    expect(await updaterReading(file, dir).configOnDisk.value)
      .toEqual({ ...pkg.build.publish[0], updaterCacheDirName: pkg.name.toLowerCase() + '-updater' });
  });

  // electron-builder adds publisherName only for a signed Windows build, and
  // electron-updater then checks each download's signature against it. The copy
  // has none, which is right only while the build is unsigned.
  it('has no publisherName, because the Windows build is not signed', async () => {
    const win = require('../package.json').build.win;
    for (const key of ['publisherName', 'signtoolOptions', 'azureSignOptions', 'certificateFile', 'certificateSubjectName']) {
      expect(win[key]).toBeUndefined();
    }
    const wf = fs.readFileSync(path.join(__dirname, '..', '..', '.github', 'workflows', 'miner-build.yml'), 'utf8');
    expect(wf).not.toMatch(/CSC_LINK/);
    const file = path.join(dir, 'app-update.yml');
    fs.writeFileSync(file, UPDATE_CONFIG);
    expect(await updaterReading(file, dir).verifySignature('setup.exe')).toBeNull();
  });
});

describe('usableUpdateConfig', () => {
  it('accepts what the build writes, with either line ending', () => {
    expect(usableUpdateConfig(UPDATE_CONFIG)).toBe(true);
    expect(usableUpdateConfig(UPDATE_CONFIG.replace(/\n/g, '\r\n'))).toBe(true);
  });

  it('rejects nothing, NUL bytes, and no provider', () => {
    expect(usableUpdateConfig(null)).toBe(false);
    expect(usableUpdateConfig('')).toBe(false);
    expect(usableUpdateConfig('\0'.repeat(96))).toBe(false);
    expect(usableUpdateConfig(UPDATE_CONFIG + '\0\0')).toBe(false);
    expect(usableUpdateConfig('owner: super3\nrepo: llmjob\n')).toBe(false);
    expect(usableUpdateConfig('provider:\n')).toBe(false);
  });
});

describe('repairUpdateConfig', () => {
  const join = path.join;
  const res = () => path.join(dir, 'resources');
  const data = () => path.join(dir, 'userData');
  beforeEach(() => {
    fs.mkdirSync(res());
    fs.mkdirSync(data());
  });
  const repair = () => repairUpdateConfig({ resourcesPath: res(), userDataPath: data(), fs, join });

  it('leaves a good file alone', () => {
    fs.writeFileSync(path.join(res(), 'app-update.yml'), UPDATE_CONFIG);
    expect(repair()).toBeNull();
    expect(fs.readdirSync(data())).toEqual([]);
  });

  // What one Windows install had after an update was cut off: the right size,
  // all zeros. electron-updater's error is the one from that user's log.
  it('writes a good copy for a file of NUL bytes, which electron-updater cannot read', async () => {
    const bad = path.join(res(), 'app-update.yml');
    fs.writeFileSync(bad, Buffer.alloc(96));
    await expect(updaterReading(bad, dir).configOnDisk.value).rejects.toThrow(/^null byte is not allowed in input \(1:1\)/);

    const file = repair();
    expect(file).toBe(path.join(data(), 'app-update.yml'));
    expect(fs.readFileSync(file, 'utf8')).toBe(UPDATE_CONFIG);
    const u = updaterReading(bad, dir);
    u.updateConfigPath = file;
    expect(await u.configOnDisk.value).toMatchObject({ provider: 'github', owner: 'super3', repo: 'llmjob' });
  });

  it('writes one for a missing file too', () => {
    expect(repair()).toBe(path.join(data(), 'app-update.yml'));
  });

  it('does nothing without a resources folder (not running under Electron)', () => {
    expect(repairUpdateConfig({ resourcesPath: undefined, userDataPath: data(), fs, join })).toBeNull();
    expect(fs.readdirSync(data())).toEqual([]);
  });

  it('throws when the copy cannot be written', () => {
    expect(() => repairUpdateConfig({ resourcesPath: res(), userDataPath: path.join(dir, 'gone'), fs, join }))
      .toThrow(/ENOENT/);
  });
});
