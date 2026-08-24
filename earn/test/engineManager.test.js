'use strict';

const path = require('path');
const { EngineManager } = require('../src/main/engineManager');
const { ENGINE, enginePackage, engineBinaryName } = require('../src/shared/engine');

const WIN = ENGINE.windows;
const LIN = ENGINE.linux;
const winPkg = enginePackage('win32', WIN);
const linPkg = enginePackage('linux', LIN);

function makeFs(installed) {
  return {
    existsSync: jest.fn(() => installed),
    mkdirSync: jest.fn(),
    unlinkSync: jest.fn(),
    renameSync: jest.fn(),
  };
}

function make(platform, version, installed, over = {}) {
  const fs = makeFs(installed);
  const download = jest.fn(() => Promise.resolve());
  const extract = jest.fn(() => Promise.resolve());
  const extractPackage = jest.fn(() => Promise.resolve());
  const chmod = jest.fn();
  const mgr = new EngineManager({
    dir: '/cache', platform, version, fs, download, extract, extractPackage, chmod, ...over,
  });
  return { mgr, fs: over.fs || fs, download, extract, extractPackage, chmod, ...over };
}

describe('EngineManager', () => {
  test('constructs with no arguments', () => {
    expect(new EngineManager()).toBeInstanceOf(EngineManager);
  });

  test('binaryPath is the versioned cache name', () => {
    const { mgr } = make('win32', WIN, false);
    expect(mgr.binaryPath()).toBe(path.join('/cache', engineBinaryName('win32', null, WIN)));
  });
});

describe('EngineManager — already installed', () => {
  test('returns the cached path without downloading', async () => {
    const { mgr, fs, download } = make('win32', WIN, true);
    await expect(mgr.ensure()).resolves.toBe(path.join('/cache', winPkg.launcher));
    expect(mgr.isInstalled()).toBe(true);
    expect(download).not.toHaveBeenCalled();
    expect(fs.mkdirSync).not.toHaveBeenCalled();
  });

  // A cached binary can be present but lack +x: the download writes 0o644 and
  // chmods afterwards, so a crash in between leaves a file that spawns EACCES
  // forever with nothing to re-trigger the install.
  test('re-asserts the execute bit off Windows', async () => {
    const { mgr, chmod } = make('linux', LIN, true);
    const dest = path.join('/cache', linPkg.launcher);
    await expect(mgr.ensure()).resolves.toBe(dest);
    expect(chmod).toHaveBeenCalledWith(dest, 0o755);
  });

  test('never chmods on Windows', async () => {
    const { mgr, chmod } = make('win32', WIN, true);
    await mgr.ensure();
    expect(chmod).not.toHaveBeenCalled();
  });

  // A chmod failure on an already-executable binary must not turn a working rig
  // into a crash.
  test('a failing chmod on the cached path is swallowed', async () => {
    const chmod = jest.fn(() => { throw new Error('EROFS: read-only file system'); });
    const { mgr } = make('linux', LIN, true, { chmod });
    await expect(mgr.ensure()).resolves.toBe(path.join('/cache', linPkg.launcher));
    expect(chmod).toHaveBeenCalled();
  });
});

describe('EngineManager — Windows zip install', () => {
  test('downloads, unpacks, drops the archive and renames the launcher', async () => {
    const { mgr, fs, download, extractPackage, chmod } = make('win32', WIN, false);
    const dest = path.join('/cache', winPkg.launcher);
    const archive = path.join('/cache', winPkg.archive);
    const onProgress = jest.fn();

    await expect(mgr.ensure(onProgress)).resolves.toBe(dest);

    expect(fs.mkdirSync).toHaveBeenCalledWith('/cache', { recursive: true });
    expect(download).toHaveBeenCalledWith(expect.stringContaining(winPkg.archive), archive, onProgress);
    expect(extractPackage).toHaveBeenCalledWith(archive, '/cache');
    expect(fs.unlinkSync).toHaveBeenCalledWith(archive);
    // Windows has no execute bit to grant; a chmod there is at best a no-op and
    // at worst a throw that fails an otherwise perfect install.
    expect(chmod).not.toHaveBeenCalled();
  });

  // Upstream's zip holds an unversioned peakminer.exe. Left at that name, the
  // NEXT release would find it sitting at the expected path and never download —
  // a silent, permanent pin to whatever version installed first.
  test('the unversioned exe is renamed so a version bump is still a cache miss', async () => {
    const { mgr, fs } = make('win32', WIN, false);
    await mgr.ensure();
    expect(fs.renameSync).toHaveBeenCalledWith(
      path.join('/cache', winPkg.archiveLauncher),
      path.join('/cache', winPkg.launcher),
    );
  });

  test('a leftover archive that will not delete is not fatal', async () => {
    const fs = makeFs(false);
    fs.unlinkSync = jest.fn(() => { throw new Error('EBUSY'); });
    const { mgr } = make('win32', WIN, false, { fs });
    await expect(mgr.ensure()).resolves.toBe(path.join('/cache', winPkg.launcher));
  });
});

describe('EngineManager — Linux bare binary install', () => {
  // Not an archive: handing a plain ELF to the extractor would fail, so it is
  // saved straight to its final path and made executable.
  test('saves it in place and makes it executable', async () => {
    const { mgr, download, extract, extractPackage, chmod } = make('linux', LIN, false);
    const dest = path.join('/cache', linPkg.launcher);

    await expect(mgr.ensure()).resolves.toBe(dest);
    expect(download).toHaveBeenCalledWith(expect.stringContaining(linPkg.archive), dest, undefined);
    expect(chmod).toHaveBeenCalledWith(dest, 0o755);
    expect(extract).not.toHaveBeenCalled();
    expect(extractPackage).not.toHaveBeenCalled();
  });

  test('reports download progress', async () => {
    const { mgr, download } = make('linux', LIN, false);
    const onProgress = jest.fn();
    await mgr.ensure(onProgress);
    expect(download).toHaveBeenCalledWith(expect.any(String), expect.any(String), onProgress);
  });

  test('nothing is renamed — the published name is already versioned', async () => {
    const { mgr, fs } = make('linux', LIN, false);
    await mgr.ensure();
    expect(fs.renameSync).not.toHaveBeenCalled();
  });
});

// A user whose download fails fetches the engine in a browser — which saves it
// under upstream's own name, never the versioned one the cache looks for. The
// app used to ignore that file and re-run the download that just failed, so "I
// downloaded it manually" changed nothing. With an antivirus in the picture this
// path matters more, not less.
describe('EngineManager — manual install (undescribed version)', () => {
  function makeFsAt(present) {
    return {
      existsSync: jest.fn((p) => p === present),
      mkdirSync: jest.fn(),
      unlinkSync: jest.fn(),
      renameSync: jest.fn(),
    };
  }

  test('adopts a hand-downloaded binary instead of hitting the network', async () => {
    const manual = path.join('/cache', 'peakminer');
    const fs = makeFsAt(manual);
    const { mgr, download, chmod } = make('linux', '9.9.9', false, { fs });
    const dest = path.join('/cache', 'peakminer-9.9.9');

    await expect(mgr.ensure()).resolves.toBe(dest);
    expect(download).not.toHaveBeenCalled();
    expect(fs.renameSync).toHaveBeenCalledWith(manual, dest);
    // Renaming leaves it 0o644, like any browser download.
    expect(chmod).toHaveBeenCalledWith(dest, 0o755);
  });

  // The second adoption candidate is the destination itself for an undescribed
  // version, so the loop must skip it rather than rename a file onto itself.
  test('a file already at the destination counts as installed, not as adoptable', async () => {
    const dest = path.join('/cache', 'peakminer-9.9.9.exe');
    const fs = makeFsAt(dest);
    const { mgr, download } = make('win32', '9.9.9', false, { fs });

    await expect(mgr.ensure()).resolves.toBe(dest);
    expect(download).not.toHaveBeenCalled();
    expect(fs.renameSync).not.toHaveBeenCalled();
  });

  test('a hand-downloaded exe under the manual name is renamed into place', async () => {
    const manual = path.join('/cache', 'peakminer.exe');
    const fs = makeFsAt(manual);
    const { mgr, download } = make('win32', '9.9.9', false, { fs });

    await mgr.ensure();
    expect(download).not.toHaveBeenCalled();
    expect(fs.renameSync).toHaveBeenCalledWith(manual, path.join('/cache', 'peakminer-9.9.9.exe'));
  });

  test('an empty engine dir still downloads', async () => {
    const fs = makeFsAt('/nothing/matches');
    const { mgr, download } = make('linux', '9.9.9', false, { fs });
    await expect(mgr.ensure()).resolves.toBe(path.join('/cache', 'peakminer-9.9.9'));
    expect(download).toHaveBeenCalled();
    expect(fs.renameSync).not.toHaveBeenCalled();
  });
});
