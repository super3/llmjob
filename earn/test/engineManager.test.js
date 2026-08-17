'use strict';

const path = require('path');
const { EngineManager } = require('../src/main/engineManager');

function makeFs(installed) {
  return {
    existsSync: jest.fn(() => installed),
    mkdirSync: jest.fn(),
    unlinkSync: jest.fn(),
  };
}

describe('EngineManager', () => {
  test('constructs with no arguments', () => {
    expect(new EngineManager()).toBeInstanceOf(EngineManager);
  });

  test('ensure returns the cached path without downloading when installed', async () => {
    const fs = makeFs(true);
    const download = jest.fn();
    const mgr = new EngineManager({ dir: '/cache', platform: 'win32', fs, download });
    const dest = path.join('/cache', 'alpha-miner-windows.exe');

    await expect(mgr.ensure()).resolves.toBe(dest);
    expect(mgr.isInstalled()).toBe(true);
    expect(download).not.toHaveBeenCalled();
    expect(fs.mkdirSync).not.toHaveBeenCalled();
  });

  test('re-asserts the execute bit on a cached binary off Windows', async () => {
    const fs = makeFs(true);
    const download = jest.fn();
    const chmod = jest.fn();
    const mgr = new EngineManager({ dir: '/cache', platform: 'linux', version: '1.8.8', fs, download, chmod });
    const dest = path.join('/cache', 'alpha-miner-1.8.8');

    await expect(mgr.ensure()).resolves.toBe(dest);
    expect(download).not.toHaveBeenCalled();
    expect(fs.mkdirSync).not.toHaveBeenCalled();
    // A cached binary that lost its +x (e.g. an interrupted first install)
    // gets it back here, so the rig stops crash-looping on spawn EACCES.
    expect(chmod).toHaveBeenCalledWith(dest, 0o755);
  });

  test('a failing chmod on the cached path is swallowed (best effort)', async () => {
    const fs = makeFs(true);
    const chmod = jest.fn(() => { throw new Error('EROFS: read-only file system'); });
    const mgr = new EngineManager({ dir: '/cache', platform: 'linux', version: '1.8.8', fs, chmod });
    const dest = path.join('/cache', 'alpha-miner-1.8.8');

    // Must still resolve: a chmod failure on an already-executable binary must
    // not turn a working rig into a crash.
    await expect(mgr.ensure()).resolves.toBe(dest);
    expect(chmod).toHaveBeenCalledWith(dest, 0o755);
  });

  test('downloads and extracts the zip on Windows, no chmod', async () => {
    const fs = makeFs(false);
    const download = jest.fn(() => Promise.resolve());
    const extract = jest.fn(() => Promise.resolve());
    const chmod = jest.fn();
    const onProgress = jest.fn();
    const mgr = new EngineManager({ dir: '/cache', platform: 'win32', fs, download, extract, chmod });
    const dest = path.join('/cache', 'alpha-miner-windows.exe');
    const zipPath = path.join('/cache', 'engine.zip');

    await expect(mgr.ensure(onProgress)).resolves.toBe(dest);

    expect(fs.mkdirSync).toHaveBeenCalledWith('/cache', { recursive: true });
    expect(download).toHaveBeenCalledWith(expect.stringMatching(/AlphaMiner-Pearl-Windows\.zip$/), zipPath, onProgress);
    expect(extract).toHaveBeenCalledWith(zipPath, dest);
    expect(fs.unlinkSync).toHaveBeenCalledWith(zipPath);
    expect(chmod).not.toHaveBeenCalled();
  });

  test('downloads the bare binary and chmods it off Windows', async () => {
    const fs = makeFs(false);
    const download = jest.fn(() => Promise.resolve());
    const extract = jest.fn(() => Promise.resolve());
    const chmod = jest.fn();
    const mgr = new EngineManager({ dir: '/cache', platform: 'linux', fs, download, extract, chmod });
    const dest = path.join('/cache', 'alpha-miner-1.8.3');

    await expect(mgr.ensure()).resolves.toBe(dest);

    expect(download).toHaveBeenCalledWith(expect.stringMatching(/alpha-miner-1\.8\.3$/), dest, undefined);
    expect(extract).not.toHaveBeenCalled();
    expect(chmod).toHaveBeenCalledWith(dest, 0o755);
  });

  test('an explicit version selects the binary name and download URL', async () => {
    const fs = makeFs(false);
    const download = jest.fn(() => Promise.resolve());
    const chmod = jest.fn();
    const mgr = new EngineManager({ dir: '/cache', platform: 'linux', version: '1.8.8', fs, download, chmod });
    const dest = path.join('/cache', 'alpha-miner-1.8.8');

    await expect(mgr.ensure()).resolves.toBe(dest);
    expect(download).toHaveBeenCalledWith(expect.stringMatching(/alpha-miner-1\.8\.8$/), dest, undefined);
  });
});

// A user whose download fails fetches the engine in a browser — which saves it
// under the pool's own name, never the versioned one the cache looks for. The
// app used to ignore that file and re-run the download that just failed, so
// "I downloaded it manually" changed nothing.
describe('EngineManager manual install', () => {
  function makeFsAt(present) {
    return {
      existsSync: jest.fn((p) => p === present),
      mkdirSync: jest.fn(),
      unlinkSync: jest.fn(),
      renameSync: jest.fn(),
    };
  }

  test('adopts a hand-downloaded alpha-miner instead of hitting the network', async () => {
    const manual = path.join('/cache', 'alpha-miner');
    const fs = makeFsAt(manual);
    const download = jest.fn();
    const chmod = jest.fn();
    const mgr = new EngineManager({ dir: '/cache', platform: 'linux', version: '1.8.8', fs, download, chmod });
    const dest = path.join('/cache', 'alpha-miner-1.8.8');

    await expect(mgr.ensure()).resolves.toBe(dest);
    expect(download).not.toHaveBeenCalled();
    expect(fs.renameSync).toHaveBeenCalledWith(manual, dest);
    // Renaming leaves it 0o644 like any browser download would.
    expect(chmod).toHaveBeenCalledWith(dest, 0o755);
  });

  test('extracts a hand-downloaded Windows zip and leaves the user\'s file alone', async () => {
    const zip = path.join('/cache', 'AlphaMiner-Pearl-Windows.zip');
    const fs = makeFsAt(zip);
    const download = jest.fn();
    const extract = jest.fn(() => Promise.resolve());
    const mgr = new EngineManager({ dir: '/cache', platform: 'win32', version: '1.8.6', fs, download, extract });
    const dest = path.join('/cache', 'alpha-miner-windows-1.8.6.exe');

    await expect(mgr.ensure()).resolves.toBe(dest);
    expect(download).not.toHaveBeenCalled();
    expect(extract).toHaveBeenCalledWith(zip, dest);
    expect(fs.unlinkSync).not.toHaveBeenCalled();
  });

  test('an empty engine dir still downloads', async () => {
    const fs = makeFsAt('/nothing/matches');
    const download = jest.fn(() => Promise.resolve());
    const mgr = new EngineManager({
      dir: '/cache', platform: 'linux', version: '1.8.8', fs, download, chmod: jest.fn(),
    });

    await expect(mgr.ensure()).resolves.toBe(path.join('/cache', 'alpha-miner-1.8.8'));
    expect(download).toHaveBeenCalled();
    expect(fs.renameSync).not.toHaveBeenCalled();
  });
});

describe('EngineManager — packaged engine (launcher + core)', () => {
  // 1.9.1b installs as a TREE, not a file. Two things must happen that the
  // bare-binary path never needed: the tarball is extracted whole, and BOTH the
  // launcher and its core are made executable. The tarball ships the core mode
  // 644 while the launcher checks -x on it, so chmodding only the launcher
  // leaves every start failing with "core missing or not executable".
  const { ENGINE, enginePackage } = require('../src/shared/engine');
  const V = ENGINE.preferred;
  const pkg = enginePackage('linux', V);

  function make(installed) {
    const fsStub = makeFs(installed);
    const download = jest.fn(() => Promise.resolve());
    const extractPackage = jest.fn(() => Promise.resolve());
    const chmod = jest.fn();
    const mgr = new EngineManager({
      dir: '/cache', platform: 'linux', version: V,
      fs: fsStub, download, extractPackage, chmod,
    });
    return { mgr, fsStub, download, extractPackage, chmod };
  }

  test('downloads the tarball, extracts it, and chmods launcher AND core', async () => {
    const { mgr, fsStub, download, extractPackage, chmod } = make(false);
    const dest = await mgr.ensure();

    expect(dest).toBe(path.join('/cache', pkg.dir, pkg.launcher));
    // fetched from the descriptor URL into the engine dir
    expect(download).toHaveBeenCalledWith(expect.stringContaining('github.com'), path.join('/cache', pkg.archive), undefined);
    expect(extractPackage).toHaveBeenCalledWith(path.join('/cache', pkg.archive), '/cache');
    expect(chmod).toHaveBeenCalledWith(dest, 0o755);
    expect(chmod).toHaveBeenCalledWith(path.join('/cache', pkg.dir, pkg.core), 0o755);
    // the scratch tarball does not linger
    expect(fsStub.unlinkSync).toHaveBeenCalledWith(path.join('/cache', pkg.archive));
  });

  test('a tarball that cannot be removed is not fatal', async () => {
    const { mgr, fsStub, chmod } = make(false);
    fsStub.unlinkSync.mockImplementation(() => { throw new Error('EBUSY'); });
    await expect(mgr.ensure()).resolves.toContain(pkg.launcher);
    expect(chmod).toHaveBeenCalledTimes(2);
  });

  test('an already-installed package re-asserts +x on the core too', async () => {
    const { mgr, chmod } = make(true);
    await mgr.ensure();
    expect(chmod).toHaveBeenCalledWith(path.join('/cache', pkg.dir, pkg.launcher), 0o755);
    expect(chmod).toHaveBeenCalledWith(path.join('/cache', pkg.dir, pkg.core), 0o755);
  });

  // Windows runs its own version on its own shape: 1.9.4 is a FLAT package —
  // one self-contained .exe at the root of the zip, no `dir`, no core half. It
  // still installs through the package path (download the archive, extract the
  // tree) but must land at the zip root, and there is no execute bit on Windows
  // to grant — a chmod there is at best a no-op, at worst a throw that fails an
  // otherwise perfect install.
  test('the flat Windows package installs at the zip root, without chmod', async () => {
    const WIN = ENGINE.windows;
    const winPkg = enginePackage('win32', WIN);
    expect(winPkg.dir).toBeUndefined();
    expect(winPkg.core).toBeUndefined();

    const fsStub = makeFs(false);
    const download = jest.fn(() => Promise.resolve());
    const extractPackage = jest.fn(() => Promise.resolve());
    const chmod = jest.fn();
    const mgr = new EngineManager({
      dir: '/cache', platform: 'win32', version: WIN,
      fs: fsStub, download, extractPackage, chmod,
    });

    const dest = await mgr.ensure();
    expect(dest).toBe(path.join('/cache', winPkg.launcher));
    expect(download).toHaveBeenCalledWith(expect.stringContaining(winPkg.archive), path.join('/cache', winPkg.archive), undefined);
    expect(extractPackage).toHaveBeenCalledWith(path.join('/cache', winPkg.archive), '/cache');
    expect(chmod).not.toHaveBeenCalled();
  });

  test('a launcher without its core is not installed — it re-downloads', async () => {
    // An interrupted extract leaves the launcher looking perfectly installed
    // while the larger half is missing. Trusting it would strand the rig on
    // "core missing or not executable" with nothing left to re-trigger the
    // download, so the core has to count as part of being installed.
    const { mgr, fsStub, download, extractPackage } = make(false);
    fsStub.existsSync.mockImplementation((p) => p === path.join('/cache', pkg.dir, pkg.launcher));

    expect(mgr.isInstalled()).toBe(false);
    await mgr.ensure();
    expect(download).toHaveBeenCalled();
    expect(extractPackage).toHaveBeenCalled();
  });
});
