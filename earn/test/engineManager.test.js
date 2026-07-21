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
