'use strict';

const path = require('path');
const {
  DOWNLOAD_BASE, ENGINE, engineVersionFor, driverTooOld, parseDriverMajor,
  engineBinaryName, engineArchiveName, engineDownloadUrl,
  isZipUrl, isArchiveUrl, looksLikeArchive, enginePath, engineFiles, bundledEnginePath, progressPercent,
  manualEngineName, manualEnginePath, enginePackage, backendForEngine, manualInstallHint,
  } = require('../src/shared/engine');

describe('engineVersionFor / driverTooOld', () => {
  test('one build per platform, both 1.9.4', () => {
    expect(engineVersionFor('win32')).toBe(ENGINE.windows);
    expect(engineVersionFor('linux')).toBe(ENGINE.linux);
    expect(engineVersionFor('freebsd')).toBe(ENGINE.linux);
    expect(ENGINE.windows).toBe('1.9.4');
    expect(ENGINE.linux).toBe('1.9.4');
  });

  // There is nothing left to fall back TO, so this only decides whether to warn.
  // An unknown driver must NOT warn: we would be guessing, and the guess would
  // scare a perfectly healthy rig into "fixing" a driver that is already fine.
  test('warns only on a driver we can read AND that is too old', () => {
    expect(driverTooOld(ENGINE.minDriverMajor - 1)).toBe(true);
    expect(driverTooOld(ENGINE.minDriverMajor)).toBe(false);
    expect(driverTooOld(999)).toBe(false);
    expect(driverTooOld(null)).toBe(false);
    expect(driverTooOld(NaN)).toBe(false);
    expect(driverTooOld(undefined)).toBe(false);
  });

  // The pre-fork fallback is gone on purpose: post-softfork it mines rank-256
  // work the network does not credit, so a rig on it burns power and earns
  // nothing while looking healthy.
  test('no fallback survives anywhere in the table', () => {
    expect(ENGINE.fallback).toBeUndefined();
    expect(ENGINE.preferred).toBeUndefined();
    expect(enginePackage('linux', '1.8.3')).toBeNull();
    expect(enginePackage('linux', '1.9.1b')).toBeNull();
  });
});

describe('parseDriverMajor', () => {
  test('reads the major out of nvidia-smi output', () => {
    expect(parseDriverMajor('581.42\n')).toBe(581);
    expect(parseDriverMajor('550.90.07\n550.90.07')).toBe(550);
  });
  test('returns null on garbage or missing output', () => {
    expect(parseDriverMajor('')).toBeNull();
    expect(parseDriverMajor(null)).toBeNull();
    expect(parseDriverMajor('NVIDIA-SMI has failed')).toBeNull();
  });
});

describe('engineBinaryName', () => {
  test('per platform and GPU vendor', () => {
    expect(engineBinaryName('win32')).toBe('alpha-miner-windows.exe'); // legacy: no version
    // ENGINE.windows is a PACKAGE now, so it resolves to the exe inside the zip
    // rather than our own alpha-miner-windows-<ver>.exe convention.
    expect(engineBinaryName('win32', 'nvidia', ENGINE.windows)).toBe('AlphaMiner-Windows-1.9.4-033f7027.exe');
    expect(engineBinaryName('win32', undefined, '1.8.6')).toBe('alpha-miner-windows-1.8.6.exe');
    expect(engineBinaryName('win32', 'amd')).toBe('alpha-miner-amd-windows-fixed.exe');
    expect(engineBinaryName('win32', 'amd', '1.8.6')).toBe('alpha-miner-amd-windows-fixed.exe'); // AMD ignores version
    expect(engineBinaryName('linux')).toBe('alpha-miner-' + ENGINE.linux);
    expect(engineBinaryName('linux', undefined, '1.8.8')).toBe('alpha-miner-1.8.8');
  });
});

describe('ENGINE.windows', () => {
  test('pins the version inside the pool Windows zip', () => {
    expect(typeof ENGINE.windows).toBe('string');
    expect(ENGINE.windows).toMatch(/^\d+\.\d+\.\d+$/);
  });
});

describe('engineArchiveName', () => {
  test('Windows ships zips, others the versioned bare binary', () => {
    expect(engineArchiveName('win32')).toBe('AlphaMiner-Pearl-Windows.zip');
    // A described engine names its own artifact, whatever the convention says.
    expect(engineArchiveName('linux', undefined, ENGINE.linux)).toBe('alphaminer-1.9.4-linux.run');
    expect(engineArchiveName('win32', 'nvidia', ENGINE.windows)).toBe('alphaminer-1.9.4-win-033f7027b.zip');
    expect(engineArchiveName('win32', 'amd')).toBe('AlphaMiner-Pearl-AMD.zip');
    expect(engineArchiveName('darwin')).toBe('alpha-miner-' + ENGINE.linux);
    expect(engineArchiveName('linux', undefined, '1.8.8')).toBe('alpha-miner-1.8.8');
  });
});

describe('engineDownloadUrl', () => {
  test('uses the default base, an override, and the version', () => {
    expect(engineDownloadUrl('win32')).toBe(DOWNLOAD_BASE + 'AlphaMiner-Pearl-Windows.zip');
    expect(engineDownloadUrl('linux', undefined, 'https://mirror/', '1.8.8')).toBe('https://mirror/alpha-miner-1.8.8');
    expect(engineDownloadUrl('linux', undefined, null, '1.8.8')).toBe(DOWNLOAD_BASE + 'alpha-miner-1.8.8');
  });
});

describe('isZipUrl', () => {
  test('detects .zip URLs', () => {
    expect(isZipUrl('https://x/AlphaMiner-Pearl-Windows.zip')).toBe(true);
    expect(isZipUrl('https://x/alpha-miner')).toBe(false);
    expect(isZipUrl(null)).toBe(false);
  });
});

describe('isArchiveUrl', () => {
  test('covers every archive we download, not just .zip', () => {
    // The llama.cpp assets: Windows .zip, Linux/macOS .tar.gz — all must extract.
    expect(isArchiveUrl('https://x/llama-b9902-bin-win-vulkan-x64.zip')).toBe(true);
    expect(isArchiveUrl('https://x/llama-b9902-bin-ubuntu-vulkan-x64.tar.gz')).toBe(true);
    expect(isArchiveUrl('https://x/llama-macos-arm64.tgz')).toBe(true);
    expect(isArchiveUrl('https://x/LLAMA.TAR.GZ')).toBe(true);
    // A bare binary is saved straight to its path, never extracted.
    expect(isArchiveUrl('https://x/llama-server')).toBe(false);
    expect(isArchiveUrl('https://x/llama-server.exe')).toBe(false);
    expect(isArchiveUrl(null)).toBe(false);
  });
});

describe('looksLikeArchive', () => {
  test('recognises gzip and zip magic, rejects an ELF or a short read', () => {
    expect(looksLikeArchive(Buffer.from([0x1f, 0x8b, 0x08, 0x00]))).toBe(true);
    expect(looksLikeArchive(Buffer.from([0x50, 0x4b, 0x03, 0x04]))).toBe(true);
    expect(looksLikeArchive(Buffer.from([0x7f, 0x45, 0x4c, 0x46]))).toBe(false); // \x7fELF
    expect(looksLikeArchive(Buffer.from([0x50, 0x4b, 0x05, 0x06]))).toBe(false); // empty-zip EOCD
    expect(looksLikeArchive(Buffer.from([0x1f]))).toBe(false);
    expect(looksLikeArchive(Buffer.alloc(0))).toBe(false);
    expect(looksLikeArchive(null)).toBe(false);
  });
});

describe('enginePath', () => {
  test('joins the cache dir and binary name', () => {
    expect(enginePath('/cache', 'win32')).toBe(path.join('/cache', 'alpha-miner-windows.exe'));
    expect(enginePath('/cache', 'linux', undefined, '1.8.8')).toBe(path.join('/cache', 'alpha-miner-1.8.8'));
  });
});

describe('manualEnginePath', () => {
  test('is the pool\'s own download name, not the versioned cache name', () => {
    // A browser saving /downloads/alpha-miner writes exactly this, so a user
    // rescuing a failed download drops the file here.
    expect(manualEnginePath('/cache', 'linux')).toBe(path.join('/cache', 'alpha-miner'));
    expect(manualEnginePath('/cache', 'win32')).toBe(path.join('/cache', 'alpha-miner.exe'));
    expect(manualEngineName('linux')).toBe('alpha-miner');
    expect(manualEnginePath('/cache', 'linux')).not.toBe(enginePath('/cache', 'linux', undefined, '1.8.8'));
  });
});

describe('bundledEnginePath', () => {
  test('resolves under the resources path when packaged', () => {
    expect(bundledEnginePath('/app/resources', 'win32')).toBe(path.join('/app/resources', 'engine', 'alpha-miner-windows.exe'));
  });
  test('honours the gpu variant', () => {
    expect(bundledEnginePath('/res', 'win32', 'amd')).toBe(path.join('/res', 'engine', 'alpha-miner-amd-windows-fixed.exe'));
  });
  test('returns null without a resources path (dev run)', () => {
    expect(bundledEnginePath(null, 'win32')).toBeNull();
    expect(bundledEnginePath(undefined, 'win32')).toBeNull();
  });
});

describe('progressPercent', () => {
  test('returns null for unknown totals and clamps otherwise', () => {
    expect(progressPercent(5, 0)).toBeNull();
    expect(progressPercent(5, -1)).toBeNull();
    expect(progressPercent(5, undefined)).toBeNull();
    expect(progressPercent(50, 100)).toBe(50);
    expect(progressPercent(200, 100)).toBe(100);
    expect(progressPercent(-5, 100)).toBe(0);
  });
});

describe('engine descriptors (1.9.4, one shape per platform)', () => {
  // Linux 1.9.4 is a makeself self-extracting bundle: downloaded, chmod +x'd and
  // spawned directly. Nothing unpacks it — it unpacks itself at each start — so
  // it must NOT be routed through the archive path.
  test('Linux is a pool-hosted self-extracting bundle, saved not unpacked', () => {
    const V = ENGINE.linux;
    const pkg = enginePackage('linux', V);
    expect(pkg.selfExtracting).toBe(true);
    expect(pkg.cli).toBe('worker-address');
    expect(pkg.sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(engineBinaryName('linux', undefined, V)).toBe('alphaminer-1.9.4-linux.run');
    expect(engineDownloadUrl('linux', undefined, null, V))
      .toBe(DOWNLOAD_BASE + 'alphaminer-1.9.4-linux.run');
    // Crucially not an archive: isArchiveUrl gates extraction elsewhere.
    expect(isArchiveUrl(engineDownloadUrl('linux', undefined, null, V))).toBe(false);
    // It installs as exactly one file, under the same name it downloads as.
    expect(engineFiles('/cache', 'linux', undefined, V))
      .toEqual([path.join('/cache', 'alphaminer-1.9.4-linux.run')]);
  });

  // Windows 1.9.4 is a FLAT zip — one self-contained exe, hosted by the pool.
  test('Windows is a flat, pool-hosted archive', () => {
    const win = enginePackage('win32', ENGINE.windows);
    expect(win.selfExtracting).toBeUndefined();
    expect(win.cli).toBe('worker-address');
    expect(engineBinaryName('win32', 'nvidia', ENGINE.windows)).toBe(win.launcher);
    expect(engineDownloadUrl('win32', 'nvidia', null, ENGINE.windows))
      .toBe(DOWNLOAD_BASE + 'alphaminer-1.9.4-win-033f7027b.zip');
    expect(engineFiles('/cache', 'win32', 'nvidia', ENGINE.windows))
      .toEqual([enginePath('/cache', 'win32', 'nvidia', ENGINE.windows)]);
  });

  // Both artifacts now live under the pool's /downloads/, so a mirror override
  // reaches them — unlike 1.9.1b, which was pinned to a GitHub release URL.
  test('a custom base redirects both platforms', () => {
    expect(engineDownloadUrl('linux', undefined, 'https://mirror.example/', ENGINE.linux))
      .toBe('https://mirror.example/alphaminer-1.9.4-linux.run');
    expect(engineDownloadUrl('win32', 'nvidia', 'https://mirror.example/', ENGINE.windows))
      .toBe('https://mirror.example/alphaminer-1.9.4-win-033f7027b.zip');
  });
});

describe('backendForEngine', () => {
  // The packaged launcher exits 2 on --force-backend and selects the backend
  // itself. Dropping the override beats letting a working rig refuse to start.
  test('drops the override on a packaged engine and says why', () => {
    const lines = [];
    expect(backendForEngine('ampere', 'linux', ENGINE.linux, (l) => lines.push(l))).toBeNull();
    expect(lines.join('')).toContain('--backend ampere is ignored on alpha-miner ' + ENGINE.linux);
  });

  test('keeps it for a bare-binary engine', () => {
    expect(backendForEngine('ampere', 'linux', '1.8.8', () => {})).toBe('ampere');
    expect(backendForEngine('ampere', 'win32', '1.8.6', () => {})).toBe('ampere'); // legacy bare exe
  });

  test('no override set stays null, and logging is optional', () => {
    expect(backendForEngine(null, 'linux', ENGINE.linux, () => {})).toBeNull();
    expect(backendForEngine('ampere', 'linux', ENGINE.linux)).toBeNull(); // no log fn
  });
});

describe('engineFiles', () => {
  // Every shape installs as exactly one file now — the launcher+core pair that
  // needed two went away with 1.9.1b.
  test('one file per engine, whatever its shape', () => {
    for (const [platform, version] of [['linux', ENGINE.linux], ['win32', ENGINE.windows], ['linux', '1.8.8']]) {
      expect(engineFiles('/cache', platform, undefined, version))
        .toEqual([enginePath('/cache', platform, undefined, version)]);
    }
  });
});

describe('manualInstallHint', () => {
  // The manual-install hint is what a user sees when the in-app download failed,
  // so it has to describe an action that can actually work: an archive saved as
  // the launcher name is not an engine, and a self-extracting bundle handed to
  // an unzip tool is not one either.
  test('an archive is extracted into the engine dir, not saved as a file', () => {
    expect(manualInstallHint('win32', ENGINE.windows, '/cache'))
      .toEqual({ manualPath: null, extractDir: '/cache' });
  });

  // The .run downloads under the very name the cache looks for, so the user
  // drops it in and it is simply there — no rename, no extraction step.
  test('a self-extracting bundle is saved under its own name', () => {
    expect(manualInstallHint('linux', ENGINE.linux, '/cache'))
      .toEqual({ manualPath: path.join('/cache', 'alphaminer-1.9.4-linux.run'), extractDir: null });
  });

  test('a bare binary is saved as the pool\'s own download name', () => {
    expect(manualInstallHint('linux', '1.8.8', '/cache'))
      .toEqual({ manualPath: manualEnginePath('/cache', 'linux'), extractDir: null });
    expect(manualInstallHint('win32', '1.8.6', '/cache'))
      .toEqual({ manualPath: manualEnginePath('/cache', 'win32'), extractDir: null });
  });
});
