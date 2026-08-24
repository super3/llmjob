'use strict';

const path = require('path');
const {
  DOWNLOAD_BASE, ENGINE, enginePackage, backendForEngine, manualInstallHint,
  engineVersionFor, driverTooOld, parseDriverMajor, engineBinaryName, engineArchiveLauncher,
  engineArchiveName, engineDownloadUrl, isZipUrl, isArchiveUrl, looksLikeArchive,
  enginePath, engineFiles, manualEngineName, manualEnginePath, bundledEnginePath,
  progressPercent,
} = require('../src/shared/engine');

const WIN = ENGINE.windows;
const LIN = ENGINE.linux;

describe('ENGINE', () => {
  test('one version per platform, the same build on both', () => {
    expect(engineVersionFor('win32')).toBe(WIN);
    expect(engineVersionFor('linux')).toBe(LIN);
    expect(engineVersionFor('darwin')).toBe(LIN);
    expect(WIN).toBe(LIN);
  });

  // Not an oversight. PeakMiner publishes no minimum driver — it embeds its own
  // CUDA runtime and picks a kernel profile by compute capability — so a number
  // here would be invented, and would warn users off drivers that work.
  test('no driver floor is claimed, so nothing is warned about', () => {
    expect(ENGINE.minDriverMajor).toBeNull();
    expect(driverTooOld(400)).toBe(false);
    expect(driverTooOld(999)).toBe(false);
  });
});

describe('driverTooOld', () => {
  // The floor is injectable so the comparison stays covered while the configured
  // one is null — otherwise setting a real threshold would light up a branch no
  // test had ever executed.
  test('compares against an explicit floor', () => {
    expect(driverTooOld(579, 580)).toBe(true);
    expect(driverTooOld(580, 580)).toBe(false);
    expect(driverTooOld(581, 580)).toBe(false);
  });

  test('an unreadable driver is never called too old', () => {
    expect(driverTooOld(null, 580)).toBe(false);
    expect(driverTooOld(undefined, 580)).toBe(false);
    expect(driverTooOld(NaN, 580)).toBe(false);
  });

  test('no floor means no verdict', () => {
    expect(driverTooOld(1, null)).toBe(false);
    expect(driverTooOld(1, undefined)).toBe(false);
  });
});

describe('parseDriverMajor', () => {
  test('reads the major off nvidia-smi output', () => {
    expect(parseDriverMajor('581.42\n581.42')).toBe(581);
    expect(parseDriverMajor('591.86')).toBe(591);
  });

  test('returns null when there is nothing to read', () => {
    expect(parseDriverMajor('')).toBeNull();
    expect(parseDriverMajor(null)).toBeNull();
    expect(parseDriverMajor(undefined)).toBeNull();
    expect(parseDriverMajor('no version here')).toBeNull();
  });
});

describe('enginePackage', () => {
  test('every shipped platform/version pair has a descriptor', () => {
    expect(enginePackage('win32', WIN)).toBeTruthy();
    expect(enginePackage('linux', LIN)).toBeTruthy();
    // Anything that is not Windows reads the Linux table.
    expect(enginePackage('darwin', LIN)).toBe(enginePackage('linux', LIN));
  });

  test('an unknown version has none', () => {
    expect(enginePackage('win32', '0.0.0')).toBeNull();
    expect(enginePackage('linux', '0.0.0')).toBeNull();
    expect(enginePackage('win32', undefined)).toBeNull();
  });
});

describe('engine descriptors', () => {
  // The Windows zip is flat and holds ONE self-contained exe — no DLLs, because
  // the CUDA runtime and kernels are embedded in the binary.
  test('Windows is a flat zip whose launcher is renamed on install', () => {
    const pkg = enginePackage('win32', WIN);
    expect(pkg.archive).toMatch(/\.zip$/);
    expect(pkg.archiveLauncher).toBe('peakminer.exe');
    expect(pkg.saveAsIs).toBeUndefined();
    expect(pkg.cli).toBe('peakminer');
    expect(pkg.sha256).toMatch(/^[0-9a-f]{64}$/);
  });

  // The unversioned name inside the zip must NOT be the installed name, or a
  // future version bump would find this exe already sitting at the expected path
  // and never download the new one.
  test('the installed Windows name carries the version, the archive one does not', () => {
    const pkg = enginePackage('win32', WIN);
    expect(pkg.launcher).toContain(WIN);
    expect(pkg.launcher).not.toBe(pkg.archiveLauncher);
  });

  // EngineManager renames unconditionally after unpacking, so an archive
  // descriptor that forgot archiveLauncher — or set it equal to launcher —
  // would crash the install. Enforced here rather than guarded at runtime,
  // where the guard could only ever be a dead branch.
  test('every archive descriptor declares a distinct archiveLauncher', () => {
    for (const [platform, version] of [['win32', WIN], ['linux', LIN]]) {
      const pkg = enginePackage(platform, version);
      if (pkg.saveAsIs) continue;
      expect(typeof pkg.archiveLauncher).toBe('string');
      expect(pkg.archiveLauncher).not.toBe(pkg.launcher);
    }
  });

  test('Linux is a bare versioned binary, saved rather than unpacked', () => {
    const pkg = enginePackage('linux', LIN);
    expect(pkg.saveAsIs).toBe(true);
    expect(pkg.archive).toBe(pkg.launcher);
    expect(pkg.archive).not.toMatch(/\.(zip|tar\.gz|tgz)$/);
    expect(pkg.sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(pkg.cli).toBe('peakminer');
  });
});

describe('engineDownloadUrl', () => {
  // Release assets live under a per-version tag directory, so the tag is part of
  // the path and the base alone is not enough.
  test('points at the real upstream release assets', () => {
    expect(engineDownloadUrl('win32', null, null, WIN))
      .toBe(DOWNLOAD_BASE + 'v' + WIN + '/peakminer-' + WIN + '-windows-x86_64.zip');
    expect(engineDownloadUrl('linux', null, null, LIN))
      .toBe(DOWNLOAD_BASE + 'v' + LIN + '/peakminer-' + LIN + '-linux-x86_64');
  });

  test('honours a base override on both platforms', () => {
    expect(engineDownloadUrl('win32', null, 'https://mirror/', WIN)).toMatch(/^https:\/\/mirror\/v/);
    expect(engineDownloadUrl('linux', null, 'https://mirror/', LIN)).toMatch(/^https:\/\/mirror\/v/);
  });

  test('falls back to the platform version when none is given', () => {
    expect(engineDownloadUrl('win32', null, null, null)).toContain('v' + WIN);
    expect(engineDownloadUrl('linux', null, null, null)).toContain('v' + LIN);
  });
});

describe('names and paths', () => {
  test('engineBinaryName uses the descriptor when there is one', () => {
    expect(engineBinaryName('win32', null, WIN)).toBe('peakminer-' + WIN + '.exe');
    expect(engineBinaryName('linux', null, LIN)).toBe('peakminer-' + LIN + '-linux-x86_64');
  });

  test('engineBinaryName still versions an undescribed build', () => {
    expect(engineBinaryName('win32', null, '9.9.9')).toBe('peakminer-9.9.9.exe');
    expect(engineBinaryName('linux', null, '9.9.9')).toBe('peakminer-9.9.9');
    expect(engineBinaryName('win32', null, null)).toBe('peakminer-' + WIN + '.exe');
    expect(engineBinaryName('linux', null, null)).toBe('peakminer-' + LIN);
  });

  test('engineArchiveLauncher is only set where the names differ', () => {
    expect(engineArchiveLauncher('win32', WIN)).toBe('peakminer.exe');
    expect(engineArchiveLauncher('linux', LIN)).toBeNull();
    expect(engineArchiveLauncher('win32', '9.9.9')).toBeNull();
  });

  test('engineArchiveName is the zip on Windows and the binary elsewhere', () => {
    expect(engineArchiveName('win32', null, WIN)).toMatch(/\.zip$/);
    expect(engineArchiveName('linux', null, LIN)).toBe('peakminer-' + LIN + '-linux-x86_64');
    expect(engineArchiveName('win32', null, '9.9.9')).toBe('peakminer-9.9.9.exe');
  });

  test('enginePath and engineFiles join the cache dir', () => {
    const p = path.join('/cache', 'peakminer-' + WIN + '.exe');
    expect(enginePath('/cache', 'win32', null, WIN)).toBe(p);
    // One self-contained executable, so there is no second half to check.
    expect(engineFiles('/cache', 'win32', null, WIN)).toEqual([p]);
  });

  test('manualEngineName is upstream\'s own download name, not the cache name', () => {
    expect(manualEngineName('win32')).toBe('peakminer.exe');
    expect(manualEngineName('linux')).toBe('peakminer');
    expect(manualEnginePath('/cache', 'win32')).toBe(path.join('/cache', 'peakminer.exe'));
  });

  test('bundledEnginePath resolves under the resources path, or nowhere', () => {
    expect(bundledEnginePath('/res', 'win32', null, WIN))
      .toBe(path.join('/res', 'engine', 'peakminer-' + WIN + '.exe'));
    expect(bundledEnginePath(null, 'win32', null, WIN)).toBeNull();
    expect(bundledEnginePath('', 'win32', null, WIN)).toBeNull();
  });
});

describe('manualInstallHint', () => {
  // A bare binary is SAVED AS its own name; an archive is EXTRACTED. Telling
  // someone to save a zip as the launcher is advice that cannot work, and this
  // hint exists precisely for people whose download failed.
  test('the Linux binary is saved under its own name', () => {
    expect(manualInstallHint('linux', LIN, '/cache')).toEqual({
      manualPath: path.join('/cache', 'peakminer-' + LIN + '-linux-x86_64'),
      extractDir: null,
    });
  });

  test('the Windows zip is extracted into the engine dir', () => {
    expect(manualInstallHint('win32', WIN, '/cache')).toEqual({ manualPath: null, extractDir: '/cache' });
  });

  test('an undescribed build falls back to the plain download name', () => {
    expect(manualInstallHint('win32', '9.9.9', '/cache')).toEqual({
      manualPath: path.join('/cache', 'peakminer.exe'), extractDir: null,
    });
  });
});

describe('backendForEngine', () => {
  // PeakMiner selects a kernel profile by compute capability and takes no
  // backend option, so passing one would be an unknown argument and an exit.
  test('drops the override on a described engine and says why', () => {
    const log = jest.fn();
    expect(backendForEngine('ampere', 'win32', WIN, log)).toBeNull();
    expect(log).toHaveBeenCalledWith(expect.stringContaining('peakminer'));
  });

  test('drops it without a logger too', () => {
    expect(backendForEngine('ampere', 'win32', WIN)).toBeNull();
  });

  test('keeps the override where no descriptor claims the version', () => {
    expect(backendForEngine('ampere', 'win32', '9.9.9', jest.fn())).toBe('ampere');
  });

  test('no backend asked for, none returned', () => {
    expect(backendForEngine(null, 'win32', WIN)).toBeNull();
    expect(backendForEngine('', 'win32', '9.9.9')).toBeNull();
  });
});

describe('archive sniffing', () => {
  test('isZipUrl and isArchiveUrl', () => {
    expect(isZipUrl('http://x/a.zip')).toBe(true);
    expect(isZipUrl('http://x/a.ZIP')).toBe(true);
    expect(isZipUrl('http://x/a.tar.gz')).toBe(false);
    expect(isArchiveUrl('http://x/a.zip')).toBe(true);
    expect(isArchiveUrl('http://x/a.tar.gz')).toBe(true);
    expect(isArchiveUrl('http://x/a.tgz')).toBe(true);
    expect(isArchiveUrl('http://x/peakminer-1-linux-x86_64')).toBe(false);
  });

  test('looksLikeArchive reads the magic bytes', () => {
    expect(looksLikeArchive(Buffer.from([0x1f, 0x8b, 0x08, 0x00]))).toBe(true);   // gzip
    expect(looksLikeArchive(Buffer.from([0x50, 0x4b, 0x03, 0x04]))).toBe(true);   // zip
    expect(looksLikeArchive(Buffer.from([0x7f, 0x45, 0x4c, 0x46]))).toBe(false);  // ELF
    expect(looksLikeArchive(Buffer.from([0x1f]))).toBe(false);
    expect(looksLikeArchive(Buffer.alloc(0))).toBe(false);
    expect(looksLikeArchive(null)).toBe(false);
    expect(looksLikeArchive('not a buffer')).toBe(false);
  });
});

describe('progressPercent', () => {
  test('scales and clamps, and gives up on an unknown total', () => {
    expect(progressPercent(50, 200)).toBe(25);
    expect(progressPercent(0, 200)).toBe(0);
    expect(progressPercent(400, 200)).toBe(100);
    expect(progressPercent(-5, 200)).toBe(0);
    expect(progressPercent(1, 0)).toBeNull();
    expect(progressPercent(1, null)).toBeNull();
    expect(progressPercent(1, -3)).toBeNull();
  });
});
