'use strict';

const path = require('path');
const {
  DOWNLOAD_BASE, ENGINE, pickEngineVersion, parseDriverMajor,
  engineBinaryName, engineArchiveName, engineDownloadUrl,
  isZipUrl, isArchiveUrl, looksLikeArchive, enginePath, engineFiles, bundledEnginePath, progressPercent,
  manualEngineName, manualEnginePath, enginePackage, backendForEngine, manualInstallHint,
  parseComputeCaps, pickWindowsEngineVersion, windowsEngineNote,
  packagedLauncherRuns, spacedLauncherNote,
} = require('../src/shared/engine');

describe('pickEngineVersion', () => {
  test('new drivers get the fast build, old or unknown drivers the compatible one', () => {
    expect(pickEngineVersion(ENGINE.minDriverMajor)).toBe(ENGINE.preferred);
    expect(pickEngineVersion(999)).toBe(ENGINE.preferred);
    expect(pickEngineVersion(ENGINE.minDriverMajor - 1)).toBe(ENGINE.fallback);
    expect(pickEngineVersion(null)).toBe(ENGINE.fallback);
    expect(pickEngineVersion(NaN)).toBe(ENGINE.fallback);
    expect(pickEngineVersion(undefined)).toBe(ENGINE.fallback);
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
    expect(engineBinaryName('win32', 'nvidia', ENGINE.windows)).toBe('alpha-miner-windows-' + ENGINE.windows + '.exe');
    expect(engineBinaryName('win32', undefined, '1.8.6')).toBe('alpha-miner-windows-1.8.6.exe');
    expect(engineBinaryName('win32', 'amd')).toBe('alpha-miner-amd-windows-fixed.exe');
    expect(engineBinaryName('win32', 'amd', '1.8.6')).toBe('alpha-miner-amd-windows-fixed.exe'); // AMD ignores version
    expect(engineBinaryName('linux')).toBe('alpha-miner-' + ENGINE.fallback);
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
    expect(engineArchiveName('win32', 'amd')).toBe('AlphaMiner-Pearl-AMD.zip');
    expect(engineArchiveName('darwin')).toBe('alpha-miner-' + ENGINE.fallback);
    expect(engineArchiveName('linux', undefined, '1.8.8')).toBe('alpha-miner-1.8.8');
  });
});

describe('engineDownloadUrl', () => {
  test('uses the default base, an override, and the version', () => {
    expect(engineDownloadUrl('win32')).toBe(DOWNLOAD_BASE + 'AlphaMiner-Pearl-Windows.zip');
    expect(engineDownloadUrl('linux', undefined, 'https://mirror/')).toBe('https://mirror/alpha-miner-' + ENGINE.fallback);
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

describe('packaged engines (1.9.1b launcher + core)', () => {
  // 1.9.1b ships as a tarball holding a /bin/sh launcher beside a hidden core,
  // published only on GitHub releases. Every bare-binary assumption has to bend
  // for it: the path is a directory, the URL is absolute, the archive is a .tar.gz.
  const V = ENGINE.preferred;

  test('preferred is the packaged hotfix, never plain 1.9.1', () => {
    expect(V).toBe('1.9.1b');
    expect(enginePackage('linux', V)).toBeTruthy();
  });

  test('the binary path is the launcher inside the package dir', () => {
    expect(engineBinaryName('linux', undefined, V)).toBe('alpha-miner-1.9.1b/alpha-miner');
    expect(enginePath('/cache', 'linux', undefined, V))
      .toBe(path.join('/cache', 'alpha-miner-1.9.1b', 'alpha-miner'));
  });

  test('the download URL is absolute — GitHub releases, not the pool', () => {
    const url = engineDownloadUrl('linux', undefined, null, V);
    expect(url.startsWith('https://github.com/AlphaMine-Tech/')).toBe(true);
    expect(url).not.toContain(DOWNLOAD_BASE);
    expect(isArchiveUrl(url)).toBe(true);
    expect(engineArchiveName('linux', undefined, V)).toBe('alpha-miner-1.9.1b-ubuntu-amd64.tar.gz');
  });

  test('a custom base cannot redirect a packaged engine', () => {
    expect(engineDownloadUrl('linux', undefined, 'https://mirror.example/', V).startsWith('https://github.com/')).toBe(true);
  });

  test('Windows has its own package — same version, different artifact', () => {
    // Upstream added a Windows build to the same release tag after the Linux
    // one. It is a .zip of two .exe files, not a .tar.gz of a shell launcher
    // beside a hidden core, so the descriptor cannot be shared.
    const linux = enginePackage('linux', V);
    const win = enginePackage('win32', V);
    expect(win).toBeTruthy();
    expect(win.archive).toBe('AlphaMiner-Windows-1.9.1.02.zip');
    expect(win.launcher).toBe('alpha-miner.exe');
    expect(win.core).toBe('alpha-miner-core.exe');
    expect(win.url).not.toBe(linux.url);
    expect(engineBinaryName('win32', 'nvidia', V)).toBe('AlphaMiner-Windows-1.9.1.02/alpha-miner.exe');
  });

  test('the Windows fallback is still a bare .exe from the pool', () => {
    expect(enginePackage('win32', ENGINE.windows)).toBeNull();
    expect(engineBinaryName('win32', 'nvidia', ENGINE.windows)).toBe('alpha-miner-windows-1.8.6.exe');
    expect(engineDownloadUrl('win32', 'nvidia', null, ENGINE.windows))
      .toBe(DOWNLOAD_BASE + 'AlphaMiner-Pearl-Windows.zip');
  });

  test('the bare-binary fallback is untouched', () => {
    expect(enginePackage('linux', ENGINE.fallback)).toBeNull();
    expect(engineDownloadUrl('linux', undefined, null, ENGINE.fallback))
      .toBe(DOWNLOAD_BASE + 'alpha-miner-' + ENGINE.fallback);
  });
});

describe('backendForEngine', () => {
  // The packaged launcher exits 2 on --force-backend and selects the backend
  // itself. Dropping the override beats letting a working rig refuse to start.
  test('drops the override on a packaged engine and says why', () => {
    const lines = [];
    expect(backendForEngine('ampere', 'linux', ENGINE.preferred, (l) => lines.push(l))).toBeNull();
    expect(lines.join('')).toContain('--backend ampere is ignored on alpha-miner ' + ENGINE.preferred);
  });

  test('keeps it for a bare-binary engine', () => {
    expect(backendForEngine('ampere', 'linux', ENGINE.fallback, () => {})).toBe('ampere');
    expect(backendForEngine('ampere', 'win32', ENGINE.windows, () => {})).toBe('ampere');
  });

  test('no override set stays null, and logging is optional', () => {
    expect(backendForEngine(null, 'linux', ENGINE.preferred, () => {})).toBeNull();
    expect(backendForEngine('ampere', 'linux', ENGINE.preferred)).toBeNull(); // no log fn
  });
});

describe('parseComputeCaps', () => {
  test('one entry per card, junk dropped', () => {
    expect(parseComputeCaps('8.9\n')).toEqual(['8.9']);
    expect(parseComputeCaps('8.6\r\n8.6\r\n')).toEqual(['8.6', '8.6']);
    expect(parseComputeCaps('')).toEqual([]);
    expect(parseComputeCaps(null)).toEqual([]);
    expect(parseComputeCaps('NVIDIA-SMI has failed')).toEqual([]);
    expect(parseComputeCaps('[N/A]')).toEqual([]);
  });
});

describe('pickWindowsEngineVersion', () => {
  // The 1.9.1b Windows package fails closed on anything but a uniform CC 8.6 or
  // 8.9 rig, so this gate is the difference between mining and not starting at
  // all. Verified against the real launcher on a 4090: `--list-devices` reports
  // `cc=8.9 backend=ada` and runs.
  test('uniform 30- or 40-series gets the compliant hotfix', () => {
    expect(pickWindowsEngineVersion(['8.6'])).toBe(ENGINE.windowsPreferred);
    expect(pickWindowsEngineVersion(['8.9'])).toBe(ENGINE.windowsPreferred);
    expect(pickWindowsEngineVersion(['8.9', '8.9', '8.9'])).toBe(ENGINE.windowsPreferred);
  });

  test('mixed, newer and older rigs stay on the fallback the launcher will run', () => {
    expect(pickWindowsEngineVersion(['8.6', '8.9'])).toBe(ENGINE.windows); // mixed: fails closed
    expect(pickWindowsEngineVersion(['12.0'])).toBe(ENGINE.windows);       // RTX 50-series
    expect(pickWindowsEngineVersion(['7.5'])).toBe(ENGINE.windows);        // Turing
  });

  test('unknown capabilities fall back rather than guess', () => {
    expect(pickWindowsEngineVersion([])).toBe(ENGINE.windows);
    expect(pickWindowsEngineVersion(null)).toBe(ENGINE.windows);
    expect(pickWindowsEngineVersion(undefined)).toBe(ENGINE.windows);
  });
});

describe('windowsEngineNote', () => {
  test('says nothing when the rig qualified', () => {
    expect(windowsEngineNote(['8.9'])).toBe('');
  });

  test('names the reason, and admits the fallback is not compliant', () => {
    // A rig left behind cannot tell from the outside whether this is our choice
    // or upstream's limit, and it is mining work the fork no longer credits.
    expect(windowsEngineNote(['12.0'])).toContain('compute capability 12.0 is not supported');
    expect(windowsEngineNote(['8.6', '8.9'])).toContain('mixed GPU generations (8.6, 8.9)');
    expect(windowsEngineNote([])).toContain('compute capability could not be read');
    expect(windowsEngineNote(null)).toContain('compute capability could not be read');
    for (const caps of [['12.0'], ['8.6', '8.9'], [], null]) {
      expect(windowsEngineNote(caps)).toContain('not rank-128 compliant');
      expect(windowsEngineNote(caps)).toContain('1.9.2');
    }
  });
});

describe('packagedLauncherRuns / spacedLauncherNote', () => {
  // Upstream's 1.9.1.02 Windows launcher passes its core the right application
  // name but an unquoted command line, so Windows splits the core's own path on
  // spaces and the core rejects the tail as an argument:
  //   C:\Program Files\LLMJob Earn\…\alpha-miner-core.exe --pool …
  //   → ERROR: unknown argument: Files\LLMJob Earn\…\alpha-miner-core.exe
  // Both our Windows locations contain a space, which is what broke 0.3.12.
  const PKG = ENGINE.windowsPreferred;

  test('a Windows package cannot run from a spaced path', () => {
    expect(packagedLauncherRuns('win32', PKG, 'C:\\Program Files\\LLMJob Earn\\resources\\engine')).toBe(false);
    expect(packagedLauncherRuns('win32', PKG, 'C:\\Users\\me\\AppData\\Roaming\\LLMJob Earn\\engine')).toBe(false);
  });

  test('and runs fine from one without', () => {
    expect(packagedLauncherRuns('win32', PKG, 'C:\\LLMJobEarn\\engine')).toBe(true);
    expect(packagedLauncherRuns('win32', PKG, null)).toBe(true); // nothing to judge
  });

  test('bare binaries and non-Windows platforms are unaffected', () => {
    // The Linux package ships a /bin/sh launcher that execs "$CORE" "$@" —
    // correctly quoted — so a spaced path is only a Windows problem.
    expect(packagedLauncherRuns('linux', ENGINE.preferred, '/home/a b/engine')).toBe(true);
    expect(packagedLauncherRuns('win32', ENGINE.windows, 'C:\\Program Files\\LLMJob Earn')).toBe(true);
  });

  test('the note explains the downgrade without advising a pointless reinstall', () => {
    expect(spacedLauncherNote('win32', PKG, 'C:\\LLMJobEarn\\engine')).toBe('');
    const note = spacedLauncherNote('win32', PKG, 'C:\\Program Files\\LLMJob Earn\\engine');
    expect(note).toContain('cannot start from a path containing a space');
    expect(note).toContain(ENGINE.windows);
    expect(note).toContain('not rank-128 compliant');
    expect(note).toContain('1.9.2');
    // Moving the app would not help — the cache stays under %APPDATA%\LLMJob Earn.
    expect(note).not.toMatch(/reinstall/i);
  });
});

describe('engineFiles', () => {
  test('a bare binary is only itself; a package is the launcher AND its core', () => {
    expect(engineFiles('/cache', 'linux', undefined, ENGINE.fallback))
      .toEqual([enginePath('/cache', 'linux', undefined, ENGINE.fallback)]);

    for (const platform of ['linux', 'win32']) {
      const pkg = enginePackage(platform, '1.9.1b');
      expect(engineFiles('/cache', platform, undefined, '1.9.1b')).toEqual([
        path.join('/cache', pkg.dir, pkg.launcher),
        path.join('/cache', pkg.dir, pkg.core),
      ]);
    }
  });
});

describe('manualInstallHint', () => {
  // The manual-install hint is what a user sees when the in-app download failed,
  // so it has to describe an action that can actually work: a tarball saved as
  // the launcher name is not an engine.
  test('a packaged engine is extracted into the engine dir, not saved as a file', () => {
    expect(manualInstallHint('linux', ENGINE.preferred, '/cache'))
      .toEqual({ manualPath: null, extractDir: '/cache' });
  });

  test('a bare binary is saved as the pool\'s own download name', () => {
    expect(manualInstallHint('linux', ENGINE.fallback, '/cache'))
      .toEqual({ manualPath: manualEnginePath('/cache', 'linux'), extractDir: null });
    expect(manualInstallHint('win32', ENGINE.windows, '/cache'))
      .toEqual({ manualPath: manualEnginePath('/cache', 'win32'), extractDir: null });
  });
});
