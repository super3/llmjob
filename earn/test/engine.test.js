'use strict';

const path = require('path');
const {
  DOWNLOAD_BASE, ENGINE, pickEngineVersion, parseDriverMajor,
  engineBinaryName, engineArchiveName, engineDownloadUrl,
  isZipUrl, enginePath, bundledEnginePath, progressPercent,
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

describe('enginePath', () => {
  test('joins the cache dir and binary name', () => {
    expect(enginePath('/cache', 'win32')).toBe(path.join('/cache', 'alpha-miner-windows.exe'));
    expect(enginePath('/cache', 'linux', undefined, '1.8.8')).toBe(path.join('/cache', 'alpha-miner-1.8.8'));
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
