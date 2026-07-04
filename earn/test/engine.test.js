'use strict';

const path = require('path');
const {
  DOWNLOAD_BASE, engineBinaryName, engineArchiveName, engineDownloadUrl,
  isZipUrl, enginePath, bundledEnginePath, progressPercent,
} = require('../src/shared/engine');

describe('engineBinaryName', () => {
  test('per platform and GPU vendor', () => {
    expect(engineBinaryName('win32')).toBe('alpha-miner-windows.exe');
    expect(engineBinaryName('win32', 'amd')).toBe('alpha-miner-amd-windows-fixed.exe');
    expect(engineBinaryName('linux')).toBe('alpha-miner');
  });
});

describe('engineArchiveName', () => {
  test('Windows ships zips, others the bare binary', () => {
    expect(engineArchiveName('win32')).toBe('AlphaMiner-Pearl-Windows.zip');
    expect(engineArchiveName('win32', 'amd')).toBe('AlphaMiner-Pearl-AMD.zip');
    expect(engineArchiveName('darwin')).toBe('alpha-miner');
  });
});

describe('engineDownloadUrl', () => {
  test('uses the default base and an override', () => {
    expect(engineDownloadUrl('win32')).toBe(DOWNLOAD_BASE + 'AlphaMiner-Pearl-Windows.zip');
    expect(engineDownloadUrl('linux', undefined, 'https://mirror/')).toBe('https://mirror/alpha-miner');
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
