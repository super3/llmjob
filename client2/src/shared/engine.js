'use strict';

const path = require('path');

// Where and how to fetch the AlphaPool `alpha-miner` engine. We never bundle the
// binary — the app downloads it for the user on first start and caches it. The
// pool ships Windows builds as zips (containing the .exe) and the bare binary
// elsewhere; URLs follow the documented /downloads/ path and are overridable.

const DOWNLOAD_BASE = 'https://pearl.alphapool.tech/downloads/';

// The engine executable name once installed.
function engineBinaryName(platform, gpu) {
  if (platform === 'win32') {
    return gpu === 'amd' ? 'alpha-miner-amd-windows-fixed.exe' : 'alpha-miner-windows.exe';
  }
  return 'alpha-miner';
}

// The downloadable artifact name (a zip on Windows, the bare binary otherwise).
function engineArchiveName(platform, gpu) {
  if (platform === 'win32') {
    return gpu === 'amd' ? 'AlphaMiner-Pearl-AMD.zip' : 'AlphaMiner-Pearl-Windows.zip';
  }
  return 'alpha-miner';
}

function engineDownloadUrl(platform, gpu, base) {
  return (base || DOWNLOAD_BASE) + engineArchiveName(platform, gpu);
}

function isZipUrl(url) {
  return /\.zip$/i.test(String(url));
}

// Absolute path to the installed engine inside a cache directory.
function enginePath(dir, platform, gpu) {
  return path.join(dir, engineBinaryName(platform, gpu));
}

// Absolute path to the engine bundled with a packaged app. electron-builder
// copies vendor/engine → <resources>/engine (see build.extraResources), so at
// runtime it lives under process.resourcesPath. Returns null when no resources
// path is available (e.g. an unpackaged dev run) so callers fall back to the
// on-demand download.
function bundledEnginePath(resourcesPath, platform, gpu) {
  if (!resourcesPath) return null;
  return path.join(resourcesPath, 'engine', engineBinaryName(platform, gpu));
}

// Download progress as a 0-100 integer, or null when the total size is unknown.
function progressPercent(received, total) {
  if (!total || total <= 0) return null;
  return Math.max(0, Math.min(100, Math.round((received / total) * 100)));
}

module.exports = {
  DOWNLOAD_BASE,
  engineBinaryName,
  engineArchiveName,
  engineDownloadUrl,
  isZipUrl,
  enginePath,
  bundledEnginePath,
  progressPercent,
};
