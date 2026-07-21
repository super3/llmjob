'use strict';

const path = require('path');

// Where and how to fetch the AlphaPool `alpha-miner` engine. We never bundle the
// binary — the app downloads it for the user on first start and caches it. The
// pool ships Windows builds as zips (containing the .exe) and the bare binary
// elsewhere; URLs follow the documented /downloads/ path and are overridable.

const DOWNLOAD_BASE = 'https://pearl.alphapool.tech/downloads/';

// Linux engine versions, both hosted by the pool under /downloads/. The
// preferred build (1.8.6+ line) carries the NoisyGEMM kernel gains — 3-8% more
// hashrate on 40/50-series — but is compiled against CUDA 13, which needs
// NVIDIA driver >= 580; older drivers must stay on the last CUDA 12 stable or
// the engine dies at cudaGetDeviceCount. Windows keeps the pool's unversioned
// "windows-fixed" zips and ignores all of this.
const ENGINE = {
  preferred: '1.8.8',
  fallback: '1.8.3',
  minDriverMajor: 580,
};

// Pick the Linux engine version a rig can actually run from its NVIDIA driver
// major version. Unknown driver (no nvidia-smi / unparseable) → fallback: an
// old-driver rig that got the preferred build would crash-loop, while a new-
// driver rig on the fallback merely mines a few percent slower.
function pickEngineVersion(driverMajor) {
  return Number.isFinite(driverMajor) && driverMajor >= ENGINE.minDriverMajor
    ? ENGINE.preferred
    : ENGINE.fallback;
}

// Parse the driver major version out of `nvidia-smi --query-gpu=driver_version`
// output ("581.42\n581.42" → 581). Returns null when it can't.
function parseDriverMajor(output) {
  const m = String(output == null ? '' : output).match(/(\d+)\.\d+/);
  return m ? parseInt(m[1], 10) : null;
}

// The engine executable name once installed. Off Windows the version is baked
// into the filename so a version bump is a cache miss — rigs re-download
// instead of running a stale cached binary forever.
function engineBinaryName(platform, gpu, version) {
  if (platform === 'win32') {
    return gpu === 'amd' ? 'alpha-miner-amd-windows-fixed.exe' : 'alpha-miner-windows.exe';
  }
  return 'alpha-miner-' + (version || ENGINE.fallback);
}

// The downloadable artifact name (a zip on Windows, the bare binary otherwise).
function engineArchiveName(platform, gpu, version) {
  if (platform === 'win32') {
    return gpu === 'amd' ? 'AlphaMiner-Pearl-AMD.zip' : 'AlphaMiner-Pearl-Windows.zip';
  }
  return 'alpha-miner-' + (version || ENGINE.fallback);
}

function engineDownloadUrl(platform, gpu, base, version) {
  return (base || DOWNLOAD_BASE) + engineArchiveName(platform, gpu, version);
}

function isZipUrl(url) {
  return /\.zip$/i.test(String(url));
}

// Absolute path to the installed engine inside a cache directory.
function enginePath(dir, platform, gpu, version) {
  return path.join(dir, engineBinaryName(platform, gpu, version));
}

// Absolute path to the engine bundled with a packaged app. electron-builder
// copies vendor/engine → <resources>/engine (see build.extraResources), so at
// runtime it lives under process.resourcesPath. Returns null when no resources
// path is available (e.g. an unpackaged dev run) so callers fall back to the
// on-demand download. Off Windows the lookup is version-aware, so a bundle can
// only ever satisfy the exact build the rig's driver selected.
function bundledEnginePath(resourcesPath, platform, gpu, version) {
  if (!resourcesPath) return null;
  return path.join(resourcesPath, 'engine', engineBinaryName(platform, gpu, version));
}

// Download progress as a 0-100 integer, or null when the total size is unknown.
function progressPercent(received, total) {
  if (!total || total <= 0) return null;
  return Math.max(0, Math.min(100, Math.round((received / total) * 100)));
}

module.exports = {
  DOWNLOAD_BASE,
  ENGINE,
  pickEngineVersion,
  parseDriverMajor,
  engineBinaryName,
  engineArchiveName,
  engineDownloadUrl,
  isZipUrl,
  enginePath,
  bundledEnginePath,
  progressPercent,
};
