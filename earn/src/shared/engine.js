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
// the engine dies at cudaGetDeviceCount.
//
// `preferred` is 1.9.1b, the emergency rank-128 hotfix. Plain 1.9.1 was the
// softfork-required build (mainnet block 96,251, passed 2026-08-06) but
// mis-selected its backend and produced rank-256/512/1024 work after the fork,
// so upstream withdrew it behind the launcher described in PACKAGED below.
// Never pin plain 1.9.1: it installs cleanly and mines invalid work.
//
// Two gaps that are not ours to fix:
//
//   • `fallback` stays on 1.8.3 and is therefore NOT softfork-compliant. Every
//     upstream release note is silent on CUDA/driver requirements — the
//     driver >= 580 split above is our own inference — so there is no way to
//     tell whether the 1.9.x line runs on a pre-580 driver. Moving it blind
//     would crash-loop exactly the rigs the fallback exists to protect, which
//     is worse than mining non-compliant. Needs an answer from the pool.
//
//   • `windows` stays on 1.8.6 — the newest Windows build that exists. Upstream
//     shipped the whole 1.9.x line for HiveOS, Linux and Docker only, and the
//     pool's `AlphaMiner-Pearl-Windows.zip` still contains
//     `alpha-miner-windows-1.8.6.exe` (verified by downloading and listing it:
//     exe dated 2026-07-01, zip touched 2026-07-24). The 1.9.1b package cannot
//     substitute — its launcher is a /bin/sh script. Bumping this without a
//     build behind it would point every Windows rig at a filename that is not
//     inside the zip.
//
// Windows has a single build: the pool ships only one `AlphaMiner-Pearl-Windows.zip`
// (no per-driver split). We still track that version in `windows` and bake it
// into the cached filename (below) so bumping it here is a cache miss — Windows
// rigs re-download the pool's newer build instead of running a stale cached .exe
// forever. Bump `windows` to match the version inside the pool's zip whenever
// they refresh it (and keep the bundling step in
// .github/workflows/miner-build.yml in sync — it reads this).
const ENGINE = {
  preferred: '1.9.1b',
  fallback: '1.8.3',
  minDriverMajor: 580,
  windows: '1.8.6',
};

// Versions that ship as a PACKAGE — a tarball holding a launcher plus a hidden
// core — instead of the bare binary the pool serves for everything else. Three
// properties of 1.9.1b break the bare-binary assumptions in this file:
//
//   • it is published ONLY on GitHub releases. The pool's /downloads/ has no
//     copy under any name (checked alpha-miner-1.9.1b, alpha-miner-1.9.1.02 and
//     both tarball names — all 404), so `url` is absolute rather than built from
//     DOWNLOAD_BASE;
//   • the archive expands to a DIRECTORY and the launcher resolves its core as a
//     sibling, so the pair must stay together — the engine path is
//     `<dir>/<launcher>`, not a flat versioned filename;
//   • the tarball ships the core mode 644 while the launcher requires `-x`, so
//     install must chmod BOTH or every start dies with "core missing or not
//     executable".
//
// The launcher also rejects rank/geometry/backend overrides and appends
// `--gemm native` itself — see minerArgs, which stops sending --force-backend
// for a packaged engine rather than letting it fail closed with exit 2.
const PACKAGED = {
  '1.9.1b': {
    url: 'https://github.com/AlphaMine-Tech/alpha-miner/releases/download/v1.9.1.02/alpha-miner-1.9.1b-ubuntu-amd64.tar.gz',
    archive: 'alpha-miner-1.9.1b-ubuntu-amd64.tar.gz',
    dir: 'alpha-miner-1.9.1b',
    launcher: 'alpha-miner',
    core: '.alpha-miner-core',
  },
};

// The package descriptor for `version`, or null when it is a plain binary.
// Windows never matches: these are Linux tarballs whose launcher is a /bin/sh
// script, and upstream has shipped no Windows build of the 1.9.x line.
function enginePackage(platform, version) {
  if (platform === 'win32') return null;
  return PACKAGED[version] || null;
}

// Decide whether a --force-backend override survives for this engine.
//
// A packaged launcher rejects rank/geometry/backend overrides and exits 2 —
// it selects the backend itself and appends `--gemm native`. Passing the
// override through would turn a working rig into one that refuses to start,
// so strip it and explain, rather than fail closed. Returns the backend to
// use (null when dropped) and calls `log` with the reason when it drops one.
// Platform and version are parameters so this is testable without faking
// process.platform.
function backendForEngine(backend, platform, version, log) {
  if (backend && enginePackage(platform, version)) {
    if (log) log('note: --backend ' + backend + ' is ignored on alpha-miner ' + version
      + ' — its launcher selects the backend itself and rejects overrides');
    return null;
  }
  return backend || null;
}

// Where a hand-downloaded engine has to go, as the two fields describeSetupError
// understands. A bare binary is SAVED AS the pool's own filename; a package is
// EXTRACTED into the engine dir. Telling someone to save a tarball as the
// launcher is advice that cannot work, and the manual-install hint exists
// precisely for users whose download failed — so getting this wrong sends them
// in circles. Platform is a parameter so both branches are testable directly.
function manualInstallHint(platform, version, dir) {
  return enginePackage(platform, version)
    ? { manualPath: null, extractDir: dir }
    : { manualPath: manualEnginePath(dir, platform), extractDir: null };
}

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

// The engine executable name once installed. The version is baked into the
// filename so a version bump is a cache miss — rigs re-download instead of
// running a stale cached binary forever. On NVIDIA Windows the versioned name
// also matches the .exe inside the pool's zip (alpha-miner-windows-<ver>.exe),
// so extraction picks it by exact name. Falsy version keeps the legacy
// unversioned Windows name (AMD, which the pool ships without a version).
function engineBinaryName(platform, gpu, version) {
  const pkg = enginePackage(platform, version);
  if (pkg) return pkg.dir + '/' + pkg.launcher; // the launcher, beside its core
  if (platform === 'win32') {
    if (gpu === 'amd') return 'alpha-miner-amd-windows-fixed.exe';
    return version ? 'alpha-miner-windows-' + version + '.exe' : 'alpha-miner-windows.exe';
  }
  return 'alpha-miner-' + (version || ENGINE.fallback);
}

// The downloadable artifact name (a zip on Windows, the bare binary otherwise).
function engineArchiveName(platform, gpu, version) {
  const pkg = enginePackage(platform, version);
  if (pkg) return pkg.archive;
  if (platform === 'win32') {
    return gpu === 'amd' ? 'AlphaMiner-Pearl-AMD.zip' : 'AlphaMiner-Pearl-Windows.zip';
  }
  return 'alpha-miner-' + (version || ENGINE.fallback);
}

function engineDownloadUrl(platform, gpu, base, version) {
  const pkg = enginePackage(platform, version);
  if (pkg) return pkg.url; // absolute: GitHub releases, not the pool
  return (base || DOWNLOAD_BASE) + engineArchiveName(platform, gpu, version);
}

function isZipUrl(url) {
  return /\.zip$/i.test(String(url));
}

// Does this URL point at an ARCHIVE that must be extracted, rather than a bare
// binary we can save straight to its final path? llama.cpp ships Windows as a
// .zip but Linux/macOS as a .tar.gz, and gating extraction on isZipUrl alone
// meant the tarball was saved AS `llama-server` and chmod +x — a gzip file with
// the execute bit, which execvp rejects (ENOEXEC) and /bin/sh then tries to
// parse ("Syntax error: word unexpected"). Every non-Windows rig hit this.
function isArchiveUrl(url) {
  return /\.(zip|tar\.gz|tgz)$/i.test(String(url));
}

// The magic bytes of the archive formats we download. A cached file starting
// with one of these is an un-extracted archive sitting where the binary belongs
// (see isArchiveUrl) — install left it there, so it must not be trusted as the
// binary. gzip = 1f 8b, zip = "PK\x03\x04".
function looksLikeArchive(head) {
  const b = Buffer.isBuffer(head) ? head : Buffer.alloc(0);
  if (b.length >= 2 && b[0] === 0x1f && b[1] === 0x8b) return true;
  return b.length >= 4 && b[0] === 0x50 && b[1] === 0x4b && b[2] === 0x03 && b[3] === 0x04;
}

// Absolute path to the installed engine inside a cache directory.
function enginePath(dir, platform, gpu, version) {
  return path.join(dir, engineBinaryName(platform, gpu, version));
}

// Every file an install of `version` must have present to count as installed.
// A bare binary is only itself; a package is the launcher AND its hidden core.
// Checking the launcher alone would call a half-extracted package installed —
// the extract is interruptible and the core is the larger half — and then every
// start dies with "core missing or not executable" with nothing to re-trigger
// the download, because the launcher is right where it belongs.
function engineFiles(dir, platform, gpu, version) {
  const pkg = enginePackage(platform, version);
  const bin = enginePath(dir, platform, gpu, version);
  return pkg ? [bin, path.join(dir, pkg.dir, pkg.core)] : [bin];
}

// The name a hand-downloaded engine lands under. The pool's documented link is
// the unversioned /downloads/alpha-miner, so a browser saves exactly this — and
// a user who fetches it because the in-app download failed (blocked HTTPS
// interception, offline rig) drops that file into the engine dir. It is NOT the
// versioned cache name, so nothing picks it up on its own; EngineManager looks
// for it explicitly and installs it (see adoptManualDownload).
function manualEngineName(platform) {
  return platform === 'win32' ? 'alpha-miner.exe' : 'alpha-miner';
}

function manualEnginePath(dir, platform) {
  return path.join(dir, manualEngineName(platform));
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
  enginePackage,
  backendForEngine,
  manualInstallHint,
  pickEngineVersion,
  parseDriverMajor,
  engineBinaryName,
  engineArchiveName,
  engineDownloadUrl,
  isZipUrl,
  isArchiveUrl,
  looksLikeArchive,
  enginePath,
  engineFiles,
  manualEngineName,
  manualEnginePath,
  bundledEnginePath,
  progressPercent,
};
