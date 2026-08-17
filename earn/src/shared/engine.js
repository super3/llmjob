'use strict';

const path = require('path');

// Where and how to fetch the AlphaPool `alpha-miner` engine. We never bundle the
// binary — the app downloads it for the user on first start and caches it. The
// pool ships Windows builds as zips (containing the .exe) and the bare binary
// elsewhere; URLs follow the documented /downloads/ path and are overridable.

const DOWNLOAD_BASE = 'https://pearl.alphapool.tech/downloads/';

// One engine version per platform, both 1.9.4, both requiring NVIDIA driver
// R580+ (CUDA 13) — upstream's own stated requirement now, not our inference.
// On an older driver the miner exits at startup with a driver notice.
//
// There is no `fallback` on either platform any more. It used to be 1.8.3 on
// Linux, kept because upstream's release notes were silent on driver
// requirements and moving blind risked crash-looping the very rigs it
// protected. That reasoning has expired twice over: the pool now documents
// R580+ explicitly, and post-fork a fallback mines rank-256 work that is no
// longer credited. Falling back would burn power for nothing while looking
// like success — worse than failing loudly with a message that names the fix
// ("update to R580+").
//
// Bumping either version is a cache miss by design (the version is baked into
// the cached filename), so rigs re-download rather than run a stale binary for
// ever. Keep the bundling step in .github/workflows/miner-build.yml in sync —
// it reads this.
const ENGINE = {
  minDriverMajor: 580,
  // Windows: one exe with automatic RTX 30/40/50 dispatch, ~4.3 MB zipped.
  windows: '1.9.4',
  // Linux: one self-extracting makeself bundle, ~530 MB, same dispatch. See
  // PACKAGED.linux for why the size is load-bearing.
  linux: '1.9.4',
};

// Engine descriptors: versions that ship as something other than the bare
// binary the pool used to serve. A descriptor owns the whole artifact story for
// one platform+version — what to download, what it installs as, how to verify
// it, and which CLI it speaks — so adding a build is a table entry rather than
// a new branch in five functions.
//
// Keyed by platform because 1.9.4 ships as two quite different artifacts: a
// ~4.3 MB .zip holding one self-contained .exe on Windows, and a ~530 MB
// makeself .run on Linux. Neither is the old launcher-plus-hidden-core shape
// that 1.9.1b used, so the `dir`/`core` handling that shape needed is gone with
// it — an unreachable branch is a liability, and git remembers how it worked.
//
// A 1.9.x engine also rejects rank/geometry/backend overrides and picks its own
// backend — verified against the real Windows binary, which answers
// `--force-backend` with "ERROR: forbidden backend option in user arguments"
// and exit 2, and which the pool documents as "rank, geometry, and backend are
// fixed to the AlphaPool mainnet rank-128 profile". See backendForEngine, which
// strips the override rather than letting a working rig fail closed.
const PACKAGED = {
  linux: {
    // 1.9.4 is a THIRD shape again: a makeself self-extracting bundle (.run,
    // Content-Type application/x-makeself) that is downloaded, chmod +x'd and
    // spawned directly. It is neither a bare binary nor an archive we unpack —
    // it unpacks itself into a temp dir at each start — so `selfExtracting`
    // tells EngineManager to skip extraction entirely and save it straight to
    // its final path. Args pass through makeself to the miner untouched, which
    // is how upstream's own documented command line works.
    //
    // It is ~530 MB because it carries the CUDA runtime plus a core per
    // architecture. That size is load-bearing: it is why the Linux engine is
    // NOT bundled into the AppImage (see .github/workflows/miner-build.yml).
    // The 4 MB `-rtx3040` HiveOS bundle is the same engine minus the Blackwell
    // sm120 core — verified by listing it — which is exactly why upstream says
    // not to put it on a 50-series card, and why we do not use it to dodge the
    // download.
    '1.9.4': {
      archive: 'alphaminer-1.9.4-linux.run',
      launcher: 'alphaminer-1.9.4-linux.run',
      sha256: 'e3dbba681c1f027b80c845f0747e35baae91dbb2ae7464dcc435e41498627f4e',
      selfExtracting: true,
      // Same 1.9.4 CLI as Windows: --host <host:port>, the payout address
      // INSIDE --worker as <address>.<rig>, --gpu <id>. Upstream's HiveOS
      // flight sheet says the same thing in its own vocabulary — the wallet
      // template is `%WAL%.%WORKER_NAME%`, with "REQUIRED — the address travels
      // inside the worker field".
      cli: 'worker-address',
    },
  },
  win32: {
    // 1.9.4 — the mandatory rank-128 build. Unlike 1.9.1b this is a FLAT,
    // pool-hosted zip holding ONE self-contained exe (plus a reference
    // start-mining.bat we ignore), so there is no `dir` and no `core`, and the
    // URL is built from DOWNLOAD_BASE rather than pointing at GitHub.
    //
    // Every one of those claims was checked against the real artifact rather
    // than the setup page, which is wrong in two places:
    //
    //   • the page still describes "the guarded alpha-miner.exe launcher checks
    //     the exact core", which was the 1.9.1b shape. 1.9.4 verifies an
    //     EMBEDDED core — the zip has exactly two entries and no sibling core;
    //   • the page's PowerShell example passes `--host HOST --port PORT`, but
    //     the miner's own --help documents `--host <endpoint>` as "endpoint:PORT"
    //     and defaults to us2.alphapool.tech:5566. Both forms parse (verified
    //     against the binary), so we send the documented combined one.
    //
    // The exe also runs happily from a path containing a space, so the
    // ProgramData relocation 1.9.1b needed is gone along with the check that
    // decided it.
    '1.9.4': {
      archive: 'alphaminer-1.9.4-win-033f7027b.zip',
      launcher: 'AlphaMiner-Windows-1.9.4-033f7027.exe',
      // Published on the setup page; matched the download byte for byte.
      sha256: 'bdaafa7806ffd742c43221babbb9018ee8237816759dc224dd4d50cb8376bd73',
      // Selects the 1.9.4 argument shape in minerArgs: --host <host:port>,
      // the payout address INSIDE --worker as <address>.<rig>, and --gpu <id>.
      cli: 'worker-address',
    },
  },
};

// The package descriptor for `version` on `platform`, or null when that pairing
// is a plain binary. Everything that is not Windows reads the Linux table — the
// app ships for those two, and a bare binary is the safe answer anywhere else.
function enginePackage(platform, version) {
  const table = PACKAGED[platform === 'win32' ? 'win32' : 'linux'];
  return table[version] || null;
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
  const pkg = enginePackage(platform, version);
  // A self-extracting bundle is saved, not extracted — and it is saved under
  // the very name the browser downloads it as, which is also the name the cache
  // looks for, so a hand-downloaded .run dropped in the engine dir is picked up
  // with no rename at all.
  if (pkg && pkg.selfExtracting) return { manualPath: path.join(dir, pkg.launcher), extractDir: null };
  if (pkg) return { manualPath: null, extractDir: dir };
  return { manualPath: manualEnginePath(dir, platform), extractDir: null };
}

// The engine version for a platform. One build each now, so this is a lookup
// rather than the driver-based choice it used to be — see ENGINE for why the
// fallback went away.
function engineVersionFor(platform) {
  return platform === 'win32' ? ENGINE.windows : ENGINE.linux;
}

// Is this rig's driver too old for the engine? Used only to warn: there is no
// older build left to fall back to, and the miner refuses to start on its own
// with a message that names the fix. An UNKNOWN driver (no nvidia-smi, or
// unparseable output) is deliberately not treated as too old — we would be
// guessing, and the guess would scare a perfectly healthy rig.
function driverTooOld(driverMajor) {
  return Number.isFinite(driverMajor) && driverMajor < ENGINE.minDriverMajor;
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
  if (pkg) return pkg.launcher;
  if (platform === 'win32') {
    if (gpu === 'amd') return 'alpha-miner-amd-windows-fixed.exe';
    return version ? 'alpha-miner-windows-' + version + '.exe' : 'alpha-miner-windows.exe';
  }
  return 'alpha-miner-' + (version || ENGINE.linux);
}

// The downloadable artifact name (a zip on Windows, the bare binary otherwise).
function engineArchiveName(platform, gpu, version) {
  const pkg = enginePackage(platform, version);
  if (pkg) return pkg.archive;
  if (platform === 'win32') {
    return gpu === 'amd' ? 'AlphaMiner-Pearl-AMD.zip' : 'AlphaMiner-Pearl-Windows.zip';
  }
  return 'alpha-miner-' + (version || ENGINE.linux);
}

function engineDownloadUrl(platform, gpu, base, version) {
  const pkg = enginePackage(platform, version);
  // Absolute only when upstream publishes it outside the pool (GitHub);
  // otherwise it is a normal /downloads/ artifact.
  if (pkg) return pkg.url || (base || DOWNLOAD_BASE) + pkg.archive;
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
  return [enginePath(dir, platform, gpu, version)];
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
  engineVersionFor,
  driverTooOld,
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
