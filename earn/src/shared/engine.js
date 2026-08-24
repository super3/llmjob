'use strict';

const path = require('path');

// Where and how to fetch the PeakMiner engine. We never bundle the binary — the
// app downloads it for the user on first start and caches it. Upstream publishes
// one artifact per platform on its GitHub releases: a zip holding a single
// self-contained `peakminer.exe` on Windows, and a bare versioned binary on
// Linux. URLs are overridable so a hand-hosted mirror drops in without a code
// change.
//
// NOT BUNDLED, AND NOT ALLOWED TO BE. PeakMiner's licence is explicit —
// "proprietary, all rights reserved; no reverse engineering / redistribution" —
// so staging it into vendor/engine and shipping it inside our installer would be
// redistribution. Fetching it to the user's own machine from upstream's own URL
// is not, which is why the Windows bundling step in
// .github/workflows/miner-build.yml is deleted rather than merely disabled.

const DOWNLOAD_BASE = 'https://github.com/peakminer/peakminer/releases/download/';

// One engine version per platform, the same build on both.
//
// `minDriverMajor` is deliberately null. alpha-miner carried 580 because
// AlphaPool documented R580+/CUDA 13 as a hard requirement and the app warned
// below it. PeakMiner publishes no such number: it embeds its own CUDA runtime
// and auto-selects a kernel profile by compute capability across
// volta/turing/ampere/ada/blackwell/h100/b200 plus a portable fallback, so the
// floor is far lower and we have no measured value for it. Inventing one would
// warn users off drivers that work, so driverTooOld answers false until somebody
// establishes a real threshold.
//
// Bumping a version is a cache miss by design (the version is baked into the
// cached filename), so rigs re-download rather than run a stale binary for ever.
const ENGINE = {
  minDriverMajor: null,
  windows: '2.11.0',
  linux: '2.11.0',
};

// Engine descriptors. A descriptor owns the whole artifact story for one
// platform+version — what to download, what it installs as, how to verify it,
// and which CLI it speaks — so adding a build is a table entry rather than a new
// branch in five functions.
const PACKAGED = {
  linux: {
    // A bare, already-versioned binary — no archive at all. `saveAsIs` tells
    // EngineManager to download it straight to its final path and chmod +x,
    // skipping extraction: handing a plain ELF to tar fails, and routing it
    // through the legacy bare-binary path would look for the wrong filenames.
    '2.11.0': {
      archive: 'peakminer-2.11.0-linux-x86_64',
      launcher: 'peakminer-2.11.0-linux-x86_64',
      sha256: '24f415716f456554a0c39ad09a6fc4c2fe52ec61ee1e346449dad6ea39de8c90',
      saveAsIs: true,
      cli: 'peakminer',
    },
  },
  win32: {
    // A flat zip holding one self-contained exe (plus reference .bat files and a
    // README we ignore). No DLLs: the CUDA runtime and GPU kernels are embedded.
    //
    // `archiveLauncher` is the name INSIDE the zip and `launcher` is what it is
    // installed as. They differ on purpose. Upstream ships an unversioned
    // `peakminer.exe`, and installing under that name would defeat the whole
    // cache-miss-on-bump rule — 2.12.0 would find 2.11.0's exe sitting at the
    // expected path and never download the new one. EngineManager renames after
    // extracting so the cached name carries the version.
    '2.11.0': {
      archive: 'peakminer-2.11.0-windows-x86_64.zip',
      archiveLauncher: 'peakminer.exe',
      launcher: 'peakminer-2.11.0.exe',
      // The release asset digest; matched the download byte for byte.
      sha256: 'ff7fcff77a458179089c05196e80e2830651e469039f06362641e8867f61f723',
      cli: 'peakminer',
    },
  },
};

// The package descriptor for `version` on `platform`, or null when that pairing
// is unknown. Everything that is not Windows reads the Linux table — the app
// ships for those two.
function enginePackage(platform, version) {
  const table = PACKAGED[platform === 'win32' ? 'win32' : 'linux'];
  return table[version] || null;
}

// Decide whether a --force-backend override survives for this engine.
//
// PeakMiner picks its kernel profile from the card's compute capability and
// exposes no backend override at all, so passing one would be an unknown
// argument and an immediate exit. Strip it and explain, rather than fail closed.
// Returns the backend to use (null when dropped) and calls `log` with the reason
// when it drops one.
function backendForEngine(backend, platform, version, log) {
  if (backend && enginePackage(platform, version)) {
    if (log) log('note: --backend ' + backend + ' is ignored on peakminer ' + version
      + ' — it selects a kernel profile by compute capability and takes no backend option');
    return null;
  }
  return backend || null;
}

// Where a hand-downloaded engine has to go, as the two fields describeSetupError
// understands. A bare binary is SAVED AS upstream's own filename; an archive is
// EXTRACTED into the engine dir. Telling someone to save a zip as the launcher
// is advice that cannot work, and the manual-install hint exists precisely for
// users whose download failed — including, now, users whose antivirus ate it.
function manualInstallHint(platform, version, dir) {
  const pkg = enginePackage(platform, version);
  // The Linux binary is saved, not extracted, and under the very name a browser
  // downloads it as — which is also the name the cache looks for, so a
  // hand-downloaded file dropped in the engine dir is picked up with no rename.
  if (pkg && pkg.saveAsIs) return { manualPath: path.join(dir, pkg.launcher), extractDir: null };
  if (pkg) return { manualPath: null, extractDir: dir };
  return { manualPath: manualEnginePath(dir, platform), extractDir: null };
}

// The engine version for a platform.
function engineVersionFor(platform) {
  return platform === 'win32' ? ENGINE.windows : ENGINE.linux;
}

// Is this rig's driver too old for the engine? Used only to warn. Answers false
// whenever no floor is configured (see ENGINE.minDriverMajor) — a warning we
// cannot substantiate is worse than none. An UNKNOWN driver (no nvidia-smi, or
// unparseable output) is likewise never treated as too old.
//
// The floor is a parameter, defaulting to the configured one, so the comparison
// stays exercisable while ENGINE.minDriverMajor is null — otherwise the moment
// somebody sets a real threshold they would be switching on a branch no test had
// ever run.
function driverTooOld(driverMajor, minMajor = ENGINE.minDriverMajor) {
  if (!Number.isFinite(minMajor)) return false;
  return Number.isFinite(driverMajor) && driverMajor < minMajor;
}

// Parse the driver major version out of `nvidia-smi --query-gpu=driver_version`
// output ("581.42\n581.42" → 581). Returns null when it can't.
function parseDriverMajor(output) {
  const m = String(output == null ? '' : output).match(/(\d+)\.\d+/);
  return m ? parseInt(m[1], 10) : null;
}

// The engine executable name once installed. The version is baked into the
// filename so a version bump is a cache miss — rigs re-download instead of
// running a stale cached binary for ever.
function engineBinaryName(platform, gpu, version) {
  const pkg = enginePackage(platform, version);
  if (pkg) return pkg.launcher;
  return platform === 'win32'
    ? 'peakminer-' + (version || ENGINE.windows) + '.exe'
    : 'peakminer-' + (version || ENGINE.linux);
}

// The name of the launcher INSIDE the downloaded archive, which is only
// sometimes the name it installs as — see PACKAGED.win32.archiveLauncher.
// Null when the artifact is not an archive, or installs under its own name.
function engineArchiveLauncher(platform, version) {
  const pkg = enginePackage(platform, version);
  return (pkg && pkg.archiveLauncher) || null;
}

// The downloadable artifact name (a zip on Windows, the bare binary otherwise).
function engineArchiveName(platform, gpu, version) {
  const pkg = enginePackage(platform, version);
  if (pkg) return pkg.archive;
  return engineBinaryName(platform, gpu, version);
}

// Release assets live under a per-version tag directory, so the base alone is
// not enough — the tag is part of the path.
function engineDownloadUrl(platform, gpu, base, version) {
  const v = version || engineVersionFor(platform);
  return (base || DOWNLOAD_BASE) + 'v' + v + '/' + engineArchiveName(platform, gpu, version);
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
// One self-contained executable on both platforms — PeakMiner carries its CUDA
// runtime and kernels inside the binary, so there is no second half to check.
function engineFiles(dir, platform, gpu, version) {
  return [enginePath(dir, platform, gpu, version)];
}

// The name a hand-downloaded engine lands under when a user fetches it in a
// browser because the in-app download failed — blocked HTTPS interception, an
// offline rig, or an antivirus that deleted it in flight. It is NOT the
// versioned cache name, so nothing picks it up on its own; EngineManager looks
// for it explicitly and installs it (see adoptManualDownload).
function manualEngineName(platform) {
  return platform === 'win32' ? 'peakminer.exe' : 'peakminer';
}

function manualEnginePath(dir, platform) {
  return path.join(dir, manualEngineName(platform));
}

// Absolute path to an engine bundled with a packaged app, if one is ever staged
// there again. Nothing stages it today and nothing may: PeakMiner's licence
// forbids redistribution, so the bundling step is gone from CI. The lookup stays
// because it is the seam a differently-licensed engine would slot into, and
// because it already fails soft — a missing bundle falls back to the on-demand
// download rather than erroring.
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
  engineArchiveLauncher,
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
