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
//   • `windows` — the Windows FALLBACK — stays on 1.8.6 and is likewise not
//     softfork-compliant. See windowsPreferred below: upstream's Windows hotfix
//     runs on only two GPU generations, and 1.8.6 is all that is left for the
//     rest. Bumping it is a cache miss by design (the version is baked into the
//     cached filename), so Windows rigs re-download the pool's newer build
//     instead of running a stale cached .exe forever. Keep the bundling step in
//     .github/workflows/miner-build.yml in sync — it reads this.
//
// `windowsPreferred` is the 1.9.1b Windows package, added to the same upstream
// release on 2026-08-07 (the release originally shipped Linux/HiveOS/Docker
// only). Unlike the Linux build it is NOT a drop-in for every rig: the package
// fails closed on anything but a uniform RTX 30-series (CC 8.6) or uniform RTX
// 40-series (CC 8.9) system, and explicitly rejects RTX 50-series/Blackwell and
// mixed 8.6/8.9 rigs. So Windows picks its engine from the rig's compute
// capability the way Linux picks from the driver version — see
// pickWindowsEngineVersion. Upstream calls it "an emergency Windows bridge
// while the genuine-source Alpha Miner 1.9.2 release is being completed"; 1.9.2
// is what finally makes non-8.6/8.9 Windows rigs compliant.
const ENGINE = {
  preferred: '1.9.1b',
  fallback: '1.8.3',
  minDriverMajor: 580,
  windows: '1.8.6',
  windowsPreferred: '1.9.1b',
  // The only compute capabilities the Windows package will run on, verbatim
  // from its README. Uniform only — a rig mixing 8.6 and 8.9 fails closed too.
  windowsComputeCaps: ['8.6', '8.9'],
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
// `--gemm native` itself — verified against the real Windows launcher, which
// answers `--force-backend` with "ERROR: forbidden backend option in user
// arguments" and exit 2. See backendForEngine, which strips the override rather
// than letting a working rig fail closed.
//
// Keyed by platform because the same version ships as two different artifacts:
// a .tar.gz holding a /bin/sh launcher beside a hidden core on Linux, and a
// .zip holding an .exe pair on Windows.
const PACKAGED = {
  linux: {
    '1.9.1b': {
      url: 'https://github.com/AlphaMine-Tech/alpha-miner/releases/download/v1.9.1.02/alpha-miner-1.9.1b-ubuntu-amd64.tar.gz',
      archive: 'alpha-miner-1.9.1b-ubuntu-amd64.tar.gz',
      dir: 'alpha-miner-1.9.1b',
      launcher: 'alpha-miner',
      core: '.alpha-miner-core',
    },
  },
  win32: {
    // Same release tag, different artifact. The Windows core is a plain sibling
    // .exe rather than a hidden file, and neither half needs a chmod — but the
    // launcher still verifies the core's SHA-256 and refuses to run without it,
    // so the pair travels together exactly as on Linux.
    '1.9.1b': {
      url: 'https://github.com/AlphaMine-Tech/alpha-miner/releases/download/v1.9.1.02/AlphaMiner-Windows-1.9.1.02.zip',
      archive: 'AlphaMiner-Windows-1.9.1.02.zip',
      dir: 'AlphaMiner-Windows-1.9.1.02',
      launcher: 'alpha-miner.exe',
      core: 'alpha-miner-core.exe',
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

// Parse `nvidia-smi --query-gpu=compute_cap` output ("8.9\n8.9" → ['8.9','8.9'])
// into one entry per card. Returns [] when there is nothing to read, which
// pickWindowsEngineVersion treats as "unknown" rather than "supported".
function parseComputeCaps(output) {
  return String(output == null ? '' : output)
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => /^\d+\.\d+$/.test(l));
}

// Pick the Windows engine from the rig's CUDA compute capabilities.
//
// The 1.9.1b Windows package fails closed on anything but a UNIFORM CC 8.6
// (RTX 30-series) or UNIFORM CC 8.9 (RTX 40-series) system — mixed 8.6/8.9,
// RTX 50-series/Blackwell and everything older are refused by the launcher
// itself. So a rig that does not qualify gets 1.8.6, which still runs but is
// NOT softfork-compliant: there is simply no compliant Windows build for it
// until upstream's 1.9.2. Shipping a version the launcher rejects would take
// those rigs from earning nothing to mining nothing, which is strictly worse.
//
// Unknown capabilities (no nvidia-smi, AMD, unparseable) fall back for the same
// reason the driver check does: guessing wrong costs the rig every share.
function pickWindowsEngineVersion(caps) {
  const list = Array.isArray(caps) ? caps.map(String) : [];
  const uniform = list.length > 0 && list.every((c) => c === list[0]);
  return uniform && ENGINE.windowsComputeCaps.includes(list[0])
    ? ENGINE.windowsPreferred
    : ENGINE.windows;
}

// Why a Windows rig got the build it got, as a log line — '' when it qualified
// for the hotfix and there is nothing to explain. A rig left on 1.8.6 is mining
// work the rank-128 fork no longer credits, and it cannot tell from the outside
// whether that is our choice or upstream's limit, so say which and say what
// unblocks it.
function windowsEngineNote(caps) {
  if (pickWindowsEngineVersion(caps) === ENGINE.windowsPreferred) return '';
  const list = Array.isArray(caps) ? caps.map(String) : [];
  const why = list.length === 0
    ? 'GPU compute capability could not be read'
    : list.every((c) => c === list[0])
      ? 'compute capability ' + list[0] + ' is not supported by it'
      : 'mixed GPU generations (' + list.join(', ') + ') are not supported by it';
  return 'note: the ' + ENGINE.windowsPreferred + ' Windows build only runs on uniform CC '
    + ENGINE.windowsComputeCaps.join(' / ') + ' rigs and ' + why
    + ' — falling back to ' + ENGINE.windows + ', which still mines but is not'
    + ' rank-128 compliant. Upstream 1.9.2 is meant to cover the rest.';
}

// Can the packaged launcher for `version` actually start from `target`?
//
// The 1.9.1.02 Windows launcher resolves its core correctly and passes that
// path as the application name, but builds the child's command line by
// concatenating the same path UNQUOTED. Windows splits it on spaces, so the
// core reads the tail of our install path as its first argument and refuses:
//
//   C:\Program Files\LLMJob Earn\…\alpha-miner-core.exe --pool …
//   → ERROR: unknown argument: Files\LLMJob Earn\…\alpha-miner-core.exe
//
// Verified against the real 1.9.1.02 binaries: an identical argument vector
// runs from a space-free directory and dies from a spaced one, and the
// launcher's string table contains no quote character anywhere. Nothing we pass
// can work around it — the path it mangles is one it builds itself, after our
// arguments are already quoted.
//
// Windows-only: the Linux package ships a /bin/sh launcher that execs
// `"$CORE" "$@"`, correctly quoted, so a spaced path is fine there.
//
// This matters because BOTH of our Windows locations contain a space — the
// product is "LLMJob Earn", so the installer defaults to
// `C:\Program Files\LLMJob Earn` and the cache is `%APPDATA%\LLMJob Earn\engine`
// — which is why 0.3.12 could not start on any rig that qualified for the
// hotfix. Callers downgrade to ENGINE.windows rather than ship a launcher that
// cannot run at all; see spacedLauncherNote.
function packagedLauncherRuns(platform, version, target) {
  if (platform !== 'win32') return true;
  if (!enginePackage(platform, version)) return true; // bare binary: unaffected
  return !/\s/.test(String(target == null ? '' : target));
}

// Why a rig that qualified for the hotfix is not getting it after all — '' when
// the launcher can run from `target`. Distinct from windowsEngineNote, which
// explains the hardware gate; this one is about where the engine landed.
//
// Deliberately does NOT tell the user to reinstall somewhere space-free: the
// download cache lives under %APPDATA%\LLMJob Earn regardless of install
// directory, so moving the app would not win the hotfix back and the advice
// would send them in circles.
function spacedLauncherNote(platform, version, target) {
  if (packagedLauncherRuns(platform, version, target)) return '';
  return 'note: the ' + version + ' Windows launcher cannot start from a path containing'
    + ' a space (' + target + ') — it loses its own core path there and exits before'
    + ' mining. Falling back to ' + ENGINE.windows + ', which still mines but is not'
    + ' rank-128 compliant. This is a bug in upstream\'s launcher; their 1.9.2 release'
    + ' is meant to fix it.';
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
  parseComputeCaps,
  pickWindowsEngineVersion,
  windowsEngineNote,
  packagedLauncherRuns,
  spacedLauncherNote,
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
