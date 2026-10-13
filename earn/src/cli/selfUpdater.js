'use strict';

// IO shell for the CLI's self-update — the real filesystem / process side of
// shared/selfUpdate.js (whose decision logic is unit-tested there). The HTTP
// side is delegated to io.js (getJson + downloadFile) so there's one hardened
// download path, not two.

const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');
const { LATEST_RELEASE_API, parseRelease, planUpdate } = require('../shared/selfUpdate');
const { getJson, downloadFile } = require('../main/io');

// Set on the re-exec'd child so it doesn't check/update again and loop.
const UPDATED_ENV = 'LLMJOB_EARN_UPDATED';

// Fetch + parse the latest release, or null if unreachable. GitHub requires a
// User-Agent header. Uses the shared best-effort JSON GET from io.js.
function fetchLatestRelease() {
  return getJson(LATEST_RELEASE_API, {
    headers: { 'User-Agent': 'llmjob-earn-cli', Accept: 'application/vnd.github+json' },
  }).then((j) => (j ? parseRelease(j) : null));
}

// True when running as the packaged single-file binary (vs `node earn-cli.js`).
// Only then can we replace ourselves from a release asset.
function isPackaged() {
  try {
    // Node Single Executable Application (how CI packages the binary).
    return require('node:sea').isSea();
  } catch (e) {
    // pkg-built binaries expose process.pkg.
    return !!process.pkg;
  }
}

// Replace the running executable with a freshly downloaded binary. On Linux a
// running binary can be renamed over (the live process keeps its open inode),
// so download beside it then atomically rename into place.
//
// Reuses io.downloadFile rather than a second, weaker copy: the CLI's old inline
// download had no response-error handler (a dropped connection mid-body became an
// uncaught exception that killed the process instead of falling back to "continue
// on the current version"), no stall timeout (a hung socket blocked mining
// forever, since auto-update runs before mining starts), and left its temp file
// behind on failure. io.downloadFile has all three (plus retry with backoff).
//
// `opts.xmr`: this rig mines Monero. The CPU worker comes along only then, or
// when one is already beside the binary to keep in step: a Pearl-only rig gets
// no coin miner it didn't ask for.
async function applyUpdate(plan, execPath, opts) {
  const exe = execPath || process.execPath;
  const tmp = exe + '.new-' + process.pid;
  await downloadFile(plan.downloadUrl, tmp);
  fs.chmodSync(tmp, 0o755);
  fs.renameSync(tmp, exe);
  // The native core updates WITH the binary, or not at all: a new CLI driving
  // an old pearl_core.node is a version skew nothing would report. Same
  // download-beside-then-rename dance, into the directory the loader probes.
  // Releases older than the split have no core asset; nothing to do then.
  const dir = require('path').dirname(exe);
  if (plan.coreUrl) {
    const core = require('path').join(dir, 'pearl_core.node');
    const coreTmp = core + '.new-' + process.pid;
    await downloadFile(plan.coreUrl, coreTmp);
    fs.renameSync(coreTmp, core);
  }
  // The CUDA 13 core (shared/coreVariant) is paired the same way but is
  // OPTIONAL, which changes the order. Last release's copy goes first, always:
  // on a Blackwell rig with driver 580+ the loader prefers that file, so one
  // left beside this release's binary is exactly the skew the rule above
  // prevents -- whether this release has no CUDA 13 core or its download fails.
  // And a failed download does not fail the update. The binary and the 12.8
  // core are already in place by now; throwing here would have the caller
  // "continue on" the old version with the NEW 12.8 core beside it, a skew of
  // its own, to save a 5090 3%. Without the file the loader says "not
  // installed" at start and mines on the 12.8 core.
  const cu13 = require('path').join(dir, 'pearl_core_cu13.node');
  fs.rmSync(cu13, { force: true });
  if (plan.coreCu13Url) {
    const cu13Tmp = cu13 + '.new-' + process.pid;
    try {
      await downloadFile(plan.coreCu13Url, cu13Tmp);
      fs.renameSync(cu13Tmp, cu13);
    } catch (e) {
      // Nothing to undo: downloadFile removes its own partial file.
    }
  }
  // The CPU worker. A failed download keeps the one already there, which the
  // CLI checks for a matching protocol when it starts it, and the next start
  // fetches it again (syncXmrWorker).
  if (plan.xmrWorkerUrl && ((opts && opts.xmr) || hasXmrWorker(exe))) {
    try {
      await fetchWorker(plan.xmrWorkerUrl, path.join(dir, WORKER));
    } catch (e) {
      // Nothing to undo: fetchWorker removes its temp file.
    }
  }
  return exe;
}

const WORKER = 'llmjob-xmr-worker';

// Whether a CPU worker sits beside the binary.
function hasXmrWorker(execPath) {
  return fs.existsSync(path.join(path.dirname(execPath || process.execPath), WORKER));
}

// Download the CPU worker beside the binary: to a temp file, executable, then
// renamed into place, so a failure never leaves half a program there.
async function fetchWorker(url, worker) {
  const tmp = worker + '.new-' + process.pid;
  try {
    await downloadFile(url, tmp);
    fs.chmodSync(tmp, 0o755);
    fs.renameSync(tmp, worker);
  } catch (e) {
    fs.rmSync(tmp, { force: true });
    throw e;
  }
}

// The version the worker states (--version), or null when it is missing or
// won't run, such as a copy downloaded by hand without chmod +x.
function workerVersion(worker) {
  const r = spawnSync(worker, ['--version'], { encoding: 'utf8', timeout: 10000 });
  if (r.error || r.status !== 0) return null;
  return String(r.stdout || '').trim() || null;
}

// Make the worker beside the binary this release's: fetch it when it is
// missing, won't run, or states another version. True when it fetched one;
// throws when the download fails.
async function syncXmrWorker(url, version, execPath) {
  const worker = path.join(path.dirname(execPath || process.execPath), WORKER);
  if (workerVersion(worker) === version) return false;
  await fetchWorker(url, worker);
  return true;
}

// Re-run the (now updated) binary with the same args, flagged so it won't loop.
// Returns the child's exit code.
function reexec(argv) {
  const env = Object.assign({}, process.env, { [UPDATED_ENV]: '1' });
  const r = spawnSync(process.execPath, argv, { stdio: 'inherit', env });
  return r.status == null ? 1 : r.status;
}

module.exports = {
  UPDATED_ENV,
  fetchLatestRelease,
  isPackaged,
  applyUpdate,
  hasXmrWorker,
  syncXmrWorker,
  reexec,
  planUpdate,
};
