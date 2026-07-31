'use strict';

// IO shell for the CLI's self-update — the real filesystem / process side of
// shared/selfUpdate.js (whose decision logic is unit-tested there). The HTTP
// side is delegated to io.js (getJson + downloadFile) so there's one hardened
// download path, not two.

const fs = require('fs');
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
async function applyUpdate(plan, execPath) {
  const exe = execPath || process.execPath;
  const tmp = exe + '.new-' + process.pid;
  await downloadFile(plan.downloadUrl, tmp);
  fs.chmodSync(tmp, 0o755);
  fs.renameSync(tmp, exe);
  return exe;
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
  reexec,
  planUpdate,
};
