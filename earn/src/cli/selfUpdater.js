'use strict';

// IO shell for the CLI's self-update — the real HTTPS / filesystem / process
// side of shared/selfUpdate.js (whose decision logic is unit-tested there).
// Covered by mocking https/fs/child_process/node:sea in selfUpdater.test.js.

const https = require('https');
const fs = require('fs');
const { spawnSync } = require('child_process');
const { LATEST_RELEASE_API, parseRelease, planUpdate } = require('../shared/selfUpdate');
const { getJson } = require('../main/io');

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

// Stream a URL to a file, following redirects. Rejects on HTTP error.
function download(url, dest, redirects) {
  redirects = redirects || 0;
  return new Promise((resolve, reject) => {
    if (redirects > 5) return reject(new Error('too many redirects'));
    const req = https.get(url, { headers: { 'User-Agent': 'llmjob-earn-cli' } }, (res) => {
      const code = res.statusCode || 0;
      if (code >= 300 && code < 400 && res.headers.location) {
        res.resume();
        return resolve(download(new URL(res.headers.location, url).toString(), dest, redirects + 1));
      }
      if (code !== 200) { res.resume(); return reject(new Error('HTTP ' + code + ' for ' + url)); }
      const out = fs.createWriteStream(dest);
      res.pipe(out);
      out.on('finish', () => out.close(() => resolve(dest)));
      out.on('error', reject);
    });
    req.on('error', reject);
  });
}

// Replace the running executable with a freshly downloaded binary. On Linux a
// running binary can be renamed over (the live process keeps its open inode),
// so download beside it then atomically rename into place.
async function applyUpdate(plan, execPath) {
  const exe = execPath || process.execPath;
  const tmp = exe + '.new-' + process.pid;
  await download(plan.downloadUrl, tmp);
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
  download,
  applyUpdate,
  reexec,
  planUpdate,
};
