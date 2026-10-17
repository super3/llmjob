'use strict';

const path = require('path');

// Where the RandomX worker (earn/xmr-worker) lives, and how it is started.
//
// It ships where the Pearl cores do: the installer's resources/native/, and
// beside the CLI binary (the HiveOS package too). LLMJOB_XMR_WORKER names a
// file to use instead, and a source checkout finds its own build. Nothing is
// downloaded.

const FILES = { win32: 'llmjob-xmr-worker.exe', linux: 'llmjob-xmr-worker' };

function workerFile(platform) {
  return FILES[platform] || null;
}

// The first candidate that exists, or null. `exists` is injected (fs.existsSync).
function findWorker({ resourcesPath, execPath, env, platform, exists }) {
  const file = workerFile(platform || process.platform);
  if (!file) return null;
  const e = env || process.env;
  const candidates = [];
  if (e.LLMJOB_XMR_WORKER) candidates.push(e.LLMJOB_XMR_WORKER);
  if (resourcesPath) candidates.push(path.join(resourcesPath, 'native', file));
  const execDir = path.dirname(execPath || process.execPath);
  candidates.push(path.join(execDir, file), path.join(execDir, 'native', file));
  const earn = path.join(__dirname, '..', '..');
  candidates.push(
    path.join(earn, 'xmr-worker', 'build', file),
    path.join(earn, 'xmr-worker', 'build', 'Release', file),
    path.join(earn, 'vendor', 'native', file),
  );
  return candidates.find((c) => exists(c)) || null;
}

// A spawnWorker for XmrEngine. Pipes on all three: the protocol is stdin and
// stdout, and stderr goes to the log. No window on Windows.
//
// On Linux the worker gets its own process group. Ctrl+C in a terminal signals
// the whole foreground group, and a worker killed that way could die before
// the CLI's shutdown ran, which logged a crash and a restart on a normal stop.
// Now only the CLI gets the signal, and it stops the worker. The worker still
// exits when its stdin closes or the app dies. (On Windows, detached would give
// it a console of its own, so it stays as it is.)
function makeSpawnWorker(spawn, platform = process.platform) {
  return (file, args) => spawn(file, args, {
    stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true, detached: platform !== 'win32',
  });
}

module.exports = { FILES, workerFile, findWorker, makeSpawnWorker };
