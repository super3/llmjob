'use strict';

const { endpointFor, DEFAULTS } = require('./config');

// Windows engine binaries shipped inside the AlphaPool zips.
const WIN_BINARIES = {
  nvidia: 'alpha-miner-windows.exe',
  amd: 'alpha-miner-amd-windows-fixed.exe',
};

// Resolve the miner binary. A configured absolute path wins; otherwise use the
// engine name for the platform/GPU (Windows is the shipped target for now).
function resolveBinary(binaryPath, platform, gpu) {
  if (binaryPath) return binaryPath;
  if (platform === 'win32') return WIN_BINARIES[gpu] || WIN_BINARIES.nvidia;
  return 'alpha-miner';
}

// Build the alpha-miner argument vector. The pool documents a static difficulty
// pinned via the Stratum password (`x;d=N`); the user is `<address>.<worker>`.
// An optional `--backend` is appended for cards that need a forced backend
// (e.g. A100/L4/L40 use `--backend ampere`).
function buildArgs(settings = {}) {
  const region = settings.region || DEFAULTS.region;
  const endpoint = settings.endpoint || endpointFor(region);
  const worker = settings.worker != null ? settings.worker : DEFAULTS.worker;
  const address = settings.address || '';
  const difficulty = settings.difficulty || DEFAULTS.difficulty;
  const algo = settings.algo || DEFAULTS.algo;
  const user = worker ? address + '.' + worker : address;

  const args = [
    '--algo', algo,
    '--url', 'stratum+tcp://' + endpoint,
    '--user', user,
    '--password', 'x;d=' + difficulty,
  ];
  if (settings.backend) args.push('--backend', settings.backend);
  return args;
}

// Environment variables for the native Windows launcher flow, where the
// start-mining .bat reads PRL_ADDRESS / WORKER / PEARL_DIFFICULTY.
function buildEnv(settings = {}) {
  return {
    PRL_ADDRESS: settings.address || '',
    WORKER: settings.worker != null ? settings.worker : DEFAULTS.worker,
    PEARL_DIFFICULTY: String(settings.difficulty || DEFAULTS.difficulty),
  };
}

module.exports = { WIN_BINARIES, resolveBinary, buildArgs, buildEnv };
