'use strict';

const { endpointFor, DEFAULTS } = require('./config');
const { combinePayoutAddress, isValidMdlAddress, normalizeAddress } = require('./address');

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

// Build the alpha-miner argument vector, matching the engine's documented CLI
// (github.com/AlphaMine-Tech/alpha-miner): --pool / --address / --worker, with
// static difficulty pinned via the Stratum password (`x;d=N`). There is no
// --algo flag — the miner is Pearl-specific — and the pool/address/worker are
// separate flags (not a combined `<address>.<worker>` user). An optional forced
// backend is appended for cards that need it (`--force-backend ampere`). A merge-
// mining MDL address, when set, rides along in the --address as `prl1…+mdl1…`.
function buildArgs(settings = {}) {
  const region = settings.region || DEFAULTS.region;
  const endpoint = settings.endpoint || endpointFor(region);
  const worker = settings.worker != null ? settings.worker : DEFAULTS.worker;
  const address = combinePayoutAddress(settings.address, settings.mdlAddress);
  const difficulty = settings.difficulty || DEFAULTS.difficulty;

  const args = ['--pool', 'stratum+tcp://' + endpoint, '--address', address];
  if (worker) args.push('--worker', worker);
  args.push('--password', 'x;d=' + difficulty);
  if (settings.backend) args.push('--force-backend', settings.backend);
  return args;
}

// Environment variables for the native Windows launcher flow, where the
// start-mining .bat reads PRL_ADDRESS / MDL_ADDRESS / WORKER / PEARL_DIFFICULTY.
function buildEnv(settings = {}) {
  const mdl = normalizeAddress(settings.mdlAddress);
  return {
    PRL_ADDRESS: settings.address || '',
    MDL_ADDRESS: isValidMdlAddress(mdl) ? mdl : '',
    WORKER: settings.worker != null ? settings.worker : DEFAULTS.worker,
    PEARL_DIFFICULTY: String(settings.difficulty || DEFAULTS.difficulty),
  };
}

module.exports = { WIN_BINARIES, resolveBinary, buildArgs, buildEnv };
