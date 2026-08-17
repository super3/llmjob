'use strict';

const { resolveEndpoint, DEFAULTS } = require('./config');
const { combinePayoutAddress, isValidMdlAddress, normalizeAddress } = require('./address');
const { enginePackage } = require('./engine');

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

// The alpha-miner 1.9.4 CLI, which shares almost nothing with the 1.8.x one.
// Verified against the shipped binary's own --help and the start-mining.bat
// inside its zip, not the setup page (which is wrong about --host):
//
//   --host <endpoint:PORT>   one argument; --help documents it as "endpoint:PORT"
//                            and defaults to us2.alphapool.tech:5566. The page's
//                            split `--host H --port P` also parses, but the
//                            combined form is the documented one and our
//                            endpointFor() already yields host:port.
//   --worker <addr>.<rig>    THERE IS NO --address. The payout address travels
//                            inside the worker field — the same thing the HiveOS
//                            template (%WAL%.%WORKER_NAME%) encodes, and what
//                            upstream flags as REQUIRED. Merge-mining rides
//                            there too: `prl1…+mdl1….rig01` is echoed back
//                            intact by the miner, so the combined payout form
//                            survives.
//   --password x;d=N         static difficulty as before.
//   --gpu <id>               one process per card; index defaults to 0.
//
// No backend override: "Rank, geometry, and backend are fixed to the AlphaPool
// mainnet rank-128 profile" — passing one is what backendForEngine strips.
function buildWorkerAddressArgs(settings, endpoint, worker, difficulty) {
  const address = combinePayoutAddress(settings.address, settings.mdlAddress);
  const login = worker ? address + '.' + worker : address;
  const password = 'x;d=' + difficulty;
  const gpu = settings.gpuIndex == null ? 0 : settings.gpuIndex;
  return ['--host', endpoint, '--worker', login, '--password', password, '--gpu', String(gpu)];
}

// Build the alpha-miner argument vector, matching the engine's documented CLI
// (github.com/AlphaMine-Tech/alpha-miner): --pool / --address / --worker, with
// static difficulty pinned via the Stratum password (`x;d=N`). There is no
// --algo flag — the miner is Pearl-specific — and the pool/address/worker are
// separate flags (not a combined `<address>.<worker>` user). An optional forced
// backend is appended for cards that need it (`--force-backend ampere`).
//
// Merge mining differs by platform. The Windows engine accepts the combined
// `prl1…+mdl1…` login in --address, but the Linux engine the pool serves by
// default (1.8.3) bech32m-validates --address as one address and rejects the
// combined form before ever connecting (usage + exit 2 — a HiveOS crash loop).
// Off Windows the MDL address therefore rides in the Stratum password's legacy
// `mdl=` field instead: the engine passes the password through verbatim and the
// pool parses both forms.
function buildArgs(settings = {}) {
  // resolveEndpoint, not a raw read: an override pasted in the old
  // `stratum+tcp://host:port` form would otherwise reach --host verbatim and the
  // engine would try to resolve the scheme as part of the hostname.
  const endpoint = resolveEndpoint(settings);
  const worker = settings.worker != null ? settings.worker : DEFAULTS.worker;
  const difficulty = settings.difficulty || DEFAULTS.difficulty;

  // Engines that use the rank-128 CLI take a completely different vector. Keyed
  // off the engine descriptor rather than a version comparison so a future build
  // opts in by declaring `cli`, not by being newer than a magic string.
  const pkg = enginePackage(settings.platform, settings.engineVersion);
  if (pkg && pkg.cli === 'worker-address') {
    return buildWorkerAddressArgs(settings, endpoint, worker, difficulty);
  }
  const combined = settings.platform === 'win32';
  const address = combined
    ? combinePayoutAddress(settings.address, settings.mdlAddress)
    : String(settings.address == null ? '' : settings.address).trim();
  const mdl = normalizeAddress(settings.mdlAddress);
  let password = 'x;d=' + difficulty;
  if (!combined && isValidMdlAddress(mdl)) password += ';mdl=' + mdl;

  const args = ['--pool', 'stratum+tcp://' + endpoint, '--address', address];
  if (worker) args.push('--worker', worker);
  args.push('--password', password);
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
