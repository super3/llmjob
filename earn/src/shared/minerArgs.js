'use strict';

const { resolveEndpoint, splitEndpoint, DEFAULTS } = require('./config');
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
//
//   --host <host> --port <port>  SPLIT, deliberately. `--help` documents a single
//                            `--host <endpoint>` as "endpoint:PORT", and the
//                            build we test against accepts that combined form —
//                            but not every 1.9.4 build in the field does. A user
//                            on a build that reads --host as host-ONLY and
//                            appends its default port turned our
//                            `--host us2.alphapool.tech:5566` into
//                            `pool us2.alphapool.tech:5566:5566`, then looped on
//                            "DNS lookup failed: No such host is known" because
//                            it was resolving `us2.alphapool.tech:5566` as a
//                            hostname. Two field reports, same signature.
//
//                            The split form is what the pool documents for BOTH
//                            Linux and Windows, is accepted by the build we test
//                            (verified: connects and mines), and CANNOT produce a
//                            doubled port on either kind of build. `--port` is
//                            undocumented in --help but parsed — checked against
//                            the real binary, not the setup page.
//   --worker <addr>.<rig>    THERE IS NO --address. The payout address travels
//                            inside the worker field — the same thing the HiveOS
//                            template (%WAL%.%WORKER_NAME%) encodes, and what
//                            upstream flags as REQUIRED.
//   --password x;d=N         static difficulty as before.
//   --gpu <id>               one process per card; index defaults to 0.
//
// No backend override: "Rank, geometry, and backend are fixed to the AlphaPool
// mainnet rank-128 profile" — passing one is what backendForEngine strips.
function buildWorkerAddressArgs(settings, endpoint, worker, difficulty) {
  const address = String(settings.address == null ? '' : settings.address).trim();
  const login = worker ? address + '.' + worker : address;
  const password = 'x;d=' + difficulty;
  const gpu = settings.gpuIndex == null ? 0 : settings.gpuIndex;
  // Port omitted only when the endpoint carries none, leaving the engine its own
  // default rather than us inventing one.
  const { host, port } = splitEndpoint(endpoint);
  const args = ['--host', host];
  if (port) args.push('--port', String(port));
  return args.concat(['--worker', login, '--password', password, '--gpu', String(gpu)]);
}

// Build the alpha-miner argument vector, matching the engine's documented CLI
// (github.com/AlphaMine-Tech/alpha-miner): --pool / --address / --worker, with
// static difficulty pinned via the Stratum password (`x;d=N`). There is no
// --algo flag — the miner is Pearl-specific — and the pool/address/worker are
// separate flags (not a combined `<address>.<worker>` user). An optional forced
// backend is appended for cards that need it (`--force-backend ampere`).
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
  const address = String(settings.address == null ? '' : settings.address).trim();
  const password = 'x;d=' + difficulty;

  const args = ['--pool', 'stratum+tcp://' + endpoint, '--address', address];
  if (worker) args.push('--worker', worker);
  args.push('--password', password);
  if (settings.backend) args.push('--force-backend', settings.backend);
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
