'use strict';

const path = require('path');

// SRBMiner-MULTI as an alternative mining engine.
//
// Why this exists: upstream's native alpha-miner has no Windows build newer
// than the 1.9.1b "emergency bridge" — 1.9.2 explicitly declines to publish
// one and 1.9.3 (Linux + HiveOS) did not either. AlphaPool's own guidance is to
// use a standard-stratum miner on port 5571 instead. SRBMiner is the one we can
// legally redistribute: its licence permits shipping the object code as part of
// an application that adds significant functionality, which this is.
//
// It is NOT the default. alpha-miner is first-party and takes 0%; SRBMiner
// takes 2% on pearlhash (from its own algorithm list), so routing a rig through
// it costs the user real money and has to be their choice, made with the fee in
// front of them.
//
// Everything in this file that could be checked was checked against the real
// 3.5.4 artifacts: the archive was downloaded and listed for the directory and
// binary names, and the flags come from the binary's own --help. The one thing
// that could not be verified here is the shape of its statistics API — see
// srbStatusEvents.

const SRB = {
  version: '3.5.4',
  algorithm: 'pearlhash',
  // Per the miner's own algorithm list: "[2.00%] [ - A N - ] pearlhash".
  // Surfaced in the UI rather than buried — a silent cut is the part of this
  // that would be indefensible.
  devFeePercent: 2,
  // Its statistics API, which we poll for hashrate and shares instead of
  // scraping the console. --api-port's documented default.
  apiPort: 21550,
  releaseBase: 'https://github.com/doktor83/SRBMiner-Multi/releases/download/',
};

// AlphaPool's plain-stratum port. 5566 speaks the pool's native protocol, which
// only alpha-miner talks; 5571 is standard stratum, added so third-party miners
// connect without a shim. Note this does not make a non-compliant miner
// compliant: rank is a property of the work computed, not the port.
const PLAIN_STRATUM_PORT = 5571;

// Verified by downloading 3.5.4 and listing it:
//   SRBMiner-Multi-3-5-4/            <- version with dots replaced by dashes
//   SRBMiner-Multi-3-5-4/SRBMiner-MULTI
// The archive is a .zip on Windows and a .tar.gz elsewhere, and like the
// alpha-miner package its top-level folder IS the install directory, so the
// extract must not strip components.
function srbArchiveName(platform) {
  const v = SRB.version.replace(/\./g, '-');
  return 'SRBMiner-Multi-' + v + (platform === 'win32' ? '-win64.zip' : '-Linux.tar.gz');
}

function srbDirName() {
  return 'SRBMiner-Multi-' + SRB.version.replace(/\./g, '-');
}

function srbBinaryName(platform) {
  return platform === 'win32' ? 'SRBMiner-MULTI.exe' : 'SRBMiner-MULTI';
}

function srbDownloadUrl(platform) {
  return SRB.releaseBase + SRB.version + '/' + srbArchiveName(platform);
}

// Absolute path to the miner inside an install dir, and every file that has to
// be present for the install to count. Only the one binary matters here — the
// archive also carries helper shell scripts, but nothing needs them.
function srbBinaryPath(dir, platform) {
  return path.join(dir, srbDirName(), srbBinaryName(platform));
}

function srbFiles(dir, platform) {
  return [srbBinaryPath(dir, platform)];
}

// Escape a Stratum password for SRBMiner's parser.
//
// This is the subtle one. Our static-difficulty password is `x;d=524288`, and
// SRBMiner documents `;` and `!` as SEPARATORS between per-pool passwords:
// "separate values with ; and ! [use #; and #! to escape separator
// characters]". Passed through raw, `x;d=524288` is read as two passwords for
// two pools, the difficulty pin is silently lost, and the rig quietly runs on
// vardiff instead. Escaping turns it into the single value the pool expects.
//
// `#` itself is not documented as needing escaping, so it is left alone rather
// than guessing at a doubling rule that may not exist.
function srbPassword(password) {
  return String(password == null ? '' : password).replace(/([;!])/g, '#$1');
}

// The plain-stratum endpoint for a region: same host, port 5571 rather than
// whatever the native endpoint uses.
function plainStratumEndpoint(endpoint) {
  const host = String(endpoint == null ? '' : endpoint).split(':')[0];
  return host ? host + ':' + PLAIN_STRATUM_PORT : '';
}

// Build SRBMiner's argument vector. Flag spellings are from the 3.5.4 binary's
// own --help, not from documentation that might lag the build:
//   --algorithm --pool --wallet --worker --password --disable-cpu
//   --api-enable --api-port
// --pool takes a bare `address:port` here, with no stratum+tcp:// scheme.
//
// --disable-cpu is deliberate: this app is about spare GPU capacity, and a
// miner that quietly also pins every core is not what anyone asked for.
function srbArgs(settings = {}) {
  const endpoint = plainStratumEndpoint(settings.endpoint);
  const address = String(settings.address == null ? '' : settings.address).trim();
  const args = [
    '--algorithm', SRB.algorithm,
    '--pool', endpoint,
    '--wallet', address,
  ];
  if (settings.worker) args.push('--worker', String(settings.worker));
  if (settings.password) args.push('--password', srbPassword(settings.password));
  args.push('--disable-cpu', '--api-enable', '--api-port', String(settings.apiPort || SRB.apiPort));
  return args;
}

// The URL its statistics API answers on.
function srbApiUrl(apiPort) {
  return 'http://127.0.0.1:' + (apiPort || SRB.apiPort) + '/';
}

// Pull a number out of the first key that exists, so one renamed field does not
// blank the whole row.
function pick(obj, keys) {
  for (const k of keys) {
    const v = obj[k];
    if (typeof v === 'number' && Number.isFinite(v)) return v;
    if (typeof v === 'string' && v.trim() !== '' && Number.isFinite(Number(v))) return Number(v);
  }
  return null;
}

function pickStr(obj, keys) {
  for (const k of keys) {
    const v = obj[k];
    if (typeof v === 'string' && v.trim() !== '') return v.trim();
  }
  return null;
}

// Map one statistics-API response onto the same `status` events parser.js emits
// for alpha-miner, so miningStats/the renderer/the reporter all stay unchanged.
//
// HONEST LIMITATION: the exact field names could not be verified. SRBMiner's
// binary is packed (no readable strings) and it exits silently on a machine
// with no supported GPU, so its API could not be observed from the development
// container — only its --help. The key lists below are therefore best-effort
// across the shapes SRBMiner is documented and reported to use.
//
// Consequently this returns [] rather than zeros when it recognises nothing:
// a blank "stats unavailable" is honest, whereas a confident 0 H/s next to a
// rig that is actually mining is the failure the pool dashboard just put a user
// through. Correcting this after one real run means editing these key lists and
// HASHRATE_TO_TH — nothing else.
//
// Units: our pipeline speaks TH/s (alpha-miner reports hashrate_th_s directly).
// SRBMiner reports raw hashes/sec, so the default divisor is 1e12. If a real
// run shows the rig reading 1e12x low, this constant is the single thing to fix.
const HASHRATE_TO_TH = 1e12;

function srbStatusEvents(json, toTh) {
  const div = toTh || HASHRATE_TO_TH;
  const root = json && typeof json === 'object' ? json : null;
  if (!root) return [];
  const devices = root.gpu_devices || root.devices || root.gpus
    || (Array.isArray(root.algorithms) && root.algorithms[0] && root.algorithms[0].gpu_devices)
    || null;
  if (!Array.isArray(devices) || devices.length === 0) return [];

  const events = [];
  for (let i = 0; i < devices.length; i++) {
    const d = devices[i] || {};
    const raw = pick(d, ['hashrate', 'hashrate_now', 'hashrate_instant', 'speed']);
    const idx = pick(d, ['device_id', 'id', 'index', 'gpu_id']);
    events.push({
      type: 'status',
      gpuIndex: idx == null ? i : idx,
      hashrate: raw == null ? null : raw / div,
      accepted: pick(d, ['accepted_shares', 'accepted', 'shares_accepted']),
      rejected: pick(d, ['rejected_shares', 'rejected', 'shares_rejected']),
      power: pick(d, ['power', 'power_usage', 'watts']),
      temp: pick(d, ['temperature', 'temp', 'core_temperature']),
      gpu: pickStr(d, ['device_name', 'name', 'gpu', 'model']),
    });
  }
  return events;
}

module.exports = {
  SRB,
  PLAIN_STRATUM_PORT,
  HASHRATE_TO_TH,
  srbArchiveName,
  srbDirName,
  srbBinaryName,
  srbDownloadUrl,
  srbBinaryPath,
  srbFiles,
  srbPassword,
  plainStratumEndpoint,
  srbArgs,
  srbApiUrl,
  srbStatusEvents,
};
