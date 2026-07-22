'use strict';

// Host- and GPU-probing IO helpers shared by the Electron main process
// (main.js) and the headless CLI (earn-cli.js). Both shells previously kept
// byte-for-byte copies of these nvidia-smi / socket wrappers, so a fix to VRAM
// summing or region timing had to be made twice. Keeping them here means it
// lands in both.
//
// Each function shells out or opens a socket. The low-level primitives
// (execFile / net / http / https) are injectable via a trailing `deps` argument
// so the branching can be unit-tested without a real GPU or network; callers in
// the app pass nothing and get the real modules.

const net = require('net');
const http = require('http');
const https = require('https');
const { execFile } = require('child_process');

const { REGIONS, DEFAULTS, NETWORK } = require('../shared/config');
const { pickFastestRegion } = require('../shared/region');
const { parseGpuStats } = require('../shared/gpu');
const { parseDriverMajor } = require('../shared/engine');

// Measure TCP connect latency (ms) to a "host:port" Stratum endpoint, or null
// if it can't be reached within the timeout. Never rejects.
function pingEndpoint(endpoint, timeoutMs, deps = {}) {
  const netlib = deps.net || net;
  const now = deps.now || Date.now;
  return new Promise((resolve) => {
    const [host, portStr] = String(endpoint).split(':');
    const start = now();
    const sock = new netlib.Socket();
    let settled = false;
    const done = (ms) => { if (!settled) { settled = true; sock.destroy(); resolve(ms); } };
    sock.setTimeout(timeoutMs || 2500);
    sock.once('connect', () => done(now() - start));
    sock.once('timeout', () => done(null));
    sock.once('error', () => done(null));
    sock.connect(Number(portStr), host);
  });
}

// Auto-detect the lowest-latency pool region by pinging every region's endpoint
// in parallel; falls back to the default when nothing is reachable.
async function detectRegion(deps = {}) {
  const keys = Object.keys(REGIONS);
  const results = await Promise.all(keys.map((region) =>
    pingEndpoint(REGIONS[region].endpoint, undefined, deps).then((ms) => ({ region, ms }))));
  return pickFastestRegion(results, DEFAULTS.region);
}

// Live GPU VRAM (used/total, MB) via nvidia-smi — NVIDIA-only. Resolves
// { usedMb, totalMb } or null (no nvidia-smi / non-NVIDIA / parse failure). Sums
// across every GPU line so a multi-GPU rig reports the rig's total; both shells
// must agree on free-VRAM decisions. Never rejects.
function detectVram(deps = {}) {
  const exec = deps.execFile || execFile;
  return new Promise((resolve) => {
    exec('nvidia-smi',
      ['--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
      { timeout: 5000 },
      (err, stdout) => {
        if (err) return resolve(null);
        let usedMb = 0;
        let totalMb = 0;
        let any = false;
        for (const row of String(stdout).split(/\r?\n/)) {
          const parts = row.split(',').map((x) => parseInt(x, 10));
          if (parts.length >= 2 && Number.isFinite(parts[0]) && Number.isFinite(parts[1])) {
            usedMb += parts[0];
            totalMb += parts[1];
            any = true;
          }
        }
        resolve(any ? { usedMb, totalMb } : null);
      });
  });
}

// Per-card live VRAM (used/total, MB) via nvidia-smi — one entry per GPU so the
// network board reports each card's own headroom rather than the rig's sum.
// Resolves [{ index, name, usedMb, totalMb }] (empty on failure). Never rejects.
function detectGpusVram(deps = {}) {
  const exec = deps.execFile || execFile;
  return new Promise((resolve) => {
    exec('nvidia-smi',
      ['--query-gpu=index,name,memory.used,memory.total', '--format=csv,noheader,nounits'],
      { timeout: 5000 },
      (err, stdout) => resolve(err ? [] : parseGpuStats(stdout)));
  });
}

// NVIDIA driver major version via nvidia-smi, or null when it can't be read.
// Decides which engine build the rig can run (CUDA 13 builds need >= 580).
function detectDriverMajor(deps = {}) {
  const exec = deps.execFile || execFile;
  return new Promise((resolve) => {
    exec('nvidia-smi', ['--query-gpu=driver_version', '--format=csv,noheader'],
      { timeout: 5000 },
      (err, stdout) => resolve(err ? null : parseDriverMajor(stdout)));
  });
}

// Publish this miner's live status to the network board (best-effort — never
// throws; timeouts and errors are swallowed so mining is never affected).
function postMinerReport(payload, deps = {}) {
  return new Promise((resolve) => {
    try {
      const body = JSON.stringify(payload);
      const u = new URL(NETWORK.reportUrl);
      const isHttp = u.protocol === 'http:';
      const lib = (isHttp ? deps.http : deps.https) || (isHttp ? http : https);
      const req = lib.request(u, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
        timeout: 8000,
      }, (res) => { res.resume(); res.on('end', resolve); });
      req.on('error', () => resolve());
      req.on('timeout', () => { req.destroy(); resolve(); });
      req.write(body);
      req.end();
    } catch (e) {
      resolve();
    }
  });
}

// Find a port llama-server can actually bind, preferring `start` (8080), walking
// forward until a bind succeeds. On Windows a just-killed server's port can stay
// unavailable for 30s+, and other software may own 8080 outright; rather than
// fail on a fixed port, try the next few. Falls back to `start` if none is free.
function findFreePort(host, start, tries, deps = {}) {
  const netlib = deps.net || net;
  const attempt = (port, left) => new Promise((resolve) => {
    if (left <= 0) return resolve(start); // give up: fall back to the default
    const srv = netlib.createServer();
    srv.once('error', () => { srv.close(); resolve(attempt(port + 1, left - 1)); });
    srv.once('listening', () => srv.close(() => resolve(port)));
    srv.listen(port, host);
  });
  return attempt(start, tries || 10);
}

module.exports = {
  pingEndpoint,
  detectRegion,
  detectVram,
  detectGpusVram,
  detectDriverMajor,
  postMinerReport,
  findFreePort,
};
