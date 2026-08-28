'use strict';

// Host- and GPU-probing IO helpers shared by the Electron main process
// (main.js) and the headless CLI (earn-cli.js). Both shells previously kept
// byte-for-byte copies of these nvidia-smi / socket wrappers, so a fix to VRAM
// summing or region timing had to be made twice. Keeping them here means it
// lands in both. Thin wrappers over net/http/https/child_process — unit-tested
// by mocking those core modules.

const net = require('net');
const http = require('http');
const https = require('https');
const { execFile } = require('child_process');

const { REGIONS, DEFAULTS, NETWORK } = require('../shared/config');
const { pickFastestRegion } = require('../shared/region');
const { parseGpuStats, pickGpu, countGpus, parseMacGpu } = require('../shared/gpu');
// Major version out of an nvidia-smi driver string. Lived in shared/engine
// until alpha-miner was removed; the driver version is a property of the
// machine, not of any engine.
function parseDriverMajor(output) {
  const m = String(output == null ? '' : output).match(/(\d+)\.\d+/);
  return m ? parseInt(m[1], 10) : null;
}

// Measure TCP connect latency (ms) to a "host:port" Stratum endpoint, or null
// if it can't be reached within the timeout. Never rejects.
function pingEndpoint(endpoint, timeoutMs) {
  return new Promise((resolve) => {
    const [host, portStr] = String(endpoint).split(':');
    const start = Date.now();
    const sock = new net.Socket();
    let settled = false;
    const done = (ms) => { if (!settled) { settled = true; sock.destroy(); resolve(ms); } };
    sock.setTimeout(timeoutMs || 2500);
    sock.once('connect', () => done(Date.now() - start));
    sock.once('timeout', () => done(null));
    sock.once('error', () => done(null));
    sock.connect(Number(portStr), host);
  });
}

// Auto-detect the lowest-latency pool region by pinging every region's endpoint
// in parallel; falls back to the default when nothing is reachable.
async function detectRegion() {
  const keys = Object.keys(REGIONS);
  const results = await Promise.all(keys.map((region) =>
    pingEndpoint(REGIONS[region].endpoint).then((ms) => ({ region, ms }))));
  return pickFastestRegion(results, DEFAULTS.region);
}

// Live GPU VRAM (used/total, MB) via nvidia-smi — NVIDIA-only. Resolves
// { usedMb, totalMb } or null (no nvidia-smi / non-NVIDIA / parse failure). Sums
// across every GPU line so a multi-GPU rig reports the rig's total; both shells
// must agree on free-VRAM decisions. Never rejects.
function detectVram() {
  return new Promise((resolve) => {
    execFile('nvidia-smi',
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
function detectGpusVram() {
  return new Promise((resolve) => {
    execFile('nvidia-smi',
      ['--query-gpu=index,name,memory.used,memory.total', '--format=csv,noheader,nounits'],
      { timeout: 5000 },
      (err, stdout) => resolve(err ? [] : parseGpuStats(stdout)));
  });
}

// Per-card core temperature (°C) via nvidia-smi, as a map of card index →
// degrees. Resolves {} on any failure (no nvidia-smi, non-NVIDIA, unparseable),
// never rejects, so a rig without it simply shows no temperature rather than a
// wrong one.
//
// Our miner is the CUDA core rather than a scraped process, so unlike
// alpha-miner it has no NVML reading of its own to forward. This is where the
// number comes from instead. It is a spawn, so PearlEngine samples it on a slow
// timer rather than per status event — those fire on every share and every
// hashrate tick.
function detectGpuTemps() {
  return new Promise((resolve) => {
    execFile('nvidia-smi',
      ['--query-gpu=index,temperature.gpu', '--format=csv,noheader,nounits'],
      { timeout: 5000 },
      (err, stdout) => {
        if (err) return resolve({});
        const out = {};
        for (const row of String(stdout).split(/\r?\n/)) {
          const parts = row.split(',').map((x) => parseInt(x, 10));
          // A card that reports "N/A" parses to NaN; skip it rather than
          // recording a zero the UI would render as a real reading.
          if (parts.length >= 2 && Number.isFinite(parts[0]) && Number.isFinite(parts[1])) {
            out[parts[0]] = parts[1];
          }
        }
        resolve(out);
      });
  });
}

// NVIDIA driver major version via nvidia-smi, or null when it can't be read.
// Decides which engine build the rig can run (CUDA 13 builds need >= 580).
function detectDriverMajor() {
  return new Promise((resolve) => {
    execFile('nvidia-smi', ['--query-gpu=driver_version', '--format=csv,noheader'],
      { timeout: 5000 },
      (err, stdout) => resolve(err ? null : parseDriverMajor(stdout)));
  });
}

// Publish this miner's live status to the network board (best-effort — never
// throws; timeouts and errors are swallowed so mining is never affected).
function postMinerReport(payload) {
  return new Promise((resolve) => {
    try {
      const body = JSON.stringify(payload);
      const u = new URL(NETWORK.reportUrl);
      const lib = u.protocol === 'http:' ? http : https;
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
function findFreePort(host, start, tries) {
  const attempt = (port, left) => new Promise((resolve) => {
    if (left <= 0) return resolve(start); // give up: fall back to the default
    const srv = net.createServer();
    srv.once('error', () => { srv.close(); resolve(attempt(port + 1, left - 1)); });
    srv.once('listening', () => srv.close(() => resolve(port)));
    srv.listen(port, host);
  });
  return attempt(start, tries || 10);
}

// The machine's GPU, as { name, count } — the representative card plus how many
// discrete GPUs the rig has. Resolves null when nothing can be identified.
//
// This is the last of the nvidia-smi wrappers to move here, and the reason it
// mattered: both shells had their own copy and they had diverged into DIFFERENT
// DETECTION METHODS. The GUI's copy asked Windows' Win32_VideoController via
// PowerShell and returned null on every other platform, so on the shipped Linux
// AppImage the device label was blank and the per-card difficulty lookup never
// ran — a Linux rig silently mined at the default difficulty. The CLI's copy
// used nvidia-smi and worked there. Now both get the union: nvidia-smi first
// (NVIDIA on any platform, which is what a mining rig runs), then WMI on Windows
// so a non-NVIDIA card still gets a name.
// The Mac's GPU as { name, count } — "Apple M3 Max" — or null. Never rejects.
//
// system_profiler is the canonical source (it is what About This Mac reads),
// and it is the only one that gets an Intel Mac's discrete card right. It is
// also slow and its JSON has been reshaped between macOS versions, so a failure
// falls back to the CPU brand string rather than giving up: on Apple silicon
// the SoC name IS the GPU name, so that fallback is the same answer by another
// route, not a guess. Only when both fail does the label stay unknown.
function detectMacGpuInfo() {
  return new Promise((resolve) => {
    execFile('system_profiler', ['SPDisplaysDataType', '-json'], { timeout: 10000 },
      (err, stdout) => {
        const info = err ? null : parseMacGpu(stdout);
        if (info) return resolve(info);
        execFile('sysctl', ['-n', 'machdep.cpu.brand_string'], { timeout: 5000 },
          (e2, out2) => {
            const name = e2 ? null : pickGpu([out2]);
            resolve(name ? { name, count: 1 } : null);
          });
      });
  });
}

function detectGpuInfo() {
  return new Promise((resolve) => {
    execFile('nvidia-smi', ['--query-gpu=name', '--format=csv,noheader'],
      { timeout: 5000 },
      (err, stdout) => {
        if (!err) {
          const names = String(stdout).split(/\r?\n/);
          const name = pickGpu(names);
          if (name) return resolve({ name, count: countGpus(names) });
        }
        // No nvidia-smi (not installed, not on PATH, or no NVIDIA card).
        // macOS has no nvidia-smi and no WMI either, so it needs its own probe —
        // otherwise the Mac build, whose whole job is running the model on that
        // GPU, reports no device at all.
        if (process.platform === 'darwin') return resolve(detectMacGpuInfo());
        if (process.platform !== 'win32') return resolve(null);
        execFile('powershell.exe',
          ['-NoProfile', '-NonInteractive', '-Command',
            'Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name'],
          { timeout: 5000 },
          (e2, out2) => {
            if (e2) return resolve(null);
            const names = String(out2).split(/\r?\n/);
            const name = pickGpu(names);
            resolve(name ? { name, count: countGpus(names) } : null);
          });
      });
  });
}

module.exports = {
  parseDriverMajor,
  pingEndpoint,
  detectRegion,
  detectVram,
  detectGpusVram,
  detectGpuTemps,
  detectDriverMajor,
  postMinerReport,
  findFreePort,
  detectGpuInfo,
};
