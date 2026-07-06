#!/usr/bin/env node
'use strict';

// Headless LLMJob Earn miner for Linux — the command-line counterpart to the
// Electron GUI. It shares all the real logic with the desktop app (config,
// address handling, engine download, argument building, the process supervisor
// and stats accumulator); this file is only the thin IO shell that wires the
// real filesystem / network / child_process around them, exactly like main.js
// does for the GUI. No window, no DOM — just stdout.

const { spawn, execFile } = require('child_process');
const path = require('path');
const os = require('os');
const fs = require('fs');
const net = require('net');
const http = require('http');
const https = require('https');

const { parseCliArgs, USAGE } = require('../shared/cliArgs');
const selfUpdater = require('./selfUpdater');
const { planUpdate } = require('../shared/selfUpdate');
const { MinerManager } = require('../main/minerManager');
const { EngineManager } = require('../main/engineManager');
const { initStats, applyEvent, snapshot } = require('../shared/miningStats');
const { NETWORK, MINER, REGIONS, DEFAULTS, endpointFor, regionLabel, difficultyForCard } = require('../shared/config');
const { progressPercent } = require('../shared/engine');
const { buildMinerReport } = require('../shared/minerReport');
const { shortenAddress } = require('../shared/address');
const { pickFastestRegion } = require('../shared/region');
const { pickGpu } = require('../shared/gpu');
const format = require('../shared/format');
const pkg = require('../../package.json');

// Write a log line. When attached to a TTY we prefix a wall-clock time; when
// piped (systemd/journald, `docker logs`, a file) we drop it, since the log
// collector adds its own timestamp and two would just be noise.
function log(line, stream) {
  const out = stream || process.stdout;
  const prefix = out.isTTY ? '[' + format.formatLogTime(new Date()) + '] ' : '';
  out.write(prefix + line + '\n');
}

// Measure TCP connect latency (ms) to a "host:port" Stratum endpoint, or null if
// it can't be reached within the timeout. Never rejects. (Mirrors main.js.)
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

// Auto-detect the lowest-latency pool region by pinging every endpoint in
// parallel; falls back to the default when nothing is reachable.
async function detectRegion() {
  const keys = Object.keys(REGIONS);
  const results = await Promise.all(keys.map((region) =>
    pingEndpoint(REGIONS[region].endpoint).then((ms) => ({ region, ms }))));
  return pickFastestRegion(results, DEFAULTS.region);
}

// Default worker name = this machine's hostname (first DNS label, sanitised to a
// safe stratum token). Unlike a shared constant like "rig01", this keeps two
// rigs mining the same payout address as distinct workers on the board instead
// of colliding into one flip-flopping entry. Falls back to the default constant
// if the hostname is empty/unusable.
function defaultWorker() {
  const host = String(os.hostname() || '').trim().toLowerCase().split('.')[0];
  const name = host.replace(/[^a-z0-9-]/g, '').replace(/^-+|-+$/g, '').slice(0, 32);
  return name || DEFAULTS.worker;
}

// Detect the discrete GPU name via nvidia-smi (Linux/NVIDIA). Resolves the card
// name or null (no nvidia-smi / non-NVIDIA / parse failure). Never rejects — the
// engine still auto-detects the real device to mine; this is only for the
// difficulty table and the status label.
function detectGpu() {
  return new Promise((resolve) => {
    execFile('nvidia-smi', ['--query-gpu=name', '--format=csv,noheader'],
      { timeout: 5000 },
      (err, stdout) => resolve(err ? null : pickGpu(String(stdout).split(/\r?\n/))));
  });
}

// Live GPU VRAM (used/total, MB) via nvidia-smi — mirrors the GUI (main.js) so
// the CLI reports VRAM to the public board too. Resolves { usedMb, totalMb } or
// null (no nvidia-smi / non-NVIDIA / parse failure). Never rejects.
function detectVram() {
  return new Promise((resolve) => {
    execFile('nvidia-smi',
      ['--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
      { timeout: 5000 },
      (err, stdout) => {
        if (err) return resolve(null);
        const parts = String(stdout).split(/\r?\n/)[0].split(',').map((x) => parseInt(x, 10));
        if (parts.length < 2 || !Number.isFinite(parts[0]) || !Number.isFinite(parts[1])) return resolve(null);
        resolve({ usedMb: parts[0], totalMb: parts[1] });
      });
  });
}

// Stream a URL to a file, following redirects and reporting download progress.
// Mirrors the GUI's downloader (main.js) — kept here so the CLI has no runtime
// dependency on Electron.
function downloadFile(url, dest, onProgress, redirects) {
  redirects = redirects || 0;
  return new Promise((resolve, reject) => {
    if (redirects > 5) return reject(new Error('too many redirects'));
    const lib = url.startsWith('https') ? https : http;
    const req = lib.get(url, (res) => {
      const code = res.statusCode || 0;
      if (code >= 300 && code < 400 && res.headers.location) {
        res.resume();
        return resolve(downloadFile(new URL(res.headers.location, url).toString(), dest, onProgress, redirects + 1));
      }
      if (code !== 200) {
        res.resume();
        return reject(new Error('HTTP ' + code + ' for ' + url));
      }
      const total = parseInt(res.headers['content-length'] || '0', 10);
      let received = 0;
      const out = fs.createWriteStream(dest);
      res.on('data', (c) => { received += c.length; if (onProgress) onProgress(progressPercent(received, total)); });
      res.pipe(out);
      out.on('finish', () => out.close(() => resolve(dest)));
      out.on('error', reject);
    });
    req.on('error', reject);
  });
}

// Publish live status to the public network board — best-effort, never throws.
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

// Zip extraction is Windows-only (the pool ships Linux as a bare binary), so the
// CLI never needs it — EngineManager only calls extract for a .zip URL.
function extractUnsupported() {
  return Promise.reject(new Error('zip extraction is not supported on the Linux CLI'));
}

async function resolveEngine(settings) {
  if (settings.binaryPath) {
    if (!fs.existsSync(settings.binaryPath)) {
      throw new Error('engine binary not found: ' + settings.binaryPath);
    }
    return settings.binaryPath;
  }

  const dir = settings.engineDir || path.join(os.homedir(), '.local', 'share', 'llmjob-earn', 'engine');
  const engine = new EngineManager({
    dir,
    platform: process.platform,
    gpu: settings.gpu,
    fs,
    download: downloadFile,
    extract: extractUnsupported,
    chmod: fs.chmodSync,
  });

  if (engine.isInstalled()) {
    log('engine found: ' + engine.binaryPath());
  } else {
    log('downloading mining engine from ' + MINER.downloadUrl + ' …');
  }
  const binaryPath = await engine.ensure((pct) => {
    if (pct != null) process.stdout.write('\r  downloading… ' + pct + '%   ');
  });
  process.stdout.write('\n');
  log('engine ready: ' + binaryPath);
  return binaryPath;
}

// Explicit `llmjob-earn-cli update` — check the latest release and, if this is
// the packaged binary, replace it in place.
async function runExplicitUpdate() {
  log('checking for updates (current v' + pkg.version + ')…');
  const release = await selfUpdater.fetchLatestRelease();
  if (!release) { log('could not reach the update server', process.stderr); return 1; }

  const plan = planUpdate({ currentVersion: pkg.version, release, platform: process.platform });
  if (!plan.updateAvailable) {
    if (plan.reason === 'up-to-date') log('already up to date (v' + pkg.version + ')');
    else if (plan.reason === 'asset-missing') log('v' + plan.latestVersion + ' is out but has no Linux CLI binary yet', process.stderr);
    else if (plan.reason === 'unsupported-platform') log('self-update is only available for the Linux binary', process.stderr);
    else log('no newer release found', process.stderr);
    return 0;
  }

  if (!selfUpdater.isPackaged()) {
    log('v' + plan.latestVersion + ' is available (you have v' + plan.currentVersion + ').');
    log('this is running from source — update via git/npm, or download: ' + plan.downloadUrl);
    return 0;
  }

  log('updating ' + plan.currentVersion + ' → ' + plan.latestVersion + ' …');
  try {
    const exe = await selfUpdater.applyUpdate(plan);
    log('updated to v' + plan.latestVersion + ' (' + exe + '). Re-run to use it.');
    return 0;
  } catch (e) {
    log('update failed: ' + e.message, process.stderr);
    return 1;
  }
}

// Best-effort auto-update on start. Returns an exit code when it replaced and
// re-ran the binary (caller should return it), or null to keep mining.
async function maybeAutoUpdate(argv) {
  if (process.env[selfUpdater.UPDATED_ENV]) return null; // already the updated child
  const release = await selfUpdater.fetchLatestRelease();
  if (!release) return null; // offline — never block mining

  const plan = planUpdate({ currentVersion: pkg.version, release, platform: process.platform });
  if (!plan.updateAvailable) return null;

  if (!selfUpdater.isPackaged()) {
    log('a newer release is available: v' + plan.latestVersion + ' (run "llmjob-earn-cli update")');
    return null;
  }

  log('updating ' + plan.currentVersion + ' → ' + plan.latestVersion + ' before starting…');
  try {
    await selfUpdater.applyUpdate(plan);
    log('updated to v' + plan.latestVersion + '; restarting');
    return selfUpdater.reexec(argv);
  } catch (e) {
    log('auto-update failed (' + e.message + '); continuing on v' + pkg.version, process.stderr);
    return null;
  }
}

async function run(argv) {
  if (argv[0] === 'update') return runExplicitUpdate();

  const parsed = parseCliArgs(argv);

  if (parsed.help) { process.stdout.write(USAGE + '\n'); return 0; }
  if (parsed.version) { process.stdout.write(pkg.version + '\n'); return 0; }

  if (parsed.errors.length) {
    for (const e of parsed.errors) log('error: ' + e, process.stderr);
    log('run with --help for usage', process.stderr);
    return 1;
  }

  const settings = parsed.settings;

  if (settings.update) {
    const code = await maybeAutoUpdate(argv);
    if (code != null) return code;
  }

  // Auto-detect the knobs the user didn't pin. Best-effort: any failure falls
  // back to the defaults already in `settings` and never blocks mining. Explicit
  // --region / --gpu / --difficulty always win.
  if (!settings.workerProvided) settings.worker = defaultWorker();
  if (!settings.regionProvided) settings.region = await detectRegion();
  if (!settings.gpuProvided) {
    const gpu = await detectGpu();
    if (gpu) {
      settings.gpu = gpu;
      if (!settings.difficultyProvided) settings.difficulty = difficultyForCard(gpu);
    }
  }

  const endpoint = endpointFor(settings.region);

  log('LLMJob Earn CLI v' + pkg.version);
  log('address:    ' + shortenAddress(settings.address) + (settings.mdlAddress ? '  (+MDL ' + shortenAddress(settings.mdlAddress) + ')' : ''));
  log('pool:       ' + endpoint + '  ' + regionLabel(settings.region) + (settings.regionProvided ? '' : '  (auto)'));
  log('worker:     ' + settings.worker + (settings.workerProvided ? '' : '  (auto)'));
  log('difficulty: ' + settings.difficulty + (settings.gpu ? '  (for ' + settings.gpu + (settings.gpuProvided ? '' : ', auto') + ')' : ''));

  let binaryPath;
  try {
    binaryPath = await resolveEngine(settings);
  } catch (e) {
    log('engine setup failed: ' + e.message, process.stderr);
    log('manual download: ' + MINER.downloadUrl, process.stderr);
    return 1;
  }

  const stats = initStats(Date.now());
  const miner = new MinerManager({ spawn });
  let reporter = null;
  let stopping = false;

  miner.on('started', ({ bin, args }) => {
    log('starting: ' + bin + ' ' + args.join(' '));
  });
  miner.on('log', (l) => log(l.line, l.level === 'error' ? process.stderr : process.stdout));
  miner.on('event', (evt) => {
    applyEvent(stats, evt);
    if (evt.type === 'status') {
      const snap = snapshot(stats, Date.now());
      log('⛏  ' + format.formatHashrate(snap.total) + ' TH/s · '
        + format.formatInt(snap.accepted) + ' accepted · ' + snap.rejected + ' rejected · up '
        + format.formatUptime(snap.uptimeSec));
    }
  });
  miner.on('error', (err) => log('engine error: ' + err.message, process.stderr));

  if (settings.report) {
    // Sample live VRAM (nvidia-smi) and fold it into the snapshot before posting,
    // just like the GUI — otherwise the board shows 0 GB for a CLI-driven rig.
    const report = async () => {
      const snap = snapshot(stats, Date.now());
      const vram = await detectVram();
      if (vram) { snap.vramUsedMb = vram.usedMb; snap.vramTotalMb = vram.totalMb; }
      return postMinerReport(buildMinerReport(settings, snap));
    };
    report();
    reporter = setInterval(report, NETWORK.reportIntervalMs);
    if (reporter.unref) reporter.unref();
  }

  const shutdown = () => {
    if (stopping) return;
    stopping = true;
    log('shutting down…');
    if (reporter) clearInterval(reporter);
    miner.stop();
  };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);

  return new Promise((resolve) => {
    miner.on('stopped', (code) => {
      if (reporter) clearInterval(reporter);
      log('engine exited (code ' + code + ')');
      resolve(stopping ? 0 : (code || 0));
    });
    try {
      miner.start(Object.assign({}, settings, { platform: process.platform, binaryPath }));
    } catch (e) {
      log('failed to launch engine: ' + e.message, process.stderr);
      resolve(1);
    }
  });
}

/* istanbul ignore next */
if (require.main === module) {
  run(process.argv.slice(2)).then((code) => { process.exitCode = code; }).catch((e) => {
    log('fatal: ' + (e && e.message ? e.message : e), process.stderr);
    process.exitCode = 1;
  });
}

module.exports = { run, downloadFile, postMinerReport, resolveEngine };
