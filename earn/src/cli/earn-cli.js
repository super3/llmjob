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
const { LlmManager } = require('../main/llmManager');
const { LlmEngineManager } = require('../main/llmEngineManager');
const { postJson, downloadFile, streamChatCompletion, extractLlamaZip } = require('../main/io');
const nodeStore = require('../main/nodeStore');
const { initStats, applyEvent, snapshot } = require('../shared/miningStats');
const { NETWORK, MINER, LLM, NODE, REGIONS, DEFAULTS, endpointFor, regionLabel, difficultyForCard } = require('../shared/config');
const nodeProto = require('../shared/node');
const { buildMinerReports } = require('../shared/minerReport');
const { statsFilePayload } = require('../shared/statsFile');
const { shortenAddress, isValidAddress } = require('../shared/address');
const { pickFastestRegion } = require('../shared/region');
const { pickGpu, countGpus, parseGpuStats } = require('../shared/gpu');
const { computeGpuLayers, requiredVramMb, hasEnoughVram } = require('../shared/vram');
const { JobWorker } = require('../main/jobWorker');
const { resolvePlan } = require('../shared/llmMode');
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
// Resolve { name, count } — the representative card plus how many discrete
// GPUs the rig actually mines with (multi-GPU rigs scale difficulty by count).
function detectGpu() {
  return new Promise((resolve) => {
    execFile('nvidia-smi', ['--query-gpu=name', '--format=csv,noheader'],
      { timeout: 5000 },
      (err, stdout) => {
        if (err) return resolve(null);
        const names = String(stdout).split(/\r?\n/);
        resolve({ name: pickGpu(names), count: countGpus(names) });
      });
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
        // Sum across every GPU line so a multi-GPU rig reports the rig's total.
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
// network board can show each card's own headroom rather than the rig's sum.
// Resolves [{ index, name, usedMb, totalMb }] (empty on no nvidia-smi / parse
// failure). Never rejects.
function detectGpusVram() {
  return new Promise((resolve) => {
    execFile('nvidia-smi',
      ['--query-gpu=index,name,memory.used,memory.total', '--format=csv,noheader,nounits'],
      { timeout: 5000 },
      (err, stdout) => resolve(err ? [] : parseGpuStats(stdout)));
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

// The mining engine ships as a bare Linux binary (no zip), so its EngineManager
// never needs an extractor — this stays a hard "unsupported" for that path.
function extractUnsupported() {
  return Promise.reject(new Error('zip extraction is not supported on the Linux CLI'));
}

// llama-server zips extract via the shared io helper; point failures at the
// CLI's escape hatch.
const LLM_UNZIP_HINT = 'install unzip, or pass --llm-binary </path/to/llama-server>';

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

// Where the CLI caches the local-LLM binary + model (mirrors the engine dir).
function llmDir(settings) {
  return settings.llmDir || path.join(os.homedir(), '.local', 'share', 'llmjob-earn', 'llm');
}

// Resolve the llama-server binary for the local LLM. An explicit --llm-binary
// wins; otherwise fall back to a previously installed one in the cache dir, and
// only then download the llama.cpp release zip and extract it (via unzip). If
// extraction isn't possible (no `unzip`), we surface a clear error pointing at
// --llm-binary as the escape hatch.
async function resolveLlmBinary(settings, dir) {
  if (settings.llmBinary) {
    if (!fs.existsSync(settings.llmBinary)) {
      throw new Error('llama-server binary not found: ' + settings.llmBinary);
    }
    return settings.llmBinary;
  }
  const engine = new LlmEngineManager({
    dir, platform: process.platform, serverUrl: LLM.serverUrl[process.platform],
    fs, download: downloadFile, extract: (zip, dest) => extractLlamaZip(zip, dest, LLM_UNZIP_HINT), chmod: fs.chmodSync,
  });
  if (engine.isServerInstalled()) {
    log('LLM server found: ' + engine.serverBinaryPath());
    return engine.serverBinaryPath();
  }
  log('downloading llama-server from ' + LLM.serverUrl[process.platform] + ' …');
  try {
    return await engine.ensureServer((pct) => {
      if (pct != null) process.stdout.write('\r  downloading… ' + pct + '%   ');
    });
  } catch (e) {
    throw new Error(e.message + ' — pass --llm-binary </path/to/llama-server> instead');
  } finally {
    process.stdout.write('\n');
  }
}

// Resolve the GGUF model path. An explicit --llm-model wins; otherwise reuse a
// cached download or fetch the small default model (a plain file, so this works
// on the CLI without zip extraction).
async function resolveLlmModel(settings, dir) {
  if (settings.llmModel) {
    if (!fs.existsSync(settings.llmModel)) {
      throw new Error('LLM model not found: ' + settings.llmModel);
    }
    return settings.llmModel;
  }
  const engine = new LlmEngineManager({ dir, platform: process.platform, fs, download: downloadFile });
  if (engine.isModelInstalled()) {
    log('LLM model found: ' + engine.modelPath());
    return engine.modelPath();
  }
  log('downloading LLM model (' + LLM.model.name + ') …');
  const modelPath = await engine.ensureModel((pct) => {
    if (pct != null) process.stdout.write('\r  downloading model… ' + pct + '%   ');
  });
  process.stdout.write('\n');
  return modelPath;
}

// ── Cluster serving state (worker + keep-alive pings while the LLM is up) ────
// Serving and pinging must live in the SAME process: only /api/nodes/ping
// updates last_seen server-side, so a worker that polls jobs without pinging
// gets marked offline (15 min) and eventually deleted — silently starving it.
let cliJobWorker = null;
let servePinger = null;
let serveLlmState = { ready: false, tps: 0 };

// The GPU name is static — probe it once and reuse, instead of spawning
// nvidia-smi twice per ping forever.
let cliGpuProbed = false;
let cliGpuName = null;
async function cachedDeviceName() {
  if (!cliGpuProbed) {
    cliGpuProbed = true;
    try { const det = await detectGpu(); cliGpuName = det && det.name ? det.name : null; } catch (e) { cliGpuName = null; }
  }
  return cliGpuName;
}

// One signed ping. `telemetry` may be sparse — fields left undefined keep the
// server's stored values (its pick() only overwrites defined fields).
async function pingServer(node, base, telemetry, verbose) {
  const body = nodeProto.buildPingBody({
    nodeId: node.nodeId, publicKey: node.publicKey, secretKey: node.secretKey,
    timestamp: Date.now(), telemetry,
  });
  try {
    const res = await postJson(base + '/api/nodes/ping', body, 15000);
    if (verbose) {
      log(res.status === 200 ? '✓ ping' : ('✗ ping failed (HTTP ' + res.status + ')'),
        res.status === 200 ? process.stdout : process.stderr);
    }
  } catch (e) {
    if (verbose) log('✗ ping error: ' + e.message, process.stderr);
  }
}

// Sparse telemetry for the standalone `connect` loop: just device + VRAM (+ the
// node's name so renames propagate). model/capabilities/tps are omitted —
// sending nulls would wipe what a serving process last reported.
async function sparseTelemetry(node) {
  const t = {};
  try {
    const vram = await detectVram();
    if (vram) { t.vramTotal = vram.totalMb; t.vramUsed = vram.usedMb; }
  } catch (e) { /* ignore */ }
  const device = await cachedDeviceName();
  if (device) t.device = device;
  if (node.name) t.name = node.name;
  return t;
}

// Full telemetry for the serving run: live model/readiness/tok-s/active jobs.
async function fullTelemetry(node) {
  let vram = null;
  try { vram = await detectVram(); } catch (e) { /* ignore */ }
  return nodeProto.buildTelemetry({
    model: LLM.model.name, quant: LLM.model.quant,
    device: await cachedDeviceName(), vram,
    tokensPerSec: serveLlmState.tps, ready: serveLlmState.ready,
    activeJobs: cliJobWorker ? cliJobWorker.activeJobs() : 0,
    name: node.name,
  });
}

function stopServe() {
  if (cliJobWorker) { cliJobWorker.stop(); cliJobWorker = null; }
  if (servePinger) { clearInterval(servePinger); servePinger = null; }
}

// Start the local LLM (llama.cpp llama-server) alongside — or instead of — the
// miner. Ensures the binary + model, sizes the GPU offload from free VRAM
// (keeping `reserveMb` free for mining), spawns the server, and logs its
// OpenAI-compatible endpoint. Returns the LlmManager, or null if setup failed
// (best-effort — a failing LLM never takes the miner down).
async function startLlm(settings, reserveMb) {
  const dir = llmDir(settings);

  // Preflight the GPU before doing anything expensive (downloading a ~5 GB
  // model): refuse to start if we can measure VRAM and there isn't enough free
  // to hold the model — spawning anyway risks an out-of-memory crash. When VRAM
  // can't be read (non-NVIDIA / no driver) we proceed and let llama.cpp decide.
  const vram = await detectVram();
  const freeMb = vram ? vram.totalMb - vram.usedMb : null;
  if (hasEnoughVram(freeMb, LLM.model) === false) {
    log('not enough free VRAM for the local LLM: ' + freeMb + ' MB free, need ~'
      + requiredVramMb(LLM.model) + ' MB for ' + LLM.model.name + ' — skipping the LLM.', process.stderr);
    return null;
  }

  log('preparing local LLM (' + LLM.model.name + ') …');

  let binaryPath, modelPath;
  try {
    binaryPath = await resolveLlmBinary(settings, dir);
    modelPath = await resolveLlmModel(settings, dir);
  } catch (e) {
    log('LLM setup failed: ' + e.message, process.stderr);
    return null;
  }

  // Size the GPU offload from the free VRAM measured above (total − used); full
  // offload when VRAM can't be read (non-NVIDIA / no driver).
  const nGpuLayers = vram
    ? computeGpuLayers(vram.totalMb - vram.usedMb, LLM.model, reserveMb || 0)
    : LLM.model.layers;

  // If this box is linked to an account, serve cluster jobs once the model is up.
  const nodeCfg = loadNodeConfig();
  const canServe = !!(nodeCfg && nodeCfg.connected);

  const llm = new LlmManager({ spawn });
  llm.on('log', (l) => log(l.line, l.level === 'error' ? process.stderr : process.stdout));
  llm.on('ready', ({ baseUrl }) => {
    serveLlmState.ready = true;
    log('🧠 local LLM ready — OpenAI endpoint ' + baseUrl + '/v1');
    if (canServe && !cliJobWorker) {
      const base = nodeCfg.serverUrl || NODE.serverUrl;
      cliJobWorker = new JobWorker({
        identity: { nodeId: nodeCfg.nodeId, publicKey: nodeCfg.publicKey, secretKey: nodeCfg.secretKey },
        serverUrl: base,
        post: (url, body) => postJson(url, body, 30000),
        runJob: (chatBody, { onDelta }) => streamChatCompletion(llm.baseUrl, chatBody, onDelta).done,
      });
      // The 'error' listener is mandatory: a listener-less EventEmitter 'error'
      // throws, and one transient poll failure would crash the whole CLI.
      cliJobWorker.on('error', (e) => log('job poll failed: ' + e.message + ' (retrying)', process.stderr));
      cliJobWorker.on('job', ({ id }) => log('cluster job ' + id + ' — running locally'));
      cliJobWorker.on('failed', ({ id, error }) => log('cluster job ' + id + ' failed: ' + error, process.stderr));
      cliJobWorker.start();
      // Keep-alive pings ride along with serving so the node stays online on
      // the dashboard (and never gets pruned) while it works.
      const pingFull = async () => pingServer(nodeCfg, base, await fullTelemetry(nodeCfg), false);
      pingFull();
      servePinger = setInterval(pingFull, NODE.pingIntervalMs);
      if (servePinger.unref) servePinger.unref();
      log('serving cluster jobs for the LLMJob network');
    }
  });
  llm.on('stats', ({ tokensPerSec }) => {
    serveLlmState.tps = Number(tokensPerSec) || 0;
    log('🧠 ' + Number(tokensPerSec).toFixed(1) + ' tok/s');
  });
  llm.on('error', (err) => log('LLM error: ' + err.message, process.stderr));

  llm.start({ platform: process.platform, binaryPath, modelPath, nGpuLayers });
  log('local LLM starting on ' + llm.baseUrl + ' — ' + nGpuLayers + ' GPU layers');
  return llm;
}

// ── Connect with LLMJob (node pairing + ping) ────────────────────────────────
// Replaces the old install.sh agent: link this headless box to an account with a
// pairing token, then ping so it shows online. The identity lives in the shared
// nodeStore — the SAME node.json the GUI uses, so one machine keeps one nodeId
// across shells. Only the public key ever leaves the machine.

const { loadNode: loadNodeConfig, saveNode: saveNodeConfig, getOrCreateNode: getOrCreateNodeConfig } = nodeStore;

// Flag parse for the `connect` subcommand (--token/-t, --name/-n, --server).
// Kept local to the shell, but strict like shared/cliArgs: unknown flags and
// missing values are reported instead of silently ignored (a typo'd --token
// must not fall through to a misleading "no pairing token yet").
function parseConnectArgs(argv) {
  const opts = { token: null, name: null, server: null, errors: [] };
  for (let i = 0; i < argv.length; i++) {
    const tok = String(argv[i]);
    const eq = tok.indexOf('=');
    const flag = eq !== -1 ? tok.slice(0, eq) : tok;
    const inline = eq !== -1 ? tok.slice(eq + 1) : null;
    const value = () => {
      if (inline != null) return inline;
      const next = i + 1 < argv.length ? String(argv[i + 1]) : null;
      if (next == null || next.startsWith('-')) { opts.errors.push('missing value for ' + flag); return null; }
      i++;
      return next;
    };
    if (flag === '--token' || flag === '-t') opts.token = value();
    else if (flag === '--name' || flag === '-n') opts.name = value();
    else if (flag === '--server') opts.server = value();
    else opts.errors.push('unknown option: ' + tok);
  }
  return opts;
}

async function runConnect(argv) {
  const opts = parseConnectArgs(argv);
  if (opts.errors.length) {
    for (const e of opts.errors) log('error: ' + e, process.stderr);
    log('usage: llmjob-earn-cli connect --token <pairing-token> [--name <rig>] [--server <url>]', process.stderr);
    return 1;
  }
  const node = getOrCreateNodeConfig();
  if (opts.server && node.serverUrl !== opts.server) { node.serverUrl = opts.server; saveNodeConfig(node); }
  const base = node.serverUrl || NODE.serverUrl;
  const name = (opts.name && opts.name.trim()) || node.name || defaultWorker();

  log('LLMJob node ' + node.nodeId + ' → ' + base);

  if (opts.token) {
    const joinBody = nodeProto.buildJoinBody({
      token: String(opts.token).trim(), nodeId: node.nodeId, publicKey: node.publicKey, name,
    });
    let res;
    try {
      res = await postJson(base + '/api/nodes/join', joinBody, 20000);
    } catch (e) {
      log('could not reach ' + base + ': ' + e.message, process.stderr);
      return 1;
    }
    if (res.status !== 200 && res.status !== 201) {
      log('join failed (HTTP ' + res.status + '): ' + ((res.data && res.data.error) || res.raw || ''), process.stderr);
      return 1;
    }
    const user = (res.data && res.data.user) || null;
    node.name = name; node.connected = true; node.user = user; saveNodeConfig(node);
    log('✓ linked' + (user ? ' to ' + user + '’s account' : ' to your account') + ' as ' + name);
  } else if (!node.connected) {
    log('no pairing token yet — run:  llmjob-earn-cli connect --token <token>', process.stderr);
    log('copy your token from the dashboard: ' + NODE.dashboardUrl, process.stderr);
    return 1;
  } else {
    log('resuming pings for ' + (node.name || node.nodeId));
  }

  // Foreground keep-alive loop. Telemetry is SPARSE (device + VRAM + name only):
  // this process isn't the one serving inference, so sending model/capabilities/
  // tps here would overwrite what a serving run last reported with nulls.
  const pingOnce = async () => pingServer(node, base, await sparseTelemetry(node), true);

  await pingOnce();
  const timer = setInterval(pingOnce, NODE.pingIntervalMs);
  return new Promise((resolve) => {
    const shutdown = () => { clearInterval(timer); log('stopped pinging'); resolve(0); };
    process.on('SIGINT', shutdown);
    process.on('SIGTERM', shutdown);
  });
}

async function run(argv) {
  if (argv[0] === 'update') return runExplicitUpdate();
  if (argv[0] === 'connect') return runConnect(argv.slice(1));

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

  // Decide which engines run from the compute mode (mirrors the GUI): mine,
  // run a local LLM, or both. `canLlm` is always true on the CLI — the LLM's
  // own setup (binary/model) fails soft below if it can't start.
  const plan = resolvePlan(settings.mode, { canMine: isValidAddress(settings.address), canLlm: true });

  log('LLMJob Earn CLI v' + pkg.version);
  log('mode:       ' + settings.mode + (settings.modeProvided ? '' : '  (default)'));

  let endpoint = null;
  if (plan.miner) {
    // Auto-detect the knobs the user didn't pin. Best-effort: any failure falls
    // back to the defaults already in `settings` and never blocks mining. Explicit
    // --region / --gpu / --difficulty always win.
    if (!settings.workerProvided) settings.worker = defaultWorker();
    if (!settings.regionProvided) settings.region = await detectRegion();
    if (!settings.gpuProvided) {
      const det = await detectGpu();
      if (det && det.name) {
        settings.gpu = det.name;
        settings.gpuCount = det.count > 1 ? det.count : 1;
        // The pool's difficulty table is per card class; a rig submits its
        // aggregate hashrate on one connection, so scale by the card count
        // (8× RTX 3070 wants the ~560 TH/s tier, not the single-card one).
        if (!settings.difficultyProvided) settings.difficulty = difficultyForCard(det.name) * settings.gpuCount;
      }
    }

    endpoint = endpointFor(settings.region);
    log('address:    ' + shortenAddress(settings.address) + (settings.mdlAddress ? '  (+MDL ' + shortenAddress(settings.mdlAddress) + ')' : ''));
    log('pool:       ' + endpoint + '  ' + regionLabel(settings.region) + (settings.regionProvided ? '' : '  (auto)'));
    log('worker:     ' + settings.worker + (settings.workerProvided ? '' : '  (auto)'));
    log('difficulty: ' + settings.difficulty + (settings.gpu ? '  (for ' + (settings.gpuCount > 1 ? settings.gpuCount + '× ' : '') + settings.gpu + (settings.gpuProvided ? '' : ', auto') + ')' : ''));
  }

  const stats = initStats(Date.now());
  let miner = null;
  let reporter = null;
  let statsWriter = null;
  let llm = null;
  let binaryPath = null;
  let stopping = false;

  // ── Miner ────────────────────────────────────────────────────────────────
  if (plan.miner) {
    try {
      binaryPath = await resolveEngine(settings);
    } catch (e) {
      log('engine setup failed: ' + e.message, process.stderr);
      log('manual download: ' + MINER.downloadUrl, process.stderr);
      return 1;
    }

    miner = new MinerManager({ spawn });
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
      // Sample per-card live VRAM (nvidia-smi) and post one board row per GPU,
      // just like the GUI — otherwise the board shows 0 GB for a CLI-driven rig.
      const report = async () => {
        const snap = snapshot(stats, Date.now());
        const gpuVram = await detectGpusVram();
        return Promise.all(buildMinerReports(settings, snap, gpuVram, pkg.version).map(postMinerReport));
      };
      report();
      reporter = setInterval(report, NETWORK.reportIntervalMs);
      if (reporter.unref) reporter.unref();
    }

    // Write live stats JSON for external consumers (HiveOS h-stats.sh reads this
    // to feed the dashboard). Atomic write (tmp + rename) so readers never see a
    // torn file; best-effort — a failed write must never affect mining.
    if (settings.statsFile) {
      const writeStats = () => {
        try {
          const payload = statsFilePayload(snapshot(stats, Date.now()), { version: pkg.version, nowMs: Date.now() });
          const tmp = settings.statsFile + '.tmp';
          fs.writeFileSync(tmp, JSON.stringify(payload));
          fs.renameSync(tmp, settings.statsFile);
        } catch (e) { /* best effort */ }
      };
      writeStats();
      statsWriter = setInterval(writeStats, 10000);
      if (statsWriter.unref) statsWriter.unref();
    }
  }

  // ── Local LLM ──────────────────────────────────────────────────────────────
  // Keep a mining reserve free only when co-running with the miner.
  if (plan.llm) {
    llm = await startLlm(settings, plan.miner ? LLM.miningReserveMb : 0);
  }

  // Nothing to run (e.g. the LLM failed to set up and there's no miner): exit
  // with an error rather than hanging on an idle process.
  if (!miner && !llm) {
    log('nothing to run — no miner and the LLM did not start', process.stderr);
    return 1;
  }

  return new Promise((resolve) => {
    let settled = false;
    const finish = (code) => { if (!settled) { settled = true; resolve(code); } };

    const shutdown = () => {
      if (stopping) return;
      stopping = true;
      log('shutting down…');
      if (reporter) clearInterval(reporter);
      if (statsWriter) clearInterval(statsWriter);
      stopServe();
      if (llm) llm.stop();
      if (miner) miner.stop();
      else finish(0); // LLM-only: no miner 'stopped' will resolve us
    };
    process.on('SIGINT', shutdown);
    process.on('SIGTERM', shutdown);

    // llama-server dying must never be a silent clean exit: without this
    // listener an LLM-only run drains the event loop and exits 0, so a systemd
    // Restart=on-failure supervisor never restarts the node.
    if (llm) {
      llm.on('stopped', (code) => {
        serveLlmState = { ready: false, tps: 0 };
        stopServe();
        if (stopping) return;
        log('local LLM exited (code ' + code + ')', process.stderr);
        if (!miner) finish(code || 1);
      });
    }

    if (miner) {
      miner.on('stopped', (code) => {
        if (reporter) clearInterval(reporter);
        if (statsWriter) clearInterval(statsWriter);
        stopServe();
        if (llm) llm.stop();
        log('engine exited (code ' + code + ')');
        finish(stopping ? 0 : (code || 0));
      });
      try {
        miner.start(Object.assign({}, settings, { platform: process.platform, binaryPath }));
      } catch (e) {
        log('failed to launch engine: ' + e.message, process.stderr);
        if (llm) llm.stop();
        finish(1);
      }
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
