'use strict';

// Electron main process. Thin shell: owns the window, persists settings, and
// bridges the renderer to the MinerManager (the real engine). Stats shown to
// the user come only from the engine's own output — no simulated data. All
// testable logic lives in ../shared and ./minerManager.

const { app, BrowserWindow, Menu, ipcMain, shell, clipboard } = require('electron');
const { spawn, execFile } = require('child_process');
const path = require('path');
const fs = require('fs');
const http = require('http');
const https = require('https');

const { autoUpdater } = require('electron-updater');

const { MinerManager } = require('./minerManager');
const { EngineManager } = require('./engineManager');
const { LlmManager } = require('./llmManager');
const { LlmEngineManager } = require('./llmEngineManager');
const { postJson, downloadFile, streamChatCompletion, extractLlamaZip } = require('./io');
const {
  detectRegion, detectVram, detectGpusVram, detectDriverMajor,
  postMinerReport, findFreePort,
} = require('./probe');
const nodeStore = require('./nodeStore');
const { initStats, applyEvent, snapshot } = require('../shared/miningStats');
const { REGIONS, DEFAULTS, MINER, NETWORK, ECON, ECON_API, LLM, NODE, endpointFor, difficultyForCard } = require('../shared/config');
const { resolveEconomics } = require('../shared/economics');
const nodeProto = require('../shared/node');
const { computeGpuLayers, requiredVramMb, hasEnoughVram } = require('../shared/vram');
const { buildChatBody } = require('../shared/llmChat');
const { JobWorker } = require('./jobWorker');
const { resolvePlan, DEFAULT_MODE } = require('../shared/llmMode');
const { buildBalanceUrl, parseBalance, buildMdlBalanceUrl, parseMdlBalance } = require('../shared/balance');
const { isValidAddress } = require('../shared/address');
const { progressPercent, bundledEnginePath, pickEngineVersion, ENGINE } = require('../shared/engine');
const { formatUpdate } = require('../shared/updateStatus');
const { describeLaunchError } = require('../shared/engineError');
const { pickGpu } = require('../shared/gpu');
const { buildMinerReports } = require('../shared/minerReport');
const { runtimeCopyPlan } = require('../shared/llmRuntime');
const earnings = require('../shared/earnings');
const format = require('../shared/format');

let win = null;
let miner = null;
let stats = null;
let ticker = null;
let reporter = null;
let llm = null;                 // LlmManager instance while the local LLM is up
let llmStatus = { ready: false, endpoint: null, webUrl: null, tokensPerSec: 0, model: LLM.model.name, error: null };

function settingsPath() {
  return path.join(app.getPath('userData'), 'settings.json');
}
function loadSettings() {
  try {
    return JSON.parse(fs.readFileSync(settingsPath(), 'utf8'));
  } catch (e) {
    return {};
  }
}
function persistSettings(s) {
  try {
    fs.writeFileSync(settingsPath(), JSON.stringify(s, null, 2));
  } catch (e) {
    /* best effort */
  }
  return s;
}

function send(channel, payload) {
  if (win && !win.isDestroyed()) win.webContents.send(channel, payload);
}

// Fetch a pool balance for a payout address. Best-effort and never rejects —
// resolves the parsed { pending, paid, earned, usd } or null (unknown address,
// offline, non-200, bad JSON). `opts.validate` guards the address shape,
// `opts.buildUrl` / `opts.parse` select the endpoint and payload shape (the
// merge-mined MDL record lives at a different route), and `opts.priceUsd`
// (optional) adds a USD figure. Runs here in the main process so it isn't
// subject to the renderer's CSP / cross-origin restrictions.
function fetchBalance(address, opts) {
  const o = opts || {};
  const isValid = o.validate || isValidAddress;
  return new Promise((resolve) => {
    if (!isValid(address)) return resolve(null);
    try {
      const u = new URL((o.buildUrl || buildBalanceUrl)(String(address).trim()));
      const lib = u.protocol === 'http:' ? http : https;
      const req = lib.get(u, { timeout: 8000 }, (res) => {
        if (res.statusCode !== 200) { res.resume(); return resolve(null); }
        let data = '';
        res.on('data', (c) => { data += c; if (data.length > 4e6) req.destroy(); });
        res.on('end', () => {
          try { resolve((o.parse || parseBalance)(JSON.parse(data), o.priceUsd, o.currency)); }
          catch (e) { resolve(null); }
        });
      });
      req.on('error', () => resolve(null));
      req.on('timeout', () => { req.destroy(); resolve(null); });
    } catch (e) {
      resolve(null);
    }
  });
}

// Live network economics for earnings estimates. Starts as the static ECON
// fallback and is refreshed from the prlscan API so the app's $/day and the
// balance's USD figure track the real network + price instead of drifting
// (a stale fallback silently overstates earnings as the network grows).
let liveEcon = Object.assign({}, ECON, { live: { price: false, net: false, reward: false } });

// GET a JSON URL, resolving the parsed object or null (never rejects). Same
// best-effort pattern as fetchBalance — a failed refresh just keeps the
// current econ values.
function getJson(url) {
  return new Promise((resolve) => {
    try {
      const u = new URL(url);
      const lib = u.protocol === 'http:' ? http : https;
      const req = lib.get(u, { timeout: 8000 }, (res) => {
        if (res.statusCode !== 200) { res.resume(); return resolve(null); }
        let data = '';
        res.on('data', (c) => { data += c; if (data.length > 4e6) req.destroy(); });
        res.on('end', () => { try { resolve(JSON.parse(data)); } catch (e) { resolve(null); } });
      });
      req.on('error', () => resolve(null));
      req.on('timeout', () => { req.destroy(); resolve(null); });
    } catch (e) {
      resolve(null);
    }
  });
}

// Refresh liveEcon from the prlscan API (price, network hashrate, emission).
// Best-effort: whatever doesn't come back stays on the previous/fallback value.
async function refreshEconomics() {
  const [market, metrics, blocks] = await Promise.all([
    getJson(ECON_API.price), getJson(ECON_API.metrics), getJson(ECON_API.blocks),
  ]);
  liveEcon = resolveEconomics({ market, metrics, blocks }, ECON);
  return liveEcon;
}

// Report a miner-engine launch failure to the UI + log, translating the common
// antivirus-quarantine case into plain guidance instead of a cryptic error.
function reportLaunchFailure(err, missing) {
  const d = describeLaunchError({ platform: process.platform, missing: !!missing, err });
  send('miner:engine', { phase: 'error', message: d.ui });
  send('miner:log', { level: 'error', line: d.log });
}

// Map a stats snapshot to the display fields the renderer expects.
function statsView(snap) {
  return {
    total: format.formatHashrate(snap.total),
    points: snap.points,
    accepted: snap.accepted,
    acceptedLabel: format.formatInt(snap.accepted),
    rejected: snap.rejected,
    load: Math.round(snap.load),
    power: snap.power,
    gpu: snap.gpu,
    uptime: format.formatUptime(snap.uptimeSec),
    estDay: earnings.estDailyUsdLabel(snap.total, liveEcon),
  };
}

// Extract a single engine .exe from a downloaded zip to `dest`. Windows-only:
// uses PowerShell's Expand-Archive, so there's no extra runtime dependency. Used
// for the miner engine, whose binary is self-contained.
function extractZip(zipPath, dest) {
  return new Promise((resolve, reject) => {
    const tmp = dest + '.unzip';
    const wanted = path.basename(dest);
    const ps = "$ErrorActionPreference='Stop';"
      + "Expand-Archive -LiteralPath '" + zipPath + "' -DestinationPath '" + tmp + "' -Force;"
      + "$e = Get-ChildItem -Path '" + tmp + "' -Recurse -Filter '" + wanted + "' | Select-Object -First 1;"
      + "if(-not $e){ $e = Get-ChildItem -Path '" + tmp + "' -Recurse -Filter '*.exe' | Select-Object -First 1 }"
      + "Copy-Item -LiteralPath $e.FullName -Destination '" + dest + "' -Force;"
      + "Remove-Item -LiteralPath '" + tmp + "' -Recurse -Force";
    execFile('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', ps], (err) => {
      if (err) return reject(err);
      resolve(dest);
    });
  });
}

// Extract a llama.cpp Windows build zip and flatten EVERY file into dest's
// directory, so llama-server.exe ends up beside all ~30 of its DLLs (a lone
// .exe can't launch — Windows resolves sibling DLLs from the exe's folder). This
// mirrors what `unzip -j` does for the Linux/macOS archives. llama.cpp's zip is
// already flat; the recursive copy also handles a build that nests the binaries.
function extractLlamaZipWin(zipPath, dest) {
  return new Promise((resolve, reject) => {
    const dir = path.dirname(dest);
    const tmp = path.join(dir, '_llama_unzip');
    const ps = "$ErrorActionPreference='Stop';"
      + "if(Test-Path -LiteralPath '" + tmp + "'){Remove-Item -LiteralPath '" + tmp + "' -Recurse -Force};"
      + "Expand-Archive -LiteralPath '" + zipPath + "' -DestinationPath '" + tmp + "' -Force;"
      + "Get-ChildItem -Path '" + tmp + "' -Recurse -File | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination '" + dir + "' -Force };"
      + "Remove-Item -LiteralPath '" + tmp + "' -Recurse -Force";
    execFile('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', ps], { maxBuffer: 8 * 1024 * 1024 }, (err) => {
      if (err) return reject(err);
      if (!fs.existsSync(dest)) return reject(new Error('llama-server was not found in the downloaded archive'));
      resolve(dest);
    });
  });
}

async function startMining(settings) {
  // Already mining (e.g. START LLM flipped the mode to 'both' while the engine
  // runs): keep the existing miner — reassigning it would orphan an unstoppable
  // engine process and spawn a second one on the same GPU.
  if (miner && miner.isRunning()) {
    persistSettings(settings);
    return;
  }
  persistSettings(settings);

  // Real stats only: the accumulator starts at zero and is filled in from the
  // engine's parsed output (see the miner 'event' handler below). The ticker
  // just re-emits the current snapshot each second so uptime advances.
  stats = initStats(Date.now());
  send('miner:stats', statsView(snapshot(stats, Date.now())));
  if (ticker) clearInterval(ticker);
  ticker = setInterval(() => send('miner:stats', statsView(snapshot(stats, Date.now()))), 1000);

  // Publish live status to the network page's board while mining, including live
  // GPU VRAM (used/total) so the board shows headroom for co-running LLMs.
  const report = async () => {
    const snap = snapshot(stats, Date.now());
    const gpuVram = await detectGpusVram();
    buildMinerReports(settings, snap, gpuVram, app.getVersion()).forEach(postMinerReport);
  };
  report();
  if (reporter) clearInterval(reporter);
  reporter = setInterval(report, NETWORK.reportIntervalMs);

  const endpoint = settings.endpoint || endpointFor(settings.region || DEFAULTS.region);
  send('miner:log', { level: 'info', line: 'connecting to ' + endpoint + ' · worker ' + (settings.worker || DEFAULTS.worker) });

  // Resolve the engine. Off Windows, pick the build the rig's driver can run
  // (the 1.8.6+ line is faster but needs NVIDIA driver >= 580). Windows has a
  // single pool build, pinned by ENGINE.windows — versioned all the same so a
  // bump busts the cache. A packaged build may ship the engine under
  // process.resourcesPath (build.extraResources) — prefer that and skip the
  // network entirely; the lookup is version-aware, so a bundle only satisfies
  // the exact build this rig selected. Otherwise download on demand.
  let binaryPath = settings.binaryPath;
  const version = process.platform === 'win32'
    ? ENGINE.windows
    : pickEngineVersion(await detectDriverMajor());
  let bundled = bundledEnginePath(process.resourcesPath, process.platform, settings.gpu, version);
  // The Windows bundling step ships the engine under the legacy unversioned
  // name (alpha-miner-windows.exe). If the version-aware lookup misses, fall
  // back to that so a shipped bundle is still used instead of re-downloading;
  // the on-demand path below stays versioned (cache-busting) either way.
  if (process.platform === 'win32' && bundled && !fs.existsSync(bundled)) {
    const legacy = bundledEnginePath(process.resourcesPath, process.platform, settings.gpu);
    if (fs.existsSync(legacy)) bundled = legacy;
  }
  if (!binaryPath && bundled && fs.existsSync(bundled)) {
    binaryPath = bundled;
    send('miner:engine', { phase: 'ready' });
    send('miner:log', { level: 'info', line: 'using bundled engine: ' + bundled });
  }
  if (!binaryPath) {
    const engine = new EngineManager({
      dir: path.join(app.getPath('userData'), 'engine'),
      platform: process.platform,
      gpu: settings.gpu,
      version,
      fs: fs,
      download: downloadFile,
      extract: extractZip,
      chmod: fs.chmodSync,
    });
    try {
      if (engine.isInstalled()) {
        send('miner:log', { level: 'info', line: 'engine found: ' + engine.binaryPath() });
      } else {
        send('miner:engine', { phase: 'downloading' });
        send('miner:log', { level: 'info', line: 'downloading mining engine from ' + MINER.downloadUrl + ' …' });
      }
      binaryPath = await engine.ensure();
      send('miner:engine', { phase: 'ready' });
      send('miner:log', { level: 'info', line: 'engine ready: ' + binaryPath });
    } catch (e) {
      binaryPath = undefined;
      send('miner:engine', { phase: 'error', message: 'Could not download or set up the mining engine — see Logs.' });
      send('miner:log', { level: 'error', line: 'engine setup failed: ' + e.message + '. Manual download: ' + MINER.downloadUrl });
    }
  }

  // Real alpha-miner engine.
  miner = new MinerManager({ spawn });
  miner.on('log', (l) => send('miner:log', l));
  miner.on('event', (e) => {
    applyEvent(stats, e);
    send('miner:event', e);
  });
  miner.on('error', (err) => reportLaunchFailure(err, false));
  miner.on('stopped', (code) => send('miner:log', { level: 'info', line: 'engine exited (code ' + code + ')' }));

  if (binaryPath && !fs.existsSync(binaryPath)) {
    // ensure() handed us a path but the file is already gone — the classic
    // antivirus-quarantined-it-right-after-download case.
    reportLaunchFailure(null, true);
  } else if (binaryPath) {
    try {
      miner.start(Object.assign({}, settings, { platform: process.platform, binaryPath: binaryPath }));
    } catch (e) {
      reportLaunchFailure(e, false);
    }
  }
}

function stopMining() {
  if (ticker) {
    clearInterval(ticker);
    ticker = null;
  }
  if (reporter) {
    clearInterval(reporter);
    reporter = null;
  }
  stats = null;
  if (miner) {
    miner.stop();
    miner = null;
  }
  send('miner:stopped');
}

function appIcon() {
  const dir = path.join(__dirname, '..', 'assets');
  return path.join(dir, process.platform === 'win32' ? 'icon.ico' : 'icon.png');
}

// Detect the machine's GPU for the settings/device label. Uses Windows'
// Win32_VideoController via PowerShell (already a dependency of extractZip);
// resolves to a display name or null. Never rejects.
function detectGpu() {
  return new Promise((resolve) => {
    if (process.platform !== 'win32') return resolve(null);
    execFile('powershell.exe',
      ['-NoProfile', '-NonInteractive', '-Command',
        'Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name'],
      { timeout: 5000 },
      (err, stdout) => resolve(err ? null : pickGpu(String(stdout).split(/\r?\n/))));
  });
}

// Wire electron-updater to the renderer's update bar. autoUpdater pulls from the
// GitHub Releases feed (see build.publish); it only works in a packaged app, so
// main.js guards the call with app.isPackaged. Downloads happen automatically;
// the user chooses when to restart via the 'app:update:install' channel.
let manualUpdateCheck = false; // true while a user-initiated check is in flight

function setupUpdater() {
  const push = (phase, payload) => send('app:update', formatUpdate(phase, payload));
  autoUpdater.autoDownload = true;
  autoUpdater.autoInstallOnAppQuit = true;
  autoUpdater.on('checking-for-update', () => push('checking'));
  autoUpdater.on('update-available', (info) => { manualUpdateCheck = false; push('available', info); });
  // A manual check that finds nothing should say so; the automatic startup check
  // stays silent (no bar) when already current.
  autoUpdater.on('update-not-available', () => {
    if (manualUpdateCheck) { manualUpdateCheck = false; push('latest'); }
    else push('none');
  });
  autoUpdater.on('download-progress', (p) => push('progress', p));
  autoUpdater.on('update-downloaded', (info) => push('ready', info));
  autoUpdater.on('error', (err) => {
    manualUpdateCheck = false;
    push('error');
    send('miner:log', { level: 'error', line: 'update error: ' + (err && err.message ? err.message : err) });
  });
  autoUpdater.checkForUpdates().catch((e) => {
    send('miner:log', { level: 'error', line: 'update check failed: ' + e.message });
  });
}

// User-initiated "Check for updates". In a dev/unpackaged run the real updater
// isn't wired, so walk the UI through checking → up-to-date so the button still
// gives feedback (the installed app runs a real check below).
function checkForUpdate() {
  if (!app.isPackaged) {
    send('app:update', formatUpdate('checking'));
    setTimeout(() => send('app:update', formatUpdate('latest')), 700);
    return;
  }
  manualUpdateCheck = true;
  send('app:update', formatUpdate('checking'));
  autoUpdater.checkForUpdates().catch((e) => {
    manualUpdateCheck = false;
    send('app:update', formatUpdate('error'));
    send('miner:log', { level: 'error', line: 'update check failed: ' + e.message });
  });
}

function createWindow() {
  win = new BrowserWindow({
    width: 620,
    height: 650,
    minWidth: 560,
    minHeight: 560,
    backgroundColor: '#fcfcfb',
    autoHideMenuBar: true,
    show: false,
    title: 'LLMJob Earn',
    icon: appIcon(),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  win.loadFile(path.join(__dirname, '..', 'renderer', 'index.html'));

  // Size the window to the rendered content so there's no default scrollbar and
  // no trailing whitespace, whatever the platform chrome / font metrics / DPI.
  // Do it before showing (window starts hidden) to avoid a resize flash. The
  // renderer keeps overflow-y:auto, so a transient taller state (update bar,
  // engine error) still scrolls rather than clipping.
  win.webContents.on('did-finish-load', () => {
    fitWindowToContent().finally(() => { if (win && !win.isDestroyed()) win.show(); });
  });
  setTimeout(() => { if (win && !win.isDestroyed() && !win.isVisible()) win.show(); }, 1500);

  // Right-click cut/copy/paste — Electron has no default context menu, so
  // pasting a payout address (or copying it) is otherwise mouse-inaccessible.
  win.webContents.on('context-menu', (_e, params) => {
    const f = params.editFlags;
    const items = [];
    if (params.isEditable || params.selectionText) {
      items.push(
        { role: 'cut', enabled: f.canCut },
        { role: 'copy', enabled: f.canCopy },
        { role: 'paste', enabled: f.canPaste },
        { type: 'separator' },
        { role: 'selectAll' }
      );
    }
    if (items.length && !win.isDestroyed()) Menu.buildFromTemplate(items).popup({ window: win });
  });
}

// Measure the rendered content (.app) and set the window's content area to it,
// so the miner view fits exactly. Best-effort — resolves regardless of errors.
function fitWindowToContent() {
  if (!win || win.isDestroyed()) return Promise.resolve();
  return win.webContents
    .executeJavaScript('Math.ceil((document.querySelector(".app") || document.body).getBoundingClientRect().height)')
    .then((h) => {
      if (win && !win.isDestroyed() && Number.isFinite(h) && h > 0) {
        win.setContentSize(win.getContentSize()[0], h);
      }
    })
    .catch(() => {});
}

// ── Local LLM (llama.cpp llama-server), run alongside the miner ─────────────

function sendLlmStatus() { send('llm:status', llmStatus); }

// Prefer a bundled llama-server (it ships with its DLLs, like the miner engine);
// otherwise download it on demand. NOTE: the llama.cpp release is a folder of
// exe + shared libs — bundling the whole folder is the reliable path; the
// download fallback needs full-folder extraction to be production-ready.
async function resolveLlmBinary(dir) {
  const name = LLM.serverBin[process.platform] || LLM.serverBin.linux;
  const bundled = process.resourcesPath && path.join(process.resourcesPath, 'llm', name);
  if (bundled && fs.existsSync(bundled)) return bundled;
  // The server ships as a folder of DLLs + the .exe, so extraction must keep them
  // together: Windows flattens the zip with PowerShell, Linux/macOS with `unzip -j`.
  const extract = process.platform === 'win32' ? extractLlamaZipWin : (zip, dest) => extractLlamaZip(zip, dest);
  const engine = new LlmEngineManager({
    dir, platform: process.platform, serverUrl: LLM.serverUrl[process.platform],
    fs, download: downloadFile, extract, chmod: fs.chmodSync,
  });
  return engine.ensureServer();
}

// GET <baseUrl>/health — resolves true when a llama-server is already listening
// and healthy on our port. Lets startLlm adopt an existing server instead of
// spawning a second one that would fail to bind 8080 (llama-server exits with
// "couldn't bind HTTP server socket" and the LLM silently never comes up). This
// happens right after an "Update & restart", when the outgoing server may still
// be releasing the port as the relaunched app tries to claim it.
function probeLlmHealth(baseUrl, timeoutMs) {
  return new Promise((resolve) => {
    let u;
    try { u = new URL(baseUrl + '/health'); } catch (e) { return resolve(false); }
    const lib = u.protocol === 'http:' ? http : https;
    const req = lib.get(u, (res) => {
      if (res.statusCode !== 200) { res.resume(); return resolve(false); }
      // Require llama-server's own health body ({"status":"ok"}), not just any
      // 200 — otherwise unrelated software on the port (a dev server, NAS UI)
      // would be adopted as "the LLM" and chat would break against it.
      let raw = '';
      res.setEncoding('utf8');
      res.on('data', (c) => { raw += c; if (raw.length > 4096) res.destroy(); });
      res.on('end', () => {
        try { resolve(JSON.parse(raw).status === 'ok'); } catch (e) { resolve(false); }
      });
      res.on('error', () => resolve(false));
    });
    req.on('error', () => resolve(false));
    req.setTimeout(timeoutMs || 2000, () => req.destroy());
  });
}

// Reset the live-status side of llmStatus and broadcast — the single cleanup
// used by both user-initiated stops and the process's own exit.
function resetLlmStatus() {
  llmStatus = Object.assign({}, llmStatus, { ready: false, tokensPerSec: 0 });
  sendLlmStatus();
}

// Start the local LLM: ensure binary + model, size the GPU offload to leave
// `reserveMb` free for mining, spawn llama-server, and surface its OpenAI
// endpoint. Best-effort — failures are logged, never thrown to the UI. Returns
// whether the server was actually started (callers use it to reset the UI when
// an LLM-only session ends up running nothing).
async function startLlm(reserveMb) {
  if (llm && llm.isRunning()) return true;

  // If a healthy llama-server is already on our port (e.g. one lingering from a
  // just-restarted session), adopt it instead of spawning a second process that
  // would fail to bind the port and leave the LLM silently down. Reusing it is
  // safe — it serves the same model on the same OpenAI endpoint.
  const targetBase = 'http://' + LLM.host + ':' + LLM.port;
  if (await probeLlmHealth(targetBase)) {
    llmStatus = Object.assign({}, llmStatus, { ready: true, error: null, endpoint: targetBase + '/v1', webUrl: targetBase });
    send('miner:log', { level: 'info', line: 'local LLM already running on ' + targetBase + ' — reusing it' });
    sendLlmStatus();
    syncWorker();
    warmUpLlm(targetBase);
    return true;
  }

  // Preflight the GPU before doing anything expensive: refuse to start if we can
  // measure VRAM and there isn't enough free to hold the model — spawning anyway
  // risks an out-of-memory crash. When VRAM can't be read (non-NVIDIA / no
  // driver) `enough` is null and we proceed, letting llama.cpp decide.
  const vram = await detectVram();
  const freeMb = vram ? vram.totalMb - vram.usedMb : null;
  if (hasEnoughVram(freeMb, LLM.model) === false) {
    const needGb = Math.round(requiredVramMb(LLM.model) / 1024);
    const line = 'not enough free VRAM for the local LLM: ' + freeMb + ' MB free, need ~'
      + requiredVramMb(LLM.model) + ' MB for ' + LLM.model.name + ' — skipping the LLM.';
    send('miner:log', { level: 'error', line });
    llmStatus = Object.assign({}, llmStatus, { ready: false, error: 'Needs ~' + needGb + ' GB free VRAM' });
    sendLlmStatus();
    return false;
  }

  const dir = path.join(app.getPath('userData'), 'llm');
  send('miner:log', { level: 'info', line: 'preparing local LLM (' + LLM.model.name + ')…' });

  let binaryPath, modelPath;
  try {
    binaryPath = await resolveLlmBinary(dir);
    const modelEngine = new LlmEngineManager({ dir, platform: process.platform, fs, download: downloadFile });
    modelPath = await modelEngine.ensureModel();
  } catch (e) {
    send('miner:log', { level: 'error', line: 'LLM setup failed: ' + e.message });
    return false;
  }

  // Make sure the VC++ runtime DLLs sit next to llama-server.exe. llama.cpp's
  // zips don't include them; on a machine without the redistributable — or one
  // where it's only reachable via the *user* PATH, which the app relaunched by
  // the elevated updater doesn't inherit — llama-server dies instantly with
  // STATUS_DLL_NOT_FOUND before it ever logs a line. The installer bundles the
  // three DLLs (build.extraResources → <resources>/llm-runtime); co-locating
  // them beats any PATH because the exe's own dir is searched first.
  try {
    const plan = runtimeCopyPlan({
      platform: process.platform, binDir: path.dirname(binaryPath),
      resourcesPath: process.resourcesPath, existsFn: fs.existsSync, joinFn: path.join,
    });
    plan.forEach((c) => {
      fs.copyFileSync(c.from, c.to);
      send('miner:log', { level: 'info', line: 'installed LLM runtime DLL: ' + path.basename(c.to) });
    });
  } catch (e) {
    send('miner:log', { level: 'error', line: 'could not install the LLM runtime DLLs: ' + e.message });
  }

  // Size the GPU offload from the free VRAM measured above (total − used); full
  // offload when VRAM can't be read (non-NVIDIA / no driver).
  const nGpuLayers = vram
    ? computeGpuLayers(vram.totalMb - vram.usedMb, LLM.model, reserveMb || 0)
    : LLM.model.layers;

  // Bind-check the port up front and walk to a free one if the default is
  // taken. This is what actually survives the post-update relaunch: the old
  // server's port can linger unavailable on Windows for longer than the spawn
  // retries last, and a permanently-occupied 8080 (other software) would never
  // clear no matter how long we retry.
  const port = await findFreePort(LLM.host, LLM.port, 10);
  if (port !== LLM.port) {
    send('miner:log', { level: 'info', line: 'port ' + LLM.port + ' is busy — using ' + port + ' for the local LLM instead' });
  }

  let becameReady = false;
  llm = new LlmManager({ spawn, startAttempts: LLM.startAttempts, retryDelayMs: LLM.startRetryMs });
  llm.on('log', (l) => send('miner:log', l));
  llm.on('ready', ({ baseUrl }) => {
    becameReady = true;
    llmStatus = Object.assign({}, llmStatus, { ready: true, endpoint: baseUrl + '/v1', webUrl: baseUrl });
    send('miner:log', { level: 'info', line: 'local LLM ready — OpenAI endpoint ' + baseUrl + '/v1' });
    sendLlmStatus();
    syncWorker(); // serve cluster jobs once the model is up, if we're linked
    warmUpLlm(baseUrl); // background generation so tok/s populates without a chat
  });
  llm.on('stats', ({ tokensPerSec }) => { llmStatus = Object.assign({}, llmStatus, { tokensPerSec }); sendLlmStatus(); });
  llm.on('stopped', () => {
    cancelChat('the local LLM stopped');
    stopWorker();
    resetLlmStatus();
    // Exited before ever going ready while mining keeps running — this is the
    // silent-failure case (typically llama-server couldn't bind port 8080), not a
    // user stop (which stops the miner too). Surface it so the UI shows why the
    // LLM never came up instead of a permanently "starting" state.
    if (!becameReady && miner && miner.isRunning()) {
      llmStatus = Object.assign({}, llmStatus, { ready: false, error: 'The local LLM stopped before it was ready. See Logs.' });
      sendLlmStatus();
    }
    // An LLM-only session ends when llama-server exits (crash/OOM) — tell the
    // renderer, or the UI keeps showing a running session with nothing running.
    if (!miner || !miner.isRunning()) send('miner:stopped');
  });

  llm.start({ platform: process.platform, binaryPath, modelPath, nGpuLayers, port });
  llmStatus = Object.assign({}, llmStatus, { ready: false, error: null, endpoint: llm.baseUrl + '/v1', webUrl: llm.baseUrl });
  send('miner:log', { level: 'info', line: 'local LLM starting on ' + llm.baseUrl + ' — ' + nGpuLayers + ' GPU layers' });
  sendLlmStatus();
  return true;
}

// Fire one tiny generation as soon as the model is ready, so the Mine tab's
// tokens/sec figure populates on its own — the user shouldn't have to open Chat
// to see the LLM is alive. Best-effort and discarded; a real request just
// overwrites the warm-up's tok/s.
function warmUpLlm(baseUrl) {
  try {
    const body = buildChatBody([{ role: 'user', content: 'Say hello.' }], { stream: true });
    body.max_tokens = 24;
    streamChatCompletion(baseUrl, body, () => {}).done.catch(() => {});
  } catch (e) { /* best effort — never blocks startup */ }
}

function stopLlm() {
  cancelChat('the local LLM was stopped');
  stopWorker();
  if (llm) { llm.stop(); llm = null; }
  resetLlmStatus();
}

// ── In-app chat: stream the local llama-server's OpenAI chat completions ──────
// The renderer can't hit http://127.0.0.1 under its CSP, so main proxies the
// request over the shared streamChatCompletion and relays batched deltas via
// IPC (delta → done / error). Only one turn runs at a time. Cancellation always
// settles the stream, so the renderer is guaranteed a done/error outcome — no
// stuck "streaming" state when the LLM is stopped mid-reply.
let chatStream = null;

function cancelChat(reason) {
  if (!chatStream) return;
  const s = chatStream;
  chatStream = null;
  s.cancel(reason || 'cancelled');
}

// Grounding for the in-app chat: a small local model has no idea what "LLMJob"
// or "PPLNS" mean, so prompts like "What is LLMJob?" produce generic guesses.
// This system message gives it the facts to answer from. In-app chat only — jobs
// relayed from the cluster (jobWorker) are arbitrary API requests and get none.
const CHAT_SYSTEM_PROMPT = [
  "You are the assistant built into LLMJob Earn, running entirely on the user's own GPU via a local server — nothing they type leaves their machine. Use this context when relevant:",
  '- LLMJob turns the spare power of a GPU the user already owns into money: while the app runs, their GPU mines Pearl (PRL, a cryptocurrency the user is paid in) and can also run you, this local AI model.',
  '- The mining pool pays with PPLNS ("Pay Per Last N Shares"): when it finds a block, the reward is split across the last N shares miners submitted, so payout reflects sustained contribution rather than luck. Payouts settle about every 4 hours with a 1 PRL minimum.',
  '- This chat is private — it runs on the user\'s machine, so prompts never leave the computer.',
  'Answer conversationally and concisely. For anything unrelated to LLMJob, just answer normally.',
].join('\n');

function llmChat(messages) {
  cancelChat('superseded by a new message');
  const base = llmStatus.webUrl || ('http://' + LLM.host + ':' + LLM.port);
  if (!llm || !llmStatus.ready) { send('llm:chat:error', { message: 'the local LLM is not running' }); return; }

  const grounded = [{ role: 'system', content: CHAT_SYSTEM_PROMPT }].concat(Array.isArray(messages) ? messages : []);
  const s = streamChatCompletion(base, buildChatBody(grounded, { stream: true }),
    (text) => send('llm:chat:delta', { text }));
  chatStream = s;
  s.done.then(
    () => { if (chatStream === s) chatStream = null; send('llm:chat:done', {}); },
    (e) => { if (chatStream === s) chatStream = null; send('llm:chat:error', { message: e.message }); },
  );
}

// ── Connect with LLMJob (node identity + pairing/ping) ───────────────────────
// The machine's Ed25519 keypair lives in the shared nodeStore (one identity for
// the GUI AND the CLI); only the public key leaves it. "Connect" self-registers
// with a pairing token (/api/nodes/join); once linked it pings (/api/nodes/ping)
// with signed telemetry so the node shows online in the user's cluster. All
// best-effort — failures never touch mining.

let nodePinger = null;
const { loadNode, saveNode, getOrCreateNode } = nodeStore;

// Renderer-safe view (no secret key).
function nodeStatus() {
  const node = loadNode();
  return {
    connected: !!(node && node.connected),
    nodeId: node ? node.nodeId : null,
    name: node ? node.name : null,
    user: node ? (node.user || null) : null,
  };
}

function sendNodeStatus() { send('node:status', nodeStatus()); }

// The GPU name is static, so probe it once instead of spawning nvidia-smi /
// PowerShell on every 5-minute ping for the life of the app.
let gpuNameProbed = false;
let cachedGpuName = null;
async function deviceName() {
  if (!gpuNameProbed) {
    gpuNameProbed = true;
    try { cachedGpuName = await detectGpu(); } catch (e) { cachedGpuName = null; }
  }
  return cachedGpuName;
}

// One signed ping with fresh telemetry. Silent on failure. Carries the node's
// current name so a Settings rename propagates to the server (see syncNodeName).
async function pingNode() {
  const node = loadNode();
  if (!node || !node.connected) return;
  let vram = null;
  try { vram = await detectVram(); } catch (e) { /* ignore */ }
  const device = await deviceName();
  const telemetry = nodeProto.buildTelemetry({
    model: LLM.model.name, quant: LLM.model.quant, device, vram,
    tokensPerSec: llmStatus.tokensPerSec, ready: llmStatus.ready,
    activeJobs: jobWorker ? jobWorker.activeJobs() : 0,
    name: node.name,
  });
  const body = nodeProto.buildPingBody({
    nodeId: node.nodeId, publicKey: node.publicKey, secretKey: node.secretKey,
    timestamp: Date.now(), telemetry,
  });
  try { await postJson(NODE.serverUrl + '/api/nodes/ping', body, 15000); } catch (e) { /* offline — try again next tick */ }
}

// Keep the linked node's name in step with the Settings worker name: the
// connected card says "rename in Settings", so a changed worker name updates
// node.json and is pushed on the next ping (the server picks up non-null names).
function syncNodeName(settings) {
  const node = loadNode();
  const worker = settings && settings.worker && String(settings.worker).trim();
  if (node && node.connected && worker && worker !== node.name) {
    saveNode(Object.assign({}, node, { name: worker }));
    sendNodeStatus();
    pingNode();
  }
}

function startNodePinger() {
  stopNodePinger();
  pingNode();
  nodePinger = setInterval(pingNode, NODE.pingIntervalMs);
  if (nodePinger.unref) nodePinger.unref();
}

function stopNodePinger() {
  if (nodePinger) { clearInterval(nodePinger); nodePinger = null; }
}

// Link this machine to an account with a pairing/join token. Returns a
// renderer-safe result; on success the node is saved connected and starts pinging.
async function connectNode({ token, name } = {}) {
  const t = String(token || '').trim();
  if (!t) return { error: 'Enter your pairing token first.' };
  const node = getOrCreateNode();
  const nm = (name && String(name).trim()) || node.name || null;
  const body = nodeProto.buildJoinBody({ token: t, nodeId: node.nodeId, publicKey: node.publicKey, name: nm });

  let res;
  try {
    res = await postJson(NODE.serverUrl + '/api/nodes/join', body, 20000);
  } catch (e) {
    return { error: 'Could not reach LLMJob — check your connection.' };
  }
  if (res.status !== 200 && res.status !== 201) {
    return { error: (res.data && res.data.error) || ('Link failed (HTTP ' + res.status + ').') };
  }

  const user = (res.data && res.data.user) || null; // account handle, if the server resolved one
  saveNode(Object.assign({}, node, { connected: true, name: nm, user, linkedAt: new Date().toISOString() }));
  startNodePinger();
  syncWorker();
  sendNodeStatus();
  return { success: true, nodeId: node.nodeId, name: nm, user };
}

function disconnectNode() {
  const node = loadNode();
  if (node) saveNode(Object.assign({}, node, { connected: false }));
  stopNodePinger();
  stopWorker();
  sendNodeStatus();
  return { ok: true };
}

// ── Cluster job worker: serve inference relayed through LLMJob ────────────────
// When this node is linked AND the local LLM is up, poll for jobs and run them
// against the local model, streaming chunks back — all outbound, so callers can
// use this GPU through the shared API without any inbound networking here.

let jobWorker = null;

// Stream a chat request to the local llama-server, forwarding batched deltas.
// Resolves when the model is done; rejects on any transport/HTTP error.
function runLocalJob(chatBody, { onDelta }) {
  const base = llmStatus.webUrl || ('http://' + LLM.host + ':' + LLM.port);
  return streamChatCompletion(base, chatBody, onDelta).done;
}

// Start/stop the worker to match "linked AND model ready". Idempotent — called
// from connect/disconnect and the LLM ready/stopped transitions.
function syncWorker() {
  const node = loadNode();
  const shouldRun = !!(node && node.connected && llmStatus.ready);
  if (shouldRun && !jobWorker) {
    jobWorker = new JobWorker({
      identity: { nodeId: node.nodeId, publicKey: node.publicKey, secretKey: node.secretKey },
      serverUrl: node.serverUrl || NODE.serverUrl,
      post: (url, body) => postJson(url, body, 30000),
      runJob: runLocalJob,
    });
    jobWorker.on('error', () => { /* transient poll failure — keep looping */ });
    jobWorker.on('job', ({ id }) => send('miner:log', { level: 'info', line: 'cluster job ' + id + ' — running locally' }));
    jobWorker.on('done', ({ id }) => send('miner:log', { level: 'info', line: 'cluster job ' + id + ' — done' }));
    jobWorker.on('failed', ({ id, error }) => send('miner:log', { level: 'error', line: 'cluster job ' + id + ' failed: ' + error }));
    jobWorker.start();
    send('miner:log', { level: 'info', line: 'serving cluster jobs for the LLMJob network' });
  } else if (!shouldRun && jobWorker) {
    stopWorker();
  }
}

function stopWorker() {
  if (jobWorker) { jobWorker.stop(); jobWorker = null; }
}

// Resolve once the running miner is actually mining — i.e. it has reported a
// non-zero hashrate, which proves the GPU is doing real work (not just connected
// to the pool). Also settles on a miner failure (exit/error) so a broken miner
// never blocks the LLM, and on a hard cap so a miner that connects but never
// produces a share doesn't hold it forever.
function waitForMinerUp(capMs) {
  return new Promise((resolve) => {
    // Capture the manager we subscribed to: `miner` is a module global that
    // stopMining() nulls, and the manager's 'stopped' event arrives async after
    // the kill — so cleanup through the global crashes the main process with
    // "Cannot read properties of null (reading 'removeListener')" whenever the
    // miner is stopped during this wait (seen in the field as the Electron
    // error dialog). Unsubscribe from the captured reference instead.
    const m = miner;
    if (!m) return resolve();
    let done = false;
    const settle = () => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      m.removeListener('event', onEvent);
      m.removeListener('stopped', settle);
      m.removeListener('error', settle);
      resolve();
    };
    const onEvent = (e) => { if (e && e.type === 'status' && Number(e.hashrate) > 0) settle(); };
    const timer = setTimeout(settle, capMs || 60000);
    m.on('event', onEvent);
    m.once('stopped', settle);
    m.once('error', settle);
  });
}

// Apply the compute mode: run the miner and/or the LLM per the plan. When the
// plan ends up running nothing (LLM-only mode with the VRAM gate refusing, or
// no engine at all), tell the renderer — otherwise its optimistic "running"
// state shows STOP for a session in which nothing runs.
async function applyPlan(settings) {
  const plan = resolvePlan(settings.mode || DEFAULT_MODE, { canMine: isValidAddress(settings.address), canLlm: true });
  if (plan.miner) {
    // Start mining FIRST, then — when co-running — wait until the miner reports a
    // non-zero hashrate before starting the LLM. Spawning the process (or even
    // connecting to the pool) isn't enough proof mining works: the LLM loads its
    // model and warms up in a few seconds, so without this wait it goes live
    // first, and its GPU-layer budgeter reads free VRAM before mining has claimed
    // its share. Waiting for real TH/s confirms the GPU is mining and its VRAM is
    // allocated, so the LLM then sizes its offload to what's actually left.
    try {
      await startMining(settings);
    } catch (e) {
      send('miner:log', { level: 'error', line: 'start failed: ' + e.message });
    }
    if (plan.llm && miner && miner.isRunning()) await waitForMinerUp();
  } else {
    persistSettings(settings); // startMining persists; do it here when the miner is off
  }
  syncNodeName(settings);
  if (plan.llm) {
    const started = await startLlm(plan.miner ? LLM.miningReserveMb : 0).catch(() => false);
    if (!started && !plan.miner) send('miner:stopped');
  } else {
    stopLlm();
    if (!plan.miner) send('miner:stopped');
  }
}

ipcMain.handle('settings:get', () => Object.assign(
  // The desktop app defaults to 'auto' (mine + serve the LLM, balanced from free
  // VRAM) — the headless CLI keeps the mining-only DEFAULT_MODE.
  { region: DEFAULTS.region, worker: DEFAULTS.worker, difficulty: DEFAULTS.difficulty, address: '', mdlAddress: '', mode: 'auto' },
  loadSettings(),
));
ipcMain.handle('llm:status', () => llmStatus);
ipcMain.handle('config:get', () => ({ regions: REGIONS, defaults: DEFAULTS, miner: MINER }));
ipcMain.handle('miner:difficultyForCard', (_e, name) => difficultyForCard(name));
ipcMain.handle('gpu:detect', () => detectGpu());
ipcMain.handle('region:detect', () => detectRegion());
ipcMain.handle('balance:get', (_e, address) => fetchBalance(address, { priceUsd: liveEcon.PRL_USD }));
// The MDL record is keyed by the PRL payout address (the pool rejects mdl1…
// addresses on the miner endpoint), so this takes the PRL address.
ipcMain.handle('balance:getMdl', (_e, address) =>
  fetchBalance(address, { buildUrl: buildMdlBalanceUrl, parse: parseMdlBalance }));
ipcMain.on('miner:start', (_e, settings) => applyPlan(settings || {}));
ipcMain.on('miner:stop', () => { stopMining(); stopLlm(); });
ipcMain.on('open-external', (_e, url) => { shell.openExternal(url); });
// Re-fit the window to its content when the renderer's layout changes (tab
// switch, mining start/stop, etc.), so the frame never leaves a gap under the
// footer or clips a taller view.
ipcMain.on('app:fit', () => { fitWindowToContent(); });
ipcMain.on('clipboard:write', (_e, text) => { clipboard.writeText(String(text == null ? '' : text)); });
ipcMain.on('llm:chat', (_e, messages) => llmChat(messages));
ipcMain.handle('node:status', () => nodeStatus());
ipcMain.handle('node:connect', (_e, opts) => connectNode(opts || {}));
ipcMain.handle('node:disconnect', () => disconnectNode());
ipcMain.on('node:dashboard', () => shell.openExternal(NODE.dashboardUrl));
ipcMain.handle('app:version', () => app.getVersion());
ipcMain.on('app:update:check', () => checkForUpdate());
ipcMain.on('app:update:install', () => {
  try {
    // If mining right now, remember to resume automatically after the restart.
    if (stats) persistSettings(Object.assign({}, loadSettings(), { resumeMining: true }));
    // Stop the LLM and miner BEFORE relaunching. llama-server binds a fixed port
    // (8080) and quitAndInstall relaunches immediately; if the outgoing server is
    // still holding the port, the resumed instance's llama-server can't bind and
    // the LLM silently fails to start (the miner binds no port, so it's fine —
    // exactly the "miner came back, LLM didn't" symptom). Killing them here frees
    // the port and VRAM deterministically before the new instance starts.
    stopLlm();
    stopMining();
    // isSilent=true: install to the existing directory without re-showing the
    // assisted-installer wizard. isForceRunAfter=true: relaunch the app afterwards.
    autoUpdater.quitAndInstall(true, true);
  } catch (e) {
    send('miner:log', { level: 'error', line: 'update install failed: ' + e.message });
  }
});

app.whenReady().then(() => {
  createWindow();
  if (app.isPackaged) setupUpdater();
  // Pull live network economics so earnings estimates and the balance's USD
  // figure reflect the real price + network, not the stale fallback constants.
  refreshEconomics();
  const econTimer = setInterval(refreshEconomics, 10 * 60 * 1000);
  if (econTimer.unref) econTimer.unref();
  // The identity used to live in Electron's userData dir; move it into the
  // store shared with the CLI so one machine keeps one nodeId across shells.
  nodeStore.migrateFrom(path.join(app.getPath('userData'), 'node.json'));
  // Resume pinging if this machine is already linked to an account.
  const node = loadNode();
  if (node && node.connected) startNodePinger();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  stopMining();
  stopLlm(); // never orphan llama-server (it would hold VRAM + port 8080)
  if (process.platform !== 'darwin') app.quit();
});
