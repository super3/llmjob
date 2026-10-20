'use strict';

// Electron main process. Thin shell: owns the window, persists settings, and
// bridges the renderer to our own Pearl miner. Stats shown to
// the user come only from the engine's own output — no simulated data. Most
// testable logic lives in ../shared and the other modules in this folder.

const {
  app, BrowserWindow, Menu, Tray, Notification, nativeImage, powerMonitor, ipcMain, shell, clipboard,
} = require('electron');
const { spawn, execFile } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');
const http = require('http');
const https = require('https');

const { autoUpdater } = require('electron-updater');

const net = require('net');
const dns = require('dns');
const { PearlEngine } = require('./pearlEngine');
const { XmrEngine } = require('./xmrEngine');
const { findWorker, makeSpawnWorker } = require('./xmrWorker');
const { readCpuFacts } = require('./cpuFacts');
const { coreFactory } = require('./pearlCore');
const { getJson, streamChatCompletion, extractLlamaZip } = require('./io');
const { detectRegion, detectGpusVram, postMinerReport } = require('./probe');
const probe = require('./probe');
const nodeStore = require('./nodeStore');
const { createNodeClient } = require('./nodeClient');
const { llmEngine, makeLlmFleet, llmRunOptions } = require('./llmLaunch');
const settingsStore = require('../shared/settingsStore');
const { initStats, applyEvent, snapshot, offPool } = require('../shared/miningStats');
const { makePoolLookup, fileCache } = require('../shared/poolLookup');
const {
  REGIONS, DEFAULTS, XMR, NETWORK, ECON, ECON_API, LLM, NODE, resolveEndpoint, migrateRegion, xmrEndpointFor,
} = require('../shared/config');
const { defaultWorker } = require('../shared/worker');
const { resolveEconomics } = require('../shared/economics');
const { requiredFreeMb, pickLlmGpu } = require('../shared/vram');
const { planLlmInstances } = require('../shared/llmPlan');
const { allModels, pickModel, ctxLadder } = require('../shared/models');
const gpuLoss = require('../shared/gpuLoss');
const { createPromptPause, cardForPrompt } = require('../shared/promptPause');
const { createLanShare } = require('./lanShare');
const { buildChatBody } = require('../shared/llmChat');
const { resolvePlan, DEFAULT_MODE } = require('../shared/llmMode');
const { minerSupported, minerUnsupportedNote, autoUpdateSupported } = require('../shared/platform');
const { placeLlmPlan, describeLlmStart, parseServedModel } = require('../shared/llama');
const { buildBalanceUrl, parseBalance, buildPoolStatsUrl, parsePoolPriceUsd } = require('../shared/balance');
const { isValidAddress } = require('../shared/address');
const { checkXmrAddress } = require('../shared/xmr/address');
const { planCpuThreads } = require('../shared/xmr/cpuPlan');
const { formatUpdate, describeUpdateError } = require('../shared/updateStatus');
const { repairUpdateConfig } = require('../shared/updateConfig');
const { buildMinerReports } = require('../shared/minerReport');
const { runtimeCopyPlan } = require('../shared/llmRuntime');
const {
  alignCudaDeviceOrder, clearCudaVisibleDevices, describeClearedCuda, readGpuIndexEnv, describeSkippedGpus,
} = require('../shared/gpu');
const { planMemClocks, readMemClockEnv, GUI_MEM_CLOCK_ENV } = require('../shared/memClock');
const earnings = require('../shared/earnings');
const format = require('../shared/format');
const appPrefs = require('../shared/appPrefs');
const autostart = require('../shared/autostart');
const trayHost = require('../shared/trayHost');

// The id the installer gives the Start menu shortcut (build.appId in
// package.json). Windows shows a notification only from an app whose id matches
// a shortcut; without it the one-time "still running in the tray" notice never
// appeared. The login item is filed under it too.
const APP_ID = 'com.llmjob.earn';
if (process.platform === 'win32') app.setAppUserModelId(APP_ID);

// Number the GPUs the way nvidia-smi does, before anything opens a CUDA device.
// Everything here — the device label, per-card VRAM, temperatures, the board's
// rows — speaks nvidia-smi's indices, and the CUDA runtime does not unless told
// to. Set at load, because the mining core initialises CUDA inside THIS process
// and reads it then. For the same reason CUDA must see every card nvidia-smi
// lists; the removed value is logged on each start, since there is no window yet.
alignCudaDeviceOrder(process.env);
const clearedCudaLine = describeClearedCuda(clearCudaVisibleDevices(process.env));

// One copy of the app at a time. With the window closing to the tray, opening
// the app again is how someone gets it back, and a second copy would mine the
// same cards beside the first. A second launch shows this one's window and
// exits.
const primaryInstance = app.requestSingleInstanceLock();
if (!primaryInstance) app.quit();

// Started by the login item or the autostart entry: stay in the tray and start
// mining. Read once; a window made later (from the tray) shows as usual.
let hiddenLaunch = autostart.launchedHidden(process.argv);
let tray = null;
// Set on a real quit (the tray's Quit, an update, logging off), so the window's
// close handler lets the window close instead of hiding it.
let quitting = false;
// The Windows message a user's close arrives as (see the window's close handler).
const WM_SYSCOMMAND = 0x0112;
const SC_CLOSE = 0xf060;

let win = null;
let miner = null;
let stats = null;
let ticker = null;
let reporter = null;
// Bumped on every stop. A start is async (the engine can download for minutes),
// during which the user may press STOP; the in-flight start captures this epoch
// and, if it has since changed, aborts instead of spawning a headless miner or
// starting the LLM after the user already stopped.
let miningEpoch = 0;
let fleet = null;               // LlmFleet (one llama-server per eligible GPU) while up
// Mining stops on a card while the LLM beside it answers a prompt, and starts
// again a little after (see shared/promptPause). Acts on whatever miner runs at
// the time, so a miner started mid-prompt starts paused (startMining calls
// reapply). In LLM-only mode there is no miner and it pauses, and says, nothing.
const promptPause = createPromptPause({
  // A miner that is not running pauses and resumes nothing (PearlMiner).
  pause: (indices) => (miner ? miner.pauseCards(indices).length > 0 : false),
  resume: (indices) => (miner ? miner.resumeCards(indices).length > 0 : false),
  log: (level, line) => send('miner:log', { level, line }),
  resumeAfterMs: LLM.promptPause.resumeAfterMs,
  maxPauseMs: LLM.promptPause.maxPauseMs,
});
// nvidia-smi's PCI bus id per card, read when the LLM starts, to find the card
// each llama-server really loaded on (see promptPause.cardForPrompt).
let llmBusIds = {};
// And nvidia-smi's cards, read at the same time, to name a card that gives up
// on its GPU (see rememberGpuLoss).
let llmCards = [];
let llmEverReady = false;       // did any instance reach "ready" this run (vs. dying first)
let serveLogged = false;        // "serving cluster jobs" logged once per serving run
// `note` is a transient "what is it doing right now" line — "Downloading model…
// 42%", "Starting…" — shown in place of the model name until the LLM is ready.
// Without it a first run logs "preparing local LLM…" and then goes silent for
// however long a ~5 GB model takes, which reads as a hang: a user on a 2-GPU rig
// hit START seventeen times over ninety seconds while the download was in fact
// progressing normally.
// `vramMb` and `ctxSize` describe the model being served, for the API tab's
// "~N GB VRAM · NK ctx" line, which was hard-coded to Gemma's old numbers.
// `held` says, in a sentence, which model a card that kept losing its GPU is
// not running, and how to try it again (see startLlm).
let llmStatus = { ready: false, endpoint: null, webUrl: null, tokensPerSec: 0, model: LLM.model.name,
  vramMb: LLM.model.vramFullMb, ctxSize: ctxLadder(LLM.model)[0], error: null, note: null, held: null };
// Which model this machine actually loaded, once a run has chosen one. Every
// place that reports a model name reads this rather than LLM.model, which stopped
// being the answer when selection became per-node: a 5090 serving Qwen3.8 would
// otherwise tell the dashboard, the telemetry and — via metrics.model — the
// `model` field of every gateway completion that it was serving Gemma. Seeded
// with the default so a report before the first start is still truthful.
let servingModel = LLM.model;
// What the running fleet's model was chosen on: the best card, its free VRAM,
// and the models it kept losing its GPU under (see llmReplan).
let llmPlannedOn = { index: null, freeMb: null, lost: [] };
// Sharing the local LLM on the network (the API tab, #257): the
// CLI's gate on port 8000, in front of llama-server. syncLanShare() decides
// when it runs.
const lanShare = createLanShare({
  upstreamUrl: () => fleet && fleet.webUrl(),
  isLlmReady: () => !!(fleet && fleet.isReady()),
  log: (line) => send('miner:log', { level: 'info', line }),
  onChange: () => {
    llmStatus = Object.assign({}, llmStatus, { lan: lanShare.status() });
    sendLlmStatus();
  },
});

function settingsPath() {
  return path.join(app.getPath('userData'), 'settings.json');
}
const settingsLog = (m) => console.error(m);
function loadSettings() {
  return settingsStore.readSettings(settingsPath(), { fs, log: settingsLog });
}
// Translate a saved AlphaPool region onto a live one. Not a write: the renderer
// hands the migrated id straight back when the user next saves, so the file
// heals itself without a startup rewrite that would touch a settings file we
// might have failed to read properly.
function withLiveRegion(s) {
  return Object.assign({}, s, { region: migrateRegion(s.region) });
}

function persistSettings(s) {
  return settingsStore.writeSettings(settingsPath(), s, { fs, log: settingsLog });
}

// Cards that kept losing their GPU, and the models they lost it under (see
// shared/gpuLoss). In their own file, for the reason appPrefs gives: every
// START rewrites settings.json from the renderer's fields.
function gpuLostPath() {
  return path.join(app.getPath('userData'), 'gpu-lost.json');
}
function loadGpuLost() {
  return gpuLoss.forVersion(settingsStore.readSettings(gpuLostPath(), { fs, log: settingsLog }), app.getVersion());
}
function saveGpuLost(record) {
  settingsStore.writeSettings(gpuLostPath(), record, { fs, log: settingsLog });
}

// The app's behaviour switches (tray, startup, idle-only), from Settings. In
// their own file, see shared/appPrefs. Read once, then kept here.
let prefs = null;
function prefsPath() {
  return path.join(app.getPath('userData'), 'preferences.json');
}
function getPrefs() {
  if (!prefs) prefs = appPrefs.normalizePrefs(settingsStore.readSettings(prefsPath(), { fs, log: settingsLog }));
  return prefs;
}
function savePrefs(change) {
  prefs = appPrefs.normalizePrefs(Object.assign({}, getPrefs(), change));
  settingsStore.writeSettings(prefsPath(), prefs, { fs, log: settingsLog });
  return prefs;
}

// Make the app start (or stop starting) with the computer. Returns whether the
// computer will now start it, which is what Settings shows.
//
// Only in the installed app: a dev run would register the bare Electron binary.
// Windows has a login item for it. Linux and macOS get a file instead (see
// loginFile). macOS has a login item too, but it can't pass --hidden, and on
// macOS 13 and later the app can't tell it was started at login.
function applyLoginItem(on) {
  if (!app.isPackaged) return false;
  if (process.platform === 'win32') {
    app.setLoginItemSettings({ openAtLogin: on, args: on ? [autostart.HIDDEN_ARG] : [] });
    return on;
  }
  const { file, body } = loginFile();
  try {
    if (on && body) {
      fs.mkdirSync(path.dirname(file), { recursive: true });
      fs.writeFileSync(file, body);
      return true;
    }
    if (fs.existsSync(file)) fs.unlinkSync(file);
  } catch (e) {
    send('miner:log', { level: 'error', line: 'could not change the start-with-computer setting: ' + e.message });
  }
  return false;
}

// The file that starts the app at login, and what it should say: a freedesktop
// autostart entry for the AppImage on Linux, a LaunchAgent on macOS. The Linux
// entry runs the AppImage file ($APPIMAGE), not the binary inside it, whose
// mount is gone after a reboot. `body` is null when this copy can't start that
// way: a Linux build that is not an AppImage.
function loginFile(appImage) {
  if (process.platform === 'darwin') {
    return { file: autostart.macLaunchAgentFile(os.homedir()), body: autostart.macLaunchAgent(process.execPath) };
  }
  const configHome = process.env.XDG_CONFIG_HOME || path.join(os.homedir(), '.config');
  const image = appImage || process.env.APPIMAGE;
  return { file: autostart.linuxAutostartFile(configHome), body: image ? autostart.linuxDesktopEntry(image) : null };
}

// While the switch is on: point the login item at this copy again if it no
// longer does (an update renamed the AppImage, a test build took it over), and
// otherwise leave it alone. Writing it every start undid the user's own switch
// for it: Windows' setLoginItemSettings re-enables an entry disabled in Task
// Manager, and a rewritten autostart entry loses Hidden=true or
// X-GNOME-Autostart-enabled=false. `appImage` is the AppImage's new name, when
// an update gave it one.
function refreshLoginItem(appImage) {
  if (!app.isPackaged || !getPrefs().runAtStartup) return;
  if (process.platform === 'win32') {
    if (!app.getLoginItemSettings({ args: [autostart.HIDDEN_ARG] }).openAtLogin) applyLoginItem(true);
    return;
  }
  const { file, body } = loginFile(appImage);
  if (!body) return;
  let current = null;
  try { current = fs.readFileSync(file, 'utf8'); } catch (e) { /* missing: written below */ }
  const next = current != null && process.platform === 'linux' ? autostart.retargetDesktopEntry(current, body) : body;
  if (next === current) return;
  try {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, next);
  } catch (e) {
    send('miner:log', { level: 'error', line: 'could not update the start-with-computer entry: ' + e.message });
  }
}

// Log lines wait here until the window is listening. The renderer subscribes
// after its first round trips (settings, then GPU and region detection), so a
// line sent at startup, such as the updater's repair warning, was dropped. The
// preload's onLog says when it is listening ('log:ready'), and a page that
// loads again starts over (see createWindow).
const LOG_BACKLOG_MAX = 200;
let logListening = false;
let logBacklog = [];

function send(channel, payload) {
  if (!win || win.isDestroyed()) return;
  if (channel === 'miner:log' && !logListening) {
    logBacklog.push(payload);
    if (logBacklog.length > LOG_BACKLOG_MAX) logBacklog.shift();
    return;
  }
  win.webContents.send(channel, payload);
}

// Open a URL in the user's browser, but only http(s). The renderer can hand any
// string to the 'open-external' channel, and shell.openExternal would otherwise
// happily launch file:, smb:, or a registered custom-protocol handler. The
// returned promise is swallowed too, so a bad URL can't become an unhandled
// rejection.
function openExternalSafe(url) {
  try {
    const u = new URL(String(url));
    if (u.protocol !== 'http:' && u.protocol !== 'https:') return;
    Promise.resolve(shell.openExternal(u.href)).catch(() => {});
  } catch (e) {
    /* not a valid URL — ignore */
  }
}

// Fetch a pool balance for a payout address. Best-effort and never rejects —
// resolves the parsed { pending, paid, earned, usd } or null (unknown address,
// offline, non-200, bad JSON). `priceUsd` (optional) adds a USD figure. Runs
// here in the main process so it isn't subject to the renderer's CSP /
// cross-origin restrictions.
//
// This used to take buildUrl/parse/validate overrides so the merge-mined MDL
// record could be read from its own route with its own payload shape. MDL is
// gone, so the indirection is gone with it — one address kind, one route.
async function fetchBalance(address, priceUsd) {
  if (!isValidAddress(address)) return null;
  let json;
  try {
    json = await getJson(buildBalanceUrl(String(address).trim()));
  } catch (e) {
    return null; // buildUrl threw
  }
  if (json == null) return null;
  try {
    return parseBalance(json, priceUsd);
  } catch (e) {
    return null; // parse threw
  }
}

// The Monero balance on HeroMiners' Monero pool, the CPU miner's. Same pool
// software and payload as the Pearl one, in 1e12 units, and with no `balance`
// field until there is one. Best-effort like fetchBalance: null on any failure.
// Its dollar value uses the pool's XMR price; without one, usd is null and
// Settings shows the balance without dollars.
async function fetchXmrBalance(address) {
  if (!checkXmrAddress(address).ok) return null;
  const json = await getJson(buildBalanceUrl(String(address).trim(), XMR.apiBase)).catch(() => null);
  if (json == null) return null;
  const price = await xmrPriceUsd();
  return parseBalance(json, price == null ? undefined : price, { units: XMR.coinUnits, absentIsZero: true });
}

// HeroMiners' XMR price, read at most every 10 minutes, as often as the PRL
// price. A failed read keeps the last price.
const XMR_PRICE_MAX_AGE_MS = 10 * 60 * 1000;
let xmrPrice = null;
let xmrPriceAt = 0;
async function xmrPriceUsd() {
  if (xmrPrice != null && Date.now() - xmrPriceAt < XMR_PRICE_MAX_AGE_MS) return xmrPrice;
  const usd = parsePoolPriceUsd(await getJson(buildPoolStatsUrl(XMR.apiBase)).catch(() => null));
  if (usd != null) {
    xmrPrice = usd;
    xmrPriceAt = Date.now();
  }
  return xmrPrice;
}

// ── CPU mining (Monero) ─────────────────────────────────────────────────────
// Opt-in and off by default (Settings → CPU Mining). runPlan starts it for any
// START that asks for it, whether or not the GPU miner runs (the CLI does the
// same), and STOP and the idle pause stop it. The hashing runs in a worker
// process of its own (see xmrEngine), so a crash there, or an antivirus
// removing the worker, stops the CPU miner and nothing else. Its line on the
// Mine tab comes on 'cpu:stats', apart from the GPU miner's ticks, which an
// LLM-only START never sends.
let cpuMiner = null;
let cpuError = null;     // why the CPU miner last stopped, for the Mine tab
let cpuFacts = null;     // read once: CPUs, cgroup limit, memory, L3

// Where this install's RandomX worker is, or null: a build without one (macOS
// has none), or an antivirus removed it.
function cpuWorkerPath() {
  return findWorker({ resourcesPath: process.resourcesPath, env: process.env, exists: fs.existsSync });
}

// The thread plan for a requested count (0 for the default).
function cpuPlan(threads) {
  if (!cpuFacts) cpuFacts = readCpuFacts({ fs, os });
  return planCpuThreads(cpuFacts, threads);
}

function cpuRunning() {
  return !!(cpuMiner && cpuMiner.isRunning());
}

function sendCpu() {
  send('cpu:stats', cpuView());
}

function startCpuMining(settings) {
  if (cpuMiner || !settings.cpuMining) return;
  cpuError = null;
  const say = (level, line) => send('miner:log', { level, line: 'cpu: ' + line });
  const addr = checkXmrAddress(settings.xmrAddress);
  if (!addr.ok) {
    cpuError = 'Monero address: ' + addr.error;
    say('warn', 'not mining Monero: ' + cpuError);
    sendCpu();
    return;
  }
  const plan = cpuPlan(Number(settings.cpuThreads) || 0);
  if (!plan.threads) {
    cpuError = plan.reason;
    say('warn', 'not mining Monero: ' + plan.reason);
    sendCpu();
    return;
  }
  const engine = new XmrEngine({
    connect: (host, port) => net.connect({ port, host, lookup: poolLookup() }),
    spawnWorker: makeSpawnWorker(spawn),
    workerPath: cpuWorkerPath(),
    agent: 'llmjob-earn/' + app.getVersion(),
  });
  engine.on('log', (l) => send('miner:log', l));
  // Kept for the Mine tab. The engine has logged it.
  engine.on('error', (e) => { cpuError = e.message; });
  // The pool took the login, so this address is one that worked.
  engine.on('authorized', (e) => poolLookup().worked(e.host, e.address, e.family));
  // Every status change and hashrate tick (every 2 s) updates the Mine tab.
  engine.on('event', () => { if (cpuMiner === engine) sendCpu(); });
  // Code 1: the engine gave up on its own. A stop we asked for is code 0, and
  // stopCpuMining lets go of the engine first anyway. With nothing else
  // running the START ends, or the window shows STOP over nothing. The line
  // keeps saying why.
  engine.on('stopped', (code) => {
    if (code !== 1 || cpuMiner !== engine) return;
    const why = cpuError;
    endIfNothingRuns((miner && miner.isRunning()) || llmRunsOrComing());
    if (!session) {
      cpuError = why;
      sendCpu();
    }
  });
  cpuMiner = engine;
  if (plan.reason) say('info', plan.reason);
  engine.start({
    address: String(settings.xmrAddress).trim(),
    worker: settings.worker || DEFAULTS.worker,
    // The Monero host follows the Pearl region.
    endpoint: xmrEndpointFor(migrateRegion(settings.region)),
    threads: plan.threads,
  });
  sendCpu();
}

function stopCpuMining() {
  if (!cpuMiner && !cpuError) return;
  const m = cpuMiner;
  cpuMiner = null;
  cpuError = null;
  if (m) m.stop();
  sendCpu();
}

// The CPU line on the Mine tab, or null when CPU mining is not running.
function cpuView() {
  if (!cpuMiner && !cpuError) return null;
  const s = cpuMiner ? cpuMiner.snapshot() : { state: 'stopped', hashrate: 0, accepted: 0, rejected: 0, threads: 0 };
  const off = s.state === 'stopped' || s.state === 'failed';
  return {
    state: s.state,
    hashrate: format.formatKhs(s.hashrate),
    accepted: s.accepted,
    rejected: s.rejected,
    threads: s.threads,
    error: off ? cpuError : null,
    // Why the pool can't be used, while the miner runs without it.
    pool: off ? null : s.poolError || null,
  };
}

// Live network economics for earnings estimates. Starts as the static ECON
// fallback and is refreshed from the prlscan API so the app's $/day and the
// balance's USD figure track the real network + price instead of drifting
// (a stale fallback silently overstates earnings as the network grows).
let liveEcon = Object.assign({}, ECON, { live: { price: false, difficulty: false, reward: false } });

// Refresh liveEcon from the prlscan API (price, and the latest block's
// difficulty and reward). Best-effort: whatever doesn't come back falls back to
// the ECON constants.
async function refreshEconomics() {
  const [market, blocks] = await Promise.all([getJson(ECON_API.price), getJson(ECON_API.blocks)]);
  liveEcon = resolveEconomics({ market, blocks }, ECON);
  return liveEcon;
}

// Report a failure to start mining. This used to translate the
// antivirus-quarantined-the-download case, which cannot happen any more: there
// is no download and no spawned binary, so the only way to fail here is that
// the CUDA core is missing or refused to initialise. PearlEngine already says
// which, in words; this puts it in front of the user.
function reportLaunchFailure(err) {
  // A core that fails to load can surface as something with no message at all,
  // and "could not start mining: undefined" tells nobody anything.
  const line = 'could not start mining: '
    + (err && err.message ? err.message : 'the Pearl core did not initialise');
  send('miner:engine', { phase: 'error', message: line });
  send('miner:log', { level: 'error', line: line });
}

// Map a stats snapshot to the display fields the renderer expects.
function statsView(snap) {
  return {
    total: format.formatHashrate(snap.total),
    points: snap.points,
    acceptedLabel: format.formatInt(snap.accepted),
    // Every card that is mining, not just the first. On a multi-card rig showing
    // one name is the same lie issue #226 was about, one level up.
    gpu: format.formatDeviceLabel(snap.gpus.map((g) => g.gpu)) || snap.gpu,
    temp: snap.temp,
    uptime: format.formatUptime(snap.uptimeSec),
    estDay: earnings.estDailyUsdLabel(snap.total, liveEcon),
  };
}

// Quote a value as a single PowerShell single-quoted literal, doubling any
// embedded single quote. Interpolating a raw path into a '...' PS string lets a
// path containing a quote break out of the quoting and inject commands; routing
// every interpolated value through this closes that.
function psQuote(value) {
  return "'" + String(value).replace(/'/g, "''") + "'";
}

// Extract a llama.cpp Windows build zip and flatten EVERY file into dest's
// directory, so llama-server.exe ends up beside all ~30 of its DLLs (a lone
// .exe can't launch — Windows resolves sibling DLLs from the exe's folder). This
// mirrors what `unzip -j` does for the Linux/macOS archives. llama.cpp's zip is
// already flat; the recursive copy also handles a build that nests the binaries.
//
// Unzips via .NET's ZipFile rather than PowerShell's Expand-Archive, which
// validates by FILE EXTENSION and accepts only `.zip`. The download lands at a
// deliberately format-neutral name (llmEngineManager's ARCHIVE_TMP,
// `llama-download.archive`, sniffed by magic bytes on POSIX), so Expand-Archive
// refused every Windows install with "'.archive' is not a supported archive file
// format" and the local LLM could never start. ZipFile reads the file, not its
// name. Add-Type is needed on Windows PowerShell 5.1 and absent-but-harmless on
// 7, hence the swallow.
function extractLlamaZipWin(zipPath, dest) {
  return new Promise((resolve, reject) => {
    const dir = path.dirname(dest);
    const tmp = path.join(dir, '_llama_unzip');
    const ps = "$ErrorActionPreference='Stop';"
      + 'try{Add-Type -AssemblyName System.IO.Compression.FileSystem}catch{};'
      + 'if(Test-Path -LiteralPath ' + psQuote(tmp) + '){Remove-Item -LiteralPath ' + psQuote(tmp) + ' -Recurse -Force};'
      + '[System.IO.Compression.ZipFile]::ExtractToDirectory(' + psQuote(zipPath) + ',' + psQuote(tmp) + ');'
      + 'Get-ChildItem -Path ' + psQuote(tmp) + ' -Recurse -File | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination ' + psQuote(dir) + ' -Force };'
      + 'Remove-Item -LiteralPath ' + psQuote(tmp) + ' -Recurse -Force';
    execFile('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', ps], { maxBuffer: 8 * 1024 * 1024 }, (err) => {
      if (err) return reject(err);
      if (!fs.existsSync(dest)) return reject(new Error('llama-server was not found in the downloaded archive'));
      resolve(dest);
    });
  });
}

// How the pool's name becomes an address: the system lookup, then fallbacks
// when it fails (see shared/poolLookup). One for the life of the app, with the
// last good addresses kept beside the node identity, shared with the CLI.
let poolLookupFn = null;
function poolLookup() {
  if (!poolLookupFn) {
    poolLookupFn = makePoolLookup({
      dns,
      cache: fileCache(path.join(path.dirname(nodeStore.nodePath()), 'pool-addresses.json'), fs, path),
      log: (line) => send('miner:log', { level: 'warn', line }),
    });
  }
  return poolLookupFn;
}

// One miner start at a time. runPlan and an idle resume can each ask for one
// while the other's is still in its nvidia-smi await, before `miner` is set; the
// second joins the first instead of building a second engine on the same cards.
let minerStarting = null;
function startMiningOnce(settings, llmCoRuns) {
  if (!minerStarting) minerStarting = startMining(settings, llmCoRuns).finally(() => { minerStarting = null; });
  return minerStarting;
}

// `llmCoRuns` is runPlan's plan.llm: whether a local LLM will share the cards
// for this run. It decides the memory clock plan below.
async function startMining(settings, llmCoRuns) {
  // Already mining (e.g. START LLM flipped the mode to 'both' while the engine
  // runs): keep the existing miner — reassigning it would orphan an unstoppable
  // engine process and spawn a second one on the same GPU.
  //
  // A miner that started mining-only may hold the Blackwell memory clock lock
  // (below). When the LLM joins it, that lock goes now, before runPlan starts
  // the model: a served model must never sit on a locked card. The other way
  // round -- the mode flipped back to mining-only mid-run -- takes no lock; the
  // card mines at the driver's clock until the next start, which is slower,
  // never wrong.
  if (miner && miner.isRunning()) {
    if (llmCoRuns) miner.releaseMemClocks();
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
    // While the pool connection is down, the board hears no hashrate: the pool
    // is getting none, and the board should agree with it.
    const live = snapshot(stats, Date.now());
    const snap = miner && !miner.poolConnected() ? offPool(live) : live;
    const gpuVram = await detectGpusVram();
    // Tag the cards serving the local LLM so the board shows which model each GPU
    // runs; null when the fleet isn't up (mining only) → blank on the board.
    // `nodeId` rides along whenever the fleet is up, because that is now exactly
    // when this machine polls the cluster — linked or not (syncWorker arms the
    // worker on the fleet, not on the account). Running the model and serving the
    // cluster are the same thing here; a machine with no identity yet reports null.
    const identity = loadNode();
    const serving = fleet
      ? {
        model: servingModel.name,
        indices: fleet.servingIndices(),
        nodeId: identity ? identity.nodeId : null,
      }
      : null;
    buildMinerReports(settings, snap, gpuVram, app.getVersion(), serving).forEach(postMinerReport);
  };
  report();
  if (reporter) clearInterval(reporter);
  reporter = setInterval(report, NETWORK.reportIntervalMs);

  const endpoint = resolveEndpoint(settings);
  send('miner:log', { level: 'info', line: 'connecting to ' + endpoint + ' · worker ' + (settings.worker || DEFAULTS.worker) });

  // The miner is this process: the GPU work is a linked N-API addon, so there
  // is no binary to resolve, download, version-gate or spawn. That whole path
  // went with alpha-miner.
  //
  // coreFactory returns null when pearl_core.node is not present, which is the
  // expected state on a machine without a CUDA build. PearlEngine then stops
  // cleanly and says so rather than opening a pool socket it could never feed.
  //
  // Which cards to mine on: every card nvidia-smi lists, one core each. This is
  // an await, short but real (it spawns nvidia-smi), so the stop-epoch check is
  // back — STOP can now land inside a start again, and starting a miner the user
  // has already stopped would leave an engine nobody is holding.
  const epoch = miningEpoch;
  if (clearedCudaLine) send('miner:log', { level: 'info', line: clearedCudaLine });
  // PEARL_GPU_INDEX can narrow the cards. A value the app can't use, and a
  // listed card nvidia-smi doesn't show, are logged rather than dropped quietly.
  const gpuEnv = readGpuIndexEnv(process.env);
  if (gpuEnv.warning) send('miner:log', { level: 'warn', line: gpuEnv.warning });
  // The same nvidia-smi round also reads each card's compute capability and the
  // driver version, which decide whether this rig loads the CUDA 12.8 core or
  // the CUDA 13 one (shared/coreVariant). The factory logs its choice.
  const [gpus, cudaCards] = await Promise.all([
    probe.detectMinerGpus(process.env, gpuEnv.indices), probe.detectCudaCards(),
  ]);
  if (epoch !== miningEpoch) return;
  for (const line of describeSkippedGpus(gpuEnv.indices, gpus, 'PEARL_GPU_INDEX')) {
    send('miner:log', { level: 'warn', line });
  }

  // Which cards lock their memory clock while mining: the RTX 5090 default
  // (shared/memClock), or LLMJOB_MINE_MEM_CLOCK from the app's environment, the
  // GUI's one switch for it. The renderer has no setting yet, so a rig that
  // regresses with the lock needs a way off it: 0 leaves every card at the
  // driver's clock, any other value locks every mining card at it, the same as
  // --mine-mem-clock.
  // Read on every start, like the CLI reads its flag; a bad value is logged and
  // ignored, so a typo keeps the default rather than silently dropping it. Not
  // while the LLM co-runs: llama-server is memory-bandwidth-bound, and here
  // 'auto' always co-runs (resolvePlan gives { miner, llm } and runPlan starts
  // both; the GUI has no demand mode), so plan.llm is the whole of the GUI's
  // co-run condition. The plan rides on the start call, not on `settings`,
  // so persistSettings never writes it to disk.
  const memRequest = readMemClockEnv(process.env);
  if (memRequest.warning) send('miner:log', { level: 'warn', line: memRequest.warning });
  const memPlan = planMemClocks({
    requestedMhz: memRequest.mhz, requestName: GUI_MEM_CLOCK_ENV,
    cards: cudaCards, gpus, llmCoRuns: !!llmCoRuns,
  });
  if (memPlan.reason) send('miner:log', { level: memPlan.dropped ? 'warn' : 'info', line: memPlan.reason });

  miner = new PearlEngine({
    connect: (host, port) => net.connect({ port, host, lookup: poolLookup() }),
    createCore: coreFactory({
      resourcesPath: process.resourcesPath,
      gpus,
      cards: cudaCards,
      log: (level, line) => send('miner:log', { level, line }),
    }),
    // The card temperature the UI shows next to the GPU name. Our core has no
    // NVML reading of its own to forward the way alpha-miner did, so the engine
    // polls nvidia-smi for it; a rig without nvidia-smi just shows the name.
    readTemps: () => probe.detectGpuTemps(),
  });
  wireMinerEvents(miner, endpoint);
  try {
    miner.start(Object.assign({}, settings, {
      endpoint, gpus, gpu: settings.gpu || null,
      mineMemClockByIndex: memPlan.byIndex, mineMemClockDefault: memPlan.isDefault,
    }));
  } catch (e) {
    reportLaunchFailure(e);
  }
  // A prompt that is running now keeps its card from mining.
  promptPause.reapply();
}

// Miner event wiring. PearlEngine translates our miner's events into the two
// the UI reads (a periodic `status` and one `connected`), so nothing
// downstream of here knows what produced them.
function wireMinerEvents(engine, endpoint) {
  engine.on('log', (l) => send('miner:log', l));
  // Said once, not on every 5s retry. The engine's own line repeats verbatim and
  // names neither the host it tried nor the fact that the name never resolved,
  // so a rig that cannot do DNS looks identical to a pool that is down — which
  // is how a user ends up staring at eight lines of "No such host is known".
  let dnsHinted = false;
  engine.on('event', (e) => {
    applyEvent(stats, e, Date.now());
    if (e.type === 'connect-failed' && e.dns && !dnsHinted) {
      dnsHinted = true;
      send('miner:log', {
        level: 'warn',
        line: 'could not resolve ' + endpoint + ' — nothing is wrong with the GPU.'
          + ' Check DNS/VPN/firewall, or pick another region in Settings.',
      });
    }
  });
  engine.on('error', (err) => reportLaunchFailure(err));
  engine.on('stopped', (code) => {
    send('miner:log', { level: 'info', line: 'engine exited (code ' + code + ')' });
    // Code 1: every card failed and the engine stopped itself. With nothing
    // else running the START ends, as when the LLM stops on its own.
    if (code === 1 && engine === miner) endIfNothingRuns(llmRunsOrComing());
  });
  // The pool took the login, so this address is one that worked.
  engine.on('authorized', (e) => poolLookup().worked(e.host, e.address, e.family));
}

function stopMining() {
  // Cancel any start still in flight (see miningEpoch) so it doesn't spawn or
  // start the LLM after this stop.
  miningEpoch++;
  haltMiner();
  stopCpuMining();
  send('miner:stopped');
}

// Stop the GPU engine, its stats ticker and its board reports, and forget it.
// Half of stopMining(): runPlan also uses it alone, when a plan drops mining
// but the session goes on with the LLM (and the CPU miner, if it runs).
function haltMiner() {
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
}

function appIcon() {
  const dir = path.join(__dirname, '..', 'assets');
  return path.join(dir, process.platform === 'win32' ? 'icon.ico' : 'icon.png');
}

// The tray icon. The macOS menu bar wants a 16-point image, and icon.png is the
// 1024 px app icon, so it is scaled down there, with a 32 px copy for Retina.
function trayIcon() {
  if (process.platform !== 'darwin') return appIcon();
  const full = nativeImage.createFromPath(appIcon());
  const icon = full.resize({ width: 16, height: 16 });
  icon.addRepresentation({ scaleFactor: 2, buffer: full.resize({ width: 32, height: 32 }).toPNG() });
  return icon;
}

// Detect the machine's GPU for the settings/device label. Uses Windows'
// Win32_VideoController via PowerShell (already a dependency of the llama unzip);
// resolves to a display name or null. Never rejects.
// Delegates to the shared probe (nvidia-smi, then WMI on Windows) rather than
// keeping a Windows-only copy here — that copy returned null on Linux, so the
// shipped AppImage showed no device at all.
// The renderer's IPC contract is a plain name string, so unwrap it here.
async function detectGpu() {
  const info = await probe.detectGpuInfo();
  return info ? info.name : null;
}

// Wire electron-updater to the renderer's update bar. autoUpdater pulls from the
// GitHub Releases feed (see build.publish); it only works in a packaged app, so
// main.js guards the call with app.isPackaged. Downloads happen automatically.
// With the window open, the user chooses when to restart via the
// 'app:update:install' channel; out of sight, it installs on its own (see
// installWhenHidden).
let manualUpdateCheck = false; // true while a user-initiated check is in flight
let updateTimer = null; // periodic re-check, so startup is not the only chance
let updateReady = false; // an update has downloaded and waits to install

// How long after an update's restart the new copy still counts as that restart,
// for starting in the tray (see resumeHidden in whenReady).
const UPDATE_RESTART_MS = 10 * 60 * 1000;

// Install the downloaded update now and restart into it. `hidden`: the window
// was out of sight, so the new copy starts in the tray too.
function installUpdate(hidden) {
  // Mining now, or paused waiting for the computer to go idle: resume after the
  // restart.
  const resume = !!session;
  // The relaunch needs this window to really close, not hide to the tray.
  quitting = true;
  try {
    // isSilent=true: install to the existing directory without re-showing the
    // assisted-installer wizard. isForceRunAfter=true: relaunch the app afterwards.
    autoUpdater.quitAndInstall(true, true);
  } catch (e) {
    quitting = false;
    send('miner:log', { level: 'error', line: 'update install failed: ' + e.message });
    return;
  }
  // An install the updater refused (no installer file, say) is reported through
  // its 'error' event and returns without quitting. The app carries on as it
  // was: a live START, and a window that closes to the tray.
  if (autoUpdater.quitAndInstallCalled === false) {
    quitting = false;
    return;
  }
  const restart = {};
  if (resume) restart.resumeMining = true;
  if (hidden) restart.resumeHidden = Date.now();
  if (resume || hidden) persistSettings(Object.assign({}, loadSettings(), restart));
  // Stop the LLM and miner now, before the installer relaunches the app.
  // llama-server binds a fixed port (8080); if the outgoing server still holds it,
  // the resumed instance's llama-server can't bind and the LLM silently fails to
  // start (the miner binds no port, so it's fine — exactly the "miner came back,
  // LLM didn't" symptom). Stopping here frees the port and VRAM before the new
  // instance starts. The quit itself comes on the next tick.
  endSession();
}

// An update installs when the app quits, and with the window closing to the
// tray the app rarely quits: a rig started at login could run an old build for
// months. So a downloaded update installs as soon as the window is out of sight
// (in the tray, or never shown after a launch at login), and the app restarts
// in the tray and picks its START back up. An open window keeps the "Update &
// restart" button instead.
//
// Not while a START is still bringing up the model: on a first run that is a
// download of several GB, which the restart would throw away. And not where the
// installer would ask for an administrator's approval, which nobody is there to
// give: an install for all users, in Program Files.
function installWhenHidden() {
  if (!updateReady || quitting || planRun) return;
  if (win && !win.isDestroyed() && win.isVisible()) return;
  if (!canInstallQuietly()) return;
  installUpdate(true);
}

// Whether this user can write to the app's own folder, which is what an update
// replaces. Windows only: it is the install for all users that elevates. Tried
// with a real file, because fs.access ignores Windows' folder permissions.
function canInstallQuietly() {
  if (process.platform !== 'win32') return true;
  const probe = path.join(path.dirname(process.execPath), '.update-probe-' + process.pid);
  try {
    fs.writeFileSync(probe, '');
    fs.unlinkSync(probe);
    return true;
  } catch (e) {
    return false;
  }
}

// electron-updater reads resources/app-update.yml before every check. A crash or
// power cut during an install can leave that file unreadable (one Windows
// install's was all NUL bytes), and every check then fails until a reinstall.
// Point the updater at a good copy instead. The next update writes a good file
// again, and this then finds nothing to do.
function useGoodUpdateConfig() {
  try {
    const file = repairUpdateConfig({
      resourcesPath: process.resourcesPath, userDataPath: app.getPath('userData'), fs, join: path.join,
    });
    if (!file) return;
    autoUpdater.updateConfigPath = file;
    send('miner:log', { level: 'warn', line: 'the updater\'s app-update.yml was missing or damaged — using a good copy from the app\'s data folder' });
  } catch (e) {
    send('miner:log', { level: 'error', line: 'could not repair the updater\'s app-update.yml: ' + e.message });
  }
}

function setupUpdater() {
  const push = (phase, payload) => send('app:update', formatUpdate(phase, payload));
  useGoodUpdateConfig();
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
  autoUpdater.on('update-downloaded', (info) => {
    updateReady = true;
    push('ready', info);
    installWhenHidden();
  });
  // An AppImage update moves to a new file named for its version and deletes
  // this one. Point a start-at-login entry at the new file now: after an install
  // on quit, the next start may be the login itself, and an entry naming the
  // deleted file would start nothing.
  autoUpdater.on('appimage-filename-updated', (file) => refreshLoginItem(file));
  autoUpdater.on('error', (err) => {
    manualUpdateCheck = false;
    push('error');
    send('miner:log', { level: 'error', line: 'update check failed: ' + describeUpdateError(err) });
  });
  // checkForUpdates() emits 'error' and THEN rethrows (AppUpdater.js), so the
  // handler above has already logged by the time the promise rejects. Logging
  // here too printed every network blip twice in the user's log.
  const check = () => autoUpdater.checkForUpdates().catch(() => {});
  check();
  // Re-check on a timer. The startup check used to be the only one for the
  // life of the process, so a rig that launched while GitHub's releases feed
  // was down never looked again — it would sit on a broken build until someone
  // restarted it, which is the opposite of what auto-update is for.
  updateTimer = setInterval(check, NETWORK.updateCheckIntervalMs);
  updateTimer.unref();
}

// User-initiated "Check for updates". In a dev/unpackaged run the real updater
// isn't wired, so walk the UI through checking → up-to-date so the button still
// gives feedback (the installed app runs a real check below).
function checkForUpdate() {
  // macOS: there is no in-app update to check for (the build is ad-hoc signed,
  // so Squirrel.Mac would refuse to install whatever it downloaded — see
  // shared/platform.autoUpdateSupported). Say so and open the Releases page,
  // which is the actual next step, rather than run a check that can only fail.
  if (!autoUpdateSupported(process.platform)) {
    send('app:update', formatUpdate('manual'));
    send('miner:log', { level: 'info', line: 'in-app updates are unavailable on macOS — opening ' + NETWORK.releasesUrl });
    openExternalSafe(NETWORK.releasesUrl);
    return;
  }
  if (!app.isPackaged) {
    send('app:update', formatUpdate('checking'));
    setTimeout(() => send('app:update', formatUpdate('latest')), 700);
    return;
  }
  manualUpdateCheck = true;
  send('app:update', formatUpdate('checking'));
  // Same as the startup check: the 'error' handler in setupUpdater has already
  // reset the flag, pushed the error state and logged the reason by the time
  // this rejects.
  autoUpdater.checkForUpdates().catch(() => {});
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
  // A page that loads, or loads again, is not listening for log lines until
  // its onLog says so (see send).
  win.webContents.on('did-start-loading', () => { logListening = false; });
  win.loadFile(path.join(__dirname, '..', 'renderer', 'index.html'));

  // A launch at login stays in the tray. Only when there is a tray to find it
  // in: without one, a hidden window could never be opened.
  const showOnLoad = !(hiddenLaunch && tray);
  hiddenLaunch = false;

  // Size the window to the rendered content so there's no default scrollbar and
  // no trailing whitespace, whatever the platform chrome / font metrics / DPI.
  // Do it before showing (window starts hidden) to avoid a resize flash. The
  // renderer keeps overflow-y:auto, so a transient taller state (update bar,
  // engine error) still scrolls rather than clipping.
  win.webContents.on('did-finish-load', () => {
    fitWindowToContent().finally(() => { if (showOnLoad && win && !win.isDestroyed()) win.show(); });
  });
  setTimeout(() => { if (showOnLoad && win && !win.isDestroyed() && !win.isVisible()) win.show(); }, 1500);

  // Closing the window hides it to the tray, and mining carries on, unless the
  // user turned that off in Settings or the app is really quitting.
  //
  // Only when the user closed it. On Windows the window's X, Alt+F4 and the
  // taskbar's Close all arrive as SC_CLOSE first. A close without one comes from
  // another program that wants the app gone: the installer before it replaces
  // the app's files, the uninstaller, Task Manager's End task. Hiding the window
  // from those left the app running until the installer force-killed it a second
  // later, so the miner never stopped cleanly. They quit it instead. A focused
  // window counts as the user's too, in case a close reaches it some other way.
  let userClose = false;
  if (process.platform === 'win32') {
    win.hookWindowMessage(WM_SYSCOMMAND, (wParam) => {
      if (Buffer.isBuffer(wParam) && (wParam.readUInt32LE(0) & 0xfff0) === SC_CLOSE) userClose = true;
    });
  }
  win.on('close', (e) => {
    const fromUser = process.platform !== 'win32' || userClose || win.isFocused();
    userClose = false;
    if (quitting || !fromUser || !tray || !getPrefs().closeToTray) return;
    e.preventDefault();
    win.hide();
    trayHint();
  });
  // Out of sight, a downloaded update can install.
  win.on('hide', installWhenHidden);
  // Logging off or shutting down on Windows: let the window close.
  win.on('session-end', () => { quitting = true; });

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

// Bring the window back: from the tray, or when the app is opened again while
// it runs (second-instance).
function showWindow() {
  if (!win || win.isDestroyed()) {
    createWindow();
    return;
  }
  if (win.isMinimized()) win.restore();
  win.show();
  win.focus();
}

function quitApp() {
  quitting = true;
  app.quit();
}

// The tray icon: the way back to a closed window, and the way to quit.
function createTray() {
  try {
    tray = new Tray(trayIcon());
  } catch (e) {
    // No tray on this desktop. Closing the window then quits, as before.
    tray = null;
    return;
  }
  tray.setToolTip('LLMJob Earn');
  tray.setContextMenu(Menu.buildFromTemplate([
    { label: 'Open LLMJob Earn', click: showWindow },
    { type: 'separator' },
    { label: 'Quit', click: quitApp },
  ]));
  tray.on('click', showWindow);
}

// GNOME draws no tray icon without an AppIndicator extension, though Electron
// makes one without complaint (see shared/trayHost). There, drop the icon, so
// closing the window quits rather than hiding it with no way back. Waits up to
// `waitMs` for the extension, which may still be loading at login.
const TRAY_WAIT_MS = 15000;
async function confirmTray(waitMs) {
  if (!tray || process.platform !== 'linux' || !trayHost.isGnome(process.env)) return;
  const deadline = Date.now() + waitMs;
  for (;;) {
    // false means no host; null means the bus couldn't be asked, which keeps
    // the icon, as before.
    if ((await trayHost.queryTrayHost(execFile)) !== false) return;
    if (Date.now() >= deadline) break;
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  tray.destroy();
  tray = null;
}

// The first time the window closes to the tray, say so once. Otherwise the app
// looks closed while the GPU is still working. Marked shown only once the
// notification was: one that never appeared is tried again on the next run.
let trayHintTried = false;
function trayHint() {
  if (getPrefs().trayHintShown || trayHintTried) return;
  trayHintTried = true;
  if (!Notification.isSupported()) {
    savePrefs({ trayHintShown: true });
    return;
  }
  const notice = new Notification({
    title: 'LLMJob Earn is still running',
    body: 'It keeps running in the tray. Click the tray icon to open it, or right-click it to quit. '
      + 'You can change this in Settings.',
  });
  notice.on('show', () => savePrefs({ trayHintShown: true }));
  notice.show();
}

// How far the frame may sit from the measured content before it is worth
// resizing. A pixel or two is invisible; chasing it is not, because the resize
// itself perturbs the layout (see below).
const FIT_TOLERANCE_PX = 2;

// Measure the rendered content (.app) and set the window's content area to it,
// so the miner view fits exactly. Best-effort — resolves regardless of errors.
//
// Only resizes when the frame is actually wrong. Re-applying a size that is
// already correct is not free: setContentSize round-trips the width back
// through getContentSize, and on a DPI-scaled display that round trip is not
// lossless, so a "no-op" fit still nudged the frame by a pixel. The renderer's
// ResizeObserver saw .app change and asked for another fit, which nudged it
// again — the window resized continuously on startup until a minimize/restore
// settled it, and any manual resize set it going again.
function fitWindowToContent() {
  if (!win || win.isDestroyed()) return Promise.resolve();
  return win.webContents
    .executeJavaScript('Math.ceil((document.querySelector(".app") || document.body).getBoundingClientRect().height)')
    .then((h) => {
      if (!win || win.isDestroyed() || !Number.isFinite(h) || h <= 0) return;
      const [width, current] = win.getContentSize();
      if (Math.abs(current - h) <= FIT_TOLERANCE_PX) return;
      win.setContentSize(width, h);
    })
    .catch(() => {});
}

// ── Local LLM (llama.cpp llama-server), run alongside the miner ─────────────

function sendLlmStatus() { send('llm:status', llmStatus); }

// The LLM row when a card gives up on its GPU and nothing else serves, and when
// no card is left to start on (see startLlm). START LLM clears it ('llm:retry').
const GPU_LOST_ERROR = 'GPU kept failing — START LLM to retry';

// An instance gave up on its card after losing the GPU again and again
// (LlmManager 'gpu-lost'). Remember the card and the model, so the next START
// or launch at login does not load that model there again.
function rememberGpuLoss(e) {
  const keys = gpuLoss.saveKeys(e.index, llmCards, llmBusIds, e.busId);
  saveGpuLost(gpuLoss.remember(loadGpuLost(), keys, servingModel.name));
  send('miner:log', { level: 'error', line: servingModel.name + ' kept losing ' + (e.index == null ? 'the GPU' : 'GPU ' + e.index)
    + '. It won\'t be started there again until the app updates or you press START LLM on the Chat tab.' });
}

// What fetches llama-server, the model and its projector into `dir`. None of
// them is bundled: each is downloaded on first use.
function llmFiles(dir) {
  // The server ships as a folder of shared libraries + the binary, so extraction
  // must keep them together: Windows flattens the zip with PowerShell; elsewhere
  // extractLlamaZip sniffs the archive (tar.gz on Linux/macOS, so `tar
  // --strip-components=1`) and flattens it the same way. macOS needs exactly
  // that: llama-server's only LC_RPATH is `@loader_path`, so it finds its
  // dylibs — including libggml-metal — beside itself and nowhere else.
  const extract = process.platform === 'win32' ? extractLlamaZipWin : (zip, dest) => extractLlamaZip(zip, dest);
  return llmEngine(dir, extract);
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
  llmStatus = Object.assign({}, llmStatus, { ready: false, tokensPerSec: 0, note: null });
  sendLlmStatus();
}

// Set (or clear) the transient stage line and push it to the renderer.
function setLlmNote(note) {
  llmStatus = Object.assign({}, llmStatus, { note: note || null });
  sendLlmStatus();
}

// A download-progress callback that reports to both the hero line and the log.
// downloadFile fires onProgress for every chunk, which at a few hundred chunks a
// second would flood the IPC and bury the log — so emit only on a 5-point move,
// or every 2s for a slow link, plus the final 100%.
function progressReporter(label, now) {
  const clock = now || Date.now;
  let lastPct = -1;
  let lastAt = 0;
  return (pct) => {
    // A download goes on after STOP (it can't be cancelled), and its progress
    // would bring back a note the stop cleared.
    if (!session) return;
    // progressPercent returns null when the server sends no content-length, and
    // Number(null) is 0 — so null must be rejected explicitly or a sizeless
    // download would sit at a permanent, wrong "0%".
    const p = Math.floor(Number(pct));
    if (pct == null || !Number.isFinite(p) || p < 0) return;
    const at = clock();
    if (p !== 100 && p - lastPct < 5 && at - lastAt < 2000) return;
    if (p === lastPct) return;
    lastPct = p;
    lastAt = at;
    setLlmNote(label + ' ' + p + '%');
    send('miner:log', { level: 'info', line: label + ' ' + p + '%' });
  };
}

// The model for a card with `freeMb` free and `reserveMb` set aside for mining,
// without the models it kept losing its GPU under (`lost`, see rememberGpuLoss).
// The default is never left out, so it is still the fallback.
function modelFor(freeMb, reserveMb, lost) {
  const models = allModels().filter((m) => m === LLM.model || !lost.includes(m.name));
  return pickModel(freeMb, reserveMb, models);
}

// Start the local LLM: ensure binary + model, size the GPU offload to leave
// `reserveMb` free for mining, spawn llama-server, and surface its OpenAI
// endpoint. Best-effort — failures are logged, never thrown to the UI. Returns
// whether the server was actually started (callers use it to reset the UI when
// an LLM-only session ends up running nothing).
//
// `epoch` is the run's miningEpoch. A STOP or an idle pause during the awaits
// below (a first run downloads several GB) moves it, and the run then stops
// short instead of starting a model for a START that is gone, or leaving the
// LLM row on "Starting…" after a STOP. A pause queues the next run, which picks
// up from the files this one downloaded.
async function startLlm(reserveMb, epoch) {
  const stale = () => epoch !== miningEpoch;
  // A fleet already up, spawned or adopted. An adopted server has no manager, so
  // hasSpawned() alone missed it, and every idle pause and resume adopted the
  // same server again, with another cluster worker each time.
  if (fleet && (fleet.hasSpawned() || fleet.isReady())) return true;

  // One fleet drives both paths: adopt a lingering server, or spawn one instance
  // per eligible GPU. Build it once with the process/GPU factories injected.
  if (!fleet) fleet = buildFleet();

  // If a healthy llama-server is already on our port (e.g. one lingering from a
  // just-restarted session), adopt it instead of spawning a second process that
  // would double-load the model and risk an OOM. Reusing it is safe — it serves
  // the same model on the same OpenAI endpoint.
  const targetBase = 'http://' + LLM.host + ':' + LLM.port;
  const healthy = await probeLlmHealth(targetBase);
  if (stale()) return false;
  if (healthy) {
    // Ask it what it serves. It is often the 27B a 5090 was running before an
    // update, and this run has planned nothing, so servingModel still held
    // the default: the hero, the board, telemetry and every cluster job named
    // the wrong model. Its name is our alias for a catalogue model; a name we
    // don't know keeps the old label. Its window is its own, not a ladder's.
    const served = parseServedModel(await getJson(targetBase + '/v1/models', { timeout: 2000 }));
    if (stale()) return false;
    const known = served && allModels().find((m) => m.name === served.name);
    if (known) servingModel = known;
    const ctxSize = (served && served.ctxSize) || ctxLadder(servingModel)[0];
    fleet.adopt(targetBase, { ctxSize });
    llmEverReady = true;
    // `held` too: a server this run did not plan was not held to any model, and
    // an old "kept losing GPU" sentence stayed in the LLM tooltip.
    llmStatus = Object.assign({}, llmStatus, {
      ready: true, error: null, note: null, held: null, endpoint: targetBase + '/v1', webUrl: targetBase,
      model: servingModel.name, vramMb: servingModel.vramFullMb, ctxSize,
    });
    send('miner:log', { level: 'info', line: 'local LLM already running on ' + targetBase + ' — reusing it' });
    sendLlmStatus();
    syncLanShare();
    syncWorker();
    warmUpLlm(targetBase);
    return true;
  }

  // Plan one llama-server per eligible GPU: the model is small enough to hold a
  // copy on every card with room, so a multi-GPU rig serves from all of them
  // (each pinned to its card) instead of only the best card. An empty plan
  // means VRAM was measured but nothing fits — refuse, as the single-card path
  // did; when VRAM can't be read the planner returns one unknown-placement
  // instance and lets llama.cpp decide.
  const [cards, busIds] = await Promise.all([detectGpusVram(), probe.detectPciBusIds()]);
  if (stale()) return false;
  llmBusIds = busIds;
  llmCards = cards;
  // A model that kept losing the GPU on a card is not loaded there again until
  // the app updates or the user presses START LLM (see rememberGpuLoss). A card
  // that lost it even under the smallest model is left out altogether. Without
  // nvidia-smi there are no cards, and the one llama.cpp picks is index null.
  const gpuLost = loadGpuLost();
  const lostOn = (index) => gpuLoss.lostModels(gpuLost, gpuLoss.lookupKeys(index, cards, busIds));
  const usable = cards.filter((c) => !lostOn(c.index).includes(LLM.model.name));
  if (cards.length ? !usable.length : lostOn(null).includes(LLM.model.name)) {
    const why = 'not starting the local LLM: it kept losing the GPU even with ' + LLM.model.name
      + '. Press START LLM on the Chat tab to try again.';
    send('miner:log', { level: 'error', line: why });
    llmStatus = Object.assign({}, llmStatus, { ready: false, note: null, error: GPU_LOST_ERROR, held: why });
    sendLlmStatus();
    return false;
  }
  // Which model this run serves. Chosen from the best card's free VRAM, because
  // the fleet loads ONE model across every instance — planLlmInstances then drops
  // any card that cannot hold it. On a mixed rig that trades breadth for
  // capability: a box with one 32 GB card and three small ones serves the large
  // model from the one card rather than the small model from four. That is the
  // intended reading of "for cards that can support it", but it is a trade, and
  // the small-card case is unaffected because those rigs resolve to the default.
  const bestCard = pickLlmGpu(usable);
  const bestIndex = bestCard ? bestCard.index : null;
  const bestFree = bestCard ? bestCard.freeMb : null;
  const lost = lostOn(bestIndex);
  const wanted = modelFor(bestFree, reserveMb || 0, []);
  // The same choice without the models this card kept losing the GPU under:
  // the next one down, or `wanted` itself when it isn't one of them.
  const model = modelFor(bestFree, reserveMb || 0, lost);
  const held = model.name === wanted.name ? null : wanted.name + ' kept losing GPU ' + bestIndex + ', so '
    + model.name + ' runs there instead. To try it again, press STOP, then START LLM on the Chat tab.';
  if (held) send('miner:log', { level: 'warn', line: held });
  servingModel = model;
  llmPlannedOn = { index: bestIndex, freeMb: bestFree, lost };
  // The hero shows this name, and it was seeded once at startup and never
  // updated — fine while every node ran the same model, wrong the moment one
  // doesn't. Set before the first failure path below so even "Needs ~N GB free
  // VRAM" names the model it was actually short for.
  llmStatus = Object.assign({}, llmStatus, { model: model.name, vramMb: model.vramFullMb, ctxSize: ctxLadder(model)[0], held });
  // Any other card that lost the GPU under this model is left out too.
  const plan = planLlmInstances(usable.filter((c) => !lostOn(c.index).includes(model.name)), model, reserveMb || 0);
  if (!plan.length) {
    // An empty plan means VRAM was measured but no card had room — so at least
    // one card parsed, and pickLlmGpu (same parse rules) returns it for the error.
    const best = pickLlmGpu(cards);
    const freeMb = best.freeMb;
    // Quote the constraint that actually refused the card, not just the preflight
    // floor: when co-running, the offload budget (model + mining reserve) is the
    // larger of the two, and reporting the floor told users they needed less than
    // they did — then refused them anyway.
    const needMb = requiredFreeMb(model, reserveMb);
    const needGb = Math.round(needMb / 1024);
    send('miner:log', { level: 'error', line: 'not enough free VRAM for the local LLM: ' + freeMb
      + ' MB free, need ~' + needMb + ' MB for ' + model.name + ' — skipping the LLM.' });
    llmStatus = Object.assign({}, llmStatus, { ready: false, note: null, error: 'Needs ~' + needGb + ' GB free VRAM' });
    sendLlmStatus();
    return false;
  }

  const dir = path.join(app.getPath('userData'), 'llm');
  send('miner:log', { level: 'info', line: 'preparing local LLM (' + model.name + ')…' });

  // Both of these are no-ops once cached, but on a first run they fetch ~100 MB
  // of llama-server and a ~5 GB model. Report progress so the wait is legible
  // instead of looking like a stall.
  let binaryPath, modelPath, mmprojPath = null;
  try {
    setLlmNote('Preparing…');
    const engine = llmFiles(dir);
    binaryPath = await engine.ensureServer(progressReporter('downloading llama-server…'));
    modelPath = await engine.ensureModel(progressReporter('downloading model ' + model.name + '…'), model);
    // A vision model needs its projector too. Fetched after the weights and
    // reported separately: it is ~1 GB against ~18, so folding it into the same
    // progress line would make the last 5% look like a stall.
    mmprojPath = await engine.ensureMmproj(progressReporter('downloading vision projector…'), model);
  } catch (e) {
    if (stale()) return false;
    // Surface the failure on the hero, not only in the log. Clearing the note
    // without setting an error dropped the row straight back to a grey dot and
    // the model name — identical to "not started yet" — so a download that died
    // at 57% after twenty minutes looked exactly like nothing having happened,
    // with the reason buried in the Logs tab. Same reasoning as the note itself:
    // the failing path deserves the legibility the working path just got.
    llmStatus = Object.assign({}, llmStatus, { ready: false, note: null, error: 'Setup failed — see Logs' });
    sendLlmStatus();
    send('miner:log', { level: 'error', line: 'LLM setup failed: ' + e.message });
    return false;
  }
  if (stale()) {
    // A pause's queued run takes over and sets its own note. After a STOP, the
    // download's last progress note goes.
    if (!session) setLlmNote(null);
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
    const dllPlan = runtimeCopyPlan({
      platform: process.platform, binDir: path.dirname(binaryPath),
      resourcesPath: process.resourcesPath, existsFn: fs.existsSync, joinFn: path.join,
    });
    dllPlan.forEach((c) => {
      fs.copyFileSync(c.from, c.to);
      send('miner:log', { level: 'info', line: 'installed LLM runtime DLL: ' + path.basename(c.to) });
    });
  } catch (e) {
    send('miner:log', { level: 'error', line: 'could not install the LLM runtime DLLs: ' + e.message });
  }

  // llama-server counts cards its own way, so ask it for its list and put each
  // instance on the card it was planned for (shared/llama.placeLlmPlan). After
  // the DLL copy: without them the listing can't run on Windows. With the
  // model, it reads each card's bus id too, which prompt pause needs.
  const placed = placeLlmPlan(plan, cards, await probe.detectLlmDevices(binaryPath, modelPath), busIds);
  if (stale()) {
    if (!session) setLlmNote(null);
    return false;
  }
  for (const w of placed.warnings) send('miner:log', { level: 'warn', line: w });
  send('miner:log', { level: 'info', line: describeLlmStart(placed.plan) });
  // Loading a multi-GB model off disk takes seconds even when nothing downloads,
  // and until 'ready' fires the hero is otherwise indistinguishable from idle.
  llmStatus = Object.assign({}, llmStatus, { ready: false, error: null, note: 'Starting…' });
  sendLlmStatus();
  await fleet.start(placed.plan, llmRunOptions(model, { binaryPath, modelPath, mmprojPath }));
  return true;
}

// Construct an LlmFleet wired to the GUI (llmLaunch.makeLlmFleet), building a
// cluster job-worker per ready instance. Aggregate fleet events map onto the
// same IPC the single-instance flow used.
function buildFleet() {
  llmEverReady = false;
  serveLogged = false;
  const f = makeLlmFleet({
    makeWorker: (baseUrl) => makeJobWorker(baseUrl),
    // An adopted server has no process of ours to watch, so it is asked.
    probeHealth: probeLlmHealth,
  });
  f.on('log', (l) => send('miner:log', l));
  f.on('first-ready', ({ baseUrl }) => warmUpLlm(baseUrl)); // one warm-up per run
  f.on('ready', ({ baseUrl }) => {
    llmEverReady = true;
    // webUrl() is the first ready instance's base URL — non-null here, since this
    // instance was just marked ready before the event fired.
    const web = f.webUrl();
    // `error: null` alongside ready, matching the adopt path: a model that is up
    // and answering must not still be wearing an error. The renderer ranks error
    // above ready in both the hero dot and its label, so a stale one outranks the
    // truth — and nothing else clears it once the fleet is running, since
    // startLlm's pre-spawn reset only runs on a fresh start.
    llmStatus = Object.assign({}, llmStatus, {
      ready: true, error: null, endpoint: web + '/v1', webUrl: web, note: null, ctxSize: f.ctxSize(),
    });
    send('miner:log', { level: 'info', line: 'local LLM ready — OpenAI endpoint ' + baseUrl + '/v1' });
    sendLlmStatus();
    syncLanShare();
    syncWorker(); // serve cluster jobs once a model is up, if we're linked
  });
  f.on('stats', ({ tokensPerSec }) => { llmStatus = Object.assign({}, llmStatus, { tokensPerSec }); sendLlmStatus(); });
  // A card's server died and is restarting. With no other card serving, the
  // row said ready all through it while chat and cluster jobs failed. It says
  // "Restarting…" until 'ready' comes back.
  f.on('crashed', () => {
    if (f.isReady()) return;
    llmStatus = Object.assign({}, llmStatus, { ready: false, note: 'Restarting…' });
    sendLlmStatus();
  });
  // A card would not load at its context size and is retrying at the next one
  // down. Show the window it will really have, not the ladder's top.
  f.on('ctx-downgrade', () => {
    llmStatus = Object.assign({}, llmStatus, { ctxSize: f.ctxSize() });
    sendLlmStatus();
  });
  f.on('busy', (e) => {
    const card = cardForPrompt(e, llmBusIds);
    if (card !== undefined) promptPause.setBusy(card, e.busy);
  });
  // An instance gave up on its card (see rememberGpuLoss), and its 'stopped'
  // follows. The row says why when no other card is still serving.
  f.on('gpu-lost', (e) => {
    rememberGpuLoss(e);
    if (!f.isReady()) {
      llmStatus = Object.assign({}, llmStatus, { error: GPU_LOST_ERROR });
      sendLlmStatus();
    }
  });
  // Logged, not swallowed: a binary that cannot be run says why only here.
  // The fleet keeps any other cards up.
  f.on('error', (e) => send('miner:log', { level: 'error', line: 'local LLM error: ' + e.message }));
  f.on('stopped', () => {
    // A fleet that stopped on its own is not used again: the next run builds a
    // new one, which starts its own "ready" and "serving" flags. A dead one
    // kept llmEverReady from its last run, which hid the error below, and it
    // never says 'stopped' a second time.
    if (fleet === f) fleet = null;
    cancelChat('the local LLM stopped');
    resetLlmStatus();
    syncLanShare();
    // Every instance died before any became ready while mining keeps running, or
    // while idle-only has it paused — the silent-failure case (typically a
    // port-bind/OOM), not a user stop.
    if (!llmEverReady && ((miner && miner.isRunning()) || (session && session.paused))) {
      llmStatus = Object.assign({}, llmStatus, { ready: false, error: 'The local LLM stopped before it was ready. See Logs.' });
      sendLlmStatus();
    }
    // An LLM-only session ends when the fleet exits, and so does one whose miner
    // never started — or the UI keeps showing a running session with nothing
    // running, and the idle watcher can bring things back behind it (see
    // runPlan).
    endIfNothingRuns(miner && miner.isRunning());
  });
  return f;
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

// The model a run that mines should serve in place of the running one, or null
// to keep it. A model planned for the card alone (Compute Mode LLM) can be
// joined by the miner with no STOP between: Auto picked in Settings, then START
// LLM. With the mining reserve the same card may get another model: the 27B
// plus the miner does not fit a 32 GB card, which serves Gemma. Judged on what the fleet was
// planned on, because a reading now counts the fleet itself, and with the same
// models left out, so a card held to Gemma (see rememberGpuLoss) keeps it.
//
// Only a fleet we spawned. An adopted server has no manager, so stopping the
// fleet would not stop it, and the next plan would adopt it again.
function llmReplan() {
  if (!fleet || !fleet.hasSpawned()) return null;
  const next = modelFor(llmPlannedOn.freeMb, LLM.miningReserveMb, llmPlannedOn.lost);
  const same = next.name === servingModel.name && ctxLadder(next)[0] === ctxLadder(servingModel)[0];
  return same ? null : next;
}

// Stop the running fleet for a fresh plan, and wait until the card is back.
// stopAndWait stops the fleet itself; stopLlm then does the rest of a STOP. A
// fleet says no 'stopped' for a stop it was asked for, so its handler neither
// ends the session nor shows an error.
async function stopLlmForReplan(needMb, epoch) {
  const exited = fleet.stopAndWait();
  stopLlm();
  await exited;
  // An exited server can still hold its VRAM for a moment (the CLI's
  // waitForFreeVram). A miner started then fails to build its core, and a plan
  // made then reads the card as full. About 8s at most, as the CLI waits; a
  // card that is really short is left for the plan to refuse. The card the
  // model was planned on, not the emptiest: on a rig with a second card, that
  // one has room at once.
  for (let i = 0; i < 32 && epoch === miningEpoch; i++) {
    const card = pickLlmGpu((await detectGpusVram()).filter((c) => c.index === llmPlannedOn.index));
    if (!card || card.freeMb >= needMb) return;
    await new Promise((r) => setTimeout(r, 250));
  }
}

function stopLlm() {
  cancelChat('the local LLM was stopped');
  // Nothing is left to answer a prompt, so every card mines now rather than
  // after the usual wait.
  promptPause.reset();
  if (fleet) { fleet.stop(); fleet = null; }
  serveLogged = false;
  resetLlmStatus();
  syncLanShare();
}

// Run the network gate while the switch is on and a model is answering, and
// only then. Called whenever either changes.
function syncLanShare() {
  const want = loadSettings().shareLlm === true && !!fleet && fleet.isReady();
  lanShare.sync(want, { name: servingModel.name, ctxSize: llmStatus.ctxSize });
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
  const base = (fleet && fleet.chatUrl()) || ('http://' + LLM.host + ':' + LLM.port);
  if (!fleet || !fleet.isReady()) { send('llm:chat:error', { message: 'the local LLM is not running' }); return; }

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
// with signed telemetry so the node shows online in the user's cluster. The
// requests themselves are nodeClient's, shared with the CLI. All best-effort —
// failures never touch mining.

let nodePinger = null;
const { loadNode, saveNode, getOrCreateNode } = nodeStore;
const nodeClient = createNodeClient({
  log: (level, line) => send('miner:log', { level, line }),
  linkHint: 'connect it to your account for private jobs',
});

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

// One signed ping with fresh telemetry. Silent on failure. Carries the node's
// current name so a Settings rename propagates to the server (see syncNodeName).
async function pingNode() {
  const node = loadNode();
  // Linked machines always ping. An unlinked one pings only while it is actually
  // serving (the fleet is up), so a machine that merely once had an identity
  // doesn't advertise itself as an online node while doing nothing.
  if (!node || !(node.connected || fleet)) return;
  const telemetry = await nodeClient.telemetry({
    model: servingModel, ready: llmStatus.ready, tokensPerSec: llmStatus.tokensPerSec,
    activeJobs: fleet ? fleet.activeJobs() : 0, name: node.name,
  });
  await nodeClient.ping(node, telemetry);
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
  nodePinger.unref();
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

  const res = await nodeClient.join(node, t, nm);
  if (res.unreachable) return { error: 'Could not reach LLMJob — check your connection.' };
  if (!res.ok) return { error: res.message || ('Link failed (HTTP ' + res.status + ').') };

  saveNode(Object.assign({}, node, {
    nodeId: res.nodeId, connected: true, name: nm, user: res.user, linkedAt: new Date().toISOString(),
  }));
  startNodePinger();
  syncWorker();
  sendNodeStatus();
  return { success: true, nodeId: node.nodeId, name: nm, user: res.user };
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

// Build a cluster job-worker for the ready LLM instance at `baseUrl` — one per
// serving GPU, each running jobs against its own llama-server. A rig that can run
// the model is useful to the network whether or not anyone linked it, so an
// unlinked box serves public work too (the server hands an unclaimed node
// non-private jobs only, so "unlinked" costs it access to private queues and
// nothing else) — matching the headless CLI's default.
function makeJobWorker(baseUrl) {
  // Reached only via the fleet after syncWorker() armed it, which already bailed
  // if no identity could be minted — so the node is always present here.
  const w = nodeClient.makeJobWorker(getOrCreateNode(), { baseUrl, servingModel: () => servingModel });
  if (!serveLogged) {
    serveLogged = true;
    send('miner:log', { level: 'info', line: 'serving cluster jobs for the LLMJob network' });
  }
  return w;
}

// Start cluster serving across every ready instance once a model is up — account
// or not. Idempotent, called from connect/disconnect and LLM ready. An unlinked
// machine mints an identity and self-registers so the server will answer its
// polls (nodeClient.enrol, retried with every ping until it lands); a linked one
// is already registered and just keeps serving. Pinging starts either way, so
// the node shows online and can be targeted for testing.
function syncWorker() {
  if (!fleet) return;
  const node = getOrCreateNode();
  if (!node) return; // no identity and none could be minted — serve nothing
  nodeClient.enrol(node);
  fleet.syncWorkers(true);
  startNodePinger();
}

function stopWorker() {
  if (fleet) fleet.syncWorkers(false);
  serveLogged = false;
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

// Which engines a START asks for.
function planFor(settings) {
  return resolvePlan(settings.mode || DEFAULT_MODE, {
    canMine: isValidAddress(settings.address) && minerSupported(process.platform),
    canLlm: true,
  });
}

// Apply the compute mode: run the miner and/or the LLM per the plan. When the
// plan ends up running nothing (LLM-only mode with the VRAM gate refusing, or
// no engine at all), tell the renderer — otherwise its optimistic "running"
// state shows STOP for a session in which nothing runs.
async function runPlan(queuedSettings) {
  // The session's settings as they are now, not as they were when this run was
  // queued: the network-share switch is saved live, and a run queued before it
  // was flipped would save the old value back.
  const settings = session ? session.settings : queuedSettings;
  const epoch = miningEpoch;
  const mode = settings.mode || DEFAULT_MODE;
  // macOS can serve the local LLM but has no mining engine to run, so the miner
  // is refused up front rather than left to fail on a Linux binary it would have
  // downloaded first (see shared/platform). The note explains the gap; without
  // it an 'auto' Mac start silently serves inference only, and a 'mining' one
  // runs nothing at all with no reason on screen.
  const note = minerUnsupportedNote(process.platform, mode);
  if (note) send('miner:log', { level: 'warn', line: note });
  const plan = planFor(settings);
  // A model planned without the miner, that the miner must not join (see
  // llmReplan). It goes before the miner starts, so the two never share the
  // card, and startLlm below plans again beside the miner. Paused or not: an
  // idle-only pause sizes the model for the miner too.
  const replan = plan.llm && plan.miner ? llmReplan() : null;
  if (replan) {
    send('miner:log', { level: 'info', line: 'restarting the local LLM: ' + servingModel.name
      + ' was sized for the card alone, and the miner is joining it' });
    await stopLlmForReplan(requiredFreeMb(replan, LLM.miningReserveMb), epoch);
    if (epoch !== miningEpoch) return;
  }
  // Idle-only, while someone uses the computer: the miner waits, the LLM runs.
  // The model is still sized with the mining reserve, so mining can start
  // beside it later without reloading it.
  const mineNow = plan.miner && !(session && session.paused);
  if (mineNow) {
    // Start mining FIRST, then — when co-running — wait until the miner reports a
    // non-zero hashrate before starting the LLM. Spawning the process (or even
    // connecting to the pool) isn't enough proof mining works: the LLM loads its
    // model and warms up in a few seconds, so without this wait it goes live
    // first, and its GPU-layer budgeter reads free VRAM before mining has claimed
    // its share. Waiting for real TH/s confirms the GPU is mining and its VRAM is
    // allocated, so the LLM then sizes its offload to what's actually left.
    try {
      await startMiningOnce(settings, plan.llm);
    } catch (e) {
      send('miner:log', { level: 'error', line: 'start failed: ' + e.message });
    }
    if (plan.llm && miner && miner.isRunning()) await waitForMinerUp();
  } else {
    // A miner from an earlier plan can still be running: LLM-only picked while
    // mining, or idle-only holding mining back. Stop it. For LLM-only, mining is
    // no longer asked for, and the model is about to be served from these cards
    // with no mining reserve, under the miner's memory clock lock if it holds
    // one (on an RTX 5090 that slows the model).
    // Not stopMining(): the session goes on, so the epoch stays (this run still
    // starts the model) and the renderer is not told it stopped. Its mining
    // figures are zeroed instead, as an LLM-only start shows them.
    if (miner) {
      haltMiner();
      send('miner:stats', statsView(snapshot(initStats(Date.now()), Date.now())));
    }
    persistSettings(settings); // startMining persists; do it here when the miner is off
  }
  // STOP arrived during miner setup or the hashrate wait: don't bring the LLM up
  // (or re-touch node state) for a session the user has already stopped.
  if (epoch !== miningEpoch) return;
  // The CPU miner runs for any START that asks for it, whatever the GPU does,
  // and waits out an idle-only pause like the GPU miner.
  if (settings.cpuMining && !(session && session.paused)) startCpuMining(settings);
  else stopCpuMining();
  syncNodeName(settings);
  if (plan.llm) {
    const started = await startLlm(plan.miner ? LLM.miningReserveMb : 0, epoch).catch(() => false);
    // A run cut short by STOP or a pause ends nothing: the START it belonged to
    // is already gone, or the next run carries it on.
    if (epoch !== miningEpoch) return;
    // Nothing runs for this START: the card had no room for an LLM-only start.
    // End the START itself, not just the window's STOP button: a session left
    // behind kept the idle watcher going, which could bring the model back while
    // the window showed START.
    if (!started) endIfNothingRuns(plan.miner);
  } else {
    stopLlm();
    endIfNothingRuns(plan.miner);
  }
}

// End the START when nothing runs for it: `others` is what the caller knows
// still runs, besides the CPU miner. A START that idle-only has paused still
// stands, though its miners are off: mining comes back on the next resume. A
// Monero-only START with idle-only on used to end itself on its first pause.
function endIfNothingRuns(others) {
  if (others || cpuRunning() || (session && session.paused)) return;
  endSession();
}

// Whether the local LLM runs, or a run is still bringing it up. A miner that
// gives up on its own leaves the START to these.
function llmRunsOrComing() {
  return !!planRun || !!(fleet && (fleet.hasSpawned() || fleet.isReady()));
}

// Single-flight wrapper around runPlan. `miner:start` is a plain IPC event with
// no re-entry guard, and the Chat tab's START LLM button fires it on every click
// while the LLM isn't ready — so a user clicking through a failing LLM setup
// launched a fresh download+extract each time, all racing on the same scratch
// files. Concurrent calls now collapse into the run already in flight.
//
// Settings that arrived mid-run are replayed once afterwards ONLY if they differ
// from the run in flight: a click that flips mining → both (START LLM while the
// miner runs) genuinely changes the plan and swallowing it would leave the LLM
// off, while a repeated click with the same settings has nothing left to do.
let planRun = null;
let planActive = null;
let planQueued = null;

// `force` replays even the same settings once the run in flight ends. An idle
// pause and resume need it: the pause cancelled that run (miningEpoch), and the
// same settings now mean a different run, with or without the miner.
function applyPlan(settings, force) {
  if (planRun) {
    if (force || JSON.stringify(settings) !== planActive) planQueued = settings;
    return planRun;
  }
  planActive = JSON.stringify(settings);
  planRun = runPlan(settings).finally(() => {
    planRun = null;
    planActive = null;
    const next = planQueued;
    planQueued = null;
    if (next) applyPlan(next);
    // An update that downloaded during the run may install now.
    installWhenHidden();
  });
  return planRun;
}

// The START the user pressed, kept until they press STOP: { settings, paused }.
// With "only mine when idle" on, the idle watcher pauses its mining while
// someone uses the computer and resumes it when they leave. The local LLM is
// not paused: it keeps serving chat, cluster jobs and the network share. The
// renderer keeps showing STOP throughout, because the user's START still stands.
let session = null;
let idleTimer = null;

function startSession(settings) {
  session = { settings, paused: false };
  syncIdleWatcher();
  // Someone just pressed START, so with idle-only on the computer is in use:
  // mining waits for it to go idle instead of starting now.
  if (getPrefs().mineWhenIdle && sessionMines() && appPrefs.idleAction(idleState(), false) === 'pause') {
    pauseSession();
    return;
  }
  applyPlan(settings);
}

function endSession() {
  session = null;
  // A pause or resume queued behind a slow start (a first model download can
  // take minutes) must not replay after STOP and start mining nobody asked for.
  planQueued = null;
  // And the cancelled run still in flight is no longer what's running, so a
  // START with the same settings must not be dropped as a duplicate of it.
  planActive = null;
  syncIdleWatcher();
  stopMining();
  stopLlm();
}

// Idle-only only ever holds back mining, GPU or CPU, so a START with neither
// in it (LLM mode with CPU mining off, say) has nothing to pause.
function sessionMines() {
  return planFor(session.settings).miner || !!session.settings.cpuMining;
}

function idleState() {
  return powerMonitor.getSystemIdleState(appPrefs.IDLE_AFTER_SEC);
}

// Stop mining for the user, keeping their START and the local LLM. Not
// stopMining(): that tells the renderer mining stopped. The epoch still moves,
// so a miner start in flight does not finish after the pause; that also cancels
// the LLM half of a run in flight, so the plan runs again without the miner,
// which brings the LLM up if it isn't already (and leaves it alone if it is).
function pauseSession() {
  session.paused = true;
  miningEpoch++;
  haltMiner();
  stopCpuMining();
  applyPlan(session.settings, true);
  send('miner:stats', statsView(snapshot(initStats(Date.now()), Date.now())));
  send('miner:idle', { paused: true, idleAfterSec: appPrefs.IDLE_AFTER_SEC });
  send('miner:log', {
    level: 'info',
    line: 'paused while the computer is in use; mining starts again after '
      + appPrefs.IDLE_AFTER_SEC / 60 + ' minutes without keyboard or mouse input',
  });
}

function resumeSession() {
  session.paused = false;
  send('miner:idle', { paused: false, idleAfterSec: appPrefs.IDLE_AFTER_SEC });
  send('miner:log', { level: 'info', line: 'the computer is idle; mining' });
  if (!planRun) {
    applyPlan(session.settings, true);
    return;
  }
  // A run is still in flight: usually a paused START bringing up the model,
  // which on a first run downloads several GB. Mining doesn't wait for it. That
  // model was planned with the mining reserve, so the miner starts beside it
  // now, and the run goes on to finish the model.
  const plan = planFor(session.settings);
  if (plan.miner) {
    startMiningOnce(session.settings, plan.llm)
      .catch((e) => send('miner:log', { level: 'error', line: 'start failed: ' + e.message }));
  }
  startCpuMining(session.settings);
}

function checkIdle() {
  if (!session.paused && !sessionMines()) return;
  const action = appPrefs.idleAction(idleState(), session.paused);
  if (action === 'pause') pauseSession();
  else if (action === 'resume') resumeSession();
}

// The watcher runs only while there is a START to pause and idle-only is on.
function syncIdleWatcher() {
  const wanted = !!session && getPrefs().mineWhenIdle;
  if (wanted && !idleTimer) idleTimer = setInterval(checkIdle, appPrefs.IDLE_POLL_MS);
  else if (!wanted && idleTimer) {
    clearInterval(idleTimer);
    idleTimer = null;
  }
}

// Settings changed one or more of the app's switches. Returns what now holds.
function setPrefs(change) {
  const next = Object.assign({}, change);
  // Starting with the computer is reported as it ended up: a dev run, or a
  // Linux build that is not an AppImage, cannot do it.
  if (typeof next.runAtStartup === 'boolean') next.runAtStartup = applyLoginItem(next.runAtStartup);
  const saved = savePrefs(next);
  // Idle-only switched off while paused: there is nothing left to wait for.
  if (!saved.mineWhenIdle && session && session.paused) resumeSession();
  syncIdleWatcher();
  return saved;
}

// The renderer gets a region that EXISTS. A saved AlphaPool id would otherwise
// reach a <select> with no matching option, which blanks it silently.
// A launch at login starts mining once, the first time the renderer asks.
let startMiningOnLaunch = hiddenLaunch;

ipcMain.handle('settings:get', () => {
  const s = withLiveRegion(Object.assign(
    // Both clients default to the shared DEFAULT_MODE ('auto': mine + serve the
    // LLM, balanced from free VRAM).
    // worker defaults to this machine's hostname, not the shared "rig01" constant:
    // two rigs on one payout address under the same name collide into a single
    // board identity (and if either is multi-GPU, the other's row is dropped
    // outright). Only fills a FRESH install — loadSettings() below wins, so an
    // existing worker name is never rewritten out from under someone's board row.
    // CPU mining is opt-in: off, no Monero address, and the default thread count.
    {
      region: DEFAULTS.region, worker: defaultWorker(), address: '', mdlAddress: '', mode: DEFAULT_MODE,
      cpuMining: false, xmrAddress: '', cpuThreads: 0,
    },
    loadSettings(),
  ));
  // The renderer already resumes a START after an update; a launch at login asks
  // for the same thing. It starts only what START could: with no payout
  // address, an LLM-only or Auto run, or the CPU miner.
  if (startMiningOnLaunch) s.resumeMining = true;
  startMiningOnLaunch = false;
  return s;
});
ipcMain.handle('prefs:get', () => getPrefs());
ipcMain.handle('prefs:set', (_e, change) => setPrefs(change || {}));
ipcMain.handle('llm:status', () => llmStatus);
// `platform` rides along on the config the renderer already fetches at startup,
// so the UI can stop offering what this OS can't do (the mining compute modes on
// macOS) without a second round trip or a new preload method.
ipcMain.handle('config:get', () => {
  // What CPU mining would do here, for its Settings section: the default,
  // whether it can run at all (one thread asked for by hand), and whether this
  // install has the worker that does it.
  const plan = cpuPlan(0);
  return {
    regions: REGIONS,
    defaults: DEFAULTS,
    platform: { minerSupported: minerSupported(process.platform) },
    cpu: {
      available: plan.available, auto: plan.auto, reason: plan.reason, canMine: cpuPlan(1).threads > 0,
      worker: cpuWorkerPath() != null,
    },
  };
});
ipcMain.handle('gpu:detect', () => detectGpu());
ipcMain.handle('region:detect', () => detectRegion());
ipcMain.handle('balance:get', (_e, address) => fetchBalance(address, liveEcon.PRL_USD));
ipcMain.handle('xmr:balance', (_e, address) => fetchXmrBalance(address));
// The renderer has no Keccak, so the address check runs here.
ipcMain.handle('xmr:check', (_e, address) => checkXmrAddress(address));
ipcMain.on('miner:start', (_e, settings) => startSession(settings || {}));
ipcMain.on('miner:stop', () => endSession());
ipcMain.on('open-external', (_e, url) => { openExternalSafe(url); });
// Re-fit the window to its content when the renderer's layout changes (tab
// switch, mining start/stop, etc.), so the frame never leaves a gap under the
// footer or clips a taller view.
ipcMain.on('app:fit', () => { fitWindowToContent(); });
// The window is listening for log lines: send what waited (see send).
ipcMain.on('log:ready', () => {
  logListening = true;
  const lines = logBacklog;
  logBacklog = [];
  for (const line of lines) send('miner:log', line);
});
ipcMain.on('clipboard:write', (_e, text) => { clipboard.writeText(String(text == null ? '' : text)); });
ipcMain.on('llm:chat', (_e, messages) => llmChat(messages));
// START LLM on the Chat or API tab: the cards that kept losing their GPU get
// another try. The renderer sends this just before that START.
ipcMain.on('llm:retry', () => {
  if (!Object.keys(loadGpuLost().cards).length) return;
  saveGpuLost({ version: app.getVersion(), cards: {} });
  send('miner:log', { level: 'info', line: 'trying the cards that kept losing their GPU again' });
});
// The network switch applies at once, beside a running model, rather than at the
// next START. Saved straight to settings.json; the renderer also carries it in
// the settings START sends, so a START never turns it back off.
ipcMain.on('llm:share', (_e, on) => {
  persistSettings(Object.assign({}, loadSettings(), { shareLlm: on === true }));
  // An idle pause or resume replays the START's settings and saves them; without
  // this they would put the switch back where it was at START.
  if (session) session.settings = Object.assign({}, session.settings, { shareLlm: on === true });
  syncLanShare();
});
ipcMain.handle('node:status', () => nodeStatus());
ipcMain.handle('node:connect', (_e, opts) => connectNode(opts || {}));
ipcMain.handle('node:disconnect', () => disconnectNode());
ipcMain.on('node:dashboard', () => openExternalSafe(NODE.dashboardUrl));
ipcMain.handle('app:version', () => app.getVersion());
ipcMain.on('app:update:check', () => checkForUpdate());
ipcMain.on('app:update:install', () => installUpdate(false));

app.on('second-instance', () => showWindow());

app.whenReady().then(async () => {
  // The second copy is on its way out (see requestSingleInstanceLock above).
  if (!primaryInstance) return;
  // Restarted by an update that installed while the window was in the tray:
  // start in the tray again. Only straight after that restart, so an install
  // that never restarted the app can't make the next ordinary launch start out
  // of sight.
  const saved = loadSettings();
  if (saved.resumeHidden) {
    if (Date.now() - Number(saved.resumeHidden) < UPDATE_RESTART_MS) hiddenLaunch = true;
    delete saved.resumeHidden;
    persistSettings(saved);
  }
  createTray();
  // A launch at login waits to know there's a tray to start in. A window on
  // screen doesn't wait: the answer is in long before anyone closes it.
  if (hiddenLaunch) await confirmTray(TRAY_WAIT_MS).catch(() => {});
  else confirmTray(0).catch(() => {});
  createWindow();
  // Logging off or shutting down on Linux: let the window close.
  powerMonitor.on('shutdown', () => { quitting = true; });
  // Point the login item at this install, if an update moved or renamed it.
  refreshLoginItem();
  // Not on macOS: Squirrel.Mac checks the downloaded bundle's signature against
  // the running app's, and this build carries only an ad-hoc one, so wiring the
  // updater there buys a periodic download that always ends in an error bar the
  // user can do nothing about. checkForUpdate() sends them to Releases instead.
  if (app.isPackaged && autoUpdateSupported(process.platform)) setupUpdater();
  // Pull live network economics so earnings estimates and the balance's USD
  // figure reflect the real price + network, not the stale fallback constants.
  refreshEconomics();
  const econTimer = setInterval(refreshEconomics, 10 * 60 * 1000);
  econTimer.unref();
  // The identity used to live in Electron's userData dir; move it into the
  // store shared with the CLI so one machine keeps one nodeId across shells.
  nodeStore.migrateFrom(path.join(app.getPath('userData'), 'node.json'));
  // Resume pinging if this machine is already linked to an account.
  const node = loadNode();
  if (node && node.connected) startNodePinger();
  // macOS: clicking the Dock icon brings the window back, from the tray too.
  // showWindow makes a new one if it was closed.
  app.on('activate', (_e, hasVisibleWindows) => {
    if (!hasVisibleWindows) showWindow();
  });
});

// Kill the children we spawned. Idempotent — both quit paths below call it, and
// on the common path both of them fire.
function shutdownChildren() {
  // endSession stops the miner and the LLM (never orphan llama-server: it would
  // hold VRAM + port 8080), and the idle watcher with them.
  endSession();
}

app.on('window-all-closed', () => {
  shutdownChildren();
  if (process.platform !== 'darwin') app.quit();
});

// The other way out, and until now the LEAKING one. Electron does not emit
// 'window-all-closed' when the quit was started programmatically — which is what
// the default menu's Quit role and Ctrl/Cmd+Q do. So the most ordinary way to
// close the app skipped the only cleanup hook and left llama-server and the
// alpha-miner running: the user's GPU stayed pinned by processes they could no
// longer see, and the next launch found port 8080 already held (the same
// symptom the update path documents at 'app:update:install').
app.on('before-quit', () => {
  // Every real quit comes through here, so the window may close now.
  quitting = true;
  shutdownChildren();
});
