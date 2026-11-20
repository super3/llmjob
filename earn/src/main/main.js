'use strict';

// Electron main process. Thin shell: owns the window, persists settings, and
// bridges the renderer to our own Pearl miner. Stats shown to
// the user come only from the engine's own output — no simulated data. Most
// testable logic lives in ../shared and the other modules in this folder.

// Before anything uses libuv's thread pool: a Quantus scan holds a pool thread
// a card (shared/threadPool.js).
require('../shared/threadPool').widenThreadPool(process.env);

const {
  app, powerMonitor, ipcMain, shell, clipboard, crashReporter, dialog,
} = require('electron');
const path = require('path');
const fs = require('fs');
const os = require('os');

const { ctx, share } = require('./appContext');
const { UPDATE_RESTART_MS, installUpdate, setupUpdater, checkForUpdate } = require('./updater');
const { TRAY_WAIT_MS, createWindow, showWindow, createTray, confirmTray, fitWindowToContent } = require('./appWindow');
const { syncLanShare, llmChat } = require('./appLlm');
const { nodeStatus, connectNode, disconnectNode, startNodePinger } = require('./appNode');
const { startSession, endSession, setPrefs, endIfNothingRuns, llmRunsOrComing } = require('./appSession');
const { createCpuMining } = require('./cpuMining');
const { coreFactory } = require('./pearlCore');
const { quantusCoreFactory } = require('./quantusCore');
const { withShapePick } = require('../shared/shapePick');
const { getJson } = require('./io');
const { detectRegion, postMinerReport } = require('./probe');
const probe = require('./probe');
const nodeStore = require('./nodeStore');
const { loadNode } = nodeStore;
const settingsStore = require('../shared/settingsStore');
const { nextRelaunches } = require('../shared/faultRelaunch');
const { initStats, applyEvent, snapshot, offPool } = require('../shared/miningStats');
const { minerDeps } = require('../shared/minerDeps');
const {
  DEFAULTS, POOLS, NETWORK, ECON, ECON_API, LLM, NODE,
  resolveEndpoint, quantusEndpointFor, liveRegion, poolId, feeShare,
} = require('../shared/config');
const { defaultWorker } = require('../shared/worker');
const { resolveEconomics } = require('../shared/economics');
const gpuLoss = require('../shared/gpuLoss');
const { makeLogFile } = require('./logFile');
const { createPromptPause } = require('../shared/promptPause');
const { DEFAULT_MODE } = require('../shared/llmMode');
const { minerSupported, autoUpdateSupported } = require('../shared/platform');
const { fetchBalance, fetchXmrBalance } = require('./balances');
const { checkXmrAddress } = require('../shared/xmr/address');
const { checkQuantusAddress, isValidQuantusAddress } = require('../shared/quantus/address');
const { isValidAddress } = require('../shared/address');
const {
  fp8On, minesFp8, forkDrops, planCoins, planIsPlain, planCoin, quantusOnly, planLines, noticeCards, noticeLine,
} = require('../shared/forkPlan');
const { hostStats, reportToBoard } = require('../shared/minerReport');
const {
  alignCudaDeviceOrder, clearCudaVisibleDevices, describeClearedCuda, readGpuIndexEnv, describeSkippedGpus,
  describeAmdPlan, describeIntelPlan, intelCount, planMiningCards,
} = require('../shared/gpu');
const { readMemClockEnv, GUI_MEM_CLOCK_ENV } = require('../shared/memClock');
const { readCoreOffsetEnv, GUI_CORE_OFFSET_ENV } = require('../shared/coreOffset');
const { planClocks } = require('../shared/clockPlan');
const { readRotationFoldEnv, CORE_ROTATION_FOLD_ENV } = require('../shared/rotationFold');
const { describeAmdDrivers } = require('../shared/amdDriver');
const earnings = require('../shared/earnings');
const format = require('../shared/format');
const appPrefs = require('../shared/appPrefs');
const autostart = require('../shared/autostart');

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

let miner = null;
// A coin per card (shared/forkPlan). On a rig that mines both, `miner` is the
// Pearl engine and `side` the Quantus engine for the cards on Quantus. `mined`
// is what the START in progress mines, as planCoin gives it: the coin, and on
// a rig mining both, the Quantus cards. `forkActive` is set, for the life of
// the app, once a pool sends a job of Pearl's FP8 scheme.
let side = null;
let mined = { coin: 'pearl', quantusCards: null };
let forkActive = false;
// The last mining start found no card with anything to mine.
let noCardMines = false;
let stats = null;
let ticker = null;
let reporter = null;
// Bumped on every stop. A start is async (the engine can download for minutes),
// during which the user may press STOP; the in-flight start captures this epoch
// and, if it has since changed, aborts instead of spawning a headless miner or
// starting the LLM after the user already stopped.
let miningEpoch = 0;
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

const settingsLog = (m) => console.error(m);
// One JSON file of the app's in userData, read and written whole
// (shared/settingsStore). A file that can't be read reads as {}; a failed write
// is logged.
function userJson(name) {
  const file = () => path.join(app.getPath('userData'), name);
  return {
    read: () => settingsStore.readSettings(file(), { fs, log: settingsLog }),
    write: (data) => settingsStore.writeSettings(file(), data, { fs, log: settingsLog }),
  };
}
const settingsFile = userJson('settings.json');
function loadSettings() {
  return settingsFile.read();
}
// A pool that exists and a server it has, with a saved AlphaPool region
// translated onto a live one. Not a write: the renderer hands the migrated id
// straight back when the user next saves, so the file heals itself without a
// startup rewrite that would touch a settings file we might have failed to
// read properly.
function withLiveRegion(s) {
  const pool = poolId(s.pool);
  return Object.assign({}, s, { pool, region: liveRegion(pool, s.region) });
}

function persistSettings(s) {
  return settingsFile.write(s);
}

// Cards that kept losing their GPU, and the models they lost it under (see
// shared/gpuLoss). In their own file, for the reason appPrefs gives: every
// START rewrites settings.json from the renderer's fields.
const gpuLostFile = userJson('gpu-lost.json');
// When the app last started itself again for a card whose CUDA state was lost (see relaunchAfterFault). In
// its own file for the same reason: START rewrites settings.json.
const faultRelaunchFile = userJson('fault-relaunches.json');
function loadGpuLost() {
  return gpuLoss.forVersion(gpuLostFile.read(), app.getVersion());
}
function saveGpuLost(record) {
  gpuLostFile.write(record);
}

// The app's behaviour switches (tray, startup, idle-only), from Settings. In
// their own file, see shared/appPrefs. Read once, then kept here.
let prefs = null;
const prefsFile = userJson('preferences.json');
function getPrefs() {
  if (!prefs) prefs = appPrefs.normalizePrefs(prefsFile.read());
  return prefs;
}
function savePrefs(change) {
  prefs = appPrefs.normalizePrefs(Object.assign({}, getPrefs(), change));
  prefsFile.write(prefs);
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
      settingsStore.writeFileAtomic(file, body, { fs });
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
    settingsStore.writeFileAtomic(file, next, { fs });
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

// Every log line also goes to disk (logFile.js), window or not, so a crash
// that takes the app down leaves its last lines behind.
const appLog = makeLogFile({ fs, dir: path.join(app.getPath('userData'), 'logs') });

// A crash in native code (the miner's core, a GPU driver) also leaves a minidump
// in app.getPath('crashDumps'). Kept on this PC: nothing is uploaded.
crashReporter.start({ uploadToServer: false });

function send(channel, payload) {
  if (channel === 'miner:log' && payload) appLog.write(payload.level, payload.line);
  const win = ctx.win;
  if (!win || win.isDestroyed()) return;
  if (channel === 'miner:log' && !logListening) {
    logBacklog.push(payload);
    if (logBacklog.length > LOG_BACKLOG_MAX) logBacklog.shift();
    return;
  }
  win.webContents.send(channel, payload);
}

const replan = (...args) => ctx.replan(...args);

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

// ── CPU mining (Monero) ─────────────────────────────────────────────────────
// See main/cpuMining. Made once, for the life of the app.
const cpu = createCpuMining({
  send,
  xmrEngine: (opts) => minerParts().xmrEngine(opts),
  onGaveUp: () => endIfNothingRuns((miner && miner.isRunning()) || sideRuns() || llmRunsOrComing()),
  inSession: () => !!ctx.session,
});

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

// Map a stats snapshot to the display fields the renderer expects. A START that
// mines Quantus (Settings → GPU Coin) reads in MH/s, beside the unit the
// renderer shows for that coin, and has no $/day estimate.
function statsView(snap) {
  const quantus = !!ctx.session && mined.coin === 'quantus';
  // What the reading is of, where it is not the coin picked in Settings: the
  // fork moved the cards to Quantus, or the rig mines both.
  const both = snap.quantusTotal != null;
  const coin = both ? 'both' : mined.coin;
  return {
    total: quantus ? format.formatMhs(snap.total) : format.formatHashrate(snap.total),
    ...(ctx.session && coin !== ctx.session.settings.coin ? { coin } : null),
    // On a rig mining both, the Quantus cards' rate in MH/s; `total` is Pearl's.
    ...(both ? { quantusTotal: format.formatMhs(snap.quantusTotal) } : null),
    points: snap.points,
    acceptedLabel: format.formatInt(snap.accepted),
    // Every card that is mining, not just the first. On a multi-card rig showing
    // one name is the same lie issue #226 was about, one level up.
    gpu: format.formatDeviceLabel(snap.gpus.map((g) => g.gpu)) || snap.gpu,
    temp: snap.temp,
    uptime: format.formatUptime(snap.uptimeSec),
    // After the fee of the pool this START mines Pearl on.
    estDay: quantus ? '—' : earnings.estDailyUsdLabel(snap.total,
      Object.assign({}, liveEcon, { FEE: feeShare(ctx.session && ctx.session.settings.pool) })),
    // Mining is stopped on a card while the local LLM answers a prompt, so the
    // Mine tab can say why Pearl reads 0 for those seconds.
    llmPaused: promptPause.pausedNow(),
  };
}

// What the miners are built from (shared/minerDeps): one for the life of the
// app, so both coins share one pool lookup, whose last good addresses are kept
// beside the node identity, shared with the CLI. The launch shapes and memory
// clock trials are kept in userData, where the app has always kept them (the
// CLI keeps its own beside node.json).
let minerPartsMemo = null;
function minerParts() {
  if (!minerPartsMemo) {
    minerPartsMemo = minerDeps({
      version: app.getVersion(),
      nodeDir: () => path.dirname(nodeStore.nodePath()),
      storeDir: () => app.getPath('userData'),
      fs,
      settingsLog,
      lookupLog: (line) => send('miner:log', { level: 'warn', line }),
    });
  }
  return minerPartsMemo;
}

// What the OS says about the AMD cards of a rig whose nvidia-smi lists none:
// probe.detectGpuInfo, whose `amd` shared/gpu.planAmdGpus counts. Null without
// asking when nvidia-smi lists cards, so an NVIDIA rig spawns nothing more, and
// on macOS, which has no AMD core and whose probe is slow.
function readAmdCards(nvidiaListsCards) {
  if (nvidiaListsCards || process.platform === 'darwin') return Promise.resolve(null);
  return probe.detectGpuInfo();
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
  // A miner that started mining-only may hold a memory clock lock, or be part
  // way through the trial for one (below). When the LLM joins it, the lock and
  // the trial go now, before runPlan starts
  // the model: a served model must never sit on a locked card. The other way
  // round -- the mode flipped back to mining-only mid-run -- takes no lock; the
  // card mines at the driver's clock until the next start, which is slower,
  // never wrong. The same call ends a core clock offset trial and puts every
  // card back at offset 0 (shared/coreOffset).
  if (miner && miner.isRunning()) {
    if (llmCoRuns) miner.releaseMemClocks();
    persistSettings(settings);
    return;
  }
  persistSettings(settings);
  // A Quantus engine left from a START whose Pearl engine has stopped.
  stopSide();
  noCardMines = false;
  // Until the cards are known: the coin picked, or Quantus once the fork has
  // ended Pearl for this build.
  const fp8 = fp8On(process.env);
  const quantusOk = isValidQuantusAddress(settings.quantusAddress);
  mined = { coin: quantusOnly({ coin: settings.coin, quantusOk, forkActive, fp8 }) ? 'quantus' : 'pearl', quantusCards: null };

  // Real stats only: the accumulator starts at zero and is filled in from the
  // engine's parsed output (see the miner 'event' handler below). The ticker
  // just re-emits the current snapshot each second so uptime advances.
  stats = initStats(Date.now());
  send('miner:stats', statsView(snapshot(stats, Date.now())));
  if (ticker) clearInterval(ticker);
  ticker = setInterval(() => send('miner:stats', statsView(snapshot(stats, Date.now()))), 1000);

  // PEARL_GPU_INDEX can narrow the cards (read here so the first board report
  // already knows). A value the app can't use, and a listed card nvidia-smi
  // doesn't show, are logged below rather than dropped quietly.
  const gpuEnv = readGpuIndexEnv(process.env);

  // Publish live status to the network page's board while mining, including live
  // GPU VRAM (used/total) so the board shows headroom for co-running LLMs.
  const report = () => reportToBoard({
    // Each card's row is its coin's (shared/minerReport).
    settings: Object.assign({}, settings, mined),
    version: app.getVersion(),
    // While the pool connection is down, the board hears no hashrate: the pool
    // is getting none, and the board should agree with it. With a Quantus
    // engine beside the Pearl one, each coin's cards follow their own.
    snapshot: () => {
      let live = snapshot(stats, Date.now());
      if (miner && !miner.poolConnected()) live = offPool(live, side ? 'pearl' : undefined);
      if (side && !side.poolConnected()) live = offPool(live, 'quantus');
      return live;
    },
    boardGpus: () => probe.detectBoardGpus(),
    // A card PEARL_GPU_INDEX left out does not mine, so it gets no row, as in the CLI.
    indices: gpuEnv.indices,
    // Tag the cards serving the local LLM so the board shows which model each GPU
    // runs; null when the fleet isn't up (mining only) → blank on the board.
    // `nodeId` rides along whenever the fleet is up, because that is now exactly
    // when this machine polls the cluster — linked or not (syncWorker arms the
    // worker on the fleet, not on the account). Running the model and serving the
    // cluster are the same thing here; a machine with no identity yet reports null.
    serving: () => {
      const fleet = ctx.fleet;
      if (!fleet) return null;
      const identity = loadNode();
      return { model: ctx.servingModel.name, indices: fleet.servingIndices(), nodeId: identity ? identity.nodeId : null };
    },
    // The machine's RAM, and the CPU miner's hashrate (null when it isn't running).
    host: () => hostStats(os, cpu.engine()),
    post: postMinerReport,
  });
  report();
  if (reporter) clearInterval(reporter);
  reporter = setInterval(report, NETWORK.reportIntervalMs);

  // Quantus mines on the same pool and server as Pearl would (Settings →
  // Connection), and its engine logs where it connects.
  const quantus = mined.coin === 'quantus';
  const quantusEndpoint = quantusEndpointFor(settings.region, settings.pool);
  const endpoint = resolveEndpoint(settings);
  if (!quantus) send('miner:log', { level: 'info', line: 'connecting to ' + endpoint + ' · worker ' + (settings.worker || DEFAULTS.worker) });

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
  if (gpuEnv.warning) send('miner:log', { level: 'warn', line: gpuEnv.warning });
  // The same nvidia-smi round also reads each card's compute capability and the
  // driver version, which decide whether this rig loads the CUDA 12.8 core or
  // the CUDA 13 one (shared/coreVariant). The factory logs its choice.
  const [nvidiaGpus, cudaCards] = await Promise.all([
    probe.detectMinerGpus(process.env, gpuEnv.indices), probe.detectCudaCards(),
  ]);
  // nvidia-smi lists no AMD card, so on an AMD rig the list above is empty and
  // one core would mine one card. The OS counts them instead (shared/gpu
  // planAmdGpus). Asked only when nvidia-smi lists no card. On Windows the same
  // answer has the AMD driver's version, logged once a start (shared/amdDriver).
  const amdCards = await readAmdCards(cudaCards.length > 0);
  if (epoch !== miningEpoch) return;
  for (const l of describeAmdDrivers(amdCards)) send('miner:log', l);
  // The same answer lists the Intel Arc cards of a rig with no AMD card, which
  // are counted the same way and tell the core loader that this is an Intel rig
  // (shared/coreVariant).
  const { gpus, amdGpus, intelGpus } = planMiningCards({
    env: process.env, det: amdCards, chosen: gpuEnv.indices, nvidiaGpus,
  });
  if (amdGpus.length) send('miner:log', { level: 'info', line: describeAmdPlan(amdGpus, 'set PEARL_GPU_INDEX') });
  if (intelGpus.length) send('miner:log', { level: 'info', line: describeIntelPlan(intelGpus, 'set PEARL_GPU_INDEX') });
  for (const line of describeSkippedGpus(gpuEnv.indices, gpus, 'PEARL_GPU_INDEX')) {
    send('miner:log', { level: 'warn', line });
  }
  // Which coin each card mines (shared/forkPlan). With no card picked for
  // another coin and no FP8 job seen, this is the coin in Settings for every
  // card, and the start goes on exactly as it always has.
  const vendor = cardMaker(nvidiaGpus, amdGpus, intelGpus);
  const coinPlan = planCoins({
    indices: gpus.map((g) => g.index), cards: cudaCards, coin: settings.coin, cardCoins: settings.cardCoins,
    pearlOk: isValidAddress(settings.address), quantusOk, forkActive, fp8,
  });
  const plain = planIsPlain(coinPlan, settings.coin);
  mined = planCoin(coinPlan);
  for (const l of planLines(coinPlan)) send('miner:log', l);
  // Before the fork: the cards it will end Pearl for are told.
  const notice = noticeLine(noticeCards({ plan: coinPlan, cards: cudaCards, vendor, forkActive, env: process.env }), quantusOk);
  if (notice) send('miner:log', { level: 'warn', line: notice });
  // The cards for the Quantus engine: PEARL_GPU_INDEX's list (null: every
  // card) unless the plan splits the rig.
  const quantusCards = plain || coinPlan.whole ? gpuEnv.indices : coinPlan.quantus;
  // Quantus has none of what follows: no clock plan, no rotation fold, no
  // launch shapes, and no local LLM beside it to pause for.
  if (!coinPlan.pearl.length) {
    if (coinPlan.quantus.length) {
      startQuantus(settings, quantusEndpoint, quantusCards, vendor, cudaCards);
      return;
    }
    // No card has anything to mine: the lines above say why. The START goes on
    // only for what else runs.
    noCardMines = true;
    endIfNothingRuns(llmRunsOrComing());
    return;
  }
  // The Pearl engine gets the Pearl cards only.
  const onPearl = (list) => (plain ? list : list.filter((g) => coinPlan.pearl.includes(g.index)));
  const pearlGpus = onPearl(gpus);
  // The run carries on through the fork only when every Pearl card has an FP8
  // core (PEARL_FP8=1, RTX 50 and RTX PRO Blackwell).
  const cudaByIndex = new Map(cudaCards.map((c) => [c.index, c]));
  const fp8Run = fp8 && coinPlan.pearl.every((index) => minesFp8(cudaByIndex.get(index), true));

  // Which cards lock their memory clock while mining: the RTX 5090 default
  // (shared/memClock), which tries the lock on each 5090 and keeps it only
  // where that card mines faster, or LLMJOB_MINE_MEM_CLOCK from the app's
  // environment, the GUI's one switch for it. The renderer has no setting yet,
  // so the variable is the way off the default and the way past its trial: 0
  // leaves every card at the driver's clock, a number locks every mining card
  // at it with no trial, and auto tries a lock on the other cards too, the
  // same as --mine-mem-clock.
  // Read on every start, like the CLI reads its flag; a bad value is logged and
  // ignored, so a typo keeps the default rather than silently dropping it. Not
  // while the LLM co-runs: llama-server is memory-bandwidth-bound, and here
  // 'auto' always co-runs (resolvePlan gives { miner, llm } and runPlan starts
  // both; the GUI has no demand mode), so plan.llm is the whole of the GUI's
  // co-run condition. The plan rides on the start call, not on `settings`,
  // so persistSettings never writes it to disk.
  // nvidia-smi's cards only: the lock is nvidia-smi's. Given the AMD plan, it
  // would try to lock each AMD card, where an empty list says "NVIDIA-only".
  // The rotation fold (shared/rotationFold) is read first: on it an RTX 40 card
  // also tries a memory clock lock by default.
  //
  // Which cards try a core clock offset (shared/coreOffset): none unless
  // LLMJOB_MINE_CORE_OFFSET=auto is in the app's environment, the GUI's one
  // switch for it, read on every start like the memory clock's. The same rules:
  // nvidia-smi's cards only, and not while the LLM co-runs.
  const foldRequest = readRotationFoldEnv(process.env);
  const memRequest = readMemClockEnv(process.env);
  const offsetRequest = readCoreOffsetEnv(process.env);
  const clocks = planClocks({
    memMhz: memRequest.mhz, memName: GUI_MEM_CLOCK_ENV, memWarning: memRequest.warning,
    offset: offsetRequest.offset, offsetName: GUI_CORE_OFFSET_ENV, offsetWarning: offsetRequest.warning,
    cards: cudaCards, gpus: onPearl(nvidiaGpus), llmCoRuns: !!llmCoRuns, rotationFold: foldRequest.on,
  });
  for (const l of clocks.lines) send('miner:log', l);

  // The rotation fold (shared/rotationFold): on unless LLMJOB_ROTATION_FOLD=0
  // is in the app's environment, read on every start like the two above. It is
  // handed to the CUDA core as its own variable, 1 or 0. Which cards can run it,
  // and what a card that cannot says, is the core's and the miner's business.
  if (foldRequest.warning) send('miner:log', { level: 'warn', line: foldRequest.warning });
  process.env[CORE_ROTATION_FOLD_ENV] = foldRequest.on ? '1' : '0';

  const coreLog = (level, line) => send('miner:log', { level, line });
  const parts = minerParts();
  miner = parts.pearlEngine({
    // Each RTX 30 card's launch shape, timed once on this PC and cached in userData
    // (shared/shapePick).
    createCore: withShapePick(coreFactory({
      resourcesPath: process.resourcesPath,
      gpus: pearlGpus,
      cards: cudaCards,
      intel: intelCount(amdCards),
      amd: amdCards && amdCards.amd,
      log: coreLog,
    }), {
      store: parts.shapeStore(),
      log: coreLog,
      gpus: pearlGpus,
    }),
    relaunch: relaunchAfterFault,
  });
  minerFp8 = fp8Run;
  wireMinerEvents(miner, endpoint);
  try {
    miner.start(Object.assign({}, settings, { endpoint, gpus: pearlGpus, gpu: settings.gpu || null },
      fp8Run ? { fp8: true } : null, clocks.start));
  } catch (e) {
    reportLaunchFailure(e);
  }
  // The cards on Quantus, on an engine of their own beside the Pearl one.
  if (coinPlan.quantus.length) {
    side = quantusEngineFor(quantusCards, vendor, cudaCards);
    wireSideEvents(side);
    side.start({ address: settings.quantusAddress, worker: settings.worker, endpoint: quantusEndpoint, cards: quantusCards });
  }
  // A prompt that is running now keeps its card from mining.
  promptPause.reapply();
}

// Whether the Pearl engine's cores are FP8 ones, so it carries on through the
// fork (see onFork).
let minerFp8 = false;

// The Quantus engine beside the Pearl one, when it mines.
function sideRuns() {
  return !!side && side.isRunning();
}

function stopSide() {
  const engine = side;
  side = null;
  if (engine) engine.stop();
}

// The Quantus engine beside the Pearl one: its log and its cards' numbers,
// which carry their coin (shared/miningStats). It ends no START by itself:
// what it says when it gives up is in the log, and the Pearl cards mine on.
function wireSideEvents(engine) {
  engine.on('log', (l) => send('miner:log', l));
  engine.on('event', (e) => applyEvent(stats, e, Date.now()));
  engine.on('error', (err) => send('miner:log', { level: 'error', line: 'quantus: ' + err.message }));
}

// A pool sent a job of Pearl's FP8 scheme (PearlMiner 'fork'): the fork is
// live. A Pearl engine whose cores are FP8 ones carries on. Any other stops
// itself, and the START is planned again, which now puts the cards that cannot
// mine FP8 on Quantus, or stops them with a reason when there is no Quantus
// address (startMining). The local LLM follows the new plan as on any START.
function onFork(engine) {
  if (engine !== miner) return;
  forkActive = true;
  if (minerFp8) return;
  haltMiner();
  replan();
}

// Whose cards this rig mines on, so the Quantus core loader tries that maker's
// core first: CUDA on NVIDIA, HIP on AMD, SYCL on Intel Arc. nvidia-smi's cards
// first, then the ones the OS counts. Null when nothing says.
function cardMaker(nvidiaGpus, amdGpus, intelGpus) {
  if (nvidiaGpus.length) return 'nvidia';
  if (amdGpus.length) return 'amd';
  return intelGpus.length ? 'intel' : null;
}

// Quantus on the GPUs (Settings → GPU Coin): the engine the CLI's --coin
// quantus runs, paid to the Quantus address. `cards` is PEARL_GPU_INDEX's
// list, or null for every card the core lists. A start that fails (no core
// here, no card would open) says why through the engine's 'error', as a Pearl
// start does, and STOP stays up for the user to press. `cudaCards` is
// nvidia-smi's list, from which the loader picks the CUDA 13 core on an RTX 50
// rig with driver 580+, as the Pearl loader does.
function startQuantus(settings, endpoint, cards, vendor, cudaCards) {
  miner = quantusEngineFor(cards, vendor, cudaCards);
  wireMinerEvents(miner, endpoint);
  miner.start({ address: settings.quantusAddress, worker: settings.worker, endpoint, cards });
}

function quantusEngineFor(cards, vendor, cudaCards) {
  const coreLog = (level, line) => send('miner:log', { level, line });
  return minerParts().quantusEngine({
    factory: quantusCoreFactory({
      resourcesPath: process.resourcesPath, vendor, log: coreLog,
      cards: cudaCards, gpus: (cards || []).map((index) => ({ index })),
    }),
  });
}

// A card's CUDA state is lost for the life of this process (PearlMiner._stuckFault): only a new process mines
// on it again. So start the app again and pick the START back up, as an update does, in the tray if the
// window was out of sight. At most FAULT_RELAUNCH_LIMIT times an hour (shared/faultRelaunch); past that it
// says no and the card is left to its restarts. The quit runs on the next tick, after the miner has stopped.
function relaunchAfterFault() {
  const now = Date.now();
  const times = nextRelaunches(faultRelaunchFile.read().times, now);
  if (!times) return false;
  faultRelaunchFile.write({ times });
  const restart = { resumeMining: true };
  // Mining means the app is up, and the window is made at ready and never dropped, so there is one.
  const win = ctx.win;
  if (win.isDestroyed() || !win.isVisible()) restart.resumeHidden = now;
  persistSettings(Object.assign({}, loadSettings(), restart));
  ctx.quitting = true;
  setImmediate(() => {
    app.relaunch();
    app.quit();
  });
  return true;
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
    if (e.type === 'fork') {
      onFork(engine);
      return;
    }
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
    if (code === 1 && engine === miner) endIfNothingRuns(sideRuns() || llmRunsOrComing());
  });
}

function stopMining() {
  // Cancel any start still in flight (see miningEpoch) so it doesn't spawn or
  // start the LLM after this stop.
  miningEpoch++;
  haltMiner();
  cpu.stop();
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
  stopSide();
  if (miner) {
    miner.stop();
    miner = null;
  }
}

// The cards a START would mine on, for the per-card coin list in Settings:
// [{ index, name, drops }], where `drops` says Pearl's FP8 fork ends Pearl for
// the card, which Settings then tells the user. LLMJOB_FORK_NOTICE=0 turns
// that off, as it does the log's notice. Empty when the cards are not known.
async function listCoinCards() {
  const gpuEnv = readGpuIndexEnv(process.env);
  const [nvidiaGpus, cudaCards] = await Promise.all([
    probe.detectMinerGpus(process.env, gpuEnv.indices), probe.detectCudaCards(),
  ]);
  const det = await readAmdCards(cudaCards.length > 0);
  const { gpus, amdGpus, intelGpus } = planMiningCards({ env: process.env, det, chosen: gpuEnv.indices, nvidiaGpus });
  const vendor = cardMaker(nvidiaGpus, amdGpus, intelGpus);
  const byIndex = new Map(cudaCards.map((c) => [c.index, c]));
  const tell = process.env.LLMJOB_FORK_NOTICE !== '0';
  return gpus.map((g) => ({ index: g.index, name: g.name || null, drops: tell && forkDrops(byIndex.get(g.index), vendor) }));
}

// Detect the machine's GPU for the settings/device label: the shared probe's
// answer (nvidia-smi, then the OS's own card list), unwrapped to the plain name
// string the renderer's IPC contract wants, or null. Never rejects.
async function detectGpu() {
  const info = await probe.detectGpuInfo();
  return info ? info.name : null;
}

// What the parts split out of main.js (see appContext) read and call here.
share({
  miner: { get: () => miner },
  forkActive: { get: () => forkActive },
  noCardMines: { get: () => noCardMines },
  miningEpoch: { get: () => miningEpoch, set: (v) => { miningEpoch = v; } },
  promptPause: { get: () => promptPause },
  cpu: { get: () => cpu },
  appLog: { get: () => appLog },
  logListening: { get: () => logListening, set: (v) => { logListening = v; } },
}, { send, loadSettings, persistSettings, refreshLoginItem, openExternalSafe, getPrefs, savePrefs, applyLoginItem,
  loadGpuLost, saveGpuLost, readAmdCards, statsView, startMiningOnce, haltMiner, stopMining });

// A launch at login starts mining once, the first time the renderer asks.
let startMiningOnLaunch = ctx.hiddenLaunch;

// The renderer gets a pool and region that EXIST. A saved AlphaPool id would
// otherwise reach a <select> with no matching option, which blanks it silently.
ipcMain.handle('settings:get', () => {
  const saved = loadSettings();
  const s = withLiveRegion(Object.assign(
    // Both clients default to the shared DEFAULT_MODE ('auto': mine + serve the
    // LLM, balanced from free VRAM).
    // worker defaults to this machine's hostname, not the shared "rig01" constant:
    // two rigs on one payout address under the same name collide into a single
    // board identity (and if either is multi-GPU, the other's row is dropped
    // outright). Only fills a FRESH install — loadSettings() below wins, so an
    // existing worker name is never rewritten out from under someone's board row.
    // CPU mining is opt-in: off, no Monero address, and the default thread count.
    // The GPUs mine Pearl unless Quantus was picked (Settings → GPU Coin).
    {
      region: DEFAULTS.region, worker: defaultWorker(), address: '', mdlAddress: '', mode: DEFAULT_MODE,
      cpuMining: false, xmrAddress: '', cpuThreads: 0, coin: 'pearl', quantusAddress: '',
    },
    saved,
  ));
  // The renderer picks the nearest region only when none was saved. The region
  // above is filled in either way, so it can't tell by itself.
  s.regionSaved = !!saved.region;
  // The renderer already resumes a START after an update; a launch at login asks
  // for the same thing. It starts only what START could: with no payout
  // address, an LLM-only or Auto run, or the CPU miner.
  if (startMiningOnLaunch) s.resumeMining = true;
  startMiningOnLaunch = false;
  return s;
});
ipcMain.handle('prefs:get', () => getPrefs());
ipcMain.handle('prefs:set', (_e, change) => setPrefs(change || {}));
ipcMain.handle('llm:status', () => ctx.llmStatus);
// `platform` rides along on the config the renderer already fetches at startup,
// so the UI can stop offering what this OS can't do (the mining compute modes on
// macOS) without a second round trip or a new preload method.
ipcMain.handle('config:get', () => {
  // What CPU mining would do here, for its Settings section: the default,
  // whether it can run at all (one thread asked for by hand), and whether this
  // install has the worker that does it.
  const plan = cpu.plan(0);
  return {
    // Each pool with its servers, for the Settings pool menu.
    pools: POOLS,
    // The worker default is this machine's, as settings:get fills it, so a
    // Worker box cleared in Settings falls back to it rather than to "rig01".
    defaults: Object.assign({}, DEFAULTS, { worker: defaultWorker() }),
    platform: { minerSupported: minerSupported(process.platform) },
    cpu: {
      available: plan.available, auto: plan.auto, reason: plan.reason, canMine: cpu.plan(1).threads > 0,
      worker: cpu.workerPath() != null,
    },
  };
});
ipcMain.handle('gpu:detect', () => detectGpu());
ipcMain.handle('gpu:cards', () => listCoinCards().catch(() => []));
ipcMain.handle('region:detect', () => detectRegion());
ipcMain.handle('balance:get', (_e, address, pool) => fetchBalance(address, liveEcon.PRL_USD, pool));
ipcMain.handle('xmr:balance', (_e, address, pool) => fetchXmrBalance(address, pool));
// The renderer has no Keccak, so the address check runs here.
ipcMain.handle('xmr:check', (_e, address) => checkXmrAddress(address));
// The same for a Quantus address, whose checksum is BLAKE2b.
ipcMain.handle('quantus:check', (_e, address) => checkQuantusAddress(address));
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
  const session = ctx.session;
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
    if (Date.now() - Number(saved.resumeHidden) < UPDATE_RESTART_MS) ctx.hiddenLaunch = true;
    delete saved.resumeHidden;
    persistSettings(saved);
  }
  createTray();
  // A launch at login waits to know there's a tray to start in. A window on
  // screen doesn't wait: the answer is in long before anyone closes it.
  if (ctx.hiddenLaunch) await confirmTray(TRAY_WAIT_MS).catch(() => {});
  else confirmTray(0).catch(() => {});
  createWindow();
  reportLastCrash();
  // Logging off or shutting down on Linux: let the window close.
  powerMonitor.on('shutdown', () => { ctx.quitting = true; });
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

// The first line of each run in the log file, and what the last run left if it
// closed while mining: a line in the log, and a box that opens the log folder,
// since a user who saw the app vanish won't know where to look.
function reportLastCrash() {
  send('miner:log', {
    level: 'info',
    line: 'LLMJob Earn ' + app.getVersion() + ' on ' + process.platform + ' ' + os.release() + '; log file: '
      + appLog.file,
  });
  const crash = appLog.takeCrash();
  if (!crash) return;
  const when = crash.startedAt ? ' (mining since ' + crash.startedAt + ')' : '';
  send('miner:log', {
    level: 'error',
    line: 'LLMJob Earn closed unexpectedly while mining' + when + '. Its log is in ' + appLog.file
      + ' and any crash report in ' + app.getPath('crashDumps'),
  });
  Promise.resolve(dialog.showMessageBox({
    type: 'warning',
    title: 'LLMJob Earn',
    message: 'LLMJob Earn closed unexpectedly while mining.',
    detail: 'Its log is saved on this PC. If you ask for help, please send the file earn.log from the log folder.',
    buttons: ['Open log folder', 'Close'],
    defaultId: 0,
    cancelId: 1,
  })).then((r) => {
    if (r && r.response === 0) shell.openPath(appLog.dir);
  }).catch(() => {});
}

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
  ctx.quitting = true;
  shutdownChildren();
});
