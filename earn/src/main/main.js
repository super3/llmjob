'use strict';

// Electron main process. Thin shell: owns the window, persists settings, and
// bridges the renderer to our own Pearl miner. Stats shown to the user come only
// from the engine's own output — no simulated data. All testable logic lives in
// ../shared and the engine modules beside this file.

const { app, BrowserWindow, Menu, ipcMain, shell, clipboard } = require('electron');
const path = require('path');
const fs = require('fs');

const { autoUpdater } = require('electron-updater');

const net = require('net');
const { PearlEngine } = require('./pearlEngine');
const { coreFactory } = require('./pearlCore');
const { getJson } = require('./io');
const { detectRegion, detectGpusVram, postMinerReport } = require('./probe');
const probe = require('./probe');
const settingsStore = require('../shared/settingsStore');
const { initStats, applyEvent, snapshot } = require('../shared/miningStats');
const {
  REGIONS, DEFAULTS, MINER, NETWORK, ECON, ECON_API, resolveEndpoint, migrateRegion,
} = require('../shared/config');
const { defaultWorker } = require('../shared/worker');
const { resolveEconomics } = require('../shared/economics');
const { minerSupported, minerUnsupportedNote, autoUpdateSupported } = require('../shared/platform');
const { buildBalanceUrl, parseBalance } = require('../shared/balance');
const { isValidAddress } = require('../shared/address');
const { formatUpdate, describeUpdateError } = require('../shared/updateStatus');
const { buildMinerReports } = require('../shared/minerReport');
const { alignCudaDeviceOrder, clearCudaVisibleDevices, describeClearedCuda } = require('../shared/gpu');
const earnings = require('../shared/earnings');
const format = require('../shared/format');

// Number the GPUs the way nvidia-smi does, before anything opens a CUDA device.
// Everything here — the device label, per-card VRAM, temperatures, the board's
// rows — speaks nvidia-smi's indices, and the CUDA runtime does not unless told
// to. Set at load, because the mining core initialises CUDA inside THIS process
// and reads it then. For the same reason CUDA must see every card nvidia-smi
// lists; the removed value is logged on each start, since there is no window yet.
alignCudaDeviceOrder(process.env);
const clearedCudaLine = describeClearedCuda(clearCudaVisibleDevices(process.env));

let win = null;
let miner = null;
let stats = null;
let ticker = null;
let reporter = null;
// Bumped on every stop. A start is async (it reads the cards first), during
// which the user may press STOP; the in-flight start captures this epoch and, if
// it has since changed, aborts instead of starting a miner the user already
// stopped.
let miningEpoch = 0;

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

function send(channel, payload) {
  if (win && !win.isDestroyed()) win.webContents.send(channel, payload);
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

// Live network economics for earnings estimates. Starts as the static ECON
// fallback and is refreshed from the prlscan API so the app's $/day and the
// balance's USD figure track the real network + price instead of drifting
// (a stale fallback silently overstates earnings as the network grows).
let liveEcon = Object.assign({}, ECON, { live: { price: false, net: false, reward: false } });

// Refresh liveEcon from the prlscan API (price, network hashrate, emission).
// Best-effort: whatever doesn't come back stays on the previous/fallback value.
async function refreshEconomics() {
  const [market, metrics, blocks] = await Promise.all([
    getJson(ECON_API.price), getJson(ECON_API.metrics), getJson(ECON_API.blocks),
  ]);
  liveEcon = resolveEconomics({ market, metrics, blocks }, ECON);
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
    accepted: snap.accepted,
    acceptedLabel: format.formatInt(snap.accepted),
    rejected: snap.rejected,
    rejectedLabel: format.formatInt(snap.rejected),
    load: Math.round(snap.load),
    power: snap.power,
    // Every card that is mining, not just the first. On a multi-card rig showing
    // one name is the same lie issue #226 was about, one level up.
    gpu: format.formatDeviceLabel(snap.gpus.map((g) => g.gpu)) || snap.gpu,
    temp: snap.temp,
    uptime: format.formatUptime(snap.uptimeSec),
    estDay: earnings.estDailyUsdLabel(snap.total, liveEcon),
  };
}

async function startMining(settings) {
  // Already mining: keep the existing miner — reassigning it would orphan an
  // unstoppable engine and start a second one on the same GPU.
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

  // Publish live status to the network page's board while mining, including
  // live per-card VRAM (used/total).
  const report = async () => {
    const snap = snapshot(stats, Date.now());
    const gpuVram = await detectGpusVram();
    buildMinerReports(settings, snap, gpuVram, app.getVersion()).forEach(postMinerReport);
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
  const gpus = await probe.detectMinerGpus();
  if (epoch !== miningEpoch) return;

  miner = new PearlEngine({
    connect: (host, port) => net.connect(port, host),
    createCore: coreFactory({ resourcesPath: process.resourcesPath }),
    // The card temperature the UI shows next to the GPU name. Our core has no
    // NVML reading of its own to forward the way alpha-miner did, so the engine
    // polls nvidia-smi for it; a rig without nvidia-smi just shows the name.
    readTemps: () => probe.detectGpuTemps(),
  });
  wireMinerEvents(miner, endpoint);
  try {
    miner.start(Object.assign({}, settings, { endpoint, gpus, gpu: settings.gpu || null }));
  } catch (e) {
    reportLaunchFailure(e);
  }
}

// Miner event wiring. PearlEngine translates our miner's events into the two
// the UI reads (a periodic `status` and one `connected`), so nothing
// downstream of here knows what produced them.
function wireMinerEvents(miner, endpoint) {
  miner.on('log', (l) => send('miner:log', l));
  // Said once, not on every 5s retry. The engine's own line repeats verbatim and
  // names neither the host it tried nor the fact that the name never resolved,
  // so a rig that cannot do DNS looks identical to a pool that is down — which
  // is how a user ends up staring at eight lines of "No such host is known".
  let dnsHinted = false;
  miner.on('event', (e) => {
    applyEvent(stats, e, Date.now());
    if (e.type === 'connect-failed' && e.dns && !dnsHinted) {
      dnsHinted = true;
      send('miner:log', {
        level: 'warn',
        line: 'could not resolve ' + endpoint + ' — nothing is wrong with the GPU.'
          + ' Check DNS/VPN/firewall, or pick another region in Settings.',
      });
    }
    send('miner:event', e);
  });
  miner.on('error', (err) => reportLaunchFailure(err));
  miner.on('stopped', (code) => send('miner:log', { level: 'info', line: 'engine exited (code ' + code + ')' }));
}

function stopMining() {
  // Cancel any start still in flight (see miningEpoch) so it doesn't start a
  // miner after this stop.
  miningEpoch++;
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

// Detect the machine's GPU for the settings/device label. Resolves to a display
// name or null. Never rejects.
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
// main.js guards the call with app.isPackaged. Downloads happen automatically;
// the user chooses when to restart via the 'app:update:install' channel.
let manualUpdateCheck = false; // true while a user-initiated check is in flight
let updateTimer = null; // periodic re-check, so startup is not the only chance

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
  if (updateTimer.unref) updateTimer.unref();
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

// Start mining if this rig can: a valid payout address, and a platform with a
// miner. When it can't, tell the renderer — otherwise its optimistic "running"
// state shows STOP for a session in which nothing runs.
async function runPlan(settings) {
  // macOS has no mining engine at all (see shared/platform); say so rather than
  // leave the user looking at a START that did nothing.
  const note = minerUnsupportedNote(process.platform);
  if (note) send('miner:log', { level: 'warn', line: note });
  if (!isValidAddress(settings.address) || !minerSupported(process.platform)) {
    persistSettings(settings);
    send('miner:stopped');
    return;
  }
  try {
    await startMining(settings);
  } catch (e) {
    send('miner:log', { level: 'error', line: 'start failed: ' + e.message });
  }
}

// Single-flight wrapper around runPlan. `miner:start` is a plain IPC event with
// no re-entry guard, and a start awaits the card probe before the miner exists —
// so without this, two quick clicks could each pass the "already mining" check
// and put two engines on the same GPU. Concurrent calls collapse into the run
// already in flight.
let planRun = null;

function applyPlan(settings) {
  if (planRun) return planRun;
  planRun = runPlan(settings).finally(() => { planRun = null; });
  return planRun;
}

// The renderer gets a region that EXISTS. A saved AlphaPool id would otherwise
// reach a <select> with no matching option, which blanks it silently.
ipcMain.handle('settings:get', () => withLiveRegion(Object.assign(
  // worker defaults to this machine's hostname, not the shared "rig01" constant:
  // two rigs on one payout address under the same name collide into a single
  // board identity (and if either is multi-GPU, the other's row is dropped
  // outright). Only fills a FRESH install — loadSettings() below wins, so an
  // existing worker name is never rewritten out from under someone's board row.
  { region: DEFAULTS.region, worker: defaultWorker(), address: '', mdlAddress: '' },
  loadSettings(),
)));
// `platform` rides along on the config the renderer already fetches at startup,
// so the UI can stop offering what this OS can't do (mining, on macOS) without a
// second round trip or a new preload method.
ipcMain.handle('config:get', () => ({
  regions: REGIONS,
  defaults: DEFAULTS,
  miner: MINER,
  platform: { minerSupported: minerSupported(process.platform) },
}));
ipcMain.handle('gpu:detect', () => detectGpu());
ipcMain.handle('region:detect', () => detectRegion());
ipcMain.handle('balance:get', (_e, address) => fetchBalance(address, liveEcon.PRL_USD));
ipcMain.on('miner:start', (_e, settings) => applyPlan(settings || {}));
ipcMain.on('miner:stop', () => { stopMining(); });
ipcMain.on('open-external', (_e, url) => { openExternalSafe(url); });
// Re-fit the window to its content when the renderer's layout changes (tab
// switch, mining start/stop, etc.), so the frame never leaves a gap under the
// footer or clips a taller view.
ipcMain.on('app:fit', () => { fitWindowToContent(); });
ipcMain.on('clipboard:write', (_e, text) => { clipboard.writeText(String(text == null ? '' : text)); });
ipcMain.handle('app:version', () => app.getVersion());
ipcMain.on('app:update:check', () => checkForUpdate());
ipcMain.on('app:update:install', () => {
  try {
    // If mining right now, remember to resume automatically after the restart.
    if (stats) persistSettings(Object.assign({}, loadSettings(), { resumeMining: true }));
    // Stop the miner BEFORE relaunching, so the GPU is released deterministically
    // before the new instance starts mining on it.
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
  // Not on macOS: Squirrel.Mac checks the downloaded bundle's signature against
  // the running app's, and this build carries only an ad-hoc one, so wiring the
  // updater there buys a periodic download that always ends in an error bar the
  // user can do nothing about. checkForUpdate() sends them to Releases instead.
  if (app.isPackaged && autoUpdateSupported(process.platform)) setupUpdater();
  // Pull live network economics so earnings estimates and the balance's USD
  // figure reflect the real price + network, not the stale fallback constants.
  refreshEconomics();
  const econTimer = setInterval(refreshEconomics, 10 * 60 * 1000);
  if (econTimer.unref) econTimer.unref();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

// Stop the miner. Idempotent — both quit paths below call it, and on the common
// path both of them fire.
function shutdownChildren() {
  stopMining();
}

app.on('window-all-closed', () => {
  shutdownChildren();
  if (process.platform !== 'darwin') app.quit();
});

// The other way out, and until now the LEAKING one. Electron does not emit
// 'window-all-closed' when the quit was started programmatically — which is what
// the default menu's Quit role and Ctrl/Cmd+Q do. So the most ordinary way to
// close the app skipped the only cleanup hook and left the miner running: the
// user's GPU stayed pinned by work they could no longer see.
app.on('before-quit', () => {
  shutdownChildren();
});
