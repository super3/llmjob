'use strict';

// Electron main process. Thin shell: owns the window, persists settings, and
// bridges the renderer to the MinerManager (the real engine). Stats shown to
// the user come only from the engine's own output — no simulated data. All
// testable logic lives in ../shared and ./minerManager.

const { app, BrowserWindow, ipcMain, shell } = require('electron');
const { spawn, execFile } = require('child_process');
const path = require('path');
const fs = require('fs');
const http = require('http');
const https = require('https');

const { autoUpdater } = require('electron-updater');

const { MinerManager } = require('./minerManager');
const { EngineManager } = require('./engineManager');
const { initStats, applyEvent, snapshot } = require('../shared/miningStats');
const { REGIONS, DEFAULTS, MINER, endpointFor, difficultyForCard } = require('../shared/config');
const { progressPercent, bundledEnginePath } = require('../shared/engine');
const { formatUpdate } = require('../shared/updateStatus');
const { describeLaunchError } = require('../shared/engineError');
const { pickGpu } = require('../shared/gpu');
const earnings = require('../shared/earnings');
const format = require('../shared/format');

let win = null;
let miner = null;
let stats = null;
let ticker = null;

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
    estDay: earnings.estDailyUsdLabel(snap.total),
  };
}

// Stream a URL to a file, following redirects and reporting download progress.
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

// Extract the engine .exe from a downloaded zip to `dest`. Windows-only: uses
// PowerShell's Expand-Archive, so there's no extra runtime dependency.
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

async function startMining(settings) {
  persistSettings(settings);

  // Real stats only: the accumulator starts at zero and is filled in from the
  // engine's parsed output (see the miner 'event' handler below). The ticker
  // just re-emits the current snapshot each second so uptime advances.
  stats = initStats(Date.now());
  send('miner:stats', statsView(snapshot(stats, Date.now())));
  if (ticker) clearInterval(ticker);
  ticker = setInterval(() => send('miner:stats', statsView(snapshot(stats, Date.now()))), 1000);

  const endpoint = settings.endpoint || endpointFor(settings.region || DEFAULTS.region);
  send('miner:log', { level: 'info', line: 'connecting to ' + endpoint + ' · worker ' + (settings.worker || DEFAULTS.worker) });

  // Resolve the engine. A packaged build ships it under process.resourcesPath
  // (build.extraResources), so prefer that and skip the network entirely; only
  // fall back to the on-demand download when no bundled copy is present (e.g. a
  // dev run, or a build that shipped without the binary).
  let binaryPath = settings.binaryPath;
  const bundled = bundledEnginePath(process.resourcesPath, process.platform, settings.gpu);
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
function setupUpdater() {
  const push = (phase, payload) => send('app:update', formatUpdate(phase, payload));
  autoUpdater.autoDownload = true;
  autoUpdater.autoInstallOnAppQuit = true;
  autoUpdater.on('checking-for-update', () => push('checking'));
  autoUpdater.on('update-available', (info) => push('available', info));
  autoUpdater.on('update-not-available', () => push('none'));
  autoUpdater.on('download-progress', (p) => push('progress', p));
  autoUpdater.on('update-downloaded', (info) => push('ready', info));
  autoUpdater.on('error', (err) => {
    push('error');
    send('miner:log', { level: 'error', line: 'update error: ' + (err && err.message ? err.message : err) });
  });
  autoUpdater.checkForUpdates().catch((e) => {
    send('miner:log', { level: 'error', line: 'update check failed: ' + e.message });
  });
}

function createWindow() {
  win = new BrowserWindow({
    width: 620,
    height: 780,
    minWidth: 560,
    minHeight: 680,
    backgroundColor: '#fcfcfb',
    autoHideMenuBar: true,
    title: 'LLMJob Earn',
    icon: appIcon(),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  win.loadFile(path.join(__dirname, '..', 'renderer', 'index.html'));
}

ipcMain.handle('settings:get', () => Object.assign(
  { region: DEFAULTS.region, worker: DEFAULTS.worker, difficulty: DEFAULTS.difficulty, address: '' },
  loadSettings(),
));
ipcMain.handle('config:get', () => ({ regions: REGIONS, defaults: DEFAULTS, miner: MINER }));
ipcMain.handle('miner:difficultyForCard', (_e, name) => difficultyForCard(name));
ipcMain.handle('gpu:detect', () => detectGpu());
ipcMain.on('miner:start', (_e, settings) => {
  startMining(settings || {}).catch((e) => send('miner:log', { level: 'error', line: 'start failed: ' + e.message }));
});
ipcMain.on('miner:stop', () => stopMining());
ipcMain.on('open-external', (_e, url) => { shell.openExternal(url); });
ipcMain.on('app:update:install', () => {
  try {
    autoUpdater.quitAndInstall();
  } catch (e) {
    send('miner:log', { level: 'error', line: 'update install failed: ' + e.message });
  }
});

app.whenReady().then(() => {
  createWindow();
  if (app.isPackaged) setupUpdater();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  stopMining();
  if (process.platform !== 'darwin') app.quit();
});
