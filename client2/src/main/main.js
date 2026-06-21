'use strict';

// Electron main process. Thin shell: owns the window, persists settings, and
// bridges the renderer to the MinerManager (real engine) and Simulator (live
// preview). All testable logic lives in ../shared and ./minerManager.

const { app, BrowserWindow, ipcMain, shell } = require('electron');
const { spawn, execFile } = require('child_process');
const path = require('path');
const fs = require('fs');
const http = require('http');
const https = require('https');

const { MinerManager } = require('./minerManager');
const { EngineManager } = require('./engineManager');
const { Simulator } = require('../shared/simulator');
const { REGIONS, DEFAULTS, MINER, endpointFor, difficultyForCard } = require('../shared/config');
const { progressPercent } = require('../shared/engine');
const earnings = require('../shared/earnings');
const format = require('../shared/format');

let win = null;
let miner = null;
let sim = null;
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

// Map a raw simulator snapshot to the display fields the renderer expects.
function statsView(snap) {
  return {
    total: format.formatHashrate(snap.total),
    points: snap.points,
    accepted: snap.accepted,
    acceptedLabel: format.formatInt(snap.accepted),
    rejected: snap.rejected,
    load: Math.round(snap.load),
    power: snap.power,
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

  // Live preview numbers (the engine itself also reports via logs/events).
  sim = new Simulator({ uptimeSec: 0 });
  send('miner:stats', statsView(sim.snapshot()));
  if (ticker) clearInterval(ticker);
  ticker = setInterval(() => send('miner:stats', statsView(sim.step())), 1000);

  const endpoint = settings.endpoint || endpointFor(settings.region || DEFAULTS.region);
  send('miner:log', { level: 'info', line: 'connecting to ' + endpoint + ' · worker ' + (settings.worker || DEFAULTS.worker) });

  // Ensure the engine is installed — download and set it up on first run.
  let binaryPath = settings.binaryPath;
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
      send('miner:engine', { phase: 'error', message: e.message });
      send('miner:log', { level: 'error', line: 'engine setup failed: ' + e.message + ' — showing simulated stats. Manual download: ' + MINER.downloadUrl });
    }
  }

  // Real alpha-miner engine.
  miner = new MinerManager({ spawn });
  miner.on('log', (l) => send('miner:log', l));
  miner.on('event', (e) => send('miner:event', e));
  miner.on('error', (err) => send('miner:log', { level: 'error', line: 'engine error: ' + err.message }));
  miner.on('stopped', (code) => send('miner:log', { level: 'info', line: 'engine exited (code ' + code + ')' }));

  if (binaryPath) {
    try {
      miner.start(Object.assign({}, settings, { platform: process.platform, binaryPath: binaryPath }));
    } catch (e) {
      send('miner:log', { level: 'error', line: 'failed to launch engine: ' + e.message });
    }
  }
}

function stopMining() {
  if (ticker) {
    clearInterval(ticker);
    ticker = null;
  }
  if (miner) {
    miner.stop();
    miner = null;
  }
  send('miner:stopped');
}

function createWindow() {
  win = new BrowserWindow({
    width: 620,
    height: 780,
    minWidth: 560,
    minHeight: 680,
    backgroundColor: '#fcfcfb',
    autoHideMenuBar: true,
    title: 'LLMJob Miner',
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
ipcMain.on('miner:start', (_e, settings) => {
  startMining(settings || {}).catch((e) => send('miner:log', { level: 'error', line: 'start failed: ' + e.message }));
});
ipcMain.on('miner:stop', () => stopMining());
ipcMain.on('open-external', (_e, url) => { shell.openExternal(url); });

app.whenReady().then(() => {
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  stopMining();
  if (process.platform !== 'darwin') app.quit();
});
