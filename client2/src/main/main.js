'use strict';

// Electron main process. Thin shell: owns the window, persists settings, and
// bridges the renderer to the MinerManager (real engine) and Simulator (live
// preview). All testable logic lives in ../shared and ./minerManager.

const { app, BrowserWindow, ipcMain, shell } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const { MinerManager } = require('./minerManager');
const { Simulator } = require('../shared/simulator');
const { REGIONS, DEFAULTS, MINER, endpointFor, difficultyForCard } = require('../shared/config');
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

function startMining(settings) {
  persistSettings(settings);

  // Live preview numbers (the engine itself reports via logs/events).
  sim = new Simulator({ uptimeSec: 0 });
  send('miner:stats', statsView(sim.snapshot()));
  if (ticker) clearInterval(ticker);
  ticker = setInterval(() => send('miner:stats', statsView(sim.step())), 1000);

  const endpoint = settings.endpoint || endpointFor(settings.region || DEFAULTS.region);
  send('miner:log', { level: 'info', line: 'connecting to ' + endpoint + ' · worker ' + (settings.worker || DEFAULTS.worker) });

  // Real alpha-miner engine.
  miner = new MinerManager({ spawn });
  miner.on('log', (l) => send('miner:log', l));
  miner.on('event', (e) => send('miner:event', e));
  miner.on('error', (err) => send('miner:log', {
    level: 'error',
    line: 'alpha-miner engine not found (' + err.message + '). Showing simulated stats — download the engine: ' + MINER.downloadUrl,
  }));
  miner.on('stopped', (code) => send('miner:log', { level: 'info', line: 'engine exited (code ' + code + ')' }));

  try {
    miner.start(Object.assign({ platform: process.platform }, settings));
  } catch (e) {
    send('miner:log', { level: 'error', line: 'failed to launch engine: ' + e.message });
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
ipcMain.on('miner:start', (_e, settings) => startMining(settings || {}));
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
