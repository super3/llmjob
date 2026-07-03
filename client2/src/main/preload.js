'use strict';

// Exposes a minimal, safe API to the renderer over the context bridge.
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('llmjob', {
  getSettings: () => ipcRenderer.invoke('settings:get'),
  getConfig: () => ipcRenderer.invoke('config:get'),
  difficultyForCard: (name) => ipcRenderer.invoke('miner:difficultyForCard', name),
  startMiner: (settings) => ipcRenderer.send('miner:start', settings),
  stopMiner: () => ipcRenderer.send('miner:stop'),
  openExternal: (url) => ipcRenderer.send('open-external', url),
  onStats: (cb) => ipcRenderer.on('miner:stats', (_e, d) => cb(d)),
  onLog: (cb) => ipcRenderer.on('miner:log', (_e, d) => cb(d)),
  onEvent: (cb) => ipcRenderer.on('miner:event', (_e, d) => cb(d)),
  onEngine: (cb) => ipcRenderer.on('miner:engine', (_e, d) => cb(d)),
  onStopped: (cb) => ipcRenderer.on('miner:stopped', () => cb()),
  onUpdate: (cb) => ipcRenderer.on('app:update', (_e, d) => cb(d)),
  installUpdate: () => ipcRenderer.send('app:update:install'),
});
