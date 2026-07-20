'use strict';

// Exposes a minimal, safe API to the renderer over the context bridge.
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('llmjob', {
  getSettings: () => ipcRenderer.invoke('settings:get'),
  getConfig: () => ipcRenderer.invoke('config:get'),
  difficultyForCard: (name) => ipcRenderer.invoke('miner:difficultyForCard', name),
  detectGpu: () => ipcRenderer.invoke('gpu:detect'),
  detectRegion: () => ipcRenderer.invoke('region:detect'),
  getBalance: (address) => ipcRenderer.invoke('balance:get', address),
  getMdlBalance: (address) => ipcRenderer.invoke('balance:getMdl', address),
  getLlmStatus: () => ipcRenderer.invoke('llm:status'),
  onLlm: (cb) => ipcRenderer.on('llm:status', (_e, d) => cb(d)),
  sendChat: (messages) => ipcRenderer.send('llm:chat', messages),
  onChatDelta: (cb) => ipcRenderer.on('llm:chat:delta', (_e, d) => cb(d)),
  onChatDone: (cb) => ipcRenderer.on('llm:chat:done', () => cb()),
  onChatError: (cb) => ipcRenderer.on('llm:chat:error', (_e, d) => cb(d)),
  getNodeStatus: () => ipcRenderer.invoke('node:status'),
  connectNode: (opts) => ipcRenderer.invoke('node:connect', opts),
  disconnectNode: () => ipcRenderer.invoke('node:disconnect'),
  onNodeStatus: (cb) => ipcRenderer.on('node:status', (_e, d) => cb(d)),
  openNodeDashboard: () => ipcRenderer.send('node:dashboard'),
  startMiner: (settings) => ipcRenderer.send('miner:start', settings),
  stopMiner: () => ipcRenderer.send('miner:stop'),
  openExternal: (url) => ipcRenderer.send('open-external', url),
  copyText: (text) => ipcRenderer.send('clipboard:write', text),
  fitWindow: () => ipcRenderer.send('app:fit'),
  onStats: (cb) => ipcRenderer.on('miner:stats', (_e, d) => cb(d)),
  onLog: (cb) => ipcRenderer.on('miner:log', (_e, d) => cb(d)),
  onEvent: (cb) => ipcRenderer.on('miner:event', (_e, d) => cb(d)),
  onEngine: (cb) => ipcRenderer.on('miner:engine', (_e, d) => cb(d)),
  onStopped: (cb) => ipcRenderer.on('miner:stopped', () => cb()),
  onUpdate: (cb) => ipcRenderer.on('app:update', (_e, d) => cb(d)),
  getVersion: () => ipcRenderer.invoke('app:version'),
  checkForUpdate: () => ipcRenderer.send('app:update:check'),
  installUpdate: () => ipcRenderer.send('app:update:install'),
});
