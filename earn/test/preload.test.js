'use strict';

jest.mock('electron', () => ({
  contextBridge: { exposeInMainWorld: jest.fn() },
  ipcRenderer: { invoke: jest.fn(), send: jest.fn(), on: jest.fn() },
}));

const { contextBridge, ipcRenderer } = require('electron');
require('../src/main/preload');

// The API object preload exposed on the context bridge.
const [bridgeName, api] = contextBridge.exposeInMainWorld.mock.calls[0];

beforeEach(() => {
  ipcRenderer.invoke.mockClear();
  ipcRenderer.send.mockClear();
  ipcRenderer.on.mockClear();
});

it('exposes the llmjob API on the context bridge', () => {
  expect(bridgeName).toBe('llmjob');
  expect(typeof api).toBe('object');
});

describe('invoke-based methods', () => {
  // [apiMethod, channel, arg]
  const cases = [
    ['getSettings', 'settings:get'],
    ['getConfig', 'config:get'],
    ['difficultyForCard', 'miner:difficultyForCard', 'RTX 4090'],
    ['detectGpu', 'gpu:detect'],
    ['detectRegion', 'region:detect'],
    ['getBalance', 'balance:get', 'prl1abc'],
    ['getMdlBalance', 'balance:getMdl', 'prl1abc'],
    ['getLlmStatus', 'llm:status'],
    ['getChatModels', 'chat:models'],
    ['getNodeStatus', 'node:status'],
    ['connectNode', 'node:connect', { token: 't' }],
    ['disconnectNode', 'node:disconnect'],
    ['getVersion', 'app:version'],
  ];

  cases.forEach(([method, channel, arg]) => {
    it(`${method} → ipcRenderer.invoke(${channel})`, () => {
      api[method](arg);
      if (arg === undefined) expect(ipcRenderer.invoke).toHaveBeenCalledWith(channel);
      else expect(ipcRenderer.invoke).toHaveBeenCalledWith(channel, arg);
    });
  });
});

describe('send-based methods', () => {
  const cases = [
    ['openNodeDashboard', 'node:dashboard'],
    ['startMiner', 'miner:start', { address: 'prl1abc' }],
    ['stopMiner', 'miner:stop'],
    ['openExternal', 'open-external', 'https://llmjob.com'],
    ['copyText', 'clipboard:write', 'text'],
    ['fitWindow', 'app:fit'],
    ['checkForUpdate', 'app:update:check'],
    ['installUpdate', 'app:update:install'],
  ];

  cases.forEach(([method, channel, arg]) => {
    it(`${method} → ipcRenderer.send(${channel})`, () => {
      api[method](arg);
      if (arg === undefined) expect(ipcRenderer.send).toHaveBeenCalledWith(channel);
      else expect(ipcRenderer.send).toHaveBeenCalledWith(channel, arg);
    });
  });

  it('sendChat wraps the messages + chosen model into one payload', () => {
    api.sendChat([{ role: 'user', content: 'hi' }], 'qwen/qwen3.6-27b');
    expect(ipcRenderer.send).toHaveBeenCalledWith('llm:chat',
      { messages: [{ role: 'user', content: 'hi' }], model: 'qwen/qwen3.6-27b' });
  });
});

describe('event subscriptions', () => {
  // [apiMethod, channel, forwardsPayload]
  const cases = [
    ['onLlm', 'llm:status', true],
    ['onChatDelta', 'llm:chat:delta', true],
    ['onChatDone', 'llm:chat:done', false],
    ['onChatError', 'llm:chat:error', true],
    ['onNodeStatus', 'node:status', true],
    ['onStats', 'miner:stats', true],
    ['onLog', 'miner:log', true],
    ['onEvent', 'miner:event', true],
    ['onEngine', 'miner:engine', true],
    ['onStopped', 'miner:stopped', false],
    ['onUpdate', 'app:update', true],
  ];

  cases.forEach(([method, channel, forwardsPayload]) => {
    it(`${method} registers ${channel} and forwards to the callback`, () => {
      const cb = jest.fn();
      api[method](cb);

      expect(ipcRenderer.on).toHaveBeenCalledWith(channel, expect.any(Function));
      const handler = ipcRenderer.on.mock.calls.find((c) => c[0] === channel)[1];

      handler({ sender: 'evt' }, { some: 'data' });
      if (forwardsPayload) expect(cb).toHaveBeenCalledWith({ some: 'data' });
      else expect(cb).toHaveBeenCalledWith();
    });
  });
});
