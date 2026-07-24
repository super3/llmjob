/** @jest-environment jsdom */
'use strict';

/* global window, document */

// Drives src/renderer/renderer.js against the real index.html markup with a
// fully stubbed window.llmjob bridge. Each load() re-requires the module into
// a fresh DOM; captured on* callbacks let tests fire main-process events.

const fs = require('fs');
const path = require('path');

const RENDERER = path.join(__dirname, '..', 'src', 'renderer', 'renderer.js');
const HTML = fs.readFileSync(path.join(__dirname, '..', 'src', 'renderer', 'index.html'), 'utf8');
const BODY = HTML
  .slice(HTML.indexOf('<body>') + '<body>'.length, HTML.indexOf('</body>'))
  .replace(/<script[\s\S]*?<\/script>/g, '');

const ADDR = 'prl1p' + 'a'.repeat(30);
const ADDR2 = 'prl1p' + 'c'.repeat(30);
const MDL = 'mdl1p' + 'b'.repeat(30);
const MDL2 = 'mdl1p' + 'd'.repeat(30);
const ENDPOINT = 'http://127.0.0.1:8080/v1';
const WEB_URL = 'http://127.0.0.1:8080';

const $ = (id) => document.getElementById(id);

let rafQueue = [];
const flushRaf = () => { rafQueue.splice(0).forEach((cb) => cb()); };

async function flush() {
  for (let i = 0; i < 30; i++) await Promise.resolve();
}

function click(elm) {
  elm.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
}

function setInput(elm, value) {
  elm.value = value;
  elm.dispatchEvent(new window.Event('input', { bubbles: true }));
}

function submitChat() {
  $('chat-form').dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true }));
}

class ROStub {
  constructor(cb) { ROStub.cb = cb; }
  observe(elm) { ROStub.observed = elm; }
}

// Full bridge: every invoke resolves realistic data, every on* captures its
// callback into cbs so tests can fire main-process events.
function makeFullApi() {
  const cbs = {};
  const api = {
    getSettings: jest.fn().mockResolvedValue({
      address: ADDR, worker: 'w1', region: 'us2', difficulty: 524288,
      mdlAddress: MDL, mode: 'auto', resumeMining: false,
    }),
    getConfig: jest.fn().mockResolvedValue({
      regions: {
        us2: { flag: 'US', label: 'US', name: 'Dallas' },
        eu1: { flag: 'EU', label: 'EU', name: 'Falkenstein' },
      },
    }),
    difficultyForCard: jest.fn().mockResolvedValue(2048),
    detectGpu: jest.fn().mockResolvedValue('RTX 4090'),
    detectRegion: jest.fn().mockResolvedValue('eu1'),
    getBalance: jest.fn().mockResolvedValue({ earned: 1234.5678, usd: 12.3 }),
    getMdlBalance: jest.fn().mockResolvedValue({ earned: 7.7, mdlAddress: MDL }),
    getLlmStatus: jest.fn().mockResolvedValue(null),
    getChatModels: jest.fn().mockResolvedValue({ proxy: [
      { id: 'qwen/qwen3.6-27b', label: 'Qwen3.6 27B' },
      { id: 'qwen/qwen3.6-35b-a3b', label: 'Qwen3.6 35B A3B' },
    ] }),
    onLlm: jest.fn((cb) => { cbs.llm = cb; }),
    sendChat: jest.fn(),
    onChatDelta: jest.fn((cb) => { cbs.chatDelta = cb; }),
    onChatDone: jest.fn((cb) => { cbs.chatDone = cb; }),
    onChatError: jest.fn((cb) => { cbs.chatError = cb; }),
    getNodeStatus: jest.fn().mockResolvedValue(null),
    connectNode: jest.fn().mockResolvedValue({ success: true, nodeId: 'n1', name: 'rig-a', user: 'alice' }),
    disconnectNode: jest.fn().mockResolvedValue(undefined),
    onNodeStatus: jest.fn((cb) => { cbs.node = cb; }),
    openNodeDashboard: jest.fn(),
    startMiner: jest.fn(),
    stopMiner: jest.fn(),
    openExternal: jest.fn(),
    copyText: jest.fn(),
    fitWindow: jest.fn(),
    onStats: jest.fn((cb) => { cbs.stats = cb; }),
    onLog: jest.fn((cb) => { cbs.log = cb; }),
    onEvent: jest.fn(),
    onEngine: jest.fn((cb) => { cbs.engine = cb; }),
    onStopped: jest.fn((cb) => { cbs.stopped = cb; }),
    onUpdate: jest.fn((cb) => { cbs.update = cb; }),
    getVersion: jest.fn().mockResolvedValue('9.9.9'),
    checkForUpdate: jest.fn(),
    installUpdate: jest.fn(),
  };
  return { api, cbs };
}

// Partial bridge: subscriptions present, action methods absent, invokes
// resolving empty/falsy values — exercises every fallback branch.
function makePartialApi() {
  const cbs = {};
  const api = {
    getSettings: jest.fn().mockResolvedValue({}),
    getConfig: jest.fn().mockResolvedValue(null),
    detectGpu: jest.fn().mockResolvedValue('GpuB'),
    detectRegion: jest.fn().mockResolvedValue(''),
    getLlmStatus: jest.fn().mockResolvedValue(undefined),
    getNodeStatus: jest.fn().mockResolvedValue(undefined),
    getVersion: jest.fn().mockResolvedValue(''),
    onLlm: jest.fn((cb) => { cbs.llm = cb; }),
    onChatDelta: jest.fn((cb) => { cbs.chatDelta = cb; }),
    onChatDone: jest.fn((cb) => { cbs.chatDone = cb; }),
    onChatError: jest.fn((cb) => { cbs.chatError = cb; }),
    onNodeStatus: jest.fn((cb) => { cbs.node = cb; }),
    onStats: jest.fn((cb) => { cbs.stats = cb; }),
    onLog: jest.fn((cb) => { cbs.log = cb; }),
    onEngine: jest.fn((cb) => { cbs.engine = cb; }),
    onStopped: jest.fn((cb) => { cbs.stopped = cb; }),
    onUpdate: jest.fn((cb) => { cbs.update = cb; }),
  };
  return { api, cbs };
}

function loadRenderer({ api, noApi, resizeObserver, mutate } = {}) {
  jest.resetModules();
  document.body.innerHTML = BODY;
  if (mutate) mutate();
  if (noApi) delete window.llmjob;
  else window.llmjob = api || {};
  rafQueue = [];
  window.requestAnimationFrame = (cb) => { rafQueue.push(cb); return rafQueue.length; };
  if (resizeObserver) window.ResizeObserver = resizeObserver;
  else delete window.ResizeObserver;
  require(RENDERER);
}

async function boot(opts) {
  loadRenderer(opts);
  await flush();
}

const makeReady = (cbs, extra) => cbs.llm(Object.assign({
  ready: true, endpoint: ENDPOINT, webUrl: WEB_URL, tokensPerSec: 12.34, model: 'gemma',
}, extra));

beforeEach(() => { jest.useFakeTimers(); });

afterEach(() => {
  jest.clearAllTimers();
  jest.useRealTimers();
  delete window.llmjob;
  delete window.ResizeObserver;
  delete ROStub.cb;
  delete ROStub.observed;
});

describe('boot with the full bridge', () => {
  it('populates settings, regions, balances, gpu and version at init', async () => {
    const { api } = makeFullApi();
    await boot({ api });
    expect($('addr-input').value).toBe(ADDR);
    expect($('set-worker').value).toBe('w1');
    expect($('set-mdl').value).toBe(MDL);
    const opts = Array.from($('set-region').options).map((o) => o.value);
    expect(opts).toEqual(['us2', 'eu1']);
    expect($('set-region').options[0].textContent).toBe('US US · Dallas');
    // detectRegion overrode the saved region, difficultyForCard the default diff
    expect($('set-region').value).toBe('eu1');
    expect($('set-difficulty').value).toBe('2048');
    expect(api.difficultyForCard).toHaveBeenCalledWith('RTX 4090');
    expect($('device-label').textContent).toBe('RTX 4090');
    expect($('balance').textContent).toBe('1,234.568');
    expect($('balance-usd').textContent).toBe('≈ $12.30');
    expect($('balance-meta').hidden).toBe(false);
    expect($('get-wallet').hidden).toBe(true);
    expect($('mdl-balance').textContent).toBe('7.700');
    expect($('mdl-balance-meta').hidden).toBe(false);
    expect($('app-version').textContent).toBe('v9.9.9');
    expect($('chat-suggestions').children).toHaveLength(3);
    expect($('btn-start').disabled).toBe(false);
    // llm defaults after a null status
    expect($('llm-hero-tps').textContent).toBe('0.0');
    expect($('llm-hero-detail').textContent).toBe('gemma-4-E4B-it');
    expect($('chat-stopped-model').textContent).toBe('the local model');
    expect($('api-model').textContent).toBe('—');
  });

  it('navigates tabs, settings, logs and unknown views', async () => {
    const { api, cbs } = makeFullApi();
    await boot({
      api,
      mutate: () => {
        const ghost = document.createElement('span');
        ghost.id = 'ghost-tab';
        ghost.setAttribute('data-tab', 'ghost');
        document.body.appendChild(ghost);
      },
    });
    click($('tab-chat'));
    expect($('view-chat').hidden).toBe(false);
    expect($('chat-stopped').hidden).toBe(false); // llm down → gate
    click($('tab-api'));
    expect($('view-api').hidden).toBe(false);
    expect($('tab-api').classList.contains('active')).toBe(true);
    // settings toggles back to the last real tab
    click($('btn-settings'));
    expect($('view-settings').hidden).toBe(false);
    click($('btn-settings'));
    expect($('view-api').hidden).toBe(false);
    // logs toggle + back link
    click($('btn-logs'));
    expect($('view-logs').hidden).toBe(false);
    click($('btn-logs'));
    expect($('view-api').hidden).toBe(false);
    click($('btn-logs'));
    click(document.querySelector('[data-back]'));
    expect($('view-api').hidden).toBe(false);
    // unknown data-tab hides every view
    click($('ghost-tab'));
    expect($('view-mine').hidden).toBe(true);
    expect($('view-chat').hidden).toBe(true);
    expect($('view-api').hidden).toBe(true);
    // chat tab focuses the composer once the model is up
    makeReady(cbs);
    click($('tab-chat'));
    jest.advanceTimersByTime(0);
    expect(document.activeElement).toBe($('chat-input'));
    click($('tab-mine'));
    expect($('view-mine').hidden).toBe(false);
  });

  it('switches compute modes and falls back on unknown ones', async () => {
    const { api } = makeFullApi();
    await boot({
      api,
      mutate: () => {
        const b = document.createElement('button');
        b.id = 'mode-empty';
        b.setAttribute('data-mode', '');
        document.getElementById('mode-seg').appendChild(b);
      },
    });
    click(document.querySelector('[data-mode="mining"]'));
    expect($('mode-hint').textContent).toMatch(/mining only/i);
    expect(document.querySelector('[data-mode="mining"]').classList.contains('active')).toBe(true);
    click(document.querySelector('[data-mode="llm"]'));
    expect($('mode-hint').textContent).toMatch(/Local model only/);
    click(document.querySelector('[data-mode="both"]'));
    click(document.querySelector('[data-mode="auto"]'));
    expect($('mode-hint').textContent).toMatch(/Balances mining/);
    // unknown mode → default hint
    click($('mode-empty'));
    expect($('mode-hint').textContent).toMatch(/Balances mining/);
    // canStart: invalid address + mining-only disables START
    setInput($('addr-input'), 'nope');
    click(document.querySelector('[data-mode="mining"]'));
    expect($('btn-start').disabled).toBe(true);
    click(document.querySelector('[data-mode="llm"]'));
    expect($('btn-start').disabled).toBe(false);
  });

  it('starts and stops mining, renders stats, logs and engine phases', async () => {
    const { api, cbs } = makeFullApi();
    await boot({
      api,
      mutate: () => {
        const b = document.createElement('button');
        b.id = 'mode-empty';
        b.setAttribute('data-mode', '');
        document.getElementById('mode-seg').appendChild(b);
      },
    });
    // stats before mining are ignored
    cbs.stats({ total: '9.9', acceptedLabel: '9', uptime: '9m', estDay: '$9', points: [1] });
    expect($('hashrate').textContent).toBe('0.0');
    // start() guard: invalid address + mining-only mode is a no-op
    setInput($('addr-input'), '');
    click(document.querySelector('[data-mode="mining"]'));
    click($('btn-start'));
    expect(api.startMiner).not.toHaveBeenCalled();
    // valid start
    setInput($('addr-input'), ADDR);
    click(document.querySelector('[data-mode="auto"]'));
    click($('btn-start'));
    expect(api.startMiner).toHaveBeenCalledWith({
      address: ADDR, mdlAddress: MDL, worker: 'w1', region: 'eu1', difficulty: 2048, mode: 'auto',
    });
    expect($('addr-static').hidden).toBe(false);
    expect($('addr-static').textContent).toBe(ADDR);
    expect($('btn-stop').hidden).toBe(false);
    expect($('mine-dot').className).toBe('dot2 on');
    expect($('log-term').textContent).toMatch(/starting LLMJob Earn/);
    // live stats: multi-point chart + gpu label
    cbs.stats({ total: '1.2', acceptedLabel: '34', uptime: '5m 00s', estDay: '$0.42', gpu: 'gpu-live', points: [1, 2, 3] });
    expect($('hashrate').textContent).toBe('1.2');
    expect($('accepted').textContent).toBe('34');
    expect($('uptime').textContent).toBe('5m 00s');
    expect($('estday').textContent).toBe('$0.42');
    expect($('device-label').textContent).toBe('gpu-live');
    expect($('mk-line').getAttribute('d')).toMatch(/^M0 .* L480 /);
    // single point (flat-span pad fallback), no gpu
    cbs.stats({ total: '1', acceptedLabel: '1', uptime: '1m', estDay: '$1', points: [5] });
    expect($('device-label').textContent).toBe('gpu-live');
    expect($('mk-line').getAttribute('d')).toMatch(/^M0 /);
    // empty + missing points → flat line
    cbs.stats({ total: '1', acceptedLabel: '1', uptime: '1m', estDay: '$1', points: [] });
    expect($('mk-line').getAttribute('d')).toBe('M0 55 L480 55');
    cbs.stats({ total: '1', acceptedLabel: '1', uptime: '1m', estDay: '$1' });
    expect($('mk-line').getAttribute('d')).toBe('M0 55 L480 55');
    // NaN points exercise the span fallback without throwing
    cbs.stats({ total: '1', acceptedLabel: '1', uptime: '1m', estDay: '$1', points: [NaN, NaN] });
    expect($('mk-line').getAttribute('d')).toMatch(/NaN/);
    // logs with and without an explicit level
    cbs.log({ line: 'warned', level: 'warn' });
    cbs.log({ line: 'plain' });
    const lines = $('log-term').querySelectorAll('.ln');
    expect(lines[lines.length - 2].className).toBe('ln warn');
    expect(lines[lines.length - 1].className).toBe('ln info');
    expect(lines[lines.length - 1].textContent).toMatch(/plain/);
    // engine phases
    cbs.engine(null);
    cbs.engine({ phase: 'downloading' });
    expect($('engine-status').hidden).toBe(false);
    expect($('engine-status').textContent).toMatch(/Downloading/);
    cbs.engine({ phase: 'ready' });
    expect($('engine-status').hidden).toBe(true);
    cbs.engine({ phase: 'error', message: 'boom' });
    expect($('engine-status').textContent).toBe('boom');
    expect($('engine-status').classList.contains('err')).toBe(true);
    cbs.engine({ phase: 'error' });
    expect($('engine-status').textContent).toMatch(/Engine setup failed/);
    cbs.engine({ phase: 'other' });
    // main-process stop resets the dashboard
    cbs.stopped();
    expect($('btn-start').hidden).toBe(false);
    expect($('hashrate').textContent).toBe('0.0');
    expect($('device-label').textContent).toBe('RTX 4090');
    expect($('engine-status').hidden).toBe(true);
    // restart with every settings fallback (empty worker/region/difficulty/mdl/mode)
    setInput($('set-worker'), '');
    $('set-region').value = 'zz'; // no such option → ''
    setInput($('set-difficulty'), '');
    setInput($('set-mdl'), 'not-an-mdl');
    click($('mode-empty'));
    click($('btn-start'));
    expect(api.startMiner).toHaveBeenLastCalledWith({
      address: ADDR, mdlAddress: '', worker: 'rig01', region: 'us2', difficulty: 524288, mode: 'mining',
    });
    // manual stop
    click($('btn-stop'));
    expect(api.stopMiner).toHaveBeenCalled();
    expect($('btn-start').hidden).toBe(false);
  });

  it('refreshes the pool balance with debounce, races and resets', async () => {
    const { api } = makeFullApi();
    await boot({ api });
    api.getBalance.mockClear();
    // null balance keeps the previous value
    api.getBalance.mockResolvedValueOnce(null);
    setInput($('addr-input'), ADDR2);
    setInput($('addr-input'), ADDR2); // second input clears the pending debounce
    jest.advanceTimersByTime(600);
    await flush();
    expect(api.getBalance).toHaveBeenCalledTimes(1);
    expect($('balance').textContent).toBe('1,234.568');
    // missing usd clears the fiat line
    api.getBalance.mockResolvedValueOnce({ earned: 2, usd: null });
    setInput($('addr-input'), ADDR);
    jest.advanceTimersByTime(600);
    await flush();
    expect($('balance').textContent).toBe('2.000');
    expect($('balance-usd').textContent).toBe('');
    // stale response for a superseded address is dropped
    let resolveBal;
    api.getBalance.mockReturnValueOnce(new Promise((r) => { resolveBal = r; }));
    setInput($('addr-input'), ADDR2);
    jest.advanceTimersByTime(600);
    await flush();
    setInput($('addr-input'), ADDR);
    resolveBal({ earned: 99, usd: 1 });
    await flush();
    expect($('balance').textContent).not.toBe('99.000');
    // invalid address resets balances and shows the wallet link
    setInput($('addr-input'), 'nope');
    expect($('balance').textContent).toBe('0.000');
    expect($('balance-usd').textContent).toBe('≈ $0.00');
    expect($('mdl-balance').textContent).toBe('0.000');
    expect($('balance-meta').hidden).toBe(true);
    expect($('get-wallet').hidden).toBe(false);
    // the minute poll ticks without a valid address (guard path)
    api.getBalance.mockClear();
    jest.advanceTimersByTime(60000);
    expect(api.getBalance).not.toHaveBeenCalled();
  });

  it('validates the MDL address, tracks its balance and drops stale replies', async () => {
    const { api } = makeFullApi();
    await boot({ api });
    setInput($('set-mdl'), '');
    expect($('mdl-note').textContent).toMatch(/Leave blank/);
    expect($('mdl-balance').textContent).toBe('0.000');
    expect($('mdl-balance-meta').hidden).toBe(true);
    setInput($('set-mdl'), 'garbage');
    expect($('mdl-note').textContent).toMatch(/Double-check it/);
    // matching mdlAddress applies
    api.getMdlBalance.mockResolvedValueOnce({ earned: 5.5, mdlAddress: MDL2.toUpperCase() });
    setInput($('set-mdl'), MDL2);
    expect($('mdl-note').textContent).toMatch(/Merge-mining MDL/);
    expect($('mdl-balance-meta').hidden).toBe(false);
    jest.advanceTimersByTime(600);
    await flush();
    expect($('mdl-balance').textContent).toBe('5.500');
    // reply without mdlAddress applies too
    api.getMdlBalance.mockResolvedValueOnce({ earned: 6.5 });
    setInput($('set-mdl'), MDL);
    jest.advanceTimersByTime(600);
    await flush();
    expect($('mdl-balance').textContent).toBe('6.500');
    // mismatched mdlAddress is ignored
    api.getMdlBalance.mockResolvedValueOnce({ earned: 7.5, mdlAddress: MDL2 });
    setInput($('set-mdl'), MDL);
    jest.advanceTimersByTime(600);
    await flush();
    expect($('mdl-balance').textContent).toBe('6.500');
    // null reply is ignored
    api.getMdlBalance.mockResolvedValueOnce(null);
    setInput($('set-mdl'), MDL);
    jest.advanceTimersByTime(600);
    await flush();
    expect($('mdl-balance').textContent).toBe('6.500');
    // mdl changed while the request was in flight
    let resolveMdl;
    api.getMdlBalance.mockReturnValueOnce(new Promise((r) => { resolveMdl = r; }));
    setInput($('set-mdl'), MDL2);
    jest.advanceTimersByTime(600);
    await flush();
    setInput($('set-mdl'), MDL);
    resolveMdl({ earned: 9.9, mdlAddress: MDL2 });
    await flush();
    expect($('mdl-balance').textContent).toBe('6.500');
    // mdl cleared entirely while the request was in flight
    api.getMdlBalance.mockReturnValueOnce(new Promise((r) => { resolveMdl = r; }));
    setInput($('set-mdl'), MDL);
    jest.advanceTimersByTime(600);
    await flush();
    $('set-mdl').value = '';
    resolveMdl({ earned: 4.4, mdlAddress: MDL });
    await flush();
    expect($('mdl-balance').textContent).toBe('6.500');
    $('set-mdl').value = MDL;
    // payout address changed while the request was in flight
    api.getMdlBalance.mockReturnValueOnce(new Promise((r) => { resolveMdl = r; }));
    setInput($('set-mdl'), MDL);
    jest.advanceTimersByTime(600);
    await flush();
    setInput($('addr-input'), ADDR2);
    resolveMdl({ earned: 8.8, mdlAddress: MDL });
    await flush();
    expect($('mdl-balance').textContent).toBe('6.500');
    // invalid payout address short-circuits the refresh
    setInput($('addr-input'), 'x');
    api.getMdlBalance.mockClear();
    setInput($('set-mdl'), MDL);
    jest.advanceTimersByTime(600);
    await flush();
    expect(api.getMdlBalance).not.toHaveBeenCalled();
  });

  it('renders llm status transitions on the hero and gates', async () => {
    const { api, cbs } = makeFullApi();
    await boot({ api });
    cbs.llm({ error: 'GPU died' });
    expect($('llm-hero-dot').className).toBe('dot2 err');
    expect($('llm-hero-detail').textContent).toBe('GPU died');
    expect($('llm-hero-detail').classList.contains('err')).toBe(true);
    expect($('llm-hero-tps').textContent).toBe('0.0');
    makeReady(cbs);
    expect($('llm-hero-tps').textContent).toBe('12.3');
    expect($('llm-hero-dot').className).toBe('dot2 on');
    expect($('llm-hero-detail').textContent).toBe('gemma');
    expect($('chat-running').hidden).toBe(false);
    expect($('chat-stopped').hidden).toBe(true);
    expect($('api-running').hidden).toBe(false);
    expect($('api-endpoint-url').textContent).toBe(ENDPOINT);
    expect($('api-model').textContent).toBe('gemma');
    // ready without endpoint keeps the last endpoint text, model is remembered
    cbs.llm({ ready: true });
    expect($('api-endpoint-url').textContent).toBe(ENDPOINT);
    expect($('llm-hero-detail').textContent).toBe('gemma');
    expect($('llm-hero-tps').textContent).toBe('0.0');
    // stopped again
    cbs.llm({});
    expect($('chat-stopped').hidden).toBe(false);
    expect($('api-stopped').hidden).toBe(false);
    expect($('chat-stopped-model').textContent).toBe('gemma');
  });

  it('runs the chat loop: send, stream, done, errors and new chat', async () => {
    const { api, cbs } = makeFullApi();
    await boot({ api });
    // submitting the pristine empty composer is a no-op
    submitChat();
    // not ready yet → submit is swallowed
    setInput($('chat-input'), 'early');
    expect($('chat-send').disabled).toBe(true);
    submitChat();
    expect(api.sendChat).not.toHaveBeenCalled();
    // ready without a model name → header falls back
    cbs.llm({ ready: true });
    setInput($('chat-input'), '  hi  ');
    expect($('chat-send').disabled).toBe(false);
    submitChat();
    expect($('chat-model').textContent).toBe('gemma-4-E4B-it');
    expect($('chat-head').hidden).toBe(false);
    expect($('chat-empty').hidden).toBe(true);
    // local model → no proxy id passed to main
    expect(api.sendChat).toHaveBeenCalledWith([{ role: 'user', content: 'hi' }], null);
    expect($('chat-input').value).toBe('');
    expect($('chat-send').disabled).toBe(true);
    let msgs = $('chat-messages').querySelectorAll('.chat-msg');
    expect(msgs).toHaveLength(2);
    expect(msgs[0].className).toBe('chat-msg user');
    expect(msgs[1].className).toBe('chat-msg assistant');
    expect(msgs[1].querySelector('.bubble').classList.contains('streaming')).toBe(true);
    // two addMsg calls queued a single rAF (throttled), flushing scrolls
    expect(rafQueue).toHaveLength(1);
    flushRaf();
    // deltas: null and empty are ignored, text appends
    cbs.chatDelta(null);
    cbs.chatDelta({ text: '' });
    cbs.chatDelta({ text: 'Hel' });
    flushRaf();
    cbs.chatDelta({ text: 'lo' });
    expect(msgs[1].querySelector('.bubble').textContent).toBe('Hello');
    // guards while streaming: submit, suggestion chip and new-chat are no-ops
    setInput($('chat-input'), 'while streaming');
    submitChat();
    click($('chat-suggestions').children[0]);
    click($('chat-new'));
    expect(api.sendChat).toHaveBeenCalledTimes(1);
    expect($('chat-messages').querySelectorAll('.chat-msg')).toHaveLength(2);
    // a ready status mid-stream does not abort the reply
    cbs.llm({ ready: true, model: 'gemma' });
    // done ends the stream; a duplicate done is ignored
    cbs.chatDone();
    expect(msgs[1].querySelector('.bubble').classList.contains('streaming')).toBe(false);
    cbs.chatDone();
    // a stray delta after the stream ended is dropped
    cbs.chatDelta({ text: 'late-delta' });
    expect(msgs[1].querySelector('.bubble').textContent).toBe('Hello');
    // a stray error with no active bubble only re-enables the composer
    cbs.chatError({ message: 'late' });
    // suggestion chip sends the canned prompt with history attached
    click($('tab-chat'));
    click($('chat-suggestions').children[0]);
    expect(api.sendChat).toHaveBeenCalledTimes(2);
    expect(api.sendChat.mock.calls[1][0]).toHaveLength(3);
    expect($('chat-model').textContent).toBe('gemma');
    // immediate done with no deltas stores an empty reply
    cbs.chatDone();
    // error without a message and without stream text
    setInput($('chat-input'), 'q1');
    submitChat();
    cbs.chatError(null);
    msgs = $('chat-messages').querySelectorAll('.chat-msg');
    let bubble = msgs[msgs.length - 1].querySelector('.bubble');
    expect(bubble.textContent).toBe('⚠ the chat request failed');
    expect(bubble.classList.contains('err')).toBe(true);
    // error after partial text keeps the partial reply
    setInput($('chat-input'), 'q2');
    submitChat();
    cbs.chatDelta({ text: 'part' });
    cbs.chatError({ message: 'oops' });
    msgs = $('chat-messages').querySelectorAll('.chat-msg');
    bubble = msgs[msgs.length - 1].querySelector('.bubble');
    expect(bubble.textContent).toBe('part\n\n⚠ oops');
    // llm dying mid-stream unbricks the composer via a synthetic error
    setInput($('chat-input'), 'q3');
    submitChat();
    cbs.llm({ ready: false });
    msgs = $('chat-messages').querySelectorAll('.chat-msg');
    bubble = msgs[msgs.length - 1].querySelector('.bubble');
    expect(bubble.textContent).toMatch(/the local LLM stopped/);
    // new chat wipes the thread (on the chat tab, then off it)
    makeReady(cbs);
    click($('chat-new'));
    expect($('chat-messages').children).toHaveLength(0);
    expect($('chat-empty').hidden).toBe(false);
    expect($('chat-head').hidden).toBe(true);
    click($('tab-mine'));
    click($('chat-new'));
    // whitespace-only submit is ignored
    setInput($('chat-input'), '   ');
    submitChat();
    expect(api.sendChat).toHaveBeenCalledTimes(5);
    // ending a stream off the chat tab skips the refocus
    click($('tab-chat'));
    setInput($('chat-input'), 'q4');
    submitChat();
    click($('tab-mine'));
    cbs.chatDone();
  });

  it('offers gateway models and routes them through the LLMJob proxy', async () => {
    const { api, cbs } = makeFullApi();
    await boot({ api });
    const sel = $('chat-model-select');
    // local option + the two gateway models, each tagged for the user
    expect(Array.from(sel.options).map((o) => o.value))
      .toEqual(['local', 'qwen/qwen3.6-27b', 'qwen/qwen3.6-35b-a3b']);
    expect(sel.options[1].textContent).toBe('Qwen3.6 27B · via LLMJob');
    // the picker sits ABOVE the running/stopped gate, so it stays reachable even
    // when the local LLM is off — the whole point of the proxy path
    expect($('chat-running').contains(sel)).toBe(false);
    click($('tab-chat'));
    // the local LLM is down, so the local model can't chat…
    expect($('chat-stopped').hidden).toBe(false);
    // …but choosing a gateway model opens the composer (proxy needs no local LLM)
    sel.value = 'qwen/qwen3.6-27b';
    sel.dispatchEvent(new window.Event('change', { bubbles: true }));
    expect($('chat-running').hidden).toBe(false);
    expect($('chat-stopped').hidden).toBe(true);
    setInput($('chat-input'), 'hola');
    expect($('chat-send').disabled).toBe(false);
    submitChat();
    expect(api.sendChat).toHaveBeenCalledWith([{ role: 'user', content: 'hola' }], 'qwen/qwen3.6-27b');
    expect($('chat-model').textContent).toBe('Qwen3.6 27B');
    // a local-LLM status change must NOT abort a proxy reply in flight
    const bubble = $('chat-messages').querySelector('.chat-msg.assistant .bubble');
    cbs.chatDelta({ text: 'ho' });
    cbs.llm({ ready: false });
    expect(bubble.textContent).toBe('ho'); // no "local LLM stopped" error injected
    expect($('chat-running').hidden).toBe(false); // gate stays open for the proxy model
    cbs.chatDone(); // the proxy stream finishes on its own
    expect(bubble.classList.contains('streaming')).toBe(false);
  });

  it('promotes the compute mode when starting the LLM from a gate', async () => {
    const { api, cbs } = makeFullApi();
    await boot({ api });
    const startLlm = document.querySelector('[data-start-llm]');
    // auto mode starts as-is
    click(startLlm);
    expect(api.startMiner).toHaveBeenCalledTimes(1);
    expect(api.startMiner.mock.calls[0][0].mode).toBe('auto');
    click($('btn-stop'));
    // mining-only + valid address → both
    click(document.querySelector('[data-mode="mining"]'));
    click(startLlm);
    expect(api.startMiner.mock.calls[1][0].mode).toBe('both');
    click($('btn-stop'));
    // mining-only + no address → llm-only
    setInput($('addr-input'), '');
    click(document.querySelector('[data-mode="mining"]'));
    click(startLlm);
    expect(api.startMiner.mock.calls[2][0].mode).toBe('llm');
    // already ready → no-op
    makeReady(cbs);
    click(startLlm);
    expect(api.startMiner).toHaveBeenCalledTimes(3);
  });

  it('copies and opens the API endpoint', async () => {
    const { api, cbs } = makeFullApi();
    await boot({ api });
    makeReady(cbs);
    click($('api-copy'));
    expect(api.copyText).toHaveBeenCalledWith(ENDPOINT);
    expect($('api-copy').textContent).toBe('Copied');
    jest.advanceTimersByTime(1200);
    expect($('api-copy').textContent).toBe('Copy API');
    click($('api-endpoint-url'));
    expect(api.copyText).toHaveBeenCalledTimes(2);
    jest.advanceTimersByTime(1200);
    click($('api-open'));
    expect(api.openExternal).toHaveBeenCalledWith(WEB_URL);
    // no webUrl → falls back to the endpoint
    cbs.llm({ ready: true, endpoint: ENDPOINT });
    click($('api-open'));
    expect(api.openExternal).toHaveBeenLastCalledWith(ENDPOINT);
    // external links go through the bridge
    click($('get-wallet'));
    expect(api.openExternal).toHaveBeenLastCalledWith('https://wallet.alphapool.tech/');
  });

  it('links and unlinks the node with pairing tokens', async () => {
    const { api, cbs } = makeFullApi();
    await boot({ api });
    // empty token → inline error
    click($('connect-link'));
    expect($('connect-error').hidden).toBe(false);
    expect($('connect-error').textContent).toMatch(/pairing token first/);
    expect(api.connectNode).not.toHaveBeenCalled();
    // pairing token flow toggle
    click($('connect-pair-toggle'));
    expect($('connect-pair').hidden).toBe(false);
    expect(document.activeElement).toBe($('connect-token'));
    click($('connect-pair-toggle'));
    expect($('connect-pair').hidden).toBe(true);
    // successful link uses the worker name
    $('connect-token').value = '  tok-1  ';
    click($('connect-link'));
    await flush();
    expect(api.connectNode).toHaveBeenCalledWith({ token: 'tok-1', name: 'w1' });
    expect($('connect-link').textContent).toBe('Link');
    expect($('connect-done').hidden).toBe(false);
    expect($('connect-form').hidden).toBe(true);
    expect($('connected-avatar').textContent).toBe('A');
    expect($('connected-title').textContent).toBe('alice');
    expect($('connected-name').textContent).toBe('rig-a');
    expect($('connect-token').value).toBe('');
    expect($('connect-hint').textContent).toBe('');
    // rename shortcut jumps to settings
    click($('connected-rename'));
    expect($('view-settings').hidden).toBe(false);
    // disconnect
    click($('connect-disconnect'));
    await flush();
    expect(api.disconnectNode).toHaveBeenCalled();
    expect($('connect-form').hidden).toBe(false);
    expect($('connect-hint').textContent).toBe('Not linked to an account');
    // failure with a server error message; empty worker omits the name
    setInput($('set-worker'), '');
    api.connectNode.mockResolvedValueOnce({ success: false, error: 'bad token' });
    $('connect-token').value = 'tok-2';
    click($('connect-link'));
    await flush();
    expect(api.connectNode).toHaveBeenLastCalledWith({ token: 'tok-2', name: undefined });
    expect($('connect-error').textContent).toBe('bad token');
    // null response → generic failure, via the Enter key
    api.connectNode.mockResolvedValueOnce(null);
    $('connect-token').value = 'tok-3';
    $('connect-token').dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Enter', cancelable: true }));
    await flush();
    expect($('connect-error').textContent).toBe('Connection failed.');
    // failure without an error string → generic failure
    api.connectNode.mockResolvedValueOnce({ success: false });
    $('connect-token').value = 'tok-4';
    click($('connect-link'));
    await flush();
    expect($('connect-error').textContent).toBe('Connection failed.');
    // other keys don't submit
    api.connectNode.mockClear();
    $('connect-token').dispatchEvent(new window.KeyboardEvent('keydown', { key: 'a' }));
    expect(api.connectNode).not.toHaveBeenCalled();
    // dashboard button
    click($('connect-dashboard'));
    expect(api.openNodeDashboard).toHaveBeenCalled();
    // pushed node status with minimal fields
    cbs.node({ connected: true });
    expect($('connected-title').textContent).toBe('Connected');
    expect($('connected-name').textContent).toBe('this rig');
    cbs.node({ connected: true, nodeId: 'n9' });
    expect($('connected-name').textContent).toBe('n9');
    cbs.node(null);
    expect($('connect-form').hidden).toBe(false);
  });

  it('drives the update checker through its phases', async () => {
    const { api, cbs } = makeFullApi();
    await boot({ api });
    click($('btn-check-update'));
    expect(api.checkForUpdate).toHaveBeenCalledTimes(1);
    expect($('btn-check-update').disabled).toBe(true);
    expect($('btn-check-update').textContent).toBe('Checking…');
    cbs.update(null);
    cbs.update({ show: true, phase: 'checking', text: 'Checking for updates…' });
    expect($('update-status').hidden).toBe(true);
    expect($('btn-check-update').textContent).toBe('Checking…');
    // transient "up to date" note auto-dismisses; a second one resets the timer
    cbs.update({ show: true, phase: 'none', text: 'Up to date', transient: true });
    expect($('update-status').hidden).toBe(false);
    expect($('update-status').textContent).toBe('Up to date');
    expect($('btn-check-update').disabled).toBe(false);
    expect($('btn-check-update').textContent).toBe('Check for updates');
    cbs.update({ show: true, phase: 'none', text: 'Still up to date', transient: true });
    jest.advanceTimersByTime(5000);
    expect($('update-status').hidden).toBe(true);
    // error state
    cbs.update({ show: true, phase: 'error', text: 'download failed', error: true });
    expect($('update-status').classList.contains('err')).toBe(true);
    // downloaded → the button becomes install-and-restart
    cbs.update({ show: true, phase: 'downloaded', text: 'Restart to update', ready: true });
    expect($('btn-check-update').textContent).toBe('Update & restart');
    expect($('btn-check-update').classList.contains('ready')).toBe(true);
    expect($('update-status').classList.contains('err')).toBe(false);
    click($('btn-check-update'));
    expect(api.installUpdate).toHaveBeenCalledTimes(1);
    expect(api.checkForUpdate).toHaveBeenCalledTimes(1);
  });

  it('debounces window refits through the resize observer', async () => {
    const { api } = makeFullApi();
    await boot({ api, resizeObserver: ROStub });
    expect(ROStub.observed).toBe(document.querySelector('.app'));
    ROStub.cb();
    jest.advanceTimersByTime(40);
    ROStub.cb(); // resets the pending debounce
    jest.advanceTimersByTime(80);
    expect(api.fitWindow).toHaveBeenCalledTimes(1);
  });
});

describe('partial bridge (fallback settings, missing action methods)', () => {
  it('applies every settings fallback and skips absent bridge calls', async () => {
    const { api, cbs } = makePartialApi();
    await boot({ api });
    expect($('addr-input').value).toBe('');
    expect($('set-worker').value).toBe('rig01');
    expect($('set-difficulty').value).toBe('524288');
    expect($('set-mdl').value).toBe('');
    expect($('set-region').options).toHaveLength(0);
    expect($('mode-hint').textContent).toMatch(/mining only/i); // s.mode → 'mining'
    expect($('btn-start').disabled).toBe(true);
    expect($('device-label').textContent).toBe('GpuB');
    expect($('app-version').textContent).toBe('—'); // empty version ignored
    // balance refreshes bail on the missing invoke methods
    setInput($('addr-input'), ADDR);
    setInput($('set-mdl'), MDL);
    jest.advanceTimersByTime(600);
    await flush();
    expect($('balance').textContent).toBe('0.000');
    expect($('mdl-balance').textContent).toBe('0.000');
    // ready llm but no sendChat → submit is swallowed
    cbs.llm({ ready: true, endpoint: ENDPOINT });
    setInput($('chat-input'), 'hello');
    submitChat();
    expect($('chat-messages').children).toHaveLength(0);
    // copy without a clipboard bridge still flashes Copied
    click($('api-copy'));
    expect($('api-copy').textContent).toBe('Copied');
    jest.advanceTimersByTime(1200);
    expect($('api-copy').textContent).toBe('Copy API');
    // open with a url but no shell bridge
    click($('api-open'));
    // connect/disconnect/dashboard without the node bridge
    $('connect-token').value = 'tok';
    click($('connect-link'));
    await flush();
    expect($('connect-link').disabled).toBe(false);
    click($('connect-disconnect'));
    await flush();
    click($('connect-dashboard'));
    // start/stop without the miner bridge still flip local state
    click($('btn-start'));
    expect($('btn-stop').hidden).toBe(false);
    click($('btn-stop'));
    expect($('btn-start').hidden).toBe(false);
    // update ready but no installer; then not-ready with no checker
    cbs.update({ show: true, phase: 'downloaded', text: 'r', ready: true });
    click($('btn-check-update'));
    cbs.update({ show: true, phase: 'none', text: 'n' });
    click($('btn-check-update'));
    expect($('btn-check-update').textContent).toBe('Check for updates');
    // external link without the shell bridge
    click($('get-wallet'));
  });
});

describe('no bridge at all', () => {
  it('boots and stays interactive with window.llmjob missing', async () => {
    await boot({ noApi: true });
    expect($('set-region').options).toHaveLength(0);
    expect($('chat-suggestions').children).toHaveLength(3);
    expect($('mode-hint').textContent).toMatch(/Balances mining/); // default auto
    expect($('device-label').textContent).toBe('GPU · auto-detect');
    // auto mode can start even with no address or miner bridge
    click($('btn-start'));
    expect($('btn-stop').hidden).toBe(false);
    expect($('log-term').textContent).toMatch(/starting LLMJob Earn/);
    click($('btn-stop'));
    // chat gated (llm never comes up)
    setInput($('chat-input'), 'x');
    submitChat();
    expect($('chat-messages').children).toHaveLength(0);
    // endpoint actions with no endpoint
    click($('api-copy'));
    expect($('api-copy').textContent).toBe('Copy API');
    click($('api-open'));
    // connect actions bail
    $('connect-token').value = 'tok';
    click($('connect-link'));
    await flush();
    expect($('connect-done').hidden).toBe(true);
    click($('connect-disconnect'));
    await flush();
    click($('connect-dashboard'));
    // update check bails
    click($('btn-check-update'));
    expect($('btn-check-update').textContent).toBe('Check for updates');
    // external links are inert but preventDefault'd
    click($('get-wallet'));
  });
});

describe('deferred init and window-fit guards', () => {
  it('waits for DOMContentLoaded when the document is still loading', async () => {
    Object.defineProperty(document, 'readyState', { configurable: true, get: () => 'loading' });
    try {
      loadRenderer({ api: { fitWindow: jest.fn() } }); // no ResizeObserver → fit bails
      expect($('chat-suggestions').children).toHaveLength(0); // init deferred
      document.dispatchEvent(new window.Event('DOMContentLoaded'));
      await flush();
      expect($('chat-suggestions').children).toHaveLength(3);
    } finally {
      delete document.readyState;
    }
  });

  it('skips the resize observer when the app root is missing', async () => {
    const api = { fitWindow: jest.fn(), detectGpu: jest.fn().mockResolvedValue('') };
    await boot({
      api,
      resizeObserver: ROStub,
      mutate: () => { document.querySelector('.app').className = ''; },
    });
    expect(ROStub.observed).toBeUndefined();
    // detectGpu resolved falsy → label untouched
    expect($('device-label').textContent).toBe('GPU · auto-detect');
  });
});

describe('init interleavings', () => {
  it('does not clobber a session started while gpu detection is in flight', async () => {
    const { api } = makeFullApi();
    let resolveGpu;
    api.detectGpu = jest.fn(() => new Promise((r) => { resolveGpu = r; }));
    api.difficultyForCard = jest.fn().mockResolvedValue(0);
    loadRenderer({ api });
    await flush(); // init parked awaiting detectGpu
    click($('btn-start')); // user starts mining mid-init
    expect(api.startMiner).toHaveBeenCalledTimes(1);
    resolveGpu('GPU-X');
    await flush();
    // mining → the label and region are left alone
    expect($('device-label').textContent).toBe('GPU · auto-detect');
    expect(api.detectRegion).not.toHaveBeenCalled();
    // difficultyForCard returned a falsy value → keep the stored one
    expect($('set-difficulty').value).toBe('524288');
  });

  it('resumes mining from saved settings', async () => {
    const { api } = makeFullApi();
    api.getSettings.mockResolvedValue({ address: ADDR, mode: 'mining', resumeMining: true });
    await boot({ api });
    expect(api.startMiner).toHaveBeenCalledTimes(1);
    expect(api.startMiner.mock.calls[0][0].mode).toBe('mining');
    expect($('btn-stop').hidden).toBe(false);
  });

  it('does not resume without a valid payout address', async () => {
    const { api } = makeFullApi();
    api.getSettings.mockResolvedValue({ address: 'bad', mode: 'mining', resumeMining: true });
    await boot({ api });
    expect(api.startMiner).not.toHaveBeenCalled();
    expect($('btn-start').hidden).toBe(false);
  });
});
