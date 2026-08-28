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
      address: ADDR, worker: 'w1', region: 'us2',
    }),
    getConfig: jest.fn().mockResolvedValue({
      regions: {
        us2: { flag: 'US', label: 'US', name: 'Dallas' },
        eu1: { flag: 'EU', label: 'EU', name: 'Falkenstein' },
      },
      // main always sends these. The region fallback reads defaults.region
      // rather than carrying its own copy of the default, which is what used to
      // pin it to a pool-specific id.
      defaults: { region: 'us2' },
    }),
    detectGpu: jest.fn().mockResolvedValue('RTX 4090'),
    detectRegion: jest.fn().mockResolvedValue('eu1'),
    getBalance: jest.fn().mockResolvedValue({ earned: 1234.5678, usd: 12.3 }),
    getLlmStatus: jest.fn().mockResolvedValue(null),
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
    const opts = Array.from($('set-region').options).map((o) => o.value);
    expect(opts).toEqual(['us2', 'eu1']);
    expect($('set-region').options[0].textContent).toBe('US US · Dallas');
    expect($('set-region').value).toBe('eu1');
    expect($('device-label').textContent).toBe('RTX 4090');
    expect($('balance').textContent).toBe('1,234.568');
    expect($('balance-usd').textContent).toBe('≈ $12.30');
    expect($('balance-meta').hidden).toBe(false);
    expect($('get-wallet').hidden).toBe(true);
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
    // logs toggle + back link — the footer link relabels, since it is also the
    // way out of the logs view
    expect($('btn-logs').textContent).toBe('VIEW LOGS');
    click($('btn-logs'));
    expect($('view-logs').hidden).toBe(false);
    expect($('btn-logs').textContent).toBe('CLOSE LOGS');
    click($('btn-logs'));
    expect($('view-api').hidden).toBe(false);
    expect($('btn-logs').textContent).toBe('VIEW LOGS');
    click($('btn-logs'));
    click(document.querySelector('[data-back]'));
    expect($('view-api').hidden).toBe(false);
    expect($('btn-logs').textContent).toBe('VIEW LOGS'); // ← Back relabels too
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

  // Merge mining is gone from the UI but an address someone already configured
  // keeps earning. Settings are persisted FROM currentSettings(), so the value
  // has to survive a round trip it is never shown in — otherwise the first
  // Start quietly erases it and ends the earnings we kept it for.
  it('carries a stored MDL address through invisibly, with nothing to set it', async () => {
    const { api } = makeFullApi();
    const MDL = 'mdl1p' + 'b'.repeat(30);
    api.getSettings = jest.fn().mockResolvedValue({
      address: ADDR, worker: 'w1', region: 'eu1',
      mode: 'auto', mdlAddress: MDL, resumeMining: false,
    });
    await boot({ api });

    // nothing in the UI exposes or edits it
    expect($('set-mdl')).toBeNull();

    click($('btn-start'));
    expect(api.startMiner).toHaveBeenCalledWith(expect.objectContaining({ mdlAddress: MDL }));
  });

  // A downloaded update used to be announced only inside Settings — the one
  // screen a user has no reason to open — so a rig could sit on a stale build
  // with the fix already on disk. The banner rides the same state, and its
  // button does exactly what the ready Settings button does.
  it('announces a downloaded update on the Mine view and installs from it', async () => {
    const { api, cbs } = makeFullApi();
    await boot({ api });
    expect($('update-bar').hidden).toBe(true);

    cbs.update({ show: true, ready: true, version: '9.9.9', text: 'Update ready' });
    expect($('update-bar').hidden).toBe(false);
    expect($('update-bar-text').textContent).toBe('Update downloaded (v9.9.9) — restart to apply.');

    click($('update-bar-btn'));
    expect(api.installUpdate).toHaveBeenCalled();

    // a later non-ready state takes it away again
    cbs.update({ show: true, phase: 'idle', text: 'Up to date' });
    expect($('update-bar').hidden).toBe(true);
  });

  // Version is optional in the payload; the banner must not read "(vundefined)".
  it('omits the version when the update payload has none', async () => {
    const { api, cbs } = makeFullApi();
    await boot({ api });
    cbs.update({ show: true, ready: true, text: 'Update ready' });
    expect($('update-bar-text').textContent).toBe('Update downloaded — restart to apply.');
  });

  // The banner button is wired even on a bridge that cannot install (an older
  // preload, or a platform with no updater); clicking it must be a no-op rather
  // than a TypeError that takes the renderer down.
  it('the update banner button is inert when the bridge cannot install', async () => {
    const { api, cbs } = makeFullApi();
    delete api.installUpdate;
    await boot({ api });
    cbs.update({ show: true, ready: true, version: '9.9.9', text: 'Update ready' });
    expect($('update-bar').hidden).toBe(false);
    expect(() => click($('update-bar-btn'))).not.toThrow();
  });

  // Wording is deliberately ours, not the v2 mock's. The mock says "Restart &
  // update" on the banner and HIDE LOGS on the toggle; we ship "Update &
  // restart" and CLOSE LOGS, both chosen after the mock was exported. Pinned so
  // a later mock sync does not quietly flip them back.
  it('keeps our wording over the mock on the update banner and the logs toggle', async () => {
    const { api, cbs } = makeFullApi();
    await boot({ api });
    expect($('update-bar-btn').textContent).toBe('Update & restart');

    cbs.update({ show: true, ready: true, version: '9.9.9', text: 'Update ready' });
    expect($('btn-check-update').textContent).toBe('Update & restart');

    click($('btn-logs'));
    expect($('btn-logs').textContent).toBe('CLOSE LOGS');
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

  // macOS: main.js reports minerSupported:false on config:get, because there is
  // no alpha-miner build for it. Leaving "Mining" and "Mining+LLM" on screen
  // would arm a START that runs nothing at all.
  it('drops the mining modes when the platform has no engine', async () => {
    const { api } = makeFullApi();
    api.getConfig.mockResolvedValue({ regions: {}, platform: { minerSupported: false } });
    // Stored 'mining' on a box that cannot mine: the button is hidden, so leaving
    // the mode selected would light nothing and arm a run that does nothing.
    api.getSettings.mockResolvedValue({ address: ADDR, mode: 'mining' });
    await boot({ api });
    expect(document.querySelector('[data-mode="auto"]').classList.contains('active')).toBe(true);

    expect(document.querySelector('[data-mode="mining"]').hidden).toBe(true);
    // Mining+LLM is not hidden — it no longer exists.
    expect(document.querySelector('[data-mode="both"]')).toBeNull();
    // 'auto' and 'llm' survive: both degrade correctly to LLM-only, and 'auto'
    // is what a fresh install lands on.
    expect(document.querySelector('[data-mode="auto"]').hidden).toBe(false);
    expect(document.querySelector('[data-mode="llm"]').hidden).toBe(false);
    expect($('mode-hint').textContent).toMatch(/Runs the local LLM on this Mac/);
    // START is never blocked there — no mode left needs a payout address.
    setInput($('addr-input'), '');
    expect($('btn-start').disabled).toBe(false);

    // Picking LLM explicitly still reads normally.
    click(document.querySelector('[data-mode="llm"]'));
    expect($('mode-hint').textContent).toMatch(/Local model only/);
  });

  // A settings file written on a Windows/Linux rig (or by an older build) can
  // carry mode:'both'. Without the correction the segment would light nothing
  // and START would arm a plan whose mining half is refused anyway.
  it('rewrites a saved mining mode to auto on a platform that cannot mine', async () => {
    const { api } = makeFullApi();
    api.getConfig.mockResolvedValue({ regions: {}, platform: { minerSupported: false } });
    api.getSettings.mockResolvedValue({ address: ADDR, mode: 'both' });
    await boot({ api });

    expect(document.querySelector('[data-mode="auto"]').classList.contains('active')).toBe(true);
    click($('btn-start'));
    expect(api.startMiner).toHaveBeenCalledWith(expect.objectContaining({ mode: 'auto' }));
  });

  it('keeps every mode where mining works, including when config omits the platform', async () => {
    const { api } = makeFullApi();
    api.getConfig.mockResolvedValue({ regions: {}, platform: { minerSupported: true } });
    api.getSettings.mockResolvedValue({ address: ADDR, mode: 'both' });
    await boot({ api });
    expect(document.querySelector('[data-mode="mining"]').hidden).toBe(false);
    // A rig that stored the retired 'both' lands on auto — the same plan, and a
    // button that actually exists to show as selected.
    expect(document.querySelector('[data-mode="auto"]').classList.contains('active')).toBe(true);
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
      address: ADDR, worker: 'w1', region: 'eu1', mode: 'auto', mdlAddress: '',
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
    // nvidia-smi's name wins over the engine's abbreviated one, so the app and
    // the network board never disagree about the same card.
    expect($('device-label').textContent).toBe('RTX 4090'); // no temp reported yet → bare name
    expect($('mk-line').getAttribute('d')).toMatch(/^M0 .* L480 /);
    // Once the engine reports a core temperature it rides alongside the name, so
    // a rig that keeps crashing can be checked for heat without nvidia-smi.
    cbs.stats({ total: '1.2', acceptedLabel: '34', uptime: '5m 00s', estDay: '$0.42', gpu: 'gpu-live', temp: 86.4, points: [1, 2, 3] });
    expect($('device-label').textContent).toBe('RTX 4090 (86°C)');
    // single point (flat-span pad fallback), no gpu — the label keeps whatever it
    // last showed, temperature included, rather than reverting.
    cbs.stats({ total: '1', acceptedLabel: '1', uptime: '1m', estDay: '$1', points: [5] });
    expect($('device-label').textContent).toBe('RTX 4090 (86°C)');

    // A temperature with NO name from the engine still lands. This is the real
    // shape our own miner reports: currentSettings() sends no `gpu`, so
    // PearlEngine reports `gpu: null` on every status. Gating the label on the
    // engine naming the card made the temperature permanently undisplayable,
    // which no unit test caught because the engine's event was correct — only
    // running the app showed the bare name.
    cbs.stats({ total: '1.2', acceptedLabel: '34', uptime: '5m 00s', estDay: '$0.42', temp: 64, points: [1, 2, 3] });
    expect($('device-label').textContent).toBe('RTX 4090 (64°C)');
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
    // restart with every settings fallback (empty worker/region/mode)
    setInput($('set-worker'), '');
    $('set-region').value = 'zz'; // no such option → '' → falls back to defaults.region
    click($('mode-empty'));
    click($('btn-start'));
    expect(api.startMiner).toHaveBeenLastCalledWith({
      address: ADDR, worker: 'rig01', region: 'us2', mode: 'mining', mdlAddress: '',
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
    expect($('balance-meta').hidden).toBe(true);
    expect($('get-wallet').hidden).toBe(false);
    // the minute poll ticks without a valid address (guard path)
    api.getBalance.mockClear();
    jest.advanceTimersByTime(60000);
    expect(api.getBalance).not.toHaveBeenCalled();
  });

  // A first run downloads a ~5 GB model; the hero has to say so. Otherwise it
  // shows the model name and a grey dot the whole time, which looks identical to
  // "nothing is happening" and gets users clicking START over and over.
  it('shows the download/startup note on the hero until the LLM is ready', async () => {
    const { api, cbs } = makeFullApi();
    await boot({ api });

    cbs.llm({ note: 'downloading model gemma… 42%' });
    expect($('llm-hero-detail').textContent).toBe('downloading model gemma… 42%');
    expect($('llm-hero-dot').className).toBe('dot2 busy');

    cbs.llm({ note: 'Starting…' });
    expect($('llm-hero-detail').textContent).toBe('Starting…');

    // A real error outranks the note — the note is only ever "still working".
    cbs.llm({ note: 'Starting…', error: 'Needs ~4 GB free VRAM' });
    expect($('llm-hero-detail').textContent).toBe('Needs ~4 GB free VRAM');
    expect($('llm-hero-dot').className).toBe('dot2 err');

    // …and so does being ready, so a stale note can't linger over a live model.
    makeReady(cbs, { note: 'Starting…' });
    expect($('llm-hero-detail').textContent).toBe('gemma');
    expect($('llm-hero-dot').className).toBe('dot2 on');
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
    expect(api.sendChat).toHaveBeenCalledWith([{ role: 'user', content: 'hi' }]);
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

  it('promotes the compute mode when starting the LLM from a gate', async () => {
    const { api, cbs } = makeFullApi();
    await boot({ api });
    const startLlm = document.querySelector('[data-start-llm]');
    // auto mode starts as-is
    click(startLlm);
    expect(api.startMiner).toHaveBeenCalledTimes(1);
    expect(api.startMiner.mock.calls[0][0].mode).toBe('auto');
    click($('btn-stop'));
    // mining-only + valid address → auto (was 'both', which meant the same)
    click(document.querySelector('[data-mode="mining"]'));
    click(startLlm);
    expect(api.startMiner.mock.calls[1][0].mode).toBe('auto');
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
    expect(api.openExternal).toHaveBeenLastCalledWith('https://github.com/pearl-research-labs/pearl/releases');
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
    const appEl = document.querySelector('.app');
    expect(ROStub.observed).toBe(appEl);
    let height = 0;
    appEl.getBoundingClientRect = () => ({ height });

    height = 500;
    ROStub.cb();
    jest.advanceTimersByTime(40);
    height = 600;
    ROStub.cb(); // a further height change resets the pending debounce
    jest.advanceTimersByTime(40);
    expect(api.fitWindow).not.toHaveBeenCalled();
    jest.advanceTimersByTime(40);
    expect(api.fitWindow).toHaveBeenCalledTimes(1);
  });

  // The reported bug: "the window keeps resizing automatically… it also starts
  // again as soon as I try to resize it manually". A ResizeObserver reports
  // width as well as height, and the fit only ever changes height — so a frame
  // resize that nudged .app's width (scrollbar, or DIP rounding on a scaled
  // display) asked for another fit, which nudged it again, forever.
  it('ignores width-only resizes so a fit cannot retrigger itself', async () => {
    const { api } = makeFullApi();
    await boot({ api, resizeObserver: ROStub });
    const appEl = document.querySelector('.app');
    appEl.getBoundingClientRect = () => ({ height: 500 }); // height never changes

    ROStub.cb(); // first observation: the height is new, so one fit is right
    jest.advanceTimersByTime(80);
    expect(api.fitWindow).toHaveBeenCalledTimes(1);

    ROStub.cb(); // the fit perturbed the width — must not schedule another
    ROStub.cb();
    jest.advanceTimersByTime(500);
    expect(api.fitWindow).toHaveBeenCalledTimes(1);
  });
});

describe('partial bridge (fallback settings, missing action methods)', () => {
  it('applies every settings fallback and skips absent bridge calls', async () => {
    const { api, cbs } = makePartialApi();
    await boot({ api });
    expect($('addr-input').value).toBe('');
    expect($('set-worker').value).toBe('rig01');
    expect($('set-region').options).toHaveLength(0);
    // A falsy stored mode falls back to DEFAULT_MODE, not to mining-only:
    // mining-only switches the LLM off silently, which reads as "the LLM is
    // broken" with nothing in the logs to say otherwise.
    expect($('mode-hint').textContent).toMatch(/Balances mining/);
    // START is live even with no payout address, because auto can serve the LLM
    // on its own. A real fresh install behaves the same way, since main sends
    // DEFAULT_MODE; only the old mining-only fallback made it look disabled.
    expect($('btn-start').disabled).toBe(false);
    expect($('device-label').textContent).toBe('GpuB');
    expect($('app-version').textContent).toBe('—'); // empty version ignored
    // balance refreshes bail on the missing invoke methods
    setInput($('addr-input'), ADDR);
    jest.advanceTimersByTime(600);
    await flush();
    expect($('balance').textContent).toBe('0.000');
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

  // A rig with no nvidia-smi has only the engine's label to go on, so it must
  // still get a name rather than an empty device row — the same fallback the
  // miner report keeps.
  it('falls back to the engine label when the GPU probe finds nothing', async () => {
    const { api, cbs } = makeFullApi();
    api.detectGpu = jest.fn().mockResolvedValue('');
    await boot({ api });
    expect($('device-label').textContent).toBe('GPU · auto-detect');

    setInput($('addr-input'), ADDR);
    click($('btn-start'));

    // Neither source has a name yet: no nvidia-smi, and the engine has not
    // named the card either. There is nothing to label the row with, so it is
    // left alone rather than reading "undefined (71°C)".
    cbs.stats({ total: '1.2', acceptedLabel: '3', uptime: '1m 00s', estDay: '$0.10', temp: 71, points: [1, 2, 3] });
    expect($('device-label').textContent).toBe('GPU · auto-detect');

    cbs.stats({ total: '1.2', acceptedLabel: '3', uptime: '1m 00s', estDay: '$0.10', gpu: 'RTX 5090', temp: 71, points: [1, 2, 3] });
    expect($('device-label').textContent).toBe('RTX 5090 (71°C)');
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
    loadRenderer({ api });
    await flush(); // init parked awaiting detectGpu
    click($('btn-start')); // user starts mining mid-init
    expect(api.startMiner).toHaveBeenCalledTimes(1);
    resolveGpu('GPU-X');
    await flush();
    // mining → the label and region are left alone
    expect($('device-label').textContent).toBe('GPU · auto-detect');
    expect(api.detectRegion).not.toHaveBeenCalled();
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
