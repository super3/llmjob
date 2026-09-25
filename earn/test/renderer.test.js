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

const $ = (id) => document.getElementById(id);

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
    getVersion: jest.fn().mockResolvedValue(''),
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
  if (resizeObserver) window.ResizeObserver = resizeObserver;
  else delete window.ResizeObserver;
  require(RENDERER);
}

async function boot(opts) {
  loadRenderer(opts);
  await flush();
}

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
    expect($('btn-start').disabled).toBe(false);
    expect($('engine-status').hidden).toBe(true);
    // External links open in the browser through the bridge, not in the window.
    click($('get-wallet'));
    expect(api.openExternal).toHaveBeenCalledWith('https://github.com/pearl-research-labs/pearl/releases');
    // The LLM's tabs, hero column and compute mode are gone from the window.
    for (const id of ['tab-chat', 'tab-api', 'view-chat', 'view-api', 'llm-hero-tps', 'mode-seg']) {
      expect($(id)).toBeNull();
    }
  });

  it('navigates between mine, settings, logs and unknown views', async () => {
    const { api } = makeFullApi();
    await boot({
      api,
      mutate: () => {
        const ghost = document.createElement('span');
        ghost.id = 'ghost-tab';
        ghost.setAttribute('data-tab', 'ghost');
        document.body.appendChild(ghost);
      },
    });
    // the settings gear toggles to settings and back to mine
    click($('btn-settings'));
    expect($('view-settings').hidden).toBe(false);
    expect($('btn-settings').classList.contains('active')).toBe(true);
    click($('btn-settings'));
    expect($('view-mine').hidden).toBe(false);
    // logs toggle + back link — the footer link relabels, since it is also the
    // way out of the logs view
    expect($('btn-logs').textContent).toBe('VIEW LOGS');
    click($('btn-logs'));
    expect($('view-logs').hidden).toBe(false);
    expect($('btn-logs').textContent).toBe('CLOSE LOGS');
    click($('btn-logs'));
    expect($('view-mine').hidden).toBe(false);
    expect($('btn-logs').textContent).toBe('VIEW LOGS');
    click($('btn-logs'));
    click(document.querySelector('[data-back]'));
    expect($('view-mine').hidden).toBe(false);
    expect($('btn-logs').textContent).toBe('VIEW LOGS'); // ← Back relabels too
    // unknown data-tab hides every view
    click($('ghost-tab'));
    expect($('view-mine').hidden).toBe(true);
    expect($('view-settings').hidden).toBe(true);
    expect($('view-logs').hidden).toBe(true);
    // the brand returns to mine
    click($('tab-mine'));
    expect($('view-mine').hidden).toBe(false);
    expect($('tab-mine').classList.contains('active')).toBe(true);
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
      mdlAddress: MDL, resumeMining: false,
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

  // macOS: main.js reports minerSupported:false on config:get, because the
  // Pearl core is CUDA and Macs have no NVIDIA GPU. START would run nothing, so
  // it stays off and the status line says why instead of staying silent.
  it('keeps START off and explains why on a platform that cannot mine', async () => {
    const { api } = makeFullApi();
    api.getConfig.mockResolvedValue({ regions: {}, platform: { minerSupported: false } });
    await boot({ api });
    expect($('btn-start').disabled).toBe(true);
    expect($('engine-status').hidden).toBe(false);
    expect($('engine-status').textContent).toMatch(/NVIDIA GPU on Windows or Linux/);
    click($('btn-start'));
    expect(api.startMiner).not.toHaveBeenCalled();
  });

  it('allows START where mining works, including when config omits the platform', async () => {
    const { api } = makeFullApi();
    api.getConfig.mockResolvedValue({ regions: {} });
    await boot({ api });
    expect($('btn-start').disabled).toBe(false);
    expect($('engine-status').hidden).toBe(true);
  });

  it('starts and stops mining, renders stats, logs and engine phases', async () => {
    const { api, cbs } = makeFullApi();
    await boot({ api });
    // stats before mining are ignored
    cbs.stats({ total: '9.9', acceptedLabel: '9', rejectedLabel: '9', uptime: '9m', estDay: '$9', points: [1] });
    expect($('hashrate').textContent).toBe('0.0');
    // start() guard: no valid address is a no-op
    setInput($('addr-input'), '');
    expect($('btn-start').disabled).toBe(true);
    click($('btn-start'));
    expect(api.startMiner).not.toHaveBeenCalled();
    // valid start
    setInput($('addr-input'), ADDR);
    click($('btn-start'));
    expect(api.startMiner).toHaveBeenCalledWith({
      address: ADDR, worker: 'w1', region: 'eu1', mdlAddress: '',
    });
    expect($('addr-static').hidden).toBe(false);
    expect($('addr-static').textContent).toBe(ADDR);
    expect($('btn-stop').hidden).toBe(false);
    expect($('mine-dot').className).toBe('dot2 on');
    expect($('log-term').textContent).toMatch(/starting LLMJob Earn/);
    // live stats: multi-point chart + gpu label
    cbs.stats({ total: '1.2', acceptedLabel: '34', rejectedLabel: '2', uptime: '5m 00s', estDay: '$0.42', gpu: 'gpu-live', points: [1, 2, 3] });
    expect($('hashrate').textContent).toBe('1.2');
    expect($('accepted').textContent).toBe('34');
    expect($('rejected').textContent).toBe('2');
    expect($('uptime').textContent).toBe('5m 00s');
    expect($('estday').textContent).toBe('$0.42');
    // The engine's name wins: it is the card the core actually opened, while the
    // detected one is a startup guess that can name a different card entirely
    // (issue #226).
    expect($('device-label').textContent).toBe('gpu-live'); // no temp reported yet → bare name
    expect($('mk-line').getAttribute('d')).toMatch(/^M0 .* L480 /);
    // Once the engine reports a core temperature it rides alongside the name, so
    // a rig that keeps crashing can be checked for heat without nvidia-smi.
    cbs.stats({ total: '1.2', acceptedLabel: '34', uptime: '5m 00s', estDay: '$0.42', gpu: 'gpu-live', temp: 86.4, points: [1, 2, 3] });
    expect($('device-label').textContent).toBe('gpu-live (86°C)');
    // single point (flat-span pad fallback), no gpu — the label keeps whatever it
    // last showed, name and temperature both, rather than reverting to the
    // startup guess for a frame.
    cbs.stats({ total: '1', acceptedLabel: '1', uptime: '1m', estDay: '$1', points: [5] });
    expect($('device-label').textContent).toBe('gpu-live (86°C)');

    // A temperature with NO name on the frame still lands, against the name last
    // reported. Gating the label on the engine naming the card in the SAME frame
    // made the temperature permanently undisplayable, which no unit test caught
    // because the engine's event was correct — only running the app showed the
    // bare name.
    cbs.stats({ total: '1.2', acceptedLabel: '34', uptime: '5m 00s', estDay: '$0.42', temp: 64, points: [1, 2, 3] });
    expect($('device-label').textContent).toBe('gpu-live (64°C)');
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
    expect($('rejected').textContent).toBe('0');
    expect($('device-label').textContent).toBe('RTX 4090');
    expect($('engine-status').hidden).toBe(true);
    // restart with every settings fallback (empty worker/region)
    setInput($('set-worker'), '');
    $('set-region').value = 'zz'; // no such option → '' → falls back to defaults.region
    click($('btn-start'));
    expect(api.startMiner).toHaveBeenLastCalledWith({
      address: ADDR, worker: 'rig01', region: 'us2', mdlAddress: '',
    });
    // A new run forgets the last run's card, so an engine that names none (an
    // older core, or a rig with no CUDA device list to report) shows the detected
    // name again rather than a card the previous run happened to pick.
    cbs.stats({ total: '2.0', acceptedLabel: '1', uptime: '1m', estDay: '$1', points: [1, 2] });
    expect($('device-label').textContent).toBe('RTX 4090 (64°C)');
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
    // No payout address yet, so there is nothing to mine to.
    expect($('btn-start').disabled).toBe(true);
    expect($('device-label').textContent).toBe('GpuB');
    expect($('app-version').textContent).toBe('—'); // empty version ignored
    // balance refreshes bail on the missing invoke methods
    setInput($('addr-input'), ADDR);
    jest.advanceTimersByTime(600);
    await flush();
    expect($('balance').textContent).toBe('0.000');
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
    expect($('device-label').textContent).toBe('GPU · auto-detect');
    // a pasted address can start even with no miner bridge
    setInput($('addr-input'), ADDR);
    click($('btn-start'));
    expect($('btn-stop').hidden).toBe(false);
    expect($('log-term').textContent).toMatch(/starting LLMJob Earn/);
    click($('btn-stop'));
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
      click($('btn-logs')); // init deferred: nothing is wired yet
      expect($('view-logs').hidden).toBe(true);
      document.dispatchEvent(new window.Event('DOMContentLoaded'));
      await flush();
      click($('btn-logs'));
      expect($('view-logs').hidden).toBe(false);
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
    api.getSettings.mockResolvedValue({ address: ADDR, resumeMining: true });
    await boot({ api });
    expect(api.startMiner).toHaveBeenCalledTimes(1);
    expect(api.startMiner.mock.calls[0][0].address).toBe(ADDR);
    expect($('btn-stop').hidden).toBe(false);
  });

  it('does not resume without a valid payout address', async () => {
    const { api } = makeFullApi();
    api.getSettings.mockResolvedValue({ address: 'bad', resumeMining: true });
    await boot({ api });
    expect(api.startMiner).not.toHaveBeenCalled();
    expect($('btn-start').hidden).toBe(false);
  });
});

// The Mine view's height, asserted against the stylesheet rather than the DOM.
//
// jsdom does no layout, so the bug this guards cannot be caught by rendering:
// #view-mine was pinned to a hard `height: 470px` alongside the views that
// scroll internally, and the update banner (which sits inside it, above content
// of a fixed size) pushed the START button straight out the bottom of its own
// section and across the footer -- 39px over VIEW LOGS / NEED HELP.
//
// Nothing clipped or scrolled either, which is why it looked like a paint bug:
// a fixed height means the overflow never grows .app, so the renderer's
// ResizeObserver never fires and the window is never refitted.
describe('mine view height', () => {
  const CSS = fs.readFileSync(path.join(__dirname, '..', 'src', 'renderer', 'styles.css'), 'utf8');

  // Every rule whose selector list mentions #view-mine, as [selector, body].
  const mineRules = CSS
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .split('}')
    .map((chunk) => chunk.split('{'))
    .filter((p) => p.length === 2 && /(^|,)\s*#view-mine\s*(,|$)/.test(p[0]))
    .map(([sel, body]) => [sel.trim(), body.trim()]);

  it('is bounded by min-height, never a fixed height', () => {
    expect(mineRules.length).toBeGreaterThan(0);
    const decls = mineRules.map(([, body]) => body).join(';');
    // A bare `height:` on this view is the regression; min-height is the fix.
    expect(/(^|;)\s*height\s*:/.test(decls)).toBe(false);
    expect(/min-height\s*:\s*\d/.test(decls)).toBe(true);
  });

  // The views that scroll internally still need a definite height, or their
  // flex children (the log terminal) grow instead of scrolling.
  it('leaves the internally-scrolling views on a fixed height', () => {
    expect(/#view-logs[^{]*{[^}]*height:\s*\d/.test(CSS.replace(/\/\*[\s\S]*?\*\//g, ''))).toBe(true);
  });
});

// The hashrate sparkline's vertical scale.
//
// It takes its range from the visible points alone, with no fixed baseline. That
// is fine while the window still holds the ramp from zero, but once a rig has
// been up a few minutes every point is steady-state and the only thing left to
// scale against is the wiggle — so a card holding a flat 226 TH/s was drawing a
// violent sawtooth from about 1% of variation. A floor on the span keeps normal
// jitter looking like jitter while leaving real drops legible.
describe('hashrate sparkline scale', () => {
  // Y coordinates out of an SVG path's "M x y L x y ..." command string.
  const ys = (d) => d.split(/[ML]\s*/).filter(Boolean).map((p) => Number(p.trim().split(/\s+/)[1]));
  const spread = (d) => { const v = ys(d); return Math.max(...v) - Math.min(...v); };
  const H = 56;

  it('renders steady-state jitter as small, not full height', async () => {
    const { api, cbs } = makeFullApi();
    await boot({ api });
    setInput($('addr-input'), ADDR);
    click($('btn-start'));

    // ~1.8% peak-to-peak around 226 — what a healthy 4090 actually produces.
    const jitter = [226.1, 224.3, 227.0, 225.2, 226.8, 224.9, 226.4, 225.6];
    cbs.stats({ total: '226.0', acceptedLabel: '749', uptime: '8h 28m', estDay: '$2.50', gpu: 'g', points: jitter });
    const quiet = spread($('mk-line').getAttribute('d'));

    // Without the floor this filled the chart; it should now use well under half.
    expect(quiet).toBeLessThan(H * 0.45);
    expect(quiet).toBeGreaterThan(0);
  });

  // The jitter is anti-correlated (a 0.5s window that grabs a 13th batch borrows
  // it from the next), so a short centred mean cancels most of it. Measured on the
  // live app: 7.2px of 56 raw, 2.2px through a five-point window.
  it('smooths the alternating quantisation, not just the zoom', async () => {
    const { api, cbs } = makeFullApi();
    await boot({ api });
    setInput($('addr-input'), ADDR);
    click($('btn-start'));

    // Perfectly alternating, which is the worst case and close to what the
    // real -0.345 lag-1 autocorrelation produces.
    const zig = Array.from({ length: 30 }, (_, i) => (i % 2 ? 224 : 228));
    cbs.stats({ total: '226.0', acceptedLabel: '9', uptime: '5m', estDay: '$2.50', gpu: 'g', points: zig });
    const smoothed = spread($('mk-line').getAttribute('d'));

    // A five-point centred mean of a pure alternation is nearly constant, so
    // this should collapse to almost nothing.
    expect(smoothed).toBeLessThan(H * 0.1);
  });
  it('still gives a real drop the full chart', async () => {
    const { api, cbs } = makeFullApi();
    await boot({ api });
    setInput($('addr-input'), ADDR);
    click($('btn-start'));

    // A card falling out mid-window — this must stay unmistakable.
    const drop = [226, 226, 225, 226, 120, 118, 119, 121];
    cbs.stats({ total: '119.0', acceptedLabel: '749', uptime: '8h 30m', estDay: '$1.30', gpu: 'g', points: drop });
    expect(spread($('mk-line').getAttribute('d'))).toBeGreaterThan(H * 0.6);
  });

  // The floor is a fraction of the mean, so it must not divide by a zero mean
  // or a single point and emit NaN into the path.
  it('survives an all-zero window and a single point', async () => {
    const { api, cbs } = makeFullApi();
    await boot({ api });
    setInput($('addr-input'), ADDR);
    click($('btn-start'));

    cbs.stats({ total: '0.0', acceptedLabel: '0', uptime: '0m 05s', estDay: '$0.00', gpu: 'g', points: [0, 0, 0] });
    expect($('mk-line').getAttribute('d')).not.toMatch(/NaN/);
    cbs.stats({ total: '226.0', acceptedLabel: '1', uptime: '0m 06s', estDay: '$2.50', gpu: 'g', points: [226] });
    expect($('mk-line').getAttribute('d')).not.toMatch(/NaN/);
  });
});
