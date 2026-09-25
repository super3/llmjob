'use strict';

// Unit tests for the Electron main process (src/main/main.js). Everything with
// side effects is mocked — Electron, the updater, fs, and the local engine/IO/
// probe modules — while the pure ../shared modules run for real. Each scenario
// re-requires main.js under fresh mocks (jest.resetModules) so the module-global
// state (win, miner, stats) starts clean, then drives the captured
// app/ipc/updater/engine callbacks.

jest.mock('electron', () => {
  const handlers = {};
  const listeners = {};
  const appEvents = {};
  const windows = [];
  const state = { readyCb: null };
  function makeWindow() {
    const wcEvents = {};
    const w = {
      loadFile: jest.fn(),
      show: jest.fn(),
      isDestroyed: jest.fn(() => false),
      isVisible: jest.fn(() => false),
      getContentSize: jest.fn(() => [620, 650]),
      setContentSize: jest.fn(),
      webContents: {
        send: jest.fn(),
        on: jest.fn((ev, fn) => { wcEvents[ev] = fn; }),
        executeJavaScript: jest.fn(() => Promise.resolve(500)),
      },
      _wcEvents: wcEvents,
    };
    windows.push(w);
    return w;
  }
  const menu = { popup: jest.fn() };
  return {
    app: {
      getPath: jest.fn(() => '/tmp/userData'),
      getVersion: jest.fn(() => '0.0.0-test'),
      isPackaged: false,
      quit: jest.fn(),
      on: jest.fn((ev, fn) => { appEvents[ev] = fn; }),
      whenReady: jest.fn(() => ({ then(fn) { state.readyCb = fn; } })),
    },
    BrowserWindow: Object.assign(jest.fn(() => makeWindow()), {
      getAllWindows: jest.fn(() => []),
    }),
    Menu: { buildFromTemplate: jest.fn(() => menu), _menu: menu },
    ipcMain: {
      handle: jest.fn((ch, fn) => { handlers[ch] = fn; }),
      on: jest.fn((ch, fn) => { listeners[ch] = fn; }),
    },
    shell: { openExternal: jest.fn(() => Promise.resolve()) },
    clipboard: { writeText: jest.fn() },
    _handlers: handlers,
    _listeners: listeners,
    _appEvents: appEvents,
    _windows: windows,
    _fireReady: () => state.readyCb && state.readyCb(),
  };
});

jest.mock('electron-updater', () => {
  const events = {};
  return {
    autoUpdater: {
      autoDownload: false,
      autoInstallOnAppQuit: false,
      on: jest.fn((ev, fn) => { events[ev] = fn; }),
      checkForUpdates: jest.fn(() => Promise.resolve()),
      quitAndInstall: jest.fn(),
      _events: events,
    },
  };
});

jest.mock('fs', () => ({
  existsSync: jest.fn(() => false),
  readFileSync: jest.fn(() => '{}'),
  writeFileSync: jest.fn(),
  copyFileSync: jest.fn(),
  chmodSync: jest.fn(),
  mkdirSync: jest.fn(),
}));

jest.mock('../src/main/io', () => ({
  getJson: jest.fn(() => Promise.resolve(null)),
}));

jest.mock('../src/main/probe', () => ({
  detectRegion: jest.fn(() => Promise.resolve('us1')),
  detectVram: jest.fn(() => Promise.resolve(null)),
  detectGpusVram: jest.fn(() => Promise.resolve([])),
  // No card list by default: one core that places itself, which is what a rig
  // with no nvidia-smi gets. The multi-card tests set it explicitly.
  detectMinerGpus: jest.fn(() => Promise.resolve([])),
  detectDriverMajor: jest.fn(() => Promise.resolve(600)),
  // Empty by default: an unknown compute capability is what keeps Windows on the
  // 1.8.6 fallback, which is the shape most of these tests were written against.
  // The tests that care about the 1.9.1b Windows package set it explicitly.
  detectComputeCaps: jest.fn(() => Promise.resolve([])),
  postMinerReport: jest.fn(() => Promise.resolve()),
  findFreePort: jest.fn(() => Promise.resolve(8080)),
  // GPU detection moved into probe so the GUI and the CLI share one
  // implementation — the GUI's own copy was Windows-only, which left the Linux
  // AppImage with no device name.
  detectGpuInfo: jest.fn(() => Promise.resolve(null)),
}));

jest.mock('../src/main/pearlEngine', () => {
  const { EventEmitter } = require('events');
  class PearlEngine extends EventEmitter {
    constructor(opts) {
      super();
      this.opts = opts;
      this._running = false;
      this.start = jest.fn((settings) => {
        this.settings = settings;
        if (PearlEngine.startError) throw PearlEngine.startError;
        this._running = true;
      });
      this.stop = jest.fn(() => { this._running = false; });
      this.isRunning = jest.fn(() => this._running);
      PearlEngine.instances.push(this);
    }
  }
  PearlEngine.instances = [];
  PearlEngine.startError = null;
  return { PearlEngine };
});

jest.mock('../src/main/pearlCore', () => ({
  loadCore: jest.fn(() => null),
  coreFactory: jest.fn(() => null),
}));

jest.mock('net', () => ({ connect: jest.fn(() => ({ on: jest.fn(), write: jest.fn(), destroy: jest.fn() })) }));

const { defaultWorker } = require('../src/shared/worker');

const VALID_ADDR = 'prl1p' + 'a'.repeat(30);

// main.js derives these from app.getPath('userData') with path.join, which
// yields backslashes on Windows — build the expectations the same way so the
// suite passes on every OS.
const path = require('path');
const SETTINGS_PATH = path.join('/tmp/userData', 'settings.json');

// ── timer capture (no real timers ever run) ──────────────────────────────────
const REAL_TIMERS = {
  setInterval: global.setInterval,
  clearInterval: global.clearInterval,
  setTimeout: global.setTimeout,
  clearTimeout: global.clearTimeout,
};
let timers = null;
function installTimers(withUnref) {
  timers = { intervals: [], timeouts: [] };
  global.setInterval = jest.fn((fn, ms) => {
    const h = { fn, ms };
    if (withUnref) h.unref = jest.fn();
    timers.intervals.push(h);
    return h;
  });
  global.setTimeout = jest.fn((fn, ms) => {
    const h = { fn, ms };
    if (withUnref) h.unref = jest.fn();
    timers.timeouts.push(h);
    return h;
  });
  global.clearInterval = jest.fn();
  global.clearTimeout = jest.fn();
}

const REAL_PLATFORM = Object.getOwnPropertyDescriptor(process, 'platform');
function setPlatform(p) {
  Object.defineProperty(process, 'platform', { value: p, configurable: true });
}

async function flush(rounds = 15) {
  for (let i = 0; i < rounds; i++) await new Promise((r) => setImmediate(r));
}

// Reset the registry, install fresh mocks/timers, and require main.js.
function loadMain(opts = {}) {
  jest.resetModules();
  installTimers(opts.unref !== false);
  setPlatform(opts.platform || 'linux');
  if (opts.resourcesPath) process.resourcesPath = opts.resourcesPath;
  else delete process.resourcesPath;

  const ctx = {};
  ctx.electron = require('electron');
  ctx.updater = require('electron-updater').autoUpdater;
  ctx.fs = require('fs');
  ctx.io = require('../src/main/io');
  ctx.probe = require('../src/main/probe');
  ctx.PearlEngine = require('../src/main/pearlEngine').PearlEngine;
  ctx.config = require('../src/shared/config');
  ctx.timers = timers;
  if (opts.isPackaged) ctx.electron.app.isPackaged = true;
  if (opts.before) opts.before(ctx);
  require('../src/main/main');
  ctx.invoke = (ch, ...a) => ctx.electron._handlers[ch]({}, ...a);
  ctx.emit = (ch, ...a) => ctx.electron._listeners[ch]({}, ...a);
  ctx.win = () => ctx.electron._windows[0];
  ctx.sent = (ch) => ctx.electron._windows
    .flatMap((w) => w.webContents.send.mock.calls)
    .filter((c) => c[0] === ch)
    .map((c) => c[1]);
  ctx.interval = (ms) => ctx.timers.intervals.find((h) => h.ms === ms);
  ctx.timeout = (ms) => ctx.timers.timeouts.find((h) => h.ms === ms);
  return ctx;
}

async function boot(opts) {
  const ctx = loadMain(opts);
  ctx.electron._fireReady();
  await flush();
  return ctx;
}

let errorSpy;
beforeEach(() => {
  errorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
});
afterEach(() => {
  errorSpy.mockRestore();
  Object.assign(global, REAL_TIMERS);
  delete process.resourcesPath;
  Object.defineProperty(process, 'platform', REAL_PLATFORM);
});

// ── boot / window lifecycle ──────────────────────────────────────────────────

describe('app boot and window lifecycle', () => {
  it('creates the window and refreshes economics on ready', async () => {
    const ctx = await boot();
    expect(ctx.electron.BrowserWindow).toHaveBeenCalledTimes(1);
    expect(ctx.win().loadFile).toHaveBeenCalledWith(expect.stringContaining('index.html'));
    // economics refreshed from the three prlscan endpoints
    expect(ctx.io.getJson).toHaveBeenCalledWith(ctx.config.ECON_API.price);
    expect(ctx.io.getJson).toHaveBeenCalledWith(ctx.config.ECON_API.metrics);
    expect(ctx.io.getJson).toHaveBeenCalledWith(ctx.config.ECON_API.blocks);
    // econ refresh interval registered and unref'd
    const econ = ctx.interval(10 * 60 * 1000);
    expect(econ).toBeTruthy();
    expect(econ.unref).toHaveBeenCalled();
  });

  it('arms the economics refresh on a runtime whose handles have no unref', async () => {
    const ctx = await boot({ unref: false });
    expect(ctx.interval(10 * 60 * 1000)).toBeTruthy();
  });

  it('fits the window to content and shows it on did-finish-load', async () => {
    const ctx = await boot();
    const w = ctx.win();
    w._wcEvents['did-finish-load']();
    await flush();
    expect(w.setContentSize).toHaveBeenCalledWith(620, 500);
    expect(w.show).toHaveBeenCalled();
  });

  it('does not show a window that was destroyed while measuring', async () => {
    const ctx = await boot();
    const w = ctx.win();
    w.webContents.executeJavaScript.mockRejectedValueOnce(new Error('gone'));
    w.isDestroyed.mockReturnValueOnce(false).mockReturnValue(true);
    w._wcEvents['did-finish-load']();
    await flush();
    expect(w.show).not.toHaveBeenCalled();
  });

  it('shows the window from the 1500ms fallback only while it is hidden', async () => {
    const ctx = await boot();
    const w = ctx.win();
    ctx.timeout(1500).fn();
    expect(w.show).toHaveBeenCalledTimes(1);
    w.isVisible.mockReturnValue(true);
    ctx.timeout(1500).fn();
    expect(w.show).toHaveBeenCalledTimes(1);
    w.isVisible.mockReturnValue(false);
    w.isDestroyed.mockReturnValue(true);
    ctx.timeout(1500).fn();
    expect(w.show).toHaveBeenCalledTimes(1);
  });

  it('builds a context menu for editable targets and selections only', async () => {
    const ctx = await boot();
    const w = ctx.win();
    const menuHandler = w._wcEvents['context-menu'];
    const Menu = ctx.electron.Menu;

    menuHandler({}, { isEditable: true, selectionText: '', editFlags: { canCut: true, canCopy: true, canPaste: true } });
    expect(Menu.buildFromTemplate).toHaveBeenCalledTimes(1);
    expect(Menu._menu.popup).toHaveBeenCalledWith({ window: w });
    const items = Menu.buildFromTemplate.mock.calls[0][0];
    expect(items.map((i) => i.role)).toEqual(['cut', 'copy', 'paste', undefined, 'selectAll']);

    menuHandler({}, { isEditable: false, selectionText: 'copy me', editFlags: {} });
    expect(Menu.buildFromTemplate).toHaveBeenCalledTimes(2);

    menuHandler({}, { isEditable: false, selectionText: '', editFlags: {} });
    expect(Menu.buildFromTemplate).toHaveBeenCalledTimes(2);

    // destroyed window: items built but never popped up
    w.isDestroyed.mockReturnValue(true);
    menuHandler({}, { isEditable: true, selectionText: '', editFlags: {} });
    expect(Menu._menu.popup).toHaveBeenCalledTimes(2);
  });

  it('recreates a window on activate only when none exist', async () => {
    const ctx = await boot();
    const activate = ctx.electron._appEvents['activate'];
    ctx.electron.BrowserWindow.getAllWindows.mockReturnValueOnce([]);
    activate();
    expect(ctx.electron.BrowserWindow).toHaveBeenCalledTimes(2);
    ctx.electron.BrowserWindow.getAllWindows.mockReturnValueOnce([{}]);
    activate();
    expect(ctx.electron.BrowserWindow).toHaveBeenCalledTimes(2);
  });

  it('window-all-closed stops everything and quits off macOS', () => {
    const ctx = loadMain();
    ctx.electron._appEvents['window-all-closed']();
    expect(ctx.electron.app.quit).toHaveBeenCalled();
  });

  it('window-all-closed does not quit on macOS', () => {
    const ctx = loadMain({ platform: 'darwin' });
    ctx.electron._appEvents['window-all-closed']();
    expect(ctx.electron.app.quit).not.toHaveBeenCalled();
  });

  // The leak this closes: Electron does NOT emit 'window-all-closed' when the
  // quit was started programmatically, which is exactly what the menu's Quit role
  // and Ctrl/Cmd+Q do. Without a before-quit hook the most ordinary way to close
  // the app skipped the only cleanup path and left the miner running — the
  // user's GPU stayed pinned by work they could no longer see.
  // 15s, not jest's default 5s: this one loads main.js, boots it and drains 15
  // rounds of the microtask queue, and on a degraded Windows runner that ran
  // past 5s and failed. It did exactly that during the v0.3.15 publish — the
  // Windows job died here, so the installer and latest.yml never uploaded and
  // the release shipped Mac + Linux only. Nothing here is slow by design, so
  // the headroom costs nothing on a healthy runner.
  it('before-quit stops the miner even when window-all-closed never fires', async () => {
    const ctx = await boot();
    ctx.emit('miner:start', { address: VALID_ADDR });
    await flush();
    const miner = ctx.PearlEngine.instances[ctx.PearlEngine.instances.length - 1];

    expect(ctx.electron._appEvents['before-quit']).toBeInstanceOf(Function);
    ctx.electron._appEvents['before-quit']();

    expect(miner.stop).toHaveBeenCalled();
  }, 15000);

  it('before-quit is safe to run with nothing started', () => {
    const ctx = loadMain();
    expect(() => ctx.electron._appEvents['before-quit']()).not.toThrow();
  });
});

// ── simple ipc handlers ──────────────────────────────────────────────────────

describe('simple ipc handlers', () => {
  it('settings:get merges saved settings over the desktop defaults', async () => {
    const ctx = loadMain();
    const s = await ctx.invoke('settings:get');
    // worker defaults to this machine's hostname, not the shared 'rig01' constant,
    // so two rigs on one payout address don't collide into one board identity.
    expect(s).toEqual({ region: 'us', worker: defaultWorker(), address: '', mdlAddress: '' });
    expect(s.worker).toMatch(/^[a-z0-9-]{1,32}$/);

    ctx.fs.existsSync.mockImplementation((p) => p === SETTINGS_PATH);
    ctx.fs.readFileSync.mockReturnValue('{"address":"prl1x","worker":"rig7"}');
    const s2 = await ctx.invoke('settings:get');
    expect(s2.address).toBe('prl1x');
    expect(s2.worker).toBe('rig7');
  });

  it('settings:get logs and falls back to defaults on a corrupt settings file', async () => {
    const ctx = loadMain();
    ctx.fs.existsSync.mockImplementation((p) => p === SETTINGS_PATH);
    ctx.fs.readFileSync.mockReturnValue('not json at all');
    const s = await ctx.invoke('settings:get');
    expect(s.address).toBe('');
    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining('Could not read settings'));
  });

  it('config:get, app:version and region:detect answer directly', async () => {
    const ctx = loadMain();
    expect(await ctx.invoke('config:get')).toEqual({
      regions: ctx.config.REGIONS, defaults: ctx.config.DEFAULTS, miner: ctx.config.MINER,
      platform: { minerSupported: true },
    });
    expect(await ctx.invoke('app:version')).toBe('0.0.0-test');
    expect(await ctx.invoke('region:detect')).toBe('us1');
    expect(ctx.probe.detectRegion).toHaveBeenCalled();
  });

  // Detection itself now lives in probe (and is tested there); main only unwraps
  // it to the plain name string the renderer's IPC contract expects. What matters
  // here is that it is NOT platform-gated any more — the old copy short-circuited
  // to null on anything but win32, which is why the Linux AppImage showed no
  // device name at all.
  it('gpu:detect returns the probed card name, on every platform', async () => {
    const ctx = loadMain({ platform: 'linux' });
    ctx.probe.detectGpuInfo.mockResolvedValue({ name: 'NVIDIA GeForce RTX 4090', count: 2 });
    expect(await ctx.invoke('gpu:detect')).toBe('NVIDIA GeForce RTX 4090');

    ctx.probe.detectGpuInfo.mockResolvedValue(null);
    expect(await ctx.invoke('gpu:detect')).toBeNull();
  });

  it('clipboard:write coerces null to an empty string', () => {
    const ctx = loadMain();
    ctx.emit('clipboard:write', null);
    expect(ctx.electron.clipboard.writeText).toHaveBeenCalledWith('');
    ctx.emit('clipboard:write', 'copied');
    expect(ctx.electron.clipboard.writeText).toHaveBeenCalledWith('copied');
  });

  it('open-external only opens http(s) URLs and swallows failures', async () => {
    const ctx = loadMain();
    const open = ctx.electron.shell.openExternal;
    ctx.emit('open-external', 'https://llmjob.com/x');
    expect(open).toHaveBeenCalledWith('https://llmjob.com/x');
    ctx.emit('open-external', 'http://llmjob.com/y');
    expect(open).toHaveBeenCalledTimes(2);
    ctx.emit('open-external', 'file:///etc/passwd');
    ctx.emit('open-external', 'not a url');
    expect(open).toHaveBeenCalledTimes(2);
    open.mockRejectedValueOnce(new Error('no browser'));
    ctx.emit('open-external', 'https://llmjob.com/z');
    await flush();
    expect(open).toHaveBeenCalledTimes(3);
  });

  // The other half of the resize loop. setContentSize round-trips the width back
  // through getContentSize, and on a DPI-scaled display that is not lossless —
  // so re-applying a size that was already correct still nudged the frame and
  // triggered the next fit. A fit that would change nothing must touch nothing.
  it('app:fit leaves the window alone when it already fits', async () => {
    const ctx = await boot();
    const w = ctx.win(); // getContentSize() → [620, 650]
    w.setContentSize.mockClear();

    w.webContents.executeJavaScript.mockResolvedValueOnce(650); // exactly right
    ctx.emit('app:fit');
    await flush();
    w.webContents.executeJavaScript.mockResolvedValueOnce(649); // within tolerance
    ctx.emit('app:fit');
    await flush();
    w.webContents.executeJavaScript.mockResolvedValueOnce(652);
    ctx.emit('app:fit');
    await flush();
    expect(w.setContentSize).not.toHaveBeenCalled();

    // …but a real mismatch still resizes, keeping the frame on the content.
    w.webContents.executeJavaScript.mockResolvedValueOnce(500);
    ctx.emit('app:fit');
    await flush();
    expect(w.setContentSize).toHaveBeenCalledWith(620, 500);
  });

  it('app:fit is a no-op before the window exists and measures it after', async () => {
    const ctx = loadMain();
    ctx.emit('app:fit'); // no window yet — early return
    ctx.electron._fireReady();
    await flush();
    const w = ctx.win();

    ctx.emit('app:fit');
    await flush();
    expect(w.setContentSize).toHaveBeenCalledWith(620, 500);

    // non-finite / non-positive heights are ignored
    w.setContentSize.mockClear();
    w.webContents.executeJavaScript.mockResolvedValueOnce(NaN);
    ctx.emit('app:fit');
    await flush();
    w.webContents.executeJavaScript.mockResolvedValueOnce(0);
    ctx.emit('app:fit');
    await flush();
    expect(w.setContentSize).not.toHaveBeenCalled();

    // measurement failure is swallowed
    w.webContents.executeJavaScript.mockRejectedValueOnce(new Error('nope'));
    ctx.emit('app:fit');
    await flush();

    // window destroyed between measure and resize
    w.isDestroyed.mockReturnValueOnce(false).mockReturnValueOnce(true);
    ctx.emit('app:fit');
    await flush();
    expect(w.setContentSize).not.toHaveBeenCalled();

    // window destroyed outright — early return
    w.webContents.executeJavaScript.mockClear();
    w.isDestroyed.mockReturnValue(true);
    ctx.emit('app:fit');
    await flush();
    expect(w.webContents.executeJavaScript).not.toHaveBeenCalled();
  });
});

// ── balance handlers ─────────────────────────────────────────────────────────

describe('balance handlers', () => {
  it('balance:get returns the parsed balance with a USD figure', async () => {
    const ctx = loadMain();
    ctx.io.getJson.mockResolvedValueOnce({
      stats: { balance: String(5 * 1e8) },
      payments: [{ amount: 10 * 1e8 }],
    });
    const b = await ctx.invoke('balance:get', VALID_ADDR);
    expect(b).toEqual({ pending: 5, paid: 10, earned: 15, usd: 15 * ctx.config.ECON.PRL_USD });
    expect(ctx.io.getJson).toHaveBeenCalledWith(
      expect.stringContaining('/api/stats_address?address=' + VALID_ADDR));
  });

  it('balance:get is null for invalid addresses, fetch failures, empty and throwing payloads', async () => {
    const ctx = loadMain();
    expect(await ctx.invoke('balance:get', 'nope')).toBeNull();
    ctx.io.getJson.mockRejectedValueOnce(new Error('offline'));
    expect(await ctx.invoke('balance:get', VALID_ADDR)).toBeNull();
    ctx.io.getJson.mockResolvedValueOnce(null);
    expect(await ctx.invoke('balance:get', VALID_ADDR)).toBeNull();
    // a payload whose property access throws exercises the parse catch
    ctx.io.getJson.mockResolvedValueOnce({ get stats() { throw new Error('boom'); } });
    expect(await ctx.invoke('balance:get', VALID_ADDR)).toBeNull();
  });

  it('live economics feed the balance USD conversion', async () => {
    const items = Array.from({ length: 12 }, () => ({ estimated_hashrate_hps: 1e18, block_time_seconds: 120 }));
    const ctx = loadMain({
      before: (c) => {
        c.io.getJson.mockImplementation((url) => {
          if (url === c.config.ECON_API.price) return Promise.resolve({ price_usd: 0.5 });
          if (url === c.config.ECON_API.metrics) return Promise.resolve({ items });
          if (url === c.config.ECON_API.blocks) return Promise.resolve({ items: [{ reward_grains: 2489e8 }] });
          return Promise.resolve(null);
        });
      },
    });
    ctx.electron._fireReady();
    await flush();
    ctx.io.getJson.mockResolvedValueOnce({ stats: { balance: String(10 * 1e8) } });
    const b = await ctx.invoke('balance:get', VALID_ADDR);
    expect(b.usd).toBe(5);
  });
});

// ── updater ──────────────────────────────────────────────────────────────────

describe('updater', () => {
  it('walks the dev flow (checking → latest) for an unpackaged manual check', async () => {
    const ctx = await boot();
    ctx.emit('app:update:check');
    expect(ctx.sent('app:update')).toEqual([expect.objectContaining({ phase: 'checking' })]);
    ctx.timeout(700).fn();
    expect(ctx.sent('app:update')[1]).toMatchObject({ phase: 'latest' });
    expect(ctx.updater.checkForUpdates).not.toHaveBeenCalled();
  });

  it('wires autoUpdater in a packaged app and relays every phase', async () => {
    const ctx = await boot({ isPackaged: true });
    expect(ctx.updater.autoDownload).toBe(true);
    expect(ctx.updater.autoInstallOnAppQuit).toBe(true);
    expect(ctx.updater.checkForUpdates).toHaveBeenCalledTimes(1);
    const ev = ctx.updater._events;

    ev['checking-for-update']();
    ev['update-available']({ version: '9.9.9' });
    ev['update-not-available']();
    ev['download-progress']({ percent: 42 });
    ev['update-downloaded']({ version: '9.9.9' });
    ev['error'](new Error('feed broke'));
    ev['error']('plain string failure');

    const phases = ctx.sent('app:update').map((u) => u.phase);
    expect(phases).toEqual(['checking', 'available', 'none', 'progress', 'ready', 'error', 'error']);
    const logs = ctx.sent('miner:log').map((l) => l.line);
    expect(logs).toContain('update check failed: feed broke');
    expect(logs).toContain('update check failed: plain string failure');
  });

  // A rig that launched while GitHub's releases feed was down used to never
  // check again for the life of the process — it would sit on a broken build
  // until someone restarted it. Observed in the wild as a five-second 503
  // window on releases.atom.
  it('re-checks for updates on a timer, not just at startup', async () => {
    const ctx = await boot({ isPackaged: true });
    expect(ctx.updater.checkForUpdates).toHaveBeenCalledTimes(1);

    ctx.interval(ctx.config.NETWORK.updateCheckIntervalMs).fn();
    expect(ctx.updater.checkForUpdates).toHaveBeenCalledTimes(2);
  });

  it('and arms that timer on a runtime whose handles have no unref', async () => {
    const ctx = await boot({ isPackaged: true, unref: false });
    expect(ctx.interval(ctx.config.NETWORK.updateCheckIntervalMs)).toBeTruthy();
  });

  it('a packaged manual check reports "latest" when nothing is found', async () => {
    const ctx = await boot({ isPackaged: true });
    ctx.emit('app:update:check');
    await flush();
    ctx.updater._events['update-not-available']();
    const phases = ctx.sent('app:update').map((u) => u.phase);
    expect(phases).toEqual(['checking', 'latest']);
  });

  // electron-updater emits 'error' and then rethrows, so a failed check used to
  // be logged twice — once by the event handler, once by the promise catch —
  // and every network blip printed a duplicate pair in the user's log.
  it('logs a failed check once, not once per code path', async () => {
    const ctx = await boot({
      isPackaged: true,
      before: (c) => c.updater.checkForUpdates.mockImplementation(() => {
        // what the real updater does: emit, then reject with the same error
        const err = new Error('rate limited');
        if (c.updater._events.error) c.updater._events.error(err);
        return Promise.reject(err);
      }),
    });
    await flush();
    const failures = ctx.sent('miner:log').map((l) => l.line)
      .filter((l) => l.startsWith('update check failed:'));
    expect(failures).toEqual(['update check failed: rate limited']);

    ctx.emit('app:update:check');
    await flush();
    expect(ctx.sent('app:update').map((u) => u.phase))
      .toEqual(['error', 'checking', 'error']);
    // the failed manual check reset the flag: the next silent result is 'none'
    ctx.updater._events['update-not-available']();
    expect(ctx.sent('app:update').map((u) => u.phase).slice(-1)).toEqual(['none']);
  });

  it('app:update:install stops the miner and relaunches; failures are logged', async () => {
    const ctx = await boot({ isPackaged: true });
    ctx.emit('app:update:install');
    expect(ctx.updater.quitAndInstall).toHaveBeenCalledWith(true, true);
    // not mining → no resumeMining persisted
    expect(ctx.fs.writeFileSync).not.toHaveBeenCalled();

    ctx.updater.quitAndInstall.mockImplementationOnce(() => { throw new Error('locked'); });
    ctx.emit('app:update:install');
    expect(ctx.sent('miner:log').map((l) => l.line)).toContain('update install failed: locked');
  });
});

// ── macOS ────────────────────────────────────────────────────────────────────
// A Mac cannot mine: the Pearl core is CUDA and Macs have no NVIDIA GPU. So a
// start on macOS runs nothing — and has to say so, or the renderer's optimistic
// "running" state shows STOP for a session in which nothing is running.

describe('macOS', () => {
  it('tells the renderer it cannot mine, so the UI stops offering START', async () => {
    const ctx = loadMain({ platform: 'darwin' });
    expect(await ctx.invoke('config:get')).toMatchObject({ platform: { minerSupported: false } });
  });

  it('a start runs nothing, ends the session, and says why', async () => {
    const ctx = await boot({ platform: 'darwin' });
    ctx.emit('miner:start', { address: VALID_ADDR });
    await flush();

    expect(ctx.PearlEngine.instances).toHaveLength(0);
    expect(ctx.sent('miner:stopped')).toHaveLength(1);
    const note = ctx.sent('miner:log').find((l) => /mining is not available on macOS/.test(l.line));
    expect(note).toMatchObject({ level: 'warn' });
  });

  // Squirrel.Mac verifies the update bundle's signature against the running
  // app's, and this build carries only an ad-hoc one — so wiring the updater
  // would buy a periodic download that always ends in an error bar.
  it('never wires the auto-updater, even packaged', async () => {
    const ctx = await boot({ platform: 'darwin', isPackaged: true });
    expect(ctx.updater.checkForUpdates).not.toHaveBeenCalled();
    expect(ctx.updater.on).not.toHaveBeenCalled();
    expect(ctx.interval(ctx.config.NETWORK.updateCheckIntervalMs)).toBeUndefined();
  });

  it('"check for updates" opens the Releases page instead of running a check that cannot install', async () => {
    const ctx = await boot({ platform: 'darwin', isPackaged: true });
    ctx.emit('app:update:check');
    await flush();

    expect(ctx.sent('app:update')).toEqual([expect.objectContaining({ phase: 'manual' })]);
    expect(ctx.electron.shell.openExternal)
      .toHaveBeenCalledWith(ctx.config.NETWORK.releasesUrl);
    expect(ctx.updater.checkForUpdates).not.toHaveBeenCalled();
    expect(ctx.sent('miner:log').map((l) => l.line).join('\n'))
      .toContain('in-app updates are unavailable on macOS — opening ' + ctx.config.NETWORK.releasesUrl);
  });
});

// ── mining ───────────────────────────────────────────────────────────────────

describe('mining', () => {
  // The engine has no NVML reading of its own the way alpha-miner did, so main
  // hands it one. Asserting the option is present is not enough -- it has to
  // actually reach nvidia-smi, or the UI silently loses the temperature again.
  it('gives the engine a card-temperature reader wired to nvidia-smi', async () => {
    const ctx = await boot();
    ctx.probe.detectGpuTemps = jest.fn(() => Promise.resolve({ 0: 68 }));
    ctx.emit('miner:start', { address: VALID_ADDR });
    await flush();
    const miner = ctx.PearlEngine.instances[0];
    expect(typeof miner.opts.readTemps).toBe('function');
    await expect(miner.opts.readTemps()).resolves.toEqual({ 0: 68 });
    expect(ctx.probe.detectGpuTemps).toHaveBeenCalled();
  });

  // Every card mines, one core each, and the engine is told which cards.
  it('hands the engine every card nvidia-smi lists', async () => {
    const ctx = await boot();
    ctx.probe.detectMinerGpus.mockResolvedValue([
      { index: 0, name: 'NVIDIA RTX PRO 4500 Blackwell' },
      { index: 1, name: 'NVIDIA GeForce RTX 4070' },
    ]);
    ctx.emit('miner:start', { address: VALID_ADDR });
    await flush();
    expect(ctx.PearlEngine.instances[0].start).toHaveBeenCalledWith(expect.objectContaining({
      gpus: [
        { index: 0, name: 'NVIDIA RTX PRO 4500 Blackwell' },
        { index: 1, name: 'NVIDIA GeForce RTX 4070' },
      ],
    }));
  });

  // CUDA_VISIBLE_DEVICES=0 hid a rig's second card from the mining core while
  // nvidia-smi still listed it. main.js removes it at load and says so on start.
  it('clears CUDA_VISIBLE_DEVICES at load and logs it when mining starts', async () => {
    const had = Object.prototype.hasOwnProperty.call(process.env, 'CUDA_VISIBLE_DEVICES');
    const before = process.env.CUDA_VISIBLE_DEVICES;
    process.env.CUDA_VISIBLE_DEVICES = '0';
    try {
      const ctx = await boot();
      expect(process.env.CUDA_VISIBLE_DEVICES).toBeUndefined();
      ctx.emit('miner:start', { address: VALID_ADDR });
      await flush();
      expect(ctx.sent('miner:log').map((l) => l.line)).toContain(
        'ignoring CUDA_VISIBLE_DEVICES=0 so every GPU can mine (set PEARL_GPU_INDEX to mine on one card)');
    } finally {
      if (had) process.env.CUDA_VISIBLE_DEVICES = before;
      else delete process.env.CUDA_VISIBLE_DEVICES;
    }
  });

  it('says nothing about CUDA_VISIBLE_DEVICES when it was not set', async () => {
    const ctx = await boot();
    ctx.emit('miner:start', { address: VALID_ADDR });
    await flush();
    expect(ctx.sent('miner:log').map((l) => l.line).join('\n')).not.toContain('CUDA_VISIBLE_DEVICES');
  });

  // Reading the card list is an await, so STOP can land inside a start again.
  // Starting a miner the user has already stopped leaves an engine nobody is
  // holding: the UI shows stopped and the cards keep mining.
  it('abandons a start that STOP overtakes while the cards are being read', async () => {
    const ctx = await boot();
    let release;
    ctx.probe.detectMinerGpus.mockReturnValue(new Promise((r) => { release = r; }));
    ctx.emit('miner:start', { address: VALID_ADDR });
    await flush();
    ctx.emit('miner:stop');
    release([{ index: 0, name: 'RTX 4090' }]);
    await flush();
    expect(ctx.PearlEngine.instances).toHaveLength(0);
  });

  // The device label names every card that is mining. One name for two working
  // cards is the same lie issue #226 was about, one level up.
  it('labels the rig with every card that is mining', async () => {
    const ctx = await boot();
    ctx.emit('miner:start', { address: VALID_ADDR });
    await flush();
    const miner = ctx.PearlEngine.instances[0];
    miner.emit('event', { type: 'status', gpuIndex: 0, hashrate: 100, gpu: 'RTX PRO 4500' });
    miner.emit('event', { type: 'status', gpuIndex: 1, hashrate: 40, gpu: 'RTX 4070' });
    ctx.interval(1000).fn();   // the stats ticker, which is what sends them
    await flush();
    const stats = ctx.sent('miner:stats');
    expect(stats[stats.length - 1].gpu).toBe('RTX PRO 4500 + RTX 4070');
  });

  // A rig whose resolver is broken looks exactly like a pool that is down: the
  // engine reprints one opaque line every 5s. Say the useful thing — which host,
  // and that it is name resolution — ONCE, so the hint does not become the same
  // spam it exists to explain.
  it('explains a DNS failure once, naming the endpoint, and not for other failures', async () => {
    const ctx = await boot();
    ctx.emit('miner:start', { address: VALID_ADDR, region: 'us1' });
    await flush();
    const miner = ctx.PearlEngine.instances[0];

    const dnsLine = '[stratum] connect failed: DNS lookup failed: No such host is known.';
    for (let i = 0; i < 3; i++) {
      miner.emit('event', { type: 'connect-failed', reason: dnsLine, dns: true });
    }
    await flush();

    const hints = ctx.sent('miner:log').map((l) => l.line)
      .filter((l) => l.includes('could not resolve'));
    expect(hints).toHaveLength(1);
    expect(hints[0]).toContain('could not resolve us.pearl.herominers.com:1200');

    // A refused connection is the pool's problem — do not blame the resolver.
    miner.emit('event', { type: 'connect-failed', reason: 'connection refused', dns: false });
    await flush();
    expect(ctx.sent('miner:log').map((l) => l.line)
      .filter((l) => l.includes('could not resolve'))).toHaveLength(1);
  });

  it('an invalid address runs nothing and tells the renderer', async () => {
    const ctx = await boot();
    ctx.emit('miner:start', { address: 'garbage' });
    await flush();
    expect(ctx.PearlEngine.instances).toHaveLength(0);
    expect(ctx.fs.writeFileSync).toHaveBeenCalled(); // still persisted
    expect(ctx.sent('miner:stopped').length).toBe(1);
  });

  // A start awaits the card probe before the miner exists, so without a
  // single-flight guard two quick clicks each pass the "already mining" check
  // and put two engines on the same GPU.
  it('coalesces a burst of miner:start events into a single run', async () => {
    const ctx = await boot();
    let release;
    ctx.probe.detectMinerGpus.mockReturnValue(new Promise((r) => { release = r; }));
    for (let i = 0; i < 5; i++) ctx.emit('miner:start', { address: VALID_ADDR });
    await flush();
    release([{ index: 0, name: 'RTX 4090' }]);
    await flush();
    expect(ctx.PearlEngine.instances).toHaveLength(1);

    // Once that run has settled, a later start is a fresh run again — which
    // finds the miner already going and keeps it rather than adding one.
    ctx.emit('miner:start', { address: VALID_ADDR });
    await flush();
    expect(ctx.PearlEngine.instances).toHaveLength(1);
  });

  it('miner:start with no payload has no address, so it runs nothing and ends the session', async () => {
    const ctx = await boot();
    ctx.emit('miner:start');
    await flush();
    expect(ctx.PearlEngine.instances).toHaveLength(0);
    expect(ctx.sent('miner:stopped')).toHaveLength(1);
  });

  it('reports each card to the network board while mining, with the app version', async () => {
    const ctx = await boot();
    ctx.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'NVIDIA GeForce RTX 4090', usedMb: 3000, totalMb: 24564 }]);
    ctx.emit('miner:start', { address: VALID_ADDR, worker: 'rig9' });
    await flush();
    const rows = ctx.probe.postMinerReport.mock.calls.map((c) => c[0]);
    expect(rows[0]).toMatchObject({ address: VALID_ADDR, worker: 'rig9', version: '0.0.0-test', vramTotalMb: 24564 });

    // …and again on the report interval, until STOP clears it.
    ctx.interval(ctx.config.NETWORK.reportIntervalMs).fn();
    await flush();
    expect(ctx.probe.postMinerReport.mock.calls.length).toBeGreaterThan(rows.length);
  });

  // The engine can stop on its own — a fatal pool error, a core that died —
  // without anyone pressing STOP, which leaves the stats ticker and board
  // reporter armed. The next START must replace them, not stack a second pair.
  it('relays engine logs and exits, and a restart after the engine died re-arms the timers once', async () => {
    const ctx = await boot({ platform: 'win32' });
    expect(ctx.electron.BrowserWindow.mock.calls[0][0].icon).toMatch(/icon\.ico$/);
    ctx.emit('miner:start', { address: VALID_ADDR });
    await flush();
    const first = ctx.PearlEngine.instances[0];
    first.emit('log', { level: 'info', line: 'job received' });
    first._running = false;
    first.emit('stopped', 3);
    const lines = ctx.sent('miner:log').map((l) => l.line);
    expect(lines).toContain('job received');
    expect(lines).toContain('engine exited (code 3)');

    global.clearInterval.mockClear();
    ctx.emit('miner:start', { address: VALID_ADDR });
    await flush();
    expect(ctx.PearlEngine.instances).toHaveLength(2);
    // the stale ticker and reporter from the first run are both cleared
    expect(global.clearInterval).toHaveBeenCalledTimes(2);
  });

  // An update that lands mid-session must bring the rig back up mining after
  // the relaunch, not leave it sitting at START.
  it('an update installed while mining remembers to resume', async () => {
    const ctx = await boot({ isPackaged: true });
    ctx.emit('miner:start', { address: VALID_ADDR });
    await flush();
    ctx.emit('app:update:install');
    const written = ctx.fs.writeFileSync.mock.calls.map((c) => JSON.parse(c[1]));
    expect(written.pop()).toMatchObject({ resumeMining: true });
    expect(ctx.PearlEngine.instances[0].stop).toHaveBeenCalled();
    expect(ctx.updater.quitAndInstall).toHaveBeenCalledWith(true, true);
  });

  it('sends formatted accepted and rejected share counts to the renderer', async () => {
    const ctx = await boot();
    ctx.emit('miner:start', { address: VALID_ADDR });
    await flush();
    const miner = ctx.PearlEngine.instances[0];
    miner.emit('event', { type: 'share', accepted: true });
    miner.emit('event', { type: 'share', accepted: false });
    ctx.interval(1000).fn();
    const last = ctx.sent('miner:stats').pop();
    expect(last).toMatchObject({ rejected: expect.any(Number), rejectedLabel: expect.any(String) });
  });
});

describe('the mining engine', () => {
  // There is one engine now. alpha-miner and AlphaPool are gone: the miner is
  // this process, the GPU work is a linked N-API addon, and the pool is
  // HeroMiners. So there is no binary to download, no version to select, no
  // driver floor to clear and nothing to spawn.
  it('starts our own core, with no binary to resolve first', async () => {
    const ctx = await boot();
    ctx.PearlEngine.instances.length = 0;
    ctx.emit('miner:start', { address: VALID_ADDR });
    await flush();
    expect(ctx.PearlEngine.instances).toHaveLength(1);
    const e = ctx.PearlEngine.instances[0];
    expect(e.start).toHaveBeenCalled();
  });

  // main.js resolves the endpoint from the region, exactly as before; the
  // engine is not asked to do it again.
  it('hands the resolved HeroMiners endpoint to the engine', async () => {
    const ctx = await boot();
    ctx.PearlEngine.instances.length = 0;
    ctx.emit('miner:start', { address: VALID_ADDR });
    await flush();
    expect(ctx.PearlEngine.instances[0].settings.endpoint)
      .toBe('us.pearl.herominers.com:1200');
  });

  // net.connect takes the PORT first and PearlMiner hands over host first, so
  // the adapter flips them. Backwards, this fails to connect with an error that
  // names neither side.
  it('the injected connect passes host and port to net in the right order', async () => {
    const ctx = await boot();
    ctx.PearlEngine.instances.length = 0;
    ctx.emit('miner:start', { address: VALID_ADDR });
    await flush();
    // AFTER boot: it resets the module registry, so a handle taken earlier
    // belongs to a mock instance main.js never saw.
    const net = require('net');
    net.connect.mockClear();
    ctx.PearlEngine.instances[0].opts.connect('pool.example', 1200);
    expect(net.connect).toHaveBeenCalledWith(1200, 'pool.example');
  });

  // Loading the addon is the one thing here that runs before the try/catch
  // around start(): a pearl_core.node that is present but throws on load would
  // otherwise take the whole main process with it.
  it('an addon that throws while loading is reported, not fatal', async () => {
    const ctx = await boot();
    const pearlCore = require('../src/main/pearlCore');
    pearlCore.coreFactory.mockImplementationOnce(() => { throw new Error('bad addon'); });
    ctx.emit('miner:start', { address: VALID_ADDR });
    await flush();
    expect(ctx.sent('miner:log').some((l) => /start failed: bad addon/.test(l.line))).toBe(true);
  });

  // An engine error with no message would otherwise read "could not start
  // mining: undefined".
  it('an error carrying no message still says something useful', async () => {
    const ctx = await boot();
    ctx.PearlEngine.instances.length = 0;
    ctx.emit('miner:start', { address: VALID_ADDR });
    await flush();
    ctx.PearlEngine.instances[0].emit('error', {});
    expect(ctx.sent('miner:log').some((l) => /the Pearl core did not initialise/.test(l.line)))
      .toBe(true);
  });

  // A throwing start is reported like any other launch failure rather than
  // taking the main process down.
  it('a failed start is reported, not thrown', async () => {
    const ctx = await boot();
    ctx.PearlEngine.instances.length = 0;
    ctx.PearlEngine.startError = new Error('no core');
    try {
      ctx.emit('miner:start', { address: VALID_ADDR });
      await flush();
      expect(ctx.PearlEngine.instances).toHaveLength(1);
      expect(ctx.sent('miner:log').some((l) => /could not start mining/.test(l.line))).toBe(true);
    } finally {
      ctx.PearlEngine.startError = null;
    }
  });
});
