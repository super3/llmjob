'use strict';

// Unit tests for the Electron main process (src/main/main.js). Everything with
// side effects is mocked — Electron, the updater, child_process, fs, http(s),
// and the local manager/IO/probe/nodeStore modules — while the pure ../shared
// modules run for real. Each scenario re-requires main.js under fresh mocks
// (jest.resetModules) so the module-global state (win, miner, llm, llmStatus)
// starts clean, then drives the captured app/ipc/updater/manager callbacks.

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

jest.mock('child_process', () => ({ spawn: jest.fn(), execFile: jest.fn() }));

jest.mock('fs', () => ({
  existsSync: jest.fn(() => false),
  readFileSync: jest.fn(() => '{}'),
  writeFileSync: jest.fn(),
  copyFileSync: jest.fn(),
  chmodSync: jest.fn(),
  mkdirSync: jest.fn(),
}));

// Default: any health probe fails fast (connection error on next tick).
jest.mock('http', () => ({
  get: jest.fn(() => {
    const req = {
      on: (ev, fn) => { if (ev === 'error') process.nextTick(fn); return req; },
      setTimeout: () => req,
      destroy: () => {},
    };
    return req;
  }),
}));
jest.mock('https', () => ({ get: jest.fn() }));

jest.mock('../src/main/io', () => ({
  postJson: jest.fn(() => Promise.resolve({ status: 200, data: {} })),
  getJson: jest.fn(() => Promise.resolve(null)),
  downloadFile: jest.fn(() => Promise.resolve()),
  streamChatCompletion: jest.fn(() => ({ done: Promise.resolve(), cancel: jest.fn() })),
  extractLlamaZip: jest.fn(() => Promise.resolve('/tmp/llm/llama-server')),
}));

jest.mock('../src/main/probe', () => ({
  detectRegion: jest.fn(() => Promise.resolve('us1')),
  detectVram: jest.fn(() => Promise.resolve(null)),
  detectGpusVram: jest.fn(() => Promise.resolve([])),
  detectDriverMajor: jest.fn(() => Promise.resolve(600)),
  postMinerReport: jest.fn(() => Promise.resolve()),
  findFreePort: jest.fn(() => Promise.resolve(8080)),
}));

jest.mock('../src/main/nodeStore', () => ({
  nodePath: jest.fn(() => '/tmp/store/node.json'),
  loadNode: jest.fn(() => null),
  saveNode: jest.fn(),
  migrateFrom: jest.fn(),
  getOrCreateNode: jest.fn(),
}));

jest.mock('../src/main/minerManager', () => {
  const { EventEmitter } = require('events');
  class MinerManager extends EventEmitter {
    constructor(opts) {
      super();
      this.opts = opts;
      this._running = false;
      this.start = jest.fn(() => {
        if (MinerManager.startError) throw MinerManager.startError;
        this._running = true;
      });
      this.stop = jest.fn(() => { this._running = false; });
      this.isRunning = jest.fn(() => this._running);
      MinerManager.instances.push(this);
    }
  }
  MinerManager.instances = [];
  MinerManager.startError = null;
  return { MinerManager };
});

jest.mock('../src/main/engineManager', () => {
  class EngineManager {
    constructor(opts) {
      this.opts = opts;
      EngineManager.instances.push(this);
    }
    isInstalled() { return EngineManager.behavior.installed; }
    binaryPath() { return EngineManager.behavior.binPath; }
    ensure() { return EngineManager.behavior.ensure(); }
  }
  EngineManager.instances = [];
  EngineManager.behavior = {
    installed: false,
    binPath: '/tmp/engine/alpha-miner',
    ensure: () => Promise.resolve('/tmp/engine/alpha-miner'),
  };
  return { EngineManager };
});

jest.mock('../src/main/llmManager', () => {
  const { EventEmitter } = require('events');
  class LlmManager extends EventEmitter {
    constructor(opts) {
      super();
      this.opts = opts;
      this.baseUrl = null;
      this._running = false;
      this.start = jest.fn((o) => {
        this._running = true;
        this.baseUrl = 'http://127.0.0.1:' + (o && o.port);
      });
      this.stop = jest.fn(() => { this._running = false; });
      this.isRunning = jest.fn(() => this._running);
      LlmManager.instances.push(this);
    }
  }
  LlmManager.instances = [];
  return { LlmManager };
});

jest.mock('../src/main/llmEngineManager', () => {
  class LlmEngineManager {
    constructor(opts) {
      this.opts = opts;
      LlmEngineManager.instances.push(this);
    }
    ensureServer() { return LlmEngineManager.behavior.ensureServer(); }
    ensureModel() { return LlmEngineManager.behavior.ensureModel(); }
  }
  LlmEngineManager.instances = [];
  LlmEngineManager.behavior = {
    ensureServer: () => Promise.resolve('/tmp/llm/llama-server'),
    ensureModel: () => Promise.resolve('/tmp/llm/model.gguf'),
  };
  return { LlmEngineManager };
});

jest.mock('../src/main/jobWorker', () => {
  const { EventEmitter } = require('events');
  class JobWorker extends EventEmitter {
    constructor(opts) {
      super();
      this.opts = opts;
      this.start = jest.fn();
      this.stop = jest.fn();
      this.activeJobs = jest.fn(() => 1);
      JobWorker.instances.push(this);
    }
  }
  JobWorker.instances = [];
  return { JobWorker };
});

const { EventEmitter } = require('events');
const nodeProto = require('../src/shared/node');

const KEYS = nodeProto.generateKeypair();
const VALID_ADDR = 'prl1p' + 'a'.repeat(30);

// main.js derives these from app.getPath('userData') with path.join, which
// yields backslashes on Windows — build the expectations the same way so the
// suite passes on every OS (same lesson as the nodeStore test).
const path = require('path');
const SETTINGS_PATH = path.join('/tmp/userData', 'settings.json');
const NODE_MIGRATE_PATH = path.join('/tmp/userData', 'node.json');

function fakeNode(extra) {
  return Object.assign({
    nodeId: 'abc123',
    publicKey: KEYS.publicKey,
    secretKey: KEYS.secretKey,
    name: null,
    connected: false,
  }, extra);
}

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
  ctx.cp = require('child_process');
  ctx.fs = require('fs');
  ctx.http = require('http');
  ctx.io = require('../src/main/io');
  ctx.probe = require('../src/main/probe');
  ctx.nodeStore = require('../src/main/nodeStore');
  ctx.MinerManager = require('../src/main/minerManager').MinerManager;
  ctx.EngineManager = require('../src/main/engineManager').EngineManager;
  ctx.LlmManager = require('../src/main/llmManager').LlmManager;
  ctx.LlmEngineManager = require('../src/main/llmEngineManager').LlmEngineManager;
  ctx.JobWorker = require('../src/main/jobWorker').JobWorker;
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

// Wire http.get so the LLM health probe gets a response built by `respond`.
function wireHealth(ctx, respond) {
  ctx.http.get.mockImplementation((u, cb) => {
    const req = new EventEmitter();
    req.setTimeout = jest.fn((_ms, fn) => { req._onTimeout = fn; return req; });
    req.destroy = jest.fn();
    process.nextTick(() => respond(cb, req));
    return req;
  });
}
function healthRes(statusCode) {
  const res = new EventEmitter();
  res.statusCode = statusCode;
  res.resume = jest.fn();
  res.setEncoding = jest.fn();
  res.destroy = jest.fn();
  return res;
}
function wireHealthOk(ctx) {
  wireHealth(ctx, (cb) => {
    const res = healthRes(200);
    cb(res);
    res.emit('data', '{"status":"ok"}');
    res.emit('end');
  });
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
  it('creates the window, refreshes economics, and migrates the node store on ready', async () => {
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
    expect(ctx.nodeStore.migrateFrom).toHaveBeenCalledWith(NODE_MIGRATE_PATH);
    // node not connected → no pinger
    expect(ctx.interval(ctx.config.NODE.pingIntervalMs)).toBeUndefined();
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

  it('starts the node pinger on boot when the machine is already linked (timers without unref)', async () => {
    const ctx = await boot({
      unref: false,
      before: (c) => {
        c.nodeStore.loadNode.mockReturnValue(fakeNode({ connected: true, name: 'rig' }));
      },
    });
    expect(ctx.interval(ctx.config.NODE.pingIntervalMs)).toBeTruthy();
    expect(ctx.io.postJson).toHaveBeenCalledWith(
      ctx.config.NODE.serverUrl + '/api/nodes/ping', expect.any(Object), 15000);
  });

  it('a GPU probe that blows up leaves the ping device null', async () => {
    const ctx = await boot({
      platform: 'win32',
      before: (c) => {
        c.nodeStore.loadNode.mockReturnValue(fakeNode({ connected: true, name: 'rig' }));
        c.cp.execFile.mockImplementation(() => { throw new Error('powershell missing'); });
      },
    });
    await flush();
    const pingBody = ctx.io.postJson.mock.calls.pop()[1];
    expect(pingBody).toMatchObject({ device: null, nodeId: 'abc123' });
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
});

// ── simple ipc handlers ──────────────────────────────────────────────────────

describe('simple ipc handlers', () => {
  it('settings:get merges saved settings over the desktop defaults', async () => {
    const ctx = loadMain();
    const s = await ctx.invoke('settings:get');
    expect(s).toMatchObject({ region: 'us2', worker: 'rig01', mode: 'auto', address: '', mdlAddress: '' });

    ctx.fs.existsSync.mockImplementation((p) => p === SETTINGS_PATH);
    ctx.fs.readFileSync.mockReturnValue('{"address":"prl1x","mode":"llm"}');
    const s2 = await ctx.invoke('settings:get');
    expect(s2.address).toBe('prl1x');
    expect(s2.mode).toBe('llm');
  });

  it('settings:get logs and falls back to defaults on a corrupt settings file', async () => {
    const ctx = loadMain();
    ctx.fs.existsSync.mockImplementation((p) => p === SETTINGS_PATH);
    ctx.fs.readFileSync.mockReturnValue('not json at all');
    const s = await ctx.invoke('settings:get');
    expect(s.mode).toBe('auto');
    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining('Could not read settings'));
  });

  it('config:get, llm:status, app:version, difficultyForCard and region:detect answer directly', async () => {
    const ctx = loadMain();
    expect(await ctx.invoke('config:get')).toEqual({
      regions: ctx.config.REGIONS, defaults: ctx.config.DEFAULTS, miner: ctx.config.MINER,
    });
    expect(await ctx.invoke('llm:status')).toMatchObject({ ready: false, model: ctx.config.LLM.model.name });
    expect(await ctx.invoke('app:version')).toBe('0.0.0-test');
    expect(await ctx.invoke('miner:difficultyForCard', 'RTX 4090')).toBe(524288);
    expect(await ctx.invoke('region:detect')).toBe('us1');
    expect(ctx.probe.detectRegion).toHaveBeenCalled();
  });

  it('gpu:detect resolves null off Windows without probing', async () => {
    const ctx = loadMain();
    expect(await ctx.invoke('gpu:detect')).toBeNull();
    expect(ctx.cp.execFile).not.toHaveBeenCalled();
  });

  it('gpu:detect parses the PowerShell output on Windows and nulls on error', async () => {
    const ctx = loadMain({ platform: 'win32' });
    ctx.cp.execFile.mockImplementation((...args) => args[args.length - 1](null, 'Intel(R) UHD Graphics\r\nNVIDIA GeForce RTX 4090\r\n'));
    expect(await ctx.invoke('gpu:detect')).toBe('NVIDIA GeForce RTX 4090');
    expect(ctx.cp.execFile).toHaveBeenCalledWith('powershell.exe', expect.any(Array), { timeout: 5000 }, expect.any(Function));
    ctx.cp.execFile.mockImplementation((...args) => args[args.length - 1](new Error('no powershell')));
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
    ctx.io.getJson.mockResolvedValueOnce({ balance_prl: 5, total_paid_prl: 10 });
    const b = await ctx.invoke('balance:get', VALID_ADDR);
    expect(b).toEqual({ pending: 5, paid: 10, earned: 15, usd: 15 * ctx.config.ECON.PRL_USD });
    expect(ctx.io.getJson).toHaveBeenCalledWith(expect.stringContaining('/api/miner/' + VALID_ADDR));
  });

  it('balance:get is null for invalid addresses, fetch failures, empty and throwing payloads', async () => {
    const ctx = loadMain();
    expect(await ctx.invoke('balance:get', 'nope')).toBeNull();
    ctx.io.getJson.mockRejectedValueOnce(new Error('offline'));
    expect(await ctx.invoke('balance:get', VALID_ADDR)).toBeNull();
    ctx.io.getJson.mockResolvedValueOnce(null);
    expect(await ctx.invoke('balance:get', VALID_ADDR)).toBeNull();
    // a payload whose property access throws exercises the parse catch
    ctx.io.getJson.mockResolvedValueOnce({ get balance_prl() { throw new Error('boom'); } });
    expect(await ctx.invoke('balance:get', VALID_ADDR)).toBeNull();
  });

  it('balance:getMdl uses the merge-mining route and parser', async () => {
    const ctx = loadMain();
    ctx.io.getJson.mockResolvedValueOnce({
      has_mdl: true, mdl_address: 'mdl1pxyz', summary: { pending_mdl: 2, total_paid_mdl: 3 },
    });
    const b = await ctx.invoke('balance:getMdl', VALID_ADDR);
    expect(b).toEqual({ pending: 2, paid: 3, earned: 5, usd: null, mdlAddress: 'mdl1pxyz' });
    expect(ctx.io.getJson).toHaveBeenCalledWith(expect.stringContaining('/mdl'));
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
    ctx.io.getJson.mockResolvedValueOnce({ balance_prl: 10, total_paid_prl: 0 });
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
    expect(logs).toContain('update error: feed broke');
    expect(logs).toContain('update error: plain string failure');
  });

  it('a packaged manual check reports "latest" when nothing is found', async () => {
    const ctx = await boot({ isPackaged: true });
    ctx.emit('app:update:check');
    await flush();
    ctx.updater._events['update-not-available']();
    const phases = ctx.sent('app:update').map((u) => u.phase);
    expect(phases).toEqual(['checking', 'latest']);
  });

  it('logs failures of both the startup and manual update checks', async () => {
    const ctx = await boot({
      isPackaged: true,
      before: (c) => c.updater.checkForUpdates.mockRejectedValue(new Error('rate limited')),
    });
    expect(ctx.sent('miner:log').map((l) => l.line))
      .toContain('update check failed: rate limited');

    ctx.emit('app:update:check');
    await flush();
    expect(ctx.sent('app:update').map((u) => u.phase)).toEqual(['checking', 'error']);
    // the failed manual check reset the flag: the next silent result is 'none'
    ctx.updater._events['update-not-available']();
    expect(ctx.sent('app:update').map((u) => u.phase)).toEqual(['checking', 'error', 'none']);
  });

  it('app:update:install stops engines and relaunches; failures are logged', async () => {
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

// ── mining ───────────────────────────────────────────────────────────────────

describe('mining', () => {
  it('downloads the engine, starts the miner, and relays engine events', async () => {
    const ctx = await boot();
    const BIN = '/tmp/engine/alpha-miner';
    ctx.fs.existsSync.mockImplementation((p) => p === BIN);

    ctx.emit('miner:start', { address: VALID_ADDR, mode: 'mining' });
    await flush();

    // settings persisted, initial stats pushed, ticker + reporter registered
    expect(ctx.fs.writeFileSync).toHaveBeenCalledWith(SETTINGS_PATH, expect.any(String));
    expect(ctx.sent('miner:stats').length).toBeGreaterThan(0);
    const ticker = ctx.interval(1000);
    expect(ticker).toBeTruthy();
    ticker.fn();
    const stats = ctx.sent('miner:stats');
    expect(stats[stats.length - 1]).toMatchObject({ accepted: 0, rejected: 0, points: [] });

    // reporter posted immediately and again on its interval
    expect(ctx.probe.postMinerReport).toHaveBeenCalled();
    const reporter = ctx.interval(ctx.config.NETWORK.reportIntervalMs);
    reporter.fn();
    await flush();
    expect(ctx.probe.detectGpusVram.mock.calls.length).toBeGreaterThanOrEqual(2);

    const logs = () => ctx.sent('miner:log').map((l) => l.line);
    expect(logs()).toContain('connecting to us2.alphapool.tech:5566 · worker rig01');
    expect(ctx.sent('miner:engine').map((e) => e.phase)).toEqual(['downloading', 'ready']);
    expect(logs()).toContain('engine ready: ' + BIN);

    const miner = ctx.MinerManager.instances[0];
    expect(miner.start).toHaveBeenCalledWith(expect.objectContaining({
      address: VALID_ADDR, platform: 'linux', binaryPath: BIN,
    }));

    // engine events flow through to the renderer + the stats accumulator
    miner.emit('log', { level: 'info', line: 'share accepted' });
    expect(logs()).toContain('share accepted');
    miner.emit('event', { type: 'status', hashrate: 12, accepted: 4 });
    expect(ctx.sent('miner:event')).toContainEqual({ type: 'status', hashrate: 12, accepted: 4 });
    ticker.fn();
    const after = ctx.sent('miner:stats');
    expect(after[after.length - 1].accepted).toBe(4);
    miner.emit('stopped', 3);
    expect(logs()).toContain('engine exited (code 3)');
    miner.emit('error', new Error('spawn EACCES'));
    expect(logs()).toContain('failed to launch engine: spawn EACCES');

    // while mining, an update install remembers to resume
    ctx.emit('app:update:install');
    const persisted = ctx.fs.writeFileSync.mock.calls.map((c) => c[1]).join('\n');
    expect(persisted).toContain('"resumeMining": true');

    // stop clears the timers and the miner
    ctx.emit('miner:stop');
    expect(global.clearInterval).toHaveBeenCalledWith(ticker);
    expect(global.clearInterval).toHaveBeenCalledWith(reporter);
    expect(miner.stop).toHaveBeenCalled();
    expect(ctx.sent('miner:stopped').length).toBeGreaterThan(0);
  });

  it('uses a custom endpoint/worker/binary and keeps the running miner on re-start', async () => {
    const ctx = await boot();
    ctx.fs.existsSync.mockImplementation((p) => p === '/custom/bin');
    ctx.emit('miner:start', {
      address: VALID_ADDR, mode: 'mining', endpoint: 'pool.example:1234', worker: 'w9',
      region: 'eu1', binaryPath: '/custom/bin',
    });
    await flush();
    expect(ctx.sent('miner:log').map((l) => l.line))
      .toContain('connecting to pool.example:1234 · worker w9');
    expect(ctx.EngineManager.instances).toHaveLength(0);
    expect(ctx.MinerManager.instances).toHaveLength(1);

    // second start while running: no new manager, settings persisted again
    ctx.emit('miner:start', { address: VALID_ADDR, mode: 'mining', binaryPath: '/custom/bin' });
    await flush();
    expect(ctx.MinerManager.instances).toHaveLength(1);
  });

  it('reports a friendly failure when the engine download fails', async () => {
    const ctx = await boot({
      before: (c) => { c.EngineManager.behavior.ensure = () => Promise.reject(new Error('404')); },
    });
    ctx.emit('miner:start', { address: VALID_ADDR, mode: 'mining' });
    await flush();
    expect(ctx.sent('miner:engine')).toContainEqual({
      phase: 'error', message: 'Could not download or set up the mining engine — see Logs.',
    });
    expect(ctx.sent('miner:log').map((l) => l.line).join('\n')).toContain('engine setup failed: 404');
    expect(ctx.MinerManager.instances[0].start).not.toHaveBeenCalled();

    // a retry while the failed session's ticker/reporter are still alive
    // replaces them instead of stacking a second pair
    const ticker = ctx.interval(1000);
    const reporter = ctx.interval(ctx.config.NETWORK.reportIntervalMs);
    ctx.emit('miner:start', { address: VALID_ADDR, mode: 'mining' });
    await flush();
    expect(global.clearInterval).toHaveBeenCalledWith(ticker);
    expect(global.clearInterval).toHaveBeenCalledWith(reporter);
  });

  it('logs "engine found" when the engine is already installed', async () => {
    const ctx = await boot({
      before: (c) => { c.EngineManager.behavior.installed = true; },
    });
    ctx.fs.existsSync.mockImplementation((p) => p === '/tmp/engine/alpha-miner');
    ctx.emit('miner:start', { address: VALID_ADDR, mode: 'mining' });
    await flush();
    expect(ctx.sent('miner:log').map((l) => l.line)).toContain('engine found: /tmp/engine/alpha-miner');
    expect(ctx.sent('miner:engine').map((e) => e.phase)).toEqual(['ready']);
  });

  it('flags an antivirus quarantine when the Windows engine vanishes after setup', async () => {
    const ctx = await boot({ platform: 'win32' });
    // ensure() resolves but the file never exists on disk
    ctx.emit('miner:start', { address: VALID_ADDR, mode: 'mining' });
    await flush();
    const engineMsgs = ctx.sent('miner:engine');
    expect(engineMsgs[engineMsgs.length - 1].message).toContain('Antivirus blocked the mining engine');
    expect(ctx.MinerManager.instances[0].start).not.toHaveBeenCalled();
  });

  it('reports a launch failure when spawning the engine throws', async () => {
    const ctx = await boot({
      before: (c) => { c.MinerManager.startError = new Error('bad exe'); },
    });
    ctx.fs.existsSync.mockImplementation((p) => p === '/tmp/engine/alpha-miner');
    ctx.emit('miner:start', { address: VALID_ADDR, mode: 'mining' });
    await flush();
    expect(ctx.sent('miner:log').map((l) => l.line)).toContain('failed to launch engine: bad exe');
  });

  it('prefers the versioned bundled engine and falls back to the legacy Windows name', async () => {
    const legacy = require('path').join('/res', 'engine', 'alpha-miner-windows.exe');
    const ctx = await boot({ platform: 'win32', resourcesPath: '/res' });
    ctx.fs.existsSync.mockImplementation((p) => p === legacy);
    ctx.emit('miner:start', { address: VALID_ADDR, mode: 'mining' });
    await flush();
    expect(ctx.sent('miner:log').map((l) => l.line)).toContain('using bundled engine: ' + legacy);
    expect(ctx.EngineManager.instances).toHaveLength(0);
    expect(ctx.MinerManager.instances[0].start)
      .toHaveBeenCalledWith(expect.objectContaining({ binaryPath: legacy }));
  });

  it('falls through to the download when neither Windows bundle exists', async () => {
    const ctx = await boot({ platform: 'win32', resourcesPath: '/res' });
    ctx.fs.existsSync.mockImplementation((p) => p === '/tmp/engine/alpha-miner');
    ctx.emit('miner:start', { address: VALID_ADDR, mode: 'mining' });
    await flush();
    expect(ctx.EngineManager.instances).toHaveLength(1);
    // Windows pins the single pool build (ENGINE.windows)
    expect(ctx.EngineManager.instances[0].opts.version).toBe(require('../src/shared/engine').ENGINE.windows);
  });

  it('logs a start failure when the driver probe throws mid-start', async () => {
    const ctx = await boot({
      before: (c) => { c.probe.detectDriverMajor.mockRejectedValue(new Error('nvidia-smi missing')); },
    });
    ctx.emit('miner:start', { address: VALID_ADDR, mode: 'mining' });
    await flush();
    expect(ctx.sent('miner:log').map((l) => l.line)).toContain('start failed: nvidia-smi missing');
  });

  it('an invalid address in mining mode runs nothing and tells the renderer', async () => {
    const ctx = await boot();
    ctx.emit('miner:start', { address: 'garbage', mode: 'mining' });
    await flush();
    expect(ctx.MinerManager.instances).toHaveLength(0);
    expect(ctx.fs.writeFileSync).toHaveBeenCalled(); // still persisted
    expect(ctx.sent('miner:stopped').length).toBe(1);
  });

  it('miner:start with no payload defaults to mining mode with no address → nothing runs', async () => {
    const ctx = await boot();
    ctx.emit('miner:start');
    await flush();
    expect(ctx.MinerManager.instances).toHaveLength(0);
    expect(ctx.LlmManager.instances).toHaveLength(0);
    expect(ctx.sent('miner:stopped')).toHaveLength(1);
  });
});

// ── zip extraction helpers (passed into the engine managers) ─────────────────

describe('zip extraction helpers', () => {
  async function minerExtract() {
    const ctx = await boot();
    ctx.emit('miner:start', { address: VALID_ADDR, mode: 'mining' });
    await flush();
    return { ctx, extract: ctx.EngineManager.instances[0].opts.extract };
  }

  it('extractZip resolves the destination via PowerShell and quotes quotes', async () => {
    const { ctx, extract } = await minerExtract();
    ctx.cp.execFile.mockImplementation((...args) => args[args.length - 1](null));
    await expect(extract("/tmp/o'brien.zip", '/tmp/dest.exe')).resolves.toBe('/tmp/dest.exe');
    const [bin, args] = ctx.cp.execFile.mock.calls.pop();
    expect(bin).toBe('powershell.exe');
    expect(args[args.length - 1]).toContain("'/tmp/o''brien.zip'");
  });

  it('extractZip rejects when PowerShell fails', async () => {
    const { ctx, extract } = await minerExtract();
    ctx.cp.execFile.mockImplementation((...args) => args[args.length - 1](new Error('expand failed')));
    await expect(extract('/tmp/a.zip', '/tmp/dest.exe')).rejects.toThrow('expand failed');
  });

  async function llamaWinExtract() {
    // win32 + no resources → the llama server manager gets extractLlamaZipWin
    const ctx = await boot({ platform: 'win32' });
    wireHealth(ctx, (cb, req) => req.emit('error', new Error('down')));
    ctx.probe.detectVram.mockResolvedValue({ totalMb: 24000, usedMb: 1000 });
    ctx.emit('miner:start', { mode: 'llm' });
    await flush();
    return { ctx, extract: ctx.LlmEngineManager.instances[0].opts.extract };
  }

  it('extractLlamaZipWin flattens the zip and verifies llama-server exists', async () => {
    const { ctx, extract } = await llamaWinExtract();
    ctx.cp.execFile.mockImplementation((...args) => args[args.length - 1](null));
    ctx.fs.existsSync.mockImplementation((p) => p === '/tmp/llm/llama-server.exe');
    await expect(extract('/tmp/llm/l.zip', '/tmp/llm/llama-server.exe')).resolves.toBe('/tmp/llm/llama-server.exe');
    ctx.fs.existsSync.mockReturnValue(false);
    await expect(extract('/tmp/llm/l.zip', '/tmp/llm/llama-server.exe'))
      .rejects.toThrow('llama-server was not found in the downloaded archive');
    ctx.cp.execFile.mockImplementation((...args) => args[args.length - 1](new Error('ps broke')));
    await expect(extract('/tmp/llm/l.zip', '/tmp/llm/llama-server.exe')).rejects.toThrow('ps broke');
  });

  it('the non-Windows llama extractor delegates to io.extractLlamaZip', async () => {
    const ctx = await boot();
    ctx.probe.detectVram.mockResolvedValue({ totalMb: 24000, usedMb: 1000 });
    ctx.emit('miner:start', { mode: 'llm' });
    await flush();
    const serverEngine = ctx.LlmEngineManager.instances[0];
    await serverEngine.opts.extract('/tmp/z.tgz', '/tmp/llm/llama-server');
    expect(ctx.io.extractLlamaZip).toHaveBeenCalledWith('/tmp/z.tgz', '/tmp/llm/llama-server');
  });
});

// ── local LLM ────────────────────────────────────────────────────────────────

describe('local LLM', () => {
  it('adopts an already-healthy llama-server instead of spawning a second one', async () => {
    const ctx = await boot({
      before: (c) => {
        c.nodeStore.loadNode.mockReturnValue(fakeNode({ connected: true, serverUrl: 'https://custom.example' }));
        // the warm-up request fails — best-effort, must be swallowed
        c.io.streamChatCompletion.mockReturnValueOnce({ done: Promise.reject(new Error('warmup')), cancel: jest.fn() });
      },
    });
    wireHealthOk(ctx);
    ctx.emit('miner:start', { mode: 'llm' });
    await flush();

    expect(ctx.LlmManager.instances).toHaveLength(0);
    const status = ctx.sent('llm:status').pop();
    expect(status).toMatchObject({ ready: true, endpoint: 'http://127.0.0.1:8080/v1', webUrl: 'http://127.0.0.1:8080' });
    expect(ctx.sent('miner:log').map((l) => l.line).join('\n')).toContain('already running on http://127.0.0.1:8080 — reusing it');
    expect(ctx.sent('miner:stopped')).toHaveLength(0);

    // linked + ready → the cluster job worker starts against the node's server
    expect(ctx.JobWorker.instances).toHaveLength(1);
    expect(ctx.JobWorker.instances[0].opts.serverUrl).toBe('https://custom.example');
    expect(ctx.JobWorker.instances[0].start).toHaveBeenCalled();

    // warm-up asked for a tiny streamed generation and discards its deltas
    const [warmBase, warmBody, warmOnDelta] = ctx.io.streamChatCompletion.mock.calls[0];
    expect(warmBase).toBe('http://127.0.0.1:8080');
    expect(warmBody.max_tokens).toBe(24);
    warmOnDelta('discarded');
    expect(ctx.sent('llm:chat:delta')).toHaveLength(0);

    // a second START adopts again, and a sync throw from the warm-up is swallowed
    ctx.io.streamChatCompletion.mockImplementationOnce(() => { throw new Error('sync boom'); });
    ctx.emit('miner:start', { mode: 'llm' });
    await flush();
    expect(ctx.sent('miner:stopped')).toHaveLength(0);
  });

  it('refuses to start the LLM without enough free VRAM (llm-only ends the session)', async () => {
    const ctx = await boot();
    ctx.probe.detectVram.mockResolvedValue({ totalMb: 8000, usedMb: 4000 });
    ctx.emit('miner:start', { mode: 'llm' });
    await flush();
    expect(ctx.sent('miner:log').map((l) => l.line).join('\n')).toContain('not enough free VRAM for the local LLM: 4000 MB free');
    expect(ctx.sent('llm:status').pop()).toMatchObject({ ready: false, error: 'Needs ~6 GB free VRAM' });
    expect(ctx.sent('miner:stopped')).toHaveLength(1);
    expect(ctx.LlmManager.instances).toHaveLength(0);
  });

  it('a rejection out of startLlm itself is caught and ends an llm-only session', async () => {
    const ctx = await boot({
      before: (c) => { c.probe.detectVram.mockRejectedValue(new Error('probe exploded')); },
    });
    ctx.emit('miner:start', { mode: 'llm' });
    await flush();
    expect(ctx.LlmManager.instances).toHaveLength(0);
    expect(ctx.sent('miner:stopped')).toHaveLength(1);
  });

  ['ensureServer', 'ensureModel'].forEach((step) => {
    it(`ends an llm-only session when ${step} fails`, async () => {
      const ctx = await boot({
        before: (c) => { c.LlmEngineManager.behavior[step] = () => Promise.reject(new Error(step + ' failed')); },
      });
      ctx.emit('miner:start', { mode: 'llm' });
      await flush();
      expect(ctx.sent('miner:log').map((l) => l.line)).toContain('LLM setup failed: ' + step + ' failed');
      expect(ctx.sent('miner:stopped')).toHaveLength(1);
    });
  });

  it('starts llama-server, goes ready, serves jobs, streams stats, and reports its exit', async () => {
    const ctx = await boot({
      before: (c) => {
        c.nodeStore.loadNode.mockReturnValue(fakeNode({ connected: true, name: 'rig' }));
        c.nodeStore.getOrCreateNode.mockReturnValue(fakeNode({ connected: true, name: 'rig' }));
        c.probe.detectVram.mockResolvedValue({ totalMb: 24000, usedMb: 2000 });
      },
    });
    ctx.emit('miner:start', { mode: 'llm' });
    await flush();

    const llm = ctx.LlmManager.instances[0];
    // free 22000 MB, no reserve → full offload
    expect(llm.start).toHaveBeenCalledWith({
      platform: 'linux', binaryPath: '/tmp/llm/llama-server', modelPath: '/tmp/llm/model.gguf',
      nGpuLayers: ctx.config.LLM.model.layers, port: 8080,
    });
    expect(ctx.sent('llm:status').pop()).toMatchObject({ ready: false, endpoint: 'http://127.0.0.1:8080/v1' });

    llm.emit('log', { level: 'info', line: 'llama says hi' });
    expect(ctx.sent('miner:log').map((l) => l.line)).toContain('llama says hi');

    llm.emit('ready', { baseUrl: llm.baseUrl });
    await flush();
    expect(ctx.sent('llm:status').pop()).toMatchObject({ ready: true, webUrl: 'http://127.0.0.1:8080' });

    // the job worker came up with the default server URL and full wiring
    const worker = ctx.JobWorker.instances[0];
    expect(worker.opts.serverUrl).toBe(ctx.config.NODE.serverUrl);
    worker.emit('error', new Error('poll blip'));
    worker.emit('job', { id: 'j1' });
    worker.emit('done', { id: 'j1' });
    worker.emit('failed', { id: 'j2', error: 'oom' });
    const logs = ctx.sent('miner:log').map((l) => l.line);
    expect(logs).toContain('cluster job j1 — running locally');
    expect(logs).toContain('cluster job j1 — done');
    expect(logs).toContain('cluster job j2 failed: oom');
    expect(logs).toContain('serving cluster jobs for the LLMJob network');

    worker.opts.post('https://x/api', { a: 1 });
    expect(ctx.io.postJson).toHaveBeenCalledWith('https://x/api', { a: 1 }, 30000);
    const onDelta = jest.fn();
    await worker.opts.runJob({ messages: [] }, { onDelta });
    expect(ctx.io.streamChatCompletion).toHaveBeenLastCalledWith('http://127.0.0.1:8080', { messages: [] }, onDelta);

    llm.emit('stats', { tokensPerSec: 33 });
    expect(ctx.sent('llm:status').pop()).toMatchObject({ tokensPerSec: 33 });

    // a linked ping while the worker runs reports its active jobs
    await ctx.invoke('node:connect', { token: 'tok' });
    await flush();
    expect(worker.activeJobs).toHaveBeenCalled();

    // unlinking makes the next worker sync stop it (linked AND ready no longer holds)
    ctx.nodeStore.loadNode.mockReturnValue(null);
    await ctx.invoke('node:connect', { token: 'tok' });
    await flush();
    expect(worker.stop).toHaveBeenCalled();
    ctx.nodeStore.loadNode.mockReturnValue(fakeNode({ connected: true, name: 'rig' }));

    // llama-server exits after ready in an llm-only session → session over
    llm.emit('stopped');
    await flush();
    expect(worker.stop).toHaveBeenCalled();
    expect(ctx.sent('miner:stopped')).toHaveLength(1);
    expect(ctx.sent('llm:status').pop()).toMatchObject({ ready: false, tokensPerSec: 0 });
  });

  it('llm mode with a second start returns early while the server runs, and STOP stops it', async () => {
    const ctx = await boot({
      before: (c) => { c.probe.detectVram.mockResolvedValue({ totalMb: 24000, usedMb: 2000 }); },
    });
    ctx.emit('miner:start', { mode: 'llm' });
    await flush();
    expect(ctx.LlmManager.instances).toHaveLength(1);
    ctx.emit('miner:start', { mode: 'llm' });
    await flush();
    expect(ctx.LlmManager.instances).toHaveLength(1); // early return, no second spawn
    ctx.emit('miner:stop');
    expect(ctx.LlmManager.instances[0].stop).toHaveBeenCalled();
    expect(ctx.sent('llm:status').pop()).toMatchObject({ ready: false });
  });

  it('co-runs mining and the LLM: waits for real hashrate, then flags a pre-ready LLM exit', async () => {
    const path = require('path');
    const bundledMiner = path.join('/res', 'engine', 'alpha-miner-1.8.8');
    const ctx = await boot({
      resourcesPath: '/res',
      before: (c) => { c.probe.findFreePort.mockResolvedValue(8081); },
    });
    ctx.fs.existsSync.mockImplementation((p) => p === bundledMiner);

    ctx.emit('miner:start', { address: VALID_ADDR, mode: 'both' });
    await flush();

    // miner is up (bundled engine), LLM is waiting for proof of hashrate
    expect(ctx.sent('miner:log').map((l) => l.line)).toContain('using bundled engine: ' + bundledMiner);
    const miner = ctx.MinerManager.instances[0];
    expect(miner.start).toHaveBeenCalled();
    expect(ctx.LlmManager.instances).toHaveLength(0);

    miner.emit('event', { type: 'log' });            // ignored — not a status
    miner.emit('event', { type: 'status', hashrate: 0 }); // ignored — no work yet
    await flush();
    expect(ctx.LlmManager.instances).toHaveLength(0);
    miner.emit('event', { type: 'status', hashrate: '2.5' });
    await flush(30);

    // the wait's cap timer was cleared; firing it late is a settled no-op
    const cap = ctx.timeout(60000);
    expect(global.clearTimeout).toHaveBeenCalledWith(cap);
    cap.fn();

    // VRAM unknown → full offload; port 8080 busy → moved to 8081
    const llm = ctx.LlmManager.instances[0];
    expect(llm.start).toHaveBeenCalledWith(expect.objectContaining({
      nGpuLayers: ctx.config.LLM.model.layers, port: 8081,
    }));
    expect(ctx.sent('miner:log').map((l) => l.line))
      .toContain('port 8080 is busy — using 8081 for the local LLM instead');

    // llama-server dies before ready while mining keeps running
    llm.emit('stopped');
    await flush();
    expect(ctx.sent('llm:status').pop()).toMatchObject({
      ready: false, error: 'The local LLM stopped before it was ready. See Logs.',
    });
    expect(ctx.sent('miner:stopped')).toHaveLength(0);
  });

  it('a miner that stops during the co-run wait releases the LLM start', async () => {
    const ctx = await boot();
    ctx.fs.existsSync.mockImplementation((p) => p === '/tmp/engine/alpha-miner');
    ctx.probe.detectVram.mockResolvedValue({ totalMb: 24000, usedMb: 2000 });
    ctx.emit('miner:start', { address: VALID_ADDR, mode: 'auto' });
    await flush();
    const miner = ctx.MinerManager.instances[0];
    expect(ctx.LlmManager.instances).toHaveLength(0);
    miner.emit('stopped', 1);
    await flush(30);
    expect(ctx.LlmManager.instances).toHaveLength(1);
  });

  it('falls back to the linux server binary name on unknown platforms', async () => {
    const ctx = await boot({ platform: 'freebsd' });
    ctx.emit('miner:start', { mode: 'llm' });
    await flush();
    expect(ctx.LlmManager.instances[0].start)
      .toHaveBeenCalledWith(expect.objectContaining({ platform: 'freebsd', binaryPath: '/tmp/llm/llama-server' }));
  });

  it('uses a bundled llama-server on Windows and installs the VC++ runtime DLLs beside it', async () => {
    const path = require('path');
    const bundledLlama = path.join('/res', 'llm', 'llama-server.exe');
    const dll = (name) => path.join('/res', 'llm-runtime', name);
    const ctx = await boot({ platform: 'win32', resourcesPath: '/res' });
    ctx.probe.detectVram.mockResolvedValue({ totalMb: 24000, usedMb: 2000 });
    // bundled exe + two of the three DLLs available in the bundle, none installed yet
    ctx.fs.existsSync.mockImplementation((p) =>
      p === bundledLlama || p === dll('msvcp140.dll') || p === dll('vcruntime140.dll'));
    // the first DLL copies fine; the second explodes → logged, start continues
    ctx.fs.copyFileSync
      .mockImplementationOnce(() => {})
      .mockImplementationOnce(() => { throw new Error('EPERM'); });

    ctx.emit('miner:start', { mode: 'llm' });
    await flush();

    const logs = ctx.sent('miner:log').map((l) => l.line);
    expect(logs).toContain('installed LLM runtime DLL: msvcp140.dll');
    expect(logs.join('\n')).toContain('could not install the LLM runtime DLLs: EPERM');
    expect(ctx.LlmManager.instances[0].start)
      .toHaveBeenCalledWith(expect.objectContaining({ binaryPath: bundledLlama }));
    // reserve 0 (llm-only), free 22000 → full offload despite the reserve arg branch
    expect(ctx.LlmManager.instances[0].start)
      .toHaveBeenCalledWith(expect.objectContaining({ nGpuLayers: ctx.config.LLM.model.layers }));
  });
});

// ── health probe variants ────────────────────────────────────────────────────

describe('llama-server health probe', () => {
  // Each case wires one degenerate response; the probe must resolve false so
  // startLlm proceeds to the (failing) VRAM gate — proof it wasn't adopted.
  const cases = [
    ['a non-200 status', (cb) => { const res = healthRes(503); cb(res); }],
    ['an unrelated 200 body', (cb) => {
      const res = healthRes(200);
      cb(res);
      res.emit('data', '{"status":"definitely-not-ok"}');
      res.emit('end');
    }],
    ['an oversized body', (cb) => {
      const res = healthRes(200);
      cb(res);
      res.emit('data', 'x'.repeat(5000));
      res.emit('end');
    }],
    ['a response stream error', (cb) => {
      const res = healthRes(200);
      cb(res);
      res.emit('error', new Error('reset'));
    }],
    ['a request timeout', (cb, req) => {
      req._onTimeout();
      req.emit('error', new Error('destroyed'));
    }],
  ];

  cases.forEach(([name, respond]) => {
    it(`treats ${name} as "no server running"`, async () => {
      const ctx = await boot();
      ctx.probe.detectVram.mockResolvedValue({ totalMb: 4000, usedMb: 0 });
      wireHealth(ctx, respond);
      ctx.emit('miner:start', { mode: 'llm' });
      await flush();
      // fell through to the VRAM gate → nothing adopted, session ended
      expect(ctx.sent('miner:stopped')).toHaveLength(1);
      expect(ctx.sent('miner:log').map((l) => l.line).join('\n')).not.toContain('reusing it');
    });
  });

  it('treats an unparseable health URL as "no server running"', async () => {
    const ctx = await boot({
      before: (c) => { c.config.LLM.host = 'bad host'; },
    });
    ctx.probe.detectVram.mockResolvedValue({ totalMb: 4000, usedMb: 0 });
    ctx.emit('miner:start', { mode: 'llm' });
    await flush();
    expect(ctx.http.get).not.toHaveBeenCalled();
    expect(ctx.sent('miner:stopped')).toHaveLength(1);
  });
});

// ── in-app chat ──────────────────────────────────────────────────────────────

describe('in-app chat', () => {
  function deferred() {
    let resolve, reject;
    const p = new Promise((a, b) => { resolve = a; reject = b; });
    return { p, resolve, reject };
  }

  it('reports an error when the LLM is not running', async () => {
    const ctx = loadMain();
    ctx.electron._fireReady();
    await flush();
    ctx.emit('llm:chat', [{ role: 'user', content: 'hi' }]);
    expect(ctx.sent('llm:chat:error')).toEqual([{ message: 'the local LLM is not running' }]);
  });

  it('streams grounded turns, supersedes stale ones, and cancels on LLM stop', async () => {
    const ctx = await boot({
      before: (c) => { c.probe.detectVram.mockResolvedValue({ totalMb: 24000, usedMb: 2000 }); },
    });
    ctx.emit('miner:start', { mode: 'llm' });
    await flush();
    const llm = ctx.LlmManager.instances[0];
    llm.emit('ready', { baseUrl: llm.baseUrl });
    await flush();
    ctx.io.streamChatCompletion.mockClear();

    // turn A: null messages still get the grounding system prompt
    const a = deferred();
    ctx.io.streamChatCompletion.mockReturnValueOnce({ done: a.p, cancel: jest.fn() });
    ctx.emit('llm:chat', null);
    const [baseA, bodyA, onDeltaA] = ctx.io.streamChatCompletion.mock.calls[0];
    expect(baseA).toBe('http://127.0.0.1:8080');
    expect(bodyA.messages).toHaveLength(1);
    expect(bodyA.messages[0].role).toBe('system');
    expect(bodyA.messages[0].content).toContain('LLMJob');
    onDeltaA('hel');
    onDeltaA('lo');
    expect(ctx.sent('llm:chat:delta')).toEqual([{ text: 'hel' }, { text: 'lo' }]);
    a.resolve();
    await flush();
    expect(ctx.sent('llm:chat:done')).toHaveLength(1);

    // turn B fails outright
    const b = deferred();
    ctx.io.streamChatCompletion.mockReturnValueOnce({ done: b.p, cancel: jest.fn() });
    ctx.emit('llm:chat', [{ role: 'user', content: 'q' }]);
    expect(ctx.io.streamChatCompletion.mock.calls[1][1].messages).toHaveLength(2);
    b.reject(new Error('model crashed'));
    await flush();
    expect(ctx.sent('llm:chat:error')).toEqual([{ message: 'model crashed' }]);

    // turn C is superseded by turn D; C's late rejection must not clear D
    const c = deferred();
    const cancelC = jest.fn();
    ctx.io.streamChatCompletion.mockReturnValueOnce({ done: c.p, cancel: cancelC });
    ctx.emit('llm:chat', [{ role: 'user', content: 'old' }]);
    const d = deferred();
    const cancelD = jest.fn();
    ctx.io.streamChatCompletion.mockReturnValueOnce({ done: d.p, cancel: cancelD });
    ctx.emit('llm:chat', [{ role: 'user', content: 'new' }]);
    expect(cancelC).toHaveBeenCalledWith('superseded by a new message');
    // C's late completion must not clear D's live stream
    c.resolve();
    await flush();
    expect(ctx.sent('llm:chat:done')).toHaveLength(2);

    // the LLM stopping cancels the in-flight turn D
    llm.emit('stopped');
    expect(cancelD).toHaveBeenCalledWith('the local LLM stopped');
    d.reject(new Error('stream closed'));
    await flush();
    expect(ctx.sent('llm:chat:error').map((e) => e.message)).toContain('stream closed');
  });
});

// ── node linking ─────────────────────────────────────────────────────────────

describe('node linking', () => {
  it('node:status is renderer-safe for missing, linked, and userless nodes', async () => {
    const ctx = loadMain();
    expect(await ctx.invoke('node:status')).toEqual({ connected: false, nodeId: null, name: null, user: null });
    ctx.nodeStore.loadNode.mockReturnValue(fakeNode({ connected: true, name: 'rig', user: 'alice' }));
    expect(await ctx.invoke('node:status')).toEqual({ connected: true, nodeId: 'abc123', name: 'rig', user: 'alice' });
    ctx.nodeStore.loadNode.mockReturnValue(fakeNode({ connected: false, name: 'rig' }));
    expect(await ctx.invoke('node:status')).toEqual({ connected: false, nodeId: 'abc123', name: 'rig', user: null });
  });

  it('node:connect rejects an empty token before any network call', async () => {
    const ctx = loadMain();
    expect(await ctx.invoke('node:connect', undefined)).toEqual({ error: 'Enter your pairing token first.' });
    expect(await ctx.invoke('node:connect', { token: '   ' })).toEqual({ error: 'Enter your pairing token first.' });
    expect(ctx.io.postJson).not.toHaveBeenCalled();
  });

  it('node:connect surfaces network failures and server rejections', async () => {
    const ctx = loadMain({
      before: (c) => { c.nodeStore.getOrCreateNode.mockReturnValue(fakeNode()); },
    });
    ctx.io.postJson.mockRejectedValueOnce(new Error('ECONNREFUSED'));
    expect(await ctx.invoke('node:connect', { token: 'tok' }))
      .toEqual({ error: 'Could not reach LLMJob — check your connection.' });

    ctx.io.postJson.mockResolvedValueOnce({ status: 401, data: { error: 'bad token' } });
    expect(await ctx.invoke('node:connect', { token: 'tok' })).toEqual({ error: 'bad token' });

    ctx.io.postJson.mockResolvedValueOnce({ status: 500, data: null });
    expect(await ctx.invoke('node:connect', { token: 'tok' })).toEqual({ error: 'Link failed (HTTP 500).' });
    expect(ctx.nodeStore.saveNode).not.toHaveBeenCalled();
  });

  it('node:connect links the node, starts pinging, and node:disconnect undoes it', async () => {
    const ctx = await boot({
      before: (c) => { c.nodeStore.getOrCreateNode.mockReturnValue(fakeNode()); },
    });
    ctx.io.postJson.mockResolvedValueOnce({ status: 201, data: { user: 'alice' } });
    const res = await ctx.invoke('node:connect', { token: ' tok ', name: '  myrig  ' });
    expect(res).toEqual({ success: true, nodeId: 'abc123', name: 'myrig', user: 'alice' });
    const joinBody = ctx.io.postJson.mock.calls[0][1];
    expect(joinBody).toMatchObject({ token: 'tok', nodeId: 'abc123', name: 'myrig', publicKey: KEYS.publicKey });
    expect(ctx.nodeStore.saveNode).toHaveBeenCalledWith(expect.objectContaining({
      connected: true, name: 'myrig', user: 'alice', linkedAt: expect.any(String),
    }));
    expect(ctx.sent('node:status').pop()).toMatchObject({ connected: false }); // loadNode mock still says unlinked

    // the immediate ping + interval
    ctx.nodeStore.loadNode.mockReturnValue(fakeNode({ connected: true, name: 'myrig' }));
    const pinger = ctx.interval(ctx.config.NODE.pingIntervalMs);
    expect(pinger).toBeTruthy();
    expect(pinger.unref).toHaveBeenCalled();
    ctx.probe.detectVram.mockResolvedValueOnce({ totalMb: 24000, usedMb: 2000 });
    pinger.fn();
    await flush();
    const pingBody = ctx.io.postJson.mock.calls.pop()[1];
    expect(pingBody).toMatchObject({ nodeId: 'abc123', vramTotal: 24000, vramUsed: 2000, name: 'myrig' });
    expect(pingBody.signature).toEqual(expect.any(String));

    // a ping survives probe failures and server unreachability
    ctx.probe.detectVram.mockRejectedValueOnce(new Error('no nvidia-smi'));
    ctx.io.postJson.mockRejectedValueOnce(new Error('offline'));
    pinger.fn();
    await flush();

    // an unlinked node makes the ping a silent no-op
    ctx.io.postJson.mockClear();
    ctx.nodeStore.loadNode.mockReturnValue(null);
    pinger.fn();
    await flush();
    expect(ctx.io.postJson).not.toHaveBeenCalled();

    // disconnect flips the stored flag and stops the pinger
    ctx.nodeStore.loadNode.mockReturnValue(fakeNode({ connected: true, name: 'myrig' }));
    expect(await ctx.invoke('node:disconnect')).toEqual({ ok: true });
    expect(ctx.nodeStore.saveNode).toHaveBeenCalledWith(expect.objectContaining({ connected: false }));
    expect(global.clearInterval).toHaveBeenCalledWith(pinger);
  });

  it('node:connect falls back to the stored name and node:disconnect tolerates a missing node', async () => {
    const ctx = loadMain({
      before: (c) => { c.nodeStore.getOrCreateNode.mockReturnValue(fakeNode({ name: 'stored' })); },
    });
    ctx.io.postJson.mockResolvedValueOnce({ status: 200, data: {} });
    const res = await ctx.invoke('node:connect', { token: 'tok' });
    expect(res).toEqual({ success: true, nodeId: 'abc123', name: 'stored', user: null });

    ctx.nodeStore.saveNode.mockClear();
    ctx.nodeStore.loadNode.mockReturnValue(null);
    expect(await ctx.invoke('node:disconnect')).toEqual({ ok: true });
    expect(ctx.nodeStore.saveNode).not.toHaveBeenCalled();
  });

  it('node:dashboard opens the dashboard URL', () => {
    const ctx = loadMain();
    ctx.emit('node:dashboard');
    expect(ctx.electron.shell.openExternal).toHaveBeenCalledWith(ctx.config.NODE.dashboardUrl);
  });

  it('a worker rename in Settings propagates to the linked node on start', async () => {
    const ctx = await boot({
      before: (c) => { c.nodeStore.loadNode.mockReturnValue(fakeNode({ connected: true, name: 'old' })); },
    });
    ctx.emit('miner:start', { address: 'bad', mode: 'mining', worker: 'renamed' });
    await flush();
    expect(ctx.nodeStore.saveNode).toHaveBeenCalledWith(expect.objectContaining({ name: 'renamed' }));
    expect(ctx.io.postJson).toHaveBeenCalledWith(
      ctx.config.NODE.serverUrl + '/api/nodes/ping', expect.any(Object), 15000);

    // unchanged name → no rewrite; missing worker → no rewrite; unlinked → no rewrite
    ctx.nodeStore.saveNode.mockClear();
    ctx.nodeStore.loadNode.mockReturnValue(fakeNode({ connected: true, name: 'renamed' }));
    ctx.emit('miner:start', { address: 'bad', mode: 'mining', worker: 'renamed' });
    await flush();
    ctx.emit('miner:start', { address: 'bad', mode: 'mining' });
    await flush();
    ctx.nodeStore.loadNode.mockReturnValue(fakeNode({ connected: false, name: 'other' }));
    ctx.emit('miner:start', { address: 'bad', mode: 'mining', worker: 'zzz' });
    await flush();
    expect(ctx.nodeStore.saveNode).not.toHaveBeenCalled();
  });
});
