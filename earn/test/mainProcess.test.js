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
  // Empty by default: an unknown compute capability is what keeps Windows on the
  // 1.8.6 fallback, which is the shape most of these tests were written against.
  // The tests that care about the 1.9.1b Windows package set it explicitly.
  detectComputeCaps: jest.fn(() => Promise.resolve([])),
  postMinerReport: jest.fn(() => Promise.resolve()),
  findFreePort: jest.fn(() => Promise.resolve(8080)),
  // GPU detection moved into probe so the GUI and the CLI share one
  // implementation — the GUI's own copy was Windows-only, which left the Linux
  // AppImage with no device name and no per-card difficulty.
  detectGpuInfo: jest.fn(() => Promise.resolve(null)),
}));

jest.mock('../src/main/nodeStore', () => ({
  nodePath: jest.fn(() => '/tmp/store/node.json'),
  loadNode: jest.fn(() => null),
  saveNode: jest.fn(),
  migrateFrom: jest.fn(),
  // Serving no longer requires an account, so the app mints an identity whenever
  // a model is up. Default to an unlinked one — tests that care about the linked
  // path override it (and fakeNode() builds the same shape).
  getOrCreateNode: jest.fn(() => ({
    nodeId: 'abc123', publicKey: 'pk-test', secretKey: 'sk-test', name: null, connected: false,
  })),
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
    ensureServer(onProgress) { return LlmEngineManager.behavior.ensureServer(onProgress); }
    ensureModel(onProgress) { return LlmEngineManager.behavior.ensureModel(onProgress); }
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
const { defaultWorker } = require('../src/shared/worker');

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

// The llama-server download is arch-aware on macOS (arm64 vs Intel), so the
// suite has to be able to pin process.arch the same way it pins the platform.
const REAL_ARCH = Object.getOwnPropertyDescriptor(process, 'arch');
function setArch(a) {
  Object.defineProperty(process, 'arch', { value: a, configurable: true });
}

async function flush(rounds = 15) {
  for (let i = 0; i < rounds; i++) await new Promise((r) => setImmediate(r));
}

// Reset the registry, install fresh mocks/timers, and require main.js.
function loadMain(opts = {}) {
  jest.resetModules();
  installTimers(opts.unref !== false);
  setPlatform(opts.platform || 'linux');
  if (opts.arch) setArch(opts.arch);
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
  Object.defineProperty(process, 'arch', REAL_ARCH);
});

// ── boot / window lifecycle ──────────────────────────────────────────────────

const { ALL_LAYERS } = require('../src/shared/vram');

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

  // The leak this closes: Electron does NOT emit 'window-all-closed' when the
  // quit was started programmatically, which is exactly what the menu's Quit role
  // and Ctrl/Cmd+Q do. Without a before-quit hook the most ordinary way to close
  // the app skipped the only cleanup path and left llama-server and the miner
  // running — the user's GPU stayed pinned and port 8080 stayed bound.
  // 15s, not jest's default 5s: this one loads main.js, boots it and drains 15
  // rounds of the microtask queue, and on a degraded Windows runner that ran
  // past 5s and failed. It did exactly that during the v0.3.15 publish — the
  // Windows job died here, so the installer and latest.yml never uploaded and
  // the release shipped Mac + Linux only. Nothing here is slow by design, so
  // the headroom costs nothing on a healthy runner.
  it('before-quit stops the miner even when window-all-closed never fires', async () => {
    const ctx = await boot();
    ctx.fs.existsSync.mockImplementation((p) => p === '/tmp/engine/alpha-miner');
    ctx.emit('miner:start', { address: VALID_ADDR, mode: 'mining' });
    await flush();
    const miner = ctx.MinerManager.instances[ctx.MinerManager.instances.length - 1];

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
    expect(s).toMatchObject({ region: 'us2', worker: defaultWorker(), mode: 'auto', address: '' });
    expect(s.worker).toMatch(/^[a-z0-9-]{1,32}$/);

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
      platform: { minerSupported: true },
    });
    expect(await ctx.invoke('llm:status')).toMatchObject({ ready: false, model: ctx.config.LLM.model.name });
    expect(await ctx.invoke('app:version')).toBe('0.0.0-test');
    expect(await ctx.invoke('miner:difficultyForCard', 'RTX 4090')).toBe(524288);
    expect(await ctx.invoke('region:detect')).toBe('us1');
    expect(ctx.probe.detectRegion).toHaveBeenCalled();
  });

  // Detection itself now lives in probe (and is tested there); main only unwraps
  // it to the plain name string the renderer's IPC contract expects. What matters
  // here is that it is NOT platform-gated any more — the old copy short-circuited
  // to null on anything but win32, which is why the Linux AppImage showed no
  // device name and never applied the per-card difficulty.
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
    // Atomic units on the wire (coinUnits 1e8), PRL out.
    ctx.io.getJson.mockResolvedValueOnce({ stats: { balance: 5 * 1e8, paid: 10 * 1e8 } });
    const b = await ctx.invoke('balance:get', VALID_ADDR);
    expect(b).toEqual({ pending: 5, paid: 10, earned: 15, usd: 15 * ctx.config.ECON.PRL_USD });
    expect(ctx.io.getJson).toHaveBeenCalledWith(expect.stringContaining('/api/stats_address?address=' + VALID_ADDR));
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
    ctx.io.getJson.mockResolvedValueOnce({ stats: { balance: 10 * 1e8, paid: 0 } });
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

// ── macOS ────────────────────────────────────────────────────────────────────
// The Mac build serves the local LLM and never mines: AlphaPool has no Darwin
// engine. The gate has to be explicit, because every non-Windows path in
// shared/engine.js resolves to the LINUX artifact — an ungated Mac would
// download an ELF binary, chmod it, and spawn something the kernel refuses.

describe('macOS', () => {
  it('tells the renderer it cannot mine, so the UI stops offering the mining modes', async () => {
    const ctx = loadMain({ platform: 'darwin' });
    expect(await ctx.invoke('config:get')).toMatchObject({ platform: { minerSupported: false } });
  });

  it('auto: serves the LLM, never resolves or spawns a mining engine, and says why', async () => {
    const ctx = await boot({ platform: 'darwin' });
    ctx.emit('miner:start', { mode: 'auto', address: VALID_ADDR });
    await flush();

    // No engine resolution at all — not a failed download, not a spawn.
    expect(ctx.EngineManager.instances).toHaveLength(0);
    expect(ctx.MinerManager.instances).toHaveLength(0);
    expect(ctx.io.downloadFile).not.toHaveBeenCalled();
    // …but the model does come up, which is the whole point of the Mac build.
    expect(ctx.LlmManager.instances).toHaveLength(1);

    const logs = ctx.sent('miner:log');
    const note = logs.find((l) => /mining is not available on macOS/.test(l.line));
    expect(note).toMatchObject({ level: 'warn' });
    expect(note.line).toMatch(/local LLM runs as usual/);
  });

  it('mining-only: runs nothing, ends the session, and points at the LLM mode', async () => {
    const ctx = await boot({ platform: 'darwin' });
    ctx.emit('miner:start', { mode: 'mining', address: VALID_ADDR });
    await flush();

    expect(ctx.EngineManager.instances).toHaveLength(0);
    expect(ctx.MinerManager.instances).toHaveLength(0);
    expect(ctx.LlmManager.instances).toHaveLength(0);
    // The renderer's optimistic "running" state must be undone, or it shows STOP
    // for a session in which nothing is running.
    expect(ctx.sent('miner:stopped')).toHaveLength(1);
    expect(ctx.sent('miner:log').map((l) => l.line).join('\n'))
      .toMatch(/Switch the compute mode to LLM/);
  });

  it('downloads the llama-server build matching the Mac architecture', async () => {
    for (const [arch, key] of [['arm64', 'darwin'], ['x64', 'darwin-x64']]) {
      const ctx = await boot({ platform: 'darwin', arch });
      ctx.emit('miner:start', { mode: 'llm' });
      await flush();
      expect(ctx.LlmEngineManager.instances[0].opts.serverUrl).toBe(ctx.config.LLM.serverUrl[key]);
    }
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
    expect(logs()).toContain('connecting to us2.pearl.herominers.com:1200 · worker rig01');
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

  it('does not spawn the miner when STOP arrives while the engine is still downloading', async () => {
    const ctx = await boot();
    // Make engine.ensure hang so the start is in flight when STOP lands.
    let resolveEnsure;
    ctx.EngineManager.behavior.installed = false;
    ctx.EngineManager.behavior.ensure = () => new Promise((res) => { resolveEnsure = res; });
    ctx.fs.existsSync.mockImplementation((p) => p === '/tmp/engine/alpha-miner');

    ctx.emit('miner:start', { address: VALID_ADDR, mode: 'mining' });
    await flush();
    expect(ctx.MinerManager.instances).toHaveLength(0); // still downloading, nothing spawned

    ctx.emit('miner:stop');          // user stops mid-download
    resolveEnsure('/tmp/engine/alpha-miner'); // download completes AFTER the stop
    await flush();

    // The stop wins: no miner is spawned behind a UI that says it's stopped.
    expect(ctx.MinerManager.instances).toHaveLength(0);
    expect(ctx.sent('miner:stopped').length).toBeGreaterThan(0);
  });

  it('does not start the LLM when STOP arrives during the miner hashrate wait', async () => {
    const ctx = await boot();
    const BIN = '/tmp/engine/alpha-miner';
    ctx.fs.existsSync.mockImplementation((p) => p === BIN);

    // mode 'both' → start the miner, then wait for a non-zero hashrate before the LLM.
    ctx.emit('miner:start', { address: VALID_ADDR, mode: 'both' });
    await flush();
    const miner = ctx.MinerManager.instances[0];
    expect(miner.start).toHaveBeenCalled(); // miner running, runPlan now awaiting waitForMinerUp

    ctx.emit('miner:stop'); // stop during the wait → miner torn down, epoch bumped
    await flush();

    // The LLM fleet must not come up for a stopped session.
    expect(ctx.LlmManager.instances).toHaveLength(0);
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

  // The reported failure: Node rejected the pool's certificate chain, and the
  // log's manual-download hint named a URL but no destination, so the user's
  // hand-downloaded engine sat in ~/Downloads and the next start failed the
  // same way.
  it('names the cause and the exact file to save when the certificate check fails', async () => {
    const ctx = await boot({
      before: (c) => {
        c.EngineManager.behavior.ensure = () => Promise.reject(Object.assign(
          new Error('unable to verify the first certificate'),
          { code: 'UNABLE_TO_VERIFY_LEAF_SIGNATURE' },
        ));
      },
    });
    ctx.emit('miner:start', { address: VALID_ADDR, mode: 'mining' });
    await flush();

    expect(ctx.sent('miner:engine')).toContainEqual({
      phase: 'error', message: 'The mining engine download failed an HTTPS certificate check — see Logs.',
    });
    const log = ctx.sent('miner:log').map((l) => l.line).join('\n');
    expect(log).toMatch(/proxy, VPN or antivirus/);
    expect(log).toContain('ca-certificates');
    // Version-agnostic on purpose: this asserts the guidance names a download URL
    // and where the file has to go, not which engine build is pinned today. A
    // The Linux engine is a bare binary, so the advice is "save it as"
    // — and it downloads under the very name the cache looks for, so there is no
    // rename to get wrong either. Telling anyone to extract it would be advice
    // that cannot work, which is the whole point of manualInstallHint.
    expect(log).toMatch(/Manual install: download \S+ and save it as \S+linux-x86_64,/);
    expect(log).not.toContain('extract it into');
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

  // A stale legacy bundle must NOT be resurrected. Windows now runs a packaged
  // 1.9.4, and the only thing that could still be sitting under the old
  // unversioned name is an 1.8.6 build — which mines rank-256 work the fork does
  // not credit. Taking it would look like success and pay nothing, so the rig
  // downloads the real engine instead.
  it('ignores a legacy bundled Windows exe and fetches the packaged engine', async () => {
    const legacy = require('path').join('/res', 'engine', 'alpha-miner-windows.exe');
    const ctx = await boot({ platform: 'win32', resourcesPath: '/res' });
    ctx.fs.existsSync.mockImplementation((p) => p === legacy);
    ctx.emit('miner:start', { address: VALID_ADDR, mode: 'mining' });
    await flush();
    expect(ctx.sent('miner:log').map((l) => l.line)).not.toContain('using bundled engine: ' + legacy);
    expect(ctx.EngineManager.instances).toHaveLength(1);
  });

  // The installer ships the engine too (see the Windows bundling step in
  // .github/workflows/miner-build.yml), so a fresh rig mines without waiting on
  // a download. Windows has no execute bit, so unlike the Linux bundle there is
  // no chmod to re-assert — calling one would be a no-op at best.
  it('spawns a bundled Windows engine as-is, with no chmod', async () => {
    const eng = require('../src/shared/engine');
    const bundled = eng.bundledEnginePath('/res', 'win32', undefined, eng.ENGINE.windows);
    const ctx = await boot({ platform: 'win32', resourcesPath: '/res' });
    ctx.fs.existsSync.mockImplementation((p) => p === bundled);
    ctx.emit('miner:start', { address: VALID_ADDR, mode: 'mining' });
    await flush();
    expect(ctx.EngineManager.instances).toHaveLength(0);
    expect(ctx.fs.chmodSync).not.toHaveBeenCalled();
    expect(ctx.MinerManager.instances[0].start)
      .toHaveBeenCalledWith(expect.objectContaining({ binaryPath: bundled }));
  });

  // The Linux AppImage now ships both engine builds (see the bundling step in
  // .github/workflows/miner-build.yml), and one spawned straight out of the
  // bundle has to be executable — so the mode is re-asserted rather than
  // trusted. Inside the read-only AppImage mount the chmod fails, and that must
  // not stop a rig whose binary squashfs already recorded as executable.
  it('re-asserts +x on a bundled Linux engine and survives a read-only bundle', async () => {
    const eng = require('../src/shared/engine');
    // Derived, not spelled out: a packaged engine lives at <dir>/<launcher>, not a
    // flat versioned filename, and bundledEnginePath is what knows the difference.
    const bundled = eng.bundledEnginePath('/res', 'linux', undefined, eng.ENGINE.linux);
    const ctx = await boot({ resourcesPath: '/res' });
    ctx.fs.existsSync.mockImplementation((p) => p === bundled);
    ctx.fs.chmodSync.mockImplementation(() => { throw new Error('EROFS: read-only file system'); });

    ctx.emit('miner:start', { address: VALID_ADDR, mode: 'mining' });
    await flush();

    expect(ctx.fs.chmodSync).toHaveBeenCalledWith(bundled, 0o755);
    expect(ctx.sent('miner:log').map((l) => l.line)).toContain('using bundled engine: ' + bundled);
    expect(ctx.MinerManager.instances[0].start)
      .toHaveBeenCalledWith(expect.objectContaining({ binaryPath: bundled }));
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

  // A Windows rig that QUALIFIES for the rank-128 hotfix must not be quietly
  // handed the bundled 1.8.6 .exe instead. The legacy-name fallback exists for
  // when the selected version has no bundle, and taking it here would downgrade
  // a compliant rig back to mining work the fork no longer credits — the exact
  // failure this release exists to fix.
  // Upstream's launcher mangles its own core path when that path contains a
  // space, and the cache is "%APPDATA%\LLMJob Earn\engine" — always spaced.
  // Rather than give up the compliant build, put the engine somewhere
  // space-free so a qualifying rig keeps it.
  // …but a locked-down box where nothing space-free can be created still has to
  // mine. A rig that cannot start earns nothing; one on 1.8.6 earns uncredited
  // work, which is worse than compliant and better than dead.
  // The same bug, reached the other way: a space-free cache keeps the hotfix
  // selected, but the bundle sitting in "C:\Program Files\LLMJob Earn" still
  // cannot run. Spawning it would fail on every start, so it is skipped and the
  // rig downloads into the path that works.
  // PeakMiner publishes no minimum driver — it embeds its own CUDA runtime and
  // picks a kernel profile by compute capability — so ENGINE.minDriverMajor is
  // null and this warning fires for nobody. Asserted as silence rather than
  // deleted, so that configuring a real floor one day makes this test fail
  // loudly instead of the warning quietly reappearing with no coverage.
  it('stays quiet about the driver while no floor is established', async () => {
    const old = await boot({ before: (c) => { c.probe.detectDriverMajor.mockResolvedValue(550); } });
    old.emit('miner:start', { address: VALID_ADDR, mode: 'mining' });
    await flush();
    expect(old.sent('miner:log').map((l) => l.line).join('\n'))
      .not.toContain('is older than R');
    // The rig starts regardless, and lets the miner deliver upstream's own
    // driver message if the driver really is too old.
    expect(old.MinerManager.instances).toHaveLength(1);

    const unknown = await boot({ before: (c) => { c.probe.detectDriverMajor.mockResolvedValue(null); } });
    unknown.emit('miner:start', { address: VALID_ADDR, mode: 'mining' });
    await flush();
    expect(unknown.sent('miner:log').map((l) => l.line).join('\n'))
      .not.toContain('is older than R');
  });

  // A rig whose resolver is broken looks exactly like a pool that is down: the
  // engine reprints one opaque line every 5s. Say the useful thing — which host,
  // and that it is name resolution — ONCE, so the hint does not become the same
  // spam it exists to explain.
  it('explains a DNS failure once, naming the endpoint, and not for other failures', async () => {
    const ctx = await boot();
    ctx.emit('miner:start', { address: VALID_ADDR, mode: 'mining', region: 'us1' });
    await flush();
    const miner = ctx.MinerManager.instances[0];

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

  // The Windows log that motivated this: 27 `miner:start` events in 11 minutes
  // (the Chat tab's START LLM fires one on every click while the LLM isn't
  // ready), each launching its own download+extract, all racing on the same
  // scratch files. Repeated identical starts must collapse into one run.
  it('coalesces a burst of identical miner:start events into a single run', async () => {
    const ctx = await boot();
    wireHealth(ctx, (cb, req) => req.emit('error', new Error('down')));
    ctx.probe.detectVram.mockResolvedValue({ totalMb: 24000, usedMb: 1000 });

    for (let i = 0; i < 5; i++) ctx.emit('miner:start', { mode: 'llm' });
    await flush();

    const prepping = ctx.sent('miner:log').filter((l) => l.line.startsWith('preparing local LLM'));
    expect(prepping).toHaveLength(1);
    expect(ctx.LlmManager.instances).toHaveLength(1);
  });

  // A first run downloads ~100 MB of llama-server and a ~5 GB model. Before this,
  // the GUI logged "preparing local LLM…" and then said nothing for however long
  // that took, which reads as a hang — a user on a 2-GPU rig hit START seventeen
  // times in ninety seconds while the download was progressing fine. Progress has
  // to reach both the log and the hero line.
  it('reports model download progress to the log and the LLM hero', async () => {
    const ctx = await boot({
      before: (c) => {
        c.LlmEngineManager.behavior.ensureModel = (onProgress) => {
          onProgress(0);
          onProgress(42);
          return Promise.resolve('/tmp/llm/model.gguf');
        };
      },
    });
    wireHealth(ctx, (cb, req) => req.emit('error', new Error('down')));
    ctx.probe.detectVram.mockResolvedValue({ totalMb: 24000, usedMb: 1000 });

    ctx.emit('miner:start', { mode: 'llm' });
    await flush();

    const lines = ctx.sent('miner:log').map((l) => l.line);
    expect(lines).toContain('downloading model ' + ctx.config.LLM.model.name + '… 0%');
    expect(lines).toContain('downloading model ' + ctx.config.LLM.model.name + '… 42%');
    const notes = ctx.sent('llm:status').map((s) => s.note).filter(Boolean);
    expect(notes).toContain('downloading model ' + ctx.config.LLM.model.name + '… 42%');
    // The model is loading off disk by now — the hero must not fall back to
    // looking idle between the download finishing and 'ready' firing.
    expect(ctx.sent('llm:status').pop()).toMatchObject({ ready: false, note: 'Starting…' });
  });

  // downloadFile fires onProgress per chunk. Emitting every one would flood the
  // IPC and bury the log, so sub-5% moves inside 2s are dropped — but 100% is
  // never dropped, or the last thing shown is a stale "…95%". A repeated 100 is
  // still only reported once: on a slow link the 2s timer expires while the
  // percent hasn't moved, and re-logging the same number looks like a stutter.
  it('throttles progress chatter but always emits 100% exactly once', async () => {
    const ctx = await boot({
      before: (c) => {
        c.LlmEngineManager.behavior.ensureServer = (onProgress) => {
          [0, 1, 2, 3, 4, 99, 100, 100].forEach(onProgress);
          return Promise.resolve('/tmp/llm/llama-server');
        };
      },
    });
    wireHealth(ctx, (cb, req) => req.emit('error', new Error('down')));
    ctx.probe.detectVram.mockResolvedValue({ totalMb: 24000, usedMb: 1000 });

    ctx.emit('miner:start', { mode: 'llm' });
    await flush();

    const pcts = ctx.sent('miner:log')
      .map((l) => l.line)
      .filter((l) => l.startsWith('downloading llama-server…'))
      .map((l) => l.replace(/\D+/g, ''));
    expect(pcts).toEqual(['0', '99', '100']); // 1–4 collapsed into the 0% report
  });

  // A garbage percent (a server with no content-length makes progressPercent
  // return null) must not paint "null%" over the hero.
  it('ignores a non-numeric progress value', async () => {
    const ctx = await boot({
      before: (c) => {
        c.LlmEngineManager.behavior.ensureModel = (onProgress) => {
          onProgress(null);
          onProgress(undefined);
          onProgress(-1);
          return Promise.resolve('/tmp/llm/model.gguf');
        };
      },
    });
    wireHealth(ctx, (cb, req) => req.emit('error', new Error('down')));
    ctx.probe.detectVram.mockResolvedValue({ totalMb: 24000, usedMb: 1000 });

    ctx.emit('miner:start', { mode: 'llm' });
    await flush();

    expect(ctx.sent('miner:log').filter((l) => l.line.startsWith('downloading model'))).toHaveLength(0);
  });

  // …but a start that actually changes the plan must not be swallowed by the
  // one in flight. This is the START LLM button while mining-only is running.
  it('replays a mid-run start whose settings differ, so mining → both still serves', async () => {
    const ctx = await boot({ before: (c) => { c.EngineManager.behavior.installed = true; } });
    ctx.fs.existsSync.mockImplementation((p) => p === '/tmp/engine/alpha-miner');
    wireHealth(ctx, (cb, req) => req.emit('error', new Error('down')));
    ctx.probe.detectVram.mockResolvedValue({ totalMb: 24000, usedMb: 1000 });

    ctx.emit('miner:start', { address: VALID_ADDR, mode: 'mining' });
    ctx.emit('miner:start', { address: VALID_ADDR, mode: 'both' });
    await flush();

    // The replay ran with mode 'both': the already-running miner is kept and the
    // LLM half now waits on proof of hashrate, which mining-only never would.
    expect(ctx.MinerManager.instances).toHaveLength(1);
    ctx.MinerManager.instances[0].emit('event', { type: 'status', hashrate: '2.5' });
    await flush(30);
    expect(ctx.LlmManager.instances).toHaveLength(1);
  });

  it('miner:start with no payload defaults to auto → no address so no miner, but the LLM serves', async () => {
    const ctx = await boot();
    ctx.emit('miner:start');
    await flush();
    // DEFAULT_MODE is 'auto': mining needs a valid payout address and there is
    // none, but serving inference doesn't — so the LLM comes up on its own and
    // the session is NOT stopped.
    expect(ctx.MinerManager.instances).toHaveLength(0);
    expect(ctx.LlmManager.instances).toHaveLength(1);
    expect(ctx.sent('miner:stopped')).toHaveLength(0);
  });
});

// ── zip extraction helpers (passed into the engine managers) ─────────────────

describe('zip extraction helpers', () => {
  // The miner's EngineManager is no longer handed an `extract` at all. Its
  // Windows artifact goes through extractEnginePackage (whole tree, launcher
  // renamed afterwards) and its Linux one is a bare binary, so the old
  // single-file extractZip had no remaining caller and was deleted with its
  // tests. This asserts the wiring stays gone rather than silently returning.
  it('the miner engine is wired without a single-file zip extractor', async () => {
    const ctx = await boot();
    ctx.emit('miner:start', { address: VALID_ADDR, mode: 'mining' });
    await flush();
    const opts = ctx.EngineManager.instances[0].opts;
    expect(opts.extract).toBeUndefined();
    expect(Object.prototype.hasOwnProperty.call(opts, 'extractPackage')).toBe(true);
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

  // Regression: the download lands at llmEngineManager's format-neutral
  // ARCHIVE_TMP ('llama-download.archive'). Expand-Archive validates by
  // extension and takes only '.zip', so it refused every Windows install with
  // "'.archive' is not a supported archive file format" and the local LLM could
  // never start. Unzip by content, not by name.
  it('extractLlamaZipWin unzips a file whose name is not .zip', async () => {
    const { ctx, extract } = await llamaWinExtract();
    ctx.cp.execFile.mockImplementation((...args) => args[args.length - 1](null));
    ctx.fs.existsSync.mockImplementation((p) => p === '/tmp/llm/llama-server.exe');
    const archive = '/tmp/llm/llama-download.archive';
    await expect(extract(archive, '/tmp/llm/llama-server.exe')).resolves.toBe('/tmp/llm/llama-server.exe');

    const ps = ctx.cp.execFile.mock.calls.pop()[1].pop();
    expect(ps).not.toContain('Expand-Archive');
    expect(ps).toContain("[System.IO.Compression.ZipFile]::ExtractToDirectory('" + archive + "',");
    // Add-Type is required on Windows PowerShell 5.1 and throws on 7, where the
    // type is already loaded — so it must be swallowed, not fatal.
    expect(ps).toContain('try{Add-Type -AssemblyName System.IO.Compression.FileSystem}catch{}');
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
        // getOrCreateNode returns the stored node when there is one, so both mocks
        // must agree — the worker signs and posts against this node's serverUrl.
        c.nodeStore.loadNode.mockReturnValue(fakeNode({ connected: true, serverUrl: 'https://custom.example' }));
        c.nodeStore.getOrCreateNode.mockReturnValue(fakeNode({ connected: true, serverUrl: 'https://custom.example' }));
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
    // One card with 4000 MB free — below the model's floor. The LLM sizes
    // against a single GPU (llama-server --split-mode none), so the per-card
    // figure is what the preflight uses.
    ctx.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'gpu', usedMb: 4000, totalMb: 8000 }]);
    ctx.emit('miner:start', { mode: 'llm' });
    await flush();
    expect(ctx.sent('miner:log').map((l) => l.line).join('\n')).toContain('not enough free VRAM for the local LLM: 4000 MB free');
    // Derived from the model's floor rather than pinned: the message quotes
    // minVramMb, which moves whenever ctxSize does (a bigger window is a bigger
    // KV cache), and a literal here just breaks on every such change.
    const needGb = Math.round(ctx.config.LLM.model.minVramMb / 1024);
    expect(ctx.sent('llm:status').pop()).toMatchObject({ ready: false, error: `Needs ~${needGb} GB free VRAM` });
    expect(ctx.sent('miner:stopped')).toHaveLength(1);
    expect(ctx.LlmManager.instances).toHaveLength(0);
  });

  it('a rejection out of startLlm itself is caught and ends an llm-only session', async () => {
    const ctx = await boot({
      before: (c) => { c.probe.detectGpusVram.mockRejectedValue(new Error('probe exploded')); },
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
      // …and the hero says so. Clearing the note without an error dropped the
      // row back to a grey dot and the model name, which is what "idle" looks
      // like — a download that died at 57% after twenty minutes appeared to
      // have simply never started.
      expect(ctx.sent('llm:status').pop()).toMatchObject({ ready: false, note: null, error: 'Setup failed — see Logs' });
    });
  });

  it('starts llama-server, goes ready, serves jobs, streams stats, and reports its exit', async () => {
    const ctx = await boot({
      before: (c) => {
        c.nodeStore.loadNode.mockReturnValue(fakeNode({ connected: true, name: 'rig' }));
        c.nodeStore.getOrCreateNode.mockReturnValue(fakeNode({ connected: true, name: 'rig' }));
        // One roomy card (22 GB free) — the whole model fits, full offload,
        // pinned to GPU 0 (--main-gpu).
        c.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 4090', usedMb: 2000, totalMb: 24000 }]);
      },
    });
    ctx.emit('miner:start', { mode: 'llm' });
    await flush();

    const llm = ctx.LlmManager.instances[0];
    // free 22000 MB, no reserve → full offload, on GPU 0
    expect(llm.start).toHaveBeenCalledWith({
      platform: 'linux', binaryPath: '/tmp/llm/llama-server', modelPath: '/tmp/llm/model.gguf',
      host: '127.0.0.1', nGpuLayers: ALL_LAYERS, port: 8080, mainGpu: 0,
    });
    expect(ctx.sent('llm:status').pop()).toMatchObject({ ready: false }); // endpoint fills in on ready

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
    const onReasoning = jest.fn();
    await worker.opts.runJob({ messages: [] }, { onDelta, onReasoning });
    expect(ctx.io.streamChatCompletion).toHaveBeenLastCalledWith('http://127.0.0.1:8080', { messages: [] }, onDelta, onReasoning);

    llm.emit('stats', { tokensPerSec: 33 });
    expect(ctx.sent('llm:status').pop()).toMatchObject({ tokensPerSec: 33 });

    // a linked ping while the worker runs reports its active jobs
    await ctx.invoke('node:connect', { token: 'tok' });
    await flush();
    expect(worker.activeJobs).toHaveBeenCalled();

    // Unlinking no longer stops serving: a machine that can run the model is
    // useful to the network account or not, so it keeps taking PUBLIC jobs (it
    // self-registers). Only losing the model stops the worker.
    ctx.nodeStore.loadNode.mockReturnValue(null);
    await ctx.invoke('node:connect', { token: 'tok' });
    await flush();
    expect(worker.stop).not.toHaveBeenCalled();
    ctx.nodeStore.loadNode.mockReturnValue(fakeNode({ connected: true, name: 'rig' }));

    // llama-server exits after ready in an llm-only session → session over
    llm.emit('stopped');
    await flush();
    expect(worker.stop).toHaveBeenCalled();
    expect(ctx.sent('miner:stopped')).toHaveLength(1);
    expect(ctx.sent('llm:status').pop()).toMatchObject({ ready: false, tokensPerSec: 0 });
  });

  it('runs one instance and worker per eligible GPU, summing their active jobs', async () => {
    const ctx = await boot({
      before: (c) => {
        c.nodeStore.loadNode.mockReturnValue(fakeNode({ connected: true, name: 'rig' }));
        c.nodeStore.getOrCreateNode.mockReturnValue(fakeNode({ connected: true, name: 'rig' }));
        // Two roomy cards → one llama-server + cluster worker pinned to each.
        c.probe.detectGpusVram.mockResolvedValue([
          { index: 0, name: 'RTX 4090', usedMb: 2000, totalMb: 24000 },
          { index: 1, name: 'RTX 4090', usedMb: 1000, totalMb: 24000 },
        ]);
        c.probe.findFreePort.mockImplementation((h, p) => Promise.resolve(p));
      },
    });
    ctx.emit('miner:start', { mode: 'llm' });
    await flush();

    // The plural log names every planned card, but they start ONE AT A TIME:
    // simultaneous multi-GB model loads thrash the page cache (see LlmFleet).
    expect(ctx.sent('miner:log').map((l) => l.line)).toContain('local LLM starting on 2 GPUs [0, 1]');
    expect(ctx.LlmManager.instances).toHaveLength(1);
    const g0 = ctx.LlmManager.instances[0];
    expect(g0.start).toHaveBeenCalledWith(expect.objectContaining({ port: 8080, mainGpu: 0 }));

    // card 0 ready → card 1 spawns on the next port; chat targets the first ready
    g0.emit('ready', { baseUrl: g0.baseUrl });
    await flush();
    expect(ctx.LlmManager.instances).toHaveLength(2);
    const g1 = ctx.LlmManager.instances[1];
    expect(g1.start).toHaveBeenCalledWith(expect.objectContaining({ port: 8081, mainGpu: 1 }));

    g1.emit('ready', { baseUrl: g1.baseUrl });
    await flush();
    expect(ctx.JobWorker.instances).toHaveLength(2);
    expect(ctx.sent('llm:status').pop()).toMatchObject({ ready: true, webUrl: 'http://127.0.0.1:8080' });

    // a transient error on one card is swallowed — the fleet keeps the others up
    g0.emit('error', new Error('transient blip'));
    expect(ctx.sent('llm:status').pop()).toMatchObject({ ready: true });

    // a telemetry ping sums active jobs across every worker (each mock reports 1)
    await ctx.invoke('node:connect', { token: 'tok' });
    await flush();
    ctx.io.postJson.mockClear();
    const pinger = ctx.interval(ctx.config.NODE.pingIntervalMs);
    pinger.fn();
    await flush();
    expect(ctx.io.postJson.mock.calls.pop()[1].activeJobs).toBe(2);

    // disconnecting the node tears every worker down through stopWorker
    await ctx.invoke('node:disconnect');
    expect(ctx.JobWorker.instances[0].stop).toHaveBeenCalled();
    expect(ctx.JobWorker.instances[1].stop).toHaveBeenCalled();
  });

  it('still serves on a card that comes up after the node unlinks (public jobs need no account)', async () => {
    const ctx = await boot({
      before: (c) => {
        c.nodeStore.loadNode.mockReturnValue(fakeNode({ connected: true, name: 'rig' }));
        c.nodeStore.getOrCreateNode.mockReturnValue(fakeNode({ connected: true, name: 'rig' }));
        c.probe.detectGpusVram.mockResolvedValue([
          { index: 0, name: 'RTX 4090', usedMb: 2000, totalMb: 24000 },
          { index: 1, name: 'RTX 4090', usedMb: 1000, totalMb: 24000 },
        ]);
        c.probe.findFreePort.mockImplementation((h, p) => Promise.resolve(p));
      },
    });
    ctx.emit('miner:start', { mode: 'llm' });
    await flush();
    // Instances start one at a time, so only GPU 0 exists until it settles.
    const g0 = ctx.LlmManager.instances[0];

    g0.emit('ready', { baseUrl: g0.baseUrl }); // linked → a worker for GPU 0
    await flush();
    expect(ctx.JobWorker.instances).toHaveLength(1);

    // The node drops before GPU 1 comes up. Serving no longer depends on the
    // account, so that card still gets a worker and takes public jobs.
    ctx.nodeStore.loadNode.mockReturnValue(null);
    const g1 = ctx.LlmManager.instances[1];
    g1.emit('ready', { baseUrl: g1.baseUrl });
    await flush();
    expect(ctx.JobWorker.instances).toHaveLength(2);
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
    // The preferred Linux engine is now a PACKAGE, so the bundled path is a
    // launcher inside a directory — ask engine.js rather than hardcoding a name.
    const { bundledEnginePath, ENGINE } = require('../src/shared/engine');
    const bundledMiner = bundledEnginePath('/res', 'linux', undefined, ENGINE.linux);
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

    // VRAM unknown → full offload; port 8080 busy → the fleet walks to 8081
    const llm = ctx.LlmManager.instances[0];
    expect(llm.start).toHaveBeenCalledWith(expect.objectContaining({
      nGpuLayers: ALL_LAYERS, port: 8081,
    }));

    // llama-server dies before ready while mining keeps running
    llm.emit('stopped');
    await flush();
    expect(ctx.sent('llm:status').pop()).toMatchObject({
      ready: false, error: 'The local LLM stopped before it was ready. See Logs.',
    });
    expect(ctx.sent('miner:stopped')).toHaveLength(0);

    // …and once an instance DOES come up, that error must not outlive it. On a
    // multi-GPU rig the next card follows the dead one, so this is the ordinary
    // case, not an edge one. The renderer ranks error above ready in both the
    // hero dot and its label, and nothing else clears it while a fleet runs —
    // startLlm's pre-spawn reset only fires on a fresh start — so a stale error
    // left the app permanently red while the model answered normally.
    llm.emit('ready', { baseUrl: 'http://127.0.0.1:8081' });
    await flush();
    const recovered = ctx.sent('llm:status').pop();
    expect(recovered).toMatchObject({ ready: true, endpoint: 'http://127.0.0.1:8081/v1' });
    expect(recovered.error).toBeNull();
  });

  it('both mode: the board report tags the GPU serving the local LLM', async () => {
    const ctx = await boot({
      before: (c) => {
        c.nodeStore.loadNode.mockReturnValue(fakeNode({ connected: true, name: 'rig' }));
        c.nodeStore.getOrCreateNode.mockReturnValue(fakeNode({ connected: true, name: 'rig' }));
        c.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 4090', usedMb: 2000, totalMb: 24000 }]);
      },
    });
    ctx.fs.existsSync.mockImplementation((p) => p === '/tmp/engine/alpha-miner');

    ctx.emit('miner:start', { address: VALID_ADDR, mode: 'both' });
    await flush();

    // the miner proves real hashrate → the LLM fleet starts
    const miner = ctx.MinerManager.instances[0];
    miner.emit('event', { type: 'status', hashrate: '5' });
    await flush(30);

    const llm = ctx.LlmManager.instances[0];
    llm.emit('ready', { baseUrl: llm.baseUrl }); // GPU 0 now serves the model
    await flush();

    // a board report now tags GPU 0 with the served model
    ctx.probe.postMinerReport.mockClear();
    const reporter = ctx.interval(ctx.config.NETWORK.reportIntervalMs);
    reporter.fn();
    await flush();
    const payloads = ctx.probe.postMinerReport.mock.calls.map((c) => c[0]);
    expect(payloads.some((p) => p.llmModel === ctx.config.LLM.model.name)).toBe(true);
    // Linked, so the rows also carry the node id — the board's "serving the
    // cluster" marker, as opposed to merely running the model.
    expect(payloads.some((p) => p.nodeId === 'abc123')).toBe(true);
  });

  it('degrades to local-only when no node identity can be minted', async () => {
    // getOrCreateNode never returns null in production, but a read-only or full
    // disk could break identity persistence. The model must still run locally:
    // no worker, no crash, and the board row reports no node id.
    const ctx = await boot({
      before: (c) => {
        c.nodeStore.loadNode.mockReturnValue(null);
        c.nodeStore.getOrCreateNode.mockReturnValue(null);
        c.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 4090', usedMb: 2000, totalMb: 24000 }]);
      },
    });
    ctx.fs.existsSync.mockImplementation((p) => p === '/tmp/engine/alpha-miner');
    ctx.emit('miner:start', { address: VALID_ADDR, mode: 'both' });
    await flush();
    ctx.MinerManager.instances[0].emit('event', { type: 'status', hashrate: '5' });
    await flush(30);
    const llm = ctx.LlmManager.instances[0];
    llm.emit('ready', { baseUrl: llm.baseUrl });
    await flush();

    expect(ctx.JobWorker.instances).toHaveLength(0); // serving declined, app alive
    ctx.probe.postMinerReport.mockClear();
    ctx.interval(ctx.config.NETWORK.reportIntervalMs).fn();
    await flush();
    const payloads = ctx.probe.postMinerReport.mock.calls.map((c) => c[0]);
    expect(payloads.length).toBeGreaterThan(0);
    expect(payloads.every((p) => p.nodeId === null)).toBe(true);
  });

  it('board report carries the node id for an UNLINKED host, which now serves public jobs', async () => {
    const ctx = await boot({
      before: (c) => {
        // Unlinked but serving: the worker is armed on the fleet, not the account,
        // so this host takes public jobs and the board must mark it as serving.
        c.nodeStore.loadNode.mockReturnValue(fakeNode({ connected: false }));
        c.nodeStore.getOrCreateNode.mockReturnValue(fakeNode({ connected: false }));
        c.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 4090', usedMb: 2000, totalMb: 24000 }]);
      },
    });
    ctx.fs.existsSync.mockImplementation((p) => p === '/tmp/engine/alpha-miner');

    ctx.emit('miner:start', { address: VALID_ADDR, mode: 'both' });
    await flush();
    ctx.MinerManager.instances[0].emit('event', { type: 'status', hashrate: '5' });
    await flush(30);
    const llm = ctx.LlmManager.instances[0];
    llm.emit('ready', { baseUrl: llm.baseUrl });
    await flush();

    ctx.probe.postMinerReport.mockClear();
    ctx.interval(ctx.config.NETWORK.reportIntervalMs).fn();
    await flush();
    const payloads = ctx.probe.postMinerReport.mock.calls.map((c) => c[0]);
    expect(payloads.length).toBeGreaterThan(0);
    expect(payloads.every((p) => p.nodeId === 'abc123')).toBe(true);
    // And it announced itself as serving without an account.
    expect(ctx.sent('miner:log').map((l) => l.line).join('\n')).toContain('unlinked node');
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
      .toHaveBeenCalledWith(expect.objectContaining({ nGpuLayers: ALL_LAYERS }));
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
      ctx.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'gpu', usedMb: 0, totalMb: 4000 }]);
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
    ctx.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'gpu', usedMb: 0, totalMb: 4000 }]);
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
