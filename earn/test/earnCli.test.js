'use strict';

// Unit tests for the headless CLI shell (src/cli/earn-cli.js). Everything with
// real IO — child_process, fs, os, the process managers, the probe/io helpers,
// nodeStore and the self-updater — is mocked; the pure shared modules (cliArgs,
// config, format, address, miningStats, minerReport, statsFile, …) run for
// real, exactly like they do in production. Each test loads a fresh copy of the
// module via jest.isolateModules so the CLI's module-level state (the GPU-probe
// cache, cluster job worker, serve pinger) never leaks between tests.

jest.mock('child_process', () => ({ spawn: jest.fn(), execFile: jest.fn(), spawnSync: jest.fn() }));
jest.mock('fs');
jest.mock('os', () => ({
  hostname: jest.fn(() => 'Rig-Host.local'),
  homedir: jest.fn(() => '/home/test'),
  tmpdir: jest.fn(() => '/tmp'),
  platform: jest.fn(() => 'linux'),
  // 32 GB with 20 free: the RAM every board row reports (12288 / 32768 MB).
  totalmem: jest.fn(() => 32 * 2 ** 30),
  freemem: jest.fn(() => 20 * 2 ** 30),
  EOL: '\n',
}));
jest.mock('../src/main/probe', () => ({
  detectRegion: jest.fn(),
  detectVram: jest.fn(),
  detectGpusVram: jest.fn(),
  detectMinerGpus: jest.fn(),
  detectCudaCards: jest.fn(),
  detectPciBusIds: jest.fn(),
  detectLlmDevices: jest.fn(),
  postMinerReport: jest.fn(),
  findFreePort: jest.fn(),
  // Shared with the GUI now — one detection path for both shells.
  detectGpuInfo: jest.fn(),
  detectGpuTemps: jest.fn(),
}));
jest.mock('../src/main/io', () => ({
  postJson: jest.fn(),
  getJson: jest.fn(),
  downloadFile: jest.fn(),
  streamChatCompletion: jest.fn(),
  extractLlamaZip: jest.fn(),
}));
jest.mock('../src/main/nodeStore', () => ({
  nodePath: jest.fn(() => '/tmp/store/node.json'),
  loadNode: jest.fn(),
  saveNode: jest.fn(),
  getOrCreateNode: jest.fn(),
}));
jest.mock('../src/cli/selfUpdater', () => ({
  UPDATED_ENV: 'LLMJOB_EARN_UPDATED',
  fetchLatestRelease: jest.fn(),
  isPackaged: jest.fn(),
  applyUpdate: jest.fn(),
  hasXmrWorker: jest.fn(),
  syncXmrWorker: jest.fn(),
  hasAmdCore: jest.fn(),
  fetchAmdCore: jest.fn(),
  reexec: jest.fn(),
}));
jest.mock('../src/shared/selfUpdate', () => ({ planUpdate: jest.fn() }));
// The HiveOS IO (folder swap, trial files, dashboard posts). Its rules in
// shared/hiveos run for real.
jest.mock('../src/cli/hiveos', () => ({
  isHiveos: jest.fn(),
  packageDir: jest.fn(),
  postMessage: jest.fn(),
  readTrial: jest.fn(),
  isSkipped: jest.fn(),
  confirmTrial: jest.fn(),
  cleanStaging: jest.fn(),
  updatePackage: jest.fn(),
  checkDriver: jest.fn(),
}));
// One engine now: our own core, in-process. There is nothing to download and
// nothing to spawn, so the CLI's engine mocks are just this.
jest.mock('../src/main/pearlEngine', () => {
  const { EventEmitter } = require('events');
  class PearlEngine extends EventEmitter {
    constructor(opts) {
      super();
      this.opts = opts;
      this.start = jest.fn((settings) => {
        this.settings = settings;
        if (PearlEngine.startError) throw PearlEngine.startError;
        // undefined by default, as the real engine returns when it started. A
        // fatal start failure returns false, which the CLI must treat as fatal.
        return PearlEngine.startReturns;
      });
      this.stop = jest.fn();
      this.isRunning = jest.fn(() => true);
      this.poolConnected = jest.fn(() => true);
      this.pauseCards = jest.fn(() => [{ index: 0, name: 'GPU0' }]);
      this.resumeCards = jest.fn(() => [{ index: 0, name: 'GPU0' }]);
      PearlEngine.instances.push(this);
    }
  }
  PearlEngine.instances = [];
  PearlEngine.startError = null;
  PearlEngine.startReturns = undefined;
  return { PearlEngine };
});
// The Monero CPU miner: an engine that records what it is asked, the CPU facts
// its thread plan reads, and where its worker file is.
jest.mock('../src/main/xmrEngine', () => {
  const { EventEmitter } = require('events');
  class XmrEngine extends EventEmitter {
    constructor(opts) {
      super();
      this.opts = opts;
      this.running = false;
      this.start = jest.fn((settings) => { this.settings = settings; this.running = true; return true; });
      this.stop = jest.fn(() => { this.running = false; });
      this.isRunning = jest.fn(() => this.running);
      this.snapshot = jest.fn(() => ({ state: 'mining', hashrate: 1845.2, accepted: 3, rejected: 0, threads: 15 }));
      XmrEngine.instances.push(this);
    }
  }
  XmrEngine.instances = [];
  return { XmrEngine };
});
jest.mock('../src/main/cpuFacts', () => ({ readCpuFacts: jest.fn() }));
jest.mock('../src/main/xmrWorker', () => ({ findWorker: jest.fn(), makeSpawnWorker: jest.fn(() => 'spawnWorker') }));
jest.mock('../src/main/pearlCore', () => ({
  // A loadable core by default: the CLI decides UP FRONT whether it can mine,
  // so a null factory now means "refuse or degrade", not "construct an engine
  // that will announce the problem later". The no-core paths have their own
  // tests; everything else models a rig that can actually mine.
  coreFactory: jest.fn(() => () => null),
}));
jest.mock('net', () => ({ connect: jest.fn(() => ({ on: jest.fn(), write: jest.fn(), destroy: jest.fn() })) }));
jest.mock('../src/main/llmManager', () => {
  const { EventEmitter } = require('events');
  class LlmManager extends EventEmitter {
    constructor(opts) {
      super();
      this.opts = opts;
      this.baseUrl = 'http://127.0.0.1:8080';
      // Mirror the real manager: the bound URL follows the port the fleet walked
      // to, so a busy 8080 lands the worker on the actual serving port.
      this.start = jest.fn((o) => { this.baseUrl = 'http://127.0.0.1:' + (o && o.port); });
      this.stop = jest.fn();
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
      this.isServerInstalled = jest.fn(() => LlmEngineManager.serverInstalled);
      this.serverBinaryPath = jest.fn(() => '/cache/llama-server');
      this.ensureServer = jest.fn(async (onPct) => {
        if (LlmEngineManager.serverError) throw LlmEngineManager.serverError;
        if (onPct) { onPct(10); onPct(null); }
        return '/cache/llama-server';
      });
      this.isModelInstalled = jest.fn(() => LlmEngineManager.modelInstalled);
      this.modelPath = jest.fn(() => '/cache/model.gguf');
      this.ensureModel = jest.fn(async (onPct) => {
        // 25 after 20: the same tenth, which a background download logs once.
        if (onPct) { onPct(20); onPct(25); onPct(null); }
        // A test can hold the download open (a promise it resolves later).
        if (LlmEngineManager.modelHold) await LlmEngineManager.modelHold;
        return '/cache/model.gguf';
      });
      // The vision projector half. Defaults to "already satisfied, none to
      // fetch", which is what a text-only model reports — the default fleet
      // model has no projector.
      this.isMmprojInstalled = jest.fn(() => LlmEngineManager.mmprojInstalled);
      this.mmprojPath = jest.fn(() => LlmEngineManager.mmprojFile);
      this.ensureMmproj = jest.fn(async (onPct) => {
        if (onPct) { onPct(30); onPct(null); }
        return LlmEngineManager.mmprojFile;
      });
      LlmEngineManager.instances.push(this);
    }
  }
  LlmEngineManager.instances = [];
  LlmEngineManager.serverInstalled = false;
  LlmEngineManager.modelInstalled = false;
  LlmEngineManager.mmprojInstalled = true;   // text-only default: nothing to fetch
  LlmEngineManager.mmprojFile = null;
  LlmEngineManager.serverError = null;
  LlmEngineManager.modelHold = null;
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
      this.activeJobs = jest.fn(() => 2);
      JobWorker.instances.push(this);
    }
  }
  JobWorker.instances = [];
  return { JobWorker };
});

const pkg = require('../package.json');
const { NETWORK, NODE, LLM } = require('../src/shared/config');
const nodeProto = require('../src/shared/node');
const { ALL_LAYERS } = require('../src/shared/vram');

const ADDR = 'prl1p' + 'a'.repeat(30);
const MDL = 'mdl1p' + 'b'.repeat(30);
const KEYS = nodeProto.generateKeypair();

function makeNode(extra) {
  return Object.assign({
    nodeId: 'abc123',
    publicKey: KEYS.publicKey,
    secretKey: KEYS.secretKey,
    name: null,
    connected: false,
    serverUrl: null,
  }, extra || {});
}

// ── Shared per-test capture state ────────────────────────────────────────────
let out; // strings written to stdout
let err; // strings written to stderr
let intervals; // { fn, ms, unref } handles captured from setInterval
let sigHandlers; // signal name -> [handler]
let origOutTty;
let origErrTty;

const allOut = () => out.join('');
const allErr = () => err.join('');
const fire = (sig) => { (sigHandlers[sig] || []).forEach((fn) => fn()); };
const intervalFor = (ms) => intervals.find((h) => h.ms === ms);
const tick = () => new Promise((resolve) => setImmediate(resolve));
async function settle(n) { for (let i = 0; i < (n || 4); i++) await tick(); }

function applyDefaults(m) {
  m.cp.execFile.mockImplementation((cmd, args, opts, cb) => cb(new Error('no nvidia-smi')));
  m.fs.existsSync.mockReturnValue(true);
  m.probe.detectRegion.mockResolvedValue('us');
  m.probe.detectVram.mockResolvedValue(null);
  m.probe.detectGpusVram.mockResolvedValue([]);
  // No card list by default: one core that places itself, which is what a rig
  // with no nvidia-smi gets.
  m.probe.detectMinerGpus.mockResolvedValue([]);
  // Nothing known about the cards: the core factory then picks the 12.8 build.
  m.probe.detectCudaCards.mockResolvedValue([]);
  m.probe.detectPciBusIds.mockResolvedValue({});
  m.probe.detectLlmDevices.mockResolvedValue([]);
  m.probe.postMinerReport.mockResolvedValue(undefined);
  m.probe.findFreePort.mockResolvedValue(8080);
  m.probe.detectGpuInfo.mockResolvedValue(null); // no identifiable GPU by default
  m.io.postJson.mockResolvedValue({ status: 200, data: {} });
  m.io.downloadFile.mockResolvedValue(undefined);
  m.io.streamChatCompletion.mockReturnValue({ done: Promise.resolve('') });
  m.io.extractLlamaZip.mockResolvedValue(undefined);
  m.nodeStore.loadNode.mockReturnValue(null);
  m.nodeStore.getOrCreateNode.mockReturnValue(makeNode());
  m.selfUpdater.fetchLatestRelease.mockResolvedValue(null);
  m.selfUpdater.isPackaged.mockReturnValue(false);
  m.selfUpdater.applyUpdate.mockResolvedValue('/opt/earn');
  m.selfUpdater.reexec.mockReturnValue(0);
  m.selfUpdater.syncXmrWorker.mockResolvedValue(false);
  m.selfUpdater.hasXmrWorker.mockReturnValue(true);
  m.selfUpdate.planUpdate.mockReturnValue({ updateAvailable: false, reason: 'up-to-date' });
  m.hive.isHiveos.mockReturnValue(false);
  m.hive.packageDir.mockReturnValue('/hive/miners/custom/llmjob-earn');
  m.hive.postMessage.mockReturnValue(true);
  m.hive.readTrial.mockReturnValue(null);
  m.hive.isSkipped.mockReturnValue(false);
  m.hive.updatePackage.mockResolvedValue(undefined);
  m.hive.checkDriver.mockResolvedValue(null);
  m.PearlEngine.startReturns = undefined;
  m.cpuFacts.readCpuFacts.mockReturnValue({ cpus: 16, quotaCpus: null, memoryBytes: 32 * 2 ** 30, l3Bytes: null });
  m.xmrWorker.findWorker.mockReturnValue('/opt/llmjob-xmr-worker');
}

// Load a fresh earn-cli plus fresh instances of every mocked dependency.
function load() {
  const m = {};
  jest.isolateModules(() => {
    m.cp = require('child_process');
    m.fs = require('fs');
    m.os = require('os');
    m.probe = require('../src/main/probe');
    m.io = require('../src/main/io');
    m.nodeStore = require('../src/main/nodeStore');
    m.selfUpdater = require('../src/cli/selfUpdater');
    m.selfUpdate = require('../src/shared/selfUpdate');
    m.hive = require('../src/cli/hiveos');
    m.net = require('net');
    m.PearlEngine = require('../src/main/pearlEngine').PearlEngine;
    m.XmrEngine = require('../src/main/xmrEngine').XmrEngine;
    m.cpuFacts = require('../src/main/cpuFacts');
    m.xmrWorker = require('../src/main/xmrWorker');
    m.autoGate = require('../src/main/autoGate');
    m.autoGate.createAutoGate.instances = [];
    m.autoGate.createServeGate.instances = [];
    m.pearlCore = require('../src/main/pearlCore');
    m.LlmManager = require('../src/main/llmManager').LlmManager;
    m.LlmEngineManager = require('../src/main/llmEngineManager').LlmEngineManager;
    m.JobWorker = require('../src/main/jobWorker').JobWorker;
    applyDefaults(m);
    m.run = require('../src/cli/earn-cli').run;
  });
  return m;
}

// The CLI is shipped for Linux, but `node src/cli/earn-cli.js` runs anywhere —
// so the macOS gate (no alpha-miner exists for it) and the arch-aware
// llama-server download both need process.platform / process.arch pinned.
//
// Every test starts pinned to linux, and that is not tidiness: process.platform
// now decides whether the CLI mines at all, so a suite that inherited the host's
// platform passed on Linux and Windows and failed 18 tests on the macOS runner
// (which was, correctly, refusing to mine). mainProcess.test.js has always
// pinned for the same reason. Same lesson as the path-separator expectations
// elsewhere in this suite: build the conditions the same way on every OS.
const REAL_PLATFORM = Object.getOwnPropertyDescriptor(process, 'platform');
const REAL_ARCH = Object.getOwnPropertyDescriptor(process, 'arch');
function setPlatform(p) {
  Object.defineProperty(process, 'platform', { value: p, configurable: true });
}
function setArch(a) {
  Object.defineProperty(process, 'arch', { value: a, configurable: true });
}

beforeEach(() => {
  setPlatform('linux'); // a mining-capable platform, whatever the host is
  setArch('x64');
  out = [];
  err = [];
  intervals = [];
  sigHandlers = {};
  origOutTty = process.stdout.isTTY;
  origErrTty = process.stderr.isTTY;
  process.stdout.isTTY = false;
  process.stderr.isTTY = false;
  jest.spyOn(process.stdout, 'write').mockImplementation((s) => { out.push(String(s)); return true; });
  jest.spyOn(process.stderr, 'write').mockImplementation((s) => { err.push(String(s)); return true; });
  jest.spyOn(global, 'setInterval').mockImplementation((fn, ms) => {
    const h = { fn, ms, unref: jest.fn() };
    intervals.push(h);
    return h;
  });
  jest.spyOn(global, 'clearInterval').mockImplementation(() => {});
  jest.spyOn(process, 'on').mockImplementation((ev, fn) => {
    (sigHandlers[ev] = sigHandlers[ev] || []).push(fn);
    return process;
  });
});

afterEach(() => {
  jest.restoreAllMocks();
  process.stdout.isTTY = origOutTty;
  process.stderr.isTTY = origErrTty;
  delete process.env.LLMJOB_EARN_UPDATED;
  Object.defineProperty(process, 'platform', REAL_PLATFORM);
  Object.defineProperty(process, 'arch', REAL_ARCH);
});

// ── help / version / bad args ────────────────────────────────────────────────

describe('help, version and argument errors', () => {
  test('--help prints usage and exits 0', async () => {
    const m = load();
    await expect(m.run(['--help'])).resolves.toBe(0);
    expect(allOut()).toContain('Usage: llmjob-earn-cli');
  });

  test('--version prints the package version and exits 0', async () => {
    const m = load();
    await expect(m.run(['--version'])).resolves.toBe(0);
    expect(allOut()).toContain(pkg.version);
  });

  test('bad arguments report every error and exit 1', async () => {
    const m = load();
    await expect(m.run(['--bogus', '--address'])).resolves.toBe(1);
    expect(allErr()).toContain('error: unknown option: --bogus');
    expect(allErr()).toContain('error: missing value for --address');
    expect(allErr()).toContain('run with --help for usage');
  });
});

// ── explicit `update` subcommand ─────────────────────────────────────────────

describe('update subcommand', () => {
  test('exits 1 when the update server is unreachable', async () => {
    const m = load();
    await expect(m.run(['update'])).resolves.toBe(1);
    expect(allErr()).toContain('could not reach the update server');
  });

  [
    { reason: 'up-to-date', text: 'already up to date' },
    { reason: 'asset-missing', text: 'no Linux CLI binary yet' },
    { reason: 'unsupported-platform', text: 'only available for the Linux binary' },
    { reason: 'something-else', text: 'no newer release found' },
  ].forEach((c) => {
    test('exits 0 when no update applies (' + c.reason + ')', async () => {
      const m = load();
      m.selfUpdater.fetchLatestRelease.mockResolvedValue({ version: '9.9.9' });
      m.selfUpdate.planUpdate.mockReturnValue({ updateAvailable: false, reason: c.reason, latestVersion: '9.9.9' });
      await expect(m.run(['update'])).resolves.toBe(0);
      expect(allOut() + allErr()).toContain(c.text);
      // One answer only: "already up to date" used to go on to "no newer
      // release found".
      if (c.reason === 'up-to-date') expect(allOut() + allErr()).not.toContain('no newer release found');
      expect(m.selfUpdate.planUpdate).toHaveBeenCalledWith({
        currentVersion: pkg.version, release: { version: '9.9.9' }, platform: process.platform,
      });
    });
  });

  // A CLI installed without its CPU worker, or whose worker download failed.
  test('up to date, it still fetches this release\'s CPU worker when it needs one', async () => {
    const m = load();
    m.selfUpdater.fetchLatestRelease.mockResolvedValue({ version: pkg.version });
    m.selfUpdater.isPackaged.mockReturnValue(true);
    m.selfUpdate.planUpdate.mockReturnValue({ updateAvailable: false, reason: 'up-to-date', xmrWorkerUrl: 'https://dl/w' });
    m.selfUpdater.syncXmrWorker.mockResolvedValue(true);
    await expect(m.run(['update'])).resolves.toBe(0);
    expect(m.selfUpdater.syncXmrWorker).toHaveBeenCalledWith('https://dl/w', pkg.version);
    expect(allOut()).toContain('fetched llmjob-xmr-worker v' + pkg.version + ' (it was missing, would not run, or was from another release)');

    // One already in step says nothing; a failed fetch says why.
    const m2 = load();
    m2.selfUpdater.fetchLatestRelease.mockResolvedValue({ version: pkg.version });
    m2.selfUpdater.isPackaged.mockReturnValue(true);
    m2.selfUpdate.planUpdate.mockReturnValue({ updateAvailable: false, reason: 'up-to-date', xmrWorkerUrl: 'https://dl/w' });
    m2.selfUpdater.syncXmrWorker.mockRejectedValue(new Error('HTTP 404'));
    await expect(m2.run(['update'])).resolves.toBe(0);
    expect(allErr()).toContain('could not fetch llmjob-xmr-worker: HTTP 404');

    // From source there is no binary to put it beside.
    const m3 = load();
    m3.selfUpdater.fetchLatestRelease.mockResolvedValue({ version: pkg.version });
    m3.selfUpdate.planUpdate.mockReturnValue({ updateAvailable: false, reason: 'up-to-date', xmrWorkerUrl: 'https://dl/w' });
    await m3.run(['update']);
    expect(m3.selfUpdater.syncXmrWorker).not.toHaveBeenCalled();

    // Nor on a rig with no worker: `update` can't tell it mines Monero.
    const m4 = load();
    m4.selfUpdater.fetchLatestRelease.mockResolvedValue({ version: pkg.version });
    m4.selfUpdater.isPackaged.mockReturnValue(true);
    m4.selfUpdater.hasXmrWorker.mockReturnValue(false);
    m4.selfUpdate.planUpdate.mockReturnValue({ updateAvailable: false, reason: 'up-to-date', xmrWorkerUrl: 'https://dl/w' });
    await expect(m4.run(['update'])).resolves.toBe(0);
    expect(m4.selfUpdater.syncXmrWorker).not.toHaveBeenCalled();
  });

  test('points a from-source run at the download instead of updating', async () => {
    const m = load();
    m.selfUpdater.fetchLatestRelease.mockResolvedValue({ version: '9.9.9' });
    m.selfUpdate.planUpdate.mockReturnValue({
      updateAvailable: true, latestVersion: '9.9.9', currentVersion: pkg.version, downloadUrl: 'https://dl/x',
    });
    await expect(m.run(['update'])).resolves.toBe(0);
    expect(allOut()).toContain('running from source');
    expect(allOut()).toContain('https://dl/x');
    expect(m.selfUpdater.applyUpdate).not.toHaveBeenCalled();
  });

  test('applies the update in place when packaged', async () => {
    const m = load();
    m.selfUpdater.fetchLatestRelease.mockResolvedValue({ version: '9.9.9' });
    m.selfUpdater.isPackaged.mockReturnValue(true);
    m.selfUpdate.planUpdate.mockReturnValue({
      updateAvailable: true, latestVersion: '9.9.9', currentVersion: pkg.version, downloadUrl: 'https://dl/x',
    });
    await expect(m.run(['update'])).resolves.toBe(0);
    expect(allOut()).toContain('updated to v9.9.9');
    // `update` takes no flags; PEARL_CORE_VARIANT=amd still says the rig is AMD.
    expect(m.selfUpdater.applyUpdate).toHaveBeenCalledWith(expect.anything(), undefined, { amd: false });
    await withEnv({ PEARL_CORE_VARIANT: 'amd' }, async () => {
      const m2 = load();
      m2.selfUpdater.fetchLatestRelease.mockResolvedValue({ version: '9.9.9' });
      m2.selfUpdater.isPackaged.mockReturnValue(true);
      m2.selfUpdate.planUpdate.mockReturnValue({
        updateAvailable: true, latestVersion: '9.9.9', currentVersion: pkg.version, downloadUrl: 'https://dl/x',
      });
      await expect(m2.run(['update'])).resolves.toBe(0);
      expect(m2.selfUpdater.applyUpdate).toHaveBeenCalledWith(expect.anything(), undefined, { amd: true });
    });
  });

  test('exits 1 when applying the update fails', async () => {
    const m = load();
    m.selfUpdater.fetchLatestRelease.mockResolvedValue({ version: '9.9.9' });
    m.selfUpdater.isPackaged.mockReturnValue(true);
    m.selfUpdater.applyUpdate.mockRejectedValue(new Error('disk full'));
    m.selfUpdate.planUpdate.mockReturnValue({
      updateAvailable: true, latestVersion: '9.9.9', currentVersion: pkg.version, downloadUrl: 'https://dl/x',
    });
    await expect(m.run(['update'])).resolves.toBe(1);
    expect(allErr()).toContain('update failed: disk full');
  });
});

// ── auto-update on start ─────────────────────────────────────────────────────
// Each test uses a run that fails fast after the update phase so it never
// reaches mining. There is no engine to resolve any more, so the fast failure
// is an address the validator rejects.

describe('auto-update on start', () => {
  // A mining run whose engine refuses to start: exits 1 promptly, and — unlike
  // an address the validator rejects — only AFTER the update phase, which is
  // what these tests are about.
  const argvQuick = ['--address', ADDR, '--mode', 'mining'];
  function quick(m) {
    m.PearlEngine.startError = new Error('core not built');
    return m.run(argvQuick);
  }

  test('skips the check entirely in the re-exec child', async () => {
    process.env.LLMJOB_EARN_UPDATED = '1';
    const m = load();
    m.fs.existsSync.mockReturnValue(false);
    await expect(quick(m)).resolves.toBe(1);
    expect(m.selfUpdater.fetchLatestRelease).not.toHaveBeenCalled();
  });

  test('continues when offline (no release)', async () => {
    const m = load();
    m.fs.existsSync.mockReturnValue(false);
    await expect(quick(m)).resolves.toBe(1);
    expect(m.selfUpdater.fetchLatestRelease).toHaveBeenCalled();
    expect(m.selfUpdate.planUpdate).not.toHaveBeenCalled();
  });

  test('continues when already up to date', async () => {
    const m = load();
    m.fs.existsSync.mockReturnValue(false);
    m.selfUpdater.fetchLatestRelease.mockResolvedValue({ version: pkg.version });
    await expect(quick(m)).resolves.toBe(1);
    expect(allOut()).not.toContain('newer release');
  });

  // Only for a rig that mines Monero: a Pearl-only one gets no coin miner it
  // did not ask for.
  test('up to date, a Monero rig fetches a missing CPU worker before it starts', async () => {
    const XMR_ADDR = '87wTMj88rEGV3BFiYXJc2UHeTGmHikMRWAvAskmPgaF3PZxuxonkKaJZsprNmuhM1Aca7P6UbyatGFHkQfUTYgoNN1JxND9';
    const m = load();
    m.fs.existsSync.mockReturnValue(false);
    m.selfUpdater.fetchLatestRelease.mockResolvedValue({ version: pkg.version });
    m.selfUpdater.isPackaged.mockReturnValue(true);
    m.selfUpdate.planUpdate.mockReturnValue({ updateAvailable: false, reason: 'up-to-date', xmrWorkerUrl: 'https://dl/w' });
    m.PearlEngine.startError = new Error('core not built');
    await m.run(['--address', ADDR, '--mode', 'mining', '--xmr-address', XMR_ADDR]);
    expect(m.selfUpdater.syncXmrWorker).toHaveBeenCalledWith('https://dl/w', pkg.version);

    const m2 = load();
    m2.fs.existsSync.mockReturnValue(false);
    m2.selfUpdater.fetchLatestRelease.mockResolvedValue({ version: pkg.version });
    m2.selfUpdater.isPackaged.mockReturnValue(true);
    m2.selfUpdate.planUpdate.mockReturnValue({ updateAvailable: false, reason: 'up-to-date', xmrWorkerUrl: 'https://dl/w' });
    await expect(quick(m2)).resolves.toBe(1);
    expect(m2.selfUpdater.syncXmrWorker).not.toHaveBeenCalled();

    // Nor from source.
    const m3 = load();
    m3.fs.existsSync.mockReturnValue(false);
    m3.selfUpdater.fetchLatestRelease.mockResolvedValue({ version: pkg.version });
    m3.selfUpdate.planUpdate.mockReturnValue({ updateAvailable: false, reason: 'up-to-date', xmrWorkerUrl: 'https://dl/w' });
    m3.PearlEngine.startError = new Error('core not built');
    await m3.run(['--address', ADDR, '--mode', 'mining', '--xmr-address', XMR_ADDR]);
    expect(m3.selfUpdater.syncXmrWorker).not.toHaveBeenCalled();
  });

  test('only mentions a newer release when running from source', async () => {
    const m = load();
    m.fs.existsSync.mockReturnValue(false);
    m.selfUpdater.fetchLatestRelease.mockResolvedValue({ version: '9.9.9' });
    m.selfUpdate.planUpdate.mockReturnValue({ updateAvailable: true, latestVersion: '9.9.9', currentVersion: pkg.version });
    await expect(quick(m)).resolves.toBe(1);
    expect(allOut()).toContain('a newer release is available: v9.9.9');
    expect(m.selfUpdater.applyUpdate).not.toHaveBeenCalled();
  });

  test('updates and re-execs when packaged, returning the child exit code', async () => {
    const m = load();
    m.selfUpdater.fetchLatestRelease.mockResolvedValue({ version: '9.9.9' });
    m.selfUpdater.isPackaged.mockReturnValue(true);
    m.selfUpdater.reexec.mockReturnValue(42);
    m.selfUpdate.planUpdate.mockReturnValue({ updateAvailable: true, latestVersion: '9.9.9', currentVersion: pkg.version });
    await expect(quick(m)).resolves.toBe(42);
    expect(m.selfUpdater.reexec).toHaveBeenCalledWith(argvQuick);
    // A Pearl-only rig: the worker comes along only if one is already there.
    // An NVIDIA one: so does the AMD core.
    expect(m.selfUpdater.applyUpdate).toHaveBeenCalledWith(expect.anything(), undefined, { xmr: false, amd: false });
  });

  // --core amd: the AMD core is this rig's core, so the update pairs it.
  test('a rig run with --core amd updates its AMD core with the binary', async () => {
    await withEnv({ PEARL_CORE_VARIANT: undefined }, async () => {
      const m = load();
      m.selfUpdater.fetchLatestRelease.mockResolvedValue({ version: '9.9.9' });
      m.selfUpdater.isPackaged.mockReturnValue(true);
      m.selfUpdater.reexec.mockReturnValue(0);
      m.selfUpdate.planUpdate.mockReturnValue({ updateAvailable: true, latestVersion: '9.9.9', currentVersion: pkg.version });
      await expect(m.run(argvQuick.concat(['--core', 'amd']))).resolves.toBe(0);
      expect(m.selfUpdater.applyUpdate).toHaveBeenCalledWith(expect.anything(), undefined, { xmr: false, amd: true });
      // The flag is what the core loader reads too.
      expect(process.env.PEARL_CORE_VARIANT).toBe('amd');
    });
  });

  // An install of the bare binary has no AMD core; asking for it fetches it.
  test('up to date, a rig run with --core amd fetches a missing AMD core before it starts', async () => {
    await withEnv({ PEARL_CORE_VARIANT: undefined }, async () => {
      const m = load();
      m.fs.existsSync.mockReturnValue(false);
      m.selfUpdater.fetchLatestRelease.mockResolvedValue({ version: pkg.version });
      m.selfUpdater.isPackaged.mockReturnValue(true);
      m.selfUpdater.hasAmdCore.mockReturnValue(false);
      m.selfUpdater.fetchAmdCore.mockResolvedValue('/opt/rig/pearl_core_hip.node');
      m.selfUpdate.planUpdate.mockReturnValue({ updateAvailable: false, reason: 'up-to-date', coreHipUrl: 'https://dl/hip' });
      m.PearlEngine.startError = new Error('core not built');
      await m.run(argvQuick.concat(['--core', 'amd']));
      expect(m.selfUpdater.fetchAmdCore).toHaveBeenCalledWith('https://dl/hip');
      expect(allOut()).toContain('fetched /opt/rig/pearl_core_hip.node (it was missing)');
    });

    // A failed fetch says why and carries on to the loader, which says what is missing.
    await withEnv({ PEARL_CORE_VARIANT: 'amd' }, async () => {
      const m = load();
      m.fs.existsSync.mockReturnValue(false);
      m.selfUpdater.fetchLatestRelease.mockResolvedValue({ version: pkg.version });
      m.selfUpdater.isPackaged.mockReturnValue(true);
      m.selfUpdater.hasAmdCore.mockReturnValue(false);
      m.selfUpdater.fetchAmdCore.mockRejectedValue(new Error('HTTP 404'));
      m.selfUpdate.planUpdate.mockReturnValue({ updateAvailable: false, reason: 'up-to-date', coreHipUrl: 'https://dl/hip' });
      await expect(quick(m)).resolves.toBe(1);
      expect(allErr()).toContain('could not fetch pearl_core_hip.node: HTTP 404');
    });

    // One already there, an NVIDIA rig, or a run from source: nothing is fetched.
    for (const [variant, packaged, has] of [['amd', true, true], [undefined, true, false], ['amd', false, false]]) {
      await withEnv({ PEARL_CORE_VARIANT: variant }, async () => {
        const m = load();
        m.fs.existsSync.mockReturnValue(false);
        m.selfUpdater.fetchLatestRelease.mockResolvedValue({ version: pkg.version });
        m.selfUpdater.isPackaged.mockReturnValue(packaged);
        m.selfUpdater.hasAmdCore.mockReturnValue(has);
        m.selfUpdate.planUpdate.mockReturnValue({ updateAvailable: false, reason: 'up-to-date', coreHipUrl: 'https://dl/hip' });
        await expect(quick(m)).resolves.toBe(1);
        expect(m.selfUpdater.fetchAmdCore).not.toHaveBeenCalled();
      });
    }
  });

  test('keeps mining on the old version when the auto-update fails', async () => {
    const m = load();
    m.fs.existsSync.mockReturnValue(false);
    m.selfUpdater.fetchLatestRelease.mockResolvedValue({ version: '9.9.9' });
    m.selfUpdater.isPackaged.mockReturnValue(true);
    m.selfUpdater.applyUpdate.mockRejectedValue(new Error('nope'));
    m.selfUpdate.planUpdate.mockReturnValue({ updateAvailable: true, latestVersion: '9.9.9', currentVersion: pkg.version });
    await expect(quick(m)).resolves.toBe(1);
    expect(allErr()).toContain('auto-update failed (nope)');
  });
});

// ── mining runs ──────────────────────────────────────────────────────────────

// Run `fn` with these environment variables set (undefined removes one), and
// put them back afterwards. The CLI reads CUDA_VISIBLE_DEVICES at load, so set
// it before load().
async function withEnv(vars, fn) {
  const saved = {};
  for (const k of Object.keys(vars)) {
    saved[k] = Object.prototype.hasOwnProperty.call(process.env, k) ? process.env[k] : undefined;
    if (vars[k] === undefined) delete process.env[k];
    else process.env[k] = vars[k];
  }
  try {
    return await fn();
  } finally {
    for (const k of Object.keys(saved)) {
      if (saved[k] === undefined) delete process.env[k];
      else process.env[k] = saved[k];
    }
  }
}

describe('mining', () => {
  test('full auto-detected run: report, stats file, SIGINT shutdown', async () => {
    const m = load();
    m.probe.detectGpuInfo.mockResolvedValue({ name: 'NVIDIA GeForce RTX 3070', count: 2 });
    m.probe.detectMinerGpus.mockResolvedValue([
      { index: 0, name: 'NVIDIA GeForce RTX 3070' },
      { index: 1, name: 'NVIDIA GeForce RTX 3070' },
    ]);
    const p = m.run(['--address', ADDR, '--mdl', MDL, '--no-update',
      '--stats-file', '/tmp/s.json']);
    await settle();

    // Auto-detected knobs: region, hostname worker, GPU name.
    // The default mode is 'auto', so a bare run mines AND serves the LLM.
    expect(allOut()).toContain('mode:       auto  (default)');
    expect(allOut()).toContain('preparing local LLM (Gemma-4-E4B-it-Q4_K_M) …');
    expect(allOut()).toContain('local LLM starting on 1 GPU [auto]');
    expect(allOut()).toContain('worker:     rig-host  (auto)');
    expect(allOut()).toContain('(+MDL');
    expect(allOut()).toContain('gpu:        2× NVIDIA GeForce RTX 3070  (auto)');
    // What will actually mine. Both cards do, one core each.
    expect(allOut()).toContain('mining on:  2 GPUs [0, 1]');

    const miner = m.PearlEngine.instances[0];
    expect(miner.start).toHaveBeenCalledWith(expect.objectContaining({
      endpoint: 'us.pearl.herominers.com:1200',
      gpus: [
        { index: 0, name: 'NVIDIA GeForce RTX 3070' },
        { index: 1, name: 'NVIDIA GeForce RTX 3070' },
      ],
    }));
    miner.emit('log', { line: 'hello', level: 'info' });
    miner.emit('log', { line: 'bad', level: 'error' });
    // The throttle reads the clock, so pin it. With the real clock, a busy Windows
    // runner that stalled a second between the two emits below printed two lines.
    const t0 = Date.now();
    const nowSpy = jest.spyOn(Date, 'now').mockReturnValue(t0);
    miner.emit('event', { type: 'status', hashrate: 3.2, accepted: 5, rejected: 1 });
    // Each card ticks about twice a second, so the rig line is throttled: this
    // second status is folded into the totals but writes no second line.
    miner.emit('event', { type: 'status', gpuIndex: 1, hashrate: 1, accepted: 0, rejected: 0 });
    miner.emit('event', { type: 'connected', gpu: 'RTX 3070' });
    miner.emit('error', new Error('boom'));
    expect(allOut()).toContain('⛏  3.2 TH/s · 5 accepted · 1 rejected');
    expect(allOut().match(/⛏/g)).toHaveLength(1);
    // A second later the line prints the rig rate averaged over that second (card 0
    // at 3.2 and card 1 at 1 the whole time: 4.2), not the latest window alone.
    nowSpy.mockReturnValue(t0 + 1500);
    miner.emit('event', { type: 'status', gpuIndex: 1, hashrate: 1, accepted: 0, rejected: 0 });
    nowSpy.mockRestore();
    expect(allOut().match(/⛏/g)).toHaveLength(2);
    expect(allOut()).toMatch(/⛏ {2}4\.2 TH\/s/);
    expect(allErr()).toContain('bad');
    expect(allErr()).toContain('engine error: boom');

    // Reporter interval posts one row per card; stats interval writes the file.
    const reporter = intervalFor(NETWORK.reportIntervalMs);
    await reporter.fn();
    expect(m.probe.postMinerReport).toHaveBeenCalled();
    const statsWriter = intervalFor(10000);
    statsWriter.fn();
    expect(m.fs.writeFileSync).toHaveBeenCalledWith('/tmp/s.json.tmp', expect.any(String));
    expect(m.fs.renameSync).toHaveBeenCalledWith('/tmp/s.json.tmp', '/tmp/s.json');

    // mode auto: the LLM comes up once the miner proves real hashrate, so by
    // here BOTH are running and shutdown has to stop both.
    await settle();
    expect(m.LlmManager.instances.length).toBeGreaterThan(0);

    fire('SIGINT');
    fire('SIGINT'); // second signal hits the `stopping` guard
    expect(m.LlmManager.instances[0].stop).toHaveBeenCalled();
    expect(miner.stop).toHaveBeenCalled();
    miner.emit('stopped', 0);
    miner.emit('stopped', 0); // second emit hits the `settled` guard
    await expect(p).resolves.toBe(0);
    expect(allOut()).toContain('shutting down…');
    expect(allOut()).toContain('engine exited (code 0)');
  });

  test('explicit knobs, TTY prefix, engine exit code passthrough', async () => {
    process.stdout.isTTY = true;
    const m = load();
    const p = m.run(['-a', ADDR, '-r', 'de', '-w', 'rig9', '-g', 'RTX 4090',
      '--no-update']);
    await settle();

    expect(m.probe.detectRegion).not.toHaveBeenCalled();
    expect(m.cp.execFile).not.toHaveBeenCalled();
    expect(/\[\d{2}:\d{2}:\d{2}\] /.test(allOut())).toBe(true);
    expect(allOut()).toContain('worker:     rig9');
    expect(allOut()).toContain('gpu:        RTX 4090');

    const miner = m.PearlEngine.instances[0];
    expect(miner.start).toHaveBeenCalledWith(expect.objectContaining({ endpoint: 'de.pearl.herominers.com:1200' }));
    miner.emit('stopped', 5); // engine died on its own → exit code passes through
    await expect(p).resolves.toBe(5);
  });

  // An old HiveOS flight sheet's --region eu1 used to stop the CLI at once.
  test('an old AlphaPool region mines in the region it maps to, and says so', async () => {
    const m = load();
    const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--region', 'eu1']);
    await settle();
    expect(allOut()).toContain('region eu1 is now de (Europe · Germany)');
    expect(allOut()).toContain('pool:       de.pearl.herominers.com:1200');
    expect(m.probe.detectRegion).not.toHaveBeenCalled();
    expect(m.PearlEngine.instances[0].start).toHaveBeenCalledWith(
      expect.objectContaining({ region: 'de', endpoint: 'de.pearl.herominers.com:1200' }));
    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  // --pool: the Salad watcher points boxes at Kryptex this way.
  test('--pool mines on that endpoint and skips region detection', async () => {
    const m = load();
    const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--pool', 'prl-us.kryptex.network:7048']);
    await settle();
    expect(allOut()).toContain('pool:       prl-us.kryptex.network:7048  (--pool)');
    expect(m.probe.detectRegion).not.toHaveBeenCalled();
    expect(m.PearlEngine.instances[0].start).toHaveBeenCalledWith(
      expect.objectContaining({ endpoint: 'prl-us.kryptex.network:7048' }));
    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  // Kryptex has no merge mining: the miner leaves the MDL address out of the
  // login there, so the startup line doesn't name it either.
  test('--pool on Kryptex does not claim an MDL address it will not send', async () => {
    const m = load();
    const p = m.run(['-a', ADDR, '--mdl', MDL, '--mode', 'mining', '--no-update', '--pool', 'prl-us.kryptex.network:7048']);
    await settle();
    expect(allOut()).toContain('address:    ');
    expect(allOut()).not.toContain('(+MDL');
    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('--pool with Monero still detects a region, for the Monero host', async () => {
    const XMR_ADDR = '87wTMj88rEGV3BFiYXJc2UHeTGmHikMRWAvAskmPgaF3PZxuxonkKaJZsprNmuhM1Aca7P6UbyatGFHkQfUTYgoNN1JxND9';
    const m = load();
    const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--pool', 'prl-us.kryptex.network:7048',
      '--xmr-address', XMR_ADDR]);
    await settle();
    expect(m.probe.detectRegion).toHaveBeenCalled();
    expect(allOut()).toContain('pool:       prl-us.kryptex.network:7048  (--pool)');
    m.PearlEngine.instances[0].emit('stopped', 0);
    await p;
  });

  test('an unknown region still exits 1 with the list', async () => {
    const m = load();
    await expect(m.run(['-a', ADDR, '--no-update', '--region', 'eu9'])).resolves.toBe(1);
    expect(allErr()).toContain('error: unknown region: eu9 (choices: us, us2, ca, br, de, fi, fr, tr, sg, hk, kr, au)');
    expect(allOut()).not.toContain('is now');
    expect(m.PearlEngine.instances).toHaveLength(0);
  });

  // CUDA_VISIBLE_DEVICES=0 hid a rig's second card from the mining core while
  // nvidia-smi still listed it. The CLI removes it at load and says so.
  test('clears CUDA_VISIBLE_DEVICES at load and logs it for a mining run', async () => {
    await withEnv({ CUDA_VISIBLE_DEVICES: '0' }, async () => {
      const m = load();
      expect(process.env.CUDA_VISIBLE_DEVICES).toBeUndefined();
      const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update']);
      await settle();
      expect(allOut()).toContain(
        'ignoring CUDA_VISIBLE_DEVICES=0 so every GPU can mine (use --gpu-index to choose the cards)');
      expect(m.probe.detectMinerGpus).toHaveBeenCalledWith(process.env, null);
      m.PearlEngine.instances[0].emit('stopped', 0);
      await expect(p).resolves.toBe(0);
    });
  });

  // HiveOS only ever writes card numbers there, so anything else (a UUID, a MIG
  // id) gets no --gpu-index from h-config.sh. The log has to say it was ignored.
  test('a CUDA_VISIBLE_DEVICES of GPU UUIDs is ignored, and the log says so', async () => {
    const uuid = 'GPU-8f0e2b1c-55aa-4c3d-9e1f-0123456789ab';
    await withEnv({ CUDA_VISIBLE_DEVICES: uuid }, async () => {
      const m = load();
      const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update']);
      await settle();
      expect(allOut()).toContain('ignoring CUDA_VISIBLE_DEVICES=' + uuid + ' so every GPU can mine');
      m.PearlEngine.instances[0].emit('stopped', 0);
      await p;
    });
  });

  // net.connect takes the PORT first and PearlMiner hands over host first, so
  // the adapter flips them. Backwards, the CLI fails to connect with an error
  // that names neither side.
  test('the injected connect passes host and port to net in the right order', async () => {
    const m = load();
    const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update']);
    await settle();
    m.net.connect.mockClear();
    m.PearlEngine.instances[0].opts.connect('pool.example', 1200);
    expect(m.net.connect).toHaveBeenCalledWith({ port: 1200, host: 'pool.example', lookup: expect.any(Function) });
    m.PearlEngine.instances[0].emit('stopped', 0);
    await p;
  });

  // The pool login names the miner, the same string the Monero miner sends.
  test('gives the engine the CLI\'s agent string for the pool login', async () => {
    const m = load();
    const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update']);
    await settle();
    expect(m.PearlEngine.instances[0].opts.agent).toBe('llmjob-earn/' + pkg.version);
    m.PearlEngine.instances[0].emit('stopped', 0);
    await p;
  });

  // The GPU keeps working on its last job while the pool connection is being
  // reopened, but the pool gets nothing, so neither should the board.
  test('reports no hashrate to the board while the pool is not connected', async () => {
    const m = load();
    const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update']);
    await settle();
    const miner = m.PearlEngine.instances[0];
    miner.emit('event', { type: 'status', hashrate: 120 });
    const reporter = intervalFor(NETWORK.reportIntervalMs);

    m.probe.postMinerReport.mockClear();
    await reporter.fn();
    expect(m.probe.postMinerReport.mock.calls.map((c) => c[0].hashrate)).toEqual([120]);

    miner.poolConnected.mockReturnValue(false);
    m.probe.postMinerReport.mockClear();
    await reporter.fn();
    expect(m.probe.postMinerReport.mock.calls.map((c) => c[0].hashrate)).toEqual([0]);

    miner.emit('stopped', 0);
    await p;
  });

  // RAM and the Monero hashrate belong to the machine, not a card, so every
  // card's row carries the same values. Without --xmr-address there is no CPU
  // miner, and its hashrate is null (unknown), not 0.
  test('every board row carries the machine\'s RAM, and no Monero hashrate without --xmr-address', async () => {
    // No container memory limit, whatever the machine running the tests has, so
    // the mocked os figures are the RAM.
    jest.spyOn(process, 'constrainedMemory').mockReturnValue(0);
    const m = load();
    m.probe.detectGpusVram.mockResolvedValue([
      { index: 0, name: 'RTX 4090', usedMb: 1000, totalMb: 24564 },
      { index: 1, name: 'RTX 4090', usedMb: 1000, totalMb: 24564 },
    ]);
    const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update']);
    await settle();
    m.probe.postMinerReport.mockClear();
    await intervalFor(NETWORK.reportIntervalMs).fn();
    expect(m.probe.postMinerReport.mock.calls.map((c) => [c[0].worker, c[0].ramUsedMb, c[0].ramTotalMb, c[0].xmrHashrate]))
      .toEqual([
        ['rig-host/gpu0', 12288, 32768, null],
        ['rig-host/gpu1', 12288, 32768, null],
      ]);

    m.PearlEngine.instances[0].emit('stopped', 0);
    await p;
  });

  // HiveOS reads the stats file, and it must agree with the board. The real
  // engine also answers poolConnected() false once it has stopped, which is how
  // demand mode leaves it while the LLM serves. The file used to keep the last
  // hashrate through both.
  test('the stats file reports no hashrate while the pool hears none', async () => {
    const m = load();
    const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--stats-file', '/tmp/s.json']);
    await settle();
    const miner = m.PearlEngine.instances[0];
    miner.emit('event', { type: 'status', gpuIndex: 0, hashrate: 120, accepted: 4 });
    const written = () => {
      m.fs.writeFileSync.mockClear();
      intervalFor(10000).fn();
      const w = JSON.parse(m.fs.writeFileSync.mock.calls[0][1]);
      return [w.ths, w.accepted, w.gpus.map((g) => g.hashrate)];
    };
    expect(written()).toEqual([120, 4, [120]]);

    miner.poolConnected.mockReturnValue(false);
    expect(written()).toEqual([0, 4, [0]]);

    miner.emit('stopped', 0);
    await p;
  });

  // A failed system lookup falls back (shared/poolLookup), and the CLI says
  // which address it used. It saves the address once the pool takes the login
  // there, not before.
  test('finds the pool by a fallback when the system lookup fails, and says so', async () => {
    const m = load();
    const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update']);
    await settle();
    m.net.connect.mockClear();
    m.PearlEngine.instances[0].opts.connect('pool.example', 1200);
    const { lookup } = m.net.connect.mock.calls[0][0];
    const dns = require('dns');
    const fail = jest.spyOn(dns, 'lookup').mockImplementation((h, o, cb) => cb(Object.assign(new Error('no data'), { code: 'ENOENT' })));
    const resolver = jest.spyOn(dns.promises, 'Resolver').mockImplementation(() => ({
      setServers: jest.fn(),
      resolve4: jest.fn(() => Promise.resolve(['203.0.113.7'])),
    }));
    const write = jest.spyOn(require('fs'), 'writeFileSync').mockImplementation(() => {});
    const mkdir = jest.spyOn(require('fs'), 'mkdirSync').mockImplementation(() => {});
    try {
      const got = await new Promise((resolve) => lookup('pool.example', {}, (err, address) => resolve({ err, address })));
      expect(got).toEqual({ err: null, address: '203.0.113.7' });
      expect(allOut()).toContain('could not look up pool.example (ENOENT); using 203.0.113.7 from DNS');
      const saved = () => m.fs.writeFileSync.mock.calls.filter((c) => String(c[0]).endsWith('pool-addresses.json'));
      expect(saved()).toEqual([]);
      m.PearlEngine.instances[0].emit('authorized', { host: 'pool.example', address: '203.0.113.7', family: 'IPv4' });
      expect(JSON.parse(saved()[0][1])).toEqual({ 'pool.example': [{ address: '203.0.113.7', family: 4 }] });
    } finally {
      fail.mockRestore();
      resolver.mockRestore();
      write.mockRestore();
      mkdir.mockRestore();
    }
    m.PearlEngine.instances[0].emit('stopped', 0);
    await p;
  });

  // The writer is unrefed, so it never holds the process open on its own.
  //
  // A stats file that cannot be written must also stay silent: it is a
  // convenience for rig dashboards, not something worth killing a miner over.
  test('the stats writer is unrefed, and a write failure is silent', async () => {
    const m = load();
    const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--stats-file', '/tmp/s.json']);
    await settle();
    m.PearlEngine.instances[0].emit('event', { type: 'status', hashrate: 1, accepted: 1 });
    m.fs.writeFileSync.mockImplementation(() => { throw new Error('read-only /tmp'); });
    expect(intervalFor(10000).unref).toHaveBeenCalled();
    intervalFor(10000).fn(); // must not throw

    // Shut down by SIGNAL with mining ONLY: the shutdown path has to cope with
    // no LLM to stop, which is a different branch from the co-run case.
    fire('SIGTERM');
    expect(m.PearlEngine.instances[0].stop).toHaveBeenCalled();
    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
    expect(allErr()).not.toContain('read-only /tmp');
  });

  test('a miner that fails to launch resolves 1', async () => {
    const m = load();
    m.PearlEngine.startError = new Error('EACCES');
    const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update']);
    await expect(p).resolves.toBe(1);
    expect(allErr()).toContain('failed to launch engine: EACCES');
  });

  // The v0.4.1 field failure: a packaged CLI with no pearl_core.node exited 0,
  // which systemd read as success — a silent ten-second restart loop that
  // mined nothing. The contract now: 'mining' with no core is a non-zero exit
  // that says what is missing and how to point at it; 'auto' says the same but
  // keeps its LLM half alive instead of looping.
  test('mining mode with no loadable core exits 1 and names PEARL_CORE_PATH', async () => {
    const m = load();
    m.pearlCore.coreFactory.mockReturnValue(null);
    await expect(m.run(['-a', ADDR, '--mode', 'mining', '--no-update'])).resolves.toBe(1);
    expect(allErr()).toContain('pearl_core.node not found');
    expect(allErr()).toContain('PEARL_CORE_PATH');
    expect(m.PearlEngine.instances.length).toBe(0);
  });

  // The AMD core is a file of its own. Telling an AMD rig to fetch
  // pearl_core.node would load a CUDA core on it.
  // The core loader has already said what failed (pearlCore.test.js); the CLI
  // adds where the AMD core goes, once, and never sends an AMD rig to fetch the
  // CUDA one.
  const AMD_FIX = 'Fix: put pearl_core_hip.node from the release next to the executable (a run from the source tree also finds earn/native/amd/build), or set PEARL_CORE_PATH to its absolute path.';
  // The loader has said why a core that is there won't load (a missing
  // library, too old a glibc); "not found" or "put it there" would contradict it.
  const WOULD_NOT_LOAD = (file) => (opts) => {
    opts.log('error', 'Pearl core: ' + file + ' is there but would not load: libamdhip64.so.7: cannot open shared object file');
    return null;
  };
  test('a core that is there but will not load gets no "not found" and no "put it there"', async () => {
    await withEnv({ PEARL_CORE_VARIANT: 'amd', PEARL_CORE_PATH: undefined }, async () => {
      const m = load();
      m.pearlCore.coreFactory.mockImplementation(WOULD_NOT_LOAD('/opt/rig/pearl_core_hip.node'));
      await expect(m.run(['-a', ADDR, '--mode', 'mining', '--no-update'])).resolves.toBe(1);
      expect(allErr()).toContain('is there but would not load');
      expect(allErr()).not.toContain('Fix: put');
    });
    await withEnv({ PEARL_CORE_VARIANT: undefined, PEARL_CORE_PATH: undefined }, async () => {
      const m = load();
      m.pearlCore.coreFactory.mockImplementation(WOULD_NOT_LOAD('/opt/rig/pearl_core.node'));
      await expect(m.run(['-a', ADDR, '--mode', 'mining', '--no-update'])).resolves.toBe(1);
      expect(allErr()).not.toContain('pearl_core.node not found');
      expect(allErr()).not.toContain('Fix: keep');
    });
  });

  test('--core amd asks for the AMD core, over PEARL_CORE_VARIANT', async () => {
    await withEnv({ PEARL_CORE_VARIANT: 'cu12', PEARL_CORE_PATH: undefined }, async () => {
      const m = load();
      m.pearlCore.coreFactory.mockReturnValue(null);
      await expect(m.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--core', 'amd'])).resolves.toBe(1);
      expect(allErr()).toContain(AMD_FIX);
      expect(process.env.PEARL_CORE_VARIANT).toBe('amd');
    });
  });

  test('with PEARL_CORE_VARIANT=amd and no AMD core, it says where the AMD core goes', async () => {
    await withEnv({ PEARL_CORE_VARIANT: ' AMD ', PEARL_CORE_PATH: undefined }, async () => {
      const m = load();
      m.pearlCore.coreFactory.mockReturnValue(null);
      await expect(m.run(['-a', ADDR, '--mode', 'mining', '--no-update'])).resolves.toBe(1);
      expect(allErr()).toContain(AMD_FIX);
      expect(allErr()).not.toContain('pearl_core.node');
    });
  });

  // Only the named file is tried, so "put it next to the executable" can't help.
  test('with PEARL_CORE_VARIANT=amd and a PEARL_CORE_PATH that did not load, no "put it there", and auto serves the LLM', async () => {
    await withEnv({ PEARL_CORE_VARIANT: 'amd', PEARL_CORE_PATH: '/opt/amd/pearl_core_hip.nod' }, async () => {
      const m = load();
      m.pearlCore.coreFactory.mockReturnValue(null);
      const p = m.run(['-a', ADDR, '--mode', 'auto', '--no-update']);
      await settle();
      expect(allErr()).not.toContain('Fix: put');
      expect(allErr()).not.toContain('pearl_core.node');
      expect(allErr()).toContain('continuing with the local LLM only.');
      fire('SIGINT');
      await expect(p).resolves.toBe(0);
    });
  });

  // Which build of the core loads is the factory's call (pearlCore.test.js); the
  // CLI's part is handing it the mining cards and nvidia-smi's answer, and
  // printing what it says -- an error on stderr, anything else on stdout.
  test('hands the core factory the mining cards and their compute capability', async () => {
    const m = load();
    const cards = [{ index: 0, major: 12, minor: 0, driverMajor: 610 }];
    m.probe.detectMinerGpus.mockResolvedValue([{ index: 0, name: 'NVIDIA GeForce RTX 5090' }]);
    m.probe.detectCudaCards.mockResolvedValue(cards);
    m.pearlCore.coreFactory.mockReturnValue(null);
    await expect(m.run(['-a', ADDR, '--mode', 'mining', '--no-update'])).resolves.toBe(1);
    const opts = m.pearlCore.coreFactory.mock.calls[0][0];
    expect(opts).toMatchObject({ gpus: [{ index: 0, name: 'NVIDIA GeForce RTX 5090' }], cards });
    opts.log('info', 'Pearl core: CUDA 13 build · driver 610, mining card is compute 12.0');
    opts.log('warn', 'Pearl core: CUDA 12.8 build · the CUDA 13 build could not start (x)');
    opts.log('error', 'core trouble');
    expect(allOut()).toContain('Pearl core: CUDA 13 build · driver 610, mining card is compute 12.0');
    expect(allOut()).toContain('the CUDA 13 build could not start (x)');
    expect(allErr()).toContain('core trouble');
    expect(allOut()).not.toContain('core trouble');
  });

  test('auto mode with no loadable core serves the LLM instead of looping', async () => {
    const m = load();
    m.pearlCore.coreFactory.mockReturnValue(null);
    const p = m.run(['-a', ADDR, '--mode', 'auto', '--no-update']);
    await settle();
    expect(allErr()).toContain('pearl_core.node not found');
    expect(allErr()).toContain('continuing with the local LLM only.');
    expect(m.PearlEngine.instances.length).toBe(0);
    // The LLM half is alive; shut it down the way an operator would.
    fire('SIGINT');
    await expect(p).resolves.toBe(0);
  });
});

// ── Monero on the CPU (--xmr-address) ────────────────────────────────────────

describe('CPU mining', () => {
  const XMR_ADDR = '87wTMj88rEGV3BFiYXJc2UHeTGmHikMRWAvAskmPgaF3PZxuxonkKaJZsprNmuhM1Aca7P6UbyatGFHkQfUTYgoNN1JxND9';

  test('mines beside the GPU on all CPUs but one, logs its numbers, and stops with the run', async () => {
    const m = load();
    const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--xmr-address', XMR_ADDR,
      '--stats-file', '/tmp/s.json']);
    await settle();
    expect(allOut()).toContain('cpu:        Monero to 87wTMj88…1JxND9 · us.monero.herominers.com:1111');
    expect(allOut()).toContain('cpu:        15 of 16 CPUs  (auto)');
    expect(m.xmrWorker.findWorker).toHaveBeenCalledWith(expect.objectContaining({ execPath: process.execPath }));
    const cpu = m.XmrEngine.instances[0];
    expect(cpu.opts).toMatchObject({ workerPath: '/opt/llmjob-xmr-worker', spawnWorker: 'spawnWorker', agent: 'llmjob-earn/' + pkg.version });
    expect(cpu.start).toHaveBeenCalledWith({
      address: XMR_ADDR, worker: 'rig-host', endpoint: 'us.monero.herominers.com:1111', threads: 15, priority: null,
    });

    // Its pool socket goes through the same lookup as the GPU miner's, and a
    // login that worked is saved to the same file.
    m.PearlEngine.instances[0].opts.connect('us.pearl.herominers.com', 1200);
    cpu.opts.connect('de.monero.herominers.com', 1111);
    expect(m.net.connect).toHaveBeenLastCalledWith({ port: 1111, host: 'de.monero.herominers.com', lookup: expect.any(Function) });
    expect(m.net.connect.mock.calls[0][0].lookup).toBe(m.net.connect.mock.calls[1][0].lookup);
    cpu.emit('authorized', { host: 'de.monero.herominers.com', address: '141.95.126.31', family: 'IPv4' });
    expect(m.fs.writeFileSync).toHaveBeenCalledWith(expect.stringMatching(/pool-addresses\.json$/), expect.stringContaining('141.95.126.31'));

    cpu.emit('log', { level: 'info', line: 'cpu: logged in to the pool' });
    cpu.emit('log', { level: 'error', line: 'cpu: XMR share rejected: [-1] Low difficulty share' });
    // The engine logs why it gave up; its 'error' is not logged again.
    cpu.emit('log', { level: 'error', line: 'cpu: the CPU miner stopped (exit code 1)' });
    cpu.emit('error', new Error('the CPU miner stopped (exit code 1)'));
    const t0 = Date.now();
    const nowSpy = jest.spyOn(Date, 'now').mockReturnValue(t0);
    cpu.emit('event', { type: 'status', state: 'dataset', hashrate: 0, accepted: 0, rejected: 0, threads: 15 });
    cpu.emit('event', { type: 'status', state: 'mining', hashrate: 0, accepted: 0, rejected: 0, threads: 15 });
    cpu.emit('event', { type: 'connect-failed', reason: 'x' });
    cpu.emit('event', { type: 'status', state: 'mining', hashrate: 1845.2, accepted: 1234, rejected: 1, threads: 15 });
    cpu.emit('event', { type: 'status', state: 'mining', hashrate: 1900, accepted: 1235, rejected: 1, threads: 15 });
    nowSpy.mockReturnValue(t0 + 30000);
    cpu.emit('event', { type: 'status', state: 'mining', hashrate: 1900, accepted: 1235, rejected: 1, threads: 15 });
    nowSpy.mockRestore();
    expect(allOut()).toContain('cpu: logged in to the pool');
    expect(allErr()).toContain('cpu: XMR share rejected: [-1] Low difficulty share');
    expect(allErr().match(/cpu: the CPU miner stopped \(exit code 1\)/g)).toHaveLength(1);
    expect(allOut()).toContain('cpu: 1.85 kH/s · 1,234 accepted · 1 rejected · 15 threads');
    expect(allOut()).toContain('cpu: 1.90 kH/s · 1,235 accepted · 1 rejected · 15 threads');
    expect(allOut().match(/cpu: \d+\.\d+ kH\/s/g)).toHaveLength(2);

    // The stats file carries it, for HiveOS's second algorithm.
    intervalFor(10000).fn();
    const written = JSON.parse(m.fs.writeFileSync.mock.calls.filter((c) => c[0] === '/tmp/s.json.tmp').pop()[1]);
    expect(written.xmr).toEqual({ algo: 'randomx', hs: 1845.2, accepted: 3, rejected: 0, threads: 15, state: 'mining' });

    // With the GPU miner running, the CPU miner giving up does not end the run.
    cpu.emit('stopped', 1);
    expect(cpu.stop).not.toHaveBeenCalled();
    fire('SIGINT');
    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
    expect(cpu.stop).toHaveBeenCalled();
    cpu.emit('stopped', 0);   // the stop it was asked for
  });

  // The board's Monero figure, on every row: the engine's H/s while it runs,
  // null once it has given up, so a CPU miner that stopped does not read as one
  // mining at 0.
  test('the board rows carry its hashrate while it runs, and null once it stops', async () => {
    const m = load();
    const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--xmr-address', XMR_ADDR]);
    await settle();
    const cpu = m.XmrEngine.instances[0];
    const reporter = intervalFor(NETWORK.reportIntervalMs);
    const xmrOf = () => m.probe.postMinerReport.mock.calls.map((c) => c[0].xmrHashrate);

    m.probe.postMinerReport.mockClear();
    await reporter.fn();
    expect(xmrOf()).toEqual([1845.2]);

    cpu.running = false;   // it gave up on its own
    m.probe.postMinerReport.mockClear();
    await reporter.fn();
    expect(xmrOf()).toEqual([null]);

    fire('SIGINT');
    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('a thread count by hand, and one cut down to leave a CPU free', async () => {
    const m = load();
    m.run(['-a', ADDR, '--mode', 'mining', '--no-update', '-w', 'rig9', '--xmr-address', XMR_ADDR, '--cpu-threads', '3',
      '--cpu-priority', 'normal']);
    await settle();
    expect(allOut()).toContain('cpu:        3 of 16 CPUs\n');
    expect(m.XmrEngine.instances[0].start).toHaveBeenCalledWith(expect.objectContaining({ threads: 3, worker: 'rig9', priority: 'normal' }));

    const m2 = load();
    m2.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--xmr-address', XMR_ADDR, '--cpu-threads', '64']);
    await settle();
    expect(allOut()).toContain('cpu:        15 of 16 CPUs  (64 threads asked for; running 15 so one of the 16 CPUs stays free for the GPU miner)');

    // The default, capped by L3 cache, says why.
    const m3 = load();
    m3.cpuFacts.readCpuFacts.mockReturnValue({ cpus: 12, memoryBytes: 16 * 2 ** 30, l3Bytes: 12 * 2 ** 20 });
    m3.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--xmr-address', XMR_ADDR]);
    await settle();
    expect(allOut()).toContain('cpu:        6 of 12 CPUs  (auto: one thread per 2 MB of L3 cache, and this CPU has 12 MB;'
      + ' more threads add power, not hashrate)');
  });

  test('off: by --cpu-threads 0, by the plan, or with no worker beside the binary', async () => {
    const m = load();
    m.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--xmr-address', XMR_ADDR, '--cpu-threads', '0']);
    await settle();
    expect(allOut()).toContain('cpu:        off (--cpu-threads 0)');
    expect(m.XmrEngine.instances).toHaveLength(0);

    const m2 = load();
    m2.cpuFacts.readCpuFacts.mockReturnValue({ cpus: 4, memoryBytes: 16 * 2 ** 30 });
    m2.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--xmr-address', XMR_ADDR]);
    await settle();
    expect(allOut()).toContain('cpu:        off: off by default with 4 CPUs (it turns on by itself above 4 CPUs and 8 GB); set a thread count to mine anyway');
    expect(m2.XmrEngine.instances).toHaveLength(0);

    const m3 = load();
    m3.xmrWorker.findWorker.mockReturnValue(null);
    m3.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--xmr-address', XMR_ADDR]);
    await settle();
    expect(allErr()).toContain('cpu:        off: llmjob-xmr-worker is not beside the executable. Keep it from the release next to it, or set LLMJOB_XMR_WORKER=/path/to/llmjob-xmr-worker.');
    expect(m3.XmrEngine.instances).toHaveLength(0);
  });

  // With nothing else running (here the LLM had no room), the CPU miner is the
  // run: a stop it made on its own ends it non-zero, so systemd restarts it.
  // The CPU miner kept the run going, so an LLM that never started exited 0
  // with the GPU idle.
  test('it does not cover for an LLM that failed: that still exits 1', async () => {
    const m = load();
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'A4000', usedMb: 7000, totalMb: 8000 }]);
    await expect(m.run(['--mode', 'llm', '--no-update', '--xmr-address', XMR_ADDR])).resolves.toBe(1);
    expect(allErr()).toContain('nothing to run — no miner and the LLM did not start');
    expect(m.XmrEngine.instances[0].start).not.toHaveBeenCalled();
  });

  test('with every GPU off, it is the run: it ends the run when it gives up, and a signal stops it', async () => {
    const m = load();
    const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--gpu-index=none', '--xmr-address', XMR_ADDR]);
    await settle();
    expect(allOut()).toContain('no GPU mines (every one is turned off); mining Monero on the CPU');
    expect(allOut()).not.toContain('nothing to run');
    const cpu = m.XmrEngine.instances[0];
    expect(cpu.start).toHaveBeenCalledWith(expect.objectContaining({ worker: 'rig-host' }));
    cpu.emit('stopped', 1);
    await expect(p).resolves.toBe(1);
    expect(allErr()).toContain('the CPU miner gave up and nothing else runs; exiting');

    const m2 = load();
    const p2 = m2.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--gpu-index=none', '--xmr-address', XMR_ADDR]);
    await settle();
    fire('SIGTERM');
    await expect(p2).resolves.toBe(0);
    expect(m2.XmrEngine.instances[0].stop).toHaveBeenCalled();
    m2.XmrEngine.instances[0].emit('stopped', 0);
  });

  // With every GPU off the LLM may still find a card. If it can't, the CPU is
  // what was meant to run.
  test('with every GPU off, an LLM that cannot start leaves the CPU mining', async () => {
    const m = load();
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'A4000', usedMb: 7000, totalMb: 8000 }]);
    const p = m.run(['--mode', 'llm', '--no-update', '--gpu-index=none', '--xmr-address', XMR_ADDR]);
    await settle();
    expect(allErr()).not.toContain('nothing to run');
    expect(m.XmrEngine.instances[0].start).toHaveBeenCalled();
    fire('SIGTERM');
    await expect(p).resolves.toBe(0);
  });

  test('a CPU flag with no address is a warning, and Pearl mines', async () => {
    const m = load();
    const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--cpu-threads', '4']);
    await settle();
    expect(allOut()).toContain('warning: --cpu-threads does nothing without --xmr-address: no CPU mining');
    expect(m.PearlEngine.instances).toHaveLength(1);
    expect(m.XmrEngine.instances).toHaveLength(0);
    fire('SIGINT');
    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  // The Monero host follows the Pearl region, detected even when no GPU mines,
  // unless --region names one.
  test('picks the Monero pool by region, like the GPU miner', async () => {
    const m = load();
    m.probe.detectRegion.mockResolvedValue('kr');
    m.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--gpu-index=none', '--xmr-address', XMR_ADDR]);
    await settle();
    expect(allOut()).toContain('cpu:        Monero to 87wTMj88…1JxND9 · kr.monero.herominers.com:1111');
    expect(m.XmrEngine.instances[0].start).toHaveBeenCalledWith(expect.objectContaining({ endpoint: 'kr.monero.herominers.com:1111' }));
    fire('SIGTERM');

    const m2 = load();
    m2.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--gpu-index=none', '-r', 'fi', '--xmr-address', XMR_ADDR]);
    await settle();
    expect(m2.probe.detectRegion).not.toHaveBeenCalled();
    expect(m2.XmrEngine.instances[0].start).toHaveBeenCalledWith(expect.objectContaining({ endpoint: 'fi.monero.herominers.com:1111' }));
    fire('SIGTERM');
  });

  // --xmr-pool names the Monero host, so Monero needs no region: with no GPU
  // miner, and with Pearl on --pool, nothing is detected.
  test('--xmr-pool mines Monero there and skips region detection Monero needed', async () => {
    const m = load();
    m.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--gpu-index=none', '--xmr-address', XMR_ADDR,
      '--xmr-pool', 'xmr-us.kryptex.network:7029']);
    await settle();
    expect(m.probe.detectRegion).not.toHaveBeenCalled();
    expect(allOut()).toContain('cpu:        Monero to 87wTMj88…1JxND9 · xmr-us.kryptex.network:7029  (--xmr-pool)');
    expect(m.XmrEngine.instances[0].start).toHaveBeenCalledWith(expect.objectContaining({ endpoint: 'xmr-us.kryptex.network:7029' }));
    fire('SIGTERM');

    const m2 = load();
    const p = m2.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--pool', 'prl-us.kryptex.network:7048',
      '--xmr-address', XMR_ADDR, '--xmr-pool', 'xmr-us.kryptex.network:7029']);
    await settle();
    expect(m2.probe.detectRegion).not.toHaveBeenCalled();
    expect(m2.XmrEngine.instances[0].start).toHaveBeenCalledWith(expect.objectContaining({ endpoint: 'xmr-us.kryptex.network:7029' }));
    m2.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });
});

// ── local LLM ────────────────────────────────────────────────────────────────

// Mining on some cards, or none. HiveOS turns a card off by naming the ones
// still on in CUDA_VISIBLE_DEVICES, which the CLI drops; h-config.sh passes the
// same numbers as --gpu-index instead.
describe('--gpu-index', () => {
  const THREE = [
    { index: 0, name: 'NVIDIA GeForce RTX 4090' },
    { index: 1, name: 'NVIDIA GeForce RTX 4090' },
    { index: 2, name: 'NVIDIA GeForce RTX 4090' },
  ];
  const stop = async (m, p) => {
    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  };

  test('mines on the cards it names, and says which and why', async () => {
    const m = load();
    m.probe.detectGpuInfo.mockResolvedValue({ name: 'NVIDIA GeForce RTX 4090', count: 3 });
    m.probe.detectMinerGpus.mockResolvedValue([THREE[0], THREE[2]]);
    const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--gpu-index', '0,2']);
    await settle();
    expect(m.probe.detectMinerGpus).toHaveBeenCalledWith(process.env, [0, 2]);
    expect(allOut()).toContain('gpu:        3× NVIDIA GeForce RTX 4090  (auto)');
    expect(allOut()).toContain('mining on:  2 GPUs [0, 2]  (--gpu-index)');
    expect(m.PearlEngine.instances[0].start).toHaveBeenCalledWith(
      expect.objectContaining({ gpus: [THREE[0], THREE[2]] }));
    await stop(m, p);
  });

  // The HiveOS case: both are set to the same cards. "so every GPU can mine"
  // would be wrong here.
  test('with CUDA_VISIBLE_DEVICES set too, it says the flag chose the cards', async () => {
    await withEnv({ CUDA_VISIBLE_DEVICES: '0,2' }, async () => {
      const m = load();
      m.probe.detectMinerGpus.mockResolvedValue([THREE[0], THREE[2]]);
      const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--gpu-index=0,2']);
      await settle();
      expect(allOut()).toContain('mining on:  2 GPUs [0, 2]  (--gpu-index)');
      expect(allOut()).toContain('ignoring CUDA_VISIBLE_DEVICES=0,2: --gpu-index picks the cards');
      expect(allOut()).not.toContain('every GPU can mine');
      await stop(m, p);
    });
  });

  // nvidia-smi has no GPU 5 (planMinerGpus drops it): mine on the rest and say so.
  test('a card nvidia-smi does not list is skipped with a line on stderr', async () => {
    const m = load();
    m.probe.detectMinerGpus.mockResolvedValue([THREE[0]]);
    const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--gpu-index', '0,5']);
    await settle();
    expect(allErr()).toContain('skipping GPU 5 from --gpu-index: nvidia-smi does not list it');
    expect(allOut()).toContain('mining on:  1 GPU [0]  (--gpu-index)');
    await stop(m, p);
  });

  // nvidia-smi still lists the card that is turned off. It must not reach the
  // board, and the two that mine keep their own hashrates rather than an even
  // split of the total across all three.
  test('the board gets a row for each mining card and none for the others', async () => {
    const m = load();
    m.probe.detectGpusVram.mockResolvedValue(THREE.map((g) => ({ ...g, usedMb: 1000, totalMb: 24564 })));
    m.probe.detectMinerGpus.mockResolvedValue([THREE[0], THREE[2]]);
    const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--gpu-index', '0,2']);
    await settle();
    const rows = () => m.probe.postMinerReport.mock.calls.map((c) => c[0]);
    const cardOf = (r) => r.worker.split('/').pop();
    // Before any card has reported: the mining cards only.
    expect(rows().map(cardOf)).toEqual(['gpu0', 'gpu2']);

    const miner = m.PearlEngine.instances[0];
    miner.emit('event', { type: 'status', gpuIndex: 0, hashrate: 10, accepted: 4, rejected: 0 });
    miner.emit('event', { type: 'status', gpuIndex: 2, hashrate: 12, accepted: 6, rejected: 0 });
    m.probe.postMinerReport.mockClear();
    await intervalFor(NETWORK.reportIntervalMs).fn();
    expect(rows().map((r) => [cardOf(r), r.hashrate, r.accepted, r.vramTotalMb])).toEqual([
      ['gpu0', 10, 4, 24564],
      ['gpu2', 12, 6, 24564],
    ]);
    await stop(m, p);
  });

  test('PEARL_GPU_INDEX takes the same list, and the flag beats it', async () => {
    await withEnv({ PEARL_GPU_INDEX: '1,2' }, async () => {
      let m = load();
      m.probe.detectMinerGpus.mockResolvedValue([THREE[1], THREE[2]]);
      let p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update']);
      await settle();
      expect(m.probe.detectMinerGpus).toHaveBeenCalledWith(process.env, [1, 2]);
      expect(allOut()).toContain('mining on:  2 GPUs [1, 2]  (PEARL_GPU_INDEX)');
      await stop(m, p);

      out = [];
      m = load();
      m.probe.detectMinerGpus.mockResolvedValue([THREE[0]]);
      p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--gpu-index', '0']);
      await settle();
      expect(m.probe.detectMinerGpus).toHaveBeenCalledWith(process.env, [0]);
      expect(allOut()).toContain('mining on:  1 GPU [0]  (--gpu-index)');
      await stop(m, p);
    });
  });

  test('a bad value exits 1 before anything starts', async () => {
    const m = load();
    await expect(m.run(['-a', ADDR, '--no-update', '--gpu-index', 'first'])).resolves.toBe(1);
    expect(allErr()).toContain('error: invalid --gpu-index: first');
    expect(m.PearlEngine.instances).toHaveLength(0);
  });

  // HiveOS passes none when every GPU is turned off there. Exiting would only
  // be restarted every few seconds and end in "Miner starting error", so the
  // run waits, and the stats file keeps saying it isn't mining.
  test('none, mining only: mines nothing, says so, and waits until stopped', async () => {
    const m = load();
    const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--gpu-index=none',
      '--stats-file', '/tmp/s.json']);
    await settle();
    expect(m.PearlEngine.instances).toHaveLength(0);
    expect(m.probe.detectMinerGpus).not.toHaveBeenCalled();
    expect(m.probe.postMinerReport).not.toHaveBeenCalled();
    expect(allOut()).toContain('mining on:  no GPU  (--gpu-index none)');
    expect(allOut()).toContain('nothing to run with every GPU turned off; waiting until stopped');
    expect(allErr()).not.toContain('nothing to run —');

    // Something has to hold the process open: everything else is unrefed.
    const idle = intervalFor(60 * 60 * 1000);
    expect(idle).toBeDefined();
    expect(idle.unref).not.toHaveBeenCalled();
    idle.fn();

    intervalFor(10000).fn();
    const call = m.fs.writeFileSync.mock.calls.filter((c) => c[0] === '/tmp/s.json.tmp').pop();
    expect(JSON.parse(call[1])).toMatchObject({ mining: false, ths: 0, gpus: [] });

    let settled = false;
    p.then(() => { settled = true; });
    await settle();
    expect(settled).toBe(false);
    fire('SIGTERM');
    await expect(p).resolves.toBe(0);
    // Left running, that timer would hold the process open after the run ends,
    // until exitWith's forced exit seconds later.
    expect(clearInterval).toHaveBeenCalledWith(idle);
  });

  test('PEARL_GPU_INDEX=none does the same as --gpu-index none', async () => {
    await withEnv({ PEARL_GPU_INDEX: 'none' }, async () => {
      const m = load();
      const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update']);
      await settle();
      expect(m.PearlEngine.instances).toHaveLength(0);
      expect(allOut()).toContain('mining on:  no GPU  (PEARL_GPU_INDEX=none)');
      expect(allOut()).toContain('waiting until stopped');
      fire('SIGTERM');
      await expect(p).resolves.toBe(0);
    });
  });

  // The same rule as the flag: a value that doesn't parse stops the CLI. Mining
  // on every card instead could put a card to work that was meant to be left out.
  test('a PEARL_GPU_INDEX that does not parse exits 1 and says why', async () => {
    for (const bad of ['0 2', '0;2', '0,2x']) {
      await withEnv({ PEARL_GPU_INDEX: bad }, async () => {
        err = [];
        const m = load();
        await expect(m.run(['-a', ADDR, '--mode', 'mining', '--no-update'])).resolves.toBe(1);
        expect(allErr()).toContain('error: invalid PEARL_GPU_INDEX: ' + bad
          + ' (give GPU numbers from nvidia-smi separated by commas, such as 0,2, or none)');
        expect(m.PearlEngine.instances).toHaveLength(0);
      });
    }
  });

  test('a bad PEARL_GPU_INDEX does not matter when --gpu-index is given', async () => {
    await withEnv({ PEARL_GPU_INDEX: 'first' }, async () => {
      const m = load();
      m.probe.detectMinerGpus.mockResolvedValue([THREE[1]]);
      const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--gpu-index', '1']);
      await settle();
      expect(allOut()).toContain('mining on:  1 GPU [1]  (--gpu-index)');
      expect(allErr()).not.toContain('PEARL_GPU_INDEX');
      await stop(m, p);
    });
  });

  // The LLM picks its own cards, so none only stops the mining half.
  test('none, auto mode: no miner, the LLM still runs', async () => {
    const m = load();
    m.LlmEngineManager.serverInstalled = true;
    m.LlmEngineManager.modelInstalled = true;
    const p = m.run(['-a', ADDR, '--mode', 'auto', '--no-update', '--no-serve', '--gpu-index', 'none']);
    await settle();
    expect(m.PearlEngine.instances).toHaveLength(0);
    expect(allOut()).toContain('mining on:  no GPU  (--gpu-index none)');
    expect(allOut()).not.toContain('waiting until stopped');
    expect(intervalFor(60 * 60 * 1000)).toBeUndefined();
    const llm = m.LlmManager.instances[0];
    expect(llm).toBeDefined();
    llm.emit('stopped', 0);
    await expect(p).resolves.toBe(1);
  });

  // An LLM asked for that fails to start is still a failure, cards or not.
  test('none, LLM mode with an LLM that cannot start: exits 1 as before', async () => {
    const m = load();
    m.fs.existsSync.mockReturnValue(false);
    await expect(m.run(['--mode', 'llm', '--no-update', '--llm-binary', '/nope', '--gpu-index', 'none']))
      .resolves.toBe(1);
    expect(allOut()).not.toContain('mining on:  no GPU');
    expect(allErr()).toContain('nothing to run — no miner and the LLM did not start');
  });
});

describe('local LLM', () => {
  test('refuses to start the LLM without enough free VRAM (nothing to run → 1)', async () => {
    const m = load();
    // One card with 1000 MB free (8000 − 7000) — below the model's floor. The
    // LLM sizes against a single GPU (llama-server --split-mode none), so the
    // per-card figure is what the preflight uses, not a summed total.
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'A4000', usedMb: 7000, totalMb: 8000 }]);
    await expect(m.run(['--mode', 'llm', '--no-update'])).resolves.toBe(1);
    expect(allErr()).toContain('not enough free VRAM on any single GPU for the local LLM: 1000 MB free on GPU 0');
    expect(allErr()).toContain('nothing to run — no miner and the LLM did not start');
  });

  test('missing --llm-binary path fails LLM setup', async () => {
    const m = load();
    m.fs.existsSync.mockReturnValue(false);
    await expect(m.run(['--mode', 'llm', '--llm-binary', '/nope', '--no-update'])).resolves.toBe(1);
    expect(allErr()).toContain('LLM setup failed: llama-server binary not found: /nope');
  });

  test('missing --llm-model path fails LLM setup', async () => {
    const m = load();
    m.fs.existsSync.mockImplementation((p) => p !== '/nope.gguf');
    await expect(m.run(['--mode', 'llm', '--llm-binary', '/lb', '--llm-model', '/nope.gguf', '--no-update']))
      .resolves.toBe(1);
    expect(allErr()).toContain('LLM setup failed: LLM model not found: /nope.gguf');
  });

  test('a failed llama-server download points at --llm-binary', async () => {
    const m = load();
    m.LlmEngineManager.serverError = new Error('unzip not found');
    await expect(m.run(['--mode', 'llm', '--no-update'])).resolves.toBe(1);
    expect(allOut()).toContain('downloading llama-server from');
    expect(allErr()).toContain('unzip not found — pass --llm-binary </path/to/llama-server> instead');
  });

  test('macOS with --mode mining has nothing to run and exits 1', async () => {
    setPlatform('darwin');
    const m = load();
    await expect(m.run(['-a', ADDR, '--mode', 'mining', '--no-update'])).resolves.toBe(1);
    expect(allErr()).toContain('Switch the compute mode to LLM');
    expect(allErr()).toContain('nothing to run — no miner and the LLM did not start');
    expect(m.PearlEngine.instances).toHaveLength(0);
  });

  test('macOS downloads the llama-server build for its own architecture', async () => {
    for (const [arch, key] of [['arm64', 'darwin'], ['x64', 'darwin-x64']]) {
      setPlatform('darwin');
      setArch(arch);
      const m = load();
      m.LlmEngineManager.serverError = new Error('stop here');
      await expect(m.run(['--mode', 'llm', '--no-update'])).resolves.toBe(1);
      expect(m.LlmEngineManager.instances[0].opts.serverUrl).toBe(LLM.serverUrl[key]);
      expect(allOut()).toContain('downloading llama-server from ' + LLM.serverUrl[key]);
      out = [];
    }
  });

  // The headless path is the one a 5090 box under systemd actually runs, so the
  // tier has to reach llama-server from HERE, not only from the GUI shell. It
  // did not until this was wired: earn-cli passed LLM.model unconditionally, so
  // a 32 GB card would have quietly kept serving the small default.
  test('a 32 GB card gets the vision tier, its projector and its tuned flags', async () => {
    const m = load();
    m.nodeStore.loadNode.mockReturnValue(makeNode({ connected: true, name: 'rig' }));
    // An idle 5090, budgeted against what CUDA actually exposes.
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 5090', usedMb: 0, totalMb: 32149 }]);
    m.LlmEngineManager.mmprojInstalled = false;
    m.LlmEngineManager.mmprojFile = '/cache/mmproj.gguf';

    m.LlmEngineManager.serverInstalled = true;
    m.LlmEngineManager.modelInstalled = true;
    const p = m.run(['--mode', 'llm', '--no-serve', '--no-update']);
    await settle();

    const tier = LLM.tiers[0];
    const llm = m.LlmManager.instances[0];
    expect(llm.start).toHaveBeenCalledWith(expect.objectContaining({
      mmprojPath: '/cache/mmproj.gguf',
      ctxSize: 262144,
      ctxLadder: tier.ctxLadder,
      extraArgs: tier.extraArgs,
    }));
    // Without a quantised KV cache the model does not load at this context.
    expect(llm.start.mock.calls[0][0].extraArgs.join(' ')).toContain('--cache-type-k q8_0');
    // And the operator is told which model they are getting, plus the extra
    // download it implies.
    expect(allOut()).toContain('preparing local LLM (' + tier.name + ')');
    expect(allOut()).toContain('downloading vision projector');

    llm.emit('ready', { baseUrl: 'http://127.0.0.1:8080' });
    await settle();
    llm.emit('stopped', 0);
    await expect(p).resolves.toBe(1);
  });

  // A mining 32 GB card serves the default model: the tier plus the mining
  // reserve does not fit. An idle 5090 goes demand-driven for the tier (see the
  // demand tests), so these cards have a desktop on them: 3 GB used, which
  // leaves 29,149 MB free.
  const argvCoRun = ['-a', ADDR, '--mode', 'both', '--no-update', '--no-report', '--llm-binary', '/lb'];
  const fiveNinety = (m, usedMb = 3000) => {
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 5090', usedMb, totalMb: 32149 }]);
    m.LlmEngineManager.serverInstalled = true;
    m.LlmEngineManager.modelInstalled = true;
    m.LlmEngineManager.mmprojInstalled = true;
  };

  test('a mining 32 GB card co-runs the default model, not the vision tier', async () => {
    const m = load();
    fiveNinety(m);
    const p = m.run(argvCoRun);
    await settle();

    const opts = m.LlmManager.instances[0].start.mock.calls[0][0];
    expect(opts.alias).toBe(LLM.model.name);
    expect(opts.extraArgs).not.toEqual(LLM.tiers[0].extraArgs);
    expect(m.PearlEngine.instances[0].start).toHaveBeenCalled();

    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  // A prompt hands its card to the LLM: mining pauses on that card and comes
  // back after (shared/promptPause has the timing). Here: the busy signal from
  // llama-server reaches the miner, and lets go when the LLM exits.
  test('a prompt pauses mining on its card while the LLM co-runs, and the card mines again when the LLM exits', async () => {
    const m = load();
    fiveNinety(m);
    const p = m.run(argvCoRun);
    await settle();

    const miner = m.PearlEngine.instances[0];
    const llm = m.LlmManager.instances[0];
    llm.emit('busy', true);
    expect(miner.pauseCards).toHaveBeenCalledWith([0]);
    expect(allOut()).toContain('mining paused on GPU 0 while the local LLM answers a prompt');

    llm.emit('stopped', 1);
    expect(miner.resumeCards).toHaveBeenCalledWith([0]);
    expect(allOut()).toContain('mining resumed on GPU 0');

    miner.emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('a card busy for 10 minutes without a break mines again, and the warning goes to stderr', async () => {
    jest.useFakeTimers({ doNotFake: ['setImmediate', 'nextTick', 'setInterval', 'clearInterval'] });
    try {
      const m = load();
      fiveNinety(m);
      const p = m.run(argvCoRun);
      await settle();
      const miner = m.PearlEngine.instances[0];
      m.LlmManager.instances[0].emit('busy', true);
      jest.advanceTimersByTime(LLM.promptPause.maxPauseMs);
      expect(miner.resumeCards).toHaveBeenCalledWith([0]);
      expect(allErr()).toContain('mining resumed on GPU 0: the local LLM has been busy for 10 minutes without a break');
      miner.emit('stopped', 0);
      await expect(p).resolves.toBe(0);
    } finally {
      jest.useRealTimers();
    }
  });

  test('on a rig of several cards, a server that never said which card it loaded on pauses nothing', async () => {
    const m = load();
    fiveNinety(m);
    m.probe.detectPciBusIds.mockResolvedValue({ 0: '00000000:01:00.0', 1: '00000000:02:00.0' });
    const p = m.run(argvCoRun);
    await settle();
    const miner = m.PearlEngine.instances[0];
    m.LlmManager.instances[0].emit('busy', true);
    expect(miner.pauseCards).not.toHaveBeenCalled();
    // Once it has said, the card it named is the one that pauses.
    m.LlmManager.instances[0].deviceBusId = '0000:02:00.0';
    m.LlmManager.instances[0].emit('busy', true);
    expect(miner.pauseCards).toHaveBeenCalledWith([1]);
    miner.emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  // Two 5090s, which Vulkan lists the other way round. The probe's bus ids put
  // the model on the planned card, and a prompt pauses that card.
  test('on a rig of several cards, a prompt pauses the card matched by bus id', async () => {
    const m = load();
    fiveNinety(m);
    m.probe.detectGpusVram.mockResolvedValue([
      { index: 0, name: 'RTX 5090', usedMb: 3000, totalMb: 32149 },
      { index: 1, name: 'RTX 5090', usedMb: 3000, totalMb: 32149 },
    ]);
    m.probe.detectPciBusIds.mockResolvedValue({ 0: '00000000:01:00.0', 1: '00000000:02:00.0' });
    m.probe.detectLlmDevices.mockResolvedValue([
      { device: 'Vulkan0', name: 'RTX 5090', busId: '0000:02:00.0' },
      { device: 'Vulkan1', name: 'RTX 5090', busId: '0000:01:00.0' },
    ]);
    const p = m.run(argvCoRun);
    await settle();
    const miner = m.PearlEngine.instances[0];
    const llm = m.LlmManager.instances[0];
    expect(llm.start).toHaveBeenCalledWith(expect.objectContaining({ mainGpu: 0, device: 'Vulkan1' }));
    llm.emit('busy', true);
    expect(miner.pauseCards).toHaveBeenCalledWith([0]);
    miner.emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  // An AMD rig: no nvidia-smi, so no bus ids, but the miner runs two cards.
  test('on a rig of several cards with no bus ids at all, a prompt pauses nothing', async () => {
    const m = load();
    fiveNinety(m);
    m.probe.detectPciBusIds.mockResolvedValue({});
    m.probe.detectMinerGpus.mockResolvedValue([{ index: 0, name: 'AMD' }, { index: 1, name: 'AMD' }]);
    const p = m.run(argvCoRun);
    await settle();
    const miner = m.PearlEngine.instances[0];
    m.LlmManager.instances[0].emit('busy', true);
    expect(miner.pauseCards).not.toHaveBeenCalled();
    miner.emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('a shutdown mid-prompt leaves the pause to the miner\'s own stop', async () => {
    const m = load();
    fiveNinety(m);
    const p = m.run(argvCoRun);
    await settle();
    const miner = m.PearlEngine.instances[0];
    m.LlmManager.instances[0].emit('busy', true);
    fire('SIGTERM');
    expect(miner.stop).toHaveBeenCalled();
    expect(miner.resumeCards).not.toHaveBeenCalled();
    miner.emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('--llm-model serves the operator file as the default model, not as the tier the card could hold', async () => {
    // The operator's own Gemma copy started with the tier's context, its MTP
    // speculation flags and its name -- which the server routes jobs on.
    const m = load();
    fiveNinety(m, 0);
    const p = m.run(['--mode', 'llm', '--no-update', '--llm-binary', '/lb', '--llm-model', '/models/my-gemma.gguf']);
    await settle();

    const llm = m.LlmManager.instances[0];
    const opts = llm.start.mock.calls[0][0];
    expect(opts.alias).toBe(LLM.model.name);
    expect(opts.modelPath).toBe('/models/my-gemma.gguf');
    expect(opts.extraArgs).not.toEqual(LLM.tiers[0].extraArgs);
    expect(opts.ctxSize).not.toBe(262144);

    fire('SIGINT');
    await expect(p).resolves.toBe(0);
  });

  test('--llm-model pointing at the tier file keeps the tier and its settings', async () => {
    const m = load();
    fiveNinety(m, 0);
    const p = m.run(['--mode', 'llm', '--no-update', '--llm-binary', '/lb', '--llm-model', '/models/' + LLM.tiers[0].file]);
    await settle();

    const llm = m.LlmManager.instances[0];
    expect(llm.start).toHaveBeenCalledWith(expect.objectContaining({
      ctxSize: 262144, extraArgs: LLM.tiers[0].extraArgs, alias: LLM.tiers[0].name,
    }));

    fire('SIGINT');
    await expect(p).resolves.toBe(0);
  });

  // Beside the miner the tier does not fit a 32 GB card, and the default is
  // what pickModel falls back to. The tier's weights must not start under
  // Gemma's name, which jobs are routed on, without the tier's flags.
  test('--llm-model naming the tier skips the LLM where the tier does not fit, and mines', async () => {
    const m = load();
    fiveNinety(m);
    const p = m.run([...argvCoRun, '--llm-model', '/models/' + LLM.tiers[0].file]);
    await settle();

    expect(m.LlmManager.instances).toHaveLength(0);
    expect(allErr()).toContain('not enough free VRAM on any single GPU for --llm-model: 29149 MB free on GPU 0');
    expect(allErr()).toContain('MB for ' + LLM.tiers[0].name + ' — skipping the LLM.');
    expect(m.PearlEngine.instances[0].start).toHaveBeenCalled();

    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('--llm-model naming the tier is served as the tier when VRAM cannot be read', async () => {
    const m = load();
    m.probe.detectGpusVram.mockResolvedValue([]);
    m.LlmEngineManager.serverInstalled = true;
    const p = m.run(['--mode', 'llm', '--no-update', '--llm-binary', '/lb', '--llm-model', '/models/' + LLM.tiers[0].file]);
    await settle();

    expect(m.LlmManager.instances[0].start).toHaveBeenCalledWith(expect.objectContaining({
      alias: LLM.tiers[0].name, extraArgs: LLM.tiers[0].extraArgs,
    }));

    fire('SIGINT');
    await expect(p).resolves.toBe(0);
  });

  test('--llm-model in auto co-runs its file instead of going demand-driven for the tier', async () => {
    // Auto on a 5090 goes demand-driven for the tier. An operator's own small
    // model has no bigger choice to wait for, so it co-runs as it always has.
    const m = load();
    fiveNinety(m, 0);
    const p = m.run(['-a', ADDR, '--no-update', '--no-report', '--no-serve', '--llm-binary', '/lb',
      '--llm-model', '/models/my-gemma.gguf']);
    await settle();

    expect(allOut()).not.toContain('needs the GPU to itself');
    expect(m.autoGate.createAutoGate.instances).toHaveLength(0);

    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  // Starting the tier is half the job; SAYING you started it is the other half.
  // Every reporting site read LLM.model directly, which was the right answer
  // only while every node ran the same model — so a serving 5090 told the
  // network board it was running Gemma, and metrics.model carried that same
  // wrong name into the `model` field of every gateway completion it served.
  test('a serving 5090 reports the tier it loaded, not the fleet default', async () => {
    const m = load();
    m.nodeStore.loadNode.mockReturnValue(makeNode({ connected: true, name: 'rig' }));
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 5090', usedMb: 0, totalMb: 32149 }]);
    m.probe.detectVram.mockResolvedValue({ totalMb: 32149, usedMb: 30150 });
    m.LlmEngineManager.serverInstalled = true;
    m.LlmEngineManager.modelInstalled = true;
    m.LlmEngineManager.mmprojInstalled = true;
    const p = m.run(['--mode', 'llm', '--no-update']);
    await settle();

    const tier = LLM.tiers[0];
    const llm = m.LlmManager.instances[0];
    llm.emit('ready', { baseUrl: 'http://127.0.0.1:8080' });
    await settle();

    // What runs the jobs — a thunk, so a fleet restart at another tier follows.
    // Compared by name, not identity: the CLI is loaded in its own module
    // registry, so its `config` is a different object graph from this file's.
    expect(m.JobWorker.instances[0].opts.servingModel().name).toBe(tier.name);

    // What the network board is told.
    const pinger = intervalFor(NODE.pingIntervalMs);
    await pinger.fn();
    const ping = m.io.postJson.mock.calls
      .filter((c) => /\/api\/nodes\/ping$/.test(c[0]))
      .pop();
    expect(ping[1]).toMatchObject({ model: tier.name, quant: tier.quant });

    fire('SIGINT');
    await expect(p).resolves.toBe(0);
  });

  test('serves cluster jobs when connected: worker, pings, telemetry, SIGINT shutdown', async () => {
    const m = load();
    m.nodeStore.loadNode.mockReturnValue(makeNode({ connected: true, name: 'rig' }));
    // One roomy card (22 GB free) — the whole model fits, so full offload, on
    // GPU 0. detectVram still feeds the keep-alive telemetry below.
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 4090', usedMb: 2000, totalMb: 24000 }]);
    m.probe.detectVram.mockResolvedValue({ totalMb: 24000, usedMb: 2000 });
    m.probe.findFreePort.mockResolvedValue(9090);
    m.probe.detectGpuInfo.mockResolvedValue({ name: 'NVIDIA GeForce RTX 4090', count: 1 });
    const p = m.run(['--mode', 'llm', '--no-update']);
    await settle();

    expect(allOut()).toContain('mode:       llm');
    expect(allOut()).toContain('local LLM starting on 1 GPU [0]');
    expect(allOut()).toContain('downloading LLM model');

    // The binary resolver's engine wires the shared extractor with no hint of
    // its own: the extractor names the tool that failed (tar on Linux), and the
    // resolver adds the --llm-binary way out. A hint here said "install unzip",
    // the wrong tool, and named --llm-binary twice.
    const le = m.LlmEngineManager.instances[0];
    await le.opts.extract('/z.zip', '/dest');
    expect(m.io.extractLlamaZip).toHaveBeenCalledWith('/z.zip', '/dest');

    const llm = m.LlmManager.instances[0];
    expect(llm.start).toHaveBeenCalledWith(expect.objectContaining({
      binaryPath: '/cache/llama-server', modelPath: '/cache/model.gguf', nGpuLayers: ALL_LAYERS, port: 9090, mainGpu: 0,
    }));
    llm.emit('log', { line: 'srv up', level: 'info' });
    llm.emit('log', { line: 'srv err', level: 'error' });
    llm.emit('ready', { baseUrl: 'http://127.0.0.1:9090' });
    await settle();
    expect(allOut()).toContain('local LLM ready — OpenAI endpoint http://127.0.0.1:9090/v1');
    expect(allOut()).toContain('serving cluster jobs for the LLMJob network');

    const jw = m.JobWorker.instances[0];
    expect(jw.start).toHaveBeenCalled();
    expect(jw.opts.serverUrl).toBe(NODE.serverUrl);

    // Exercise the worker's wiring back into io.
    // The worker's abort signal reaches the request (see JobWorker.stop).
    const signal = new AbortController().signal;
    await jw.opts.post('http://u', { a: 1 }, { signal });
    expect(m.io.postJson).toHaveBeenCalledWith('http://u', { a: 1 }, 30000, { signal });
    const onDelta = jest.fn();
    const onReasoning = jest.fn();
    await jw.opts.runJob({ messages: [] }, { onDelta, onReasoning });
    expect(m.io.streamChatCompletion).toHaveBeenCalledWith(llm.baseUrl, { messages: [] }, onDelta, onReasoning);

    jw.emit('error', new Error('poll down'));
    jw.emit('job', { id: 'j1' });
    jw.emit('failed', { id: 'j1', error: 'nope' });
    expect(allErr()).toContain('job poll failed: poll down (retrying)');
    expect(allOut()).toContain('cluster job j1 — running locally');
    expect(allErr()).toContain('cluster job j1 failed: nope');

    llm.emit('ready', { baseUrl: 'http://127.0.0.1:9090' }); // no second worker
    expect(m.JobWorker.instances.length).toBe(1);
    llm.emit('stats', { tokensPerSec: 12.34 });
    llm.emit('stats', { tokensPerSec: 'garbage' });
    llm.emit('error', new Error('cuda oom'));
    expect(allOut()).toContain('🧠 12.3 tok/s');
    expect(allErr()).toContain('LLM error: cuda oom');

    // Keep-alive ping with full telemetry, including the VRAM-read failure path
    // and a ping POST failure (silent — serving pings are not verbose).
    const pinger = intervalFor(NODE.pingIntervalMs);
    m.probe.detectVram.mockRejectedValueOnce(new Error('smi gone'));
    m.io.postJson.mockRejectedValueOnce(new Error('ping down'));
    await pinger.fn();
    expect(allErr()).not.toContain('ping down');

    fire('SIGINT');
    await expect(p).resolves.toBe(0);
    expect(jw.stop).toHaveBeenCalled();
    expect(llm.stop).toHaveBeenCalled();
    llm.emit('stopped', 0); // during shutdown: fleet.stop() already suppressed it
    await pinger.fn(); // after stopServe: telemetry reports 0 active jobs
    expect(allErr()).not.toContain('local LLM exited');
  });

  test('a crashing llama-server fails an LLM-only run with its exit code', async () => {
    const m = load();
    m.nodeStore.loadNode.mockReturnValue(makeNode({ connected: true, serverUrl: 'https://custom.example' }));
    const p = m.run(['--mode', 'llm', '--no-update']);
    await settle();
    const llm = m.LlmManager.instances[0];
    llm.emit('ready', { baseUrl: 'http://127.0.0.1:8080' });
    await settle();
    expect(m.JobWorker.instances[0].opts.serverUrl).toBe('https://custom.example');
    llm.emit('stopped', 3);
    await expect(p).resolves.toBe(3);
    expect(allErr()).toContain('local LLM exited (code 3)');
  });

  // Serving is the DEFAULT, account or not: an unlinked rig self-registers and
  // takes public jobs. Linking adds access to private queues, nothing else.
  test('an unconnected node registers itself and serves public jobs anyway', async () => {
    const m = load();
    m.nodeStore.loadNode.mockReturnValue(makeNode({ connected: false }));
    m.nodeStore.getOrCreateNode.mockReturnValue(makeNode({ connected: false }));
    m.LlmEngineManager.serverInstalled = true;
    m.LlmEngineManager.modelInstalled = true;
    const p = m.run(['--mode', 'llm', '--no-update']);
    await settle();
    expect(allOut()).toContain('LLM server found: /cache/llama-server');
    expect(allOut()).toContain('LLM model found: /cache/model.gguf');

    // It registered as an unclaimed node before serving.
    const registerCall = m.io.postJson.mock.calls.find((c) => String(c[0]).endsWith('/api/nodes/register'));
    expect(registerCall).toBeTruthy();
    expect(allOut()).toContain('serving public jobs as an unlinked node');

    const llm = m.LlmManager.instances[0];
    expect(llm.start).toHaveBeenCalledWith(expect.objectContaining({ nGpuLayers: ALL_LAYERS })); // no VRAM → full offload
    llm.emit('ready', { baseUrl: 'http://127.0.0.1:8080' });
    await settle();
    expect(m.JobWorker.instances.length).toBe(1);
    llm.emit('stopped', 0);
    await expect(p).resolves.toBe(1);
  });

  // The server decides which id a machine is enrolled under and can hand back
  // one the client did not compute — a machine whose old narrow id turns out to
  // be taken by someone else's key is enrolled on the wide one instead. Keeping
  // the local id left it signing every later call as a row that does not exist,
  // i.e. serving nothing with nothing to show why.
  test('adopts and persists a node id the server reassigns', async () => {
    const m = load();
    const node = makeNode({ nodeId: '5840fc', connected: false });
    m.nodeStore.loadNode.mockReturnValue(node);
    m.nodeStore.getOrCreateNode.mockReturnValue(node);
    m.LlmEngineManager.serverInstalled = true;
    m.LlmEngineManager.modelInstalled = true;
    m.io.postJson.mockResolvedValue({ status: 200, data: { nodeId: 'a1b2c3d4e5f60789' } });

    const p = m.run(['--mode', 'llm', '--no-update']);
    await settle();

    expect(node.nodeId).toBe('a1b2c3d4e5f60789');
    expect(m.nodeStore.saveNode).toHaveBeenCalledWith(expect.objectContaining({ nodeId: 'a1b2c3d4e5f60789' }));
    expect(allOut()).toContain('node id updated by the network to a1b2c3d4e5f60789');

    m.LlmManager.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(1);
  });

  test('keeps its own node id when the server agrees', async () => {
    const m = load();
    const node = makeNode({ nodeId: '5840fc', connected: false });
    m.nodeStore.loadNode.mockReturnValue(node);
    m.nodeStore.getOrCreateNode.mockReturnValue(node);
    m.LlmEngineManager.serverInstalled = true;
    m.LlmEngineManager.modelInstalled = true;
    m.io.postJson.mockResolvedValue({ status: 200, data: { nodeId: '5840fc' } });

    const p = m.run(['--mode', 'llm', '--no-update']);
    await settle();

    expect(node.nodeId).toBe('5840fc');
    expect(allOut()).not.toContain('node id updated by the network');

    m.LlmManager.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(1);
  });

  // The server now REFUSES a registration whose node id is held by a different
  // key (a fingerprint collision) instead of reporting a false success. That
  // arrives as a non-200, not a transport failure, and must read as a failure
  // rather than being mistaken for a successful enrolment.
  test('a refused registration (node id already held) is reported, not treated as success', async () => {
    const m = load();
    const node = makeNode({ nodeId: '5840fc', connected: false });
    m.nodeStore.loadNode.mockReturnValue(node);
    m.nodeStore.getOrCreateNode.mockReturnValue(node);
    m.LlmEngineManager.serverInstalled = true;
    m.LlmEngineManager.modelInstalled = true;
    m.io.postJson.mockResolvedValue({ status: 400, data: { error: 'Node key mismatch' } });

    const p = m.run(['--mode', 'llm', '--no-update']);
    await settle();

    expect(allErr()).toContain('could not register with the network — running the LLM locally only');
    expect(node.nodeId).toBe('5840fc');                     // identity untouched
    expect(m.nodeStore.saveNode).not.toHaveBeenCalled();
    m.LlmManager.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(1);
  });

  test('a failed registration is survivable — the model still runs locally', async () => {
    const m = load();
    m.nodeStore.loadNode.mockReturnValue(null);
    m.nodeStore.getOrCreateNode.mockReturnValue(makeNode({ connected: false }));
    m.LlmEngineManager.serverInstalled = true;
    m.LlmEngineManager.modelInstalled = true;
    m.io.postJson.mockRejectedValue(new Error('offline'));
    const p = m.run(['--mode', 'llm', '--no-update']);
    await settle();

    expect(allErr()).toContain('could not register with the network — running the LLM locally only');
    const llm = m.LlmManager.instances[0];
    expect(llm.start).toHaveBeenCalled(); // the LLM came up regardless
    llm.emit('stopped', 0);
    await expect(p).resolves.toBe(1);
  });

  // A register that failed was the only one the process ever tried, so a server
  // that was down at that moment left the node polling for jobs it could never
  // be given. The keep-alive tries again.
  test('a failed registration is retried by the next keep-alive ping', async () => {
    const m = load();
    m.nodeStore.loadNode.mockReturnValue(makeNode({ connected: false }));
    m.LlmEngineManager.serverInstalled = true;
    m.LlmEngineManager.modelInstalled = true;
    m.io.postJson.mockRejectedValueOnce(new Error('offline'));
    const p = m.run(['--mode', 'llm', '--no-update']);
    await settle();
    expect(allErr()).toContain('could not register with the network');
    expect(allOut()).not.toContain('serving public jobs as an unlinked node');

    const llm = m.LlmManager.instances[0];
    llm.emit('ready', { baseUrl: llm.baseUrl });
    await settle();
    const registers = () => m.io.postJson.mock.calls.filter((c) => String(c[0]).endsWith('/api/nodes/register'));
    expect(registers()).toHaveLength(2);   // the first ping's retry
    expect(allOut()).toContain('serving public jobs as an unlinked node (abc123)');
    await intervalFor(NODE.pingIntervalMs).fn();
    expect(registers()).toHaveLength(2);   // registered: no more

    llm.emit('stopped', 0);
    await expect(p).resolves.toBe(1);
  });

  // Minting an identity writes node.json. A disk that refuses it ended an llm
  // run with 'fatal' before any model started.
  test('an identity that cannot be written still runs the LLM locally', async () => {
    const m = load();
    m.nodeStore.getOrCreateNode.mockImplementation(() => { throw new Error('EROFS: read-only file system'); });
    const p = m.run(['--mode', 'llm', '--no-update']);
    await settle();

    expect(allErr()).toContain('could not set up this node\'s identity (EROFS: read-only file system) — running the LLM locally only');
    const llm = m.LlmManager.instances[0];
    expect(llm.start).toHaveBeenCalled();
    llm.emit('ready', { baseUrl: llm.baseUrl });
    await settle();
    expect(m.JobWorker.instances).toHaveLength(0);
    llm.emit('stopped', 0);
    await expect(p).resolves.toBe(1);
  });

  // The serving pings run from a timer. Signing outside the try made a corrupt
  // key in node.json an unhandled rejection there.
  test('a corrupt key in node.json cannot break a serving ping', async () => {
    const m = load();
    m.nodeStore.loadNode.mockReturnValue(makeNode({ connected: true, secretKey: 'not-a-key' }));
    const p = m.run(['--mode', 'llm', '--no-update']);
    await settle();
    const llm = m.LlmManager.instances[0];
    llm.emit('ready', { baseUrl: llm.baseUrl });
    await settle();
    await expect(intervalFor(NODE.pingIntervalMs).fn()).resolves.toEqual({ error: expect.any(String) });
    llm.emit('stopped', 0);
    await expect(p).resolves.toBe(1);
  });

  // A cached binary has to go through ensureServer: that is where a binary left
  // without its execute bit gets it back. Returning the cached path skipped it,
  // and such a binary failed with EACCES on every start.
  test('a cached llama-server goes through ensureServer, which repairs its execute bit', async () => {
    const m = load();
    m.LlmEngineManager.serverInstalled = true;
    m.LlmEngineManager.modelInstalled = true;
    const p = m.run(['--mode', 'llm', '--no-update']);
    await settle();
    expect(allOut()).toContain('LLM server found: /cache/llama-server');
    const engine = m.LlmEngineManager.instances.find((e) => e.isServerInstalled.mock.calls.length);
    expect(engine.ensureServer).toHaveBeenCalled();
    expect(m.LlmManager.instances[0].start).toHaveBeenCalledWith(expect.objectContaining({ binaryPath: '/cache/llama-server' }));
    m.LlmManager.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(1);
  });

  test('--no-serve keeps the model entirely local: no registration, no worker', async () => {
    const m = load();
    // No stored identity, and --no-serve must not mint one: an opted-out box
    // never registers a keypair with the network.
    m.nodeStore.loadNode.mockReturnValue(null);
    m.LlmEngineManager.serverInstalled = true;
    m.LlmEngineManager.modelInstalled = true;
    const p = m.run(['--mode', 'llm', '--no-serve', '--no-update']);
    await settle();

    expect(m.io.postJson.mock.calls.some((c) => String(c[0]).endsWith('/api/nodes/register'))).toBe(false);
    expect(m.nodeStore.getOrCreateNode).not.toHaveBeenCalled();
    expect(allOut()).not.toContain('serving public jobs as an unlinked node');
    const llm = m.LlmManager.instances[0];
    llm.emit('ready', { baseUrl: 'http://127.0.0.1:8080' });
    await settle();
    expect(m.JobWorker.instances.length).toBe(0);
    llm.emit('stopped', 0);
    await expect(p).resolves.toBe(1);
  });

  test('--llm-max-instances caps the fleet and says so', async () => {
    const m = load();
    m.probe.detectGpusVram.mockResolvedValue([
      { index: 0, name: 'RTX 4090', usedMb: 1000, totalMb: 24000 },
      { index: 1, name: 'RTX 4090', usedMb: 1000, totalMb: 24000 },
      { index: 2, name: 'RTX 4090', usedMb: 1000, totalMb: 24000 },
    ]);
    m.probe.findFreePort.mockImplementation((h, p) => Promise.resolve(p));
    const p = m.run(['--mode', 'llm', '--llm-max-instances', '2', '--no-update']);
    await settle();

    expect(allOut()).toContain('serving on 2 of 3 GPUs — capped by --llm-max-instances 2');
    expect(allOut()).toContain('local LLM starting on 2 GPUs [0, 1]');

    fire('SIGINT');
    m.LlmManager.instances[0].emit('stopped', 0);
    await p;
  });

  // The report this fix came from: nvidia-smi's GPU 0 is the 32 GB card, but
  // Vulkan lists the 12 GB card that drives the desktop first.
  test('loads on the planned card by llama-server\'s own name for it', async () => {
    const m = load();
    m.probe.detectGpusVram.mockResolvedValue([
          { index: 0, name: 'NVIDIA RTX PRO 4500 Blackwell', usedMb: 500, totalMb: 32623 },
      { index: 1, name: 'NVIDIA GeForce RTX 4070', usedMb: 1500, totalMb: 12282 },
    ]);
    m.probe.detectLlmDevices.mockResolvedValue([
      { device: 'Vulkan0', name: 'NVIDIA GeForce RTX 4070' },
      { device: 'Vulkan1', name: 'NVIDIA RTX PRO 4500 Blackwell' },
    ]);
    const p = m.run(['--mode', 'llm', '--no-update', '--llm-binary', '/lb']);
    await settle();
    expect(m.probe.detectLlmDevices).toHaveBeenCalledWith('/lb', '/cache/model.gguf');
    expect(m.LlmManager.instances[0].start).toHaveBeenCalledWith(expect.objectContaining({ mainGpu: 0, device: 'Vulkan1' }));
    expect(allOut()).toContain('local LLM starting on 1 GPU [0] (llama-server calls it Vulkan1)');
    fire('SIGINT');
    await expect(p).resolves.toBe(0);
  });

  test('says so when it cannot tell which card is which', async () => {
    const m = load();
    m.probe.detectGpusVram.mockResolvedValue([
      { index: 0, name: 'RTX 4090', usedMb: 3000, totalMb: 24000 },
      { index: 1, name: 'RTX 3060', usedMb: 1000, totalMb: 12000 },
    ]);
    const p = m.run(['--mode', 'llm', '--no-update', '--llm-binary', '/lb']);
    await settle();
    expect(allOut()).toContain('llama-server did not list its cards, so the local LLM is placed by GPU number (--main-gpu), which may be a different card');
    expect(m.LlmManager.instances[0].start).toHaveBeenCalledWith(expect.objectContaining({ mainGpu: 0 }));
    fire('SIGINT');
    await expect(p).resolves.toBe(0);
  });

  test('runs one instance and worker per eligible GPU, summing their active jobs', async () => {
    const m = load();
    m.nodeStore.loadNode.mockReturnValue(makeNode({ connected: true, name: 'rig' }));
    // Two roomy cards → one llama-server + cluster worker pinned to each.
    // Below the Qwen tier's offer floor (65536 needs ~22,333 free) so both cards
    // resolve the SAME small model: this test is about multi-GPU fan-out, not
    // about which tier a card qualifies for.
    m.probe.detectGpusVram.mockResolvedValue([
      { index: 0, name: 'RTX 4090', usedMb: 3000, totalMb: 24000 },
      { index: 1, name: 'RTX 4090', usedMb: 2500, totalMb: 24000 },
    ]);
    m.probe.detectVram.mockResolvedValue({ totalMb: 24000, usedMb: 2000 });
    m.probe.findFreePort.mockImplementation((h, p) => Promise.resolve(p));
    const p = m.run(['--mode', 'llm', '--no-update']);
    await settle();

    // The plural log names every planned card, but they start ONE AT A TIME:
    // simultaneous multi-GB model loads thrash the page cache (see LlmFleet).
    expect(allOut()).toContain('local LLM starting on 2 GPUs [0, 1]');
    expect(m.LlmManager.instances.length).toBe(1);
    const g0 = m.LlmManager.instances[0];
    expect(g0.start).toHaveBeenCalledWith(expect.objectContaining({ port: 8080, mainGpu: 0 }));

    // card 0 ready → card 1 is spawned on the next port
    g0.emit('ready', { baseUrl: g0.baseUrl });
    await settle();
    expect(m.LlmManager.instances.length).toBe(2);
    const g1 = m.LlmManager.instances[1];
    expect(g1.start).toHaveBeenCalledWith(expect.objectContaining({ port: 8081, mainGpu: 1 }));

    // both cards up → a cluster worker per card
    g1.emit('ready', { baseUrl: g1.baseUrl });
    await settle();
    expect(m.JobWorker.instances.length).toBe(2);

    // one keep-alive ping loop for the whole fleet, summing active jobs (2 each)
    const pinger = intervalFor(NODE.pingIntervalMs);
    await pinger.fn();
    expect(m.io.postJson.mock.calls.pop()[1].activeJobs).toBe(4);

    // SIGINT stops every instance and resolves the LLM-only run
    fire('SIGINT');
    await expect(p).resolves.toBe(0);
    expect(g0.stop).toHaveBeenCalled();
    expect(g1.stop).toHaveBeenCalled();
  });
});

// ── both mode (miner + LLM together) ─────────────────────────────────────────

describe('both mode', () => {
  const argvBoth = ['-a', ADDR, '--mode', 'both', '--no-update', '--no-report',
    '--llm-binary', '/lb', '--llm-model', '/lm'];

  test('co-runs; an LLM death does not stop mining; the miner exit code wins', async () => {
    const m = load();
    // One card with 15000 MB free (16000 − 1000): room for the whole model even
    // after the 2048 MB mining reserve. Pinned to that GPU via --main-gpu.
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'A4000', usedMb: 1000, totalMb: 16000 }]);
    const p = m.run(argvBoth);
    await settle();

    expect(allOut()).toContain('mode:       both');
    const llm = m.LlmManager.instances[0];
    // Offload is all-or-nothing, so an eligible card always gets ALL_LAYERS.
    expect(llm.start).toHaveBeenCalledWith(expect.objectContaining({ nGpuLayers: ALL_LAYERS, port: 8080, mainGpu: 0 }));
    llm.emit('ready', { baseUrl: 'http://127.0.0.1:8080' }); // not connected → no worker
    llm.emit('stopped', 2); // LLM dies; the miner keeps running
    expect(allErr()).toContain('local LLM exited (code 2)');

    const miner = m.PearlEngine.instances[0];
    miner.emit('stopped', 7);
    expect(llm.stop).toHaveBeenCalled();
    await expect(p).resolves.toBe(7);
  });

  // Half a model on the GPU means the other half in host RAM, which OOM'd an
  // 8 GB rig and got the miner killed. Such a card is skipped; mining continues.
  test('a card that can only hold part of the model is skipped, and mining carries on', async () => {
    const m = load();
    // 5000 MB free minus the 2048 reserve = 2952, short of the model's 3800. It
    // clears the preflight floor, so the all-or-nothing offload gate is what skips it.
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'A4000', usedMb: 1000, totalMb: 6000 }]);
    const p = m.run(argvBoth);
    await settle();

    expect(m.LlmManager.instances.length).toBe(0);
    expect(allErr()).toContain('not enough free VRAM');
    const miner = m.PearlEngine.instances[0];
    expect(miner.start).toHaveBeenCalled();
    miner.emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  // The LLM now starts after the miner, so a miner that cannot launch ends the
  // run before any model is fetched or any llama-server spawned.
  test('a miner that fails to launch exits 1 before any LLM is fetched or started', async () => {
    const m = load();
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'A4000', usedMb: 1000, totalMb: 16000 }]);
    m.PearlEngine.startError = new Error('spawn ENOENT');
    const p = m.run(['-a', ADDR, '--mode', 'auto', '--no-update', '--no-report']);
    await expect(p).resolves.toBe(1);
    await settle();
    expect(allErr()).toContain('failed to launch engine: spawn ENOENT');
    expect(m.LlmManager.instances).toHaveLength(0);
    expect(m.LlmEngineManager.instances).toHaveLength(0);
  });

  // The point of the change: on a first run the model is several GB, and the
  // miner used to wait for it. Now the miner starts first and the download runs
  // beside it, printed as whole lines so it doesn't overwrite the mining line.
  test('mines before it fetches the model, then starts the LLM beside the miner', async () => {
    const m = load();
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'A4000', usedMb: 1000, totalMb: 16000 }]);
    const p = m.run(['-a', ADDR, '--mode', 'auto', '--no-update', '--no-report', '--no-serve']);
    await settle(8);

    const miner = m.PearlEngine.instances[0];
    const server = m.LlmEngineManager.instances.find((e) => e.ensureServer.mock.calls.length);
    const model = m.LlmEngineManager.instances.find((e) => e.ensureModel.mock.calls.length);
    expect(miner.start.mock.invocationCallOrder[0]).toBeLessThan(server.ensureServer.mock.invocationCallOrder[0]);
    expect(miner.start.mock.invocationCallOrder[0]).toBeLessThan(model.ensureModel.mock.invocationCallOrder[0]);
    expect(allOut()).toContain('downloading… 10%\n');
    expect(allOut()).toContain('downloading model… 20%\n');
    expect(allOut()).not.toContain('\r  downloading');
    // The mock reports 20% then 25%: the same tenth, so one line, not two.
    expect(allOut().match(/downloading model… 20%/g)).toHaveLength(1);

    // Planned against the empty card, as before: the whole model on GPU 0.
    const llm = m.LlmManager.instances[0];
    expect(llm.start).toHaveBeenCalledWith(expect.objectContaining({ nGpuLayers: ALL_LAYERS, mainGpu: 0 }));
    llm.emit('stopped', 3);   // the LLM it started is watched like the up-front one
    expect(allErr()).toContain('local LLM exited (code 3)');

    miner.emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('an LLM that fails to set up beside the miner leaves mining running', async () => {
    const m = load();
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'A4000', usedMb: 1000, totalMb: 16000 }]);
    m.LlmEngineManager.serverError = new Error('HTTP 404');
    const p = m.run(['-a', ADDR, '--mode', 'auto', '--no-update', '--no-report', '--no-serve']);
    await settle(8);

    expect(allErr()).toContain('LLM setup failed: HTTP 404');
    expect(m.LlmManager.instances).toHaveLength(0);
    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('an error setting up the LLM beside the miner is logged, not fatal', async () => {
    const m = load();
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'A4000', usedMb: 1000, totalMb: 16000 }]);
    m.LlmEngineManager.serverInstalled = true;
    m.LlmEngineManager.modelInstalled = true;
    // Anything that throws in the not-awaited start.
    m.probe.detectPciBusIds.mockRejectedValue(new Error('nvidia-smi exploded'));
    const p = m.run(['-a', ADDR, '--mode', 'auto', '--no-update', '--no-report']);
    await settle(8);

    expect(allErr()).toContain('LLM start failed: nvidia-smi exploded — mining carries on');
    let done = false;
    p.then(() => { done = true; });
    await settle();
    expect(done).toBe(false);   // still mining
    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  // Serving is on and there is no stored identity, so one is minted -- and the
  // key file can't be written. The model runs, locally.
  test('an identity that cannot be written leaves the LLM beside the miner local-only', async () => {
    const m = load();
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'A4000', usedMb: 1000, totalMb: 16000 }]);
    m.LlmEngineManager.serverInstalled = true;
    m.LlmEngineManager.modelInstalled = true;
    m.nodeStore.getOrCreateNode.mockImplementation(() => { throw new Error('ENOSPC: no space left on device'); });
    const p = m.run(['-a', ADDR, '--mode', 'auto', '--no-update', '--no-report']);
    await settle(8);

    expect(allErr()).toContain('could not set up this node\'s identity (ENOSPC: no space left on device) — running the LLM locally only');
    const llm = m.LlmManager.instances[0];
    llm.emit('ready', { baseUrl: llm.baseUrl });
    await settle();
    expect(m.JobWorker.instances).toHaveLength(0);
    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('a run stopped while the model downloads leaves no llama-server behind', async () => {
    const m = load();
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'A4000', usedMb: 1000, totalMb: 16000 }]);
    let release;
    m.LlmEngineManager.modelHold = new Promise((r) => { release = r; });
    const p = m.run(['-a', ADDR, '--mode', 'auto', '--no-update', '--no-report', '--no-serve']);
    await settle(8);
    expect(m.LlmManager.instances).toHaveLength(0);   // still downloading

    fire('SIGINT');
    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);

    release();
    await settle(8);
    // The download finished after the run ended: the server it spawned is stopped.
    expect(m.LlmManager.instances[0].stop).toHaveBeenCalled();
  });

  test('the board report tags the GPU serving the local LLM', async () => {
    const m = load();
    m.nodeStore.loadNode.mockReturnValue(makeNode({ connected: true, name: 'rig' }));
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 4090', usedMb: 2000, totalMb: 24000 }]);
    // 'both' with reporting on (no --no-report), so the miner report path runs
    // while the LLM serves.
    const p = m.run(['-a', ADDR, '--mode', 'both', '--no-update', '--llm-binary', '/lb', '--llm-model', '/lm']);
    await settle();

    const llm = m.LlmManager.instances[0];
    llm.emit('ready', { baseUrl: llm.baseUrl }); // GPU 0 now serves the model
    await settle();

    m.probe.postMinerReport.mockClear();
    const reporter = intervalFor(NETWORK.reportIntervalMs);
    await reporter.fn();
    const payloads = m.probe.postMinerReport.mock.calls.map((c) => c[0]);
    expect(payloads.some((pl) => pl.llmModel === LLM.model.name)).toBe(true);

    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });
});

// ── connect subcommand ───────────────────────────────────────────────────────

describe('connect subcommand', () => {
  test('reports every flag-parse error with usage', async () => {
    const m = load();
    await expect(m.run(['connect', '--bogus', '--token', '--name'])).resolves.toBe(1);
    expect(allErr()).toContain('error: unknown option: --bogus');
    expect(allErr()).toContain('error: missing value for --token');
    expect(allErr()).toContain('error: missing value for --name');
    expect(allErr()).toContain('usage: llmjob-earn-cli connect --token');
  });

  test('without a token on an unlinked node it points at the dashboard', async () => {
    const m = load();
    await expect(m.run(['connect'])).resolves.toBe(1);
    expect(allErr()).toContain('no pairing token yet');
    expect(allErr()).toContain(NODE.dashboardUrl);
  });

  test('joins with a token, saves the link, then pings until SIGINT', async () => {
    const m = load();
    const node = makeNode();
    m.nodeStore.getOrCreateNode.mockReturnValue(node);
    m.io.postJson.mockResolvedValueOnce({ status: 200, data: { user: 'bob' } });
    m.probe.detectVram.mockResolvedValue({ totalMb: 100, usedMb: 10 });
    m.probe.detectGpuInfo.mockResolvedValue({ name: 'NVIDIA GeForce RTX 4090', count: 1 });

    const p = m.run(['connect', '--token=tok123', '--name', 'MyRig', '--server', 'https://srv.example']);
    await settle();

    expect(node.serverUrl).toBe('https://srv.example');
    expect(node.connected).toBe(true);
    expect(node.name).toBe('MyRig');
    expect(m.nodeStore.saveNode).toHaveBeenCalledTimes(2);
    expect(m.io.postJson.mock.calls[0][0]).toBe('https://srv.example/api/nodes/join');
    expect(m.io.postJson.mock.calls[0][1]).toEqual(expect.objectContaining({ token: 'tok123', name: 'MyRig' }));
    expect(allOut()).toContain('✓ linked to bob’s account as MyRig');
    expect(allOut()).toContain('✓ ping'); // verbose keep-alive started
    // Sparse telemetry carried the probe results.
    expect(m.io.postJson.mock.calls[1][1]).toEqual(expect.objectContaining({
      vramTotal: 100, vramUsed: 10, device: 'NVIDIA GeForce RTX 4090', name: 'MyRig',
    }));

    fire('SIGINT');
    await expect(p).resolves.toBe(0);
    expect(allOut()).toContain('stopped pinging');
  });

  // The GPU name is only a display hint, so a probe that blows up must
  // degrade to "unknown device" rather than take the ping (or the run) down.
  test('a GPU probe that rejects leaves the ping device null', async () => {
    const m = load();
    const node = makeNode();
    m.nodeStore.getOrCreateNode.mockReturnValue(node);
    m.io.postJson.mockResolvedValueOnce({ status: 200, data: { user: 'bob' } });
    m.probe.detectVram.mockResolvedValue({ totalMb: 100, usedMb: 10 });
    m.probe.detectGpuInfo.mockRejectedValue(new Error('nvidia-smi exploded'));

    const p = m.run(['connect', '--token=tok123', '--server', 'https://srv.example']);
    await settle();

    // Telemetry omits the field entirely rather than carrying a bad name.
    const pings = m.io.postJson.mock.calls.slice(1).map((c) => c[1]);
    expect(pings.length).toBeGreaterThan(0);
    for (const body of pings) expect(body.device == null).toBe(true);

    fire('SIGINT');
    await expect(p).resolves.toBe(0);
  });

  test('a 201 join without a user links "your account" under the hostname worker name', async () => {
    const m = load();
    m.io.postJson.mockResolvedValueOnce({ status: 201, data: {} });
    // The GPU probe explodes synchronously — nodeClient.deviceName absorbs it.
    m.cp.execFile.mockImplementation(() => { throw new Error('spawn fail'); });
    const p = m.run(['connect', '-t', 'tok']);
    await settle();
    expect(allOut()).toContain('✓ linked to your account as rig-host');
    fire('SIGTERM');
    await expect(p).resolves.toBe(0);
  });

  test('exits 1 when the server is unreachable during join', async () => {
    const m = load();
    m.io.postJson.mockRejectedValueOnce(new Error('ECONNREFUSED'));
    await expect(m.run(['connect', '-t', 'tok'])).resolves.toBe(1);
    expect(allErr()).toContain('could not reach ' + NODE.serverUrl + ': ECONNREFUSED');
  });

  [
    { res: { status: 403, data: { error: 'bad token' }, raw: 'x' }, text: 'join failed (HTTP 403): bad token' },
    { res: { status: 500, data: null, raw: 'boom' }, text: 'join failed (HTTP 500): boom' },
    { res: { status: 500, data: {}, raw: null }, text: 'join failed (HTTP 500): ' },
  ].forEach((c) => {
    test('exits 1 on join HTTP ' + c.res.status + ' (' + (c.res.raw || 'no raw') + ')', async () => {
      const m = load();
      // A node whose stored server matches --server (no re-save needed).
      m.nodeStore.getOrCreateNode.mockReturnValue(makeNode({ serverUrl: 'https://s.example' }));
      m.io.postJson.mockResolvedValueOnce(c.res);
      await expect(m.run(['connect', '-t', 'tok', '--server', 'https://s.example'])).resolves.toBe(1);
      expect(m.nodeStore.saveNode).not.toHaveBeenCalled();
      expect(allErr()).toContain(c.text);
    });
  });

  test('resumes pinging a linked node by name, logging ping failures verbosely', async () => {
    const m = load();
    m.nodeStore.getOrCreateNode.mockReturnValue(makeNode({ connected: true, name: 'rig' }));
    const p = m.run(['connect']);
    await settle();
    expect(allOut()).toContain('resuming pings for rig');
    expect(allOut()).toContain('✓ ping');

    const pinger = intervalFor(NODE.pingIntervalMs);
    m.io.postJson.mockResolvedValueOnce({ status: 500 });
    await pinger.fn();
    expect(allErr()).toContain('✗ ping failed (HTTP 500)');
    m.io.postJson.mockRejectedValueOnce(new Error('net down'));
    m.probe.detectVram.mockRejectedValueOnce(new Error('smi gone')); // sparse telemetry absorbs it
    await pinger.fn();
    expect(allErr()).toContain('✗ ping error: net down');

    fire('SIGINT');
    await expect(p).resolves.toBe(0);
  });

  test('resumes pinging an unnamed node by its nodeId', async () => {
    const m = load();
    m.nodeStore.getOrCreateNode.mockReturnValue(makeNode({ connected: true }));
    const p = m.run(['connect']);
    await settle();
    expect(allOut()).toContain('resuming pings for abc123');
    fire('SIGTERM');
    await expect(p).resolves.toBe(0);
  });
});

// ── demand-driven auto ───────────────────────────────────────────────────────

// autoGate binds a real HTTP server, and this suite mocks `net` down to
// `connect` — so http can never listen here. Same treatment as llmManager and
// the fleet: the wiring is asserted, and autoGate's own lifecycle is covered by
// test/autoGate.test.js against a real socket.
jest.mock('../src/main/autoGate', () => {
  const createAutoGate = jest.fn((opts) => {
    const inst = {
      opts,
      started: false,
      switching: false,
      // The real createAutoGate returns its LlmGate; anything reading node state
      // (the stats file, and the watch view) goes through it.
      gate: { state: 'MINING', inFlight: 0, quietMs: 60000, quietFor: () => 0 },
      isSwitching() { return this.switching; },
      setCtxSize: jest.fn(),
      start() { this.started = true; return this; },
      stop: jest.fn(),
    };
    createAutoGate.instances.push(inst);
    return inst;
  });
  createAutoGate.instances = [];
  // llm mode stands up the same server with nothing to switch, so the CLI can
  // keep one `auto` variable and one teardown path.
  const createServeGate = jest.fn((opts) => {
    const inst = {
      opts,
      started: false,
      isSwitching() { return false; },
      gate: { state: 'SERVING', inFlight: 0, quietMs: Infinity, quietFor: () => 0 },
      setCtxSize: jest.fn(),
      start() { this.started = true; return this; },
      stop: jest.fn(),
    };
    createServeGate.instances.push(inst);
    return inst;
  });
  createServeGate.instances = [];
  return { createAutoGate, createServeGate };
});


describe('--gate-quiet', () => {
  const demandRig = (m) => {
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 5090', usedMb: 0, totalMb: 32149 }]);
    m.LlmEngineManager.serverInstalled = true;
    m.LlmEngineManager.modelInstalled = true;
    m.LlmEngineManager.mmprojInstalled = true;
  };

  test('the product default is used when the flag is absent', async () => {
    const m = load();
    demandRig(m);
    m.run(['--address', ADDR, '--no-update', '--no-serve', '--gate-port', '0']);
    await settle();
    expect(m.autoGate.createAutoGate.instances[0].opts.quietMs).toBe(60000);
    expect(allOut()).toContain('60s with no requests hands the GPU back to mining');
  });

  // The reason the flag exists: an agent pauses between turns to think or run a
  // tool, and a release drops the prompt cache, so the next turn re-prefills the
  // whole context.
  test('an explicit window reaches the gate and is announced', async () => {
    const m = load();
    demandRig(m);
    m.run(['--address', ADDR, '--no-update', '--no-serve', '--gate-port', '0',
      '--gate-quiet', '900']);
    await settle();
    expect(m.autoGate.createAutoGate.instances[0].opts.quietMs).toBe(900000);
    expect(allOut()).toContain('900s with no requests hands the GPU back to mining');
  });

  test('0 pins the card to serving and says so in words', async () => {
    const m = load();
    demandRig(m);
    m.run(['--address', ADDR, '--no-update', '--no-serve', '--gate-port', '0',
      '--gate-quiet', '0']);
    await settle();
    expect(m.autoGate.createAutoGate.instances[0].opts.quietMs).toBe(Infinity);
    expect(allOut()).toContain('no amount of quiet with no requests hands the GPU back to mining');
  });
});

// The lock itself is PearlMiner's (see pearlMiner.test.js) and the per-card
// plan is shared/memClock's; the CLI's part is feeding the plan what it knows
// -- the request, nvidia-smi's compute capabilities, the mining list, and
// whether an LLM co-runs -- and handing the result to the miner. The lock must
// never be in place while an LLM is served, and a co-running LLM is served for
// the whole run.
describe('--mine-mem-clock', () => {
  const IGNORED = '--mine-mem-clock ignored: the LLM co-runs with the miner and needs full memory bandwidth';
  const RTX5090 = { index: 0, major: 12, minor: 0, driverMajor: 610 };
  const RTX4090 = { index: 0, major: 8, minor: 9, driverMajor: 610 };
  const oneCard = (m, card, name) => {
    m.probe.detectMinerGpus.mockResolvedValue([{ index: 0, name }]);
    m.probe.detectCudaCards.mockResolvedValue([card]);
  };

  test('a miner on its own gets the clock', async () => {
    const m = load();
    oneCard(m, RTX4090, 'NVIDIA GeForce RTX 4090');
    const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--no-report', '--mine-mem-clock', '7001']);
    await settle();
    const miner = m.PearlEngine.instances[0];
    expect(miner.settings.mineMemClockByIndex).toEqual({ 0: 7001 });
    expect(miner.settings.mineMemClockDefault).toBe(false);
    expect(allOut()).toContain('--mine-mem-clock 7001 MHz on GPU 0');
    expect(allErr()).not.toContain(IGNORED);
    miner.emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  // On the AMD core the plan would be nvidia-smi's: an RTX 5090 beside the AMD
  // cards would have its memory clock locked while it sits idle.
  test('an AMD core locks no clock, even with an RTX 5090 in the rig', async () => {
    await withEnv({ PEARL_CORE_VARIANT: undefined }, async () => {
      const m = load();
      oneCard(m, RTX5090, 'NVIDIA GeForce RTX 5090');
      const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--no-report', '--core', 'amd', '--gpu-index', '0,1']);
      await settle();
      const miner = m.PearlEngine.instances[0];
      expect(miner.settings.mineMemClockByIndex).toEqual({});
      expect(miner.settings.mineMemClockDefault).toBe(false);
      expect(allOut()).not.toContain('memory clock 7001');
      // HIP's indices go to the core as given, not filtered by nvidia-smi's list.
      expect(m.pearlCore.coreFactory.mock.calls[0][0]).toMatchObject({
        coreFlag: true, gpus: [{ index: 0, name: null }, { index: 1, name: null }],
      });
      expect(allErr()).not.toContain('nvidia-smi does not list it');
      miner.emit('stopped', 0);
      await expect(p).resolves.toBe(0);
    });
  });

  // The default. An RTX 5090 is locked without being asked, and the log
  // says so and names the way to turn it off.
  test('an RTX 5090 is locked by default, and told how to opt out', async () => {
    const m = load();
    oneCard(m, RTX5090, 'NVIDIA GeForce RTX 5090');
    const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--no-report']);
    await settle();
    const miner = m.PearlEngine.instances[0];
    expect(miner.settings.mineMemClockByIndex).toEqual({ 0: 7001 });
    expect(miner.settings.mineMemClockDefault).toBe(true);
    expect(allOut()).toContain(
      'memory clock 7001 MHz by default on GPU 0 (RTX 5090; --mine-mem-clock 0 leaves the driver\'s clock)');
    miner.emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('--mine-mem-clock 0 turns the default off', async () => {
    const m = load();
    oneCard(m, RTX5090, 'NVIDIA GeForce RTX 5090');
    const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--no-report', '--mine-mem-clock', '0']);
    await settle();
    const miner = m.PearlEngine.instances[0];
    expect(miner.settings.mineMemClockByIndex).toEqual({});
    expect(allOut()).toContain('--mine-mem-clock 0: memory clocks left to the driver');
    miner.emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  // Not Blackwell, nothing asked: the plan is empty and nothing is said about
  // memory clocks at all, which is what every 3090/4090 rig saw before.
  test('any other card with no flag gets no plan and no line', async () => {
    const m = load();
    oneCard(m, RTX4090, 'NVIDIA GeForce RTX 4090');
    const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--no-report']);
    await settle();
    const miner = m.PearlEngine.instances[0];
    expect(miner.settings.mineMemClockByIndex).toEqual({});
    expect(miner.settings.mineMemClockDefault).toBe(false);
    expect(allOut() + allErr()).not.toMatch(/memory clock|mine-mem-clock/);
    miner.emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  // Demand mode never serves while mining: the miner's stop() releases the lock
  // before the gate starts llama-server, and every restart after serving asks
  // for it again.
  test('demand mode keeps it, on the first start and every restart', async () => {
    const m = load();
    oneCard(m, RTX5090, 'NVIDIA GeForce RTX 5090');
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 5090', usedMb: 0, totalMb: 32149 }]);
    m.LlmEngineManager.serverInstalled = true;
    m.LlmEngineManager.modelInstalled = true;
    m.LlmEngineManager.mmprojInstalled = true;
    const p = m.run(['--address', ADDR, '--no-update', '--no-serve', '--no-report', '--gate-port', '0',
      '--mine-mem-clock', '7001']);
    await settle();

    expect(allOut()).toContain('mining until a request arrives');
    const miner = m.PearlEngine.instances[0];
    expect(miner.settings.mineMemClockByIndex).toEqual({ 0: 7001 });
    expect(m.autoGate.createAutoGate.instances[0].opts.startMinerArgs().mineMemClockByIndex).toEqual({ 0: 7001 });
    expect(allErr()).not.toContain(IGNORED);
    miner.emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('a co-running LLM drops it, and says so once', async () => {
    const m = load();
    oneCard(m, RTX4090, 'NVIDIA GeForce RTX 4090');
    // 24 GB free: the same model wins with and without the mining reserve, so
    // auto co-runs.
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 4090', usedMb: 2000, totalMb: 24000 }]);
    m.LlmEngineManager.serverInstalled = true;
    m.LlmEngineManager.modelInstalled = true;
    const p = m.run(['--address', ADDR, '--no-update', '--no-serve', '--no-report', '--mine-mem-clock', '7001']);
    await settle();

    expect(m.LlmManager.instances.length).toBeGreaterThan(0);
    const miner = m.PearlEngine.instances[0];
    expect(miner.settings.mineMemClockByIndex).toEqual({});
    expect(allErr().split(IGNORED)).toHaveLength(2);
    miner.emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  // The default is dropped the same way, but nobody asked for it, so the line
  // is information on stdout, not a complaint on stderr.
  test('a co-running LLM skips the RTX 5090 default, and says so on stdout', async () => {
    const m = load();
    oneCard(m, RTX5090, 'NVIDIA GeForce RTX 5090');
    // Plenty free: the same model wins with and without the mining reserve.
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 5090', usedMb: 2000, totalMb: 24000 }]);
    m.LlmEngineManager.serverInstalled = true;
    m.LlmEngineManager.modelInstalled = true;
    const p = m.run(['--address', ADDR, '--no-update', '--no-serve', '--no-report']);
    await settle();

    expect(m.LlmManager.instances.length).toBeGreaterThan(0);
    const miner = m.PearlEngine.instances[0];
    expect(miner.settings.mineMemClockByIndex).toEqual({});
    expect(allOut()).toContain(
      'memory clock left to the driver on GPU 0: the LLM co-runs with the miner and needs full memory bandwidth');
    expect(allErr()).not.toContain(IGNORED);
    miner.emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });
});

describe('auto mode on a card that cannot co-run its best model', () => {
  test('mines first and does not start the LLM until something asks for it', async () => {
    const m = load();
    // An idle 5090: 32,149 MiB is what CUDA exposes. Qwen3.8 needs 30,720, which
    // fits alone but not alongside the mining reserve — so auto goes demand-driven
    // instead of quietly downgrading to the small default.
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 5090', usedMb: 0, totalMb: 32149 }]);
    m.LlmEngineManager.serverInstalled = true;
    m.LlmEngineManager.modelInstalled = true;
    m.LlmEngineManager.mmprojInstalled = true;
    const p = m.run(['--address', ADDR, '--no-update', '--no-serve', '--gate-port', '0']);
    await settle();

    expect(allOut()).toContain('needs the GPU to itself');
    expect(allOut()).toContain('mining until a request arrives');
    // The point of demand mode: the miner runs, the model does not.
    expect(m.PearlEngine.instances).toHaveLength(1);
    expect(m.PearlEngine.instances[0].start).toHaveBeenCalled();
    expect(m.LlmManager.instances).toHaveLength(0);

    // The gate was handed working callbacks, not just constructed. Drive them the
    // way a request would, and check the LLM is sized against the WHOLE card --
    // no mining reserve, because nothing is co-resident in demand mode.
    const gate = m.autoGate.createAutoGate.instances[0];
    expect(gate.started).toBe(true);
    expect(gate.opts.modelName).toBe(LLM.tiers[0].name);
    expect(gate.opts.quietMs).toBe(LLM.gate.quietMs);
    expect(gate.opts.startMinerArgs()).toEqual(expect.objectContaining({ endpoint: expect.any(String) }));
    expect(gate.opts.isLlmReady()).toBe(false);

    const waking = gate.opts.startLlm();
    await settle();
    const llm = m.LlmManager.instances[0];
    expect(llm.start).toHaveBeenCalledWith(expect.objectContaining({ ctxSize: 262144 }));
    // The gate holds the request until the server actually reports ready, which
    // is what lets the proxy park a caller through the ~4s load.
    llm.emit('ready', { baseUrl: 'http://127.0.0.1:8080' });
    await waking;
    expect(gate.opts.isLlmReady()).toBe(true);

    // An LLM that dies on its own, NOT during a switch, must still tear serving
    // down — the switching flag is what tells those apart.
    llm.emit('stopped', 1);
    await settle();

    await gate.opts.stopLlm();
    expect(llm.stop).toHaveBeenCalled();
    expect(gate.opts.isLlmReady()).toBe(false);

    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
    expect(gate.stop).toHaveBeenCalled();   // torn down with everything else
  });

  test('a card whose best model co-runs is untouched — it still serves immediately', async () => {
    const m = load();
    // 24 GB free: the same model wins with and without the reserve, so nothing
    // about this node's behaviour changes.
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 4090', usedMb: 2000, totalMb: 24000 }]);
    m.LlmEngineManager.serverInstalled = true;
    m.LlmEngineManager.modelInstalled = true;
    const p = m.run(['--address', ADDR, '--no-update', '--no-serve']);
    await settle();

    expect(allOut()).not.toContain('needs the GPU to itself');
    expect(m.LlmManager.instances.length).toBeGreaterThan(0);   // co-running as before

    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });
});

describe('demand-driven auto: the model is fetched while mining', () => {
  test('downloads the model beside the miner, and a wake waits on that download', async () => {
    const m = load();
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 5090', usedMb: 0, totalMb: 32149 }]);
    m.LlmEngineManager.mmprojInstalled = true;
    let release;
    m.LlmEngineManager.modelHold = new Promise((r) => { release = r; });
    const p = m.run(['--address', ADDR, '--no-update', '--no-serve', '--gate-port', '0']);
    await settle(8);

    const miner = m.PearlEngine.instances[0];
    const model = m.LlmEngineManager.instances.find((e) => e.ensureModel.mock.calls.length);
    expect(miner.start.mock.invocationCallOrder[0]).toBeLessThan(model.ensureModel.mock.invocationCallOrder[0]);
    // Whole lines beside the miner, never the foreground's \r redraw.
    expect(allOut()).toContain('downloading model… 20%\n');
    expect(allOut()).not.toContain('\r  downloading');

    // The gate's hook hands back the same download, so a wake that arrives
    // mid-download waits for it while the miner keeps the card, instead of
    // stopping the miner and starting a second download.
    const gate = m.autoGate.createAutoGate.instances[0];
    let ready = false;
    const waiting = gate.opts.prepareLlm().then(() => { ready = true; });
    await settle();
    expect(ready).toBe(false);   // still downloading

    release();
    await waiting;
    expect(allOut()).toContain(LLM.tiers[0].name + ' is on disk; a request now waits only for it to load');
    const downloads = m.LlmEngineManager.instances.reduce((n, e) => n + e.ensureModel.mock.calls.length, 0);
    expect(downloads).toBe(1);
    expect(m.LlmManager.instances).toHaveLength(0);   // fetched, not loaded

    miner.emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('a failed download is logged and left for the first request to retry', async () => {
    const m = load();
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 5090', usedMb: 0, totalMb: 32149 }]);
    m.LlmEngineManager.serverError = new Error('HTTP 404');
    const p = m.run(['--address', ADDR, '--no-update', '--no-serve', '--gate-port', '0']);
    await settle(8);

    expect(allErr()).toContain('LLM download failed: HTTP 404');
    expect(allErr()).toContain('it will be tried again when a request arrives');
    expect(m.PearlEngine.instances[0].start).toHaveBeenCalled();   // still mining

    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });
});

describe('demand-driven auto: the switching flag and failure paths', () => {
  test('a deliberate stop is ignored, a real one is not; and the default port is used', async () => {
    const m = load();
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 5090', usedMb: 0, totalMb: 32149 }]);
    m.LlmEngineManager.serverInstalled = true;
    m.LlmEngineManager.modelInstalled = true;
    m.LlmEngineManager.mmprojInstalled = true;
    // No --gate-port: the configured default is what a real node uses.
    const p = m.run(['--address', ADDR, '--no-update', '--no-serve']);
    await settle();
    const gate = m.autoGate.createAutoGate.instances[0];
    expect(gate.opts.port).toBe(LLM.gate.port);

    // stopLlm before anything started must be a no-op, not a crash.
    await gate.opts.stopLlm();

    // While switching, an engine's exit is the gate's doing and must NOT tear the
    // process down — otherwise every wake would end the run.
    gate.switching = true;
    m.PearlEngine.instances[0].emit('stopped', 0);
    await settle();
    let done = false;
    p.then(() => { done = true; });
    await settle();
    expect(done).toBe(false);

    // The same event with the flag clear is a genuine exit.
    gate.switching = false;
    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('an LLM that fails to come up surfaces as an error the gate can report', async () => {
    const m = load();
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 5090', usedMb: 0, totalMb: 32149 }]);
    m.LlmEngineManager.serverInstalled = true;
    m.LlmEngineManager.modelInstalled = true;
    m.LlmEngineManager.mmprojInstalled = true;
    const p = m.run(['--address', ADDR, '--no-update', '--no-serve', '--gate-port', '0']);
    await settle();
    const gate = m.autoGate.createAutoGate.instances[0];

    // A card that no longer has room: the planner returns no instances, so the
    // callback hands back null. Turning that into an error is the gate's job
    // (test/autoGate.test.js), not the CLI's.
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'A4000', usedMb: 7900, totalMb: 8000 }]);
    await expect(gate.opts.startLlm()).resolves.toBeNull();

    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
    // Slow on purpose: this exercises the real VRAM-release wait giving up.
  }, 20000);

});

test('an LLM stopping during a switch does not tear serving down', async () => {
  const m = load();
  m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 5090', usedMb: 0, totalMb: 32149 }]);
  m.LlmEngineManager.serverInstalled = true;
  m.LlmEngineManager.modelInstalled = true;
  m.LlmEngineManager.mmprojInstalled = true;
  const p = m.run(['--address', ADDR, '--no-update', '--no-serve', '--gate-port', '0']);
  await settle();
  const gate = m.autoGate.createAutoGate.instances[0];

  const waking = gate.opts.startLlm();
  await settle();
  const llm = m.LlmManager.instances[0];
  llm.emit('ready', { baseUrl: 'http://127.0.0.1:8080' });
  await waking;

  // The gate stopping the LLM to hand the card back is not the LLM dying: the
  // handler must take the early return and leave the node's serve registration
  // alone, rather than unregistering it on every idle flip.
  gate.switching = true;
  llm.emit('stopped', 0);
  await settle();
  let ended = false;
  p.then(() => { ended = true; });
  await settle();
  expect(ended).toBe(false);
  gate.switching = false;

  m.PearlEngine.instances[0].emit('stopped', 0);
  await expect(p).resolves.toBe(0);
});

test('an engine that cannot start exits non-zero, not 0', async () => {
  // The core failing to construct -- no VRAM for the rank-128 profile, no
  // pearl_core.node -- makes PearlMiner.start() return false. It emits 'error'
  // but never 'stopped', because nothing started. Before this was checked the
  // run simply ran out of work and resolved 0, which under Restart=always is a
  // silent restart loop that mines nothing and looks healthy to systemd.
  const m = load();
  m.PearlEngine.startReturns = false;
  const code = await m.run(['--address', ADDR, '--mode', 'mining', '--no-update', '--no-serve']);
  expect(code).toBe(1);
  expect(allErr()).toContain('engine failed to start');
});

test('a fatal engine start ends an auto run before any LLM comes up', async () => {
  // auto mode starts the LLM only once the miner has started, so a fatal engine
  // failure ends the run with no server to orphan behind a dead node.
  const m = load();
  m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 4090', usedMb: 3000, totalMb: 24000 }]);
  m.LlmEngineManager.serverInstalled = true;
  m.LlmEngineManager.modelInstalled = true;
  m.PearlEngine.startReturns = false;
  const code = await m.run(['--address', ADDR, '--no-update', '--no-serve']);
  expect(code).toBe(1);
  await settle();
  expect(m.LlmManager.instances).toHaveLength(0);
});

test('a fatal engine start closes the demand gate rather than leaking its port', async () => {
  // The gate binds the public port. Created after the miner start, a fatal start
  // would call finish() while `auto` was still null and the gate would bind
  // afterwards on a run that had already resolved.
  const m = load();
  m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 5090', usedMb: 0, totalMb: 32149 }]);
  m.LlmEngineManager.serverInstalled = true;
  m.LlmEngineManager.modelInstalled = true;
  m.LlmEngineManager.mmprojInstalled = true;
  m.PearlEngine.startReturns = false;
  const code = await m.run(['--address', ADDR, '--no-update', '--no-serve', '--gate-port', '0']);
  expect(code).toBe(1);
  const gate = m.autoGate.createAutoGate.instances[0];
  expect(gate.started).toBe(true);
  expect(gate.stop).toHaveBeenCalled();
});

test('the gate proxies to the port the fleet actually bound, not the default', async () => {
  // llmFleet probes upward from LLM.port when it is busy, so a gate pinned to the
  // constant can dial a port the model is not on. Worse than a 502: a foreign
  // process sitting on 8080 answers in the model's place, through our endpoint.
  const m = load();
  m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 5090', usedMb: 0, totalMb: 32149 }]);
  m.LlmEngineManager.serverInstalled = true;
  m.LlmEngineManager.modelInstalled = true;
  m.LlmEngineManager.mmprojInstalled = true;
  const p = m.run(['--address', ADDR, '--no-update', '--no-serve', '--gate-port', '0']);
  await settle();
  const gate = m.autoGate.createAutoGate.instances[0];

  // Nothing serving yet: fall back to the configured port.
  expect(gate.opts.upstreamPort()).toBe(LLM.port);

  const waking = gate.opts.startLlm();
  await settle();
  const llm = m.LlmManager.instances[0];
  llm.emit('ready', { baseUrl: 'http://127.0.0.1:8087' });   // walked past a busy 8080
  await waking;
  expect(gate.opts.upstreamPort()).toBe(8087);

  gate.switching = false;
  m.PearlEngine.instances[0].emit('stopped', 0);
  await expect(p).resolves.toBe(0);
});

test('a miner that fails to restart after serving exits non-zero', async () => {
  // The same contract as the up-front start: a false return means the core did
  // not construct. Swallowed, the node reported the last hashrate forever while
  // the card did nothing, and the supervisor saw a healthy process.
  const m = load();
  m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 5090', usedMb: 0, totalMb: 32149 }]);
  m.LlmEngineManager.serverInstalled = true;
  m.LlmEngineManager.modelInstalled = true;
  m.LlmEngineManager.mmprojInstalled = true;
  const p = m.run(['--address', ADDR, '--no-update', '--no-serve', '--gate-port', '0']);
  await settle();
  const gate = m.autoGate.createAutoGate.instances[0];

  const waking = gate.opts.startLlm();
  await settle();
  m.LlmManager.instances[0].emit('ready', { baseUrl: 'http://127.0.0.1:8080' });
  await waking;

  gate.opts.onMinerFailed();
  await expect(p).resolves.toBe(1);
  expect(allErr()).toContain('engine failed to restart after serving');
  expect(m.LlmManager.instances[0].stop).toHaveBeenCalled();
});

test('a failed restart with no model loaded still exits non-zero', async () => {
  // The release path stops the LLM before restarting the miner, so the failure
  // can land with nothing to tear down.
  const m = load();
  m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 5090', usedMb: 0, totalMb: 32149 }]);
  m.LlmEngineManager.serverInstalled = true;
  m.LlmEngineManager.modelInstalled = true;
  m.LlmEngineManager.mmprojInstalled = true;
  const p = m.run(['--address', ADDR, '--no-update', '--no-serve', '--gate-port', '0']);
  await settle();
  const gate = m.autoGate.createAutoGate.instances[0];

  expect(m.LlmManager.instances).toHaveLength(0);   // never woke
  gate.opts.onMinerFailed();
  await expect(p).resolves.toBe(1);
  expect(allErr()).toContain('engine failed to restart after serving');
});

// A request during a crash restart: the gate is SERVING with no card ready, so
// it wakes the model again. That planned a second fleet on top of the one
// restarting and lost the handle to the first, which nothing could then stop.
test('a wake during a crash restart waits for the same fleet instead of starting another', async () => {
  const m = load();
  m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 5090', usedMb: 0, totalMb: 32149 }]);
  m.LlmEngineManager.serverInstalled = true;
  m.LlmEngineManager.modelInstalled = true;
  m.LlmEngineManager.mmprojInstalled = true;
  const p = m.run(['--address', ADDR, '--no-update', '--no-serve', '--gate-port', '0']);
  await settle();
  const gate = m.autoGate.createAutoGate.instances[0];

  const waking = gate.opts.startLlm();
  await settle();
  const llm = m.LlmManager.instances[0];
  llm.emit('ready', { baseUrl: 'http://127.0.0.1:8080' });
  const fleet = await waking;
  expect(gate.setCtxSize).toHaveBeenLastCalledWith(262144);

  llm.emit('crashed', { code: 139, restartInMs: 3000, attempt: 1 });
  expect(gate.opts.isLlmReady()).toBe(false);
  m.probe.detectGpusVram.mockClear();
  await expect(gate.opts.startLlm()).resolves.toBe(fleet);
  expect(m.LlmManager.instances).toHaveLength(1);
  expect(m.probe.detectGpusVram).not.toHaveBeenCalled();   // nothing planned

  // The restart could not load at the top rung: the gate is told the real window.
  llm.emit('ctx-downgrade', { from: 262144, to: 131072 });
  expect(gate.setCtxSize).toHaveBeenLastCalledWith(131072);
  llm.emit('ready', { baseUrl: 'http://127.0.0.1:8080' });
  expect(gate.opts.isLlmReady()).toBe(true);

  gate.switching = false;
  m.PearlEngine.instances[0].emit('stopped', 0);
  await expect(p).resolves.toBe(0);
});

test('handing the card back waits for the model to actually release its VRAM', async () => {
  // LlmFleet.stop() signals llama-server and returns; it does not join. Restarting
  // the miner straight away put it on a card still holding ~30 GB, and its core
  // failed to construct -- fatal, now that a failed restart is reported. Observed
  // on a 5090: 1,469 MiB free at the restart against the ~2,081 MiB it needed.
  const m = load();
  m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 5090', usedMb: 0, totalMb: 32149 }]);
  m.LlmEngineManager.serverInstalled = true;
  m.LlmEngineManager.modelInstalled = true;
  m.LlmEngineManager.mmprojInstalled = true;
  const p = m.run(['--address', ADDR, '--no-update', '--no-serve', '--gate-port', '0']);
  await settle();
  const gate = m.autoGate.createAutoGate.instances[0];

  const waking = gate.opts.startLlm();
  await settle();
  const llm = m.LlmManager.instances[0];
  llm.emit('ready', { baseUrl: 'http://127.0.0.1:8080' });
  await waking;

  m.probe.detectGpusVram.mockClear();
  await gate.opts.stopLlm();
  expect(llm.stop).toHaveBeenCalled();
  // The card was read before the gate handed control back to the miner.
  expect(m.probe.detectGpusVram).toHaveBeenCalled();

  gate.switching = false;
  m.PearlEngine.instances[0].emit('stopped', 0);
  await expect(p).resolves.toBe(0);
});

test('the engine is given a temperature reader, as the GUI has always been', async () => {
  // Without it PearlEngine never starts its temp poll, so every headless rig
  // reported temp 0 -- to the stats file, the miner report and the network board.
  const m = load();
  m.probe.detectGpuTemps.mockResolvedValue([{ index: 0, temp: 61 }]);
  const p = m.run(['--address', ADDR, '--no-update', '--no-serve', '--no-report', '--mode', 'mining']);
  await settle();
  const eng = m.PearlEngine.instances[0];
  expect(typeof eng.opts.readTemps).toBe('function');
  eng.opts.readTemps();
  expect(m.probe.detectGpuTemps).toHaveBeenCalled();
  eng.emit('stopped', 0);
  await expect(p).resolves.toBe(0);
});

test('the stats file records what the node is doing, not just its counters', async () => {
  // It used to live inside `if (plan.miner)`, so --mode llm wrote nothing, and it
  // could not see the gate or the model even when both were up. A rig reading
  // 0 TH/s in demand mode is busy serving, not broken -- the file has to say so.
  const m = load();
  m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 5090', usedMb: 0, totalMb: 32149 }]);
  m.LlmEngineManager.serverInstalled = true;
  m.LlmEngineManager.modelInstalled = true;
  m.LlmEngineManager.mmprojInstalled = true;
  const p = m.run(['--address', ADDR, '--no-update', '--no-serve', '--gate-port', '0',
    '--stats-file', '/tmp/s.json']);
  await settle();

  m.fs.writeFileSync.mockClear();
  intervalFor(10000).fn();
  const written = JSON.parse(m.fs.writeFileSync.mock.calls[0][1]);
  expect(written.mode).toBe('auto');
  expect(written.strategy).toBe('demand');
  expect(written.gate).toBe('MINING');
  expect(written.schema).toBe(1);

  m.PearlEngine.instances[0].emit('stopped', 0);
  await expect(p).resolves.toBe(0);
});

// HiveOS's h-stats.sh sends each card's bus as bus_numbers, which is what puts
// the card's hashrate on its own dashboard row rather than matching by position.
test('the stats file gives each mining card its PCI bus id', async () => {
  const m = load();
  m.probe.detectPciBusIds.mockResolvedValue({ 0: '00000000:01:00.0', 1: '00000000:02:00.0' });
  const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--stats-file', '/tmp/s.json']);
  await settle();
  expect(m.probe.detectPciBusIds).toHaveBeenCalledTimes(1);

  const eng = m.PearlEngine.instances[0];
  eng.emit('event', { type: 'status', gpuIndex: 1, hashrate: 90, accepted: 2, rejected: 0 });
  eng.emit('event', { type: 'status', gpuIndex: 0, hashrate: 100, accepted: 3, rejected: 0 });
  m.fs.writeFileSync.mockClear();
  intervalFor(10000).fn();
  const written = JSON.parse(m.fs.writeFileSync.mock.calls[0][1]);
  expect(written.gpus.map((g) => [g.index, g.hashrate, g.pciBusId])).toEqual([
    [0, 100, '00000000:01:00.0'],
    [1, 90, '00000000:02:00.0'],
  ]);

  eng.emit('stopped', 0);
  await expect(p).resolves.toBe(0);
});

test('no stats file, no bus id query', async () => {
  const m = load();
  const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update']);
  await settle();
  expect(m.probe.detectPciBusIds).not.toHaveBeenCalled();
  m.PearlEngine.instances[0].emit('stopped', 0);
  await expect(p).resolves.toBe(0);
});

test('llm mode serves on the gate port, not on llama-server\'s own', async () => {
  // The gate used to exist only in auto, so choosing llm mode silently moved
  // callers from the documented endpoint to llama-server's port -- at exactly the
  // moment the operator had committed to serving.
  const m = load();
  m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 5090', usedMb: 0, totalMb: 32149 }]);
  m.LlmEngineManager.serverInstalled = true;
  m.LlmEngineManager.modelInstalled = true;
  m.LlmEngineManager.mmprojInstalled = true;
  const p = m.run(['--mode', 'llm', '--no-update', '--no-serve', '--gate-port', '0']);
  await settle();
  const llm = m.LlmManager.instances[0];
  llm.emit('ready', { baseUrl: 'http://127.0.0.1:8080' });
  await settle();

  expect(m.autoGate.createAutoGate.instances).toHaveLength(0);   // nothing to switch
  const gate = m.autoGate.createServeGate.instances[0];
  expect(gate.started).toBe(true);
  expect(gate.opts.port).toBe(0);
  expect(gate.opts.isLlmReady()).toBe(true);
  expect(gate.opts.upstreamPort()).toBe(8080);
  expect(allOut()).toContain('serving on :0');

  // 1, not 0: an LLM-only run whose fleet dies is never a clean exit, or a
  // Restart=on-failure supervisor would never bring the node back.
  llm.emit('stopped', 0);
  await expect(p).resolves.toBe(1);
});

// The 27B's ladder halves the window per step. The gate is the endpoint
// clients read the window from while the model is down, and it kept the top
// rung, so one step down told them twice what the server had.
test('the llm-mode gate reports the window the card really loaded at', async () => {
  const m = load();
  m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 5090', usedMb: 0, totalMb: 32149 }]);
  m.LlmEngineManager.serverInstalled = true;
  m.LlmEngineManager.modelInstalled = true;
  m.LlmEngineManager.mmprojInstalled = true;
  const p = m.run(['--mode', 'llm', '--no-update', '--no-serve', '--gate-port', '0']);
  await settle();
  const gate = m.autoGate.createServeGate.instances[0];
  expect(gate.opts.ctxSize).toBe(262144);
  const llm = m.LlmManager.instances[0];
  llm.emit('ctx-downgrade', { from: 262144, to: 131072 });
  expect(gate.setCtxSize).toHaveBeenLastCalledWith(131072);
  llm.emit('ready', { baseUrl: 'http://127.0.0.1:8080' });
  expect(gate.setCtxSize).toHaveBeenLastCalledWith(131072);
  llm.emit('stopped', 0);
  await expect(p).resolves.toBe(1);
});

// Telemetry and the stats file read serveLlmState.ready. A restart after a crash
// is a model that is down, and they said ready all through it.
test('a crash restart is not reported as ready', async () => {
  const m = load();
  m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 5090', usedMb: 0, totalMb: 32149 }]);
  m.LlmEngineManager.serverInstalled = true;
  m.LlmEngineManager.modelInstalled = true;
  m.LlmEngineManager.mmprojInstalled = true;
  const p = m.run(['--mode', 'llm', '--no-update', '--no-serve', '--gate-port', '0', '--stats-file', '/tmp/s.json']);
  await settle();
  const llm = m.LlmManager.instances[0];
  const written = () => {
    m.fs.writeFileSync.mockClear();
    intervalFor(10000).fn();
    return JSON.parse(m.fs.writeFileSync.mock.calls[0][1]);
  };
  llm.emit('ready', { baseUrl: 'http://127.0.0.1:8080' });
  expect(written().model.ready).toBe(true);
  llm.emit('crashed', { code: 139, restartInMs: 3000, attempt: 1 });
  expect(written().model.ready).toBe(false);
  llm.emit('ready', { baseUrl: 'http://127.0.0.1:8080' });   // the restart landed
  expect(written().model.ready).toBe(true);
  llm.emit('stopped', 0);
  await expect(p).resolves.toBe(1);
});

test('the llm-mode gate falls back to the configured port before the fleet is up', async () => {
  const m = load();
  m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 5090', usedMb: 0, totalMb: 32149 }]);
  m.LlmEngineManager.serverInstalled = true;
  m.LlmEngineManager.modelInstalled = true;
  m.LlmEngineManager.mmprojInstalled = true;
  const p = m.run(['--mode', 'llm', '--no-update', '--no-serve', '--gate-port', '0']);
  await settle();
  const gate = m.autoGate.createServeGate.instances[0];
  expect(gate.opts.upstreamPort()).toBe(LLM.port);   // nothing ready yet

  m.LlmManager.instances[0].emit('stopped', 0);
  await expect(p).resolves.toBe(1);
});

describe('demand mode still serves the cluster', () => {
  // Cluster work is PULLED, outbound, so a NAT'd node can serve with no inbound
  // networking. But a JobWorker is only built per READY llama-server instance,
  // and demand mode has none while mining -- so the best cards in the fleet
  // stopped pulling jobs entirely and served only what reached :8000 directly.
  function serving(m) {
    m.nodeStore.loadNode.mockReturnValue(makeNode({ connected: true }));
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 5090', usedMb: 0, totalMb: 32149 }]);
    m.LlmEngineManager.serverInstalled = true;
    m.LlmEngineManager.modelInstalled = true;
    m.LlmEngineManager.mmprojInstalled = true;
  }

  test('polls for cluster jobs while mining, before any model exists', async () => {
    const m = load();
    serving(m);
    const p = m.run(['--address', ADDR, '--no-update', '--gate-port', '0']);
    await settle();

    expect(m.LlmManager.instances).toHaveLength(0);      // nothing loaded
    expect(m.JobWorker.instances).toHaveLength(1);       // yet a poller exists
    expect(m.JobWorker.instances[0].start).toHaveBeenCalled();
    expect(allOut()).toContain('polling for cluster jobs while mining');

    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('a cluster job wakes the model and holds the card for its whole run', async () => {
    const m = load();
    serving(m);
    const p = m.run(['--address', ADDR, '--no-update', '--gate-port', '0']);
    await settle();
    const gate = m.autoGate.createAutoGate.instances[0];
    const worker = m.JobWorker.instances[0];

    // Drive runJob the way JobWorker does when a job is assigned.
    let ensured = false;
    gate.gate.ensureServing = jest.fn(async () => { ensured = true; });
    gate.gate.begin = jest.fn();
    gate.gate.end = jest.fn();
    m.io.streamChatCompletion.mockReturnValue({ done: Promise.resolve() });

    await worker.opts.runJob({ messages: [] }, { onDelta: () => {}, onReasoning: () => {} });

    expect(ensured).toBe(true);
    // begin()/end() bracket the WHOLE job: shouldRelease() needs inFlight === 0,
    // so without them the quiet timer kills a long generation mid-flight.
    expect(gate.gate.begin).toHaveBeenCalled();
    expect(gate.gate.end).toHaveBeenCalled();

    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('the card is released even when the job throws', async () => {
    const m = load();
    serving(m);
    const p = m.run(['--address', ADDR, '--no-update', '--gate-port', '0']);
    await settle();
    const gate = m.autoGate.createAutoGate.instances[0];
    const worker = m.JobWorker.instances[0];
    gate.gate.ensureServing = jest.fn(async () => {});
    gate.gate.begin = jest.fn();
    gate.gate.end = jest.fn();
    m.io.streamChatCompletion.mockReturnValue({ done: Promise.reject(new Error('model died')) });

    await expect(worker.opts.runJob({ messages: [] }, { onDelta: () => {} })).rejects.toThrow('model died');
    expect(gate.gate.end).toHaveBeenCalled();   // or the node never mines again

    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('the fleet does not arm a second poller when the model wakes', async () => {
    const m = load();
    serving(m);
    const p = m.run(['--address', ADDR, '--no-update', '--gate-port', '0']);
    await settle();
    const gate = m.autoGate.createAutoGate.instances[0];

    const waking = gate.opts.startLlm();
    await settle();
    m.LlmManager.instances[0].emit('ready', { baseUrl: 'http://127.0.0.1:8080' });
    await waking;

    // Two pollers would double-poll and race each other for the same jobs.
    expect(m.JobWorker.instances).toHaveLength(1);

    // And the one poller now resolves the port the fleet actually bound, rather
    // than the null it correctly reported while nothing was loaded.
    const gate2 = m.autoGate.createAutoGate.instances[0];
    gate2.gate.ensureServing = jest.fn(async () => {});
    gate2.gate.begin = jest.fn(); gate2.gate.end = jest.fn();
    m.io.streamChatCompletion.mockReturnValue({ done: Promise.resolve() });
    await m.JobWorker.instances[0].opts.runJob({ messages: [] }, { onDelta: () => {} });
    expect(m.io.streamChatCompletion.mock.calls[0][0]).toBe('http://127.0.0.1:8080');

    gate.switching = false;
    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('stays online on the board while mining, not only while serving', async () => {
    // The fleet's ping loop is armed on its first ready card, so in demand mode
    // the node only appeared online during the seconds it happened to be
    // serving and went stale the rest of the time.
    const m = load();
    serving(m);
    const p = m.run(['--address', ADDR, '--no-update', '--gate-port', '0']);
    await settle();

    const pinger = intervalFor(NODE.pingIntervalMs);
    expect(pinger).toBeTruthy();
    expect(m.LlmManager.instances).toHaveLength(0);   // pinging with no model loaded
    m.io.postJson.mockClear();
    await pinger.fn();
    expect(m.io.postJson).toHaveBeenCalled();

    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  // A demand wake starts its fleet without workers: jobs run on demand mode's
  // own worker. Counting the fleet alone reported 0 active jobs all along.
  test('counts the jobs on its own worker in the telemetry', async () => {
    const m = load();
    serving(m);
    const p = m.run(['--address', ADDR, '--no-update', '--gate-port', '0']);
    await settle();
    m.io.postJson.mockClear();
    await intervalFor(NODE.pingIntervalMs).fn();
    expect(m.io.postJson.mock.calls[0][1].activeJobs).toBe(2);   // the mock worker runs 2
    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  // Demand mode resolves its identity up front. A disk that refused node.json
  // ended the run with 'fatal'; it mines on, serving nothing.
  test('an identity that cannot be written leaves it mining, with no poller', async () => {
    const m = load();
    serving(m);
    m.nodeStore.loadNode.mockReturnValue(null);
    m.nodeStore.getOrCreateNode.mockImplementation(() => { throw new Error('ENOSPC: no space left on device'); });
    const p = m.run(['--address', ADDR, '--no-update', '--gate-port', '0']);
    await settle();
    expect(allErr()).toContain('could not set up this node\'s identity (ENOSPC: no space left on device)');
    expect(m.JobWorker.instances).toHaveLength(0);
    expect(m.PearlEngine.instances[0].start).toHaveBeenCalled();
    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('the keep-alive handle is unrefed', async () => {
    const m = load();
    serving(m);
    const p = m.run(['--address', ADDR, '--no-update', '--gate-port', '0']);
    await settle();
    expect(intervalFor(NODE.pingIntervalMs).unref).toHaveBeenCalled();
    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('advertises the tier it will serve, not the default it has loaded', async () => {
    // Nothing is loaded while mining, so serveLlmState kept the small default --
    // and that is what the ping publishes and the server now ROUTES on. The board
    // named a model this node never serves, a job asking for the tier was never
    // offered to the one node running it, and a job asking for the default would
    // have been answered by the tier.
    const m = load();
    serving(m);
    const p = m.run(['--address', ADDR, '--no-update', '--gate-port', '0']);
    await settle();

    expect(m.LlmManager.instances).toHaveLength(0);   // nothing loaded
    const pinger = intervalFor(NODE.pingIntervalMs);
    m.io.postJson.mockClear();
    await pinger.fn();
    const body = m.io.postJson.mock.calls[0][1];
    expect(body.model).toBe(LLM.tiers[0].name);
    expect(body.model).not.toBe(LLM.model.name);

    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('--no-serve still means no poller', async () => {
    const m = load();
    serving(m);
    const p = m.run(['--address', ADDR, '--no-update', '--no-serve', '--gate-port', '0']);
    await settle();
    expect(m.JobWorker.instances).toHaveLength(0);
    expect(allOut()).not.toContain('polling for cluster jobs');
    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });
});

describe('shutdown and startup edges the gate introduced', () => {
  const serving = (m) => {
    m.nodeStore.loadNode.mockReturnValue(makeNode({ connected: true }));
    m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 5090', usedMb: 0, totalMb: 32149 }]);
    m.LlmEngineManager.serverInstalled = true;
    m.LlmEngineManager.modelInstalled = true;
    m.LlmEngineManager.mmprojInstalled = true;
  };

  test('a signal while the gate is serving resolves the run', async () => {
    // The gate has already stopped the miner, and PearlEngine.stop() on a stopped
    // miner emits nothing -- but `miner` was still truthy, so the else arm never
    // ran and finish() was never called. The process sat until TimeoutStopSec
    // while its release timer restarted the miner underneath the shutdown. A
    // self-update IS a restart, so the rollout mechanism hit this every time.
    const m = load();
    serving(m);
    const p = m.run(['--address', ADDR, '--no-update', '--no-serve', '--gate-port', '0']);
    await settle();
    const gate = m.autoGate.createAutoGate.instances[0];

    const waking = gate.opts.startLlm();
    await settle();
    m.LlmManager.instances[0].emit('ready', { baseUrl: 'http://127.0.0.1:8080' });
    await waking;

    // What the gate leaves behind: the engine is down but the object remains.
    m.PearlEngine.instances[0].isRunning.mockReturnValue(false);
    fire('SIGTERM');
    await expect(p).resolves.toBe(0);
    expect(gate.stop).toHaveBeenCalled();     // and the public port is released
  });

  test('the board keeps showing the model a demand node serves while it mines', async () => {
    // In demand mode nothing is loaded between requests, so keying the board row
    // off a live fleet left the model column blank for a node whose whole purpose
    // is to serve that model -- it appeared only for the seconds either side of a
    // request. Not resident is not the same as not available.
    const m = load();
    serving(m);
    const p = m.run(['--address', ADDR, '--no-update', '--gate-port', '0']);
    await settle();

    expect(m.LlmManager.instances).toHaveLength(0);   // nothing loaded
    m.probe.postMinerReport.mockClear();
    await intervalFor(NETWORK.reportIntervalMs).fn();
    const rows = m.probe.postMinerReport.mock.calls.map((c) => c[0]);
    const withModel = rows.filter((r) => r && r.llmModel);
    expect(withModel.length).toBeGreaterThan(0);
    expect(withModel[0].llmModel).toBe(LLM.tiers[0].name);
    expect(withModel[0].nodeId).toBeTruthy();

    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('--no-report skips the board row without tripping the demand refresh', async () => {
    const m = load();
    serving(m);
    const p = m.run(['--address', ADDR, '--no-update', '--no-report', '--gate-port', '0']);
    await settle();
    expect(m.probe.postMinerReport).not.toHaveBeenCalled();
    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('a demand node with no readable card reports no serving index', async () => {
    // pickLlmGpu can come back empty (no driver, unreadable VRAM); the row is
    // then posted without a card tagged rather than throwing.
    const m = load();
    serving(m);
    const p = m.run(['--address', ADDR, '--no-update', '--gate-port', '0']);
    await settle();
    m.probe.detectGpusVram.mockResolvedValue([]);
    m.probe.postMinerReport.mockClear();
    await expect(intervalFor(NETWORK.reportIntervalMs).fn()).resolves.toBeDefined();

    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('a card that cannot mine still serves instead of exiting 1', async () => {
    // plan.miner stays true when the mining core fails to load, so the node
    // believed it was in demand mode, skipped the up-front LLM start, never built
    // the gate (which needs a miner) and fell through to "nothing to run".
    const m = load();
    serving(m);
    m.pearlCore.coreFactory.mockReturnValue(null);   // the mining core will not load
    const p = m.run(['--address', ADDR, '--no-update', '--no-serve', '--gate-port', '0']);
    await settle();
    expect(m.LlmManager.instances.length).toBeGreaterThan(0);   // it serves
    m.LlmManager.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(1);
  });
});

// ── HiveOS ───────────────────────────────────────────────────────────────────
// h-run.sh sets LLMJOB_HIVEOS=1. The CLI then updates its whole package folder,
// runs a new version on trial, and posts problems to the dashboard.
describe('under HiveOS', () => {
  const { TRIAL_CHECK_MS } = require('../src/shared/hiveos');
  const DIR = '/hive/miners/custom/llmjob-earn';
  const UPDATE = {
    updateAvailable: true,
    reason: 'update-available',
    currentVersion: pkg.version,
    latestVersion: '9.9.9',
    packageUrl: 'https://dl/llmjob-earn-9.9.9.tar.gz',
  };
  function hiveLoad() {
    const m = load();
    m.hive.isHiveos.mockReturnValue(true);
    return m;
  }
  let now;
  function clock() {
    now = 1e12;
    jest.spyOn(Date, 'now').mockImplementation(() => now);
  }

  test('an argument error goes to the dashboard as well as the log', async () => {
    const m = hiveLoad();
    await expect(m.run(['--address', 'krx123', '--mode', 'mining'])).resolves.toBe(1);
    expect(allErr()).toContain('error: invalid Pearl address: krx123');
    expect(m.hive.postMessage).toHaveBeenCalledWith('danger', 'LLMJob Earn can\'t start: krx123 is not a Pearl '
      + 'address. Put your prl1p… address in the flight sheet\'s wallet template.');
  });

  test('off HiveOS none of it happens', async () => {
    const m = load();
    m.selfUpdater.isPackaged.mockReturnValue(true);
    await expect(m.run(['--address', 'krx123'])).resolves.toBe(1);
    expect(m.hive.postMessage).not.toHaveBeenCalled();

    const m2 = load();
    m2.selfUpdater.isPackaged.mockReturnValue(true);
    const p = m2.run(['--address', ADDR, '--mode', 'mining']);
    await settle();
    expect(m2.hive.cleanStaging).not.toHaveBeenCalled();
    expect(m2.hive.readTrial).not.toHaveBeenCalled();
    expect(m2.hive.checkDriver).not.toHaveBeenCalled();
    expect(intervalFor(NETWORK.updateCheckIntervalMs)).toBeUndefined();
    expect(intervalFor(TRIAL_CHECK_MS)).toBeUndefined();
    m2.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('a new release replaces the whole package, and the CLI exits 0 for HiveOS to start it', async () => {
    const m = hiveLoad();
    m.selfUpdater.fetchLatestRelease.mockResolvedValue({ version: '9.9.9' });
    m.selfUpdater.isPackaged.mockReturnValue(true);
    m.selfUpdate.planUpdate.mockReturnValue(UPDATE);
    await expect(m.run(['--address', ADDR, '--mode', 'mining'])).resolves.toBe(0);
    expect(m.hive.cleanStaging).toHaveBeenCalledWith(DIR);
    expect(m.hive.isSkipped).toHaveBeenCalledWith(DIR, '9.9.9', expect.any(Number));
    expect(m.hive.updatePackage).toHaveBeenCalledWith(UPDATE);
    expect(m.selfUpdater.applyUpdate).not.toHaveBeenCalled();
    expect(m.selfUpdater.reexec).not.toHaveBeenCalled();
    expect(m.PearlEngine.instances).toHaveLength(0);
    expect(allOut()).toContain('updating the HiveOS package ' + pkg.version + ' → 9.9.9…');
    expect(allOut()).toContain('updated to v9.9.9; HiveOS starts it in a few seconds');
  });

  test('a release that failed here lately, has no package yet, or will not install: mining carries on', async () => {
    const cases = [
      [(m) => m.hive.isSkipped.mockReturnValue(true),
        'v9.9.9 is out, but it failed on this rig in the last day; staying on v' + pkg.version],
      [(m) => m.selfUpdate.planUpdate.mockReturnValue(Object.assign({}, UPDATE, { packageUrl: null })),
        'v9.9.9 is out, but its HiveOS package is not up yet; staying on v' + pkg.version],
      [(m) => m.hive.updatePackage.mockRejectedValue(new Error('HTTP 404')),
        'update failed: HTTP 404; staying on v' + pkg.version],
    ];
    for (const [setup, said] of cases) {
      out = [];
      err = [];
      const m = hiveLoad();
      m.selfUpdater.fetchLatestRelease.mockResolvedValue({ version: '9.9.9' });
      m.selfUpdater.isPackaged.mockReturnValue(true);
      m.selfUpdate.planUpdate.mockReturnValue(UPDATE);
      setup(m);
      m.PearlEngine.startError = new Error('core not built');
      await expect(m.run(['--address', ADDR, '--mode', 'mining'])).resolves.toBe(1);
      expect(allOut() + allErr()).toContain(said);
      expect(m.PearlEngine.instances).toHaveLength(1);
    }
  });

  test('a core that will not load is posted; a missing one is only logged', async () => {
    const m = hiveLoad();
    m.pearlCore.coreFactory.mockImplementation((opts) => {
      opts.log('error', 'Pearl core: /x/pearl_core.node is there but would not load: /lib/libc.so.6: '
        + "version `GLIBC_2.34' not found (required by /x/pearl_core.node)");
      return null;
    });
    await expect(m.run(['--address', ADDR, '--mode', 'mining', '--no-update'])).resolves.toBe(1);
    expect(m.hive.postMessage).toHaveBeenCalledWith('danger', 'LLMJob Earn can\'t mine: this HiveOS image is too '
      + 'old (the mining core needs GLIBC_2.34). Update the rig to the latest HiveOS image.');

    const m2 = hiveLoad();
    m2.pearlCore.coreFactory.mockReturnValue(null);
    await expect(m2.run(['--address', ADDR, '--mode', 'mining', '--no-update'])).resolves.toBe(1);
    expect(m2.hive.postMessage).not.toHaveBeenCalled();
  });

  test('Blackwell cards on an old driver are logged and posted', async () => {
    const msg = 'LLMJob Earn: the RTX 50 or other Blackwell card at 02:00.0 needs NVIDIA driver 570 or newer, '
      + 'and this rig has 550. Run nvidia-driver-update in Hive Shell.';
    const m = hiveLoad();
    m.hive.checkDriver.mockResolvedValue(msg);
    const p = m.run(['--address', ADDR, '--mode', 'mining', '--no-update']);
    await settle();
    expect(allErr()).toContain(msg);
    expect(m.hive.postMessage).toHaveBeenCalledWith('warning', msg);
    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('checks for a release every few hours, and restarts into one', async () => {
    const m = hiveLoad();
    m.selfUpdater.isPackaged.mockReturnValue(true);
    const p = m.run(['--address', ADDR, '--mode', 'mining']);
    await settle();
    const timer = intervalFor(NETWORK.updateCheckIntervalMs);
    expect(timer.unref).toHaveBeenCalled();
    const miner = m.PearlEngine.instances[0];

    // Offline, up to date, a failed check, and from source: mining carries on.
    timer.fn();
    await settle();
    m.selfUpdater.fetchLatestRelease.mockResolvedValue({ version: pkg.version });
    timer.fn();
    await settle();
    m.selfUpdater.fetchLatestRelease.mockRejectedValue(new Error('socket hang up'));
    timer.fn();
    await settle();
    expect(allErr()).toContain('update check failed: socket hang up');
    m.selfUpdater.fetchLatestRelease.mockResolvedValue({ version: '9.9.9' });
    m.selfUpdate.planUpdate.mockReturnValue(UPDATE);
    m.selfUpdater.isPackaged.mockReturnValue(false);
    timer.fn();
    await settle();
    expect(m.hive.updatePackage).not.toHaveBeenCalled();
    expect(miner.stop).not.toHaveBeenCalled();

    // A new release: swap it in, then shut down cleanly, exit code 0.
    m.selfUpdater.isPackaged.mockReturnValue(true);
    timer.fn();
    timer.fn(); // still checking: nothing
    await settle();
    expect(m.hive.updatePackage).toHaveBeenCalledTimes(1);
    expect(allOut()).toContain('shutting down…');
    expect(miner.stop).toHaveBeenCalled();
    timer.fn(); // stopping: nothing
    await settle();
    expect(m.hive.updatePackage).toHaveBeenCalledTimes(1);
    miner.emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('--no-update keeps a rig on its version', async () => {
    const m = hiveLoad();
    m.selfUpdater.isPackaged.mockReturnValue(true);
    const p = m.run(['--address', ADDR, '--mode', 'mining', '--no-update']);
    await settle();
    expect(m.selfUpdater.fetchLatestRelease).not.toHaveBeenCalled();
    expect(intervalFor(NETWORK.updateCheckIntervalMs)).toBeUndefined();
    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('a version on trial passes once it hashes, and says so', async () => {
    const m = hiveLoad();
    m.hive.readTrial.mockReturnValue({ from: '0.6.9', to: pkg.version });
    const p = m.run(['--address', ADDR, '--mode', 'mining', '--no-update']);
    await settle();
    expect(m.hive.readTrial).toHaveBeenCalledWith(DIR);
    const trial = intervalFor(TRIAL_CHECK_MS);
    expect(trial.unref).toHaveBeenCalled();
    const miner = m.PearlEngine.instances[0];
    trial.fn();
    miner.emit('event', { type: 'status', gpuIndex: 0, hashrate: 0, accepted: 0, rejected: 0 });
    trial.fn();
    expect(m.hive.confirmTrial).not.toHaveBeenCalled();
    miner.emit('event', { type: 'status', gpuIndex: 0, hashrate: 120, accepted: 0, rejected: 0 });
    trial.fn();
    expect(m.hive.confirmTrial).toHaveBeenCalledWith(DIR);
    expect(clearInterval).toHaveBeenCalledWith(trial);
    expect(allOut()).toContain('the update from v0.6.9 to v' + pkg.version + ' is done');
    expect(m.hive.postMessage).toHaveBeenCalledWith('success', 'LLMJob Earn updated to v' + pkg.version + ' from v0.6.9.');
    miner.emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('a version on trial that has not hashed in ten minutes exits 75, for h-run.sh to roll back', async () => {
    clock();
    const m = hiveLoad();
    m.hive.readTrial.mockReturnValue({ from: '0.6.9', to: pkg.version });
    const p = m.run(['--address', ADDR, '--mode', 'mining', '--no-update']);
    await settle();
    const trial = intervalFor(TRIAL_CHECK_MS);
    now += 10 * 60 * 1000 - 1;
    trial.fn();
    const miner = m.PearlEngine.instances[0];
    expect(miner.stop).not.toHaveBeenCalled();
    now += 1;
    trial.fn();
    expect(allErr()).toContain('v' + pkg.version + ' has not hashed in 10 minutes since the update from v0.6.9; '
      + 'exiting so HiveOS puts the old version back');
    expect(miner.stop).toHaveBeenCalled();
    expect(m.hive.confirmTrial).not.toHaveBeenCalled();
    miner.emit('stopped', 0);
    await expect(p).resolves.toBe(75);
  });

  // `miner stop` on a version still on trial: the trial is neither passed nor
  // failed, and runs again on the next start.
  test('a stop during a trial ends it undecided, with exit code 0', async () => {
    const m = hiveLoad();
    m.hive.readTrial.mockReturnValue({ from: '0.6.9', to: pkg.version });
    const p = m.run(['--address', ADDR, '--mode', 'mining', '--no-update']);
    await settle();
    const trial = intervalFor(TRIAL_CHECK_MS);
    fire('SIGINT');
    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
    expect(clearInterval).toHaveBeenCalledWith(trial);
    expect(m.hive.confirmTrial).not.toHaveBeenCalled();
  });

  test('with every GPU off a trial passes after two minutes; a confirm that fails is retried next start', async () => {
    clock();
    const m = hiveLoad();
    m.hive.readTrial.mockReturnValue({ from: null, to: pkg.version });
    m.hive.confirmTrial.mockImplementation(() => { throw new Error('EBUSY'); });
    const p = m.run(['--address', ADDR, '--mode', 'mining', '--no-update', '--gpu-index', 'none']);
    await settle();
    const trial = intervalFor(TRIAL_CHECK_MS);
    trial.fn();
    expect(m.hive.confirmTrial).not.toHaveBeenCalled();
    now += 2 * 60 * 1000;
    trial.fn();
    expect(m.hive.confirmTrial).toHaveBeenCalled();
    expect(allOut()).toContain('the update to v' + pkg.version + ' is done');
    expect(m.hive.postMessage).toHaveBeenCalledWith('success', 'LLMJob Earn updated to v' + pkg.version + '.');
    fire('SIGINT');
    trial.fn(); // stopping: nothing
    await expect(p).resolves.toBe(0);
  });
});
