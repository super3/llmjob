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
  EOL: '\n',
}));
jest.mock('../src/main/probe', () => ({
  detectRegion: jest.fn(),
  detectVram: jest.fn(),
  detectGpusVram: jest.fn(),
  detectDriverMajor: jest.fn(),
  postMinerReport: jest.fn(),
  findFreePort: jest.fn(),
  // Shared with the GUI now — one detection path for both shells.
  detectGpuInfo: jest.fn(),
}));
jest.mock('../src/main/io', () => ({
  postJson: jest.fn(),
  getJson: jest.fn(),
  downloadFile: jest.fn(),
  streamChatCompletion: jest.fn(),
  extractLlamaZip: jest.fn(),
  extractEnginePackage: jest.fn(),
}));
jest.mock('../src/main/nodeStore', () => ({
  loadNode: jest.fn(),
  saveNode: jest.fn(),
  getOrCreateNode: jest.fn(),
}));
jest.mock('../src/cli/selfUpdater', () => ({
  UPDATED_ENV: 'LLMJOB_EARN_UPDATED',
  fetchLatestRelease: jest.fn(),
  isPackaged: jest.fn(),
  applyUpdate: jest.fn(),
  reexec: jest.fn(),
}));
jest.mock('../src/shared/selfUpdate', () => ({ planUpdate: jest.fn() }));
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
      PearlEngine.instances.push(this);
    }
  }
  PearlEngine.instances = [];
  PearlEngine.startError = null;
  PearlEngine.startReturns = undefined;
  return { PearlEngine };
});
jest.mock('../src/main/pearlCore', () => ({
  loadCore: jest.fn(() => null),
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
        if (onPct) { onPct(20); onPct(null); }
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
let intervals; // { fn, ms, unref? } handles captured from setInterval
let intervalUnref; // whether captured handles carry an unref()
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
  m.probe.detectDriverMajor.mockResolvedValue(600);
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
  m.selfUpdate.planUpdate.mockReturnValue({ updateAvailable: false, reason: 'up-to-date' });
  m.PearlEngine.startReturns = undefined;
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
    m.net = require('net');
    m.PearlEngine = require('../src/main/pearlEngine').PearlEngine;
    m.autoGate = require('../src/main/autoGate');
    m.autoGate.createAutoGate.instances = [];
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
  intervalUnref = true;
  sigHandlers = {};
  origOutTty = process.stdout.isTTY;
  origErrTty = process.stderr.isTTY;
  process.stdout.isTTY = false;
  process.stderr.isTTY = false;
  jest.spyOn(process.stdout, 'write').mockImplementation((s) => { out.push(String(s)); return true; });
  jest.spyOn(process.stderr, 'write').mockImplementation((s) => { err.push(String(s)); return true; });
  jest.spyOn(global, 'setInterval').mockImplementation((fn, ms) => {
    const h = intervalUnref ? { fn, ms, unref: jest.fn() } : { fn, ms };
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
      expect(m.selfUpdate.planUpdate).toHaveBeenCalledWith({
        currentVersion: pkg.version, release: { version: '9.9.9' }, platform: process.platform,
      });
    });
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

describe('mining', () => {
  test('full auto-detected run: report, stats file, SIGINT shutdown', async () => {
    intervalUnref = false; // cover the interval handles without unref()
    const m = load();
    m.probe.detectGpuInfo.mockResolvedValue({ name: 'NVIDIA GeForce RTX 3070', count: 2 });
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

    const miner = m.PearlEngine.instances[0];
    expect(miner.start).toHaveBeenCalledWith(expect.objectContaining({
      endpoint: 'us.pearl.herominers.com:1200',
    }));
    miner.emit('log', { line: 'hello', level: 'info' });
    miner.emit('log', { line: 'bad', level: 'error' });
    miner.emit('event', { type: 'status', hashrate: 3.2, accepted: 5, rejected: 1 });
    miner.emit('event', { type: 'connected', gpu: 'RTX 3070' });
    miner.emit('error', new Error('boom'));
    expect(allOut()).toContain('⛏  3.2 TH/s · 5 accepted · 1 rejected');
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

  // net.connect takes the PORT first and PearlMiner hands over host first, so
  // the adapter flips them. Backwards, the CLI fails to connect with an error
  // that names neither side.
  test('the injected connect passes host and port to net in the right order', async () => {
    const m = load();
    const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update']);
    await settle();
    m.net.connect.mockClear();
    m.PearlEngine.instances[0].opts.connect('pool.example', 1200);
    expect(m.net.connect).toHaveBeenCalledWith(1200, 'pool.example');
    m.PearlEngine.instances[0].emit('stopped', 0);
    await p;
  });

  // The default interval handle DOES carry unref(), so the writer is unrefed
  // and never holds the process open on its own. The full run above covers the
  // other side, where the handle has no unref to call.
  //
  // A stats file that cannot be written must also stay silent: it is a
  // convenience for rig dashboards, not something worth killing a miner over.
  test('the stats writer is unrefed, and a write failure is silent', async () => {
    const m = load();
    const p = m.run(['-a', ADDR, '--mode', 'mining', '--no-update', '--stats-file', '/tmp/s.json']);
    await settle();
    m.PearlEngine.instances[0].emit('event', { type: 'status', hashrate: 1, accepted: 1 });
    m.fs.writeFileSync.mockImplementation(() => { throw new Error('read-only /tmp'); });
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

// ── local LLM ────────────────────────────────────────────────────────────────

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
    intervalUnref = false; // cover the servePinger without unref()
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

    // The binary resolver's engine wires the shared extractor with the CLI hint.
    const le = m.LlmEngineManager.instances[0];
    await le.opts.extract('/z.zip', '/dest');
    expect(m.io.extractLlamaZip).toHaveBeenCalledWith('/z.zip', '/dest', expect.stringContaining('unzip'));

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
    await jw.opts.post('http://u', { a: 1 });
    expect(m.io.postJson).toHaveBeenCalledWith('http://u', { a: 1 }, 30000);
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

  test('a miner that fails to launch also stops the LLM', async () => {
    const m = load();
    m.PearlEngine.startError = new Error('spawn ENOENT');
    const p = m.run(argvBoth);
    await expect(p).resolves.toBe(1);
    expect(allErr()).toContain('failed to launch engine: spawn ENOENT');
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
    // The GPU probe explodes synchronously — cachedDeviceName absorbs it.
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
      isSwitching() { return this.switching; },
      start() { this.started = true; return this; },
      stop: jest.fn(),
    };
    createAutoGate.instances.push(inst);
    return inst;
  });
  createAutoGate.instances = [];
  return { createAutoGate };
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

test('a fatal engine start also stops an LLM that had already come up', async () => {
  // auto mode starts the LLM before the miner, so a fatal engine failure must
  // tear the LLM down rather than leave a server orphaned behind a dead node.
  const m = load();
  m.probe.detectGpusVram.mockResolvedValue([{ index: 0, name: 'RTX 4090', usedMb: 3000, totalMb: 24000 }]);
  m.LlmEngineManager.serverInstalled = true;
  m.LlmEngineManager.modelInstalled = true;
  m.PearlEngine.startReturns = false;
  const code = await m.run(['--address', ADDR, '--no-update', '--no-serve']);
  expect(code).toBe(1);
  expect(m.LlmManager.instances[0].stop).toHaveBeenCalled();
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
