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
}));
jest.mock('../src/main/io', () => ({
  postJson: jest.fn(),
  getJson: jest.fn(),
  downloadFile: jest.fn(),
  streamChatCompletion: jest.fn(),
  extractLlamaZip: jest.fn(),
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
jest.mock('../src/main/minerManager', () => {
  const { EventEmitter } = require('events');
  class MinerManager extends EventEmitter {
    constructor(opts) {
      super();
      this.opts = opts;
      this.start = jest.fn(() => { if (MinerManager.startError) throw MinerManager.startError; });
      this.stop = jest.fn();
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
      this.isInstalled = jest.fn(() => EngineManager.installed);
      this.binaryPath = jest.fn(() => '/cache/alpha-miner');
      this.ensure = jest.fn(async (onPct) => {
        if (onPct) { onPct(50); onPct(null); }
        return '/cache/alpha-miner';
      });
      EngineManager.instances.push(this);
    }
  }
  EngineManager.instances = [];
  EngineManager.installed = false;
  return { EngineManager };
});
jest.mock('../src/main/llmManager', () => {
  const { EventEmitter } = require('events');
  class LlmManager extends EventEmitter {
    constructor(opts) {
      super();
      this.opts = opts;
      this.baseUrl = 'http://127.0.0.1:8080';
      this.start = jest.fn();
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
      LlmEngineManager.instances.push(this);
    }
  }
  LlmEngineManager.instances = [];
  LlmEngineManager.serverInstalled = false;
  LlmEngineManager.modelInstalled = false;
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
const { NETWORK, NODE } = require('../src/shared/config');
const nodeProto = require('../src/shared/node');

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
  m.probe.detectRegion.mockResolvedValue('us1');
  m.probe.detectVram.mockResolvedValue(null);
  m.probe.detectGpusVram.mockResolvedValue([]);
  m.probe.detectDriverMajor.mockResolvedValue(600);
  m.probe.postMinerReport.mockResolvedValue(undefined);
  m.probe.findFreePort.mockResolvedValue(8080);
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
    m.MinerManager = require('../src/main/minerManager').MinerManager;
    m.EngineManager = require('../src/main/engineManager').EngineManager;
    m.LlmManager = require('../src/main/llmManager').LlmManager;
    m.LlmEngineManager = require('../src/main/llmEngineManager').LlmEngineManager;
    m.JobWorker = require('../src/main/jobWorker').JobWorker;
    applyDefaults(m);
    m.run = require('../src/cli/earn-cli').run;
  });
  return m;
}

beforeEach(() => {
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
// Each test uses a run that fails fast after the update phase (--binary that
// does not exist → engine setup error → exit 1) so it never reaches mining.

describe('auto-update on start', () => {
  const argvQuick = ['--address', ADDR, '--binary', '/nope'];

  test('skips the check entirely in the re-exec child', async () => {
    process.env.LLMJOB_EARN_UPDATED = '1';
    const m = load();
    m.fs.existsSync.mockReturnValue(false);
    await expect(m.run(argvQuick)).resolves.toBe(1);
    expect(m.selfUpdater.fetchLatestRelease).not.toHaveBeenCalled();
    expect(allErr()).toContain('engine setup failed: engine binary not found: /nope');
  });

  test('continues when offline (no release)', async () => {
    const m = load();
    m.fs.existsSync.mockReturnValue(false);
    await expect(m.run(argvQuick)).resolves.toBe(1);
    expect(m.selfUpdater.fetchLatestRelease).toHaveBeenCalled();
    expect(m.selfUpdate.planUpdate).not.toHaveBeenCalled();
  });

  test('continues when already up to date', async () => {
    const m = load();
    m.fs.existsSync.mockReturnValue(false);
    m.selfUpdater.fetchLatestRelease.mockResolvedValue({ version: pkg.version });
    await expect(m.run(argvQuick)).resolves.toBe(1);
    expect(allOut()).not.toContain('newer release');
  });

  test('only mentions a newer release when running from source', async () => {
    const m = load();
    m.fs.existsSync.mockReturnValue(false);
    m.selfUpdater.fetchLatestRelease.mockResolvedValue({ version: '9.9.9' });
    m.selfUpdate.planUpdate.mockReturnValue({ updateAvailable: true, latestVersion: '9.9.9', currentVersion: pkg.version });
    await expect(m.run(argvQuick)).resolves.toBe(1);
    expect(allOut()).toContain('a newer release is available: v9.9.9');
    expect(m.selfUpdater.applyUpdate).not.toHaveBeenCalled();
  });

  test('updates and re-execs when packaged, returning the child exit code', async () => {
    const m = load();
    m.selfUpdater.fetchLatestRelease.mockResolvedValue({ version: '9.9.9' });
    m.selfUpdater.isPackaged.mockReturnValue(true);
    m.selfUpdater.reexec.mockReturnValue(42);
    m.selfUpdate.planUpdate.mockReturnValue({ updateAvailable: true, latestVersion: '9.9.9', currentVersion: pkg.version });
    await expect(m.run(argvQuick)).resolves.toBe(42);
    expect(m.selfUpdater.reexec).toHaveBeenCalledWith(argvQuick);
  });

  test('keeps mining on the old version when the auto-update fails', async () => {
    const m = load();
    m.fs.existsSync.mockReturnValue(false);
    m.selfUpdater.fetchLatestRelease.mockResolvedValue({ version: '9.9.9' });
    m.selfUpdater.isPackaged.mockReturnValue(true);
    m.selfUpdater.applyUpdate.mockRejectedValue(new Error('nope'));
    m.selfUpdate.planUpdate.mockReturnValue({ updateAvailable: true, latestVersion: '9.9.9', currentVersion: pkg.version });
    await expect(m.run(argvQuick)).resolves.toBe(1);
    expect(allErr()).toContain('auto-update failed (nope)');
  });
});

// ── mining runs ──────────────────────────────────────────────────────────────

describe('mining', () => {
  test('full auto-detected run: download engine, report, stats file, SIGINT shutdown', async () => {
    intervalUnref = false; // cover the interval handles without unref()
    const m = load();
    m.cp.execFile.mockImplementation((cmd, args, opts, cb) =>
      cb(null, 'NVIDIA GeForce RTX 3070\nNVIDIA GeForce RTX 3070\n'));
    const p = m.run(['--address', ADDR, '--mdl', MDL, '--no-update',
      '--stats-file', '/tmp/s.json', '--engine-dir', '/ed']);
    await settle();

    // Auto-detected knobs: region, hostname worker, GPU → scaled difficulty.
    expect(allOut()).toContain('mode:       mining  (default)');
    expect(allOut()).toContain('worker:     rig-host  (auto)');
    expect(allOut()).toContain('(+MDL');
    expect(allOut()).toContain('difficulty: 262144  (for 2× NVIDIA GeForce RTX 3070, auto)');
    expect(allOut()).toContain('engine:     alpha-miner 1.8.8');
    expect(allOut()).toContain('downloading mining engine from');
    expect(allOut()).toContain('downloading… 50%');

    // The Linux CLI never extracts zips — its EngineManager extractor rejects.
    const eng = m.EngineManager.instances[0];
    expect(eng.opts.dir).toBe('/ed');
    await expect(eng.opts.extract()).rejects.toThrow('zip extraction is not supported on the Linux CLI');

    const miner = m.MinerManager.instances[0];
    expect(miner.start).toHaveBeenCalledWith(expect.objectContaining({ binaryPath: '/cache/alpha-miner' }));
    miner.emit('started', { bin: '/cache/alpha-miner', args: ['-x'] });
    miner.emit('log', { line: 'hello', level: 'info' });
    miner.emit('log', { line: 'bad', level: 'error' });
    miner.emit('event', { type: 'status', hashrate: 3.2, accepted: 5, rejected: 1 });
    miner.emit('event', { type: 'connected', gpu: 'RTX 3070' });
    miner.emit('error', new Error('boom'));
    expect(allOut()).toContain('starting: /cache/alpha-miner -x');
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

    fire('SIGINT');
    fire('SIGINT'); // second signal hits the `stopping` guard
    expect(miner.stop).toHaveBeenCalled();
    miner.emit('stopped', 0);
    miner.emit('stopped', 0); // second emit hits the `settled` guard
    await expect(p).resolves.toBe(0);
    expect(allOut()).toContain('shutting down…');
    expect(allOut()).toContain('engine exited (code 0)');
  });

  test('explicit knobs, provided binary, TTY prefix, engine exit code passthrough', async () => {
    process.stdout.isTTY = true;
    const m = load();
    const p = m.run(['-a', ADDR, '-r', 'eu1', '-w', 'rig9', '-g', 'RTX 4090', '-d', '1024',
      '--binary', '/bin/eng', '--no-update']);
    await settle();

    expect(m.probe.detectRegion).not.toHaveBeenCalled();
    expect(m.cp.execFile).not.toHaveBeenCalled();
    expect(/\[\d{2}:\d{2}:\d{2}\] /.test(allOut())).toBe(true);
    expect(allOut()).toContain('worker:     rig9');
    expect(allOut()).toContain('difficulty: 1024  (for RTX 4090)');

    const miner = m.MinerManager.instances[0];
    expect(miner.start).toHaveBeenCalledWith(expect.objectContaining({ binaryPath: '/bin/eng' }));
    miner.emit('stopped', 5); // engine died on its own → exit code passes through
    await expect(p).resolves.toBe(5);
  });

  test('engine setup failure prints the manual download URL and exits 1', async () => {
    const m = load();
    m.fs.existsSync.mockReturnValue(false);
    await expect(m.run(['-a', ADDR, '--binary', '/nope', '--no-update'])).resolves.toBe(1);
    expect(allErr()).toContain('engine setup failed: engine binary not found: /nope');
    expect(allErr()).toContain('manual download:');
  });

  test('unknown driver picks the compatible build; no-report; hostname fallback', async () => {
    const m = load();
    m.os.hostname.mockReturnValue('');
    m.probe.detectDriverMajor.mockResolvedValue(null);
    m.EngineManager.installed = true;
    const p = m.run(['-a', ADDR, '--no-update', '--no-report']);
    await settle();

    expect(allOut()).toContain('driver version unknown — using the compatible build');
    expect(allOut()).toContain('engine found: /cache/alpha-miner');
    expect(allOut()).toContain('worker:     rig01  (auto)'); // unusable hostname → default
    expect(intervalFor(NETWORK.reportIntervalMs)).toBeUndefined();

    const miner = m.MinerManager.instances[0];
    miner.emit('stopped', null); // null exit code maps to 0
    await expect(p).resolves.toBe(0);
  });

  test('old driver picks the fallback build; single GPU; stats write failures stay silent', async () => {
    const m = load();
    m.probe.detectDriverMajor.mockResolvedValue(550);
    m.cp.execFile.mockImplementation((cmd, args, opts, cb) => cb(null, 'NVIDIA GeForce RTX 3070\n'));
    m.fs.writeFileSync.mockImplementation(() => { throw new Error('read-only fs'); });
    const p = m.run(['-a', ADDR, '-d', '4096', '--no-update', '--stats-file', '/tmp/s2.json']);
    await settle();

    expect(allOut()).toContain('driver 550 < 580');
    expect(allOut()).toContain('difficulty: 4096  (for NVIDIA GeForce RTX 3070, auto)');
    intervalFor(10000).fn(); // must not throw
    expect(m.fs.renameSync).not.toHaveBeenCalled();

    fire('SIGTERM');
    m.MinerManager.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('a miner that fails to launch resolves 1', async () => {
    const m = load();
    m.MinerManager.startError = new Error('EACCES');
    const p = m.run(['-a', ADDR, '--binary', '/bin/eng', '--no-update']);
    await expect(p).resolves.toBe(1);
    expect(allErr()).toContain('failed to launch engine: EACCES');
  });
});

// ── local LLM ────────────────────────────────────────────────────────────────

describe('local LLM', () => {
  test('refuses to start the LLM without enough free VRAM (nothing to run → 1)', async () => {
    const m = load();
    m.probe.detectVram.mockResolvedValue({ totalMb: 8000, usedMb: 7000 });
    await expect(m.run(['--mode', 'llm', '--no-update'])).resolves.toBe(1);
    expect(allErr()).toContain('not enough free VRAM for the local LLM: 1000 MB free');
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

  test('serves cluster jobs when connected: worker, pings, telemetry, SIGINT shutdown', async () => {
    intervalUnref = false; // cover the servePinger without unref()
    const m = load();
    m.nodeStore.loadNode.mockReturnValue(makeNode({ connected: true, name: 'rig' }));
    m.probe.detectVram.mockResolvedValue({ totalMb: 24000, usedMb: 2000 });
    m.probe.findFreePort.mockResolvedValue(9090);
    m.cp.execFile.mockImplementation((cmd, args, opts, cb) => cb(null, 'NVIDIA GeForce RTX 4090\n'));
    const p = m.run(['--mode', 'llm', '--no-update']);
    await settle();

    expect(allOut()).toContain('mode:       llm');
    expect(allOut()).toContain('port 8080 is busy — using 9090 for the local LLM instead');
    expect(allOut()).toContain('downloading LLM model');

    // The binary resolver's engine wires the shared extractor with the CLI hint.
    const le = m.LlmEngineManager.instances[0];
    await le.opts.extract('/z.zip', '/dest');
    expect(m.io.extractLlamaZip).toHaveBeenCalledWith('/z.zip', '/dest', expect.stringContaining('unzip'));

    const llm = m.LlmManager.instances[0];
    expect(llm.start).toHaveBeenCalledWith(expect.objectContaining({
      binaryPath: '/cache/llama-server', modelPath: '/cache/model.gguf', nGpuLayers: 42, port: 9090,
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
    await jw.opts.runJob({ messages: [] }, { onDelta });
    expect(m.io.streamChatCompletion).toHaveBeenCalledWith(llm.baseUrl, { messages: [] }, onDelta);

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
    llm.emit('stopped', 0); // during shutdown: the `stopping` guard swallows it
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

  test('an unconnected node runs the LLM without serving; clean exit still fails (0 → 1)', async () => {
    const m = load();
    m.nodeStore.loadNode.mockReturnValue(makeNode({ connected: false }));
    m.LlmEngineManager.serverInstalled = true;
    m.LlmEngineManager.modelInstalled = true;
    const p = m.run(['--mode', 'llm', '--no-update']);
    await settle();
    expect(allOut()).toContain('LLM server found: /cache/llama-server');
    expect(allOut()).toContain('LLM model found: /cache/model.gguf');
    const llm = m.LlmManager.instances[0];
    expect(llm.start).toHaveBeenCalledWith(expect.objectContaining({ nGpuLayers: 42 })); // no VRAM → full offload
    llm.emit('ready', { baseUrl: 'http://127.0.0.1:8080' });
    expect(m.JobWorker.instances.length).toBe(0);
    llm.emit('stopped', 0);
    await expect(p).resolves.toBe(1);
  });
});

// ── both mode (miner + LLM together) ─────────────────────────────────────────

describe('both mode', () => {
  const argvBoth = ['-a', ADDR, '--mode', 'both', '--no-update', '--no-report',
    '--llm-binary', '/lb', '--llm-model', '/lm'];

  test('co-runs; an LLM death does not stop mining; the miner exit code wins', async () => {
    const m = load();
    m.EngineManager.installed = true;
    m.probe.detectVram.mockResolvedValue({ totalMb: 8000, usedMb: 1000 });
    const p = m.run(argvBoth);
    await settle();

    expect(allOut()).toContain('mode:       both');
    const llm = m.LlmManager.instances[0];
    // free 7000 − 2048 mining reserve = 4952 of 5800 → partial offload
    expect(llm.start).toHaveBeenCalledWith(expect.objectContaining({ nGpuLayers: 35, port: 8080 }));
    llm.emit('ready', { baseUrl: 'http://127.0.0.1:8080' }); // not connected → no worker
    llm.emit('stopped', 2); // LLM dies; the miner keeps running
    expect(allErr()).toContain('local LLM exited (code 2)');

    const miner = m.MinerManager.instances[0];
    miner.emit('stopped', 7);
    expect(llm.stop).toHaveBeenCalled();
    await expect(p).resolves.toBe(7);
  });

  test('a miner that fails to launch also stops the LLM', async () => {
    const m = load();
    m.EngineManager.installed = true;
    m.MinerManager.startError = new Error('spawn ENOENT');
    const p = m.run(argvBoth);
    await expect(p).resolves.toBe(1);
    expect(allErr()).toContain('failed to launch engine: spawn ENOENT');
    expect(m.LlmManager.instances[0].stop).toHaveBeenCalled();
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
    m.cp.execFile.mockImplementation((cmd, args, opts, cb) => cb(null, 'NVIDIA GeForce RTX 4090\n'));

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
