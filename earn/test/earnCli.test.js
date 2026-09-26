'use strict';

// Unit tests for the headless CLI shell (src/cli/earn-cli.js). Everything with
// real IO — fs, os, the engine, the probe helpers and the self-updater — is
// mocked; the pure shared modules (cliArgs, config, format, address,
// miningStats, minerReport, statsFile, …) run for real, exactly like they do in
// production. Each test loads a fresh copy of the module via
// jest.isolateModules so the CLI's module-level state never leaks between tests.

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
  detectGpusVram: jest.fn(),
  detectMinerGpus: jest.fn(),
  postMinerReport: jest.fn(),
  detectGpuTelemetry: jest.fn(),
  // Shared with the GUI now — one detection path for both shells.
  detectGpuInfo: jest.fn(),
  detectGpuTemps: jest.fn(),
}));
jest.mock('../src/main/nodeStore', () => ({ getOrCreateNode: jest.fn() }));
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
      this.isRunning = jest.fn(() => PearlEngine.running);
      PearlEngine.instances.push(this);
    }
  }
  PearlEngine.instances = [];
  PearlEngine.startError = null;
  PearlEngine.startReturns = undefined;
  PearlEngine.running = true;
  return { PearlEngine };
});
jest.mock('../src/main/pearlCore', () => ({
  loadCore: jest.fn(() => null),
  // A loadable core by default: the CLI decides UP FRONT whether it can mine,
  // so a null factory now means "refuse", not "construct an engine
  // that will announce the problem later". The no-core paths have their own
  // tests; everything else models a rig that can actually mine.
  coreFactory: jest.fn(() => () => null),
}));
jest.mock('net', () => ({ connect: jest.fn(() => ({ on: jest.fn(), write: jest.fn(), destroy: jest.fn() })) }));
const pkg = require('../package.json');
const { NETWORK } = require('../src/shared/config');

const ADDR = 'prl1p' + 'a'.repeat(30);
const MDL = 'mdl1p' + 'b'.repeat(30);
const KEYS = require('../src/shared/node').generateKeypair();

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
  m.fs.existsSync.mockReturnValue(true);
  m.probe.detectRegion.mockResolvedValue('us');
  m.probe.detectGpusVram.mockResolvedValue([]);
  // No card list by default: one core that places itself, which is what a rig
  // with no nvidia-smi gets.
  m.probe.detectMinerGpus.mockResolvedValue([]);
  m.probe.postMinerReport.mockResolvedValue(undefined);
  m.probe.detectGpuTelemetry.mockResolvedValue([]);
  m.nodeStore.getOrCreateNode.mockReturnValue({ nodeId: 'a1b2c3d4e5f60789', publicKey: KEYS.publicKey, secretKey: KEYS.secretKey });
  m.probe.detectGpuInfo.mockResolvedValue(null); // no identifiable GPU by default
  m.selfUpdater.fetchLatestRelease.mockResolvedValue(null);
  m.selfUpdater.isPackaged.mockReturnValue(false);
  m.selfUpdater.applyUpdate.mockResolvedValue('/opt/earn');
  m.selfUpdater.reexec.mockReturnValue(0);
  m.selfUpdate.planUpdate.mockReturnValue({ updateAvailable: false, reason: 'up-to-date' });
  m.PearlEngine.startReturns = undefined;
  m.PearlEngine.running = true;
}

// Load a fresh earn-cli plus fresh instances of every mocked dependency.
function load() {
  const m = {};
  jest.isolateModules(() => {
    m.fs = require('fs');
    m.os = require('os');
    m.probe = require('../src/main/probe');
    m.nodeStore = require('../src/main/nodeStore');
    m.selfUpdater = require('../src/cli/selfUpdater');
    m.selfUpdate = require('../src/shared/selfUpdate');
    m.net = require('net');
    m.PearlEngine = require('../src/main/pearlEngine').PearlEngine;
    m.pearlCore = require('../src/main/pearlCore');
    applyDefaults(m);
    m.run = require('../src/cli/earn-cli').run;
  });
  return m;
}

// The CLI is shipped for Linux, but `node src/cli/earn-cli.js` runs anywhere —
// so the macOS gate (a Mac has no NVIDIA GPU to mine on) needs process.platform
// pinned.
//
// Every test starts pinned to linux, and that is not tidiness: process.platform
// now decides whether the CLI mines at all, so a suite that inherited the host's
// platform passed on Linux and Windows and failed 18 tests on the macOS runner
// (which was, correctly, refusing to mine). mainProcess.test.js has always
// pinned for the same reason. Same lesson as the path-separator expectations
// elsewhere in this suite: build the conditions the same way on every OS.
const REAL_PLATFORM = Object.getOwnPropertyDescriptor(process, 'platform');
function setPlatform(p) {
  Object.defineProperty(process, 'platform', { value: p, configurable: true });
}

beforeEach(() => {
  setPlatform('linux'); // a mining-capable platform, whatever the host is
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
  const argvQuick = ['--address', ADDR];
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
    m.probe.detectMinerGpus.mockResolvedValue([
      { index: 0, name: 'NVIDIA GeForce RTX 3070' },
      { index: 1, name: 'NVIDIA GeForce RTX 3070' },
    ]);
    const p = m.run(['--address', ADDR, '--mdl', MDL, '--no-update',
      '--stats-file', '/tmp/s.json']);
    await settle();

    // Auto-detected knobs: region, hostname worker, GPU name.
    expect(allOut()).toContain('pool:       us.pearl.herominers.com:1200');
    expect(allOut()).toContain('(auto)');
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
    miner.emit('event', { type: 'status', hashrate: 3.2, accepted: 5, rejected: 1 });
    // Each card ticks about twice a second, so the rig line is throttled: this
    // second status is folded into the totals but writes no second line.
    miner.emit('event', { type: 'status', gpuIndex: 1, hashrate: 1, accepted: 0, rejected: 0 });
    miner.emit('event', { type: 'connected', gpu: 'RTX 3070' });
    miner.emit('error', new Error('boom'));
    expect(allOut()).toContain('⛏  3.2 TH/s · 5 accepted · 1 rejected');
    expect(allOut().match(/⛏/g)).toHaveLength(1);
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
    expect(miner.stop).toHaveBeenCalledTimes(1);
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
    expect(/\[\d{2}:\d{2}:\d{2}\] /.test(allOut())).toBe(true);
    expect(allOut()).toContain('worker:     rig9');
    expect(allOut()).toContain('gpu:        RTX 4090');

    const miner = m.PearlEngine.instances[0];
    expect(miner.start).toHaveBeenCalledWith(expect.objectContaining({ endpoint: 'de.pearl.herominers.com:1200' }));
    miner.emit('stopped', 5); // engine died on its own → exit code passes through
    await expect(p).resolves.toBe(5);
  });

  // CUDA_VISIBLE_DEVICES=0 hid a rig's second card from the mining core while
  // nvidia-smi still listed it. The CLI removes it at load and says so.
  test('clears CUDA_VISIBLE_DEVICES at load and logs it for a mining run', async () => {
    const had = Object.prototype.hasOwnProperty.call(process.env, 'CUDA_VISIBLE_DEVICES');
    const before = process.env.CUDA_VISIBLE_DEVICES;
    process.env.CUDA_VISIBLE_DEVICES = '0';
    try {
      const m = load();
      expect(process.env.CUDA_VISIBLE_DEVICES).toBeUndefined();
      const p = m.run(['-a', ADDR, '--no-update']);
      await settle();
      expect(allOut()).toContain(
        'ignoring CUDA_VISIBLE_DEVICES=0 so every GPU can mine (set PEARL_GPU_INDEX to mine on one card)');
      m.PearlEngine.instances[0].emit('stopped', 0);
      await expect(p).resolves.toBe(0);
    } finally {
      if (had) process.env.CUDA_VISIBLE_DEVICES = before;
      else delete process.env.CUDA_VISIBLE_DEVICES;
    }
  });

  // net.connect takes the PORT first and PearlMiner hands over host first, so
  // the adapter flips them. Backwards, the CLI fails to connect with an error
  // that names neither side.
  test('the injected connect passes host and port to net in the right order', async () => {
    const m = load();
    const p = m.run(['-a', ADDR, '--no-update']);
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
    const p = m.run(['-a', ADDR, '--no-update', '--stats-file', '/tmp/s.json']);
    await settle();
    m.PearlEngine.instances[0].emit('event', { type: 'status', hashrate: 1, accepted: 1 });
    m.fs.writeFileSync.mockImplementation(() => { throw new Error('read-only /tmp'); });
    intervalFor(10000).fn(); // must not throw

    fire('SIGTERM');
    expect(m.PearlEngine.instances[0].stop).toHaveBeenCalled();
    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
    expect(allErr()).not.toContain('read-only /tmp');
  });

  test('a miner that fails to launch resolves 1', async () => {
    const m = load();
    m.PearlEngine.startError = new Error('EACCES');
    const p = m.run(['-a', ADDR, '--no-update']);
    await expect(p).resolves.toBe(1);
    expect(allErr()).toContain('failed to launch engine: EACCES');
  });

  // The v0.4.1 field failure: a packaged CLI with no pearl_core.node exited 0,
  // which systemd read as success — a silent ten-second restart loop that
  // mined nothing. The contract now: no core is a non-zero exit that says what
  // is missing and how to point at it.
  test('no loadable core exits 1 and names PEARL_CORE_PATH', async () => {
    const m = load();
    m.pearlCore.coreFactory.mockReturnValue(null);
    await expect(m.run(['-a', ADDR, '--no-update'])).resolves.toBe(1);
    expect(allErr()).toContain('pearl_core.node not found');
    expect(allErr()).toContain('PEARL_CORE_PATH');
    expect(m.PearlEngine.instances.length).toBe(0);
  });

  // A false return from start() means the core did not construct: no socket, no
  // job, and no 'stopped' event coming. Left unchecked the run would just end
  // with exit 0 — a restart loop under systemd that mines nothing.
  test('a start that returns false is fatal', async () => {
    const m = load();
    m.PearlEngine.startReturns = false;
    await expect(m.run(['-a', ADDR, '--no-update'])).resolves.toBe(1);
    expect(allErr()).toContain('engine failed to start — see the error above');
  });

  // PearlEngine.stop() on a miner that already stopped emits nothing, so
  // shutdown must not wait on a 'stopped' that will never come.
  test('a signal after the engine already stopped resolves at once', async () => {
    const m = load();
    const p = m.run(['-a', ADDR, '--no-update', '--no-report']);
    await settle();
    m.PearlEngine.running = false;
    fire('SIGINT');
    await expect(p).resolves.toBe(0);
    expect(m.PearlEngine.instances[0].stop).not.toHaveBeenCalled();
  });

  test('the engine is given a temperature reader, as the GUI has always been', async () => {
    // Without it PearlEngine never starts its temp poll, so every headless rig
    // reported temp 0 -- to the stats file, the miner report and the network board.
    const m = load();
    m.probe.detectGpuTemps.mockResolvedValue([{ index: 0, temp: 61 }]);
    const p = m.run(['--address', ADDR, '--no-update', '--no-report']);
    await settle();
    const eng = m.PearlEngine.instances[0];
    expect(typeof eng.opts.readTemps).toBe('function');
    eng.opts.readTemps();
    expect(m.probe.detectGpuTemps).toHaveBeenCalled();
    eng.emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  // The stats file keeps every key HiveOS's h-stats.sh (and anything else)
  // reads by name. The LLM's fields stay in the payload as nulls rather than
  // disappearing, and the mode now always says what the rig does: mine.
  test('the stats file keeps its shape, with the LLM fields null', async () => {
    const m = load();
    const p = m.run(['--address', ADDR, '--no-update', '--no-report', '--stats-file', '/tmp/s.json']);
    await settle();
    const written = JSON.parse(m.fs.writeFileSync.mock.calls[0][1]);
    expect(written).toMatchObject({
      algo: 'pearlhash', schema: 1, mode: 'mining', mining: true,
      strategy: null, gate: null, model: null, tps: { gen: 0, prefill: 0 },
    });
    m.PearlEngine.instances[0].emit('stopped', 0);
    await p;
  });

  test('board reports carry per-card telemetry and a signed rig identity', async () => {
    const m = load();
    m.probe.detectGpuTelemetry.mockResolvedValue([{ index: 0, tempC: 58, powerW: 290, driver: '580.82' }]);
    const p = m.run(['-a', ADDR, '--no-update']);
    await settle();
    const row = m.probe.postMinerReport.mock.calls[0][0];
    expect(row).toMatchObject({ client: 'cli', os: 'linux', tempC: 58, powerW: 290, driver: '580.82', rigId: 'a1b2c3d4e5f60789' });
    expect(typeof row.signature).toBe('string');
    expect(row).not.toHaveProperty('secretKey');
    m.PearlEngine.instances[0].emit('stopped', 0);
    await p;
  });

  // Some rigs run with a read-only home. That must cost the rig its signature,
  // not its place on the board.
  test('an identity store that cannot be written means unsigned reports, not none', async () => {
    const m = load();
    m.nodeStore.getOrCreateNode.mockImplementation(() => { throw new Error('EROFS'); });
    const p = m.run(['-a', ADDR, '--no-update']);
    await settle();
    const row = m.probe.postMinerReport.mock.calls[0][0];
    expect(row.address).toBe(ADDR);
    expect(row).not.toHaveProperty('rigId');
    m.PearlEngine.instances[0].emit('stopped', 0);
    await p;
  });

  test('--no-report posts nothing to the board and arms no reporter', async () => {
    const m = load();
    const p = m.run(['-a', ADDR, '--no-update', '--no-report']);
    await settle();
    expect(m.probe.postMinerReport).not.toHaveBeenCalled();
    expect(intervalFor(NETWORK.reportIntervalMs)).toBeUndefined();
    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });
});

// ── the retired LLM surface ──────────────────────────────────────────────────

describe('retired LLM options and commands', () => {
  // The CLI auto-updates on start, so a unit written for an older build lands
  // on this one with its old flags. It must keep mining and say what it ignored.
  test('the LLM flags of an old unit are ignored, announced once, and it still mines', async () => {
    const m = load();
    const p = m.run(['-a', ADDR, '--no-update', '--mode', 'auto', '--no-serve', '--gate-port', '8000']);
    await settle();
    expect(allErr()).toContain('ignoring retired options: --mode, --no-serve, --gate-port'
      + ' (the local LLM was removed; this build only mines)');
    expect(m.PearlEngine.instances).toHaveLength(1);
    m.PearlEngine.instances[0].emit('stopped', 0);
    await expect(p).resolves.toBe(0);
  });

  test('a single retired flag reads in the singular', async () => {
    const m = load();
    const p = m.run(['-a', ADDR, '--no-update', '--mode', 'mining']);
    await settle();
    expect(allErr()).toContain('ignoring retired option: --mode (');
    m.PearlEngine.instances[0].emit('stopped', 0);
    await p;
  });

  test('an old LLM-only unit is told it now needs an address', async () => {
    const m = load();
    await expect(m.run(['--mode', 'llm', '--no-update'])).resolves.toBe(1);
    expect(allErr()).toContain('--mode llm was retired and this build only mines');
  });

  test('`connect` explains that it was retired instead of "unknown option"', async () => {
    const m = load();
    await expect(m.run(['connect', '--token', 't'])).resolves.toBe(1);
    expect(allErr()).toContain('"connect" was retired with the local LLM');
    expect(allErr()).toContain('llmjob-earn-cli --address');
  });

  // A Mac has no NVIDIA GPU, and with the LLM gone there is nothing else to run.
  test('macOS has nothing to run and exits 1 without building an engine', async () => {
    setPlatform('darwin');
    const m = load();
    await expect(m.run(['-a', ADDR, '--no-update'])).resolves.toBe(1);
    expect(allErr()).toContain('mining is not available on macOS');
    expect(m.PearlEngine.instances).toHaveLength(0);
  });
});
