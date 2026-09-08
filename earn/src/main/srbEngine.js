'use strict';

const { EventEmitter } = require('events');
const { combinePayoutAddress } = require('../shared/address');
const { DEV_FEE_PCT } = require('../shared/minerSelect');

// SRBMiner-Multi, wearing the same clothes PearlEngine wears.
//
// PearlEngine IS the miner (our core, driven in-process). This one WATCHES a
// miner: SRBMiner is a separate closed-source process, so everything the UI
// reads has to be recovered from outside it. The surface is identical on
// purpose -- start/stop/isRunning/gpuIndex plus log/event/error/stopped -- so
// choosing an engine stays a choice of constructor, which is what keeps main.js
// and earn-cli.js from growing a second code path.
//
// Telemetry comes from SRBMiner's own statistics API (--api-enable, JSON on
// 127.0.0.1:21550), NOT from scraping its console. The console is a redrawn,
// column-aligned table with ANSI colour, written for humans and free to change
// between releases; the JSON is stable and gives hashrate, shares, pool state
// and per-GPU temperature directly. alpha-miner was scraped and that is exactly
// how it broke.
//
// The 2% dev fee is disclosed on every start. Our own core takes nothing, so a
// user swapping engines is agreeing to something they were not agreeing to
// before, and it should not take reading the vendor's site to find that out.
//
// Everything IO is injected (spawn, fetchStats, kill) so the whole lifecycle is
// testable with no binary, no GPU and no sockets.

const POLL_MS = 5000;
const KILL_GRACE_MS = 8000;
const API_PORT = 21550;
// The escape character is the point: this strips the colour SRBMiner writes to
// its console, so the log lines we forward are readable in journalctl.
// eslint-disable-next-line no-control-regex
const ANSI = /\x1B\[[0-9;]*[a-zA-Z]/g;

// A stats fetcher over an injected http module. Kept here rather than in the
// callers so both entry points share one implementation, and injectable so the
// tests never open a socket.
function httpStatsFetcher(http) {
  return (port) => new Promise((resolve, reject) => {
    const req = http.get({ host: '127.0.0.1', port, path: '/', timeout: 4000 }, (res) => {
      let body = '';
      res.on('data', (c) => { body += c; });
      res.on('end', () => {
        try {
          resolve(JSON.parse(body));
        } catch (e) {
          reject(e);
        }
      });
    });
    req.on('timeout', () => req.destroy(new Error('stats timeout')));
    req.on('error', reject);
  });
}

class SrbEngine extends EventEmitter {
  // No readTemps here, unlike PearlEngine: the stats API already carries the
  // card temperature, so there is nothing to poll nvidia-smi for.
  constructor({
    binPath, spawn, fetchStats, kill,
    apiPort, pollMs, killGraceMs,
  } = {}) {
    super();
    this.binPath = binPath || null;
    this.spawn = spawn;
    this.fetchStats = fetchStats;
    this.kill = kill || ((pid, sig) => process.kill(pid, sig));
    this.apiPort = apiPort || API_PORT;
    this.pollMs = pollMs || POLL_MS;
    this.killGraceMs = killGraceMs || KILL_GRACE_MS;

    this.child = null;
    this.pollTimer = null;
    this.killTimer = null;
    this.running = false;

    this.accepted = 0;
    this.rejected = 0;
    this.hashrate = 0;
    this.temp = null;
    this.gpu = null;
    this.endpoint = null;
    this._announced = false;
    this._out = '';
    this._err = '';
  }

  isRunning() {
    return this.running;
  }

  // One card, index 0 -- same reading as PearlEngine, kept as a method so it can
  // follow if that stops being true.
  gpuIndex() {
    return 0;
  }

  start(settings = {}) {
    if (this.running) return false;

    this.accepted = 0;
    this.rejected = 0;
    this.hashrate = 0;
    this.temp = null;
    this._announced = false;
    this._out = '';
    this._err = '';
    this.gpu = settings.gpu || null;
    this.endpoint = settings.endpoint || null;

    if (!this.binPath) {
      this.emit('log', {
        level: 'error',
        line: 'SRBMiner-Multi binary not found -- nothing to mine with.',
      });
      this.emit('stopped', 0);
      return false;
    }

    const wallet = combinePayoutAddress(settings.address, settings.mdlAddress);
    const worker = settings.worker || 'rig01';
    const args = [
      '--disable-cpu',
      '--algorithm', 'pearlhash',
      '--pool', this.endpoint,
      '--wallet', wallet,
      '--worker', worker,
      // Without this the stats API never binds, and the only telemetry left is
      // the console table -- the thing we are deliberately not parsing.
      '--api-enable',
      '--api-port', String(this.apiPort),
    ];

    // Said every start, not once at install: the fee is a standing cost and our
    // own engine's is zero, so the difference should be visible in the log the
    // operator actually reads.
    this.emit('log', {
      level: 'info',
      line: 'mining with SRBMiner-Multi (' + this.binPath + ') — '
        + 'closed source, ' + DEV_FEE_PCT + '% dev fee. '
        + 'Use --miner native for the built-in zero-fee core.',
    });

    let child;
    try {
      // detached so the miner leads its own process group. SRBMiner outlives a
      // plain kill of the pid we hold -- that is how a "stopped" miner kept a
      // card busy through a whole benchmark window -- so stop() signals the
      // GROUP, which requires being a group leader.
      child = this.spawn(this.binPath, args, {
        detached: true,
        stdio: ['ignore', 'pipe', 'pipe'],
      });
    } catch (e) {
      this.emit('error', e);
      this.emit('stopped', 0);
      return false;
    }

    this.child = child;
    this.running = true;

    if (child.stdout) child.stdout.on('data', (d) => this._lines('_out', d, 'info'));
    if (child.stderr) child.stderr.on('data', (d) => this._lines('_err', d, 'error'));

    child.on('error', (e) => this.emit('error', e));
    child.on('exit', (code) => {
      this._stopTimers();
      this.running = false;
      this.child = null;
      this.emit('stopped', code == null ? 0 : code);
    });

    this._startPolling();
    return true;
  }

  stop() {
    this._stopPolling();
    const child = this.child;
    if (!child) return;
    // Negative pid: signal the whole group, not just the leader.
    this._signal(-child.pid, 'SIGTERM');
    this.killTimer = setTimeout(() => {
      this.killTimer = null;
      // Still there after the grace period. SIGKILL the group; a miner that
      // ignores SIGTERM must not be able to hold the card.
      if (this.child) this._signal(-this.child.pid, 'SIGKILL');
    }, this.killGraceMs);
    if (this.killTimer.unref) this.killTimer.unref();
  }

  // A signal to a process that has already gone is not an error worth
  // surfacing -- it is the normal race between our timer and its exit.
  _signal(pid, sig) {
    try {
      this.kill(pid, sig);
    } catch (e) { /* already gone */ }
  }

  _stopTimers() {
    this._stopPolling();
    if (this.killTimer) clearTimeout(this.killTimer);
    this.killTimer = null;
  }

  _stopPolling() {
    if (this.pollTimer) clearInterval(this.pollTimer);
    this.pollTimer = null;
  }

  _startPolling() {
    const tick = () => {
      let p;
      try {
        p = Promise.resolve(this.fetchStats(this.apiPort));
      } catch (e) {
        return;
      }
      p.then((s) => this._applyStats(s)).catch(() => { /* miner still starting */ });
    };
    this.pollTimer = setInterval(tick, this.pollMs);
    if (this.pollTimer.unref) this.pollTimer.unref();
    tick();
  }

  // Translate one stats document into the two events the UI reads.
  //
  // SRBMiner can run several algorithms at once, so the interesting numbers sit
  // under algorithms[]. We only ever launch it with one (--algorithm pearlhash),
  // so entry 0 is ours; reading it by index rather than searching for the name
  // keeps this working if the algorithm is ever spelled differently upstream.
  _applyStats(s) {
    if (!s || typeof s !== 'object') return;
    const algo = Array.isArray(s.algorithms) && s.algorithms.length ? s.algorithms[0] : null;
    const gpu0 = Array.isArray(s.gpu_devices) && s.gpu_devices.length ? s.gpu_devices[0] : null;

    if (algo) {
      // The API reports H/s; every counter downstream of here is TH/s. The GPU
      // total rather than the 1min average: the average reads 0 for the first
      // minute, which would render as a rig that is connected but not hashing.
      const hs = algo.hashrate && algo.hashrate.gpu ? Number(algo.hashrate.gpu.total) : NaN;
      if (Number.isFinite(hs)) this.hashrate = hs / 1e12;
      if (algo.shares) {
        const acc = Number(algo.shares.accepted);
        if (Number.isFinite(acc)) this.accepted = acc;
        const rej = Number(algo.shares.rejected);
        if (Number.isFinite(rej)) this.rejected = rej;
      }
    }

    if (gpu0) {
      const t = Number(gpu0.temperature);
      this.temp = Number.isFinite(t) && t > 0 ? t : null;
      if (!this.gpu && gpu0.model) this.gpu = gpu0.model;
    }

    // The pool has us: this is the moment the card is actually mining, and the
    // same point PearlEngine announces on its first job. SRBMiner has no
    // explicit connected flag; time_connected appears once the session is up.
    if (!this._announced && algo && algo.pool && algo.pool.time_connected) {
      this._announced = true;
      this.emit('event', {
        type: 'connected',
        gpuIndex: this.gpuIndex(),
        endpoint: this.endpoint,
        gpu: this.gpu,
      });
    }
    this._status();
  }

  _lines(buf, chunk, level) {
    this[buf] += String(chunk);
    const parts = this[buf].split('\n');
    this[buf] = parts.pop();
    for (const raw of parts) {
      const line = raw.replace(ANSI, '').trimEnd();
      if (line) this.emit('log', { level, line });
    }
  }

  _status() {
    this.emit('event', {
      type: 'status',
      gpuIndex: this.gpuIndex(),
      hashrate: this.hashrate,
      accepted: this.accepted,
      rejected: this.rejected,
      // Deliberately null, matching PearlEngine: the per-card figure would go to
      // the network board, and starting to send it because the engine changed
      // would be a behaviour change disguised as a display one. SRBMiner does
      // report asic_power, so this is a choice, not a limitation.
      power: null,
      temp: this.temp,
      gpu: this.gpu,
    });
  }
}

module.exports = { SrbEngine, httpStatsFetcher, POLL_MS, API_PORT };
