'use strict';

const { EventEmitter } = require('events');
const { combinePayoutAddress } = require('../shared/address');
const { DEV_FEE_PCT } = require('../shared/minerSelect');

// PeakMiner, wearing the same clothes PearlEngine wears.
//
// PearlEngine IS the miner (our core, driven in-process). This one WATCHES a
// miner: PeakMiner is a separate proprietary process, so everything the UI reads
// has to be recovered from outside it. The surface is identical on purpose --
// start/stop/isRunning/gpuIndex plus log/event/error/stopped -- so choosing an
// engine stays a choice of constructor and nothing else, which is what keeps
// main.js and earn-cli.js from growing a second code path.
//
// Telemetry comes from PeakMiner's own HTTP API (127.0.0.1:4068/summary), NOT
// from scraping its console. The console output is a redrawn box-drawing table
// with ANSI colour, aligned for humans and free to change between releases;
// the JSON is stable and gives per-GPU hashrate, shares, temperature and power
// directly. alpha-miner was scraped and that is exactly how it broke.
//
// The 2% dev fee is disclosed on every start. Our own core takes nothing, so a
// user swapping engines is agreeing to something they were not agreeing to
// before, and it should not take reading the vendor's site to find that out.
//
// Everything IO is injected (spawn, fetchSummary, kill) so the whole lifecycle
// is testable with no binary, no GPU and no sockets.

const POLL_MS = 5000;
const STATUS_INTERVAL_S = 30;
const KILL_GRACE_MS = 8000;
const API_PORT = 4068;
// The escape character is the point: this strips the colour PeakMiner writes to
// its console, so the log lines we forward are readable in journalctl.
// eslint-disable-next-line no-control-regex
const ANSI = /\x1B\[[0-9;]*[a-zA-Z]/g;

// A summary fetcher over an injected http module. Kept here rather than in the
// callers so both entry points share one implementation, and injectable so the
// tests never open a socket.
function httpSummaryFetcher(http) {
  return (port) => new Promise((resolve, reject) => {
    const req = http.get({ host: '127.0.0.1', port, path: '/summary', timeout: 4000 }, (res) => {
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
    req.on('timeout', () => req.destroy(new Error('summary timeout')));
    req.on('error', reject);
  });
}

class PeakEngine extends EventEmitter {
  // No readTemps here, unlike PearlEngine: /summary already carries the card
  // temperature, so there is nothing to poll nvidia-smi for.
  constructor({
    binPath, spawn, fetchSummary, kill,
    apiPort, pollMs, statusIntervalS, killGraceMs,
  } = {}) {
    super();
    this.binPath = binPath || null;
    this.spawn = spawn;
    this.fetchSummary = fetchSummary;
    this.kill = kill || ((pid, sig) => process.kill(pid, sig));
    this.apiPort = apiPort || API_PORT;
    this.pollMs = pollMs || POLL_MS;
    this.statusIntervalS = statusIntervalS || STATUS_INTERVAL_S;
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
        line: 'PeakMiner binary not found -- nothing to mine with.',
      });
      this.emit('stopped', 0);
      return false;
    }

    const wallet = combinePayoutAddress(settings.address, settings.mdlAddress);
    const worker = settings.worker || 'rig01';
    const args = [
      '--coin', 'pearl',
      '-o', 'stratum+tcp://' + this.endpoint,
      '-u', wallet,
      '-w', worker,
      '--status-interval', String(this.statusIntervalS),
    ];

    // Said every start, not once at install: the fee is a standing cost and our
    // own engine's is zero, so the difference should be visible in the log the
    // operator actually reads.
    this.emit('log', {
      level: 'info',
      line: 'mining with PeakMiner (' + this.binPath + ') — '
        + 'proprietary, ' + DEV_FEE_PCT + '% dev fee. '
        + 'Use --miner native for the built-in zero-fee core.',
    });

    let child;
    try {
      // detached so the miner leads its own process group. PeakMiner outlives a
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
        p = Promise.resolve(this.fetchSummary(this.apiPort));
      } catch (e) {
        return;
      }
      p.then((s) => this._applySummary(s)).catch(() => { /* miner still starting */ });
    };
    this.pollTimer = setInterval(tick, this.pollMs);
    if (this.pollTimer.unref) this.pollTimer.unref();
    tick();
  }

  // Translate one /summary document into the two events the UI reads.
  _applySummary(s) {
    if (!s || typeof s !== 'object') return;
    const gpu0 = Array.isArray(s.gpus) && s.gpus.length ? s.gpus[0] : null;

    // The API reports H/s; every counter downstream of here is TH/s.
    const hs = Number(s.hashrate);
    if (Number.isFinite(hs)) this.hashrate = hs / 1e12;
    const acc = Number(s.accepted_shares);
    if (Number.isFinite(acc)) this.accepted = acc;
    const inv = Number(s.invalid_shares);
    if (Number.isFinite(inv)) this.rejected = inv;
    if (gpu0) {
      const t = Number(gpu0.temperature_c);
      this.temp = Number.isFinite(t) && t > 0 ? t : null;
      if (!this.gpu && gpu0.name) this.gpu = gpu0.name;
    }

    // The pool has us: this is the moment the card is actually mining, and the
    // same point PearlEngine announces on its first job.
    if (!this._announced && s.pool && s.pool.connected) {
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
      // would be a behaviour change disguised as a display one. PeakMiner does
      // report power_w, so this is a choice, not a limitation.
      power: null,
      temp: this.temp,
      gpu: this.gpu,
    });
  }
}

module.exports = { PeakEngine, httpSummaryFetcher, POLL_MS, API_PORT };
