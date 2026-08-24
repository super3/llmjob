'use strict';

const { EventEmitter } = require('events');
const {
  buildAuthorize, buildSubmit, encode, parseMessage,
} = require('../shared/miner/stratum');
const { PROFILE, meetsTarget, rankMatches, shareBound } = require('../shared/miner/pearlhash');
const { combinePayoutAddress } = require('../shared/address');

// The host for our own Pearl miner: it owns the pool socket and the job/lifecycle
// state machine, and drives a PearlCore (the CUDA addon) for the actual search.
// It is the JS counterpart to a competitive miner's thin host layer — everything
// here is protocol, bookkeeping and safety; none of it touches the GPU directly.
//
// Everything IO is injected (connect → a duplex-ish socket, createCore → the
// native core factory) so the entire state machine is unit-testable against a
// fake socket and a fake core, with no network and no GPU.
//
// Emits, for the app to relay to the renderer exactly like MinerManager does:
//   started        { pool, wallet, worker }
//   log            { level, line }
//   authorized     {}
//   job            { jobId, height }
//   share          { jobId, accepted:true }        (pool confirmed)
//   rejected       { jobId, reason }
//   hashrate       thPerSec
//   stopped        {}
//   error          Error
//
// The dev fee is ZERO and there is no dev-address code path, by design — this is
// our own implementation written against the ISC-licensed reference, not a
// derivative of any fee-bearing miner, so there is nothing to disclose or route.

const RECONNECT_MS = 5000;

class PearlMiner extends EventEmitter {
  constructor({ connect, createCore, reconnectMs } = {}) {
    super();
    this.connect = connect;                 // (host, port) -> socket
    this.createCore = createCore || null;   // (profile) -> core, or null when unbuilt
    this.reconnectMs = reconnectMs == null ? RECONNECT_MS : reconnectMs;

    this.sock = null;
    this.core = null;
    this.running = false;
    this.authorized = false;
    this.job = null;          // the current parsed job the core is searching
    this.buf = '';            // partial-line accumulator for the socket
    this.submitId = 100;      // submit request ids start clear of the authorize id (1)
    this.pending = new Map(); // submit id -> { jobId }
    this.settings = null;
  }

  isRunning() { return this.running; }

  start(settings = {}) {
    if (this.running) return false;
    this.settings = settings;
    this.running = true;
    this.authorized = false;
    this.job = null;
    this.buf = '';

    // No core means the native addon is not built for this machine. That is a
    // clean, explicable stop — not a crash — so the host says exactly that and
    // does not open a pool socket it could never feed a share to.
    if (!this.createCore) {
      this.emit('log', {
        level: 'error',
        line: 'Pearl core is not built for this platform yet — nothing to mine with. '
          + 'Build earn/native (CUDA) or use a release that bundles pearl_core.node.',
      });
      this.running = false;
      this.emit('stopped', {});
      return false;
    }

    const wallet = combinePayoutAddress(settings.address, settings.mdlAddress);
    const worker = settings.worker || 'rig01';
    const [host, port] = String(settings.endpoint || '').split(':');
    this.emit('started', { pool: settings.endpoint, wallet, worker });

    try {
      this.core = this.createCore(settings.profile || PROFILE);
      this._wireCore(this.core, wallet, worker);
    } catch (e) {
      this.emit('error', e);
      this.running = false;
      return false;
    }

    this._openSocket(host, Number(port), wallet, worker);
    return true;
  }

  _openSocket(host, port, wallet, worker) {
    let sock;
    try {
      sock = this.connect(host, port);
    } catch (e) {
      this.emit('error', e);
      return;
    }
    this.sock = sock;

    sock.on('connect', () => {
      this.emit('log', { level: 'info', line: 'connecting to ' + host + ':' + port + ' · worker ' + worker });
      sock.write(encode(buildAuthorize(wallet, worker)));
    });
    sock.on('data', (chunk) => this._onData(chunk, wallet, worker));
    sock.on('error', (err) => this.emit('log', { level: 'error', line: 'pool socket error: ' + err.message }));
    sock.on('close', () => this._onClose(host, port, wallet, worker));
  }

  _onData(chunk, wallet, worker) {
    this.buf += String(chunk);
    let i;
    while ((i = this.buf.indexOf('\n')) >= 0) {
      const line = this.buf.slice(0, i);
      this.buf = this.buf.slice(i + 1);
      if (line.trim()) this._onMessage(line, wallet, worker);
    }
  }

  _onMessage(line, wallet, worker) {
    const m = parseMessage(line);
    switch (m.kind) {
      case 'auth-ok':
        this.authorized = true;
        this.emit('authorized', {});
        this.emit('log', { level: 'info', line: 'authorized' });
        break;
      case 'auth-fail':
        this.emit('log', { level: 'error', line: 'pool rejected the wallet: ' + errText(m.error) });
        break;
      case 'job':
        this._onJob(m);
        break;
      case 'bad-job':
        this.emit('log', { level: 'error', line: 'ignoring an unusable job' + (m.jobId ? ' ' + m.jobId : '') });
        break;
      case 'difficulty':
        // vardiff arrives independently of the job; the next mining.notify carries
        // the widened/narrowed target, so there is nothing to do but note it.
        this.emit('log', { level: 'info', line: 'pool difficulty → ' + m.difficulty });
        break;
      case 'submit-accepted': {
        const p = this.pending.get(m.id);
        this.pending.delete(m.id);
        this.emit('share', { jobId: p ? p.jobId : null, accepted: true });
        this.emit('log', { level: 'info', line: 'share accepted' });
        break;
      }
      case 'submit-rejected': {
        const p = this.pending.get(m.id);
        this.pending.delete(m.id);
        this.emit('rejected', { jobId: p ? p.jobId : null, reason: errText(m.error) });
        this.emit('log', { level: 'error', line: 'share rejected: ' + errText(m.error) });
        break;
      }
      case 'unparseable':
      case 'unknown':
        this.emit('log', { level: 'info', line: 'pool: ' + m.raw.slice(0, 200) });
        break;
      // No default: blank lines are filtered before dispatch, so 'empty' never
      // reaches this switch and a default would be a branch that cannot run.
    }
  }

  _onJob(job) {
    // Refuse a job whose stated rank is not the one the core was built for.
    // Post-softfork, mining any other rank produces work the network does not
    // credit — which is precisely how alpha-miner's Ada builds ended up running
    // rank 512 and earning nothing while looking perfectly healthy. A job that
    // states no rank (HeroMiners does not) is trusted: the pool decides.
    if (!rankMatches(job.rank, this.settings && this.settings.profile)) {
      this.emit('log', {
        level: 'error',
        line: 'refusing job ' + job.jobId + ': rank ' + job.rank + ' is not the credited '
          + (((this.settings && this.settings.profile) || PROFILE).rank) + ' — not mining uncredited work',
      });
      return;
    }
    this.job = job;
    this.emit('job', { jobId: job.jobId, height: job.height });
    // No null guard: start() only reaches here with a live core (a createCore
    // that throws OR returns falsy both land in its catch and abort the start).
    // The core compares against the SCALED bound, not the pool's raw target.
    // The protocol makes the bound easier in proportion to the work one attempt
    // costs — that is the same factor that makes a hashrate here count
    // multiply-accumulates rather than attempts. Comparing against the raw
    // target makes shares 65536x rarer than the pool intends, which is
    // indistinguishable from simply being slow.
    const bound = shareBound(job.target, PROFILE);
    if (bound == null) {
      this.emit('log', { level: 'error', line: 'pool target is too easy to scale for this profile; ignoring the job' });
      return;
    }
    this.core.setJob({ header: job.header, target: bound, jobId: job.jobId });
  }

  _wireCore(core, wallet, worker) {
    core.on('hashrate', (th) => this.emit('hashrate', th));
    core.on('error', (err) => this.emit('error', err));
    core.on('hit', (hit) => this._onHit(hit, wallet, worker));
  }

  // A candidate the core found. Re-verify it against the CURRENT job's target in
  // JS before submitting: the core may have been searching a job that vardiff has
  // since moved, and a bad submit earns a ban. Stale hits are dropped silently —
  // they are not errors, just races.
  _onHit(hit, wallet, worker) {
    const job = this.job;
    if (!job || hit.jobId !== job.jobId) return;
    if (!meetsTarget(hit.jackpotHash, shareBound(job.target, PROFILE))) {
      this.emit('log', { level: 'info', line: 'dropping a hit that no longer meets target (vardiff moved)' });
      return;
    }
    const id = this.submitId++;
    this.pending.set(id, { jobId: job.jobId });
    const msg = buildSubmit(id, {
      wallet, worker, jobId: job.jobId, nonce: hit.nonce,
      aSeed: hit.aSeed, bSeed: hit.bSeed, proof: hit.proof,
    });
    this.sock.write(encode(msg));
  }

  _onClose(host, port, wallet, worker) {
    this.authorized = false;
    if (!this.running) return;
    // The core keeps its current job loaded across a reconnect, so a brief pool
    // blip does not idle the GPU. Reopen after a backoff.
    this.emit('log', { level: 'info', line: 'pool connection closed; reconnecting' });
    this._scheduleReconnect(host, port, wallet, worker);
  }

  _scheduleReconnect(host, port, wallet, worker) {
    this._reconnectTimer = setTimeout(() => {
      this._openSocket(host, port, wallet, worker);
    }, this.reconnectMs);
    // Do not keep the event loop alive on a timer alone.
    this._reconnectTimer.unref();
  }

  stop() {
    if (!this.running) return false;
    this.running = false;
    if (this._reconnectTimer) { clearTimeout(this._reconnectTimer); this._reconnectTimer = null; }
    try { this.core.stop(); } catch (e) { /* core already gone */ }
    this.core = null;
    if (this.sock) { try { this.sock.destroy(); } catch (e) { /* already closed */ } this.sock = null; }
    this.job = null;
    this.pending.clear();
    this.emit('stopped', {});
    return true;
  }
}

// Only ever called for an auth-fail or submit-rejected, both of which carry a
// non-null error by construction — hence no null guard.
function errText(err) {
  return (err.code != null ? '[' + err.code + '] ' : '') + (err.message || '');
}

module.exports = { PearlMiner, RECONNECT_MS };
