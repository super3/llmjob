'use strict';

const { EventEmitter } = require('events');
const {
  buildAuthorize, buildSubmit, encode, parseMessage,
} = require('../shared/miner/stratum');
const {
  PROFILE, meetsTarget, rankMatches, shareBound, buildConfig52,
} = require('../shared/miner/pearlhash');
const { hash } = require('../shared/miner/blake3');
const { buildShareProof } = require('../shared/miner/shareProof');
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

// The card a core opened, as { index, name }, or null when it won't say.
//
// A core built before the device choice existed has no `device` at all, and the
// host must stay usable against it — an older pearl_core.node beside a newer
// app is the normal state of a rig mid-upgrade. Null then means "unknown", and
// every consumer falls back to what it did before.
function readDevice(core) {
  const d = core && core.device;
  if (!d || !Number.isInteger(d.index) || d.index < 0) return null;
  return { index: d.index, name: d.name ? String(d.name) : 'GPU ' + d.index };
}

class PearlMiner extends EventEmitter {
  constructor({ connect, createCore, reconnectMs } = {}) {
    super();
    this.connect = connect;                 // (host, port) -> socket
    this.createCore = createCore || null;   // (profile) -> core, or null when unbuilt
    this.reconnectMs = reconnectMs == null ? RECONNECT_MS : reconnectMs;

    this.sock = null;
    // One core per card. `cores` is [{ core, device }] in the order they were
    // started; `device` is what that core reported, so it is the card really
    // mining rather than the one we asked for.
    this.cores = [];
    this.hashrates = new Map();   // card index -> its latest TH/s
    this.running = false;
    this.authorized = false;
    this.job = null;          // the current parsed job the core is searching
    this.buf = '';            // partial-line accumulator for the socket
    this.submitId = 100;      // submit request ids start clear of the authorize id (1)
    // submit id -> { jobId, index }. The card index rides along so an accept or
    // a reject lands on the card that found the share, not on card 0.
    this.pending = new Map();
    this.settings = null;
  }

  isRunning() { return this.running; }

  // The cards this rig is mining on, as [{ index, name }]. Empty before a start,
  // and [null] entries are filtered out — a core that won't name its card leaves
  // the caller on its old single-card behaviour.
  devices() {
    return this.cores.map((c) => c.device).filter(Boolean);
  }

  start(settings = {}) {
    if (this.running) return false;
    this.settings = settings;
    this.running = true;
    this.authorized = false;
    this.job = null;
    this.buf = '';
    // Cleared here, not left from the last run: if no core starts this time, the
    // honest answer is "no cards", not the ones a previous run got.
    this.cores = [];
    this.hashrates = new Map();   // card index -> its latest TH/s

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

    if (!this._startCores(settings, wallet, worker)) {
      this.running = false;
      return false;
    }

    this._openSocket(host, Number(port), wallet, worker);
    return true;
  }

  // Start one core per card, and keep the ones that start.
  //
  // `settings.gpus` is the card list (shared/gpu.planMinerGpus, via
  // probe.detectMinerGpus). An empty list means nvidia-smi told us nothing, so
  // we start a single core with no index and let it choose — the single-card
  // behaviour that shipped before.
  //
  // A card that refuses is skipped, not fatal. The usual reason is no room: the
  // local LLM holds most of the VRAM on that card. One full card used to take
  // the whole rig's mining down with it; now the rest keep mining.
  //
  // Each core gets its own slice of the search space (saltBase/saltStride), or
  // every card would search the same operands and find the same shares.
  _startCores(settings, wallet, worker) {
    const profile = settings.profile || PROFILE;
    const cards = Array.isArray(settings.gpus) && settings.gpus.length
      ? settings.gpus
      : [null];                       // no list: one core, its own choice of card
    const failures = [];

    for (let i = 0; i < cards.length; i++) {
      const card = cards[i];
      const opts = { saltBase: i, saltStride: cards.length };
      if (card && Number.isInteger(card.index)) opts.deviceIndex = card.index;
      let core;
      let device;
      try {
        core = this.createCore(profile, opts);
        // A factory that returns nothing rather than throwing: no core, so the
        // same answer as one that refused. Wiring is inside the try for the same
        // reason — a core we cannot listen to is a core we cannot mine with.
        if (!core) throw new Error('the Pearl core did not initialise');
        device = readDevice(core);
        this._wireCore(core, device, wallet, worker);
      } catch (e) {
        failures.push({ card, message: (e && e.message) || String(e) });
        continue;
      }
      this.cores.push({ core, device });
      // Name the card in the log, every run and every card. The core chooses it
      // — the host cannot see CUDA's device list — so this is the only place the
      // two halves of "which GPU is mining" are written down together. A rig
      // whose UI names one card and whose fan spins up on another (issue #226)
      // is diagnosable from a log file because of it.
      if (device) {
        this.emit('log', {
          level: 'info',
          line: 'mining on GPU ' + device.index + ' · ' + device.name,
        });
      }

      // A core that won't say which card it opened is one built before any of
      // this existed, and it ignores the card we asked for — it mines on CUDA's
      // device 0 whatever we pass. Starting a second such core would stack two
      // searches on that one card and label them as two. Mine on the one, which
      // is exactly what that build did on its own.
      if (!device && cards.length > 1) {
        this.emit('log', {
          level: 'info',
          line: 'this pearl_core.node predates per-card mining — mining on one card. '
            + 'Update to mine on all ' + cards.length + '.',
        });
        break;
      }
    }

    for (const f of failures) {
      const where = f.card
        ? 'GPU ' + f.card.index + (f.card.name ? ' (' + f.card.name + ')' : '')
        : 'the GPU';
      this.emit('log', {
        level: this.cores.length ? 'info' : 'error',
        line: 'skipping ' + where + ': ' + f.message,
      });
    }

    // Nothing started at all. Report the first reason as the error: with one
    // card that IS the reason, and the log above has already listed the rest.
    // There is always a reason — every card either started or failed.
    if (!this.cores.length) {
      this.emit('error', new Error(failures[0].message));
      return false;
    }
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
    sock.on('error', (err) => {
      this.emit('log', { level: 'error', line: 'pool socket error: ' + err.message });
      // Structured, because a name that never resolves and a pool that is down
      // look identical in the log line above. The host says so once, naming the
      // host it tried — a rig that cannot do DNS otherwise just looks broken.
      this.emit('connect-failed', {
        reason: err.message,
        dns: /ENOTFOUND|EAI_AGAIN|getaddrinfo/i.test(String(err.message)),
      });
    });
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
        // `index` is the card that found it, so a multi-card rig credits the
        // right one. Card 0 when the submit is unknown to us, which is the same
        // bucket a single-card rig has always used.
        this.emit('share', { jobId: p ? p.jobId : null, accepted: true, index: p ? p.index : 0 });
        this.emit('log', { level: 'info', line: 'share accepted' });
        break;
      }
      case 'submit-rejected': {
        const p = this.pending.get(m.id);
        this.pending.delete(m.id);
        this.emit('rejected', {
          jobId: p ? p.jobId : null, reason: errText(m.error), index: p ? p.index : 0,
        });
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
    // job_key binds the header to the mining configuration, and everything
    // downstream -- both operands, both commitment roots, the noise seeds --
    // hangs off it. The pool never sends the configuration, so both sides
    // derive this independently and a single wrong byte is silent.
    this.jobKey = hash(Buffer.concat([job.header, buildConfig52(this.profile())]));
    this.emit('job', { jobId: job.jobId, height: job.height });
    // No null guard: start() only reaches here with a live core (a createCore
    // that throws OR returns falsy both land in its catch and abort the start).
    // The core compares against the SCALED bound, not the pool's raw target.
    // The protocol makes the bound easier in proportion to the work one attempt
    // costs — that is the same factor that makes a hashrate here count
    // multiply-accumulates rather than attempts. Comparing against the raw
    // target makes shares 65536x rarer than the pool intends, which is
    // indistinguishable from simply being slow.
    const bound = shareBound(job.target, this.profile());
    if (bound == null) {
      this.emit('log', { level: 'error', line: 'pool target is too easy to scale for this profile; ignoring the job' });
      return;
    }
    // Every card gets the same job. They search different salts (see
    // _startCores), so the same job is a different search on each of them.
    for (const c of this.cores) c.core.setJob({ header: job.header, target: bound, jobId: job.jobId });
  }

  // The profile this miner mines, which a caller may override wholesale.
  profile() {
    return (this.settings && this.settings.profile) || PROFILE;
  }

  _wireCore(core, device, wallet, worker) {
    core.on('hashrate', (th) => {
      // Per card for the UI and the board, and summed for what we tell the pool.
      if (device) this.hashrates.set(device.index, th);
      this.hashrate = this.totalHashrate();
      this.emit('hashrate', th, device);
    });
    core.on('error', (err) => this.emit('error', err));
    core.on('hit', (hit) => this._onHit(hit, wallet, worker, device));
  }

  // The rig's throughput: every card's latest tick, added up. A card that has
  // not ticked yet contributes nothing rather than a guess.
  totalHashrate() {
    let total = 0;
    for (const th of this.hashrates.values()) total += Number(th) || 0;
    return total;
  }

  // A candidate the core found. Re-verify it against the CURRENT job's target in
  // JS before submitting: the core may have been searching a job that vardiff has
  // since moved, and a bad submit earns a ban. Stale hits are dropped silently —
  // they are not errors, just races.
  _onHit(hit, wallet, worker, device) {
    const job = this.job;
    if (!job || hit.jobId !== job.jobId) return;
    if (!meetsTarget(hit.jackpotHash, shareBound(job.target, this.profile()))) {
      this.emit('log', { level: 'info', line: 'dropping a hit that no longer meets target (vardiff moved)' });
      return;
    }
    // Certify the hit before it goes anywhere. The proof was captured on the
    // device at the moment of the hit, because the search re-draws its operands
    // every few tens of milliseconds and a proof read back afterwards belongs to
    // a different matrix than the hash it certifies.
    const plainProof = buildShareProof(hit, this.jobKey, this.profile());
    if (!plainProof) {
      this.emit('log', { level: 'error', line: 'dropping a hit whose proof does not verify locally' });
      return;
    }
    const id = this.submitId++;
    this.pending.set(id, { jobId: job.jobId, index: device ? device.index : 0 });
    this.sock.write(encode(buildSubmit(id, {
      jobId: job.jobId, plainProof, hashrate: (this.hashrate || 0) * 1e12,
    })));
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
    for (const c of this.cores) {
      try { c.core.stop(); } catch (e) { /* that core is already gone */ }
    }
    this.cores = [];
    this.hashrates.clear();
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
