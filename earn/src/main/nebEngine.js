'use strict';

const { EventEmitter } = require('events');
const crypto = require('crypto');
const stratum = require('../shared/neb/stratum');
const nh = require('../shared/neb/nebhash2');
const hdr = require('../shared/neb/header');
const { shortenNebAddress } = require('../shared/neb/address');
const { wireEndpoint } = require('../shared/config');
const { hitList } = require('./nebCore');

// The NEB (Neural Engine Byte) miner: the pool connection (shared/neb/stratum,
// the chain's pool protocol) and one core per card (main/nebCore.js). It has
// the surface the shells drive the Pearl, Monero and Quantus engines by:
//
//   start(settings) -> bool   stop()   isRunning()   poolConnected()   snapshot()
//   'log'        { level, line }
//   'event'      { type: 'status', gpuIndex, hashrate (H/s), accepted, rejected, gpu, coin: 'neb' }
//                { type: 'connected', gpuIndex, endpoint, gpu }
//                { type: 'connect-failed', reason, dns }
//   'authorized' { host, address, family }
//   'stopped'    code: 1 when it gave up on its own, else 0
//   'error'      Error: why it gave up
//
// settings: { address, worker, endpoint, cards, datasets, datasetHash }.
// `datasets` is the network's dataset file parsed (nebhash2.parseDatasets) and
// `datasetHash` its BLAKE3 as hex; the pool names the hash it expects when the
// miner subscribes, and a miner with another file stops.
//
// The pool sends each block's model (13,000 bytes) before the job that uses
// it. Blocks alternate between the network's two datasets, so a card holds a
// model for each slot. Each card loops: take the pool's current job, give the
// core the job's model if it does not have it, scan a batch of nonces under a
// fresh extranonce2, evaluate every hit again on the reference, submit.
//
// The hashrate is attempts evaluated a second, not nonces looked at: a core
// skips the nonces whose samples are mostly live (nebCore.js says why), and
// the skipped ones earn nothing. It moves with every model.
//
// The triple table. A core always builds a model's pair table. A table of row
// triples lets it evaluate the attempts with 3 live samples too, which on a
// well-trained model are as many again; but it is rows^3 / 6 entries of 32
// bytes, and the card mines nothing while it is built. So for each model the
// engine picks how many rows the table covers (triplePlan): as many as the
// card's memory holds, cut back until the build pays for itself within the
// time a block of that slot has been lasting. The slot with fewer live rows
// gains most from a table, so it has first call on the memory.

const RECONNECT_MS = 5000;
const LOGIN_TIMEOUT_MS = 20 * 1000;
const LOGIN_RETRY_MS = 60 * 1000;
const SILENCE_MS = 5 * 60 * 1000;
const WATCHDOG_MS = 30 * 1000;
const STATUS_MS = 1000;
const BATCH_MS = 250;
const FIRST_BATCH = 2 ** 22;
const MAX_BATCH = 2 ** 32;
const MAX_HITS = 256;
const BAD_HIT_LIMIT = 3;
const POOL_GRACE_MS = 30 * 1000;
// Models kept: the two slots' current ones and the two before.
const MODELS_KEPT = 4;
const LANE_BYTES = 6;
// The triple table (triplePlan). A core covers at most 2,000 rows.
const TRIPLE_MAX_ROWS = 2000;
// Card memory a triple table leaves free.
const TRIPLE_RESERVE = 512 * 1048576;
// Below this many rows a table adds too little to bother.
const TRIPLE_MIN_ROWS = 64;
// Until measured: how long a slot's block lasts, and table entries a card builds a millisecond.
const BLOCK_MS_GUESS = 30000;
const TRIPLE_RATE_GUESS = 100000;
// A table must be expected to add this much to a block's work to be built.
const TRIPLE_MIN_GAIN = 0.02;

const tripleEntries = (rows) => rows * (rows + 1) * (rows + 2) / 6;
const tripleBytes = (rows) => tripleEntries(rows) * 32;

// The most rows a triple table of `bytes` covers.
function tripleRowsFor(bytes) {
  let rows = Math.max(0, Math.floor(Math.cbrt(Math.max(bytes, 0) * 6 / 32)));
  while (tripleBytes(rows) > bytes) rows--;
  return rows;
}

// The triple table worth building for a model, as { rows, bytes, gain }, or
// null for none.
//   live, samples   the model's live samples and the dataset's size
//   bytes           card memory the table may take
//   rate            table entries the card builds a millisecond
//   life            how long a block of this slot lasts, in milliseconds
// With a fraction c^3 of the triples covered (c = rows / live) the card
// evaluates 1 + g c^3 times the attempts, g being the model's 3-live share over
// its 2-or-fewer share. The build costs entries / rate milliseconds of a
// block that ends at a random time, so the expected work of the block is
// (1 + g c^3) e^(-cost / life) of what it is without a table.
function triplePlan({ live, samples, bytes, rate, life }) {
  const most = Math.min(live, TRIPLE_MAX_ROWS, tripleRowsFor(bytes));
  if (most < TRIPLE_MIN_ROWS) return null;
  const p = live / samples;
  const g = nh.exactShare(p, 3) / nh.candidateShare(p, 2);
  let best = null;
  for (let k = 1; k <= 16; k++) {
    const rows = Math.floor(most * k / 16);
    const gain = (1 + g * (rows / live) ** 3) * Math.exp(-tripleEntries(rows) / rate / life) - 1;
    if (gain >= TRIPLE_MIN_GAIN && (!best || gain > best.gain)) best = { rows, bytes: tripleBytes(rows), gain };
  }
  return best;
}

function errText(e) {
  return (e && e.message) || String(e);
}

class NebEngine extends EventEmitter {
  constructor({ connect, factory, agent, now, random, timing } = {}) {
    super();
    this.connect = connect;
    this.factory = factory || null;
    this.agent = agent || 'llmjob-earn';
    this.now = now || Date.now;
    this.random = random || crypto.randomBytes;
    this.t = Object.assign({
      reconnect: RECONNECT_MS, loginTimeout: LOGIN_TIMEOUT_MS, loginRetry: LOGIN_RETRY_MS, silence: SILENCE_MS,
      watchdog: WATCHDOG_MS, status: STATUS_MS, batchMs: BATCH_MS, firstBatch: FIRST_BATCH,
    }, timing);
    this.running = false;
    this.cards = [];
    // How long each slot's blocks have been lasting, in milliseconds: kept across reconnects.
    this.blockMs = [];
    this._reset();
  }

  _reset() {
    this.sock = null;
    this.buf = '';
    this.session = null;       // { extranonce1, extranonce2Size }
    this.authorized = false;
    this.lostAt = null;
    this.loginRefused = false;
    this.poolError = null;
    this.poolFails = 0;
    this.target = null;
    this.models = new Map();   // weights hash -> { slot, weights (Buffer, one int8 a weight), m (the reference's model) }
    this.job = null;
    this.blockStart = null;
    this.sent = new Set();
    this.submitId = 10;
    this.pending = new Map();  // submit id -> card
    this.lastRx = 0;
    this.timers = new Set();
    this.waiters = [];
    this.announced = false;
  }

  isRunning() { return this.running; }

  poolConnected(now = this.now()) {
    if (!this.running) return false;
    if (this.authorized) return true;
    return this.lostAt != null && now - this.lostAt < POOL_GRACE_MS;
  }

  snapshot(now = this.now()) {
    const on = this.poolConnected(now);
    const sum = (k) => this.cards.reduce((a, c) => a + c[k], 0);
    return {
      state: this.running ? 'mining' : 'stopped',
      hashrate: on ? sum('rate') : 0,
      accepted: sum('accepted'),
      rejected: sum('rejected'),
      cards: this.cards.length,
      poolError: on ? null : this.poolError,
    };
  }

  _log(level, line) {
    this.emit('log', { level, line: 'neb: ' + line });
  }

  _timer(fn, ms, repeat) {
    const t = (repeat ? setInterval : setTimeout)(() => {
      if (!repeat) this.timers.delete(t);
      fn();
    }, ms);
    this.timers.add(t);
    return t;
  }

  start(settings = {}) {
    if (this.running) return false;
    this._reset();
    this.settings = Object.assign({}, settings);
    const fail = (message) => {
      this._log('error', message);
      this.emit('error', new Error(message));
      return false;
    };
    if (!this.factory) return fail('there is no NEB core in this build');
    const datasets = settings.datasets;
    if (!Array.isArray(datasets) || !datasets.length) return fail('the NEB dataset file is not loaded');
    const indices = Array.isArray(settings.cards) ? settings.cards : this.factory.devices.map((d) => d.index);
    // A random lane per run and card: two runs, or two cards, never try the same header.
    const runLane = this.random(LANE_BYTES - 1);
    this.cards = [];
    for (const index of indices) {
      let core;
      try {
        core = this.factory.createCore(index);
        datasets.forEach((ds, slot) => core.setDataset(slot, ds.arch, ds.values, ds.labels));
      } catch (e) {
        this._log('error', 'card ' + index + ' would not open: ' + errText(e));
        continue;
      }
      const name = (core.device && core.device.name) || 'card ' + index;
      this.cards.push({
        index, core, name, lane: Buffer.concat([runLane, Buffer.from([index & 0xff])]), counter: this.random(4).readUInt32LE(0), loaded: [],
        batch: this.t.firstBatch, rate: 0, screened: 0, accepted: 0, rejected: 0, bad: 0, dead: false,
        tri: [], tripleRate: 0,
      });
      this._log('info', 'card ' + index + ': ' + name);
    }
    if (!this.cards.length) return fail('no card would open a NEB core');
    this.running = true;
    this.login = stratum.loginOf(settings.address, settings.worker);
    const { host, port, tls } = wireEndpoint(settings.endpoint);
    this.host = host;
    this.port = Number(port);
    this.tls = tls;
    this._log('info', 'mining NEB for ' + shortenNebAddress(settings.address) + ' on ' + this.cards.length
      + ' card' + (this.cards.length === 1 ? '' : 's') + ' · ' + settings.endpoint);
    this._openSocket();
    this._timer(() => this._checkSilence(), this.t.watchdog, true);
    this._timer(() => { for (const c of this.cards) this._status(c); }, this.t.status, true);
    for (const c of this.cards) this._run(c).catch((e) => this._cardFailed(c, 'it answered with ' + errText(e)));
    return true;
  }

  // ── the cards ────────────────────────────────────────────────────────────

  _waitForWork() {
    return new Promise((resolve) => this.waiters.push(resolve));
  }

  _wake() {
    const w = this.waiters;
    this.waiters = [];
    for (const resolve of w) resolve();
  }

  // The job a card may mine now: the pool's current one, until its expiry.
  _liveJob() {
    const job = this.authorized ? this.job : null;
    if (!job) return null;
    if (job.expiry != null && this.now() / 1000 >= job.expiry) return null;
    return job;
  }

  async _run(card) {
    while (this.running) {
      const job = this._liveJob();
      if (!job) {
        await this._waitForWork();
        continue;
      }
      try {
        if (card.loaded[job.slot] !== job.weightsHash) {
          const info = await card.core.setModel(job.slot, job.model.weights);
          if (!this.running) return;
          card.loaded[job.slot] = job.weightsHash;
          this._modelLoaded(card, job, info);
          await this._triples(card, job.slot, info);
          continue;  // the pool may have moved on while the tables were built
        }
        const count = card.batch;
        const extranonce2 = this._extranonce2(card);
        const header = hdr.buildHeader(job, this.session.extranonce1, extranonce2, job.timeStart);
        const t0 = this.now();
        const res = await card.core.scan({ slot: job.slot, header, target: job.poolTarget, start: 0, count, maxHits: MAX_HITS });
        if (!this.running) return;
        this._timed(card, count, res.evaluated, Math.max(this.now() - t0, 1));
        if (res.overflow) this._log('warn', 'card ' + card.index + ' found more than ' + MAX_HITS + ' hits in one batch');
        // A share is good for its parent block: a job that replaced this one
        // on the same parent does not make its hits stale. A new block does.
        if (!this.job || !this.job.parent.equals(job.parent)) continue;
        for (const nonce of hitList(res)) {
          this._check(card, job, header, extranonce2, nonce);
          if (card.dead) return;
        }
      } catch (e) {
        if (!this.running) return;
        this._cardFailed(card, 'its core failed: ' + errText(e));
        return;
      }
    }
  }

  // The card's lane, then a counter: a new header for every batch.
  _extranonce2(card) {
    const size = this.session.extranonce2Size;
    const b = Buffer.alloc(size);
    const cb = Math.min(6, size);
    card.lane.copy(b, 0, 0, Math.min(LANE_BYTES, size - cb));
    b.writeUIntBE(card.counter % 2 ** (8 * cb), size - cb, cb);
    card.counter++;
    return b;
  }

  _modelLoaded(card, job, info) {
    const share = nh.candidateShare(info.live / info.samples, info.maxLive);
    this._log('info', 'card ' + card.index + ' model ' + job.weightsHash.slice(0, 8) + ' (slot ' + job.slot + '): '
      + info.live + ' of ' + info.samples + ' samples live · ' + (share * 100).toFixed(2) + '% of attempts evaluated · table '
      + Math.round(info.tableBytes / 1048576) + ' MB in ' + Math.round(info.cpuMs + info.gpuMs) + ' ms');
  }

  // The model's triple table, when one is worth building on this card now.
  async _triples(card, slot, info) {
    if (info.maxLive < 2 || typeof card.core.buildTriples !== 'function') return;
    const mine = card.tri[slot] || (card.tri[slot] = { live: 0, want: 0, have: 0 });
    mine.live = info.live;
    mine.want = tripleBytes(Math.min(info.live, TRIPLE_MAX_ROWS));
    // The card's free memory and this slot's own buffer, less what the slots with fewer live rows have yet to take.
    let bytes = info.freeBytes + info.tripleBytes - TRIPLE_RESERVE;
    card.tri.forEach((other, i) => {
      if (i !== slot && other.live < info.live) bytes -= Math.max(0, other.want - other.have);
    });
    const plan = triplePlan({
      live: info.live, samples: info.samples, bytes, rate: card.tripleRate || TRIPLE_RATE_GUESS,
      life: this.blockMs[slot] || BLOCK_MS_GUESS,
    });
    if (!plan) return;
    const t = await card.core.buildTriples(slot, plan.rows, plan.bytes);
    if (!this.running) return;
    mine.have = t.tableBytes;
    if (t.entries > 0 && t.ms > 0) card.tripleRate = t.entries / t.ms;
    this._log('info', 'card ' + card.index + ' triple table (slot ' + slot + '): ' + t.rows + ' of ' + info.live + ' live rows · '
      + Math.round(t.tableBytes / 1048576) + ' MB in ' + Math.round(t.ms) + ' ms');
  }

  // A batch's speed: the card's rate (attempts evaluated a second), and the
  // next batch's size.
  _timed(card, count, evaluated, ms) {
    const rate = evaluated * 1000 / ms;
    const screened = count * 1000 / ms;
    card.rate = card.rate ? card.rate * 0.8 + rate * 0.2 : rate;
    card.screened = card.screened ? card.screened * 0.8 + screened * 0.2 : screened;
    card.batch = Math.max(2 ** 16, Math.min(MAX_BATCH, count * 8, Math.round(count * this.t.batchMs / ms)));
  }

  // A hit, evaluated in full on the reference before anything is sent.
  _check(card, job, header, extranonce2, nonce) {
    const r = nh.shareEval(job.model.m, this.settings.datasets[job.slot], hdr.attemptKey(header, nonce));
    if (!nh.meetsTarget(r.ticket, job.poolTarget)) {
      card.bad++;
      this._log('warn', 'card ' + card.index + ' reported a nonce that is not a hit (' + card.bad + ' of ' + BAD_HIT_LIMIT + ')');
      if (card.bad >= BAD_HIT_LIMIT) this._cardFailed(card, 'its core keeps reporting nonces that are not hits');
      return;
    }
    const share = { jobId: job.jobId, extranonce2: extranonce2.toString('hex'), time: job.timeStart, nonce };
    const key = share.extranonce2 + ':' + nonce;
    if (this.sent.has(key)) return;
    this.sent.add(key);
    const id = this.submitId++;
    this.pending.set(id, card);
    this.sock.write(stratum.encode(stratum.buildSubmit(id, this.login, share)));
  }

  _cardFailed(card, why) {
    card.dead = true;
    card.rate = 0;
    try { card.core.close(); } catch (e) { /* already gone */ }
    this._log('error', 'card ' + card.index + ' stopped: ' + why);
    this._status(card);
    if (this.cards.every((c) => c.dead)) {
      const message = 'every card stopped; the NEB miner gives up';
      this._log('error', message);
      this.emit('error', new Error(message));
      this.stop(true);
    }
  }

  _status(card) {
    this.emit('event', {
      type: 'status', gpuIndex: card.index, hashrate: this.poolConnected() && !card.dead ? card.rate : 0,
      screened: card.screened, accepted: card.accepted, rejected: card.rejected, gpu: card.name, coin: 'neb',
    });
  }

  // ── the pool ─────────────────────────────────────────────────────────────

  _openSocket() {
    let sock;
    try {
      sock = this.connect(this.host, this.port, this.tls);
    } catch (e) {
      this._log('error', 'pool connection failed: ' + errText(e));
      this._timer(() => this._openSocket(), this.t.reconnect);
      return;
    }
    this.sock = sock;
    this.buf = '';
    this.lastRx = this.now();
    this._timer(() => {
      if (this.sock !== sock || this.authorized) return;
      this.poolError = 'the NEB pool did not answer the login';
      if (this.poolFails === 0) this._log('warn', 'no login at the pool in ' + Math.round(this.t.loginTimeout / 1000) + ' s; reconnecting');
      sock.destroy();
    }, this.t.loginTimeout);
    sock.on('connect', () => {
      this._log('info', 'connecting to ' + this.host + ':' + this.port + ' · ' + shortenNebAddress(this.login));
      sock.write(stratum.encode(stratum.buildSubscribe(this.agent)));
    });
    sock.on('data', (chunk) => {
      if (this.sock !== sock) return;
      this.lastRx = this.now();
      this.buf += String(chunk);
      let i;
      while (this.sock === sock && (i = this.buf.indexOf('\n')) >= 0) {
        const line = this.buf.slice(0, i);
        this.buf = this.buf.slice(i + 1);
        this._onPoolMessage(stratum.parseMessage(line));
      }
    });
    sock.on('error', (err) => {
      this.poolError = 'can\'t reach the NEB pool: ' + err.message;
      if (this.poolFails === 0) this._log('error', 'pool socket error: ' + err.message);
      this.emit('event', {
        type: 'connect-failed', reason: err.message, dns: /ENOTFOUND|EAI_AGAIN|getaddrinfo/i.test(String(err.message)),
      });
    });
    sock.on('close', () => {
      if (this.sock !== sock) return;
      this.sock = null;
      if (this.authorized) this.lostAt = this.now();
      else {
        this.poolFails++;
        if (!this.poolError) this.poolError = 'the NEB pool closed the connection before the login';
      }
      this.authorized = false;
      this.session = null;
      this.job = null;
      this.pending.clear();
      const wait = this.loginRefused ? this.t.loginRetry : this.t.reconnect;
      this.loginRefused = false;
      if (this.poolFails <= 1) this._log('info', 'pool connection closed; reconnecting in ' + Math.round(wait / 1000) + ' s');
      this._timer(() => this._openSocket(), wait);
    });
  }

  _refuse(why) {
    this.loginRefused = true;
    this.poolError = why;
    this._log('error', why);
    this.sock.destroy();
  }

  _setModel(m) {
    const ds = this.settings.datasets[m.slot];
    if (!ds) return this._log('error', 'ignoring a model for dataset ' + m.slot + ', which the dataset file does not have');
    if (m.weights.length * 2 !== nh.archOf(ds.arch).nweights) return this._log('error', 'ignoring a model of ' + m.weights.length + ' bytes');
    if (nh.weightsHash(m.weights).toString('hex') !== m.weightsHash) return this._log('error', 'ignoring a model whose hash is not the one the pool names');
    const model = nh.modelFromPacked(ds.arch, m.weights);
    this.models.set(m.weightsHash, { slot: m.slot, m: model, weights: Buffer.from(model.w.buffer, model.w.byteOffset, model.w.length) });
    while (this.models.size > MODELS_KEPT) this.models.delete(this.models.keys().next().value);
    return undefined;
  }

  _setJob(m) {
    const model = this.models.get(m.weightsHash);
    if (!model || model.slot !== m.slot || !this.target) {
      this._log('error', 'ignoring job ' + m.jobId + ': ' + (this.target ? 'the pool has not sent its model' : 'the pool has set no target'));
      this.job = null;
      return;
    }
    if (!this.job || !this.job.parent.equals(m.parent)) {
      // A new block: note how long the last one of its slot took.
      const now = this.now();
      if (this.job) {
        const took = now - this.blockStart;
        const before = this.blockMs[this.job.slot];
        this.blockMs[this.job.slot] = before ? before * 0.7 + took * 0.3 : took;
      }
      this.blockStart = now;
      this.sent.clear();
    }
    this.job = Object.assign({}, m, { model, poolTarget: this.target });
    if (!this.announced) {
      this.announced = true;
      for (const c of this.cards) {
        this.emit('event', { type: 'connected', gpuIndex: c.index, endpoint: this.settings.endpoint, gpu: c.name });
      }
    }
    this._wake();
  }

  _onPoolMessage(m) {
    switch (m.kind) {
      case 'subscribed':
        if (this.settings.datasetHash && m.datasetHash !== this.settings.datasetHash) {
          this._refuse('the NEB pool mines another dataset file (' + m.datasetHash.slice(0, 16) + '…); this miner has '
            + this.settings.datasetHash.slice(0, 16) + '…');
          break;
        }
        this.session = { extranonce1: m.extranonce1, extranonce2Size: m.extranonce2Size };
        this.sock.write(stratum.encode(stratum.buildAuthorize(this.login)));
        break;
      case 'subscribe-fail':
        this._refuse('the NEB pool refused the connection: ' + stratum.errorText(m.error));
        break;
      case 'authorized':
        this.authorized = true;
        this.lostAt = null;
        this.poolError = null;
        this.poolFails = 0;
        this.emit('authorized', { host: this.host, address: this.sock.remoteAddress, family: this.sock.remoteFamily });
        this._log('info', 'logged in to the pool');
        break;
      case 'login-fail':
        this._refuse('the NEB pool refused the login: ' + stratum.errorText(m.error));
        break;
      case 'target':
        this.target = m.target;
        break;
      case 'model':
        this._setModel(m);
        break;
      case 'bad-model':
        this._log('error', 'ignoring ' + m.reason);
        break;
      case 'job':
        this._setJob(m);
        break;
      case 'bad-job':
        this._log('error', 'ignoring a job this miner cannot mine' + (m.jobId ? ' (' + m.jobId + ')' : '') + ': ' + m.reason);
        this.job = null;
        break;
      case 'submit-accepted':
      case 'submit-rejected': {
        const card = this.pending.get(m.id);
        if (!card) break;
        this.pending.delete(m.id);
        if (m.kind === 'submit-accepted') {
          card.accepted++;
          this._log('info', 'NEB share accepted (card ' + card.index + ')');
        } else {
          card.rejected++;
          this._log('error', 'NEB share rejected (card ' + card.index + '): ' + stratum.errorText(m.error));
        }
        this._status(card);
        break;
      }
      case 'pool-error':
        this._log('error', 'pool: ' + stratum.errorText(m.error));
        break;
      case 'unknown':
      case 'unparseable':
        this._log('info', 'pool: ' + m.raw);
        break;
    }
  }

  _checkSilence() {
    if (!this.sock || this.now() - this.lastRx < this.t.silence) return;
    this._log('warn', 'no word from the pool in ' + Math.round(this.t.silence / 60000) + ' minutes; reconnecting');
    this.sock.destroy();
  }

  stop(failed = false) {
    if (!this.running) return false;
    this.running = false;
    for (const t of this.timers) { clearTimeout(t); clearInterval(t); }
    this.timers.clear();
    if (this.sock) {
      const s = this.sock;
      this.sock = null;
      try { s.destroy(); } catch (e) { /* already closed */ }
    }
    this.authorized = false;
    for (const c of this.cards) {
      if (c.dead) continue;
      c.rate = 0;
      try { c.core.close(); } catch (e) { /* already gone */ }
    }
    this._wake();
    this.emit('stopped', failed ? 1 : 0);
    return true;
  }
}

module.exports = {
  NebEngine, RECONNECT_MS, LOGIN_TIMEOUT_MS, LOGIN_RETRY_MS, SILENCE_MS, WATCHDOG_MS, STATUS_MS, BATCH_MS, FIRST_BATCH,
  MAX_BATCH, MAX_HITS, BAD_HIT_LIMIT, POOL_GRACE_MS, MODELS_KEPT, TRIPLE_MAX_ROWS, TRIPLE_RESERVE, TRIPLE_MIN_ROWS,
  BLOCK_MS_GUESS, TRIPLE_RATE_GUESS, TRIPLE_MIN_GAIN, tripleEntries, tripleBytes, tripleRowsFor, triplePlan,
};
