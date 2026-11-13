'use strict';

const { EventEmitter } = require('events');
const crypto = require('crypto');
const stratum = require('../shared/quantus/stratum');
const q = require('../shared/quantus/qpow');
const { shortenQuantusAddress } = require('../shared/quantus/address');
const { u64Bytes, hitList } = require('./quantusCore');

// The Quantus (QTC) miner: the pool connection, and one core per card driven
// through the batch interface in main/quantusCore.js. It has the surface the
// shells drive PearlEngine and XmrEngine by:
//
//   start(settings) -> bool   stop()   isRunning()   poolConnected()   snapshot()
//   'log'        { level, line }
//   'event'      { type: 'status', gpuIndex, hashrate (H/s), accepted, rejected, gpu, coin: 'quantus' }
//                { type: 'connected', gpuIndex, endpoint, gpu }
//                { type: 'connect-failed', reason, dns }
//   'authorized' { host, address, family }
//   'stopped'    code: 1 when it gave up on its own, else 0
//   'error'      Error: why it gave up
//
// settings: { address, worker, endpoint, cards } where cards is the core's
// device indices to mine on (null: every card the core lists).
//
// Each card loops: take the pool's current job, scan a batch of nonces on its
// core, check every hit on the CPU against the full hash, submit the shares.
// A batch is sized to take about BATCH_MS, so a new job waits at most that
// long. Hits from a batch whose job was replaced meanwhile are dropped: the
// pool would call them stale. A core that reports a nonce that is not a hit
// BAD_HIT_LIMIT times is taken off mining.

const RECONNECT_MS = 5000;
// After the pool refuses the login: most often a wrong address.
const LOGIN_RETRY_MS = 60 * 1000;
const KEEPALIVE_MS = 60 * 1000;
const SILENCE_MS = 5 * 60 * 1000;
// A block every 12 to 15 seconds, and a job with each: three minutes without
// one is a pool that has stopped following the chain.
const JOB_SILENCE_MS = 3 * 60 * 1000;
const WATCHDOG_MS = 30 * 1000;
const STATUS_MS = 1000;
const BATCH_MS = 100;
const FIRST_BATCH = 256;
const MAX_BATCH = 2 ** 32;
const MAX_HITS = 256;
const BAD_HIT_LIMIT = 3;
const POOL_GRACE_MS = 30 * 1000;
const SESSION_LOST = /unauthenticated|banned|invalid job id|unknown session|not logged in/i;

function errText(e) {
  return (e && e.message) || String(e);
}

class QuantusEngine extends EventEmitter {
  constructor({ connect, factory, agent, now, random, timing } = {}) {
    super();
    this.connect = connect;
    this.factory = factory || null;
    this.agent = agent || 'llmjob-earn';
    this.now = now || Date.now;
    this.random = random || crypto.randomBytes;
    const t = timing || {};
    const pick = (k, d) => (t[k] == null ? d : t[k]);
    this.t = {
      reconnect: pick('reconnect', RECONNECT_MS),
      loginRetry: pick('loginRetry', LOGIN_RETRY_MS),
      keepalive: pick('keepalive', KEEPALIVE_MS),
      silence: pick('silence', SILENCE_MS),
      jobSilence: pick('jobSilence', JOB_SILENCE_MS),
      watchdog: pick('watchdog', WATCHDOG_MS),
      status: pick('status', STATUS_MS),
      batchMs: pick('batchMs', BATCH_MS),
      firstBatch: pick('firstBatch', FIRST_BATCH),
    };
    this.running = false;
    this.cards = [];
    this._reset();
  }

  _reset() {
    this.sock = null;
    this.buf = '';
    this.sessionId = null;
    this.authorized = false;
    this.closingAuthorized = false;
    this.lostAt = null;
    this.loginRefused = false;
    this.poolError = null;
    this.poolFails = 0;
    this.lastSockError = null;
    this.job = null;
    this.sent = new Set();
    this.submitId = 10;
    this.pending = new Map();      // submit id -> card
    this.keepaliveIds = new Set();
    this.lastRx = 0;
    this.lastJobRx = 0;
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
    this.emit('log', { level, line: 'quantus: ' + line });
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
    if (!this.factory) return fail('there is no Quantus core in this build');
    const indices = Array.isArray(settings.cards) ? settings.cards : this.factory.devices.map((d) => d.index);
    // A random lane per run and card: two runs, or two cards, never send the
    // same nonce for the same job.
    const runLane = this.random(6);
    this.cards = [];
    for (const index of indices) {
      let core;
      try {
        core = this.factory.createCore(index);
      } catch (e) {
        this._log('error', 'card ' + index + ' would not open: ' + errText(e));
        continue;
      }
      const lane = Buffer.alloc(8);
      runLane.copy(lane, 0);
      lane.writeUInt16BE(index & 0xffff, 6);
      const name = (core.device && core.device.name) || 'card ' + index;
      this.cards.push({
        index, core, name, lane, tail: Buffer.concat([Buffer.alloc(16), lane]), counter: 0n, batch: this.t.firstBatch, rate: 0,
        accepted: 0, rejected: 0, bad: 0, dead: false,
      });
      this._log('info', 'card ' + index + ': ' + name);
    }
    if (!this.cards.length) return fail('no card would open a Quantus core');
    this.running = true;
    const [host, port] = String(settings.endpoint || '').split(':');
    this.host = host;
    this.port = Number(port);
    this._log('info', 'mining Quantus for ' + shortenQuantusAddress(settings.address) + ' on ' + this.cards.length
      + ' card' + (this.cards.length === 1 ? '' : 's') + ' · ' + settings.endpoint);
    this._openSocket();
    this._timer(() => this._checkSilence(), this.t.watchdog, true);
    this._timer(() => this._sendKeepalive(), this.t.keepalive, true);
    this._timer(() => { for (const c of this.cards) this._status(c); }, this.t.status, true);
    // A core that answers a scan with something that is not a result takes
    // its card off mining, rather than ending the process.
    for (const c of this.cards) this._run(c).catch((e) => this._cardFailed(c, 'it answered a scan with ' + errText(e)));
    return true;
  }

  // ── the cards ────────────────────────────────────────────────────────────

  // Resolves when there may be new work: a job, a stop, a lost connection.
  _waitForWork() {
    return new Promise((resolve) => this.waiters.push(resolve));
  }

  _wake() {
    const w = this.waiters;
    this.waiters = [];
    for (const resolve of w) resolve();
  }

  async _run(card) {
    while (this.running) {
      const job = this.authorized ? this.job : null;
      if (!job) {
        await this._waitForWork();
        continue;
      }
      const count = card.batch;
      const t0 = this.now();
      let res;
      try {
        res = await card.core.scan({
          midstate: job.midstate, tail: card.tail,
          start: u64Bytes(card.counter), count, targetHigh: job.targetHigh, maxHits: MAX_HITS,
        });
      } catch (e) {
        if (!this.running) return;
        this._cardFailed(card, 'its core failed: ' + errText(e));
        return;
      }
      if (!this.running) return;
      card.counter += BigInt(count);
      this._timed(card, count, Math.max(this.now() - t0, 1));
      if (res.overflow) this._log('warn', 'card ' + card.index + ' found more than ' + MAX_HITS + ' hits in one batch');
      // The pool moved on while this batch ran: its hits would be stale.
      if (this.job !== job) continue;
      for (const counter of hitList(res)) {
        this._check(card, job, counter);
        if (card.dead) return;
      }
    }
  }

  // A batch's speed: the card's rate, and the next batch's size.
  _timed(card, count, ms) {
    const rate = count * 1000 / ms;
    card.rate = card.rate ? card.rate * 0.7 + rate * 0.3 : rate;
    const next = Math.round(count * this.t.batchMs / ms);
    card.batch = Math.max(1, Math.min(MAX_BATCH, count * 16, next));
  }

  // A hit, checked on the full reference hash before anything is sent.
  _check(card, job, counter) {
    const nonce = q.buildNonce(job.extranonce, card.lane, counter);
    const hash = q.nonceHash(job.miningHash, nonce);
    if (!q.meetsTarget(hash, job.target)) {
      // A hash whose top 64 bits tie the target's can still lose on the rest.
      if (q.toBig(hash) >> 448n === q.targetHigh(job.target)) return;
      card.bad++;
      this._log('warn', 'card ' + card.index + ' reported a nonce that is not a hit (' + card.bad + ' of ' + BAD_HIT_LIMIT + ')');
      if (card.bad >= BAD_HIT_LIMIT) this._cardFailed(card, 'its core keeps reporting nonces that are not hits');
      return;
    }
    this._submit(card, job, nonce.toString('hex'));
  }

  _cardFailed(card, why) {
    card.dead = true;
    card.rate = 0;
    try { card.core.close(); } catch (e) { /* already gone */ }
    this._log('error', 'card ' + card.index + ' stopped: ' + why);
    this._status(card);
    if (this.cards.every((c) => c.dead)) {
      const message = 'every card stopped; the Quantus miner gives up';
      this._log('error', message);
      this.emit('error', new Error(message));
      this.stop(true);
    }
  }

  _status(card) {
    this.emit('event', {
      type: 'status', gpuIndex: card.index, hashrate: this.poolConnected() && !card.dead ? card.rate : 0,
      accepted: card.accepted, rejected: card.rejected, gpu: card.name, coin: 'quantus',
    });
  }

  // ── the pool ─────────────────────────────────────────────────────────────

  _openSocket() {
    let sock;
    try {
      sock = this.connect(this.host, this.port);
    } catch (e) {
      this._log('error', 'pool connection failed: ' + errText(e));
      this._timer(() => this._openSocket(), this.t.reconnect);
      return;
    }
    this.sock = sock;
    this.buf = '';
    this.lastRx = this.now();
    this.lastJobRx = this.lastRx;
    sock.on('connect', () => {
      this._log('info', 'connecting to ' + this.host + ':' + this.port + ' · worker ' + (this.settings.worker || 'x'));
      sock.write(stratum.encode(stratum.buildLogin({
        address: this.settings.address, worker: this.settings.worker, agent: this.agent, host: this.host,
      })));
    });
    sock.on('data', (chunk) => {
      if (this.sock !== sock) return;
      this.lastRx = this.now();
      this.buf += String(chunk);
      let i;
      while (this.sock === sock && (i = this.buf.indexOf('\n')) >= 0) {
        const line = this.buf.slice(0, i);
        this.buf = this.buf.slice(i + 1);
        if (line.trim()) this._onPoolMessage(stratum.parseMessage(line, this.keepaliveIds));
      }
    });
    sock.on('error', (err) => {
      this.poolError = 'can\'t reach the Quantus pool: ' + err.message;
      if (this.poolFails === 0 || err.message !== this.lastSockError) this._log('error', 'pool socket error: ' + err.message);
      this.lastSockError = err.message;
      this.emit('event', {
        type: 'connect-failed', reason: err.message, dns: /ENOTFOUND|EAI_AGAIN|getaddrinfo/i.test(String(err.message)),
      });
    });
    sock.on('close', () => {
      if (this.sock !== sock) return;
      this.sock = null;
      const wasIn = this.authorized || this.closingAuthorized;
      this.closingAuthorized = false;
      if (wasIn) this.lostAt = this.now();
      else {
        this.poolFails++;
        if (!this.poolError) this.poolError = 'the Quantus pool closed the connection before the login';
      }
      this.authorized = false;
      this.sessionId = null;
      // The next session's jobs are its own.
      this.job = null;
      this.pending.clear();
      this.keepaliveIds.clear();
      const refused = this.loginRefused;
      const wait = refused ? this.t.loginRetry : this.t.reconnect;
      this.loginRefused = false;
      if (this.poolFails <= 1) this._log('info', 'pool connection closed; reconnecting in ' + Math.round(wait / 1000) + ' s');
      this._timer(() => this._openSocket(), wait);
    });
  }

  _closeSocket() {
    this.closingAuthorized = this.authorized;
    this.authorized = false;
    this.sock.destroy();
  }

  _setJob(m) {
    this.lastJobRx = this.now();
    // The job the cards already have: sending it again would restart nothing.
    const key = [m.jobId, m.miningHash.toString('hex'), m.target.toString(16), m.extranonce.toString('hex')].join(':');
    if (this.job && this.job.key === key) return;
    if (!this.job || this.job.jobId !== m.jobId) this.sent.clear();
    const high = Buffer.alloc(32);
    m.extranonce.copy(high, 0);
    const job = Object.assign({}, m, {
      key,
      midstate: q.midstateBytes(q.midstate(m.miningHash, high)),
      targetHigh: u64Bytes(q.targetHigh(m.target)),
    });
    if (!this.job || this.job.difficulty !== job.difficulty) this._log('info', 'pool difficulty ' + job.difficulty);
    this.job = job;
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
      case 'login-ok':
        this.sessionId = m.sessionId;
        this.authorized = true;
        this.lostAt = null;
        this.poolError = null;
        this.poolFails = 0;
        this.lastSockError = null;
        this.emit('authorized', { host: this.host, address: this.sock.remoteAddress, family: this.sock.remoteFamily });
        this._log('info', 'logged in to the pool');
        if (m.job) this._onPoolMessage(m.job);
        break;
      case 'login-fail':
        this.loginRefused = true;
        this.poolError = 'the Quantus pool refused the login: ' + stratum.errorText(m.error);
        this._log('error', 'the pool refused the login: ' + stratum.errorText(m.error));
        this._closeSocket();
        break;
      case 'job':
        this._setJob(m);
        break;
      case 'bad-job':
        this._log('error', 'ignoring a job this miner cannot mine' + (m.jobId ? ' (' + m.jobId + ')' : '') + ': ' + m.reason);
        this.job = null;
        break;
      case 'submit-accepted': {
        const card = this.pending.get(m.id);
        if (card) {
          this.pending.delete(m.id);
          card.accepted++;
          this._log('info', 'QTC share accepted (card ' + card.index + ')');
          this._status(card);
        }
        break;
      }
      case 'submit-rejected': {
        const card = this.pending.get(m.id);
        if (card) {
          this.pending.delete(m.id);
          card.rejected++;
          this._log('error', 'QTC share rejected (card ' + card.index + '): ' + stratum.errorText(m.error));
          this._status(card);
        }
        this._checkSessionLost(m.error);
        break;
      }
      case 'keepalive':
        this.keepaliveIds.delete(m.id);
        this._checkSessionLost(m.error);
        break;
      case 'pool-error':
        this._log('error', 'pool: ' + stratum.errorText(m.error));
        this._checkSessionLost(m.error);
        break;
      case 'unknown':
      case 'unparseable':
        this._log('info', 'pool: ' + m.raw.slice(0, 200));
        break;
    }
  }

  // A share for the pool's current job, sent once. The caller has checked
  // that the job is still the pool's.
  _submit(card, job, nonce) {
    if (this.sent.has(nonce)) {
      this._log('warn', 'dropping a share already sent (job ' + job.jobId + ')');
      return;
    }
    this.sent.add(nonce);
    const id = this.submitId++;
    this.pending.set(id, card);
    this.sock.write(stratum.encode(stratum.buildSubmit(id, this.sessionId, { jobId: job.jobId, nonce })));
  }

  _checkSessionLost(error) {
    if (!error || !SESSION_LOST.test(error.message)) return;
    this._log('warn', 'the pool dropped this session (' + stratum.errorText(error) + '); logging in again');
    if (/banned/i.test(error.message)) this.loginRefused = true;
    this._closeSocket();
  }

  _sendKeepalive() {
    if (!this.authorized) return;
    const id = this.submitId++;
    this.keepaliveIds.add(id);
    if (this.keepaliveIds.size > 16) this.keepaliveIds.delete(this.keepaliveIds.values().next().value);
    this.sock.write(stratum.encode(stratum.buildKeepalive(id, this.sessionId)));
  }

  _checkSilence() {
    if (!this.sock) return;
    const now = this.now();
    const mins = (ms) => Math.round(ms / 60000) + ' minutes';
    if (now - this.lastRx >= this.t.silence) this._log('warn', 'no word from the pool in ' + mins(this.t.silence) + '; reconnecting');
    else if (now - this.lastJobRx >= this.t.jobSilence) this._log('warn', 'no new job from the pool in ' + mins(this.t.jobSilence) + '; reconnecting');
    else return;
    this._closeSocket();
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
  QuantusEngine, RECONNECT_MS, LOGIN_RETRY_MS, KEEPALIVE_MS, SILENCE_MS, JOB_SILENCE_MS, WATCHDOG_MS, STATUS_MS,
  BATCH_MS, FIRST_BATCH, MAX_BATCH, MAX_HITS, BAD_HIT_LIMIT, POOL_GRACE_MS,
};
