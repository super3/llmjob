'use strict';

const { EventEmitter } = require('events');
const stratum = require('../shared/xmr/stratum');
const workerProto = require('../shared/xmr/worker');
const { shortenXmrAddress } = require('../shared/xmr/address');

// The Monero CPU miner: the pool connection, and the RandomX worker process
// (earn/xmr-worker) that does the hashing. It has the start/stop/event surface
// PearlEngine has, so both shells drive it the same way:
//
//   start(settings) -> bool   stop()   isRunning()   poolConnected()   snapshot()
//   'log'        { level, line }
//   'event'      { type: 'status', hashrate (H/s), accepted, rejected, threads, state }
//                { type: 'connect-failed', reason, dns }
//   'authorized' { host, address, family }   (for the pool-address cache)
//   'stopped'    code: 1 when it gave up on its own, else 0
//   'error'      Error
//
// The worker runs as its own process, never inside this one. If it crashes, or
// an antivirus deletes or kills it, the CPU miner stops and is restarted, and
// the GPU miner in this process carries on. Everything IO is injected (connect
// -> socket, spawnWorker -> child process), so the tests run with neither.
//
// settings: { address, worker, endpoint, threads, priority, hugePages }
// priority is the worker's (idle, low or normal); without one it is
// worker.defaultPriority for this platform.

const RECONNECT_MS = 5000;
// After the pool refuses the login. It means a wrong address far more often
// than a pool hiccup, so it is not worth asking every 5 seconds.
const LOGIN_RETRY_MS = 60 * 1000;
const KEEPALIVE_MS = 60 * 1000;
// Nothing from the pool for this long, not even a new job (Monero has a block
// every 2 minutes): the connection is dead without having closed.
const SILENCE_MS = 5 * 60 * 1000;
const WATCHDOG_MS = 30 * 1000;
// A worker that stops on its own is started again after RESTART_MS, unless it
// stopped within STABLE_MS of each of its last RESTART_LIMIT starts.
const RESTART_MS = 30 * 1000;
const RESTART_LIMIT = 3;
const STABLE_MS = 10 * 60 * 1000;
// How long a worker gets to exit after "quit" before it is killed.
const KILL_MS = 5000;
const POOL_GRACE_MS = 30 * 1000;

const PAGES = {
  hugetlb: 'huge pages',
  thp: 'transparent huge pages',
  normal: 'no huge pages (hashrate is up to 30% lower without them)',
};

function errText(e) {
  return (e && e.message) || String(e);
}

class XmrEngine extends EventEmitter {
  constructor({ connect, spawnWorker, workerPath, agent, now, timing, platform } = {}) {
    super();
    this.connect = connect;
    this.spawnWorker = spawnWorker;
    this.workerPath = workerPath || null;
    this.agent = agent || 'llmjob-earn';
    this.now = now || Date.now;
    this.platform = platform || process.platform;
    const t = timing || {};
    this.t = {
      reconnect: t.reconnect == null ? RECONNECT_MS : t.reconnect,
      loginRetry: t.loginRetry == null ? LOGIN_RETRY_MS : t.loginRetry,
      keepalive: t.keepalive == null ? KEEPALIVE_MS : t.keepalive,
      silence: t.silence == null ? SILENCE_MS : t.silence,
      watchdog: t.watchdog == null ? WATCHDOG_MS : t.watchdog,
      restart: t.restart == null ? RESTART_MS : t.restart,
      kill: t.kill == null ? KILL_MS : t.kill,
    };
    this.running = false;
    this._reset();
  }

  _reset() {
    this.sock = null;
    this.buf = '';
    this.sessionId = null;
    this.authorized = false;
    this.lostAt = null;
    this.loginRefused = false;
    this.job = null;
    this.sentNonces = new Set();   // nonces submitted for this.job
    this.difficulty = null;
    this.submitId = 10;
    this.pending = new Map();      // submit id -> job id
    this.keepaliveIds = new Set();
    this.lastRx = 0;
    this.child = null;
    this.childBuf = '';
    this.state = 'stopped';
    this.hashrate = 0;
    this.accepted = 0;
    this.rejected = 0;
    this.lastShareMs = null;
    this.pages = null;
    this.startedAt = 0;
    this.timers = new Set();
    this.restart = { fails: 0, startedAt: 0 };
  }

  isRunning() { return this.running; }

  // Whether the pool has this rig's login, give or take a routine reconnect.
  poolConnected(now = this.now()) {
    if (!this.running) return false;
    if (this.authorized) return true;
    return this.lostAt != null && now - this.lostAt < POOL_GRACE_MS;
  }

  // The numbers the UI and the stats file show.
  snapshot(now = this.now()) {
    return {
      state: this.running ? this.state : 'stopped',
      hashrate: this.running ? this.hashrate : 0,
      accepted: this.accepted,
      rejected: this.rejected,
      threads: this.settings ? this.settings.threads : 0,
      pages: this.pages,
      difficulty: this.difficulty,
      lastShareMs: this.lastShareMs,
      uptimeSec: this.running ? Math.max(0, Math.floor((now - this.startedAt) / 1000)) : 0,
    };
  }

  _log(level, line) {
    this.emit('log', { level, line: 'cpu: ' + line });
  }

  _status() {
    const s = this.snapshot();
    this.emit('event', {
      type: 'status', hashrate: s.hashrate, accepted: s.accepted, rejected: s.rejected,
      threads: s.threads, state: s.state,
    });
  }

  _setState(state) {
    this.state = state;
    this._status();
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
    if (!this.settings.priority) this.settings.priority = workerProto.defaultPriority(this.platform);
    if (!this.workerPath || !this.spawnWorker) {
      this.emit('error', new Error('the CPU miner (llmjob-xmr-worker) is not installed beside the app.'
        + ' An antivirus may have removed it; reinstalling puts it back.'));
      return false;
    }
    this.running = true;
    this.startedAt = this.now();
    const [host, port] = String(settings.endpoint || '').split(':');
    this.host = host;
    this.port = Number(port);
    this._log('info', 'mining Monero for ' + shortenXmrAddress(settings.address) + ' on ' + settings.threads
      + ' CPU thread' + (settings.threads === 1 ? '' : 's') + ' at ' + this.settings.priority + ' priority · '
      + settings.endpoint);
    if (!this._startWorker()) return false;
    this._openSocket();
    this._timer(() => this._checkSilence(), this.t.watchdog, true);
    this._timer(() => this._sendKeepalive(), this.t.keepalive, true);
    return true;
  }

  // ── the worker ───────────────────────────────────────────────────────────

  _startWorker() {
    const args = workerProto.workerArgs(this.settings);
    let child;
    try {
      child = this.spawnWorker(this.workerPath, args);
    } catch (e) {
      this._workerFailed('could not start ' + this.workerPath + ': ' + errText(e), false);
      return false;
    }
    this.child = child;
    this.childBuf = '';
    this.restart.startedAt = this.now();
    this.hashrate = 0;
    this._setState('starting');
    child.stdout.on('data', (chunk) => {
      if (this.child !== child) return;
      this.childBuf += String(chunk);
      let i;
      // Re-checked per line: a line can end this worker, and the rest of the
      // chunk is then from a process that is gone.
      while (this.child === child && (i = this.childBuf.indexOf('\n')) >= 0) {
        const line = this.childBuf.slice(0, i);
        this.childBuf = this.childBuf.slice(i + 1);
        this._onWorkerLine(workerProto.parseWorkerLine(line));
      }
    });
    child.stderr.on('data', (chunk) => {
      for (const line of String(chunk).split('\n')) if (line.trim()) this._log('warn', line.trim());
    });
    // A missing or unrunnable file is reported here, after spawn returned.
    child.on('error', (e) => {
      if (this.child !== child) return;
      this.child = null;
      const missing = e && e.code === 'ENOENT';
      this._workerFailed(missing
        ? 'the CPU miner is missing (' + this.workerPath + '). An antivirus may have removed it; reinstalling puts it back.'
        : 'the CPU miner could not run: ' + errText(e), !missing);
    });
    child.on('exit', (code, signal) => {
      if (this.child !== child) return;
      this.child = null;
      const how = signal ? 'signal ' + signal : 'exit code ' + code;
      const hint = signal === 'SIGKILL' ? ' (killed: out of memory, or an antivirus)' : '';
      this._workerFailed('the CPU miner stopped (' + how + ')' + hint, true);
    });
    // stdin breaks when the worker dies; the exit handler says so.
    child.stdin.on('error', () => {});
    // The pool won't send this job again. jobCommand gives it a new start
    // nonce, so the new worker doesn't find the old one's shares again.
    if (this.job) this._send(workerProto.jobCommand(this.job));
    return true;
  }

  _send(line) {
    const c = this.child;
    if (!c || c.stdin.destroyed) return;
    try { c.stdin.write(line); } catch (e) { /* it is exiting; the exit handler reports it */ }
  }

  _onWorkerLine(m) {
    switch (m.kind) {
      case 'hello':
        if (m.protocol !== workerProto.PROTOCOL) {
          this._killWorker();
          this._workerFailed('this llmjob-xmr-worker speaks protocol ' + m.protocol + ', the app speaks '
            + workerProto.PROTOCOL + ': it is from another release. Reinstall to get a matching one.', false);
          return;
        }
        this._log('info', 'RandomX worker ' + m.version + ' (RandomX ' + m.randomx + ') · '
          + (m.jit ? 'JIT' : 'no JIT') + ' · ' + (m.hardAes ? 'hardware AES' : 'software AES'));
        break;
      case 'selftest':
        if (!m.ok) {
          // The same build on the same CPU gives the same wrong answer every
          // time, so this is not worth a restart.
          this._killWorker();
          this._workerFailed('RandomX self-test failed, so nothing was mined: ' + m.message, false);
          return;
        }
        this._log('info', 'RandomX self-test passed (' + m.ms + ' ms)');
        break;
      case 'dataset':
        if (m.state === 'building') {
          this._setState('dataset');
          this._log('info', 'building the 2080 MiB RandomX dataset on ' + m.threads + ' thread'
            + (m.threads === 1 ? '' : 's') + ' (about a minute; again when Monero changes its seed, every 2048 blocks)');
        } else {
          this.pages = m.pages;
          this._log('info', 'dataset ready in ' + (Math.round(m.ms / 100) / 10) + ' s · ' + (PAGES[m.pages] || m.pages));
          this._setState('mining');
        }
        break;
      case 'share':
        this._submit(m);
        break;
      case 'hashrate':
        this.hashrate = m.hs;
        this._status();
        break;
      case 'log':
        this._log(m.level, m.message);
        break;
      case 'error':
        // Not every fatal error ends the worker on its own (a dataset that
        // fails its check leaves it waiting), so a fatal one ends it here.
        if (m.fatal) {
          this._killWorker();
          this._workerFailed(m.message, true);
          return;
        }
        this._log('error', m.message);
        break;
      case 'unparseable':
      case 'unknown':
        this._log('info', 'worker: ' + m.raw.slice(0, 200));
        break;
      // 'empty' and 'bench' need nothing.
    }
  }

  // Only called while handling a line from the current worker.
  _killWorker() {
    const c = this.child;
    this.child = null;
    try { c.kill('SIGKILL'); } catch (e) { /* already gone */ }
  }

  // The worker is gone. Start it again in a while, unless it keeps failing or
  // retrying can't help, in which case the CPU miner stops.
  _workerFailed(message, retry) {
    this.hashrate = 0;
    this.pages = null;
    this._log('error', message);
    if (retry) {
      if (this.now() - this.restart.startedAt >= STABLE_MS) this.restart.fails = 0;
      if (this.restart.fails < RESTART_LIMIT) {
        this.restart.fails++;
        this._setState('restarting');
        this._log('info', 'restarting the CPU miner in ' + Math.round(this.t.restart / 1000) + ' s');
        // stop() clears this timer, so it only fires while still running.
        this._timer(() => this._startWorker(), this.t.restart);
        return;
      }
      this._log('error', 'leaving the CPU miner off: it stopped within ' + STABLE_MS / 60000
        + ' minutes of each of ' + RESTART_LIMIT + ' restarts');
    }
    this.emit('error', new Error(message));
    this.stop(true);
  }

  // ── the pool ─────────────────────────────────────────────────────────────

  _openSocket() {
    let sock;
    try {
      sock = this.connect(this.host, this.port);
    } catch (e) {
      this._log('error', 'pool connection failed: ' + errText(e));
      this._scheduleReconnect(this.t.reconnect);
      return;
    }
    this.sock = sock;
    this.buf = '';
    this.lastRx = this.now();
    sock.on('connect', () => {
      this._log('info', 'connecting to ' + this.host + ':' + this.port + ' · worker ' + (this.settings.worker || 'x'));
      sock.write(stratum.encode(stratum.buildLogin({
        address: this.settings.address, worker: this.settings.worker, agent: this.agent,
      })));
    });
    sock.on('data', (chunk) => {
      if (this.sock !== sock) return;
      this.lastRx = this.now();
      this.buf += String(chunk);
      let i;
      while ((i = this.buf.indexOf('\n')) >= 0) {
        const line = this.buf.slice(0, i);
        this.buf = this.buf.slice(i + 1);
        if (line.trim()) this._onPoolMessage(stratum.parseMessage(line, this.keepaliveIds));
      }
    });
    sock.on('error', (err) => {
      this._log('error', 'pool socket error: ' + err.message);
      this.emit('event', {
        type: 'connect-failed', reason: err.message,
        dns: /ENOTFOUND|EAI_AGAIN|getaddrinfo/i.test(String(err.message)),
      });
    });
    sock.on('close', () => {
      if (this.sock !== sock) return;
      this.sock = null;
      if (this.authorized) this.lostAt = this.now();
      this.authorized = false;
      this.sessionId = null;
      this.pending.clear();
      this.keepaliveIds.clear();
      // stop() lets go of the socket before closing it, so this is a close
      // nobody asked for.
      const wait = this.loginRefused ? this.t.loginRetry : this.t.reconnect;
      this.loginRefused = false;
      this._log('info', 'pool connection closed; reconnecting in ' + Math.round(wait / 1000) + ' s');
      this._scheduleReconnect(wait);
    });
  }

  // stop() clears this timer, so it only fires while still running.
  _scheduleReconnect(ms) {
    this._timer(() => this._openSocket(), ms);
  }

  _onPoolMessage(m) {
    switch (m.kind) {
      case 'login-ok':
        this.sessionId = m.sessionId;
        this.authorized = true;
        this.lostAt = null;
        // The reply came in on this.sock, so that is the address that worked.
        this.emit('authorized', { host: this.host, address: this.sock.remoteAddress, family: this.sock.remoteFamily });
        this._log('info', 'logged in to the pool');
        if (m.job) this._onPoolMessage(m.job);
        break;
      case 'login-fail':
        this.loginRefused = true;
        this._log('error', 'the pool refused the login: ' + stratum.errorText(m.error));
        this.sock.destroy();
        break;
      case 'job':
        // The job the worker already has. Sending it again would restart its
        // search for no gain.
        if (this.job && m.jobId === this.job.jobId && m.blob === this.job.blob && m.target === this.job.target) break;
        if (!this.job || m.jobId !== this.job.jobId) this.sentNonces.clear();
        this.job = m;
        if (m.difficulty !== this.difficulty) {
          this.difficulty = m.difficulty;
          this._log('info', 'pool difficulty ' + m.difficulty);
        }
        this._send(workerProto.jobCommand(m));
        break;
      case 'bad-job':
        this._log('error', 'ignoring a job the CPU miner cannot mine' + (m.jobId ? ' (' + m.jobId + ')' : '') + ': ' + m.reason);
        break;
      case 'submit-accepted':
        if (this.pending.delete(m.id)) {
          this.accepted++;
          this.lastShareMs = this.now();
          this._log('info', 'XMR share accepted');
          this._status();
        }
        break;
      case 'submit-rejected':
        if (this.pending.delete(m.id)) {
          this.rejected++;
          this._log('error', 'XMR share rejected: ' + stratum.errorText(m.error));
          this._status();
        }
        break;
      case 'keepalive':
        this.keepaliveIds.delete(m.id);
        break;
      case 'pool-error':
        this._log('error', 'pool: ' + stratum.errorText(m.error));
        break;
      case 'unknown':
      case 'unparseable':
        this._log('info', 'pool: ' + m.raw.slice(0, 200));
        break;
    }
  }

  // A share the worker found and checked. Only shares of the job the pool is
  // on now go out: HeroMiners holds back pay from a miner whose stale shares
  // pass 2-5%, and a share of the last job is usually stale. A share sent once
  // never goes again: cryptonote-nodejs-pool, which HeroMiners runs, can ban
  // the IP for a duplicate.
  _submit(share) {
    if (!this.authorized || !this.job || share.jobId !== this.job.jobId) {
      this._log('info', 'dropping a share for a job the pool has moved on from');
      return;
    }
    if (this.sentNonces.has(share.nonce)) {
      this._log('warn', 'dropping a share already sent (job ' + share.jobId + ', nonce ' + share.nonce + ')');
      return;
    }
    this.sentNonces.add(share.nonce);
    const id = this.submitId++;
    this.pending.set(id, share.jobId);
    this.sock.write(stratum.encode(stratum.buildSubmit(id, this.sessionId, share)));
  }

  _sendKeepalive() {
    if (!this.authorized) return;
    const id = this.submitId++;
    this.keepaliveIds.add(id);
    // A pool that never answers keepalives must not grow this set forever.
    if (this.keepaliveIds.size > 16) this.keepaliveIds.delete(this.keepaliveIds.values().next().value);
    this.sock.write(stratum.encode(stratum.buildKeepalive(id, this.sessionId)));
  }

  _checkSilence() {
    if (!this.sock || this.now() - this.lastRx < this.t.silence) return;
    this._log('warn', 'no word from the pool in ' + Math.round(this.t.silence / 60000) + ' minutes; reconnecting');
    this.sock.destroy();
  }

  // `failed` marks the stop the engine makes on its own, which 'stopped' carries
  // as code 1 (see PearlEngine).
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
    const c = this.child;
    this.child = null;
    if (c) {
      // Ask first: the worker frees its 2 GiB and exits in milliseconds. Kill
      // it if it hasn't gone in KILL_MS.
      try { c.stdin.write(workerProto.quitCommand()); c.stdin.end(); } catch (e) { /* already exiting */ }
      const t = setTimeout(() => { try { c.kill('SIGKILL'); } catch (e) { /* gone */ } }, this.t.kill);
      t.unref();
      c.once('exit', () => clearTimeout(t));
    }
    this.hashrate = 0;
    this.state = failed ? 'failed' : 'stopped';
    this._status();
    this.emit('stopped', failed ? 1 : 0);
    return true;
  }
}

module.exports = {
  XmrEngine, RECONNECT_MS, LOGIN_RETRY_MS, KEEPALIVE_MS, SILENCE_MS, RESTART_MS, RESTART_LIMIT, STABLE_MS,
  KILL_MS, POOL_GRACE_MS,
};
