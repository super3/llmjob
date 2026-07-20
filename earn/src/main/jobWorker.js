'use strict';

const EventEmitter = require('events');
const { signedBody } = require('../shared/node');
const { jobToChatBody } = require('../shared/jobs');

// Polls the LLMJob server for inference jobs and runs them against the local
// model, streaming the result back in chunks. Every call is outbound (poll +
// chunk POSTs), so a node behind NAT / a provider network can serve the cluster
// with no inbound networking. All IO is injected — `post` (signed HTTP), `runJob`
// (drive the local model), `now`, and the scheduler — so it's unit-tested with
// fakes, no network or GPU. Mirrors the manager pattern used elsewhere.
//
// Protocol notes (must match server/src/services/jobService.js):
// - A heartbeat POST is what flips a job 'assigned'→'running' and renews its
//   10-minute lock, so one is sent immediately and then every heartbeatMs while
//   the job runs — without it, long jobs lose their lock and get re-executed.
// - Every chunk/complete POST's HTTP status is checked: a rejected chunk fails
//   the job instead of silently completing with missing content.
// - The final chunk always carries `isFinal` plus generation metrics
//   (totalTokens / tokensPerSecond / elapsedSeconds / model) for the job record.
class JobWorker extends EventEmitter {
  constructor(opts = {}) {
    super();
    this.identity = opts.identity;         // { nodeId, publicKey, secretKey }
    this.serverUrl = opts.serverUrl;
    this.post = opts.post;                 // (url, body) -> Promise<{ status, data }>
    this.runJob = opts.runJob;             // (chatBody, { onDelta }) -> Promise (rejects on error)
    this.now = opts.now || Date.now;
    this.schedule = opts.schedule || ((fn, ms) => { const t = setTimeout(fn, ms); t.unref(); return t; });
    this.cancel = opts.cancel || clearTimeout;
    this.idleMs = opts.idleMs || 5000;     // poll cadence right after activity
    this.maxIdleMs = opts.maxIdleMs || 60000; // backoff ceiling for empty/error polls
    this.heartbeatMs = opts.heartbeatMs || 30000; // per-job lock renewal cadence
    this.chunkChars = opts.chunkChars || 60; // flush a result chunk every N chars…
    this.flushMs = opts.flushMs || 1000;     // …or at least this often while text flows
    this.running = false;
    this.active = 0;
    this._timer = null;
    this._delay = this.idleMs;
  }

  activeJobs() { return this.active; }

  _sign(extra) {
    return signedBody(Object.assign({}, this.identity, { timestamp: this.now() }), extra);
  }

  _ok(res) { return !!(res && res.status >= 200 && res.status < 300); }

  start() {
    if (this.running) return;
    this.running = true;
    this._delay = this.idleMs;
    this._tick();
  }

  stop() {
    this.running = false;
    if (this._timer) { this.cancel(this._timer); this._timer = null; }
  }

  // One poll → run any assigned jobs → schedule the next poll. Never rejects; a
  // failure is emitted and the loop keeps going. Empty polls and errors back off
  // exponentially up to maxIdleMs so an idle fleet (or a down server) isn't
  // hammered; any assigned job snaps the cadence back to idleMs.
  _tick() {
    if (!this.running) return;
    this.pollOnce()
      .then((count) => {
        this._delay = count > 0 ? this.idleMs : Math.min(this._delay * 2, this.maxIdleMs);
      })
      .catch((e) => {
        this._delay = Math.min(this._delay * 2, this.maxIdleMs);
        try { this.emit('error', e); } catch (e2) { /* listener-less 'error' must not kill the loop */ }
      })
      .then(() => { if (this.running) this._timer = this.schedule(() => this._tick(), this._delay); });
  }

  // Ask the server for work and process whatever it assigns. Returns the count.
  async pollOnce() {
    const res = await this.post(this.serverUrl + '/api/jobs/poll', this._sign({ maxJobs: 1 }));
    const jobs = (res && res.data && res.data.jobs) || [];
    for (const job of jobs) await this.processJob(job);
    return jobs.length;
  }

  // Run one job against the local model, streaming chunks back in order. The
  // server assembles by chunk index, so ordered enqueue (not awaited per delta)
  // keeps generation from stalling on the network.
  async processJob(job) {
    this.active++;
    this.emit('job', { id: job.id, active: this.active });
    const base = this.serverUrl + '/api/jobs/' + job.id;
    const chatBody = jobToChatBody(job);

    // Keep the server's job lock alive for the whole run: immediately (which
    // also flips the job to 'running' so callers see streamed partials), then
    // every heartbeatMs. Best-effort — a missed beat is caught by the next.
    let hbTimer = null;
    const beat = () => {
      this.post(base + '/heartbeat', this._sign({})).catch(() => {});
      hbTimer = this.schedule(beat, this.heartbeatMs);
    };
    beat();

    let idx = 0;
    let buf = '';
    let chunkError = null;
    let chain = Promise.resolve();
    const startedAt = this.now();
    let tokens = 0;
    let lastFlushAt = startedAt;
    const enqueueFlush = (isFinal, metrics) => {
      if (!buf && !isFinal) return;
      const content = buf;
      const i = idx++;
      buf = '';
      lastFlushAt = this.now();
      const body = { chunkIndex: i, content, isFinal: !!isFinal };
      if (metrics) body.metrics = metrics;
      chain = chain.then(() => this.post(base + '/chunks', this._sign(body))).then((res) => {
        if (!this._ok(res) && !chunkError) {
          chunkError = new Error('chunk ' + i + ' rejected (HTTP ' + ((res && res.status) || 0) + ')');
        }
      });
    };

    try {
      await this.runJob(chatBody, {
        // `count` lets a batching stream report several tokens per callback.
        onDelta: (text, count) => {
          buf += text;
          tokens += Number.isFinite(count) ? count : 1;
          if (buf.length >= this.chunkChars || this.now() - lastFlushAt >= this.flushMs) enqueueFlush(false);
        },
      });
      const elapsedSeconds = Math.max(0.001, (this.now() - startedAt) / 1000);
      enqueueFlush(true, {
        totalTokens: tokens,
        tokensPerSecond: +(tokens / elapsedSeconds).toFixed(2),
        elapsedSeconds: +elapsedSeconds.toFixed(3),
        model: chatBody.model,
      });
      await chain;
      if (chunkError) throw chunkError;
      const done = await this.post(base + '/complete', this._sign({}));
      if (!this._ok(done)) {
        // The server refused the completion (lock lost, job re-queued/deleted);
        // don't pretend success, and don't POST /fail — our lock is gone anyway.
        this.emit('failed', { id: job.id, error: 'complete rejected (HTTP ' + ((done && done.status) || 0) + ')' });
      } else {
        this.emit('done', { id: job.id });
      }
    } catch (e) {
      await chain.catch(() => {});
      await this.post(base + '/fail', this._sign({ error: e.message })).catch(() => {});
      this.emit('failed', { id: job.id, error: e.message });
    } finally {
      if (hbTimer) this.cancel(hbTimer);
      this.active--;
    }
  }
}

module.exports = { JobWorker };
