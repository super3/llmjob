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
    this.idleMs = opts.idleMs || 3000;     // poll cadence
    this.chunkChars = opts.chunkChars || 60; // flush a result chunk every N chars
    this.running = false;
    this.active = 0;
    this._timer = null;
  }

  activeJobs() { return this.active; }

  _sign(extra) {
    return signedBody(Object.assign({}, this.identity, { timestamp: this.now() }), extra);
  }

  start() {
    if (this.running) return;
    this.running = true;
    this._tick();
  }

  stop() {
    this.running = false;
    if (this._timer) { this.cancel(this._timer); this._timer = null; }
  }

  // One poll → run any assigned jobs → schedule the next poll. Never rejects; a
  // failure is emitted and the loop keeps going.
  _tick() {
    if (!this.running) return;
    this.pollOnce()
      .catch((e) => this.emit('error', e))
      .then(() => { if (this.running) this._timer = this.schedule(() => this._tick(), this.idleMs); });
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
    const chunksUrl = this.serverUrl + '/api/jobs/' + job.id + '/chunks';
    let idx = 0;
    let buf = '';
    let chain = Promise.resolve();
    const enqueueFlush = (isFinal) => {
      if (!buf) return; // only send chunks that carry content
      const content = buf;
      const i = idx++;
      buf = '';
      chain = chain.then(() => this.post(chunksUrl, this._sign({ chunkIndex: i, content, isFinal: !!isFinal })));
    };

    try {
      await this.runJob(jobToChatBody(job), {
        onDelta: (text) => { buf += text; if (buf.length >= this.chunkChars) enqueueFlush(false); },
      });
      enqueueFlush(true);
      await chain;
      await this.post(this.serverUrl + '/api/jobs/' + job.id + '/complete', this._sign({}));
      this.emit('done', { id: job.id });
    } catch (e) {
      await chain.catch(() => {});
      await this.post(this.serverUrl + '/api/jobs/' + job.id + '/fail', this._sign({ error: e.message }));
      this.emit('failed', { id: job.id, error: e.message });
    } finally {
      this.active--;
    }
  }
}

module.exports = { JobWorker };
