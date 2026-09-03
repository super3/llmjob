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
    // What this node has loaded, as a thunk rather than a value: the fleet can be
    // stopped and restarted at a different tier under a worker that outlives it,
    // and metrics.model must follow the model that actually ran the job. Omitted
    // by a caller that has not been taught about tiers, which then gets the fleet
    // default from jobToChatBody.
    this.servingModel = opts.servingModel || (() => null);
    this.now = opts.now || Date.now;
    this.schedule = opts.schedule || ((fn, ms) => { const t = setTimeout(fn, ms); t.unref(); return t; });
    this.cancel = opts.cancel || clearTimeout;
    // Gap between a completed poll and re-opening. Small because the WAIT now
    // happens inside the request: the server holds an empty poll open, so this is
    // only breathing room between round trips, not a dispatch delay.
    this.pollMinMs = opts.pollMinMs || 250;
    this.pollMaxMs = opts.pollMaxMs || 60000;  // backoff ceiling, errors only
    // How long an empty poll must have taken for us to believe the server HELD
    // it. A holding server returns either with work or at its hold deadline
    // (~25s); one that does not hold answers empty in milliseconds. Anything
    // above a second is a hold by any reasonable network.
    this.holdHintMs = opts.holdHintMs || 1000;
    this.heartbeatMs = opts.heartbeatMs || 30000; // per-job lock renewal cadence
    this.chunkChars = opts.chunkChars || 60; // flush a result chunk every N chars…
    this.flushMs = opts.flushMs || 1000;     // …or at least this often while text flows
    this.running = false;
    this.active = 0;
    this._timer = null;
    this._delay = this.pollMinMs;
  }

  activeJobs() { return this.active; }

  _sign(extra) {
    return signedBody(Object.assign({}, this.identity, { timestamp: this.now() }), extra);
  }

  _ok(res) { return !!(res && res.status >= 200 && res.status < 300); }

  start() {
    if (this.running) return;
    this.running = true;
    this._delay = this.pollMinMs;
    this._tick();
  }

  stop() {
    this.running = false;
    if (this._timer) { this.cancel(this._timer); this._timer = null; }
  }

  // One poll → run any assigned jobs → re-open. Never rejects; a failure is
  // emitted and the loop keeps going.
  //
  // A SUCCESSFUL poll re-opens immediately, because the server now holds an empty
  // poll open rather than answering nothing: the request itself is the wait, so
  // returning means either there is work or the hold expired, and in both cases
  // the right move is to ask again at once. The old exponential backoff -- 5s
  // doubling to 60s on empty polls -- put an idle rig on a 20-40s rung, so a job
  // waited that long just to be ASKED for. That was the largest term in
  // time-to-first-token, larger than loading the model.
  //
  // ERRORS still back off. A server that is down or rejecting must not be
  // hammered once per round trip, and that is the one case where an empty return
  // is not a completed wait.
  _tick() {
    if (!this.running) return;
    const startedAt = this.now();
    this.pollOnce()
      .then((count) => {
        // Did the server HOLD this poll, or answer empty at once?
        //
        // This matters because a node can be newer than the server it talks to.
        // Against a holding server an empty return means "nothing for 25s", and
        // re-opening immediately is right. Against one that does not hold, an
        // empty return means nothing at all -- and re-opening immediately would
        // be four polls a second, forever, on every idle node in the fleet.
        // Elapsed time tells them apart with no version negotiation.
        const held = (this.now() - startedAt) >= this.holdHintMs;
        if (count > 0 || held) this._delay = this.pollMinMs;
        else this._delay = Math.min(Math.max(this._delay, this.pollMinMs) * 2, this.pollMaxMs);
      })
      .catch((e) => {
        this._delay = Math.min(Math.max(this._delay, this.pollMinMs) * 2, this.pollMaxMs);
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
    const chatBody = jobToChatBody(job, this.servingModel());
    // Fences this attempt. Every worker on a rig signs as the same node id (one
    // per GPU that fits the model, and the GUI and CLI share one node.json), so
    // the server cannot tell our writes from a sibling's on the node id alone.
    // Echoing the token the assignment came with lets it reject us — and not the
    // sibling — if this job was requeued and picked up elsewhere while we ran.
    // Undefined against a server too old to issue one, which omits it entirely.
    const lock = job.lockToken ? { lockToken: job.lockToken } : {};

    // Keep the server's job lock alive for the whole run: immediately (which
    // also flips the job to 'running' so callers see streamed partials), then
    // every heartbeatMs. Best-effort — a missed beat is caught by the next.
    let hbTimer = null;
    const beat = () => {
      this.post(base + '/heartbeat', this._sign({ ...lock })).catch(() => {});
      hbTimer = this.schedule(beat, this.heartbeatMs);
    };
    beat();

    let idx = 0;
    let buf = '';
    let reasoning = '';
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
      const body = { ...lock, chunkIndex: i, content, isFinal: !!isFinal };
      // Reasoning isn't streamed to the caller (it isn't part of the answer), so
      // it rides on the final chunk — enough for the gateway to report it and to
      // explain an empty completion.
      if (isFinal && reasoning) body.reasoning = reasoning;
      if (metrics) body.metrics = metrics;
      chain = chain.then(() => this.post(base + '/chunks', this._sign(body))).then((res) => {
        if (!this._ok(res) && !chunkError) {
          chunkError = new Error('chunk ' + i + ' rejected (HTTP ' + ((res && res.status) || 0) + ')');
        }
      });
    };

    try {
      const outcome = await this.runJob(chatBody, {
        // `count` lets a batching stream report several tokens per callback.
        onDelta: (text, count) => {
          buf += text;
          tokens += Number.isFinite(count) ? count : 1;
          if (buf.length >= this.chunkChars || this.now() - lastFlushAt >= this.flushMs) enqueueFlush(false);
        },
        // A thinking model spends real tokens here before it writes a word of
        // the answer, so they count toward the total the caller is billed for.
        onReasoning: (text, count) => {
          reasoning += text;
          tokens += Number.isFinite(count) ? count : 1;
        },
      });
      const elapsedSeconds = Math.max(0.001, (this.now() - startedAt) / 1000);
      enqueueFlush(true, {
        totalTokens: tokens,
        tokensPerSecond: +(tokens / elapsedSeconds).toFixed(2),
        elapsedSeconds: +elapsedSeconds.toFixed(3),
        model: chatBody.model,
        finishReason: (outcome && outcome.finishReason) || 'stop',
      });
      await chain;
      if (chunkError) throw chunkError;
      const done = await this.post(base + '/complete', this._sign({ ...lock }));
      if (!this._ok(done)) {
        // The server refused the completion (lock lost, job re-queued/deleted);
        // don't pretend success, and don't POST /fail — our lock is gone anyway.
        this.emit('failed', { id: job.id, error: 'complete rejected (HTTP ' + ((done && done.status) || 0) + ')' });
      } else {
        this.emit('done', { id: job.id });
      }
    } catch (e) {
      await chain.catch(() => {});
      await this.post(base + '/fail', this._sign({ ...lock, error: e.message })).catch(() => {});
      this.emit('failed', { id: job.id, error: e.message });
    } finally {
      if (hbTimer) this.cancel(hbTimer);
      this.active--;
    }
  }
}

module.exports = { JobWorker };
