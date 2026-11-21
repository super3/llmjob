'use strict';

// Runs rented container jobs from our server on this rig (issue #318).
//
// The rig asks the server for work on a timer, offering its cards unless the
// owner's LLM is answering a prompt, docker is missing, or the owner took the
// PC back in the last hour. When it is handed a job it stops mining on that
// card, checks the image (ours, pinned by digest, and signed when cosign is
// installed), pulls it, and runs it on that one card with no host mounts, not
// privileged, and capped CPU and RAM. It streams the logs back with a
// heartbeat every 15 s, and stops the container when the job time is up, the
// server says stop, the card runs too hot, or the owner takes the PC back.
// Then mining starts again on that card.
//
// Everything with side effects comes in as `deps` (see rentalHost.js for the
// real ones), so the whole lifecycle is tested with a fake docker.

const { EventEmitter } = require('events');
const R = require('../shared/rental');

const NEXT_POLL_AFTER_JOB_MS = 5000;
const LLM_WAIT_MS = 2 * 60 * 1000;
const LLM_WAIT_STEP_MS = 2000;
const STATUS_LOG_EVERY = 4;            // heartbeats, so about once a minute
const END_LOG_LINES = 500;             // the most the server takes in one call

const tail = (s) => String(s || '').trim().split(/\r?\n/).slice(-3).join(' | ').slice(0, 300);

class RentalRunner extends EventEmitter {
  // deps: { api, docker, cosign, cards, driver, pauseCard, resumeCard, llmBusy,
  // temps, takeBackRequested, clearTakeBack, log, platform, worker, caps,
  // registry?, requireSignature?, now?, setTimer?, clearTimer?, every?,
  // stopEvery?, sleep? }
  constructor(deps) {
    super();
    this.d = Object.assign({
      registry: R.REGISTRY,
      requireSignature: false,
      now: Date.now,
      setTimer: (fn, ms) => { const t = setTimeout(fn, ms); t.unref(); return t; },
      clearTimer: (t) => clearTimeout(t),
      every: (fn, ms) => { const t = setInterval(fn, ms); t.unref(); return t; },
      stopEvery: (t) => clearInterval(t),
      sleep: (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
    }, deps);
    this.job = null;
    this.stopped = true;
    this.dockerOk = false;
    this.holdUntil = 0;
    this.pollTimer = null;
    this.takeBackTimer = null;
    this.lastCards = [];
    this.unsignedWarned = false;
  }

  log(level, line) {
    this.d.log(level, 'rented containers: ' + line);
  }

  // Start asking for work. Resolves false where there is no runner (Windows,
  // macOS), after saying so once.
  async start() {
    if (!this.stopped) return true;
    const note = R.rentalUnavailable(this.d.platform);
    if (note) {
      this.d.log('info', note);
      return false;
    }
    this.stopped = false;
    this.dockerOk = await this.d.docker.available();
    if (this.stopped) return false;
    if (this.dockerOk) await this.d.docker.cleanup();
    else this.log('info', 'docker is not installed or not running, so this rig takes no container jobs');
    this.takeBackTimer = this.d.every(() => this._checkTakeBack(), R.TAKEBACK_CHECK_MS);
    this._schedule(0);
    return true;
  }

  // Stop asking for work, and end a running job at once (the owner stopped the
  // miner). Resolves once the job's end has been reported.
  stop() {
    this.stopped = true;
    if (this.pollTimer) { this.d.clearTimer(this.pollTimer); this.pollTimer = null; }
    if (this.takeBackTimer) { this.d.stopEvery(this.takeBackTimer); this.takeBackTimer = null; }
    if (!this.job) return Promise.resolve(false);
    return this._finish(this.job, 'taken_back', { detail: 'the owner stopped the miner', fast: true }).then(() => true);
  }

  // The owner takes the PC back: end the job now, and take no new one for an
  // hour. Resolves whether a job was running.
  takeBack() {
    this.holdUntil = this.d.now() + R.HOLD_AFTER_TAKEBACK_MS;
    if (!this.job) {
      this.log('info', 'the PC is yours; no container jobs for the next hour');
      return Promise.resolve(false);
    }
    return this._finish(this.job, 'taken_back', { detail: 'the owner took the PC back' }).then(() => {
      this.log('info', 'the PC is yours again; no container jobs for the next hour');
      return true;
    });
  }

  // Pause mining again on a rented card, after something else (prompt pause)
  // resumed it.
  reassert() {
    if (this.job && this.job.paused && !this.job.ending) this.d.pauseCard(this.job.card);
  }

  // The job on this rig, or null: { card, state, endsAt, jobId, name }.
  status() {
    const j = this.job;
    if (!j || j.ending) return null;
    return { card: j.card, state: j.state, endsAt: j.endsAt, jobId: j.jobId, name: j.name };
  }

  // "Rented: running a container · 42m left", or null.
  label() {
    return R.rentedLabel(this.status(), this.d.now());
  }

  // ── Polling ─────────────────────────────────────────────────────────────────

  _schedule(ms) {
    if (this.stopped) return;
    if (this.pollTimer) this.d.clearTimer(this.pollTimer);
    this.pollTimer = this.d.setTimer(() => { this.pollTimer = null; this.tick(); }, ms);
  }

  async tick() {
    if (this.stopped || this.job) return;
    let next = R.POLL_MS;
    try {
      const cards = await this.d.cards();
      this.lastCards = cards;
      // No card is offered while the owner's LLM is answering, without docker,
      // or for an hour after the owner took the PC back.
      const offer = this.dockerOk && this.d.now() >= this.holdUntil && !this.d.llmBusy();
      const res = await this.d.api.poll({
        worker: this.d.worker,
        platform: this.d.platform,
        docker: this.dockerOk,
        driver: await this.d.driver(),
        cards: cards.map((c) => Object.assign({}, c, { free: offer })),
      });
      if (res && Number(res.pollMs) > 0) next = Number(res.pollMs);
      if (res && res.job && !this.stopped) {
        this.runJob(res.job);
        return;
      }
    } catch (e) {
      this.log('warn', 'could not ask the server for work: ' + e.message);
    }
    this._schedule(next);
  }

  _checkTakeBack() {
    if (!this.d.takeBackRequested()) return;
    this.d.clearTakeBack();
    this.takeBack();
  }

  // ── A job ───────────────────────────────────────────────────────────────────

  // Run a job the server handed us. Resolves once the container is up (or the
  // job has ended); the job's end is _finish's.
  async runJob(offer) {
    const card = this.lastCards.find((c) => c.index === offer.card);
    const job = Object.assign({}, offer, {
      name: R.containerName(offer.runId),
      vendor: (card && card.vendor) || 'nvidia',
      state: 'pulling', paused: false, started: false, ending: false, endsAt: null, logs: [], beats: 0,
    });
    this.job = job;
    const problem = R.imageProblem(offer.image, this.d.registry) || R.envProblem(offer.env);
    if (problem) {
      this.log('warn', 'refused a job: ' + problem);
      return this._finish(job, 'refused', { detail: problem });
    }
    this.d.pauseCard(job.card);
    job.paused = true;
    this.log('info', 'GPU ' + job.card + ' is rented for ' + R.formatLeft(job.durationSec * 1000)
      + '; mining paused on that card. Run `llmjob-earn-cli take-back` to end it.');
    this.emit('change');
    job.hb = this.d.every(() => this._heartbeat(job), R.HEARTBEAT_MS);
    job.tempTimer = this.d.every(() => this._checkTemp(job), R.TEMP_CHECK_MS);
    try {
      const sig = await this._checkSignature(job.image);
      if (job.ending) return job.done;
      if (sig) return this._finish(job, 'refused', { detail: sig });
      const pull = await this.d.docker.run(['pull', job.image], {});
      if (job.ending) return job.done;
      if (pull.code !== 0) return this._finish(job, 'failed', { detail: 'docker pull failed: ' + tail(pull.stderr) });
      if (!(await this._waitLlmIdle(job))) {
        if (job.ending) return job.done;
        return this._finish(job, 'refused', { detail: "the owner's LLM stayed busy" });
      }
      await this.d.docker.run(['rm', '-f', job.name], {});
      const args = R.dockerRunArgs({
        name: job.name, image: job.image, env: job.env, card: job.card, vendor: job.vendor, caps: this.d.caps,
      });
      const run = await this.d.docker.run(args, job.env);
      if (run.code !== 0) {
        if (job.ending) return job.done;
        return this._finish(job, 'failed', { detail: 'docker run failed: ' + tail(run.stderr) });
      }
      // Ended while the container was starting: _finish didn't know about it.
      if (job.ending) {
        await this.d.docker.run(['rm', '-f', job.name], {});
        return job.done;
      }
      job.started = true;
      job.state = 'running';
      job.endsAt = this.d.now() + job.durationSec * 1000;
      job.follow = this.d.docker.logs(job.name, (line) => this._addLog(job, line));
      this.d.docker.wait(job.name).then(
        (code) => this._finish(job, 'exited', { exitCode: code }),
        (e) => this._finish(job, 'failed', { detail: 'docker wait failed: ' + e.message }),
      );
      job.endTimer = this.d.setTimer(() => this._finish(job, 'expired'), job.durationSec * 1000);
      this.log('info', 'GPU ' + job.card + ' · ' + this.label());
      this.emit('change');
      this._heartbeat(job);
    } catch (e) {
      return this._finish(job, 'failed', { detail: e.message });
    }
  }

  async _checkSignature(image) {
    if (await this.d.cosign.available()) {
      const v = await this.d.cosign.verify(image);
      return v.code === 0 ? null : 'signature check failed: ' + tail(v.stderr);
    }
    if (this.d.requireSignature) return 'cosign is not installed and LLMJOB_RENT_REQUIRE_SIGNATURE=1';
    if (!this.unsignedWarned) {
      this.unsignedWarned = true;
      this.log('warn', 'image signatures are not checked: cosign is not installed');
    }
    return null;
  }

  // Never start a container while the owner's LLM is answering: wait up to two
  // minutes for it to finish. Resolves whether it did.
  async _waitLlmIdle(job) {
    for (let waited = 0; waited <= LLM_WAIT_MS; waited += LLM_WAIT_STEP_MS) {
      if (job.ending) return false;
      if (!this.d.llmBusy()) return true;
      await this.d.sleep(LLM_WAIT_STEP_MS);
    }
    return false;
  }

  _addLog(job, line) {
    job.logs.push(String(line).slice(0, R.MAX_LINE_CHARS));
    if (job.logs.length > R.MAX_LOG_BUFFER) job.logs.splice(0, job.logs.length - R.MAX_LOG_BUFFER);
  }

  async _heartbeat(job) {
    if (job.ending) return;
    const logs = job.logs.splice(0, R.MAX_LOG_BATCH);
    job.beats++;
    if (job.state === 'running' && job.beats % STATUS_LOG_EVERY === 0) this.log('info', 'GPU ' + job.card + ' · ' + this.label());
    let res = null;
    try {
      res = await this.d.api.heartbeat(job.runId, { state: job.state, logs });
    } catch (e) {
      // Kept for the next one; the server allows three minutes of silence.
      job.logs.unshift(...logs);
      return;
    }
    if (res && res.action === 'stop') await this._finish(job, 'stopped', { detail: res.reason || 'the server said stop' });
  }

  async _checkTemp(job) {
    if (job.ending) return;
    const readings = await this.d.temps();
    const why = R.tempProblem(readings[job.card]);
    if (!why) return;
    this.log('warn', 'stopping the container on GPU ' + job.card + ': ' + why);
    await this._finish(job, 'temperature', { detail: why });
  }

  // End a job, once: stop and remove its container, start mining on the card
  // again, and tell the server how it ended. `fast` skips docker's grace period
  // (a shutdown has only a few seconds).
  _finish(job, reason, { exitCode = null, detail = null, fast = false } = {}) {
    if (job.ending) return job.done;
    job.ending = true;
    job.done = (async () => {
      for (const t of [job.hb, job.tempTimer]) if (t) this.d.stopEvery(t);
      if (job.endTimer) this.d.clearTimer(job.endTimer);
      if (job.follow) job.follow.stop();
      if (job.started) {
        if (!fast && reason !== 'exited') await this.d.docker.run(['stop', '-t', '10', job.name], {});
        await this.d.docker.run(['rm', '-f', job.name], {});
      }
      if (job.paused) this.d.resumeCard(job.card);
      this.job = null;
      this.log(reason === 'exited' || reason === 'expired' ? 'info' : 'warn',
        'job on GPU ' + job.card + ' ended (' + reason + (exitCode != null ? ', exit ' + exitCode : '')
        + (detail ? ': ' + detail : '') + ')' + (job.paused ? '; mining resumed on that card' : ''));
      this.emit('change');
      try {
        await this.d.api.end(job.runId, { reason, exitCode, detail, logs: job.logs.splice(-END_LOG_LINES) });
      } catch (e) {
        this.log('warn', 'could not report the end of the job: ' + e.message);
      }
      this._schedule(NEXT_POLL_AFTER_JOB_MS);
    })();
    return job.done;
  }
}

module.exports = { RentalRunner, LLM_WAIT_MS, LLM_WAIT_STEP_MS, NEXT_POLL_AFTER_JOB_MS };
