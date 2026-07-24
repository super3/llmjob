'use strict';

// Demand-driven mining pause for "auto" mode (the client's Phase-4 behavior).
// The GPU can mine and serve inference at once, but PoW mining saturates the
// SMs, so co-running inference is 2–4× slower. This controller pauses mining on
// the serving card(s) while a request is in flight and resumes after a short
// idle debounce — the model stays resident in VRAM, so only compute is freed and
// time-to-first-token stays instant.
//
// Pure event/timer logic with injected pause/resume callbacks and scheduler, so
// it's unit-tested without a process or GPU (same pattern as JobWorker). main.js
// / earn-cli.js wire the callbacks to MinerManager.pause()/resume() and feed it
// the job + chat start/end events. It does nothing until start() — i.e. only in
// auto mode; `both` keeps co-running as before. (A future refinement could pause
// only a sharded model's cards on a multi-GPU rig, rather than the whole miner.)
class ServeController {
  constructor({ pause, resume, log, now, schedule, cancel, resumeDelayMs, maxPauseMs } = {}) {
    // pause() returns false when the miner declines (Windows has no SIGSTOP, or
    // nothing is running) — then we don't consider ourselves paused.
    this.pauseFn = pause || (() => {});
    this.resumeFn = resume || (() => {});
    this.log = log || (() => {});
    this.now = now || Date.now;
    this.schedule = schedule || ((fn, ms) => setTimeout(fn, ms));
    this.cancel = cancel || clearTimeout;
    // Stay paused this long after the last request ends, so bursty traffic
    // doesn't flap mining on/off (each flap wastes a resume + warm-up).
    this.resumeDelayMs = resumeDelayMs != null ? resumeDelayMs : 15000;
    // Never stay paused longer than this: a stuck / stream-forever job must not
    // starve mining, and a long freeze risks the pool dropping the stratum
    // connection. Past the cap we resume and co-run until the current burst ends.
    this.maxPauseMs = maxPauseMs != null ? maxPauseMs : 120000;

    this.enabled = false;
    this.active = 0;
    this.paused = false;
    this.capped = false; // hit the max-pause cap → co-run until this burst ends
    this._resumeTimer = null;
    this._capTimer = null;
  }

  isPaused() { return this.paused; }
  activeJobs() { return this.active; }

  // Enable demand-driven pausing (auto mode). Before this, every event is inert.
  start() { this.enabled = true; }

  // A request became active on the serving card(s).
  jobStarted() {
    this.active++;
    this._clearResumeTimer();
    if (this.enabled && !this.paused && !this.capped) this._pauseNow();
  }

  // A request finished. When the last one drains, debounce a resume.
  jobEnded() {
    if (this.active > 0) this.active--;
    if (this.active === 0) {
      this.capped = false; // burst over — the cap latch lifts
      if (this.paused) this._scheduleResume();
    }
  }

  _pauseNow() {
    if (this.pauseFn() === false) return; // miner declined (Windows / not running)
    this.paused = true;
    this.log('paused mining to serve inference');
    this._capTimer = this.schedule(() => this._capReached(), this.maxPauseMs);
  }

  _capReached() {
    // Only ever armed while paused (and cleared on resume), so we're paused here.
    this._capTimer = null;
    this.capped = true;
    this.paused = false;
    this.resumeFn();
    this.log('max pause reached — mining alongside inference');
  }

  _scheduleResume() {
    this._clearResumeTimer();
    this._resumeTimer = this.schedule(() => this._resumeNow(), this.resumeDelayMs);
  }

  _resumeNow() {
    this._resumeTimer = null;
    if (this.active > 0) return; // a new job arrived during the debounce
    this._clearCapTimer();
    if (this.paused) {
      this.paused = false;
      this.resumeFn();
      this.log('resumed mining (idle)');
    }
  }

  _clearResumeTimer() { if (this._resumeTimer) { this.cancel(this._resumeTimer); this._resumeTimer = null; } }
  _clearCapTimer() { if (this._capTimer) { this.cancel(this._capTimer); this._capTimer = null; } }

  // Tear down: cancel timers and make sure mining is resumed (LLM/worker stopped
  // or mode changed). Idempotent.
  stop() {
    this.enabled = false;
    this.active = 0;
    this.capped = false;
    this._clearResumeTimer();
    this._clearCapTimer();
    if (this.paused) { this.paused = false; this.resumeFn(); }
  }
}

module.exports = { ServeController };
