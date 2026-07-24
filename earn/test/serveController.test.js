'use strict';

const { ServeController } = require('../src/main/serveController');

// Injected fake scheduler: timers are collected so a test fires them by hand.
function harness(over = {}) {
  const timers = [];
  const schedule = jest.fn((fn, ms) => { const t = { fn, ms, cancelled: false }; timers.push(t); return t; });
  const cancel = jest.fn((t) => { if (t) t.cancelled = true; });
  const pause = jest.fn(() => true);
  const resume = jest.fn();
  const log = jest.fn();
  const ctl = new ServeController(Object.assign(
    { pause, resume, log, schedule, cancel, resumeDelayMs: 15000, maxPauseMs: 120000 }, over));
  const timerFor = (ms) => timers.filter((t) => t.ms === ms && !t.cancelled).pop();
  return { ctl, timers, timerFor, schedule, cancel, pause, resume, log };
}

describe('ServeController', () => {
  test('does nothing until start() — events are inert before auto mode enables it', () => {
    const h = harness();
    h.ctl.jobStarted();
    expect(h.pause).not.toHaveBeenCalled();
    expect(h.ctl.isPaused()).toBe(false);
    expect(h.ctl.activeJobs()).toBe(1);
    h.ctl.jobEnded();
    expect(h.resume).not.toHaveBeenCalled();
  });

  test('pauses on the first active job and resumes after the idle debounce', () => {
    const h = harness();
    h.ctl.start();

    h.ctl.jobStarted();
    expect(h.pause).toHaveBeenCalledTimes(1);
    expect(h.ctl.isPaused()).toBe(true);
    // a cap timer was armed
    expect(h.timerFor(120000)).toBeTruthy();

    h.ctl.jobEnded();
    expect(h.resume).not.toHaveBeenCalled(); // debounced, not immediate
    const resumeTimer = h.timerFor(15000);
    expect(resumeTimer).toBeTruthy();

    resumeTimer.fn(); // debounce elapses while still idle
    expect(h.resume).toHaveBeenCalledTimes(1);
    expect(h.ctl.isPaused()).toBe(false);
    expect(h.cancel).toHaveBeenCalled(); // cap timer cleared on resume
  });

  test('overlapping jobs pause once and only resume when the last drains', () => {
    const h = harness();
    h.ctl.start();
    h.ctl.jobStarted();
    h.ctl.jobStarted(); // second concurrent job — no second pause
    expect(h.pause).toHaveBeenCalledTimes(1);
    h.ctl.jobEnded();   // one still running
    expect(h.timerFor(15000)).toBeFalsy(); // no resume scheduled yet
    h.ctl.jobEnded();   // last one drains
    expect(h.timerFor(15000)).toBeTruthy();
  });

  test('a job arriving during the debounce cancels the pending resume', () => {
    const h = harness();
    h.ctl.start();
    h.ctl.jobStarted();
    h.ctl.jobEnded();
    const resumeTimer = h.timerFor(15000);
    h.ctl.jobStarted(); // new work before the debounce fired
    expect(resumeTimer.cancelled).toBe(true);
    expect(h.pause).toHaveBeenCalledTimes(1); // already paused → not re-paused
    // if the stale timer still fires, it's a no-op (active > 0)
    resumeTimer.fn();
    expect(h.resume).not.toHaveBeenCalled();
    expect(h.ctl.isPaused()).toBe(true);
  });

  test('max-pause cap resumes mining and co-runs until the burst ends', () => {
    const h = harness();
    h.ctl.start();
    h.ctl.jobStarted();
    const capTimer = h.timerFor(120000);
    capTimer.fn(); // paused too long → resume + latch
    expect(h.resume).toHaveBeenCalledTimes(1);
    expect(h.ctl.isPaused()).toBe(false);

    // still serving, but the latch prevents re-pausing this burst
    h.ctl.jobStarted();
    expect(h.pause).toHaveBeenCalledTimes(1);
    // burst ends → latch lifts; a fresh job pauses again
    h.ctl.jobEnded();
    h.ctl.jobEnded();
    h.ctl.jobStarted();
    expect(h.pause).toHaveBeenCalledTimes(2);
  });

  test('a resume timer firing after the cap already resumed is a no-op', () => {
    const h = harness();
    h.ctl.start();
    h.ctl.jobStarted();
    h.ctl.jobEnded();                       // arms the resume debounce (15s)
    const resumeTimer = h.timerFor(15000);
    const capTimer = h.timerFor(120000);
    capTimer.fn();                          // cap fires first → already resumed + capped
    expect(h.resume).toHaveBeenCalledTimes(1);
    resumeTimer.fn();                       // stale debounce fires: paused already false
    expect(h.resume).toHaveBeenCalledTimes(1); // no double resume
  });

  test('a miner that declines the pause (Windows / not running) is not marked paused', () => {
    const h = harness({ pause: jest.fn(() => false) });
    h.ctl.start();
    h.ctl.jobStarted();
    expect(h.ctl.isPaused()).toBe(false);
    expect(h.timerFor(120000)).toBeFalsy(); // no cap timer when pause didn't take
    h.ctl.jobEnded();
    expect(h.resume).not.toHaveBeenCalled(); // nothing to resume
  });

  test('stop() cancels timers and resumes mining if paused', () => {
    const h = harness();
    h.ctl.start();
    h.ctl.jobStarted();
    expect(h.ctl.isPaused()).toBe(true);
    h.ctl.stop();
    expect(h.resume).toHaveBeenCalledTimes(1);
    expect(h.ctl.isPaused()).toBe(false);
    expect(h.ctl.activeJobs()).toBe(0);
    // idempotent: stopping again does nothing
    h.ctl.stop();
    expect(h.resume).toHaveBeenCalledTimes(1);
  });

  test('jobEnded underflow is clamped (a stray end never goes negative)', () => {
    const h = harness();
    h.ctl.start();
    h.ctl.jobEnded(); // no active jobs
    expect(h.ctl.activeJobs()).toBe(0);
  });

  test('defaults: no callbacks/scheduler → real timers drive pause/resume', () => {
    jest.useFakeTimers();
    try {
      const ctl = new ServeController(); // all defaults, no-op pause/resume
      ctl.start();
      ctl.jobStarted();             // default pause returns undefined (≠ false) → paused
      expect(ctl.isPaused()).toBe(true);
      ctl.jobEnded();
      jest.advanceTimersByTime(15000); // default 15s debounce via real setTimeout
      expect(ctl.isPaused()).toBe(false);
    } finally {
      jest.useRealTimers();
    }
  });
});
