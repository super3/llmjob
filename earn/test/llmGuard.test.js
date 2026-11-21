'use strict';

// The local LLM's safety checks (shared/llmGuard): the preflight before it
// starts, the watch once it runs, the guard's one retry, and the board's codes.

const path = require('path');
const {
  LIMITS, REASONS, CODES, preflight, freeDiskMb, createLlmWatch, createLlmGuard, llmState, desktopLlmState,
} = require('../src/shared/llmGuard');

describe('preflight', () => {
  const ok = { cpus: 16, miningCards: 6, cpuMinerThreads: 0, freeRamMb: 8000, freeDiskMb: 50000, needDiskMb: 5000 };

  test('passes a rig with room for the LLM', () => {
    expect(preflight(ok)).toBeNull();
  });

  test('a thread per mining card, the Monero threads and 2 for the LLM', () => {
    expect(preflight({ ...ok, cpus: 8 })).toBeNull();
    expect(preflight({ ...ok, cpus: 2 })).toEqual({
      reason: 'weak-cpu', message: 'only 2 CPU threads for 6 GPUs; the LLM needs 2 more',
    });
    expect(preflight({ ...ok, cpus: 1, miningCards: 1 }).message).toBe('only 1 CPU thread for 1 GPU; the LLM needs 2 more');
    expect(preflight({ ...ok, cpus: 16, cpuMinerThreads: 9 })).toEqual({
      reason: 'weak-cpu', message: 'only 16 CPU threads for 6 GPUs and 9 Monero threads; the LLM needs 2 more',
    });
    expect(preflight({ ...ok, cpus: 8, miningCards: 6, cpuMinerThreads: 1 }).message).toContain('and 1 Monero thread;');
  });

  test('free RAM under 2 GB', () => {
    expect(preflight({ ...ok, freeRamMb: 2048 })).toBeNull();
    expect(preflight({ ...ok, freeRamMb: 1500.4 })).toEqual({
      reason: 'low-ram', message: 'only 1500 MB of free RAM; the LLM needs 2048',
    });
  });

  test('free disk under the download plus 2 GB, only when there is a download', () => {
    expect(preflight({ ...ok, freeDiskMb: 7048 })).toBeNull();
    expect(preflight({ ...ok, freeDiskMb: 6000 })).toEqual({
      reason: 'low-disk', message: 'only 6000 MB of free disk; the model needs 5000 MB and 2048 MB must stay free',
    });
    expect(preflight({ ...ok, freeDiskMb: 100, needDiskMb: 0 })).toBeNull();
  });

  test('figures that could not be read pass, and limits can be changed', () => {
    expect(preflight({ miningCards: 6, freeRamMb: null, freeDiskMb: null, needDiskMb: 5000 })).toBeNull();
    expect(preflight({ cpus: 0, miningCards: 'x', cpuMinerThreads: -1 })).toBeNull();
    expect(preflight()).toBeNull();
    expect(preflight({ ...ok, cpus: 8 }, { llmThreads: 4 }).reason).toBe('weak-cpu');
  });
});

describe('freeDiskMb', () => {
  test('measures the nearest folder that exists', () => {
    const fs = {
      existsSync: (d) => d === path.dirname(path.dirname('/a/b/c')),
      statfsSync: jest.fn(() => ({ bavail: 2048, bsize: 1024 * 1024 })),
    };
    expect(freeDiskMb(fs, path, '/a/b/c')).toBe(2048);
    expect(fs.statfsSync).toHaveBeenCalledWith(path.dirname(path.dirname('/a/b/c')));
  });

  test('stops at the root, and is null when it cannot read', () => {
    const fs = { existsSync: () => false, statfsSync: jest.fn(() => ({ bavail: 1, bsize: 'x' })) };
    expect(freeDiskMb(fs, path, '/a')).toBeNull();
    expect(fs.statfsSync).toHaveBeenCalledWith('/');
    expect(freeDiskMb({ existsSync: () => true, statfsSync: () => { throw new Error('ENOSYS'); } }, path, '/x')).toBeNull();
  });
});

describe('createLlmWatch', () => {
  let t;
  let trips;
  const make = (limits) => createLlmWatch({ limits, now: () => t, onTrip: (x) => trips.push(x) });
  beforeEach(() => { t = 1000; trips = []; });

  test('trips once, and says so', () => {
    const w = make();
    expect(w.isTripped()).toBe(false);
    expect(w.startFailed('no binary')).toBe(true);
    expect(w.died(1)).toBe(true);
    expect(trips).toEqual([{ reason: 'start-failed', message: 'it failed to start (no binary)', retry: true }]);
    expect(w.isTripped()).toBe(true);
  });

  test('a failed start with no detail', () => {
    make().startFailed();
    expect(trips[0].message).toBe('it failed to start');
  });

  test('a death retries', () => {
    make().died(3);
    expect(trips).toEqual([{ reason: 'crashed', message: 'llama-server exited (code 3)', retry: true }]);
  });

  test('three crashes in one start', () => {
    const w = make();
    expect(w.crashed()).toBe(false);
    expect(w.crashed()).toBe(false);
    expect(w.crashed()).toBe(true);
    expect(trips).toEqual([{ reason: 'crashed', message: 'llama-server crashed 3 times', retry: true }]);
  });

  test('health misses count only once it is ready, and an answer resets them', () => {
    const w = make();
    expect(w.health(false)).toBe(false);
    expect(w.isReady()).toBe(false);
    w.ready();
    expect(w.isReady()).toBe(true);
    w.health(false);
    w.health(false);
    w.health(true);
    w.health(false);
    w.health(false);
    expect(trips).toEqual([]);
    expect(w.health(false)).toBe(true);
    expect(trips[0]).toEqual({ reason: 'health', message: 'it missed 3 health checks in a row', retry: true });
  });

  test('too slow: three generation rates in a row under 2 tokens/s, no retry', () => {
    const w = make();
    w.tokens(0);
    w.tokens(1);
    w.tokens(1.5);
    w.tokens(30);
    w.tokens(1);
    w.tokens('x');
    w.tokens(1);
    expect(trips).toEqual([]);
    expect(w.tokens(0.5)).toBe(true);
    expect(trips[0]).toEqual({ reason: 'too-slow', message: 'only 0.5 tokens/s; the floor is 2', retry: false });
  });

  test('hashrate: five checks in a row under 75% of before, skipping paused ones', () => {
    const w = make();
    for (let i = 0; i < 4; i++) w.hashrate({ rate: 50, baseline: 100 });
    w.hashrate({ rate: 0, baseline: 100, paused: true });
    w.hashrate({ rate: 0, baseline: 0 });
    w.hashrate({ rate: 80, baseline: 100 });
    for (let i = 0; i < 4; i++) w.hashrate({ rate: 60, baseline: 100 });
    expect(trips).toEqual([]);
    expect(w.hashrate({ rate: 60, baseline: 100 })).toBe(true);
    expect(trips[0]).toEqual({ reason: 'hashrate-drop', message: 'the shared cards hash at 60% of their rate before the LLM', retry: false });
  });

  test('rejected shares: 5 or more, over 10% of the new ones, or twice the share before', () => {
    const w = make();
    expect(w.shares({ accepted: 100, rejected: 0 })).toBe(false);
    expect(w.shares({ accepted: 200, rejected: 4 })).toBe(false);   // 4: too few
    expect(w.shares({ accepted: 200, rejected: 10 })).toBe(false);  // 10 of 110: 9%
    expect(w.shares({ accepted: 110, rejected: 12 })).toBe(true);   // 12 of 22
    expect(trips[0]).toEqual({ reason: 'rejects', message: '12 of 22 shares rejected since the LLM started', retry: false });
  });

  test('a rig with no shares yet', () => {
    const w = make();
    w.shares({ accepted: 0, rejected: 0 });
    expect(w.shares({ accepted: 0, rejected: 5 })).toBe(true);
  });

  test('a rig that already had rejects needs twice its old share', () => {
    const w = make();
    w.shares({ accepted: 80, rejected: 20 });                        // 20% before
    expect(w.shares({ accepted: 120, rejected: 30 })).toBe(false);  // 10 of 50: 20%
    expect(w.shares({ accepted: 'x', rejected: 'y' })).toBe(false);
    expect(w.shares({ accepted: 100, rejected: 40 })).toBe(true);   // 20 of 40
  });

  test('the CPU lags: three ticks in a row 2 s late', () => {
    const w = make();
    w.lag(2500);
    w.lag(3000);
    w.lag(100);
    w.lag(2000);
    w.lag(2100);
    expect(trips).toEqual([]);
    expect(w.lag(4000)).toBe(true);
    expect(trips[0]).toEqual({ reason: 'cpu-lag', message: 'the timers ran 4.0 s late 3 times in a row', retry: false });
  });

  test('a start that has not answered in 15 minutes has failed; a ready one never times out', () => {
    const w = make();
    t += LIMITS.readyTimeoutMs - 1;
    expect(w.tick()).toBe(false);
    t += 1;
    expect(w.tick()).toBe(true);
    expect(trips[0]).toEqual({ reason: 'start-failed', message: 'it did not answer within 15 minutes', retry: true });
    const ready = make();
    ready.ready();
    t += LIMITS.readyTimeoutMs * 2;
    expect(ready.tick()).toBe(false);
  });

  test('uses the real clock when given none', () => {
    const w = createLlmWatch({ onTrip: () => {} });
    expect(w.tick()).toBe(false);
  });
});

describe('createLlmGuard', () => {
  let lines;
  let stop;
  let retry;
  let timers;
  const make = (enabled = true, extra) => createLlmGuard(Object.assign({
    enabled, now: () => 0, log: (l) => lines.push(l), stop, retry,
    schedule: (fn, ms) => timers.push({ fn, ms }),
  }, extra));
  beforeEach(() => { lines = []; stop = jest.fn(); retry = jest.fn(); timers = []; });

  test('a failed preflight turns the LLM off for the run, in one line', () => {
    const g = make();
    expect(g.enabled).toBe(true);
    expect(g.limits.minFreeRamMb).toBe(2048);
    expect(g.preflight({ cpus: 2, miningCards: 6 })).toBe(false);
    expect(lines).toEqual(['LLM off: only 2 CPU threads for 6 GPUs; the LLM needs 2 more. Mining carries on.']);
    expect(g.off()).toEqual({ reason: 'weak-cpu', message: 'only 2 CPU threads for 6 GPUs; the LLM needs 2 more' });
    expect(g.begin()).toBeNull();
    expect(g.watch()).toBeNull();
  });

  test('a passing preflight leaves it on', () => {
    const g = make();
    expect(g.preflight({ cpus: 16, miningCards: 2 })).toBe(true);
    expect(g.off()).toBeNull();
  });

  test('without the checks: every preflight passes, there is no watch, and it only records why it is off', () => {
    const g = make(false);
    expect(g.enabled).toBe(false);
    expect(g.preflight({ cpus: 1, miningCards: 8 })).toBe(true);
    expect(g.begin()).toBeNull();
    g.note('crashed', 3);
    expect(g.off()).toEqual({ reason: 'crashed', message: 'it kept crashing (3)' });
    g.note('vram');
    expect(g.off().reason).toBe('crashed');
    g.clear();
    expect(g.off()).toBeNull();
    g.note('vram', '');
    expect(g.off()).toEqual({ reason: 'vram', message: 'not enough free VRAM' });
    expect(lines).toEqual([]);
  });

  test('a crash stops the LLM and retries once; a second one turns it off', () => {
    const g = make();
    const w1 = g.begin();
    w1.died(2);
    expect(stop).toHaveBeenCalledTimes(1);
    expect(lines).toEqual(['LLM: llama-server exited (code 2); trying once more in 60 s']);
    expect(timers[0].ms).toBe(60000);
    timers[0].fn();
    expect(retry).toHaveBeenCalledTimes(1);
    const w2 = g.begin();
    expect(w2).not.toBe(w1);
    expect(g.watch()).toBe(w2);
    w2.crashed(); w2.crashed(); w2.crashed();
    expect(stop).toHaveBeenCalledTimes(2);
    expect(timers).toHaveLength(1);
    expect(g.off()).toEqual({ reason: 'crashed', message: 'llama-server crashed 3 times' });
    expect(lines[1]).toBe('LLM off: llama-server crashed 3 times. Mining carries on.');
    // Off stays off: clear() is for runs without the checks.
    g.clear();
    expect(g.off().reason).toBe('crashed');
  });

  test('slowing mining down turns it off at once, with no retry', () => {
    const g = make();
    const w = g.begin();
    for (let i = 0; i < 3; i++) w.lag(5000);
    expect(timers).toEqual([]);
    expect(g.off().reason).toBe('cpu-lag');
  });

  test('a failed start or a death noted during a start goes to its watch', () => {
    const g = make();
    g.begin();
    g.note('start-failed', 'download failed');
    expect(lines[0]).toBe('LLM: it failed to start (download failed); trying once more in 60 s');
    g.begin();
    g.note('crashed', 9);
    expect(g.off()).toEqual({ reason: 'crashed', message: 'llama-server exited (code 9)' });
    // A VRAM refusal never goes to the watch.
    const h = make();
    h.begin();
    h.note('vram');
    expect(h.off().reason).toBe('vram');
  });

  test('a retry is skipped when the LLM went off while it waited, and a stop that throws is ignored', () => {
    stop = jest.fn(() => { throw new Error('gone'); });
    const g = make();
    g.begin().startFailed();
    g.note('vram');
    timers[0].fn();
    expect(retry).not.toHaveBeenCalled();
  });
});

describe('llmState and the board codes', () => {
  test('mine is off, whatever else is said', () => {
    expect(llmState({ mode: 'mining', off: 'vram' })).toEqual({ mode: 'mine', llm: 'off' });
  });

  test('wanted: on, or the short code for why not', () => {
    expect(llmState({ mode: 'auto' })).toEqual({ mode: 'auto', llm: 'on' });
    expect(llmState({ mode: 'both' })).toEqual({ mode: 'auto', llm: 'on' });
    expect(llmState({ mode: 'llm', off: 'vram' })).toEqual({ mode: 'llm', llm: 'vram' });
    for (const r of ['weak-cpu', 'low-ram', 'low-disk']) expect(llmState({ mode: 'auto', off: r }).llm).toBe('cpu');
    for (const r of ['start-failed', 'crashed', 'health', 'too-slow', 'hashrate-drop', 'rejects', 'cpu-lag']) {
      expect(llmState({ mode: 'auto', off: r }).llm).toBe('error');
      expect(REASONS[r]).toEqual(expect.any(String));
    }
    expect(Object.values(CODES).every((c) => ['vram', 'cpu'].includes(c))).toBe(true);
  });

  test('an unknown mode reports null', () => {
    expect(llmState({ mode: 'turbo' })).toEqual({ mode: null, llm: 'on' });
  });

  test('the desktop app: from its mode and its LLM status row', () => {
    expect(desktopLlmState({ mode: 'mining', status: {} })).toEqual({ mode: 'mine', llm: 'off' });
    expect(desktopLlmState({ mode: 'auto', status: { ready: true } })).toEqual({ mode: 'auto', llm: 'on' });
    expect(desktopLlmState({ mode: 'auto', status: { error: 'Needs ~5 GB free VRAM' } }).llm).toBe('vram');
    expect(desktopLlmState({ mode: 'auto', status: { error: 'Setup failed — see Logs' } }).llm).toBe('error');
    expect(desktopLlmState({ mode: 'llm' })).toEqual({ mode: 'llm', llm: 'on' });
    expect(desktopLlmState({ status: {} })).toEqual({ mode: 'auto', llm: 'on' });
  });
});
