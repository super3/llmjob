'use strict';

// shared/kTrial: which k an RX 9070 XT mines on its PC, k 8192 or k 16384.
// pearlMinerKTrial.test.js has the miner's half.

const k = require('../src/shared/kTrial');
const { MAX_AGE_MS } = require('../src/shared/shapePick');

// Runs a whole trial with each turn's rate given by `rate(k, turn)`, a sample
// every 500 ms. Returns every step that was not null.
function run(rate, { every = 500, from = 0 } = {}) {
  const t = k.createTrial(from);
  const steps = [];
  for (let now = from; ; now += every) {
    const s = k.feed(t, t.turn < 0 ? rate(8192, -1) : rate(k.TURNS[t.turn], t.turn), now);
    if (!s) continue;
    steps.push(s);
    if (s.done) return steps;
  }
}

describe('kTrial', () => {
  test('warms up, then takes turns at k 8192, 16384, 16384 and 8192, about 110 s in all', () => {
    expect(k.TURNS).toEqual([8192, 16384, 16384, 8192]);
    expect(k.WARM_MS + k.TURNS.length * k.TURN_MS).toBe(110000);
    const steps = run((kk) => (kk === 16384 ? 101 : 100));
    // Only a change of k asks for anything: two in the trial.
    expect(steps.slice(0, 2)).toEqual([{ k: 16384 }, { k: 8192 }]);
    expect(steps).toHaveLength(3);
    // Nothing from the warm-up is counted, however slow.
    const t = k.createTrial(0);
    k.feed(t, 1, 1000);
    k.feed(t, 1, k.WARM_MS - 1);
    expect(t.counts).toEqual([0, 0, 0, 0]);
    expect(k.feed(t, 1, k.WARM_MS)).toBeNull();
    expect(t.turn).toBe(0);
  });

  test('keeps k 16384 when it is ahead by the bar in every turn', () => {
    const done = run((kk) => (kk === 16384 ? 100.65 : 100)).pop();
    expect(done).toMatchObject({ done: true, k: 16384, baseTh: 100, reason: null });
    expect(done.wideTh).toBeCloseTo(100.65, 9);
    expect(done.gain).toBeCloseTo(0.0065, 9);
  });

  test('keeps k 8192 on a PC where the two are level, or k 16384 is slower', () => {
    expect(run((kk) => (kk === 16384 ? 100.1 : 100)).pop()).toMatchObject({
      k: 8192, reason: 'k 16384 was not ahead by 0.3%',
    });
    const slower = run((kk) => (kk === 16384 ? 98.4 : 100)).pop();
    expect(slower.k).toBe(8192);
    expect(slower.gain).toBeCloseTo(-0.016, 9);
  });

  test('keeps k 8192 when k 16384 is ahead on the means but not in every turn', () => {
    // A PC whose last turn at k 8192 beat k 16384's first.
    const rates = [100, 101.5, 102, 101.6];
    expect(run((kk, turn) => rates[turn]).pop()).toMatchObject({
      k: 8192, reason: 'k 16384 was not ahead in every turn',
    });
  });

  test('leaves out each turn\'s settling time and nothing at 0 TH/s', () => {
    // A slow start in every turn, and no rate at all from a core still opening.
    const done = run((kk) => (kk === 16384 ? 101 : 100), { every: 500 }).pop();
    const t = k.createTrial(0);
    k.feed(t, 100, k.WARM_MS);
    k.feed(t, 50, k.WARM_MS + 1);
    k.feed(t, 0, k.WARM_MS + k.SETTLE_MS);
    k.feed(t, 100, k.WARM_MS + k.SETTLE_MS + 1);
    expect(t.counts[0]).toBe(1);
    expect(t.sums[0]).toBe(100);
    expect(done.baseTh).toBe(100);
  });

  test('is no answer when a turn had too few samples', () => {
    // One sample every 2 s: 7 a turn after the settling time.
    const done = run(() => 100, { every: 2000 }).pop();
    expect(done).toEqual({
      done: true, k: 8192, baseTh: null, wideTh: null, gain: null, reason: 'too few hashrate samples to compare',
    });
    expect(k.remember(done, 5)).toBeNull();
  });

  test('times only a gfx1201 card that opened at k 8192', () => {
    expect(k.eligible('gfx1201', 8192)).toBe(true);
    expect(k.eligible('gfx1201', 4096)).toBe(false);
    expect(k.eligible('gfx1201', 16384)).toBe(false);
    expect(k.eligible('gfx1200', 8192)).toBe(false);
    expect(k.eligible(null, 8192)).toBe(false);
  });

  test('remembers an answer for a week, under the launch shape key', () => {
    const v = { k: 16384, baseTh: 136.304, wideTh: 137.2049, gain: 0.0066 };
    const saved = k.remember(v, 1000);
    expect(saved).toEqual({ k: 16384, at: 1000, baseTh: 136.3, wideTh: 137.2 });
    expect(k.recall(saved, 1000 + MAX_AGE_MS - 1, false)).toBe(saved);
    expect(k.recall(saved, 1000 + MAX_AGE_MS, false)).toBeNull();
    // Dated after now: the PC's clock was set back.
    expect(k.recall(saved, 999, false)).toBeNull();
    expect(k.recall(saved, 2000, true)).toBeNull();
    expect(k.recall(null, 2000, false)).toBeNull();
    expect(k.recall({ k: 4096, at: 1000 }, 2000, false)).toBeNull();
    expect(k.cardKey(0, [{ index: 0, name: 'AMD Radeon RX 9070 XT' }])).toBe('gpu0 AMD Radeon RX 9070 XT');
    expect(k.cardKey(undefined, [])).toBe('auto');
  });

  test('reads its switches from the environment', () => {
    expect(k.trialsOff({ LLMJOB_K_TRIAL: '0' })).toBe(true);
    expect(k.trialsOff({ LLMJOB_K_TRIAL: '1' })).toBe(false);
    expect(k.trialsOff(null)).toBe(false);
    expect(k.retryAsked({ LLMJOB_K_TRIAL_RETRY: '1' })).toBe(true);
    expect(k.retryAsked({})).toBe(false);
    expect(k.retryAsked(null)).toBe(false);
    expect(k.pct(0.0065)).toBe('+0.65%');
    expect(k.pct(-0.016)).toBe('-1.60%');
  });
});
