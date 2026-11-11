'use strict';

const {
  TURN_MS, SETTLE_MS, STEPS_MHZ, MIN_GAIN, MIN_SAMPLES, CHECK_HITS, CHECK_HITS_PER_S, CHECK_MS, CHECK_MIN_HITS,
  stepsFor, checkBound, createTrial, feed, hit,
} = require('../src/shared/coreOffsetTrial');
const { PROFILE, difficultyAdjustmentFactor } = require('../src/shared/miner/pearlhash');

// Run a whole trial as the miner does. The card reports `rates[mhz]` TH/s every
// half second while it is at offset `mhz`; `rates.close` is the last turn at 0
// (default: the same as the first). A step's check gets `hits` right hits one
// after another (default: all it needs), or what `checks[mhz]` says:
//   { wrong: 'why', after: n }  n right hits, then a wrong one
//   { hits: n }                 n right hits, then nothing until the check times out
// `skip` lists offsets whose turn reports no samples but the one that ends it.
// Returns { steps, end, at }: every step the trial gave, the last one, and the
// offset the card was told to be at after each.
function run(rates, { steps: climb, checks = {}, skip = [] } = {}) {
  const trial = climb ? createTrial(0, climb) : createTrial(0);
  const out = [];
  let now = 0;
  let at = 0;
  let closing = false;
  for (let guard = 0; guard < 100000; guard += 1) {
    now += 500;
    const th = at === 0 && closing && rates.close != null ? rates.close : rates[at];
    const quiet = skip.includes(closing && at === 0 ? 'close' : at) && now - trial.since < TURN_MS;
    let step = quiet ? null : feed(trial, th, now);
    while (step) {
      out.push(step);
      if (step.done) return { steps: out, end: step };
      if (step.offset === 0) closing = true;
      at = step.offset;
      if (!step.check) break;
      // The check: the hits the card finds for it.
      const plan = checks[at] || {};
      const right = plan.after != null ? plan.after : (plan.hits != null ? plan.hits : CHECK_HITS);
      step = null;
      for (let i = 0; i < right && !step; i += 1) step = hit(trial, null, now);
      if (!step && plan.wrong) step = hit(trial, plan.wrong, now);
    }
  }
  throw new Error('the trial did not end');
}

describe('the core clock offset trial', () => {
  // The numbers the README and the log lines quote.
  test('is 75-second turns with 15 seconds left out, at 0, +50, +100, +150 MHz and 0 again', () => {
    expect(TURN_MS).toBe(75000);
    expect(SETTLE_MS).toBe(15000);
    expect(STEPS_MHZ).toEqual([50, 100, 150]);
    expect(MIN_GAIN).toBe(0.003);
    expect(MIN_SAMPLES).toBe(20);
    expect(createTrial(0).turns).toEqual([0, 50, 100, 150, 0]);
  });

  test('checks a step with 100 hits, or as many as 10 seconds bring if that is 20 or more', () => {
    expect(CHECK_HITS).toBe(100);
    expect(CHECK_MS).toBe(10000);
    expect(CHECK_MIN_HITS).toBe(20);
    expect(CHECK_HITS_PER_S).toBe(30);
  });

  // rig01's RTX 4090 at 450 W (tuning/benchmark.md, "core clock offset on the RTX 4090").
  const RIG01 = { 0: 317.4, 50: 320.9, 100: 325.3, 150: 331.2 };

  test('climbs a step at a time, each checked first, and goes back to 0 for its last turn', () => {
    const { steps, end } = run(RIG01);
    expect(steps.slice(0, -1)).toEqual([
      { offset: 50, check: true }, { offset: 50, check: false },
      { offset: 100, check: true }, { offset: 100, check: false },
      { offset: 150, check: true }, { offset: 150, check: false },
      { offset: 0, check: false },
    ]);
    expect(end.done).toBe(true);
    expect(end.check).toBe(false);
  });

  test('keeps the highest step when every step was right and faster, and says what it measured', () => {
    const { end } = run(RIG01);
    expect(end.offset).toBe(150);
    expect(end.reason).toBe(null);
    expect(end.failed).toBe(null);
    expect(end.baseTh).toBeCloseTo(317.4, 6);
    expect(end.keptTh).toBeCloseTo(331.2, 6);
    expect(end.gain).toBeCloseTo(331.2 / 317.4 - 1, 9);
    expect(end.rates.map((r) => r.mhz)).toEqual([50, 100, 150]);
    expect(end.rates.map((r) => +r.th.toFixed(1))).toEqual([320.9, 325.3, 331.2]);
  });

  test('keeps a lower step when the one above it was not faster', () => {
    expect(run({ ...RIG01, 150: 325.4 }).end.offset).toBe(100);
    expect(run({ ...RIG01, 100: 321.0, 150: 321.1 }).end.offset).toBe(50);
    // A step has to beat the one before it by 0.3%: just under is not enough.
    expect(run({ 0: 100, 50: 101, 100: 102, 150: 102.3 }).end.offset).toBe(100);
    expect(run({ 0: 100, 50: 101, 100: 102, 150: 102.31 }).end.offset).toBe(150);
  });

  // Each step is judged against the one before it and against offset 0, so a
  // dip in the middle does not rule out the step above it.
  test('keeps a step that beat the one before it, even after a step that did not', () => {
    const { end } = run({ 0: 317.4, 50: 317.5, 100: 325.3, 150: 331.2 });
    expect(end.offset).toBe(150);
  });

  // A step that beat the one before it is not thereby the fastest: after a dip
  // the next step up can pass that test and still be slower than a lower one.
  test('keeps the fastest step, not the highest one that beat the step before it', () => {
    const dip = run({ 0: 100, 50: 102, 100: 100.5, 150: 100.9 }).end;
    expect(dip.offset).toBe(50);
    expect(dip.keptTh).toBeCloseTo(102, 6);
    expect(dip.gain).toBeCloseTo(0.02, 6);
    // A higher step takes a lower one's place only when it is faster by 0.3%:
    // level with it, the card keeps the lower offset.
    expect(run({ 0: 100, 50: 102, 100: 100.5, 150: 102.3 }).end.offset).toBe(50);
    expect(run({ 0: 100, 50: 102, 100: 100.5, 150: 102.31 }).end.offset).toBe(150);
  });

  test('keeps nothing when no step was faster than offset 0 by 0.3%', () => {
    const { end } = run({ 0: 317.4, 50: 317.5, 100: 317.6, 150: 317.7 });
    expect(end.offset).toBe(0);
    expect(end.reason).toBe('no step was faster by 0.3%');
    expect(end.baseTh).toBeCloseTo(317.4, 6);
    expect(end.keptTh).toBe(null);
    expect(end.gain).toBe(null);
    expect(end.rates).toHaveLength(3);
  });

  // The two turns at 0 sit either side of the steps. A card that was still
  // speeding up on its own reads faster at every step than its first turn.
  test('measures against both turns at 0, so a card warming up keeps nothing', () => {
    const { end } = run({ 0: 300, 50: 301, 100: 302, 150: 303, close: 306 });
    expect(end.baseTh).toBeCloseTo(303, 6);
    expect(end.offset).toBe(0);
  });

  describe('the hit check', () => {
    test('one wrong hit fails the step, ends the climb, and the card keeps the last step that passed', () => {
      const { steps, end } = run(RIG01, { checks: { 150: { wrong: 'jackpot hash mismatch (fold/transcript)', after: 57 } } });
      // Straight from the failed check to the last turn at 0: +150 gets no turn.
      expect(steps.slice(-3, -1)).toEqual([{ offset: 150, check: true }, { offset: 0, check: false }]);
      expect(end.offset).toBe(100);
      expect(end.failed).toEqual({ mhz: 150, why: 'jackpot hash mismatch (fold/transcript)' });
      expect(end.rates.map((r) => r.mhz)).toEqual([50, 100]);
    });

    test('a first step that fails leaves the card at 0, with no step to quote', () => {
      const { steps, end } = run(RIG01, { checks: { 50: { wrong: 'Merkle proof does not verify', after: 0 } } });
      expect(steps.slice(0, -1)).toEqual([{ offset: 50, check: true }, { offset: 0, check: false }]);
      expect(end.offset).toBe(0);
      expect(end.reason).toBe('no step passed its hit check');
      expect(end.failed).toEqual({ mhz: 50, why: 'Merkle proof does not verify' });
      expect(end.rates).toEqual([]);
      expect(end.baseTh).toBeCloseTo(317.4, 6);
    });

    test('passes at the time limit with 20 hits or more, and fails with fewer', () => {
      expect(run(RIG01, { checks: { 100: { hits: CHECK_MIN_HITS } } }).end.offset).toBe(150);
      const few = run(RIG01, { checks: { 100: { hits: CHECK_MIN_HITS - 1 } } }).end;
      expect(few.offset).toBe(50);
      expect(few.failed).toEqual({ mhz: 100, why: 'only 19 hits in 10 s, too few to check' });
    });

    test('a turn starts when its check passes, not before', () => {
      const trial = createTrial(0);
      expect(feed(trial, 317, TURN_MS)).toEqual({ offset: 50, check: true });
      // Samples during the check count for nothing and change nothing.
      expect(feed(trial, 999, TURN_MS + 500)).toBe(null);
      expect(feed(trial, 999, TURN_MS + CHECK_MS - 1)).toBe(null);
      for (let i = 1; i < CHECK_HITS; i += 1) expect(hit(trial, null, TURN_MS + 3000)).toBe(null);
      expect(hit(trial, null, TURN_MS + 3000)).toEqual({ offset: 50, check: false });
      expect(trial.since).toBe(TURN_MS + 3000);
      expect(trial.counts[1]).toBe(0);
      // A hit that arrives once the check is over changes nothing, right or wrong.
      expect(hit(trial, null, TURN_MS + 3100)).toBe(null);
      expect(hit(trial, 'jackpot hash mismatch (fold/transcript)', TURN_MS + 3100)).toBe(null);
      expect(trial.failed).toBe(null);
      // The turn is 75 s from the pass.
      expect(feed(trial, 320, 2 * TURN_MS + 2999)).toBe(null);
      expect(feed(trial, 320, 2 * TURN_MS + 3000)).toEqual({ offset: 100, check: true });
    });
  });

  describe('a turn with too few samples is no answer', () => {
    test('at 0, first or last: nothing is kept and no rates are quoted', () => {
      for (const turn of [0, 'close']) {
        const { end } = run(RIG01, { skip: [turn] });
        expect(end.offset).toBe(0);
        expect(end.reason).toBe('too few hashrate samples to compare');
        expect(end.baseTh).toBe(null);
        expect(end.gain).toBe(null);
      }
    });

    test('at a step: that step is not kept, and neither is the one above it', () => {
      const top = run(RIG01, { skip: [150] }).end;
      expect(top.offset).toBe(100);
      expect(top.rates[2]).toEqual({ mhz: 150, th: null });
      // +150 cannot be shown to beat the step before it.
      expect(run(RIG01, { skip: [100] }).end.offset).toBe(50);
      expect(run(RIG01, { skip: [50] }).end.offset).toBe(150);
      expect(run(RIG01, { skip: [50, 100] }).end.offset).toBe(0);
    });

    test('a sample of zero is not counted', () => {
      const { end } = run({ 0: 317.4, 50: 0, 100: 325.3, 150: 331.2 });
      expect(end.rates[0]).toEqual({ mhz: 50, th: null });
    });
  });

  test('leaves the start of each turn and the sample that ends it out of the mean', () => {
    const trial = createTrial(0);
    feed(trial, 999, SETTLE_MS - 1);
    expect(trial.counts[0]).toBe(0);
    feed(trial, 317, SETTLE_MS);
    expect([trial.sums[0], trial.counts[0]]).toEqual([317, 1]);
    expect(feed(trial, 999, TURN_MS)).toEqual({ offset: 50, check: true });
    expect([trial.sums[0], trial.counts[0]]).toEqual([317, 1]);
  });

  describe('a card whose driver allows less', () => {
    test('tries only the steps inside its range', () => {
      expect(stepsFor(1000)).toEqual([50, 100, 150]);
      expect(stepsFor(150)).toEqual([50, 100, 150]);
      expect(stepsFor(149)).toEqual([50, 100]);
      expect(stepsFor(50)).toEqual([50]);
      expect(stepsFor(49)).toEqual([]);
      expect(stepsFor(0)).toEqual([]);
    });

    test('runs the shorter climb', () => {
      const { steps, end } = run(RIG01, { steps: [50, 100] });
      expect(steps.map((s) => s.offset)).toEqual([50, 50, 100, 100, 0, 100]);
      expect(end.offset).toBe(100);
    });
  });

  // The check's job is set to hit about 30 times a second at the card's own
  // rate, so a slow card's check takes no longer than a fast one's.
  describe('the bound of a check\'s job', () => {
    const macs = difficultyAdjustmentFactor(PROFILE);
    const hitsPerSec = (th, bound) => (th * 1e12 / macs) * Number(bound) / 2 ** 256;

    test('gives any card about 30 hits a second', () => {
      for (const th of [12.5, 95.7, 317.4, 700]) {
        expect(hitsPerSec(th, checkBound(th, macs))).toBeCloseTo(CHECK_HITS_PER_S, 3);
      }
    });

    // 2^232 is the bound native/probes/verify-hits.js uses, which gave rig01's
    // RTX 4090 about 36 hits a second.
    test('is near the probe\'s 2^232 on an RTX 4090', () => {
      const bits = Math.log2(Number(checkBound(317.4, macs)));
      expect(bits).toBeGreaterThan(231.5);
      expect(bits).toBeLessThan(232);
    });

    test('takes a card with no rate yet as 1 TH/s', () => {
      const one = checkBound(1, macs);
      for (const none of [0, undefined, null, NaN, -5, 0.2]) expect(checkBound(none, macs)).toBe(one);
      expect(one < (1n << 256n)).toBe(true);
    });
  });
});
