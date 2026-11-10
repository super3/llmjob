'use strict';

const {
  TURN_MS, SETTLE_MS, TURNS, MIN_GAIN, MIN_SAMPLES, createTrial, feed,
} = require('../src/shared/memClockTrial');

// Run a whole trial: the card reports `rates[turn]` TH/s every `everyMs`
// through each of the four turns. Returns every step feed() gave, in order.
function run(rates, { everyMs = 500, skipTurn = null } = {}) {
  const trial = createTrial(0);
  const steps = [];
  let now = 0;
  for (let turn = 0; turn < TURNS.length; turn += 1) {
    const end = (turn + 1) * TURN_MS;
    while (now + everyMs < end) {
      now += everyMs;
      if (turn === skipTurn) continue;
      const step = feed(trial, rates[turn], now);
      if (step) steps.push(step);
    }
    // The sample that ends the turn.
    now = end;
    const step = feed(trial, rates[turn], now);
    if (step) steps.push(step);
  }
  return steps;
}

describe('the memory clock trial', () => {
  // The numbers the README and the log line quote.
  test('is four 75-second turns, alternating, with 15 seconds left out of each', () => {
    expect(TURN_MS).toBe(75000);
    expect(SETTLE_MS).toBe(15000);
    expect(TURNS).toEqual([false, true, false, true]);
    expect(MIN_GAIN).toBe(0.003);
    expect(MIN_SAMPLES).toBe(20);
  });

  // rig01's RTX 4090 at 450 W: 317.5 at the driver's clock, 320.2 at 5001 MHz.
  test('takes and gives back the lock between turns, and keeps a lock that is faster', () => {
    const steps = run([317.5, 320.2, 317.6, 320.3]);
    expect(steps.slice(0, 3)).toEqual([{ lock: true }, { lock: false }, { lock: true }]);
    expect(steps).toHaveLength(4);
    const end = steps[3];
    expect(end.done).toBe(true);
    expect(end.keep).toBe(true);
    expect(end.reason).toBe(null);
    expect(end.offTh).toBeCloseTo(317.55, 6);
    expect(end.onTh).toBeCloseTo(320.25, 6);
    expect(end.gain).toBeCloseTo(320.25 / 317.55 - 1, 9);
  });

  // The 810 MHz speed on the same card: 11% slower.
  test('refuses a lock that is slower, and says by how much', () => {
    const end = run([317.5, 283.1, 317.5, 283.1]).pop();
    expect(end.done).toBe(true);
    expect(end.keep).toBe(false);
    expect(end.reason).toBe('the lock was not ahead by 0.3%');
    expect(end.offTh).toBeCloseTo(317.5, 6);
    expect(end.onTh).toBeCloseTo(283.1, 6);
    expect(end.gain).toBeCloseTo(283.1 / 317.5 - 1, 9);
  });

  test('refuses a lock that is ahead by less than the bar', () => {
    const end = run([300, 300.6, 300, 300.6]).pop();
    expect(end.keep).toBe(false);
    expect(end.reason).toBe('the lock was not ahead by 0.3%');
  });

  // Ahead on the means only because one unlocked turn was slow: a card still
  // warming up, or the pool's difficulty changing under it.
  test('refuses a lock that was not ahead in every turn', () => {
    const end = run([290, 301, 302, 303]).pop();
    expect(end.gain).toBeGreaterThan(MIN_GAIN);
    expect(end.keep).toBe(false);
    expect(end.reason).toBe('the lock was not ahead in every turn');
  });

  test('leaves the start of each turn out of the means', () => {
    const trial = createTrial(1000);
    // Inside the settle time: not counted, whatever it reads.
    expect(feed(trial, 999, 1000 + SETTLE_MS - 1)).toBe(null);
    expect(trial.counts[0]).toBe(0);
    expect(feed(trial, 300, 1000 + SETTLE_MS)).toBe(null);
    expect(trial.counts[0]).toBe(1);
    expect(trial.sums[0]).toBe(300);
  });

  // A core reports 0 while it has no job; that is not a rate to average.
  test('does not count a sample of 0', () => {
    const trial = createTrial(0);
    expect(feed(trial, 0, SETTLE_MS + 1)).toBe(null);
    expect(trial.counts[0]).toBe(0);
  });

  test('does not count the sample that ends a turn', () => {
    const trial = createTrial(0);
    expect(feed(trial, 300, TURN_MS)).toEqual({ lock: true });
    expect(trial.counts).toEqual([0, 0, 0, 0]);
    expect(trial.turn).toBe(1);
    expect(trial.since).toBe(TURN_MS);
  });

  // A card paused or without a job through a turn: no answer is not a yes.
  test('keeps the driver\'s clock when a turn had too few samples', () => {
    const end = run([317.5, 320.2, 317.6, 320.3], { skipTurn: 2 }).pop();
    expect(end).toEqual({
      done: true, keep: false, offTh: null, onTh: null, gain: null,
      reason: 'too few hashrate samples to compare',
    });
  });

  test('a turn is long enough for a slow sampler', () => {
    // One sample every 2.5 s still gives each turn over MIN_SAMPLES.
    const end = run([317.5, 320.2, 317.6, 320.3], { everyMs: 2500 }).pop();
    expect(end.keep).toBe(true);
  });
});
