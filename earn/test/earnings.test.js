'use strict';

const {
  estDailyPrl, estDailyUsd, estDailyUsdLabel, prlToUsd, prlToUsdLabel, HASHES_PER_DIFFICULTY,
} = require('../src/shared/earnings');

// prlscan on 2026-10-06, block 124012: the latest block's difficulty and reward,
// and the price. HeroMiners' calculator said 6.190208 PRL/day ($7.05) for 305
// TH/s at that moment; the app said 6.70 ($7.64) before this formula.
const OCT6 = { DIFFICULTY: 34452600.30923175, BLOCK_REWARD: 2277.90297138, FEE: 1, PRL_USD: 1.1407684455824119 };

describe('earnings', () => {
  test('estDailyPrl scales with hashrate and handles bad input', () => {
    expect(estDailyPrl(354)).toBeCloseTo(7.1852, 3);
    expect(estDailyPrl(708)).toBeCloseTo(2 * estDailyPrl(354), 9);
    expect(estDailyPrl(0)).toBe(0);
    expect(estDailyPrl('not a number')).toBe(0);
  });

  test('estDailyUsd and label', () => {
    expect(estDailyUsd(354)).toBeCloseTo(8.1912, 3);
    expect(estDailyUsdLabel(354)).toBe('$8.19');
    expect(estDailyUsdLabel(0)).toBe('$0.00');
  });

  test('prlToUsd and label', () => {
    expect(prlToUsd(128.407)).toBeCloseTo(146.384, 3);
    expect(prlToUsd('x')).toBe(0);
    expect(prlToUsdLabel(128.407)).toBe('$146.38');
    expect(prlToUsdLabel(null)).toBe('$0.00');
  });

  // The figure the pool's own calculator gives, from the same difficulty and
  // reward, to well under a cent a day.
  test('matches HeroMiners on the same block', () => {
    expect(estDailyPrl(305, OCT6)).toBeCloseTo(6.190, 2);
    expect(estDailyUsdLabel(305, OCT6)).toBe('$7.06');
  });

  test('a block every 2^48 hashes per unit of difficulty, times the reward and the fee', () => {
    // At this difficulty, 1 TH/s finds exactly one block a day.
    const oneADay = { DIFFICULTY: (1e12 * 86400) / HASHES_PER_DIFFICULTY, BLOCK_REWARD: 2000, FEE: 1, PRL_USD: 0.5 };
    expect(estDailyPrl(1, oneADay)).toBeCloseTo(2000, 6);
    // Twice the difficulty halves it; a 2% fee takes 2%.
    expect(estDailyPrl(1, { ...oneADay, DIFFICULTY: 2 * oneADay.DIFFICULTY })).toBeCloseTo(1000, 6);
    expect(estDailyPrl(1, { ...oneADay, FEE: 0.98 })).toBeCloseTo(1960, 6);
    expect(estDailyUsd(1, oneADay)).toBeCloseTo(1000, 6);
    expect(prlToUsd(10, oneADay)).toBeCloseTo(5, 5);
    expect(prlToUsdLabel(10, oneADay)).toBe('$5.00');
  });
});
