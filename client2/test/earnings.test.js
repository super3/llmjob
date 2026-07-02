'use strict';

const { estDailyPrl, estDailyUsd, estDailyUsdLabel, prlToUsd, prlToUsdLabel } = require('../src/shared/earnings');

describe('earnings', () => {
  test('estDailyPrl scales with hashrate and handles bad input', () => {
    expect(estDailyPrl(354)).toBeCloseTo(137.1365, 3);
    expect(estDailyPrl(0)).toBe(0);
    expect(estDailyPrl('not a number')).toBe(0);
  });

  test('estDailyUsd and label', () => {
    expect(estDailyUsd(354)).toBeCloseTo(11.2452, 3);
    expect(estDailyUsdLabel(354)).toBe('$11.25');
    expect(estDailyUsdLabel(0)).toBe('$0.00');
  });

  test('prlToUsd and label match the mock balance', () => {
    expect(prlToUsd(128.407)).toBeCloseTo(10.5294, 3);
    expect(prlToUsd('x')).toBe(0);
    expect(prlToUsdLabel(128.407)).toBe('$10.53');
    expect(prlToUsdLabel(null)).toBe('$0.00');
  });
});
