'use strict';

const { estDailyPrl, estDailyUsd, estDailyUsdLabel, prlToUsd, prlToUsdLabel } = require('../src/shared/earnings');

describe('earnings', () => {
  test('estDailyPrl scales with hashrate and handles bad input', () => {
    expect(estDailyPrl(354)).toBeCloseTo(13.6587, 3);
    expect(estDailyPrl(0)).toBe(0);
    expect(estDailyPrl('not a number')).toBe(0);
  });

  test('estDailyUsd and label', () => {
    expect(estDailyUsd(354)).toBeCloseTo(6.4196, 3);
    expect(estDailyUsdLabel(354)).toBe('$6.42');
    expect(estDailyUsdLabel(0)).toBe('$0.00');
  });

  test('prlToUsd and label match the mock balance', () => {
    expect(prlToUsd(128.407)).toBeCloseTo(60.3513, 3);
    expect(prlToUsd('x')).toBe(0);
    expect(prlToUsdLabel(128.407)).toBe('$60.35');
    expect(prlToUsdLabel(null)).toBe('$0.00');
  });
});
