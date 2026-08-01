'use strict';

const { estDailyPrl, estDailyUsd, estDailyUsdLabel } = require('../src/shared/earnings');

describe('earnings', () => {
  test('estDailyPrl scales with hashrate and handles bad input', () => {
    expect(estDailyPrl(354)).toBeCloseTo(9.3073, 3);
    expect(estDailyPrl(0)).toBe(0);
    expect(estDailyPrl('not a number')).toBe(0);
  });

  test('estDailyUsd and label', () => {
    expect(estDailyUsd(354)).toBeCloseTo(2.7922, 3);
    expect(estDailyUsdLabel(354)).toBe('$2.79');
    expect(estDailyUsdLabel(0)).toBe('$0.00');
  });


  test('an econ override (live values) is used instead of the constants', () => {
    const econ = { NET_TH: 50e6, DAILY_NET_PRL: 1e6, FEE: 0.99, PRL_USD: 0.5 };
    expect(estDailyPrl(100, econ)).toBeCloseTo(1.98, 5);
    expect(estDailyUsd(100, econ)).toBeCloseTo(0.99, 5);
    expect(estDailyUsdLabel(100, econ)).toBe('$0.99');
  });
});
