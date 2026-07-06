'use strict';

const { POOL_BASE, buildBalanceUrl, parseBalance } = require('../src/shared/balance');

const ADDR = 'prl1pql8r6m4z9x7v2k0t3whu8e2snd4p6c';

describe('buildBalanceUrl', () => {
  test('targets the pool miner endpoint and encodes the address', () => {
    expect(buildBalanceUrl(ADDR)).toBe(POOL_BASE + '/api/miner/' + ADDR);
    expect(buildBalanceUrl('  ' + ADDR + '  ')).toBe(POOL_BASE + '/api/miner/' + ADDR); // trimmed
    expect(buildBalanceUrl('a+b')).toBe(POOL_BASE + '/api/miner/a%2Bb');                // encoded
  });

  test('honors a custom base and handles nullish input', () => {
    expect(buildBalanceUrl(ADDR, 'http://localhost:9')).toBe('http://localhost:9/api/miner/' + ADDR);
    expect(buildBalanceUrl(null)).toBe(POOL_BASE + '/api/miner/');
  });
});

describe('parseBalance', () => {
  test('earned = pending + lifetime paid, with USD priced off that total', () => {
    expect(parseBalance({ balance_prl: 3.0933774, total_paid_prl: 330.64 }, 0.47)).toEqual({
      pending: 3.0933774, paid: 330.64, earned: 3.0933774 + 330.64, usd: (3.0933774 + 330.64) * 0.47,
    });
  });

  test('omits USD when no price is supplied and defaults missing paid to 0', () => {
    expect(parseBalance({ balance_prl: 5 })).toEqual({ pending: 5, paid: 0, earned: 5, usd: null });
    expect(parseBalance({ balance_prl: 5, total_paid_prl: -1 }, 0.47).earned).toBe(5); // bad paid → 0, earned = pending
  });

  test('treats a zero balance as valid (a real, credited-nothing-yet account)', () => {
    expect(parseBalance({ balance_prl: 0 }, 0.47)).toEqual({ pending: 0, paid: 0, earned: 0, usd: 0 });
  });

  test('reads the mdl denomination when currency is "mdl"', () => {
    expect(parseBalance({ balance_mdl: 12.5, total_paid_mdl: 4 }, undefined, 'mdl')).toEqual({
      pending: 12.5, paid: 4, earned: 16.5, usd: null,
    });
    // the prl fields are ignored for an mdl lookup, and vice versa
    expect(parseBalance({ balance_prl: 9 }, undefined, 'mdl')).toBeNull();
    expect(parseBalance({ balance_mdl: 9 })).toBeNull();
  });

  test('returns null for unusable payloads', () => {
    expect(parseBalance(null)).toBeNull();
    expect(parseBalance('nope')).toBeNull();
    expect(parseBalance([1, 2])).toBeNull();
    expect(parseBalance({})).toBeNull();                     // no balance_prl
    expect(parseBalance({ balance_prl: 'x' }, 0.47)).toBeNull(); // non-numeric
    expect(parseBalance({ balance_prl: -2 }, 0.47)).toBeNull();  // negative
  });
});
