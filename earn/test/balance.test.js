'use strict';

const {
  POOL_BASE, COIN_UNITS, buildBalanceUrl, parseBalance,
} = require('../src/shared/balance');

const ADDR = 'prl1pql8r6m4z9x7v2k0t3whu8e2snd4p6c';
const URL = (a) => POOL_BASE + '/api/stats_address?address=' + a + '&longpoll=false';

// AlphaPool's /api/miner/<address> returned PRL directly. HeroMiners returns
// atomic units and nests everything under `stats`, so both halves of this
// module changed at once — the URL shape and the scaling.

describe('buildBalanceUrl', () => {
  test('targets the address stats endpoint and encodes the address', () => {
    expect(buildBalanceUrl(ADDR)).toBe(URL(ADDR));
    expect(buildBalanceUrl('  ' + ADDR + '  ')).toBe(URL(ADDR)); // trimmed
    expect(buildBalanceUrl('a+b')).toBe(URL('a%2Bb'));           // encoded
  });

  // longpoll is pinned off: the endpoint otherwise holds the connection open
  // waiting for a change, which is not what a one-shot balance read wants.
  test('disables longpoll', () => {
    expect(buildBalanceUrl(ADDR)).toContain('longpoll=false');
  });

  test('honors a custom base and handles nullish input', () => {
    expect(buildBalanceUrl(ADDR, 'http://localhost:9'))
      .toBe('http://localhost:9/api/stats_address?address=' + ADDR + '&longpoll=false');
    expect(buildBalanceUrl(null)).toBe(URL(''));
  });
});

describe('parseBalance', () => {
  // The pool's own config reports coinUnits 1e8. Showing the raw integer would
  // overstate a balance by a hundred million.
  test('scales atomic units into PRL', () => {
    const b = parseBalance({ stats: { balance: String(3 * COIN_UNITS) } });
    expect(b.pending).toBe(3);
    expect(b.earned).toBe(3);
  });

  // Atomic units arrive as a STRING on this pool.
  test('accepts the balance as a string or a number', () => {
    expect(parseBalance({ stats: { balance: '5461345' } }).pending).toBe(0.05461345);
    expect(parseBalance({ stats: { balance: 5461345 } }).pending).toBe(0.05461345);
  });

  // There is no lifetime-paid field on this pool, so paid is summed from the
  // payments list. The coin has paid nothing yet, which is why the record shape
  // could not be observed and both plausible shapes are accepted.
  test('sums lifetime paid out of the payments list, in either record shape', () => {
    const asObjects = parseBalance({
      stats: { balance: String(COIN_UNITS) },
      payments: [{ amount: COIN_UNITS }, { amount: 2 * COIN_UNITS }],
    });
    expect(asObjects.paid).toBe(3);
    expect(asObjects.earned).toBe(4);

    const asStrings = parseBalance({
      stats: { balance: '0' },
      payments: ['abc123:' + COIN_UNITS + ':0:0:1700000000'],
    });
    expect(asStrings.paid).toBe(1);
  });

  test('an unreadable payment contributes nothing rather than NaN', () => {
    const b = parseBalance({
      stats: { balance: '0' },
      payments: [null, 42, 'nocolon', { amount: 'x' }, { amount: -5 }, ['a']],
    });
    expect(b.paid).toBe(0);
    expect(b.earned).toBe(0);
  });

  test('no payments list at all means nothing paid', () => {
    expect(parseBalance({ stats: { balance: '0' } }).paid).toBe(0);
    expect(parseBalance({ stats: { balance: '0' }, payments: 'nope' }).paid).toBe(0);
  });

  test('prices the total when given a price, and omits USD otherwise', () => {
    const priced = parseBalance({
      stats: { balance: String(COIN_UNITS) },
      payments: [{ amount: COIN_UNITS }],
    }, 0.47);
    expect(priced.usd).toBeCloseTo(2 * 0.47, 10);
    expect(parseBalance({ stats: { balance: String(COIN_UNITS) } }).usd).toBeNull();
  });

  test('treats a zero balance as valid (a real, credited-nothing-yet account)', () => {
    expect(parseBalance({ stats: { balance: '0' } }, 0.47))
      .toEqual({ pending: 0, paid: 0, earned: 0, usd: 0 });
  });

  test('returns null for unusable payloads', () => {
    expect(parseBalance(null)).toBeNull();
    expect(parseBalance('nope')).toBeNull();
    expect(parseBalance([1, 2])).toBeNull();
    expect(parseBalance({})).toBeNull();                          // no stats
    expect(parseBalance({ stats: 'x' })).toBeNull();
    expect(parseBalance({ stats: [1] })).toBeNull();              // an array is not the object
    expect(parseBalance({ stats: {} })).toBeNull();               // no balance
    expect(parseBalance({ stats: { balance: 'x' } }, 0.47)).toBeNull(); // non-numeric
    expect(parseBalance({ stats: { balance: -2 } }, 0.47)).toBeNull();  // negative
  });
});
