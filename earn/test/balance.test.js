'use strict';

const {
  POOL_BASE, COIN_UNITS, buildBalanceUrl, parseBalance, buildPoolStatsUrl, parsePoolPriceUsd,
  buildKryptexBalanceUrl, parseKryptexBalance,
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

  // The pool reports a lifetime total, and it is the one to believe. Summing
  // `payments` instead undercounted an address that had been paid 63.63 PRL down
  // to 14.95, because the list is capped at the most recent ten payouts.
  test('takes lifetime paid from stats.paid, not the truncated payments list', () => {
    const b = parseBalance({
      stats: { balance: '36079192', paid: '6363049415' },
      // Ten payouts as the pool really serialises them: tx:amount:fee entries
      // alternating with timestamps, and only the newest ten of many.
      payments: ['aa:185968376:52289', '1788308205', 'bb:185385657:48150', '1788297706'],
    });
    expect(b.pending).toBeCloseTo(0.36079192, 10);
    expect(b.paid).toBeCloseTo(63.63049415, 10);
    expect(b.earned).toBeCloseTo(63.99128607, 10);
  });

  // A pool that reports no lifetime total still gets one, from the list.
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

  // An unusable `paid` must not shadow the list, or a pool that reports it as
  // null would show nothing ever paid.
  test('falls back to the payments list when stats.paid is unusable', () => {
    for (const paid of [undefined, null, '', 'x', -1, NaN]) {
      const b = parseBalance({
        stats: { balance: '0', paid },
        payments: [{ amount: 2 * COIN_UNITS }],
      });
      expect(b.paid).toBe(2);
    }
    // A pool genuinely reporting zero paid is believed, not overridden.
    expect(parseBalance({
      stats: { balance: '0', paid: '0' },
      payments: [{ amount: 2 * COIN_UNITS }],
    }).paid).toBe(0);
  });

  // Blocks that have not matured are the pool's own separate line, and they can
  // still vanish if a block is orphaned, so they stay out of what we call earned.
  test('ignores the unconfirmed block list', () => {
    const b = parseBalance({
      stats: { balance: '0', paid: String(COIN_UNITS) },
      unconfirmed: [{ height: 107357, reward: '41227439', shares: '119537664' }],
    });
    expect(b.earned).toBe(1);
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

// The same pool software runs HeroMiners' Monero pool, the CPU miner's.
describe('parseBalance for Monero', () => {
  const XMR = { units: 1e12, absentIsZero: true };

  test('scales by the units it is given', () => {
    expect(parseBalance({ stats: { balance: '12500000000', paid: '1000000000000' } }, 300, XMR))
      .toEqual({ pending: 0.0125, paid: 1, earned: 1.0125, usd: 1.0125 * 300 });
  });

  // An address with shares and no balance yet: the Monero pool leaves
  // `balance` out, and that is 0, not an unreadable payload. Captured from
  // monero.herominers.com for an address with 12 good shares.
  test('reads a missing balance as 0 when told to', () => {
    const payload = { stats: { shares_good: 12, hashes: '5760000', payments_24h: 0 }, payments: [] };
    expect(parseBalance(payload, undefined, XMR)).toEqual({ pending: 0, paid: 0, earned: 0, usd: null });
    expect(parseBalance(payload)).toBeNull();
    expect(parseBalance({ stats: { balance: null } }, null, XMR).pending).toBe(0);
    expect(parseBalance({ stats: { balance: 'x' } }, null, XMR)).toBeNull();
  });

  test('falls back to the default units for a bad one', () => {
    expect(parseBalance({ stats: { balance: String(COIN_UNITS) } }, null, { units: 0 }).pending).toBe(1);
  });
});

// The Monero balance's dollar figure uses the pool's own XMR price, from
// /api/stats, where HeroMiners gives it as a string under pool.price.
describe('the pool price', () => {
  test('is read from /api/stats on the given pool', () => {
    expect(buildPoolStatsUrl('https://monero.herominers.com')).toBe('https://monero.herominers.com/api/stats');
    expect(buildPoolStatsUrl()).toBe(POOL_BASE + '/api/stats');
  });

  test('is pool.price.usd as a number', () => {
    expect(parsePoolPriceUsd({ pool: { price: { usd: '605.41', btc: '0.00739276' } } })).toBe(605.41);
    expect(parsePoolPriceUsd({ pool: { price: { usd: 605.41 } } })).toBe(605.41);
  });

  test('is null when the payload has none', () => {
    expect(parsePoolPriceUsd(null)).toBeNull();
    expect(parsePoolPriceUsd({})).toBeNull();
    expect(parsePoolPriceUsd({ pool: {} })).toBeNull();
    expect(parsePoolPriceUsd({ pool: { price: {} } })).toBeNull();
    expect(parsePoolPriceUsd({ pool: { price: { usd: '' } } })).toBeNull();
    expect(parsePoolPriceUsd({ pool: { price: { usd: '0' } } })).toBeNull();
    expect(parsePoolPriceUsd({ pool: { price: { usd: 'n/a' } } })).toBeNull();
  });
});

// Kryptex's payout stats, in whole coins: what is confirmed and unpaid, and
// what it has paid.
describe('Kryptex balances', () => {
  test('targets the payout stats of the coin for the address', () => {
    expect(buildKryptexBalanceUrl('https://pool.kryptex.com', 'prl', ' ' + ADDR + ' '))
      .toBe('https://pool.kryptex.com/prl/api/v1/miner/payouts/' + ADDR + '/stats');
    expect(buildKryptexBalanceUrl('https://k', 'xmr', 'a/b')).toBe('https://k/xmr/api/v1/miner/payouts/a%2Fb/stats');
    expect(buildKryptexBalanceUrl('https://k', 'xmr', null)).toBe('https://k/xmr/api/v1/miner/payouts//stats');
  });

  test('earned is unpaid plus paid, with dollars when there is a price', () => {
    const live = { reward: { week: 0.114, month: 0.114 }, paid: 0.0, unpaid: 0.11421937459744771 };
    expect(parseKryptexBalance(live)).toEqual({ pending: 0.11421937459744771, paid: 0, earned: 0.11421937459744771, usd: null });
    expect(parseKryptexBalance({ paid: '2', unpaid: '0.5' }, 2)).toEqual({ pending: 0.5, paid: 2, earned: 2.5, usd: 5 });
    // An address it has never seen.
    expect(parseKryptexBalance({ reward: { week: 0, month: 0 }, paid: 0, unpaid: 0 }, 1)).toEqual({ pending: 0, paid: 0, earned: 0, usd: 0 });
  });

  test('anything else is null, not a guess', () => {
    for (const bad of [null, undefined, 'x', [], {}, { paid: 1 }, { unpaid: 1 }, { paid: '', unpaid: 1 },
      { paid: 1, unpaid: -1 }, { paid: 'x', unpaid: 1 }, { paid: Infinity, unpaid: 1 }, { paid: 1e308, unpaid: 1e308 }]) {
      expect(parseKryptexBalance(bad)).toBeNull();
    }
  });
});
