'use strict';

const {
  POOL_BASE, COIN_UNITS, buildBalanceUrl, parseBalance,
} = require('../src/shared/balance');

const ADDR = 'prl1pql8r6m4z9x7v2k0t3whu8e2snd4p6c';
const Q = '/api/stats_address?address=';
const LP = '&longpoll=false';

describe('buildBalanceUrl', () => {
  test('targets the pool address endpoint and encodes the address', () => {
    expect(buildBalanceUrl(ADDR)).toBe(POOL_BASE + Q + ADDR + LP);
    expect(buildBalanceUrl('  ' + ADDR + '  ')).toBe(POOL_BASE + Q + ADDR + LP); // trimmed
    expect(buildBalanceUrl('a+b')).toBe(POOL_BASE + Q + 'a%2Bb' + LP);           // encoded
  });

  // Without longpoll=false the endpoint holds the connection open until the
  // address next submits a share, which turns a poll into a hang.
  test('always disables long polling', () => {
    expect(buildBalanceUrl(ADDR)).toContain(LP);
  });

  test('honors a custom base and handles nullish input', () => {
    expect(buildBalanceUrl(ADDR, 'http://localhost:9')).toBe('http://localhost:9' + Q + ADDR + LP);
    expect(buildBalanceUrl(null)).toBe(POOL_BASE + Q + LP);
  });
});

describe('parseBalance', () => {
  // The pool reports atomic units (coinUnits 1e8), so every figure here is
  // scaled on the way out. Getting this wrong would overstate a balance by
  // eight orders of magnitude, which is the kind of bug a user notices.
  test('converts atomic units to PRL, earned = pending + lifetime paid', () => {
    const json = { stats: { balance: 3.0933774 * COIN_UNITS, paid: 330.64 * COIN_UNITS } };
    const b = parseBalance(json, 0.47);
    expect(b.pending).toBeCloseTo(3.0933774, 6);
    expect(b.paid).toBeCloseTo(330.64, 6);
    expect(b.earned).toBeCloseTo(333.7333774, 6);
    expect(b.usd).toBeCloseTo(333.7333774 * 0.47, 6);
  });

  test('omits USD when no price is supplied', () => {
    expect(parseBalance({ stats: { balance: 5 * COIN_UNITS } }))
      .toEqual({ pending: 5, paid: 0, earned: 5, usd: null });
  });

  // A brand-new address really does come back with neither key — verified
  // against one that had just submitted its first shares — so absent must mean
  // zero, not "unusable". Showing "—" to somebody who just started mining reads
  // as the app being broken.
  test('a stats object with no balance fields is zero, not null', () => {
    expect(parseBalance({ stats: { shares_good: 7, hashrate_1h: 12088.5 } }, 0.47))
      .toEqual({ pending: 0, paid: 0, earned: 0, usd: 0 });
  });

  test('accepts the alternate spellings the pool software may use', () => {
    expect(parseBalance({ stats: { pending: 2 * COIN_UNITS } }).pending).toBe(2);
    expect(parseBalance({ stats: { unpaid: 3 * COIN_UNITS } }).pending).toBe(3);
    expect(parseBalance({ stats: { total_paid: 4 * COIN_UNITS } }).paid).toBe(4);
    expect(parseBalance({ stats: { totalPaid: 5 * COIN_UNITS } }).paid).toBe(5);
  });

  test('skips unusable values and falls through to the next spelling', () => {
    expect(parseBalance({ stats: { balance: 'x', pending: 7 * COIN_UNITS } }).pending).toBe(7);
    expect(parseBalance({ stats: { balance: -2, pending: 8 * COIN_UNITS } }).pending).toBe(8);
    expect(parseBalance({ stats: { paid: 'x' } }).paid).toBe(0);
  });

  test('returns null for unusable payloads', () => {
    expect(parseBalance(null)).toBeNull();
    expect(parseBalance('nope')).toBeNull();
    expect(parseBalance([1, 2])).toBeNull();
    expect(parseBalance({})).toBeNull();               // no stats object at all
    expect(parseBalance({ stats: null })).toBeNull();
    expect(parseBalance({ stats: 'nope' })).toBeNull();
    expect(parseBalance({ stats: [1, 2] })).toBeNull();
  });
});
