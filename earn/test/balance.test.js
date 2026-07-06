'use strict';

const {
  POOL_BASE, buildBalanceUrl, parseBalance, buildMdlBalanceUrl, parseMdlBalance,
} = require('../src/shared/balance');

const ADDR = 'prl1pql8r6m4z9x7v2k0t3whu8e2snd4p6c';
const MDL_ADDR = 'mdl1pdu94f4vwf97fauryx37rzs654qc0p578h0ew9wr2zhvecfr6n3lscruajj';

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

describe('buildMdlBalanceUrl', () => {
  test('targets the PRL miner record\'s /mdl route (not the mdl1… address)', () => {
    expect(buildMdlBalanceUrl(ADDR)).toBe(POOL_BASE + '/api/miner/' + ADDR + '/mdl');
    expect(buildMdlBalanceUrl(ADDR, 'http://localhost:9')).toBe('http://localhost:9/api/miner/' + ADDR + '/mdl');
  });
});

describe('parseMdlBalance', () => {
  const payload = (over) => Object.assign({
    has_mdl: true,
    mdl_address: MDL_ADDR,
    summary: { blocks_found: 0, pending_mdl: 0.75457042, total_paid_mdl: 1.08184323 },
    recent_payouts: [],
  }, over);

  test('earned = summary pending + lifetime paid, echoing the paired address', () => {
    expect(parseMdlBalance(payload())).toEqual({
      pending: 0.75457042, paid: 1.08184323, earned: 0.75457042 + 1.08184323,
      usd: null, mdlAddress: MDL_ADDR,
    });
  });

  test('defaults missing/bad paid to 0 and missing mdl_address to ""', () => {
    expect(parseMdlBalance(payload({ summary: { pending_mdl: 2 } })).earned).toBe(2);
    expect(parseMdlBalance(payload({ summary: { pending_mdl: 2, total_paid_mdl: -1 } })).paid).toBe(0);
    expect(parseMdlBalance(payload({ mdl_address: undefined })).mdlAddress).toBe('');
  });

  test('treats a zero balance as valid (paired but credited nothing yet)', () => {
    expect(parseMdlBalance(payload({ summary: { pending_mdl: 0 } })).earned).toBe(0);
  });

  test('returns null when the record has no MDL pairing', () => {
    expect(parseMdlBalance(payload({ has_mdl: false }))).toBeNull();
    expect(parseMdlBalance({ mdl_address: MDL_ADDR, summary: { pending_mdl: 1 } })).toBeNull(); // has_mdl absent
  });

  test('returns null for unusable payloads', () => {
    expect(parseMdlBalance(null)).toBeNull();
    expect(parseMdlBalance('nope')).toBeNull();
    expect(parseMdlBalance([1, 2])).toBeNull();
    expect(parseMdlBalance(payload({ summary: undefined }))).toBeNull();
    expect(parseMdlBalance(payload({ summary: 'x' }))).toBeNull();
    expect(parseMdlBalance(payload({ summary: [1] }))).toBeNull();
    expect(parseMdlBalance(payload({ summary: { pending_mdl: 'x' } }))).toBeNull(); // non-numeric
    expect(parseMdlBalance(payload({ summary: { pending_mdl: -2 } }))).toBeNull();  // negative
  });
});
