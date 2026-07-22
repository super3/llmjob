'use strict';

const { parsePrice, parseNetTh, parseDailyPrl, resolveEconomics } = require('../src/shared/economics');

// Build N block-metrics items with the given per-block hashrate (hps) and block
// time (seconds).
function metrics(n, hps, blockSec) {
  return { items: Array.from({ length: n }, () => ({
    estimated_hashrate_hps: hps, block_time_seconds: blockSec,
  })) };
}

const FALLBACK = { NET_TH: 61e6, DAILY_NET_PRL: 1.62e6, FEE: 0.99, PRL_USD: 0.30 };

describe('parsePrice', () => {
  test('reads a valid sub-$1000 price', () => {
    expect(parsePrice({ price_usd: 0.2998 })).toBeCloseTo(0.2998, 4);
  });
  test('rejects missing, zero, and absurdly high prices', () => {
    expect(parsePrice(null)).toBeNull();
    expect(parsePrice({})).toBeNull();
    expect(parsePrice({ price_usd: 0 })).toBeNull();
    expect(parsePrice({ price_usd: 1500 })).toBeNull();
  });
});

describe('parseNetTh', () => {
  test('averages the per-block hashrate into TH/s', () => {
    // 60 EH/s per block → 60e6 TH/s.
    expect(parseNetTh(metrics(12, 60e18, 132))).toBeCloseTo(60e6, 0);
  });
  test('needs at least 10 samples', () => {
    expect(parseNetTh(metrics(9, 60e18, 132))).toBeNull();
    expect(parseNetTh(null)).toBeNull();
  });
  test('rejects an out-of-range average', () => {
    expect(parseNetTh(metrics(10, 1, 132))).toBeNull(); // ~1e-12 TH/s
  });
});

describe('parseDailyPrl', () => {
  const blocks = { items: [{ reward_grains: 248900000000 }] }; // 2489 PRL
  test('reward x blocks-per-day at the observed pace', () => {
    // 2489 PRL every 132s → ~1.63M PRL/day.
    expect(parseDailyPrl(blocks, metrics(12, 60e18, 132))).toBeCloseTo(1.629e6, -4);
  });
  test('needs a reward and enough block-time samples', () => {
    expect(parseDailyPrl({ items: [{}] }, metrics(12, 60e18, 132))).toBeNull();
    expect(parseDailyPrl(null, metrics(12, 60e18, 132))).toBeNull();
    expect(parseDailyPrl(blocks, metrics(9, 60e18, 132))).toBeNull();
  });
  test('rejects an out-of-range emission', () => {
    expect(parseDailyPrl({ items: [{ reward_grains: 1 }] }, metrics(12, 60e18, 132))).toBeNull();
  });
});

describe('resolveEconomics', () => {
  test('uses live values when all three parse, flagging them live', () => {
    const econ = resolveEconomics({
      market: { price_usd: 0.30 },
      metrics: metrics(12, 60e18, 132),
      blocks: { items: [{ reward_grains: 248900000000 }] },
    }, FALLBACK);
    expect(econ.PRL_USD).toBeCloseTo(0.30, 4);
    expect(econ.NET_TH).toBeCloseTo(60e6, 0);
    expect(econ.DAILY_NET_PRL).toBeCloseTo(1.629e6, -4);
    expect(econ.FEE).toBe(0.99);
    expect(econ.live).toEqual({ price: true, net: true, reward: true });
  });

  test('falls back per-field and flags nothing live when the API is empty', () => {
    const econ = resolveEconomics({}, FALLBACK);
    expect(econ).toMatchObject(FALLBACK);
    expect(econ.live).toEqual({ price: false, net: false, reward: false });
  });

  test('tolerates missing payloads/fallback entirely', () => {
    const econ = resolveEconomics(undefined, FALLBACK);
    expect(econ.NET_TH).toBe(FALLBACK.NET_TH);
    expect(econ.live.price).toBe(false);
  });

  test('with no fallback at all, live fields are undefined but it never throws', () => {
    const econ = resolveEconomics();
    expect(econ.NET_TH).toBeUndefined();
    expect(econ.FEE).toBeUndefined();
    expect(econ.live).toEqual({ price: false, net: false, reward: false });
  });
});
