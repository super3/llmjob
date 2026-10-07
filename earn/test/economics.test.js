'use strict';

const { parsePrice, parseDifficulty, parseReward, resolveEconomics } = require('../src/shared/economics');

// /v1/blocks?limit=1 as prlscan returns it: the latest block first.
function blocks(over = {}) {
  return { items: [{ height: 124012, difficulty: 34452600.30923175, reward_grains: 227790297138, ...over }] };
}

const FALLBACK = { DIFFICULTY: 34.45e6, BLOCK_REWARD: 2278, FEE: 1, PRL_USD: 1.14 };

describe('parsePrice', () => {
  test('reads a valid sub-$1000 price', () => {
    expect(parsePrice({ price_usd: 1.1407 })).toBeCloseTo(1.1407, 4);
  });
  test('rejects missing, zero, and absurdly high prices', () => {
    expect(parsePrice(null)).toBeNull();
    expect(parsePrice({})).toBeNull();
    expect(parsePrice({ price_usd: 0 })).toBeNull();
    expect(parsePrice({ price_usd: 1500 })).toBeNull();
  });
});

describe('parseDifficulty', () => {
  test('reads the latest block\'s difficulty', () => {
    expect(parseDifficulty(blocks())).toBeCloseTo(34452600.31, 2);
  });
  test('needs a block with a difficulty', () => {
    expect(parseDifficulty(null)).toBeNull();
    expect(parseDifficulty({})).toBeNull();
    expect(parseDifficulty({ items: [] })).toBeNull();
    expect(parseDifficulty(blocks({ difficulty: undefined }))).toBeNull();
  });
  test('rejects an out-of-range difficulty', () => {
    expect(parseDifficulty(blocks({ difficulty: 0.5 }))).toBeNull();
    expect(parseDifficulty(blocks({ difficulty: 1e16 }))).toBeNull();
  });
});

describe('parseReward', () => {
  test('reads the latest block\'s reward in PRL', () => {
    expect(parseReward(blocks())).toBeCloseTo(2277.903, 3);
  });
  test('needs a block with a reward', () => {
    expect(parseReward(null)).toBeNull();
    expect(parseReward({ items: [{}] })).toBeNull();
  });
  test('rejects an out-of-range reward', () => {
    expect(parseReward(blocks({ reward_grains: 1 }))).toBeNull();
    expect(parseReward(blocks({ reward_grains: 1e15 }))).toBeNull();
  });
});

describe('resolveEconomics', () => {
  test('uses live values when they parse, flagging them live', () => {
    const econ = resolveEconomics({ market: { price_usd: 1.1407 }, blocks: blocks() }, FALLBACK);
    expect(econ.PRL_USD).toBeCloseTo(1.1407, 4);
    expect(econ.DIFFICULTY).toBeCloseTo(34452600.31, 2);
    expect(econ.BLOCK_REWARD).toBeCloseTo(2277.903, 3);
    expect(econ.FEE).toBe(1);
    expect(econ.live).toEqual({ price: true, difficulty: true, reward: true });
  });

  test('falls back per-field and flags nothing live when the API is empty', () => {
    const econ = resolveEconomics({}, FALLBACK);
    expect(econ).toMatchObject(FALLBACK);
    expect(econ.live).toEqual({ price: false, difficulty: false, reward: false });
  });

  test('keeps the live fields that parse and falls back on the rest', () => {
    const econ = resolveEconomics({ blocks: blocks({ reward_grains: null }) }, FALLBACK);
    expect(econ.DIFFICULTY).toBeCloseTo(34452600.31, 2);
    expect(econ.BLOCK_REWARD).toBe(FALLBACK.BLOCK_REWARD);
    expect(econ.live).toEqual({ price: false, difficulty: true, reward: false });
  });

  test('tolerates missing payloads/fallback entirely', () => {
    const econ = resolveEconomics(undefined, FALLBACK);
    expect(econ.DIFFICULTY).toBe(FALLBACK.DIFFICULTY);
    expect(econ.live.price).toBe(false);
  });

  test('with no fallback at all, live fields are undefined but it never throws', () => {
    const econ = resolveEconomics();
    expect(econ.DIFFICULTY).toBeUndefined();
    expect(econ.FEE).toBeUndefined();
    expect(econ.live).toEqual({ price: false, difficulty: false, reward: false });
  });
});
