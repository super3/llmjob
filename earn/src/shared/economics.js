'use strict';

// Live Pearl economics from the prlscan API (api.prlscan.com) — the same source
// the website earnings calculator uses. Pure parsing so it's unit-testable; the
// IO (the actual fetch) lives in main.js. Any field that fails validation keeps
// the caller's fallback, and `live` records which fields came from the API so
// the app can tell a live number from a stale constant.

const num = (v) => { const n = Number(v); return Number.isFinite(n) && n > 0 ? n : null; };

// PRL/USD from /v1/market/prl. Capped at $1000, so a garbage feed can't inflate
// every dollar figure in the app.
function parsePrice(marketJson) {
  const p = marketJson && num(marketJson.price_usd);
  return p && p < 1000 ? p : null;
}

// The latest block from /v1/blocks?limit=1, or null.
function latestBlock(blocksJson) {
  return (blocksJson && Array.isArray(blocksJson.items) && blocksJson.items[0]) || null;
}

// The network difficulty of the latest block. The next block's is within a
// fraction of a percent of it. Returns null when missing or out of range.
function parseDifficulty(blocksJson) {
  const b = latestBlock(blocksJson);
  const d = b && num(b.difficulty);
  return d && d > 1 && d < 1e15 ? d : null;
}

// The latest block's reward in PRL. reward_grains is in 1e-8 PRL. Returns null
// when missing or out of range.
function parseReward(blocksJson) {
  const b = latestBlock(blocksJson);
  const grains = b && num(b.reward_grains);
  const prl = grains ? grains / 1e8 : null;
  return prl && prl > 1 && prl < 1e6 ? prl : null;
}

// Merge live values over a fallback econ ({ DIFFICULTY, BLOCK_REWARD, FEE,
// PRL_USD }), keeping the fallback for any field the API didn't return a sane
// value for. `live` flags which fields are actually current.
function resolveEconomics(payloads, fallback) {
  const p = payloads || {};
  const base = fallback || {};
  const price = parsePrice(p.market);
  const difficulty = parseDifficulty(p.blocks);
  const reward = parseReward(p.blocks);
  return {
    DIFFICULTY: difficulty || base.DIFFICULTY,
    BLOCK_REWARD: reward || base.BLOCK_REWARD,
    FEE: base.FEE,
    PRL_USD: price || base.PRL_USD,
    live: { price: !!price, difficulty: !!difficulty, reward: !!reward },
  };
}

module.exports = { parsePrice, parseDifficulty, parseReward, resolveEconomics };
