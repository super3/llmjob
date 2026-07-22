'use strict';

// Live Pearl economics from the prlscan API (api.prlscan.com) — the same source
// the website earnings calculator uses. Pure parsing so it's unit-testable; the
// IO (the actual fetch) lives in main.js. Any field that fails validation keeps
// the caller's fallback, and `live` records which fields came from the API so
// the app can tell a live number from a stale constant.

const num = (v) => { const n = Number(v); return Number.isFinite(n) && n > 0 ? n : null; };

// PRL/USD from /v1/market/prl. Guarded to well under $1 (PRL trades in cents),
// so a garbage feed can't inflate every dollar figure in the app.
function parsePrice(marketJson) {
  const p = marketJson && num(marketJson.price_usd);
  return p && p < 1000 ? p : null;
}

// Network hashrate in TH/s, averaged over /v1/analytics/block-metrics' ~100-block
// window — single-block estimates swing wildly, so a point sample would be
// noise. Returns null with too few samples or an out-of-range result.
function parseNetTh(metricsJson) {
  const items = metricsJson && Array.isArray(metricsJson.items) ? metricsJson.items : [];
  const hps = items.map((i) => num(i && i.estimated_hashrate_hps)).filter(Boolean);
  if (hps.length < 10) return null;
  const th = hps.reduce((a, b) => a + b, 0) / hps.length / 1e12;
  return th > 1 && th < 1e12 ? th : null;
}

// Daily emission (PRL/day) = the latest block reward × blocks/day at the observed
// average block time. reward_grains is in 1e-8 PRL; block times come from the
// metrics window. Returns null without a reward or enough block-time samples.
function parseDailyPrl(blocksJson, metricsJson) {
  const grains = blocksJson && Array.isArray(blocksJson.items) && blocksJson.items[0]
    && num(blocksJson.items[0].reward_grains);
  const items = metricsJson && Array.isArray(metricsJson.items) ? metricsJson.items : [];
  const times = items.map((i) => num(i && i.block_time_seconds)).filter(Boolean);
  if (!grains || times.length < 10) return null;
  const rewardPrl = grains / 1e8;
  const avgSec = times.reduce((a, b) => a + b, 0) / times.length;
  const daily = rewardPrl * (86400 / avgSec);
  return daily > 1e4 && daily < 1e9 ? daily : null;
}

// Merge live values over a fallback econ ({ NET_TH, DAILY_NET_PRL, FEE, PRL_USD }),
// keeping the fallback for any field the API didn't return a sane value for.
// `live` flags which fields are actually current.
function resolveEconomics(payloads, fallback) {
  const p = payloads || {};
  const base = fallback || {};
  const price = parsePrice(p.market);
  const netTh = parseNetTh(p.metrics);
  const dailyPrl = parseDailyPrl(p.blocks, p.metrics);
  return {
    NET_TH: netTh || base.NET_TH,
    DAILY_NET_PRL: dailyPrl || base.DAILY_NET_PRL,
    FEE: base.FEE,
    PRL_USD: price || base.PRL_USD,
    live: { price: !!price, net: !!netTh, reward: !!dailyPrl },
  };
}

module.exports = { parsePrice, parseNetTh, parseDailyPrl, resolveEconomics };
