'use strict';

// Pool balance lookup for a payout address. AlphaPool exposes an unauthenticated
// per-miner endpoint, GET /api/miner/<address>, whose `balance_prl` is the
// pending (unpaid) balance and `total_paid_prl` the lifetime payout. The actual
// HTTPS GET runs in the main process (no CORS/CSP there); this module just builds
// the URL and parses the response so both are unit-testable.

const POOL_BASE = 'https://pearl.alphapool.tech';

function buildBalanceUrl(address, base) {
  const a = String(address == null ? '' : address).trim();
  return (base || POOL_BASE) + '/api/miner/' + encodeURIComponent(a);
}

// Reduce the pool payload to the display fields, or null when it's unusable.
// `earned` is the balance we show: pending payout (balance_prl) plus lifetime
// paid (total_paid_prl) — i.e. everything the address has earned. priceUsd
// (optional) converts that to USD; omit it for PRL-only.
function parseBalance(json, priceUsd) {
  if (!json || typeof json !== 'object' || Array.isArray(json)) return null;
  const prl = Number(json.balance_prl);
  if (!Number.isFinite(prl) || prl < 0) return null;
  const paidRaw = Number(json.total_paid_prl);
  const paid = Number.isFinite(paidRaw) && paidRaw >= 0 ? paidRaw : 0;
  const earned = prl + paid;
  const price = Number(priceUsd);
  const usd = Number.isFinite(price) && price >= 0 ? earned * price : null;
  return { prl, paid, earned, usd };
}

module.exports = { POOL_BASE, buildBalanceUrl, parseBalance };
