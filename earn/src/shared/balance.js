'use strict';

// Pool balance lookup for a payout address. AlphaPool exposes an unauthenticated
// per-miner endpoint, GET /api/miner/<address>, whose `balance_<cur>` is the
// pending (unpaid) balance and `total_paid_<cur>` the lifetime payout. Merge
// mining means the same endpoint also serves an mdl1… address (currency 'mdl').
// The actual HTTPS GET runs in the main process (no CORS/CSP there); this module
// just builds the URL and parses the response so both are unit-testable.

const POOL_BASE = 'https://pearl.alphapool.tech';

function buildBalanceUrl(address, base) {
  const a = String(address == null ? '' : address).trim();
  return (base || POOL_BASE) + '/api/miner/' + encodeURIComponent(a);
}

// Reduce the pool payload to the display fields, or null when it's unusable.
// `earned` is the balance we show: pending payout (balance_<cur>) plus lifetime
// paid (total_paid_<cur>) — i.e. everything the address has earned. `currency`
// selects the denomination ('prl' default, or 'mdl' for merge-mined balances).
// priceUsd (optional) converts the total to USD; omit it to show the coin only.
function parseBalance(json, priceUsd, currency) {
  if (!json || typeof json !== 'object' || Array.isArray(json)) return null;
  const cur = currency || 'prl';
  const pending = Number(json['balance_' + cur]);
  if (!Number.isFinite(pending) || pending < 0) return null;
  const paidRaw = Number(json['total_paid_' + cur]);
  const paid = Number.isFinite(paidRaw) && paidRaw >= 0 ? paidRaw : 0;
  const earned = pending + paid;
  const price = Number(priceUsd);
  const usd = Number.isFinite(price) && price >= 0 ? earned * price : null;
  return { pending, paid, earned, usd };
}

module.exports = { POOL_BASE, buildBalanceUrl, parseBalance };
