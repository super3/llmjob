'use strict';

// Pool balance lookup for a payout address.
//
// HeroMiners exposes an unauthenticated per-address endpoint,
// GET /api/stats_address?address=<address>, and everything it returns is in
// ATOMIC units — the pool's own config reports coinUnits 1e8, so a `balance` of
// 100000000 is one PRL. Reporting the raw integer as a coin amount would
// overstate a balance by a hundred million.
//
// Unlike AlphaPool, which this replaced, there is no lifetime-paid total in the
// payload: `stats.balance` is the pending (unpaid) balance and the only record
// of what has been paid is the `payments` list. So lifetime paid is summed from
// that list rather than read from a field.
//
// The actual HTTPS GET runs in the main process (no CORS/CSP there); this module
// just builds the URL and parses the response so both are unit-testable.

const POOL_BASE = 'https://pearl.herominers.com';

// Atomic units per PRL, from the pool's own /api/stats config block.
const COIN_UNITS = 100000000;

function buildBalanceUrl(address, base) {
  const a = String(address == null ? '' : address).trim();
  return (base || POOL_BASE) + '/api/stats_address?address=' + encodeURIComponent(a)
    + '&longpoll=false';
}

// One payment's amount in atomic units, or 0 when it cannot be read.
//
// The pool has paid nothing on this coin yet, so the record shape could not be
// observed — cryptonote pools serialise a payment either as an object with an
// `amount`, or as a colon-delimited string whose second field is the amount.
// Both are accepted, and anything else contributes nothing rather than NaN.
function paymentAmount(entry) {
  if (entry && typeof entry === 'object') {
    const n = Number(entry.amount);
    return Number.isFinite(n) && n >= 0 ? n : 0;
  }
  if (typeof entry === 'string') {
    const n = Number(entry.split(':')[1]);
    return Number.isFinite(n) && n >= 0 ? n : 0;
  }
  return 0;
}

// Reduce the pool payload to the display fields, or null when it's unusable.
// `earned` is the balance we show: pending payout plus everything already paid.
// priceUsd (optional) converts the total to USD; omit it to show the coin only.
function parseBalance(json, priceUsd) {
  if (!json || typeof json !== 'object' || Array.isArray(json)) return null;
  const stats = json.stats;
  if (!stats || typeof stats !== 'object') return null;

  // Atomic units arrive as a STRING on this pool, which Number() handles — but
  // a missing or negative balance is a payload we cannot show.
  const pendingRaw = Number(stats.balance);
  if (!Number.isFinite(pendingRaw) || pendingRaw < 0) return null;

  const paidRaw = Array.isArray(json.payments)
    ? json.payments.reduce((a, p) => a + paymentAmount(p), 0)
    : 0;

  const pending = pendingRaw / COIN_UNITS;
  const paid = paidRaw / COIN_UNITS;
  const earned = pending + paid;
  const price = Number(priceUsd);
  const usd = Number.isFinite(price) && price >= 0 ? earned * price : null;
  return { pending, paid, earned, usd };
}

module.exports = { POOL_BASE, COIN_UNITS, buildBalanceUrl, parseBalance };
