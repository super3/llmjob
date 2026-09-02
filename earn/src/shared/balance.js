'use strict';

// Pool balance lookup for a payout address.
//
// HeroMiners exposes an unauthenticated per-address endpoint,
// GET /api/stats_address?address=<address>, and everything it returns is in
// ATOMIC units — the pool's own config reports coinUnits 1e8, so a `balance` of
// 100000000 is one PRL. Reporting the raw integer as a coin amount would
// overstate a balance by a hundred million.
//
// `stats.balance` is the pending (unpaid) balance and `stats.paid` is the
// lifetime total paid out. Both are authoritative and both are what the pool's
// own address page prints.
//
// Lifetime paid used to be summed from the `payments` list, because when this
// was written the pool had paid nothing on this coin and no `paid` field had
// ever been observed in a response. That undercounted badly once payments
// started: `payments` is TRUNCATED to the most recent ten (twenty entries —
// they alternate `tx:amount:fee` with a timestamp), so an address paid 63.63
// PRL over many payouts displayed 14.95. The list is still the fallback for a
// payload with no `paid` field, but the field wins when it is there.
//
// NOT counted: the top-level `unconfirmed` array, which is the miner's share of
// blocks that have not matured yet (the pool page shows it on its own line).
// That is money which still disappears if a block is orphaned, so it is not
// folded into a figure labelled as earned.
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

  // The pool's own lifetime total, when it reports one. Only fall back to
  // adding up `payments` if it does not — that list is capped at the last ten
  // payouts and silently undercounts everything before them.
  // Absent has to be distinguished from zero before Number() sees it: both
  // Number(null) and Number('') are 0, which is finite and non-negative, so a
  // pool that omits the field would read as "paid nothing" and never fall back.
  const paidField = (stats.paid === null || stats.paid === undefined || stats.paid === '')
    ? NaN
    : Number(stats.paid);
  const paidRaw = Number.isFinite(paidField) && paidField >= 0
    ? paidField
    : (Array.isArray(json.payments)
      ? json.payments.reduce((a, p) => a + paymentAmount(p), 0)
      : 0);

  const pending = pendingRaw / COIN_UNITS;
  const paid = paidRaw / COIN_UNITS;
  const earned = pending + paid;
  const price = Number(priceUsd);
  const usd = Number.isFinite(price) && price >= 0 ? earned * price : null;
  return { pending, paid, earned, usd };
}

module.exports = { POOL_BASE, COIN_UNITS, buildBalanceUrl, parseBalance };
