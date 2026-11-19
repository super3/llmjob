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
//
// opts:
//   units          atomic units per coin (XMR: 1e12; default COIN_UNITS)
//   absentIsZero   a payload with no `balance` reads as 0. Both HeroMiners
//                  pools leave the field out for an address that has shares
//                  but no balance yet. The Pearl pool always sent it until its
//                  dashboard relaunch on 18 Nov 2026, which also reset every
//                  address's history.
function parseBalance(json, priceUsd, opts) {
  if (!json || typeof json !== 'object' || Array.isArray(json)) return null;
  const stats = json.stats;
  if (!stats || typeof stats !== 'object') return null;
  const o = opts || {};
  const units = Number(o.units) > 0 ? Number(o.units) : COIN_UNITS;

  // Atomic units arrive as a STRING on this pool, which Number() handles — but
  // a missing or negative balance is a payload we cannot show.
  const pendingRaw = o.absentIsZero && stats.balance == null ? 0 : Number(stats.balance);
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

  const pending = pendingRaw / units;
  const paid = paidRaw / units;
  const earned = pending + paid;
  const price = Number(priceUsd);
  const usd = Number.isFinite(price) && price >= 0 ? earned * price : null;
  return { pending, paid, earned, usd };
}

// The pool's own price for its coin. HeroMiners' /api/stats carries it in USD as
// pool.price.usd, a string. The Monero balance uses it for its dollar figure;
// the Pearl balance uses prlscan's price instead (ECON_API).
function buildPoolStatsUrl(base) {
  return (base || POOL_BASE) + '/api/stats';
}

// The USD price, or null when the payload has none.
function parsePoolPriceUsd(json) {
  const price = json && json.pool && json.pool.price;
  const usd = Number(price && price.usd);
  return Number.isFinite(usd) && usd > 0 ? usd : null;
}

// Kryptex (pool.kryptex.com, its openapi.yaml) answers per coin at
// GET /{coin}/api/v1/miner/payouts/{address}/stats with
// {"reward":{…},"paid":0.0,"unpaid":0.114}, in whole coins, not atomic units.
// `unpaid` is the confirmed balance; the blocks still maturing are left out,
// as they are on HeroMiners. An address it has never seen gets zeros.
// `base` is config's KRYPTEX.apiBase.
function buildKryptexBalanceUrl(base, coin, address) {
  const a = String(address == null ? '' : address).trim();
  return base + '/' + coin + '/api/v1/miner/payouts/' + encodeURIComponent(a) + '/stats';
}

// The same display fields as parseBalance, or null when unusable.
function parseKryptexBalance(json, priceUsd) {
  if (!json || typeof json !== 'object' || Array.isArray(json)) return null;
  const amount = (v) => (v === null || v === undefined || v === '' ? NaN : Number(v));
  const pending = amount(json.unpaid);
  const paid = amount(json.paid);
  if (!(pending >= 0) || !(paid >= 0) || !Number.isFinite(pending + paid)) return null;
  const earned = pending + paid;
  const price = Number(priceUsd);
  const usd = Number.isFinite(price) && price >= 0 ? earned * price : null;
  return { pending, paid, earned, usd };
}

// What the address has received on chain, from prlscan's address record:
// GET /v1/addresses/{address} → {"external_received_grains": …, …}, in grains
// (1e-8 PRL). HeroMiners' 18 Nov 2026 relaunch wiped its `paid` totals, so the
// pool alone now reports 0 for addresses it has paid for weeks. The chain still
// has every payout. "External" leaves out the address paying itself.
const CHAIN_API = 'https://api.prlscan.com/v1/addresses/';
const GRAINS_PER_PRL = 1e8;

function buildChainAddressUrl(address) {
  return CHAIN_API + encodeURIComponent(String(address == null ? '' : address).trim());
}

// PRL received, or null when the payload has no usable figure.
function parseChainReceived(json) {
  if (!json || typeof json !== 'object' || Array.isArray(json)) return null;
  const v = json.external_received_grains;
  if (v === null || v === undefined || v === '') return null;
  const g = Number(v);
  return Number.isFinite(g) && g >= 0 ? g / GRAINS_PER_PRL : null;
}

// A HeroMiners balance with its lifetime paid raised to what the chain shows
// the address received, when that is more. Either side may be null; null when
// both are. On chain counts every payment in, from any pool or person, which
// for a mining payout address is its payouts.
function withChainPaid(bal, receivedPrl, priceUsd) {
  if (bal == null && receivedPrl == null) return null;
  const pending = bal ? bal.pending : 0;
  const poolPaid = bal ? bal.paid : 0;
  const paid = receivedPrl != null && receivedPrl > poolPaid ? receivedPrl : poolPaid;
  if (bal && paid === poolPaid) return bal;
  const earned = pending + paid;
  const price = Number(priceUsd);
  const usd = Number.isFinite(price) && price >= 0 ? earned * price : null;
  return { pending, paid, earned, usd };
}

// The pre-relaunch HeroMiners Pearl pool. Since 18 Nov 2026 it keeps each
// address's old statistics and whatever it had not paid yet; once its last
// blocks mature and pay out, it moves the unpaid balances to the new pool.
const OLD_POOL_BASE = 'https://oldpearl.herominers.com';

// Two pool balances added together (the new and old HeroMiners pools). Either
// may be null; null when both are.
function sumBalances(a, b, priceUsd) {
  if (a == null) return b;
  if (b == null) return a;
  const pending = a.pending + b.pending;
  const paid = a.paid + b.paid;
  const earned = pending + paid;
  const price = Number(priceUsd);
  const usd = Number.isFinite(price) && price >= 0 ? earned * price : null;
  return { pending, paid, earned, usd };
}

module.exports = {
  POOL_BASE, COIN_UNITS, buildBalanceUrl, parseBalance, buildPoolStatsUrl, parsePoolPriceUsd,
  buildKryptexBalanceUrl, parseKryptexBalance, buildChainAddressUrl, parseChainReceived, withChainPaid,
  OLD_POOL_BASE, sumBalances,
};
