'use strict';

// Pool balance lookup for a payout address. HeroMiners exposes an
// unauthenticated per-miner endpoint, GET /api/stats_address?address=<address>,
// which returns a `stats` object alongside payment history, per-worker rows and
// chart data. The actual HTTPS GET runs in the main process (no CORS/CSP there);
// this module just builds the URL and parses the response so both are
// unit-testable.
//
// `longpoll=false` matters: without it the endpoint holds the connection open
// waiting for a share update, which is fine for a dashboard that wants a live
// push and wrong for a poller that wants an answer now.

const POOL_BASE = 'https://pearl.herominers.com';

// The pool reports amounts in atomic units — its own /api/stats config declares
// coinUnits 100000000, i.e. 1 PRL = 1e8.
const COIN_UNITS = 1e8;

// Field names, most specific first. The endpoint omits balance fields entirely
// for an address that has never been credited — verified against a freshly-mined
// address, whose `stats` carried shares_good/hashrate/payments_24h and no
// balance key at all — so the shape for a FUNDED address could not be observed
// directly and these are the candidate spellings the pool software is known to
// use. Reading several and defaulting to zero means the worst case is a balance
// that reads 0 until someone with a real one reports it, never a wrong number.
// If balances show as zero for a paid address, this list is the first place to
// look.
const PENDING_KEYS = ['balance', 'pending', 'unpaid'];
const PAID_KEYS = ['paid', 'total_paid', 'totalPaid'];

function buildBalanceUrl(address, base) {
  const a = String(address == null ? '' : address).trim();
  return (base || POOL_BASE) + '/api/stats_address?address=' + encodeURIComponent(a)
    + '&longpoll=false';
}

// First finite, non-negative value among `keys`, in PRL. Absent everywhere → 0.
function coinField(stats, keys) {
  for (const k of keys) {
    const n = Number(stats[k]);
    if (Number.isFinite(n) && n >= 0) return n / COIN_UNITS;
  }
  return 0;
}

// Reduce the pool payload to the display fields, or null when it's unusable.
// `earned` is the balance we show: pending payout plus lifetime paid — i.e.
// everything the address has earned.
// priceUsd (optional) converts the total to USD; omit it to show the coin only.
function parseBalance(json, priceUsd) {
  if (!json || typeof json !== 'object' || Array.isArray(json)) return null;
  const stats = json.stats;
  if (!stats || typeof stats !== 'object' || Array.isArray(stats)) return null;
  const pending = coinField(stats, PENDING_KEYS);
  const paid = coinField(stats, PAID_KEYS);
  const earned = pending + paid;
  const price = Number(priceUsd);
  const usd = Number.isFinite(price) && price >= 0 ? earned * price : null;
  return { pending, paid, earned, usd };
}

module.exports = { POOL_BASE, COIN_UNITS, buildBalanceUrl, parseBalance };
