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

// Merge-mined MDL lives on the PRL miner's record — GET /api/miner/<prl…>/mdl —
// not under the mdl1… address (the pool 400s those). The response is
// { has_mdl, mdl_address, summary: { pending_mdl, total_paid_mdl, … },
// recent_payouts: […] }.
function buildMdlBalanceUrl(prlAddress, base) {
  return buildBalanceUrl(prlAddress, base) + '/mdl';
}

// Reduce the merge-mining payload to the display fields, or null when the
// payload is unusable or the address has no MDL pairing. Totals come from
// `summary` — recent_payouts is a capped window and undercounts lifetime paid.
// `mdlAddress` echoes the pool's linked mdl1… address so the UI can cross-check
// it against the address in Settings.
function parseMdlBalance(json) {
  if (!json || typeof json !== 'object' || Array.isArray(json)) return null;
  if (!json.has_mdl) return null;
  const s = json.summary;
  if (!s || typeof s !== 'object' || Array.isArray(s)) return null;
  const pending = Number(s.pending_mdl);
  if (!Number.isFinite(pending) || pending < 0) return null;
  const paidRaw = Number(s.total_paid_mdl);
  const paid = Number.isFinite(paidRaw) && paidRaw >= 0 ? paidRaw : 0;
  return { pending, paid, earned: pending + paid, usd: null, mdlAddress: String(json.mdl_address || '') };
}

module.exports = { POOL_BASE, buildBalanceUrl, parseBalance, buildMdlBalanceUrl, parseMdlBalance };
