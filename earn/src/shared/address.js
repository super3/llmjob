'use strict';

// Pearl (PRL) payout addresses are bech32-style strings beginning `prl1p`.
// We only need light validation: the miner/pool rejects a malformed address,
// so this just guards the UI from obviously-wrong input.

const { isKryptexHost } = require('./miner/stratum');

const ADDRESS_RE = /^prl1p[0-9a-z]{20,80}$/;
// ModelOS (MDL) merge-mining payout addresses share the bech32-style shape but
// begin `mdl1p`. AlphaPool grades every Pearl share against MDL too (AuxPoW), so
// a rig can earn MDL on the same hashrate by appending its MDL address.
const MDL_ADDRESS_RE = /^mdl1p[0-9a-z]{20,80}$/;

function normalizeAddress(addr) {
  return String(addr == null ? '' : addr).trim().toLowerCase();
}

function isValidAddress(addr) {
  return ADDRESS_RE.test(normalizeAddress(addr));
}

function isValidMdlAddress(addr) {
  return MDL_ADDRESS_RE.test(normalizeAddress(addr));
}

// Merge mining is enabled by handing alpha-miner a combined `prl1…+mdl1…` as its
// --address. Append the MDL address only when it's well-formed, so a typo can
// never corrupt the Pearl address and break mining outright (Pearl-only still
// works — merge mining is purely additive).
function combinePayoutAddress(prl, mdl) {
  const p = String(prl == null ? '' : prl).trim();
  const m = normalizeAddress(mdl);
  return p && isValidMdlAddress(m) ? p + '+' + m : p;
}

// Kryptex also pays its own accounts ("account mining", issue #303): the
// account's email or its Mining Username (krx…) goes where the address goes,
// and the login is `account/worker`, the same as an address's on Kryptex
// (pool.kryptex.com/articles/account-mining-en). No other pool here does this,
// and HeroMiners would take an email for an address it can't pay.
const KRYPTEX_USER_RE = /^krx[0-9a-z]{4,30}$/i;
const EMAIL_RE = /^[^\s@/]+@[^\s@/.]+(\.[^\s@/.]+)+$/;

function isKryptexAccount(login) {
  const s = String(login == null ? '' : login).trim();
  return s.length <= 254 && (KRYPTEX_USER_RE.test(s) || EMAIL_RE.test(s));
}

// Why `login` can't mine Pearl on the pool at `host`, or null when it can: a
// Pearl address anywhere, or a Kryptex account on a Kryptex host. The messages
// start the way cliArgs' errors always have (shared/hiveos reads them).
function payoutProblem(login, host) {
  if (isValidAddress(login)) return null;
  const s = String(login == null ? '' : login).trim();
  if (!isKryptexAccount(s)) return 'invalid Pearl address: ' + s;
  if (isKryptexHost(host)) return null;
  return 'a Kryptex account works only on a Kryptex pool: ' + s;
}

// Compact form for tables/log lines, e.g. `prl1pql8…d4p6c`.
function shortenAddress(addr, head = 8, tail = 5) {
  const s = normalizeAddress(addr);
  if (s.length <= head + tail + 1) return s;
  return s.slice(0, head) + '…' + s.slice(-tail);
}

module.exports = {
  normalizeAddress,
  isValidAddress, isValidMdlAddress, combinePayoutAddress, shortenAddress,
  isKryptexAccount, payoutProblem,
};
