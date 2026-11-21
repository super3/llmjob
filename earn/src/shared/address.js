'use strict';

// Pearl (PRL) payout addresses are bech32-style strings beginning `prl1p`.
// We only need light validation: the miner/pool rejects a malformed address,
// so this just guards the UI from obviously-wrong input.

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

// A Kryptex account (its email or krx… Mining Username) where the address
// goes. Kryptex paid these until v0.7.8; now every pool needs the Pearl
// address. Recognised only to say so, rather than call it a bad address.
const KRYPTEX_USER_RE = /^krx[0-9a-z]{4,30}$/i;
const EMAIL_RE = /^[^\s@/]+@[^\s@/.]+(\.[^\s@/.]+)+$/;
const KRYPTEX_ACCOUNT_GONE = 'Kryptex accounts can\'t be paid any more. Enter your own Pearl address (prl1p…).';

function isKryptexAccount(login) {
  const s = String(login == null ? '' : login).trim();
  return s.length <= 254 && (KRYPTEX_USER_RE.test(s) || EMAIL_RE.test(s));
}

// The Kryptex message when `settings` (the app's) would mine Pearl to a
// Kryptex account, else null. Quantus has its own address.
function kryptexAccountNote(settings) {
  const s = settings || {};
  return s.coin !== 'quantus' && isKryptexAccount(s.address) ? KRYPTEX_ACCOUNT_GONE : null;
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
  isKryptexAccount, kryptexAccountNote, KRYPTEX_ACCOUNT_GONE,
};
