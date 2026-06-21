'use strict';

// Pearl (PRL) payout addresses are bech32-style strings beginning `prl1p`.
// We only need light validation: the miner/pool rejects a malformed address,
// so this just guards the UI from obviously-wrong input.

const ADDRESS_RE = /^prl1p[0-9a-z]{20,80}$/;

function normalizeAddress(addr) {
  return String(addr == null ? '' : addr).trim().toLowerCase();
}

function isValidAddress(addr) {
  return ADDRESS_RE.test(normalizeAddress(addr));
}

// Compact form for tables/log lines, e.g. `prl1pql8…d4p6c`.
function shortenAddress(addr, head = 8, tail = 5) {
  const s = normalizeAddress(addr);
  if (s.length <= head + tail + 1) return s;
  return s.slice(0, head) + '…' + s.slice(-tail);
}

module.exports = { ADDRESS_RE, normalizeAddress, isValidAddress, shortenAddress };
