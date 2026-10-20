'use strict';

// Monero payout addresses.
//
// A Monero address is base58 in Monero's own block form: the bytes are split
// into 8-byte blocks, each written as 11 characters, and a short last block
// takes fewer (see BLOCK_CHARS). It is not Bitcoin's base58, which encodes the
// whole number at once. Decoded, an address is
//
//   network byte | 32-byte spend key | 32-byte view key | 4-byte checksum
//
// where the checksum is the first 4 bytes of Keccak-256 over everything before
// it. Mainnet uses network byte 18 for a standard address (95 characters,
// starts with 4) and 42 for a subaddress (95, starts with 8). Byte 19 is an
// integrated address (106 characters, an 8-byte payment ID before the
// checksum). We take standard addresses and subaddresses only: pools use "."
// after an address for a payment ID, and an exchange deposit that needs one
// has a high minimum and is not what the app's single payout field is for.

const { keccak256 } = require('./keccak');

const ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz';
// Characters per block, indexed by the block's byte length (0-8).
const BLOCK_CHARS = [0, 2, 3, 5, 6, 7, 9, 10, 11];
const FULL_BLOCK_BYTES = 8;
const FULL_BLOCK_CHARS = 11;

const NETWORKS = {
  18: 'standard',
  42: 'subaddress',
  19: 'integrated',
  // Testnet and stagenet, recognised only to say so.
  53: 'testnet', 63: 'testnet', 54: 'testnet',
  24: 'stagenet', 36: 'stagenet', 25: 'stagenet',
};

const DIGITS = new Map(Array.from(ALPHABET, (ch, i) => [ch, BigInt(i)]));

// One block of up to 11 characters to its bytes, or null when it isn't valid.
function decodeBlock(chars) {
  const size = BLOCK_CHARS.indexOf(chars.length);
  if (size <= 0) return null;
  let n = 0n;
  for (const ch of chars) {
    const d = DIGITS.get(ch);
    if (d === undefined) return null;
    n = n * 58n + d;
  }
  if (n >= 1n << BigInt(8 * size)) return null;
  const out = Buffer.alloc(size);
  for (let i = size - 1; i >= 0; i--) {
    out[i] = Number(n & 0xffn);
    n >>= 8n;
  }
  return out;
}

// Monero base58 to bytes, or null when it isn't valid.
function decodeBase58(text) {
  const s = String(text == null ? '' : text);
  if (!s) return null;
  // Slicing by 11 leaves only the last block short.
  const parts = [];
  for (let i = 0; i < s.length; i += FULL_BLOCK_CHARS) {
    const block = decodeBlock(s.slice(i, i + FULL_BLOCK_CHARS));
    if (!block) return null;
    parts.push(block);
  }
  return Buffer.concat(parts);
}

// Bytes to Monero base58. Only the tests need it, to build addresses of every
// kind; the app never writes an address.
function encodeBase58(bytes) {
  const data = Buffer.from(bytes);
  let out = '';
  for (let i = 0; i < data.length; i += FULL_BLOCK_BYTES) {
    const block = data.subarray(i, i + FULL_BLOCK_BYTES);
    let n = 0n;
    for (const b of block) n = (n << 8n) | BigInt(b);
    let chars = '';
    for (let k = 0; k < BLOCK_CHARS[block.length]; k++) {
      chars = ALPHABET[Number(n % 58n)] + chars;
      n /= 58n;
    }
    out += chars;
  }
  return out;
}

// What kind of address this is, or why it can't be used:
//   { ok: true, kind: 'standard' | 'subaddress' }
//   { ok: false, error: '…' }
function checkXmrAddress(addr) {
  const s = String(addr == null ? '' : addr).trim();
  if (!s) return { ok: false, error: 'enter a Monero address' };
  const bytes = decodeBase58(s);
  if (!bytes || bytes.length < 5) return { ok: false, error: 'not a Monero address' };
  const body = bytes.subarray(0, bytes.length - 4);
  const sum = bytes.subarray(bytes.length - 4);
  if (!keccak256(body).subarray(0, 4).equals(sum)) {
    return { ok: false, error: 'the checksum does not match: check for a typo' };
  }
  const kind = NETWORKS[bytes[0]];
  if (kind === 'integrated') {
    return { ok: false, error: 'integrated addresses (with a payment ID) are not supported: use your wallet\'s main address or a subaddress' };
  }
  if (kind === 'testnet' || kind === 'stagenet') {
    return { ok: false, error: 'this is a ' + kind + ' address, not a mainnet one' };
  }
  // A standard address or subaddress is 1 + 32 + 32 + 4 bytes.
  if (!kind || bytes.length !== 69) return { ok: false, error: 'not a Monero address' };
  return { ok: true, kind };
}

// Compact form for log lines, e.g. `87wTMj88…1JxND9`. Not lowercased: base58
// is case-sensitive.
function shortenXmrAddress(addr, head = 8, tail = 6) {
  const s = String(addr == null ? '' : addr).trim();
  if (s.length <= head + tail + 1) return s;
  return s.slice(0, head) + '…' + s.slice(-tail);
}

module.exports = {
  ALPHABET, BLOCK_CHARS, decodeBase58, encodeBase58, checkXmrAddress, shortenXmrAddress,
};
