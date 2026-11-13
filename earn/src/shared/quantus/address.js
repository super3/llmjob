'use strict';

// Quantus (QTC) addresses: SS58, the Substrate format, with network prefix 189,
// so they start "qz" (docs.quantus.com, deep-dives/qpow.md, "SS58 prefix 189").
// The pools take one as the login: Kryptex's example is qz… (49 characters).
//
// SS58: base58 of [2 prefix bytes][32-byte account][2 checksum bytes], where
// the checksum is the first two bytes of BLAKE2b-512("SS58PRE" || prefix ||
// account). Checked with Node's BLAKE2b, so a typo is caught before a pool
// would refuse the login.

const crypto = require('crypto');

const PREFIX = 189;
const ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz';
// Two prefix bytes (a prefix of 64 or more takes two), the account, the checksum.
const DECODED_BYTES = 2 + 32 + 2;

function base58Decode(s) {
  let n = 0n;
  for (const ch of s) {
    const v = ALPHABET.indexOf(ch);
    if (v < 0) return null;
    n = n * 58n + BigInt(v);
  }
  let hex = n === 0n ? '' : n.toString(16);
  if (hex.length % 2) hex = '0' + hex;
  const zeros = s.length - s.replace(/^1+/, '').length;
  return Buffer.concat([Buffer.alloc(zeros), Buffer.from(hex, 'hex')]);
}

function base58Encode(buf) {
  let n = BigInt('0x' + (buf.toString('hex') || '0'));
  let out = '';
  while (n > 0n) {
    out = ALPHABET[Number(n % 58n)] + out;
    n /= 58n;
  }
  for (const b of buf) {
    if (b !== 0) break;
    out = '1' + out;
  }
  return out;
}

// The two bytes SS58 writes a prefix from 64 to 16383 as.
function prefixBytes(ident) {
  return Buffer.from([((ident & 0xfc) >> 2) | 0x40, (ident >> 8) | ((ident & 0x03) << 6)]);
}

function checksum(body) {
  return crypto.createHash('blake2b512').update(Buffer.concat([Buffer.from('SS58PRE'), body])).digest().subarray(0, 2);
}

// { ok: true } or { ok: false, error } saying what is wrong, for the CLI.
function checkQuantusAddress(address) {
  const a = String(address == null ? '' : address).trim();
  if (!a) return { ok: false, error: 'empty' };
  if (!a.startsWith('qz')) return { ok: false, error: 'a Quantus address starts with qz' };
  const raw = base58Decode(a);
  if (!raw) return { ok: false, error: 'not base58' };
  if (raw.length !== DECODED_BYTES) return { ok: false, error: 'wrong length' };
  if (!raw.subarray(0, 2).equals(prefixBytes(PREFIX))) return { ok: false, error: 'not a Quantus network address' };
  if (!raw.subarray(34).equals(checksum(raw.subarray(0, 34)))) return { ok: false, error: 'bad checksum (a typo?)' };
  return { ok: true };
}

function isValidQuantusAddress(address) {
  return checkQuantusAddress(address).ok;
}

// The address for a 32-byte account. For tests and tools: the miner only checks.
function encodeQuantusAddress(account) {
  if (!Buffer.isBuffer(account) || account.length !== 32) throw new Error('an account is 32 bytes');
  const body = Buffer.concat([prefixBytes(PREFIX), account]);
  return base58Encode(Buffer.concat([body, checksum(body)]));
}

// For logs: the start and the end, as shortenAddress does for Pearl.
function shortenQuantusAddress(address) {
  const a = String(address == null ? '' : address).trim();
  return a.length > 16 ? a.slice(0, 8) + '…' + a.slice(-6) : a;
}

module.exports = {
  PREFIX, checkQuantusAddress, isValidQuantusAddress, encodeQuantusAddress, shortenQuantusAddress, base58Decode, base58Encode,
};
