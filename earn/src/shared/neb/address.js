'use strict';

// NEB addresses: bech32m (BIP-350) with the prefix "neb" over the 32 bytes of
// a single-key lock, with no witness-version character, so 62 characters that
// start "neb1" (spec/node.md section 5.2 of github.com/SilasEigen/neb). Pools
// take one as the login. Checked here so a typo is caught before a pool
// refuses it.

const HRP = 'neb';
const CHARSET = 'qpzry9x8gf2tvdw0s3jn54khce6mua7l';
const BECH32M = 0x2bc830a3;
const GEN = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3];
const ADDRESS_LENGTH = 62;

function polymod(values) {
  let chk = 1;
  for (const v of values) {
    const top = chk >>> 25;
    chk = ((chk & 0x1ffffff) << 5) ^ v;
    for (let i = 0; i < 5; i++) if ((top >>> i) & 1) chk ^= GEN[i];
  }
  return chk >>> 0;
}

function hrpExpand(hrp) {
  const out = [];
  for (const c of hrp) out.push(c.charCodeAt(0) >> 5);
  out.push(0);
  for (const c of hrp) out.push(c.charCodeAt(0) & 31);
  return out;
}

// The 32-byte lock of an address, or null when it is not a NEB address.
function decodeNebAddress(address) {
  const s = String(address == null ? '' : address).trim();
  if (s.length !== ADDRESS_LENGTH || s !== s.toLowerCase() || !s.startsWith(HRP + '1')) return null;
  const data = [];
  for (const c of s.slice(HRP.length + 1)) {
    const v = CHARSET.indexOf(c);
    if (v < 0) return null;
    data.push(v);
  }
  if (polymod(hrpExpand(HRP).concat(data)) !== BECH32M) return null;
  // 5-bit groups to bytes: 52 groups are 32 bytes and 4 padding bits, which must be zero.
  const out = [];
  let acc = 0;
  let bits = 0;
  for (const v of data.slice(0, -6)) {
    acc = ((acc << 5) | v) & 0xfff;
    bits += 5;
    if (bits >= 8) {
      bits -= 8;
      out.push((acc >> bits) & 0xff);
    }
  }
  if ((acc & ((1 << bits) - 1)) !== 0) return null;
  return Buffer.from(out);
}

// The address of a list of 5-bit groups: the prefix, the groups, their checksum.
function encodeGroups(data) {
  const mod = polymod(hrpExpand(HRP).concat(data, [0, 0, 0, 0, 0, 0])) ^ BECH32M;
  const sum = [];
  for (let i = 0; i < 6; i++) sum.push((mod >>> (5 * (5 - i))) & 31);
  return HRP + '1' + data.concat(sum).map((v) => CHARSET[v]).join('');
}

// The address of a 32-byte lock.
function encodeNebAddress(lock) {
  const data = [];
  let acc = 0;
  let bits = 0;
  for (const b of lock) {
    acc = ((acc << 8) | b) & 0xfff;
    bits += 8;
    while (bits >= 5) {
      bits -= 5;
      data.push((acc >> bits) & 31);
    }
  }
  // 256 bits leave one over: the last group is that bit and four zero bits.
  data.push((acc << (5 - bits)) & 31);
  return encodeGroups(data);
}

function isNebAddress(address) {
  return decodeNebAddress(address) != null;
}

// For log lines: neb1kwgy…vwr4n.
function shortenNebAddress(address) {
  const s = String(address == null ? '' : address).trim();
  return s.length > 16 ? s.slice(0, 8) + '…' + s.slice(-5) : s;
}

module.exports = { ADDRESS_LENGTH, CHARSET, decodeNebAddress, encodeGroups, encodeNebAddress, isNebAddress, shortenNebAddress };
