'use strict';

// BLAKE2b-512 (RFC 7693), unkeyed, in plain JavaScript. A Quantus address's
// checksum is BLAKE2b-512, and Node's crypto has it, but Electron's does not:
// Electron builds Node against BoringSSL, where createHash('blake2b512') throws
// "Digest method not supported". So the desktop app could not check an
// address, and a START with Quantus picked ran nothing (19 Nov 2026). The CLI
// ran on Node and never saw it. This is slow next to OpenSSL's, which doesn't
// matter for an address of 36 bytes. It is checked against Node's in the tests.

const MASK = (1n << 64n) - 1n;

const IV = [
  0x6a09e667f3bcc908n, 0xbb67ae8584caa73bn, 0x3c6ef372fe94f82bn, 0xa54ff53a5f1d36f1n,
  0x510e527fade682d1n, 0x9b05688c2b3e6c1fn, 0x1f83d9abfb41bd6bn, 0x5be0cd19137e2179n,
];

const SIGMA = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
  [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
  [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
  [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
  [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
  [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
  [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
  [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
  [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

const BLOCK = 128;

function rotr(x, n) {
  return ((x >> n) | (x << (64n - n))) & MASK;
}

function mix(v, a, b, c, d, x, y) {
  v[a] = (v[a] + v[b] + x) & MASK;
  v[d] = rotr(v[d] ^ v[a], 32n);
  v[c] = (v[c] + v[d]) & MASK;
  v[b] = rotr(v[b] ^ v[c], 24n);
  v[a] = (v[a] + v[b] + y) & MASK;
  v[d] = rotr(v[d] ^ v[a], 16n);
  v[c] = (v[c] + v[d]) & MASK;
  v[b] = rotr(v[b] ^ v[c], 63n);
}

// One 128-byte block into the state. `t` is the bytes hashed so far, this
// block included; `last` marks the final block.
function compress(h, block, t, last) {
  const m = [];
  for (let i = 0; i < 16; i++) m.push(block.readBigUInt64LE(i * 8));
  const v = h.concat(IV);
  v[12] ^= t & MASK;
  v[13] ^= t >> 64n;
  if (last) v[14] ^= MASK;
  for (let r = 0; r < 12; r++) {
    const s = SIGMA[r % 10];
    mix(v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
    mix(v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
    mix(v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
    mix(v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
    mix(v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
    mix(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
    mix(v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
    mix(v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
  }
  for (let i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i + 8];
}

// The 64-byte digest of a Buffer.
function blake2b512(data) {
  const h = IV.slice();
  // Parameter block: 64-byte digest, no key, fanout 1, depth 1.
  h[0] ^= 0x01010040n;
  // Every block but the last; the last, padded with zeros, is the final one
  // even when the input fills it (and an empty input is one block of zeros).
  const blocks = Math.max(1, Math.ceil(data.length / BLOCK));
  for (let i = 0; i < blocks; i++) {
    const block = Buffer.alloc(BLOCK);
    data.copy(block, 0, i * BLOCK, Math.min(data.length, (i + 1) * BLOCK));
    const last = i === blocks - 1;
    compress(h, block, BigInt(last ? data.length : (i + 1) * BLOCK), last);
  }
  const out = Buffer.alloc(64);
  h.forEach((w, i) => out.writeBigUInt64LE(w, i * 8));
  return out;
}

module.exports = { blake2b512 };
