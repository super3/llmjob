'use strict';

// Keccak-256, the hash Monero uses for its address checksum.
//
// Node's crypto has SHA3-256 but not Keccak-256. The two share the permutation
// and differ only in the padding byte (0x06 for SHA3, 0x01 for the original
// Keccak that Monero adopted before SHA3 was standardised), so the tests check
// this permutation against Node's sha3-256 with the padding switched.
//
// Lanes are BigInts. Address checks hash 65 bytes a few times per edit, so
// speed doesn't matter here and the plain form is easier to check.

const MASK = (1n << 64n) - 1n;
const RATE = 136; // bytes: 1600-bit state minus twice the 256-bit output

const ROUND_CONSTANTS = [
  0x0000000000000001n, 0x0000000000008082n, 0x800000000000808an, 0x8000000080008000n,
  0x000000000000808bn, 0x0000000080000001n, 0x8000000080008081n, 0x8000000000008009n,
  0x000000000000008an, 0x0000000000000088n, 0x0000000080008009n, 0x000000008000000an,
  0x000000008000808bn, 0x800000000000008bn, 0x8000000000008089n, 0x8000000000008003n,
  0x8000000000008002n, 0x8000000000000080n, 0x000000000000800an, 0x800000008000000an,
  0x8000000080008081n, 0x8000000000008080n, 0x0000000080000001n, 0x8000000080008008n,
];

// Rotation offsets, indexed [x + 5y].
const ROTATIONS = [
  0, 1, 62, 28, 27,
  36, 44, 6, 55, 20,
  3, 10, 43, 25, 39,
  41, 45, 15, 21, 8,
  18, 2, 61, 56, 14,
];

function rotl(v, n) {
  if (n === 0) return v;
  const s = BigInt(n);
  return ((v << s) | (v >> (64n - s))) & MASK;
}

function permute(state) {
  const c = new Array(5);
  const b = new Array(25);
  for (let round = 0; round < 24; round++) {
    // theta
    for (let x = 0; x < 5; x++) {
      c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
    }
    for (let x = 0; x < 5; x++) {
      const d = c[(x + 4) % 5] ^ rotl(c[(x + 1) % 5], 1);
      for (let y = 0; y < 25; y += 5) state[x + y] ^= d;
    }
    // rho and pi
    for (let x = 0; x < 5; x++) {
      for (let y = 0; y < 5; y++) {
        b[y + 5 * ((2 * x + 3 * y) % 5)] = rotl(state[x + 5 * y], ROTATIONS[x + 5 * y]);
      }
    }
    // chi
    for (let y = 0; y < 25; y += 5) {
      for (let x = 0; x < 5; x++) {
        state[x + y] = b[x + y] ^ (~b[((x + 1) % 5) + y] & MASK & b[((x + 2) % 5) + y]);
      }
    }
    // iota
    state[0] ^= ROUND_CONSTANTS[round];
  }
}

// The sponge, with the padding's domain byte as a parameter (0x01 Keccak,
// 0x06 SHA3). Returns 32 bytes.
function sponge(input, domain) {
  const data = Buffer.from(input);
  const blocks = Math.floor(data.length / RATE) + 1;
  const padded = Buffer.alloc(blocks * RATE);
  data.copy(padded);
  padded[data.length] ^= domain;
  padded[padded.length - 1] ^= 0x80;

  const state = new Array(25).fill(0n);
  for (let off = 0; off < padded.length; off += RATE) {
    for (let i = 0; i < RATE / 8; i++) state[i] ^= padded.readBigUInt64LE(off + 8 * i);
    permute(state);
  }
  const out = Buffer.alloc(32);
  for (let i = 0; i < 4; i++) out.writeBigUInt64LE(state[i], 8 * i);
  return out;
}

function keccak256(input) {
  return sponge(input, 0x01);
}

module.exports = { keccak256, sponge };
