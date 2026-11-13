'use strict';

// Poseidon2 over the Goldilocks field, as Quantus (QTC) hashes with it: width
// 12, rate 8, capacity 4, 8 full rounds and 22 partial rounds, S-box x^7.
// This is the CPU reference our miner checks itself against. It is written for
// clarity, not speed: every field element is a BigInt.
//
// Ported from qp-poseidon-core 3.1.0 (github.com/Quantus-Network/qp-poseidon,
// src/poseidon2.rs, src/lib.rs, src/serialization.rs), the crate the chain's
// qpow-math calls. The constants below are copied from it. Its licence:
//
//   Copyright 2025 Quantus Network developers
//
//   Permission is hereby granted, free of charge, to any person obtaining a copy
//   of this software and associated documentation files (the "Software"), to
//   deal in the Software without restriction, including without limitation the
//   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
//   sell copies of the Software, and to permit persons to whom the Software is
//   furnished to do so, subject to the following conditions:
//
//   The above copyright notice and this permission notice shall be included in
//   all copies or substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//   IN THE SOFTWARE.

// The Goldilocks prime, 2^64 - 2^32 + 1.
const P = 0xffffffff00000001n;

const WIDTH = 12;
const RATE = 8;
const OUTPUT = 4;

const big = (hex) => BigInt('0x' + hex);

const INTERNAL_CONSTANTS = [
  '97f7798a784ad863', 'd1d2bf082f60d4f0', '69a377a79f9ad206', 'a9d06906a3858e24',
  '295275001eede5b5', '5874e441117bd746', '8a084bbba8ed86cc', '3defd7645cde6425',
  '3998cfe6871cc137', '3e52ef8bca48314a', '964a209f85dc9ecc', '3fcc9ee82cc4577e',
  '8e79b4a5d0096d6d', '8492362ad2392556', 'ee72f470262574d6', '1e0e18496da2444a',
  '0f3a74bf215eaac6', '1b061b76a1c0ded3', '192c42d86803d7a6', 'f6d49ff997ae0260',
  '3ec372e7a0fa3786', '5538cdf4f23445d3',
].map(big);

// The internal layer is state[i] = sum(state) + state[i] * MATRIX_DIAG[i].
const MATRIX_DIAG = [
  'c3b6c08e23ba9300', 'd84b5de94a324fb6', '0d0c371c5b35b84f', '7964f570e7188037',
  '5daf18bbd996604b', '6743bc47b9595257', '5528b9362c59bb70', 'ac45e25b7127b68b',
  'a2077d7dfbb606b5', 'f3faac6faee378ae', '0c6388b51545e883', 'd27dbb6944917b60',
].map(big);

const INITIAL_EXTERNAL_CONSTANTS = [
  ['c002e770975b1607', 'bca51a8dfe14593a', '72938dfbe774f7f9', 'e4f2fe29e03234ac', 'd5e0ba2f541b6449', 'ec33b868f3cc46c1',
    '486dcb55419d475a', '6c1cb2a358cc24f1', 'e3f30d509a1436bb', 'd9a64f068dca7c29', 'e59b3f57aabba1ae', '2a3dd4505b478fdc'],
  ['ada1f8dc7676ed25', '2711aa8b5509d516', '4ae6acd0c9c92897', '56eb3d6b5256d67a', '1f7a9d55923bf51e', '3600427d397a7f68',
    'e5076df75b72c3d0', 'fcd59aa12c6090ad', 'cd895e8c68b57a9e', '41df7ef9d730ae3e', 'ee3e2b889abe977d', 'd29bb7edbeb9c405'],
  ['7d5c08eef608e382', '89ae889caaf0802c', 'b35a8e976d2af617', 'db14234eafaf5173', '78f04462d48b1c98', '265293b0e47ce88a',
    '999a649b69b9d32f', '64b0a186698e01d3', 'ee0b22d0dfae8bb8', '4fd53e50ca04a7ee', '5762bfe181f25047', 'f51593e2beb5e3bd'],
  ['1e5e2b5760e32477', '622462a1f9aaaeed', 'aa284b3ecdb222ae', '63c8e72f542bf3fc', '3ba588cacb43b5e0', '23eda6f3c99150dd',
    'aad3bea4baac9a5a', 'e9da8d699b94184a', 'cdb13f4cd93e024c', '902cbd0956f655e3', '5b4e40ffc759532f', 'de795c20a2357af7'],
].map((row) => row.map(big));

const TERMINAL_EXTERNAL_CONSTANTS = [
  ['7b72c539e0ea4c6e', '144573dae2ce9976', '802028b68f35fc88', '6d36c5022c4fe7c2', 'a205d0ffa9b9def3', 'f6e7e38b1ea6ba2f',
    '34f7909ae5258d64', 'b0464d9d77b97fca', '64ddb9d5de7e00a6', '0ed0d75c27975d97', '1cbb36f11127338b', '6673e505cfd0b6ba'],
  ['605f902830872e01', '3fd5eb927e95fe4f', 'e81025b5a24c69cd', 'f7d0ce75de23f74e', 'f39942b6a8585089', '6d808a08f7b71df6',
    'f8806b6588f49a8b', '57df2d8c2a32107a', '16e7c2074d654a2d', '213de241fcf33835', 'b0f2b8905a0976f6', 'd8e3cf2bbd355417'],
  ['e498691679d9330f', '763b45d2a3821b28', '0908bf65eb0a1f0d', '7691eb2d194b24f4', '0e43551233ae13b2', '93c393dbfc2fe76f',
    '98f607485d48cdea', 'e3d95f30309819c0', '1ef581a93eaf6acf', '0b24c1b7a030fca4', '624370be5670b327', '5f1e28615a11e486'],
  ['fe04051f909e042b', '7257e5b147fd3803', 'e6ae134bb82f2e78', '5711fd5cf4784511', 'f83a42660c08c0bc', '2cd8c96d9a3ce855',
    '7d2ffb1bb0e17271', '85ae1528caea3811', '52a345d5c7adb0b8', '504c4c51f3faee94', 'bce34a649cfccaf9', 'e0a3389266fb6dc9'],
].map((row) => row.map(big));

function add(a, b) {
  return (a + b) % P;
}

function mul(a, b) {
  return (a * b) % P;
}

function exp7(x) {
  const x2 = mul(x, x);
  return mul(mul(x2, x), mul(x2, x2));
}

// The 4x4 matrix [[2,3,1,1],[1,2,3,1],[1,1,2,3],[3,1,1,2]] on s[o..o+4].
function mat4(s, o) {
  const [x0, x1, x2, x3] = [s[o], s[o + 1], s[o + 2], s[o + 3]];
  s[o] = (2n * x0 + 3n * x1 + x2 + x3) % P;
  s[o + 1] = (x0 + 2n * x1 + 3n * x2 + x3) % P;
  s[o + 2] = (x0 + x1 + 2n * x2 + 3n * x3) % P;
  s[o + 3] = (3n * x0 + x1 + x2 + 2n * x3) % P;
}

// M_4 on each group of four, then each element gets the sum of its column
// across the three groups.
function externalLinear(s) {
  for (let o = 0; o < WIDTH; o += 4) mat4(s, o);
  for (let k = 0; k < 4; k++) {
    const sum = s[k] + s[k + 4] + s[k + 8];
    for (let j = k; j < WIDTH; j += 4) s[j] = (s[j] + sum) % P;
  }
}

function internalLinear(s) {
  let sum = 0n;
  for (const x of s) sum += x;
  for (let i = 0; i < WIDTH; i++) s[i] = (sum + s[i] * MATRIX_DIAG[i]) % P;
}

function externalRound(s, rc) {
  for (let i = 0; i < WIDTH; i++) s[i] = exp7(add(s[i], rc[i]));
  externalLinear(s);
}

// The permutation, in place on 12 canonical field elements (BigInts < P).
function permute(s) {
  externalLinear(s);
  for (const rc of INITIAL_EXTERNAL_CONSTANTS) externalRound(s, rc);
  for (const rc of INTERNAL_CONSTANTS) {
    s[0] = exp7(add(s[0], rc));
    internalLinear(s);
  }
  for (const rc of TERMINAL_EXTERNAL_CONSTANTS) externalRound(s, rc);
  return s;
}

// Bytes to field elements, the injective encoding: each 4 bytes is one element
// (little-endian u32), and the last element holds what is left followed by a
// 0x01 byte. An input whose length is a multiple of 4 ends with the element 1.
function bytesToFelts(bytes) {
  const b = Buffer.from(bytes);
  const out = [];
  let pos = 0;
  for (; pos + 4 <= b.length; pos += 4) out.push(BigInt(b.readUInt32LE(pos)));
  const last = Buffer.alloc(4);
  b.copy(last, 0, pos);
  last[b.length - pos] = 1;
  out.push(BigInt(last.readUInt32LE(0)));
  return out;
}

// The sponge: absorb the elements 8 at a time (added into state[0..8], then the
// permutation), then pad with a 1 and zeros to a full block.
function absorb(felts) {
  const s = new Array(WIDTH).fill(0n);
  const all = felts.concat([1n]);
  while (all.length % RATE) all.push(0n);
  for (let o = 0; o < all.length; o += RATE) {
    for (let i = 0; i < RATE; i++) s[i] = add(s[i], all[o + i]);
    permute(s);
  }
  return s;
}

// A digest is state[0..4], each element 8 bytes little-endian.
function digestBytes(s) {
  const out = Buffer.alloc(OUTPUT * 8);
  for (let i = 0; i < OUTPUT; i++) out.writeBigUInt64LE(s[i], i * 8);
  return out;
}

// qp-poseidon-core's hash_bytes: 32 bytes.
function hashBytes(bytes) {
  return digestBytes(absorb(bytesToFelts(bytes)));
}

// qp-poseidon-core's hash_squeeze_twice: the 32-byte digest, then the
// permutation once more and the next 32. The first half equals hashBytes.
function hashSqueezeTwice(bytes) {
  const s = absorb(bytesToFelts(bytes));
  const first = digestBytes(s);
  return Buffer.concat([first, digestBytes(permute(s))]);
}

module.exports = {
  P, WIDTH, RATE, OUTPUT, INTERNAL_CONSTANTS, MATRIX_DIAG, INITIAL_EXTERNAL_CONSTANTS, TERMINAL_EXTERNAL_CONSTANTS,
  add, mul, exp7, externalLinear, internalLinear, permute, bytesToFelts, absorb, digestBytes, hashBytes, hashSqueezeTwice,
};
