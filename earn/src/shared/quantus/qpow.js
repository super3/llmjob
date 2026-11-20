'use strict';

// Quantus's proof of work (QPoW), from the chain's qpow-math crate
// (github.com/Quantus-Network/chain, qpow-math/src/lib.rs) and QIP-0013:
//
//   hash  = Poseidon2 hash_squeeze_twice(header (32 bytes) || nonce (64 bytes)),
//           64 bytes read as one big-endian 512-bit number
//   valid = hash < (2^512 - 1) / difficulty
//
// The 96 input bytes are 24 field elements and a 25th that is the encoding's
// terminator (1), so the sponge runs four permutations to absorb them and a
// fifth for the second squeeze:
//
//   block 1: the header                 the same for a whole job
//   block 2: nonce bytes 0..31          the same for one card's job
//   block 3: nonce bytes 32..63         changes with every nonce
//   block 4: [1, 1, 0 x 6]              the terminator and the sponge's pad
//   squeeze: state[0..4], permutation, state[0..4]
//
// A miner keeps the state after block 2 (the midstate), so a nonce costs the
// permutations of blocks 3 and 4. The hash's top 64 bits are state[0] after
// block 4, byte-swapped (its little-endian bytes come first in the big-endian
// number). A nonce whose top 64 bits are above the target's cannot be a share,
// so the fifth permutation is needed only for the rare nonce that ties or wins.
// The quantus-miner pow-core crate's MiningHasher does the same.
//
// This is the reference every GPU core is checked against. The nonce layout
// our miner uses, and what it saves a core, is in the second half of the file.

const {
  P, WIDTH, INITIAL_EXTERNAL_CONSTANTS, permute, permuteFromRound2, add, exp7, externalLinear, hashSqueezeTwice,
} = require('./poseidon2');

const HEADER_BYTES = 32;
const NONCE_BYTES = 64;
const HASH_BYTES = 64;
const MAX_TARGET = (1n << 512n) - 1n;
const U64_MAX = (1n << 64n) - 1n;

function need(buf, bytes, what) {
  if (!Buffer.isBuffer(buf) || buf.length !== bytes) throw new Error(what + ' must be ' + bytes + ' bytes');
}

// The full 64-byte hash of one header and nonce.
function nonceHash(header, nonce) {
  need(header, HEADER_BYTES, 'header');
  need(nonce, NONCE_BYTES, 'nonce');
  return hashSqueezeTwice(Buffer.concat([header, nonce]));
}

function toBig(buf) {
  return BigInt('0x' + buf.toString('hex'));
}

// The target a difficulty stands for. Difficulty 0 has none.
function targetFromDifficulty(difficulty) {
  const d = BigInt(difficulty);
  if (d <= 0n) throw new Error('difficulty must be at least 1');
  return MAX_TARGET / d;
}

// How many hashes a share at this target is worth on average.
function difficultyFromTarget(target) {
  return target > 0n ? MAX_TARGET / target : MAX_TARGET;
}

// A share: strictly below the target.
function meetsTarget(hash, target) {
  return toBig(hash) < target;
}

// The target's top 64 bits: what a core compares a nonce's top 64 bits with.
function targetHigh(target) {
  return target >> 448n;
}

// Absorb 8 little-endian u32 words of `bytes` into state[0..8].
function absorbWords(s, bytes) {
  for (let i = 0; i < bytes.length / 4; i++) s[i] = add(s[i], BigInt(bytes.readUInt32LE(i * 4)));
}

// The state after blocks 1 and 2: 12 canonical field elements.
function midstate(header, nonceHigh) {
  need(header, HEADER_BYTES, 'header');
  need(nonceHigh, 32, 'nonce high half');
  const s = new Array(WIDTH).fill(0n);
  absorbWords(s, header);
  permute(s);
  absorbWords(s, nonceHigh);
  return permute(s);
}

// ── The nonce layout (version 2) ────────────────────────────────────────────
//
//   bytes 0..23   the pool's extranonce, then zeros   } in the midstate:
//   bytes 24..31  the card's lane (8 bytes)           } one per card and job
//   bytes 32..63  block 3: eight little-endian u32 words w[0..8]
//
// The lane keeps two cards, and two runs of the miner, off each other's
// nonces. A pool may ban a miner that sends the same share twice.
//
// A core does not count through block 3. It walks a line:
//
//   w[i] = base[i] + STEP[i] * t        t = 0, 1, 2, ...
//
// STEP is chosen so that most of the first round does not depend on t. Block
// 3 is added to state[0..8], and the permutation starts with the external
// linear layer E (M4 on each group of four, then each group plus the sum of
// the three). With u = (-4, 17, -11, 3), M4 u = (35, 0, 0, 0). STEP is u in
// group 0 and -u in group 1, so the two groups' M4 outputs cancel in the sum
// and E(STEP) = (35, 0, 0, 0, -35, 0, 0, 0, 0, 0, 0, 0). After the first
// layer only elements 0 and 4 move with t:
//
//   x0 = X0 + 35 t        x4 = X4 - 35 t        (mod p)
//
// The other ten first-round S-boxes are the same for the whole line, so the
// host works them out once (lineWork). The first round's linear layer is
// linear too: with a = x0^7 and b = x4^7, its output is
//
//   s[4g + k] = C[4g + k] + M[k] * (a * (g == 0 ? 2 : 1) + b * (g == 1 ? 2 : 1))
//
// with M = (2, 1, 1, 3) (M4's first column) and C the layer's output for
// a = b = 0, the second round's constants added. A core does two S-boxes and
// those twelve sums, then the permutation from its second round's S-boxes
// (indexState, indexTop64). That is 10 of a nonce's 236 S-boxes and most of
// one of its 17 linear layers saved.
//
// Every word must stay a u32, so a line has SPAN points: STEP's largest
// factor is 17. Lines are numbered: a line's number is in words 0 and 4 and
// the other six words start at their end of the range (lineBase). Two points
// of two lines are never the same nonce: word 1 is 17 t, which gives t, and
// then words 0 and 4 give the line.

const LANE_BYTES = 8;
const LANE_OFFSET = 24;
const BLOCK3_OFFSET = 32;
const STEP = [-4, 17, -11, 3, 4, -17, 11, -3];
// What one step of t adds to element 0, and takes from element 4, after the
// first linear layer.
const SLOPE = 35n;
// Points on a line: the most t for which 17 t is still a u32, plus one.
const SPAN = Math.floor(0xffffffff / 17) + 1;
// Lines a lane has: 31 bits in word 0 and 31 in word 4.
const LINES = 1n << 62n;
// M4's first column.
const COLUMN = [2n, 1n, 1n, 3n];
const LINE_ELEMENTS = WIDTH + 2;
const LINE_BYTES = LINE_ELEMENTS * 8;

// Line `line`'s eight words at t = 0. Words 0 and 4 hold the line's number;
// each word has room for SPAN steps of its own STEP.
function lineBase(line) {
  const n = BigInt(line);
  if (n < 0n || n >= LINES) throw new Error('line must be 0 to 2^62 - 1');
  const lo = Number(n & 0x7fffffffn);
  const hi = Number(n >> 31n);
  return [0x80000000 + lo, 0, 0xffffffff, 0, hi, 0xffffffff, 0, 0xffffffff];
}

// The words `t` steps along a line from `base`.
function lineWords(base, t) {
  if (!Number.isInteger(t) || t < 0) throw new Error('t must be a whole number, 0 or more');
  return base.map((w, i) => {
    const v = w + STEP[i] * t;
    if (v < 0 || v > 0xffffffff) throw new Error('t = ' + t + ' takes word ' + i + ' out of a u32');
    return v;
  });
}

// The nonce for point `t` of line `line` in this lane.
function buildNonce(extranonce, lane, line, t) {
  const prefix = extranonce || Buffer.alloc(0);
  if (prefix.length > LANE_OFFSET) {
    throw new Error('an extranonce of more than ' + LANE_OFFSET + ' bytes leaves no room for the lane');
  }
  need(lane, LANE_BYTES, 'lane');
  const nonce = Buffer.alloc(NONCE_BYTES);
  prefix.copy(nonce, 0);
  lane.copy(nonce, LANE_OFFSET);
  lineWords(lineBase(line), t).forEach((w, i) => nonce.writeUInt32LE(w, BLOCK3_OFFSET + 4 * i));
  return nonce;
}

// Any nonce as a midstate's half and a line's base (its own point 0).
function splitNonce(nonce) {
  need(nonce, NONCE_BYTES, 'nonce');
  const base = [];
  for (let i = 0; i < 8; i++) base.push(nonce.readUInt32LE(BLOCK3_OFFSET + 4 * i));
  return { high: nonce.subarray(0, BLOCK3_OFFSET), base };
}

// What a core needs for one line, as 14 canonical field elements: C[0..12]
// (see above), then X0 and X4. `mid` is the midstate, `base` the line's eight
// words at t = 0.
function lineWork(mid, base) {
  const s = mid.slice();
  for (let i = 0; i < 8; i++) s[i] = add(s[i], BigInt(base[i]));
  externalLinear(s);
  for (let i = 0; i < WIDTH; i++) s[i] = add(s[i], INITIAL_EXTERNAL_CONSTANTS[0][i]);
  const x0 = s[0];
  const x4 = s[4];
  const c = s.map((x, i) => (i === 0 || i === 4 ? 0n : exp7(x)));
  externalLinear(c);
  for (let i = 0; i < WIDTH; i++) c[i] = add(c[i], INITIAL_EXTERNAL_CONSTANTS[1][i]);
  return c.concat([x0, x4]);
}

// A line's work as cores take it: 14 x u64 little-endian, 112 bytes.
function lineBytes(work) {
  const out = Buffer.alloc(LINE_BYTES);
  work.forEach((x, i) => out.writeBigUInt64LE(x, i * 8));
  return out;
}

function lineFromBytes(buf) {
  need(buf, LINE_BYTES, 'line');
  const w = [];
  for (let i = 0; i < LINE_ELEMENTS; i++) {
    const x = buf.readBigUInt64LE(i * 8);
    if (x >= P) throw new Error('line element ' + i + ' is not below the field prime');
    w.push(x);
  }
  return w;
}

// Point `t` of a line: the state after the first permutation's first round,
// with the second round's constants added. This is the loop a GPU kernel runs
// before its rounds.
function indexState(work, t) {
  const d = (SLOPE * BigInt(t)) % P;
  const a = exp7(add(work[WIDTH], d));
  const b = exp7(add(work[WIDTH + 1], P - d));
  const s = [];
  for (let g = 0; g < 3; g++) {
    const v = (a * (g === 0 ? 2n : 1n) + b * (g === 1 ? 2n : 1n)) % P;
    for (let k = 0; k < 4; k++) s.push((work[4 * g + k] + COLUMN[k] * v) % P);
  }
  return s;
}

// The top 64 bits of the hash of point `t` of a line: the number a core
// compares with targetHigh.
function indexTop64(work, t) {
  const s = permuteFromRound2(indexState(work, t));
  s[0] = add(s[0], 1n);
  s[1] = add(s[1], 1n);
  const b = Buffer.alloc(8);
  b.writeBigUInt64LE(permute(s)[0]);
  return b.readBigUInt64BE(0);
}

// ── The nonce layout version 1, for the Intel Arc core only ────────────────
//
// Version 2 has not run on an Arc card yet (no Arc card to test on), so the
// SYCL core stays on version 1 and the miner speaks both: core API 1 for the
// Intel core, 2 for the rest (quantusCore.js).
//
//   bytes 0..31   extranonce, then zeros      in the midstate: one per job
//   bytes 32..47  zeros                       } the tail: one per card
//   bytes 48..55  lane (8 bytes)              }
//   bytes 56..63  counter, big-endian u64       walked by the core

// Nonce bytes 32..55, block 3's first six elements: fixed for one card's batch.
const TAIL_BYTES = 24;
const V1_LANE_OFFSET = 48;
const COUNTER_OFFSET = 56;

// The midstate as a version 1 core takes it: 12 x u64 little-endian, 96 bytes.
function midstateBytes(s) {
  const out = Buffer.alloc(WIDTH * 8);
  s.forEach((x, i) => out.writeBigUInt64LE(x, i * 8));
  return out;
}

function midstateFromBytes(buf) {
  need(buf, WIDTH * 8, 'midstate');
  const s = [];
  for (let i = 0; i < WIDTH; i++) {
    const x = buf.readBigUInt64LE(i * 8);
    if (x >= P) throw new Error('midstate element ' + i + ' is not below the field prime');
    s.push(x);
  }
  return s;
}

// The top 64 bits of the hash of the nonce `tail || counter` under this
// midstate: what a version 1 core compares with targetHigh.
function nonceTop64(mid, tail, counter) {
  const s = mid.slice();
  const block = Buffer.alloc(32);
  tail.copy(block, 0);
  block.writeBigUInt64BE(BigInt(counter), TAIL_BYTES);
  absorbWords(s, block);
  permute(s);
  s[0] = add(s[0], 1n);
  s[1] = add(s[1], 1n);
  const b = Buffer.alloc(8);
  b.writeBigUInt64LE(permute(s)[0]);
  return b.readBigUInt64BE(0);
}

function buildNonceV1(extranonce, lane, counter) {
  const prefix = extranonce || Buffer.alloc(0);
  if (prefix.length > BLOCK3_OFFSET) throw new Error('an extranonce of more than 32 bytes does not fit the nonce\'s high half');
  need(lane, LANE_BYTES, 'lane');
  const nonce = Buffer.alloc(NONCE_BYTES);
  prefix.copy(nonce, 0);
  lane.copy(nonce, V1_LANE_OFFSET);
  nonce.writeBigUInt64BE(BigInt(counter), COUNTER_OFFSET);
  return nonce;
}

function splitNonceV1(nonce) {
  need(nonce, NONCE_BYTES, 'nonce');
  return {
    high: nonce.subarray(0, BLOCK3_OFFSET),
    tail: nonce.subarray(BLOCK3_OFFSET, BLOCK3_OFFSET + TAIL_BYTES),
    counter: nonce.readBigUInt64BE(COUNTER_OFFSET),
  };
}

module.exports = {
  HEADER_BYTES, NONCE_BYTES, HASH_BYTES, MAX_TARGET, U64_MAX,
  LANE_BYTES, LANE_OFFSET, BLOCK3_OFFSET, STEP, SLOPE, SPAN, LINES, LINE_BYTES,
  nonceHash, toBig, targetFromDifficulty, difficultyFromTarget, meetsTarget, targetHigh, midstate,
  lineBase, lineWords, buildNonce, splitNonce, lineWork, lineBytes, lineFromBytes, indexState, indexTop64,
  TAIL_BYTES, midstateBytes, midstateFromBytes, nonceTop64, buildNonceV1, splitNonceV1,
};
