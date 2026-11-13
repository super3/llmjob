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
//   block 2: nonce bytes 0..31          the same while only the low bytes move
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
// our miner uses is at the end (buildNonce).

const { P, WIDTH, permute, add, hashSqueezeTwice, digestBytes } = require('./poseidon2');

const HEADER_BYTES = 32;
const NONCE_BYTES = 64;
const HASH_BYTES = 64;
// Nonce bytes 32..55, block 3's first six elements: fixed for one card's batch.
const TAIL_BYTES = 24;
const TAIL_OFFSET = 32;
// Nonce bytes 56..63: the counter a core walks, a big-endian u64.
const COUNTER_OFFSET = 56;
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

// The midstate as cores take it: 12 x u64 little-endian, 96 bytes.
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

// Block 3 and block 4 from a midstate: the state whose first four elements are
// the hash's first 32 bytes. `counter` is the nonce's last 8 bytes as a u64.
function finishBlocks(mid, tail, counter) {
  const s = mid.slice();
  const block = Buffer.alloc(32);
  tail.copy(block, 0);
  block.writeBigUInt64BE(BigInt(counter), TAIL_BYTES);
  absorbWords(s, block);
  permute(s);
  s[0] = add(s[0], 1n);
  s[1] = add(s[1], 1n);
  return permute(s);
}

// The top 64 bits of the hash of the nonce `tail || counter` under this
// midstate: the number a core compares with targetHigh. One per nonce, so this
// is the inner loop a GPU kernel runs.
function nonceTop64(mid, tail, counter) {
  const s0 = finishBlocks(mid, tail, counter)[0];
  const b = Buffer.alloc(8);
  b.writeBigUInt64LE(s0);
  return b.readBigUInt64BE(0);
}

// The full hash from a midstate, for a nonce a core reported.
function hashFromMidstate(mid, tail, counter) {
  const s = finishBlocks(mid, tail, counter);
  const first = digestBytes(s);
  return Buffer.concat([first, digestBytes(permute(s))]);
}

// Our nonce layout. The pool's extranonce leads (it checks that), the rest of
// the high half is zero, then the card's lane, then the counter:
//
//   bytes 0..31   extranonce, then zeros      in the midstate: one per job
//   bytes 32..47  zeros                       } the tail: one per card
//   bytes 48..55  lane (8 bytes)              }
//   bytes 56..63  counter, big-endian u64       walked by the core
//
// The lane keeps two cards, and two runs of the miner, off each other's nonces.
// A pool may ban a miner that sends the same share twice.
function buildNonce(extranonce, lane, counter) {
  const prefix = extranonce || Buffer.alloc(0);
  if (prefix.length > 32) throw new Error('an extranonce of more than 32 bytes does not fit the nonce\'s high half');
  need(lane, 8, 'lane');
  const nonce = Buffer.alloc(NONCE_BYTES);
  prefix.copy(nonce, 0);
  lane.copy(nonce, 48);
  nonce.writeBigUInt64BE(BigInt(counter), COUNTER_OFFSET);
  return nonce;
}

// The pieces of a nonce a core and the midstate take.
function splitNonce(nonce) {
  need(nonce, NONCE_BYTES, 'nonce');
  return {
    high: nonce.subarray(0, 32),
    tail: nonce.subarray(TAIL_OFFSET, TAIL_OFFSET + TAIL_BYTES),
    counter: nonce.readBigUInt64BE(COUNTER_OFFSET),
  };
}

module.exports = {
  HEADER_BYTES, NONCE_BYTES, HASH_BYTES, TAIL_BYTES, TAIL_OFFSET, COUNTER_OFFSET, MAX_TARGET, U64_MAX,
  nonceHash, toBig, targetFromDifficulty, difficultyFromTarget, meetsTarget, targetHigh,
  midstate, midstateBytes, midstateFromBytes, finishBlocks, nonceTop64, hashFromMidstate, buildNonce, splitNonce,
};
