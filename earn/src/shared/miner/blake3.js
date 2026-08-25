'use strict';

// BLAKE3, in pure JavaScript.
//
// This exists to anchor the Pearl PoW reference implementation: every hash in
// pearlhash is BLAKE3 (plain or keyed), so without a trustworthy BLAKE3 there is
// nothing to check the CUDA core against. Crucially this one is verifiable
// against evidence OUTSIDE our own codebase — the official BLAKE3 test vectors —
// so it is the one piece of the miner that is provably correct rather than
// merely self-consistent. See test/minerBlake3.test.js.
//
// Implemented straight from the BLAKE3 specification's reference pseudocode:
// 7 rounds, 1024-byte chunks, a binary Merkle tree over chunk chaining values,
// and an extendable root output. Both `hash` and `keyedHash` are provided; the
// PoW uses keyed hashing for the operand commitments and the transcript.
//
// It is not fast and is not meant to be — the GPU does the real hashing. This is
// the oracle.

const OUT_LEN = 32;
const KEY_LEN = 32;
const BLOCK_LEN = 64;
const CHUNK_LEN = 1024;

const CHUNK_START = 1 << 0;
const CHUNK_END = 1 << 1;
const PARENT = 1 << 2;
const ROOT = 1 << 3;
const KEYED_HASH = 1 << 4;

const IV = [
  0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
  0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

const MSG_PERMUTATION = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

function rotr(x, n) {
  return ((x >>> n) | (x << (32 - n))) >>> 0;
}

// The G mixing function. Additions are done in doubles and masked back to 32
// bits: three 32-bit values sum to well under 2^53, so no precision is lost.
function g(s, a, b, c, d, mx, my) {
  s[a] = (s[a] + s[b] + mx) >>> 0;
  s[d] = rotr((s[d] ^ s[a]) >>> 0, 16);
  s[c] = (s[c] + s[d]) >>> 0;
  s[b] = rotr((s[b] ^ s[c]) >>> 0, 12);
  s[a] = (s[a] + s[b] + my) >>> 0;
  s[d] = rotr((s[d] ^ s[a]) >>> 0, 8);
  s[c] = (s[c] + s[d]) >>> 0;
  s[b] = rotr((s[b] ^ s[c]) >>> 0, 7);
}

function round(s, m) {
  g(s, 0, 4, 8, 12, m[0], m[1]);
  g(s, 1, 5, 9, 13, m[2], m[3]);
  g(s, 2, 6, 10, 14, m[4], m[5]);
  g(s, 3, 7, 11, 15, m[6], m[7]);
  g(s, 0, 5, 10, 15, m[8], m[9]);
  g(s, 1, 6, 11, 12, m[10], m[11]);
  g(s, 2, 7, 8, 13, m[12], m[13]);
  g(s, 3, 4, 9, 14, m[14], m[15]);
}

// The compression function, returning all 16 output words. The first 8 are the
// chaining value; the root output uses all 16 to produce extendable output.
function compress(cv, blockWords, counter, blockLen, flags) {
  const s = [
    cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
    IV[0], IV[1], IV[2], IV[3],
    counter >>> 0,
    Math.floor(counter / 0x100000000) >>> 0,
    blockLen >>> 0,
    flags >>> 0,
  ];
  let m = blockWords.slice();
  for (let r = 0; r < 7; r++) {
    round(s, m);
    if (r < 6) {
      const permuted = new Array(16);
      for (let i = 0; i < 16; i++) permuted[i] = m[MSG_PERMUTATION[i]];
      m = permuted;
    }
  }
  for (let i = 0; i < 8; i++) {
    s[i] = (s[i] ^ s[i + 8]) >>> 0;
    s[i + 8] = (s[i + 8] ^ cv[i]) >>> 0;
  }
  return s;
}

function wordsFromLE(buf, off) {
  const w = new Array(16);
  for (let i = 0; i < 16; i++) {
    const j = off + i * 4;
    w[i] = ((buf[j] | (buf[j + 1] << 8) | (buf[j + 2] << 16) | (buf[j + 3] << 24)) >>> 0);
  }
  return w;
}

// A node awaiting root/extendable output. Kept as data rather than hashed
// immediately, because the ROOT flag is only known once the whole input is in.
function makeOutput(inputCv, blockWords, counter, blockLen, flags) {
  return { inputCv, blockWords, counter, blockLen, flags };
}

function outputChainingValue(o) {
  return compress(o.inputCv, o.blockWords, o.counter, o.blockLen, o.flags).slice(0, 8);
}

// Extendable output: successive compressions with an incrementing counter.
function outputRootBytes(o, outLen) {
  const out = Buffer.alloc(outLen);
  let counter = 0;
  let pos = 0;
  while (pos < outLen) {
    const words = compress(o.inputCv, o.blockWords, counter, o.blockLen, o.flags | ROOT);
    for (let i = 0; i < 16 && pos < outLen; i++) {
      for (let b = 0; b < 4 && pos < outLen; b++) {
        out[pos++] = (words[i] >>> (8 * b)) & 0xff;
      }
    }
    counter++;
  }
  return out;
}

class ChunkState {
  constructor(key, chunkCounter, flags) {
    this.cv = key.slice();
    this.chunkCounter = chunkCounter;
    this.block = Buffer.alloc(BLOCK_LEN);
    this.blockLen = 0;
    this.blocksCompressed = 0;
    this.flags = flags;
  }

  len() { return BLOCK_LEN * this.blocksCompressed + this.blockLen; }

  startFlag() { return this.blocksCompressed === 0 ? CHUNK_START : 0; }

  update(input) {
    let off = 0;
    while (off < input.length) {
      if (this.blockLen === BLOCK_LEN) {
        const words = wordsFromLE(this.block, 0);
        const out = compress(this.cv, words, this.chunkCounter, BLOCK_LEN,
          this.flags | this.startFlag());
        this.cv = out.slice(0, 8);
        this.blocksCompressed++;
        this.block = Buffer.alloc(BLOCK_LEN);
        this.blockLen = 0;
      }
      const want = Math.min(BLOCK_LEN - this.blockLen, input.length - off);
      input.copy(this.block, this.blockLen, off, off + want);
      this.blockLen += want;
      off += want;
    }
  }

  output() {
    return makeOutput(this.cv, wordsFromLE(this.block, 0), this.chunkCounter,
      this.blockLen, this.flags | this.startFlag() | CHUNK_END);
  }
}

function parentOutput(leftCv, rightCv, key, flags) {
  const words = leftCv.concat(rightCv);
  return makeOutput(key.slice(), words, 0, BLOCK_LEN, PARENT | flags);
}

class Hasher {
  constructor(key, flags) {
    this.key = key.slice();
    this.flags = flags;
    this.chunk = new ChunkState(this.key, 0, flags);
    this.stack = [];
  }

  // Merge completed subtrees. A chunk's chaining value is merged upward once for
  // every trailing zero bit in the total chunk count — the standard way of
  // maintaining a binary tree with O(log n) state and no buffering.
  _addChunkCv(cv, totalChunks) {
    let newCv = cv;
    let total = totalChunks;
    while ((total & 1) === 0) {
      const left = this.stack.pop();
      newCv = outputChainingValue(parentOutput(left, newCv, this.key, this.flags));
      total >>= 1;
    }
    this.stack.push(newCv);
  }

  update(input) {
    let buf = Buffer.isBuffer(input) ? input : Buffer.from(input);
    let off = 0;
    while (off < buf.length) {
      if (this.chunk.len() === CHUNK_LEN) {
        const cv = outputChainingValue(this.chunk.output());
        const totalChunks = this.chunk.chunkCounter + 1;
        this._addChunkCv(cv, totalChunks);
        this.chunk = new ChunkState(this.key, totalChunks, this.flags);
      }
      const want = Math.min(CHUNK_LEN - this.chunk.len(), buf.length - off);
      this.chunk.update(buf.slice(off, off + want));
      off += want;
    }
    return this;
  }

  digest(outLen = OUT_LEN) {
    let output = this.chunk.output();
    for (let i = this.stack.length - 1; i >= 0; i--) {
      output = parentOutput(this.stack[i], outputChainingValue(output), this.key, this.flags);
    }
    return outputRootBytes(output, outLen);
  }
}

function keyWords(key) {
  const k = Buffer.isBuffer(key) ? key : Buffer.from(key);
  if (k.length !== KEY_LEN) throw new Error('BLAKE3 key must be exactly 32 bytes');
  const w = new Array(8);
  for (let i = 0; i < 8; i++) {
    const j = i * 4;
    w[i] = ((k[j] | (k[j + 1] << 8) | (k[j + 2] << 16) | (k[j + 3] << 24)) >>> 0);
  }
  return w;
}

// Unkeyed BLAKE3.
function hash(input, outLen = OUT_LEN) {
  return new Hasher(IV, 0).update(input).digest(outLen);
}

// Keyed BLAKE3 — what every hash in the Pearl PoW uses.
function keyedHash(key, input, outLen = OUT_LEN) {
  return new Hasher(keyWords(key), KEYED_HASH).update(input).digest(outLen);
}

module.exports = {
  OUT_LEN, KEY_LEN, BLOCK_LEN, CHUNK_LEN,
  IV, MSG_PERMUTATION,
  compress, hash, keyedHash, Hasher,
};
