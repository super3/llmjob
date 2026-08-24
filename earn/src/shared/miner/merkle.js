'use strict';

const { compress, CHUNK_LEN, OUT_LEN, keyedHash } = require('./blake3');

// The operand commitment as a Merkle TREE, not just a root.
//
// A share is not accepted on its hash alone. The pool has to be able to check
// the work, and it cannot do that without the operand data the tile actually
// touched -- so a submitted proof carries the 1024-byte chunks holding those
// rows, plus the sibling digests that authenticate them against the committed
// root. That is what `plain_proof` is, and sending the 64-byte transcript
// instead is what earned:
//
//   {"code":23,"message":"not a valid PlainProof (tried current and legacy V1
//    formats)"}
//
// The tree is keyed BLAKE3: leaves are chunk chaining values, parents are
// non-root merges, and the final merge carries ROOT. With a power-of-two chunk
// count that is exactly standard BLAKE3's own tree, which is why the root here
// must equal keyedHash(key, data) -- asserted in the tests rather than assumed.

const CHUNK_START = 1 << 0;
const CHUNK_END = 1 << 1;
const PARENT = 1 << 2;
const ROOT = 1 << 3;
const KEYED_HASH = 1 << 4;

function keyWords(key) {
  const k = Buffer.isBuffer(key) ? key : Buffer.from(key);
  if (k.length !== 32) throw new Error('BLAKE3 key must be exactly 32 bytes');
  const w = new Array(8);
  for (let i = 0; i < 8; i++) w[i] = k.readUInt32LE(i * 4) >>> 0;
  return w;
}

function wordsToBytes(words) {
  const out = Buffer.alloc(words.length * 4);
  for (let i = 0; i < words.length; i++) out.writeUInt32LE(words[i] >>> 0, i * 4);
  return out;
}

function blockWords(buf, offset) {
  const w = new Array(16);
  for (let i = 0; i < 16; i++) w[i] = buf.readUInt32LE(offset + i * 4) >>> 0;
  return w;
}

// The chaining value of one 1024-byte chunk at `index`. The counter is the
// CHUNK index and is the same for all sixteen blocks inside it.
function chunkCV(key, chunk, index) {
  let cv = keyWords(key);
  const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
  for (let b = 0; b < CHUNK_LEN / 64; b++) {
    const flags = KEYED_HASH
      | (b === 0 ? CHUNK_START : 0)
      | (b === CHUNK_LEN / 64 - 1 ? CHUNK_END : 0);
    cv = compress(cv, blockWords(buf, b * 64), index, 64, flags).slice(0, 8);
  }
  return wordsToBytes(cv);
}

// Merge two child CVs. The final merge of a tree carries ROOT and its output is
// the digest; every other merge does not.
function parentCV(key, left, right, isRoot) {
  const block = Buffer.concat([Buffer.from(left), Buffer.from(right)]);
  const flags = PARENT | KEYED_HASH | (isRoot ? ROOT : 0);
  const out = compress(keyWords(key), blockWords(block, 0), 0, 64, flags).slice(0, 8);
  return wordsToBytes(out);
}

// Build every layer, leaves first. Layer 0 is the chunk CVs; the last layer is
// the single root.
function buildLayers(key, data) {
  const buf = Buffer.isBuffer(data) ? data : Buffer.from(data);
  if (buf.length <= CHUNK_LEN) return [[keyedHash(key, buf)]];

  const numChunks = Math.ceil(buf.length / CHUNK_LEN);
  const leaves = [];
  for (let i = 0; i < numChunks; i++) {
    const chunk = Buffer.alloc(CHUNK_LEN);
    buf.copy(chunk, 0, i * CHUNK_LEN, Math.min((i + 1) * CHUNK_LEN, buf.length));
    leaves.push(chunkCV(key, chunk, i));
  }

  const layers = [leaves];
  while (layers[layers.length - 1].length > 2) {
    const prev = layers[layers.length - 1];
    const next = [];
    for (let i = 0; i < prev.length; i += 2) {
      // An odd node is promoted rather than merged with itself.
      next.push(i + 1 < prev.length ? parentCV(key, prev[i], prev[i + 1], false) : prev[i]);
    }
    layers.push(next);
  }
  // The loop above runs while the top layer has more than two nodes, and a
  // multi-chunk operand always starts with at least two, so it exits at exactly
  // two. The final merge is therefore unconditional.
  const last = layers[layers.length - 1];
  layers.push([parentCV(key, last[0], last[1], true)]);
  return layers;
}

function rootOf(layers) {
  return layers[layers.length - 1][0];
}

// Which 1024-byte chunks hold the given matrix rows. A row of `cols` bytes can
// straddle a chunk boundary, so this is a range per row, not one index.
function leafIndicesFromRows(rowIndices, cols) {
  const set = new Set();
  for (const row of rowIndices) {
    const first = Math.floor((row * cols) / CHUNK_LEN);
    const last = Math.floor(((row + 1) * cols - 1) / CHUNK_LEN);
    for (let i = first; i <= last; i++) set.add(i);
  }
  return Array.from(set).sort((a, b) => a - b);
}

// A multi-leaf proof: the requested chunks, and the sibling digests needed to
// recompute the root from them.
//
// The sibling walk must match the verifier's exactly, including ORDER: it goes
// level by level, and within a level it visits the live node set in ascending
// index order, emitting a sibling only when that sibling is not itself live.
function multiLeafProof(key, data, layers, leafIndices) {
  const buf = Buffer.isBuffer(data) ? data : Buffer.from(data);
  const totalLeaves = layers[0].length;
  const sorted = Array.from(new Set(leafIndices)).sort((a, b) => a - b);
  if (!sorted.length) throw new Error('leaf_indices must be non-empty');
  if (sorted[sorted.length - 1] >= totalLeaves) throw new Error('leaf index out of bounds');

  const leafData = sorted.map((i) => {
    const chunk = Buffer.alloc(CHUNK_LEN);
    buf.copy(chunk, 0, i * CHUNK_LEN, Math.min((i + 1) * CHUNK_LEN, buf.length));
    return chunk;
  });

  const siblings = [];
  let current = sorted.slice();
  let levelLen = totalLeaves;
  let level = 0;
  while (levelLen > 1 && current.length) {
    const live = new Set(current);
    const nodes = layers[level];
    for (const i of current) {
      if (i % 2 === 1) {
        if (!live.has(i - 1)) siblings.push(nodes[i - 1]);
      } else if (!live.has(i + 1) && i + 1 < levelLen) {
        siblings.push(nodes[i + 1]);
      }
    }
    current = Array.from(new Set(current.map((i) => Math.floor(i / 2)))).sort((a, b) => a - b);
    levelLen = Math.ceil(levelLen / 2);
    level++;
  }

  return { leafData, leafIndices: sorted, totalLeaves, root: rootOf(layers), siblings };
}

// Recompute the root from a proof alone. This is the verifier's side, kept here
// so a proof can be checked before it is submitted -- a rejected share costs a
// round trip and, on some pools, reputation.
function verifyProof(key, proof) {
  let level = proof.leafIndices.map((idx, i) => ({ idx, cv: chunkCV(key, proof.leafData[i], idx) }));
  let levelLen = proof.totalLeaves;
  let s = 0;
  while (levelLen > 1) {
    const live = new Map(level.map((n) => [n.idx, n.cv]));
    const next = [];
    const seen = new Set();
    for (const { idx } of level) {
      const parent = Math.floor(idx / 2);
      if (seen.has(parent)) continue;
      seen.add(parent);
      const leftIdx = parent * 2;
      const rightIdx = leftIdx + 1;
      const left = live.get(leftIdx);
      const right = live.get(rightIdx);
      let cv;
      if (left && right) cv = parentCV(key, left, right, levelLen <= 2);
      else if (left) {
        // A right child that is absent is either a sibling or, at an odd tail,
        // nothing at all — in which case the node is promoted unchanged.
        cv = rightIdx < levelLen ? parentCV(key, left, proof.siblings[s++], levelLen <= 2) : left;
      } else {
        cv = parentCV(key, proof.siblings[s++], right, levelLen <= 2);
      }
      next.push({ idx: parent, cv });
    }
    level = next.sort((a, b) => a.idx - b.idx);
    levelLen = Math.ceil(levelLen / 2);
  }
  return Buffer.from(level[0].cv).equals(Buffer.from(proof.root));
}

module.exports = {
  CHUNK_LEN, OUT_LEN,
  chunkCV, parentCV, buildLayers, rootOf,
  leafIndicesFromRows, multiLeafProof, verifyProof,
};
