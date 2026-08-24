'use strict';

const { keyedHash } = require('./blake3');

// Pearl's low-rank noise, ported directly from the reference implementation
// (zk-pow/src/circuit/pearl_noise.rs) rather than inferred from prose.
//
// The structure is NOT "two dense random factors", which is what an earlier
// version of this file assumed and which is wrong in a way that produces a
// perfectly self-consistent miner that can never have a share accepted:
//
//   E_AL, E_BR  dense,  (rows x rank), values in [-32, 32)
//   E_AR, E_BL  SPARSE, (k x rank), each row exactly one +1 and one -1
//
// So the noise for one element is a difference of two dense entries:
//
//   noise_A[row][kk] = E_AL[row][p0(kk)] - E_AL[row][p1(kk)]
//
// Two lookups and a subtract — not a rank-length dot product. That is worth
// stating plainly because the cost difference is a factor of `rank` (128 at
// mainnet) in the hot loop.
//
// It also explains the name Int7xInt7ToInt32. The operands are int7 (+-63) and
// the noise is a difference of two [-32, 32) draws, so also int7. Their sum
// therefore lands inside int8 and the reference's saturating convert-down is a
// guard rail that essentially never fires.

const BLAKE3_DIGEST_SIZE = 32;
const NOISE_RANGE = 128;
const IDXS_PER_COL = 2;
const UNIFORM_NOISE_RANGE = NOISE_RANGE / IDXS_PER_COL; // 64
const ZERO_POINT_TRANSLATION = UNIFORM_NOISE_RANGE / 2; // 32
const RANGE_MASK = UNIFORM_NOISE_RANGE - 1;             // 63
const LINES_PER_HASH = BLAKE3_DIGEST_SIZE / 4;          // 8

// The seed labels are literal ASCII, zero-padded to 32 bytes. Note the naming
// in the reference is a trap: this LABEL is passed as `seed` and goes into the
// message, while the commitment-derived value is passed as `key` and is the
// BLAKE3 key. Swapping the two is silent and fatal.
function paddedSeedLabel(label) {
  const b = Buffer.alloc(32);
  Buffer.from(label, 'ascii').copy(b);
  return b;
}
const SEED_LABEL_A = paddedSeedLabel('A_tensor');
const SEED_LABEL_B = paddedSeedLabel('B_tensor');

// blake3(message64, key), where message64 is
//   [ 8 zero int32 slots | 32-byte seed label ]
// with slot `prependIndex` overwritten by (1 + index) as little-endian int32.
// The dense and sparse generators use slots 0 and 1 respectively, which is what
// keeps their streams distinct despite sharing a seed and key.
function getRandomHash(index, seedLabel, key, prependIndex) {
  const message = Buffer.alloc(64);
  message.writeInt32LE((1 + index) | 0, prependIndex * 4);
  seedLabel.copy(message, 32);
  return keyedHash(key, message);
}

// Dense factor: one row per requested index, `numCols` values in [-32, 32).
// The byte stream is global — row `i` reads at offset i*numCols — so the rows
// are NOT independently seeded and the tile offset genuinely changes the noise.
function generateUniformRandomMatrix(seedLabel, key, rowIndices, numCols) {
  return rowIndices.map((rowIdx) => {
    const startIdx = rowIdx * numCols;
    const out = new Int8Array(numCols);
    const firstBlock = Math.floor(startIdx / BLAKE3_DIGEST_SIZE);
    const lastBlock = Math.ceil((startIdx + numCols) / BLAKE3_DIGEST_SIZE);
    for (let block = firstBlock; block < lastBlock; block++) {
      const h = getRandomHash(block, seedLabel, key, 0);
      for (let i = 0; i < BLAKE3_DIGEST_SIZE; i++) {
        const idx = block * BLAKE3_DIGEST_SIZE + i;
        if (idx >= startIdx && idx < startIdx + numCols) {
          out[idx - startIdx] = (h[i] & RANGE_MASK) - ZERO_POINT_TRANSLATION;
        }
      }
    }
    return out;
  });
}

// High 32 bits of a 32x32 unsigned product. Done in BigInt because the exact
// product reaches 2^64 and doubles silently lose the low bits above 2^53.
function mulHiU32(a, b) {
  return Number((BigInt(a >>> 0) * BigInt(b >>> 0)) >> 32n) >>> 0;
}

// Sparse factor: k rows, each carrying +1 at p0 and -1 at p1. Returned as a
// flat Int32Array of [p0, p1] pairs since that is what both the fold and the
// device want.
//
// p1 = p0 ^ (1 + mulhi(rank-1, u)) is always distinct from p0 and always inside
// [0, rank): mulhi(rank-1, u) <= rank-2, so the xor operand is in [1, rank-1],
// and rank is a power of two.
function generatePermutationMatrix(seedLabel, key, k, noiseRank) {
  const mask = (noiseRank - 1) >>> 0;
  const out = new Int32Array(k * 2);
  const chunks = Math.ceil(k / LINES_PER_HASH);
  for (let i = 0; i < chunks; i++) {
    const h = getRandomHash(i, seedLabel, key, 1);
    for (let j = 0; j < LINES_PER_HASH; j++) {
      const row = i * LINES_PER_HASH + j;
      if (row >= k) break;
      const u = h.readUInt32LE(j * 4);
      const p0 = (u & mask) >>> 0;
      const p1 = (p0 ^ (1 + mulHiU32(noiseRank - 1, u))) >>> 0;
      out[row * 2] = p0;
      out[row * 2 + 1] = p1;
    }
  }
  return out;
}

// noise[row][kk] = dense[row][p0(kk)] - dense[row][p1(kk)], the sparse matvec.
function matvecSparsePerm(perm, dense, k) {
  const out = new Int8Array(k);
  for (let kk = 0; kk < k; kk++) {
    out[kk] = dense[perm[kk * 2]] - dense[perm[kk * 2 + 1]];
  }
  return out;
}

// Everything the fold needs for one tile offset: a noise row per A row and a
// noise column per B column.
//
// The A side is keyed by aSeed and the B side by bSeed. That pairing looks
// obvious but the reference unpacks its tuple as
//   let (b_noise_seed, a_noise_seed) = commitment_hash;
// i.e. b first — so reading that line too quickly is an easy way to end up with
// the two seeds swapped, which is silent and produces a wrong transcript.
function computeNoiseForIndices(opts) {
  const { k, rank, aSeed, bSeed, rowIndices, colIndices } = opts;

  const eAL = generateUniformRandomMatrix(SEED_LABEL_A, aSeed, rowIndices, rank);
  const eARt = generatePermutationMatrix(SEED_LABEL_A, aSeed, k, rank);
  const eBL = generatePermutationMatrix(SEED_LABEL_B, bSeed, k, rank);
  const eBRt = generateUniformRandomMatrix(SEED_LABEL_B, bSeed, colIndices, rank);

  return {
    noiseA: eAL.map((row) => matvecSparsePerm(eARt, row, k)),
    noiseB: eBRt.map((col) => matvecSparsePerm(eBL, col, k)),
    eAL, eARt, eBL, eBRt,
  };
}

// Saturating int32 -> int8, matching the reference's convert_type_out, which is
// a cutlass::NumericArrayConverter<int8_t, int32_t, 4> and therefore a
// saturating cvt.pack.sat.s8.s32 — it clamps, it does not wrap.
function satInt8(v) {
  return v < -128 ? -128 : (v > 127 ? 127 : v);
}

module.exports = {
  BLAKE3_DIGEST_SIZE, UNIFORM_NOISE_RANGE, ZERO_POINT_TRANSLATION, RANGE_MASK,
  SEED_LABEL_A, SEED_LABEL_B,
  paddedSeedLabel, getRandomHash, generateUniformRandomMatrix,
  mulHiU32, generatePermutationMatrix, matvecSparsePerm,
  computeNoiseForIndices, satInt8,
};
