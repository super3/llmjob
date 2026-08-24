'use strict';

const n = require('../src/shared/miner/noise');
const { keyedHash } = require('../src/shared/miner/blake3');

// These pin the noise construction against the reference implementation
// (zk-pow/src/circuit/pearl_noise.rs). They matter more than most tests here
// because the previous version of this code used a structure that was internally
// consistent, passed its own vectors, and was completely wrong: two dense factors
// multiplied at full rank, instead of one dense factor and one sparse +-1
// selector. A miner built on that mines happily forever and never earns a thing.

describe('noise constants', () => {
  test('match the reference implementation', () => {
    expect(n.UNIFORM_NOISE_RANGE).toBe(64); // NOISE_RANGE / IDXS_PER_COL
    expect(n.ZERO_POINT_TRANSLATION).toBe(32);
    expect(n.RANGE_MASK).toBe(63);
    expect(n.BLAKE3_DIGEST_SIZE).toBe(32);
  });

  // The labels are the literal ASCII tensor names, zero-padded to 32 bytes.
  test('the seed labels are the padded ASCII names', () => {
    expect(n.SEED_LABEL_A.toString('hex')).toBe('415f74656e736f72' + '00'.repeat(24));
    expect(n.SEED_LABEL_B.toString('hex')).toBe('425f74656e736f72' + '00'.repeat(24));
    expect(n.SEED_LABEL_A).toHaveLength(32);
    expect(n.paddedSeedLabel('A_tensor')).toEqual(n.SEED_LABEL_A);
    expect(n.paddedSeedLabel('')).toEqual(Buffer.alloc(32));
  });
});

describe('getRandomHash', () => {
  const key = Buffer.alloc(32, 9);

  // The label is the MESSAGE and the commitment seed is the KEY. The reference
  // names its parameters the other way round, which is exactly the sort of thing
  // that gets transposed silently.
  test('is blake3(zero-padded index then label, key=seed)', () => {
    const msg = Buffer.alloc(64);
    msg.writeInt32LE(1 + 5, 0);
    n.SEED_LABEL_A.copy(msg, 32);
    expect(n.getRandomHash(5, n.SEED_LABEL_A, key, 0)).toEqual(keyedHash(key, msg));
  });

  // Dense uses slot 0, sparse uses slot 1. That is the only thing keeping the
  // two streams apart when they share a seed and key.
  test('the prepend slot separates the dense and sparse streams', () => {
    expect(n.getRandomHash(5, n.SEED_LABEL_A, key, 0))
      .not.toEqual(n.getRandomHash(5, n.SEED_LABEL_A, key, 1));
  });

  // The +1 offset exists so index 0 of the two streams cannot coincide.
  test('the index is offset by one', () => {
    const msg = Buffer.alloc(64);
    msg.writeInt32LE(1, 0);
    n.SEED_LABEL_A.copy(msg, 32);
    expect(n.getRandomHash(0, n.SEED_LABEL_A, key, 0)).toEqual(keyedHash(key, msg));
  });

  test('depends on the label and on the key', () => {
    expect(n.getRandomHash(0, n.SEED_LABEL_A, key, 0))
      .not.toEqual(n.getRandomHash(0, n.SEED_LABEL_B, key, 0));
    expect(n.getRandomHash(0, n.SEED_LABEL_A, key, 0))
      .not.toEqual(n.getRandomHash(0, n.SEED_LABEL_A, Buffer.alloc(32, 8), 0));
  });
});

describe('generateUniformRandomMatrix', () => {
  const key = Buffer.alloc(32, 7);

  test('draws every value into [-32, 32)', () => {
    const rows = n.generateUniformRandomMatrix(n.SEED_LABEL_A, key, [0, 8, 64, 72], 128);
    expect(rows).toHaveLength(4);
    for (const row of rows) {
      expect(row).toHaveLength(128);
      for (const v of row) {
        expect(v).toBeGreaterThanOrEqual(-32);
        expect(v).toBeLessThanOrEqual(31);
      }
    }
  });

  // The byte stream is global and indexed by row*numCols, so a row's noise is
  // tied to its absolute index. This is what makes the tile OFFSET change the
  // noise -- if rows were independently seeded, every offset would see the same
  // draw and the search space would collapse.
  test('a row depends on its absolute index', () => {
    const [r0] = n.generateUniformRandomMatrix(n.SEED_LABEL_A, key, [0], 128);
    const [r1] = n.generateUniformRandomMatrix(n.SEED_LABEL_A, key, [1], 128);
    expect(Array.from(r0)).not.toEqual(Array.from(r1));
  });

  test('selecting a row alone matches selecting it in a batch', () => {
    const batch = n.generateUniformRandomMatrix(n.SEED_LABEL_A, key, [0, 8], 128);
    const [alone] = n.generateUniformRandomMatrix(n.SEED_LABEL_A, key, [8], 128);
    expect(Array.from(batch[1])).toEqual(Array.from(alone));
  });

  // A width that is not a whole number of digests exercises the block-boundary
  // filter in both directions.
  test('handles a width that straddles digest boundaries', () => {
    const [row] = n.generateUniformRandomMatrix(n.SEED_LABEL_A, key, [3], 40);
    expect(row).toHaveLength(40);
    expect(Array.from(row).some((v) => v !== 0)).toBe(true);
  });

  test('an empty row list yields nothing', () => {
    expect(n.generateUniformRandomMatrix(n.SEED_LABEL_A, key, [], 128)).toEqual([]);
  });
});

describe('mulHiU32', () => {
  // Exactness above 2^53 is the whole point: a double would drop the low bits
  // and the derived index would be subtly wrong.
  test('is the exact high word of a 32x32 product', () => {
    expect(n.mulHiU32(0xffffffff, 0xffffffff)).toBe(4294967294);
    expect(n.mulHiU32(127, 0xffffffff)).toBe(126);
    expect(n.mulHiU32(1, 1)).toBe(0);
    expect(n.mulHiU32(0x10000, 0x10000)).toBe(1);
  });
});

describe('generatePermutationMatrix', () => {
  const key = Buffer.alloc(32, 3);

  test('every index is in range and the pair is always distinct', () => {
    const p = n.generatePermutationMatrix(n.SEED_LABEL_A, key, 2048, 128);
    expect(p).toHaveLength(4096);
    for (let i = 0; i < 2048; i++) {
      const a = p[2 * i];
      const b = p[2 * i + 1];
      expect(a).toBeGreaterThanOrEqual(0);
      expect(a).toBeLessThan(128);
      expect(b).toBeGreaterThanOrEqual(0);
      expect(b).toBeLessThan(128);
      expect(a).not.toBe(b); // +1 and -1 must not land on the same column
    }
  });

  // k is not required to be a multiple of the 8 pairs a digest yields, so the
  // tail of the last hash has to be discarded rather than written past the end.
  test('handles a k that is not a whole number of digests', () => {
    const p = n.generatePermutationMatrix(n.SEED_LABEL_A, key, 5, 128);
    expect(p).toHaveLength(10);
    const full = n.generatePermutationMatrix(n.SEED_LABEL_A, key, 8, 128);
    expect(Array.from(p)).toEqual(Array.from(full.slice(0, 10)));
  });

  test('depends on the rank', () => {
    const a = n.generatePermutationMatrix(n.SEED_LABEL_A, key, 64, 128);
    const b = n.generatePermutationMatrix(n.SEED_LABEL_A, key, 64, 64);
    expect(Array.from(a)).not.toEqual(Array.from(b));
    for (let i = 0; i < 128; i++) expect(b[i]).toBeLessThan(64);
  });
});

describe('matvecSparsePerm', () => {
  test('is dense[p0] - dense[p1] per row', () => {
    const perm = Int32Array.from([0, 1, 2, 0]);
    const dense = Int8Array.from([10, 3, -4]);
    expect(Array.from(n.matvecSparsePerm(perm, dense, 2))).toEqual([7, -14]);
  });

  // Two [-32, 32) draws differ by at most 63, so the result is int7 and adding
  // it to an int7 operand still fits int8. That is what Int7xInt7ToInt32 means.
  test('stays inside the int7 range', () => {
    const key = Buffer.alloc(32, 5);
    const [dense] = n.generateUniformRandomMatrix(n.SEED_LABEL_A, key, [0], 128);
    const perm = n.generatePermutationMatrix(n.SEED_LABEL_A, key, 512, 128);
    for (const v of n.matvecSparsePerm(perm, dense, 512)) {
      expect(v).toBeGreaterThanOrEqual(-63);
      expect(v).toBeLessThanOrEqual(63);
    }
  });
});

describe('computeNoiseForIndices', () => {
  const aSeed = Buffer.alloc(32, 1);
  const bSeed = Buffer.alloc(32, 2);
  const base = { k: 256, rank: 128, aSeed, bSeed, rowIndices: [0, 8], colIndices: [0, 1, 8] };

  test('produces one noise row per A row and one per B column', () => {
    const out = n.computeNoiseForIndices(base);
    expect(out.noiseA).toHaveLength(2);
    expect(out.noiseB).toHaveLength(3);
    expect(out.noiseA[0]).toHaveLength(256);
    expect(out.noiseB[0]).toHaveLength(256);
  });

  // A is keyed by a_seed and B by b_seed. The reference unpacks them as
  // `let (b_noise_seed, a_noise_seed) = commitment_hash`, which is easy to
  // uncross by accident and impossible to notice afterwards.
  test('the A and B sides use different seeds', () => {
    const out = n.computeNoiseForIndices(base);
    const swapped = n.computeNoiseForIndices({ ...base, aSeed: bSeed, bSeed: aSeed });
    expect(Array.from(out.noiseA[0])).not.toEqual(Array.from(swapped.noiseA[0]));
  });

  test('exposes the underlying factors with the reference shapes', () => {
    const out = n.computeNoiseForIndices(base);
    expect(out.eAL).toHaveLength(2); // (rows, rank)
    expect(out.eAL[0]).toHaveLength(128);
    expect(out.eBRt).toHaveLength(3); // (cols, rank)
    expect(out.eARt).toHaveLength(512); // (k, 2)
    expect(out.eBL).toHaveLength(512);
  });
});

describe('satInt8', () => {
  // Saturating, matching cutlass::NumericArrayConverter<int8_t, int32_t, 4>,
  // which lowers to cvt.pack.sat.s8.s32 -- it clamps rather than wrapping.
  test('clamps rather than wrapping', () => {
    expect(n.satInt8(0)).toBe(0);
    expect(n.satInt8(127)).toBe(127);
    expect(n.satInt8(-128)).toBe(-128);
    expect(n.satInt8(128)).toBe(127);
    expect(n.satInt8(-129)).toBe(-128);
    expect(n.satInt8(500000)).toBe(127);
    expect(n.satInt8(-500000)).toBe(-128);
  });
});
