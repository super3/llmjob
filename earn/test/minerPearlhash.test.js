'use strict';

const {
  PROFILE, CONFIG_BYTES, JACKPOT_BUCKETS, ROTL_BITS,
  buildConfig52, leBytesToBigInt, meetsTarget, rotl13, rankMatches,
  patternToList, patternFromList, patternToBytes, difficultyAdjustmentFactor,
  PENALTY_BASE_RANK, penalizedAdjustmentFactor, shareBound,
  ROWS_MASK, COLS_MASK, offsetIsValid, expandOffset, regionToTile,
} = require('../src/shared/miner/pearlhash');

describe('PROFILE', () => {
  // The floor the sanity checks allow, which is also where the reference miner
  // deliberately sits: "sitting on the rank-penalty floor keeps mined blocks
  // valid once that rule activates, without paying the penalty a larger rank
  // would" (zk-pow/bindings/go/src/mine.rs).
  //
  // This profile briefly carried rank 256 / k 4096 instead, on the grounds that
  // an in-the-wild miner quotes those as its defaults. That was the wrong thing
  // to copy. The bound a share is measured against scales as
  // tile*(k/rank)*PENALTY_BASE_RANK, but an attempt COSTS tile*k, so doubling
  // the rank doubles the arithmetic per attempt and leaves the bound where it
  // was -- an exact factor of two in accepted shares, paid for nothing.
  test('sits on the rank-penalty floor, where a share costs the least work', () => {
    expect(PROFILE.rank).toBe(PENALTY_BASE_RANK);
    expect(PROFILE.rank).toBe(128);
    expect(PROFILE.k).toBe(16 * PROFILE.rank); // the sanity checks require k >= 16r
    expect(PROFILE.k).toBe(2048);
    expect(PROFILE.mmaType).toBe(0);         // Int7xInt7ToInt32
  });

  // k/rank chunks map one-to-one onto the transcript lanes, so every lane is
  // written exactly once and the rotation never wraps.
  test('the chunk count equals the lane count', () => {
    expect(PROFILE.k / PROFILE.rank).toBe(JACKPOT_BUCKETS);
  });

  // 16x16 contiguous. The size is chosen for what it costs to READ OUT, not for
  // anything protocol: tile size cancels out of the share rate exactly, since
  // the bound it earns and the work it costs both scale with it.
  //
  // A 16x16 tile is exactly one int8 wmma accumulator fragment, so the fold's
  // XOR is the XOR over each lane's registers and one warp reduction -- no
  // shared memory and no need to know the fragment layout. The 4x16 tile this
  // replaced was a QUARTER of a fragment, so every chunk boundary had to spill
  // the accumulator to shared and gather rows back. On a 4090 that gather cost
  // about two thirds of the kernel's runtime.
  //
  // 256 is the largest h*w the sanity checks allow, and both dimensions are
  // divisible by TILE_H = 2.
  test('the tile is a contiguous 16x16 block, the largest allowed', () => {
    expect(PROFILE.rows).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    expect(PROFILE.cols).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    expect(PROFILE.rows.length * PROFILE.cols.length).toBe(256);
    expect(PROFILE.rows.length % 2).toBe(0); // TILE_H
    expect(PROFILE.cols.length % 2).toBe(0);
  });

  // m and n are the miner's own workload dimensions and are NOT protocol.
  test('carries m and n, which are not part of the configuration', () => {
    expect(PROFILE.m).toBeGreaterThan(Math.max(...PROFILE.rows));
    expect(PROFILE.n).toBeGreaterThan(Math.max(...PROFILE.cols));
    expect(buildConfig52(PROFILE))
      .toEqual(buildConfig52({ ...PROFILE, m: 131072, n: 65536 }));
  });
});

describe('periodic patterns', () => {
  // Three (stride, length) dimensions, serialised as (factor-1, length-1).
  test('round-trip through the shape representation', () => {
    expect(patternToList(patternFromList(PROFILE.rows)).sort((a, b) => a - b))
      .toEqual(PROFILE.rows);
    expect(patternToList(patternFromList(PROFILE.cols)).sort((a, b) => a - b))
      .toEqual(PROFILE.cols);
  });

  // The exact six-byte encodings the reference produces for these defaults.
  // A contiguous run is a single (stride 1, length N) dimension, so the factor
  // byte is 0 and the length byte is N-1.
  test('encode to the reference bytes', () => {
    expect(patternToBytes(patternFromList(PROFILE.rows)).toString('hex')).toBe('000f00000000');
    expect(patternToBytes(patternFromList(PROFILE.cols)).toString('hex')).toBe('000f00000000');
  });

  test('a non-periodic index list is refused rather than mis-encoded', () => {
    expect(() => patternFromList([0, 1, 5])).toThrow(/not periodic/);
  });
});

describe('buildConfig52', () => {
  // Layout is MiningConfiguration::to_bytes: common_dim u32 | rank u16 |
  // mma_type u16 | rows(6) | cols(6) | MoE trailer(32).
  test('matches the reference layout field for field', () => {
    const b = buildConfig52();
    expect(b).toHaveLength(CONFIG_BYTES);
    expect(b.readUInt32LE(0)).toBe(2048);
    expect(b.readUInt16LE(4)).toBe(128);
    expect(b.readUInt16LE(6)).toBe(0);
    expect(b.slice(8, 14).toString('hex')).toBe('000f00000000');
    expect(b.slice(14, 20).toString('hex')).toBe('000f00000000');
  });

  test('the MoE trailer is zero for a standard job', () => {
    expect(buildConfig52().slice(20).every((x) => x === 0)).toBe(true);
  });

  test('honours a custom k and rank', () => {
    const b = buildConfig52({ k: 4096, rank: 256, mmaType: 0 });
    expect(b.readUInt32LE(0)).toBe(4096);
    expect(b.readUInt16LE(4)).toBe(256);
  });
});

describe('leBytesToBigInt', () => {
  // Little-endian: the FIRST byte is least significant. This is the single
  // easiest thing to get backwards, so it is pinned hard.
  test('reads bytes least-significant-first', () => {
    expect(leBytesToBigInt(Buffer.from([0x01, 0x00]))).toBe(1n);
    expect(leBytesToBigInt(Buffer.from([0x00, 0x01]))).toBe(256n);
    expect(leBytesToBigInt(Buffer.from([0xff, 0xff]))).toBe(65535n);
    expect(leBytesToBigInt(Buffer.alloc(0))).toBe(0n);
  });

  test('accepts a plain array too', () => {
    expect(leBytesToBigInt([0x00, 0x01])).toBe(256n);
    expect(leBytesToBigInt(null)).toBe(0n);
  });

  test('round-trips a full 32-byte value', () => {
    const b = Buffer.alloc(32);
    b[0] = 0x2a; // least significant
    expect(leBytesToBigInt(b)).toBe(42n);
    const hi = Buffer.alloc(32);
    hi[31] = 0x01; // most significant
    expect(leBytesToBigInt(hi)).toBe(1n << 248n);
  });
});

describe('meetsTarget', () => {
  const target = 1n << 200n;
  test('accepts a hash at or below the target, rejects above', () => {
    const below = Buffer.alloc(32); below[24] = 0x01; // = 2^192 < 2^200
    const above = Buffer.alloc(32); above[26] = 0x01; // = 2^208 > 2^200
    expect(meetsTarget(below, target)).toBe(true);
    expect(meetsTarget(above, target)).toBe(false);
  });

  test('equality counts as a share', () => {
    const eq = Buffer.alloc(32); eq[25] = 0x01; // = 2^200 exactly
    expect(meetsTarget(eq, target)).toBe(true);
  });

  test('a null target is never met (no job = no share)', () => {
    expect(meetsTarget(Buffer.alloc(32), null)).toBe(false);
  });
});

describe('rotl13', () => {
  test('rotates a 32-bit lane left by 13', () => {
    expect(rotl13(1)).toBe(1 << 13);
    // top bits wrap into the bottom
    expect(rotl13(0x80000000 >>> 0)).toBe((0x80000000 >>> (32 - 13)) >>> 0);
    expect(rotl13(0)).toBe(0);
    expect(rotl13(0xffffffff)).toBe(0xffffffff >>> 0);
    expect(ROTL_BITS).toBe(13);
  });
});

describe('rankMatches', () => {
  test('accepts the profile rank and an unstated one, refuses a mismatch', () => {
    expect(rankMatches(128)).toBe(true);
    expect(rankMatches(null)).toBe(true);       // pool did not state a rank
    expect(rankMatches(256)).toBe(false);       // not what we mine
    expect(rankMatches(512)).toBe(false);
    expect(rankMatches(128, { rank: 256 })).toBe(false);
    expect(rankMatches(256, { rank: 256 })).toBe(true);
  });
});

describe('difficultyAdjustmentFactor', () => {
  // The protocol scales the jackpot bound in proportion to the work one attempt
  // costs, so a hashrate is MACs per second, not attempts per second. Reporting
  // attempts as hashes under-reported this miner by 65536x at mainnet.
  test('is tile size times dot product length', () => {
    expect(difficultyAdjustmentFactor()).toBe(16 * 16 * 2048);
    expect(difficultyAdjustmentFactor()).toBe(524288);
  });

  // The sanity check that identified the unit in the first place: a competing
  // miner's 296 TH/s is ~45% of an RTX 4090's int8 tensor-core peak, which is a
  // plausible GEMM efficiency. As attempts per second it would have required
  // 3e14 BLAKE3 hashes a second, which no GPU can do.
  test("makes a competitor quoted hashrate physically plausible", () => {
    const attemptsPerSec = 2.96e14 / difficultyAdjustmentFactor();
    expect(attemptsPerSec).toBeLessThan(1e10);
    expect(attemptsPerSec).toBeGreaterThan(1e8);
  });

  test('defaults to the mainnet profile when given none', () => {
    expect(difficultyAdjustmentFactor(undefined)).toBe(524288);
    expect(difficultyAdjustmentFactor(null)).toBe(524288);
  });

  // The tile patterns default when a profile omits them, exactly as buildConfig52
  // does — they are protocol constants rather than per-profile knobs.
  test('defaults the tile patterns when a profile omits them', () => {
    expect(difficultyAdjustmentFactor({ k: 4096 })).toBe(1048576);
    expect(difficultyAdjustmentFactor({ k: 2048 })).toBe(524288);
  });

  test('scales with k and with the tile', () => {
    expect(difficultyAdjustmentFactor({ ...PROFILE, k: 8192 })).toBe(16 * 16 * 8192);
    expect(difficultyAdjustmentFactor({ k: 256, rows: [0, 8], cols: [0, 1] })).toBe(2 * 2 * 256);
  });
});


describe('the share bound', () => {
  // A miner scaling a SHARE target uses the rank-penalized factor, not the
  // consensus one. Both are 65536 at the mandated rank-128 profile, which is
  // exactly why they are easy to conflate -- they diverge as soon as rank moves.
  test('the penalized factor divides out the rank and re-multiplies by the base', () => {
    expect(PENALTY_BASE_RANK).toBe(128);
    // Consensus scales by tile*k; a miner scales a SHARE target by
    // tile*(k/rank)*PENALTY_BASE_RANK. At the floor the two coincide, and that
    // is the point of mining there: whichever rule the pool applies, the bound
    // is the same, so no attempt is thrown away by guessing wrong.
    expect(penalizedAdjustmentFactor()).toBe(256 * 16 * 128);
    expect(difficultyAdjustmentFactor()).toBe(256 * 2048);
    expect(penalizedAdjustmentFactor()).toBe(difficultyAdjustmentFactor());
    // Off the floor they part company, and the penalized one is the smaller --
    // which is exactly the work a higher rank forfeits.
    const highP = { ...PROFILE, rank: 256, k: 4096 };
    expect(penalizedAdjustmentFactor(highP)).toBe(524288);
    expect(difficultyAdjustmentFactor(highP)).toBe(1048576);
  });

  test('defaults to the mainnet profile', () => {
    expect(penalizedAdjustmentFactor(undefined)).toBe(524288);
    expect(penalizedAdjustmentFactor(null)).toBe(524288);
    expect(penalizedAdjustmentFactor({ k: 4096, rank: 256 })).toBe(524288);
  });

  // The tile patterns are protocol constants rather than per-profile knobs, so
  // a profile that omits them still gets the right factor -- same as
  // buildConfig52 and difficultyAdjustmentFactor.
  test('defaults the tile patterns when a profile omits them', () => {
    expect(penalizedAdjustmentFactor({ k: 4096, rank: 256 })).toBe(524288);
  });

  // The bound is the pool's target made easier in proportion to the work one
  // attempt costs. Comparing against the raw target instead makes every share
  // 65536x rarer than the pool intends -- which looks exactly like being slow.
  test('scales the target by the penalized factor', () => {
    expect(shareBound(1n)).toBe(524288n);
    expect(shareBound(1000n)).toBe(524288000n);
    expect(shareBound(BigInt('0x' + '00'.repeat(6) + '07fff8' + '00'.repeat(23))))
      .toBe(BigInt('0x' + '00'.repeat(6) + '07fff8' + '00'.repeat(23)) * 524288n);
  });

  // Refuse rather than saturate: a bound of U256::MAX is met by every hash.
  test('returns null when the product will not fit 256 bits', () => {
    expect(shareBound((1n << 256n) - 1n)).toBeNull();
    expect(shareBound(1n << 250n)).toBeNull();
    expect(shareBound((1n << 237n) - 1n)).not.toBeNull();
  });

  test('a null target has no bound', () => {
    expect(shareBound(null)).toBeNull();
    expect(shareBound(undefined)).toBeNull();
  });

  // A degenerate profile whose k is below its rank gives a zero factor. Scaling
  // by it would be a divide-by-nothing, so it is refused outright.
  test('a degenerate profile yields no bound', () => {
    expect(shareBound(1n, { ...PROFILE, k: 64, rank: 128 })).toBeNull();
  });
});


describe('valid tile offsets', () => {
  // A transcription of PeriodicPattern::offset_is_valid from the reference:
  // reduce the offset modulo each (stride, length) dimension, largest stride
  // first, and require it to stay below the stride.
  function referenceIsValid(offset, shape) {
    let o = offset;
    for (let i = shape.length - 1; i >= 0; i--) {
      const [stride, length] = shape[i];
      o %= stride * length;
      if (o >= stride) return false;
    }
    return true;
  }

  // The masks are the OR of each pattern's own values, and the patterns are
  // exactly the subsets of those bits.
  test('the mask is the pattern bits', () => {
    expect(ROWS_MASK).toBe(15); // bits 0 to 3
    expect(COLS_MASK).toBe(15); // bits 0 to 3
    for (const r of PROFILE.rows) expect(r & ~ROWS_MASK).toBe(0);
    for (const c of PROFILE.cols) expect(c & ~COLS_MASK).toBe(0);
  });

  // The whole reason the search enumerates offsets the way it does. Getting
  // this wrong is not slow, it is unusable: the pool rejects the share with
  // "offset N is not valid for pattern" and the work is lost.
  test('the bit test agrees with the reference rule everywhere', () => {
    for (let o = 0; o < 4096; o++) {
      expect(offsetIsValid(o, ROWS_MASK)).toBe(referenceIsValid(o, [[1, 16]]));
      expect(offsetIsValid(o, COLS_MASK)).toBe(referenceIsValid(o, [[1, 16]]));
    }
  });

  test('expansion enumerates exactly the valid offsets, in order', () => {
    for (const mask of [ROWS_MASK, COLS_MASK]) {
      const enumerated = [];
      for (let i = 0; i < 64; i++) enumerated.push(expandOffset(i, mask));
      const brute = [];
      for (let o = 0; brute.length < 64; o++) if (offsetIsValid(o, mask)) brute.push(o);
      expect(enumerated).toEqual(brute);
    }
  });

  // One offset in rows_count down and one in cols_count across, so one region
  // in 32 is submittable -- and the valid tiles partition the grid.
  test('valid offsets are one in the pattern length', () => {
    let rows = 0;
    for (let o = 0; o < 1024; o++) if (offsetIsValid(o, ROWS_MASK)) rows++;
    expect(rows).toBe(1024 / PROFILE.rows.length);
    let cols = 0;
    for (let o = 0; o < 1024; o++) if (offsetIsValid(o, COLS_MASK)) cols++;
    expect(cols).toBe(1024 / PROFILE.cols.length);
  });

  test('a tile is the offset OR-ed with the pattern', () => {
    const t = regionToTile(0, { ...PROFILE, m: 4096, n: 4096 });
    expect(t.rows).toEqual(PROFILE.rows);
    expect(t.cols).toEqual(PROFILE.cols);
    // The next valid row offset is 16, not 1: a valid offset has the pattern's
    // bits clear, and the contiguous tile occupies bits 0 to 3.
    const t2 = regionToTile(1, { ...PROFILE, m: 4096, n: 4096 });
    expect(t2.rows).toEqual([16, 17, 18, 19, 20, 21, 22, 23,
                             24, 25, 26, 27, 28, 29, 30, 31]);
  });

  // Tiles partitioning the grid is what makes the search non-redundant: no two
  // regions share a cell, so no work is ever repeated.
  test('distinct regions touch disjoint cells', () => {
    const seen = new Set();
    const p = { ...PROFILE, m: 256, n: 256 };
    for (let region = 0; region < 200; region++) {
      const t = regionToTile(region, p);
      for (const r of t.rows) {
        for (const c of t.cols) {
          const cell = r * 1024 + c;
          expect(seen.has(cell)).toBe(false);
          seen.add(cell);
        }
      }
    }
  });

  test('defaults to the mainnet profile', () => {
    expect(regionToTile(0).rows).toEqual(PROFILE.rows);
  });

  // The tile patterns are protocol constants, so a profile that omits them
  // still names the right tile.
  test('defaults the tile patterns when a profile omits them', () => {
    const t = regionToTile(3, { m: 4096, n: 4096 });
    expect(t.rows).toEqual(PROFILE.rows.map((r) => t.rowOff | r));
    expect(t.cols).toEqual(PROFILE.cols.map((c) => t.colOff | c));
  });
});

describe('constants', () => {
  test('jackpot bucket count matches the fold', () => {
    expect(JACKPOT_BUCKETS).toBe(16);
  });
});
