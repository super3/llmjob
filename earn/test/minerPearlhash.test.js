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

  // Rows {0,1,2,3}+8j by columns {0,1}+8i. The shape is chosen for what it costs
  // to READ OUT, not for anything protocol: tile size cancels out of the share
  // rate exactly, since the bound it earns and the work it costs both scale
  // with it, and the verifier rebuilds the pattern from a share's own indices.
  //
  // 256 is the largest h*w the sanity checks allow, and both dimensions are
  // divisible by TILE_H = 2.
  test('the tile is 16 rows over 32 by 16 columns over 64, the largest allowed', () => {
    expect(PROFILE.rows).toEqual([0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]);
    expect(PROFILE.cols).toEqual([0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57]);
    expect(PROFILE.rows.length * PROFILE.cols.length).toBe(256);
    expect(PROFILE.rows.length % 2).toBe(0); // TILE_H
    expect(PROFILE.cols.length % 2).toBe(0);
  });

  // Why this shape: it is what the int8 tensor cores already hand over. In the
  // m16n8k32 accumulator lane L holds rows g, g+8 and columns 2t, 2t+1 (g = L >> 2,
  // t = L & 3), and across the fold's 32x64 warp tile (two m16 tiles by eight n8
  // tiles) each lane's 64 values fall in exactly ONE region, shared with lanes
  // L^4, L^8 and L^12. That is what lets the fold XOR a region in registers plus
  // two shuffles. This pins the claim the kernel's readout is built on, so the
  // pattern cannot drift from it unnoticed.
  test('each lane of a 32x64 m16n8k32 warp tile holds a quarter of exactly one region', () => {
    // A cell's region is its row and column offset: the pattern's bits cleared.
    const regionOf = (r, c) => (r & ~ROWS_MASK) + ',' + (c & ~COLS_MASK);
    const cellsOf = new Map();
    const lanesOf = new Map();
    for (let lane = 0; lane < 32; lane++) {
      const g = lane >> 2, t = lane & 3, mine = new Set();
      for (let mb = 0; mb < 2; mb++) for (let nb = 0; nb < 8; nb++) for (let i = 0; i < 4; i++) {
        const r = mb * 16 + g + (i >= 2 ? 8 : 0), c = nb * 8 + 2 * t + (i & 1);
        const reg = regionOf(r, c);
        mine.add(reg);
        cellsOf.set(reg, (cellsOf.get(reg) || 0) + 1);
      }
      // Row offset 0 or 4 (lane bit 4), column offset 2t: what the kernel's
      // hand-off assumes when it files each lane's words under a region.
      expect([...mine]).toEqual([(4 * ((lane >> 4) & 1)) + ',' + (2 * t)]);
      lanesOf.set([...mine][0], [...(lanesOf.get([...mine][0]) || []), lane]);
    }
    expect(cellsOf.size).toBe(8);
    for (const n of cellsOf.values()) expect(n).toBe(256); // whole regions, 64 cells a lane
    for (const lanes of lanesOf.values()) {
      expect(lanes.map((l) => l ^ lanes[0]).sort((a, b) => a - b)).toEqual([0, 4, 8, 12]);
    }
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

  // The exact six-byte encodings, (factor-1, length-1) per dimension, where the
  // factor is the stride over the running product of the dimensions before it:
  //   rows: (stride 1, length 4), (stride 8, length 4) -> factors 1, 8/4 = 2
  //   cols: (stride 1, length 2), (stride 8, length 8) -> factors 1, 8/2 = 4
  // and the unused third dimension pads as factor 1, length 1.
  test('encode to the reference bytes', () => {
    expect(patternFromList(PROFILE.rows)).toEqual([[1, 4], [8, 4]]);
    expect(patternFromList(PROFILE.cols)).toEqual([[1, 2], [8, 8]]);
    expect(patternToBytes(patternFromList(PROFILE.rows)).toString('hex')).toBe('000301030000');
    expect(patternToBytes(patternFromList(PROFILE.cols)).toString('hex')).toBe('000103070000');
  });

  // A contiguous run is a single (stride 1, length N) dimension, so its factor
  // byte is 0 and its length byte N-1: the encoding the previous 0..15 tile had.
  test('a contiguous run encodes as one dimension', () => {
    const run = Array.from({ length: 16 }, (_, i) => i);
    expect(patternToBytes(patternFromList(run)).toString('hex')).toBe('000f00000000');
  });

  // The bytes the reference produces for MiningConfiguration's own strided
  // defaults, two and three dimensions. The mined tile has two, so this pins the
  // factor rule it depends on to the reference rather than to our own reading.
  test("reproduces the reference's encoding of its strided defaults", () => {
    expect(patternToBytes(patternFromList([0, 8, 64, 72])).toString('hex')).toBe('070103010000');
    expect(patternToBytes(patternFromList([0, 1, 8, 9, 32, 33, 40, 41])).toString('hex'))
      .toBe('000103010101');
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
    expect(b.slice(8, 14).toString('hex')).toBe('000301030000');
    expect(b.slice(14, 20).toString('hex')).toBe('000103070000');
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
  // exactly the subsets of those bits -- which is what makes the bit test below
  // the reference rule, and what makes valid tiles partition the grid.
  test('the mask is the pattern bits', () => {
    expect(ROWS_MASK).toBe(0x1b); // bits 0, 1, 3, 4
    expect(COLS_MASK).toBe(0x39); // bits 0, 3, 4, 5
    const subsets = (m) => {
      const out = [];
      for (let v = 0; v <= m; v++) if ((v & ~m) === 0) out.push(v);
      return out;
    };
    expect(PROFILE.rows).toEqual(subsets(ROWS_MASK));
    expect(PROFILE.cols).toEqual(subsets(COLS_MASK));
  });

  // The whole reason the search enumerates offsets the way it does. Getting
  // this wrong is not slow, it is unusable: the pool rejects the share with
  // "offset N is not valid for pattern" and the work is lost.
  test('the bit test agrees with the reference rule everywhere', () => {
    const rowShape = patternFromList(PROFILE.rows);
    const colShape = patternFromList(PROFILE.cols);
    for (let o = 0; o < 4096; o++) {
      expect(offsetIsValid(o, ROWS_MASK)).toBe(referenceIsValid(o, rowShape));
      expect(offsetIsValid(o, COLS_MASK)).toBe(referenceIsValid(o, colShape));
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
    const p = { ...PROFILE, m: 4096, n: 4096 };
    const t = regionToTile(0, p);
    expect(t.rows).toEqual(PROFILE.rows);
    expect(t.cols).toEqual(PROFILE.cols);
    // The next valid row offset is 4, not 1: a valid offset has the pattern's
    // bits clear, and bit 2 is the first the rows leave free. The one after is
    // 32, where bits 0 to 4 run out.
    expect(regionToTile(1, p).rows).toEqual([4, 5, 6, 7, 12, 13, 14, 15,
                                             20, 21, 22, 23, 28, 29, 30, 31]);
    expect(regionToTile(2, p).rowOff).toBe(32);
    // Columns likewise: 2, 4, 6, then 64. Region index = column index * rows + row.
    const rowsValid = p.m / PROFILE.rows.length;
    expect([1, 2, 3, 4].map((j) => regionToTile(j * rowsValid, p).colOff)).toEqual([2, 4, 6, 64]);
    expect(regionToTile(rowsValid, p).cols).toEqual(PROFILE.cols.map((c) => c + 2));
  });

  // Four column offsets tile 64 columns and two row offsets 32 rows with no
  // gaps, so a run of offset indices that starts on a multiple of four (two)
  // is one contiguous block of the operand, beginning at index * 16. The fold
  // stages each tile on exactly that assumption.
  test('span-aligned runs of offsets are contiguous blocks', () => {
    const p = { ...PROFILE, m: 4096, n: 4096 };
    const rowsValid = p.m / PROFILE.rows.length;
    for (const j0 of [0, 4, 16, 36]) {
      const cols = [];
      for (let j = j0; j < j0 + 16; j++) cols.push(...regionToTile(j * rowsValid, p).cols);
      expect(cols.sort((a, b) => a - b)).toEqual(Array.from({ length: 256 }, (_, i) => j0 * 16 + i));
    }
    for (const i0 of [0, 8, 24]) {
      const rows = [];
      for (let i = i0; i < i0 + 8; i++) rows.push(...regionToTile(i, p).rows);
      expect(rows.sort((a, b) => a - b)).toEqual(Array.from({ length: 128 }, (_, i) => i0 * 16 + i));
    }
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
