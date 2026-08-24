'use strict';

const {
  PROFILE, CONFIG_BYTES, JACKPOT_BUCKETS, ROTL_BITS,
  buildConfig52, leBytesToBigInt, meetsTarget, rotl13, rankMatches,
  patternToList, patternFromList, patternToBytes,
} = require('../src/shared/miner/pearlhash');

describe('PROFILE', () => {
  // Taken from the reference implementation's own defaults, not inferred.
  test('is the mandated protocol geometry', () => {
    expect(PROFILE.rank).toBe(128);          // the rank-penalty floor
    expect(PROFILE.k).toBe(16 * PROFILE.rank); // smallest k the sanity checks allow
    expect(PROFILE.k).toBe(2048);
    expect(PROFILE.mmaType).toBe(0);         // Int7xInt7ToInt32
  });

  // k/rank chunks map one-to-one onto the transcript lanes, so every lane is
  // written exactly once and the rotation never wraps.
  test('the chunk count equals the lane count', () => {
    expect(PROFILE.k / PROFILE.rank).toBe(JACKPOT_BUCKETS);
  });

  test('the tile is the reference 4x8, with its exact index sets', () => {
    expect(PROFILE.rows).toEqual([0, 8, 64, 72]);
    expect(PROFILE.cols).toEqual([0, 1, 8, 9, 32, 33, 40, 41]);
    expect(PROFILE.rows.length * PROFILE.cols.length).toBe(32);
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
  test('encode to the reference bytes', () => {
    expect(patternToBytes(patternFromList(PROFILE.rows)).toString('hex')).toBe('070103010000');
    expect(patternToBytes(patternFromList(PROFILE.cols)).toString('hex')).toBe('000103010101');
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
    expect(b.slice(8, 14).toString('hex')).toBe('070103010000');
    expect(b.slice(14, 20).toString('hex')).toBe('000103010101');
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
    expect(rankMatches(256)).toBe(false);       // the pre-fork rank — uncredited
    expect(rankMatches(512)).toBe(false);
    expect(rankMatches(128, { rank: 256 })).toBe(false);
    expect(rankMatches(256, { rank: 256 })).toBe(true);
  });
});

describe('constants', () => {
  test('jackpot bucket count matches the fold', () => {
    expect(JACKPOT_BUCKETS).toBe(16);
  });
});
