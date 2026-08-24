'use strict';

const {
  PROFILE, CONFIG_BYTES, JACKPOT_BUCKETS, ROTL_BITS,
  buildConfig52, leBytesToBigInt, meetsTarget, rotl13, rankMatches,
} = require('../src/shared/miner/pearlhash');

describe('PROFILE', () => {
  // The softfork mandates rank 128; anything else is uncredited work.
  test('is the mandated rank-128 mainnet geometry', () => {
    expect(PROFILE.rank).toBe(128);
    expect(PROFILE.m).toBe(131072);
    expect(PROFILE.n).toBe(131072);
    expect(PROFILE.k).toBe(4096);
    expect(PROFILE.hashTile).toBe(16);
  });

  test('the default column pattern is the (i, i+1)-per-8 shape, 64 wide', () => {
    expect(PROFILE.cols).toHaveLength(64);
    expect(PROFILE.cols.slice(0, 6)).toEqual([0, 1, 8, 9, 16, 17]);
    expect(PROFILE.cols[PROFILE.cols.length - 1]).toBe(249);
    expect(PROFILE.rows).toEqual([0, 8]);
  });
});

describe('buildConfig52', () => {
  test('is exactly 52 bytes with the geometry in little-endian', () => {
    const b = buildConfig52();
    expect(b).toHaveLength(CONFIG_BYTES);
    expect(b.readUInt32LE(0)).toBe(131072);
    expect(b.readUInt32LE(4)).toBe(131072);
    expect(b.readUInt32LE(8)).toBe(4096);
    expect(b.readUInt16LE(12)).toBe(128);
    expect(b.readUInt16LE(14)).toBe(16);
    expect(b.readUInt16LE(16)).toBe(2);   // rows count
    expect(b.readUInt16LE(18)).toBe(64);  // cols count
  });

  test('the reserved tail is zeroed so the job_key is deterministic', () => {
    expect(buildConfig52().slice(20).equals(Buffer.alloc(CONFIG_BYTES - 20))).toBe(true);
  });

  test('honours a custom profile', () => {
    const b = buildConfig52({ m: 2048, n: 2048, k: 1024, rank: 256, hashTile: 16, rows: [0], cols: [0, 1, 2] });
    expect(b.readUInt32LE(0)).toBe(2048);
    expect(b.readUInt16LE(12)).toBe(256);
    expect(b.readUInt16LE(16)).toBe(1);
    expect(b.readUInt16LE(18)).toBe(3);
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
