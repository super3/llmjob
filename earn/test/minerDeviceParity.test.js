'use strict';

const { hash, keyedHash } = require('../src/shared/miner/blake3');
const {
  buildConfig52, JACKPOT_BUCKETS, regionToTile,
  SEED_SALT_A, SEED_SALT_B, bindMessage,
} = require('../src/shared/miner/pearlhash');
const {
  SEED_LABEL_A, SEED_LABEL_B,
  generateUniformRandomMatrix, generatePermutationMatrix, satInt8,
} = require('../src/shared/miner/noise');

// DEVICE PARITY VECTORS, captured from a CI-built pearl_core.node running on a
// real RTX 4090. The JS below recomputes the whole device pipeline and must
// reproduce them exactly.
//
// The profile is deliberately small so the oracle runs in milliseconds, but
// STRUCTURALLY it is the real thing: k/rank = 16 chunks (one per transcript
// lane), the mined 16x16 strided tile, valid offsets only, and the operand salt.
//
// Recaptured when the tile went from contiguous 0..15 to rows {0,1,2,3}+8j by
// cols {0,1}+8i: the pattern is hashed into job_key, so the seeds changed with
// it, and the fold now reads each region out of four lanes' own accumulators.
// Captured at col_batch 16 (two batches a salt), so the regions include column
// indices past 16 and every row and column sub-offset of a warp tile.
//
// This is the check that closes the loop. The JS rests on a BLAKE3 that passes
// the official published vectors, so agreement means the GPU computes what we
// believe. Every bug this project hit was silent -- a search that did not vary,
// a collapsed nonce space, identical seeds, a single-chunk tree hash, a config
// block wrong in every field, a job_key hashed keyed instead of unkeyed, a
// sparse noise selector reconstructed as a dense factor, a partial table strided
// by the wrong row count, and a tile pattern that disagreed between the header
// and the oracle. None produced an error message; this is what makes the next
// one loud.
//
// The vectors span several SALTS on purpose. The operands are re-drawn whenever
// the region space is exhausted, and that mechanism is as capable of silent
// breakage as anything else here.
const PROFILE = { k: 512, rank: 32, mmaType: 0, m: 512, n: 512 };
const { m, n, k, rank } = PROFILE;

const DEVICE = {
  aSeed: '2311ccf1262161fcaf096d5f4d6bef16816cac4c56743d413460d819d266cdb7',
  bSeed: '06949add2af428da0353d85140466cc28d36534f584fca71bdb19e2c8aa98cf5',
  // salt, region -> jackpot hash
  regions: [
    [0, 198, '23e8428b4da026b0d0efbb550e959e176998473de2682e3de4fce2f8f2047d02'],
    [0, 528, 'caf0096df932a4d4eec1a67ebf30761da5ca1d9bc2129ca447d93964b4686500'],
    // Salts after the first are RESTAMPS, not fresh draws (see forSalt).
    [1, 36, '8b492c720cdf8ec2ac307a96d4e50cc46607164b8d33c282d79ecf0588183c00'],
    [1, 636, '8cd6a4ef48bb311c01b3b00d6e7d076d19234102513d6c23fb44b997e2eb0402'],
    [2, 21, '7dd6ee2d0628b9064f8dbfd41b8202a641d8f26e106f9ba7e7788233d7530f00'],
    [2, 517, '6d69763da21187bb6c1ad66f455941e6aac28762fc8e78ef381157f2d2dbce02'],
    [3, 99, 'a9c0548621763e4d6ed6d0297125925c182813e045fe032ecb7fe8f33c8d3b01'],
    [3, 580, 'a3458250b0d203ffaf87bf321559c24fe53c2baf99446b5b3e64588d34e2f302'],
  ],
};

const HEADER = Buffer.alloc(76, 1);
const rotl13 = (x) => (((x << 13) | (x >>> 19)) >>> 0);
const idx = (len) => Array.from({ length: len }, (_, i) => i);
const jobKey = hash(Buffer.concat([HEADER, buildConfig52(PROFILE)]));

// The miner's own operands. Contents are our choice; the int7 RANGE is not,
// because the noise adds another int7 and the sum must stay inside int8.
//
// The salt sits at bytes 8..15 of the RNG message. At salt 0 that leaves the
// message byte-identical to the unsalted version, which is why introducing it
// did not disturb the earlier vectors.
function genOperand(label, total, salt) {
  const out = Buffer.alloc(total);
  for (let base = 0; base < total; base += 32) {
    const msg = Buffer.alloc(64);
    msg.writeUInt32LE((base / 32 + 1) >>> 0, 0);
    msg.writeBigUInt64LE(BigInt(salt), 8);
    label.copy(msg, 32);
    const h = keyedHash(jobKey, msg);
    for (let i = 0; i < 32 && base + i < total; i++) out[base + i] = ((h[i] % 127) - 63) & 0xff;
  }
  return out;
}

const s8 = (b) => (b > 127 ? b - 256 : b);
const cache = new Map();

// A job's first salt is a full draw. Every later salt in the same job RESTAMPS
// it: B, and so b_seed, stay as drawn, and A's first PEARL_STAMP_BYTES bytes are
// overwritten with six bits of the salt each. A new A root means new noise, which
// is a fresh search space for about a tenth of the cost of drawing both operands
// again. The job here starts at salt 0, as a one-card rig's does.
const STAMP_BYTES = 11;

function forSalt(salt) {
  if (cache.has(salt)) return cache.get(salt);
  const A = genOperand(SEED_LABEL_A, m * k, 0);
  const B = genOperand(SEED_LABEL_B, n * k, 0);
  if (salt > 0) {
    for (let i = 0; i < STAMP_BYTES; i++) A[i] = Number((BigInt(salt) >> BigInt(6 * i)) & 63n);
  }

  // cert-v3: salt each operand root with its dimension before the seed chain.
  // This is the only thing that commits m and n, which config52 does not carry.
  const boundA = keyedHash(SEED_SALT_A, bindMessage(keyedHash(jobKey, A), m));
  const boundB = keyedHash(SEED_SALT_B, bindMessage(keyedHash(jobKey, B), n));
  const bSeed = hash(Buffer.concat([jobKey, boundB]));
  const aSeed = hash(Buffer.concat([bSeed, boundA]));

  const eAL = generateUniformRandomMatrix(SEED_LABEL_A, aSeed, idx(m), rank);
  const eBR = generateUniformRandomMatrix(SEED_LABEL_B, bSeed, idx(n), rank);
  const permA = generatePermutationMatrix(SEED_LABEL_A, aSeed, k, rank);
  const permB = generatePermutationMatrix(SEED_LABEL_B, bSeed, k, rank);

  // Two lookups and a subtract, then saturate: E_AR and E_BL are sparse +-1
  // selectors, not dense factors.
  const noise = (dense, perm, kk) => dense[perm[kk * 2]] - dense[perm[kk * 2 + 1]];
  const Ap = new Int8Array(m * k);
  const Bp = new Int8Array(n * k);
  for (let r = 0; r < m; r++) {
    for (let kk = 0; kk < k; kk++) Ap[r * k + kk] = satInt8(s8(A[r * k + kk]) + noise(eAL[r], permA, kk));
  }
  for (let c = 0; c < n; c++) {
    for (let kk = 0; kk < k; kk++) Bp[c * k + kk] = satInt8(s8(B[c * k + kk]) + noise(eBR[c], permB, kk));
  }

  const v = { aSeed, bSeed, Ap, Bp };
  cache.set(salt, v);
  return v;
}

function foldRegion(region, salt) {
  const { aSeed, Ap, Bp } = forSalt(salt);
  const tile = regionToTile(region, PROFILE);
  const chunks = Math.ceil(k / rank);
  const j = new Uint32Array(JACKPOT_BUCKETS);
  // CUMULATIVE across chunks. The reference declares jackpot_tile outside the
  // chunk loop and never resets it, so the value XORed at chunk c is the dot
  // product over all of k up to that point. Resetting per chunk hashes a
  // different function, and the only symptom is that no pool accepts a share.
  const acc = new Int32Array(tile.rows.length * tile.cols.length);
  for (let chunk = 0; chunk < chunks; chunk++) {
    const k0 = chunk * rank;
    let tileXor = 0;
    for (let ri = 0; ri < tile.rows.length; ri++) {
      for (let ci = 0; ci < tile.cols.length; ci++) {
        const r = tile.rows[ri], c = tile.cols[ci];
        let v = acc[ri * tile.cols.length + ci];
        for (let t = 0; t < rank; t++) v = (v + Ap[r * k + k0 + t] * Bp[c * k + k0 + t]) | 0;
        acc[ri * tile.cols.length + ci] = v;
        tileXor = (tileXor ^ v) >>> 0;
      }
    }
    j[chunk % JACKPOT_BUCKETS] = rotl13(j[chunk % JACKPOT_BUCKETS]) ^ tileXor;
  }
  const t = Buffer.alloc(64);
  for (let i = 0; i < JACKPOT_BUCKETS; i++) t.writeUInt32LE(j[i] >>> 0, i * 4);
  return { transcript: t, jackpotHash: keyedHash(aSeed, t) };
}

describe('CUDA core parity — seeds', () => {
  // Matching 32-byte seeds means job_key (unkeyed, over the contiguous-tile
  // config52), both synthesised operands, the keyed BLAKE3 TREE over each, and
  // the cert-v3 root binding all agree with the device.
  test('the device derives the same b_seed and a_seed', () => {
    const s = forSalt(0);
    expect(s.bSeed.toString('hex')).toBe(DEVICE.bSeed);
    expect(s.aSeed.toString('hex')).toBe(DEVICE.aSeed);
  });

  // b_seed is derived first and feeds a_seed, so swapping them is silent.
  test('the two seeds are distinct', () => {
    const s = forSalt(0);
    expect(s.aSeed).not.toEqual(s.bSeed);
  });

  // Re-drawing the operands must actually change the job, or the search has
  // nowhere to go and re-mines what it already tried.
  test('a new operand draw changes the seeds', () => {
    expect(forSalt(1).aSeed).not.toEqual(forSalt(0).aSeed);
  });
});

describe('CUDA core parity — the fold', () => {
  for (const [salt, region, want] of DEVICE.regions) {
    test('salt ' + salt + ' region ' + region + ' matches the device', () => {
      expect(foldRegion(region, salt).jackpotHash.toString('hex')).toBe(want);
    });
  }

  // The bug that made the miner grind at 76% GPU and find nothing.
  test('distinct regions produce distinct hashes', () => {
    const hs = DEVICE.regions.map(([s, r]) => foldRegion(r, s).jackpotHash.toString('hex'));
    expect(new Set(hs).size).toBe(hs.length);
  });
});

describe('the mirrored pipeline', () => {
  test('every transcript lane is written exactly once', () => {
    // k/rank = 16 chunks against 16 lanes, so the rotation never wraps.
    expect(k / rank).toBe(JACKPOT_BUCKETS);
    const t = foldRegion(3, 0).transcript;
    for (let i = 0; i < JACKPOT_BUCKETS; i++) expect(t.readUInt32LE(i * 4)).not.toBe(0);
  });

  // Saturation should be a guard rail that essentially never fires. If it fired
  // often the operand would have collapsed towards a sign pattern and the work
  // would no longer be useful.
  test('saturation is rare, so the operands are not clipped to signs', () => {
    const { Ap } = forSalt(0);
    let clipped = 0;
    for (let i = 0; i < Ap.length; i++) if (Ap[i] === 127 || Ap[i] === -128) clipped++;
    expect(clipped / Ap.length).toBeLessThan(0.02);
  });
});
