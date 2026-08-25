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
// lane), the contiguous 4x16 tile, valid offsets only, and the operand salt.
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
  aSeed: 'ff6556a61358e74b7adb65f03b81bac8868d330c7a8d386a1460d7d7f57dc024',
  bSeed: 'edb3483789036d710f24aa23a16be7a471def174b90a748783bceaab724614e7',
  // salt, region -> jackpot hash
  regions: [
    [0, 3, '0f826edbfe51e32eac38f100d578d7b7da6858a959338b5b691b2edbff313503'],
    [0, 629, 'd502d7e96219fff91e953c6d92c69004885d43879062329ec2b8e9b398cc5302'],
    [1, 9, '16b9a97be6b5d62a0d432620b86f208f2b90e6a363ba4449c964d7fc98be7e02'],
    [1, 637, '4eadf6ff2b144561ed5e4b2cbd51bdfefab8b630e931db0a54650c7fc6e02e02'],
    [2, 57, '91d7162abf7a4e028a49f1e32dc1ff761d2f998ad2205743b8697802c527cc02'],
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

function forSalt(salt) {
  if (cache.has(salt)) return cache.get(salt);
  const A = genOperand(SEED_LABEL_A, m * k, salt);
  const B = genOperand(SEED_LABEL_B, n * k, salt);

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
