'use strict';

const { hash, keyedHash } = require('../src/shared/miner/blake3');
const { buildConfig52, ROWS_PATTERN, COLS_PATTERN, JACKPOT_BUCKETS } = require('../src/shared/miner/pearlhash');

// DEVICE PARITY VECTORS, captured from a CI-built pearl_core.node running on a
// real RTX 4090. The JS below recomputes the whole device pipeline and must
// reproduce them exactly.
//
// The profile is deliberately small in m/n/k so the oracle finishes in
// milliseconds, but STRUCTURALLY identical to mainnet: k/rank = 16 chunks (one
// per transcript lane) and the real 4x8 tile patterns. Mainnet's own k = 2048
// would need ~1M keyed hashes in JS per run, which is not a unit test.
//
// This is the check that closes the loop. The JS rests on a BLAKE3 that passes
// the official published vectors, so agreement means the GPU computes what we
// believe. Every bug this project hit was silent — a search that did not vary, a
// collapsed nonce space, identical seeds, a single-chunk tree hash, and a config
// block that was wrong in every field. A frozen parity vector is what makes the
// next one loud.

const PROFILE = { k: 256, rank: 16, mmaType: 0, m: 128, n: 128 };

// Captured 2026-08-24, RTX 4090, driver 560.94.
const DEVICE = {
  aSeed: '2656d847de13920c520f4295be9e7a9dd1ee850c75dcd63dbafb689e47af919b',
  bSeed: 'e76b85096f9f16c67826e5bd8f0d898c1dee9effe74757a5bd53b70ca1eeef52',
  transcript0: 'fdcff21840ceb8e8ae14c1fee9eae8ef54c37cf6865dabecc01b76f0783c44fb'
    + '8b952cf56c5b1ff8971cccf35d2c3a0caaa86ce048cf5007095a18182c1274f1',
  regions: {
    0: '1d0e4c9fe3a44f7c46ba4aa03f612d2108c7cdf31500df594de56b1914235fd6',
    4096: '1e87df1d193c607596c39cf8fd601fc479c55e2775d268aecbc3cc29f48f39f0',
    8192: '3b507df09fd142fc39acda103e58d79ef32d7e0cd5079c629367bd32fed51fb4',
    12288: '424f5d6e45bfaf3198c1a7113845a4c20586b4bf645483b5423c0448c2b68ed3',
  },
};

const HEADER = Buffer.alloc(76, 1);
const rotl13 = (x) => (((x << 13) | (x >>> 19)) >>> 0);

function noiseDraw(seed, index) {
  const idx = Buffer.alloc(8);
  idx.writeBigUInt64LE(BigInt(index));
  return (keyedHash(seed, idx)[0] % 127) - 63;
}

function genNoise(seed, count, base) {
  const out = new Int8Array(count);
  for (let i = 0; i < count; i++) out[i] = noiseDraw(seed, base + i);
  return out;
}

const raw = (i8) => Buffer.from(i8.buffer, i8.byteOffset, i8.byteLength);

let CACHE = null;
function derive() {
  if (CACHE) return CACHE;
  const { m, n, k, rank } = PROFILE;
  const jobKey = keyedHash(Buffer.alloc(32), Buffer.concat([HEADER, buildConfig52(PROFILE)]));
  const A = genNoise(jobKey, m * k, 0);
  // 2**40, not 1<<40: JS bit shifts are 32-bit and would give 0, colliding B with A.
  const B = genNoise(jobKey, n * k, 2 ** 40);
  const bSeed = hash(Buffer.concat([jobKey, keyedHash(jobKey, raw(B))]));
  const aSeed = hash(Buffer.concat([bSeed, keyedHash(jobKey, raw(A))]));
  // The operands are materialised once per job on the device too.
  const Ap = new Int32Array(m * k);
  const Bp = new Int32Array(n * k);
  const EAL = genNoise(bSeed, m * rank, 0);
  const EAR = genNoise(bSeed, rank * k, 1 << 20);
  const EBL = genNoise(aSeed, n * rank, 0);
  const EBR = genNoise(aSeed, rank * k, 1 << 20);
  for (let r = 0; r < m; r++) {
    for (let kk = 0; kk < k; kk++) {
      let v = A[r * k + kk];
      for (let j = 0; j < rank; j++) v += EAL[r * rank + j] * EAR[j * k + kk];
      Ap[r * k + kk] = v;
    }
  }
  for (let c = 0; c < n; c++) {
    for (let kk = 0; kk < k; kk++) {
      let v = B[c * k + kk];
      for (let j = 0; j < rank; j++) v += EBL[c * rank + j] * EBR[j * k + kk];
      Bp[c * k + kk] = v;
    }
  }
  CACHE = { jobKey, aSeed, bSeed, Ap, Bp };
  return CACHE;
}

function foldRegion(region) {
  const { m, n, k, rank } = PROFILE;
  const d = derive();
  const rowOff = region % m;
  const colOff = Math.floor(region / m) % n;
  const chunks = Math.ceil(k / rank);
  const jackpot = new Uint32Array(JACKPOT_BUCKETS);

  for (let chunk = 0; chunk < chunks; chunk++) {
    const k0 = chunk * rank;
    let tileXor = 0;
    for (const r0 of ROWS_PATTERN) {
      for (const c0 of COLS_PATTERN) {
        const r = (r0 + rowOff) % m;
        const c = (c0 + colOff) % n;
        let acc = 0;
        for (let t = 0; t < rank; t++) {
          const kk = k0 + t;
          if (kk >= k) continue;
          acc = (acc + Math.imul(d.Ap[r * k + kk], d.Bp[c * k + kk])) | 0;
        }
        tileXor = (tileXor ^ acc) >>> 0;
      }
    }
    jackpot[chunk % JACKPOT_BUCKETS] = rotl13(jackpot[chunk % JACKPOT_BUCKETS]) ^ tileXor;
  }
  const transcript = Buffer.alloc(64);
  for (let i = 0; i < JACKPOT_BUCKETS; i++) transcript.writeUInt32LE(jackpot[i] >>> 0, i * 4);
  return { transcript, jackpotHash: keyedHash(d.aSeed, transcript) };
}

describe('CUDA core parity — seeds', () => {
  // Matching 32-byte seeds means job_key (and so the corrected config52), both
  // operands, and the keyed BLAKE3 TREE over each all agree with the device.
  test('the device derives the same b_seed and a_seed', () => {
    const d = derive();
    expect(d.bSeed.toString('hex')).toBe(DEVICE.bSeed);
    expect(d.aSeed.toString('hex')).toBe(DEVICE.aSeed);
  });

  test('the two seeds are distinct', () => {
    const d = derive();
    expect(d.aSeed).not.toEqual(d.bSeed);
  });
});

describe('CUDA core parity — the fold', () => {
  for (const region of Object.keys(DEVICE.regions).map(Number)) {
    test('region ' + region + ' matches the device hash', () => {
      expect(foldRegion(region).jackpotHash.toString('hex')).toBe(DEVICE.regions[region]);
    });
  }

  test('region 0 matches the device transcript byte for byte', () => {
    expect(foldRegion(0).transcript.toString('hex')).toBe(DEVICE.transcript0);
  });

  // The bug that made the miner grind at 76% GPU and find nothing.
  test('distinct regions produce distinct hashes', () => {
    const hs = Object.keys(DEVICE.regions).map(Number).map((r) => foldRegion(r).jackpotHash.toString('hex'));
    expect(new Set(hs).size).toBe(hs.length);
  });

  // k/rank = 16 chunks and there are 16 lanes, so every lane is written exactly
  // once and the rotation never wraps. At the old k = 4096 each lane folded
  // twice and the trailing lanes stayed zero.
  test('all sixteen lanes are populated exactly once', () => {
    const t = foldRegion(0).transcript;
    for (let i = 0; i < JACKPOT_BUCKETS; i++) {
      expect(t.readUInt32LE(i * 4)).not.toBe(0);
    }
  });
});
