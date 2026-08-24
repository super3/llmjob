'use strict';

const { hash, keyedHash } = require('../src/shared/miner/blake3');
const {
  buildConfig52, ROWS_PATTERN, COLS_PATTERN, JACKPOT_BUCKETS,
  SEED_SALT_A, SEED_SALT_B, bindMessage,
} = require('../src/shared/miner/pearlhash');
const {
  SEED_LABEL_A, SEED_LABEL_B, getRandomHash,
  generateUniformRandomMatrix, generatePermutationMatrix, satInt8,
} = require('../src/shared/miner/noise');

// DEVICE PARITY VECTORS, captured from a CI-built pearl_core.node running on a
// real RTX 4090. The JS below recomputes the whole device pipeline and must
// reproduce them exactly.
//
// The profile is deliberately small in m/n/k so the oracle finishes in
// milliseconds, but STRUCTURALLY identical to mainnet: k/rank = 16 chunks (one
// per transcript lane) and the real 4x8 tile patterns.
//
// This is the check that closes the loop. The JS rests on a BLAKE3 that passes
// the official published vectors, so agreement means the GPU computes what we
// believe. Every bug this project hit was silent -- a search that did not vary,
// a collapsed nonce space, identical seeds, a single-chunk tree hash, a config
// block wrong in every field, a job_key hashed keyed instead of unkeyed, and a
// noise construction that reconstructed a sparse selector as a dense factor.
// None of them produced an error message. A frozen parity vector is what makes
// the next one loud.
//
// rank is 32, not 16: the reference requires it to be a power of two AND a
// multiple of the BLAKE3 digest size, because the dense noise factor is
// generated one 32-byte digest at a time.
const PROFILE = { k: 512, rank: 32, mmaType: 0, m: 128, n: 128 };

// Captured 2026-08-24 from a CI-built core on an RTX 4090, driver 560.94,
// after the noise and seed corrections.
const DEVICE = {
  aSeed: '9971330c53c0358910e12f43e85603e07ffaf8703cf9c6948f87f576488e8a68',
  bSeed: 'a1755737e834c0c2cb383c18f37a59d5cbfc34f60cb839e7098acc9964c066bc',
  regions: {
    0: '01bb140ae6bdcbf7557e935a9947a7a955cfc99b01ade7df63c9c1bcd214bd19',
    4096: 'da04ea1350e5c9c087b70adc0cbb1d4cd0ce79f638b01198369cf4be48912316',
    8192: '452b0a1e4086e31daf925e5563c1c591c2693dc94ad5efb58791f31f0ba89801',
    12288: '08a341a3804c2a3cf541a2954e9aa2e7cc401c4539cad59dfc8aead3c1b8ef38',
  },
};

const HEADER = Buffer.alloc(76, 1);
const rotl13 = (x) => (((x << 13) | (x >>> 19)) >>> 0);
const raw = (i8) => Buffer.from(i8.buffer, i8.byteOffset, i8.byteLength);

// The miner's own operands. Contents are our choice; the int7 RANGE is not,
// because the noise adds another int7 and the sum must stay inside int8.
// One digest per 32 bytes, exactly as pearl_gen_operand does it.
function genOperand(key, label, total) {
  const out = new Int8Array(total);
  for (let base = 0; base < total; base += 32) {
    const h = getRandomHash(base / 32, label, key, 0);
    for (let i = 0; i < 32 && base + i < total; i++) {
      out[base + i] = (h[i] % 127) - 63;
    }
  }
  return out;
}

let CACHE = null;
function derive() {
  if (CACHE) return CACHE;
  const { m, n, k, rank } = PROFILE;

  // job_key is UNKEYED. A zero KEY is a different function and gives a
  // different digest -- the bug this test now pins.
  const jobKey = hash(Buffer.concat([HEADER, buildConfig52(PROFILE)]));

  const A = genOperand(jobKey, SEED_LABEL_A, m * k);
  const B = genOperand(jobKey, SEED_LABEL_B, n * k);

  // The commitments are keyed BLAKE3 TREES over 1024-byte chunks, not one long
  // chain. blake3.js implements the real thing and is checked against the
  // official vectors, so this is where that pays off.
  const hashA = keyedHash(jobKey, raw(A));
  const hashB = keyedHash(jobKey, raw(B));

  // cert-v3: salt each root with its dimension before the chain. This is the
  // only thing that commits m and n.
  const boundA = keyedHash(SEED_SALT_A, bindMessage(hashA, m));
  const boundB = keyedHash(SEED_SALT_B, bindMessage(hashB, n));

  const bSeed = hash(Buffer.concat([jobKey, boundB]));
  const aSeed = hash(Buffer.concat([bSeed, boundA]));

  // The whole operand is noised, so a row's index is its position.
  const rowsAll = Array.from({ length: m }, (_, i) => i);
  const colsAll = Array.from({ length: n }, (_, i) => i);
  const eAL = generateUniformRandomMatrix(SEED_LABEL_A, aSeed, rowsAll, rank);
  const eBR = generateUniformRandomMatrix(SEED_LABEL_B, bSeed, colsAll, rank);
  const permA = generatePermutationMatrix(SEED_LABEL_A, aSeed, k, rank);
  const permB = generatePermutationMatrix(SEED_LABEL_B, bSeed, k, rank);

  // Two lookups and a subtract per element, then saturate to int8.
  const noise = (dense, perm, kk) => dense[perm[kk * 2]] - dense[perm[kk * 2 + 1]];
  const Ap = new Int8Array(m * k);
  const Bp = new Int8Array(n * k);
  for (let r = 0; r < m; r++) {
    for (let kk = 0; kk < k; kk++) {
      Ap[r * k + kk] = satInt8(A[r * k + kk] + noise(eAL[r], permA, kk));
    }
  }
  for (let c = 0; c < n; c++) {
    for (let kk = 0; kk < k; kk++) {
      Bp[c * k + kk] = satInt8(B[c * k + kk] + noise(eBR[c], permB, kk));
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
          acc = (acc + d.Ap[r * k + kk] * d.Bp[c * k + kk]) | 0;
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

const CAPTURED = DEVICE.aSeed !== '__PENDING__';
const whenCaptured = CAPTURED ? describe : describe.skip;

whenCaptured('CUDA core parity — seeds', () => {
  // Matching 32-byte seeds means job_key (unkeyed, over the corrected config52),
  // both synthesised operands, the keyed BLAKE3 TREE over each, and the cert-v3
  // root binding all agree with the device.
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

whenCaptured('CUDA core parity — the fold', () => {
  for (const region of Object.keys(DEVICE.regions).map(Number)) {
    test('region ' + region + ' matches the device hash', () => {
      expect(foldRegion(region).jackpotHash.toString('hex')).toBe(DEVICE.regions[region]);
    });
  }

  // The bug that made the miner grind at 76% GPU and find nothing.
  test('distinct regions produce distinct hashes', () => {
    const hs = Object.keys(DEVICE.regions).map(Number)
      .map((r) => foldRegion(r).jackpotHash.toString('hex'));
    expect(new Set(hs).size).toBe(hs.length);
  });
});

// These hold with or without a device capture: they are properties of the
// pipeline, not of any particular card.
describe('the mirrored pipeline', () => {
  test('every transcript lane is written exactly once', () => {
    // k/rank = 16 chunks against 16 lanes, so the rotation never wraps.
    expect(PROFILE.k / PROFILE.rank).toBe(JACKPOT_BUCKETS);
    const t = foldRegion(0).transcript;
    for (let i = 0; i < JACKPOT_BUCKETS; i++) {
      expect(t.readUInt32LE(i * 4)).not.toBe(0);
    }
  });

  test('distinct regions fold to distinct transcripts', () => {
    const hs = [0, 4096, 8192, 12288]
      .map((r) => foldRegion(r).jackpotHash.toString('hex'));
    expect(new Set(hs).size).toBe(hs.length);
  });

  // The noised operands must stay int7-ish: saturation should be a guard rail
  // that essentially never fires, not the main effect. If it fires often, the
  // operand has collapsed towards a sign pattern and the work is not useful.
  test('saturation is rare, so the operands are not clipped to signs', () => {
    const d = derive();
    let clipped = 0;
    for (let i = 0; i < d.Ap.length; i++) {
      if (d.Ap[i] === 127 || d.Ap[i] === -128) clipped++;
    }
    expect(clipped / d.Ap.length).toBeLessThan(0.02);
  });

  test('the seeds are distinct and the operands are noised', () => {
    const d = derive();
    expect(d.aSeed).not.toEqual(d.bSeed);
    expect(Array.from(d.Ap.slice(0, 64))).not.toEqual(Array.from(d.Bp.slice(0, 64)));
  });
});
