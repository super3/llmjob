'use strict';

const { hash, keyedHash } = require('../src/shared/miner/blake3');
const { buildConfig52 } = require('../src/shared/miner/pearlhash');

// DEVICE PARITY VECTORS.
//
// These are the values the CUDA core produced on a real RTX 4090, captured from
// a CI-built pearl_core.node run on hardware. The JS below recomputes the whole
// device pipeline — job_key, operand generation, the keyed BLAKE3 TREE over both
// operands, both seeds, the region-offset fold, the final keyed hash — and must
// reproduce them exactly.
//
// This is the check that closes the loop. The JS side rests on a BLAKE3 that
// passes the official published vectors, so agreement here means the GPU is
// computing what we believe it is, not merely something self-consistent. Every
// bug this project has hit was silent — a search that did not vary, a collapsed
// nonce space, identical seeds — and each produced a miner that looked perfectly
// healthy. A frozen parity vector is what makes the next one loud.
//
// If this fails after a kernel change, the kernel changed behaviour. That is the
// point.

const JACKPOT = 16;
const PROFILE = { m: 512, n: 512, k: 128, rank: 32, hashTile: 16 };

// Captured from the device, 2026-08-24, RTX 4090, driver 560.94.
const DEVICE = {
  bSeed: 'a7a7d3799f6835cb6aca3d893574b3568ad9bf5503fe6d44032eaa7d1f1f7cb8',
  aSeed: 'b51a12f98a738a5c02b3d6c6f04724afbc648dc6fd6eff15666b21039c7a2763',
  transcript0: '86d89fe150fa81f369766cedf7dd3902' + '0'.repeat(96),
  regions: {
    0: '5cbd348d52beba4fe4d530b368f54081b244dae86aaf7c669e1291ea33f9908f',
    1: 'd36ded066aa5ea0549db36084e6def522430661422242432fcfa2a72511731fc',
    2: '05763b4acf2cd74aa2db84fff26f5705c7de0c92416b4ac6752139401fc3061c',
    3: 'eddc8e05f2616415adf5b131b7ba915d897f1dfb7fe4dd34d7779d7ff01bdb9a',
  },
};

function rotl13(x) { return (((x << 13) | (x >>> 19)) >>> 0); }

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

// Mirrors build_patterns() in pearl_host.cu.
function buildPatterns(rowsCount, colsCount) {
  const rows = [];
  for (let i = 0; i < rowsCount; i++) rows.push(i * 8);
  const cols = [];
  for (let base = 0; cols.length < colsCount; base += 8) {
    cols.push(base);
    if (cols.length < colsCount) cols.push(base + 1);
  }
  return { rows, cols };
}

// The seeds are job-scoped, so derive them once for the whole suite rather than
// per region — this recomputes 128k keyed hashes and a 64-chunk tree.
let CACHE = null;
function seeds(header) {
  if (CACHE) return CACHE;
  const { m, n, k } = PROFILE;
  const cfg = buildConfig52({ ...PROFILE, rows: [0, 8], cols: new Array(64) });
  // The device hashes header ‖ config52 under an all-zero key, which is the
  // unkeyed construction for this input.
  const jobKey = keyedHash(Buffer.alloc(32), Buffer.concat([header, cfg]));
  const A = genNoise(jobKey, m * k, 0);
  // 2**40, not 1<<40: the device uses a 64-bit index base and JS bit shifts are
  // 32-bit, so `1 << 40` would be 0 and every B value would collide with A.
  const B = genNoise(jobKey, n * k, 2 ** 40);
  const hashA = keyedHash(jobKey, raw(A));
  const hashB = keyedHash(jobKey, raw(B));
  const bSeed = hash(Buffer.concat([jobKey, hashB]));
  const aSeed = hash(Buffer.concat([bSeed, hashA]));
  CACHE = { jobKey, A, B, hashA, hashB, bSeed, aSeed };
  return CACHE;
}

function foldRegion(header, region) {
  const { m, n, k, rank } = PROFILE;
  const s = seeds(header);
  const EAL = genNoise(s.bSeed, m * rank, 0);
  const EAR = genNoise(s.bSeed, rank * k, 1 << 20);
  const EBL = genNoise(s.aSeed, n * rank, 0);
  const EBR = genNoise(s.aSeed, rank * k, 1 << 20);

  const { rows, cols } = buildPatterns(2, 64);
  const rowOff = region % m;
  const colOff = Math.floor(region / m) % n;
  const chunks = Math.ceil(k / rank);
  const jackpot = new Uint32Array(JACKPOT);

  for (let chunk = 0; chunk < chunks; chunk++) {
    const k0 = chunk * rank;
    let tileXor = 0;
    for (const r0 of rows) {
      for (const c0 of cols) {
        const r = (r0 + rowOff) % m;
        const c = (c0 + colOff) % n;
        let acc = 0;
        for (let kk = k0; kk < k0 + rank && kk < k; kk++) {
          let a = s.A[r * k + kk];
          let b = s.B[c * k + kk];
          for (let j = 0; j < rank; j++) {
            a += EAL[r * rank + j] * EAR[j * k + kk];
            b += EBL[c * rank + j] * EBR[j * k + kk];
          }
          acc = (acc + Math.imul(a, b)) | 0;
        }
        tileXor = (tileXor ^ acc) >>> 0;
      }
    }
    jackpot[chunk % JACKPOT] = rotl13(jackpot[chunk % JACKPOT]) ^ tileXor;
  }

  const transcript = Buffer.alloc(64);
  for (let i = 0; i < JACKPOT; i++) transcript.writeUInt32LE(jackpot[i] >>> 0, i * 4);
  return { transcript, jackpotHash: keyedHash(s.aSeed, transcript) };
}

const HEADER = Buffer.alloc(76, 1);

describe('CUDA core parity — seeds', () => {
  // Matching 32-byte seeds means job_key, both operands, and the keyed BLAKE3
  // TREE over 64 KiB of each all agree. The tree is the part that was wrong
  // first time round: the device hashed only a single chunk, which is correct
  // for <=1024 bytes and silently wrong for everything larger.
  test('the device derives the same b_seed and a_seed', () => {
    const s = seeds(HEADER);
    expect(s.bSeed.toString('hex')).toBe(DEVICE.bSeed);
    expect(s.aSeed.toString('hex')).toBe(DEVICE.aSeed);
  });

  // b_seed feeds a_seed; identical seeds meant the derivation was stubbed out.
  test('the two seeds are distinct', () => {
    const s = seeds(HEADER);
    expect(s.aSeed).not.toEqual(s.bSeed);
  });
});

describe('CUDA core parity — the fold', () => {
  for (const region of [0, 1, 2, 3]) {
    test('region ' + region + ' matches the device hash', () => {
      expect(foldRegion(HEADER, region).jackpotHash.toString('hex'))
        .toBe(DEVICE.regions[region]);
    });
  }

  test('region 0 matches the device transcript byte for byte', () => {
    expect(foldRegion(HEADER, 0).transcript.toString('hex')).toBe(DEVICE.transcript0);
  });

  // The bug that made the miner grind at 76% GPU and find nothing: every region
  // produced an identical transcript because the kernel had no region parameter.
  test('distinct regions produce distinct hashes', () => {
    const hashes = [0, 1, 2, 3].map((r) => foldRegion(HEADER, r).jackpotHash.toString('hex'));
    expect(new Set(hashes).size).toBe(4);
  });

  // k/rank = 4 chunks, so only lanes 0..3 are ever touched. A fold that wrote
  // every lane would be folding the wrong number of chunks.
  test('only the lanes the chunk count reaches are populated', () => {
    const t = foldRegion(HEADER, 0).transcript;
    expect(t.readUInt32LE(0)).not.toBe(0);
    expect(t.readUInt32LE(12)).not.toBe(0);
    expect(t.slice(16).every((b) => b === 0)).toBe(true);
  });
});
