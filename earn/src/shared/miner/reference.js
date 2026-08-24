'use strict';

const { keyedHash, hash } = require('./blake3');
const { buildConfig52, JACKPOT_BUCKETS, ROTL_BITS, leBytesToBigInt } = require('./pearlhash');

// A complete, executable PearlHash in JavaScript.
//
// It is far too slow to mine with — that is not what it is for. It is the ORACLE
// the CUDA core is checked against. The failure mode this exists to prevent is
// specific and nasty: a GPU core with a subtly wrong hash, fold, or endianness
// mines at full speed, reports a healthy hashrate, and every share it finds is
// rejected — or worse, it finds none and looks merely unlucky. There is no error
// message for that. The only defence is a slow implementation you trust, and
// known-answer vectors generated from it.
//
// Built on ./blake3, which is validated against the official BLAKE3 vectors, so
// the foundation is externally verified rather than self-consistent.
//
// SCOPE, STATED HONESTLY: these vectors pin OUR implementation's semantics —
// they make the JS and CUDA sides provably agree, and they catch any future edit
// to either. They do NOT by themselves prove agreement with the Pearl network;
// that needs a share accepted by a real pool, which needs the core compiled on a
// GPU box. What they buy is that when that day comes, a rejection points at the
// protocol rather than at arithmetic.

// Derive the per-job key: blake3(header76 ‖ config52), unkeyed.
function jobKey(header, profile) {
  return hash(Buffer.concat([Buffer.from(header), buildConfig52(profile)]));
}

// Pad an operand to a whole number of 1024-byte chunks, as the spec requires
// before committing it.
function pad1024(bytes) {
  const b = Buffer.from(bytes);
  const padded = Math.ceil(b.length / 1024) * 1024 || 1024;
  const out = Buffer.alloc(padded);
  b.copy(out);
  return out;
}

// The operand commitments and the two seeds derived from them.
//   hash_a = blake3(pad1024(A), key=job_key)
//   hash_b = blake3(pad1024(Bᵀ), key=job_key)
//   b_seed = blake3(job_key ‖ hash_b)
//   a_seed = blake3(b_seed ‖ hash_a)
// The ORDER matters and is not symmetric: b_seed is derived first and then feeds
// a_seed, so swapping the two silently produces a different transcript.
function deriveSeeds(key, A, Bt) {
  const hashA = keyedHash(key, pad1024(A));
  const hashB = keyedHash(key, pad1024(Bt));
  const bSeed = hash(Buffer.concat([key, hashB]));
  const aSeed = hash(Buffer.concat([bSeed, hashA]));
  return { hashA, hashB, bSeed, aSeed };
}

// One int7 noise draw: keyed BLAKE3 over the little-endian index, mapped into
// [-63, 63]. Mirrors noise_draw() in pearl_kernel.cu exactly — the range is
// chosen so products stay inside int32 without saturating.
function noiseDraw(seed, index) {
  const idx = Buffer.alloc(8);
  idx.writeBigUInt64LE(BigInt(index));
  const h = keyedHash(seed, idx);
  return (h[0] % 127) - 63;
}

// A low-rank noise factor, [rows × rank], drawn from a seed. E_A = E_AL·E_AR and
// E_B = E_BL·E_BR are each built from two of these, which is what lets the GPU
// regenerate noise on the fly instead of storing a dense matrix.
function noiseFactor(seed, rows, rank, indexBase) {
  const out = new Int8Array(rows * rank);
  for (let i = 0; i < rows * rank; i++) out[i] = noiseDraw(seed, indexBase + i);
  return out;
}

// Rotate a 32-bit lane left by 13 — the transcript fold's mixing step.
function rotl13(x) {
  return (((x << ROTL_BITS) | (x >>> (32 - ROTL_BITS))) >>> 0);
}

// The heart of it: accumulate C in `rank`-sized chunks and fold the mandated
// sub-tile of each chunk into a 16-lane transcript.
//
//   jackpot[chunk % 16] = rotl13(jackpot[chunk % 16]) ^ xor(tile)
//
// The noised operands are reconstructed on the fly:
//   A'[r,kk] = A[r,kk] + Σ_j E_AL[r,j]·E_AR[j,kk]
//   B'[c,kk] = B[c,kk] + Σ_j E_BL[c,j]·E_BR[j,kk]
function foldTranscript(opts) {
  const { A, Bt, EAL, EAR, EBL, EBR, rows, cols, k, rank } = opts;
  const chunks = Math.ceil(k / rank);
  const jackpot = new Uint32Array(JACKPOT_BUCKETS);

  for (let chunk = 0; chunk < chunks; chunk++) {
    const k0 = chunk * rank;
    let tileXor = 0;
    for (const r of rows) {
      for (const c of cols) {
        let acc = 0;
        for (let kk = k0; kk < k0 + rank && kk < k; kk++) {
          let a = A[r * k + kk];
          let b = Bt[c * k + kk];
          for (let j = 0; j < rank; j++) {
            a += EAL[r * rank + j] * EAR[j * k + kk];
            b += EBL[c * rank + j] * EBR[j * k + kk];
          }
          // Wrap to int32 exactly as the device accumulator does.
          acc = (acc + Math.imul(a, b)) | 0;
        }
        tileXor = (tileXor ^ acc) >>> 0;
      }
    }
    const lane = chunk % JACKPOT_BUCKETS;
    jackpot[lane] = rotl13(jackpot[lane]) ^ tileXor;
  }
  return jackpot;
}

// The 64-byte transcript, little-endian per lane, then hashed under a_seed.
function transcriptBytes(jackpot) {
  const out = Buffer.alloc(JACKPOT_BUCKETS * 4);
  for (let i = 0; i < JACKPOT_BUCKETS; i++) out.writeUInt32LE(jackpot[i] >>> 0, i * 4);
  return out;
}

// Run the whole pipeline for one job and one set of operands. Returns everything
// a test or a submit needs, including the intermediate seeds so a mismatch can
// be localised to a stage rather than just "the answer is wrong".
function computePow(opts) {
  const { header, profile, A, Bt } = opts;
  const key = jobKey(header, profile);
  const { hashA, hashB, bSeed, aSeed } = deriveSeeds(key, A, Bt);

  const { k, rank, rows, cols } = profile;
  const mRows = Math.max(...rows) + 1;
  const nCols = Math.max(...cols) + 1;
  const EAL = noiseFactor(bSeed, mRows, rank, 0);
  const EAR = noiseFactor(bSeed, rank, k, 1 << 20);
  const EBL = noiseFactor(aSeed, nCols, rank, 0);
  const EBR = noiseFactor(aSeed, rank, k, 1 << 20);

  const jackpot = foldTranscript({ A, Bt, EAL, EAR, EBL, EBR, rows, cols, k, rank });
  const transcript = transcriptBytes(jackpot);
  const jackpotHash = keyedHash(aSeed, transcript);

  return {
    jobKey: key, hashA, hashB, bSeed, aSeed,
    jackpot: Array.from(jackpot), transcript, jackpotHash,
    value: leBytesToBigInt(jackpotHash),
  };
}

module.exports = {
  jobKey, pad1024, deriveSeeds, noiseDraw, noiseFactor,
  rotl13, foldTranscript, transcriptBytes, computePow,
};
