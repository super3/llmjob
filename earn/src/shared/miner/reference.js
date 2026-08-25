'use strict';

const { keyedHash, hash } = require('./blake3');
const {
  buildConfig52, JACKPOT_BUCKETS, ROTL_BITS, leBytesToBigInt,
  SEED_SALT_A, SEED_SALT_B, bindMessage,
} = require('./pearlhash');
const { computeNoiseForIndices, satInt8 } = require('./noise');

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
// the foundation is externally verified rather than self-consistent. The noise
// construction in ./noise is a direct port of the reference implementation's
// pearl_noise.rs rather than an inference from prose — an earlier version of
// this file guessed at it and got a structure that was internally consistent and
// completely wrong.

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

// Bind the operand roots before they enter the seed chain.
//
// Under cert-v3 ('salted') each root is re-hashed under a domain-separation salt
// together with its dimension, which is what commits m and n — they are the
// miner's own choice and are deliberately absent from config52, so nothing else
// in the chain pins them. 'legacy' passes the raw roots straight through.
function bindRoots(hashA, hashB, m, n, mode) {
  if (mode === 'legacy') return { hashA, hashB };
  return {
    hashA: keyedHash(SEED_SALT_A, bindMessage(hashA, m)),
    hashB: keyedHash(SEED_SALT_B, bindMessage(hashB, n)),
  };
}

// The operand commitments and the two seeds derived from them.
//   hash_a = blake3(pad1024(A), key=job_key)
//   hash_b = blake3(pad1024(Bᵀ), key=job_key)
//   (hash_a, hash_b) = bind_roots(...)          [cert-v3 only]
//   b_seed = blake3(job_key ‖ hash_b)
//   a_seed = blake3(b_seed ‖ hash_a)
// The ORDER matters and is not symmetric: b_seed is derived first and then feeds
// a_seed, so swapping the two silently produces a different transcript. Both
// hashes here are UNKEYED — the reference passes None as the key, and keying
// them with a zero key (which is a different function in BLAKE3, not the same
// one) is a mistake with no visible symptom.
function deriveSeeds(key, A, Bt, profile) {
  const p = profile || {};
  const rawA = keyedHash(key, pad1024(A));
  const rawB = keyedHash(key, pad1024(Bt));
  const { hashA, hashB } = bindRoots(rawA, rawB, p.m, p.n, p.seedDerivation);
  const bSeed = hash(Buffer.concat([key, hashB]));
  const aSeed = hash(Buffer.concat([bSeed, hashA]));
  return { hashA: rawA, hashB: rawB, boundA: hashA, boundB: hashB, bSeed, aSeed };
}

// Rotate a 32-bit lane left by 13 — the transcript fold's mixing step.
function rotl13(x) {
  return (((x << ROTL_BITS) | (x >>> (32 - ROTL_BITS))) >>> 0);
}

// Accumulate C in `rank`-sized chunks and fold the mandated sub-tile of each
// chunk into a 16-lane transcript.
//
//   jackpot[chunk % 16] = rotl13(jackpot[chunk % 16]) ^ xor(tile)
//
// The operands are noised first and saturated to int8, which is what the
// reference's noising kernel hands to the main GEMM:
//
//   A'[r,kk] = sat_i8(A[r,kk] + noiseA[r,kk])
//   B'[c,kk] = sat_i8(B[c,kk] + noiseB[c,kk])
//
// Both are int7-ranged, so the product is well inside int32 and the accumulator
// is a real accumulator rather than a wraparound artefact.
function foldTranscript(opts) {
  const { A, Bt, noiseA, noiseB, rows, cols, k, rank } = opts;
  const chunks = Math.ceil(k / rank);
  const jackpot = new Uint32Array(JACKPOT_BUCKETS);

  // Materialise the noised tile rows and columns once.
  const Ap = rows.map((r, ri) => {
    const out = new Int8Array(k);
    for (let kk = 0; kk < k; kk++) out[kk] = satInt8(A[r * k + kk] + noiseA[ri][kk]);
    return out;
  });
  const Bp = cols.map((c, ci) => {
    const out = new Int8Array(k);
    for (let kk = 0; kk < k; kk++) out[kk] = satInt8(Bt[c * k + kk] + noiseB[ci][kk]);
    return out;
  });

  // The tile accumulator is CUMULATIVE across chunks. It is declared here, not
  // inside the loop, exactly as the reference does:
  //
  //   let mut jackpot_tile = vec![vec![0; tile_w]; tile_h];   // outside
  //   for ll in (rank..=k).step_by(rank) {
  //       ... jackpot_tile[u][v] += a_noised[..][l] * b_noised_t[..][l];
  //       let xored_tile = jackpot_tile.iter().flatten().fold(0u32, |a, &x| a ^ x as u32);
  //
  // So the value XORed at chunk c is the dot product over ALL of k up to that
  // point, not that chunk's slice. Resetting per chunk -- which this did --
  // computes a completely different function, and the only symptom is that no
  // pool ever accepts a share.
  const tile = new Int32Array(rows.length * cols.length);

  for (let chunk = 0; chunk < chunks; chunk++) {
    const k0 = chunk * rank;
    const kEnd = Math.min(k0 + rank, k);
    let tileXor = 0;
    for (let ri = 0; ri < rows.length; ri++) {
      for (let ci = 0; ci < cols.length; ci++) {
        let acc = tile[ri * cols.length + ci];
        for (let kk = k0; kk < kEnd; kk++) acc = (acc + Ap[ri][kk] * Bp[ci][kk]) | 0;
        tile[ri * cols.length + ci] = acc;
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
  const { hashA, hashB, bSeed, aSeed } = deriveSeeds(key, A, Bt, profile);

  const { k, rank, rows, cols } = profile;
  const { noiseA, noiseB } = computeNoiseForIndices({
    k, rank, aSeed, bSeed, rowIndices: rows, colIndices: cols,
  });

  const jackpot = foldTranscript({ A, Bt, noiseA, noiseB, rows, cols, k, rank });
  const transcript = transcriptBytes(jackpot);
  const jackpotHash = keyedHash(aSeed, transcript);

  return {
    jobKey: key, hashA, hashB, bSeed, aSeed,
    jackpot: Array.from(jackpot), transcript, jackpotHash,
    value: leBytesToBigInt(jackpotHash),
  };
}

module.exports = {
  jobKey, pad1024, deriveSeeds, bindRoots,
  rotl13, foldTranscript, transcriptBytes, computePow,
};
