'use strict';

// A genuine hit, built in JS the way the device builds one, for stand-in cores: real Merkle proofs over
// the operands under the job's job_key, the salted seed chain, the noise, the cumulative fold and the
// transcript hash. Small profiles only (m and n of a few hundred): the operands are held whole.
//
// hitFor(profile, header, { bound, jobId, salt }) tries each valid region in turn and returns the first
// whose hash meets `bound` (default: any), or null when none does. `A` and `B` replace the operands
// (m x k and n x k bytes), and `tamper(jackpot)` may change the transcript's 16 words before they are
// hashed, as a card whose fold is wrong would: the proofs and seeds stay right.

const path = require('path');
const S = path.join(__dirname, '..', '..', 'src', 'shared', 'miner') + path.sep;
const M = require(S + 'merkle');
const {
  buildConfig52, regionToTile, leBytesToBigInt, SEED_SALT_A, SEED_SALT_B, bindMessage,
} = require(S + 'pearlhash');
const { hash, keyedHash } = require(S + 'blake3');
const { computeNoiseForIndices } = require(S + 'noise');
const { foldTranscript, transcriptBytes } = require(S + 'reference');

// int7 operands that differ by row and column, so a wrong row or k offset changes the hash.
function operand(rows, k, mul) {
  const b = Buffer.alloc(rows * k);
  for (let i = 0; i < b.length; i++) b[i] = (((i * mul) % 61) - 30) & 0xff;
  return b;
}

function side(jobKey, data, layers, rows, k) {
  const p = M.multiLeafProof(jobKey, data, layers, M.leafIndicesFromRows(rows, k));
  return {
    leafIndices: p.leafIndices, leafData: Buffer.concat(p.leafData),
    siblings: Buffer.concat(p.siblings), root: p.root, totalLeaves: p.totalLeaves,
  };
}

function hitFor(profile, header, { bound = (1n << 256n) - 1n, jobId = 'job', salt = 0, A: a, B: b, tamper } = {}) {
  const { k, rank, m, n } = profile;
  const jobKey = hash(Buffer.concat([Buffer.from(header), buildConfig52(profile)]));
  const A = a || operand(m, k, 7), B = b || operand(n, k, 11);
  const layersA = M.buildLayers(jobKey, A), layersB = M.buildLayers(jobKey, B);
  const rootA = M.rootOf(layersA), rootB = M.rootOf(layersB);
  const bSeed = hash(Buffer.concat([jobKey, keyedHash(SEED_SALT_B, bindMessage(rootB, n))]));
  const aSeed = hash(Buffer.concat([bSeed, keyedHash(SEED_SALT_A, bindMessage(rootA, m))]));
  const regions = (m / 16) * (n / 16);   // valid offsets: the patterns' own bits clear (regionToTile)
  const idx = Array.from({ length: 16 }, (_, i) => i);
  for (let nonce = 0; nonce < regions; nonce++) {
    const { rows, cols } = regionToTile(nonce, profile);
    if (rows[rows.length - 1] >= m || cols[cols.length - 1] >= n) continue;
    const { noiseA, noiseB } = computeNoiseForIndices({ k, rank, aSeed, bSeed, rowIndices: rows, colIndices: cols });
    const At = new Int8Array(16 * k), Bt = new Int8Array(16 * k);
    rows.forEach((r, i) => At.set(new Int8Array(A.buffer, A.byteOffset + r * k, k), i * k));
    cols.forEach((c, i) => Bt.set(new Int8Array(B.buffer, B.byteOffset + c * k, k), i * k));
    const jackpot = foldTranscript({ A: At, Bt, noiseA, noiseB, rows: idx, cols: idx, k, rank });
    if (tamper) tamper(jackpot);
    const jackpotHash = keyedHash(aSeed, transcriptBytes(jackpot));
    if (leBytesToBigInt(jackpotHash) > bound) continue;
    return {
      jobId, nonce, salt, jackpotHash, aSeed: aSeed.toString('hex'), bSeed: bSeed.toString('hex'),
      proofA: side(jobKey, A, layersA, rows, k), proofBt: side(jobKey, B, layersB, cols, k),
    };
  }
  return null;
}

module.exports = { hitFor };
