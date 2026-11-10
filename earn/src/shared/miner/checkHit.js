'use strict';

// A hit recomputed from scratch, the way the pool's verifier checks a share:
// the Merkle proofs and the rows they cover, the seed chain from the two roots,
// the operands' range ([-64, 64], as the spec checks), the noise, the cumulative fold and the transcript
// hash, which must equal the core's jackpot hash byte for byte.
//
// buildShareProof checks only the first of those. A core whose fold is wrong on
// its card passes it: the proofs are of operands the card drew correctly, and
// the hash it reports is wrong yet meets the target, so every share would be
// sent and rejected, and a pool can ban the address for that. This catches it
// before anything is sent.
//
// Two callers, so they cannot drift apart. PearlMiner checks every hit of a card
// whose fold has not run on its kind of card in a release (core.untested; on
// AMD, RX 6000 and Instinct cards) before it sends one. The probes
// (native/probes/verify-hits.js, native/amd/test/verify-hits-profile.js) check
// a core's first 400 hits with it.
//
// Cost, measured on 2 November with Node 22.23 on a 12-core QEMU VM: 4.6 ms a
// hit at k 2048 and 8.6 ms at k 4096, at the mainnet m and n (the median of 30
// synthetic hits each; buildShareProof alone took 1.8 and 3.4). Of that, the
// Merkle proofs' leaves took about 1.3 and 2.3 ms, the noise's BLAKE3 calls 1.5
// and 3.4, and the fold's 16 x 16 x k MACs 1.4 and 2.2. It runs on the miner's
// event loop, once for each share it sends.

const { hash, keyedHash } = require('./blake3');
const { PROFILE, regionToTile } = require('./pearlhash');
const { verifyProof, leafIndicesFromRows } = require('./merkle');
const { computeNoiseForIndices } = require('./noise');
const { bindRoots, foldTranscript, transcriptBytes } = require('./reference');
const { CHUNK_LEN, unpackSide, sameIndices } = require('./shareProof');

// The tile's own rows and columns, once gathered: 0..15.
const TILE = Array.from({ length: 16 }, (_, i) => i);

// The rows a proof side covers, gathered from its leaves into one buffer, row
// after row. Row r is bytes [r*k, r*k+k) of the operand. Every leaf is there:
// the caller has checked the side's leaves are exactly the ones these rows
// live in.
function gather(side, rows, k) {
  const leaf = new Map(side.leafIndices.map((li, i) => [li, side.leafData[i]]));
  const out = new Int8Array(rows.length * k);
  rows.forEach((r, i) => {
    for (let b = 0; b < k;) {
      const off = r * k + b;
      const chunk = leaf.get(Math.floor(off / CHUNK_LEN));
      const at = off % CHUNK_LEN;
      const len = Math.min(CHUNK_LEN - at, k - b);
      out.set(new Int8Array(chunk.buffer, chunk.byteOffset + at, len), i * k + b);
      b += len;
    }
  });
  return out;
}

function mismatch(hit, jobKey, p) {
  const { k, rank, m, n } = p;
  const a = unpackSide(hit.proofA);
  const bt = unpackSide(hit.proofBt);
  if (!a || !bt || !verifyProof(jobKey, a) || !verifyProof(jobKey, bt)) return 'Merkle proof does not verify';
  const { rows, cols } = regionToTile(hit.nonce, p);
  if (!sameIndices(a.leafIndices, leafIndicesFromRows(rows, k))
      || !sameIndices(bt.leafIndices, leafIndicesFromRows(cols, k))) {
    return 'proof leaves are not the rows of the region';
  }
  const { hashA, hashB } = bindRoots(a.root, bt.root, m, n, p.seedDerivation);
  const bSeed = hash(Buffer.concat([jobKey, hashB]));
  const aSeed = hash(Buffer.concat([bSeed, hashA]));
  if (aSeed.toString('hex') !== hit.aSeed || bSeed.toString('hex') !== hit.bSeed) return 'seed chain mismatch';
  const A = gather(a, rows, k);
  const Bt = gather(bt, cols, k);
  for (const op of [A, Bt]) for (const v of op) if (v < -64 || v > 64) return 'operand outside [-64, 64]';
  const { noiseA, noiseB } = computeNoiseForIndices({ k, rank, aSeed, bSeed, rowIndices: rows, colIndices: cols });
  const jackpot = foldTranscript({ A, Bt, noiseA, noiseB, rows: TILE, cols: TILE, k, rank });
  if (!keyedHash(aSeed, transcriptBytes(jackpot)).equals(Buffer.from(hit.jackpotHash))) {
    return 'jackpot hash mismatch (fold/transcript)';
  }
  return null;
}

// Null when the hit is right, or a few words naming the first thing that
// differs. `jobKey` is the hit's job's, at `profile`, which is the profile at
// the k the core mines (core.k). Whether the hash meets a target is the
// caller's to check. A hit too malformed to read fails rather than throws.
function checkHit(hit, jobKey, profile) {
  try {
    return mismatch(hit, jobKey, profile || PROFILE);
  } catch (e) {
    return 'the check could not read the hit: ' + ((e && e.message) || String(e));
  }
}

module.exports = { checkHit };
