'use strict';

// Turning a core hit into the bytes a pool will accept.
//
// The core hands back the two Merkle proofs already, captured on the device at
// the moment the hit was found. That timing is not an optimisation: the search
// re-draws its operands every few tens of milliseconds, so a proof read back
// afterwards belongs to a different matrix than the hash it is supposed to
// certify. It is also why the proof travels WITH the hit rather than living in
// one buffer on the context — a later hit would overwrite it before this one
// had been read.
//
// What arrives here is flat: leaves concatenated into one buffer, siblings
// likewise. This module unpacks that into the shape the encoder wants, checks
// it, and produces the base64 `plain_proof` field.

const { regionToTile, PROFILE } = require('./pearlhash');
const { verifyProof, leafIndicesFromRows } = require('./merkle');
const { encodePlainProof } = require('./plainProof');

const CHUNK_LEN = 1024;
const DIGEST_LEN = 32;

// One side of the proof, as the native layer emits it. leafData and siblings
// arrive as single concatenated buffers because that is what a device copy
// naturally produces; splitting them is this function's whole job.
function unpackSide(side) {
  if (!side || !side.leafIndices || !side.leafIndices.length) return null;
  const leafIndices = Array.from(side.leafIndices);
  const leafData = [];
  for (let i = 0; i < leafIndices.length; i++) {
    const off = i * CHUNK_LEN;
    if (off + CHUNK_LEN > side.leafData.length) return null;
    leafData.push(Buffer.from(side.leafData.subarray(off, off + CHUNK_LEN)));
  }
  const siblings = [];
  for (let off = 0; off + DIGEST_LEN <= side.siblings.length; off += DIGEST_LEN) {
    siblings.push(Buffer.from(side.siblings.subarray(off, off + DIGEST_LEN)));
  }
  return {
    leafData,
    leafIndices,
    totalLeaves: Number(side.totalLeaves),
    root: Buffer.from(side.root),
    siblings,
  };
}

// The wire form of a hit, or null if it cannot be certified.
//
// The local verify is not belt-and-braces. A proof that does not check out here
// cannot check out at the pool either, and a rejected share costs a round trip
// and counts against the worker — so a hit we cannot prove is worth strictly
// less than no hit at all.
function buildShareProof(hit, jobKey, profile) {
  const p = profile || PROFILE;
  const a = unpackSide(hit && hit.proofA);
  const bt = unpackSide(hit && hit.proofBt);
  if (!a || !bt) return null;
  if (!verifyProof(jobKey, a) || !verifyProof(jobKey, bt)) return null;

  const { rows, cols } = regionToTile(hit.nonce, p);

  // The leaves must be the ones these rows actually live in. verifyProof
  // cannot check that: it shows only that the leaves hash to the root, which
  // is equally true of the WRONG leaves under the same tree.
  //
  // The verifier reads each claimed row as bytes [row*k, row*k+k) out of the
  // leaves we supply, and answers "Failed to extract strip" when they are not
  // covered. That is what a stale row mask on the device produced: the fold was
  // right, the hash was right, the proof verified against its own root, and the
  // rows it described were a quarter of the way up the matrix.
  if (!sameIndices(a.leafIndices, leafIndicesFromRows(rows, p.k))
      || !sameIndices(bt.leafIndices, leafIndicesFromRows(cols, p.k))) {
    return null;
  }

  return encodePlainProof({
    m: p.m,
    n: p.n,
    k: p.k,
    noiseRank: p.rank,
    a: { proof: a, rowIndices: rows },
    bt: { proof: bt, rowIndices: cols },
  });
}

function sameIndices(a, b) {
  return a.length === b.length && a.every((x, i) => x === b[i]);
}

module.exports = { CHUNK_LEN, DIGEST_LEN, unpackSide, buildShareProof, sameIndices };
