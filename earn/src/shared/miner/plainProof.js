'use strict';

const { CHUNK_LEN } = require('./merkle');

// The `plain_proof` field of a share submission: base64 of a bincode-encoded
// PlainProof, as defined in zk-pow/src/ffi/plain_proof.rs.
//
// This is what the pool verifies a share with. It is NOT the 64-byte
// transcript, which is what this miner sent at first and what earned:
//
//   {"code":23,"message":"not a valid PlainProof (tried current and legacy V1
//    formats)"}
//
// A share is a claim that a particular tile of a particular matrix product
// meets the target, so the pool needs the operand data that tile touched --
// carried here as the 1024-byte chunks holding those rows, plus the sibling
// digests that authenticate them against the committed root.
//
// The encoding is bincode with FIXED-WIDTH integers, little-endian:
//   usize          -> u64
//   Vec<T>         -> u64 length, then the elements
//   [u8; N]        -> N bytes, no length prefix (it is a tuple, not a sequence)
//   Option<T>      -> one tag byte, 0 for None
//   leaf_data      -> serialised as Vec<&[u8]>, so each chunk carries its OWN
//                     u64 length as well as the outer count
//
// The verifier accepts a legacy V1 blob too, which is the same bytes without
// the trailing Option tag. We always write the current form.

const BINCODE_OPTION_NONE = 0x00;

function u64le(n) {
  const b = Buffer.alloc(8);
  b.writeBigUInt64LE(BigInt(n));
  return b;
}

function seqOfU64(values) {
  return Buffer.concat([u64le(values.length), ...values.map((v) => u64le(v))]);
}

// Vec<&[u8]>: outer count, then each element length-prefixed.
function seqOfByteVecs(items) {
  const parts = [u64le(items.length)];
  for (const it of items) {
    const b = Buffer.from(it);
    parts.push(u64le(b.length), b);
  }
  return Buffer.concat(parts);
}

// Vec<Digest>: outer count, then 32 raw bytes each. Fixed-size arrays are
// tuples in serde, so they get NO length prefix of their own — unlike leaf_data.
function seqOfDigests(digests) {
  const parts = [u64le(digests.length)];
  for (const d of digests) {
    const b = Buffer.from(d);
    if (b.length !== 32) throw new Error('digest must be 32 bytes, got ' + b.length);
    parts.push(b);
  }
  return Buffer.concat(parts);
}

function serializeMerkleProof(p) {
  for (const leaf of p.leafData) {
    if (Buffer.from(leaf).length !== CHUNK_LEN) {
      throw new Error('leaf data must be exactly ' + CHUNK_LEN + ' bytes');
    }
  }
  const root = Buffer.from(p.root);
  if (root.length !== 32) throw new Error('root must be 32 bytes');
  return Buffer.concat([
    seqOfByteVecs(p.leafData),
    seqOfU64(p.leafIndices),
    u64le(p.totalLeaves),
    root,
    seqOfDigests(p.siblings),
  ]);
}

function serializeMatrixMerkleProof(mp) {
  return Buffer.concat([serializeMerkleProof(mp.proof), seqOfU64(mp.rowIndices)]);
}

// { m, n, k, noiseRank, a, bt } -> the bincode bytes.
//
// `n` is the operand's column count for a dense job. MoE is not produced here:
// this miner only mines standard (non-GROUPED_GEMM) jobs, so the trailing
// Option is always None.
function serializePlainProof(pp) {
  return Buffer.concat([
    u64le(pp.m),
    u64le(pp.n),
    u64le(pp.k),
    u64le(pp.noiseRank),
    serializeMatrixMerkleProof(pp.a),
    serializeMatrixMerkleProof(pp.bt),
    Buffer.from([BINCODE_OPTION_NONE]),
  ]);
}

function encodePlainProof(pp) {
  return serializePlainProof(pp).toString('base64');
}

module.exports = {
  BINCODE_OPTION_NONE,
  u64le, seqOfU64, seqOfByteVecs, seqOfDigests,
  serializeMerkleProof, serializeMatrixMerkleProof,
  serializePlainProof, encodePlainProof,
};
