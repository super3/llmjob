'use strict';

const PP = require('../src/shared/miner/plainProof');
const M = require('../src/shared/miner/merkle');

// The bincode encoding of a share's proof. Confirmed against the live pool:
// with the wrong shape it answered "not a valid PlainProof (tried current and
// legacy V1 formats)", and with this one it parsed the proof, checked the
// pattern, verified both Merkle proofs and got as far as judging the hash
// against the target.

describe('bincode primitives', () => {
  // Fixed-width, little-endian. Not varint: the verifier reads with
  // with_fixint_encoding().
  test('a usize is eight little-endian bytes', () => {
    expect(PP.u64le(0).toString('hex')).toBe('0000000000000000');
    expect(PP.u64le(1).toString('hex')).toBe('0100000000000000');
    expect(PP.u64le(2048).toString('hex')).toBe('0008000000000000');
  });

  test('a Vec<usize> is a count then the values', () => {
    expect(PP.seqOfU64([7, 9]).toString('hex'))
      .toBe('0200000000000000' + '0700000000000000' + '0900000000000000');
    expect(PP.seqOfU64([])).toHaveLength(8);
  });

  // leaf_data is serialised as Vec<&[u8]>, so each chunk carries its OWN length
  // as well as the outer count.
  test('a Vec of byte slices length-prefixes every element', () => {
    const out = PP.seqOfByteVecs([Buffer.from([1, 2]), Buffer.from([3])]);
    expect(out.toString('hex')).toBe(
      '0200000000000000' + '0200000000000000' + '0102' + '0100000000000000' + '03');
  });

  // A Digest is [u8; 32], a fixed-size array. serde treats that as a tuple, so
  // it gets NO length prefix of its own -- unlike leaf_data's elements.
  test('a Vec<Digest> prefixes only the count', () => {
    const d = Buffer.alloc(32, 0xab);
    const out = PP.seqOfDigests([d, d]);
    expect(out).toHaveLength(8 + 64);
    expect(out.slice(0, 8).toString('hex')).toBe('0200000000000000');
    expect(out.slice(8, 40)).toEqual(d);
  });

  test('a digest of the wrong width is refused', () => {
    expect(() => PP.seqOfDigests([Buffer.alloc(31)])).toThrow(/32 bytes/);
  });
});

describe('a serialised PlainProof', () => {
  const KEY = Buffer.alloc(32, 5);
  const data = Buffer.alloc(8 * 1024);
  for (let i = 0; i < data.length; i++) data[i] = (i * 7) & 0xff;
  const layers = M.buildLayers(KEY, data);
  const proof = M.multiLeafProof(KEY, data, layers, [0, 1]);
  const pp = {
    m: 4096, n: 4096, k: 2048, noiseRank: 128,
    a: { proof, rowIndices: [0, 8, 64, 72] },
    bt: { proof, rowIndices: [0, 1, 8, 9, 32, 33, 40, 41] },
  };

  test('begins with the four dimensions', () => {
    const b = PP.serializePlainProof(pp);
    expect(b.readBigUInt64LE(0)).toBe(4096n);
    expect(b.readBigUInt64LE(8)).toBe(4096n);
    expect(b.readBigUInt64LE(16)).toBe(2048n);
    expect(b.readBigUInt64LE(24)).toBe(128n);
  });

  // The trailing field is moe: Option<_>, and this miner only mines dense jobs.
  // The verifier also accepts a legacy V1 blob, which is these bytes without
  // that tag -- we always write the current form.
  test('ends with the None tag for the MoE field', () => {
    const b = PP.serializePlainProof(pp);
    expect(b[b.length - 1]).toBe(PP.BINCODE_OPTION_NONE);
    expect(PP.BINCODE_OPTION_NONE).toBe(0);
  });

  test('carries both matrices and their row indices', () => {
    const b = PP.serializePlainProof(pp);
    const one = PP.serializeMatrixMerkleProof(pp.a);
    expect(b.includes(one)).toBe(true);
    // 8 KiB of operand rides along in each half, so the encoding is large.
    expect(b.length).toBeGreaterThan(2 * 2 * 1024);
  });

  test('the base64 form decodes back to the same bytes', () => {
    const b64 = PP.encodePlainProof(pp);
    expect(Buffer.from(b64, 'base64')).toEqual(PP.serializePlainProof(pp));
  });

  test('a leaf that is not a whole chunk is refused', () => {
    const bad = { ...pp, a: { proof: { ...proof, leafData: [Buffer.alloc(10)] }, rowIndices: [0] } };
    expect(() => PP.serializePlainProof(bad)).toThrow(/1024 bytes/);
  });

  test('a root of the wrong width is refused', () => {
    const bad = { ...pp, a: { proof: { ...proof, root: Buffer.alloc(31) }, rowIndices: [0] } };
    expect(() => PP.serializePlainProof(bad)).toThrow(/32 bytes/);
  });
});
