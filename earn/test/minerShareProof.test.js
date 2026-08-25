'use strict';

const { unpackSide, buildShareProof, sameIndices, CHUNK_LEN } = require('../src/shared/miner/shareProof');
const { hash, keyedHash } = require('../src/shared/miner/blake3');
const { buildConfig52, regionToTile } = require('../src/shared/miner/pearlhash');
const M = require('../src/shared/miner/merkle');

// The last gate before a share leaves the machine. Everything it rejects is
// something that would otherwise cost a pool round trip and count against the
// worker, and every rejection here is a failure mode this miner actually hit.

// Small enough to build real trees in milliseconds, same shape as mainnet:
// 1024-byte leaves, a power-of-two leaf count, and the 16x16 tile.
const TINY = { k: 1024, rank: 128, mmaType: 0, m: 64, n: 64 };
const HEADER = Buffer.alloc(76, 3);
const jobKey = hash(Buffer.concat([HEADER, buildConfig52(TINY)]));

function matrix(seed, bytes) {
  return keyedHash(Buffer.alloc(32, seed), Buffer.alloc(64), bytes);
}

function side(mat, rows, k) {
  const layers = M.buildLayers(jobKey, mat);
  const p = M.multiLeafProof(jobKey, mat, layers, M.leafIndicesFromRows(rows, k));
  return {
    leafIndices: p.leafIndices,
    leafData: Buffer.concat(p.leafData),
    siblings: Buffer.concat(p.siblings),
    root: p.root,
    totalLeaves: p.totalLeaves,
  };
}

function hitAt(nonce = 0) {
  const { rows, cols } = regionToTile(nonce, TINY);
  return {
    nonce,
    proofA: side(matrix(7, TINY.m * TINY.k), rows, TINY.k),
    proofBt: side(matrix(9, TINY.n * TINY.k), cols, TINY.k),
  };
}

describe('unpackSide', () => {
  test('splits the flat device buffers into leaves and siblings', () => {
    const s = unpackSide(hitAt().proofA);
    expect(s.leafData).toHaveLength(s.leafIndices.length);
    expect(s.leafData[0]).toHaveLength(CHUNK_LEN);
    expect(s.root).toHaveLength(32);
    s.siblings.forEach((d) => expect(d).toHaveLength(32));
  });

  // A side with no leaves is not a proof of anything.
  test('an absent or empty side is not a proof', () => {
    expect(unpackSide(null)).toBeNull();
    expect(unpackSide(undefined)).toBeNull();
    expect(unpackSide({ leafIndices: [] })).toBeNull();
  });

  // A short copy off the device would otherwise be padded with zeros and
  // silently certify leaves that were never read.
  test('leaf data shorter than the indices claim is refused', () => {
    const s = hitAt().proofA;
    s.leafData = s.leafData.subarray(0, s.leafData.length - 1);
    expect(unpackSide(s)).toBeNull();
  });

  // A side with no siblings is legal: the tile can cover a whole small tree.
  test('an empty sibling list is allowed', () => {
    const s = hitAt().proofA;
    s.siblings = Buffer.alloc(0);
    expect(unpackSide(s).siblings).toEqual([]);
  });
});

describe('sameIndices', () => {
  test('compares by length and order', () => {
    expect(sameIndices([1, 2, 3], [1, 2, 3])).toBe(true);
    expect(sameIndices([1, 2], [1, 2, 3])).toBe(false);
    expect(sameIndices([1, 3, 2], [1, 2, 3])).toBe(false);
  });
});

describe('buildShareProof', () => {
  test('a sound hit encodes to a plain proof', () => {
    const b = buildShareProof(hitAt(), jobKey, TINY);
    expect(typeof b).toBe('string');
    expect(Buffer.from(b, 'base64').length).toBeGreaterThan(1024);
  });

  // Omitting the profile means the mainnet one, exactly as everywhere else in
  // this module -- not "no geometry", which would silently accept any tile.
  test('an omitted profile means the mainnet profile', () => {
    const h = hitAt();
    const { PROFILE } = require('../src/shared/miner/pearlhash');
    expect(buildShareProof(h, jobKey)).toBe(buildShareProof(h, jobKey, PROFILE));
  });

  test('a hit with a side missing cannot be certified', () => {
    const h = hitAt();
    delete h.proofBt;
    expect(buildShareProof(h, jobKey, TINY)).toBeNull();
    expect(buildShareProof(null, jobKey, TINY)).toBeNull();
  });

  // Tampered leaves no longer hash to the committed root.
  test('a proof that does not verify is refused', () => {
    const h = hitAt();
    h.proofA.leafData[0] ^= 0xff;
    expect(buildShareProof(h, jobKey, TINY)).toBeNull();
  });

  // The one hashing cannot catch. These leaves are real leaves of the real
  // tree and verify perfectly -- they are simply not the rows the region names,
  // which is what the device produced when its row mask went stale. The
  // verifier reads bytes [row*k, row*k+k) from the leaves it is given and
  // answers "Failed to extract strip".
  test('a proof over the wrong rows is refused even though it verifies', () => {
    const h = hitAt(0);
    const { rows } = regionToTile(0, TINY);
    const wrong = rows.map((r) => r + 16);
    const A = matrix(7, TINY.m * TINY.k);
    h.proofA = side(A, wrong, TINY.k);

    // It really does verify — that is the point.
    const unpacked = unpackSide(h.proofA);
    expect(M.verifyProof(jobKey, unpacked)).toBe(true);
    expect(buildShareProof(h, jobKey, TINY)).toBeNull();
  });

  test('the column side is checked the same way', () => {
    const h = hitAt(0);
    const { cols } = regionToTile(0, TINY);
    h.proofBt = side(matrix(9, TINY.n * TINY.k), cols.map((c) => c + 16), TINY.k);
    expect(buildShareProof(h, jobKey, TINY)).toBeNull();
  });
});
