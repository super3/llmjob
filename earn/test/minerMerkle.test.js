'use strict';

const M = require('../src/shared/miner/merkle');
const { keyedHash } = require('../src/shared/miner/blake3');

// The operand commitment as a TREE, which is what a share has to be proved
// against. Sending the 64-byte transcript instead is what the pool meant by
// "not a valid PlainProof".

const KEY = Buffer.alloc(32, 7);

function data(chunks) {
  const b = Buffer.alloc(chunks * 1024);
  for (let i = 0; i < b.length; i++) b[i] = (i * 31 + chunks) & 0xff;
  return b;
}

describe('the commitment tree', () => {
  // The anchor for everything else here. At a power-of-two chunk count this
  // tree IS standard BLAKE3's, so its root must equal a plain keyed hash of the
  // whole operand -- which ties it to an implementation already checked against
  // the official published vectors.
  test('its root is exactly keyed BLAKE3 of the operand', () => {
    for (const chunks of [2, 4, 8, 16, 64]) {
      const d = data(chunks);
      expect(Buffer.from(M.rootOf(M.buildLayers(KEY, d)))).toEqual(keyedHash(KEY, d));
    }
  });

  test('a single chunk or less is hashed directly', () => {
    const small = Buffer.alloc(500, 9);
    expect(Buffer.from(M.rootOf(M.buildLayers(KEY, small)))).toEqual(keyedHash(KEY, small));
    const exact = Buffer.alloc(1024, 3);
    expect(Buffer.from(M.rootOf(M.buildLayers(KEY, exact)))).toEqual(keyedHash(KEY, exact));
  });

  test('layer counts follow the halving', () => {
    expect(M.buildLayers(KEY, data(2))).toHaveLength(2);
    expect(M.buildLayers(KEY, data(8))).toHaveLength(4);
    expect(M.buildLayers(KEY, data(64))).toHaveLength(7);
  });

  test('the key matters', () => {
    const d = data(8);
    expect(Buffer.from(M.rootOf(M.buildLayers(KEY, d))))
      .not.toEqual(Buffer.from(M.rootOf(M.buildLayers(Buffer.alloc(32, 8), d))));
  });

  // An odd node has no partner and is promoted unchanged rather than merged
  // with itself, which would be a different tree.
  test('an odd tail is promoted, not self-merged', () => {
    const d = data(6);
    const layers = M.buildLayers(KEY, d);
    expect(layers[0]).toHaveLength(6);
    expect(layers[1]).toHaveLength(3);
    expect(Buffer.from(layers[2][1])).toEqual(Buffer.from(layers[1][2]));
  });
});

describe('chunk and parent CVs', () => {
  test('a chunk CV depends on its index', () => {
    const chunk = Buffer.alloc(1024, 5);
    expect(Buffer.from(M.chunkCV(KEY, chunk, 0)))
      .not.toEqual(Buffer.from(M.chunkCV(KEY, chunk, 1)));
  });

  // The final merge of a tree carries ROOT and its output is the digest; every
  // other merge does not. Confusing the two gives a wrong root with no symptom.
  test('the root merge differs from an interior merge', () => {
    const a = Buffer.alloc(32, 1);
    const b = Buffer.alloc(32, 2);
    expect(Buffer.from(M.parentCV(KEY, a, b, true)))
      .not.toEqual(Buffer.from(M.parentCV(KEY, a, b, false)));
  });
});

describe('row to leaf mapping', () => {
  // A k=2048 row spans exactly two 1024-byte chunks.
  test('maps each row to the chunks that hold it', () => {
    expect(M.leafIndicesFromRows([0, 8, 64, 72], 2048)).toEqual([0, 1, 16, 17, 128, 129, 144, 145]);
  });

  test('a row narrower than a chunk can share one', () => {
    expect(M.leafIndicesFromRows([0, 1], 512)).toEqual([0]);
    expect(M.leafIndicesFromRows([0, 3], 512)).toEqual([0, 1]);
  });

  test('the result is sorted and free of duplicates', () => {
    const out = M.leafIndicesFromRows([72, 0, 8, 0], 2048);
    expect(out).toEqual([...out].sort((a, b) => a - b));
    expect(new Set(out).size).toBe(out.length);
  });
});

describe('multi-leaf proofs', () => {
  const d = data(64);
  const layers = M.buildLayers(KEY, d);

  test('every shape of request round-trips', () => {
    const cases = [[0], [1], [0, 1], [5], [0, 63], [3, 4, 5], [2, 3, 10, 11, 40, 41],
      [0, 1, 2, 3, 4, 5, 6, 7], [0, 16, 17, 128 % 64, 63]];
    for (const idxs of cases) {
      const p = M.multiLeafProof(KEY, d, layers, idxs);
      expect(M.verifyProof(KEY, p)).toBe(true);
    }
  });

  test('it carries the requested chunks, sorted and deduplicated', () => {
    const p = M.multiLeafProof(KEY, d, layers, [5, 2, 5]);
    expect(p.leafIndices).toEqual([2, 5]);
    expect(p.leafData).toHaveLength(2);
    expect(p.leafData[0]).toHaveLength(1024);
    expect(p.totalLeaves).toBe(64);
    expect(Buffer.from(p.root)).toEqual(keyedHash(KEY, d));
  });

  // Adjacent leaves authenticate each other, so asking for both costs fewer
  // siblings than asking for either alone.
  test('adjacent leaves need fewer siblings than separated ones', () => {
    const together = M.multiLeafProof(KEY, d, layers, [4, 5]).siblings.length;
    const apart = M.multiLeafProof(KEY, d, layers, [4, 40]).siblings.length;
    expect(together).toBeLessThan(apart);
  });

  test('tampering with a leaf is caught', () => {
    const p = M.multiLeafProof(KEY, d, layers, [5]);
    p.leafData[0][0] ^= 1;
    expect(M.verifyProof(KEY, p)).toBe(false);
  });

  test('tampering with a sibling is caught', () => {
    const p = M.multiLeafProof(KEY, d, layers, [5]);
    p.siblings[0] = Buffer.alloc(32, 0xff);
    expect(M.verifyProof(KEY, p)).toBe(false);
  });

  test('a proof under the wrong key does not verify', () => {
    const p = M.multiLeafProof(KEY, d, layers, [5]);
    expect(M.verifyProof(Buffer.alloc(32, 8), p)).toBe(false);
  });

  test('an empty or out-of-range request is refused', () => {
    expect(() => M.multiLeafProof(KEY, d, layers, [])).toThrow(/non-empty/);
    expect(() => M.multiLeafProof(KEY, d, layers, [64])).toThrow(/out of bounds/);
  });
});

describe('defensive paths', () => {
  test('a chunk may be a typed array rather than a Buffer', () => {
    const chunk = new Uint8Array(1024).fill(11);
    expect(Buffer.from(M.chunkCV(KEY, chunk, 0)))
      .toEqual(Buffer.from(M.chunkCV(KEY, Buffer.from(chunk), 0)));
  });

  test('a key that is not 32 bytes is refused', () => {
    expect(() => M.chunkCV(Buffer.alloc(31), Buffer.alloc(1024), 0))
      .toThrow(/32 bytes/);
  });

  // Callers pass Uint8Array as readily as Buffer; neither should be special.
  test('accepts plain typed arrays as well as Buffers', () => {
    const key = new Uint8Array(32).fill(7);
    const d = new Uint8Array(2048).fill(3);
    const layers = M.buildLayers(key, d);
    const p = M.multiLeafProof(key, d, layers, [0]);
    expect(M.verifyProof(key, p)).toBe(true);
  });

  // An odd tail is promoted rather than merged, so the verifier has to promote
  // it too — with no sibling consumed. Getting that wrong desynchronises the
  // sibling stream and every later level is garbage.
  test('a tree with an odd tail verifies', () => {
    const key = Buffer.alloc(32, 4);
    for (const chunks of [3, 5, 6, 7]) {
      const d = Buffer.alloc(chunks * 1024);
      for (let i = 0; i < d.length; i++) d[i] = (i * 13 + chunks) & 0xff;
      const layers = M.buildLayers(key, d);
      for (const idxs of [[0], [chunks - 1], [0, chunks - 1]]) {
        expect(M.verifyProof(key, M.multiLeafProof(key, d, layers, idxs))).toBe(true);
      }
    }
  });
});
