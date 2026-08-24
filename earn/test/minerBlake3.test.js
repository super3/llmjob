'use strict';

const { hash, keyedHash, OUT_LEN, CHUNK_LEN, BLOCK_LEN } = require('../src/shared/miner/blake3');

// The OFFICIAL BLAKE3 test vectors, from the reference repository's
// test_vectors.json. This is the only part of the miner that can be checked
// against evidence outside our own codebase, which makes it the anchor for
// everything else: every hash in the Pearl PoW is BLAKE3, so if this is right
// the reference implementation has a trustworthy foundation, and if it is wrong
// nothing downstream can be believed.
//
// Input pattern is the one the vectors specify: byte i = i mod 251.
function input(n) {
  return Buffer.from(Array.from({ length: n }, (_, i) => i % 251));
}

const KEY = Buffer.from('whats the Elvish word for friend');

describe('BLAKE3 — official unkeyed vectors', () => {
  // The lengths are chosen by the spec to straddle every structural boundary:
  // sub-block, exact block, exact chunk, and the multi-chunk tree.
  const CASES = [
    [0, 'af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262'],
    [1, '2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213'],
    [2, '7b7015bb92cf0b318037702a6cdd81dee41224f734684c2c122cd6359cb1ee63'],
    [3, 'e1be4d7a8ab5560aa4199eea339849ba8e293d55ca0a81006726d184519e647f'],
    [1023, '10108970eeda3eb932baac1428c7a2163b0e924c9a9e25b35bba72b28f70bd11'],
    [1024, '42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7'],
    [1025, 'd00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444'],
    [2048, 'e776b6028c7cd22a4d0ba182a8bf62205d2ef576467e838ed6f2529b85fba24a'],
    [2049, '5f4d72f40d7a5f82b15ca2b2e44b1de3c2ef86c426c95c1af0b6879522563030'],
  ];

  for (const [len, expected] of CASES) {
    test('hash of ' + len + ' bytes', () => {
      expect(hash(input(len)).toString('hex')).toBe(expected);
    });
  }

  // 1025 and 2049 are the ones that matter structurally: they force a parent
  // node and then an unbalanced tree. A chunk-only implementation passes
  // everything up to 1024 and fails from there, so these are the real test.
  test('the multi-chunk cases exercise the Merkle tree, not just chunks', () => {
    expect(1025).toBeGreaterThan(CHUNK_LEN);
    expect(hash(input(1025))).not.toEqual(hash(input(1024)));
  });
});

describe('BLAKE3 — official keyed vectors', () => {
  const CASES = [
    [0, '92b2b75604ed3c761f9d6f62392c8a9227ad0ea3f09573e783f1498a4ed60d26'],
    [1024, '75c46f6f3d9eb4f55ecaaee480db732e6c2105546f1e675003687c31719c7ba4'],
  ];

  for (const [len, expected] of CASES) {
    test('keyed hash of ' + len + ' bytes', () => {
      expect(keyedHash(KEY, input(len)).toString('hex')).toBe(expected);
    });
  }

  // Keying must actually change the output — a keyed implementation that
  // silently ignores its key would still pass the unkeyed vectors above.
  test('the key changes the digest', () => {
    expect(keyedHash(KEY, input(64))).not.toEqual(hash(input(64)));
  });

  test('a key of the wrong length is refused rather than truncated', () => {
    expect(() => keyedHash(Buffer.alloc(31), input(0))).toThrow(/32 bytes/);
    expect(() => keyedHash(Buffer.alloc(33), input(0))).toThrow(/32 bytes/);
  });
});

describe('BLAKE3 — shape', () => {
  test('defaults to 32 bytes and honours extendable output', () => {
    expect(hash(input(10))).toHaveLength(OUT_LEN);
    expect(hash(input(10), 64)).toHaveLength(64);
    // Extendable output must EXTEND the 32-byte digest, not restart it.
    expect(hash(input(10), 64).slice(0, 32)).toEqual(hash(input(10)));
  });

  test('accepts a plain array as the key too, not just a Buffer', () => {
    const arr = Array.from({ length: 32 }, (_, i) => i);
    expect(keyedHash(arr, input(8))).toEqual(keyedHash(Buffer.from(arr), input(8)));
  });

  test('accepts a plain array as input', () => {
    expect(hash([0, 1, 2])).toEqual(hash(Buffer.from([0, 1, 2])));
  });

  test('block and chunk lengths are the spec values', () => {
    expect(BLOCK_LEN).toBe(64);
    expect(CHUNK_LEN).toBe(1024);
  });

  test('incremental update matches a single-shot hash', () => {
    const { Hasher, IV } = require('../src/shared/miner/blake3');
    const h = new Hasher(IV, 0);
    h.update(input(500));
    h.update(input(600).slice(0, 600));
    const all = Buffer.concat([input(500), input(600).slice(0, 600)]);
    expect(h.digest()).toEqual(hash(all));
  });
});
