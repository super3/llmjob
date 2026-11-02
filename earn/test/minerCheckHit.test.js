'use strict';

// shared/miner/checkHit: a hit recomputed as the pool's verifier does, which the miner runs on every hit of a
// card whose fold has not run on its kind of card (core.untested) before it sends it. The hits are genuine,
// built in JS the way the device builds them (helpers/pearlHit.js), and then each is spoiled in one place.

const { checkHit } = require('../src/shared/miner/checkHit');
const { PROFILE, buildConfig52, profileForK } = require('../src/shared/miner/pearlhash');
const { hash } = require('../src/shared/miner/blake3');
const { hitFor } = require('./helpers/pearlHit');

const SMALL = { ...PROFILE, m: 64, n: 64 };
const K4 = profileForK(SMALL, 4096);
const HEADER = Buffer.alloc(76, 0x5A);
const keyOf = (p, header = HEADER) => hash(Buffer.concat([header, buildConfig52(p)]));

// A copy deep enough to spoil one field without touching the hit the next test uses.
function copy(hit) {
  const side = (s) => ({ ...s, leafIndices: s.leafIndices.slice(), leafData: Buffer.from(s.leafData), siblings: Buffer.from(s.siblings) });
  return { ...hit, jackpotHash: Buffer.from(hit.jackpotHash), proofA: side(hit.proofA), proofBt: side(hit.proofBt) };
}

describe('checkHit', () => {
  const good = hitFor(SMALL, HEADER);
  const key = keyOf(SMALL);

  it('passes a right hit, at k 2048 and at k 4096, and at a k of a single leaf a row', () => {
    expect(checkHit(good, key, SMALL)).toBeNull();
    expect(checkHit(hitFor(K4, HEADER), keyOf(K4), K4)).toBeNull();
    const k1 = { ...SMALL, k: 1024 };
    expect(checkHit(hitFor(k1, HEADER), keyOf(k1), k1)).toBeNull();
  });

  it('fails a hit whose transcript has one word wrong, though its proofs and seeds are right', () => {
    for (const word of [0, 7, 15]) {
      const bad = hitFor(SMALL, HEADER, { tamper: (jp) => { jp[word] ^= 1; } });
      expect([word, bad.aSeed, bad.nonce]).toEqual([word, good.aSeed, good.nonce]);
      expect([word, checkHit(bad, key, SMALL)]).toEqual([word, 'jackpot hash mismatch (fold/transcript)']);
    }
    // At k 4096 too, where chunk c goes into word c % 16.
    const bad4 = hitFor(K4, HEADER, { tamper: (jp) => { jp[3] ^= 0x80000000; } });
    expect(checkHit(bad4, keyOf(K4), K4)).toBe('jackpot hash mismatch (fold/transcript)');
  });

  it('fails a hit with one byte of a row changed in its proof', () => {
    for (const side of ['proofA', 'proofBt']) {
      const bad = copy(good);
      bad[side].leafData[1000] ^= 1;
      expect([side, checkHit(bad, key, SMALL)]).toEqual([side, 'Merkle proof does not verify']);
    }
  });

  it('fails a hit with one byte of its hash changed', () => {
    for (const at of [0, 31]) {
      const bad = copy(good);
      bad.jackpotHash[at] ^= 0x10;
      expect([at, checkHit(bad, key, SMALL)]).toEqual([at, 'jackpot hash mismatch (fold/transcript)']);
    }
  });

  it('fails a hit whose seeds are not the ones its roots give', () => {
    for (const f of ['aSeed', 'bSeed']) {
      const bad = { ...good, [f]: 'ff' + good[f].slice(2) };
      expect([f, checkHit(bad, key, SMALL)]).toEqual([f, 'seed chain mismatch']);
    }
    // Under another job's key, the proofs themselves fail.
    expect(checkHit(good, keyOf(SMALL, Buffer.alloc(76, 1)), SMALL)).toBe('Merkle proof does not verify');
    // The profile's seed derivation is the one used: these seeds are the salted ones.
    expect(checkHit(good, key, { ...SMALL, seedDerivation: 'legacy' })).toBe('seed chain mismatch');
  });

  it('fails a hit whose proofs cover other rows than its region\'s', () => {
    expect(checkHit({ ...good, nonce: good.nonce + 1 }, key, SMALL)).toBe('proof leaves are not the rows of the region');
    // The proofs have the rows of k 2048; at k 4096 a row is twice as long.
    expect(checkHit(good, key, K4)).toBe('proof leaves are not the rows of the region');
  });

  it('fails a hit whose operands are not int7, though everything else about it is right', () => {
    const A = Buffer.alloc(SMALL.m * SMALL.k, 5);
    A[3 * SMALL.k + 100] = 64;
    const bad = hitFor(SMALL, HEADER, { A });
    expect(checkHit(bad, key, SMALL)).toBe('operand outside int7');
    expect(checkHit(hitFor(SMALL, HEADER, { A: Buffer.alloc(SMALL.m * SMALL.k, 0xC1) }), key, SMALL)).toBeNull();
  });

  it('fails a hit with a proof missing, and one it cannot read, without throwing', () => {
    expect(checkHit({ ...good, proofBt: undefined }, key, SMALL)).toBe('Merkle proof does not verify');
    expect(checkHit({ ...good, proofA: { ...good.proofA, leafIndices: [] } }, key, SMALL)).toBe('Merkle proof does not verify');
    expect(checkHit(null, key, SMALL)).toMatch(/^the check could not read the hit: /);
    expect(checkHit({ ...good, proofA: { ...good.proofA, leafData: null } }, key, SMALL))
      .toMatch(/^the check could not read the hit: /);
    // Something thrown that is not an Error is named as it is.
    const odd = { ...good };
    Object.defineProperty(odd, 'proofA', { get() { throw 'no proof'; } });
    expect(checkHit(odd, key, SMALL)).toBe('the check could not read the hit: no proof');
  });

  it('takes the mainnet profile when none is given', () => {
    // The same rows and leaves at mainnet's k, but its m and n bind other seeds.
    expect(checkHit(good, key)).toBe('seed chain mismatch');
  });
});
