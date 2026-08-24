'use strict';

const { hash, keyedHash } = require('../src/shared/miner/blake3');
const {
  PROFILE, SEED_SALT_A, SEED_SALT_B, bindMessage, seedDerivationCode,
} = require('../src/shared/miner/pearlhash');
const ref = require('../src/shared/miner/reference');

// The seed chain, which is where two silent bugs lived at once.
//
// The first: job_key is UNKEYED. The reference passes None as the BLAKE3 key,
// and the device was hashing it keyed with a zero key. Those are different
// functions in BLAKE3 -- keyed mode seeds the chaining value from the key and
// sets a flag -- so the device and the oracle derived different job keys while
// each stayed perfectly self-consistent.
//
// The second: cert-v3 salting. m and n are the miner's own choice and are
// deliberately NOT in config52, so under the legacy derivation nothing in the
// chain commits them. Salting the roots is what closes that.

describe('cert-v3 seed salts', () => {
  // Re-derived from the strings rather than trusted as constants. This is also
  // an outside check on our BLAKE3: the expected bytes come from the reference
  // implementation, not from us.
  test('are blake3 of their domain-separation strings', () => {
    expect(hash(Buffer.from('pearl/cert-v3/noise-seed/A', 'ascii'))).toEqual(SEED_SALT_A);
    expect(hash(Buffer.from('pearl/cert-v3/noise-seed/B', 'ascii'))).toEqual(SEED_SALT_B);
    expect(SEED_SALT_A).not.toEqual(SEED_SALT_B);
  });

  test('the bind message is root then dimension then zero padding', () => {
    const root = Buffer.alloc(32, 0xab);
    const msg = bindMessage(root, 6144);
    expect(msg).toHaveLength(64);
    expect(msg.slice(0, 32)).toEqual(root);
    expect(msg.readUInt32LE(32)).toBe(6144);
    expect(msg.slice(36).every((b) => b === 0)).toBe(true);
  });
});

describe('bindRoots', () => {
  const hashA = Buffer.alloc(32, 1);
  const hashB = Buffer.alloc(32, 2);

  test('legacy passes the raw roots through', () => {
    const out = ref.bindRoots(hashA, hashB, 6144, 6144, 'legacy');
    expect(out.hashA).toEqual(hashA);
    expect(out.hashB).toEqual(hashB);
  });

  test('salted re-hashes each root under its own salt', () => {
    const out = ref.bindRoots(hashA, hashB, 6144, 4096, 'salted');
    expect(out.hashA).toEqual(keyedHash(SEED_SALT_A, bindMessage(hashA, 6144)));
    expect(out.hashB).toEqual(keyedHash(SEED_SALT_B, bindMessage(hashB, 4096)));
    expect(out.hashA).not.toEqual(hashA);
  });

  test('salted is the default when no mode is given', () => {
    expect(ref.bindRoots(hashA, hashB, 6144, 6144, undefined))
      .toEqual(ref.bindRoots(hashA, hashB, 6144, 6144, 'salted'));
  });

  // The whole point of cert-v3: m and n reach the seeds. Under legacy they
  // cannot, because they are not in config52 either.
  test('salting commits m and n; legacy does not', () => {
    const a = ref.bindRoots(hashA, hashB, 6144, 6144, 'salted');
    const b = ref.bindRoots(hashA, hashB, 8192, 6144, 'salted');
    expect(a.hashA).not.toEqual(b.hashA);

    const la = ref.bindRoots(hashA, hashB, 6144, 6144, 'legacy');
    const lb = ref.bindRoots(hashA, hashB, 8192, 6144, 'legacy');
    expect(la.hashA).toEqual(lb.hashA);
  });

  // A and B use different salts, so a square problem cannot collapse them.
  test('the two sides cannot collide at equal dimensions', () => {
    const out = ref.bindRoots(hashA, hashA, 6144, 6144, 'salted');
    expect(out.hashA).not.toEqual(out.hashB);
  });
});

describe('deriveSeeds', () => {
  const TINY = { ...PROFILE, k: 512, rank: 32, m: 128, n: 128 };
  const A = new Int8Array(128 * 512);
  const B = new Int8Array(128 * 512);
  for (let i = 0; i < A.length; i++) A[i] = ((i * 7) % 127) - 63;
  for (let i = 0; i < B.length; i++) B[i] = ((i * 13) % 127) - 63;
  const key = Buffer.alloc(32, 5);

  test('reports the raw roots and the bound ones separately', () => {
    const s = ref.deriveSeeds(key, A, B, TINY);
    expect(s.hashA).not.toEqual(s.boundA); // salted by default
    expect(s.boundA).not.toEqual(s.boundB);
  });

  // b_seed is derived first and feeds a_seed. Not symmetric.
  test('chains b_seed into a_seed', () => {
    const s = ref.deriveSeeds(key, A, B, TINY);
    expect(s.bSeed).toEqual(hash(Buffer.concat([key, s.boundB])));
    expect(s.aSeed).toEqual(hash(Buffer.concat([s.bSeed, s.boundA])));
    expect(s.aSeed).not.toEqual(s.bSeed);
  });

  // The bug the device had: a zero KEY is not the same as no key.
  test('the seed hashes are unkeyed, not keyed with zeros', () => {
    const s = ref.deriveSeeds(key, A, B, TINY);
    const zeroKeyed = keyedHash(Buffer.alloc(32), Buffer.concat([key, s.boundB]));
    expect(s.bSeed).not.toEqual(zeroKeyed);
  });

  test('the derivation mode changes the seeds', () => {
    const salted = ref.deriveSeeds(key, A, B, TINY);
    const legacy = ref.deriveSeeds(key, A, B, { ...TINY, seedDerivation: 'legacy' });
    expect(salted.aSeed).not.toEqual(legacy.aSeed);
    expect(salted.hashA).toEqual(legacy.hashA); // the raw root is the same
  });

  test('an absent profile still derives, in legacy-free salted form', () => {
    const s = ref.deriveSeeds(key, A, B, undefined);
    expect(s.aSeed).toHaveLength(32);
    expect(s.bSeed).toHaveLength(32);
  });
});

describe('the mainnet profile', () => {
  test('defaults to the cert-v3 derivation', () => {
    expect(PROFILE.seedDerivation).toBe('salted');
  });

  // The addon takes the derivation as a NUMBER. It used to be handed the
  // string through Uint32Value(), which renders any non-numeric text as 0 --
  // and 0 is salted, so asking for legacy would have been silently ignored and
  // the wrong derivation mined with no symptom but rejected shares.
  test('the numeric code the addon reads agrees with the string', () => {
    expect(seedDerivationCode()).toBe(0);
    expect(seedDerivationCode({ ...PROFILE, seedDerivation: 'legacy' })).toBe(1);
    expect(PROFILE.seedDerivationCode).toBe(seedDerivationCode(PROFILE));
  });

  // m and n are bound into the seeds by cert-v3, so the JS profile and the core
  // must agree on them or every seed differs.
  test('carries the measured workload dimensions and batch width', () => {
    expect(PROFILE.m).toBe(12288);
    expect(PROFILE.n).toBe(12288);
    expect(PROFILE.colBatch).toBeGreaterThan(0);
  });

  // rank must be a power of two AND a multiple of the BLAKE3 digest size: the
  // dense noise factor is generated one 32-byte digest at a time.
  test('carries a rank the noise generator can actually produce', () => {
    expect(PROFILE.rank % 32).toBe(0);
    expect(PROFILE.rank & (PROFILE.rank - 1)).toBe(0);
  });
});
