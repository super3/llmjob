'use strict';

// The Quantus CPU reference against the project's own published vectors. Every
// expected value below is copied from Quantus's code, not computed here:
//
//   - the permutation of the zero state and of 1..12: qp-poseidon 3.1.0,
//     src/poseidon2.rs (P3_ZERO_RESULT, P3_SEQ_RESULT), which it took from
//     Plonky3's Goldilocks;
//   - hash_bytes of 18 inputs: qp-poseidon src/lib.rs, test_known_value_hashes;
//   - the nonce hash and midstate of 5 header/nonce pairs: quantus-miner
//     crates/pow-core/src/lib.rs, NONCE_HASH_KVS (the same five are in
//     not-only-mining-pool's engine/quantus/testdata/nonce_hash.json).

const p2 = require('../src/shared/quantus/poseidon2');
const q = require('../src/shared/quantus/qpow');

const hex = (s) => Buffer.from(s, 'hex');
const u64s = (list) => list.map((h) => BigInt('0x' + h));

describe('Poseidon2 permutation', () => {
  test('the zero state', () => {
    const s = new Array(12).fill(0n);
    p2.permute(s);
    expect(s).toEqual(u64s([
      'c9bc9432e1686884', '03ecbab0dcdd2189', '5e7ac885b3dc1215', '6ac07513801d191f', 'ca5c593fb184dcfc', '414dec5f3e455287',
      '1a17df170127ae41', 'e7e592bd0af9b0a5', 'c71a9b27edc66a4c', '2728671759ac43c2', 'b9969c20f7f672f9', 'c5140b586823b92f',
    ]));
  });

  test('the state 1..12', () => {
    const s = Array.from({ length: 12 }, (_, i) => BigInt(i + 1));
    p2.permute(s);
    expect(s).toEqual(u64s([
      '7e9574e2a3d6c48b', '9d7bc16d282d2f2b', '798826626d94a498', '0831011bb22304c7', 'bdccb5931fffd16c', 'e98687714dacbefc',
      'c6a1ed29dd75e027', '1aec96681d15f765', 'c74b2c710b170a23', '5fb4aff45e9c24fb', '1fb3d228db0127eb', 'e201a7e214b16e74',
    ]));
  });

  test('the layers and the S-box on their own, as qp-poseidon tests them', () => {
    const s = Array.from({ length: 12 }, (_, i) => BigInt(i + 1));
    p2.externalLinear(s);
    expect(s).toEqual([144n, 156n, 168n, 148n, 172n, 184n, 196n, 176n, 200n, 212n, 224n, 204n]);
    const t = Array.from({ length: 12 }, (_, i) => BigInt(i + 1));
    p2.internalLinear(t);
    expect(t).toEqual([
      14102670999874605902n, 12724564314584031161n, 2820561051427350843n, 16542800896675938601n,
      15306463738286039492n, 7752500014678011478n, 6061018803481092188n, 7074894295206835361n,
      12845226504185330854n, 9784840285965170449n, 9819782087245364719n, 15989125545469266117n,
    ]);
    expect([0n, 1n, 2n, 5n, 0xffffffffn].map(p2.exp7)).toEqual([0n, 1n, 128n, 78125n, 0xffffffffn]);
  });
});

describe('hash_bytes', () => {
  const zeros = (n) => Buffer.alloc(n);
  const vectors = [
    [zeros(0), '4d8d22af81f6c27a005a07028590ef4ee480f6c4b93f813daf9de47a07c8ae86'],
    [zeros(1), '8f5b42e350ff5a12788210c86c2bcd49243b8f9350de818b3b0c56839a42ebad'],
    [zeros(2), '3e6ee24fb61a22f4d825b72fc8ebd359e3b3b9566e246c71c3e450ebe3262f9c'],
    [zeros(3), '34f4338a6f1b671062a3ac00b37ca05a47b43e16e589ccaa5b063416ba42356b'],
    [zeros(4), '7bac8c6bc49b0b750f2ce0912b815a2cb4ae20c75ac430850257882d9d321afa'],
    [zeros(5), 'c95cfddb573adf4070b3d7c8d2dfbbee48b4b973d80cbda2b458abe7bb6f0def'],
    [zeros(6), 'a4dc08d0a8c5ea44007462fe1fd8e45962d4ea85c420eab4140fbb30b5b5e111'],
    [zeros(7), 'b01975012df91d9f9f040c34655f23f3ec1f6d1738d85679e9848143756637c9'],
    [zeros(8), 'eacd9e48d2e968131e48c8e69f2a211cc06c7778db6c5467348b45418fc7f585'],
    [zeros(9), '00df670e8ec0751d3fb9b5f0281d0af9a7a82f62ad35a21247a9d6117daec151'],
    [zeros(14), 'd6182896f274c5d9640972e2bf2a5e893e516a21adfdd8ebd39969128d619934'],
    [zeros(15), '15fc2f3c3bc51c96797b889d4fecfcd3535b959f510c007598a87f099e356303'],
    [Buffer.from([0, 0, 0, 1]), '6ffff0c97262139567c426e916c1fd70c924010153c366bb2a8957ea89902942'],
    [Buffer.from([1, 2, 3, 4, 5, 6, 7, 8]), '131020b2e74819343f8568258ae2e9717e9b2253d57baabab78a518bc7499a8b'],
    [Buffer.alloc(32, 255), '41260a4322e97dc3dda2b5f70b5ffb1b43071ad5510e101f34209721042c0987'],
    [Buffer.from('hello world'), 'fd1f5d7d4701c25bbdd5dd6e3be6abb474fffbaa402f814dce95f8283abbf3e7'],
    [Buffer.from(Array.from({ length: 32 }, (_, i) => i)), '36884f9093be80632397f5736dce2fece627a4182daf3cdbf8bf12c8e3e02668'],
    [Buffer.from(Array.from({ length: 64 }, (_, i) => i)), 'dd0d06fbe4e7575d0eeac53706482cbbe592e269a35bcd5591a495814371724e'],
  ];
  test.each(vectors.map(([input, want]) => [input.toString('hex') || '(empty)', input, want]))('%s', (_, input, want) => {
    expect(p2.hashBytes(input).toString('hex')).toBe(want);
  });

  test('hash_squeeze_twice starts with hash_bytes, as qp-poseidon tests it', () => {
    const input = Buffer.from('test 512-bit hash');
    const h = p2.hashSqueezeTwice(input);
    expect(h).toHaveLength(64);
    expect(h.subarray(0, 32)).toEqual(p2.hashBytes(input));
  });

  test('the byte encoding: 4 bytes an element, then the 0x01 terminator', () => {
    expect(p2.bytesToFelts(Buffer.from('hello'))).toEqual([0x6c6c6568n, 0x16fn]);
    expect(p2.bytesToFelts(Buffer.alloc(4))).toEqual([0n, 1n]);
  });
});

// quantus-miner crates/pow-core/src/lib.rs, NONCE_HASH_KVS.
const KVS = [
  {
    header: '0000000000000000000000000000000000000000000000000000000000000000',
    nonce: '00'.repeat(64),
    hash: '8e64e3d8e0f38f882e8501f9e525df0a95d2e91e9cfc32c9248d756fb07780e2f8fdca2c5a54441e6fcd8d774a5f6aae72f36d1c76bc19f691a0d4f6c607e8cc',
    mid: ['8646d336b5a0fccd', '818ca59548916345', '49475e8bc9c928bc', 'c469163bcdc4900b', 'e4421fafacd59087', '633f49b998698d66',
      '4a9067e3883e76da', '193039e85ea14472', '203d2d3ee6a8eba8', 'ef5b9ce39bcff072', 'c16a3c8dd5680003', 'd77c52ab5bf7cf1f'],
  },
  {
    header: '0101010101010101010101010101010101010101010101010101010101010101',
    nonce: '00'.repeat(63) + '01',
    hash: 'e4ef79db80b642a093d7e38e6a6daac7ec1cca7293ddbe3710b4e6781be4add0b93f7e3440626df5bb23b757436787ba8d3fd0a2652e387f8af5bd8a5cc2ce2f',
    mid: ['788cf743676a5c85', 'dce1d727e1189b53', '36e0e019f5a0bf3a', '806487dd9c3bf83f', 'cb6d7300de1ab5fe', '7dbe191fa64fc6b7',
      '95dedc7fe99861be', 'b1a3885303f7467d', '891eb002a34a0f17', '4be4cf057a87208a', 'e9113e7dc541745f', 'faeda908097d00f1'],
  },
  {
    header: '000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f',
    nonce: '00'.repeat(56) + '1234567890abcdef',
    hash: '7f6c410ce62e7fc54811ad3b8b92664ac0f8b1d2cd6eb163129251146e7ee34d27794aaae1b31eaaeb98ab835bae6a2b9ac4ad6f08c95903e1423f0b865816ff',
    mid: ['f6fc848ab2dddc18', 'fd88d1f0ccda99fe', '27cfa43b8ab9947a', '6a505b37f02183d3', '5f98f1d3524fb0c6', 'd6de4efd98793e2e',
      'f3f205bd1e4d4526', 'c0c9046201e91dd6', 'cd2d28eef32b18de', 'a303ecbc8e98000c', '3db245e7738f3aca', 'bc55f7553023863e'],
  },
  {
    header: 'ff'.repeat(32),
    nonce: '00'.repeat(56) + 'ffffffffffffffff',
    hash: '822f70999ee517f610b51bbb79ba88bfbb19cb77b316df25d8aeb4c2682673555e9fc6a588ee96dcce4ff0d2b8d97f4aa89d0f46c97f9027b386d42b14be5100',
    mid: ['d1fe93daa1f07409', '0ea440f2b57a0ecf', '3b7a1f3cb2d48e39', '43a5a1255658d1b2', 'ebe00ad8f0fe28c0', 'b5618e7c6d6072d1',
      '158b7b19f80ec62c', '329a91c8de5575c2', 'd1637d851a474212', '2d927735708f3b7d', '6219c7cf8ef5af51', '802028c39ccd8f13'],
  },
  {
    header: '07'.repeat(32),
    nonce: '00000000000000000000000000000000000000000deadbeefcafe000000000000000000000000000000000000000000000000000000000001234567890abcdef',
    hash: 'eab81cdb33db293aef7e1f10f85cb08d17bc197c284d46a25d9dc57cdb7ba878adf3f5c354318121876795aab08c0941afda21d835383faa2f139c26be31a30a',
    mid: ['41885398f5b05a0d', 'b7f17cee3f70d242', '307aecc5bca54c7e', 'b74164c0d6de7e91', '0c0a9b27281c8181', 'd84ddf12fdddf0ee',
      '88f8cdf552c0a15e', '88da4cfbfbf92d22', '8c344261349149db', '0c0fd0d6a8444652', '7a57b00736d58568', '9ad40597f4fe2abc'],
  },
];

describe('the nonce hash (qpow-math get_nonce_hash)', () => {
  test.each(KVS.map((v, i) => [i, v]))('vector %i: hash, midstate, and the midstate path', (_, v) => {
    const header = hex(v.header);
    const nonce = hex(v.nonce);
    expect(q.nonceHash(header, nonce).toString('hex')).toBe(v.hash);

    const { high, tail, counter } = q.splitNonce(nonce);
    const mid = q.midstate(header, high);
    expect(mid).toEqual(u64s(v.mid));
    // The midstate as cores take it, and back.
    expect(q.midstateFromBytes(q.midstateBytes(mid))).toEqual(mid);
    // What a core computes per nonce, against the full hash.
    expect(q.hashFromMidstate(mid, tail, counter).toString('hex')).toBe(v.hash);
    expect(q.nonceTop64(mid, tail, counter)).toBe(BigInt('0x' + v.hash.slice(0, 16)));
  });

  test('the target check is strict, as qpow-math is_valid_nonce', () => {
    const v = KVS[2];
    const hash = hex(v.hash);
    const h = q.toBig(hash);
    expect(q.meetsTarget(hash, h)).toBe(false);
    expect(q.meetsTarget(hash, h + 1n)).toBe(true);
    // Difficulty 1 takes everything below the maximum.
    expect(q.targetFromDifficulty(1)).toBe(q.MAX_TARGET);
    expect(q.meetsTarget(hash, q.targetFromDifficulty(1n))).toBe(true);
    expect(() => q.targetFromDifficulty(0)).toThrow('difficulty must be at least 1');
  });

  test('a difficulty and its target: LuckyPool\'s difficulty 1e10 job carries this target', () => {
    // engine/quantus/testdata/luckypool_login.json (not-only-mining-pool), a
    // recording of LuckyPool's job.
    const target = q.targetFromDifficulty(10000000000n);
    expect(target.toString(16).padStart(128, '0')).toBe(
      '000000006df37f675ef6eadf5ab9a2072d44268d97df837e6748956e5c6c2117501e68855669e4b8356cf292464d9e16cc8d4655f8fb96f429b149fc87d74da4');
    expect(q.targetHigh(target)).toBe(0x6df37f67n);
    expect(q.difficultyFromTarget(target)).toBe(10000000000n);
    expect(q.difficultyFromTarget(0n)).toBe(q.MAX_TARGET);
  });

  test('the nonce layout: extranonce first, lane, then the counter', () => {
    const lane = hex('a1a2a3a4a5a60003');
    const nonce = q.buildNonce(hex('0004ee45'), lane, 0x0102030405060708n);
    expect(nonce.toString('hex')).toBe('0004ee45' + '00'.repeat(44) + 'a1a2a3a4a5a60003' + '0102030405060708');
    const parts = q.splitNonce(nonce);
    expect(parts.high.toString('hex')).toBe('0004ee45' + '00'.repeat(28));
    expect(parts.tail.toString('hex')).toBe('00'.repeat(16) + 'a1a2a3a4a5a60003');
    expect(parts.counter).toBe(0x0102030405060708n);
    expect(q.buildNonce(null, lane, 5).subarray(0, 48)).toEqual(Buffer.alloc(48));
    expect(() => q.buildNonce(Buffer.alloc(33), lane, 0)).toThrow('does not fit');
    expect(() => q.buildNonce(Buffer.alloc(4), Buffer.alloc(7), 0)).toThrow('lane must be 8 bytes');
  });

  test('wrong sizes and a midstate that is not field elements are refused', () => {
    expect(() => q.nonceHash(Buffer.alloc(31), Buffer.alloc(64))).toThrow('header must be 32 bytes');
    expect(() => q.nonceHash(Buffer.alloc(32), 'x')).toThrow('nonce must be 64 bytes');
    expect(() => q.midstate(Buffer.alloc(32), Buffer.alloc(31))).toThrow('nonce high half must be 32 bytes');
    expect(() => q.midstateFromBytes(Buffer.alloc(95))).toThrow('midstate must be 96 bytes');
    expect(() => q.midstateFromBytes(Buffer.alloc(96, 0xff))).toThrow('midstate element 0 is not below the field prime');
  });
});
