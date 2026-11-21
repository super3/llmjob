'use strict';

const f = require('../src/shared/miner/fp8');
const { hash, keyedHash } = require('../src/shared/miner/blake3');
const b200Vectors = require('./fixtures/fp8B200Vectors.json');
const rtx5080 = require('./fixtures/fp8Rtx5080Testnet2.json');

// The PIP-3 (FP8) reference, checked against numbers Pearl pinned in its own
// tests on the fp8-scheme branch (zk-pow/src/v4, commit 1dd75a52). Each block
// names the Rust test its vectors come from. Those vectors were produced by
// Pearl's Python simulator of the B200, by torch, or by the Python blake3
// package, so a pass here is agreement with code outside this repo.
//
// What is NOT pinned by Pearl, and so is only self-consistent here: a whole
// tile from seeds to ticket. Pearl has no fixture for one. The end-to-end test
// at the bottom guards against regressions, not against a wrong port.

const hex = (s) => Buffer.from(s, 'hex');
const B200 = f.DEVICE_B200;
const H100 = f.DEVICE_H100;

// A small deterministic generator for sweeps.
function lcg(seed) {
  let x = seed >>> 0;
  return () => {
    x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
    return x;
  };
}

describe('the flag', () => {
  test('is off unless PEARL_FP8=1', () => {
    expect(f.enabled({})).toBe(false);
    expect(f.enabled({ PEARL_FP8: '0' })).toBe(false);
    expect(f.enabled({ PEARL_FP8: '1' })).toBe(true);
  });
});

describe('E4M3 (dtype.rs)', () => {
  // fp8_e4m3_exhaustive, f32_to_fp8_e4m3_round_trips
  test('every code decodes to the format definition and encodes back', () => {
    for (let bits = 0; bits < 256; bits++) {
      if ((bits & 0x7F) === 0x7F) continue;
      const exp = (bits >> 3) & 15;
      const man = bits & 7;
      const want = (bits & 0x80 ? -1 : 1)
        * (exp === 0 ? (man / 8) * 2 ** -6 : (1 + man / 8) * 2 ** (exp - 7));
      expect(Object.is(f.e4m3ToF32(bits), want)).toBe(true);
      expect(f.f32ToE4m3(want)).toBe(bits);
    }
    expect(f.e4m3ToF32(0x7E)).toBe(448);
    expect(f.e4m3ToF32(0x01)).toBe(2 ** -9);
  });

  test('the NaN encodings are refused', () => {
    expect(() => f.e4m3ToF32(0x7F)).toThrow('NaN encoding');
    expect(() => f.e4m3ToF32(0xFF)).toThrow('NaN encoding');
  });

  // f32_to_fp8_e4m3_rounds_and_saturates
  test('rounds to nearest even and saturates at 448', () => {
    expect(f.f32ToE4m3(1)).toBe(0x38);
    expect(f.f32ToE4m3(-1)).toBe(0xB8);
    expect(f.f32ToE4m3(1.0625)).toBe(0x38);
    expect(f.f32ToE4m3(Math.fround(1.0626))).toBe(0x39);
    expect(f.f32ToE4m3(1.1875)).toBe(0x3A); // a tie on an odd mantissa rounds up
    expect(f.f32ToE4m3(1.96875)).toBe(0x40); // the mantissa carries into the exponent
    expect(f.f32ToE4m3(2 ** -10)).toBe(0x00);
    expect(f.f32ToE4m3(2 ** -9)).toBe(0x01);
    expect(f.f32ToE4m3(1.5 * 2 ** -9)).toBe(0x02);
    expect(f.f32ToE4m3(2.5 * 2 ** -9)).toBe(0x02);
    expect(f.f32ToE4m3(1e30)).toBe(0x7E);
    expect(f.f32ToE4m3(-1e30)).toBe(0xFE);
    expect(f.f32ToE4m3(500)).toBe(0x7E);
    expect(f.f32ToE4m3(464)).toBe(0x7E); // exp 15 with a full mantissa is the NaN slot
    expect(f.f32ToE4m3(0)).toBe(0x00);
    expect(f.f32ToE4m3(-0)).toBe(0x80);
    expect(() => f.f32ToE4m3(NaN)).toThrow('non-finite');
    expect(() => f.f32ToE4m3(Infinity)).toThrow('non-finite');
  });
});

describe('BF16 (dtype.rs)', () => {
  // bf16_round_trip, f32_to_bf16_rejects_non_finite
  test('is the top half of an f32', () => {
    for (let bits = 0; bits < 65536; bits += 7) {
      if ((bits & 0x7FFF) >= 0x7F80) continue;
      expect(f.f32Bits(f.bf16ToF32(bits))).toBe((bits << 16) >>> 0);
      expect(f.f32ToBf16(f.bf16ToF32(bits))).toBe(bits);
    }
    expect(f.f32ToBf16(1)).toBe(0x3F80);
    expect(f.f32ToBf16(448)).toBe(0x43E0);
    expect(f.f32ToBf16(f.bf16ToF32(0x7F7F))).toBe(0x7F7F);
  });

  test('rounds ties to even', () => {
    expect(f.f32ToBf16(f.f32FromBits(0x3F808000))).toBe(0x3F80);
    expect(f.f32ToBf16(f.f32FromBits(0x3F818000))).toBe(0x3F82);
    expect(f.f32ToBf16(f.f32FromBits(0x3F808001))).toBe(0x3F81);
  });

  test('refuses non-finite values and overflow', () => {
    expect(() => f.f32ToBf16(NaN)).toThrow('non-finite');
    expect(() => f.f32ToBf16(-Infinity)).toThrow('non-finite');
    expect(() => f.f32ToBf16(f.f32FromBits(0x7F7FFFFF))).toThrow('overflows bf16');
    expect(() => f.bf16ToF32(0x7FC0)).toThrow('non-finite encoding');
    expect(() => f.bf16ToF32(0xFF80)).toThrow('non-finite encoding');
  });

  test('f32Step moves one float in either direction', () => {
    expect(f.f32Bits(f.f32Step(0, true))).toBe(1);
    expect(f.f32Bits(f.f32Step(0, false))).toBe(0x80000001);
    expect(f.f32Bits(f.f32Step(1, true))).toBe(0x3F800001);
    expect(f.f32Bits(f.f32Step(1, false))).toBe(0x3F7FFFFF);
    expect(f.f32Bits(f.f32Step(-1, true))).toBe(0xBF7FFFFF);
    expect(f.f32Bits(f.f32Step(-1, false))).toBe(0xBF800001);
  });
});

// a*b + c rounded once to `keep` significant bits, in exact integers.
function exactFma(a, b, c, keep) {
  const parts = (x) => {
    const bits = f.f32Bits(x);
    const e = (bits >>> 23) & 0xFF;
    const sig = BigInt(e ? (bits & 0x7FFFFF) | 0x800000 : bits & 0x7FFFFF);
    return [bits >>> 31 ? -sig : sig, (e || 1) - 150];
  };
  const [sa, ea] = parts(a);
  const [sb, eb] = parts(b);
  const [sc, ec] = parts(c);
  const e = Math.min(ea + eb, ec);
  let v = ((sa * sb) << BigInt(ea + eb - e)) + (sc << BigInt(ec - e));
  const neg = v < 0n;
  if (neg) v = -v;
  const drop = Math.max(v.toString(2).length - keep, 0);
  if (drop > 0) {
    const q = v >> BigInt(drop);
    const r = v - (q << BigInt(drop));
    const half = 1n << BigInt(drop - 1);
    v = q + (r > half || (r === half && (q & 1n) === 1n) ? 1n : 0n);
  }
  return (neg ? -1 : 1) * Number(v) * 2 ** (e + drop);
}

describe('element-wise arithmetic (compute.rs)', () => {
  // bf16_fma_matches_reference_vectors
  test('bf16Fma matches the torch vectors', () => {
    const cases = [
      [0x3F80, 0x4000, 0x3F80, 0x4040],
      [0x4049, 0x3FCC, 0xBF80, 0x4080],
      [0x3F00, 0x3F00, 0x3F00, 0x3F40],
      [0x4228, 0x3D23, 0x40B0, 0x40E5],
      [0xBF80, 0x4000, 0x4080, 0x4000],
      [0x4315, 0x3F80, 0x3B95, 0x4315],
    ];
    for (const [a, b, c, want] of cases) expect(f.bf16Fma(a, b, c)).toBe(want);
  });

  test('bf16Fma rounds once, against exact integer arithmetic', () => {
    const next = lcg(7);
    const code = () => ((next() >>> 31) << 15) | ((110 + (next() % 40)) << 7) | (next() % 128);
    for (let i = 0; i < 4000; i++) {
      const [a, b, c] = [code(), code(), code()];
      const want = exactFma(f.bf16ToF32(a), f.bf16ToF32(b), f.bf16ToF32(c), 8);
      expect(f.bf16ToF32(f.bf16Fma(a, b, c))).toBe(want);
    }
  });

  test('bf16Fma refuses an overflowing result', () => {
    expect(() => f.bf16Fma(0x7F00, 0x7F00, 0x3F80)).toThrow('non-finite');
  });

  test('fp32Fma rounds once, against exact integer arithmetic', () => {
    const next = lcg(11);
    const value = () => f.f32FromBits((((next() >>> 31) << 31) | ((100 + (next() % 50)) << 23) | (next() & 0x7FFFFF)) >>> 0);
    for (let i = 0; i < 4000; i++) {
      const [a, b, c] = [value(), value(), value()];
      expect(f.fp32Fma(a, b, c)).toBe(exactFma(a, b, c, 24));
    }
    expect(f.fp32Fma(2, 3, 4)).toBe(10);
  });

  test('fp32Fma refuses an overflowing result', () => {
    const big = f.f32FromBits(0x7F7FFFFF);
    expect(() => f.fp32Fma(big, big, 0)).toThrow('not finite');
  });

  // bf16_max_min_are_exact
  test('mul, div and max', () => {
    expect(f.bf16Mul(0x4000, 0x4040)).toBe(0x40C0); // 2 * 3
    expect(f.bf16Div(0x40C0, 0x4000)).toBe(0x4040); // 6 / 2
    expect(f.bf16Max(0x3F80, 0x4000)).toBe(0x4000);
    expect(f.bf16Max(0x4000, 0x3F80)).toBe(0x4000);
    expect(f.bf16Max(0x0000, 0x2F80)).toBe(0x2F80);
    expect(() => f.bf16Div(0x3F80, 0x0000)).toThrow('non-finite');
  });
});

describe('labelled hashing (transcript.rs)', () => {
  const PARENT = Buffer.from(Array.from({ length: 32 }, (_, i) => i));

  // every_role_has_pinned_default_and_keyed_subkeys
  test('every role label has the pinned subkeys', () => {
    const roles = [
      ['keyA', '8bd812084a635e5161f84d2fb34f5c50d4e979becc07437fd4d4f6f4896cbc9d',
        'e97059b9fd3a5c744225db08e9dc3e63d795a51ddce01e2ea82f5e9c06925165'],
      ['keyB', '521d46f9dd031c281000380a382c3fdcd393ff59c71336c13b7347ad2c1a938f',
        '028107616c23fc723994b6e249e45b58f4e2e36aa1b4263be9c1041002ad8a58'],
      ['seedA', '060fe1ea0482eb3b2cfb927fbcf7e5dd48118192fcb45dca34a58faec0469299',
        '9b454df4917a237f845a35fec59bda9e598b0c1222e09abe52cea8bf118890e6'],
      ['seedB', '5472d30cc74f93d9273d7d3612bde3702b8f2f5958b4efc5e9235b0ec86b4e3a',
        '8330940458d1f040028c474dff27882dcf717656c5ed49bb9ca083bbff39faf1'],
      ['noiseLine', '7aadc28ccbbef879bd58974d4b38c9d538c33b21ee6460b3185e933df1179e90',
        '3f5176a01068c0b9e3c364a8670767a1b909dc5e2762ff99110edbcfe424c198'],
      ['jackpot', 'bce6af9f8fc1313d6904c8315a10e58b476a0cda822c1be77233c98eb1127eb5',
        '1abccbdca824c1fa9707f6456144fcd48a2567a143ac8dc85c7b6ff11f3dc28f'],
      ['zkPublic', 'ffc5d44fcccba998161cc9890c2d2ceb6133a1689a4d6699d3552d646fa04ff3',
        'b6ab192fc0b2d2351210f87dfe8575353c1d9c6f03f20ec92762f1b2f5789aad'],
    ];
    for (const [role, plain, keyed] of roles) {
      expect(f.subkey(f.LABELS[role], null).toString('hex')).toBe(plain);
      expect(f.subkey(f.LABELS[role], PARENT).toString('hex')).toBe(keyed);
    }
  });

  // hash_labelled_matches_python_default_and_keyed_messages
  test('hashLabelled matches the Python blake3 vectors', () => {
    const long = Buffer.from(Array.from({ length: 64 }, (_, i) => i));
    const cases = [
      [Buffer.alloc(0), 'f9c774347cc10b0dd5729b13925f246e8d21487dce485be8ab4013727c2a2d15',
        '79311dd1640edf6873c8d7a956b501b2562e203e41a11bf24ff24adf1c1a32f9'],
      [Buffer.from('x'), '7cdc56fbcde7155cbe3604c99a35831cca9fbeb53e1964d7b6f0cb7b5ce65c69',
        '66e8588103d496ebb3ee49e8f3d6a46309fe808f354bab28f8d058449f88df54'],
      [long, '8cd0fb004b120832471a26ce8958f469c55cb19684b6dee0687bbc6dbf890bbb',
        'c9bfd8a818a577b8aa717b0b58294381fff818e0af65ed1283189e644f778d11'],
    ];
    for (const [message, plain, keyed] of cases) {
      expect(f.hashLabelled(message, f.LABELS.seedA, null).toString('hex')).toBe(plain);
      expect(f.hashLabelled(message, f.LABELS.seedA, PARENT).toString('hex')).toBe(keyed);
    }
  });

  // operand_digest_fp10_matches_reference
  test('operandDigest matches the pinned digest', () => {
    const scalesRoot = Buffer.from(Array.from({ length: 32 }, (_, i) => (i * 7) & 0xFF));
    expect(f.operandDigest(PARENT, scalesRoot, PARENT).toString('hex'))
      .toBe('64eaeed16fa66e94a72bd3833ef80f083dac14894c2311ba93fe7d0a8139a748');
  });

  // encode_pins (dense_params: k 2048, B200, A 256 rows tile 4, B 128 rows tile 128)
  const job = {
    k: 2048,
    device: B200,
    a: { numRows: 256, hashId: 3, tileSize: 4 },
    b: { numRows: 128, hashId: 3, tileSize: 128 },
  };

  test('pA and pB match the pinned encodings', () => {
    expect(f.encodePA(job).toString('hex')).toBe('000100000304000000');
    expect(f.encodePB(job).toString('hex')).toBe('00080000200000018000000003800000000000');
  });

  test('keys, seeds and the ticket follow the transcript', () => {
    const proposedHeader = Buffer.alloc(76, 5);
    const ancestorHeader = Buffer.alloc(108, 6);
    const hashA = Buffer.alloc(32, 7);
    const hashB = Buffer.alloc(32, 8);
    const s = f.noiseSeeds({ proposedHeader, ancestorHeader, hashA, hashB, job });
    expect(s.keyA).toEqual(keyedHash(hash(f.LABELS.keyA), proposedHeader));
    expect(s.keyB).toEqual(keyedHash(hash(f.LABELS.keyB), ancestorHeader));
    expect(s.seedB).toEqual(f.hashLabelled(Buffer.concat([hashB, s.keyB, f.encodePB(job)]), f.LABELS.seedB, null));
    expect(s.seedA).toEqual(
      f.hashLabelled(Buffer.concat([hashA, s.seedB, s.keyA, f.encodePA(job)]), f.LABELS.seedA, null));
    // B's seed does not move with the proposed header; A's does.
    const other = f.noiseSeeds({ proposedHeader: Buffer.alloc(76, 9), ancestorHeader, hashA, hashB, job });
    expect(other.seedB).toEqual(s.seedB);
    expect(other.seedA).not.toEqual(s.seedA);

    // ticket_depends_only_on_seed_and_message
    const message = Buffer.alloc(64, 0x5A);
    const seed = Buffer.alloc(32, 0x11);
    expect(f.jackpotTicket(seed, message)).toEqual(f.hashLabelled(message, f.LABELS.jackpot, seed));
    expect(f.jackpotTicket(seed, message)).not.toEqual(keyedHash(seed, message));
  });
});

describe('noise lines (noise.rs)', () => {
  const seedA = Buffer.alloc(32, 0x22);
  const seedB = Buffer.alloc(32, 0x11);
  const E7 = 'be3adfead867e55be469dd50e9e85c50dfdc64e2dfe4eae2e8d6596a6746dd65';
  const F0 = 'c65268603369d96362e16a5f6ae6615fe6db43eae463e2d3eae55ae3bb5f54e3';

  // sample_line_pins_fixed_seed_address_bytes
  test('the four pinned lines', () => {
    const cases = [
      [seedA, f.SIDE_A, f.FACTOR_E, 7, E7],
      [seedB, f.SIDE_A, f.FACTOR_F, 0, F0],
      [seedB, f.SIDE_B, f.FACTOR_E, 7, 'e6dce3e05ce4684ae1dee56555596ae3605de9e6e968e961dce1c2e8d75ee559'],
      [seedB, f.SIDE_B, f.FACTOR_F, 0, '59de69d9696352dee8cee7d4e6da66615be9df69dd66dfe0d7e44f685fe96553'],
    ];
    for (const [seed, side, factor, line, want] of cases) {
      const got = f.sampleLine(f.noiseLineKey(seed), side, factor, line, 32);
      expect(Buffer.from(got).toString('hex')).toBe(want);
    }
  });

  // noise_line_is_normalized_to_target_norm
  test('a line has an L2 norm near 256', () => {
    const line = f.sampleLine(f.noiseLineKey(seedA), f.SIDE_A, f.FACTOR_E, 3, 32);
    const norm = Math.sqrt(Array.from(line).reduce((s, c) => s + f.e4m3ToF32(c) ** 2, 0));
    expect(Math.abs(norm - 256)).toBeLessThan(16);
  });

  test('normalizeLine handles both signs and the extremes', () => {
    const out = f.normalizeLine(Uint8Array.from([0x00, 0x80, 0x7F, 0xFF]));
    expect(f.e4m3ToF32(out[0])).toBeGreaterThan(0);
    expect(f.e4m3ToF32(out[1])).toBe(-f.e4m3ToF32(out[0]));
    expect(f.e4m3ToF32(out[3])).toBe(-f.e4m3ToF32(out[2]));
  });

  // noise_shapes_and_determinism, noise_lines_are_keyed_by_address_side_and_seed
  test('sampleNoise lays the four factors out by row index', () => {
    const n = f.sampleNoise({ k: 4, seedA, seedB, aRows: [7, 8], bRows: [7] });
    expect(n.a.e).toHaveLength(64);
    expect(n.b.e).toHaveLength(32);
    expect(n.a.f).toHaveLength(128);
    expect(n.b.f).toHaveLength(128);
    expect(Buffer.from(n.a.e.subarray(0, 32)).toString('hex')).toBe(E7);
    expect(Buffer.from(n.a.f.subarray(0, 32)).toString('hex')).toBe(F0);
    expect(Buffer.from(n.b.e).toString('hex')).toBe('e6dce3e05ce4684ae1dee56555596ae3605de9e6e968e961dce1c2e8d75ee559');
    expect(Buffer.from(n.b.f.subarray(0, 32)).toString('hex'))
      .toBe('59de69d9696352dee8cee7d4e6da66615be9df69dd66dfe0d7e44f685fe96553');
    expect(Buffer.from(n.a.e.subarray(32))).not.toEqual(Buffer.from(n.a.e.subarray(0, 32)));
  });
});

describe('the FP10 planes and row norms (prequant.rs)', () => {
  const ints = Int8Array.from([
    5, -3, 127, 0, 1, -1, 64, -64, 2, 100, -100, 7, -7, 33, -33, 12,
    -1, 1, -2, 2, 3, -3, 4, -4, 8, -8, 16, -16, 32, -32, 63, -63,
  ]);
  const scales = Uint16Array.from([0x3CA0, 0x4020, 0x3F00, 0x3D00]);

  // open_prequant_matches_reference_vectors
  test('openPrequant matches the torch vectors', () => {
    expect(Array.from(f.openPrequant(ints, scales, 2, 16))).toEqual([
      0x3DC8, 0xBD70, 0x401F, 0x0000, 0x3CA0, 0xBCA0, 0x3FA0, 0xBFA0,
      0x40A0, 0x437A, 0xC37A, 0x418C, 0xC18C, 0x42A5, 0xC2A5, 0x41F0,
      0xBF00, 0x3F00, 0xBF80, 0x3F80, 0x3FC0, 0xBFC0, 0x4000, 0xC000,
      0x3E80, 0xBE80, 0x3F00, 0xBF00, 0x3F80, 0xBF80, 0x3FFC, 0xBFFC,
    ]);
  });

  // open_prequant_validates_shapes, exact_norms_validates_shapes, exact_norms_rejects_non_finite_scales
  test('shapes and scales are validated', () => {
    const one = Uint16Array.of(0x3F80);
    expect(() => f.openPrequant(new Int8Array(3), one, 1, 3)).toThrow('multiple of 8');
    expect(() => f.openPrequant(new Int8Array(0), new Uint16Array(0), 1, 0)).toThrow('multiple of 8');
    expect(() => f.openPrequant(new Int8Array(8), Uint16Array.of(0x3F80, 0x3F80), 1, 8)).toThrow('scales must be');
    expect(() => f.openPrequant(new Int8Array(7), one, 1, 8)).toThrow('int_values must be');
    expect(() => f.openPrequant(new Int8Array(8), Uint16Array.of(0x7F80), 1, 8)).toThrow('not finite');
    expect(() => f.exactNorms(new Int8Array(8).fill(1), Uint16Array.of(0x7F80), 1, 8)).toThrow('not finite');
  });

  // exact_norms_matches_reference_vectors
  test('exactNorms matches the torch vectors', () => {
    expect(f.exactNorms(ints, scales, 2, 16)).toEqual([[0x42BC, 0x437A], [0x3FA0, 0x4000]]);
  });

  // exact_norms_l2_matches_the_zk_certificate
  test('exactNorms matches the certificate fixtures', () => {
    const seededRow = (seed, k) => {
      const mask = (1n << 64n) - 1n;
      let x = ((BigInt(seed) * 0x9E3779B97F4A7C15n) & mask) | 1n;
      const next = () => {
        x ^= (x << 13n) & mask;
        x ^= x >> 7n;
        x ^= (x << 17n) & mask;
        return x;
      };
      const row = Int8Array.from({ length: k }, () => Number(BigInt.asIntN(8, next())));
      const sc = Uint16Array.from({ length: k / 8 }, () => 0x3C00 + Number(next() % 0x400n));
      return [row, sc];
    };
    for (const [seed, k, l2] of [[877130, 1024, 0x4224], [110113, 2048, 0x422C]]) {
      const [row, sc] = seededRow(seed, k);
      expect(f.exactNorms(row, sc, 1, k)[0][0]).toBe(l2);
    }
    // Every element 2^60.
    const big = f.exactNorms(new Int8Array(1024).fill(1), new Uint16Array(128).fill((127 + 60) << 7), 1, 1024);
    expect(big[0][0]).toBe(0x5D80);
  });

  test('zero rows, dead blocks and tiny rows', () => {
    // An all-zero row has both norms zero.
    expect(f.exactNorms(new Int8Array(16), Uint16Array.of(0x3F80, 0x3F80), 1, 16)).toEqual([[0, 0]]);
    // A block with a zero scale or all-zero values adds nothing to l2.
    const row = new Int8Array(24);
    row.fill(4, 0, 16);
    const mixed = f.exactNorms(row, Uint16Array.of(0x3F80, 0x0000, 0x4500), 1, 24);
    const alone = f.exactNorms(row.subarray(0, 8), Uint16Array.of(0x3F80), 1, 8);
    expect(mixed[0][1]).toBe(alone[0][1]);
    expect(f.bf16ToF32(mixed[0][0])).toBeCloseTo(4 / Math.sqrt(3), 1);
    // The smallest subnormal scale: sqrt lands in the subnormal BF16 codes.
    const sub = f.exactNorms(new Int8Array(8).fill(1), Uint16Array.of(0x0001), 1, 8);
    expect(sub[0]).toEqual([0x0000, 0x0001]);
    const sub2 = f.exactNorms(new Int8Array(8).fill(100), Uint16Array.of(0x0001), 1, 8);
    expect(sub2[0]).toEqual([100, 100]);
    // One tiny entry in a long row rounds to zero.
    const lone = new Int8Array(1024);
    lone[0] = 1;
    expect(f.exactNorms(lone, new Uint16Array(128).fill(0x0001), 1, 1024)[0][0]).toBe(0);
  });

  test('an l2 past the BF16 range is refused', () => {
    expect(() => f.exactNorms(new Int8Array(1024).fill(127), new Uint16Array(128).fill(0x7F7F), 1, 1024))
      .toThrow();
  });

  // l2_frame_width_pins_the_protocol_formula
  test('l2FrameWidth', () => {
    expect(f.l2FrameWidth(1024)).toBe(17);
    expect(f.l2FrameWidth(2080)).toBe(15);
    expect(f.l2FrameWidth(1 << 16)).toBe(11);
    expect(f.l2FrameWidth(1 << 27)).toBe(0);
    expect(() => f.l2FrameWidth((1 << 27) + 1)).toThrow('L2 frame width');
  });

  // round_l2_clears_tail_bits_and_rounds_to_nearest
  test('roundL2ToGrid', () => {
    expect(f.roundL2ToGrid(0x3F80)).toBe(0x3F80);
    expect(f.roundL2ToGrid(0x3F81)).toBe(0x3F80);
    expect(f.roundL2ToGrid(0x3F82)).toBe(0x3F84);
    expect(f.roundL2ToGrid(0x3F83)).toBe(0x3F84);
    expect(f.roundL2ToGrid(0x3FFF)).toBe(0x4000);
  });

  test('the square root does not depend on its starting guess', () => {
    // sqrt(s * 2^(eMax - 14 - wl2) / k) for a few sums, from seeds off in both
    // directions and across a binade edge.
    const cases = [[123456789n, 254, 17, 1024], [1n << 40n, 254, 17, 1024], [987654321987n, 260, 11, 65536]];
    for (const [s, eMax, wl2, k] of cases) {
      const want = f.rneSqrtHat(s, eMax, wl2, k);
      const [t, e] = f.sqrtSeed(s, eMax, wl2, k);
      for (const seed of [[t, e - 1], [t, e + 1], [128, e], [255, e], [t - 3 < 128 ? 128 : t - 3, e]]) {
        expect(f.rneSqrtHat(s, eMax, wl2, k, seed)).toBe(want);
      }
    }
    expect(() => f.rneSqrtHat(123456789n, 254, 17, 1024, [128, 200])).toThrow('did not converge');
    // An exact tie: sqrt = 128.5 units. It rounds to the even code, from either side.
    const tie = BigInt(514 * 514 * 512);
    expect(f.rneSqrtHat(tie, 254, 17, 1024, [128, 120])).toBe(120 << 7);
    expect(f.rneSqrtHat(tie, 254, 17, 1024, [129, 120])).toBe(120 << 7);
    expect(f.rneSqrtHat(tie, 254, 17, 1024)).toBe(120 << 7);
    // A value just under a power of two rounds its guess up into the next binade.
    expect(f.sqrtSeed(BigInt(Math.floor(2 ** 41 * 0.998)), 254, 17, 1024)).toEqual([128, 127]);
  });
});

describe('the FP8 matmul (utils.rs)', () => {
  const bits = (arr) => Array.from(arr, (x) => f.f32Bits(x).toString(16).padStart(8, '0'));

  // b200_matmul_fp8_matches_python_reference_vectors
  test('B200 matches the nine vectors from Pearl\'s B200 simulator', () => {
    expect(b200Vectors).toHaveLength(9);
    for (const v of b200Vectors) {
      const acc = v.c && Float32Array.from(v.c, (h) => f.f32FromBits(parseInt(h, 16)));
      const out = f.matmulFp8(B200, hex(v.a), hex(v.b), acc, v.m, v.n, v.k);
      expect(bits(out)).toEqual(v.expected);
    }
  });

  // b200_fp8_window_keeps_small_products
  test('B200 keeps 25 bits below the largest term and drops what is further', () => {
    const a = new Uint8Array(32);
    const b = new Uint8Array(32);
    a[0] = b[0] = 0x38;
    a[1] = b[1] = 0x01;
    expect(bits(f.matmulFp8(B200, a, b, null, 1, 1, 32))).toEqual(['3f800020']);
    // Against a carry of 2^24 the same 2^-18 product is 42 bits down: gone.
    const carry = Float32Array.of(2 ** 24);
    const a2 = new Uint8Array(32);
    a2[0] = 0x01;
    expect(bits(f.matmulFp8(B200, a2, b, carry, 1, 1, 32))).toEqual(['4b800000']);
  });

  const pattern = (len, codes) => Uint8Array.from({ length: len }, (_, i) => codes[i % codes.length]);

  // matmul_group_boundaries_align_to_32_products, matmul_fp8_partials_chain_the_group_accumulator
  test('B200 collapses every 32 products, and the partials are that chain', () => {
    const [m, n, k] = [2, 2, 96];
    const a = pattern(m * k, [0x00, 0x38, 0xB0, 0x42, 0x2C]);
    const b = pattern(n * k, [0x38, 0x00, 0x3C, 0xB4, 0x29]);
    const full = f.matmulFp8(B200, a, b, null, m, n, k);
    const { out, partials } = f.matmulB200(a, b, null, m, n, k, true);
    expect(bits(out)).toEqual(bits(full));
    let acc = null;
    for (let col = 0, g = 0; col < k; col += 32, g++) {
      const cut = (x, rows) => Uint8Array.from({ length: rows * 32 }, (_, i) => x[Math.floor(i / 32) * k + col + (i % 32)]);
      acc = f.matmulFp8(B200, cut(a, m), cut(b, n), acc, m, n, 32);
      for (let cell = 0; cell < m * n; cell++) expect(partials[cell][g]).toBe(acc[cell]);
    }
    expect(bits(acc)).toEqual(bits(full));
    expect(partials.every((p) => p.length === 3)).toBe(true);
    const short = f.matmulB200(a.subarray(0, 40), b.subarray(0, 40), null, 1, 1, 40, true);
    expect(short.partials[0]).toHaveLength(2);
  });

  // matmul_fp8_all_zeros_is_zero, matmul_fp8_handles_zeros_without_panicking
  test('zeros', () => {
    expect(bits(f.matmulFp8(B200, new Uint8Array(32), new Uint8Array(32), null, 2, 2, 16)))
      .toEqual(['00000000', '00000000', '00000000', '00000000']);
    const a = Uint8Array.from({ length: 32 }, (_, i) => (i % 3 === 0 ? 0x00 : 0x38));
    const b = Uint8Array.from({ length: 32 }, (_, i) => (i % 2 === 0 ? 0x38 : 0x00));
    const once = f.matmulFp8(B200, a, b, null, 2, 2, 16);
    const dot = (i, j) => {
      let sum = 0;
      for (let u = 0; u < 16; u++) sum += f.e4m3ToF32(a[i * 16 + u]) * f.e4m3ToF32(b[j * 16 + u]);
      return sum;
    };
    const want = [dot(0, 0), dot(0, 1), dot(1, 0), dot(1, 1)];
    expect(Array.from(once)).toEqual(want);
    expect(Array.from(f.matmulFp8(B200, a, b, once, 2, 2, 16))).toEqual(want.map((x) => 2 * x));
  });

  // subnormal_carry_renormalizes_through_the_fp8_window
  test('a subnormal carry survives the window', () => {
    const zeros = new Uint8Array(32);
    for (const carry of [0x00400000, 0x00000200, 0x00000001, 0x807FFFFF]) {
      const out = f.matmulFp8(B200, zeros, zeros, Float32Array.of(f.f32FromBits(carry)), 1, 1, 32);
      expect(f.f32Bits(out[0])).toBe(carry);
    }
  });

  // group_sum_zero_result_preserves_sign_on_underflow
  test('a sum that underflows keeps its sign', () => {
    expect(f.f32Bits(f.groupSumOf([{ neg: true, exp: -252, sig: 0x800000 }], 26))).toBe(0x80000000);
    expect(f.f32Bits(f.groupSumOf([{ neg: false, exp: -252, sig: 0x800000 }], 26))).toBe(0);
    // Exact cancellation is +0.
    const one = { exp: 0, sig: 0x800000 };
    expect(f.f32Bits(f.groupSumOf([{ neg: true, ...one }, { neg: false, ...one }], 26))).toBe(0);
  });

  test('a non-finite carry and an unknown device are refused', () => {
    const z = new Uint8Array(32);
    expect(() => f.matmulFp8(B200, z, z, Float32Array.of(Infinity), 1, 1, 32)).toThrow('not finite');
    expect(() => f.matmulFp8(2, z, z, null, 1, 1, 32)).toThrow('unsupported FP8 device 2');
  });

  // h100_replay_resets_every_four_atoms_and_promotes_in_fp32
  test('H100 restarts every 128 products and promotes in FP32', () => {
    const k = 1024;
    const a = new Uint8Array(k);
    a.fill(0x78, 0, 128);
    a.fill(0x38, 128, 256);
    const b = new Uint8Array(k).fill(0x38);
    expect(Array.from(f.matmulFp8(H100, a, b, null, 1, 1, k))).toEqual([32896]);
  });

  // h100_replay_handles_partial_windows_and_canonicalizes_zero
  test('H100 handles a short last window and returns +0', () => {
    for (const [k, expected] of [[1056, 32], [1088, 64], [1120, 96], [1152, 128]]) {
      const a = new Uint8Array(k);
      a.fill(0x38, 1024);
      expect(Array.from(f.matmulFp8(H100, a, new Uint8Array(k).fill(0x38), null, 1, 1, k))).toEqual([expected]);
    }
    const a = new Uint8Array(1024);
    a[0] = 0x78;
    a[1] = 0xF8;
    expect(f.f32Bits(f.matmulFp8(H100, a, new Uint8Array(1024).fill(0x38), null, 1, 1, 1024)[0])).toBe(0);
    // A window that sums to -0 still promotes to +0.
    const neg = new Uint8Array(32);
    neg[0] = 0x81;
    const tiny = new Uint8Array(32);
    tiny[0] = 0x01;
    expect(f.f32Bits(f.matmulFp8(H100, neg, tiny, null, 1, 1, 32)[0])).toBe(f.f32Bits(-(2 ** -18)));
  });

  test('H100 refuses a carry and a k that is not a multiple of 32', () => {
    const z = new Uint8Array(40);
    expect(() => f.matmulFp8(H100, z, z, Float32Array.of(0), 1, 1, 32)).toThrow('carry-in');
    expect(() => f.matmulFp8(H100, z, z, null, 1, 1, 40)).toThrow('divisible by 32');
    expect(() => f.matmulFp8(H100, z, z, null, 1, 1, 0)).toThrow('divisible by 32');
  });
});

describe('row scales and the noised operand (quantization.rs)', () => {
  // derive_row_scales_matches_reference_vectors
  test('deriveRowScales matches the torch vectors', () => {
    const cases = [
      [0x3F80, 0x3F80, 16, 0x4315, 0x3B95],
      [0x3F80, 0x4080, 16, 0x4295, 0x3B15],
      [0x2F80, 0x2F80, 16, 0x5315, 0x3B95],
      [0x3F00, 0x4040, 64, 0x42B3, 0x3B33],
    ];
    for (const [l2, linf, r, alpha, beta] of cases) {
      expect(f.deriveRowScales(B200, l2, linf, r)).toEqual([alpha, beta]);
    }
  });

  // device_quantization_pins_noise_alpha_beta_and_code
  test('each device pins its own noise value, scales and code', () => {
    const e = hex('be3adfead867e55be469dd50e9e85c50dfdc64e2dfe4eae2e8d6596a6746dd65');
    const line = hex('c65268603369d96362e16a5f6ae6615fe6db43eae463e2d3eae55ae3bb5f54e3');
    const noise = { e, f: Buffer.concat(Array(8).fill(line)) };
    const ints = new Int8Array(8).fill(1);
    const scales = Uint16Array.of(16256);
    const norms = [[16256, 16384]];
    for (const [device, noiseBits, alpha, beta, code] of [
      [H100, 1161306112, 17002, 15269, 105],
      [B200, 1161306624, 17082, 15236, 109],
    ]) {
      expect(f.f32Bits(f.matmulFp8(device, e, line, null, 1, 1, 32)[0])).toBe(noiseBits);
      const built = f.noisyQuantize(device, ints, scales, noise, norms);
      expect(built.alpha[0]).toBe(alpha);
      expect(built.beta[0]).toBe(beta);
      expect(built.noised[0]).toBe(code);
    }
  });

  test('a zero row is floored, and a row with too little RMS is refused', () => {
    const noise = {
      e: Uint8Array.from({ length: 16 }, (_, i) => (i % 3 === 0 ? 0xB0 : 0x30)),
      f: Uint8Array.from({ length: 8 * 16 }, (_, i) => (i % 5 === 0 ? 0xB0 : 0x30)),
    };
    const built = f.noisyQuantize(B200, new Int8Array(8), Uint16Array.of(0x3F80), noise, [[0, 0]]);
    expect(built.l2[0]).toBe(0x2F80);
    expect(f.bf16ToF32(built.beta[0])).toBeGreaterThan(0);
    // A scaled RMS between 4 and 8 takes the wider strip.
    const small = f.noisyQuantize(B200, new Int8Array(8).fill(1), Uint16Array.of(0x3F80), noise, [[0x3F80, 0x4290]]);
    const rms = f.bf16ToF32(small.alpha[0]) * f.bf16ToF32(small.l2[0]);
    expect(rms).toBeGreaterThanOrEqual(4);
    expect(rms).toBeLessThan(8);
    // l2 1 against a peak of 448: alpha is about 1, so the scaled RMS is too.
    expect(() => f.noisyQuantize(B200, new Int8Array(8), Uint16Array.of(0x3F80), noise, [[0x3F80, 0x43E0]]))
      .toThrow('scaled_rms >= 4');
  });
});

describe('tile layout and the fold (layout.rs, utils.rs)', () => {
  test('checkLotteryLayout', () => {
    for (const [h, w] of [[4, 64], [4, 128], [16, 32], [16, 16], [4, 512], [8, 256]]) {
      expect(() => f.checkLotteryLayout(h, w)).not.toThrow();
    }
    expect(() => f.checkLotteryLayout(2, 128)).toThrow('at least 4 x 16');
    expect(() => f.checkLotteryLayout(16, 8)).toThrow('at least 4 x 16');
    expect(() => f.checkLotteryLayout(4, 1024)).toThrow('256..=2048');
    expect(() => f.checkLotteryLayout(4, 32)).toThrow('256..=2048');
    expect(() => f.checkLotteryLayout(5, 100)).toThrow('256..=2048');
  });

  test('foldTile is sixteen rotate-multiply-add lanes over consecutive runs', () => {
    const tile = Float32Array.from({ length: 64 }, (_, i) => (i % 5 === 0 ? -0 : i * 1.25 - 20));
    const got = f.foldTile(tile);
    expect(got).toHaveLength(64);
    for (let lane = 0; lane < 16; lane++) {
      let v = 0n;
      for (let i = lane * 4; i < lane * 4 + 4; i++) {
        v = (v * 0x9E3779B1n + BigInt(f.f32Bits(tile[i]))) & 0xFFFFFFFFn;
        v = ((v << 13n) | (v >> 19n)) & 0xFFFFFFFFn;
      }
      expect(got.readUInt32LE(lane * 4)).toBe(Number(v));
    }
    // -0 and +0 fold differently.
    expect(f.foldTile(new Float32Array(16).fill(-0))).not.toEqual(f.foldTile(new Float32Array(16)));
  });
});

describe('the policy\'s integer scores (unpredictability.rs)', () => {
  // log2_fixed_matches_its_real_form
  test('log2Fixed is floor(64 log2 n)', () => {
    for (let n = 1; n < 65536; n++) {
      if (f.log2Fixed(n) !== Math.floor(64 * Math.log2(n))) throw new Error(`log2Fixed(${n})`);
    }
    expect(f.log2Fixed(1 << 13)).toBe(832);
    expect(f.log2Fixed(65535)).toBe(1023);
  });

  // lambda_stays_in_the_one_sided_band
  test('lambda stays in its one-sided band around 64 (log2 S + 508)', () => {
    const xs = [0x0000, 0x8000, 0x0001, 0x007F, 0x0080, 0x3F80, 0xBF80, 0x7F7F, 0x0100];
    for (let i = 0; i < 600; i++) {
      const code = (Math.imul(i, 2654435761) >>> 16) & 0xFFFF;
      if (((code >> 7) & 0xFF) !== 255) xs.push(code);
    }
    const alphas = Array.from({ length: 40 }, (_, i) => ((7 + i * 4) << 7) | ((i * 11) % 128));
    const l2s = Array.from({ length: 30 }, (_, i) => ((95 + i * 5) << 7) | ((i * 23) % 128));
    let max = 0;
    for (const [device, delta] of [[B200, 0.5], [H100, 1]]) {
      alphas.forEach((alpha, i) => {
        xs.forEach((x, j) => {
          const l2f = l2s[(i * 7 + j) % 30];
          const lam = f.lambda(device, alpha, x, l2f);
          const au = f.bf16ToF32(alpha) * f.bf16ToF32(x);
          const sigma = delta * f.bf16ToF32(alpha) * f.bf16ToF32(l2f);
          const ideal = 64 * (Math.log2(au * au + sigma * sigma) + 508);
          if (!(lam > ideal - 1.1 && lam <= ideal + 1e-6)) throw new Error(`lambda ${lam} vs ${ideal}`);
          max = Math.max(max, lam);
        });
      });
    }
    expect(max).toBeGreaterThan(12928);
    expect(max).toBeLessThanOrEqual(54207);
  });

  test('lambda refuses codes outside the scheme', () => {
    expect(() => f.lambda(B200, 0x7F80, 0x3F80, 0x3F80)).toThrow('non-finite');
    expect(() => f.lambda(B200, 0x0001, 0x3F80, 0x3F80)).toThrow('must be normal');
    expect(() => f.lambda(B200, 0x3F80, 0x3F80, 0x0001)).toThrow('must be normal');
  });

  test('skip', () => {
    expect(f.skip(0, 0, 0)).toBe(false);
    expect(f.skip(20000, 20000, 100)).toBe(true);
    expect(f.skip(30000, 28752, 100)).toBe(false);
    expect(f.skip(30000, 28751, 100)).toBe(true);
  });
});

// jackpot_policy.rs `build`: one side from clean rows, with mesh-like noise.
function buildSide(rows, k, r, seed) {
  const numRows = rows.length / k;
  const ints = Int8Array.from(rows, (x) => Math.trunc(f.bf16ToF32(x) * 16));
  const scales = new Uint16Array(numRows * (k / 8)).fill(0x3D80);
  const noise = {
    e: Uint8Array.from({ length: numRows * r }, (_, i) => ((i + seed) % 3 === 0 ? 0xB0 : 0x30)),
    f: Uint8Array.from({ length: k * r }, (_, i) => ((i + seed) % 5 === 0 ? 0xB0 : 0x30)),
  };
  const side = f.buildSide(B200, ints, scales, numRows, k, noise);
  expect(Array.from(side.clean)).toEqual(Array.from(rows));
  return side;
}

// jackpot_policy.rs `honest_rows`.
function honestRows(numRows, k) {
  const vals = [0x3F80, 0xBF00, 0x3E80, 0xBFC0, 0x3F00, 0x4000, 0xBF80, 0x3D80];
  return Uint16Array.from({ length: numRows * k }, (_, i) => vals[(i * 7 + Math.floor(i / k)) % 8]);
}

describe('the jackpot policy (jackpot_policy.rs)', () => {
  // gauss_kernel_matches_its_table_and_vanishes_past_290
  test('gaussKernel matches the pinned values', () => {
    expect(f.gaussKernel(0)).toBe(65536);
    expect(f.gaussKernel(1)).toBe(65180);
    expect(f.gaussKernel(100)).toBe(24674);
    expect(f.gaussKernel(150)).toBe(8537);
    expect(f.gaussKernel(290)).toBe(1);
    let sum = 0;
    for (let d = 0; d < 291; d++) sum += f.gaussKernel(d);
    expect(sum).toBe(5617275);
    expect(f.gaussKernel(291)).toBe(0);
    expect(f.gaussKernel(54207)).toBe(0);
  });

  // gauss_row_constants_are_exact
  test('gaussRowConstants matches the pinned rows', () => {
    expect(f.gaussRowConstants(0x4180, 0x3F80, B200)).toEqual([6720, 65536]);
    expect(f.gaussRowConstants(0x4080, 0x3F80, B200)).toEqual([8192, 23488]);
    expect(() => f.gaussRowConstants(0x4040, 0x3F80, B200)).toThrow('RMS in [4, 256)');
    expect(() => f.gaussRowConstants(0x4380, 0x3F80, B200)).toThrow('RMS in [4, 256)');
  });

  // honest_tile_is_admissible_with_default_thresholds
  test('an honest tile is admissible and returns the replayed tile', () => {
    const k = 64;
    const a = buildSide(honestRows(2, k), k, 16, 0);
    const b = buildSide(honestRows(3, k), k, 16, 1);
    const tile = f.runChecks(B200, a, b, k);
    expect(tile).toHaveLength(6);
    expect(Array.from(tile)).toEqual(Array.from(f.matmulFp8(B200, a.noised, b.noised, null, 2, 3, k)));
  });

  // sigma_floor_rejects_rows_with_vanishing_noise_scale
  test('a row with no noise scale is rejected, on either side', () => {
    const k = 64;
    const make = () => [buildSide(honestRows(2, k), k, 16, 0), buildSide(honestRows(2, k), k, 16, 1)];
    // With so small an l2 every entry of the row also counts as dead, so the
    // liveness check is switched off to reach the floor itself.
    const policy = { ...f.POLICY, epsIdle: 1 };
    let [a, b] = make();
    a.l2[0] = 0x2F80;
    expect(f.runChecks(B200, a, b, k)).toBeNull();
    expect(f.runChecks(B200, a, b, k, policy)).toBeNull();
    [a, b] = make();
    b.l2[1] = 0x2F80;
    expect(f.runChecks(B200, a, b, k, policy)).toBeNull();
    [a, b] = make();
    expect(f.runChecks(B200, a, b, k, policy)).not.toBeNull();
  });

  // liveness_rejects_spike_dominated_rows
  test('rows that are one spike are rejected, on either side', () => {
    const k = 16;
    const spikes = new Uint16Array(2 * k);
    spikes[0] = 0x3F80;
    spikes[k] = 0x3F80;
    const policy = { ...f.POLICY, rmsMinA: 0, rmsMinB: 0 };
    const spiky = buildSide(spikes, k, 16, 0);
    const honest = buildSide(honestRows(2, k), k, 16, 1);
    expect(f.runChecks(B200, spiky, honest, k, policy)).toBeNull();
    expect(f.runChecks(B200, honest, spiky, k, policy)).toBeNull();
  });

  // unpredictability_rejects_when_summands_are_predictable
  test('a tile whose summands are predictable is rejected', () => {
    const k = 1024;
    const rows = new Uint16Array(2 * k);
    rows[0] = 0x3F80;
    rows[k] = 0x3F80;
    const a = buildSide(rows, k, 16, 0);
    const b = buildSide(rows, k, 16, 1);
    a.l2.fill(0x2F80);
    b.l2.fill(0x2F80);
    const policy = { ...f.POLICY, epsIdle: 1, rmsMinA: 0, rmsMinB: 0 };
    expect(f.runChecks(B200, a, b, k, policy)).toBeNull();
  });

  // row_noise_coverage_rejects_flat_rows_and_accepts_varied_rows
  test('flat rows fail noise coverage, on either side', () => {
    const k = 64;
    const flatRows = Uint16Array.from({ length: 2 * k }, (_, i) => (i % 3 === 0 ? 0xBF80 : 0x3F80));
    const flat = buildSide(flatRows, k, 16, 0);
    const honest = buildSide(honestRows(2, k), k, 16, 1);
    expect(f.rowNoiseCoverageOk(honest, B200, k)).toBe(true);
    expect(f.rowNoiseCoverageOk(flat, B200, k)).toBe(false);
    expect(f.runChecks(B200, flat, honest, k)).toBeNull();
    expect(f.runChecks(B200, honest, flat, k)).toBeNull();
  });

  test('an all-zero tile has nothing to skip', () => {
    const k = 64;
    const a = buildSide(honestRows(2, k), k, 16, 0);
    const b = buildSide(honestRows(2, k), k, 16, 1);
    a.noised.fill(0);
    expect(Array.from(f.runChecks(B200, a, b, k))).toEqual([0, 0, 0, 0]);
  });

  test('only B200 has a policy here', () => {
    expect(() => f.runChecks(H100, {}, {}, 64)).toThrow('B200 only');
  });
});

describe('difficulty (proof_utils.rs)', () => {
  // test_difficulty_conversion
  test('nbitsToTarget is the compact form', () => {
    expect(f.nbitsToTarget(0x1D00FFFF)).toBe(0xFFFFn << 208n);
    expect(f.nbitsToTarget(0x1B0404CB)).toBe(0x0404CBn << 192n);
    expect(f.nbitsToTarget(0x03123456)).toBe(0x123456n);
    expect(f.nbitsToTarget(0x01123456)).toBe(0x12n);
    expect(f.nbitsToTarget(0x1D000000)).toBe(0n);
    expect(f.nbitsToTarget(0x00123456)).toBe(0n);
    expect(f.nbitsToTarget(0x1D800000)).toBe(0n);
  });

  test('a ticket wins at or below target * h * w * k', () => {
    const target = f.nbitsToTarget(0x1D00FFFF) * BigInt(16 * 16 * 1024);
    const le = (v) => Buffer.from(v.toString(16).padStart(64, '0'), 'hex').reverse();
    expect(f.meetsDifficulty(le(target), 0x1D00FFFF, 16, 16, 1024)).toBe(true);
    expect(f.meetsDifficulty(le(target + 1n), 0x1D00FFFF, 16, 16, 1024)).toBe(false);
    // An easy target times the work overflows 256 bits and is capped.
    expect(f.meetsDifficulty(Buffer.alloc(32, 0xFF), 0x207FFFFF, 16, 16, 1024)).toBe(true);
  });
});

describe('one tile, end to end (verify.rs)', () => {
  // The data mine.rs uses for its CPU miner: centred triangular int8 values
  // with unit scales, 16 x 16 x 1024.
  const k = 1024;
  const plane = (seed) => {
    const next = lcg(seed);
    return Int8Array.from({ length: 16 * k }, () => {
      const x = next();
      return ((x >>> 8) & 127) + ((x >>> 20) & 127) - 127;
    });
  };
  const opts = () => ({
    device: B200,
    k,
    seedA: Buffer.alloc(32, 0xA1),
    seedB: Buffer.alloc(32, 0xB2),
    aRows: Array.from({ length: 16 }, (_, i) => 32 + i),
    bRows: Array.from({ length: 16 }, (_, i) => 16 + i),
    aInts: plane(1),
    aScales: new Uint16Array(16 * k / 8).fill(0x3F80),
    bInts: plane(2),
    bScales: new Uint16Array(16 * k / 8).fill(0x3F80),
  });

  test('random data is admissible and gives a ticket', () => {
    const r = f.evaluateTile(opts());
    expect(r.tile).toHaveLength(256);
    expect(r.message).toEqual(f.foldTile(r.tile));
    expect(r.ticket).toEqual(f.jackpotTicket(Buffer.alloc(32, 0xA1), r.message));
    expect(Array.from(r.tile)).toEqual(Array.from(f.matmulFp8(B200, r.a.noised, r.b.noised, null, 16, 16, k)));
    // Our own regression pin. Pearl publishes no end-to-end vector to check it against.
    expect(r.ticket.toString('hex')).toBe(TICKET_PIN);
  });

  test('a tile the policy rejects gives null', () => {
    const o = opts();
    // Rows whose entries all have the same size fail the noise-coverage check.
    o.aInts = Int8Array.from({ length: 16 * k }, (_, i) => (i % 3 === 0 ? -64 : 64));
    expect(f.evaluateTile(o)).toBeNull();
  });

  test('k and the tile shape are checked', () => {
    expect(() => f.evaluateTile({ ...opts(), k: 1000 })).toThrow('k must be');
    expect(() => f.evaluateTile({ ...opts(), k: 512 })).toThrow('k must be');
    expect(() => f.evaluateTile({ ...opts(), k: 131072 })).toThrow('k must be');
    expect(() => f.evaluateTile({ ...opts(), aRows: [0, 1] })).toThrow('at least 4 x 16');
  });
});

// Real hardware. An RTX 5080 running Pearl's own fp8-scheme miner (1dd75a52)
// mined testnet2 block 150,688 on 20 November 2026, 405 blocks after the fork:
// 9a40173418fdbeec1923b8baec53215455ac9a32dd1081c05df65288574738cd. The fixture
// is the PlainProofV4 its gateway received, unpacked: the two headers, the four
// plane roots, and the opened rows of the winning 4 x 64 tile (k 2048). Pearl's
// node accepted it and the network built on it, so a pass here is agreement
// with consensus on a whole tile, from the roots to the ticket. The Merkle
// openings themselves are not replayed.
describe('a block an RTX 5080 mined on testnet2', () => {
  const hex = (s) => Buffer.from(s, 'hex');
  const ints = (b64) => new Int8Array(Buffer.from(b64, 'base64'));
  const scales = (b64) => {
    const raw = Buffer.from(b64, 'base64');
    return Uint16Array.from({ length: raw.length / 2 }, (_, i) => raw.readUInt16LE(2 * i));
  };
  const v = rtx5080;
  const proposedHeader = hex(v.proposedHeader);
  const ancestorHeader = hex(v.ancestorHeader);
  const seeds = () => f.noiseSeeds({
    proposedHeader,
    ancestorHeader,
    hashA: f.operandDigest(hex(v.roots.aValues), hex(v.roots.aScales), f.keyA(proposedHeader)),
    hashB: f.operandDigest(hex(v.roots.bValues), hex(v.roots.bScales), f.keyB(ancestorHeader)),
    job: { k: v.k, device: v.device, a: v.a, b: v.b },
  });
  const evaluate = (seedA, seedB) => f.evaluateTile({
    device: v.device,
    k: v.k,
    seedA,
    seedB,
    aRows: v.aRows,
    bRows: v.bRows,
    aInts: ints(v.aInts),
    aScales: scales(v.aScales),
    bInts: ints(v.bInts),
    bScales: scales(v.bScales),
  });

  test('the fixture is the job the block committed', () => {
    expect(v.device).toBe(B200);
    expect(proposedHeader).toHaveLength(76);
    expect(ancestorHeader).toHaveLength(108);
    expect(proposedHeader.readUInt32LE(72)).toBe(v.nbits);
    expect(f.nbitsToTarget(v.nbits)).toBe(BigInt(v.target));
    expect(v.aRows).toEqual([1568, 1569, 1570, 1571]);
    expect(v.bRows).toHaveLength(64);
    expect(v.bRows[0]).toBe(448);
  });

  test('the reference gets a ticket under the network target', () => {
    const { seedA, seedB } = seeds();
    const r = evaluate(seedA, seedB);
    expect(r.tile).toHaveLength(4 * 64);
    expect(r.ticket.toString('hex')).toBe(RTX5080_TICKET);
    expect(f.meetsDifficulty(r.ticket, v.nbits, 4, 64, v.k)).toBe(true);
    // The ticket's top 32 bits are zero. A wrong bit anywhere gives a random
    // ticket, which meets this target about once in 57 billion tries.
    expect(r.ticket.readUInt32LE(28)).toBe(0);
  });

  test('another seed does not win', () => {
    const { seedA, seedB } = seeds();
    const other = Buffer.from(seedA);
    other[0] ^= 1;
    const r = evaluate(other, seedB);
    expect(r === null || !f.meetsDifficulty(r.ticket, v.nbits, 4, 64, v.k)).toBe(true);
  });
});

const TICKET_PIN = '53f0b0d80efabba1e3e93039449e937d724f53fbe8a04033e969f55276f2566d';
const RTX5080_TICKET = 'dff324097fee6a158eedee360147e58f2244da047ecfe18d6566fd0100000000';
