'use strict';

const { hash, keyedHash } = require('./blake3');

// Pearl's PIP-3 proof of work (certificate version 4, "FP8"), on the CPU.
//
// PIP-3 replaces the INT7 matrix multiply with a floating-point one: FP8 E4M3
// operands, an FP32 accumulator, and a verifier that repeats the declared
// chip's arithmetic bit for bit. This file is a port of the plain verifier in
// Pearl's fp8-scheme branch (zk-pow/src/v4/api, commit 1dd75a52), for the
// Blackwell device ("B200": B200, RTX PRO 6000, GeForce RTX 50). The Hopper
// ("H100") matmul and quantization constants are here too; its jackpot policy
// is not.
//
// Nothing in the miner requires this file. It is the oracle a future FP8 CUDA
// core gets checked against, the way ./reference checks the INT7 cores. It is
// slow on purpose. `enabled()` is the switch any later wiring must ask.
//
// Every function names the Rust function it ports. Where the Rust guards a
// case that cannot happen for in-range inputs (an FP32 overflow inside the
// matmul, a u32 cast that cannot truncate), the guard is left out and said so,
// because this repo holds every line to 100% test coverage.
//
// See earn/tuning/pip3-fp8.md for the spec, what is still draft, and the plan.

// The mainnet fork has no height yet, so this stays off unless asked for.
function enabled(env) {
  return env.PEARL_FP8 === '1';
}

const DEVICE_H100 = 0;
const DEVICE_B200 = 1;

// The committed parameters of each device (public_params.rs `Device`).
//   delta       noise-to-signal ratio the row scales are built from
//   width       significand bits of one accumulation group's window
//   g, gSmall   strip width of the downscale rewrite; gSmall below RMS 8
const DEVICES = {
  [DEVICE_H100]: { delta: 1, width: 14, g: Math.fround(0.105), gSmall: Math.fround(0.105) },
  [DEVICE_B200]: { delta: 0.5, width: 26, g: Math.fround(0.08), gSmall: Math.fround(0.1) },
};

function deviceParams(device) {
  const p = DEVICES[device];
  if (!p) throw new Error(`unsupported FP8 device ${device}`);
  return p;
}

const RANK = 32;                 // the only rank the fork accepts
const BLOCK_SIZE = 8;            // int8 values sharing one BF16 scale (FP10)
const MMA_GROUP_PRODUCTS = 32;   // products one MMA instruction adds to its carry
const H100_WINDOW_PRODUCTS = 128;
const MAX_E4M3 = 448;
const JACKPOT_ENTRIES = 16;
const MIN_K = 1 << 10;
const MAX_K = 1 << 16;
const FOLD_MUL = 0x9E3779B1;

// ---------------------------------------------------------------------------
// Raw float views
// ---------------------------------------------------------------------------

const F32 = new Float32Array(1);
const U32 = new Uint32Array(F32.buffer);
const F64 = new Float64Array(1);
const U64 = new BigUint64Array(F64.buffer);

function f32Bits(x) {
  F32[0] = x;
  return U32[0];
}

function f32FromBits(bits) {
  U32[0] = bits;
  return F32[0];
}

// The next f32 above (`up`) or below `x`.
function f32Step(x, up) {
  if (x === 0) return f32FromBits(up ? 1 : 0x80000001);
  const bits = f32Bits(x);
  return f32FromBits((x > 0) === up ? bits + 1 : bits - 1);
}

// The next f64 above (`up`) or below a nonzero `x`.
function f64Step(x, up) {
  F64[0] = x;
  U64[0] += (x > 0) === up ? 1n : -1n;
  return F64[0];
}

function f64LowBit(x) {
  F64[0] = x;
  return Number(U64[0] & 1n);
}

// floor(log2 |x|) of a nonzero double that is not subnormal.
function f64Exponent(x) {
  F64[0] = x;
  return Number((U64[0] >> 52n) & 0x7FFn) - 1023;
}

// ---------------------------------------------------------------------------
// dtype.rs: FP8 E4M3 and BF16
// ---------------------------------------------------------------------------

// fp8_e4m3_to_f32. Exact. S.1111.111 is the format's only NaN and is banned.
function e4m3ToF32(bits) {
  if ((bits & 0x7F) === 0x7F) throw new Error('FP8 E4M3 NaN encoding is not allowed');
  const sign = bits & 0x80 ? -1 : 1;
  const exp = (bits >> 3) & 0x0F;
  const man = bits & 0x07;
  if (exp === 0) return sign * (man / 512);
  return sign * (1 + man / 8) * 2 ** (exp - 7);
}

function roundTiesEven(v) {
  const r = Math.round(v);
  return v % 1 === 0.5 && r % 2 === 1 ? r - 1 : r;
}

// f32_to_fp8_e4m3. Round to nearest, ties to even, saturating at +-448.
function f32ToE4m3(x) {
  if (!Number.isFinite(x)) throw new Error(`non-finite value ${x} unsupported`);
  const bits = f32Bits(x);
  const sign = bits >>> 31 ? 0x80 : 0;
  const mbits = bits & 0x7FFFFFFF;
  const e = (mbits >>> 23) - 127;
  if (e < -6) return sign | roundTiesEven(f32FromBits(mbits) * 512);
  let exp = e + 7;
  const fm = mbits & 0x7FFFFF;
  let mant = fm >>> 20;
  const rem = fm & 0xFFFFF;
  if (rem > 0x80000 || (rem === 0x80000 && (mant & 1) === 1)) {
    mant += 1;
    if (mant === 8) {
      mant = 0;
      exp += 1;
    }
  }
  if (exp > 15 || (exp === 15 && mant >= 7)) return sign | 0x7E;
  return sign | (exp << 3) | mant;
}

// f32_to_bf16. Round to nearest, ties to even; overflow is an error.
function f32ToBf16(x) {
  if (!Number.isFinite(x)) throw new Error(`non-finite value ${x} unsupported`);
  const bits = f32Bits(x);
  const bf16 = ((bits + 0x7FFF + ((bits >>> 16) & 1)) >>> 16) & 0xFFFF;
  if ((bf16 & 0x7FFF) >= 0x7F80) throw new Error(`value ${x} overflows bf16`);
  return bf16;
}

// bf16_to_f32. Exact: BF16 is the top half of an f32.
function bf16ToF32(bits) {
  if ((bits & 0x7FFF) >= 0x7F80) throw new Error('bf16 non-finite encoding is not allowed');
  return f32FromBits((bits << 16) >>> 0);
}

// ---------------------------------------------------------------------------
// compute.rs: element-wise BF16 and FP32 arithmetic
// ---------------------------------------------------------------------------

// A double holds the exact product of two f32 values, and a quotient rounded to
// double then to f32 is the correctly rounded f32 quotient, so Math.fround of
// the double result is the f32 operation.
function bf16Mul(a, b) {
  return f32ToBf16(Math.fround(bf16ToF32(a) * bf16ToF32(b)));
}

function bf16Div(a, b) {
  return f32ToBf16(Math.fround(bf16ToF32(a) / bf16ToF32(b)));
}

function bf16Max(a, b) {
  return bf16ToF32(a) >= bf16ToF32(b) ? a : b;
}

// bf16_fma: a*b + c with one rounding to BF16.
function bf16Fma(a, b, c) {
  const p = bf16ToF32(a) * bf16ToF32(b); // exact
  const c64 = bf16ToF32(c);
  const s = p + c64;
  const t = s - c64;
  const r = (p - t) + (c64 - (s - t)); // TwoSum: the exact error of s
  const s32 = Math.fround(s);
  const err = (s - s32) + r;
  // Round to odd, so the final cast to BF16 cannot round twice.
  const fix = err !== 0 && (f32Bits(s32) & 1) === 0 && Number.isFinite(s32);
  return f32ToBf16(fix ? f32Step(s32, err > 0) : s32);
}

// fp32_fma: a*b + c with one rounding to f32.
function fp32Fma(a, b, c) {
  const product = a * b; // exact for f32 inputs
  const total = product + c;
  const shift = total - c;
  const r = (product - shift) + (c - (total - shift));
  const inexactEven = r !== 0 && f64LowBit(total) === 0 && Number.isFinite(total);
  const rounded = Math.fround(inexactEven ? f64Step(total, r > 0) : total);
  if (!Number.isFinite(rounded)) throw new Error('f32 value is not finite');
  return rounded;
}

// ---------------------------------------------------------------------------
// transcript.rs: labelled BLAKE3, keys, seeds, the ticket
// ---------------------------------------------------------------------------

const LABELS = {
  keyA: Buffer.from('pearl/v4/FP8/key-A'),
  keyB: Buffer.from('pearl/v4/FP8/key-B'),
  seedA: Buffer.from('pearl/v4/FP8/seed-A'),
  seedB: Buffer.from('pearl/v4/FP8/seed-B'),
  noiseLine: Buffer.from('pearl/v4/FP8/noise-line'),
  jackpot: Buffer.from('pearl/v4/FP8/jackpot'),
  zkPublic: Buffer.from('pearl/v4/FP8/zk-public'),
};

// subkey: a role key from a label, under a parent key or unkeyed.
function subkey(label, key) {
  return key ? keyedHash(key, label) : hash(label);
}

function hashLabelled(message, label, key) {
  return keyedHash(subkey(label, key), message);
}

// keyA = H_"key-A"(the 76-byte proposed header). Keys operand A's trees.
function keyA(proposedHeader) {
  return hashLabelled(proposedHeader, LABELS.keyA, null);
}

// keyB = H_"key-B"(the 108-byte ancestor header). Keys operand B's trees.
function keyB(ancestorHeader) {
  return hashLabelled(ancestorHeader, LABELS.keyB, null);
}

// operand_digest_fp10: one operand's commitment, binding its two plane roots.
function operandDigest(valuesRoot, scalesRoot, key) {
  return keyedHash(key, Buffer.concat([valuesRoot, scalesRoot]));
}

function u32le(v) {
  const b = Buffer.alloc(4);
  b.writeUInt32LE(v);
  return b;
}

// OperandParams::encode: (num_rows u32, hash_id u8, tile_size u32).
function encodeOperand(o) {
  return Buffer.concat([u32le(o.numRows), Buffer.from([o.hashId]), u32le(o.tileSize)]);
}

// encode_p_b: (k u32, r u16, quant u8, device u8, operand B, experts u16).
// Quant 0 is the only scheme; experts 0 means a dense job.
function encodePB(job) {
  const head = Buffer.alloc(8);
  head.writeUInt32LE(job.k, 0);
  head.writeUInt16LE(RANK, 4);
  head[6] = 0;
  head[7] = job.device;
  return Buffer.concat([head, encodeOperand(job.b), Buffer.alloc(2)]);
}

// encode_p_a, dense: operand A alone.
function encodePA(job) {
  return encodeOperand(job.a);
}

// noise_seeds, dense:
//   seedB = H_"seed-B"(HB || keyB || pB)
//   seedA = H_"seed-A"(HA || seedB || keyA || pA)
function noiseSeeds(opts) {
  const kA = keyA(opts.proposedHeader);
  const kB = keyB(opts.ancestorHeader);
  const seedB = hashLabelled(Buffer.concat([opts.hashB, kB, encodePB(opts.job)]), LABELS.seedB, null);
  const seedA = hashLabelled(
    Buffer.concat([opts.hashA, seedB, kA, encodePA(opts.job)]), LABELS.seedA, null);
  return { keyA: kA, keyB: kB, seedA, seedB };
}

// compute_jackpot_ticket: the 64-byte message hashed under seedA's jackpot key.
function jackpotTicket(seedA, message) {
  return hashLabelled(message, LABELS.jackpot, seedA);
}

// ---------------------------------------------------------------------------
// noise.rs: the noise lines
// ---------------------------------------------------------------------------

const SIDE_A = 0;
const SIDE_B = 1;
const FACTOR_E = 0;
const FACTOR_F = 1;

// normalize_line: bytes to signed integers in +-[1, 128], scaled to an L2 norm
// of 256 and cast to E4M3. floor(sqrt) is exact here: the argument is far below
// 2^52 and Math.sqrt is correctly rounded.
function normalizeLine(bytes) {
  let sumsq = 0;
  for (const b of bytes) sumsq += ((b & 0x7F) + 1) ** 2;
  const normScaled = Math.floor(Math.sqrt(sumsq * 1024));
  const scale = bf16Div(f32ToBf16(8192), f32ToBf16(normScaled));
  const out = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) {
    const xi = (bytes[i] >> 7 ? -1 : 1) * ((bytes[i] & 0x7F) + 1);
    out[i] = f32ToE4m3(bf16ToF32(bf16Mul(f32ToBf16(xi), scale)));
  }
  return out;
}

// sample_line: keyed BLAKE3 XOF of side | factor | line (u32 LE), zero-padded
// to one 64-byte block, then normalized. `lineKey` is noiseLineKey(seed).
function sampleLine(lineKey, side, factor, line, rank) {
  const material = Buffer.alloc(64);
  material[0] = side;
  material[1] = factor;
  material.writeUInt32LE(line, 2);
  return normalizeLine(keyedHash(lineKey, material, rank));
}

function noiseLineKey(seed) {
  return subkey(LABELS.noiseLine, seed);
}

function sampleLines(lineKey, side, factor, indices, rank) {
  const out = new Uint8Array(indices.length * rank);
  indices.forEach((line, i) => out.set(sampleLine(lineKey, side, factor, line, rank), i * rank));
  return out;
}

// sample_noise. E lines are addressed by the opened rows' global indices; F
// lines are 0..k. Only A's E lines use seedA; both F bases and B's E lines use
// seedB, so B's whole noise is fixed once B is committed.
function sampleNoise(opts) {
  const kA = noiseLineKey(opts.seedA);
  const kB = noiseLineKey(opts.seedB);
  const all = Array.from({ length: opts.k }, (_, i) => i);
  return {
    a: {
      e: sampleLines(kA, SIDE_A, FACTOR_E, opts.aRows, RANK),
      f: sampleLines(kB, SIDE_A, FACTOR_F, all, RANK),
    },
    b: {
      e: sampleLines(kB, SIDE_B, FACTOR_E, opts.bRows, RANK),
      f: sampleLines(kB, SIDE_B, FACTOR_F, all, RANK),
    },
  };
}

// ---------------------------------------------------------------------------
// prequant.rs: the FP10 planes, and the row norms taken from them
// ---------------------------------------------------------------------------

function checkPlanes(ints, scales, numRows, k) {
  if (k <= 0 || k % BLOCK_SIZE !== 0) throw new Error(`k=${k} must be a positive multiple of ${BLOCK_SIZE}`);
  if (ints.length !== numRows * k) throw new Error('int_values must be num_rows x k');
  if (scales.length !== numRows * (k / BLOCK_SIZE)) throw new Error('scales must be num_rows x (k / 8)');
  for (const s of scales) {
    if ((s & 0x7F80) === 0x7F80) throw new Error('scale is not finite');
  }
}

// open_prequant: value = int8 * its block's BF16 scale, rounded once to BF16.
function openPrequant(ints, scales, numRows, k) {
  checkPlanes(ints, scales, numRows, k);
  const blocks = k / BLOCK_SIZE;
  const out = new Uint16Array(numRows * k);
  for (let i = 0; i < numRows; i++) {
    for (let j = 0; j < k; j++) {
      const scale = bf16ToF32(scales[i * blocks + (j >> 3)]);
      out[i * k + j] = f32ToBf16(Math.fround(ints[i * k + j] * scale));
    }
  }
  return out;
}

// round_l2_to_grid: clear the two low mantissa bits, rounding ties up.
function roundL2ToGrid(l2) {
  return (l2 + 2) & 0xFFFC;
}

// l2_frame_width: 27 - ceil(log2 k).
function l2FrameWidth(k) {
  const ceilLog2 = 32 - Math.clz32(k - 1);
  if (ceilLog2 > 27) throw new Error(`k=${k} exceeds the L2 frame width`);
  return 27 - ceilLog2;
}

// sqrt_bracket_ok: does the claim (t, f) round sqrt(s * 2^(eMax-14-wl2) / k)
// correctly? Returns [lowerOk, upperOk]. BigInt keeps the comparison exact.
function sqrtBracketOk(s, eMax, wl2, k, t, f) {
  const d = eMax + 4 - wl2 - 2 * f;
  const lhs = d >= 0 ? s << BigInt(d) : s;
  const cmp = (bound) => {
    const c = BigInt(bound * bound) * BigInt(k);
    const rhs = d >= 0 ? c : c << BigInt(-d);
    return lhs < rhs ? -1 : Number(lhs > rhs);
  };
  const even = t % 2 === 0;
  const hi = cmp(4 * t + 2);
  const hiOk = hi < 0 || (hi === 0 && even);
  if (t === 0) return [true, hiOk];
  const lo = cmp(4 * t - 2 + (t === 128 && f >= 2 ? 1 : 0));
  return [lo > 0 || (lo === 0 && even), hiOk];
}

// The double-precision guess rne_sqrt_hat starts from. It only has to land
// near the answer; the exact bracket below decides.
function sqrtSeed(s, eMax, wl2, k) {
  const log2y = (Math.log2(Number(s)) - Math.log2(k) + (eMax - 14 - wl2)) / 2;
  let f = Math.floor(log2y);
  if (f < 1) return [Math.min(Math.max(Math.round(2 ** log2y * 64), 0), 128), 1];
  let t = Math.round(2 ** (log2y - f) * 128);
  if (t >= 256) {
    t = 128;
    f += 1;
  }
  return [Math.max(t, 128), f];
}

// rne_sqrt_hat: the BF16 code of the correctly rounded square root, found by
// stepping the seed until the exact bracket holds. `seed` is a test hook.
function rneSqrtHat(s, eMax, wl2, k, seed) {
  let [t, f] = seed || sqrtSeed(s, eMax, wl2, k);
  for (let i = 0; i < 1024; i++) {
    const [loOk, hiOk] = sqrtBracketOk(s, eMax, wl2, k, t, f);
    if (loOk && hiOk) return t >= 128 ? (f << 7) + (t - 128) : t;
    if (!loOk) {
      t -= 1;
      if (t === 127 && f >= 2) {
        t = 255;
        f -= 1;
      }
    } else {
      t += 1;
      if (t === 256) {
        t = 128;
        f += 1;
      }
    }
  }
  throw new Error('sqrt bracket search did not converge');
}

// |scale| = M * 2^(E - 134); returns [M, 2E].
function decodeScale(code) {
  const expField = (code & 0x7FFF) >> 7;
  return [(code & 0x7F) + (expField === 0 ? 0 : 128), 2 * Math.max(expField, 1)];
}

// row_norms: (l2, linf) of one row as BF16 codes, from exact integers.
function rowNorms(ints, scales, wl2) {
  const blocks = [];
  let eMax = 0;
  let linf = 0;
  for (let b = 0; b < scales.length; b++) {
    let amax = 0;
    let sumsq = 0;
    for (let j = b * BLOCK_SIZE; j < (b + 1) * BLOCK_SIZE; j++) {
      amax = Math.max(amax, Math.abs(ints[j]));
      sumsq += ints[j] * ints[j];
    }
    const [m, doubledExp] = decodeScale(scales[b]);
    if (amax !== 0 && m !== 0) eMax = Math.max(eMax, doubledExp);
    linf = Math.max(linf, amax * bf16ToF32(scales[b] & 0x7FFF));
    blocks.push([m * m * sumsq, doubledExp]);
  }
  let s = 0n;
  for (const [p, doubledExp] of blocks) {
    s += (BigInt(p) << BigInt(wl2)) >> BigInt(Math.max(eMax - doubledExp, 0));
  }
  const l2 = s === 0n ? 0 : roundL2ToGrid(rneSqrtHat(s, eMax, wl2, ints.length));
  if (l2 >= 0x7F80) throw new Error('l2 snaps into the bf16 infinity code');
  return [l2, f32ToBf16(linf)];
}

// exact_norms: per-row [l2, linf] straight from the committed planes.
function exactNorms(ints, scales, numRows, k) {
  checkPlanes(ints, scales, numRows, k);
  const wl2 = l2FrameWidth(k);
  const blocks = k / BLOCK_SIZE;
  const out = [];
  for (let i = 0; i < numRows; i++) {
    out.push(rowNorms(ints.subarray(i * k, (i + 1) * k), scales.subarray(i * blocks, (i + 1) * blocks), wl2));
  }
  return out;
}

// ---------------------------------------------------------------------------
// utils.rs: the FP8 matmul, as each chip adds it up
// ---------------------------------------------------------------------------

// Every E4M3 product, exact and unnormalized: value = sig * 2^(exp - 23).
// A zero product has sig 0 and never sets a group's anchor.
const PRODUCT_SIG = new Uint32Array(65536);
const PRODUCT_EXP = new Int8Array(65536);
const PRODUCT_NEG = new Uint8Array(65536);
for (let a = 0; a < 256; a++) {
  for (let b = 0; b < 256; b++) {
    const expA = (a >> 3) & 0x0F;
    const expB = (b >> 3) & 0x0F;
    const sigA = expA !== 0 ? (a & 7) | 8 : a & 7;
    const sigB = expB !== 0 ? (b & 7) | 8 : b & 7;
    const i = (a << 8) | b;
    PRODUCT_SIG[i] = (sigA * sigB) << 17;
    PRODUCT_EXP[i] = Math.max(expA, 1) + Math.max(expB, 1) - 14;
    PRODUCT_NEG[i] = (a ^ b) >> 7;
  }
}

// One group's terms, reused across calls: the carry in slot 0, then products.
const TERM_SIG = new Uint32Array(MMA_GROUP_PRODUCTS + 1);
const TERM_EXP = new Int32Array(MMA_GROUP_PRODUCTS + 1);
const TERM_NEG = new Uint8Array(MMA_GROUP_PRODUCTS + 1);

// GFloat::from(f32): the carry as a term. Subnormals keep their true scale.
function setCarry(value) {
  const bits = f32Bits(value);
  const exp = (bits >>> 23) & 0xFF;
  TERM_NEG[0] = bits >>> 31;
  TERM_EXP[0] = Math.max(exp, 1) - 127;
  TERM_SIG[0] = (bits & 0x7FFFFF) | (exp !== 0 ? 0x800000 : 0);
}

function setProduct(slot, a, b) {
  const i = (a << 8) | b;
  TERM_SIG[slot] = PRODUCT_SIG[i];
  TERM_EXP[slot] = PRODUCT_EXP[i];
  TERM_NEG[slot] = PRODUCT_NEG[i];
}

// windowed_group_sum: what one MMA instruction computes. The terms are aligned
// to the largest exponent among them, each is truncated toward zero to the
// window's width, they are added exactly, and the sum is truncated to FP32.
// Nothing rounds to nearest.
//
// Left out, because neither can happen for a finite carry and E4M3 products:
// the Rust `as u32` cast of the sum (at most 33 terms below 2^27), and the
// finite check on the result.
function windowedGroupSum(count, width) {
  let maxExp = null;
  for (let i = 0; i < count; i++) {
    if (TERM_SIG[i] !== 0 && (maxExp === null || TERM_EXP[i] > maxExp)) maxExp = TERM_EXP[i];
  }
  if (maxExp === null) return 0;
  const narrow = width < 24;
  const diff = Math.abs(width - 24);
  let acc = 0;
  for (let i = 0; i < count; i++) {
    const shift = maxExp - TERM_EXP[i];
    if (TERM_SIG[i] === 0 || shift >= 32) continue;
    const scaled = narrow ? TERM_SIG[i] >>> diff : TERM_SIG[i] * 2 ** diff;
    const aligned = Math.floor(scaled / 2 ** shift);
    acc += TERM_NEG[i] ? -aligned : aligned;
  }
  if (acc === 0) return 0;
  const negative = acc < 0;
  let sig = Math.abs(acc);
  const bitLen = 32 - Math.clz32(sig);
  let exponent = maxExp + bitLen - width;
  sig = bitLen > width ? sig >>> (bitLen - width) : sig * 2 ** (width - bitLen);
  if (exponent < -126) {
    sig = Math.floor(sig / 2 ** (-126 - exponent));
    exponent = -126;
  }
  sig = narrow ? sig * 2 ** diff : sig >>> diff;
  if (sig === 0) return negative ? -0 : 0;
  const expField = exponent + 127 - (sig & 0x800000 ? 0 : 1);
  return f32FromBits(((negative ? 0x80000000 : 0) + expField * 0x800000 + (sig & 0x7FFFFF)) >>> 0);
}

// windowedGroupSum over explicit terms [{ neg, exp, sig }], value sig * 2^(exp-23).
// The matmuls never need this; it lets the tests reach the underflow cases
// Pearl's own tests reach by building terms directly.
function groupSumOf(terms, width) {
  terms.forEach((t, i) => {
    TERM_NEG[i] = t.neg ? 1 : 0;
    TERM_EXP[i] = t.exp;
    TERM_SIG[i] = t.sig;
  });
  return windowedGroupSum(terms.length, width);
}

// matmul_fp8_windowed at Blackwell's width. `a` is m x k and `b` is n x k, both
// row-major E4M3, so cell (i, j) is row i of a dotted with row j of b. Each
// cell takes k in ascending groups of 32 products, each group added to the
// running FP32 value by windowedGroupSum. `acc`, if given, seeds the cells.
// With `partials`, the running value after every group is returned as well.
function matmulB200(a, b, acc, m, n, k, partials) {
  const out = new Float32Array(m * n);
  const steps = [];
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      const initial = acc ? acc[i * n + j] : 0;
      if (!Number.isFinite(initial)) throw new Error('f32 value is not finite');
      const cell = [];
      setCarry(initial);
      let used = 1;
      for (let u = 0; u < k; u++) {
        setProduct(used++, a[i * k + u], b[j * k + u]);
        if (used === MMA_GROUP_PRODUCTS + 1) {
          const collapsed = windowedGroupSum(used, 26);
          cell.push(collapsed);
          setCarry(collapsed);
          used = 1;
        }
      }
      // The last, possibly short, group. On a single value this is the
      // identity, except that it turns -0 into +0.
      const value = windowedGroupSum(used, 26);
      if (used > 1) cell.push(value);
      out[i * n + j] = value;
      steps.push(cell);
    }
  }
  return partials ? { out, partials: steps } : out;
}

// H100::matmul_fp8_replay, output only. Hopper keeps 13 fractional bits per
// group, restarts a local accumulator at +0 every 128 products, and adds each
// window to the total with one ordinary FP32 round-to-nearest add.
function matmulH100(a, b, acc, m, n, k) {
  if (acc) throw new Error('H100 replay does not accept a carry-in');
  if (k <= 0 || k % MMA_GROUP_PRODUCTS !== 0) throw new Error('H100 replay requires positive k divisible by 32');
  const out = new Float32Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let total = 0;
      for (let w = 0; w < k; w += H100_WINDOW_PRODUCTS) {
        let local = 0;
        for (let g = w; g < Math.min(w + H100_WINDOW_PRODUCTS, k); g += MMA_GROUP_PRODUCTS) {
          setCarry(local);
          for (let u = 0; u < MMA_GROUP_PRODUCTS; u++) setProduct(u + 1, a[i * k + g + u], b[j * k + g + u]);
          local = windowedGroupSum(MMA_GROUP_PRODUCTS + 1, 14);
        }
        total = Math.fround(total + local) + 0; // + 0 turns -0 into +0
      }
      out[i * n + j] = total;
    }
  }
  return out;
}

// Device::matmul_fp8.
function matmulFp8(device, a, b, acc, m, n, k) {
  deviceParams(device);
  return device === DEVICE_B200 ? matmulB200(a, b, acc, m, n, k, false) : matmulH100(a, b, acc, m, n, k);
}

// ---------------------------------------------------------------------------
// quantization.rs: row scales, the noised FP8 operand, the downscale rewrite
// ---------------------------------------------------------------------------

const NORM_FLOOR_BF16 = 0x2F80; // 2^-32

// derive_row_scales:
//   alpha = 448 / (linf + delta*sqrt(r) * l2)
//   beta  = alpha * l2 * delta*sqrt(r) / 256^2
// every step a BF16 operation.
function deriveRowScales(device, l2, linf, r) {
  const { delta } = deviceParams(device);
  const deltaR = f32ToBf16(Math.fround(delta * Math.sqrt(r)));
  const deltaOverStd = f32ToBf16(Math.fround(delta * Math.sqrt(r) / 65536));
  const alpha = bf16Div(f32ToBf16(MAX_E4M3), bf16Fma(deltaR, l2, linf));
  return [alpha, bf16Mul(bf16Mul(alpha, l2), deltaOverStd)];
}

// Fp8E4M3Quant::downscale. After the cast to E4M3, a code whose magnitude is
// below e4m3(g * rms) is rewritten to the row's "safe" exponent with its low
// mantissa bit set, if the coin of its 4-column group passes. The coin is bits
// 14..8 of the group's first FP32 value, below 112.
function downscale(mixed, scaledRms, device) {
  const { g, gSmall } = deviceParams(device);
  const k = mixed.length / scaledRms.length;
  const out = new Uint8Array(mixed.length);
  for (let row = 0; row < scaledRms.length; row++) {
    const rms = scaledRms[row];
    if (!(Number.isFinite(rms) && rms >= 4)) throw new Error('downscaling requires finite scaled_rms >= 4');
    const safeExp = ((f32Bits(rms) >>> 23) & 0xFF) - 127 - 9 + 7;
    const threshold = f32ToE4m3(Math.fround((rms >= 8 ? g : gSmall) * rms));
    for (let j = 0; j < k; j++) {
      const i = row * k + j;
      const code = f32ToE4m3(mixed[i]);
      const coin = (f32Bits(mixed[i - (j % 4)]) >>> 8) & 0x7F;
      if ((code & 0x7F) < threshold && coin < 112) {
        const safe = ((code & 0x80) | (safeExp << 3) | (code & 7) | 1) & 0xFF;
        out[i] = f32ToE4m3(e4m3ToF32(safe));
      } else {
        out[i] = code;
      }
    }
  }
  return out;
}

// noisy_quantize: X' = Q(alpha*X + beta*(E @ F)) for a stack of opened rows.
// `noise.e` is rows x r and `noise.f` is k x r. E @ F runs on the device's
// matmul; the mix is one FP32 fused multiply-add per element.
// The planes must have been through exactNorms (which validates them).
function noisyQuantize(device, ints, scales, noise, norms) {
  const numRows = norms.length;
  const k = ints.length / numRows;
  const r = noise.e.length / numRows;
  const blocks = k / BLOCK_SIZE;
  const product = matmulFp8(device, noise.e, noise.f, null, numRows, k, r);
  const mixed = new Float32Array(numRows * k);
  const alpha = new Uint16Array(numRows);
  const beta = new Uint16Array(numRows);
  const l2 = new Uint16Array(numRows);
  const scaledRms = new Float32Array(numRows);
  for (let i = 0; i < numRows; i++) {
    l2[i] = bf16Max(norms[i][0], NORM_FLOOR_BF16);
    [alpha[i], beta[i]] = deriveRowScales(device, l2[i], bf16Max(norms[i][1], NORM_FLOOR_BF16), r);
    const alphaF = bf16ToF32(alpha[i]);
    const betaF = bf16ToF32(beta[i]);
    for (let j = 0; j < k; j++) {
      const scale = bf16ToF32(scales[i * blocks + (j >> 3)]);
      const signal = Math.fround(Math.fround(alphaF * scale) * ints[i * k + j]);
      mixed[i * k + j] = fp32Fma(betaF, product[i * k + j], signal);
    }
    scaledRms[i] = alphaF * bf16ToF32(l2[i]);
  }
  return { noised: downscale(mixed, scaledRms, device), alpha, beta, l2 };
}

// ---------------------------------------------------------------------------
// layout.rs and the fold
// ---------------------------------------------------------------------------

// check_lottery_layout: h >= 4, w >= 16, 256 <= h*w <= 2048, 16 | h*w.
function checkLotteryLayout(h, w) {
  const cells = h * w;
  if (h < 4 || w < 16) throw new Error(`tile must be at least 4 x 16 || h=${h} w=${w}`);
  if (cells > 2048 || cells < 256 || cells % JACKPOT_ENTRIES !== 0) {
    throw new Error(`h*w must be a multiple of 16 in 256..=2048 || h=${h} w=${w}`);
  }
}

// xor_fold_extract over lane_assignment: the row-major tile is cut into sixteen
// equal consecutive runs, each folded into one u32 as
// lane = rotl32(lane * 0x9E3779B1 + bits, 13). The sixteen words, little
// endian, are the 64-byte jackpot message.
function foldTile(tile) {
  const per = tile.length / JACKPOT_ENTRIES;
  const out = Buffer.alloc(4 * JACKPOT_ENTRIES);
  for (let lane = 0; lane < JACKPOT_ENTRIES; lane++) {
    let v = 0;
    for (let i = lane * per; i < (lane + 1) * per; i++) {
      v = (Math.imul(v, FOLD_MUL) + f32Bits(tile[i])) >>> 0;
      v = ((v << 13) | (v >>> 19)) >>> 0;
    }
    out.writeUInt32LE(v, lane * 4);
  }
  return out;
}

// ---------------------------------------------------------------------------
// unpredictability.rs: the integer scores of the jackpot policy
// ---------------------------------------------------------------------------

const SKIP_THRESHOLD_OFFSET = 45952;

// ceil(2^(15 + i/64)).
const LOG_BOUNDS = [
  32768, 33125, 33486, 33851, 34219, 34592, 34969, 35349, 35734, 36123, 36517, 36914, 37316,
  37723, 38133, 38549, 38968, 39393, 39822, 40255, 40694, 41137, 41585, 42038, 42495, 42958,
  43426, 43899, 44377, 44860, 45348, 45842, 46341, 46846, 47356, 47872, 48393, 48920, 49453,
  49991, 50536, 51086, 51642, 52205, 52773, 53348, 53929, 54516, 55109, 55710, 56316, 56929,
  57549, 58176, 58810, 59450, 60097, 60752, 61413, 62082, 62758, 63441, 64132, 64831,
];

// log2_fixed: floor(64 * log2 n) for n in [1, 2^16).
function log2Fixed(n) {
  const w = 31 - Math.clz32(n);
  const normalized = n << (15 - w);
  let fraction = 63;
  while (LOG_BOUNDS[fraction] > normalized) fraction -= 1;
  return 64 * w + fraction;
}

// bf16_e_star_m: a finite BF16 code as [E, M], value = +-M * 2^(E - 134).
function bf16EStarM(code) {
  const exp = (code >> 7) & 0xFF;
  if (exp === 255) throw new Error('non-finite bf16 code is banned');
  return exp === 0 ? [1, code & 0x7F] : [exp, 128 + (code & 0x7F)];
}

// lambda: about 64 * (log2((alpha*x)^2 + sigma^2) + 508), in exact integers,
// where sigma = delta * alpha * l2f is the row's noise deviation.
function lambda(device, alpha, x, l2f) {
  const { delta } = deviceParams(device);
  const [alphaE, alphaM] = bf16EStarM(alpha);
  const [xE, xM] = bf16EStarM(x);
  const [l2E, l2M] = bf16EStarM(l2f);
  if (alphaM < 128 || l2M < 128) throw new Error('alpha and l2f must be normal');
  const product = alphaM * xM;
  const width = 32 - Math.clz32(product);
  const xNorm = xM === 0 ? 0 : product << (16 - width);
  const xEnc = xM === 0 ? 0 : alphaE + xE - 1 + width;
  const sigmaProduct = alphaM * l2M;
  const wide = sigmaProduct >= 1 << 15 ? 1 : 0;
  const sigmaNorm = sigmaProduct << (1 - wide);
  const sigmaEnc = alphaE + l2E + wide + 14 + Math.log2(delta);
  // lambda_witness
  const xDominates = xEnc >= sigmaEnc && xNorm !== 0;
  const gap = Math.abs(xEnc - sigmaEnc);
  const big = xDominates ? xNorm : sigmaNorm;
  const small = xDominates ? sigmaNorm : xNorm;
  const quotient = gap >= 16 ? 0 : Math.floor(Math.floor(small * small / 2 ** gap) / 2 ** gap);
  const top = Math.floor((big * big + quotient) / 131072);
  return 128 * Math.max(xEnc, sigmaEnc) - 2624 + log2Fixed(top);
}

// skip: can this summand be dropped without changing the cell?
function skip(lambdaA, lambdaB, eGrid) {
  return eGrid !== 0 && lambdaA + lambdaB < 128 * eGrid + SKIP_THRESHOLD_OFFSET;
}

// ---------------------------------------------------------------------------
// jackpot_policy.rs: which tiles may win (Blackwell)
// ---------------------------------------------------------------------------

const POLICY = { tauIdle: 8, epsIdle: 1 / 64, rmsMinA: 4, rmsMinB: 8 };
const COVERAGE_SCALE = 65536;

// gauss_kernel: floor(2^16 * exp(-(2^(d/64) - 1) / 2)). Pearl's own test shows
// no value is within 1e-3 of an integer except d = 0, so any libm agrees.
function gaussKernel(d) {
  return Math.floor(COVERAGE_SCALE * Math.exp(-0.5 * (2 ** (d / 64) - 1)));
}

// gauss_row_constants: check 4's per-row [coefficient, kernel cap].
function gaussRowConstants(alpha, l2, device) {
  const { delta, g, gSmall } = deviceParams(device);
  const scaledRms = bf16ToF32(alpha) * bf16ToF32(l2); // exact
  if (!(scaledRms >= 4 && scaledRms < 256)) throw new Error('noise coverage requires signal RMS in [4, 256)');
  const code = e4m3ToF32(f32ToE4m3(Math.fround((scaledRms >= 8 ? g : gSmall) * scaledRms)));
  const noiseRms = delta * scaledRms;
  if (noiseRms >= 4) return [Math.floor(COVERAGE_SCALE * 0.65625 * code / noiseRms), COVERAGE_SCALE];
  const cap = Math.floor(COVERAGE_SCALE * (0.0431 + 0.2738 * (code - 0.4)));
  return [COVERAGE_SCALE / 8, 8 * cap];
}

// Check 1: at most 1/64 of a side's entries may be "dead", meaning
// |x| >= tau * delta * l2 of its row.
function livenessOk(side, device, policy) {
  const { delta } = deviceParams(device);
  const k = side.clean.length / side.l2.length;
  let dead = 0;
  side.clean.forEach((x, i) => {
    const bound = policy.tauIdle * delta * bf16ToF32(side.l2[Math.floor(i / k)]);
    if (Math.abs(bf16ToF32(x)) >= bound) dead += 1;
  });
  return dead <= policy.epsIdle * side.clean.length;
}

// Check 2: every row's noise deviation delta*alpha*l2 reaches the floor.
function noiseFloorOk(side, device, rmsMin) {
  return side.alpha.every((alpha, i) => bf16ToF32(alpha) * bf16ToF32(side.l2[i]) >= rmsMin);
}

function lambdas(side, device, k) {
  return Array.from(side.clean, (x, i) => {
    const row = Math.floor(i / k);
    return lambda(device, side.alpha[row], x, side.l2[row]);
  });
}

// Check 3 (census_ok, B200): replay A' @ B'^T, and count the summands that sit
// below the accumulator's precision at their cell's magnitude. More than a
// twentieth of them and the tile is rejected. Returns the FP32 tile or null.
function census(a, b, k, h, w) {
  const { out, partials } = matmulB200(a.noised, b.noised, null, h, w, k, true);
  const lamA = lambdas(a, DEVICE_B200, k);
  const lamB = lambdas(b, DEVICE_B200, k);
  let skippable = 0;
  for (let i = 0; i < h; i++) {
    for (let j = 0; j < w; j++) {
      let magnitude = 0;
      for (let u = 0; u < k; u++) {
        magnitude = Math.max(magnitude, Math.abs(e4m3ToF32(a.noised[i * k + u]) * e4m3ToF32(b.noised[j * k + u])));
      }
      for (const p of partials[i * w + j]) magnitude = Math.max(magnitude, Math.abs(p));
      // On B200 the window and cell magnitudes are the same number, so
      // E_GRID = e(M) - 18 + 157.
      const eGrid = magnitude === 0 ? 0 : f64Exponent(magnitude) + 139;
      for (let u = 0; u < k; u++) {
        if (skip(lamA[i * k + u], lamB[j * k + u], eGrid)) skippable += 1;
      }
    }
  }
  return skippable > Math.floor(k * h * w / 20) ? null : out;
}

// Check 4 (row_noise_coverage_ok): every row's mean noise coverage is at least
// 1/25, in the policy's fixed-point units.
function rowNoiseCoverageOk(side, device, k) {
  const threshold = ((BigInt(k) << 32n) + 24n) / 25n;
  for (let row = 0; row < side.alpha.length; row++) {
    const [coefficient, cap] = gaussRowConstants(side.alpha[row], side.l2[row], device);
    const base = lambda(device, side.alpha[row], 0, side.l2[row]);
    let sum = 0;
    for (let u = 0; u < k; u++) {
      const d = lambda(device, side.alpha[row], side.clean[row * k + u], side.l2[row]) - base;
      sum += Math.min(gaussKernel(d), cap);
    }
    if (BigInt(coefficient) * BigInt(sum) < threshold) return false;
  }
  return true;
}

// JackpotPolicy::run_checks. Each side is { clean, noised, alpha, l2 }.
// Returns the bit-exact FP32 tile, or null when the tile is not admissible.
// `policy` overrides thresholds in tests, as Pearl's own tests do.
function runChecks(device, a, b, k, policy = POLICY) {
  if (device !== DEVICE_B200) throw new Error('the jackpot policy is ported for B200 only');
  const h = a.alpha.length;
  const w = b.alpha.length;
  if (!livenessOk(a, device, policy) || !livenessOk(b, device, policy)) return null;
  if (!noiseFloorOk(a, device, policy.rmsMinA) || !noiseFloorOk(b, device, policy.rmsMinB)) return null;
  const tile = census(a, b, k, h, w);
  if (!tile) return null;
  if (!rowNoiseCoverageOk(a, device, k) || !rowNoiseCoverageOk(b, device, k)) return null;
  return tile;
}

// ---------------------------------------------------------------------------
// verify.rs: one tile, end to end
// ---------------------------------------------------------------------------

// open_and_noisy_quantize: one operand's opened rows to the policy's view.
function buildSide(device, ints, scales, numRows, k, noise) {
  const clean = openPrequant(ints, scales, numRows, k);
  const norms = exactNorms(ints, scales, numRows, k);
  return { clean, ...noisyQuantize(device, ints, scales, noise, norms) };
}

// nbits_to_difficulty: Bitcoin's compact target.
function nbitsToTarget(nbits) {
  const exponent = nbits >>> 24;
  const mantissa = nbits & 0x00FFFFFF;
  if (mantissa === 0 || exponent === 0 || (mantissa & 0x00800000) !== 0) return 0n;
  const m = BigInt(mantissa);
  return exponent <= 3 ? m >> BigInt(8 * (3 - exponent)) : m << BigInt(8 * (exponent - 3));
}

// check_jackpot_difficulty: ticket <= target * h * w * k, read little endian
// and capped at 2^256 - 1. Pools pass their own nbits for a share.
function meetsDifficulty(ticket, nbits, h, w, k) {
  const bound = nbitsToTarget(nbits) * BigInt(Math.min(h * w * k, 0xFFFFFFFF));
  const max = (1n << 256n) - 1n;
  const value = BigInt(`0x${Buffer.from(ticket).reverse().toString('hex')}`);
  return value <= (bound > max ? max : bound);
}

// verify_plain_proof after the Merkle openings: from the opened rows of A and B
// and the two noise seeds to the ticket. `aRows` and `bRows` are the global row
// indices of the tile, h and w of them, consecutive, starting at a multiple of
// the tile size. Returns null when the jackpot policy rejects the tile.
function evaluateTile(opts) {
  const { device, k, aRows, bRows } = opts;
  const h = aRows.length;
  const w = bRows.length;
  checkLotteryLayout(h, w);
  if (k < MIN_K || k > MAX_K || k % 32 !== 0) throw new Error(`k must be a multiple of 32 in 1024..=65536 || k=${k}`);
  const noise = sampleNoise({ k, seedA: opts.seedA, seedB: opts.seedB, aRows, bRows });
  const a = buildSide(device, opts.aInts, opts.aScales, h, k, noise.a);
  const b = buildSide(device, opts.bInts, opts.bScales, w, k, noise.b);
  const tile = runChecks(device, a, b, k);
  if (!tile) return null;
  const message = foldTile(tile);
  return { a, b, tile, message, ticket: jackpotTicket(opts.seedA, message) };
}

module.exports = {
  enabled,
  DEVICE_H100, DEVICE_B200, RANK, BLOCK_SIZE, MMA_GROUP_PRODUCTS, MAX_E4M3, MIN_K, MAX_K,
  SIDE_A, SIDE_B, FACTOR_E, FACTOR_F, LABELS, POLICY,
  f32Bits, f32FromBits, f32Step,
  e4m3ToF32, f32ToE4m3, f32ToBf16, bf16ToF32,
  bf16Mul, bf16Div, bf16Max, bf16Fma, fp32Fma,
  subkey, hashLabelled, keyA, keyB, operandDigest, encodePA, encodePB, noiseSeeds, jackpotTicket,
  normalizeLine, sampleLine, noiseLineKey, sampleNoise,
  openPrequant, roundL2ToGrid, l2FrameWidth, sqrtSeed, rneSqrtHat, exactNorms,
  groupSumOf, matmulFp8, matmulB200,
  deriveRowScales, noisyQuantize,
  checkLotteryLayout, foldTile,
  log2Fixed, lambda, skip,
  gaussKernel, gaussRowConstants, rowNoiseCoverageOk, runChecks,
  buildSide, nbitsToTarget, meetsDifficulty, evaluateTile,
};
