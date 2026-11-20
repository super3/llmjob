// Pearl PIP-3 (FP8) mining core for Blackwell RTX cards (sm_120: RTX 50, RTX PRO 6000).
//
// The fork replaces the INT7 product with C' = A' . B'^T over FP8 E4M3 operands and an FP32
// accumulator, and the verifier repeats the declared chip's arithmetic bit for bit. On sm_120
// that arithmetic is one instruction, mma.sync m16n8k32 with E4M3 operands: it adds 32 exact
// products to the running FP32 value, every term cut toward zero 25 bits below the group's
// largest exponent. Pearl commits these cards as device B200. So this core never models the
// sum; it runs the instruction, one accumulator a cell, k ascending, and the CPU reference
// (earn/src/shared/miner/fp8.js) is what it is checked against (probes/fp8-verify.js).
//
// The kernels:
//   fp8_noise_kernel   A' = Q(alpha.A + beta.(E @ F)) for a stack of rows: the rank-32 noise
//                      product on the same instruction, the FP32 mix, the cast to E4M3 and
//                      the downscale rewrite (quantization.rs).
//   fp8_search_kernel  the product, 64 rows of A by 64 rows of B a warp, then each 256-cell
//                      tile folded to its 64-byte message, hashed, and compared with the bound.
//   fp8_lines_kernel   the noise lines E and F from a seed (keyed BLAKE3, one BF16 scale).
//   fp8_leaf_kernel, fp8_parent_kernel   the keyed BLAKE3 Merkle tree of a committed plane.
//
// The host half below owns device memory and is the C interface pearl_fp8_core.cc binds.
// The row norms, alpha and beta need exact wide integers and BF16 steps, so the JS host
// (fp8Host.js) computes them and hands them in.
//
// See earn/tuning/pip3-fp8.md for the spec.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <vector>

// A warp's share: FP8_ATOMS 16-row atoms of A by FP8_NB 8-column atoms of B, 128 accumulators
// a lane either way. 4 x 8 beats 2 x 16 on an RTX 5080: each load of B feeds more
// instructions.
//
// A block is FP8_WA x FP8_WB warps laid out as a patch of the product: warps in a column of
// the patch read the same rows of A and warps in a row of it the same rows of B. With
// FP8_SMEM the block copies those rows into shared memory once a plane; without it each
// warp loads its own and relies on the SM's L1 cache to share them.
#ifndef FP8_WA
#define FP8_WA 2
#endif
#ifndef FP8_WB
#define FP8_WB 2
#endif
#ifndef FP8_ATOMS
#define FP8_ATOMS 4
#endif
#ifndef FP8_NB
#define FP8_NB 8
#endif
#define FP8_WARPS (FP8_WA * FP8_WB)
#define FP8_WARP_ROWS (16 * FP8_ATOMS)          // rows of A a warp
#define FP8_WARP_COLS (8 * FP8_NB)              // rows of B a warp
#define FP8_ROWS_A (FP8_WARP_ROWS * FP8_WA)     // rows of A a block
#define FP8_ROWS_B (FP8_WARP_COLS * FP8_WB)     // rows of B a block
#define FP8_THREADS (32 * FP8_WARPS)
#define FP8_DUMP_A 64                  // the window of cells the verify tool reads
#define FP8_DUMP_B 128
#if (FP8_WARP_ROWS * FP8_WARP_COLS) != 4096 || 64 % FP8_WARP_ROWS || 128 % FP8_WARP_COLS
#error "a warp must hold 4096 cells (16 tiles) and divide 64 x 128"
#endif

// FP8_SMEM 1: a block stages its rows of A' and B' through shared memory (cp.async, FP8_STAGES
// deep), so each byte comes from the L2 cache once a block. 0: every warp reads its operands
// straight from global memory.
#ifndef FP8_SMEM
#define FP8_SMEM 1
#endif
#ifndef FP8_STAGES
#define FP8_STAGES 3
#endif

// How A' and B' are stored. k is cut into 64-byte "planes" (two 32-byte steps), and plane p
// of every row comes before plane p + 1 of any row, so the rows a block needs for one plane
// are one contiguous run of memory. k is padded to a multiple of 64 with zero codes.
//
// One instruction reads, in lane t of a row, bytes 4t..4t+3 and 16+4t..16+4t+3 of a step.
// B' keeps each row's 64 bytes of a plane in "lane order": lane t's 16 bytes sit together at
// 16t, the first step's 8 then the second's, so one 16-byte load feeds a lane for two steps.
__host__ __device__ __forceinline__ uint32_t fp8_pos(uint32_t j) {
  const uint32_t r = j & 63u, x = r & 31u;
  return (j & ~63u) + 16u * ((x & 15u) >> 2) + 8u * (r >> 5) + 4u * (x >> 4) + (x & 3u);
}

// A' is kept "atom-major": the 16 rows of an atom share 1024 bytes a plane, 512 a step, and
// lane l = 4g + t holds 16 bytes at 16l in the order the instruction takes them, a0 a1 a2 a3
// (row g then g + 8, at 4t, then both at 16 + 4t). One 16-byte load fills the four registers,
// so nothing has to be moved before the instruction.
__host__ __device__ __forceinline__ size_t fp8_at(int side, uint32_t rows, uint32_t r, uint32_t j) {
  const size_t plane = (size_t)(j >> 6) * rows * 64;
  const uint32_t x = j & 63u;
  if (side) return plane + (size_t)r * 64 + fp8_pos(x);
  const uint32_t rr = r & 15u, y = x & 31u;
  return plane + (size_t)(r >> 4) * 1024 + 512u * (x >> 5) + 16u * (4u * (rr & 7u) + ((y & 15u) >> 2)) +
         4u * (2u * (y >> 4) + (rr >> 3)) + (y & 3u);
}

// ---------------------------------------------------------------------------------------
// The instruction
// ---------------------------------------------------------------------------------------

// One group of 32 products for a 16 x 8 block of cells, added to the running values c.
// The block-scaled form with every scale 1.0 (ue8m0 code 127) computes the same atom at
// twice the rate of the plain form (Pearl's _kernel_sm120.py); FP8_MMA_PLAIN picks the plain
// one, and fp8-verify.js is run against both.
__device__ __forceinline__ void fp8_mma(float &c0, float &c1, float &c2, float &c3,
                                        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
#ifdef FP8_MMA_PLAIN
  asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(c0), "=f"(c1), "=f"(c2), "=f"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#else
  const uint32_t one = 127u;
  const uint16_t zero = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, "
      "{%14}, {%15, %16}, {%17}, {%18, %19};\n"
      : "=f"(c0), "=f"(c1), "=f"(c2), "=f"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
        "r"(one), "h"(zero), "h"(zero), "r"(one), "h"(zero), "h"(zero));
#endif
}

// Where a lane's registers sit in the 16 x 8 x 32 atom (g = lane >> 2, t = lane & 3):
//   a0 row g,   k 4t..4t+3      a1 row g+8, k 4t..4t+3
//   a2 row g,   k 16+4t..       a3 row g+8, k 16+4t..
//   b0 col g,   k 4t..4t+3      b1 col g,   k 16+4t..
//   c0 (g, 2t)  c1 (g, 2t+1)    c2 (g+8, 2t)  c3 (g+8, 2t+1)

// ---------------------------------------------------------------------------------------
// The noised operand
// ---------------------------------------------------------------------------------------

// f32_to_fp8_e4m3 (dtype.rs): round to nearest even, saturating at +-448.
__device__ __forceinline__ uint32_t fp8_cast(float x) {
  const uint32_t bits = __float_as_uint(x);
  const uint32_t sign = (bits >> 31) << 7;
  const uint32_t mag = bits & 0x7FFFFFFFu;
  const int e = (int)(mag >> 23) - 127;
  if (e < -6) return sign | (uint32_t)__float2int_rn(__uint_as_float(mag) * 512.0f);
  uint32_t exp = (uint32_t)(e + 7);
  const uint32_t fm = mag & 0x7FFFFFu;
  uint32_t mant = fm >> 20;
  const uint32_t rem = fm & 0xFFFFFu;
  if (rem > 0x80000u || (rem == 0x80000u && (mant & 1u))) {
    mant += 1;
    if (mant == 8) {
      mant = 0;
      exp += 1;
    }
  }
  if (exp > 15 || (exp == 15 && mant >= 7)) return sign | 0x7Eu;
  return sign | (exp << 3) | mant;
}

// Fp8E4M3Quant::downscale for one value. `first` is the FP32 value that opens its
// 4-column group; `rewrite` is the row's threshold code (bits 0..7) and safe exponent,
// already shifted (bits 8..15).
__device__ __forceinline__ uint32_t fp8_code(float mixed, float first, uint32_t rewrite) {
  uint32_t code = fp8_cast(mixed);
  const uint32_t coin = (__float_as_uint(first) >> 8) & 0x7Fu;
  if ((code & 0x7Fu) < (rewrite & 0xFFu) && coin < 112u) {
    code = (code & 0x80u) | ((rewrite >> 8) & 0xFFu) | (code & 7u) | 1u;
  }
  return code;
}

// One warp a block: 16 rows by 256 columns.
// ints is rows x k int8; e is rows x 32 and f is k x 32 E4M3 codes; out is rows x k
// codes laid out as fp8_at has them for `side` (`rows` rows in all).
// signal is rows x k/8: fl(alpha * scale) of each 8-value block, so the clean term is one
// more FP32 multiply by the int8, as quantization.rs has it.
extern "C" __global__ void fp8_noise_kernel(const int8_t *ints, const uint32_t *e,
                                            const uint32_t *f, const float *signal,
                                            const float *beta, const uint32_t *rewrite,
                                            uint8_t *out, uint32_t k, int side, uint32_t rows) {
  const uint32_t lane = threadIdx.x;
  const uint32_t g = lane >> 2, t = lane & 3;
  const uint32_t r0 = blockIdx.x * 16 + g, r1 = r0 + 8;
  const uint32_t a0 = e[r0 * 8 + t], a1 = e[r1 * 8 + t];
  const uint32_t a2 = e[r0 * 8 + 4 + t], a3 = e[r1 * 8 + 4 + t];
  const float be0 = beta[r0], be1 = beta[r1];
  const uint32_t rw0 = rewrite[r0], rw1 = rewrite[r1];
  for (uint32_t nb = 0; nb < 32; nb++) {
    const uint32_t j0 = blockIdx.y * 256 + nb * 8;
    if (j0 >= k) break;
    const uint32_t b0 = f[(j0 + g) * 8 + t], b1 = f[(j0 + g) * 8 + 4 + t];
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    fp8_mma(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    const uint32_t j = j0 + 2 * t;
    const size_t at0 = (size_t)r0 * k + j, at1 = (size_t)r1 * k + j;
    const float al0 = signal[(size_t)r0 * (k >> 3) + (j >> 3)];
    const float al1 = signal[(size_t)r1 * (k >> 3) + (j >> 3)];
    // fp32_fma(beta, product, signal * int8): the multiply and the fused add each round once.
    const float m0 = __fmaf_rn(be0, c0, __fmul_rn(al0, (float)ints[at0]));
    const float m1 = __fmaf_rn(be0, c1, __fmul_rn(al0, (float)ints[at0 + 1]));
    const float m2 = __fmaf_rn(be1, c2, __fmul_rn(al1, (float)ints[at1]));
    const float m3 = __fmaf_rn(be1, c3, __fmul_rn(al1, (float)ints[at1 + 1]));
    // Columns 4q..4q+3 share the coin of column 4q, which lane t & ~1 holds.
    const float p0 = __shfl_xor_sync(0xFFFFFFFFu, m0, 1);
    const float p2 = __shfl_xor_sync(0xFFFFFFFFu, m2, 1);
    const float f0 = (t & 1) ? p0 : m0, f2 = (t & 1) ? p2 : m2;
    // j is even, so its two codes are neighbours in either layout.
    uint8_t *o0 = out + fp8_at(side, rows, r0, j);
    uint8_t *o1 = out + fp8_at(side, rows, r1, j);
    o0[0] = (uint8_t)fp8_code(m0, f0, rw0);
    o0[1] = (uint8_t)fp8_code(m1, f0, rw0);
    o1[0] = (uint8_t)fp8_code(m2, f2, rw1);
    o1[1] = (uint8_t)fp8_code(m3, f2, rw1);
  }
}

// ---------------------------------------------------------------------------------------
// BLAKE3, one 64-byte keyed block: the jackpot ticket
// ---------------------------------------------------------------------------------------

#define FP8_ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define FP8_G(a, b, c, d, x, y)                                                                  \
  a = a + b + (x); d = FP8_ROTR(d ^ a, 16); c = c + d; b = FP8_ROTR(b ^ c, 12);                  \
  a = a + b + (y); d = FP8_ROTR(d ^ a, 8);  c = c + d; b = FP8_ROTR(b ^ c, 7);

// One BLAKE3 compression: the first 8 words of the output. cv is the key or the running
// chaining value; flags are the BLAKE3 domain flags.
#define FP8_CHUNK_START 1u
#define FP8_CHUNK_END 2u
#define FP8_PARENT 4u
#define FP8_ROOT 8u
#define FP8_KEYED 16u

__device__ __forceinline__ void fp8_compress(const uint32_t *cv, const uint32_t *msg, uint32_t counter,
                                             uint32_t flags, uint32_t *out) {
  uint32_t s0 = cv[0], s1 = cv[1], s2 = cv[2], s3 = cv[3];
  uint32_t s4 = cv[4], s5 = cv[5], s6 = cv[6], s7 = cv[7];
  uint32_t s8 = 0x6A09E667u, s9 = 0xBB67AE85u, s10 = 0x3C6EF372u, s11 = 0xA54FF53Au;
  uint32_t s12 = counter, s13 = 0, s14 = 64, s15 = flags;
  uint32_t m[16];
#pragma unroll
  for (int i = 0; i < 16; i++) m[i] = msg[i];
#pragma unroll
  for (int r = 0; r < 7; r++) {
    FP8_G(s0, s4, s8, s12, m[0], m[1]);
    FP8_G(s1, s5, s9, s13, m[2], m[3]);
    FP8_G(s2, s6, s10, s14, m[4], m[5]);
    FP8_G(s3, s7, s11, s15, m[6], m[7]);
    FP8_G(s0, s5, s10, s15, m[8], m[9]);
    FP8_G(s1, s6, s11, s12, m[10], m[11]);
    FP8_G(s2, s7, s8, s13, m[12], m[13]);
    FP8_G(s3, s4, s9, s14, m[14], m[15]);
    const uint32_t p0 = m[2], p1 = m[6], p2 = m[3], p3 = m[10];
    const uint32_t p4 = m[7], p5 = m[0], p6 = m[4], p7 = m[13];
    const uint32_t p8 = m[1], p9 = m[11], p10 = m[12], p11 = m[5];
    const uint32_t p12 = m[9], p13 = m[14], p14 = m[15], p15 = m[8];
    m[0] = p0; m[1] = p1; m[2] = p2; m[3] = p3; m[4] = p4; m[5] = p5; m[6] = p6; m[7] = p7;
    m[8] = p8; m[9] = p9; m[10] = p10; m[11] = p11; m[12] = p12; m[13] = p13; m[14] = p14;
    m[15] = p15;
  }
  out[0] = s0 ^ s8; out[1] = s1 ^ s9; out[2] = s2 ^ s10; out[3] = s3 ^ s11;
  out[4] = s4 ^ s12; out[5] = s5 ^ s13; out[6] = s6 ^ s14; out[7] = s7 ^ s15;
}

// keyed_hash(key, msg) for a 64-byte msg: one compression, the whole input and the root.
__device__ __forceinline__ void fp8_ticket(const uint32_t *key, const uint32_t *msg, uint32_t *out) {
  fp8_compress(key, msg, 0, FP8_CHUNK_START | FP8_CHUNK_END | FP8_ROOT | FP8_KEYED, out);
}

// ---------------------------------------------------------------------------------------
// Noise lines (noise.rs sample_line and normalize_line)
// ---------------------------------------------------------------------------------------

// f32_to_bf16 and back: round to nearest even to the top 16 bits.
__device__ __forceinline__ float fp8_bf16(float x) {
  const uint32_t bits = __float_as_uint(x);
  return __uint_as_float(((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16) << 16);
}

// One thread a line: 32 bytes of keyed BLAKE3 for (side, factor, line), each byte a signed
// integer in +-[1, 128], the line scaled to an L2 norm of 256 with one BF16 division, cast
// to E4M3. out is count x 32 codes.
extern "C" __global__ void fp8_lines_kernel(const uint32_t *key, uint32_t side, uint32_t factor,
                                            uint32_t count, uint32_t *out) {
  const uint32_t line = blockIdx.x * blockDim.x + threadIdx.x;
  if (line >= count) return;
  uint32_t msg[16];
#pragma unroll
  for (int i = 0; i < 16; i++) msg[i] = 0;
  msg[0] = side | (factor << 8) | (line << 16);
  msg[1] = line >> 16;
  uint32_t w[8];
  fp8_ticket(key, msg, w);
  uint32_t sumsq = 0;
#pragma unroll
  for (int i = 0; i < 32; i++) {
    const uint32_t v = ((w[i >> 2] >> (8 * (i & 3))) & 0x7Fu) + 1u;
    sumsq += v * v;
  }
  // floor(sqrt(sumsq * 1024)), exact: the argument is below 2^30.
  const uint32_t arg = sumsq * 1024u;
  uint32_t root = (uint32_t)sqrt((double)arg);
  while ((uint64_t)root * root > arg) root--;
  while ((uint64_t)(root + 1) * (root + 1) <= arg) root++;
  const float scale = fp8_bf16(__fdiv_rn(8192.0f, fp8_bf16((float)root)));
#pragma unroll
  for (int q = 0; q < 8; q++) {
    uint32_t codes = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const uint32_t byte = (w[q] >> (8 * j)) & 0xFFu;
      const float x = (float)((byte & 0x7Fu) + 1u);
      const float scaled = fp8_bf16(__fmul_rn((byte & 0x80u) ? -x : x, scale));
      codes |= fp8_cast(scaled) << (8 * j);
    }
    out[line * 8 + q] = codes;
  }
}

// ---------------------------------------------------------------------------------------
// The Merkle tree of a committed plane: keyed BLAKE3 over 1024-byte chunks
// ---------------------------------------------------------------------------------------

// One thread a chunk: its chaining value, 16 compressions with the chunk index as counter.
extern "C" __global__ void fp8_leaf_kernel(const uint32_t *key, const uint32_t *data,
                                           uint32_t chunks, uint32_t *out) {
  const uint32_t at = blockIdx.x * blockDim.x + threadIdx.x;
  if (at >= chunks) return;
  uint32_t cv[8];
#pragma unroll
  for (int i = 0; i < 8; i++) cv[i] = key[i];
  const uint32_t *chunk = data + (size_t)at * 256;
  for (uint32_t b = 0; b < 16; b++) {
    const uint32_t flags = FP8_KEYED | (b == 0 ? FP8_CHUNK_START : 0u) | (b == 15 ? FP8_CHUNK_END : 0u);
    fp8_compress(cv, chunk + b * 16, at, flags, cv);
  }
#pragma unroll
  for (int i = 0; i < 8; i++) out[(size_t)at * 8 + i] = cv[i];
}

// One thread a parent: the merge of two children. flags carries ROOT for the last merge.
extern "C" __global__ void fp8_parent_kernel(const uint32_t *key, const uint32_t *in,
                                             uint32_t count, uint32_t flags, uint32_t *out) {
  const uint32_t at = blockIdx.x * blockDim.x + threadIdx.x;
  if (at >= count) return;
  uint32_t cv[8];
  fp8_compress(key, in + (size_t)at * 16, 0, FP8_PARENT | FP8_KEYED | flags, cv);
#pragma unroll
  for (int i = 0; i < 8; i++) out[(size_t)at * 8 + i] = cv[i];
}

// ---------------------------------------------------------------------------------------
// The search
// ---------------------------------------------------------------------------------------

// A block computes FP8_ROWS_A rows of A against FP8_ROWS_B rows of B; block (x, y) of the
// launch is block (x0 + x, y0 + y) of the product, and warps past the end of an operand do
// nothing. x runs fastest, so the blocks in flight share a strip of B and all of A stays in
// the L2 cache.
//
// A warp owns 64 rows of A by 64 rows of B: 4 x 8 atoms, 128 accumulators a lane, all in
// registers. Per plane (two steps) a lane loads 16 bytes for each atom of A and each 8 rows
// of B, then issues the 64 instructions.
//
// a, b       the noised operands, laid out as fp8_at has them (m and n rows)
// h          tile height, 4, 8 or 16; the tile is h x 256/h
// keybound   the jackpot key (8 words) then the winning bound (8 words, little endian)
// hits       pairs (tile row, tile column) of winning tiles; count is how many were found
// tickets    if set, every tile's 32-byte ticket at (tile row * tiles_b + tile column)
// dump       if set, the FP32 cells of the 64 x 128 window at (win_a, win_b), row-major
//            (launch only the blocks that cover it)

typedef float fp8_acc_t[FP8_ATOMS][FP8_NB][4];

// The 64 instructions of one plane. a0/a1: an atom's registers for the first and second
// step; bb: 8 rows of B, the first step in x, y and the second in z, w.
__device__ __forceinline__ void fp8_plane(fp8_acc_t &acc, const uint4 (&a0)[FP8_ATOMS],
                                          const uint4 (&a1)[FP8_ATOMS], const uint4 (&bb)[FP8_NB],
                                          bool two) {
#pragma unroll
  for (int nb = 0; nb < FP8_NB; nb++) {
#pragma unroll
    for (int at = 0; at < FP8_ATOMS; at++) {
      fp8_mma(acc[at][nb][0], acc[at][nb][1], acc[at][nb][2], acc[at][nb][3],
              a0[at].x, a0[at].y, a0[at].z, a0[at].w, bb[nb].x, bb[nb].y);
    }
  }
  if (two) {
#pragma unroll
    for (int nb = 0; nb < FP8_NB; nb++) {
#pragma unroll
      for (int at = 0; at < FP8_ATOMS; at++) {
        fp8_mma(acc[at][nb][0], acc[at][nb][1], acc[at][nb][2], acc[at][nb][3],
                a1[at].x, a1[at].y, a1[at].z, a1[at].w, bb[nb].z, bb[nb].w);
      }
    }
  }
}

// After the product: the window dump, then every tile folded, hashed and compared.
// words: this warp's 256 words of shared memory.
__device__ __forceinline__ void fp8_finish(fp8_acc_t &acc, uint32_t *words, uint32_t lane,
                                           uint32_t base_a, uint32_t base_b, uint32_t h,
                                           uint32_t win_a, uint32_t win_b,
                                           const uint32_t *keybound, uint32_t *hits,
                                           uint32_t *count, uint32_t max_hits, uint8_t *tickets,
                                           uint32_t tiles_b, float *dump) {
  const uint32_t g = lane >> 2, t = lane & 3;
  if (dump) {
#pragma unroll
    for (int at = 0; at < FP8_ATOMS; at++) {
#pragma unroll
      for (int nb = 0; nb < FP8_NB; nb++) {
        if (base_a - win_a >= FP8_DUMP_A || base_b - win_b >= FP8_DUMP_B) continue;
        const uint32_t to = (base_a - win_a + at * 16 + g) * FP8_DUMP_B + (base_b - win_b) + nb * 8 + 2 * t;
        dump[to] = acc[at][nb][0];
        dump[to + 1] = acc[at][nb][1];
        dump[to + 8 * FP8_DUMP_B] = acc[at][nb][2];
        dump[to + 8 * FP8_DUMP_B + 1] = acc[at][nb][3];
      }
    }
  }

#if FP8_DIAG == 1
  // Tuning only (wrong tickets): no fold and no hash, to time the product alone.
  uint32_t x = 0;
#pragma unroll
  for (int at = 0; at < FP8_ATOMS; at++) {
#pragma unroll
    for (int nb = 0; nb < FP8_NB; nb++) {
      x ^= __float_as_uint(acc[at][nb][0]) ^ __float_as_uint(acc[at][nb][1]) ^
           __float_as_uint(acc[at][nb][2]) ^ __float_as_uint(acc[at][nb][3]);
    }
  }
  if (x == 0x9E3779B1u) atomicAdd(count, 1u);
  return;
#endif
  // xor_fold_extract: a tile, row-major, is 16 runs of 16 cells, and a run is 16 consecutive
  // columns of one row: two 8-column atoms, whose cells sit two a lane in the four lanes of
  // the row. The fold is sequential, so its value walks those lanes: lane t folds its two
  // cells and hands the value to lane t + 1, twice round. It ends in the row's lane 0.
  const uint32_t cpr = 16 / h;       // runs across one tile
  const uint32_t across = FP8_WARP_COLS * h / 256;  // tiles across the warp's columns
  const uint32_t from = (lane & ~3u) | ((lane - 1) & 3u);
#pragma unroll
  for (int at = 0; at < FP8_ATOMS; at++) {
#pragma unroll
    for (int half = 0; half < 2; half++) {
#pragma unroll
      for (int cr = 0; cr < FP8_NB / 2; cr++) {
        uint32_t v = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) {
          const int nb = 2 * cr + (i >> 2);
          v = v * 0x9E3779B1u + __float_as_uint(acc[at][nb][2 * half]);
          v = (v << 13) | (v >> 19);
          v = v * 0x9E3779B1u + __float_as_uint(acc[at][nb][2 * half + 1]);
          v = (v << 13) | (v >> 19);
          // Only the lane whose turn it is (t == i & 3) holds the run's value.
          v = __shfl_sync(0xFFFFFFFFu, v, from);
        }
        if (t == 0) {
          const uint32_t row = at * 16 + half * 8 + g;
          const uint32_t tile = (row / h) * across + cr / cpr;
          words[tile * 16 + (row % h) * cpr + cr % cpr] = v;
        }
      }
    }
  }
  __syncwarp();
#if FP8_DIAG == 2
  // Tuning only (wrong tickets): the fold without the hash.
  if (lane < 16 && words[lane * 16] == 0x9E3779B1u) atomicAdd(count, 1u);
  return;
#endif
  if (lane < 16) {
    uint32_t ticket[8];
    fp8_ticket(keybound, words + lane * 16, ticket);
    const uint32_t tile_a = base_a / h + lane / across;
    const uint32_t tile_b = base_b / (256 / h) + lane % across;
    if (tickets) {
      uint32_t *o = reinterpret_cast<uint32_t *>(tickets) + ((size_t)tile_a * tiles_b + tile_b) * 8;
#pragma unroll
      for (int i = 0; i < 8; i++) o[i] = ticket[i];
    }
    // ticket <= bound, both 256-bit little endian.
    bool wins = true;
    for (int i = 7; i >= 0; i--) {
      if (ticket[i] != keybound[8 + i]) {
        wins = ticket[i] < keybound[8 + i];
        break;
      }
    }
    if (wins) {
      const uint32_t at = atomicAdd(count, 1u);
      if (at < max_hits) {
        hits[2 * at] = tile_a;
        hits[2 * at + 1] = tile_b;
      }
    }
  }
}

#define FP8_STAGE_BYTES ((FP8_ROWS_A + FP8_ROWS_B) * 64)
#define FP8_SMEM_BYTES (FP8_SMEM ? FP8_STAGES * FP8_STAGE_BYTES : 0)

__device__ __forceinline__ void fp8_cp16(void *dst, const void *src) {
  const uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
}

// FP8_SMEM 2: one bulk copy (the TMA engine) a run, which completes on the stage's mbarrier.
__device__ __forceinline__ void fp8_bulk(void *dst, const void *src, uint32_t bytes, uint64_t *bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src), "r"(bytes),
               "r"((uint32_t)__cvta_generic_to_shared(bar)) : "memory");
}

__device__ __forceinline__ void fp8_wait(uint64_t *bar, uint32_t parity) {
  const uint32_t b = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile(
      "{\n .reg .pred P;\n WAIT_%=:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
      " @!P bra WAIT_%=;\n}\n" ::"r"(b), "r"(parity) : "memory");
}

// FP8_MINB: blocks an SM the compiler must leave room for (registers), for tuning occupancy.
#ifndef FP8_MINB
#define FP8_MINB 1
#endif
extern "C" __global__ void __launch_bounds__(FP8_THREADS, FP8_MINB)
fp8_search_kernel(const uint8_t *a, const uint8_t *b, uint32_t k, uint32_t h, uint32_t m,
                  uint32_t n, uint32_t x0, uint32_t y0, uint32_t nbx, uint32_t tiles,
                  uint32_t win_a, uint32_t win_b,
                  const uint32_t *keybound, uint32_t *hits, uint32_t *count,
                  uint32_t max_hits, uint8_t *tickets, uint32_t tiles_b, float *dump) {
#ifdef FP8_PAD_SMEM
  // Tuning only: unused shared memory, to cap how many warps an SM runs at once.
  __shared__ volatile uint32_t spad[FP8_PAD_SMEM / 4];
  if (k == 0) spad[0] = 0;
#endif
  // Block t of the launch's `tiles` is block (x0 + t % nbx, y0 + t / nbx) of the product. A
  // launch block takes t = blockIdx.x, then t + gridDim.x and on (FP8_PERSIST launches one
  // block a slot of the card, so the next tile's first planes load during this one's fold).
  const uint32_t tid = threadIdx.x;
  const uint32_t warp = tid >> 5, lane = tid & 31;
  const uint32_t wa = warp % FP8_WA, wb = warp / FP8_WA;
  const size_t plane_a = (size_t)m * 64, plane_b = (size_t)n * 64;
  const uint32_t steps = k / 32, planes = (steps + 1) / 2;

  fp8_acc_t acc;
#pragma unroll
  for (int at = 0; at < FP8_ATOMS; at++) {
#pragma unroll
    for (int nb = 0; nb < FP8_NB; nb++) {
      acc[at][nb][0] = 0.0f; acc[at][nb][1] = 0.0f; acc[at][nb][2] = 0.0f; acc[at][nb][3] = 0.0f;
    }
  }

#if FP8_SMEM
  extern __shared__ uint4 fp8_dsm[];
  uint8_t *sm = reinterpret_cast<uint8_t *>(fp8_dsm);
  // Planes are counted across this launch block's tiles: plane q is plane q % planes of its
  // (q / planes)th tile. A block's rows of a plane are one run in global memory.
#if FP8_SMEM == 2
  __shared__ uint64_t bars[FP8_STAGES];
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < FP8_STAGES; i++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"((uint32_t)__cvta_generic_to_shared(&bars[i])));
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
#endif
  // The writer: the next plane to stage is plane wp of launch tile wt, whose rows start at
  // wa_ and wb_ in plane 0 (wca, wcb 16-byte chunks); wq counts planes staged. Kept as a
  // cursor so the per-plane path has no division.
  uint32_t wt = blockIdx.x, wp = 0, wq = 0, wca = 0, wcb = 0;
  const uint8_t *wa_ = a, *wb_ = b;
  auto seek = [&]() {
    if (wt >= tiles) return;
    const uint32_t ra = (x0 + wt % nbx) * FP8_ROWS_A, rb = (y0 + wt / nbx) * FP8_ROWS_B;
    wca = min(m - ra, (uint32_t)FP8_ROWS_A) * 4;
    wcb = min(n - rb, (uint32_t)FP8_ROWS_B) * 4;
    wa_ = a + (size_t)ra * 64;
    wb_ = b + (size_t)rb * 64;
  };
  seek();
  auto issue = [&]() {
    const uint32_t st = wq++ % FP8_STAGES;
    if (wt >= tiles) return;
    uint8_t *sa = sm + st * FP8_STAGE_BYTES;
    uint8_t *sb = sa + FP8_ROWS_A * 64;
    const uint8_t *pa = wa_ + wp * plane_a, *pb = wb_ + wp * plane_b;
#if FP8_SMEM == 2
    if (tid == 0) {
      uint64_t *bar = &bars[st];
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                   ::"r"((uint32_t)__cvta_generic_to_shared(bar)), "r"((wca + wcb) * 16) : "memory");
      fp8_bulk(sa, pa, wca * 16, bar);
      fp8_bulk(sb, pb, wcb * 16, bar);
    }
#else
    for (uint32_t c = tid; c < wca; c += FP8_THREADS) fp8_cp16(sa + 16 * c, pa + 16 * c);
    for (uint32_t c = tid; c < wcb; c += FP8_THREADS) fp8_cp16(sb + 16 * c, pb + 16 * c);
#endif
    if (++wp == planes) {
      wp = 0;
      wt += gridDim.x;
      seek();
    }
  };
#if FP8_SMEM == 2
#define FP8_COMMIT()
#define FP8_WAIT_STAGE(q) fp8_wait(&bars[(q) % FP8_STAGES], ((q) / FP8_STAGES) & 1u)
#else
#define FP8_COMMIT() asm volatile("cp.async.commit_group;\n" ::)
#define FP8_WAIT_STAGE(q) asm volatile("cp.async.wait_group %0;\n" ::"n"(FP8_STAGES - 2))
#endif
#pragma unroll
  for (int i = 0; i < FP8_STAGES - 1; i++) {
    issue();
    FP8_COMMIT();
  }
  const uint32_t off_a = wa * FP8_WARP_ROWS * 64 + 16 * lane;
  const uint32_t off_b = FP8_ROWS_A * 64 + wb * FP8_WARP_COLS * 64 + 16 * lane;
  uint32_t q = 0;   // planes read
  for (uint32_t t = blockIdx.x; t < tiles; t += gridDim.x) {
    const uint32_t base_a = (x0 + t % nbx) * FP8_ROWS_A + wa * FP8_WARP_ROWS;
    const uint32_t base_b = (y0 + t / nbx) * FP8_ROWS_B + wb * FP8_WARP_COLS;
    const bool active = base_a < m && base_b < n;
    const uint8_t *s = sm;
    for (uint32_t p = 0; p < planes; p++, q++) {
      FP8_WAIT_STAGE(q);
      __syncthreads();
      s = sm + (q % FP8_STAGES) * FP8_STAGE_BYTES;
      if (active) {
        uint4 a0[FP8_ATOMS], a1[FP8_ATOMS], bb[FP8_NB];
#pragma unroll
        for (int at = 0; at < FP8_ATOMS; at++) {
          a0[at] = *reinterpret_cast<const uint4 *>(s + off_a + at * 1024);
          a1[at] = *reinterpret_cast<const uint4 *>(s + off_a + at * 1024 + 512);
        }
#pragma unroll
        for (int nb = 0; nb < FP8_NB; nb++) bb[nb] = *reinterpret_cast<const uint4 *>(s + off_b + nb * 512);
        fp8_plane(acc, a0, a1, bb, 2 * p + 1 < steps);
      }
      issue();
      FP8_COMMIT();
    }
    {
      // The stage just read stays free until the next pass refills it (after its
      // __syncthreads), so the fold borrows it: 1 KB a warp.
      if (active) {
        uint32_t *words = reinterpret_cast<uint32_t *>(const_cast<uint8_t *>(s)) + warp * 256;
        fp8_finish(acc, words, lane, base_a, base_b, h, win_a, win_b, keybound, hits, count, max_hits,
                   tickets, tiles_b, dump);
#if FP8_SMEM == 2
        // The next refill of this stage is a bulk copy (the async proxy): order the fold's
        // writes before it.
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
      }
#pragma unroll
      for (int at = 0; at < FP8_ATOMS; at++) {
#pragma unroll
        for (int nb = 0; nb < FP8_NB; nb++) {
          acc[at][nb][0] = 0.0f; acc[at][nb][1] = 0.0f; acc[at][nb][2] = 0.0f; acc[at][nb][3] = 0.0f;
        }
      }
    }
  }
#if FP8_SMEM == 1
  asm volatile("cp.async.wait_group 0;\n" ::);
#endif
#else
  // The 64-byte messages of a warp's tiles (16 of them, whatever the tile's shape).
  __shared__ uint32_t swords[FP8_WARPS * 256];
  uint32_t *words = swords + warp * 256;
  for (uint32_t t = blockIdx.x; t < tiles; t += gridDim.x) {
    const uint32_t base_a = (x0 + t % nbx) * FP8_ROWS_A + wa * FP8_WARP_ROWS;
    const uint32_t base_b = (y0 + t / nbx) * FP8_ROWS_B + wb * FP8_WARP_COLS;
    if (base_a >= m || base_b >= n) continue;
    const uint8_t *pa = a + (size_t)(base_a / 16) * 1024 + 16 * lane;
    const uint8_t *pb = b + (size_t)base_b * 64 + 16 * lane;
    for (uint32_t p = 0; p < planes; p++, pa += plane_a, pb += plane_b) {
      // Every load of the plane first, then the instructions: a warp issues in order, so
      // loads placed together overlap.
      uint4 a0[FP8_ATOMS], a1[FP8_ATOMS], bb[FP8_NB];
#pragma unroll
      for (int at = 0; at < FP8_ATOMS; at++) {
        a0[at] = *reinterpret_cast<const uint4 *>(pa + at * 1024);
        a1[at] = *reinterpret_cast<const uint4 *>(pa + at * 1024 + 512);
      }
#pragma unroll
      for (int nb = 0; nb < FP8_NB; nb++) bb[nb] = *reinterpret_cast<const uint4 *>(pb + nb * 512);
      fp8_plane(acc, a0, a1, bb, 2 * p + 1 < steps);
    }
    fp8_finish(acc, words, lane, base_a, base_b, h, win_a, win_b, keybound, hits, count, max_hits,
               tickets, tiles_b, dump);
#pragma unroll
    for (int at = 0; at < FP8_ATOMS; at++) {
#pragma unroll
      for (int nb = 0; nb < FP8_NB; nb++) {
        acc[at][nb][0] = 0.0f; acc[at][nb][1] = 0.0f; acc[at][nb][2] = 0.0f; acc[at][nb][3] = 0.0f;
      }
    }
  }
#endif
}

// ---------------------------------------------------------------------------------------
// Host half
// ---------------------------------------------------------------------------------------

namespace {

struct Side {
  uint32_t rows = 0;
  uint8_t *codes = nullptr;
  int8_t *ints = nullptr;
  uint32_t *e = nullptr, *f = nullptr, *rewrite = nullptr;
  float *signal = nullptr, *beta = nullptr;
};

struct Ctx {
  bool open = false;
  uint32_t slots = 0;   // search blocks the whole card runs at once
  uint32_t k = 0, stride = 0, h = 0, max_hits = 0;
  Side side[2];
  uint32_t *keybound = nullptr, *hits = nullptr, *count = nullptr, *linekey = nullptr;
  float *dump = nullptr;
  uint8_t *tickets = nullptr;
  uint8_t *plane = nullptr;   // a plane being hashed into its tree
  uint32_t *cvs = nullptr;    // and the tree's nodes
  size_t plane_bytes = 0;
  char err[256] = "";
};

Ctx ctx;

// FP8_PERSIST: launch only as many search blocks as the card runs at once (ctx.slots), each
// taking tile after tile. 0 (the default): one launch block a tile, which ran 4% faster on an
// RTX 5080 (round 4 in earn/tuning/salad/fp8tune.txt).
#ifndef FP8_PERSIST
#define FP8_PERSIST 0
#endif
uint32_t launch_blocks(uint32_t tiles) {
  return FP8_PERSIST && ctx.slots && ctx.slots < tiles ? ctx.slots : tiles;
}

int fail(const char *what, cudaError_t e) {
  snprintf(ctx.err, sizeof(ctx.err), "%s: %s", what, cudaGetErrorString(e));
  return 1;
}

int fail(const char *what) {
  snprintf(ctx.err, sizeof(ctx.err), "%s", what);
  return 1;
}

template <typename T>
void release(T *&p) {
  if (p) cudaFree(p);
  p = nullptr;
}

void release_all() {
  for (Side &s : ctx.side) {
    release(s.codes); release(s.ints); release(s.e); release(s.f); release(s.rewrite);
    release(s.signal); release(s.beta);
  }
  release(ctx.keybound); release(ctx.hits); release(ctx.count); release(ctx.linekey);
  release(ctx.dump); release(ctx.tickets); release(ctx.plane); release(ctx.cvs);
  ctx.plane_bytes = 0;
}

#define FP8_TRY(what, call)                               \
  do {                                                    \
    const cudaError_t e_ = (call);                        \
    if (e_ != cudaSuccess) return fail(what, e_);         \
  } while (0)

#define FP8_READY()                                                          \
  do {                                                                       \
    if (!ctx.side[0].codes) return fail("fp8_configure was not called");     \
  } while (0)

}  // namespace

extern "C" {

const char *fp8_error() { return ctx.err; }

// Select the card. It must be compute capability 12.x: no other card has this kernel.
int fp8_open(int device, char *name, int name_len, int *major, int *minor) {
  FP8_TRY("cudaSetDevice", cudaSetDevice(device));
  cudaDeviceProp prop;
  FP8_TRY("cudaGetDeviceProperties", cudaGetDeviceProperties(&prop, device));
  snprintf(name, (size_t)name_len, "%s", prop.name);
  *major = prop.major;
  *minor = prop.minor;
  if (prop.major != 12) return fail("the FP8 core needs an RTX 50 or RTX PRO Blackwell card (compute 12.x)");
  // Always raise the limit: 48 KB of stages plus the static barriers is already past the default.
  if (FP8_SMEM_BYTES > 0) {
    FP8_TRY("search shared memory", cudaFuncSetAttribute(fp8_search_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                         FP8_SMEM_BYTES));
  }
  int per_sm = 0;
  FP8_TRY("search occupancy", cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, fp8_search_kernel, FP8_THREADS,
                                                                             FP8_SMEM_BYTES));
  ctx.slots = (uint32_t)(per_sm > 0 ? per_sm : 1) * (uint32_t)prop.multiProcessorCount;
  ctx.open = true;
  return 0;
}

// Allocate for a job shape. m is a multiple of 64, n of 128, k of 32; h is 4, 8 or 16.
int fp8_configure(uint32_t m, uint32_t n, uint32_t k, uint32_t h, uint32_t max_hits) {
  if (!ctx.open) return fail("fp8_open was not called");
  if (m == 0 || n == 0 || m % FP8_DUMP_A || n % FP8_DUMP_B) return fail("m must be a multiple of 64 and n of 128");
  if (k < 1024 || k > 65536 || k % 32) return fail("k must be a multiple of 32 in 1024..65536");
  if (h != 4 && h != 8 && h != 16) return fail("the tile height must be 4, 8 or 16");
  if (max_hits == 0) return fail("max_hits must be positive");
  release_all();
  ctx.k = k; ctx.h = h; ctx.max_hits = max_hits;
  ctx.stride = (k + 63u) & ~63u;
  ctx.side[0].rows = m;
  ctx.side[1].rows = n;
  for (Side &s : ctx.side) {
    FP8_TRY("alloc codes", cudaMalloc(&s.codes, (size_t)s.rows * ctx.stride));
    FP8_TRY("clear codes", cudaMemset(s.codes, 0, (size_t)s.rows * ctx.stride));
    FP8_TRY("alloc values", cudaMalloc(&s.ints, (size_t)s.rows * k));
    FP8_TRY("alloc E", cudaMalloc(&s.e, (size_t)s.rows * 32));
    FP8_TRY("alloc F", cudaMalloc(&s.f, (size_t)k * 32));
    FP8_TRY("alloc signal", cudaMalloc(&s.signal, (size_t)s.rows * (k / 8) * 4));
    FP8_TRY("alloc beta", cudaMalloc(&s.beta, (size_t)s.rows * 4));
    FP8_TRY("alloc rewrite", cudaMalloc(&s.rewrite, (size_t)s.rows * 4));
  }
  const size_t tiles = (size_t)(m / h) * (n / (256 / h));
  FP8_TRY("alloc key", cudaMalloc(&ctx.keybound, 64));
  FP8_TRY("alloc line key", cudaMalloc(&ctx.linekey, 32));
  FP8_TRY("alloc hits", cudaMalloc(&ctx.hits, (size_t)max_hits * 8));
  FP8_TRY("alloc count", cudaMalloc(&ctx.count, 4));
  FP8_TRY("alloc cells", cudaMalloc(&ctx.dump, (size_t)FP8_DUMP_A * FP8_DUMP_B * 4));
  // Every tile's ticket, for the verify tool; only small shapes ask for it.
  if (tiles <= (1u << 20)) FP8_TRY("alloc tickets", cudaMalloc(&ctx.tickets, tiles * 32));
  return 0;
}

// Load E4M3 codes as they are (side 0 is A', 1 is B'). For the verify tool.
int fp8_set_codes(int side, const uint8_t *codes) {
  FP8_READY();
  const Side &s = ctx.side[side];
  std::vector<uint8_t> laned((size_t)s.rows * ctx.stride, 0);
  for (uint32_t r = 0; r < s.rows; r++) {
    for (uint32_t j = 0; j < ctx.k; j++) laned[fp8_at(side, s.rows, r, j)] = codes[(size_t)r * ctx.k + j];
  }
  FP8_TRY("upload codes", cudaMemcpy(s.codes, laned.data(), laned.size(), cudaMemcpyHostToDevice));
  return 0;
}

int fp8_get_codes(int side, uint32_t row0, uint32_t rows, uint8_t *out) {
  FP8_READY();
  const Side &s = ctx.side[side];
  if ((uint64_t)row0 + rows > s.rows) return fail("rows out of range");
  // Rows lo..hi (whole atoms) of every plane: one strided copy.
  const uint32_t lo = row0 & ~15u, hi = (row0 + rows + 15u) & ~15u;
  const uint32_t span = (hi < s.rows ? hi : s.rows) - lo, planes = ctx.stride / 64;
  std::vector<uint8_t> part((size_t)span * ctx.stride);
  FP8_TRY("download codes", cudaMemcpy2D(part.data(), (size_t)span * 64, s.codes + (size_t)lo * 64, (size_t)s.rows * 64,
                                         (size_t)span * 64, planes, cudaMemcpyDeviceToHost));
  for (uint32_t r = 0; r < rows; r++) {
    for (uint32_t j = 0; j < ctx.k; j++) out[(size_t)r * ctx.k + j] = part[fp8_at(side, span, row0 + r - lo, j)];
  }
  return 0;
}

// An operand's committed values and the numbers that noise its rows (fp8Host.rowParams):
// ints rows x k, signal rows x k/8, and a row's beta and rewrite constants. They stay on
// the card until the next load.
int fp8_load(int side, const int8_t *ints, const float *signal, const float *beta, const uint32_t *rewrite) {
  FP8_READY();
  const Side &s = ctx.side[side];
  FP8_TRY("upload values", cudaMemcpy(s.ints, ints, (size_t)s.rows * ctx.k, cudaMemcpyHostToDevice));
  FP8_TRY("upload signal", cudaMemcpy(s.signal, signal, (size_t)s.rows * (ctx.k / 8) * 4, cudaMemcpyHostToDevice));
  FP8_TRY("upload beta", cudaMemcpy(s.beta, beta, (size_t)s.rows * 4, cudaMemcpyHostToDevice));
  FP8_TRY("upload rewrite", cudaMemcpy(s.rewrite, rewrite, (size_t)s.rows * 4, cudaMemcpyHostToDevice));
  return 0;
}

// Overwrite len loaded values from `offset` on: how a new attempt changes A.
int fp8_patch(int side, uint64_t offset, const int8_t *ints, uint32_t len) {
  FP8_READY();
  const Side &s = ctx.side[side];
  if (offset + len > (uint64_t)s.rows * ctx.k) return fail("patch out of range");
  FP8_TRY("patch values", cudaMemcpy(s.ints + offset, ints, len, cudaMemcpyHostToDevice));
  return 0;
}

// Noise lines given as codes: factor 0 is E (rows x 32), 1 is F (k x 32).
int fp8_set_lines(int side, int factor, const uint8_t *lines) {
  FP8_READY();
  const Side &s = ctx.side[side];
  FP8_TRY("upload lines", cudaMemcpy(factor ? s.f : s.e, lines, (size_t)(factor ? ctx.k : s.rows) * 32, cudaMemcpyHostToDevice));
  return 0;
}

// Noise lines sampled on the card: lines 0.. of (side, factor) under the 32-byte line key
// (fp8.noiseLineKey of the seed).
int fp8_seed_lines(int side, int factor, const uint8_t *key) {
  FP8_READY();
  const Side &s = ctx.side[side];
  const uint32_t count = factor ? ctx.k : s.rows;
  FP8_TRY("upload line key", cudaMemcpy(ctx.linekey, key, 32, cudaMemcpyHostToDevice));
  fp8_lines_kernel<<<(count + 127) / 128, 128>>>(ctx.linekey, (uint32_t)side, (uint32_t)factor, count, factor ? s.f : s.e);
  FP8_TRY("lines launch", cudaGetLastError());
  FP8_TRY("lines", cudaDeviceSynchronize());
  return 0;
}

int fp8_get_lines(int side, int factor, uint8_t *out) {
  FP8_READY();
  const Side &s = ctx.side[side];
  FP8_TRY("download lines", cudaMemcpy(out, factor ? s.f : s.e, (size_t)(factor ? ctx.k : s.rows) * 32, cudaMemcpyDeviceToHost));
  return 0;
}

// Noise the loaded operand with its loaded lines: A' or B' = Q(alpha.X + beta.(E @ F)).
int fp8_renoise(int side) {
  FP8_READY();
  const Side &s = ctx.side[side];
  fp8_noise_kernel<<<dim3(s.rows / 16, (ctx.k + 255) / 256), 32>>>(
      s.ints, s.e, s.f, s.signal, s.beta, s.rewrite, s.codes, ctx.k, side, s.rows);
  FP8_TRY("noise launch", cudaGetLastError());
  FP8_TRY("noise", cudaDeviceSynchronize());
  return 0;
}

// The keyed Merkle tree of a plane of `chunks` 1024-byte chunks (a power of two, at least
// 2). out receives every node, 32 bytes each: the chunks' values, then each layer above,
// the root last; 2 * chunks - 1 nodes.
int fp8_tree(const uint8_t *key, const uint8_t *data, uint32_t chunks, uint8_t *out) {
  if (!ctx.open) return fail("fp8_open was not called");
  if (chunks < 2 || (chunks & (chunks - 1))) return fail("a plane must be a power-of-two number of chunks, at least 2");
  const size_t bytes = (size_t)chunks * 1024;
  if (bytes > ctx.plane_bytes) {
    release(ctx.plane);
    release(ctx.cvs);
    ctx.plane_bytes = 0;
    FP8_TRY("alloc plane", cudaMalloc(&ctx.plane, bytes + 32));
    FP8_TRY("alloc tree", cudaMalloc(&ctx.cvs, (size_t)chunks * 64));
    ctx.plane_bytes = bytes;
  }
  uint32_t *dkey = reinterpret_cast<uint32_t *>(ctx.plane + ctx.plane_bytes);
  FP8_TRY("upload plane", cudaMemcpy(ctx.plane, data, bytes, cudaMemcpyHostToDevice));
  FP8_TRY("upload tree key", cudaMemcpy(dkey, key, 32, cudaMemcpyHostToDevice));
  fp8_leaf_kernel<<<(chunks + 127) / 128, 128>>>(dkey, reinterpret_cast<const uint32_t *>(ctx.plane), chunks, ctx.cvs);
  FP8_TRY("leaf launch", cudaGetLastError());
  uint32_t *below = ctx.cvs;
  for (uint32_t count = chunks / 2; count >= 1; count /= 2) {
    uint32_t *above = below + (size_t)count * 16;
    fp8_parent_kernel<<<(count + 127) / 128, 128>>>(dkey, below, count, count == 1 ? FP8_ROOT : 0u, above);
    FP8_TRY("parent launch", cudaGetLastError());
    below = above;
  }
  FP8_TRY("tree", cudaDeviceSynchronize());
  FP8_TRY("download tree", cudaMemcpy(out, ctx.cvs, ((size_t)chunks * 2 - 1) * 32, cudaMemcpyDeviceToHost));
  return 0;
}

// Search every tile of A' and B'. key is the 32-byte jackpot key, bound the 32-byte
// little-endian winning bound. hits receives up to max_hits (tile row, tile column) pairs
// and *count how many tiles won (it can exceed max_hits). tickets, if not null, receives
// every tile's ticket (small shapes only).
int fp8_search(const uint8_t *key, const uint8_t *bound, uint32_t *hits, uint32_t *count, uint8_t *tickets) {
  FP8_READY();
  if (tickets && !ctx.tickets) return fail("this shape is too large to return every ticket");
  const uint32_t m = ctx.side[0].rows, n = ctx.side[1].rows;
  uint8_t kb[64];
  memcpy(kb, key, 32);
  memcpy(kb + 32, bound, 32);
  FP8_TRY("upload key", cudaMemcpy(ctx.keybound, kb, 64, cudaMemcpyHostToDevice));
  FP8_TRY("clear count", cudaMemset(ctx.count, 0, 4));
  const uint32_t tiles_b = n / (256 / ctx.h);
  const uint32_t nbx = (m + FP8_ROWS_A - 1) / FP8_ROWS_A, tiles = nbx * ((n + FP8_ROWS_B - 1) / FP8_ROWS_B);
  fp8_search_kernel<<<launch_blocks(tiles), FP8_THREADS, FP8_SMEM_BYTES>>>(
      ctx.side[0].codes, ctx.side[1].codes, ctx.k, ctx.h, m, n, 0, 0, nbx, tiles, 0, 0, ctx.keybound, ctx.hits, ctx.count,
      ctx.max_hits, tickets ? ctx.tickets : nullptr, tiles_b, nullptr);
  FP8_TRY("search launch", cudaGetLastError());
  FP8_TRY("search", cudaDeviceSynchronize());
  FP8_TRY("download count", cudaMemcpy(count, ctx.count, 4, cudaMemcpyDeviceToHost));
  const uint32_t got = *count < ctx.max_hits ? *count : ctx.max_hits;
  if (got) FP8_TRY("download hits", cudaMemcpy(hits, ctx.hits, (size_t)got * 8, cudaMemcpyDeviceToHost));
  if (tickets) {
    FP8_TRY("download tickets", cudaMemcpy(tickets, ctx.tickets, (size_t)(m / ctx.h) * tiles_b * 32, cudaMemcpyDeviceToHost));
  }
  return 0;
}

// The FP32 cells of one block: rows row_a..+64 of A' by rows row_b..+128 of B', row-major.
// For the verify tool, which compares their bits with the CPU reference.
int fp8_cells(uint32_t row_a, uint32_t row_b, float *out) {
  FP8_READY();
  if (row_a % FP8_DUMP_A || row_b % FP8_DUMP_B || row_a >= ctx.side[0].rows || row_b >= ctx.side[1].rows) return fail("block out of range");
  uint8_t kb[64];
  memset(kb, 0, sizeof(kb));
  FP8_TRY("upload key", cudaMemcpy(ctx.keybound, kb, 64, cudaMemcpyHostToDevice));
  // The blocks that cover the window, from the one that holds its first cell.
  const uint32_t x0 = row_a / FP8_ROWS_A, y0 = row_b / FP8_ROWS_B;
  const uint32_t nx = (row_a + FP8_DUMP_A - 1) / FP8_ROWS_A - x0 + 1;
  const uint32_t ny = (row_b + FP8_DUMP_B - 1) / FP8_ROWS_B - y0 + 1;
  fp8_search_kernel<<<launch_blocks(nx * ny), FP8_THREADS, FP8_SMEM_BYTES>>>(
      ctx.side[0].codes, ctx.side[1].codes, ctx.k, ctx.h, ctx.side[0].rows, ctx.side[1].rows,
      x0, y0, nx, nx * ny, row_a, row_b, ctx.keybound, ctx.hits, ctx.count,
      0, nullptr, 1, ctx.dump);
  FP8_TRY("cells launch", cudaGetLastError());
  FP8_TRY("cells", cudaDeviceSynchronize());
  FP8_TRY("download cells", cudaMemcpy(out, ctx.dump, (size_t)FP8_DUMP_A * FP8_DUMP_B * 4, cudaMemcpyDeviceToHost));
  return 0;
}

void fp8_close() {
  release_all();
  ctx.k = ctx.stride = ctx.h = 0;
  ctx.side[0].rows = ctx.side[1].rows = 0;
}

}  // extern "C"
