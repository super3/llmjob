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
// Two kernels:
//   fp8_noise_kernel   A' = Q(alpha.A + beta.(E @ F)) for a stack of rows: the rank-32 noise
//                      product on the same instruction, the FP32 mix, the cast to E4M3 and
//                      the downscale rewrite (quantization.rs).
//   fp8_search_kernel  the product, 64 rows of A by 128 rows of B a block, then each 256-cell
//                      tile folded to its 64-byte message, hashed, and compared with the bound.
//
// The host half below owns device memory and is the C interface pearl_fp8_core.cc binds.
// Everything the GPU cannot do exactly the way the verifier does (row norms, alpha and beta,
// noise lines, Merkle trees) is done by the JS host (fp8Host.js) and handed in.
//
// See earn/tuning/pip3-fp8.md for the spec and earn/native/fp8/README.md for the layout.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define FP8_WARPS 4                    // warps a block; each owns 16 rows of A
#define FP8_NB 16                      // 8-column accumulator blocks a warp holds in registers
#define FP8_KC 128                     // k staged in shared memory at a time
#define FP8_ROWS_A (16 * FP8_WARPS)    // 64
#define FP8_ROWS_B (8 * FP8_NB)        // 128
#define FP8_THREADS (32 * FP8_WARPS)

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
// ints is rows x k int8; e is rows x 32 and f is k x 32 E4M3 codes; out is rows x k codes.
// signal is rows x k/8: fl(alpha * scale) of each 8-value block, so the clean term is one
// more FP32 multiply by the int8, as quantization.rs has it.
extern "C" __global__ void fp8_noise_kernel(const int8_t *ints, const uint32_t *e,
                                            const uint32_t *f, const float *signal,
                                            const float *beta, const uint32_t *rewrite,
                                            uint8_t *out, uint32_t k) {
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
    out[at0] = (uint8_t)fp8_code(m0, f0, rw0);
    out[at0 + 1] = (uint8_t)fp8_code(m1, f0, rw0);
    out[at1] = (uint8_t)fp8_code(m2, f2, rw1);
    out[at1 + 1] = (uint8_t)fp8_code(m3, f2, rw1);
  }
}

// ---------------------------------------------------------------------------------------
// BLAKE3, one 64-byte keyed block: the jackpot ticket
// ---------------------------------------------------------------------------------------

#define FP8_ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define FP8_G(a, b, c, d, x, y)                                                                  \
  a = a + b + (x); d = FP8_ROTR(d ^ a, 16); c = c + d; b = FP8_ROTR(b ^ c, 12);                  \
  a = a + b + (y); d = FP8_ROTR(d ^ a, 8);  c = c + d; b = FP8_ROTR(b ^ c, 7);

// keyed_hash(key, msg) for a 64-byte msg: one compression with CHUNK_START | CHUNK_END |
// ROOT | KEYED_HASH. out is the 32-byte digest as 8 little-endian words.
__device__ __forceinline__ void fp8_ticket(const uint32_t *key, const uint32_t *msg, uint32_t *out) {
  uint32_t s0 = key[0], s1 = key[1], s2 = key[2], s3 = key[3];
  uint32_t s4 = key[4], s5 = key[5], s6 = key[6], s7 = key[7];
  uint32_t s8 = 0x6A09E667u, s9 = 0xBB67AE85u, s10 = 0x3C6EF372u, s11 = 0xA54FF53Au;
  uint32_t s12 = 0, s13 = 0, s14 = 64, s15 = 1u | 2u | 8u | 16u;
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

// ---------------------------------------------------------------------------------------
// The search
// ---------------------------------------------------------------------------------------

// Block (x, y) computes rows row_a + 64x .. of A against rows row_b + 128y .. of B. x runs
// fastest, so the blocks in flight share a strip of B and all of A stays in the L2 cache.
//
// a, b       the noised operands, rows x k E4M3 codes, row-major
// h          tile height, 4, 8 or 16; the tile is h x 256/h
// keybound   the jackpot key (8 words) then the winning bound (8 words, little endian)
// hits       pairs (tile row, tile column) of winning tiles; count is how many were found
// tickets    if set, every tile's 32-byte ticket at (tile row * tiles_b + tile column)
// dump       if set, this block's 64 x 128 FP32 cells, row-major (launch one block)
extern "C" __global__ void fp8_search_kernel(const uint8_t *a, const uint8_t *b, uint32_t k,
                                             uint32_t h, uint32_t row_a, uint32_t row_b,
                                             const uint32_t *keybound, uint32_t *hits,
                                             uint32_t *count, uint32_t max_hits,
                                             uint8_t *tickets, uint32_t tiles_b, float *dump) {
  // The staged operands and, after the last chunk, the cells: the larger is the cells.
  __shared__ __align__(16) uint8_t sbuf[FP8_ROWS_A * FP8_ROWS_B * 4];
  __shared__ uint32_t swords[FP8_WARPS * 8 * 16];
  const uint32_t tid = threadIdx.x;
  const uint32_t warp = tid >> 5, lane = tid & 31;
  const uint32_t g = lane >> 2, t = lane & 3;
  const uint32_t base_a = row_a + blockIdx.x * FP8_ROWS_A;
  const uint32_t base_b = row_b + blockIdx.y * FP8_ROWS_B;
  const uint32_t *sa = reinterpret_cast<const uint32_t *>(sbuf);
  const uint32_t *sb = reinterpret_cast<const uint32_t *>(sbuf + FP8_ROWS_A * FP8_KC);

  float acc[FP8_NB * 4];
#pragma unroll
  for (int i = 0; i < FP8_NB * 4; i++) acc[i] = 0.0f;

  for (uint32_t k0 = 0; k0 < k; k0 += FP8_KC) {
    const uint32_t kc = (k - k0 < FP8_KC) ? (k - k0) : FP8_KC;
    // Stage kc columns of the block's 64 + 128 rows, 16 bytes a move.
    for (uint32_t i = tid; i < (FP8_ROWS_A + FP8_ROWS_B) * (FP8_KC / 16); i += FP8_THREADS) {
      const uint32_t row = i / (FP8_KC / 16), part = i % (FP8_KC / 16);
      if (part * 16 >= kc) continue;
      const uint8_t *src = row < FP8_ROWS_A
          ? a + (size_t)(base_a + row) * k + k0 + part * 16
          : b + (size_t)(base_b + row - FP8_ROWS_A) * k + k0 + part * 16;
      *reinterpret_cast<uint4 *>(sbuf + row * FP8_KC + part * 16) =
          *reinterpret_cast<const uint4 *>(src);
    }
    __syncthreads();
    for (uint32_t ks = 0; ks < kc / 32; ks++) {
      const uint32_t col = ks * 8 + t;  // in 4-byte words
      const uint32_t ra = (warp * 16 + g) * (FP8_KC / 4) + col;
      const uint32_t a0 = sa[ra], a1 = sa[ra + 8 * (FP8_KC / 4)];
      const uint32_t a2 = sa[ra + 4], a3 = sa[ra + 8 * (FP8_KC / 4) + 4];
#pragma unroll
      for (int nb = 0; nb < FP8_NB; nb++) {
        const uint32_t rb = (nb * 8 + g) * (FP8_KC / 4) + col;
        fp8_mma(acc[nb * 4], acc[nb * 4 + 1], acc[nb * 4 + 2], acc[nb * 4 + 3],
                a0, a1, a2, a3, sb[rb], sb[rb + 4]);
      }
    }
    __syncthreads();
  }

  // The cells to shared memory, row-major, 64 x 128.
  float *cells = reinterpret_cast<float *>(sbuf);
#pragma unroll
  for (int nb = 0; nb < FP8_NB; nb++) {
    const uint32_t at = (warp * 16 + g) * FP8_ROWS_B + nb * 8 + 2 * t;
    cells[at] = acc[nb * 4];
    cells[at + 1] = acc[nb * 4 + 1];
    cells[at + 8 * FP8_ROWS_B] = acc[nb * 4 + 2];
    cells[at + 8 * FP8_ROWS_B + 1] = acc[nb * 4 + 3];
  }
  __syncthreads();
  if (dump) {
    for (uint32_t i = tid; i < FP8_ROWS_A * FP8_ROWS_B; i += FP8_THREADS) dump[i] = cells[i];
  }

  // xor_fold_extract: a tile, row-major, is 16 runs of 16 cells, so a run is 16 consecutive
  // columns of one row. A warp's 16 x 128 cells are 128 runs and 8 tiles; each lane folds 4.
  const uint32_t cpr = 16 / h;       // runs across one tile
  const uint32_t across = h / 2;     // tiles across the warp's 128 columns
#pragma unroll
  for (int q = 0; q < 4; q++) {
    const uint32_t run = lane + 32 * q;
    const uint32_t row = run >> 3, cr = run & 7;
    const uint32_t *p = reinterpret_cast<const uint32_t *>(cells) + (warp * 16 + row) * FP8_ROWS_B + cr * 16;
    uint32_t v = 0;
#pragma unroll
    for (int i = 0; i < 16; i++) {
      v = v * 0x9E3779B1u + p[i];
      v = (v << 13) | (v >> 19);
    }
    const uint32_t tile = (row / h) * across + cr / cpr;
    swords[(warp * 8 + tile) * 16 + (row % h) * cpr + cr % cpr] = v;
  }
  __syncwarp();
  if (lane < 8) {
    uint32_t ticket[8];
    fp8_ticket(keybound, swords + (warp * 8 + lane) * 16, ticket);
    const uint32_t tile_a = (base_a + warp * 16) / h + lane / across;
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

// ---------------------------------------------------------------------------------------
// Host half
// ---------------------------------------------------------------------------------------

namespace {

struct Ctx {
  bool open = false;
  uint32_t m = 0, n = 0, k = 0, h = 0;
  uint32_t max_rows = 0, max_hits = 0;
  uint8_t *codes[2] = {nullptr, nullptr};
  int8_t *ints = nullptr;
  uint32_t *e = nullptr, *f = nullptr, *rewrite = nullptr, *keybound = nullptr;
  uint32_t *hits = nullptr, *count = nullptr;
  float *signal = nullptr, *beta = nullptr, *dump = nullptr;
  uint8_t *tickets = nullptr;
  char err[256] = "";
};

Ctx ctx;

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
  release(ctx.codes[0]); release(ctx.codes[1]); release(ctx.ints);
  release(ctx.e); release(ctx.f); release(ctx.rewrite); release(ctx.keybound);
  release(ctx.hits); release(ctx.count); release(ctx.signal); release(ctx.beta);
  release(ctx.dump); release(ctx.tickets);
}

#define FP8_TRY(what, call)                               \
  do {                                                    \
    const cudaError_t e_ = (call);                        \
    if (e_ != cudaSuccess) return fail(what, e_);         \
  } while (0)

uint32_t rows_of(int side) { return side == 0 ? ctx.m : ctx.n; }

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
  ctx.open = true;
  return 0;
}

// Allocate for a job shape. m is a multiple of 64, n of 128, k of 32; h is 4, 8 or 16.
int fp8_configure(uint32_t m, uint32_t n, uint32_t k, uint32_t h, uint32_t max_hits) {
  if (!ctx.open) return fail("fp8_open was not called");
  if (m == 0 || n == 0 || m % FP8_ROWS_A || n % FP8_ROWS_B) return fail("m must be a multiple of 64 and n of 128");
  if (k < 1024 || k > 65536 || k % 32) return fail("k must be a multiple of 32 in 1024..65536");
  if (h != 4 && h != 8 && h != 16) return fail("the tile height must be 4, 8 or 16");
  if (max_hits == 0) return fail("max_hits must be positive");
  release_all();
  ctx.m = m; ctx.n = n; ctx.k = k; ctx.h = h; ctx.max_hits = max_hits;
  ctx.max_rows = m > n ? m : n;
  const size_t tiles = (size_t)(m / h) * (n / (256 / h));
  FP8_TRY("alloc A'", cudaMalloc(&ctx.codes[0], (size_t)m * k));
  FP8_TRY("alloc B'", cudaMalloc(&ctx.codes[1], (size_t)n * k));
  FP8_TRY("alloc values", cudaMalloc(&ctx.ints, (size_t)ctx.max_rows * k));
  FP8_TRY("alloc E", cudaMalloc(&ctx.e, (size_t)ctx.max_rows * 32));
  FP8_TRY("alloc F", cudaMalloc(&ctx.f, (size_t)k * 32));
  FP8_TRY("alloc signal", cudaMalloc(&ctx.signal, (size_t)ctx.max_rows * (k / 8) * 4));
  FP8_TRY("alloc beta", cudaMalloc(&ctx.beta, (size_t)ctx.max_rows * 4));
  FP8_TRY("alloc rewrite", cudaMalloc(&ctx.rewrite, (size_t)ctx.max_rows * 4));
  FP8_TRY("alloc key", cudaMalloc(&ctx.keybound, 64));
  FP8_TRY("alloc hits", cudaMalloc(&ctx.hits, (size_t)max_hits * 8));
  FP8_TRY("alloc count", cudaMalloc(&ctx.count, 4));
  FP8_TRY("alloc cells", cudaMalloc(&ctx.dump, (size_t)FP8_ROWS_A * FP8_ROWS_B * 4));
  // Every tile's ticket, for the verify tool; only small shapes ask for it.
  if (tiles <= (1u << 20)) FP8_TRY("alloc tickets", cudaMalloc(&ctx.tickets, tiles * 32));
  return 0;
}

// Load E4M3 codes as they are (side 0 is A', 1 is B'). For the verify tool.
int fp8_set_codes(int side, const uint8_t *codes) {
  if (!ctx.codes[0]) return fail("fp8_configure was not called");
  FP8_TRY("upload codes", cudaMemcpy(ctx.codes[side], codes, (size_t)rows_of(side) * ctx.k, cudaMemcpyHostToDevice));
  return 0;
}

int fp8_get_codes(int side, uint32_t row0, uint32_t rows, uint8_t *out) {
  if (!ctx.codes[0]) return fail("fp8_configure was not called");
  if (row0 + rows > rows_of(side)) return fail("rows out of range");
  FP8_TRY("download codes", cudaMemcpy(out, ctx.codes[side] + (size_t)row0 * ctx.k, (size_t)rows * ctx.k, cudaMemcpyDeviceToHost));
  return 0;
}

// Noise one operand on the card: ints rows x k, e rows x 32, f k x 32, signal rows x k/8,
// and a row's beta and rewrite constants (all from fp8Host.rowParams).
int fp8_noise(int side, const int8_t *ints, const uint8_t *e, const uint8_t *f,
              const float *signal, const float *beta, const uint32_t *rewrite) {
  if (!ctx.codes[0]) return fail("fp8_configure was not called");
  const uint32_t rows = rows_of(side);
  FP8_TRY("upload values", cudaMemcpy(ctx.ints, ints, (size_t)rows * ctx.k, cudaMemcpyHostToDevice));
  FP8_TRY("upload E", cudaMemcpy(ctx.e, e, (size_t)rows * 32, cudaMemcpyHostToDevice));
  FP8_TRY("upload F", cudaMemcpy(ctx.f, f, (size_t)ctx.k * 32, cudaMemcpyHostToDevice));
  FP8_TRY("upload signal", cudaMemcpy(ctx.signal, signal, (size_t)rows * (ctx.k / 8) * 4, cudaMemcpyHostToDevice));
  FP8_TRY("upload beta", cudaMemcpy(ctx.beta, beta, (size_t)rows * 4, cudaMemcpyHostToDevice));
  FP8_TRY("upload rewrite", cudaMemcpy(ctx.rewrite, rewrite, (size_t)rows * 4, cudaMemcpyHostToDevice));
  fp8_noise_kernel<<<dim3(rows / 16, (ctx.k + 255) / 256), 32>>>(
      ctx.ints, ctx.e, ctx.f, ctx.signal, ctx.beta, ctx.rewrite, ctx.codes[side], ctx.k);
  FP8_TRY("noise launch", cudaGetLastError());
  FP8_TRY("noise", cudaDeviceSynchronize());
  return 0;
}

// Search every tile of the loaded A' and B'. key is the 32-byte jackpot key, bound the
// 32-byte little-endian winning bound. hits receives up to max_hits (tile row, tile column)
// pairs and *count how many tiles won (it can exceed max_hits). tickets, if not null,
// receives every tile's ticket (small shapes only).
int fp8_search(const uint8_t *key, const uint8_t *bound, uint32_t *hits, uint32_t *count, uint8_t *tickets) {
  if (!ctx.codes[0]) return fail("fp8_configure was not called");
  if (tickets && !ctx.tickets) return fail("this shape is too large to return every ticket");
  uint8_t kb[64];
  memcpy(kb, key, 32);
  memcpy(kb + 32, bound, 32);
  FP8_TRY("upload key", cudaMemcpy(ctx.keybound, kb, 64, cudaMemcpyHostToDevice));
  FP8_TRY("clear count", cudaMemset(ctx.count, 0, 4));
  const uint32_t tiles_b = ctx.n / (256 / ctx.h);
  fp8_search_kernel<<<dim3(ctx.m / FP8_ROWS_A, ctx.n / FP8_ROWS_B), FP8_THREADS>>>(
      ctx.codes[0], ctx.codes[1], ctx.k, ctx.h, 0, 0, ctx.keybound, ctx.hits, ctx.count,
      ctx.max_hits, tickets ? ctx.tickets : nullptr, tiles_b, nullptr);
  FP8_TRY("search launch", cudaGetLastError());
  FP8_TRY("search", cudaDeviceSynchronize());
  FP8_TRY("download count", cudaMemcpy(count, ctx.count, 4, cudaMemcpyDeviceToHost));
  const uint32_t got = *count < ctx.max_hits ? *count : ctx.max_hits;
  if (got) FP8_TRY("download hits", cudaMemcpy(hits, ctx.hits, (size_t)got * 8, cudaMemcpyDeviceToHost));
  if (tickets) {
    FP8_TRY("download tickets", cudaMemcpy(tickets, ctx.tickets, (size_t)(ctx.m / ctx.h) * tiles_b * 32, cudaMemcpyDeviceToHost));
  }
  return 0;
}

// The FP32 cells of one block: rows row_a..+64 of A' by rows row_b..+128 of B', row-major.
// For the verify tool, which compares their bits with the CPU reference.
int fp8_cells(uint32_t row_a, uint32_t row_b, float *out) {
  if (!ctx.codes[0]) return fail("fp8_configure was not called");
  if (row_a % FP8_ROWS_A || row_b % FP8_ROWS_B || row_a >= ctx.m || row_b >= ctx.n) return fail("block out of range");
  uint8_t kb[64];
  memset(kb, 0, sizeof(kb));
  FP8_TRY("upload key", cudaMemcpy(ctx.keybound, kb, 64, cudaMemcpyHostToDevice));
  fp8_search_kernel<<<dim3(1, 1), FP8_THREADS>>>(
      ctx.codes[0], ctx.codes[1], ctx.k, ctx.h, row_a, row_b, ctx.keybound, ctx.hits, ctx.count,
      0, nullptr, 1, ctx.dump);
  FP8_TRY("cells launch", cudaGetLastError());
  FP8_TRY("cells", cudaDeviceSynchronize());
  FP8_TRY("download cells", cudaMemcpy(out, ctx.dump, (size_t)FP8_ROWS_A * FP8_ROWS_B * 4, cudaMemcpyDeviceToHost));
  return 0;
}

void fp8_close() {
  release_all();
  ctx.m = ctx.n = ctx.k = ctx.h = 0;
}

}  // extern "C"
