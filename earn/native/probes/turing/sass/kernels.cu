// The inner loop of Turing's B-direct fold (src/pearl_fold_bd.cuh), alone, for
// sm_75. Three kernels, all at the fold's geometry: 256 threads, eight 64x64
// warp tiles over a 128x256 tile, one block an SM, 40 KB of dynamic shared (so
// L1 keeps the fold's 32 KB).
//
//   sass_ceiling  IMMA.8816 from registers, no memory: the denominator. The
//                 same 128 IMMA a k-step, on the same 64 accumulator pairs and
//                 operand registers as the fold's k-step.
//   sass_ptxas    one k-step of the fold, 64 to a tile: B fragments straight
//                 from global (fragment-ordered, ld.global.nc 128-bit, one
//                 k-step ahead), A from shared with ldmatrix, the fold's
//                 swizzle and stage layout, 128 IMMA into 128 accumulators.
//                 Compiled by ptxas: the baseline.
//   sass_hand1-3  the same source as sass_ptxas, so the cubin carries its
//   sass_hceil    parameters and attributes. hand.py replaces their code
//                 with the hand-scheduled loop (B 1, 2 or 3 k-steps ahead)
//                 and a hand-written ceiling before anything runs them.
//
// What is not here: A's staging copies, the chunk barrier, the readout and the
// hash. A sits in shared for the whole run (two 16 KB stages, chunk c reading
// stage c & 1), so only B moves. The seams cost what they cost on top of this.
//
// B is the fold's footprint: ncg column groups of 512 KB (256 columns x k), a
// band that stays in L2. Block b's tile i reads column group (b + i) % ncg,
// so about SMs / ncg blocks read each group at once, as in the fold's band
// walk. With kmask = 1 every B load after the first two of a tile is an L1
// hit (the fold's PEARL_ABLATE_BHIT); kmask = 63 is the real loop.
//
// The accumulators are never reset, so a block's result is the sum over its
// tiles, and every block writes all of it: out[(block * 128 + i) * 256 + tid],
// i = mb * 32 + nb * 4 + c. Each warp's lane 0 writes clock64 before and after
// its tile loop to cyc[(block * 8 + warp) * 2 + 0/1].

#include <cstdint>

#define SK 2048u                       // k, the mainnet's
#define KSTEPS (SK / 32u)              // 64 k-steps a tile
#define FRAG_BLOCK ((SK / 32u) * 512u) // one 16-column block of B': 32 KB
#define ABUF 16384u                    // one A stage: 128 rows x 128 bytes

// Turing's int8 mma is m8n8k16. The fold keeps m16n8k32's register layout and
// issues four of them (pearl_kernel.cu, pearl_mma_m16n8k32).
__device__ __forceinline__ void mma16832(int32_t &c0, int32_t &c1, int32_t &c2, int32_t &c3,
                                         uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%4}, {%8}, {%0,%1};\n"
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%6}, {%9}, {%0,%1};\n"
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%2,%3}, {%5}, {%8}, {%2,%3};\n"
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%2,%3}, {%7}, {%9}, {%2,%3};\n"
      : "+&r"(c0), "+&r"(c1), "+&r"(c2), "+&r"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldsm4(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3,
                                      uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(a));
}

// The fold's B load (pearl_bd_ldg): volatile and ordered against memory.
__device__ __forceinline__ int4 ldg_nc(const int8_t *p) {
  int4 v;
  asm volatile("ld.global.nc.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}

__device__ __forceinline__ long long clk64() {
  long long c;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(c));
  return c;
}

// Block b's two A stages: rows b * 128 .. + 127 of A (256 bytes a row, the two
// chunks), stored as the fold stores them: unit q of row r at q ^ (r & 7).
__device__ __forceinline__ void stage_a(const int8_t *__restrict__ A, uint32_t sA) {
  for (uint32_t i = threadIdx.x; i < 2u * 128u * 8u; i += 256u) {
    const uint32_t st = i >> 10, r = (i >> 3) & 127u, q = i & 7u;
    const int4 v = *reinterpret_cast<const int4 *>(A + ((size_t)blockIdx.x * 128u + r) * 256u
                                                    + st * 128u + q * 16u);
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(sA + st * ABUF + r * 128u
                                                                + ((q ^ (r & 7u)) << 4)),
                 "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
  }
}

#define FOLD_BODY                                                                              \
  extern __shared__ __align__(128) uint32_t smem_u32[];                                        \
  const uint32_t sA = (uint32_t)__cvta_generic_to_shared(smem_u32);                            \
  const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;                            \
  const uint32_t wr = warp >> 2, wc = warp & 3u;                                               \
  stage_a(A, sA);                                                                              \
  __syncthreads();                                                                             \
  const uint32_t alrow = (lane & 7u) + ((lane >> 3) & 1u) * 8u;                                \
  const uint32_t albyte = ((lane >> 4) & 1u) * 16u;                                            \
  const uint32_t swz = (lane & 7u) << 4;                                                       \
  const uint32_t aLane0 =                                                                      \
      sA + (wr * 64u + alrow) * 128u + ((albyte ^ swz) & 0x10u) + (swz & 0x60u);              \
  int32_t acc[4][8][4];                                                                        \
  _Pragma("unroll") for (uint32_t mb = 0; mb < 4u; mb++)                                       \
  _Pragma("unroll") for (uint32_t nb = 0; nb < 8u; nb++)                                       \
  _Pragma("unroll") for (uint32_t i = 0; i < 4u; i++) acc[mb][nb][i] = 0;                      \
  const int8_t *bsrc = B + (size_t)((blockIdx.x % ncg) * 16u + wc * 4u) * FRAG_BLOCK + lane * 16u; \
  int4 bf[2][4];                                                                               \
  _Pragma("unroll") for (uint32_t j = 0; j < 4u; j++) bf[0][j] = ldg_nc(bsrc + j * FRAG_BLOCK); \
  const long long t0 = clk64();                                                                \
  for (uint32_t tile = 0; tile < tiles; tile++) {                                              \
    const uint32_t cgn = (blockIdx.x + tile + 1u) % ncg;                                       \
    const int8_t *bnext =                                                                      \
        B + (size_t)(cgn * 16u + wc * 4u) * FRAG_BLOCK + lane * 16u;                           \
    for (uint32_t ch = 0; ch < KSTEPS / 4u; ch++) {                                            \
      const uint32_t buf = (ch & 1u) * ABUF;                                                   \
      _Pragma("unroll") for (uint32_t t = 0; t < 4u; t++) {                                    \
        const uint32_t s = ch * 4u + t;                                                        \
        const int8_t *bs = s + 1u < KSTEPS ? bsrc + ((s + 1u) & kmask) * 512u : bnext;         \
        _Pragma("unroll") for (uint32_t j = 0; j < 4u; j++)                                    \
          bf[(t + 1u) & 1u][j] = ldg_nc(bs + j * FRAG_BLOCK);                                  \
        uint32_t af[4][4];                                                                     \
        _Pragma("unroll") for (uint32_t mb = 0; mb < 4u; mb++)                                 \
          ldsm4(af[mb][0], af[mb][1], af[mb][2], af[mb][3],                                    \
                ((aLane0 + buf) ^ (t * 32u)) + mb * 16u * 128u);                               \
        _Pragma("unroll") for (uint32_t j = 0; j < 4u; j++) {                                  \
          const uint32_t b0 = (uint32_t)bf[t & 1u][j].x, b1 = (uint32_t)bf[t & 1u][j].y;       \
          const uint32_t b2 = (uint32_t)bf[t & 1u][j].z, b3 = (uint32_t)bf[t & 1u][j].w;       \
          _Pragma("unroll") for (uint32_t mb = 0; mb < 4u; mb++)                               \
            mma16832(acc[mb][2u * j][0], acc[mb][2u * j][1], acc[mb][2u * j][2],               \
                     acc[mb][2u * j][3], af[mb][0], af[mb][1], af[mb][2], af[mb][3], b0, b1);  \
          _Pragma("unroll") for (uint32_t mb = 0; mb < 4u; mb++)                               \
            mma16832(acc[mb][2u * j + 1u][0], acc[mb][2u * j + 1u][1],                         \
                     acc[mb][2u * j + 1u][2], acc[mb][2u * j + 1u][3], af[mb][0], af[mb][1],   \
                     af[mb][2], af[mb][3], b2, b3);                                            \
        }                                                                                      \
      }                                                                                        \
    }                                                                                          \
    bsrc = bnext;                                                                              \
  }                                                                                            \
  const long long t1 = clk64();                                                                \
  if (lane == 0u) {                                                                            \
    cyc[(blockIdx.x * 8u + warp) * 2u] = t0;                                                   \
    cyc[(blockIdx.x * 8u + warp) * 2u + 1u] = t1;                                              \
  }                                                                                            \
  _Pragma("unroll") for (uint32_t mb = 0; mb < 4u; mb++)                                       \
  _Pragma("unroll") for (uint32_t nb = 0; nb < 8u; nb++)                                       \
  _Pragma("unroll") for (uint32_t i = 0; i < 4u; i++)                                          \
    out[((size_t)blockIdx.x * 128u + mb * 32u + nb * 4u + i) * 256u + threadIdx.x] =           \
        acc[mb][nb][i];

extern "C" __global__ __launch_bounds__(256, 1) void sass_ptxas(
    const int8_t *__restrict__ A, const int8_t *__restrict__ B, int32_t *__restrict__ out,
    long long *__restrict__ cyc, uint32_t tiles, uint32_t ncg, uint32_t kmask) {
  FOLD_BODY
}

// Slots for hand.py's kernels: copies of sass_ptxas, so each carries the
// parameters, shared size and attributes; hand.py replaces their code.
#define HAND_SLOT(name)                                                                        \
  extern "C" __global__ __launch_bounds__(256, 1) void name(                                   \
      const int8_t *__restrict__ A, const int8_t *__restrict__ B, int32_t *__restrict__ out,   \
      long long *__restrict__ cyc, uint32_t tiles, uint32_t ncg, uint32_t kmask) {             \
    FOLD_BODY                                                                                  \
  }
HAND_SLOT(sass_hand1)
HAND_SLOT(sass_hand2)
HAND_SLOT(sass_hand3)
HAND_SLOT(sass_hceil)

// The ceiling: the fold's k-step with its loads taken out. The operands are
// registers that change once a tile (so nothing is loop-invariant), the
// accumulators and the IMMA are the fold's. Same signature, same output.
extern "C" __global__ __launch_bounds__(256, 1) void sass_ceiling(
    const int8_t *__restrict__ A, const int8_t *__restrict__ B, int32_t *__restrict__ out,
    long long *__restrict__ cyc, uint32_t tiles, uint32_t ncg, uint32_t kmask) {
  (void)A; (void)B; (void)ncg; (void)kmask;
  const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
  int32_t acc[4][8][4];
#pragma unroll
  for (uint32_t mb = 0; mb < 4u; mb++)
#pragma unroll
    for (uint32_t nb = 0; nb < 8u; nb++)
#pragma unroll
      for (uint32_t i = 0; i < 4u; i++) acc[mb][nb][i] = 0;
  uint32_t af[4][4], bf[4][4];
#pragma unroll
  for (uint32_t i = 0; i < 4u; i++)
#pragma unroll
    for (uint32_t j = 0; j < 4u; j++) {
      af[i][j] = 0x01010101u * (threadIdx.x + i * 4u + j);
      bf[i][j] = 0x01030507u ^ (threadIdx.x * 7u + i * 4u + j);
    }
  const long long t0 = clk64();
  for (uint32_t tile = 0; tile < tiles; tile++) {
    for (uint32_t s = 0; s < KSTEPS; s++) {
#pragma unroll
      for (uint32_t j = 0; j < 4u; j++) {
#pragma unroll
        for (uint32_t mb = 0; mb < 4u; mb++)
          mma16832(acc[mb][2u * j][0], acc[mb][2u * j][1], acc[mb][2u * j][2],
                   acc[mb][2u * j][3], af[mb][0], af[mb][1], af[mb][2], af[mb][3], bf[j][0],
                   bf[j][1]);
#pragma unroll
        for (uint32_t mb = 0; mb < 4u; mb++)
          mma16832(acc[mb][2u * j + 1u][0], acc[mb][2u * j + 1u][1], acc[mb][2u * j + 1u][2],
                   acc[mb][2u * j + 1u][3], af[mb][0], af[mb][1], af[mb][2], af[mb][3],
                   bf[j][2], bf[j][3]);
      }
    }
#pragma unroll
    for (uint32_t i = 0; i < 4u; i++)
#pragma unroll
      for (uint32_t j = 0; j < 4u; j++) {
        af[i][j] += 0x01010101u;
        bf[i][j] ^= 0x00010001u << (j & 1u);
      }
  }
  const long long t1 = clk64();
  if (lane == 0u) {
    cyc[(blockIdx.x * 8u + warp) * 2u] = t0;
    cyc[(blockIdx.x * 8u + warp) * 2u + 1u] = t1;
  }
#pragma unroll
  for (uint32_t mb = 0; mb < 4u; mb++)
#pragma unroll
    for (uint32_t nb = 0; nb < 8u; nb++)
#pragma unroll
      for (uint32_t i = 0; i < 4u; i++)
        out[((size_t)blockIdx.x * 128u + mb * 32u + nb * 4u + i) * 256u + threadIdx.x] =
            acc[mb][nb][i];
}
