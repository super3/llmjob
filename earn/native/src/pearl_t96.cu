// The RTX 20 (sm_75) fold with a 96 x 64 warp tile. NOT part of pearl_kernel.cu and not linked into the addon:
// this file is compiled alone into a cubin (nvcc -arch=sm_75 -cubin), probes/turing/build_t96.py replaces the
// code of pearl_t96_loop with a loop written by hand (probes/turing/t96gen.py), and the host loads the result at
// run time through the driver (src/pearl_host_t96.inc). The release kernels are untouched by it.
//
// Why by hand: 192 sums a lane need about 274 registers under ptxas (goal-nv20s) and a kernel has 255. The hand
// loop keeps each operand in one register quad (single-buffered) and fits: 192 sums + 40 operand registers.
//
// Kernels:
//   pearl_t96_fold   the fold the miner runs. Its k loop, pearl_t96_loop, is compiled here from C: that code is
//                    the room and the register set for the hand loop, and is never run as compiled.
//   t96_ref_fold     the same kernel with the C loop left as ptxas made it (t96_ref_loop). Slow (it spills) and
//                    right: the host runs a few tiles through both when a card opens and compares every
//                    transcript word (pearl_host_t96.inc, t96_startup_check).
//   t96_aux_mat      a noised operand in this fold's fragment order.
// The helpers have no "pearl" in their names on purpose: the build patches the one function that has.
//
// The fold's contract (the same tile as the rotation fold's, pearl_rot.cu):
//   A tile is 192 rows by 256 columns, 256 threads: 2 row slots of 96 rows by 4 column slots of 64 columns.
//   Thread (warp w, lane l), g = l >> 2, t = l & 3, holds the sums of rows 96 (w >> 2) + g + 8 i (i 0 .. 11) and
//   columns 64 (w & 3) + 2 t + {0, 1} + 8 j (j 0 .. 7) of the tile: one region of the pattern rows {0, 8, .. 88}
//   by columns {0, 1, 8, 9, .. 56, 57}, 12 by 16. Word c of its transcript is the XOR of its 192 sums after chunk
//   c (k 128 c + 127), as the reference's (src/shared/miner/reference.js, foldTranscript).
//   Tile v of a launch is tile0 + v of the salt, walked in bands of column groups (t96_coords, here and in
//   the host). A hit's index is 256 v + thread.
//
// The operands' order (t96_aux_mat). One LDG.128 gives a lane four IMMA.8816 operand words of ONE k16 half, so
// a register quad dies as a unit: plane P is rows 32 P .. 32 P + 31 (four m8 blocks; for B' four n8 blocks of
// columns), [k / 16][lane 32][block 4][4 bytes]: byte (row r, kk) is at
//   (r >> 5) * 32 k + (kk >> 4) * 512 + ((r & 7) * 4 + ((kk >> 2) & 3)) * 16 + ((r >> 3) & 3) * 4 + (kk & 3).
// So lane 4 g + t's sixteen bytes of half-step hs are rows g, g + 8, g + 16, g + 24 of the plane at k bytes
// 16 hs + 4 t .. + 3, which is the m8n8k16 A operand of each of the four blocks (B's the same, a column for a
// row). A k-step (k 32) of a warp is 6 loads of A and 4 of B, 192 IMMA.

#include <cuda_runtime.h>
#include <stdint.h>

#include "pearl_config.h"

#define T96_THREADS 256u
#define T96_K 2048u
#define T96_PLANE (32u * T96_K)   // bytes of a plane of 32 rows: 64 KB
#define T96_KSTEPS (T96_K / 32u)  // 64
#define T96_CHUNKS 16u
// Shared memory: the transcripts, [buffer 12][chunk 16][lane 32] words. A warp that hashes (T96_HASHER) has
// one buffer, at 2048 (w & 3). Its partner on the scheduler, whose words it reads after the tile's barrier,
// has two, used in turn, at 8192 + 2048 (2 (w & 3) + (tile count & 1)): the partner is already storing the
// next tile's words while the hasher reads.
#define T96_TRB 2048u
#define T96_SMEM (12u * T96_TRB)
// Timing only (a race's -DT96_PASSES=0 or 1): how many of the hash's two passes run. A build that mines has 2.
#ifndef T96_PASSES
#define T96_PASSES 2u
#endif
// The hand loop's second copy of the constants it bakes in: build_t96.py refuses a cubin whose values differ.
extern "C" __device__ const uint32_t t96_shape[4] = {T96_K, T96_PLANE, T96_TRB, T96_SMEM};

static __device__ __forceinline__ void t96_g(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d, uint32_t mx,
                                             uint32_t my) {
  a = a + b + mx;
  d = ((d ^ a) >> 16) | ((d ^ a) << 16);
  c = c + d;
  b = ((b ^ c) >> 12) | ((b ^ c) << 20);
  a = a + b + my;
  d = ((d ^ a) >> 8) | ((d ^ a) << 24);
  c = c + d;
  b = ((b ^ c) >> 7) | ((b ^ c) << 25);
}
// BLAKE3's keyed hash of one 64-byte block: one compression, counter 0, flags chunk start, chunk end, root, keyed.
static __device__ __forceinline__ void t96_blake3_keyed64(const uint32_t key[8], const uint32_t block[16],
                                                          uint32_t out[8]) {
  uint32_t s[16], m[16];
#pragma unroll
  for (int i = 0; i < 8; i++) s[i] = key[i];
  s[8] = 0x6A09E667u;
  s[9] = 0xBB67AE85u;
  s[10] = 0x3C6EF372u;
  s[11] = 0xA54FF53Au;
  s[12] = 0u;
  s[13] = 0u;
  s[14] = 64u;
  s[15] = 1u | 2u | 8u | 16u;
#pragma unroll
  for (int i = 0; i < 16; i++) m[i] = block[i];
#pragma unroll
  for (int round = 0; round < 7; round++) {
    t96_g(s[0], s[4], s[8], s[12], m[0], m[1]);
    t96_g(s[1], s[5], s[9], s[13], m[2], m[3]);
    t96_g(s[2], s[6], s[10], s[14], m[4], m[5]);
    t96_g(s[3], s[7], s[11], s[15], m[6], m[7]);
    t96_g(s[0], s[5], s[10], s[15], m[8], m[9]);
    t96_g(s[1], s[6], s[11], s[12], m[10], m[11]);
    t96_g(s[2], s[7], s[8], s[13], m[12], m[13]);
    t96_g(s[3], s[4], s[9], s[14], m[14], m[15]);
    if (round < 6) {
      // The message permutation {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8}, with literal indices.
      const uint32_t p0 = m[2], p1 = m[6], p2 = m[3], p3 = m[10];
      const uint32_t p4 = m[7], p5 = m[0], p6 = m[4], p7 = m[13];
      const uint32_t p8 = m[1], p9 = m[11], p10 = m[12], p11 = m[5];
      const uint32_t p12 = m[9], p13 = m[14], p14 = m[15], p15 = m[8];
      m[0] = p0; m[1] = p1; m[2] = p2; m[3] = p3;
      m[4] = p4; m[5] = p5; m[6] = p6; m[7] = p7;
      m[8] = p8; m[9] = p9; m[10] = p10; m[11] = p11;
      m[12] = p12; m[13] = p13; m[14] = p14; m[15] = p15;
    }
  }
#pragma unroll
  for (int i = 0; i < 8; i++) out[i] = s[i] ^ s[i + 8];
}

// One tile of one thread, in C: 64 k-steps, each its low k16 half and then its high one, and after every fourth
// the XOR of the lane's 192 sums into the chunk's word. ao and bo are the thread's byte offsets of k-step 0 of
// its first plane in A' and B'; tr is the shared-memory address of its word of chunk 0 (chunk c is 128 c on).
// A chunk's four k-steps are written out so that the compiled function is longer than the hand loop.
static __device__ __forceinline__ void t96_loop_c(const int8_t *__restrict__ Ap, const int8_t *__restrict__ Bp,
                                                  uint32_t ao, uint32_t bo, uint32_t tr) {
  int32_t acc[12][8][2];
#pragma unroll
  for (uint32_t i = 0; i < 12u; i++)
#pragma unroll
    for (uint32_t j = 0; j < 8u; j++) acc[i][j][0] = acc[i][j][1] = 0;
  auto ld = [](const int8_t *p) -> int4 {
    int4 v;
    asm volatile("ld.global.nc.v4.s32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p)
                 : "memory");
    return v;
  };
  auto half = [&](uint32_t off) {
    int4 a[3], b[2];
#pragma unroll
    for (uint32_t q = 0; q < 3u; q++) a[q] = ld(Ap + ao + q * T96_PLANE + off);
#pragma unroll
    for (uint32_t q = 0; q < 2u; q++) b[q] = ld(Bp + bo + q * T96_PLANE + off);
#pragma unroll
    for (uint32_t i = 0; i < 12u; i++) {
      const int4 av = a[i / 4u];
      const uint32_t aw = (uint32_t)(i % 4u == 0u ? av.x : i % 4u == 1u ? av.y : i % 4u == 2u ? av.z : av.w);
#pragma unroll
      for (uint32_t j = 0; j < 8u; j++) {
        const int4 bv = b[j / 4u];
        const uint32_t bw = (uint32_t)(j % 4u == 0u ? bv.x : j % 4u == 1u ? bv.y : j % 4u == 2u ? bv.z : bv.w);
        asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                     : "+r"(acc[i][j][0]), "+r"(acc[i][j][1])
                     : "r"(aw), "r"(bw));
      }
    }
  };
  for (uint32_t c = 0; c < T96_CHUNKS; c++) {
#pragma unroll
    for (uint32_t s = 0; s < 4u; s++) {
      half((c * 4u + s) * 1024u);
      half((c * 4u + s) * 1024u + 512u);
    }
    uint32_t x = 0u;
#pragma unroll
    for (uint32_t i = 0; i < 12u; i++)
#pragma unroll
      for (uint32_t j = 0; j < 8u; j++) x ^= (uint32_t)acc[i][j][0] ^ (uint32_t)acc[i][j][1];
    asm volatile("st.shared.u32 [%0], %1;" ::"r"(tr + c * 128u), "r"(x) : "memory");
  }
}

// The function the build replaces. It takes its operands' bases from the kernel's own first two parameters,
// which is where the hand loop reads them (the constant bank): keep Ap and Bp first in both kernels.
static __device__ __noinline__ void pearl_t96_loop(const int8_t *__restrict__ Ap, const int8_t *__restrict__ Bp,
                                                   uint32_t ao, uint32_t bo, uint32_t tr) {
  t96_loop_c(Ap, Bp, ao, bo, tr);
}
static __device__ __noinline__ void t96_ref_loop(const int8_t *__restrict__ Ap, const int8_t *__restrict__ Bp,
                                                 uint32_t ao, uint32_t bo, uint32_t tr) {
  t96_loop_c(Ap, Bp, ao, bo, tr);
}

// Tile vg of the salt: its row group and column group. Bands of `band` column groups: a band's B stays in the
// L2 while every row group's A streams past it, a band's columns first (the release's walk, PEARL_BD_WALK 1).
// The host's t96_coords (pearl_host_t96.inc) is the same arithmetic.
static __device__ __forceinline__ void t96_coords(uint32_t vg, uint32_t row_groups, uint32_t col_groups,
                                                  uint32_t band, uint32_t &rbg, uint32_t &cbg) {
  const uint32_t band_tiles = band * row_groups;
  const uint32_t bnd = vg / band_tiles, inb = vg % band_tiles;
  const uint32_t first = bnd * band;
  const uint32_t cols_in = col_groups - first < band ? col_groups - first : band;
  cbg = first + inb % cols_in;
  rbg = inb / cols_in;
}

// Persistent: one block an SM, block b starts on tile b and takes every later one from the counter ctr (the
// host zeroes it before the launch). Thread 0 asks for the tile after next at the start of each tile and
// publishes it at the tile's barrier, so the atomic's round trip hides under the tile.
// One warp of each scheduler (w & 3) hashes, its own 32 regions and then its partner's (w ^ 4), while the
// partner starts the next tile: the tensor unit of the scheduler is never idle for a hash.
// dump (the startup check): when not null, every region's transcript is written to dump[(256 v + thread) * 16]
// and nothing is hashed.
template <bool kHand>
static __device__ __forceinline__ void t96_fold(const int8_t *__restrict__ Ap, const int8_t *__restrict__ Bp,
                                                uint32_t *__restrict__ ctr, uint32_t *__restrict__ dump,
                                                uint32_t row_groups, uint32_t col_groups, uint32_t band,
                                                uint32_t tile0, uint32_t tiles, const PearlTranscriptTest &test,
                                                const PearlHitList &hits) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  extern __shared__ __align__(128) uint32_t t96_smem[];
  __shared__ uint32_t sNext[2];
  const uint32_t tid = threadIdx.x;
  if (blockDim.x != T96_THREADS || blockIdx.x >= tiles) return;
  if (tid == 0u) sNext[0] = gridDim.x + atomicAdd(ctr, 1u);
  __syncthreads();
  uint32_t it = 0u;
  for (uint32_t v = blockIdx.x; v < tiles; it++) {
    uint32_t vgrab = 0u;
    if (tid == 0u) vgrab = gridDim.x + atomicAdd(ctr, 1u);
    {
      const uint32_t warp = tid >> 5, lane = tid & 31u, wr = warp >> 2, ws = warp & 3u;
      uint32_t rbg, cbg;
      t96_coords(tile0 + v, row_groups, col_groups, band, rbg, cbg);
      const uint32_t ao = (rbg * 2u + wr) * (3u * T96_PLANE) + lane * 16u;
      const uint32_t bo = (cbg * 4u + ws) * (2u * T96_PLANE) + lane * 16u;
      const bool hasher = wr == (ws >> 1);
      const uint32_t buf = hasher ? ws * T96_TRB : 4u * T96_TRB + (2u * ws + (it & 1u)) * T96_TRB;
      const uint32_t tr = (uint32_t)__cvta_generic_to_shared(t96_smem) + buf + lane * 4u;
      // For the hand loop: the tile this block folds next, which thread 0 published at the barrier before, and
      // two flags. The loop's last k-step loads that tile's first operands, so the tile does not start by
      // waiting for them. They go in the thread's transcript rows 13 to 15, which are free until the loop
      // writes them (probes/turing/t96gen.py has the contract). The compiled loop does not read them.
      {
        const uint32_t vn = sNext[it & 1u];
        uint32_t ao2 = ao, bo2 = bo, flags = it == 0u ? 1u : 0u;   // 1: nothing was loaded for this tile
        if (vn < tiles) {
          uint32_t rbg2, cbg2;
          t96_coords(tile0 + vn, row_groups, col_groups, band, rbg2, cbg2);
          ao2 = (rbg2 * 2u + wr) * (3u * T96_PLANE) + lane * 16u;
          bo2 = (cbg2 * 4u + ws) * (2u * T96_PLANE) + lane * 16u;
          flags |= 2u;                                              // 2: there is a next tile
        }
        asm volatile("st.shared.u32 [%0], %1;" ::"r"(tr + 13u * 128u), "r"(ao2) : "memory");
        asm volatile("st.shared.u32 [%0], %1;" ::"r"(tr + 14u * 128u), "r"(bo2) : "memory");
        asm volatile("st.shared.u32 [%0], %1;" ::"r"(tr + 15u * 128u), "r"(flags) : "memory");
      }
      if constexpr (kHand)
        pearl_t96_loop(Ap, Bp, ao, bo, tr);
      else
        t96_ref_loop(Ap, Bp, ao, bo, tr);
    }
    const uint32_t vnext = sNext[it & 1u];
    if (tid == 0u) sNext[(it + 1u) & 1u] = vgrab;
    __syncthreads();
    {
      // Worked out again from an opaque copy of the thread index, so nothing of it lives through the loop.
      uint32_t t2 = threadIdx.x;
      asm volatile("" : "+r"(t2));
      const uint32_t warp = t2 >> 5, lane = t2 & 31u, wr = warp >> 2, ws = warp & 3u;
      if (wr == (ws >> 1)) {
        const uint32_t *const sT = t96_smem;
#pragma unroll 1
        for (uint32_t pass = 0; pass < T96_PASSES; pass++) {
          const uint32_t owner = pass ? t2 ^ 128u : t2;
          const uint32_t buf = pass ? 4u * T96_TRB + (2u * ws + (it & 1u)) * T96_TRB : ws * T96_TRB;
          uint32_t m[16];
#pragma unroll
          for (uint32_t c = 0; c < 16u; c++) m[c] = sT[(buf >> 2) + c * 32u + lane];
          const uint32_t index = v * T96_THREADS + owner;
          if (dump) {
#pragma unroll
            for (uint32_t c = 0; c < 16u; c++) dump[(size_t)index * 16u + c] = m[c];
            continue;
          }
          uint32_t key[8], h[8];
#pragma unroll
          for (uint32_t i = 0; i < 8u; i++) key[i] = test.key[i];
          t96_blake3_keyed64(key, m, h);
          // The hash, a little-endian number, at or under the bound (target_w: its words, most significant first).
          bool le = true;
          for (int i = 0; i < 8; i++) {
            const uint32_t hw = h[7 - i], b = test.target_w[i];
            if (hw != b) {
              le = hw < b;
              break;
            }
          }
          if (le) {
            const uint32_t n = atomicAdd(hits.count, 1u);
            if (n < PEARL_MAX_HITS) {
              hits.index[n] = index;
#pragma unroll
              for (uint32_t j = 0; j < 8u; j++) hits.hash[n * 8u + j] = h[j];
#pragma unroll
              for (uint32_t j = 0; j < 16u; j++) hits.transcript[n * 16u + j] = m[j];
            }
          }
        }
      }
    }
    v = vnext;
  }
#endif
}

extern "C" __global__ __launch_bounds__(T96_THREADS, 1) void pearl_t96_fold(
    const int8_t *__restrict__ Ap, const int8_t *__restrict__ Bp, uint32_t *__restrict__ ctr,
    uint32_t *__restrict__ dump, uint32_t row_groups, uint32_t col_groups, uint32_t band, uint32_t tile0,
    uint32_t tiles, const PearlTranscriptTest test, const PearlHitList hits) {
  t96_fold<true>(Ap, Bp, ctr, dump, row_groups, col_groups, band, tile0, tiles, test, hits);
}

extern "C" __global__ __launch_bounds__(T96_THREADS, 1) void t96_ref_fold(
    const int8_t *__restrict__ Ap, const int8_t *__restrict__ Bp, uint32_t *__restrict__ ctr,
    uint32_t *__restrict__ dump, uint32_t row_groups, uint32_t col_groups, uint32_t band, uint32_t tile0,
    uint32_t tiles, const PearlTranscriptTest test, const PearlHitList hits) {
  t96_fold<false>(Ap, Bp, ctr, dump, row_groups, col_groups, band, tile0, tiles, test, hits);
}

// A noised operand in the fold's fragment order (the head of this file). The values are pearl_materialize16's
// (pearl_kernel.cu, pearl_noise16): byte (r, kk) is base's plus dense[r][p0(kk)] - dense[r][p1(kk)], clamped to
// int8; only the first read_vecs sixteen-byte groups of base are read, the rest are fill_word (the constant
// fill). One thread a sixteen-byte group. Needs rows a multiple of 32 and k a power of two of at least 16.
extern "C" __global__ void t96_aux_mat(const int8_t *__restrict__ base, const int8_t *__restrict__ dense,
                                       const uint32_t *__restrict__ perm, int8_t *__restrict__ out, uint32_t rows,
                                       uint32_t k_log2, uint32_t rank, uint64_t read_vecs, uint32_t fill_word) {
  const uint64_t v = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= (((uint64_t)rows << k_log2) >> 4)) return;
  const uint64_t idx = v << 4;
  const uint32_t r = (uint32_t)(idx >> k_log2);
  const uint32_t kk = (uint32_t)(idx & ((1ull << k_log2) - 1u));
  const int8_t *__restrict__ row = dense + (size_t)r * rank;
  const int4 b4 = v < read_vecs ? reinterpret_cast<const int4 *>(base)[v]
                                : make_int4((int)fill_word, (int)fill_word, (int)fill_word, (int)fill_word);
  const uint32_t bw[4] = {(uint32_t)b4.x, (uint32_t)b4.y, (uint32_t)b4.z, (uint32_t)b4.w};
  const uint4 *__restrict__ pp = reinterpret_cast<const uint4 *>(perm + (size_t)kk * 2u);
  uint32_t *o = reinterpret_cast<uint32_t *>(out + (((uint64_t)(r >> 5)) << (k_log2 + 5u)) + (kk >> 4) * 512u
                                             + (r & 7u) * 64u)
                + ((r >> 3) & 3u);
#pragma unroll
  for (int w = 0; w < 4; w++) {
    const uint4 q0 = pp[w * 2];
    const uint4 q1 = pp[w * 2 + 1];
    const uint32_t p0[4] = {q0.x, q0.z, q1.x, q1.z};
    const uint32_t p1[4] = {q0.y, q0.w, q1.y, q1.w};
    uint32_t packed = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int32_t a = (int32_t)(int8_t)(bw[w] >> (j * 8));
      int32_t sv = a + (int32_t)row[p0[j]] - (int32_t)row[p1[j]];
      sv = sv < -128 ? -128 : (sv > 127 ? 127 : sv);
      packed |= ((uint32_t)sv & 0xffu) << (j * 8);
    }
    o[w * 4] = packed;  // lane 4 * (r & 7) + w, 16 bytes a lane
  }
}
