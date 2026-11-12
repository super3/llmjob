// The rotation fold's device code: a second sm_89 fold, OFF by default (PEARL_FOLD_ROT=1 turns it on, see
// pearl_host_rot.inc). It is NOT part of pearl_kernel.cu and nothing here is linked into the addon: this file is
// compiled alone into a cubin (nvcc -arch=sm_89 -cubin), probes/hand89/rfold.py then replaces the chunk loop's
// code with the loop written by hand, and the host loads the result at run time through the driver. The release
// kernels and their cubin are untouched by it.
//
// Three kernels:
//   pearl_rot_fold   the fold. Its chunk loop as compiled here is the gather fold (probes/hand89/gfold_h.cuh,
//                    every warp makes all its fragments): it is the room, the registers and the reference for
//                    the hand-written rotation loop, which reads offsets the compiled loop cannot use. The host
//                    loads only the patched file, pearl_rot_sm89.cubin; the base has another name.
//   rot_aux_tab      a dense noise factor (pearl_gen_dense's [rows][128] bytes) as the fold's group tables.
//   rot_aux_offs     the fold's offsets for one salt, from the two index lists and the order of k in each chunk.
// The two helpers have no "pearl" in their names on purpose: hand89/asm89.py patches the one code section that
// has.
//
// The fold's contract (the layouts are gfold_h.cuh's, the offsets gprobe.cu's `rot` form):
//   A tile is 192 rows by 256 columns, 256 threads: 2 row bands of 96 rows by 4 column slots of 64 columns.
//   Thread (warp w, lane l), g = l >> 2, q = l & 3, holds the sums of rows 96 (w >> 2) + g + 8 j (j 0 .. 11) and
//   columns 64 (w & 3) + 2 q + {0, 1} + 8 i (i 0 .. 7) of the tile: one region of the pattern rows {0, 8, .. 88}
//   by columns {0, 1, 8, 9, .. 56, 57}, 12 by 16. Tile v of a batch is row group rg0 + (tile0 + v) % row_groups
//   and column block (tile0 + v) / row_groups. A hit's index is 256 v + thread.
//   Shared memory: A's 12 group tables, B's 16, the transcripts, 256 spare bytes, the hand loop's stage.

#include <cuda_runtime.h>
#include <stdint.h>

#include "pearl_config.h"

#define ROT_MT 6u
#define ROT_NA (2u * ROT_MT)                        // A's groups a block
#define ROT_TABB (2048u * ROT_NA)                   // B's 16 groups
#define ROT_TR (ROT_TABB + 2048u * 16u)             // transcripts, [thread][16 words]
#define ROT_ST (ROT_TR + 256u * 64u)                // 256 spare bytes (the probe's stamps were here)
#define ROT_STG (ROT_ST + 256u)                     // the hand loop's stage of ready A fragments
#define ROT_STAGE 24576u
#define ROT_SMEM (ROT_STG + ROT_STAGE)
#define ROT_THREADS 256u
#define ROT_FILL 63
#define ROT_FILL4 ((uint32_t)ROT_FILL * 0x01010101u)
// The sm_120 loop's form (pearl_rot_c.inc's RC_FLAGS) and whether it is inlined. The shipped form, raced on an
// RTX 5090 (tuning/benchmark.md): 11 (B gathered a turn ahead, barriers after a chunk's first and last turn
// only), inlined: 401.9 and 401.4 TH/s by hashrate.js against 393.7, 391.8 and 391.3 for the release fold.
// The file a release ships is no longer this form: build_rot.py --arch sm_120 builds ROT_INLINE=0 ROT_C=16387
// (the C fold outlined, with room) and puts rfold.py's hand loop in that function (probes/hand89/rfold120.py).
#ifndef ROT_C
#define ROT_C 11
#endif
#ifndef ROT_INLINE
#define ROT_INLINE 1
#endif

static __device__ __forceinline__ void rot_g(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d, uint32_t mx,
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
static __device__ void rot_blake3_keyed64(const uint32_t key[8], const uint32_t block[16], uint32_t out[8]) {
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
    rot_g(s[0], s[4], s[8], s[12], m[0], m[1]);
    rot_g(s[1], s[5], s[9], s[13], m[2], m[3]);
    rot_g(s[2], s[6], s[10], s[14], m[4], m[5]);
    rot_g(s[3], s[7], s[11], s[15], m[6], m[7]);
    rot_g(s[0], s[5], s[10], s[15], m[8], m[9]);
    rot_g(s[1], s[6], s[11], s[12], m[10], m[11]);
    rot_g(s[2], s[7], s[8], s[13], m[12], m[13]);
    rot_g(s[3], s[4], s[9], s[14], m[14], m[15]);
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

// The tile's fold. It takes nothing from the kernel but these two values, which the hand-written loop reads
// from the kernel's own parameters (the third and the fourth: keep them there).
// On sm_120 the loop is the compiled C fold, nothing is put in its place, and ROT_INLINE=1 lets ptxas inline it:
// the offsets pointer then stays in the constant bank, which frees registers (the outlined loop spills one a chunk).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && ROT_INLINE
#define ROT_LOOP_ATTR __forceinline__
#else
#define ROT_LOOP_ATTR __noinline__
#endif
static __device__ ROT_LOOP_ATTR void pearl_rot_fold_loop(const uint4 *__restrict__ offs, uint32_t chunks) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  // RTX 50 (sm_120): the rotation fold itself, compiled from C (pearl_rot_c.inc, which the probe's GP_ROTC
  // builds too). The hand loop is written in sm_89's instructions; this cubin ships as ptxas makes it, with no
  // loop put in. ROT_C picks its form (pearl_rot_c.inc's RC_FLAGS).
#define RC_MT ROT_MT
#define RC_TR ROT_TR
#define RC_STG ROT_STG
#define RC_FILL4 ROT_FILL4
#define RC_FLAGS ROT_C
#include "pearl_rot_c.inc"
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr uint32_t MT = ROT_MT, NB = 8u;
  const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
  const uint32_t trL = ROT_TR + tid * 64u;
  auto ldmt = [](uint32_t r[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
  };
  auto gather = [&](uint32_t &f0, uint32_t &f1, uint32_t a) {
    uint32_t m[4];
    ldmt(m, a);
    const uint32_t x = m[0] + ROT_FILL4 - m[2], y = m[1] + ROT_FILL4 - m[3];
    f0 = __byte_perm(x, y, 0x6420u);
    f1 = __byte_perm(x, y, 0x7531u);
  };
  int32_t acc[MT][NB][4];
  uint32_t af[4], bf[NB][2];
#pragma unroll
  for (uint32_t mb = 0; mb < MT; mb++)
#pragma unroll
    for (uint32_t nb = 0; nb < NB; nb++)
#pragma unroll
      for (uint32_t i = 0; i < 4; i++) acc[mb][nb][i] = 0;
  const uint4 *po = offs + (size_t)warp * ((size_t)chunks * 128u + 32u) + lane;
  auto kstep = [&](const uint32_t zero) {
    const uint4 o = *po;
    po += 32;
#pragma unroll
    for (uint32_t P = 0; P < 4u; P++) {
      gather(bf[2u * P][0], bf[2u * P + 1u][0], o.z + 2048u * P);
      gather(bf[2u * P][1], bf[2u * P + 1u][1], o.w + 2048u * P);
    }
#pragma unroll
    for (uint32_t mb = 0; mb < MT; mb++) {
      gather(af[0], af[1], o.x + zero + 2048u * mb);
      gather(af[2], af[3], o.y + zero + 2048u * mb);
#pragma unroll
      for (uint32_t nb = 0; nb < NB; nb++)
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9}, {%0,%1,%2,%3};"
            : "+r"(acc[mb][nb][0]), "+r"(acc[mb][nb][1]), "+r"(acc[mb][nb][2]), "+r"(acc[mb][nb][3])
            : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bf[nb][0]), "r"(bf[nb][1]));
    }
  };
  auto chunk_loop = [&](const uint32_t zero) {
    for (uint32_t c = 0; c < chunks; c++) {
      kstep(zero);
      kstep(zero);
      kstep(zero);
      kstep(zero);
      // The chunk's end: the lane's XOR of its own sums into its own word of the transcript.
      uint32_t x = 0;
#pragma unroll
      for (uint32_t mb = 0; mb < MT; mb++)
#pragma unroll
        for (uint32_t nb = 0; nb < NB; nb++)
          x ^= (uint32_t)acc[mb][nb][0] ^ (uint32_t)acc[mb][nb][1] ^ (uint32_t)acc[mb][nb][2]
               ^ (uint32_t)acc[mb][nb][3];
      uint32_t w;
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(w) : "r"(trL + (c & 15u) * 4u) : "memory");
      w = __funnelshift_l(w, w, 13u) ^ x;
      asm volatile("st.shared.u32 [%0], %1;" ::"r"(trL + (c & 15u) * 4u), "r"(w) : "memory");
    }
  };
  // Two copies of the same loop, which is the room the hand-written loop is given (it is longer than one).
  // The second adds a value the compiler cannot know is zero, or it would make one of the two.
  if (warp < 4u)
    chunk_loop(0u);
  else
    chunk_loop(chunks >> 24);
#endif
}

extern "C" __global__ __launch_bounds__(ROT_THREADS, 1) void pearl_rot_fold(
    const uint4 *__restrict__ tabA, const uint4 *__restrict__ tabB, const uint4 *__restrict__ offs,
    uint32_t chunks, uint32_t row_groups, uint32_t rg0, uint32_t tile0, uint32_t tiles,
    const PearlTranscriptTest test, const PearlHitList hits) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  extern __shared__ __align__(128) uint32_t rot_smem[];
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(rot_smem);
  const uint32_t tid = threadIdx.x;
  if (blockDim.x != ROT_THREADS) return;
  for (uint32_t v = blockIdx.x; v < tiles; v += gridDim.x) {
    const uint32_t vg = tile0 + v;
    const uint32_t rbg = rg0 + vg % row_groups, cbg = vg / row_groups;
    // The tile's tables, from the host's: A's 128 ROT_NA entries, then B's 2048. And the thread's transcript.
    {
      const uint4 *const ta = tabA + (size_t)rbg * (128u * ROT_NA);
      const uint4 *const tb = tabB + (size_t)cbg * 2048u;
      for (uint32_t i = tid; i < 128u * ROT_NA + 2048u; i += ROT_THREADS) {
        const uint4 w = i < 128u * ROT_NA ? ta[i] : tb[i - 128u * ROT_NA];
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};"
                     ::"r"(sbase + i * 16u), "r"(w.x), "r"(w.y), "r"(w.z), "r"(w.w)
                     : "memory");
      }
#pragma unroll
      for (uint32_t q = 0; q < 4u; q++)
        asm volatile("st.shared.v4.u32 [%0], {%1,%1,%1,%1};" ::"r"(sbase + ROT_TR + tid * 64u + q * 16u), "r"(0u)
                     : "memory");
    }
    __syncthreads();
    pearl_rot_fold_loop(offs, chunks);
    // Every thread hashes its own transcript and reports itself when the hash, a little-endian number, is at
    // or under the bound (target_w: the bound's words, most significant first).
    {
      uint32_t m[16], key[8], h[8];
#pragma unroll
      for (uint32_t q = 0; q < 4u; q++)
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(m[4u * q]), "=r"(m[4u * q + 1u]), "=r"(m[4u * q + 2u]), "=r"(m[4u * q + 3u])
                     : "r"(sbase + ROT_TR + tid * 64u + q * 16u)
                     : "memory");
#pragma unroll
      for (uint32_t i = 0; i < 8u; i++) key[i] = test.key[i];
      rot_blake3_keyed64(key, m, h);
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
          hits.index[n] = v * ROT_THREADS + tid;
#pragma unroll
          for (uint32_t j = 0; j < 8u; j++) hits.hash[n * 8u + j] = h[j];
#pragma unroll
          for (uint32_t j = 0; j < 16u; j++) hits.transcript[n * 16u + j] = m[j];
        }
      }
    }
    __syncthreads();
  }
#endif
}

// A dense factor as group tables. dense is pearl_gen_dense's output, [rows][128] bytes in -32 .. 31. Group gi is
// rows 16 gi .. 16 gi + 15; its entry p is 16 bytes: byte 2 e the factor plus 32 of row 16 gi + e at rank index
// p, byte 2 e + 1 of row 16 gi + 8 + e. One thread an entry.
// ROT_TAB8 1 (the sm_120 hand loop's b8 key, build_rot.py): byte c is row 16 gi + c, in plain order. The loop
// then gathers with ldmatrix's byte transpose, which gives each lane rows g and g + 8 in registers of their own.
#ifndef ROT_TAB8
#define ROT_TAB8 0
#endif
extern "C" __global__ void rot_aux_tab(const int8_t *__restrict__ dense, uint4 *__restrict__ out,
                                       uint32_t groups) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= groups * 128u) return;
  const uint32_t gi = idx >> 7, p = idx & 127u;
  uint32_t w[4] = {0u, 0u, 0u, 0u};
#pragma unroll
  for (uint32_t e = 0; e < 8u; e++) {
    const uint32_t lo = (uint32_t)((int32_t)dense[(size_t)(16u * gi + e) * 128u + p] + 32) & 0xffu;
    const uint32_t hi = (uint32_t)((int32_t)dense[(size_t)(16u * gi + 8u + e) * 128u + p] + 32) & 0xffu;
#if ROT_TAB8
    w[e >> 2] |= lo << (8u * (e & 3u));
    w[2u + (e >> 2)] |= hi << (8u * (e & 3u));
#else
    w[e >> 1] |= (lo | (hi << 8)) << (16u * (e & 1u));
#endif
  }
  out[idx] = make_uint4(w[0], w[1], w[2], w[3]);
}

// The fold's offsets for one salt (gprobe.cu's `rot` 2 form). permA and permB are pearl_gen_perm's [k][2] (plus
// index, minus index); order[128 c + x] is which k of chunk c sits in slot x. offs is [warp][k-step][lane] of
// two words, one k-step more than the tile has at each warp's end. One thread a (k-step, lane).
//
// A record is 8 bytes, not the 16 of four plain offsets. The hand loop reads one record a warp a k-step from
// the L2, and that trip is a large part of what holds the card's clock down (probes/README.md, "Where the
// rotation fold's time goes"). Every offset is a shared-memory address under 65536, so two fit a word: a
// first turn's record is {A's two, B's two}, a later turn's {the stage address of the k-step's A fragments,
// B's two}. The loop parts them with two ALU steps.
extern "C" __global__ void rot_aux_offs(const uint32_t *__restrict__ permA, const uint32_t *__restrict__ permB,
                                        const uint8_t *__restrict__ order, uint32_t *__restrict__ offs,
                                        uint32_t ksteps) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (ksteps + 1u) * 32u) return;
  const uint32_t ks = idx >> 5, l = idx & 31u;
  const uint32_t ch = ks >> 2, s = ks & 3u;
  const uint32_t sign = l >> 4, slot = 4u * ((l & 7u) >> 1) + 2u * ((l >> 3) & 1u) + (l & 1u);
  const size_t warpWords = (size_t)(ksteps + 1u) * 32u * 2u;
  uint32_t pa[2] = {0u, 0u}, pb[2] = {0u, 0u};
  if (ks < ksteps)
    for (uint32_t h = 0; h < 2u; h++) {
      const uint32_t kk = ch * 128u + order[(size_t)ch * 128u + 32u * s + 16u * h + slot];
      pa[h] = permA[2u * kk + sign];
      pb[h] = permB[2u * kk + sign];
    }
  for (uint32_t w = 0; w < 8u; w++) {
    const uint32_t j = w & 3u, band = w >> 2;
    const uint32_t turn = ks == ksteps ? s : band ? (((s + 4u - j) & 3u) ^ 1u) : ((s + 4u - j) & 3u);
    uint32_t *o = offs + w * warpWords + ((size_t)(ch * 4u + turn) * 32u + l) * 2u;
    const uint32_t a0 = 16u * pa[0] + 2048u * ROT_MT * band, a1 = 16u * pa[1] + 2048u * ROT_MT * band;
    const uint32_t b0 = 16u * pb[0] + ROT_TABB + 8192u * j, b1 = 16u * pb[1] + ROT_TABB + 8192u * j;
    o[0] = turn && ks < ksteps ? ROT_STG + band * (2048u * ROT_MT) + s * (512u * ROT_MT) + 16u * l
                               : a0 | (a1 << 16);
    o[1] = b0 | (b1 << 16);
  }
}
