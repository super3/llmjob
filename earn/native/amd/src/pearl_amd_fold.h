// The AMD folds: C = A' x B'^T in int32 over k, read out per region at every chunk end,
// and the transcript hash that follows them.
//
// What a fold computes is fixed by the protocol and described in pearl_kernel.cu: for each
// 16x16 region (rows {0-3, 8-11, 16-19, 24-27} + row offset, columns {0, 1, 8, 9, ..., 56,
// 57} + column offset), transcript word c is the XOR of the region's 256 running sums after
// chunk c. Here every fold writes all 16 words of every region of the batch to `tr`, and
// pearl_amd_hash hashes them in a kernel of its own. The CUDA folds hash inside the fold on
// most cards; storing first is what GA100 and Hopper do, and it is the simplest thing that
// is correct on every AMD target.
//
// Five folds:
//   pearl_amd_fold_ref     plain dot products from LDS. Any AMD GPU (sdot4/sudot4 where
//                          the card has them), and the fold for RDNA2 (gfx103x).
//   pearl_amd_fold_mfma16  CDNA3/CDNA4 (gfx942, gfx950): v_mfma_i32_32x32x16_i8, wave64.
//   pearl_amd_fold_mfma8   CDNA1/CDNA2 (gfx908, gfx90a): v_mfma_i32_32x32x8_i8, wave64.
//   pearl_amd_fold_wmma11  RDNA3 (gfx11): v_wmma_i32_16x16x16_iu8, wave32.
//   pearl_amd_fold_wmma12  RDNA4 (gfx12): v_wmma_i32_16x16x16_iu8, wave32, RDNA4's layout.
//
// Each matrix fold is the same code (pearl_amd_fold_body) over a policy that says how the
// instruction lays out its operands and results across lanes. The layouts are AMD's, from
// the Matrix Instruction Calculator (github.com/ROCm/amd_matrix_instruction_calculator).
//
// No GPU was available to write this. So every policy also has an emulation of its
// instruction, written from the same documented layout, and the HIP-CPU build
// (__HIP_CPU_RT__) runs the folds with it: each lane puts its fragments in shared memory,
// the block synchronises, and each lane computes the outputs the layout says it holds.
// Everything else in the fold -- staging, fragment loads, the readout, the transcript
// stores -- is the same source on the CPU and on the GPU. What the CPU run cannot show is
// that the hardware matches the documented layout; test/pearl_amd_foldtest.cpp checks that
// on a real card.
#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "pearl_config.h"

#if defined(__HIP_CPU_RT__)
#define PEARL_AMD_EMU 1
#else
#define PEARL_AMD_EMU 0
#endif

// Which instructions this compile pass has. Only the device pass of a real target sets
// any of these; a builtin compiled for a target without it stops the compiler, so every
// one is behind its own gate.
#if !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__)
#if defined(__gfx942__) || defined(__gfx950__)
#define PEARL_AMD_MFMA16 1
#endif
#if defined(__gfx908__) || defined(__gfx90a__)
#define PEARL_AMD_MFMA8 1
#endif
#if defined(__GFX11__)
#define PEARL_AMD_WMMA11 1
#endif
#if defined(__GFX12__)
#define PEARL_AMD_WMMA12 1
#endif
// v_dot4_i32_i8 (or v_dot4c on RDNA2): Vega 20, CDNA, RDNA1's gfx1011/1012 and RDNA2.
#if defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__) || \
    defined(__gfx950__) || defined(__gfx1011__) || defined(__gfx1012__) || defined(__gfx1030__) || \
    defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || defined(__gfx1034__) || \
    defined(__gfx1035__) || defined(__gfx1036__)
#define PEARL_AMD_SDOT4 1
#endif
// v_dot4_i32_iu8 on RDNA3 and RDNA4.
#if defined(__GFX11__) || defined(__GFX12__)
#define PEARL_AMD_SUDOT4 1
#endif
#endif

#ifndef PEARL_AMD_MFMA16
#define PEARL_AMD_MFMA16 0
#endif
#ifndef PEARL_AMD_MFMA8
#define PEARL_AMD_MFMA8 0
#endif
#ifndef PEARL_AMD_WMMA11
#define PEARL_AMD_WMMA11 0
#endif
#ifndef PEARL_AMD_WMMA12
#define PEARL_AMD_WMMA12 0
#endif

// Which folds have a body in the code object that runs (pearl_amd_caps). The host asks
// once a context: a fold without a body for the card returns at once, finds nothing and
// still reports hashrate, so the host must never launch one.
#define PEARL_AMD_CAP_REF 1u
#define PEARL_AMD_CAP_MFMA16 2u
#define PEARL_AMD_CAP_MFMA8 4u
#define PEARL_AMD_CAP_WMMA11 8u
#define PEARL_AMD_CAP_WMMA12 16u
#define PEARL_AMD_CAP_EMULATED 0x100u  // the HIP-CPU build: every fold, emulated

// Every fold is 256 threads a block.
#define PEARL_AMD_THREADS 256u
// The matrix folds' tile: 128 rows of A' by 256 columns (rows of B'), staged through LDS
// PEARL_AMD_SLAB bytes of k at a time, double-buffered. A row in LDS is the slab plus 16
// bytes, which keeps 16-byte stores aligned and spreads rows over the banks.
#define PEARL_AMD_BM 128u
#define PEARL_AMD_BN 256u
#define PEARL_AMD_SLAB 32u
// The reference fold's tile: one 32x64 block of C, which holds eight whole regions.
#define PEARL_AMD_REF_BM 32u
#define PEARL_AMD_REF_BN 64u

typedef int32_t pearl_amd_v16i __attribute__((ext_vector_type(16)));
typedef int32_t pearl_amd_v8i __attribute__((ext_vector_type(8)));
typedef int32_t pearl_amd_v4i __attribute__((ext_vector_type(4)));
typedef int32_t pearl_amd_v2i __attribute__((ext_vector_type(2)));

namespace {

__device__ __forceinline__ int32_t pearl_amd_sbyte(uint64_t v, int i) {
  return (int32_t)(int8_t)(uint8_t)(v >> (8 * i));
}

// ---------------------------------------------------------------------------------------
// The policies. Each gives the instruction's shape (kTM x kTN x kTK), its wave size, how a
// wave's tiles and a block's waves are arranged, and three things about its layout:
//   load_a/load_b  the fragment one lane holds, read from an LDS tile whose rows are A' rows
//                  (or B' rows, which are columns of C), `stride` bytes apart, with the k
//                  offset already applied;
//   a_elem/b_elem  element A[i][kk] (B[kk][j]) of the tile, read out of the wave's fragments
//                  the way the hardware reads it -- the emulation's only view of them;
//   c_row/c_col    which element of C accumulator register g of lane `lane` is.
// And for the readout, which region of a 32x64 block a register's row falls in (ro_of).
// ---------------------------------------------------------------------------------------

// CDNA3 and CDNA4: v_mfma_i32_32x32x16_i8. Lane l holds A row l % 32 at k 8*(l/32) .. +7 and
// B column l % 32 at the same k; accumulator g of lane l is C[8*(g/4) + 4*(l/32) + g%4][l%32].
struct PearlAmdMfma16 {
  static constexpr int kWave = 64, kTM = 32, kTN = 32, kTK = 16;
  static constexpr int kWR = 2, kWC = 2, kMT = 2, kNT = 4;
  static constexpr int kRegs = 16;
  typedef int64_t AFrag;
  typedef int64_t BFrag;
  typedef pearl_amd_v16i Acc;
  static __device__ __forceinline__ AFrag load_a(const int8_t *s, int stride, int lane) {
    return *reinterpret_cast<const int64_t *>(s + (lane & 31) * stride + 8 * (lane >> 5));
  }
  static __device__ __forceinline__ BFrag load_b(const int8_t *s, int stride, int lane) {
    return load_a(s, stride, lane);
  }
  static __device__ __forceinline__ int32_t a_elem(const AFrag *f, int i, int kk, int) {
    return pearl_amd_sbyte((uint64_t)f[32 * (kk / 8) + i], kk % 8);
  }
  static __device__ __forceinline__ int32_t b_elem(const BFrag *f, int kk, int j, int) {
    return pearl_amd_sbyte((uint64_t)f[32 * (kk / 8) + j], kk % 8);
  }
  static __device__ __forceinline__ int c_row(int lane, int g) {
    return 8 * (g / 4) + 4 * (lane / 32) + g % 4;
  }
  static __device__ __forceinline__ int c_col(int lane, int) { return lane % 32; }
  static __device__ __forceinline__ void mma(Acc &c, AFrag a, BFrag b) {
#if PEARL_AMD_MFMA16
    c = __builtin_amdgcn_mfma_i32_32x32x16_i8(a, b, c, 0, 0, 0);
#else
    (void)c; (void)a; (void)b;
#endif
  }
  // A lane's rows are one region's: which one is lane bit 5, not the register.
  static constexpr bool kRoByLane = true;
  static __device__ __forceinline__ int ro_of(int) { return 0; }
};

// CDNA1 and CDNA2: v_mfma_i32_32x32x8_i8. The same as above with 4 bytes of k a lane:
// k 4*(l/32) .. +3.
struct PearlAmdMfma8 {
  static constexpr int kWave = 64, kTM = 32, kTN = 32, kTK = 8;
  static constexpr int kWR = 2, kWC = 2, kMT = 2, kNT = 4;
  static constexpr int kRegs = 16;
  typedef int32_t AFrag;
  typedef int32_t BFrag;
  typedef pearl_amd_v16i Acc;
  static __device__ __forceinline__ AFrag load_a(const int8_t *s, int stride, int lane) {
    return *reinterpret_cast<const int32_t *>(s + (lane & 31) * stride + 4 * (lane >> 5));
  }
  static __device__ __forceinline__ BFrag load_b(const int8_t *s, int stride, int lane) {
    return load_a(s, stride, lane);
  }
  static __device__ __forceinline__ int32_t a_elem(const AFrag *f, int i, int kk, int) {
    return pearl_amd_sbyte((uint32_t)f[32 * (kk / 4) + i], kk % 4);
  }
  static __device__ __forceinline__ int32_t b_elem(const BFrag *f, int kk, int j, int) {
    return pearl_amd_sbyte((uint32_t)f[32 * (kk / 4) + j], kk % 4);
  }
  static __device__ __forceinline__ int c_row(int lane, int g) {
    return 8 * (g / 4) + 4 * (lane / 32) + g % 4;
  }
  static __device__ __forceinline__ int c_col(int lane, int) { return lane % 32; }
  static __device__ __forceinline__ void mma(Acc &c, AFrag a, BFrag b) {
#if PEARL_AMD_MFMA8
    c = __builtin_amdgcn_mfma_i32_32x32x8i8(a, b, c, 0, 0, 0);
#else
    (void)c; (void)a; (void)b;
#endif
  }
  static constexpr bool kRoByLane = true;
  static __device__ __forceinline__ int ro_of(int) { return 0; }
};

// RDNA3: v_wmma_i32_16x16x16_iu8 in wave32. Lanes l and l + 16 both hold A row l % 16, all
// 16 bytes of k (and B column l % 16 the same way); accumulator g of lane l is
// C[2g + l/16][l % 16]. The emulation reads a lane's A and B from its own half of the wave,
// so a fold that filled only one half gives wrong sums in the other.
struct PearlAmdWmma11 {
  static constexpr int kWave = 32, kTM = 16, kTN = 16, kTK = 16;
  static constexpr int kWR = 2, kWC = 4, kMT = 4, kNT = 4;
  static constexpr int kRegs = 8;
  typedef pearl_amd_v4i AFrag;
  typedef pearl_amd_v4i BFrag;
  typedef pearl_amd_v8i Acc;
  static __device__ __forceinline__ AFrag load_a(const int8_t *s, int stride, int lane) {
    return *reinterpret_cast<const pearl_amd_v4i *>(s + (lane & 15) * stride);
  }
  static __device__ __forceinline__ BFrag load_b(const int8_t *s, int stride, int lane) {
    return load_a(s, stride, lane);
  }
  static __device__ __forceinline__ int32_t a_elem(const AFrag *f, int i, int kk, int lane) {
    return pearl_amd_sbyte((uint32_t)f[i + 16 * (lane / 16)][kk / 4], kk % 4);
  }
  static __device__ __forceinline__ int32_t b_elem(const BFrag *f, int kk, int j, int lane) {
    return pearl_amd_sbyte((uint32_t)f[j + 16 * (lane / 16)][kk / 4], kk % 4);
  }
  static __device__ __forceinline__ int c_row(int lane, int g) { return 2 * g + lane / 16; }
  static __device__ __forceinline__ int c_col(int lane, int) { return lane % 16; }
  static __device__ __forceinline__ void mma(Acc &c, AFrag a, BFrag b) {
#if PEARL_AMD_WMMA11
    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
#else
    (void)c; (void)a; (void)b;
#endif
  }
  // Registers {0, 1, 4, 5} hold rows 0-3 and 8-11 of a 16-row tile (region row offset 0),
  // {2, 3, 6, 7} rows 4-7 and 12-15 (offset 4).
  static constexpr bool kRoByLane = false;
  static __device__ __forceinline__ int ro_of(int g) { return (g >> 1) & 1; }
};

// RDNA4: v_wmma_i32_16x16x16_iu8 in wave32, gfx12's layout. Lane l holds A row l % 16 at k
// 8*(l/16) .. +7 (and B column l % 16 the same way); accumulator g of lane l is
// C[8*(l/16) + g][l % 16].
struct PearlAmdWmma12 {
  static constexpr int kWave = 32, kTM = 16, kTN = 16, kTK = 16;
  static constexpr int kWR = 2, kWC = 4, kMT = 4, kNT = 4;
  static constexpr int kRegs = 8;
  typedef pearl_amd_v2i AFrag;
  typedef pearl_amd_v2i BFrag;
  typedef pearl_amd_v8i Acc;
  static __device__ __forceinline__ AFrag load_a(const int8_t *s, int stride, int lane) {
    return *reinterpret_cast<const pearl_amd_v2i *>(s + (lane & 15) * stride + 8 * (lane >> 4));
  }
  static __device__ __forceinline__ BFrag load_b(const int8_t *s, int stride, int lane) {
    return load_a(s, stride, lane);
  }
  static __device__ __forceinline__ int32_t a_elem(const AFrag *f, int i, int kk, int) {
    return pearl_amd_sbyte((uint32_t)f[16 * (kk / 8) + i][(kk / 4) % 2], kk % 4);
  }
  static __device__ __forceinline__ int32_t b_elem(const BFrag *f, int kk, int j, int) {
    return pearl_amd_sbyte((uint32_t)f[16 * (kk / 8) + j][(kk / 4) % 2], kk % 4);
  }
  static __device__ __forceinline__ int c_row(int lane, int g) { return 8 * (lane / 16) + g; }
  static __device__ __forceinline__ int c_col(int lane, int) { return lane % 16; }
  static __device__ __forceinline__ void mma(Acc &c, AFrag a, BFrag b) {
#if PEARL_AMD_WMMA12
    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
#else
    (void)c; (void)a; (void)b;
#endif
  }
  // Registers 0-3 hold rows 0-3 or 8-11 of a 16-row tile (row offset 0), 4-7 the others.
  static constexpr bool kRoByLane = false;
  static __device__ __forceinline__ int ro_of(int g) { return g >> 2; }
};

// One k-step of a wave: every tile of the wave's kMT x kNT, from the wave's A and B
// fragments. On the GPU, one instruction a tile. In the HIP-CPU build, the emulation. Every
// thread of the block calls it at the same point, which the emulation's barriers need.
template <class P>
__device__ __forceinline__ void pearl_amd_mma_step(typename P::Acc (&acc)[P::kMT][P::kNT],
                                                   const typename P::AFrag (&a)[P::kMT],
                                                   const typename P::BFrag (&b)[P::kNT]) {
#if PEARL_AMD_EMU
  constexpr int W = P::kWave, WAVES = P::kWR * P::kWC;
  __shared__ typename P::AFrag ea[WAVES][P::kMT][W];
  __shared__ typename P::BFrag eb[WAVES][P::kNT][W];
  const int wave = (int)threadIdx.x / W, lane = (int)threadIdx.x % W;
  for (int mt = 0; mt < P::kMT; mt++) ea[wave][mt][lane] = a[mt];
  for (int nt = 0; nt < P::kNT; nt++) eb[wave][nt][lane] = b[nt];
  __syncthreads();
  for (int mt = 0; mt < P::kMT; mt++)
    for (int nt = 0; nt < P::kNT; nt++)
      for (int g = 0; g < P::kRegs; g++) {
        const int i = P::c_row(lane, g), j = P::c_col(lane, g);
        int32_t s = 0;
        for (int kk = 0; kk < P::kTK; kk++)
          s += P::a_elem(ea[wave][mt], i, kk, lane) * P::b_elem(eb[wave][nt], kk, j, lane);
        acc[mt][nt][g] += s;
      }
  __syncthreads();
#else
#pragma unroll
  for (int mt = 0; mt < P::kMT; mt++)
#pragma unroll
    for (int nt = 0; nt < P::kNT; nt++) P::mma(acc[mt][nt], a[mt], b[nt]);
#endif
}

// XOR across the eight lanes that share a region: lane bits 0, 3 and 4 in every layout here.
// (Bits 1 and 2 pick the region's column offset; on CDNA bit 5 picks its row offset.)
__device__ __forceinline__ uint32_t pearl_amd_group_xor(uint32_t x) {
  x ^= (uint32_t)__shfl_xor((int)x, 1);
  x ^= (uint32_t)__shfl_xor((int)x, 8);
  x ^= (uint32_t)__shfl_xor((int)x, 16);
  return x;
}

// The matrix fold over one 128x256 tile of C: rows blockIdx.y * 128 of A', and B' rows
// col0 + blockIdx.x * 256 (the batch's columns start at col0). Every region of the tile gets
// its 16 transcript words in `tr`, at its batch-local number: column offset index within
// the batch times rows_valid, plus row offset index.
//
// A wave holds kMT x kNT tiles: 64 rows by 128 columns on CDNA, 64 by 64 on RDNA. That is
// whole 32x64 blocks, and each block holds eight whole regions. At every chunk end a lane
// XORs the accumulators it holds of each region in the block (32 values), the eight lanes
// that share the region combine theirs, and lane r of those eight keeps transcript words r
// and r + 8. So all 16 words of a region end up in registers, two a lane, with no shared
// memory and no rotate (16 chunks: each word is written once, onto zero).
template <class P>
__device__ __forceinline__ void pearl_amd_fold_body(const int8_t *__restrict__ Ap,
                                                    const int8_t *__restrict__ Bp, uint32_t k,
                                                    uint32_t rank, uint32_t col0,
                                                    uint32_t rows_valid,
                                                    uint32_t *__restrict__ tr) {
  constexpr int W = P::kWave, T = W * P::kWR * P::kWC;
  constexpr int WM = P::kMT * P::kTM, WN = P::kNT * P::kTN;
  constexpr int BM = P::kWR * WM, BN = P::kWC * WN;
  constexpr int KS = (int)PEARL_AMD_SLAB, S = KS + 16;
  constexpr int QR = KS / 16;                          // 16-byte units in a slab row
  constexpr int UA = BM * QR, UB = BN * QR, U = (UA + UB) / T;
  constexpr int RB = WM / 32, CB = WN / 64;            // 32x64 blocks a wave
  constexpr int NRO = P::kRoByLane ? 1 : 2;            // regions a lane holds in a block
  static_assert(T == (int)PEARL_AMD_THREADS && BM == (int)PEARL_AMD_BM && BN == (int)PEARL_AMD_BN,
                "the host launches every matrix fold as 256 threads over 128x256 tiles");
  static_assert((UA + UB) % T == 0 && UA % T == 0, "the staging must split evenly");
  static_assert(WM % 32 == 0 && WN % 64 == 0, "a wave must hold whole 32x64 blocks");
  static_assert(KS % P::kTK == 0, "a slab must hold whole k-steps");

  __shared__ __attribute__((aligned(16))) int8_t sA[2][BM * S];
  __shared__ __attribute__((aligned(16))) int8_t sB[2][BN * S];

  const int t = (int)threadIdx.x, wave = t / W, lane = t % W;
  const int wr = wave / P::kWC, wc = wave % P::kWC;
  const uint32_t tileRow = blockIdx.y * (uint32_t)BM;   // first A' row of the tile
  const uint32_t tileCol = blockIdx.x * (uint32_t)BN;   // first column, within the batch
  const int8_t *gA = Ap + (size_t)tileRow * k;
  const int8_t *gB = Bp + (size_t)(col0 + tileCol) * k;

  // The next slab, in registers while the current one is used. A vector type, not HIP's
  // int4: int4 is a struct, which the compiler copied through scratch memory.
  pearl_amd_v4i pre[U];
#define PEARL_AMD_LOAD(slab)                                                                 \
  _Pragma("unroll") for (int u = 0; u < U; u++) {                                            \
    const int idx = t + u * T;                                                               \
    const int8_t *src = idx < UA ? gA + (size_t)(idx / QR) * k                               \
                                 : gB + (size_t)((idx - UA) / QR) * k;                       \
    pre[u] = *reinterpret_cast<const pearl_amd_v4i *>(src + (size_t)(slab) * KS + 16 * (idx % QR)); \
  }
#define PEARL_AMD_STORE(buf)                                                                 \
  _Pragma("unroll") for (int u = 0; u < U; u++) {                                            \
    const int idx = t + u * T;                                                               \
    int8_t *dst = idx < UA ? sA[buf] + (idx / QR) * S : sB[buf] + ((idx - UA) / QR) * S;     \
    *reinterpret_cast<pearl_amd_v4i *>(dst + 16 * (idx % QR)) = pre[u];                      \
  }

  typename P::Acc acc[P::kMT][P::kNT];
#pragma unroll
  for (int mt = 0; mt < P::kMT; mt++)
#pragma unroll
    for (int nt = 0; nt < P::kNT; nt++)
#pragma unroll
      for (int g = 0; g < P::kRegs; g++) acc[mt][nt][g] = 0;
  // Transcript words r and r + 8 of each region this lane shares.
  uint32_t keepLo[RB][CB][NRO], keepHi[RB][CB][NRO];
#pragma unroll
  for (int rb = 0; rb < RB; rb++)
#pragma unroll
    for (int cb = 0; cb < CB; cb++)
#pragma unroll
      for (int q = 0; q < NRO; q++) keepLo[rb][cb][q] = keepHi[rb][cb][q] = 0u;
  const uint32_t r = (uint32_t)((lane & 1) | ((lane >> 2) & 2) | ((lane >> 2) & 4));

  const uint32_t slabs = k / (uint32_t)KS, perChunk = rank / (uint32_t)KS;
  PEARL_AMD_LOAD(0)
  PEARL_AMD_STORE(0)
  __syncthreads();
  for (uint32_t s = 0; s < slabs; s++) {
    const int buf = (int)(s & 1u);
    if (s + 1 < slabs) { PEARL_AMD_LOAD(s + 1) }
    const int8_t *aBase = sA[buf] + wr * WM * S;
    const int8_t *bBase = sB[buf] + wc * WN * S;
#pragma unroll
    for (int kk = 0; kk < KS / P::kTK; kk++) {
      typename P::AFrag a[P::kMT];
      typename P::BFrag b[P::kNT];
#pragma unroll
      for (int mt = 0; mt < P::kMT; mt++)
        a[mt] = P::load_a(aBase + mt * P::kTM * S + kk * P::kTK, S, lane);
#pragma unroll
      for (int nt = 0; nt < P::kNT; nt++)
        b[nt] = P::load_b(bBase + nt * P::kTN * S + kk * P::kTK, S, lane);
      pearl_amd_mma_step<P>(acc, a, b);
    }
    if ((s + 1) % perChunk == 0) {
      // The end of chunk c: its transcript word, for every region the wave holds.
      const uint32_t c = (s + 1) / perChunk - 1;
      const bool mine = (c & 7u) == r, hi = c >= 8u;
#pragma unroll
      for (int rb = 0; rb < RB; rb++)
#pragma unroll
        for (int cb = 0; cb < CB; cb++) {
          uint32_t x[NRO];
#pragma unroll
          for (int q = 0; q < NRO; q++) x[q] = 0u;
          // The block's tiles: kTM = 32 is one tile down, 16 is two; kTN = 32 is two tiles
          // across, 16 is four.
#pragma unroll
          for (int mt = rb * (32 / P::kTM); mt < (rb + 1) * (32 / P::kTM); mt++)
#pragma unroll
            for (int nt = cb * (64 / P::kTN); nt < (cb + 1) * (64 / P::kTN); nt++)
#pragma unroll
              for (int g = 0; g < P::kRegs; g++) x[P::ro_of(g)] ^= (uint32_t)acc[mt][nt][g];
#pragma unroll
          for (int q = 0; q < NRO; q++) {
            const uint32_t v = pearl_amd_group_xor(x[q]);
            keepLo[rb][cb][q] = (mine && !hi) ? v : keepLo[rb][cb][q];
            keepHi[rb][cb][q] = (mine && hi) ? v : keepHi[rb][cb][q];
          }
        }
    }
    if (s + 1 < slabs) { PEARL_AMD_STORE(buf ^ 1) }
    __syncthreads();
  }

  // Each lane writes words r and r + 8 of its regions.
  const uint32_t coIdx = (uint32_t)((lane >> 1) & 3);
#pragma unroll
  for (int rb = 0; rb < RB; rb++)
#pragma unroll
    for (int cb = 0; cb < CB; cb++)
#pragma unroll
      for (int q = 0; q < NRO; q++) {
        const uint32_t roIdx = P::kRoByLane ? (uint32_t)(lane >> 5) : (uint32_t)q;
        const uint32_t rowIdx = 2u * ((tileRow + (uint32_t)(wr * WM + rb * 32)) / 32u) + roIdx;
        const uint32_t colIdx = 4u * ((tileCol + (uint32_t)(wc * WN + cb * 64)) / 64u) + coIdx;
        uint32_t *w = tr + ((size_t)colIdx * rows_valid + rowIdx) * PEARL_JACKPOT_BUCKETS;
        w[r] = keepLo[rb][cb][q];
        w[r + 8] = keepHi[rb][cb][q];
      }
#undef PEARL_AMD_LOAD
#undef PEARL_AMD_STORE
}

__device__ __forceinline__ int32_t pearl_amd_dot4(int32_t a, int32_t b, int32_t c) {
#if defined(PEARL_AMD_SDOT4)
  return __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(PEARL_AMD_SUDOT4)
  return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
  return c + pearl_amd_sbyte((uint32_t)a, 0) * pearl_amd_sbyte((uint32_t)b, 0)
         + pearl_amd_sbyte((uint32_t)a, 1) * pearl_amd_sbyte((uint32_t)b, 1)
         + pearl_amd_sbyte((uint32_t)a, 2) * pearl_amd_sbyte((uint32_t)b, 2)
         + pearl_amd_sbyte((uint32_t)a, 3) * pearl_amd_sbyte((uint32_t)b, 3);
#endif
}

}  // namespace

#define PEARL_AMD_FOLD_PARAMS                                                              \
  const int8_t *__restrict__ Ap, const int8_t *__restrict__ Bp, uint32_t k, uint32_t rank, \
      uint32_t col0, uint32_t rows_valid, uint32_t *__restrict__ tr

extern "C" __global__ __launch_bounds__(PEARL_AMD_THREADS) void pearl_amd_fold_mfma16(
    PEARL_AMD_FOLD_PARAMS) {
#if PEARL_AMD_EMU || PEARL_AMD_MFMA16
  pearl_amd_fold_body<PearlAmdMfma16>(Ap, Bp, k, rank, col0, rows_valid, tr);
#else
  (void)Ap; (void)Bp; (void)k; (void)rank; (void)col0; (void)rows_valid; (void)tr;
#endif
}

extern "C" __global__ __launch_bounds__(PEARL_AMD_THREADS) void pearl_amd_fold_mfma8(
    PEARL_AMD_FOLD_PARAMS) {
#if PEARL_AMD_EMU || PEARL_AMD_MFMA8
  pearl_amd_fold_body<PearlAmdMfma8>(Ap, Bp, k, rank, col0, rows_valid, tr);
#else
  (void)Ap; (void)Bp; (void)k; (void)rank; (void)col0; (void)rows_valid; (void)tr;
#endif
}

extern "C" __global__ __launch_bounds__(PEARL_AMD_THREADS) void pearl_amd_fold_wmma11(
    PEARL_AMD_FOLD_PARAMS) {
#if PEARL_AMD_EMU || PEARL_AMD_WMMA11
  pearl_amd_fold_body<PearlAmdWmma11>(Ap, Bp, k, rank, col0, rows_valid, tr);
#else
  (void)Ap; (void)Bp; (void)k; (void)rank; (void)col0; (void)rows_valid; (void)tr;
#endif
}

extern "C" __global__ __launch_bounds__(PEARL_AMD_THREADS) void pearl_amd_fold_wmma12(
    PEARL_AMD_FOLD_PARAMS) {
#if PEARL_AMD_EMU || PEARL_AMD_WMMA12
  pearl_amd_fold_body<PearlAmdWmma12>(Ap, Bp, k, rank, col0, rows_valid, tr);
#else
  (void)Ap; (void)Bp; (void)k; (void)rank; (void)col0; (void)rows_valid; (void)tr;
#endif
}

// The reference fold: one 32x64 block of C, eight regions, a block. Thread t works on region
// t / 32 of the block and holds 8 of its 256 sums: one row, eight columns. Each chunk's
// rows of A' and B' go through LDS; the sums are dot products of 4 bytes at a time. Slow
// next to the matrix folds, but it needs nothing from the card, so it is the baseline to
// compare against on hardware (PEARL_AMD_FOLD=ref) and the fold RDNA2 runs.
extern "C" __global__ __launch_bounds__(PEARL_AMD_THREADS) void pearl_amd_fold_ref(
    PEARL_AMD_FOLD_PARAMS) {
  constexpr uint32_t T = PEARL_AMD_THREADS, RB = PEARL_AMD_REF_BM, CB = PEARL_AMD_REF_BN;
  constexpr uint32_t KMAX = 128u, S = KMAX + 16u;
  __shared__ __attribute__((aligned(16))) int8_t sA[RB * S];
  __shared__ __attribute__((aligned(16))) int8_t sB[CB * S];
  const uint32_t t = threadIdx.x, q = t / 32u, p = t % 32u;
  // Region q of the block: row offset 4 * (q / 4), column offset 2 * (q % 4). Its 256 sums
  // in pattern order: thread p holds pattern row p / 2 and pattern columns 8 * (p % 2) + u.
  const uint32_t ri = p >> 1;
  const uint32_t row = 4u * (q >> 2) + ((ri & 3u) | ((ri >> 2) << 3));
  uint32_t cols[8];
#pragma unroll
  for (uint32_t u = 0; u < 8u; u++) {
    const uint32_t ci = 8u * (p & 1u) + u;
    cols[u] = 2u * (q & 3u) + ((ci & 1u) | ((ci >> 1) << 3));
  }
  const int8_t *gA = Ap + (size_t)blockIdx.y * RB * k;
  const int8_t *gB = Bp + (size_t)(col0 + blockIdx.x * CB) * k;
  const uint32_t units = rank / 16u;   // 16-byte units in a row's chunk
  int32_t acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  uint32_t keep = 0u;
  const uint32_t chunks = k / rank;
  for (uint32_t c = 0; c < chunks; c++) {
    for (uint32_t u = t; u < (RB + CB) * units; u += T) {
      const uint32_t rr = u / units, qq = u % units;
      const int8_t *src = rr < RB ? gA + (size_t)rr * k : gB + (size_t)(rr - RB) * k;
      int8_t *dst = rr < RB ? sA + rr * S : sB + (rr - RB) * S;
      *reinterpret_cast<int4 *>(dst + 16u * qq) =
          *reinterpret_cast<const int4 *>(src + (size_t)c * rank + 16u * qq);
    }
    __syncthreads();
    for (uint32_t kk = 0; kk < rank; kk += 4u) {
      const int32_t a = *reinterpret_cast<const int32_t *>(sA + row * S + kk);
#pragma unroll
      for (uint32_t u = 0; u < 8u; u++)
        acc[u] = pearl_amd_dot4(a, *reinterpret_cast<const int32_t *>(sB + cols[u] * S + kk), acc[u]);
    }
    __syncthreads();
    uint32_t x = 0u;
#pragma unroll
    for (uint32_t u = 0; u < 8u; u++) x ^= (uint32_t)acc[u];
    x ^= (uint32_t)__shfl_xor((int)x, 1);
    x ^= (uint32_t)__shfl_xor((int)x, 2);
    x ^= (uint32_t)__shfl_xor((int)x, 4);
    x ^= (uint32_t)__shfl_xor((int)x, 8);
    x ^= (uint32_t)__shfl_xor((int)x, 16);
    keep = p == c ? x : keep;
  }
  if (p < PEARL_JACKPOT_BUCKETS) {
    const uint32_t rowIdx = 2u * blockIdx.y + (q >> 2);
    const uint32_t colIdx = 4u * blockIdx.x + (q & 3u);
    tr[((size_t)colIdx * rows_valid + rowIdx) * PEARL_JACKPOT_BUCKETS + p] = keep;
  }
}

namespace {
// pearl_transcript_hash_again, for AMD. That one hides its inputs from the compiler with an
// empty asm on "r" operands, which on AMD GPUs means scalar registers: the inputs are
// per-lane, so the backend refuses it ("illegal VGPR to SGPR copy"). "v" is a vector
// register. The point is the same: keep the full hash, which only the rare region that
// passes the first word needs, from being merged into every region's.
__device__ __forceinline__ void pearl_amd_hash_again(const uint32_t key[8], const uint32_t m[16],
                                                     uint32_t h[8]) {
  uint32_t k2[8], m2[16];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    k2[i] = key[i];
#if !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__)
    asm volatile("" : "+v"(k2[i]));
#endif
  }
#pragma unroll
  for (int i = 0; i < 16; i++) {
    m2[i] = m[i];
#if !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__)
    asm volatile("" : "+v"(m2[i]));
#endif
  }
  pearl_transcript_hash(k2, m2, h);
}
}  // namespace

// The transcript hash after a fold: one thread a region of the batch, the same tests and
// the same hit list as pearl_tall_hash80 in pearl_kernel.cu.
extern "C" __global__ __launch_bounds__(PEARL_AMD_THREADS) void pearl_amd_hash(
    const uint32_t *__restrict__ tr, uint32_t regions, const PearlTranscriptTest test,
    const PearlHitList hits) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= regions) return;
  uint32_t tm[16];
#pragma unroll
  for (uint32_t q = 0; q < 4u; q++) {
    const uint4 w = reinterpret_cast<const uint4 *>(tr)[(size_t)i * 4u + q];
    tm[4 * q] = w.x; tm[4 * q + 1] = w.y; tm[4 * q + 2] = w.z; tm[4 * q + 3] = w.w;
  }
  if (pearl_transcript_msw(test.key, tm, test.hash_big_endian) > test.target_w[0]) return;
  uint32_t h[8];
  pearl_amd_hash_again(test.key, tm, h);
  if (!pearl_hash_meets_words(h, test.target_w, test.hash_big_endian)) return;
  const uint32_t slot = atomicAdd(hits.count, 1u);
  if (slot >= PEARL_MAX_HITS) return;
  hits.index[slot] = i;
#pragma unroll
  for (int j = 0; j < 8; j++) hits.hash[slot * 8u + j] = h[j];
#pragma unroll
  for (int j = 0; j < 16; j++) hits.transcript[slot * 16u + j] = tm[j];
}

// Which folds this code object has bodies for, and its wave size: out[0] and out[1].
extern "C" __global__ void pearl_amd_caps(uint32_t *out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
#if PEARL_AMD_EMU
  out[0] = PEARL_AMD_CAP_REF | PEARL_AMD_CAP_MFMA16 | PEARL_AMD_CAP_MFMA8 | PEARL_AMD_CAP_WMMA11
           | PEARL_AMD_CAP_WMMA12 | PEARL_AMD_CAP_EMULATED;
#else
  out[0] = PEARL_AMD_CAP_REF | (PEARL_AMD_MFMA16 ? PEARL_AMD_CAP_MFMA16 : 0u)
           | (PEARL_AMD_MFMA8 ? PEARL_AMD_CAP_MFMA8 : 0u)
           | (PEARL_AMD_WMMA11 ? PEARL_AMD_CAP_WMMA11 : 0u)
           | (PEARL_AMD_WMMA12 ? PEARL_AMD_CAP_WMMA12 : 0u);
#endif
  out[1] = (uint32_t)warpSize;
}
