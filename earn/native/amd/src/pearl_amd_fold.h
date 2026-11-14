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
// Six folds:
//   pearl_amd_fold_ref     plain dot products from LDS, 32x64 a block. Any AMD GPU
//                          (sdot4/sudot4 where the card has them); slow, the baseline.
//   pearl_amd_fold_mfma16  CDNA3/CDNA4 (gfx942, gfx950): v_mfma_i32_32x32x16_i8, wave64.
//   pearl_amd_fold_mfma8   CDNA1/CDNA2 (gfx908, gfx90a): v_mfma_i32_32x32x8_i8, wave64.
//   pearl_amd_fold_wmma11  RDNA3 (gfx11): v_wmma_i32_16x16x16_iu8, wave32.
//   pearl_amd_fold_wmma12  RDNA4 (gfx12): v_wmma_i32_16x16x16_iu8, wave32, RDNA4's layout.
//   pearl_amd_fold_dot4    RDNA2 (gfx103x), which has no matrix instruction: v_dot4 from
//                          registers, wave32, in the matrix folds' tile. An RDNA3 or RDNA4
//                          build with -DPEARL_AMD_PF2=0 has it too, to run it on those cards.
//
// Each of the last five is the same code (pearl_amd_fold_body) over a policy that says how
// the instruction lays out its operands and results across lanes. The matrix layouts are
// AMD's, from the Matrix Instruction Calculator
// (github.com/ROCm/amd_matrix_instruction_calculator); dot4's is chosen here (PearlAmdDot4).
//
// No GPU was available to write this. So every matrix policy also has an emulation of its
// instruction, written from the same documented layout, and the HIP-CPU build
// (__HIP_CPU_RT__) runs the folds with it: each lane puts its fragments in shared memory,
// the block synchronises, and each lane computes the outputs the layout says it holds.
// Everything else in the fold -- staging, fragment loads, the readout, the transcript
// stores -- is the same source on the CPU and on the GPU. What the CPU run cannot show is
// that the hardware matches the documented layout; test/pearl_amd_foldtest.cpp checks that
// on a real card. The dot4 fold needs no emulation: the CPU runs its own mma.
#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>

#include <type_traits>

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
// The two RDNA 3 chips the buffer loads were raced on (PEARL_AMD_RDNA3_BUFLD): Navi 31 and Navi 32.
#if defined(__gfx1100__) || defined(__gfx1101__)
#define PEARL_AMD_GFX1100_1101 1
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
// The dot4 fold (PearlAmdDot4) needs a dot4 instruction in wave32: RDNA2, RDNA3 and RDNA4. CDNA
// has v_dot4 too, but runs its MFMA fold, so its code objects leave this one out. Which passes
// build it is PEARL_AMD_DOT4_FOLD, below.
#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || \
    defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) || defined(__GFX11__) || \
    defined(__GFX12__)
#define PEARL_AMD_DOT4 1
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
#ifndef PEARL_AMD_GFX1100_1101
#define PEARL_AMD_GFX1100_1101 0
#endif
#ifndef PEARL_AMD_DOT4
#define PEARL_AMD_DOT4 0
#endif

// Which folds have a body in the code object that runs (pearl_amd_caps). The host asks
// once a context: a fold without a body for the card returns at once, finds nothing and
// still reports hashrate, so the host must never launch one.
#define PEARL_AMD_CAP_REF 1u
#define PEARL_AMD_CAP_MFMA16 2u
#define PEARL_AMD_CAP_MFMA8 4u
#define PEARL_AMD_CAP_WMMA11 8u
#define PEARL_AMD_CAP_WMMA12 16u
#define PEARL_AMD_CAP_DOT4 32u
#define PEARL_AMD_CAP_EMULATED 0x100u  // the HIP-CPU build: every fold, the matrix ones emulated
#define PEARL_AMD_CAP_DROP 0x200u      // a PEARL_AMD_READOUT_DROP build: timing only, never mines
#define PEARL_AMD_CAP_KWRAP 0x400u     // the wrap readout (PEARL_AMD_KWRAP): k past 2048, the k RDNA 4 mines
#define PEARL_AMD_CAP_TIMING 0x800u    // a PEARL_AMD_ABLATE build: timing only, never mines
#define PEARL_AMD_CAP_STAMP 0x1000u    // a PEARL_AMD_STAMP build: clock stamps (pearl_amd_stamps); TIMING too
#define PEARL_AMD_CAP_RANK 0x2000u     // PEARL_AMD_CRANK: the matrix fold takes rank PEARL_FOLD_RANK only
#define PEARL_AMD_CAP_FUSE 0x4000u     // PEARL_AMD_FUSE: the matrix folds and dot4 hash their own transcripts

// Every fold is 256 threads a block.
#define PEARL_AMD_THREADS 256u
// The matrix folds' tile: 128 rows of A' by 256 columns (rows of B'), staged through LDS
// PEARL_AMD_SLAB bytes of k at a time, double-buffered. A row in LDS is the slab plus 16
// bytes, which keeps 16-byte stores aligned and spreads rows over the banks.
#define PEARL_AMD_BM 128u
#define PEARL_AMD_BN 256u
#ifndef PEARL_AMD_SLAB
#define PEARL_AMD_SLAB 32u
#endif
// The dot4 fold's slab (RDNA 2's), 64 bytes since 12 November: half the barriers and LDS stores a chunk of 32's.
// Its double-buffered LDS is 61,440 bytes a block, so a WGP still takes 2 blocks (4 waves a SIMD), as at 32 (36,864).
// RX 6800, 1400 MHz cap, hashrate.js, 3 rounds: 17.77-17.79 TH/s against 17.25-17.29 at 32 (+3.0%). The host lays
// out a fold's operands by that fold's slab (FoldInfo::slab). -DPEARL_AMD_DOT4_SLAB=32 builds the old slab; with
// RDNA 2's own slab loop (PEARL_AMD_DOT4_ADDR, the Linux default) it also needs -DPEARL_AMD_DOT4_ADDR=0.
#ifndef PEARL_AMD_DOT4_SLAB
#define PEARL_AMD_DOT4_SLAB 64u
#endif
// The reference fold's tile: one 32x64 block of C, which holds eight whole regions.
#define PEARL_AMD_REF_BM 32u
#define PEARL_AMD_REF_BN 64u

// The operand layout the matrix folds read. A' and B' are stored in per-tile order: each
// 128-row tile of A' (256-row tile of B') is k / PEARL_AMD_SLAB slabs of [tile rows][slab],
// so one slab of a tile is a contiguous run of whole cache lines. Row-major, a slab was 32
// bytes out of every row's 128-byte line, and the next three slabs fetched the same lines
// again: on an RX 9060 XT (4 MB of L2) the fold ran at 15.4 TMAC/s at the mainnet batch
// against 52.1 tiled (lane-a-sweep.sh, PC a777486d, 23 October). The host materialises them
// this way (pearl_materialize16_tiled); the reference fold reads row-major operands.
// -DPEARL_AMD_TILED=0 builds the row-major layout, to compare on a card.
#ifndef PEARL_AMD_TILED
#define PEARL_AMD_TILED 1
#endif

// PEARL_AMD_WALK: the order the matrix folds take their tiles in (issue 287, item 5). 1 by default on RDNA 4 (gfx1200
// and gfx1201), where it was raced; 0 (the old order) on every other target. -DPEARL_AMD_WALK=0 builds the old order
// everywhere, to race against. The Windows core is built with it off (build-amd.sh's SET_OS): it mines k 2048, which
// the walk wasn't timed at.
// Salad, 7 November, each walk build against v0.6.13's order on the same PC, hashrate.js (the whole miner):
//   RX 9060 XT  +0.38% (PC 9ac12ceb) and +1.85% (ffa9f87b, where the fold alone gained +1.91%)
//   RX 9070 XT  +0.56% (c10c6949) and +0.75% (1dba3ce9), at the k 8192 it mines
// Every walk build passed the fold check and verify-hits (400 of 400), and Kryptex took every share of 1 (8 on the
// 9060 XT, 6 on the 9070 XT, none rejected). 2 read 0.04-0.20% above 1 on every RDNA 4 PC, under the +0.3% bar, so 1
// is the default. On an RX 7900 XTX (45c94e40) neither is faster (hashrate.js +0.06% and -0.04%), and the stamps show
// the dispatcher doesn't deal blocks to the shader arrays in turn (19% of blocks, not 100%), so 2's premise is wrong
// there.
// The old order takes them x fastest, so the blocks on the card at once (3 a WGP) cover every column tile of the
// batch and only a few row tiles: 16 x 6 tiles on an RX 9070 XT (96 blocks), 16 x 3 on an RX 9060 XT (48). Those
// blocks read each slab of their tiles at about the same time, so the L2 fetches that slab once for all of them,
// from the Infinity Cache or memory. At RDNA 4's k one such group reads more than the L2 holds (28-38 MB on the
// 9070 XT against 8 MB), so nothing stays in L2 for the next group, and the bytes past the L2 are 16 B' tiles and
// 6 A' tiles a group. Cut into 8 x 12 tiles the same 96 blocks need 8 and 12, about a quarter fewer bytes; on the
// 9060 XT 4 x 12 needs about 40% fewer. Every block still reads its whole slab from L2, as before.
//   1  supertiles: the grid in W column tiles by H row tiles, each about the blocks the card holds at once, x
//      fastest inside one. A row of supertiles crosses the batch, every other row backwards, so the turn starts on
//      the B' the last supertile read.
//   2  supertiles, and inside each the blocks are dealt in turn to Q sub-tiles of SW x SH: block j of a supertile
//      folds a tile of sub-tile j % Q. If the dispatcher hands blocks to the shader arrays in turn (a guess: AMD
//      doesn't document it), each array then holds one compact sub-tile, so its L1 and its WGPs' L0 see fewer
//      different lines. Q is the card's shader arrays.
// The tested group-of-4 walk (removed 23 October) took every row tile for 4 column tiles, so it read all of A' from
// memory 4 times a batch; a supertile row reads A' from memory once, as the old order does.
// PEARL_AMD_WALK_W, _H, _SW and _SH set the shape; the targets below have defaults, and on any other target the walk
// is off unless W is given. The map is one-to-one on the grid (amdWalk.test.js), so every tile is folded once and
// the transcripts are the old order's. A grid that W doesn't divide, and the rows after the last whole supertile row
// (4 of 1024 at H 12), keep the old order.
#ifndef PEARL_AMD_WALK
#if defined(__gfx1200__) || defined(__gfx1201__)
#define PEARL_AMD_WALK 1
#else
#define PEARL_AMD_WALK 0
#endif
#endif
#if PEARL_AMD_WALK != 0 && PEARL_AMD_WALK != 1 && PEARL_AMD_WALK != 2
#error "PEARL_AMD_WALK is 0, 1 or 2"
#endif
// The defaults: W x H is about the blocks a card holds (3 a WGP) in the batch's column tiles (16 at batch 256, 8 at
// gfx1100's and gfx1101's 128), shaped for the fewest bytes past the L2 (an A' tile is half a B' tile), and Q is the
// shader arrays. gfx1201 (RX 9070 XT, 32 WGPs in 8 arrays): 8 x 12, sub-tiles 2 x 6. gfx1200 (RX 9060 XT, 16 WGPs
// in 4): 4 x 12, 2 x 6. gfx1100 (RX 7900 XTX, 48 WGPs in 12): 8 x 18, 2 x 6. gfx1101 (RX 7800 XT, 30 WGPs in 6):
// 8 x 12, 4 x 4. At RDNA 3's batch of 128 (8 column tiles) 1 is the old order, so there 1 against 0 times what
// compiling the map in costs, and 2 against 1 the sub-tiles alone.
// The code (ROCm 7.2, CI on 6 November): the map is 43-53 scalar instructions in the prologue, with the release's
// VGPRs, LDS and waves on all four targets. On gfx12 the rest of the fold is the release's (one VALU moved in the
// prologue, one operand order swapped in the epilogue). On gfx11 the compiler also reorders some selects and readout
// steps (753 VALU to 750), the same way in 1 and 2.
#if PEARL_AMD_WALK && !defined(PEARL_AMD_WALK_W)
#if defined(__gfx1201__)
#define PEARL_AMD_WALK_W 8
#define PEARL_AMD_WALK_H 12
#define PEARL_AMD_WALK_SW 2
#define PEARL_AMD_WALK_SH 6
#elif defined(__gfx1200__)
#define PEARL_AMD_WALK_W 4
#define PEARL_AMD_WALK_H 12
#define PEARL_AMD_WALK_SW 2
#define PEARL_AMD_WALK_SH 6
#elif defined(__gfx1100__)
#define PEARL_AMD_WALK_W 8
#define PEARL_AMD_WALK_H 18
#define PEARL_AMD_WALK_SW 2
#define PEARL_AMD_WALK_SH 6
#elif defined(__gfx1101__)
#define PEARL_AMD_WALK_W 8
#define PEARL_AMD_WALK_H 12
#define PEARL_AMD_WALK_SW 4
#define PEARL_AMD_WALK_SH 4
#endif
#endif
#ifndef PEARL_AMD_WALK_W
#define PEARL_AMD_WALK_W 0
#endif
#ifndef PEARL_AMD_WALK_H
#define PEARL_AMD_WALK_H 1
#endif
#ifndef PEARL_AMD_WALK_SW
#define PEARL_AMD_WALK_SW PEARL_AMD_WALK_W
#endif
#ifndef PEARL_AMD_WALK_SH
#define PEARL_AMD_WALK_SH PEARL_AMD_WALK_H
#endif
// Whether this pass remaps its tiles: the walk asked for, and a shape for this target.
#define PEARL_AMD_WALK_ON (PEARL_AMD_WALK != 0 && PEARL_AMD_WALK_W > 0)

typedef int32_t pearl_amd_v16i __attribute__((ext_vector_type(16)));
typedef int32_t pearl_amd_v8i __attribute__((ext_vector_type(8)));
typedef int32_t pearl_amd_v4i __attribute__((ext_vector_type(4)));
typedef int32_t pearl_amd_v2i __attribute__((ext_vector_type(2)));

namespace {

__device__ __forceinline__ int32_t pearl_amd_sbyte(uint64_t v, int i) {
  return (int32_t)(int8_t)(uint8_t)(v >> (8 * i));
}

// c plus the dot product of a's and b's four signed bytes: one v_dot4 where the card has it,
// else four multiplies (the HIP-CPU build, and a GPU without one).
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
// kEmulated says the HIP-CPU build replaces mma with the emulation built from a_elem, b_elem,
// c_row and c_col; the dot4 policy has none of those, since the CPU runs its mma as it is.
// ---------------------------------------------------------------------------------------

// CDNA3 and CDNA4: v_mfma_i32_32x32x16_i8. Lane l holds A row l % 32 at k 8*(l/32) .. +7 and
// B column l % 32 at the same k; accumulator g of lane l is C[8*(g/4) + 4*(l/32) + g%4][l%32].
struct PearlAmdMfma16 {
  static constexpr int kWave = 64, kTM = 32, kTN = 32, kTK = 16;
  static constexpr int kWR = 2, kWC = 2, kMT = 2, kNT = 4;
  static constexpr int kRegs = 16;
  static constexpr bool kEmulated = true;
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
  static constexpr bool kEmulated = true;
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
  static constexpr bool kEmulated = true;
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
  static constexpr bool kEmulated = true;
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

// PEARL_AMD_DOT4_K: the bytes of k a dot4 k-step reads from each row, 4, 8 or 16 (one ds_read_b32,
// b64 or b128 a row). More bytes a step means fewer LDS reads, each wider, and more VGPRs for the
// fragments: 24, 48 or 96 a lane. 4 and 8 timed the same in RDNA 2's loop on an RX 7800 XT (PC
// 16f1685c, 2 November: 18.09 and 17.99 TMAC/s against 18.01 and 17.91, two rounds), and 4 takes
// fewer VGPRs (201 against 225 on gfx1030), so 4.
#ifndef PEARL_AMD_DOT4_K
#define PEARL_AMD_DOT4_K 4
#endif
#if PEARL_AMD_DOT4_K != 4 && PEARL_AMD_DOT4_K != 8 && PEARL_AMD_DOT4_K != 16
#error "PEARL_AMD_DOT4_K is 4, 8 or 16"
#endif
// PEARL_AMD_DOT4_SPLIT=1: an empty asm with a memory clobber before each k-step's loads, so the
// compiler keeps each k-step's LDS reads apart. Without it the compiler merged a row's 8 k-steps of
// a slab into two ds_read_b128 and waited for each one (lgkmcnt(0)) right before the 32 dot4s that
// use it, so no read was in flight while dot4s ran, and on gfx11 the slab's fragments didn't fit:
// 256 VGPRs and 64 spilled, scratch loads in the slab loop. With it each k-step's 24 words are read
// while earlier k-steps' dot4s run (most waits leave 2 to 9 reads in flight), gfx1101 takes 205
// VGPRs and spills none, and gfx1030 201 (211 without). In RDNA 2's loop on an RX 7800 XT the fold went from 10.26
// and 10.21 TMAC/s to 18.09 and 17.99 (PC 16f1685c, 2 November), where the card's v_dot4 peaks at
// 18.5-19.2 with nothing else to do (RDNA 3 runs v_dot4_i32_iu8 at a little over half the rate of
// its WMMA), so this is 94-98% of it. On by default; =0 builds the merged reads, to compare.
#ifndef PEARL_AMD_DOT4_SPLIT
#define PEARL_AMD_DOT4_SPLIT 1
#endif

// RDNA2 (gfx103x) has no matrix instruction, but by AMD's figures v_dot4_i32_i8 runs at full rate
// there: 4 int8 MACs a lane a clock, 256 a CU a clock, the rate RDNA3's v_wmma_i32_16x16x16_iu8
// has. (RDNA3's v_dot4_i32_iu8 measured about half that, 140 a CU a clock on an RX 7800 XT.) So
// each lane sums its own part of C with dot4s from registers, and the rest is the matrix folds'
// (the tile, the per-tile operands, the slab staging, the readout, the store). Here one "tile" is
// a whole 32x64 block of C; a wave holds two, 64x64, as wmma11 and wmma12 do.
//
// Lane l, with i = (l >> 3) & 3, holds rows 4*(rho & 1) + i + 8*(rho >> 1) for rho = 0..7 and
// columns (l & 7) + 8*j for j = 0..7 of the block: accumulator g = 8*rho + j is
// C[row(rho)][col(j)], 64 a tile. Rows 4*ro + {0-3, 8-11, 16-19, 24-27} are row offset ro, so g bit
// 3 is the region's row offset (ro_of). Columns {0, 1, 8, 9, ..., 56, 57} + 2*co are column offset
// co = (l >> 1) & 3, lane bits 1 and 2, where the store looks for it. A lane holds 32 sums of each of
// its two regions (4 rows by 8 columns), and the eight lanes of a region (lane bits 0, 3 and 4) hold
// its 256, so pearl_amd_group_xor finishes it as it does the matrix folds'.
//
// A k-step (kTK bytes) reads the lane's 8 A' rows of each tile and its 8 B' rows (columns of C),
// and does 64 dot4s a tile for each 4 bytes: at kTK 4, 24 words from LDS (the compiler pairs them
// into ds_read2_b32) and 128 v_dot4 a lane over the wave's two tiles (PEARL_AMD_DOT4_SPLIT says how
// the reads are kept apart). LDS rows are 48 bytes apart, 12 banks. An A' read touches 4 rows (i), banks
// 0, 12, 24, 4 past the k offset, each word read by 8 lanes at once; a B' read touches 8 consecutive
// rows, banks 0, 12, 24, 4, 16, 28, 8, 20, each word read by 4 lanes. No two rows of a read share a
// bank, of 32 or of 64, at b32, and the 2 or 4 banks each row takes at b64 or b128 don't overlap.
//
// No emulation (kEmulated false): v_dot4 is a sum of four products, so the HIP-CPU build runs this
// mma itself, with pearl_amd_dot4's scalar form. The layout the CPU check tests is the GPU's.
struct PearlAmdDot4 {
  static constexpr int kWave = 32, kTM = 32, kTN = 64, kTK = PEARL_AMD_DOT4_K;
  static constexpr int kWR = 2, kWC = 4, kMT = 2, kNT = 1;
  static constexpr int kRegs = 64;
  static constexpr bool kEmulated = false;
  static constexpr int kQ = kTK / 4;   // 4-byte words of k a row holds in a fragment
  typedef int32_t AFrag __attribute__((ext_vector_type(8 * kQ)));   // word q of row rho: [kQ * rho + q]
  typedef AFrag BFrag;                                              // word q of column j: [kQ * j + q]
  typedef int32_t Acc[kRegs];
  // kQ words at p, into f from word w.
  static __device__ __forceinline__ void load_row(AFrag &f, int w, const int8_t *p) {
    if constexpr (kQ == 1) {
      f[w] = *reinterpret_cast<const int32_t *>(p);
    } else if constexpr (kQ == 2) {
      const pearl_amd_v2i v = *reinterpret_cast<const pearl_amd_v2i *>(p);
      f[w] = v[0]; f[w + 1] = v[1];
    } else {
      const pearl_amd_v4i v = *reinterpret_cast<const pearl_amd_v4i *>(p);
      f[w] = v[0]; f[w + 1] = v[1]; f[w + 2] = v[2]; f[w + 3] = v[3];
    }
  }
  static __device__ __forceinline__ AFrag load_a(const int8_t *s, int stride, int lane) {
#if PEARL_AMD_DOT4_SPLIT && !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__)
    asm volatile("" ::: "memory");
#endif
    AFrag f;
#pragma unroll
    for (int rho = 0; rho < 8; rho++)
      load_row(f, kQ * rho, s + (4 * (rho & 1) + ((lane >> 3) & 3) + 8 * (rho >> 1)) * stride);
    return f;
  }
  static __device__ __forceinline__ BFrag load_b(const int8_t *s, int stride, int lane) {
    BFrag f;
#pragma unroll
    for (int j = 0; j < 8; j++) load_row(f, kQ * j, s + ((lane & 7) + 8 * j) * stride);
    return f;
  }
  static __device__ __forceinline__ void mma(Acc &c, AFrag a, BFrag b) {
#pragma unroll
    for (int q = 0; q < kQ; q++)
#pragma unroll
      for (int rho = 0; rho < 8; rho++)
#pragma unroll
        for (int j = 0; j < 8; j++) c[8 * rho + j] = pearl_amd_dot4(a[kQ * rho + q], b[kQ * j + q], c[8 * rho + j]);
  }
  static constexpr bool kRoByLane = false;
  static __device__ __forceinline__ int ro_of(int g) { return (g >> 3) & 1; }
};

// PEARL_AMD_DOT4_DPPA (RDNA 2, on by default): a k-step reads one A' word a lane a tile from LDS, not eight. The
// eight lanes of a dpp8 group (lane bits 0-2) hold the same eight A' rows, so lane r of a group reads row rho = r,
// and each v_dot4c takes row rho's word from lane rho of its group through its own dpp8 operand
// (v_dot4c_i32_i8_dpp: no extra instruction). A k-step's LDS reads fall from 24 words a lane to 10.
// RX 6800, 1400 MHz cap, 12 November: hashrate.js 18.04 TH/s against 17.26 (+4.5%); 191 VGPRs, 4 waves. A
// dot4c with a dpp8 operand ran 243 MACs a CU a clock in a probe of nothing else (253 without), so this only
// pays because the LDS reads were what held the loop: a timing-only build that read each slab's fragments once
// was +10.9% (no barrier +6.2%, no staging +3.8%; same card, fold alone, 17.33 TMAC/s for the release). -DPEARL_AMD_DOT4_DPPA=0 builds the old k-step, to compare.
#ifndef PEARL_AMD_DOT4_DPPA
#define PEARL_AMD_DOT4_DPPA 1
#endif
// PEARL_AMD_DOT4_DPPA_KB: bytes of k each LDS read of the DPPA k-step takes, 8 (default since 13 November) or 4.
// At 8 a pass reads each A' and B' word pair with one ds_read_b64 and runs two k-steps on them: half the LDS
// instructions. RX 6800 and RX 6600, 1400 MHz cap, fold alone: +2.9% and +3.1% (18.97 against 18.44 TMAC/s, 8.74
// against 8.48); 217 VGPRs, still 4 waves. 16 (ds_read_b128, four k-steps) took 249 VGPRs and ran 12% slower.
// Timing-only builds of the 8-byte loop then priced the slab barrier at +3.1%, the staging at +2.6% and the readout
// at +1.7% (earn/tuning/benchmark.md). -DPEARL_AMD_DOT4_DPPA_KB=4 builds the 4-byte k-step, to compare.
#ifndef PEARL_AMD_DOT4_DPPA_KB
#define PEARL_AMD_DOT4_DPPA_KB 8
#endif
#if PEARL_AMD_DOT4_DPPA_KB != 4 && PEARL_AMD_DOT4_DPPA_KB != 8
#error "PEARL_AMD_DOT4_DPPA_KB is 4 or 8"
#endif
#if PEARL_AMD_DOT4_DPPA && PEARL_AMD_SDOT4 && PEARL_AMD_DOT4 && !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__)
#define PEARL_AMD_DOT4_DPPA_ON 1
static_assert(PEARL_AMD_DOT4_K == 4, "PEARL_AMD_DOT4_DPPA is for K 4");
#define PEARL_AMD_DPP8_ALL(R) "dpp8:[" #R "," #R "," #R "," #R "," #R "," #R "," #R "," #R "]"
#define PEARL_AMD_D4P(R) \
  if constexpr (RHO == R) asm("v_dot4c_i32_i8_dpp %0, %1, %2 " PEARL_AMD_DPP8_ALL(R) : "+v"(c) : "v"(a), "v"(b))
// c plus the dot4 of a from lane RHO of this lane's dpp8 group with this lane's b.
template <int RHO>
__device__ __forceinline__ void pearl_amd_dot4_from(int32_t &c, int32_t a, int32_t b) {
  PEARL_AMD_D4P(0); PEARL_AMD_D4P(1); PEARL_AMD_D4P(2); PEARL_AMD_D4P(3);
  PEARL_AMD_D4P(4); PEARL_AMD_D4P(5); PEARL_AMD_D4P(6); PEARL_AMD_D4P(7);
}
#undef PEARL_AMD_D4P
#else
#define PEARL_AMD_DOT4_DPPA_ON 0
#endif

// One k-step of a wave: every tile of the wave's kMT x kNT, from the wave's A and B
// fragments. On the GPU, one instruction a tile (dot4: 64 v_dot4 a 4 bytes). In the HIP-CPU
// build, the emulation, but dot4's own mma. Every thread of the block calls it at the same
// point, which the emulation's barriers need.
template <class P>
__device__ __forceinline__ void pearl_amd_mma_step(typename P::Acc (&acc)[P::kMT][P::kNT],
                                                   const typename P::AFrag (&a)[P::kMT],
                                                   const typename P::BFrag (&b)[P::kNT]) {
#if PEARL_AMD_EMU
  if constexpr (P::kEmulated) {
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
    return;
  }
#endif
#pragma unroll
  for (int mt = 0; mt < P::kMT; mt++)
#pragma unroll
    for (int nt = 0; nt < P::kNT; nt++) P::mma(acc[mt][nt], a[mt], b[nt]);
}

// PEARL_AMD_PF2=1: global loads two slabs ahead instead of one (two register sets, the slab
// loop unrolled by two so each set and LDS buffer is fixed), meant to give a load two slabs of
// matrix work to arrive in before its LDS store waits on it. The compiler doesn't keep it that
// far ahead (the review of 23 October, in the ROCm 7.2 -O3 ISA): on gfx1201 it moves slab s+2's
// loads below slab s's matrix instructions, about one slab ahead, and on gfx1100 (no PFX) the
// LDS store and the barrier still wait for every load. The measured gains are real, but they
// come mostly from the unrolled loop, the DPP readout and, on RDNA 4, PFX's barrier. Pinning
// the loads at the top of the step: PEARL_AMD_LPIN and PEARL_AMD_PF2PIN. On by default on
// RDNA 3 and RDNA 4. An RX
// 7800 XT (PC 16f1685c) ran hashrate.js at 31.86-31.93 TH/s with it against 31.10-31.16
// without, and 32.17-32.28 with the DPP readout below as well. RDNA 4: see PFX.
#ifndef PEARL_AMD_PF2
#define PEARL_AMD_PF2 (PEARL_AMD_WMMA11 || PEARL_AMD_WMMA12)
#endif
// PEARL_AMD_DOT4_ADDR (RDNA 2, on by default since 13 November): RDNA 2's slab loop (DPPA, 8-byte reads) with its
// addresses out of the loop. The shipped loop spent about 51 VALU a slab of 2048 dot4s on addresses: 24 on the slab
// loads' 64-bit addresses, 10 on the LDS stores' and 12 on the LDS read bases, which the loop's parity (s & 1) made
// new every slab. Here the slab loads are buffer loads (PEARL_AMD_BUFLD, with gfx10's word 3, and PEARL_AMD_TID);
// LDS is [A' 0][B' 0][A' 1][B' 1], so slab s's buffers are one SGPR offset from buffer 0's, the same for A' and B';
// and every LDS address is a lane base plus that offset plus an immediate: 6 VALU a slab (four read bases, the
// store's, and the offset's flip, in the gfx1030 assembly with ROCm 7.2's clang). RDNA 2 is VALU-bound
// there, so the VALU are time: RX 6800, 1400 MHz cap, hashrate.js 19.08-19.12 TH/s against 18.95-18.96 (+0.8%).
// PEARL_AMD_DOT4_SWZ (in that loop, on by default): LDS bank conflicts. Rows are 80 bytes, 20 banks apart, so rows 8
// apart share a bank: a k-pair's A' read (rows 0, 4, 8, ... 28 and 1, 5, ... 29 in each 16 lanes) hit 4 rows a
// bank. A' row r of a tile goes to LDS row r ^ (((r >> 3) & 3) << 1), which spreads them 2 a bank (at 80-byte rows,
// 8-byte reads can't do better). And each eight lanes of a 16-byte staging store wrote rows r and r + 1, whose banks
// overlap by 4; here they write rows 4 apart, which cover the 32 banks once. B' reads were conflict-free and stay
// so. With ADDR: 19.31-19.34 TH/s (+2.0% on the shipped loop, the same 1366-1368 MHz and 120 W).
// The Windows core builds with -DPEARL_AMD_DOT4_ADDR=0 (build-amd.sh's SET_OS): both were timed only with ROCm's clang
// 22 on Linux. -DPEARL_AMD_DOT4_ADDR=0 builds the loop before them, -DPEARL_AMD_DOT4_SWZ=0 ADDR without the swizzle.
#ifndef PEARL_AMD_DOT4_ADDR
#define PEARL_AMD_DOT4_ADDR 1
#endif
#if PEARL_AMD_DOT4_ADDR != 0 && PEARL_AMD_DOT4_ADDR != 1
#error "PEARL_AMD_DOT4_ADDR is 0 or 1"
#endif
#if PEARL_AMD_DOT4_ADDR && PEARL_AMD_DOT4_DPPA_ON && PEARL_AMD_DOT4_DPPA_KB == 8 && !PEARL_AMD_PF2
#define PEARL_AMD_DOT4_ADDR_ON 1
#else
#define PEARL_AMD_DOT4_ADDR_ON 0
#endif
#ifndef PEARL_AMD_DOT4_SWZ
#define PEARL_AMD_DOT4_SWZ 1
#endif
#if PEARL_AMD_DOT4_SWZ != 0 && PEARL_AMD_DOT4_SWZ != 1
#error "PEARL_AMD_DOT4_SWZ is 0 or 1"
#endif
// PEARL_AMD_DOT4_STKP=n, in PEARL_AMD_DOT4_ADDR's loop: the next slab's LDS store issued after k-pair n's reads
// (0-7), not after the slab's dot4s (8, the default), so it completes before the barrier. At 6, +0.2% on the RX 6800
// and +0.1% on an RX 6600 (hashrate.js 19.36 against 19.31-19.34, 8.92 against 8.91); at 7, +0.0%. Off: too small to
// ship without a pool run of its own.
#ifndef PEARL_AMD_DOT4_STKP
#define PEARL_AMD_DOT4_STKP 8
#endif
// PEARL_AMD_RDNA3_BUFLD: RDNA 3's buffer loads, PEARL_AMD_BUFLD and PEARL_AMD_TID together (see BUFLD, below). On by
// default on gfx1100 and gfx1101 only, the two chips it was raced on; gfx1102, gfx1150 and gfx1151 keep their global
// loads. The Windows core leaves it out (build-amd.sh's SET_OS). -DPEARL_AMD_RDNA3_BUFLD=0 builds the release before
// it, to race against.
#ifndef PEARL_AMD_RDNA3_BUFLD
#define PEARL_AMD_RDNA3_BUFLD PEARL_AMD_GFX1100_1101
#endif
#if PEARL_AMD_RDNA3_BUFLD != 0 && PEARL_AMD_RDNA3_BUFLD != 1
#error "PEARL_AMD_RDNA3_BUFLD is 0 or 1"
#endif
// PEARL_AMD_RDNA3_SET (issue #287 item 8): RDNA 3's staging cuts, on by default where its buffer loads are
// (PEARL_AMD_RDNA3_BUFLD, so gfx1100 and gfx1101 in the Linux core), the only code they were raced on. In gfx11 code
// it sets PEARL_AMD_PFX and, with RDNA 4's set, PEARL_AMD_CRANK and PEARL_AMD_SCHEDB (each below). CRANK + SCHEDB
// drop the per-slab division and the readout code an even slab carried but never ran (the fold's code has 32 LDS
// loads, not 48), but alone they lost 1.2% on an RX 7800 XT: three of a pair's six slab loads then have no WMMA
// between their issue and the barrier's wait for them. With PFX the barrier no longer waits for the loads, and they get 32 to 96 WMMAs. The wmma11 fold takes
// 226 VGPRs, 6 waves. Against the buffer-load fold on the same PC, 7 November, with a twin of it for the noise:
// on an RX 7900 XTX (28a12f15) the fold alone +0.60 to +0.88% a round and hashrate.js +0.89 to +1.20%; on an
// RX 7800 XT (72dd708c) +0.83 to +0.89% and +0.83 to +0.93%, verify-hits 400 of 400; on Kryptex
// the XTX got 6 accepted shares and 0 rejected in 10 minutes.
// -DPEARL_AMD_RDNA3_SET=0 builds item 2's fold to race against; the Windows core leaves it out (SET_OS).
#ifndef PEARL_AMD_RDNA3_SET
#define PEARL_AMD_RDNA3_SET PEARL_AMD_RDNA3_BUFLD
#endif
#if PEARL_AMD_RDNA3_SET != 0 && PEARL_AMD_RDNA3_SET != 1
#error "PEARL_AMD_RDNA3_SET is 0 or 1"
#endif
#define PEARL_AMD_RDNA3_SET_ON (PEARL_AMD_RDNA3_SET && PEARL_AMD_WMMA11)
// PEARL_AMD_PF2PIN=1 (issue #287 item 8): RDNA 3's PF2 loop with every load two slabs ahead in fact. In gfx11 code
// only, it sets PEARL_AMD_PFX, PEARL_AMD_TID and PEARL_AMD_LPIN=2 (each below), which can still be set one by one.
// Off by default; the races below show why. What the gfx1100 assembly showed (ROCm 7.2):
//   - The release issues slab s+2's loads at the top of step s, but __syncthreads()'s fence waits for them at the
//     end of step s (vmcnt(0)): each load gets the 32 WMMAs of one slab, and its registers then sit a slab unused.
//   - PFX alone (timed level on an RX 7900 XT) lets the compiler sink the loads to the end of the step: one slab.
//   - PFX + LPIN=1 keeps the loads of one step of the unrolled pair at the top; the other step's sink past the
//     sched_barrier into the block after its readout. Machine sinking stops a load only at a later store in its
//     block, and a sched_barrier isn't one.
//   - PFX + LPIN=2 keeps all six at the top: each gets 64 WMMAs. 235 VGPRs against 240, 6 waves, no spills; 233
//     with PEARL_AMD_BUFLD, 230 with PEARL_AMD_CRANK and PEARL_AMD_SCHEDB as well, each still two slabs ahead.
// Raced on top of the buffer loads, 7 November: the stamps of the buffer-load fold put the memory wait at the release
// fence at 0.76-0.79% of the slab loop on two RX 7900 XTX PCs and 1.85% on an RX 7800 XT. PF2PIN alone ran the fold
// +0.05%, +0.42% and +0.25% on those three PCs (hashrate.js +0.24% on the second, inside its noise). With CRANK and
// SCHEDB it ran +0.66% in hashrate.js on the 7800 XT, but PEARL_AMD_RDNA3_SET without the pin (226 VGPRs, three loads
// one slab ahead and three two) ran +0.88% there, and +1.04% against +0.38% on an RX 7900 XTX (the fold +0.70%
// against +0.52%).
#ifndef PEARL_AMD_PF2PIN
#define PEARL_AMD_PF2PIN 0
#endif
#define PEARL_AMD_PF2PIN_ON (PEARL_AMD_PF2PIN && PEARL_AMD_WMMA11)
// PEARL_AMD_PFX=1, with PF2: keep the loads two slabs ahead in flight across the barrier. Two
// things stopped that. __syncthreads() is a workgroup fence for every address space, which
// waits for every outstanding global load at each slab barrier (vmcnt(0) and an L0 invalidate
// on gfx11; loadcnt(0) and global_inv scope:SCOPE_SE on gfx12). And the loads were issued
// under `if`, so where the two paths met the compiler waited for all of them before the LDS
// stores (vmcnt(0), not 3). So the slab barrier fences LDS only (the loads go to registers;
// nothing else is shared), and the last two steps load the last slab again instead of skipping
// (two slabs in 64, from L2). PEARL_AMD_TID=1 tells the compiler threadIdx.x < 256, which drops
// an exec-mask branch in the staging addresses (wmma12 215 -> 198 VGPRs; wmma11 240 -> 231).
// PFX is on by default on RDNA 4 and with PEARL_AMD_RDNA3_SET or PEARL_AMD_PF2PIN; TID on RDNA 4, with RDNA 3's
// buffer loads (above) and with either of those. An RX 9060 XT (PC ca4f64e4) ran hashrate.js at 52.88 TH/s on
// the old path (one slab ahead, shuffle readout), 54.52 with DPP, 57.41 with PF2 + DPP, 59.99
// with PF2 + DPP + PFX + TID; the old path against all four on other PCs: 9060 XT d324328f
// 56.27-56.34 against 63.92-63.94, 54b217a9 28.16 against 32.85; 9070 XT 83151dba 81.26-81.33
// against 89.05-89.81, 77904d36 97.81-97.85 against 110.09-110.49. On RDNA 3 PFX + TID changed
// nothing (RX 7900 XT 67c308d2, 47.13-47.23 against 47.22-47.26), so PFX alone stays off there.
#ifndef PEARL_AMD_PFX
#define PEARL_AMD_PFX (PEARL_AMD_WMMA12 || PEARL_AMD_PF2PIN_ON || PEARL_AMD_RDNA3_SET_ON)
#endif
#ifndef PEARL_AMD_TID
#define PEARL_AMD_TID (PEARL_AMD_WMMA12 || (PEARL_AMD_WMMA11 && PEARL_AMD_RDNA3_BUFLD) || PEARL_AMD_PF2PIN_ON || PEARL_AMD_RDNA3_SET_ON || PEARL_AMD_DOT4_ADDR_ON)
#endif
// PEARL_AMD_KPAIR=1 (RDNA 4, wmma12): each lane reads its fragments for two k-steps with one 16-byte LDS load.
// A slab's k-steps don't have to take its bytes in order: the WMMAs of a slab add up over all of them, and a chunk
// ends only on a slab boundary. So lanes 0-15 take bytes 0-15 of each 32 and lanes 16-31 bytes 16-31, the first
// 8 for the even k-step and the next 8 for the odd one, in A' and B' alike. The k-step pair then reads 16 bytes a
// lane at a 16-bit offset (ds_load_b128) where the old order read two 8-byte words 16 bytes apart
// (ds_load_2addr_b64, whose 8-bit offsets needed a new address register, a VALU, for most fragments). On RDNA 4 a
// VALU beside the WMMAs costs about a cycle of WMMA time (pearl_amd_peak's valui variants), so the fold alone ran
// 5.9-6.8% faster on an RX 9070 XT (earn/tuning/benchmark.md, 4 November). On by default on RDNA 4; -DPEARL_AMD_KPAIR=0 builds
// the old order to race against. RDNA 3's fragments are 16 bytes already.
#ifndef PEARL_AMD_KPAIR
#define PEARL_AMD_KPAIR PEARL_AMD_WMMA12
#endif
// PEARL_AMD_BUFLD=1: the slab loads as buffer loads. A buffer resource for the block's tile of A' and of B' (in
// SGPRs), the lane's place in the slab as the VGPR offset, which never changes, and the slab's start as the SGPR
// offset: no VALU a load, where the global loads took a 64-bit VALU add of the slab to each lane's address (two
// VALU a load). Word 3 is the one AMD's Composable Kernel uses on gfx11 and gfx12. Every load stays inside its
// tile, the resource's size (amdBufld.test.js). On top of PEARL_AMD_KPAIR the fold alone ran another 2.2%
// faster on an RX 9070 XT (earn/tuning/benchmark.md, 4 November). On by default on RDNA 4.
// RDNA 3 (wmma11) takes it with PEARL_AMD_TID on gfx1100 and gfx1101 (PEARL_AMD_RDNA3_BUFLD, above; issue #287,
// item 2). Its fragments, LDS layout and loads two slabs ahead stay as they are; only the slab loads' addressing
// changes. Without TID the first staging unit's A'-or-B' test stays a run-time branch. The fold without them spent
// 12 VALU a slab on load addresses (8 of them that test's two arms, both run under an exec mask, in chains of 64-bit
// adds); with both the loads take none, and the fold 232 VGPRs, not 240, still 6 waves (ROCm 7.2.0). On RDNA 3 a
// chained VALU beside the WMMAs costs about 4-5 clocks of WMMA time and a lone one about 1 (pearl_amd_peak on an
// RX 7900 XTX and an RX 7800 XT), so it paid more than the 1.2% a clock each would. Against the release on the same
// PC, 7 November: on the RX 7900 XTX (57cc1801) the fold alone +1.02 to +1.25% a round and hashrate.js +0.66 to
// +0.77%, with TID alone +0.16% and +0.04%; on the RX 7800 XT (72dd708c) +0.81 to +1.34% and +0.75 to +0.82%. Only
// gfx11 and gfx12 code takes buffer loads, whatever the setting: word 3 is theirs, and CDNA and RDNA 2 keep their
// global loads.
#ifndef PEARL_AMD_BUFLD
#define PEARL_AMD_BUFLD (PEARL_AMD_WMMA12 || (PEARL_AMD_WMMA11 && PEARL_AMD_RDNA3_BUFLD) || PEARL_AMD_DOT4_ADDR_ON)
#endif
// PEARL_AMD_RDNA4_SET: the RDNA 4 cuts of 4 November, on by default in RDNA 4 (gfx12) code: PEARL_AMD_RDSEL=1,
// PEARL_AMD_RDPERM=2, PEARL_AMD_CRANK=1, PEARL_AMD_SCHEDB=1, PEARL_AMD_LDS32=1 and PEARL_AMD_LDS32_PIN=1, each
// described below. The wmma12 fold then takes 201 VGPRs and 24,576 bytes of LDS, not 193 and 36,864, and still runs
// 6 waves a SIMD (7 by registers, but a block is 8 waves). hashrate.js against the fold without them, interleaved
// on one PC: +4.4 to +4.6% on an RX 9070 XT and +5.1 to +5.4% on an RX 9060 XT (two PCs each), then +4.3% and
// +4.6% on fresh PCs, with verify-hits 400 of 400 on both cards every time (earn/tuning/benchmark.md, 4 November).
// -DPEARL_AMD_RDNA4_SET=0 builds the fold without them, to race against; each switch can still be set on its own.
// The Windows core builds with it off (build-amd.sh): it was timed with ROCm's clang 22 on Linux, one of its parts
// alone (CRANK) lost 1.8% there to the scheduler, and the HIP SDK's clang 21 has never built it.
// Alone the parts read +1.2 to +3.4%. They overlap, but each still pays on top of the others (9070 XT): RDSEL +
// RDPERM +0.7%, CRANK + SCHEDB +1.9%, LDS32 + PIN +0.9% (without PIN 0.5% less; PIN alone at 48-byte rows lost a
// little). Left out, because they stop paying in the set: STOFF, level once LDS32's pinned store offset is in; the
// split slab barrier (s_barrier_signal after a slab's LDS work, the wait before the next slab's reads), 1.1 to 1.4%
// behind on top of CRANK + SCHEDB; and PEARL_AMD_WAVES=8, which fits the set in 189 VGPRs for 8 waves but lost 2.1
// to 2.6%. The cards are at their power limit: a slab falls from 2083 to 1888 cycles, but the clock from 2316 to
// 2266 MHz on the 9070 XT and from 2343 to 2140 MHz on the 9060 XT (stamps).
#ifndef PEARL_AMD_RDNA4_SET
#define PEARL_AMD_RDNA4_SET PEARL_AMD_WMMA12
#endif
#if PEARL_AMD_RDNA4_SET != 0 && PEARL_AMD_RDNA4_SET != 1
#error "PEARL_AMD_RDNA4_SET is 0 or 1"
#endif
// Wrap-readout cuts for RDNA 4 (wmma12, PEARL_AMD_KWRAP), in PEARL_AMD_RDNA4_SET:
//   PEARL_AMD_RDSEL=1    c & 8 picks keepLo or keepHi with a scalar branch. The compiler had made it selects: each
//                        keep word picked, XORed and written back, three v_cndmask a word. An asm comment in each
//                        arm keeps the arms apart, and the arms change copies local to the readout (see there).
//   PEARL_AMD_RDPERM=2   v_permlanex16's old value from an empty asm, so the compiler needn't copy x first (the
//                        v_dual_mov before it). The old value is a register nothing wrote, which is right only while
//                        every lane is active at the readout, as it is: 256-thread blocks, no early return, and only
//                        uniform branches around it. A lane off would get that register, not its x. =1 sets
//                        fetch-inactive and bound_ctrl instead, so the compiler drops the old value: the same code,
//                        timed 0.3 points lower on a quiet PC.
// RX 9070 XT, PC 3fd19936, 4 November, fold alone at k 4096, rounds 2-6 against the default then: RDSEL=1 +0.74 to
// +0.80%, RDPERM=2 +0.21 to +0.40%, both +1.07 to +1.30% (three runs). The bound for any cut here is
// PEARL_AMD_ABLATE bit 64, the readout with no cross-lane step, select or branch: +3.64 to +3.78%.
// Tried and dropped (earn/tuning/benchmark.md): the keeping lane by exec mask (+0.35 to +0.83%, below RDSEL=1 alone); each
// lane keeping its own part of all 16 words and combining the 8 lanes after the loop (rotl 13 and XOR commute):
// in registers it needs 256 VGPRs and spills, or 230 with a lane-pair DPP step, but the 16-way switch or the
// indexed vector that picks word c % 16 cost 30-50 v_mov a readout (-0.7 to +0.9%); in scratch, one store a chunk
// at an SGPR offset and no VALU, it lost 6.6 to 8.2%.
#ifndef PEARL_AMD_RDSEL
#define PEARL_AMD_RDSEL PEARL_AMD_RDNA4_SET
#endif
#ifndef PEARL_AMD_RDPERM
#define PEARL_AMD_RDPERM (2 * PEARL_AMD_RDNA4_SET)
#endif
// Readout scheduling for RDNA 4 (wmma12 with the set's KPAIR, CRANK and PF2 loop, and the wrap readout), issue #287
// item 4. On by default with PEARL_AMD_RDNA4_SET: RDLATE=2 and RDTREE=1 (RDLATE's default is below PEARL_AMD_KWRAP's,
// which it needs). Before them the slab loop had no VALU but the readout's, about 88 a chunk (64 XORs of the 128
// sums, 16 cross-lane steps, the keep), and it ran after the chunk's last WMMAs, just before that slab's stores and
// barrier. Every slab then starts with each wave waiting on its fragment loads from LDS, and the blocks on a WGP run
// in step (item 14), so nothing else filled that wait.
//   PEARL_AMD_RDLATE=1   chunk c's readout runs in the next slab (4c + 4) after its fragment loads are issued and
//                        before its WMMAs, which change the sums only after it. Its VALU then run while the loads are
//                        in flight, and the chunk's last slab reaches its barrier sooner. The last chunk's readout
//                        runs after the slab loop.
//   PEARL_AMD_RDLATE=2   as 1 for the XORs of the sums only; the cross-lane steps and the keep run one slab later
//                        (4c + 5), at the same point, from 4 VGPRs kept across that slab. Two waits filled, not one.
//   PEARL_AMD_RDTREE=1   each region's 32 sums XORed as three chains joined at the end, not one chain: 12
//                        independent chains a chunk, not 4, the same 16 instructions a region. The release's 4
//                        chains wait on each other (an s_delay_alu before most XORs), and a VALU that waits costs
//                        RDNA 4 about 4.5 cycles of WMMA time against 1.1 for one that doesn't (pearl_amd_peak).
// The transcript is the same with any of them. They change only where and in what order the readout runs.
// hashrate.js against the release, interleaved on one PC each (7 November), while a copy of the release read within
// 0.1%: on an RX 9070 XT at k 8192 RDTREE alone +1.2 to +1.3%, RDLATE=1 or 2 alone +1.3%, both (RDLATE=2, RDTREE=1)
// +1.7%; on an RX 9060 XT at k 4096 RDTREE alone +1.0%, both +1.4%. RDLATE=1 with RDTREE read +1.7% there, 0.3% more
// than =2, but it hasn't run on a 9070 XT or been checked on a pool. The fold takes 213 VGPRs, not 201, and still runs
// 6 waves a SIMD (earn/tuning/benchmark.md).
#ifndef PEARL_AMD_RDTREE
#define PEARL_AMD_RDTREE PEARL_AMD_RDNA4_SET
#endif
#if PEARL_AMD_RDTREE != 0 && PEARL_AMD_RDTREE != 1
#error "PEARL_AMD_RDTREE is 0 or 1"
#endif
// Staging cuts (RDNA 4). CRANK and SCHEDB are in PEARL_AMD_RDNA4_SET and PEARL_AMD_RDNA3_SET; the rest are off by
// default.
// PEARL_AMD_CRANK=1: the PF2 slab loop takes rank as PEARL_FOLD_RANK (128) at compile time, so a chunk is a
// constant 4 slabs. The runtime `% perChunk` was a scalar division every slab, and every even slab of the
// unrolled pair carried a readout it never ran. A code object built with it says so (PEARL_AMD_CAP_RANK), and the
// host then refuses any other rank for its matrix fold.
// PEARL_AMD_UNROLL4=1 (needs CRANK): the slab loop four slabs at a time, so the readout runs unconditionally
// after the fourth.
// PEARL_AMD_STOFF=1: both tiles' LDS buffers in one array, and each staging store at one lane register plus an
// immediate offset.
// PEARL_AMD_CMASK=1: perChunk stays a kernel argument but is written as 1 << its log, so `%` and `/` by it are a
// mask and a shift, not a division. rank is a power of two: it divides k, which the tiled folds need a power of two.
// PEARL_AMD_SCHEDB=1: a sched_barrier(0) after each slab's WMMAs and readout, before its stores and barrier, so
// the compiler can't move a slab's WMMAs past its barrier when both slabs of a pair are one basic block (CRANK).
// PEARL_AMD_LPIN=1: a sched_barrier(0) after the slab loads at the top of each step, so they aren't sunk below
// the slab's WMMAs and readout: each load gets two slabs to arrive in, as PF2 meant. That binds the scheduler only,
// inside one block. On gfx1100 machine sinking still moved one step's loads into the block after its readout,
// since nothing after them in their block stores. =2 adds an empty asm with a memory clobber after the loads,
// which counts as a store there, so the loads stay. It adds no instruction. =2 with PEARL_AMD_PF2PIN.
#ifndef PEARL_AMD_CRANK
#define PEARL_AMD_CRANK (PEARL_AMD_RDNA4_SET || PEARL_AMD_RDNA3_SET_ON)
#endif
#ifndef PEARL_AMD_UNROLL4
#define PEARL_AMD_UNROLL4 0
#endif
#ifndef PEARL_AMD_STOFF
#define PEARL_AMD_STOFF 0
#endif
#ifndef PEARL_AMD_CMASK
#define PEARL_AMD_CMASK 0
#endif
#ifndef PEARL_AMD_SCHEDB
#define PEARL_AMD_SCHEDB (PEARL_AMD_RDNA4_SET || PEARL_AMD_RDNA3_SET_ON)
#endif
#ifndef PEARL_AMD_LPIN
#define PEARL_AMD_LPIN (2 * PEARL_AMD_PF2PIN_ON)
#endif
#if PEARL_AMD_LPIN < 0 || PEARL_AMD_LPIN > 2
#error "PEARL_AMD_LPIN is 0, 1 or 2"
#endif
#if (PEARL_AMD_SCHEDB || PEARL_AMD_LPIN) && !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__)
#define PEARL_AMD_SCHED_PIN() __builtin_amdgcn_sched_barrier(0)
#else
#define PEARL_AMD_SCHED_PIN() ((void)0)
#endif
#if PEARL_AMD_UNROLL4 && !PEARL_AMD_CRANK
#error "PEARL_AMD_UNROLL4 needs PEARL_AMD_CRANK"
#endif
// PEARL_AMD_LDS32=1 (RDNA 4, wmma12 with PEARL_AMD_KPAIR, 32-byte slab): LDS rows of 32 bytes with no padding, so a
// block takes 24,576 bytes of LDS, not 36,864, and 4 blocks fit a WGP (8 waves a SIMD) where 3 did. The padding
// spread rows over the banks; here the row's two 16-byte halves swap instead, on row bits PEARL_AMD_LDS32_SWZ (a
// mask; the swap is the parity of the masked bits). Bits 2 and 3 (12) keep both an 8-lane pass over 32 banks and a
// 16-lane pass over 64 banks free of conflicts, for the 16-byte fragment reads and the 16-byte stores alike. The
// bytes a fragment holds are the same, so the fold is bit-exact. In PEARL_AMD_RDNA4_SET with PEARL_AMD_LDS32_PIN
// (in the fold body), which it needs: the rows and the pin pay only together. With the rest of the set the fold
// takes 201 VGPRs, so it still runs 6 waves a SIMD, not 8.
// PEARL_AMD_WAVES=N: amdgpu_waves_per_eu(N) on the wmma12 fold, so the compiler keeps it within 1536 / N VGPRs
// (192 for 8 waves a SIMD). Off by default.
#ifndef PEARL_AMD_LDS32
#define PEARL_AMD_LDS32 PEARL_AMD_RDNA4_SET
#endif
#ifndef PEARL_AMD_LDS32_SWZ
#define PEARL_AMD_LDS32_SWZ 12
#endif
#ifndef PEARL_AMD_LDS32_PIN
#define PEARL_AMD_LDS32_PIN PEARL_AMD_RDNA4_SET
#endif
#ifndef PEARL_AMD_LDS_PAD
#define PEARL_AMD_LDS_PAD 0
#endif
#if PEARL_AMD_LDS_PAD && PEARL_AMD_STOFF
#error "PEARL_AMD_LDS_PAD pads the separate B' buffers: build it without PEARL_AMD_STOFF"
#endif
// PEARL_AMD_LDS32_S48=1, timing only: PEARL_AMD_LDS32's code (pinned with _PIN) at today's 48-byte rows; with
// -DPEARL_AMD_LDS32_SWZ=0 the layout is today's exactly, so it times the pinned offsets alone.
#ifndef PEARL_AMD_LDS32_S48
#define PEARL_AMD_LDS32_S48 0
#endif
#if PEARL_AMD_LDS32
__device__ __forceinline__ int pearl_amd_l32_swz(int row) {
  return __builtin_popcount((unsigned)row & (unsigned)PEARL_AMD_LDS32_SWZ) & 1;
}
#endif
__device__ __forceinline__ void pearl_amd_lds_barrier() {
#if !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__)
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#else
  __syncthreads();
#endif
}

// PEARL_AMD_DPP=1 (RDNA, wave32): VALU lane moves instead of __shfl_xor's ds_bpermute, which
// goes through the LDS pipe: lane ^ 1 by a quad permute, ^ 8 by rotating each 16-lane row by 8,
// ^ 16 by swapping the two rows (v_permlanex16 with the identity selects). On by default on
// RDNA 3 and RDNA 4: 31.30-31.41 TH/s against 31.10-31.16 on the same 7800 XT, 54.52 against
// 52.88 on the 9060 XT above. And on RDNA 2 (gfx10 has the same DPP row moves and v_permlanex16),
// since 12 November: on an RX 6800 (1400 MHz core cap) hashrate.js read 17.28-17.29 TH/s against
// 17.22-17.24 for the old readout, three rounds each, a copy of the release within 0.06%; 200 VGPRs
// against 201, still 4 waves.
// The passes whose wave32 has these lane moves: RDNA 3 and 4, and RDNA 2's dot4 fold.
#define PEARL_AMD_DPP_HW (PEARL_AMD_WMMA11 || PEARL_AMD_WMMA12 || (PEARL_AMD_DOT4 && !PEARL_AMD_EMU))
#ifndef PEARL_AMD_DPP
#define PEARL_AMD_DPP PEARL_AMD_DPP_HW
#endif
// XOR across the eight lanes that share a region: lane bits 0, 3 and 4 in every layout here.
// (Bits 1 and 2 pick the region's column offset; on CDNA bit 5 picks its row offset.)
__device__ __forceinline__ uint32_t pearl_amd_group_xor(uint32_t x) {
#if PEARL_AMD_DPP && PEARL_AMD_DPP_HW
  x ^= (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0xB1, 0xF, 0xF, false);
  x ^= (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x128, 0xF, 0xF, false);
#if PEARL_AMD_RDPERM == 1 && PEARL_AMD_WMMA12
  // PEARL_AMD_RDPERM=1 (above): every lane is active and picks a lane in range, so fetch-inactive and bound_ctrl
  // change nothing, and with either set the compiler drops the old value (AMDGPU's instcombine).
  x ^= __builtin_amdgcn_permlanex16(x, x, 0x76543210u, 0xFEDCBA98u, true, true);
#endif
#if PEARL_AMD_RDPERM == 2 && PEARL_AMD_WMMA12
  {   // PEARL_AMD_RDPERM=2: the old value a fresh register from an empty asm, which costs no instruction.
    uint32_t old;
    asm("" : "=v"(old) : "v"(x));
    x ^= __builtin_amdgcn_permlanex16(old, x, 0x76543210u, 0xFEDCBA98u, false, false);
  }
#endif
#if !(PEARL_AMD_RDPERM && PEARL_AMD_WMMA12)
  x ^= __builtin_amdgcn_permlanex16(x, x, 0x76543210u, 0xFEDCBA98u, false, false);
#endif
#else
  x ^= (uint32_t)__shfl_xor((int)x, 1);
  x ^= (uint32_t)__shfl_xor((int)x, 8);
  x ^= (uint32_t)__shfl_xor((int)x, 16);
#endif
  return x;
}

// PEARL_AMD_READOUT_DROP=1: a build for timing only, whose transcripts are wrong. In the RDNA 4
// fold (wmma12) only, each chunk's readout skips its cross-lane steps: no pearl_amd_group_xor
// (three lane moves and their XORs), no compare and no selects. Each lane XORs its own part of
// each word into keepLo instead, and keepHi stays zero. The matrix work, the loads and the
// barriers are unchanged, so a hashrate.js race against the default build times the VALU those
// steps cost. About 68 VALU a chunk a wave are left, close to the 68-70 of two-lane 32x8
// regions (round 4 item 2), which would remove the same steps for real. This build is that
// item's gate: two-lane goes ahead only if dropping the steps is worth at least +1.0% in 3 of 3
// scored rounds on both an RX 9070 XT and an RX 9060 XT.
// It must never mine or ship. Every code object built with it reports PEARL_AMD_CAP_DROP, and
// the host then refuses any target but zero, the one hashrate.js sets. build-amd.sh refuses it in
// the builds that ship, and test/sweep.sh times it only when SWEEP_TIMEONLY names it.
#ifndef PEARL_AMD_READOUT_DROP
#define PEARL_AMD_READOUT_DROP 0
#endif

// PEARL_AMD_ABLATE: a build for timing only, whose transcripts are wrong. Each bit leaves one part of
// the matrix fold out, to see what that part costs:
//   2   no s_barrier at the end of a slab. The data is the same; nothing orders the LDS stores and
//       loads. The fences around the barrier stay, so the memory waits they add (vmcnt(0) and an L0
//       invalidate on RDNA 3) are still timed: only the wait for the other waves goes.
//   8   no LDS fragment loads. On a GPU an empty asm gives each k-step new fragments, so they live
//       no longer than loaded ones and the fold needs no more VGPRs (gfx11's is at 240, the most
//       for 6 waves). Their values are whatever the registers hold and may stop changing, and a
//       card at its power limit may then clock up, so the time it saves overstates the wait on
//       LDS. One real fragment read a slab is kept, 1 of the release's KS / kTK * (kMT + kNT): with
//       none, the RDNA 4 compiler dropped the LDS stores and the global loads that feed them (0
//       bytes of LDS and 8 waves, not 36864 and 6), so the build timed no operand staging at a
//       higher occupancy. The read takes turns over all four LDS buffers, picked at run time, so
//       every store has a reader. build-amd.sh check's row shows LDS 36864 and 6 waves, as the
//       release; the run scripts also count the fold's LDS stores and global loads against the
//       release's. The HIP-CPU build uses the first slab's fragments at every k-step instead.
//   16  every block reads tile 0's operands, which stay in L2. The loads still run.
//   32  no staging in the slab loop: no global loads of the slabs ahead and no LDS stores of them. Slab 0 is
//       still loaded and stored before the loop, and the fragments are read from LDS as before (stale words,
//       which the barrier's fences keep the compiler from hoisting). With 2 and 8 the loop is the WMMAs, the
//       readout and the loop itself, which pearl_amd_peak's register-fed kernel times without the readout.
//   64  the wrap readout (PEARL_AMD_KWRAP) on RDNA 4 without its cross-lane steps, selects or branch: each lane
//       XORs its own part of each word into keepLo. The bound for any cut to that readout at k 4096, where
//       PEARL_AMD_READOUT_DROP times only the k 2048 one.
// Bits 2, 8 and 32 are in the PF2 loop only (RDNA 3 and RDNA 4); bit 16 is in either loop. They combine.
// These are bits 2, 8 and 16 of the old knob, which 534e285 removed: it only ever worked in the
// one-slab loop.
// Every word a matrix fold stores is XORed with a constant, so the fold check always fails, even where
// an ablation leaves the sums right (with no barrier the words can still come out right). test/sweep.sh
// times a build named in SWEEP_TIMEONLY only when its check fails. The reference fold is left alone.
// It must never mine or ship. Its code objects report PEARL_AMD_CAP_TIMING, and the host then refuses
// any target but zero, the one hashrate.js sets. build-amd.sh refuses it in the builds that ship. Its k
// is the release's (pearl_host_card_k), so on RDNA 4 it times the k 4096 fold the card mines.
#ifndef PEARL_AMD_ABLATE
#define PEARL_AMD_ABLATE 0
#endif
#if PEARL_AMD_ABLATE & ~(2 | 8 | 16 | 32 | 64)
#error "PEARL_AMD_ABLATE takes bits 2, 8, 16, 32 and 64 only"
#endif
// Whether this pass has the dot4 fold's body: a pass with PEARL_AMD_DOT4 that builds the one-slab
// loop (no PF2), so RDNA 2's, and RDNA 3 or RDNA 4 built with -DPEARL_AMD_PF2=0, which runs RDNA 2's
// fold on those cards; and HIP-CPU. The release's RDNA 3 and RDNA 4 code objects have no dot4 fold
// and stay as they were: it is slower there than their WMMA folds (18.1 TMAC/s against wmma11's 32.8
// on an RX 7800 XT), and built with their PF2 loop (and DPP, and on gfx12 PFX, TID and the wrap
// readout) it needed more than the 256 VGPRs a wave can have, 21 to 101 spilled on gfx11 and 217 on
// gfx12 (ROCm 7.2, 2 November, before PEARL_AMD_DOT4_SPLIT; not tried since). An ablation build leaves it
// out too: it times the RDNA 3 and RDNA 4 matrix folds, and bits 2 and 8 have no dot4 form.
#if PEARL_AMD_ABLATE
#define PEARL_AMD_DOT4_FOLD 0
#elif PEARL_AMD_EMU || (PEARL_AMD_DOT4 && !PEARL_AMD_PF2)
#define PEARL_AMD_DOT4_FOLD 1
#else
#define PEARL_AMD_DOT4_FOLD 0
#endif

// PEARL_AMD_STAMP=1: a build for timing only, with clock stamps. In the PF2 loop of the RDNA folds (gfx11
// and gfx12) every wave keeps these in scalar registers, and lane 0 of each wave stores them at the end of
// its block, PEARL_AMD_STAMP_WORDS words, if the host has set pearl_amd_stamps:
//   - SHADER_CYCLES and REALTIME read side by side at the block's start and end. SHADER_CYCLES counts the
//     shader clock (20 bits on gfx11, the low 32 of 64 on gfx12); REALTIME is a constant clock, 100 MHz
//     on RDNA (hipDeviceAttributeWallClockRate). Cycles over ticks is the wave's clock in MHz.
//   - The cycles of the prologue (start to the slab loop), the slab loop and the epilogue (the loop's end
//     to the end), each a sum of short deltas in 32 bits. gfx11's counter wraps every 2^20 cycles, about
//     0.4 ms, which a whole block at k 4096 can pass.
//   - The cycles each slab waits at its s_barrier: the counter read just before and just after it, inside
//     the barrier's fence pair. Without PFX the release fence waits for every load and store; on gfx11 an
//     s_waitcnt makes that wait explicit before the first read, so it isn't counted as barrier wait.
//   - The cycles each slab waits at the release fence: one more read just before it. That is the memory
//     wait: without PFX, the global loads still in flight (gfx11's PF2 loads); with PFX, the LDS only.
//   - The counter at the first barrier exit after its bit 16 changes, and after its bit 19 changes. The
//     ISA says SHADER_CYCLES isn't synchronised across SIMDs, so these compare only between waves on one
//     SIMD: two blocks sharing a WGP that cross the same boundary differ there by their true phase, how
//     far apart their barriers are. Bit 16 comes round about every 10 slabs, so its crossing is near a
//     block's start; bit 19 about every 70, so it lands anywhere in a block.
//   - HW_ID1 at the start and the end: the wave's SIMD, WGP, shader array and engine. A context switch
//     can change it; test/stamp-summary.js drops such waves.
// REALTIME is a message whose answer the wave waits for (lgkmcnt on gfx11, which counts LDS loads too;
// kmcnt on gfx12), so only the prologue and epilogue read it. Nothing is stored in the slab loop.
// Every stamp is the same in all lanes of a wave, so all of them belong in SGPRs, and the stamps must not
// take a VGPR: the run scripts keep a stamp build only with its release twin's VGPRs. The first build took
// 218 against 198 on gfx1201 and 245 against 240 on gfx1100, a wave less. So every value the fold keeps goes
// through pearl_amd_sgpr (readfirstlane), the crossings are one compare and one select each, the 16 words
// are built only after the loop, and the pointer is in constant memory, which a wave reads with a scalar
// load. That build also had one more vector load than the release (13 global loads, not 12), most likely
// the pointer: a __device__ variable read after the counter reads, which the compiler can't prove unchanged.
// The transcripts are right: a stamp build passes the fold check. It must never mine or ship, as
// PEARL_AMD_ABLATE: its code objects report PEARL_AMD_CAP_TIMING and PEARL_AMD_CAP_STAMP, the host refuses
// any target but zero, and build-amd.sh refuses it in the builds that ship. pearl_amd_foldtest's stamps
// argument writes the last launch's stamps to a file, and test/stamp-summary.js summarises them.
#ifndef PEARL_AMD_STAMP
#define PEARL_AMD_STAMP 0
#endif
#if PEARL_AMD_STAMP != 0 && PEARL_AMD_STAMP != 1
#error "PEARL_AMD_STAMP is 0 or 1"
#endif
#if PEARL_AMD_STAMP && PEARL_AMD_EMU
#error "PEARL_AMD_STAMP reads a GPU's clocks: build it for a card (build-amd.sh gpu), not HIP-CPU"
#endif
#if PEARL_AMD_STAMP
// The words a wave stores, and its counter's width (0 where the fold has no stamps). The record's layout is
// the one test/stamp-summary.js reads (version 2).
#define PEARL_AMD_STAMP_WORDS 16u
#define PEARL_AMD_STAMP_BITS (PEARL_AMD_WMMA11 ? 20u : PEARL_AMD_WMMA12 ? 32u : 0u)
}  // namespace
// Where the stamps go: wave w of block b at word (b * waves a block + w) * PEARL_AMD_STAMP_WORDS, with
// b = blockIdx.y * gridDim.x + blockIdx.x. The host sets it (hipMemcpyToSymbol); null stores nothing. In
// constant memory, so the fold reads it with a scalar load into an SGPR pair.
__constant__ uint32_t *pearl_amd_stamps = nullptr;
namespace {
// A stamp value, in an SGPR. It is the same in every lane, and readfirstlane says so: the compiler then keeps
// it in an SGPR, and a value it had put in a VGPR costs one v_readfirstlane here, not a VGPR across the loop.
// On a value in an SGPR already it is a copy.
__device__ __forceinline__ uint32_t pearl_amd_sgpr(uint32_t x) {
#if PEARL_AMD_WMMA11 || PEARL_AMD_WMMA12
  return (uint32_t)__builtin_amdgcn_readfirstlane((int)x);
#else
  return x;
#endif
}
// SHADER_CYCLES: gfx11's 20 bits (hwreg 29, SHADER_CYCLES); gfx12's low 32 (hwreg 29, SHADER_CYCLES_LO).
__device__ __forceinline__ uint32_t pearl_amd_cycles() {
#if PEARL_AMD_WMMA11
  return (uint32_t)__builtin_amdgcn_s_getreg(0x981D) & 0xFFFFFu;
#elif PEARL_AMD_WMMA12
  return (uint32_t)__builtin_amdgcn_s_getreg(0xF81D);
#else
  return 0u;
#endif
}
// b - a in cycles, for two reads of one wave's counter less than a wrap apart.
__device__ __forceinline__ uint32_t pearl_amd_cycles_since(uint32_t a, uint32_t b) {
  return PEARL_AMD_WMMA11 ? (b - a) & 0xFFFFFu : b - a;
}
// REALTIME's low 32 bits: s_sendmsg_rtn_b64 with MSG_RTN_GET_REALTIME (0x83).
__device__ __forceinline__ uint32_t pearl_amd_realtime() {
#if PEARL_AMD_WMMA11 || PEARL_AMD_WMMA12
  return (uint32_t)__builtin_amdgcn_s_sendmsg_rtnl(0x83);
#else
  return 0u;
#endif
}
// HW_ID1 (hwreg 23 on gfx11 and gfx12): WAVE_ID 4:0, SIMD_ID 9:8, WGP_ID 13:10, SA_ID 16, SE_ID 20:18.
__device__ __forceinline__ uint32_t pearl_amd_hw_id1() {
#if PEARL_AMD_WMMA11 || PEARL_AMD_WMMA12
  return (uint32_t)__builtin_amdgcn_s_getreg(0xF817);
#else
  return 0u;
#endif
}
#endif

// PEARL_AMD_KWRAP: the readout for k past 2048 (round 4 item 5). The folds take k at run time, but past
// 16 chunks today's readout is wrong: it writes each word once, onto zero, where chunk c must go into
// word c % 16 after that word's rotl 13 (the reference's rule, reference.js). With this on, every fold
// does that. At k 4096 a hash covers twice the MACs, so hashes and readouts per MAC halve, and credit
// per MAC is the same at rank 128.
//
// RDNA 4 mines k 4096 or 8192 (pearl_host_card_k), so its Linux code objects always have this readout: +1.0 to
// +1.2% on an RX 9070 XT and an RX 9060 XT (earn/tuning/benchmark.md). Every other target keeps today's readout and
// k 2048: the wrap readout at k 2048 costs about 0.4%. The Windows core (build-amd.sh winlib) builds
// with -DPEARL_AMD_KWRAP=0, so RDNA 4 mines k 2048 there: its compiler put part of the gfx12 fold in
// scratch memory with this readout, and no Windows Radeon has run it. Only a gfx12 device pass defines __GFX12__, so one
// build holds both, and the host learns which a card has from its caps (PEARL_AMD_CAP_KWRAP), never from
// this macro. -DPEARL_FOLD_K=4096 turns it on for every target, as before; -DPEARL_AMD_KWRAP=0 or 1 sets
// it for every target (=0 builds today's release, to race against). A readout-drop build leaves it off.
#ifndef PEARL_AMD_KWRAP
#if PEARL_FOLD_CHUNKS > PEARL_JACKPOT_BUCKETS
#define PEARL_AMD_KWRAP 1
#elif !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__) && defined(__GFX12__) && !PEARL_AMD_READOUT_DROP
#define PEARL_AMD_KWRAP 1
#else
#define PEARL_AMD_KWRAP 0
#endif
#endif
#if PEARL_AMD_KWRAP != 0 && PEARL_AMD_KWRAP != 1
#error "PEARL_AMD_KWRAP is 0 or 1"
#endif
#if !PEARL_AMD_KWRAP && PEARL_FOLD_CHUNKS > PEARL_JACKPOT_BUCKETS
#error "PEARL_FOLD_K past 16 chunks needs PEARL_AMD_KWRAP: the default readout writes each word once"
#endif
#if PEARL_AMD_KWRAP && PEARL_AMD_READOUT_DROP
#error "PEARL_AMD_KWRAP and PEARL_AMD_READOUT_DROP don't go together: build one or the other"
#endif

// PEARL_AMD_RDLATE (with RDTREE, above PEARL_AMD_CRANK): 2 in RDNA 4 code with the set, where its slab loop is (the
// wrap readout, KPAIR, CRANK and PF2, no UNROLL4), as it was timed and verified. A build that takes a part of that loop
// out, or ablates the fragment loads (bit 8) or the readout (bit 64), gets 0, the readout as it was, not an error.
// PEARL_AMD_DIRECT=1 (RDNA 4, wmma12; off by default, goal-rdna4a, 14 November): no LDS. Each lane loads its own
// fragments from the per-tile operands with buffer loads, one slab ahead in registers, so the slab loop has no LDS
// stores, no LDS reads and no barrier. The four waves that share A' rows (and the two that share B' rows) each load
// them, from the vector cache. A lane's 16 bytes of a row are what its KPAIR read took from LDS, so the fold is the
// same, bit for bit. The readout runs at the chunk's end (RDLATE 0). =2 loads two slabs ahead, in three register sets.
#ifndef PEARL_AMD_DIRECT
#define PEARL_AMD_DIRECT 0
#endif
#ifndef PEARL_AMD_RDLATE
#if (PEARL_AMD_ABLATE & (8 | 64)) || PEARL_AMD_DIRECT
#define PEARL_AMD_RDLATE 0
#else
#if PEARL_AMD_RDNA4_SET && PEARL_AMD_KWRAP && PEARL_AMD_KPAIR && PEARL_AMD_CRANK && PEARL_AMD_PF2 && !PEARL_AMD_UNROLL4
#define PEARL_AMD_RDLATE 2
#else
#define PEARL_AMD_RDLATE 0
#endif
#endif
#endif
#if PEARL_AMD_RDLATE != 0 && PEARL_AMD_RDLATE != 1 && PEARL_AMD_RDLATE != 2
#error "PEARL_AMD_RDLATE is 0, 1 or 2"
#endif
// PEARL_AMD_RDLATE's pin: no load moves across it (the memory clobber; with RDLATE=2 the compiler had moved a slab's
// fragment loads below the XORs) and nothing is scheduled across it. It emits no instruction and waits for nothing.
#if PEARL_AMD_RDLATE && !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__)
#define PEARL_AMD_RDLATE_PIN() \
  do {                         \
    asm volatile("" ::: "memory"); \
    __builtin_amdgcn_sched_barrier(0); \
  } while (0)
#else
#define PEARL_AMD_RDLATE_PIN() ((void)0)
#endif
// PEARL_AMD_WALK (above): the tile (x, y) that block (bx, by) of a gx x gy grid folds. In a whole supertile row r,
// off counts the row's blocks in launch order, n is the supertile, j the block in it, q its sub-tile and p its place
// there. One-to-one on the grid. Every value is the same in all lanes, so the compiler keeps it scalar.
__device__ __forceinline__ void pearl_amd_walk(uint32_t bx, uint32_t by, uint32_t gx, uint32_t gy, uint32_t &x,
                                               uint32_t &y) {
  x = bx;
  y = by;
#if PEARL_AMD_WALK_ON
  constexpr uint32_t W = PEARL_AMD_WALK_W, H = PEARL_AMD_WALK_H;
  constexpr uint32_t SW = PEARL_AMD_WALK == 2 ? PEARL_AMD_WALK_SW : W, SH = PEARL_AMD_WALK == 2 ? PEARL_AMD_WALK_SH : H;
  static_assert(H > 0u && SW > 0u && SH > 0u && W % SW == 0u && H % SH == 0u,
                "PEARL_AMD_WALK: sub-tiles must tile a supertile");
  constexpr uint32_t Q = (W / SW) * (H / SH);
  if (gx % W != 0u || by >= gy / H * H) return;
  const uint32_t r = by / H;
  const uint32_t off = (by - r * H) * gx + bx;
  const uint32_t n = off / (W * H);
  const uint32_t j = off - n * (W * H);
  const uint32_t b = (r & 1u) != 0u ? gx / W - 1u - n : n;
  const uint32_t q = j % Q, p = j / Q;
  x = b * W + q % (W / SW) * SW + p % SW;
  y = r * H + q / (W / SW) * SH + p / SW;
#else
  (void)gx;
  (void)gy;
#endif
}

// PEARL_AMD_FUSE: the matrix folds and dot4 hash their own transcripts (issue #287, item 6). Without it every
// fold stores all 16 words of every region, 128 MiB for a batch of 256 at the mainnet m, and pearl_amd_hash
// reads them back in a kernel of its own: 0.41-0.45 ms of a 30-31 ms batch on an RX 9060 XT at k 4096 (1.4%),
// and 0.15-0.17 ms of a 20.6 ms batch on an RX 7900 XTX (0.8%). With it, a block ends its tile by putting the
// tile's 128 regions in LDS (the B' staging buffers are free by then), and 128 of its threads hash one region
// each and test it as pearl_amd_hash does, into the same hit list with the same region numbers. No transcript
// is stored and no hash kernel runs (the reference fold still stores, and the host still hashes after it). That
// saves the hash kernel's reads and launch and the fold's stores. It costs the hash's VALU inside the fold,
// where on RDNA 4 a VALU instruction doesn't overlap the WMMAs: on the 9060 XT the fold got 0.33 ms slower and
// the batch 0.11 ms faster. A fold given a transcript buffer (the fold test does) stores the words as well.
// Off by default on every target. Against the release alone it ran hashrate.js +0.34% on an RX 9060 XT at k 4096
// (+0.30 to +0.38% a round, two copies of each build within 0.08% of each other) and +0.36 to +0.44% on another,
// with verify-hits 400 of 400 and 7 Kryptex shares, none rejected. On an RX 9070 XT (gfx1201) at k 8192 it lost
// 0.50%: its hash kernel takes only 0.23 ms of a 33.6 ms batch, and the fold got 0.36 ms slower (6 November).
// Raced on top of the late readout and the tile order (items 4 and 5) on 7 November, it lost 0.47% on a
// 9060 XT and read level on a second, noisier one, so it stays off (earn/tuning/benchmark.md). -DPEARL_AMD_FUSE=1
// turns it on for every target, to race again. The host learns whether a card has it from its caps
// (PEARL_AMD_CAP_FUSE), never from this macro.
#ifndef PEARL_AMD_FUSE
#define PEARL_AMD_FUSE 0
#endif
#if PEARL_AMD_FUSE != 0 && PEARL_AMD_FUSE != 1
#error "PEARL_AMD_FUSE is 0 or 1"
#endif
// Every fold takes the test and the hit list, so one host launch fits every target's code object. A fold that
// doesn't hash ignores them.
#define PEARL_AMD_FUSE_PARAMS , [[maybe_unused]] const PearlTranscriptTest test, [[maybe_unused]] const PearlHitList hits
#define PEARL_AMD_FUSE_ARGS , test, hits

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

// One region's transcript hashed and tested, as pearl_tall_hash80 in pearl_kernel.cu does: the top word
// first, and only for the rare region that passes, the whole hash against the target. A hit goes in the list
// with region number i. pearl_amd_hash runs it for every region of a batch, a PEARL_AMD_FUSE fold for every
// region of its tile.
__device__ __forceinline__ void pearl_amd_hash_region(const uint32_t tm[16], uint32_t i,
                                                      const PearlTranscriptTest &test, const PearlHitList &hits) {
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

// The matrix fold over one 128x256 tile of C: rows blockIdx.y * 128 of A', and B' rows
// col0 + blockIdx.x * 256 (the batch's columns start at col0), or the tile PEARL_AMD_WALK gives.
// Every region of the tile gets its 16 transcript words in `tr`, at its batch-local number:
// column offset index within the batch times rows_valid, plus row offset index.
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
                                                    uint32_t *__restrict__ tr PEARL_AMD_FUSE_PARAMS) {
  constexpr int W = P::kWave, T = W * P::kWR * P::kWC;
  constexpr int WM = P::kMT * P::kTM, WN = P::kNT * P::kTN;
  constexpr int BM = P::kWR * WM, BN = P::kWC * WN;
#if PEARL_AMD_LDS32
  // PEARL_AMD_LDS32 (above): 32-byte rows, halves swapped by row bits, in the RDNA 4 fold with paired reads only.
  constexpr int KS = std::is_same<P, PearlAmdDot4>::value ? (int)PEARL_AMD_DOT4_SLAB : (int)PEARL_AMD_SLAB;
  constexpr bool kL32 = PEARL_AMD_KPAIR && std::is_same<P, PearlAmdWmma12>::value;
  static_assert(!kL32 || KS == 32, "PEARL_AMD_LDS32 is for 32-byte slabs");
  constexpr int S = kL32 && !PEARL_AMD_LDS32_S48 ? KS : KS + 16;
#else
  // The dot4 fold has a slab of its own (PEARL_AMD_DOT4_SLAB, above).
  constexpr int KS = std::is_same<P, PearlAmdDot4>::value ? (int)PEARL_AMD_DOT4_SLAB : (int)PEARL_AMD_SLAB, S = KS + 16;
#endif
  constexpr int QR = KS / 16;                          // 16-byte units in a slab row
  constexpr int UA = BM * QR, UB = BN * QR, U = (UA + UB) / T;
  constexpr int RB = WM / 32, CB = WN / 64;            // 32x64 blocks a wave
  constexpr int NRO = P::kRoByLane ? 1 : 2;            // regions a lane holds in a block
  // PEARL_AMD_READOUT_DROP (above): the RDNA 4 fold only, for timing.
  constexpr bool kDrop = PEARL_AMD_READOUT_DROP && std::is_same<P, PearlAmdWmma12>::value;
  static_assert(T == (int)PEARL_AMD_THREADS && BM == (int)PEARL_AMD_BM && BN == (int)PEARL_AMD_BN,
                "the host launches every matrix fold as 256 threads over 128x256 tiles");
  static_assert((UA + UB) % T == 0 && UA % T == 0, "the staging must split evenly");
  static_assert(WM % 32 == 0 && WN % 64 == 0, "a wave must hold whole 32x64 blocks");
  static_assert(KS % P::kTK == 0, "a slab must hold whole k-steps");

#if PEARL_AMD_DOT4_ADDR_ON
  // PEARL_AMD_DOT4_ADDR (above): [A' 0][B' 0][A' 1][B' 1]. sA[0] and sB[0] are buffer 0's, which the first slab's
  // store uses; the slab loop finds buffer 1 at (BM + BN) * S past them.
  static_assert(!PEARL_AMD_STOFF && !PEARL_AMD_LDS32, "PEARL_AMD_DOT4_ADDR has a layout of its own");
  __shared__ __attribute__((aligned(16))) int8_t sL[2 * (BM + BN) * S];
  int8_t (*const sA)[BM * S] = reinterpret_cast<int8_t (*)[BM * S]>(sL);
  int8_t (*const sB)[BN * S] = reinterpret_cast<int8_t (*)[BN * S]>(sL + BM * S);
#elif PEARL_AMD_STOFF
  // PEARL_AMD_STOFF (above): A' buffers 0 and 1, then B' buffers 0 and 1, in one array.
  __shared__ __attribute__((aligned(16))) int8_t sL[2 * (BM + BN) * S];
  int8_t (*const sA)[BM * S] = reinterpret_cast<int8_t (*)[BM * S]>(sL);
  int8_t (*const sB)[BN * S] = reinterpret_cast<int8_t (*)[BN * S]>(sL + 2 * BM * S);
#else
  __shared__ __attribute__((aligned(16))) int8_t sA[2][BM * S];
#if PEARL_AMD_LDS_PAD
  // PEARL_AMD_LDS_PAD: unused bytes after each B' buffer, for timing only: with PEARL_AMD_LDS32, 6144 brings a block
  // back to 36,864 bytes, so 3 blocks fit a WGP (6 waves a SIMD) with the 32-byte layout's code.
  __shared__ __attribute__((aligned(16))) int8_t sB[2][BN * S + PEARL_AMD_LDS_PAD];
#else
  __shared__ __attribute__((aligned(16))) int8_t sB[2][BN * S];
#endif
#endif

#if PEARL_AMD_STAMP
  // PEARL_AMD_STAMP (above): the start, and what the slab loop adds up, in SGPRs (pearl_amd_sgpr). stLast is the
  // counter at its last read: the loop's start, each barrier exit, then the loop's end. stSeen says which
  // crossings stX16 and stX19 hold (bits 0 and 1).
  const uint32_t stHw = pearl_amd_sgpr(pearl_amd_hw_id1());
  const uint32_t stCyc0 = pearl_amd_sgpr(pearl_amd_cycles());
  const uint32_t stRt0 = pearl_amd_sgpr(pearl_amd_realtime());
  uint32_t stPro = 0u, stLast = stCyc0, stLoop = 0u, stBar = 0u, stMem = 0u, stX16 = 0u, stX19 = 0u, stSeen = 0u;
#endif
  const int t = (int)threadIdx.x, wave = t / W, lane = t % W;
#if PEARL_AMD_STAMP
  const uint32_t stWave = pearl_amd_sgpr((uint32_t)wave);   // the record needs it after the loop: an SGPR, not a VGPR
#endif
#if PEARL_AMD_TID
  // The compiler doesn't know threadIdx.x < 256 here, so it guarded the first staging unit
  // (always A') with an exec-mask branch; told, the branch goes and the fold needs ~17 fewer VGPRs.
  __builtin_assume(t >= 0 && t < T);
#endif
  const int wr = wave / P::kWC, wc = wave % P::kWC;
#if PEARL_AMD_WALK_ON
  uint32_t tileX, tileY;   // PEARL_AMD_WALK (above): this block's tile
  pearl_amd_walk(blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, tileX, tileY);
  const uint32_t tileRow = tileY * (uint32_t)BM;
  const uint32_t tileCol = tileX * (uint32_t)BN;
#else
  const uint32_t tileRow = blockIdx.y * (uint32_t)BM;   // first A' row of the tile
  const uint32_t tileCol = blockIdx.x * (uint32_t)BN;   // first column, within the batch
#endif
#if PEARL_AMD_ABLATE & 16
  // PEARL_AMD_ABLATE bit 16: every block reads tile 0's operands, which stay in L2.
  const int8_t *gA = Ap;
  const int8_t *gB = Bp + (size_t)col0 * k;
#else
  const int8_t *gA = Ap + (size_t)tileRow * k;
  const int8_t *gB = Bp + (size_t)(col0 + tileCol) * k;
#endif

  // The next slab, in registers while the current one is used. A vector type, not HIP's
  // int4: int4 is a struct, which the compiler copied through scratch memory.
  pearl_amd_v4i pre[U];
#if PEARL_AMD_BUFLD && PEARL_AMD_TILED && (PEARL_AMD_WMMA11 || PEARL_AMD_WMMA12 || PEARL_AMD_DOT4_ADDR_ON)
  // PEARL_AMD_BUFLD (above): the tiles' resources. PEARL_AMD_WMMA11 and _WMMA12 are set only in a gfx11 or gfx12
  // device pass, never in HIP-CPU's, and word 3 0x31004000 is right on those two only; gfx10's is 0x31014000
  // (PEARL_AMD_DOT4_ADDR_ON: a gfx103x device pass).
  constexpr int kRsrcW3 = PEARL_AMD_DOT4_ADDR_ON ? 0x31014000 : 0x31004000;
  const __amdgpu_buffer_rsrc_t rsA =
      __builtin_amdgcn_make_buffer_rsrc(const_cast<int8_t *>(gA), (short)0, (int)((uint32_t)BM * k), kRsrcW3);
  const __amdgpu_buffer_rsrc_t rsB =
      __builtin_amdgcn_make_buffer_rsrc(const_cast<int8_t *>(gB), (short)0, (int)((uint32_t)BN * k), kRsrcW3);
#define PEARL_AMD_LOADR(slab, R)                                                             \
  _Pragma("unroll") for (int u = 0; u < U; u++) {                                            \
    const int idx = t + u * T;                                                               \
    R[u] = __builtin_bit_cast(pearl_amd_v4i,                                                 \
        idx < UA ? __builtin_amdgcn_raw_buffer_load_b128(rsA, 16 * idx, (int)((slab) * (uint32_t)(BM * KS)), 0)\
                 : __builtin_amdgcn_raw_buffer_load_b128(rsB, 16 * (idx - UA), (int)((slab) * (uint32_t)(BN * KS)), 0));\
  }
#else
#define PEARL_AMD_LOADR(slab, R)                                                             \
  _Pragma("unroll") for (int u = 0; u < U; u++) {                                            \
    const int idx = t + u * T;                                                               \
    const pearl_amd_v4i *p = reinterpret_cast<const pearl_amd_v4i *>(                       \
        PEARL_AMD_TILED                                                                      \
            ? (idx < UA ? gA + (size_t)(slab) * (BM * KS) + 16 * idx                          \
                        : gB + (size_t)(slab) * (BN * KS) + 16 * (idx - UA))                  \
            : (idx < UA ? gA + (size_t)(idx / QR) * k                                         \
                        : gB + (size_t)((idx - UA) / QR) * k) + (size_t)(slab) * KS + 16 * (idx % QR)); \
    R[u] = *p;                                                                               \
  }
#endif
#if PEARL_AMD_LDS32
  // A thread's rows are t / QR plus multiples of 128, whose bits 2 and 3 are t / QR's: one swap a thread.
  static_assert(!kL32 || ((T / QR) % 16 == 0 && (UA / QR) % 16 == 0 && (PEARL_AMD_LDS32_SWZ & ~15) == 0),
                "PEARL_AMD_LDS32's swap must be the same for every row a thread stores");
  const int l32sw = kL32 ? pearl_amd_l32_swz(t / QR) : 0;
#if PEARL_AMD_LDS32_PIN
  // PEARL_AMD_LDS32_PIN: the thread's store offset (row t / QR, its half swapped) through an empty asm, so the
  // compiler keeps it in a VGPR and can't recompute it in the slab loop. The other rows are whole 128 further.
  static_assert(UA % T == 0 && T % QR == 0, "PEARL_AMD_LDS32_PIN needs whole rows a unit");
  int l32st = (t / QR) * S + 16 * ((t % QR) ^ l32sw);
#if !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__)
  if constexpr (kL32) asm volatile("" : "+v"(l32st));
#endif
  // And the lane's fragment-read offset (slab_step's off at k-step 0), the same way.
  int l32rd = (lane & 15) * S + 16 * ((lane >> 4) ^ (kL32 ? pearl_amd_l32_swz(lane & 15) : 0));
#if !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__)
  if constexpr (kL32) asm volatile("" : "+v"(l32rd));
#endif
#define PEARL_AMD_STORER(buf, R)                                                             \
  _Pragma("unroll") for (int u = 0; u < U; u++) {                                            \
    int8_t *dst = (u * T < UA ? sA[buf] + ((u * T) / QR) * S : sB[buf] + ((u * T - UA) / QR) * S) + l32st; \
    *reinterpret_cast<pearl_amd_v4i *>(dst) = R[u];                                          \
  }
#else
#define PEARL_AMD_STORER(buf, R)                                                             \
  _Pragma("unroll") for (int u = 0; u < U; u++) {                                            \
    const int idx = t + u * T;                                                               \
    int8_t *dst = idx < UA ? sA[buf] + (idx / QR) * S : sB[buf] + ((idx - UA) / QR) * S;     \
    *reinterpret_cast<pearl_amd_v4i *>(dst + 16 * ((idx % QR) ^ l32sw)) = R[u];              \
  }
#endif
#elif PEARL_AMD_STOFF
  // PEARL_AMD_STOFF (above): unit u of thread t is row t / QR + u * T / QR of its tile's slab (less BM rows for
  // B'), at 16 * (t % QR) in the row, so its place in sL is the thread's ldsOff plus a constant.
  static_assert(T % QR == 0 && UA % QR == 0, "a thread's staging units must share its place in a row");
  const uint32_t ldsOff = (uint32_t)((t / QR) * S + 16 * (t % QR));
#define PEARL_AMD_STORER(buf, R)                                                             \
  _Pragma("unroll") for (int u = 0; u < U; u++) {                                            \
    const uint32_t c = u * T < UA ? (uint32_t)((buf) * (BM * S) + u * (T / QR) * S)           \
                                  : (uint32_t)(2 * BM * S + (buf) * (BN * S) + (u * T - UA) / QR * S); \
    *reinterpret_cast<pearl_amd_v4i *>(sL + (ldsOff + c)) = R[u];                             \
  }
#else
#define PEARL_AMD_STORER(buf, R)                                                             \
  _Pragma("unroll") for (int u = 0; u < U; u++) {                                            \
    const int idx = t + u * T;                                                               \
    int8_t *dst = idx < UA ? sA[buf] + (idx / QR) * S : sB[buf] + ((idx - UA) / QR) * S;     \
    *reinterpret_cast<pearl_amd_v4i *>(dst + 16 * (idx % QR)) = R[u];                        \
  }
#endif
#if PEARL_AMD_DOT4_ADDR_ON
  // PEARL_AMD_DOT4_ADDR (above), RDNA 2's slab loop: a gfx103x device pass, whose only fold body is dot4's. sL is
  // [A' 0][B' 0][A' 1][B' 1], so buffer 1 is kFlip bytes past buffer 0, A' and B' alike.
  static_assert(std::is_same<P, PearlAmdDot4>::value, "PEARL_AMD_DOT4_ADDR is the dot4 fold's");
  constexpr uint32_t kFlip = (uint32_t)((BM + BN) * S);
  static_assert(2 * (BM + BN) * S <= 65536, "every LDS offset fits a DS instruction's 16-bit offset");
  static_assert(UA % T == 0 && QR == 4 && T % 32 == 0 && (T / QR) % 8 == 0,
                "a staging unit is A' or B' by its number alone, and the staging map is for 64-byte slabs");
  // PEARL_AMD_DOT4_SWZ (above): A' row r of a tile sits at LDS row swz(r) (bits 1-2 XOR bits 3-4), and each eight
  // lanes of a staging store write rows 4 apart. swz is its own inverse and keeps every row in its 32-row block.
  constexpr bool kSwz = PEARL_AMD_DOT4_SWZ != 0;
  auto swz = [](int row) { return kSwz ? row ^ (((row >> 3) & 3) << 1) : row; };
  // Staging unit u of thread t: LDS row prow0 + 64 u of its tile (A' for u < 2, then B' for u - 2), 16-byte column
  // c16. Its byte offset in the tile's slab, the buffer load's VGPR offset, is that row's: the LDS row for B', the
  // swizzled row for A'.
  const int tl = t & 31, c16 = t & 3;
  const int prow0 = 8 * (t >> 5) + (kSwz ? ((tl >> 2) & 1) * 4 + (tl >> 3) : (tl >> 2));
  uint32_t vo[U];
#pragma unroll
  for (int u = 0; u < U; u++) {
    const bool isA = u * T < UA;
    const int prow = prow0 + (isA ? u * T / QR : (u * T - UA) / QR);
    vo[u] = (uint32_t)(16 * (QR * (isA ? swz(prow) : prow) + c16));
  }
  // The lane's bases in buffer 0: its A' row in each tile (load_a's row rho = lane & 7, swizzled), its B' rows
  // j = 0-3 and 4-7 (8 * j rows apart), and its staging place. The empty asm hides how they relate, so the compiler
  // neither rebuilds one from another each slab nor merges the two A' reads into a ds_read2st64 (whose offsets
  // can't take a k-pair's 8 bytes).
  uint32_t la0 = (uint32_t)((wr * WM + swz(4 * (lane & 1) + ((lane >> 3) & 3) + 8 * ((lane & 7) >> 1))) * S);
  uint32_t la1 = la0 + (uint32_t)(P::kTM * S);
  uint32_t lb0 = (uint32_t)(BM * S + wc * WN * S + (lane & 7) * S);
  uint32_t lb4 = lb0 + (uint32_t)(32 * S);
  uint32_t lst = (uint32_t)(prow0 * S + 16 * c16);
  asm volatile("" : "+v"(la0), "+v"(la1), "+v"(lb0), "+v"(lb4), "+v"(lst));
  uint32_t par = 0u;   // the slab loop's buffers: 0 or kFlip bytes past buffer 0's
#define PEARL_AMD_LOAD(slab)                                                                 \
  _Pragma("unroll") for (int u = 0; u < U; u++) {                                            \
    pre[u] = __builtin_bit_cast(pearl_amd_v4i,                                               \
        u * T < UA ? __builtin_amdgcn_raw_buffer_load_b128(rsA, (int)vo[u], (int)((slab) * (uint32_t)(BM * KS)), 0) \
                   : __builtin_amdgcn_raw_buffer_load_b128(rsB, (int)vo[u], (int)((slab) * (uint32_t)(BN * KS)), 0)); \
  }
// The staged slab into the buffers po bytes past buffer 0's.
#define PEARL_AMD_STORE_AT(po)                                                               \
  _Pragma("unroll") for (int u = 0; u < U; u++) {                                            \
    int8_t *const dst = sL + (lst + (po));                                                   \
    *reinterpret_cast<pearl_amd_v4i *>(dst + (u * T < UA ? (u * T / QR) * S : BM * S + ((u * T - UA) / QR) * S)) = \
        pre[u];                                                                              \
  }
#define PEARL_AMD_STORE(buf) PEARL_AMD_STORE_AT((uint32_t)(buf) * kFlip)
#else
#define PEARL_AMD_LOAD(slab) PEARL_AMD_LOADR(slab, pre)
#define PEARL_AMD_STORE(buf) PEARL_AMD_STORER(buf, pre)
#endif

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
#if PEARL_AMD_ABLATE & 8
  uint32_t sink = 0u;   // PEARL_AMD_ABLATE bit 8: the words of the one fragment read a slab (slab_step)
#endif
  const uint32_t r = (uint32_t)((lane & 1) | ((lane >> 2) & 2) | ((lane >> 2) & 4));

#if PEARL_AMD_CRANK && PEARL_AMD_PF2
  // PEARL_AMD_CRANK (above): the host launches only rank PEARL_FOLD_RANK, so k is a whole number of chunks of 4.
  static_assert(PEARL_FOLD_RANK % (uint32_t)KS == 0u && ((PEARL_FOLD_RANK / (uint32_t)KS) & 1u) == 0u,
                "PEARL_AMD_CRANK: a chunk is an even number of slabs");
  constexpr uint32_t perChunk = PEARL_FOLD_RANK / (uint32_t)KS;
  const uint32_t slabs = k / (uint32_t)KS;
  __builtin_assume((slabs & (perChunk - 1u)) == 0u);
  (void)rank;
#elif PEARL_AMD_CMASK
  // PEARL_AMD_CMASK (above).
  const uint32_t slabs = k / (uint32_t)KS, perChunk = 1u << (uint32_t)__builtin_ctz(rank / (uint32_t)KS);
#else
  const uint32_t slabs = k / (uint32_t)KS, perChunk = rank / (uint32_t)KS;
#endif
  // PEARL_AMD_DIRECT (above): the RDNA 4 fold with no LDS, in a gfx12 device pass only.
#if PEARL_AMD_DIRECT && PEARL_AMD_WMMA12 && PEARL_AMD_BUFLD && PEARL_AMD_TILED && PEARL_AMD_KPAIR && PEARL_AMD_PF2 && \
    PEARL_AMD_CRANK && !PEARL_AMD_ABLATE && !PEARL_AMD_STAMP && !PEARL_AMD_FUSE
  constexpr bool kDirect = std::is_same<P, PearlAmdWmma12>::value;
#else
  constexpr bool kDirect = false;
#endif
  if constexpr (!kDirect) {
    PEARL_AMD_LOAD(0)
    PEARL_AMD_STORE(0)
    __syncthreads();
  }
#if PEARL_AMD_ABLATE & 8
#if PEARL_AMD_EMU || !defined(__HIP_DEVICE_COMPILE__)
  // PEARL_AMD_ABLATE bit 8, HIP-CPU: the first slab's fragments, loaded once and used at every k-step.
  typename P::AFrag a0[P::kMT];
  typename P::BFrag b0[P::kNT];
#pragma unroll
  for (int mt = 0; mt < P::kMT; mt++) a0[mt] = P::load_a(sA[0] + wr * WM * S + mt * P::kTM * S, S, lane);
#pragma unroll
  for (int nt = 0; nt < P::kNT; nt++) b0[nt] = P::load_b(sB[0] + wc * WN * S + nt * P::kTN * S, S, lane);
#endif
#endif
  // The end of chunk c: its transcript word, for every region the wave holds.
#if PEARL_AMD_KWRAP
  // PEARL_AMD_KWRAP: chunk c goes into word c % 16 after that word's rotl 13. Each word is written once
  // in every 16 chunks, so rotating them all when a pass starts (c % 16 == 0; at c = 0 they are still
  // zero) is the same as rotating each just before its write, and costs 8 VALU per 16 chunks. A chunk
  // then costs one select and one XOR a region, into the keep its word is in, where the default costs
  // two selects: c & 8 is the same in every lane, so choosing the keep is a branch, not a select. The
  // host takes past 16 only a multiple of 16 chunks, the whole passes this is right for.
  static_assert(!kDrop, "the readout-drop probe has no wrap form");
  // A chunk's readout is three parts, which PEARL_AMD_RDLATE=2 runs apart: the rotation every 16 chunks, the XOR of
  // the sums of each region the lane holds (x), and the cross-lane steps and the keep.
  auto readout_rot = [&](uint32_t c) {
    if ((c & 15u) == 0u) {
#pragma unroll
      for (int rb = 0; rb < RB; rb++)
#pragma unroll
        for (int cb = 0; cb < CB; cb++)
#pragma unroll
          for (int q = 0; q < NRO; q++) {
            keepLo[rb][cb][q] = pearl_rotl13(keepLo[rb][cb][q]);
            keepHi[rb][cb][q] = pearl_rotl13(keepHi[rb][cb][q]);
          }
    }
  };
  auto readout_sum = [&](uint32_t (&x)[RB][CB][NRO]) {
#pragma unroll
    for (int rb = 0; rb < RB; rb++)
#pragma unroll
      for (int cb = 0; cb < CB; cb++) {
#if PEARL_AMD_RDTREE
        if constexpr (std::is_same<P, PearlAmdWmma12>::value) {
          // PEARL_AMD_RDTREE (above): the region's sums in turn into three chains, li being a sum's place among its
          // region's 32 (registers g and g + 4 of a tile are the two regions), so the chains take 11, 11 and 10 sums.
          // On a GPU an empty asm hides each chain's end, so the compiler can't make the three one chain again.
          static_assert(P::kRegs == 8 && NRO == 2, "PEARL_AMD_RDTREE is for wmma12's registers");
          uint32_t y[NRO][3];
#pragma unroll
          for (int q = 0; q < NRO; q++) y[q][0] = y[q][1] = y[q][2] = 0u;
#pragma unroll
          for (int mt = rb * (32 / P::kTM); mt < (rb + 1) * (32 / P::kTM); mt++)
#pragma unroll
            for (int nt = cb * (64 / P::kTN); nt < (cb + 1) * (64 / P::kTN); nt++)
#pragma unroll
              for (int g = 0; g < P::kRegs; g++) {
                const int li = ((mt - rb * (32 / P::kTM)) * (64 / P::kTN) + (nt - cb * (64 / P::kTN))) * 4 + (g & 3);
                y[P::ro_of(g)][li % 3] ^= (uint32_t)acc[mt][nt][g];
              }
#pragma unroll
          for (int q = 0; q < NRO; q++) {
#if !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__)
            asm("" : "+v"(y[q][0]), "+v"(y[q][1]), "+v"(y[q][2]));
#endif  // a GPU
            x[rb][cb][q] = y[q][0] ^ y[q][1] ^ y[q][2];
          }
          continue;
        }
#endif  // PEARL_AMD_RDTREE
        uint32_t xr[NRO];
#pragma unroll
        for (int q = 0; q < NRO; q++) xr[q] = 0u;
#pragma unroll
        for (int mt = rb * (32 / P::kTM); mt < (rb + 1) * (32 / P::kTM); mt++)
#pragma unroll
          for (int nt = cb * (64 / P::kTN); nt < (cb + 1) * (64 / P::kTN); nt++)
#pragma unroll
            for (int g = 0; g < P::kRegs; g++) xr[P::ro_of(g)] ^= (uint32_t)acc[mt][nt][g];
#pragma unroll
        for (int q = 0; q < NRO; q++) x[rb][cb][q] = xr[q];
      }
  };
  auto readout_keep = [&](uint32_t c, const uint32_t (&xs)[RB][CB][NRO]) {
    const bool mine = (c & 7u) == r;
    uint32_t v[RB][CB][NRO];
#pragma unroll
    for (int rb = 0; rb < RB; rb++)
#pragma unroll
      for (int cb = 0; cb < CB; cb++) {
        const uint32_t *x = xs[rb][cb];
#pragma unroll
        for (int q = 0; q < NRO; q++) v[rb][cb][q] = pearl_amd_group_xor(x[q]);
      }
#if PEARL_AMD_RDSEL && !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__)
    if constexpr (std::is_same<P, PearlAmdWmma12>::value) {
      // PEARL_AMD_RDSEL (above): the keep by a scalar branch; the comment in each arm keeps the arms apart. The
      // arms change copies local to this lambda, put back after: with the arms storing to keepLo and keepHi
      // themselves, through the lambda's capture, the compiler merged the two arms' stores into one through a
      // phi of the two addresses, and the arrays went to scratch memory (48 bytes, as PEARL_AMD_RDBR did).
      static_assert(PEARL_AMD_RDSEL == 1, "PEARL_AMD_RDSEL is 0 or 1");
      uint32_t kLo[RB][CB][NRO], kHi[RB][CB][NRO];
#pragma unroll
      for (int rb = 0; rb < RB; rb++)
#pragma unroll
        for (int cb = 0; cb < CB; cb++)
#pragma unroll
          for (int q = 0; q < NRO; q++) {
            kLo[rb][cb][q] = keepLo[rb][cb][q];
            kHi[rb][cb][q] = keepHi[rb][cb][q];
          }
      {
        if (c & 8u) {
          asm volatile("; rdsel hi");
#pragma unroll
          for (int rb = 0; rb < RB; rb++)
#pragma unroll
            for (int cb = 0; cb < CB; cb++)
#pragma unroll
              for (int q = 0; q < NRO; q++) kHi[rb][cb][q] ^= mine ? v[rb][cb][q] : 0u;
        } else {
          asm volatile("; rdsel lo");
#pragma unroll
          for (int rb = 0; rb < RB; rb++)
#pragma unroll
            for (int cb = 0; cb < CB; cb++)
#pragma unroll
              for (int q = 0; q < NRO; q++) kLo[rb][cb][q] ^= mine ? v[rb][cb][q] : 0u;
        }
      }
#pragma unroll
      for (int rb = 0; rb < RB; rb++)
#pragma unroll
        for (int cb = 0; cb < CB; cb++)
#pragma unroll
          for (int q = 0; q < NRO; q++) {
            keepLo[rb][cb][q] = kLo[rb][cb][q];
            keepHi[rb][cb][q] = kHi[rb][cb][q];
          }
      return;
    }
#endif  // PEARL_AMD_RDSEL
    if (c & 8u) {
#pragma unroll
      for (int rb = 0; rb < RB; rb++)
#pragma unroll
        for (int cb = 0; cb < CB; cb++)
#pragma unroll
          for (int q = 0; q < NRO; q++) keepHi[rb][cb][q] ^= mine ? v[rb][cb][q] : 0u;
    } else {
#pragma unroll
      for (int rb = 0; rb < RB; rb++)
#pragma unroll
        for (int cb = 0; cb < CB; cb++)
#pragma unroll
          for (int q = 0; q < NRO; q++) keepLo[rb][cb][q] ^= mine ? v[rb][cb][q] : 0u;
    }
  };
  auto readout = [&](uint32_t c) {
#if PEARL_AMD_ABLATE & 64
#if !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__)
    if constexpr (std::is_same<P, PearlAmdWmma12>::value) {
      // PEARL_AMD_ABLATE bit 64 (timing only): each lane XORs its own part into keepLo. No rotate, cross-lane
      // step, select or branch.
#pragma unroll
      for (int rb = 0; rb < RB; rb++)
#pragma unroll
        for (int cb = 0; cb < CB; cb++)
#pragma unroll
          for (int mt = rb * (32 / P::kTM); mt < (rb + 1) * (32 / P::kTM); mt++)
#pragma unroll
            for (int nt = cb * (64 / P::kTN); nt < (cb + 1) * (64 / P::kTN); nt++)
#pragma unroll
              for (int g = 0; g < P::kRegs; g++) keepLo[rb][cb][P::ro_of(g)] ^= (uint32_t)acc[mt][nt][g];
      return;
    }
#endif  // a GPU
#endif  // PEARL_AMD_ABLATE & 64
    readout_rot(c);
    uint32_t x[RB][CB][NRO];
    readout_sum(x);
    readout_keep(c, x);
  };
#else
  auto readout = [&](uint32_t c) {
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
          if constexpr (kDrop) {
            keepLo[rb][cb][q] ^= x[q];   // timing only: the transcript is wrong
          } else {
            const uint32_t v = pearl_amd_group_xor(x[q]);
            keepLo[rb][cb][q] = (mine && !hi) ? v : keepLo[rb][cb][q];
            keepHi[rb][cb][q] = (mine && hi) ? v : keepHi[rb][cb][q];
          }
        }
      }
  };
#endif
  // PEARL_AMD_RDLATE (above), in the RDNA 4 fold only. It needs the set's slab loop: the fragment loads it runs after
  // are KPAIR's, and with CRANK a chunk is a constant 4 slabs, so only slabs 4c + 4 (and 4c + 5) test for it.
  constexpr bool kLate = PEARL_AMD_RDLATE != 0 && PEARL_AMD_KWRAP && std::is_same<P, PearlAmdWmma12>::value;
  static_assert(!(PEARL_AMD_RDLATE && std::is_same<P, PearlAmdWmma12>::value) ||
                    (PEARL_AMD_KWRAP && PEARL_AMD_KPAIR && PEARL_AMD_CRANK && PEARL_AMD_PF2 && !PEARL_AMD_UNROLL4),
                "PEARL_AMD_RDLATE needs the RDNA 4 set's slab loop (KWRAP, KPAIR, CRANK, PF2) and no UNROLL4");
#if PEARL_AMD_ABLATE & 8
  static_assert(!kLate, "PEARL_AMD_RDLATE runs the readout after the fragment loads, which ablation bit 8 leaves out");
#endif
#if PEARL_AMD_ABLATE & 64
  static_assert(!kLate || PEARL_AMD_RDLATE == 1, "ablation bit 64 is in the readout RDLATE=1 runs, not in =2's parts");
#endif
  (void)kLate;
#if PEARL_AMD_RDLATE && PEARL_AMD_KWRAP && PEARL_AMD_CRANK && PEARL_AMD_PF2
#if PEARL_AMD_RDLATE == 2
  uint32_t xl[RB][CB][NRO];   // the sums of the chunk that ended last, from slab 4c + 4 to the keep in 4c + 5
#endif
  // In slab s, after its fragment loads are issued and before its WMMAs: chunk s / 4 - 1 if it ended with the slab
  // before (s = 4c + 4), and with =2 the rest of it a slab later (s = 4c + 5).
  auto readout_late = [&](uint32_t s, auto parity) {
    constexpr int B = decltype(parity)::value;
    if constexpr (B == 0 || PEARL_AMD_RDLATE == 2) {
      PEARL_AMD_RDLATE_PIN();   // the fragment loads stay above the readout, in flight while it runs
      if (s % perChunk == (uint32_t)B && s >= perChunk) {
        const uint32_t c = s / perChunk - 1u;
#if PEARL_AMD_RDLATE == 1
        readout(c);
#else
        if constexpr (B == 0) {
          readout_sum(xl);
        } else {
          readout_rot(c);
          readout_keep(c, xl);
        }
#endif
      }
    }
  };
  // A use of every sum right after a slab's WMMAs. With the readout's branch in the slab, the compiler had moved the
  // even slab's WMMAs below its stores and barrier, beside the odd slab's, so two slabs' fragments were live at once
  // (234 VGPRs). No instruction: the asm is empty.
  auto readout_hold = [&]() {
#if !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
    for (int mt = 0; mt < P::kMT; mt++)
#pragma unroll
      for (int nt = 0; nt < P::kNT; nt++) asm volatile("" ::"v"(acc[mt][nt]));
#endif
  };
#else
  auto readout_late = [](uint32_t, auto) {};   // PEARL_AMD_RDLATE off
  auto readout_hold = []() {};
#endif  // PEARL_AMD_RDLATE
  (void)readout_late;
  (void)readout_hold;
#if PEARL_AMD_PF2
#if PEARL_AMD_BUFLD && PEARL_AMD_TILED && PEARL_AMD_WMMA12
  if constexpr (kDirect) {
    // PEARL_AMD_DIRECT (above). Lane l reads row l % 16 of each 16-row tile at byte 16 * (l / 16) of the slab's 32,
    // the bytes KPAIR's LDS read gave it: [0, 1] for the even k-step, [2, 3] for the odd one.
    static_assert(KS == 32 && P::kTK == 16, "PEARL_AMD_DIRECT reads a 32-byte slab as KPAIR does");
    constexpr int NF = P::kMT + P::kNT;
    uint32_t voA = (uint32_t)((wr * WM + (lane & 15)) * KS + 16 * (lane >> 4));
    uint32_t voB = (uint32_t)((wc * WN + (lane & 15)) * KS + 16 * (lane >> 4));
#if !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__)
    asm volatile("" : "+v"(voA), "+v"(voB));
#endif
    auto dload = [&](uint32_t sl, pearl_amd_v4i (&f)[NF]) {
#pragma unroll
      for (int mt = 0; mt < P::kMT; mt++)
        f[mt] = __builtin_bit_cast(pearl_amd_v4i, __builtin_amdgcn_raw_buffer_load_b128(
                    rsA, (int)(voA + (uint32_t)(mt * P::kTM * KS)), (int)(sl * (uint32_t)(BM * KS)), 0));
#pragma unroll
      for (int nt = 0; nt < P::kNT; nt++)
        f[P::kMT + nt] = __builtin_bit_cast(pearl_amd_v4i, __builtin_amdgcn_raw_buffer_load_b128(
                             rsB, (int)(voB + (uint32_t)(nt * P::kTN * KS)), (int)(sl * (uint32_t)(BN * KS)), 0));
    };
    auto dstep = [&](uint32_t s, const pearl_amd_v4i (&cur)[NF], pearl_amd_v4i (&nxt)[NF]) {
      // PEARL_AMD_DIRECT=2: the load is for two slabs ahead. Always a load, no branch; past the end it reloads the last.
      constexpr uint32_t kAhead = PEARL_AMD_DIRECT == 2 ? 2u : 1u;
      dload(s + kAhead < slabs ? s + kAhead : slabs - 1u, nxt);
      typename P::AFrag a[2][P::kMT];
      typename P::BFrag b[2][P::kNT];
#pragma unroll
      for (int mt = 0; mt < P::kMT; mt++) {
        a[0][mt] = pearl_amd_v2i{cur[mt][0], cur[mt][1]};
        a[1][mt] = pearl_amd_v2i{cur[mt][2], cur[mt][3]};
      }
#pragma unroll
      for (int nt = 0; nt < P::kNT; nt++) {
        b[0][nt] = pearl_amd_v2i{cur[P::kMT + nt][0], cur[P::kMT + nt][1]};
        b[1][nt] = pearl_amd_v2i{cur[P::kMT + nt][2], cur[P::kMT + nt][3]};
      }
      pearl_amd_mma_step<P>(acc, a[0], b[0]);
      pearl_amd_mma_step<P>(acc, a[1], b[1]);
      if ((s + 1) % perChunk == 0) readout((s + 1) / perChunk - 1);
    };
#if PEARL_AMD_DIRECT == 2
    // Three register sets: slab s is in set s mod 3 and its step loads slab s + 2 into set (s + 2) mod 3.
    pearl_amd_v4i fr0[NF], fr1[NF], fr2[NF];
    dload(0, fr0);
    dload(slabs > 1u ? 1u : 0u, fr1);
    uint32_t s = 0;
    for (; s + 2 < slabs; s += 3) {
      dstep(s, fr0, fr2);
      dstep(s + 1, fr1, fr0);
      dstep(s + 2, fr2, fr1);
    }
    if (s < slabs) dstep(s, fr0, fr2);
    if (s + 1 < slabs) dstep(s + 1, fr1, fr0);
#else
    pearl_amd_v4i fr0[NF], fr1[NF];
    dload(0, fr0);
    uint32_t s = 0;
    for (; s + 1 < slabs; s += 2) {
      dstep(s, fr0, fr1);
      dstep(s + 1, fr1, fr0);
    }
    if (s < slabs) dstep(s, fr0, fr1);
#endif
  }
#endif  // PEARL_AMD_DIRECT's loop
  if constexpr (!kDirect) {
  // pre2[q] holds the slab whose number is q mod 2. Slab 0 is in LDS buffer 0 already.
  pearl_amd_v4i pre2[2][U];
  if (slabs > 1u) { PEARL_AMD_LOADR(1, pre2[1]) }
#if PEARL_AMD_STAMP
  stLast = pearl_amd_sgpr(pearl_amd_cycles());   // the slab loop starts: before here is the prologue
  stPro = pearl_amd_sgpr(pearl_amd_cycles_since(stCyc0, stLast));
  // PEARL_AMD_STAMP: the slab barrier as its parts, the counter read each side of the s_barrier inside the fence
  // pair, and once before the pair, so the release fence's memory wait has a number of its own. PFX's pair is
  // LDS-only; without PFX it is __syncthreads()'s, for every address space. Bit 2 of PEARL_AMD_ABLATE leaves out
  // the s_barrier here too. Each sum goes through pearl_amd_sgpr, and each crossing is one compare and one
  // select, never an && of two conditions.
  auto stamp_barrier = [&]() {
    const uint32_t cm = pearl_amd_cycles();
#if PEARL_AMD_PFX
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
#else
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
#if PEARL_AMD_WMMA11
    __builtin_amdgcn_s_waitcnt(0);   // vmcnt, lgkmcnt and expcnt 0: the release's wait, before the first read
#endif
#endif
    const uint32_t c0 = pearl_amd_cycles();
#if !(PEARL_AMD_ABLATE & 2)
    __builtin_amdgcn_s_barrier();
#endif
    const uint32_t c1 = pearl_amd_cycles();
#if PEARL_AMD_PFX
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#else
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
#endif
    stMem = pearl_amd_sgpr(stMem + pearl_amd_cycles_since(cm, c0));
    stBar = pearl_amd_sgpr(stBar + pearl_amd_cycles_since(c0, c1));
    stLoop = pearl_amd_sgpr(stLoop + pearl_amd_cycles_since(stLast, c1));
    // The first barrier exits after bits 16 and 19 change: bits 0 and 1 of nx, less the crossings already seen.
    const uint32_t chg = c1 ^ stLast;
    const uint32_t nx = (((chg >> 16) != 0u ? 1u : 0u) | ((chg >> 19) != 0u ? 2u : 0u)) & ~stSeen;
    stX16 = pearl_amd_sgpr((nx & 1u) != 0u ? c1 : stX16);
    stX19 = pearl_amd_sgpr((nx & 2u) != 0u ? c1 : stX19);
    stSeen = pearl_amd_sgpr(stSeen | nx);
    stLast = pearl_amd_sgpr(c1);
  };
#endif
  auto slab_step = [&](uint32_t s, auto parity) {
    constexpr int B = decltype(parity)::value;   // s mod 2: slab s's LDS buffer and register set
#if PEARL_AMD_ABLATE & 32
    // PEARL_AMD_ABLATE bit 32: no slab ahead is loaded.
#elif PEARL_AMD_PFX
    {   // set B's slab s went to LDS last step
      const uint32_t n = s + 2 < slabs ? s + 2 : slabs - 1u;
      PEARL_AMD_LOADR(n, pre2[B])
    }
#else
    if (s + 2 < slabs) { PEARL_AMD_LOADR(s + 2, pre2[B]) }   // set B's slab s went to LDS last step
#endif
#if PEARL_AMD_LPIN == 2 && !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__)
    asm volatile("" ::: "memory");   // PEARL_AMD_LPIN=2 (above)
#endif
#if PEARL_AMD_LPIN
    PEARL_AMD_SCHED_PIN();   // PEARL_AMD_LPIN (above)
#endif
    const int8_t *aBase = sA[B] + wr * WM * S;
    const int8_t *bBase = sB[B] + wc * WN * S;
#pragma unroll
    for (int kk = 0; kk < KS / P::kTK; kk++) {
#if PEARL_AMD_ABLATE & 8
      (void)aBase; (void)bBase;
#if PEARL_AMD_EMU || !defined(__HIP_DEVICE_COMPILE__)
      pearl_amd_mma_step<P>(acc, a0, b0);
#else
      // PEARL_AMD_ABLATE bit 8: new fragments with no load. Fragments kept across the loop would
      // hold 16-32 more VGPRs. The asm isn't volatile and touches no memory, so nothing waits on
      // it; its input differs at every k-step, so the compiler can neither hoist nor merge it.
      static_assert(P::kMT == 4 && P::kNT == 4, "PEARL_AMD_ABLATE bit 8 is for the RDNA folds");
      typename P::AFrag a[P::kMT];
      typename P::BFrag b[P::kNT];
      asm("" : "=v"(a[0]), "=v"(a[1]), "=v"(a[2]), "=v"(a[3]), "=v"(b[0]), "=v"(b[1]), "=v"(b[2]), "=v"(b[3])
          : "s"(s * (uint32_t)(KS / P::kTK) + (uint32_t)kk));
      pearl_amd_mma_step<P>(acc, a, b);
#endif
#else
#if PEARL_AMD_KPAIR
      if constexpr (std::is_same<P, PearlAmdWmma12>::value) {
        // PEARL_AMD_KPAIR (above): k-steps kk and kk + 1 from one 16-byte read a fragment, at the even kk.
        static_assert((KS / P::kTK) % 2 == 0, "PEARL_AMD_KPAIR pairs the k-steps of a slab");
        if ((kk & 1) == 0) {
          typename P::AFrag a[2][P::kMT];
          typename P::BFrag b[2][P::kNT];
#if PEARL_AMD_LDS32 && PEARL_AMD_LDS32_PIN
          const int off = l32rd + kk * P::kTK;
#elif PEARL_AMD_LDS32
          // PEARL_AMD_LDS32: the half is swapped by the row's bits (tile offsets are whole 16 rows: bits 0-3 kept).
          const int off = (lane & 15) * S + 16 * ((lane >> 4) ^ (kL32 ? pearl_amd_l32_swz(lane & 15) : 0)) + kk * P::kTK;
#else
          const int off = (lane & 15) * S + 16 * (lane >> 4) + kk * P::kTK;
#endif
          if constexpr (kLate) {
            // PEARL_AMD_RDLATE: the readout's branch comes between these loads and the WMMAs, so the compiler can't
            // order the loads by first use, as it does in the release. They go in the WMMAs' order here: A' tile 0,
            // every B' tile, then the rest of A', so the first WMMA waits for two loads, not five. A sched_barrier
            // after each keeps that order (the scheduler put B' first).
#pragma unroll
            for (int i = 0; i < P::kMT + P::kNT; i++) {
              const bool isB = i > 0 && i <= P::kNT;
              const int t = isB ? i - 1 : (i == 0 ? 0 : i - P::kNT);
              const pearl_amd_v4i q = *reinterpret_cast<const pearl_amd_v4i *>(
                  isB ? bBase + t * P::kTN * S + off : aBase + t * P::kTM * S + off);
              PEARL_AMD_RDLATE_PIN();
              if (isB) {
                b[0][t] = pearl_amd_v2i{q[0], q[1]};
                b[1][t] = pearl_amd_v2i{q[2], q[3]};
              } else {
                a[0][t] = pearl_amd_v2i{q[0], q[1]};
                a[1][t] = pearl_amd_v2i{q[2], q[3]};
              }
            }
          } else {
#pragma unroll
            for (int mt = 0; mt < P::kMT; mt++) {
              const pearl_amd_v4i q = *reinterpret_cast<const pearl_amd_v4i *>(aBase + mt * P::kTM * S + off);
              a[0][mt] = pearl_amd_v2i{q[0], q[1]};
              a[1][mt] = pearl_amd_v2i{q[2], q[3]};
            }
#pragma unroll
            for (int nt = 0; nt < P::kNT; nt++) {
              const pearl_amd_v4i q = *reinterpret_cast<const pearl_amd_v4i *>(bBase + nt * P::kTN * S + off);
              b[0][nt] = pearl_amd_v2i{q[0], q[1]};
              b[1][nt] = pearl_amd_v2i{q[2], q[3]};
            }
          }
          if constexpr (kLate) readout_late(s, parity);   // PEARL_AMD_RDLATE (above)
          pearl_amd_mma_step<P>(acc, a[0], b[0]);
          pearl_amd_mma_step<P>(acc, a[1], b[1]);
          if constexpr (kLate) readout_hold();
        }
      } else
#endif
      {
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
#endif
    }
    if constexpr (!kLate)   // PEARL_AMD_RDLATE runs it in the next slab instead
#if PEARL_AMD_CRANK
    // PEARL_AMD_CRANK (above): s mod 2 is B and a chunk is an even number of slabs, so only an odd slab ends one.
    if constexpr (B == 1)
#endif
    if ((s + 1) % perChunk == 0) readout((s + 1) / perChunk - 1);
#if PEARL_AMD_SCHEDB
    PEARL_AMD_SCHED_PIN();   // PEARL_AMD_SCHEDB (above)
#endif
#if PEARL_AMD_ABLATE & 8
#if !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__)
    // PEARL_AMD_ABLATE bit 8: one real fragment read a slab, the release's load at k-step 0 of tile 0. Slabs 0
    // and 1 of every 4 read A', slabs 2 and 3 B', and B is s mod 2, so the four LDS buffers take turns: each
    // has a reader, and the pick is made at run time, so the compiler can't drop a store as unread. load_b is
    // load_a on both RDNA folds, so one call takes either address. Only `sink` uses the words, after the
    // barrier, and only the stored words use `sink`: its wait falls in the barrier's, not in the matrix work.
    const typename P::AFrag kept = P::load_a((s & 2u) ? bBase : aBase, S, lane);
#endif
#endif
#if !(PEARL_AMD_ABLATE & 32)
    if (s + 1 < slabs) { PEARL_AMD_STORER(B ^ 1, pre2[B ^ 1]) }   // slab s + 1, loaded a step ago
#endif
#if PEARL_AMD_STAMP
    stamp_barrier();
#elif PEARL_AMD_ABLATE & 2
    // PEARL_AMD_ABLATE bit 2: the barrier's fences without its s_barrier. __syncthreads() is the
    // workgroup pair for every address space, PFX's barrier the LDS-only pair.
#if !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__)
#if PEARL_AMD_PFX
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#else
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
#endif
#endif
#elif PEARL_AMD_PFX
    pearl_amd_lds_barrier();
#else
    __syncthreads();
#endif
#if PEARL_AMD_ABLATE & 8
#if !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
    for (int i = 0; i < (int)(sizeof kept / sizeof(int32_t)); i++) sink ^= (uint32_t)kept[i];
#endif
#endif
  };
  uint32_t s = 0;
#if PEARL_AMD_UNROLL4
  // PEARL_AMD_UNROLL4 (above): s is a multiple of 4 here, so the fourth step's readout needs no test.
  for (; s + 3 < slabs; s += 4) {
    slab_step(s, std::integral_constant<int, 0>());
    slab_step(s + 1, std::integral_constant<int, 1>());
    slab_step(s + 2, std::integral_constant<int, 0>());
    slab_step(s + 3, std::integral_constant<int, 1>());
  }
#endif
#if PEARL_AMD_RDLATE
#pragma clang loop unroll(disable)   // PEARL_AMD_RDLATE: the compiler peeled the slabs before the first readout
#endif
  for (; s + 1 < slabs; s += 2) {
    slab_step(s, std::integral_constant<int, 0>());
    slab_step(s + 1, std::integral_constant<int, 1>());
  }
  if constexpr (!kLate)   // RDLATE has CRANK: slabs is a whole number of 4-slab chunks, so the pairs ran them all
  if (s < slabs) slab_step(s, std::integral_constant<int, 0>());
#if PEARL_AMD_STAMP
  {   // PEARL_AMD_STAMP: the slab loop ends. Its sum runs to here, just after the loop's last barrier, and the
      // epilogue starts here.
    const uint32_t c = pearl_amd_cycles();
    stLoop = pearl_amd_sgpr(stLoop + pearl_amd_cycles_since(stLast, c));
    stLast = pearl_amd_sgpr(c);
  }
#endif
  }  // !kDirect
#else
#if PEARL_AMD_ABLATE & 42
  // Only a compile pass that builds this fold stops here: the condition depends on P.
  static_assert(sizeof(P) == 0, "PEARL_AMD_ABLATE bits 2, 8 and 32 are in the PF2 loop only: add -DPEARL_AMD_PF2=1");
#endif
#if PEARL_AMD_STAMP
  static_assert(!std::is_same<P, PearlAmdWmma11>::value && !std::is_same<P, PearlAmdWmma12>::value,
                "PEARL_AMD_STAMP is in the PF2 loop only: add -DPEARL_AMD_PF2=1");
#endif
  for (uint32_t s = 0; s < slabs; s++) {
    const int buf = (int)(s & 1u);
    if (s + 1 < slabs) { PEARL_AMD_LOAD(s + 1) }
#if PEARL_AMD_DOT4_ADDR_ON
    {   // PEARL_AMD_DOT4_ADDR (above): RDNA 2's slab, from the buffers par bytes past buffer 0's, then the next one.
      const int8_t *const aR0 = sL + (la0 + par);
      const int8_t *const aR1 = sL + (la1 + par);
      const int8_t *const bR0 = sL + (lb0 + par);
      const int8_t *const bR4 = sL + (lb4 + par);
#pragma unroll
      for (int kp = 0; kp < KS / 8; kp++) {
#if PEARL_AMD_DOT4_SPLIT
        asm volatile("" ::: "memory");
#endif
        // k-pair kp: the lane's A' row of each tile and its eight B' rows, 8 bytes each, two k-steps.
        const pearl_amd_v2i aw2[2] = {*reinterpret_cast<const pearl_amd_v2i *>(aR0 + 8 * kp),
                                      *reinterpret_cast<const pearl_amd_v2i *>(aR1 + 8 * kp)};
        pearl_amd_v2i b2[8];
#pragma unroll
        for (int j = 0; j < 8; j++)
          b2[j] = *reinterpret_cast<const pearl_amd_v2i *>((j < 4 ? bR0 + 8 * j * S : bR4 + 8 * (j - 4) * S) + 8 * kp);
        // PEARL_AMD_DOT4_STKP (above): the next slab's store once k-pair STKP's reads are issued.
        if constexpr (PEARL_AMD_DOT4_STKP < KS / 8) {
          if (kp == PEARL_AMD_DOT4_STKP && s + 1 < slabs) { PEARL_AMD_STORE_AT(kFlip - par) }
        }
#pragma unroll
        for (int q = 0; q < 2; q++) {
#define PEARL_AMD_DPPA_ROW2(R) \
  _Pragma("unroll") for (int j = 0; j < 8; j++) pearl_amd_dot4_from<R>(acc[mt][0][8 * R + j], aw2[mt][q], b2[j][q]);
#pragma unroll
          for (int mt = 0; mt < 2; mt++) {
            PEARL_AMD_DPPA_ROW2(0) PEARL_AMD_DPPA_ROW2(1) PEARL_AMD_DPPA_ROW2(2) PEARL_AMD_DPPA_ROW2(3)
            PEARL_AMD_DPPA_ROW2(4) PEARL_AMD_DPPA_ROW2(5) PEARL_AMD_DPPA_ROW2(6) PEARL_AMD_DPPA_ROW2(7)
          }
#undef PEARL_AMD_DPPA_ROW2
        }
      }
      // The next slab's store before the readout, as below (kStoreFirst).
      if constexpr (PEARL_AMD_DOT4_STKP >= KS / 8) {
        if (s + 1 < slabs) { PEARL_AMD_STORE_AT(kFlip - par) }
      }
      if ((s + 1) % perChunk == 0) readout((s + 1) / perChunk - 1);
      __syncthreads();
      par = kFlip - par;
      continue;
    }
#endif  // PEARL_AMD_DOT4_ADDR_ON
    const int8_t *aBase = sA[buf] + wr * WM * S;
    const int8_t *bBase = sB[buf] + wc * WN * S;
#if PEARL_AMD_DOT4_DPPA_ON
    if constexpr (std::is_same<P, PearlAmdDot4>::value) {
      // PEARL_AMD_DOT4_DPPA (above): this lane's A' row is rho = lane & 7 of its group's eight (load_a's rows).
      const int8_t *aRow = aBase + (4 * (lane & 1) + ((lane >> 3) & 3) + 8 * ((lane & 7) >> 1)) * S;
#if PEARL_AMD_DOT4_DPPA_KB == 8
      // PEARL_AMD_DOT4_DPPA_KB (above): 8 bytes of k a read, two k-steps a pass.
#pragma unroll
      for (int kp = 0; kp < KS / 8; kp++) {
#if PEARL_AMD_DOT4_SPLIT
        asm volatile("" ::: "memory");
#endif
        const pearl_amd_v2i aw2[2] = {*reinterpret_cast<const pearl_amd_v2i *>(aRow + 8 * kp),
                                      *reinterpret_cast<const pearl_amd_v2i *>(aRow + P::kTM * S + 8 * kp)};
        pearl_amd_v2i b2[8];
#pragma unroll
        for (int j = 0; j < 8; j++)
          b2[j] = *reinterpret_cast<const pearl_amd_v2i *>(bBase + ((lane & 7) + 8 * j) * S + 8 * kp);
#pragma unroll
        for (int q = 0; q < 2; q++) {
#define PEARL_AMD_DPPA_ROW2(R) \
  _Pragma("unroll") for (int j = 0; j < 8; j++) pearl_amd_dot4_from<R>(acc[mt][0][8 * R + j], aw2[mt][q], b2[j][q]);
#pragma unroll
          for (int mt = 0; mt < 2; mt++) {
            PEARL_AMD_DPPA_ROW2(0) PEARL_AMD_DPPA_ROW2(1) PEARL_AMD_DPPA_ROW2(2) PEARL_AMD_DPPA_ROW2(3)
            PEARL_AMD_DPPA_ROW2(4) PEARL_AMD_DPPA_ROW2(5) PEARL_AMD_DPPA_ROW2(6) PEARL_AMD_DPPA_ROW2(7)
          }
#undef PEARL_AMD_DPPA_ROW2
        }
      }
#else
#pragma unroll
      for (int kk = 0; kk < KS / P::kTK; kk++) {
#if PEARL_AMD_DOT4_SPLIT
        asm volatile("" ::: "memory");
#endif
        const int32_t aw[2] = {*reinterpret_cast<const int32_t *>(aRow + kk * P::kTK),
                               *reinterpret_cast<const int32_t *>(aRow + P::kTM * S + kk * P::kTK)};
        const typename P::BFrag b = P::load_b(bBase + kk * P::kTK, S, lane);
        // Row by row, as the old k-step's mma: each row's eight dot4s take the same A' word.
#define PEARL_AMD_DPPA_ROW(R) \
  _Pragma("unroll") for (int j = 0; j < 8; j++) pearl_amd_dot4_from<R>(acc[mt][0][8 * R + j], aw[mt], b[j]);
#pragma unroll
        for (int mt = 0; mt < 2; mt++) {
          PEARL_AMD_DPPA_ROW(0) PEARL_AMD_DPPA_ROW(1) PEARL_AMD_DPPA_ROW(2) PEARL_AMD_DPPA_ROW(3)
          PEARL_AMD_DPPA_ROW(4) PEARL_AMD_DPPA_ROW(5) PEARL_AMD_DPPA_ROW(6) PEARL_AMD_DPPA_ROW(7)
        }
#undef PEARL_AMD_DPPA_ROW
      }
#endif  // PEARL_AMD_DOT4_DPPA_KB
    } else
#endif
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
    // RDNA 2's DPPA fold stores the next slab before the readout, not after, so the stores finish during the
    // readout's VALU instead of in the wait before the barrier: +0.4% on the RX 6800 and the RX 6600 (hashrate.js
    // 18.95-18.98 against 18.89-18.90 TH/s, 13 November). Every other fold keeps the old order.
    constexpr bool kStoreFirst = PEARL_AMD_DOT4_DPPA_ON && std::is_same<P, PearlAmdDot4>::value;
    if constexpr (kStoreFirst) {
      if (s + 1 < slabs) { PEARL_AMD_STORE(buf ^ 1) }
    }
    if ((s + 1) % perChunk == 0) readout((s + 1) / perChunk - 1);
    if constexpr (!kStoreFirst) {
      if (s + 1 < slabs) { PEARL_AMD_STORE(buf ^ 1) }
    }
    __syncthreads();
  }

#endif

  if constexpr (kLate) readout(slabs / perChunk - 1u);   // PEARL_AMD_RDLATE: the last chunk has no next slab
#if PEARL_AMD_FUSE
  {   // PEARL_AMD_FUSE (above): the tile's regions through LDS to a thread each, which hashes and tests one.
    constexpr int RR = 2 * BM / 32, RC = 4 * BN / 64;   // regions down and across the tile: 8 and 16
    static_assert(2 * RR * RC == T, "half the block hashes, a region a thread");
    static_assert(2 * BN * S >= RR * RC * (int)PEARL_JACKPOT_BUCKETS * 4, "the B' staging holds the transcripts");
    uint32_t *const sT = reinterpret_cast<uint32_t *>(&sB[0][0]);   // region j's words at 16 j
    __syncthreads();   // every wave is done reading the last slab
    const uint32_t co = (uint32_t)((lane >> 1) & 3);
#pragma unroll
    for (int rb = 0; rb < RB; rb++)
#pragma unroll
      for (int cb = 0; cb < CB; cb++)
#pragma unroll
        for (int q = 0; q < NRO; q++) {
          // The store loop's row and column offsets below, less the tile's first.
          const uint32_t lr = 2u * (uint32_t)((wr * WM + rb * 32) / 32) + (P::kRoByLane ? (uint32_t)(lane >> 5) : (uint32_t)q);
          const uint32_t lc = 4u * (uint32_t)((wc * WN + cb * 64) / 64) + co;
          uint32_t *const o = sT + (lc * RR + lr) * PEARL_JACKPOT_BUCKETS;
          o[r] = keepLo[rb][cb][q];
          o[r + 8] = keepHi[rb][cb][q];
        }
    __syncthreads();
    // Threads 0-127 hash in half the blocks and 128-255 in the other half, so the work doesn't always land on
    // the same waves' SIMDs, however the hardware spreads a block's waves over them.
    const uint32_t j = (uint32_t)t ^ (((blockIdx.x ^ blockIdx.y) & 1u) << 7);
    if (j < (uint32_t)(RR * RC)) {
      uint32_t tm[16];
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const pearl_amd_v4i v = reinterpret_cast<const pearl_amd_v4i *>(sT + j * PEARL_JACKPOT_BUCKETS)[u];
        tm[4 * u] = (uint32_t)v[0]; tm[4 * u + 1] = (uint32_t)v[1];
        tm[4 * u + 2] = (uint32_t)v[2]; tm[4 * u + 3] = (uint32_t)v[3];
      }
      const uint32_t rowIdx = 2u * (tileRow / 32u) + j % (uint32_t)RR;
      const uint32_t colIdx = 4u * (tileCol / 64u) + j / (uint32_t)RR;
      pearl_amd_hash_region(tm, colIdx * rows_valid + rowIdx, test, hits);
    }
  }
#if !PEARL_AMD_STAMP
  if (tr == nullptr) return;   // the miner's launch; the fold test's passes a buffer and gets the words too
#endif
#endif

  // Each lane writes words r and r + 8 of its regions.
  const uint32_t coIdx = (uint32_t)((lane >> 1) & 3);
#pragma unroll
  for (int rb = 0; rb < RB; rb++)
#pragma unroll
    for (int cb = 0; cb < CB; cb++)
#pragma unroll
      for (int q = 0; q < NRO; q++) {
#if PEARL_AMD_STAMP
#if PEARL_AMD_FUSE
        if (tr == nullptr) break;   // no words to store, but the stamps below still are
#endif
#endif
        const uint32_t roIdx = P::kRoByLane ? (uint32_t)(lane >> 5) : (uint32_t)q;
        const uint32_t rowIdx = 2u * ((tileRow + (uint32_t)(wr * WM + rb * 32)) / 32u) + roIdx;
        const uint32_t colIdx = 4u * ((tileCol + (uint32_t)(wc * WN + cb * 64)) / 64u) + coIdx;
        uint32_t *w = tr + ((size_t)colIdx * rows_valid + rowIdx) * PEARL_JACKPOT_BUCKETS;
#if PEARL_AMD_ABLATE & 8
        // PEARL_AMD_ABLATE: wrong words on purpose, so the fold check always fails. Bit 8's sink goes in
        // too, so its fragment reads are needed.
        w[r] = keepLo[rb][cb][q] ^ 0x5A5A5A5Au ^ sink;
        w[r + 8] = keepHi[rb][cb][q] ^ 0x5A5A5A5Au ^ sink;
#elif PEARL_AMD_ABLATE
        // PEARL_AMD_ABLATE: wrong words on purpose, so the fold check always fails.
        w[r] = keepLo[rb][cb][q] ^ 0x5A5A5A5Au;
        w[r + 8] = keepHi[rb][cb][q] ^ 0x5A5A5A5Au;
#else
        w[r] = keepLo[rb][cb][q];
        w[r + 8] = keepHi[rb][cb][q];
#endif
      }
#if PEARL_AMD_STAMP
  {   // PEARL_AMD_STAMP: the end, then lane 0 of each wave stores its words, built here from SGPRs.
    const uint32_t cyc = pearl_amd_sgpr(pearl_amd_cycles());
    const uint32_t rt = pearl_amd_sgpr(pearl_amd_realtime());
    const uint32_t hw = pearl_amd_sgpr(pearl_amd_hw_id1());
    uint32_t *const out = pearl_amd_stamps;   // constant memory: a scalar load
    if (out && lane == 0) {
      const uint32_t block = blockIdx.y * gridDim.x + blockIdx.x;
      // Word 1: the wave (bits 7:0), the counter's bits (13:8), the crossings seen (15:14) and the block's low
      // 16 bits, which the summary checks against the record's place.
      const uint32_t v[PEARL_AMD_STAMP_WORDS] = {
          0x53540002u,                                                  // "ST", version 2: a written record
          stWave | (PEARL_AMD_STAMP_BITS << 8) | (stSeen << 14) | (block << 16),
          stHw, hw, stCyc0, stRt0, cyc, rt,                             // HW_ID1 and the clocks, start and end
          stPro, stLoop, pearl_amd_cycles_since(stLast, cyc), stBar,    // prologue, loop, epilogue, barrier wait
          stX16, stX19, slabs, stMem};                                  // the crossings, the slabs, memory wait
      uint32_t *o = out + ((size_t)block * (uint32_t)(T / W) + stWave) * PEARL_AMD_STAMP_WORDS;
#pragma unroll
      for (int i = 0; i < (int)PEARL_AMD_STAMP_WORDS; i++) o[i] = v[i];
    }
  }
#endif
#undef PEARL_AMD_LOAD
#undef PEARL_AMD_LOADR
#undef PEARL_AMD_STORER
#undef PEARL_AMD_STORE
#undef PEARL_AMD_STORE_AT
}

}  // namespace

#define PEARL_AMD_FOLD_PARAMS                                                              \
  const int8_t *__restrict__ Ap, const int8_t *__restrict__ Bp, uint32_t k, uint32_t rank, \
      uint32_t col0, uint32_t rows_valid, uint32_t *__restrict__ tr PEARL_AMD_FUSE_PARAMS

extern "C" __global__ __launch_bounds__(PEARL_AMD_THREADS) void pearl_amd_fold_mfma16(
    PEARL_AMD_FOLD_PARAMS) {
#if PEARL_AMD_EMU || PEARL_AMD_MFMA16
  pearl_amd_fold_body<PearlAmdMfma16>(Ap, Bp, k, rank, col0, rows_valid, tr PEARL_AMD_FUSE_ARGS);
#else
  (void)Ap; (void)Bp; (void)k; (void)rank; (void)col0; (void)rows_valid; (void)tr;
#endif
}

extern "C" __global__ __launch_bounds__(PEARL_AMD_THREADS) void pearl_amd_fold_mfma8(
    PEARL_AMD_FOLD_PARAMS) {
#if PEARL_AMD_EMU || PEARL_AMD_MFMA8
  pearl_amd_fold_body<PearlAmdMfma8>(Ap, Bp, k, rank, col0, rows_valid, tr PEARL_AMD_FUSE_ARGS);
#else
  (void)Ap; (void)Bp; (void)k; (void)rank; (void)col0; (void)rows_valid; (void)tr;
#endif
}

extern "C" __global__ __launch_bounds__(PEARL_AMD_THREADS) void pearl_amd_fold_wmma11(
    PEARL_AMD_FOLD_PARAMS) {
#if PEARL_AMD_EMU || PEARL_AMD_WMMA11
  pearl_amd_fold_body<PearlAmdWmma11>(Ap, Bp, k, rank, col0, rows_valid, tr PEARL_AMD_FUSE_ARGS);
#else
  (void)Ap; (void)Bp; (void)k; (void)rank; (void)col0; (void)rows_valid; (void)tr;
#endif
}

#ifdef PEARL_AMD_WAVES
#define PEARL_AMD_WMMA12_OCC __attribute__((amdgpu_waves_per_eu(PEARL_AMD_WAVES)))
#else
#define PEARL_AMD_WMMA12_OCC
#endif
extern "C" __global__ __launch_bounds__(PEARL_AMD_THREADS) PEARL_AMD_WMMA12_OCC void pearl_amd_fold_wmma12(
    PEARL_AMD_FOLD_PARAMS) {
#if PEARL_AMD_EMU || PEARL_AMD_WMMA12
  pearl_amd_fold_body<PearlAmdWmma12>(Ap, Bp, k, rank, col0, rows_valid, tr PEARL_AMD_FUSE_ARGS);
#else
  (void)Ap; (void)Bp; (void)k; (void)rank; (void)col0; (void)rows_valid; (void)tr;
#endif
}

// RDNA2's fold (PearlAmdDot4), in the one-slab loop. RDNA3 and RDNA4 have it only in a build with
// -DPEARL_AMD_PF2=0 (PEARL_AMD_DOT4_FOLD), which runs there RDNA2's loop and readout (DPP on both
// since 12 November), with v_dot4_i32_iu8 for v_dot4_i32_i8 (and the wrap readout on RDNA4).
// PEARL_AMD_DOT4_WAVES=n asks the compiler for n waves a SIMD (amdgpu_waves_per_eu), which caps the
// VGPRs: on gfx103x, 4 waves at 201 VGPRs is 2 blocks a WGP, and 6 (168 VGPRs) would be 3, which the
// 36 KB of LDS a block allows. Unset, the compiler picks (201 VGPRs, 4 waves).
#ifdef PEARL_AMD_DOT4_WAVES
#define PEARL_AMD_DOT4_OCC __attribute__((amdgpu_waves_per_eu(PEARL_AMD_DOT4_WAVES)))
#else
#define PEARL_AMD_DOT4_OCC
#endif
extern "C" __global__ __launch_bounds__(PEARL_AMD_THREADS) PEARL_AMD_DOT4_OCC void pearl_amd_fold_dot4(
    PEARL_AMD_FOLD_PARAMS) {
#if PEARL_AMD_DOT4_FOLD
  pearl_amd_fold_body<PearlAmdDot4>(Ap, Bp, k, rank, col0, rows_valid, tr PEARL_AMD_FUSE_ARGS);
#else
  (void)Ap; (void)Bp; (void)k; (void)rank; (void)col0; (void)rows_valid; (void)tr;
#endif
}

// The reference fold: one 32x64 block of C, eight regions, a block. Thread t works on region
// t / 32 of the block and holds 8 of its 256 sums: one row, eight columns. Each chunk's
// rows of A' and B' go through LDS; the sums are dot products of 4 bytes at a time. Slow
// next to the matrix folds (each sum reads its A' and B' words from LDS), but it needs nothing
// from the card, so it is the baseline to compare against on hardware (PEARL_AMD_FOLD=ref).
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
#if PEARL_AMD_KWRAP
    keep = p == (c & 15u) ? pearl_rotl13(keep) ^ x : keep;   // chunk c into word c % 16 (PEARL_AMD_KWRAP)
#else
    keep = p == c ? x : keep;
#endif
  }
  if (p < PEARL_JACKPOT_BUCKETS) {
    const uint32_t rowIdx = 2u * blockIdx.y + (q >> 2);
    const uint32_t colIdx = 4u * blockIdx.x + (q & 3u);
    tr[((size_t)colIdx * rows_valid + rowIdx) * PEARL_JACKPOT_BUCKETS + p] = keep;
  }
}

// The transcript hash after a fold: one thread a region of the batch (pearl_amd_hash_region). A
// PEARL_AMD_FUSE build runs it only after the reference fold.
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
  pearl_amd_hash_region(tm, i, test, hits);
}

// Which folds this code object has bodies for, and its wave size: out[0] and out[1]. A
// PEARL_AMD_READOUT_DROP build says so on every target, the ones its drop leaves alone too, so
// the host refuses to mine with it on any card. A PEARL_AMD_ABLATE build says so the same way,
// with PEARL_AMD_CAP_TIMING, and a PEARL_AMD_STAMP build with that and PEARL_AMD_CAP_STAMP, which
// also tells the host to give the folds somewhere to store their stamps. A code object with the
// wrap readout says so too: it is how the host knows a card can mine k 4096 (pearl_host_card_k).
// And one whose PF2 loop has PEARL_AMD_CRANK (RDNA 4's, by default) says its matrix fold takes
// rank PEARL_FOLD_RANK only: the host pass never sees the gfx12 default, so the macro there would
// say nothing about the card. The same goes for one whose folds hash their own transcripts
// (PEARL_AMD_FUSE): the host then launches no hash kernel after them.
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
  if (PEARL_AMD_DOT4_FOLD) out[0] |= PEARL_AMD_CAP_DOT4;
  if (PEARL_AMD_READOUT_DROP) out[0] |= PEARL_AMD_CAP_DROP;
  if (PEARL_AMD_CRANK && PEARL_AMD_PF2) out[0] |= PEARL_AMD_CAP_RANK;
  if (PEARL_AMD_FUSE) out[0] |= PEARL_AMD_CAP_FUSE;
  if (PEARL_AMD_KWRAP) out[0] |= PEARL_AMD_CAP_KWRAP;
#if PEARL_AMD_ABLATE
  out[0] |= PEARL_AMD_CAP_TIMING;
#endif
#if PEARL_AMD_STAMP
  out[0] |= PEARL_AMD_CAP_TIMING | PEARL_AMD_CAP_STAMP;
#endif
  out[1] = (uint32_t)warpSize;
}
