// pearl_sycl_peak: what an Intel GPU's XMX matrix engines do with little or
// nothing in the way, and where the fold's time goes between that and the rate
// it mines at. The AMD port's pearl_amd_peak asks the same of WMMA.
//
//   pearl_sycl_peak [--secs S] [--only LIST] [--reps R] [--trace]
//                   [--m M] [--n N] [--col-batch C] [--band B] [--sg 8|16] [--rb 2]
//   pearl_sycl_peak --frag [--a-stride BYTES] [--b-stride BYTES] [--lane-stride BYTES]
//                   [--wpg WINDOWS] [--row-mask M] [--col-mask M] [--k-mask M]
//                   [--stagger N] [--a-mix N] [--b-mix N] [--fill-skip 1]
//                   [--pfa D] [--pfb D] [--pf-hint H] [--touch-a D] [--touch-b D] [--spin N]
//                   and the options above
//
// The second line of --frag switches came with the rows that read by
// sub-group block reads (f-k4..., g-k4...): what a work-group that fills an
// Xe-core shares (--wpg 32 with --stagger, PF 12's chunk fill), what a few
// windows reading elsewhere cost (--a-mix, --b-mix), and prefetches, one-byte
// loads ahead and work in registers (the PF 11 rows only). PeakArgs says what
// each does; README.md, "What was timed toward 100", what they showed.
//
// The image rows (i-..., j-...), where part of an operand comes through the
// sampler, are described above PeakImg.
//
// --frag: the rows of the fragment fold (fold_xmxf, what an Arc A-series card
// mines with since 11 November 2026) in place of the rows below, which are the
// first fold's. They are listed above PeakFrag and in add_frag_rows; README.md,
// "Where its time goes now", has what they showed. A build with --jit --peak
// has them: the 256-register mode is a kernel property there.
//
// Every row is the fold's own grid and thread shape: a work-group of 32 a
// window (4 sub-groups of 8 on Alchemist, 2 of 16 on Battlemage and newer), 8
// DPAS accumulators a sub-group, 64 k-steps of 32 a window. What changes from
// row to row is what feeds the DPAS and what follows it.
//
// The matrix engines alone:
//   regs        operands in registers, two A sets taking turns. Nothing else in
//               the loop. A thread runs --reps windows' worth of k-steps (16),
//               so starting threads costs next to nothing. The peak.
//   regs-half, regs-quarter, regs-one
//               regs with a half, a quarter or one of the accumulators in use:
//               does a DPAS wait for the one before it on the same accumulator?
//   regs-short  regs with one window a thread, as the fold has it: what
//               starting 16 times as many threads costs.
//   regs-rd     regs-short plus the fold's readout: the XOR and lane shuffles
//               at each chunk, the barrier, the hash and the test.
//
// The fold's load instructions, from memory that stays in the cache:
//   fixed       every load is the same fragment.
//   window0     every window reads window 0 (A' 64 KB and B' 128 KB at k 2048).
//   a-win0, a-res16, a-res256, a-res4096
//               A' only (B in registers), from 1, 16, 256 or 4096 row windows
//               in turn: 64 KB, 1 MB, 16 MB, 256 MB.
//   b-win0, b-res4, b-res16, b-res64
//               B' only (A in registers), from 1, 4, 16 or 64 column windows
//               in turn: 128 KB, 512 KB, 2 MB, 8 MB.
//
// The fold's loads from its own addresses, in its own order:
//   load-a      A' as the fold loads it, B in registers.
//   load-b      B' as the fold loads it, A in registers.
//   own-a       A' as the fold loads it, B' from window 0.
//   own-b       B' as the fold loads it, A' from window 0.
//   loads       both. No readout.
//   loads-1dpas loads with one DPAS a k-step and the other operands
//               XORed into a register: nearly the loads alone. Its figure is
//               the rate those loads would allow, not work done.
//   loads-xor   loads plus the chunk XOR, lane shuffles and local store.
//   fold-copy   and the barrier, hash and test: this file's copy of the fold.
//   fold        the kernel the core mines with (pearl_sycl_kernels.hpp). It
//               should match fold-copy, or the rows above describe another fold.
//
// What a change to the operands' layout could give. These are the fold's own
// load instructions, reading the same number of bytes from the same windows in
// the same order, at the addresses the bytes would have if A' and B' were
// stored in the order DPAS takes them: the 8 lanes' 32 bytes of a B fragment
// side by side (4 cache lines, where the fold's 8 columns are 8 rows of B'
// 2048 bytes apart), and a sub-group's 8 rows of A the same way.
//   frag-a      A that way, B in registers.
//   frag-b      B that way, A in registers.
//   frag        both. No readout.
//   frag-fold   and the readout, barrier, hash and test.
// And with one load a fragment: a lane loads its 8 dwords of a fragment as one
// vector, where the fold loads a B fragment in 2 and an A fragment in 8 (A's
// 8 rows are 2048 bytes apart, so this needs A in fragment order, each lane's
// 8 rows together):
//   vec-a       A that way, B in registers.
//   vec-b       B' that way at the fold's own addresses, A in registers.
//   vec         both, B' at the fold's own addresses.
//   vec-fold    and the readout, barrier, hash and test.
//   vecfrag, vecfrag-fold
//               the same with B' in fragment order too.
// Their bits are not a fold's: no row here is checked, and these could not be.
//
// --rb 2 (a build with --large-grf --peak, on Alchemist): the same rows with
// 16 rows a sub-group instead of 8, so 16 accumulators, 2 sub-groups a window,
// and one load of B' for twice the DPAS. It needs the 256-register mode, which
// leaves an EU 4 threads instead of 8.
//
// On an Arc A750 (README.md, "The A750's ceiling"): regs 113 T-MAC/s, the fold
// 16.9, and vecfrag-fold with --rb 2 at --band 32, the best row shaped like a
// fold, 42.7.
//
// T-MAC/s is int8 multiply-accumulates a second / 1e12, the unit the pool's
// TH/s is in: a DPAS is 8 rows by a sub-group's columns by 32 k. Clock, watts
// and fan are the kernel driver's counters for the card (i915 or xe, through
// sysfs), read while the row runs; a card without them prints none. i915 has
// no temperature for Arc on kernel 6.12, so none is printed there.
// MAC/EU/clk is the rate over the card's EUs and that clock.
//
// It checks nothing. It is for timing; verify.sh checks the folds.
// PEARL_GPU_INDEX picks the card (default 0), PEARL_SYCL_DEVICE=cpu the SYCL
// CPU device, where DPAS is the reference code and the figures mean nothing:
// that is for seeing that it builds and runs.

#include <sycl/sycl.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "pearl_sycl_kernels.hpp"

// A fragment as one lane holds it, at a byte address: the 8 dwords of B, or the
// 8 rows of A (a dword each at sub-group 8, a word each at 16).
typedef int peak_v8i __attribute__((ext_vector_type(8), aligned(4)));
typedef short peak_v8s __attribute__((ext_vector_type(8), aligned(2)));
// Half a fragment, and a fragment as four qwords: other forms of the same load.
typedef int peak_v4i __attribute__((ext_vector_type(4), aligned(4)));
typedef long long peak_v4q __attribute__((ext_vector_type(4), aligned(4)));
typedef long long peak_v8q __attribute__((ext_vector_type(8), aligned(4)));
// What a sub-group block read returns to a lane: 4 dwords (a message of 128 bytes at sub-group 8) or 4 qwords
// (256 bytes).
typedef uint32_t peak_b4u __attribute__((ext_vector_type(4)));
typedef uint64_t peak_b4q __attribute__((ext_vector_type(4)));

// A sub-group block read (SPV_INTEL_subgroups): one message with one address for the whole sub-group, where a
// vector load is a gather with an address a lane. Lane l's component c is word c * 8 + l at p.
// A prefetch with one of the extension's eight hints: cache level 1 to 4, then the same four non-temporal.
inline void peak_prefetch(const int8_t *p, uint32_t hint) {
  namespace sx = sycl::ext::oneapi::experimental;
  switch (hint) {
    case 0: sx::prefetch(p, sx::properties{sx::prefetch_hint_L1}); break;
    case 1: sx::prefetch(p, sx::properties{sx::prefetch_hint_L2}); break;
    case 2: sx::prefetch(p, sx::properties{sx::prefetch_hint_L3}); break;
    case 3: sx::prefetch(p, sx::properties{sx::prefetch_hint_L4}); break;
    case 4: sx::prefetch(p, sx::properties{sx::prefetch_hint_L1_nt}); break;
    case 5: sx::prefetch(p, sx::properties{sx::prefetch_hint_L2_nt}); break;
    case 6: sx::prefetch(p, sx::properties{sx::prefetch_hint_L3_nt}); break;
    default: sx::prefetch(p, sx::properties{sx::prefetch_hint_L4_nt}); break;
  }
}

template <typename V, typename E> inline V peak_block_read(const int8_t *p) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT)
  return __spirv_SubgroupBlockReadINTEL<V>((const __attribute__((opencl_global)) E *)p);
#else
  // A prebuilt probe has an image for the CPU device, which has no block read:
  // there a plain load of the same size stands in, and the row is not a block
  // read's. Build with --jit --peak for these rows.
  V v;
  __builtin_memcpy(&v, p, sizeof(V));
  return v;
#endif
}

namespace {

using pearl_sycl::AFrag;
using pearl_sycl::FoldArgs;
using pearl_sycl::HitBuf;

struct PeakArgs {
  FoldArgs f;        // the fold's own arguments: operands, grid, band, test, hits
  uint32_t reps;     // windows' worth of k-steps a thread runs
  uint32_t seed;     // the register operands' bits
  uint32_t rowMask;  // "masked" A loads: the row windows read (window & mask)
  uint32_t colMask;  // "masked" B loads: the column windows read
  uint32_t kMask;    // "masked" loads: the k offsets read (0: one fragment)
  uint32_t zero;     // 0, which the compiler cannot know: makes one value wait for another
  uint32_t laneStride;  // fragment rows: bytes from one lane's 32 of a fragment to the next lane's (32: packed)
  uint32_t wpg;         // fragment rows: windows a work-group (1: the fold's; a band's: every sub-group of a
                        // work-group reads the same B')
  uint32_t stagger;     // fragment rows: 0, or N: a sub-group reads the k-step (i + j) % N ahead of the one it
                        // is at, i its block of A' and j its column window in the work-group. Stands in for
                        // sub-groups that share an operand starting a k-step apart: a sharer then finds a
                        // fragment another loaded a k-step ago
  uint32_t aMix;        // fragment rows: 0, or N: only one window in N reads its own A', the others window 0's
  uint32_t bMix;        // the same for B'
  uint32_t spin;        // fragment rows: extra adds on registers a k-step, to price work done in registers
  uint32_t pfHint;      // which of the prefetch extension's hints (peak_prefetch)
  uint32_t fillSkip;    // PF 12: 1 leaves the chunk's fill reads out
  uint32_t touchA, touchB;  // fragment rows: 0, or D + 1: each k-step loads one byte of each cache line of the
                        // A' (B') of the k-step D ahead, 8 lines a message, and keeps it: a load, not a
                        // prefetch, so the lines come into the first cache level
  uint32_t pfA, pfB;    // fragment rows: 0, or D: each k-step sends prefetches for the A' (B') of the k-step D
                        // ahead, 8 cache lines a message (one a lane), so its loads find them in the cache
  uint32_t sga, sgb;    // "s-" rows: a work-group is sga 16-row blocks of A' by sgb 64-column blocks of B'
  uint32_t sgp;         // producer rows: sub-groups a work-group that only copy the ring
  uint32_t *sink;    // one word a sub-group, so no accumulator can be dropped
};

inline uint32_t mix(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352du;
  x ^= x >> 15; x *= 0x846ca68bu;
  return x ^ (x >> 16);
}
// Four int8 in [-64, 63]: bit 6 of each byte copied into bit 7.
inline uint32_t bytes7(uint32_t x) {
  const uint32_t t = x & 0x7f7f7f7fu;
  return t | ((t & 0x40404040u) << 1);
}

template <int SG> inline v8i widen(typename AFrag<SG>::type a) {
  if constexpr (SG == 16) return __builtin_convertvector(a, v8i);
  else return a;
}

// One row's kernel. LA and LB say where A and B come from:
//   0 registers
//   1 the fold's loads from masked windows and k offsets (PeakArgs's masks)
//   2 the fold's loads from its own addresses
//   3 the fold's loads at fragment-order addresses (a fragment's bytes together)
//   4 one vector load a lane and fragment: B at the fold's own address, A at a
//     fragment-order address with each lane's 8 rows together
//   5 (B only) one vector load a lane at the fragment-order address
// RD: 0 no readout, 1 the chunk XOR, lane shuffles and local store, 2 and the
// barrier, hash and test. NA: accumulators that get a DPAS, of 8 x RB x 8 / SG;
// the operands of the others are XORed into a register so their loads stay.
// Modes 1 and 2 and the readout are pearl_sycl::fold_xmx's, line for line.
template <int SG, int RB, int LA, int LB, int RD, int NA>
inline sycl::event peak_step(sycl::queue &q, const PeakArgs &args, uint32_t groups) {
  constexpr int S = 4 / RB;
  constexpr int NCB = 64 / SG;
  constexpr int WG = S * SG;
  constexpr int BG = SG == 16 ? 4 : 2;
  typedef typename AFrag<SG>::type afrag;
  return q.submit([&](sycl::handler &h) {
    sycl::local_accessor<uint32_t, 1> part(S * 8 * 16, h);
    const PeakArgs P = args;
    h.parallel_for(sycl::nd_range<1>((size_t)groups * WG, WG),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
      const FoldArgs &A = P.f;
      sycl::sub_group sg = it.get_sub_group();
      const uint32_t lane = (uint32_t)sg.get_local_linear_id();
      const uint32_t sgi = (uint32_t)sg.get_group_linear_id();
      const uint32_t g = (uint32_t)it.get_group(0);
      const uint32_t rowWins = A.rowsValid / 2u, colWins = A.colBatch / 4u;
      uint32_t wr, wc;
      pearl_sycl::window_of(g, rowWins, colWins, A.band, wr, wc);
      const uint32_t k = A.k;
      const uint32_t awr = LA == 1 ? (wr & P.rowMask) : wr;
      const uint32_t bcol = LB == 1 ? 64u * (wc & P.colMask) : 16u * A.colStart + 64u * wc;
      // The sub-group's rows of A' and, for modes 1 and 2, the lane's column of B'.
      const int8_t *Ab = A.Ap + (size_t)(32u * awr + sgi * 8u * RB) * k;
      const int8_t *Bb = A.Bp + (size_t)(bcol + lane) * k;
      const int8_t *Bw = A.Bp + (size_t)bcol * k;  // mode 3: the column window
      afrag ar[2][RB];
      v8i br[BG];
      if constexpr (LA == 0) {
        for (int set = 0; set < 2; set++)
          for (int rb = 0; rb < RB; rb++)
            for (int r = 0; r < 8; r++) {
              const uint32_t w = bytes7(mix(P.seed + lane * 131u + sgi * 17u + (uint32_t)(set * 64 + rb * 8 + r)));
              if constexpr (SG == 16) ar[set][rb][r] = (short)w;
              else ar[set][rb][r] = (int)w;
            }
      }
      if constexpr (LB == 0) {
        for (int i = 0; i < BG; i++)
          for (int d = 0; d < 8; d++)
            br[i][d] = (int)bytes7(mix((P.seed ^ 0xB5297A4Du) + lane * 257u + (uint32_t)(i * 8 + d)));
      }
      v8i c[RB][NCB];
      for (int rb = 0; rb < RB; rb++)
        for (int cb = 0; cb < NCB; cb++) c[rb][cb] = 0;
      v8i keep = 0;
      for (uint32_t rep = 0; rep < P.reps; rep++) {
        for (uint32_t ch = 0; ch < 16u; ch++) {
          for (uint32_t s = 0; s < 4u; s++) {
            const uint32_t k0 = ch * 128u + s * 32u;
            const uint32_t ka = LA == 1 ? (k0 & P.kMask) : k0, kb = LB == 1 ? (k0 & P.kMask) : k0;
            afrag af[RB];
            for (int rb = 0; rb < RB; rb++) {
              if constexpr (LA == 0) {
                af[rb] = ar[s & 1u][rb];
              } else if constexpr (LA == 3) {
                // The sub-group's 8 x RB rows by 32 k of this step, stored
                // together: 32 bytes a row, 256 bytes a fragment.
                const int8_t *p = Ab + (size_t)(k0 / 32u) * (256u * RB) + (size_t)rb * 256u;
                for (int r = 0; r < 8; r++) {
                  if constexpr (SG == 16)
                    af[rb][r] = *reinterpret_cast<const short *>(p + 32 * r + 2u * lane);
                  else
                    af[rb][r] = *reinterpret_cast<const int *>(p + 32 * r + 4u * lane);
                }
              } else if constexpr (LA == 4) {
                // The same 256 bytes with each lane's 8 rows together, loaded
                // as one vector: one load a fragment where the fold has 8.
                const int8_t *p = Ab + (size_t)(k0 / 32u) * (256u * RB) + (size_t)rb * 256u;
                if constexpr (SG == 16)
                  af[rb] = *reinterpret_cast<const peak_v8s *>(p + 16u * lane);
                else
                  af[rb] = *reinterpret_cast<const peak_v8i *>(p + 32u * lane);
              } else {
                for (int r = 0; r < 8; r++) {
                  const int8_t *p = Ab + (size_t)(8 * rb + r) * k + ka;
                  if constexpr (SG == 16)
                    af[rb][r] = *reinterpret_cast<const short *>(p + 2u * lane);
                  else
                    af[rb][r] = *reinterpret_cast<const int *>(p + 4u * lane);
                }
              }
            }
            for (int gb = 0; gb < NCB / BG; gb++) {
              v8i bf[BG];
              for (int i = 0; i < BG; i++) {
                if constexpr (LB == 0) {
                  bf[i] = br[i];
                } else if constexpr (LB == 3) {
                  // The window's 64 columns by 32 k of this step, stored
                  // together: a fragment is its SG lanes' 32 bytes side by side.
                  const uint32_t *p = reinterpret_cast<const uint32_t *>(
                      Bw + (size_t)(k0 / 32u) * 2048u + (size_t)(BG * gb + i) * (32u * SG) + 32u * lane);
                  for (int d = 0; d < 8; d++) bf[i][d] = (int)p[d];
                } else if constexpr (LB == 4) {
                  // The lane's 32 bytes at the fold's own address, loaded as
                  // one vector: one load a fragment where the fold has 2.
                  bf[i] = *reinterpret_cast<const peak_v8i *>(Bb + (size_t)(SG * (BG * gb + i)) * k + kb);
                } else if constexpr (LB == 5) {
                  // One vector a lane at the fragment-order address.
                  bf[i] = *reinterpret_cast<const peak_v8i *>(
                      Bw + (size_t)(k0 / 32u) * 2048u + (size_t)(BG * gb + i) * (32u * SG) + 32u * lane);
                } else {
                  const uint32_t *p = reinterpret_cast<const uint32_t *>(
                      Bb + (size_t)(SG * (BG * gb + i)) * k + kb);
                  for (int d = 0; d < 8; d++) bf[i][d] = (int)p[d];
                }
              }
              for (int rb = 0; rb < RB; rb++)
                for (int i = 0; i < BG; i++) {
                  const int cb = BG * gb + i;
                  if (rb * NCB + cb < NA) c[rb][cb] = pearl_sycl::dpas<SG, true>(sg, af[rb], bf[i], c[rb][cb]);
                }
              // A fragment no DPAS took, once, so its load stays.
              for (int i = 0; i < BG; i++)
                if (BG * gb + i >= NA) keep ^= bf[i];
            }
            for (int rb = 0; rb < RB; rb++)
              if (rb * NCB >= NA) keep ^= widen<SG>(af[rb]);
          }
          if constexpr (RD >= 1) {
            uint32_t x0 = 0, x1 = 0;
            for (int rb = 0; rb < RB; rb++)
              for (int cb = 0; cb < NCB; cb++) {
                for (int r = 0; r < 4; r++) x0 ^= (uint32_t)c[rb][cb][r];
                for (int r = 4; r < 8; r++) x1 ^= (uint32_t)c[rb][cb][r];
              }
            x0 ^= sycl::permute_group_by_xor(sg, x0, 1u);
            x1 ^= sycl::permute_group_by_xor(sg, x1, 1u);
            if constexpr (SG == 16) {
              x0 ^= sycl::permute_group_by_xor(sg, x0, 8u);
              x1 ^= sycl::permute_group_by_xor(sg, x1, 8u);
            }
            if (lane < 8u) part[(sgi * 8u + lane) * 16u + ch] = (lane & 1u) ? x1 : x0;
          }
        }
      }
      if constexpr (RD == 2) {
        sycl::group_barrier(it.get_group());
        if (sgi == 0 && lane < 8u) {
          const uint32_t qr = lane;
          uint32_t words[16];
          for (int j = 0; j < 16; j++) {
            uint32_t x = 0;
            for (int s = 0; s < S; s++) x ^= part[((uint32_t)s * 8u + qr) * 16u + (uint32_t)j];
            words[j] = x;
          }
          const uint32_t local = (2u * wr + (qr & 1u)) + A.rowsValid * (4u * wc + (qr >> 1));
          pearl_sycl::region_epilogue(words, local, A.test, A.hits, A.allTr);
        }
      } else {
        // Every accumulator into one word a sub-group, stored: the rows without
        // the fold's own readout keep their DPAS and their loads this way.
        uint32_t x = 0;
        for (int rb = 0; rb < RB; rb++)
          for (int cb = 0; cb < NCB; cb++)
            for (int r = 0; r < 8; r++) x ^= (uint32_t)c[rb][cb][r];
        for (int r = 0; r < 8; r++) x ^= (uint32_t)keep[r];
        x = sycl::reduce_over_group(sg, x, sycl::bit_xor<uint32_t>());
        if (lane == 0) P.sink[(size_t)g * S + sgi] = x;
      }
    });
  });
}

// The fragment fold's rows (--frag): pearl_sycl::FoldXmxF's grid, sub-group 8
// in the 256-register mode, 16 accumulators a sub-group, its operands at
// fragment-order addresses.
//   RB, NCB the sub-group's tile, in 8-row and 8-column blocks: 2 by 8 is the
//           fold's 16 rows by 64 columns (a window is two sub-groups, split by
//           rows), 4 by 4 is 32 rows by 32 columns (split by columns). A
//           k-step is 16 DPAS either way; it loads 2 fragments of A' and 8 of
//           B', or 4 and 4.
//   LA, LB  where an operand's fragment comes from:
//           0 registers
//           1 one vector of 8 dwords a lane (the fold's load)
//           2 two vectors of 4 dwords
//           6 one vector of 4 qwords, kept a qword load by adding a zero the
//             compiler cannot see, then taken as 8 dwords
//           7 two sub-group block reads of 4 dwords a lane: 128 bytes a
//             message, one address for the sub-group
//           8 one sub-group block read of 4 qwords a lane: 256 bytes
//   RD      0 no readout, 2 the fold's readout, barrier, hash and test
//   PF      how a k-step's loads and DPAS are ordered:
//           0 as the fold has them: A', then each B' fragment before its DPAS
//           3 every load first, and every DPAS made to wait for all of them
//             (a word of each fragment, ANDed with a zero the compiler cannot
//             see, is ORed into A), so the 16 DPAS run as one unbroken run
//           4 the loads are sent and kept by an XOR, and the DPAS take
//             register operands: loads and DPAS side by side, no load's
//             result going into a DPAS
//           5 no loads: every operand is written by an add before its k-step
//           6 as 3 in two halves: A' and the first half of B', their DPAS,
//             then the second half of B' and its DPAS. No more than 12
//             loads are in flight (a thread has 16 scoreboard tokens)
//           7 6 with each half's loads sent before the other half's DPAS, so
//             they are in flight while it runs (A' is held twice)
//           8 every fragment as one block read of 256 bytes (4 qwords a
//             lane), all sent first; each is taken as 8 dwords only once all
//             have landed. LA and LB choose which operands are loaded so:
//             8 the 256-byte read, 7 two reads of 128 bytes, 0 registers
//           10 3 with B' sent before A'
//           11 3 with the probe's prefetch, touch and spin switches in the loop
//           12 3 for a work-group that fills an Xe-core (--wpg 32 --band 8: 64 sub-groups,
//             16 blocks of A' by 4 column windows, or 8 by 8 for the 32 by 32 tile), a
//             chunk at a time: at each chunk's start a sub-group reads its 1/64 of
//             the chunk's cache lines (8 block reads, kept), the group meets at a
//             barrier, and the chunk's 4 k-steps then find every line in the
//             first cache level. --fill-skip 1 leaves the reads out (the barrier
//             alone)
//           9 block reads only kept: one word of each goes into an XOR and
//             the DPAS take register operands. What the messages alone
//             cost, in either size (LA, LB: 8 or 7)
// The masks cut the windows and k-steps read down to a set that stays in a
// cache; aStride and bStride (FoldArgs) space the blocks, laneStride the
// lanes' 32 bytes of a fragment.
template <int RB, int NCB, int LA, int LB, int RD, int PF> struct PeakFrag {
  static constexpr int S = (4 / RB) * (8 / NCB);  // sub-groups a window
  PeakArgs P;
  sycl::local_accessor<uint32_t, 1> part;

  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    namespace syclex = sycl::ext::oneapi::experimental;
    return syclex::properties{syclex::sub_group_size<8>, sycl::ext::intel::experimental::grf_size<256>};
  }

  void operator()(sycl::nd_item<1> it) const {
    const FoldArgs &A = P.f;
    sycl::sub_group sg = it.get_sub_group();
    const uint32_t lane = (uint32_t)sg.get_local_linear_id();
    // A work-group is wpg windows in the walk's order, S sub-groups each.
    const uint32_t sgl = (uint32_t)sg.get_group_linear_id();
    const uint32_t sgi = sgl % (uint32_t)S;
    const uint32_t widx = sgl / (uint32_t)S;
    const uint32_t g = (uint32_t)it.get_group(0) * P.wpg + widx;
    const uint32_t rot = P.stagger ? ((widx % A.band) * (uint32_t)S + sgi + widx / A.band) % P.stagger : 0u;
    const uint32_t rowWins = A.rowsValid / 2u, colWins = A.colBatch / 4u;
    uint32_t wr, wc;
    pearl_sycl::window_of(g, rowWins, colWins, A.band, wr, wc);
    // The sub-group's first row and first column block of the window.
    const uint32_t wra = (P.aMix && (g % P.aMix)) ? 0u : (wr & P.rowMask);
    const uint32_t wcb = (P.bMix && (g % P.bMix)) ? 0u : ((A.colStart / 4u + wc) & P.colMask);
    const uint32_t row0 = 32u * wra + (RB == 2 ? sgi * 16u : 0u);
    const uint32_t cb0 = NCB == 8 ? 0u : sgi * 4u;
    // A fragment is 8 lanes of laneStride bytes: 256 bytes when packed.
    const size_t fragBytes = 8u * (size_t)P.laneStride;
    // The sub-group's block, which a block read addresses, and this lane's bytes in it.
    const int8_t *Au = A.Ap + (size_t)(row0 >> 4) * A.aStride;
    const int8_t *Bu = A.Bp + (size_t)wcb * A.bStride + (size_t)cb0 * fragBytes;
    const size_t laneOff = (size_t)P.laneStride * lane;
    v8i ar[2][RB], br[2];
    if constexpr (LA == 0 || PF >= 4) {
      for (int set = 0; set < 2; set++)
        for (int rb = 0; rb < RB; rb++)
          for (int r = 0; r < 8; r++)
            ar[set][rb][r] = (int)bytes7(mix(P.seed + lane * 131u + sgi * 17u + (uint32_t)(set * 64 + rb * 8 + r)));
    }
    if constexpr (LB == 0 || PF >= 4) {
      for (int i = 0; i < 2; i++)
        for (int d = 0; d < 8; d++)
          br[i][d] = (int)bytes7(mix((P.seed ^ 0xB5297A4Du) + lane * 257u + (uint32_t)(i * 8 + d)));
    }
    // A lane's 32 bytes of a fragment at p, loaded in form F.
    auto load_frag = [&](const int8_t *u, int F) -> v8i {
      if (F == 7) {
        const peak_b4u lo = peak_block_read<peak_b4u, uint32_t>(u);
        const peak_b4u hi = peak_block_read<peak_b4u, uint32_t>(u + 128);
        return (v8i)__builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
      } else if (F == 8) {
        return (v8i)peak_block_read<peak_b4q, uint64_t>(u);
      }
      const int8_t *p = u + laneOff;
      if (F == 2) {
        const peak_v4i lo = *reinterpret_cast<const peak_v4i *>(p);
        const peak_v4i hi = *reinterpret_cast<const peak_v4i *>(p + 16);
        return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
      } else if (F == 6) {
        const peak_v4q q = *reinterpret_cast<const peak_v4q *>(p) + (long long)P.zero;
        return (v8i)q;
      }
      return *reinterpret_cast<const v8iu *>(p);
    };
    auto load_a = [&](uint32_t ks, int rb) -> v8i {
      if constexpr (LA == 0) return ar[ks & 1u][rb];
      else return load_frag(Au + ((size_t)(((ks + rot) & 63u) & P.kMask) * RB + (size_t)rb) * fragBytes, LA);
    };
    auto load_b = [&](uint32_t ks, int cb) -> v8i {
      if constexpr (LB == 0) return br[(ks + (uint32_t)cb) & 1u];
      else return load_frag(Bu + ((size_t)(((ks + rot) & 63u) & P.kMask) * 8u + (size_t)cb) * fragBytes, LB);
    };
    v8i c[RB][NCB];
    for (int rb = 0; rb < RB; rb++)
      for (int cb = 0; cb < NCB; cb++) c[rb][cb] = 0;
    v8i keep = 0;
    v8i af[RB], bf[NCB];
    // A word of each of n fragments, ANDed with the unseen zero: ORed into an
    // operand, it makes whatever reads that operand wait for all n loads.
    auto gate = [&](const v8i *a, int na, const v8i *b, int nb) -> int {
      int t = 0;
      for (int i = 0; i < na; i++) t ^= a[i][0];
      for (int i = 0; i < nb; i++) t ^= b[i][0];
      return t & (int)P.zero;
    };
    constexpr int H = NCB / 2;
    if constexpr (PF == 7) {
      for (int rb = 0; rb < RB; rb++) af[rb] = load_a(0, rb);
      for (int cb = 0; cb < H; cb++) bf[cb] = load_b(0, cb);
    }
    for (uint32_t rep = 0; rep < P.reps; rep++) {
      for (uint32_t ch = 0; ch < 16u; ch++) {
        for (uint32_t s = 0; s < 4u; s++) {
          const uint32_t ks = ch * 4u + s;
          // Prefetches: a message is 8 cache lines, one a lane, and brings nothing back.
          if constexpr (PF == 11) {
          if (P.pfA) {
            const uint32_t kp = ((ks + rot + P.pfA) & 63u) & P.kMask;
            for (int i = 0; i < RB / 2; i++)
              peak_prefetch(Au + ((size_t)kp * RB + 2u * (size_t)i) * fragBytes + 64u * lane, P.pfHint);
          }
          if (P.pfB) {
            const uint32_t kp = ((ks + rot + P.pfB) & 63u) & P.kMask;
            for (int i = 0; i < NCB / 2; i++)
              peak_prefetch(Bu + ((size_t)kp * 8u + 2u * (size_t)i) * fragBytes + 64u * lane, P.pfHint);
          }
          if (P.touchA) {
            const uint32_t kp = ((ks + rot + P.touchA - 1u) & 63u) & P.kMask;
            for (int i = 0; i < RB / 2; i++)
              keep[1] ^= (int)*reinterpret_cast<const uint8_t *>(Au + ((size_t)kp * RB + 2u * (size_t)i) * fragBytes + 64u * lane);
          }
          if (P.touchB) {
            const uint32_t kp = ((ks + rot + P.touchB - 1u) & 63u) & P.kMask;
            for (int i = 0; i < NCB / 2; i++)
              keep[2] ^= (int)*reinterpret_cast<const uint8_t *>(Bu + ((size_t)kp * 8u + 2u * (size_t)i) * fragBytes + 64u * lane);
          }
          // Work in registers: `spin` adds that depend on each other, kept by the sink.
          for (uint32_t i = 0; i < P.spin; i++) keep += keep + (int)(ks + i);
          }
          if constexpr (PF == 0) {
            for (int rb = 0; rb < RB; rb++) af[rb] = load_a(ks, rb);
            for (int cb = 0; cb < NCB; cb++) {
              const v8i b = load_b(ks, cb);
              for (int rb = 0; rb < RB; rb++) c[rb][cb] = pearl_sycl::dpas<8, true>(sg, af[rb], b, c[rb][cb]);
            }
          } else if constexpr (PF == 3 || PF == 10 || PF == 11 || PF == 12) {
            if constexpr (PF == 12) {
              if (s == 0u) {
                // The sub-group's block of A' and of B' among the work-group's, and how many share each.
                const uint32_t ia = RB == 2 ? (widx % A.band) * 2u + sgi : widx % A.band;
                const uint32_t ib = RB == 2 ? widx / A.band : (widx / A.band) * 2u + sgi;
                if (!P.fillSkip) {
                  const uint32_t k0 = (ks & 63u) & P.kMask;
                  const size_t runB = (size_t)NCB * fragBytes;
                  const size_t pb = (size_t)ia * 512u;
                  const int8_t *fa = Au + (size_t)k0 * RB * fragBytes + (size_t)ib * 512u;
                  const int8_t *fb = Bu + ((size_t)k0 + pb / runB) * 8u * fragBytes + pb % runB;
                  uint32_t t = 0;
                  for (int m = 0; m < 4; m++) {
                    t ^= peak_block_read<peak_b4u, uint32_t>(fa + 128 * m)[0];
                    t ^= peak_block_read<peak_b4u, uint32_t>(fb + 128 * m)[0];
                  }
                  keep[3] ^= (int)t;
                }
                sycl::group_barrier(it.get_group());
              }
            }
            // 10: B' sent before A', so the loads that wait for a free token are A's.
            if constexpr (PF == 10) {
              for (int cb = 0; cb < NCB; cb++) bf[cb] = load_b(ks, cb);
              for (int rb = 0; rb < RB; rb++) af[rb] = load_a(ks, rb);
            } else {
              for (int rb = 0; rb < RB; rb++) af[rb] = load_a(ks, rb);
              for (int cb = 0; cb < NCB; cb++) bf[cb] = load_b(ks, cb);
            }
            int t = 0;
            for (int rb = 0; rb < RB; rb++) t ^= af[rb][0];
            for (int cb = 0; cb < NCB; cb++) t ^= bf[cb][0];
            t &= (int)P.zero;
            for (int rb = 0; rb < RB; rb++) af[rb][0] |= t;
            for (int cb = 0; cb < NCB; cb++)
              for (int rb = 0; rb < RB; rb++) c[rb][cb] = pearl_sycl::dpas<8, true>(sg, af[rb], bf[cb], c[rb][cb]);
          } else if constexpr (PF == 6) {
            for (int rb = 0; rb < RB; rb++) af[rb] = load_a(ks, rb);
            for (int cb = 0; cb < H; cb++) bf[cb] = load_b(ks, cb);
            const int g1 = gate(af, RB, bf, H);
            for (int rb = 0; rb < RB; rb++) af[rb][0] |= g1;
            for (int cb = 0; cb < H; cb++)
              for (int rb = 0; rb < RB; rb++) c[rb][cb] = pearl_sycl::dpas<8, true>(sg, af[rb], bf[cb], c[rb][cb]);
            for (int cb = H; cb < NCB; cb++) bf[cb] = load_b(ks, cb);
            const int g2 = gate(af, 0, bf + H, H);
            for (int rb = 0; rb < RB; rb++) af[rb][1] |= g2;
            for (int cb = H; cb < NCB; cb++)
              for (int rb = 0; rb < RB; rb++) c[rb][cb] = pearl_sycl::dpas<8, true>(sg, af[rb], bf[cb], c[rb][cb]);
          } else if constexpr (PF == 7) {
            // On entry A' and the first half of B' of this k-step are loaded
            // or in flight. The second half goes out before the first DPAS.
            const uint32_t nx = (ks + 1u) & 63u;
            const int g1 = gate(af, RB, bf, H);
            for (int cb = H; cb < NCB; cb++) bf[cb] = load_b(ks, cb);
            for (int rb = 0; rb < RB; rb++) af[rb][0] |= g1;
            for (int cb = 0; cb < H; cb++)
              for (int rb = 0; rb < RB; rb++) c[rb][cb] = pearl_sycl::dpas<8, true>(sg, af[rb], bf[cb], c[rb][cb]);
            const int g2 = gate(af, 0, bf + H, H) | (c[RB - 1][H - 1][0] & (int)P.zero);
            v8i an[RB];
            for (int rb = 0; rb < RB; rb++) an[rb] = load_a(nx, rb);
            for (int cb = 0; cb < H; cb++) bf[cb] = load_b(nx + (uint32_t)(g2 & 0), cb);
            for (int rb = 0; rb < RB; rb++) af[rb][1] |= g2;
            for (int cb = H; cb < NCB; cb++)
              for (int rb = 0; rb < RB; rb++) c[rb][cb] = pearl_sycl::dpas<8, true>(sg, af[rb], bf[cb], c[rb][cb]);
            for (int rb = 0; rb < RB; rb++) af[rb] = an[rb];
          } else if constexpr (PF == 8) {
            const uint32_t kr = ((ks + rot) & 63u) & P.kMask;
            peak_b4q qa[RB], qb[NCB];
            uint64_t t = 0;
            if constexpr (LA == 8) {
              for (int rb = 0; rb < RB; rb++) {
                qa[rb] = peak_block_read<peak_b4q, uint64_t>(Au + ((size_t)kr * RB + (size_t)rb) * fragBytes);
                t ^= qa[rb][0];
              }
            } else {
              for (int rb = 0; rb < RB; rb++) { af[rb] = load_a(ks, rb); t ^= (uint64_t)(uint32_t)af[rb][0]; }
            }
            if constexpr (LB == 8) {
              for (int cb = 0; cb < NCB; cb++) {
                qb[cb] = peak_block_read<peak_b4q, uint64_t>(Bu + ((size_t)kr * 8u + (size_t)cb) * fragBytes);
                t ^= qb[cb][0];
              }
            } else {
              for (int cb = 0; cb < NCB; cb++) { bf[cb] = load_b(ks, cb); t ^= (uint64_t)(uint32_t)bf[cb][0]; }
            }
            t &= (uint64_t)P.zero;
            if constexpr (LA == 8) {
              for (int rb = 0; rb < RB; rb++) { qa[rb][0] |= t; af[rb] = (v8i)qa[rb]; }
            } else {
              for (int rb = 0; rb < RB; rb++) af[rb][0] |= (int)t;
            }
            if constexpr (LB == 8) {
              for (int cb = 0; cb < NCB; cb++) { qb[cb][0] |= t; bf[cb] = (v8i)qb[cb]; }
            }
            for (int cb = 0; cb < NCB; cb++)
              for (int rb = 0; rb < RB; rb++) c[rb][cb] = pearl_sycl::dpas<8, true>(sg, af[rb], bf[cb], c[rb][cb]);
          } else if constexpr (PF == 9) {
            const uint32_t kr = ((ks + rot) & 63u) & P.kMask;
            uint32_t t = 0;
            for (int rb = 0; rb < RB; rb++) {
              const int8_t *u = Au + ((size_t)kr * RB + (size_t)rb) * fragBytes;
              if constexpr (LA == 8) t ^= (uint32_t)peak_block_read<peak_b4q, uint64_t>(u)[0];
              else if constexpr (LA == 7)
                t ^= peak_block_read<peak_b4u, uint32_t>(u)[0] ^ peak_block_read<peak_b4u, uint32_t>(u + 128)[0];
            }
            for (int cb = 0; cb < NCB; cb++) {
              const int8_t *u = Bu + ((size_t)kr * 8u + (size_t)cb) * fragBytes;
              if constexpr (LB == 8) t ^= (uint32_t)peak_block_read<peak_b4q, uint64_t>(u)[0];
              else if constexpr (LB == 7)
                t ^= peak_block_read<peak_b4u, uint32_t>(u)[0] ^ peak_block_read<peak_b4u, uint32_t>(u + 128)[0];
            }
            keep[0] ^= (int)t;
            for (int cb = 0; cb < NCB; cb++)
              for (int rb = 0; rb < RB; rb++)
                c[rb][cb] = pearl_sycl::dpas<8, true>(sg, ar[ks & 1u][rb], br[(ks + (uint32_t)cb) & 1u], c[rb][cb]);
          } else if constexpr (PF == 4) {
            for (int rb = 0; rb < RB; rb++) keep ^= load_a(ks, rb) << (1 + rb);
            for (int cb = 0; cb < NCB; cb++) keep ^= load_b(ks, cb) << (9 + cb);
            for (int cb = 0; cb < NCB; cb++)
              for (int rb = 0; rb < RB; rb++)
                c[rb][cb] = pearl_sycl::dpas<8, true>(sg, ar[ks & 1u][rb], br[(ks + (uint32_t)cb) & 1u], c[rb][cb]);
          } else {
            const int z = (int)(ks & P.zero);
            for (int rb = 0; rb < RB; rb++) af[rb] = ar[ks & 1u][rb] + z;
            for (int cb = 0; cb < NCB; cb++) bf[cb] = br[(ks + (uint32_t)cb) & 1u] + z;
            for (int cb = 0; cb < NCB; cb++)
              for (int rb = 0; rb < RB; rb++) c[rb][cb] = pearl_sycl::dpas<8, true>(sg, af[rb], bf[cb], c[rb][cb]);
          }
        }
        if constexpr (RD >= 1) {
          uint32_t x0 = 0, x1 = 0;
          for (int rb = 0; rb < RB; rb++)
            for (int cb = 0; cb < NCB; cb++) {
              for (int r = 0; r < 4; r++) x0 ^= (uint32_t)c[rb][cb][r];
              for (int r = 4; r < 8; r++) x1 ^= (uint32_t)c[rb][cb][r];
            }
          x0 ^= sycl::permute_group_by_xor(sg, x0, 1u);
          x1 ^= sycl::permute_group_by_xor(sg, x1, 1u);
          if (lane < 8u) part[(sgl * 8u + lane) * 16u + ch] = (lane & 1u) ? x1 : x0;
        }
      }
    }
    if constexpr (RD == 2) {
      sycl::group_barrier(it.get_group());
      if (sgi == 0 && lane < 8u) {
        const uint32_t qr = lane;
        uint32_t words[16];
        for (int j = 0; j < 16; j++) {
          uint32_t x = 0;
          for (int s = 0; s < S; s++) x ^= part[((sgl + (uint32_t)s) * 8u + qr) * 16u + (uint32_t)j];
          words[j] = x;
        }
        const uint32_t local = (2u * wr + (qr & 1u)) + A.rowsValid * (4u * wc + (qr >> 1));
        pearl_sycl::region_epilogue(words, local, A.test, A.hits, A.allTr);
      }
    } else {
      uint32_t x = 0;
      for (int rb = 0; rb < RB; rb++)
        for (int cb = 0; cb < NCB; cb++)
          for (int r = 0; r < 8; r++) x ^= (uint32_t)c[rb][cb][r];
      for (int r = 0; r < 8; r++) x ^= (uint32_t)keep[r];
      x = sycl::reduce_over_group(sg, x, sycl::bit_xor<uint32_t>());
      if (lane == 0) P.sink[(size_t)g * S + sgi] = x;
    }
  }
};

template <int RB, int NCB, int LA, int LB, int RD, int PF>
inline sycl::event peak_frag(sycl::queue &q, const PeakArgs &args, uint32_t groups) {
  constexpr int S = (4 / RB) * (8 / NCB);
  const size_t wg = (size_t)S * 8 * args.wpg;
  return q.submit([&](sycl::handler &h) {
    sycl::local_accessor<uint32_t, 1> part(wg * 16, h);
    h.parallel_for(sycl::nd_range<1>((size_t)(groups / args.wpg) * wg, wg), PeakFrag<RB, NCB, LA, LB, RD, PF>{args, part});
  });
}

// The "s-" rows: operands staged through shared local memory, the way the AMD folds stage 32 bytes of k
// at a time through LDS (../amd/README.md). A work-group is sga by sgb sub-groups, each with the fold's
// 16 by 64 tile: 16 sga rows by 64 sgb columns. A stage is KS k-steps. Every stage the group copies the
// next stage of its A' (512 bytes a 16-row block and k-step) and B' (2,048 bytes a 64-column block and
// k-step) from memory into a ring of D + 1 stages, 128 bytes a message (a lane's 16), and each sub-group
// reads its fragments from the ring. So A' is read from memory once for sgb sub-groups and B' once for
// sga, where the fold reads A' once a sub-group. Timing only: the copy is real, the results are not a
// fold's.
//   OPS   which operands go through the ring: 1 A', 2 B', 3 both. The other is loaded from memory as
//         the fold loads it (A' one vector of 8 dwords, B' two of 4)
//   MODE  0 the ring filled once, then only fragment reads from it (what reads from local memory cost)
//         1 the whole pipeline: a stage sends the loads of the stage D ahead, runs its k-steps (fragment
//           reads from the ring, 16 DPAS each), stores the loaded stage and meets the group at a barrier
//         2 1 without the copy (the barrier alone)
//         3 1 without the barrier (the copy alone)
//   D     stages in flight ahead of the one being used
//   BF    how a B' fragment is read from the ring: 1 one vector of 8 dwords, 2 two of 4
//   PPS   128-byte pieces a sub-group copies a stage, at most (registers held across the stage: 4 each)
//   KS    k-steps a stage
//         4 1 with a split barrier (SPV_INTEL_split_barrier, AMD's s_barrier_signal and s_barrier_wait;
//           modes 4, 5, 8, 13 to 16 need a probe built with
//           PEARL_SYCL_EXTRA="-DPEARL_PEAK_SPLIT -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier"
//           ./build.sh --jit --peak, which no release does):
//           a sub-group arrives after storing a stage and waits for that phase only just before its next
//           store, a stage later, so it is held only when another is a stage behind. D must be 2
//         5 4 without the copy (the split barrier alone)
//         6 1 with the copy's loads pinned: the stage's last k-step waits for them, so the compiler cannot
//           sink them to the store (AMD's PEARL_AMD_LPIN)
//         7 6 without the barrier
#if defined(__SYCL_DEVICE_ONLY__)
#define PEAK_LOCAL __attribute__((opencl_local))
#else
#define PEAK_LOCAL
#endif
#if defined(__SYCL_DEVICE_ONLY__) && defined(PEARL_SYCL_XMX_JIT) && defined(PEARL_PEAK_SPLIT)
// Scope 2 is the work-group; semantics 0x100 its local memory, with 0x4 release (arrive), 0x2 acquire (wait).
}  // namespace
SYCL_EXTERNAL void __spirv_ControlBarrierArriveINTEL(int, int, int) noexcept;
SYCL_EXTERNAL void __spirv_ControlBarrierWaitINTEL(int, int, int) noexcept;
namespace {
inline void peak_arrive() { __spirv_ControlBarrierArriveINTEL(2, 2, 0x104); }
inline void peak_wait() { __spirv_ControlBarrierWaitINTEL(2, 2, 0x102); }
#else
inline void peak_arrive() {}
inline void peak_wait() {}
#endif
template <int OPS, int MODE, int D, int BF, int PPS, int KS, int G = 0> struct PeakSlm {
  static constexpr int TR = G ? 4 : 2, TC = G ? 4 : 8;  // a sub-group's tile in 8-row and 8-column blocks
  static constexpr uint32_t R = (uint32_t)D + 1u;
  static constexpr bool RA = (OPS & 1) != 0, RBp = (OPS & 2) != 0;
  PeakArgs P;
  sycl::local_accessor<uint32_t, 1> ring;

  // Bytes of one k-step in the ring, and of a stage.
  // A group holds nAb 16-row blocks of A' and nBb 64-column blocks of B'.
  static uint32_t a_blocks(uint32_t ga) { return G ? 2u * ga : ga; }
  static uint32_t b_blocks(uint32_t gb) { return G ? gb / 2u : gb; }
  static uint32_t step_bytes(uint32_t ga, uint32_t gb) {
    return (RA ? 512u * a_blocks(ga) : 0u) + (RBp ? 2048u * b_blocks(gb) : 0u);
  }

  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    namespace syclex = sycl::ext::oneapi::experimental;
    return syclex::properties{syclex::sub_group_size<8>, sycl::ext::intel::experimental::grf_size<256>};
  }

  void operator()(sycl::nd_item<1> it) const {
    const FoldArgs &A = P.f;
    sycl::sub_group sg = it.get_sub_group();
    const uint32_t lane = (uint32_t)sg.get_local_linear_id();
    const uint32_t sgl = (uint32_t)sg.get_group_linear_id();
    const uint32_t ga = P.sga, gb = P.sgb, n = ga * gb;
    // 13, 14, 15: sub-groups n and up are producers. They copy the ring and run no DPAS.
    constexpr bool PROD = MODE == 13 || MODE == 14 || MODE == 15;
    const uint32_t copiers = PROD ? P.sgp : n, ci = PROD ? sgl - n : sgl;
    const bool producer = PROD && sgl >= n, copier = !PROD || producer;
    const uint32_t si = (sgl % n) % ga, sj = (sgl % n) / ga;
    const uint32_t nAb = a_blocks(ga), nBb = b_blocks(gb);
    const uint32_t hr = G ? ga : ga / 2u, cw = nBb;  // row and column windows a group
    const uint32_t rowWins = A.rowsValid / 2u, colWins = A.colBatch / 4u;
    uint32_t bandS = A.band / hr;
    if (bandS < 1u) bandS = 1u;
    uint32_t swr, swc;
    pearl_sycl::window_of((uint32_t)it.get_group(0), rowWins / hr, colWins / cw, bandS, swr, swc);
    const uint32_t ablk0 = 2u * hr * swr, bblk0 = A.colStart / 4u + cw * swc;
    // A k-step in the ring: [A' of the ga blocks, 512 bytes each][B' of the gb blocks, 2,048 bytes each],
    // as present; a stage is KS of them.
    const uint32_t aPart = RA ? 512u * nAb : 0u;
    const uint32_t T1 = step_bytes(ga, gb), T = T1 * (uint32_t)KS;
    const uint32_t NP1 = T1 / 128u, NP = T / 128u;
    PEAK_LOCAL int8_t *ls = reinterpret_cast<PEAK_LOCAL int8_t *>(ring.template get_multi_ptr<sycl::access::decorated::yes>().get_decorated());
    // Piece q of k-step ks in memory.
    auto src = [&](uint32_t q, uint32_t ks) -> const int8_t * {
      if (RA && q < 4u * nAb)
        return MODE == 12 ? A.Ap + (size_t)(q % 4u) * 128u
                          : A.Ap + (size_t)(ablk0 + q / 4u) * A.aStride + (size_t)ks * 512u + (size_t)(q % 4u) * 128u;
      const uint32_t r = q - (RA ? 4u * nAb : 0u);
      return A.Bp + (size_t)(bblk0 + r / 16u) * A.bStride + (size_t)ks * 2048u + (size_t)(r % 16u) * 128u;
    };
    peak_v4i held[PPS], nxt[PPS];
    auto load_into = [&](uint32_t st, peak_v4i *out) {
      for (int i = 0; i < PPS; i++) {
        const uint32_t p = ci + (uint32_t)i * copiers;
        if (p < NP) out[i] = *reinterpret_cast<const peak_v4i *>(src(p % NP1, (st * (uint32_t)KS + p / NP1) & 63u) + 16u * lane);
      }
    };
    auto load_stage = [&](uint32_t st) { load_into(st, held); };
    auto store_stage = [&](uint32_t slot) {
      for (int i = 0; i < PPS; i++) {
        const uint32_t p = ci + (uint32_t)i * copiers;
        if (p < NP) *reinterpret_cast<PEAK_LOCAL peak_v4i *>(ls + (size_t)slot * T + p * 128u + 16u * lane) = held[i];
      }
    };
    const uint32_t aOwn = G ? 2u * si : si;  // the sub-group's first 16-row block in the group
    const uint32_t bOwn = G ? (sj / 2u) * 2048u + (sj % 2u) * 1024u : sj * 2048u;  // its B' in a k-step
    const int8_t *Ag = A.Ap + (size_t)(ablk0 + aOwn) * A.aStride + 32u * lane;
    const int8_t *Bg = A.Bp + (size_t)(bblk0 + (G ? sj / 2u : sj)) * A.bStride + (G ? (sj % 2u) * 1024u : 0u) + 32u * lane;
    if (copier) {
      for (uint32_t st = 0; st < (MODE == 0 ? R : (uint32_t)D); st++) {
        load_stage(st);
        store_stage(st);
      }
    }
    constexpr bool CARRY = MODE == 8 || MODE == 9 || MODE == 15;
    if constexpr (CARRY) {
      if (copier) load_stage((uint32_t)D);
    }
    sycl::group_barrier(it.get_group());
    v8i c[TR][TC];
    for (int rb = 0; rb < TR; rb++)
      for (int cb = 0; cb < TC; cb++) c[rb][cb] = 0;
    constexpr uint32_t NST = 64u / (uint32_t)KS;
    constexpr bool COPY = MODE == 1 || MODE == 3 || MODE == 4 || MODE == 6 || MODE == 7 || MODE == 16 || CARRY;  // 16: 4 without the pin
    // 7 with only its loads, or only its stores; 12 its loads all from one place that stays in the cache
    constexpr bool LOADONLY = MODE == 10 || MODE == 12, STOREONLY = MODE == 11;
    constexpr bool PIN = MODE == 4 || MODE == 6 || MODE == 7 || LOADONLY;
    constexpr bool FULLBAR = MODE == 1 || MODE == 2 || MODE == 6;
    constexpr bool SPLIT = MODE == 4 || MODE == 5 || MODE == 8 || MODE == 16 || PROD;
    bool arrived = false;
    if (producer) {
      // Copy stage st + D each stage (15: load stage st + D + 1 and store the one loaded a stage before),
      // meeting the consumers at the same split barrier.
      for (uint32_t rep = 0; rep < P.reps; rep++) {
#pragma unroll 1
        for (uint32_t st = 0; st < NST; st++) {
          if constexpr (MODE == 13) load_stage(st + (uint32_t)D);
          if constexpr (MODE == 15) load_into(st + (uint32_t)D + 1u, nxt);
          if (arrived) peak_wait();
          if constexpr (MODE != 14) store_stage((st + (uint32_t)D) % R);
          peak_arrive();
          arrived = true;
          if constexpr (MODE == 15) {
            for (int i = 0; i < PPS; i++) held[i] = nxt[i];
          }
        }
      }
      peak_wait();
      return;
    }
    for (uint32_t rep = 0; rep < P.reps; rep++) {
      // One stage an iteration: unrolled, the compiler hoists the next stages' reads and spills.
#pragma unroll 1
      for (uint32_t st = 0; st < NST; st++) {
        int pin = 0;
        if constexpr (PROD) {
        } else if constexpr (CARRY) load_into(st + (uint32_t)D + 1u, nxt);
        else if constexpr (COPY || LOADONLY) load_stage(st + (uint32_t)D);
        if constexpr (PIN) {
          for (int i = 0; i < PPS; i++) pin ^= held[i][0];
          pin &= (int)P.zero;
        }
        const PEAK_LOCAL int8_t *sl0 = ls + (size_t)(st % R) * T;
        for (int kk = 0; kk < KS; kk++) {
          const uint32_t ks = st * (uint32_t)KS + (uint32_t)kk;
          const PEAK_LOCAL int8_t *sl = sl0 + (uint32_t)kk * T1;
          v8i af[TR], bf[TC];
          for (int rb = 0; rb < TR; rb++) {
            if constexpr (RA) af[rb] = *reinterpret_cast<const PEAK_LOCAL v8iu *>(sl + aOwn * 512u + (uint32_t)rb * 256u + 32u * lane);
            else af[rb] = *reinterpret_cast<const v8iu *>(Ag + (size_t)(rb / 2) * A.aStride + (size_t)ks * 512u + (size_t)(rb % 2) * 256u);
          }
          for (int cb = 0; cb < TC; cb++) {
            if constexpr (RBp) {
              const PEAK_LOCAL int8_t *p = sl + aPart + bOwn + (uint32_t)cb * 256u + 32u * lane;
              if constexpr (BF == 1) {
                bf[cb] = *reinterpret_cast<const PEAK_LOCAL v8iu *>(p);
              } else {
                const peak_v4i lo = *reinterpret_cast<const PEAK_LOCAL peak_v4i *>(p);
                const peak_v4i hi = *reinterpret_cast<const PEAK_LOCAL peak_v4i *>(p + 16);
                bf[cb] = __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
              }
            } else {
              bf[cb] = pearl_sycl::load_halves(Bg + (size_t)ks * 2048u + (size_t)cb * 256u);
            }
          }
          int t = 0;
          for (int rb = 0; rb < TR; rb++) t ^= af[rb][0];
          for (int cb = 0; cb < TC; cb++) t ^= bf[cb][0];
          t &= (int)P.zero;
          if (kk == KS - 1) t |= pin;  // the stage's last k-step waits for the copy's loads
          for (int rb = 0; rb < TR; rb++) af[rb][0] |= t;
          for (int cb = 0; cb < TC; cb++)
            for (int rb = 0; rb < TR; rb++) c[rb][cb] = pearl_sycl::dpas<8, true>(sg, af[rb], bf[cb], c[rb][cb]);
        }
        if constexpr (SPLIT) {
          if (arrived) peak_wait();
        }
        if constexpr ((COPY || STOREONLY) && !PROD) store_stage((st + (uint32_t)D) % R);
        if constexpr (FULLBAR) sycl::group_barrier(it.get_group());
        if constexpr (SPLIT) {
          peak_arrive();
          arrived = true;
        }
        if constexpr (CARRY && !PROD) {
          for (int i = 0; i < PPS; i++) held[i] = nxt[i];
        }
      }
    }
    if constexpr (SPLIT) peak_wait();
    uint32_t x = 0;
    for (int rb = 0; rb < TR; rb++)
      for (int cb = 0; cb < TC; cb++)
        for (int r = 0; r < 8; r++) x ^= (uint32_t)c[rb][cb][r];
    x = sycl::reduce_over_group(sg, x, sycl::bit_xor<uint32_t>());
    if (lane == 0) P.sink[(size_t)it.get_group(0) * n + sgl] = x;
  }
};

template <int OPS, int MODE, int D, int BF, int PPS, int KS, int G = 0>
inline sycl::event peak_slm(sycl::queue &q, const PeakArgs &args, uint32_t groups) {
  typedef PeakSlm<OPS, MODE, D, BF, PPS, KS, G> K;
#if !defined(PEARL_PEAK_SPLIT)
  if (MODE == 4 || MODE == 5 || MODE == 8 || MODE == 13 || MODE == 14 || MODE == 15 || MODE == 16)
    throw std::runtime_error("a split-barrier row: build with PEARL_PEAK_SPLIT (see the s- rows' comment)");
#endif
  if (G && (args.sgb & 1u)) throw std::runtime_error("the 32 by 32 rows need an even --sgb");
  const bool prod = MODE == 13 || MODE == 14 || MODE == 15;
  const uint32_t n = args.sga * args.sgb, all = n + (prod ? args.sgp : 0u);
  if (all > 64u) throw std::runtime_error("more than 64 sub-groups a work-group");
  const uint32_t T = K::step_bytes(args.sga, args.sgb) * (uint32_t)KS;
  if (T / 128u > (uint32_t)PPS * (prod ? args.sgp : n)) throw std::runtime_error("--sga by --sgb leaves more than PPS pieces a sub-group");
  if (K::R * T > 65536u) throw std::runtime_error("the ring is over 64 KB at this --sga, --sgb");
  const uint32_t wgs = groups / (args.sga * args.sgb / 2u);
  return q.submit([&](sycl::handler &h) {
    sycl::local_accessor<uint32_t, 1> ring(K::R * T / 4u, h);
    h.parallel_for(sycl::nd_range<1>((size_t)wgs * all * 8u, (size_t)all * 8u), K{args, ring});
  });
}

// The image rows are in the build the driver compiles only (./build.sh --jit --peak): the prebuilt
// form's CPU image does not compile them.
#if defined(PEARL_SYCL_XMX_JIT)
// ---------------------------------------------------------------------------
// The image rows ("i-"): does an Xe-core have a second way in for operands?
//
// Every row above reads through the core's data port (untyped loads). A core
// also has a sampler, with caches of its own. These rows keep the fragment
// fold's tile (16 rows by 64 columns, 16 DPAS a k-step) and read one or both
// operands as texels of a 2D image, 16 bytes a texel (RGBA, 32-bit unsigned),
// so one fetch is a lane's half fragment and one message is 128 bytes, the
// same as the fold's half-fragment load of B'. Timing only: the images hold
// other bytes than the operands, and the bits are not a fold's.
//
//   LA, LB  0 registers, 1 one vector of 8 dwords from memory, 2 two vectors
//           of 4 dwords from memory, 9 two texel fetches from the image.
//           LB 10, 11, 14, 12, 15: the first 1, 2, 3, 4 or 6 of a k-step's 8
//           fragments of B' from the image and the rest from memory. LA 13: A'
//           fetched from B's image where B' fragments 6 and 7 are, which every
//           sub-group of a band shares: not a real A'
// The "i-" rows read the old SYCL image class (sycl::image). Those images are
// never written: a kernel that writes one hung the card, one made from host
// bytes crashed the runtime, and so did a kernel that reads only its first
// image or reads two. On images never written the rows come out faster than
// on a filled one, so they are not to be trusted. The "j-" rows read a
// bindless image filled from host memory and checked: those are the figures
// (README.md, "A second way in: the sampler"). Both run only with --only.
//   PF      3 every load first and every DPAS made to wait for all of them,
//           4 the loads only kept by an XOR and the DPAS on register operands
//             (what the messages alone cost)
//   SMP     0 a read with no sampler (the compiler picks the message), 1 a
//           read through a sampler (nearest, unnormalized, integer coordinates),
//           2 a fetch from the bindless image (the "j-" rows; B' only)
//
// An image is 4096 texels wide (64 KB a row) and holds the operand's bytes at
// the same offsets as the memory rows read them: texel = byte offset / 16.
typedef sycl::vec<uint32_t, 4> peak_texel;
typedef sycl::accessor<peak_texel, 2, sycl::access::mode::read, sycl::access::target::image> peak_img_acc;
struct PeakImages {
  sycl::image<2> *a = nullptr, *b = nullptr;
  uint32_t yMaskA = 0, yMaskB = 0;  // rows of the image, less one (a power of two)
  // The "j-" rows: B's image again as a bindless image (sycl_ext_oneapi_bindless_images), which is a plain
  // handle in the kernel, is filled by a copy from memory, and is read with fetch_image.
  sycl::ext::oneapi::experimental::unsampled_image_handle hb{};
  bool haveBindless = false;
};
PeakImages gImg;

template <int LA, int LB, int PF, int SMP> struct PeakImg {
  PeakArgs P;
  uint32_t yMaskA, yMaskB;
  peak_img_acc ia, ib;
  sycl::sampler smp;
  sycl::ext::oneapi::experimental::unsampled_image_handle hb;

  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    namespace syclex = sycl::ext::oneapi::experimental;
    return syclex::properties{syclex::sub_group_size<8>, sycl::ext::intel::experimental::grf_size<256>};
  }

  void operator()(sycl::nd_item<1> it) const {
    const FoldArgs &A = P.f;
    sycl::sub_group sg = it.get_sub_group();
    const uint32_t lane = (uint32_t)sg.get_local_linear_id();
    const uint32_t sgi = (uint32_t)sg.get_group_linear_id();
    const uint32_t g = (uint32_t)it.get_group(0);
    const uint32_t rowWins = A.rowsValid / 2u, colWins = A.colBatch / 4u;
    uint32_t wr, wc;
    pearl_sycl::window_of(g, rowWins, colWins, A.band, wr, wc);
    const uint32_t wra = wr & P.rowMask;
    const uint32_t wcb = (A.colStart / 4u + wc) & P.colMask;
    const uint32_t blkA = 2u * wra + sgi;  // the sub-group's 16-row block of A'
    const int8_t *Au = A.Ap + (size_t)blkA * A.aStride + 32u * lane;
    const int8_t *Bu = A.Bp + (size_t)wcb * A.bStride + 32u * lane;
    const uint32_t tA = blkA * (A.aStride >> 4) + 2u * lane;  // this lane's first texel of the block
    const uint32_t tB = wcb * (A.bStride >> 4) + 2u * lane;
    v8i ar[2][2], br[2];
    for (int set = 0; set < 2; set++)
      for (int rb = 0; rb < 2; rb++)
        for (int r = 0; r < 8; r++)
          ar[set][rb][r] = (int)bytes7(mix(P.seed + lane * 131u + sgi * 17u + (uint32_t)(set * 64 + rb * 8 + r)));
    for (int i = 0; i < 2; i++)
      for (int d = 0; d < 8; d++) br[i][d] = (int)bytes7(mix((P.seed ^ 0xB5297A4Du) + lane * 257u + (uint32_t)(i * 8 + d)));
    auto fetch = [&](const peak_img_acc &im, uint32_t t, uint32_t ymask) -> peak_v4i {
      const sycl::vec<int, 2> xy{(int)(t & 4095u), (int)((t >> 12) & ymask)};
      peak_texel v;
      if constexpr (SMP == 1) v = im.read(xy, smp);
      else if constexpr (SMP == 2) v = sycl::ext::oneapi::experimental::fetch_image<peak_texel>(hb, xy);
      else v = im.read(xy);
      return peak_v4i{(int)v[0], (int)v[1], (int)v[2], (int)v[3]};
    };
    auto load_a = [&](uint32_t ks, int rb) -> v8i {
      const uint32_t f = (ks & P.kMask) * 2u + (uint32_t)rb;  // fragment of the block
      if constexpr (LA == 0) return ar[ks & 1u][rb];
      else if constexpr (LA == 13) {
        // A' read where B' fragments 6 and 7 of the k-step are, from B's image: the cost of A' by the
        // sampler without A's own coordinates, which hung the card.
        const uint32_t fb = (ks & P.kMask) * 8u + 6u + (uint32_t)rb;
        const peak_v4i lo = fetch(ib, tB + fb * 16u, yMaskB), hi = fetch(ib, tB + fb * 16u + 1u, yMaskB);
        return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
      } else if constexpr (LA == 9) {
        // With A' alone read from an image it goes in as the kernel's second image: a kernel that used its
        // first image and not its second hung the card (GPU HANG in the kernel log, device lost).
        const peak_img_acc &im = LB >= 9 ? ia : ib;
        const peak_v4i lo = fetch(im, tA + f * 16u, yMaskA), hi = fetch(im, tA + f * 16u + 1u, yMaskA);
        return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
      } else if constexpr (LA == 2) {
        const peak_v4i lo = *reinterpret_cast<const peak_v4i *>(Au + (size_t)f * 256u);
        const peak_v4i hi = *reinterpret_cast<const peak_v4i *>(Au + (size_t)f * 256u + 16);
        return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
      } else return *reinterpret_cast<const v8iu *>(Au + (size_t)f * 256u);
    };
    auto load_b = [&](uint32_t ks, int cb) -> v8i {
      const uint32_t f = (ks & P.kMask) * 8u + (uint32_t)cb;
      if constexpr (LB == 0) return br[(ks + (uint32_t)cb) & 1u];
      else if constexpr (LB >= 9) {
        // 9: every fragment from the image. 10, 11, 12, 14, 15: the first 1, 2, 4, 3 or 6 of the 8, the rest
        // from memory.
        if (LB != 9 && cb >= (LB == 10 ? 1 : LB == 11 ? 2 : LB == 12 ? 4 : LB == 14 ? 3 : 6)) {
          const peak_v4i lo = *reinterpret_cast<const peak_v4i *>(Bu + (size_t)f * 256u);
          const peak_v4i hi = *reinterpret_cast<const peak_v4i *>(Bu + (size_t)f * 256u + 16);
          return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
        }
        const peak_v4i lo = fetch(ib, tB + f * 16u, yMaskB), hi = fetch(ib, tB + f * 16u + 1u, yMaskB);
        return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
      } else if constexpr (LB == 2) {
        const peak_v4i lo = *reinterpret_cast<const peak_v4i *>(Bu + (size_t)f * 256u);
        const peak_v4i hi = *reinterpret_cast<const peak_v4i *>(Bu + (size_t)f * 256u + 16);
        return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
      } else return *reinterpret_cast<const v8iu *>(Bu + (size_t)f * 256u);
    };
    v8i c[2][8];
    for (int rb = 0; rb < 2; rb++)
      for (int cb = 0; cb < 8; cb++) c[rb][cb] = 0;
    v8i keep = 0;
    v8i af[2], bf[8];
    for (uint32_t ks = 0; ks < 64u; ks++) {
      if constexpr (PF == 3) {
        for (int rb = 0; rb < 2; rb++) af[rb] = load_a(ks, rb);
        for (int cb = 0; cb < 8; cb++) bf[cb] = load_b(ks, cb);
        int t = 0;
        for (int rb = 0; rb < 2; rb++) t ^= af[rb][0] ^ af[rb][4];
        for (int cb = 0; cb < 8; cb++) t ^= bf[cb][0] ^ bf[cb][4];
        t &= (int)P.zero;
        for (int rb = 0; rb < 2; rb++) af[rb][0] |= t;
        for (int cb = 0; cb < 8; cb++)
          for (int rb = 0; rb < 2; rb++) c[rb][cb] = pearl_sycl::dpas<8, true>(sg, af[rb], bf[cb], c[rb][cb]);
      } else {
        if constexpr (LA != 0)
          for (int rb = 0; rb < 2; rb++) keep ^= load_a(ks, rb) << (1 + rb);
        if constexpr (LB != 0)
          for (int cb = 0; cb < 8; cb++) keep ^= load_b(ks, cb) << (9 + cb);
        for (int cb = 0; cb < 8; cb++)
          for (int rb = 0; rb < 2; rb++)
            c[rb][cb] = pearl_sycl::dpas<8, true>(sg, ar[ks & 1u][rb], br[(ks + (uint32_t)cb) & 1u], c[rb][cb]);
      }
    }
    uint32_t x = 0;
    for (int rb = 0; rb < 2; rb++)
      for (int cb = 0; cb < 8; cb++)
        for (int r = 0; r < 8; r++) x ^= (uint32_t)c[rb][cb][r];
    for (int r = 0; r < 8; r++) x ^= (uint32_t)keep[r];
    x = sycl::reduce_over_group(sg, x, sycl::bit_xor<uint32_t>());
    if (lane == 0) P.sink[(size_t)g * 2u + sgi] = x;
  }
};

template <int LA, int LB, int PF, int SMP>
inline sycl::event peak_img(sycl::queue &q, const PeakArgs &args, uint32_t groups) {
  return q.submit([&](sycl::handler &h) {
    peak_img_acc ia = gImg.a->get_access<peak_texel, sycl::access::mode::read>(h);
    peak_img_acc ib = (LB >= 9 || LA == 13 ? gImg.b : gImg.a)->get_access<peak_texel, sycl::access::mode::read>(h);
    sycl::sampler smp(sycl::coordinate_normalization_mode::unnormalized, sycl::addressing_mode::clamp_to_edge,
                      sycl::filtering_mode::nearest);
    h.parallel_for(sycl::nd_range<1>((size_t)groups * 16, 16),
                   PeakImg<LA, LB, PF, SMP>{args, gImg.yMaskA, gImg.yMaskB, ia, ib, smp, gImg.hb});
  });
}

#endif  // PEARL_SYCL_XMX_JIT

struct Row {
  const char *name;
  const char *what;
  int acc;        // accumulators that get a DPAS
  int counted;    // accumulators the rate is counted over: all of them where `acc` is cut to show the loads
  int all;        // accumulators a sub-group has
  uint32_t reps;  // 0: the --reps value
  uint32_t rowMask, colMask, kMask;
  std::function<sycl::event(sycl::queue &, const PeakArgs &, uint32_t)> run;
};

// FOLD: the shape is one the core has a fold for, so the "fold" row can run it.
template <int SG, int RB, bool FOLD> void add_rows(std::vector<Row> &v) {
  constexpr int ALL = RB * (64 / SG);
  const uint32_t all = ~0u;
#define PEAK_ROW(NAME, WHAT, LA, LB, RD, NA, COUNTED, REPS, RMASK, CMASK, KMASK)               \
  v.push_back({NAME, WHAT, NA, COUNTED, ALL, REPS, RMASK, CMASK, KMASK,                         \
               [](sycl::queue &q, const PeakArgs &a, uint32_t g) {                              \
                 return peak_step<SG, RB, LA, LB, RD, NA>(q, a, g);                             \
               }})
  PEAK_ROW("regs", "DPAS alone, operands in registers, long threads", 0, 0, 0, ALL, ALL, 0u, all, all, all);
  PEAK_ROW("regs-half", "regs on half of the accumulators", 0, 0, 0, ALL / 2, ALL / 2, 0u, all, all, all);
  PEAK_ROW("regs-quarter", "regs on a quarter of the accumulators", 0, 0, 0, ALL / 4, ALL / 4, 0u, all, all, all);
  PEAK_ROW("regs-one", "regs on one accumulator", 0, 0, 0, 1, 1, 0u, all, all, all);
  PEAK_ROW("regs-short", "regs, one window a thread as the fold has it", 0, 0, 0, ALL, ALL, 1u, all, all, all);
  PEAK_ROW("regs-rd", "regs-short plus the fold's readout, barrier, hash and test", 0, 0, 2, ALL, ALL, 1u, all, all, all);
  PEAK_ROW("fixed", "the fold's loads, all of one fragment", 1, 1, 0, ALL, ALL, 1u, 0u, 0u, 0u);
  PEAK_ROW("window0", "the fold's loads, every window reading window 0", 1, 1, 0, ALL, ALL, 1u, 0u, 0u, all);
  PEAK_ROW("a-win0", "A' loads only, from 1 row window (64 KB)", 1, 0, 0, ALL, ALL, 1u, 0u, all, all);
  PEAK_ROW("a-res16", "A' loads only, from 16 row windows (1 MB)", 1, 0, 0, ALL, ALL, 1u, 15u, all, all);
  PEAK_ROW("a-res256", "A' loads only, from 256 row windows (16 MB)", 1, 0, 0, ALL, ALL, 1u, 255u, all, all);
  PEAK_ROW("a-res4096", "A' loads only, from 4096 row windows (256 MB)", 1, 0, 0, ALL, ALL, 1u, 4095u, all, all);
  PEAK_ROW("b-win0", "B' loads only, from 1 column window (128 KB)", 0, 1, 0, ALL, ALL, 1u, all, 0u, all);
  PEAK_ROW("b-res4", "B' loads only, from 4 column windows (512 KB)", 0, 1, 0, ALL, ALL, 1u, all, 3u, all);
  PEAK_ROW("b-res16", "B' loads only, from 16 column windows (2 MB)", 0, 1, 0, ALL, ALL, 1u, all, 15u, all);
  PEAK_ROW("b-res64", "B' loads only, from 64 column windows (8 MB)", 0, 1, 0, ALL, ALL, 1u, all, 63u, all);
  PEAK_ROW("load-a", "A' as the fold loads it, B in registers", 2, 0, 0, ALL, ALL, 1u, all, all, all);
  PEAK_ROW("load-b", "B' as the fold loads it, A in registers", 0, 2, 0, ALL, ALL, 1u, all, all, all);
  PEAK_ROW("own-a", "A' as the fold loads it, B' from window 0", 2, 1, 0, ALL, ALL, 1u, all, 0u, all);
  PEAK_ROW("own-b", "B' as the fold loads it, A' from window 0", 1, 2, 0, ALL, ALL, 1u, 0u, all, all);
  PEAK_ROW("loads", "both operands as the fold loads them, no readout", 2, 2, 0, ALL, ALL, 1u, all, all, all);
  PEAK_ROW("loads-1dpas", "loads with one DPAS a k-step: the rate the loads alone would allow", 2, 2, 0, 1, ALL, 1u, all, all, all);
  PEAK_ROW("loads-xor", "loads plus the chunk XOR, lane shuffles and local store", 2, 2, 1, ALL, ALL, 1u, all, all, all);
  PEAK_ROW("fold-copy", "and the barrier, hash and test: this file's copy of the fold", 2, 2, 2, ALL, ALL, 1u, all, all, all);
  if constexpr (FOLD) {
    v.push_back({"fold", "the kernel the core mines with", ALL, ALL, ALL, 1u, all, all, all,
                 [](sycl::queue &q, const PeakArgs &a, uint32_t g) {
                   pearl_sycl::fold_xmx<SG, RB, true>(q, a.f, g);
                   return sycl::event();
                 }});
  }
  PEAK_ROW("frag-a", "A' at fragment-order addresses, B in registers", 3, 0, 0, ALL, ALL, 1u, all, all, all);
  PEAK_ROW("frag-b", "B' at fragment-order addresses, A in registers", 0, 3, 0, ALL, ALL, 1u, all, all, all);
  PEAK_ROW("frag", "both at fragment-order addresses, no readout", 3, 3, 0, ALL, ALL, 1u, all, all, all);
  PEAK_ROW("frag-fold", "both at fragment-order addresses, with the readout, barrier, hash and test", 3, 3, 2, ALL, ALL, 1u, all, all, all);
  PEAK_ROW("vec-a", "A as one vector load a fragment (fragment order), B in registers", 4, 0, 0, ALL, ALL, 1u, all, all, all);
  PEAK_ROW("vec-b", "B' as one vector load a fragment at the fold's addresses, A in registers", 0, 4, 0, ALL, ALL, 1u, all, all, all);
  PEAK_ROW("vec", "both as one vector load a fragment, B' at the fold's addresses", 4, 4, 0, ALL, ALL, 1u, all, all, all);
  PEAK_ROW("vec-fold", "vec with the readout, barrier, hash and test", 4, 4, 2, ALL, ALL, 1u, all, all, all);
  PEAK_ROW("vecfrag", "both as one vector load a fragment, both in fragment order", 4, 5, 0, ALL, ALL, 1u, all, all, all);
  PEAK_ROW("vecfrag-fold", "vecfrag with the readout, barrier, hash and test", 4, 5, 2, ALL, ALL, 1u, all, all, all);
#undef PEAK_ROW
}

// The fragment fold's rows. Every one is 16 accumulators a sub-group: "f-" 16
// rows by 64 columns, the fold's tile, "g-" 32 rows by 32 columns.
void add_frag_rows(std::vector<Row> &v) {
  const uint32_t all = ~0u;
#define FRAG_ROW(NAME, WHAT, RB, NCB, LA, LB, RD, PF, REPS, RMASK, CMASK, KMASK)               \
  v.push_back({NAME, WHAT, 16, 16, 16, REPS, RMASK, CMASK, KMASK,                               \
               [](sycl::queue &q, const PeakArgs &a, uint32_t g) {                              \
                 return peak_frag<RB, NCB, LA, LB, RD, PF>(q, a, g);                            \
               }})
  FRAG_ROW("f-regs", "DPAS alone on 16 accumulators, operands in registers, long threads", 2, 8, 0, 0, 0, 0, 0u, all, all, all);
  FRAG_ROW("f-regs-short", "f-regs, one window a thread as the fold has it", 2, 8, 0, 0, 0, 0, 1u, all, all, all);
  FRAG_ROW("f-fold", "the fragment fold: this file's copy", 2, 8, 1, 1, 2, 0, 1u, all, all, all);
  FRAG_ROW("f-loads", "its loads and DPAS, no readout", 2, 8, 1, 1, 0, 0, 1u, all, all, all);
  FRAG_ROW("f-win0", "its loads, every window reading window 0 (64 KB of A', 128 KB of B')", 2, 8, 1, 1, 0, 0, 1u, 0u, 0u, all);
  FRAG_ROW("f-fixed", "its loads, all of one k-step of window 0", 2, 8, 1, 1, 0, 0, 1u, 0u, 0u, 0u);
  FRAG_ROW("f-a", "A' loaded from its own addresses, B in registers", 2, 8, 1, 0, 0, 0, 1u, all, all, all);
  FRAG_ROW("f-b", "B' loaded from its own addresses, A in registers", 2, 8, 0, 1, 0, 0, 1u, all, all, all);
  FRAG_ROW("f-pf3", "f-loads with every DPAS of a k-step made to wait for its ten loads", 2, 8, 1, 1, 0, 3, 1u, all, all, all);
  FRAG_ROW("f-keep", "ten loads a k-step kept by an XOR, the 16 DPAS on register operands", 2, 8, 1, 1, 0, 4, 1u, all, all, all);
  FRAG_ROW("f-keep-b", "f-keep with the eight B' loads only", 2, 8, 0, 1, 0, 4, 1u, all, all, all);
  FRAG_ROW("f-alu", "no loads: every operand written by an add before its k-step", 2, 8, 0, 0, 0, 5, 1u, all, all, all);
  // The same bytes in other load forms.
  FRAG_ROW("f-v4", "f-loads with each fragment loaded as two vectors of 4 dwords", 2, 8, 2, 2, 0, 0, 1u, all, all, all);
  FRAG_ROW("f-v4-a", "A' as two vectors of 4 dwords, B in registers", 2, 8, 2, 0, 0, 0, 1u, all, all, all);
  FRAG_ROW("f-v4-b", "B' as two vectors of 4 dwords, A in registers", 2, 8, 0, 2, 0, 0, 1u, all, all, all);
  FRAG_ROW("f-v4-pf3", "f-v4 with every DPAS of a k-step made to wait for its loads", 2, 8, 2, 2, 0, 3, 1u, all, all, all);
  FRAG_ROW("f-v4-b-pf3", "f-v4-b that way", 2, 8, 0, 2, 0, 3, 1u, all, all, all);
  FRAG_ROW("f-v84-pf3", "A' as one vector of 8 dwords, B' as two of 4, every DPAS waiting for the loads", 2, 8, 1, 2, 0, 3, 1u, all, all, all);
  FRAG_ROW("f-v4-pf3-fold", "f-v4-pf3 with the readout, barrier, hash and test", 2, 8, 2, 2, 2, 3, 1u, all, all, all);
  FRAG_ROW("f-v84-pf3-fold", "f-v84-pf3 with the readout, barrier, hash and test", 2, 8, 1, 2, 2, 3, 1u, all, all, all);
  FRAG_ROW("f-v84-pf6", "A' one vector, B' halves, in two halves of a k-step", 2, 8, 1, 2, 0, 6, 1u, all, all, all);
  FRAG_ROW("f-v4-pf6", "both operands as halves, in two halves of a k-step", 2, 8, 2, 2, 0, 6, 1u, all, all, all);
  FRAG_ROW("f-v84-pf7", "f-v84-pf6 with each half's loads in flight during the other half's DPAS", 2, 8, 1, 2, 0, 7, 1u, all, all, all);
  FRAG_ROW("f-v4-pf7", "f-v4-pf6 that way", 2, 8, 2, 2, 0, 7, 1u, all, all, all);
  FRAG_ROW("f-v84-pf7-fold", "f-v84-pf7 with the readout, barrier, hash and test", 2, 8, 1, 2, 2, 7, 1u, all, all, all);
  FRAG_ROW("f-v84-pf6-fold", "f-v84-pf6 with the readout, barrier, hash and test", 2, 8, 1, 2, 2, 6, 1u, all, all, all);
  FRAG_ROW("g-v4-pf6", "32 by 32, halves, in two halves of a k-step", 4, 4, 2, 2, 0, 6, 1u, all, all, all);
  FRAG_ROW("g-v4-pf7", "32 by 32, halves, each half's loads in flight during the other half's DPAS", 4, 4, 2, 2, 0, 7, 1u, all, all, all);
  FRAG_ROW("f-q4", "f-loads with each fragment loaded as one vector of 4 qwords", 2, 8, 6, 6, 0, 0, 1u, all, all, all);
  FRAG_ROW("f-q4-b", "B' as one vector of 4 qwords, A in registers", 2, 8, 0, 6, 0, 0, 1u, all, all, all);
  FRAG_ROW("f-q4-pf3", "f-q4 with every DPAS of a k-step made to wait for its loads", 2, 8, 6, 6, 0, 3, 1u, all, all, all);
  FRAG_ROW("f-q4-b-pf3", "f-q4-b that way", 2, 8, 0, 6, 0, 3, 1u, all, all, all);
  // 32 rows by 32 columns a sub-group: 4 fragments of each operand a k-step.
  FRAG_ROW("g-regs", "DPAS alone, 32 by 32", 4, 4, 0, 0, 0, 0, 0u, all, all, all);
  FRAG_ROW("g-loads", "32 by 32, one vector of 8 dwords a fragment", 4, 4, 1, 1, 0, 0, 1u, all, all, all);
  FRAG_ROW("g-pf3", "g-loads with every DPAS of a k-step made to wait for its loads", 4, 4, 1, 1, 0, 3, 1u, all, all, all);
  FRAG_ROW("g-v4", "32 by 32, two vectors of 4 dwords a fragment", 4, 4, 2, 2, 0, 0, 1u, all, all, all);
  FRAG_ROW("g-v4-pf3", "g-v4 with every DPAS of a k-step made to wait for its loads", 4, 4, 2, 2, 0, 3, 1u, all, all, all);
  FRAG_ROW("g-v4-a-pf3", "g-v4-pf3 with A' loaded and B in registers", 4, 4, 2, 0, 0, 3, 1u, all, all, all);
  FRAG_ROW("g-v4-b-pf3", "g-v4-pf3 with B' loaded and A in registers", 4, 4, 0, 2, 0, 3, 1u, all, all, all);
  FRAG_ROW("g-v4-pf3-fold", "g-v4-pf3 with the readout, barrier, hash and test", 4, 4, 2, 2, 2, 3, 1u, all, all, all);
  FRAG_ROW("g-v84-pf3", "32 by 32, A' one vector of 8 dwords, B' two of 4, every DPAS waiting (the fold's forms)", 4, 4, 1, 2, 0, 3, 1u, all, all, all);
  FRAG_ROW("g-q4-pf3", "32 by 32, one vector of 4 qwords a fragment, every DPAS waiting for the loads", 4, 4, 6, 6, 0, 3, 1u, all, all, all);
  FRAG_ROW("g-win0", "g-v4-pf3 with every window reading window 0", 4, 4, 2, 2, 0, 3, 1u, 0u, 0u, all);
  // Sub-group block reads: one address a message.
  FRAG_ROW("f-k4", "both operands as two block reads of 128 bytes a fragment", 2, 8, 7, 7, 0, 0, 1u, all, all, all);
  FRAG_ROW("f-k4-pf3", "f-k4 with every DPAS of a k-step made to wait for its loads", 2, 8, 7, 7, 0, 3, 1u, all, all, all);
  FRAG_ROW("f-k4-a", "A' as block reads of 128 bytes, B in registers", 2, 8, 7, 0, 0, 0, 1u, all, all, all);
  FRAG_ROW("f-k4-b", "B' as block reads of 128 bytes, A in registers", 2, 8, 0, 7, 0, 0, 1u, all, all, all);
  FRAG_ROW("f-k8", "both operands as one block read of 256 bytes a fragment", 2, 8, 8, 8, 0, 0, 1u, all, all, all);
  FRAG_ROW("f-k8-pf3", "f-k8 with every DPAS of a k-step made to wait for its loads", 2, 8, 8, 8, 0, 3, 1u, all, all, all);
  FRAG_ROW("f-k8-a", "A' as one block read of 256 bytes a fragment, B in registers", 2, 8, 8, 0, 0, 0, 1u, all, all, all);
  FRAG_ROW("f-k8-b", "B' as one block read of 256 bytes a fragment, A in registers", 2, 8, 0, 8, 0, 0, 1u, all, all, all);
  FRAG_ROW("f-k84-pf3", "A' as one block read of 256 bytes, B' as two of 128, every DPAS waiting", 2, 8, 8, 7, 0, 3, 1u, all, all, all);
  FRAG_ROW("f-k48-pf3", "A' as two block reads of 128 bytes, B' as one of 256, every DPAS waiting", 2, 8, 7, 8, 0, 3, 1u, all, all, all);
  FRAG_ROW("f-k8-pf3-fold", "f-k8-pf3 with the readout, barrier, hash and test", 2, 8, 8, 8, 2, 3, 1u, all, all, all);
  FRAG_ROW("f-k4-pf3-fold", "f-k4-pf3 with the readout, barrier, hash and test", 2, 8, 7, 7, 2, 3, 1u, all, all, all);
  FRAG_ROW("f-k8-pf8", "one block read of 256 bytes a fragment, all ten sent first, taken as dwords after", 2, 8, 8, 8, 0, 8, 1u, all, all, all);
  FRAG_ROW("f-k84-pf8", "A' as one block read of 256 bytes a fragment, B' as two of 128, all sent first", 2, 8, 8, 7, 0, 8, 1u, all, all, all);
  FRAG_ROW("f-k48-pf8", "A' as two block reads of 128 bytes a fragment, B' as one of 256, all sent first", 2, 8, 7, 8, 0, 8, 1u, all, all, all);
  FRAG_ROW("f-k8-a-pf8", "A' as one block read of 256 bytes a fragment, sent first; B in registers", 2, 8, 8, 0, 0, 8, 1u, all, all, all);
  FRAG_ROW("f-k8-b-pf8", "B' as one block read of 256 bytes a fragment, sent first; A in registers", 2, 8, 0, 8, 0, 8, 1u, all, all, all);
  FRAG_ROW("g-k8-pf8", "32 by 32, one block read of 256 bytes a fragment, all eight sent first", 4, 4, 8, 8, 0, 8, 1u, all, all, all);
  FRAG_ROW("f-v8k4-pf3", "A' as one vector of 8 dwords a fragment, B' as two block reads of 128 bytes, every DPAS waiting", 2, 8, 1, 7, 0, 3, 1u, all, all, all);
  FRAG_ROW("f-v8k4-pf3-fold", "f-v8k4-pf3 with the readout, barrier, hash and test", 2, 8, 1, 7, 2, 3, 1u, all, all, all);
  FRAG_ROW("f-v4k4-pf3", "A' as two vectors of 4 dwords a fragment, B' as two block reads of 128 bytes, every DPAS waiting", 2, 8, 2, 7, 0, 3, 1u, all, all, all);
  FRAG_ROW("f-k4-pf11", "f-k4-pf3 with the prefetch, touch and spin switches", 2, 8, 7, 7, 0, 11, 1u, all, all, all);
  FRAG_ROW("f-v4-pf10", "both operands as two vectors of 4 dwords, B' sent first, every DPAS waiting", 2, 8, 2, 2, 0, 10, 1u, all, all, all);
  FRAG_ROW("f-v84-pf10", "A' one vector of 8 dwords, B' two of 4, B' sent first, every DPAS waiting", 2, 8, 1, 2, 0, 10, 1u, all, all, all);
  FRAG_ROW("f-k4-pf10", "both operands as block reads of 128 bytes, B' sent first, every DPAS waiting", 2, 8, 7, 7, 0, 10, 1u, all, all, all);
  FRAG_ROW("f-v4k4-pf10", "A' as two vectors of 4 dwords, B' as block reads of 128 bytes, B' sent first", 2, 8, 2, 7, 0, 10, 1u, all, all, all);
  FRAG_ROW("f-k4v4-pf3", "A' as block reads of 128 bytes, B' as two vectors of 4 dwords, every DPAS waiting", 2, 8, 7, 2, 0, 3, 1u, all, all, all);
  FRAG_ROW("f-k4-pf12", "16 by 64, block reads, a chunk filled and met at a barrier before its 4 k-steps", 2, 8, 7, 7, 0, 12, 1u, all, all, all);
  FRAG_ROW("g-k4-pf12", "32 by 32, block reads, a chunk filled and met at a barrier before its 4 k-steps", 4, 4, 7, 7, 0, 12, 1u, all, all, all);
  FRAG_ROW("f-k4-keep", "20 block reads of 128 bytes a k-step only kept, DPAS on register operands", 2, 8, 7, 7, 0, 9, 1u, all, all, all);
  FRAG_ROW("f-k8-keep", "10 block reads of 256 bytes a k-step only kept, DPAS on register operands", 2, 8, 8, 8, 0, 9, 1u, all, all, all);
  FRAG_ROW("f-k8-keep-b", "8 block reads of 256 bytes a k-step only kept, DPAS on register operands", 2, 8, 0, 8, 0, 9, 1u, all, all, all);
  FRAG_ROW("f-k4-keep-b", "16 block reads of 128 bytes a k-step only kept, DPAS on register operands", 2, 8, 0, 7, 0, 9, 1u, all, all, all);
  FRAG_ROW("f-k4-keep-a", "4 block reads of 128 bytes a k-step only kept, DPAS on register operands", 2, 8, 7, 0, 0, 9, 1u, all, all, all);
  FRAG_ROW("g-k4", "32 by 32, two block reads of 128 bytes a fragment", 4, 4, 7, 7, 0, 0, 1u, all, all, all);
  FRAG_ROW("g-k4-pf3", "g-k4 with every DPAS of a k-step made to wait for its loads", 4, 4, 7, 7, 0, 3, 1u, all, all, all);
  FRAG_ROW("g-k8", "32 by 32, one block read of 256 bytes a fragment", 4, 4, 8, 8, 0, 0, 1u, all, all, all);
  FRAG_ROW("g-k8-pf3", "g-k8 with every DPAS of a k-step made to wait for its loads", 4, 4, 8, 8, 0, 3, 1u, all, all, all);
#undef FRAG_ROW
  // Operands staged through shared local memory (PeakSlm), --sga by --sgb sub-groups a work-group (default
  // 4 by 4). PPS is sized for 4 by 4; a shape that needs more pieces a sub-group fails with a message.
#define SLM_ROW(NAME, WHAT, OPS, MODE, D, BF, PPS, KS)                                           \
  v.push_back({NAME, WHAT, 16, 16, 16, 1u, all, all, all,                                       \
               [](sycl::queue &q, const PeakArgs &a, uint32_t g) {                              \
                 return peak_slm<OPS, MODE, D, BF, PPS, KS>(q, a, g);                           \
               }})
  SLM_ROW("s-ceil", "both operands only read from local memory, filled once (B' halves)", 3, 0, 1, 2, 5, 1);
  SLM_ROW("s-ceil-v8", "s-ceil with B' read as one vector of 8 dwords", 3, 0, 1, 1, 5, 1);
  SLM_ROW("s-ceil-a", "A' only read from local memory, filled once; B' from its own addresses", 1, 0, 1, 2, 1, 1);
  SLM_ROW("s-ceil-b", "B' only read from local memory, filled once; A' from its own addresses", 2, 0, 1, 2, 4, 1);
  SLM_ROW("s-ceil-b-v8", "s-ceil-b with B' read as one vector of 8 dwords", 2, 0, 1, 1, 4, 1);
  SLM_ROW("s-pipe", "both through a ring in local memory, copied a k-step ahead, a barrier a k-step", 3, 1, 1, 2, 5, 1);
  SLM_ROW("s-pipe-b", "B' through the ring, A' from its own addresses", 2, 1, 1, 2, 4, 1);
  SLM_ROW("s-pipe-a", "A' through the ring a k-step a stage, B' from its own addresses", 1, 1, 1, 2, 1, 1);
  SLM_ROW("s-pipe-a2", "s-pipe-a, 2 k-steps a stage", 1, 1, 1, 2, 2, 2);
  SLM_ROW("s-pipe-a4", "s-pipe-a, 4 k-steps a stage (a chunk)", 1, 1, 1, 2, 4, 4);
  SLM_ROW("s-pipe-a8", "s-pipe-a, 8 k-steps a stage", 1, 1, 1, 2, 8, 8);
  SLM_ROW("s-pipe-a4-d2", "s-pipe-a4, two stages ahead", 1, 1, 2, 2, 4, 4);
  SLM_ROW("s-pipe-a4w", "s-pipe-a4 with room for 8 pieces a sub-group (--sga 8 --sgb 2)", 1, 1, 1, 2, 8, 4);
  SLM_ROW("s-bar-a", "s-pipe-a without the copy: the barrier alone", 1, 2, 1, 2, 1, 1);
  SLM_ROW("s-bar-a4", "s-pipe-a4 without the copy", 1, 2, 1, 2, 4, 4);
  SLM_ROW("s-copy-a", "s-pipe-a without the barrier: the copy alone", 1, 3, 1, 2, 1, 1);
  SLM_ROW("s-copy-a4", "s-pipe-a4 without the barrier", 1, 3, 1, 2, 4, 4);
  SLM_ROW("s-pin-a4", "s-pipe-a4 with the copy's loads pinned", 1, 6, 1, 2, 4, 4);
  SLM_ROW("s-pin-a4-d2", "s-pin-a4 two stages ahead", 1, 6, 2, 2, 4, 4);
  SLM_ROW("s-pinc-a4", "s-pin-a4 without the barrier", 1, 7, 1, 2, 4, 4);
  SLM_ROW("s-split-a", "A' through the ring, pinned, split barrier, a k-step a stage", 1, 4, 2, 2, 1, 1);
  SLM_ROW("s-split-a2", "s-split-a, 2 k-steps a stage", 1, 4, 2, 2, 2, 2);
  SLM_ROW("s-split-a4", "s-split-a, 4 k-steps a stage", 1, 4, 2, 2, 4, 4);
  SLM_ROW("s-sbar-a", "s-split-a without the copy: the split barrier alone", 1, 5, 2, 2, 1, 1);
  SLM_ROW("s-sbar-a4", "s-split-a4 without the copy", 1, 5, 2, 2, 4, 4);
  SLM_ROW("s-split", "both operands through the ring, pinned, split barrier, a k-step a stage", 3, 4, 2, 2, 5, 1);
#undef SLM_ROW
  // The same at 32 rows by 32 columns a sub-group ("t-"): 4 fragments of each operand a k-step, so a k-step
  // sends 12 messages (4 of A', 8 half fragments of B') where the 16 by 64 tile sends 18.
#define SLM_ROW(NAME, WHAT, OPS, MODE, D, BF, PPS, KS)                                           \
  v.push_back({NAME, WHAT, 16, 16, 16, 1u, all, all, all,                                       \
               [](sycl::queue &q, const PeakArgs &a, uint32_t g) {                              \
                 return peak_slm<OPS, MODE, D, BF, PPS, KS, 1>(q, a, g);                        \
               }})
  SLM_ROW("t-ceil-a", "32 by 32: A' only read from local memory, filled once; B' from its own addresses", 1, 0, 1, 2, 2, 1);
  SLM_ROW("t-ceil-b", "32 by 32: B' only read from local memory, filled once; A' from its own addresses", 2, 0, 1, 2, 2, 1);
  SLM_ROW("t-ceil", "32 by 32: both only read from local memory, filled once", 3, 0, 1, 2, 4, 1);
  SLM_ROW("t-ceil-v8", "t-ceil with B' read as one vector of 8 dwords", 3, 0, 1, 1, 4, 1);
  SLM_ROW("t-ceil-b-v8", "t-ceil-b with B' read as one vector of 8 dwords", 2, 0, 1, 1, 2, 1);
  SLM_ROW("t-split-v8", "t-split with B' read as one vector of 8 dwords", 3, 4, 2, 1, 4, 1);
  SLM_ROW("t-split-b-v8", "t-split-b with B' read as one vector of 8 dwords", 2, 4, 2, 1, 2, 1);
  SLM_ROW("t-free", "32 by 32, both through the ring (8-dword reads), copy loads at the top and stores after the DPAS, no pin, split barrier", 3, 16, 2, 1, 4, 1);
  SLM_ROW("t-free-b", "t-free with B' only through the ring", 2, 16, 2, 1, 2, 1);
  SLM_ROW("t-free-a", "t-free with A' only through the ring", 1, 16, 2, 1, 2, 1);
  SLM_ROW("t-freec", "t-free without the barrier", 3, 3, 2, 1, 4, 1);
  SLM_ROW("t-sbar-v8", "t-free without the copy", 3, 5, 2, 1, 4, 1);
  SLM_ROW("t-pinc-a", "32 by 32: A' copied into the ring a k-step ahead, pinned, no barrier", 1, 7, 1, 2, 2, 1);
  SLM_ROW("t-split-a", "32 by 32: A' through the ring, pinned, split barrier, a k-step a stage", 1, 4, 2, 2, 2, 1);
  SLM_ROW("t-split-a2", "t-split-a, 2 k-steps a stage", 1, 4, 2, 2, 4, 2);
  SLM_ROW("t-split-b", "32 by 32: B' through the ring, pinned, split barrier; A' from its own addresses", 2, 4, 2, 2, 2, 1);
  SLM_ROW("t-split", "32 by 32: both through the ring, pinned, split barrier", 3, 4, 2, 2, 4, 1);
  SLM_ROW("t-sbar", "t-split without the copy: the split barrier alone", 3, 5, 2, 2, 4, 1);
  SLM_ROW("t-sbar-a", "t-split-a without the copy", 1, 5, 2, 2, 2, 1);
  SLM_ROW("t-carry-a", "32 by 32: A' through a ring of 3, each stage's loads stored a stage later, split barrier", 1, 8, 2, 2, 2, 1);
  SLM_ROW("t-carryc-a", "t-carry-a without the barrier", 1, 9, 2, 2, 2, 1);
  SLM_ROW("t-carry-a2", "t-carry-a, 2 k-steps a stage", 1, 8, 2, 2, 4, 2);
  SLM_ROW("t-carry", "32 by 32: both operands through the ring, carried, split barrier", 3, 8, 2, 2, 4, 1);
  SLM_ROW("t-ld-a", "t-pinc-a with only the copy's loads (kept, not stored)", 1, 10, 1, 2, 2, 1);
  SLM_ROW("t-st-a", "t-pinc-a with only the copy's stores (of registers)", 1, 11, 1, 2, 2, 1);
  SLM_ROW("t-ldc-a", "t-ld-a with every load from one place that stays in the cache", 1, 12, 1, 2, 2, 1);
  SLM_ROW("t-pinc-a2", "t-pinc-a, 2 k-steps a stage, the copy pinned to the second", 1, 7, 1, 2, 2, 2);
  SLM_ROW("t-pinc-a4", "t-pinc-a, 4 k-steps a stage, the copy pinned to the fourth", 1, 7, 1, 2, 4, 4);
  SLM_ROW("t-split-a4", "t-split-a, 4 k-steps a stage", 1, 4, 2, 2, 4, 4);
  SLM_ROW("t-split-a2p", "t-split-a, 2 k-steps a stage, room for 2 pieces", 1, 4, 2, 2, 2, 2);
  SLM_ROW("t-sbar-a4", "t-split-a4 without the copy", 1, 5, 2, 2, 4, 4);
  SLM_ROW("u-prod-a", "32 by 32, A' through the ring, copied by --sgp producer sub-groups that run no DPAS, split barrier", 1, 13, 2, 2, 8, 1);
  SLM_ROW("u-prodc-a", "u-prod-a with each producer load stored a stage later", 1, 15, 2, 2, 8, 1);
  SLM_ROW("u-idle-a", "u-prod-a with producers that copy nothing (their threads and the barrier alone)", 1, 14, 2, 2, 8, 1);
  SLM_ROW("u-prod-a16", "u-prod-a with room for 16 pieces a producer", 1, 13, 2, 2, 16, 1);
  SLM_ROW("u-prodc-a16", "u-prodc-a with room for 16 pieces a producer", 1, 15, 2, 2, 16, 1);
#undef SLM_ROW

#if defined(PEARL_SYCL_XMX_JIT)
  // Operands read as texels of an image: the sampler, or whatever message a read with no sampler becomes.
#define IMG_ROW(NAME, WHAT, LA, LB, PF, SMP)                                                    \
  v.push_back({NAME, WHAT, 16, 16, 16, 1u, all, all, all,                                       \
               [](sycl::queue &q, const PeakArgs &a, uint32_t g) {                              \
                 return peak_img<LA, LB, PF, SMP>(q, a, g);                                     \
               }})
  IMG_ROW("i-regs", "the image rows' kernel with every operand in registers", 0, 0, 4, 0);
  IMG_ROW("i-base", "A' one vector of 8 dwords, B' two of 4, from memory, every DPAS waiting (f-v84-pf3 again)", 1, 2, 3, 0);
  IMG_ROW("i-b-smp", "B' as 16 sampler fetches a k-step, A' from memory, every DPAS waiting", 1, 9, 3, 1);
  IMG_ROW("i-b-rd", "B' as 16 image reads with no sampler, A' from memory, every DPAS waiting", 1, 9, 3, 0);
  IMG_ROW("i-keep-b-smp", "16 sampler fetches a k-step only kept, DPAS on register operands", 0, 9, 4, 1);
  IMG_ROW("i-keep-b-rd", "16 image reads with no sampler a k-step only kept, DPAS on register operands", 0, 9, 4, 0);
  IMG_ROW("i-keep-b-mem", "16 half-fragment loads from memory a k-step only kept, DPAS on register operands", 0, 2, 4, 0);
  IMG_ROW("i-keep-a-mem", "4 half-fragment loads of A' from memory a k-step only kept, DPAS on register operands", 2, 0, 4, 0);
  IMG_ROW("i-b1-smp", "1 of the 8 B' fragments a k-step by the sampler (2 fetches), the rest and A' from memory", 1, 10, 3, 1);
  IMG_ROW("i-b2-smp", "2 of the 8 B' fragments a k-step by the sampler (4 fetches), the rest and A' from memory", 1, 11, 3, 1);
  IMG_ROW("i-b4-smp", "4 of the 8 B' fragments a k-step by the sampler (8 fetches), the rest and A' from memory", 1, 12, 3, 1);
  IMG_ROW("i-keep-b1-smp", "i-keep-b-mem with 1 of the 8 fragments by the sampler", 0, 10, 4, 1);
  IMG_ROW("i-keep-b2-smp", "i-keep-b-mem with 2 of the 8 fragments by the sampler", 0, 11, 4, 1);
  IMG_ROW("i-keep-b4-smp", "i-keep-b-mem with 4 of the 8 fragments by the sampler", 0, 12, 4, 1);
  IMG_ROW("i-a4-b1-smp", "i-b1-smp with A' from memory as halves (4 loads)", 2, 10, 3, 1);
  IMG_ROW("i-a4-b2-smp", "i-b2-smp with A' from memory as halves (4 loads)", 2, 11, 3, 1);
  IMG_ROW("i-a4", "A' and B' from memory, both as halves (f-v4-pf3 again)", 2, 2, 3, 0);
  IMG_ROW("i-b1-rd", "i-b1-smp with the image read with no sampler", 1, 10, 3, 0);
  IMG_ROW("i-b2-rd", "i-b2-smp with the image read with no sampler", 1, 11, 3, 0);
  IMG_ROW("i-ab-smp2", "A' by the sampler (4 fetches, at B' coordinates), B' from memory in halves", 13, 2, 3, 1);
  IMG_ROW("i-ab1-smp2", "A' by the sampler (at B' coordinates) and 1 of the 8 B' fragments too", 13, 10, 3, 1);
  IMG_ROW("i-b3-rd", "3 of the 8 B' fragments a k-step as image reads with no sampler", 1, 14, 3, 0);
  IMG_ROW("i-b4-rd", "4 of the 8", 1, 12, 3, 0);
  IMG_ROW("i-b6-rd", "6 of the 8", 1, 15, 3, 0);
  IMG_ROW("i-ab-rd2", "A' as image reads with no sampler (at B' coordinates), B' from memory in halves", 13, 2, 3, 0);
  IMG_ROW("i-ab2-rd2", "A' and 2 of the 8 B' fragments as image reads with no sampler", 13, 11, 3, 0);
  IMG_ROW("i-ab4-rd2", "A' and 4 of the 8 B' fragments as image reads with no sampler", 13, 12, 3, 0);
  IMG_ROW("j-b1", "1 of the 8 B' fragments a k-step fetched from a bindless image", 1, 10, 3, 2);
  IMG_ROW("j-b2", "2 of the 8", 1, 11, 3, 2);
  IMG_ROW("j-b3", "3 of the 8", 1, 14, 3, 2);
  IMG_ROW("j-b4", "4 of the 8", 1, 12, 3, 2);
  IMG_ROW("j-b", "all 8", 1, 9, 3, 2);
  IMG_ROW("j-a4-b2", "j-b2 with A' from memory as halves", 2, 11, 3, 2);
  IMG_ROW("j-ab", "A' fetched from the bindless image (at B' coordinates), B' from memory in halves", 13, 2, 3, 2);
  IMG_ROW("j-ab2", "A' and 2 of the 8 B' fragments fetched from the bindless image", 13, 11, 3, 2);
  IMG_ROW("j-a", "A' fetched from the bindless image at its own blocks' texels, B' from memory in halves", 9, 2, 3, 2);
  IMG_ROW("j-a-b1", "j-a with 1 of the 8 B' fragments fetched too", 9, 10, 3, 2);
  IMG_ROW("j-a-b2", "j-a with 2 of the 8 B' fragments fetched too", 9, 11, 3, 2);
  IMG_ROW("j-keep-a", "4 fetches of A' a k-step only kept, DPAS on register operands", 9, 0, 4, 2);
  IMG_ROW("j-keep-b2", "4 fetches of B' a k-step and 12 half loads only kept, DPAS on register operands", 0, 11, 4, 2);
#undef IMG_ROW
#endif
}

bool read_num(const std::string &path, double &v) {
  if (path.empty()) return false;
  std::ifstream f(path);
  return static_cast<bool>(f >> v);
}

// The first file under dir/*/ that is named `name`, or whose name starts with
// `prefix` and ends with `suffix`.
std::string find_file(const std::string &dir, const std::string &name, const std::string &prefix = "",
                      const std::string &suffix = "") {
  namespace fs = std::filesystem;
  std::error_code ec;
  for (const auto &sub : fs::directory_iterator(dir, ec)) {
    if (!sub.is_directory(ec)) continue;
    for (const auto &f : fs::directory_iterator(sub.path(), ec)) {
      const std::string n = f.path().filename().string();
      if (n == name) return f.path().string();
      if (!prefix.empty() && n.rfind(prefix, 0) == 0 && n.size() >= suffix.size()
          && n.compare(n.size() - suffix.size(), suffix.size(), suffix) == 0)
        return f.path().string();
    }
  }
  return "";
}

// The card's counters in sysfs, by its PCI address. Any may be missing.
struct Sensors {
  std::string energy, freq, temp, fan;
  void find(const std::string &pci) {
    if (pci.empty()) return;
    const std::string base = "/sys/bus/pci/devices/" + pci;
    freq = find_file(base + "/drm", "gt_act_freq_mhz");
    energy = find_file(base + "/hwmon", "energy1_input");
    temp = find_file(base + "/hwmon", "temp1_input", "temp", "_input");
    fan = find_file(base + "/hwmon", "fan1_input");
  }
};

// Reads the clock a few times a second while a row runs, and the energy
// counter at both ends.
struct Sampler {
  const Sensors &s;
  std::atomic<bool> stop{false};
  std::thread th;
  double mhzSum = 0, tempSum = 0, fanSum = 0, e0 = 0, e1 = 0;
  int mhzN = 0, tempN = 0, fanN = 0;
  bool haveE = false;
  std::chrono::steady_clock::time_point t0, t1;
  explicit Sampler(const Sensors &sensors) : s(sensors) {}
  void start() {
    haveE = read_num(s.energy, e0);
    t0 = std::chrono::steady_clock::now();
    th = std::thread([this]() {
      while (!stop.load()) {
        double v;
        if (read_num(s.freq, v)) { mhzSum += v; mhzN++; }
        if (read_num(s.temp, v)) { tempSum += v; tempN++; }
        if (read_num(s.fan, v)) { fanSum += v; fanN++; }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
      }
    });
  }
  void finish() {
    stop.store(true);
    th.join();
    haveE = haveE && read_num(s.energy, e1);
    t1 = std::chrono::steady_clock::now();
  }
  double watts() const {  // the counter is in microjoules
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    return haveE && secs > 0 ? (e1 - e0) / 1e6 / secs : -1;
  }
  double mhz() const { return mhzN ? mhzSum / mhzN : -1; }
  double tempC() const { return tempN ? tempSum / tempN / 1000.0 : -1; }
  double fan() const { return fanN ? fanSum / fanN : -1; }
};

uint32_t floor_pow2(uint64_t v) {
  uint32_t p = 1;
  while ((uint64_t)p * 2 <= v) p *= 2;
  return p;
}

}  // namespace

int main(int argc, char **argv) {
  PearlProfile p = PEARL_MAINNET_PROFILE;
  double secs = 5;
  uint32_t reps = 16, bandArg = 0, colBatch = 256;
  uint32_t aStride = 0, bStride = 0, laneStride = 32, wpg = 1, stagger = 0, aMix = 0, bMix = 0, spin = 0, pfA = 0, pfB = 0, pfHint = 0, touchA = 0, touchB = 0, fillSkip = 0, sga = 4, sgb = 4, sgp = 2;
  uint32_t maskArg[3] = {0, 0, 0};
  bool maskSet[3] = {false, false, false};
  int sgArg = 0, rbArg = 0;
  bool trace = false, frag = false;
  std::string only;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    const char *v = i + 1 < argc ? argv[i + 1] : nullptr;
    if (a == "--secs" && v) { secs = atof(v); i++; }
    else if (a == "--only" && v) { only = std::string(",") + v + ","; i++; }
    else if (a == "--reps" && v) { reps = (uint32_t)atoi(v); i++; }
    else if (a == "--m" && v) { p.m = (uint32_t)atoi(v); i++; }
    else if (a == "--n" && v) { p.n = (uint32_t)atoi(v); i++; }
    else if (a == "--col-batch" && v) { colBatch = (uint32_t)atoi(v); i++; }
    else if (a == "--band" && v) { bandArg = (uint32_t)atoi(v); i++; }
    else if (a == "--sg" && v) { sgArg = atoi(v); i++; }
    else if (a == "--rb" && v) { rbArg = atoi(v); i++; }
    else if (a == "--trace") { trace = true; }
    else if (a == "--frag") { frag = true; }
    else if (a == "--a-stride" && v) { aStride = (uint32_t)atoi(v); i++; }
    else if (a == "--b-stride" && v) { bStride = (uint32_t)atoi(v); i++; }
    else if (a == "--lane-stride" && v) { laneStride = (uint32_t)atoi(v); i++; }
    else if (a == "--wpg" && v) { wpg = (uint32_t)atoi(v); i++; }
    else if (a == "--stagger" && v) { stagger = (uint32_t)atoi(v); i++; }
    else if (a == "--a-mix" && v) { aMix = (uint32_t)atoi(v); i++; }
    else if (a == "--b-mix" && v) { bMix = (uint32_t)atoi(v); i++; }
    else if (a == "--spin" && v) { spin = (uint32_t)atoi(v); i++; }
    else if (a == "--pfa" && v) { pfA = (uint32_t)atoi(v); i++; }
    else if (a == "--pfb" && v) { pfB = (uint32_t)atoi(v); i++; }
    else if (a == "--pf-hint" && v) { pfHint = (uint32_t)atoi(v); i++; }
    else if (a == "--touch-a" && v) { touchA = (uint32_t)atoi(v); i++; }
    else if (a == "--fill-skip" && v) { fillSkip = (uint32_t)atoi(v); i++; }
    else if (a == "--touch-b" && v) { touchB = (uint32_t)atoi(v); i++; }
    else if (a == "--sga" && v) { sga = (uint32_t)atoi(v); i++; }
    else if (a == "--sgb" && v) { sgb = (uint32_t)atoi(v); i++; }
    else if (a == "--sgp" && v) { sgp = (uint32_t)atoi(v); i++; }
    else if (a == "--row-mask" && v) { maskArg[0] = (uint32_t)strtoul(v, nullptr, 0); maskSet[0] = true; i++; }
    else if (a == "--col-mask" && v) { maskArg[1] = (uint32_t)strtoul(v, nullptr, 0); maskSet[1] = true; i++; }
    else if (a == "--k-mask" && v) { maskArg[2] = (uint32_t)strtoul(v, nullptr, 0); maskSet[2] = true; i++; }
    else {
      fprintf(stderr, "usage: %s [--secs S] [--only LIST] [--reps R] [--trace] [--m M] [--n N] "
                      "[--col-batch C] [--band B] [--sg 8|16] [--rb 2]\n"
                      "       %s --frag [--a-stride BYTES] [--b-stride BYTES] [--row-mask M] [--col-mask M] "
                      "[--k-mask M] [--wpg W] [--stagger N] [--a-mix N] [--b-mix N] [--fill-skip 1] [--pfa D] "
                      "[--pfb D] [--pf-hint H] [--touch-a D] [--touch-b D] [--spin N] and the options above\n",
              argv[0], argv[0]);
      return 2;
    }
  }
  if (reps < 1) reps = 1;

  // The device: Intel GPUs through Level Zero, else through OpenCL, in the
  // runtime's order, or the CPU device.
  std::vector<sycl::device> devs;
  const char *want = getenv("PEARL_SYCL_DEVICE");
  const bool cpu = want && !strcmp(want, "cpu");
  try {
    if (cpu) {
      devs = sycl::device::get_devices(sycl::info::device_type::cpu);
    } else {
      for (const auto be : {sycl::backend::ext_oneapi_level_zero, sycl::backend::opencl}) {
        for (const auto &d : sycl::device::get_devices(sycl::info::device_type::gpu))
          if (d.get_backend() == be && d.get_info<sycl::info::device::vendor_id>() == 0x8086) devs.push_back(d);
        if (!devs.empty()) break;
      }
    }
  } catch (const std::exception &e) {
    fprintf(stderr, "device list: %s\n", e.what());
    return 2;
  }
  size_t index = 0;
  if (const char *g = getenv("PEARL_GPU_INDEX")) index = (size_t)atoi(g);
  if (index >= devs.size()) {
    fprintf(stderr, "no device %zu: %zu found\n", index, devs.size());
    return 2;
  }
  sycl::device dev = devs[index];
  const std::string name = dev.get_info<sycl::info::device::name>();
  const auto sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
  auto has_sg = [&](size_t s) {
    for (size_t x : sizes)
      if (x == s) return true;
    return false;
  };
  const int SG = sgArg ? sgArg : (has_sg(8) && !cpu ? 8 : 16);
  if ((SG != 8 && SG != 16) || !has_sg((size_t)SG)) {
    fprintf(stderr, "%s has no sub-group size %d\n", name.c_str(), SG);
    return 2;
  }
  uint32_t eus = 0, threadsPerEu = 0;
  std::string pci;
  try {
    if (dev.has(sycl::aspect::ext_intel_gpu_eu_count))
      eus = dev.get_info<sycl::ext::intel::info::device::gpu_eu_count>();
    if (dev.has(sycl::aspect::ext_intel_gpu_hw_threads_per_eu))
      threadsPerEu = dev.get_info<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>();
    if (dev.has(sycl::aspect::ext_intel_pci_address))
      pci = dev.get_info<sycl::ext::intel::info::device::pci_address>();
  } catch (const std::exception &) {
  }
  Sensors sensors;
  if (const char *forced = getenv("PEARL_PEAK_PCI")) pci = forced;
  sensors.find(pci);

  sycl::queue q(dev, sycl::property::queue::in_order());
  const uint32_t k = p.k;
  const uint32_t rowsValid = p.m / 16, colsValid = p.n / 16;
  if (colBatch < 4 || colBatch > colsValid) colBatch = colsValid;
  uint64_t cache = 0;
  try {
    cache = dev.get_info<sycl::info::device::global_mem_cache_size>();
  } catch (const std::exception &) {
  }
  uint64_t band = bandArg ? bandArg : (cache ? cache / (32ull * k) : 16);  // the host's rule for the XMX folds
  if (frag && !bandArg && cache) band = cache / 4 / (32ull * k);           // and for the fragment fold
  if (band < 1) band = 1;
  if (band > rowsValid / 2u) band = rowsValid / 2u;
  band = floor_pow2(band);
  const uint32_t groups = (rowsValid / 2u) * (colBatch / 4u);
  const uint32_t batches = colsValid / colBatch;

  fprintf(stderr, "device: %s (%s)%s%s; %u EUs, %u threads an EU; sub-group %d; cache %llu KB\n", name.c_str(),
          dev.get_backend() == sycl::backend::opencl ? "opencl" : "level_zero", pci.empty() ? "" : " at ",
          pci.c_str(), eus, threadsPerEu, SG, (unsigned long long)(cache >> 10));
  fprintf(stderr, "grid: m %u n %u k %u, %u column offsets a batch, band %u, %u work-groups of 32, %u s a row\n",
          p.m, p.n, k, colBatch, (uint32_t)band, groups, (unsigned)secs);
  fprintf(stderr, "sensors: clock %s, energy %s, temperature %s, fan %s\n", sensors.freq.empty() ? "none" : "yes",
          sensors.energy.empty() ? "none" : "yes", sensors.temp.empty() ? "none" : "yes",
          sensors.fan.empty() ? "none" : "yes");

  // Operands: uniform int8 in [-64, 63], about what the miner's are.
  // The fragment rows' blocks: 16 rows of A', 64 columns of B', or further apart.
  if (laneStride < 32) laneStride = 32;
  if (aStride < (k / 32u) * 16u * laneStride) aStride = (k / 32u) * 16u * laneStride;
  if (bStride < (k / 32u) * 64u * laneStride) bStride = (k / 32u) * 64u * laneStride;
  const size_t aBytes = (size_t)(p.m / 16u) * aStride, bBytes = (size_t)(p.n / 64u) * bStride;
  int8_t *dA = sycl::malloc_device<int8_t>(aBytes, q);
  int8_t *dB = sycl::malloc_device<int8_t>(bBytes, q);
  HitBuf *hits = sycl::malloc_device<HitBuf>(1, q);
  uint32_t *sink = sycl::malloc_device<uint32_t>((size_t)groups * 4, q);
  if (!dA || !dB || !hits || !sink) {
    fprintf(stderr, "out of device memory (%zu MB of operands)\n", (aBytes + bBytes) >> 20);
    return 2;
  }
  try {
    for (int which = 0; which < 2; which++) {
      uint32_t *out = reinterpret_cast<uint32_t *>(which ? dB : dA);
      const size_t words = (which ? bBytes : aBytes) / 4;
      const uint32_t salt = which ? 0x51ED270Bu : 0x9E3779B9u;
      q.parallel_for(sycl::range<1>(words), [=](sycl::id<1> i) { out[i[0]] = bytes7(mix((uint32_t)i[0] ^ salt)); });
    }
    q.memset(hits, 0, sizeof(HitBuf));
    q.wait_and_throw();
  } catch (const std::exception &e) {
    fprintf(stderr, "fill: %s\n", e.what());
    return 2;
  }

  PeakArgs args{};
  args.f.Ap = dA;
  args.f.Bp = dB;
  args.f.k = k;
  args.f.rowsValid = rowsValid;
  args.f.colStart = 0;
  args.f.colBatch = colBatch;
  args.f.band = (uint32_t)band;
  args.f.aStride = aStride;
  args.f.bStride = bStride;
  args.f.zero = 0;
  for (int i = 0; i < 8; i++) { args.f.test.key[i] = mix(0xA5EEDu + (uint32_t)i); args.f.test.target_w[i] = 0; }
  args.f.test.hash_big_endian = 0;
  args.f.hits = hits;
  args.f.allTr = nullptr;
  args.seed = 0x1234567u;
  args.zero = 0;
  args.laneStride = laneStride;
  // Windows a work-group: a divisor of the band, so a group stays in one column window, or a multiple of it
  // that divides a band's walk, so a group is whole column windows of one band.
  if (wpg < 1 || ((band % wpg) && ((wpg % band) || ((band * (colBatch / 4u)) % wpg)))) wpg = 1;
  args.wpg = wpg;
  args.stagger = stagger;
  args.aMix = aMix;
  args.bMix = bMix;
  args.spin = spin;
  args.pfA = pfA;
  args.pfB = pfB;
  args.pfHint = pfHint;
  args.touchA = touchA;
  args.fillSkip = fillSkip;
  args.touchB = touchB;
  // The "s-" rows' work-group: an even number of 16-row blocks, whole row windows of a band, whole column
  // windows of a batch, at most 64 sub-groups (an Xe-core's threads in this mode). A row whose ring would
  // pass 64 KB at this shape fails with a message.
  if (sga < 2 || (sga & 1u) || ((rowsValid / 2u) % (sga / 2u)) || sgb < 1 || ((colBatch / 4u) % sgb)
      || sga * sgb > 64u) {
    if (sga != 4 || sgb != 4) fprintf(stderr, "--sga %u --sgb %u does not fit: 4 by 4 instead\n", sga, sgb);
    sga = 4;
    sgb = 4;
  }
  args.sga = sga;
  args.sgb = sgb;
  args.sgp = sgp < 1 ? 1 : sgp;
  args.sink = sink;

  std::vector<Row> rows;
  if (frag) {
    if (SG != 8 || cpu) {
      fprintf(stderr, "--frag is the sub-group 8 hardware fold's: an Arc A-series card\n");
      return 2;
    }
    fprintf(stderr, "fragment rows: A' blocks %u bytes apart, B' blocks %u, a lane's 32 bytes every %u, %u windows a work-group\n",
            aStride, bStride, laneStride, wpg);
    add_frag_rows(rows);
#if defined(PEARL_SYCL_XMX_JIT)
    // The image rows' two images, made only when one of those rows is asked for by name.
    if (only.find(",i-") != std::string::npos || only.find(",j-") != std::string::npos) {
      try {
        // The images are left as the driver hands them out: a kernel that wrote the texels hung the card, and
        // an image made from bytes on the host crashed the runtime. What they hold does not matter to the clock.
        auto make = [&](size_t bytes, uint32_t &ymask, uint32_t) {
          uint64_t rowsWanted = (bytes / 16u + 4095u) / 4096u;
          if (rowsWanted > 16384u) rowsWanted = 16384u;
          const uint32_t h = floor_pow2(rowsWanted);
          ymask = h - 1u;
          return new sycl::image<2>(sycl::image_channel_order::rgba, sycl::image_channel_type::unsigned_int32,
                                    sycl::range<2>(4096, h));
        };
        gImg.a = make(aBytes, gImg.yMaskA, 0x9E3779B9u);
        gImg.b = make(bBytes, gImg.yMaskB, 0x51ED270Bu);
        // The bindless image: B's bytes as the operands are filled, copied in from host memory, then checked
        // by a kernel that fetches every texel back.
        if (only.find(",j-") != std::string::npos) {
          namespace sx = sycl::ext::oneapi::experimental;
          const uint32_t rowsB = gImg.yMaskB + 1u;
          const size_t words = (size_t)4096 * rowsB * 4u;
          const uint32_t salt = 0x51ED270Bu;
          std::vector<uint32_t> host(words);
          for (size_t i = 0; i < words; i++) host[i] = bytes7(mix((uint32_t)i ^ salt));
          sx::image_descriptor desc(sycl::range<2>{4096, rowsB}, 4, sycl::image_channel_type::unsigned_int32);
          const auto t0 = std::chrono::steady_clock::now();
          sx::image_mem_handle mem = sx::alloc_image_mem(desc, q);
          q.ext_oneapi_copy(host.data(), mem, desc);
          q.wait_and_throw();
          const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
          gImg.hb = sx::create_image(mem, desc, q);
          gImg.haveBindless = true;
          const sx::unsampled_image_handle hb = gImg.hb;
          uint32_t *bad = sycl::malloc_shared<uint32_t>(1, q);
          *bad = 0;
          q.parallel_for(sycl::range<2>(rowsB, 4096), sycl::reduction(bad, sycl::plus<uint32_t>()),
                         [=](sycl::id<2> i, auto &sum) {
            const uint32_t x = (uint32_t)i[1], y = (uint32_t)i[0];
            const uint32_t t = (y * 4096u + x) * 4u;
            const peak_texel v = sx::fetch_image<peak_texel>(hb, sycl::vec<int, 2>{(int)x, (int)y});
            uint32_t n = 0;
            for (uint32_t c = 0; c < 4u; c++) n += v[c] != bytes7(mix((t + c) ^ salt));
            sum += n;
          });
          q.wait_and_throw();
          fprintf(stderr, "bindless image: 4096 by %u texels (%u MB) copied from host memory in %.3f s; "
                          "%u of %zu words read back wrong\n", rowsB, rowsB >> 4, dt, *bad, words);
          sycl::free(bad, q);
        }
        fprintf(stderr, "image rows: A' image 4096 by %u texels (%u MB of %zu), B' image 4096 by %u (%u MB of %zu)\n",
                gImg.yMaskA + 1u, (gImg.yMaskA + 1u) >> 4, aBytes >> 20, gImg.yMaskB + 1u, (gImg.yMaskB + 1u) >> 4,
                bBytes >> 20);
      } catch (const std::exception &e) {
        fprintf(stderr, "images: %s\n", e.what());
        return 2;
      }
    }
#endif
  } else if (rbArg == 2 && SG == 8) {
#if defined(PEARL_PEAK_LARGE_GRF)
    add_rows<8, 2, false>(rows);
#else
    fprintf(stderr, "--rb 2 needs the 256-register build: ./build.sh --large-grf --peak\n");
    return 2;
#endif
  } else if (SG == 8) {
    add_rows<8, 1, true>(rows);
  } else {
    add_rows<16, 2, true>(rows);
  }

  int failed = 0;
  for (const Row &r : rows) {
    if (!only.empty() && only.find(std::string(",") + r.name + ",") == std::string::npos) continue;
#if defined(PEARL_SYCL_XMX_JIT)
    // The image rows run only when asked for by name (--only), which is when their images are made.
    if (r.name[1] == '-' && ((r.name[0] == 'i' && !gImg.a) || (r.name[0] == 'j' && !gImg.haveBindless))) continue;
#endif
    PeakArgs a = args;
    a.reps = r.reps ? r.reps : reps;
    a.rowMask = maskSet[0] ? maskArg[0] : r.rowMask;
    a.colMask = maskSet[1] ? maskArg[1] : r.colMask;
    a.kMask = maskSet[2] ? maskArg[2] : r.kMask;
    // MACs a launch: a window is 32 rows by 64 columns by 64 k-steps of 32;
    // `counted` of a sub-group's `all` DPAS are counted.
    const double macs = (double)groups * 32.0 * 64.0 * 64.0 * 32.0 * a.reps * r.counted / r.all;
    std::string err;
    uint32_t n = 0;
    double total = 0, best = 1e30, lastTrace = 0, traceMacs = 0;
    Sampler smp(sensors);
    try {
      r.run(q, a, groups);  // the first launch includes any JIT compile; it is dropped
      q.wait_and_throw();
      smp.start();
      while (total < secs) {
        a.f.colStart = (n % batches) * colBatch;
        const auto t0 = std::chrono::steady_clock::now();
        r.run(q, a, groups);
        q.wait_and_throw();
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        total += dt;
        if (dt < best) best = dt;
        n++;
        traceMacs += macs;
        if (trace && total - lastTrace >= 1.0) {
          double mhz = -1;
          read_num(sensors.freq, mhz);
          fprintf(stderr, "  %s %5.0f s: %.2f T-MAC/s, %.0f MHz\n", r.name, total,
                  traceMacs / (total - lastTrace) / 1e12, mhz);
          lastTrace = total;
          traceMacs = 0;
        }
      }
      smp.finish();
    } catch (const std::exception &e) {
      err = e.what();
      for (char &ch : err)
        if (ch == '"' || ch == '\\' || ch == '\n') ch = ' ';
      if (smp.th.joinable()) smp.finish();
    }
    const double rate = n ? macs * n / total / 1e12 : 0;
    const double mhz = smp.mhz();
    const bool real = r.acc == r.counted;  // every DPAS counted was run
    printf("{\"row\":\"%s\",\"what\":\"%s\",\"sg\":%d,\"dpas\":\"%d of %d\",\"reps\":%u,\"launches\":%u,"
           "\"msLaunch\":%.2f,\"tmacs\":%.2f,\"bestTmacs\":%.2f",
           r.name, r.what, SG, r.acc, r.all, a.reps, n, n ? total / n * 1000.0 : 0.0, rate,
           n ? macs / best / 1e12 : 0.0);
    if (mhz > 0) printf(",\"mhz\":%.0f", mhz);
    if (smp.watts() > 0) printf(",\"watts\":%.0f", smp.watts());
    if (smp.tempC() > 0) printf(",\"tempC\":%.0f", smp.tempC());
    if (smp.fan() > 0) printf(",\"fanRpm\":%.0f", smp.fan());
    if (real && eus && mhz > 0) printf(",\"macEuClk\":%.1f", rate * 1e12 / eus / (mhz * 1e6));
    if (!err.empty()) { printf(",\"error\":\"%s\"", err.c_str()); failed = 1; }
    printf("}\n");
    fflush(stdout);
  }
  sycl::free(dA, q);
  sycl::free(dB, q);
  sycl::free(hits, q);
  sycl::free(sink, q);
  return failed;
}
