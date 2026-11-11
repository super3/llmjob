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
  uint32_t slmBytes;    // price and local-memory rows ("p-", "s-"): local memory a work-group asks for
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

#if defined(PEARL_SYCL_XMX_JIT)
// ---------------------------------------------------------------------------
// The price rows ("p-") and the local-memory rows ("s-"), added 12 November
// 2026 to carry the NVIDIA work's methods over. There each kind of step was
// priced by adding N of it to a loop of matrix instructions only, and reading
// the cost off the rate, the clock and the watts. These rows do that in the
// fragment fold's grid and tile (16 rows by 64 columns a sub-group, 16 DPAS a
// k-step), so a message's price is per k-step of the fold:
//   KIND 1  N block reads of 128 bytes from local memory (one address a message)
//        2  N vector loads of 4 dwords a lane from local memory (128 bytes, an address a lane)
//        3  N block writes of 128 bytes to local memory
//        4  N block reads of 128 bytes from global memory every sub-group reads (64 KB: first cache level)
//        5  N block reads of 128 bytes from global memory at the sub-group's own block of A' (64 KB each)
//        6  a work-group barrier every k-step (N unused)
//        7  a split barrier every k-step: arrive at its start, wait at its end (N unused)
//   KIND 1 to 7 run the DPAS on register operands and only keep what they load.
// The fold's operands through local memory (DPAS on what is loaded, every DPAS
// of a k-step waiting for its loads, as the fold does):
//        10 A' and B' both from local memory, block reads of 128 bytes (20 a k-step); nothing copies them
//        11 B' from local memory (16 block reads), A' from global memory at its own address (one vector of
//           8 dwords a fragment, as the fold); nothing copies B'
//        12 the copy ring: every k-step each sub-group of the work-group copies its share of the NEXT
//           k-step's B' (2,048 bytes over 2 W sub-groups) from its column window in global memory into local
//           memory (block reads, block writes), then loads this k-step's B' from local memory and its A'
//           from global memory, runs the 16 DPAS, and the group meets at a barrier. Two slots.
//        13 12 with a split barrier: wait for the k-step before, copy, arrive, then the DPAS. Three slots.
// --wpg W puts W windows of one column window in a work-group (2 W sub-groups), all sharing B'.
// --slm BYTES is the local memory a work-group asks for (at least what the row needs).
// Timing only: the bytes are not a fold's and nothing is checked.
inline void peak_arrive() {
#if defined(__SYCL_DEVICE_ONLY__)
  __spirv_ControlBarrierArriveINTEL(2, 2, 0x104);  // work-group; work-group memory, release
#endif
}
inline void peak_wait() {
#if defined(__SYCL_DEVICE_ONLY__)
  __spirv_ControlBarrierWaitINTEL(2, 2, 0x102);  // work-group; work-group memory, acquire
#endif
}
inline peak_b4u slm_read(uint32_t *p) {
#if defined(__SYCL_DEVICE_ONLY__)
  return __spirv_SubgroupBlockReadINTEL<peak_b4u>((const __attribute__((opencl_local)) uint32_t *)p);
#else
  peak_b4u v;
  __builtin_memcpy(&v, p, sizeof(v));
  return v;
#endif
}
inline void slm_write(uint32_t *p, peak_b4u v) {
#if defined(__SYCL_DEVICE_ONLY__)
  __spirv_SubgroupBlockWriteINTEL((__attribute__((opencl_local)) uint32_t *)p, v);
#else
  __builtin_memcpy(p, &v, sizeof(v));
#endif
}

template <int KIND, int N, int ND = 16> struct PeakPrice {
  PeakArgs P;
  sycl::local_accessor<uint32_t, 1> slm;

  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    namespace syclex = sycl::ext::oneapi::experimental;
    return syclex::properties{syclex::sub_group_size<8>, sycl::ext::intel::experimental::grf_size<256>};
  }

  void operator()(sycl::nd_item<1> it) const {
    const FoldArgs &A = P.f;
    sycl::sub_group sg = it.get_sub_group();
    const uint32_t lane = (uint32_t)sg.get_local_linear_id();
    const uint32_t sgl = (uint32_t)sg.get_group_linear_id();
    const uint32_t nsg = 2u * P.wpg;
    const uint32_t sgi = sgl & 1u, widx = sgl >> 1;
    const uint32_t g = (uint32_t)it.get_group(0) * P.wpg + widx;
    const uint32_t rowWins = A.rowsValid / 2u, colWins = A.colBatch / 4u;
    uint32_t wr, wc;
    pearl_sycl::window_of(g, rowWins, colWins, A.band, wr, wc);
    const uint32_t row0 = 32u * wr + sgi * 16u;
    // The sub-group's own A' block and its column window's B', as the fold addresses them.
    const int8_t *Au = A.Ap + (size_t)(row0 >> 4) * A.aStride;
    const int8_t *Bu = A.Bp + (size_t)(A.colStart / 4u + wc) * A.bStride;
    uint32_t *L = &slm[0];
    const uint32_t slmWords = P.slmBytes / 4u;
    v8i ar[2][2], br[2];
    for (int set = 0; set < 2; set++)
      for (int rb = 0; rb < 2; rb++)
        for (int r = 0; r < 8; r++)
          ar[set][rb][r] = (int)bytes7(mix(P.seed + lane * 131u + sgi * 17u + (uint32_t)(set * 64 + rb * 8 + r)));
    for (int i = 0; i < 2; i++)
      for (int d = 0; d < 8; d++) br[i][d] = (int)bytes7(mix((P.seed ^ 0xB5297A4Du) + lane * 257u + (uint32_t)(i * 8 + d)));
    v8i c[2][8];
    pearl_sycl::each<2>([&](int rb) { pearl_sycl::each<8>([&](int cb) { c[rb][cb] = 0; }); });
    peak_b4u keep4 = 0;
    // B' of k-step ks as 16 blocks of 128 bytes; slot s of the ring is 512 words.
    constexpr uint32_t kSlotWords = 512u;
    // This sub-group's blocks of a k-step's B': sgl, sgl + nsg, ... of 16. (Until 23:25 UTC on 11 November
    // this was 16 / nsg blocks from sgl * that, which is none at 32 or more sub-groups: the first s-ring rows
    // at --wpg 16 and 32 copied nothing.)
    auto copy_in = [&](uint32_t ks, uint32_t slot) {
      const uint32_t kr = ks & 63u;
      for (uint32_t blk = sgl; blk < 16u; blk += nsg) {
        const peak_b4u v = peak_block_read<peak_b4u, uint32_t>(Bu + (size_t)kr * 2048u + (size_t)blk * 128u);
        slm_write(L + slot * kSlotWords + blk * 32u, v);
      }
    };
    if constexpr (KIND == 12) {
      copy_in(0, 0);
      sycl::group_barrier(it.get_group());
    } else if constexpr (KIND == 13 || KIND == 14) {
      copy_in(0, 0);
      peak_arrive();
    } else if constexpr (KIND == 15) {
      copy_in(0, 0);
    } else if constexpr (KIND == 16) {
      copy_in(0, 0);
      copy_in(1, 1);
      peak_arrive();
    }
    for (uint32_t rep = 0; rep < P.reps; rep++) {
      for (uint32_t ks = 0; ks < 64u; ks++) {
        if constexpr ((KIND >= 1 && KIND <= 7) || KIND == 8) {
          if constexpr (KIND == 7) peak_arrive();
          if constexpr (KIND == 1 || KIND == 2 || KIND == 3) {
            pearl_sycl::each<N>([&](int i) {
              const uint32_t w = (((ks * (uint32_t)N + (uint32_t)i) * 32u) + sgl * 64u) % slmWords;
              if constexpr (KIND == 1) keep4 ^= slm_read(L + w);
              else if constexpr (KIND == 2) keep4 ^= *reinterpret_cast<const peak_b4u *>(L + w + 4u * lane);
              else slm_write(L + w, keep4 + (uint32_t)i);
            });
          } else if constexpr (KIND == 4 || KIND == 5) {
            const int8_t *base = KIND == 4 ? A.Bp : Au;
            pearl_sycl::each<N>([&](int i) {
              const size_t off = (((size_t)ks * (size_t)N + (size_t)i) * 128u) % 65536u;
              keep4 ^= peak_block_read<peak_b4u, uint32_t>(base + off);
            });
          } else if constexpr (KIND == 8) {
            // A stream of its own through 1 MB a sub-group, the whole A' area in all: from memory.
            const size_t area = (size_t)A.rowsValid * A.aStride;
            const size_t mine = (size_t)(row0 >> 4) * 16u * A.aStride;
            pearl_sycl::each<N>([&](int i) {
              const size_t off = (mine + ((((size_t)rep * 64u + ks) * (size_t)N + (size_t)i) * 128u) % (16u * (size_t)A.aStride)) % area;
              keep4 ^= peak_block_read<peak_b4u, uint32_t>(A.Ap + off);
            });
          }
          const int z = (int)(keep4[0] & P.zero);
          pearl_sycl::each<8>([&](int cb) {
            pearl_sycl::each<2>([&](int rb) {
              if (cb * 2 + rb < ND) {
                v8i a = ar[ks & 1u][rb];
                a[0] |= z;
                c[rb][cb] = pearl_sycl::dpas<8, true>(sg, a, br[(ks + (uint32_t)cb) & 1u], c[rb][cb]);
              }
            });
          });
          if constexpr (KIND == 6) sycl::group_barrier(it.get_group());
          if constexpr (KIND == 7) peak_wait();
        } else {
          // The fold's 16 DPAS on loaded fragments.
          v8i af[2], bf[8];
          uint32_t slot = 0;
          if constexpr (KIND == 12) {
            slot = ks & 1u;
            copy_in(ks + 1u, slot ^ 1u);
          } else if constexpr (KIND == 13 || KIND == 14) {
            slot = ks % 3u;
            peak_wait();
            copy_in(ks + 1u, (ks + 1u) % 3u);
            peak_arrive();
          } else if constexpr (KIND == 15) {
            slot = ks % 3u;
            copy_in(ks + 1u, (ks + 1u) % 3u);
          } else if constexpr (KIND == 16) {
            // Pairs of k-steps: slots of two k-steps, a barrier every second k-step.
            slot = ((ks >> 1) % 3u) * 2u + (ks & 1u);
            if ((ks & 1u) == 0) {
              peak_wait();
              const uint32_t nx = ((ks >> 1) + 1u) % 3u;
              copy_in(ks + 2u, nx * 2u);
              copy_in(ks + 3u, nx * 2u + 1u);
              peak_arrive();
            }
          }
          // KIND 10 and 11: four slots of 768 words, B' (512) then each sub-group's A' (128).
          if constexpr (KIND == 10) {
            const uint32_t base = (ks & 3u) * 768u + 512u + sgi * 128u;
            pearl_sycl::each<2>([&](int rb) {
              const peak_b4u lo = slm_read(L + base + (uint32_t)rb * 64u);
              const peak_b4u hi = slm_read(L + base + (uint32_t)rb * 64u + 32u);
              af[rb] = (v8i)__builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
            });
          } else if constexpr (KIND == 14 || KIND == 15 || KIND == 16) {
            pearl_sycl::each<2>([&](int rb) {
              const int8_t *u = Au + (size_t)(ks & 63u) * 512u + (size_t)rb * 256u;
              const peak_b4u lo = peak_block_read<peak_b4u, uint32_t>(u);
              const peak_b4u hi = peak_block_read<peak_b4u, uint32_t>(u + 128);
              af[rb] = (v8i)__builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
            });
          } else {
            pearl_sycl::each<2>([&](int rb) {
              af[rb] = *reinterpret_cast<const v8iu *>(Au + (size_t)(ks & 63u) * 512u + (size_t)rb * 256u + 32u * lane);
            });
          }
          const uint32_t bbase = KIND == 10 || KIND == 11 ? (ks & 3u) * 768u : slot * kSlotWords;
          pearl_sycl::each<8>([&](int cb) {
            const peak_b4u lo = slm_read(L + bbase + (uint32_t)cb * 64u);
            const peak_b4u hi = slm_read(L + bbase + (uint32_t)cb * 64u + 32u);
            bf[cb] = (v8i)__builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
          });
          int t = 0;
          pearl_sycl::each<2>([&](int rb) { t ^= af[rb][0]; });
          pearl_sycl::each<8>([&](int cb) { t ^= bf[cb][0]; });
          t &= (int)P.zero;
          pearl_sycl::each<2>([&](int rb) { af[rb][0] |= t; });
          pearl_sycl::each<8>([&](int cb) {
            pearl_sycl::each<2>([&](int rb) { c[rb][cb] = pearl_sycl::dpas<8, true>(sg, af[rb], bf[cb], c[rb][cb]); });
          });
          if constexpr (KIND == 12) sycl::group_barrier(it.get_group());
        }
      }
    }
    if constexpr (KIND == 13 || KIND == 14 || KIND == 16) peak_wait();
    uint32_t x = keep4[0] ^ keep4[1] ^ keep4[2] ^ keep4[3];
    pearl_sycl::each<2>([&](int rb) {
      pearl_sycl::each<8>([&](int cb) {
        for (int r = 0; r < 8; r++) x ^= (uint32_t)c[rb][cb][r];
      });
    });
    if constexpr (KIND == 3) x ^= L[(lane * 7u) % slmWords];
    x = sycl::reduce_over_group(sg, x, sycl::bit_xor<uint32_t>());
    if (lane == 0) P.sink[(size_t)g * 2u + sgi] = x;
  }
};

template <int KIND, int N, int ND = 16>
inline sycl::event peak_price(sycl::queue &q, const PeakArgs &args, uint32_t groups) {
  const size_t wg = 16u * args.wpg;
  uint32_t need = 4096u;
  if (KIND >= 10 && KIND <= 11) need = 4u * 768u * 4u;
  if (KIND == 12) need = 2u * 2048u;
  if (KIND == 13 || KIND == 14 || KIND == 15) need = 3u * 2048u;
  if (KIND == 16) need = 6u * 2048u;
  PeakArgs a = args;
  if (a.slmBytes < need) a.slmBytes = need;
  return q.submit([&](sycl::handler &h) {
    sycl::local_accessor<uint32_t, 1> slm(a.slmBytes / 4u, h);
    h.parallel_for(sycl::nd_range<1>((size_t)(groups / a.wpg) * wg, wg), PeakPrice<KIND, N, ND>{a, slm});
  });
}
#endif  // PEARL_SYCL_XMX_JIT

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
  // The price rows and the local-memory rows (PeakPrice).
#define PRICE_ROW(NAME, WHAT, KIND, N)                                                          \
  v.push_back({NAME, WHAT, 16, 16, 16, 1u, all, all, all,                                       \
               [](sycl::queue &q, const PeakArgs &a, uint32_t g) {                              \
                 return peak_price<KIND, N>(q, a, g);                                           \
               }})
  PRICE_ROW("p-skel", "DPAS on register operands, the fold's grid: the skeleton the p- rows add to", 1, 0);
  PRICE_ROW("p-slm4", "4 block reads of 128 bytes from local memory a k-step", 1, 4);
  PRICE_ROW("p-slm8", "8 block reads of 128 bytes from local memory a k-step", 1, 8);
  PRICE_ROW("p-slm16", "16 block reads of 128 bytes from local memory a k-step", 1, 16);
  PRICE_ROW("p-slm20", "20 block reads of 128 bytes from local memory a k-step", 1, 20);
  PRICE_ROW("p-slmv8", "8 vector loads of 4 dwords a lane from local memory a k-step", 2, 8);
  PRICE_ROW("p-slmv16", "16 vector loads of 4 dwords a lane from local memory a k-step", 2, 16);
  PRICE_ROW("p-slmw2", "2 block writes of 128 bytes to local memory a k-step", 3, 2);
  PRICE_ROW("p-slmw4", "4 block writes of 128 bytes to local memory a k-step", 3, 4);
  PRICE_ROW("p-slmw8", "8 block writes of 128 bytes to local memory a k-step", 3, 8);
  PRICE_ROW("p-l1x8", "8 block reads of 128 bytes from 64 KB of global memory every sub-group reads", 4, 8);
  PRICE_ROW("p-l1x16", "16 block reads of 128 bytes from 64 KB of global memory every sub-group reads", 4, 16);
  PRICE_ROW("p-l1x20", "20 block reads of 128 bytes from 64 KB of global memory every sub-group reads", 4, 20);
  PRICE_ROW("p-own2", "2 block reads of 128 bytes from the sub-group's own 64 KB block of A'", 5, 2);
  PRICE_ROW("p-own4", "4 block reads of 128 bytes from the sub-group's own 64 KB block of A'", 5, 4);
  PRICE_ROW("p-own8", "8 block reads of 128 bytes from the sub-group's own 64 KB block of A'", 5, 8);
  PRICE_ROW("p-bar", "a work-group barrier every k-step", 6, 0);
  PRICE_ROW("p-sbar", "a split barrier every k-step: arrive at its start, wait at its end", 7, 0);
  PRICE_ROW("s-ab", "A' and B' both from local memory, 20 block reads a k-step, nothing copied", 10, 0);
  PRICE_ROW("s-b", "B' from local memory (16 block reads), A' from global memory as the fold loads it, nothing copied", 11, 0);
  PRICE_ROW("s-ring", "the copy ring: each sub-group copies its share of the next k-step's B' into local memory, barrier a k-step", 12, 0);
  PRICE_ROW("s-ring3", "the copy ring with a split barrier and three slots", 13, 0);
  PRICE_ROW("s-ring3k", "s-ring3 with A' as block reads of 128 bytes (4 a k-step)", 14, 0);
  PRICE_ROW("s-ringnb", "s-ring3k with no barrier at all: what the copies cost alone (reads may see a slot early)", 15, 0);
  PRICE_ROW("s-ring2k", "s-ring3k with slots of two k-steps and a split barrier every second k-step", 16, 0);
  PRICE_ROW("p-far8", "8 block reads of 128 bytes a k-step from a 1 MB stream of the sub-group's own (memory)", 8, 8);
  PRICE_ROW("p-far16", "16 block reads of 128 bytes a k-step from a 1 MB stream of the sub-group's own (memory)", 8, 16);
#undef PRICE_ROW
  // The two folds themselves, as the core launches them (their bits are the probe's operands, not a job's).
  v.push_back({"x-fold", "the fragment fold (fold_xmxf), the kernel itself", 16, 16, 16, 1u, all, all, all,
               [](sycl::queue &q, const PeakArgs &a, uint32_t g) {
                 pearl_sycl::fold_xmxf<8, 2, true>(q, a.f, g);
                 return sycl::event();
               }});
  v.push_back({"r-fold", "the ring fold (fold_xmxr), the kernel itself, --wpg windows a work-group", 16, 16, 16, 1u,
               all, all, all, [](sycl::queue &q, const PeakArgs &a, uint32_t g) {
                 pearl_sycl::FoldArgs f = a.f;
                 f.wpg = a.wpg;
                 pearl_sycl::fold_xmxr<true>(q, f, g);
                 return sycl::event();
               }});
  v.push_back({"r-fold-ro", "r-fold with each chunk's readout after the next k-step's arrival and loads", 16, 16, 16, 1u, all, all, all,
               [](sycl::queue &q, const PeakArgs &a, uint32_t g) {
                 pearl_sycl::FoldArgs f = a.f;
                 f.wpg = a.wpg;
                 pearl_sycl::fold_xmxr<true, 1>(q, f, g);
                 return sycl::event();
               }});
  v.push_back({"r-fold-p", "r-fold with the copy loaded a k-step early (VAR 4)", 16, 16, 16, 1u, all, all, all,
               [](sycl::queue &q, const PeakArgs &a, uint32_t g) {
                 pearl_sycl::FoldArgs f = a.f;
                 f.wpg = a.wpg;
                 pearl_sycl::fold_xmxr<true, 4>(q, f, g);
                 return sycl::event();
               }});
  v.push_back({"r-fold-pw", "r-fold-p with no acquire fence at the wait (VAR 12)", 16, 16, 16, 1u, all, all, all,
               [](sycl::queue &q, const PeakArgs &a, uint32_t g) {
                 pearl_sycl::FoldArgs f = a.f;
                 f.wpg = a.wpg;
                 pearl_sycl::fold_xmxr<true, 12>(q, f, g);
                 return sycl::event();
               }});
  v.push_back({"r-fold-v8", "r-fold with A' as one vector a fragment (timing only)", 16, 16, 16, 1u, all, all, all,
               [](sycl::queue &q, const PeakArgs &a, uint32_t g) {
                 pearl_sycl::FoldArgs f = a.f;
                 f.wpg = a.wpg;
                 pearl_sycl::fold_xmxr<true, 2>(q, f, g);
                 return sycl::event();
               }});
  // Energy rows: one DPAS a k-step instead of 16, so the card stays under its power limit and the watts
  // are the messages' and the clock's. Their T-MAC/s is counted as if all 16 ran: k-steps a second are
  // T-MAC/s x 1e12 / 32768.
#define EROW(NAME, WHAT, KIND, N)                                                                \
  v.push_back({NAME, WHAT, 1, 16, 16, 1u, all, all, all,                                        \
               [](sycl::queue &q, const PeakArgs &a, uint32_t g) {                              \
                 return peak_price<KIND, N, 1>(q, a, g);                                        \
               }})
  EROW("e-skel", "1 DPAS a k-step, nothing else", 1, 0);
  EROW("e-l1x8", "1 DPAS and 8 block reads from first-cache-level memory a k-step", 4, 8);
  EROW("e-l1x16", "1 DPAS and 16 block reads from first-cache-level memory a k-step", 4, 16);
  EROW("e-own8", "1 DPAS and 8 block reads from the sub-group's own 64 KB a k-step", 5, 8);
  EROW("e-own16", "1 DPAS and 16 block reads from the sub-group's own 64 KB a k-step", 5, 16);
  EROW("e-far8", "1 DPAS and 8 block reads from a 1 MB stream of its own a k-step", 8, 8);
  EROW("e-far16", "1 DPAS and 16 block reads from a 1 MB stream of its own a k-step", 8, 16);
  EROW("e-slm8", "1 DPAS and 8 block reads from local memory a k-step", 1, 8);
#undef EROW
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
  uint32_t aStride = 0, bStride = 0, laneStride = 32, wpg = 1, stagger = 0, aMix = 0, bMix = 0, spin = 0, pfA = 0, pfB = 0, pfHint = 0, touchA = 0, touchB = 0, fillSkip = 0, slmBytes = 0;
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
    else if (a == "--slm" && v) { slmBytes = (uint32_t)atoi(v); i++; }
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
  args.slmBytes = slmBytes;
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
