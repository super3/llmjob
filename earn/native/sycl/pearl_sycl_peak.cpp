// pearl_sycl_peak: what an Intel GPU's XMX matrix engines do with little or
// nothing in the way, and where the fold's time goes between that and the rate
// it mines at. The AMD port's pearl_amd_peak asks the same of WMMA.
//
//   pearl_sycl_peak [--secs S] [--only LIST] [--reps R] [--trace]
//                   [--m M] [--n N] [--col-batch C] [--band B] [--sg 8|16] [--rb 2]
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
  int sgArg = 0, rbArg = 0;
  bool trace = false;
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
    else {
      fprintf(stderr, "usage: %s [--secs S] [--only LIST] [--reps R] [--trace] [--m M] [--n N] "
                      "[--col-batch C] [--band B] [--sg 8|16] [--rb 2]\n", argv[0]);
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
  const size_t aBytes = (size_t)p.m * k, bBytes = (size_t)p.n * k;
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
  for (int i = 0; i < 8; i++) { args.f.test.key[i] = mix(0xA5EEDu + (uint32_t)i); args.f.test.target_w[i] = 0; }
  args.f.test.hash_big_endian = 0;
  args.f.hits = hits;
  args.f.allTr = nullptr;
  args.seed = 0x1234567u;
  args.sink = sink;

  std::vector<Row> rows;
  if (rbArg == 2 && SG == 8) {
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
    PeakArgs a = args;
    a.reps = r.reps ? r.reps : reps;
    a.rowMask = r.rowMask;
    a.colMask = r.colMask;
    a.kMask = r.kMask;
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
