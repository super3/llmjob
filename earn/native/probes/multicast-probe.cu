// multicast-probe.cu -- RTX 50 (sm_120): what TMA multicast saves the tall fold's B' feed, and
// whether the cluster build's barriers hang. Standalone: nothing of the miner.
//
//   nvcc -O3 -std=c++17 -gencode arch=compute_120,code=sm_120 multicast-probe.cu -o multicast-probe -ldl
//   ./multicast-probe [secs=6] [rounds=3] [pace=3240] [settle=1.5] [wait_ms=1000] [launch_ms=300] [hash=3000]
//
// compilers.sh runs it as its "multicast" stage, and multicast-read.js turns its lines into f.
// Item 16, step 2: PEARL_TALL_CLUSTER hung on a 5090 on 1 November, and nothing had measured
// what multicast saves. This answers both before more cluster work.
//
// The kernel is the fold's feed without the mma. Clusters of two CTAs of 256 threads, each with
// the fold's 99840 bytes of shared (so one CTA an SM): a ring of three 28 KB buffers, each with a
// FULL and an EMPTY mbarrier behind it, the fold's TMA boxes (B' 256 columns by 64 bytes, A' 192
// rows by 64 bytes, SWIZZLE_64B) from k-blocked operands small enough for the L2 (16 MB and 6 MB),
// and a producer (thread 128) that refills two stages ahead behind its EMPTY wait. Each warp reads
// its column slot's B' with the fold's eight ldmatrix a stage, so the B' region is read twice a
// stage, and sums the words. Then it waits out the stage time (pace, in SM cycles: 3240 is a
// stage of the fold on a 5090 at 2613 MHz) and releases the buffer. The pace holds every arm to
// the same stage rate, so the energy a pair-stage compares across arms.
//
// Arms, by what a pair does a stage:
//   e   no loads: the warps read what the buffers hold.
//   a   each CTA loads its own B' box: two L2 reads, two fills.
//   b   rank 0 multicasts one B' box to both, as PEARL_TALL_CLUSTER does: one L2 read if
//       multicast is one, two fills.
//   d   only rank 0 loads its B' box: one L2 read, one fill. Rank 1 reads stale bytes.
//   c   b's loads with the fold's barrier protocol: every warp arrives on both CTAs' EMPTY (a
//       mapa remote arrive and a local one, both released at cluster scope), EMPTY counts both
//       CTAs, each producer acquires at cluster scope, and FULL has no remote arrival.
//   cA  c plus each CTA's own A' box, so FULL waits for a whole stage, as in the fold.
//   cAH cA plus the fold's tile seam (every 32 stages, a tile): warp w and w + 4 meet at a named
//       barrier, w sleeps a hash's cycles (hash: 3000, a 4090's hash phase in pearl_config.h's
//       PEARL_TALL_STUB_SPIN note) while w + 4 runs on, and at the end of the tile's first chunk
//       (two stages) warps 4-7 wait at CTA scope on the EMPTY of the tile's stage 0, which counts
//       the peer's arrivals too: the fold's hand-off.
// The push arms (issue #287, item 7): B' shared between the pair without multicast. Each CTA loads
// half of the pair's box and copies that half into the peer's buffer over distributed shared memory
// (cp.async.bulk shared::cta to shared::cluster), so the L2 serves one box a pair-stage, not two. All
// three use c's barrier protocol (cluster-wide EMPTY, local FULL):
//   a2  each CTA loads its own whole B' box: two L2 reads, two fills. a with c's barriers.
//   h2  each CTA loads its half of the box and nothing else: one L2 read, one fill. Each CTA reads
//       half stale bytes.
//   p   h2's loads, then each CTA's producer waits for its half to land (a HALF mbarrier behind
//       FULL), copies it to the same offset in the peer's buffer, completing on the peer's FULL, and
//       arrives on its own FULL. One L2 read, two fills, one box CTA to CTA.
// fp = (E_a2 - E_p) / (E_a2 - E_h2): the share of an L2 box's cost the copy saves. A stock 5090 pays
// about 5.2% of its rate for a B' box a CTA-stage (item 16, skip B'), so a fold that shares B' this
// way could gain about 2.6% x fp, before its own costs. If p sticks and a2 does not, the copy never
// lands on this card. MC_ARMS picks the phases: "push" (these three, with e), "multicast" (the arms
// above), or both, push first (the default: on a 5090 the multicast arm b hangs, which ends the run).
// e, a, b and d share another coupling, not the fold's: rank 1's warps also arrive on rank 0's
// EMPTY, which is what lets rank 0 refill both CTAs' buffers in b, and rank 0's producer also
// arrives on rank 1's FULL with each fill, so rank 1 can never run a phase ahead of rank 0 (in
// b the multicast already holds it back). So those four differ only in their loads. c differs
// from b in both barriers: the cluster-wide EMPTY, and FULL without rank 0's remote arrival.
// Each own box (a, d, and the A' boxes of cA and cAH) uses the fold's .shared::cta form where
// the toolkit has it (12.8 on; CARD says which).
//
// Reading it: f = (E_a - E_b) / (E_a - E_d), E the energy a pair-stage. a and b fill the same
// bytes, so f is the share of a whole box's cost (L2 read, crossbar, shared fill) that multicast
// saves. The plan's gate for item 16's #1 and #3 is f >= 0.5. (E_a - E_e) against
// 2 (E_d - E_e) checks that the cost is linear in boxes. The hang ladder: b and not a, the
// multicast itself; c and not b, the fold's barrier protocol; cA and not c, multicast beside the
// CTA's own box; cAH and not cA, the fold's tile seam. cAH's stages carry the hash's sleep, so
// its energy is not compared.
//
// Watchdog: every mbarrier wait gives up after wait_ms of global time, sets an abort flag the
// other waits see, and writes what it waited on (and the barrier's raw 64-bit word) to host
// memory, so a hang prints a TIMEOUT line instead of sticking. The host gives each launch a
// deadline too; past it the process prints HANG and exits. The arms that can hang run last, cAH
// last of all: its named barrier cannot give up, so after a timeout elsewhere in the CTA a warp
// can be left at it, which the deadline then reports as HANG after the TIMEOUT line.
//
// Lines: CARD, NVML, CLUSTERS (the occupancy query and its return code per arm), SMOKE (one
// short launch per arm: the launch's return code, the sync's, and the checksums), ROUND (one
// measured window), TIMEOUT, FAULT, HANG, DONE. Energy is NVML's total-energy counter when the
// driver has it, else mean power times time.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace {

constexpr uint32_t THREADS = 256u;
constexpr uint32_t SK = 64u;                          // bytes of k a stage: one k-block
constexpr uint32_t BN = 256u, BM = 192u;              // B' columns and A' rows a box
constexpr uint32_t BOXB = BN * SK, BOXA = BM * SK;    // 16384 and 12288
constexpr uint32_t BNH = BN / 2u, BOXH = BOXB / 2u;   // a push arm's half box: 128 columns, 8192
constexpr uint32_t STAGE = BOXB + BOXA;
constexpr uint32_t STRIDE = STAGE + 512u;             // the fold's ring: barriers in the pad
constexpr uint32_t NST = 3u;
constexpr uint32_t SMEM = NST * STRIDE + BM * BN / 4u;  // 99840, the fold's footprint
constexpr uint32_t PRODUCER = 128u;                   // lane 0 of warp 4, as in the fold
constexpr uint32_t KB = 32u;                          // k-blocks: k 2048
constexpr uint32_t CG = 32u;                          // B' column groups: 8192 columns, 16 MB
constexpr uint32_t RG = 16u;                          // A' row groups: 3072 rows, 6 MB
static_assert(SMEM == 99840u, "the fold's shared footprint");

enum Arm { ARM_E, ARM_A, ARM_B, ARM_D, ARM_C, ARM_CA, ARM_CAH, ARM_A2, ARM_H2, ARM_P, NARMS };

// Bytes a pair-stage: what TMA lands in shared (both CTAs), and what the boxes read from the
// L2 (a multicast counted as one read). Which ranks' sums can be checked: a rank that gets no
// B' reads stale bytes.
struct ArmInfo {
  const char *name;
  int arm;
  uint32_t fillPair, l2Pair;
  bool check0, check1;
};
const ArmInfo ARMS[] = {
    {"e", ARM_E, 0u, 0u, false, false},
    {"a", ARM_A, 2u * BOXB, 2u * BOXB, true, true},
    {"b", ARM_B, 2u * BOXB, BOXB, true, true},
    {"d", ARM_D, BOXB, BOXB, true, false},
    {"c", ARM_C, 2u * BOXB, BOXB, true, true},
    {"cA", ARM_CA, 2u * (BOXB + BOXA), BOXB + 2u * BOXA, true, true},
    {"cAH", ARM_CAH, 2u * (BOXB + BOXA), BOXB + 2u * BOXA, true, true},
    {"a2", ARM_A2, 2u * BOXB, 2u * BOXB, true, true},
    {"h2", ARM_H2, BOXB, BOXB, false, false},
    {"p", ARM_P, 2u * BOXB, BOXB, true, true},
};
static_assert(sizeof ARMS / sizeof ARMS[0] == NARMS, "one row an arm, in the enum's order");

// Whether the own boxes use the fold's .shared::cta form (see tma_3d). The host pass reads it too,
// for the CARD line. -DMC_SHARED_CTA=1 forces it, for a PTX-only check under an older toolkit.
#ifndef MC_SHARED_CTA
#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__ >= 1208
#define MC_SHARED_CTA 1
#else
#define MC_SHARED_CTA 0
#endif
#endif

// A tensor map as a kernel parameter: 128-byte aligned in the device pass only (see
// earn/native/src/pearl_tensor_map.h for why not cuda.h's own alignment).
#if defined(__CUDA_ARCH__)
#define MC_MAP_ALIGN alignas(128)
#else
#define MC_MAP_ALIGN
#endif
struct MC_MAP_ALIGN Map {
  CUtensorMap map;
};

// What the first wait to give up saw, in host memory the host can read whatever the device does.
struct Status {
  unsigned set, role, block, rank, thread, stage, parity, bar;
  unsigned long long state, waited_ns;
};
// Each CTA's result.
struct CtaReport {
  unsigned sum, stages;
  unsigned long long cycles;
};

// ---------------------------------------------------------------------------------------------
// Device: the fold's PTX, on 32-bit shared addresses.
__device__ __forceinline__ uint32_t smem_addr(const void *p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t n) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(bar), "r"(n) : "memory");
}
// Released at CTA scope.
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("{\n\t.reg .b64 st;\n\tmbarrier.arrive.shared.b64 st, [%0];\n\t}" ::"r"(bar) : "memory");
}
// This CTA's barrier, released at cluster scope (the fold's pearl_mbar_arrive_cluster).
__device__ __forceinline__ void mbar_arrive_cluster(uint32_t bar) {
  asm volatile("{\n\t.reg .b64 st;\n\tmbarrier.arrive.release.cluster.shared::cta.b64 st, [%0];\n\t}" ::"r"(
                   bar)
               : "memory");
}
// The peer's barrier at a shared::cluster address (pearl_mbar_arrive_remote).
__device__ __forceinline__ void mbar_arrive_remote(uint32_t bar) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
// One try: acquire at CTA scope, or at cluster scope (the producer's EMPTY wait in the fold).
template <bool CL>
__device__ __forceinline__ bool mbar_try(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  if (CL)
    asm volatile(
        "{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t}"
        : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  else
    asm volatile(
        "{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t}"
        : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0u;
}
// A CTA's own box: the fold's pearl_tma_3d, with a .shared::cta destination, on toolkits that have
// it (PTX 8.6, CUDA 12.8). Before that (the sm_90 compile check under 12.6) .shared::cluster, which
// in a cluster launch names this CTA's own window for its own shared address.
__device__ __forceinline__ void tma_3d(uint32_t dst, const CUtensorMap *map, uint32_t c0, uint32_t c1,
                                       uint32_t c2, uint32_t bar) {
  asm volatile(
#if MC_SHARED_CTA
      "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
#else
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
#endif
      " [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(dst), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1),
      "r"(c2), "r"(bar)
      : "memory");
}
// The multicast: .shared::cluster, the only form it has.
__device__ __forceinline__ void tma_3d_mc(uint32_t dst, const CUtensorMap *map, uint32_t c0, uint32_t c1,
                                          uint32_t c2, uint32_t bar, uint16_t mask) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      ".multicast::cluster [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(bar), "h"(mask)
      : "memory");
}
// The push arm's copy: `bytes` of this CTA's shared memory at src into the peer's at dst, completing
// on the peer's barrier at bar (dst and bar from mapa). Distributed shared memory, not the L2.
__device__ __forceinline__ void push_bulk(uint32_t dst, uint32_t src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                   dst),
               "r"(src), "r"(bytes), "r"(bar)
               : "memory");
}
__device__ __forceinline__ uint32_t ctarank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ uint32_t mapa(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
__device__ __forceinline__ uint64_t gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3,
                                            uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(a));
}
__device__ __forceinline__ uint32_t mix(uint32_t h) {
  h = (h ^ (h >> 16)) * 0x85EBCA6Bu;
  h = (h ^ (h >> 13)) * 0xC2B2AE35u;
  return h ^ (h >> 16);
}

// A wait that gives up: after waitNs of global time, or once another wait has given up. The
// first to give up records what it waited on. role 0 is a FULL wait, 1 the producer's EMPTY, 2
// the hand-off's EMPTY (cAH), 3 the producer's HALF (p).
template <bool CL>
__device__ bool wait_bounded(uint32_t bar, uint32_t parity, uint64_t waitNs, unsigned *abortFlag,
                             volatile Status *st, uint32_t role, uint32_t g) {
  if (mbar_try<CL>(bar, parity)) return true;
  const uint64_t t0 = gtime();
  for (;;) {
    if (mbar_try<CL>(bar, parity)) return true;
    if (*(volatile unsigned *)abortFlag) return false;
    const uint64_t dt = gtime() - t0;
    if (dt > waitNs) {
      if (atomicCAS(abortFlag, 0u, 1u) == 0u) {
        unsigned long long word;
        asm volatile("ld.shared.b64 %0, [%1];" : "=l"(word) : "r"(bar));
        st->role = role;
        st->block = blockIdx.x;
        st->rank = ctarank();
        st->thread = threadIdx.x;
        st->stage = g;
        st->parity = parity;
        st->bar = bar;
        st->state = word;
        st->waited_ns = dt;
        __threadfence_system();
        st->set = 1u;
        __threadfence_system();
      }
      return false;
    }
  }
}

template <int ARM>
__global__ void __launch_bounds__(THREADS, 1)
    mc_probe(const __grid_constant__ Map tmB, const __grid_constant__ Map tmA, const __grid_constant__ Map tmBh,
             uint32_t tiles, uint32_t pace, uint32_t hashCyc, unsigned long long waitNs, unsigned *abortFlag,
             CtaReport *rep, Status *st) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // The fold's barrier protocol (c, cA, cAH, and the push arms), or the one-way coupling of the
  // others; each CTA's own A' box (cA, cAH); the fold's tile seam (cAH).
  constexpr bool SYM = ARM == ARM_C || ARM == ARM_CA || ARM == ARM_CAH || ARM == ARM_A2 || ARM == ARM_H2 || ARM == ARM_P;
  constexpr bool OWNA = ARM == ARM_CA || ARM == ARM_CAH;
  constexpr bool SEAM = ARM == ARM_CAH;
  // Half a box a CTA (h2, p), and the copy of it into the peer (p).
  constexpr bool HALF = ARM == ARM_H2 || ARM == ARM_P;
  constexpr bool PUSH = ARM == ARM_P;
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sbase = smem_addr(smem);
  if ((sbase & 1023u) != 0u) __trap();
  const uint32_t rank = ctarank();
  const uint32_t pair = blockIdx.x / 2u, pairs = gridDim.x / 2u;
  const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
  // FULL: the producer's arrival, and rank 0's producer's too on rank 1 outside the arms with the
  // fold's barriers; in p the producer's second, made once its half has landed and gone to the peer.
  // EMPTY: one a warp, this CTA's eight, and the peer's eight where they arrive too (rank 0
  // always; both ranks with the fold's barriers). HALF (p): the producer's expect_tx, and its half
  // box's bytes.
  if (threadIdx.x < NST) {
    mbar_init(sbase + threadIdx.x * STRIDE + STAGE, PUSH || (!SYM && rank == 1u) ? 2u : 1u);
    mbar_init(sbase + threadIdx.x * STRIDE + STAGE + 8u, SYM || rank == 0u ? 16u : 8u);
    if (PUSH) mbar_init(sbase + threadIdx.x * STRIDE + STAGE + 16u, 1u);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // The B' regions start as noise, so a rank that loads nothing (e, d's rank 1) reads bytes like
  // the others'. TMA writes them later, so the generic writes are fenced against the async proxy.
  for (uint32_t o = threadIdx.x * 16u; o < NST * BOXB; o += THREADS * 16u) {
    const uint32_t at = sbase + (o / BOXB) * STRIDE + o % BOXB, h = blockIdx.x * 0x9E3779B9u ^ o;
    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(at), "r"(mix(h)), "r"(mix(h + 1u)),
                 "r"(mix(h + 2u)), "r"(mix(h + 3u))
                 : "memory");
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  // The peer's barriers exist before anything arrives on them or lands in its shared memory.
  cluster_sync();

  // Stage g of this launch: tile g / KB of the pair's walk, k-block g % KB, buffer g % 3.
  auto fill = [&](uint32_t g) {
    const uint32_t buf = sbase + (g % NST) * STRIDE, full = buf + STAGE;
    const uint32_t t = pair + (g / KB) * pairs, kb = g % KB;
    const uint32_t col = (t % CG) * BN, row = ((2u * t + rank) % RG) * BM;
    if (!SYM && rank == 0u) mbar_arrive_remote(mapa(full, 1u));
    if (ARM == ARM_E || (ARM == ARM_D && rank != 0u)) {
      mbar_arrive(full);
      return;
    }
    if (HALF) {
      // This rank's half of the pair's box: columns 128 rank on, at the same offset in the buffer.
      // In p it lands on HALF, and FULL waits for the peer's half instead (push, below).
      const uint32_t ho = rank * BOXH;
      expect_tx(full, BOXH);
      if (PUSH) expect_tx(buf + STAGE + 16u, BOXH);
      tma_3d(buf + ho, &tmBh.map, 0u, col + rank * BNH, kb, PUSH ? buf + STAGE + 16u : full);
      return;
    }
    expect_tx(full, OWNA ? STAGE : BOXB);
    if (ARM == ARM_A || ARM == ARM_D || ARM == ARM_A2)
      tma_3d(buf, &tmB.map, 0u, col, kb, full);
    else if (rank == 0u)
      tma_3d_mc(buf, &tmB.map, 0u, col, kb, full, (uint16_t)3u);
    if (OWNA) tma_3d(buf + BOXB, &tmA.map, 0u, row, kb, full);
  };
  // p: once this rank's half of stage g's box has landed, copy it to the same offset in the peer's
  // buffer, completing on the peer's FULL, and make this CTA's second FULL arrival. A release after
  // the HALF wait's acquire, so the warps that see FULL complete see this half too. The peer's
  // buffer is free: this producer's EMPTY wait before the fill covered both CTAs' warps. The source
  // stays put until the copy is done: its next refill waits on an EMPTY that needs the peer's warps,
  // which read it only after the copy landed. The proxy fence orders what the wait made visible
  // before the copy, which reads it as TMA does. False if the wait gave up.
  auto push = [&](uint32_t g) -> bool {
    const uint32_t buf = sbase + (g % NST) * STRIDE, full = buf + STAGE, ho = rank * BOXH;
    if (!wait_bounded<false>(full + 16u, (g / NST) & 1u, waitNs, abortFlag, st, 3u, g)) return false;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    push_bulk(mapa(buf + ho, rank ^ 1u), buf + ho, BOXH, mapa(full, rank ^ 1u));
    mbar_arrive(full);
    return true;
  };
  // The fold's lane bases for a warp's column slot: eight ldmatrix a stage read its 4 KB once.
  const uint32_t f = (lane >> 1) & 3u;
  const uint32_t blcol = (lane & 7u) + ((lane >> 4) & 1u) * 8u;
  const uint32_t bLane = ((warp & 3u) * 64u + blcol) * SK + 16u * (((lane >> 3) & 1u) ^ f);
  const uint32_t S = tiles * KB;
  uint32_t sum = 0u, g = 0u;
  bool ok = true;
  if (threadIdx.x == PRODUCER) {
    fill(0u);
    if (S > 1u) fill(1u);
    if (PUSH) ok = push(0u);
  }
  const long long c0 = clock64();
  long long due = c0;
  for (; g < S; g++) {
    const uint32_t buf = sbase + (g % NST) * STRIDE;
    if (SEAM && g > 0u && g % KB == 0u) {
      // The fold's tile seam (cAH): warp w and w + 4 meet (pearl_bar_sync(1u + wc, 64u)), then
      // w hashes, here a sleep of hashCyc cycles, while w + 4 runs on into the next tile. Both
      // keep the stage pace from here. A warp leaves at the seam once a wait has given up; its
      // partner can have passed the test just before, and then waits at the barrier for good.
      const unsigned quit = __shfl_sync(0xFFFFFFFFu, lane == 0u ? *(volatile unsigned *)abortFlag : 0u, 0);
      if (quit) break;
      asm volatile("bar.sync %0, %1;" ::"r"(1u + (warp & 3u)), "r"(64u) : "memory");
      if (warp < 4u) {
        const long long end = clock64() + hashCyc;
        while (clock64() < end) __nanosleep(32);
      }
      due = clock64();
    }
    // p: the next stage's half goes to the peer first; it has had a stage to land.
    if (PUSH && threadIdx.x == PRODUCER && ok && g + 1u < S) ok = push(g + 1u);
    // The producer refills two stages ahead, once both CTAs (or this one) released the buffer.
    if (threadIdx.x == PRODUCER && ok && g + 2u < S) {
      const uint32_t n = g + 2u, empty = sbase + (n % NST) * STRIDE + STAGE + 8u;
      const uint32_t par = (n / NST - 1u) & 1u;
      // Acquired at cluster scope where the peer's warps arrive (the fold's wait in c, cA, cAH).
      if (n >= NST)
        ok = SYM || rank == 0u ? wait_bounded<true>(empty, par, waitNs, abortFlag, st, 1u, n)
                               : wait_bounded<false>(empty, par, waitNs, abortFlag, st, 1u, n);
      if (ok) fill(n);
    }
    const bool got = ok && wait_bounded<false>(buf + STAGE, (g / NST) & 1u, waitNs, abortFlag, st, 0u, g);
    if (__any_sync(0xFFFFFFFFu, !got)) break;
#pragma unroll
    for (uint32_t t = 0; t < 2u; t++)
#pragma unroll
      for (uint32_t q = 0; q < 4u; q++) {
        uint32_t r0, r1, r2, r3;
        ldmatrix_x4(r0, r1, r2, r3, ((buf + bLane) ^ (t * 32u)) + q * 16u * SK);
        sum += r0 + r1 + r2 + r3;
      }
    // The stage's mma, as time: sleep to the pace's schedule (none when behind it).
    due += pace;
    while (clock64() < due) __nanosleep(32);
    __syncwarp();
    if (lane == 0u) {
      const uint32_t empty = buf + STAGE + 8u;
      if (SYM) {
        mbar_arrive_remote(mapa(empty, rank ^ 1u));
        mbar_arrive_cluster(empty);
      } else {
        if (rank == 1u) mbar_arrive_remote(mapa(empty, 0u));
        mbar_arrive(empty);
      }
    }
    if (SEAM && g % KB == 1u && warp >= 4u) {
      // The fold's hand-off (cAH): at the end of the tile's first chunk every thread of warps
      // 4-7 waits, acquiring at CTA scope, until the tile's stage 0 is released, a phase that
      // counts the peer's arrivals too: so until the hashers have caught up.
      const uint32_t h = g - 1u;
      const uint32_t empty = sbase + (h % NST) * STRIDE + STAGE + 8u;
      const bool held = wait_bounded<false>(empty, (h / NST) & 1u, waitNs, abortFlag, st, 2u, h);
      if (__any_sync(0xFFFFFFFFu, !held)) break;
    }
  }
  // Every thread's words into the CTA's sum (mod 2^32, as the host adds them).
#pragma unroll
  for (uint32_t m = 16u; m > 0u; m >>= 1) sum += __shfl_xor_sync(0xFFFFFFFFu, sum, m);
  if (lane == 0u) atomicAdd(&rep[blockIdx.x].sum, sum);
  if (threadIdx.x == 0u) {
    rep[blockIdx.x].stages = g;
    rep[blockIdx.x].cycles = (unsigned long long)(clock64() - c0);
  }
  __syncwarp();
  // No CTA leaves while its peer may still arrive on it or multicast into it.
  cluster_sync();
#else
  (void)tmB;
  (void)tmA;
  (void)tmBh;
  (void)tiles;
  (void)pace;
  (void)hashCyc;
  (void)waitNs;
  (void)abortFlag;
  (void)rep;
  (void)st;
#endif
}

// ---------------------------------------------------------------------------------------------
// Host.
typedef void (*Kernel)(Map, Map, Map, uint32_t, uint32_t, uint32_t, unsigned long long, unsigned *, CtaReport *,
                       Status *);
Kernel kernel_for(int arm) {
  switch (arm) {
    case ARM_E: return mc_probe<ARM_E>;
    case ARM_A: return mc_probe<ARM_A>;
    case ARM_B: return mc_probe<ARM_B>;
    case ARM_D: return mc_probe<ARM_D>;
    case ARM_C: return mc_probe<ARM_C>;
    case ARM_CA: return mc_probe<ARM_CA>;
    case ARM_A2: return mc_probe<ARM_A2>;
    case ARM_H2: return mc_probe<ARM_H2>;
    case ARM_P: return mc_probe<ARM_P>;
    default: return mc_probe<ARM_CAH>;
  }
}

// NVML through dlopen, as probes/race.cu: the driver ships libnvidia-ml.so.1, and the toolkit
// the boxes install has no nvml.h. None on Windows (a compile check only there).
typedef int (*NvmlInitFn)();
typedef int (*NvmlHandleFn)(unsigned, void **);
typedef int (*NvmlClockFn)(void *, int, unsigned *);
typedef int (*NvmlPowerFn)(void *, unsigned *);
typedef int (*NvmlTempFn)(void *, int, unsigned *);
typedef int (*NvmlEnergyFn)(void *, unsigned long long *);
void *g_dev = nullptr;
NvmlClockFn nvClock;
NvmlPowerFn nvPower;
NvmlTempFn nvTemp;
NvmlEnergyFn nvEnergy;

bool nvml_open() {
#ifdef _WIN32
  return false;
#else
  void *h = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!h) return false;
  NvmlInitFn init = (NvmlInitFn)dlsym(h, "nvmlInit_v2");
  NvmlHandleFn handle = (NvmlHandleFn)dlsym(h, "nvmlDeviceGetHandleByIndex_v2");
  nvClock = (NvmlClockFn)dlsym(h, "nvmlDeviceGetClockInfo");
  nvPower = (NvmlPowerFn)dlsym(h, "nvmlDeviceGetPowerUsage");
  nvTemp = (NvmlTempFn)dlsym(h, "nvmlDeviceGetTemperature");
  nvEnergy = (NvmlEnergyFn)dlsym(h, "nvmlDeviceGetTotalEnergyConsumption");
  if (!init || !handle || !nvClock || !nvPower || !nvTemp) return false;
  // Index 0, as compilers.sh's nvidia-smi -i 0: the boxes have one card.
  return init() == 0 && handle(0, &g_dev) == 0;
#endif
}
// Total energy in joules, or a negative number when the driver has no counter.
double energy_j() {
  unsigned long long mj = 0;
  if (!g_dev || !nvEnergy || nvEnergy(g_dev, &mj) != 0) return -1.0;
  return mj / 1000.0;
}

// The sampler: while g_on, one sample every 20 ms into the sums (SM MHz, W, C).
std::atomic<bool> g_on{false}, g_quit{false};
std::mutex g_mu;
double g_sum[3];
unsigned g_n;
void sampler() {
  while (!g_quit) {
    if (g_on && g_dev) {
      unsigned mhz = 0, mw = 0, c = 0;
      if (nvClock(g_dev, 1 /* NVML_CLOCK_SM */, &mhz) == 0 && nvPower(g_dev, &mw) == 0
          && nvTemp(g_dev, 0 /* NVML_TEMPERATURE_GPU */, &c) == 0) {
        std::lock_guard<std::mutex> l(g_mu);
        g_sum[0] += mhz;
        g_sum[1] += mw / 1000.0;
        g_sum[2] += c;
        g_n++;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Probe {
  int sms = 0;
  uint32_t pace = 3240u, waitMs = 1000u, hashCyc = 3000u;
  double launchMs = 300.0;
  Map tmB{}, tmA{}, tmBh{};   // tmBh: B' in half boxes, 128 columns (h2, p)
  unsigned *abortDev = nullptr;
  CtaReport *repDev = nullptr;
  Status *stHost = nullptr, *stDev = nullptr;
  std::vector<uint32_t> colsum;   // per B' column group: the words of its 32 boxes, summed
  int clusters[NARMS] = {};
  bool dropped[NARMS] = {};
  double cyclesPerMs = 0.0;
  double stageCycles = 0.0;   // the pace, or e's own stage time when that is longer
};

// What one launch did: whether it ran to the end, its stages and cycles, and checksum misses.
struct LaunchOut {
  bool ran = false, timedOut = false;
  unsigned long long ctaStages = 0, cycles = 0;
  int bad = 0;
};

void print_status(const Probe &p, const char *arm) {
  const volatile Status *s = p.stHost;
  if (!s->set) return;
  printf("TIMEOUT arm=%s role=%s block=%u rank=%u thread=%u stage=%u parity=%u bar=0x%x state=0x%016llx "
         "waited_ms=%.0f\n",
         arm, s->role == 3u ? "producer-half" : s->role == 2u ? "handoff-empty" : s->role ? "producer-empty" : "consumer-full",
         s->block, s->rank,
         s->thread, s->stage, s->parity, s->bar, (unsigned long long)s->state, s->waited_ns / 1e6);
}

// One launch of an arm over `tiles` tiles a pair, with events around the kernel if given. Exits
// the process on a fault or a hang: the context is gone or stuck either way.
LaunchOut launch(Probe &p, const ArmInfo &a, uint32_t tiles, bool smoke, cudaEvent_t before = nullptr,
                 cudaEvent_t after = nullptr) {
  LaunchOut out;
  const int grid = 2 * p.clusters[a.arm];
  memset((void *)p.stHost, 0, sizeof(Status));
  cudaMemset(p.abortDev, 0, sizeof(unsigned));
  cudaMemset(p.repDev, 0, sizeof(CtaReport) * grid);
  cudaLaunchConfig_t cfg = {};
  cudaLaunchAttribute attr[1] = {};
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 2;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = SMEM;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  const double t0 = now_s();
  if (before) cudaEventRecord(before, 0);
  const cudaError_t lr = cudaLaunchKernelEx(&cfg, kernel_for(a.arm), p.tmB, p.tmA, p.tmBh, tiles, p.pace, p.hashCyc,
                                            (unsigned long long)p.waitMs * 1000000ull, p.abortDev, p.repDev,
                                            p.stDev);
  if (after) cudaEventRecord(after, 0);
  const cudaError_t le = cudaGetLastError();
  if (lr != cudaSuccess || le != cudaSuccess) {
    printf("FAULT arm=%s launch rc=%d (%s) last=%d (%s)\n", a.name, (int)lr, cudaGetErrorName(lr), (int)le,
           cudaGetErrorName(le));
    p.dropped[a.arm] = true;
    return out;
  }
  // Expected time, from the calibration (or a generous guess before it), and a deadline.
  const double expect =
      p.cyclesPerMs > 0.0 ? tiles * KB * p.stageCycles / p.cyclesPerMs / 1000.0 : 1.0;
  const double deadline = t0 + 4.0 * expect + 2.0 * p.waitMs / 1000.0 + 5.0;
  cudaError_t q;
  while ((q = cudaStreamQuery(0)) == cudaErrorNotReady) {
    if (now_s() > deadline) {
      print_status(p, a.name);
      printf("HANG arm=%s launch did not finish in %.1f s (expected %.2f s)%s\n", a.name, now_s() - t0, expect,
             p.stHost->set ? "" : "; no wait gave up");
      printf("DONE early\n");
      fflush(stdout);
      std::_Exit(3);
    }
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  if (q != cudaSuccess) {
    print_status(p, a.name);
    printf("FAULT arm=%s sync rc=%d (%s)\n", a.name, (int)q, cudaGetErrorName(q));
    printf("DONE early\n");
    fflush(stdout);
    std::_Exit(2);
  }
  std::vector<CtaReport> rep(grid);
  cudaMemcpy(rep.data(), p.repDev, sizeof(CtaReport) * grid, cudaMemcpyDeviceToHost);
  out.ran = true;
  out.timedOut = p.stHost->set != 0u;
  const uint32_t pairs = (uint32_t)grid / 2u, S = tiles * KB;
  for (int b = 0; b < grid; b++) {
    out.ctaStages += rep[b].stages;
    out.cycles += rep[b].cycles;
    if (rep[b].stages != S) {
      out.bad++;
      continue;
    }
    const uint32_t pair = (uint32_t)b / 2u, rank = (uint32_t)b % 2u;
    if (!(rank ? a.check1 : a.check0)) continue;
    uint32_t want = 0u;
    for (uint32_t j = 0; j < tiles; j++) want += 2u * p.colsum[(pair + j * pairs) % CG];
    if (rep[b].sum != want) out.bad++;
  }
  if (out.timedOut) {
    print_status(p, a.name);
    p.dropped[a.arm] = true;
  }
  if (smoke)
    printf("SMOKE arm=%s launch=%d sync=0 tiles=%u ctas=%d stages=%llu cyc_stage=%.0f bad=%d%s\n", a.name,
           (int)lr, tiles, grid, out.ctaStages, out.ctaStages ? (double)out.cycles / out.ctaStages : 0.0, out.bad,
           out.timedOut ? " timeout" : "");
  return out;
}

// A measured window: launches for `settle` seconds unmeasured, then for `secs` seconds measured.
void window(Probe &p, const ArmInfo &a, int round, double secs, double settle, uint32_t tiles) {
  for (double t0 = now_s(); now_s() - t0 < settle;) {
    if (!launch(p, a, tiles, false).ran || p.dropped[a.arm]) return;
  }
  {
    std::lock_guard<std::mutex> l(g_mu);
    g_sum[0] = g_sum[1] = g_sum[2] = 0.0;
    g_n = 0;
  }
  const double e0 = energy_j(), t0 = now_s();
  g_on = true;
  unsigned long long stages = 0, cycles = 0;
  int launches = 0, bad = 0;
  while (now_s() - t0 < secs) {
    const LaunchOut o = launch(p, a, tiles, false);
    if (!o.ran || p.dropped[a.arm]) {
      g_on = false;
      return;
    }
    stages += o.ctaStages;
    cycles += o.cycles;
    bad += o.bad;
    launches++;
  }
  g_on = false;
  const double t = now_s() - t0, e1 = energy_j();
  double mhz = 0.0, w = 0.0, c = 0.0;
  {
    std::lock_guard<std::mutex> l(g_mu);
    if (g_n) {
      mhz = g_sum[0] / g_n;
      w = g_sum[1] / g_n;
      c = g_sum[2] / g_n;
    }
  }
  const bool counter = e0 >= 0.0 && e1 >= e0;
  const double j = counter ? e1 - e0 : w * t;
  printf("ROUND r=%d arm=%s secs=%.3f launches=%d ctas=%d cta_stages=%llu fill_gb=%.3f l2_gb=%.3f j=%.3f "
         "j_src=%s w=%.2f mhz=%.0f c=%.0f cyc_stage=%.0f pace=%u bad=%d\n",
         round, a.name, t, launches, 2 * p.clusters[a.arm], stages, stages / 2.0 * a.fillPair / 1e9,
         stages / 2.0 * a.l2Pair / 1e9, j, counter ? "energy" : (g_n ? "power" : "none"), w, mhz, c,
         stages ? (double)cycles / stages : 0.0, p.pace, bad);
  fflush(stdout);
}

bool encode(PFN_cuTensorMapEncodeTiled_v12000 enc, Map *m, void *base, uint64_t rows, uint32_t boxRows) {
  const cuuint64_t dims[3] = {(cuuint64_t)SK, (cuuint64_t)rows, (cuuint64_t)KB};
  const cuuint64_t strides[2] = {(cuuint64_t)SK, (cuuint64_t)rows * SK};
  const cuuint32_t box[3] = {SK, boxRows, 1u};
  const cuuint32_t estr[3] = {1u, 1u, 1u};
  return enc(&m->map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, base, dims, strides, box, estr,
             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
         == CUDA_SUCCESS;
}

}  // namespace

int main(int argc, char **argv) {
  const double secs = argc > 1 ? atof(argv[1]) : 6.0;
  const int rounds = argc > 2 ? atoi(argv[2]) : 3;
  Probe p;
  p.pace = argc > 3 ? (uint32_t)atoi(argv[3]) : 3240u;
  const double settle = argc > 4 ? atof(argv[4]) : 1.5;
  p.waitMs = argc > 5 ? (uint32_t)atoi(argv[5]) : 1000u;
  p.launchMs = argc > 6 ? atof(argv[6]) : 300.0;
  p.hashCyc = argc > 7 ? (uint32_t)atoi(argv[7]) : 3000u;
  setvbuf(stdout, nullptr, _IOLBF, 0);
  // MC_ARMS: which phases run, "push" and/or "multicast" (both when unset or empty).
  const char *armsEnv = getenv("MC_ARMS");
  const bool all = !armsEnv || !*armsEnv;
  const bool runPush = all || strstr(armsEnv, "push"), runMc = all || strstr(armsEnv, "multicast");

  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    printf("FAULT no device\nDONE early\n");
    return 2;
  }
  int clusterLaunch = 0;
  cudaDeviceGetAttribute(&clusterLaunch, cudaDevAttrClusterLaunch, 0);
  p.sms = prop.multiProcessorCount;
  printf("CARD name=%s cc=%d.%d sms=%d l2_mb=%.0f cluster_launch=%d smem_optin=%zu pace=%u secs=%.1f rounds=%d "
         "settle=%.1f wait_ms=%u hash=%u own_dst=%s arms=%s%s\n",
         prop.name, prop.major, prop.minor, p.sms, prop.l2CacheSize / 1048576.0, clusterLaunch,
         (size_t)prop.sharedMemPerBlockOptin, p.pace, secs, rounds, settle, p.waitMs, p.hashCyc,
         MC_SHARED_CTA ? "cta" : "cluster", runPush ? "push" : "", runPush && runMc ? ",multicast" : runMc ? "multicast" : "");
  if (prop.major < 9) {
    printf("SKIP clusters and TMA need sm_90 or newer\nDONE\n");
    return 0;
  }
  const bool nvml = nvml_open();
  printf("NVML %s energy=%s\n", nvml ? "yes" : "no", energy_j() >= 0.0 ? "yes" : "no");

  // The operands: random bytes, k-blocked [KB][rows][64] as the fold reads them. The host keeps
  // each B' column group's word sum over its 32 boxes for the checksums.
  const size_t rowsB = (size_t)CG * BN, rowsA = (size_t)RG * BM;
  std::vector<uint8_t> hB(rowsB * SK * KB), hA(rowsA * SK * KB);
  uint64_t x = 0x9E3779B97F4A7C15ull;
  auto next = [&]() {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
  };
  for (size_t i = 0; i < hB.size(); i += 8) {
    const uint64_t v = next();
    memcpy(&hB[i], &v, 8);
  }
  for (size_t i = 0; i < hA.size(); i += 8) {
    const uint64_t v = next();
    memcpy(&hA[i], &v, 8);
  }
  p.colsum.assign(CG, 0u);
  for (uint32_t kb = 0; kb < KB; kb++)
    for (uint32_t cg = 0; cg < CG; cg++) {
      const uint8_t *box = &hB[((size_t)kb * rowsB + (size_t)cg * BN) * SK];
      for (uint32_t w = 0; w < BOXB / 4u; w++) {
        uint32_t v;
        memcpy(&v, box + 4u * w, 4);
        p.colsum[cg] += v;
      }
    }
  void *dB = nullptr, *dA = nullptr;
  if (cudaMalloc(&dB, hB.size()) != cudaSuccess || cudaMalloc(&dA, hA.size()) != cudaSuccess
      || cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice) != cudaSuccess
      || cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
    printf("FAULT operand upload\nDONE early\n");
    return 2;
  }
  void *encp = nullptr;
  cudaDriverEntryPointQueryResult qres;
  if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &encp, 12000, cudaEnableDefault, &qres)
          != cudaSuccess
      || qres != cudaDriverEntryPointSuccess
      || !encode(reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(encp), &p.tmB, dB, rowsB, BN)
      || !encode(reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(encp), &p.tmA, dA, rowsA, BM)
      || !encode(reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(encp), &p.tmBh, dB, rowsB, BNH)) {
    printf("FAULT tensor maps\nDONE early\n");
    return 2;
  }
  cudaMalloc(&p.abortDev, sizeof(unsigned));
  cudaMalloc(&p.repDev, sizeof(CtaReport) * (size_t)std::max(p.sms, 2) * 2);
  cudaHostAlloc(&p.stHost, sizeof(Status), cudaHostAllocMapped);
  cudaHostGetDevicePointer(&p.stDev, p.stHost, 0);

  // How many clusters of two fit, per arm, with the launch's own configuration. The grid is
  // whole clusters of what fits; a refused query falls back to half the SMs and says so.
  for (const ArmInfo &a : ARMS) {
    const Kernel k = kernel_for(a.arm);
    const cudaError_t fa = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM);
    cudaLaunchConfig_t cfg = {};
    cudaLaunchAttribute attr[1] = {};
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 2;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.gridDim = dim3(2);
    cfg.blockDim = dim3(THREADS);
    cfg.dynamicSmemBytes = SMEM;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    int n = 0;
    const cudaError_t oq = cudaOccupancyMaxActiveClusters(&n, k, &cfg);
    if (oq != cudaSuccess || n < 1) {
      (void)cudaGetLastError();
      p.clusters[a.arm] = std::max(p.sms / 2, 1);
      printf("CLUSTERS arm=%s rc=%d (%s) smem_attr=%d clusters=%d fallback=%d\n", a.name, (int)oq,
             cudaGetErrorName(oq), (int)fa, n, p.clusters[a.arm]);
    } else {
      p.clusters[a.arm] = n;
      printf("CLUSTERS arm=%s rc=0 (cudaSuccess) smem_attr=%d clusters=%d ctas=%d\n", a.name, (int)fa, n, 2 * n);
    }
  }

  // Calibration on e (no loads, cannot hang on a load): SM cycles a millisecond, from a short
  // launch's cycles and its time on the GPU, to size the launches.
  {
    const ArmInfo &e = ARMS[ARM_E];
    launch(p, e, 4u, true);   // warm
    const uint32_t tiles = 512u;
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    const LaunchOut o = launch(p, e, tiles, true, ev0, ev1);
    cudaEventSynchronize(ev1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    if (!o.ran || o.ctaStages == 0 || ms <= 0.0f) {
      printf("FAULT calibration\nDONE early\n");
      return 2;
    }
    p.cyclesPerMs = (double)o.cycles / (o.ctaStages / (tiles * KB)) / ms;
    p.stageCycles = std::max((double)p.pace, (double)o.cycles / o.ctaStages);
  }
  const uint32_t tiles = std::max(1u, (uint32_t)(p.launchMs * p.cyclesPerMs / (p.stageCycles * KB)));
  printf("LAUNCH tiles=%u stages=%u cycles_per_ms=%.0f stage_cycles=%.0f\n", tiles, tiles * KB,
         p.cyclesPerMs, p.stageCycles);
  // The sampler from here: every return before this leaves without it, and every exit after it
  // is either the end below or launch()'s _Exit.
  std::thread samp(sampler);

  // Phase 1: the loads (a, d, b) as a smoke launch each, then the rounds that give f.
  // Phase 2: the fold's barrier protocol (c) and the fold's whole stage (cA), with b again.
  auto rounds_over = [&](std::vector<int> arms, int first, int n) {
    for (int r = 0; r < n; r++) {
      std::vector<int> order = arms;
      if (r % 2) std::reverse(order.begin(), order.end());
      for (int arm : order)
        if (!p.dropped[arm]) window(p, ARMS[arm], first + r, secs, settle, tiles);
    }
  };
  // The push phase first (MC_ARMS): a2, h2 and p as a smoke launch each, p last (it is the one that
  // can stick), then rounds with e, numbered from 101 so they never share a round with the others.
  if (runPush) {
    for (int arm : {ARM_A2, ARM_H2, ARM_P}) launch(p, ARMS[arm], 8u, true);
    rounds_over({ARM_E, ARM_A2, ARM_H2, ARM_P}, 101, rounds);
  }
  if (runMc) {
    for (int arm : {ARM_A, ARM_D, ARM_B}) launch(p, ARMS[arm], 8u, true);
    rounds_over({ARM_E, ARM_A, ARM_D, ARM_B}, 1, rounds);
    for (int arm : {ARM_C, ARM_CA}) launch(p, ARMS[arm], 8u, true);
    const int phase2 = std::max(1, rounds - 1);
    rounds_over({ARM_B, ARM_C, ARM_CA}, rounds + 1, phase2);
    // Phase 3: the fold's tile seam on top (cAH), a smoke launch and one window, last of all: its
    // named barrier can leave a warp stuck after a timeout, which ends the process.
    launch(p, ARMS[ARM_CAH], 8u, true);
    rounds_over({ARM_CAH}, rounds + phase2 + 1, 1);
  }
  g_quit = true;
  samp.join();
  printf("DONE\n");
  return 0;
}
