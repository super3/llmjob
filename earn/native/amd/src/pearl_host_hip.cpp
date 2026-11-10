// Host-side driver for the AMD core: the same extern "C" entry points pearl_host.cu gives
// pearl_core.cc, on HIP. pearl_core.cc itself is compiled unchanged against this file, so
// the addon the app loads (pearl_core_hip.node) has exactly the CUDA core's interface.
//
// This is a port of pearl_host.cu with the NVIDIA folds taken out. What is the same, and
// why it matters, is explained there; the comments here say what differs:
//   - Which fold runs is read off the card's gfx architecture (resolve_fold), and checked
//     against the folds the loaded code object has bodies for (pearl_amd_caps).
//     PEARL_AMD_FOLD=ref|mfma16|mfma8|wmma11|wmma12|dot4 picks one by hand.
//   - A card whose fold has not run on its kind of card in a release (gfx9, gfx103x, or an
//     architecture the host could not read) mines only for an opener that checks every hit
//     before sending it (pearl_host_create_checked). PEARL_AMD_UNTESTED=1 also opens it for one
//     that checks nothing, for hashrate.js, which sends no shares (resolve_fold).
//   - A card whose code object has the wrap readout (RDNA 4) mines k 8192 on gfx1201 with 56+
//     CUs and k 4096 on the rest, when the miner lets the core choose (pearl_host_card_k); every
//     other card mines the profile's k. PEARL_AMD_K (2048, 4096 or 8192) picks the k instead.
//   - Every fold stores every region's transcript, and pearl_amd_hash hashes them after
//     it, so each pipeline slot has a transcript buffer: 64 bytes a region of the batch.
//     The batch is narrower than the CUDA core's to keep those buffers small
//     (PEARL_AMD_COL_BATCH).
//   - A' and B' are in per-tile order for the matrix folds (pearl_materialize16_tiled, see
//     PEARL_AMD_TILED in pearl_amd_fold.h) and row-major for the reference fold.
//   - Everything runs on one stream of the core's own, in order; proof reads use a side
//     stream and are waited for once a hit (HitReads).
//
// The same file builds two ways (build-amd.sh): with ROCm's hip-clang for real cards, and
// as plain C++ against HIP-CPU (__HIP_CPU_RT__), which runs the kernels on the CPU. The
// CPU build exists to check the core without a GPU, and is never shipped.

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "pearl_amd_kernels.h"

// GCC 11's libstdc++ headers make `new T[n]` and std::allocator call
// std::__throw_bad_array_new_length, which is GLIBCXX_3.4.29, so the Linux core built with them
// wouldn't load on Ubuntu 20.04 (GCC 10's libstdc++, 3.4.28), HiveOS's image: an RX 6700 XT rig
// on 9 November. Defined here, hidden, the core doesn't ask the system's libstdc++ for it. It
// runs only for an array size that overflows, which the core never asks for.
#if defined(__linux__) && defined(_GLIBCXX_RELEASE) && _GLIBCXX_RELEASE >= 11
namespace std {
__attribute__((visibility("hidden"), noreturn)) void __throw_bad_array_new_length() { abort(); }
}
#endif

// How many column offsets one batch covers at most. Each pipeline slot keeps a transcript
// buffer of col_batch x (m / 16) regions x 64 bytes: 128 MiB at 256 and the mainnet m.
// -DPEARL_AMD_COL_BATCH=N sets another width; the host narrows it further if it has to.
// Without it the host also picks a narrower width for the cards measured faster there
// (amd_col_batch); a build that sets it gets exactly that cap, so a sweep can still race 256.
// PEARL_AMD_HASHQ: the hash kernel on a queue of its own (Ctx::hashq). 0 on no card; 1 on every card; 2, the default,
// on RDNA 3's gfx1100 (RX 7900) only. The environment variable of the same name (0 or 1) overrides it at run time.
#ifndef PEARL_AMD_HASHQ
#define PEARL_AMD_HASHQ 2
#endif
#if PEARL_AMD_HASHQ != 0 && PEARL_AMD_HASHQ != 1 && PEARL_AMD_HASHQ != 2
#error "PEARL_AMD_HASHQ is 0, 1 or 2"
#endif
// PEARL_AMD_HASHQ_ZERO: with the hash on its own queue, the hit count is zeroed there too, just before the hash, and
// not before the fold (hipMemsetAsync is a kernel of its own, so it sat between every two folds). PEARL_AMD_HASHQ_WAIT:
// the fold's queue waits for the slot's last hash before the fold (hipStreamWaitEvent). The host frees a slot only in
// pearl_host_collect, which has waited for that hash, so the wait is a second check; 0 leaves it out. The environment
// variables of the same names (0 or 1) override both at run time, to race them on one build.
#ifndef PEARL_AMD_HASHQ_ZERO
#define PEARL_AMD_HASHQ_ZERO 0
#endif
#ifndef PEARL_AMD_HASHQ_WAIT
#define PEARL_AMD_HASHQ_WAIT 1
#endif
// PEARL_AMD_STRIPS: how many batches of the card's width one launch folds, side by side (Ctx::strips), on a card whose
// code object takes such a launch (PEARL_AMD_CAP_STRIP, pearl_amd_fold.h). 1 is a launch a batch. The environment
// variable of the same name (1 to 16) overrides it at run time.
#ifndef PEARL_AMD_STRIPS
#define PEARL_AMD_STRIPS 1
#endif
#if PEARL_AMD_STRIPS < 1 || PEARL_AMD_STRIPS > 16
#error "PEARL_AMD_STRIPS is 1 to 16"
#endif
// PEARL_AMD_TICK_US: a thread launches a one-thread kernel on a queue of its own every that many microseconds
// (Ctx::ticker), 0 for none. On an RX 7900 XTX PC where the fold's queue was held up (its slots in use 86-88% of the
// time, on one queue), the hash queue's launch every batch got most of that back; this asks whether a faster tick
// gets the rest. The environment variable of the same name (0, or 100 to 100000) overrides it at run time.
#ifndef PEARL_AMD_TICK_US
#define PEARL_AMD_TICK_US 0
#endif
#ifndef PEARL_AMD_COL_BATCH
#define PEARL_AMD_COL_BATCH 256u
#define PEARL_AMD_COL_BATCH_BY_ARCH 1
#else
#define PEARL_AMD_COL_BATCH_BY_ARCH 0
#endif

// The most chunks (k / rank) a card with the wrap readout takes: 64, k 8192 at rank 128, the most
// gfx1201 mines by default (pearl_host_card_k); more in a build made for a larger PEARL_FOLD_K. Without
// that readout a card takes 16.
#define PEARL_AMD_KWRAP_CHUNKS (PEARL_FOLD_CHUNKS > 64u ? PEARL_FOLD_CHUNKS : 64u)

// Mirrors the struct pearl_core.cc declares, as pearl_host.cu does.
struct PearlProofSide {
  std::vector<uint32_t> leaf_indices;
  std::vector<uint8_t> leaves;    // leaf_indices.size() * 1024
  std::vector<uint8_t> siblings;  // n * 32
  uint8_t root[PEARL_HASH_BYTES];
  uint64_t total_leaves;
};

struct PearlSearchResult {
  uint8_t jackpot_hash[PEARL_HASH_BYTES];
  uint8_t a_seed[PEARL_HASH_BYTES];
  uint8_t b_seed[PEARL_HASH_BYTES];
  uint64_t nonce;
  uint64_t salt;
  std::vector<uint8_t> proof;
  PearlProofSide proof_a;
  PearlProofSide proof_bt;
  bool found;
};

namespace {

// The folds, in the order of their PEARL_AMD_CAP_* bits. Where a fold hashes its transcripts depends on the
// card's code object (Ctx::fuse), so pearl_host_fold_name adds it.
struct FoldInfo {
  const char *name;
  uint32_t cap;
  uint32_t bm, bn;  // the fold's tile: rows of A', columns (rows of B')
  const char *text;
};
const FoldInfo kFolds[] = {
    {"ref", PEARL_AMD_CAP_REF, PEARL_AMD_REF_BM, PEARL_AMD_REF_BN,
     "ref 32x64, dot products from LDS"},
    {"mfma16", PEARL_AMD_CAP_MFMA16, PEARL_AMD_BM, PEARL_AMD_BN,
     "mfma 128x256, 4 waves of 64x128, v_mfma_i32_32x32x16_i8"},
    {"mfma8", PEARL_AMD_CAP_MFMA8, PEARL_AMD_BM, PEARL_AMD_BN,
     "mfma 128x256, 4 waves of 64x128, v_mfma_i32_32x32x8_i8"},
    {"wmma11", PEARL_AMD_CAP_WMMA11, PEARL_AMD_BM, PEARL_AMD_BN,
     "wmma 128x256, 8 waves of 64x64, v_wmma_i32_16x16x16_iu8 (RDNA3)"},
    {"wmma12", PEARL_AMD_CAP_WMMA12, PEARL_AMD_BM, PEARL_AMD_BN,
     "wmma 128x256, 8 waves of 64x64, v_wmma_i32_16x16x16_iu8 (RDNA4)"},
    {"dot4", PEARL_AMD_CAP_DOT4, PEARL_AMD_BM, PEARL_AMD_BN,
     "dot4 128x256, 8 waves of 64x64, v_dot4 from registers"},
};
const int kFoldCount = (int)(sizeof kFolds / sizeof kFolds[0]);

int fold_by_name(const char *name) {
  for (int i = 0; i < kFoldCount; i++)
    if (strcmp(kFolds[i].name, name) == 0) return i;
  return -1;
}

// The fold a gfx architecture runs by default: its matrix instruction; on RDNA2, which has
// none, the dot4 fold; and the reference fold on anything else.
int native_fold(const char *arch) {
  if (!strncmp(arch, "gfx942", 6) || !strncmp(arch, "gfx950", 6)) return fold_by_name("mfma16");
  if (!strncmp(arch, "gfx90a", 6) || !strncmp(arch, "gfx908", 6)) return fold_by_name("mfma8");
  if (!strncmp(arch, "gfx11", 5)) return fold_by_name("wmma11");
  if (!strncmp(arch, "gfx12", 5)) return fold_by_name("wmma12");
  if (!strncmp(arch, "gfx103", 6)) return fold_by_name("dot4");
  return fold_by_name("ref");
}

struct Ctx {
  PearlProfile profile;
  int device = 0;
  char arch[64] = {0};      // gcnArchName, e.g. "gfx942:sramecc+:xnack-"
  int fold = -1;            // index into kFolds; -1 until resolve_fold settles it
  bool emulated = false;    // the HIP-CPU build
  bool readoutDrop = false; // a PEARL_AMD_READOUT_DROP build: timing only (pearl_host_submit)
  bool timing = false;      // a PEARL_AMD_CAP_TIMING build (PEARL_AMD_ABLATE): timing only, the same way
  bool kwrap = false;       // the card's code object has the wrap readout, so k may pass 2048
  bool rank128 = false;     // and its matrix fold takes rank PEARL_FOLD_RANK only (PEARL_AMD_CAP_RANK)
  bool fuse = false;        // and every fold but ref hashes its own transcripts (PEARL_AMD_CAP_FUSE)
  bool checkHits = false;   // the opener recomputes every hit before it sends it (pearl_host_create_checked)
  bool untested = false;    // the card's fold has not run on its kind of card, and mines because of checkHits
#if PEARL_AMD_STAMP
  bool stamp = false;           // a PEARL_AMD_CAP_STAMP build: the folds store clock stamps (pearl_amd_fold.h)
  uint32_t *dStamps = nullptr;  // where they go: PEARL_AMD_STAMP_WORDS words for each wave of a batch
  // PEARL_AMD_STAMP_DUMP=file with PEARL_AMD_STAMP_DUMP_AT=n: once n batches have been collected, the search waits
  // for what is queued and writes the stamps of the last launch to the file, in the fold test's format, for
  // test/stamp-summary.js. It is how the miner's own launches are read (the fold test's run back to back on one queue).
  size_t stampBlocks = 0;
  uint64_t stampSeen = 0, stampAt = 0;
#endif
  char foldErr[512] = {0};
  char foldName[320] = {0};

  int8_t *dA = nullptr;
  int8_t *dB = nullptr;
  bool compact = false;
  int8_t *dAp = nullptr;
  int8_t *dBp = nullptr;
  int8_t *dEAL = nullptr;
  int8_t *dEBR = nullptr;
  uint32_t *dPermA = nullptr;
  uint32_t *dPermB = nullptr;
  uint8_t *dLabelA = nullptr;
  uint8_t *dLabelB = nullptr;
  uint8_t *dSaltA = nullptr;
  uint8_t *dSaltB = nullptr;
  uint8_t *dBoundA = nullptr;
  uint8_t *dBoundB = nullptr;
  uint32_t *dHitTranscript = nullptr;
  uint8_t *dHashes = nullptr;
  uint32_t *dHitCount = nullptr;
  uint32_t *dHitIndex = nullptr;
  uint32_t *dCaps = nullptr;
  uint32_t colBatch = 1;
  uint32_t rowsValid = 1;
  uint32_t colsValid = 1;
  std::vector<uint32_t> hHitIndex;
  uint32_t batch = 0;
  uint32_t *dJobKey = nullptr;
  uint32_t *dASeed = nullptr;
  uint32_t *dBSeed = nullptr;
  uint32_t *dCvs = nullptr;
  uint32_t *dTreeA = nullptr;
  uint32_t *dTreeB = nullptr;
  std::vector<uint64_t> layerOffA;
  std::vector<uint64_t> layerOffB;
  uint8_t *dSeedBuf = nullptr;
  uint8_t *dHashA = nullptr;
  uint8_t *dHashB = nullptr;
  uint64_t cvCapacity = 0;
  uint8_t aSeed[PEARL_HASH_BYTES] = {0};
  uint8_t bSeed[PEARL_HASH_BYTES] = {0};
  bool haveJob = false;
  uint8_t header[PEARL_HEADER_BYTES] = {0};
  uint8_t target[PEARL_HASH_BYTES] = {0};
  uint64_t salt = 0;
  uint8_t *dSeedInput = nullptr;
  bool baseDrawn = false;

  static const int kSlots = 2;
  struct Pending {
    uint64_t nonceBase;
    uint64_t salt;
    uint8_t aSeed[PEARL_HASH_BYTES];
    uint32_t regions;
  };
  Pending pend[kSlots];
  int pendHead = 0, pendCount = 0;
  hipEvent_t slotDone[kSlots] = {};
  uint32_t *hSlotCount = nullptr;  // pinned, [kSlots]
  hipStream_t side = nullptr;
  // Every kernel and every asynchronous copy runs here, in order. A stream of its own
  // rather than the null stream: HIP-CPU can run work queued on the null stream while it
  // is still running earlier null-stream work (its scheduler drains that stream from two
  // threads), which let a redraw rewrite A' under a fold still reading it. A blocking
  // stream behaves as the null stream did on a real GPU: the synchronous copies wait for it.
  //
  // A stream for each pipeline slot, as the CUDA core has, lets a batch's hash overlap the
  // next fold. On an RX 9070 XT that measured no faster (100.36 against 100.35 TH/s, three
  // rounds each, 23 October), so the batches stay on this one stream.
  hipStream_t work = nullptr;
  // PEARL_AMD_HASHQ (above; PR #290 research item 23, 10 November): the hash kernel, the hit count's copy and the
  // slot's event on a queue of their own, after the fold's event, so the next batch's fold follows this one on `work`
  // with no hash between them. The slot's fold waits for that slot's last hash (slotDone) before it rewrites its
  // transcripts and hit list. On an RX 7900 XTX (PC 44d3716f) a batch fell from 10.67 to 10.33 ms and hashrate.js
  // rose 3.5% (51.39-51.64 to 53.22-53.28 TH/s, 5 rounds, a twin within 0.2%): the fold itself ran 10.57 -> 10.31 ms,
  // more than the 0.084 ms hash it no longer waits for. A second run on that PC read +3.4%. On two RX 7800 XTs it was
  // level (72dd708c, within 0.03%) and behind (16f1685c, -0.03 to -0.27%): the fold got 0.1 ms slower with the hash
  // beside it. On an RX 9060 XT (8c96a2a7, k 4096) it lost 0.21-0.36% in 6 of 6 rounds. A second RX 7900 XTX
  // (3bf70206) read +0.2 to +1.0% in 10 of 10 rounds (56.2 to 56.4 TH/s, a twin level), so it is on for gfx1100
  // only. No RX 7900 XT or GRE has run it (benchmark.md, PR #290 item 23).
  hipStream_t hashq = nullptr;
  hipEvent_t foldDone[kSlots] = {};
  bool hashqZero = false;   // PEARL_AMD_HASHQ_ZERO: the hit count is zeroed on the hash queue
  bool hashqWait = true;    // PEARL_AMD_HASHQ_WAIT: the fold's queue waits for the slot's last hash
  // PEARL_AMD_STRIPS: a launch folds `strips` batches of stripBatch column offsets, each a strip of the grid's z
  // (colBatch = strips x stripBatch). The blocks of one strip run before the next strip's, so the B' in use at any
  // time is one strip's, as in a launch of one batch.
  // PEARL_AMD_TICK_US (above): the ticker, its queue and the two words its kernel writes.
  std::thread ticker;
  std::atomic<bool> tickStop{false};
  hipStream_t tickq = nullptr;
  uint32_t *dTick = nullptr;
  uint32_t tickUs = 0;
  bool stripCap = false;    // the card's code object takes such a launch (PEARL_AMD_CAP_STRIP)
  uint32_t strips = 1;
  uint32_t stripBatch = 1;
  uint32_t *dTr[kSlots] = {nullptr, nullptr};  // each slot's transcripts, 16 words a region
  std::vector<PearlSearchResult> extraHits;
  size_t extraNext = 0;
  // Pinned landing space for a hit's device reads (HitReads), so they are all queued and
  // waited for once. Allocated on the first hit; null if pinned memory was refused.
  uint8_t *hStage = nullptr;
  size_t hStageBytes = 0;

  // PEARL_AMD_TRACE=1: every ~10 s, one line on stderr with what the GPU and the search
  // thread spent their time on (trace_line). Timing events exist only then.
  struct Trace {
    bool on = false;
    hipEvent_t f0[kSlots] = {}, f1[kSlots] = {}, h1[kSlots] = {};
    double t0 = 0, foldMs = 0, hashMs = 0, spanMs = 0, waitMs = 0, drawMs = 0, jobMs = 0, hitMs = 0;
    double idleMs = 0;
    uint32_t batches = 0, draws = 0, jobs = 0, hits = 0, lines = 0, gaps = 0;
  } trace;

  bool hostSeeds = false;
  uint32_t jobKeyW[8] = {0};
  uint8_t leaf0[1024] = {0};
  uint32_t sib[PEARL_RESTAMP_MAX_LEVELS][8] = {};
  static const int kRecords = 4;
  PearlRestampRecord rec[kRecords] = {};
  uint64_t recSalt[kRecords] = {0};
  bool recValid[kRecords] = {false, false, false, false};
  int recNext = 0;
};

bool pearl_compact_operands(const PearlProfile *profile) {
  const uint32_t k = profile->k;
  return profile->operand_fill == PEARL_OPERAND_CONST && k >= 16u && (k & (k - 1u)) == 0u;
}

const int8_t *operand_chunk(const Ctx *ctx, const int8_t *operand, uint64_t i) {
  return ctx->compact ? operand + (i == 0 ? 0 : 1024) : operand + i * 1024;
}

bool fail(char *err, size_t err_len, const char *msg) {
  if (err && err_len) snprintf(err, err_len, "%s", msg);
  return false;
}

double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// hipEventQuery, which HIP-CPU doesn't have: there it waits for the event instead.
hipError_t event_query(hipEvent_t ev) {
#if defined(__HIP_CPU_RT__)
  return hipEventSynchronize(ev);
#else
  return hipEventQuery(ev);
#endif
}

// The batch width: the profile's col_batch, at most the valid column offsets, at most
// PEARL_AMD_COL_BATCH, halved until it divides the valid offsets. A batch must be whole
// tiles of every fold (16 column offsets, the matrix folds' 256 columns), so a width
// under 16 is refused when the search starts.
//
// arch is the card's gcnArchName ("gfx1100" or "gfx1100:xnack-"). On gfx1100 (RX 7900 XTX,
// 7900 XT, 7900 GRE) and gfx1101 (RX 7800 XT, 7700 XT) the cap is 128: in the miner's own loop
// 128 read +0.47 to +0.64% and +1.44 to +1.84% a round on two 7900 XTX PCs and +0.12 to +0.37%
// on a 7800 XT, ahead in every round, and 64 was slower on both chips (earn/tuning/benchmark.md, AMD, item
// 13). RDNA 4 was level at 128 (a 9060 XT, README.md; a 9070 XT, earn/tuning/benchmark.md), and the other
// targets weren't measured, so they keep 256.
uint32_t amd_col_batch(const PearlProfile *profile, const char *arch) {
  const uint32_t colsValid = profile->n / PEARL_COLS_COUNT;
  uint32_t cb = profile->col_batch ? profile->col_batch : 1u;
  // PEARL_AMD_COL_BATCH in the environment lowers the cap, for measuring widths on one build.
  // It never raises it: the two transcript buffers are sized by the width (128 MiB each at
  // 256), so 2048 would ask for 1 GiB each. Its leading digits are read by hand: atoi under
  // glibc 2.38's headers calls __isoc23_strtol, which would tie the addon to glibc 2.38
  // (pearl_host.cu's pearl_bd_sweep_env says more).
  uint32_t limit = PEARL_AMD_COL_BATCH;
#if PEARL_AMD_COL_BATCH_BY_ARCH
  if (arch && (strncmp(arch, "gfx1100", 7) == 0 || strncmp(arch, "gfx1101", 7) == 0)
      && (arch[7] == '\0' || arch[7] == ':'))
    limit = 128u;
#endif
  const char *env = getenv("PEARL_AMD_COL_BATCH");
  uint32_t want = 0;
  for (const char *p = env; p && *p >= '0' && *p <= '9' && want < limit; p++) want = want * 10u + (uint32_t)(*p - '0');
  if (want > 0 && want < limit) limit = want;
  if (cb > colsValid) cb = colsValid;
  if (cb > limit) cb = limit;
  while (cb > 1u && colsValid % cb != 0u) cb /= 2u;
  return cb;
}

// How many batches of the card's width (ctx->stripBatch) one launch folds: PEARL_AMD_STRIPS, or the environment's, on
// a card whose code object takes strips and runs a matrix fold or dot4; 1 on any other. A power of two, halved until
// the launch divides the valid column offsets, as a batch must.
uint32_t amd_strips(const Ctx *ctx) {
  if (!ctx->stripCap || ctx->fold < 0 || kFolds[ctx->fold].cap == PEARL_AMD_CAP_REF) return 1u;
  // A strip is whole tiles of the fold.
  if ((ctx->stripBatch * PEARL_COLS_COUNT) % kFolds[ctx->fold].bn != 0u) return 1u;
  uint32_t n = PEARL_AMD_STRIPS;
  const char *env = getenv("PEARL_AMD_STRIPS");
  uint32_t want = 0;
  for (const char *p = env; p && *p >= '0' && *p <= '9' && want <= 16u; p++) want = want * 10u + (uint32_t)(*p - '0');
  if (want >= 1u && want <= 16u) n = want;
  while (n > 1u && ((n & (n - 1u)) != 0u || ctx->colsValid % (ctx->stripBatch * n) != 0u)) n--;
  return n;
}

// What one instance of `profile` costs on a card, in bytes: pearl_host.cu's needed_bytes
// for row-major operands, plus the two transcript buffers.
size_t needed_bytes(const PearlProfile *profile) {
  const size_t k = profile->k;
  const size_t rank = profile->rank;
  const size_t aBytes = (size_t)profile->m * k;
  const size_t bBytes = (size_t)profile->n * k;
  const bool compact = pearl_compact_operands(profile);
  const size_t aStored = compact ? 2048 : aBytes, bStored = compact ? 2048 : bBytes;
  const size_t noiseBytes = (size_t)profile->m * rank + (size_t)profile->n * rank
                            + 2 * k * 2 * sizeof(uint32_t) + 64;
  const size_t primeBytes = aBytes + bBytes;
  const size_t batchBytes =
      (size_t)Ctx::kSlots * PEARL_MAX_HITS
      * (PEARL_HASH_BYTES + 2 * sizeof(uint32_t) + PEARL_JACKPOT_BUCKETS * sizeof(uint32_t));
  const size_t aLeaves = aBytes / 1024, bLeaves = bBytes / 1024;
  const size_t treeBytes = 2 * (aLeaves + bLeaves) * 32
                           + (aLeaves > bLeaves ? aLeaves : bLeaves) * 32;
  // The width before any per-card cap (nullptr): the most a card can need, so the check never
  // passes a card that then runs short.
  const size_t trBytes = (size_t)Ctx::kSlots * amd_col_batch(profile, nullptr)
                         * (profile->m / PEARL_ROWS_COUNT) * PEARL_JACKPOT_BUCKETS * 4u;
  return aStored + bStored + primeBytes + noiseBytes + batchBytes + treeBytes + trBytes
         + (1u << 20);
}

void operand_commitment(Ctx *ctx, const uint8_t *data, size_t len, uint8_t *out32,
                        uint32_t *tree, std::vector<uint64_t> *offsets) {
  const uint64_t chunks = len / 1024;
  const uint32_t threads = 256;
  if (offsets) offsets->clear();
  if (chunks <= 1) {
    hipLaunchKernelGGL(pearl_hash_operands, dim3(1), dim3(1), 0, ctx->work, (const uint32_t *)ctx->dJobKey,
                       data, (uint32_t)len, out32);
    return;
  }
  uint32_t *dst = tree ? tree : ctx->dCvs;
  uint64_t base = 0;
  if (offsets) offsets->push_back(0);
  hipLaunchKernelGGL(pearl_blake3_chunk_cvs, dim3((unsigned)((chunks + threads - 1) / threads)),
                     dim3(threads), 0, ctx->work, (const uint32_t *)ctx->dJobKey, data, chunks, dst,
                     ctx->compact ? 1u : 0u);
  uint64_t count = chunks;
  while (count > 1) {
    const uint64_t pairs = count / 2;
    const uint32_t isRoot = (pairs == 1) ? 1u : 0u;
    const uint64_t next = tree ? base + count : 0;
    hipLaunchKernelGGL(pearl_blake3_parent_layer, dim3((unsigned)((pairs + threads - 1) / threads)),
                       dim3(threads), 0, ctx->work, (const uint32_t *)ctx->dJobKey,
                       (const uint32_t *)(dst + base * 8), pairs, isRoot, dst + next * 8);
    if (offsets) offsets->push_back(next);
    base = next;
    count = pairs;
  }
  hipMemcpy(out32, dst + base * 8, PEARL_HASH_BYTES, hipMemcpyDeviceToDevice);
}

// ---------------------------------------------------------------------------------------
// BLAKE3 on the host, for host_record only. A copy of pearl_host.cu's; it is checked
// against the device once a job (setup_host_seeds) and not trusted until it agrees.
// ---------------------------------------------------------------------------------------
const uint32_t kB3Iv[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                           0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};
const uint32_t kB3ChunkStart = 1u, kB3ChunkEnd = 2u, kB3Parent = 4u, kB3Root = 8u,
               kB3Keyed = 16u;

inline uint32_t hb3_rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void hb3_compress(const uint32_t cv[8], const uint32_t block[16], uint64_t counter,
                  uint32_t len, uint32_t flags, uint32_t out[16]) {
  static const int perm[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
  uint32_t v[16], m[16], t[16];
  for (int i = 0; i < 8; i++) v[i] = cv[i];
  for (int i = 0; i < 4; i++) v[8 + i] = kB3Iv[i];
  v[12] = (uint32_t)counter; v[13] = (uint32_t)(counter >> 32); v[14] = len; v[15] = flags;
  for (int i = 0; i < 16; i++) m[i] = block[i];
  auto g = [&](int a, int b, int c, int d, uint32_t x, uint32_t y) {
    v[a] = v[a] + v[b] + x; v[d] = hb3_rotr(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];     v[b] = hb3_rotr(v[b] ^ v[c], 12);
    v[a] = v[a] + v[b] + y; v[d] = hb3_rotr(v[d] ^ v[a], 8);
    v[c] = v[c] + v[d];     v[b] = hb3_rotr(v[b] ^ v[c], 7);
  };
  for (int r = 0; r < 7; r++) {
    g(0, 4, 8, 12, m[0], m[1]);   g(1, 5, 9, 13, m[2], m[3]);
    g(2, 6, 10, 14, m[4], m[5]);  g(3, 7, 11, 15, m[6], m[7]);
    g(0, 5, 10, 15, m[8], m[9]);  g(1, 6, 11, 12, m[10], m[11]);
    g(2, 7, 8, 13, m[12], m[13]); g(3, 4, 9, 14, m[14], m[15]);
    for (int i = 0; i < 16; i++) t[i] = m[perm[i]];
    for (int i = 0; i < 16; i++) m[i] = t[i];
  }
  for (int i = 0; i < 8; i++) { out[i] = v[i] ^ v[i + 8]; out[i + 8] = v[i + 8] ^ cv[i]; }
}

void hb3_words(const uint8_t *b, uint32_t *w, int n) {
  for (int i = 0; i < n; i++)
    w[i] = (uint32_t)b[4 * i] | ((uint32_t)b[4 * i + 1] << 8) | ((uint32_t)b[4 * i + 2] << 16)
           | ((uint32_t)b[4 * i + 3] << 24);
}

void hb3_chunk_cv(const uint32_t key[8], const uint8_t *chunk, uint64_t counter, uint32_t out[8]) {
  uint32_t cv[8], block[16], o[16];
  for (int i = 0; i < 8; i++) cv[i] = key[i];
  for (int b = 0; b < 16; b++) {
    hb3_words(chunk + b * 64, block, 16);
    const uint32_t flags = kB3Keyed | (b == 0 ? kB3ChunkStart : 0u) | (b == 15 ? kB3ChunkEnd : 0u);
    hb3_compress(cv, block, counter, 64, flags, o);
    for (int i = 0; i < 8; i++) cv[i] = o[i];
  }
  for (int i = 0; i < 8; i++) out[i] = cv[i];
}

void hb3_hash64(const uint32_t *key, const uint8_t msg[64], uint32_t out[8]) {
  uint32_t block[16], o[16];
  hb3_words(msg, block, 16);
  hb3_compress(key ? key : kB3Iv, block, 0, 64,
               kB3ChunkStart | kB3ChunkEnd | kB3Root | (key ? kB3Keyed : 0u), o);
  for (int i = 0; i < 8; i++) out[i] = o[i];
}

void hb3_bytes(const uint32_t w[8], uint8_t out[32]) {
  for (int i = 0; i < 32; i++) out[i] = (uint8_t)(w[i >> 2] >> ((i & 3) * 8));
}

void host_record(const Ctx *ctx, uint64_t salt, bool stamp, PearlRestampRecord *r) {
  uint8_t leaf[1024];
  memcpy(leaf, ctx->leaf0, sizeof leaf);
  if (stamp)
    for (int i = 0; i < PEARL_STAMP_BYTES; i++) leaf[i] = (uint8_t)pearl_stamp_byte(salt, i);
  memcpy(r->head, leaf, sizeof r->head);
  const uint32_t levels = (uint32_t)ctx->layerOffA.size();
  r->levels = levels;
  for (uint32_t L = 0; L < levels; L++) r->node_off[L] = (uint32_t)ctx->layerOffA[L];
  hb3_chunk_cv(ctx->jobKeyW, leaf, 0, r->path[0]);
  for (uint32_t L = 0; L + 1 < levels; L++) {
    uint32_t block[16], o[16];
    for (int i = 0; i < 8; i++) { block[i] = r->path[L][i]; block[8 + i] = ctx->sib[L][i]; }
    const uint32_t flags = kB3Parent | kB3Keyed | (L + 2 == levels ? kB3Root : 0u);
    hb3_compress(ctx->jobKeyW, block, 0, 64, flags, o);
    for (int i = 0; i < 8; i++) r->path[L + 1][i] = o[i];
  }
  uint8_t root[32], bound[32];
  hb3_bytes(r->path[levels - 1], root);
  if (ctx->profile.seed_derivation == PEARL_SEED_LEGACY) {
    memcpy(bound, root, 32);
  } else {
    uint8_t msg[64] = {0};
    memcpy(msg, root, 32);
    const uint32_t dim = ctx->profile.m;
    msg[32] = (uint8_t)dim; msg[33] = (uint8_t)(dim >> 8);
    msg[34] = (uint8_t)(dim >> 16); msg[35] = (uint8_t)(dim >> 24);
    uint32_t key[8], bw[8];
    hb3_words(PEARL_SEED_SALT_A, key, 8);
    hb3_hash64(key, msg, bw);
    hb3_bytes(bw, bound);
  }
  hb3_words(bound, r->bound, 8);
  uint8_t msg[64];
  memcpy(msg, ctx->bSeed, 32);
  memcpy(msg + 32, bound, 32);
  hb3_hash64(nullptr, msg, r->a_seed);
}

void keep_record(Ctx *ctx, uint64_t salt, const PearlRestampRecord &r) {
  const int i = ctx->recNext;
  ctx->rec[i] = r;
  ctx->recSalt[i] = salt;
  ctx->recValid[i] = true;
  ctx->recNext = (i + 1) % Ctx::kRecords;
}

const PearlRestampRecord *find_record(const Ctx *ctx, uint64_t salt) {
  for (int i = 0; i < Ctx::kRecords; i++)
    if (ctx->recValid[i] && ctx->recSalt[i] == salt) return &ctx->rec[i];
  return nullptr;
}

void side_copy(Ctx *ctx, void *dst, const void *src, size_t n) {
  hipMemcpyAsync(dst, src, n, hipMemcpyDeviceToHost, ctx->side);
  hipStreamSynchronize(ctx->side);
}

// The device reads one hit needs (pearl_host.cu's HitReads): its hash and transcript, the
// proof's leaves and sibling nodes, and the roots, about 120 of them. Each was its own copy
// and its own wait, and a wait costs far more under WSL2 (every Salad PC) than on Linux.
// Queued, they cost one wait. A read lands at a fixed address, or at an offset into a
// vector that may still grow until the reads run.
struct HitReads {
  struct Read {
    std::vector<uint8_t> *vec;
    uint8_t *ptr;
    size_t off;
    const void *src;
    size_t n;
  };
  // Host bytes laid over a read once it has landed: leaf 0's head from a restamp record.
  struct Patch {
    std::vector<uint8_t> *vec;
    size_t off;
    uint8_t bytes[sizeof(PearlRestampRecord::head)];
  };
  std::vector<Read> reads;
  std::vector<Patch> patches;
  size_t bytes = 0;
  void to(void *dst, const void *src, size_t n) {
    reads.push_back({nullptr, static_cast<uint8_t *>(dst), 0, src, n});
    bytes += n;
  }
  void into(std::vector<uint8_t> *vec, size_t off, const void *src, size_t n) {
    reads.push_back({vec, nullptr, off, src, n});
    bytes += n;
  }
};

// Run a hit's reads: all queued on the side stream into the pinned stage, one wait, then
// each copied to where it belongs and the patches laid over them. Without a stage (pinned
// memory refused) each read is a side_copy, as before.
void run_reads(Ctx *ctx, HitReads *r) {
  auto land = [](const HitReads::Read &x) { return x.vec ? x.vec->data() + x.off : x.ptr; };
  if (r->bytes > ctx->hStageBytes) {
    if (ctx->hStage) hipHostFree(ctx->hStage);
    ctx->hStage = nullptr;
    ctx->hStageBytes = 0;
    const size_t want = r->bytes > ((size_t)128 << 10) ? r->bytes : ((size_t)128 << 10);
    if (hipHostMalloc(reinterpret_cast<void **>(&ctx->hStage), want, hipHostMallocDefault) == hipSuccess) {
      ctx->hStageBytes = want;
    } else {
      ctx->hStage = nullptr;
      (void)hipGetLastError();
    }
  }
  if (ctx->hStage) {
    size_t off = 0;
    for (const HitReads::Read &x : r->reads) {
      hipMemcpyAsync(ctx->hStage + off, x.src, x.n, hipMemcpyDeviceToHost, ctx->side);
      off += x.n;
    }
    hipStreamSynchronize(ctx->side);
    off = 0;
    for (const HitReads::Read &x : r->reads) {
      memcpy(land(x), ctx->hStage + off, x.n);
      off += x.n;
    }
  } else {
    for (const HitReads::Read &x : r->reads) side_copy(ctx, land(x), x.src, x.n);
  }
  for (const HitReads::Patch &p : r->patches) memcpy(p.vec->data() + p.off, p.bytes, sizeof p.bytes);
}

void leafIndicesForRows(const uint32_t *rows, uint32_t nrows, uint32_t k,
                        std::vector<uint32_t> *out) {
  std::set<uint32_t> s;
  for (uint32_t i = 0; i < nrows; i++) {
    const uint64_t first = (uint64_t)rows[i] * k / 1024;
    const uint64_t last = ((uint64_t)(rows[i] + 1) * k - 1) / 1024;
    for (uint64_t j = first; j <= last; j++) s.insert((uint32_t)j);
  }
  out->assign(s.begin(), s.end());
}

// pearl_host.cu's snapshotProof: one side's proof, with the salt-dependent part of A
// (leaf 0's head and node 0 of every level) from the hit's own salt's record. The device
// reads go into `reads`, which the caller runs once it has queued all of the hit's
// (run_reads); until then the leaf and sibling bytes are placeholders.
void snapshotProof(Ctx *ctx, bool isA, const uint32_t *rows, uint32_t nrows,
                   std::vector<uint32_t> *leafIdx, std::vector<uint8_t> *leaves,
                   std::vector<uint8_t> *sibs, const PearlRestampRecord *rec, HitReads *reads) {
  if (!isA) rec = nullptr;
  const uint32_t k = ctx->profile.k;
  const int8_t *operand = isA ? ctx->dA : ctx->dB;
  const uint32_t *tree = isA ? ctx->dTreeA : ctx->dTreeB;
  const std::vector<uint64_t> &offs = isA ? ctx->layerOffA : ctx->layerOffB;
  const uint64_t totalLeaves = (uint64_t)(isA ? ctx->profile.m : ctx->profile.n) * k / 1024;

  leafIndicesForRows(rows, nrows, k, leafIdx);
  leaves->resize(leafIdx->size() * 1024);
  for (size_t i = 0; i < leafIdx->size(); i++) {
    reads->into(leaves, i * 1024, operand_chunk(ctx, operand, (*leafIdx)[i]), 1024);
    if (rec && (*leafIdx)[i] == 0) {
      HitReads::Patch patch;
      patch.vec = leaves;
      patch.off = i * 1024;
      memcpy(patch.bytes, rec->head, sizeof patch.bytes);
      reads->patches.push_back(patch);
    }
  }

  sibs->clear();
  std::vector<uint32_t> current = *leafIdx;
  uint64_t levelLen = totalLeaves;
  uint32_t level = 0;
  while (levelLen > 1 && !current.empty() && level + 1 < offs.size()) {
    const std::set<uint32_t> live(current.begin(), current.end());
    for (uint32_t i : current) {
      uint32_t want;
      if (i % 2 == 1) {
        if (live.count(i - 1)) continue;
        want = i - 1;
      } else {
        if (live.count(i + 1) || (uint64_t)i + 1 >= levelLen) continue;
        want = i + 1;
      }
      if (rec && want == 0) {
        uint8_t node[PEARL_HASH_BYTES];
        hb3_bytes(rec->path[level], node);
        sibs->insert(sibs->end(), node, node + PEARL_HASH_BYTES);
      } else {
        const size_t at = sibs->size();
        sibs->resize(at + PEARL_HASH_BYTES);
        reads->into(sibs, at, tree + (offs[level] + want) * 8, PEARL_HASH_BYTES);
      }
    }
    std::set<uint32_t> next;
    for (uint32_t i : current) next.insert(i / 2);
    current.assign(next.begin(), next.end());
    levelLen = (levelLen + 1) / 2;
    level++;
  }
}

// The calling thread's current device for the length of one call (see pearl_host.cu).
struct DeviceScope {
  int prev = -1;
  explicit DeviceScope(int device) {
    if (hipGetDevice(&prev) != hipSuccess) prev = -1;
    if (prev != device) hipSetDevice(device);
  }
  ~DeviceScope() {
    int now = -1;
    if (prev >= 0 && hipGetDevice(&now) == hipSuccess && now != prev) hipSetDevice(prev);
  }
};

// The targets this build has code for, from build-amd.sh ("gfx908,gfx90a,...").
#ifndef PEARL_AMD_TARGETS
#define PEARL_AMD_TARGETS "the targets build-amd.sh was given"
#endif

// Whether this build has code for an architecture ("gfx1100" or "gfx942:sramecc+:xnack-").
bool in_build(const char *arch) {
  char base[32] = {0};
  size_t i = 0;
  for (; arch[i] && arch[i] != ':' && i + 1 < sizeof base; i++) base[i] = arch[i];
  if (!i) return false;
  const char *list = PEARL_AMD_TARGETS;
  for (const char *p = list; (p = strstr(p, base)) != nullptr; p += i) {
    const bool start = p == list || p[-1] == ',';
    const bool end = p[i] == ',' || p[i] == '\0';
    if (start && end) return true;
  }
  return false;
}

// The card count, or why there isn't one. After a GPU fault ROCm refuses every later call in
// the process, hipGetDeviceCount included, and on Linux by default it never recovers. That
// must not read as a missing driver.
bool count_devices(int *devices, char *err, size_t err_len) {
  const hipError_t e = hipGetDeviceCount(devices);
  (void)hipGetLastError();
  if (e == hipSuccess && *devices > 0) return true;
  if (e == hipSuccess || e == hipErrorNoDevice) {
#ifdef _WIN32
    fail(err, err_len, "no AMD GPU found: is AMD's graphics driver installed?");
#else
    fail(err, err_len, "no AMD GPU found: is the ROCm driver (amdgpu) installed, and can this user open /dev/kfd?");
#endif
  } else if (err && err_len) {
    snprintf(err, err_len,
             "HIP stopped working in this process (%s): after a GPU error ROCm refuses every later call; restart "
             "the miner", hipGetErrorString(e));
  }
  return false;
}

// What the code object loaded for the current card says of itself (pearl_amd_caps): its folds
// and flags in caps[0], its wave size in caps[1]. dCaps is two words of device memory. It waits
// for the launch before the copy: HIP-CPU can run a copy beside earlier work on the same stream.
hipError_t read_caps(hipStream_t stream, uint32_t *dCaps, uint32_t caps[2]) {
  hipLaunchKernelGGL(pearl_amd_caps, dim3(1), dim3(1), 0, stream, dCaps);
  hipError_t e = hipGetLastError();
  if (e == hipSuccess) e = hipStreamSynchronize(stream);
  if (e == hipSuccess) e = hipMemcpy(caps, dCaps, 2 * sizeof(uint32_t), hipMemcpyDeviceToHost);
  return e;
}

// Which fold this card runs, decided once a context, before the first operand draw.
//
// The card's architecture picks the fold (native_fold); PEARL_AMD_FOLD overrides it. Then
// the code object that actually loaded says which folds it has bodies for: a build that
// left this card's architecture out has no code object for it at all, and the caps launch
// itself fails. Either way the reason is kept in foldErr and the first search reports it.
void resolve_fold(Ctx *ctx) {
  char *err = ctx->foldErr;
  const size_t err_len = sizeof ctx->foldErr;
  err[0] = 0;
  uint32_t caps[2] = {0, 0};
  const hipError_t e = read_caps(ctx->work, ctx->dCaps, caps);
  if (e != hipSuccess) {
    (void)hipGetLastError();
    // HIP loads the code objects for every card in its order and stops at the first card the
    // build has no code for, so every card after that one gets none either. Name that card.
    int other = -1;
    char otherArch[64] = {0};
#if !defined(__HIP_CPU_RT__)
    int n = 0;
    if (hipGetDeviceCount(&n) == hipSuccess) {
      for (int d = 0; d < n && other < 0; d++) {
        hipDeviceProp_t prop;
        if (hipGetDeviceProperties(&prop, d) != hipSuccess || in_build(prop.gcnArchName)) continue;
        other = d;
        snprintf(otherArch, sizeof otherArch, "%s", prop.gcnArchName);
      }
    }
    (void)hipGetLastError();
#endif
    if (other >= 0 && other != ctx->device)
      snprintf(err, err_len,
               "HIP device %d (%s) has no code in this build, and HIP then loads none for the cards after it: hide it "
               "with HIP_VISIBLE_DEVICES, listing only the cards to mine on (%s)",
               other, otherArch, hipGetErrorString(e));
    else
      snprintf(err, err_len, "this core has no code for %s (%s): this build is for %s",
               ctx->arch[0] ? ctx->arch : "this card", hipGetErrorString(e), PEARL_AMD_TARGETS);
    return;
  }
  ctx->emulated = (caps[0] & PEARL_AMD_CAP_EMULATED) != 0;
  ctx->readoutDrop = (caps[0] & PEARL_AMD_CAP_DROP) != 0;
  ctx->timing = (caps[0] & PEARL_AMD_CAP_TIMING) != 0;
  ctx->kwrap = (caps[0] & PEARL_AMD_CAP_KWRAP) != 0;
  ctx->rank128 = (caps[0] & PEARL_AMD_CAP_RANK) != 0;
  ctx->fuse = (caps[0] & PEARL_AMD_CAP_FUSE) != 0;
  ctx->stripCap = (caps[0] & PEARL_AMD_CAP_STRIP) != 0;
#if PEARL_AMD_STAMP
  ctx->stamp = (caps[0] & PEARL_AMD_CAP_STAMP) != 0;
#endif
  // Past 16 chunks only the wrap readout is right. Said here, at create, rather than at the first
  // batch: a k 4096 asked of an RDNA 3 card (PEARL_PROFILE_K) fails at once with the reason.
  const uint32_t rank = ctx->profile.rank;
  if (!ctx->kwrap && rank && ctx->profile.k / rank > PEARL_JACKPOT_BUCKETS) {
    snprintf(err, err_len, "k %u needs the wrap readout, and this build's code for %s has none (a release has it "
             "only for RDNA 4)", ctx->profile.k, ctx->arch[0] ? ctx->arch : "this card");
    return;
  }
  int want = ctx->emulated ? fold_by_name("mfma16") : native_fold(ctx->arch);
  const char *forced = getenv("PEARL_AMD_FOLD");
  if (forced && forced[0]) {
    want = fold_by_name(forced);
    if (want < 0) {
      snprintf(err, err_len, "PEARL_AMD_FOLD=%s is not a fold: use ref, mfma16, mfma8, wmma11, wmma12 or dot4",
               forced);
      return;
    }
  }
  if (!(caps[0] & kFolds[want].cap)) {
    snprintf(err, err_len, "the %s fold has no code for %s in this build (folds present: 0x%x)",
             kFolds[want].name, ctx->arch[0] ? ctx->arch : "this card", caps[0]);
    return;
  }
  // The folds on CDNA (gfx9: mfma8, mfma16) have run only through the CPU emulation, and the
  // one on RDNA 2 (gfx103x: dot4) on RDNA 3 cards and the CPU, never on an RDNA 2 card. If a
  // card's layout differs from the one the fold was written for, every hash is wrong yet meets
  // the target, so every share is sent and rejected, and a pool can ban the address for it. A
  // card whose architecture could not be read (hipGetDeviceProperties failed) gets the reference
  // fold and could be one of these, so it counts as one. Such a card mines with every hit
  // checked: its opener recomputes each hit in JS before sending it (checkHits, pearlMiner.js),
  // and ctx->untested tells it to. PEARL_AMD_UNTESTED=1 does not change that. It only opens the
  // card for an opener that checks nothing, which is for hashrate.js: it sends no shares. In
  // v0.6.8 and v0.6.9 it let such a card mine with no check, and the error there told RX 6000
  // owners to set it for the reference fold those cards ran then. They now run dot4, so a setting
  // carried over must not vouch for it. A kind of card leaves this list here, once one has passed
  // the checks in README.md ("Testing on real hardware"). Any other opener is refused.
  const char *unchecked = getenv("PEARL_AMD_UNTESTED");
  if (!ctx->emulated && (!ctx->arch[0] || !strncmp(ctx->arch, "gfx9", 4) || !strncmp(ctx->arch, "gfx103", 6))) {
    if (ctx->checkHits) {
      ctx->untested = true;
    } else if (!(unchecked && !strcmp(unchecked, "1"))) {
      snprintf(err, err_len,
               "the %s fold has not run on %s%s card yet, so every share could be rejected: it mines only with "
               "every hit checked, as the app and verify-hits.js do. For an opener that sends no shares, set "
               "PEARL_AMD_UNTESTED=1 (earn/native/amd/README.md, \"Testing on real hardware\")",
               kFolds[want].name, ctx->arch[0] ? "a " : "", ctx->arch[0] ? ctx->arch : "this unidentified");
      return;
    }
  }
  ctx->fold = want;
}

}  // namespace

// Choose the card this core mines on, and make it the calling thread's device. The same
// rule as pearl_host.cu: an explicit index wins; otherwise the card with the most compute
// units times clock, among those whose memory can hold the profile, ties to the lower
// index. HIP numbers cards the way rocm-smi does unless HIP_VISIBLE_DEVICES says otherwise.
extern "C" int pearl_host_select_device(const PearlProfile *profile, int requested, char *name,
                                        size_t name_len, char *err, size_t err_len) {
  if (name && name_len) name[0] = '\0';
  if (!profile) { fail(err, err_len, "no profile supplied"); return -1; }
  int devices = 0;
  if (!count_devices(&devices, err, err_len)) return -1;
  int chosen = -1;
  if (requested >= 0) {
    if (requested >= devices) {
      if (err && err_len)
        snprintf(err, err_len,
                 "GPU %d was asked for (PEARL_GPU_INDEX) but this machine has %d: valid indices are 0..%d",
                 requested, devices, devices - 1);
      return -1;
    }
    chosen = requested;
  } else {
    double best = -1.0;
    const size_t need = needed_bytes(profile);
    for (int d = 0; d < devices; d++) {
      hipDeviceProp_t prop;
      if (hipGetDeviceProperties(&prop, d) != hipSuccess) continue;
      if (prop.totalGlobalMem < need) continue;
      const double clock = prop.clockRate > 0 ? (double)prop.clockRate : 1.0;
      const double score = (double)prop.multiProcessorCount * clock;
      if (score > best) { best = score; chosen = d; }
    }
    if (chosen < 0) chosen = 0;  // let create's pre-flight say why, with the numbers
  }
  const hipError_t e = hipSetDevice(chosen);
  if (e != hipSuccess) {
    if (err && err_len) snprintf(err, err_len, "could not open GPU %d: %s", chosen, hipGetErrorString(e));
    return -1;
  }
  hipDeviceProp_t prop;
  if (name && name_len && hipGetDeviceProperties(&prop, chosen) == hipSuccess)
    snprintf(name, name_len, "%s", prop.name);
  return chosen;
}

// Whether k 8192 is a card's default. An RX 9070 XT (gfx1201, 64 CUs) read +0.5 to +0.9% over
// k 4096 in hashrate.js, and +0.7% again with HeroMiners taking 13 of 13 k 8192 shares; an RX
// 9060 XT (gfx1200, 32 CUs) read about 6% slower (4 November, with the combined fold flags in
// pearl_amd_fold.h; earn/tuning/benchmark.md). So: gfx1201 with at least 56 CUs, the full Navi 48 cards (RX
// 9070 XT and R9700, 64 CUs) and the RX 9070 (56). The RX 9070 GRE (gfx1201, 48 CUs) stays at
// k 4096, since nothing between 32 and 64 CUs has been timed. `arch` is gcnArchName, which may
// carry feature flags after a colon ("gfx1201:xnack-"). `units` is multiProcessorCount, which
// on RDNA counts WGPs, two CUs each: an RX 9070 XT reads 32 under ROCm 7.2, where rocminfo says
// 64 CUs (a box on 4 November; a gate at 56 kept it at k 4096). Either count is taken, so a
// runtime that reports CUs picks the same cards: 28-39 is 56-78 CUs counted as WGPs, 56 and up
// is CUs. The GRE (24 WGPs, 48 CUs) is in neither.
static bool k8192_by_default(const char *arch, int units) {
  const bool big = (units >= 28 && units < 40) || units >= 56;
  return !strncmp(arch, "gfx1201", 7) && (arch[7] == '\0' || arch[7] == ':') && big;
}

// The k a card mines when the miner lets the core choose (pearl_core.cc's cardK): where the
// card's code object has the wrap readout (RDNA 4, PEARL_AMD_CAP_KWRAP), k 8192 where
// k8192_by_default says so and k 4096 on the rest; the profile's k everywhere else. A share says
// its own k and a MAC earns the same at rank 128, so the pool needs nothing from us. Only the
// mainnet k moves: 2048 at rank 128. PEARL_AMD_K picks the k on any such card: 2048, 4096 or 8192;
// any other value is ignored, and the note says so. Without the free VRAM the chosen k needs, the
// card tries k 4096 (from 8192), then stays at k 2048, so it still mines; `note` says why, for
// the log. A HIP error leaves the profile's k too, and pearl_host_create reports the error.
// An ablation build (PEARL_AMD_CAP_TIMING) gets the release's k, unlike a readout-drop build, so it
// times the fold the card mines; pearl_host_submit keeps it from mining.
extern "C" uint32_t pearl_host_card_k(const PearlProfile *profile, int device, char *note, size_t note_len) {
  if (note && note_len) note[0] = '\0';
  if (!profile) return 0;
  const uint32_t k = profile->k;
  if (k != 2048u || profile->rank != 128u) return k;
  DeviceScope scope(device);
  uint32_t *dCaps = nullptr;
  uint32_t caps[2] = {0, 0};
  hipError_t e = hipMalloc(&dCaps, 2 * sizeof(uint32_t));
  if (e == hipSuccess) e = read_caps(nullptr, dCaps, caps);
  if (dCaps) hipFree(dCaps);
  (void)hipGetLastError();
  if (e != hipSuccess || !(caps[0] & PEARL_AMD_CAP_KWRAP) || (caps[0] & PEARL_AMD_CAP_DROP)) return k;
  const char *pin = getenv("PEARL_AMD_K");
  if (pin && !strcmp(pin, "2048")) {
    if (note && note_len) snprintf(note, note_len, "k 2048 (PEARL_AMD_K=2048)");
    return k;
  }
  hipDeviceProp_t prop;
  const bool big = hipGetDeviceProperties(&prop, device) == hipSuccess
                   && k8192_by_default(prop.gcnArchName, prop.multiProcessorCount);
  (void)hipGetLastError();
  const bool pin8 = pin && !strcmp(pin, "8192"), pin4 = pin && !strcmp(pin, "4096");
  const uint32_t want = pin8 ? 8192u : pin4 ? 4096u : big ? 8192u : 4096u;
  // A pin names itself; an unknown value is ignored, out loud. A VRAM step-down below overwrites it.
  if (note && note_len) {
    if (pin8 || pin4) snprintf(note, note_len, "k %u (PEARL_AMD_K=%s)", want, pin);
    else if (pin && *pin) snprintf(note, note_len, "PEARL_AMD_K=%.16s ignored: not 2048, 4096 or 8192", pin);
  }
  PearlProfile wide = *profile;
  wide.k = want;
  const size_t need = needed_bytes(&wide);
  size_t freeMem = 0, totalMem = 0;
  const bool known = hipMemGetInfo(&freeMem, &totalMem) == hipSuccess;
  (void)hipGetLastError();
  if (known && freeMem < need) {
    // Asked for k 8192 without the VRAM for it: k 4096 still beats the mainnet k, so try it first.
    wide.k = 4096u;
    if (want == 8192u && freeMem >= needed_bytes(&wide)) {
      if (note && note_len)
        snprintf(note, note_len, "k 4096: k 8192 needs ~%zu MiB and %zu MiB is free", need >> 20, freeMem >> 20);
      return 4096u;
    }
    if (note && note_len)
      snprintf(note, note_len, "k 2048: k %u needs ~%zu MiB and %zu MiB is free", want, need >> 20, freeMem >> 20);
    return k;
  }
  return want;
}

extern "C" void pearl_host_destroy(void *handle);

#define HIP_OK(expr, msg)                                                         \
  do {                                                                            \
    hipError_t _e = (expr);                                                       \
    if (_e != hipSuccess) {                                                       \
      if (err && err_len) snprintf(err, err_len, "%s: %s", msg, hipGetErrorString(_e)); \
      pearl_host_destroy(ctx);                                                    \
      (void)hipGetLastError();                                                    \
      return nullptr;                                                             \
    }                                                                             \
  } while (0)

// pearl_host_create for an opener that recomputes every hit before it sends it (check_hits: the
// app's PearlMiner, with shared/miner/checkHit.js). Such an opener gets a card whose fold has not
// run on its kind of card (resolve_fold), and *untested says when it got one, so that it checks
// that card's hits, whatever PEARL_AMD_UNTESTED says. pearl_host_create is this with neither.
extern "C" void *pearl_host_create_checked(const PearlProfile *profile, bool check_hits, bool *untested,
                                           char *err, size_t err_len) {
  if (untested) *untested = false;
  if (!profile) { fail(err, err_len, "no profile supplied"); return nullptr; }
  int devices = 0;
  if (!count_devices(&devices, err, err_len)) return nullptr;
  {
    const uint64_t aChunks = (uint64_t)profile->m * profile->k / 1024u;
    const uint64_t bChunks = (uint64_t)profile->n * profile->k / 1024u;
    const bool aOk = aChunks && (aChunks & (aChunks - 1)) == 0;
    const bool bOk = bChunks && (bChunks & (bChunks - 1)) == 0;
    if (!aOk || !bOk) {
      if (err && err_len)
        snprintf(err, err_len,
                 "m*k/1024 and n*k/1024 must each be a power of two (got %llu and %llu): the "
                 "commitment tree fold assumes it",
                 (unsigned long long)aChunks, (unsigned long long)bChunks);
      return nullptr;
    }
  }
  const size_t k = profile->k;
  const size_t rank = profile->rank;
  const size_t aBytes = (size_t)profile->m * k;
  const size_t bBytes = (size_t)profile->n * k;
  const size_t need = needed_bytes(profile);
  size_t freeMem = 0, totalMem = 0;
  if (hipMemGetInfo(&freeMem, &totalMem) == hipSuccess && freeMem < need) {
    if (err && err_len)
      snprintf(err, err_len,
               "not enough free VRAM for the rank-%u profile: need ~%zu MiB, %zu MiB free of %zu MiB",
               (unsigned)profile->rank, need >> 20, freeMem >> 20, totalMem >> 20);
    return nullptr;
  }

  Ctx *ctx = new Ctx();
  ctx->profile = *profile;
  ctx->checkHits = check_hits;
  if (hipGetDevice(&ctx->device) != hipSuccess) ctx->device = 0;
#if !defined(__HIP_CPU_RT__)
  {
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, ctx->device) == hipSuccess)
      snprintf(ctx->arch, sizeof ctx->arch, "%s", prop.gcnArchName);
  }
#endif

  ctx->compact = pearl_compact_operands(profile);
  HIP_OK(hipMalloc(&ctx->dA, ctx->compact ? 2048 : aBytes), "allocating A");
  HIP_OK(hipMalloc(&ctx->dB, ctx->compact ? 2048 : bBytes), "allocating B");
  HIP_OK(hipMalloc(&ctx->dAp, aBytes), "allocating the noised A");
  HIP_OK(hipMalloc(&ctx->dBp, bBytes), "allocating the noised B");
  HIP_OK(hipMalloc(&ctx->dEAL, (size_t)profile->m * rank), "allocating E_AL");
  HIP_OK(hipMalloc(&ctx->dEBR, (size_t)profile->n * rank), "allocating E_BR");
  HIP_OK(hipMalloc(&ctx->dPermA, k * 2 * sizeof(uint32_t)), "allocating E_AR");
  HIP_OK(hipMalloc(&ctx->dPermB, k * 2 * sizeof(uint32_t)), "allocating E_BL");
  HIP_OK(hipMalloc(&ctx->dLabelA, 32), "allocating the A label");
  HIP_OK(hipMalloc(&ctx->dLabelB, 32), "allocating the B label");
  {
    uint8_t lab[32];
    memset(lab, 0, 32);
    memcpy(lab, "A_tensor", 8);
    hipMemcpy(ctx->dLabelA, lab, 32, hipMemcpyHostToDevice);
    memset(lab, 0, 32);
    memcpy(lab, "B_tensor", 8);
    hipMemcpy(ctx->dLabelB, lab, 32, hipMemcpyHostToDevice);
  }
  HIP_OK(hipMalloc(&ctx->dSaltA, 32), "allocating the A salt");
  HIP_OK(hipMalloc(&ctx->dSaltB, 32), "allocating the B salt");
  HIP_OK(hipMalloc(&ctx->dBoundA, PEARL_HASH_BYTES), "allocating the bound A root");
  HIP_OK(hipMalloc(&ctx->dBoundB, PEARL_HASH_BYTES), "allocating the bound B root");
  hipMemcpy(ctx->dSaltA, PEARL_SEED_SALT_A, 32, hipMemcpyHostToDevice);
  hipMemcpy(ctx->dSaltB, PEARL_SEED_SALT_B, 32, hipMemcpyHostToDevice);

  ctx->rowsValid = profile->m / PEARL_ROWS_COUNT;
  ctx->colsValid = profile->n / PEARL_COLS_COUNT;
  ctx->colBatch = amd_col_batch(profile, ctx->arch);
  ctx->stripBatch = ctx->colBatch;
  ctx->batch = ctx->colBatch * ctx->rowsValid;
  ctx->hHitIndex.resize(PEARL_MAX_HITS);
  const size_t S = Ctx::kSlots;
  HIP_OK(hipMalloc(&ctx->dHitTranscript, S * PEARL_MAX_HITS * PEARL_JACKPOT_BUCKETS * sizeof(uint32_t)),
         "allocating the hit transcripts");
  HIP_OK(hipMalloc(&ctx->dHashes, S * PEARL_MAX_HITS * PEARL_HASH_BYTES), "allocating the batch hashes");
  HIP_OK(hipMalloc(&ctx->dHitCount, S * sizeof(uint32_t)), "allocating the hit counter");
  HIP_OK(hipMalloc(&ctx->dHitIndex, S * PEARL_MAX_HITS * sizeof(uint32_t)), "allocating the hit list");
  HIP_OK(hipMalloc(&ctx->dCaps, 2 * sizeof(uint32_t)), "allocating the fold check");
  HIP_OK(hipHostMalloc(&ctx->hSlotCount, S * sizeof(uint32_t), hipHostMallocDefault),
         "allocating the pinned hit counts");
  for (int i = 0; i < Ctx::kSlots; i++) {
    HIP_OK(hipEventCreateWithFlags(&ctx->slotDone[i], hipEventDisableTiming), "creating the batch events");
  }
  HIP_OK(hipStreamCreateWithFlags(&ctx->side, hipStreamNonBlocking), "creating the proof stream");
  HIP_OK(hipStreamCreateWithFlags(&ctx->work, hipStreamDefault), "creating the search stream");
#if !defined(__HIP_CPU_RT__)   // a GPU only: HIP-CPU's streams (Ctx::work) stay as they are
  {
    bool q = PEARL_AMD_HASHQ == 1 || (PEARL_AMD_HASHQ == 2 && !strncmp(ctx->arch, "gfx1100", 7));
    const char *hq = getenv("PEARL_AMD_HASHQ");
    if (hq && (!strcmp(hq, "0") || !strcmp(hq, "1"))) q = hq[0] == '1';
    if (q) {
      HIP_OK(hipStreamCreateWithFlags(&ctx->hashq, hipStreamDefault), "creating the hash stream");
      for (int i = 0; i < Ctx::kSlots; i++)
        HIP_OK(hipEventCreateWithFlags(&ctx->foldDone[i], hipEventDisableTiming), "creating the fold events");
      ctx->hashqZero = PEARL_AMD_HASHQ_ZERO != 0;
      ctx->hashqWait = PEARL_AMD_HASHQ_WAIT != 0;
      const char *hz = getenv("PEARL_AMD_HASHQ_ZERO");
      if (hz && (!strcmp(hz, "0") || !strcmp(hz, "1"))) ctx->hashqZero = hz[0] == '1';
      const char *hw = getenv("PEARL_AMD_HASHQ_WAIT");
      if (hw && (!strcmp(hw, "0") || !strcmp(hw, "1"))) ctx->hashqWait = hw[0] == '1';
    }
  }
#endif
  {
    const char *tr = getenv("PEARL_AMD_TRACE");
    if (tr && tr[0] && strcmp(tr, "0") != 0) {
      ctx->trace.on = true;
      for (int i = 0; i < Ctx::kSlots; i++) {
        HIP_OK(hipEventCreate(&ctx->trace.f0[i]), "creating the trace events");
        HIP_OK(hipEventCreate(&ctx->trace.f1[i]), "creating the trace events");
        HIP_OK(hipEventCreate(&ctx->trace.h1[i]), "creating the trace events");
      }
    }
  }
  HIP_OK(hipMalloc(&ctx->dJobKey, 8 * sizeof(uint32_t)), "allocating job_key");
  HIP_OK(hipMalloc(&ctx->dASeed, 8 * sizeof(uint32_t)), "allocating a_seed");
  HIP_OK(hipMalloc(&ctx->dBSeed, 8 * sizeof(uint32_t)), "allocating b_seed");
  ctx->cvCapacity = (aBytes > bBytes ? aBytes : bBytes) / 1024;
  if (ctx->cvCapacity < 1) ctx->cvCapacity = 1;
  HIP_OK(hipMalloc(&ctx->dCvs, ctx->cvCapacity * 8 * sizeof(uint32_t)), "allocating the BLAKE3 tree scratch");
  {
    const uint64_t aLeaves = (uint64_t)profile->m * profile->k / 1024;
    const uint64_t bLeaves = (uint64_t)profile->n * profile->k / 1024;
    HIP_OK(hipMalloc(&ctx->dTreeA, 2 * aLeaves * 8 * sizeof(uint32_t)), "allocating the A commitment tree");
    HIP_OK(hipMalloc(&ctx->dTreeB, 2 * bLeaves * 8 * sizeof(uint32_t)), "allocating the B commitment tree");
  }
  HIP_OK(hipMalloc(&ctx->dSeedBuf, 64), "allocating the seed buffer");
  HIP_OK(hipMalloc(&ctx->dSeedInput, PEARL_HEADER_BYTES + PEARL_CONFIG_BYTES), "allocating the job_key input");
  HIP_OK(hipMalloc(&ctx->dHashA, PEARL_HASH_BYTES), "allocating hash_a");
  HIP_OK(hipMalloc(&ctx->dHashB, PEARL_HASH_BYTES), "allocating hash_b");

  // A card with no fold it may run fails here, once, with the reason, rather than after a
  // full operand draw at the first search, and again at each restart.
  resolve_fold(ctx);
  if (ctx->fold < 0) {
    fail(err, err_len, ctx->foldErr);
    pearl_host_destroy(ctx);
    return nullptr;
  }
  // The launch's width, now that the card's code object has said whether it takes strips, and then the transcript
  // buffers, which are as wide.
  ctx->strips = amd_strips(ctx);
  ctx->colBatch = ctx->stripBatch * ctx->strips;
  ctx->batch = ctx->colBatch * ctx->rowsValid;
  for (int i = 0; i < Ctx::kSlots; i++)
    HIP_OK(hipMalloc(&ctx->dTr[i], (size_t)ctx->batch * PEARL_JACKPOT_BUCKETS * sizeof(uint32_t)),
           "allocating the transcript buffers");
#if PEARL_AMD_STAMP
  // A PEARL_AMD_STAMP build's folds store clock stamps, each wave of a block its own. They get room for a
  // whole batch, so a hashrate.js run times the same stores as the fold test; nothing here reads them.
  if (ctx->stamp) {
    const FoldInfo &f = kFolds[ctx->fold];
    const size_t blocks = (size_t)(ctx->colBatch * PEARL_COLS_COUNT / f.bn) * (profile->m / f.bm);
    HIP_OK(hipMalloc(&ctx->dStamps, blocks * (PEARL_AMD_THREADS / 32u) * PEARL_AMD_STAMP_WORDS * sizeof(uint32_t)),
           "allocating the clock stamps");
    HIP_OK(hipMemcpyToSymbol(HIP_SYMBOL(pearl_amd_stamps), &ctx->dStamps, sizeof ctx->dStamps),
           "pointing the folds at the clock stamps");
    ctx->stampBlocks = blocks;
    const char *at = getenv("PEARL_AMD_STAMP_DUMP_AT");
    for (const char *q = at; q && *q >= '0' && *q <= '9' && ctx->stampAt < 100000000u; q++)
      ctx->stampAt = ctx->stampAt * 10u + (uint64_t)(*q - '0');
  }
#endif
#if !defined(__HIP_CPU_RT__)   // a GPU only
  {
    uint32_t us = PEARL_AMD_TICK_US;
    const char *env = getenv("PEARL_AMD_TICK_US");
    if (env && env[0] >= '0' && env[0] <= '9') {
      us = 0;
      for (const char *p = env; *p >= '0' && *p <= '9' && us <= 100000u; p++) us = us * 10u + (uint32_t)(*p - '0');
    }
    if (us >= 100u && us <= 100000u) {
      HIP_OK(hipStreamCreateWithFlags(&ctx->tickq, hipStreamDefault), "creating the ticker's stream");
      HIP_OK(hipMalloc(&ctx->dTick, 2 * sizeof(uint32_t)), "allocating the ticker's words");
      ctx->tickUs = us;
      Ctx *c = ctx;
      ctx->ticker = std::thread([c] {
        hipSetDevice(c->device);
        while (!c->tickStop.load(std::memory_order_relaxed)) {
          hipLaunchKernelGGL(pearl_amd_caps, dim3(1), dim3(1), 0, c->tickq, c->dTick);
          (void)hipStreamSynchronize(c->tickq);
          (void)hipGetLastError();
          std::this_thread::sleep_for(std::chrono::microseconds(c->tickUs));
        }
      });
    }
  }
#endif
  if (untested) *untested = ctx->untested;
  return ctx;
}

// For an opener that checks no hits: a card whose fold has not run on its kind of card is
// refused, unless PEARL_AMD_UNTESTED=1 (for an opener that sends no shares, resolve_fold).
extern "C" void *pearl_host_create(const PearlProfile *profile, char *err, size_t err_len) {
  return pearl_host_create_checked(profile, false, nullptr, err, err_len);
}

extern "C" void pearl_host_bind_thread(void *handle) {
  Ctx *ctx = (Ctx *)handle;
  if (!ctx) return;
  hipSetDevice(ctx->device);
}

extern "C" void pearl_host_destroy(void *handle) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx) return;
  // The ticker first: it launches on this context's device until it is told to stop.
  ctx->tickStop.store(true);
  if (ctx->ticker.joinable()) ctx->ticker.join();
  DeviceScope scope(ctx->device);
  hipDeviceSynchronize();
  if (ctx->tickq) hipStreamDestroy(ctx->tickq);
  hipFree(ctx->dTick);
  hipFree(ctx->dA); hipFree(ctx->dB);
  hipFree(ctx->dAp); hipFree(ctx->dBp);
  hipFree(ctx->dEAL); hipFree(ctx->dEBR);
  hipFree(ctx->dPermA); hipFree(ctx->dPermB);
  hipFree(ctx->dLabelA); hipFree(ctx->dLabelB);
  hipFree(ctx->dSaltA); hipFree(ctx->dSaltB);
  hipFree(ctx->dBoundA); hipFree(ctx->dBoundB);
  hipFree(ctx->dHashes); hipFree(ctx->dHitCount); hipFree(ctx->dHitIndex); hipFree(ctx->dCaps);
  for (int s = 0; s < Ctx::kSlots; s++) hipFree(ctx->dTr[s]);
#if PEARL_AMD_STAMP
  if (ctx->dStamps) {   // no fold may store to it once it is freed
    uint32_t *none = nullptr;
    hipMemcpyToSymbol(HIP_SYMBOL(pearl_amd_stamps), &none, sizeof none);
    hipFree(ctx->dStamps);
  }
#endif
  hipFree(ctx->dTreeA); hipFree(ctx->dTreeB);
  hipFree(ctx->dCvs); hipFree(ctx->dSeedBuf); hipFree(ctx->dSeedInput);
  hipFree(ctx->dHashA); hipFree(ctx->dHashB);
  hipFree(ctx->dHitTranscript); hipFree(ctx->dJobKey);
  hipFree(ctx->dASeed); hipFree(ctx->dBSeed);
  for (int i = 0; i < Ctx::kSlots; i++) {
    if (ctx->slotDone[i]) hipEventDestroy(ctx->slotDone[i]);
    if (ctx->trace.f0[i]) hipEventDestroy(ctx->trace.f0[i]);
    if (ctx->trace.f1[i]) hipEventDestroy(ctx->trace.f1[i]);
    if (ctx->trace.h1[i]) hipEventDestroy(ctx->trace.h1[i]);
    if (ctx->foldDone[i]) hipEventDestroy(ctx->foldDone[i]);
  }
  if (ctx->hashq) hipStreamDestroy(ctx->hashq);
  if (ctx->side) hipStreamDestroy(ctx->side);
  if (ctx->work) hipStreamDestroy(ctx->work);
  if (ctx->hSlotCount) hipHostFree(ctx->hSlotCount);
  if (ctx->hStage) hipHostFree(ctx->hStage);
  (void)hipGetLastError();
  delete ctx;
}

extern "C" void pearl_host_reseed(void *handle, uint64_t salt);

extern "C" void pearl_host_set_job_salted(void *handle, const uint8_t *header,
                                          const uint8_t *target, uint64_t salt) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx) return;
  DeviceScope scope(ctx->device);
  const double t0 = ctx->trace.on ? now_ms() : 0.0;
  memcpy(ctx->header, header, PEARL_HEADER_BYTES);
  memcpy(ctx->target, target, PEARL_HASH_BYTES);
  ctx->baseDrawn = false;
  pearl_host_reseed(handle, salt);
  if (ctx->trace.on) {
    ctx->trace.jobs++;
    ctx->trace.jobMs += now_ms() - t0;
  }
}

extern "C" void pearl_host_set_job(void *handle, const uint8_t *header, const uint8_t *target) {
  pearl_host_set_job_salted(handle, header, target, 0);
}

namespace {

const uint32_t kDrawThreads = 256;
unsigned draw_blocks(size_t n) { return (unsigned)((n + kDrawThreads - 1) / kDrawThreads); }

// b_seed = blake3(job_key || bound_b), then a_seed = blake3(b_seed || bound_a), the roots
// bound to their dimensions first under cert-v3 (see pearl_host.cu).
void derive_seeds(Ctx *ctx, bool withB) {
  const bool legacy = ctx->profile.seed_derivation == PEARL_SEED_LEGACY;
  if (legacy) {
    hipMemcpy(ctx->dBoundA, ctx->dHashA, PEARL_HASH_BYTES, hipMemcpyDeviceToDevice);
    if (withB) hipMemcpy(ctx->dBoundB, ctx->dHashB, PEARL_HASH_BYTES, hipMemcpyDeviceToDevice);
  } else {
    hipLaunchKernelGGL(pearl_bind_root, dim3(1), dim3(1), 0, ctx->work, (const uint8_t *)ctx->dSaltA,
                       (const uint8_t *)ctx->dHashA, ctx->profile.m, ctx->dBoundA);
    if (withB) {
      hipLaunchKernelGGL(pearl_bind_root, dim3(1), dim3(1), 0, ctx->work, (const uint8_t *)ctx->dSaltB,
                         (const uint8_t *)ctx->dHashB, ctx->profile.n, ctx->dBoundB);
    }
  }
  if (withB) {
    hipMemcpy(ctx->dSeedBuf, ctx->dJobKey, PEARL_HASH_BYTES, hipMemcpyDeviceToDevice);
    hipMemcpy(ctx->dSeedBuf + PEARL_HASH_BYTES, ctx->dBoundB, PEARL_HASH_BYTES, hipMemcpyDeviceToDevice);
    hipLaunchKernelGGL(pearl_blake3_unkeyed, dim3(1), dim3(1), 0, ctx->work, (const uint8_t *)ctx->dSeedBuf,
                       64u, reinterpret_cast<uint8_t *>(ctx->dBSeed));
  }
  hipMemcpy(ctx->dSeedBuf, ctx->dBSeed, PEARL_HASH_BYTES, hipMemcpyDeviceToDevice);
  hipMemcpy(ctx->dSeedBuf + PEARL_HASH_BYTES, ctx->dBoundA, PEARL_HASH_BYTES, hipMemcpyDeviceToDevice);
  hipLaunchKernelGGL(pearl_blake3_unkeyed, dim3(1), dim3(1), 0, ctx->work, (const uint8_t *)ctx->dSeedBuf,
                     64u, reinterpret_cast<uint8_t *>(ctx->dASeed));
}

// One side's noise, and the noised operand: the dense factor, the sparse selectors, then
// pearl_materialize16_tiled for the matrix folds, pearl_materialize16 (row-major) for the
// reference fold, or pearl_materialize when k is not a power of two.
void draw_noise(Ctx *ctx, bool isA) {
  const uint32_t rank = ctx->profile.rank;
  const uint32_t k = ctx->profile.k;
  const uint32_t rows = isA ? ctx->profile.m : ctx->profile.n;
  const uint32_t *seed = isA ? ctx->dASeed : ctx->dBSeed;
  const uint8_t *label = isA ? ctx->dLabelA : ctx->dLabelB;
  int8_t *dense = isA ? ctx->dEAL : ctx->dEBR;
  uint32_t *perm = isA ? ctx->dPermA : ctx->dPermB;
  const int8_t *src = isA ? ctx->dA : ctx->dB;
  int8_t *dst = isA ? ctx->dAp : ctx->dBp;
  const size_t len = (size_t)rows * k;
  const bool kPow2 = (k & (k - 1u)) == 0u && k >= 16u;
  uint32_t kLog2 = 0;
  while ((1u << kLog2) < k) kLog2++;
  const bool constFill = ctx->profile.operand_fill == PEARL_OPERAND_CONST;
  const uint64_t readVecs = constFill ? (PEARL_STAMP_BYTES + 15u) / 16u : (uint64_t)(len / 16);
  const uint32_t fillWord = (uint32_t)(uint8_t)PEARL_OPERAND_FILL * 0x01010101u;
  hipLaunchKernelGGL(pearl_gen_dense, dim3(draw_blocks((size_t)rows * (rank / 32))), dim3(kDrawThreads),
                     0, ctx->work, seed, label, (const uint32_t *)nullptr, dense, rows, rank);
  hipLaunchKernelGGL(pearl_gen_perm, dim3(draw_blocks((k + 7) / 8)), dim3(kDrawThreads), 0, ctx->work, seed,
                     label, perm, k, rank);
  // The matrix folds read the noised operands in per-tile order under PEARL_AMD_TILED (see
  // pearl_amd_fold.h): tiles of the fold's rows (A') or columns (B'), each k / PEARL_AMD_SLAB
  // slabs of [tile rows][slab]. The reference fold reads them row-major.
  // Only for whole tiles: a profile whose rows aren't (m = 64, say) would be written past the
  // end of the buffer. pearl_host_submit refuses such a profile for the matrix folds anyway,
  // but the first draw runs before that, in set_job.
  const FoldInfo *tf = PEARL_AMD_TILED && ctx->fold >= 0 && kFolds[ctx->fold].cap != PEARL_AMD_CAP_REF &&
                               rows % (isA ? kFolds[ctx->fold].bm : kFolds[ctx->fold].bn) == 0u
                           ? &kFolds[ctx->fold] : nullptr;
  if (kPow2 && tf) {
    uint32_t kbLog2 = 0;
    while ((1u << kbLog2) < PEARL_AMD_SLAB) kbLog2++;
    hipLaunchKernelGGL(pearl_materialize16_tiled, dim3(draw_blocks(len / 16)), dim3(kDrawThreads), 0, ctx->work,
                       src, (const int8_t *)dense, (const uint32_t *)perm, dst, rows, kLog2, rank, kbLog2,
                       isA ? tf->bm : tf->bn, readVecs, fillWord);
  } else if (kPow2) {
    hipLaunchKernelGGL(pearl_materialize16, dim3(draw_blocks(len / 16)), dim3(kDrawThreads), 0, ctx->work, src,
                       (const int8_t *)dense, (const uint32_t *)perm, dst, rows, kLog2, rank, readVecs,
                       fillWord);
  } else {
    hipLaunchKernelGGL(pearl_materialize, dim3(draw_blocks(len)), dim3(kDrawThreads), 0, ctx->work, src,
                       (const int8_t *)dense, (const uint32_t *)perm, dst, rows, k, rank);
  }
}

void full_draw(Ctx *ctx, uint64_t salt) {
  uint8_t seedInput[PEARL_HEADER_BYTES + PEARL_CONFIG_BYTES];
  memcpy(seedInput, ctx->header, PEARL_HEADER_BYTES);
  pearl_write_config52(&ctx->profile, seedInput + PEARL_HEADER_BYTES);
  hipMemcpy(ctx->dSeedInput, seedInput, sizeof(seedInput), hipMemcpyHostToDevice);
  hipLaunchKernelGGL(pearl_blake3_unkeyed, dim3(1), dim3(1), 0, ctx->work, (const uint8_t *)ctx->dSeedInput,
                     (uint32_t)sizeof(seedInput), reinterpret_cast<uint8_t *>(ctx->dJobKey));
  const size_t aLen = (size_t)ctx->profile.m * ctx->profile.k;
  const size_t bLen = (size_t)ctx->profile.n * ctx->profile.k;
  if (ctx->profile.operand_fill == PEARL_OPERAND_CONST) {
    hipMemset(ctx->dA, PEARL_OPERAND_FILL, ctx->compact ? 2048 : aLen);
    hipMemset(ctx->dB, PEARL_OPERAND_FILL, ctx->compact ? 2048 : bLen);
    int8_t stamp[PEARL_STAMP_BYTES];
    for (int i = 0; i < PEARL_STAMP_BYTES; i++) stamp[i] = pearl_stamp_byte(salt, i);
    hipMemcpy(ctx->dA, stamp, sizeof(stamp), hipMemcpyHostToDevice);
  } else {
    hipLaunchKernelGGL(pearl_gen_operand, dim3(draw_blocks(aLen / 32 + 1)), dim3(kDrawThreads), 0, ctx->work,
                       (const uint32_t *)ctx->dJobKey, (const uint8_t *)ctx->dLabelA, ctx->dA,
                       (uint64_t)aLen, salt);
    hipLaunchKernelGGL(pearl_gen_operand, dim3(draw_blocks(bLen / 32 + 1)), dim3(kDrawThreads), 0, ctx->work,
                       (const uint32_t *)ctx->dJobKey, (const uint8_t *)ctx->dLabelB, ctx->dB,
                       (uint64_t)bLen, salt);
  }
  operand_commitment(ctx, reinterpret_cast<const uint8_t *>(ctx->dA), aLen, ctx->dHashA, ctx->dTreeA,
                     &ctx->layerOffA);
  operand_commitment(ctx, reinterpret_cast<const uint8_t *>(ctx->dB), bLen, ctx->dHashB, ctx->dTreeB,
                     &ctx->layerOffB);
  derive_seeds(ctx, true);
  draw_noise(ctx, true);
  draw_noise(ctx, false);
  hipMemcpy(ctx->bSeed, ctx->dBSeed, PEARL_HASH_BYTES, hipMemcpyDeviceToHost);
}

void restamp(Ctx *ctx, uint64_t salt) {
  const uint64_t aChunks = (uint64_t)ctx->profile.m * ctx->profile.k / 1024;
  hipLaunchKernelGGL(pearl_restamp_operand, dim3(1), dim3(1), 0, ctx->work, (const uint32_t *)ctx->dJobKey,
                     ctx->dA, salt, aChunks, ctx->dTreeA, ctx->dHashA);
  derive_seeds(ctx, false);
  draw_noise(ctx, true);
}

void device_record(Ctx *ctx, PearlRestampRecord *r) {
  const uint32_t levels = (uint32_t)ctx->layerOffA.size();
  memset(r, 0, sizeof *r);
  r->levels = levels < PEARL_RESTAMP_MAX_LEVELS ? levels : PEARL_RESTAMP_MAX_LEVELS;
  hipMemcpy(r->head, ctx->dA, sizeof r->head, hipMemcpyDeviceToHost);
  for (uint32_t L = 0; L < r->levels; L++) {
    r->node_off[L] = (uint32_t)ctx->layerOffA[L];
    hipMemcpy(r->path[L], ctx->dTreeA + ctx->layerOffA[L] * 8, 32, hipMemcpyDeviceToHost);
  }
  hipMemcpy(r->bound, ctx->dBoundA, 32, hipMemcpyDeviceToHost);
  hb3_words(ctx->aSeed, r->a_seed, 8);
}

void setup_host_seeds(Ctx *ctx) {
  ctx->hostSeeds = false;
  for (int i = 0; i < Ctx::kRecords; i++) ctx->recValid[i] = false;
  const uint32_t levels = (uint32_t)ctx->layerOffA.size();
  PearlRestampRecord r;
  if (levels >= 2 && levels <= PEARL_RESTAMP_MAX_LEVELS) {
    hipMemcpy(ctx->jobKeyW, ctx->dJobKey, 32, hipMemcpyDeviceToHost);
    hipMemcpy(ctx->leaf0, ctx->dA, sizeof ctx->leaf0, hipMemcpyDeviceToHost);
    for (uint32_t L = 0; L + 1 < levels; L++)
      hipMemcpy(ctx->sib[L], ctx->dTreeA + (ctx->layerOffA[L] + 1) * 8, 32, hipMemcpyDeviceToHost);
    host_record(ctx, ctx->salt, false, &r);
    uint8_t root[32], hroot[32], hseed[32];
    hipMemcpy(root, ctx->dHashA, 32, hipMemcpyDeviceToHost);
    hb3_bytes(r.path[levels - 1], hroot);
    hb3_bytes(r.a_seed, hseed);
    if (memcmp(root, hroot, 32) == 0 && memcmp(hseed, ctx->aSeed, 32) == 0) {
      ctx->hostSeeds = true;
      keep_record(ctx, ctx->salt, r);
      return;
    }
    fprintf(stderr, "pearl: the host's restamp hashing disagrees with the device; "
                    "restamps will run on the device\n");
  }
  device_record(ctx, &r);
  keep_record(ctx, ctx->salt, r);
}

}  // namespace

// Re-draw the operands under a new salt: the first draw of a job is full, every later
// one restamps A (see pearl_host.cu).
extern "C" void pearl_host_reseed(void *handle, uint64_t salt) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx) return;
  DeviceScope scope(ctx->device);
  const double t0 = ctx->trace.on ? now_ms() : 0.0;
  ctx->salt = salt;
  const uint64_t aChunks = (uint64_t)ctx->profile.m * ctx->profile.k / 1024;
  const bool canRestamp = ctx->baseDrawn && aChunks >= 2;
  if (canRestamp && ctx->hostSeeds) {
    PearlRestampRecord r;
    host_record(ctx, salt, true, &r);
    hipLaunchKernelGGL(pearl_restamp_commit, dim3(1), dim3(256), 0, ctx->work, r, ctx->dA, ctx->dTreeA,
                       ctx->dHashA, ctx->dBoundA, ctx->dASeed);
    draw_noise(ctx, true);
    hb3_bytes(r.a_seed, ctx->aSeed);
    keep_record(ctx, salt, r);
  } else {
    if (canRestamp)
      restamp(ctx, salt);
    else
      full_draw(ctx, salt);
    hipMemcpy(ctx->aSeed, ctx->dASeed, PEARL_HASH_BYTES, hipMemcpyDeviceToHost);
    if (canRestamp) {
      PearlRestampRecord r;
      device_record(ctx, &r);
      keep_record(ctx, salt, r);
    } else {
      setup_host_seeds(ctx);
    }
  }
  ctx->baseDrawn = true;
  ctx->haveJob = true;
  if (ctx->trace.on) {
    ctx->trace.draws++;
    ctx->trace.drawMs += now_ms() - t0;
  }
}

namespace {
// One hit of a collected batch, as a result with its share proof (see pearl_host.cu).
void fill_hit(Ctx *ctx, const Ctx::Pending &p, int slot, uint32_t i, PearlSearchResult *out) {
  const uint8_t *dHashes = ctx->dHashes + (size_t)slot * PEARL_MAX_HITS * PEARL_HASH_BYTES;
  const uint32_t *dTranscripts = ctx->dHitTranscript + (size_t)slot * PEARL_MAX_HITS * PEARL_JACKPOT_BUCKETS;
  // Every device read below is queued here and run once, at the end (HitReads).
  HitReads reads;
  reads.to(out->jackpot_hash, dHashes + (size_t)i * PEARL_HASH_BYTES, PEARL_HASH_BYTES);
  memcpy(out->a_seed, p.aSeed, PEARL_HASH_BYTES);
  memcpy(out->b_seed, ctx->bSeed, PEARL_HASH_BYTES);
  out->nonce = p.nonceBase + ctx->hHitIndex[i];
  out->salt = p.salt;
  const PearlRestampRecord *rec = find_record(ctx, p.salt);
  const uint64_t region = out->nonce;
  const uint32_t rowIdx = (uint32_t)(region % ctx->rowsValid);
  const uint32_t colIdx = (uint32_t)((region / ctx->rowsValid) % ctx->colsValid);
  const uint32_t rowOff = pearl_expand_offset(rowIdx, PEARL_ROWS_MASK);
  const uint32_t colOff = pearl_expand_offset(colIdx, PEARL_COLS_MASK);
  uint32_t rows[PEARL_ROWS_COUNT], cols[PEARL_COLS_COUNT];
  for (int j = 0; j < PEARL_ROWS_COUNT; j++) rows[j] = rowOff | PEARL_ROWS_PATTERN[j];
  for (int j = 0; j < PEARL_COLS_COUNT; j++) cols[j] = colOff | PEARL_COLS_PATTERN[j];
  snapshotProof(ctx, true, rows, PEARL_ROWS_COUNT, &out->proof_a.leaf_indices, &out->proof_a.leaves,
                &out->proof_a.siblings, rec, &reads);
  snapshotProof(ctx, false, cols, PEARL_COLS_COUNT, &out->proof_bt.leaf_indices, &out->proof_bt.leaves,
                &out->proof_bt.siblings, nullptr, &reads);
  if (rec)
    hb3_bytes(rec->path[rec->levels - 1], out->proof_a.root);
  else
    reads.to(out->proof_a.root, ctx->dHashA, PEARL_HASH_BYTES);
  reads.to(out->proof_bt.root, ctx->dHashB, PEARL_HASH_BYTES);
  out->proof_a.total_leaves = (uint64_t)ctx->profile.m * ctx->profile.k / 1024;
  out->proof_bt.total_leaves = (uint64_t)ctx->profile.n * ctx->profile.k / 1024;
  out->proof.assign(PEARL_JACKPOT_BUCKETS * 4, 0);
  reads.to(out->proof.data(), dTranscripts + (size_t)i * PEARL_JACKPOT_BUCKETS, PEARL_JACKPOT_BUCKETS * 4);
  run_reads(ctx, &reads);
  out->found = true;
}

// PEARL_AMD_TRACE: one line on stderr every ~10 s (more often in the first minute) with what
// the last window's batches cost. "fold" and "hash" are each kernel's own GPU time, "span" a
// batch's fold start to its hash's end. "wait" is the search thread blocked on a
// batch, "draw" and "job" its time queuing redraws (a job's includes its draw), "hits" its
// time reading hits back.
void trace_line(Ctx *ctx, double now) {
  Ctx::Trace &t = ctx->trace;
  const double win = now - t.t0;
  if (t.batches == 0 || win < (t.lines < 6 ? 2000.0 : 10000.0)) return;
  const double b = (double)t.batches;
  fprintf(stderr,
          "[pearl-amd] %.1f s: %u batches, %.2f ms each; GPU fold %.2f ms, hash %.3f ms, span %.2f ms, "
          "idle %.3f ms a batch (%u seen); wait %.2f ms; %u draws %.2f ms, %u jobs %.1f ms; %u hits %.1f ms\n",
          win / 1000.0, t.batches, win / b, t.foldMs / b, t.hashMs / b, t.spanMs / b,
          t.gaps ? t.idleMs / t.gaps : 0.0, t.gaps, t.waitMs / b, t.draws, t.draws ? t.drawMs / t.draws : 0.0,
          t.jobs, t.jobs ? t.jobMs / t.jobs : 0.0, t.hits, t.hits ? t.hitMs / t.hits : 0.0);
  t.t0 = now;
  t.foldMs = t.hashMs = t.spanMs = t.waitMs = t.drawMs = t.jobMs = t.hitMs = t.idleMs = 0;
  t.batches = t.draws = t.jobs = t.hits = t.gaps = 0;
  t.lines++;
}
}  // namespace

// Queue one batch at nonce_base into the next free pipeline slot: the fold over the
// batch's columns, then the hash over its transcripts, then the hit count back to the
// host behind the slot's event.
extern "C" bool pearl_host_submit(void *handle, uint64_t nonce_base, uint32_t batch,
                                  uint64_t *regions_out, char *err, size_t err_len) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (regions_out) *regions_out = 0;
  if (!ctx || !ctx->haveJob) return false;
  if (ctx->pendCount >= Ctx::kSlots) {
    if (err && err_len) snprintf(err, err_len, "search pipeline full: collect a batch first");
    return false;
  }
  (void)batch;
  DeviceScope scope(ctx->device);
  if (ctx->fold < 0) {
    if (err && err_len) snprintf(err, err_len, "%s", ctx->foldErr);
    return false;
  }
#if PEARL_AMD_STAMP
  // A PEARL_AMD_STAMP build is for timing too: the same rule as below, in its own words.
  if (ctx->stamp) {
    bool zero = true;
    for (int i = 0; i < PEARL_HASH_BYTES; i++) zero = zero && ctx->target[i] == 0;
    if (!zero) {
      if (err && err_len)
        snprintf(err, err_len, "clock-stamp build is timing-only: it runs only at target 0 (hashrate.js), "
                 "never on a job; build without PEARL_AMD_STAMP to mine");
      return false;
    }
  }
#endif
  // A PEARL_AMD_READOUT_DROP or PEARL_AMD_ABLATE build (pearl_amd_fold.h) computes wrong
  // transcripts, so a hit it reported would not be a real one. It runs only at target zero, the
  // hardest, which is what hashrate.js sets: it times the fold and finds nothing. Any other
  // target is a job it must not mine.
  if (ctx->readoutDrop || ctx->timing) {
    bool zero = true;
    for (int i = 0; i < PEARL_HASH_BYTES; i++) zero = zero && ctx->target[i] == 0;
    if (!zero) {
      if (err && err_len && ctx->readoutDrop)
        snprintf(err, err_len, "readout-drop build is timing-only: it runs only at target 0 (hashrate.js), "
                 "never on a job; build without PEARL_AMD_READOUT_DROP to mine");
      else if (err && err_len)
        snprintf(err, err_len, "ablation build is timing-only: it runs only at target 0 (hashrate.js), "
                 "never on a job; build without PEARL_AMD_ABLATE to mine");
      return false;
    }
  }
  const FoldInfo &f = kFolds[ctx->fold];
  const uint32_t k = ctx->profile.k, rank = ctx->profile.rank;
  const uint32_t chunks = rank ? k / rank : 0;
  // Every fold reads whole PEARL_AMD_SLAB-byte slabs, a chunk at a time, and writes each of
  // the 16 transcript words at most once. The reference fold holds one chunk of a row in
  // LDS, 128 bytes at most.
  // Under PEARL_AMD_TILED the matrix folds need the per-tile operands, which only the
  // power-of-two materialise writes.
  const bool kPow2 = k >= 16u && (k & (k - 1u)) == 0u;
  // A code object with the wrap readout (PEARL_AMD_KWRAP in pearl_amd_fold.h: RDNA 4's) folds chunks
  // past 16 into the 16 words, up to PEARL_AMD_KWRAP_CHUNKS. It rotates every word at the start of each
  // 16 chunks, which is the reference's rule only for whole passes, so past 16 the count must be a
  // multiple of 16. The card's caps say whether it has it (ctx->kwrap): this host pass never sees
  // __GFX12__, so the macro here would say nothing about the card.
  const uint32_t maxChunks = ctx->kwrap ? PEARL_AMD_KWRAP_CHUNKS : PEARL_JACKPOT_BUCKETS;
  const bool chunksOk = chunks <= maxChunks && (chunks <= PEARL_JACKPOT_BUCKETS || chunks % PEARL_JACKPOT_BUCKETS == 0u);
  if (rank == 0 || k % rank != 0 || rank % PEARL_AMD_SLAB != 0 || !chunksOk
      || (f.cap == PEARL_AMD_CAP_REF && rank > 128u)
      || (ctx->rank128 && f.cap != PEARL_AMD_CAP_REF && rank != PEARL_FOLD_RANK)   // PEARL_AMD_CRANK (caps)
      || (PEARL_AMD_TILED && f.cap != PEARL_AMD_CAP_REF && !kPow2)) {
    if (err && err_len)
      snprintf(err, err_len, "the AMD folds need rank a multiple of %u (at most 128 for the reference fold%s) "
               "dividing k, at most %u chunks%s, and k a power of two (got rank %u, k %u)",
               (unsigned)PEARL_AMD_SLAB, ctx->rank128 ? ", exactly 128 for this card's matrix fold" : "",
               (unsigned)maxChunks, ctx->kwrap ? " (a multiple of 16 past 16)" : "", rank, k);
    return false;
  }
  const uint32_t batchCols = ctx->colBatch * PEARL_COLS_COUNT;
  if (ctx->profile.m % f.bm != 0 || batchCols % f.bn != 0 || ctx->colsValid % ctx->colBatch != 0) {
    if (err && err_len)
      snprintf(err, err_len, "the %s fold tiles %ux%u: m %u, a batch of %u columns (%u of %u column offsets)",
               f.name, f.bm, f.bn, ctx->profile.m, batchCols, ctx->colBatch, ctx->colsValid);
    return false;
  }
  const int slot = (ctx->pendHead + ctx->pendCount) % Ctx::kSlots;
  const uint32_t regions = ctx->batch;
  // The batch's first column offset index, and the first column (row of B') it is: the
  // batch starts on a whole number of batches, so on a whole 64-column span.
  const uint32_t col_off = (uint32_t)((nonce_base / ctx->rowsValid) % ctx->colsValid);
  const uint32_t col0 = col_off * PEARL_COLS_COUNT;

  PearlTranscriptTest test;
  memcpy(test.key, ctx->aSeed, sizeof(test.key));
  for (int i = 0; i < 8; i++) {
    const uint8_t *t = ctx->target + i * 4;
    test.target_w[i] = ((uint32_t)t[0] << 24) | ((uint32_t)t[1] << 16) | ((uint32_t)t[2] << 8) | (uint32_t)t[3];
  }
  test.hash_big_endian = (int)ctx->profile.hash_big_endian;
  PearlHitList hitList;
  hitList.count = ctx->dHitCount + slot;
  hitList.index = ctx->dHitIndex + (size_t)slot * PEARL_MAX_HITS;
  hitList.hash = reinterpret_cast<uint32_t *>(ctx->dHashes + (size_t)slot * PEARL_MAX_HITS * PEARL_HASH_BYTES);
  hitList.transcript = ctx->dHitTranscript + (size_t)slot * PEARL_MAX_HITS * PEARL_JACKPOT_BUCKETS;
  hipStream_t st = ctx->work;
  const bool fusedQ = ctx->fuse && kFolds[ctx->fold].cap != PEARL_AMD_CAP_REF;
  // PEARL_AMD_HASHQ (Ctx): this slot's last hash read its transcripts and counted into its hit list; wait for it.
  const bool hashq = ctx->hashq && !fusedQ;
  if (hashq && ctx->hashqWait) hipStreamWaitEvent(st, ctx->slotDone[slot], 0);
  // PEARL_AMD_HASHQ_ZERO: only the hash kernel counts into the list, so with the hash on its own queue the count is
  // zeroed there, and nothing sits between this fold and the last.
  const bool zeroLate = hashq && ctx->hashqZero;
  if (!zeroLate) hipMemsetAsync(hitList.count, 0, sizeof(uint32_t), st);
  if (ctx->trace.on) hipEventRecord(ctx->trace.f0[slot], st);

  // PEARL_AMD_STRIPS: the grid's z is the strips, each a batch of the card's width (grid.x column tiles).
  const dim3 grid(batchCols / f.bn / ctx->strips, ctx->profile.m / f.bm, ctx->strips), block(PEARL_AMD_THREADS);
  const int8_t *ap = ctx->dAp, *bp = ctx->dBp;
  const uint32_t rv = ctx->rowsValid;
  uint32_t *tr = ctx->dTr[slot];
  // In a code object with PEARL_AMD_FUSE (pearl_amd_fold.h; the caps say, ctx->fuse) every fold but the
  // reference hashes its own transcripts into the hit list, so it is given no transcript buffer (it stores
  // nothing) and no hash kernel follows it. Every fold takes the test and the list either way.
  const bool fused = ctx->fuse && f.cap != PEARL_AMD_CAP_REF;
  if (fused) tr = nullptr;
  switch (f.cap) {
    case PEARL_AMD_CAP_MFMA16:
      hipLaunchKernelGGL(pearl_amd_fold_mfma16, grid, block, 0, st, ap, bp, k, rank, col0, rv, tr, test, hitList);
      break;
    case PEARL_AMD_CAP_MFMA8:
      hipLaunchKernelGGL(pearl_amd_fold_mfma8, grid, block, 0, st, ap, bp, k, rank, col0, rv, tr, test, hitList);
      break;
    case PEARL_AMD_CAP_WMMA11:
      hipLaunchKernelGGL(pearl_amd_fold_wmma11, grid, block, 0, st, ap, bp, k, rank, col0, rv, tr, test, hitList);
      break;
    case PEARL_AMD_CAP_WMMA12:
      hipLaunchKernelGGL(pearl_amd_fold_wmma12, grid, block, 0, st, ap, bp, k, rank, col0, rv, tr, test, hitList);
      break;
    case PEARL_AMD_CAP_DOT4:
      hipLaunchKernelGGL(pearl_amd_fold_dot4, grid, block, 0, st, ap, bp, k, rank, col0, rv, tr, test, hitList);
      break;
    default:
      hipLaunchKernelGGL(pearl_amd_fold_ref, grid, block, 0, st, ap, bp, k, rank, col0, rv, tr, test, hitList);
      break;
  }
  if (ctx->trace.on) hipEventRecord(ctx->trace.f1[slot], st);
  if (hashq) {   // PEARL_AMD_HASHQ (Ctx): the hash, the count and the slot's event on the hash queue
    hipEventRecord(ctx->foldDone[slot], st);
    st = ctx->hashq;
    hipStreamWaitEvent(st, ctx->foldDone[slot], 0);
    if (zeroLate) hipMemsetAsync(hitList.count, 0, sizeof(uint32_t), st);
  }
  if (!fused)
    hipLaunchKernelGGL(pearl_amd_hash, dim3((regions + PEARL_AMD_THREADS - 1) / PEARL_AMD_THREADS),
                       dim3(PEARL_AMD_THREADS), 0, st, (const uint32_t *)tr, regions, test, hitList);
  if (ctx->trace.on) hipEventRecord(ctx->trace.h1[slot], st);
  hipMemcpyAsync(ctx->hSlotCount + slot, hitList.count, sizeof(uint32_t), hipMemcpyDeviceToHost, st);
  hipEventRecord(ctx->slotDone[slot], st);
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) {
    if (err && err_len) snprintf(err, err_len, "HIP error during search: %s", hipGetErrorString(e));
    return false;
  }
  Ctx::Pending &p = ctx->pend[slot];
  p.nonceBase = nonce_base;
  p.salt = ctx->salt;
  memcpy(p.aSeed, ctx->aSeed, PEARL_HASH_BYTES);
  p.regions = regions;
  ctx->pendCount++;
  if (regions_out) *regions_out = regions;
  return true;
}

extern "C" int pearl_host_pending(void *handle) {
  const Ctx *ctx = static_cast<const Ctx *>(handle);
  return ctx ? ctx->pendCount : 0;
}

extern "C" bool pearl_host_collect(void *handle, PearlSearchResult *out, uint64_t *attempts,
                                   char *err, size_t err_len) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (attempts) *attempts = 0;
  if (!ctx || !out || ctx->pendCount == 0) return false;
  DeviceScope scope(ctx->device);
  const int slot = ctx->pendHead;
  const Ctx::Pending p = ctx->pend[slot];
  const double tw = ctx->trace.on ? now_ms() : 0.0;
  hipError_t e = hipEventSynchronize(ctx->slotDone[slot]);
  if (e == hipSuccess) e = hipGetLastError();
  ctx->pendHead = (ctx->pendHead + 1) % Ctx::kSlots;
  ctx->pendCount--;
  if (e != hipSuccess) {
    if (err && err_len) snprintf(err, err_len, "HIP error during search: %s", hipGetErrorString(e));
    return false;
  }
  if (ctx->trace.on) {
    Ctx::Trace &t = ctx->trace;
    const double now = now_ms();
    if (t.t0 == 0) t.t0 = now;
    float fold = 0, hash = 0, span = 0;
    hipEventElapsedTime(&fold, t.f0[slot], t.f1[slot]);
    hipEventElapsedTime(&hash, t.f1[slot], t.h1[slot]);
    hipEventElapsedTime(&span, t.f0[slot], t.h1[slot]);
    t.foldMs += fold;
    t.hashMs += hash;
    t.spanMs += span;
    t.waitMs += now - tw;
    t.batches++;
    // The next batch was queued before this one was waited for. If it reached the GPU
    // only after this one ended, the GPU had nothing to do in between: the search thread
    // came back from the batch before too late (or a redraw sat between them).
    if (ctx->pendCount > 0) {
      const int next = ctx->pendHead;
      float gap = 0;
      if (event_query(t.f0[next]) == hipSuccess
          && hipEventElapsedTime(&gap, t.h1[slot], t.f0[next]) == hipSuccess) {
        t.gaps++;
        if (gap > 0) t.idleMs += gap;
      }
      (void)hipGetLastError();
    }
  }
#if PEARL_AMD_STAMP
  if (ctx->stamp && ctx->stampAt && ++ctx->stampSeen == ctx->stampAt) {
    const char *path = getenv("PEARL_AMD_STAMP_DUMP");
    const size_t words = ctx->stampBlocks * (PEARL_AMD_THREADS / 32u) * PEARL_AMD_STAMP_WORDS;
    std::vector<uint32_t> w(words);
    int wallKhz = 0;
    if (hipDeviceGetAttribute(&wallKhz, hipDeviceAttributeWallClockRate, ctx->device) != hipSuccess) wallKhz = 0;
    FILE *fp = path && path[0] ? fopen(path, "wb") : nullptr;
    // Every queued fold ends first, so the stamps are one whole launch's: the last one's.
    if (fp && hipStreamSynchronize(ctx->work) == hipSuccess
        && hipMemcpy(w.data(), ctx->dStamps, words * 4, hipMemcpyDeviceToHost) == hipSuccess) {
      const FoldInfo &sf = kFolds[ctx->fold];
      const uint32_t head[16] = {0x31545350u, PEARL_AMD_STAMP_WORDS, PEARL_AMD_THREADS / 32u, (uint32_t)ctx->stampBlocks,
                                 ctx->colBatch * PEARL_COLS_COUNT / sf.bn, (uint32_t)wallKhz, ctx->profile.k,
                                 (uint32_t)ctx->stampSeen, ctx->profile.m, ctx->colBatch * PEARL_COLS_COUNT,
                                 0, 0, 0, 0, 0, 0};
      fwrite(head, 4, 16, fp);
      fwrite(w.data(), 4, words, fp);
    }
    if (fp) fclose(fp);
    (void)hipGetLastError();
  }
#endif
  const uint32_t hits = ctx->hSlotCount[slot];
  if (attempts) *attempts = p.regions;
  ctx->extraHits.clear();
  ctx->extraNext = 0;
  if (hits > 0) {
    const double th = ctx->trace.on ? now_ms() : 0.0;
    const uint32_t *dIndex = ctx->dHitIndex + (size_t)slot * PEARL_MAX_HITS;
    const uint32_t n_hits = hits < PEARL_MAX_HITS ? hits : PEARL_MAX_HITS;
    side_copy(ctx, ctx->hHitIndex.data(), dIndex, (size_t)n_hits * sizeof(uint32_t));
    std::vector<uint32_t> order(n_hits);
    for (uint32_t i = 0; i < n_hits; i++) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](uint32_t a, uint32_t b) { return ctx->hHitIndex[a] < ctx->hHitIndex[b]; });
    fill_hit(ctx, p, slot, order[0], out);
    ctx->extraHits.resize(n_hits - 1);
    for (uint32_t i = 1; i < n_hits; i++) fill_hit(ctx, p, slot, order[i], &ctx->extraHits[i - 1]);
    if (ctx->trace.on) {
      const double now = now_ms();
      ctx->trace.hits += n_hits;
      ctx->trace.hitMs += now - th;
      trace_line(ctx, now);
    }
    return true;
  }
  if (ctx->trace.on) trace_line(ctx, now_ms());
  out->found = false;
  return false;
}

extern "C" bool pearl_host_next_hit(void *handle, PearlSearchResult *out) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx || !out || ctx->extraNext >= ctx->extraHits.size()) return false;
  *out = ctx->extraHits[ctx->extraNext++];
  return true;
}

extern "C" bool pearl_host_search(void *handle, uint64_t nonce_base, uint32_t batch,
                                  PearlSearchResult *out, uint64_t *attempts, char *err,
                                  size_t err_len) {
  if (attempts) *attempts = 0;
  if (!handle || !out) return false;
  Ctx *ctx = static_cast<Ctx *>(handle);
  while (ctx->pendCount > 0) {
    PearlSearchResult drop;
    uint64_t a = 0;
    pearl_host_collect(handle, &drop, &a, nullptr, 0);
  }
  if (!pearl_host_submit(handle, nonce_base, batch, nullptr, err, err_len)) return false;
  return pearl_host_collect(handle, out, attempts, err, err_len);
}

// Which fold this context runs, for the bench probe and the logs.
extern "C" const char *pearl_host_fold_name(void *handle) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx || ctx->fold < 0) return "unresolved";
  const bool fused = ctx->fuse && kFolds[ctx->fold].cap != PEARL_AMD_CAP_REF;
  char strips[48] = {0};
  if (ctx->strips > 1u) snprintf(strips, sizeof strips, " in %u strips of %u", ctx->strips, ctx->stripBatch);
  snprintf(ctx->foldName, sizeof ctx->foldName, "%s, %s, batch %u column offsets%s, k %u%s%s%s%s",
           kFolds[ctx->fold].text, fused ? "hash in the fold" : "hash in its own kernel", ctx->colBatch, strips,
           ctx->profile.k, ctx->kwrap ? ", wrap readout" : "",
           ctx->emulated ? ", emulated on the CPU (HIP-CPU)" : "", ctx->arch[0] ? ", " : "", ctx->arch);
  return ctx->foldName;
}

// RTX 30's launch shapes (pearl_host.cu): an AMD card has none to choose, so the core runs no
// trial and reports no shapes.
extern "C" int pearl_host_shape_count(void *) { return 0; }
extern "C" void pearl_host_shape_name(void *, int, char *buf, size_t len) {
  if (buf && len) snprintf(buf, len, "none");
}
extern "C" bool pearl_host_set_shape(void *, int) { return false; }

extern "C" bool pearl_host_leaf_chunks(void *handle, int isA, const uint32_t *leaf_indices,
                                       uint32_t count, uint8_t *out) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx || !leaf_indices || !out) return false;
  DeviceScope scope(ctx->device);
  const int8_t *src = isA ? ctx->dA : ctx->dB;
  const uint64_t bytes = isA ? (uint64_t)ctx->profile.m * ctx->profile.k : (uint64_t)ctx->profile.n * ctx->profile.k;
  for (uint32_t i = 0; i < count; i++) {
    const uint64_t off = (uint64_t)leaf_indices[i] * 1024;
    if (off + 1024 > bytes) return false;
    hipMemcpy(out + (size_t)i * 1024, operand_chunk(ctx, src, leaf_indices[i]), 1024, hipMemcpyDeviceToHost);
  }
  return true;
}

extern "C" bool pearl_host_tree_nodes(void *handle, int isA, uint32_t level, const uint32_t *indices,
                                      uint32_t count, uint8_t *out) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx || !indices || !out) return false;
  DeviceScope scope(ctx->device);
  const std::vector<uint64_t> &offs = isA ? ctx->layerOffA : ctx->layerOffB;
  const uint32_t *tree = isA ? ctx->dTreeA : ctx->dTreeB;
  if (!tree || level >= offs.size()) return false;
  const uint64_t base = offs[level];
  for (uint32_t i = 0; i < count; i++)
    hipMemcpy(out + (size_t)i * PEARL_HASH_BYTES, tree + (base + indices[i]) * 8, PEARL_HASH_BYTES,
              hipMemcpyDeviceToHost);
  return true;
}

extern "C" uint32_t pearl_host_tree_levels(void *handle, int isA) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx) return 0;
  return (uint32_t)(isA ? ctx->layerOffA.size() : ctx->layerOffB.size());
}
