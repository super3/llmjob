// Host driver for the Pearl core on SYCL (Intel GPUs through Level Zero or
// OpenCL, and the SYCL CPU device for testing). It implements the extern "C"
// pearl_host_* API that ../src/pearl_core.cc calls, so the N-API addon is built
// from that file unchanged and the app loads the result like any other core.
//
// The logic follows ../src/pearl_host.cu: the same draw (job_key, constant or
// hashed fill, salt stamp, commitment trees, cert-v3 seeds, noise), the same
// region numbering, the same two-batch pipeline and the same share proofs. What
// is simpler here, on purpose, for a first version:
//
//   - A same-job redraw rebuilds A's whole tree on the device and waits for the
//     root, where pearl_host.cu repairs leaf 0's path and hashes the new a_seed
//     on the host without waiting. One wait a salt.
//   - Before a redraw touches A, any batch still queued is finished and its hits
//     read (resolve_pending), so a proof is always read from its own salt's tree.
//     pearl_host.cu instead keeps per-salt records of what a restamp changes.
//   - Both pipeline slots run on one in-order queue. Proof reads go on a second
//     queue so they do not wait behind the next batch.
//
// What pearl_host.cu has that this host answers plainly. One kind of Intel card
// has run it (Arc A750, on 10 November 2026), and nothing has been tuned on it:
//
//   - k: the folds are built for PEARL_FOLD_K only, so pearl_host_card_k gives
//     the profile's k back.
//   - Launch shapes: none, so pearl_core.cc runs no startup trial.
//   - Untested cards: every GPU counts as one. It mines only for an opener
//     that checks each hit before sending it (pearl_host_create_checked), and
//     tells that opener so. PEARL_SYCL_UNTESTED=1 opens it for an opener that
//     sends nothing (bench.js).
//
// Device choice: Level Zero GPUs, else OpenCL GPUs, ranked by compute units x
// clock. PEARL_SYCL_DEVICE=cpu takes the SYCL CPU device instead, which is how
// the correctness gate runs without an Intel GPU. See README.md for the other
// switches (PEARL_SYCL_FOLD, PEARL_SYCL_COL_BATCH, PEARL_SYCL_BAND, PEARL_SYCL_DUTY).

#include <sycl/sycl.hpp>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#if defined(PEARL_SYCL_XMX_HW) || defined(PEARL_SYCL_XMX_JIT)
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#endif

#include "../src/pearl_config.h"
#include "pearl_sycl_blake3.h"
#include "pearl_sycl_kernels.hpp"

// The same two structs pearl_core.cc declares (and pearl_host.cu mirrors). Both
// files must agree on the layout; both are built by the same compiler and
// standard library in build.sh.
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

using namespace pearl_sycl;

// Which fold a context runs. See pearl_sycl_kernels.hpp.
// The ring fold's variant (pearl_sycl_kernels.hpp, FoldXmxR): 12 is what has been verified and mined.
#ifndef PEARL_RING_VAR
#define PEARL_RING_VAR 12
#endif

enum Fold { kDot = 0, kXmx16 = 1, kXmx8 = 2, kXmx16Emu = 3, kXmx8Emu = 4, kXmx8F = 5, kXmx8FEmu = 6, kXmx8R = 7,
            kXmx8REmu = 8, kXmx8W = 11, kXmx8WEmu = 12 };  // 9 and 10: xmx8s, below

// The folds that read the noised operands in fragment order (pearl_sycl_kernels.hpp, Layout).
#if defined(PEARL_SYCL_XMX8S)
// The shared-A' fold, an experiment (pearl_sycl_kernels.hpp, FoldXmxS): same operand layouts as xmx8f.
enum : int { kXmx8S = 9, kXmx8SEmu = 10 };
bool fold_is_frag(int f) {
  return f == kXmx8F || f == kXmx8FEmu || f == kXmx8R || f == kXmx8REmu || f == kXmx8W || f == kXmx8WEmu || f == kXmx8S
         || f == kXmx8SEmu;
}
#else
bool fold_is_frag(int f) {
  return f == kXmx8F || f == kXmx8FEmu || f == kXmx8R || f == kXmx8REmu || f == kXmx8W || f == kXmx8WEmu;
}
#endif
// How a fold wants the noised operands stored: 0 row-major, 1 fragment order (kFragA, kFragB), 2 fragment
// order for block reads (kFragAT, kFragBT).
#if (PEARL_RING_VAR & 262144)
// 3, for the ring fold's timing variant PEARL_RING_VAR & 262144 only: A' for block reads, B' in fragment order.
int fold_layout(int f) { return f == kXmx8R || f == kXmx8REmu ? 3 : fold_is_frag(f) ? 1 : 0; }
#else
// The split fold reads the ring fold's layouts.
int fold_layout(int f) {
  return f == kXmx8R || f == kXmx8REmu || f == kXmx8W || f == kXmx8WEmu ? 2 : fold_is_frag(f) ? 1 : 0;
}
#endif

const char *fold_label(int f) {
  switch (f) {
    case kDot: return "dot (int8 dot products, local memory)";
    case kXmx16: return "xmx16 (DPAS 8x16x32, sub-group 16)";
    case kXmx8: return "xmx8 (DPAS 8x8x32, sub-group 8)";
    case kXmx16Emu: return "xmx16-emu (DPAS reference code, sub-group 16)";
    case kXmx8Emu: return "xmx8-emu (DPAS reference code, sub-group 8)";
    case kXmx8F: return "xmx8f (DPAS 8x8x32, sub-group 8, 16 rows a sub-group, operands in fragment order)";
    case kXmx8FEmu: return "xmx8f-emu (DPAS reference code, sub-group 8, operands in fragment order)";
    case kXmx8R: return "xmx8r (DPAS 8x8x32, sub-group 8, 16 rows a sub-group, B' shared through local memory)";
    case kXmx8REmu: return "xmx8r-emu (DPAS reference code, sub-group 8, B' shared through local memory)";
    case kXmx8W: return "xmx8w (split DPAS 8x8x32, sub-group 8, 32 by 32 a sub-group, A' shared by pairs, B' through local memory)";
    case kXmx8WEmu: return "xmx8w-emu (DPAS reference code, sub-group 8, 32 by 32 a sub-group, B' through local memory)";
#if defined(PEARL_SYCL_XMX8S)
    case kXmx8S: return "xmx8s (DPAS 8x8x32, sub-group 8, 32 by 32 a sub-group, A' shared in local memory)";
    case kXmx8SEmu: return "xmx8s-emu (DPAS reference code, A' shared in local memory)";
#endif
    default: return "?";
  }
}

const int kSlots = 2;

struct Slot {
  bool busy = false;
  bool resolved = false;  // hits already read (resolve_pending)
  uint64_t nonceBase = 0;
  uint64_t salt = 0;
  uint8_t aSeed[PEARL_HASH_BYTES] = {0};
  uint32_t regions = 0;
  sycl::event done;
  std::vector<PearlSearchResult> hits;
  std::string err;
};

struct Ctx {
  PearlProfile profile;
  sycl::device dev;
  sycl::queue *q = nullptr;     // draws and folds, in order
  sycl::queue *side = nullptr;  // proof reads
  int deviceIndex = 0;
  // PEARL_SYCL_DUTY, 10 to 100: the share of the time the card folds. Under 100 each batch waits for the one
  // before it and then leaves the card idle for the rest, for a card that cannot shed the heat of a full load.
  int dutyPct = 100;
  bool dutyStarted = false;
  std::chrono::steady_clock::time_point dutyMark;
  std::string deviceName;
  int fold = kDot;
  bool xmxHw = false;  // the fold runs the hardware instruction
  int hwSg = 0;        // the sub-group size this device has hardware DPAS at: 16, 8 or 0
  bool isGpu = false;  // a GPU, not the SYCL CPU device
  bool untested = false;  // a GPU, which no fold has run on yet, opened because its opener checks every hit
  std::string foldName;   // pearl_host_fold_name's answer, written once at create
  uint64_t memTotal = 0;  // the device's memory, bytes; this and memUsed are written once at create
  uint64_t memUsed = 0;   // what this context allocated on the device, bytes
  uint32_t band = 1;
  bool compact = false;

  int8_t *dA = nullptr, *dB = nullptr;    // operands (compact: two chunks each)
  int8_t *dAp = nullptr, *dBp = nullptr;  // noised operands, row-major or in fragment order (frag)
  int frag = 0;                           // how dAp and dBp are stored now (fold_layout)
  bool straddle = false;                  // dAp or dBp crosses a 4 GB boundary: no split fold (launch_fold)
  int seqp = 0;                           // xmx8w's set stream: 0 off, 1 on, 2 in 24.35's forms (goal-arc21/23; launch_fold)
  uint32_t ringWpg = 1;                   // xmx8r: windows a work-group
  uint32_t aStride = 0, bStride = 0;      // fragment order: bytes a 16-row block of A', a 64-column block of B'
  int8_t *dEAL = nullptr, *dEBR = nullptr;
  uint32_t *dPermA = nullptr, *dPermB = nullptr;
  uint32_t *dTreeA = nullptr, *dTreeB = nullptr;  // levels end to end, 8 words a node
  std::vector<uint64_t> layerOffA, layerOffB;
  HitBuf *dHits[kSlots] = {nullptr, nullptr};
  HitBuf *hHits[kSlots] = {nullptr, nullptr};  // pinned host copies
  uint32_t *hRoot = nullptr;                   // pinned, 8 words
#if (defined(PEARL_SYCL_ESIMD) || defined(PEARL_W_SEPEPI)) && defined(PEARL_SYCL_XMX_JIT)
  uint32_t *dWords = nullptr;  // goal-arc9: the ESIMD xmx8w's words, 16 a region (PEARL_SYCL_ESIMD=1)
#endif
#if defined(PEARL_W_SEPQ) && defined(PEARL_SYCL_XMX_JIT)
  uint32_t *dWordsS[2] = {nullptr, nullptr};  // goal-arc17: a words buffer a slot, the hash runs on `side`
  sycl::event sepEv;
  bool sepEvSet = false;
#endif

  uint32_t rowsValid = 1, colsValid = 1, colBatch = 1, batch = 1;
  uint8_t jobKey[32] = {0};
  uint8_t rootA[32] = {0}, rootB[32] = {0};
  uint8_t aSeed[32] = {0}, bSeed[32] = {0};
  uint8_t header[PEARL_HEADER_BYTES] = {0};
  uint8_t target[PEARL_HASH_BYTES] = {0};
  uint64_t salt = 0;
  bool haveJob = false;
  bool baseDrawn = false;

  Slot slots[kSlots];
  int head = 0, count = 0;
  std::vector<PearlSearchResult> extra;
  size_t extraNext = 0;

  std::mutex errMu;
  std::string asyncErr;
};

// The card pearl_host_select_device chose, for pearl_host_create on the same
// thread (pearl_core.cc calls the two back to back).
thread_local int g_selected = -1;

bool fail(char *err, size_t err_len, const std::string &msg) {
  if (err && err_len) snprintf(err, err_len, "%s", msg.c_str());
  return false;
}

std::string env_str(const char *name, const char *dflt) {
  const char *v = getenv(name);
  return v && *v ? std::string(v) : std::string(dflt);
}

// Options for Intel's finalizer (vISA), set once before the driver compiles anything: the driver reads
// IGC_VISAOptions when it builds a kernel. On an A750 (goal-arc9, 12 November) xmx8w ran 91.8 TH/s with
// "-insertDPASRSDummyMov -DPASTokenReduction" against 90.6 without (card 5, three rounds), and 85.2 against 84.2
// on HiveOS's 24.35 driver; both drivers take the options and the fold check passes. The first swaps some of the
// dummy DPAS a read-suppression workaround puts before a run of DPAS (6 to 4 in the loop) for movs, the second
// lets DPAS share scoreboard tokens. goal-arc11 added -noloopra (no separate register allocation for loops):
// xmx8w's loop then has 3 dummy DPAS where it had 4, 93.45 TH/s against 92.92 (card 5, three rounds); the 24.35
// driver takes it too (check PASS). Each dummy costs xmx8w about 0.5%. goal-arc12 added -noroundrobin and
// -keepDPASMacroInSchedule: with xmx8w's 8x4 DPAS (no dummies left to shift) 96.4 TH/s against 94.85 with the
// three before (card 5, two rounds). goal-arc13 dropped -keepDPASMacroInSchedule again: with A' and B' read as
// 256-byte inline-vISA messages, HiveOS's 24.35 finalizer under it gets most regions wrong (16,3xx of 16,384, a
// different count each run); without it the check passes there, and the new driver runs level (98.0-98.1 against
// 97.9-98.0, card 5). A user's own IGC_VISAOptions wins; PEARL_SYCL_VISA=none sets nothing.
void set_finalizer_options() {
  static std::once_flag once;
  std::call_once(once, []() {
    const std::string o = env_str("PEARL_SYCL_VISA", "-insertDPASRSDummyMov -DPASTokenReduction -noloopra -noroundrobin");
    if (o == "none" || getenv("IGC_VISAOptions")) return;
#if defined(_WIN32)
    _putenv_s("IGC_VISAOptions", o.c_str());
#else
    setenv("IGC_VISAOptions", o.c_str(), 0);
#endif
  });
}

// The devices this core may open, in index order. Level Zero GPUs if there are
// any (the same GPU is usually listed by OpenCL too), else OpenCL GPUs.
// PEARL_SYCL_DEVICE=cpu: the CPU device, for the correctness gate.
std::vector<sycl::device> list_devices() {
  set_finalizer_options();
  std::vector<sycl::device> l0, ocl, cpu;
  try {
    for (const auto &p : sycl::platform::get_platforms()) {
      const sycl::backend be = p.get_backend();
      for (const auto &d : p.get_devices()) {
        if (d.is_gpu()) {
          if (be == sycl::backend::ext_oneapi_level_zero) l0.push_back(d);
          else if (be == sycl::backend::opencl) ocl.push_back(d);
        } else if (d.is_cpu()) {
          cpu.push_back(d);
        }
      }
    }
  } catch (const std::exception &) {
  }
  if (env_str("PEARL_SYCL_DEVICE", "gpu") == "cpu") return cpu;
  return !l0.empty() ? l0 : ocl;
}

// The gap after each block of the noised operands in fragment order (Layout), in bytes: 16-row blocks of A',
// 64-column blocks of B'. Without one the blocks sit a power of two apart (32 KB and 128 KB at k 2048), so
// the 16 A' streams of a work-group of the split fold land in the same cache sets. On an A750 (goal-arc8,
// benchmark.md) 256 bytes after each A' block took xmx8w from 86.5 to 87.6-88.0 TH/s, both at the 188 W
// limit (128 to 3,200 all gained, 256 most); a gap after B' blocks gained nothing. PEARL_SYCL_APAD and
// PEARL_SYCL_BPAD force others (rounded down to multiples of 128).
uint32_t operand_pad(bool isA) {
  const std::string s = env_str(isA ? "PEARL_SYCL_APAD" : "PEARL_SYCL_BPAD", "");
  unsigned long v = s.empty() ? (isA ? 256ul : 0ul) : strtoul(s.c_str(), nullptr, 10);
  if (v > 65536ul) v = 65536ul;
  return (uint32_t)(v / 128ul * 128ul);
}
// goal-arc21: and 16 KB after the last block. xmx8w's k-step (PEARL_W_SA) loads A' a k-step ahead and the copy's
// B' five k-steps ahead with no clamp, so its last k-steps read up to 10 KB past an operand's last block.
size_t padded_bytes(size_t rows, size_t k, bool isA) {
  const size_t blk = isA ? 16u : 64u;
  return (rows + blk - 1) / blk * (blk * k + operand_pad(isA)) + 16384u;
}

// What one context of `profile` allocates on the device, in bytes.
size_t needed_bytes(const PearlProfile *p) {
  const size_t k = p->k, rank = p->rank;
  const size_t aBytes = (size_t)p->m * k, bBytes = (size_t)p->n * k;
  const bool compact = p->operand_fill == PEARL_OPERAND_CONST;
  const size_t stored = compact ? 4096 : aBytes + bBytes;
  const size_t primes = padded_bytes(p->m, k, true) + padded_bytes(p->n, k, false);  // blocks and their gaps
  const size_t noise = ((size_t)p->m + p->n) * rank + 4 * k * sizeof(uint32_t);
  const size_t trees = 2 * (aBytes / 1024 + bBytes / 1024) * 32;
  return stored + primes + noise + trees + kSlots * sizeof(HitBuf) + (1u << 20);
}

uint32_t floor_pow2(uint64_t v) {
  uint32_t p = 1;
  while ((uint64_t)p * 2 <= v && p < 0x80000000u) p *= 2;
  return p;
}

#if defined(PEARL_SYCL_XMX_HW) || defined(PEARL_SYCL_XMX_JIT)
// Which XMX DPAS the device's architecture has (16, 8), or 0 for none. Asked of
// the device at run time, so it is the same answer whether this build's kernels
// were prebuilt for named GPUs or are compiled by the driver.
int xmx_sg(const sycl::device &d) {
  namespace syclex = sycl::ext::oneapi::experimental;
  using arch = syclex::architecture;
  try {
    const arch a = d.get_info<syclex::info::device::architecture>();
    if (a == arch::intel_gpu_pvc || a == arch::intel_gpu_bmg_g21 || a == arch::intel_gpu_bmg_g31
        || a == arch::intel_gpu_lnl_m || a == arch::intel_gpu_ptl_h || a == arch::intel_gpu_ptl_u)
      return 16;
    if (a == arch::intel_gpu_acm_g10 || a == arch::intel_gpu_acm_g11 || a == arch::intel_gpu_acm_g12)
      return 8;
  } catch (const std::exception &) {
  }
  return 0;
}
#else
int xmx_sg(const sycl::device &) { return 0; }
#endif

// The fold for this context: PEARL_SYCL_FOLD, or by default the hardware DPAS
// where this binary has it for the device's architecture, else the plain fold.
bool pick_fold(Ctx *ctx, char *err, size_t err_len) {
  const std::string want = env_str("PEARL_SYCL_FOLD", "auto");
  const int sg = xmx_sg(ctx->dev);
  if (want == "auto") {
    // Alchemist (sub-group 8): the split fold since 12 Nov 2026 (88 TH/s a cool A750 against 68 for the ring
    // fold xmx8r, an hour on four mining cards with no share rejected; launch_fold runs xmx8r when the work-group
    // is not 8 windows). PEARL_SYCL_FOLD=xmx8r is the way back.
    ctx->fold = sg == 16 ? kXmx16 : sg == 8 ? kXmx8W : kDot;
  } else if (want == "dot") {
    ctx->fold = kDot;
  } else if (want == "xmx16" || want == "xmx") {
    ctx->fold = kXmx16;
  } else if (want == "xmx8") {
    ctx->fold = kXmx8;
  } else if (want == "xmx16-emu") {
    ctx->fold = kXmx16Emu;
  } else if (want == "xmx8-emu") {
    ctx->fold = kXmx8Emu;
  } else if (want == "xmx8f") {
    ctx->fold = kXmx8F;
  } else if (want == "xmx8f-emu") {
    ctx->fold = kXmx8FEmu;
  } else if (want == "xmx8r") {
    ctx->fold = kXmx8R;
  } else if (want == "xmx8r-emu") {
    ctx->fold = kXmx8REmu;
  } else if (want == "xmx8w") {
    ctx->fold = kXmx8W;
  } else if (want == "xmx8w-emu") {
    ctx->fold = kXmx8WEmu;
#if defined(PEARL_SYCL_XMX8S)
  } else if (want == "xmx8s") {
    ctx->fold = kXmx8S;
  } else if (want == "xmx8s-emu") {
    ctx->fold = kXmx8SEmu;
#endif
  } else {
    return fail(err, err_len, "PEARL_SYCL_FOLD=" + want
                                  + " is not one of auto, dot, xmx16, xmx8, xmx8f, xmx8r, xmx8w, xmx16-emu, xmx8-emu, xmx8f-emu,"
                                  + " xmx8r-emu, xmx8w-emu");
  }
  ctx->hwSg = sg;
  ctx->xmxHw = (ctx->fold == kXmx16 && sg == 16)
                || ((ctx->fold == kXmx8 || ctx->fold == kXmx8F || ctx->fold == kXmx8R || ctx->fold == kXmx8W) && sg == 8);
#if defined(PEARL_SYCL_XMX8S)
  if (ctx->fold == kXmx8S && sg == 8) ctx->xmxHw = true;
#endif
  return true;
}

void record_async(Ctx *ctx, sycl::exception_list el) {
  for (auto &e : el) {
    try {
      std::rethrow_exception(e);
    } catch (const std::exception &x) {
      std::lock_guard<std::mutex> lock(ctx->errMu);
      if (ctx->asyncErr.empty()) ctx->asyncErr = x.what();
    }
  }
}

// Any error a queue reported since the last call, or an empty string.
std::string take_async(Ctx *ctx) {
  try {
    ctx->q->throw_asynchronous();
    ctx->side->throw_asynchronous();
  } catch (...) {
  }
  std::lock_guard<std::mutex> lock(ctx->errMu);
  std::string e = ctx->asyncErr;
  ctx->asyncErr.clear();
  return e;
}

void free_all(Ctx *ctx) {
  if (!ctx->q) return;
  try {
    ctx->q->wait();
    if (ctx->side) ctx->side->wait();  // null when the second queue could not be made
  } catch (...) {
  }
  void *ps[] = {ctx->dA, ctx->dB, ctx->dAp, ctx->dBp, ctx->dEAL, ctx->dEBR, ctx->dPermA,
                ctx->dPermB, ctx->dTreeA, ctx->dTreeB, ctx->dHits[0], ctx->dHits[1],
                ctx->hHits[0], ctx->hHits[1], ctx->hRoot};
  for (void *p : ps)
    if (p) sycl::free(p, *ctx->q);
  delete ctx->side;
  delete ctx->q;
  ctx->q = ctx->side = nullptr;
}

// Keyed BLAKE3 over a whole operand as a tree over 1024-byte chunks, every
// level kept (the share proofs read siblings out of it). The chunk count is a
// power of two (pearl_host_create checks), so the pairwise fold is BLAKE3's
// tree. The root lands in out32, which waits for the device.
void commitment(Ctx *ctx, const int8_t *data, uint64_t len, uint32_t *tree,
                std::vector<uint64_t> *offs, uint8_t out32[32]) {
  sycl::queue &q = *ctx->q;
  const Words8 key = words_of(ctx->jobKey);
  const uint64_t chunks = len / 1024;
  offs->assign(1, 0);
  chunk_cvs(q, key, reinterpret_cast<const uint8_t *>(data), chunks, ctx->compact, tree);
  uint64_t base = 0, count = chunks;
  while (count > 1) {
    const uint64_t pairs = count / 2, next = base + count;
    parent_layer(q, key, tree + base * 8, pairs, pairs == 1, tree + next * 8);
    offs->push_back(next);
    base = next;
    count = pairs;
  }
  q.memcpy(ctx->hRoot, tree + base * 8, 32).wait();
  for (int i = 0; i < 8; i++) pearl_b3::store_le32(out32 + 4 * i, ctx->hRoot[i]);
}

// cert-v3: blake3(root || dim_le32 || 28 zeros, key = salt), or the raw root
// under legacy.
void bind_root(const Ctx *ctx, const uint8_t root[32], uint32_t dim, const uint8_t salt[32],
               uint8_t out[32]) {
  if (ctx->profile.seed_derivation == PEARL_SEED_LEGACY) {
    memcpy(out, root, 32);
    return;
  }
  uint8_t msg[64] = {0};
  memcpy(msg, root, 32);
  pearl_b3::store_le32(msg + 32, dim);
  uint32_t key[8], m[16], h[8];
  for (int i = 0; i < 8; i++) key[i] = pearl_b3::load_le32(salt + 4 * i);
  for (int i = 0; i < 16; i++) m[i] = pearl_b3::load_le32(msg + 4 * i);
  pearl_b3::hash64(key, true, m, h);
  for (int i = 0; i < 8; i++) pearl_b3::store_le32(out + 4 * i, h[i]);
}

// blake3(a || b), unkeyed: both seed links.
void seed_link(const uint8_t a[32], const uint8_t b[32], uint8_t out[32]) {
  uint32_t m[16], h[8];
  for (int i = 0; i < 8; i++) {
    m[i] = pearl_b3::load_le32(a + 4 * i);
    m[8 + i] = pearl_b3::load_le32(b + 4 * i);
  }
  pearl_b3::hash64(nullptr, false, m, h);
  for (int i = 0; i < 8; i++) pearl_b3::store_le32(out + 4 * i, h[i]);
}

Words8 label_words(const char *s) {
  uint8_t b[32] = {0};
  memcpy(b, s, strlen(s));
  return words_of(b);
}

// One side's noise and noised operand: E_AL (or E_BR) dense, E_AR (or E_BL) the
// sparse pairs, then the materialise. A is keyed by a_seed, B by b_seed.
void draw_noise(Ctx *ctx, bool isA) {
  sycl::queue &q = *ctx->q;
  const uint32_t k = ctx->profile.k, rank = ctx->profile.rank;
  const uint32_t rows = isA ? ctx->profile.m : ctx->profile.n;
  const Words8 seed = words_of(isA ? ctx->aSeed : ctx->bSeed);
  const Words8 label = label_words(isA ? "A_tensor" : "B_tensor");
  int8_t *dense = isA ? ctx->dEAL : ctx->dEBR;
  uint32_t *perm = isA ? ctx->dPermA : ctx->dPermB;
  gen_dense(q, seed, label, dense, rows, rank);
  gen_perm(q, seed, label, perm, k, rank);
  uint32_t kLog2 = 0;
  while ((1u << kLog2) < k) kLog2++;
  const uint64_t len = (uint64_t)rows * k;
  const bool constFill = ctx->profile.operand_fill == PEARL_OPERAND_CONST;
  const uint64_t readVecs = constFill ? (PEARL_STAMP_BYTES + 15u) / 16u : len / 16;
  const uint32_t fillWord = (uint32_t)(uint8_t)PEARL_OPERAND_FILL * 0x01010101u;
  Layout lay{kRowMajor, 0};
  if (ctx->frag == 1) lay = isA ? Layout{kFragA, ctx->aStride / 4u} : Layout{kFragB, ctx->bStride / 4u};
  if (ctx->frag == 2) lay = isA ? Layout{kFragAT, ctx->aStride / 4u} : Layout{kFragBT, ctx->bStride / 4u};
  if (ctx->frag == 3) lay = isA ? Layout{kFragAT, ctx->aStride / 4u} : Layout{kFragB, ctx->bStride / 4u};
  materialize16(q, isA ? ctx->dA : ctx->dB, dense, perm, isA ? ctx->dAp : ctx->dBp, rows, kLog2,
                rank, readVecs, fillWord, lay);
}

// The noised operands stored the way `fold` reads them. A context draws them
// for its own fold; only the check tool, which runs other folds on the same
// context, ever changes it. Both are drawn again from the same seeds.
void set_layout(Ctx *ctx, int frag) {
  if (ctx->frag == frag) return;
  ctx->frag = frag;
  if (!ctx->haveJob) return;
  draw_noise(ctx, true);
  draw_noise(ctx, false);
}

void a_seed_from_root(Ctx *ctx) {
  uint8_t bound[32];
  bind_root(ctx, ctx->rootA, ctx->profile.m, PEARL_SEED_SALT_A, bound);
  seed_link(ctx->bSeed, bound, ctx->aSeed);
}

// A job's first draw: both operands, both trees, both seeds, both noises.
void full_draw(Ctx *ctx, uint64_t salt) {
  sycl::queue &q = *ctx->q;
  uint8_t in[PEARL_HEADER_BYTES + PEARL_CONFIG_BYTES];
  memcpy(in, ctx->header, PEARL_HEADER_BYTES);
  pearl_write_config52(&ctx->profile, in + PEARL_HEADER_BYTES);
  pearl_b3::hash_small(nullptr, in, sizeof in, ctx->jobKey);

  const uint64_t aLen = (uint64_t)ctx->profile.m * ctx->profile.k;
  const uint64_t bLen = (uint64_t)ctx->profile.n * ctx->profile.k;
  if (ctx->profile.operand_fill == PEARL_OPERAND_CONST) {
    q.memset(ctx->dA, PEARL_OPERAND_FILL, ctx->compact ? 2048 : aLen);
    q.memset(ctx->dB, PEARL_OPERAND_FILL, ctx->compact ? 2048 : bLen);
    stamp_a(q, ctx->dA, salt);
  } else {
    const Words8 key = words_of(ctx->jobKey);
    gen_operand(q, key, label_words("A_tensor"), ctx->dA, aLen, salt);
    gen_operand(q, key, label_words("B_tensor"), ctx->dB, bLen, salt);
  }
  commitment(ctx, ctx->dA, aLen, ctx->dTreeA, &ctx->layerOffA, ctx->rootA);
  commitment(ctx, ctx->dB, bLen, ctx->dTreeB, &ctx->layerOffB, ctx->rootB);
  uint8_t boundB[32];
  bind_root(ctx, ctx->rootB, ctx->profile.n, PEARL_SEED_SALT_B, boundB);
  seed_link(ctx->jobKey, boundB, ctx->bSeed);
  a_seed_from_root(ctx);
  draw_noise(ctx, true);
  draw_noise(ctx, false);
}

// A same-job redraw: stamp the salt into A, rebuild A's tree and a_seed, noise A
// again. B, its tree, b_seed and B' stay (they do not depend on the salt).
void restamp(Ctx *ctx, uint64_t salt) {
#if defined(PEARL_W_PROF)
  // goal-arc25 (measurement only): each step's wall time, to stderr.
  const bool lg = getenv("PEARL_W_RSLOG") != nullptr;
  auto tq = [&]() {
    ctx->q->wait();
    return std::chrono::steady_clock::now();
  };
  auto ta = lg ? tq() : std::chrono::steady_clock::now();
  stamp_a(*ctx->q, ctx->dA, salt);
  auto tb = lg ? tq() : ta;
  const uint64_t aLen = (uint64_t)ctx->profile.m * ctx->profile.k;
  commitment(ctx, ctx->dA, aLen, ctx->dTreeA, &ctx->layerOffA, ctx->rootA);
  auto tc = lg ? tq() : ta;
  a_seed_from_root(ctx);
  {
    sycl::queue &q = *ctx->q;
    const Words8 seed = words_of(ctx->aSeed);
    const Words8 label = label_words("A_tensor");
    gen_dense(q, seed, label, ctx->dEAL, ctx->profile.m, ctx->profile.rank);
    auto td = lg ? tq() : ta;
    gen_perm(q, seed, label, ctx->dPermA, ctx->profile.k, ctx->profile.rank);
    auto te = lg ? tq() : ta;
    draw_noise(ctx, true);
    auto tf = lg ? tq() : ta;
    if (lg) {
      auto d = [](std::chrono::steady_clock::time_point x, std::chrono::steady_clock::time_point y) {
        return std::chrono::duration<double, std::milli>(y - x).count();
      };
      fprintf(stderr, "rslog stamp %.3f tree %.3f dense %.3f perm %.3f noise(all three again) %.3f\n", d(ta, tb),
              d(tb, tc), d(tc, td), d(td, te), d(te, tf));
    }
  }
#else
  stamp_a(*ctx->q, ctx->dA, salt);
  const uint64_t aLen = (uint64_t)ctx->profile.m * ctx->profile.k;
  commitment(ctx, ctx->dA, aLen, ctx->dTreeA, &ctx->layerOffA, ctx->rootA);
  a_seed_from_root(ctx);
  draw_noise(ctx, true);
#endif
}

// Which 1024-byte chunks hold these rows.
void leaf_indices_for_rows(const uint32_t *rows, uint32_t nrows, uint32_t k,
                           std::vector<uint32_t> *out) {
  std::set<uint32_t> s;
  for (uint32_t i = 0; i < nrows; i++) {
    const uint64_t first = (uint64_t)rows[i] * k / 1024;
    const uint64_t last = ((uint64_t)(rows[i] + 1) * k - 1) / 1024;
    for (uint64_t j = first; j <= last; j++) s.insert((uint32_t)j);
  }
  out->assign(s.begin(), s.end());
}

// One side's share proof: the leaves the rows live in and the sibling digests,
// in the verifier's order (level by level, the live set ascending, a sibling
// only when it is not itself live). Same walk as snapshotProof in pearl_host.cu.
// All reads go on the side queue and are waited for once.
void snapshot_proof(Ctx *ctx, bool isA, const uint32_t *rows, uint32_t nrows, PearlProofSide *out) {
  const uint32_t k = ctx->profile.k;
  const int8_t *operand = isA ? ctx->dA : ctx->dB;
  const uint32_t *tree = isA ? ctx->dTreeA : ctx->dTreeB;
  const std::vector<uint64_t> &offs = isA ? ctx->layerOffA : ctx->layerOffB;
  const uint64_t totalLeaves = (uint64_t)(isA ? ctx->profile.m : ctx->profile.n) * k / 1024;
  sycl::queue &side = *ctx->side;

  leaf_indices_for_rows(rows, nrows, k, &out->leaf_indices);
  out->leaves.resize(out->leaf_indices.size() * 1024);
  for (size_t i = 0; i < out->leaf_indices.size(); i++) {
    const uint64_t li = out->leaf_indices[i];
    const int8_t *src = ctx->compact ? operand + (li == 0 ? 0 : 1024) : operand + li * 1024;
    side.memcpy(out->leaves.data() + i * 1024, src, 1024);
  }

  std::vector<std::pair<uint32_t, uint64_t>> want;  // (level, node index)
  std::vector<uint32_t> current = out->leaf_indices;
  uint64_t levelLen = totalLeaves;
  uint32_t level = 0;
  while (levelLen > 1 && !current.empty() && level + 1 < offs.size()) {
    const std::set<uint32_t> live(current.begin(), current.end());
    for (uint32_t i : current) {
      if (i % 2 == 1) {
        if (live.count(i - 1)) continue;
        want.push_back({level, i - 1});
      } else {
        if (live.count(i + 1) || (uint64_t)i + 1 >= levelLen) continue;
        want.push_back({level, (uint64_t)i + 1});
      }
    }
    std::set<uint32_t> next;
    for (uint32_t i : current) next.insert(i / 2);
    current.assign(next.begin(), next.end());
    levelLen = (levelLen + 1) / 2;
    level++;
  }
  out->siblings.resize(want.size() * PEARL_HASH_BYTES);
  for (size_t i = 0; i < want.size(); i++)
    side.memcpy(out->siblings.data() + i * PEARL_HASH_BYTES,
                tree + (offs[want[i].first] + want[i].second) * 8, PEARL_HASH_BYTES);
  side.wait();
  memcpy(out->root, isA ? ctx->rootA : ctx->rootB, PEARL_HASH_BYTES);
  out->total_leaves = totalLeaves;
}

// Hit `i` of a finished batch as a result, proofs included.
void fill_hit(Ctx *ctx, const Slot &s, const HitBuf *hb, uint32_t i, PearlSearchResult *out) {
  for (int w = 0; w < 8; w++) pearl_b3::store_le32(out->jackpot_hash + 4 * w, hb->hash[i * 8 + w]);
  memcpy(out->a_seed, s.aSeed, PEARL_HASH_BYTES);
  memcpy(out->b_seed, ctx->bSeed, PEARL_HASH_BYTES);
  out->nonce = s.nonceBase + hb->index[i];
  out->salt = s.salt;
  const uint64_t region = out->nonce;
  const uint32_t rowIdx = (uint32_t)(region % ctx->rowsValid);
  const uint32_t colIdx = (uint32_t)((region / ctx->rowsValid) % ctx->colsValid);
  const uint32_t rowOff = pearl_expand_offset(rowIdx, PEARL_ROWS_MASK);
  const uint32_t colOff = pearl_expand_offset(colIdx, PEARL_COLS_MASK);
  uint32_t rows[PEARL_ROWS_COUNT], cols[PEARL_COLS_COUNT];
  for (int j = 0; j < PEARL_ROWS_COUNT; j++) rows[j] = rowOff | PEARL_ROWS_PATTERN[j];
  for (int j = 0; j < PEARL_COLS_COUNT; j++) cols[j] = colOff | PEARL_COLS_PATTERN[j];
  snapshot_proof(ctx, true, rows, PEARL_ROWS_COUNT, &out->proof_a);
  snapshot_proof(ctx, false, cols, PEARL_COLS_COUNT, &out->proof_bt);
  out->proof.resize(PEARL_JACKPOT_BUCKETS * 4);
  for (int j = 0; j < PEARL_JACKPOT_BUCKETS; j++)
    pearl_b3::store_le32(out->proof.data() + 4 * j, hb->transcript[i * 16 + j]);
  out->found = true;
}

// Wait for slot `si` and read its hits, lowest region first (the fold appends
// in no particular order). Returns false on a device error, in s.err.
bool finish_slot(Ctx *ctx, int si) {
  Slot &s = ctx->slots[si];
  if (s.resolved) return s.err.empty();
  s.resolved = true;
  s.hits.clear();
  try {
    s.done.wait_and_throw();
  } catch (const std::exception &e) {
    s.err = std::string("SYCL error during search: ") + e.what();
    return false;
  }
  const std::string ae = take_async(ctx);
  if (!ae.empty()) {
    s.err = "SYCL error during search: " + ae;
    return false;
  }
  const HitBuf *hb = ctx->hHits[si];
  const uint32_t n = hb->count < PEARL_MAX_HITS ? hb->count : PEARL_MAX_HITS;
  std::vector<uint32_t> order(n);
  for (uint32_t i = 0; i < n; i++) order[i] = i;
  std::sort(order.begin(), order.end(),
            [&](uint32_t a, uint32_t b) { return hb->index[a] < hb->index[b]; });
  try {
    s.hits.resize(n);
    for (uint32_t i = 0; i < n; i++) fill_hit(ctx, s, hb, order[i], &s.hits[i]);
  } catch (const std::exception &e) {
    s.hits.clear();
    s.err = std::string("SYCL error reading a share proof: ") + e.what();
    return false;
  }
  return true;
}

// Every queued batch finished and its hits read, before a redraw rewrites the
// operands and trees its proofs come from.
void resolve_pending(Ctx *ctx) {
  for (int i = 0; i < ctx->count; i++) finish_slot(ctx, (ctx->head + i) % kSlots);
}

// A hardware fold is launched only on a device that has the instruction at
// that sub-group size; anywhere else the same name is the reference code. In a
// build whose kernels the driver compiles, the hardware fold's kernel has
// nothing but the instruction, and would not compile on another device.
// xmx8w's column windows a work-group: PEARL_W_CW, or 1 where CW does not divide the batch's column windows
// or the card runs under duty (see launch_fold). PEARL_SYCL_CW=1 or 2 forces it (2 still needs the division).
uint32_t xmx8w_cw(const Ctx *ctx, const FoldArgs &a) {
  if ((a.colBatch / 4u) % (uint32_t)PEARL_W_CW) return 1u;
  const std::string f = env_str("PEARL_SYCL_CW", "");
  if (f == "1") return 1u;
  if (f.empty() && ctx->dutyPct < 100) return 1u;
  return (uint32_t)PEARL_W_CW;
}

#if defined(PEARL_W_SMP)
// goal-arc20 timing: one bindless image of 4,096 by 2,048 texels (128 MB) of random bytes, made once, for
// FoldXmxW's PEARL_W_SMP fragments.
uint64_t smp_image(sycl::queue &q) {
  static uint64_t h = 0;
  if (h) return h;
  namespace sx = sycl::ext::oneapi::experimental;
  sx::image_descriptor desc(sycl::range<2>{4096, 2048}, 4, sycl::image_channel_type::unsigned_int32);
  std::vector<uint32_t> host((size_t)4096 * 2048 * 4);
  uint32_t x = 0x9E3779B9u;
  for (auto &w : host) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    w = x & 0x7F7F7F7Fu;
  }
  sx::image_mem_handle mem = sx::alloc_image_mem(desc, q);
  q.ext_oneapi_copy(host.data(), mem, desc);
  q.wait_and_throw();
  h = sx::create_image(mem, desc, q).raw_handle;
  return h;
}
#endif

void launch_fold(Ctx *ctx, int fold, const FoldArgs &a, uint32_t groups) {
  sycl::queue &q = *ctx->q;
  if (fold == kXmx16 && ctx->hwSg != 16) fold = kXmx16Emu;
  if (fold == kXmx8 && ctx->hwSg != 8) fold = kXmx8Emu;
  if (fold == kXmx8F && ctx->hwSg != 8) fold = kXmx8FEmu;
  if (fold == kXmx8R && ctx->hwSg != 8) fold = kXmx8REmu;
  if (fold == kXmx8W && ctx->hwSg != 8) fold = kXmx8WEmu;
  // The split fold copies one block a sub-group, so it runs only at 8 windows a work-group; else the ring fold.
  if ((fold == kXmx8W || fold == kXmx8WEmu) && a.wpg != 8u) fold = fold == kXmx8W ? kXmx8R : kXmx8REmu;
  // goal-arc13: a work-group takes PEARL_W_RW (8; 16 before goal-arc15) row windows, so RW must divide the band
  // and the row windows.
  if ((fold == kXmx8W || fold == kXmx8WEmu) && ((a.band % (uint32_t)PEARL_W_RW) || ((a.rowsValid / 2u) % (uint32_t)PEARL_W_RW)))
    fold = fold == kXmx8W ? kXmx8R : kXmx8REmu;
  // It adds offsets to the low 32 bits of the operands' addresses only, so each must lie inside one 4 GB window.
  if ((fold == kXmx8W || fold == kXmx8WEmu) && ctx->straddle) fold = fold == kXmx8W ? kXmx8R : kXmx8REmu;
#if defined(PEARL_SYCL_XMX8S)
  if (fold == kXmx8S && ctx->hwSg != 8) fold = kXmx8SEmu;
  // A batch it cannot tile in fours (the check's 4-column batches): xmx8f, which reads the same layouts.
  if ((fold == kXmx8S || fold == kXmx8SEmu) && (((a.rowsValid / 2u) % 4u) || ((a.colBatch / 4u) % 4u)))
    fold = fold == kXmx8S ? kXmx8F : kXmx8FEmu;
  if (fold == kXmx8S) { fold_xmxs<true>(q, a, groups); return; }
  if (fold == kXmx8SEmu) { fold_xmxs<false>(q, a, groups); return; }
#endif
  // goal-arc20 timing builds only: their extra fields (every caller passes a temporary, so this is allowed).
#if defined(PEARL_W_SMP)
  const_cast<FoldArgs &>(a).img = smp_image(q);
#endif
#if defined(PEARL_W_TSC)
  // goal-arc23 measurement: launch PEARL_W_TSCN (default 30) of this process gets each work-group's first start and
  // last end (GPU timestamps) and Xe-core, written to the file PEARL_W_TSCOUT (4 uint64 a work-group).
  static int tscCalls = 0;
  const bool tscDump = ++tscCalls == atoi(env_str("PEARL_W_TSCN", "30").c_str()) && !env_str("PEARL_W_TSCOUT", "").empty();
  {
    static uint64_t *tsc = nullptr;
    const size_t n = 1u << 16;
    if (!tsc) tsc = sycl::malloc_device<uint64_t>(n * 4, q);
    std::vector<uint64_t> init(n * 4, 0);
    for (size_t i = 0; i < n; i++) init[i * 4] = ~0ull;
    q.memcpy(tsc, init.data(), n * 4 * sizeof(uint64_t)).wait();
    const_cast<FoldArgs &>(a).tsc = tsc;
  }
#endif
#if defined(PEARL_W_STAG)
  {
    static uint32_t *tick = nullptr;
    if (!tick) tick = sycl::malloc_device<uint32_t>(32, q);
    q.memset(tick, 0, 32 * sizeof(uint32_t));
    const_cast<FoldArgs &>(a).tick = tick;
  }
#endif
  switch (fold) {
    // VAR 12: the copy loaded a k-step early, no acquire fence at the wait (pearl_sycl_kernels.hpp).
    case kXmx8R: fold_xmxr<true, PEARL_RING_VAR>(q, a, groups); break;
    case kXmx8REmu: fold_xmxr<false, PEARL_RING_VAR>(q, a, groups); break;
    case kXmx8W:
#if defined(PEARL_SYCL_ESIMD) && defined(PEARL_SYCL_XMX_JIT)
      if (env_str("PEARL_SYCL_ESIMD", "") == "1") {
        if (!ctx->dWords) ctx->dWords = sycl::malloc_device<uint32_t>((size_t)ctx->batch * 16u, q);
        fold_xmxe(q, a, groups, ctx->dWords);
        break;
      }
#endif
#if defined(PEARL_W_SEPEPI) && defined(PEARL_SYCL_XMX_JIT)
#if defined(PEARL_W_SEPQ)
      {
        const int si = a.hits == ctx->dHits[1] ? 1 : 0;
        if (!ctx->dWordsS[si]) ctx->dWordsS[si] = sycl::malloc_device<uint32_t>((size_t)ctx->batch * 16u, q);
        fold_xmxw_sep(q, a, groups, ctx->dWordsS[si], ctx->side, &ctx->sepEv);
        ctx->sepEvSet = true;
        break;
      }
#endif
      if (!ctx->dWords) ctx->dWords = sycl::malloc_device<uint32_t>((size_t)ctx->batch * 16u, q);
      fold_xmxw_sep(q, a, groups, ctx->dWords);
      break;
#endif
#if defined(PEARL_SYCL_XMX_JIT)
      // goal-arc10: two column windows a work-group, one after the other, when the batch has an even number
      // of them (PEARL_SYCL_WSEQ=1 runs one): the hash at the end is paid once for twice the work. Card 5:
      // 92.9 TH/s against 92.1.
      // goal-arc15: a work-group spans PEARL_W_CW column windows, SEQ times over, so CW times SEQ must divide
      // the batch's column windows; where CW does not (the check's 4-column batch, one window), CW 1.
      // A card under duty runs CW 1 too. CW 2 saves energy (A' from L1), which only counts on a card at its
      // power limit; a card resting between batches is not, and there CW 1 does more work a clock. bench.js on
      // card 5: duty 75 CW 1 77.0, CW 2 74.5; duty 60 60.7 / 58.7; full speed 98.2 / 98.9.
      if (xmx8w_cw(ctx, a) == 1) {
        if (env_str("PEARL_SYCL_WSEQ", "2") != "1" && (a.colBatch / 4u) % 2u == 0u)
          fold_xmxw<true, 2, PEARL_W_KPS, (PEARL_W_AH != 0), PEARL_W_RW, 1>(q, a, groups);
        else
          fold_xmxw<true, 1, PEARL_W_KPS, (PEARL_W_AH != 0), PEARL_W_RW, 1>(q, a, groups);
        break;
      }
      if (env_str("PEARL_SYCL_WSEQ", "2") == "16" && (a.colBatch / 4u) % (16u * PEARL_W_CW) == 0u) {
        fold_xmxw<true, 16>(q, a, groups);
        break;
      }
      // goal-arc22: PEARL_SYCL_WSEQ 4 and 8 take the set stream too when PEARL_SYCL_SEQP=1 (timing; see benchmark.md).
      if (env_str("PEARL_SYCL_WSEQ", "2") == "8" && (a.colBatch / 4u) % (8u * PEARL_W_CW) == 0u) {
        if (ctx->seqp) fold_xmxw<true, 8, PEARL_W_KPS, (PEARL_W_AH != 0), PEARL_W_RW, PEARL_W_CW, true>(q, a, groups);
        else fold_xmxw<true, 8>(q, a, groups);
        break;
      }
      if (env_str("PEARL_SYCL_WSEQ", "2") == "4" && (a.colBatch / 4u) % (4u * PEARL_W_CW) == 0u) {
        if (ctx->seqp) fold_xmxw<true, 4, PEARL_W_KPS, (PEARL_W_AH != 0), PEARL_W_RW, PEARL_W_CW, true>(q, a, groups);
        else fold_xmxw<true, 4>(q, a, groups);
        break;
      }
      if (env_str("PEARL_SYCL_WSEQ", "2") != "1" && (a.colBatch / 4u) % (2u * PEARL_W_CW) == 0u) {
        // goal-arc21: the set stream (FoldXmxW's kSeqP), opt-in on the newer driver. goal-arc23: seqp 2 is the stream
        // in the forms HiveOS's 24.35 driver compiles right (FoldXmxW's S24), the default there (see Ctx::seqp).
        if (ctx->seqp == 2) fold_xmxw<true, 2, PEARL_W_KPS, (PEARL_W_AH != 0), PEARL_W_RW, PEARL_W_CW, true, true>(q, a, groups);
        else if (ctx->seqp) fold_xmxw<true, 2, PEARL_W_KPS, (PEARL_W_AH != 0), PEARL_W_RW, PEARL_W_CW, true>(q, a, groups);
        else fold_xmxw<true, 2>(q, a, groups);
        break;
      }
#endif
      if (xmx8w_cw(ctx, a) == PEARL_W_CW) fold_xmxw<true>(q, a, groups);
      else fold_xmxw<true, 1, PEARL_W_KPS, (PEARL_W_AH != 0), PEARL_W_RW, 1>(q, a, groups);
      break;
    case kXmx8WEmu:
      if (xmx8w_cw(ctx, a) == PEARL_W_CW) fold_xmxw<false>(q, a, groups);
      else fold_xmxw<false, 1, PEARL_W_KPS, (PEARL_W_AH != 0), PEARL_W_RW, 1>(q, a, groups);
      break;
    case kXmx8F: fold_xmxf<8, 2, true>(q, a, groups); break;
    case kXmx8FEmu: fold_xmxf<8, 2, false>(q, a, groups); break;
    case kXmx16: fold_xmx<16, 2, true>(q, a, groups); break;
    case kXmx8: fold_xmx<8, 1, true>(q, a, groups); break;
    case kXmx16Emu: fold_xmx<16, 2, false>(q, a, groups); break;
    case kXmx8Emu: fold_xmx<8, 1, false>(q, a, groups); break;
    default: fold_dot(q, a, groups); break;
  }
#if defined(PEARL_W_TSC)
  if (tscDump) {
    std::vector<uint64_t> h((size_t)1u << 18);
    q.memcpy(h.data(), a.tsc, h.size() * sizeof(uint64_t)).wait();
    if (FILE *f = fopen(env_str("PEARL_W_TSCOUT", "").c_str(), "wb")) {
      fwrite(h.data(), sizeof(uint64_t), h.size(), f);
      fclose(f);
    }
  }
#endif
}

FoldArgs fold_args(Ctx *ctx, uint64_t nonce_base, int slot, uint32_t *allTr) {
  FoldArgs a;
  a.Ap = ctx->dAp;
  a.Bp = ctx->dBp;
  a.k = ctx->profile.k;
  a.rowsValid = ctx->rowsValid;
  a.colStart = (uint32_t)((nonce_base / ctx->rowsValid) % ctx->colsValid);
  a.colBatch = ctx->colBatch;
  a.band = ctx->band;
  a.aStride = ctx->aStride;
  a.bStride = ctx->bStride;
  a.zero = 0;  // always: the fragment fold ORs (a loaded word & zero) into an operand to order its loads
  a.wpg = ctx->ringWpg;
  for (int i = 0; i < 8; i++) a.test.key[i] = pearl_b3::load_le32(ctx->aSeed + 4 * i);
  for (int i = 0; i < 8; i++) {
    const uint8_t *t = ctx->target + i * 4;
    a.test.target_w[i] = ((uint32_t)t[0] << 24) | ((uint32_t)t[1] << 16) | ((uint32_t)t[2] << 8) | t[3];
  }
  a.test.hash_big_endian = (int)ctx->profile.hash_big_endian;
  a.hits = ctx->dHits[slot];
  a.allTr = allTr;
  return a;
}

uint32_t fold_groups(const Ctx *ctx) { return (ctx->rowsValid / 2u) * (ctx->colBatch / 4u); }

}  // namespace

extern "C" int pearl_host_select_device(const PearlProfile *profile, int requested, char *name,
                                        size_t name_len, char *err, size_t err_len) {
  if (name && name_len) name[0] = 0;
  g_selected = -1;
  if (!profile) { fail(err, err_len, "no profile supplied"); return -1; }
  const std::vector<sycl::device> devs = list_devices();
  if (devs.empty()) {
    fail(err, err_len, env_str("PEARL_SYCL_DEVICE", "gpu") == "cpu"
                           ? "no SYCL CPU device found (is the OpenCL CPU runtime installed?)"
                           : "no Intel GPU found through Level Zero or OpenCL");
    return -1;
  }
  int chosen = -1;
  if (requested >= 0) {
    if (requested >= (int)devs.size()) {
      fail(err, err_len, "GPU " + std::to_string(requested) + " was asked for but SYCL lists "
                             + std::to_string(devs.size()) + ": valid indices are 0.."
                             + std::to_string(devs.size() - 1));
      return -1;
    }
    chosen = requested;
  } else {
    // Compute units x clock, skipping cards too small for the profile; ties
    // keep the lower index. A proxy, but it picks the big card over the iGPU.
    double best = -1.0;
    const size_t need = needed_bytes(profile);
    for (size_t d = 0; d < devs.size(); d++) {
      try {
        if (devs[d].get_info<sycl::info::device::global_mem_size>() < need) continue;
        const double score = (double)devs[d].get_info<sycl::info::device::max_compute_units>()
                             * (double)std::max(1u, devs[d].get_info<sycl::info::device::max_clock_frequency>());
        if (score > best) { best = score; chosen = (int)d; }
      } catch (const std::exception &) {
      }
    }
    if (chosen < 0) chosen = 0;  // create's check then says why, with numbers
  }
  g_selected = chosen;
  if (name && name_len) {
    try {
      snprintf(name, name_len, "%s", devs[chosen].get_info<sycl::info::device::name>().c_str());
    } catch (const std::exception &) {
    }
  }
  return chosen;
}

// The k a card mines when the miner lets the core choose (pearl_core.cc's cardK).
// Both folds are built for one k (PEARL_FOLD_K, which pearl_host_create_checked
// checks), so a card mines the profile's, as on the CUDA host. No Intel GPU has
// been timed at another k, so there is no larger one to prefer.
extern "C" uint32_t pearl_host_card_k(const PearlProfile *profile, int, char *note, size_t note_len) {
  if (note && note_len) note[0] = '\0';
  return profile ? profile->k : 0;
}

// SYCL queues are bound to their device, so there is nothing to do per thread.
extern "C" void pearl_host_bind_thread(void *) {}

extern "C" void pearl_host_destroy(void *handle) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx) return;
  free_all(ctx);
  delete ctx;
}

// The create pearl_core.cc calls, with whether the opener recomputes every hit
// before it sends it (check_hits: the app's PearlMiner, verify-hits.js). Such an
// opener gets a GPU, and *untested tells it to check that card's hits.
// pearl_host_create is this with neither.
extern "C" void *pearl_host_create_checked(const PearlProfile *profile, bool check_hits, bool *untested,
                                           char *err, size_t err_len) {
  if (untested) *untested = false;
  if (!profile) { fail(err, err_len, "no profile supplied"); return nullptr; }
  const uint32_t k = profile->k, rank = profile->rank;
  if (k != PEARL_FOLD_K || rank != PEARL_FOLD_RANK) {
    fail(err, err_len, "the SYCL fold is built for k " + std::to_string(PEARL_FOLD_K) + ", rank "
                           + std::to_string(PEARL_FOLD_RANK) + " (got k " + std::to_string(k)
                           + ", rank " + std::to_string(rank) + ")");
    return nullptr;
  }
  const uint64_t aChunks = (uint64_t)profile->m * k / 1024, bChunks = (uint64_t)profile->n * k / 1024;
  if (aChunks < 2 || bChunks < 2 || (aChunks & (aChunks - 1)) || (bChunks & (bChunks - 1))) {
    fail(err, err_len, "m*k/1024 and n*k/1024 must each be a power of two, at least 2 (got "
                           + std::to_string(aChunks) + " and " + std::to_string(bChunks) + ")");
    return nullptr;
  }
  // Whole windows: 2 valid row offsets (32 rows) by 4 valid column offsets (64 columns).
  if (profile->m % 32u || profile->n % 64u) {
    fail(err, err_len, "m must be a multiple of 32 and n of 64");
    return nullptr;
  }
  const std::vector<sycl::device> devs = list_devices();
  int idx = g_selected;
  if (idx < 0 || idx >= (int)devs.size()) idx = 0;
  if (devs.empty()) { fail(err, err_len, "no SYCL device"); return nullptr; }

  Ctx *ctx = new Ctx();
  ctx->profile = *profile;
  ctx->dev = devs[idx];
  ctx->deviceIndex = idx;
  ctx->isGpu = ctx->dev.is_gpu();
  try {
    ctx->deviceName = ctx->dev.get_info<sycl::info::device::name>();
    const size_t need = needed_bytes(profile);
    const size_t have = ctx->dev.get_info<sycl::info::device::global_mem_size>();
    ctx->memTotal = have;
    if (have < need) {
      fail(err, err_len, "not enough device memory for this profile: need ~"
                             + std::to_string(need >> 20) + " MiB, the device has "
                             + std::to_string(have >> 20) + " MiB");
      delete ctx;
      return nullptr;
    }
    auto handler = [ctx](sycl::exception_list el) { record_async(ctx, el); };
    sycl::context sctx(ctx->dev, handler);
#if defined(PEARL_W_PROF)
    // goal-arc23 (measurement only): profiling on, so the check bench can time the fold on the GPU itself.
    ctx->q = new sycl::queue(sctx, ctx->dev, handler, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
#else
    ctx->q = new sycl::queue(sctx, ctx->dev, handler, sycl::property::queue::in_order());
#endif
    ctx->side = new sycl::queue(sctx, ctx->dev, handler, sycl::property::queue::in_order());
  } catch (const std::exception &e) {
    fail(err, err_len, std::string("could not open the SYCL device: ") + e.what());
    pearl_host_destroy(ctx);  // not a bare delete: the first queue may exist
    return nullptr;
  }
  // A build carries kernels only for its own targets (build.sh). On any other
  // device the first kernel would fail with "No kernel named ... was found";
  // say what to do instead.
  bool haveImage = true;
  try {
    haveImage = sycl::has_kernel_bundle<sycl::bundle_state::executable>(ctx->q->get_context(), {ctx->dev});
  } catch (const std::exception &) {
  }
  if (!haveImage) {
    fail(err, err_len, "this build has no kernels for " + ctx->deviceName
                           + ": rebuild with ./build.sh --jit, or with --targets naming this GPU"
                             " (README.md, \"Which build\")");
    pearl_host_destroy(ctx);
    return nullptr;
  }
  if (!pick_fold(ctx, err, err_len)) {
    pearl_host_destroy(ctx);
    return nullptr;
  }
  // No Intel GPU has mined with these folds in a release. They were written
  // against the SYCL CPU device, where DPAS is the specification's reference
  // code, and have run on one kind of card since: Arc A750, where the hardware
  // fold was right and that reference code was not (README.md, "Results on an
  // Arc A750 rig"). If a fold is wrong on a card, every hash is wrong yet meets
  // the target, so every share is sent and rejected, and a pool can ban the
  // address for that. So a GPU mines only with every hit checked: its opener
  // recomputes each hit in JS before sending it (checkHits, pearlMiner.js),
  // and ctx->untested tells it to. PEARL_SYCL_UNTESTED=1 does not change that.
  // It only opens a GPU for an opener that checks nothing, which is for
  // bench.js: it sends no shares. A kind of card leaves this gate by a change
  // here, once it has passed README.md's "Test plan" and the founder says so.
  // The CPU device is not gated: nobody mines on it.
  const char *unchecked = getenv("PEARL_SYCL_UNTESTED");
  if (ctx->isGpu) {
    if (check_hits) {
      ctx->untested = true;
    } else if (!(unchecked && !strcmp(unchecked, "1"))) {
      fail(err, err_len,
           "no Intel GPU has mined with this core in a release yet, so its shares could be rejected: it "
           "mines only with every hit checked, as the app and verify-hits.js do. For an opener that sends "
           "no shares, set PEARL_SYCL_UNTESTED=1 (earn/native/sycl/README.md)");
      pearl_host_destroy(ctx);
      return nullptr;
    }
  }

  sycl::queue &q = *ctx->q;
  ctx->compact = profile->operand_fill == PEARL_OPERAND_CONST;
  // goal-arc23: PEARL_SYCL_SEQP 0 / 1 / 2 forces the set stream off / on / in 24.35's forms. Unset, a 24.x OpenCL
  // driver (HiveOS: 24.35, where it is +1.0%, 94.5-94.7 against 93.6 on card 5) gets 2, every other driver 0 (the
  // newer one runs 2 slower than the stream off, and 1 is still opt-in).
  {
    const std::string s = env_str("PEARL_SYCL_SEQP", "");
    if (s == "0" || s == "1" || s == "2") {
      ctx->seqp = s[0] - '0';
    } else {
      std::string drv;
      try {
        if (ctx->dev.get_backend() == sycl::backend::opencl) drv = ctx->dev.get_info<sycl::info::device::driver_version>();
      } catch (const std::exception &) {
      }
      ctx->seqp = drv.rfind("24.", 0) == 0 ? 2 : 0;
    }
  }
  const size_t aBytes = (size_t)profile->m * k, bBytes = (size_t)profile->n * k;
  bool ok = true;
  // `window`: the noised operands, each inside one 4 GB-aligned window if five tries allow (the split fold
  // needs it). An allocation that crosses a boundary is held while the next is made, so the next lands
  // elsewhere; then the held ones are freed.
  auto crosses = [](void *p, size_t n) { return ((uintptr_t)p >> 32) != (((uintptr_t)p + n - 1) >> 32); };
  auto dev_alloc = [&](size_t n, bool window = false) -> void * {
    std::vector<void *> held;
    void *p = nullptr;
    for (int i = 0; i < (window ? 5 : 1); i++) {
      if (p) held.push_back(p);
      p = sycl::malloc_device(n, q);
      if (!p || !crosses(p, n)) break;
    }
    for (void *h : held) sycl::free(h, q);
    if (!p) ok = false;
    else ctx->memUsed += n;
    if (window && p && crosses(p, n)) ctx->straddle = true;
    return p;
  };
  ctx->dA = (int8_t *)dev_alloc(ctx->compact ? 2048 : aBytes);
  ctx->dB = (int8_t *)dev_alloc(ctx->compact ? 2048 : bBytes);
  ctx->dAp = (int8_t *)dev_alloc(padded_bytes(profile->m, k, true), true);
  ctx->dBp = (int8_t *)dev_alloc(padded_bytes(profile->n, k, false), true);
  if (ctx->straddle)
    fprintf(stderr, "[pearl-sycl] an operand crosses a 4 GB boundary after 5 tries: xmx8w runs as xmx8r\n");
  ctx->dEAL = (int8_t *)dev_alloc((size_t)profile->m * rank);
  ctx->dEBR = (int8_t *)dev_alloc((size_t)profile->n * rank);
  ctx->dPermA = (uint32_t *)dev_alloc((size_t)k * 2 * sizeof(uint32_t));
  ctx->dPermB = (uint32_t *)dev_alloc((size_t)k * 2 * sizeof(uint32_t));
  ctx->dTreeA = (uint32_t *)dev_alloc(2 * aChunks * 32);
  ctx->dTreeB = (uint32_t *)dev_alloc(2 * bChunks * 32);
  for (int s = 0; s < kSlots; s++) {
    ctx->dHits[s] = (HitBuf *)dev_alloc(sizeof(HitBuf));
    ctx->hHits[s] = (HitBuf *)sycl::malloc_host(sizeof(HitBuf), q);
    if (!ctx->hHits[s]) ok = false;
  }
  ctx->hRoot = (uint32_t *)sycl::malloc_host(32, q);
  if (!ctx->hRoot) ok = false;
  if (!ok) {
    fail(err, err_len, "could not allocate the core's device memory (need ~"
                           + std::to_string(needed_bytes(profile) >> 20) + " MiB)");
    pearl_host_destroy(ctx);
    return nullptr;
  }

  ctx->frag = fold_layout(ctx->fold);
  ctx->aStride = 16u * k + operand_pad(true);
  ctx->bStride = 64u * k + operand_pad(false);
  ctx->rowsValid = profile->m / PEARL_ROWS_COUNT;
  ctx->colsValid = profile->n / PEARL_COLS_COUNT;
  {
    // One figure for every card, or a list by card index ("100,60,60"; a card past its end takes the last).
    std::string dEnv = env_str("PEARL_SYCL_DUTY", "");
    for (int i = 0; i < ctx->deviceIndex; i++) {
      const size_t comma = dEnv.find(',');
      if (comma == std::string::npos) break;
      dEnv = dEnv.substr(comma + 1);
    }
    if (!dEnv.empty()) {
      const int d = atoi(dEnv.c_str());
      ctx->dutyPct = d < 10 ? 10 : d > 100 ? 100 : d;
    }
  }
  // The batch width: the profile's, at most kMaxColBatch, or PEARL_SYCL_COL_BATCH;
  // as a power of two of at least one window's 4 column offsets and at most the
  // valid offsets (a power of two too, since n is), so a salt splits into whole
  // batches. The cap keeps a batch short on a slow fold: at mainnet 256 column
  // offsets are 2^21 regions, 1.1e12 MACs, 0.2 s at 5 TH/s, where the profile's
  // 2048 would be 1.8 s, near the GPU drivers' job timeouts. The B' one batch
  // sweeps is 256 x 16 columns of 2048 bytes, 8 MB: inside the L2 of the A770,
  // A750 and B580 (16-18 MB), but not of the A380 (6 MB), where bench.sh
  // --sweep's narrower batches may do better.
  // goal-arc25: xmx8w under duty takes 1,024 column offsets, walked as chunks of 256 (FoldXmxW, kChunkWins) so
  // its B' in L2 stays 8 MB: ~42 ms a batch at mainnet on an A750. Under duty every batch waits, rests and
  // launches again, and the card's clock comes back up after each rest: a fixed cost a launch, which four
  // times fewer launches cut. bench.js, card 4, stream on: duty 75 79.7-79.8 TH/s against 78.2 (172 W against
  // 176), duty 60 64.0-64.1 against 61.4-61.6 (same watts). At full speed the batches queue back to back and
  // it was worth only 0.35% (103.3 / 102.9, bench.js), against up to four times as much work searched for an
  // old job after a job switch, so a card at full speed keeps 256.
  // Only where the duty path's CW 1 instance runs, the one that walks chunks (xmx8w_cw).
  const std::string cwEnv = env_str("PEARL_SYCL_CW", "");
  const bool wide = ctx->fold == kXmx8W && ctx->dutyPct < 100 && (cwEnv.empty() || cwEnv == "1");
  const uint64_t kMaxColBatch = wide ? 1024 : 256;
  uint64_t cb = profile->col_batch ? profile->col_batch : 4u;
  if (cb > kMaxColBatch) cb = kMaxColBatch;
  const std::string cbEnv = env_str("PEARL_SYCL_COL_BATCH", "");
  if (!cbEnv.empty()) cb = strtoull(cbEnv.c_str(), nullptr, 10);
  if (cb > ctx->colsValid) cb = ctx->colsValid;
  if (cb < 4) cb = 4;
  ctx->colBatch = floor_pow2(cb);
  // The plain fold launches 32 work-items a region, and a SYCL launch range must
  // fit in an int, so a batch is at most 2^25 regions (4096 column offsets at
  // mainnet). Only a large PEARL_SYCL_COL_BATCH gets here.
  while ((uint64_t)ctx->colBatch * ctx->rowsValid > (1ull << 25) && ctx->colBatch > 4) ctx->colBatch /= 2;
  ctx->batch = ctx->colBatch * ctx->rowsValid;
  // Row windows a band: for fold_xmx as many as make a band's A' (32 rows of k
  // bytes a window) the size of the device's last-level cache, for the plain
  // fold and the fragment fold a quarter of that. PEARL_SYCL_BAND forces it.
  // Measured on an Arc A750, which reports 16 MB, so 256 and 64 (TH/s). xmx8,
  // bench: 4 -> 14.1, 16 -> 14.3, 64 -> 15.2 to 15.5, 128 -> 16.3, 256 -> 16.6
  // to 16.9, 512 -> 13.8, 1024 to 4096 -> 13.4 to 13.5. dot, 400 hits through
  // the addon: 64 -> 3.91, 256 -> 3.74. A quarter was the rule for both, and
  // not measured, until that run. xmx8f, bench, 11 November, with B' loaded
  // in halves: 16 -> 55.8, 32 -> 60.6, 64 -> 64.2, 128 -> 44.1 (with one load
  // a fragment it was 42.1 to 42.2 at 16, 32 and 64, 31.7 at 128, 35.4 at 256).
  {
    uint64_t cache = 0;
    try {
      cache = ctx->dev.get_info<sycl::info::device::global_mem_cache_size>();
    } catch (const std::exception &) {
    }
    // xmx8w takes an eighth (32 row windows on an A750): with its 256-byte A' and B' messages (goal-arc13)
    // 98.5-98.6 TH/s against 98.0-98.1 at 64 and 98.0-98.3 at 16 (card 5, three rounds).
    const uint64_t share = ctx->fold == kXmx8W ? 8 : ctx->fold == kDot || fold_is_frag(ctx->fold) ? 4 : 1;
    uint64_t band = cache ? cache / share / (32ull * k) : 16;
    const std::string bEnv = env_str("PEARL_SYCL_BAND", "");
    if (!bEnv.empty()) band = strtoull(bEnv.c_str(), nullptr, 10);
    if (band < 1) band = 1;
    if (band > ctx->rowsValid / 2u) band = ctx->rowsValid / 2u;
    ctx->band = floor_pow2(band);
    // xmx8r: windows a work-group, all of one column window. The walk goes down a band's rows first, so a
    // number that divides the band and the row windows keeps every work-group inside one column window.
    // 8 (16 sub-groups) was the best of 4, 8 and 16 for the real kernel on an A750 (README.md, "The ring
    // fold"). PEARL_SYCL_RING_WPG forces another.
    const uint32_t rowWins = ctx->rowsValid / 2u;
    uint32_t w = (uint32_t)strtoul(env_str("PEARL_SYCL_RING_WPG", "8").c_str(), nullptr, 10);
    if (w < 1 || w > 32) w = 8;
    w = floor_pow2(w);
    while (w > 1 && (ctx->band % w || rowWins % w)) w /= 2;
    ctx->ringWpg = w;
  }
  // What pearl_host_fold_name says, for core.device.fold and the logs. Written
  // here, once: the search thread and the JS thread both read it. xmx16 and
  // xmx8 say whether they got the hardware instruction or the reference code.
  bool namedXmx = ctx->fold == kXmx16 || ctx->fold == kXmx8 || ctx->fold == kXmx8F || ctx->fold == kXmx8R
                  || ctx->fold == kXmx8W;
#if defined(PEARL_SYCL_XMX8S)
  namedXmx = namedXmx || ctx->fold == kXmx8S;
#endif
  ctx->foldName = std::string(fold_label(ctx->fold))
                  + (!namedXmx ? "" : ctx->xmxHw ? ", hardware DPAS" : ", DPAS reference code")
                  + ", col_batch " + std::to_string(ctx->colBatch) + ", band " + std::to_string(ctx->band)
                  + " row windows, k " + std::to_string(k) + (ctx->isGpu ? "" : ", SYCL CPU device")
                  + (ctx->dutyPct < 100 ? ", duty " + std::to_string(ctx->dutyPct) + "%" : "");
  if (getenv("PEARL_SYCL_VERBOSE"))
    fprintf(stderr, "[pearl-sycl] %s: fold %s%s, col_batch %u, band %u row windows\n",
            ctx->deviceName.c_str(), fold_label(ctx->fold), ctx->xmxHw ? " (hardware)" : "",
            ctx->colBatch, ctx->band);
  if (untested) *untested = ctx->untested;
  return ctx;
}

// For an opener that does not check its hits. A GPU is refused, unless
// PEARL_SYCL_UNTESTED=1 (see the gate in pearl_host_create_checked).
extern "C" void *pearl_host_create(const PearlProfile *profile, char *err, size_t err_len) {
  return pearl_host_create_checked(profile, false, nullptr, err, err_len);
}

extern "C" void pearl_host_reseed(void *handle, uint64_t salt) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx) return;
  resolve_pending(ctx);
  ctx->salt = salt;
  try {
    if (ctx->baseDrawn) restamp(ctx, salt);
    else full_draw(ctx, salt);
  } catch (const std::exception &e) {
    std::lock_guard<std::mutex> lock(ctx->errMu);
    if (ctx->asyncErr.empty()) ctx->asyncErr = std::string("operand draw: ") + e.what();
  }
  ctx->baseDrawn = true;
  ctx->haveJob = true;
}

extern "C" void pearl_host_set_job_salted(void *handle, const uint8_t *header, const uint8_t *target,
                                          uint64_t salt) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx) return;
  resolve_pending(ctx);
  memcpy(ctx->header, header, PEARL_HEADER_BYTES);
  memcpy(ctx->target, target, PEARL_HASH_BYTES);
  ctx->baseDrawn = false;
  pearl_host_reseed(handle, salt);
}

extern "C" void pearl_host_set_job(void *handle, const uint8_t *header, const uint8_t *target) {
  pearl_host_set_job_salted(handle, header, target, 0);
}

extern "C" bool pearl_host_submit(void *handle, uint64_t nonce_base, uint32_t batch,
                                  uint64_t *regions_out, char *err, size_t err_len) {
  (void)batch;  // a batch is always the context's own width, as in pearl_host.cu
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (regions_out) *regions_out = 0;
  if (!ctx || !ctx->haveJob) return false;
  if (ctx->count >= kSlots) return fail(err, err_len, "search pipeline full: collect a batch first");
  {
    const std::string ae = take_async(ctx);
    if (!ae.empty()) return fail(err, err_len, "SYCL error: " + ae);
  }
  const int si = (ctx->head + ctx->count) % kSlots;
  Slot &s = ctx->slots[si];
  try {
    sycl::queue &q = *ctx->q;
    set_layout(ctx, fold_layout(ctx->fold));  // the check tool may have left the other one
    if (ctx->dutyPct < 100) {
      // Let the card finish what it has, then rest it in proportion to the time that took (2 s at most, so
      // a wait for a job is not paid for twice).
      q.wait();
#if defined(PEARL_W_SEPQ) && defined(PEARL_SYCL_XMX_JIT)
      ctx->side->wait();
#endif
      const auto now = std::chrono::steady_clock::now();
      if (ctx->dutyStarted) {
        auto rest = (now - ctx->dutyMark) * (100 - ctx->dutyPct) / ctx->dutyPct;
        if (rest > std::chrono::seconds(2)) rest = std::chrono::seconds(2);
        std::this_thread::sleep_for(rest);
      }
      ctx->dutyMark = std::chrono::steady_clock::now();
      ctx->dutyStarted = true;
    }
    q.memset(&ctx->dHits[si]->count, 0, sizeof(uint32_t));
    launch_fold(ctx, ctx->fold, fold_args(ctx, nonce_base, si, nullptr), fold_groups(ctx));
#if defined(PEARL_W_SEPQ) && defined(PEARL_SYCL_XMX_JIT)
    if (ctx->sepEvSet) {
      ctx->sepEvSet = false;
      const sycl::event he = ctx->sepEv;
      s.done = ctx->side->submit([&](sycl::handler &h) {
        h.depends_on(he);
        h.memcpy(ctx->hHits[si], ctx->dHits[si], sizeof(HitBuf));
      });
    } else
#endif
    s.done = q.memcpy(ctx->hHits[si], ctx->dHits[si], sizeof(HitBuf));
  } catch (const std::exception &e) {
    return fail(err, err_len, std::string("SYCL error queuing a batch: ") + e.what());
  }
  s.busy = true;
  s.resolved = false;
  s.err.clear();
  s.hits.clear();
  s.nonceBase = nonce_base;
  s.salt = ctx->salt;
  memcpy(s.aSeed, ctx->aSeed, PEARL_HASH_BYTES);
  s.regions = ctx->batch;
  ctx->count++;
  if (regions_out) *regions_out = ctx->batch;
  return true;
}

extern "C" int pearl_host_pending(void *handle) {
  const Ctx *ctx = static_cast<const Ctx *>(handle);
  return ctx ? ctx->count : 0;
}

extern "C" bool pearl_host_collect(void *handle, PearlSearchResult *out, uint64_t *attempts, char *err,
                                   size_t err_len) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (attempts) *attempts = 0;
  if (!ctx || !out || ctx->count == 0) return false;
  const int si = ctx->head;
  Slot &s = ctx->slots[si];
  const bool ok = finish_slot(ctx, si);
  ctx->head = (ctx->head + 1) % kSlots;
  ctx->count--;
  s.busy = false;
  ctx->extra.clear();
  ctx->extraNext = 0;
  if (!ok) {
    fail(err, err_len, s.err.empty() ? "SYCL error during search" : s.err);
    return false;
  }
  if (attempts) *attempts = s.regions;
  if (s.hits.empty()) {
    out->found = false;
    return false;
  }
  *out = s.hits[0];
  ctx->extra.assign(s.hits.begin() + 1, s.hits.end());
  s.hits.clear();
  return true;
}

extern "C" bool pearl_host_next_hit(void *handle, PearlSearchResult *out) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx || !out || ctx->extraNext >= ctx->extra.size()) return false;
  *out = ctx->extra[ctx->extraNext++];
  return true;
}

extern "C" bool pearl_host_search(void *handle, uint64_t nonce_base, uint32_t batch,
                                  PearlSearchResult *out, uint64_t *attempts, char *err,
                                  size_t err_len) {
  if (attempts) *attempts = 0;
  if (!handle || !out) return false;
  Ctx *ctx = static_cast<Ctx *>(handle);
  while (ctx->count > 0) {
    PearlSearchResult drop;
    uint64_t a = 0;
    pearl_host_collect(handle, &drop, &a, nullptr, 0);
  }
  if (!pearl_host_submit(handle, nonce_base, batch, nullptr, err, err_len)) return false;
  return pearl_host_collect(handle, out, attempts, err, err_len);
}

// Which fold this context runs, in words, for core.device.fold and the logs
// ("unresolved" without a context: pearl_core.cc asks after stop() too).
extern "C" const char *pearl_host_fold_name(void *handle) {
  const Ctx *ctx = static_cast<const Ctx *>(handle);
  return ctx && !ctx->foldName.empty() ? ctx->foldName.c_str() : "unresolved";
}

// The device's memory and what this context allocated on it, in bytes, for
// core.device (pearl_core.cc, PEARL_HOST_MEM_INFO). Both were written at
// create, so the JS thread may read them while the search thread runs. False
// without a context: pearl_core.cc asks after stop() too.
extern "C" bool pearl_host_mem_info(void *handle, uint64_t *total_bytes, uint64_t *used_bytes) {
  const Ctx *ctx = static_cast<const Ctx *>(handle);
  if (!ctx || !ctx->memTotal) return false;
  if (total_bytes) *total_bytes = ctx->memTotal;
  if (used_bytes) *used_bytes = ctx->memUsed;
  return true;
}

// RTX 30's launch shapes (pearl_host.cu): this host has one way to launch a
// fold, so there is nothing to choose, pearl_core.cc runs no startup trial and
// the app caches no pick. The batch width and band come from the profile, the
// device's cache size, or PEARL_SYCL_COL_BATCH and PEARL_SYCL_BAND.
extern "C" int pearl_host_shape_count(void *) { return 0; }
extern "C" void pearl_host_shape_name(void *, int, char *buf, size_t len) {
  if (buf && len) snprintf(buf, len, "none");
}
extern "C" bool pearl_host_set_shape(void *, int) { return false; }

// ---------------------------------------------------------------------------
// For the check and bench tool (pearl_sycl_check.cpp), not for the addon.
// ---------------------------------------------------------------------------

// Every device this core may open, one line each: its index (the number
// PEARL_GPU_INDEX and deviceIndex take), its name, the backend it comes through
// and the fold `auto` gives it in this build. Returns how many there are. On a
// rig this is how to tell which index is which card, in the core's own order.
extern "C" int pearl_sycl_list_devices(char *buf, size_t len) {
  const std::vector<sycl::device> devs = list_devices();
  std::string out;
  for (size_t i = 0; i < devs.size(); i++) {
    std::string name = "?", via = "?";
    try {
      name = devs[i].get_info<sycl::info::device::name>();
      const sycl::backend be = devs[i].get_backend();
      via = be == sycl::backend::ext_oneapi_level_zero ? "level_zero" : be == sycl::backend::opencl ? "opencl" : "other";
    } catch (const std::exception &) {
    }
    const int sg = xmx_sg(devs[i]);
    out += std::to_string(i) + ": " + name + " [" + via + "], auto fold "
           + (sg == 16 ? "xmx16 (hardware DPAS)" : sg == 8 ? "xmx8w (hardware DPAS)" : "dot") + "\n";
  }
  if (buf && len) snprintf(buf, len, "%s", out.c_str());
  return (int)devs.size();
}

// The fold this context runs, and its label.
extern "C" const char *pearl_sycl_fold_name(void *handle) {
  const Ctx *ctx = static_cast<const Ctx *>(handle);
  return ctx ? fold_label(ctx->fold) : "none";
}

// The fold as a Fold value (0 dot, 1 xmx16, 2 xmx8, 3 xmx16-emu, 4 xmx8-emu,
// 5 xmx8f, 6 xmx8f-emu).
extern "C" int pearl_sycl_fold_code(void *handle) {
  const Ctx *ctx = static_cast<const Ctx *>(handle);
  return ctx ? ctx->fold : 0;
}

extern "C" int pearl_sycl_fold_is_hw(void *handle) {
  const Ctx *ctx = static_cast<const Ctx *>(handle);
  return ctx && ctx->xmxHw ? 1 : 0;
}

extern "C" const char *pearl_sycl_device_name(void *handle) {
  const Ctx *ctx = static_cast<const Ctx *>(handle);
  return ctx ? ctx->deviceName.c_str() : "";
}

extern "C" uint32_t pearl_sycl_batch_regions(void *handle) {
  const Ctx *ctx = static_cast<const Ctx *>(handle);
  return ctx ? ctx->batch : 0;
}

// One batch at nonce_base with fold `fold` (an Fold value, or -1 for the
// context's own), synchronously. With `transcripts` (batch regions x 16 words)
// every region's transcript is written there; `hits` gets the batch's hit count.
// `ms` the wall time. Returns false with a message on an error.
extern "C" bool pearl_sycl_run_batch(void *handle, uint64_t nonce_base, int fold, uint32_t *transcripts,
                                     uint32_t *hits, double *ms, char *err, size_t err_len) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx || !ctx->haveJob) return fail(err, err_len, "no job");
  resolve_pending(ctx);
  uint32_t *dTr = nullptr;
  try {
    sycl::queue &q = *ctx->q;
    if (transcripts) {
      dTr = sycl::malloc_device<uint32_t>((size_t)ctx->batch * 16, q);
      if (!dTr) return fail(err, err_len, "could not allocate the transcript buffer");
    }
    q.memset(&ctx->dHits[0]->count, 0, sizeof(uint32_t)).wait();
    set_layout(ctx, fold_layout(fold < 0 ? ctx->fold : fold));
    q.wait_and_throw();
    const auto t0 = std::chrono::steady_clock::now();
#if defined(PEARL_W_PROF)
    const sycl::event pe0 = q.ext_oneapi_submit_barrier();
#endif
    launch_fold(ctx, fold < 0 ? ctx->fold : fold, fold_args(ctx, nonce_base, 0, dTr), fold_groups(ctx));
#if defined(PEARL_W_PROF)
    const sycl::event pe1 = q.ext_oneapi_submit_barrier();
#endif
    q.wait_and_throw();
#if defined(PEARL_W_SEPQ) && defined(PEARL_SYCL_XMX_JIT)
    ctx->sepEvSet = false;
    ctx->side->wait_and_throw();
#endif
    if (ms) *ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
#if defined(PEARL_W_PROF)
    // goal-arc25 (measurement only): each batch's wall time, GPU span and the gap from the barrier's end to the
    // fold's start, to stderr.
    if (getenv("PEARL_W_BLOG")) {
      const uint64_t b0 = pe0.get_profiling_info<sycl::info::event_profiling::command_end>();
      const uint64_t b1 = pe1.get_profiling_info<sycl::info::event_profiling::command_end>();
      const uint64_t s1 = pe1.get_profiling_info<sycl::info::event_profiling::command_submit>();
      fprintf(stderr, "blog %.4f %.4f %.4f\n",
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
              (double)(b1 - b0) / 1e6, (double)(b1 > s1 ? b1 - s1 : 0) / 1e6);
    }
    if (ms && getenv("PEARL_W_PROFGPU"))
      *ms = (double)(pe1.get_profiling_info<sycl::info::event_profiling::command_end>()
                     - pe0.get_profiling_info<sycl::info::event_profiling::command_end>()) / 1e6;
#endif
    q.memcpy(ctx->hHits[0], ctx->dHits[0], sizeof(HitBuf)).wait();
    if (hits) *hits = ctx->hHits[0]->count;
    if (transcripts) {
      q.memcpy(transcripts, dTr, (size_t)ctx->batch * 16 * sizeof(uint32_t)).wait();
      sycl::free(dTr, q);
      dTr = nullptr;
    }
  } catch (const std::exception &e) {
    if (dTr) sycl::free(dTr, *ctx->q);
    return fail(err, err_len, std::string("SYCL error: ") + e.what());
  }
  const std::string ae = take_async(ctx);
  if (!ae.empty()) return fail(err, err_len, "SYCL error: " + ae);
  return true;
}

// Measurement only (pearl_sycl_check's PEARL_W_RESEED): wait for everything queued.
extern "C" void pearl_sycl_wait(void *handle) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (ctx && ctx->q) ctx->q->wait();
}

// The noised operands, row-major, and the current a_seed, for a host-side
// reference fold. A context in fragment order draws them row-major for this;
// the next batch of a fragment fold draws them back (set_layout).
extern "C" bool pearl_sycl_read_operands(void *handle, int8_t *Ap, int8_t *Bp, uint8_t a_seed[32]) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx || !ctx->haveJob) return false;
  resolve_pending(ctx);
  try {
    set_layout(ctx, 0);
    ctx->q->wait();
    ctx->q->memcpy(Ap, ctx->dAp, (size_t)ctx->profile.m * ctx->profile.k).wait();
    ctx->q->memcpy(Bp, ctx->dBp, (size_t)ctx->profile.n * ctx->profile.k).wait();
  } catch (const std::exception &) {
    return false;
  }
  memcpy(a_seed, ctx->aSeed, 32);
  return true;
}
