// Host-side driver for the Pearl CUDA core: owns device memory, runs the kernel
// pipeline, and implements the four extern "C" entry points pearl_core.cc calls.
// Without this the addon compiles and fails to link on four undefined symbols.
//
// MEMORY BUDGET, because it is the design constraint that shapes everything
// here. The mandated profile is m=n=131072, k=4096, which makes a full int8 A
// (m×k) and Bᵀ (n×k) 512 MiB each — 1 GiB resident, before noise. That fits a
// 24 GB card comfortably and would not fit an 8 GB one alongside a co-running
// LLM, so `pearl_host_create` checks free VRAM up front and fails with a
// readable message rather than dying inside a kernel launch. The app already
// treats a failed core as "engine unavailable" and says so.
//
// WHY A AND B ARE RESIDENT AT ALL. The commitments hash the WHOLE operands
// (hash_a over pad1024(A), hash_b over pad1024(Bᵀ)), so they must exist somewhere
// once per job. They are generated on-device from job_key rather than uploaded,
// which keeps the PCIe bus out of it entirely: generation and hashing are both
// GPU-side and happen once per job, after which the search reads them.
//
// WHAT THE SEARCH VARIES. Not a header nonce — re-deriving job_key per attempt
// would mean re-hashing 1 GiB per attempt. Per job the commitments are computed
// once, and the search then walks output sub-regions: each region index selects
// the offset of the mandated row/column tile, giving a distinct transcript and so
// a distinct jackpot hash. That index is what travels back as `nonce` and goes
// into the share, and it is why the miner reports progress as "regions".
//
// STATUS: never compiled — the development box has no CUDA toolkit and no MSVC.
// Reviewed source with the build wired up, checked against the JS reference's
// known-answer vectors for semantics (test/minerReference.test.js), not against
// a running GPU. Treat every performance claim as absent rather than optimistic.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <chrono>
#include <set>
#include <vector>

#include <cstdlib>
#include <cstdio>

#include "pearl_config.h"

// Declared in pearl_kernel.cu.
extern "C" __global__ void pearl_gen_dense(const uint32_t *seed,
                                           const uint8_t *label,
                                           const uint32_t *row_indices,
                                           int8_t *out, uint32_t num_rows,
                                           uint32_t rank);
extern "C" __global__ void pearl_gen_perm(const uint32_t *seed,
                                          const uint8_t *label, uint32_t *out,
                                          uint32_t k, uint32_t rank);
extern "C" __global__ void pearl_gen_operand(const uint32_t *key,
                                             const uint8_t *label, int8_t *out,
                                             uint64_t total, uint64_t salt);
extern "C" __global__ void pearl_hash_operands(const uint32_t *job_key,
                                               const uint8_t *padded,
                                               uint32_t len, uint8_t *out);
extern "C" __global__ void pearl_materialize(const int8_t *base,
                                             const int8_t *dense,
                                             const uint32_t *perm, int8_t *out,
                                             uint32_t rows, uint32_t k,
                                             uint32_t rank);
extern "C" __global__ void pearl_materialize16(const int8_t *base,
                                               const int8_t *dense,
                                               const uint32_t *perm, int8_t *out,
                                               uint32_t rows, uint32_t k_log2,
                                               uint32_t rank);
extern "C" __global__ void pearl_restamp_operand(const uint32_t *key,
                                                 int8_t *operand, uint64_t salt,
                                                 uint64_t chunks, uint32_t *tree,
                                                 uint8_t *root_out);
extern "C" __global__ void pearl_tile_fold_wmma(
    const int8_t *Aprime, const int8_t *Bprime, uint32_t m, uint32_t n,
    uint32_t k, uint32_t rank, uint32_t chunks, uint32_t col_off,
    uint32_t rows_valid, uint32_t col_groups, uint32_t tiles,
    const PearlTranscriptTest test, const PearlHitList hits);
extern "C" __global__ void pearl_partials(const int8_t *Aprime, const int8_t *Bprime,
                                          const uint32_t *cols_pattern,
                                          uint32_t cols_count, uint32_t m, uint32_t n,
                                          uint32_t k, uint32_t rank, uint32_t chunks,
                                          uint32_t col_off, uint32_t col_groups,
                                          int32_t *D);
extern "C" __global__ void pearl_gemm_fold(
    const int32_t *D, const uint32_t *rows_pattern, uint32_t rows_count,
    uint32_t cols_count, uint32_t m, uint32_t rows_valid, uint32_t chunks,
    uint64_t region_base, uint32_t *jackpot_out);
extern "C" __global__ void pearl_blake3_chunk_cvs(const uint32_t *key,
                                                  const uint8_t *data,
                                                  uint64_t chunks,
                                                  uint32_t *cvs_out);
extern "C" __global__ void pearl_blake3_parent_layer(const uint32_t *key,
                                                     const uint32_t *in_cvs,
                                                     uint64_t pairs,
                                                     uint32_t is_root,
                                                     uint32_t *out_cvs);
extern "C" __global__ void pearl_blake3_unkeyed(const uint8_t *in, uint32_t len,
                                                uint8_t *out);
extern "C" __global__ void pearl_bind_root(const uint8_t *salt,
                                           const uint8_t *root, uint32_t dim,
                                           uint8_t *out);
extern "C" __global__ void pearl_finalize(const uint32_t *a_seed,
                                          const uint32_t *jackpot,
                                          const uint8_t *target_be,
                                          uint8_t *hash_out, int *is_share);

// Mirrors the struct pearl_core.cc declares. Kept in this one header-free form
// deliberately: the two files must agree on the layout, and a shared header that
// pulled in <napi.h> would drag Node headers into nvcc.
// One side of a share proof, carried WITH the hit.
//
// A single shared snapshot on the context was not enough: the search keeps
// running after a hit, and a later hit overwrote the buffer before the host had
// read the earlier one. The same leaf range would verify for one region and
// fail for the next. The proof has to travel with the result it belongs to.
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
  // Which operand draw this came from. The proof a pool needs is over the
  // operand data, so a share is unprovable without it.
  uint64_t salt;
  std::vector<uint8_t> proof;
  // The share proof for this hit, captured while the operands still belong
  // to it.
  PearlProofSide proof_a;
  PearlProofSide proof_bt;
  bool found;
};

namespace {

struct Ctx {
  PearlProfile profile;

  // The card every allocation below lives on, as chosen by
  // pearl_host_select_device. Kept because the current device is a PER-THREAD
  // setting: the search runs on its own thread, and a thread that never set it
  // would launch kernels on device 0 against pointers belonging to this one.
  // pearl_host_bind_thread is how that thread inherits the choice.
  int device = 0;

  // Whether the fold's shared-memory opt-in has been made for this context's
  // card. The attribute belongs to a DEVICE, so one flag for the whole process
  // (which this was) opted in only the first card to search; every other card's
  // fold launch then asked for 96 KB it had not been granted.
  bool smemOptedIn = false;

  // Fold blocks that fit on this context's card at once: SMs times blocks per
  // SM at the fold's shared-memory footprint. The fold is persistent, so this
  // is its grid. Per context for the same reason as the flag above -- two
  // cards need not have the same SM count. 0 until the first search asks.
  unsigned foldResident = 0;

  // Operands, generated once per job and then read by every region.
  int8_t *dA = nullptr;   // [m, k]
  int8_t *dB = nullptr;   // [n, k]  (Bᵀ, row-major)

  // The noised operands, computed once per commitment. int8, matching the
  // reference's saturating convert-down: operand and noise are both int7, so the
  // sum fits, and an int8 operand is what lets the fold use __dp4a at all.
  int8_t *dAp = nullptr; // [m, k]
  int8_t *dBp = nullptr; // [n, k]

  // The noise factors. Only two of the four are dense — E_AR and E_BL are sparse
  // +-1 selectors, stored as one (p0, p1) index pair per k, and they depend only
  // on the seeds rather than on the tile offset.
  int8_t *dEAL = nullptr;    // dense  [m, rank]
  int8_t *dEBR = nullptr;    // dense  [n, rank]
  uint32_t *dPermA = nullptr; // sparse [k, 2]
  uint32_t *dPermB = nullptr; // sparse [k, 2]

  // The two 32-byte seed labels, "A_tensor" and "B_tensor", zero-padded. They go
  // into the RNG message while the commitment seed is the key.
  uint8_t *dLabelA = nullptr;
  uint8_t *dLabelB = nullptr;

  // The cert-v3 domain-separation salts, and the bound roots they produce.
  uint8_t *dSaltA = nullptr;
  uint8_t *dSaltB = nullptr;
  uint8_t *dBoundA = nullptr;
  uint8_t *dBoundB = nullptr;

  uint32_t *dRows = nullptr;
  uint32_t *dCols = nullptr;
  uint32_t *dHitTranscript = nullptr;  // [PEARL_MAX_HITS][16] — hits only
  uint8_t *dHashes = nullptr;     // [PEARL_MAX_HITS][32] — hits only
  uint32_t *dHitCount = nullptr;  // one counter per batch
  uint32_t *dHitIndex = nullptr;  // [PEARL_MAX_HITS]
  uint32_t colBatch = 1;          // valid column offsets per launch
  uint32_t rowsValid = 1;         // count of valid row offsets
  uint32_t colsValid = 1;         // count of valid column offsets
  std::vector<uint32_t> hHitIndex;
  uint32_t batch = 0;
  uint32_t *dJobKey = nullptr;
  uint32_t *dASeed = nullptr;
  uint32_t *dBSeed = nullptr;
  uint32_t *dCvs = nullptr;      // BLAKE3 tree scratch (leaf CVs, reduced in place)

  // The commitment TREE, kept rather than reduced away.
  //
  // A share has to be proved against the operand data the tile touched, and
  // that proof needs sibling digests from every level. The device already
  // computes all of them to get the root; discarding them forced the host to
  // rebuild the whole tree in JS, which at m=32768, k=4096 took 21 seconds a
  // share -- long enough that the job could rotate underneath it.
  //
  // Layers are stored end to end: level 0 (the leaves) first, then each parent
  // level. About 2*leaves nodes in total, 8 MiB at the mainnet geometry.
  uint32_t *dTreeA = nullptr;
  uint32_t *dTreeB = nullptr;
  std::vector<uint64_t> layerOffA;  // node offset of each level
  std::vector<uint64_t> layerOffB;

  // The proof for the most recent hit, captured AT HIT TIME.
  //
  // The operands are re-drawn every time the region space is exhausted, which
  // at this geometry is every few tens of milliseconds. Fetching the proof
  // afterwards therefore reads a DIFFERENT draw's tree than the one the hit was
  // found under -- sometimes a level that no longer exists, sometimes a proof
  // that simply does not verify. Both were observed before this existed.
  std::vector<uint8_t> snapLeavesA, snapLeavesB;
  std::vector<uint8_t> snapSibsA, snapSibsB;
  std::vector<uint32_t> snapLeafIdxA, snapLeafIdxB;
  bool snapValid = false;
  uint8_t *dSeedBuf = nullptr;   // 64-byte concat for b_seed / a_seed
  uint8_t *dHashA = nullptr;
  uint8_t *dHashB = nullptr;
  uint64_t cvCapacity = 0;
  uint8_t *dTarget = nullptr;
  uint8_t *dHash = nullptr;
  int *dIsShare = nullptr;

  uint8_t aSeed[PEARL_HASH_BYTES] = {0};
  uint8_t bSeed[PEARL_HASH_BYTES] = {0};
  bool haveJob = false;
  // The job is kept so the operands can be re-drawn under a new salt without
  // the caller having to hand the header back.
  uint8_t header[PEARL_HEADER_BYTES] = {0};
  uint8_t target[PEARL_HASH_BYTES] = {0};
  uint64_t salt = 0;
  // header76 ‖ config52 on the device, for job_key. Allocated once: this was a
  // cudaMalloc/cudaFree pair inside every redraw.
  uint8_t *dSeedInput = nullptr;
  // True once the CURRENT job has had a full draw: A, B, both trees, b_seed and
  // B' all belong to it. Only then may a redraw restamp A instead of redrawing
  // everything. set_job clears it, so a new job never mixes with an old tree.
  bool baseDrawn = false;
};

// The row/column patterns the tile folds over. Derived from the counts plus the
// fixed stride, exactly as the reference does, so the host and the config block
// cannot disagree about what they mean.
void build_patterns(const PearlProfile &, std::vector<uint32_t> *rows,
                    std::vector<uint32_t> *cols) {
  // Straight from the reference's defaults. This used to DERIVE the indices from
  // counts plus an assumed stride, which produced a 2x64 tile of the wrong
  // indices entirely.
  rows->assign(PEARL_ROWS_PATTERN, PEARL_ROWS_PATTERN + PEARL_ROWS_COUNT);
  cols->assign(PEARL_COLS_PATTERN, PEARL_COLS_PATTERN + PEARL_COLS_COUNT);
}

// Keyed BLAKE3 over a whole operand: hash each 1024-byte chunk into a leaf CV,
// then fold pairs until one root remains. The final fold carries ROOT and its
// output IS the digest.
//
// Chunk counts are powers of two here (m*k with power-of-two m and k), so the
// tree is balanced and the fold is exact. A non-power-of-two operand would need
// BLAKE3's left-heavy layout, which this deliberately does not pretend to do.
void operand_commitment(Ctx *ctx, const uint8_t *data, size_t len,
                        uint8_t *out32, uint32_t *tree,
                        std::vector<uint64_t> *offsets) {
  const uint64_t chunks = len / 1024;
  const int threads = 256;
  if (offsets) offsets->clear();
  if (chunks <= 1) {
    // One chunk or less: the chunk's own final compression carries ROOT.
    pearl_hash_operands<<<1, 1>>>(ctx->dJobKey, data, (uint32_t)len, out32);
    return;
  }

  // Each level is written to its own place rather than reduced in place, so the
  // sibling digests a share proof needs are still there afterwards.
  uint32_t *dst = tree ? tree : ctx->dCvs;
  uint64_t base = 0;
  if (offsets) offsets->push_back(0);
  pearl_blake3_chunk_cvs<<<(unsigned)((chunks + threads - 1) / threads), threads>>>(
      ctx->dJobKey, data, chunks, dst);

  uint64_t count = chunks;
  while (count > 1) {
    const uint64_t pairs = count / 2;
    const uint32_t isRoot = (pairs == 1) ? 1u : 0u;
    const uint64_t next = tree ? base + count : 0;
    pearl_blake3_parent_layer<<<(unsigned)((pairs + threads - 1) / threads), threads>>>(
        ctx->dJobKey, dst + base * 8, pairs, isRoot, dst + next * 8);
    if (offsets) offsets->push_back(next);
    base = next;
    count = pairs;
  }
  cudaMemcpy(out32, dst + base * 8, PEARL_HASH_BYTES, cudaMemcpyDeviceToDevice);
}

// Which 1024-byte chunks hold these matrix rows. A row of k bytes can straddle
// a boundary, so this is a range per row.
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

// Capture one side's proof while the tree still belongs to the hit.
//
// The sibling order must match the verifier's exactly: level by level, visiting
// the live set in ascending index order, emitting a sibling only when it is not
// itself live.
void snapshotProof(Ctx *ctx, bool isA, const uint32_t *rows, uint32_t nrows,
                   std::vector<uint32_t> *leafIdx, std::vector<uint8_t> *leaves,
                   std::vector<uint8_t> *sibs) {
  const uint32_t k = ctx->profile.k;
  const int8_t *operand = isA ? ctx->dA : ctx->dB;
  const uint32_t *tree = isA ? ctx->dTreeA : ctx->dTreeB;
  const std::vector<uint64_t> &offs = isA ? ctx->layerOffA : ctx->layerOffB;
  const uint64_t totalLeaves =
      (uint64_t)(isA ? ctx->profile.m : ctx->profile.n) * k / 1024;

  leafIndicesForRows(rows, nrows, k, leafIdx);

  leaves->resize(leafIdx->size() * 1024);
  for (size_t i = 0; i < leafIdx->size(); i++) {
    cudaMemcpy(leaves->data() + i * 1024,
               operand + (uint64_t)(*leafIdx)[i] * 1024, 1024,
               cudaMemcpyDeviceToHost);
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
      uint8_t node[PEARL_HASH_BYTES];
      cudaMemcpy(node, tree + (offs[level] + want) * 8, PEARL_HASH_BYTES,
                 cudaMemcpyDeviceToHost);
      sibs->insert(sibs->end(), node, node + PEARL_HASH_BYTES);
    }
    std::set<uint32_t> next;
    for (uint32_t i : current) next.insert(i / 2);
    current.assign(next.begin(), next.end());
    levelLen = (levelLen + 1) / 2;
    level++;
  }
}

bool fail(char *err, size_t err_len, const char *msg) {
  if (err && err_len) snprintf(err, err_len, "%s", msg);
  return false;
}

// What one instance of `profile` costs on a card, in bytes.
//
// Two callers ask: the pre-flight in pearl_host_create, and the device choice
// in pearl_host_select_device. They have to ask the SAME question — a card
// chosen against one number and then refused against another is the worst of
// both answers.
//
// The terms moved here wholesale from that pre-flight, comments and all; each
// one is a thing that was got wrong once.
size_t needed_bytes(const PearlProfile *profile) {
  const size_t k = profile->k;
  const size_t rank = profile->rank;
  const size_t aBytes = (size_t)profile->m * k;
  const size_t bBytes = (size_t)profile->n * k;
  const size_t noiseBytes = (size_t)profile->m * rank + (size_t)profile->n * rank
                            + 2 * k * 2 * sizeof(uint32_t) + 64;
  // The materialised operands are int8, the same size as the sources. They were
  // int32 while the noise was (wrongly) reconstructed at full rank, which cost
  // 2 GiB at mainnet on top of the 1 GiB of sources.
  const size_t primeBytes = aBytes + bBytes;
  // What a batch really costs: one transcript per REGION, and regions are
  // row OFFSETS by column offsets, not rows by columns. Two stale terms lived
  // here and together overstated it by about 25x:
  //
  //   - profile->m instead of m/PEARL_ROWS_COUNT, which is the number of valid
  //     row offsets and therefore the batch's real height;
  //   - a 32-byte hash and a flag PER REGION, from when finalize wrote every
  //     region's hash and the host read back a flag array. It writes only on a
  //     hit now, into a fixed PEARL_MAX_HITS list.
  //
  // The consequence was not cosmetic: this check refused geometries the miner
  // runs fine on, which is what kept the search pinned to the smaller operand
  // draw and paid the redraw cost four times more often than it had to.
  const size_t colBatch = profile->col_batch ? profile->col_batch : 1u;
  const size_t rowsValid = profile->m / PEARL_ROWS_COUNT;
  const size_t colsValid = profile->n / PEARL_COLS_COUNT;
  const size_t batchCols = colBatch > colsValid ? colsValid : colBatch;
  const size_t batchBytes = batchCols * rowsValid * PEARL_JACKPOT_BUCKETS * sizeof(uint32_t)
                            + (size_t)PEARL_MAX_HITS * (PEARL_HASH_BYTES + sizeof(uint32_t));
  return aBytes + bBytes + primeBytes + noiseBytes + batchBytes + (1u << 20);
}

#define CUDA_OK(expr, msg)                                   \
  do {                                                       \
    cudaError_t _e = (expr);                                 \
    if (_e != cudaSuccess) {                                 \
      if (err && err_len)                                    \
        snprintf(err, err_len, "%s: %s", msg,                \
                 cudaGetErrorString(_e));                    \
      return nullptr;                                        \
    }                                                        \
  } while (0)

// Make a context's card the current device for the length of one call, and put
// the caller's back afterwards.
//
// The current device is PER THREAD, and these entry points run on two kinds of
// thread: the context's own search thread (pinned once by pearl_host_bind_thread)
// and the JS thread, which loads every job for every core. Constructing a core
// leaves the JS thread on that core's card, so on a two-card rig it sat on the
// LAST card created -- and the first core's operand draw then launched its
// kernels there, against pointers that live on the other card. That is an
// illegal memory access, and it is sticky: both cores died on their first search
// (an RTX PRO 4500 + RTX 4070 rig on v0.5.3). One card never shows it, because
// there is only one device to be current.
struct DeviceScope {
  int prev = -1;
  explicit DeviceScope(int device) {
    if (cudaGetDevice(&prev) != cudaSuccess) prev = -1;
    if (prev != device) cudaSetDevice(device);
  }
  ~DeviceScope() {
    int now = -1;
    if (prev >= 0 && cudaGetDevice(&now) == cudaSuccess && now != prev) cudaSetDevice(prev);
  }
};

}  // namespace

// Choose the card this core mines on, and make it the calling thread's device.
//
// Nothing used to choose it. The core allocated on whatever the CUDA runtime
// calls device 0, and device 0 is NOT the first card nvidia-smi lists: the
// runtime orders devices by its own "fastest first" heuristic unless
// CUDA_DEVICE_ORDER says otherwise, while nvidia-smi always orders by PCI bus.
// On a single-card rig the two agree and nobody notices. On a two-card rig they
// need not, and then the app names one card and mines on another — reported
// from the field as a 32 GB RTX PRO 4500 shown in the UI while an RTX 4070 did
// the work, at a fraction of the hashrate (issue #226).
//
// Half the fix is that the shells now pin the ordering
// (CUDA_DEVICE_ORDER=PCI_BUS_ID — see shared/gpu.alignCudaDeviceOrder), so an
// index means the same card here as it does in nvidia-smi. This is the other
// half: pick which of those indices to mine on, and report it back with the
// device's own name so the host can SAY which card it took instead of assuming.
//
// `requested` >= 0 is an operator's explicit choice (PEARL_GPU_INDEX) and wins
// outright, including over a card too small for the profile — create's VRAM
// pre-flight speaks to that, and it quotes real numbers.
//
// Otherwise rank by SM count x clock: "how much machine per second". That is a
// proxy rather than a benchmark, but it is the proxy that survives across
// generations, and the only judgement it has to get right is "mine on the big
// card, not the little one". Cards whose TOTAL memory cannot hold the profile
// are skipped first — no amount of free VRAM would save them — and ties keep
// the lower index, so a rig of identical cards picks the same one every run.
//
// Returns the chosen device index, or -1 with a message in `err`.
extern "C" int pearl_host_select_device(const PearlProfile *profile, int requested,
                                        char *name, size_t name_len, char *err,
                                        size_t err_len) {
  if (name && name_len) name[0] = '\0';
  if (!profile) { fail(err, err_len, "no profile supplied"); return -1; }

  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    fail(err, err_len, "no CUDA device found — is an NVIDIA driver installed?");
    return -1;
  }

  int chosen = -1;
  if (requested >= 0) {
    if (requested >= devices) {
      if (err && err_len) {
        snprintf(err, err_len,
                 "GPU %d was asked for (PEARL_GPU_INDEX) but this machine has "
                 "%d: valid indices are 0..%d",
                 requested, devices, devices - 1);
      }
      return -1;
    }
    chosen = requested;
  } else {
    const size_t need = needed_bytes(profile);
    double bestScore = -1.0;
    for (int d = 0; d < devices; d++) {
      cudaDeviceProp prop;
      if (cudaGetDeviceProperties(&prop, d) != cudaSuccess) continue;
      // A card in an exclusive or prohibited compute mode cannot take our
      // context at all; choosing it would fail the whole start on a rig that
      // has a perfectly good second card.
      //
      // Read through cudaDeviceGetAttribute rather than prop.computeMode: CUDA 13
      // removed that field from cudaDeviceProp (as it did clockRate, read the same
      // way just below), so the struct member does not compile there. The
      // attribute exists in 12 and 13 alike.
      int computeMode = cudaComputeModeDefault;
      if (cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, d) == cudaSuccess
          && computeMode == cudaComputeModeProhibited) continue;
      if (prop.totalGlobalMem < need) continue;
      int clockKHz = 0;
      if (cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, d) != cudaSuccess
          || clockKHz <= 0) {
        clockKHz = 1;  // unreadable clock: rank on SM count alone rather than drop the card
      }
      const double score = (double)prop.multiProcessorCount * (double)clockKHz;
      if (score > bestScore) { bestScore = score; chosen = d; }
    }
    // Every card was skipped — all too small, or none would report itself. Take
    // the first one anyway: create's pre-flight then refuses with the numbers,
    // which is a far better answer than "no CUDA device found".
    if (chosen < 0) chosen = 0;
  }

  const cudaError_t e = cudaSetDevice(chosen);
  if (e != cudaSuccess) {
    if (err && err_len) {
      snprintf(err, err_len, "could not open GPU %d: %s", chosen,
               cudaGetErrorString(e));
    }
    return -1;
  }

  cudaDeviceProp prop;
  if (name && name_len && cudaGetDeviceProperties(&prop, chosen) == cudaSuccess) {
    snprintf(name, name_len, "%s", prop.name);
  }
  return chosen;
}

extern "C" void *pearl_host_create(const PearlProfile *profile, char *err,
                                   size_t err_len) {
  if (!profile) { fail(err, err_len, "no profile supplied"); return nullptr; }

  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    fail(err, err_len, "no CUDA device found — is an NVIDIA driver installed?");
    return nullptr;
  }

  // The commitment folds the chunk tree as a balanced pairwise reduction, which
  // is only BLAKE3's tree when the chunk count is a power of two. Refuse
  // anything else rather than computing a wrong root in silence — that produces
  // wrong seeds and a share no pool accepts, with no symptom to follow.
  {
    const uint64_t aChunks = (uint64_t)profile->m * profile->k / 1024u;
    const uint64_t bChunks = (uint64_t)profile->n * profile->k / 1024u;
    const bool aOk = aChunks && (aChunks & (aChunks - 1)) == 0;
    const bool bOk = bChunks && (bChunks & (bChunks - 1)) == 0;
    if (!aOk || !bOk) {
      if (err && err_len) {
        snprintf(err, err_len,
                 "m*k/1024 and n*k/1024 must each be a power of two (got %llu "
                 "and %llu): the commitment tree fold assumes it",
                 (unsigned long long)aChunks, (unsigned long long)bChunks);
      }
      return nullptr;
    }
  }

  const size_t k = profile->k;
  const size_t rank = profile->rank;
  const size_t aBytes = (size_t)profile->m * k;
  const size_t bBytes = (size_t)profile->n * k;
  const size_t need = needed_bytes(profile);

  // Check the budget BEFORE allocating, so an 8 GB card gets a sentence it can
  // act on instead of an out-of-memory abort three kernels deep. This reads the
  // CURRENT device, which pearl_host_select_device has already chosen — so the
  // card measured here is the card that will mine.
  size_t freeMem = 0, totalMem = 0;
  if (cudaMemGetInfo(&freeMem, &totalMem) == cudaSuccess && freeMem < need) {
    if (err && err_len) {
      snprintf(err, err_len,
               "not enough free VRAM for the rank-%u profile: need ~%zu MiB, "
               "%zu MiB free of %zu MiB",
               (unsigned)profile->rank, need >> 20, freeMem >> 20, totalMem >> 20);
    }
    return nullptr;
  }

  Ctx *ctx = new Ctx();
  ctx->profile = *profile;
  // Whatever pearl_host_select_device left current is where these allocations
  // land, so that is the card the context belongs to.
  if (cudaGetDevice(&ctx->device) != cudaSuccess) ctx->device = 0;

  CUDA_OK(cudaMalloc(&ctx->dA, aBytes), "allocating A");
  CUDA_OK(cudaMalloc(&ctx->dB, bBytes), "allocating B");
  CUDA_OK(cudaMalloc(&ctx->dAp, aBytes), "allocating the noised A");
  CUDA_OK(cudaMalloc(&ctx->dBp, bBytes), "allocating the noised B");
  CUDA_OK(cudaMalloc(&ctx->dEAL, (size_t)profile->m * rank), "allocating E_AL");
  CUDA_OK(cudaMalloc(&ctx->dEBR, (size_t)profile->n * rank), "allocating E_BR");
  CUDA_OK(cudaMalloc(&ctx->dPermA, k * 2 * sizeof(uint32_t)), "allocating E_AR");
  CUDA_OK(cudaMalloc(&ctx->dPermB, k * 2 * sizeof(uint32_t)), "allocating E_BL");

  // The seed labels are fixed ASCII, so they are uploaded once here rather than
  // rebuilt per job.
  CUDA_OK(cudaMalloc(&ctx->dLabelA, 32), "allocating the A label");
  CUDA_OK(cudaMalloc(&ctx->dLabelB, 32), "allocating the B label");
  {
    uint8_t lab[32];
    memset(lab, 0, 32);
    memcpy(lab, "A_tensor", 8);
    cudaMemcpy(ctx->dLabelA, lab, 32, cudaMemcpyHostToDevice);
    memset(lab, 0, 32);
    memcpy(lab, "B_tensor", 8);
    cudaMemcpy(ctx->dLabelB, lab, 32, cudaMemcpyHostToDevice);
  }

  CUDA_OK(cudaMalloc(&ctx->dSaltA, 32), "allocating the A salt");
  CUDA_OK(cudaMalloc(&ctx->dSaltB, 32), "allocating the B salt");
  CUDA_OK(cudaMalloc(&ctx->dBoundA, PEARL_HASH_BYTES), "allocating the bound A root");
  CUDA_OK(cudaMalloc(&ctx->dBoundB, PEARL_HASH_BYTES), "allocating the bound B root");
  cudaMemcpy(ctx->dSaltA, PEARL_SEED_SALT_A, 32, cudaMemcpyHostToDevice);
  cudaMemcpy(ctx->dSaltB, PEARL_SEED_SALT_B, 32, cudaMemcpyHostToDevice);
  // A batch is col_batch column offsets by m row offsets. Widening it past a
  // single column offset is what took the search from launch-bound to
  // compute-bound: the fixed per-batch cost (three launches and a synchronising
  // copy) was flat at 134-213us regardless of the work inside it.
  //
  // Only offsets with the pattern's bits clear are valid, so there are
  // m/PEARL_ROWS_COUNT of them down the rows and n/PEARL_COLS_COUNT across the
  // columns. Searching the other 31/32 produced hashes no pool would take.
  ctx->rowsValid = profile->m / PEARL_ROWS_COUNT;
  ctx->colsValid = profile->n / PEARL_COLS_COUNT;
  ctx->colBatch = profile->col_batch ? profile->col_batch : 1u;
  if (ctx->colBatch > ctx->colsValid) ctx->colBatch = ctx->colsValid;
  ctx->batch = ctx->colBatch * ctx->rowsValid;
  ctx->hHitIndex.resize(PEARL_MAX_HITS);
  // No partial-product table any more. The fold was two passes once -- compute
  // every partial dot product into a table, then gather it -- and the fused
  // tile fold replaced both, because a CUMULATIVE accumulator cannot be
  // decomposed into reusable partials. The allocation outlived its only
  // consumer and was still sized from col_batch, so raising col_batch to 2048
  // silently reserved GIGABYTES that nothing ever read, and the pre-flight VRAM
  // check refused geometries the miner would have run fine.
  // The same fate befell the per-region transcript buffer: the fold hashes its
  // own transcripts now, so only a hit's transcript is ever stored -- 64 slots
  // of 64 bytes where the batch used to take 64 bytes a region, 1 GiB at the
  // mainnet geometry.
  CUDA_OK(cudaMalloc(&ctx->dHitTranscript,
                     (size_t)PEARL_MAX_HITS * PEARL_JACKPOT_BUCKETS * sizeof(uint32_t)),
          "allocating the hit transcripts");
  CUDA_OK(cudaMalloc(&ctx->dHashes, (size_t)PEARL_MAX_HITS * PEARL_HASH_BYTES),
          "allocating the batch hashes");
  CUDA_OK(cudaMalloc(&ctx->dHitCount, sizeof(uint32_t)), "allocating the hit counter");
  CUDA_OK(cudaMalloc(&ctx->dHitIndex, (size_t)PEARL_MAX_HITS * sizeof(uint32_t)),
          "allocating the hit list");
  CUDA_OK(cudaMalloc(&ctx->dJobKey, 8 * sizeof(uint32_t)), "allocating job_key");
  CUDA_OK(cudaMalloc(&ctx->dASeed, 8 * sizeof(uint32_t)), "allocating a_seed");
  CUDA_OK(cudaMalloc(&ctx->dBSeed, 8 * sizeof(uint32_t)), "allocating b_seed");
  // Leaf CVs for the larger of the two operands: 8 words per 1024-byte chunk.
  ctx->cvCapacity = (aBytes > bBytes ? aBytes : bBytes) / 1024;
  if (ctx->cvCapacity < 1) ctx->cvCapacity = 1;
  CUDA_OK(cudaMalloc(&ctx->dCvs, ctx->cvCapacity * 8 * sizeof(uint32_t)),
          "allocating the BLAKE3 tree scratch");
  // The kept commitment trees. Levels sum to just under 2*leaves nodes.
  {
    const uint64_t aLeaves = (uint64_t)profile->m * profile->k / 1024;
    const uint64_t bLeaves = (uint64_t)profile->n * profile->k / 1024;
    CUDA_OK(cudaMalloc(&ctx->dTreeA, 2 * aLeaves * 8 * sizeof(uint32_t)),
            "allocating the A commitment tree");
    CUDA_OK(cudaMalloc(&ctx->dTreeB, 2 * bLeaves * 8 * sizeof(uint32_t)),
            "allocating the B commitment tree");
  }
  CUDA_OK(cudaMalloc(&ctx->dSeedBuf, 64), "allocating the seed buffer");
  CUDA_OK(cudaMalloc(&ctx->dSeedInput, PEARL_HEADER_BYTES + PEARL_CONFIG_BYTES),
          "allocating the job_key input");
  CUDA_OK(cudaMalloc(&ctx->dHashA, PEARL_HASH_BYTES), "allocating hash_a");
  CUDA_OK(cudaMalloc(&ctx->dHashB, PEARL_HASH_BYTES), "allocating hash_b");
  CUDA_OK(cudaMalloc(&ctx->dTarget, PEARL_HASH_BYTES), "allocating the target");
  CUDA_OK(cudaMalloc(&ctx->dHash, PEARL_HASH_BYTES), "allocating the hash");
  CUDA_OK(cudaMalloc(&ctx->dIsShare, sizeof(int)), "allocating the share flag");

  std::vector<uint32_t> rows, cols;
  build_patterns(*profile, &rows, &cols);
  CUDA_OK(cudaMalloc(&ctx->dRows, rows.size() * sizeof(uint32_t)), "allocating rows");
  CUDA_OK(cudaMalloc(&ctx->dCols, cols.size() * sizeof(uint32_t)), "allocating cols");
  cudaMemcpy(ctx->dRows, rows.data(), rows.size() * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(ctx->dCols, cols.data(), cols.size() * sizeof(uint32_t),
             cudaMemcpyHostToDevice);

  return ctx;
}

// Point the CALLING thread at the card this context lives on.
//
// cudaSetDevice is per-thread state, and the search runs on a thread of its own
// (PearlCore::SearchLoop). Without this the search thread would default to
// device 0 and launch kernels there against pointers allocated on the chosen
// card — on a single-card rig an invisible no-op, on a two-card rig an illegal
// access at the first batch. Called once when the thread starts, not per batch:
// a batch is only ~200us, and this is not free.
extern "C" void pearl_host_bind_thread(void *handle) {
  Ctx *ctx = (Ctx *)handle;
  if (!ctx) return;
  cudaSetDevice(ctx->device);
}

extern "C" void pearl_host_destroy(void *handle) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx) return;
  DeviceScope scope(ctx->device);
  cudaFree(ctx->dA); cudaFree(ctx->dB);
  cudaFree(ctx->dAp); cudaFree(ctx->dBp);
  cudaFree(ctx->dEAL); cudaFree(ctx->dEBR);
  cudaFree(ctx->dPermA); cudaFree(ctx->dPermB);
  cudaFree(ctx->dLabelA); cudaFree(ctx->dLabelB);
  cudaFree(ctx->dSaltA); cudaFree(ctx->dSaltB);
  cudaFree(ctx->dBoundA); cudaFree(ctx->dBoundB);
  cudaFree(ctx->dRows); cudaFree(ctx->dCols);

  cudaFree(ctx->dHashes); cudaFree(ctx->dHitCount); cudaFree(ctx->dHitIndex);
  cudaFree(ctx->dTreeA); cudaFree(ctx->dTreeB);
  cudaFree(ctx->dCvs); cudaFree(ctx->dSeedBuf); cudaFree(ctx->dSeedInput);
  cudaFree(ctx->dHashA); cudaFree(ctx->dHashB);
  cudaFree(ctx->dHitTranscript); cudaFree(ctx->dJobKey);
  cudaFree(ctx->dASeed); cudaFree(ctx->dBSeed);
  cudaFree(ctx->dTarget); cudaFree(ctx->dHash); cudaFree(ctx->dIsShare);
  delete ctx;
}

extern "C" void pearl_host_reseed(void *handle, uint64_t salt);

// Load a job and draw its operands under `salt`.
//
// The salt is what keeps two cards off each other's work. One salt is worth m*n
// regions; a card searches those, then re-draws under the next salt it owns. Give
// every card a different starting salt and a stride equal to the number of cards
// and they never draw the same operands, so no work and no share is done twice.
// Without it every card would start at salt 0 and walk 1, 2, 3 in step, and a
// second card would earn exactly nothing.
extern "C" void pearl_host_set_job_salted(void *handle, const uint8_t *header,
                                          const uint8_t *target, uint64_t salt) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx) return;
  DeviceScope scope(ctx->device);
  memcpy(ctx->header, header, PEARL_HEADER_BYTES);
  memcpy(ctx->target, target, PEARL_HASH_BYTES);
  // A new job always gets a full draw. Clearing this is what stops a restamp
  // from grafting new bytes onto a tree the previous job's key built.
  ctx->baseDrawn = false;
  pearl_host_reseed(handle, salt);
}

// The single-card spelling, kept so the bench probe and anything else built
// against the old entry point still link.
extern "C" void pearl_host_set_job(void *handle, const uint8_t *header,
                                   const uint8_t *target) {
  pearl_host_set_job_salted(handle, header, target, 0);
}

namespace {

// Diagnostic only (PEARL_RESEED_STAGES): the wall time of each restamp stage,
// synchronising between them, averaged over 32 redraws and printed to stderr.
// Measured on a 4090 at mainnet geometry: tree 0.04 ms, seeds 0.04, dense+perm
// 0.04, materialize 0.59 -- 0.72 ms a redraw, against 4.6 ms for a full draw
// with the same kernels. What is left is one read and one write of A at DRAM
// speed; a new a_seed changes the noise on every row, so that part cannot go.
#ifdef PEARL_RESEED_STAGES
double g_stage[4] = {0, 0, 0, 0};
int g_stageCalls = 0;
bool g_stageOn = false;  // only restamps are timed, not the draw a job starts with
std::chrono::steady_clock::time_point g_stageLast;
void stage_lap(int i) {
  cudaDeviceSynchronize();
  const auto now = std::chrono::steady_clock::now();
  if (i < 0) {
    g_stageOn = true;
  } else if (g_stageOn) {
    g_stage[i] += std::chrono::duration<double, std::milli>(now - g_stageLast).count();
  }
  g_stageLast = now;
}
void stage_report() {
  g_stageOn = false;
  if (++g_stageCalls < 32) return;
  fprintf(stderr, "restamp ms: tree %.3f seeds %.3f dense+perm %.3f materialize %.3f\n",
          g_stage[0] / 32, g_stage[1] / 32, g_stage[2] / 32, g_stage[3] / 32);
  g_stage[0] = g_stage[1] = g_stage[2] = g_stage[3] = 0;
  g_stageCalls = 0;
}
#define PEARL_LAP(i) stage_lap(i)
#define PEARL_LAP_REPORT() stage_report()
#else
#define PEARL_LAP(i) (void)0
#define PEARL_LAP_REPORT() (void)0
#endif

const int kDrawThreads = 256;
unsigned draw_blocks(size_t n) {
  return (unsigned)((n + kDrawThreads - 1) / kDrawThreads);
}

// b_seed = blake3(job_key ‖ bound_b), then a_seed = blake3(b_seed ‖ bound_a).
// The order is NOT symmetric: b_seed is derived first and feeds a_seed. With
// `withB` false only the A link is recomputed -- b_seed depends on B alone, so a
// restamp of A leaves it exactly as it was.
//
// Bind the roots before they enter the chain. Under cert-v3 each root is
// re-hashed with its dimension under a domain-separation salt, which is what
// commits m and n; legacy passes the raw roots straight through.
//
// No synchronisation in here: every step is on the one stream, in order. The
// three cudaDeviceSynchronize calls this replaced ordered nothing the stream
// did not already order.
void derive_seeds(Ctx *ctx, bool withB) {
  const bool legacy = ctx->profile.seed_derivation == PEARL_SEED_LEGACY;
  if (legacy) {
    cudaMemcpy(ctx->dBoundA, ctx->dHashA, PEARL_HASH_BYTES, cudaMemcpyDeviceToDevice);
    if (withB)
      cudaMemcpy(ctx->dBoundB, ctx->dHashB, PEARL_HASH_BYTES, cudaMemcpyDeviceToDevice);
  } else {
    pearl_bind_root<<<1, 1>>>(ctx->dSaltA, ctx->dHashA, ctx->profile.m, ctx->dBoundA);
    if (withB)
      pearl_bind_root<<<1, 1>>>(ctx->dSaltB, ctx->dHashB, ctx->profile.n, ctx->dBoundB);
  }
  if (withB) {
    cudaMemcpy(ctx->dSeedBuf, ctx->dJobKey, PEARL_HASH_BYTES, cudaMemcpyDeviceToDevice);
    cudaMemcpy(ctx->dSeedBuf + PEARL_HASH_BYTES, ctx->dBoundB, PEARL_HASH_BYTES,
               cudaMemcpyDeviceToDevice);
    pearl_blake3_unkeyed<<<1, 1>>>(ctx->dSeedBuf, 64,
                                   reinterpret_cast<uint8_t *>(ctx->dBSeed));
  }
  cudaMemcpy(ctx->dSeedBuf, ctx->dBSeed, PEARL_HASH_BYTES, cudaMemcpyDeviceToDevice);
  cudaMemcpy(ctx->dSeedBuf + PEARL_HASH_BYTES, ctx->dBoundA, PEARL_HASH_BYTES,
             cudaMemcpyDeviceToDevice);
  pearl_blake3_unkeyed<<<1, 1>>>(ctx->dSeedBuf, 64,
                                 reinterpret_cast<uint8_t *>(ctx->dASeed));
}

// One side's noise, and the noised operand it produces.
//
// The A side is keyed by a_seed and the B side by b_seed. Obvious as written,
// but the reference destructures its tuple as
//   let (b_noise_seed, a_noise_seed) = commitment_hash;
// i.e. b first, so it is easy to end up with these swapped — silently, and
// with no symptom other than shares that are never accepted.
//
// row_indices is null here because the whole operand is noised, so a row's
// index IS its position. The parameter exists for the verifier's path, which
// only ever wants the handful of rows in one tile.
//
// Then fold the noise into the operand ONCE. Each element costs two lookups and
// a subtract, because E_AR and E_BL are sparse +-1 selectors rather than dense
// factors — the version that reconstructed at full rank did rank times this
// much work and computed the wrong thing.
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

  pearl_gen_dense<<<draw_blocks((size_t)rows * (rank / 32)), kDrawThreads>>>(
      seed, label, nullptr, dense, rows, rank);
  pearl_gen_perm<<<draw_blocks((k + 7) / 8), kDrawThreads>>>(seed, label, perm, k, rank);
  PEARL_LAP(2);
  if ((k & (k - 1u)) == 0u && k >= 16u) {
    uint32_t kLog2 = 0;
    while ((1u << kLog2) < k) kLog2++;
    pearl_materialize16<<<draw_blocks(len / 16), kDrawThreads>>>(src, dense, perm, dst,
                                                                 rows, kLog2, rank);
  } else {
    pearl_materialize<<<draw_blocks(len), kDrawThreads>>>(src, dense, perm, dst, rows,
                                                          k, rank);
  }
  PEARL_LAP(3);
}

// The full draw: both operands from scratch under `salt`, both commitment trees,
// both seeds, both sides of the noise. What every job starts with.
void full_draw(Ctx *ctx, uint64_t salt) {
  // job_key = blake3(header76 ‖ config52), UNKEYED. Computed on-device so there
  // is one BLAKE3 implementation in the binary rather than two that can
  // disagree.
  //
  // This used to hash it KEYED with an all-zero key, under the belief that a
  // zero key is the same as no key. It is not: keyed mode seeds the chaining
  // value from the key and sets KEYED_HASH, so the two produce different
  // digests. The device and the oracle therefore derived different job keys —
  // and, both being internally consistent, nothing anywhere said so.
  uint8_t seedInput[PEARL_HEADER_BYTES + PEARL_CONFIG_BYTES];
  memcpy(seedInput, ctx->header, PEARL_HEADER_BYTES);
  pearl_write_config52(&ctx->profile, seedInput + PEARL_HEADER_BYTES);
  uint8_t *dSeedInput = ctx->dSeedInput;
  cudaMemcpy(dSeedInput, seedInput, sizeof(seedInput), cudaMemcpyHostToDevice);
  pearl_blake3_unkeyed<<<1, 1>>>(dSeedInput, sizeof(seedInput),
                                 reinterpret_cast<uint8_t *>(ctx->dJobKey));

  const size_t aLen = (size_t)ctx->profile.m * ctx->profile.k;
  const size_t bLen = (size_t)ctx->profile.n * ctx->profile.k;

  // The operands are the miner's own workload, so their contents are our choice
  // — but their RANGE is not. They must be int7: the noise adds another int7 and
  // the sum has to stay inside int8 for the Int7xInt7ToInt32 MMA.
  //
  // Keyed by job_key rather than by a commitment seed, so these streams cannot
  // collide with the noise streams even though they share the labels.
  pearl_gen_operand<<<draw_blocks(aLen / 32 + 1), kDrawThreads>>>(
      ctx->dJobKey, ctx->dLabelA, ctx->dA, aLen, salt);
  pearl_gen_operand<<<draw_blocks(bLen / 32 + 1), kDrawThreads>>>(
      ctx->dJobKey, ctx->dLabelB, ctx->dB, bLen, salt);

  // hash_a and hash_b: keyed BLAKE3 over the WHOLE operands. These are Merkle
  // trees over 1024-byte chunks, not one long chain — hashing them as a single
  // chunk (which this did until the device run showed a_seed == b_seed) gives the
  // wrong digest for anything over 1024 bytes and so the wrong seeds.
  operand_commitment(ctx, reinterpret_cast<const uint8_t *>(ctx->dA), aLen,
                     ctx->dHashA, ctx->dTreeA, &ctx->layerOffA);
  operand_commitment(ctx, reinterpret_cast<const uint8_t *>(ctx->dB), bLen,
                     ctx->dHashB, ctx->dTreeB, &ctx->layerOffB);

  derive_seeds(ctx, true);
  draw_noise(ctx, true);
  draw_noise(ctx, false);

  cudaMemcpy(ctx->dTarget, ctx->target, PEARL_HASH_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(ctx->bSeed, ctx->dBSeed, PEARL_HASH_BYTES, cudaMemcpyDeviceToHost);
}

// A same-job redraw: a new A root from a handful of bytes, and nothing on B.
//
// The full draw regenerates 512 MiB of operands, hashes all of it into two
// trees and noises both sides, every ~150 ms of search: 4% of the hashrate the
// app shows (about 6 ms a draw; 4.6 ms with the faster noise kernels). Most of that is redundant within a job. The search space is keyed
// by a_seed alone (it keys the jackpot hash and seeds A's noise), and
// a_seed = blake3(b_seed ‖ bound(root_A)), where b_seed depends only on B. So a
// fresh space needs only a fresh root_A:
//
//   - stamp the salt into A's first bytes and repair leaf 0's path in the
//     stored tree (pearl_restamp_operand). Share proofs read leaves from dA and
//     siblings from dTreeA (snapshotProof), so both are updated in place and
//     keep agreeing with the root;
//   - re-bind root_A and derive the new a_seed from the unchanged b_seed;
//   - redraw A's noise and re-materialise A'. B, B', B's tree and b_seed stay.
//
// Distinctness: within a job every salt writes a different stamp, so a
// different root_A. Across cards the first draws differ (each card's full draw
// uses its own salt), so B and b_seed already differ. A stamp can only repeat
// the first draw's own random bytes by chance, about 1 in 127^11.
void restamp(Ctx *ctx, uint64_t salt) {
  const uint64_t aChunks = (uint64_t)ctx->profile.m * ctx->profile.k / 1024;
  PEARL_LAP(-1);
  pearl_restamp_operand<<<1, 1>>>(ctx->dJobKey, ctx->dA, salt, aChunks, ctx->dTreeA,
                                  ctx->dHashA);
  PEARL_LAP(0);
  derive_seeds(ctx, false);
  PEARL_LAP(1);
  draw_noise(ctx, true);
  PEARL_LAP_REPORT();
}

}  // namespace

// Re-draw the operands under a new salt and rebuild everything downstream of
// them: the commitments, the seeds, the noise, and the noised operands.
//
// This is the outer loop of the search. One salt yields m*n regions and nothing
// more, because the region index is just (row offset, column offset) -- so the
// miner must periodically pick new operands or it re-mines what it has already
// tried, at full reported hashrate and with no chance of a share.
//
// The first draw of a job is full; every later one restamps A (see restamp).
// Measured on a 4090, full miner loop, interleaved A/B: 222.8 -> 229.0 TH/s.
// Forcing the full draw every time (PEARL_FULL_REDRAW: the old behaviour, but
// with the faster noise kernels) measured 224.5, so most of the gain is the
// restamp itself. PEARL_FULL_REDRAW is also how the frozen device parity
// vectors for salts above 0 were produced.
extern "C" void pearl_host_reseed(void *handle, uint64_t salt) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx) return;
  DeviceScope scope(ctx->device);
  ctx->salt = salt;

  // A restamp needs a real tree to repair: at least two leaves, so leaf 0 has a
  // path. Profiles smaller than that always take the full draw.
  const uint64_t aChunks = (uint64_t)ctx->profile.m * ctx->profile.k / 1024;
#ifdef PEARL_FULL_REDRAW
  const bool canRestamp = false;
  (void)aChunks;
#else
  const bool canRestamp = ctx->baseDrawn && aChunks >= 2;
#endif
  if (canRestamp) {
    restamp(ctx, salt);
  } else {
    full_draw(ctx, salt);
  }

  // The one synchronising copy: the search must not launch against half-built
  // seeds, and the host copy of a_seed travels with every hit.
  cudaMemcpy(ctx->aSeed, ctx->dASeed, PEARL_HASH_BYTES, cudaMemcpyDeviceToHost);

#ifdef PEARL_RESTAMP_CHECK
  // Diagnostic only: after a restamp, rebuild A's whole tree from scratch and
  // demand it match the repaired one node for node. Share proofs only carry
  // leaf 0 when the tile covers row 0, so an end-to-end run could miss a stale
  // leaf; this cannot. 192 of 192 restamps matched on a 4090.
  if (canRestamp) {
    const size_t aLen = (size_t)ctx->profile.m * ctx->profile.k;
    const size_t nodes = 2 * (aLen / 1024) * 8;
    static uint32_t *dCheck = nullptr;
    static uint8_t *dRoot = nullptr;
    if (!dCheck) {
      cudaMalloc(&dCheck, nodes * sizeof(uint32_t));
      cudaMalloc(&dRoot, PEARL_HASH_BYTES);
    }
    std::vector<uint64_t> offs;
    operand_commitment(ctx, reinterpret_cast<const uint8_t *>(ctx->dA), aLen, dRoot,
                       dCheck, &offs);
    const size_t used = (size_t)(offs.back() + 1) * 8;
    std::vector<uint32_t> want(used), got(used);
    uint8_t rootWant[PEARL_HASH_BYTES], rootGot[PEARL_HASH_BYTES];
    cudaMemcpy(want.data(), dCheck, used * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(got.data(), ctx->dTreeA, used * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(rootWant, dRoot, PEARL_HASH_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(rootGot, ctx->dHashA, PEARL_HASH_BYTES, cudaMemcpyDeviceToHost);
    const bool same = offs == ctx->layerOffA && want == got
                      && memcmp(rootWant, rootGot, PEARL_HASH_BYTES) == 0;
    static int checked = 0, bad = 0;
    checked++;
    if (!same) bad++;
    if (!same || checked % 32 == 0)
      fprintf(stderr, "restamp check: salt %llu %s (%d checked, %d bad)\n",
              (unsigned long long)salt, same ? "ok" : "MISMATCH", checked, bad);
  }
#endif
  ctx->baseDrawn = true;
  ctx->haveJob = true;
}

extern "C" bool pearl_host_search(void *handle, uint64_t nonce_base,
                                  uint32_t batch, PearlSearchResult *out,
                                  uint64_t *attempts, char *err,
                                  size_t err_len) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (attempts) *attempts = 0;
  if (!ctx || !ctx->haveJob || !out) return false;
  // A no-op on the search thread, which is already bound to this card; the
  // guard is for any other caller.
  DeviceScope scope(ctx->device);

  const uint32_t k = ctx->profile.k;
  const uint32_t rank = ctx->profile.rank;
  const uint32_t chunks = (k + rank - 1) / rank;
  // The fold is compiled for exactly the mandated geometry (see PEARL_FOLD_*).
  if (rank != PEARL_FOLD_RANK || k != PEARL_FOLD_K) {
    if (err && err_len)
      snprintf(err, err_len, "fold kernel is built for rank %u, k %u (got rank %u, k %u)",
               (unsigned)PEARL_FOLD_RANK, (unsigned)PEARL_FOLD_K, rank, k);
    return false;
  }
  // Clamped to m so one launch shares a single col_off: D is built for exactly
  // the columns that batch touches. nonce_base stays a multiple of m because the
  // caller advances by the attempt count we report back.
  // The caller's batch hint is advisory; the real width is the context's, since
  // the partial table was allocated for exactly that many column groups.
  (void)batch;
  const uint32_t regions = ctx->batch;
  const uint32_t col_groups = ctx->colBatch;
  // Block size for the search kernels. Tunable because occupancy against
  // register pressure is not something to guess at.
  static const int threads = []() {
    const char *e = getenv("PEARL_BLOCK");
    // 512 by default: sixteen warps make the CTA tile 128x256, which raises
    // the MACs bought per staged byte from 64 to 85 -- and the two-stage
    // pipeline hides the staging that a single 512-thread block used to expose.
    const int v = e ? atoi(e) : 512;
    return (v == 64 || v == 128 || v == 256 || v == 512 || v == 1024) ? v : 256;
  }();
  const int warps_per_block = threads / 32;
  // A valid-offset INDEX; the kernel expands it into an actual offset.
  const uint32_t col_off =
      (uint32_t)((nonce_base / ctx->rowsValid) % ctx->colsValid);

  // One fused launch: the tile fold keeps its accumulator across chunks, so
  // there are no reusable partials to stage and no second pass.
  const uint32_t warpsPerBlock = threads / 32;
  const uint32_t regionsPerWarp = PEARL_WMMA_ROW_TILES * (PEARL_WMMA_ROWS / PEARL_ROWS_COUNT);
  // A block is a 2D grid of warps: PEARL_WARP_ROWS down, the rest across. Both
  // dimensions have to tile exactly, because a warp that falls outside cannot
  // return -- staging is a block-wide cooperative load and a __syncthreads()
  // some warps skip hangs the launch.
  const uint64_t rowBlocks = ctx->rowsValid / regionsPerWarp;
  const uint32_t warpCols = warpsPerBlock / PEARL_WARP_ROWS;
  const uint64_t colBlocks = col_groups / PEARL_WMMA_COL_BLK;
  // The fold is compiled for exactly this block size (see PEARL_FOLD_THREADS).
  if ((uint32_t)threads != PEARL_FOLD_THREADS) {
    if (err && err_len)
      snprintf(err, err_len, "fold kernel is built for %u threads a block (got %d)",
               (unsigned)PEARL_FOLD_THREADS, threads);
    return false;
  }
  // The staging walks base + stride rather than a table of addresses, which is
  // only the same sequence when quads divides the staging thread count and the
  // column step lands on a whole number of column groups.
  const uint32_t quadsPerRow = rank / 16u;
  const uint32_t stageThreads = (uint32_t)threads;
  if (quadsPerRow == 0 || stageThreads % quadsPerRow != 0
      || (stageThreads / quadsPerRow) % PEARL_COLS_COUNT != 0) {
    if (err && err_len)
      snprintf(err, err_len,
               "staging stride is not uniform: %u threads, %u quads a row, %u columns a group",
               stageThreads, quadsPerRow, (unsigned)PEARL_COLS_COUNT);
    return false;
  }
  if (warpsPerBlock % PEARL_WARP_ROWS != 0 || rowBlocks % PEARL_WARP_ROWS != 0
      || warpCols == 0 || colBlocks % warpCols != 0) {
    if (err && err_len)
      snprintf(err, err_len,
               "warp grid %ux%u does not tile %llu row blocks by %llu column blocks",
               PEARL_WARP_ROWS, warpCols, (unsigned long long)rowBlocks,
               (unsigned long long)colBlocks);
    return false;
  }
  const unsigned tiles =
      (unsigned)((rowBlocks / PEARL_WARP_ROWS) * (colBlocks / warpCols));
  // Two full-chunk stages; the transcripts live in registers and global now.
  const size_t smem = (size_t)PEARL_STAGE_BUFS
                      * ((size_t)warpCols * PEARL_WMMA_COL_BLK * 16
                         + (size_t)PEARL_WARP_ROWS * regionsPerWarp * PEARL_ROWS_COUNT)
                      * PEARL_SB_STRIDE;
  // The fold writes each transcript slot exactly once only when every chunk
  // has its own bucket. A geometry with more chunks than buckets would fold
  // into whatever the buffer already held; refuse it rather than mine garbage.
  if (chunks > PEARL_JACKPOT_BUCKETS) {
    if (err && err_len)
      snprintf(err, err_len, "%u chunks exceed %u transcript buckets", chunks,
               (unsigned)PEARL_JACKPOT_BUCKETS);
    return false;
  }
  // Staging both operands puts this past the 48 KB a block gets by default.
  // Ada allows 99 KB per block, but only when asked; without this the launch
  // fails with an invalid-configuration error rather than running slowly.
  // Once per CONTEXT, not once per process: the attribute is per device (see
  // Ctx::smemOptedIn).
  if (!ctx->smemOptedIn) {
    cudaFuncSetAttribute(reinterpret_cast<const void *>(pearl_tile_fold_wmma),
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    ctx->smemOptedIn = true;
  }
  // The fold is persistent: exactly as many blocks as can be resident, each
  // walking tiles a grid-width apart, so no tile starts with its first chunk
  // exposed (see the kernel). Launching more would only queue blocks behind
  // the resident ones and bring the exposed starts back. Asked once per
  // context, after the opt-in, because the answer depends on the footprint.
  if (ctx->foldResident == 0) {
    int sms = 0, perSm = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, ctx->device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &perSm, reinterpret_cast<const void *>(pearl_tile_fold_wmma), threads, smem);
    ctx->foldResident = (unsigned)(sms > 0 ? sms : 1) * (unsigned)(perSm > 0 ? perSm : 1);
  }
  const unsigned blocks = tiles < ctx->foldResident ? tiles : ctx->foldResident;
  // The fold hashes every transcript itself and tests it against the bound. It
  // writes only on a hit and appends to a compact list, so the readback below
  // is four bytes rather than one flag per region.
  //
  // The key and target go in as words, by value (see PearlTranscriptTest):
  // a_seed as the little-endian words BLAKE3 keys with, the target as
  // big-endian words so the kernel compares whole words most significant first.
  PearlTranscriptTest test;
  memcpy(test.key, ctx->aSeed, sizeof(test.key));
  for (int i = 0; i < 8; i++) {
    const uint8_t *t = ctx->target + i * 4;
    test.target_w[i] = ((uint32_t)t[0] << 24) | ((uint32_t)t[1] << 16) |
                       ((uint32_t)t[2] << 8) | (uint32_t)t[3];
  }
  test.hash_big_endian = (int)ctx->profile.hash_big_endian;
  PearlHitList hitList;
  hitList.count = ctx->dHitCount;
  hitList.index = ctx->dHitIndex;
  hitList.hash = reinterpret_cast<uint32_t *>(ctx->dHashes);
  hitList.transcript = ctx->dHitTranscript;
  cudaMemsetAsync(ctx->dHitCount, 0, sizeof(uint32_t));
  pearl_tile_fold_wmma<<<blocks, threads, smem>>>(
      ctx->dAp, ctx->dBp, ctx->profile.m, ctx->profile.n, k, rank, chunks,
      col_off, ctx->rowsValid, col_groups, tiles, test, hitList);

  uint32_t hits = 0;
  cudaMemcpy(&hits, ctx->dHitCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);

  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    if (err && err_len)
      snprintf(err, err_len, "CUDA error during search: %s", cudaGetErrorString(e));
    return false;
  }
  if (attempts) *attempts = regions;

  if (hits > 0) {
    const uint32_t n_hits = hits < PEARL_MAX_HITS ? hits : PEARL_MAX_HITS;
    cudaMemcpy(ctx->hHitIndex.data(), ctx->dHitIndex, (size_t)n_hits * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    // The kernel appends with an atomic, so the list is in an arbitrary order.
    // Take the LOWEST region index, which is what a sequential scan would have
    // returned — otherwise which share gets submitted varies run to run.
    uint32_t best = 0;
    for (uint32_t i = 1; i < n_hits; i++) {
      if (ctx->hHitIndex[i] < ctx->hHitIndex[best]) best = i;
    }
    cudaMemcpy(out->jackpot_hash, ctx->dHashes + (size_t)best * PEARL_HASH_BYTES,
               PEARL_HASH_BYTES, cudaMemcpyDeviceToHost);
    memcpy(out->a_seed, ctx->aSeed, PEARL_HASH_BYTES);
    memcpy(out->b_seed, ctx->bSeed, PEARL_HASH_BYTES);
    out->nonce = nonce_base + ctx->hHitIndex[best];
    out->salt = ctx->salt;

    // Capture the proof NOW, while the operands and tree still belong to this
    // hit. A few tens of milliseconds later they will have been re-drawn.
    {
      // The GLOBAL region index, not the batch-local one. Both give the same
      // row offset, because nonce_base is a multiple of rowsValid -- but the
      // COLUMN offset is (region / rowsValid) % colsValid, and the local index
      // drops the batch base entirely. The columns in the snapshot then belong
      // to a different tile than the row indices the proof declares, which the
      // pool reports as "Failed to extract strip".
      const uint64_t region = out->nonce;
      const uint32_t rowIdx = (uint32_t)(region % ctx->rowsValid);
      const uint32_t colIdx = (uint32_t)((region / ctx->rowsValid) % ctx->colsValid);
      const uint32_t rowOff = pearl_expand_offset(rowIdx, PEARL_ROWS_MASK);
      const uint32_t colOff = pearl_expand_offset(colIdx, PEARL_COLS_MASK);

      uint32_t rows[PEARL_ROWS_COUNT], cols[PEARL_COLS_COUNT];
      for (int i = 0; i < PEARL_ROWS_COUNT; i++) rows[i] = rowOff | PEARL_ROWS_PATTERN[i];
      for (int i = 0; i < PEARL_COLS_COUNT; i++) cols[i] = colOff | PEARL_COLS_PATTERN[i];

      snapshotProof(ctx, true, rows, PEARL_ROWS_COUNT, &out->proof_a.leaf_indices,
                    &out->proof_a.leaves, &out->proof_a.siblings);
      snapshotProof(ctx, false, cols, PEARL_COLS_COUNT, &out->proof_bt.leaf_indices,
                    &out->proof_bt.leaves, &out->proof_bt.siblings);
      cudaMemcpy(out->proof_a.root, ctx->dHashA, PEARL_HASH_BYTES, cudaMemcpyDeviceToHost);
      cudaMemcpy(out->proof_bt.root, ctx->dHashB, PEARL_HASH_BYTES, cudaMemcpyDeviceToHost);
      out->proof_a.total_leaves = (uint64_t)ctx->profile.m * ctx->profile.k / 1024;
      out->proof_bt.total_leaves = (uint64_t)ctx->profile.n * ctx->profile.k / 1024;
    }
    out->proof.assign(PEARL_JACKPOT_BUCKETS * 4, 0);
    // Indexed by the hit's SLOT, like the hash: the fold keeps no per-region
    // transcripts, only the ones that hit.
    cudaMemcpy(out->proof.data(),
               ctx->dHitTranscript + (size_t)best * PEARL_JACKPOT_BUCKETS,
               PEARL_JACKPOT_BUCKETS * 4, cudaMemcpyDeviceToHost);
    out->found = true;
    return true;
  }
  out->found = false;
  return false;
}

// ---------------------------------------------------------------------------
// Share-proof accessors.
//
// A submitted share carries the 1024-byte operand chunks its tile touched plus
// the sibling digests that authenticate them against the committed root. Both
// already exist on the device -- the chunks in the operand, the digests in the
// commitment tree -- so the host copies out the handful it needs instead of
// rebuilding the tree, which is thousands of times more work.
// ---------------------------------------------------------------------------

// Copy whole 1024-byte leaf chunks out of an operand.// Copy whole 1024-byte leaf chunks out of an operand.
extern "C" bool pearl_host_leaf_chunks(void *handle, int isA,
                                       const uint32_t *leaf_indices,
                                       uint32_t count, uint8_t *out) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx || !leaf_indices || !out) return false;
  DeviceScope scope(ctx->device);
  const int8_t *src = isA ? ctx->dA : ctx->dB;
  const uint64_t bytes = isA ? (uint64_t)ctx->profile.m * ctx->profile.k
                             : (uint64_t)ctx->profile.n * ctx->profile.k;
  for (uint32_t i = 0; i < count; i++) {
    const uint64_t off = (uint64_t)leaf_indices[i] * 1024;
    if (off + 1024 > bytes) return false;
    cudaMemcpy(out + (size_t)i * 1024, src + off, 1024, cudaMemcpyDeviceToHost);
  }
  return true;
}

// Copy 32-byte nodes out of one level of a commitment tree.
extern "C" bool pearl_host_tree_nodes(void *handle, int isA, uint32_t level,
                                      const uint32_t *indices, uint32_t count,
                                      uint8_t *out) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx || !indices || !out) return false;
  DeviceScope scope(ctx->device);
  const std::vector<uint64_t> &offs = isA ? ctx->layerOffA : ctx->layerOffB;
  const uint32_t *tree = isA ? ctx->dTreeA : ctx->dTreeB;
  if (!tree || level >= offs.size()) return false;
  const uint64_t base = offs[level];
  for (uint32_t i = 0; i < count; i++) {
    cudaMemcpy(out + (size_t)i * PEARL_HASH_BYTES,
               tree + (base + indices[i]) * 8, PEARL_HASH_BYTES,
               cudaMemcpyDeviceToHost);
  }
  return true;
}

// How many levels the tree has, so the host knows where to stop walking.
extern "C" uint32_t pearl_host_tree_levels(void *handle, int isA) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx) return 0;
  return (uint32_t)(isA ? ctx->layerOffA.size() : ctx->layerOffB.size());
}
