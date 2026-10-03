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
// once per job. They are made on-device (a constant fill plus a salt stamp, or
// hashed from job_key; see PEARL_OPERAND_CONST) rather than uploaded, which
// keeps the PCIe bus out of it entirely: filling and hashing are both GPU-side
// and happen once per job, after which the search reads them.
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

#include <cuda.h>
#include <cudaTypedefs.h>  // PFN_cuTensorMapEncodeTiled
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <chrono>
#include <mutex>
#include <set>
#include <vector>

#include <cstdlib>
#include <cstdio>

#include "pearl_config.h"
#include "pearl_tensor_map.h"

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
extern "C" __global__ void pearl_materialize16_kblocked(const int8_t *base,
                                                        const int8_t *dense,
                                                        const uint32_t *perm, int8_t *out,
                                                        uint32_t rows, uint32_t k_log2,
                                                        uint32_t rank, uint32_t kb_log2);
extern "C" __global__ void pearl_restamp_operand(const uint32_t *key,
                                                 int8_t *operand, uint64_t salt,
                                                 uint64_t chunks, uint32_t *tree,
                                                 uint8_t *root_out);
extern "C" __global__ void pearl_tile_fold_wmma(
    const int8_t *Aprime, const int8_t *Bprime, uint32_t m, uint32_t n,
    uint32_t k, uint32_t rank, uint32_t chunks, uint32_t col_off,
    uint32_t rows_valid, uint32_t col_groups, uint32_t tiles,
    const PearlTranscriptTest test, const PearlHitList hits);
extern "C" __global__ void pearl_tile_fold_tall(
    const int8_t *Aprime, const int8_t *Bprime, uint32_t m, uint32_t n,
    uint32_t k, uint32_t rank, uint32_t chunks, uint32_t col_off,
    uint32_t rows_valid, uint32_t col_groups, uint32_t tiles,
    const PearlTranscriptTest test, const PearlHitList hits,
    const PearlTensorMap tmA, const PearlTensorMap tmB);
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
  // Whether the fold binary this card loaded is the persistent build (see
  // PEARL_FOLD_PERSISTENT). Only then is the grid foldResident; otherwise it
  // is one block per tile, as before.
  bool foldPersistent = false;
  // Whether it is the eight-warp build (PEARL_FOLD_WIDE_WARPS), which must be
  // launched 256 threads a block rather than 512. Both are read off the loaded
  // binary once, when the context is created (resolve_fold).
  bool foldWide = false;
  // Whether this card runs the tall fold instead (PEARL_FOLD_TALL): its own
  // kernel, pearl_tile_fold_tall, over 192x256 tiles. Read off that kernel's
  // loaded binary (PEARL_TALL_ARCH of its binaryVersion) with the others.
  bool foldTall = false;
  // Whether that is Blackwell's build, which stages with TMA (PEARL_TALL_TMA),
  // through the tensor maps below. Both tall builds read A' and B' k-blocked, so
  // the operand draw writes them that way whenever foldTall. The draw runs before
  // any search, so all of this is resolved when the context is created
  // (resolve_fold) and never changes after.
  bool foldTma = false;
  // Whether that TMA build is the two-CTA cluster one (PEARL_TALL_CLUSTER, off by
  // default): the tall fold is then launched in clusters of PEARL_TALL_CLUSTER_SIZE
  // over tiles of two row groups, and its resident count is in clusters.
  bool foldCluster = false;
  bool foldKnown = false;
  // Why not, when foldKnown is false: reported by the first search, as it was when
  // the search itself asked.
  char foldErr[256] = {0};
  // TMA descriptors for the k-blocked noised operands, encoded once against dAp and
  // dBp (which never move) when foldTma. Zero, and ignored, for every other build.
  // cuTensorMapEncodeTiled wants the map it writes 64-byte aligned. The host pass
  // sees PearlTensorMap unaligned (MSVC cannot pass an over-aligned kernel
  // parameter by value; see pearl_tensor_map.h), so the alignment is declared on
  // the members instead, which MSVC allows. tmA has one map per noised-A buffer
  // (see dAp); tmA[1] is encoded only with the host overlap on.
  alignas(128) PearlTensorMap tmA[2]{};
  alignas(128) PearlTensorMap tmB{};

  // Operands, generated once per job and then read by every region.
  int8_t *dA = nullptr;   // [m, k]
  int8_t *dB = nullptr;   // [n, k]  (Bᵀ, row-major)

  // The noised operands, computed once per commitment. int8, matching the
  // reference's saturating convert-down: operand and noise are both int7, so the
  // sum fits, and an int8 operand is what lets the fold use __dp4a at all.
  // dAp[apCur] is the buffer the fold reads. dAp[1] exists only with the host
  // overlap on (see the block at the end of this struct), for the next salt's
  // materialize to write while the fold reads dAp[apCur].
  int8_t *dAp[2] = {nullptr, nullptr}; // [PEARL_TALL_A_ROWS(m), k] each
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
  // The hit list a fold appends to. Set 0 is the one every search used to have;
  // set 1 exists only with the host overlap on, so consecutive batches can
  // alternate and the next batch's list can be cleared while the current batch's
  // is still unread.
  uint32_t *dHitTranscript[2] = {nullptr, nullptr};  // [PEARL_MAX_HITS][16] — hits only
  uint8_t *dHashes[2] = {nullptr, nullptr};     // [PEARL_MAX_HITS][32] — hits only
  uint32_t *dHitCount[2] = {nullptr, nullptr};  // one counter per batch
  uint32_t *dHitIndex[2] = {nullptr, nullptr};  // [PEARL_MAX_HITS]
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

  // ---------------------------------------------------------------------------
  // Host overlap (PEARL_HOST_OVERLAP=1; see pearl_overlap_enabled).
  //
  // Where the card idles today. pearl_host_search launches one fold and reads
  // the hit count back with a synchronous copy, so between every batch the card
  // waits a host round trip for the next launch. And whenever the nonce reaches
  // the salt span, pearl_host_reseed restamps A on the same stream with the
  // tensor cores idle: 0.72 ms a redraw on a 4090, 0.59 of it materialize (see
  // restamp). The log puts the full loop at 311 TH/s against 313 for the bench
  // (probes/README.md), which has no reseeds and a tighter loop; that gap is the
  // most the prepared restamp can be worth on a 4090. The per-batch round trip
  // is in the bench too, which calls pearl_host_search one batch at a time, so
  // nothing in the log bounds that part. None of it has been run on a GPU; every
  // figure here is the log's, not this code's.
  //
  // With the switch on the host keeps one batch in flight: the next batch is
  // launched before the current one's hits are read, so the card has work queued
  // when a fold ends, and a salt's last batch has the next salt's restamp issued
  // beside it rather than after it. How much of that restamp actually runs beside
  // the fold is the card's business: the tall fold takes 253-255 registers a
  // thread (probes/README.md), the whole register file at 256 threads, so on that
  // build the restamp's blocks start only as the fold's drain, and what overlaps
  // is the batch's tail plus the host round trips. Not measured.
  //
  // Streams. The fold stays on the legacy default stream, so set_job's full draw
  // (launched from the JS thread, on that stream) still queues behind an in-flight
  // fold exactly as today. The legacy stream synchronises with blocking streams
  // and NOT with cudaStreamNonBlocking ones, and both streams below are
  // non-blocking: sideStream runs the next salt's restamp, which must not wait
  // for the fold; readStream carries every readback (hit count, hit data, proof
  // copies), because a copy on the legacy stream would queue behind the fold
  // launched AFTER the one it reads, and that wait is the round trip being
  // removed. Events (cudaEventDisableTiming) order the three: one is recorded
  // after each fold and readStream waits on it before reading that batch's list;
  // one is recorded after the shadow save and readStream waits on it before
  // reading the shadow.
  //
  // Buffers. The noised A is double-buffered (dAp[2], a TMA descriptor each) so
  // the next salt's materialize writes one while the fold reads the other; apCur
  // is the live one. Two hit lists alternate between consecutive batches, and a
  // list is cleared only once its previous batch has been read or dropped.
  //
  // The shadow. A restamp rewrites, in place, exactly leaf 0 of A (the stamp is
  // A's first PEARL_STAMP_BYTES bytes), node 0 of every tree level, and root_A.
  // A hit found under salt s but read after salt s+1's restamp has run needs salt
  // s's values for those, so they are copied aside on sideStream before the
  // restamp kernel, and snapshotProof reads them from there whenever the shadow
  // belongs to the hit's salt. Everything else the proof reads -- the other
  // leaves and siblings, all of B's side -- a restamp does not touch.
  //
  // Why it defaults off. Nothing here has been run on a GPU. The repo's gate for
  // a change that touches which nonce or salt a hit belongs to is
  // probes/verify-hits.js at 400/400 plus a pool-accepted share, and until
  // someone has run that on a 4090 with the switch on, the proven path stays the
  // default. With the switch off every entry point does the same device work on
  // the same stream in the same order as before. What did change on that path:
  // derive_seeds' device-to-device copies are cudaMemcpyAsync on stream 0 rather
  // than cudaMemcpy, snapshotProof copies its siblings after the level walk
  // instead of during it, and mu (below) is taken around search, reseed and
  // set_job, so set_job on the JS thread waits for a search in progress.
  //
  // Locking. The search thread (pearl_host_search_next, pearl_host_reseed) and
  // the JS thread (pearl_host_set_job_salted) both reach into this state, so mu
  // guards apCur, salt, aSeed, jobGen and the prepared / in-flight / shadow
  // records, and is held for the whole of each of those three calls, switch on
  // or off. pearl_core.cc takes its job_mu_ around set_job_salted and around
  // reseed, never around a search, and nothing here calls back into it, so the
  // order is job_mu_ then mu on both threads. mu is not recursive:
  // set_job_salted reaches the redraw through reseed_locked, which assumes it is
  // held.
  // ---------------------------------------------------------------------------
  bool overlap = false;
  std::mutex mu;
  cudaStream_t sideStream = nullptr;
  cudaStream_t readStream = nullptr;
  int apCur = 0;
  // The hit list the next launch takes. Flipped by every launch.
  int nextList = 0;
  // Recorded on the legacy stream right after the fold that writes that list.
  cudaEvent_t listEvent[2] = {nullptr, nullptr};
  // Bumped by set_job. A batch launched under an older job is dropped unread.
  uint64_t jobGen = 0;
  // One launched batch: what reading its hits and building a hit's result takes.
  // The salt and a_seed are the batch's own, because by the time a hit is read the
  // context may already name the prepared salt.
  struct Batch {
    bool valid = false;
    uint64_t nonce = 0;  // nonce_base
    uint64_t salt = 0;
    uint8_t aSeed[PEARL_HASH_BYTES] = {0};
    int apBuf = 0;
    int list = 0;
    uint64_t jobGen = 0;
  };
  Batch inflight;  // launched, not yet read
  // The next salt, restamped into dAp[prepared.apBuf] ahead of pearl_host_reseed,
  // which swaps to it instead of drawing.
  struct Prepared {
    bool valid = false;
    uint64_t salt = 0;
    int apBuf = 0;
    uint8_t aSeed[PEARL_HASH_BYTES] = {0};
  };
  Prepared prepared;
  // The shadow (above): leaf 0 of A, node 0 of each level, root_A, end to end.
  uint8_t *dShadow = nullptr;
  uint32_t shadowLevels = 0;  // tree levels the allocation has room for
  bool shadowValid = false;
  uint64_t shadowSalt = 0;
  cudaEvent_t shadowEvent = nullptr;
};

// Where the shadow keeps each piece. `levels` is layerOffA.size(), which a
// restamp never changes, so the save and the read agree on it.
uint8_t *shadow_leaf0(uint8_t *s) { return s; }
uint8_t *shadow_node0(uint8_t *s, size_t level) { return s + 1024 + level * PEARL_HASH_BYTES; }
uint8_t *shadow_root(uint8_t *s, size_t levels) { return s + 1024 + levels * PEARL_HASH_BYTES; }

// The process-wide switch, read once. Off unless PEARL_HOST_OVERLAP is exactly
// "1" (see Ctx for why off is the default).
bool pearl_overlap_enabled() {
  static const bool on = [] {
    const char *v = std::getenv("PEARL_HOST_OVERLAP");
    return v != nullptr && strcmp(v, "1") == 0;
  }();
  return on;
}

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

// One device-to-host copy of a hit's data. With the host overlap on it goes on
// readStream, behind the batch's event, so it does not queue behind a fold
// launched after the one it reads (see Ctx). Every destination is pageable (a
// std::vector, a field of the result, a stack word; nothing here is pinned), and
// the runtime completes a device-to-pageable-host cudaMemcpyAsync before it
// returns, so each copy is a host sync of its own. The caller's copies_done is a
// safety net, not the sync; pinning the destinations (cudaMallocHost) is what
// would make it one synchronisation per hit. Otherwise it is the synchronous
// cudaMemcpy it always was.
void copy_out(Ctx *ctx, void *dst, const void *src, size_t n) {
  if (ctx->overlap)
    cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, ctx->readStream);
  else
    cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost);
}
void copies_done(Ctx *ctx) {
  if (ctx->overlap) cudaStreamSynchronize(ctx->readStream);
}

// Capture one side's proof while the tree still belongs to the hit.
//
// The sibling order must match the verifier's exactly: level by level, visiting
// the live set in ascending index order, emitting a sibling only when it is not
// itself live.
//
// `shadow` is non-null only for A, with the host overlap on, when the live leaf 0,
// node 0 of each level and root already belong to the NEXT salt (see Ctx, "The
// shadow"): those three then come from the copy saved before the restamp, and
// everything else from the live arrays, which a restamp does not touch.
void snapshotProof(Ctx *ctx, bool isA, const uint32_t *rows, uint32_t nrows,
                   std::vector<uint32_t> *leafIdx, std::vector<uint8_t> *leaves,
                   std::vector<uint8_t> *sibs, uint8_t *shadow) {
  const uint32_t k = ctx->profile.k;
  const int8_t *operand = isA ? ctx->dA : ctx->dB;
  const uint32_t *tree = isA ? ctx->dTreeA : ctx->dTreeB;
  const std::vector<uint64_t> &offs = isA ? ctx->layerOffA : ctx->layerOffB;
  const uint64_t totalLeaves =
      (uint64_t)(isA ? ctx->profile.m : ctx->profile.n) * k / 1024;

  leafIndicesForRows(rows, nrows, k, leafIdx);

  leaves->resize(leafIdx->size() * 1024);
  for (size_t i = 0; i < leafIdx->size(); i++) {
    const uint32_t idx = (*leafIdx)[i];
    const void *src = (shadow && idx == 0)
                          ? static_cast<const void *>(shadow_leaf0(shadow))
                          : static_cast<const void *>(operand + (uint64_t)idx * 1024);
    copy_out(ctx, leaves->data() + i * 1024, src, 1024);
  }

  sibs->clear();
  std::vector<const void *> srcs;  // one device address per sibling, in order
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
      srcs.push_back(shadow && want == 0
                         ? static_cast<const void *>(shadow_node0(shadow, level))
                         : static_cast<const void *>(tree + (offs[level] + want) * 8));
    }
    std::set<uint32_t> next;
    for (uint32_t i : current) next.insert(i / 2);
    current.assign(next.begin(), next.end());
    levelLen = (levelLen + 1) / 2;
    level++;
  }
  // The sources are worked out first and the output sized once, so every copy
  // lands in memory that no later step moves. With pageable destinations each
  // copy completes before it returns (copy_out); copies_done at the end is what
  // would make the snapshot one synchronisation if they were pinned.
  sibs->resize(srcs.size() * PEARL_HASH_BYTES);
  for (size_t j = 0; j < srcs.size(); j++)
    copy_out(ctx, sibs->data() + j * PEARL_HASH_BYTES, srcs[j], PEARL_HASH_BYTES);
  copies_done(ctx);
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
  // 2 GiB at mainnet on top of the 1 GiB of sources. The noised A has the tall
  // fold's padding rows on the end (PEARL_TALL_A_ROWS): 128 KB at mainnet.
  const size_t primeBytes = (size_t)PEARL_TALL_A_ROWS(profile->m) * k + bBytes;
  // What a batch costs now: nothing per region. The fold hashes its own
  // transcripts, so only a hit's transcript, hash and index are stored, in a
  // fixed PEARL_MAX_HITS list. This term used to be a transcript PER REGION --
  // 1 GiB at the mainnet geometry -- and it stayed here after the buffer went,
  // so a card whose free VRAM the local LLM had taken could be refused for a
  // gigabyte the miner no longer asks for.
  const size_t batchBytes =
      (size_t)PEARL_MAX_HITS * (PEARL_HASH_BYTES + sizeof(uint32_t)
                                + PEARL_JACKPOT_BUCKETS * sizeof(uint32_t));
  // The kept commitment trees (just under 2 nodes a leaf, 32 bytes a node, for
  // both operands) and the leaf-CV scratch for the larger one. Real
  // allocations that were never counted: 40 MiB at the mainnet geometry.
  const size_t aLeaves = aBytes / 1024, bLeaves = bBytes / 1024;
  const size_t treeBytes = 2 * (aLeaves + bLeaves) * 32
                           + (aLeaves > bLeaves ? aLeaves : bLeaves) * 32;
  // The host overlap (PEARL_HOST_OVERLAP=1, see Ctx) double-buffers the noised A
  // and the hit list and keeps the shadow: leaf 0, a node a level with room for
  // 64 levels (more than any power-of-two leaf count has), and the root. Counted
  // only when it is on, which both callers see the same way.
  const size_t overlapBytes =
      pearl_overlap_enabled()
          ? (size_t)PEARL_TALL_A_ROWS(profile->m) * k + batchBytes
                + 1024 + 64 * PEARL_HASH_BYTES + PEARL_HASH_BYTES
          : 0;
  return aBytes + bBytes + primeBytes + noiseBytes + batchBytes + treeBytes + overlapBytes
         + (1u << 20);
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

// cuTensorMapEncodeTiled is a driver API entry point; fetching it through the
// runtime keeps the addon's cudart-only link line. Resolved once per process.
PFN_cuTensorMapEncodeTiled_v12000 pearl_encode_tiled() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  static bool tried = false;
  if (!tried) {
    tried = true;
    void *p = nullptr;
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault,
                                         &q) == cudaSuccess
        && q == cudaDriverEntryPointSuccess)
      fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}

// One k-blocked noised operand as the tall fold's TMA reads it (PEARL_TALL_TMA):
// `rows` rows of k int8, stored [k / kBlock][rows][kBlock] by
// pearl_materialize16_kblocked, read as boxes of kBlock bytes by `boxRows` rows of
// one k-block, swizzled SWIZZLE_64B (unit q of box row r at q ^ ((r >> 1) & 3), what
// the fold's ldmatrix lane bases expect). The map is 3-D -- kBlock bytes, rows,
// k-blocks -- rather than a 2-D flattening, so the row dimension stays bounded by
// `rows`: the last row group's rows past m come back zero-filled instead of being the
// next k-block's first rows.
bool pearl_encode_operand(PearlTensorMap *map, const void *base, uint64_t rows, uint64_t k,
                          uint32_t kBlock, uint32_t boxRows) {
  PFN_cuTensorMapEncodeTiled_v12000 enc = pearl_encode_tiled();
  if (!enc || kBlock != 64u || k % kBlock != 0u || boxRows == 0u || boxRows > 256u) return false;
  const cuuint64_t dims[3] = {(cuuint64_t)kBlock, (cuuint64_t)rows, (cuuint64_t)(k / kBlock)};
  const cuuint64_t strides[2] = {(cuuint64_t)kBlock, (cuuint64_t)rows * kBlock};  // bytes
  const cuuint32_t box[3] = {kBlock, boxRows, 1u};
  const cuuint32_t estr[3] = {1u, 1u, 1u};
  return enc(&map->map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, const_cast<void *>(base), dims, strides,
             box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
             CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
         == CUDA_SUCCESS;
}

// Which fold this card loaded, and what launching it takes. Ask which binary
// actually loaded rather than which card it is -- the launch has to match the code
// that runs. The Ada-layout builds of pearl_tile_fold_wmma (sm_89, and sm_86, which
// takes Ada's switches on the shared SM layout, unmeasured) are the persistent one
// (PEARL_FOLD_PERSISTENT) and the eight-warp one (PEARL_FOLD_WIDE_WARPS); the tall
// fold is its own kernel, with a body only in the builds PEARL_TALL_ARCH names, so
// its binary answers for itself, and Blackwell's also says the noised operands are
// read k-blocked through tensor maps (PEARL_TALL_TMA).
//
// Once per context, when it is created: the job's first operand draw has to know
// the layout, and it runs before any search. Nothing here changes afterwards, so the
// draw (on the JS thread) and the search (on its own) read it without a lock. A
// failure is kept in foldErr and reported by the first search.
void resolve_fold(Ctx *ctx) {
  char *err = ctx->foldErr;
  const size_t err_len = sizeof ctx->foldErr;
  err[0] = 0;
  cudaFuncAttributes fa;
  const bool haveAttrs =
      cudaFuncGetAttributes(&fa, reinterpret_cast<const void *>(pearl_tile_fold_wmma))
      == cudaSuccess;
  // An Ada-layout build: sm_89, or sm_86 compiled with Ada's switches. The same
  // architectures pearl_config.h's gates spell out; the launch bound check below
  // catches a build whose gates disagree.
  const bool adaLayout = haveAttrs && (fa.binaryVersion == 86 || fa.binaryVersion == 89);
#ifdef PEARL_FOLD_PERSISTENT_FORCED
  ctx->foldPersistent = PEARL_FOLD_PERSISTENT != 0;
#else
  ctx->foldPersistent = adaLayout;
#endif
#ifdef PEARL_FOLD_WIDE_WARPS_FORCED
  ctx->foldWide = PEARL_FOLD_WIDE_WARPS != 0;
#else
  ctx->foldWide = adaLayout;
#endif
  // The tall fold is persistent like the Ada-layout fold.
  {
    cudaFuncAttributes ft;
    const bool haveTall =
        cudaFuncGetAttributes(&ft, reinterpret_cast<const void *>(pearl_tile_fold_tall))
        == cudaSuccess;
    const bool tallBody = haveTall && PEARL_TALL_ARCH(ft.binaryVersion)
                          && (uint32_t)ft.maxThreadsPerBlock == PEARL_TALL_THREADS;
#ifdef PEARL_FOLD_TALL_FORCED
    ctx->foldTall = PEARL_FOLD_TALL != 0 && tallBody;
#else
    ctx->foldTall = tallBody;
#endif
    // -DPEARL_TALL_TMA=0 builds Blackwell's tall fold on cp.async instead; the host
    // pass sees the same value.
    ctx->foldTma = ctx->foldTall && PEARL_TALL_TMA != 0 && PEARL_TALL_TMA_ARCH(ft.binaryVersion);
    // -DPEARL_TALL_CLUSTER=1 builds that TMA fold as a two-CTA cluster; the host pass
    // sees the same value, so the launch shape follows the body.
    ctx->foldCluster = ctx->foldTma && PEARL_TALL_CLUSTER != 0;
  }
  // The fold is compiled for exactly one block size, which is also its launch
  // bound. A disagreement would not fail loudly: a block of the wrong size
  // returns at once, and the search would report hashrate while finding
  // nothing. So refuse it.
  const uint32_t want = ctx->foldWide ? PEARL_FOLD_WIDE_THREADS : PEARL_FOLD_THREADS;
  if (!haveAttrs) {
    snprintf(err, err_len, "no fold kernel for this card: %s",
             cudaGetErrorString(cudaGetLastError()));
    return;
  }
  if ((uint32_t)fa.maxThreadsPerBlock != want) {
    snprintf(err, err_len, "fold binary takes %d threads a block, the host would launch %u",
             fa.maxThreadsPerBlock, want);
    return;
  }
  // The TMA fold's boxes: 64 bytes of k (one k-block) by the tile's 192 A rows and
  // its 256 B columns. A's map is bounded by m, not by the padded PEARL_TALL_A_ROWS:
  // TMA zero-fills the last row group's rows past it (see PEARL_TALL_TMA).
  // One map per noised-A buffer; the second buffer exists only with the host
  // overlap on (see Ctx), and the launch passes the map of the buffer it reads.
  if (ctx->foldTma
      && (!pearl_encode_operand(&ctx->tmA[0], ctx->dAp[0], ctx->profile.m, ctx->profile.k,
                                PEARL_TALL_STAGE_K, PEARL_TALL_BM)
          || (ctx->overlap
              && !pearl_encode_operand(&ctx->tmA[1], ctx->dAp[1], ctx->profile.m,
                                       ctx->profile.k, PEARL_TALL_STAGE_K, PEARL_TALL_BM))
          || !pearl_encode_operand(&ctx->tmB, ctx->dBp, ctx->profile.n, ctx->profile.k,
                                   PEARL_TALL_STAGE_K, PEARL_TALL_BN))) {
    snprintf(err, err_len, "could not encode the tall fold's TMA descriptors (k %u, %u-byte k-blocks)",
             ctx->profile.k, (unsigned)PEARL_TALL_STAGE_K);
    return;
  }
  ctx->foldKnown = true;
}

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
  // Read once, here: everything below that the switch adds is allocated against it,
  // and needed_bytes counted it the same way.
  ctx->overlap = pearl_overlap_enabled();
  // Whatever pearl_host_select_device left current is where these allocations
  // land, so that is the card the context belongs to.
  if (cudaGetDevice(&ctx->device) != cudaSuccess) ctx->device = 0;

  CUDA_OK(cudaMalloc(&ctx->dA, aBytes), "allocating A");
  CUDA_OK(cudaMalloc(&ctx->dB, bBytes), "allocating B");
  // The tall fold's last row group of tiles reads past A's m rows (see
  // PEARL_TALL_A_ROWS): the cp.async build (Ada and Ampere) up to 64 rows past the
  // end of the last k-block. Nothing generates the bytes past m * k; they are zeroed
  // once so the fold reads defined bytes there, and it hashes no region from them.
  // The second buffer, with the host overlap on, is padded the same way: a restamp
  // writes only rows 0..m-1 of it.
  {
    const size_t apBytes = (size_t)PEARL_TALL_A_ROWS(profile->m) * k;
    CUDA_OK(cudaMalloc(&ctx->dAp[0], apBytes), "allocating the noised A");
    if (apBytes > aBytes) cudaMemset(ctx->dAp[0] + aBytes, 0, apBytes - aBytes);
    if (ctx->overlap) {
      CUDA_OK(cudaMalloc(&ctx->dAp[1], apBytes), "allocating the second noised A");
      if (apBytes > aBytes) cudaMemset(ctx->dAp[1] + aBytes, 0, apBytes - aBytes);
    }
  }
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
  CUDA_OK(cudaMalloc(&ctx->dHitTranscript[0],
                     (size_t)PEARL_MAX_HITS * PEARL_JACKPOT_BUCKETS * sizeof(uint32_t)),
          "allocating the hit transcripts");
  CUDA_OK(cudaMalloc(&ctx->dHashes[0], (size_t)PEARL_MAX_HITS * PEARL_HASH_BYTES),
          "allocating the batch hashes");
  CUDA_OK(cudaMalloc(&ctx->dHitCount[0], sizeof(uint32_t)), "allocating the hit counter");
  CUDA_OK(cudaMalloc(&ctx->dHitIndex[0], (size_t)PEARL_MAX_HITS * sizeof(uint32_t)),
          "allocating the hit list");
  if (ctx->overlap) {
    // The second hit list, the two streams, the events and the shadow: the host
    // overlap's own state (see Ctx). None of it exists with the switch off.
    CUDA_OK(cudaMalloc(&ctx->dHitTranscript[1],
                       (size_t)PEARL_MAX_HITS * PEARL_JACKPOT_BUCKETS * sizeof(uint32_t)),
            "allocating the second hit transcripts");
    CUDA_OK(cudaMalloc(&ctx->dHashes[1], (size_t)PEARL_MAX_HITS * PEARL_HASH_BYTES),
            "allocating the second batch hashes");
    CUDA_OK(cudaMalloc(&ctx->dHitCount[1], sizeof(uint32_t)),
            "allocating the second hit counter");
    CUDA_OK(cudaMalloc(&ctx->dHitIndex[1], (size_t)PEARL_MAX_HITS * sizeof(uint32_t)),
            "allocating the second hit list");
    CUDA_OK(cudaStreamCreateWithFlags(&ctx->sideStream, cudaStreamNonBlocking),
            "creating the restamp stream");
    CUDA_OK(cudaStreamCreateWithFlags(&ctx->readStream, cudaStreamNonBlocking),
            "creating the readback stream");
    CUDA_OK(cudaEventCreateWithFlags(&ctx->listEvent[0], cudaEventDisableTiming),
            "creating the first batch event");
    CUDA_OK(cudaEventCreateWithFlags(&ctx->listEvent[1], cudaEventDisableTiming),
            "creating the second batch event");
    CUDA_OK(cudaEventCreateWithFlags(&ctx->shadowEvent, cudaEventDisableTiming),
            "creating the shadow event");
    // A's tree has one level per halving of its (power-of-two, checked above)
    // chunk count, plus the leaves: what operand_commitment's layerOffA will hold.
    {
      const uint64_t aChunks = aBytes / 1024;
      uint32_t lg = 0;
      while ((1ull << lg) < aChunks) lg++;
      ctx->shadowLevels = lg + 1;
    }
    CUDA_OK(cudaMalloc(&ctx->dShadow,
                       1024 + (size_t)ctx->shadowLevels * PEARL_HASH_BYTES + PEARL_HASH_BYTES),
            "allocating the proof shadow");
  }
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

  // Which fold runs, before any operand is drawn (see resolve_fold).
  resolve_fold(ctx);

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
  if (ctx->overlap) {
    // A pre-launched batch may still be running, and a restamp may be on
    // sideStream; today's last search always waited for its own fold, so
    // nothing else here expects work in flight when the memory goes.
    cudaDeviceSynchronize();
  }
  cudaFree(ctx->dA); cudaFree(ctx->dB);
  cudaFree(ctx->dAp[0]); cudaFree(ctx->dBp);
  cudaFree(ctx->dEAL); cudaFree(ctx->dEBR);
  cudaFree(ctx->dPermA); cudaFree(ctx->dPermB);
  cudaFree(ctx->dLabelA); cudaFree(ctx->dLabelB);
  cudaFree(ctx->dSaltA); cudaFree(ctx->dSaltB);
  cudaFree(ctx->dBoundA); cudaFree(ctx->dBoundB);
  cudaFree(ctx->dRows); cudaFree(ctx->dCols);

  cudaFree(ctx->dHashes[0]); cudaFree(ctx->dHitCount[0]); cudaFree(ctx->dHitIndex[0]);
  cudaFree(ctx->dTreeA); cudaFree(ctx->dTreeB);
  cudaFree(ctx->dCvs); cudaFree(ctx->dSeedBuf); cudaFree(ctx->dSeedInput);
  cudaFree(ctx->dHashA); cudaFree(ctx->dHashB);
  cudaFree(ctx->dHitTranscript[0]); cudaFree(ctx->dJobKey);
  cudaFree(ctx->dASeed); cudaFree(ctx->dBSeed);
  cudaFree(ctx->dTarget); cudaFree(ctx->dHash); cudaFree(ctx->dIsShare);
  if (ctx->overlap) {
    cudaFree(ctx->dAp[1]);
    cudaFree(ctx->dHashes[1]); cudaFree(ctx->dHitCount[1]); cudaFree(ctx->dHitIndex[1]);
    cudaFree(ctx->dHitTranscript[1]);
    cudaFree(ctx->dShadow);
    if (ctx->listEvent[0]) cudaEventDestroy(ctx->listEvent[0]);
    if (ctx->listEvent[1]) cudaEventDestroy(ctx->listEvent[1]);
    if (ctx->shadowEvent) cudaEventDestroy(ctx->shadowEvent);
    if (ctx->sideStream) cudaStreamDestroy(ctx->sideStream);
    if (ctx->readStream) cudaStreamDestroy(ctx->readStream);
  }
  delete ctx;
}

extern "C" void pearl_host_reseed(void *handle, uint64_t salt);

// Load a job and draw its operands under `salt`.
//
// The salt is what keeps two cards off each other's work. One salt is worth m*n
// regions; a card searches those, then re-draws under the next salt it owns. Give
// every card a different starting salt and a stride equal to the number of cards
// and no two cards ever draw under the same salt, so no work and no share is done
// twice. Without it every card would start at salt 0 and walk 1, 2, 3 in step, and
// a second card would earn exactly nothing.
//
// Different salts mean different work because of A, not B. Every draw stamps the
// salt into A's first bytes (pearl_stamp_byte, one-to-one in the salt), so each
// salt gets its own root_A and so its own a_seed, and a_seed keys the search: the
// jackpot hash and A's noise. B can be shared. Under the constant fill
// (PEARL_OPERAND_CONST) every card draws the same B, root_B, b_seed and B' for a
// job, and A is the fill plus the stamp, so the stamp is all that keeps two cards
// apart. Under the hashed fill B differs between cards as well.
namespace {
void reseed_locked(Ctx *ctx, uint64_t salt);
}

extern "C" void pearl_host_set_job_salted(void *handle, const uint8_t *header,
                                          const uint8_t *target, uint64_t salt) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx) return;
  // Called on the JS thread while the search thread may be inside a search or a
  // reseed; the lock order is job_mu_ then mu (see Ctx).
  std::lock_guard<std::mutex> lock(ctx->mu);
  DeviceScope scope(ctx->device);
  memcpy(ctx->header, header, PEARL_HEADER_BYTES);
  memcpy(ctx->target, target, PEARL_HASH_BYTES);
  if (ctx->overlap) {
    // The next salt's restamp may be rewriting A's leaf 0 and tree on sideStream.
    // The full draw below rewrites both on the legacy stream, which does not wait
    // for a non-blocking stream, so wait it out here. Then forget everything that
    // belonged to the old job: the prepared salt, the shadow, and the in-flight
    // batch, which the next search drops unread because its jobGen differs. Its
    // fold is ahead of the full draw on the legacy stream, so the draw's writes
    // to dAp[apCur] wait for it.
    cudaStreamSynchronize(ctx->sideStream);
    if (ctx->inflight.valid) cudaEventSynchronize(ctx->listEvent[ctx->inflight.list]);
    ctx->prepared.valid = false;
    ctx->shadowValid = false;
    ctx->inflight.valid = false;
    ctx->jobGen++;
  }
  // A new job always gets a full draw. Clearing this is what stops a restamp
  // from grafting new bytes onto a tree the previous job's key built.
  ctx->baseDrawn = false;
  reseed_locked(ctx, salt);
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
//
// Meant for the synchronous path. With PEARL_HOST_OVERLAP=1 the restamp runs on
// sideStream beside a fold, and each lap here synchronises the whole device, so the
// stage times would include whatever the fold was doing: measure with the switch
// off, where they mean what they say.
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
//
// `stream` is 0 (the legacy default stream) for every draw a job starts with and
// for a synchronous reseed, and sideStream for a restamp prepared beside a fold
// (see Ctx). On stream 0 these asynchronous copies are the same device-to-device
// copies as before, on the same stream, in the same order.
void derive_seeds(Ctx *ctx, bool withB, cudaStream_t stream) {
  const bool legacy = ctx->profile.seed_derivation == PEARL_SEED_LEGACY;
  if (legacy) {
    cudaMemcpyAsync(ctx->dBoundA, ctx->dHashA, PEARL_HASH_BYTES, cudaMemcpyDeviceToDevice,
                    stream);
    if (withB)
      cudaMemcpyAsync(ctx->dBoundB, ctx->dHashB, PEARL_HASH_BYTES, cudaMemcpyDeviceToDevice,
                      stream);
  } else {
    pearl_bind_root<<<1, 1, 0, stream>>>(ctx->dSaltA, ctx->dHashA, ctx->profile.m,
                                         ctx->dBoundA);
    if (withB)
      pearl_bind_root<<<1, 1, 0, stream>>>(ctx->dSaltB, ctx->dHashB, ctx->profile.n,
                                           ctx->dBoundB);
  }
  if (withB) {
    cudaMemcpyAsync(ctx->dSeedBuf, ctx->dJobKey, PEARL_HASH_BYTES, cudaMemcpyDeviceToDevice,
                    stream);
    cudaMemcpyAsync(ctx->dSeedBuf + PEARL_HASH_BYTES, ctx->dBoundB, PEARL_HASH_BYTES,
                    cudaMemcpyDeviceToDevice, stream);
    pearl_blake3_unkeyed<<<1, 1, 0, stream>>>(ctx->dSeedBuf, 64,
                                              reinterpret_cast<uint8_t *>(ctx->dBSeed));
  }
  cudaMemcpyAsync(ctx->dSeedBuf, ctx->dBSeed, PEARL_HASH_BYTES, cudaMemcpyDeviceToDevice,
                  stream);
  cudaMemcpyAsync(ctx->dSeedBuf + PEARL_HASH_BYTES, ctx->dBoundA, PEARL_HASH_BYTES,
                  cudaMemcpyDeviceToDevice, stream);
  pearl_blake3_unkeyed<<<1, 1, 0, stream>>>(ctx->dSeedBuf, 64,
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
//
// `apBuf` is which noised-A buffer the A side writes (see Ctx::dAp); the B side
// has one. `stream` as in derive_seeds.
void draw_noise(Ctx *ctx, bool isA, cudaStream_t stream, int apBuf) {
  const uint32_t rank = ctx->profile.rank;
  const uint32_t k = ctx->profile.k;
  const uint32_t rows = isA ? ctx->profile.m : ctx->profile.n;
  const uint32_t *seed = isA ? ctx->dASeed : ctx->dBSeed;
  const uint8_t *label = isA ? ctx->dLabelA : ctx->dLabelB;
  int8_t *dense = isA ? ctx->dEAL : ctx->dEBR;
  uint32_t *perm = isA ? ctx->dPermA : ctx->dPermB;
  const int8_t *src = isA ? ctx->dA : ctx->dB;
  int8_t *dst = isA ? ctx->dAp[apBuf] : ctx->dBp;
  const size_t len = (size_t)rows * k;

  pearl_gen_dense<<<draw_blocks((size_t)rows * (rank / 32)), kDrawThreads, 0, stream>>>(
      seed, label, nullptr, dense, rows, rank);
  pearl_gen_perm<<<draw_blocks((k + 7) / 8), kDrawThreads, 0, stream>>>(seed, label, perm, k,
                                                                        rank);
  PEARL_LAP(2);
  if ((k & (k - 1u)) == 0u && k >= 16u) {
    uint32_t kLog2 = 0;
    while ((1u << kLog2) < k) kLog2++;
    if (ctx->foldTall && k >= PEARL_TALL_STAGE_K) {
      // The same values, stored [k / 64][rows][64] for the tall fold's staging: each
      // 64-byte stage of a tile is then whole L2 lines rather than half of every line,
      // for Blackwell's TMA boxes (PEARL_TALL_TMA) and the cp.async copies of Ada and
      // Ampere alike.
      // resolve_fold decided this for the context, before its first draw, and the
      // search launches the fold that reads it. (Any other k is one the search
      // refuses.)
      uint32_t kbLog2 = 0;
      while ((1u << kbLog2) < PEARL_TALL_STAGE_K) kbLog2++;
      pearl_materialize16_kblocked<<<draw_blocks(len / 16), kDrawThreads, 0, stream>>>(
          src, dense, perm, dst, rows, kLog2, rank, kbLog2);
    } else {
      pearl_materialize16<<<draw_blocks(len / 16), kDrawThreads, 0, stream>>>(
          src, dense, perm, dst, rows, kLog2, rank);
    }
  } else {
    pearl_materialize<<<draw_blocks(len), kDrawThreads, 0, stream>>>(src, dense, perm, dst,
                                                                     rows, k, rank);
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
  if (ctx->profile.operand_fill == PEARL_OPERAND_CONST) {
    // A constant costs the fold less energy than random bytes (about 3.5% in the
    // rate, see PEARL_OPERAND_CONST). Two memsets, cheaper than hashing.
    //
    // Then the salt stamp, the same bytes a restamp at this salt would write.
    // It is the only thing that differs between salts: skip it and every card of
    // a rig draws the same A and B for a new job, and they all search one space
    // until their first restamp. With it, a_seed depends on the job and the salt
    // alone, and the cards' salts never meet.
    cudaMemset(ctx->dA, PEARL_OPERAND_FILL, aLen);
    cudaMemset(ctx->dB, PEARL_OPERAND_FILL, bLen);
    int8_t stamp[PEARL_STAMP_BYTES];
    for (int i = 0; i < PEARL_STAMP_BYTES; i++) stamp[i] = pearl_stamp_byte(salt, i);
    cudaMemcpy(ctx->dA, stamp, sizeof(stamp), cudaMemcpyHostToDevice);
  } else {
    // Keyed by job_key rather than by a commitment seed, so these streams cannot
    // collide with the noise streams even though they share the labels.
    pearl_gen_operand<<<draw_blocks(aLen / 32 + 1), kDrawThreads>>>(
        ctx->dJobKey, ctx->dLabelA, ctx->dA, aLen, salt);
    pearl_gen_operand<<<draw_blocks(bLen / 32 + 1), kDrawThreads>>>(
        ctx->dJobKey, ctx->dLabelB, ctx->dB, bLen, salt);
  }

  // hash_a and hash_b: keyed BLAKE3 over the WHOLE operands. These are Merkle
  // trees over 1024-byte chunks, not one long chain — hashing them as a single
  // chunk (which this did until the device run showed a_seed == b_seed) gives the
  // wrong digest for anything over 1024 bytes and so the wrong seeds.
  operand_commitment(ctx, reinterpret_cast<const uint8_t *>(ctx->dA), aLen,
                     ctx->dHashA, ctx->dTreeA, &ctx->layerOffA);
  operand_commitment(ctx, reinterpret_cast<const uint8_t *>(ctx->dB), bLen,
                     ctx->dHashB, ctx->dTreeB, &ctx->layerOffB);

  // All on the legacy stream, into the live noised-A buffer: a full draw is what a
  // job starts with, and it queues behind any fold still running (see Ctx).
  derive_seeds(ctx, true, 0);
  draw_noise(ctx, true, 0, ctx->apCur);
  draw_noise(ctx, false, 0, ctx->apCur);

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
// different root_A. Across cards, it depends on the fill:
//   - hashed: each card's full draw uses its own salt, so B and b_seed already
//     differ. A stamp can only repeat the first draw's own random bytes by
//     chance, about 1 in 127^11.
//   - constant: B and b_seed are the same on every card, and the full draw
//     stamps A exactly as this does. So A is the fill plus stamp(salt) whichever
//     way it was drawn, and cards differ because their salts do.
//
// `stream` and `apBuf` as in draw_noise: the legacy stream and the live buffer for
// a synchronous reseed; sideStream and the other buffer for a restamp prepared
// beside a fold (see Ctx). Either way the stamp, the tree path and root_A are
// rewritten in place, which is why the prepared case saves the shadow first.
void restamp(Ctx *ctx, uint64_t salt, cudaStream_t stream, int apBuf) {
  const uint64_t aChunks = (uint64_t)ctx->profile.m * ctx->profile.k / 1024;
  PEARL_LAP(-1);
  pearl_restamp_operand<<<1, 1, 0, stream>>>(ctx->dJobKey, ctx->dA, salt, aChunks,
                                             ctx->dTreeA, ctx->dHashA);
  PEARL_LAP(0);
  derive_seeds(ctx, false, stream);
  PEARL_LAP(1);
  draw_noise(ctx, true, stream, apBuf);
  PEARL_LAP_REPORT();
}

// Whether a reseed may restamp A rather than draw everything: the current job has
// had its full draw, and A's tree has at least two leaves, so leaf 0 has a path to
// repair. PEARL_FULL_REDRAW forces the full draw every time (see
// pearl_host_reseed).
bool can_restamp(const Ctx *ctx) {
  const uint64_t aChunks = (uint64_t)ctx->profile.m * ctx->profile.k / 1024;
#ifdef PEARL_FULL_REDRAW
  (void)ctx;
  (void)aChunks;
  return false;
#else
  return ctx->baseDrawn && aChunks >= 2;
#endif
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
//
// With the host overlap on (see Ctx) the search usually has this salt PREPARED
// already: restamped into the other noised-A buffer while the previous salt's
// last batch ran, with its first batch in flight. Then this only swaps buffers.
// Only when nothing is prepared for this salt does it take the path below.
namespace {
void reseed_locked(Ctx *ctx, uint64_t salt) {
  DeviceScope scope(ctx->device);
  if (ctx->overlap) {
    if (ctx->prepared.valid && ctx->prepared.salt == salt) {
      // A, its tree, root_A, dASeed and dAp[prepared.apBuf] already belong to this
      // salt (pearl_host_search_next waited for that restamp before launching the
      // salt's first batch). The shadow held the previous salt's leaf 0, nodes and
      // root for a hit from its last batch, and that batch has been read by now;
      // from here the live arrays are this salt's, so the shadow goes.
      // PEARL_RESTAMP_CHECK does not run on this path; it stays with the
      // synchronous restamp below, where it was measured.
      ctx->apCur = ctx->prepared.apBuf;
      ctx->salt = salt;
      memcpy(ctx->aSeed, ctx->prepared.aSeed, PEARL_HASH_BYTES);
      ctx->prepared.valid = false;
      ctx->shadowValid = false;
      return;
    }
    // Nothing prepared for this salt: today's synchronous path. First wait out any
    // restamp still on sideStream, since the one below rewrites the same leaf, tree
    // path and root on the legacy stream, which does not wait for a non-blocking
    // stream. Whatever was prepared is for some other salt, so forget it, and the
    // in-flight batch is dropped unread: its A' is about to be rewritten under it.
    // Its fold is ahead of this restamp on the legacy stream, so the rewrite would
    // wait for it anyway; waiting here as well means nothing after this has to
    // know the batch existed (a later restamp on sideStream could otherwise be
    // writing a buffer it still reads).
    cudaStreamSynchronize(ctx->sideStream);
    if (ctx->inflight.valid) cudaEventSynchronize(ctx->listEvent[ctx->inflight.list]);
    ctx->prepared.valid = false;
    ctx->shadowValid = false;
    ctx->inflight.valid = false;
  }
  ctx->salt = salt;

  // A restamp needs a real tree to repair: at least two leaves, so leaf 0 has a
  // path. Profiles smaller than that always take the full draw.
  const bool canRestamp = can_restamp(ctx);
  if (canRestamp) {
    restamp(ctx, salt, 0, ctx->apCur);
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
}  // namespace

extern "C" void pearl_host_reseed(void *handle, uint64_t salt) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx) return;
  std::lock_guard<std::mutex> lock(ctx->mu);
  reseed_locked(ctx, salt);
}

namespace {

// The cluster build's launch configuration (PEARL_TALL_CLUSTER): `threads` a block,
// `smem` dynamic shared, the legacy default stream, and one attribute, clusters of
// `ctas` x 1 x 1. The grid is one cluster, which the occupancy query needs;
// launch_fold sets the real one. One place, so the occupancy query and the launch
// cannot disagree. `attr` must outlive the config.
void pearl_cluster_config(cudaLaunchConfig_t *cfg, cudaLaunchAttribute attr[1], unsigned ctas,
                          uint32_t threads, size_t smem) {
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = ctas;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg->gridDim = dim3(ctas);
  cfg->blockDim = dim3(threads);
  cfg->dynamicSmemBytes = smem;
  cfg->stream = 0;
  cfg->attrs = attr;
  cfg->numAttrs = 1;
}

// Everything a fold launch needs that does not change between batches: the
// geometry checks, the shared-memory opt-in and the resident block count, in the
// order pearl_host_search always ran them. Returns false with a message when the
// search must be refused.
struct FoldShape {
  uint32_t k = 0, rank = 0, chunks = 0;
  uint32_t regions = 0;     // what one batch tries, and what it reports
  uint32_t col_groups = 0;
  uint32_t threads = 0;
  unsigned tiles = 0, blocks = 0;
  size_t smem = 0;
};

bool fold_shape(Ctx *ctx, FoldShape *s, char *err, size_t err_len) {
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
  const uint32_t regions = ctx->batch;
  const uint32_t col_groups = ctx->colBatch;
  // Which fold this card loaded: resolved when the context was created (see
  // resolve_fold), because the operand draw has to know it too.
  if (!ctx->foldKnown) {
    if (err && err_len) snprintf(err, err_len, "%s", ctx->foldErr);
    return false;
  }
  // The block's shape: sixteen 32x64 warp tiles, or eight 64x64 ones. Either
  // way the CTA tile is 128x256, so the tile count, the grid and the shared
  // footprint below come out the same.
  // The tall fold (ctx->foldTall) is 256 threads over 192x256 tiles; its tile
  // count and shared footprint are worked out apart from these, below.
  const uint32_t threads = ctx->foldTall ? PEARL_TALL_THREADS
                           : ctx->foldWide ? PEARL_FOLD_WIDE_THREADS : PEARL_FOLD_THREADS;
  const uint32_t warpRows = ctx->foldWide ? PEARL_FOLD_WIDE_WARP_ROWS : PEARL_WARP_ROWS;
  const uint32_t rowTiles = ctx->foldWide ? PEARL_FOLD_WIDE_ROW_TILES : PEARL_WMMA_ROW_TILES;
  const void *foldFn = ctx->foldTall ? reinterpret_cast<const void *>(pearl_tile_fold_tall)
                                     : reinterpret_cast<const void *>(pearl_tile_fold_wmma);

  // One fused launch: the tile fold keeps its accumulator across chunks, so
  // there are no reusable partials to stage and no second pass.
  const uint32_t warpsPerBlock = threads / 32;
  const uint32_t regionsPerWarp = rowTiles * (PEARL_WMMA_ROWS / PEARL_ROWS_COUNT);
  // A block is a 2D grid of warps: warpRows down, the rest across. Both
  // dimensions have to tile exactly, because a warp that falls outside cannot
  // return -- staging is a block-wide cooperative load and a __syncthreads()
  // some warps skip hangs the launch.
  const uint64_t rowBlocks = ctx->rowsValid / regionsPerWarp;
  const uint32_t warpCols = warpsPerBlock / warpRows;
  const uint64_t colBlocks = col_groups / PEARL_WMMA_COL_BLK;
  // The staging walks base + stride rather than a table of addresses, which is
  // only the same sequence when quads divides the staging thread count and the
  // column step lands on a whole number of column groups.
  const uint32_t quadsPerRow = rank / 16u;
  const uint32_t stageThreads = threads;
  if (quadsPerRow == 0 || stageThreads % quadsPerRow != 0
      || (stageThreads / quadsPerRow) % PEARL_COLS_COUNT != 0) {
    if (err && err_len)
      snprintf(err, err_len,
               "staging stride is not uniform: %u threads, %u quads a row, %u columns a group",
               stageThreads, quadsPerRow, (unsigned)PEARL_COLS_COUNT);
    return false;
  }
  // The fold stages a tile's columns as one contiguous block, which holds only
  // while every batch starts on a whole column span (see PEARL_COLS_SPAN): four
  // valid column offsets for the strided pattern. col_off steps by col_groups
  // and wraps at colsValid, so both must be whole spans.
  const uint32_t colsPerSpan = PEARL_COLS_SPAN / PEARL_COLS_COUNT;
  if (col_groups % colsPerSpan != 0 || ctx->colsValid % colsPerSpan != 0) {
    if (err && err_len)
      snprintf(err, err_len,
               "column offsets must come in whole spans of %u: col_batch %u, %u valid",
               colsPerSpan, col_groups, ctx->colsValid);
    return false;
  }
  if (!ctx->foldTall && (warpsPerBlock % warpRows != 0 || rowBlocks % warpRows != 0
                         || warpCols == 0 || colBlocks % warpCols != 0)) {
    if (err && err_len)
      snprintf(err, err_len,
               "warp grid %ux%u does not tile %llu row blocks by %llu column blocks",
               warpRows, warpCols, (unsigned long long)rowBlocks,
               (unsigned long long)colBlocks);
    return false;
  }
  // The tall fold's tiles are 12 valid row offsets by 16 valid column offsets. The
  // columns must tile exactly; the last row group may run past m, into the rows
  // PEARL_TALL_A_ROWS pads the noised A with, and the fold hashes none of those.
  if (ctx->foldTall && (col_groups % PEARL_TALL_COL_OFFSETS != 0
                        || (uint64_t)PEARL_TALL_A_ROWS(ctx->profile.m) * k
                               > (uint64_t)0xFFFFFFFFu)) {
    if (err && err_len)
      snprintf(err, err_len, "tall fold: col_batch %u is not a whole number of %u-column tiles",
               col_groups, (unsigned)PEARL_TALL_BN);
    return false;
  }
  // The cluster build (ctx->foldCluster) walks tiles of a row-group PAIR by a column
  // group, one per cluster of two CTAs; the odd last pair's second CTA has no rows
  // and hashes nothing (see the kernel). Its grid is counted in CTAs, two a tile.
  const unsigned tallRowGroups =
      (unsigned)((ctx->rowsValid + PEARL_TALL_ROW_OFFSETS - 1u) / PEARL_TALL_ROW_OFFSETS);
  const unsigned ctasPerTile = ctx->foldCluster ? (unsigned)PEARL_TALL_CLUSTER_SIZE : 1u;
  const unsigned tiles =
      ctx->foldTall
          ? ((tallRowGroups + ctasPerTile - 1u) / ctasPerTile)
                * (unsigned)(col_groups / PEARL_TALL_COL_OFFSETS)
          : (unsigned)((rowBlocks / warpRows) * (colBlocks / warpCols));
  const unsigned tileCtas = tiles * ctasPerTile;
  // Two full-chunk stages; the transcripts live in registers and global now. The
  // tall fold: three 64-deep stages, its ring's barriers, and the transcripts, laid
  // out differently by the cp.async and TMA builds (see PEARL_TALL_SMEM).
  const size_t smem = ctx->foldTall
                          ? (ctx->foldTma ? (size_t)PEARL_TALL_SMEM_TMA : (size_t)PEARL_TALL_SMEM)
                          : (size_t)PEARL_STAGE_BUFS
                                * ((size_t)warpCols * PEARL_WMMA_COL_BLK * 16
                                   + (size_t)warpRows * regionsPerWarp * PEARL_ROWS_COUNT)
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
    // Refuse a footprint the card cannot grant rather than launch into an
    // invalid-configuration error: the fold's static shared counts too.
    int optin = 0;
    cudaFuncAttributes fs{};
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, ctx->device);
    cudaFuncGetAttributes(&fs, foldFn);
    if (optin > 0 && smem + fs.sharedSizeBytes > (size_t)optin) {
      if (err && err_len)
        snprintf(err, err_len, "fold needs %zu B of shared a block (+%zu static), card allows %d",
                 smem, (size_t)fs.sharedSizeBytes, optin);
      return false;
    }
    const cudaError_t se =
        cudaFuncSetAttribute(foldFn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (se != cudaSuccess) {
      if (err && err_len)
        snprintf(err, err_len, "fold shared-memory opt-in (%zu B) failed: %s", smem,
                 cudaGetErrorString(se));
      return false;
    }
    ctx->smemOptedIn = true;
  }
  // The fold is persistent: exactly as many blocks as can be resident, each
  // walking tiles a grid-width apart, so no tile starts with its first chunk
  // exposed (see the kernel). Launching more would only queue blocks behind
  // the resident ones and bring the exposed starts back. Asked once per
  // context, after the opt-in, because the answer depends on the footprint.
  // Only the persistent build (ctx->foldPersistent, above) uses it.
  if (ctx->foldResident == 0) {
    int sms = 0, perSm = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, ctx->device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, foldFn, (int)threads, smem);
    ctx->foldResident = (unsigned)(sms > 0 ? sms : 1) * (unsigned)(perSm > 0 ? perSm : 1);
    if (ctx->foldCluster) {
      // In CTAs, but whole clusters of them: what the runtime says fit at once with the
      // launch's own configuration (the same block size, shared footprint and cluster
      // shape as launch_fold), or the card's resident block count rounded down to a
      // multiple of the cluster size if it cannot say.
      cudaLaunchConfig_t cfg = {};
      cudaLaunchAttribute attr[1] = {};
      pearl_cluster_config(&cfg, attr, PEARL_TALL_CLUSTER_SIZE, threads, smem);
      int clusters = 0;
      if (cudaOccupancyMaxActiveClusters(&clusters, foldFn, &cfg) == cudaSuccess && clusters > 0) {
        ctx->foldResident = (unsigned)clusters * PEARL_TALL_CLUSTER_SIZE;
      } else {
        // A failed query is recorded as the thread's last error, and the search reads
        // cudaGetLastError after its first launch; drop it, or the fallback fails the
        // search it exists for.
        (void)cudaGetLastError();
        ctx->foldResident -= ctx->foldResident % PEARL_TALL_CLUSTER_SIZE;
      }
      if (ctx->foldResident == 0) ctx->foldResident = PEARL_TALL_CLUSTER_SIZE;
    }
  }
  const unsigned blocks =
      ((ctx->foldPersistent || ctx->foldTall) && ctx->foldResident < tileCtas) ? ctx->foldResident
                                                                              : tileCtas;
  s->k = k;
  s->rank = rank;
  s->chunks = chunks;
  s->regions = regions;
  s->col_groups = col_groups;
  s->threads = threads;
  s->tiles = tiles;
  s->blocks = blocks;
  s->smem = smem;
  return true;
}

// One fold over the batch at nonce_base, reading dAp[apBuf] under `key` and
// appending its hits to list `list`, which is cleared first. Always on the legacy
// default stream (see Ctx): a full draw from the JS thread and a synchronous
// reseed queue behind it there, as they always have.
void launch_fold(Ctx *ctx, const FoldShape &s, uint64_t nonce_base, int apBuf, int list,
                 const uint8_t *key) {
  // A valid-offset INDEX; the kernel expands it into an actual offset. From THIS
  // batch's nonce_base, which for a pre-launched batch is not the caller's.
  const uint32_t col_off = (uint32_t)((nonce_base / ctx->rowsValid) % ctx->colsValid);
  // The fold hashes every transcript itself and tests it against the bound. It
  // writes only on a hit and appends to a compact list, so the readback is four
  // bytes rather than one flag per region.
  //
  // The key and target go in as words, by value (see PearlTranscriptTest):
  // a_seed as the little-endian words BLAKE3 keys with, the target as
  // big-endian words so the kernel compares whole words most significant first.
  PearlTranscriptTest test;
  memcpy(test.key, key, sizeof(test.key));
  for (int i = 0; i < 8; i++) {
    const uint8_t *t = ctx->target + i * 4;
    test.target_w[i] = ((uint32_t)t[0] << 24) | ((uint32_t)t[1] << 16) |
                       ((uint32_t)t[2] << 8) | (uint32_t)t[3];
  }
  test.hash_big_endian = (int)ctx->profile.hash_big_endian;
  PearlHitList hitList;
  hitList.count = ctx->dHitCount[list];
  hitList.index = ctx->dHitIndex[list];
  hitList.hash = reinterpret_cast<uint32_t *>(ctx->dHashes[list]);
  hitList.transcript = ctx->dHitTranscript[list];
  cudaMemsetAsync(ctx->dHitCount[list], 0, sizeof(uint32_t));
  if (ctx->foldTall && ctx->foldCluster) {
    // The cluster build: the same launch, in clusters of PEARL_TALL_CLUSTER_SIZE CTAs
    // (s.blocks is a multiple of it, see fold_shape), the same arguments by value, on
    // the same stream. cudaLaunchKernelEx coerces each argument to the kernel's
    // parameter type, the tensor maps included.
    cudaLaunchConfig_t cfg = {};
    cudaLaunchAttribute attr[1] = {};
    pearl_cluster_config(&cfg, attr, PEARL_TALL_CLUSTER_SIZE, s.threads, s.smem);
    cfg.gridDim = dim3(s.blocks);
    cudaLaunchKernelEx(&cfg, pearl_tile_fold_tall, ctx->dAp[apBuf], ctx->dBp, ctx->profile.m,
                       ctx->profile.n, s.k, s.rank, s.chunks, col_off, ctx->rowsValid,
                       s.col_groups, s.tiles, test, hitList, ctx->tmA[apBuf], ctx->tmB);
  } else if (ctx->foldTall)
    pearl_tile_fold_tall<<<s.blocks, s.threads, s.smem>>>(
        ctx->dAp[apBuf], ctx->dBp, ctx->profile.m, ctx->profile.n, s.k, s.rank, s.chunks,
        col_off, ctx->rowsValid, s.col_groups, s.tiles, test, hitList, ctx->tmA[apBuf],
        ctx->tmB);
  else
    pearl_tile_fold_wmma<<<s.blocks, s.threads, s.smem>>>(
        ctx->dAp[apBuf], ctx->dBp, ctx->profile.m, ctx->profile.n, s.k, s.rank, s.chunks,
        col_off, ctx->rowsValid, s.col_groups, s.tiles, test, hitList);
}

// Host overlap only: launch a batch, record the event its readback waits on, and
// hand the other list to the next launch.
void launch_batch(Ctx *ctx, const FoldShape &s, const Ctx::Batch &b) {
  launch_fold(ctx, s, b.nonce, b.apBuf, b.list, b.aSeed);
  cudaEventRecord(ctx->listEvent[b.list], 0);
  ctx->nextList = b.list ^ 1;
}

// A finished batch with `hits` in its list: build the result for the lowest region
// index, and capture its proof NOW, while the operands and tree still belong to
// this hit. A few tens of milliseconds later they will have been re-drawn.
//
// The salt and a_seed are the batch's own (Ctx::Batch), not the context's: with
// the host overlap on, the context may already name the prepared salt.
bool build_hit(Ctx *ctx, const Ctx::Batch &b, uint32_t hits, PearlSearchResult *out) {
  const uint32_t n_hits = hits < PEARL_MAX_HITS ? hits : PEARL_MAX_HITS;
  copy_out(ctx, ctx->hHitIndex.data(), ctx->dHitIndex[b.list],
           (size_t)n_hits * sizeof(uint32_t));
  copies_done(ctx);
  // The kernel appends with an atomic, so the list is in an arbitrary order.
  // Take the LOWEST region index, which is what a sequential scan would have
  // returned — otherwise which share gets submitted varies run to run.
  uint32_t best = 0;
  for (uint32_t i = 1; i < n_hits; i++) {
    if (ctx->hHitIndex[i] < ctx->hHitIndex[best]) best = i;
  }
  copy_out(ctx, out->jackpot_hash, ctx->dHashes[b.list] + (size_t)best * PEARL_HASH_BYTES,
           PEARL_HASH_BYTES);
  memcpy(out->a_seed, b.aSeed, PEARL_HASH_BYTES);
  memcpy(out->b_seed, ctx->bSeed, PEARL_HASH_BYTES);
  out->nonce = b.nonce + ctx->hHitIndex[best];
  out->salt = b.salt;

  // Leaf 0, the node-0s and root_A come from the shadow when the live ones have
  // already been restamped for the next salt (see Ctx, "The shadow"). The
  // readback stream waits for the save first; the save was also waited for on the
  // host before the restamp's a_seed was read, so this is belt and braces.
  uint8_t *shadow = (ctx->overlap && ctx->shadowValid && ctx->shadowSalt == b.salt)
                        ? ctx->dShadow
                        : nullptr;
  if (shadow) cudaStreamWaitEvent(ctx->readStream, ctx->shadowEvent, 0);
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
                  &out->proof_a.leaves, &out->proof_a.siblings, shadow);
    snapshotProof(ctx, false, cols, PEARL_COLS_COUNT, &out->proof_bt.leaf_indices,
                  &out->proof_bt.leaves, &out->proof_bt.siblings, nullptr);
    copy_out(ctx, out->proof_a.root,
             shadow ? shadow_root(shadow, ctx->layerOffA.size()) : ctx->dHashA,
             PEARL_HASH_BYTES);
    copy_out(ctx, out->proof_bt.root, ctx->dHashB, PEARL_HASH_BYTES);
    out->proof_a.total_leaves = (uint64_t)ctx->profile.m * ctx->profile.k / 1024;
    out->proof_bt.total_leaves = (uint64_t)ctx->profile.n * ctx->profile.k / 1024;
  }
  out->proof.assign(PEARL_JACKPOT_BUCKETS * 4, 0);
  // Indexed by the hit's SLOT, like the hash: the fold keeps no per-region
  // transcripts, only the ones that hit.
  copy_out(ctx, out->proof.data(),
           ctx->dHitTranscript[b.list] + (size_t)best * PEARL_JACKPOT_BUCKETS,
           PEARL_JACKPOT_BUCKETS * 4);
  copies_done(ctx);
  out->found = true;
  return true;
}

// Today's search, call for call: one fold on the legacy stream, a synchronous copy
// of its hit count, then the hits. What every rig runs unless PEARL_HOST_OVERLAP=1.
// Buffer 0 and list 0 are the only ones there are on this path.
bool search_sync(Ctx *ctx, const FoldShape &s, uint64_t nonce_base, PearlSearchResult *out,
                 uint64_t *attempts, char *err, size_t err_len) {
  Ctx::Batch b;
  b.valid = true;
  b.nonce = nonce_base;
  b.salt = ctx->salt;
  memcpy(b.aSeed, ctx->aSeed, PEARL_HASH_BYTES);
  b.apBuf = 0;
  b.list = 0;
  b.jobGen = ctx->jobGen;
  launch_fold(ctx, s, nonce_base, 0, 0, ctx->aSeed);

  uint32_t hits = 0;
  cudaMemcpy(&hits, ctx->dHitCount[0], sizeof(uint32_t), cudaMemcpyDeviceToHost);

  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    if (err && err_len)
      snprintf(err, err_len, "CUDA error during search: %s", cudaGetErrorString(e));
    return false;
  }
  if (attempts) *attempts = s.regions;

  if (hits > 0) return build_hit(ctx, b, hits, out);
  out->found = false;
  return false;
}

// Host overlap only: copy aside what the next salt's restamp is about to rewrite
// (see Ctx, "The shadow"), on sideStream ahead of that restamp, and mark the copy
// as salt `salt`'s. The event is what a reader waits on.
void save_shadow(Ctx *ctx, uint64_t salt) {
  const size_t levels = ctx->layerOffA.size();
  cudaMemcpyAsync(shadow_leaf0(ctx->dShadow), ctx->dA, 1024, cudaMemcpyDeviceToDevice,
                  ctx->sideStream);
  for (size_t L = 0; L < levels; L++)
    cudaMemcpyAsync(shadow_node0(ctx->dShadow, L), ctx->dTreeA + ctx->layerOffA[L] * 8,
                    PEARL_HASH_BYTES, cudaMemcpyDeviceToDevice, ctx->sideStream);
  cudaMemcpyAsync(shadow_root(ctx->dShadow, levels), ctx->dHashA, PEARL_HASH_BYTES,
                  cudaMemcpyDeviceToDevice, ctx->sideStream);
  cudaEventRecord(ctx->shadowEvent, ctx->sideStream);
  ctx->shadowValid = true;
  ctx->shadowSalt = salt;
}

// The search with the host overlap on (see Ctx). One batch is kept in flight:
//
//   1. the CURRENT batch, at nonce_base, is the in-flight one when that was
//      launched for this nonce, salt and job; otherwise any stale in-flight batch
//      is waited for and dropped, and this one is launched now;
//   2. the NEXT batch is launched behind it on the legacy stream: the next nonce
//      under the same salt, or, at the salt's last batch when the caller named the
//      next salt, that salt's first batch, after its restamp has been run on
//      sideStream into the other noised-A buffer. The host waits for that restamp
//      before launching, because the fold takes the new a_seed by value;
//   3. only then is the current batch's hit count read, on readStream, behind the
//      event its fold recorded, so the read does not wait for the batch launched in
//      step 2.
//
// Which nonce the next batch gets is fixed by the caller's contract: a batch
// always runs its whole `regions` and reports exactly that, and the caller
// advances by what it reports. A caller that does anything else simply misses
// the in-flight batch at step 1 and pays a launch, as today.
bool search_overlapped(Ctx *ctx, const FoldShape &s, uint64_t nonce_base, int have_next,
                       uint64_t next_salt, PearlSearchResult *out, uint64_t *attempts,
                       char *err, size_t err_len) {
  Ctx::Batch cur;
  Ctx::Batch &in = ctx->inflight;
  if (in.valid && in.nonce == nonce_base && in.salt == ctx->salt && in.jobGen == ctx->jobGen) {
    cur = in;
    in.valid = false;
  } else {
    if (in.valid) {
      // Another nonce, a salt no reseed swapped to, or an older job. Nothing reads
      // it, but its list is reused below and its fold is ahead of everything that
      // follows on the legacy stream, so wait for it rather than reason about it.
      cudaEventSynchronize(ctx->listEvent[in.list]);
      in.valid = false;
    }
    cur.valid = true;
    cur.nonce = nonce_base;
    cur.salt = ctx->salt;
    memcpy(cur.aSeed, ctx->aSeed, PEARL_HASH_BYTES);
    cur.apBuf = ctx->apCur;
    cur.list = ctx->nextList;
    cur.jobGen = ctx->jobGen;
    launch_batch(ctx, s, cur);
  }

  // The next batch. A caller with no next salt (the bench) walks nonce_base past
  // the span and lets col_off wrap, as it always has, so it is kept fed the same
  // way.
  const uint64_t span = (uint64_t)ctx->rowsValid * ctx->colsValid;
  const uint64_t nextNonce = nonce_base + s.regions;
  if (nextNonce < span || !have_next) {
    Ctx::Batch nx = cur;
    nx.nonce = nextNonce;
    nx.list = ctx->nextList;
    launch_batch(ctx, s, nx);
    in = nx;
  } else if (ctx->prepared.valid) {
    // Prepared on an earlier call whose pre-launched batch was dropped. The arrays
    // already hold that salt, so there is nothing to restamp; its first batch is
    // relaunched if it is the salt asked for, and otherwise reseed will take the
    // synchronous path.
    if (ctx->prepared.salt == next_salt) {
      Ctx::Batch nx;
      nx.valid = true;
      nx.nonce = 0;
      nx.salt = next_salt;
      memcpy(nx.aSeed, ctx->prepared.aSeed, PEARL_HASH_BYTES);
      nx.apBuf = ctx->prepared.apBuf;
      nx.list = ctx->nextList;
      nx.jobGen = ctx->jobGen;
      launch_batch(ctx, s, nx);
      in = nx;
    }
  } else if (can_restamp(ctx) && ctx->layerOffA.size() <= ctx->shadowLevels
             && cur.salt == ctx->salt) {
    // The salt's last batch: prepare the next salt beside it. The shadow first,
    // on the same stream as the restamp so it reads the values the restamp is
    // about to replace; then the restamp into the other buffer, whose last reader
    // was the previous salt's last batch, read on an earlier call. The fold has
    // its key by value, so dASeed and dHashA may change under it, and B is not
    // touched. Then wait for the restamp here: the next batch's key is the new
    // a_seed, and the host needs it to launch.
    const int other = 1 - ctx->apCur;
    save_shadow(ctx, cur.salt);
    restamp(ctx, next_salt, ctx->sideStream, other);
    cudaMemcpyAsync(ctx->prepared.aSeed, ctx->dASeed, PEARL_HASH_BYTES,
                    cudaMemcpyDeviceToHost, ctx->sideStream);
    cudaStreamSynchronize(ctx->sideStream);
    ctx->prepared.valid = true;
    ctx->prepared.salt = next_salt;
    ctx->prepared.apBuf = other;
    Ctx::Batch nx;
    nx.valid = true;
    nx.nonce = 0;
    nx.salt = next_salt;
    memcpy(nx.aSeed, ctx->prepared.aSeed, PEARL_HASH_BYTES);
    nx.apBuf = other;
    nx.list = ctx->nextList;
    nx.jobGen = ctx->jobGen;
    launch_batch(ctx, s, nx);
    in = nx;
  }
  // Otherwise nothing is pre-launched: the last batch of a salt with no next one,
  // or a profile that cannot restamp. reseed then draws synchronously, as today.

  // Now the current batch's hits, behind ITS fold and nothing later.
  cudaStreamWaitEvent(ctx->readStream, ctx->listEvent[cur.list], 0);
  uint32_t hits = 0;
  cudaMemcpyAsync(&hits, ctx->dHitCount[cur.list], sizeof(uint32_t), cudaMemcpyDeviceToHost,
                  ctx->readStream);
  cudaStreamSynchronize(ctx->readStream);

  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    if (err && err_len)
      snprintf(err, err_len, "CUDA error during search: %s", cudaGetErrorString(e));
    return false;
  }
  if (attempts) *attempts = s.regions;

  if (hits > 0) return build_hit(ctx, cur, hits, out);
  out->found = false;
  return false;
}

}  // namespace

// Search the batch at nonce_base. Returns true and fills `out` on a hit; `attempts`
// is the batch's region count either way. `have_next_salt` and `next_salt` tell the
// host overlap (PEARL_HOST_OVERLAP=1, see Ctx) which salt the caller will reseed to
// when this salt's regions run out, so it can be prepared beside the last batch;
// with the switch off they are ignored, and this is pearl_host_search.
extern "C" bool pearl_host_search_next(void *handle, uint64_t nonce_base, uint32_t batch,
                                       int have_next_salt, uint64_t next_salt,
                                       PearlSearchResult *out, uint64_t *attempts,
                                       char *err, size_t err_len) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (attempts) *attempts = 0;
  if (!ctx || !out) return false;
  // Held for the whole call, switch on or off (see Ctx, "Locking"), so set_job's
  // full draw on the JS thread cannot land in the middle of a launch; set_job then
  // waits for the search in progress instead of queuing behind its fold on the
  // stream.
  std::lock_guard<std::mutex> lock(ctx->mu);
  if (!ctx->haveJob) return false;
  // A no-op on the search thread, which is already bound to this card; the
  // guard is for any other caller.
  DeviceScope scope(ctx->device);

  // The caller's batch hint is advisory; the real width is the context's, since
  // the partial table was allocated for exactly that many column groups.
  (void)batch;
  FoldShape shape;
  if (!fold_shape(ctx, &shape, err, err_len)) return false;

  if (!ctx->overlap) return search_sync(ctx, shape, nonce_base, out, attempts, err, err_len);
  return search_overlapped(ctx, shape, nonce_base, have_next_salt, next_salt, out, attempts,
                           err, err_len);
}

// The spelling the bench probe and pearl_core.cc were built against: no next salt,
// so nothing is prepared ahead of a reseed.
extern "C" bool pearl_host_search(void *handle, uint64_t nonce_base,
                                  uint32_t batch, PearlSearchResult *out,
                                  uint64_t *attempts, char *err,
                                  size_t err_len) {
  return pearl_host_search_next(handle, nonce_base, batch, 0, 0, out, attempts, err, err_len);
}

// Which fold this context's card runs, as resolve_fold read it off the loaded
// binaries, for the bench probe to print: host-pass macros describe the host compile,
// not the cubin that runs, and would credit a reading to the wrong fold.
extern "C" const char *pearl_host_fold_name(void *handle) {
  const Ctx *ctx = static_cast<const Ctx *>(handle);
  if (!ctx || !ctx->foldKnown) return "unresolved";
  if (ctx->foldTall && ctx->foldCluster)
    return "tall 192x256, 8 warps of 96x64, TMA ring, k-blocked operands, "
           "2-CTA cluster sharing B by multicast";
  if (ctx->foldTall)
    return ctx->foldTma ? "tall 192x256, 8 warps of 96x64, TMA ring, k-blocked operands"
                        : "tall 192x256, 8 warps of 96x64, cp.async ring";
  if (ctx->foldWide) return "wmma 128x256, 8 warps of 64x64";
  return ctx->foldPersistent ? "wmma 128x256, 16 warps of 32x64, persistent"
                             : "wmma 128x256, 16 warps of 32x64";
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
