// Host-side driver for the Pearl CUDA core: owns device memory, runs the kernel
// pipeline, and implements the four extern "C" entry points pearl_core.cc calls.
// Without this the addon compiles and fails to link on four undefined symbols.
//
// MEMORY BUDGET, because it is the design constraint that shapes everything
// here. The mainnet profile is m=131072, n=262144, k=2048. Under the hashed fill
// a full int8 A (m×k) is 256 MiB and Bᵀ (n×k) 512 MiB, before noise; the
// constant fill stores both compact (see pearl_host_create). That fits a
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

#include <algorithm>
#include <chrono>
#include <set>
#include <vector>

#include <cstdlib>
#include <cstdio>

#include "pearl_config.h"
#include "pearl_tensor_map.h"

// How the search thread waits for a batch: spinning, as the synchronous copy it
// replaced did. cudaEventBlockingSync, which sleeps instead, measured 0.4% slower on a
// 4090 with the next batch already queued: the fold itself ran slower by the GPU's own
// timer, at the same clock. Why is not known.
#ifndef PEARL_SLOT_EVENT_FLAGS
#define PEARL_SLOT_EVENT_FLAGS cudaEventDisableTiming
#endif
// The pipeline's two slots on two streams, so one batch starts on the SMs the one before
// it has already left (Ctx::foldStream). 0 puts both on the default stream.
#ifndef PEARL_FOLD_STREAMS
#define PEARL_FOLD_STREAMS 1
#endif
// Report every hit a batch holds, not only its lowest region (pearl_host_next_hit).
#ifndef PEARL_ALL_HITS
#define PEARL_ALL_HITS 1
#endif

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
                                               uint32_t rank, uint64_t read_vecs,
                                               uint32_t fill_word);
extern "C" __global__ void pearl_materialize16_kblocked(const int8_t *base,
                                                        const int8_t *dense,
                                                        const uint32_t *perm, int8_t *out,
                                                        uint32_t rows, uint32_t k_log2,
                                                        uint32_t rank, uint32_t kb_log2,
                                                        uint64_t read_vecs, uint32_t fill_word);
extern "C" __global__ void pearl_materialize16_tiled(const int8_t *base,
                                                     const int8_t *dense,
                                                     const uint32_t *perm, int8_t *out,
                                                     uint32_t rows, uint32_t k_log2,
                                                     uint32_t rank, uint32_t kb_log2,
                                                     uint32_t block_rows, uint64_t read_vecs,
                                                     uint32_t fill_word);
extern "C" __global__ void pearl_noise_materialize_kblocked(
    const uint32_t *seed, const uint8_t *label, const uint32_t *perm, const int8_t *base,
    int8_t *out, uint32_t rows, uint32_t k_log2, uint32_t rank, uint32_t kb_log2,
    uint64_t read_vecs, uint32_t fill_word, uint32_t block_rows);
extern "C" __global__ void pearl_restamp_operand(const uint32_t *key,
                                                 int8_t *operand, uint64_t salt,
                                                 uint64_t chunks, uint32_t *tree,
                                                 uint8_t *root_out);
extern "C" __global__ void pearl_restamp_commit(const PearlRestampRecord rec, int8_t *operand,
                                                uint32_t *tree, uint8_t *root_out,
                                                uint8_t *bound_out, uint32_t *a_seed_out);
#if PEARL_TURING_BDIRECT
// Built only with the B-direct fold, the only reader of its layout.
extern "C" __global__ void pearl_materialize16_frag(const int8_t *base,
                                                    const int8_t *dense,
                                                    const uint32_t *perm, int8_t *out,
                                                    uint32_t rows, uint32_t k_log2,
                                                    uint32_t rank, uint64_t read_vecs,
                                                    uint32_t fill_word);
#endif
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
#if PEARL_HOPPER_WGMMA
// Hopper's wgmma fold (PEARL_HOPPER_WGMMA; a body only in an sm_90a build with the switch).
extern "C" __global__ void pearl_tile_fold_hopper(uint32_t k_arg, uint32_t rank_arg,
                                                  uint32_t chunks_arg, uint32_t col_off,
                                                  uint32_t rows_valid, uint32_t col_groups,
                                                  uint32_t tiles, uint32_t band_depth, uint4 *trg,
                                                  const PearlTensorMap tmA,
                                                  const PearlTensorMap tmB);
#endif
// The transcript hash after a fold that stores its transcripts: GA100's unfused fold
// (PEARL_TALL_UNFUSED, sm_80) and Hopper's wgmma fold (PEARL_HOPPER_WGMMA, sm_90a). Its
// body is empty in every other build.
extern "C" __global__ void pearl_tall_hash80(const uint4 *tr, uint32_t regions,
                                             const PearlTranscriptTest test,
                                             const PearlHitList hits, uint32_t one);
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
                                                  uint32_t *cvs_out, uint32_t compact);
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
  // Whether it is Turing's build (binaryVersion 75): 256 threads, and a fold
  // that fits the 64 KB of shared Turing grants a block (PEARL_TURING_BDIRECT
  // and PEARL_TURING_WIDE pick which).
  bool foldTuring = false;
  // Whether that is the B-direct fold (PEARL_TURING_BDIRECT), which reads B'
  // fragment-ordered. The operand draw writes this card's B' that way exactly
  // when this is set. It is per context, so in a rig with a 2080 Ti beside a
  // 3090 or a 4090 each card's B' is in the layout its own fold reads.
  bool foldBDirect = false;
  // The B-direct fold's band depth on this card (see PEARL_BD_L2_SHARE).
  uint32_t bdBand = 0;
  // pearl_host_fold_name's text when it has to be formatted.
  char foldName[224] = {0};
  // Whether this card runs the tall fold instead (PEARL_FOLD_TALL): its own
  // kernel, pearl_tile_fold_tall, over 192x256 tiles. Read off that kernel's
  // loaded binary (PEARL_TALL_ARCH of its binaryVersion) with the others.
  bool foldTall = false;
  // Whether that is Blackwell's build, which stages with TMA (PEARL_TALL_TMA),
  // through the tensor maps below, and reads A' and B' k-blocked. Ada's reads them
  // in per-tile order (foldTiled, PEARL_TALL_TILE_ORDER). The operand draw writes the
  // order the fold reads; it runs before any search, so all of this is resolved when
  // the context is created (resolve_fold) and never changes after.
  bool foldTma = false;
  bool foldTiled = false;
  // Whether that TMA build is the two-CTA cluster one (PEARL_TALL_CLUSTER, off by
  // default): the tall fold is then launched in clusters of PEARL_TALL_CLUSTER_SIZE
  // over tiles of two row groups, and its resident count is in clusters.
  bool foldCluster = false;
  // Whether it is Ampere's cp.async build (binaryVersion 80 or 86, and Hopper's 90), whose
  // band depth the host picks from the L2 (PEARL_AMPERE_BAND_L2_SHARE), and the depth it
  // picked.
  bool foldAmpere = false;
  // Whether that build's A copies carry the evict_last hint the persisting slice is for
  // (PEARL_AMPERE_PERSIST_ARCH: Ampere's, not Hopper's).
  bool foldPersistA = false;
  uint32_t tallBand = 0;
  // Whether that is GA100's unfused fold (binaryVersion 80, PEARL_TALL_UNFUSED): it writes
  // each slot's transcripts to dTrG[slot], 64 bytes a region of the batch, and
  // pearl_tall_hash80 hashes them after it on the slot's stream.
  bool foldUnfused = false;
  // Whether Hopper's wgmma fold runs instead (PEARL_HOPPER_WGMMA): pearl_tile_fold_hopper
  // on the operands through tmA and tmB, per-tile (128-row A blocks) or k-blocked as
  // PEARL_HOPPER_TILED says, writing dTrG[slot] for pearl_tall_hash80, on hopperBlocks
  // CTAs (one an SM).
  bool foldHopper = false;
  unsigned hopperBlocks = 0;
  // With PEARL_HOPPER_CLUSTER: CTAs a launch, whole clusters of two (asked on first launch).
  unsigned hopperClusterCtas = 0;
  uint4 *dTrG[2] = {nullptr, nullptr};
  bool foldKnown = false;
  // Why not, when foldKnown is false: reported by the first search, as it was when
  // the search itself asked.
  char foldErr[256] = {0};
  // TMA descriptors for the k-blocked noised operands, encoded once against dAp and
  // dBp (which never move) when foldTma. Zero, and ignored, for every other build.
  // cuTensorMapEncodeTiled wants the map it writes 64-byte aligned. The host pass
  // sees PearlTensorMap unaligned (MSVC cannot pass an over-aligned kernel
  // parameter by value; see pearl_tensor_map.h), so the alignment is declared on
  // the members instead, which MSVC allows.
  alignas(128) PearlTensorMap tmA{}, tmB{};

  // Operands, generated once per job and then read by every region.
  int8_t *dA = nullptr;   // [m, k]
  int8_t *dB = nullptr;   // [n, k]  (Bᵀ, row-major)
  // COMPACT operands (pearl_compact_operands): under the constant fill every 1024-byte
  // chunk of A but chunk 0 (the salt stamp's) is the fill, and every chunk of B is. So
  // each is stored as two chunks, chunk 0 and the fill chunk, and read through
  // operand_chunk. Nothing reads them whole: the noised operands are drawn from the fill
  // (read_vecs), the tree hashes the two chunks, and a proof's leaves are copied chunk by
  // chunk. That is m*k + n*k bytes not allocated, 512 MiB at m = n = 131072.
  bool compact = false;

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

  // THE PIPELINE. The search keeps kSlots batches queued on the device, so the fold
  // never waits for the host between launches: batch j+1 is already queued when the
  // host reads batch j's hits. Each slot has its own hit list, and the hit count comes
  // back through pinned memory behind an event, not a synchronising copy. Before this
  // the GPU sat idle ~0.1 ms at every launch (WDDM submission plus the round trip),
  // 0.36% of the time at four launches a salt; after it, ~2 us.
  static const int kSlots = 2;
  struct Pending {
    uint64_t nonceBase;
    uint64_t salt;
    uint8_t aSeed[PEARL_HASH_BYTES];
    uint32_t regions;
  };
  Pending pend[kSlots];
  int pendHead = 0, pendCount = 0;
  cudaEvent_t slotDone[kSlots] = {};
  uint32_t *hSlotCount = nullptr;    // pinned, [kSlots]
  // Proof reads run here, not on the search's stream: that stream holds the next
  // batch, and a copy queued behind it would wait for it and then leave the GPU idle.
  cudaStream_t side = nullptr;
  // The stream each slot's batches run on. Blocking streams, so the redraw, which runs
  // on the legacy default stream, still waits for every queued batch and every later
  // batch waits for it; two batches of one salt may overlap. With PEARL_FOLD_STREAMS 0
  // both are the default stream and batches run strictly one after another.
  cudaStream_t foldStream[kSlots] = {};
  // Batches to submit since a queued redraw (see pearl_host_reseed). Both batches queued
  // behind a redraw wait for it, so when it finishes they start together and share the
  // SMs: each takes two batch periods and they finish within ~0.1 ms of each other. The
  // GPU loses nothing, but the core's hashrate windows, which count work as batches
  // complete, then alternate (on a 4060, 3 and 5 batches a window around 4). So the
  // second batch after a redraw waits for the first to finish, and they run in order.
  int afterRedraw = 0;
  // The collected batch's hits after the first, for pearl_host_next_hit.
  std::vector<PearlSearchResult> extraHits;
  size_t extraNext = 0;

  // Host-side restamps (host_record). What the job's full draw leaves for them: job_key,
  // leaf 0's bytes and node 1 of every level (the siblings of leaf 0's path, which no
  // restamp changes). hostSeeds is set only once a record the host hashed matched the
  // device's own root and a_seed for the job's first draw; until then every restamp
  // takes the device path and synchronises, as before.
  bool hostSeeds = false;
  uint32_t jobKeyW[8] = {0};
  uint8_t leaf0[1024] = {0};
  uint32_t sib[PEARL_RESTAMP_MAX_LEVELS][8] = {};
  // The records of the last few salts. A hit's proof is read while later work may
  // already have restamped A, so its salt-dependent part (leaf 0's head, node 0 of each
  // level, the root) comes from its own salt's record instead of from the device.
  static const int kRecords = 4;
  PearlRestampRecord rec[kRecords] = {};
  uint64_t recSalt[kRecords] = {0};
  bool recValid[kRecords] = {false, false, false, false};
  int recNext = 0;
};

// Whether A and B are stored compact (Ctx::compact): the constant fill, at a k every
// materialise path reads through read_vecs (a power of two, at least 16).
bool pearl_compact_operands(const PearlProfile *profile) {
  const uint32_t k = profile->k;
  return profile->operand_fill == PEARL_OPERAND_CONST && k >= 16u && (k & (k - 1u)) == 0u;
}

// Where chunk `i` of an operand lives: its own place, or one of the two a compact
// operand stores.
const int8_t *operand_chunk(const Ctx *ctx, const int8_t *operand, uint64_t i) {
  return ctx->compact ? operand + (i == 0 ? 0 : 1024) : operand + i * 1024;
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
      ctx->dJobKey, data, chunks, dst, ctx->compact ? 1u : 0u);

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

// ---------------------------------------------------------------------------
// BLAKE3 on the host, for host_record only: one chunk CV, parent compressions and
// two one-block hashes a redraw, about 40 compressions, some tens of microseconds on
// one core. The device's BLAKE3 (pearl_kernel.cu) is the one every other hash uses;
// this one is checked against it once a job (see setup_host_seeds) and not trusted
// until it agrees.
// ---------------------------------------------------------------------------
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

// The CV of a whole 1024-byte chunk at `counter`, keyed.
void hb3_chunk_cv(const uint32_t key[8], const uint8_t *chunk, uint64_t counter,
                  uint32_t out[8]) {
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

// BLAKE3 of exactly 64 bytes as a root: keyed when key is given, else unkeyed.
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

// The record a restamp at `salt` produces, hashed on the host: leaf 0 with the stamp
// written over it (or as it stands, when stamp is false: the check against the job's
// first draw), its path to the root with the stored siblings, the binding, a_seed.
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

// A device-to-host copy on the side stream, so it does not queue behind the fold.
void side_copy(Ctx *ctx, void *dst, const void *src, size_t n) {
  cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, ctx->side);
  cudaStreamSynchronize(ctx->side);
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
//
// `rec` is the record of the hit's salt, when there is one. Later work may already have
// restamped A for the next salt, so the parts a restamp changes -- leaf 0's head and node
// 0 of every level -- come from the record; every other leaf and node is the same for
// every salt of the job.
void snapshotProof(Ctx *ctx, bool isA, const uint32_t *rows, uint32_t nrows,
                   std::vector<uint32_t> *leafIdx, std::vector<uint8_t> *leaves,
                   std::vector<uint8_t> *sibs, const PearlRestampRecord *rec) {
  if (!isA) rec = nullptr;
  const uint32_t k = ctx->profile.k;
  const int8_t *operand = isA ? ctx->dA : ctx->dB;
  const uint32_t *tree = isA ? ctx->dTreeA : ctx->dTreeB;
  const std::vector<uint64_t> &offs = isA ? ctx->layerOffA : ctx->layerOffB;
  const uint64_t totalLeaves =
      (uint64_t)(isA ? ctx->profile.m : ctx->profile.n) * k / 1024;

  leafIndicesForRows(rows, nrows, k, leafIdx);

  leaves->resize(leafIdx->size() * 1024);
  for (size_t i = 0; i < leafIdx->size(); i++) {
    side_copy(ctx, leaves->data() + i * 1024, operand_chunk(ctx, operand, (*leafIdx)[i]), 1024);
    if (rec && (*leafIdx)[i] == 0) memcpy(leaves->data() + i * 1024, rec->head, sizeof rec->head);
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
      if (rec && want == 0)
        hb3_bytes(rec->path[level], node);
      else
        side_copy(ctx, node, tree + (offs[level] + want) * 8, PEARL_HASH_BYTES);
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

uint32_t pearl_ampere_col_batch(uint32_t colBatch, uint32_t colsValid);

// Two folds write every region's transcript to a buffer a pipeline slot, 64 bytes a region,
// for pearl_tall_hash80 to read: GA100's unfused fold (PEARL_TALL_UNFUSED) and Hopper's
// wgmma fold (PEARL_HOPPER_WGMMA, an sm_90a build). Which fold a card runs is known only
// once its fold binary has loaded (resolve_fold), so both VRAM checks ask by compute
// capability: 8.0 is GA100, the only card the release gives sm_80 code, and 9.0 is
// Hopper. A card the build counts but that runs another fold (an sm_90 build without the
// wgmma body, or a forced fold) is over-counted, which only makes the check stricter.
static bool transcript_buffer_card(int device) {
  if (!PEARL_TALL_UNFUSED && !PEARL_HOPPER_WGMMA) return false;
  int major = 0, minor = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess
      || cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device)
             != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  // Both are counted at the Ampere batch width (Hopper's fold runs only where foldAmpere
  // does). Hopper's own narrower width (PEARL_HOPPER_COL_BATCH) is over-counted, which only
  // makes the check stricter.
  return (PEARL_TALL_UNFUSED && major == 8 && minor == 0)
         || (PEARL_HOPPER_WGMMA && major == 9 && minor == 0);
}

// Those buffers' bytes: the batch width pearl_host_create settles on for Ampere (the
// profile's col_batch, at most the valid column offsets, then PEARL_AMPERE_COL_BATCH's
// narrower width) by the valid row offsets, 64 bytes a region, one buffer a slot. 512 MiB
// at the mainnet geometry.
static size_t unfused_transcript_bytes(const PearlProfile *profile) {
  const uint32_t rowsValid = profile->m / PEARL_ROWS_COUNT;
  const uint32_t colsValid = profile->n / PEARL_COLS_COUNT;
  uint32_t cb = profile->col_batch ? profile->col_batch : 1u;
  if (cb > colsValid) cb = colsValid;
  cb = pearl_ampere_col_batch(cb, colsValid);
  return (size_t)Ctx::kSlots * cb * rowsValid * 16u * sizeof(uint32_t);
}

// What one instance of `profile` costs on a card, in bytes. `unfused` is whether the
// card runs a fold that stores every transcript (transcript_buffer_card).
//
// Two callers ask: the pre-flight in pearl_host_create, and the device choice
// in pearl_host_select_device. They have to ask the SAME question — a card
// chosen against one number and then refused against another is the worst of
// both answers.
//
// The terms moved here wholesale from that pre-flight, comments and all; each
// one is a thing that was got wrong once.
size_t needed_bytes(const PearlProfile *profile, bool unfused) {
  const size_t k = profile->k;
  const size_t rank = profile->rank;
  const size_t aBytes = (size_t)profile->m * k;
  const size_t bBytes = (size_t)profile->n * k;
  const bool compact = pearl_compact_operands(profile);
  const size_t aStored = compact ? 2048 : aBytes, bStored = compact ? 2048 : bBytes;
  const size_t noiseBytes = (size_t)profile->m * rank + (size_t)profile->n * rank
                            + 2 * k * 2 * sizeof(uint32_t) + 64;
  // The materialised operands are int8, the same size as the sources. They were
  // int32 while the noise was (wrongly) reconstructed at full rank, which cost
  // 2 GiB at mainnet on top of the 1 GiB of sources. The noised A has the tall
  // fold's padding rows on the end (PEARL_TALL_A_ROWS): 128 KB at mainnet.
  const size_t primeBytes = (size_t)PEARL_TALL_A_ROWS(profile->m) * k + bBytes;
  // What a batch costs now: nothing per region, except on GA100 (trBytes below).
  // The fold hashes its own transcripts, so only a hit's transcript, hash and
  // index are stored, in a fixed PEARL_MAX_HITS list. This term used to be a
  // transcript PER REGION -- 1 GiB at the mainnet geometry -- and it stayed here
  // after the buffer went, so a card whose free VRAM the local LLM had taken could
  // be refused for a gigabyte the miner no longer asks for.
  // One hit list per pipeline slot (Ctx::kSlots).
  const size_t batchBytes =
      (size_t)Ctx::kSlots * PEARL_MAX_HITS
      * (PEARL_HASH_BYTES + 2 * sizeof(uint32_t) + PEARL_JACKPOT_BUCKETS * sizeof(uint32_t));
  // The kept commitment trees (just under 2 nodes a leaf, 32 bytes a node, for
  // both operands) and the leaf-CV scratch for the larger one. Real
  // allocations that were never counted: 40 MiB at the mainnet geometry.
  const size_t aLeaves = aBytes / 1024, bLeaves = bBytes / 1024;
  const size_t treeBytes = 2 * (aLeaves + bLeaves) * 32
                           + (aLeaves > bLeaves ? aLeaves : bLeaves) * 32;
  // GA100's unfused fold does keep a transcript a region, in a buffer a slot, for the
  // hash kernel that follows it (unfused_transcript_bytes). Left out, the pre-flight
  // passed a card that then failed on these buffers, the last allocation create makes.
  const size_t trBytes = unfused ? unfused_transcript_bytes(profile) : 0;
  return aStored + bStored + primeBytes + noiseBytes + batchBytes + treeBytes + trBytes
         + (1u << 20);
}

// pearl_host_create's check on each allocation, and the only place that uses it (after
// `ctx` exists). A failure frees what the context already holds: pearl_host_destroy takes
// a partly built one, since every member starts null. Without it those buffers stayed
// held until the process exited, and the restart's pre-flight refused the card for them.
// The failed call's error is cleared too, so nothing later on this thread reads it as its
// own.
extern "C" void pearl_host_destroy(void *handle);
#define CUDA_OK(expr, msg)                                   \
  do {                                                       \
    cudaError_t _e = (expr);                                 \
    if (_e != cudaSuccess) {                                 \
      if (err && err_len)                                    \
        snprintf(err, err_len, "%s: %s", msg,                \
                 cudaGetErrorString(_e));                    \
      pearl_host_destroy(ctx);                               \
      (void)cudaGetLastError();                              \
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
  const bool adaLayout =
      haveAttrs && (PEARL_AMPERE_ARCH(fa.binaryVersion * 10) || fa.binaryVersion == 89);
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
  // Turing's fold has its own block size (PEARL_FOLD_TURING_THREADS); a wide
  // build forced onto every arch takes precedence, as it does in the kernel.
  ctx->foldTuring = haveAttrs && fa.binaryVersion == 75 && !ctx->foldWide;
  // The sm_75 pass builds the B-direct fold exactly when PEARL_TURING_BDIRECT
  // (PEARL_TURING_BD_BODY), which this pass reads the same, unless the block is
  // forced wide, which foldTuring already excludes.
  ctx->foldBDirect = ctx->foldTuring && PEARL_TURING_BDIRECT != 0;
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
    ctx->foldTiled = ctx->foldTall && !ctx->foldTma && PEARL_TALL_TILE_ORDER != 0;
    // -DPEARL_TALL_CLUSTER=1 builds that TMA fold as a two-CTA cluster; the host pass
    // sees the same value, so the launch shape follows the body.
    ctx->foldCluster = ctx->foldTma && PEARL_TALL_CLUSTER != 0;
    ctx->foldAmpere = ctx->foldTall && !ctx->foldTma && PEARL_AMPERE_ARCH(ft.binaryVersion * 10);
    ctx->foldPersistA = ctx->foldAmpere && PEARL_AMPERE_PERSIST_ARCH(ft.binaryVersion * 10);
    ctx->foldUnfused = ctx->foldAmpere && ft.binaryVersion == 80 && PEARL_TALL_UNFUSED != 0;
    // Hopper's wgmma fold: only when this binary has its body, which alone carries the
    // 288-thread launch bound (an sm_90a build with PEARL_HOPPER_WGMMA). It reads the
    // operands through TMA in the order its build has, per-tile in 128-row A blocks or
    // k-blocked (PEARL_HOPPER_TILED), and the draw writes them that way.
#if PEARL_HOPPER_WGMMA
    // Per-tile, the A' allocation (padded to 192-row tiles, PEARL_TALL_A_ROWS) must hold
    // whole 128-row blocks; at mainnet m is a multiple of both. Else the cp.async fold runs.
    const uint64_t hRows =
        ((uint64_t)ctx->profile.m + PEARL_HOPPER_BM - 1u) / PEARL_HOPPER_BM * PEARL_HOPPER_BM;
    if (ctx->foldAmpere && ft.binaryVersion == 90
        && (!PEARL_HOPPER_TILED || hRows <= PEARL_TALL_A_ROWS((uint64_t)ctx->profile.m))) {
      cudaFuncAttributes fh;
      int sms = 0;
      if (cudaFuncGetAttributes(&fh, reinterpret_cast<const void *>(pearl_tile_fold_hopper))
              == cudaSuccess
          && (uint32_t)fh.maxThreadsPerBlock == PEARL_HOPPER_THREADS
          && cudaFuncSetAttribute(reinterpret_cast<const void *>(pearl_tile_fold_hopper),
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  (int)PEARL_HOPPER_SMEM) == cudaSuccess
          && cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, ctx->device)
                 == cudaSuccess && sms > 0) {
        ctx->foldHopper = true;
        ctx->hopperBlocks = (unsigned)sms;
        // The draw follows the kernel's operand order, whatever PEARL_TALL_TILE_ORDER says.
        ctx->foldTiled = PEARL_HOPPER_TILED != 0;
      } else {
        (void)cudaGetLastError();
      }
    }
#endif
  }
  // The fold is compiled for exactly one block size, which is also its launch
  // bound. A disagreement would not fail loudly: a block of the wrong size
  // returns at once, and the search would report hashrate while finding
  // nothing. So refuse it.
  const uint32_t want = ctx->foldWide     ? PEARL_FOLD_WIDE_THREADS
                        : ctx->foldTuring ? PEARL_FOLD_TURING_THREADS
                                          : PEARL_FOLD_THREADS;
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
#if PEARL_HOPPER_WGMMA
  // Hopper's wgmma fold on per-tile operands (PEARL_HOPPER_TILED): each tile block's k-blocks
  // are consecutive boxes of 128 (A) or 256 (B) rows of 64 bytes, so the maps are
  // {64 bytes, box rows, blocks x k-blocks}, the A map over m rounded up to 128 rows.
  if (ctx->foldHopper && ctx->foldTiled) {
    PFN_cuTensorMapEncodeTiled_v12000 enc = pearl_encode_tiled();
    const uint32_t kbs = ctx->profile.k / PEARL_TALL_STAGE_K;
    bool ok = enc != nullptr && ctx->profile.k % PEARL_TALL_STAGE_K == 0u
              && ctx->profile.n % PEARL_TALL_BN == 0u;
    for (int side = 0; ok && side < 2; side++) {
      const uint32_t blockRows = side ? PEARL_TALL_BN : PEARL_HOPPER_BM;
      // With the cluster each CTA loads half of B's 256 rows (PEARL_HOPPER_CLUSTER).
      const uint32_t boxRows = side && PEARL_HOPPER_CLUSTER ? PEARL_TALL_BN / 2u : blockRows;
      const uint64_t blocks = side ? ctx->profile.n / PEARL_TALL_BN
                                   : (ctx->profile.m + PEARL_HOPPER_BM - 1u) / PEARL_HOPPER_BM;
      const cuuint64_t dims[3] = {(cuuint64_t)PEARL_TALL_STAGE_K, (cuuint64_t)blockRows,
                                  (cuuint64_t)(blocks * kbs)};
      const cuuint64_t strides[2] = {(cuuint64_t)PEARL_TALL_STAGE_K,
                                     (cuuint64_t)blockRows * PEARL_TALL_STAGE_K};
      const cuuint32_t box[3] = {PEARL_TALL_STAGE_K, boxRows, 1u};
      const cuuint32_t estr[3] = {1u, 1u, 1u};
      ok = enc(&(side ? ctx->tmB : ctx->tmA).map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3,
               side ? (void *)ctx->dBp : (void *)ctx->dAp, dims, strides, box, estr,
               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
               CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
           == CUDA_SUCCESS;
    }
    if (!ok) {
      snprintf(err, err_len, "could not encode the wgmma fold's per-tile tensor maps");
      return;
    }
  }
#endif
  if ((ctx->foldTma || (ctx->foldHopper && !ctx->foldTiled))
      && (!pearl_encode_operand(&ctx->tmA, ctx->dAp, ctx->profile.m, ctx->profile.k,
                                PEARL_TALL_STAGE_K,
#if PEARL_HOPPER_WGMMA
                                ctx->foldHopper ? PEARL_HOPPER_BM :
#endif
                                PEARL_TALL_BM)
          || !pearl_encode_operand(&ctx->tmB, ctx->dBp, ctx->profile.n, ctx->profile.k,
                                   PEARL_TALL_STAGE_K,
#if PEARL_HOPPER_WGMMA
                                   ctx->foldHopper && PEARL_HOPPER_CLUSTER ? PEARL_TALL_BN / 2u :
#endif
                                   PEARL_TALL_BN))) {
    snprintf(err, err_len, "could not encode the tall fold's TMA descriptors (k %u, %u-byte k-blocks)",
             ctx->profile.k, (unsigned)PEARL_TALL_STAGE_K);
    return;
  }
  ctx->foldKnown = true;
}

// The column offsets one batch of Blackwell's TMA fold covers (see PEARL_TMA_L2_SHARE):
// the profile's col_batch, halved while the B' a row group sweeps -- col_batch * 16
// columns of k bytes -- is more than PEARL_TMA_L2_SHARE percent of the L2. Never below
// one tile's 16 column offsets, and only to a width that divides the valid offsets, so
// a salt still splits into whole batches. -DPEARL_TMA_COL_BATCH=N forces N instead.
uint32_t pearl_tma_col_batch(uint32_t colBatch, uint32_t colsValid, uint32_t k, uint64_t l2) {
#ifdef PEARL_TMA_COL_BATCH
  (void)k;
  (void)l2;
  const uint32_t forced = (uint32_t)PEARL_TMA_COL_BATCH;
  if (forced >= PEARL_TALL_COL_OFFSETS && forced <= colBatch && forced % PEARL_TALL_COL_OFFSETS == 0u
      && colsValid % forced == 0u)
    return forced;
  return colBatch;
#else
  if (l2 == 0u) return colBatch;
  const uint64_t colBytes = (uint64_t)PEARL_COLS_COUNT * k;
  uint32_t cb = colBatch;
  while ((uint64_t)cb * colBytes * 100u > l2 * (uint64_t)PEARL_TMA_L2_SHARE && cb % 2u == 0u
         && (cb / 2u) % PEARL_TALL_COL_OFFSETS == 0u && colsValid % (cb / 2u) == 0u)
    cb /= 2u;
  return cb;
#endif
}

// Turing's band depth for the B-direct fold (see PEARL_BD_L2_SHARE). Bands of
// column groups (PEARL_BD_WALK 1): 8, halved while a band's B -- depth x 256
// columns x k bytes -- is more than PEARL_BD_L2_SHARE percent of the L2, never
// below 2. Bands of row groups (PEARL_BD_WALK 0): 16, halved while a band's A --
// depth x 128 rows x k bytes -- is, never below 4. -DPEARL_BD_BAND=N forces N.
uint32_t pearl_bd_band_for(uint32_t k, uint64_t l2) {
#ifdef PEARL_BD_BAND
  (void)k;
  (void)l2;
  return (uint32_t)PEARL_BD_BAND;
#else
  const uint64_t unit = (PEARL_BD_WALK ? 256u : 128u) * (uint64_t)k;
  const uint32_t floor = PEARL_BD_WALK ? 2u : 4u;
  uint32_t band = PEARL_BD_WALK ? 8u : 16u;
  if (l2 == 0u) return band;
  while (band > floor && (uint64_t)band * unit * 100u > l2 * (uint64_t)PEARL_BD_L2_SHARE)
    band /= 2u;
  return band;
#endif
}

// The column offsets one batch of Ampere's tall fold covers (see PEARL_AMPERE_COL_BATCH):
// the profile's col_batch, or PEARL_AMPERE_COL_BATCH if that is narrower, is a whole
// number of tiles' 16 column offsets, and divides the valid offsets, so a salt still
// splits into whole batches.
uint32_t pearl_ampere_col_batch(uint32_t colBatch, uint32_t colsValid) {
  const uint32_t want = (uint32_t)PEARL_AMPERE_COL_BATCH;
  if (want >= PEARL_TALL_COL_OFFSETS && want < colBatch && want % PEARL_TALL_COL_OFFSETS == 0u
      && colsValid % want == 0u)
    return want;
  return colBatch;
}

// Ampere's band depth for the tall fold (see PEARL_AMPERE_BAND_L2_SHARE): 16, halved
// while a band's A' -- depth x 192 rows x k bytes -- is more than
// PEARL_AMPERE_BAND_L2_SHARE percent of the L2, never below 4. GA100's unfused fold
// (`ga100`) goes to 32 where that A' is at most PEARL_GA100_BAND32_L2_SHARE percent of
// the L2. -DPEARL_AMPERE_BAND=N forces N, a power of two.
uint32_t pearl_ampere_band_for(uint32_t k, uint64_t l2, bool ga100) {
#ifdef PEARL_AMPERE_BAND
  (void)k;
  (void)l2;
  (void)ga100;
  return (uint32_t)PEARL_AMPERE_BAND;
#else
  uint32_t band = 16u;
  if (l2 == 0u) return band;
  if (ga100 && PEARL_GA100_BAND32_L2_SHARE != 0u
      && 32ull * PEARL_TALL_BM * k * 100u <= l2 * (uint64_t)PEARL_GA100_BAND32_L2_SHARE)
    return 32u;
  while (band > 4u
         && (uint64_t)band * PEARL_TALL_BM * k * 100u > l2 * (uint64_t)PEARL_AMPERE_BAND_L2_SHARE)
    band /= 2u;
  return band;
#endif
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
    double bestScore = -1.0;
    for (int d = 0; d < devices; d++) {
      cudaDeviceProp prop;
      if (cudaGetDeviceProperties(&prop, d) != cudaSuccess) continue;
      const size_t need = needed_bytes(profile, transcript_buffer_card(d));
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
  int current = 0;
  if (cudaGetDevice(&current) != cudaSuccess) current = 0;
  const size_t need = needed_bytes(profile, transcript_buffer_card(current));

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

  ctx->compact = pearl_compact_operands(profile);
  CUDA_OK(cudaMalloc(&ctx->dA, ctx->compact ? 2048 : aBytes), "allocating A");
  CUDA_OK(cudaMalloc(&ctx->dB, ctx->compact ? 2048 : bBytes), "allocating B");
  // The tall fold's last row group of tiles reads past A's m rows (see
  // PEARL_TALL_A_ROWS): the cp.async build (Ada and Ampere) up to 64 rows past the
  // end of the last k-block. Nothing generates the bytes past m * k; they are zeroed
  // once so the fold reads defined bytes there, and it hashes no region from them.
  {
    const size_t apBytes = (size_t)PEARL_TALL_A_ROWS(profile->m) * k;
    CUDA_OK(cudaMalloc(&ctx->dAp, apBytes), "allocating the noised A");
    if (apBytes > aBytes) cudaMemset(ctx->dAp + aBytes, 0, apBytes - aBytes);
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
  // mainnet geometry. GA100 is the exception: its unfused fold leaves the hashing
  // to a kernel of its own, so it keeps 64 bytes a region again, in a buffer a
  // slot (dTrG, allocated once the width is final; needed_bytes counts it).
  // One hit list per pipeline slot: a batch's hits must survive until the host reads
  // them, which is after the next batch has been queued.
  const size_t S = Ctx::kSlots;
  CUDA_OK(cudaMalloc(&ctx->dHitTranscript,
                     S * PEARL_MAX_HITS * PEARL_JACKPOT_BUCKETS * sizeof(uint32_t)),
          "allocating the hit transcripts");
  CUDA_OK(cudaMalloc(&ctx->dHashes, S * PEARL_MAX_HITS * PEARL_HASH_BYTES),
          "allocating the batch hashes");
  // Turing's B-direct fold keeps each slot's tile counter (PEARL_BD_PERSIST) and
  // its band depth (PEARL_BD_L2_SHARE) after the hit counters: [hit counts][tile
  // counters][band depths], one word a slot each.
  static_assert(PEARL_BD_CTR_SLOTS == Ctx::kSlots && PEARL_BD_BAND_WORD == 2u * Ctx::kSlots,
                "the B-direct fold's words, one a pipeline slot");
  // Ampere's tall fold reads its band depth from the same word (PEARL_AMPERE_BAND_L2_SHARE).
  CUDA_OK(cudaMalloc(&ctx->dHitCount, S * 3u * sizeof(uint32_t)),
          "allocating the hit counter");
  CUDA_OK(cudaMalloc(&ctx->dHitIndex, S * PEARL_MAX_HITS * sizeof(uint32_t)),
          "allocating the hit list");
  CUDA_OK(cudaHostAlloc(&ctx->hSlotCount, S * sizeof(uint32_t), cudaHostAllocDefault),
          "allocating the pinned hit counts");
  for (int i = 0; i < Ctx::kSlots; i++)
    CUDA_OK(cudaEventCreateWithFlags(&ctx->slotDone[i], PEARL_SLOT_EVENT_FLAGS),
            "creating the batch events");
  CUDA_OK(cudaStreamCreateWithFlags(&ctx->side, cudaStreamNonBlocking),
          "creating the proof stream");
#if PEARL_FOLD_STREAMS
  for (int i = 0; i < Ctx::kSlots; i++)
    CUDA_OK(cudaStreamCreateWithFlags(&ctx->foldStream[i], cudaStreamDefault),
            "creating the fold streams");
#endif
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
  // Blackwell (the TMA fold): a batch narrow enough that the B' it sweeps fits the L2.
  // See PEARL_TMA_L2_SHARE.
  if (ctx->foldTma) {
    int l2 = 0;
    if (cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, ctx->device) != cudaSuccess) {
      (void)cudaGetLastError();
      l2 = 0;
    }
    const uint32_t cb = pearl_tma_col_batch(ctx->colBatch, ctx->colsValid, profile->k, (uint64_t)l2);
    if (cb != ctx->colBatch) {
      ctx->colBatch = cb;
      ctx->batch = ctx->colBatch * ctx->rowsValid;
    }
  }
  // Ampere (the sm_80, sm_86 and sm_90 tall fold): a band of row groups whose A' fits the
  // L2. See PEARL_AMPERE_BAND_L2_SHARE.
  if (ctx->foldAmpere) {
    // A narrower batch than the profile's (PEARL_AMPERE_COL_BATCH).
    {
      uint32_t cb = pearl_ampere_col_batch(ctx->colBatch, ctx->colsValid);
#if PEARL_HOPPER_WGMMA
      // Hopper's wgmma fold narrower still (PEARL_HOPPER_COL_BATCH).
      if (ctx->foldHopper) {
        const uint32_t hw = (uint32_t)PEARL_HOPPER_COL_BATCH;
        if (hw >= PEARL_TALL_COL_OFFSETS && hw < cb && hw % PEARL_TALL_COL_OFFSETS == 0u
            && ctx->colsValid % hw == 0u)
          cb = hw;
      }
#endif
      if (cb != ctx->colBatch) {
        ctx->colBatch = cb;
        ctx->batch = ctx->colBatch * ctx->rowsValid;
      }
    }
    int l2 = 0;
    if (cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, ctx->device) != cudaSuccess) {
      (void)cudaGetLastError();
      l2 = 0;
    }
    ctx->tallBand = pearl_ampere_band_for(profile->k, (uint64_t)l2, ctx->foldUnfused);
    // The fold reads it PEARL_BD_BAND_WORD words past its slot's hit counter.
    const uint32_t bands[Ctx::kSlots] = {ctx->tallBand, ctx->tallBand};
    CUDA_OK(cudaMemcpy(ctx->dHitCount + PEARL_BD_BAND_WORD, bands, sizeof bands,
                       cudaMemcpyHostToDevice),
            "setting the tall fold's band depth");
#if PEARL_AMPERE_PERSIST_A
    // A persisting L2 slice for the band's A' (PEARL_AMPERE_PERSIST_A). Without one the
    // fold still runs, only slower, so a refusal is dropped.
    int maxp = 0;
    if (!ctx->foldPersistA) {
      // Hopper: its A copies carry no hint, so a slice would only take L2 from the rest.
    } else if (cudaDeviceGetAttribute(&maxp, cudaDevAttrMaxPersistingL2CacheSize, ctx->device)
                   == cudaSuccess && maxp > 0) {
      size_t slice = (size_t)ctx->tallBand * PEARL_TALL_BM * profile->k;
      if (slice > (size_t)maxp) slice = (size_t)maxp;
      if (cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, slice) != cudaSuccess)
        (void)cudaGetLastError();
    } else {
      (void)cudaGetLastError();
    }
#endif
  }
  // GA100's unfused fold (PEARL_TALL_UNFUSED) and Hopper's wgmma fold (PEARL_HOPPER_WGMMA):
  // a transcript buffer a slot, 64 bytes a region of the batch. The batch width is final
  // here. The pre-flight counted these (transcript_buffer_card, unfused_transcript_bytes);
  // if they still do not fit, CUDA_OK frees the rest.
  if (ctx->foldUnfused || ctx->foldHopper) {
    static_assert(Ctx::kSlots == 2, "one transcript buffer a pipeline slot");
    for (int s = 0; s < Ctx::kSlots; s++)
      CUDA_OK(cudaMalloc(&ctx->dTrG[s], (size_t)ctx->batch * 16u * sizeof(uint32_t)),
              "allocating the transcript buffers (unfused or wgmma fold)");
  }
  // Turing (the B-direct fold): a band of row groups whose A fits the L2. See
  // PEARL_BD_L2_SHARE.
  if (ctx->foldBDirect) {
    int l2 = 0;
    if (cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, ctx->device) != cudaSuccess) {
      (void)cudaGetLastError();
      l2 = 0;
    }
    ctx->bdBand = pearl_bd_band_for(profile->k, (uint64_t)l2);
    // The fold reads it PEARL_BD_BAND_WORD words past its slot's hit counter.
    const uint32_t bands[Ctx::kSlots] = {ctx->bdBand, ctx->bdBand};
    CUDA_OK(cudaMemcpy(ctx->dHitCount + PEARL_BD_BAND_WORD, bands, sizeof bands,
                       cudaMemcpyHostToDevice),
            "setting the B-direct fold's band depth");
  }
  // In per-tile order A's padding rows sit inside every slab of its last block, not
  // after m * k, so the whole noised A is zeroed once.
  if (ctx->foldTiled)
    cudaMemset(ctx->dAp, 0, (size_t)PEARL_TALL_A_ROWS(profile->m) * profile->k);

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
  cudaDeviceSynchronize();  // nothing still queued may read what is freed below
  cudaFree(ctx->dA); cudaFree(ctx->dB);
  cudaFree(ctx->dAp); cudaFree(ctx->dBp);
  cudaFree(ctx->dEAL); cudaFree(ctx->dEBR);
  cudaFree(ctx->dPermA); cudaFree(ctx->dPermB);
  cudaFree(ctx->dLabelA); cudaFree(ctx->dLabelB);
  cudaFree(ctx->dSaltA); cudaFree(ctx->dSaltB);
  cudaFree(ctx->dBoundA); cudaFree(ctx->dBoundB);
  cudaFree(ctx->dRows); cudaFree(ctx->dCols);

  cudaFree(ctx->dHashes); cudaFree(ctx->dHitCount); cudaFree(ctx->dHitIndex);
  for (int s = 0; s < 2; s++) cudaFree(ctx->dTrG[s]);
  cudaFree(ctx->dTreeA); cudaFree(ctx->dTreeB);
  cudaFree(ctx->dCvs); cudaFree(ctx->dSeedBuf); cudaFree(ctx->dSeedInput);
  cudaFree(ctx->dHashA); cudaFree(ctx->dHashB);
  cudaFree(ctx->dHitTranscript); cudaFree(ctx->dJobKey);
  cudaFree(ctx->dASeed); cudaFree(ctx->dBSeed);
  cudaFree(ctx->dTarget); cudaFree(ctx->dHash); cudaFree(ctx->dIsShare);
  for (int i = 0; i < Ctx::kSlots; i++)
    if (ctx->slotDone[i]) cudaEventDestroy(ctx->slotDone[i]);
  if (ctx->side) cudaStreamDestroy(ctx->side);
  for (int i = 0; i < Ctx::kSlots; i++)
    if (ctx->foldStream[i]) cudaStreamDestroy(ctx->foldStream[i]);
  if (ctx->hSlotCount) cudaFreeHost(ctx->hSlotCount);
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
  const bool kPow2 = (k & (k - 1u)) == 0u && k >= 16u;
  uint32_t kLog2 = 0;
  while ((1u << kLog2) < k) kLog2++;
  uint32_t kbLog2 = 0;
  while ((1u << kbLog2) < PEARL_TALL_STAGE_K) kbLog2++;
  // Under the constant fill the operand is the fill everywhere except the stamp in
  // A's first PEARL_STAMP_BYTES bytes, so only the sixteen-byte groups holding it are
  // read (see the kernels): a restamp then reads nothing of A but that.
  const bool constFill = ctx->profile.operand_fill == PEARL_OPERAND_CONST;
  const uint64_t readVecs = constFill ? (PEARL_STAMP_BYTES + 15u) / 16u : (uint64_t)(len / 16);
  const uint32_t fillWord = (uint32_t)(uint8_t)PEARL_OPERAND_FILL * 0x01010101u;
  // The tall fold's operands (k-blocked, or per-tile on Ada), drawn by the fused kernel
  // whenever its shape allows (pearl_noise_materialize_kblocked): the mainnet geometry
  // always does.
  const bool tallLayout = ctx->foldTall && kPow2 && k >= PEARL_TALL_STAGE_K;
  const bool fused = tallLayout && k >= 64u && k <= 4096u && rank % 32u == 0u && rank <= 128u
                     && rows % 32u == 0u;

  if (!fused)
    pearl_gen_dense<<<draw_blocks((size_t)rows * (rank / 32)), kDrawThreads>>>(
        seed, label, nullptr, dense, rows, rank);
  pearl_gen_perm<<<draw_blocks((k + 7) / 8), kDrawThreads>>>(seed, label, perm, k, rank);
  PEARL_LAP(2);
  // The cp.async fold (Ada, and Ampere) reads them in per-tile order (foldTiled,
  // PEARL_TALL_TILE_ORDER):
  // blocks of a tile's 192 rows of A' or 256 columns of B', each k / 64 slabs. 0 is the
  // k-blocked order Blackwell's TMA reads.
  const uint32_t blockRows =
      ctx->foldTiled ? (isA ? (ctx->foldHopper ? PEARL_HOPPER_BM : PEARL_TALL_BM) : PEARL_TALL_BN)
                     : 0u;
#if PEARL_TURING_BDIRECT
  if (ctx->foldBDirect && !isA && kPow2 && rows % 16u == 0u && k >= 32u) {
    // The same values, in the fragment order Turing's B-direct fold loads B'
    // in (pearl_materialize16_frag). A' stays row-major, so a restamp, which
    // redraws only A', is unchanged. Decided per card when its context was
    // created (resolve_fold). The search refuses the B-direct fold for any
    // geometry this branch does not take. Turing has no tall fold, so `fused`
    // is never set here and pearl_gen_dense above has written the dense factor
    // this reads; under the constant fill it reads B through read_vecs, like
    // every other materialise.
    pearl_materialize16_frag<<<draw_blocks(len / 16), kDrawThreads>>>(
        src, dense, perm, dst, rows, kLog2, rank, readVecs, fillWord);
  } else
#endif
  if (fused) {
    pearl_noise_materialize_kblocked<<<rows / 32u, 256>>>(seed, label, perm, src, dst, rows,
                                                         kLog2, rank, kbLog2, readVecs,
                                                         fillWord, blockRows);
  } else if (tallLayout) {
    // The same values, stored for the tall fold's staging: each 64-byte stage of a tile
    // is then whole L2 lines rather than half of every line, for Blackwell's TMA boxes
    // (PEARL_TALL_TMA, k-blocked) and the cp.async copies of Ada and Ampere (per-tile)
    // alike.
    // resolve_fold decided this for the context, before its first draw, and the
    // search launches the fold that reads it. (Any other k is one the search
    // refuses.)
    if (ctx->foldTiled)
      pearl_materialize16_tiled<<<draw_blocks(len / 16), kDrawThreads>>>(
          src, dense, perm, dst, rows, kLog2, rank, kbLog2, blockRows, readVecs, fillWord);
    else
      pearl_materialize16_kblocked<<<draw_blocks(len / 16), kDrawThreads>>>(
          src, dense, perm, dst, rows, kLog2, rank, kbLog2, readVecs, fillWord);
  } else if (kPow2) {
    pearl_materialize16<<<draw_blocks(len / 16), kDrawThreads>>>(src, dense, perm, dst, rows,
                                                                 kLog2, rank, readVecs,
                                                                 fillWord);
  } else {
    pearl_materialize<<<draw_blocks(len), kDrawThreads>>>(src, dense, perm, dst, rows, k,
                                                          rank);
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
    cudaMemset(ctx->dA, PEARL_OPERAND_FILL, ctx->compact ? 2048 : aLen);
    cudaMemset(ctx->dB, PEARL_OPERAND_FILL, ctx->compact ? 2048 : bLen);
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
// different root_A. Across cards, it depends on the fill:
//   - hashed: each card's full draw uses its own salt, so B and b_seed already
//     differ. A stamp can only repeat the first draw's own random bytes by
//     chance, about 1 in 127^11.
//   - constant: B and b_seed are the same on every card, and the full draw
//     stamps A exactly as this does. So A is the fill plus stamp(salt) whichever
//     way it was drawn, and cards differ because their salts do.
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

// A record read back from the device, for a draw the host did not hash itself.
void device_record(Ctx *ctx, PearlRestampRecord *r) {
  const uint32_t levels = (uint32_t)ctx->layerOffA.size();
  memset(r, 0, sizeof *r);
  r->levels = levels < PEARL_RESTAMP_MAX_LEVELS ? levels : PEARL_RESTAMP_MAX_LEVELS;
  cudaMemcpy(r->head, ctx->dA, sizeof r->head, cudaMemcpyDeviceToHost);
  for (uint32_t L = 0; L < r->levels; L++) {
    r->node_off[L] = (uint32_t)ctx->layerOffA[L];
    cudaMemcpy(r->path[L], ctx->dTreeA + ctx->layerOffA[L] * 8, 32, cudaMemcpyDeviceToHost);
  }
  cudaMemcpy(r->bound, ctx->dBoundA, 32, cudaMemcpyDeviceToHost);
  hb3_words(ctx->aSeed, r->a_seed, 8);
}

// After a job's full draw: fetch what host_record needs -- job_key, leaf 0, and node 1
// of every level -- and check the host's hashing against the device's own root and
// a_seed for this draw. Only a match turns host-side restamps on for the job.
void setup_host_seeds(Ctx *ctx) {
  ctx->hostSeeds = false;
  for (int i = 0; i < Ctx::kRecords; i++) ctx->recValid[i] = false;
  const uint32_t levels = (uint32_t)ctx->layerOffA.size();
  PearlRestampRecord r;
  if (levels >= 2 && levels <= PEARL_RESTAMP_MAX_LEVELS) {
    cudaMemcpy(ctx->jobKeyW, ctx->dJobKey, 32, cudaMemcpyDeviceToHost);
    cudaMemcpy(ctx->leaf0, ctx->dA, sizeof ctx->leaf0, cudaMemcpyDeviceToHost);
    for (uint32_t L = 0; L + 1 < levels; L++)
      cudaMemcpy(ctx->sib[L], ctx->dTreeA + (ctx->layerOffA[L] + 1) * 8, 32,
                 cudaMemcpyDeviceToHost);
    host_record(ctx, ctx->salt, false, &r);
    uint8_t root[32], hroot[32], hseed[32];
    cudaMemcpy(root, ctx->dHashA, 32, cudaMemcpyDeviceToHost);
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
  if (canRestamp && ctx->hostSeeds) {
    // The host hashes the new path and a_seed itself (host_record), so nothing here
    // waits for the device: the commit and A's noise queue behind whatever batches are
    // already queued, and the next batch queues behind them with an a_seed the host
    // already holds. That wait was one synchronising copy a salt, and the serial
    // restamp and seed kernels were ~0.08 ms of a single device thread.
    PearlRestampRecord r;
    host_record(ctx, salt, true, &r);
    pearl_restamp_commit<<<1, 256>>>(r, ctx->dA, ctx->dTreeA, ctx->dHashA, ctx->dBoundA,
                                     ctx->dASeed);
    draw_noise(ctx, true);
    hb3_bytes(r.a_seed, ctx->aSeed);
    keep_record(ctx, salt, r);
    ctx->afterRedraw = 2;
  } else {
    if (canRestamp) {
      restamp(ctx, salt);
    } else {
      full_draw(ctx, salt);
    }
    // The one synchronising copy on this path: the search must not launch against
    // half-built seeds, and the host copy of a_seed travels with every hit.
    cudaMemcpy(ctx->aSeed, ctx->dASeed, PEARL_HASH_BYTES, cudaMemcpyDeviceToHost);
    if (canRestamp) {
      PearlRestampRecord r;
      device_record(ctx, &r);
      keep_record(ctx, salt, r);
    } else {
      setup_host_seeds(ctx);
    }
  }

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


namespace {
// One hit of a collected batch -- entry `i` of slot `slot`'s hit list -- as a result:
// its hash, region and seeds, the transcript that hashed to it, and its share proof.
// The proof is read now, while the tree still belongs to this job; within the job only
// leaf 0's head and node 0 of each level change between salts, and those come from the
// salt's own record (snapshotProof), since later work may already have restamped A.
void fill_hit(Ctx *ctx, const Ctx::Pending &p, int slot, uint32_t i, PearlSearchResult *out) {
  const uint8_t *dHashes = ctx->dHashes + (size_t)slot * PEARL_MAX_HITS * PEARL_HASH_BYTES;
  const uint32_t *dTranscripts =
      ctx->dHitTranscript + (size_t)slot * PEARL_MAX_HITS * PEARL_JACKPOT_BUCKETS;
  side_copy(ctx, out->jackpot_hash, dHashes + (size_t)i * PEARL_HASH_BYTES, PEARL_HASH_BYTES);
  memcpy(out->a_seed, p.aSeed, PEARL_HASH_BYTES);
  memcpy(out->b_seed, ctx->bSeed, PEARL_HASH_BYTES);
  out->nonce = p.nonceBase + ctx->hHitIndex[i];
  out->salt = p.salt;
  const PearlRestampRecord *rec = find_record(ctx, p.salt);

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
  for (int j = 0; j < PEARL_ROWS_COUNT; j++) rows[j] = rowOff | PEARL_ROWS_PATTERN[j];
  for (int j = 0; j < PEARL_COLS_COUNT; j++) cols[j] = colOff | PEARL_COLS_PATTERN[j];

  snapshotProof(ctx, true, rows, PEARL_ROWS_COUNT, &out->proof_a.leaf_indices,
                &out->proof_a.leaves, &out->proof_a.siblings, rec);
  snapshotProof(ctx, false, cols, PEARL_COLS_COUNT, &out->proof_bt.leaf_indices,
                &out->proof_bt.leaves, &out->proof_bt.siblings, nullptr);
  if (rec)
    hb3_bytes(rec->path[rec->levels - 1], out->proof_a.root);
  else
    side_copy(ctx, out->proof_a.root, ctx->dHashA, PEARL_HASH_BYTES);
  side_copy(ctx, out->proof_bt.root, ctx->dHashB, PEARL_HASH_BYTES);
  out->proof_a.total_leaves = (uint64_t)ctx->profile.m * ctx->profile.k / 1024;
  out->proof_bt.total_leaves = (uint64_t)ctx->profile.n * ctx->profile.k / 1024;

  out->proof.assign(PEARL_JACKPOT_BUCKETS * 4, 0);
  // Indexed by the hit's SLOT in the list, like the hash: the fold keeps no
  // per-region transcripts, only the ones that hit.
  side_copy(ctx, out->proof.data(), dTranscripts + (size_t)i * PEARL_JACKPOT_BUCKETS,
            PEARL_JACKPOT_BUCKETS * 4);
  out->found = true;
}


// The cluster build's launch configuration (PEARL_TALL_CLUSTER): `threads` a block,
// `smem` dynamic shared, stream `st`, and one attribute, clusters of `ctas` x 1 x 1.
// The grid is one cluster, which the occupancy query needs; the launch sets the real
// one. One place, so the occupancy query and the launch cannot disagree. `attr` must
// outlive the config.
void pearl_cluster_config(cudaLaunchConfig_t *cfg, cudaLaunchAttribute attr[1], unsigned ctas,
                          uint32_t threads, size_t smem, cudaStream_t st) {
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = ctas;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg->gridDim = dim3(ctas);
  cfg->blockDim = dim3(threads);
  cfg->dynamicSmemBytes = smem;
  cfg->stream = st;
  cfg->attrs = attr;
  cfg->numAttrs = 1;
}

#if PEARL_HOPPER_WGMMA
// Launch Hopper's wgmma fold over one batch on `st`. Its tiles are row groups of 8 row
// offsets (128 rows) by the batch's column groups: one CTA an SM, or with
// PEARL_HOPPER_CLUSTER clusters of two CTAs over row-group pairs, as many clusters as the
// runtime says fit at once (asked on the first launch).
void pearl_launch_hopper(Ctx *ctx, cudaStream_t st, uint32_t k, uint32_t rank, uint32_t chunks,
                         uint32_t col_off, uint32_t col_groups, uint4 *trg) {
  const unsigned rowGroups =
      (unsigned)((ctx->rowsValid + PEARL_HOPPER_ROW_OFFSETS - 1u) / PEARL_HOPPER_ROW_OFFSETS);
  const unsigned colGroups = (unsigned)(col_groups / PEARL_TALL_COL_OFFSETS);
#if PEARL_HOPPER_CLUSTER
  const unsigned htiles = (rowGroups + 1u) / 2u * colGroups;   // pair tiles
  if (ctx->hopperClusterCtas == 0) {
    cudaLaunchConfig_t cfg = {};
    cudaLaunchAttribute attr[1] = {};
    pearl_cluster_config(&cfg, attr, 2u, PEARL_HOPPER_THREADS, PEARL_HOPPER_SMEM, 0);
    int clusters = 0;
    if (cudaOccupancyMaxActiveClusters(&clusters, pearl_tile_fold_hopper, &cfg) == cudaSuccess
        && clusters > 0) {
      ctx->hopperClusterCtas = 2u * (unsigned)clusters;
    } else {
      (void)cudaGetLastError();
      ctx->hopperClusterCtas = ctx->hopperBlocks & ~1u;
    }
    if (ctx->hopperClusterCtas < 2u) ctx->hopperClusterCtas = 2u;
  }
  const unsigned ctas = ctx->hopperClusterCtas < 2u * htiles ? ctx->hopperClusterCtas : 2u * htiles;
  cudaLaunchConfig_t cfg = {};
  cudaLaunchAttribute attr[1] = {};
  pearl_cluster_config(&cfg, attr, 2u, PEARL_HOPPER_THREADS, PEARL_HOPPER_SMEM, st);
  cfg.gridDim = dim3(ctas);
  cudaLaunchKernelEx(&cfg, pearl_tile_fold_hopper, k, rank, chunks, col_off, ctx->rowsValid,
                     col_groups, htiles, ctx->tallBand, trg, ctx->tmA, ctx->tmB);
#else
  const unsigned htiles = rowGroups * colGroups;
  pearl_tile_fold_hopper<<<ctx->hopperBlocks < htiles ? ctx->hopperBlocks : htiles,
                           PEARL_HOPPER_THREADS, PEARL_HOPPER_SMEM, st>>>(
      k, rank, chunks, col_off, ctx->rowsValid, col_groups, htiles, ctx->tallBand, trg,
      ctx->tmA, ctx->tmB);
#endif
}
#endif
}  // namespace

// Queue one batch at nonce_base, under the current salt, into the next free pipeline
// slot, without waiting for anything. pearl_host_collect returns its results, oldest
// batch first; at most Ctx::kSlots batches are queued at once.
extern "C" bool pearl_host_submit(void *handle, uint64_t nonce_base, uint32_t batch,
                                  uint64_t *regions_out, char *err, size_t err_len) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (regions_out) *regions_out = 0;
  if (!ctx || !ctx->haveJob) return false;
  if (ctx->pendCount >= Ctx::kSlots) {
    if (err && err_len) snprintf(err, err_len, "search pipeline full: collect a batch first");
    return false;
  }
  // A no-op on the search thread, which is already bound to this card; the
  // guard is for any other caller.
  DeviceScope scope(ctx->device);
  const int slot = (ctx->pendHead + ctx->pendCount) % Ctx::kSlots;
  // Each slot's batches run on the slot's own stream, so a batch can start on the SMs
  // the one before it has already left (see Ctx::foldStream).
  cudaStream_t st = ctx->foldStream[slot];
  // The second batch after a queued redraw runs after the first, not beside it (see
  // Ctx::afterRedraw). The other slot holds the batch submitted just before this one.
  if (ctx->afterRedraw > 0) {
    if (ctx->afterRedraw == 1 && ctx->pendCount > 0)
      cudaStreamWaitEvent(st, ctx->slotDone[(slot + Ctx::kSlots - 1) % Ctx::kSlots], 0);
    ctx->afterRedraw--;
  }

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
  // Which fold this card loaded: resolved when the context was created (see
  // resolve_fold), because the operand draw has to know it too.
  if (!ctx->foldKnown) {
    if (err && err_len) snprintf(err, err_len, "%s", ctx->foldErr);
    return false;
  }
  // The block's shape: sixteen 32x64 warp tiles, or eight 64x64 ones. Either
  // way the CTA tile is 128x256, so the tile count, the grid and the shared
  // footprint below come out the same.
  // Turing's folds (see PEARL_TURING_BDIRECT) are eight 64x64 warp tiles over
  // 128x256 too: by default the B-direct fold (ctx->foldBDirect), whose shared
  // footprint is its own, below, and with PEARL_TURING_BDIRECT=0 one stage of
  // 48 KB instead of two. With PEARL_TURING_WIDE=0 it is eight 32x64 warp tiles
  // in a 4x2 grid: a 128x128 tile, so twice the tiles and two stages of 32 KB,
  // both of which follow from warpCols below.
  // The tall fold (ctx->foldTall) is 256 threads over 192x256 tiles; its tile
  // count and shared footprint are worked out apart from these, below.
  const bool wideWarps = ctx->foldWide || (ctx->foldTuring && PEARL_TURING_WIDE);
  const uint32_t threads = ctx->foldTall     ? PEARL_TALL_THREADS
                           : ctx->foldWide   ? PEARL_FOLD_WIDE_THREADS
                           : ctx->foldTuring ? PEARL_FOLD_TURING_THREADS
                                             : PEARL_FOLD_THREADS;
  const uint32_t warpRows = wideWarps ? PEARL_FOLD_WIDE_WARP_ROWS : PEARL_WARP_ROWS;
  const uint32_t rowTiles = wideWarps ? PEARL_FOLD_WIDE_ROW_TILES : PEARL_WMMA_ROW_TILES;
  const uint32_t stageBufs =
      ctx->foldTuring && PEARL_TURING_WIDE ? PEARL_TURING_WIDE_STAGE_BUFS : PEARL_STAGE_BUFS;
  const void *foldFn = ctx->foldTall ? reinterpret_cast<const void *>(pearl_tile_fold_tall)
                                     : reinterpret_cast<const void *>(pearl_tile_fold_wmma);
  // A valid-offset INDEX; the kernel expands it into an actual offset.
  const uint32_t col_off =
      (uint32_t)((nonce_base / ctx->rowsValid) % ctx->colsValid);

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
                        || (ctx->foldTiled && (col_off % PEARL_TALL_COL_OFFSETS != 0
                                               || ctx->profile.n % PEARL_TALL_BN != 0))
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
  // Turing's B-direct fold reads B' in blocks of 16 columns, and the draw writes
  // it that way only when n is a whole number of them (draw_noise).
  if (ctx->foldBDirect && ctx->profile.n % 16u != 0u) {
    if (err && err_len)
      snprintf(err, err_len, "B-direct fold: n %u is not a multiple of 16", ctx->profile.n);
    return false;
  }
  // Two full-chunk stages (one on Turing's one-stage fold); the transcripts live
  // in registers and global now. The tall fold: three 64-deep stages, its ring's
  // barriers, and the transcripts, laid out differently by the cp.async and TMA
  // builds (see PEARL_TALL_SMEM). Turing's B-direct fold: PEARL_BD_A_STAGES
  // stages of the tile's A rows, then the block's transcripts, 64 bytes a
  // region (pearl_fold_bd.cuh): 40 KB.
  const size_t smem =
      ctx->foldTall ? (ctx->foldTma ? (size_t)PEARL_TALL_SMEM_TMA : (size_t)PEARL_TALL_SMEM)
      : ctx->foldBDirect
          ? (size_t)PEARL_BD_A_STAGES * warpRows * regionsPerWarp * PEARL_ROWS_COUNT
                    * PEARL_SB_STRIDE
                + (size_t)warpsPerBlock * regionsPerWarp * PEARL_WMMA_COL_BLK
                      * PEARL_JACKPOT_BUCKETS * 4u
          : (size_t)stageBufs
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
  // On Turing the B-direct fold takes 40 KB and the one-stage fold 48 KB, so
  // neither needs the opt-in, and setting it anyway changes nothing. The
  // 128x128 fold (PEARL_TURING_WIDE=0) takes 64 KB and does.
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
      // shape as the launch below), or the card's resident block count rounded down to
      // a multiple of the cluster size if it cannot say.
      cudaLaunchConfig_t cfg = {};
      cudaLaunchAttribute attr[1] = {};
      pearl_cluster_config(&cfg, attr, PEARL_TALL_CLUSTER_SIZE, threads, smem, 0);
      int clusters = 0;
      if (cudaOccupancyMaxActiveClusters(&clusters, foldFn, &cfg) == cudaSuccess && clusters > 0) {
        ctx->foldResident = (unsigned)clusters * PEARL_TALL_CLUSTER_SIZE;
      } else {
        // A failed query is recorded as the thread's last error, and the search reads
        // cudaGetLastError after its launch; drop it, or the fallback fails the search
        // it exists for.
        (void)cudaGetLastError();
        ctx->foldResident -= ctx->foldResident % PEARL_TALL_CLUSTER_SIZE;
      }
      if (ctx->foldResident == 0) ctx->foldResident = PEARL_TALL_CLUSTER_SIZE;
    }
  }
  const unsigned blocks =
      ((ctx->foldPersistent || ctx->foldTall || (ctx->foldBDirect && PEARL_BD_PERSIST))
       && ctx->foldResident < tileCtas)
          ? ctx->foldResident
          : tileCtas;
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
  hitList.count = ctx->dHitCount + slot;
  hitList.index = ctx->dHitIndex + (size_t)slot * PEARL_MAX_HITS;
  hitList.hash = reinterpret_cast<uint32_t *>(ctx->dHashes + (size_t)slot * PEARL_MAX_HITS
                                                                 * PEARL_HASH_BYTES);
  hitList.transcript =
      ctx->dHitTranscript + (size_t)slot * PEARL_MAX_HITS * PEARL_JACKPOT_BUCKETS;
  cudaMemsetAsync(hitList.count, 0, sizeof(uint32_t), st);
  // Turing's persistent B-direct fold takes its tiles from this slot's counter.
  if (ctx->foldBDirect && PEARL_BD_PERSIST)
    cudaMemsetAsync(hitList.count + PEARL_BD_CTR_SLOTS, 0, sizeof(uint32_t), st);
#if PEARL_HOPPER_WGMMA
  if (ctx->foldHopper) {
    // Hopper's wgmma fold (PEARL_HOPPER_WGMMA), then the hash kernel over the transcripts
    // it wrote, on the same stream.
    pearl_launch_hopper(ctx, st, k, rank, chunks, col_off, col_groups, ctx->dTrG[slot]);
    pearl_tall_hash80<<<(regions + 255u) / 256u, 256, 0, st>>>(ctx->dTrG[slot], regions, test,
                                                               hitList, 1u);
  } else
#endif
  if (ctx->foldTall && ctx->foldCluster) {
    // The cluster build: the same launch, in clusters of PEARL_TALL_CLUSTER_SIZE CTAs
    // (blocks is a multiple of it, see foldResident), the same arguments by value, on
    // the slot's stream. cudaLaunchKernelEx coerces each argument to the kernel's
    // parameter type, the tensor maps included.
    cudaLaunchConfig_t cfg = {};
    cudaLaunchAttribute attr[1] = {};
    pearl_cluster_config(&cfg, attr, PEARL_TALL_CLUSTER_SIZE, threads, smem, st);
    cfg.gridDim = dim3(blocks);
    cudaLaunchKernelEx(&cfg, pearl_tile_fold_tall, ctx->dAp, ctx->dBp, ctx->profile.m,
                       ctx->profile.n, k, rank, chunks, col_off, ctx->rowsValid, col_groups,
                       tiles, test, hitList, ctx->tmA, ctx->tmB);
  } else if (ctx->foldTall && ctx->foldUnfused) {
    // GA100's unfused fold (PEARL_TALL_UNFUSED): the slot's transcript buffer travels in
    // the first bytes of tmA, which the cp.async fold does not otherwise read; the hash
    // kernel follows on the same stream, before the count is read back.
    PearlTensorMap trm;
    memset(&trm, 0, sizeof trm);
    memcpy(&trm, &ctx->dTrG[slot], sizeof ctx->dTrG[slot]);
    pearl_tile_fold_tall<<<blocks, threads, smem, st>>>(
        ctx->dAp, ctx->dBp, ctx->profile.m, ctx->profile.n, k, rank, chunks,
        col_off, ctx->rowsValid, col_groups, tiles, test, hitList, trm, ctx->tmB);
    pearl_tall_hash80<<<(regions + 255u) / 256u, 256, 0, st>>>(ctx->dTrG[slot], regions, test,
                                                               hitList, 1u);
  } else if (ctx->foldTall)
    pearl_tile_fold_tall<<<blocks, threads, smem, st>>>(
        ctx->dAp, ctx->dBp, ctx->profile.m, ctx->profile.n, k, rank, chunks,
        col_off, ctx->rowsValid, col_groups, tiles, test, hitList, ctx->tmA, ctx->tmB);
  else
    pearl_tile_fold_wmma<<<blocks, threads, smem, st>>>(
        ctx->dAp, ctx->dBp, ctx->profile.m, ctx->profile.n, k, rank, chunks,
        col_off, ctx->rowsValid, col_groups, tiles, test, hitList);

  // The count comes back through pinned memory behind the slot's event; the host reads
  // it once the event has passed, and the hit list itself only when it is not zero.
  cudaMemcpyAsync(ctx->hSlotCount + slot, hitList.count, sizeof(uint32_t),
                  cudaMemcpyDeviceToHost, st);
  cudaEventRecord(ctx->slotDone[slot], st);
  // WDDM batches submissions until something forces them out; this does, so the
  // batch starts behind the one running rather than when the host next waits.
  cudaStreamQuery(st);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    if (err && err_len)
      snprintf(err, err_len, "CUDA error during search: %s", cudaGetErrorString(e));
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

// How many batches are queued and not yet collected.
extern "C" int pearl_host_pending(void *handle) {
  const Ctx *ctx = static_cast<const Ctx *>(handle);
  return ctx ? ctx->pendCount : 0;
}

// Wait for the oldest queued batch and report it: `attempts` is its region count, and
// the return is true with `out` filled when it hit. The hit's proof is read now, on the
// side stream, with the salt-dependent part from its own salt's record (snapshotProof),
// so it is right even when later work has already restamped A.
extern "C" bool pearl_host_collect(void *handle, PearlSearchResult *out, uint64_t *attempts,
                                   char *err, size_t err_len) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (attempts) *attempts = 0;
  if (!ctx || !out || ctx->pendCount == 0) return false;
  DeviceScope scope(ctx->device);
  const int slot = ctx->pendHead;
  const Ctx::Pending p = ctx->pend[slot];
  cudaError_t e = cudaEventSynchronize(ctx->slotDone[slot]);
  if (e == cudaSuccess) e = cudaGetLastError();
  ctx->pendHead = (ctx->pendHead + 1) % Ctx::kSlots;
  ctx->pendCount--;
  if (e != cudaSuccess) {
    if (err && err_len)
      snprintf(err, err_len, "CUDA error during search: %s", cudaGetErrorString(e));
    return false;
  }
  const uint32_t hits = ctx->hSlotCount[slot];
  if (attempts) *attempts = p.regions;

  ctx->extraHits.clear();
  ctx->extraNext = 0;
  if (hits > 0) {
    const uint32_t *dIndex = ctx->dHitIndex + (size_t)slot * PEARL_MAX_HITS;
    const uint32_t n_hits = hits < PEARL_MAX_HITS ? hits : PEARL_MAX_HITS;
    side_copy(ctx, ctx->hHitIndex.data(), dIndex, (size_t)n_hits * sizeof(uint32_t));
    // The kernel appends with an atomic, so the list is in an arbitrary order. Report
    // it in region order, the LOWEST index first, which is what a sequential scan
    // would have returned -- otherwise which share goes first varies run to run.
    std::vector<uint32_t> order(n_hits);
    for (uint32_t i = 0; i < n_hits; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
      return ctx->hHitIndex[a] < ctx->hHitIndex[b];
    });
    fill_hit(ctx, p, slot, order[0], out);
    // Every other hit of the batch as well (PEARL_ALL_HITS). Only the lowest used to be
    // reported and the rest dropped: each is a distinct region that met the target, so
    // at a pool's difficulty that was a share lost whenever a batch held two.
#if PEARL_ALL_HITS
    ctx->extraHits.resize(n_hits - 1);
    for (uint32_t i = 1; i < n_hits; i++) fill_hit(ctx, p, slot, order[i], &ctx->extraHits[i - 1]);
#endif
    return true;
  }
  out->found = false;
  return false;
}

// The collected batch's other hits, one a call, after the one pearl_host_collect
// returned (PEARL_ALL_HITS). False when there are no more.
extern "C" bool pearl_host_next_hit(void *handle, PearlSearchResult *out) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx || !out || ctx->extraNext >= ctx->extraHits.size()) return false;
  *out = ctx->extraHits[ctx->extraNext++];
  return true;
}

// The synchronous search the bench probe and any older caller use: queue one batch and
// wait for it. The miner's own loop (pearl_core.cc) keeps the pipeline full instead.
extern "C" bool pearl_host_search(void *handle, uint64_t nonce_base, uint32_t batch,
                                  PearlSearchResult *out, uint64_t *attempts, char *err,
                                  size_t err_len) {
  if (attempts) *attempts = 0;
  if (!handle || !out) return false;
  Ctx *ctx = static_cast<Ctx *>(handle);
  while (ctx->pendCount > 0) {  // not meant to be mixed with the pipelined calls
    PearlSearchResult drop;
    uint64_t a = 0;
    pearl_host_collect(handle, &drop, &a, nullptr, 0);
  }
  if (!pearl_host_submit(handle, nonce_base, batch, nullptr, err, err_len)) return false;
  return pearl_host_collect(handle, out, attempts, err, err_len);
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
  if (ctx->foldHopper) {
    snprintf(const_cast<Ctx *>(ctx)->foldName, sizeof ctx->foldName,
             "hopper 128x256, 2 warpgroups of wgmma m64n256 and a producer warp, TMA ring, %s "
             "operands, band %u, hash in its own kernel%s",
             ctx->foldTiled ? "per-tile" : "k-blocked", ctx->tallBand,
             PEARL_HOPPER_CLUSTER ? ", 2-CTA clusters sharing B by multicast" : "");
    return ctx->foldName;
  }
  if (ctx->foldAmpere) {
    snprintf(const_cast<Ctx *>(ctx)->foldName, sizeof ctx->foldName,
             "tall 192x256, 8 warps of 96x64, cp.async ring, %s operands, band %u%s",
             ctx->foldTiled ? "per-tile" : "k-blocked", ctx->tallBand,
             ctx->foldUnfused ? ", hash in its own kernel" : "");
    return ctx->foldName;
  }
  if (ctx->foldTall)
    return ctx->foldTma ? "tall 192x256, 8 warps of 96x64, TMA ring, k-blocked operands"
                        : (ctx->foldTiled ? "tall 192x256, 8 warps of 96x64, cp.async ring, per-tile operands"
                                          : "tall 192x256, 8 warps of 96x64, cp.async ring");
  if (ctx->foldWide) return "wmma 128x256, 8 warps of 64x64";
  if (ctx->foldBDirect) {
    snprintf(const_cast<Ctx *>(ctx)->foldName, sizeof ctx->foldName,
             "B-direct 128x256, 8 warps of 64x64, %d row group%s, %s, band %u, m8n8k16 (Turing)",
             PEARL_BD_GROUPS, PEARL_BD_GROUPS == 2 ? "s" : "",
             PEARL_BD_PERSIST ? "persistent" : "a block a tile", ctx->bdBand);
    return ctx->foldName;
  }
  if (ctx->foldTuring)
    return PEARL_TURING_WIDE ? "wmma 128x256, 8 warps of 64x64, one stage, m8n8k16 (Turing)"
                             : "wmma 128x128, 8 warps of 32x64, m8n8k16 (Turing)";
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
    cudaMemcpy(out + (size_t)i * 1024, operand_chunk(ctx, src, leaf_indices[i]), 1024,
               cudaMemcpyDeviceToHost);
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
