// Host-side driver for the AMD core: the same extern "C" entry points pearl_host.cu gives
// pearl_core.cc, on HIP. pearl_core.cc itself is compiled unchanged against this file, so
// the addon the app loads (pearl_core_hip.node) has exactly the CUDA core's interface.
//
// This is a port of pearl_host.cu with the NVIDIA folds taken out. What is the same, and
// why it matters, is explained there; the comments here say what differs:
//   - Which fold runs is read off the card's gfx architecture (resolve_fold), and checked
//     against the folds the loaded code object has bodies for (pearl_amd_caps).
//     PEARL_AMD_FOLD=ref|mfma16|mfma8|wmma11|wmma12 picks one by hand.
//   - Every fold stores every region's transcript, and pearl_amd_hash hashes them after
//     it, so each pipeline slot has a transcript buffer: 64 bytes a region of the batch.
//     The batch is narrower than the CUDA core's to keep those buffers small
//     (PEARL_AMD_COL_BATCH).
//   - A' and B' are row-major (pearl_materialize16), the layout every AMD fold reads.
//   - Everything runs on the null stream, in order; proof reads use a side stream.
//
// The same file builds two ways (build-amd.sh): with ROCm's hip-clang for real cards, and
// as plain C++ against HIP-CPU (__HIP_CPU_RT__), which runs the kernels on the CPU. The
// CPU build exists to check the core without a GPU, and is never shipped.

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

#include "pearl_amd_kernels.h"

// How many column offsets one batch covers at most. Each pipeline slot keeps a transcript
// buffer of col_batch x (m / 16) regions x 64 bytes: 128 MiB at 256 and the mainnet m.
// -DPEARL_AMD_COL_BATCH=N sets another width; the host narrows it further if it has to.
#ifndef PEARL_AMD_COL_BATCH
#define PEARL_AMD_COL_BATCH 256u
#endif

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

// The folds, in the order of their PEARL_AMD_CAP_* bits.
struct FoldInfo {
  const char *name;
  uint32_t cap;
  uint32_t bm, bn;  // the fold's tile: rows of A', columns (rows of B')
  const char *text;
};
const FoldInfo kFolds[] = {
    {"ref", PEARL_AMD_CAP_REF, PEARL_AMD_REF_BM, PEARL_AMD_REF_BN,
     "ref 32x64, dot products from LDS, hash in its own kernel"},
    {"mfma16", PEARL_AMD_CAP_MFMA16, PEARL_AMD_BM, PEARL_AMD_BN,
     "mfma 128x256, 4 waves of 64x128, v_mfma_i32_32x32x16_i8, hash in its own kernel"},
    {"mfma8", PEARL_AMD_CAP_MFMA8, PEARL_AMD_BM, PEARL_AMD_BN,
     "mfma 128x256, 4 waves of 64x128, v_mfma_i32_32x32x8_i8, hash in its own kernel"},
    {"wmma11", PEARL_AMD_CAP_WMMA11, PEARL_AMD_BM, PEARL_AMD_BN,
     "wmma 128x256, 8 waves of 64x64, v_wmma_i32_16x16x16_iu8 (RDNA3), hash in its own kernel"},
    {"wmma12", PEARL_AMD_CAP_WMMA12, PEARL_AMD_BM, PEARL_AMD_BN,
     "wmma 128x256, 8 waves of 64x64, v_wmma_i32_16x16x16_iu8 (RDNA4), hash in its own kernel"},
};
const int kFoldCount = (int)(sizeof kFolds / sizeof kFolds[0]);

int fold_by_name(const char *name) {
  for (int i = 0; i < kFoldCount; i++)
    if (strcmp(kFolds[i].name, name) == 0) return i;
  return -1;
}

// The fold a gfx architecture runs by default: its matrix instruction, or the reference
// fold where it has none (RDNA2 and older).
int native_fold(const char *arch) {
  if (!strncmp(arch, "gfx942", 6) || !strncmp(arch, "gfx950", 6)) return fold_by_name("mfma16");
  if (!strncmp(arch, "gfx90a", 6) || !strncmp(arch, "gfx908", 6)) return fold_by_name("mfma8");
  if (!strncmp(arch, "gfx11", 5)) return fold_by_name("wmma11");
  if (!strncmp(arch, "gfx12", 5)) return fold_by_name("wmma12");
  return fold_by_name("ref");
}

struct Ctx {
  PearlProfile profile;
  int device = 0;
  char arch[64] = {0};      // gcnArchName, e.g. "gfx942:sramecc+:xnack-"
  int fold = -1;            // index into kFolds; -1 until resolve_fold settles it
  bool emulated = false;    // the HIP-CPU build
  char foldErr[256] = {0};
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
  hipStream_t work = nullptr;
  uint32_t *dTr[kSlots] = {nullptr, nullptr};  // each slot's transcripts, 16 words a region
  std::vector<PearlSearchResult> extraHits;
  size_t extraNext = 0;

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

// The batch width: the profile's col_batch, at most the valid column offsets, at most
// PEARL_AMD_COL_BATCH, halved until it divides the valid offsets. A batch must be whole
// tiles of every fold (16 column offsets, the matrix folds' 256 columns), so a width
// under 16 is refused when the search starts.
uint32_t amd_col_batch(const PearlProfile *profile) {
  const uint32_t colsValid = profile->n / PEARL_COLS_COUNT;
  uint32_t cb = profile->col_batch ? profile->col_batch : 1u;
  if (cb > colsValid) cb = colsValid;
  if (cb > PEARL_AMD_COL_BATCH) cb = PEARL_AMD_COL_BATCH;
  while (cb > 1u && colsValid % cb != 0u) cb /= 2u;
  return cb;
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
  const size_t trBytes = (size_t)Ctx::kSlots * amd_col_batch(profile)
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
// (leaf 0's head and node 0 of every level) from the hit's own salt's record.
void snapshotProof(Ctx *ctx, bool isA, const uint32_t *rows, uint32_t nrows,
                   std::vector<uint32_t> *leafIdx, std::vector<uint8_t> *leaves,
                   std::vector<uint8_t> *sibs, const PearlRestampRecord *rec) {
  if (!isA) rec = nullptr;
  const uint32_t k = ctx->profile.k;
  const int8_t *operand = isA ? ctx->dA : ctx->dB;
  const uint32_t *tree = isA ? ctx->dTreeA : ctx->dTreeB;
  const std::vector<uint64_t> &offs = isA ? ctx->layerOffA : ctx->layerOffB;
  const uint64_t totalLeaves = (uint64_t)(isA ? ctx->profile.m : ctx->profile.n) * k / 1024;

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
  hipLaunchKernelGGL(pearl_amd_caps, dim3(1), dim3(1), 0, ctx->work, ctx->dCaps);
  uint32_t caps[2] = {0, 0};
  hipError_t e = hipGetLastError();
  if (e == hipSuccess) e = hipMemcpy(caps, ctx->dCaps, sizeof caps, hipMemcpyDeviceToHost);
  if (e != hipSuccess) {
    snprintf(err, err_len, "this core has no code for %s (%s): build it with build-amd.sh for that target",
             ctx->arch[0] ? ctx->arch : "this card", hipGetErrorString(e));
    (void)hipGetLastError();
    return;
  }
  ctx->emulated = (caps[0] & PEARL_AMD_CAP_EMULATED) != 0;
  int want = ctx->emulated ? fold_by_name("mfma16") : native_fold(ctx->arch);
  const char *forced = getenv("PEARL_AMD_FOLD");
  if (forced && forced[0]) {
    want = fold_by_name(forced);
    if (want < 0) {
      snprintf(err, err_len, "PEARL_AMD_FOLD=%s is not a fold: use ref, mfma16, mfma8, wmma11 or wmma12",
               forced);
      return;
    }
  }
  if (!(caps[0] & kFolds[want].cap)) {
    snprintf(err, err_len, "the %s fold has no code for %s in this build (folds present: 0x%x)",
             kFolds[want].name, ctx->arch[0] ? ctx->arch : "this card", caps[0]);
    return;
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
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    (void)hipGetLastError();
    fail(err, err_len, "no AMD GPU found: is the ROCm driver (amdgpu) installed, and can this user open /dev/kfd?");
    return -1;
  }
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

extern "C" void *pearl_host_create(const PearlProfile *profile, char *err, size_t err_len) {
  if (!profile) { fail(err, err_len, "no profile supplied"); return nullptr; }
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    (void)hipGetLastError();
    fail(err, err_len, "no AMD GPU found: is the ROCm driver (amdgpu) installed, and can this user open /dev/kfd?");
    return nullptr;
  }
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
  ctx->colBatch = amd_col_batch(profile);
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
  for (int i = 0; i < Ctx::kSlots; i++)
    HIP_OK(hipEventCreateWithFlags(&ctx->slotDone[i], hipEventDisableTiming), "creating the batch events");
  HIP_OK(hipStreamCreateWithFlags(&ctx->side, hipStreamNonBlocking), "creating the proof stream");
  HIP_OK(hipStreamCreateWithFlags(&ctx->work, hipStreamDefault), "creating the search stream");
  for (int i = 0; i < Ctx::kSlots; i++)
    HIP_OK(hipMalloc(&ctx->dTr[i], (size_t)ctx->batch * PEARL_JACKPOT_BUCKETS * sizeof(uint32_t)),
           "allocating the transcript buffers");
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

  resolve_fold(ctx);
  return ctx;
}

extern "C" void pearl_host_bind_thread(void *handle) {
  Ctx *ctx = (Ctx *)handle;
  if (!ctx) return;
  hipSetDevice(ctx->device);
}

extern "C" void pearl_host_destroy(void *handle) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx) return;
  DeviceScope scope(ctx->device);
  hipDeviceSynchronize();
  hipFree(ctx->dA); hipFree(ctx->dB);
  hipFree(ctx->dAp); hipFree(ctx->dBp);
  hipFree(ctx->dEAL); hipFree(ctx->dEBR);
  hipFree(ctx->dPermA); hipFree(ctx->dPermB);
  hipFree(ctx->dLabelA); hipFree(ctx->dLabelB);
  hipFree(ctx->dSaltA); hipFree(ctx->dSaltB);
  hipFree(ctx->dBoundA); hipFree(ctx->dBoundB);
  hipFree(ctx->dHashes); hipFree(ctx->dHitCount); hipFree(ctx->dHitIndex); hipFree(ctx->dCaps);
  for (int s = 0; s < Ctx::kSlots; s++) hipFree(ctx->dTr[s]);
  hipFree(ctx->dTreeA); hipFree(ctx->dTreeB);
  hipFree(ctx->dCvs); hipFree(ctx->dSeedBuf); hipFree(ctx->dSeedInput);
  hipFree(ctx->dHashA); hipFree(ctx->dHashB);
  hipFree(ctx->dHitTranscript); hipFree(ctx->dJobKey);
  hipFree(ctx->dASeed); hipFree(ctx->dBSeed);
  for (int i = 0; i < Ctx::kSlots; i++)
    if (ctx->slotDone[i]) hipEventDestroy(ctx->slotDone[i]);
  if (ctx->side) hipStreamDestroy(ctx->side);
  if (ctx->work) hipStreamDestroy(ctx->work);
  if (ctx->hSlotCount) hipHostFree(ctx->hSlotCount);
  (void)hipGetLastError();
  delete ctx;
}

extern "C" void pearl_host_reseed(void *handle, uint64_t salt);

extern "C" void pearl_host_set_job_salted(void *handle, const uint8_t *header,
                                          const uint8_t *target, uint64_t salt) {
  Ctx *ctx = static_cast<Ctx *>(handle);
  if (!ctx) return;
  DeviceScope scope(ctx->device);
  memcpy(ctx->header, header, PEARL_HEADER_BYTES);
  memcpy(ctx->target, target, PEARL_HASH_BYTES);
  ctx->baseDrawn = false;
  pearl_host_reseed(handle, salt);
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

// One side's noise, and the noised operand, row-major: the dense factor, the sparse
// selectors, then pearl_materialize16 (pearl_materialize when k is not a power of two).
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
  if (kPow2) {
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
}

namespace {
// One hit of a collected batch, as a result with its share proof (see pearl_host.cu).
void fill_hit(Ctx *ctx, const Ctx::Pending &p, int slot, uint32_t i, PearlSearchResult *out) {
  const uint8_t *dHashes = ctx->dHashes + (size_t)slot * PEARL_MAX_HITS * PEARL_HASH_BYTES;
  const uint32_t *dTranscripts = ctx->dHitTranscript + (size_t)slot * PEARL_MAX_HITS * PEARL_JACKPOT_BUCKETS;
  side_copy(ctx, out->jackpot_hash, dHashes + (size_t)i * PEARL_HASH_BYTES, PEARL_HASH_BYTES);
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
                &out->proof_a.siblings, rec);
  snapshotProof(ctx, false, cols, PEARL_COLS_COUNT, &out->proof_bt.leaf_indices, &out->proof_bt.leaves,
                &out->proof_bt.siblings, nullptr);
  if (rec)
    hb3_bytes(rec->path[rec->levels - 1], out->proof_a.root);
  else
    side_copy(ctx, out->proof_a.root, ctx->dHashA, PEARL_HASH_BYTES);
  side_copy(ctx, out->proof_bt.root, ctx->dHashB, PEARL_HASH_BYTES);
  out->proof_a.total_leaves = (uint64_t)ctx->profile.m * ctx->profile.k / 1024;
  out->proof_bt.total_leaves = (uint64_t)ctx->profile.n * ctx->profile.k / 1024;
  out->proof.assign(PEARL_JACKPOT_BUCKETS * 4, 0);
  side_copy(ctx, out->proof.data(), dTranscripts + (size_t)i * PEARL_JACKPOT_BUCKETS,
            PEARL_JACKPOT_BUCKETS * 4);
  out->found = true;
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
  const FoldInfo &f = kFolds[ctx->fold];
  const uint32_t k = ctx->profile.k, rank = ctx->profile.rank;
  const uint32_t chunks = rank ? k / rank : 0;
  // Every fold reads whole PEARL_AMD_SLAB-byte slabs, a chunk at a time, and writes each of
  // the 16 transcript words at most once. The reference fold holds one chunk of a row in
  // LDS, 128 bytes at most.
  if (rank == 0 || k % rank != 0 || rank % PEARL_AMD_SLAB != 0 || chunks > PEARL_JACKPOT_BUCKETS
      || (f.cap == PEARL_AMD_CAP_REF && rank > 128u)) {
    if (err && err_len)
      snprintf(err, err_len, "the AMD folds need rank a multiple of %u (at most 128 for the reference fold) "
               "dividing k, and at most %u chunks (got rank %u, k %u)",
               (unsigned)PEARL_AMD_SLAB, (unsigned)PEARL_JACKPOT_BUCKETS, rank, k);
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
  hipMemsetAsync(hitList.count, 0, sizeof(uint32_t), ctx->work);

  const dim3 grid(batchCols / f.bn, ctx->profile.m / f.bm), block(PEARL_AMD_THREADS);
  const int8_t *ap = ctx->dAp, *bp = ctx->dBp;
  const uint32_t rv = ctx->rowsValid;
  uint32_t *tr = ctx->dTr[slot];
  switch (f.cap) {
    case PEARL_AMD_CAP_MFMA16:
      hipLaunchKernelGGL(pearl_amd_fold_mfma16, grid, block, 0, ctx->work, ap, bp, k, rank, col0, rv, tr);
      break;
    case PEARL_AMD_CAP_MFMA8:
      hipLaunchKernelGGL(pearl_amd_fold_mfma8, grid, block, 0, ctx->work, ap, bp, k, rank, col0, rv, tr);
      break;
    case PEARL_AMD_CAP_WMMA11:
      hipLaunchKernelGGL(pearl_amd_fold_wmma11, grid, block, 0, ctx->work, ap, bp, k, rank, col0, rv, tr);
      break;
    case PEARL_AMD_CAP_WMMA12:
      hipLaunchKernelGGL(pearl_amd_fold_wmma12, grid, block, 0, ctx->work, ap, bp, k, rank, col0, rv, tr);
      break;
    default:
      hipLaunchKernelGGL(pearl_amd_fold_ref, grid, block, 0, ctx->work, ap, bp, k, rank, col0, rv, tr);
      break;
  }
  hipLaunchKernelGGL(pearl_amd_hash, dim3((regions + PEARL_AMD_THREADS - 1) / PEARL_AMD_THREADS),
                     dim3(PEARL_AMD_THREADS), 0, ctx->work, (const uint32_t *)tr, regions, test, hitList);
  hipMemcpyAsync(ctx->hSlotCount + slot, hitList.count, sizeof(uint32_t), hipMemcpyDeviceToHost, ctx->work);
  hipEventRecord(ctx->slotDone[slot], ctx->work);
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
  hipError_t e = hipEventSynchronize(ctx->slotDone[slot]);
  if (e == hipSuccess) e = hipGetLastError();
  ctx->pendHead = (ctx->pendHead + 1) % Ctx::kSlots;
  ctx->pendCount--;
  if (e != hipSuccess) {
    if (err && err_len) snprintf(err, err_len, "HIP error during search: %s", hipGetErrorString(e));
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
    std::vector<uint32_t> order(n_hits);
    for (uint32_t i = 0; i < n_hits; i++) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](uint32_t a, uint32_t b) { return ctx->hHitIndex[a] < ctx->hHitIndex[b]; });
    fill_hit(ctx, p, slot, order[0], out);
    ctx->extraHits.resize(n_hits - 1);
    for (uint32_t i = 1; i < n_hits; i++) fill_hit(ctx, p, slot, order[i], &ctx->extraHits[i - 1]);
    return true;
  }
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
  snprintf(ctx->foldName, sizeof ctx->foldName, "%s, batch %u column offsets%s%s%s",
           kFolds[ctx->fold].text, ctx->colBatch, ctx->emulated ? ", emulated on the CPU (HIP-CPU)" : "",
           ctx->arch[0] ? ", " : "", ctx->arch);
  return ctx->foldName;
}

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
