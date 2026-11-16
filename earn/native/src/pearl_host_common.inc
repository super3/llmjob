// Host code the CUDA core (pearl_host.cu) and the HIP core (amd/src/pearl_host_hip.cpp)
// share: the operand commitment, the host BLAKE3 and restamp records, a hit's proof reads,
// and the seed links. Each host file includes it inside its anonymous namespace, after its
// Ctx, with the runtime calls named through gpu_api.h. Before including it, a host file
// defines pearl_draw_stream(ctx), the stream the draw's kernels and copies go on (the
// CUDA core's legacy default stream, the HIP core's work stream), and declares
// draw_noise(ctx, isA), its own.
//
// Every field used here has the same name and meaning in both Ctx structs.

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

// Keyed BLAKE3 over a whole operand: hash each 1024-byte chunk into a leaf CV,
// then fold pairs until one root remains. The final fold carries ROOT and its
// output IS the digest.
//
// Chunk counts are powers of two here (m*k with power-of-two m and k), so the
// tree is balanced and the fold is exact. A non-power-of-two operand would need
// BLAKE3's left-heavy layout, which this deliberately does not pretend to do.
//
// `level0` (compact operands only): chunk CVs already made for another operand of the
// same fill under the same job key, at least `len` long. A chunk's CV depends only on
// the key, its 1024 bytes and its index, so every chunk past chunk 0 has the CV that
// operand's chunk of the same index has; they are copied, and only chunk 0 (the salt
// stamp's) is hashed. The rotation fold's A (m*k/1024 = 8.4M chunks) takes B's this
// way: a third of a job's chunk hashing, about 8 ms of a 4070's 39 ms draw.
void operand_commitment(Ctx *ctx, const uint8_t *data, size_t len, uint8_t *out32, uint32_t *tree,
                        std::vector<uint64_t> *offsets, const uint32_t *level0 = nullptr) {
  const gpuStream_t st = pearl_draw_stream(ctx);
  const uint64_t chunks = len / 1024;
  const uint32_t threads = 256;
  if (offsets) offsets->clear();
  if (chunks <= 1) {
    // One chunk or less: the chunk's own final compression carries ROOT.
    PEARL_GPU_LAUNCH(pearl_hash_operands, 1, 1, st, (const uint32_t *)ctx->dJobKey, data, (uint32_t)len, out32);
    return;
  }

  // Each level is written to its own place rather than reduced in place, so the
  // sibling digests a share proof needs are still there afterwards.
  uint32_t *dst = tree ? tree : ctx->dCvs;
  uint64_t base = 0;
  if (offsets) offsets->push_back(0);
  if (level0 && ctx->compact) {
    gpuMemcpyAsync(dst + 8, level0 + 8, (size_t)(chunks - 1) * PEARL_HASH_BYTES, gpuMemcpyDeviceToDevice, st);
    PEARL_GPU_LAUNCH(pearl_blake3_chunk_cvs, 1, 1, st, (const uint32_t *)ctx->dJobKey, data, (uint64_t)1, dst, 1u);
  } else {
    PEARL_GPU_LAUNCH(pearl_blake3_chunk_cvs, (unsigned)((chunks + threads - 1) / threads), threads, st,
                     (const uint32_t *)ctx->dJobKey, data, chunks, dst, ctx->compact ? 1u : 0u);
  }

  uint64_t count = chunks;
  while (count > 1) {
    const uint64_t pairs = count / 2;
    const uint32_t isRoot = (pairs == 1) ? 1u : 0u;
    const uint64_t next = tree ? base + count : 0;
    PEARL_GPU_LAUNCH(pearl_blake3_parent_layer, (unsigned)((pairs + threads - 1) / threads), threads, st,
                     (const uint32_t *)ctx->dJobKey, (const uint32_t *)(dst + base * 8), pairs, isRoot,
                     dst + next * 8);
    if (offsets) offsets->push_back(next);
    base = next;
    count = pairs;
  }
  gpuMemcpy(out32, dst + base * 8, PEARL_HASH_BYTES, gpuMemcpyDeviceToDevice);
}

// ---------------------------------------------------------------------------
// BLAKE3 on the host, for host_record only: one chunk CV, parent compressions and
// two one-block hashes a redraw, about 40 compressions, some tens of microseconds on
// one core. The device's BLAKE3 (pearl_kernel.cu) is the one every other hash uses;
// this one is checked against it once a job (see setup_host_seeds) and not trusted
// until it agrees. Its compression is pearl_blake3.h's, which the SYCL core runs on
// both sides.
// ---------------------------------------------------------------------------
const uint32_t kB3Iv[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                           0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};
const uint32_t kB3ChunkStart = 1u, kB3ChunkEnd = 2u, kB3Parent = 4u, kB3Root = 8u,
               kB3Keyed = 16u;

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
    pearl_b3::compress(cv, block, counter, 64, flags, o);
    for (int i = 0; i < 8; i++) cv[i] = o[i];
  }
  for (int i = 0; i < 8; i++) out[i] = cv[i];
}

// BLAKE3 of exactly 64 bytes as a root: keyed when key is given, else unkeyed.
void hb3_hash64(const uint32_t *key, const uint8_t msg[64], uint32_t out[8]) {
  uint32_t block[16], o[16];
  hb3_words(msg, block, 16);
  pearl_b3::compress(key ? key : kB3Iv, block, 0, 64,
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
    pearl_b3::compress(ctx->jobKeyW, block, 0, 64, flags, o);
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
  gpuMemcpyAsync(dst, src, n, gpuMemcpyDeviceToHost, ctx->side);
  gpuStreamSynchronize(ctx->side);
}

// The device reads one hit needs: its hash and transcript, the proof's leaves and
// sibling nodes, and the roots. Each used to be its own copy and its own wait, about
// 120 a hit, and the search thread does this between batches, so the GPU runs dry
// once the one batch queued behind it is done. A wait is cheap on Linux and costs far
// more under WDDM and WSL2 (Windows rigs, Salad), where nothing had measured it: a
// card with no tensor cores never found a hit at a pool's target. Queued, the reads
// cost one wait. A read lands at a fixed address, or at an offset into a vector that
// may still grow until the reads run.
//
// A read of bytes an earlier read of the hit already fetches (the same device address and
// length) is not queued again: it takes the earlier one's copy. Under the constant fill a
// compact operand keeps two 1024-byte chunks (operand_chunk), so a proof's 768 + 1024
// leaves (the rotation fold's 12 rows and 16 columns at k 65536) are two distinct reads,
// not 1,792 copies queued one by one. Those calls held the search thread long enough
// that the GPU ran dry for about 20 ms a share on rig01's 4090 (goal-nvk23).
struct HitReads {
  struct Read {
    std::vector<uint8_t> *vec;
    uint8_t *ptr;
    size_t off;
    const void *src;
    size_t n;
    size_t same;   // the index of the read whose copy this one takes, or kOwn when it is fetched
  };
  static constexpr size_t kOwn = ~(size_t)0;
  // Host bytes laid over a read once it has landed: leaf 0's head from a restamp record.
  struct Patch {
    std::vector<uint8_t> *vec;
    size_t off;
    uint8_t bytes[sizeof(PearlRestampRecord::head)];
  };
  std::vector<Read> reads;
  std::vector<Patch> patches;
  std::map<const void *, size_t> first;   // a source address -> its first read
  size_t bytes = 0;
  void add(const Read &r0) {
    // PEARL_HIT_DEDUP=0 queues every read as before (for racing the two).
    static const bool dedup = [] {
      const char *v = getenv("PEARL_HIT_DEDUP");
      return !(v && v[0] == '0');
    }();
    Read r = r0;
    r.same = kOwn;
    const auto it = dedup ? first.find(r.src) : first.end();
    if (it != first.end() && reads[it->second].n == r.n) {
      r.same = it->second;
    } else {
      first[r.src] = reads.size();
      bytes += r.n;
    }
    reads.push_back(r);
  }
  void to(void *dst, const void *src, size_t n) {
    add({nullptr, static_cast<uint8_t *>(dst), 0, src, n, kOwn});
  }
  void into(std::vector<uint8_t> *vec, size_t off, const void *src, size_t n) {
    add({vec, nullptr, off, src, n, kOwn});
  }
};

// Run a hit's reads: all queued on the side stream into the pinned stage, one wait,
// then each copied to where it belongs and the patches laid over them. Without a stage
// (pinned memory refused) each read is a side_copy, as before.
void run_reads(Ctx *ctx, HitReads *r) {
  auto land = [](const HitReads::Read &x) { return x.vec ? x.vec->data() + x.off : x.ptr; };
  if (r->bytes > ctx->hStageBytes) {
    if (ctx->hStage) gpuFreeHost(ctx->hStage);
    ctx->hStage = nullptr;
    ctx->hStageBytes = 0;
    const size_t want = r->bytes > ((size_t)128 << 10) ? r->bytes : ((size_t)128 << 10);
    if (gpuHostAlloc(reinterpret_cast<void **>(&ctx->hStage), want) == gpuSuccess) {
      ctx->hStageBytes = want;
    } else {
      ctx->hStage = nullptr;
      (void)gpuGetLastError();
    }
  }
  if (ctx->hStage) {
    std::vector<size_t> at(r->reads.size());
    size_t off = 0;
    for (size_t i = 0; i < r->reads.size(); i++) {
      const HitReads::Read &x = r->reads[i];
      if (x.same != HitReads::kOwn) {
        at[i] = at[x.same];
        continue;
      }
      at[i] = off;
      gpuMemcpyAsync(ctx->hStage + off, x.src, x.n, gpuMemcpyDeviceToHost, ctx->side);
      off += x.n;
    }
    gpuStreamSynchronize(ctx->side);
    for (size_t i = 0; i < r->reads.size(); i++)
      memcpy(land(r->reads[i]), ctx->hStage + at[i], r->reads[i].n);
  } else {
    for (const HitReads::Read &x : r->reads) {
      if (x.same == HitReads::kOwn)
        side_copy(ctx, land(x), x.src, x.n);
      else
        memcpy(land(x), land(r->reads[x.same]), x.n);
    }
  }
  for (const HitReads::Patch &p : r->patches)
    memcpy(p.vec->data() + p.off, p.bytes, sizeof p.bytes);
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
// The device reads go into `reads`, which the caller runs once it has queued all of the
// hit's (run_reads); until then the leaf and sibling bytes are placeholders.
void snapshotProof(Ctx *ctx, bool isA, const uint32_t *rows, uint32_t nrows,
                   std::vector<uint32_t> *leafIdx, std::vector<uint8_t> *leaves,
                   std::vector<uint8_t> *sibs, const PearlRestampRecord *rec,
                   HitReads *reads) {
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

bool fail(char *err, size_t err_len, const char *msg) {
  if (err && err_len) snprintf(err, err_len, "%s", msg);
  return false;
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
  const gpuStream_t st = pearl_draw_stream(ctx);
  const bool legacy = ctx->profile.seed_derivation == PEARL_SEED_LEGACY;
  if (legacy) {
    gpuMemcpy(ctx->dBoundA, ctx->dHashA, PEARL_HASH_BYTES, gpuMemcpyDeviceToDevice);
    if (withB) gpuMemcpy(ctx->dBoundB, ctx->dHashB, PEARL_HASH_BYTES, gpuMemcpyDeviceToDevice);
  } else {
    PEARL_GPU_LAUNCH(pearl_bind_root, 1, 1, st, (const uint8_t *)ctx->dSaltA, (const uint8_t *)ctx->dHashA,
                     ctx->profile.m, ctx->dBoundA);
    if (withB)
      PEARL_GPU_LAUNCH(pearl_bind_root, 1, 1, st, (const uint8_t *)ctx->dSaltB, (const uint8_t *)ctx->dHashB,
                       ctx->profile.n, ctx->dBoundB);
  }
  if (withB) {
    gpuMemcpy(ctx->dSeedBuf, ctx->dJobKey, PEARL_HASH_BYTES, gpuMemcpyDeviceToDevice);
    gpuMemcpy(ctx->dSeedBuf + PEARL_HASH_BYTES, ctx->dBoundB, PEARL_HASH_BYTES, gpuMemcpyDeviceToDevice);
    PEARL_GPU_LAUNCH(pearl_blake3_unkeyed, 1, 1, st, (const uint8_t *)ctx->dSeedBuf, 64u,
                     reinterpret_cast<uint8_t *>(ctx->dBSeed));
  }
  gpuMemcpy(ctx->dSeedBuf, ctx->dBSeed, PEARL_HASH_BYTES, gpuMemcpyDeviceToDevice);
  gpuMemcpy(ctx->dSeedBuf + PEARL_HASH_BYTES, ctx->dBoundA, PEARL_HASH_BYTES, gpuMemcpyDeviceToDevice);
  PEARL_GPU_LAUNCH(pearl_blake3_unkeyed, 1, 1, st, (const uint8_t *)ctx->dSeedBuf, 64u,
                   reinterpret_cast<uint8_t *>(ctx->dASeed));
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
  PEARL_GPU_LAUNCH(pearl_restamp_operand, 1, 1, pearl_draw_stream(ctx), (const uint32_t *)ctx->dJobKey, ctx->dA,
                   salt, aChunks, ctx->dTreeA, ctx->dHashA);
  derive_seeds(ctx, false);
  draw_noise(ctx, true);
}

// A record read back from the device, for a draw the host did not hash itself.
void device_record(Ctx *ctx, PearlRestampRecord *r) {
  const uint32_t levels = (uint32_t)ctx->layerOffA.size();
  memset(r, 0, sizeof *r);
  r->levels = levels < PEARL_RESTAMP_MAX_LEVELS ? levels : PEARL_RESTAMP_MAX_LEVELS;
  gpuMemcpy(r->head, ctx->dA, sizeof r->head, gpuMemcpyDeviceToHost);
  for (uint32_t L = 0; L < r->levels; L++) {
    r->node_off[L] = (uint32_t)ctx->layerOffA[L];
    gpuMemcpy(r->path[L], ctx->dTreeA + ctx->layerOffA[L] * 8, 32, gpuMemcpyDeviceToHost);
  }
  gpuMemcpy(r->bound, ctx->dBoundA, 32, gpuMemcpyDeviceToHost);
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
    gpuMemcpy(ctx->jobKeyW, ctx->dJobKey, 32, gpuMemcpyDeviceToHost);
    gpuMemcpy(ctx->leaf0, ctx->dA, sizeof ctx->leaf0, gpuMemcpyDeviceToHost);
    for (uint32_t L = 0; L + 1 < levels; L++)
      gpuMemcpy(ctx->sib[L], ctx->dTreeA + (ctx->layerOffA[L] + 1) * 8, 32, gpuMemcpyDeviceToHost);
    host_record(ctx, ctx->salt, false, &r);
    uint8_t root[32], hroot[32], hseed[32];
    gpuMemcpy(root, ctx->dHashA, 32, gpuMemcpyDeviceToHost);
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
