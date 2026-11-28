// The host side of a NEB core (neb_host.h), written once for CUDA and HIP. The including file has
// already included the vendor's runtime and defined
//   G(x)          the vendor's name of runtime call or constant x: cuda##x or hip##x
//   NB_VENDOR     "nvidia" or "amd"
//   NB_PROP       the type of a card's properties
//   NB_MIN_OK(p)  whether a card with these device properties can run the core
//   NEB_DEV, NEB_HD, NEB_ATOMIC_INC  for neb_kernel.h
// CUDA and HIP share the kernel syntax (__global__, blockIdx, the <<< >>> launch), so the kernels are here too.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "neb_host.h"
#include "neb_kernel.h"
extern "C" {
#include "ref/nebhash2.h"
}

#define NB_THREADS 256
// Nonces one screen launch takes, and so the most candidates it can list. About 2 ms on an RTX 4090.
#define NB_CHUNK (1u << 24)
// Table entries one build launch takes: short enough for Windows' 2-second limit on a kernel on a slow card.
#define NB_BUILD_CHUNK (1u << 17)
#define NB_EVAL_CHUNK 4096u
// Card memory left free when the pair table is sized.
#define NB_RESERVE (384ull << 20)

// ── kernels ─────────────────────────────────────────────────────────────────

__global__ void nbk_screen(NebScan S, nu32 count, const unsigned short *live, nu32 *ncand, NebCand *cand) {
  const nu32 gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < count) nebk_screen(gid, &S, live, ncand, cand);
}

__global__ void nbk_finish(NebScan S, nu32 count, const NebCand *cand, const nu32 *table, nu32 *hits) {
  const nu32 gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < count) nebk_finish(gid, &S, cand, table, hits);
}

__global__ void nbk_pair(NebRows R, nu32 first, nu32 count, nu32 pairs, const short *conv, const nu32 *tag,
                         const unsigned char *lane, const signed char *d5, const signed char *x5, const nu32 *zero5,
                         nu32 *table) {
  const nu32 gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < count) nebk_pair(first + gid, pairs, &R, conv, tag, lane, d5, x5, zero5, table);
}

__global__ void nbk_full(NebRows R, nu32 count, const nu32 *keys, const unsigned short *live, const short *conv,
                         const nu32 *tag, const unsigned char *lane, const signed char *d5, const signed char *x5,
                         nu32 *out) {
  const nu32 gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < count) nebk_full(keys + 8u * gid, &R, live, conv, tag, lane, d5, x5, out + 16u * gid);
}

// ── the core ────────────────────────────────────────────────────────────────

struct NbSlot {
  int arch = 0;
  uint32_t n = 0;
  std::vector<uint8_t> values, labels;
  bool ready = false;
  NebRows R;
  uint32_t max_live = 0;
  uint64_t entries = 0;
  unsigned short *d_live = NULL;
  short *d_conv = NULL;
  nu32 *d_tag = NULL;
  unsigned char *d_lane = NULL;
  signed char *d_d5 = NULL, *d_x5 = NULL;
  nu32 *d_zero5 = NULL, *d_table = NULL;
};

struct nb_core {
  int index;
  std::mutex mu;
  NbSlot slot[NB_SLOTS];
  NebCand *d_cand;
  nu32 *d_ncand, *d_hits, *d_keys, *d_out;
};

typedef G(Error_t) nb_err;

static int nb_fail(char *err, int errlen, const char *what, nb_err e) {
  if (err && errlen > 0) snprintf(err, (size_t)errlen, "%s: %s", what, G(GetErrorString)(e));
  G(GetLastError)();
  return -1;
}

static int nb_say(char *err, int errlen, const char *what) {
  if (err && errlen > 0) snprintf(err, (size_t)errlen, "%s", what);
  return -1;
}

#define NB_TRY(call, what)                                      \
  do {                                                          \
    const nb_err e_ = (call);                                   \
    if (e_ != G(Success)) return nb_fail(err, errlen, what, e_); \
  } while (0)

static double nb_ms(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

extern "C" int nb_device_count(void) {
  int n = 0;
  if (G(GetDeviceCount)(&n) != G(Success)) {
    G(GetLastError)();
    return 0;
  }
  return n;
}

extern "C" int nb_device_info(int index, char *name, int len) {
  NB_PROP p;
  if (G(GetDeviceProperties)(&p, index) != G(Success)) {
    G(GetLastError)();
    return -1;
  }
  if (name && len > 0) snprintf(name, (size_t)len, "%s", p.name);
  return NB_MIN_OK(p) ? 0 : -1;
}

extern "C" const char *nb_vendor(void) { return NB_VENDOR; }

static void nb_free_model(NbSlot *s) {
  G(Free)(s->d_live); G(Free)(s->d_conv); G(Free)(s->d_tag); G(Free)(s->d_lane);
  G(Free)(s->d_d5); G(Free)(s->d_x5); G(Free)(s->d_zero5); G(Free)(s->d_table);
  s->d_live = NULL; s->d_conv = NULL; s->d_tag = NULL; s->d_lane = NULL;
  s->d_d5 = NULL; s->d_x5 = NULL; s->d_zero5 = NULL; s->d_table = NULL;
  s->ready = false;
}

extern "C" void nb_close(nb_core *c) {
  if (!c) return;
  {
    std::lock_guard<std::mutex> lock(c->mu);
    if (G(SetDevice)(c->index) == G(Success)) {
      for (int i = 0; i < NB_SLOTS; i++) nb_free_model(&c->slot[i]);
      G(Free)(c->d_cand); G(Free)(c->d_ncand); G(Free)(c->d_hits); G(Free)(c->d_keys); G(Free)(c->d_out);
    }
    G(GetLastError)();
  }
  delete c;
}

static int nb_open_buffers(nb_core *c, char *err, int errlen) {
  NB_TRY(G(SetDevice)(c->index), "selecting the card");
  NB_TRY(G(Malloc)((void **)&c->d_cand, (size_t)NB_CHUNK * sizeof(NebCand)), "memory for the candidate list");
  NB_TRY(G(Malloc)((void **)&c->d_ncand, 4), "memory for the candidate count");
  NB_TRY(G(Malloc)((void **)&c->d_hits, ((size_t)NB_MAX_HITS + 1) * 4), "memory for the hits");
  NB_TRY(G(Malloc)((void **)&c->d_keys, (size_t)NB_EVAL_CHUNK * 32), "memory for the keys");
  NB_TRY(G(Malloc)((void **)&c->d_out, (size_t)NB_EVAL_CHUNK * 64), "memory for the tickets");
  return 0;
}

extern "C" nb_core *nb_open(int index, char *err, int errlen) {
  nb_core *c = new nb_core();
  c->index = index;
  c->d_cand = NULL;
  c->d_ncand = c->d_hits = c->d_keys = c->d_out = NULL;
  if (nb_open_buffers(c, err, errlen) != 0) {
    nb_close(c);
    return NULL;
  }
  return c;
}

extern "C" int nb_set_dataset(nb_core *c, int slot, int arch, uint32_t n, const uint8_t *values,
                              const uint8_t *labels, char *err, int errlen) {
  if (slot < 0 || slot >= NB_SLOTS) return nb_say(err, errlen, "no such dataset slot");
  const neb2_arch *A = neb2_arch_get(arch);
  // KEP1 only: a sample's direct gradients fit int16 there (583 * 8 * 7 = 32,648), and the main network has no other.
  if (!A || arch != NEB2_KEP1) return nb_say(err, errlen, "this core runs the KEP1 network only");
  if (n < 1 || n > 65534u) return nb_say(err, errlen, "a dataset of 1 to 65,534 samples");
  std::lock_guard<std::mutex> lock(c->mu);
  NbSlot *s = &c->slot[slot];
  if (G(SetDevice)(c->index) == G(Success)) nb_free_model(s);
  s->arch = arch;
  s->n = n;
  s->values.assign(values, values + (size_t)n * A->nin);
  s->labels.assign(labels, labels + n);
  return 0;
}

// One live sample's row on the host.
struct NbRow {
  std::vector<int16_t> conv;
  std::vector<int8_t> d5, x5;
};

template <typename T>
static int nb_upload(T **dst, const void *src, size_t bytes, char *err, int errlen, const char *what) {
  NB_TRY(G(Malloc)((void **)dst, bytes ? bytes : 1), what);
  if (bytes) NB_TRY(G(Memcpy)(*dst, src, bytes, G(MemcpyHostToDevice)), what);
  return 0;
}

extern "C" int nb_set_model(nb_core *c, int slot, const int8_t *weights, uint32_t nweights, nb_model_info *info,
                            char *err, int errlen) {
  if (slot < 0 || slot >= NB_SLOTS) return nb_say(err, errlen, "no such dataset slot");
  std::lock_guard<std::mutex> lock(c->mu);
  NbSlot *s = &c->slot[slot];
  if (!s->n) return nb_say(err, errlen, "the slot has no dataset");
  const neb2_arch *A = neb2_arch_get(s->arch);
  if (nweights != (uint32_t)A->nweights) return nb_say(err, errlen, "a model of the wrong size for this network");
  for (uint32_t i = 0; i < nweights; i++)
    if (weights[i] < -8 || weights[i] > 7) return nb_say(err, errlen, "a weight outside -8..7");

  // The factored layer is the largest linear one; every other layer's entries are kept directly.
  int lf = -1;
  for (int i = 0; i < A->nlayers; i++)
    if (A->l[i].kind == NEB2_LINEAR && (lf < 0 || A->l[i].nw > A->l[lf].nw)) lf = i;
  const neb2_layer &F = A->l[lf];
  if (F.cin % 8 != 0) return nb_say(err, errlen, "a factored layer whose width is not a multiple of 8");
  // Where the factored layer's errors and inputs sit in neb2_sample_grad's factors, and the last layer's errors.
  int fac_at = 0, fac_last = 0, at = 0;
  for (int i = 0; i < A->nlayers; i++) {
    if (A->l[i].kind != NEB2_LINEAR) continue;
    if (i == lf) fac_at = at;
    if (i == A->nlayers - 1) fac_last = at;
    at += A->l[i].cout + A->l[i].cin;
  }
  const int fac_len = at;
  if (A->l[A->nlayers - 1].kind != NEB2_LINEAR) return nb_say(err, errlen, "a network whose last layer is not linear");
  std::vector<uint32_t> src;     // model index of each direct entry
  std::vector<nu32> tag;
  std::vector<unsigned char> lane;
  for (int i = 0; i < A->nlayers; i++) {
    if (i == lf) continue;
    for (int f = 0; f < A->l[i].nw; f++) {
      src.push_back((uint32_t)(A->l[i].w_off + f));
      tag.push_back((nu32)f + 7919u * (nu32)(i + 1));
      lane.push_back((unsigned char)(f & 7));
    }
  }
  const uint32_t nc = (uint32_t)src.size();

  // Every sample once, on the host's threads, with the chain's reference.
  const auto t0 = std::chrono::steady_clock::now();
  std::unique_ptr<neb2_model> m(new neb2_model());
  memset(m.get(), 0, sizeof(neb2_model));
  m->arch = s->arch;
  memcpy(m->w, weights, nweights);
  std::vector<std::unique_ptr<NbRow>> rows(s->n);
  unsigned nt = std::thread::hardware_concurrency();
  if (nt == 0) nt = 4;
  if (nt > 32) nt = 32;
  std::vector<int> bad(nt, 0);
  std::vector<std::thread> th;
  for (unsigned t = 0; t < nt; t++)
    th.emplace_back([&, t] {
      std::vector<int32_t> g(NEB2_MAX_WEIGHTS);
      std::vector<int8_t> fac(2 * NEB2_MAX_WEIGHTS);
      for (uint32_t i = t; i < s->n; i += nt) {
        if (neb2_sample_grad(m.get(), s->values.data() + (size_t)i * A->nin, s->labels[i], g.data(), fac.data()) != 0) {
          bad[t] = 1;
          return;
        }
        bool live = false;
        for (int o = 0; o < A->l[A->nlayers - 1].cout; o++) live = live || fac[fac_last + o] != 0;
        if (!live) continue;
        std::unique_ptr<NbRow> r(new NbRow());
        r->conv.resize(nc);
        for (uint32_t e = 0; e < nc; e++) {
          const int32_t v = g[src[e]];
          if (v < -32768 || v > 32767) bad[t] = 2;
          r->conv[e] = (int16_t)v;
        }
        r->d5.assign(fac.begin() + fac_at, fac.begin() + fac_at + F.cout);
        r->x5.assign(fac.begin() + fac_at + F.cout, fac.begin() + fac_at + F.cout + F.cin);
        rows[i] = std::move(r);
      }
    });
  for (auto &t : th) t.join();
  (void)fac_len;
  for (int b : bad) {
    if (b == 1) return nb_say(err, errlen, "the reference could not evaluate a sample");
    if (b == 2) return nb_say(err, errlen, "a sample's gradient does not fit 16 bits");
  }
  // Row 0 is the zero row; live samples take rows 1 and up in dataset order.
  std::vector<unsigned short> live(s->n, 0);
  uint32_t nlive = 0;
  for (uint32_t i = 0; i < s->n; i++)
    if (rows[i]) live[i] = (unsigned short)(++nlive);
  const uint32_t m1 = nlive + 1;
  std::vector<int16_t> conv((size_t)m1 * nc, 0);
  std::vector<int8_t> d5((size_t)m1 * F.cout, 0), x5((size_t)m1 * F.cin, 0);
  for (uint32_t i = 0; i < s->n; i++) {
    if (!rows[i]) continue;
    const uint32_t r = live[i];
    memcpy(conv.data() + (size_t)r * nc, rows[i]->conv.data(), (size_t)nc * 2);
    memcpy(d5.data() + (size_t)r * F.cout, rows[i]->d5.data(), (size_t)F.cout);
    memcpy(x5.data() + (size_t)r * F.cin, rows[i]->x5.data(), (size_t)F.cin);
    rows[i].reset();
  }
  const nu32 base = 7919u * (nu32)(lf + 1);
  std::vector<nu32> zero5((size_t)F.cout * 8, 0);
  for (int o = 0; o < F.cout; o++)
    for (int j = 0; j < F.cin; j++) zero5[(size_t)o * 8 + (j & 7)] += (nu32)(o * F.cin + j) + base;
  const double cpu_ms = nb_ms(t0);

  // The card: the rows, then the table.
  const auto t1 = std::chrono::steady_clock::now();
  NB_TRY(G(SetDevice)(c->index), "selecting the card");
  nb_free_model(s);
  size_t free_b = 0, total_b = 0;
  NB_TRY(G(MemGetInfo)(&free_b, &total_b), "reading the card's free memory");
  const uint64_t pair_entries = (uint64_t)m1 * (m1 + 1) / 2;
  const bool pairs = pair_entries < 0xffffffffull && free_b > NB_RESERVE && pair_entries * 32 <= free_b - NB_RESERVE;
  const uint64_t entries = pairs ? pair_entries : m1;
  NebRows R;
  R.m1 = m1; R.nc = nc; R.cin = (nu32)F.cin; R.cout = (nu32)F.cout; R.base = base; R.n = s->n;
  if (nb_upload(&s->d_live, live.data(), live.size() * 2, err, errlen, "memory for the live list") != 0 ||
      nb_upload(&s->d_conv, conv.data(), conv.size() * 2, err, errlen, "memory for the rows") != 0 ||
      nb_upload(&s->d_tag, tag.data(), tag.size() * 4, err, errlen, "memory for the rows") != 0 ||
      nb_upload(&s->d_lane, lane.data(), lane.size(), err, errlen, "memory for the rows") != 0 ||
      nb_upload(&s->d_d5, d5.data(), d5.size(), err, errlen, "memory for the rows") != 0 ||
      nb_upload(&s->d_x5, x5.data(), x5.size(), err, errlen, "memory for the rows") != 0 ||
      nb_upload(&s->d_zero5, zero5.data(), zero5.size() * 4, err, errlen, "memory for the rows") != 0) {
    nb_free_model(s);
    return -1;
  }
  nb_err e = G(Malloc)((void **)&s->d_table, (size_t)entries * 32);
  if (e != G(Success)) {
    nb_free_model(s);
    return nb_fail(err, errlen, "memory for the model's table", e);
  }
  for (uint64_t done = 0; done < entries; done += NB_BUILD_CHUNK) {
    const nu32 part = (nu32)(entries - done < NB_BUILD_CHUNK ? entries - done : NB_BUILD_CHUNK);
    nbk_pair<<<(part + NB_THREADS - 1) / NB_THREADS, NB_THREADS>>>(R, (nu32)done, part, pairs ? 1u : 0u, s->d_conv,
                                                                   s->d_tag, s->d_lane, s->d_d5, s->d_x5, s->d_zero5,
                                                                   s->d_table);
    e = G(GetLastError)();
    if (e == G(Success) && (done / NB_BUILD_CHUNK) % 8 == 7) e = G(DeviceSynchronize)();
    if (e != G(Success)) {
      nb_free_model(s);
      return nb_fail(err, errlen, "building the model's table", e);
    }
  }
  e = G(DeviceSynchronize)();
  if (e != G(Success)) {
    nb_free_model(s);
    return nb_fail(err, errlen, "building the model's table", e);
  }
  s->R = R;
  s->max_live = pairs ? 2 : 1;
  s->entries = entries;
  s->ready = true;
  if (info) {
    info->samples = s->n;
    info->live = nlive;
    info->max_live = s->max_live;
    info->entries = entries;
    info->table_bytes = entries * 32;
    info->cpu_ms = cpu_ms;
    info->gpu_ms = nb_ms(t1);
  }
  return 0;
}

static nu32 nb_le32(const uint8_t *p) {
  return (nu32)p[0] | ((nu32)p[1] << 8) | ((nu32)p[2] << 16) | ((nu32)p[3] << 24);
}

extern "C" int nb_scan(nb_core *c, const nb_work *w, uint32_t *hits, uint32_t *stored, uint64_t *found,
                       uint64_t *evaluated, char *err, int errlen) {
  *stored = 0;
  *found = 0;
  *evaluated = 0;
  if (w->slot < 0 || w->slot >= NB_SLOTS) return nb_say(err, errlen, "no such dataset slot");
  if (w->count < 1 || w->count > 4294967296ull - w->start) return nb_say(err, errlen, "start + count passes 2^32");
  std::lock_guard<std::mutex> lock(c->mu);
  NbSlot *s = &c->slot[w->slot];
  if (!s->ready) return nb_say(err, errlen, "the slot has no model");
  NebScan S;
  // BLAKE3's state after the header's first block: CHUNK_START, 64 bytes.
  nu32 blk[16];
  const nu32 iv[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au, 0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};
  for (int i = 0; i < 16; i++) blk[i] = nb_le32(w->header + 4 * i);
  nebk_b3(iv, blk, 64u, 1u, S.cv);
  for (int i = 0; i < 3; i++) S.tail[i] = nb_le32(w->header + 64 + 4 * i);
  for (int i = 0; i < 8; i++) S.bound[i] = nb_le32(w->target + 4 * i);
  S.n = s->n;
  S.max_live = s->max_live;
  S.cap = NB_CHUNK;
  S.max_hits = w->max_hits > NB_MAX_HITS ? NB_MAX_HITS : w->max_hits;
  NB_TRY(G(SetDevice)(c->index), "selecting the card");
  NB_TRY(G(Memset)(c->d_hits, 0, 4), "clearing the hits");
  for (uint64_t done = 0; done < w->count; done += NB_CHUNK) {
    const nu32 part = (nu32)(w->count - done < NB_CHUNK ? w->count - done : NB_CHUNK);
    S.start = w->start + (nu32)done;
    NB_TRY(G(Memset)(c->d_ncand, 0, 4), "clearing the candidate count");
    nbk_screen<<<(part + NB_THREADS - 1) / NB_THREADS, NB_THREADS>>>(S, part, s->d_live, c->d_ncand, c->d_cand);
    NB_TRY(G(GetLastError)(), "the screen kernel");
    nu32 ncand = 0;
    NB_TRY(G(Memcpy)(&ncand, c->d_ncand, 4, G(MemcpyDeviceToHost)), "the screen kernel");
    if (ncand > S.cap) return nb_say(err, errlen, "the screen listed more candidates than nonces");
    *evaluated += ncand;
    if (!ncand) continue;
    nbk_finish<<<(ncand + NB_THREADS - 1) / NB_THREADS, NB_THREADS>>>(S, ncand, c->d_cand, s->d_table, c->d_hits);
    NB_TRY(G(GetLastError)(), "the ticket kernel");
  }
  nu32 n = 0;
  NB_TRY(G(Memcpy)(&n, c->d_hits, 4, G(MemcpyDeviceToHost)), "the ticket kernel");
  *found = n;
  const nu32 keep = n < S.max_hits ? n : S.max_hits;
  if (keep) NB_TRY(G(Memcpy)(hits, c->d_hits + 1, (size_t)keep * 4, G(MemcpyDeviceToHost)), "reading the hits");
  *stored = keep;
  return 0;
}

extern "C" int nb_eval(nb_core *c, int slot, const uint8_t *keys, uint32_t nkeys, uint8_t *out, char *err,
                       int errlen) {
  if (slot < 0 || slot >= NB_SLOTS) return nb_say(err, errlen, "no such dataset slot");
  std::lock_guard<std::mutex> lock(c->mu);
  NbSlot *s = &c->slot[slot];
  if (!s->ready) return nb_say(err, errlen, "the slot has no model");
  NB_TRY(G(SetDevice)(c->index), "selecting the card");
  for (uint32_t done = 0; done < nkeys; done += NB_EVAL_CHUNK) {
    const nu32 part = nkeys - done < NB_EVAL_CHUNK ? nkeys - done : NB_EVAL_CHUNK;
    // Keys and tickets are bytes in little-endian word order, which is the host's.
    NB_TRY(G(Memcpy)(c->d_keys, keys + (size_t)done * 32, (size_t)part * 32, G(MemcpyHostToDevice)), "sending the keys");
    nbk_full<<<(part + 63) / 64, 64>>>(s->R, part, c->d_keys, s->d_live, s->d_conv, s->d_tag, s->d_lane, s->d_d5,
                                       s->d_x5, c->d_out);
    NB_TRY(G(GetLastError)(), "the full kernel");
    NB_TRY(G(Memcpy)(out + (size_t)done * 64, c->d_out, (size_t)part * 64, G(MemcpyDeviceToHost)), "the full kernel");
  }
  return 0;
}
