// The host side of the Quantus AMD core: the card list, opening a card, and a scan, as a small C API
// that qh_napi.cc wraps. Built with hip-clang (-x hip) together with the kernel (qh_kernel.h).

#include "qh_host.h"

#include <hip/hip_runtime.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <mutex>

#include "qh_kernel.h"

#ifndef QH_TARGETS
#define QH_TARGETS "?"
#endif

// Counters a launch covers. A scan of 2^32 is 2^10 launches, all queued at once; each takes well under a
// second, so a card's queue never holds one kernel long enough for a driver timeout.
#ifndef QH_LAUNCH_N
#define QH_LAUNCH_N (1u << 22)
#endif

namespace {

typedef unsigned __int128 u128;

const u64 P = 0xffffffff00000001ull;

// qp-poseidon 3.1.0's constants (src/shared/quantus/poseidon2.js has them with their licence).
const u64 INTERNAL[22] = {
  0x97f7798a784ad863ull, 0xd1d2bf082f60d4f0ull, 0x69a377a79f9ad206ull, 0xa9d06906a3858e24ull,
  0x295275001eede5b5ull, 0x5874e441117bd746ull, 0x8a084bbba8ed86ccull, 0x3defd7645cde6425ull,
  0x3998cfe6871cc137ull, 0x3e52ef8bca48314aull, 0x964a209f85dc9eccull, 0x3fcc9ee82cc4577eull,
  0x8e79b4a5d0096d6dull, 0x8492362ad2392556ull, 0xee72f470262574d6ull, 0x1e0e18496da2444aull,
  0x0f3a74bf215eaac6ull, 0x1b061b76a1c0ded3ull, 0x192c42d86803d7a6ull, 0xf6d49ff997ae0260ull,
  0x3ec372e7a0fa3786ull, 0x5538cdf4f23445d3ull,
};
const u64 DIAG[12] = {
  0xc3b6c08e23ba9300ull, 0xd84b5de94a324fb6ull, 0x0d0c371c5b35b84full, 0x7964f570e7188037ull,
  0x5daf18bbd996604bull, 0x6743bc47b9595257ull, 0x5528b9362c59bb70ull, 0xac45e25b7127b68bull,
  0xa2077d7dfbb606b5ull, 0xf3faac6faee378aeull, 0x0c6388b51545e883ull, 0xd27dbb6944917b60ull,
};
const u64 INIT[4][12] = {
  {0xc002e770975b1607ull, 0xbca51a8dfe14593aull, 0x72938dfbe774f7f9ull, 0xe4f2fe29e03234acull, 0xd5e0ba2f541b6449ull,
   0xec33b868f3cc46c1ull, 0x486dcb55419d475aull, 0x6c1cb2a358cc24f1ull, 0xe3f30d509a1436bbull, 0xd9a64f068dca7c29ull,
   0xe59b3f57aabba1aeull, 0x2a3dd4505b478fdcull},
  {0xada1f8dc7676ed25ull, 0x2711aa8b5509d516ull, 0x4ae6acd0c9c92897ull, 0x56eb3d6b5256d67aull, 0x1f7a9d55923bf51eull,
   0x3600427d397a7f68ull, 0xe5076df75b72c3d0ull, 0xfcd59aa12c6090adull, 0xcd895e8c68b57a9eull, 0x41df7ef9d730ae3eull,
   0xee3e2b889abe977dull, 0xd29bb7edbeb9c405ull},
  {0x7d5c08eef608e382ull, 0x89ae889caaf0802cull, 0xb35a8e976d2af617ull, 0xdb14234eafaf5173ull, 0x78f04462d48b1c98ull,
   0x265293b0e47ce88aull, 0x999a649b69b9d32full, 0x64b0a186698e01d3ull, 0xee0b22d0dfae8bb8ull, 0x4fd53e50ca04a7eeull,
   0x5762bfe181f25047ull, 0xf51593e2beb5e3bdull},
  {0x1e5e2b5760e32477ull, 0x622462a1f9aaaeedull, 0xaa284b3ecdb222aeull, 0x63c8e72f542bf3fcull, 0x3ba588cacb43b5e0ull,
   0x23eda6f3c99150ddull, 0xaad3bea4baac9a5aull, 0xe9da8d699b94184aull, 0xcdb13f4cd93e024cull, 0x902cbd0956f655e3ull,
   0x5b4e40ffc759532full, 0xde795c20a2357af7ull},
};
const u64 TERM[4][12] = {
  {0x7b72c539e0ea4c6eull, 0x144573dae2ce9976ull, 0x802028b68f35fc88ull, 0x6d36c5022c4fe7c2ull, 0xa205d0ffa9b9def3ull,
   0xf6e7e38b1ea6ba2full, 0x34f7909ae5258d64ull, 0xb0464d9d77b97fcaull, 0x64ddb9d5de7e00a6ull, 0x0ed0d75c27975d97ull,
   0x1cbb36f11127338bull, 0x6673e505cfd0b6baull},
  {0x605f902830872e01ull, 0x3fd5eb927e95fe4full, 0xe81025b5a24c69cdull, 0xf7d0ce75de23f74eull, 0xf39942b6a8585089ull,
   0x6d808a08f7b71df6ull, 0xf8806b6588f49a8bull, 0x57df2d8c2a32107aull, 0x16e7c2074d654a2dull, 0x213de241fcf33835ull,
   0xb0f2b8905a0976f6ull, 0xd8e3cf2bbd355417ull},
  {0xe498691679d9330full, 0x763b45d2a3821b28ull, 0x0908bf65eb0a1f0dull, 0x7691eb2d194b24f4ull, 0x0e43551233ae13b2ull,
   0x93c393dbfc2fe76full, 0x98f607485d48cdeaull, 0xe3d95f30309819c0ull, 0x1ef581a93eaf6acfull, 0x0b24c1b7a030fca4ull,
   0x624370be5670b327ull, 0x5f1e28615a11e486ull},
  {0xfe04051f909e042bull, 0x7257e5b147fd3803ull, 0xe6ae134bb82f2e78ull, 0x5711fd5cf4784511ull, 0xf83a42660c08c0bcull,
   0x2cd8c96d9a3ce855ull, 0x7d2ffb1bb0e17271ull, 0x85ae1528caea3811ull, 0x52a345d5c7adb0b8ull, 0x504c4c51f3faee94ull,
   0xbce34a649cfccaf9ull, 0xe0a3389266fb6dc9ull},
};

u64 addp(u64 a, u64 b) { return (u64)(((u128)a + b) % P); }

// The external layer on canonical values (the host's one use of it: block 3's first layer).
void external(u64 *s) {
  for (int o = 0; o < 12; o += 4) {
    u128 x0 = s[o], x1 = s[o + 1], x2 = s[o + 2], x3 = s[o + 3];
    s[o] = (u64)((2 * x0 + 3 * x1 + x2 + x3) % P);
    s[o + 1] = (u64)((x0 + 2 * x1 + 3 * x2 + x3) % P);
    s[o + 2] = (u64)((x0 + x1 + 2 * x2 + 3 * x3) % P);
    s[o + 3] = (u64)((3 * x0 + x1 + x2 + 2 * x3) % P);
  }
  for (int k = 0; k < 4; k++) {
    u128 sum = (u128)s[k] + s[k + 4] + s[k + 8];
    for (int j = k; j < 12; j += 4) s[j] = (u64)((s[j] + sum) % P);
  }
}

void build_tables(QhTables *t) {
  memset(t, 0, sizeof(*t));
  for (int i = 0; i < 12; i++) {
    t->full[0][i] = INIT[1][i];
    t->full[1][i] = INIT[2][i];
    t->full[2][i] = INIT[3][i];
    t->full[4][i] = TERM[1][i];
    t->full[5][i] = TERM[2][i];
    t->full[6][i] = TERM[3][i];
    t->full[8][i] = INIT[0][i];
    t->full[9][i] = INIT[1][i];
    t->full[10][i] = INIT[2][i];
    t->full[11][i] = INIT[3][i];
    t->full[13][i] = TERM[1][i];
    t->full[14][i] = TERM[2][i];
    t->full[15][i] = TERM[3][i];
    t->term0[i] = TERM[0][i];
    t->diag[i] = DIAG[i];
  }
  t->full[3][0] = INTERNAL[0];
  t->full[7][0] = 1;
  t->full[7][1] = 1;
  t->full[12][0] = INTERNAL[0];
  for (int r = 0; r < 21; r++) t->part0[r] = INTERNAL[r + 1];
  t->part0[21] = TERM[0][0];
}

u32 bswap32(u32 v) { return __builtin_bswap32(v); }

void say(char *err, size_t len, const char *fmt, ...) {
  if (!err || !len) return;
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(err, len, fmt, ap);
  va_end(ap);
}

bool ok(hipError_t e, const char *what, char *err, size_t len) {
  if (e == hipSuccess) return true;
  say(err, len, "%s: %s", what, hipGetErrorString(e));
  return false;
}

}  // namespace

struct QhCore {
  int dev;
  hipStream_t stream;
  u64 *hits;       // [count, hit, hit, ...] on the card
  u32 cap;         // hit slots in `hits`
  u64 *host;       // pinned copy of it
  u32 npt;         // nonces a thread in this card's code object (qh_npt)
  std::atomic<bool> closing;
};

extern "C" int qh_device_count(char *err, size_t len) {
  int n = 0;
  hipError_t e = hipGetDeviceCount(&n);
  if (e == hipErrorNoDevice) return 0;
  if (!ok(e, "hipGetDeviceCount", err, len)) return -1;
  return n;
}

extern "C" int qh_device_info(int index, QhDeviceInfo *info, char *err, size_t len) {
  hipDeviceProp_t p;
  if (!ok(hipGetDeviceProperties(&p, index), "hipGetDeviceProperties", err, len)) return -1;
  memset(info, 0, sizeof(*info));
  snprintf(info->name, sizeof(info->name), "%s", p.name);
  snprintf(info->arch, sizeof(info->arch), "%s", p.gcnArchName);
  char *colon = strchr(info->arch, ':');  // "gfx1032:xnack-" -> "gfx1032"
  if (colon) *colon = 0;
  snprintf(info->pci, sizeof(info->pci), "%04x:%02x:%02x.0", p.pciDomainID, p.pciBusID, p.pciDeviceID);
  info->cus = p.multiProcessorCount;
  info->clock_khz = p.clockRate;
  return 0;
}

extern "C" QhCore *qh_open(int index, char *err, size_t len) {
  int n = qh_device_count(err, len);
  if (n < 0) return nullptr;
  if (index < 0 || index >= n) {
    say(err, len, "no AMD card %d (HIP lists %d)", index, n);
    return nullptr;
  }
  if (!ok(hipSetDevice(index), "hipSetDevice", err, len)) return nullptr;
  // A card whose target this build has no code for fails here, not on the first scan.
  hipFuncAttributes fa;
  hipError_t fe = hipFuncGetAttributes(&fa, reinterpret_cast<const void *>(qh_search));
  if (fe != hipSuccess) {
    QhDeviceInfo info;
    qh_device_info(index, &info, nullptr, 0);
    say(err, len, "this core has no code for card %d (%s, %s): built for %s (%s)", index, info.name, info.arch,
        QH_TARGETS, hipGetErrorString(fe));
    return nullptr;
  }
  QhTables t;
  build_tables(&t);
  if (!ok(hipMemcpyToSymbol(HIP_SYMBOL(qh_tab), &t, sizeof(t)), "constant tables", err, len)) return nullptr;
  u32 npt = 1;
  if (!ok(hipMemcpyFromSymbol(&npt, HIP_SYMBOL(qh_npt), sizeof(npt)), "nonces a thread", err, len)) return nullptr;
  if (npt < 1 || npt > 8) npt = 1;
  QhCore *c = new QhCore();
  c->npt = npt;
  c->dev = index;
  c->cap = 0;
  c->hits = nullptr;
  c->host = nullptr;
  c->closing = false;
  if (!ok(hipStreamCreateWithFlags(&c->stream, hipStreamNonBlocking), "hipStreamCreate", err, len)) {
    delete c;
    return nullptr;
  }
  return c;
}

static bool reserve(QhCore *c, u32 cap, char *err, size_t len) {
  if (cap <= c->cap && c->hits) return true;
  u32 want = cap < 64 ? 64 : cap;
  if (c->hits) hipFree(c->hits);
  if (c->host) hipHostFree(c->host);
  c->hits = nullptr;
  c->host = nullptr;
  c->cap = 0;
  if (!ok(hipMalloc(&c->hits, (size_t)(want + 1) * 8), "hipMalloc", err, len)) return false;
  if (!ok(hipHostMalloc(&c->host, (size_t)(want + 1) * 8, 0), "hipHostMalloc", err, len)) return false;
  c->cap = want;
  return true;
}

extern "C" int qh_scan(QhCore *c, const QhWork *w, u64 *hits, u32 *nhits, int *overflow, char *err, size_t len) {
  if (c->closing) {
    say(err, len, "the core is closed");
    return -1;
  }
  if (w->count < 1 || w->count > (1ull << 32) || w->start + (w->count - 1) < w->start) {
    say(err, len, "count must be 1 to 2^32 and stay below 2^64");
    return -1;
  }
  if (!ok(hipSetDevice(c->dev), "hipSetDevice", err, len)) return -1;
  u32 maxHits = w->max_hits;
  if (!reserve(c, maxHits, err, len)) return -1;
  if (!ok(hipMemsetAsync(c->hits, 0, 8, c->stream), "hipMemset", err, len)) return -1;

  u64 done = 0;
  while (done < w->count) {
    u64 ctr = w->start + done;
    u32 hi = (u32)(ctr >> 32), lo = (u32)ctr;
    // Counters up to the next 2^32 boundary share word 6, so the host folds it in once.
    u64 left = w->count - done;
    u64 to_wrap = (1ull << 32) - lo;
    u64 run = left < to_wrap ? left : to_wrap;
    QhArgs a;
    u64 s[12];
    for (int i = 0; i < 12; i++) s[i] = w->mid[i];
    for (int i = 0; i < 6; i++) {
      u32 word = (u32)w->tail[4 * i] | ((u32)w->tail[4 * i + 1] << 8) | ((u32)w->tail[4 * i + 2] << 16) |
                 ((u32)w->tail[4 * i + 3] << 24);
      s[i] = addp(s[i], word);
    }
    s[6] = addp(s[6], bswap32(hi));
    external(s);
    for (int i = 0; i < 12; i++) a.k[i] = addp(s[i], INIT[0][i]);
    a.target = w->target;
    a.hi = hi;
    a.cap = c->cap < maxHits ? c->cap : maxHits;
    for (u64 off = 0; off < run; off += QH_LAUNCH_N) {
      if (c->closing) break;
      u64 n = run - off < QH_LAUNCH_N ? run - off : QH_LAUNCH_N;
      a.lo0 = lo + (u32)off;
      a.n = (u32)n;
      u32 blocks = (u32)((n + (u64)QH_BLOCK * c->npt - 1) / ((u64)QH_BLOCK * c->npt));
      hipLaunchKernelGGL(qh_search, dim3(blocks), dim3(QH_BLOCK), 0, c->stream, a, c->hits);
      if (!ok(hipGetLastError(), "kernel launch", err, len)) return -1;
    }
    done += run;
  }
  if (!ok(hipMemcpyAsync(c->host, c->hits, (size_t)(c->cap + 1) * 8, hipMemcpyDeviceToHost, c->stream), "hipMemcpy",
          err, len))
    return -1;
  if (!ok(hipStreamSynchronize(c->stream), "the scan", err, len)) return -1;
  if (c->closing) {
    say(err, len, "the core is closed");
    return -1;
  }
  u64 found = c->host[0];
  u32 kept = found < maxHits ? (u32)found : maxHits;
  for (u32 i = 0; i < kept; i++) hits[i] = c->host[1 + i];
  *nhits = kept;
  *overflow = found > maxHits ? 1 : 0;
  return 0;
}

extern "C" void qh_cancel(QhCore *c) { c->closing = true; }

extern "C" void qh_close(QhCore *c) {
  if (!c) return;
  hipSetDevice(c->dev);
  hipStreamSynchronize(c->stream);
  if (c->hits) hipFree(c->hits);
  if (c->host) hipHostFree(c->host);
  hipStreamDestroy(c->stream);
  delete c;
}
