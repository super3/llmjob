// Host side of the Quantus CUDA core: cards, launches, the prestate. See
// qc_host.h for the interface and qc_kernel.cuh for the kernel.
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mutex>

#include "qc_host.h"
#include "qc_kernel.cuh"

// Counters one launch takes at most: about 50 ms on an RTX 4090, well inside
// Windows' 2-second limit on a kernel even on a slow card.
#define QC_CHUNK (1u << 26)

struct qc_core {
  int index;
  cudaStream_t stream;
  unsigned long long *d_nhits;
  u64 *d_hits;
  u32 cap;
  int grid;  // blocks in one full wave
  std::mutex mu;
};

static void set_err(char *err, int errlen, const char *what, cudaError_t e) {
  if (err && errlen > 0) snprintf(err, errlen, "%s: %s", what, cudaGetErrorString(e));
}

// ── the prestate, on the host ───────────────────────────────────────────────

static const u64 HP = 0xFFFFFFFF00000001ull;

static const u64 H_RC_I0[12] = {
    0xc002e770975b1607ull, 0xbca51a8dfe14593aull, 0x72938dfbe774f7f9ull, 0xe4f2fe29e03234acull,
    0xd5e0ba2f541b6449ull, 0xec33b868f3cc46c1ull, 0x486dcb55419d475aull, 0x6c1cb2a358cc24f1ull,
    0xe3f30d509a1436bbull, 0xd9a64f068dca7c29ull, 0xe59b3f57aabba1aeull, 0x2a3dd4505b478fdcull};

// a + b mod p for canonical a and b; canonical result.
static u64 hadd(u64 a, u64 b) {
  u64 r = a + b;
  if (r < a) return r + 0xFFFFFFFFull;  // a + b - p = r + 2^64 - p
  return r >= HP ? r - HP : r;
}

static void hmat4(u64 *x) {
  u64 t01 = hadd(x[0], x[1]), t23 = hadd(x[2], x[3]), t0123 = hadd(t01, t23);
  u64 t01123 = hadd(t0123, x[1]), t01233 = hadd(t0123, x[3]);
  u64 y3 = hadd(hadd(t01233, x[0]), x[0]);
  u64 y1 = hadd(hadd(t01123, x[2]), x[2]);
  x[0] = hadd(t01123, t01);
  x[1] = y1;
  x[2] = hadd(t01233, t23);
  x[3] = y3;
}

static void hext(u64 *s) {
  for (int g = 0; g < 12; g += 4) hmat4(s + g);
  for (int k = 0; k < 4; k++) {
    u64 col = hadd(hadd(s[k], s[k + 4]), s[k + 8]);
    for (int g = 0; g < 12; g += 4) s[g + k] = hadd(s[g + k], col);
  }
}

static u32 bswap32h(u32 v) {
  return (v >> 24) | ((v >> 8) & 0xff00u) | ((v << 8) & 0xff0000u) | (v << 24);
}

// The state after the first permutation's initial layer and round constants,
// for counters whose high word is `hi`, without the counter's low word (the
// kernel adds that).
static void prestate(const qc_work *w, u32 hi, u64 *c) {
  for (int i = 0; i < 12; i++) c[i] = w->mid[i];
  for (int i = 0; i < 6; i++) c[i] = hadd(c[i], w->tail[i]);
  c[6] = hadd(c[6], bswap32h(hi));
  hext(c);
  for (int i = 0; i < 12; i++) c[i] = hadd(c[i], H_RC_I0[i]);
}

// ── cards ───────────────────────────────────────────────────────────────────

extern "C" int qc_device_count(char *err, int errlen) {
  int n = 0;
  cudaError_t e = cudaGetDeviceCount(&n);
  if (e != cudaSuccess) {
    set_err(err, errlen, "cudaGetDeviceCount", e);
    cudaGetLastError();
    return 0;
  }
  return n;
}

extern "C" int qc_device_info(int index, char *name, int len, int *major, int *minor) {
  cudaDeviceProp p;
  if (cudaGetDeviceProperties(&p, index) != cudaSuccess) {
    cudaGetLastError();
    return -1;
  }
  if (name && len > 0) snprintf(name, len, "%s", p.name);
  if (major) *major = p.major;
  if (minor) *minor = p.minor;
  return 0;
}

static int grow(qc_core *c, u32 cap, char *err, int errlen) {
  if (cap <= c->cap) return 0;
  if (c->d_hits) cudaFree(c->d_hits);
  c->d_hits = NULL;
  c->cap = 0;
  cudaError_t e = cudaMalloc((void **)&c->d_hits, (size_t)cap * sizeof(u64));
  if (e != cudaSuccess) {
    set_err(err, errlen, "cudaMalloc (hits)", e);
    return -1;
  }
  c->cap = cap;
  return 0;
}

extern "C" qc_core *qc_open(int index, char *err, int errlen) {
  cudaError_t e = cudaSetDevice(index);
  if (e != cudaSuccess) {
    set_err(err, errlen, "cudaSetDevice", e);
    return NULL;
  }
  // Wait for a launch by sleeping, not by spinning a CPU core. Refused when the
  // card's context already exists (another core in this process): harmless.
  cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
  cudaGetLastError();
  cudaFuncAttributes fa;
  e = cudaFuncGetAttributes(&fa, qc_scan_kernel);
  if (e != cudaSuccess) {
    set_err(err, errlen, "this build has no kernel for this card", e);
    cudaGetLastError();
    return NULL;
  }
  int sms = 0, per = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, index);
  e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, qc_scan_kernel, QC_THREADS, 0);
  if (e != cudaSuccess || per < 1) per = 1;
  qc_core *c = new qc_core();
  c->index = index;
  c->grid = sms * per;
  const char *g = getenv("QUANTUS_CUDA_GRID");  // for tuning runs only
  if (g && atoi(g) > 0) c->grid = atoi(g);
  c->d_nhits = NULL;
  c->d_hits = NULL;
  c->cap = 0;
  e = cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking);
  if (e != cudaSuccess) {
    set_err(err, errlen, "cudaStreamCreate", e);
    delete c;
    return NULL;
  }
  e = cudaMalloc((void **)&c->d_nhits, sizeof(unsigned long long));
  if (e != cudaSuccess || grow(c, 256, err, errlen)) {
    if (e != cudaSuccess) set_err(err, errlen, "cudaMalloc", e);
    qc_close(c);
    return NULL;
  }
  return c;
}

extern "C" int qc_scan(qc_core *c, const qc_work *w, uint64_t *hits, uint32_t *stored, uint64_t *found, char *err,
                       int errlen) {
  std::lock_guard<std::mutex> lock(c->mu);
  *stored = 0;
  *found = 0;
  cudaError_t e = cudaSetDevice(c->index);
  if (e != cudaSuccess) {
    set_err(err, errlen, "cudaSetDevice", e);
    return -1;
  }
  u32 keep = w->max_hits < QC_MAX_SLOTS ? w->max_hits : QC_MAX_SLOTS;
  if (grow(c, keep > 256 ? keep : 256, err, errlen)) return -1;
  e = cudaMemsetAsync(c->d_nhits, 0, sizeof(unsigned long long), c->stream);
  if (e != cudaSuccess) {
    set_err(err, errlen, "cudaMemsetAsync", e);
    return -1;
  }
  QcLaunch L;
  L.target = w->target;
  L.cap = keep;
  L.zero = 0;
  u64 at = w->start, left = w->count;
  u32 hi = 0;
  int have = 0;
  while (left) {
    // A launch never changes the counter's high word: it stops at the next
    // multiple of 2^32 (the "exact" check in verify-core.js crosses one).
    u64 room = (1ull << 32) - (u32)at;
    u64 n = left < room ? left : room;
    if (n > QC_CHUNK) n = QC_CHUNK;
    if (!have || hi != (u32)(at >> 32)) {
      hi = (u32)(at >> 32);
      prestate(w, hi, L.c);
      have = 1;
    }
    L.base = at;
    L.count = (u32)n;
    u64 blocks = (n + QC_THREADS - 1) / QC_THREADS;
    int grid = blocks < (u64)c->grid ? (int)blocks : c->grid;
    qc_scan_kernel<<<grid, QC_THREADS, 0, c->stream>>>(L, c->d_nhits, c->d_hits);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
      set_err(err, errlen, "kernel launch", e);
      return -1;
    }
    at += n;  // wraps to 0 after the last counter, when left is then 0
    left -= n;
  }
  unsigned long long nh = 0;
  e = cudaMemcpyAsync(&nh, c->d_nhits, sizeof(nh), cudaMemcpyDeviceToHost, c->stream);
  if (e == cudaSuccess) e = cudaStreamSynchronize(c->stream);
  if (e != cudaSuccess) {
    set_err(err, errlen, "scan", e);
    return -1;
  }
  u64 n = nh < keep ? nh : keep;
  if (n) {
    e = cudaMemcpyAsync(hits, c->d_hits, n * sizeof(u64), cudaMemcpyDeviceToHost, c->stream);
    if (e == cudaSuccess) e = cudaStreamSynchronize(c->stream);
    if (e != cudaSuccess) {
      set_err(err, errlen, "reading hits", e);
      return -1;
    }
  }
  *stored = (u32)n;
  *found = nh;
  return 0;
}

extern "C" void qc_close(qc_core *c) {
  if (!c) return;
  {
    std::lock_guard<std::mutex> lock(c->mu);
    if (cudaSetDevice(c->index) == cudaSuccess) {
      if (c->stream) cudaStreamSynchronize(c->stream);
      if (c->d_hits) cudaFree(c->d_hits);
      if (c->d_nhits) cudaFree(c->d_nhits);
      if (c->stream) cudaStreamDestroy(c->stream);
    }
    cudaGetLastError();
  }
  delete c;
}
