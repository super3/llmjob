// Host side of the Quantus CUDA core: cards and launches. See
// qc_host.h for the interface and qc_kernel.cuh for the kernel.
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mutex>
#include <vector>

#include "qc_host.h"
#include "qc_kernel.cuh"
#include "qc_cubin.inc"

// Points one launch takes at most: about 50 ms on an RTX 4090, well inside
// Windows' 2-second limit on a kernel even on a slow card. The kernel counts
// on it too: 35 * (QC_CHUNK - 1) must fit a u32 (qc_scan_kernel).
#define QC_CHUNK (1u << 26)

struct qc_core {
  int index;
  cudaStream_t stream;
  unsigned long long *d_nhits;
  u64 *d_hits;
  u32 cap;
  int grid;  // blocks in one full wave
  int threads;  // threads a block: the kernel's own bound for this card (QC_THREADS_K)
  QcCubin k;  // the patched kernel, when this card runs it (qc_cubin.inc)
  std::mutex mu;
};

// Opens and checks take turns, so two never run the patched kernel's check on a card at once.
static std::mutex g_open_mu;

static void set_err(char *err, int errlen, const char *what, cudaError_t e) {
  if (err && errlen > 0) snprintf(err, errlen, "%s: %s", what, cudaGetErrorString(e));
}

// ── a launch's start on the line, on the host ───────────────────────────────

static const u64 HP = 0xFFFFFFFF00000001ull;

// a + b mod p for canonical a and b; canonical result.
static u64 hadd(u64 a, u64 b) {
  u64 r = a + b;
  if (r < a) return r + 0xFFFFFFFFull;  // a + b - p = r + 2^64 - p
  return r >= HP ? r - HP : r;
}

// The launch that starts at point `at` of the line: element 0 there is
// X0 + 35 at and element 4 is X4 - 35 at (qpow.js, the nonce layout). at is
// below 2^32, so 35 at is far below p.
static void launch_at(const qc_work *w, u64 at, QcLaunch *L) {
  const u64 d = QC_SLOPE * at;
  for (int i = 0; i < 12; i++) L->c[i] = w->line[i];
  L->x0 = hadd(w->line[12], d);
  L->x4 = hadd(w->line[13], HP - d);
  L->base = at;
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

static void free_core(qc_core *c);

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

// Selects card `index` and finds the compiled kernel's block size and one full wave there. 0, or -1 with err set.
static int card_shape(int index, int *threads, int *grid, char *err, int errlen) {
  cudaError_t e = cudaSetDevice(index);
  if (e != cudaSuccess) {
    set_err(err, errlen, "cudaSetDevice", e);
    return -1;
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
    return -1;
  }
  int sms = 0, per = 0;
  *threads = fa.maxThreadsPerBlock > 0 ? fa.maxThreadsPerBlock : QC_THREADS;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, index);
  e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, qc_scan_kernel, *threads, 0);
  if (e != cudaSuccess || per < 1) per = 1;
  cudaGetLastError();
  *grid = sms * per;
  return 0;
}

extern "C" int qc_cubin_wanted(int index) {
  std::vector<char> img;
  const char *what = "";
  int major = 0, minor = 0;
  qc_cubin_image(index, &img, &what, &major, &minor);
  return img.empty() ? 0 : 1;
}

extern "C" int qc_cubin_test(int index, int *lost, char *why, int whylen) {
  std::lock_guard<std::mutex> open_lock(g_open_mu);
  *lost = 0;
  if (!qc_cubin_wanted(index)) {
    if (why && whylen > 0) snprintf(why, whylen, "this card has none");
    return -1;
  }
  int threads = 0, grid = 0;
  if (card_shape(index, &threads, &grid, why, whylen)) return 0;
  QcCubin k;
  bool l = false;
  const char *w = "";
  const bool ok = qc_cubin_open(&k, index, grid, threads, &l, &w);
  if (ok) {
    k.unload(k.mod);
    cudaGetLastError();
  }
  *lost = l ? 1 : 0;
  if (why && whylen > 0) snprintf(why, whylen, "%s", ok ? "" : w);
  return ok ? 1 : 0;
}

extern "C" qc_core *qc_open(int index, char *err, int errlen) { return qc_open2(index, 1, err, errlen); }

extern "C" qc_core *qc_open2(int index, int cubin, char *err, int errlen) {
  std::lock_guard<std::mutex> open_lock(g_open_mu);
  int threads = 0, grid = 0;
  if (card_shape(index, &threads, &grid, err, errlen)) return NULL;
  cudaError_t e;
  qc_core *c = new qc_core();
  c->index = index;
  c->threads = threads;
  c->grid = grid;
  const char *g = getenv("QUANTUS_CUDA_GRID");  // for tuning runs only
  if (g && atoi(g) > 0) c->grid = atoi(g);
  c->d_nhits = NULL;
  c->d_hits = NULL;
  c->cap = 0;
  c->stream = NULL;
  // The patched kernel first, before the core makes anything on the card. A check that faults leaves the card's
  // context dead for the rest of this process: no reset brings it back (cudaDeviceReset, then every call answers
  // "busy or unavailable": RTX 4090 on Windows, driver 610.88, and RTX 3060 on Linux, driver 617.42, 28 Nov). So the
  // app checks in a child process first (qc_cubin_test) and opens with cubin 0 when that failed; should the check
  // still fault here, the open fails with a clear error instead of handing back a core whose first scan fails.
  bool lost = false;
  const char *why = "";
  if (cubin) qc_cubin_open(&c->k, index, c->grid, c->threads, &lost, &why);
  if (lost) {
    if (err && errlen > 0) snprintf(err, errlen, "the patched kernel's check broke the card's context (%s)", why);
    delete c;
    return NULL;
  }
  e = cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking);
  if (e != cudaSuccess) {
    set_err(err, errlen, "cudaStreamCreate", e);
    c->stream = NULL;
    free_core(c);
    return NULL;
  }
  e = cudaMalloc((void **)&c->d_nhits, sizeof(unsigned long long));
  if (e != cudaSuccess || grow(c, 256, err, errlen)) {
    if (e != cudaSuccess) set_err(err, errlen, "cudaMalloc", e);
    free_core(c);
    return NULL;
  }
  if (c->k.fn && g && atoi(g) > 0) c->k.grid = c->grid;
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
  if (left < 1 || at > (1ull << 32) || left > (1ull << 32) - at) {
    if (err && errlen > 0) snprintf(err, errlen, "start + count passes 2^32");
    return -1;
  }
  for (int i = 0; i < QC_LINE; i++) {
    if (w->line[i] >= HP) {
      if (err && errlen > 0) snprintf(err, errlen, "line element %d is not below the field prime", i);
      return -1;
    }
  }
  while (left) {
    u64 n = left < QC_CHUNK ? left : QC_CHUNK;
    launch_at(w, at, &L);
    L.count = (u32)n;
    u64 blocks = (n + c->threads - 1) / c->threads;
    if (c->k.fn) {
      int grid = blocks < (u64)c->k.grid ? (int)blocks : c->k.grid;
      void *args[] = {&L, &c->d_nhits, &c->d_hits};
      CUresult r = c->k.launch(c->k.fn, grid, 1, 1, c->threads, 1, 1, 0, (CUstream)c->stream, args, nullptr);
      if (r != CUDA_SUCCESS) {
        if (err && errlen > 0) snprintf(err, errlen, "kernel launch (patched kernel): CUresult %d", (int)r);
        return -1;
      }
    } else {
      int grid = blocks < (u64)c->grid ? (int)blocks : c->grid;
      qc_scan_kernel<<<grid, c->threads, 0, c->stream>>>(L, c->d_nhits, c->d_hits);
      e = cudaGetLastError();
      if (e != cudaSuccess) {
        set_err(err, errlen, "kernel launch", e);
        return -1;
      }
    }
    at += n;
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

// What a core made on its card, and the core. The caller holds g_open_mu.
static void free_core(qc_core *c) {
  {
    std::lock_guard<std::mutex> lock(c->mu);
    if (cudaSetDevice(c->index) == cudaSuccess) {
      if (c->stream) cudaStreamSynchronize(c->stream);
      if (c->d_hits) cudaFree(c->d_hits);
      if (c->d_nhits) cudaFree(c->d_nhits);
      if (c->stream) cudaStreamDestroy(c->stream);
      if (c->k.mod && c->k.unload) c->k.unload(c->k.mod);
    }
    cudaGetLastError();
  }
  delete c;
}

extern "C" void qc_close(qc_core *c) {
  if (!c) return;
  std::lock_guard<std::mutex> open_lock(g_open_mu);
  free_core(c);
}
