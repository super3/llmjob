// The patched RTX 30, RTX 40 and RTX 50 kernels, on the host. Included by qc_host.cu.
//
// Since 28 Nov 2026 the RTX 20 (sm_75) and RTX 30 (sm_86) images are the compact kernel with a hand-written body: every
// step of both permutations after the first round as SASS from sm75/full75.py (QC_HAND75 2, qc_hand_loopall), 15% faster
// on an RTX 2060 SUPER than the compiled kernel and 6 to 7% faster on an RTX 3070 than the QC_RED 3 kernel described
// below, which RTX 40 and RTX 50 still use. They are built, listed, embedded and checked on the card the same way.
// RTX 50's CUDA 13 image is a hand-written body too since that day (sm120/full120.py: 502 MH/s on an RTX 5070 where
// the QC_RED 4 kernel ran 421); its 12.8 image is still the QC_RED 3 kernel.
//
// sm86/build.py makes qc_sm86_cubin.h: the scan kernel for sm_86 built with QC_RED 3 and its reduction tails made
// right by sm86/carryfix.py (the file says why that is exact), 2% faster on an RTX 3070 than the kernel as compiled;
// with --arch sm_89 it makes qc_sm89_cubin.h, the same for RTX 40 (QC_RED 3 and QC_KZ 1), and with --arch sm_120
// qc_sm120_cubin.h for RTX 50 (QC_RED 3 with CUDA 12.8, the hand-written body with CUDA 13; each RTX 50 core embeds
// the one its own nvcc built). build.sh / build.ps1 embed
// them when they are given their folders (QC_SM86_CUBIN, QC_SM89_CUBIN, QC_SM120_CUBIN are then defined). On a compute 8.6 (8.9, 12.0)
// card the core loads its one through the driver (cuModuleLoadData, cuLaunchKernel) on the runtime's own context.
// The check runs it beside the compiled kernel on 2^20 nonces of a fixed line, and the core uses it only when the two
// find the same hits (up to 4 nonces apart: QC_RED 2, which the compiled kernel uses, is 2^32 off in a case that
// comes up about once in 2^33 reductions, and the patched tail is right there). On any failure it keeps the
// compiled kernel.
//
// A cubin that faults in the check (an illegal instruction: a 12.8-built hand kernel on an RTX 5060 with driver
// 621.40, 28 Nov) leaves an error that no call clears, and nothing gets the card back in that process: after
// cudaDeviceReset every call answers "busy or unavailable" (RTX 4090 on Windows, driver 610.88; RTX 3060 on Linux,
// driver 617.42), and on Windows a fault in a second context (cuCtxCreate) poisons the runtime's context too. So the
// app runs the check first in a process of its own (cubinCheck, from src/main/quantusCubinCheck.js) and opens the
// core with the cubin only when that passed (qc_open2's cubin argument). The check in the core's own process stays,
// as before; should it still fault there, qc_open fails with a clear error instead of handing back a dead core.
// tools/qc_fault.cu builds a cubin that faults, for testing all this.
//
// QUANTUS_CUDA_CUBIN=0 keeps the compiled kernel; QUANTUS_CUDA_CUBIN_FILE=<file> loads that cubin instead of the
// embedded one, whatever its hash, on any card (for trying a new build before its hash is listed);
// QUANTUS_CUDA_LOG=1 says on stderr which kernel each card runs.
#include <cuda.h>
#include <iterator>
#include <string>
#include <vector>
#include <algorithm>
#if defined(QC_SM75_CUBIN)
#include "qc_sm75_cubin.h"
#endif
#if defined(QC_SM86_CUBIN)
#include "qc_sm86_cubin.h"
#endif
#if defined(QC_SM89_CUBIN)
#include "qc_sm89_cubin.h"
#endif
#if defined(QC_SM120_CUBIN)
#include "qc_sm120_cubin.h"
#endif

typedef CUresult(CUDAAPI *QcLoadData)(CUmodule *, const void *);
typedef CUresult(CUDAAPI *QcUnload)(CUmodule);
typedef CUresult(CUDAAPI *QcGetFunction)(CUfunction *, CUmodule, const char *);
typedef CUresult(CUDAAPI *QcLaunchK)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                     unsigned, CUstream, void **, void **);
typedef CUresult(CUDAAPI *QcOccupancy)(int *, CUfunction, int, size_t);

struct QcCubin {
  CUmodule mod = nullptr;
  CUfunction fn = nullptr;
  QcLaunchK launch = nullptr;
  QcUnload unload = nullptr;
  int grid = 0;
};

static bool qc_log() {
  const char *v = getenv("QUANTUS_CUDA_LOG");
  return v && *v && strcmp(v, "0") != 0;
}

template <class F>
static bool qc_sym(const char *name, F *out) {
  void *p = nullptr;
  cudaDriverEntryPointQueryResult q;
  if (cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q) != cudaSuccess
      || q != cudaDriverEntryPointSuccess || !p) {
    cudaGetLastError();
    return false;
  }
  *out = reinterpret_cast<F>(p);
  return true;
}

// The kernel's name inside a cubin: the one ".text._Z14qc_scan_kernel..." section (its mangling differs between
// Linux and Windows).
static bool qc_cubin_kernel(const std::vector<char> &img, std::string *name) {
  static const char kPre[] = ".text._Z14qc_scan_kernel";
  auto it = std::search(img.begin(), img.end(), kPre, kPre + sizeof kPre - 1);
  if (it == img.end()) return false;
  auto end = std::find(it, img.end(), '\0');
  if (end == img.end()) return false;
  *name = std::string(it + 6, end);  // without ".text."
  return true;
}

static bool qc_read_file(const char *path, std::vector<char> *out) {
  FILE *f = fopen(path, "rb");
  if (!f) return false;
  char buf[65536];
  size_t n;
  while ((n = fread(buf, 1, sizeof buf, f)) > 0) out->insert(out->end(), buf, buf + n);
  fclose(f);
  return !out->empty();
}

// One launch of either kernel over L (count nonces), its hits sorted.
static bool qc_cubin_hits(QcCubin *k, cudaStream_t stream, QcLaunch L, unsigned long long *d_n, u64 *d_h, int grid,
                          int threads, std::vector<u64> *hits) {
  if (cudaMemsetAsync(d_n, 0, sizeof(unsigned long long), stream) != cudaSuccess) return false;
  u64 blocks = ((u64)L.count + threads - 1) / threads;
  int g = blocks < (u64)grid ? (int)blocks : grid;
  if (k) {
    void *args[] = {&L, &d_n, &d_h};
    if (k->launch(k->fn, g, 1, 1, threads, 1, 1, 0, (CUstream)stream, args, nullptr) != CUDA_SUCCESS) return false;
  } else {
    qc_scan_kernel<<<g, threads, 0, stream>>>(L, d_n, d_h);
    if (cudaGetLastError() != cudaSuccess) return false;
  }
  unsigned long long n = 0;
  if (cudaMemcpyAsync(&n, d_n, sizeof n, cudaMemcpyDeviceToHost, stream) != cudaSuccess) return false;
  if (cudaStreamSynchronize(stream) != cudaSuccess) return false;
  if (n > L.cap) return false;
  hits->resize((size_t)n);
  if (n && (cudaMemcpyAsync(hits->data(), d_h, n * sizeof(u64), cudaMemcpyDeviceToHost, stream) != cudaSuccess
            || cudaStreamSynchronize(stream) != cudaSuccess))
    return false;
  std::sort(hits->begin(), hits->end());
  return true;
}

// The startup check: both kernels over 2^20 nonces at a target one nonce in 16 meets, on a fixed line.
// The hit lists may differ by at most 4 nonces (see the top of this file); a wrong patch differs in thousands.
static bool qc_cubin_check(QcCubin *k, cudaStream_t stream, int grid, int threads, const char **why) {
  const u32 count = 1u << 20, cap = 1u << 17;
  unsigned long long *d_n = nullptr;
  u64 *d_h = nullptr;
  bool ok = false;
  std::vector<u64> a, b, diff;
  QcLaunch L;
  memset(&L, 0, sizeof L);
  u64 x = 0x9e3779b97f4a7c15ull;
  u64 line[QC_LINE];
  for (int i = 0; i < QC_LINE; i++) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    line[i] = x % QC_P;
  }
  for (int i = 0; i < 12; i++) L.c[i] = line[i];
  L.x0 = line[12];
  L.x4 = line[13];
  L.target = 1ull << 60;
  L.base = 0x12345678u;
  L.count = count;
  L.cap = cap;
  *why = "could not allocate the check's buffers";
  if (cudaMalloc((void **)&d_n, sizeof(unsigned long long)) != cudaSuccess) goto done;
  if (cudaMalloc((void **)&d_h, cap * sizeof(u64)) != cudaSuccess) goto done;
  *why = "the compiled kernel's check run failed";
  if (!qc_cubin_hits(nullptr, stream, L, d_n, d_h, grid, threads, &a)) goto done;
  *why = "the cubin's check run failed";
  if (!qc_cubin_hits(k, stream, L, d_n, d_h, k->grid, threads, &b)) goto done;
  std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(diff));
  *why = "its hits differ from the compiled kernel's";
  ok = a.size() > count / 32 && diff.size() <= 4;
done:
  if (d_h) cudaFree(d_h);
  if (d_n) cudaFree(d_n);
  cudaGetLastError();
  return ok;
}

// The cubin card `index` would try, nul-terminated in *img, with a name for the log in *what; *img is left empty
// when it has none (QUANTUS_CUDA_CUBIN=0, no image for its compute capability, or a QUANTUS_CUDA_CUBIN_FILE that
// cannot be read).
// Makes no context on the card.
static void qc_cubin_image(int index, std::vector<char> *out, const char **what, int *mj, int *mn) {
  std::vector<char> &img = *out;
  const char *off = getenv("QUANTUS_CUDA_CUBIN");
  if (off && strcmp(off, "0") == 0) return;
  *mj = 0;
  *mn = 0;
  cudaDeviceGetAttribute(mj, cudaDevAttrComputeCapabilityMajor, index);
  cudaDeviceGetAttribute(mn, cudaDevAttrComputeCapabilityMinor, index);
  cudaGetLastError();
  const char *file = getenv("QUANTUS_CUDA_CUBIN_FILE");
  *what = "";
  if (file && *file) {
    if (!qc_read_file(file, &img)) {
      if (qc_log()) fprintf(stderr, "quantus_cuda: card %d: cannot read %s; the compiled kernel runs\n", index, file);
      img.clear();
      return;
    }
    *what = file;
  } else {
    const int major = *mj, minor = *mn;
    if (major != 7 && major != 8 && major != 12) return;
#if defined(QC_SM75_CUBIN)
    // RTX 20: the kernel with the hand-written body (sm86/build.py --arch sm_75, sm75/full75.py).
    if (major == 7 && minor == 5) {
      img.assign((const char *)qc_sm75_image, (const char *)qc_sm75_image + sizeof qc_sm75_image);
      *what = qc_sm75_hash;
    }
#endif
#if defined(QC_SM86_CUBIN)
    if (major == 8 && minor == 6) {
      img.assign((const char *)qc_sm86_image, (const char *)qc_sm86_image + sizeof qc_sm86_image);
      *what = qc_sm86_hash;
    }
#endif
#if defined(QC_SM120_CUBIN)
    if (major == 12 && minor == 0) {
      img.assign((const char *)qc_sm120_image, (const char *)qc_sm120_image + sizeof qc_sm120_image);
      *what = qc_sm120_hash;
    }
#endif
#if defined(QC_SM89_CUBIN)
    if (major == 8 && minor == 9) {
      img.assign((const char *)qc_sm89_image, (const char *)qc_sm89_image + sizeof qc_sm89_image);
      *what = qc_sm89_hash;
    }
#endif
    (void)minor;
    if (img.empty()) return;
  }
  img.push_back('\0');
}

// Loads the cubin for this card if there is one for it and it passes the check; true when the card runs it. On any
// failure leaves k empty (the compiled kernel runs) and says why in *why. threads: the compiled kernel's block size
// for this card; the cubin, built from the same source, runs it too. *lost is set when the card's context no longer
// works after a failed check (the cubin faulted): this process can no longer use the card. So call this before
// making anything else on the card.
static bool qc_cubin_open(QcCubin *k, int index, int grid, int threads, bool *lost, const char **why) {
  *lost = false;
  std::vector<char> img;
  const char *what = "";
  int major = 0, minor = 0;
  *why = "this card has none";
  qc_cubin_image(index, &img, &what, &major, &minor);
  if (img.empty()) return false;
  std::string name;
  QcLoadData load = nullptr;
  QcGetFunction getfn = nullptr;
  QcOccupancy occ = nullptr;
  cudaStream_t stream = nullptr;
  int per = 0, sms = 0;
  *why = "the driver lacks an entry point it needs";
  if (!qc_sym("cuModuleLoadData", &load) || !qc_sym("cuModuleGetFunction", &getfn) || !qc_sym("cuLaunchKernel", &k->launch)
      || !qc_sym("cuModuleUnload", &k->unload) || !qc_sym("cuOccupancyMaxActiveBlocksPerMultiprocessor", &occ))
    goto fail;
  *why = "no scan kernel in it";
  if (!qc_cubin_kernel(img, &name)) goto fail;
  *why = "the driver would not load it";
  if (load(&k->mod, img.data()) != CUDA_SUCCESS) goto fail;
  *why = "its kernel was not found";
  if (getfn(&k->fn, k->mod, name.c_str()) != CUDA_SUCCESS) goto fail;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, index);
  if (occ(&per, k->fn, threads, 0) != CUDA_SUCCESS || per < 1) per = 1;
  k->grid = sms * per;
  *why = "no stream for the check";
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) goto fail;
  if (!qc_cubin_check(k, stream, grid, threads, why)) goto fail;
  cudaStreamDestroy(stream);
  if (qc_log()) fprintf(stderr, "quantus_cuda: card %d (sm_%d%d) runs the patched kernel (%s)\n", index, major, minor, what);
  return true;
fail:
  // A cubin that faulted leaves an error on the context that every later call returns, cudaGetLastError included.
  // Then nothing made in it is touched again.
  *lost = cudaDeviceSynchronize() != cudaSuccess;
  if (qc_log()) fprintf(stderr, "quantus_cuda: card %d: the patched kernel is not used (%s)%s; the compiled kernel runs\n",
                        index, *why, *lost ? ", and it broke the card's context" : "");
  if (!*lost) {
    if (stream) cudaStreamDestroy(stream);
    if (k->mod && k->unload) k->unload(k->mod);
  }
  k->mod = nullptr;
  k->fn = nullptr;
  cudaGetLastError();
  return false;
}
