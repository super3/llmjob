// The patched RTX 30, RTX 40 and RTX 50 kernels, on the host. Included by qc_host.cu.
//
// sm86/build.py makes qc_sm86_cubin.h: the scan kernel for sm_86 built with QC_RED 3 and its reduction tails made
// right by sm86/carryfix.py (the file says why that is exact), 2% faster on an RTX 3070 than the kernel as compiled;
// with --arch sm_89 it makes qc_sm89_cubin.h, the same for RTX 40 (QC_RED 3 and QC_KZ 1), and with --arch sm_120
// qc_sm120_cubin.h for RTX 50 (QC_RED 3; each RTX 50 core embeds the one its own nvcc built). build.sh / build.ps1 embed
// them when they are given their folders (QC_SM86_CUBIN, QC_SM89_CUBIN, QC_SM120_CUBIN are then defined). On a compute 8.6 (8.9, 12.0)
// card the core loads its one through the driver (cuModuleLoadData, cuLaunchKernel, on the runtime's own context and
// stream), runs it beside the compiled kernel on 2^20 nonces of a fixed line, and uses it only when the two
// find the same hits (up to 4 nonces apart: QC_RED 2, which the compiled kernel uses, is 2^32 off in a case that
// comes up about once in 2^33 reductions, and the patched tail is right there). On any failure it keeps the
// compiled kernel. QUANTUS_CUDA_CUBIN=0 keeps the compiled kernel; QUANTUS_CUDA_CUBIN_FILE=<file> loads that cubin
// instead of the embedded one, whatever its hash, on any card (for trying a new build before its hash is listed);
// QUANTUS_CUDA_LOG=1 says on stderr which kernel each card runs.
#include <cuda.h>
#include <iterator>
#include <string>
#include <vector>
#include <algorithm>
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

// Loads the cubin for this card if there is one for it; on any failure leaves k empty (the compiled kernel runs).
// threads: the compiled kernel's block size for this card; the cubin, built from the same source, runs it too.
static void qc_cubin_open(QcCubin *k, int index, cudaStream_t stream, int grid, int threads) {
  const char *off = getenv("QUANTUS_CUDA_CUBIN");
  if (off && strcmp(off, "0") == 0) return;
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, index);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, index);
  std::vector<char> img;
  const char *file = getenv("QUANTUS_CUDA_CUBIN_FILE");
  const char *what = "";
  if (file && *file) {
    if (!qc_read_file(file, &img)) {
      if (qc_log()) fprintf(stderr, "quantus_cuda: card %d: cannot read %s; the compiled kernel runs\n", index, file);
      return;
    }
    what = file;
  } else {
    if (major != 8 && major != 12) return;
#if defined(QC_SM86_CUBIN)
    if (major == 8 && minor == 6) {
      img.assign((const char *)qc_sm86_image, (const char *)qc_sm86_image + sizeof qc_sm86_image);
      what = qc_sm86_hash;
    }
#endif
#if defined(QC_SM120_CUBIN)
    if (major == 12 && minor == 0) {
      img.assign((const char *)qc_sm120_image, (const char *)qc_sm120_image + sizeof qc_sm120_image);
      what = qc_sm120_hash;
    }
#endif
#if defined(QC_SM89_CUBIN)
    if (major == 8 && minor == 9) {
      img.assign((const char *)qc_sm89_image, (const char *)qc_sm89_image + sizeof qc_sm89_image);
      what = qc_sm89_hash;
    }
#endif
    if (img.empty()) return;
  }
  img.push_back('\0');
  std::string name;
  QcLoadData load = nullptr;
  QcGetFunction getfn = nullptr;
  QcOccupancy occ = nullptr;
  const char *why = "the driver lacks an entry point it needs";
  int per = 0, sms = 0;
  if (!qc_sym("cuModuleLoadData", &load) || !qc_sym("cuModuleGetFunction", &getfn) || !qc_sym("cuLaunchKernel", &k->launch)
      || !qc_sym("cuModuleUnload", &k->unload) || !qc_sym("cuOccupancyMaxActiveBlocksPerMultiprocessor", &occ))
    goto fail;
  why = "no scan kernel in it";
  if (!qc_cubin_kernel(img, &name)) goto fail;
  why = "the driver would not load it";
  if (load(&k->mod, img.data()) != CUDA_SUCCESS) goto fail;
  why = "its kernel was not found";
  if (getfn(&k->fn, k->mod, name.c_str()) != CUDA_SUCCESS) goto fail;
  if (occ(&per, k->fn, threads, 0) != CUDA_SUCCESS || per < 1) per = 1;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, index);
  k->grid = sms * per;
  if (!qc_cubin_check(k, stream, grid, threads, &why)) goto fail;
  if (qc_log()) fprintf(stderr, "quantus_cuda: card %d (sm_%d%d) runs the patched kernel (%s)\n", index, major, minor, what);
  return;
fail:
  if (qc_log()) fprintf(stderr, "quantus_cuda: card %d: the patched kernel is not used (%s); the compiled kernel runs\n",
                        index, why);
  if (k->mod && k->unload) k->unload(k->mod);
  k->mod = nullptr;
  k->fn = nullptr;
  cudaGetLastError();
}
