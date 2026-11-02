// HIP smoke test for a Salad AMD box (amdtest.sh's "hip" task): does HIP compute work on this GPU?
// Prints the runtime and driver versions, every device HIP sees, the HIP and HSA libraries the process
// actually loaded (from /proc/self/maps, so Salad's WSL2 bridge shows if it is the one in use), then runs
// three kernels on device 0 and checks them on the CPU: an integer multiply-add over 16M words, a
// wave-wide XOR shuffle (the folds' readout uses __shfl_xor), and a timed copy, so a figure far below the
// card's memory bandwidth would show. Ends with PASS, or FAIL and the step that failed. Exit 0 on PASS.
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// HIP-CPU (used to check this file without a GPU) has no gcnArchName.
#if defined(__HIP_CPU_RT__)
#define ARCH_NAME(p) "cpu"
#else
#define ARCH_NAME(p) (p).gcnArchName
#endif

#define TRY(x)                                                                    \
  do {                                                                            \
    hipError_t e_ = (x);                                                          \
    if (e_ != hipSuccess) {                                                       \
      printf("FAIL %s: %s (%d)\n", #x, hipGetErrorString(e_), (int)e_);           \
      return 1;                                                                   \
    }                                                                             \
  } while (0)

__global__ void madd(const unsigned *a, const unsigned *b, unsigned *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] * b[i] + (unsigned)i;
}

// Each lane XORs its value with the lanes at distance 1, 8 and 16, as the folds' readout does.
__global__ void shuffle(const unsigned *in, unsigned *out, int *wave) {
  unsigned v = in[threadIdx.x];
  v ^= __shfl_xor(v, 1);
  v ^= __shfl_xor(v, 8);
  v ^= __shfl_xor(v, 16);
  out[threadIdx.x] = v;
  if (threadIdx.x == 0) *wave = warpSize;
}

__global__ void copy(const int4 *in, int4 *out, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    out[i] = in[i];
}

static void loaded_libraries() {
  FILE *f = fopen("/proc/self/maps", "r");
  if (!f) return;
  char line[1024];
  std::vector<std::string> seen;
  while (fgets(line, sizeof line, f)) {
    const char *p = strchr(line, '/');
    if (!p) continue;
    std::string path(p);
    while (!path.empty() && (path.back() == '\n' || path.back() == ' ')) path.pop_back();
    if (path.find("amdhip") == std::string::npos && path.find("hsa-runtime") == std::string::npos &&
        path.find("dxg") == std::string::npos && path.find("dxcore") == std::string::npos &&
        path.find("rocm-wsl") == std::string::npos && path.find("amdgpu") == std::string::npos)
      continue;
    bool dup = false;
    for (auto &s : seen) dup |= s == path;
    if (!dup) {
      seen.push_back(path);
      printf("loaded %s\n", path.c_str());
    }
  }
  fclose(f);
}

int main() {
  int rv = 0, dv = 0;
  hipError_t ie = hipInit(0);
  printf("hipInit: %s\n", hipGetErrorString(ie));
  hipRuntimeGetVersion(&rv);
  hipDriverGetVersion(&dv);
  printf("runtime %d, driver %d\n", rv, dv);
  int n = 0;
  hipError_t ce = hipGetDeviceCount(&n);
  printf("devices %d (%s)\n", n, hipGetErrorString(ce));
  loaded_libraries();
  if (ce != hipSuccess || n == 0) {
    printf("FAIL no HIP device\n");
    return 1;
  }
  for (int d = 0; d < n; d++) {
    hipDeviceProp_t p;
    if (hipGetDeviceProperties(&p, d) != hipSuccess) {
      printf("device %d: no properties\n", d);
      continue;
    }
    printf("device %d: %s, %s, %d CUs, %d MHz, %.1f GiB, wave %d, pci %02x:%02x\n", d, p.name, ARCH_NAME(p),
           p.multiProcessorCount, p.clockRate / 1000, p.totalGlobalMem / 1073741824.0, p.warpSize, p.pciBusID,
           p.pciDeviceID);
  }
  TRY(hipSetDevice(0));

  // 1. Multiply-add over 16M words.
  const int N = 1 << 24;
  std::vector<unsigned> a(N), b(N), c(N);
  for (int i = 0; i < N; i++) {
    a[i] = 0x9E3779B9u * (unsigned)i + 3u;
    b[i] = (unsigned)(i % 1000) - 500u;
  }
  unsigned *da, *db, *dc;
  TRY(hipMalloc(&da, N * 4));
  TRY(hipMalloc(&db, N * 4));
  TRY(hipMalloc(&dc, N * 4));
  TRY(hipMemcpy(da, a.data(), N * 4, hipMemcpyHostToDevice));
  TRY(hipMemcpy(db, b.data(), N * 4, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(madd, dim3(N / 256), dim3(256), 0, 0, da, db, dc, N);
  TRY(hipGetLastError());
  TRY(hipDeviceSynchronize());
  TRY(hipMemcpy(c.data(), dc, N * 4, hipMemcpyDeviceToHost));
  long bad = 0;
  for (int i = 0; i < N; i++) bad += c[i] != a[i] * b[i] + (unsigned)i;
  printf("madd: %ld of %d wrong\n", bad, N);
  if (bad) {
    printf("FAIL madd\n");
    return 1;
  }

  // 2. XOR shuffle over 64 threads (two waves of 32 or one of 64; distances 1, 8 and 16 stay inside 32 lanes, so
  // the answer is the same either way), checked lane by lane.
  const int W = 64;
  std::vector<unsigned> in(W), out(W);
  for (int i = 0; i < W; i++) in[i] = 0x9E3779B9u * (i + 1);
  unsigned *din, *dout;
  int *dw, wave = 0;
  TRY(hipMalloc(&din, W * 4));
  TRY(hipMalloc(&dout, W * 4));
  TRY(hipMalloc(&dw, 4));
  TRY(hipMemcpy(din, in.data(), W * 4, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(shuffle, dim3(1), dim3(W), 0, 0, din, dout, dw);
  TRY(hipGetLastError());
  TRY(hipDeviceSynchronize());
  TRY(hipMemcpy(out.data(), dout, W * 4, hipMemcpyDeviceToHost));
  TRY(hipMemcpy(&wave, dw, 4, hipMemcpyDeviceToHost));
  int sbad = 0;
  for (int i = 0; i < W; i++) {
    unsigned want = 0;
    for (int m = 0; m < 8; m++) {
      int j = i ^ ((m & 1 ? 1 : 0) | (m & 2 ? 8 : 0) | (m & 4 ? 16 : 0));
      want ^= in[j];
    }
    sbad += out[i] != want;
  }
  printf("shuffle: wave %d, %d of %d lanes wrong\n", wave, sbad, W);
  if (sbad) {
    printf("FAIL shuffle\n");
    return 1;
  }

  // 3. A timed 512 MiB device-to-device copy, 10 times; the rate counts the read and the write.
  const size_t bytes = 1ull << 29, n4 = bytes / 16;
  int4 *s, *t;
  TRY(hipMalloc(&s, bytes));
  TRY(hipMalloc(&t, bytes));
  TRY(hipMemset(s, 1, bytes));
  hipLaunchKernelGGL(copy, dim3(4096), dim3(256), 0, 0, s, t, n4);
  TRY(hipDeviceSynchronize());
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < 10; r++) hipLaunchKernelGGL(copy, dim3(4096), dim3(256), 0, 0, s, t, n4);
  TRY(hipGetLastError());
  TRY(hipDeviceSynchronize());
  double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  printf("copy: %.0f GB/s (read + write)\n", 10.0 * 2 * bytes / secs / 1e9);
  hipFree(s); hipFree(t); hipFree(da); hipFree(db); hipFree(dc); hipFree(din); hipFree(dout); hipFree(dw);
  printf("PASS\n");
  return 0;
}
