// partbench: SM-cycles a thread-op of each part of the Quantus kernel (qc_kernel.cuh), each part alone in a loop on
// 12 independent state words a thread, at the scan kernel's launch shape (QC_THREADS a block, QC_MINB blocks an SM).
// Compare with the part's SASS (nvdisasm) to see how close each part runs to its pipe bound.
//
//   partbench [iters]
//
// Parts: sbox (12 S-boxes), ext (the external layer), full (12 S-boxes and the external layer), partial (one partial
// round, as the scan kernel's partial loop runs it), perms (both permutations, the scan kernel's whole per-nonce work),
// full2 and partial2 (two nonces a thread, see PB_KERNEL2).
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <map>

#include "../qc_kernel.cuh"

struct Rec {
  unsigned sm;
  long long t0, t1;
};

__device__ __forceinline__ unsigned pb_smid() {
  unsigned r;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
  return r;
}

#define PB_KERNEL(name, body)                                                                     \
  __global__ void __launch_bounds__(QC_THREADS, QC_MINB)                                          \
  name(int iters, u64 *out, Rec *rec, u64 seed, u32 z) {                                          \
    u64 s[12];                                                                                    \
    _Pragma("unroll") for (int i = 0; i < 12; i++)                                                \
      s[i] = (seed ^ (0x9e3779b97f4a7c15ull * (threadIdx.x + blockIdx.x * 977 + i * 31))) >> 1;   \
    __syncthreads();                                                                              \
    if (QC_DESYNC && ((threadIdx.x >> 7) & 1)) {                                                  \
      _Pragma("unroll 1") for (int k = 0; k < QC_DESYNC; k++) s[k % 12] = qc_sbox(s[k % 12]);     \
    }                                                                                             \
    long long t0 = clock64();                                                                     \
    _Pragma("unroll 1") for (int it = 0; it < iters; it++) { body; }                              \
    __syncthreads();                                                                              \
    long long t1 = clock64();                                                                     \
    u64 acc = 0;                                                                                  \
    _Pragma("unroll") for (int i = 0; i < 12; i++) acc ^= s[i];                                   \
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;                                             \
    if (threadIdx.x == 0) {                                                                       \
      rec[blockIdx.x].sm = pb_smid();                                                             \
      rec[blockIdx.x].t0 = t0;                                                                    \
      rec[blockIdx.x].t1 = t1;                                                                    \
    }                                                                                             \
  }

PB_KERNEL(k_sbox, {
  _Pragma("unroll")
  for (int i = 0; i < 12; i++) s[i] = qc_sbox<QC_SQRF>(s[i]);
})
PB_KERNEL(k_ext, { qc_ext(s, QC_RC_F1[it & 3], z + 2); })
PB_KERNEL(k_full, { qc_full(s, QC_RC_F1[it & 3], z + 2); })
PB_KERNEL(k_partial, {
  u64 x = s[0];
  _Pragma("unroll 1")
  for (int r = 0; r < 21; r++) x = qc_sbox(qc_partial(s, x, QC_RC_P[r], z));
  s[0] = x;
})
PB_KERNEL(k_mul, {
  _Pragma("unroll")
  for (int i = 0; i < 12; i++) s[i] = qc_mul(s[i], s[(i + 5) % 12] | 1);
})
PB_KERNEL(k_sqr, {
  _Pragma("unroll")
  for (int i = 0; i < 12; i++) s[i] = qc_sqr(s[i]);
})
PB_KERNEL(k_perms, { s[0] ^= qc_two_perms(s, z); })

// Two nonces a thread. full2: nonce b half a round behind nonce a, so one nonce's S-boxes share a basic block with
// the other's external layer (S-boxes fill both pipes, the layer only the ALU). partial2: both nonces' partial rounds
// in one loop body. Each counts per nonce (a round of full2 is two nonce-rounds).
#define PB_KERNEL2(name, body)                                                                    \
  __global__ void __launch_bounds__(QC_THREADS, QC_MINB)                                          \
  name(int iters, u64 *out, Rec *rec, u64 seed, u32 z) {                                          \
    u64 s[12], b[12];                                                                             \
    _Pragma("unroll") for (int i = 0; i < 12; i++) {                                              \
      s[i] = (seed ^ (0x9e3779b97f4a7c15ull * (threadIdx.x + blockIdx.x * 977 + i * 31))) >> 1;   \
      b[i] = (seed ^ (0xc2b2ae3d27d4eb4full * (threadIdx.x + blockIdx.x * 991 + i * 37))) >> 1;   \
    }                                                                                             \
    __syncthreads();                                                                              \
    long long t0 = clock64();                                                                     \
    _Pragma("unroll 1") for (int it = 0; it < iters; it++) { body; }                              \
    __syncthreads();                                                                              \
    long long t1 = clock64();                                                                     \
    u64 acc = 0;                                                                                  \
    _Pragma("unroll") for (int i = 0; i < 12; i++) acc ^= s[i] ^ b[i];                            \
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;                                             \
    if (threadIdx.x == 0) {                                                                       \
      rec[blockIdx.x].sm = pb_smid();                                                             \
      rec[blockIdx.x].t0 = t0;                                                                    \
      rec[blockIdx.x].t1 = t1;                                                                    \
    }                                                                                             \
  }

PB_KERNEL2(k_full2, {
  _Pragma("unroll")
  for (int i = 0; i < 12; i++) s[i] = qc_sbox<QC_SQRF>(s[i]);
  qc_ext(b, QC_RC_F1[it & 3], z + 2);
  qc_ext(s, QC_RC_F1[(it + 1) & 3], z + 2);
  _Pragma("unroll")
  for (int i = 0; i < 12; i++) b[i] = qc_sbox<QC_SQRF>(b[i]);
})
PB_KERNEL2(k_partial2, {
  u64 x = s[0];
  u64 y = b[0];
  _Pragma("unroll 1")
  for (int r = 0; r < 21; r++) {
    x = qc_sbox(qc_partial(s, x, QC_RC_P[r], z));
    y = qc_sbox(qc_partial(b, y, QC_RC_P[r], z));
  }
  s[0] = x;
  b[0] = y;
})

typedef void (*Kern)(int, u64 *, Rec *, u64, u32);
struct Part {
  const char *name;
  Kern k;
  double per;  // thread-ops an iteration (S-boxes, layers, rounds, ...)
  const char *unit;
};

int main(int argc, char **argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 200;
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  int threads = QC_THREADS, blocks = p.multiProcessorCount * QC_MINB;
  printf("card %s sm_%d%d, %d SMs, %d blocks x %d threads, %d iters\n", p.name, p.major, p.minor,
         p.multiProcessorCount, blocks, threads, iters);
  u64 *out;
  Rec *rec;
  cudaMalloc(&out, (size_t)blocks * threads * 8);
  cudaMalloc(&rec, blocks * sizeof(Rec));
  std::vector<Rec> h(blocks);
  Part parts[] = {{"sbox", k_sbox, 12, "S-box"},          {"mul", k_mul, 12, "mul"},
                  {"sqr", k_sqr, 12, "square"},           {"ext", k_ext, 1, "layer"},
                  {"full", k_full, 1, "round"},           {"partial", k_partial, 21, "round"},
                  {"perms", k_perms, 1, "nonce"},         {"full2", k_full2, 2, "round"},
                  {"partial2", k_partial2, 42, "round"}};
  for (auto &t : parts) {
    int it = t.k == k_perms ? iters / 20 + 1 : ((t.k == k_partial || t.k == k_partial2) ? iters / 4 + 1 : iters);
    t.k<<<blocks, threads>>>(2, out, rec, 7, 0);
    cudaDeviceSynchronize();
    t.k<<<blocks, threads>>>(it, out, rec, 7, 0);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
      printf("%s: %s\n", t.name, cudaGetErrorString(e));
      return 1;
    }
    cudaMemcpy(h.data(), rec, blocks * sizeof(Rec), cudaMemcpyDeviceToHost);
    std::map<unsigned, std::pair<long long, long long>> span;
    std::map<unsigned, int> nb;
    for (auto &r : h) {
      auto f = span.find(r.sm);
      if (f == span.end()) span[r.sm] = {r.t0, r.t1};
      else {
        if (r.t0 < f->second.first) f->second.first = r.t0;
        if (r.t1 > f->second.second) f->second.second = r.t1;
      }
      nb[r.sm]++;
    }
    double cyc = 0;
    for (auto &s : span) {
      double ops = (double)nb[s.first] * threads * (double)it * t.per;
      cyc += (double)(s.second.second - s.second.first) / ops;
    }
    cyc /= span.size();
    printf("%-8s %8.3f SM-cycles a thread-%s\n", t.name, cyc, t.unit);
  }
  return 0;
}
