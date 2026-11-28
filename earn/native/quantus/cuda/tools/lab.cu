// lab: timing probes for the sm_120 arithmetic redesign (round 5). Each probe runs a loop body on independent
// chains at a chosen launch shape and prints SMSP-cycles a warp-op: the cycles one of an SM's four schedulers
// spends on one op of one warp (SM-cycles a thread-op times 128). A probe also prints a checksum of its outputs, so
// two forms of the same arithmetic can be compared.
//
//   lab <threads a block> <blocks an SM> <iters> [name prefix]
//
// Synthetic probes (group "s") mix IMAD.WIDE with cheap two-input adds on independent chains, as ubench does, to see
// how much ALU work hides under a WIDE at this launch shape. Field probes (group "f") run qc_field.cuh's routines.
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <map>

#include "../qc_kernel.cuh"
#include "../qc_lab.cuh"

struct Rec {
  unsigned sm;
  long long t0, t1;
};

__device__ __forceinline__ unsigned lab_smid() {
  unsigned r;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
  return r;
}

#define CH 8
#define LAB(name, decl, body, fin)                                                                   \
  __global__ void name(int iters, u64 *out, Rec *rec, u64 seed, u32 z) {                          \
    u64 s[12];                                                                                    \
    u32 x[CH], y[CH];                                                                             \
    _Pragma("unroll") for (int i = 0; i < 12; i++)                                                \
      s[i] = (seed ^ (0x9e3779b97f4a7c15ull * (threadIdx.x + blockIdx.x * 977 + i * 31))) >> 1;   \
    _Pragma("unroll") for (int k = 0; k < CH; k++) {                                              \
      x[k] = (u32)s[k] + z;                                                                       \
      y[k] = (u32)(s[k] >> 32) | 1;                                                               \
    }                                                                                             \
    decl;                                                                                         \
    __syncthreads();                                                                              \
    long long t0 = clock64();                                                                     \
    _Pragma("unroll 1") for (int it = 0; it < iters; it++) { body; }                              \
    __syncthreads();                                                                              \
    long long t1 = clock64();                                                                     \
    fin;                                                                                          \
    u64 acc = 0;                                                                                  \
    _Pragma("unroll") for (int i = 0; i < 12; i++) acc ^= qc_canon(s[i]) * (2 * i + 3);           \
    _Pragma("unroll") for (int k = 0; k < CH; k++) acc += x[k] ^ ((u64)y[k] << 32);               \
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;                                             \
    if (threadIdx.x == 0) {                                                                       \
      rec[blockIdx.x].sm = lab_smid();                                                            \
      rec[blockIdx.x].t0 = t0;                                                                    \
      rec[blockIdx.x].t1 = t1;                                                                    \
    }                                                                                             \
  }

// (x, y) = x * y as one IMAD.WIDE with no addend; (x, y) = x * y + (x, y) with one.
#define SW0(k) asm volatile("{.reg .u64 t; mul.wide.u32 t, %0, %1; mov.b64 {%0, %1}, t;}" : "+r"(x[k]), "+r"(y[k]))
#define SWA(k) asm volatile("{.reg .u64 t, c; mov.b64 c, {%0, %1}; mad.wide.u32 t, %0, %1, c; mov.b64 {%0, %1}, t;}" : "+r"(x[k]), "+r"(y[k]))
// Two cheap adds: a 64-bit add as two two-input IADD3s with a carry between them, on registers of another chain.
#define SA2(d, e) asm volatile("add.cc.u32 %0, %0, %2; addc.u32 %1, %1, %3;" : "+r"(p[d]), "+r"(q[d]) : "r"(p[e]), "r"(q[e]))
#define SDECL u32 p[CH], q[CH]; _Pragma("unroll") for (int k = 0; k < CH; k++) { p[k] = x[k] * 3 + 1; q[k] = y[k] ^ 0x55; }
#define SEND _Pragma("unroll") for (int k = 0; k < CH; k++) { x[k] ^= p[k]; y[k] += q[k]; }
#define SLOOP _Pragma("unroll") for (int k = 0; k < CH; k++)

LAB(s_w0, , SLOOP { SW0(k); }, )
LAB(s_wa, , SLOOP { SWA(k); }, )
LAB(s_a2, SDECL, { SLOOP { SA2(k, (k + 3) % CH); } }, SEND)
LAB(s_w0_2a, SDECL, { SLOOP { SW0(k); SA2(k, (k + 3) % CH); } }, SEND)
LAB(s_w0_4a, SDECL, { SLOOP { SW0(k); SA2(k, (k + 3) % CH); SA2((k + 1) % CH, (k + 5) % CH); } }, SEND)
LAB(s_w0_6a, SDECL, { SLOOP { SW0(k); SA2(k, (k + 3) % CH); SA2((k + 1) % CH, (k + 5) % CH); SA2((k + 2) % CH, (k + 7) % CH); } }, SEND)
LAB(s_wa_2a, SDECL, { SLOOP { SWA(k); SA2(k, (k + 3) % CH); } }, SEND)
LAB(s_wa_4a, SDECL, { SLOOP { SWA(k); SA2(k, (k + 3) % CH); SA2((k + 1) % CH, (k + 5) % CH); } }, SEND)

#define FLOOP _Pragma("unroll") for (int i = 0; i < 12; i++)
LAB(f_mul, , FLOOP { s[i] = qc_mul(s[i], s[(i + 5) % 12]); }, )
LAB(f_sqr, , FLOOP { s[i] = qc_sqr(s[i]); }, )
LAB(f_sbox, , FLOOP { s[i] = qc_sbox<QC_SQRF>(s[i]); }, )
LAB_EXTRA

typedef void (*Kern)(int, u64 *, Rec *, u64, u32);
struct Probe {
  const char *name;
  Kern k;
  double per;
};

int main(int argc, char **argv) {
  int threads = argc > 1 ? atoi(argv[1]) : 768;
  int bps = argc > 2 ? atoi(argv[2]) : 1;
  int iters = argc > 3 ? atoi(argv[3]) : 400;
  const char *pre = argc > 4 ? argv[4] : "";
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  int blocks = p.multiProcessorCount * bps;
  printf("lab %s sm_%d%d, %d SMs, %d blocks x %d threads (%d warps a scheduler), %d iters\n", p.name, p.major,
         p.minor, p.multiProcessorCount, blocks, threads, threads * bps / 128, iters);
  u64 *out;
  Rec *rec;
  cudaMalloc(&out, (size_t)blocks * threads * 8);
  cudaMalloc(&rec, blocks * sizeof(Rec));
  std::vector<Rec> h(blocks);
  std::vector<u64> ho((size_t)blocks * threads);
  Probe probes[] = {{"s_w0", s_w0, CH},       {"s_wa", s_wa, CH},       {"s_a2", s_a2, CH},
                    {"s_w0_2a", s_w0_2a, CH}, {"s_w0_4a", s_w0_4a, CH}, {"s_w0_6a", s_w0_6a, CH},
                    {"s_wa_2a", s_wa_2a, CH}, {"s_wa_4a", s_wa_4a, CH}, {"f_mul", f_mul, 12},
                    {"f_sqr", f_sqr, 12},     {"f_sbox", f_sbox, 12}    LAB_EXTRA_LIST};
  for (auto &t : probes) {
    if (strncmp(t.name, pre, strlen(pre))) continue;
    t.k<<<blocks, threads>>>(2, out, rec, 7, 0);
    cudaDeviceSynchronize();
    t.k<<<blocks, threads>>>(iters, out, rec, 7, 0);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
      printf("%s: %s\n", t.name, cudaGetErrorString(e));
      return 1;
    }
    cudaMemcpy(h.data(), rec, blocks * sizeof(Rec), cudaMemcpyDeviceToHost);
    cudaMemcpy(ho.data(), out, ho.size() * 8, cudaMemcpyDeviceToHost);
    u64 sum = 0;
    for (auto v : ho) sum = sum * 0x100000001b3ull + v;
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
      double ops = (double)nb[s.first] * threads * (double)iters * t.per;
      cyc += (double)(s.second.second - s.second.first) / ops;
    }
    cyc /= span.size();
    printf("%-12s %8.2f SMSP-cycles a warp-op  sum %016llx  (%d SMs, %.2f blocks an SM)\n", t.name, cyc * 128,
           (unsigned long long)sum, (int)span.size(), (double)blocks / span.size());
  }
  return 0;
}
