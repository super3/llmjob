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

// Synthetic probes, ubench style: 8 chains a thread, 8 copies a loop trip (64 ops a trip, so the loop's own
// branch is small). w = hi(w) * y is an IMAD.WIDE with no addend and no MOV; WA adds w as a 64-bit addend.
#define SDECL                                                                 \
  u64 w[CH];                                                                  \
  u32 p[CH], q[CH];                                                           \
  _Pragma("unroll") for (int k = 0; k < CH; k++) {                            \
    w[k] = s[k] | 1;                                                          \
    p[k] = x[k] * 3 + 1;                                                      \
    q[k] = y[k] ^ 0x55;                                                       \
  }
#define SEND _Pragma("unroll") for (int k = 0; k < CH; k++) { x[k] ^= p[k] ^ (u32)w[k]; y[k] += q[k] ^ (u32)(w[k] >> 32); }
#define SLOOP _Pragma("unroll") for (int u = 0; u < 8; u++) _Pragma("unroll") for (int k = 0; k < CH; k++)
#define SW0(k) asm volatile("mul.wide.u32 %0, %1, %2;" : "=l"(w[k]) : "r"((u32)(w[k] >> 32)), "r"(y[k]))
#define SWA(k) asm volatile("mad.wide.u32 %0, %1, %2, %0;" : "+l"(w[k]) : "r"((u32)(w[k] >> 32)), "r"(y[k]))
// A plain two-register add, a three-register add, a carry pair (two two-register adds with a carry between), a MOV.
#define SP(d, e) asm volatile("add.u32 %0, %0, %1;" : "+r"(p[(d) % CH]) : "r"(q[(e) % CH]))
#define SQ(d, e) asm volatile("add.u32 %0, %0, %1;" : "+r"(q[(d) % CH]) : "r"(p[(e) % CH]))
#define S3(d, e, f) asm volatile("{.reg .u32 t; add.u32 t, %0, %1; add.u32 %0, t, %2;}" : "+r"(p[(d) % CH]) : "r"(q[(e) % CH]), "r"(y[(f) % CH]))
#define SC(d, e) asm volatile("add.cc.u32 %0, %0, %2; addc.u32 %1, %1, %3;" : "+r"(p[(d) % CH]), "+r"(q[(d) % CH]) : "r"(q[(e) % CH]), "r"(p[(e) % CH]))

LAB(s_w0, SDECL, SLOOP { SW0(k); }, SEND)
LAB(s_wa, SDECL, SLOOP { SWA(k); }, SEND)
LAB(s_p1, SDECL, SLOOP { SP(k, k + 3); }, SEND)
LAB(s_c1, SDECL, SLOOP { SC(k, k + 3); }, SEND)
LAB(s_31, SDECL, SLOOP { S3(k, k + 3, k + 1); }, SEND)
LAB(s_w0_p1, SDECL, SLOOP { SW0(k); SP(k, k + 3); }, SEND)
LAB(s_w0_p2, SDECL, SLOOP { SW0(k); SP(k, k + 3); SQ(k + 1, k + 5); }, SEND)
LAB(s_w0_p3, SDECL, SLOOP { SW0(k); SP(k, k + 3); SQ(k + 1, k + 5); SP(k + 2, k + 6); }, SEND)
LAB(s_w0_p4, SDECL, SLOOP { SW0(k); SP(k, k + 3); SQ(k + 1, k + 5); SP(k + 2, k + 6); SQ(k + 3, k + 7); }, SEND)
LAB(s_w0_p6, SDECL, SLOOP { SW0(k); SP(k, k + 3); SQ(k + 1, k + 5); SP(k + 2, k + 6); SQ(k + 3, k + 7); SP(k + 4, k + 1); SQ(k + 5, k + 2); }, SEND)
LAB(s_w0_c1, SDECL, SLOOP { SW0(k); SC(k, k + 3); }, SEND)
LAB(s_w0_c2, SDECL, SLOOP { SW0(k); SC(k, k + 3); SC(k + 1, k + 5); }, SEND)
LAB(s_w0_c3, SDECL, SLOOP { SW0(k); SC(k, k + 3); SC(k + 1, k + 5); SC(k + 2, k + 6); }, SEND)
LAB(s_w0_31, SDECL, SLOOP { SW0(k); S3(k, k + 3, k + 1); }, SEND)
LAB(s_w0_32, SDECL, SLOOP { SW0(k); S3(k, k + 3, k + 1); S3(k + 1, k + 5, k + 2); }, SEND)
LAB(s_wa_p2, SDECL, SLOOP { SWA(k); SP(k, k + 3); SQ(k + 1, k + 5); }, SEND)
LAB(s_wa_p4, SDECL, SLOOP { SWA(k); SP(k, k + 3); SQ(k + 1, k + 5); SP(k + 2, k + 6); SQ(k + 3, k + 7); }, SEND)
LAB(s_wa_c2, SDECL, SLOOP { SWA(k); SC(k, k + 3); SC(k + 1, k + 5); }, SEND)

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
  Probe probes[] = {{"s_w0", s_w0, 64}, {"s_wa", s_wa, 64}, {"s_p1", s_p1, 64}, {"s_c1", s_c1, 64}, {"s_31", s_31, 64}, {"s_w0_p1", s_w0_p1, 64}, {"s_w0_p2", s_w0_p2, 64}, {"s_w0_p3", s_w0_p3, 64}, {"s_w0_p4", s_w0_p4, 64}, {"s_w0_p6", s_w0_p6, 64}, {"s_w0_c1", s_w0_c1, 64}, {"s_w0_c2", s_w0_c2, 64}, {"s_w0_c3", s_w0_c3, 64}, {"s_w0_31", s_w0_31, 64}, {"s_w0_32", s_w0_32, 64}, {"s_wa_p2", s_wa_p2, 64}, {"s_wa_p4", s_wa_p4, 64}, {"s_wa_c2", s_wa_c2, 64},
                    {"f_mul", f_mul, 12}, {"f_sqr", f_sqr, 12}, {"f_sbox", f_sbox, 12} LAB_EXTRA_LIST};
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
