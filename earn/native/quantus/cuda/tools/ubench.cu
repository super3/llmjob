// ubench: integer issue rates per SM per clock on this card, for the ceiling
// estimate of the Quantus kernel. Each test runs 8 independent chains a
// thread of one instruction kind (or a mix), 1024 threads a block, enough
// blocks for 4 a SM, and times each SM with clock64 (so the SM clock does not
// matter). Prints warp-instructions per SM per cycle for each kind.
//
//   ubench [iters]
//
// Check the SASS (nvdisasm) once per architecture: each test's loop must hold
// the instruction it names and little else.
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <map>

#define CH 8
#define UNR 16

__device__ __forceinline__ unsigned smid() {
  unsigned r;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
  return r;
}

struct Rec {
  unsigned sm;
  long long t0, t1;
};

// One test body: x[k] = OP(x[k], ...) for k in 0..CH, UNR times a loop trip.
#define BENCH(name, decl, body)                                                         \
  __global__ void name(int iters, unsigned *out, Rec *rec, unsigned a, unsigned b) {   \
    unsigned x[CH], y[CH];                                                              \
    unsigned long long w[CH];                                                           \
    for (int k = 0; k < CH; k++) {                                                      \
      x[k] = threadIdx.x * (k + 3) + a;                                                 \
      y[k] = threadIdx.x ^ (k * 77 + b);                                                \
      w[k] = ((unsigned long long)y[k] << 32) | x[k];                                   \
    }                                                                                   \
    decl;                                                                               \
    __syncthreads();                                                                    \
    long long t0 = clock64();                                                           \
    for (int it = 0; it < iters; it++) {                                                \
      _Pragma("unroll") for (int u = 0; u < UNR; u++) {                                 \
        _Pragma("unroll") for (int k = 0; k < CH; k++) { body; }                        \
      }                                                                                 \
    }                                                                                   \
    __syncthreads();                                                                    \
    long long t1 = clock64();                                                           \
    unsigned acc = 0;                                                                   \
    for (int k = 0; k < CH; k++) acc ^= x[k] ^ y[k] ^ (unsigned)w[k] ^ (unsigned)(w[k] >> 32); \
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;                                   \
    if (threadIdx.x == 0) {                                                             \
      rec[blockIdx.x].sm = smid();                                                      \
      rec[blockIdx.x].t0 = t0;                                                          \
      rec[blockIdx.x].t1 = t1;                                                          \
    }                                                                                   \
  }

// IMAD: x = x * y + a
BENCH(k_imad, , asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(x[k]) : "r"(y[k]), "r"(a)))
// IMAD.WIDE.U32: (x, y) = x * a + (x, y)
#define WIDE(k, m) asm volatile("mul.wide.u32 %0, %1, %2;" : "=l"(w[k]) : "r"((unsigned)(w[k] >> 32)), "r"(m))
BENCH(k_imadwide, , WIDE(k, a))
// IMAD.HI.U32: x = hi(x * y) + a
BENCH(k_imadhi, , asm volatile("mad.hi.u32 %0, %0, %1, %2;" : "+r"(x[k]) : "r"(y[k]), "r"(a)))
// IADD3: x = x + y + a
BENCH(k_iadd3, , asm volatile("{.reg .u32 t; add.u32 t, %0, %1; add.u32 %0, t, %2;}" : "+r"(x[k]) : "r"(y[k]), "r"(a)))
// LOP3: x = (x & y) ^ a
BENCH(k_lop3, , asm volatile("lop3.b32 %0, %0, %1, %2, 0x6a;" : "+r"(x[k]) : "r"(y[k]), "r"(a)))
// SHF: x = funnel(x, y) << 3
BENCH(k_shf, , asm volatile("shf.l.wrap.b32 %0, %0, %1, 3;" : "+r"(x[k]) : "r"(y[k])))
// IMAD and IADD3, one each
BENCH(k_imad_iadd3, , {
  asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(x[k]) : "r"(y[k]), "r"(a));
  asm volatile("{.reg .u32 t; add.u32 t, %0, %1; add.u32 %0, t, %2;}" : "+r"(y[k]) : "r"(x[(k + 1) % CH]), "r"(b));
})
// IMAD.WIDE and IADD3, one each
BENCH(k_wide_iadd3, , {
  WIDE(k, a);
  asm volatile("{.reg .u32 t; add.u32 t, %0, %1; add.u32 %0, t, %2;}" : "+r"(y[k]) : "r"(x[(k + 3) % CH]), "r"(b));
})
// IMAD.WIDE and two IADD3
BENCH(k_wide_2iadd3, , {
  WIDE(k, a);
  asm volatile("{.reg .u32 t; add.u32 t, %0, %1; add.u32 %0, t, %2;}" : "+r"(y[k]) : "r"(x[(k + 3) % CH]), "r"(b));
  asm volatile("{.reg .u32 t; add.u32 t, %0, %1; add.u32 %0, t, %2;}" : "+r"(x[k]) : "r"(y[(k + 5) % CH]), "r"(a));
})
// IMAD.WIDE and IMAD (both on the FMA pipe)
BENCH(k_wide_imad, , {
  WIDE(k, a);
  asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(x[k]) : "r"(y[k]), "r"(b));
})
// IMAD.HI and IADD3
BENCH(k_hi_iadd3, , {
  asm volatile("mad.hi.u32 %0, %0, %1, %2;" : "+r"(x[k]) : "r"(y[k]), "r"(a));
  asm volatile("{.reg .u32 t; add.u32 t, %0, %1; add.u32 %0, t, %2;}" : "+r"(y[k]) : "r"(x[(k + 3) % CH]), "r"(b));
})
// IMAD and LOP3 and IADD3
BENCH(k_imad_lop3_iadd3, , {
  asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(x[k]) : "r"(y[k]), "r"(a));
  asm volatile("lop3.b32 %0, %0, %1, %2, 0x6a;" : "+r"(y[k]) : "r"(x[(k + 1) % CH]), "r"(a));
  asm volatile("{.reg .u32 t; add.u32 t, %0, %1; add.u32 %0, t, %2;}" : "+r"(y[k]) : "r"(x[(k + 3) % CH]), "r"(b));
})
// IMAD.WIDE.U32 with a 64-bit addend: w = hi(w) * a + w
BENCH(k_madwide, , asm volatile("mad.wide.u32 %0, %1, %2, %0;" : "+l"(w[k]) : "r"((unsigned)(w[k] >> 32)), "r"(a)))
// IMAD.WIDE.U32 with a carry out (a product added into a 64-bit sum) and the
// carry counted by an IADD3.X: the shape of every product in the kernel
BENCH(k_widecc, , asm volatile("{.reg .u64 t; mul.wide.u32 t, %1, %2; add.cc.u64 %0, %0, t; addc.u32 %1, %1, 0;}"
                               : "+l"(w[k]), "+r"(x[k]) : "r"(a)))
// IADD3 with a carry out and IADD3.X taking it: a 64-bit add as two 32-bit ones
BENCH(k_addcc, , asm volatile("add.cc.u32 %0, %0, %1; addc.u32 %1, %1, %0;" : "+r"(x[k]), "+r"(y[k])))
// FFMA (the FP32 rate, for scale)
BENCH(k_ffma, , asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+r"(x[k]) : "r"(y[k]), "r"(a)))
// FFMA and IMAD, one each (unified INT32/FP32 cores would run both at FP32 rate)
BENCH(k_ffma_imad, , {
  asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+r"(x[k]) : "r"(y[k]), "r"(a));
  asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(y[k]) : "r"(x[(k + 1) % CH]), "r"(b));
})
// FFMA and IADD3
BENCH(k_ffma_iadd3, , {
  asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+r"(x[k]) : "r"(y[k]), "r"(a));
  asm volatile("{.reg .u32 t; add.u32 t, %0, %1; add.u32 %0, t, %2;}" : "+r"(y[k]) : "r"(x[(k + 1) % CH]), "r"(b));
})

// Register-only forms (no constant-bank operand: an earlier IADD3 ubench read 2.0 a clock from its constant
// operand where register operands give 4.0). ra, rb are runtime values ptxas cannot fold.
#define RDECL unsigned ra = x[0] ^ 5u, rb = y[1] | 3u; asm volatile("" : "+r"(ra), "+r"(rb));
#define RADD3(d, s1, s2) asm volatile("{.reg .u32 t; add.u32 t, %0, %1; add.u32 %0, t, %2;}" : "+r"(d) : "r"(s1), "r"(s2))
BENCH(r_imad, RDECL, asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(x[k]) : "r"(y[k]), "r"(ra)))
BENCH(r_imadhi, RDECL, asm volatile("mad.hi.u32 %0, %0, %1, %2;" : "+r"(x[k]) : "r"(y[k]), "r"(ra)))
BENCH(r_madwide, RDECL, asm volatile("mad.wide.u32 %0, %1, %2, %0;" : "+l"(w[k]) : "r"((unsigned)(w[k] >> 32)), "r"(ra)))
BENCH(r_lohi, RDECL, {
  asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(x[k]) : "r"(y[k]), "r"(ra));
  asm volatile("mad.hi.u32 %0, %0, %1, %2;" : "+r"(y[k]) : "r"(x[(k + 3) % CH]), "r"(rb));
})
BENCH(r_imad_iadd3, RDECL, {
  asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(x[k]) : "r"(y[k]), "r"(ra));
  RADD3(y[k], x[(k + 1) % CH], rb);
})
BENCH(r_hi_iadd3, RDECL, {
  asm volatile("mad.hi.u32 %0, %0, %1, %2;" : "+r"(x[k]) : "r"(y[k]), "r"(ra));
  RADD3(y[k], x[(k + 3) % CH], rb);
})
BENCH(r_wide_iadd3, RDECL, {
  asm volatile("mad.wide.u32 %0, %1, %2, %0;" : "+l"(w[k]) : "r"((unsigned)(w[k] >> 32)), "r"(ra));
  RADD3(y[k], x[(k + 3) % CH], rb);
})
BENCH(r_wide_2iadd3, RDECL, {
  asm volatile("mad.wide.u32 %0, %1, %2, %0;" : "+l"(w[k]) : "r"((unsigned)(w[k] >> 32)), "r"(ra));
  RADD3(y[k], x[(k + 3) % CH], rb);
  RADD3(x[k], y[(k + 5) % CH], ra);
})
BENCH(r_lohi_2iadd3, RDECL, {
  asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(x[k]) : "r"(y[k]), "r"(ra));
  asm volatile("mad.hi.u32 %0, %0, %1, %2;" : "+r"(y[k]) : "r"(x[(k + 3) % CH]), "r"(rb));
  asm volatile("{.reg .u32 a, b; mov.b64 {a, b}, %0; add.u32 a, a, %1; add.u32 a, a, %2; mov.b64 %0, {a, b};}" : "+l"(w[k]) : "r"(x[(k + 5) % CH]), "r"(rb));
})
BENCH(r_wide_imad, RDECL, {
  asm volatile("mad.wide.u32 %0, %1, %2, %0;" : "+l"(w[k]) : "r"((unsigned)(w[k] >> 32)), "r"(ra));
  asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(x[k]) : "r"(y[k]), "r"(rb));
})
BENCH(r_ffma_imad, RDECL, {
  asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+r"(x[k]) : "r"(y[k]), "r"(ra));
  asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(y[k]) : "r"(x[(k + 1) % CH]), "r"(rb));
})

// Operand-count forms of IMAD.WIDE, alone and beside IADD3s that read 2 registers and an immediate (a2) or 3
// registers (a3): does a WIDE's cost depend on how many registers it reads?
#define HW(k) ((unsigned)(w[k] >> 32))
#define W2(k) asm volatile("mul.wide.u32 %0, %1, %2;" : "=l"(w[k]) : "r"(HW(k)), "r"(y[k]))
#define W1I(k) asm volatile("mul.wide.u32 %0, %1, 0x9E3779B9;" : "=l"(w[k]) : "r"(HW(k)))
#define W3(k) asm volatile("mad.wide.u32 %0, %1, %2, %0;" : "+l"(w[k]) : "r"(HW(k)), "r"(y[k]))
#define W3I(k) asm volatile("mad.wide.u32 %0, %1, 0x9E3779B9, %0;" : "+l"(w[k]) : "r"(HW(k)))
#define A2(d, s1) asm volatile("{.reg .u32 t; add.u32 t, %0, %1; add.u32 %0, t, 0x5bd1e995;}" : "+r"(d) : "r"(s1))
#define A3(d, s1, s2) RADD3(d, s1, s2)
BENCH(o_w2, RDECL, W2(k))
BENCH(o_w1i, RDECL, W1I(k))
BENCH(o_w3, RDECL, W3(k))
BENCH(o_w3i, RDECL, W3I(k))
BENCH(o_w2a2, RDECL, { W2(k); A2(x[k], y[(k + 3) % CH]); })
BENCH(o_w2a3, RDECL, { W2(k); A3(x[k], y[(k + 3) % CH], ra); })
BENCH(o_w3a2, RDECL, { W3(k); A2(x[k], y[(k + 3) % CH]); })
BENCH(o_w3a3, RDECL, { W3(k); A3(x[k], y[(k + 3) % CH], ra); })
BENCH(o_w3ia2, RDECL, { W3I(k); A2(x[k], y[(k + 3) % CH]); })
BENCH(o_w3ia3, RDECL, { W3I(k); A3(x[k], y[(k + 3) % CH], ra); })
BENCH(o_w1ia3, RDECL, { W1I(k); A3(x[k], y[(k + 3) % CH], ra); })
BENCH(o_w3i2a2, RDECL, { W3I(k); A2(x[k], y[(k + 3) % CH]); A2(y[k], x[(k + 5) % CH]); })
BENCH(o_w3i2a3, RDECL, { W3I(k); A3(x[k], y[(k + 3) % CH], ra); A3(y[k], x[(k + 5) % CH], rb); })
BENCH(o_w1i3a3, RDECL, { W1I(k); A3(x[k], y[(k + 3) % CH], ra); A3(y[k], x[(k + 5) % CH], rb); A3(x[k], y[(k + 6) % CH], rb); })
BENCH(o_w2_3a2, RDECL, { W2(k); A2(x[k], y[(k + 3) % CH]); A2(y[k], x[(k + 5) % CH]); A2(x[k], y[(k + 6) % CH]); })
BENCH(o_a2, RDECL, A2(x[k], y[(k + 3) % CH]))
BENCH(o_a3, RDECL, A3(x[k], y[(k + 3) % CH], ra))

typedef void (*Kern)(int, unsigned *, Rec *, unsigned, unsigned);
struct Test {
  const char *name;
  Kern k;
  int per;  // instructions a chain step
};

int main(int argc, char **argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 2000;
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  int threads = 1024, blocks = p.multiProcessorCount * 2;
  printf("card %s sm_%d%d, %d SMs, %d blocks x %d threads, %d iters\n", p.name, p.major, p.minor,
         p.multiProcessorCount, blocks, threads, iters);
  unsigned *out;
  Rec *rec;
  cudaMalloc(&out, (size_t)blocks * threads * 4);
  cudaMalloc(&rec, blocks * sizeof(Rec));
  std::vector<Rec> h(blocks);
  Test tests[] = {{"IMAD", k_imad, 1},           {"IMAD.WIDE", k_imadwide, 1},  {"IMAD.HI", k_imadhi, 1},
                  {"IADD3", k_iadd3, 1},         {"LOP3", k_lop3, 1},           {"SHF", k_shf, 1},
                  {"IMAD+IADD3", k_imad_iadd3, 2}, {"WIDE+IADD3", k_wide_iadd3, 2}, {"WIDE+2IADD3", k_wide_2iadd3, 3},
                  {"WIDE+IMAD", k_wide_imad, 2}, {"HI+IADD3", k_hi_iadd3, 2},
                  {"IMAD+LOP3+IADD3", k_imad_lop3_iadd3, 3}, {"FFMA", k_ffma, 1}, {"FFMA+IMAD", k_ffma_imad, 2},
                  {"FFMA+IADD3", k_ffma_iadd3, 2}, {"MAD.WIDE(addend)", k_madwide, 1},
                  {"WIDE.cc+IADD3.X", k_widecc, 2}, {"IADD3.cc+IADD3.X", k_addcc, 2},
                  {"r:IMAD", r_imad, 1}, {"r:IMAD.HI", r_imadhi, 1}, {"r:MAD.WIDE", r_madwide, 1},
                  {"r:IMAD+HI", r_lohi, 2}, {"r:IMAD+IADD3", r_imad_iadd3, 2}, {"r:HI+IADD3", r_hi_iadd3, 2},
                  {"r:WIDE+IADD3", r_wide_iadd3, 2}, {"r:WIDE+2IADD3", r_wide_2iadd3, 3},
                  {"r:IMAD+HI+IADD3", r_lohi_2iadd3, 3}, {"r:WIDE+IMAD", r_wide_imad, 2},
                  {"r:FFMA+IMAD", r_ffma_imad, 2},
                  {"o:W2", o_w2, 1}, {"o:W1I", o_w1i, 1}, {"o:W3", o_w3, 1}, {"o:W3I", o_w3i, 1},
                  {"o:W2+A2", o_w2a2, 2}, {"o:W2+A3", o_w2a3, 2}, {"o:W3+A2", o_w3a2, 2}, {"o:W3+A3", o_w3a3, 2},
                  {"o:W3I+A2", o_w3ia2, 2}, {"o:W3I+A3", o_w3ia3, 2}, {"o:W1I+A3", o_w1ia3, 2},
                  {"o:W3I+2A2", o_w3i2a2, 3}, {"o:W3I+2A3", o_w3i2a3, 3}, {"o:W1I+3A3", o_w1i3a3, 4},
                  {"o:W2+3A2", o_w2_3a2, 4}, {"o:A2", o_a2, 1}, {"o:A3", o_a3, 1}};
  for (auto &t : tests) {
    t.k<<<blocks, threads>>>(10, out, rec, 7, 9);  // warm up
    cudaDeviceSynchronize();
    t.k<<<blocks, threads>>>(iters, out, rec, 7, 9);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
      printf("%s: %s\n", t.name, cudaGetErrorString(e));
      return 1;
    }
    cudaMemcpy(h.data(), rec, blocks * sizeof(Rec), cudaMemcpyDeviceToHost);
    std::map<unsigned, std::pair<long long, long long>> span;
    std::map<unsigned, int> nb;
    for (auto &r : h) {
      auto it = span.find(r.sm);
      if (it == span.end()) span[r.sm] = {r.t0, r.t1};
      else {
        if (r.t0 < it->second.first) it->second.first = r.t0;
        if (r.t1 > it->second.second) it->second.second = r.t1;
      }
      nb[r.sm]++;
    }
    double rate = 0;
    for (auto &s : span) {
      double warp_instr = (double)nb[s.first] * (threads / 32) * (double)iters * UNR * CH * t.per;
      rate += warp_instr / (double)(s.second.second - s.second.first);
    }
    rate /= span.size();
    printf("%-18s %6.3f warp-instr/SM/clk  (%6.1f thread-ops/SM/clk)\n", t.name, rate, rate * 32);
  }
  return 0;
}
