// pipeprobe2 -- the fold's stream with nothing decided at run time: 8 warps, two a scheduler, 48 mma a k-step,
// 4 k-steps a chunk, and between them only what the build's switches put there. No branch stands between
// two mma, so ptxas adds no convergence fences. Our own code, for the nv-kernel study.
//
//   pipeprobe2 <name> <chunks> [secs]
//
// Switches (compile time):
//   KLOAD   1: reload the held fragments by ldmatrix after every k-step (RT + CT / 2 ldmatrix)
//   RD      0: no readout. 1: one run of 96 XOR3 at the chunk's end. 2: three runs of 32, each after its tiles'
//           last mma (after mma 24 and 40 of the last k-step, after mma 8 of the next chunk's first).
//           3: six runs of 16 (after mma 12, 20, 28, 36, 44 of the last k-step and mma 4 of the first).
//   RCH     XOR chains the readout keeps (registers)
//   TRW     1: the chunk's word goes into the thread's own transcript in shared (load, rotate, store)
//   DRAW    draw-shaped units a k-step (0: none); each is 16 bytes from four 8-byte table reads
//   DPOS    where: 0 at the k-step's end, 1 after mma 24, 2 half after mma 16 and half after mma 40,
//           3 one unit after every 48 / DRAW mma
//   DGRP    1: two units' steps grouped by kind (loads, subtracts, byte selects, stores)
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#ifndef KLOAD
#define KLOAD 1
#endif
#ifndef RD
#define RD 0
#endif
#ifndef RCH
#define RCH 8
#endif
#ifndef TRW
#define TRW 1
#endif
#ifndef DRAW
#define DRAW 0
#endif
#ifndef DPOS
#define DPOS 0
#endif
#ifndef DGRP
#define DGRP 0
#endif
#ifndef DREAL
#define DREAL 0
#endif
#ifndef DSYNC
#define DSYNC 0
#endif
#ifndef RT
#define RT 2
#endif
#ifndef CT
#define CT 4
#endif
#define ACC 48
#define WARPS 8
#define THREADS (WARPS * 32)
#define SHBYTES (56 * 1024)

__device__ __forceinline__ void ldsm4(unsigned &r0, unsigned &r1, unsigned &r2, unsigned &r3, const void *p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

extern "C" __global__ __launch_bounds__(THREADS, 1) void pipeprobe2(long long *times, int *sink, int chunks) {
  extern __shared__ __align__(16) unsigned char sh[];
  const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
  unsigned *const shw = reinterpret_cast<unsigned *>(sh);
  for (unsigned i = tid; i < SHBYTES / 4u; i += THREADS)
    shw[i] = (i >= 96u && i < 2048u) ? ((i * 2654435761u) >> 7) & 2040u : i * 2654435761u;
  __syncthreads();
  unsigned a[RT][4], b[CT][2];
  int c[ACC][4];
  unsigned t[RCH];
  const unsigned char *const frag = sh + 1024u + (lane & 7u) * 16u + (lane >> 3) * 128u;
  const unsigned char *const dtab = sh + 8192u + warp * 4096u + lane * 8u;
  unsigned char *const dslab = sh + 44u * 1024u + warp * 1024u + lane * 16u;
  unsigned *const jw = shw + 64u + tid * 16u;
  unsigned dro = lane * 24u;
#pragma unroll
  for (int i = 0; i < RT; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) a[i][j] = 0x01010101u + tid + i + j;
#pragma unroll
  for (int i = 0; i < CT; i++)
#pragma unroll
    for (int j = 0; j < 2; j++) b[i][j] = 0x01010101u + tid + 2 * i + j;
#pragma unroll
  for (int i = 0; i < ACC; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) c[i][j] = 0;
#pragma unroll
  for (int i = 0; i < RCH; i++) t[i] = 0;

  auto mma = [&](int i) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+r"(c[i][0]), "+r"(c[i][1]), "+r"(c[i][2]), "+r"(c[i][3])
                 : "r"(a[i / CT % RT][0]), "r"(a[i / CT % RT][1]), "r"(a[i / CT % RT][2]), "r"(a[i / CT % RT][3]),
                   "r"(b[i % CT][0]), "r"(b[i % CT][1]));
  };
  auto xrun = [&](int lo, int hi) {
#pragma unroll
    for (int i = lo; i < hi; i++) {
      asm volatile("xor.b32 %0, %0, %1;\n xor.b32 %0, %0, %2;\n" : "+r"(t[(2 * i) % RCH])
                   : "r"(c[i][0]), "r"(c[i][1]));
      asm volatile("xor.b32 %0, %0, %1;\n xor.b32 %0, %0, %2;\n" : "+r"(t[(2 * i + 1) % RCH])
                   : "r"(c[i][2]), "r"(c[i][3]));
    }
  };
  auto word = [&](int ch) {
    unsigned x = 0;
#pragma unroll
    for (int i = 0; i < RCH; i++) {
      x ^= t[i];
      t[i] = 0;
    }
#if TRW
    const unsigned j0 = jw[ch & 15];
    jw[ch & 15] = ((j0 << 13) | (j0 >> 19)) ^ x;
#else
    atomicXor(shw + 16u + (lane & 7u), x);
#endif
  };
  auto unit = [&]() {
#if DREAL
    const uint4 o4 = *reinterpret_cast<const uint4 *>(sh + 384u + (dro & 4080u));
    dro += 16u;
    const unsigned long long a0 = *reinterpret_cast<const unsigned long long *>(dtab + o4.x);
    const unsigned long long a1 = *reinterpret_cast<const unsigned long long *>(dtab + o4.y);
    const unsigned long long a2 = *reinterpret_cast<const unsigned long long *>(dtab + o4.z);
    const unsigned long long a3 = *reinterpret_cast<const unsigned long long *>(dtab + o4.w);
#else
    const unsigned long long *q = reinterpret_cast<const unsigned long long *>(dtab + (dro & 2040u));
    dro += 40u;
    const unsigned long long a0 = q[0], a1 = q[5], a2 = q[9], a3 = q[14];
#endif
    const unsigned w0 = (unsigned)a0 - (unsigned)(a1 >> 32), w1 = (unsigned)(a0 >> 32) - (unsigned)a1;
    const unsigned w2 = (unsigned)a2 - (unsigned)(a3 >> 32), w3 = (unsigned)(a2 >> 32) - (unsigned)a3;
    *reinterpret_cast<uint4 *>(dslab + (dro & 496u)) =
        make_uint4(__byte_perm(w0, w1, 0x5140u), __byte_perm(w2, w3, 0x5140u), __byte_perm(w0, w1, 0x7362u),
                   __byte_perm(w2, w3, 0x7362u));
  };
  auto unit2 = [&]() {
    const unsigned long long *q = reinterpret_cast<const unsigned long long *>(dtab + (dro & 2040u));
    dro += 72u;
    unsigned long long a0 = q[0], a1 = q[5], a2 = q[9], a3 = q[14], a4 = q[17], a5 = q[22], a6 = q[26], a7 = q[31];
    asm volatile("" : "+l"(a0), "+l"(a1), "+l"(a2), "+l"(a3), "+l"(a4), "+l"(a5), "+l"(a6), "+l"(a7));
    unsigned w0 = (unsigned)a0 - (unsigned)(a1 >> 32), w1 = (unsigned)(a0 >> 32) - (unsigned)a1;
    unsigned w2 = (unsigned)a2 - (unsigned)(a3 >> 32), w3 = (unsigned)(a2 >> 32) - (unsigned)a3;
    unsigned w4 = (unsigned)a4 - (unsigned)(a5 >> 32), w5 = (unsigned)(a4 >> 32) - (unsigned)a5;
    unsigned w6 = (unsigned)a6 - (unsigned)(a7 >> 32), w7 = (unsigned)(a6 >> 32) - (unsigned)a7;
    asm volatile("" : "+r"(w0), "+r"(w1), "+r"(w2), "+r"(w3), "+r"(w4), "+r"(w5), "+r"(w6), "+r"(w7));
    *reinterpret_cast<uint4 *>(dslab + (dro & 480u)) =
        make_uint4(__byte_perm(w0, w1, 0x5140u), __byte_perm(w2, w3, 0x5140u), __byte_perm(w0, w1, 0x7362u),
                   __byte_perm(w2, w3, 0x7362u));
    *reinterpret_cast<uint4 *>(dslab + (dro & 480u) + 16) =
        make_uint4(__byte_perm(w4, w5, 0x5140u), __byte_perm(w6, w7, 0x5140u), __byte_perm(w4, w5, 0x7362u),
                   __byte_perm(w6, w7, 0x7362u));
  };
  auto draw = [&](int n) {
#if DGRP
#pragma unroll
    for (int u = 0; u < n / 2; u++) unit2();
    if (n & 1) unit();
#else
#pragma unroll
    for (int u = 0; u < n; u++) unit();
#endif
  };
  // One k-step: 48 mma, with the readout's runs and the draw where the switches put them. s is 0 .. 3.
  auto kstep = [&](int s, int ch) {
#pragma unroll
    for (int i = 0; i < ACC; i++) {
      mma(i);
      const int n = i + 1;   // mma issued so far in this k-step
#if RD == 2
      if (s == 3 && n == 24) xrun(0, 16);
      if (s == 3 && n == 40) xrun(16, 32);
      if (s == 0 && n == 8) { xrun(32, 48); word(ch + 15); }
#elif RD == 3
      if (s == 3 && n >= 12 && n <= 44 && (n - 12) % 8 == 0) xrun((n - 12), (n - 12) + 8);
      if (s == 0 && n == 4) { xrun(40, 48); word(ch + 15); }
#endif
#if DRAW > 0 && DPOS == 1
      if (n == 24) draw(DRAW);
#elif DRAW > 0 && DPOS == 2
      if (n == 16) draw(DRAW / 2);
      if (n == 40) draw(DRAW - DRAW / 2);
#elif DRAW > 0 && DPOS == 3
      if (n % (ACC / DRAW) == 0) draw(1);
#endif
    }
#if DRAW > 0 && DPOS == 0
    draw(DRAW);
#endif
#if DSYNC
    atomicAdd(shw + 8u + ((unsigned)s & 1u), 1u);
    {
      const unsigned seen = shw[8u + (((unsigned)s + 1u) & 1u)];
      if (__any_sync(0xffffffffu, seen == 0xdeadbeefu)) asm volatile("nanosleep.u32 64;\n");
    }
#endif
#if KLOAD
    const unsigned char *const f = frag + s * 1024;
#pragma unroll
    for (int i = 0; i < RT; i++) ldsm4(a[i][0], a[i][1], a[i][2], a[i][3], f + i * 1024);
#pragma unroll
    for (int i = 0; i < CT; i += 2) ldsm4(b[i][0], b[i][1], b[i + 1][0], b[i + 1][1], f + 4096 + i * 512);
#endif
#if RD == 1
    if (s == 3) { xrun(0, 48); word(ch); }
#endif
  };

  const long long t0 = clock64();
  for (int ch = 0; ch < chunks; ch++) {
    kstep(0, ch);
    kstep(1, ch);
    kstep(2, ch);
    kstep(3, ch);
  }
  const long long t1 = clock64();
  if (lane == 0) {
    times[blockIdx.x * (2 * WARPS) + 2 * warp] = t0;
    times[blockIdx.x * (2 * WARPS) + 2 * warp + 1] = t1;
  }
  int sum = (int)dro;
#pragma unroll
  for (int i = 0; i < RCH; i++) sum ^= (int)t[i];
#pragma unroll
  for (int i = 0; i < ACC; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) sum += c[i][j];
  sink[blockIdx.x * THREADS + tid] = sum;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "pipeprobe2 name chunks [secs]\n");
    return 2;
  }
  const char *name = argv[1];
  const int chunks = atoi(argv[2]);
  const double secs = argc > 3 ? atof(argv[3]) : 0.0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  const int blocks = prop.multiProcessorCount;
  long long *times;
  int *sink;
  cudaMalloc(&times, sizeof(long long) * blocks * 2 * WARPS);
  cudaMalloc(&sink, sizeof(int) * blocks * THREADS);
  long long *h = (long long *)malloc(sizeof(long long) * blocks * 2 * WARPS);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaFuncSetAttribute(pipeprobe2, cudaFuncAttributeMaxDynamicSharedMemorySize, SHBYTES);
  float ms = 0, total = 0;
  int launches = 0;
  do {
    cudaEventRecord(e0);
    pipeprobe2<<<blocks, THREADS, SHBYTES>>>(times, sink, chunks);
    cudaError_t err = cudaGetLastError();
    cudaEventRecord(e1);
    if (err == cudaSuccess) err = cudaEventSynchronize(e1);
    if (err != cudaSuccess) {
      fprintf(stderr, "%s: %s\n", name, cudaGetErrorString(err));
      return 1;
    }
    cudaEventElapsedTime(&ms, e0, e1);
    total += ms;
    launches++;
  } while (total < secs * 1000.0);
  cudaMemcpy(h, times, sizeof(long long) * blocks * 2 * WARPS, cudaMemcpyDeviceToHost);
  double use_sum = 0, use_min = 1e9, use_max = 0;
  int nsched = 0;
  for (int bk = 0; bk < blocks; bk++) {
    for (int s = 0; s < 4; s++) {
      long long lo = 0, hi = 0;
      bool any = false;
      for (int w = s; w < WARPS; w += 4) {
        const long long a0 = h[bk * 2 * WARPS + 2 * w], a1 = h[bk * 2 * WARPS + 2 * w + 1];
        if (!any || a0 < lo) lo = a0;
        if (!any || a1 > hi) hi = a1;
        any = true;
      }
      const double use = 16.0 * 2.0 * chunks * 4.0 * ACC / (double)(hi - lo);
      use_sum += use;
      if (use < use_min) use_min = use;
      if (use > use_max) use_max = use;
      nsched++;
    }
  }
  const double use = use_sum / nsched;
  printf("%-26s pipe %6.2f%% (%.2f-%.2f) %6.3f cyc/IMMA pg 0 %.0f ms x%d\n", name, use * 100, use_min * 100,
         use_max * 100, 16.0 / use, ms, launches);
  return 0;
}
