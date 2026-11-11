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
//   DREAL   1: a unit reads its four table offsets from a slice first, as a real draw must
//   DSYNC   1: one shared add after the draw and one shared read and warp-uniform test before the loads, a k-step
//   WARPS   warps a block: 4, 8, 12 or 16 (one to four a scheduler, if warp w runs on scheduler w % 4), with
//   ACC     mma a k-step a warp (48; 32 fits twelve warps' registers, 24 sixteen's). Only the mma-only and
//           fragment-load streams build at other values than 8 and 48.
//   WMASK   the warps that run (bit w); the others leave at once. Two warps that share a tensor pipe take
//           twice the cycles an mma of two that do not, which is how a card's warp-to-scheduler map is read.
//   FILLRND 1: the fragments are loaded from hashed bytes, not from the mostly-zero offset slice (the draw
//           then must not be DREAL). Operand bytes change the watts, not the pipe use.
//   PAD     steps after every mma (0: none), all of the kind
//   PKIND   0: an add into one of RCH registers no mma writes (independent steps); 1: an XOR3 of two sums the
//           mma 8 earlier wrote (the readout's step, spread one an mma); 2: a shared load and an XOR of what
//           it read; 3: a shared store; 4: an add into one register (each waits on the one before).
//           What a step between the mma costs by kind differs by architecture: on sm_120 the readout and
//           the draw cost several times what they cost on sm_86 and sm_89.
//
// The result line: pipe use at 16 cycles an mma a scheduler (mean, and the least and most over schedulers),
// cycles an mma a scheduler, the last launch's time, the launches, MACs a second by count (4096 an mma, over
// the launches' own time; in a run of more than 8 s, over the launches that began after its first 8 s), and the
// SM's cycle counter in MHz (cycles over the last launch's time). A fourth argument adds each running warp's
// own cycles an mma. The streams compute nothing real: a rate by count is not a mining rate.
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
#ifndef ACC
#define ACC 48
#endif
#ifndef WARPS
#define WARPS 8
#endif
#ifdef WMASK
#define RUNMASK (WMASK)
#else
#define RUNMASK 0xffffffffu
#endif
#ifndef FILLRND
#define FILLRND 0
#endif
#ifndef PAD
#define PAD 0
#endif
#ifndef PKIND
#define PKIND 0
#endif
#if (WARPS != 8 || ACC != 48) && (RD != 0 || DRAW != 0)
#error the readout and the draw are laid out for 8 warps of 48 mma
#endif
#if FILLRND && DREAL
#error DREAL reads its offsets from the slice FILLRND fills with hashed bytes
#endif
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
    shw[i] = (!FILLRND && i >= 96u && i < 2048u) ? ((i * 2654435761u) >> 7) & 2040u : i * 2654435761u;
  __syncthreads();
  unsigned a[RT][4], b[CT][2];
  int c[ACC][4];
  unsigned t[RCH];
  const unsigned char *const frag = sh + 1024u + (lane & 7u) * 16u + (lane >> 3) * 128u;
  const unsigned char *const dtab = sh + 8192u + warp * 4096u + lane * 8u;
  unsigned char *const dslab = sh + 44u * 1024u + warp * 1024u + lane * 16u;
  unsigned *const jw = shw + 64u + tid * 16u;
  unsigned dro = lane * 24u;
#if PAD > 0
  unsigned px = tid * 2654435761u + 1u, pchain = tid;
  const unsigned pa = (unsigned)__cvta_generic_to_shared(sh + 44u * 1024u + warp * 1024u + lane * 16u);
#endif
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
#if PAD > 0
  // One step after mma i, of the kind PKIND.
  auto pad = [&](int i, int p) {
    const int r = (i * PAD + p) % RCH;
#if PKIND == 0
    asm volatile("add.u32 %0, %0, %1;\n" : "+r"(t[r]) : "r"(px));
#elif PKIND == 1
    const int j = (i + ACC - 8) % ACC;
    asm volatile("xor.b32 %0, %0, %1;\n xor.b32 %0, %0, %2;\n" : "+r"(t[r])
                 : "r"(c[j][(2 * p) & 3]), "r"(c[j][(2 * p + 1) & 3]));
#elif PKIND == 2
    unsigned v;
    asm volatile("ld.volatile.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(pa));
    asm volatile("xor.b32 %0, %0, %1;\n" : "+r"(t[r]) : "r"(v));
#elif PKIND == 3
    asm volatile("st.volatile.shared.u32 [%0], %1;\n" :: "r"(pa), "r"(px) : "memory");
#else
    asm volatile("add.u32 %0, %0, %1;\n" : "+r"(pchain) : "r"(px));
#endif
  };
#endif
  // One k-step: 48 mma, with the readout's runs and the draw where the switches put them. s is 0 .. 3.
  auto kstep = [&](int s, int ch) {
#pragma unroll
    for (int i = 0; i < ACC; i++) {
      mma(i);
#if PAD > 0
#pragma unroll
      for (int p = 0; p < PAD; p++) pad(i, p);
#endif
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

#ifdef WMASK
  const int nch = ((RUNMASK >> warp) & 1u) ? chunks : 0;
#else
  const int nch = chunks;
#endif
  const long long t0 = clock64();
  for (int ch = 0; ch < nch; ch++) {
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
#if PAD > 0
  sum ^= (int)(px + pchain);
#endif
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
    fprintf(stderr, "pipeprobe2 name chunks [secs] [v]\n");
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
  float ms = 0, total = 0, late = 0;
  int launches = 0, nlate = 0;
  do {
    const bool settled = total >= 8000.0f;   // a capped card's clock has come down by then
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
    if (settled) {
      late += ms;
      nlate++;
    }
  } while (total < secs * 1000.0);
  cudaMemcpy(h, times, sizeof(long long) * blocks * 2 * WARPS, cudaMemcpyDeviceToHost);
  double use_sum = 0, use_min = 1e9, use_max = 0, cyc_sum = 0, own[WARPS] = {0};
  int nsched = 0;
  for (int bk = 0; bk < blocks; bk++) {
    for (int s = 0; s < 4; s++) {
      long long lo = 0, hi = 0;
      bool any = false;
      int ran = 0;
      for (int w = s; w < WARPS; w += 4) {
        const long long a0 = h[bk * 2 * WARPS + 2 * w], a1 = h[bk * 2 * WARPS + 2 * w + 1];
        if (!((RUNMASK >> w) & 1u)) continue;
        ran++;
        own[w] += (double)(a1 - a0) / ((double)chunks * 4.0 * ACC) / blocks;
        if (!any || a0 < lo) lo = a0;
        if (!any || a1 > hi) hi = a1;
        any = true;
      }
      if (!ran) continue;
      const double use = 16.0 * ran * chunks * 4.0 * ACC / (double)(hi - lo);
      cyc_sum += (double)(hi - lo);
      use_sum += use;
      if (use < use_min) use_min = use;
      if (use > use_max) use_max = use;
      nsched++;
    }
  }
  const double use = use_sum / nsched;
  int running = 0;
  for (int w = 0; w < WARPS; w++) running += (RUNMASK >> w) & 1u;
  // By count: over the launches that began 8 s or more into the run, or over all of a shorter run.
  const double macs = 4096.0 * (nlate ? nlate : launches) * blocks * running * chunks * 4.0 * ACC /
                      ((nlate ? late : total) / 1000.0);
  printf("%-26s pipe %6.2f%% (%.2f-%.2f) %6.3f cyc/IMMA %.0f ms x%d | %.2f T-MAC/s by count, counter %.0f MHz, "
         "%d SMs, %d of %d warps\n", name, use * 100, use_min * 100, use_max * 100, 16.0 / use, ms, launches,
         macs / 1e12, cyc_sum / nsched / (ms * 1000.0), blocks, running, WARPS);
  if (argc > 4) {
    printf("%-26s own cycles an mma, by warp:", name);
    for (int w = 0; w < WARPS; w++)
      if ((RUNMASK >> w) & 1u) printf(" %d:%.2f", w, own[w]);
    printf("\n");
  }
  return 0;
}
