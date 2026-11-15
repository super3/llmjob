// powcap.cu (goal-nv50w): which kind of work trips a capped card's "SW Power Cap", and at what board power.
//
// On a 575 W RTX 5090 our fold sits at 539 W under "SW Power Cap", a timing-only variant of it at 510 W at the same
// clock, and SRBMiner at 571 W. So the board total is not what binds; some part of it is. Each mode here is one
// kind of work alone, at full occupancy, for a few seconds, while the program reads NVML every 100 ms: board power,
// SM clock and the clock-event reasons. A mode that is held by the cap at a low board power loads the part that
// binds. Plain observation; nothing is set on the card.
//
//   nvcc -arch=sm_120 -O3 -o powcap powcap.cu -ldl && ./powcap <seconds a mode> [mode ...]
//
// Modes: lds (LDS.128, no bank conflicts), ldsm (ldmatrix x4), imma (mma m16n8k32 s8, random operands), immaz (the
// same, operands zero), ffma (FP32 FMA chains), mix<N> (one LDS.128 every N IMMA, as a fold's loop mixes them),
// idle. A line a mode: work a second, mean board watts, mean SM MHz, and the share of samples with each reason bit
// (b2 = SW power cap, b3 = HW slowdown, b5 = SW thermal, b6 = HW thermal, b7 = HW power brake, b8 = display clock).
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define THREADS 256
#define ITERS 4096

__device__ __forceinline__ void mma8(int (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// mode: 0 lds, 1 ldsm, 2 imma, 3 immaz, 4 ffma, 5 mix (every `mixn` IMMA one LDS.128)
__global__ void __launch_bounds__(THREADS) work(int mode, int mixn, uint32_t seed, unsigned long long *sink) {
  __shared__ __align__(16) uint32_t sm[THREADS * 4 * 4];
  const int t = threadIdx.x;
  uint32_t x = seed ^ (t * 2654435761u) ^ (blockIdx.x * 40503u);
  for (int i = t; i < THREADS * 16; i += THREADS) {
    x = x * 1664525u + 1013904223u;
    sm[i] = x;
  }
  __syncthreads();
  unsigned long long acc = 0;
  if (mode == 0) {
    uint4 r = make_uint4(0, 0, 0, 0);
    const uint4 *p = reinterpret_cast<const uint4 *>(sm);
    int idx = t;
#pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#pragma unroll
      for (int j = 0; j < 16; j++) {
        uint4 v = p[(idx + j * 32) & (THREADS * 4 - 1)];
        r.x ^= v.x; r.y += v.y; r.z ^= v.z; r.w += v.w;
      }
      idx += r.x & 32;
    }
    acc = r.x + r.y + r.z + r.w;
  } else if (mode == 1) {
    uint32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0;
    uint32_t base = (uint32_t)__cvta_generic_to_shared(sm) + (t & 31) * 16 + (t >> 5) * 512;
#pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#pragma unroll
      for (int j = 0; j < 16; j++) {
        uint32_t a0, a1, a2, a3;
        uint32_t ad = base + ((j * 4096 + (r0 & 16)) & 16383) ;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(ad & ~15u));
        r0 ^= a0; r1 += a1; r2 ^= a2; r3 += a3;
      }
    }
    acc = r0 + r1 + r2 + r3;
  } else if (mode == 2 || mode == 3 || mode == 5) {
    unsigned a[6][4], b[4][2];
    for (int i = 0; i < 6; i++)
      for (int j = 0; j < 4; j++) { x = x * 1664525u + 1013904223u; a[i][j] = mode == 3 ? 0 : x; }
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 2; j++) { x = x * 1664525u + 1013904223u; b[i][j] = mode == 3 ? 0 : x; }
    int d[6][4][4];
    for (int i = 0; i < 6; i++)
      for (int j = 0; j < 4; j++)
        for (int k = 0; k < 4; k++) d[i][j][k] = 0;
    const uint4 *p = reinterpret_cast<const uint4 *>(sm);
    int idx = t;
    int n = 0;
#pragma unroll 1
    for (int it = 0; it < ITERS / 4; it++) {
#pragma unroll
      for (int i = 0; i < 6; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
          mma8(d[i][j], a[i], b[j]);
          if (mode == 5 && ++n >= mixn) {
            n = 0;
            uint4 v = p[(idx + i * 32 + j * 64) & (THREADS * 4 - 1)];
            // The load feeds the operands, so it is not dead, and the operands change as a fold's do.
            a[i][0] ^= v.x; a[i][1] ^= v.y; b[j][0] ^= v.z; b[j][1] ^= v.w;
          }
        }
      }
      idx += 32;
    }
    for (int i = 0; i < 6; i++)
      for (int j = 0; j < 4; j++)
        for (int k = 0; k < 4; k++) acc += (unsigned)d[i][j][k];
  } else if (mode == 4) {
    float f[16];
    for (int i = 0; i < 16; i++) f[i] = (float)(x >> (i + 1)) * 1e-9f;
    const float m = 0.9999f, c = 1e-7f;
#pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#pragma unroll
      for (int j = 0; j < 16; j++)
#pragma unroll
        for (int k = 0; k < 4; k++) f[j] = fmaf(f[j], m, c);
    }
    for (int i = 0; i < 16; i++) acc += (unsigned)__float_as_uint(f[i]);
  }
  if (acc == 0x123456789ull) sink[0] = acc;  // never true; keeps the work live
}

typedef void *nd_t;
static int (*n_init)(void), (*n_idx)(unsigned, nd_t *), (*n_pw)(nd_t, unsigned *), (*n_clk)(nd_t, int, unsigned *),
    (*n_rs)(nd_t, unsigned long long *);
static nd_t dev;

static double now_s(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
  double secs = argc > 1 ? atof(argv[1]) : 30;
  void *h = dlopen("libnvidia-ml.so.1", RTLD_NOW);
  if (!h) { printf("powcap no NVML\n"); return 1; }
  n_init = (int (*)(void))dlsym(h, "nvmlInit_v2");
  n_idx = (int (*)(unsigned, nd_t *))dlsym(h, "nvmlDeviceGetHandleByIndex_v2");
  n_pw = (int (*)(nd_t, unsigned *))dlsym(h, "nvmlDeviceGetPowerUsage");
  n_clk = (int (*)(nd_t, int, unsigned *))dlsym(h, "nvmlDeviceGetClockInfo");
  n_rs = (int (*)(nd_t, unsigned long long *))dlsym(h, "nvmlDeviceGetCurrentClocksEventReasons");
  if (!n_rs) n_rs = (int (*)(nd_t, unsigned long long *))dlsym(h, "nvmlDeviceGetCurrentClocksThrottleReasons");
  if (!n_init || n_init() != 0 || n_idx(0, &dev) != 0) { printf("powcap NVML init failed\n"); return 1; }
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  unsigned long long *sink;
  cudaMalloc(&sink, 8);
  const char *defs[] = {"idle", "lds", "imma", "immaz", "ffma", "mix8", "mix4", "mix16", "ldsm", "imma"};
  int nm = argc > 2 ? argc - 2 : (int)(sizeof defs / sizeof defs[0]);
  cudaStream_t st;
  cudaStreamCreate(&st);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  for (int m = 0; m < nm; m++) {
    const char *name = argc > 2 ? argv[m + 2] : defs[m];
    int mode = -1, mixn = 0;
    if (!strcmp(name, "lds")) mode = 0;
    else if (!strcmp(name, "ldsm")) mode = 1;
    else if (!strcmp(name, "imma")) mode = 2;
    else if (!strcmp(name, "immaz")) mode = 3;
    else if (!strcmp(name, "ffma")) mode = 4;
    else if (!strncmp(name, "mix", 3)) { mode = 5; mixn = atoi(name + 3); if (mixn < 1) mixn = 8; }
    else if (strcmp(name, "idle")) { printf("powcap unknown mode %s\n", name); continue; }
    const int blocks = sms * 4;
    double t0 = now_s(), warm = 10.0, wsum = 0, csum = 0;
    int ns = 0, launches = 0, counted = 0;
    int bits[32] = {0};
    float kms = 0;
    double next = t0;
    while (now_s() - t0 < secs + warm) {
      if (mode >= 0) {
        // A few launches queued, then one sample of NVML: the card is never idle between them.
        cudaEventRecord(e0, st);
        for (int q = 0; q < 8; q++) work<<<blocks, THREADS, 0, st>>>(mode, mixn, 12345u + launches * 7u + q, sink);
        cudaEventRecord(e1, st);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        launches += 8;
        if (now_s() - t0 >= warm) { kms += ms; counted += 8; }
      } else {
        usleep(50000);
      }
      double tn = now_s();
      if (tn - t0 >= warm && tn >= next) {
        next = tn + 0.1;
        unsigned mw = 0, mhz = 0;
        unsigned long long r = 0;
        if (n_pw(dev, &mw) == 0 && n_clk(dev, 1, &mhz) == 0) {
          wsum += mw / 1000.0; csum += mhz; ns++;
          if (n_rs && n_rs(dev, &r) == 0)
            for (int b = 0; b < 32; b++) if (r >> b & 1) bits[b]++;
        }
      }
    }
    cudaError_t err = cudaGetLastError();
    // Work a second: kernel launches a second, as each mode's own unit (a launch is blocks x THREADS x ITERS steps).
    printf("powcap %s launches/s=%.2f W=%.1f MHz=%.0f n=%d", name, counted && kms > 0 ? counted / (kms / 1000.0) : 0.0,
           ns ? wsum / ns : 0.0, ns ? csum / ns : 0.0, ns);
    for (int b = 0; b < 32; b++) if (bits[b]) printf(" b%d=%.0f", b, 100.0 * bits[b] / ns);
    printf("%s%s\n", err ? " err=" : "", err ? cudaGetErrorString(err) : "");
    fflush(stdout);
  }
  return 0;
}
