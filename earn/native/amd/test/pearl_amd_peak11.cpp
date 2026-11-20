// pearl_amd_peak's first question for a wave64 RDNA 3 fold (goal-f-rdna3, 20 November): does gfx11 run
// v_wmma_i32_16x16x16_iu8 at the same rate and clock in wave64 as in wave32, and what do the fragment reads cost in
// each? Timing only; it checks nothing. Build for gfx1100 or gfx1101, wave32 as it is or wave64 with
// -mwavefrontsize64 -DPK_W=64:
//
//   clang++ -x hip --offload-arch=gfx1100 -O3 test/pearl_amd_peak11.cpp -o peak11
//   peak11 [secs=20] [variant]
//
// A block is 256 threads over 128 x 256 of C with the same 128 accumulator VGPRs a wave either way: 8 waves32 of
// 64 x 64 (4 x 4 WMMA tiles, 8 VGPRs each) or 4 waves64 of 128 x 64 (8 x 4 tiles, 4 VGPRs each). In both a lane holds
// all 16 bytes of its A row and B column, so a k-step reads 4 + 4 fragments for 16 WMMAs in wave32 and 8 + 4 for 32
// in wave64. Variants (PK_MT x PK_NT is the wave's tile count; -DPK_MT=4 -DPK_NT=8 builds 64 x 128 waves64):
//   regs    fragments stay in registers
//   lds     every k-step reads its fragments from LDS (ds_load_b128 a lane), no barrier
//   ldsa    A from LDS, B stays in registers (the fold with B' straight from memory, less the memory loads)
//   ldsbar  lds with the fold's LDS barrier every two k-steps
//   ldsagb  A from LDS, B from global memory with one load a lane (a small buffer that stays in L2)
//   gbbar   ldsagb with the barrier: the shape of the fold with B' straight to registers
//   ldsm, ldsam, ldsagbm, gbbarm  (wave64 only) the same with every load issued in lanes 0-31 only. gfx11's wave64
//           WMMA reads A and B from lanes 0-31 and never from lanes 32-63 (test/pearl_amd_lay64_11.cpp), so a
//           masked load delivers the 512 bytes a wave32 load does, to a wave of twice the tiles.
// Operands are the miner's fill, 63. Prints TMAC/s; run the clock probe beside it for the clock.
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef PK_W
#define PK_W 32
#endif
#ifndef PK_MT
#define PK_MT (PK_W == 64 ? 8 : 4)
#endif
#ifndef PK_NT
#define PK_NT 4
#endif
#define PK_ROW 48   // an LDS row: 32 bytes of k and 16 of padding, as the wave32 fold's

typedef int32_t pk_v4i __attribute__((ext_vector_type(4)));
typedef int32_t pk_v8i __attribute__((ext_vector_type(8)));
#if PK_W == 64
typedef pk_v4i pk_acc;
#else
typedef pk_v8i pk_acc;
#endif

__device__ __forceinline__ void pk_mma(pk_acc &c, pk_v4i a, pk_v4i b) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX11__)
#if PK_W == 64
  c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(true, a, true, b, c, false);
#else
  c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
#endif
#else
  (void)c; (void)a; (void)b;
#endif
}

// MODE bit 0: A from LDS; bit 1: B from LDS; bit 2: the barrier; bit 3: loads in lanes 0-31 only; bit 4: B from gb.
template <int MODE>
__device__ __forceinline__ void pk_body(uint32_t steps, uint32_t *out, const int8_t *gb) {
  constexpr int MT = PK_MT, NT = PK_NT, ROWS = 16 * (MT + NT) * 2;
  __shared__ __attribute__((aligned(16))) int8_t sF[2][ROWS * PK_ROW];
  const int t = (int)threadIdx.x, lane = t % PK_W;
  const pk_v4i fill = {0x3F3F3F3F, 0x3F3F3F3F, 0x3F3F3F3F, 0x3F3F3F3F};
  for (int i = t; i < 2 * ROWS * 3; i += 256)
    *reinterpret_cast<pk_v4i *>(&sF[0][0] + (i / 3) * PK_ROW + 16 * (i % 3)) = fill;
  __syncthreads();
  pk_v4i a[MT], b[NT];
#pragma unroll
  for (int i = 0; i < MT; i++) { a[i] = fill; asm volatile("" : "+v"(a[i])); }
#pragma unroll
  for (int i = 0; i < NT; i++) { b[i] = fill; asm volatile("" : "+v"(b[i])); }
  pk_acc acc[MT][NT];
#pragma unroll
  for (int i = 0; i < MT; i++)
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
      for (int g = 0; g < 256 / PK_W; g++) acc[i][j][g] = 0;
  const int8_t *base = &sF[0][0] + (lane & 15) * PK_ROW;
  const int8_t *gbase = gb + (size_t)(blockIdx.x & 31u) * (NT * 16 * 64) + (lane & 15) * 64;
  for (uint32_t s = 0; s < steps; s++) {
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
      const int8_t *p = base + ((s & 1u) ? ROWS * PK_ROW : 0) + 16 * kk;
      const int8_t *g = gbase + ((s & 1u) ? 32 : 0) + 16 * kk;
      // The k-step's fragments: new registers each time, so a masked load leaves the other lanes undefined, not
      // selected from the last k-step's.
      pk_v4i a2[MT], b2[NT];
      auto loads = [&]() {
        if constexpr ((MODE & 1) != 0) {
#pragma unroll
          for (int i = 0; i < MT; i++) a2[i] = *reinterpret_cast<const pk_v4i *>(p + i * 16 * PK_ROW);
        }
        if constexpr ((MODE & 2) != 0) {
#pragma unroll
          for (int j = 0; j < NT; j++) b2[j] = *reinterpret_cast<const pk_v4i *>(p + (MT + j) * 16 * PK_ROW);
        }
        if constexpr ((MODE & 16) != 0) {
#pragma unroll
          for (int j = 0; j < NT; j++) b2[j] = *reinterpret_cast<const pk_v4i *>(g + j * 16 * 64);
        }
      };
      if constexpr ((MODE & 8) != 0) {
        if (lane < 32) {
          asm volatile("" ::: "memory");   // keeps the loads in the masked block
          loads();
        }
      } else {
        loads();
      }
#pragma unroll
      for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NT; j++)
          pk_mma(acc[i][j], (MODE & 1) ? a2[i] : a[i], (MODE & 18) ? b2[j] : b[j]);
    }
    if constexpr ((MODE & 4) != 0) {
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
    }
  }
  uint32_t x = 0u;
#pragma unroll
  for (int i = 0; i < MT; i++)
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
      for (int g = 0; g < 256 / PK_W; g++) x ^= (uint32_t)acc[i][j][g];
  if (t == 0 && blockIdx.x == 0) out[0] = x;
}

#define PK_KERNEL(name, mode)   __global__ __launch_bounds__(256) void pk_##name(uint32_t steps, uint32_t *out, const int8_t *gb) { pk_body<mode>(steps, out, gb); }
PK_KERNEL(regs, 0)
PK_KERNEL(lds, 3)
PK_KERNEL(ldsa, 1)
PK_KERNEL(ldsbar, 7)
PK_KERNEL(ldsagb, 17)
PK_KERNEL(gbbar, 21)
#if PK_W == 64
PK_KERNEL(ldsm, 11)
PK_KERNEL(ldsam, 9)
PK_KERNEL(ldsagbm, 25)
PK_KERNEL(gbbarm, 29)
#endif

int main(int argc, char **argv) {
  const double secs = argc > 1 ? atof(argv[1]) : 20.0;
  const char *only = argc > 2 && argv[2][0] ? argv[2] : nullptr;
  uint32_t *d = nullptr;
  if (hipMalloc(&d, 64) != hipSuccess) return 1;
  const uint32_t blocks = 4096u, steps = 1024u;
  // A step is two k-steps of PK_MT x PK_NT WMMAs a wave; a WMMA is 16 x 16 x 16 MACs.
  const double macs = (double)blocks * (256 / PK_W) * steps * 2.0 * PK_MT * PK_NT * 4096.0;
  // B's buffer: 32 block places of PK_NT tiles x 16 rows x 64 bytes, all the fill.
  const size_t gbBytes = (size_t)32 * PK_NT * 16 * 64 + 4096;
  int8_t *gb = nullptr;
  if (hipMalloc(&gb, gbBytes) != hipSuccess || hipMemset(gb, 0x3F, gbBytes) != hipSuccess) return 1;
#if PK_W == 64
  constexpr int NV = 10;
#else
  constexpr int NV = 6;
#endif
  const char *names[10] = {"regs", "lds", "ldsa", "ldsbar", "ldsagb", "gbbar", "ldsm", "ldsam", "ldsagbm", "gbbarm"};
  for (int v = 0; v < NV; v++) {
    if (only && strcmp(only, names[v]) != 0) continue;
    auto launch = [&]() {
#define PK_GO(i, name) if (v == i) hipLaunchKernelGGL(pk_##name, dim3(blocks), dim3(256), 0, 0, steps, d, (const int8_t *)gb);
      PK_GO(0, regs) PK_GO(1, lds) PK_GO(2, ldsa) PK_GO(3, ldsbar) PK_GO(4, ldsagb) PK_GO(5, gbbar)
#if PK_W == 64
      PK_GO(6, ldsm) PK_GO(7, ldsam) PK_GO(8, ldsagbm) PK_GO(9, gbbarm)
#endif
#undef PK_GO
    };
    launch();
    if (hipDeviceSynchronize() != hipSuccess) { printf("peak11 %s launch failed\n", names[v]); return 1; }
    const auto t0 = std::chrono::steady_clock::now();
    int n = 0;
    double el = 0.0;
    do {
      for (int i = 0; i < 4; i++) launch();
      hipDeviceSynchronize();
      n += 4;
      el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    } while (el < secs);
    printf("peak11 %-6s wave%d %dx%d tiles a wave: %.2f TMAC/s (%d launches, %.1f s)\n", names[v], PK_W, PK_MT, PK_NT,
           macs * n / el / 1e12, n, el);
    fflush(stdout);
  }
  return 0;
}
