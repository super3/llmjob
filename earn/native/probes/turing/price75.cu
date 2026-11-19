// price75 -- the price-list method (probes/hand89/price.py, RTX 4090) on Turing (sm_75): what each kind of
// step costs the tensor pipe and the clock when it is added to the B-direct fold's k-step, and what the
// rotation fold's operand stream would cost on these cards. TIMING ONLY: every stream folds made-up operands
// and its sums are meaningless. Our own code, for branch goal/nv-20.
//
// Every kernel is the release fold's geometry (src/pearl_fold_bd.cuh): 256 threads, eight 64x64 warp tiles
// over a 128x256 tile, one block an SM, a k-step 128 IMMA.8816 (k 32) on 128 accumulators a thread, 64
// k-steps a tile (k 2048). What differs is where each k-step's operands come from (A_SRC, B_SRC), loaded or
// made one k-step ahead, and what is added to each k-step (XK, XN):
//
//   source 0  registers, the same all tile (the denominator: no memory at all)
//   source 1  the release's: fragment-ordered A' / B' in global memory, ld.global.nc 16 bytes a lane, A shared
//             by the four warps of a row slot and B by the two of a column slot, both resident in the L2
//   source 2  ready fragments in shared memory, one ldmatrix.x4 a fragment (the least a shared-memory fold
//             can do: Ada's "skeleton")
//   source 3  (A only) the rotation fold's: a warp gathers A at one turn of a chunk's four (two ldmatrix and
//             eight ALU steps a 16-byte fragment) and stores it to a stage; at the other three turns it loads
//             a partner's from the stage (ld.shared 16 bytes). A barrier for the row slot's four warps at every
//             k-step's end when BAR is 1.
//   source 4  the gather fold's: every warp gathers every fragment every k-step (2 ldmatrix + 8 ALU each)
//   Gathers read their addresses from an offsets record in the L2: one 8-byte load a lane a k-step.
//
//   XK 1 ld.shared 16 bytes, 2 ld.global 16 bytes that L1 holds, 3 ld.global 16 bytes from the L2 (a 2 MB
//   ring), 4 ALU steps (PRMT and IADD3 pairs on the next k-step's B fragments), 5 st.shared 16 bytes; XN a
//   k-step a warp.
//
//   price75 <kernel> <seconds> [tiles a launch]     prints "RES <kernel> pipe=... launches=... ms=..."
//   price75 list                                     the kernels
// pipe is IMMA.8816 a clock an SM over its peak of 1 (1024 MACs), from the SM's own cycle counter; the clock
// and the watts come from nvidia-smi beside it (earn/tuning/salad/nv20probe.py).
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                    \
  do {                                                                           \
    cudaError_t e_ = (x);                                                        \
    if (e_ != cudaSuccess) {                                                     \
      printf("FAIL %s: %s\n", #x, cudaGetErrorString(e_));                       \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

#define KS 64u           // k-steps a tile (k 2048)
#define SMEM 65536u      // A table 16 KB, B table 32 KB, stage 16 KB
#define S_A 0u
#define S_B 16384u
#define S_STAGE 49152u
#define OFFS_N 65536u    // offsets ring, 8 bytes each (512 KB)
#define XRING (2u << 20) // the L2 ring of XK 3

__device__ __forceinline__ void imma4(int32_t &c0, int32_t &c1, int32_t &c2, int32_t &c3, uint32_t a0,
                                      uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%4}, {%8}, {%0,%1};\n"
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%6}, {%9}, {%0,%1};\n"
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%2,%3}, {%5}, {%8}, {%2,%3};\n"
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%2,%3}, {%7}, {%9}, {%2,%3};\n"
      : "+&r"(c0), "+&r"(c1), "+&r"(c2), "+&r"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a)
               : "memory");
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a)
               : "memory");
}
__device__ cudaTextureObject_t g_tex;  // XK 12: a 64 KB uint4 buffer the L1 holds
__device__ __forceinline__ void lds4(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a)
               : "memory");
}
__device__ __forceinline__ void sts4(uint32_t a, const uint32_t (&v)[4]) {
  asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3])
               : "memory");
}
__device__ __forceinline__ void ldg4(uint32_t (&r)[4], const void *p) {
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "l"(p)
               : "memory");
}
__device__ __forceinline__ void ldg4c(uint32_t (&r)[4], const void *p) {  // coherent: L1 caches it too
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "l"(p)
               : "memory");
}
__device__ __forceinline__ uint2 ldg2(const void *p) {
  uint2 v;
  asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "n"(0x6420));
  (void)s;
  return r;
}
__device__ __forceinline__ long long clk64() {
  long long c;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(c));
  return c;
}
__device__ __forceinline__ uint32_t mix(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
  return x;
}

// A gathered 16-byte fragment: two ldmatrix.x4 of table rows, then two ALU steps a register (the regroup's
// byte permute and the subtract of the noise, as gfold's draw does it).
__device__ __forceinline__ void gather(uint32_t (&f)[4], uint32_t a0, uint32_t a1) {
  uint32_t x[4], y[4];
  ldsm4(x, a0);
  ldsm4(y, a1);
#pragma unroll
  for (int i = 0; i < 4; i++) f[i] = prmt(x[i], y[i], 0) + 0x3f3f3f3fu - y[i];
}

template <int AS, int BS, int XK, int XN, int BAR>
__global__ __launch_bounds__(256, 1) void pk(const uint8_t *__restrict__ A, const uint8_t *__restrict__ B,
                                             const uint2 *__restrict__ offs, const uint8_t *__restrict__ X,
                                             int32_t *__restrict__ out, long long *__restrict__ cyc,
                                             uint32_t tiles, uint32_t ncg, uint32_t nga) {
  extern __shared__ __align__(128) uint32_t smem[];
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
  const uint32_t wr = warp >> 2, wc = warp & 3u;
  constexpr bool kSmem = AS >= 2 || BS >= 2 || XK == 1 || XK == 5 || (XK >= 6 && XK <= 14 && XK != 9 && XK != 12);
  for (uint32_t i = tid; kSmem && i < SMEM / 4u; i += 256u) {
    const uint32_t h = mix(i * 2654435761u + blockIdx.x);
    smem[i] = h & 0x7f7f7f7fu;  // bytes an int8 operand can hold
    if ((BS == 7 || BS == 8) && i >= (S_B - 64u) / 4u && i < S_B / 4u) smem[i] = 0u;  // the ring's k-step counts
  }
  __syncthreads();
  int32_t acc[4][8][4];
#pragma unroll
  for (int mb = 0; mb < 4; mb++)
#pragma unroll
    for (int nb = 0; nb < 8; nb++)
#pragma unroll
      for (int i = 0; i < 4; i++) acc[mb][nb][i] = 0;
  uint32_t af[4][4], bf[4][4];
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) {
      af[i][j] = mix(tid * 16u + i * 4u + j) & 0x7f7f7f7fu;
      bf[i][j] = mix(tid * 16u + i * 4u + j + 4096u) & 0x7f7f7f7fu;
    }
  const uint32_t zz = nga >> 16;  // 0, which ptxas cannot know: the added loads of one k-step and the next differ
  uint32_t sink = 0u;  // the added loads' results end here (ptxas drops a load nothing reads)
  const uint32_t ldsm_lane = lane * 16u;  // ready fragments: a matrix's rows in a row, no bank collisions
  const long long t0 = clk64();
  for (uint32_t tile = 0; tile < tiles; tile++) {
    const uint32_t rg = (blockIdx.x + tile) % nga, cg = (blockIdx.x * 7u + tile) % ncg;
    const uint8_t *Ab = A + ((size_t)(rg * 8u + wr * 4u) * KS) * 512u + lane * 16u;
    const uint8_t *Bb = B + ((size_t)(cg * 16u + wc * 4u) * KS) * 512u + lane * 16u;
    const uint32_t ob = ((blockIdx.x * 8u + warp) * 977u + tile * KS) * 32u + lane;  // a warp's records: 256 bytes in a row
#pragma unroll 1
    for (uint32_t ch = 0; ch < KS / 4u; ch++) {
#pragma unroll
      for (uint32_t t = 0; t < 4u; t++) {
        const uint32_t s = ch * 4u + t;
        const uint32_t sn = (s + 1u) & (KS - 1u), tn = (t + 1u) & 3u;  // the k-step made now, its turn
        uint32_t an[4][4], bn[4][4];
        uint2 o = make_uint2(0u, 0u);
        if (AS >= 3 || BS >= 3 || XK == 10) o = ldg2(offs + ((ob + sn * 32u) % OFFS_N));
        // A's next fragments
#pragma unroll
        for (int mb = 0; mb < 4; mb++) {
          if (AS == 0) {
#pragma unroll
            for (int i = 0; i < 4; i++) an[mb][i] = af[mb][i];
          } else if (AS == 1) {
            ldg4(an[mb], Ab + ((size_t)mb * KS + sn) * 512u);
          } else if (AS == 2) {
            ldsm4(an[mb], sb + S_A + (((wr * 4u + mb) * 4u + (sn & 3u)) * 512u) + ldsm_lane);
          } else if (AS == 4) {
            gather(an[mb], sb + S_A + o.y + mb * 1024u, sb + S_A + o.y + mb * 1024u + 512u);
          }
        }
        if (AS == 3) {
          // Turn 0 of a chunk: gather this warp's k-step and store it; turns 1-3: a partner's from the stage.
          const uint32_t stg = sb + S_STAGE + wr * 8192u;
          if (tn == 0u) {
#pragma unroll
            for (int mb = 0; mb < 4; mb++) {
              gather(an[mb], sb + S_A + o.y + mb * 1024u, sb + S_A + o.y + mb * 1024u + 512u);
              sts4(stg + ((wc * 4u + mb) * 512u) + lane * 16u, an[mb]);
            }
          } else {
#pragma unroll
            for (int mb = 0; mb < 4; mb++)
              lds4(an[mb], stg + ((((wc + tn) & 3u) * 4u + mb) * 512u) + lane * 16u);
          }
        }
        // goal-nv20v: B through a shared ring. The column slot's two warps share one scheduler (warp wc and wc + 4)
        // and the same B. Row slot 0's warp loads B from global and stores it into the pair's ring; row slot 1's
        // reads it from the ring. A slot is a k-step of the pair's B (2 KB); 6 slots a pair, 48 KB for the four
        // pairs, over the B table (the A table keeps its 16 KB, as the fold's transcripts would). Sources:
        //   5  stores word-major (16 st.shared 4 bytes), reads with 16 ld.shared 4 bytes, no flow control
        //   6  stores fragment order (4 st.shared 16 bytes), reads with 4 ld.shared 16 bytes, no flow control
        //   7  as 5, with flow control: each warp publishes its k-step count in shared memory after its stores or
        //      reads, and waits on its partner's (the writer at most 6 k-steps ahead, the reader behind the writer)
        //   8  as 6, with that flow control
        constexpr bool kRing = BS >= 5 && BS <= 8, kWord = BS == 5 || BS == 7, kFlow = BS == 7 || BS == 8;
        if (kRing) {
          const uint32_t g = tile * KS + s;  // k-steps this warp has made before this one, all tiles
          const uint32_t ring = sb + S_B + wc * 2048u + ((g + 1u) % 6u) * 8192u;
          const uint32_t ctr = sb + S_B - 64u + (wc * 2u + wr) * 4u;  // in the A table's last bytes: timing only
          if (wr == 0u) {
            if (kFlow) {
              uint32_t rc;
              do {
                asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(rc) : "r"(ctr + 4u) : "memory");
              } while (rc + 5u < g + 1u);  // the slot k-step g + 1 takes was read (k-step g - 5)
            }
#pragma unroll
            for (int j = 0; j < 4; j++) ldg4(bn[j], Bb + ((size_t)j * KS + sn) * 512u);
#pragma unroll
            for (int j = 0; j < 4; j++) {
              if (kWord) {
#pragma unroll
                for (int i = 0; i < 4; i++)
                  asm volatile("st.shared.b32 [%0], %1;" ::"r"(ring + j * 512u + i * 128u + lane * 4u), "r"(bn[j][i]) : "memory");
              } else {
                sts4(ring + j * 512u + lane * 16u, bn[j]);
              }
            }
            if (kFlow) {
              asm volatile("membar.cta;" ::: "memory");
              asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"(ctr), "r"(g + 2u) : "memory");
            }
          } else {
            if (kFlow) {
              uint32_t wc2;
              do {
                asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(wc2) : "r"(ctr - 4u) : "memory");
              } while (wc2 < g + 2u);  // k-step g + 1 is stored
              asm volatile("membar.cta;" ::: "memory");
            }
#pragma unroll
            for (int j = 0; j < 4; j++) {
              if (kWord) {
#pragma unroll
                for (int i = 0; i < 4; i++)
                  asm volatile("ld.shared.b32 %0, [%1];" : "=r"(bn[j][i]) : "r"(ring + j * 512u + i * 128u + lane * 4u) : "memory");
              } else {
                lds4(bn[j], ring + j * 512u + lane * 16u);
              }
            }
            if (kFlow) asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"(ctr), "r"(g + 2u) : "memory");
          }
        }
        // B's next fragments
#pragma unroll
        for (int j = 0; j < 4 && !kRing; j++) {
          if (BS == 0) {
#pragma unroll
            for (int i = 0; i < 4; i++) bn[j][i] = bf[j][i];
          } else if (BS == 1) {
            ldg4(bn[j], Bb + ((size_t)j * KS + sn) * 512u);
          } else if (BS == 2) {
            ldsm4(bn[j], sb + S_B + (((wc * 4u + j) * 4u + (sn & 3u)) * 512u) + ldsm_lane);
          } else {
            gather(bn[j], sb + S_B + o.x + j * 1024u, sb + S_B + o.x + j * 1024u + 512u);
          }
        }
        // The added steps
        uint32_t dx[4];  // XK 14: the even step's ldmatrix, subtracted from the odd one's
#pragma unroll
        for (int n = 0; n < XN; n++) {
          uint32_t v[4];
          // goal-nv20t: what each kind of shared-memory step costs Turing's load path, a step at a time.
          // 6 ld.shared 4 bytes a lane; 7 ld.shared 8 bytes; 8 ld.shared 16 bytes, every lane the same address
          // (broadcast); 9 shfl.idx from a lane that changes; 10 ldmatrix.x4.trans at a record's rows (bank
          // collisions as the offsets give them: argv[5] "rnd" for random rows); 11 ldmatrix.x4 on ready rows;
          // 12 a texture fetch of 16 bytes the L1 holds; 13 ldmatrix.x4.trans on ready rows; 14 a staged draw: pairs
          // of ldmatrix.x4.trans on ready rows, subtracted and regrouped (4 IADD3 and 4 PRMT a pair, as the
          // gather's) and stored to the stage (st.shared 16 bytes a pair).
          if (XK == 6) {
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v[0]) : "r"(sb + S_STAGE + s * zz + ((n & 15) * 128u + (warp & 1u) * 4096u) + lane * 4u) : "memory");
            v[1] = v[2] = v[3] = 0u;
          } else if (XK == 7) {
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];" : "=r"(v[0]), "=r"(v[1]) : "r"(sb + S_STAGE + s * zz + ((n & 15) * 256u + (warp & 1u) * 4096u) + lane * 8u) : "memory");
            v[2] = v[3] = 0u;
          } else if (XK == 8) {
            lds4(v, sb + S_STAGE + s * zz + (n & 15) * 16u + warp * 256u);
          } else if (XK == 9) {
            v[0] = __shfl_sync(0xffffffffu, bn[n & 3][(n >> 2) & 3], (lane + (uint32_t)n * 5u + s) & 31u);
            v[1] = v[2] = v[3] = 0u;
          } else if (XK == 10) {
            ldsm4t(v, sb + S_A + ((o.y + (uint32_t)n * 2048u) & 0x3fffu));
          } else if (XK == 11) {
            ldsm4(v, sb + S_STAGE + s * zz + ((n & 7) * 512u + (warp & 1u) * 4096u) + lane * 16u);
          } else if (XK == 12) {
            const uint4 t4 = tex1Dfetch<uint4>(g_tex, (int)((warp * 512u + (n & 15) * 32u + lane + s * zz) & 4095u));
            v[0] = t4.x; v[1] = t4.y; v[2] = t4.z; v[3] = t4.w;
          } else if (XK == 13) {
            ldsm4t(v, sb + S_STAGE + s * zz + ((n & 7) * 512u + (warp & 1u) * 4096u) + lane * 16u);
          } else if (XK == 14) {
            const uint32_t ra = sb + S_A + ((((warp * 8u + (uint32_t)n) * 512u) + s * zz) & 0x3fffu) + lane * 16u;
            if ((n & 1) == 0) {
              ldsm4t(dx, ra);
            } else {
              ldsm4t(v, ra);
              uint32_t w[4];
              const uint32_t x0 = dx[0] + 0x3f3f3f3fu - dx[2], y0 = dx[1] + 0x3f3f3f3fu - dx[3];
              const uint32_t x1 = v[0] + 0x3f3f3f3fu - v[2], y1 = v[1] + 0x3f3f3f3fu - v[3];
              w[0] = __byte_perm(x0, y0, 0x6420u);
              w[1] = __byte_perm(x0, y0, 0x7531u);
              w[2] = __byte_perm(x1, y1, 0x6420u);
              w[3] = __byte_perm(x1, y1, 0x7531u);
              sts4(sb + S_STAGE + ((warp * 4u + ((uint32_t)n >> 1)) & 31u) * 512u + lane * 16u, w);
            }
            v[0] = v[1] = v[2] = v[3] = 0u;
          }
          if (XK == 1) {
            lds4(v, sb + S_STAGE + s * zz + ((n & 15) * 512u + (warp & 1u) * 4096u) + lane * 16u);
          } else if (XK == 2) {
            ldg4c(v, X + (XRING + s * zz + warp * 4096u + (n & 7) * 512u) + lane * 16u);
          } else if (XK == 3) {
            ldg4(v, X + ((((blockIdx.x * 8u + warp) * 4099u + (ch * 4u + t) * XN + n) * 512u) % XRING) + lane * 16u);
          } else if (XK == 5) {
            v[0] = n; v[1] = s; v[2] = t; v[3] = lane;
            sts4(sb + S_STAGE + ((n & 7) * 512u + (warp & 1u) * 4096u) + lane * 16u, v);
          }
          if ((XK >= 1 && XK <= 3) || (XK >= 6 && XK <= 13)) sink ^= v[0] ^ v[1] ^ v[2] ^ v[3];  // two LOP3 a load
          if (XK == 4 && (n & 1) == 0) {
            uint32_t &r = bn[(n >> 3) & 3][(n >> 1) & 3];
            r = prmt(r, an[(n >> 3) & 3][(n >> 1) & 3], 0) + 0x01010101u;
          }
        }
        // This k-step's 128 IMMA on the fragments made a k-step ago
#pragma unroll
        for (int j = 0; j < 4; j++) {
#pragma unroll
          for (int mb = 0; mb < 4; mb++)
            imma4(acc[mb][2 * j][0], acc[mb][2 * j][1], acc[mb][2 * j][2], acc[mb][2 * j][3], af[mb][0],
                  af[mb][1], af[mb][2], af[mb][3], bf[j][0], bf[j][1]);
#pragma unroll
          for (int mb = 0; mb < 4; mb++)
            imma4(acc[mb][2 * j + 1][0], acc[mb][2 * j + 1][1], acc[mb][2 * j + 1][2], acc[mb][2 * j + 1][3],
                  af[mb][0], af[mb][1], af[mb][2], af[mb][3], bf[j][2], bf[j][3]);
        }
        if (BAR == 1) asm volatile("bar.sync %0, 128;" ::"r"(1u + wr) : "memory");
        if (BAR == 2) __syncthreads();  // goal-nv20t: the whole block, as a stage the eight warps fill needs
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
          for (int j = 0; j < 4; j++) {
            af[i][j] = an[i][j];
            bf[i][j] = bn[i][j];
          }
      }
    }
    if (AS == 0 && BS == 0) {
#pragma unroll
      for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) {
          af[i][j] += 0x01010101u;
          bf[i][j] ^= 0x00010001u << (j & 1);
        }
    }
  }
  const long long t1 = clk64();
  if (sink == 0x9e3779b9u) out[tid] = (int32_t)sink;
  if (lane == 0u) {
    cyc[(blockIdx.x * 8u + warp) * 2u] = t0;
    cyc[(blockIdx.x * 8u + warp) * 2u + 1u] = t1;
  }
#pragma unroll
  for (int mb = 0; mb < 4; mb++)
#pragma unroll
    for (int nb = 0; nb < 8; nb++)
#pragma unroll
      for (int i = 0; i < 4; i++)
        out[((size_t)blockIdx.x * 128u + mb * 32u + nb * 4u + i) * 256u + tid] = acc[mb][nb][i];
}

// goal/f-rtx20: the 256 x 128 block (TIMING ONLY, made-up operands as pk's). Eight warps, each a 128 x 32 warp
// tile. The 16-wide ("m") operand is 128 lines of one operand and the same 128 for all eight warps, so they
// share its lines in L1 and the block pulls 1/256 byte a MAC of it from the L2 (pk's block pulls 1/128 of B and
// 1/256 of A). The 8-wide ("n") operand is 32 lines of the other, a warp's own: 32 bytes a lane a k-step. This
// is the block SRBMiner 3.8.9's shares point to on Turing (earn/tuning/benchmark.md, goal/f-rtx20). A k-step is
// 128 IMMA.8816 on 128 sums a thread, as pk's, so pipe and count compare.
//   MS 0  m in registers                  1  from global, 8 loads of 16 bytes, every warp the same lines
//   NS 0  n in registers                  1  from global, 2 loads of 16 bytes, a warp's own lines
//      4  gathered from the warp's own 4 KB table in shared: 2 gathers a k-step (4 ldmatrix, 16 ALU steps)
template <int MS, int NS>
__global__ __launch_bounds__(256, 1) void pr(const uint8_t *__restrict__ A, const uint8_t *__restrict__ B,
                                             const uint2 *__restrict__ offs, const uint8_t *__restrict__ X,
                                             int32_t *__restrict__ out, long long *__restrict__ cyc,
                                             uint32_t tiles, uint32_t ncg, uint32_t nga) {
  extern __shared__ __align__(128) uint32_t smem[];
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
  (void)X;
  for (uint32_t i = tid; NS == 4 && i < SMEM / 4u; i += 256u)
    smem[i] = mix(i * 2654435761u + blockIdx.x) & 0x7f7f7f7fu;
  __syncthreads();
  int32_t acc[8][4][4];
#pragma unroll
  for (int mb = 0; mb < 8; mb++)
#pragma unroll
    for (int nb = 0; nb < 4; nb++)
#pragma unroll
      for (int i = 0; i < 4; i++) acc[mb][nb][i] = 0;
  uint32_t mf[8][4], nf[2][4];
#pragma unroll
  for (int i = 0; i < 8; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) mf[i][j] = mix(tid * 32u + i * 4u + j) & 0x7f7f7f7fu;
#pragma unroll
  for (int i = 0; i < 2; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) nf[i][j] = mix(tid * 32u + i * 4u + j + 8192u) & 0x7f7f7f7fu;
  const long long t0 = clk64();
  for (uint32_t tile = 0; tile < tiles; tile++) {
    const uint32_t rg = (blockIdx.x + tile) % nga, cg = (blockIdx.x * 7u + tile) % ncg;
    // m: eight 16-line blocks of B's buffer, the same for every warp of the block.
    const uint8_t *Mb = B + ((size_t)(cg * 16u + (tile & 1u) * 8u) * KS) * 512u + lane * 16u;
    // n: this warp's own block in two of A's row groups (2 x 16 bytes a lane a k-step).
    const uint8_t *N0 = A + ((size_t)(rg * 8u + warp) * KS) * 512u + lane * 16u;
    const uint8_t *N1 = A + ((size_t)(((rg + 1u) % nga) * 8u + warp) * KS) * 512u + lane * 16u;
    const uint32_t ob = ((blockIdx.x * 8u + warp) * 977u + tile * KS) * 32u + lane;
#pragma unroll 1
    for (uint32_t ch = 0; ch < KS / 4u; ch++) {
#pragma unroll
      for (uint32_t t = 0; t < 4u; t++) {
        const uint32_t s = ch * 4u + t;
        const uint32_t sn = (s + 1u) & (KS - 1u);
        uint32_t mn[8][4], nn[2][4];
#pragma unroll
        for (int mb = 0; mb < 8; mb++) {
          if (MS == 0) {
#pragma unroll
            for (int i = 0; i < 4; i++) mn[mb][i] = mf[mb][i];
          } else {
            ldg4(mn[mb], Mb + ((size_t)mb * KS + sn) * 512u);
          }
        }
        if (NS == 0) {
#pragma unroll
          for (int j = 0; j < 2; j++)
#pragma unroll
            for (int i = 0; i < 4; i++) nn[j][i] = nf[j][i];
        } else if (NS == 1) {
          ldg4(nn[0], N0 + (size_t)sn * 512u);
          ldg4(nn[1], N1 + (size_t)sn * 512u);
        } else {
          // The warp's table is 4 KB of B's 32 KB: 24 rows of 128 bytes to draw from, and room for the four
          // matrices' 256-byte steps.
          const uint2 o = ldg2(offs + ((ob + sn * 32u) % OFFS_N));
          const uint32_t base = sb + S_B + warp * 4096u + (o.y % 3072u);
          gather(nn[0], base, base + 512u);
          gather(nn[1], base + 256u, base + 768u);
        }
#pragma unroll
        for (int nb = 0; nb < 4; nb++)
#pragma unroll
          for (int mb = 0; mb < 8; mb++)
            imma4(acc[mb][nb][0], acc[mb][nb][1], acc[mb][nb][2], acc[mb][nb][3], mf[mb][0], mf[mb][1], mf[mb][2],
                  mf[mb][3], nf[nb >> 1][(nb & 1) * 2], nf[nb >> 1][(nb & 1) * 2 + 1]);
#pragma unroll
        for (int i = 0; i < 8; i++)
#pragma unroll
          for (int j = 0; j < 4; j++) mf[i][j] = mn[i][j];
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
          for (int j = 0; j < 4; j++) nf[i][j] = nn[i][j];
      }
    }
    if (MS == 0) {
#pragma unroll
      for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) mf[i][j] += 0x01010101u;
    }
    if (NS == 0) {
#pragma unroll
      for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) nf[i][j] ^= 0x00010001u << (j & 1);
    }
  }
  const long long t1 = clk64();
  if (lane == 0u) {
    cyc[(blockIdx.x * 8u + warp) * 2u] = t0;
    cyc[(blockIdx.x * 8u + warp) * 2u + 1u] = t1;
  }
#pragma unroll
  for (int mb = 0; mb < 8; mb++)
#pragma unroll
    for (int nb = 0; nb < 4; nb++)
#pragma unroll
      for (int i = 0; i < 4; i++)
        out[((size_t)blockIdx.x * 128u + mb * 16u + nb * 4u + i) * 256u + tid] = acc[mb][nb][i];
}

typedef void (*KFn)(const uint8_t *, const uint8_t *, const uint2 *, const uint8_t *, int32_t *, long long *,
                    uint32_t, uint32_t, uint32_t);
struct Kern {
  const char *name;
  KFn f;
  const char *what;
};
static const Kern kKerns[] = {
    {"ceil", pk<0, 0, 0, 0, 0>, "operands in registers, no memory"},
    {"rel", pk<1, 1, 0, 0, 0>, "the release's: A' and B' from global, 8 x 16-byte loads a k-step"},
    {"relA", pk<1, 0, 0, 0, 0>, "A' from global, B in registers"},
    {"relB", pk<0, 1, 0, 0, 0>, "B' from global, A in registers"},
    {"sm8", pk<2, 2, 0, 0, 0>, "ready fragments in shared: 8 ldmatrix a k-step"},
    {"rot", pk<3, 4, 0, 0, 1>, "rotation fold: B gathered, A gathered at one turn in four and shared, a barrier"},
    {"rotnb", pk<3, 4, 0, 0, 0>, "the same with no barrier"},
    {"rotrel", pk<3, 1, 0, 0, 1>, "A as the rotation fold makes it, B' from global as the release loads it"},
    {"gath", pk<4, 4, 0, 0, 0>, "gather fold: every fragment gathered by every warp"},
    {"gathA", pk<4, 1, 0, 0, 0>, "A gathered every k-step, B' from global"},
    // goal-nv20t round 2: B drawn on the chip by each warp, A as before, against the same stream with B loaded.
    {"gathB", pk<1, 4, 0, 0, 0>, "A' from global, B gathered every k-step (against rel)"},
    {"smAgB", pk<2, 4, 0, 0, 0>, "A ready in shared, B gathered every k-step (against smArB)"},
    {"smArB", pk<2, 1, 0, 0, 0>, "A ready in shared, B' from global as the release loads it"},
    {"rgAgB", pk<0, 4, 0, 0, 0>, "A in registers, B gathered every k-step (against relB)"},
    {"sm8+lds4", pk<2, 2, 1, 4, 0>, "sm8 and 4 ld.shared 16 bytes"},
    {"sm8+lds8", pk<2, 2, 1, 8, 0>, "sm8 and 8 ld.shared 16 bytes"},
    {"sm8+lds16", pk<2, 2, 1, 16, 0>, "sm8 and 16 ld.shared 16 bytes"},
    {"sm8+l1g8", pk<2, 2, 2, 8, 0>, "sm8 and 8 ld.global 16 bytes from L1"},
    {"sm8+l2g4", pk<2, 2, 3, 4, 0>, "sm8 and 4 ld.global 16 bytes from the L2"},
    {"sm8+l2g8", pk<2, 2, 3, 8, 0>, "sm8 and 8 ld.global 16 bytes from the L2"},
    {"sm8+alu32", pk<2, 2, 4, 32, 0>, "sm8 and 32 ALU steps"},
    {"sm8+alu64", pk<2, 2, 4, 64, 0>, "sm8 and 64 ALU steps"},
    {"sm8+sts4", pk<2, 2, 5, 4, 0>, "sm8 and 4 st.shared 16 bytes"},
    {"rel+alu32", pk<1, 1, 4, 32, 0>, "rel and 32 ALU steps"},
    // goal-nv20t: the load path's price of each kind of step (a step is one instruction a warp a k-step).
    {"sm8+lds32x16", pk<2, 2, 6, 16, 0>, "sm8 and 16 ld.shared 4 bytes"},
    {"sm8+lds32x32", pk<2, 2, 6, 32, 0>, "sm8 and 32 ld.shared 4 bytes"},
    {"sm8+lds64x16", pk<2, 2, 7, 16, 0>, "sm8 and 16 ld.shared 8 bytes"},
    {"sm8+ldsbc16", pk<2, 2, 8, 16, 0>, "sm8 and 16 ld.shared 16 bytes, one address a warp"},
    {"sm8+shfl16", pk<2, 2, 9, 16, 0>, "sm8 and 16 shfl.idx"},
    {"sm8+shfl32", pk<2, 2, 9, 32, 0>, "sm8 and 32 shfl.idx"},
    {"sm8+ldsm8", pk<2, 2, 11, 8, 0>, "sm8 and 8 ldmatrix.x4 on ready rows"},
    {"sm8+ldsmt8", pk<2, 2, 13, 8, 0>, "sm8 and 8 ldmatrix.x4.trans on ready rows"},
    {"sm8+ldsmo8", pk<2, 2, 10, 8, 0>, "sm8 and 8 ldmatrix.x4.trans at a record's rows (rnd: random rows)"},
    {"sm8+tex8", pk<2, 2, 12, 8, 0>, "sm8 and 8 texture fetches of 16 bytes"},
    {"sm8+tex16", pk<2, 2, 12, 16, 0>, "sm8 and 16 texture fetches of 16 bytes"},
    {"stg2", pk<2, 2, 14, 6, 2>, "staged draw of both: sm8, the warp's share of the block's draw (6 ldmatrix.trans, 3 st.shared), a block barrier"},
    {"stg2nb", pk<2, 2, 14, 6, 0>, "stg2 with no barrier"},
    {"stg1", pk<2, 2, 14, 4, 2>, "staged draw of B only (4 ldmatrix.trans, 2 st.shared), a block barrier"},
    // goal-nv20v: B through a shared ring, row slot 0's warp loading it for its partner (256 threads only).
    {"smAringW", pk<2, 5, 0, 0, 0>, "A ready in shared, B ring: 16 st.shared / 16 ld.shared 4 bytes, no flow control (against smArB)"},
    {"smAringQ", pk<2, 6, 0, 0, 0>, "A ready in shared, B ring: 4 st.shared / 4 ld.shared 16 bytes, no flow control"},
    {"smAringWF", pk<2, 7, 0, 0, 0>, "smAringW with flow control"},
    {"smAringQF", pk<2, 8, 0, 0, 0>, "smAringQ with flow control"},
    {"rgAringWF", pk<0, 7, 0, 0, 0>, "A in registers, B ring of 4-byte steps with flow control (against relB)"},
    {"relringWF", pk<1, 7, 0, 0, 0>, "A' from global, B ring of 4-byte steps with flow control (against rel)"},
    {"relringQF", pk<1, 8, 0, 0, 0>, "A' from global, B ring of 16-byte steps with flow control (against rel)"},
    // goal/f-rtx20: the 256 x 128 block, its 16-wide operand shared by all eight warps (kernel pr).
    {"r8c", pr<0, 0>, "256 x 128 block, operands in registers"},
    {"r8L", pr<1, 0>, "256 x 128 block: m from global, shared by eight warps; n in registers"},
    {"r8LL", pr<1, 1>, "256 x 128 block: m from global, shared by eight warps; n from global, a warp's own"},
    {"r8G", pr<1, 4>, "256 x 128 block: m from global, shared by eight warps; n gathered from the warp's own table"},
    {"r8cG", pr<0, 4>, "256 x 128 block: m in registers; n gathered from the warp's own table"},
};

static uint32_t lcg(uint64_t &s) {
  s = s * 6364136223846793005ull + 1442695040888963407ull;
  return (uint32_t)(s >> 33);
}

int main(int argc, char **argv) {
  if (argc < 2 || !strcmp(argv[1], "list")) {
    for (const Kern &k : kKerns) printf("%-10s %s\n", k.name, k.what);
    return argc < 2 ? 2 : 0;
  }
  const Kern *kp = nullptr;
  for (const Kern &k : kKerns)
    if (!strcmp(k.name, argv[1])) kp = &k;
  if (!kp) {
    printf("FAIL no kernel %s\n", argv[1]);
    return 2;
  }
  const double secs = argc > 2 ? atof(argv[2]) : 20.0;
  const uint32_t tiles = argc > 3 ? (uint32_t)atoi(argv[3]) : 1500u;
  // Threads a block (goal-nv20j): 256, two warps a scheduler as the fold runs, or 128, one warp a scheduler (row
  // slot 0 only), to see whether one warp alone can keep Turing's tensor pipe busy.
  const uint32_t nthr = argc > 4 ? (uint32_t)atoi(argv[4]) : 256u;
  if (nthr != 256u && nthr != 128u) {
    printf("FAIL threads must be 256 or 128, not %u\n", nthr);
    return 2;
  }
  const uint32_t nw = nthr / 32u;
  CK(cudaFree(0));
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  const uint32_t sms = (uint32_t)p.multiProcessorCount;
  // A: a band of nga row groups (128 rows x k each, 256 KB); B: ncg column groups (256 columns, 512 KB), both
  // within three quarters of the L2, as the release's band walk keeps them.
  const uint32_t nga = 4;
  uint32_t ncg = 1;
  while ((uint64_t)(ncg + 1) * 512u * 1024u + nga * 256u * 1024u <= (uint64_t)p.l2CacheSize * 3u / 4u) ncg++;
  const size_t aB = (size_t)nga * 8 * KS * 512, bB = (size_t)ncg * 16 * KS * 512;
  const size_t xB = XRING + 8 * 4096;
  std::vector<uint8_t> h(std::max(aB, std::max(bB, xB)));
  uint64_t seed = 12345;
  for (auto &c : h) c = (uint8_t)(lcg(seed) & 0x7f);
  uint8_t *dA, *dB, *dX;
  uint2 *dO;
  int32_t *dOut;
  long long *dC;
  CK(cudaMalloc(&dA, aB));
  CK(cudaMalloc(&dB, bB));
  CK(cudaMalloc(&dX, xB));
  CK(cudaMemcpy(dA, h.data(), aB, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, h.data(), bB, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dX, h.data(), xB, cudaMemcpyHostToDevice));
  // Offsets: x for B's table (32 KB), y for A's (16 KB). A lane's row sits at 16 x (lane % 8) modulo 128, so
  // the eight rows of a matrix fall in different banks (the order search's best case).
  std::vector<uint2> ho(OFFS_N);
  const bool rnd = argc > 5 && !strcmp(argv[5], "rnd");  // goal-nv20t: random rows, collisions as a real table's
  // goal-nv20t: "pair": the lane-spread rows, but in every other matrix two rows share a bank group (1.5 wavefronts
  // a matrix, about what the order search reaches with one gathered operand).
  const bool pair = argc > 5 && !strcmp(argv[5], "pair");
  for (uint32_t i = 0; i < OFFS_N; i++) {
    const uint32_t ln = i & 31u;
    ho[i].x = (lcg(seed) % 224u) * 128u + (ln & 7u) * 16u;  // < 28 KB; + 3584 stays in the 32 KB table
    ho[i].y = (lcg(seed) % 96u) * 128u + (ln & 7u) * 16u;   // < 12 KB; + 3584 stays in the 16 KB table
    if (pair && ((ln >> 3) & 1u) && (ln & 7u) == 7u) {
      ho[i].x -= 16u;
      ho[i].y -= 16u;
    }
    if (rnd) {
      ho[i].x = (lcg(seed) % 1792u) * 16u;
      ho[i].y = (lcg(seed) % 768u) * 16u;
    }
  }
  {
    // XK 12's texture: the first 64 KB of the X ring as uint4.
    cudaResourceDesc rd;
    memset(&rd, 0, sizeof(rd));
    rd.resType = cudaResourceTypeLinear;
    rd.res.linear.devPtr = dX;
    rd.res.linear.desc = cudaCreateChannelDesc<uint4>();
    rd.res.linear.sizeInBytes = 65536;
    cudaTextureDesc td;
    memset(&td, 0, sizeof(td));
    td.readMode = cudaReadModeElementType;
    cudaTextureObject_t tx = 0;
    CK(cudaCreateTextureObject(&tx, &rd, &td, nullptr));
    CK(cudaMemcpyToSymbol(g_tex, &tx, sizeof(tx)));
  }
  CK(cudaMalloc(&dO, OFFS_N * sizeof(uint2)));
  CK(cudaMemcpy(dO, ho.data(), OFFS_N * sizeof(uint2), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dOut, (size_t)sms * 128 * 256 * 4));
  CK(cudaMalloc(&dC, (size_t)sms * 16 * 8));
  // A kernel that uses no shared memory gets none, and the most L1 (64 KB on Turing), as the release fold asks
  // for (PEARL_BD_A_DIRECT_CARVEOUT); the others get all 64 KB of shared and 32 KB of L1.
  const std::string kn = kp->name;
  const bool noSmem = kn == "ceil" || kn == "rel" || kn == "relA" || kn == "relB" || kn == "rel+alu32"
                     || kn == "r8c" || kn == "r8L" || kn == "r8LL";
  const uint32_t smem = noSmem ? 0u : SMEM;
  CK(cudaFuncSetAttribute(kp->f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM));
  CK(cudaFuncSetAttribute(kp->f, cudaFuncAttributePreferredSharedMemoryCarveout, noSmem ? 0 : 100));
  cudaFuncAttributes fa;
  CK(cudaFuncGetAttributes(&fa, kp->f));
  std::vector<long long> hc((size_t)sms * 16);
  std::vector<double> pipes;
  const auto T0 = std::chrono::steady_clock::now();
  double ms_sum = 0;
  int launches = 0;
  for (;;) {
    const auto a = std::chrono::steady_clock::now();
    kp->f<<<sms, nthr, smem>>>(dA, dB, dO, dX, dOut, dC, tiles, ncg, nga);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    const auto b = std::chrono::steady_clock::now();
    CK(cudaMemcpy(hc.data(), dC, hc.size() * 8, cudaMemcpyDeviceToHost));
    double pipe = 0;
    for (uint32_t blk = 0; blk < sms; blk++) {
      long long lo = hc[blk * 16], hi = hc[blk * 16 + 1];
      for (uint32_t w = 1; w < nw; w++) {
        lo = std::min(lo, hc[blk * 16 + w * 2]);
        hi = std::max(hi, hc[blk * 16 + w * 2 + 1]);
      }
      pipe += (double)tiles * KS * 128.0 * (double)nw / (double)(hi - lo);
    }
    pipes.push_back(pipe / sms);
    ms_sum += std::chrono::duration<double, std::milli>(b - a).count();
    launches++;
    if (std::chrono::duration<double>(b - T0).count() >= secs) break;
  }
  // The mean of the launches after the first third (the card's clock and heat settle).
  const size_t from = pipes.size() / 3;
  double ps = 0;
  for (size_t i = from; i < pipes.size(); i++) ps += pipes[i];
  printf("RES %s%s threads=%u pipe=%.5f launches=%d ms=%.1f regs=%d local=%d sms=%u ncg=%u l2=%.1fMB\n", kp->name, rnd ? "/rnd" : pair ? "/pair" : "", nthr,
         ps / (double)(pipes.size() - from), launches, ms_sum / launches, fa.numRegs, (int)fa.localSizeBytes, sms,
         ncg, p.l2CacheSize / 1048576.0, smem);
  return 0;
}
