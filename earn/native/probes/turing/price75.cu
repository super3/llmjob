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

// goal-rtx20b: the timing skeleton of goal/algebra's form 3. With both fills 0, a chunk's update is
// dC = La H_c, H_c = M_c Lb^T (128 x n), M_c = S_A^T S_B. La (a row's 128 int6 noise entries) is the same in
// every chunk, so it is the IMMA's A operand held in registers all tile; H_c is built once a block a chunk into
// shared memory and read as the B operand. The block is 512 rows x 64 columns: warp w owns rows 64w..64w+63
// over all 64 columns, 8 x 8 IMMA.8816 a k16 (acc[mb][nb][2]: rows g + 8 mb, columns 8 nb + 2 t + j).
//
// H is built in two steps, one chunk apart, so a chunk needs one barrier:
//   scatter (chunk x - 2): each k of chunk x adds v = Lb[q0] - Lb[q1] to H[p0] and -v to H[p1], for the lane's
//     two columns in 16-bit halves (shared atomics into Hacc[i][lane], biased so nothing borrows across halves);
//     warp w takes the chunk's k 16w..16w+15, so every warp does the same work.
//   convert (chunk x - 1): warp w reads its k16 block of Hacc (i 16w..16w+15), sees the wraps (entries outside
//     int8), packs the low bytes into H's fragment order and sets Hacc back to the bias for chunk x + 1.
// TIMING ONLY: made-up values, and the wraps are only counted.
//   BUILD 0: H stays as initialised (the core: La in registers, H from shared, a barrier a chunk)
//   RO    1: a readout every chunk (an XOR of the 128 sums and one store), as the fold's
//   CORR  1: the wrap corrections every chunk, warp-uniform over the block's wrap list: each wrap (i, col, d)
//            adds 256 d La[r][i] to the 8 sums a lane holds in that column (an 8-byte load of La's bytes for
//            the lane's rows from a global table, a jump on the column)
// Shared: Hfrag 2 x 8 KB at 0, Hacc 2 x 16 KB at 16 KB, Lb 8 KB at 48 KB ([q 128][col 64] bytes, +32).
// Records in offs: 2 KB a chunk, 64 of them: words 2k and 2k + 1 for each k, the byte offsets of Lb rows q0 and q1
// (64 q, low and high half) and of Hacc rows p0 and p1 (128 p); words 256-287
// each rank index's degree (bytes), 400 the wrap count, 401.. the wraps (i | col << 8 | (d + 128) << 16).
#define F3_HF 0u
#define F3_HACC 16384u
#define F3_LB 49152u
#define F3_SMEM 57344u
#define F3_REC 512u
#define F3_NREC 64u
#define F3_BIAS 0x40004000u

__device__ __forceinline__ void imma1(int32_t &c0, int32_t &c1, uint32_t a, uint32_t b) {
  asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
               : "+r"(c0), "+r"(c1)
               : "r"(a), "r"(b));
}
__device__ __forceinline__ uint32_t lds1(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t lds16(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.u16 %0, [%1];" : "=r"(v) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ void atoms_add(uint32_t a, uint32_t v) {
  asm volatile("red.shared.add.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}

template <int K>
__device__ __forceinline__ void f3_fix(int32_t (&acc)[8][8][2], const int32_t (&d)[8]) {
#pragma unroll
  for (int mb = 0; mb < 8; mb++) acc[mb][K >> 1][K & 1] += d[mb];
}

// CORR 2: La[r][i] for the lane's 8 rows, from the La registers of the lane that holds them (row g's k-quad
// (i / 4) % 4 is in lane 4 g + that quad, register la[mb][i / 16]), so no memory: a jump on i / 16.
template <int KB>
__device__ __forceinline__ void f3_lav(const uint32_t (&la)[8][8], uint32_t src, uint32_t (&v)[8]) {
#pragma unroll
  for (int mb = 0; mb < 8; mb++) v[mb] = __shfl_sync(0xffffffffu, la[mb][KB], src);
}

template <int BUILD, int RO, int CORR>
__global__ __launch_bounds__(256, 1) void f3k(const uint8_t *__restrict__ A, const uint8_t *__restrict__ B,
                                              const uint2 *__restrict__ offs, const uint8_t *__restrict__ X,
                                              int32_t *__restrict__ out, long long *__restrict__ cyc,
                                              uint32_t tiles, uint32_t ncg, uint32_t nga) {
  extern __shared__ __align__(128) uint32_t smem[];
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
  const uint32_t *rec0 = reinterpret_cast<const uint32_t *>(offs);
  for (uint32_t i = tid; i < F3_SMEM / 4u; i += 256u) {
    const uint32_t h = mix(i * 2654435761u + blockIdx.x);
    smem[i] = i < F3_HACC / 4u ? (h & 0x3f3f3f3fu) - 0x20202020u : i < F3_LB / 4u ? F3_BIAS : (h & 0x3f3f3f3fu);
  }
  __shared__ uint32_t sRo[256];
  __syncthreads();
  uint32_t la[8][8];
#pragma unroll
  for (int mb = 0; mb < 8; mb++)
#pragma unroll
    for (int kb = 0; kb < 8; kb++) {
      // Loaded once, volatile, so ptxas holds La in registers and can't remake it in the loop.
      uint32_t v;
      asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(A + 65536u + (((size_t)(blockIdx.x & 1u) * 64u + mb * 8u + kb) * 256u + tid) * 4u) : "memory");
      la[mb][kb] = (v & 0x3f3f3f3fu) - 0x20202020u;
    }
  int32_t acc[8][8][2];
#pragma unroll
  for (int mb = 0; mb < 8; mb++)
#pragma unroll
    for (int nb = 0; nb < 8; nb++) acc[mb][nb][0] = acc[mb][nb][1] = 0;
  // The lane's two columns' places in H's fragment order: column c = 2 lane (+1) is nb = lane / 4,
  // n = c % 8, fragment lane 4 n + t (t: the quad of i), word nb % 4 of half nb / 4.
  const uint32_t nbw = lane >> 2, n0 = 2u * (lane & 3u);
  const uint32_t stoff = ((warp * 2u + (nbw >> 2)) * 32u + 4u * n0) * 16u + (nbw & 3u) * 4u;
  const uint32_t lbl = sb + F3_LB + 2u * lane;  // the lane's two columns of Lb row 0
  const uint8_t *latab = A + warp * 8192u + (lane >> 2) * 8u;  // [i][g][mb] bytes, the correction's La
  uint32_t wraps = 0u, sink = 0u;
  const long long t0 = clk64();
  uint32_t gch = 0u;
#pragma unroll 1
  for (uint32_t tile = 0; tile < tiles; tile++) {
#pragma unroll 1
    for (uint32_t ch = 0; ch < KS / 4u; ch++, gch++) {
      const uint32_t cur = sb + F3_HF + (gch & 1u) * 8192u, nxt = sb + F3_HF + ((gch + 1u) & 1u) * 8192u;
      const uint32_t accS = sb + F3_HACC + (gch & 1u) * 16384u;          // scatter chunk gch + 2 into this
      const uint32_t accL = accS + lane * 4u;
      const uint32_t accC = sb + F3_HACC + ((gch + 1u) & 1u) * 16384u;   // convert chunk gch + 1 from this
      const uint32_t *recS = rec0 + ((gch + 2u) % F3_NREC) * F3_REC;
      const uint32_t *recR = rec0 + ((gch + 3u) % F3_NREC) * F3_REC;     // the degrees Hacc is reset for
      // Loaded at the chunk's top so their latency hides under the IMMA: the lane's k word (lanes 0-15), and
      // the reset degrees of the warp's 16 rank indices (four bytes a word, lanes 0-3).
      const uint32_t kw = BUILD ? __ldg(recS + 32u * warp + lane) : 0u;
      const uint32_t dg = BUILD ? __ldg(recR + 256u + 4u * warp + (lane & 3u)) : 0u;
      // CORR 2: this chunk's wraps, one a lane (lane 31: the count; at most 31 are applied).
      const uint32_t *wrec = rec0 + (gch % F3_NREC) * F3_REC;
      const uint32_t wl = CORR == 2 ? __ldg(wrec + (lane == 31u ? 400u : 401u + lane)) : 0u;
      uint32_t hb[8];
      lds4(*reinterpret_cast<uint32_t(*)[4]>(&hb[0]), cur + ((0u * 2u + 0u) * 32u + lane) * 16u);
      lds4(*reinterpret_cast<uint32_t(*)[4]>(&hb[4]), cur + ((0u * 2u + 1u) * 32u + lane) * 16u);
#pragma unroll
      for (uint32_t kb = 0; kb < 8u; kb++) {
        if (BUILD && kb < 4u) {
          // Scatter: four of the warp's 16 k's.
#pragma unroll
          for (uint32_t u = 0; u < 4u; u++) {
            const uint32_t wq = __shfl_sync(0xffffffffu, kw, 2u * (4u * kb + u));
            const uint32_t wp = __shfl_sync(0xffffffffu, kw, 2u * (4u * kb + u) + 1u);
            const uint32_t x = lds16(lbl + (wq & 0xffffu)), y = lds16(lbl + (wq >> 16));
            // Bytes to 16-bit halves; v + 64 is in [1, 127] in each half, so neither add borrows.
            const uint32_t d = __byte_perm(x, 0u, 0x7170u) + 0x00400040u - __byte_perm(y, 0u, 0x7170u);
            atoms_add(accL + (wp & 0xffffu), d);
            atoms_add(accL + (wp >> 16), 0x00800080u - d);
          }
        } else if (BUILD) {
          // Convert: quad q = kb - 4 of the warp's k16 block (i = 16 warp + 4 q + u).
          const uint32_t q = kb - 4u;
          const uint32_t dw = __shfl_sync(0xffffffffu, dg, q);
          uint32_t hv[4];
#pragma unroll
          for (uint32_t u = 0; u < 4u; u++) {
            const uint32_t ad = accC + ((16u * warp + 4u * q + u) * 32u + lane) * 4u;
            const uint32_t a16 = lds1(ad);
            // Each of the entry's degree adds 64 to both halves; take it off, then see the wrap.
            const uint32_t deg = (dw >> (8u * u)) & 0xffu;
            asm volatile("st.shared.b32 [%0], %1;" ::"r"(ad), "r"(F3_BIAS - deg * 0x00400040u) : "memory");
            wraps += ((a16 + 0x00800080u - F3_BIAS) & 0xff00ff00u) != 0u;
            hv[u] = a16;
          }
          const uint32_t t01 = __byte_perm(hv[0], hv[1], 0x6240u), t23 = __byte_perm(hv[2], hv[3], 0x6240u);
          asm volatile("st.shared.b32 [%0], %1;" ::"r"(nxt + stoff + q * 16u), "r"(__byte_perm(t01, t23, 0x5410u)) : "memory");
          asm volatile("st.shared.b32 [%0], %1;" ::"r"(nxt + stoff + q * 16u + 64u), "r"(__byte_perm(t01, t23, 0x7632u)) : "memory");
        }
#pragma unroll
        for (uint32_t h = 0; h < 2u; h++) {
#pragma unroll
          for (uint32_t nb = 4u * h; nb < 4u * h + 4u; nb++)
#pragma unroll
            for (int mb = 0; mb < 8; mb++) imma1(acc[mb][nb][0], acc[mb][nb][1], la[mb][kb], hb[nb]);
          if (kb < 7u)
            lds4(*reinterpret_cast<uint32_t(*)[4]>(&hb[4u * h]), cur + (((kb + 1u) * 2u + h) * 32u + lane) * 16u);
        }
      }
      if (CORR == 2) {
        // This chunk's wraps, warp-uniform; the list is loaded a lane a wrap at the chunk's top (wl).
        const uint32_t nw = __shfl_sync(0xffffffffu, wl, 31u);
#pragma unroll 1
        for (uint32_t w = 0; w < nw && w < 31u; w++) {
          const uint32_t e = __shfl_sync(0xffffffffu, wl, w);
          const uint32_t i = e & 0xffu, col = (e >> 8) & 0xffu;
          const int32_t sc = ((int32_t)(e >> 16) - 128) * 256;
          const uint32_t src = (lane & ~3u) | ((i >> 2) & 3u), bsel = i & 3u;
          const uint32_t sel = bsel | ((bsel | 8u) << 4) | ((bsel | 8u) << 8) | ((bsel | 8u) << 12);
          uint32_t v[8];
          switch (i >> 4) {
            case 0: f3_lav<0>(la, src, v); break;
            case 1: f3_lav<1>(la, src, v); break;
            case 2: f3_lav<2>(la, src, v); break;
            case 3: f3_lav<3>(la, src, v); break;
            case 4: f3_lav<4>(la, src, v); break;
            case 5: f3_lav<5>(la, src, v); break;
            case 6: f3_lav<6>(la, src, v); break;
            default: f3_lav<7>(la, src, v); break;
          }
          if ((lane & 3u) != ((col & 7u) >> 1)) continue;
          int32_t d[8];
#pragma unroll
          for (int mb = 0; mb < 8; mb++) d[mb] = (int32_t)__byte_perm(v[mb], 0u, sel) * sc;
          switch (((col >> 3) << 1) | (col & 1u)) {
            case 0: f3_fix<0>(acc, d); break;
            case 1: f3_fix<1>(acc, d); break;
            case 2: f3_fix<2>(acc, d); break;
            case 3: f3_fix<3>(acc, d); break;
            case 4: f3_fix<4>(acc, d); break;
            case 5: f3_fix<5>(acc, d); break;
            case 6: f3_fix<6>(acc, d); break;
            case 7: f3_fix<7>(acc, d); break;
            case 8: f3_fix<8>(acc, d); break;
            case 9: f3_fix<9>(acc, d); break;
            case 10: f3_fix<10>(acc, d); break;
            case 11: f3_fix<11>(acc, d); break;
            case 12: f3_fix<12>(acc, d); break;
            case 13: f3_fix<13>(acc, d); break;
            case 14: f3_fix<14>(acc, d); break;
            default: f3_fix<15>(acc, d); break;
          }
        }
      }
      if (CORR == 1) {
        // This chunk's wraps, warp-uniform.
        const uint32_t *wr = rec0 + (gch % F3_NREC) * F3_REC;
        const uint32_t nw = __ldg(wr + 400u);
#pragma unroll 1
        for (uint32_t w = 0; w < nw; w++) {
          const uint32_t e = __ldg(wr + 401u + w);
          const uint32_t i = e & 0xffu, col = (e >> 8) & 0xffu;
          const int32_t sc = ((int32_t)(e >> 16) - 128) * 256;
          if ((lane & 3u) != ((col & 7u) >> 1)) continue;
          const uint2 lb = *reinterpret_cast<const uint2 *>(latab + i * 64u);
          int32_t d[8];
#pragma unroll
          for (int mb = 0; mb < 8; mb++)
            d[mb] = ((int32_t)((mb < 4 ? lb.x : lb.y) << (24 - 8 * (mb & 3))) >> 24) * sc;
          switch (((col >> 3) << 1) | (col & 1u)) {
            case 0: f3_fix<0>(acc, d); break;
            case 1: f3_fix<1>(acc, d); break;
            case 2: f3_fix<2>(acc, d); break;
            case 3: f3_fix<3>(acc, d); break;
            case 4: f3_fix<4>(acc, d); break;
            case 5: f3_fix<5>(acc, d); break;
            case 6: f3_fix<6>(acc, d); break;
            case 7: f3_fix<7>(acc, d); break;
            case 8: f3_fix<8>(acc, d); break;
            case 9: f3_fix<9>(acc, d); break;
            case 10: f3_fix<10>(acc, d); break;
            case 11: f3_fix<11>(acc, d); break;
            case 12: f3_fix<12>(acc, d); break;
            case 13: f3_fix<13>(acc, d); break;
            case 14: f3_fix<14>(acc, d); break;
            default: f3_fix<15>(acc, d); break;
          }
        }
      }
      if (RO) {
        uint32_t x = 0u;
#pragma unroll
        for (int mb = 0; mb < 8; mb++)
#pragma unroll
          for (int nb = 0; nb < 8; nb++) x ^= (uint32_t)acc[mb][nb][0] ^ (uint32_t)acc[mb][nb][1];
        sRo[tid] = x;
      }
      __syncthreads();
    }
  }
  const long long t1 = clk64();
  sink = wraps + (RO ? sRo[tid ^ 1u] : 0u);
  if (sink == 0x9e3779b9u) out[tid] = (int32_t)sink;
  if (lane == 0u) {
    cyc[(blockIdx.x * 8u + warp) * 2u] = t0;
    cyc[(blockIdx.x * 8u + warp) * 2u + 1u] = t1;
  }
#pragma unroll
  for (int mb = 0; mb < 8; mb++)
#pragma unroll
    for (int nb = 0; nb < 8; nb++)
#pragma unroll
      for (int j = 0; j < 2; j++)
        out[((size_t)blockIdx.x * 128u + mb * 16u + nb * 2u + j) * 256u + tid] = acc[mb][nb][j];
}

// goal-rtx20b: form 3, round 2 (goal/algebra 591259a1). H_c is built once per column block by a kernel of its
// own (not timed here: its IMMA are 128 / m of the main GEMM's, 0.4% at m 34816) and main blocks copy it from L2.
// The corrections read La from shared memory. Block 256 rows x 128 columns: warp w is row warp w >> 1 (64 rows)
// and column warp w & 1 (64 columns), 8 x 8 IMMA.8816 a k16 as f3k.
//   copy: each chunk a thread loads 64 bytes of the next chunk's H (4 LDG.128 from a 2 MB ring the L2 holds) and
//     stores them to the other H buffer (4 STS.128) after k16 block 5, so the loads' latency hides under the IMMA.
//   RO 1: the readout every chunk (XOR of the 128 sums, stored to global: shared is full).
//   CORR 1: rounds over the 16 accumulator columns (nb, j) of a lane's m8n8 sums. In a round, lanes of column
//     class t = lane % 4 apply one wrap whose column is 8 nb + 2 t + j: one LDS.64 of La[p][g][8 rows], then for
//     each of the 8 rows a PRMT (the byte, sign-extended) and an IMAD by 256 d (0 where the class has none).
//     The record's entries are loaded a lane each at the chunk's top and handed out by shuffle.
// Shared (64 KB, all of it): H 2 x 16 KB at 0 ([column warp][k16][half][lane][4 words]), La 32 KB at 32 KB
// ([row warp][p 128][g 8][8 rows]). Records in offs (2 KB a chunk, 64): for column warp c, word 128 c + 0 and
// + 1 the rounds of each (nb, j), 4 bits each; + 2 .. + 65 the entries, round-major, 4 a round (class t):
// p | (d + 128) << 8.
#define F3S_LA 32768u

// A correction slot: slot s = 2 kb + h follows half h of k16 block kb, and corrects accumulator column
// (nb, j) = (4 (1 - h) + kb / 2, kb % 2), whose IMMA are not the ones just issued.
template <uint32_t S>
__device__ __forceinline__ void f3s_slot(int32_t (&acc)[8][8][2], uint32_t mc0, uint32_t mc1, uint32_t en0,
                                         uint32_t en1, uint32_t en2, uint32_t lag, uint32_t t4, uint32_t &R) {
  constexpr uint32_t kb = S >> 1, h = S & 1u, nb = 4u * (1u - h) + (kb >> 1), j = kb & 1u;
  const uint32_t n = ((S < 8u ? mc0 : mc1) >> (4u * (S & 7u))) & 15u;
#pragma unroll 1
  for (uint32_t r = 0; r < n && R < 24u; r++, R++) {
    const uint32_t idx = 4u * R + t4;
    const uint32_t e0 = __shfl_sync(0xffffffffu, en0, idx & 31u), e1 = __shfl_sync(0xffffffffu, en1, idx & 31u);
    const uint32_t e2 = __shfl_sync(0xffffffffu, en2, idx & 31u);
    const uint32_t e = idx < 32u ? e0 : idx < 64u ? e1 : e2;
    const int32_t sc = ((int32_t)((e >> 8) & 0xffu) - 128) * 256;
    uint32_t v0, v1;
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];" : "=r"(v0), "=r"(v1) : "r"(lag + (e & 0xffu) * 64u) : "memory");
#pragma unroll
    for (int mb = 0; mb < 8; mb++) {
      const uint32_t b = (uint32_t)mb & 3u;
      acc[mb][nb][j] += (int32_t)__byte_perm(mb < 4 ? v0 : v1, 0u, b | ((8u | b) << 4) | ((8u | b) << 8) | ((8u | b) << 12)) * sc;
    }
  }
}

// IL 1: the correction slots between the IMMA groups instead of after them (they only add, so any order works
// before the readout); the wrap list then has to be loaded at the chunk's top.
template <int RO, int CORR, int IL>
__global__ __launch_bounds__(256, 1) void f3s(const uint8_t *__restrict__ A, const uint8_t *__restrict__ B,
                                              const uint2 *__restrict__ offs, const uint8_t *__restrict__ X,
                                              int32_t *__restrict__ out, long long *__restrict__ cyc,
                                              uint32_t tiles, uint32_t ncg, uint32_t nga) {
  extern __shared__ __align__(128) uint32_t smem[];
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
  const uint32_t wr = warp >> 1, wc = warp & 1u;
  const uint32_t *rec0 = reinterpret_cast<const uint32_t *>(offs);
  for (uint32_t i = tid; i < 65536u / 4u; i += 256u) {
    const uint32_t h = mix(i * 2654435761u + blockIdx.x);
    smem[i] = (h & 0x3f3f3f3fu) - 0x20202020u;
  }
  __syncthreads();
  uint32_t la[8][8];
#pragma unroll
  for (int mb = 0; mb < 8; mb++)
#pragma unroll
    for (int kb = 0; kb < 8; kb++) {
      uint32_t v;
      asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(A + 65536u + (((size_t)(blockIdx.x & 1u) * 64u + mb * 8u + kb) * 256u + tid) * 4u) : "memory");
      la[mb][kb] = (v & 0x3f3f3f3fu) - 0x20202020u;
    }
  int32_t acc[8][8][2];
#pragma unroll
  for (int mb = 0; mb < 8; mb++)
#pragma unroll
    for (int nb = 0; nb < 8; nb++) acc[mb][nb][0] = acc[mb][nb][1] = 0;
  const uint32_t lag = sb + F3S_LA + wr * 8192u + (lane >> 2) * 8u;  // + 64 p: lane g's 8 rows at p
  const uint32_t t4 = lane & 3u;
  uint32_t rx = 0u;
  const long long t0 = clk64();
  uint32_t gch = 0u;
#pragma unroll 1
  for (uint32_t tile = 0; tile < tiles; tile++) {
#pragma unroll 1
    for (uint32_t ch = 0; ch < KS / 4u; ch++, gch++) {
      const uint32_t cur = sb + (gch & 1u) * 16384u + wc * 8192u, nxt = sb + ((gch + 1u) & 1u) * 16384u;
      const uint8_t *hsrc = X + ((gch + 1u) % 128u) * 16384u + tid * 16u;
      uint32_t hc[2][4];  // the copy in two halves of 8 registers: loads at k16 blocks 0 and 3, stores at 2 and 5
      const uint32_t *wrec = rec0 + (gch % F3_NREC) * F3_REC + wc * 128u;
      uint32_t mc0 = 0u, mc1 = 0u, en0 = 0u, en1 = 0u, en2 = 0u;  // the wrap list
      if (CORR && IL) {
        mc0 = __ldg(wrec);
        mc1 = __ldg(wrec + 1u);
        en0 = __ldg(wrec + 2u + lane);
        en1 = __ldg(wrec + 34u + lane);
        en2 = __ldg(wrec + 66u + lane);
      }
      uint32_t R = 0u;
      uint32_t hb[8];
      lds4(*reinterpret_cast<uint32_t(*)[4]>(&hb[0]), cur + ((0u * 2u + 0u) * 32u + lane) * 16u);
      lds4(*reinterpret_cast<uint32_t(*)[4]>(&hb[4]), cur + ((0u * 2u + 1u) * 32u + lane) * 16u);
#pragma unroll
      for (uint32_t kb = 0; kb < 8u; kb++) {
        if (kb == 0u || kb == 3u) {
#pragma unroll
          for (int j = 0; j < 2; j++) ldg4(hc[j], hsrc + ((kb == 0u ? 0u : 2u) + j) * 4096u);
        }
        if (kb == 2u) {
#pragma unroll
          for (int j = 0; j < 2; j++) sts4(nxt + j * 4096u + tid * 16u, hc[j]);
        }
        if (kb == 5u) {
#pragma unroll
          for (int j = 0; j < 2; j++) sts4(nxt + (2u + j) * 4096u + tid * 16u, hc[j]);
          if (CORR && !IL) {
            mc0 = __ldg(wrec);
            mc1 = __ldg(wrec + 1u);
            en0 = __ldg(wrec + 2u + lane);
            en1 = __ldg(wrec + 34u + lane);
            en2 = __ldg(wrec + 66u + lane);
          }
        }
#pragma unroll
        for (uint32_t h = 0; h < 2u; h++) {
#pragma unroll
          for (uint32_t nb = 4u * h; nb < 4u * h + 4u; nb++)
#pragma unroll
            for (int mb = 0; mb < 8; mb++) imma1(acc[mb][nb][0], acc[mb][nb][1], la[mb][kb], hb[nb]);
          if (kb < 7u)
            lds4(*reinterpret_cast<uint32_t(*)[4]>(&hb[4u * h]), cur + (((kb + 1u) * 2u + h) * 32u + lane) * 16u);
          if (CORR && IL) {
            switch (2u * kb + h) {  // static after unrolling
#define F3S_SLOT(S) case S: f3s_slot<S>(acc, mc0, mc1, en0, en1, en2, lag, t4, R); break;
              F3S_SLOT(0) F3S_SLOT(1) F3S_SLOT(2) F3S_SLOT(3) F3S_SLOT(4) F3S_SLOT(5) F3S_SLOT(6) F3S_SLOT(7)
              F3S_SLOT(8) F3S_SLOT(9) F3S_SLOT(10) F3S_SLOT(11) F3S_SLOT(12) F3S_SLOT(13) F3S_SLOT(14) F3S_SLOT(15)
#undef F3S_SLOT
            }
          }
        }
      }
      if (CORR && !IL) {
#define F3S_SLOT(S) f3s_slot<S>(acc, mc0, mc1, en0, en1, en2, lag, t4, R);
        F3S_SLOT(0) F3S_SLOT(1) F3S_SLOT(2) F3S_SLOT(3) F3S_SLOT(4) F3S_SLOT(5) F3S_SLOT(6) F3S_SLOT(7)
        F3S_SLOT(8) F3S_SLOT(9) F3S_SLOT(10) F3S_SLOT(11) F3S_SLOT(12) F3S_SLOT(13) F3S_SLOT(14) F3S_SLOT(15)
#undef F3S_SLOT
      }
      if (RO) {
        uint32_t x = 0u;
#pragma unroll
        for (int mb = 0; mb < 8; mb++)
#pragma unroll
          for (int nb = 0; nb < 8; nb++) x ^= (uint32_t)acc[mb][nb][0] ^ (uint32_t)acc[mb][nb][1];
        rx += x;
        out[(size_t)blockIdx.x * 256u + tid] = (int32_t)x;
      }
      __syncthreads();
    }
  }
  const long long t1 = clk64();
  if (rx == 0x9e3779b9u) out[tid] = (int32_t)rx;
  if (lane == 0u) {
    cyc[(blockIdx.x * 8u + warp) * 2u] = t0;
    cyc[(blockIdx.x * 8u + warp) * 2u + 1u] = t1;
  }
#pragma unroll
  for (int mb = 0; mb < 8; mb++)
#pragma unroll
    for (int nb = 0; nb < 8; nb++)
#pragma unroll
      for (int j = 0; j < 2; j++)
        out[((size_t)blockIdx.x * 128u + mb * 16u + nb * 2u + j) * 256u + tid] = acc[mb][nb][j];
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
    // goal-rtx20b: goal/algebra's form 3 (see f3k), 512 x 64 blocks.
    {"f3c", f3k<0, 0, 0>, "form 3 core: La in registers, H from shared, a barrier a chunk; no build"},
    {"f3b", f3k<1, 0, 0>, "f3c and H built every chunk from the incidence lists"},
    {"f3r", f3k<1, 1, 0>, "f3b and a readout every chunk"},
    {"f3x", f3k<1, 1, 1>, "f3r and the wrap corrections at the measured wrap rate (La from a global table)"},
    {"f3y", f3k<1, 1, 2>, "f3r and the wrap corrections, La from the La registers by shuffle"},
    {"f3cy", f3k<0, 0, 2>, "f3c and f3y's corrections only (no build, no readout)"},
    // goal-rtx20b, form 3 round 2 (see f3s): H copied from L2 a chunk, corrections with La in shared.
    {"f3s0", f3s<0, 0, 0>, "form 3 round 2: La in registers, H copied from L2 to shared every chunk"},
    {"f3s1", f3s<1, 0, 0>, "f3s0 and the readout"},
    {"f3s2", f3s<1, 1, 0>, "f3s1 and the wrap corrections, La from shared, 4 column classes a round"},
    {"f3s3", f3s<0, 1, 0>, "f3s0 and the corrections, no readout"},
    {"f3s4", f3s<1, 1, 1>, "f3s2 with the correction slots between the IMMA groups"},
    {"f3s5", f3s<0, 1, 1>, "f3s4 without the readout"},
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
  if (!strncmp(kp->name, "f3s", 3)) {
    // Form 3 round 2's records (see f3s): the real wraps of H for a made-up Lb of 128 columns, per column warp
    // as rounds over the 16 accumulator columns (nb, j), one wrap a column class (t) a round.
    uint32_t *rw = reinterpret_cast<uint32_t *>(ho.data());
    std::vector<int> lb(128 * 128);
    for (auto &v : lb) v = (int)(lcg(seed) & 63u) - 32;
    double wsum = 0, rsum = 0, cut = 0;
    for (uint32_t c = 0; c < F3_NREC; c++) {
      uint32_t *r = rw + c * F3_REC;
      memset(r, 0, F3_REC * 4);
      std::vector<std::vector<uint32_t>> inc(128);
      for (uint32_t kk = 0; kk < 128; kk++) {
        const uint32_t u = lcg(seed) * 2u + (lcg(seed) & 1u), v = lcg(seed) * 2u + (lcg(seed) & 1u);
        const uint32_t p0 = u & 127u, p1 = p0 ^ (1u + (uint32_t)(((uint64_t)127u * u) >> 32));
        const uint32_t q0 = v & 127u, q1 = q0 ^ (1u + (uint32_t)(((uint64_t)127u * v) >> 32));
        inc[p0].push_back(q0 | (q1 << 16));
        inc[p1].push_back(q1 | (q0 << 16));
      }
      for (uint32_t wc = 0; wc < 2; wc++) {
        std::vector<uint32_t> lists[16][4];
        for (uint32_t i = 0; i < 128; i++)
          for (uint32_t lc = 0; lc < 64; lc++) {
            const uint32_t col = 64 * wc + lc;
            int hsum = 0;
            for (uint32_t e : inc[i]) hsum += lb[col * 128 + (e & 0xffffu)] - lb[col * 128 + (e >> 16)];
            const int wrapped = (int)(int8_t)(uint8_t)(hsum & 0xff);
            if (hsum == wrapped) continue;
            wsum += 1;
            const uint32_t nbj = (lc >> 3) * 2u + (lc & 1u), t = (lc & 7u) >> 1;
            lists[nbj][t].push_back(i | ((uint32_t)((hsum - wrapped) / 256 + 128) << 8));
          }
        uint32_t *w = r + wc * 128u, R = 0;
        for (uint32_t sl = 0; sl < 16; sl++) {
          const uint32_t kb = sl >> 1, hh = sl & 1u, nbj = 2u * (4u * (1u - hh) + (kb >> 1)) + (kb & 1u);
          uint32_t n = 0;
          for (uint32_t t = 0; t < 4; t++) n = std::max(n, (uint32_t)lists[nbj][t].size());
          n = std::min(n, 15u);
          uint32_t kept = 0;
          for (uint32_t k2 = 0; k2 < n; k2++) {
            if (R >= 24) { for (uint32_t t = 0; t < 4; t++) cut += k2 < lists[nbj][t].size(); continue; }
            for (uint32_t t = 0; t < 4; t++)
              w[2 + 4 * R + t] = k2 < lists[nbj][t].size() ? lists[nbj][t][k2] : (128u << 8);
            R++;
            kept++;
          }
          w[sl < 8 ? 0 : 1] |= kept << (4 * (sl & 7));
          rsum += n;
        }
      }
    }
    printf("F3S records %u, wraps a chunk a 64-column warp %.1f (%.3f%% of H), rounds %.1f, entries cut %.2f\n",
           F3_NREC, wsum / F3_NREC / 2, 100.0 * wsum / F3_NREC / 16384.0, rsum / F3_NREC / 2, cut / F3_NREC / 2);
  } else if (!strncmp(kp->name, "f3", 2)) {
    // Form 3's chunk records (see f3k), from noise.js's rule: p1 = p0 ^ (1 + mulhi(127, u)), the same for q.
    // The wraps are the real ones for a made-up Lb of 64 columns: H entries outside int8.
    uint32_t *rw = reinterpret_cast<uint32_t *>(ho.data());
    std::vector<int> lb(64 * 128);
    for (auto &v : lb) v = (int)(lcg(seed) & 63u) - 32;
    double wsum = 0;
    for (uint32_t c = 0; c < F3_NREC; c++) {
      uint32_t *r = rw + c * F3_REC;
      memset(r, 0, F3_REC * 4);
      std::vector<std::vector<uint32_t>> inc(128);
      for (uint32_t kk = 0; kk < 128; kk++) {
        const uint32_t u = lcg(seed) * 2u + (lcg(seed) & 1u), v = lcg(seed) * 2u + (lcg(seed) & 1u);
        const uint32_t p0 = u & 127u, p1 = p0 ^ (1u + (uint32_t)(((uint64_t)127u * u) >> 32));
        const uint32_t q0 = v & 127u, q1 = q0 ^ (1u + (uint32_t)(((uint64_t)127u * v) >> 32));
        inc[p0].push_back(q0 | (q1 << 16));
        inc[p1].push_back(q1 | (q0 << 16));
        r[2 * kk] = (q0 * 64u) | ((q1 * 64u) << 16);     // byte offsets of Lb rows q0, q1
        r[2 * kk + 1] = (p0 * 128u) | ((p1 * 128u) << 16);  // and of Hacc rows p0, p1
      }
      uint32_t nw = 0;
      for (uint32_t i = 0; i < 128; i++) {
        reinterpret_cast<uint8_t *>(r + 256)[i] = (uint8_t)inc[i].size();
        for (uint32_t col = 0; col < 64; col++) {
          int hsum = 0;
          for (uint32_t e : inc[i]) hsum += lb[col * 128 + (e & 0xffffu)] - lb[col * 128 + (e >> 16)];
          const int wrapped = (int)(int8_t)(uint8_t)(hsum & 0xff);
          if (hsum != wrapped && nw < 100)
            r[401 + nw++] = i | (col << 8) | ((uint32_t)((hsum - wrapped) / 256 + 128) << 16);
        }
      }
      r[400] = nw;
      wsum += nw;
    }
    printf("F3 records %u, wraps a chunk a block %.1f (%.3f%% of H)\n", F3_NREC, wsum / F3_NREC,
           100.0 * wsum / F3_NREC / 8192.0);
  }
  CK(cudaMalloc(&dO, OFFS_N * sizeof(uint2)));
  CK(cudaMemcpy(dO, ho.data(), OFFS_N * sizeof(uint2), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dOut, (size_t)sms * 128 * 256 * 4));
  CK(cudaMalloc(&dC, (size_t)sms * 16 * 8));
  // A kernel that uses no shared memory gets none, and the most L1 (64 KB on Turing), as the release fold asks
  // for (PEARL_BD_A_DIRECT_CARVEOUT); the others get all 64 KB of shared and 32 KB of L1.
  const std::string kn = kp->name;
  const bool noSmem = kn == "ceil" || kn == "rel" || kn == "relA" || kn == "relB" || kn == "rel+alu32";
  // Form 3 has 1 KB of static shared beside its 56 KB, within Turing's 64 KB a block.
  const uint32_t smem = noSmem ? 0u : !strncmp(kp->name, "f3s", 3) ? 65536u : !strncmp(kp->name, "f3", 2) ? F3_SMEM : SMEM;
  CK(cudaFuncSetAttribute(kp->f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
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
