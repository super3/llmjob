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
  constexpr bool kSmem = AS >= 2 || BS >= 2 || XK == 1 || XK == 5;
  for (uint32_t i = tid; kSmem && i < SMEM / 4u; i += 256u) {
    const uint32_t h = mix(i * 2654435761u + blockIdx.x);
    smem[i] = h & 0x7f7f7f7fu;  // bytes an int8 operand can hold
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
        if (AS >= 3 || BS >= 3) o = ldg2(offs + ((ob + sn * 32u) % OFFS_N));
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
        // B's next fragments
#pragma unroll
        for (int j = 0; j < 4; j++) {
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
#pragma unroll
        for (int n = 0; n < XN; n++) {
          uint32_t v[4];
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
          if (XK >= 1 && XK <= 3) sink ^= v[0] ^ v[1] ^ v[2] ^ v[3];  // two LOP3 a load
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
        if (BAR) asm volatile("bar.sync %0, 128;" ::"r"(1u + wr) : "memory");
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
  for (uint32_t i = 0; i < OFFS_N; i++) {
    const uint32_t ln = i & 31u;
    ho[i].x = (lcg(seed) % 224u) * 128u + (ln & 7u) * 16u;  // < 28 KB; + 3584 stays in the 32 KB table
    ho[i].y = (lcg(seed) % 96u) * 128u + (ln & 7u) * 16u;   // < 12 KB; + 3584 stays in the 16 KB table
  }
  CK(cudaMalloc(&dO, OFFS_N * sizeof(uint2)));
  CK(cudaMemcpy(dO, ho.data(), OFFS_N * sizeof(uint2), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dOut, (size_t)sms * 128 * 256 * 4));
  CK(cudaMalloc(&dC, (size_t)sms * 16 * 8));
  // A kernel that uses no shared memory gets none, and the most L1 (64 KB on Turing), as the release fold asks
  // for (PEARL_BD_A_DIRECT_CARVEOUT); the others get all 64 KB of shared and 32 KB of L1.
  const std::string kn = kp->name;
  const bool noSmem = kn == "ceil" || kn == "rel" || kn == "relA" || kn == "relB" || kn == "rel+alu32";
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
  printf("RES %s threads=%u pipe=%.5f launches=%d ms=%.1f regs=%d local=%d sms=%u ncg=%u l2=%.1fMB\n", kp->name, nthr,
         ps / (double)(pipes.size() - from), launches, ms_sum / launches, fa.numRegs, (int)fa.localSizeBytes, sms,
         ncg, p.l2CacheSize / 1048576.0, smem);
  return 0;
}
