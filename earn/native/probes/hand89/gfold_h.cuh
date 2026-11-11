// The gather fold: eight warps over a tile of 32 GP_MT rows by 256 columns, each warp building its own mma
// fragments with ldmatrix.trans used as a gather on static tables. No stage, no store, no barrier in the chunk
// loop, nothing drawn is shared. Branch goal/nv-kernel, 11 November 2026. The form is the maths review's
// (proved bit for bit on the CPU); this file is the kernel of probes/hand89/gprobe.cu, with the chunk loop in a
// function of its own that is never inlined, so that hand89/gfold.py can replace that function's code in the
// cubin with a loop written by hand. As compiled here it is the reference the hand loop is checked against.
//
// A warp is GP_MT m16 tiles (16 rows each) by eight n8 tiles (8 columns each); the eight warps are 2 row bands
// by 4 column slots. A lane's sums are rows g and g + 8 of each m16 tile and columns 2t and 2t + 1 of each n8
// tile (g = lane >> 2, t = lane & 3): that is its proof tile, 2 GP_MT rows by 16 columns, and its XOR needs
// nothing from any other lane.
//
// Layouts the host shares (gprobe.cu has a writer of each):
//   group table  128 entries of 16 bytes, entry p at 16 p: byte 2g is the dense factor plus 32 (0 .. 63) of the
//                group's element row e0(g) at rank index p, byte 2g + 1 of e1(g), g 0 .. 7. A's group of m16
//                tile x of row band i is rows 16 x + g and 16 x + 8 + g of the band; B's group of pair P of
//                column slot j is columns 16 P + g and 16 P + 8 + g of the slot.
//   shared       A's 2 GP_MT groups (band i, tile x at 2048 (GP_MT i + x)), then B's 16 (slot j, pair P at
//                2048 (2 GP_MT + 4 j + P)), then every thread's 16 transcript words.
//   offsets      [warp][k-step][lane 0 .. 31] of four 32-bit words {oA0, oA1, oB0, oB1}: 16 times the rank index
//                of the lane's k slot, for A and B and the k-step's two halves of 16 k, plus where the warp's
//                first group of that operand is in shared memory (A's 2048 GP_MT (warp >> 2), B's GP_TABB +
//                8192 (warp & 3)): a loaded word is an address as it stands. The lane's slot of a half
//                is 4 ((lane & 7) >> 1) + 2 ((lane >> 3) & 1) + (lane & 1); lanes 0-15 take the plus rank
//                index, lanes 16-31 the minus. Which k sits in which slot of its chunk is the host's choice
//                (any order inside a chunk gives the same sums at the chunk's end). One k-step more than the
//                tile has, at each warp's end. The host keeps the table's addresses from crossing a multiple
//                of 4 GiB, so the hand-written loop moves its pointer with a 32-bit add.
//
// One gather (a group, a k half): ldmatrix.x4.trans at [offset + group], X = M0 + fill - M2, Y = M1 + fill - M3,
// F0 = byte_perm(X, Y, 0x6420), F1 = byte_perm(X, Y, 0x7531): the two fragment words of the group's two element
// rows at the lane's four k. A k-step is 2 GP_MT gathers of A and 8 of B for 8 GP_MT mma.

#ifndef GP_FN
#error "gfold_h.cuh: define GP_FN before including it"
#endif
#ifndef GP_MT
#define GP_MT 6
#endif
#if GP_MT != 6 && GP_MT != 5
#error "GP_MT is 6 m16 tiles a warp (192 sums a thread) or 5 (160)"
#endif
// GP_DIRECT 1 is another fold in the same harness, the DIRECT fold: no draw and no tables. A' and B' are made
// beforehand (as the release makes them) and kept in global memory in the layout of the mma's own fragments, so
// a fragment is one 16-byte load a lane, straight into its registers: 2 GP_MT / 2 loads of A and 4 of B a warp a
// k-step and nothing else. No shared stage, no ldmatrix, no store, no warp waits for another: the SM's own
// cache is what the warps of a block share. tabA is then A' as [row group][k-step][row band][m16 tile][lane] of
// 16 bytes {row g at the lane's four k of the first half, row g + 8 the same, row g at the second half's, row
// g + 8}, and tabB is B' as [column block][k-step][column slot][pair][lane] of {column g of the pair's first
// n8 tile at the first half's four k, at the second half's, column g of its second tile the same two}. One
// k-step more than the tile has at each table's end. offs is not used.
#ifndef GP_DIRECT
#define GP_DIRECT 0
#endif
#define GP_NA (2u * GP_MT)                          // A's groups a block
#if GP_DIRECT
#define GP_TR 0u                                    // transcripts, [thread][16 words]
#else
#define GP_TABB (2048u * GP_NA)                     // B's 16 groups
#define GP_TR (GP_TABB + 2048u * 16u)               // transcripts, [thread][16 words]
#endif
#define GP_ST (GP_TR + 256u * 64u)                  // the probe's stamps
#define GP_PTR (GP_ST + 192u)                       // direct: where the tile's A' and B' begin (two pointers)
// GP_STAGE: bytes of shared memory behind the probe's stamps that the compiled kernel never touches. The
// rotation fold written by hand (hand89/rfold.py) keeps its stage of ready fragments there: A's at GP_STG as
// [row band][k-step of the chunk][m16 tile][lane] of 16 bytes (the mma's four A words of that lane), 3072 bytes
// a k-step of a band.
#ifndef GP_STAGE
#define GP_STAGE 0u
#endif
#define GP_STG (GP_ST + 256u)
#define GP_SMEM (GP_STG + GP_STAGE)
#define GP_THREADS 256u
#define GP_NI (8u * GP_MT)                          // mma a k-step
#ifndef GP_FILL
#define GP_FILL 63
#endif
#if GP_FILL != 63 && GP_FILL != 64
#error "GP_FILL is 63 or 64: under 63 a byte's subtract can borrow from the next"
#endif
#define GP_FILL4 ((uint32_t)GP_FILL * 0x01010101u)

#define GP_CAT2(a, b) a##b
#define GP_CAT(a, b) GP_CAT2(a, b)
#define GP_LOOPFN GP_CAT(GP_FN, _loop)

// The tile's fold. It takes nothing from the kernel but these two values.
static __device__ __noinline__ void GP_LOOPFN(const uint4 *__restrict__ offs, uint32_t chunks) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#if GP_DIRECT
  constexpr uint32_t MT = GP_MT, NB = 8u;
  const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
  const uint32_t trL = GP_TR + tid * 64u;
  (void)offs;
  unsigned long long a0, b0;
  asm volatile("ld.shared.u64 %0, [%1];" : "=l"(a0) : "r"(GP_PTR) : "memory");
  asm volatile("ld.shared.u64 %0, [%1];" : "=l"(b0) : "r"(GP_PTR + 8u) : "memory");
  const uint4 *pa = reinterpret_cast<const uint4 *>(a0) + (warp >> 2) * (32u * MT) + lane;
  const uint4 *pb = reinterpret_cast<const uint4 *>(b0) + (warp & 3u) * 128u + lane;
  int32_t acc[MT][NB][4];
#pragma unroll
  for (uint32_t mb = 0; mb < MT; mb++)
#pragma unroll
    for (uint32_t nb = 0; nb < NB; nb++)
#pragma unroll
      for (uint32_t i = 0; i < 4; i++) acc[mb][nb][i] = 0;
  // One k-step: B's four loads, then tile by tile A's load and the mma. `zero` is nothing: see below.
  auto kstep = [&](const uint32_t zero) {
    uint32_t bf[NB][2];
#pragma unroll
    for (uint32_t P = 0; P < 4u; P++) {
      const uint4 w = pb[32u * P + zero];
      bf[2u * P][0] = w.x;
      bf[2u * P][1] = w.y;
      bf[2u * P + 1u][0] = w.z;
      bf[2u * P + 1u][1] = w.w;
    }
#pragma unroll
    for (uint32_t mb = 0; mb < MT; mb++) {
      const uint4 w = pa[32u * mb + zero];
#pragma unroll
      for (uint32_t nb = 0; nb < NB; nb++)
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9}, {%0,%1,%2,%3};"
            : "+r"(acc[mb][nb][0]), "+r"(acc[mb][nb][1]), "+r"(acc[mb][nb][2]), "+r"(acc[mb][nb][3])
            : "r"(w.x), "r"(w.y), "r"(w.z), "r"(w.w), "r"(bf[nb][0]), "r"(bf[nb][1]));
    }
    pa += 64u * MT;
    pb += 512u;
  };
#else
  constexpr uint32_t MT = GP_MT, NB = 8u;
  const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
  const uint32_t trL = GP_TR + tid * 64u;
  auto ldmt = [](uint32_t r[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
  };
  auto gather = [&](uint32_t &f0, uint32_t &f1, uint32_t a) {
    uint32_t m[4];
    ldmt(m, a);
    const uint32_t x = m[0] + GP_FILL4 - m[2], y = m[1] + GP_FILL4 - m[3];
    f0 = __byte_perm(x, y, 0x6420u);
    f1 = __byte_perm(x, y, 0x7531u);
  };
  int32_t acc[MT][NB][4];
  uint32_t af[4], bf[NB][2];
#pragma unroll
  for (uint32_t mb = 0; mb < MT; mb++)
#pragma unroll
    for (uint32_t nb = 0; nb < NB; nb++)
#pragma unroll
      for (uint32_t i = 0; i < 4; i++) acc[mb][nb][i] = 0;
  const uint4 *po = offs + (size_t)warp * ((size_t)chunks * 128u + 32u) + lane;
  // One k-step: B's eight fragments, then tile by tile A's and the mma. `zero` is nothing: see below.
  auto kstep = [&](const uint32_t zero) {
    const uint4 o = *po;
    po += 32;
#pragma unroll
    for (uint32_t P = 0; P < 4u; P++) {
      gather(bf[2u * P][0], bf[2u * P + 1u][0], o.z + 2048u * P);
      gather(bf[2u * P][1], bf[2u * P + 1u][1], o.w + 2048u * P);
    }
#pragma unroll
    for (uint32_t mb = 0; mb < MT; mb++) {
      gather(af[0], af[1], o.x + zero + 2048u * mb);
      gather(af[2], af[3], o.y + zero + 2048u * mb);
#pragma unroll
      for (uint32_t nb = 0; nb < NB; nb++)
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9}, {%0,%1,%2,%3};"
            : "+r"(acc[mb][nb][0]), "+r"(acc[mb][nb][1]), "+r"(acc[mb][nb][2]), "+r"(acc[mb][nb][3])
            : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bf[nb][0]), "r"(bf[nb][1]));
    }
  };
#endif
  auto chunk_loop = [&](const uint32_t zero) {
    for (uint32_t c = 0; c < chunks; c++) {
      kstep(zero);
      kstep(zero);
      kstep(zero);
      kstep(zero);
      // The chunk's end: the lane's XOR of its own sums into its own word of the transcript.
      uint32_t x = 0;
#pragma unroll
      for (uint32_t mb = 0; mb < MT; mb++)
#pragma unroll
        for (uint32_t nb = 0; nb < NB; nb++)
          x ^= (uint32_t)acc[mb][nb][0] ^ (uint32_t)acc[mb][nb][1] ^ (uint32_t)acc[mb][nb][2]
               ^ (uint32_t)acc[mb][nb][3];
      uint32_t w;
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(w) : "r"(trL + (c & 15u) * 4u) : "memory");
      w = __funnelshift_l(w, w, 13u) ^ x;
      asm volatile("st.shared.u32 [%0], %1;" ::"r"(trL + (c & 15u) * 4u), "r"(w) : "memory");
    }
  };
  // Two copies of the same loop, which is the room the hand-written loop is given (it is longer than one).
  // The second adds a value the compiler cannot know is zero, or it would make one of the two.
  if (warp < 4u)
    chunk_loop(0u);
  else
    chunk_loop(chunks >> 24);
#endif
}

extern "C" __global__ __launch_bounds__(GP_THREADS, 1) void GP_FN(
    const uint4 *__restrict__ tabA, const uint4 *__restrict__ tabB, const uint4 *__restrict__ offs,
    uint32_t chunks, uint32_t row_groups, uint32_t tiles, unsigned long long *cyc, uint32_t *dump,
    uint32_t dump_tiles, uint32_t col_blocks) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  extern __shared__ __align__(128) uint32_t gp_smem[];
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(gp_smem);
  const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
  if (blockDim.x != GP_THREADS) return;
  for (uint32_t v = blockIdx.x; v < tiles; v += gridDim.x) {
    // The probe's long runs fold more tiles than it has tables for: the column blocks go round.
    const uint32_t rbg = v % row_groups, cbg = (v / row_groups) % col_blocks;
    if (lane == 0u) {
      const long long pt0 = clock64();
      asm volatile("st.shared.u64 [%0], %1;" ::"r"(sbase + GP_ST + warp * 24u), "l"(pt0) : "memory");
    }
    // The tile's tables, from the host's: A's 128 GP_NA entries, then B's 2048. And the thread's transcript.
    {
#if GP_DIRECT
      // Direct: nothing is copied. Where the tile's A' and B' begin goes into shared memory for the loop.
      if (tid == 0u) {
        const size_t ks1 = (size_t)chunks * 4u + 1u;
        const unsigned long long a0 = (unsigned long long)(tabA + (size_t)rbg * ks1 * (64u * GP_MT));
        const unsigned long long b0 = (unsigned long long)(tabB + (size_t)cbg * ks1 * 512u);
        asm volatile("st.shared.u64 [%0], %1;" ::"r"(sbase + GP_PTR), "l"(a0) : "memory");
        asm volatile("st.shared.u64 [%0], %1;" ::"r"(sbase + GP_PTR + 8u), "l"(b0) : "memory");
      }
#else
      const uint4 *const ta = tabA + (size_t)rbg * (128u * GP_NA);
      const uint4 *const tb = tabB + (size_t)cbg * 2048u;
      for (uint32_t i = tid; i < 128u * GP_NA + 2048u; i += GP_THREADS) {
        const uint4 w = i < 128u * GP_NA ? ta[i] : tb[i - 128u * GP_NA];
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};"
                     ::"r"(sbase + i * 16u), "r"(w.x), "r"(w.y), "r"(w.z), "r"(w.w)
                     : "memory");
      }
#endif
#pragma unroll
      for (uint32_t q = 0; q < 4u; q++)
        asm volatile("st.shared.v4.u32 [%0], {%1,%1,%1,%1};" ::"r"(sbase + GP_TR + tid * 64u + q * 16u), "r"(0u)
                     : "memory");
    }
    __syncthreads();
    if (lane == 0u) {
      const long long pt1 = clock64();
      asm volatile("st.shared.u64 [%0], %1;" ::"r"(sbase + GP_ST + 8u + warp * 24u), "l"(pt1) : "memory");
    }
    GP_LOOPFN(offs, chunks);
    if (lane == 0u) {
      const long long pt2 = clock64();
      asm volatile("st.shared.u64 [%0], %1;" ::"r"(sbase + GP_ST + 16u + warp * 24u), "l"(pt2) : "memory");
    }
    // The probe: the thread's transcript goes out as it is.
    if (v < dump_tiles) {
#pragma unroll
      for (uint32_t q = 0; q < 4u; q++) {
        uint32_t w[4];
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3])
                     : "r"(sbase + GP_TR + tid * 64u + q * 16u)
                     : "memory");
        uint32_t *const d = dump + ((size_t)v * GP_THREADS + tid) * 16u + q * 4u;
        d[0] = w[0];
        d[1] = w[1];
        d[2] = w[2];
        d[3] = w[3];
      }
    }
    if (lane == 0u) {
      // Cycles: the tile's prologue, its chunk loop, and its end, summed a warp; and the tiles.
      const long long pt3 = clock64();
      long long pt0, pt1, pt2;
      asm volatile("ld.shared.u64 %0, [%1];" : "=l"(pt0) : "r"(sbase + GP_ST + warp * 24u) : "memory");
      asm volatile("ld.shared.u64 %0, [%1];" : "=l"(pt1) : "r"(sbase + GP_ST + 8u + warp * 24u) : "memory");
      asm volatile("ld.shared.u64 %0, [%1];" : "=l"(pt2) : "r"(sbase + GP_ST + 16u + warp * 24u) : "memory");
      atomicAdd(cyc + (blockIdx.x * 8u + warp) * 4u + 0u, (unsigned long long)(pt1 - pt0));
      atomicAdd(cyc + (blockIdx.x * 8u + warp) * 4u + 1u, (unsigned long long)(pt2 - pt1));
      atomicAdd(cyc + (blockIdx.x * 8u + warp) * 4u + 2u, (unsigned long long)(pt3 - pt2));
      atomicAdd(cyc + (blockIdx.x * 8u + warp) * 4u + 3u, 1ull);
    }
    __syncthreads();
  }
#endif
}
