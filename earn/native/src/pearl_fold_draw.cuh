// The draw fold: the tall fold's 192x256 tile, eight warps and readout, with both operands
// drawn on the SM instead of copied from memory. Branch goal/nv-kernel, 11 November 2026.
//
// A' and B' are never stored. A tile keeps, in shared memory, the dense noise factors of its
// 192 rows and 256 columns (transposed: one 16-byte word holds sixteen rows at one rank
// index), and two k-step stages of the noised operands. Each warp draws the next k-step's
// stage while it folds this one: a "unit" is sixteen rows by one k-pair, four 16-byte table
// reads, eight subtracts, eight byte selects and two 16-byte stores. The stage is laid out so
// ldmatrix.trans hands every lane its k-packed fragment word.
//
// Included by pearl_kernel.cu (the miner's kernel) and by probes/drawprobe.cu (the same body
// with cycle stamps and a transcript dump in place of the hash). Before including, define
//   PD_FN      the kernel's name
//   PD_PROBE   1 for the probe's build, 0 for the miner's
// and, for the miner's build, everything the tall fold's hash uses (pearl_kernel.cu has it).
//
// The chunk loop is written out once a warp (eight copies), so everything that depends on the
// warp is a constant in its copy: a table read's address is its offset plus a constant, with
// no add. That leaves a thread four registers that live through the loop beside its sums and
// fragments, and ptxas the room to weave the draw between the mma by itself (at 255 registers
// it runs every unit as one clump instead).
//
// Layouts the host shares (see the probe for a writer of each):
//   table    [quad q][e][rank index p] of 16 bytes: a quad is 32 rows (or columns); word (q, e, p)
//            holds the dense factor plus 32 of rows 32q + 8e + 0..7 (bytes 0-7) and
//            32q + 8e + 16 + 0..7 (bytes 8-15) at rank index p.
//   offsets  [k-step][B, A][lane 0..31] of four 32-bit words: for the lane's k-pair, its first
//            k's plus and minus rank index, then its second k's, each times 16, plus 2048 e.
//            B's lane L draws e = L >> 4, k-pair L & 15; A's lane L draws e = (L >> 3) & 1,
//            k-pair 8 (L >> 4) + (L & 7).
//   stage    16 bytes for a group of 8 rows and a k-pair: byte 2c is row c at the pair's
//            first k and byte 2c + 1 at its second. A's group G, k-pair j is at
//            PD_AOFF + 256 G + 16 j. B's is at 512 (G >> 1) + 256 (j >> 3) + 128 (G & 1)
//            + 16 (j & 7): half-major inside a pair of groups, so that one lane register
//            addresses both operands' ldmatrix rows (the mma wants a tile's two k halves in
//            adjacent registers for B, and its two row groups for A).

#ifndef PD_FN
#error "pearl_fold_draw.cuh: define PD_FN and PD_PROBE before including it"
#endif

// Shared memory, bytes from the block's base.
#define PD_TABB 0u            // B's table: 8 quads
#define PD_TABA 32768u        // A's table: 6 quads
#define PD_STG 57344u         // two stages, PD_STAGE bytes each
#define PD_STAGE 14336u
#define PD_AOFF 8192u         // A's groups in a stage, after B's
#define PD_TR 86016u          // transcripts, [region][16 words]
#define PD_CNT 98304u         // the stage counter, then the probe's stamps
#define PD_SCR 98560u         // a 16-byte row a lane (512 bytes), for the holds
#define PD_SMEM 99072u
#define PD_THREADS 256u

// The pieces' places in a k-step, as the index of the mma they follow (0 .. 47).
#ifndef PD_ADD_AT
#define PD_ADD_AT 33          // the counter add: after the draw's last store and A's last load
#endif
#ifndef PD_WAIT_AT
#define PD_WAIT_AT 39         // the wait for the next stage; its first loads follow
#endif
// A half unit's four steps, as the mma of its m16 tile (0 .. 7) each follows: table reads,
// subtracts, byte selects, store.
#ifndef PD_L
#define PD_L 0u
#endif
#ifndef PD_A
#define PD_A 3u
#endif
#ifndef PD_P
#define PD_P 5u
#endif
#ifndef PD_S
#define PD_S 6u
#endif
// The readout's runs: the mma each of a run's four parts follows, then its end. Region pair 0
// and pair 1 in a chunk's last k-step, pair 2 (of the chunk before) in its first.
#ifndef PD_RO0
#define PD_RO0 {32u, 33u, 35u, 36u, 37u}
#endif
#ifndef PD_RO1
#define PD_RO1 {40u, 42u, 44u, 46u, 47u}
#endif
#ifndef PD_RO2
#define PD_RO2 {0u, 2u, 4u, 6u, 7u}
#endif
// 1: a half unit's subtracts wait for mma PD_CP of their tile (see the chunk loop).
#ifndef PD_CPIN
#define PD_CPIN 1
#endif
#ifndef PD_CP
#define PD_CP 1u
#endif
// 1: the next k-step's B fragments wait for the mma that free their registers (the holds).
#ifndef PD_HOLDS
#define PD_HOLDS 1
#endif
// 1: every warp draws a unit of A every k-step (warps 6 and 7 draw warps 0 and 1's again).
// 0: A's twelve units go to six warps a k-step; scheduler s's two warps sit out k-step s.
#ifndef PD_ADUP
#define PD_ADUP 0
#endif
// Timing only: pieces off. PD_NO_DRAW folds the stage the prologue drew, over and over;
// PD_NO_READOUT reads out at the tile's end only; PD_NO_SYNC drops the add and the wait.
#ifndef PD_NO_DRAW
#define PD_NO_DRAW 0
#endif
#ifndef PD_NO_READOUT
#define PD_NO_READOUT 0
#endif
#ifndef PD_NO_SYNC
#define PD_NO_SYNC 0
#endif
// 1: the readout's shared XOR behind an always-true predicate, as the tall fold has it.
#ifndef PD_RED_PRED
#define PD_RED_PRED 0
#endif
// Probe only: count the waits that found the next stage not whole (adds code to the spin's path).
#ifndef PD_SPINCOUNT
#define PD_SPINCOUNT 0
#endif
// A fence after these mma of every k-step, a bit an mma: ptxas does not move an instruction
// across one (see PD_SLOT). Each costs register moves where the blocks join; measured, not free.
#ifndef PD_FENCE_MASK
#define PD_FENCE_MASK 0ull
#endif
// 1: the draw's stores address through a register of their own (see dstL).
#ifndef PD_DST_REG
#define PD_DST_REG 1
#endif

extern "C" __global__ __launch_bounds__(PD_THREADS, 1) void PD_FN(
    const uint4 *__restrict__ tabA, const uint4 *__restrict__ tabB, const uint4 *__restrict__ offs,
    const int8_t *__restrict__ stamp, uint32_t stamp_bytes, uint32_t chunks, uint32_t col_off,
    uint32_t rows_valid, uint32_t col_groups, uint32_t tiles
#if PD_PROBE
    , unsigned long long *cyc, uint32_t *dump, uint32_t dump_tiles
#else
    , const PearlTranscriptTest test, const PearlHitList hits
#endif
    ) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr uint32_t MT = 6u, NB = 8u, RPL = 3u, WARP_REGIONS = 24u;
  if (blockDim.x != PD_THREADS) return;
  extern __shared__ __align__(128) uint32_t pd_smem[];
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(pd_smem);
  const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;

  const uint32_t row_groups = (rows_valid + 11u) / 12u;
  (void)col_groups;

  // ---- lane constants: the four registers a thread keeps through the chunk loop ----
  // fragL: this lane's 16-byte row in a 512-byte block of a stage (two groups by two k
  // halves), less everything the warp and the k-step fix. ldmatrix.trans reads it and the
  // draw writes it. For A bit 3 of the lane is the group and bit 4 the half; for B bit 3 is
  // the half and bit 4 the group (see "stage" above).
  uint32_t fragL = sbase + PD_STG + ((lane >> 3) & 1u) * 256u + (lane >> 4) * 128u + (lane & 7u) * 16u;
  // trL: the lane's region words, less the warp's part (the tall fold's layout).
  uint32_t trL = sbase + PD_TR + ((lane & 3u) * (2u * RPL) + ((lane >> 4) & 1u)) * 64u;
  // dstL: the same address, for the draw's stores. A second register on purpose (PD_DST_REG):
  // with one, ptxas sees that a store and a fragment load never overlap and moves every
  // ldmatrix of the k-step to its top, into registers of their own, and the fold spills.
  // With two it must keep a load after the stores before it, which holds each where it is put.
  // (Plus chunks >> 16, which is zero for every k the fold takes: an empty asm alone leaves
  // ptxas a copy it folds away.)
#if PD_DST_REG
  uint32_t dstL = fragL + (chunks >> 16);
  asm volatile("" : "+r"(fragL), "+r"(trL), "+r"(dstL));
  // A zero ptxas cannot see is one (the same term as dstL's), for PD_CPIN.
  const uint32_t zero = chunks >> 16;
#else
  asm volatile("" : "+r"(fragL), "+r"(trL));
  const uint32_t dstL = fragL;
#endif
  const uint32_t sCnt = sbase + PD_CNT;

  // ---- small pieces ----
  auto ld4 = [](uint32_t r[4], uint32_t a) {
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
  };
  auto st4 = [](uint32_t a, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3)
                 : "memory");
  };
  auto ldg4 = [](uint32_t r[4], const uint4 *p) {
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "l"(p));
  };
  auto ldmt = [](uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory");
  };
  auto xor3 = [](uint32_t a, uint32_t b, uint32_t c) -> uint32_t {
    uint32_t d;
    asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
  };

  // A unit: sixteen rows of one k-pair. o holds its four table offsets; x and y the sixteen
  // noised bytes of each k. The table's bytes are the dense factor plus 32, so
  // 64 + plus - minus has no borrow between bytes: the fill is 64 and the sum is the noised
  // operand, 1 .. 127. tab is the table's address less the offset: a constant in the chunk loop.
  struct Unit {
    uint32_t o[4], x[4], y[4];
  };
  auto u_half = [&](uint32_t out[4], uint32_t tab, uint32_t op, uint32_t om) {
    uint32_t p[4], q[4];
    ld4(p, tab + op);
    ld4(q, tab + om);
#pragma unroll
    for (int i = 0; i < 4; i++) out[i] = p[i] + 0x40404040u - q[i];
  };
  // Half a unit, as the chunk loop draws it: rows 8 hi .. 8 hi + 7 of the sixteen, from
  // 64-bit reads, in one store.
  auto ld2 = [](uint32_t r[2], uint32_t a) {
    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(a) : "memory");
  };
  struct Half {
    uint32_t p0[2], m0[2], p1[2], m1[2];
  };
  auto u_load_half = [&](Half &f, const uint32_t o[4], uint32_t tab, uint32_t hi) {
    ld2(f.p0, tab + 8u * hi + o[0]);
    ld2(f.m0, tab + 8u * hi + o[1]);
    ld2(f.p1, tab + 8u * hi + o[2]);
    ld2(f.m1, tab + 8u * hi + o[3]);
  };
  auto u_rest_half = [&](const Half &f, uint32_t dst, uint32_t hi) {
    const uint32_t x0 = f.p0[0] + 0x40404040u - f.m0[0], x1 = f.p0[1] + 0x40404040u - f.m0[1];
    const uint32_t y0 = f.p1[0] + 0x40404040u - f.m1[0], y1 = f.p1[1] + 0x40404040u - f.m1[1];
    st4(dst + 512u * hi, __byte_perm(x0, y0, 0x5140u), __byte_perm(x0, y0, 0x7362u),
        __byte_perm(x1, y1, 0x5140u), __byte_perm(x1, y1, 0x7362u));
  };
  auto u_store = [&](const Unit &u, uint32_t dst) {
    st4(dst, __byte_perm(u.x[0], u.y[0], 0x5140u), __byte_perm(u.x[0], u.y[0], 0x7362u),
        __byte_perm(u.x[1], u.y[1], 0x5140u), __byte_perm(u.x[1], u.y[1], 0x7362u));
    st4(dst + 512u, __byte_perm(u.x[2], u.y[2], 0x5140u), __byte_perm(u.x[2], u.y[2], 0x7362u),
        __byte_perm(u.x[3], u.y[3], 0x5140u), __byte_perm(u.x[3], u.y[3], 0x7362u));
  };

  int32_t acc[MT][NB][4];
  uint32_t af[2][4];        // A's fragment of the m16 tile being folded, and the next one's
  uint32_t bf[NB][2];       // B's eight fragments of the k-step

  auto fold_pair = [&](uint32_t x, uint32_t rl, uint32_t np) -> uint32_t {
    const int32_t *p0 = acc[2u * rl][2u * np], *p1 = acc[2u * rl + 1u][2u * np];
    const int32_t *p2 = acc[2u * rl][2u * np + 1u], *p3 = acc[2u * rl + 1u][2u * np + 1u];
    const uint32_t ta = xor3((uint32_t)p0[0], (uint32_t)p0[1], (uint32_t)p0[2]);
    const uint32_t tb = xor3((uint32_t)p0[3], (uint32_t)p1[0], (uint32_t)p1[1]);
    const uint32_t tc = xor3((uint32_t)p1[2], (uint32_t)p1[3], (uint32_t)p2[0]);
    const uint32_t td = xor3((uint32_t)p2[1], (uint32_t)p2[2], (uint32_t)p2[3]);
    const uint32_t te = xor3(ta, tb, tc);
    const uint32_t rest = np == 0u ? (te ^ td) : xor3(te, td, x);
    const uint32_t t3 = xor3((uint32_t)p3[0], (uint32_t)p3[1], (uint32_t)p3[2]);
    return xor3(t3, (uint32_t)p3[3], rest);
  };
  // Region pair rl's part of chunk q's word: the lane's 64 sums in 32 xor3, rotated once for
  // every later chunk of the same word (the reference rotates the word left 13 before each
  // chunk's XOR), then XORed into the region's word by every lane that holds a quarter of it.
  // tr is the lane's region base: trL plus the warp's part.
  auto readout_end = [&](uint32_t x, uint32_t tr, uint32_t rl, uint32_t q) {
    const uint32_t later = (chunks >> 4) - 1u - (q >> 4);
    x = __funnelshift_l(x, x, (13u * later) & 31u);
#if PD_RED_PRED
    asm volatile("{\n .reg .pred p;\n setp.ne.u32 p, %2, 99;\n @p red.shared.xor.b32 [%0], %1;\n}"
                 ::"r"(tr + rl * 128u + (q & 15u) * 4u), "r"(x), "r"(trL) : "memory");
#else
    asm volatile("red.shared.xor.b32 [%0], %1;" ::"r"(tr + rl * 128u + (q & 15u) * 4u), "r"(x)
                 : "memory");
#endif
  };
  auto readout = [&](uint32_t tr, uint32_t rl, uint32_t q) {
    uint32_t x = 0u;
#pragma unroll
    for (uint32_t np = 0; np < NB / 2u; np++) x = fold_pair(x, rl, np);
    readout_end(x, tr, rl, q);
  };

  // The fences' branches (see PD_SLOT): one a label, never taken (the host never asks for
  // this many tiles), so each label has a second way in and ptxas keeps its block apart.
  // Each tests its own value: with one value for all, ptxas keeps the first branch only.
#define PD_FENCED(M, N) (((PD_FENCE_MASK) >> ((M) * 8 + (N))) & 1ull)
#define PD_SLOT(W, S, M, N)                                                                \
  if (PD_FENCED(M, N))                                                                     \
    asm volatile("{\n .reg .pred p;\n setp.eq.u32 p, %0, %1;\n @p bra PDL_" #W "_" #S "_" #M "_" #N ";\n}" \
                 ::"r"(tiles), "r"(0x7ffff000u + W * 192u + S * 48u + M * 8u + N));
#define PD_TILE(W, S, M)                                                                   \
  PD_SLOT(W, S, M, 0) PD_SLOT(W, S, M, 1) PD_SLOT(W, S, M, 2) PD_SLOT(W, S, M, 3)          \
  PD_SLOT(W, S, M, 4) PD_SLOT(W, S, M, 5) PD_SLOT(W, S, M, 6) PD_SLOT(W, S, M, 7)
#define PD_KSTEP(W, S)                                                                     \
  PD_TILE(W, S, 0) PD_TILE(W, S, 1) PD_TILE(W, S, 2) PD_TILE(W, S, 3) PD_TILE(W, S, 4) PD_TILE(W, S, 5)
#define PD_CHUNK(W) PD_KSTEP(W, 0) PD_KSTEP(W, 1) PD_KSTEP(W, 2) PD_KSTEP(W, 3)
#define PD_WARPS PD_CHUNK(0) PD_CHUNK(1) PD_CHUNK(2) PD_CHUNK(3) PD_CHUNK(4) PD_CHUNK(5) PD_CHUNK(6) PD_CHUNK(7)
#if PD_FENCE_MASK
  PD_WARPS
#endif
#undef PD_SLOT

  for (uint32_t v = blockIdx.x; v < tiles; v += gridDim.x) {
    const uint32_t rbg = v % row_groups, cbg = v / row_groups;
#if PD_PROBE
    // The probe's stamps live in shared (sCnt + 16 + 24 * warp), not in registers.
    if (lane == 0u) {
      const long long pt0 = clock64();
      asm volatile("st.shared.u64 [%0], %1;" ::"r"(sCnt + 16u + warp * 24u), "l"(pt0) : "memory");
    }
#endif
    // ---- the tile's tables, from the host's ----
    {
      const uint4 *const tb = tabB + (size_t)(col_off + cbg * 16u) * 128u;
      const uint4 *const ta = tabA + (size_t)(rbg * 12u) * 128u;
      for (uint32_t i = threadIdx.x; i < 3584u; i += PD_THREADS) {
        const uint4 w = i < 2048u ? tb[i] : ta[i - 2048u];
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};"
                     ::"r"(sbase + PD_TABB + i * 16u), "r"(w.x), "r"(w.y), "r"(w.z), "r"(w.w)
                     : "memory");
      }
      if (threadIdx.x == 0u) asm volatile("st.shared.u32 [%0], %1;" ::"r"(sCnt), "r"(0u) : "memory");
    }
    __syncthreads();
    // ---- stage 0, every unit of it: B's eight quads (a warp each) and A's six ----
    {
      Unit u;
      ldg4(u.o, offs + lane);
      u_half(u.x, sbase + PD_TABB + warp * 4096u, u.o[0], u.o[1]);
      u_half(u.y, sbase + PD_TABB + warp * 4096u, u.o[2], u.o[3]);
      u_store(u, fragL + warp * 1024u);
      if (warp < 6u) {
        ldg4(u.o, offs + 32u + lane);
        u_half(u.x, sbase + PD_TABA + warp * 4096u, u.o[0], u.o[1]);
        u_half(u.y, sbase + PD_TABA + warp * 4096u, u.o[2], u.o[3]);
        u_store(u, fragL + PD_AOFF + warp * 1024u);
      }
    }
    __syncthreads();
    // The salt stamp: A's row 0 holds it in place of the fill for its first bytes, so those
    // bytes of A' are the stamp plus the noise (saturated, as the materialise does).
    if (rbg == 0u && threadIdx.x < stamp_bytes) {
      unsigned char *const b = reinterpret_cast<unsigned char *>(pd_smem) + PD_STG + PD_AOFF
                               + (threadIdx.x >> 1) * 16u + (threadIdx.x & 1u);
      int s = (int)(signed char)*b - 64 + (int)stamp[threadIdx.x];
      s = s < -128 ? -128 : (s > 127 ? 127 : s);
      *b = (unsigned char)s;
    }
    __syncthreads();

#pragma unroll
    for (uint32_t mb = 0; mb < MT; mb++)
#pragma unroll
      for (uint32_t nb = 0; nb < NB; nb++)
#pragma unroll
        for (uint32_t i = 0; i < 4; i++) acc[mb][nb][i] = 0;
    // The first k-step's fragments: A's first m16 tile and B's eight. A's four matrices are
    // (rows 0-7, k 0-15), (rows 8-15, k 0-15), (rows 0-7, k 16-31), (rows 8-15, k 16-31); B's
    // of a pair of n8 tiles are (tile 0, k 0-15), (tile 0, k 16-31), (tile 1, k 0-15),
    // (tile 1, k 16-31).
    ldmt(af[0][0], af[0][1], af[0][2], af[0][3], fragL + PD_AOFF + (warp >> 2) * 3072u);
#pragma unroll
    for (uint32_t p = 0; p < NB / 2u; p++)
      ldmt(bf[2u * p][0], bf[2u * p][1], bf[2u * p + 1u][0], bf[2u * p + 1u][1],
           fragL + (warp & 3u) * 2048u + p * 512u);
    const uint4 *pOff = offs + lane;                  // this lane's offsets, k-step 0, B's
#if PD_PROBE
    if (lane == 0u) {
      const long long pt1 = clock64();
      asm volatile("st.shared.u64 [%0], %1;" ::"r"(sCnt + 24u + warp * 24u), "l"(pt1) : "memory");
    }
#endif

    // One mma of a k-step and what follows it. W (the warp), s, mb and nb are constants at
    // every call (PD_SLOT below), so every test folds and a slot is straight-line code.
    Unit ub, ua;
    Half hf;
    uint32_t xy[4], ww[4], rx = 0u;
    uint32_t c = 0u;
    auto slot = [&](const uint32_t W, const uint32_t s, const uint32_t mb, const uint32_t nb) {
      const uint32_t wr = W >> 2, wc = W & 3u;
      // Stage of this k-step and of the next: buffers alternate, and a chunk is four.
      const uint32_t cur = (s & 1u) * PD_STAGE, nxt = PD_STAGE - cur;
      const uint32_t i = mb * NB + nb;
      const uint32_t as = mb & 1u;
      const uint32_t tr = trL + (wc * 2u * WARP_REGIONS + wr * WARP_REGIONS) * 64u;
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
          "{%8,%9}, {%0,%1,%2,%3};\n"
          : "+r"(acc[mb][nb][0]), "+r"(acc[mb][nb][1]), "+r"(acc[mb][nb][2]), "+r"(acc[mb][nb][3])
          : "r"(af[as][0]), "r"(af[as][1]), "r"(af[as][2]), "r"(af[as][3]), "r"(bf[nb][0]),
            "r"(bf[nb][1]));
      // What follows the mma: every piece cut into steps, each written after the mma it is
      // to follow. ptxas issues what is ready in the order of the source, lifts a load up to
      // the shared store before it, and pulls a later instruction forward only when the next
      // one's inputs do not exist yet. So a step is written at least two mma after the step
      // that feeds it, and comes out where it is written; a warp then never waits for a
      // table read with mma queued behind it (measured, with a half unit written as one
      // run: 14.7% of the pipe).
      {
#if PD_ADUP
        const bool drawA = true;
        const uint32_t qa = W < 6u ? W : W - 6u;
#else
        const bool drawA = wc != s;
        const uint32_t qa = (wc - (wc > s ? 1u : 0u)) * 2u + wr;
#endif
        const uint32_t tb = sbase + PD_TABB + W * 4096u, ta = sbase + PD_TABA + qa * 4096u;
        const uint32_t db = dstL + nxt + W * 1024u, da = dstL + nxt + PD_AOFF + qa * 1024u;
#if PD_NO_DRAW
        const uint32_t nh = 0u;
#else
        const uint32_t nh = drawA ? 4u : 2u;                           // half units this k-step
#endif
        // A's next m16 tile of this k-step, into the other fragment set, eight mma ahead.
        if (nb == 0u && mb + 1u < MT)
          ldmt(af[as ^ 1u][0], af[as ^ 1u][1], af[as ^ 1u][2], af[as ^ 1u][3],
               fragL + PD_AOFF + wr * 3072u + cur + (mb + 1u) * 512u);
        if (i == 0u && nh != 0u) {
          // The offsets of this k-step's units (global loads: ptxas lifts them to the top of
          // the block, the wait of the k-step before).
          ldg4(ub.o, pOff + (s + 1u) * 64u);
          if (drawA) ldg4(ua.o, pOff + (s + 1u) * 64u + 32u);
        }
        // Half unit h (B's rows 0-7 and 8-15 of its sixteen, then A's) goes with m16 tile h:
        // table reads, subtracts, byte selects, store.
        if (mb < nh) {
          const uint32_t h = mb;
          if (nb == PD_L) u_load_half(hf, h < 2u ? ub.o : ua.o, h < 2u ? tb : ta, h & 1u);
          if (nb == PD_A) {
#if PD_CPIN
            // The fill's constant, made to wait for mma PD_CP of this tile: the sum ANDed with
            // zero, ORed with the constant, in one lop3. ptxas rates a table read as ready at
            // once and would issue the subtracts right behind the reads, where the warp then
            // sits; this is what puts the tile's first mma between them.
            uint32_t fc;
            asm("lop3.b32 %0, %1, 0x40404040, %2, 0xEC;" : "=r"(fc) : "r"(acc[mb][PD_CP][0]), "r"(zero));
#else
            const uint32_t fc = 0x40404040u;
#endif
            xy[0] = hf.p0[0] + fc - hf.m0[0];
            xy[1] = hf.p0[1] + fc - hf.m0[1];
            xy[2] = hf.p1[0] + fc - hf.m1[0];
            xy[3] = hf.p1[1] + fc - hf.m1[1];
          }
          if (nb == PD_P) {
            ww[0] = __byte_perm(xy[0], xy[2], 0x5140u);
            ww[1] = __byte_perm(xy[0], xy[2], 0x7362u);
            ww[2] = __byte_perm(xy[1], xy[3], 0x5140u);
            ww[3] = __byte_perm(xy[1], xy[3], 0x7362u);
          }
          if (nb == PD_S) st4((h < 2u ? db : da) + 512u * (h & 1u), ww[0], ww[1], ww[2], ww[3]);
        }
      }
#if !PD_NO_READOUT
      // The readout's three runs, each in four parts and an end (the rotate and the shared
      // XOR). Region pair 2 of the chunk before, over the first k-step's tile 0 (zero sums in
      // a tile's first chunk, so it XORs nothing); pair 0 over the last k-step's tile 4;
      // pair 1 over its tile 5.
      {
        constexpr uint32_t part0[5] = PD_RO0, part1[5] = PD_RO1, part2[5] = PD_RO2;
#pragma unroll
        for (uint32_t np = 0; np < 4u; np++) {
          if (s == 0u && i == part2[np]) rx = fold_pair(np == 0u ? 0u : rx, 2u, np);
          if (s == 3u && i == part0[np]) rx = fold_pair(np == 0u ? 0u : rx, 0u, np);
          if (s == 3u && i == part1[np]) rx = fold_pair(np == 0u ? 0u : rx, 1u, np);
        }
        if (s == 0u && i == part2[4]) readout_end(rx, tr, 2u, c - 1u);
        if (s == 3u && i == part0[4]) readout_end(rx, tr, 0u, c);
        if (s == 3u && i == part1[4]) readout_end(rx, tr, 1u, c);
      }
#endif
#if !PD_NO_SYNC
      // After the draw's last store and A's last load of this stage: this warp has drawn
      // its part of the next stage and loaded all of this one.
      if (i == PD_ADD_AT) atomicAdd(reinterpret_cast<unsigned *>(pd_smem) + PD_CNT / 4u, 1u);
      // Every warp has added: the next stage is whole, and this one's buffer is free to draw
      // into from the next k-step on. The counter is 256 a k-step of the tile.
      if (i == PD_WAIT_AT) {
        for (;;) {
          uint32_t seen;
          asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(seen) : "r"(sCnt) : "memory");
          if (!__any_sync(0xffffffffu, (int)(seen - (c << 10)) < (int)((s + 1u) * PD_THREADS))) break;
#if PD_PROBE && PD_SPINCOUNT
          atomicAdd(cyc + (blockIdx.x * 8u + warp) * 5u + 4u, 1ull);
#endif
        }
      }
#endif
      // The next k-step's fragments, in the last m16 tile: A's first tile, and B's pairs
      // into the registers of the pairs they replace. Each pair's load is held (PD_HOLDS) by a
      // 4-byte store of the sum the mma two back made, into the lane's scratch row: without
      // it ptxas lifts all four to the wait, into sixteen registers of their own.
      if (i == PD_WAIT_AT + 1u)
        ldmt(af[0][0], af[0][1], af[0][2], af[0][3], fragL + PD_AOFF + wr * 3072u + nxt);
      if (mb + 1u == MT && (nb & 1u)) {
#if PD_HOLDS
        asm volatile("st.shared.u32 [%0], %1;" ::"r"(dstL + (PD_SCR - PD_STG)), "r"(acc[mb][nb - 1u][0]) : "memory");
#endif
        ldmt(bf[nb - 1u][0], bf[nb - 1u][1], bf[nb][0], bf[nb][1],
             fragL + wc * 2048u + nxt + (nb >> 1) * 512u);
      }
    };
    // A fence is a label: ptxas orders instructions inside a basic block only, and a label
    // that something can branch to ends one. The branches are at the kernel's start, behind
    // tests that are never true; the stream itself gets no instruction.
#define PD_SLOT(W, S, M, N)                                                                \
  slot(W##u, S##u, M##u, N##u);                                                            \
  if (PD_FENCED(M, N)) asm volatile("PDL_" #W "_" #S "_" #M "_" #N ":" ::: "memory");
#define PD_LOOP(W)                                                                         \
  case W:                                                                                  \
    for (; c < chunks; c++) {                                                              \
      PD_CHUNK(W)                                                                          \
      pOff += 256; /* four k-steps of B's and A's offsets, 32 lanes each */ \
    }                                                                                      \
    break;
    switch (warp) {
      PD_LOOP(0)
      PD_LOOP(1)
      PD_LOOP(2)
      PD_LOOP(3)
      PD_LOOP(4)
      PD_LOOP(5)
      PD_LOOP(6)
      PD_LOOP(7)
      default:
        break;
    }
#undef PD_SLOT
#undef PD_LOOP
#if PD_PROBE
    if (lane == 0u) {
      const long long pt2 = clock64();
      asm volatile("st.shared.u64 [%0], %1;" ::"r"(sCnt + 32u + warp * 24u), "l"(pt2) : "memory");
    }
#endif
    // The last chunk's region pair 2 (pairs 0 and 1 were read out in its last k-step).
    {
      const uint32_t tr = trL + ((warp & 3u) * 2u * WARP_REGIONS + (warp >> 2) * WARP_REGIONS) * 64u;
#if PD_NO_READOUT
      readout(tr, 0u, chunks - 1u);
      readout(tr, 1u, chunks - 1u);
#endif
      readout(tr, 2u, chunks - 1u);
    }

    // Hand-off: the column slot's two warps meet, so every word is in shared.
    asm volatile("bar.sync %0, %1;" ::"r"(1u + (warp & 3u)), "r"(64u) : "memory");
#if PD_PROBE
    if (warp < 4u) {
      // The probe: the slot's 48 regions go out as they are, then are zeroed.
#pragma unroll
      for (uint32_t i = 0; i < 6u; i++) {
        const uint32_t o = warp * 3072u + i * 512u + lane * 16u;
        uint32_t w[4];
        ld4(w, sbase + PD_TR + o);
        if (v < dump_tiles) {
          uint32_t *const d = dump + (size_t)v * 3072u + o / 4u;
          d[0] = w[0];
          d[1] = w[1];
          d[2] = w[2];
          d[3] = w[3];
        }
        st4(sbase + PD_TR + o, 0u, 0u, 0u, 0u);
      }
    }
    if (lane == 0u) {
      // Cycles: the tile's prologue, its chunk loop, and its end, summed a warp.
      const long long pt3 = clock64();
      long long pt0, pt1, pt2;
      asm volatile("ld.shared.u64 %0, [%1];" : "=l"(pt0) : "r"(sCnt + 16u + warp * 24u) : "memory");
      asm volatile("ld.shared.u64 %0, [%1];" : "=l"(pt1) : "r"(sCnt + 24u + warp * 24u) : "memory");
      asm volatile("ld.shared.u64 %0, [%1];" : "=l"(pt2) : "r"(sCnt + 32u + warp * 24u) : "memory");
      atomicAdd(cyc + (blockIdx.x * 8u + warp) * 5u + 0u, (unsigned long long)(pt1 - pt0));
      atomicAdd(cyc + (blockIdx.x * 8u + warp) * 5u + 1u, (unsigned long long)(pt2 - pt1));
      atomicAdd(cyc + (blockIdx.x * 8u + warp) * 5u + 2u, (unsigned long long)(pt3 - pt2));
      atomicAdd(cyc + (blockIdx.x * 8u + warp) * 5u + 3u, 1ull);
    }
#else
#error "the miner's hash is not written yet"
#endif
    __syncthreads();
  }
#undef PD_FENCED
#undef PD_TILE
#undef PD_KSTEP
#undef PD_CHUNK
#undef PD_WARPS
#endif
}
