// The wave64 fold of RDNA 3's gfx1100 and gfx1101 (PEARL_AMD_W11K). pearl_amd_fold.h includes this file after its own
// folds, inside no namespace; it uses that file's switches, types and helpers and nothing else.
#pragma once

// PEARL_AMD_W11K (goal-f-rdna3, 20 November; opt-in): RDNA 3's fold in wave64, for a gfx1100 or gfx1101 device pass
// given -mwavefrontsize64 -DPEARL_AMD_WAVE64=1 in PEARL_AMD_CFLAGS. No build turns it on: on an RX 7900 XTX it read
// level with the wave32 fold, 0.5% behind in every round (earn/tuning/benchmark.md). gfx11's wave64 v_wmma_i32_16x16x16_iu8
// takes 16 bytes of A and of B a lane, as wave32 does, and 4 sums a lane: sum g of lane l is C[4 g + l / 16][l % 16].
// It reads A and B from lanes 0-31 only (sums in lanes 0-15 and 32-47 take lanes 0-15's copy, the rest lanes
// 16-31's; test/pearl_amd_lay64_11.cpp on an RX 7900 XTX). So one 16-byte load issued in lanes 0-31 alone feeds a
// WMMA tile, as in wave32, and a wave64 holds 8 x 4 tiles (128 x 64 of C) in the 128 accumulator VGPRs that hold
// 4 x 4 in wave32. A block is one row of 4 waves64 over the same 128 x 256 tile, 256 threads and 32-byte slab, so the
// host launches it as it launches the wave32 fold and lays the operands out the same way. Against the wave32 fold
// (PEARL_AMD_GB11), per block and slab:
//   A' fragment reads from LDS   64 of 512 bytes, as before (4 waves x 8 tiles x 2 k-steps)
//   B' loads from memory         32 of 512 bytes, not 64: no second wave of a column loads the same rows
//   barriers                     4 waves, not 8
// PEARL_AMD_W11K_MASK: how the loads are kept to lanes 0-31. 1: under `if (lane < 32)`, the compiler's exec mask; it
// copies every loaded register once (a v_mov a VGPR) where the two paths meet. 2: the fold clears exec's high half
// itself around each burst of loads (two s_mov_b32, no branch and no copy). The compiler doesn't know, so a burst
// holds nothing but the loads: a sched_barrier each side keeps other instructions out, and every address in it is a
// register set before the loop plus a constant. 0: every load in all 64 lanes (1,024 bytes a load), to time what
// the mask is worth. 3: as 2 for the B' loads only, one burst a slab; the A' reads go to all 64 lanes, whose upper
// half reads the lower half's addresses.
// The profile's 16 x 16 regions: rows 0-3 and 8-11 of a WMMA tile are registers 0 and 2, rows 4-7 and 12-15
// registers 1 and 3, so a lane holds two regions a 32 x 64 block, 16 sums of each, and the 16 lanes of a region
// differ in lane bits 0, 3, 4 and 5. The tile's transcripts are in LDS (128 regions x 16 words). At a chunk's end
// each lane XORs its 16 sums of a region, four cross-lane steps (two DPP, v_permlanex16, v_permlane64) make that the
// region's word in all 16 lanes, and the four lanes of the wave with those bits clear (0, 2, 4, 6: exec 0x55) store
// it. No word is kept in a register: with 8 kept a lane the fold needed 199 VGPRs, 7 more than 4 waves64 a SIMD
// leave. PEARL_AMD_W11K_RDX=1 has every lane XOR its own part into the word with an atomic LDS instruction instead
// (no cross-lane step): 16 lanes on one address, which ran 4.5% slower on an RX 7900 XTX.
#ifndef PEARL_AMD_W11K
#if PEARL_AMD_WMMA11 && PEARL_AMD_GFX1100_1101 && !PEARL_AMD_EMU && defined(PEARL_AMD_WAVE64) && PEARL_AMD_WAVE64 && \
    PEARL_AMD_RDNA3_SET && PEARL_AMD_GB11 && PEARL_AMD_BUFLD && PEARL_AMD_TILED && !PEARL_AMD_STAMP && !PEARL_AMD_ABLATE && \
    !PEARL_AMD_KWRAP
#define PEARL_AMD_W11K 1
#else
#define PEARL_AMD_W11K 0
#endif
#endif
#if !PEARL_AMD_W11K && PEARL_AMD_WMMA11 && !PEARL_AMD_EMU && defined(PEARL_AMD_WAVE64) && PEARL_AMD_WAVE64
#error "a wave64 gfx11 pass builds the wave64 fold only (PEARL_AMD_W11K: gfx1100 and gfx1101 with RDNA 3's set and GB11)"
#endif
#ifndef PEARL_AMD_W11K_MASK
#define PEARL_AMD_W11K_MASK 2
#endif
// PEARL_AMD_XDLY=1 (from q3-amd's finding): an s_delay_alu after each of the fold's own writes of exec. The
// compiler adds none after inline asm, and the instruction after the write depends on it.
#ifndef PEARL_AMD_XDLY
#define PEARL_AMD_XDLY 0
#endif
#if PEARL_AMD_XDLY
#define PEARL_AMD_XDLY_S "\n\ts_delay_alu instid0(SALU_CYCLE_1)"
#else
#define PEARL_AMD_XDLY_S ""
#endif
#if PEARL_AMD_W11K_MASK < 0 || PEARL_AMD_W11K_MASK > 4
#error "PEARL_AMD_W11K_MASK is 0, 1, 2, 3 or 4"
#endif
// =4: as 2, and the A' reads in the 16 lanes the WMMA reads A from: the even lanes of 0-15 and the odd lanes of
// 16-31 (row 4 g + q of C takes A row r from lane r of lanes 0-15 for q 0 and 2, of lanes 16-31 for q 1 and 3).
#if PEARL_AMD_W11K_MASK == 4
#define PEARL_AMD_W11K_ALO()                                  \
  do {                                                        \
    __builtin_amdgcn_sched_barrier(0);                        \
    asm volatile("s_mov_b32 exec_lo, 0xAAAA5555" PEARL_AMD_XDLY_S ::: "memory"); \
    asm volatile("s_mov_b32 exec_hi, 0" PEARL_AMD_XDLY_S ::: "memory");        \
  } while (0)
#define PEARL_AMD_W11K_A16() asm volatile("s_mov_b32 exec_lo, 0xAAAA5555" PEARL_AMD_XDLY_S ::: "memory")
#else
#define PEARL_AMD_W11K_ALO() PEARL_AMD_W11K_LO()
#define PEARL_AMD_W11K_A16() ((void)0)
#endif
#if PEARL_AMD_W11K_MASK >= 2
#define PEARL_AMD_W11K_LO()                                   \
  do {                                                        \
    __builtin_amdgcn_sched_barrier(0);                        \
    asm volatile("s_mov_b32 exec_hi, 0" PEARL_AMD_XDLY_S ::: "memory");        \
  } while (0)
#define PEARL_AMD_W11K_ALL()                                  \
  do {                                                        \
    asm volatile("s_mov_b64 exec, -1" PEARL_AMD_XDLY_S ::: "memory");          \
    __builtin_amdgcn_sched_barrier(0);                        \
  } while (0)
#else
#define PEARL_AMD_W11K_LO() asm volatile("" ::: "memory")   /* keeps the loads in the masked block: no hoisting */
#define PEARL_AMD_W11K_ALL() ((void)0)
#endif
// The A' tiles read in one burst, then their WMMAs.
#ifndef PEARL_AMD_W11K_GRP
#define PEARL_AMD_W11K_GRP 2
#endif
#ifndef PEARL_AMD_W11K_RDX
#define PEARL_AMD_W11K_RDX 0
#endif
// PEARL_AMD_W11K_ROLL=1: B' one k-step ahead. The wave holds two k-steps of B' (32 VGPRs) and, as each k-step
// starts, loads the next k-step's four tiles into the registers the last one freed, in the burst of that k-step's
// first A' reads. The loads no longer follow the slab, so the slab can be 64 bytes (-DPEARL_AMD_SLAB=64u: four
// k-steps and one barrier where the 32-byte slab has two; a thread then stages two units of A', each loaded at an
// even k-step and stored after the next). =0: both k-steps of the next slab's B' after the slab's last WMMA.
#ifndef PEARL_AMD_W11K_ROLL
#define PEARL_AMD_W11K_ROLL 0
#endif
// PEARL_AMD_W11K_PAD: bytes of LDS a block takes beyond what it uses, to hold a WGP to fewer blocks (4 fit unpadded).
#ifndef PEARL_AMD_W11K_PAD
#define PEARL_AMD_W11K_PAD 0
#endif
#if PEARL_AMD_W11K_GRP != 1 && PEARL_AMD_W11K_GRP != 2 && PEARL_AMD_W11K_GRP != 4 && PEARL_AMD_W11K_GRP != 8
#error "PEARL_AMD_W11K_GRP is 1, 2, 4 or 8"
#endif
#if PEARL_AMD_W11K
namespace {
__device__ __forceinline__ void pearl_amd_fold_w11k(const int8_t *__restrict__ Ap, const int8_t *__restrict__ Bp,
                                                    uint32_t k, uint32_t rank, uint32_t col0, uint32_t rows_valid,
                                                    uint32_t *__restrict__ tr PEARL_AMD_FUSE_PARAMS) {
  constexpr int W = 64, MT = 8, NT = 4, WM = 16 * MT, WN = 16 * NT;
  constexpr int BM = (int)PEARL_AMD_BM, BN = (int)PEARL_AMD_BN, T = (int)PEARL_AMD_THREADS;
  constexpr int KS = (int)PEARL_AMD_SLAB, S = KS + 16, G = PEARL_AMD_W11K_GRP;
  constexpr int RB = WM / 32;                         // 32 x 64 blocks a wave, each two regions a lane
  constexpr int RR = 2 * BM / 32, RC = 4 * BN / 64;   // regions down and across the tile: 8 and 16
  static_assert(BM == WM && BN == 4 * WN && T == 4 * W, "a block is one row of 4 waves64 of 128 x 64");
  constexpr int KJ = KS / 16, U = KS / 32;   // k-steps a slab, and the 16-byte units of A' a thread stages
  static_assert((KS == 32 || (KS == 64 && PEARL_AMD_W11K_ROLL)) && BM * KJ == U * T,
                "the slab is 32 bytes, or 64 with PEARL_AMD_W11K_ROLL; a thread stages one unit of A' or two");
  constexpr uint32_t perChunk = PEARL_FOLD_RANK / (uint32_t)KS;
  static_assert(perChunk * (uint32_t)KS == PEARL_FOLD_RANK && (perChunk == 4u || perChunk == 2u), "a chunk is four slabs or two");
  static_assert(2 * RR * RC == T, "half the block hashes, a region a thread");
  __shared__ __attribute__((aligned(16))) int8_t sA[2][BM * S + PEARL_AMD_W11K_PAD / 2];
  __shared__ __attribute__((aligned(16))) uint32_t sT[RR * RC * PEARL_JACKPOT_BUCKETS];   // region j's words at 16 j
  static_assert(RR * RC * (int)PEARL_JACKPOT_BUCKETS == 8 * T, "a thread clears 8 words");
  const int t = (int)threadIdx.x;
  __builtin_assume(t >= 0 && t < T);
  const int wc = t / W, lane = t % W;
#if PEARL_AMD_WALK_ON
  uint32_t tileX, tileY;
  pearl_amd_walk(blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, tileX, tileY);
  const uint32_t tileRow = tileY * (uint32_t)BM, tileCol = tileX * (uint32_t)BN;
#else
  const uint32_t tileRow = blockIdx.y * (uint32_t)BM, tileCol = blockIdx.x * (uint32_t)BN;
#endif
  const int8_t *gA = Ap + (size_t)tileRow * k;
  const int8_t *gB = Bp + (size_t)(col0 + tileCol) * k;
  const __amdgpu_buffer_rsrc_t rsA =
      __builtin_amdgcn_make_buffer_rsrc(const_cast<int8_t *>(gA), (short)0, (int)((uint32_t)BM * k), 0x31004000);
  const __amdgpu_buffer_rsrc_t rsB =
      __builtin_amdgcn_make_buffer_rsrc(const_cast<int8_t *>(gB), (short)0, (int)((uint32_t)BN * k), 0x31004000);
  // Staging: thread t loads units t and, at a 64-byte slab, t + T of the tile's slab of A' (row t / KJ + u T / KJ,
  // 16 bytes t % KJ) and stores them in LDS.
  int stOff = (t / KJ) * S + 16 * (t % KJ);
  // The lane's fragment row in each LDS tile, and in its wave's B' tiles in memory.
  int rdOff = (lane & 15) * S;
  int gbOff = (wc * WN + (lane & 15)) * KS;
  asm volatile("" : "+v"(stOff), "+v"(rdOff), "+v"(gbOff));
  const bool lo = PEARL_AMD_W11K_MASK != 1 || lane < 32;   // the lanes the WMMA reads its operands from
  auto stage_load = [&](uint32_t slab, int u = 0) -> pearl_amd_v4i {
    return __builtin_bit_cast(pearl_amd_v4i, __builtin_amdgcn_raw_buffer_load_b128(
        rsA, 16 * t, (int)(slab * (uint32_t)(BM * KS) + (uint32_t)(16 * u * T)), 0));
  };
  auto stage_store = [&](int buf, pearl_amd_v4i v, int u = 0) {
    *reinterpret_cast<pearl_amd_v4i *>(&sA[buf][0] + stOff + u * (T / KJ) * S) = v;
  };
  // B' for a slab: both k-steps of the wave's four tiles, in lanes 0-31. A slab past the last reads zeros (the
  // buffer's bound), and nothing uses them.
  pearl_amd_v4i bq[2][NT];
  auto b_load = [&](uint32_t slab) {
    pearl_amd_v4i nb[2][NT];
    if (lo) {
      PEARL_AMD_W11K_LO();
#pragma unroll
      for (int j = 0; j < 2; j++)
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
          nb[j][nt] = __builtin_bit_cast(pearl_amd_v4i, __builtin_amdgcn_raw_buffer_load_b128(
              rsB, gbOff + nt * 16 * KS + 16 * j, (int)(slab * (uint32_t)(BN * KS)), 0));
      PEARL_AMD_W11K_ALL();
    }
#pragma unroll
    for (int j = 0; j < 2; j++)
#pragma unroll
      for (int nt = 0; nt < NT; nt++) bq[j][nt] = nb[j][nt];
  };
  pearl_amd_v4i acc[MT][NT];
#pragma unroll
  for (int mt = 0; mt < MT; mt++)
#pragma unroll
    for (int nt = 0; nt < NT; nt++) acc[mt][nt] = pearl_amd_v4i{0, 0, 0, 0};
  // The transcripts start at zero: thread t clears words 8 t .. 8 t + 7. The barrier before the slab loop orders it.
  reinterpret_cast<pearl_amd_v4i *>(sT + 8 * t)[0] = pearl_amd_v4i{0, 0, 0, 0};
  reinterpret_cast<pearl_amd_v4i *>(sT + 8 * t)[1] = pearl_amd_v4i{0, 0, 0, 0};
  static_assert(PEARL_AMD_CRANK && PEARL_AMD_PF2, "the code object says its fold takes rank 128 only (PEARL_AMD_CAP_RANK)");
  const uint32_t slabs = k / (uint32_t)KS;
  (void)rank;
  __builtin_assume(slabs % perChunk == 0u && slabs >= perChunk);
  auto readout = [&](uint32_t c) {
    // Chunk c's word of the lane's first region: column offset co = (lane >> 1) & 3 of its wave's 64 columns, rows
    // 2 rb + ro of the tile's 8. The lane's place comes from gbOff (64 wc + lane % 16, times the slab), which the
    // fold keeps anyway, so no register holds it between chunks. Its byte offset goes through an empty asm, so the
    // stores below need no address arithmetic of their own.
    const uint32_t x = (uint32_t)gbOff / (uint32_t)KS;
    uint32_t wb = ((((x >> 6) << 2) | ((x >> 1) & 3u)) * (uint32_t)(RR * PEARL_JACKPOT_BUCKETS) + c) * 4u;
    asm volatile("" : "+v"(wb));
    uint32_t *const w = reinterpret_cast<uint32_t *>(reinterpret_cast<int8_t *>(sT) + wb);
    uint32_t v[RB][2];
    (void)v;
#pragma unroll
    for (int rb = 0; rb < RB; rb++)
#pragma unroll
      for (int ro = 0; ro < 2; ro++) {
        uint32_t x = 0u;
#pragma unroll
        for (int mt = 2 * rb; mt < 2 * rb + 2; mt++)
#pragma unroll
          for (int nt = 0; nt < NT; nt++) x ^= (uint32_t)acc[mt][nt][ro] ^ (uint32_t)acc[mt][nt][ro + 2];
#if PEARL_AMD_W11K_RDX
        __hip_atomic_fetch_xor(w + (2 * rb + ro) * (int)PEARL_JACKPOT_BUCKETS, x, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_WORKGROUP);
#else
        x ^= (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0xB1, 0xF, 0xF, false);    // lane ^ 1
        x ^= (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x128, 0xF, 0xF, false);   // lane ^ 8
        x ^= __builtin_amdgcn_permlanex16(x, x, 0x76543210u, 0xFEDCBA98u, false, false);   // lane ^ 16
        x ^= (uint32_t)__builtin_amdgcn_permlane64(x);                                    // lane ^ 32
        v[rb][ro] = x;
#endif
      }
#if !PEARL_AMD_W11K_RDX
    // The keeper lanes store: exec is lanes 0, 2, 4 and 6 for the stores alone (the compiler doesn't know; a
    // sched_barrier each side keeps everything else out).
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_mov_b64 exec, 0x55" PEARL_AMD_XDLY_S ::: "memory");
#pragma unroll
    for (int rb = 0; rb < RB; rb++)
#pragma unroll
      for (int ro = 0; ro < 2; ro++) w[(2 * rb + ro) * (int)PEARL_JACKPOT_BUCKETS] = v[rb][ro];
    asm volatile("s_mov_b64 exec, -1" PEARL_AMD_XDLY_S ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
#endif
  };
  auto slab_step = [&](uint32_t s, auto parity, auto last) {
    constexpr int B = decltype(parity)::value;      // s mod 2: slab s's LDS buffer
    pearl_amd_v4i pre;
    const int8_t *aBase = &sA[B][0] + rdOff;
#pragma unroll
    for (int j = 0; j < 2; j++)
#pragma unroll
      for (int g0 = 0; g0 < MT; g0 += G) {
        if (j == 1 && g0 == 0) {
          // Slab s + 1's unit of A', stored when this slab's WMMAs are done: issued here, when the first k-step's B'
          // registers are free, so it has a k-step's WMMAs to arrive and takes no register at the fold's fullest.
          pre = stage_load(s + 1u);
          asm volatile("" ::: "memory");
          __builtin_amdgcn_sched_barrier(0);
        }
        pearl_amd_v4i qa[G];
        if (lo) {
          if constexpr (PEARL_AMD_W11K_MASK != 3) PEARL_AMD_W11K_ALO();
#pragma unroll
          for (int i = 0; i < G; i++)
            qa[i] = *reinterpret_cast<const pearl_amd_v4i *>(aBase + (g0 + i) * 16 * S + 16 * j);
          if constexpr (PEARL_AMD_W11K_MASK != 3) PEARL_AMD_W11K_ALL();
        }
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (int i = 0; i < G; i++)
#pragma unroll
          for (int nt = 0; nt < NT; nt++)
            acc[g0 + i][nt] =
                __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(true, qa[i], true, bq[j][nt], acc[g0 + i][nt], false);
        __builtin_amdgcn_sched_barrier(0);
      }
    b_load(s + 1u);   // the next slab's B', after this slab's last WMMA
    __builtin_amdgcn_sched_barrier(0);
    if constexpr (decltype(last)::value) readout(s >> 2);
    __builtin_amdgcn_sched_barrier(0);
    stage_store(B ^ 1, pre);
    pearl_amd_lds_barrier();
  };
  (void)slab_step;
  (void)b_load;
#if PEARL_AMD_W11K_ROLL
  // PEARL_AMD_W11K_ROLL (above). B' tile nt of k-step j of a slab; k-step j's set is j & 1 (a slab is an even number
  // of k-steps).
  auto b_tile = [&](uint32_t slab, int j, int nt) -> pearl_amd_v4i {
    return __builtin_bit_cast(pearl_amd_v4i, __builtin_amdgcn_raw_buffer_load_b128(
        rsB, gbOff + nt * 16 * KS + 16 * j, (int)(slab * (uint32_t)(BN * KS)), 0));
  };
  static_assert(PEARL_AMD_W11K_MASK != 1, "PEARL_AMD_W11K_ROLL masks by exec itself (2, 3 or 4) or not at all (0)");
  auto roll_step = [&](uint32_t s, auto parity, auto last) {
    constexpr int B = decltype(parity)::value;
    pearl_amd_v4i pre;
    const int8_t *aBase = &sA[B][0] + rdOff;
#pragma unroll
    for (int j = 0; j < KJ; j++) {
      if ((j & 1) == 0) {   // a unit of slab s + 1's A': stored after the next k-step
        pre = stage_load(s + 1u, j / 2);
        asm volatile("" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
      }
#pragma unroll
      for (int g0 = 0; g0 < MT; g0 += G) {
        pearl_amd_v4i qa[G];
        if constexpr (PEARL_AMD_W11K_MASK == 3) {
#pragma unroll
          for (int i = 0; i < G; i++)
            qa[i] = *reinterpret_cast<const pearl_amd_v4i *>(aBase + (g0 + i) * 16 * S + 16 * j);
          __builtin_amdgcn_sched_barrier(0);
        }
        if constexpr (PEARL_AMD_W11K_MASK == 2 || PEARL_AMD_W11K_MASK == 4) {
          if (g0 == 0) PEARL_AMD_W11K_LO();
          else PEARL_AMD_W11K_ALO();
        } else if (g0 == 0 && PEARL_AMD_W11K_MASK == 3) PEARL_AMD_W11K_LO();
        if (g0 == 0) {   // the next k-step's B', into the set the last k-step used
#pragma unroll
          for (int nt = 0; nt < NT; nt++)
            bq[(j + 1) & 1][nt] = j + 1 < KJ ? b_tile(s, j + 1, nt) : b_tile(s + 1u, 0, nt);
          PEARL_AMD_W11K_A16();   // =4: from lanes 0-31 to A's 16
        }
        if constexpr (PEARL_AMD_W11K_MASK != 3) {
#pragma unroll
          for (int i = 0; i < G; i++)
            qa[i] = *reinterpret_cast<const pearl_amd_v4i *>(aBase + (g0 + i) * 16 * S + 16 * j);
        }
        if constexpr (PEARL_AMD_W11K_MASK == 2 || PEARL_AMD_W11K_MASK == 4) PEARL_AMD_W11K_ALL();
        else if (g0 == 0 && PEARL_AMD_W11K_MASK == 3) PEARL_AMD_W11K_ALL();
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (int i = 0; i < G; i++)
#pragma unroll
          for (int nt = 0; nt < NT; nt++)
            acc[g0 + i][nt] =
                __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(true, qa[i], true, bq[j & 1][nt], acc[g0 + i][nt], false);
        __builtin_amdgcn_sched_barrier(0);
      }
      if ((j & 1) == 1) {
        stage_store(B ^ 1, pre, j / 2);
        __builtin_amdgcn_sched_barrier(0);
      }
    }
    if constexpr (decltype(last)::value) readout(s / perChunk);
    __builtin_amdgcn_sched_barrier(0);
    pearl_amd_lds_barrier();
  };
#pragma unroll
  for (int u = 0; u < U; u++) stage_store(0, stage_load(0u, u), u);
  // The first k-step's B', and the second's as well, in every lane: the loop's loads then refresh lanes 0-31 and
  // leave operand bytes, not whatever the registers held, in the lanes the WMMA doesn't read.
#pragma unroll
  for (int nt = 0; nt < NT; nt++) { bq[0][nt] = b_tile(0u, 0, nt); bq[1][nt] = b_tile(0u, 1, nt); }
  __syncthreads();
  for (uint32_t s = 0; s < slabs; s += perChunk) {
    if constexpr (perChunk == 4u) {
      roll_step(s, std::integral_constant<int, 0>(), std::false_type());
      roll_step(s + 1, std::integral_constant<int, 1>(), std::false_type());
      roll_step(s + 2, std::integral_constant<int, 0>(), std::false_type());
      roll_step(s + 3, std::integral_constant<int, 1>(), std::true_type());
    } else {
      roll_step(s, std::integral_constant<int, 0>(), std::false_type());
      roll_step(s + 1, std::integral_constant<int, 1>(), std::true_type());
    }
  }
#else
  stage_store(0, stage_load(0u));
  b_load(0u);
  __syncthreads();
  for (uint32_t s = 0; s < slabs; s += 4) {
    slab_step(s, std::integral_constant<int, 0>(), std::false_type());
    slab_step(s + 1, std::integral_constant<int, 1>(), std::false_type());
    slab_step(s + 2, std::integral_constant<int, 0>(), std::false_type());
    slab_step(s + 3, std::integral_constant<int, 1>(), std::true_type());
  }
#endif
  __syncthreads();   // every lane's last word is in
  // A thread a region: it hashes and tests the region (PEARL_AMD_FUSE), and stores its words when the launch gave
  // somewhere to (the fold test's; the miner's fused launch gives none). Threads 0-127 in half the blocks and
  // 128-255 in the other half, as in the wave32 fold.
  const uint32_t j = (uint32_t)t ^ (((blockIdx.x ^ blockIdx.y) & 1u) << 7);
  if (j < (uint32_t)(RR * RC)) {
    pearl_amd_v4i tv[4];
#pragma unroll
    for (int u = 0; u < 4; u++) tv[u] = reinterpret_cast<const pearl_amd_v4i *>(sT + j * PEARL_JACKPOT_BUCKETS)[u];
    const uint32_t rowIdx = 2u * (tileRow / 32u) + j % (uint32_t)RR;
    const uint32_t colIdx = 4u * (tileCol / 64u) + j / (uint32_t)RR;
#if PEARL_AMD_FUSE
    uint32_t tm[16];
#pragma unroll
    for (int u = 0; u < 4; u++) {
      tm[4 * u] = (uint32_t)tv[u][0]; tm[4 * u + 1] = (uint32_t)tv[u][1];
      tm[4 * u + 2] = (uint32_t)tv[u][2]; tm[4 * u + 3] = (uint32_t)tv[u][3];
    }
    pearl_amd_hash_region(tm, colIdx * rows_valid + rowIdx, test, hits);
    if (tr != nullptr)
#endif
    {
      pearl_amd_v4i *const o =
          reinterpret_cast<pearl_amd_v4i *>(tr + ((size_t)colIdx * rows_valid + rowIdx) * PEARL_JACKPOT_BUCKETS);
#pragma unroll
      for (int u = 0; u < 4; u++) o[u] = tv[u];
    }
  }
}
}  // namespace
#endif  // PEARL_AMD_W11K
