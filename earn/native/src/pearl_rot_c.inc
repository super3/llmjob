// The rotation fold's chunk loop in C: the body of a loop function (const uint4 *offs, uint32_t chunks), for
// cards the hand-written loop (probes/hand89/rfold.py, sm_89 instructions) is not written for: RTX 50, sm_120.
// Included in two places, so that the probe and the miner run the same code: src/pearl_rot.cu (the miner's
// cubin for sm_120) and probes/hand89/gfold_h.cuh (GP_ROTC, the probe's kernel). The includer defines:
//   RC_MT      m16 tiles a warp (6)          RC_TR     the transcripts' place in the dynamic shared memory
//   RC_STG     the stage's place             RC_FILL4  the operand fill in each byte of a word (63)
//   RC_FLAGS   1, plus 2 (B a turn ahead), 4 (the readout spread over the mma), 8 (barriers after the first
//              and the last turn only), 16 (the readout spread over two turns: half a turn behind the sums) and
//              32 (the readout's steps written out, so ptxas keeps them as they are), 4096 (a turn's barrier goes
//              ahead of its last m16 tile's mma, behind that tile's load or store: the loads after the barrier then
//              go out while those eight mma run), 8192 (the first turn's A gather loads a tile ahead). TIMING ONLY, wrong results on purpose, to price a part: 64 (no
//              barriers), 128 (no readout), 256 (B gathered once a chunk), 512 (no stage loads), 1024 (no A gathers
//              or stores in the first turn), 2048 (no B gathers)
// It reads the 8-byte offsets records of rot_aux_offs (gprobe's `rot` 2).
//
// The fold. A warp takes a chunk's four k-steps in its own order (the records are in that order). At its first
// turn it gathers its k-step's A fragments, stores them to the stage and folds; at the other three it loads the
// fragments another warp of its band stored. B is gathered every turn. The four warps of a band meet at a
// barrier of their own after every k-step, so every load has a barrier between it and the store that wrote it
// and the store that overwrites it.
{
  constexpr uint32_t MT = RC_MT, NB = 8u;
  const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
  // Shared addresses: the offsets and the layout count from the start of the dynamic shared memory, which is not
  // address 0 on sm_90 and later (1 KB is reserved ahead of it).
  extern __shared__ __align__(128) uint32_t rc_smem[];
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(rc_smem);
  const uint32_t trL = sb + RC_TR + tid * 64u;
  const uint32_t band = warp >> 2, j = warp & 3u;
  const uint32_t stw = sb + RC_STG + band * (2048u * MT) + ((j + band) & 3u) * (512u * MT) + 16u * lane;
  auto ldmt = [](uint32_t r[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
  };
  auto gather = [&](uint32_t &f0, uint32_t &f1, uint32_t a) {
    uint32_t m[4];
    ldmt(m, a);
    const uint32_t x = m[0] + RC_FILL4 - m[2], y = m[1] + RC_FILL4 - m[3];
    f0 = __byte_perm(x, y, 0x6420u);
    f1 = __byte_perm(x, y, 0x7531u);
  };
  int32_t acc[MT][NB][4];
  uint32_t af[4], bf[NB][2];
#if RC_FLAGS & 8192
  uint32_t ma[8];
#endif
#pragma unroll
  for (uint32_t mb = 0; mb < MT; mb++)
#pragma unroll
    for (uint32_t nb = 0; nb < NB; nb++)
#pragma unroll
      for (uint32_t i = 0; i < 4; i++) acc[mb][nb][i] = 0;
  const uint2 *po = reinterpret_cast<const uint2 *>(offs) + (size_t)warp * ((size_t)chunks * 128u + 32u) + lane;
  auto gatherB = [&](const uint32_t P, const uint32_t ob) {
#if RC_FLAGS & 2048
    // TIMING ONLY: no B gathers (B stays what the prologue left: zero).
    if (P < 4u) return;
#endif
    gather(bf[2u * P][0], bf[2u * P + 1u][0], (ob & 0xffffu) + sb + 2048u * P);
    gather(bf[2u * P][1], bf[2u * P + 1u][1], (ob >> 16) + sb + 2048u * P);
  };
#if RC_FLAGS & 2
  // B a turn ahead. The tables never change in the loop, so the next turn's B pairs go out in the last m16
  // tile, each as soon as that tile's two mma that read the pair have been issued, ahead of the barrier; the
  // next record is loaded a turn ahead (a warp's records end with one spare).
  uint2 o = *po;
  po += 32;
#pragma unroll
  for (uint32_t P = 0; P < 4u; P++) gatherB(P, o.y);
#endif
  // The readout spread over the mma (flag 4). In a chunk's last turn a tile's sums are read out (two
  // three-input XOR a mma) while the next tile's mma go out; the last tile's in the next chunk's first turn,
  // during its first tile (which does not touch them), and that chunk's transcript word is written then. The
  // first chunk writes nothing real that way: the sums and the word are zero, and rotl(0) ^ 0 is 0.
  // Two chains of XOR (even and odd n8 tiles), so the readout is not one serial chain of 96 steps.
  uint32_t xp = 0u, xq = 0u;
  bool live = !(RC_FLAGS & 128);
  auto rd = [&](const uint32_t mb, const uint32_t nb) {
#if RC_FLAGS & 128
    // TIMING ONLY (wrong results on purpose): no readout in the loop, to price it; the sums are read once after
    // it, so the mma stay live.
    if (!live) return;
#endif
    uint32_t &x = (nb & 1u) ? xq : xp;
#if RC_FLAGS & 32
    // Flag 32: each step written out (lop3 0x96 is a three-input XOR), so ptxas keeps the chain in x and does
    // not regroup it into temporaries that must wait for registers the mma still read.
    asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(x) : "r"((uint32_t)acc[mb][nb][0]), "r"((uint32_t)acc[mb][nb][1]));
    asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(x) : "r"((uint32_t)acc[mb][nb][2]), "r"((uint32_t)acc[mb][nb][3]));
#else
    x ^= (uint32_t)acc[mb][nb][0] ^ (uint32_t)acc[mb][nb][1] ^ (uint32_t)acc[mb][nb][2] ^ (uint32_t)acc[mb][nb][3];
#endif
  };
  auto word = [&](const uint32_t c) {
#ifdef RC_TRL
    // RC_TRL (the 96 x 512 block, goal-nv50k): the transcript is in the kernel's frame, `trl`, the loop function's
    // third argument, so that the kernel keeps it there for the hand loop that replaces this one.
    (void)trL;
    trl[c & 15u] = __funnelshift_l(trl[c & 15u], trl[c & 15u], 13u) ^ xp ^ xq;
#else
    uint32_t w;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(w) : "r"(trL + (c & 15u) * 4u) : "memory");
    w = __funnelshift_l(w, w, 13u) ^ xp ^ xq;
    asm volatile("st.shared.u32 [%0], %1;" ::"r"(trL + (c & 15u) * 4u), "r"(w) : "memory");
#endif
    xp = 0u;
    xq = 0u;
  };
  // The band's barrier: 1 for warps 0 to 3, 2 for warps 4 to 7, 128 threads each. Flag 8: only after the first
  // turn (every store behind it, before any load) and the last (every load behind it, before the next chunk's
  // stores); nothing is stored in the turns between, so the two barriers there order nothing.
  auto bar = [&](const bool first, const bool last) {
    (void)first;
    (void)last;
#if RC_FLAGS & 64
    // TIMING ONLY (wrong results on purpose): no barriers, to price them.
    if (false)
#elif RC_FLAGS & 8
    if (first || last)
#endif
    asm volatile("{.reg .pred p; setp.ne.u32 p, %0, 0; @p bar.sync 2, 128; @!p bar.sync 1, 128;}" ::"r"(band) : "memory");
  };
  auto turn = [&](const bool first, const bool last, const uint32_t c) {
    (void)last;
    (void)c;
#if RC_FLAGS & 2
    const uint2 on = *po;
    po += 32;
#else
    const uint2 o = *po;
    po += 32;
#pragma unroll
    for (uint32_t P = 0; P < 4u; P++) gatherB(P, o.y);
#endif
#pragma unroll
    for (uint32_t mb = 0; mb < MT; mb++) {
      if (first) {
#if RC_FLAGS & 1024
        // TIMING ONLY: no A gathers or stores in the first turn (it folds the last A fragment it has).
        if (o.x == 0xffffffffu)
#endif
        {
#if RC_FLAGS & 8192
        // Flag 8192: a tile's two gather loads go out a tile ahead, before the mma of the tile before, and the
        // fragments are made from them when their tile comes.
        if (mb == 0u) {
          ldmt(ma, (o.x & 0xffffu) + sb);
          ldmt(ma + 4, (o.x >> 16) + sb);
        }
        {
          const uint32_t x0 = ma[0] + RC_FILL4 - ma[2], y0 = ma[1] + RC_FILL4 - ma[3];
          const uint32_t x1 = ma[4] + RC_FILL4 - ma[6], y1 = ma[5] + RC_FILL4 - ma[7];
          af[0] = __byte_perm(x0, y0, 0x6420u);
          af[1] = __byte_perm(x0, y0, 0x7531u);
          af[2] = __byte_perm(x1, y1, 0x6420u);
          af[3] = __byte_perm(x1, y1, 0x7531u);
        }
        if (mb + 1u < MT) {
          ldmt(ma, (o.x & 0xffffu) + sb + 2048u * (mb + 1u));
          ldmt(ma + 4, (o.x >> 16) + sb + 2048u * (mb + 1u));
        }
#else
        gather(af[0], af[1], (o.x & 0xffffu) + sb + 2048u * mb);
        gather(af[2], af[3], (o.x >> 16) + sb + 2048u * mb);
#endif
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};"
                     ::"r"(stw + 512u * mb), "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]) : "memory");
        }
      } else {
#if RC_FLAGS & 512
        // TIMING ONLY: no stage loads (later turns fold the last A fragment they have).
        if (mb == 0u && o.x == 0xffffffffu)
#endif
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(af[0]), "=r"(af[1]), "=r"(af[2]), "=r"(af[3]) : "r"(o.x + sb + 512u * mb) : "memory");
      }
#if RC_FLAGS & 4096
      // Flag 4096: the barrier behind the last tile's load or store and ahead of its mma.
      if (mb == MT - 1u) bar(first, last);
#endif
#pragma unroll
      for (uint32_t nb = 0; nb < NB; nb++) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9}, {%0,%1,%2,%3};"
            : "+r"(acc[mb][nb][0]), "+r"(acc[mb][nb][1]), "+r"(acc[mb][nb][2]), "+r"(acc[mb][nb][3])
            : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bf[nb][0]), "r"(bf[nb][1]));
#if RC_FLAGS & 2
#if RC_FLAGS & 256
        // TIMING ONLY: B gathered for the next chunk's first turn only (turns 1 to 3 fold the B they have).
        if (last && mb == MT - 1u && (nb & 1u)) gatherB(nb >> 1, on.y);
#else
        if (mb == MT - 1u && (nb & 1u)) gatherB(nb >> 1, on.y);
#endif
#endif
#if RC_FLAGS & 16
        // Flag 16: sum i (= 8 mb + nb) of a chunk is final at its mma in the last turn and is next changed at its
        // mma in the next chunk's first turn, a turn later. It is read out half a turn after it is final: sums 0
        // to 23 in the last turn's second half, 24 to 47 in the next first turn's first half, one sum a mma.
        {
          const uint32_t i = 8u * mb + nb, h = 4u * MT;
          if (last && i >= h) rd((i - h) >> 3, (i - h) & 7u);
          if (first && i < h) rd((i + h) >> 3, (i + h) & 7u);
          if (first && i == h - 1u) word(c + 15u);
        }
#elif RC_FLAGS & 4
        if (last && mb > 0u) rd(mb - 1u, nb);
        if (first && mb == 0u) rd(MT - 1u, nb);
#endif
      }
#if (RC_FLAGS & 4) && !(RC_FLAGS & 16)
      if (first && mb == 0u) word(c + 15u);
#endif
    }
#if !(RC_FLAGS & 4096)
    bar(first, last);
#endif
#if RC_FLAGS & 2
    o = on;
#endif
  };
  for (uint32_t c = 0; c < chunks; c++) {
    turn(true, false, c);
    turn(false, false, c);
    turn(false, false, c);
    turn(false, true, c);
#if RC_FLAGS & 16
  }
  // The last chunk's sums 24 to 47, and its word.
#pragma unroll
  for (uint32_t i = 4u * MT; i < 8u * MT; i++) rd(i >> 3, i & 7u);
  word(chunks - 1u);
  {
#elif RC_FLAGS & 4
  }
  // The last chunk's last tile, and its word.
#pragma unroll
  for (uint32_t nb = 0; nb < NB; nb++) rd(MT - 1u, nb);
  word(chunks - 1u);
  {
#else
#pragma unroll
    for (uint32_t mb = 0; mb < MT; mb++)
#pragma unroll
      for (uint32_t nb = 0; nb < NB; nb++) rd(mb, nb);
    word(c);
#endif
  }
#if RC_FLAGS & 16384
  // Flag 16384: room. 128 stores that never run (no record count is 2^32 - 1), so that the compiled function is
  // long enough for probes/hand89/rfold120.py to put the hand-written loop in its place. Not for the miner.
  if (chunks == 0xffffffffu) {
#pragma unroll
    for (uint32_t i = 0; i < 128u; i++)
      asm volatile("st.shared.u32 [%0], %1;" ::"r"(trL + 4u * (i & 15u)), "r"(i * 0x9e3779b9u) : "memory");
#if RC_FLAGS & 32768
    // Flag 32768 (goal-nv50k): a barrier numbered 7 that never runs, so that ptxas gives the kernel eight barriers
    // and a hand loop may use 0 to 7 (an sm_120 kernel compiled with three faults on a fourth: illegal instruction).
    asm volatile("bar.sync 7, 256;" ::: "memory");
#endif
  }
#endif
#if RC_FLAGS & 128
  live = true;
#pragma unroll
  for (uint32_t mb = 0; mb < MT; mb++)
#pragma unroll
    for (uint32_t nb = 0; nb < NB; nb++) rd(mb, nb);
  word(0u);
#endif
}
