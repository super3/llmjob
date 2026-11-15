// The rotation fold's chunk loop for RTX 20 (sm_75, pearl_rot.cu's ROT_T75): the body of the loop function
// (const uint4 *offs, uint32_t chunks, uint32_t *trl). Turing has no m16n8k32 mma, so each one of the other cards'
// loop is four m8n8k16 mma here, on the same fragments: ldmatrix's gather gives a lane the rows g and g + 8 of a
// 16-row group at four k each (f0, f1), which are the A fragments of two m8n8k16 mma; B's two k halves are two B
// fragments; the sums of rows g and g + 8 (c0 c1, c2 c3) are the sums of the two m8 tiles. So the tile, its sums
// and the transcript are the same as on RTX 30 and 40, and the host drives it the same way.
//
// What is different: Turing has 64 KB of shared memory, which holds the tables (A's 12 groups, B's 16: 56 KB) and
// no stage, so every warp gathers its own A every turn (the gather fold) and there are no barriers; and the
// transcript lives in the kernel's frame (trl). Records are rot_aux_offs's per-warp form with A's two offsets in
// every turn (ROT_T75 there).
//
// RT_FLAGS (pearl_rot.cu's ROT_T75 value): 1 the plain loop; 2 B a turn ahead (gathered in the last m16 tile, as
// RC_FLAGS 2); 4 the readout spread over the next chunk's first turn (as RC_FLAGS 4); 8192 A's two gather loads a
// tile ahead (as RC_FLAGS 8192). TIMING ONLY, wrong results on purpose: 128 no readout, 2048 no B gathers, 1024 no
// A gathers, 4096 B's gathers at addresses with no bank conflict, 16384 A's the same. ROT_MT (m16 tiles a warp) is 6 (the RTX 30/40 tile, 12 x 16) or 4 (8 x 16, a 128 x 256 block: 128
// sums a thread, which leaves Turing's 255 registers room to load ahead).
{
  constexpr uint32_t MT = ROT_MT, NB = 8u;
  const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
  extern __shared__ __align__(128) uint32_t rt_smem[];
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(rt_smem);
  auto ldmt = [](uint32_t r[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
  };
  auto gather = [&](uint32_t &f0, uint32_t &f1, uint32_t a) {
    uint32_t m[4];
    ldmt(m, a);
    const uint32_t x = m[0] + ROT_FILL4 - m[2], y = m[1] + ROT_FILL4 - m[3];
    f0 = __byte_perm(x, y, 0x6420u);
    f1 = __byte_perm(x, y, 0x7531u);
  };
  auto imma = [](int32_t &c0, int32_t &c1, uint32_t a, uint32_t b) {
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
        : "+r"(c0), "+r"(c1) : "r"(a), "r"(b));
  };
  int32_t acc[MT][NB][4];
  uint32_t af[4], bf[NB][2];
#if ROT_T75 & 8192
  uint32_t ma[8];
#endif
#pragma unroll
  for (uint32_t mb = 0; mb < MT; mb++)
#pragma unroll
    for (uint32_t nb = 0; nb < NB; nb++)
#pragma unroll
      for (uint32_t i = 0; i < 4; i++) acc[mb][nb][i] = 0;
#pragma unroll
  for (uint32_t nb = 0; nb < NB; nb++) bf[nb][0] = bf[nb][1] = 0u;
  const uint2 *po = reinterpret_cast<const uint2 *>(offs) + (size_t)warp * ((size_t)chunks * 128u + 32u) + lane;
  // TIMING ONLY (4096, 16384): a record's offsets moved to the lane's own row of its table group, so the eight rows
  // a matrix reads sit in eight different bank groups.
  auto cf = [&](const uint32_t v) {
    const uint32_t r = 16u * lane;
    return ((v & 0xf800u) | r) | (((v >> 16) & 0xf800u) | (r + 1024u)) << 16;
  };
  (void)cf;
  auto gatherB = [&](const uint32_t P, const uint32_t ob) {
#if ROT_T75 & 2048
    if (P < 4u) return;
#endif
#if ROT_T75 & 4096
    const uint32_t obc = cf(ob);
#define RT_OB obc
#else
#define RT_OB ob
#endif
    gather(bf[2u * P][0], bf[2u * P + 1u][0], (RT_OB & 0xffffu) + sb + 2048u * P);
    gather(bf[2u * P][1], bf[2u * P + 1u][1], (RT_OB >> 16) + sb + 2048u * P);
#undef RT_OB
  };
#if ROT_T75 & 2
  uint2 o = *po;
  po += 32;
#pragma unroll
  for (uint32_t P = 0; P < 4u; P++) gatherB(P, o.y);
#endif
  uint32_t xp = 0u, xq = 0u;
  auto rd = [&](const uint32_t mb, const uint32_t nb) {
#if ROT_T75 & 128
    (void)mb;
    (void)nb;
#else
    uint32_t &x = (nb & 1u) ? xq : xp;
    x ^= (uint32_t)acc[mb][nb][0] ^ (uint32_t)acc[mb][nb][1] ^ (uint32_t)acc[mb][nb][2] ^ (uint32_t)acc[mb][nb][3];
#endif
  };
  auto word = [&](const uint32_t c) {
    trl[c & 15u] = __funnelshift_l(trl[c & 15u], trl[c & 15u], 13u) ^ xp ^ xq;
    xp = 0u;
    xq = 0u;
  };
  auto turn = [&](const bool first, const bool last, const uint32_t c) {
    (void)first;
    (void)last;
    (void)c;
#if ROT_T75 & 2
    const uint2 on = *po;
    po += 32;
#elif ROT_T75 & 16384
    uint2 o = *po;
    po += 32;
    o.x = cf(o.x);
#pragma unroll
    for (uint32_t P = 0; P < 4u; P++) gatherB(P, o.y);
#else
    const uint2 o = *po;
    po += 32;
#pragma unroll
    for (uint32_t P = 0; P < 4u; P++) gatherB(P, o.y);
#endif
#pragma unroll
    for (uint32_t mb = 0; mb < MT; mb++) {
#if ROT_T75 & 1024
      if (mb == 0u && o.x == 0xffffffffu)
#endif
      {
#if ROT_T75 & 8192
        if (mb == 0u) {
          ldmt(ma, (o.x & 0xffffu) + sb);
          ldmt(ma + 4, (o.x >> 16) + sb);
        }
        {
          const uint32_t x0 = ma[0] + ROT_FILL4 - ma[2], y0 = ma[1] + ROT_FILL4 - ma[3];
          const uint32_t x1 = ma[4] + ROT_FILL4 - ma[6], y1 = ma[5] + ROT_FILL4 - ma[7];
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
      }
      // The k-lo half of all eight n8 tiles, then the k-hi half: 16 mma between two that add into one sum.
#pragma unroll
      for (uint32_t nb = 0; nb < NB; nb++) {
        imma(acc[mb][nb][0], acc[mb][nb][1], af[0], bf[nb][0]);
        imma(acc[mb][nb][2], acc[mb][nb][3], af[1], bf[nb][0]);
      }
#pragma unroll
      for (uint32_t nb = 0; nb < NB; nb++) {
        imma(acc[mb][nb][0], acc[mb][nb][1], af[2], bf[nb][1]);
        imma(acc[mb][nb][2], acc[mb][nb][3], af[3], bf[nb][1]);
#if ROT_T75 & 2
        if (mb == MT - 1u && (nb & 1u)) gatherB(nb >> 1, on.y);
#endif
#if ROT_T75 & 4
        if (last && mb > 0u) rd(mb - 1u, nb);
        if (first && mb == 0u) rd(MT - 1u, nb);
#endif
      }
#if ROT_T75 & 4
      if (first && mb == 0u) word(c + 15u);
#endif
    }
#if ROT_T75 & 2
    o = on;
#endif
  };
  for (uint32_t c = 0; c < chunks; c++) {
    turn(true, false, c);
    turn(false, false, c);
    turn(false, false, c);
    turn(false, true, c);
#if ROT_T75 & 4
  }
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
#if ROT_T75 & 128
#pragma unroll
  for (uint32_t mb = 0; mb < MT; mb++)
#pragma unroll
    for (uint32_t nb = 0; nb < NB; nb++)
      xp ^= (uint32_t)acc[mb][nb][0] ^ (uint32_t)acc[mb][nb][1] ^ (uint32_t)acc[mb][nb][2] ^ (uint32_t)acc[mb][nb][3];
  word(0u);
#endif
}
