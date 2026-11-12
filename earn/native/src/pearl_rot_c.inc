// The rotation fold's chunk loop in C: the body of a loop function (const uint4 *offs, uint32_t chunks), for
// cards the hand-written loop (probes/hand89/rfold.py, sm_89 instructions) is not written for: RTX 50, sm_120.
// Included in two places, so that the probe and the miner run the same code: src/pearl_rot.cu (the miner's
// cubin for sm_120) and probes/hand89/gfold_h.cuh (GP_ROTC, the probe's kernel). The includer defines:
//   RC_MT      m16 tiles a warp (6)          RC_TR     the transcripts' place in the dynamic shared memory
//   RC_STG     the stage's place             RC_FILL4  the operand fill in each byte of a word (63)
//   RC_FLAGS   1, plus 2 (B a turn ahead) and 4 (the readout spread over the mma)
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
#pragma unroll
  for (uint32_t mb = 0; mb < MT; mb++)
#pragma unroll
    for (uint32_t nb = 0; nb < NB; nb++)
#pragma unroll
      for (uint32_t i = 0; i < 4; i++) acc[mb][nb][i] = 0;
  const uint2 *po = reinterpret_cast<const uint2 *>(offs) + (size_t)warp * ((size_t)chunks * 128u + 32u) + lane;
  auto gatherB = [&](const uint32_t P, const uint32_t ob) {
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
  uint32_t xp = 0u;
  auto rd = [&](const uint32_t mb, const uint32_t nb) {
    xp ^= (uint32_t)acc[mb][nb][0] ^ (uint32_t)acc[mb][nb][1] ^ (uint32_t)acc[mb][nb][2] ^ (uint32_t)acc[mb][nb][3];
  };
  auto word = [&](const uint32_t c) {
    uint32_t w;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(w) : "r"(trL + (c & 15u) * 4u) : "memory");
    w = __funnelshift_l(w, w, 13u) ^ xp;
    asm volatile("st.shared.u32 [%0], %1;" ::"r"(trL + (c & 15u) * 4u), "r"(w) : "memory");
    xp = 0u;
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
        gather(af[0], af[1], (o.x & 0xffffu) + sb + 2048u * mb);
        gather(af[2], af[3], (o.x >> 16) + sb + 2048u * mb);
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};"
                     ::"r"(stw + 512u * mb), "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]) : "memory");
      } else {
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(af[0]), "=r"(af[1]), "=r"(af[2]), "=r"(af[3]) : "r"(o.x + sb + 512u * mb) : "memory");
      }
#pragma unroll
      for (uint32_t nb = 0; nb < NB; nb++) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9}, {%0,%1,%2,%3};"
            : "+r"(acc[mb][nb][0]), "+r"(acc[mb][nb][1]), "+r"(acc[mb][nb][2]), "+r"(acc[mb][nb][3])
            : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bf[nb][0]), "r"(bf[nb][1]));
#if RC_FLAGS & 2
        if (mb == MT - 1u && (nb & 1u)) gatherB(nb >> 1, on.y);
#endif
#if RC_FLAGS & 4
        if (last && mb > 0u) rd(mb - 1u, nb);
        if (first && mb == 0u) rd(MT - 1u, nb);
#endif
      }
#if RC_FLAGS & 4
      if (first && mb == 0u) word(c + 15u);
#endif
    }
    // The band's barrier: 1 for warps 0 to 3, 2 for warps 4 to 7, 128 threads each.
    asm volatile("{.reg .pred p; setp.ne.u32 p, %0, 0; @p bar.sync 2, 128; @!p bar.sync 1, 128;}" ::"r"(band) : "memory");
#if RC_FLAGS & 2
    o = on;
#endif
  };
  for (uint32_t c = 0; c < chunks; c++) {
    turn(true, false, c);
    turn(false, false, c);
    turn(false, false, c);
    turn(false, true, c);
#if RC_FLAGS & 4
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
}
