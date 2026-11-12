// Timing variants of the fragment fold (goal-arc5, 12 November 2026). Included by
// pearl_sycl_kernels.hpp only in a build with PEARL_SYCL_EXTRA=-DPEARL_A5=N; no release has them.
// Each N is one shape raced against the shipped fold; README.md has the results.
#ifndef PEARL_SYCL_A5_HPP
#define PEARL_SYCL_A5_HPP

#ifndef PEARL_A5_STAG
#define PEARL_A5_STAG 0u
#endif
#ifndef PEARL_A5_ALLOC
#if defined(PEARL_A5) && (PEARL_A5 == 2 || PEARL_A5 == 3)
#define PEARL_A5_ALLOC 17
#else
#define PEARL_A5_ALLOC 16
#endif
#endif
#if defined(PEARL_A5) && (PEARL_A5 == 2 || PEARL_A5 == 3)
constexpr uint32_t kPW = 17;
#else
constexpr uint32_t kPW = 16;
#endif

template <int SG, int RB, bool HW> struct FoldXmxFA5 {
  static_assert(SG == 8, "the fragment layouts are sub-group 8's");
  static constexpr int S = 4 / RB;     // sub-groups a window
  static constexpr int NCB = 64 / SG;  // DPAS column blocks
  FoldArgs A;
  sycl::local_accessor<uint32_t, 1> part;

  // The sub-group size, as a kernel property since the kernel has another:
  // the hardware kernel with 16 accumulators needs the 256-register mode. The
  // reference kernel is left alone: the CPU device has no such mode.
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    namespace syclex = sycl::ext::oneapi::experimental;
    if constexpr (HW && RB > 1)
      return syclex::properties{syclex::sub_group_size<SG>, sycl::ext::intel::experimental::grf_size<256>};
    else
      return syclex::properties{syclex::sub_group_size<SG>};
  }

  void operator()(sycl::nd_item<1> it) const {
    sycl::sub_group sg = it.get_sub_group();
    const uint32_t lane = (uint32_t)sg.get_local_linear_id();
    const uint32_t sgi = (uint32_t)sg.get_group_linear_id();
    const uint32_t rowWins = A.rowsValid / 2u, colWins = A.colBatch / 4u;
    uint32_t wr, wc;
    window_of((uint32_t)it.get_group(0), rowWins, colWins, A.band, wr, wc);
    // The sub-group's rows in their 16-row block of A' and the window's block
    // of B', each at this lane's 32 bytes of a fragment.
    const uint32_t row0 = 32u * wr + sgi * 8u * RB;
    const int8_t *Ab = A.Ap + (size_t)(row0 >> 4) * A.aStride + (size_t)((row0 >> 3) & 1u) * 256u + 32u * lane;
    const int8_t *Bb = A.Bp + (size_t)(A.colStart / 4u + wc) * A.bStride + 32u * lane;
    v8i c[RB][NCB];
    each<RB>([&](int rb) { each<NCB>([&](int cb) { c[rb][cb] = 0; }); });
#if defined(PEARL_A5) && PEARL_A5 == 12
    // Two row-adjacent windows a work-group of 4 sub-groups, each sub-group a
    // 32 by 32 tile: sub-group sgi takes window 2 wp + (sgi >> 1) and its
    // column half sgi & 1. A' is shared by sub-groups 0,1 and 2,3, B' by 0,2
    // and 1,3. The words go into 1 KB of local memory with local XOR atomics
    // (two windows of 8 regions by 16 words), so the group stays at 1 KB.
    {
      uint32_t wp, wc2;
      window_of((uint32_t)it.get_group(0), rowWins / 2u, colWins, A.band > 1u ? A.band / 2u : 1u, wp, wc2);
      const uint32_t w2 = 2u * wp + (sgi >> 1);
      const uint32_t li = (uint32_t)it.get_local_linear_id();
      for (uint32_t i = li; i < 256u; i += 32u) part[i] = 0u;
      const int8_t *A2 = A.Ap + (size_t)w2 * 2u * A.aStride + 32u * lane;
      const int8_t *B2 = A.Bp + (size_t)(A.colStart / 4u + wc2) * A.bStride + 32u * lane + (size_t)(sgi & 1u) * 1024u;
      sycl::group_barrier(it.get_group());
      v8i d[4][4];
      each<4>([&](int rb) { each<4>([&](int cb) { d[rb][cb] = 0; }); });
      for (uint32_t ch = 0; ch < 16u; ch++) {
        for (uint32_t s = 0; s < 4u; s++) {
          const uint32_t ks = ch * 4u + ((s + PEARL_A5_STAG * (sgi & 1u)) & 3u);
          v8i af[4], bf[4];
          each<4>([&](int rb) {
            af[rb] = load_halves(A2 + (size_t)(rb >> 1) * A.aStride + (size_t)ks * 512u + (size_t)(rb & 1) * 256u);
          });
          each<4>([&](int cb) { bf[cb] = load_halves(B2 + (size_t)ks * 2048u + (size_t)cb * 256u); });
          int wait = 0;
          each<4>([&](int rb) { wait ^= af[rb][0]; });
          each<4>([&](int cb) { wait ^= bf[cb][0]; });
          wait &= (int)A.zero;
#ifndef PEARL_A5_NOWAIT
          each<4>([&](int rb) { af[rb][0] |= wait; });
#endif
          each<4>([&](int cb) {
            each<4>([&](int rb) { d[rb][cb] = dpas<SG, HW>(sg, af[rb], bf[cb], d[rb][cb]); });
          });
        }
        uint32_t x0 = 0, x1 = 0;
        each<4>([&](int rb) {
          each<4>([&](int cb) {
            const v8i v = d[rb][cb];
            x0 ^= (uint32_t)(v[0] ^ v[1] ^ v[2] ^ v[3]);
            x1 ^= (uint32_t)(v[4] ^ v[5] ^ v[6] ^ v[7]);
          });
        });
        x0 ^= sycl::permute_group_by_xor(sg, x0, 1u);
        x1 ^= sycl::permute_group_by_xor(sg, x1, 1u);
        if (lane < 8u) {
          sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                           sycl::access::address_space::local_space>
              at(part[((sgi >> 1) * 8u + lane) * 16u + ch]);
          at.fetch_xor((lane & 1u) ? x1 : x0);
        }
      }
      sycl::group_barrier(it.get_group());
      if ((sgi & 1u) == 0 && lane < 8u) {
        const uint32_t qr = lane;
        uint32_t words[16];
        for (int j = 0; j < 16; j++) words[j] = part[((sgi >> 1) * 8u + qr) * 16u + (uint32_t)j];
        const uint32_t local = (2u * w2 + (qr & 1u)) + A.rowsValid * (4u * wc2 + (qr >> 1));
        region_epilogue(words, local, A.test, A.hits, A.allTr);
      }
      return;
    }
#elif defined(PEARL_A5) && PEARL_A5 == 13
    // PEARL_A5_WPG windows a work-group, all in one column window (consecutive
    // groups of the band walk), each window two sub-groups of 16 by 64 as in
    // the built fold: B' is read by 2 WPG sub-groups in step. Each lane keeps
    // its 16 words in registers; at the end the windows are combined two at a
    // time through 1 KB of local memory.
    {
      const uint32_t j = sgi >> 1, h = sgi & 1u;
      uint32_t w3, wc3;
      window_of((uint32_t)it.get_group(0) * PEARL_A5_WPG + j, rowWins, colWins, A.band, w3, wc3);
      const uint32_t r0 = 32u * w3 + h * 16u;
      const int8_t *A3 = A.Ap + (size_t)(r0 >> 4) * A.aStride + 32u * lane;
      const int8_t *B3 = A.Bp + (size_t)(A.colStart / 4u + wc3) * A.bStride + 32u * lane;
      uint32_t t[16];
      each<16>([&](int q) { t[q] = 0; });
      for (uint32_t ch = 0; ch < 16u; ch++) {
        for (uint32_t s = 0; s < 4u; s++) {
          const uint32_t ks = ch * 4u + s;
          v8i af[RB], bf[NCB];
          each<RB>([&](int rb) {
            af[rb] = *reinterpret_cast<const v8iu *>(A3 + (size_t)ks * 512u + (size_t)rb * 256u);
          });
          each<NCB>([&](int cb) { bf[cb] = load_halves(B3 + (size_t)ks * 2048u + (size_t)cb * 256u); });
          int wait = 0;
          each<RB>([&](int rb) { wait ^= af[rb][0]; });
          each<NCB>([&](int cb) { wait ^= bf[cb][0]; });
          wait &= (int)A.zero;
          each<RB>([&](int rb) { af[rb][0] |= wait; });
          each<NCB>([&](int cb) {
            each<RB>([&](int rb) { c[rb][cb] = dpas<SG, HW>(sg, af[rb], bf[cb], c[rb][cb]); });
          });
        }
        uint32_t x0 = 0, x1 = 0;
        each<RB>([&](int rb) {
          each<NCB>([&](int cb) {
            const v8i v = c[rb][cb];
            x0 ^= (uint32_t)(v[0] ^ v[1] ^ v[2] ^ v[3]);
            x1 ^= (uint32_t)(v[4] ^ v[5] ^ v[6] ^ v[7]);
          });
        });
        x0 ^= sycl::permute_group_by_xor(sg, x0, 1u);
        x1 ^= sycl::permute_group_by_xor(sg, x1, 1u);
        each<15>([&](int q) { t[q] = t[q + 1]; });
        t[15] = (lane & 1u) ? x1 : x0;
      }
      for (uint32_t r = 0; r < PEARL_A5_WPG / 2u; r++) {
        const bool mine = (j >> 1) == r;
        if (mine && h == 1u && lane < 8u)
          each<16>([&](int q) { part[((j & 1u) * 8u + lane) * 16u + (uint32_t)q] = t[q]; });
        sycl::group_barrier(it.get_group());
        if (mine && h == 0u && lane < 8u) {
          uint32_t words[16];
          each<16>([&](int q) { words[q] = t[q] ^ part[((j & 1u) * 8u + lane) * 16u + (uint32_t)q]; });
          const uint32_t qr = lane;
          const uint32_t local = (2u * w3 + (qr & 1u)) + A.rowsValid * (4u * wc3 + (qr >> 1));
          region_epilogue(words, local, A.test, A.hits, A.allTr);
        }
        sycl::group_barrier(it.get_group());
      }
      return;
    }
#elif defined(PEARL_A5) && PEARL_A5 > 0
    // Timing variants (goal-arc5). 1: the chunk's 4 k-steps written out, so no
    // loop is left inside the chunk loop. 2: as 1, and chunk ch's readout sits
    // in chunk ch+1's first k-step, after its loads are sent and before its
    // DPAS, so the XORs run while the loads are in flight. 3: as 2, two chunks
    // an iteration. Chunk j's word lands at column j + 1 of a 17-word row
    // (column 0 takes the empty readout before chunk 0).
    auto kstep_load = [&](uint32_t ks, v8i (&af)[RB], v8i (&bf)[NCB]) {
      each<RB>([&](int rb) {
        af[rb] = *reinterpret_cast<const v8iu *>(Ab + (size_t)ks * 512u + (size_t)rb * 256u);
      });
      each<NCB>([&](int cb) { bf[cb] = load_halves(Bb + (size_t)ks * 2048u + (size_t)cb * 256u); });
    };
    auto kstep_mma = [&](v8i (&af)[RB], v8i (&bf)[NCB]) {
      int wait = 0;
      each<RB>([&](int rb) { wait ^= af[rb][0]; });
      each<NCB>([&](int cb) { wait ^= bf[cb][0]; });
      wait &= (int)A.zero;
      each<RB>([&](int rb) { af[rb][0] |= wait; });
      each<NCB>([&](int cb) {
        each<RB>([&](int rb) { c[rb][cb] = dpas<SG, HW>(sg, af[rb], bf[cb], c[rb][cb]); });
      });
    };
    auto readout = [&](uint32_t col) {
      uint32_t x0 = 0, x1 = 0;
      each<RB>([&](int rb) {
        each<NCB>([&](int cb) {
          const v8i v = c[rb][cb];
          x0 ^= (uint32_t)(v[0] ^ v[1] ^ v[2] ^ v[3]);
          x1 ^= (uint32_t)(v[4] ^ v[5] ^ v[6] ^ v[7]);
        });
      });
      x0 ^= sycl::permute_group_by_xor(sg, x0, 1u);
      x1 ^= sycl::permute_group_by_xor(sg, x1, 1u);
      if (lane < 8u) part[(sgi * 8u + lane) * kPW + col] = (lane & 1u) ? x1 : x0;
    };
#if PEARL_A5 == 8
    // A 32 by 32 tile a sub-group: all 32 rows of the window (both 16-row
    // blocks of A') by sub-group sgi's 32 columns (B' fragments 4 sgi .. 4 sgi
    // + 3). 16 accumulators as before, 4 + 4 fragments a k-step (2,048 bytes)
    // in place of 2 + 8 (2,560). The readout is unchanged: a lane's column is
    // still 8 cb + lane and rows 0-3 / 4-7 of each 8-row block.
    {
      const int8_t *A2 = A.Ap + (size_t)wr * 2u * A.aStride + 32u * lane;
      const int8_t *B2 = Bb + (size_t)sgi * 1024u;
      v8i d[4][4];
      each<4>([&](int rb) { each<4>([&](int cb) { d[rb][cb] = 0; }); });
      for (uint32_t ch = 0; ch < 16u; ch++) {
        for (uint32_t s = 0; s < 4u; s++) {
          const uint32_t ks = ch * 4u + ((s + PEARL_A5_STAG * sgi) & 3u);
          v8i af[4], bf[4];
          each<4>([&](int rb) {
#if defined(PEARL_A5_AH)
            af[rb] = load_halves(A2 + (size_t)(rb >> 1) * A.aStride + (size_t)ks * 512u + (size_t)(rb & 1) * 256u);
#else
            af[rb] = *reinterpret_cast<const v8iu *>(A2 + (size_t)(rb >> 1) * A.aStride + (size_t)ks * 512u
                                                     + (size_t)(rb & 1) * 256u);
#endif
          });
          each<4>([&](int cb) { bf[cb] = load_halves(B2 + (size_t)ks * 2048u + (size_t)cb * 256u); });
          int wait = 0;
          each<4>([&](int rb) { wait ^= af[rb][0]; });
          each<4>([&](int cb) { wait ^= bf[cb][0]; });
          wait &= (int)A.zero;
#ifndef PEARL_A5_NOWAIT
          each<4>([&](int rb) { af[rb][0] |= wait; });
#endif
          each<4>([&](int cb) {
            each<4>([&](int rb) { d[rb][cb] = dpas<SG, HW>(sg, af[rb], bf[cb], d[rb][cb]); });
          });
        }
        uint32_t x0 = 0, x1 = 0;
        each<4>([&](int rb) {
          each<4>([&](int cb) {
            const v8i v = d[rb][cb];
            x0 ^= (uint32_t)(v[0] ^ v[1] ^ v[2] ^ v[3]);
            x1 ^= (uint32_t)(v[4] ^ v[5] ^ v[6] ^ v[7]);
          });
        });
        x0 ^= sycl::permute_group_by_xor(sg, x0, 1u);
        x1 ^= sycl::permute_group_by_xor(sg, x1, 1u);
        if (lane < 8u) part[(sgi * 8u + lane) * 16u + ch] = (lane & 1u) ? x1 : x0;
      }
    }
#elif PEARL_A5 == 7
    // Timing only: no local memory and no barrier. Each lane keeps its 16
    // words in registers (a shift, so indices stay constant) and every
    // sub-group hashes its own half: wrong transcripts on purpose.
    uint32_t t[16];
    each<16>([&](int j) { t[j] = 0; });
    for (uint32_t ch = 0; ch < 16u; ch++) {
      for (uint32_t s = 0; s < 4u; s++) {
        v8i af[RB], bf[NCB];
        kstep_load(ch * 4u + s, af, bf);
        kstep_mma(af, bf);
      }
      uint32_t x0 = 0, x1 = 0;
      each<RB>([&](int rb) {
        each<NCB>([&](int cb) {
          const v8i v = c[rb][cb];
          x0 ^= (uint32_t)(v[0] ^ v[1] ^ v[2] ^ v[3]);
          x1 ^= (uint32_t)(v[4] ^ v[5] ^ v[6] ^ v[7]);
        });
      });
      x0 ^= sycl::permute_group_by_xor(sg, x0, 1u);
      x1 ^= sycl::permute_group_by_xor(sg, x1, 1u);
      each<15>([&](int j) { t[j] = t[j + 1]; });
      t[15] = (lane & 1u) ? x1 : x0;
    }
    if (lane < 8u) {
      const uint32_t qr = lane;
      const uint32_t local = (2u * wr + (qr & 1u)) + A.rowsValid * (4u * wc + (qr >> 1));
      region_epilogue(t, local, A.test, A.hits, A.allTr);
    }
    return;
#elif PEARL_A5 == 1
    // PEARL_A5_CPI chunks an iteration (16: no loop at all).
#ifndef PEARL_A5_CPI
#define PEARL_A5_CPI 1
#endif
    for (uint32_t ch0 = 0; ch0 < 16u; ch0 += PEARL_A5_CPI) {
      each<PEARL_A5_CPI>([&](int cc) {
        const uint32_t ch = ch0 + (uint32_t)cc;
        each<4>([&](int s) {
          v8i af[RB], bf[NCB];
          kstep_load(ch * 4u + (uint32_t)s, af, bf);
          kstep_mma(af, bf);
        });
        readout(ch);
      });
    }
#else
    constexpr int CPI = PEARL_A5 == 3 ? 2 : 1;  // chunks an iteration
    for (uint32_t ch = 0; ch < 16u; ch += CPI) {
      each<4 * CPI>([&](int s) {
        v8i af[RB], bf[NCB];
        kstep_load(ch * 4u + (uint32_t)s, af, bf);
        if ((s & 3) == 0) readout(ch + (uint32_t)(s >> 2));
        kstep_mma(af, bf);
      });
    }
    readout(16u);
#endif
    sycl::group_barrier(it.get_group());
    if (sgi == 0 && lane < 8u) {
      const uint32_t qr = lane;
      uint32_t words[16];
      for (int j = 0; j < 16; j++) {
        uint32_t x = 0;
        for (int s = 0; s < S; s++) x ^= part[((uint32_t)s * 8u + qr) * kPW + (uint32_t)j + (kPW - 16u)];
        words[j] = x;
      }
      const uint32_t local = (2u * wr + (qr & 1u)) + A.rowsValid * (4u * wc + (qr >> 1));
      region_epilogue(words, local, A.test, A.hits, A.allTr);
    }
    return;
#endif
    for (uint32_t ch = 0; ch < 16u; ch++) {
      for (uint32_t s = 0; s < 4u; s++) {
        const uint32_t ks = ch * 4u + ((s + PEARL_A5_STAG * sgi) & 3u);
        v8i af[RB], bf[NCB];
        each<RB>([&](int rb) {
          af[rb] = *reinterpret_cast<const v8iu *>(Ab + (size_t)ks * 512u + (size_t)rb * 256u);
        });
        each<NCB>([&](int cb) { bf[cb] = load_halves(Bb + (size_t)ks * 2048u + (size_t)cb * 256u); });
        // Every DPAS of the k-step waits for every load of it.
        int wait = 0;
        each<RB>([&](int rb) { wait ^= af[rb][0]; });
        each<NCB>([&](int cb) { wait ^= bf[cb][0]; });
        wait &= (int)A.zero;
        each<RB>([&](int rb) { af[rb][0] |= wait; });
        each<NCB>([&](int cb) {
          each<RB>([&](int rb) { c[rb][cb] = dpas<SG, HW>(sg, af[rb], bf[cb], c[rb][cb]); });
        });
      }
      uint32_t x0 = 0, x1 = 0;
      each<RB>([&](int rb) {
        each<NCB>([&](int cb) {
          const v8i v = c[rb][cb];
          x0 ^= (uint32_t)(v[0] ^ v[1] ^ v[2] ^ v[3]);
          x1 ^= (uint32_t)(v[4] ^ v[5] ^ v[6] ^ v[7]);
        });
      });
      x0 ^= sycl::permute_group_by_xor(sg, x0, 1u);
      x1 ^= sycl::permute_group_by_xor(sg, x1, 1u);
      if (lane < 8u) part[(sgi * 8u + lane) * 16u + ch] = (lane & 1u) ? x1 : x0;
    }
    sycl::group_barrier(it.get_group());
    if (sgi == 0 && lane < 8u) {
      const uint32_t qr = lane;
      uint32_t words[16];
      for (int j = 0; j < 16; j++) {
        uint32_t x = 0;
        for (int s = 0; s < S; s++) x ^= part[((uint32_t)s * 8u + qr) * 16u + (uint32_t)j];
        words[j] = x;
      }
      const uint32_t local = (2u * wr + (qr & 1u)) + A.rowsValid * (4u * wc + (qr >> 1));
      region_epilogue(words, local, A.test, A.hits, A.allTr);
    }
  }
};

template <int SG, int RB, bool HW>
inline void fold_xmxf_a5(sycl::queue &q, const FoldArgs &a, uint32_t groups) {
  constexpr int S = 4 / RB;
#if defined(PEARL_A5) && PEARL_A5 == 12
  // Two windows a group of 4 sub-groups (the host's row windows must be even).
  constexpr int WG = 2 * S * SG;
  groups /= 2u;
#elif defined(PEARL_A5) && PEARL_A5 == 13
  constexpr int WG = PEARL_A5_WPG * S * SG;
  groups /= PEARL_A5_WPG;
#else
  constexpr int WG = S * SG;
#endif
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<uint32_t, 1> part(S * 8 * PEARL_A5_ALLOC, h);
    h.parallel_for(sycl::nd_range<1>((size_t)groups * WG, WG), FoldXmxFA5<SG, RB, HW>{a, part});
  });
}

#endif  // PEARL_SYCL_A5_HPP
