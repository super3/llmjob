// pipeprobe -- what one warp scheduler's tensor pipe does with the instruction streams we give it.
// Our own code, written for the nv-kernel study. One block an SM, WARPS warps a block (warp w runs on
// scheduler w % 4). A warp either folds (runs of ACC mma.m16n8k32 with other work between them), produces
// (no mma: loads, ALU and stores in a loop, paced by its scheduler's folding warp), or leaves at once.
// Every folding warp times itself with the SM's cycle counter (clock64), so the result is cycles an
// IMMA a scheduler, whatever the clock.
//
//   pipeprobe <name> <foldmask> <prodmask> <periods> <ksteps> <kload> <rd> <burst> <lead8> <pkind> <pn> [secs]
//
//   a folding warp:  lead8 x (8 mma)                        its stagger, run once
//                    periods x { ksteps x { ACC mma [INTER sham ALU after each]; kload ? reload fragments }
//                                rd ? readout (2 XOR3 an mma tile, 3 shared XOR)
//                                burst x (8 sham ALU) }
//   lead8 is multiplied by the warp's row on its scheduler (w / 4), so a scheduler's warps start apart.
//   a producing warp: pn groups of kind pkind for every k-step its scheduler's first folding warp ends.
//   secs: repeat the launch for that long (for clock and watts); the last launch is the one reported.
//   rd: 1 a run of XOR3 and three shared XORs, 2 the same with the thread's own transcript word, 3 two XOR3 woven
//       after each mma of the last k-step (tile i - LAG), 4 one after each mma of the last and first k-steps
//       (tile i - LAGW), 5 three runs of 32 between mma segments (with run-time tests between the segments, which
//       is what makes this path slow: see pipeprobe2.cu for the same stream with none).
//   pkind: for a producing warp 0 ALU, 1 wide shared loads, 2 wide shared stores, 3 a draw-shaped unit, 4 ldmatrix;
//       pn < 0 runs it unpaced. With no producers: 9 makes the burst's steps independent, 10 / 11 / 12 put a
//       draw-shaped run of pn groups in the folding warp (k-step end / after mma 32 / the same, steps grouped).
//   This probe decides roles and counts at run time, and those tests cost pipe time themselves; compare its
//   rows with each other only. Build: -DWARPS=8|12|16 -DACC=48|32|24 [-DRT -DCT -DRCH -DINTER -DLAG -DLAGW].
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#ifndef WARPS
#define WARPS 8
#endif
#ifndef ACC
#define ACC 48          // mma a k-step; a thread holds 4 * ACC accumulators
#endif
#ifndef RT
#define RT 4            // A fragments (row tiles) held
#endif
#ifndef CT
#define CT 8            // B fragments (column tiles) held
#endif
#ifndef INTER
#define INTER 0         // sham ALU instructions after every mma
#endif
#ifndef LAG
#define LAG 2            // woven readout: tile i - LAG is XORed after mma i
#endif
#ifndef RCH
#define RCH 4            // XOR chains the readout keeps
#endif
#ifndef LAGW
#define LAGW 24          // rd 4: the XOR3 after mma i reads tile i - LAGW
#endif
#define THREADS (WARPS * 32)
#define SHBYTES (56 * 1024)

__device__ __forceinline__ void ldsm4(unsigned &r0, unsigned &r1, unsigned &r2, unsigned &r3, const void *p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

extern "C" __global__ __launch_bounds__(THREADS, 1)
void pipeprobe(long long *times, int *sink, unsigned foldmask, unsigned prodmask, int periods, int ksteps,
               int kload, int rd, int burst, int lead8, int pkind, int pn) {
  extern __shared__ __align__(16) unsigned char sh[];
  const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u, sched = warp & 3u;
  unsigned *const shw = reinterpret_cast<unsigned *>(sh);
  volatile unsigned *const tick = reinterpret_cast<volatile unsigned *>(sh) + sched;       // k-steps done
  volatile unsigned *const done = reinterpret_cast<volatile unsigned *>(sh) + 4u + sched;  // folding over
  for (unsigned i = tid; i < SHBYTES / 4u; i += THREADS) shw[i] = (i < 8u) ? 0u : i * 2654435761u;
  // (words 64 .. 64 + 16 * THREADS hold the per-thread transcripts when rd is 2; the fragment and table reads may overlap them)
  __syncthreads();
  const bool folds = (foldmask >> warp) & 1u, prods = (prodmask >> warp) & 1u;
  // The scheduler's first folding warp keeps the tick its producers pace themselves by.
  const bool ticker = folds && (foldmask & ((1u << warp) - 1u) & (0x11111111u << sched)) == 0u;
  unsigned s0 = tid * 0x9E3779B9u + 1u, s1 = tid ^ 0x85EBCA6Bu, s2 = tid + 0xC2B2AE35u, s3 = ~tid;
  asm volatile("" : "+r"(s0), "+r"(s1), "+r"(s2), "+r"(s3));

  if (prods) {
    // Work areas in shared memory: a table to read (8 KB a warp) and a slab to write (2 KB a warp).
    unsigned char *const tab = sh + 1024u + warp * 8192u % (40u * 1024u);
    unsigned char *const slab = sh + 44u * 1024u + (warp & 3u) * 2048u + lane * 64u;
    unsigned target = 0, groups = 0, o = lane * 16u;
    while (!*done) {
      const int pn1 = pn < 0 ? 16 : pn;
      for (int j = 0; j < pn1; j++) {
        if (pkind == 0) {            // 8 ALU
          asm volatile("add.u32 %0, %0, %1;\n xor.b32 %1, %1, %2;\n add.u32 %2, %2, %3;\n xor.b32 %3, %3, %0;\n"
                       "add.u32 %0, %0, %2;\n xor.b32 %1, %1, %3;\n add.u32 %2, %2, %1;\n xor.b32 %3, %3, %2;\n"
                       : "+r"(s0), "+r"(s1), "+r"(s2), "+r"(s3));
        } else if (pkind == 1) {     // 4 wide shared loads, folded into two words
          o = (o + 80u) & 8176u;
          const uint4 x = *reinterpret_cast<const uint4 *>(tab + o);
          const uint4 y = *reinterpret_cast<const uint4 *>(tab + ((o + 2048u) & 8176u));
          const uint4 z = *reinterpret_cast<const uint4 *>(tab + ((o + 4096u) & 8176u));
          const uint4 w = *reinterpret_cast<const uint4 *>(tab + ((o + 6144u) & 8176u));
          s0 ^= x.x ^ y.y; s1 ^= z.z ^ w.w; s2 ^= x.z ^ z.x; s3 ^= y.w ^ w.y;
        } else if (pkind == 2) {     // 4 wide shared stores
          *reinterpret_cast<uint4 *>(slab) = make_uint4(s0, s1, s2, s3);
          *reinterpret_cast<uint4 *>(slab + 16) = make_uint4(s1, s2, s3, s0);
          *reinterpret_cast<uint4 *>(slab + 32) = make_uint4(s2, s3, s0, s1);
          *reinterpret_cast<uint4 *>(slab + 48) = make_uint4(s3, s0, s1, s2);
          s0 += 1u;
        } else if (pkind == 3) {     // a draw-shaped unit: 16 bytes out of 8 looked-up 8-byte reads
          o = (o + 72u) & 8184u;
          const unsigned long long a0 = *reinterpret_cast<const unsigned long long *>(tab + o);
          const unsigned long long a1 = *reinterpret_cast<const unsigned long long *>(tab + ((o + 1032u) & 8184u));
          const unsigned long long a2 = *reinterpret_cast<const unsigned long long *>(tab + ((o + 3080u) & 8184u));
          const unsigned long long a3 = *reinterpret_cast<const unsigned long long *>(tab + ((o + 5128u) & 8184u));
          const unsigned w0 = (unsigned)a0 - (unsigned)(a1 >> 32), w1 = (unsigned)(a0 >> 32) - (unsigned)a1;
          const unsigned w2 = (unsigned)a2 - (unsigned)(a3 >> 32), w3 = (unsigned)(a2 >> 32) - (unsigned)a3;
          *reinterpret_cast<uint4 *>(slab) = make_uint4(__byte_perm(w0, w1, 0x5140u), __byte_perm(w2, w3, 0x5140u),
                                                        __byte_perm(w0, w1, 0x7362u), __byte_perm(w2, w3, 0x7362u));
        } else {                     // 4 ldmatrix
          o = (o + 64u) & 4032u;
          unsigned r0, r1, r2, r3, r4, r5, r6, r7;
          ldsm4(r0, r1, r2, r3, tab + o + (lane & 7u) * 16u);
          ldsm4(r4, r5, r6, r7, tab + ((o + 1024u) & 4032u) + (lane & 7u) * 16u);
          s0 ^= r0 ^ r5; s1 ^= r1 ^ r6; s2 ^= r2 ^ r7; s3 ^= r3 ^ r4;
          ldsm4(r0, r1, r2, r3, tab + ((o + 2048u) & 4032u) + (lane & 7u) * 16u);
          ldsm4(r4, r5, r6, r7, tab + ((o + 3072u) & 4032u) + (lane & 7u) * 16u);
          s0 ^= r0 ^ r5; s1 ^= r1 ^ r6; s2 ^= r2 ^ r7; s3 ^= r3 ^ r4;
        }
      }
      groups += (unsigned)pn1;
      target += 1u;
      while (pn >= 0 && *tick < target && !*done) asm volatile("nanosleep.u32 64;\n");
    }
    if (lane == 0) times[blockIdx.x * (2 * WARPS) + 2 * warp] = -(long long)groups;
    if (lane == 0) times[blockIdx.x * (2 * WARPS) + 2 * warp + 1] = 0;
    sink[blockIdx.x * THREADS + tid] = (int)(s0 ^ s1 ^ s2 ^ s3);
    return;
  }
  if (!folds) {
    if (lane == 0) times[blockIdx.x * (2 * WARPS) + 2 * warp] = 0;
    if (lane == 0) times[blockIdx.x * (2 * WARPS) + 2 * warp + 1] = 0;
    return;
  }

  unsigned a[RT][4], b[CT][2];
  int c[ACC][4];
  const unsigned char *const frag = sh + 1024u + (lane & 7u) * 16u + (lane >> 3) * 128u;
#pragma unroll
  for (int i = 0; i < RT; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) a[i][j] = 0x01010101u + tid + i + j;
#pragma unroll
  for (int i = 0; i < CT; i++)
#pragma unroll
    for (int j = 0; j < 2; j++) b[i][j] = 0x01010101u + tid + 2 * i + j;
#pragma unroll
  for (int i = 0; i < ACC; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) c[i][j] = 0;
  unsigned t[RCH];
#pragma unroll
  for (int i = 0; i < RCH; i++) t[i] = 0;

#define MMA(i)                                                                                              \
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "    \
               "{%0,%1,%2,%3};\n"                                                                           \
               : "+r"(c[i][0]), "+r"(c[i][1]), "+r"(c[i][2]), "+r"(c[i][3])                                 \
               : "r"(a[(i) / CT % RT][0]), "r"(a[(i) / CT % RT][1]), "r"(a[(i) / CT % RT][2]),              \
                 "r"(a[(i) / CT % RT][3]), "r"(b[(i) % CT][0]), "r"(b[(i) % CT][1]))

  const unsigned char *const dtab = sh + 8192u + (warp & 7u) * 4096u + lane * 8u;
  unsigned char *const dslab = sh + 44u * 1024u + warp * 1024u % 8192u + lane * 64u % 960u;
  unsigned dro = lane * 24u;
  const long long t0 = clock64();
  for (int l = 0; l < lead8 * (int)(warp >> 2); l++) {
#pragma unroll
    for (int i = 0; i < 8; i++) MMA(i);
  }
  for (int p = 0; p < periods; p++) {
    for (int s = 0; s < ksteps; s++) {
      if (rd == 5 || pkind >= 10) {
        const bool xl = rd == 5 && s == ksteps - 1, xf = rd == 5 && s == 0;
#pragma unroll
        for (int i = 0; i < 8; i++) MMA(i);
        if (xf) {
#pragma unroll
          for (int i = 32; i < 48; i++) {
            asm volatile("xor.b32 %0, %0, %1;\n xor.b32 %0, %0, %2;\n" : "+r"(t[(2 * i) % RCH])
                         : "r"(c[i][0]), "r"(c[i][1]));
            asm volatile("xor.b32 %0, %0, %1;\n xor.b32 %0, %0, %2;\n" : "+r"(t[(2 * i + 1) % RCH])
                         : "r"(c[i][2]), "r"(c[i][3]));
          }
          unsigned x = 0;
#pragma unroll
          for (int i = 0; i < RCH; i++) x ^= t[i];
          unsigned *const jw = shw + 64u + tid * 16u + ((unsigned)p & 15u);
          const unsigned j0 = *jw;
          *jw = ((j0 << 13) | (j0 >> 19)) ^ x;
        }
#pragma unroll
        for (int i = 8; i < 24; i++) MMA(i);
        if (xl) {
#pragma unroll
          for (int i = 0; i < 16; i++) {
            asm volatile("xor.b32 %0, %0, %1;\n xor.b32 %0, %0, %2;\n" : "+r"(t[(2 * i) % RCH])
                         : "r"(c[i][0]), "r"(c[i][1]));
            asm volatile("xor.b32 %0, %0, %1;\n xor.b32 %0, %0, %2;\n" : "+r"(t[(2 * i + 1) % RCH])
                         : "r"(c[i][2]), "r"(c[i][3]));
          }
        }
#pragma unroll
        for (int i = 24; i < 32; i++) MMA(i);
        if (pkind == 12) {
          for (int j = 0; j < 2 * pn; j++) {
            const unsigned long long *q = reinterpret_cast<const unsigned long long *>(dtab + (dro & 4088u));
            dro += 72u;
            unsigned long long a0 = q[0], a1 = q[5], a2 = q[9], a3 = q[14], a4 = q[17], a5 = q[22], a6 = q[26], a7 = q[31];
            asm volatile("" : "+l"(a0), "+l"(a1), "+l"(a2), "+l"(a3), "+l"(a4), "+l"(a5), "+l"(a6), "+l"(a7));
            unsigned w0 = (unsigned)a0 - (unsigned)(a1 >> 32), w1 = (unsigned)(a0 >> 32) - (unsigned)a1;
            unsigned w2 = (unsigned)a2 - (unsigned)(a3 >> 32), w3 = (unsigned)(a2 >> 32) - (unsigned)a3;
            unsigned w4 = (unsigned)a4 - (unsigned)(a5 >> 32), w5 = (unsigned)(a4 >> 32) - (unsigned)a5;
            unsigned w6 = (unsigned)a6 - (unsigned)(a7 >> 32), w7 = (unsigned)(a6 >> 32) - (unsigned)a7;
            asm volatile("" : "+r"(w0), "+r"(w1), "+r"(w2), "+r"(w3), "+r"(w4), "+r"(w5), "+r"(w6), "+r"(w7));
            *reinterpret_cast<uint4 *>(dslab + (j & 1) * 32) =
                make_uint4(__byte_perm(w0, w1, 0x5140u), __byte_perm(w2, w3, 0x5140u),
                           __byte_perm(w0, w1, 0x7362u), __byte_perm(w2, w3, 0x7362u));
            *reinterpret_cast<uint4 *>(dslab + (j & 1) * 32 + 16) =
                make_uint4(__byte_perm(w4, w5, 0x5140u), __byte_perm(w6, w7, 0x5140u),
                           __byte_perm(w4, w5, 0x7362u), __byte_perm(w6, w7, 0x7362u));
          }
        }
        if (pkind == 11) {
          for (int j = 0; j < pn; j++) {
            const unsigned long long *q = reinterpret_cast<const unsigned long long *>(dtab + (dro & 4088u));
            dro += 136u;
#pragma unroll
            for (int u = 0; u < 4; u++) {
              const unsigned long long a0 = q[u * 17], a1 = q[u * 17 + 5], a2 = q[u * 17 + 9], a3 = q[u * 17 + 14];
              const unsigned w0 = (unsigned)a0 - (unsigned)(a1 >> 32), w1 = (unsigned)(a0 >> 32) - (unsigned)a1;
              const unsigned w2 = (unsigned)a2 - (unsigned)(a3 >> 32), w3 = (unsigned)(a2 >> 32) - (unsigned)a3;
              *reinterpret_cast<uint4 *>(dslab + u * 16) =
                  make_uint4(__byte_perm(w0, w1, 0x5140u), __byte_perm(w2, w3, 0x5140u),
                             __byte_perm(w0, w1, 0x7362u), __byte_perm(w2, w3, 0x7362u));
            }
          }
        }
#pragma unroll
        for (int i = 32; i < 40; i++) MMA(i);
        if (xl) {
#pragma unroll
          for (int i = 16; i < 32; i++) {
            asm volatile("xor.b32 %0, %0, %1;\n xor.b32 %0, %0, %2;\n" : "+r"(t[(2 * i) % RCH])
                         : "r"(c[i][0]), "r"(c[i][1]));
            asm volatile("xor.b32 %0, %0, %1;\n xor.b32 %0, %0, %2;\n" : "+r"(t[(2 * i + 1) % RCH])
                         : "r"(c[i][2]), "r"(c[i][3]));
          }
        }
#pragma unroll
        for (int i = 40; i < 48; i++) MMA(i);
        if (pkind == 10) {
          for (int j = 0; j < pn; j++) {
            const unsigned long long *q = reinterpret_cast<const unsigned long long *>(dtab + (dro & 4088u));
            dro += 136u;
#pragma unroll
            for (int u = 0; u < 4; u++) {
              const unsigned long long a0 = q[u * 17], a1 = q[u * 17 + 5], a2 = q[u * 17 + 9], a3 = q[u * 17 + 14];
              const unsigned w0 = (unsigned)a0 - (unsigned)(a1 >> 32), w1 = (unsigned)(a0 >> 32) - (unsigned)a1;
              const unsigned w2 = (unsigned)a2 - (unsigned)(a3 >> 32), w3 = (unsigned)(a2 >> 32) - (unsigned)a3;
              *reinterpret_cast<uint4 *>(dslab + u * 16) =
                  make_uint4(__byte_perm(w0, w1, 0x5140u), __byte_perm(w2, w3, 0x5140u),
                             __byte_perm(w0, w1, 0x7362u), __byte_perm(w2, w3, 0x7362u));
            }
          }
        }
      } else if (rd == 4 && s == ksteps - 1) {
#pragma unroll
        for (int i = 0; i < ACC; i++) {
          MMA(i);
          asm volatile("xor.b32 %0, %0, %1;\n xor.b32 %0, %0, %2;\n" : "+r"(t[i % RCH])
                       : "r"(c[(i + ACC - LAGW) % ACC][0]), "r"(c[(i + ACC - LAGW) % ACC][1]));
        }
      } else if (rd == 4 && s == 0) {
#pragma unroll
        for (int i = 0; i < ACC; i++) {
          MMA(i);
          asm volatile("xor.b32 %0, %0, %1;\n xor.b32 %0, %0, %2;\n" : "+r"(t[i % RCH])
                       : "r"(c[(i + ACC - LAGW) % ACC][2]), "r"(c[(i + ACC - LAGW) % ACC][3]));
        }
        unsigned x = 0;
#pragma unroll
        for (int i = 0; i < RCH; i++) x ^= t[i];
        unsigned *const jw = shw + 64u + tid * 16u + ((unsigned)p & 15u);
        const unsigned j0 = *jw;
        *jw = ((j0 << 13) | (j0 >> 19)) ^ x;
      } else if (rd == 3 && s == ksteps - 1) {
#pragma unroll
        for (int i = 0; i < ACC; i++) {
          MMA(i);
          if (i >= LAG) {
            asm volatile("xor.b32 %0, %0, %1;\n xor.b32 %0, %0, %2;\n" : "+r"(t[0]) : "r"(c[i - LAG][0]), "r"(c[i - LAG][1]));
            asm volatile("xor.b32 %0, %0, %1;\n xor.b32 %0, %0, %2;\n" : "+r"(t[1]) : "r"(c[i - LAG][2]), "r"(c[i - LAG][3]));
          }
        }
#pragma unroll
        for (int i = ACC - LAG; i < ACC; i++) {
          t[0] ^= (unsigned)c[i][0] ^ (unsigned)c[i][1];
          t[1] ^= (unsigned)c[i][2] ^ (unsigned)c[i][3];
        }
        unsigned *const jw = shw + 64u + tid * 16u + ((unsigned)p & 15u);
        const unsigned j0 = *jw;
        *jw = ((j0 << 13) | (j0 >> 19)) ^ t[0] ^ t[1];
      } else {
#pragma unroll
        for (int i = 0; i < ACC; i++) {
          MMA(i);
  #if INTER >= 1
          asm volatile("add.u32 %0, %0, %1;\n" : "+r"(s0) : "r"(s1));
  #endif
  #if INTER >= 2
          asm volatile("xor.b32 %0, %0, %1;\n" : "+r"(s1) : "r"(s2));
  #endif
  #if INTER >= 3
          asm volatile("add.u32 %0, %0, %1;\n" : "+r"(s2) : "r"(s3));
  #endif
  #if INTER >= 4
          asm volatile("xor.b32 %0, %0, %1;\n" : "+r"(s3) : "r"(s0));
  #endif
        }
      }
      if (kload) {
        const unsigned char *const f = frag + ((unsigned)s & 3u) * 1024u;
#pragma unroll
        for (int i = 0; i < RT; i++) ldsm4(a[i][0], a[i][1], a[i][2], a[i][3], f + i * 1024);
#pragma unroll
        for (int i = 0; i < CT; i += 2) ldsm4(b[i][0], b[i][1], b[i + 1][0], b[i + 1][1], f + 8192 + i * 512);
      }
      if (ticker) *tick = (unsigned)(p * ksteps + s + 1);
    }
    if (rd == 1 || rd == 2) {
#pragma unroll
      for (int i = 0; i < ACC; i++) {
        t[(2 * i) % RCH] ^= (unsigned)c[i][0] ^ (unsigned)c[i][1];
        t[(2 * i + 1) % RCH] ^= (unsigned)c[i][2] ^ (unsigned)c[i][3];
      }
      if (rd == 1) {            // the tile is shared by lanes: three shared XORs
        unsigned x = 0;
#pragma unroll
        for (int i = 3; i < RCH; i++) x ^= t[i];
        atomicXor(shw + 16u + (lane & 7u), t[0]);
        atomicXor(shw + 24u + (lane & 7u), t[1]);
        atomicXor(shw + 32u + (lane & 7u), t[2] ^ x);
      } else {                  // the tile is the thread's own: one transcript word of sixteen, in shared
        unsigned *const jw = shw + 64u + tid * 16u + ((unsigned)p & 15u);
        unsigned x = 0;
#pragma unroll
        for (int i = 0; i < RCH; i++) x ^= t[i];
        const unsigned j0 = *jw;
        *jw = ((j0 << 13) | (j0 >> 19)) ^ x;
      }
    }
    if (pkind == 9) {           // a burst of independent steps (no chain longer than two)
      for (int j = 0; j < burst; j++) {
        asm volatile("add.u32 %0, %0, %4;\n xor.b32 %1, %1, %4;\n add.u32 %2, %2, %4;\n xor.b32 %3, %3, %4;\n"
                     "add.u32 %0, %0, %5;\n xor.b32 %1, %1, %5;\n add.u32 %2, %2, %5;\n xor.b32 %3, %3, %5;\n"
                     : "+r"(s0), "+r"(s1), "+r"(s2), "+r"(s3) : "r"(tid), "r"(lane));
      }
    } else {
      for (int j = 0; j < burst; j++) {
        asm volatile("add.u32 %0, %0, %1;\n xor.b32 %1, %1, %2;\n add.u32 %2, %2, %3;\n xor.b32 %3, %3, %0;\n"
                     "add.u32 %0, %0, %2;\n xor.b32 %1, %1, %3;\n add.u32 %2, %2, %1;\n xor.b32 %3, %3, %2;\n"
                     : "+r"(s0), "+r"(s1), "+r"(s2), "+r"(s3));
      }
    }
  }
  const long long t1 = clock64();
  *done = 1u;
  if (lane == 0) {
    times[blockIdx.x * (2 * WARPS) + 2 * warp] = t0;
    times[blockIdx.x * (2 * WARPS) + 2 * warp + 1] = t1;
  }
  int sum = (int)(s0 ^ s1 ^ s2 ^ s3);
#pragma unroll
  for (int i = 0; i < RCH; i++) sum ^= (int)t[i];
#pragma unroll
  for (int i = 0; i < ACC; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) sum += c[i][j];
  sink[blockIdx.x * THREADS + tid] = sum;
}

int main(int argc, char **argv) {
  if (argc < 12) {
    fprintf(stderr, "pipeprobe name foldmask prodmask periods ksteps kload rd burst lead8 pkind pn [secs]\n");
    return 2;
  }
  const char *name = argv[1];
  const unsigned foldmask = strtoul(argv[2], 0, 0), prodmask = strtoul(argv[3], 0, 0);
  const int periods = atoi(argv[4]), ksteps = atoi(argv[5]), kload = atoi(argv[6]), rd = atoi(argv[7]);
  const int burst = atoi(argv[8]), lead8 = atoi(argv[9]), pkind = atoi(argv[10]), pn = atoi(argv[11]);
  const double secs = argc > 12 ? atof(argv[12]) : 0.0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  const int blocks = prop.multiProcessorCount;
  long long *times;
  int *sink;
  cudaMalloc(&times, sizeof(long long) * blocks * 2 * WARPS);
  cudaMalloc(&sink, sizeof(int) * blocks * THREADS);
  long long *h = (long long *)malloc(sizeof(long long) * blocks * 2 * WARPS);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaFuncSetAttribute(pipeprobe, cudaFuncAttributeMaxDynamicSharedMemorySize, SHBYTES);
  float ms = 0, total = 0;
  int launches = 0;
  do {
    cudaMemset(times, 0, sizeof(long long) * blocks * 2 * WARPS);
    cudaEventRecord(e0);
    pipeprobe<<<blocks, THREADS, SHBYTES>>>(times, sink, foldmask, prodmask, periods, ksteps, kload, rd, burst,
                                            lead8, pkind, pn);
    cudaError_t err = cudaGetLastError();
    cudaEventRecord(e1);
    if (err == cudaSuccess) err = cudaEventSynchronize(e1);
    if (err != cudaSuccess) {
      fprintf(stderr, "%s: %s\n", name, cudaGetErrorString(err));
      return 1;
    }
    cudaEventElapsedTime(&ms, e0, e1);
    total += ms;
    launches++;
  } while (total < secs * 1000.0);
  cudaMemcpy(h, times, sizeof(long long) * blocks * 2 * WARPS, cudaMemcpyDeviceToHost);
  // A scheduler's pipe use: 16 cycles for each IMMA its folding warps issued, over the span they ran.
  double use_sum = 0, use_min = 1e9, use_max = 0, groups = 0;
  long long n_imma_sched = 0;
  int nsched = 0;
  for (int bk = 0; bk < blocks; bk++) {
    for (int s = 0; s < 4; s++) {
      long long lo = 0, hi = 0, imma = 0;
      bool any = false;
      for (int w = s; w < WARPS; w += 4) {
        const long long a0 = h[bk * 2 * WARPS + 2 * w], a1 = h[bk * 2 * WARPS + 2 * w + 1];
        if (a0 < 0) groups += (double)(-a0);
        if (a1 == 0) continue;
        if (!any || a0 < lo) lo = a0;
        if (!any || a1 > hi) hi = a1;
        any = true;
        imma += (long long)periods * ksteps * ACC + (long long)lead8 * (w >> 2) * 8;
      }
      if (!any) continue;
      const double use = 16.0 * (double)imma / (double)(hi - lo);
      use_sum += use;
      if (use < use_min) use_min = use;
      if (use > use_max) use_max = use;
      n_imma_sched = imma;
      nsched++;
    }
  }
  const double use = nsched ? use_sum / nsched : 0;
  const double pg = nsched && n_imma_sched ? groups / nsched / (double)n_imma_sched : 0;
  printf("%-22s pipe %6.2f%% (%.2f-%.2f) %6.3f cyc/IMMA pg %.4f %.0f ms x%d\n", name, use * 100, use_min * 100,
         use_max * 100, use > 0 ? 16.0 / use : 0.0, pg, ms, launches);
  return 0;
}
