// The device code of the NEB (nebhash v2) cores, written once for every vendor.
//
// Nothing here names CUDA or HIP. The file that includes it defines
//   NEB_DEV            the qualifier of a device function
//   NEB_HD             the qualifier of a function the host calls too (the hashes)
//   NEB_ATOMIC_INC(p)  adds 1 to the u32 at p, atomically, and gives the value before
// and wraps each nebk_* body in a kernel of its own (neb_host.inc).
//
// How a card mines NEB. An attempt's gradient is the sum of the gradients of its 16 samples, and a
// sample whose output already equals its target has a zero gradient: a dead row. On a trained model
// most samples are dead, so most of the work is finding the attempts that touch few live rows.
//
//   nebk_screen  one nonce: key = BLAKE3(header), the 16 sample indices
//                from ChaCha8(key), and a look at which are live (nebk_count). An attempt with at
//                most max_live live rows (2, or 1 when the pair table does not fit), or with 3 that
//                are all among the rows the triple table covers, is a candidate: its key and the
//                table entry of its rows (nebk_cand).
//   nebk_finish  one candidate: the digest is read from the model's table,
//                ticket = BLAKE3(digest || key), and a ticket at or under the bound is a hit.
//   nebk_pair    one entry of the pair table: the digest of the clipped sum of two rows.
//   nebk_out1, nebk_g2, nebk_out2, nebk_triple   the triple table, one slab (the entries
//                that share their highest row) at a time: per-output digests of each row alone, of
//                each pair, and of each row with the slab's row, then every entry of the slab from them.
//   nebk_full    one attempt in full, any number of live rows: the 16 rows summed, clipped and hashed.
//                Slow. It is how the core is checked against the chain's vectors.
//
// The attempts the screen passes over are never evaluated. That costs no luck: every nonce is an
// independent draw, and a skipped one is simply not tried. The rate that earns is the candidates
// evaluated a second, and that is what the core reports.
//
// The hashes follow the chain's reference (pow/ref and pow/gpu of github.com/SilasEigen/neb, Apache-2.0;
// see ref/NOTICE): BLAKE3's compression with 7 rounds, ChaCha8 block 0, the digest of
// spec/shares.md section 5.3 (D[f mod 8] += clip(g) XOR (f + 7919 * layer)).
#ifndef NEB_KERNEL_H
#define NEB_KERNEL_H

typedef unsigned int nu32;
typedef unsigned long long nu64;

#define NEBK_NB 16
#define NEBK_CLIP 256
// A candidate's entry with this bit set is in the triple table.
#define NEBK_TRIPLE 0x80000000u

// One scan's constants.
struct NebScan {
  nu32 pre[16];    // BLAKE3's state for the header's last block, after the part of round 1 no nonce touches
  nu32 tail[3];    // header words 16, 17, 18; word 19 is the nonce
  nu32 start;      // the first nonce of thread 0
  nu32 threads;    // thread t tries start + t, start + t + threads, ...
  nu32 iters;      // ... this many nonces
  nu32 n;          // samples in the dataset
  nu32 max_live;   // 2: the pair table; 1: the table of single rows
  nu32 t_rows;     // rows 1 .. t_rows are in the triple table; 0: no triple table
  nu32 one;        // 1: see NEB_ADDB
  nu32 bound[8];   // the target, a 256-bit little-endian integer
  nu32 max_hits;
  nu32 cap;        // nbk_lean: candidates a block's list holds
  nu32 cap2;       // nbk_lean: and its list of attempts with a sparse third row
  nu32 blocks;     // nbk_lean: blocks of the launch that screen
  nu32 bcap;       // records a sparse row's bucket holds (nbk_third)
  nu32 dense;      // rows 1 .. dense are dense; the rows after them are the sparse ones
};

struct NebCand {
  nu32 nonce;
  nu32 entry;      // its table entry
  nu32 key[8];
};

// An attempt with three live rows, the highest of them sparse, waiting in that row's bucket for nbk_third: its
// other two rows, p <= q (p in the low half), and its key.
struct NebRec {
  nu32 nonce;
  nu32 pq;
  nu32 key[8];
};

// A model's rows on the card, and the layout of its digest.
struct NebRows {
  nu32 m1;         // rows, counting row 0, the zero row
  nu32 nc;         // direct entries a row: every layer but the factored one, int16 each
  nu32 cin, cout;  // the factored layer: its gradient is the outer product of cout errors and cin inputs
  nu32 base;       // 7919 * (its layer number)
  nu32 n;          // samples in the dataset
  nu32 nclip;      // the first direct entries (a multiple of 8): the ones three rows' sum can clip at
};

NEB_HD nu32 nebk_rotr(nu32 x, int n) { return (x >> n) | (x << (32 - n)); }
NEB_HD nu32 nebk_rotl(nu32 x, int n) { return (x << n) | (x >> (32 - n)); }
// The adds of the two hashes. `one` is 1, from the scan's constants, where the compiler cannot see its
// value. A vendor file may define these as one * b + a: on NVIDIA cards a multiply-add runs on a pipe of
// its own beside the adder's, which every add, xor and rotate otherwise shares (neb_cuda.cu).
#ifndef NEB_ADDB
#define NEB_ADDB(a, b, one) ((a) + (b))
#endif
#ifndef NEB_ADDC
#define NEB_ADDC(a, b, one) ((a) + (b))
#endif
// The lowest set bit's position, for x != 0, and the count of set bits.
#ifndef NEB_CTZ
#define NEB_CTZ(x) ((nu32)__builtin_ctz(x))
#endif
#ifndef NEB_POPC
#define NEB_POPC(x) ((nu32)__builtin_popcount(x))
#endif

// One step of BLAKE3's round. `one` is in scope wherever it is used: see NEB_ADDB.
#define NEBK_G(a, b, c, d, mx, my)                                                              \
  v[a] = NEB_ADDB(NEB_ADDB(v[a], v[b], one), (mx), one); v[d] = nebk_rotr(v[d] ^ v[a], 16);          \
  v[c] = NEB_ADDB(v[c], v[d], one);                      v[b] = nebk_rotr(v[b] ^ v[c], 12);          \
  v[a] = NEB_ADDB(NEB_ADDB(v[a], v[b], one), (my), one); v[d] = nebk_rotr(v[d] ^ v[a], 8);           \
  v[c] = NEB_ADDB(v[c], v[d], one);                      v[b] = nebk_rotr(v[b] ^ v[c], 7);

// A round's four column steps, its four diagonal steps, and the message permutation between rounds.
#define NEBK_COLUMNS                                                   \
  NEBK_G(0, 4, 8, 12, m[0], m[1])   NEBK_G(1, 5, 9, 13, m[2], m[3])    \
  NEBK_G(2, 6, 10, 14, m[4], m[5])  NEBK_G(3, 7, 11, 15, m[6], m[7])
#define NEBK_DIAGONALS                                                 \
  NEBK_G(0, 5, 10, 15, m[8], m[9])  NEBK_G(1, 6, 11, 12, m[10], m[11]) \
  NEBK_G(2, 7, 8, 13, m[12], m[13]) NEBK_G(3, 4, 9, 14, m[14], m[15])
#define NEBK_PERMUTE                                                                                                 \
  t[0] = m[2]; t[1] = m[6]; t[2] = m[3]; t[3] = m[10]; t[4] = m[7]; t[5] = m[0]; t[6] = m[4]; t[7] = m[13];          \
  t[8] = m[1]; t[9] = m[11]; t[10] = m[12]; t[11] = m[5]; t[12] = m[9]; t[13] = m[14]; t[14] = m[15]; t[15] = m[8]; \
  for (int i_ = 0; i_ < 16; i_++) m[i_] = t[i_];

// BLAKE3's compression function for block counter 0: the first 8 words of the output.
NEB_HD void nebk_b3(const nu32 *cv, const nu32 *blk, nu32 blen, nu32 flags, nu32 *out) {
  const nu32 one = 1u;
  nu32 m[16], v[16], t[16];
  for (int i = 0; i < 16; i++) m[i] = blk[i];
  for (int i = 0; i < 8; i++) v[i] = cv[i];
  v[8] = 0x6A09E667u; v[9] = 0xBB67AE85u; v[10] = 0x3C6EF372u; v[11] = 0xA54FF53Au;
  v[12] = 0; v[13] = 0; v[14] = blen; v[15] = flags;
  for (int r = 0; r < 7; r++) {
    NEBK_COLUMNS
    NEBK_DIAGONALS
    if (r < 6) { NEBK_PERMUTE }
  }
  for (int i = 0; i < 8; i++) out[i] = v[i] ^ v[i + 8];
}

// The header's last block is 16 bytes: words 16 to 18 and the nonce, then zeros. In round 1 the
// columns work on separate words, the nonce is the second message word of column 1 alone, and the
// other three columns take no nonce at all. This does those three and the first half of column 1,
// once a header. cv is the state after the header's first 64 bytes.
NEB_HD void nebk_pre(const nu32 *cv, const nu32 *tail, nu32 *v) {
  const nu32 one = 1u;
  for (int i = 0; i < 8; i++) v[i] = cv[i];
  v[8] = 0x6A09E667u; v[9] = 0xBB67AE85u; v[10] = 0x3C6EF372u; v[11] = 0xA54FF53Au;
  v[12] = 0; v[13] = 0; v[14] = 16u; v[15] = 2u | 8u;  // 16 bytes; CHUNK_END | ROOT
  NEBK_G(0, 4, 8, 12, tail[0], tail[1])
  NEBK_G(2, 6, 10, 14, 0u, 0u)
  NEBK_G(3, 7, 11, 15, 0u, 0u)
  v[1] = v[1] + v[5] + tail[2]; v[13] = nebk_rotr(v[13] ^ v[1], 16);
  v[9] = v[9] + v[13];          v[5] = nebk_rotr(v[5] ^ v[9], 12);
}

// The key of the header with this nonce: the rest of the compression nebk_pre began. Only four of the
// block's 16 message words are not zero (words 16 to 18 and the nonce), so each round's steps are written
// out with the words they take, where BLAKE3's message permutation puts them.
NEB_HD void nebk_key(const nu32 *pre, const nu32 *tail, nu32 nonce, nu32 one, nu32 *key) {
  nu32 v[16];
  for (int i = 0; i < 16; i++) v[i] = pre[i];
  const nu32 m0 = tail[0], m1 = tail[1], m2 = tail[2];
  v[1] = NEB_ADDB(NEB_ADDB(v[1], v[5], one), nonce, one); v[13] = nebk_rotr(v[13] ^ v[1], 8);
  v[9] = NEB_ADDB(v[9], v[13], one);                      v[5] = nebk_rotr(v[5] ^ v[9], 7);
  // round 1
  NEBK_G(0, 5, 10, 15, 0u, 0u)
  NEBK_G(1, 6, 11, 12, 0u, 0u)
  NEBK_G(2, 7, 8, 13, 0u, 0u)
  NEBK_G(3, 4, 9, 14, 0u, 0u)
  // round 2
  NEBK_G(0, 4, 8, 12, m2, 0u)
  NEBK_G(1, 5, 9, 13, nonce, 0u)
  NEBK_G(2, 6, 10, 14, 0u, m0)
  NEBK_G(3, 7, 11, 15, 0u, 0u)
  NEBK_G(0, 5, 10, 15, m1, 0u)
  NEBK_G(1, 6, 11, 12, 0u, 0u)
  NEBK_G(2, 7, 8, 13, 0u, 0u)
  NEBK_G(3, 4, 9, 14, 0u, 0u)
  // round 3
  NEBK_G(0, 4, 8, 12, nonce, 0u)
  NEBK_G(1, 5, 9, 13, 0u, 0u)
  NEBK_G(2, 6, 10, 14, 0u, m2)
  NEBK_G(3, 7, 11, 15, 0u, 0u)
  NEBK_G(0, 5, 10, 15, 0u, 0u)
  NEBK_G(1, 6, 11, 12, 0u, m0)
  NEBK_G(2, 7, 8, 13, 0u, 0u)
  NEBK_G(3, 4, 9, 14, 0u, m1)
  // round 4
  NEBK_G(0, 4, 8, 12, 0u, 0u)
  NEBK_G(1, 5, 9, 13, 0u, 0u)
  NEBK_G(2, 6, 10, 14, 0u, nonce)
  NEBK_G(3, 7, 11, 15, 0u, 0u)
  NEBK_G(0, 5, 10, 15, 0u, m0)
  NEBK_G(1, 6, 11, 12, 0u, m2)
  NEBK_G(2, 7, 8, 13, 0u, 0u)
  NEBK_G(3, 4, 9, 14, m1, 0u)
  // round 5
  NEBK_G(0, 4, 8, 12, 0u, 0u)
  NEBK_G(1, 5, 9, 13, 0u, 0u)
  NEBK_G(2, 6, 10, 14, 0u, 0u)
  NEBK_G(3, 7, 11, 15, 0u, 0u)
  NEBK_G(0, 5, 10, 15, 0u, m2)
  NEBK_G(1, 6, 11, 12, 0u, nonce)
  NEBK_G(2, 7, 8, 13, m0, m1)
  NEBK_G(3, 4, 9, 14, 0u, 0u)
  // round 6
  NEBK_G(0, 4, 8, 12, 0u, 0u)
  NEBK_G(1, 5, 9, 13, 0u, 0u)
  NEBK_G(2, 6, 10, 14, 0u, 0u)
  NEBK_G(3, 7, 11, 15, 0u, m1)
  NEBK_G(0, 5, 10, 15, 0u, nonce)
  NEBK_G(1, 6, 11, 12, m0, 0u)
  NEBK_G(2, 7, 8, 13, m2, 0u)
  NEBK_G(3, 4, 9, 14, 0u, 0u)
  // round 7
  NEBK_G(0, 4, 8, 12, 0u, 0u)
  NEBK_G(1, 5, 9, 13, 0u, m0)
  NEBK_G(2, 6, 10, 14, m1, 0u)
  NEBK_G(3, 7, 11, 15, 0u, 0u)
  NEBK_G(0, 5, 10, 15, 0u, 0u)
  NEBK_G(1, 6, 11, 12, m2, 0u)
  NEBK_G(2, 7, 8, 13, nonce, 0u)
  NEBK_G(3, 4, 9, 14, 0u, 0u)
  for (int i = 0; i < 8; i++) key[i] = v[i] ^ v[i + 8];
  (void)m0; (void)m1; (void)m2;
}

#define NEBK_QR(a, b, c, d)                                                                        \
  x[a] = NEB_ADDC(x[a], x[b], one); x[d] = nebk_rotl(x[d] ^ x[a], 16);                                   \
  x[c] = NEB_ADDC(x[c], x[d], one); x[b] = nebk_rotl(x[b] ^ x[c], 12);                                 \
  x[a] = NEB_ADDC(x[a], x[b], one); x[d] = nebk_rotl(x[d] ^ x[a], 8);                                    \
  x[c] = NEB_ADDC(x[c], x[d], one); x[b] = nebk_rotl(x[b] ^ x[c], 7);

// Block 0 of ChaCha8(key): 16 words.
NEB_HD void nebk_chacha8(const nu32 *key, nu32 one, nu32 *out) {
  nu32 s[16], x[16];
  s[0] = 0x61707865u; s[1] = 0x3320646eu; s[2] = 0x79622d32u; s[3] = 0x6b206574u;
  for (int i = 0; i < 8; i++) s[4 + i] = key[i];
  s[12] = 0; s[13] = 0; s[14] = 0; s[15] = 0;
  for (int i = 0; i < 16; i++) x[i] = s[i];
  // Unrolled: left a loop, each trip cost a count, a compare and a branch on the adder.
#pragma unroll
  for (int r = 0; r < 4; r++) {
    NEBK_QR(0, 4, 8, 12) NEBK_QR(1, 5, 9, 13) NEBK_QR(2, 6, 10, 14) NEBK_QR(3, 7, 11, 15)
    NEBK_QR(0, 5, 10, 15) NEBK_QR(1, 6, 11, 12) NEBK_QR(2, 7, 8, 13) NEBK_QR(3, 4, 9, 14)
  }
  for (int i = 0; i < 16; i++) out[i] = NEB_ADDC(x[i], s[i], one);
}

// The pair-table entry of rows p <= q (0 is the zero row): q (q + 1) / 2 + p.
NEB_HD nu32 nebk_entry(nu32 p, nu32 q) { return q * (q + 1u) / 2u + p; }

// The triple-table entry of a <= b <= c, each a row less 1: c (c + 1) (c + 2) / 6 + b (b + 1) / 2 + a.
NEB_HD nu32 nebk_entry3(nu32 a, nu32 b, nu32 c) {
  return (nu32)((nu64)c * (c + 1u) * (c + 2u) / 6u) + b * (b + 1u) / 2u + a;
}

// The pair (p, q), p <= q, that nebk_entry numbers `entry`.
NEB_DEV void nebk_unpair(nu32 entry, nu32 *p, nu32 *q) {
  nu32 b = (nu32)((sqrtf(8.0f * (float)entry + 1.0f) - 1.0f) * 0.5f);
  while ((nu64)b * (b + 1u) / 2u > entry) b--;
  while ((nu64)(b + 1u) * (b + 2u) / 2u <= entry) b++;
  *q = b;
  *p = entry - b * (b + 1u) / 2u;
}

// 1: the triple build keeps each entry's errors on the thread's stack (nebk_triple). A vendor file sets it where
// a stack is cheap: on NVIDIA it saved a third of the loop's adder instructions; on AMD a stack is scratch
// memory, and the build of a TESS table took 15 s with it against 5.4 s without.
#ifndef NEBK_ERRS_ON_STACK
#define NEBK_ERRS_ON_STACK 0
#endif

// floor(u * n / 2^32). A vendor file may define NEB_MULHI as its own instruction for the high word.
#ifndef NEB_MULHI
#define NEB_MULHI(u, n) ((nu32)(((nu64)(u) * (n)) >> 32))
#endif

// The sum of the bytes of an attempt's 16 samples, from its ChaCha8 words. alive is one byte a sample: 0 for a
// dead one, 1 for a live one, NEBK_OUT for a live one whose row a triple table that is up does not cover, or
// NEBK_SPARSE for one whose row is sparse.
#define NEBK_OUT 17u
#define NEBK_SPARSE 241u
NEB_DEV nu32 nebk_count(const nu32 *u, const NebScan *S, const unsigned char *alive) {
  nu32 cnt = 0;
  for (int j = 0; j < NEBK_NB; j++) cnt += alive[NEB_MULHI(u[j], S->n)];
  return cnt;
}

// What an attempt with that sum and a sparse row is: 0 none to evaluate, 1 one the pair table answers (a sparse row
// alone, beside one dense row, or two sparse rows), 2 three live rows with a sparse one among them (nbk_third). A
// sparse row's byte is 241: no mix of 1s and 17s over 16 samples reaches 241 to 243, 258, 259, 275, 482, 483, 499
// or 723, so each of these sums has one make-up. Sums under 241 have no sparse row.
NEB_DEV nu32 nebk_sparse_kind(nu32 sum) {
  const nu32 r = sum - NEBK_SPARSE;
  if (r < 35u) return (r <= 1u || r == 17u) ? 1u : ((r == 2u || r == 18u || r == 34u) ? 2u : 0u);
  if (sum == 2u * NEBK_SPARSE) return 1u;
  return (sum == 2u * NEBK_SPARSE + 1u || sum == 2u * NEBK_SPARSE + NEBK_OUT || sum == 3u * NEBK_SPARSE) ? 2u : 0u;
}

// Whether an attempt with that sum is one to evaluate: at most max_live live samples, exactly 3 that the triple
// table covers, or exactly 3 with a sparse one among them. With no table and no sparse rows the sum is the count
// of live samples. With a table, a sum of 3 or less means that many live samples, all covered; 17 and 18 are one
// uncovered sample alone or with a covered one, and 34 two uncovered ones: the pair table has those. No other mix
// of 16 samples gives these sums. Without the bytes' third value a card looked up the rows of every attempt with
// 3 live samples to find the few its table covers: on a Kepler model, where a table covers 3% of them, that cost
// more than the table gave.
NEB_DEV bool nebk_pass(nu32 sum, const NebScan *S) {
  if (sum >= NEBK_SPARSE) return nebk_sparse_kind(sum) != 0u;
  if (sum <= (S->t_rows ? 3u : S->max_live)) return true;
  return S->t_rows != 0u && (sum - NEBK_OUT <= 1u || sum == 2u * NEBK_OUT);
}

// An attempt the count let through that a table answers. live[s] is 0 for a dead sample and the sample's row (1
// and up) otherwise. Returns 1 and fills *out when the attempt is one to evaluate (its rows are in a table), else 0.
NEB_DEV nu32 nebk_cand(nu32 nonce, const nu32 *key, const nu32 *u, const NebScan *S, const unsigned short *live,
                       NebCand *out) {
  nu32 a = 0, b = 0, c = 0;
  nu32 cnt = 0;
  for (int j = 0; j < NEBK_NB; j++) {
    const nu32 r = live[NEB_MULHI(u[j], S->n)];
    if (r) {
      if (cnt == 0) a = r;
      else if (cnt == 1) b = r;
      else c = r;
      cnt++;
    }
  }
  nu32 entry;
  if (cnt <= S->max_live) {
    // a and b are the live rows, or 0
    const nu32 p = a < b ? a : b, q = a < b ? b : a;
    entry = S->max_live >= 2u ? nebk_entry(p, q) : q;
  } else if (cnt == 3u && a <= S->t_rows && b <= S->t_rows && c <= S->t_rows) {
    nu32 t;
    if (a > b) { t = a; a = b; b = t; }
    if (b > c) { t = b; b = c; c = t; }
    if (a > b) { t = a; a = b; b = t; }
    entry = NEBK_TRIPLE | nebk_entry3(a - 1u, b - 1u, c - 1u);
  } else {
    return 0;
  }
  out->nonce = nonce;
  out->entry = entry;
  for (int i = 0; i < 8; i++) out->key[i] = key[i];
  return 1;
}

// An attempt with three live rows, the highest sparse: its record for that row's bucket. Returns the bucket (the
// sparse row less dense + 1) and fills *out, or returns ~0u when the attempt is not one.
NEB_DEV nu32 nebk_third_rec(nu32 nonce, const nu32 *key, const nu32 *u, const NebScan *S, const unsigned short *live,
                            NebRec *out) {
  nu32 r3[3] = {0, 0, 0}, cnt = 0;
  for (int j = 0; j < NEBK_NB; j++) {
    const nu32 r = live[NEB_MULHI(u[j], S->n)];
    if (r) {
      if (cnt < 3u) r3[cnt] = r;
      cnt++;
    }
  }
  nu32 a = r3[0], b = r3[1], c = r3[2], t;
  if (a > b) { t = a; a = b; b = t; }
  if (b > c) { t = b; b = c; c = t; }
  if (a > b) { t = a; a = b; b = t; }
  if (cnt != 3u || c <= S->dense) return ~0u;
  out->nonce = nonce;
  out->pq = a | (b << 16);
  for (int i = 0; i < 8; i++) out->key[i] = key[i];
  return c - S->dense - 1u;
}

// One nonce. Nearly every nonce is passed over, so the count of live samples comes from the byte table alone, and
// the rows are looked up only for the few that stay. Returns 1 and fills *out when the attempt is one to
// evaluate, else 0.
NEB_DEV nu32 nebk_screen(nu32 nonce, const NebScan *S, const unsigned short *live, const unsigned char *alive,
                         NebCand *out) {
  nu32 key[8], u[16];
  nebk_key(S->pre, S->tail, nonce, S->one, key);
  nebk_chacha8(key, S->one, u);
  if (!nebk_pass(nebk_count(u, S, alive), S)) return 0;
  return nebk_cand(nonce, key, u, S, live, out);
}

// One candidate: the digest is read from the model's table, and a ticket at or under the bound is a hit.
// hits[0] counts the hits; the nonces follow.
NEB_DEV void nebk_ticket(const NebScan *S, nu32 nonce, const nu32 *key, const nu32 *dg, nu32 *hits) {
  nu32 blk[16], iv[8], t[8];
  for (int i = 0; i < 8; i++) { blk[i] = dg[i]; blk[8 + i] = key[i]; }
  iv[0] = 0x6A09E667u; iv[1] = 0xBB67AE85u; iv[2] = 0x3C6EF372u; iv[3] = 0xA54FF53Au;
  iv[4] = 0x510E527Fu; iv[5] = 0x9B05688Cu; iv[6] = 0x1F83D9ABu; iv[7] = 0x5BE0CD19u;
  nebk_b3(iv, blk, 64u, 1u | 2u | 8u, t);  // CHUNK_START | CHUNK_END | ROOT
  bool ok = true;
  for (int k = 7; k >= 0; k--) {
    if (t[k] != S->bound[k]) { ok = t[k] < S->bound[k]; break; }
  }
  if (!ok) return;
  const nu32 h = NEB_ATOMIC_INC(hits);
  if (h < S->max_hits) hits[1u + h] = nonce;
}

NEB_DEV void nebk_finish(const NebScan *S, const NebCand *at, const nu32 *pairs, const nu32 *triples, nu32 *hits) {
  const nu32 e = at->entry;
  nebk_ticket(S, at->nonce, at->key, (e & NEBK_TRIPLE) ? triples + (nu64)(e & ~NEBK_TRIPLE) * 8u : pairs + (nu64)e * 8u, hits);
}

NEB_HD int nebk_clip(int g) { return g < -NEBK_CLIP ? -NEBK_CLIP : (g > NEBK_CLIP ? NEBK_CLIP : g); }

// The digest's 8 lanes, kept in 8 variables so they stay in registers: entry f goes to lane f mod 8, and every
// layer's size is a multiple of 8, so 8 entries in a row are lanes 0 to 7 in order.
//   NEBK_ACC8(G, T)   adds clip(G(k)) XOR T(k) to lane k, for k = 0 .. 7
//   NEBK_ACC8N(G, T)  the same without the clip, where the sum cannot reach it: the factored layer's entries
//                     are at most 64 a row, so 4 rows or fewer stay within 256
#define NEBK_LANES nu32 D0 = 0, D1 = 0, D2 = 0, D3 = 0, D4 = 0, D5 = 0, D6 = 0, D7 = 0;
#define NEBK_ACC8(G, T)                                                                        \
  D0 += (nu32)nebk_clip(G(0)) ^ (T(0)); D1 += (nu32)nebk_clip(G(1)) ^ (T(1));                  \
  D2 += (nu32)nebk_clip(G(2)) ^ (T(2)); D3 += (nu32)nebk_clip(G(3)) ^ (T(3));                  \
  D4 += (nu32)nebk_clip(G(4)) ^ (T(4)); D5 += (nu32)nebk_clip(G(5)) ^ (T(5));                  \
  D6 += (nu32)nebk_clip(G(6)) ^ (T(6)); D7 += (nu32)nebk_clip(G(7)) ^ (T(7));
#define NEBK_ACC8N(G, T)                                                                       \
  D0 += (nu32)(G(0)) ^ (T(0)); D1 += (nu32)(G(1)) ^ (T(1)); D2 += (nu32)(G(2)) ^ (T(2));       \
  D3 += (nu32)(G(3)) ^ (T(3)); D4 += (nu32)(G(4)) ^ (T(4)); D5 += (nu32)(G(5)) ^ (T(5));       \
  D6 += (nu32)(G(6)) ^ (T(6)); D7 += (nu32)(G(7)) ^ (T(7));
#define NEBK_ADD8(P)                                                                           \
  D0 += (P)[0]; D1 += (P)[1]; D2 += (P)[2]; D3 += (P)[3]; D4 += (P)[4]; D5 += (P)[5]; D6 += (P)[6]; D7 += (P)[7];
#define NEBK_PUT8(P)                                                                           \
  (P)[0] = D0; (P)[1] = D1; (P)[2] = D2; (P)[3] = D3; (P)[4] = D4; (P)[5] = D5; (P)[6] = D6; (P)[7] = D7;
// Input values are read 4 at a time, as one word: NEBK_X8(lo, hi, k) is value k of 8, signed.
#define NEBK_X(w, k) ((int)(signed char)((w) >> (8 * (k))))
#define NEBK_X8(lo, hi, k) ((k) < 4 ? NEBK_X(lo, (k) & 3) : NEBK_X(hi, (k) & 3))
#define NEBK_LOAD4(p) (*(const nu32 *)(p))

// One pair-table entry: the digest of rows p and q summed. With pairs == 0 the table holds single rows
// and entry e is row e alone.
//   conv   m1 x nc int16: the direct entries of each row
//   tag    nc words: f + 7919 * layer of each direct entry
//   d5     m1 x cout int8, x5 m1 x cin int8: the factored layer
//   zero5  cout x 8 words: what a row of that layer adds to each lane when its gradient is all zero
NEB_DEV void nebk_pair(nu32 entry, nu32 pairs, const NebRows *R, const short *conv, const nu32 *tag,
                       const signed char *d5, const signed char *x5, const nu32 *zero5, nu32 *table) {
  nu32 p = 0, q = entry;
  if (pairs) nebk_unpair(entry, &p, &q);
  NEBK_LANES
  const short *cp = conv + (nu64)p * R->nc, *cq = conv + (nu64)q * R->nc;
  for (nu32 e = 0; e < R->nc; e += 8u) {
#define NEBK_G_(k) ((int)cp[e + k] + (int)cq[e + k])
#define NEBK_T_(k) tag[e + k]
    NEBK_ACC8(NEBK_G_, NEBK_T_)
#undef NEBK_G_
#undef NEBK_T_
  }
  const signed char *xp = x5 + (nu64)p * R->cin, *xq = x5 + (nu64)q * R->cin;
  nu32 live = 0;
  for (nu32 o = 0; o < R->cout; o++) {
    const int dp = d5[(nu64)p * R->cout + o], dq = d5[(nu64)q * R->cout + o];
    if ((dp | dq) == 0) {
      NEBK_ADD8(zero5 + o * 8u)
      continue;
    }
    live |= 1u << o;
  }
  // The outputs with an error, summed entry by entry. The inputs are walked once, 8 at a time, and each 8 go to
  // every such output before the next 8 are read: walked once an output, the rows were read from the card's
  // memory again each time, and a build was as fast as the memory, not the card.
  for (nu32 j = 0; live && j < R->cin; j += 8u) {
    const nu32 p0 = NEBK_LOAD4(xp + j), p1 = NEBK_LOAD4(xp + j + 4u), q0 = NEBK_LOAD4(xq + j), q1 = NEBK_LOAD4(xq + j + 4u);
    for (nu32 m = live; m; m &= m - 1u) {
      const nu32 o = NEB_CTZ(m);
      const int dp = d5[(nu64)p * R->cout + o], dq = d5[(nu64)q * R->cout + o];
      const nu32 f0 = o * R->cin + R->base;
#define NEBK_G_(k) (dp * NEBK_X8(p0, p1, k) + dq * NEBK_X8(q0, q1, k))
#define NEBK_T_(k) (f0 + j + k)
      NEBK_ACC8N(NEBK_G_, NEBK_T_)
#undef NEBK_G_
#undef NEBK_T_
    }
  }
  nu32 *out = table + (nu64)entry * 8u;
  NEBK_PUT8(out)
}

// The triple table is built a slab at a time: slab c holds the entries (a, b, c) with a <= b <= c.
// Most of an entry's work is the factored layer, and most of that can be shared. For one output o of
// that layer, which of the rows have an error there decides the work (mask[r] has a bit an output where
// row r has one):
//   none               the constant zero5[o]
//   one row            that row alone: out1, built once a model
//   a and c, or c      a with c: out2, built once a slab
//   a and b            the pair's own sum: g2, built once a model for every pair of rows the table covers
//   b and c            summed entry by entry, whatever a's error there
// So an entry sums entry by entry only where b and c both have an error, and its neighbours in the slab
// (the same b and c, the next a) do so at the same outputs: a warp's threads stay in step. Summing where
// a and b both have one, as this did before, left each thread with its own count of outputs, and a warp
// took as long as its slowest: a table of 22 GB took 5 s on an RTX 4090.
//
// Tried and lost: the slab's sums db x_b + dc x_c kept in a table a slab, so that an entry adds only its
// own row's part. One multiply an entry less, one 16-bit load more, from 18 KB a row b that no SM keeps
// near: 8.4 s for the table that took 5.4 on an RTX 4090. The card's memory is slower than its adders.

// out1[row][o]: the 8 lanes output o adds for `row` alone. `index` is row * cout + o.
NEB_DEV void nebk_out1(nu32 index, const NebRows *R, const signed char *d5, const signed char *x5, nu32 *out1) {
  const nu32 row = index / R->cout, o = index % R->cout;
  const int d = d5[index];
  NEBK_LANES
  if (d != 0) {
    const signed char *x = x5 + (nu64)row * R->cin;
    const nu32 f0 = o * R->cin + R->base;
    for (nu32 j = 0; j < R->cin; j += 8u) {
      const nu32 w0 = NEBK_LOAD4(x + j), w1 = NEBK_LOAD4(x + j + 4u);
#define NEBK_G_(k) (d * NEBK_X8(w0, w1, k))
#define NEBK_T_(k) (f0 + j + k)
      NEBK_ACC8N(NEBK_G_, NEBK_T_)
#undef NEBK_G_
#undef NEBK_T_
    }
  }
  nu32 *out = out1 + (nu64)index * 8u;
  NEBK_PUT8(out)
}

// out2[row][o]: the same for `row` summed with the slab's row c (c is a row number, 1 and up).
NEB_DEV void nebk_out2(nu32 index, nu32 c, const NebRows *R, const signed char *d5, const signed char *x5, nu32 *out2) {
  const nu32 row = index / R->cout, o = index % R->cout;
  const int d = d5[index], dc = d5[(nu64)c * R->cout + o];
  NEBK_LANES
  if ((d | dc) != 0) {
    const signed char *x = x5 + (nu64)row * R->cin, *xc = x5 + (nu64)c * R->cin;
    const nu32 f0 = o * R->cin + R->base;
    for (nu32 j = 0; j < R->cin; j += 8u) {
      const nu32 w0 = NEBK_LOAD4(x + j), w1 = NEBK_LOAD4(x + j + 4u), c0 = NEBK_LOAD4(xc + j), c1 = NEBK_LOAD4(xc + j + 4u);
#define NEBK_G_(k) (d * NEBK_X8(w0, w1, k) + dc * NEBK_X8(c0, c1, k))
#define NEBK_T_(k) (f0 + j + k)
      NEBK_ACC8N(NEBK_G_, NEBK_T_)
#undef NEBK_G_
#undef NEBK_T_
    }
  }
  nu32 *out = out2 + (nu64)index * 8u;
  NEBK_PUT8(out)
}

// g2: pair e's outputs where both rows have an error, in order of output, 8 lanes each. The pair is rows
// a <= b from nebk_unpair(e), each plus 1; off2[e] is where its outputs start, counted in outputs.
NEB_DEV void nebk_g2(nu32 e, const NebRows *R, const nu32 *mask, const nu32 *off2, const signed char *d5,
                     const signed char *x5, nu32 *g2) {
  nu32 a, b;
  nebk_unpair(e, &a, &b);
  a++;
  b++;
  const signed char *xa = x5 + (nu64)a * R->cin, *xb = x5 + (nu64)b * R->cin;
  nu32 *out = g2 + (nu64)off2[e] * 8u;
  for (nu32 m = mask[a] & mask[b]; m; m &= m - 1u, out += 8) {
    const nu32 o = NEB_CTZ(m);
    const int da = d5[(nu64)a * R->cout + o], db = d5[(nu64)b * R->cout + o];
    const nu32 f0 = o * R->cin + R->base;
    NEBK_LANES
    for (nu32 j = 0; j < R->cin; j += 8u) {
      const nu32 a0 = NEBK_LOAD4(xa + j), a1 = NEBK_LOAD4(xa + j + 4u), b0 = NEBK_LOAD4(xb + j), b1 = NEBK_LOAD4(xb + j + 4u);
#define NEBK_G_(k) (da * NEBK_X8(a0, a1, k) + db * NEBK_X8(b0, b1, k))
#define NEBK_T_(k) (f0 + j + k)
      NEBK_ACC8N(NEBK_G_, NEBK_T_)
#undef NEBK_G_
#undef NEBK_T_
    }
    NEBK_PUT8(out)
  }
}

// `count` direct entries (a multiple of 8) of three rows summed, added to the 8 lanes D: ca, cb and cc point at
// the rows' entries, tag at their tags. The first `clip` of them are clipped. The host puts the entries where
// three rows can pass 256 first (NebRows.nclip): on the pool's models that is a quarter to a half of them,
// and the rest need no clip.
NEB_DEV void nebk_triple_conv(const short *ca, const short *cb, const short *cc, const nu32 *tag, nu32 count, nu32 clip,
                              nu32 *D) {
  nu32 D0 = D[0], D1 = D[1], D2 = D[2], D3 = D[3], D4 = D[4], D5 = D[5], D6 = D[6], D7 = D[7];
#define NEBK_G_(k) ((int)ca[i + k] + (int)cb[i + k] + (int)cc[i + k])
#define NEBK_T_(k) tag[i + k]
  nu32 i = 0;
  for (; i < count && i < clip; i += 8u) {
    NEBK_ACC8(NEBK_G_, NEBK_T_)
  }
  for (; i < count; i += 8u) {
    NEBK_ACC8N(NEBK_G_, NEBK_T_)
  }
#undef NEBK_G_
#undef NEBK_T_
  NEBK_PUT8(D)
}

// The same for a row and a sum of two rows the caller made (bc[i] = b's entry plus c's): the triple build keeps
// both in shared memory (NB_TRIPLE_NEAR).
NEB_DEV void nebk_triple_conv2(const short *ca, const short *bc, const nu32 *tag, nu32 count, nu32 clip, nu32 *D) {
  nu32 D0 = D[0], D1 = D[1], D2 = D[2], D3 = D[3], D4 = D[4], D5 = D[5], D6 = D[6], D7 = D[7];
#define NEBK_G_(k) ((int)ca[i + k] + (int)bc[i + k])
#define NEBK_T_(k) tag[i + k]
  nu32 i = 0;
  for (; i < count && i < clip; i += 8u) {
    NEBK_ACC8(NEBK_G_, NEBK_T_)
  }
  for (; i < count; i += 8u) {
    NEBK_ACC8N(NEBK_G_, NEBK_T_)
  }
#undef NEBK_G_
#undef NEBK_T_
  NEBK_PUT8(D)
}

// The entry of rows a <= b <= c in slab c (its number there is that of the pair a - 1, b - 1), from the lanes D
// that nebk_triple_conv made of its direct entries: the factored layer, then the entry is written. `slab` is
// where the slab starts in the table.
NEB_DEV void nebk_triple(nu32 a, nu32 b, nu32 c, const NebRows *R, const nu32 *mask, const signed char *d5,
                         const signed char *x5, const nu32 *zero5, const nu32 *out1, const nu32 *out2, const nu32 *off2,
                         const nu32 *g2, const nu32 *D, nu32 *slab) {
  const nu32 e = nebk_entry(a - 1u, b - 1u);
  nu32 D0 = D[0], D1 = D[1], D2 = D[2], D3 = D[3], D4 = D[4], D5 = D[5], D6 = D[6], D7 = D[7];
  // The outputs a shared digest answers.
  const nu32 ma = mask[a], mb = mask[b], mc = mask[c], direct = mb & mc;
  const nu32 *pair = g2 + (nu64)off2[e] * 8u;
  for (nu32 o = 0; o < R->cout; o++) {
    const nu32 bit = 1u << o;
    const nu32 *part;
    if (direct & bit) continue;
    if (mc & bit) part = out2 + ((nu64)a * R->cout + o) * 8u;
    else if (mb & bit) part = (ma & bit) ? pair + (nu64)NEB_POPC(ma & mb & (bit - 1u)) * 8u : out1 + ((nu64)b * R->cout + o) * 8u;
    else part = (ma & bit) ? out1 + ((nu64)a * R->cout + o) * 8u : zero5 + o * 8u;
    NEBK_ADD8(part)
  }
  // The outputs where b and c both have an error, summed entry by entry. The inputs are walked once, 8 at a
  // time, and each 8 go to every such output before the next 8 are read: walked once an output, the rows were
  // read from the card's memory again each time, and a build was as fast as the memory, not the card.
  const signed char *xa = x5 + (nu64)a * R->cin, *xb = x5 + (nu64)b * R->cin, *xc = x5 + (nu64)c * R->cin;
#if NEBK_ERRS_ON_STACK
  // Each such output's three errors and first tag, read once: in the loop below they are a look-up on the
  // thread's stack, where reading them from the rows each time was three loads and three 64-bit addresses.
  signed char ea[32], eb[32], ec[32];
  nu32 tag0[32], n = 0;
  for (nu32 m = direct; m; m &= m - 1u, n++) {
    const nu32 o = NEB_CTZ(m);
    ea[n] = d5[(nu64)a * R->cout + o];
    eb[n] = d5[(nu64)b * R->cout + o];
    ec[n] = d5[(nu64)c * R->cout + o];
    tag0[n] = o * R->cin + R->base;
  }
#define NEBK_EACH_ for (nu32 i = 0; i < n; i++)
#define NEBK_ERRS_ const int da = ea[i], db = eb[i], dc = ec[i]; const nu32 t0 = tag0[i];
#else
#define NEBK_EACH_ for (nu32 m = direct; m; m &= m - 1u)
#define NEBK_ERRS_                                                                                                const nu32 o = NEB_CTZ(m);                                                                                      const int da = d5[(nu64)a * R->cout + o], db = d5[(nu64)b * R->cout + o], dc = d5[(nu64)c * R->cout + o];       const nu32 t0 = o * R->cin + R->base;
#endif
#define NEBK_G_(k) (da * NEBK_X8(a0, a1, k) + db * NEBK_X8(b0, b1, k) + dc * NEBK_X8(c0, c1, k))
  if ((R->base & 7u) == 3u) {
    // The tags of 8 entries in a row are t, t + 1, ... t + 7 with t = 3 mod 8 (KEP1's factored layer: its base
    // is 7919 * 5 and its rows are 768 long). So with h = t - 3 they are h ^ 3 .. h ^ 7, then h + 8, (h + 8) ^ 1,
    // (h + 8) ^ 2: an xor with a constant in place of an add an entry.
    for (nu32 j = 0; direct && j < R->cin; j += 8u) {
      const nu32 a0 = NEBK_LOAD4(xa + j), a1 = NEBK_LOAD4(xa + j + 4u), b0 = NEBK_LOAD4(xb + j), b1 = NEBK_LOAD4(xb + j + 4u);
      const nu32 c0 = NEBK_LOAD4(xc + j), c1 = NEBK_LOAD4(xc + j + 4u);
      NEBK_EACH_ {
        NEBK_ERRS_
        const nu32 h = t0 + j - 3u, h8 = h + 8u;
        D0 += (nu32)NEBK_G_(0) ^ h ^ 3u; D1 += (nu32)NEBK_G_(1) ^ h ^ 4u; D2 += (nu32)NEBK_G_(2) ^ h ^ 5u;
        D3 += (nu32)NEBK_G_(3) ^ h ^ 6u; D4 += (nu32)NEBK_G_(4) ^ h ^ 7u; D5 += (nu32)NEBK_G_(5) ^ h8;
        D6 += (nu32)NEBK_G_(6) ^ h8 ^ 1u; D7 += (nu32)NEBK_G_(7) ^ h8 ^ 2u;
      }
    }
  } else {
    for (nu32 j = 0; direct && j < R->cin; j += 8u) {
      const nu32 a0 = NEBK_LOAD4(xa + j), a1 = NEBK_LOAD4(xa + j + 4u), b0 = NEBK_LOAD4(xb + j), b1 = NEBK_LOAD4(xb + j + 4u);
      const nu32 c0 = NEBK_LOAD4(xc + j), c1 = NEBK_LOAD4(xc + j + 4u);
      NEBK_EACH_ {
        NEBK_ERRS_
#define NEBK_T_(k) (t0 + j + k)
        NEBK_ACC8N(NEBK_G_, NEBK_T_)
#undef NEBK_T_
      }
    }
  }
#undef NEBK_G_
#undef NEBK_EACH_
#undef NEBK_ERRS_
  nu32 *out = slab + (nu64)e * 8u;
  NEBK_PUT8(out)
}

// Attempts with three live rows, the highest of them sparse (nbk_third). A sparse row s is a live sample whose
// gradient has few entries (nb_set_model picks them). The attempt's digest is the pair table's digest of its
// other two rows p and q, plus, at each entry where s is not zero, what the entry is with s's value less what it
// was without: clip(G + v) ^ t - clip(G) ^ t, with G = p's value + q's there. Everywhere else s adds nothing.
//
// The attempts wait in a bucket a sparse row and are done a bucket at a time: a block's threads all have the same
// s, so they walk the same entries in step, and they read p's and q's values at those entries from the gathered
// rows of s (gt): for every row x, x's values at s's entries, one after the other. Read from the rows themselves,
// each value cost a load of its own from anywhere in the row, and the card's memory could not keep up. A bucket's
// gathered rows (about 5 MB) stay in the card's L2 cache while its blocks run.
//
// A sparse row's description (nb_set_model writes it; the block copies it to shared memory), in words:
//   0..7    clip runs: chunks of 8 direct entries of lane 0..7 where the sum can pass the clip
//   8..15   the same where it cannot
//   16..23  chunks of 8 inputs of the factored layer, of lane 0..7
//   24      ne, outputs of the factored layer where s has an error;  25  bytes of a gathered row (stride)
//   26, 27  where s's gathered rows start in gt, in bytes (low, high);  28  pc, direct slots;  29  pj, input slots
//   32 ..   v[pc] (s's value at each direct slot), t[pc] (its tag), code[pc] (its place in a row, for the gather),
//           then e[ne rounded up to 4] (output o in the low half, s's error there in the high half, signed),
//           then j[pj] (each input slot's input), xs[pj] (s's input there)
// The slots are in the order of the runs: clip runs lane 0 to 7, then the others, then the inputs. A run is padded
// to a multiple of 8 with slots where s's value is 0, which add nothing. A gathered row is pc int8 (the row's
// values at the direct slots), then pj int8 (its inputs at the input slots), then a byte that is 1 when a value at
// a direct slot does not fit a byte (then nbk_third reads the row itself), then zeros to a multiple of 16 bytes.
// On the pool's Kepler model 99.97% of the direct values fit a byte, and the bytes read were what an attempt cost.
#define NEBK_M_HEAD 32u
#define NEBK_M_NE 24u
#define NEBK_M_STRIDE 25u
#define NEBK_M_AT 26u
#define NEBK_M_PC 28u
#define NEBK_M_PJ 29u

NEB_HD nu32 nebk_m_e(nu32 pc) { return NEBK_M_HEAD + 3u * pc; }

// Byte i of a uint2, signed.
#define NEBK_B8(w, i) ((int)(signed char)((((i) < 4 ? (w).x : (w).y) >> (8 * ((i) & 3))) & 0xffu))
#define NEBK_W4(v, i) ((i) < 2 ? ((i) == 0 ? (v).x : (v).y) : ((i) == 2 ? (v).z : (v).w))

// One run of direct slots, all of lane L, into DL. CLIPPED says whether the sum can pass 256. FETCH(i) sets ga and
// gb, p's and q's values at slot + i.
#define NEBK_RUN(DL, chunks, CLIPPED, LOAD, FETCH)                                                        \
  for (nu32 c_ = 0; c_ < (chunks); c_++, slot += 8u) {                                                    \
    LOAD                                                                                                  \
    const int4 v0 = *(const int4 *)(V + slot), v1 = *(const int4 *)(V + slot + 4u);                       \
    const uint4 t0 = *(const uint4 *)(T + slot), t1 = *(const uint4 *)(T + slot + 4u);                    \
    nu32 acc = 0;                                                                                         \
    _Pragma("unroll") for (int i = 0; i < 8; i++) {                                                       \
      int ga, gb;                                                                                         \
      FETCH(i)                                                                                            \
      const int g = ga + gb;                                                                              \
      const int v = i < 4 ? NEBK_W4(v0, i) : NEBK_W4(v1, i - 4);                                          \
      const nu32 t = i < 4 ? NEBK_W4(t0, i) : NEBK_W4(t1, i - 4);                                         \
      if (CLIPPED) acc += ((nu32)nebk_clip(g + v) ^ t) - ((nu32)nebk_clip(g) ^ t);                        \
      else acc += ((nu32)(g + v) ^ t) - ((nu32)g ^ t);                                                    \
    }                                                                                                     \
    DL += acc;                                                                                            \
  }
// From the gathered rows, a byte a value.
#define NEBK_LOAD8 const uint2 pa = *(const uint2 *)(gp + slot); const uint2 pb = *(const uint2 *)(gq + slot);
#define NEBK_FETCH8(i) ga = NEBK_B8(pa, i); gb = NEBK_B8(pb, i);
// From the rows themselves, for the few rows with a value there that a byte does not hold.
#define NEBK_LOADW
#define NEBK_FETCHW(i) ga = cp[C[slot + (i)]]; gb = cq[C[slot + (i)]];
#define NEBK_RUNS(LOAD, FETCH)                                                                                         \
  NEBK_RUN(D0, M[0], 1, LOAD, FETCH) NEBK_RUN(D1, M[1], 1, LOAD, FETCH) NEBK_RUN(D2, M[2], 1, LOAD, FETCH)           \
  NEBK_RUN(D3, M[3], 1, LOAD, FETCH) NEBK_RUN(D4, M[4], 1, LOAD, FETCH) NEBK_RUN(D5, M[5], 1, LOAD, FETCH)           \
  NEBK_RUN(D6, M[6], 1, LOAD, FETCH) NEBK_RUN(D7, M[7], 1, LOAD, FETCH)                                              \
  NEBK_RUN(D0, M[8], 0, LOAD, FETCH) NEBK_RUN(D1, M[9], 0, LOAD, FETCH) NEBK_RUN(D2, M[10], 0, LOAD, FETCH)          \
  NEBK_RUN(D3, M[11], 0, LOAD, FETCH) NEBK_RUN(D4, M[12], 0, LOAD, FETCH) NEBK_RUN(D5, M[13], 0, LOAD, FETCH)        \
  NEBK_RUN(D6, M[14], 0, LOAD, FETCH) NEBK_RUN(D7, M[15], 0, LOAD, FETCH)

// One run of input slots, all of lane L, into DL: each 8 inputs go to every output where s has an error. The
// factored layer's entries are at most 64 a row, so three rows' sum never reaches the clip.
#define NEBK_FRUN(DL, chunks)                                                                             \
  for (nu32 c_ = 0; c_ < (chunks); c_++, fs += 8u) {                                                      \
    const uint2 xa = *(const uint2 *)(gp + pc + fs), xb = *(const uint2 *)(gq + pc + fs);                 \
    int ap[8], bq[8], xs[8];                                                                              \
    nu32 jj[8];                                                                                           \
    _Pragma("unroll") for (int i = 0; i < 8; i++) {                                                       \
      ap[i] = (int)(signed char)(((i < 4 ? xa.x : xa.y) >> (8 * (i & 3))) & 0xffu);                       \
      bq[i] = (int)(signed char)(((i < 4 ? xb.x : xb.y) >> (8 * (i & 3))) & 0xffu);                       \
      jj[i] = J[fs + i];                                                                                  \
      xs[i] = XS[fs + i];                                                                                 \
    }                                                                                                     \
    nu32 acc = 0;                                                                                         \
    for (nu32 k_ = 0; k_ < ne; k_++) {                                                                    \
      const nu32 w = E[k_], o = w & 0xffffu;                                                              \
      const nu32 dd = sd[k_ * sstride];                                                               \
      const int ds = (int)w >> 16, dp = (int)(signed char)(dd & 0xffu), dq = (int)(signed char)(dd >> 8); \
      const nu32 tb = o * R->cin + R->base;                                                               \
      _Pragma("unroll") for (int i = 0; i < 8; i++) {                                                     \
        const int g = dp * ap[i] + dq * bq[i];                                                            \
        const nu32 t = tb + jj[i];                                                                        \
        acc += ((nu32)(g + ds * xs[i]) ^ t) - ((nu32)g ^ t);                                              \
      }                                                                                                   \
    }                                                                                                     \
    DL += acc;                                                                                            \
  }

// One record of sparse row s's bucket. M is s's description, gt the gathered rows of every sparse row. sd is the
// thread's own place in shared memory for p's and q's errors at s's outputs (a u16 each, sstride apart): read
// from d5 at every 8 inputs, they were two loads from anywhere in the card's memory each time.
NEB_DEV void nebk_third(const NebScan *S, const NebRows *R, const NebRec *rec, const nu32 *M, const unsigned char *gt,
                        const nu32 *pairs, const short *conv, const signed char *d5, unsigned short *sd, nu32 sstride,
                        nu32 *hits) {
  const nu32 p = rec->pq & 0xffffu, q = rec->pq >> 16;
  const nu32 *dg = pairs + (nu64)nebk_entry(p, q) * 8u;
  nu32 D0 = dg[0], D1 = dg[1], D2 = dg[2], D3 = dg[3], D4 = dg[4], D5 = dg[5], D6 = dg[6], D7 = dg[7];
  const nu32 stride = M[NEBK_M_STRIDE], pc = M[NEBK_M_PC], pj = M[NEBK_M_PJ], ne = M[NEBK_M_NE];
  const unsigned char *at = gt + ((nu64)M[NEBK_M_AT] | ((nu64)M[NEBK_M_AT + 1u] << 32));
#ifdef NB_DBG_ROWMASK
  // Timing only (wrong digests): every attempt reads the gathered rows of the first NB_DBG_ROWMASK + 1 rows.
  const unsigned char *gp = at + (nu64)(p & NB_DBG_ROWMASK) * stride, *gq = at + (nu64)(q & NB_DBG_ROWMASK) * stride;
#else
  const unsigned char *gp = at + (nu64)p * stride, *gq = at + (nu64)q * stride;
#endif
  const int *V = (const int *)(M + NEBK_M_HEAD);
  const nu32 *T = M + NEBK_M_HEAD + pc;
  const nu32 *E = M + nebk_m_e(pc);
  const nu32 *J = E + ((ne + 3u) & ~3u), *XS = J + pj;
  nu32 slot = 0;
#ifndef NB_DBG_NOCONV
  if (!(gp[pc + pj] | gq[pc + pj])) {
    NEBK_RUNS(NEBK_LOAD8, NEBK_FETCH8)
  } else {
    const nu32 *C = T + pc;
    const short *cp = conv + (nu64)p * R->nc, *cq = conv + (nu64)q * R->nc;
    NEBK_RUNS(NEBK_LOADW, NEBK_FETCHW)
  }
#endif
  const signed char *dpp = d5 + (nu64)p * R->cout, *dqq = d5 + (nu64)q * R->cout;
  for (nu32 k = 0; k < ne; k++) {
    const nu32 o = E[k] & 0xffffu;
    sd[k * sstride] = (unsigned short)((nu32)(unsigned char)dpp[o] | ((nu32)(unsigned char)dqq[o] << 8));
  }
  nu32 fs = 0;
#ifndef NB_DBG_NOFAC
  NEBK_FRUN(D0, M[16]) NEBK_FRUN(D1, M[17]) NEBK_FRUN(D2, M[18]) NEBK_FRUN(D3, M[19])
  NEBK_FRUN(D4, M[20]) NEBK_FRUN(D5, M[21]) NEBK_FRUN(D6, M[22]) NEBK_FRUN(D7, M[23])
#endif
  const nu32 D[8] = {D0, D1, D2, D3, D4, D5, D6, D7};
  nebk_ticket(S, rec->nonce, rec->key, D, hits);
}

// The gathered row of row x for sparse row s (description M): see nebk_third.
NEB_DEV void nebk_gather(nu32 x, const NebRows *R, const nu32 *M, const short *conv, const signed char *x5,
                         unsigned char *gt) {
  const nu32 stride = M[NEBK_M_STRIDE], pc = M[NEBK_M_PC], pj = M[NEBK_M_PJ], ne = M[NEBK_M_NE];
  unsigned char *out = gt + ((nu64)M[NEBK_M_AT] | ((nu64)M[NEBK_M_AT + 1u] << 32)) + (nu64)x * stride;
  const nu32 *code = M + NEBK_M_HEAD + 2u * pc;
  const nu32 *J = M + nebk_m_e(pc) + ((ne + 3u) & ~3u);
  const short *row = conv + (nu64)x * R->nc;
  unsigned char wide = 0;
  for (nu32 i = 0; i < pc; i++) {
    const int v = row[code[i]];
    out[i] = (unsigned char)(signed char)v;
    wide |= (v < -128 || v > 127) ? 1u : 0u;
  }
  const signed char *xr = x5 + (nu64)x * R->cin;
  for (nu32 i = 0; i < pj; i++) out[pc + i] = (unsigned char)xr[J[i]];
  out[pc + pj] = wide;
  for (nu32 i = pc + pj + 1u; i < stride; i++) out[i] = 0;
}

// One attempt in full, from its key (8 words): the digest then the ticket, 16 words.
NEB_DEV void nebk_full(const nu32 *key, const NebRows *R, const unsigned short *live, const short *conv,
                       const nu32 *tag, const signed char *d5, const signed char *x5, nu32 *out) {
  nu32 u[16], row[NEBK_NB];
  nebk_chacha8(key, 1u, u);
  for (int j = 0; j < NEBK_NB; j++) row[j] = live[(nu32)(((nu64)u[j] * R->n) >> 32)];
  nu32 D[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (nu32 e = 0; e < R->nc; e++) {
    int g = 0;
    for (int b = 0; b < NEBK_NB; b++) g += conv[(nu64)row[b] * R->nc + e];
    D[e & 7u] += (nu32)nebk_clip(g) ^ tag[e];
  }
  for (nu32 o = 0; o < R->cout; o++) {
    int d[NEBK_NB];
    for (int b = 0; b < NEBK_NB; b++) d[b] = d5[(nu64)row[b] * R->cout + o];
    const nu32 f0 = o * R->cin + R->base;
    for (nu32 j = 0; j < R->cin; j++) {
      int g = 0;
      for (int b = 0; b < NEBK_NB; b++) g += d[b] * (int)x5[(nu64)row[b] * R->cin + j];
      D[j & 7u] += (nu32)nebk_clip(g) ^ (f0 + j);
    }
  }
  nu32 blk[16], iv[8];
  for (int i = 0; i < 8; i++) { blk[i] = D[i]; blk[8 + i] = key[i]; out[i] = D[i]; }
  iv[0] = 0x6A09E667u; iv[1] = 0xBB67AE85u; iv[2] = 0x3C6EF372u; iv[3] = 0xA54FF53Au;
  iv[4] = 0x510E527Fu; iv[5] = 0x9B05688Cu; iv[6] = 0x1F83D9ABu; iv[7] = 0x5BE0CD19u;
  nebk_b3(iv, blk, 64u, 1u | 2u | 8u, out + 8);
}

#endif
