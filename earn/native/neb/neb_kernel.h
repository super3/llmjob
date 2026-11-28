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
//   nebk_screen  a thread's nonces, one after another: key = BLAKE3(header), the 16 sample indices
//                from ChaCha8(key), and a look at which are live. An attempt with at most max_live
//                live rows (2, or 1 when the pair table does not fit), or with 3 that are all among
//                the rows the triple table covers, goes on the candidate list with its key and the
//                table entry of its rows.
//   nebk_finish  a thread's candidates: the digest is read from the model's table,
//                ticket = BLAKE3(digest || key), and a ticket at or under the bound is a hit.
//   nebk_pair    one entry of the pair table: the digest of the clipped sum of two rows.
//   nebk_out1, nebk_out2, nebk_triple   the triple table, one slab (the entries that share their
//                highest row) at a time: per-output digests of each row alone and of each row with
//                the slab's row, then every entry of the slab from them.
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
  nu32 cap;        // candidates the list holds
  nu32 bound[8];   // the target, a 256-bit little-endian integer
  nu32 max_hits;
};

struct NebCand {
  nu32 nonce;
  nu32 entry;      // the table entry of its rows
  nu32 key[8];
};

// A model's rows on the card, and the layout of its digest.
struct NebRows {
  nu32 m1;         // rows, counting row 0, the zero row
  nu32 nc;         // direct entries a row: every layer but the factored one, int16 each
  nu32 cin, cout;  // the factored layer: its gradient is the outer product of cout errors and cin inputs
  nu32 base;       // 7919 * (its layer number)
  nu32 n;          // samples in the dataset
};

NEB_HD nu32 nebk_rotr(nu32 x, int n) { return (x >> n) | (x << (32 - n)); }
NEB_HD nu32 nebk_rotl(nu32 x, int n) { return (x << n) | (x >> (32 - n)); }

#define NEBK_G(a, b, c, d, mx, my)                              \
  v[a] = v[a] + v[b] + (mx); v[d] = nebk_rotr(v[d] ^ v[a], 16); \
  v[c] = v[c] + v[d];        v[b] = nebk_rotr(v[b] ^ v[c], 12); \
  v[a] = v[a] + v[b] + (my); v[d] = nebk_rotr(v[d] ^ v[a], 8);  \
  v[c] = v[c] + v[d];        v[b] = nebk_rotr(v[b] ^ v[c], 7);

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
  for (int i = 0; i < 8; i++) v[i] = cv[i];
  v[8] = 0x6A09E667u; v[9] = 0xBB67AE85u; v[10] = 0x3C6EF372u; v[11] = 0xA54FF53Au;
  v[12] = 0; v[13] = 0; v[14] = 16u; v[15] = 2u | 8u;  // 16 bytes; CHUNK_END | ROOT
  NEBK_G(0, 4, 8, 12, tail[0], tail[1])
  NEBK_G(2, 6, 10, 14, 0u, 0u)
  NEBK_G(3, 7, 11, 15, 0u, 0u)
  v[1] = v[1] + v[5] + tail[2]; v[13] = nebk_rotr(v[13] ^ v[1], 16);
  v[9] = v[9] + v[13];          v[5] = nebk_rotr(v[5] ^ v[9], 12);
}

// The key of the header with this nonce: the rest of the compression nebk_pre began.
NEB_HD void nebk_key(const nu32 *pre, const nu32 *tail, nu32 nonce, nu32 *key) {
  nu32 m[16], v[16], t[16];
  for (int i = 0; i < 16; i++) v[i] = pre[i];
  m[0] = tail[0]; m[1] = tail[1]; m[2] = tail[2]; m[3] = nonce;
  for (int i = 4; i < 16; i++) m[i] = 0;
  v[1] = v[1] + v[5] + nonce; v[13] = nebk_rotr(v[13] ^ v[1], 8);
  v[9] = v[9] + v[13];        v[5] = nebk_rotr(v[5] ^ v[9], 7);
  NEBK_DIAGONALS
  for (int r = 1; r < 7; r++) {
    NEBK_PERMUTE
    NEBK_COLUMNS
    NEBK_DIAGONALS
  }
  for (int i = 0; i < 8; i++) key[i] = v[i] ^ v[i + 8];
}

#define NEBK_QR(a, b, c, d)                                        \
  x[a] += x[b]; x[d] = nebk_rotl(x[d] ^ x[a], 16); x[c] += x[d]; x[b] = nebk_rotl(x[b] ^ x[c], 12); \
  x[a] += x[b]; x[d] = nebk_rotl(x[d] ^ x[a], 8);  x[c] += x[d]; x[b] = nebk_rotl(x[b] ^ x[c], 7);

// Block 0 of ChaCha8(key): 16 words.
NEB_HD void nebk_chacha8(const nu32 *key, nu32 *out) {
  nu32 s[16], x[16];
  s[0] = 0x61707865u; s[1] = 0x3320646eu; s[2] = 0x79622d32u; s[3] = 0x6b206574u;
  for (int i = 0; i < 8; i++) s[4 + i] = key[i];
  s[12] = 0; s[13] = 0; s[14] = 0; s[15] = 0;
  for (int i = 0; i < 16; i++) x[i] = s[i];
  for (int r = 0; r < 4; r++) {
    NEBK_QR(0, 4, 8, 12) NEBK_QR(1, 5, 9, 13) NEBK_QR(2, 6, 10, 14) NEBK_QR(3, 7, 11, 15)
    NEBK_QR(0, 5, 10, 15) NEBK_QR(1, 6, 11, 12) NEBK_QR(2, 7, 8, 13) NEBK_QR(3, 4, 9, 14)
  }
  for (int i = 0; i < 16; i++) out[i] = x[i] + s[i];
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

// A thread's nonces. live[s] is 0 for a dead sample and the sample's row (1 and up) otherwise.
NEB_DEV void nebk_screen(nu32 gid, const NebScan *S, const unsigned short *live, nu32 *ncand, NebCand *cand) {
  for (nu32 it = 0; it < S->iters; it++) {
    const nu32 nonce = S->start + gid + it * S->threads;
    nu32 key[8], u[16];
    nebk_key(S->pre, S->tail, nonce, key);
    nebk_chacha8(key, u);
    nu32 cnt = 0, a = 0, b = 0, c = 0;
    for (int j = 0; j < NEBK_NB; j++) {
      const nu32 r = live[(nu32)(((nu64)u[j] * S->n) >> 32)];
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
      continue;
    }
    const nu32 at = NEB_ATOMIC_INC(ncand);
    if (at < S->cap) {
      cand[at].nonce = nonce;
      cand[at].entry = entry;
      for (int i = 0; i < 8; i++) cand[at].key[i] = key[i];
    }
  }
}

// A thread's candidates: gid, gid + threads, ... below the count the screen left. hits[0] counts the
// hits; the nonces follow.
NEB_DEV void nebk_finish(nu32 gid, nu32 threads, const NebScan *S, const nu32 *ncand, const NebCand *cand,
                         const nu32 *pairs, const nu32 *triples, nu32 *hits) {
  nu32 n = *ncand;
  if (n > S->cap) n = S->cap;
  for (nu32 at = gid; at < n; at += threads) {
    const nu32 e = cand[at].entry;
    const nu32 *dg = (e & NEBK_TRIPLE) ? triples + (nu64)(e & ~NEBK_TRIPLE) * 8u : pairs + (nu64)e * 8u;
    nu32 blk[16], iv[8], t[8];
    for (int i = 0; i < 8; i++) { blk[i] = dg[i]; blk[8 + i] = cand[at].key[i]; }
    iv[0] = 0x6A09E667u; iv[1] = 0xBB67AE85u; iv[2] = 0x3C6EF372u; iv[3] = 0xA54FF53Au;
    iv[4] = 0x510E527Fu; iv[5] = 0x9B05688Cu; iv[6] = 0x1F83D9ABu; iv[7] = 0x5BE0CD19u;
    nebk_b3(iv, blk, 64u, 1u | 2u | 8u, t);  // CHUNK_START | CHUNK_END | ROOT
    bool ok = true;
    for (int k = 7; k >= 0; k--) {
      if (t[k] != S->bound[k]) { ok = t[k] < S->bound[k]; break; }
    }
    if (!ok) continue;
    const nu32 h = NEB_ATOMIC_INC(hits);
    if (h < S->max_hits) hits[1u + h] = cand[at].nonce;
  }
}

NEB_HD int nebk_clip(int g) { return g < -NEBK_CLIP ? -NEBK_CLIP : (g > NEBK_CLIP ? NEBK_CLIP : g); }

// One pair-table entry: the digest of rows p and q summed. With pairs == 0 the table holds single rows
// and entry e is row e alone.
//   conv   m1 x nc int16: the direct entries of each row
//   tag    nc words: f + 7919 * layer of each direct entry
//   lane   nc bytes: f mod 8 of each
//   d5     m1 x cout int8, x5 m1 x cin int8: the factored layer
//   zero5  cout x 8 words: what a row of that layer adds to each lane when its gradient is all zero
NEB_DEV void nebk_pair(nu32 entry, nu32 pairs, const NebRows *R, const short *conv, const nu32 *tag,
                       const unsigned char *lane, const signed char *d5, const signed char *x5, const nu32 *zero5,
                       nu32 *table) {
  nu32 p = 0, q = entry;
  if (pairs) nebk_unpair(entry, &p, &q);
  nu32 D[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const short *cp = conv + (nu64)p * R->nc, *cq = conv + (nu64)q * R->nc;
  for (nu32 e = 0; e < R->nc; e++) D[lane[e]] += (nu32)nebk_clip((int)cp[e] + (int)cq[e]) ^ tag[e];
  const signed char *xp = x5 + (nu64)p * R->cin, *xq = x5 + (nu64)q * R->cin;
  for (nu32 o = 0; o < R->cout; o++) {
    const int dp = d5[(nu64)p * R->cout + o], dq = d5[(nu64)q * R->cout + o];
    if ((dp | dq) == 0) {
      for (int k = 0; k < 8; k++) D[k] += zero5[o * 8u + k];
      continue;
    }
    const nu32 f0 = o * R->cin + R->base;
    for (nu32 j = 0; j < R->cin; j++) D[j & 7u] += (nu32)nebk_clip(dp * (int)xp[j] + dq * (int)xq[j]) ^ (f0 + j);
  }
  nu32 *out = table + (nu64)entry * 8u;
  for (int k = 0; k < 8; k++) out[k] = D[k];
}

// The triple table is built a slab at a time: slab c holds the entries (a, b, c) with a <= b <= c.
// Most of an entry's work is the factored layer, and most of that can be shared. For one output o of
// that layer, the rows' errors d decide the work:
//   all three zero     the constant zero5[o]
//   only one non-zero  that row alone: out1, built once a model
//   a or b zero        the other with c: out2, built once a slab
//   a and b non-zero   summed entry by entry
// On a live model a third to a half of a row's outputs have an error, so the last case, the costly one,
// is one output in ten to one in four.

// out1[row][o]: the 8 lanes output o adds for `row` alone. `index` is row * cout + o.
NEB_DEV void nebk_out1(nu32 index, const NebRows *R, const signed char *d5, const signed char *x5, nu32 *out1) {
  const nu32 row = index / R->cout, o = index % R->cout;
  const int d = d5[index];
  nu32 D[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  if (d != 0) {
    const signed char *x = x5 + (nu64)row * R->cin;
    const nu32 f0 = o * R->cin + R->base;
    for (nu32 j = 0; j < R->cin; j++) D[j & 7u] += (nu32)nebk_clip(d * (int)x[j]) ^ (f0 + j);
  }
  for (int k = 0; k < 8; k++) out1[(nu64)index * 8u + k] = D[k];
}

// out2[row][o]: the same for `row` summed with the slab's row c (c is a row number, 1 and up).
NEB_DEV void nebk_out2(nu32 index, nu32 c, const NebRows *R, const signed char *d5, const signed char *x5, nu32 *out2) {
  const nu32 row = index / R->cout, o = index % R->cout;
  const int d = d5[index], dc = d5[(nu64)c * R->cout + o];
  nu32 D[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  if ((d | dc) != 0) {
    const signed char *x = x5 + (nu64)row * R->cin, *xc = x5 + (nu64)c * R->cin;
    const nu32 f0 = o * R->cin + R->base;
    for (nu32 j = 0; j < R->cin; j++) D[j & 7u] += (nu32)nebk_clip(d * (int)x[j] + dc * (int)xc[j]) ^ (f0 + j);
  }
  for (int k = 0; k < 8; k++) out2[(nu64)index * 8u + k] = D[k];
}

// Entry `e` of slab c: rows a <= b <= c from nebk_unpair(e), each plus 1. `slab` is where the slab starts in the table.
NEB_DEV void nebk_triple(nu32 e, nu32 c, const NebRows *R, const short *conv, const nu32 *tag,
                         const unsigned char *lane, const signed char *d5, const signed char *x5, const nu32 *zero5,
                         const nu32 *out1, const nu32 *out2, nu32 *slab) {
  nu32 a, b;
  nebk_unpair(e, &a, &b);
  a++;
  b++;
  nu32 D[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const short *ca = conv + (nu64)a * R->nc, *cb = conv + (nu64)b * R->nc, *cc = conv + (nu64)c * R->nc;
  for (nu32 i = 0; i < R->nc; i++) D[lane[i]] += (nu32)nebk_clip((int)ca[i] + (int)cb[i] + (int)cc[i]) ^ tag[i];
  const signed char *xa = x5 + (nu64)a * R->cin, *xb = x5 + (nu64)b * R->cin, *xc = x5 + (nu64)c * R->cin;
  for (nu32 o = 0; o < R->cout; o++) {
    const int da = d5[(nu64)a * R->cout + o], db = d5[(nu64)b * R->cout + o], dc = d5[(nu64)c * R->cout + o];
    const nu32 *part;
    if (da != 0 && db != 0) {
      const nu32 f0 = o * R->cin + R->base;
      for (nu32 j = 0; j < R->cin; j++)
        D[j & 7u] += (nu32)nebk_clip(da * (int)xa[j] + db * (int)xb[j] + dc * (int)xc[j]) ^ (f0 + j);
      continue;
    }
    if (da != 0) part = dc != 0 ? out2 + ((nu64)a * R->cout + o) * 8u : out1 + ((nu64)a * R->cout + o) * 8u;
    else if (db != 0) part = dc != 0 ? out2 + ((nu64)b * R->cout + o) * 8u : out1 + ((nu64)b * R->cout + o) * 8u;
    else if (dc != 0) part = out1 + ((nu64)c * R->cout + o) * 8u;
    else part = zero5 + o * 8u;
    for (int k = 0; k < 8; k++) D[k] += part[k];
  }
  nu32 *out = slab + (nu64)e * 8u;
  for (int k = 0; k < 8; k++) out[k] = D[k];
}

// One attempt in full, from its key (8 words): the digest then the ticket, 16 words.
NEB_DEV void nebk_full(const nu32 *key, const NebRows *R, const unsigned short *live, const short *conv,
                       const nu32 *tag, const unsigned char *lane, const signed char *d5, const signed char *x5,
                       nu32 *out) {
  nu32 u[16], row[NEBK_NB];
  nebk_chacha8(key, u);
  for (int j = 0; j < NEBK_NB; j++) row[j] = live[(nu32)(((nu64)u[j] * R->n) >> 32)];
  nu32 D[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (nu32 e = 0; e < R->nc; e++) {
    int g = 0;
    for (int b = 0; b < NEBK_NB; b++) g += conv[(nu64)row[b] * R->nc + e];
    D[lane[e]] += (nu32)nebk_clip(g) ^ tag[e];
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
