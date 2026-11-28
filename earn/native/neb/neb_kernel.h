// The device code of the NEB (nebhash v2) cores, written once for every vendor.
//
// Nothing here names CUDA or HIP. The file that includes it defines
//   NEB_DEV            the qualifier of a device function
//   NEB_HD             the qualifier of a function the host calls too (the hashes)
//   NEB_ATOMIC_INC(p)  adds 1 to the u32 at p, atomically, and gives the value before
// and wraps each nebk_* body in a kernel of its own (cuda/neb_cuda.cu, amd/neb_hip.cpp).
//
// How a card mines NEB. An attempt's gradient is the sum of the gradients of its 16 samples, and a
// sample whose output already equals its target has a zero gradient: a dead row. On a trained model
// most samples are dead, so most of the work is finding the attempts that touch few live rows.
//
//   nebk_screen  one nonce: key = BLAKE3(header), the 16 sample indices from ChaCha8(key), and a look
//                at which are live. An attempt with at most max_live live rows (2, or 1 when the pair
//                table does not fit) goes on the candidate list with the index of its row pair.
//   nebk_finish  one candidate: its digest is read from the model's table, ticket = BLAKE3(digest ||
//                key), and a ticket at or under the bound is a hit.
//   nebk_pair    one entry of the model's table: the digest of the clipped sum of two rows.
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

// One scan's constants.
struct NebScan {
  nu32 cv[8];      // BLAKE3's state after the header's first 64 bytes
  nu32 tail[3];    // header words 16, 17, 18; word 19 is the nonce
  nu32 start;      // the nonce of thread 0
  nu32 n;          // samples in the dataset
  nu32 max_live;   // 2: the pair table; 1: the table of single rows
  nu32 cap;        // candidates the list holds
  nu32 bound[8];   // the target, a 256-bit little-endian integer
  nu32 max_hits;
};

struct NebCand {
  nu32 nonce;
  nu32 entry;      // the table entry of its rows
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

// BLAKE3's compression function for block counter 0: the first 8 words of the output.
NEB_HD void nebk_b3(const nu32 *cv, const nu32 *blk, nu32 blen, nu32 flags, nu32 *out) {
  nu32 m[16], v[16], t[16];
  for (int i = 0; i < 16; i++) m[i] = blk[i];
  for (int i = 0; i < 8; i++) v[i] = cv[i];
  v[8] = 0x6A09E667u; v[9] = 0xBB67AE85u; v[10] = 0x3C6EF372u; v[11] = 0xA54FF53Au;
  v[12] = 0; v[13] = 0; v[14] = blen; v[15] = flags;
  for (int r = 0; r < 7; r++) {
    NEBK_G(0, 4, 8, 12, m[0], m[1])   NEBK_G(1, 5, 9, 13, m[2], m[3])
    NEBK_G(2, 6, 10, 14, m[4], m[5])  NEBK_G(3, 7, 11, 15, m[6], m[7])
    NEBK_G(0, 5, 10, 15, m[8], m[9])  NEBK_G(1, 6, 11, 12, m[10], m[11])
    NEBK_G(2, 7, 8, 13, m[12], m[13]) NEBK_G(3, 4, 9, 14, m[14], m[15])
    if (r < 6) {
      t[0] = m[2]; t[1] = m[6]; t[2] = m[3]; t[3] = m[10]; t[4] = m[7]; t[5] = m[0]; t[6] = m[4]; t[7] = m[13];
      t[8] = m[1]; t[9] = m[11]; t[10] = m[12]; t[11] = m[5]; t[12] = m[9]; t[13] = m[14]; t[14] = m[15]; t[15] = m[8];
      for (int i = 0; i < 16; i++) m[i] = t[i];
    }
  }
  for (int i = 0; i < 8; i++) out[i] = v[i] ^ v[i + 8];
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

// The key of the header with this nonce.
NEB_DEV void nebk_key(const NebScan *S, nu32 nonce, nu32 *key) {
  nu32 blk[16];
  blk[0] = S->tail[0]; blk[1] = S->tail[1]; blk[2] = S->tail[2]; blk[3] = nonce;
  for (int i = 4; i < 16; i++) blk[i] = 0;
  nebk_b3(S->cv, blk, 16u, 2u | 8u, key);  // CHUNK_END | ROOT
}

// The table entry of rows p <= q (0 is the zero row): q (q + 1) / 2 + p.
NEB_HD nu32 nebk_entry(nu32 p, nu32 q) { return q * (q + 1u) / 2u + p; }

// One nonce. live[s] is 0 for a dead sample and the sample's row (1 and up) otherwise.
NEB_DEV void nebk_screen(nu32 gid, const NebScan *S, const unsigned short *live, nu32 *ncand, NebCand *cand) {
  const nu32 nonce = S->start + gid;
  nu32 key[8], u[16];
  nebk_key(S, nonce, key);
  nebk_chacha8(key, u);
  nu32 cnt = 0, p = 0, q = 0;
  for (int j = 0; j < NEBK_NB; j++) {
    const nu32 s = (nu32)(((nu64)u[j] * S->n) >> 32);
    const nu32 r = live[s];
    if (r) {
      if (cnt == 0) q = r;
      else if (cnt == 1) p = r;
      cnt++;
    }
  }
  if (cnt > S->max_live) return;
  if (p > q) { const nu32 t = p; p = q; q = t; }
  const nu32 at = NEB_ATOMIC_INC(ncand);
  if (at < S->cap) {
    cand[at].nonce = nonce;
    cand[at].entry = S->max_live >= 2u ? nebk_entry(p, q) : q;
  }
}

// One candidate. hits[0] counts the hits; the nonces follow.
NEB_DEV void nebk_finish(nu32 gid, const NebScan *S, const NebCand *cand, const nu32 *table, nu32 *hits) {
  const nu32 nonce = cand[gid].nonce;
  const nu32 *dg = table + (nu64)cand[gid].entry * 8u;
  nu32 blk[16], iv[8], t[8];
  nebk_key(S, nonce, blk + 8);
  for (int i = 0; i < 8; i++) blk[i] = dg[i];
  iv[0] = 0x6A09E667u; iv[1] = 0xBB67AE85u; iv[2] = 0x3C6EF372u; iv[3] = 0xA54FF53Au;
  iv[4] = 0x510E527Fu; iv[5] = 0x9B05688Cu; iv[6] = 0x1F83D9ABu; iv[7] = 0x5BE0CD19u;
  nebk_b3(iv, blk, 64u, 1u | 2u | 8u, t);  // CHUNK_START | CHUNK_END | ROOT
  bool ok = true;
  for (int k = 7; k >= 0; k--) {
    if (t[k] != S->bound[k]) { ok = t[k] < S->bound[k]; break; }
  }
  if (!ok) return;
  const nu32 at = NEB_ATOMIC_INC(hits);
  if (at < S->max_hits) hits[1u + at] = nonce;
}

NEB_HD int nebk_clip(int g) { return g < -NEBK_CLIP ? -NEBK_CLIP : (g > NEBK_CLIP ? NEBK_CLIP : g); }

// One table entry: the digest of rows p and q summed. With pairs == 0 the table holds single rows
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
  if (pairs) {
    q = (nu32)((sqrtf(8.0f * (float)entry + 1.0f) - 1.0f) * 0.5f);
    while ((nu64)q * (q + 1u) / 2u > entry) q--;
    while ((nu64)(q + 1u) * (q + 2u) / 2u <= entry) q++;
    p = entry - q * (q + 1u) / 2u;
  }
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
