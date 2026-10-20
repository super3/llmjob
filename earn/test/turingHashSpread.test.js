'use strict';

const fs = require('fs');
const path = require('path');
const { keyedHash, IV, MSG_PERMUTATION } = require('../src/shared/miner/blake3');

// Turing's spread transcript hash (PEARL_BD_HASH_SPREAD, pearl_fold_bd.cuh). Every
// warp hashes its own 16 regions of the last tile while it folds the next: two lanes
// a region, two of the state's four columns a lane, one step (two G and a fixed
// exchange between the lanes) at each chunk. Nothing here runs the device code, so
// this runs the same schedule on the CPU, with the message table and the exchange
// as the source has them, and checks the word the target test reads against BLAKE3
// itself. It also checks the shared addresses the steps read against the words the
// readout writes. The source lines the schedule copies are pinned below.

const SRC = path.join(__dirname, '..', 'native', 'src');
const NL = String.fromCharCode(10);
const read = (name) => fs.readFileSync(path.join(SRC, name), 'utf8')
  .split(String.fromCharCode(13) + NL).join(NL);
const KERNEL = read('pearl_kernel.cu');
const FOLD = read('pearl_fold_bd.cuh');
const HEADER = read('pearl_config.h');

const FLAGS = 1 | 2 | 8 | 16;  // CHUNK_START | CHUNK_END | ROOT | KEYED_HASH

// The table as the source writes it: 30 words, step q and lane parity h at 2q + h.
function sourceTable() {
  const at = FOLD.indexOf('#define PEARL_BD_HS_MSG');
  expect(at).toBeGreaterThanOrEqual(0);
  const open = FOLD.indexOf('{', at);
  const close = FOLD.indexOf('}', open);
  return FOLD.slice(open + 1, close).split(',').map((s) => s.replace(/\\/g, '').trim())
    .filter((s) => s.length).map((s) => parseInt(s.replace(/u$/, ''), 16) >>> 0);
}

// The table rebuilt from BLAKE3's message permutation. Step q is half a round:
// round q >> 1, columns when q is even and diagonals when odd. After the exchange
// slot k holds diagonal (1 - k) mod 4, so its G reads message words 8 + 2i and
// 9 + 2i of that round with i = (1 - k) & 3; on a column step slot k reads 2k and
// 2k + 1. Lane h has slots 2h and 2h + 1. Step 14 reads step 0's words.
function builtTable() {
  const out = [];
  for (let q = 0; q < 15; q++) {
    const qq = q < 14 ? q : 0;
    let p = [...Array(16).keys()];
    for (let r = 0; r < qq >> 1; r++) p = p.map((_, j) => p[MSG_PERMUTATION[j]]);
    for (let h = 0; h < 2; h++) {
      let w = 0;
      [2 * h, 2 * h + 1].forEach((k, i) => {
        const j = qq & 1 ? 8 + 2 * ((1 - k) & 3) : 2 * k;
        w |= (p[j] << 2) << (16 * i);
        w |= (p[j + 1] << 2) << (16 * i + 8);
      });
      out.push(w >>> 0);
    }
  }
  return out;
}

const rotr = (x, n) => ((x >>> n) | (x << (32 - n))) >>> 0;
function g(s, a, b, c, d, mx, my) {
  s[a] = (s[a] + s[b] + mx) >>> 0;
  s[d] = rotr(s[d] ^ s[a], 16);
  s[c] = (s[c] + s[d]) >>> 0;
  s[b] = rotr(s[b] ^ s[c], 12);
  s[a] = (s[a] + s[b] + my) >>> 0;
  s[d] = rotr(s[d] ^ s[a], 8);
  s[c] = (s[c] + s[d]) >>> 0;
  s[b] = rotr(s[b] ^ s[c], 7);
}

// __byte_perm(x, y, s): byte n of the result is byte (s >> 4n) & 7 of y:x.
function bytePerm(x, y, s) {
  const b = [0, 1, 2, 3].map((i) => (x >>> (8 * i)) & 0xff)
    .concat([0, 1, 2, 3].map((i) => (y >>> (8 * i)) & 0xff));
  let r = 0;
  for (let n = 0; n < 4; n++) r |= b[(s >>> (4 * n)) & 7] << (8 * n);
  return r >>> 0;
}

// hs_init, hs_step and the test's word, for one region (two lanes), as the fold
// runs them: fifteen steps, the word kept at the start of each.
function spreadMsw(key, m, table, bigEndian) {
  const lanes = [0, 1].map((h) => [
    key[2 * h], key[2 * h + 1], key[4 + 2 * h], key[5 + 2 * h],
    IV[2 * h], IV[2 * h + 1], h ? 64 : 0, h ? FLAGS : 0,
  ].map((v) => v >>> 0));
  const w = [0, 0];
  for (let q = 0; q < 15; q++) {
    for (let h = 0; h < 2; h++) {
      const hs = lanes[h];
      w[h] = bigEndian ? (hs[0] ^ hs[4]) >>> 0 : (hs[3] ^ hs[7]) >>> 0;
      const t = table[2 * q + h];
      const mw = [0, 1, 2, 3].map((i) => m[((t >>> (8 * i)) & 0xff) >> 2]);
      g(hs, 0, 2, 4, 6, mw[0], mw[1]);
      g(hs, 1, 3, 5, 7, mw[2], mw[3]);
    }
    // T: a_f and a_s swap; b_f and d_s from the other lane; c_f and c_s from the
    // other lane's c_s and c_f.
    const [p0, p1] = lanes.map((hs) => hs.slice());
    for (let h = 0; h < 2; h++) {
      const o = h ? p1 : p0;
      const p = h ? p0 : p1;
      const hs = lanes[h];
      hs[0] = o[1];
      hs[1] = o[0];
      hs[2] = p[2];
      hs[4] = p[5];
      hs[5] = p[4];
      hs[7] = p[7];
    }
  }
  // The lane that holds the word: 0 for out16[0], 1 for out16[7].
  const word = bigEndian ? w[0] : w[1];
  if (!bigEndian) return word;
  return (((word & 0xff) << 24) | ((word & 0xff00) << 8) | ((word >>> 8) & 0xff00)
    | (word >>> 24)) >>> 0;
}

function words(buf) {
  return [...Array(buf.length / 4).keys()].map((i) => buf.readUInt32LE(4 * i));
}

describe('Turing spread hash (PEARL_BD_HASH_SPREAD)', () => {
  test('is off by default', () => {
    const at = HEADER.indexOf('#ifndef PEARL_BD_HASH_SPREAD');
    expect(at).toBeGreaterThanOrEqual(0);
    expect(HEADER.slice(at).split(NL)[1].trim()).toBe('#define PEARL_BD_HASH_SPREAD 0');
  });

  test('the message table is BLAKE3\'s permutation, step by step', () => {
    const table = sourceTable();
    expect(table).toHaveLength(30);
    expect(table).toEqual(builtTable());
  });

  test('fifteen steps give the word the target test reads, in both byte orders', () => {
    const table = sourceTable();
    let x = 0x2545f491;
    const next = () => {
      x ^= x << 13; x >>>= 0;
      x ^= x >>> 17;
      x ^= x << 5; x >>>= 0;
      return x;
    };
    for (let n = 0; n < 300; n++) {
      const keyBytes = Buffer.alloc(32);
      const transcript = Buffer.alloc(64);
      for (let i = 0; i < 32; i++) keyBytes[i] = next() & 0xff;
      for (let i = 0; i < 64; i++) transcript[i] = next() & 0xff;
      const out = words(keyedHash(keyBytes, transcript));
      const key = words(keyBytes);
      const m = words(transcript);
      // pearl_transcript_msw: out16[7], or out16[0] with its bytes reversed.
      expect(spreadMsw(key, m, table, false)).toBe(out[7]);
      const be = out[0];
      expect(spreadMsw(key, m, table, true)).toBe((((be & 0xff) << 24) | ((be & 0xff00) << 8)
        | ((be >>> 8) & 0xff00) | (be >>> 24)) >>> 0);
    }
  });

  // A step's four words: byte i of the table word, xored with hs_k, put in byte 0 of
  // hs_base. For every region of every warp, both buffers and every word, that is
  // the word the readout stored there and hash_tile reads.
  test('each step reads its region\'s words, where the readout wrote them', () => {
    const TWORDS = 8 * 16 * 16;
    for (let warp = 0; warp < 8; warp++) {
      for (let lane = 0; lane < 32; lane++) {
        const L = warp * 16 + (lane >> 1);
        const k = ((((L >> 1) & 15) << 2 | (L & 3) << 6) * 0x01010101) >>> 0;
        for (const buf of [0, TWORDS]) {
          const base = buf * 4 + (L >> 2) * 256;
          for (let c = 0; c < 16; c++) {
            const t = ((c << 2) * 0x01010101) >>> 0;
            for (let i = 0; i < 4; i++) {
              expect(bytePerm((t ^ k) >>> 0, base, 0x7650 + i))
                .toBe(4 * (buf + L * 16 + (c ^ ((L >> 1) & 15))));
            }
          }
        }
        // The readout's writers of this warp cover exactly regions 16 warp ..+15.
      }
      const written = new Set();
      for (let lane = 0; lane < 32; lane++) {
        if (((lane >> 2) & 3) !== 0) continue;
        for (let rl = 0; rl < 2; rl++) {
          written.add(warp * 16 + (lane & 3) * 4 + 2 * rl + ((lane >> 4) & 1));
        }
      }
      expect([...written].sort((a, b) => a - b))
        .toEqual([...Array(16).keys()].map((r) => warp * 16 + r));
    }
  });

  // The late warps are the ones that hashed at the hand-off: one of each scheduler's
  // two warps (w and w + 4) and two of each row group's four.
  test('the late warps split every scheduler and row group', () => {
    const late = [...Array(8).keys()].filter((w) => ((w & 3) < 2) !== ((w >> 2) !== 0));
    expect(late).toEqual([0, 1, 6, 7]);
    for (let s = 0; s < 4; s++) expect(late.includes(s) !== late.includes(s + 4)).toBe(true);
  });

  test('the schedule above matches the source', () => {
    expect(FOLD).toContain(
      'const bool hs_late = __any_sync(0xffffffffu, ((warp & 3u) < 2u) != (wr != 0u));');
    expect(FOLD).toContain('const uint32_t hs_L = warp * warp_regions + (lane >> 1);');
    expect(FOLD).toContain(
      'const uint32_t hs_k = (((hs_L >> 1) & 15u) << 2 | (hs_L & 3u) << 6) * 0x01010101u;');
    expect(FOLD).toContain('hs[0] = h1 ? test.key[2] : test.key[0];');
    expect(FOLD).toContain('hs[1] = h1 ? test.key[3] : test.key[1];');
    expect(FOLD).toContain('hs[2] = h1 ? test.key[6] : test.key[4];');
    expect(FOLD).toContain('hs[3] = h1 ? test.key[7] : test.key[5];');
    expect(FOLD).toContain('hs[4] = h1 ? 0x3C6EF372u : 0x6A09E667u;');
    expect(FOLD).toContain('hs[5] = h1 ? 0xA54FF53Au : 0xBB67AE85u;');
    expect(FOLD).toContain('hs[6] = h1 ? 64u : 0u;');
    expect(FOLD).toContain('hs[7] = h1 ? (uint32_t)PEARL_TRANSCRIPT_FLAGS : 0u;');
    expect(FOLD).toContain('hs_base = buf * 4u + (hs_L >> 2) * 256u;');
    expect(FOLD).toContain('hs_w = test.hash_big_endian ? hs[0] ^ hs[4] : hs[3] ^ hs[7];');
    expect(FOLD).toContain('const uint32_t t = ((lane & 1u) ? e.y : e.x) ^ hs_k;');
    for (let i = 0; i < 4; i++) {
      expect(FOLD).toContain('const uint32_t m' + i
        + ' = *reinterpret_cast<const uint32_t *>(tb + __byte_perm(t, hs_base, 0x765' + i + 'u));');
    }
    expect(FOLD).toContain('pearl_bd_g(hs[0], hs[2], hs[4], hs[6], m0, m1);');
    expect(FOLD).toContain('pearl_bd_g(hs[1], hs[3], hs[5], hs[7], m2, m3);');
    expect(FOLD).toContain('const uint32_t a = hs[0];' + NL + '    hs[0] = hs[1];' + NL
      + '    hs[1] = a;' + NL + '    hs[2] = __shfl_xor_sync(0xffffffffu, hs[2], 1);' + NL
      + '    const uint32_t c = __shfl_xor_sync(0xffffffffu, hs[5], 1);' + NL
      + '    hs[5] = __shfl_xor_sync(0xffffffffu, hs[4], 1);' + NL + '    hs[4] = c;' + NL
      + '    hs[7] = __shfl_xor_sync(0xffffffffu, hs[7], 1);');
    // Fifteen steps a tile, one a chunk; fifteen again for the last tile.
    expect(FOLD).toContain('      hs_step(chunk);');
    expect(FOLD).toContain('for (uint32_t q = 0; q < chunks - 1u; q++) hs_step(q);');
    // The test: lane 0 of the pair for out16[0], lane 1 for out16[7], then hash_tile's
    // own path with the same words.
    expect(FOLD).toContain(
      'const bool mine = test.hash_big_endian ? (lane & 1u) == 0u : (lane & 1u) != 0u;');
    expect(FOLD).toContain(
      'const uint32_t msw = test.hash_big_endian ? pearl_bswap32(hs_w) : hs_w;');
    expect(FOLD).toContain('const uint32_t *src = sT + hs_base / 4u + (hs_L & 3u) * 16u;');
    expect(FOLD).toContain('tm[c] = src[c ^ ((hs_L >> 1) & 15u)];');
    expect(FOLD).toContain('sT[tbuf + L * 16u + (c ^ ((L >> 1) & 15u))] = x;');
    // G is BLAKE3's, and the constants are the kernel's.
    expect(FOLD).toContain('  a = a + b + mx;' + NL + '  d = rotr32(d ^ a, 16);' + NL
      + '  c = c + d;' + NL + '  b = rotr32(b ^ c, 12);' + NL + '  a = a + b + my;' + NL
      + '  d = rotr32(d ^ a, 8);' + NL + '  c = c + d;' + NL + '  b = rotr32(b ^ c, 7);');
    expect(KERNEL).toContain('__constant__ uint32_t BLAKE3_IV[8] = {0x6A09E667u, 0xBB67AE85u, '
      + '0x3C6EF372u,' + NL + '                                      0xA54FF53Au,');
    expect(KERNEL).toContain('#define PEARL_TRANSCRIPT_FLAGS (CHUNK_START | CHUNK_END | ROOT | KEYED_HASH)');
    expect(IV.slice(0, 4)).toEqual([0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A]);
  });
});
