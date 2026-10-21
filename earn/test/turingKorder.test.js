'use strict';

const fs = require('fs');
const path = require('path');

// Turing's k-lo-first fold (PEARL_BD_KORDER, on by default). The draw writes A'
// and B' in pairs of 16-row blocks: block 2p's slot holds both blocks' k 0-15
// words and block 2p + 1's their k 16-31 words. The fold reads each IMMA's A and
// B word from there instead of from one block's four words. Nothing here can run
// the device code, so this checks on the CPU, word for word, that every IMMA of
// a k-step gets the same A and B words as with the default layout, and that the
// paired draw writes every byte exactly once. The transcriptions are pinned to
// the source lines they copy.

const SRC = path.join(__dirname, '..', 'native', 'src');
const NL = String.fromCharCode(10);
const read = (name) => fs.readFileSync(path.join(SRC, name), 'utf8')
  .split(String.fromCharCode(13) + NL).join(NL);
const KERNEL = read('pearl_kernel.cu');
const FOLD = read('pearl_fold_bd.cuh');
const HEADER = read('pearl_config.h');

function body(src, signature) {
  const at = src.indexOf(signature);
  expect(at).toBeGreaterThanOrEqual(0);
  const end = src.indexOf(NL + '}', at);
  expect(end).toBeGreaterThan(at);
  return src.slice(at, end);
}

function bytes(n, seed) {
  const a = new Uint8Array(n);
  let x = seed >>> 0;
  for (let i = 0; i < n; i++) {
    x ^= x << 13; x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5; x >>>= 0;
    a[i] = x & 0xff;
  }
  return a;
}

function word(b, at) {
  return (b[at] | (b[at + 1] << 8) | (b[at + 2] << 16) | (b[at + 3] << 24)) >>> 0;
}

// The draw's store, thread by thread (pearl_materialize16_frag and _afrag): thread
// v holds the sixteen row-major bytes at v * 16 and stores word w at o[w * 4],
// o = out + blk * 512 + (r & 7) * 64 + reg * 4 bytes. isA picks the default word
// order (A: k before rows; B: rows before k); paired is PEARL_BD_KORDER's.
function draw(src, rows, kLog2, isA, paired) {
  const k = 2 ** kLog2;
  const out = new Uint8Array(rows * k);
  const written = new Uint8Array(rows * k);
  for (let v = 0; v < (rows * k) / 16; v++) {
    const idx = v * 16;
    const r = Math.floor(idx / k);
    const kk = idx & (k - 1);
    let blk;
    let reg;
    const bl = r >> 4;
    if (paired && (bl | 1) < (rows >> 4)) {
      const dbl = (bl & ~1) | ((kk >> 4) & 1);
      blk = dbl * 2 ** (kLog2 - 5) + (kk >> 5);
      reg = (bl & 1) * 2 + ((r >> 3) & 1);
    } else {
      blk = bl * 2 ** (kLog2 - 5) + (kk >> 5);
      reg = isA ? ((kk >> 4) & 1) * 2 + ((r >> 3) & 1) : ((r >> 3) & 1) * 2 + ((kk >> 4) & 1);
    }
    const o = blk * 512 + (r & 7) * 64 + reg * 4;
    for (let w = 0; w < 4; w++) {
      for (let j = 0; j < 4; j++) {
        out[o + w * 16 + j] = src[idx + 4 * w + j];
        written[o + w * 16 + j]++;
      }
    }
  }
  return { out, written };
}

// A lane's four words of 16-row block blk at k-step s (the fold's LDG.128).
function quad(frag, kLog2, blk, s, lane) {
  const at = blk * (2 ** (kLog2 - 5)) * 512 + s * 512 + lane * 16;
  return [word(frag, at), word(frag, at + 4), word(frag, at + 8), word(frag, at + 12)];
}

// The 128 IMMA of a k-step for a warp whose four A blocks start at aBase and
// four B blocks at bBase, as (accumulator, A word, B word), read the default way
// (pearl_mma_m16n8k32: c01 += a0 b0, a2 b1; c23 += a1 b0, a3 b1, with b2, b3 for
// the odd n-block) or the paired way (kstep's a_word and b_word).
function immas(A, B, kLog2, aBase, bBase, s, lane, paired) {
  const out = [];
  const aw = (mb, hi, u) => (paired
    ? quad(A, kLog2, aBase + Math.floor(mb / 2) * 2 + hi, s, lane)[(mb & 1) * 2 + u]
    : quad(A, kLog2, aBase + mb, s, lane)[hi * 2 + u]);
  const bw = (j, hi, odd) => (paired
    ? quad(B, kLog2, bBase + Math.floor(j / 2) * 2 + hi, s, lane)[(j & 1) * 2 + odd]
    : quad(B, kLog2, bBase + j, s, lane)[odd * 2 + hi]);
  for (let hi = 0; hi < 2; hi++) {
    for (let j = 0; j < 4; j++) {
      for (let odd = 0; odd < 2; odd++) {
        for (let mb = 0; mb < 4; mb++) {
          for (let u = 0; u < 2; u++) {
            out.push([`${mb}.${2 * j + odd}.${u}.${hi}`, aw(mb, hi, u), bw(j, hi, odd)]);
          }
        }
      }
    }
  }
  return out;
}

describe('Turing k-lo first (PEARL_BD_KORDER)', () => {
  test('is on by default with A-direct, as measured, and needs it', () => {
    const at = HEADER.indexOf('#ifndef PEARL_BD_KORDER');
    expect(at).toBeGreaterThanOrEqual(0);
    expect(HEADER.slice(at).split(NL)[1].trim()).toBe('#define PEARL_BD_KORDER PEARL_BD_A_DIRECT');
    expect(HEADER).toContain('#if PEARL_BD_KORDER && !PEARL_BD_A_DIRECT');
  });

  test('the transcriptions match the source', () => {
    for (const name of ['void pearl_materialize16_frag(', 'void pearl_materialize16_afrag(']) {
      const mat = body(KERNEL, name);
      expect(mat).toContain('const bool paired = (bl | 1u) < (rows >> 4);');
      expect(mat).toContain('const uint32_t dbl = paired ? (bl & ~1u) | ((kk >> 4) & 1u) : bl;');
      expect(mat).toContain('const uint64_t blk = ((uint64_t)dbl << (k_log2 - 5)) + (kk >> 5);');
      expect(mat).toContain('const uint32_t reg = paired ? (bl & 1u) * 2u + ((r >> 3) & 1u)');
      expect(mat).toContain(
        'uint32_t *o = reinterpret_cast<uint32_t *>(out + blk * 512u + (r & 7u) * 64u) + reg;');
    }
    expect(body(KERNEL, 'void pearl_materialize16_frag('))
      .toContain(': ((r >> 3) & 1u) * 2u + ((kk >> 4) & 1u);');
    expect(body(KERNEL, 'void pearl_materialize16_afrag('))
      .toContain(': ((kk >> 4) & 1u) * 2u + ((r >> 3) & 1u);');
    // The fold's reads: quad (mb / 2) * 2 + hi, word (mb & 1) * 2 + u.
    expect(FOLD).toContain('const int4 v = af[cur][(mb / 2u) * 2u + hi];');
    expect(FOLD).toContain('return (uint32_t)(mb & 1u ? (u ? v.w : v.z) : (u ? v.y : v.x));');
    expect(FOLD).toContain('const int4 v = bf[cur][(j / 2u) * 2u + hi];');
    expect(FOLD).toContain('return (uint32_t)(j & 1u ? (odd ? v.w : v.z) : (odd ? v.y : v.x));');
    // Each IMMA's accumulator, A word and B word, k-lo then k-hi.
    expect(FOLD).toContain('b0[rl] = b_word(j, 0u, 0u);');
    expect(FOLD).toContain('b2[rl] = b_word(j, 0u, 1u);');
    expect(FOLD).toContain(
      'pearl_bd_imma(acc[mb][2u * j][0], acc[mb][2u * j][1], a_word(mb, 0u, 0u), b0[mb / 2u]);');
    expect(FOLD).toContain(
      'pearl_bd_imma(acc[mb][2u * j][2], acc[mb][2u * j][3], a_word(mb, 0u, 1u), b0[mb / 2u]);');
    expect(FOLD).toContain('pearl_bd_imma(acc[mb][2u * j + 1u][0], acc[mb][2u * j + 1u][1], a_word(mb, 0u, 0u),'
      + NL + '                      b2[mb / 2u]);');
    expect(FOLD).toContain('pearl_bd_imma(acc[mb][2u * j + 1u][2], acc[mb][2u * j + 1u][3], a_word(mb, 0u, 1u),'
      + NL + '                      b2[mb / 2u]);');
    expect(FOLD).toContain('const uint32_t b1 = b_word(j, 1u, 0u), b3 = b_word(j, 1u, 1u);');
    expect(FOLD).toContain(
      'pearl_bd_imma(acc[mb][2u * j][0], acc[mb][2u * j][1], a_word(mb, 1u, 0u), b1);');
    expect(FOLD).toContain(
      'pearl_bd_imma(acc[mb][2u * j][2], acc[mb][2u * j][3], a_word(mb, 1u, 1u), b1);');
    expect(FOLD).toContain(
      'pearl_bd_imma(acc[mb][2u * j + 1u][0], acc[mb][2u * j + 1u][1], a_word(mb, 1u, 0u), b3);');
    expect(FOLD).toContain(
      'pearl_bd_imma(acc[mb][2u * j + 1u][2], acc[mb][2u * j + 1u][3], a_word(mb, 1u, 1u), b3);');
    // The default: four m8n8k16 a mma, a0 b0 and a2 b1 into c0 c1, a1 b0 and a3 b1
    // into c2 c3.
    expect(KERNEL).toContain('"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%4}, {%8}, {%0,%1};' + String.fromCharCode(92) + 'n"');
    expect(KERNEL).toContain('"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%6}, {%9}, {%0,%1};' + String.fromCharCode(92) + 'n"');
    expect(KERNEL).toContain('"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%2,%3}, {%5}, {%8}, {%2,%3};' + String.fromCharCode(92) + 'n"');
    expect(KERNEL).toContain('"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%2,%3}, {%7}, {%9}, {%2,%3};' + String.fromCharCode(92) + 'n"');
    // A warp's four blocks start on a pair: A's at (rbg * 8 + wr * 4), B's at
    // col_off + cbg * 16 + wc * 4, with col_off even.
    expect(FOLD).toContain('return (rbg_ * 8u + wr * 4u) * FRAG_BLOCK + lane * 16u;');
    expect(FOLD).toContain('return (col_off + cbg_ * 16u + wc * 4u) * FRAG_BLOCK + lane * 16u;');
    expect(FOLD).toContain('if (col_off & 1u) __trap();');
  });

  test('the paired draw writes every byte once, an unpaired last block too', () => {
    for (const rows of [64, 80, 272]) {
      for (const isA of [true, false]) {
        const { written } = draw(bytes(rows * 128, 7 + rows), rows, 7, isA, true);
        expect(written.every((n) => n === 1)).toBe(true);
      }
    }
  });

  test('every IMMA gets the same A and B words as with the default layout', () => {
    // 6 blocks of A (one warp's four, at 0 or at 2, the next pair), 10 of B; k 256.
    const kLog2 = 8;
    const k = 2 ** kLog2;
    const mRows = 6 * 16;
    const nRows = 10 * 16;
    const a = bytes(mRows * k, 11);
    const b = bytes(nRows * k, 13);
    const A0 = draw(a, mRows, kLog2, true, false).out;
    const A1 = draw(a, mRows, kLog2, true, true).out;
    const B0 = draw(b, nRows, kLog2, false, false).out;
    const B1 = draw(b, nRows, kLog2, false, true).out;
    const bad = [];
    let checked = 0;
    for (const aBase of [0, 2]) {
      for (const bBase of [0, 4, 6]) {
        for (let s = 0; s < k / 32; s++) {
          for (let lane = 0; lane < 32; lane++) {
            const want = immas(A0, B0, kLog2, aBase, bBase, s, lane, false);
            const got = immas(A1, B1, kLog2, aBase, bBase, s, lane, true);
            for (let i = 0; i < want.length; i++) {
              if ((want[i][1] !== got[i][1] || want[i][2] !== got[i][2]) && bad.length < 5) {
                bad.push(`A ${aBase} B ${bBase} k-step ${s} lane ${lane} IMMA ${want[i][0]}`);
              }
              checked++;
            }
          }
        }
      }
    }
    expect(bad).toEqual([]);
    expect(checked).toBe(2 * 3 * (k / 32) * 32 * 128);
  });
});
