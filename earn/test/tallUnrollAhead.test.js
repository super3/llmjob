'use strict';

const fs = require('fs');
const path = require('path');
const { preprocess: resolve } = require('./helpers/amdSwitches');

// PEARL_TALL_CHUNK_UNROLL and PEARL_TALL_A_AHEAD (pearl_config.h, off by default): unrolling and
// prefetch distance on the cp.async tall fold (sm_86 and sm_89). Two A/B switches for a race;
// nothing ships them.
//
// Nothing here runs device code. These check:
// - off by default, the values allowed, and on only in the sm_86 and sm_89 device passes;
// - off, the fold is the shipped code: no pragma on the chunk loop, the same stage body;
// - A_AHEAD on: the same mma from the same fragments, one ldmatrix moved a tile early at each
//   copy point (modelled below for every copy-point pair the fold allows);
// - no build that ships passes either switch.

const SRC = path.join(__dirname, '..', 'native', 'src');
const NL = String.fromCharCode(10);
const rawRead = (file) => fs.readFileSync(file, 'utf8').split(String.fromCharCode(13) + NL).join(NL);
const read = (name) => rawRead(path.join(SRC, name));
const HEADER = read('pearl_config.h');
const FOLD = read('pearl_fold_tall.cuh');
const ROOT = path.join(__dirname, '..', '..');
const UNROLL = 'PEARL_TALL_CHUNK_UNROLL';
const AHEAD = 'PEARL_TALL_A_AHEAD';
const OFF = { PEARL_TALL_CHUNK_UNROLL_ON: 1, PEARL_TALL_A_AHEAD_ON: 0 };

// Source without comment lines or trailing comments.
function code(text) {
  return text.split(NL).filter((l) => !l.trim().startsWith('//')).map((l) => {
    const at = l.indexOf('//');
    return at > 0 && l.slice(0, at).trim() ? l.slice(0, at).trimEnd() : l;
  }).join(NL);
}
const lines = (...l) => l.join(NL);
const count = (text, s) => text.split(s).length - 1;

describe('tall fold unroll and A-ahead switches', () => {
  test('off by default, 1/2/4 and 0/1 only, on only for sm_86 and sm_89', () => {
    expect(HEADER).toContain(lines('#ifndef PEARL_TALL_CHUNK_UNROLL', '#define PEARL_TALL_CHUNK_UNROLL 1', '#endif'));
    expect(HEADER).toContain(lines('#ifndef PEARL_TALL_A_AHEAD', '#define PEARL_TALL_A_AHEAD 0', '#endif'));
    expect(HEADER).toContain(lines(
      '#if PEARL_TALL_CHUNK_UNROLL != 1 && PEARL_TALL_CHUNK_UNROLL != 2 && PEARL_TALL_CHUNK_UNROLL != 4',
      '#error "PEARL_TALL_CHUNK_UNROLL is 1 (off), 2 or 4"',
      '#endif',
      '#if PEARL_TALL_A_AHEAD != 0 && PEARL_TALL_A_AHEAD != 1',
      '#error "PEARL_TALL_A_AHEAD is 0 or 1"',
      '#endif',
      '#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 890)',
      '#define PEARL_TALL_CHUNK_UNROLL_ON PEARL_TALL_CHUNK_UNROLL',
      '#define PEARL_TALL_A_AHEAD_ON PEARL_TALL_A_AHEAD',
      '#else',
      '#define PEARL_TALL_CHUNK_UNROLL_ON 1',
      '#define PEARL_TALL_A_AHEAD_ON 0',
      '#endif'));
    // The fold reads only the gated names, so the TMA build (sm_120, sm_100) and GA100 never
    // see either switch.
    const F = code(FOLD);
    expect(F).not.toMatch(new RegExp('\\b' + UNROLL + '\\b'));
    expect(F).not.toMatch(new RegExp('\\b' + AHEAD + '\\b'));
  });

  test('off, the chunk loop has no pragma and the stage body is the shipped one', () => {
    const off = code(resolve(FOLD, OFF));
    expect(off).not.toMatch(/#pragma unroll [24]/);
    expect(off).not.toMatch(/\ban[0-3]\b|\bptc\b|\bnmb\b/);
    expect(off).toContain(lines(
      '        for (uint32_t i = 0; i < 4; i++) acc[mb][nb][i] = 0;',
      '',
      '    for (uint32_t chunk = 0; chunk < chunks; chunk++) {'));
    expect(off).toContain(lines(
      '          for (uint32_t mb = 0; mb < MT; mb++) {',
      '            uint32_t a0, a1, a2, a3;',
      '            pearl_ldmatrix_x4(a0, a1, a2, a3, ((aLane0 + so) ^ kt) + mb * 16u * SK);',
      '#pragma unroll',
      '            for (uint32_t nb = 0; nb < NB; nb++)'));
  });

  test('unroll on puts its pragma on the chunk loop alone', () => {
    for (const n of [2, 4]) {
      const on = code(resolve(FOLD, { ...OFF, PEARL_TALL_CHUNK_UNROLL_ON: n }));
      expect([n, count(on, '#pragma unroll ' + n)]).toEqual([n, 1]);
      expect(on).toContain(lines('#pragma unroll ' + n, '    for (uint32_t chunk = 0; chunk < chunks; chunk++) {'));
      expect(on.replace('#pragma unroll ' + n + NL, '')).toBe(code(resolve(FOLD, OFF)));
    }
  });

  test('A-ahead on: the same mma, one ldmatrix a copy point moved a tile early', () => {
    const off = code(resolve(FOLD, OFF));
    const on = code(resolve(FOLD, { ...OFF, PEARL_TALL_A_AHEAD_ON: 1 }));
    expect(count(on, 'pearl_mma_m16n8k32(')).toBe(count(off, 'pearl_mma_m16n8k32('));
    // Two ldmatrix sites for A in the cp.async body instead of one: the tile's own, and the
    // next tile's at a copy point.
    expect(count(on, 'pearl_ldmatrix_x4(') - count(off, 'pearl_ldmatrix_x4(')).toBe(1);
    expect(on).toContain(lines(
      '              const uint32_t ptc = t * MT + mb;',
      '              if (ptc == PEARL_TALL_APT + 1u || ptc == PEARL_TALL_BPT + 1u) {',
      '                a0 = an0;',
      '                a1 = an1;',
      '                a2 = an2;',
      '                a3 = an3;',
      '              } else {',
      '                pearl_ldmatrix_x4(a0, a1, a2, a3, ((aLane0 + so) ^ kt) + mb * 16u * SK);',
      '              }'));
    expect(on).toContain(lines(
      '              if (ptc == PEARL_TALL_APT || ptc == PEARL_TALL_BPT) {',
      '                const uint32_t nt = (mb + 1u == MT) ? t + 1u : t;',
      '                const uint32_t nmb = (mb + 1u == MT) ? 0u : mb + 1u;',
      '                pearl_ldmatrix_x4(an0, an1, an2, an3,',
      '                                  ((aLane0 + so) ^ (nt * 32u)) + nmb * 16u * SK);',
      '              }'));
    // Both sit ahead of the tile's mma, and the copy points after them, as before.
    const at = on.indexOf('const uint32_t ptc = t * MT + mb;');
    expect(at).toBeGreaterThan(0);
    expect(on.indexOf('pearl_mma_m16n8k32(acc[mb][nb][0]', at)).toBeGreaterThan(on.indexOf('nmb * 16u * SK);', at));
    expect(on).toContain('static_assert(PEARL_TALL_BPT + 1u < KS * MT, "the tile after B\'s copy point is in the stage");');
  });

  test('A-ahead reads every tile\'s own fragment, for every copy-point pair it allows', () => {
    // The stage body's index logic, modelled: a fragment is named by its k-step and m16 tile.
    const KS = 2;
    const MT = 6;
    for (let apt = 0; apt < KS * MT; apt++) {
      for (let bpt = apt + 1; bpt + 1 < KS * MT; bpt++) {
        let an = null;
        for (let t = 0; t < KS; t++) {
          for (let mb = 0; mb < MT; mb++) {
            const ptc = t * MT + mb;
            const used = (ptc === apt + 1 || ptc === bpt + 1) ? an : [t, mb];
            if (ptc === apt || ptc === bpt) an = [mb + 1 === MT ? t + 1 : t, mb + 1 === MT ? 0 : mb + 1];
            expect([apt, bpt, t, mb, used]).toEqual([apt, bpt, t, mb, [t, mb]]);
          }
        }
      }
    }
  });

  test('no build that ships passes either switch', () => {
    for (const file of ['.github/workflows/miner-build.yml', 'earn/native/build-local.ps1', 'earn/native/binding.gyp']) {
      const text = rawRead(path.join(ROOT, file));
      expect(text.length).toBeGreaterThan(0);
      expect([file, text.includes(UNROLL) || text.includes(AHEAD)]).toEqual([file, false]);
    }
    // native-core.yml names them in its compile-only G7 step alone, after the shipped compile.
    const ci = rawRead(path.join(ROOT, '.github/workflows/native-core.yml'));
    const step = ci.indexOf('- name: Compile the unroll and A-ahead builds of the tall fold');
    const shipped = ci.indexOf('- name: Compile the CUDA kernels');
    expect(shipped).toBeGreaterThan(0);
    expect(step).toBeGreaterThan(shipped);
    const next = ci.indexOf('- name:', step + 10);
    expect(ci.slice(0, step)).not.toMatch(/PEARL_TALL_CHUNK_UNROLL=|PEARL_TALL_A_AHEAD=/);
    expect(ci.slice(next)).not.toMatch(/PEARL_TALL_CHUNK_UNROLL=|PEARL_TALL_A_AHEAD=/);
    expect(ci.slice(step, next)).toContain('continue-on-error: true');
  });
});
