'use strict';

// PEARL_AMD_RDLATE and PEARL_AMD_RDTREE (earn/native/amd/src/pearl_amd_fold.h, issue #287 item 4): where and in what
// order the RDNA 4 fold runs a chunk's readout. Both are off by default. RDLATE runs chunk c's readout in the next
// slab, after that slab's fragment loads; that is right only if every chunk is read out once, after its last slab's
// WMMAs and before the next slab's change the sums, with the keeps in chunk order. RDTREE XORs a region's 32 sums as
// three chains; that is right only if the chains split the 32 sums between them. No HIP compiler runs here, so the
// C++ is read as text and its conditions run as JS. The fold test checks both bit for bit, on a GPU and under HIP-CPU.

const fs = require('fs');
const path = require('path');
const { preprocess } = require('./helpers/amdSwitches');

const NATIVE = path.join(__dirname, '..', 'native');
const read = (...f) => fs.readFileSync(path.join(NATIVE, ...f), 'utf8').replace(/\r\n/g, '\n');
const FOLD = read('amd', 'src', 'pearl_amd_fold.h');
const CONFIG = read('src', 'pearl_config.h');
const BUILD = read('amd', 'build-amd.sh');

function slice(text, from, to) {
  const at = text.indexOf(from);
  if (at < 0) throw new Error('not found: ' + from);
  return text.slice(at, text.indexOf(to, at + from.length) + to.length);
}
function grab(text, re) {
  const m = re.exec(text);
  if (!m) throw new Error('not found: ' + re);
  return m[1];
}
// A C condition or expression of these sources as JS: no u suffixes or casts.
const toJs = (c) => c.replace(/\b(\d+)u\b/g, '$1').replace(/\(uint32_t\)/g, '');

const KS = Number(grab(FOLD, /#define PEARL_AMD_SLAB (\d+)u/));
const RANK = Number(grab(CONFIG, /#define PEARL_FOLD_RANK (\d+)u/));
const late = slice(FOLD, '  auto readout_late = [&](uint32_t s, auto parity) {', '\n  };\n');

describe('readout scheduling on RDNA 4 (PEARL_AMD_RDLATE, PEARL_AMD_RDTREE)', () => {
  it('is off unless a build sets it, and takes 0, 1 or 2 (RDTREE 0 or 1)', () => {
    expect(FOLD).toContain('#ifndef PEARL_AMD_RDLATE\n#define PEARL_AMD_RDLATE 0\n#endif\n'
      + '#if PEARL_AMD_RDLATE != 0 && PEARL_AMD_RDLATE != 1 && PEARL_AMD_RDLATE != 2\n#error');
    expect(FOLD).toContain('#ifndef PEARL_AMD_RDTREE\n#define PEARL_AMD_RDTREE 0\n#endif\n'
      + '#if PEARL_AMD_RDTREE != 0 && PEARL_AMD_RDTREE != 1\n#error');
    // Nothing that builds a release sets them.
    expect(BUILD).not.toMatch(/RDLATE|RDTREE/);
    // The RDNA 4 fold only, with the set's slab loop: a constant chunk of 4 slabs and the paired fragment reads.
    expect(FOLD).toContain(
      'constexpr bool kLate = PEARL_AMD_RDLATE != 0 && PEARL_AMD_KWRAP && std::is_same<P, PearlAmdWmma12>::value;');
    expect(FOLD).toContain('(PEARL_AMD_KWRAP && PEARL_AMD_KPAIR && PEARL_AMD_CRANK && PEARL_AMD_PF2 && !PEARL_AMD_UNROLL4)');
    expect(RANK / KS).toBe(4);
  });

  it('leaves the release\'s readout where it was when RDLATE is off', () => {
    const off = preprocess(FOLD, { PEARL_AMD_RDLATE: 0, PEARL_AMD_ABLATE: 0, PEARL_AMD_STAMP: 0 });
    expect(off).toContain('  auto readout_late = [](uint32_t, auto) {};   // PEARL_AMD_RDLATE off\n'
      + '  auto readout_hold = []() {};\n');
    expect(off).not.toContain('#pragma clang loop unroll(disable)');
    const step = slice(off, 'auto slab_step = [&](uint32_t s, auto parity) {', '\n  };\n');
    // The one readout in the slab: after the chunk's last WMMAs, behind kLate, which is false.
    expect(step).toContain('    if constexpr (!kLate)   // PEARL_AMD_RDLATE runs it in the next slab instead\n'
      + '#if PEARL_AMD_CRANK\n');
    expect(step.match(/readout\(/g)).toHaveLength(1);
    expect(step.match(/readout_late\(s, parity\)|readout_hold\(\)/g)).toEqual(['readout_late(s, parity)', 'readout_hold()']);
    expect(step).toContain('if constexpr (kLate) readout_late(s, parity);');
    expect(step).toContain('if constexpr (kLate) readout_hold();');
  });

  for (const mode of [1, 2]) {
    it(`with RDLATE=${mode}, reads out every chunk once, between its last slab's WMMAs and the next slab's`, () => {
      const on = preprocess(FOLD, { PEARL_AMD_RDLATE: mode });
      const body = slice(on, '  auto readout_late = [&](uint32_t s, auto parity) {', '\n  };\n');
      // Which slabs test for it, the test, and the chunk it reads out.
      // C's unsigned division is whole: s / perChunk as Math.floor in JS.
      const whole = (e) => e.replace(/(\w+) \/ perChunk/g, 'Math.floor($1 / perChunk)');
      const steps = toJs(grab(body, /if constexpr \((B == 0 \|\| PEARL_AMD_RDLATE == 2)\) \{/))
        .replace('PEARL_AMD_RDLATE', String(mode));
      const when = toJs(grab(body, /\n {6}if \((.*)\) \{\n/));
      const chunk = whole(toJs(grab(body, /\n {8}const uint32_t c = (.*);\n/)));
      const after = whole(toJs(grab(FOLD, /if constexpr \(kLate\) readout\((.*)\);   \/\/ PEARL_AMD_RDLATE: the last chunk/)));
      const f = (expr, ...names) => new Function(...names, 'return (' + expr + ');');
      const stepOn = f(steps, 'B'), test = f(when, 's', 'B', 'perChunk'), chunkOf = f(chunk, 's', 'perChunk');
      const lastOf = f(after, 'slabs', 'perChunk');
      // In mode 1 the readout runs whole at the slab; in mode 2 the sums at an even slab, the keep at an odd one.
      if (mode === 1) expect(body).toContain('        readout(c);\n');
      else expect(body).toContain('if constexpr (B == 0) {\n          readout_sum(xl);\n        } else {\n'
        + '          readout_rot(c);\n          readout_keep(c, xl);\n        }');
      const perChunk = RANK / KS;
      for (const k of [2048, 4096, 8192]) {
        const slabs = k / KS, chunks = slabs / perChunk;
        // Points in time: slab s's readout point is 2s (after its fragment loads, before its WMMAs), its WMMAs 2s + 1.
        const sumAt = new Array(chunks).fill(null), keepAt = new Array(chunks).fill(null), keeps = [];
        const at = (arr, c, t) => { expect([k, c, arr[c]]).toEqual([k, c, null]); arr[c] = t; };
        for (let s = 0; s < slabs; s++) {
          const B = s % 2;
          if (!stepOn(B) || !test(s, B, perChunk)) continue;
          const c = chunkOf(s, perChunk);
          if (mode === 1 || B === 0) at(sumAt, c, 2 * s);
          if (mode === 1 || B === 1) { at(keepAt, c, 2 * s); keeps.push(c); }
        }
        const c = lastOf(slabs, perChunk);
        at(sumAt, c, 2 * slabs);
        at(keepAt, c, 2 * slabs);
        keeps.push(c);
        for (let c = 0; c < chunks; c++) {
          // After the WMMAs of the chunk's last slab, 4c + 3, and no later than the next slab's readout point.
          expect([k, c, sumAt[c] > 2 * (perChunk * c + perChunk - 1) + 1]).toEqual([k, c, true]);
          expect([k, c, sumAt[c] <= 2 * perChunk * (c + 1)]).toEqual([k, c, true]);
          expect([k, c, keepAt[c] >= sumAt[c]]).toEqual([k, c, true]);
          // Mode 2 keeps the sums in one place (xl): the next chunk's sums come after this chunk's keep.
          if (mode === 2 && c + 1 < chunks) expect([k, c, sumAt[c + 1] > keepAt[c]]).toEqual([k, c, true]);
        }
        expect(keeps).toEqual([...Array(chunks).keys()]);
      }
    });
  }

  it('keeps each slab\'s WMMAs in the slab with an empty asm, and runs the readout after the fragment loads', () => {
    expect(late).toContain('__builtin_amdgcn_sched_barrier(0);   // the fragment loads stay above the readout');
    const hold = slice(FOLD, '  auto readout_hold = [&]() {', '\n  };\n');
    expect(hold).toContain('asm volatile("" ::"v"(acc[mt][nt]));');
    expect(hold).toContain('#if !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__)');
    // In the paired-read block, after the loads and before the WMMAs, and the hold right after the WMMAs.
    const pair = slice(FOLD, '        if ((kk & 1) == 0) {\n', '\n        }\n');
    const iLate = pair.indexOf('if constexpr (kLate) readout_late(s, parity);');
    const iHold = pair.indexOf('if constexpr (kLate) readout_hold();');
    expect(pair.lastIndexOf('b[1][nt] = pearl_amd_v2i{q[2], q[3]};')).toBeLessThan(iLate);
    expect(iLate).toBeLessThan(pair.indexOf('pearl_amd_mma_step<P>(acc, a[0], b[0]);'));
    expect(pair.indexOf('pearl_amd_mma_step<P>(acc, a[1], b[1]);')).toBeLessThan(iHold);
  });

  it('with RDTREE, XORs each region\'s 32 sums once, in three chains of 11, 11 and 10', () => {
    const tree = slice(FOLD, '#if PEARL_AMD_RDTREE\n', '#endif  // PEARL_AMD_RDTREE');
    const li = toJs(grab(tree, /const int li = (.*);/));
    const into = grab(tree, /\n\s+y\[P::ro_of\(g\)\]\[(.*)\] \^= \(uint32_t\)acc\[mt\]\[nt\]\[g\];/);
    expect(tree).toContain('asm("" : "+v"(y[q][0]), "+v"(y[q][1]), "+v"(y[q][2]));');
    expect(tree).toContain('x[rb][cb][q] = y[q][0] ^ y[q][1] ^ y[q][2];');
    // wmma12's tile: 16 x 16, 8 registers, and register g is row offset g >> 2.
    const pol = slice(FOLD, 'struct PearlAmdWmma12 {', '\n};\n');
    expect(pol).toContain('static constexpr int kWave = 32, kTM = 16, kTN = 16, kTK = 16;');
    expect(pol).toContain('static constexpr int kWR = 2, kWC = 4, kMT = 4, kNT = 4;');
    expect(pol).toContain('static __device__ __forceinline__ int ro_of(int g) { return g >> 2; }');
    const P = { kTM: 16, kTN: 16, kRegs: 8 };
    const liOf = new Function('mt', 'nt', 'g', 'rb', 'cb', 'P', 'return ' + li.replace(/P::/g, 'P.') + ';');
    const chainOf = new Function('li', 'return ' + into + ';');
    const RB = (4 * P.kTM) / 32, CB = (4 * P.kTN) / 64;
    for (let rb = 0; rb < RB; rb++) {
      for (let cb = 0; cb < CB; cb++) {
        for (let q = 0; q < 2; q++) {
          const chains = [[], [], []];
          for (let mt = rb * (32 / P.kTM); mt < (rb + 1) * (32 / P.kTM); mt++) {
            for (let nt = cb * (64 / P.kTN); nt < (cb + 1) * (64 / P.kTN); nt++) {
              for (let g = 0; g < P.kRegs; g++) {
                if (g >> 2 === q) chains[chainOf(liOf(mt, nt, g, rb, cb, P))].push(`${mt}.${nt}.${g}`);
              }
            }
          }
          expect(chains.map((c) => c.length)).toEqual([11, 11, 10]);
          expect(new Set(chains.flat()).size).toBe(32);
        }
      }
    }
  });
});
