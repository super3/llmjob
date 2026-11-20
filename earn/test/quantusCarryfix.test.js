'use strict';

// The patched RTX 30 kernel (native/quantus/cuda/sm86): QC_RED 3's reduction tail with one carry-in inverted in the
// cubin by carryfix.py. carryfix.py's header gives the argument; these tests pin it.
//
// The tail after E = r2 * EPS + (r1:r0) (mod 2^64, carry c) must give (hi:lo) = E + c * EPS - r3 (mod 2^64). The
// PTX it is compiled from (pinned below) becomes, in SASS,
//   lo = E0 + ~r3 + carryIn   (one IADD3.X; Pb its carry out)
//   hi = E1 - 1 + Pb + c      (the next IADD3.X)
// with carryIn = c as ptxas emits it (wrong) and !c once patched (right, for every E, r3 and c). The diagonal
// multiplies' tail (qc_dmul) is the same with cy, a 0/1 carry, as r3. QC_RED 2, the compiled default, gets the same
// result but for r3 = 2^32 - 1 with c = 1, where its r3 + c wraps.

const fs = require('fs');
const path = require('path');

const P = (1n << 64n) - (1n << 32n) + 1n;
const EPS = (1n << 32n) - 1n;
const M32 = (1n << 32n) - 1n;
const M64 = (1n << 64n) - 1n;
const cuda = path.join(__dirname, '../native/quantus/cuda');
const field = fs.readFileSync(path.join(cuda, 'qc_field.cuh'), 'utf8');
const kernel = fs.readFileSync(path.join(cuda, 'qc_kernel.cuh'), 'utf8');

// The two SASS instructions, word for word. inv says whether the first one's carry-in is inverted.
function tail(E, r3, c, inv) {
  const e0 = E & M32;
  const e1 = E >> 32n;
  const cin = inv ? 1n - c : c;
  const s = e0 + (~r3 & M32) + cin;
  const lo = s & M32;
  const pb = s >> 32n;
  const hi = (e1 - 1n + pb + c) & M32;
  return (hi << 32n) | lo;
}
// QC_RED 2's four instructions: k = r3 + c on 32 bits, lo = E0 - k with its borrow, hi = E1 + c - borrow.
function red2(E, r3, c) {
  const k = (r3 + c) & M32;
  const e0 = E & M32;
  const e1 = E >> 32n;
  const b = e0 < k ? 1n : 0n;
  const lo = (e0 - k) & M32;
  const hi = (e1 + c - b) & M32;
  return (hi << 32n) | lo;
}
const exact = (E, r3, c) => (E + c * EPS - r3) & M64;

// A small xorshift, so the run is the same every time.
let seed = 0x9e3779b97f4a7c15n;
function rnd64() {
  seed ^= (seed << 13n) & M64;
  seed ^= seed >> 7n;
  seed ^= (seed << 17n) & M64;
  return seed;
}

const edges32 = [0n, 1n, 2n, (1n << 31n) - 1n, 1n << 31n, M32 - 1n, M32];

describe('carryfix: the QC_RED 3 tail with its carry-in inverted', () => {
  test('the PTX that compiles to the tail is the one the argument is about', () => {
    const red3 = /#elif QC_RED == 3([\s\S]*?)#else/.exec(field);
    expect(red3).not.toBeNull();
    // The asm block's strings that are instructions (not braces, .reg lines or operand constraints).
    const ptx = (text) => text.match(/"[^"]*"/g).map((s) => s.slice(1, -1).replace(/\\n\\t$/, ''))
      .filter((s) => / /.test(s) && /;\}?$/.test(s) && !s.startsWith('.reg'));
    expect(ptx(red3[1])).toEqual([
      'mov.b64 L, {%2, %3};',
      'mul.wide.u32 E, %4, %6;',
      'add.cc.u64 E, E, L;',
      'mov.b64 {%0, %1}, E;',
      'addc.u32 hc, %1, 0;',
      'subc.cc.u32 %0, %0, %5;',
      'subc.u32 %1, hc, 0;',
    ]);
    // qc_dmul's tail with cy, under QC_RED 3.
    const dm = /#if QC_RED == 3[\s\S]*?#define QC_DM_T_CY (.*)/.exec(kernel);
    expect(dm).not.toBeNull();
    expect(ptx(dm[1])).toEqual([
      'addc.u32 c, %1, 0;',
      'subc.cc.u32 %0, %0, cy;',
      'subc.u32 %1, c, 0;}',
    ]);
    // ... and without cy (r3 = 0, which ptxas writes as the immediate -0x1 = ~0), on by default with QC_RED 3.
    const dm0 = /#if QC_RED == 3 && QC_DM_T0[\s\S]*?#define QC_DM_T (.*)/.exec(kernel);
    expect(dm0).not.toBeNull();
    expect(ptx(dm0[1])).toEqual([
      'addc.u32 c, %1, 0;',
      'subc.cc.u32 %0, %0, 0;',
      'subc.u32 %1, c, 0;}',
    ]);
    expect(kernel).toMatch(/#ifndef QC_DM_T0\r?\n#define QC_DM_T0 1\r?\n#endif/);
  });

  test('patched, the tail is exact for every edge value of E, r3 and c', () => {
    for (const e0 of edges32) {
      for (const e1 of edges32) {
        const E = (e1 << 32n) | e0;
        for (const r3 of edges32) {
          for (const c of [0n, 1n]) expect(tail(E, r3, c, true)).toBe(exact(E, r3, c));
        }
      }
    }
  });

  test('patched, the tail is exact on random inputs; as compiled it is not', () => {
    let wrong = 0;
    for (let i = 0; i < 20000; i++) {
      const E = rnd64();
      const r3 = rnd64() & M32;
      const c = rnd64() & 1n;
      expect(tail(E, r3, c, true)).toBe(exact(E, r3, c));
      if (tail(E, r3, c, false) !== exact(E, r3, c)) wrong++;
    }
    expect(wrong).toBeGreaterThan(9000);  // whenever c is 0 the compiled tail is 1 off
  });

  test('QC_RED 2 differs from the exact tail only where r3 + c wraps', () => {
    for (const e0 of edges32) {
      for (const e1 of edges32) {
        const E = (e1 << 32n) | e0;
        for (const r3 of edges32) {
          for (const c of [0n, 1n]) {
            if (r3 === M32 && c === 1n) expect(red2(E, r3, c)).not.toBe(exact(E, r3, c));
            else expect(red2(E, r3, c)).toBe(exact(E, r3, c));
          }
        }
      }
    }
  });

  test('the diagonal multiplies\' tail (cy as r3) is exact patched', () => {
    for (let i = 0; i < 5000; i++) {
      const E = rnd64();
      for (const cy of [0n, 1n]) {
        for (const c of [0n, 1n]) expect(tail(E, cy, c, true)).toBe(exact(E, cy, c));
      }
    }
  });

  test('a whole reduction with the patched tail is congruent to the product, bar the 64-bit wrap', () => {
    let wraps = 0;
    for (let i = 0; i < 20000; i++) {
      const a = rnd64();
      const b = rnd64();
      const x = a * b;
      const r0 = x & M32;
      const r1 = (x >> 32n) & M32;
      const r2 = (x >> 64n) & M32;
      const r3 = x >> 96n;
      const s = r2 * EPS + ((r1 << 32n) | r0);
      const E = s & M64;
      const c = s >> 64n;
      const z = tail(E, r3, c, true);
      const v = E + c * EPS - r3;  // the value before the 64-bit wrap
      if (v < 0n || v > M64) wraps++;
      else expect(z % P).toBe(x % P);
    }
    expect(wraps).toBeLessThan(5);  // about one in 2^32 reductions
  });
});

describe('carryfix: the build pins', () => {
  const sm86 = path.join(cuda, 'sm86');
  test('build.py builds sm_86 with QC_RED 3 and nothing else', () => {
    const py = fs.readFileSync(path.join(sm86, 'build.py'), 'utf8');
    expect(/FLAGS = \[(.*)\]/.exec(py)[1]).toBe('"-cubin", "-O3", "-std=c++17", "-arch=sm_86", "-DQC_RED=3"');
  });

  test('carryfix.py inverts bit 90, the first carry-in\'s negation', () => {
    const py = fs.readFileSync(path.join(sm86, 'carryfix.py'), 'utf8');
    expect(py).toMatch(/^NEG1 = 1 << \(90 - 64\)/m);
  });

  test('qc_sm86.ref lists toolkits and hashes, each verified on a card', () => {
    const lines = fs.readFileSync(path.join(sm86, 'qc_sm86.ref'), 'utf8').split('\n')
      .map((l) => l.split('#')[0].trim()).filter(Boolean);
    expect(lines.length).toBeGreaterThan(0);
    for (const l of lines) expect(l).toMatch(/^V\d+\.\d+\.\d+ [0-9a-f]{64} \S/);
  });
});

// The patched RTX 40 kernel: the same tails in an sm_89 build (sm86/build.py --arch sm_89), with QC_KZ 1, which makes
// the squares' carry by "madc.lo.u32 cm, kz, kz, 0" on a constant zero instead of "addc.u32 cm, 0, 0". kz * kz + 0 +
// carry is the carry when kz is 0, so the square is unchanged; it only moves that instruction to the multiplier pipe.
describe('carryfix: the RTX 40 (sm_89) build', () => {
  const sm86 = path.join(cuda, 'sm86');
  const py = fs.readFileSync(path.join(sm86, 'build.py'), 'utf8');

  test('build.py builds sm_89 with QC_RED 3 and QC_KZ 1, and sm_86 as before by default', () => {
    expect(/FLAGS_SM89 = \[(.*)\]/.exec(py)[1]).toBe('"-cubin", "-O3", "-std=c++17", "-arch=sm_89", "-DQC_RED=3", "-DQC_KZ=1"');
    expect(py).toContain('ARCHS = {"sm_86": (FLAGS, "RTX 30"), "sm_89": (FLAGS_SM89, "RTX 40")}');
    expect(py).toContain('ap.add_argument("--arch", default="sm_86", choices=sorted(ARCHS))');
  });

  test('QC_KZ 1 makes the squares\' carry a madc of a zero the compiler cannot see, and nothing else', () => {
    expect(field).toContain('__constant__ unsigned qc_kzero = 0;');
    expect(field).toContain('#define QC_CARRY_KZ(c, kz) "madc.lo.u32 " c ", " kz ", " kz ", 0;\\n\\t"');
    expect(field).toMatch(/#if QC_KZ & 1\r?\n#define QC_CARRY_SQ QC_CARRY_KZ\r?\n/);
    expect(field).toMatch(/#if QC_KZ & 2\r?\n#define QC_CARRY_DM QC_CARRY_KZ\r?\n/);
    // madc.lo: kz * kz + 0 + carry, with kz = 0, for both values of the carry.
    for (const c of [0n, 1n]) expect((0n * 0n + 0n + c) & M32).toBe(c);
  });

  test('the core loads the sm_89 image on compute 8.9 cards only, and the sm_86 one on 8.6 only', () => {
    const inc = fs.readFileSync(path.join(cuda, 'qc_cubin.inc'), 'utf8');
    expect(inc).toMatch(/#if defined\(QC_SM89_CUBIN\)\r?\n#include "qc_sm89_cubin.h"\r?\n#endif/);
    expect(inc).toMatch(/if \(major != 8\) return;/);
    expect(inc).toMatch(/if \(minor == 6\) \{\r?\n\s+img\.assign\(\(const char \*\)qc_sm86_image/);
    expect(inc).toMatch(/if \(minor == 9\) \{\r?\n\s+img\.assign\(\(const char \*\)qc_sm89_image/);
  });

  test('qc_sm89.ref lists toolkits and hashes, each verified on an RTX 40 card', () => {
    const lines = fs.readFileSync(path.join(sm86, 'qc_sm89.ref'), 'utf8').split('\n')
      .map((l) => l.split('#')[0].trim()).filter(Boolean);
    expect(lines.length).toBeGreaterThan(0);
    for (const l of lines) expect(l).toMatch(/^V\d+\.\d+\.\d+ [0-9a-f]{64} RTX 40\S*/);
  });
});
