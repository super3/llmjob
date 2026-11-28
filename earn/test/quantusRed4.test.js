/**
 * @jest-environment node
 */
// The RTX 50 kernel's reduction, QC_RED 4 (native/quantus/cuda/qc_field.cuh): a 128-bit product folded with four
// adds and no multiply. With f = 2^32, p = f^2 - f + 1, so f^2 = f - 1 and f^3 = -1 (mod p), and the product
// r0 + r1 f + r2 f^2 + r3 f^3 is (r0 - r2 - r3) + (r1 + r2) f. ptxas compiles the PTX to exactly the four
// instructions modelled here; in the third, PTX's subc reads the carry with the wrong sign, and sm120/fuse.py
// inverts that carry-in in the cubin (bit 90, as sm86/carryfix.py does for QC_RED 3). This file pins the PTX the
// instructions come from and shows, word for word, that the patched tail is the product mod p for every input
// except when its last add goes below zero, which the test bounds.
const fs = require('fs');
const path = require('path');

const P = (1n << 64n) - (1n << 32n) + 1n;
const M32 = (1n << 32n) - 1n;
const F = 1n << 32n;
const cuda = path.join(__dirname, '../native/quantus/cuda');
const read = (...p) => fs.readFileSync(path.join(...p), 'utf8').replace(/\r\n/g, '\n');
const field = read(cuda, 'qc_field.cuh');
const kernel = read(cuda, 'qc_kernel.cuh');
const fuse = read(cuda, 'sm120', 'fuse.py');
const build = read(cuda, 'sm86', 'build.py');

// The four SASS instructions. inv says whether the third one's carry-in is inverted (the patched cubin).
//   1: IADD3 l, Pc, r0, -r2, RZ
//   2: IADD3.X h, Pc, r1, r2, RZ, Pc, !PT
//   3: IADD3.X lo, Pb, ~r3, l, RZ, [!]Pc, !PT
//   4: IADD3.X hi, RZ, -0x2, h, Pb, Pc
function tail(r0, r1, r2, r3, inv) {
  const s1 = r0 + (F - r2);
  const l = s1 & M32;
  const nb = s1 >> 32n;
  const s2 = r1 + r2 + nb;
  const h = s2 & M32;
  const c = s2 >> 32n;
  const s3 = l + (~r3 & M32) + (inv ? 1n - c : c);
  const lo = s3 & M32;
  const pb = s3 >> 32n;
  const t = h - 2n + pb + c;
  return { value: ((t & M32) << 32n) | lo, slip: t < 0n || t > M32, carries: [nb, c, pb] };
}
const exact = (r0, r1, r2, r3) => (((r0 + r1 * F + r2 * F * F + r3 * F * F * F) % P) + P) % P;

let seed = 0x9e3779b97f4a7c15n;
function rnd32() {
  seed ^= (seed << 13n) & ((1n << 64n) - 1n);
  seed ^= seed >> 7n;
  seed ^= (seed << 17n) & ((1n << 64n) - 1n);
  return seed >> 32n;
}
const edges = [0n, 1n, 2n, 3n, (1n << 31n) - 1n, 1n << 31n, M32 - 2n, M32 - 1n, M32];

describe('QC_RED 4: the reduction as four adds', () => {
  test('qc_reduce is written as the four adds', () => {
    const m = /#elif QC_RED == 4([\s\S]*?)\n#elif QC_RED == 3\b/.exec(field);
    expect(m).not.toBeNull();
    const ptx = [...m[1].matchAll(/"([^"]*)\\n\\t"/g)].map((x) => x[1]).filter((l) => /^(add|sub)/.test(l));
    expect(ptx).toEqual([
      'sub.cc.u32 l, %2, %4;',
      'addc.cc.u32 h, %3, %4;',
      'addc.u32 h2, h, 0;',
      'subc.cc.u32 %0, l, %5;',
      'subc.u32 %1, h2, 1;',
    ]);
  });

  test('the diagonal multiply folds the same way, with cy or nothing as r3', () => {
    const m = /#if QC_RED == 4\n([\s\S]*?)\n#else\n/.exec(kernel);
    expect(m).not.toBeNull();
    expect(m[1]).toContain('"sub.cc.u32 x0, x0, e;\\n\\t"');
    expect(m[1]).toContain('"addc.cc.u32 z1, z1, e;\\n\\t"');
    expect(m[1]).toContain('"addc.u32 c, z1, 0;\\n\\t"');
    expect(m[1]).toContain('#define QC_DM_T_CY "subc.cc.u32 %0, x0, cy;\\n\\t" "subc.u32 %1, c, 1;}"');
    expect(m[1]).toContain('#define QC_DM_T "subc.cc.u32 %0, x0, 0;\\n\\t" "subc.u32 %1, c, 1;}"');
  });

  test('the patched tail is the product mod p on every edge value', () => {
    let slips = 0;
    for (const r0 of edges) {
      for (const r1 of edges) {
        for (const r2 of edges) {
          for (const r3 of edges) {
            const t = tail(r0, r1, r2, r3, true);
            if (t.slip) {
              slips += 1;
              // the only slip: the last add below zero, which needs a high digit of 0 or 1 before its carries
              expect((r1 + r2 + t.carries[0]) & M32).toBeLessThan(2n);
            } else {
              expect(t.value % P).toBe(exact(r0, r1, r2, r3));
            }
          }
        }
      }
    }
    expect(slips).toBeGreaterThan(0);
  });

  test('and on random words, with no slip in 200,000 of them', () => {
    for (let i = 0; i < 200000; i += 1) {
      const [r0, r1, r2, r3] = [rnd32(), rnd32(), rnd32(), rnd32()];
      const t = tail(r0, r1, r2, i % 3 === 0 ? 0n : r3, true);
      expect(t.slip).toBe(false);
      expect(t.value % P).toBe(exact(r0, r1, r2, i % 3 === 0 ? 0n : r3));
    }
  });

  test('as compiled (carry-in not inverted) it is wrong, which is why the cubin is patched', () => {
    const t = tail(5n, 7n, 3n, 2n, false);
    expect(t.value % P).not.toBe(exact(5n, 7n, 3n, 2n));
    expect(tail(5n, 7n, 3n, 2n, true).value % P).toBe(exact(5n, 7n, 3n, 2n));
  });

  test('a slip needs the high digit to be 0 or 1: about 3 in 2^32 reductions', () => {
    // h + pb + c < 2 with h uniform: h = 0 (pb + c <= 1) or h = 1 (pb = c = 0). At most 2 of 2^32 values of h.
    for (const r1 of [0n, 1n, 2n, 3n]) {
      for (const r2 of [0n, 1n, 2n]) {
        const t = tail(9n, r1, r2, 4n, true);
        if (t.slip) expect(r1 + r2).toBeLessThan(3n);
      }
    }
  });
});

describe('QC_RED 4: the build', () => {
  test('build.py builds sm_120 with QC_RED 4 and runs fuse.py on it', () => {
    expect(/FLAGS_SM120 = \[(.*)\]/.exec(build)[1]).toBe('"-cubin", "-O3", "-std=c++17", "-arch=sm_120", "-DQC_RED=4"');
    expect(build).toContain('fusing = (a.arch == "sm_120" and "-DQC_FUSE=0" not in a.defs)');
    expect(build).toContain('fuse.apply(a.nvdisasm, out + ".unverified", out + ".unverified", kernel)');
  });

  test('fuse.py inverts one carry-in per tail (bit 90) and stops on a tail it cannot match', () => {
    expect(fuse).toContain('hi=x["hi"] ^ (1 << (90 - 64))');
    expect(fuse).toContain("looks like a QC_RED 4 tail's last, but its tail was not matched");
    expect(fuse).toContain('def find_red4(ins):');
  });

  test('fuse.py reads every changed instruction back through nvdisasm', () => {
    expect(fuse).toContain('sys.exit("fuse: read back at %x: %r, meant %r"');
  });
});
