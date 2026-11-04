'use strict';

// PEARL_AMD_KPAIR (earn/native/amd/src/pearl_amd_fold.h): on RDNA 4 each lane reads the fragments of two k-steps
// with one 16-byte LDS load, which takes a slab's bytes in another order. That is right only if the two k-steps
// together still take every byte of the slab once, and A' and B' take them in the same order, so every product
// of the slab is added once. No HIP compiler runs here, so the C++ is read as text and its arithmetic run as JS.

const fs = require('fs');
const path = require('path');
const { preprocess } = require('./helpers/amdSwitches');

const FOLD = fs.readFileSync(path.join(__dirname, '..', 'native', 'amd', 'src', 'pearl_amd_fold.h'), 'utf8')
  .replace(/\r\n/g, '\n');

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

const release = preprocess(FOLD, { PEARL_AMD_ABLATE: 0, PEARL_AMD_STAMP: 0 });
const step = slice(release, 'auto slab_step = [&](uint32_t s, auto parity) {', '\n  };\n');
const pair = slice(step, '#if PEARL_AMD_KPAIR\n', '\n#endif\n');
const KS = Number(grab(FOLD, /#define PEARL_AMD_SLAB (\d+)u/));
const pol = slice(FOLD, 'struct PearlAmdWmma12 {', '\n};\n');
const kTK = Number(grab(pol, /kTK = (\d+);/));

describe('paired fragment reads on RDNA 4 (PEARL_AMD_KPAIR)', () => {
  it('is on by default in RDNA 4 code only, and can be turned off', () => {
    expect(FOLD).toContain('#ifndef PEARL_AMD_KPAIR\n#define PEARL_AMD_KPAIR PEARL_AMD_WMMA12\n#endif');
    expect(pair).toContain('if constexpr (std::is_same<P, PearlAmdWmma12>::value) {');
    expect(pair).toContain('static_assert((KS / P::kTK) % 2 == 0, "PEARL_AMD_KPAIR pairs the k-steps of a slab");');
    // The slab's k-steps are run in pairs, at the even ones.
    expect(pair).toContain('if ((kk & 1) == 0) {');
    expect(KS / kTK).toBe(2);
  });

  it('reads A\' and B\' at the same place in their rows, 16 bytes a lane', () => {
    const off = grab(pair, /const int off = (.*);/);
    expect(pair).toContain('*reinterpret_cast<const pearl_amd_v4i *>(aBase + mt * P::kTM * S + off);');
    expect(pair).toContain('*reinterpret_cast<const pearl_amd_v4i *>(bBase + nt * P::kTN * S + off);');
    // Each load is 16-byte aligned: rows are S = KS + 16 bytes apart, and the lane's place is 0 or 16.
    const S = KS + 16;
    const at = new Function('lane', 'S', 'kk', 'P', 'return ' + off.replace(/P::kTK/g, 'P.kTK') + ';');
    for (let lane = 0; lane < 32; lane++) expect(at(lane, S, 0, { kTK }) % 16).toBe(0);
  });

  it('takes every byte of the slab once over its two k-steps, in A\' and B\' alike', () => {
    // Words 0 and 1 of the 16 bytes are the even k-step's fragment, 2 and 3 the odd one's.
    expect(pair).toMatch(/a\[0\]\[mt\] = pearl_amd_v2i\{q\[0\], q\[1\]\};\n\s+a\[1\]\[mt\] = pearl_amd_v2i\{q\[2\], q\[3\]\};/);
    expect(pair).toMatch(/b\[0\]\[nt\] = pearl_amd_v2i\{q\[0\], q\[1\]\};\n\s+b\[1\]\[nt\] = pearl_amd_v2i\{q\[2\], q\[3\]\};/);
    expect(pair).toContain('pearl_amd_mma_step<P>(acc, a[0], b[0]);\n          pearl_amd_mma_step<P>(acc, a[1], b[1]);');
    // The instruction's own k index of a fragment byte: lane l holds k 8 * (l / 16) .. + 7 of its k-step, by
    // load_a's layout (PearlAmdWmma12, which a_elem reads the same way).
    expect(pol).toContain('8 * (lane >> 4)');
    const off = grab(pair, /const int off = (.*);/);
    const S = KS + 16;
    const at = new Function('lane', 'S', 'kk', 'P', 'return ' + off.replace(/P::kTK/g, 'P.kTK') + ';');
    for (const row of [0, 5, 15]) {
      // For each k-step of the pair and each instruction k, the slab byte of row `row` it takes.
      const taken = [];
      for (let half = 0; half < 2; half++) {
        const lane = 16 * half + row;
        const base = at(lane, S, 0, { kTK }) - row * S;   // the byte within the row's slab
        for (let ks = 0; ks < 2; ks++) {
          for (let b = 0; b < 8; b++) taken.push({ ks, k: 8 * half + b, byte: base + 8 * ks + b });
        }
      }
      // Every slab byte once, and each k-step a full 16 instruction k.
      expect(taken.map((t) => t.byte).sort((x, y) => x - y)).toEqual([...Array(KS).keys()]);
      for (let ks = 0; ks < 2; ks++) {
        expect(taken.filter((t) => t.ks === ks).map((t) => t.k).sort((x, y) => x - y)).toEqual([...Array(kTK).keys()]);
      }
    }
  });

  it('leaves the other folds\' k-steps as they were', () => {
    const rest = slice(step, '      } else\n#endif\n      {\n', '\n      }\n');
    expect(rest).toContain('a[mt] = P::load_a(aBase + mt * P::kTM * S + kk * P::kTK, S, lane);');
    expect(rest).toContain('b[nt] = P::load_b(bBase + nt * P::kTN * S + kk * P::kTK, S, lane);');
    expect(rest).toContain('pearl_amd_mma_step<P>(acc, a, b);');
  });
});
