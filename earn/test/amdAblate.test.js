'use strict';

// PEARL_AMD_ABLATE (earn/native/amd/src/pearl_amd_fold.h) builds an AMD core whose fold leaves one part
// out, for timing: bit 2 the slab barrier's s_barrier, 8 the LDS fragment loads, 16 the spread of the
// operands over memory, 32 the slab loop's staging (its global loads and LDS stores). Its transcripts are wrong on purpose. These pin the guards that keep it from mining or
// shipping, that a release build gets the same code, the fill a tuning build can now set, and the
// fold test's miner operands that sweep.sh times with. No HIP compiler runs here, so the C++ is read as
// text, its #if blocks resolved as the preprocessor would; the shell parts run under bash where there
// is one.

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const noise = require('../src/shared/miner/noise');
const { preprocess } = require('./helpers/amdSwitches');

const NATIVE = path.join(__dirname, '..', 'native');
const ROOT = path.join(NATIVE, '..', '..');
const AMD = path.join(NATIVE, 'amd');
const read = (...f) => fs.readFileSync(path.join(NATIVE, ...f), 'utf8').replace(/\r\n/g, '\n');
const FOLD = read('amd', 'src', 'pearl_amd_fold.h');
const HOST = read('amd', 'src', 'pearl_host_hip.cpp');
const BUILD = read('amd', 'build-amd.sh');
const SWEEP = read('amd', 'test', 'sweep.sh');
const FOLDTEST = read('amd', 'test', 'pearl_amd_foldtest.cpp');
const CONFIG = read('src', 'pearl_config.h');
const HASHRATE = read('probes', 'hashrate.js');

const hasBash = spawnSync('bash', ['--version'], { stdio: 'ignore' }).status === 0;
const bashIt = hasBash ? it : it.skip;
const bash = (script, env) => spawnSync('bash', ['-c', script],
  { env: Object.assign({ PATH: process.env.PATH }, env), encoding: 'utf8' });

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
// The text from the line starting with `first` to the next line that is exactly `last`.
function block(text, first, last) {
  const lines = text.split('\n');
  const a = lines.findIndex((l) => l.startsWith(first));
  const b = lines.findIndex((l, i) => i > a && l === last);
  if (a < 0 || b < 0) throw new Error('block not found: ' + first);
  return lines.slice(a, b + 1).join('\n') + '\n';
}

// The text as the preprocessor leaves it with PEARL_AMD_ABLATE = v, in a build without clock stamps
// (helpers/amdSwitches.js).
const ablated = (text, v) => preprocess(text, { PEARL_AMD_ABLATE: v, PEARL_AMD_STAMP: 0 });
const BITS = [2, 8, 16, 32, 10, 18, 24, 26, 34, 40, 42, 58];
const slabStep = (t) => slice(t, 'auto slab_step = [&](uint32_t s, auto parity) {', '\n  };\n');
const kSteps = (step) => slice(step, '    for (int kk = 0; kk < KS / P::kTK; kk++) {', '\n    }\n');
const oneSlab = (t) => slice(t, '  if (s < slabs) slab_step(s, std::integral_constant<int, 0>());\n#else\n', '\n#endif\n');
const caps = (t) => slice(t, 'extern "C" __global__ void pearl_amd_caps(uint32_t *out) {', '\n}\n');
const refFold = (t) => slice(t, 'void pearl_amd_fold_ref(', '\n}\n');

describe('the ablation knob (PEARL_AMD_ABLATE)', () => {
  it('is 0 unless a build sets it, and takes bits 2, 8, 16, 32 and 64 only', () => {
    expect(FOLD).toContain('#ifndef PEARL_AMD_ABLATE\n#define PEARL_AMD_ABLATE 0\n#endif\n#if PEARL_AMD_ABLATE & ~(2 | 8 | 16 | 32 | 64)\n#error');
    const cond = grab(FOLD, /^#if (PEARL_AMD_ABLATE & ~\(.*\))\n#error "PEARL_AMD_ABLATE takes bits 2, 8, 16, 32 and 64 only"/m);
    const refused = (v) => new Function('PEARL_AMD_ABLATE', 'return (' + cond + ') !== 0;')(v);
    const allowed = [];
    for (let v = 0; v < 128; v++) if (!refused(v)) allowed.push(v);
    expect(allowed).toEqual([0, 2, 8, 10, 16, 18, 24, 26, 32, 34, 40, 42, 48, 50, 56, 58,
      64, 66, 72, 74, 80, 82, 88, 90, 96, 98, 104, 106, 112, 114, 120, 122]);
    // -DPEARL_AMD_ABLATE alone defines it as 1.
    for (const v of [-1, 1, 4, 128]) expect([v, refused(v)]).toEqual([v, true]);
    // And no block tests a bit it refuses.
    const tests = [...FOLD.matchAll(/^#(?:el)?if !?\(?PEARL_AMD_ABLATE & (\d+)\)?$/gm)].map((m) => Number(m[1]));
    expect(tests).toContain(32);
    for (const b of tests) expect(b & ~122).toBe(0);
  });

  it('leaves a release build as it was: at 0 every block it guards is out', () => {
    const rel = ablated(FOLD, 0);
    const code = rel.split('\n').filter((l) => !/^\s*\/\//.test(l));
    // Only the knob's default and the cap's name are left.
    expect(code.filter((l) => /ABLATE|CAP_TIMING|0x5A5A5A5A|\b[ab]0\b/.test(l))).toEqual([
      '#define PEARL_AMD_CAP_TIMING 0x800u    // a PEARL_AMD_ABLATE build: timing only, never mines',
      '#ifndef PEARL_AMD_ABLATE',
      '#define PEARL_AMD_ABLATE 0',
    ]);
    // Today's code, where each bit cuts in.
    expect(rel).toContain('  const int8_t *gA = Ap + (size_t)tileRow * k;\n  const int8_t *gB = Bp + (size_t)(col0 + tileCol) * k;\n');
    expect(rel).toContain('  PEARL_AMD_STORE(0)\n  __syncthreads();\n  // The end of chunk c: its transcript word');
    // The k-steps: RDNA 4's paired fragment reads (PEARL_AMD_KPAIR), then everyone else's.
    expect(slabStep(rel)).toContain([
      '    for (int kk = 0; kk < KS / P::kTK; kk++) {',
      '#if PEARL_AMD_KPAIR',
      '      if constexpr (std::is_same<P, PearlAmdWmma12>::value) {'].join('\n'));
    expect(slabStep(rel)).toContain([
      '      } else',
      '#endif',
      '      {',
      '        typename P::AFrag a[P::kMT];',
      '        typename P::BFrag b[P::kNT];'].join('\n'));
    expect(slabStep(rel)).toContain('        pearl_amd_mma_step<P>(acc, a, b);\n      }\n    }\n');
    expect(slabStep(rel)).toContain([
      '    if (s + 1 < slabs) { PEARL_AMD_STORER(B ^ 1, pre2[B ^ 1]) }   // slab s + 1, loaded a step ago',
      '#if PEARL_AMD_PFX',
      '    pearl_amd_lds_barrier();',
      '#else',
      '    __syncthreads();',
      '#endif',
      '  };'].join('\n'));
    expect(oneSlab(rel)).toMatch(/^ {2}if \(s < slabs\) slab_step\(s, std::integral_constant<int, 0>\(\)\);\n#else\n {2}for \(uint32_t s = 0;/);
    expect(rel).toContain('        w[r] = keepLo[rb][cb][q];\n        w[r + 8] = keepHi[rb][cb][q];\n      }\n');
    expect(caps(rel)).toContain('  if (PEARL_AMD_KWRAP) out[0] |= PEARL_AMD_CAP_KWRAP;\n  out[1] = (uint32_t)warpSize;');
  });

  it('bit 2 takes out the s_barrier and keeps the fences around it', () => {
    const code = (t) => t.split('\n').filter((l) => !/^\s*\/\//.test(l)).join('\n');
    const base = slabStep(ablated(FOLD, 0));
    const step = slabStep(ablated(FOLD, 2));
    expect(code(step)).not.toMatch(/__syncthreads\(\)|pearl_amd_lds_barrier\(\)|s_barrier/);
    expect(step).toContain('P::load_a(aBase');
    // On a GPU, the pair each barrier puts round its s_barrier: PFX's LDS-only pair, else
    // __syncthreads()'s, which is for every address space. HIP-CPU has neither.
    const pair = (scope) => ['RELEASE', 'ACQUIRE']
      .map((o) => '    __builtin_amdgcn_fence(__ATOMIC_' + o + ', "workgroup"' + scope + ');').join('\n');
    expect(code(step)).toBe(code(base).replace('#if PEARL_AMD_PFX\n    pearl_amd_lds_barrier();\n#else\n    __syncthreads();\n#endif\n',
      ['#if !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__)', '#if PEARL_AMD_PFX', pair(', "local"'), '#else', pair(''),
        '#endif', '#endif', ''].join('\n')));
    expect(slice(FOLD, 'void pearl_amd_lds_barrier() {', '\n}\n')).toContain([
      '  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");',
      '  __builtin_amdgcn_s_barrier();',
      '  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");'].join('\n'));
    expect(FOLD).toContain('__syncthreads() is a workgroup fence for every address space');
    // The barrier after the first slab's stores stays: every block starts with whole LDS.
    expect(ablated(FOLD, 2)).toContain('  PEARL_AMD_STORE(0)\n  __syncthreads();\n');
  });

  it('bit 8 gives each k-step new fragments with no LDS load, and no fragment lives longer', () => {
    const t = ablated(FOLD, 8);
    const step = slabStep(t);
    // The k-steps load nothing (the one read a slab that keeps LDS in use is tested below).
    expect(kSteps(step)).not.toMatch(/P::load_[ab]\(/);
    // On a GPU: one asm a k-step that defines every fragment the k-step uses, for the RDNA folds.
    const gpu = grab(step, /\n#if PEARL_AMD_EMU \|\| !defined\(__HIP_DEVICE_COMPILE__\)\n {6}pearl_amd_mma_step<P>\(acc, a0, b0\);\n#else\n([\s\S]*?)#endif\n/);
    expect(gpu).toContain('static_assert(P::kMT == 4 && P::kNT == 4, "PEARL_AMD_ABLATE bit 8 is for the RDNA folds");');
    expect(gpu).toContain('      typename P::AFrag a[P::kMT];\n      typename P::BFrag b[P::kNT];\n');
    expect(gpu).toMatch(/\);\n {6}pearl_amd_mma_step<P>\(acc, a, b\);\n$/);
    const asm = grab(gpu, /\n {6}(asm[\s\S]*?\);)\n/);
    // Not volatile and no memory clobber: it orders no load, store or wait.
    expect(asm).toMatch(/^asm\("" : /);
    expect(asm).not.toMatch(/volatile|memory|"m"/);
    expect([...asm.matchAll(/"=v"\(([ab]\[\d\])\)/g)].map((m) => m[1]))
      .toEqual(['a[0]', 'a[1]', 'a[2]', 'a[3]', 'b[0]', 'b[1]', 'b[2]', 'b[3]']);
    for (const p of ['PearlAmdWmma11', 'PearlAmdWmma12']) {
      expect(slice(FOLD, 'struct ' + p + ' {', '\n};\n')).toContain('kMT = 4, kNT = 4;');
    }
    // Its one input differs at every k-step, the two steps of the unrolled pair too, so the compiler
    // can't hoist it out of the loop or merge two of them.
    const input = grab(asm, /: "s"\((.*)\)\);$/);
    const KS = Number(grab(FOLD, /#define PEARL_AMD_SLAB (\d+)u/));
    const kTK = Number(grab(slice(FOLD, 'struct PearlAmdWmma12 {', '\n};\n'), /kTK = (\d+);/));
    const tag = new Function('s', 'kk', 'return ' + input.replace(/\(uint32_t\)/g, '')
      .replace('KS', String(KS)).replace('P::kTK', String(kTK)) + ';');
    const seen = new Set();
    for (let s = 0; s < 64; s++) for (let kk = 0; kk < KS / kTK; kk++) seen.add(tag(s, kk));
    expect(seen.size).toBe(64 * (KS / kTK));
    // No fragment is kept across the loop on a GPU: a0 and b0 are HIP-CPU's.
    const kept = grab(t, /\n(#if PEARL_AMD_EMU \|\| !defined\(__HIP_DEVICE_COMPILE__\)\n {2}\/\/ PEARL_AMD_ABLATE bit 8, HIP-CPU[\s\S]*?\n#endif)\n/);
    expect(kept).toContain('  typename P::AFrag a0[P::kMT];\n  typename P::BFrag b0[P::kNT];\n');
    // The next slab still goes to LDS, behind the barrier.
    expect(step).toContain('PEARL_AMD_STORER(B ^ 1, pre2[B ^ 1])');
    expect(step).toContain('    pearl_amd_lds_barrier();\n#else\n    __syncthreads();\n');
    // HIP-CPU's are loaded after slab 0 is in LDS, before the loop.
    const a0 = t.indexOf('a0[mt] = P::load_a(sA[0] + wr * WM * S + mt * P::kTM * S, S, lane);');
    expect(a0).toBeGreaterThan(t.indexOf('  PEARL_AMD_STORE(0)\n  __syncthreads();\n'));
    expect(a0).toBeLessThan(t.indexOf('auto slab_step'));
    expect(t).toContain('b0[nt] = P::load_b(sB[0] + wc * WN * S + nt * P::kTN * S, S, lane);');
  });

  it('bit 16 points every block at tile 0\'s operands, and still stores each block\'s own words', () => {
    const t = ablated(FOLD, 16);
    expect(t).toContain('  const int8_t *gA = Ap;\n  const int8_t *gB = Bp + (size_t)col0 * k;\n');
    expect(t).not.toContain('(size_t)tileRow * k');
    expect(t).toContain('2u * ((tileRow + (uint32_t)(wr * WM + rb * 32)) / 32u)');
    expect(slabStep(t)).toBe(slabStep(ablated(FOLD, 0)));
  });

  it('bit 32 takes the slab loop\'s staging out, and leaves the rest of the step as it was', () => {
    const code = (t) => t.split('\n').filter((l) => !/^\s*\/\//.test(l)).join('\n');
    const base = code(slabStep(ablated(FOLD, 0)));
    const step = code(slabStep(ablated(FOLD, 32)));
    // No slab ahead is loaded, and nothing is stored to LDS in the loop; the release does both.
    expect(step).not.toMatch(/PEARL_AMD_LOADR|PEARL_AMD_STORER/);
    expect(base).toContain('PEARL_AMD_LOADR(n, pre2[B])');
    expect(base).toContain('if (s + 2 < slabs) { PEARL_AMD_LOADR(s + 2, pre2[B]) }');
    expect(base).toContain('if (s + 1 < slabs) { PEARL_AMD_STORER(B ^ 1, pre2[B ^ 1]) }');
    // Everything else is the release's: take its loads and its store out, and the two match.
    const loads = slice(base, '#if PEARL_AMD_PFX\n    {', '#endif\n');
    const store = '    if (s + 1 < slabs) { PEARL_AMD_STORER(B ^ 1, pre2[B ^ 1]) }   // slab s + 1, loaded a step ago\n';
    expect(base).toContain(store);
    expect(step).toBe(base.replace(loads, '').replace(store, ''));
    // Slab 0 still goes to LDS before the loop, so every block starts with whole LDS.
    expect(ablated(FOLD, 32)).toContain('  PEARL_AMD_LOAD(0)\n  PEARL_AMD_STORE(0)\n  __syncthreads();\n');
    // With 2 and 8 the loop is the WMMAs, the readout and the one LDS read a slab that bit 8 keeps.
    const lean = code(slabStep(ablated(FOLD, 42)));
    expect(lean).not.toMatch(/PEARL_AMD_LOADR|PEARL_AMD_STORER|__syncthreads\(\)|pearl_amd_lds_barrier\(\)|s_barrier/);
    expect(lean).toContain('pearl_amd_mma_step<P>(acc, a, b);');
    expect(lean).toContain('readout((s + 1) / perChunk - 1);');
    expect(lean.match(/P::load_[ab]\(/g)).toEqual(['P::load_a(']);
  });

  it('stops a pass that builds the one-slab loop with bit 2, 8 or 32, which only the PF2 loop has', () => {
    const stop = 'static_assert(sizeof(P) == 0, "PEARL_AMD_ABLATE bits 2, 8 and 32 are in the PF2 loop only';
    for (const v of [2, 8, 10, 18, 24, 26, 32, 34, 40, 42, 48, 58]) {
      expect([v, oneSlab(ablated(FOLD, v)).includes(stop)]).toEqual([v, true]);
    }
    for (const v of [0, 16]) expect([v, oneSlab(ablated(FOLD, v)).includes('static_assert')]).toEqual([v, false]);
    // Inside the template: a pass that never builds a matrix fold (the host's, gfx1030's) isn't stopped.
    const body = slice(FOLD, '__device__ __forceinline__ void pearl_amd_fold_body(', '\n}\n');
    expect(body).toContain(stop);
  });

  it('makes every word wrong in every ablation build, so the fold check always fails', () => {
    for (const v of BITS) {
      const t = ablated(FOLD, v);
      // Bit 8's sink goes in too (below).
      const sink = v & 8 ? ' ^ sink' : '';
      expect(t).toContain('        w[r] = keepLo[rb][cb][q] ^ 0x5A5A5A5Au' + sink + ';\n'
        + '        w[r + 8] = keepHi[rb][cb][q] ^ 0x5A5A5A5Au' + sink + ';\n');
      // The reference fold is left alone.
      expect(refFold(t)).toBe(refFold(ablated(FOLD, 0)));
    }
    expect(FOLD.match(/\^ 0x5A5A5A5Au;/g)).toHaveLength(2);
    expect(FOLD.match(/\^ 0x5A5A5A5Au \^ sink;/g)).toHaveLength(2);
  });

  it('bit 8 keeps one real fragment read a slab, the release\'s, and the stored words need it', () => {
    const t = ablated(FOLD, 8);
    const step = slabStep(t);
    // One read a slab on a GPU, through the release's loader, at an address picked at run time.
    const reads = step.match(/P::load_[ab]\([^;]*;/g);
    expect(reads).toEqual(['P::load_a((s & 2u) ? bBase : aBase, S, lane);']);
    const read = '    const typename P::AFrag kept = P::load_a((s & 2u) ? bBase : aBase, S, lane);';
    // In a GPU's device pass only: the last directive before it is that guard, and it ends there.
    const before = step.slice(0, step.indexOf(read));
    expect(before.slice(before.lastIndexOf('\n#'))).toMatch(/^\n#if !PEARL_AMD_EMU && defined\(__HIP_DEVICE_COMPILE__\)\n/);
    expect(step).toContain(read + '\n#endif\n');
    // Every one of the four LDS buffers gets a reader. The loop runs step(s, B = 0) and step(s + 1, B = 1) for
    // even s, then step(s, 0) for an odd count; B picks the half, the C's pick the array: run here as JS.
    expect(t).toContain('    slab_step(s, std::integral_constant<int, 0>());\n    slab_step(s + 1, std::integral_constant<int, 1>());');
    expect(step).toContain('    const int8_t *aBase = sA[B] + wr * WM * S;\n    const int8_t *bBase = sB[B] + wc * WN * S;\n');
    const pick = new Function('s', 'aBase', 'bBase', 'return ' + grab(read, /P::load_a\((.*), S, lane\)/).replace(/2u/, '2') + ';');
    const seen = new Set();
    for (let s = 0; s < 4; s++) seen.add(pick(s, 'sA', 'sB') + '[' + (s % 2) + ']');
    expect([...seen].sort()).toEqual(['sA[0]', 'sA[1]', 'sB[0]', 'sB[1]']);
    // Those are the release's addresses at k-step 0 of tile 0: its loads, with mt, nt and kk all 0.
    const rel = kSteps(slabStep(ablated(FOLD, 0)));
    const at = (re) => grab(rel, re).replace(/\b(mt|nt|kk)\b/g, '0');
    expect(at(/a\[mt\] = (P::load_a\(.*\));/)).toBe('P::load_a(aBase + 0 * P::kTM * S + 0 * P::kTK, S, lane)');
    expect(at(/b\[nt\] = (P::load_b\(.*\));/)).toBe('P::load_b(bBase + 0 * P::kTN * S + 0 * P::kTK, S, lane)');
    // The release reads KS / kTK * (kMT + kNT) fragments a slab; this reads 1 of them.
    const KS = Number(grab(FOLD, /#define PEARL_AMD_SLAB (\d+)u/));
    for (const p of ['PearlAmdWmma11', 'PearlAmdWmma12']) {
      const pol = slice(FOLD, 'struct ' + p + ' {', '\n};\n');
      const kTK = Number(grab(pol, /kTK = (\d+);/));
      const [kMT, kNT] = grab(pol, /(kMT = \d+, kNT = \d+);/).match(/\d+/g).map(Number);
      expect([p, (KS / kTK) * (kMT + kNT)]).toEqual([p, 16]);
      // Both readers return the same fragment type, and load_b is load_a, so one call reads either array.
      expect(pol).toMatch(/typedef (\w+) AFrag;\n {2}typedef \1 BFrag;/);
      expect(pol).toContain('  static __device__ __forceinline__ BFrag load_b(const int8_t *s, int stride, int lane) {\n'
        + '    return load_a(s, stride, lane);\n  }');
    }
    // After the k-steps and the readout, before the next slab's store: so not in the matrix work.
    expect(step.indexOf(read)).toBeGreaterThan(step.indexOf('readout((s + 1) / perChunk - 1);'));
    expect(step.indexOf(read)).toBeLessThan(step.indexOf('PEARL_AMD_STORER(B ^ 1, pre2[B ^ 1])'));
    // Its words go into the sink after the barrier, so its wait is the barrier's.
    const xor = '    for (int i = 0; i < (int)(sizeof kept / sizeof(int32_t)); i++) sink ^= (uint32_t)kept[i];';
    expect(step.endsWith('    pearl_amd_lds_barrier();\n#else\n    __syncthreads();\n#endif\n'
      + '#if !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__)\n#pragma unroll\n' + xor + '\n#endif\n  };\n')).toBe(true);
    expect(t).toContain('  uint32_t sink = 0u;   // PEARL_AMD_ABLATE bit 8');
    // The sink is used only there and in the stored words.
    expect(t.split('\n').filter((l) => /\bsink\b/.test(l) && !/^\s*\/\//.test(l))).toEqual([
      '  uint32_t sink = 0u;   // PEARL_AMD_ABLATE bit 8: the words of the one fragment read a slab (slab_step)',
      xor,
      '        w[r] = keepLo[rb][cb][q] ^ 0x5A5A5A5Au ^ sink;',
      '        w[r + 8] = keepHi[rb][cb][q] ^ 0x5A5A5A5Au ^ sink;',
    ]);
    // Only in bit 8 builds; HIP-CPU's path is still the first slab's fragments, with no read a slab.
    const code = (x) => x.split('\n').map((l) => l.replace(/(^|\s)\/\/.*$/, '')).join('\n');
    for (const v of [0, 2, 16, 18]) expect([v, /\bsink\b|\bkept\b/.test(code(ablated(FOLD, v)))]).toEqual([v, false]);
    expect(step).toContain('#if PEARL_AMD_EMU || !defined(__HIP_DEVICE_COMPILE__)\n      pearl_amd_mma_step<P>(acc, a0, b0);\n#else\n');
  });

  it('marks every code object it is in with a caps bit of its own', () => {
    const bits = [...FOLD.matchAll(/#define PEARL_AMD_CAP_(\w+) (0x[0-9a-f]+|\d+)u/g)].map((m) => [m[1], Number(m[2])]);
    expect(bits.find(([n]) => n === 'TIMING')[1]).toBe(0x800);
    for (const [n, v] of bits) if (n !== 'TIMING') expect([n, v & 0x800]).toEqual([n, 0]);
    expect(caps(ablated(FOLD, 0))).not.toContain('CAP_TIMING');
    for (const v of BITS) {
      const c = caps(ablated(FOLD, v));
      // After the per-target bits, so every target says so, one with no matrix fold too.
      expect(c).toContain('  if (PEARL_AMD_KWRAP) out[0] |= PEARL_AMD_CAP_KWRAP;\n  out[0] |= PEARL_AMD_CAP_TIMING;\n');
      expect(c.indexOf('out[0] |= PEARL_AMD_CAP_TIMING;')).toBeGreaterThan(c.lastIndexOf('#endif'));
    }
  });
});

describe('the AMD host with a timing build (PEARL_AMD_CAP_TIMING)', () => {
  const submit = HOST.slice(HOST.indexOf('extern "C" bool pearl_host_submit('));
  const capValue = (n) => Number(grab(FOLD, new RegExp('#define PEARL_AMD_CAP_' + n + ' (0x[0-9a-f]+|\\d+)u')));

  // The target guard, run: its C as JS, with ctx a plain object and snprintf keeping the message.
  function guard(ctx) {
    const start = submit.indexOf('  if (ctx->readoutDrop || ctx->timing) {');
    expect(start).toBeGreaterThan(0);
    const js = submit.slice(start, submit.indexOf('\n  }\n', start) + 4)
      .replace(/ctx->/g, 'ctx.')
      .replace(/\b(bool|int)\b/g, 'let')
      .replace(/"\s*\n\s*"/g, '')   // C joins adjacent string literals
      .replace(/PEARL_HASH_BYTES/g, '32');
    let msg = null;
    const ok = new Function('ctx', 'err', 'err_len', 'snprintf', js + '\nreturn true;')(
      ctx, 'buf', 512, (e, n, text) => { msg = text; });
    return { ok, msg };
  }
  const target = (byte) => Array.from({ length: 32 }, (_, i) => (i === byte ? 1 : 0));

  it('reads the cap, and refuses every target but zero before any fold is launched', () => {
    expect(HOST).toContain('ctx->timing = (caps[0] & PEARL_AMD_CAP_TIMING) != 0;');
    const at = submit.indexOf('if (ctx->readoutDrop || ctx->timing) {');
    expect(at).toBeGreaterThan(0);
    expect(at).toBeLessThan(submit.indexOf('hipLaunchKernelGGL(pearl_amd_fold_'));
    const timingOnly = 'ablation build is timing-only: it runs only at target 0 (hashrate.js), never on a job; '
      + 'build without PEARL_AMD_ABLATE to mine';
    // hashrate.js's target, zero, runs; a job's doesn't.
    expect(guard({ readoutDrop: false, timing: true, target: target(-1) })).toEqual({ ok: true, msg: null });
    expect(guard({ readoutDrop: false, timing: true, target: target(31) })).toEqual({ ok: false, msg: timingOnly });
    expect(guard({ readoutDrop: false, timing: true, target: target(0) })).toEqual({ ok: false, msg: timingOnly });
    // A release core mines any target; a readout-drop build keeps its own message.
    expect(guard({ readoutDrop: false, timing: false, target: target(3) })).toEqual({ ok: true, msg: null });
    for (const timing of [false, true]) {
      const r = guard({ readoutDrop: true, timing, target: target(3) });
      expect([r.ok, r.msg]).toEqual([false, 'readout-drop build is timing-only: it runs only at target 0 '
        + '(hashrate.js), never on a job; build without PEARL_AMD_READOUT_DROP to mine']);
    }
    expect(HASHRATE).toContain("core.setJob({ header, target: 0n, jobId: 'bench-' + jobs++ });");
  });

  it('chooses the k a release would, so an RDNA 4 ablation times k 4096', () => {
    const cardK = slice(HOST, 'extern "C" uint32_t pearl_host_card_k(', '\n}\n');
    expect(cardK).not.toContain('CAP_TIMING');
    const cond = grab(cardK, /\n {2}if \((e != hipSuccess \|\| [^\n]*)\) return k;/);
    const keeps = (c) => new Function('c', 'return !!(' + cond.replace('e != hipSuccess', 'false')
      .replace(/caps\[0\]/g, 'c').replace(/PEARL_AMD_CAP_(\w+)/g, (_, n) => String(capValue(n))) + ');')(c);
    const gfx12 = capValue('REF') | capValue('WMMA12') | capValue('KWRAP');
    expect(keeps(gfx12)).toBe(false);
    expect(keeps(gfx12 | capValue('TIMING'))).toBe(false);
    expect(keeps(gfx12 | capValue('DROP'))).toBe(true);
    // The wrap readout's default doesn't look at the knob, so a gfx12 ablation keeps it.
    expect(slice(FOLD, '#ifndef PEARL_AMD_KWRAP\n#if ', '#endif\n#endif\n')).not.toContain('ABLATE');
  });
});

describe('build-amd.sh and an ablation build', () => {
  const caseBlock = block(BUILD, 'case "${PEARL_AMD_CFLAGS:-}" in', 'esac');
  const tryGuard = (mode, flags, ci) => bash('mode=' + mode + '\n' + caseBlock + 'echo allowed',
    Object.assign({ PEARL_AMD_CFLAGS: flags }, ci || {}));

  bashIt('refuses it in the builds that ship, and allows it on a test box', () => {
    for (const [mode, flags, ci, ok] of [
      ['winlib', '-DPEARL_AMD_ABLATE=8', null, false],
      ['winlib', '-DPEARL_AMD_ABLATE=0', null, false],
      ['gpu', '-O3 -DPEARL_AMD_ABLATE=2', { CI: 'true' }, false],
      ['gpu', '-DPEARL_AMD_ABLATE=16', { GITHUB_ACTIONS: 'true' }, false],
      ['gpu', '-DPEARL_AMD_ABLATE=2', null, true],                        // sweep.sh on a Salad box
      ['gpu', '-DPEARL_AMD_PFX=1 -DPEARL_AMD_TID=1 -DPEARL_AMD_ABLATE=16', null, true],
      ['check', '-DPEARL_AMD_ABLATE=8', { CI: 'true' }, true],           // compiles only
      ['cpu', '-DPEARL_AMD_PF2=1 -DPEARL_AMD_ABLATE=2', null, true],
    ]) {
      const r = tryGuard(mode, flags, ci);
      expect([mode, flags, ci, r.stdout.trim(), r.status]).toEqual([mode, flags, ci, ok ? 'allowed' : '', ok ? 0 : 1]);
      if (!ok) expect(r.stderr).toContain('ablation build is timing-only: PEARL_AMD_ABLATE is refused in a ' + mode);
    }
    // With the readout drop as well, still refused.
    const both = tryGuard('winlib', '-DPEARL_AMD_READOUT_DROP=1 -DPEARL_AMD_ABLATE=2');
    expect([both.status, both.stdout]).toEqual([1, '']);
  });

  bashIt('stops before it builds anything', () => {
    const r = spawnSync('bash', [path.join(AMD, 'build-amd.sh'), 'winlib'], {
      env: Object.assign({}, process.env, { PEARL_AMD_CFLAGS: '-DPEARL_AMD_ABLATE=2' }), encoding: 'utf8',
    });
    expect(r.status).toBe(1);
    expect(r.stderr).toContain('ablation build is timing-only');
    expect(r.stdout).toBe('');
  });
});

describe('PEARL_OPERAND_FILL', () => {
  it('a build flag sets it, it is 48 without one, and it must be int7', () => {
    expect(CONFIG).toContain('#ifndef PEARL_OPERAND_FILL\n#define PEARL_OPERAND_FILL 48\n#endif\n');
    expect(CONFIG.match(/#define PEARL_OPERAND_FILL\b/g)).toHaveLength(1);
    const cond = grab(CONFIG, /^#if (PEARL_OPERAND_FILL < [^\n]*)\n#error "PEARL_OPERAND_FILL must be in \[-63, 63\]: int7/m);
    const refused = (f) => new Function('PEARL_OPERAND_FILL', 'return ' + cond + ';')(f);
    expect([-64, -63, 0, 32, 48, 63, 64].map(refused)).toEqual([true, false, false, false, false, false, true]);
    // int7 here is +-63, as noise.js has it: the miner's own operands are drawn in [-63, 63].
    expect(read('..', 'src', 'shared', 'miner', 'noise.js')).toContain('The operands are int7 (+-63)');
  });

  it('is set by no build that ships', () => {
    for (const f of ['.github/workflows/native-core.yml', '.github/workflows/miner-build.yml',
      'earn/native/build-local.ps1', 'earn/native/binding.gyp', 'earn/native/amd/build-amd.sh']) {
      const text = fs.readFileSync(path.join(ROOT, f), 'utf8');
      expect([f, /PEARL_OPERAND_FILL/.test(text)]).toEqual([f, false]);
    }
  });
});

describe('the fold test\'s miner operands', () => {
  it('takes int8, the default and the same draws as before, or miner, and nothing else', () => {
    expect(FOLDTEST).toContain('//   pearl_amd_foldtest [m=256] [cols=512] [seed=1] [reps=1] [check=1] [fold|all] [k] [int8|miner]');
    expect(FOLDTEST).toContain('const bool miner = argc > 8 && strcmp(argv[8], "miner") == 0;');
    expect(FOLDTEST).toContain('if (argc > 8 && !miner && strcmp(argv[8], "int8") != 0) {');
    expect(FOLDTEST).toContain([
      '  if (miner) {',
      '    miner_operands(A, m, k, rank);',
      '    miner_operands(B, n, k, rank);',
      '  } else {',
      '    for (auto &v : A) v = (int8_t)next_rand();',
      '    for (auto &v : B) v = (int8_t)next_rand();',
      '  }'].join('\n'));
    // k 0 is the k the card mines, which sweep.sh passes when it pins none.
    expect(FOLDTEST).toContain('const uint32_t k = kArg ? kArg : (wrap && PEARL_FOLD_K < 4096u ? 4096u : (uint32_t)PEARL_FOLD_K);');
  });

  it('fills as the miner does: the fill plus the protocol\'s noise, in noise.js\'s shape', () => {
    const fn = slice(FOLDTEST, 'void miner_operands(', '\n}\n');
    // One (p0, p1) a k, p1 != p0, as noise.js's generatePermutationMatrix: p1 = p0 ^ (1 + mulhi(rank - 1, u)).
    expect(fn).toContain('p0[kk] = u & (rank - 1u);');
    expect(fn).toContain('p1[kk] = p0[kk] ^ (1u + (uint32_t)(((uint64_t)(rank - 1u) * u) >> 32));');
    // A row's draws in [-32, 32), as generateUniformRandomMatrix.
    const m = /v = \(int32_t\)\(next_rand\(\) & (\d+)u\) - (\d+);/.exec(fn);
    expect([Number(m[1]), Number(m[2])]).toEqual([noise.RANGE_MASK, noise.ZERO_POINT_TRANSLATION]);
    // Noise e[p0] - e[p1] on the fill, saturated as the reference's convert.
    expect(fn).toContain('const int32_t v = PEARL_OPERAND_FILL + e[p0[kk]] - e[p1[kk]];');
    expect(fn).toContain('(int8_t)(v < -128 ? -128 : v > 127 ? 127 : v)');
    // At fill 48 that is [-15, 111], as pearl_config.h says: no byte saturates.
    const fill = Number(grab(CONFIG, /#define PEARL_OPERAND_FILL (\d+)/));
    const span = (noise.RANGE_MASK - noise.ZERO_POINT_TRANSLATION) - -noise.ZERO_POINT_TRANSLATION;   // max - min draw
    expect([fill - span, fill + span]).toEqual([-15, 111]);
    // The protocol's own noise, a few rows of it, lands there too.
    const key = Buffer.alloc(32, 7);
    const perm = noise.generatePermutationMatrix(noise.SEED_LABEL_A, key, 2048, 128);
    const rows = noise.generateUniformRandomMatrix(noise.SEED_LABEL_A, key, [0, 1, 2, 3], 128);
    let lo = 0, hi = 0;
    for (const r of rows) for (const x of noise.matvecSparsePerm(perm, r, 2048)) { lo = Math.min(lo, x); hi = Math.max(hi, x); }
    expect(lo).toBeGreaterThanOrEqual(-span);
    expect(hi).toBeLessThanOrEqual(span);
  });
});

describe('sweep.sh SWEEP_OPERANDS, with an ablation build', () => {
  // A stand-in build-amd.sh whose fold test logs its arguments, finds wrong words in an ablation
  // build's check, and times anything; and a stand-in hashrate.js.
  const STUB = [
    '#!/bin/bash',
    'mkdir -p build',
    'printf \'#!/bin/bash\\nf="%s"\\n\' "${PEARL_AMD_CFLAGS:-}" > build/pearl_amd_foldtest',
    'cat >> build/pearl_amd_foldtest <<\'FT\'',
    'echo "$*" >> "$ARGS_LOG"',
    'if [ "$5" = 0 ]; then echo "$6  timed only: $1 x $2, $4 launches, 1.0000 s, 100.000 TMAC/s"; exit 0; fi',
    'case "$f" in *ABLATE*) echo "$6  FAIL: 4096 of 4096 words differ; first: region 0 word 0, got 1 want 2"; echo FAIL; exit 1 ;; esac',
    'echo "$6  ok: $1 x $2, all 16 words match; 1 launches, 0.0100 s, 1.000 TMAC/s"; echo PASS',
    'FT',
    'chmod +x build/pearl_amd_foldtest',
    'echo "${PEARL_AMD_CFLAGS:-}" > build/pearl_core_hip.node',
    '',
  ].join('\n');

  let tmp;
  beforeEach(() => { tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'amd-sweep-abl-')); });
  afterEach(() => { fs.rmSync(tmp, { recursive: true, force: true }); });

  // Runs a copy of sweep.sh with the stand-ins, writing nothing outside tmp: the container-log copy of
  // each line goes to /dev/null.
  function sweep(env) {
    const amd = path.join(tmp, 'amd');
    fs.mkdirSync(path.join(amd, 'test'), { recursive: true });
    fs.mkdirSync(path.join(tmp, 'probes'), { recursive: true });
    fs.mkdirSync(path.join(tmp, 'root'), { recursive: true });
    fs.writeFileSync(path.join(amd, 'build-amd.sh'), STUB, { mode: 0o755 });
    fs.writeFileSync(path.join(tmp, 'probes', 'hashrate.js'), 'console.log(JSON.stringify({ th: 100 }));\n');
    fs.writeFileSync(path.join(amd, 'test', 'sweep.sh'), SWEEP.replace('/proc/1/fd/1', '/dev/null'));
    const log = path.join(tmp, 'args.log');
    fs.rmSync(log, { force: true });
    const e = { ...process.env, ARCH: 'gfx1100', SWEEP_ROOT: path.join(tmp, 'root').replace(/\\/g, '/'),
      ARGS_LOG: log.replace(/\\/g, '/'), SWEEP_ROUNDS: '1', ...env };
    delete e.PEARL_PROFILE_K;
    if (!('SWEEP_OPERANDS' in env)) delete e.SWEEP_OPERANDS;
    const r = spawnSync('bash', ['test/sweep.sh'], { cwd: amd, encoding: 'utf8', env: e });
    return {
      status: r.status,
      lines: r.stdout.split('\n').filter(Boolean).map((l) => l.replace(/^\[amdtest\] \S+ sweep /, '')),
      args: fs.existsSync(log) ? fs.readFileSync(log, 'utf8').split('\n').filter(Boolean) : [],
    };
  }

  bashIt('times on the miner\'s operands when asked, checks on int8, and times an ablation build', () => {
    const { status, lines, args } = sweep({
      SWEEP: 'base=;abl2=-DPEARL_AMD_ABLATE=2', SWEEP_TIMEONLY: 'abl2', SWEEP_OPERANDS: 'miner',
    });
    expect(status).toBe(0);
    expect(lines[0]).toBe('start gfx1100 fold wmma11; variants: base=;abl2=-DPEARL_AMD_ABLATE=2 (timing only: abl2); '
      + 'fold timing 131072:4096:10 on miner operands; hashrate 40s+10s x 1');
    expect(lines).toContain('abl2 check FAIL (expected: timing only)');
    expect(args).toEqual([
      '1024 1024 7 1 1 wmma11', '131072 4096 1 10 0 wmma11 0 miner',
      '1024 1024 7 1 1 wmma11', '131072 4096 1 10 0 wmma11 0 miner',
    ]);
    expect(lines).toContain('abl2 fold 131072 x 4096: wmma11 timed only: 131072 x 4096, 10 launches, 1.0000 s, 100.000 TMAC/s');
    expect(lines.filter((l) => / hashrate: /.test(l))).toEqual(['base hashrate: {"th":100}', 'abl2 hashrate: {"th":100}']);
  }, 30000);

  bashIt('times on int8 by default, at the k the card mines', () => {
    const { status, lines, args } = sweep({ SWEEP: 'base=' });
    expect(status).toBe(0);
    expect(lines[0]).toBe('start gfx1100 fold wmma11; variants: base=; fold timing 131072:4096:10; hashrate 40s+10s x 1');
    expect(args).toEqual(['1024 1024 7 1 1 wmma11', '131072 4096 1 10 0 wmma11 0']);
    expect(sweep({ SWEEP: 'base=', SWEEP_OPERANDS: 'int8' }).args[1]).toBe('131072 4096 1 10 0 wmma11 0 int8');
  }, 30000);

  bashIt('refuses operands it doesn\'t know before it builds anything', () => {
    const { status, lines, args } = sweep({ SWEEP: 'base=', SWEEP_OPERANDS: 'mining' });
    expect([status, lines, args]).toEqual([2, ['SWEEP_OPERANDS is int8 or miner, not mining'], []]);
  }, 30000);
});
