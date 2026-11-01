'use strict';

// PEARL_AMD_LDS_AHEAD (earn/native/amd/src/pearl_amd_fold.h, item 15): RDNA 4's fold loads each slab's WMMA
// fragments from LDS a k-step before their WMMAs, with the next slab's LDS store and the slab barrier a k-step
// earlier. These pin that it is off by default and leaves every release pass, and every pass but gfx12's and
// HIP-CPU's, as it was; that it can ship; and that it moves loads, not math. For that, its step is run here as JS,
// out of the C, against a model of the two LDS buffers, the two register sets and the barriers, over many slab
// counts: every WMMA must get the fragments of the slab and k-step the release gives it, in the release's order,
// and no LDS buffer may be written and read between the same two barriers. No HIP compiler runs here: the C is
// read as text (helpers/amdSwitches.js).

const fs = require('fs');
const path = require('path');
const { preprocess } = require('./helpers/amdSwitches');

const NATIVE = path.join(__dirname, '..', 'native');
const read = (...f) => fs.readFileSync(path.join(NATIVE, ...f), 'utf8').replace(/\r\n/g, '\n');
const FOLD = read('amd', 'src', 'pearl_amd_fold.h');
const HOST = read('amd', 'src', 'pearl_host_hip.cpp');
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
// Code lines: no comments, and no directives (the preprocessor leaves none in a pass's tokens).
const codeLines = (t) => t.split('\n').filter((l) => !/^\s*(\/\/|#(?!pragma))/.test(l) && l.trim() !== '');
const RELEASE = { PEARL_AMD_ABLATE: 0, PEARL_AMD_STAMP: 0, PEARL_AMD_LDS_AHEAD: 0, PEARL_AMD_AHEAD_HERE: 0 };
const at = (sw) => preprocess(FOLD, Object.assign({}, RELEASE, sw));
// A gfx12 device pass with the switch at `form`, as far as these switches go: PFX on, RDNA 4's default.
const gfx12 = (form, sw) => at(Object.assign({ PEARL_AMD_LDS_AHEAD: form, PEARL_AMD_AHEAD_HERE: form ? 1 : 0,
  PEARL_AMD_WMMA11: 0, PEARL_AMD_WMMA12: 1, PEARL_AMD_PFX: 1, PEARL_AMD_EMU: 0 }, sw));
const aheadBlock = (t) => slice(t, '  if constexpr (std::is_same<P, PearlAmdWmma12>::value) {', '\n  } else {\n');
const aheadStep = (t) => slice(t, 'auto ahead_step = [&](uint32_t s, auto parity) {', '\n    };\n');
const slabStep = (t) => slice(t, 'auto slab_step = [&](uint32_t s, auto parity) {', '\n  };\n');
const releaseLoop = (t) => slice(t, '  auto slab_step = [&](uint32_t s, auto parity) {',
  '  if (s < slabs) slab_step(s, std::integral_constant<int, 0>());\n');
const AHEAD_NAMES = /ahead_step|pearl_amd_ahead_|\bfrags\b|\bf[ab]\[|sched_group_barrier|sched_barrier|\bNA\b|\bKK\b/;

describe('the LDS-ahead switch (PEARL_AMD_LDS_AHEAD)', () => {
  it('is 0 unless a build sets it, takes 0, 1 or 2, and no ablation bit 2 or 8', () => {
    expect(FOLD).toContain('#ifndef PEARL_AMD_LDS_AHEAD\n#define PEARL_AMD_LDS_AHEAD 0\n#endif\n');
    const fn = (cond, ...names) => new Function(...names, 'return !!(' + cond + ');');
    const bad = fn(grab(FOLD, /^#if (.*)\n#error "PEARL_AMD_LDS_AHEAD is 0, 1 or 2"/m), 'PEARL_AMD_LDS_AHEAD');
    expect([0, 1, 2, 3, -1].map((v) => bad(v))).toEqual([false, false, false, true, true]);
    const abl = fn(grab(FOLD, /^#if (.*)\n#error "PEARL_AMD_LDS_AHEAD moves what ablation bits 2 and 8 take out/m),
      'PEARL_AMD_LDS_AHEAD', 'PEARL_AMD_ABLATE');
    const refused = [];
    for (const a of [0, 1, 2]) for (const b of [0, 2, 8, 10, 16, 18, 24, 26]) if (abl(a, b)) refused.push([a, b]);
    expect(refused).toEqual([[1, 2], [1, 8], [1, 10], [1, 18], [1, 24], [1, 26], [2, 2], [2, 8], [2, 10], [2, 18],
      [2, 24], [2, 26]]);
  });

  it('applies in a gfx12 device pass and in HIP-CPU only, and there to the wmma12 fold only', () => {
    const cond = grab(FOLD, /^#if (.*)\n#define PEARL_AMD_AHEAD_HERE 1\n#else\n#define PEARL_AMD_AHEAD_HERE 0\n#endif\n/m);
    const here = new Function('PEARL_AMD_LDS_AHEAD', 'PEARL_AMD_WMMA12', 'PEARL_AMD_EMU', 'return !!(' + cond + ');');
    // The passes: the host's and every other device pass's (neither set), gfx12's, HIP-CPU's.
    const on = [];
    for (const a of [0, 1, 2]) for (const [w, e] of [[0, 0], [1, 0], [0, 1]]) if (here(a, w, e)) on.push([a, w, e]);
    expect(on).toEqual([[1, 1, 0], [1, 0, 1], [2, 1, 0], [2, 0, 1]]);
    // PEARL_AMD_WMMA12 is set only in a gfx12 device pass; PEARL_AMD_EMU only in HIP-CPU.
    expect(FOLD).toContain('#if defined(__GFX12__)\n#define PEARL_AMD_WMMA12 1\n#endif\n');
    expect(FOLD).toContain('#if defined(__HIP_CPU_RT__)\n#define PEARL_AMD_EMU 1\n');
    // Every line of it is behind that: a gfx11 or host pass with the switch on has the release's code, line for line.
    expect(codeLines(at({ PEARL_AMD_LDS_AHEAD: 1, PEARL_AMD_AHEAD_HERE: 0 })))
      .toEqual(codeLines(at({})));
    expect(codeLines(at({})).filter((l) => AHEAD_NAMES.test(l))).toEqual([]);
    // The switch itself guards only its refusals and PEARL_AMD_AHEAD_HERE, which guards the code.
    expect([...FOLD.matchAll(/^#if PEARL_AMD_LDS_AHEAD\b.*\n(#\w+ \S+)/gm)].map((m) => m[1]))
      .toEqual(['#error "PEARL_AMD_LDS_AHEAD', '#error "PEARL_AMD_LDS_AHEAD', '#define PEARL_AMD_AHEAD_HERE']);
    // The helpers, the ahead loop, its closing brace, the one-slab loop's stop.
    expect(FOLD.match(/^#if PEARL_AMD_AHEAD_HERE$/gm)).toHaveLength(4);
    // In such a pass, only the wmma12 fold: HIP-CPU builds all five, and the others keep the release's loop.
    const t = gfx12(1);
    expect(aheadBlock(t).split('\n')[0]).toBe('  if constexpr (std::is_same<P, PearlAmdWmma12>::value) {');
    expect(t.slice(t.indexOf('\n  } else {\n'), t.indexOf('\n  } else {\n') + 80))
      .toContain('\n  } else {\n  auto slab_step = [&](uint32_t s, auto parity) {');
    // The one-slab loop has none of it, and stops a pass that builds the wmma12 fold there with it on.
    const one = t.slice(t.indexOf('\n#else\n', t.indexOf('  if (s < slabs) slab_step(s, std::integral_constant<int, 0>());\n')));
    expect(one).toContain('static_assert(!std::is_same<P, PearlAmdWmma12>::value,\n'
      + '                "PEARL_AMD_LDS_AHEAD is in the PF2 loop only: add -DPEARL_AMD_PF2=1");');
    expect(codeLines(one.slice(0, one.indexOf('\n#endif\n'))).filter((l) => AHEAD_NAMES.test(l))).toEqual([]);
  });

  it('can ship: no caps bit, and neither the host nor build-amd.sh refuses it', () => {
    expect(BUILD).not.toContain('LDS_AHEAD');
    expect(HOST).not.toContain('LDS_AHEAD');
    const caps = (t) => slice(t, 'extern "C" __global__ void pearl_amd_caps(uint32_t *out) {', '\n}\n');
    expect(caps(gfx12(1))).toBe(caps(gfx12(0)));
    expect(caps(FOLD)).not.toMatch(/AHEAD/);
  });
});

describe('the ahead step', () => {
  it('reads each fragment the release reads, from the release\'s address with the release\'s loader', () => {
    for (const form of [1, 2]) {
      const t = gfx12(form);
      const blk = aheadBlock(t);
      const rel = slabStep(t);
      // The loads: the release's expressions, the buffer and set its own.
      const relA = grab(rel, /\n {6}for \(int mt = 0; mt < P::kMT; mt\+\+\)\n {8}a\[mt\] = (P::load_a\(.*\));/);
      const relB = grab(rel, /\n {6}for \(int nt = 0; nt < P::kNT; nt\+\+\)\n {8}b\[nt\] = (P::load_b\(.*\));/);
      expect(grab(blk, /fa\[Q\]\[kk\]\[mt\] = (P::load_a\(.*\));/)).toBe(relA);
      expect(grab(blk, /fb\[Q\]\[kk\]\[nt\] = (P::load_b\(.*\));/)).toBe(relB);
      expect(grab(blk, /\n {6}(const int8_t \*aBase = .*;)/)).toBe(grab(rel, /\n {4}(const int8_t \*aBase = .*;)/).replace('[B]', '[Q]'));
      expect(grab(blk, /\n {6}(const int8_t \*bBase = .*;)/)).toBe(grab(rel, /\n {4}(const int8_t \*bBase = .*;)/).replace('[B]', '[Q]'));
      // The WMMAs: the release's step function, on whole fragment sets.
      expect(blk.match(/pearl_amd_mma_step<P>\(acc, [^;]*\);/g)).toEqual([
        'pearl_amd_mma_step<P>(acc, fa[B][kk], fb[B][kk]);', 'pearl_amd_mma_step<P>(acc, fa[B][KK - 1], fb[B][KK - 1]);']);
      expect(rel).toContain('      pearl_amd_mma_step<P>(acc, a, b);\n');
      // The next slab's global loads, its LDS store, the slab barrier and the readout: the release's lines.
      const ah = codeLines(aheadStep(t)).map((l) => l.trim());
      const re = codeLines(rel).map((l) => l.trim());
      for (const l of ['const uint32_t n = s + 2 < slabs ? s + 2 : slabs - 1u;', 'PEARL_AMD_LOADR(n, pre2[B])',
        'if (s + 1 < slabs) { PEARL_AMD_STORER(B ^ 1, pre2[B ^ 1]) }   // slab s + 1, loaded a step ago',
        'pearl_amd_lds_barrier();', 'if ((s + 1) % perChunk == 0) readout((s + 1) / perChunk - 1);']) {
        expect([form, l, ah.filter((x) => x === l).length, re.filter((x) => x === l).length]).toEqual([form, l, 1, 1]);
      }
      // And how many k-steps it loads after the barrier.
      expect(blk).toContain('constexpr int NA = PEARL_AMD_LDS_AHEAD == 2 ? KK : 1;');
      expect(blk).toContain('constexpr int KK = KS / P::kTK;');
    }
    // Without PFX, the release's guarded global loads; with stamps, the release's stamp barrier.
    const noPfx = codeLines(aheadStep(gfx12(1, { PEARL_AMD_PFX: 0 }))).map((l) => l.trim());
    expect(noPfx).toContain('if (s + 2 < slabs) { PEARL_AMD_LOADR(s + 2, pre2[B]) }   // set B\'s slab s went to LDS a step ago');
    expect(noPfx).toContain('__syncthreads();');
    const st = codeLines(aheadStep(gfx12(2, { PEARL_AMD_STAMP: 1 }))).filter((l) => /barrier|__syncthreads|fence/.test(l));
    expect(st.map((l) => l.trim())).toEqual(['stamp_barrier();', 'pearl_amd_ahead_fence();']);
  });

  // The ahead loop (or the release's), from the C, as JS: the model below gives it the loads, stores, barriers,
  // WMMAs and readouts as functions.
  function asJs(c) {
    return c.split('\n').filter((l) => !/^\s*(#pragma|\/\/)/.test(l)).join('\n')
      .replace(/\/\/[^\n]*/g, '')
      .replace(/^ {2}if constexpr \(std::is_same<P, PearlAmdWmma12>::value\) \{\n/, '')
      .replace(/\n {2}\} else \{\n$/, '\n')
      .replace(/constexpr int KK = KS \/ P::kTK;/, 'const KK = KSTEPS;')
      .replace(/constexpr int NA = /, 'const NA = ')
      .replace(/typename P::[AB]Frag (f[ab])\[2\]\[KK\]\[P::k[MN]T\];/g,
        'const $1 = [0, 1].map(() => Array.from({ length: KK }, () => []));')
      .replace(/typename P::[AB]Frag ([ab])\[P::k[MN]T\];/g, 'const $1 = [];')
      .replace(/auto (\w+) = \[&\]\(auto q, int kk\) \{/g, 'const $1 = (q, kk) => {')
      .replace(/auto (\w+) = \[&\]\(uint32_t s, auto parity\) \{/g, 'const $1 = (s, parity) => {')
      .replace(/constexpr int (\w) = decltype\((\w+)\)::value;/g, 'const $1 = $2;')
      .replace(/constexpr int /g, 'const ')
      .replace(/const int8_t \*([ab])Base = s[AB]\[(\w+)\] \+ w[rc] \* W[MN] \* S;/g, 'const $1Base = $2;')
      .replace(/P::load_([ab])\(([ab])Base \+ (mt|nt) \* P::kT[MN] \* S \+ kk \* P::kTK, S, lane\)/g, "lds('$1', $2Base, $3, kk)")
      .replace(/KS \/ P::kTK/g, 'KSTEPS')
      .replace(/P::k([MN])T/g, 'k$1T')
      .replace(/std::integral_constant<int, ([^>]+)>\(\)/g, '($1)')
      .replace(/\bint (\w+) = /g, 'let $1 = ')
      .replace(/\buint32_t s = 0;/, 'let s = 0;')
      .replace(/const uint32_t /g, 'const ')
      .replace(/(0x[0-9A-Fa-f]+|\d+)u\b/g, '$1')
      .replace(/PEARL_AMD_LOADR\(([^,]+), pre2\[(\w+)\]\)/g, 'gload($1, $2);')
      .replace(/PEARL_AMD_STORER\(([^,]+), pre2\[([^\]]+)\]\)/g, 'store($1, $2);')
      .replace(/pearl_amd_mma_step<P>\(acc, ([^;]*)\);/g, 'mma($1);')
      .replace(/\b(stamp_barrier|pearl_amd_lds_barrier|__syncthreads)\(\);/g, 'barrier();')
      .replace(/pearl_amd_ahead_fence\(\);/g, '')
      .replace(/(if constexpr \([^)]*\) )?pearl_amd_ahead_pin<[^;]*>\(\);/g, '');
  }

  // Runs one wave's part of a block's slab loop on the model. The block's waves all run the same code between two
  // barriers in any order, so a buffer written and read between the same two barriers is a race; the model flags
  // it, and checks every WMMA's fragments (which slab, which k-step, which tile) and every readout's place.
  function simulate(js, { slabs, perChunk, kSteps, form, kMT = 4, kNT = 4 }) {
    const errs = [];
    const st = {
      epoch: 1,                               // the prologue's __syncthreads() came after slab 0's store
      buf: [0, undefined],                    // the slab in each LDS buffer
      reads: new Set(), writes: new Set(),    // buffers read and written since the last barrier
      pre: [undefined, slabs > 1 ? 1 : undefined],   // the slab in each register set: the prologue loaded slab 1
      pending: [false, slabs > 1],            // loaded and not yet stored
      stored: [0], mmas: 0, readouts: [],
    };
    const env = {
      KSTEPS: kSteps, PEARL_AMD_LDS_AHEAD: form, kMT, kNT, slabs, perChunk,
      lds: (arr, b, idx, kk) => { st.reads.add(b); return { arr, idx, kk, slab: st.buf[b] }; },
      gload: (n, q) => {
        if (st.pending[q]) errs.push(`slab ${n} loaded into set ${q} before its slab ${st.pre[q]} was stored`);
        st.pre[q] = n;
        st.pending[q] = true;
      },
      store: (b, q) => {
        if (b !== q || !st.pending[q]) errs.push(`set ${q} stored to buffer ${b} with nothing new in it`);
        st.writes.add(b);
        st.buf[b] = st.pre[q];
        st.pending[q] = false;
        st.stored.push(st.pre[q]);
      },
      barrier: () => {
        for (const b of st.writes) if (st.reads.has(b)) errs.push(`buffer ${b} written and read between barriers ${st.epoch - 1} and ${st.epoch}`);
        st.reads.clear();
        st.writes.clear();
        st.epoch++;
      },
      mma: (a, b) => {
        const slab = Math.floor(st.mmas / kSteps), kk = st.mmas % kSteps;
        const want = (arr, n) => Array.from({ length: n }, (_, idx) => ({ arr, idx, kk, slab }));
        const got = [...a, ...b].map((f) => (f ? { arr: f.arr, idx: f.idx, kk: f.kk, slab: f.slab } : null));
        if (JSON.stringify(got) !== JSON.stringify(want('a', kMT).concat(want('b', kNT)))) {
          errs.push(`WMMA ${st.mmas} (slab ${slab}, k-step ${kk}) got ${JSON.stringify(got[0])} ...`);
        }
        st.mmas++;
      },
      readout: (c) => {
        if (c !== st.readouts.length || st.mmas !== (c + 1) * perChunk * kSteps) errs.push(`readout ${c} after ${st.mmas} WMMAs`);
        st.readouts.push(c);
      },
    };
    new Function('env', 'const { KSTEPS, PEARL_AMD_LDS_AHEAD, kMT, kNT, slabs, perChunk, lds, gload, store, barrier, mma, '
      + 'readout } = env;\n' + js)(env);
    env.barrier();   // the block's end: what happened since its last barrier counts too
    if (st.mmas !== slabs * kSteps) errs.push(`${st.mmas} WMMAs, not ${slabs * kSteps}`);
    if (st.readouts.length !== Math.floor(slabs / perChunk)) errs.push(`${st.readouts.length} readouts`);
    if (JSON.stringify(st.stored) !== JSON.stringify(Array.from({ length: slabs }, (_, i) => i))) {
      errs.push(`slabs went to LDS in the order ${st.stored.join(' ')}`);
    }
    return errs;
  }
  const CASES = [];
  for (const slabs of [1, 2, 3, 4, 5, 6, 7, 8, 9, 64, 65, 128]) {
    for (const perChunk of [1, 2, 4]) for (const kSteps of [2, 4, 6]) CASES.push({ slabs, perChunk, kSteps });
  }

  it('gives every WMMA the release\'s fragments in the release\'s order, and races on no LDS buffer', () => {
    // The prologue the model starts from: slab 0 stored and synchronised, slab 1 loaded into set 1, then the loop.
    const t = gfx12(1);
    const pro = ['  PEARL_AMD_LOAD(0)\n  PEARL_AMD_STORE(0)\n  __syncthreads();\n', '  if (slabs > 1u) { PEARL_AMD_LOADR(1, pre2[1]) }\n',
      '  if constexpr (std::is_same<P, PearlAmdWmma12>::value) {\n'].map((s) => t.indexOf(s));
    expect(pro.every((x, i) => x > 0 && (i === 0 || x > pro[i - 1]))).toBe(true);
    // Each form, with and without PFX (its global loads differ), at 2 k-steps a slab (RDNA 4's), 4 and 6.
    let runs = 0;
    for (const form of [1, 2]) {
      for (const pfx of [1, 0]) {
        const js = asJs(aheadBlock(gfx12(form, { PEARL_AMD_PFX: pfx })));
        for (const c of CASES) {
          expect([form, pfx, c, simulate(js, Object.assign({ form }, c))]).toEqual([form, pfx, c, []]);
          runs++;
        }
      }
    }
    // The release's loop on the same model: the same WMMAs, the same order.
    const rel = asJs(releaseLoop(gfx12(0)));
    for (const c of CASES) expect([c, simulate(rel, Object.assign({ form: 0 }, c))]).toEqual([c, []]);
    expect(runs).toBe(4 * CASES.length);
  });

  it('would catch a step that reads a slab early, keeps the wrong set, or loses the barrier', () => {
    const t = aheadBlock(gfx12(2));
    const next = '      for (int kk = 0; kk < NA; kk++) frags(std::integral_constant<int, B ^ 1>(), kk);\n';
    const store = '      if (s + 1 < slabs) { PEARL_AMD_STORER(B ^ 1, pre2[B ^ 1]) }   // slab s + 1, loaded a step ago\n';
    const barrier = '      pearl_amd_lds_barrier();\n';
    for (const s of [next, store, barrier]) expect(t).toContain(s);
    const mutants = {
      'the next slab read before the barrier': t.replace(next, '').replace(barrier, next + barrier),
      'the next slab stored after the barrier': t.replace(store, '').replace(barrier, barrier + store),
      'the next slab read into this slab\'s set': t.replace(next, next.replace('B ^ 1', 'B')),
      'the last k-step run on the other set': t.replace('fa[B][KK - 1], fb[B][KK - 1]', 'fa[B ^ 1][KK - 1], fb[B ^ 1][KK - 1]'),
      'no barrier': t.replace(barrier, ''),
    };
    for (const [name, m] of Object.entries(mutants)) {
      expect([name, m === t]).toEqual([name, false]);
      const errs = simulate(asJs(m), { slabs: 8, perChunk: 4, kSteps: 2, form: 2 });
      expect([name, errs.length > 0]).toEqual([name, true]);
    }
  });

  it('keeps the WMMAs after the slab barrier, and puts each LDS read among them', () => {
    for (const form of [1, 2]) {
      const s = codeLines(aheadStep(gfx12(form))).map((l) => l.trim()).filter((l) => l !== '#pragma unroll');
      const i = (l) => s.indexOf(l);
      const order = ['pearl_amd_lds_barrier();', 'pearl_amd_ahead_fence();',
        'for (int kk = 0; kk < NA; kk++) frags(std::integral_constant<int, B ^ 1>(), kk);',
        'pearl_amd_mma_step<P>(acc, fa[B][KK - 1], fb[B][KK - 1]);',
        'pearl_amd_ahead_pin<RA, MM, pearl_amd_gcd(RA, MM)>();   // the reads spread among the WMMAs',
        'if ((s + 1) % perChunk == 0) readout((s + 1) / perChunk - 1);'];
      expect(order.map(i)).toEqual(order.map((l, k) => i(order[0]) + k));
      // With 1, the loads at the step's top go before its first WMMAs, pinned there before the next slab's store:
      // nothing before the slab barrier uses them, and unpinned they could sink to it.
      const top = ['if (kk + 1 >= NA) frags(parity, kk + 1);', 'pearl_amd_mma_step<P>(acc, fa[B][kk], fb[B][kk]);', '}',
        'if constexpr (NA < KK) pearl_amd_ahead_pin<(KK - NA) * FR, (KK - 1) * MM, 1>();',
        'if (s + 1 < slabs) { PEARL_AMD_STORER(B ^ 1, pre2[B ^ 1]) }   // slab s + 1, loaded a step ago'];
      expect(s.slice(i(top[0]), i(top[0]) + top.length)).toEqual(top);
    }
    // A full scheduling boundary, and the pin: kGroups groups of kReads / kGroups LDS reads (mask 0x100) then
    // kMmas / kGroups WMMAs (0x008, the MFMA mask, which takes WMMAs), each in a gfx12 device pass only.
    const fence = slice(FOLD, 'void pearl_amd_ahead_fence() {', '\n}\n');
    expect(fence).toContain('#if PEARL_AMD_WMMA12\n  __builtin_amdgcn_sched_barrier(0);\n#endif');
    const pin = slice(FOLD, 'void pearl_amd_ahead_pin() {', '\n}\n');
    expect(pin).toContain('#if PEARL_AMD_WMMA12\n#pragma unroll\n  for (int i = 0; i < kGroups; i++) {\n'
      + '    __builtin_amdgcn_sched_group_barrier(0x100, kReads / kGroups, 0);   // its LDS reads\n'
      + '    __builtin_amdgcn_sched_group_barrier(0x008, kMmas / kGroups, 0);    // then its WMMAs\n  }\n#endif');
    expect(FOLD).toContain('template <int kReads, int kMmas, int kGroups>\n__device__ __forceinline__ void pearl_amd_ahead_pin() {');
  });

  it('sizes the pins so they compile at any slab the staging takes, and as the compiler merges the reads', () => {
    // The C's arithmetic, as JS: the pin's static_assert, the gcd, the counts and the two pins' arguments.
    const js = (c) => c.replace(/P::k([MN])T/g, 'k$1T').replace(/pearl_amd_gcd/g, 'gcd');
    const pin = slice(FOLD, 'void pearl_amd_ahead_pin() {', '\n}\n');
    const ok = new Function('kReads', 'kMmas', 'kGroups', 'return ' + grab(pin, /static_assert\(([^,]*),\n/) + ';');
    const gcdBody = grab(FOLD, /\nconstexpr int pearl_amd_gcd\(int a, int b\) \{ return (.*); \}\n/);
    const gcd = new Function('a', 'b', 'gcd', 'return ' + gcdBody.replace('pearl_amd_gcd(b, a % b)', 'gcd(b, a % b, gcd)') + ';');
    const g = (a, b) => gcd(a, b, gcd);
    const blk = aheadBlock(gfx12(1));
    const na = grab(blk, /constexpr int NA = (.*);/);
    const [fr, mm] = grab(blk, /constexpr int FR = (.*);/).split(', MM = ');
    const ra = grab(blk, /constexpr int RA = (.*);/);
    const [topIf, topArgs] = /if constexpr \((.*)\) pearl_amd_ahead_pin<(.*)>\(\);/.exec(blk).slice(1);
    const afterArgs = grab(blk, /\n {6}pearl_amd_ahead_pin<(.*)>\(\);/);
    const split = (a) => {
      const out = [];
      let depth = 0, cur = '';
      for (const ch of a) {
        if (ch === '(') depth++;
        if (ch === ')') depth--;
        if (ch === ',' && depth === 0) { out.push(cur.trim()); cur = ''; } else cur += ch;
      }
      return out.concat(cur.trim());
    };
    const pol = slice(FOLD, 'struct PearlAmdWmma12 {', '\n};\n');
    const [kMT, kNT] = grab(pol, /(kMT = \d+, kNT = \d+);/).match(/\d+/g).map(Number);
    const counts = (form, KK) => {
      const v = new Function('PEARL_AMD_LDS_AHEAD', 'KK', 'kMT', 'kNT', 'gcd', `const NA = ${js(na)};
        const FR = ${js(fr)}, MM = ${js(mm)}; const RA = ${js(ra)};
        return { top: (${js(topIf)}) ? [${split(js(topArgs)).join(', ')}] : null, after: [${split(js(afterArgs)).join(', ')}] };`);
      return v(form, KK, kMT, kNT, g);
    };
    // The staging splits evenly at a slab of 32, 64, 96 or 128 bytes: 2, 4, 6 or 8 k-steps (a slab of 16 or 48 doesn't).
    for (const KK of [2, 4, 6, 8]) {
      for (const form of [1, 2]) {
        const c = counts(form, KK);
        for (const p of [c.top, c.after]) if (p) expect([form, KK, p, ok(...p)]).toEqual([form, KK, p, true]);
        expect([form, KK, c.top === null]).toEqual([form, KK, form === 2]);
      }
    }
    // RDNA 4's slab, 2 k-steps of 8 fragments and 16 WMMAs. After the barrier: with 1, 4 reads (8 fragments, two a
    // ds_load_2addr_b64) a group each, 4 WMMAs a read; with 2, 8 reads, 2 WMMAs a read. That 8 is the release's: its
    // gfx1201 fold has 24 LDS loads in its assembly, 8 in each of the three copies of the step. With 1, the step's
    // top: one group of at most 8 reads (merged or not), then the first k-step's 16 WMMAs.
    expect(counts(1, 2)).toEqual({ top: [8, 16, 1], after: [4, 16, 4] });
    expect(counts(2, 2)).toEqual({ top: null, after: [8, 16, 8] });
    expect(counts(2, 2).after[0]).toBe(24 / 3);
    // More k-steps than WMMAs a k-step: groups of more than one read (the first form asserted here).
    expect(counts(2, 8).after).toEqual([32, 16, 16]);
    expect(counts(2, 6).after).toEqual([24, 16, 8]);
  });

  it('with stamps, times the whole loop: the same stamp barrier, and the loop\'s sum runs to its end', () => {
    const t = gfx12(1, { PEARL_AMD_STAMP: 1 });
    expect(codeLines(aheadStep(t)).filter((l) => /stamp_barrier/.test(l))).toEqual(['      stamp_barrier();']);
    expect(codeLines(slabStep(t)).filter((l) => /stamp_barrier/.test(l))).toEqual(['    stamp_barrier();']);
    // The loop's end read comes after both loops, so a last barrier a k-step before the end still counts the rest.
    const end = t.indexOf('  {   // PEARL_AMD_STAMP: the slab loop ends.');
    expect(end).toBeGreaterThan(t.indexOf('  if (s < slabs) slab_step(s, std::integral_constant<int, 0>());\n  }\n'));
    expect(end).toBeGreaterThan(t.indexOf('    if (s < slabs) ahead_step(s, std::integral_constant<int, 0>());'));
  });
});
