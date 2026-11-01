'use strict';

// PEARL_AMD_STAMP (earn/native/amd/src/pearl_amd_fold.h) builds an AMD core whose folds store clock
// stamps, for timing: each wave's shader clock and REALTIME at its block's start and end, the cycles of
// its prologue, slab loop, epilogue, barrier waits and memory waits, and its first barrier exits after the
// counter's bit 16 and bit 19 change. test/stamp-summary.js turns them into item 13's readings. These pin the
// guards that keep a stamp build from mining or shipping, that a release build gets the same code, the
// stamps' arithmetic (run here as JS, out of the C), the record the fold, the fold test and the summary
// share, the summary's math on made-up stamps, and sweep.sh's SWEEP_STAMP. No HIP compiler runs here:
// the C is read as text (helpers/amdSwitches.js), and the shell parts run under bash where there is one.

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const { preprocess } = require('./helpers/amdSwitches');

const NATIVE = path.join(__dirname, '..', 'native');
const AMD = path.join(NATIVE, 'amd');
const SUMMARY = path.join(AMD, 'test', 'stamp-summary.js');
const read = (...f) => fs.readFileSync(path.join(NATIVE, ...f), 'utf8').replace(/\r\n/g, '\n');
const FOLD = read('amd', 'src', 'pearl_amd_fold.h');
const HOST = read('amd', 'src', 'pearl_host_hip.cpp');
const BUILD = read('amd', 'build-amd.sh');
const SWEEP = read('amd', 'test', 'sweep.sh');
const FOLDTEST = read('amd', 'test', 'pearl_amd_foldtest.cpp');
const summary = require(SUMMARY);

const hasBash = spawnSync('bash', ['--version'], { stdio: 'ignore' }).status === 0;
const bashIt = hasBash ? it : it.skip;

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
// The C as a build with these switches leaves it (the others as a release has them).
const at = (text, sw) => preprocess(text, Object.assign({ PEARL_AMD_ABLATE: 0, PEARL_AMD_STAMP: 0 }, sw));
const code = (t) => t.split('\n').filter((l) => !/^\s*\/\//.test(l));
const slabStep = (t) => slice(t, 'auto slab_step = [&](uint32_t s, auto parity) {', '\n  };\n');
const caps = (t) => slice(t, 'extern "C" __global__ void pearl_amd_caps(uint32_t *out) {', '\n}\n');
const body = (t) => slice(t, 'void pearl_amd_fold_body(', '\n}\n');
// The top-level items of a C brace list, comments dropped.
const items = (list) => {
  const out = [];
  let depth = 0, cur = '';
  for (const ch of list.replace(/\/\/[^\n]*/g, '')) {
    if (ch === '(') depth++;
    if (ch === ')') depth--;
    if (ch === ',' && depth === 0) { out.push(cur.trim()); cur = ''; } else cur += ch;
  }
  if (cur.trim()) out.push(cur.trim());
  return out;
};

describe('the clock-stamp switch (PEARL_AMD_STAMP)', () => {
  it('is 0 unless a build sets it, takes 0 or 1, and refuses HIP-CPU', () => {
    expect(FOLD).toContain('#ifndef PEARL_AMD_STAMP\n#define PEARL_AMD_STAMP 0\n#endif\n'
      + '#if PEARL_AMD_STAMP != 0 && PEARL_AMD_STAMP != 1\n#error "PEARL_AMD_STAMP is 0 or 1"\n#endif\n'
      + '#if PEARL_AMD_STAMP && PEARL_AMD_EMU\n#error "PEARL_AMD_STAMP reads a GPU\'s clocks');
    const cond = grab(FOLD, /^#if (PEARL_AMD_STAMP != 0 && PEARL_AMD_STAMP != 1)\n/m);
    const refused = (v) => new Function('PEARL_AMD_STAMP', 'return ' + cond + ';')(v);
    expect([0, 1, 2, -1, 8].map(refused)).toEqual([false, false, true, true, true]);
  });

  it('leaves a release build as it was: at 0 nothing it adds is left', () => {
    // The token check is tokdiff.sh's (cl /EP on every pass); here, what the switch guards.
    const rel = at(FOLD, {});
    const names = /STAMP|pearl_amd_stamps|pearl_amd_cycles|pearl_amd_realtime|pearl_amd_hw_id1|\bst[A-Z]\w*|\bc[01m]\b|\bx1[69]\b|s_getreg|s_sendmsg|s_waitcnt/;
    expect(code(rel).filter((l) => names.test(l))).toEqual([
      '#define PEARL_AMD_CAP_STAMP 0x1000u    // a PEARL_AMD_STAMP build: clock stamps (pearl_amd_stamps); TIMING too',
      '#ifndef PEARL_AMD_STAMP',
      '#define PEARL_AMD_STAMP 0',
      '#if PEARL_AMD_STAMP && PEARL_AMD_EMU',
      '#error "PEARL_AMD_STAMP reads a GPU\'s clocks: build it for a card (build-amd.sh gpu), not HIP-CPU"',
    ]);
    // The slab barrier is the release's: the LDS-only barrier under PFX, else __syncthreads().
    expect(slabStep(rel).endsWith('    if (s + 1 < slabs) { PEARL_AMD_STORER(B ^ 1, pre2[B ^ 1]) }   // slab s + 1, loaded a step ago\n'
      + '#if PEARL_AMD_PFX\n    pearl_amd_lds_barrier();\n#else\n    __syncthreads();\n#endif\n  };\n')).toBe(true);
    expect(rel).toContain('  if (slabs > 1u) { PEARL_AMD_LOADR(1, pre2[1]) }\n  auto slab_step');
    expect(rel).toContain('        w[r + 8] = keepHi[rb][cb][q];\n      }\n#undef PEARL_AMD_LOAD\n');
    expect(caps(rel)).not.toContain('CAP_STAMP');
    // The host and the fold test: nothing in the host; the fold test only reads its argument and refuses it.
    expect(code(at(HOST, {})).filter((l) => /PEARL_AMD_STAMP|CAP_STAMP|pearl_amd_stamps|dStamps|->stamp\b/.test(l))).toEqual([]);
    expect(code(at(FOLDTEST, {})).filter((l) => /stamp|STAMP/i.test(l))).toEqual([
      '  const char *stampPath = argc > 9 && argv[9][0] ? argv[9] : nullptr;',
      '  if (stampPath && !(caps[0] & PEARL_AMD_CAP_STAMP)) {',
      '    fprintf(stderr, "stamps: this build has no clock stamps; build it with -DPEARL_AMD_STAMP=1\\n");',
    ]);
  });

  it('marks every code object it is in with a caps bit of its own, and as timing only', () => {
    const bits = [...FOLD.matchAll(/#define PEARL_AMD_CAP_(\w+) (0x[0-9a-f]+|\d+)u/g)].map((m) => [m[1], Number(m[2])]);
    const v = Object.fromEntries(bits);
    expect([v.STAMP, v.TIMING]).toEqual([0x1000, 0x800]);
    for (const [n, x] of bits) if (n !== 'STAMP') expect([n, x & v.STAMP]).toEqual([n, 0]);
    // After the per-target bits, so every target says so, one with no matrix fold too.
    const c = caps(at(FOLD, { PEARL_AMD_STAMP: 1 }));
    expect(c).toContain('  if (PEARL_AMD_KWRAP) out[0] |= PEARL_AMD_CAP_KWRAP;\n'
      + '  out[0] |= PEARL_AMD_CAP_TIMING | PEARL_AMD_CAP_STAMP;\n  out[1] = (uint32_t)warpSize;');
    expect(c.indexOf('CAP_STAMP')).toBeGreaterThan(c.lastIndexOf('#endif'));
    expect(caps(at(FOLD, { PEARL_AMD_STAMP: 1, PEARL_AMD_ABLATE: 8 }))).toContain('  out[0] |= PEARL_AMD_CAP_TIMING;\n'
      + '  out[0] |= PEARL_AMD_CAP_TIMING | PEARL_AMD_CAP_STAMP;\n');
  });

  it('stops a pass that builds an RDNA fold in the one-slab loop, which has no stamps', () => {
    const one = (t) => slice(t, '  if (s < slabs) slab_step(s, std::integral_constant<int, 0>());\n#else\n', '\n#endif\n');
    const stop = 'static_assert(!std::is_same<P, PearlAmdWmma11>::value && !std::is_same<P, PearlAmdWmma12>::value,\n'
      + '                "PEARL_AMD_STAMP is in the PF2 loop only: add -DPEARL_AMD_PF2=1");';
    expect(one(at(FOLD, { PEARL_AMD_STAMP: 1 }))).toContain(stop);
    expect(one(at(FOLD, {}))).not.toContain('static_assert');
  });
});

describe('the stamps in the fold', () => {
  // A device pass, as far as these switches go: gfx11 has no PFX by default, gfx12 has it.
  const pass = (sw) => at(FOLD, Object.assign({ PEARL_AMD_STAMP: 1, PEARL_AMD_PFX: 0, PEARL_AMD_WMMA11: 0,
    PEARL_AMD_WMMA12: 0 }, sw));
  const gfx11 = { PEARL_AMD_WMMA11: 1 }, gfx12 = { PEARL_AMD_WMMA12: 1, PEARL_AMD_PFX: 1 };

  it('reads the counters the ISA names, with gfx11\'s 20 bits masked', () => {
    // s_getreg's operand: the hwreg's number in bits 5:0, the first bit 10:6, the width less one 15:11.
    const hwreg = (x) => ({ id: x & 63, offset: (x >> 6) & 31, size: (x >> 11) + 1 });
    expect(hwreg(0x981D)).toEqual({ id: 29, offset: 0, size: 20 });   // SHADER_CYCLES, gfx11
    expect(hwreg(0xF81D)).toEqual({ id: 29, offset: 0, size: 32 });   // SHADER_CYCLES_LO, gfx12
    expect(hwreg(0xF817)).toEqual({ id: 23, offset: 0, size: 32 });   // HW_ID1, both
    const cyc = (sw) => slice(pass(sw), 'uint32_t pearl_amd_cycles() {', '\n}\n');
    expect(cyc(gfx11)).toContain('  return (uint32_t)__builtin_amdgcn_s_getreg(0x981D) & 0xFFFFFu;\n}');
    expect(cyc(gfx12)).toContain('  return (uint32_t)__builtin_amdgcn_s_getreg(0xF81D);\n}');
    expect(cyc({})).toContain('  return 0u;\n}');
    expect(0xFFFFF).toBe(2 ** 20 - 1);
    // REALTIME: s_sendmsg_rtn_b64 with MSG_RTN_GET_REALTIME, and HW_ID1, on both.
    for (const sw of [gfx11, gfx12]) {
      expect(slice(pass(sw), 'uint32_t pearl_amd_realtime() {', '\n}\n')).toContain('return (uint32_t)__builtin_amdgcn_s_sendmsg_rtnl(0x83);');
      expect(slice(pass(sw), 'uint32_t pearl_amd_hw_id1() {', '\n}\n')).toContain('return (uint32_t)__builtin_amdgcn_s_getreg(0xF817);');
    }
    expect(FOLD).toContain('#define PEARL_AMD_STAMP_BITS (PEARL_AMD_WMMA11 ? 20u : PEARL_AMD_WMMA12 ? 32u : 0u)');
  });

  it('reads only the shader clock in the slab loop, before the release fence and inside the pair, and stores nothing there', () => {
    // The barrier's steps, in order.
    const steps = (sw) => code(slabStep(pass(sw))).map((l) => {
      let m;
      if ((m = /__builtin_amdgcn_fence\(__ATOMIC_(\w+), "workgroup"(, "local")?\)/.exec(l))) {
        return m[1].toLowerCase() + (m[2] ? ' lds' : ' all');
      }
      if (/__builtin_amdgcn_s_waitcnt\(0\);/.test(l)) return 'waitcnt';
      if (/pearl_amd_cycles\(\)/.test(l)) return 'cycles';
      if (/__builtin_amdgcn_s_barrier\(\);/.test(l)) return 'barrier';
      if (/__syncthreads|pearl_amd_lds_barrier/.test(l)) return 'other barrier';
      return null;
    }).filter(Boolean);
    // A read before the release fence, so its wait (the memory wait) has a number of its own. gfx11 without
    // PFX: __syncthreads()'s pair for every address space, its wait made explicit before the second read.
    // gfx12 with PFX: the LDS-only pair, as pearl_amd_lds_barrier, and no extra wait.
    expect(steps(gfx11)).toEqual(['cycles', 'release all', 'waitcnt', 'cycles', 'barrier', 'cycles', 'acquire all']);
    expect(steps(gfx12)).toEqual(['cycles', 'release lds', 'cycles', 'barrier', 'cycles', 'acquire lds']);
    expect(steps({ PEARL_AMD_WMMA11: 1, PEARL_AMD_PFX: 1 }))
      .toEqual(['cycles', 'release lds', 'cycles', 'barrier', 'cycles', 'acquire lds']);
    expect(steps({ PEARL_AMD_WMMA12: 1 })).toEqual(['cycles', 'release all', 'cycles', 'barrier', 'cycles', 'acquire all']);
    // Bit 2 of the ablation drops only the s_barrier.
    expect(steps(Object.assign({ PEARL_AMD_ABLATE: 2 }, gfx11)))
      .toEqual(['cycles', 'release all', 'waitcnt', 'cycles', 'cycles', 'acquire all']);
    expect(steps(Object.assign({ PEARL_AMD_ABLATE: 2 }, gfx12))).toEqual(['cycles', 'release lds', 'cycles', 'cycles', 'acquire lds']);
    // The first read is the next line of code after the next slab's LDS store, so the store's wait is counted
    // there too, and the bit-8 read (before the store) has its wait in the fence's.
    for (const sw of [gfx11, gfx12, Object.assign({ PEARL_AMD_ABLATE: 8 }, gfx11)]) {
      const t = slabStep(pass(sw));
      expect(code(t.slice(t.indexOf('    if (s + 1 < slabs) { PEARL_AMD_STORER(B ^ 1'))).slice(0, 2)).toEqual([
        '    if (s + 1 < slabs) { PEARL_AMD_STORER(B ^ 1, pre2[B ^ 1]) }   // slab s + 1, loaded a step ago',
        '    const uint32_t cm = pearl_amd_cycles();']);
    }
    // The pairs are the release's barriers' (amdAblate.test.js pins pearl_amd_lds_barrier's).
    expect(FOLD).toContain('__syncthreads() is a workgroup fence for every address space');
    // Nothing else: no REALTIME (its wait would be an lgkmcnt wait on gfx11, which counts the LDS loads too),
    // no HW_ID1 and no store.
    for (const sw of [gfx11, gfx12]) {
      const s = code(slabStep(pass(sw))).join('\n');
      expect(s).not.toMatch(/pearl_amd_realtime|pearl_amd_hw_id1|pearl_amd_stamps|\bout\b/);
      expect(s.match(/pearl_amd_cycles\(\)/g)).toHaveLength(3);
    }
  });

  it('adds up the slab loop and finds the crossings wrap-safe, run as JS out of the C', () => {
    const sinceC = grab(slice(FOLD, 'uint32_t pearl_amd_cycles_since(uint32_t a, uint32_t b) {', '\n}\n'), /return (.*);/);
    const sinceJs = (w11) => new Function('a', 'b', 'return (' + sinceC.replace('PEARL_AMD_WMMA11', String(w11))
      .replace(/(0x[0-9A-Fa-f]+|\d+)u\b/g, '$1') + ') >>> 0;');
    const s = slabStep(pass(gfx11));
    const upd = grab(s, /\n( {4}stMem \+= [\s\S]*? {4}stLast = c1;)\n/);
    const js = upd.replace(/const bool /g, 'const ').replace(/pearl_amd_cycles_since\(/g, 'since(')
      .replace(/(0x[0-9A-Fa-f]+|\d+)u\b/g, '$1').replace(/>>/g, '>>>');
    const run = (bits) => new Function('st', 'cm', 'c0', 'c1', 'since',
      'let { stMem, stBar, stLoop, stX16, stX19, stSeen, stLast } = st;\n' + js
      + '\nreturn { stMem: stMem >>> 0, stBar: stBar >>> 0, stLoop: stLoop >>> 0, stX16, stX19, stSeen, stLast };');
    expect(grab(body(pass(gfx11)), /\n {2}(stPro = .*;)\n/)).toBe('stPro = pearl_amd_cycles_since(stCyc0, stLast);');
    for (const [bits, w11] of [[20, 1], [32, 0]]) {
      const mask = 2 ** bits;
      const cyc = (t) => t % mask;   // the counter at absolute cycle t
      const since = sinceJs(w11), step = run(bits);
      // A block whose counter starts near the top, 128 slabs of 9000 cycles with 120 at the release fence and
      // 300 at the barrier: more than 2^20 cycles in all, so gfx11's counter wraps on the way.
      const t0 = mask - 70000, pro = 4000, P = 9000, mem = 120, wait = 300, slabs = 128, epi = 2500;
      let st = { stMem: 0, stBar: 0, stLoop: 0, stX16: 0, stX19: 0, stSeen: 0, stLast: cyc(t0 + pro) };
      const stPro = since(cyc(t0), st.stLast);
      let first16 = null, first19 = null, prev = t0 + pro;
      for (let i = 1; i <= slabs; i++) {
        const t1 = t0 + pro + i * P;
        if (first16 === null && Math.floor(t1 / 2 ** 16) !== Math.floor(prev / 2 ** 16)) first16 = t1;
        if (first19 === null && Math.floor(t1 / 2 ** 19) !== Math.floor(prev / 2 ** 19)) first19 = t1;
        st = step(st, cyc(t1 - wait - mem), cyc(t1 - wait), cyc(t1), since);
        prev = t1;
      }
      const stEpi = since(st.stLast, cyc(prev + epi));
      expect([bits, stPro, st.stLoop, st.stBar, st.stMem, stEpi])
        .toEqual([bits, pro, slabs * P, slabs * wait, slabs * mem, epi]);
      // The block's whole length from the three sums, where its own delta has wrapped on 20 bits.
      const total = stPro + st.stLoop + stEpi;
      expect(total).toBe(pro + slabs * P + epi);
      if (bits === 20) expect(since(cyc(t0), cyc(t0 + total))).not.toBe(total);
      // The first barrier exits after bits 16 and 19 change, both seen.
      expect([st.stX16, st.stX19, st.stSeen]).toEqual([cyc(first16), cyc(first19), 3]);
      // The summary's arithmetic is the fold's.
      for (const [a, b] of [[0, 5], [mask - 3, 2], [123456, 99], [mask - 1, 0]]) {
        expect([bits, a, b, summary.since(a, b, bits)]).toEqual([bits, a, b, since(a, b)]);
      }
    }
  });

  it('reads REALTIME and HW_ID1 only at a block\'s start and end, each next to the shader clock', () => {
    const b = body(pass(gfx12));
    expect(b.match(/pearl_amd_realtime\(\)/g)).toHaveLength(2);
    expect(b.match(/pearl_amd_hw_id1\(\)/g)).toHaveLength(2);
    expect(b).toContain('  const uint32_t stCyc0 = pearl_amd_cycles();\n  const uint32_t stRt0 = pearl_amd_realtime();\n');
    expect(b).toContain('    const uint32_t cyc = pearl_amd_cycles();\n    const uint32_t rt = pearl_amd_realtime();\n');
    // The start comes before anything else in the block; the end after the transcript words are stored.
    expect(b.indexOf('stRt0 = pearl_amd_realtime()')).toBeLessThan(b.indexOf('const int t = (int)threadIdx.x'));
    expect(b.indexOf('rt = pearl_amd_realtime()')).toBeGreaterThan(b.indexOf('w[r + 8] = keepHi[rb][cb][q];'));
    // The loop's start: after the second slab's loads, before the first step.
    expect(b).toContain('  if (slabs > 1u) { PEARL_AMD_LOADR(1, pre2[1]) }\n  stLast = pearl_amd_cycles();');
  });

  it('stores one record a wave, from its lane 0, in the order stamp-summary.js reads', () => {
    const b = body(pass(gfx11));
    const v = items(grab(b, /const uint32_t v\[PEARL_AMD_STAMP_WORDS\] = \{([\s\S]*?)\};/));
    expect(v).toEqual(['0x53540002u', '(uint32_t)wave | (PEARL_AMD_STAMP_BITS << 8) | (stSeen << 14) | (block << 16)', 'stHw',
      'hw', 'stCyc0', 'stRt0', 'cyc', 'rt', 'stPro', 'stLoop', 'pearl_amd_cycles_since(stLast, cyc)', 'stBar', 'stX16',
      'stX19', 'slabs', 'stMem']);
    expect(FOLD).toContain('#define PEARL_AMD_STAMP_WORDS 16u');
    expect(b).toContain('    if (out && lane == 0) {\n      const uint32_t block = blockIdx.y * gridDim.x + blockIdx.x;');
    expect(b).toContain('uint32_t *o = out + ((size_t)block * (uint32_t)(T / W) + (uint32_t)wave) * PEARL_AMD_STAMP_WORDS;');
    // Word 1's fields fit their bits: 8 waves of a block, 20 or 32 bits, 2 crossing flags.
    expect([8 < 2 ** 8, 32 < 2 ** 6, 3 < 2 ** 2]).toEqual([true, true, true]);
    // The summary reads those words as those fields; word 1's top half is the block's low 16 bits.
    const words = Array.from({ length: 16 }, (_, i) => 1000 + i);
    words[0] = 0x53540002;
    const block = 0x10001;
    words[1] = 3 | (32 << 8) | (2 << 14) | ((block & 0xffff) << 16);
    const f = file([], { blocks: block + 1, waves: 8 });
    words.forEach((w, i) => f.writeUInt32LE(w >>> 0, 4 * (16 + (block * 8 + 3) * 16 + i)));
    const r = summary.load(f).waves[0];
    expect(r).toEqual({ wave: 3, bits: 32, seen: 2, block, hw0: 1002, hw1: 1003, cyc0: 1004, rt0: 1005, cyc1: 1006,
      rt1: 1007, pro: 1008, loop: 1009, epi: 1010, bar: 1011, x16: 1012, x19: 1013, slabs: 1014, mem: 1015 });
    // The fold test's file header, as the summary reads it.
    const head = items(grab(FOLDTEST, /const uint32_t head\[16\] = \{([\s\S]*?)\};/));
    expect(head).toEqual(['0x31545350u', 'PEARL_AMD_STAMP_WORDS', 'PEARL_AMD_THREADS / 32u', '(uint32_t)stampBlocks',
      'cols / PEARL_AMD_BN', '(uint32_t)wallKhz', 'k', '(uint32_t)n', 'm', 'cols', '0', '0', '0', '0', '0', '0']);
    expect(Buffer.from('PST1').readUInt32LE(0)).toBe(0x31545350);
    const h = summary.load(file([], { blocks: 3, waves: 8, gridX: 5, wallKhz: 99000, k: 2048, launches: 7, m: 384,
      cols: 1280 })).head;
    expect(h).toEqual({ words: 16, waves: 8, blocks: 3, gridX: 5, wallKhz: 99000, k: 2048, launches: 7, m: 384, cols: 1280 });
  });
});

describe('the fold test\'s stamps argument', () => {
  it('needs a stamp build, and stamps only the timed launches of the matrix folds', () => {
    expect(FOLDTEST).toContain('[int8|miner] [stamps file]');
    const main = FOLDTEST.slice(FOLDTEST.indexOf('int main('));
    const refuse = main.indexOf('if (stampPath && !(caps[0] & PEARL_AMD_CAP_STAMP)) {');
    expect(refuse).toBeGreaterThan(main.indexOf('hipLaunchKernelGGL(pearl_amd_caps'));
    expect(refuse).toBeLessThan(main.indexOf('std::vector<int8_t> A('));
    // Room for every wave of a launch: the grid the matrix folds launch, 8 waves a block of 256 at wave32.
    expect(main).toContain('const size_t stampBlocks = (size_t)(cols / PEARL_AMD_BN) * (m / PEARL_AMD_BM);');
    expect(main).toContain('const dim3 grid = f.tall ? dim3(cols / PEARL_AMD_BN, m / PEARL_AMD_BM)');
    expect(main).toContain('const size_t stampWords = stampBlocks * (PEARL_AMD_THREADS / 32u) * PEARL_AMD_STAMP_WORDS;');
    expect(main).toContain('hipDeviceGetAttribute(&wallKhz, hipDeviceAttributeWallClockRate, 0)');
    // Set after the check launch, cleared after the timed ones, then the last launch's stamps are written.
    const set = main.indexOf('hipMemcpyToSymbol(HIP_SYMBOL(pearl_amd_stamps), &stampTo, sizeof stampTo)');
    const timed = main.indexOf('for (int i = 0; i < n; i++) launch(');
    const cleared = main.indexOf('hipMemcpyToSymbol(HIP_SYMBOL(pearl_amd_stamps), &none, sizeof none);');
    const written = main.indexOf('write_stamps(stampPath, dStamps, stampWords, head)');
    expect(main.indexOf('launch(f, grid, fa, fb, k, rank, col0, rowsValid, dTr);')).toBeLessThan(set);
    expect([set < timed, timed < cleared, cleared < written]).toEqual([true, true, true]);
    expect(main).toContain('uint32_t *stampTo = f.tall ? dStamps : nullptr;');
    expect(main).toContain('hipMemset(stampTo, 0, stampWords * 4);');
  });
});

describe('the AMD host with a stamp build', () => {
  const host = at(HOST, { PEARL_AMD_STAMP: 1 });
  const submit = host.slice(host.indexOf('extern "C" bool pearl_host_submit('));

  it('reads the cap, gives the folds room for a batch\'s stamps, and takes it back before freeing it', () => {
    expect(slice(host, 'void resolve_fold(Ctx *ctx) {', '\n}\n')).toContain('  ctx->stamp = (caps[0] & PEARL_AMD_CAP_STAMP) != 0;');
    const create = slice(host, 'extern "C" void *pearl_host_create(', '\n}\n');
    const room = create.indexOf('  if (ctx->stamp) {');
    expect(room).toBeGreaterThan(create.indexOf('  if (ctx->fold < 0) {'));
    expect(create).toContain('const size_t blocks = (size_t)(ctx->colBatch * PEARL_COLS_COUNT / f.bn) * (profile->m / f.bm);');
    expect(create).toContain('HIP_OK(hipMalloc(&ctx->dStamps, blocks * (PEARL_AMD_THREADS / 32u) * PEARL_AMD_STAMP_WORDS * sizeof(uint32_t)),');
    expect(create).toContain('HIP_OK(hipMemcpyToSymbol(HIP_SYMBOL(pearl_amd_stamps), &ctx->dStamps, sizeof ctx->dStamps),');
    // The same grid the fold is launched with.
    expect(submit).toContain('const uint32_t batchCols = ctx->colBatch * PEARL_COLS_COUNT;');
    expect(submit).toContain('const dim3 grid(batchCols / f.bn, ctx->profile.m / f.bm), block(PEARL_AMD_THREADS);');
    const destroy = slice(host, 'extern "C" void pearl_host_destroy(void *handle) {', '\n}\n');
    const unset = destroy.indexOf('hipMemcpyToSymbol(HIP_SYMBOL(pearl_amd_stamps), &none, sizeof none);');
    expect(unset).toBeGreaterThan(destroy.indexOf('hipDeviceSynchronize();'));
    expect(unset).toBeLessThan(destroy.indexOf('hipFree(ctx->dStamps);'));
  });

  it('refuses every target but zero before any fold is launched', () => {
    const start = submit.indexOf('  if (ctx->stamp) {');
    expect(start).toBeGreaterThan(0);
    expect(start).toBeLessThan(submit.indexOf('if (ctx->readoutDrop || ctx->timing) {'));
    expect(start).toBeLessThan(submit.indexOf('hipLaunchKernelGGL(pearl_amd_fold_'));
    const js = submit.slice(start, submit.indexOf('\n  }\n', start) + 4)
      .replace(/ctx->/g, 'ctx.').replace(/\b(bool|int)\b/g, 'let').replace(/"\s*\n\s*"/g, '')
      .replace(/PEARL_HASH_BYTES/g, '32');
    const guard = (ctx) => {
      let msg = null;
      const ok = new Function('ctx', 'err', 'err_len', 'snprintf', js + '\nreturn true;')(ctx, 'buf', 512,
        (e, n, text) => { msg = text; });
      return { ok, msg };
    };
    const target = (byte) => Array.from({ length: 32 }, (_, i) => (i === byte ? 1 : 0));
    const refusal = 'clock-stamp build is timing-only: it runs only at target 0 (hashrate.js), never on a job; '
      + 'build without PEARL_AMD_STAMP to mine';
    expect(guard({ stamp: true, target: target(-1) })).toEqual({ ok: true, msg: null });
    expect(guard({ stamp: true, target: target(0) })).toEqual({ ok: false, msg: refusal });
    expect(guard({ stamp: true, target: target(31) })).toEqual({ ok: false, msg: refusal });
    expect(guard({ stamp: false, target: target(5) })).toEqual({ ok: true, msg: null });
    // And it is a timing build, so the readout-drop and ablation rule below holds it too.
    expect(caps(at(FOLD, { PEARL_AMD_STAMP: 1 }))).toContain('PEARL_AMD_CAP_TIMING | PEARL_AMD_CAP_STAMP');
  });

  it('mines the k a release would (the cap that picks it is untouched)', () => {
    expect(slice(HOST, 'extern "C" uint32_t pearl_host_card_k(', '\n}\n')).not.toMatch(/STAMP|TIMING/);
  });
});

describe('build-amd.sh and a stamp build', () => {
  // Both refusal blocks, from the first case on PEARL_AMD_CFLAGS to the end of the second.
  const first = BUILD.indexOf('case "${PEARL_AMD_CFLAGS:-}" in');
  const second = BUILD.indexOf('case "${PEARL_AMD_CFLAGS:-}" in', first + 1);
  const guards = BUILD.slice(first, BUILD.indexOf('\nesac\n', second) + 6);
  const tryGuard = (mode, flags, ci) => spawnSync('bash', ['-c', 'mode=' + mode + '\n' + guards + 'echo allowed'],
    { env: Object.assign({ PATH: process.env.PATH, PEARL_AMD_CFLAGS: flags }, ci || {}), encoding: 'utf8' });

  bashIt('refuses it in the builds that ship, and allows it on a test box', () => {
    expect(second).toBeGreaterThan(first);
    for (const [mode, flags, ci, ok] of [
      ['winlib', '-DPEARL_AMD_STAMP=1', null, false],
      ['winlib', '-DPEARL_AMD_STAMP=0', null, false],
      ['gpu', '-O3 -DPEARL_AMD_STAMP=1', { CI: 'true' }, false],
      ['gpu', '-DPEARL_AMD_STAMP=1', { GITHUB_ACTIONS: 'true' }, false],
      ['gpu', '-DPEARL_AMD_STAMP=1', null, true],                          // sweep.sh on a Salad box
      ['gpu', '-DPEARL_AMD_ABLATE=8 -DPEARL_AMD_STAMP=1', null, true],
      ['check', '-DPEARL_AMD_STAMP=1', { CI: 'true' }, true],             // compiles only
      ['cpu', '-DPEARL_AMD_STAMP=1', null, true],                          // cpu takes no PEARL_AMD_CFLAGS
    ]) {
      const r = tryGuard(mode, flags, ci);
      expect([mode, flags, ci, r.stdout.trim(), r.status]).toEqual([mode, flags, ci, ok ? 'allowed' : '', ok ? 0 : 1]);
      if (!ok) expect(r.stderr).toContain('clock-stamp build is timing-only: PEARL_AMD_STAMP is refused in a ' + mode);
    }
    // With an ablation as well, the ablation's refusal says so first.
    const both = tryGuard('winlib', '-DPEARL_AMD_ABLATE=2 -DPEARL_AMD_STAMP=1');
    expect([both.status, both.stdout]).toEqual([1, '']);
    expect(both.stderr).toContain('PEARL_AMD_ABLATE is refused');
  });

  bashIt('stops before it builds anything', () => {
    const r = spawnSync('bash', [path.join(AMD, 'build-amd.sh'), 'winlib'], {
      env: Object.assign({}, process.env, { PEARL_AMD_CFLAGS: '-DPEARL_AMD_STAMP=1' }), encoding: 'utf8',
    });
    expect([r.status, r.stdout]).toEqual([1, '']);
    expect(r.stderr).toContain('clock-stamp build is timing-only');
  });
});

// A stamps file as pearl_amd_foldtest writes it, from wave records (the fold's 16 words, by name).
function file(waves, head) {
  const h = Object.assign({ waves: 8, gridX: 1, wallKhz: 100000, k: 4096, launches: 300, m: 131072, cols: 4096 }, head);
  if (h.blocks === undefined) h.blocks = waves.reduce((m, w) => Math.max(m, w.block + 1), 0);
  const buf = Buffer.alloc(4 * (16 + h.blocks * h.waves * 16));
  const put = (i, v) => buf.writeUInt32LE(v >>> 0, 4 * i);
  [0x31545350, 16, h.waves, h.blocks, h.gridX, h.wallKhz, h.k, h.launches, h.m, h.cols].forEach((v, i) => put(i, v));
  for (const w of waves) {
    [0x53540002, w.wave | (w.bits << 8) | (w.seen << 14) | ((w.block & 0xffff) << 16), w.hw0, w.hw1, w.cyc0, w.rt0,
      w.cyc1, w.rt1, w.pro, w.loop, w.epi, w.bar, w.x16, w.x19, w.slabs, w.mem || 0]
      .forEach((v, i) => put(16 + (w.block * h.waves + w.wave) * 16 + i, v));
  }
  return buf;
}
// HW_ID1 for a wave on SIMD `simd` of WGP `wgp` in shader array `sa` of engine `se`.
const hwId = (se, sa, wgp, simd, wave = 0) => ((se << 18) | (sa << 16) | (wgp << 10) | (simd << 8) | wave) >>> 0;
// A wave that ran `ticks` REALTIME ticks from `start` at `mhz` (REALTIME at 100 MHz), its SIMD's counter at
// `cyc0` when it started; by default wave w of its block sits on SIMD w % 4.
function wave(o) {
  const bits = o.bits === undefined ? 20 : o.bits;
  const w = o.wave || 0;
  const cycles = Math.round((o.ticks * (o.mhz || 2500)) / 100);
  const pro = o.pro === undefined ? 2000 : o.pro, epi = o.epi === undefined ? 1000 : o.epi;
  const cyc0 = (o.cyc0 || 0) % 2 ** bits;
  const hw0 = hwId(o.se || 0, o.sa || 0, o.wgp || 0, o.simd === undefined ? w % 4 : o.simd, w);
  return {
    block: o.block, wave: w, bits, seen: o.seen || 0, hw0, hw1: o.hw1 === undefined ? hw0 : o.hw1, cyc0,
    rt0: o.start >>> 0, cyc1: (cyc0 + cycles) % 2 ** bits, rt1: (o.start + o.ticks) >>> 0, pro, loop: cycles - pro - epi,
    epi, bar: o.bar || 0, x16: o.x16 || 0, x19: o.x19 || 0, slabs: o.slabs || 128, mem: o.mem || 0,
  };
}
const summarize = (waves, head) => summary.summarize(summary.load(file(waves, head)));

describe('stamp-summary.js', () => {
  it('reads the clock, the slab and the shares by launch position, from each block\'s wave 0', () => {
    const ws = [
      // The first wave: three blocks that start before any block ends (at tick 10900).
      wave({ block: 0, start: 900, ticks: 10000, mhz: 2400, bar: 12800, mem: 5120, pro: 3000, epi: 1500 }),
      wave({ block: 1, start: 905, ticks: 10000, mhz: 2500, bar: 12800, mem: 5120, pro: 3000, epi: 1500 }),
      wave({ block: 2, start: 910, ticks: 10000, mhz: 2600, bar: 12800, mem: 5120, pro: 3000, epi: 1500 }),
      // The rest, at 2500 MHz: one starts as the first block ends, and REALTIME's low word wraps under one.
      wave({ block: 3, start: 10900, ticks: 10000, bar: 6400, mem: 1280 }),
      wave({ block: 4, start: 2 ** 32 - 50, ticks: 10000, bar: 6400, mem: 1280 }),
    ];
    // REALTIME's low word: block 4 started 50 ticks before it wrapped, so last, after the others.
    for (const w of ws.slice(0, 4)) {
      w.rt0 = (w.rt0 + 2 ** 32 - 20000) >>> 0;
      w.rt1 = (w.rt1 + 2 ** 32 - 20000) >>> 0;
    }
    const s = summarize(ws);
    expect([s.blocks, s.first.n, s.rest.n]).toEqual([5, 3, 2]);
    expect([s.first.mhz.p10, s.first.mhz.med, s.first.mhz.p90]).toEqual([2400, 2500, 2600]);
    expect(s.rest.mhz.med).toBe(2500);
    // Slab: the loop over 128 slabs; barrier: its share of the loop, and a slab's.
    expect(s.rest.slab.med).toBe((250000 - 3000) / 128);
    expect(s.rest.barSlab.med).toBe(50);
    expect(s.rest.bar).toBeCloseTo(6400 / 247000, 12);
    // Memory wait the same way, and the work a slab is the rest of it.
    expect(s.rest.memSlab.med).toBe(10);
    expect(s.rest.mem).toBeCloseTo(1280 / 247000, 12);
    expect(s.rest.work.med).toBe((247000 - 6400 - 1280) / 128);
    expect(s.first.pro).toBeCloseTo(9000 / 750000, 12);
    expect(s.first.epi).toBeCloseTo(4500 / 750000, 12);
    // REALTIME's rate from the file; 100 MHz when the card said nothing.
    expect(summarize(ws, { wallKhz: 50000 }).rest.mhz.med).toBe(1250);
    expect(summarize(ws, { wallKhz: 0 }).tickMhz).toBe(100);
    const lines = summary.format(s);
    expect(lines[1]).toBe('first: 3 blocks, 2500 MHz (p10 2400, p90 2600); slab 1918 cycles (p90 1996); barrier '
      + '5.21% of the loop, 100 a slab; memory wait 2.09%, 40 a slab; prologue 1.20%, epilogue 0.60%');
    expect(lines.every((l) => l.length <= 300)).toBe(true);
  });

  it('drops waves that moved or have no stamps, and says what it read', () => {
    const ws = [
      wave({ block: 0, start: 0, ticks: 1000 }),
      wave({ block: 1, start: 0, ticks: 1000, hw1: 7 }),           // HW_ID1 changed: a context switch
      wave({ block: 2, start: 0, ticks: 1000, bits: 0 }),          // a fold with no stamps on this card
    ];
    const f = file(ws, { blocks: 4 });
    f.writeUInt32LE(0x53540002, 4 * (16 + (3 * 8 + 1) * 16));
    f.writeUInt32LE((1 | (9 << 16)) >>> 0, 4 * (16 + (3 * 8 + 1) * 16 + 1));   // block 3's wave 1 claims block 9
    f.writeUInt32LE(0x53540002, 4 * (16 + (3 * 8 + 2) * 16));
    f.writeUInt32LE((1 | (3 << 16)) >>> 0, 4 * (16 + (3 * 8 + 2) * 16 + 1));   // block 3's wave 2 claims to be wave 1
    f.writeUInt32LE(0x53540001, 4 * (16 + (3 * 8 + 3) * 16));                // version 1: not read
    const d = summary.load(f);
    expect([d.waves.length, d.missing, d.misplaced]).toEqual([3, 4 * 8 - 5, 2]);
    const s = summary.summarize(d);
    expect([s.blocks, s.moved, s.unstamped, s.bits]).toEqual([1, 1, 1, 20]);
    expect(summary.format(s)[0]).toBe('run: k 4096, 131072 x 4096, the last of 300 launches; 1 blocks, 3 waves read, '
      + '1 dropped (HW_ID1 changed), 1 unstamped, 27 slots empty, 2 misplaced; REALTIME 100 MHz; counter 20 bits');
    // With no blocks to read, it says why: no record, no stamps on this card, or every block dropped.
    const reason = (waves, head) => summary.format(summarize(waves, head));
    expect(reason([], { blocks: 2 })).toEqual([expect.stringMatching(/^run: .* 0 blocks, 0 waves read/),
      'no stamps: no wave stored a record in its place (16 slots empty, 0 misplaced)']);
    expect(reason([ws[2]])).toEqual([expect.stringMatching(/^run: .* 0 blocks, 1 waves read/),
      'no stamps: this fold has none on this card (only gfx11 and gfx12 do)']);
    // Every wave moved (gfx12's HW_ID1 changing in a wave would do it), or a block's wave 0 missing: the counter's
    // bits are still those of the waves read.
    const moved = [0, 1].map((w) => wave({ block: 0, wave: w, start: 0, ticks: 1000, bits: 32, hw1: 7 }));
    const noLead = [wave({ block: 1, wave: 1, start: 0, ticks: 1000, bits: 32 })];
    const lines = reason(moved.concat(noLead));
    expect(lines[0]).toMatch(/ 0 blocks, 3 waves read, 2 dropped \(HW_ID1 changed\), .* counter 32 bits$/);
    expect(lines[1]).toBe('no usable blocks: 2 waves dropped (HW_ID1 changed), 1 blocks without their wave 0');
    expect(reason(moved)).toEqual([expect.stringMatching(/ 0 blocks, 2 waves read, 2 dropped \(HW_ID1 changed\), .* counter 32 bits$/),
      'no usable blocks: 2 waves dropped (HW_ID1 changed), 0 blocks without their wave 0']);
    expect(() => summary.load(Buffer.alloc(64))).toThrow('not a stamps file');
    const bad = file([], { blocks: 1 });
    bad.writeUInt32LE(8, 4);
    expect(() => summary.load(bad)).toThrow('8 words a wave, not 16');
    expect(() => summary.load(file([], { blocks: 2 }).subarray(0, 100))).toThrow('stamps file is short');
  });

  it('measures how far apart a block\'s waves read the counter, on one SIMD and across SIMDs', () => {
    // Eight waves leave their last barrier together; each SIMD's counter has its own offset, and the two
    // waves on a SIMD read it 4 cycles apart.
    const offset = [0, 300000, 600000, 900000];
    const ws = [];
    for (let w = 0; w < 8; w++) {
      const x = wave({ block: 0, wave: w, start: 0, ticks: 1000, cyc0: offset[w % 4] + (w >= 4 ? 4 : 0) });
      ws.push(x);
    }
    const s = summarize(ws);
    expect([s.counter.one.n, s.counter.one.med]).toEqual([4, 4]);
    expect(s.counter.other.n).toBe(24);
    expect(s.counter.other.med).toBeGreaterThan(100000);
    // Distances either way round, across a wrap.
    expect(summary.apart(2 ** 20 - 10, 15, 20)).toBe(25);
    expect(summary.apart(15, 2 ** 20 - 10, 20)).toBe(25);
    expect(summary.apart(5, 2 ** 31 + 5, 32)).toBe(2 ** 31);
  });

  it('checks across blocks that a SIMD\'s counter follows REALTIME, which the phases rest on', () => {
    // One wave a block on SIMD 0 of WGP 0, each starting 40 ticks after the one before ends. The counter is one
    // free-running counter at 25 cycles a tick, wrapping 20 bits during block 1; block 3's reads 5000 cycles off
    // (not that counter), and block 4 starts too long after (500000 cycles: past gfx11's quarter wrap).
    const cpt = 25, c = (t) => (2 ** 20 - 30000 + t * cpt) % 2 ** 20;
    const starts = [0, 1040, 2080, 3120, 24120];
    const ws = starts.map((start, b) => wave({ block: b, start, ticks: 1000, cyc0: c(start) + (b === 3 ? 5000 : 0) }));
    ws.push(wave({ block: 5, start: 1040, ticks: 1000, simd: 1, cyc0: 123 }));   // another SIMD: no successor
    const s = summarize(ws);
    expect(s.counter.across).toEqual({ n: 3, p10: 0, med: 0, p90: 5000 });
    expect(summary.format(s)[4]).toBe('counter across blocks: 3 waves that started on a SIMD after another wave there '
      + 'ended; counter gap less REALTIME gap median 0 cycles (p90 5000)');
    // A counter gap below zero (the successor's read 1200 early: -200 against REALTIME's 1000) is read as -200,
    // not as nearly a whole wrap.
    const early = starts.slice(0, 2).map((start, b) => wave({ block: b, start, ticks: 1000, cyc0: c(start) - (b ? 1200 : 0) }));
    expect(summarize(early).counter.across).toEqual({ n: 1, p10: 1200, med: 1200, p90: 1200 });
  });

  // Two blocks on WGP 3 at once. Their waves on SIMD 0 (A's wave 0, B's wave 1) read one counter, `c(t)` at
  // tick t, 25 cycles a tick from `base`; B's wave 0 is on SIMD 2 and crossed nothing. The crossings are given.
  function pairOf(o) {
    const cpt = 25, c = (t) => o.base + t * cpt;
    const A0 = wave({ block: o.a, start: o.startA, ticks: 10000, wgp: 3, cyc0: c(o.startA), seen: 3, x16: o.xa16, x19: o.xa19 });
    const B1 = wave({ block: o.b, wave: 1, simd: 0, start: o.startB, ticks: 10000, wgp: 3, cyc0: c(o.startB), seen: 3,
      x16: o.xb16, x19: o.xb19 });
    const B0 = wave({ block: o.b, start: o.startB, ticks: 10000, wgp: 3, simd: 2, cyc0: 777 });
    return [A0, B0, B1];
  }

  it('finds the phase of two blocks sharing a WGP at the boundary both crossed, from waves on one SIMD', () => {
    // A starts at tick 0, B at 3000. On SIMD 0 the counter crosses 0x80000 (bit 19) at tick 5000: A's barrier
    // exit after it reads 0x80000 + 300 and B's 0x80000 + 1100. A slab is 247000 / 128 cycles, about 1930.
    const base = 0x80000 - 5000 * 25;
    const ws = pairOf({ a: 0, b: 1, startA: 0, startB: 3000, base, xa19: 0x80000 + 300, xb19: 0x80000 + 1100,
      xa16: 0x30000 + 50, xb16: 0x40000 + 60 });   // different bit-16 boundaries: no reading there
    // A block on another WGP, and one on WGP 3 after both ended: neither pairs.
    ws.push(wave({ block: 2, start: 0, ticks: 10000, wgp: 4 }), wave({ block: 3, start: 20000, ticks: 10000, wgp: 3 }));
    const s = summarize(ws);
    expect([s.pairs.n, s.pairs.simd]).toEqual([1, 1]);
    expect(s.pairs.gap.med).toBe(3000 * 25);
    expect([s.phase[19].n, s.phase[19].med]).toEqual([1, 800]);
    expect(s.phase[16].n).toBe(0);
    // Folded into one slab: 1500 apart is about 430 from the next barrier.
    const far = pairOf({ a: 0, b: 1, startA: 0, startB: 3000, base, xa19: 0x80000 + 100, xb19: 0x80000 + 1600 });
    expect(summarize(far).phase[19].med).toBeCloseTo(247000 / 128 - 1500, 9);
    const lines = summary.format(s);
    expect(lines[5]).toBe('pairs: 1 pairs of blocks on one WGP at once, 1 with a SIMD in common; start gap median 75000 cycles');
    // REALTIME puts the two crossings 32 ticks apart: 800 cycles, under a slab, so one boundary.
    expect(lines[7]).toBe('phase bit 19: 1 pairs on one boundary, median 800, p10 800, p90 800 cycles (rest slab 1930); '
      + 'the crossings 800 cycles apart on REALTIME (p90 800); 0 dropped: more than 3 slabs apart there');
    // The same counter values, but REALTIME puts B's crossing 20000 cycles later: more than 3 slabs, and no whole
    // number of wraps, so the counter didn't agree with REALTIME there. Dropped and counted.
    const late = pairOf({ a: 0, b: 1, startA: 0, startB: 3000, base, xa19: 0x80000 + 300, xb19: 0x80000 + 1100,
      xa16: 0x30000 + 50, xb16: 0x40000 + 60 });
    late[2].cyc0 -= 20000;
    late[2].cyc1 -= 20000;
    const r = summarize(late);
    expect([r.phase[19].n, r.phase[19].rejected, r.phase[16].rejected]).toEqual([0, 1, 0]);
    expect(summary.format(r)[7]).toMatch(/^phase bit 19: 0 pairs on one boundary, .*; 1 dropped: more than 3 slabs apart there$/);
    expect(summary.MAX_SLABS_APART).toBe(3);
  });

  it('pairs only waves on one SIMD, and not a boundary that came round again on a 20-bit counter', () => {
    const base = 0x80000 - 5000 * 25;
    // B's SIMD-0 wave on SIMD 1 instead: no wave of B shares a SIMD with A's.
    const off = pairOf({ a: 0, b: 1, startA: 0, startB: 3000, base, xa19: 0x80000 + 300, xb19: 0x80000 + 1100 });
    off[2].hw0 = off[2].hw1 = hwId(0, 0, 3, 1, 1);
    const s = summarize(off);
    expect([s.pairs.n, s.pairs.simd, s.phase[19].n]).toEqual([1, 0, 0]);
    // A long A crosses 0x80000 at tick 1000; B starts at 25000 and its first bit-19 crossing is 0x80000 again,
    // 2^20 cycles later (tick 42943): the same counter value, a different boundary.
    const cpt = 25, b2 = 0x80000 - 1000 * cpt, c = (t) => b2 + t * cpt;
    const A = wave({ block: 0, start: 0, ticks: 50000, wgp: 3, cyc0: c(0), seen: 2, x19: 0x80000 + 100 });
    const B = wave({ block: 1, start: 25000, ticks: 50000, wgp: 3, cyc0: c(25000), seen: 2, x19: 0x80000 + 400 });
    const r = summarize([A, B]);
    expect([r.pairs.n, r.pairs.simd, r.phase[19].n]).toEqual([1, 1, 0]);
    // REALTIME puts them one wrap apart, so they are skipped, not counted as the counter disagreeing.
    expect(r.phase[19].rejected).toBe(0);
    // On gfx12's 32 bits the two values differ above bit 19 as well.
    const A32 = Object.assign({}, A, { bits: 32, cyc0: c(0) + 2 ** 20, x19: 0x80000 + 2 ** 20 + 100 });
    const B32 = Object.assign({}, B, { bits: 32, cyc0: c(25000) + 2 ** 20, x19: 0x80000 + 2 * 2 ** 20 + 400 });
    expect(summarize([A32, B32]).phase[19].n).toBe(0);
  });

  it('gives D and the readings for items 14 and 15 from a stamp build and its abl8 twin', () => {
    // `n` pairs of blocks, each pair on a WGP of its own, after one short block that ends first (or, with `long`,
    // a long one that leaves no block for "rest"); the stamp build's barriers `phase` cycles apart.
    const build = (n, mhz, phase, bar, mem, long) => {
      const ws = [wave({ block: 0, start: 0, ticks: long ? 20000 : 100, wgp: 15, se: 7 })];
      for (let i = 0; i < n; i++) {
        const base = 0x80000 - 5000 * 25;
        const p = pairOf({ a: 1 + 2 * i, b: 2 + 2 * i, startA: 200, startB: 3000, base, xa19: 0x80000 + 300,
          xb19: 0x80000 + 300 + phase, xa16: 0x10000, xb16: 0x10000 + phase });
        for (const w of p) {
          w.hw0 = w.hw1 = (w.hw0 & ~(15 << 10) & ~(7 << 18)) | ((i % 16) << 10) | ((i >> 4) << 18);
          const x = wave({ block: w.block, wave: w.wave, start: w.rt0, ticks: 10000, mhz, bar, mem });
          Object.assign(w, { loop: x.loop, pro: x.pro, epi: x.epi, bar: x.bar, mem: x.mem, rt1: x.rt1 });
        }
        ws.push(...p);
      }
      return summarize(ws);
    };
    // abl8: 10000 cycles less a block at the same ticks, 3840 of them barrier wait (30 a slab).
    const st = build(40, 2500, 600, 128 * 150, 128 * 50);
    const ab = build(40, 2400, 600, 128 * 120, 128 * 50);
    const g = summary.gates(st, ab);
    expect([g.slab, g.slab8, g.barSlab, g.memSlab]).toEqual([247000 / 128, 237000 / 128, 150, 50]);
    // D's LDS part is the work a slab (the loop less barrier and memory wait), so abl8's shorter barrier wait
    // isn't counted twice: 6160 cycles a block, not 10000.
    expect([g.work, g.work8]).toEqual([(247000 - 19200 - 6400) / 128, (237000 - 15360 - 6400) / 128]);
    expect(g.lds).toBeCloseTo(6160 / 128, 9);
    expect(g.D).toBeCloseTo(150 + 6160 / 128, 9);
    expect(g.Dm).toBeCloseTo(200 + 6160 / 128, 9);
    expect([g.bit, g.phase.n, g.phase.med]).toEqual([19, 40, 600]);
    expect([g.item14, g.item14m]).toEqual(['kill', 'kill']);   // 600 > D (198) and D with memory wait (248)
    // Item 15 counts whole slabs, barrier wait and all.
    expect(g.save).toBeCloseTo(10000 / 128, 9);
    expect(g.share).toBeCloseTo(10000 / 247000, 12);
    expect(g.item15).toBe('pass');                              // 4.05% >= 3%
    // No successions on a SIMD in these blocks, so the counter across blocks is unchecked; the gate says so.
    expect([g.counter, g.across.n]).toEqual(['unchecked', 0]);
    expect(summary.formatGates(g)).toEqual([
      'D: 198 cycles a slab = barrier wait 150 + LDS fragment loads 48 (work a slab, the loop less barrier and memory '
        + 'wait: 1730 less abl8\'s 1682); with memory wait 50: 248; medians over the rest of the launch',
      'item 14: phase 600 cycles (bit 19, 40 pairs) against D 198: kill; against D with memory wait 248: kill (kill if '
        + 'the phase is larger); the counter across blocks is unchecked (0 waves)',
      'item 15: abl8 saves 78 of 1930 cycles a slab, 4.05%: pass (kill under 3%)',
    ]);
    // A phase between the two D's: memory wait decides it, and both readings show that.
    const mid = summary.gates(build(40, 2500, 220, 128 * 150, 128 * 50), ab);
    expect([mid.item14, mid.item14m]).toEqual(['kill', 'pass']);
    // In step, and a small saving: pass and kill.
    const g2 = summary.gates(build(40, 2500, 100, 128 * 150), build(40, 2490, 100, 128 * 150));
    expect([g2.item14, g2.item15]).toEqual(['pass', 'kill']);
    // The counter across blocks: following REALTIME it adds nothing; half a slab off or more, no reading.
    const across = (a) => summary.gates(Object.assign({}, st, { counter: Object.assign({}, st.counter, { across: a }) }), ab);
    const ok = across({ n: 500, med: 40, p90: 90 });
    expect([ok.counter, ok.item14, summary.formatGates(ok)[1]]).toEqual(['ok', 'kill',
      'item 14: phase 600 cycles (bit 19, 40 pairs) against D 198: kill; against D with memory wait 248: kill (kill if '
        + 'the phase is larger)']);
    const off = across({ n: 500, med: 262519, p90: 471830 });
    expect([off.counter, off.item14, off.item14m, off.item15]).toEqual(['off', 'no reading', 'no reading', 'pass']);
    expect(summary.formatGates(off)[1]).toBe('item 14: phase 600 cycles (bit 19, 40 pairs) against D 198: no reading; '
      + 'against D with memory wait 248: no reading (kill if the phase is larger); the counter is 262519 cycles off '
      + 'REALTIME across blocks on one SIMD (median), so it isn\'t one counter a SIMD');
    expect(across({ n: 500, med: 964, p90: 2000 }).counter).toBe('ok');      // half a slab is 964.8
    expect(across({ n: 500, med: 965, p90: 2000 }).counter).toBe('off');
    expect(across({ n: summary.MIN_PAIRS - 1, med: 262519, p90: 471830 }).counter).toBe('unchecked');
    // Too few bit-19 pairs: bit 16's; too few of either: no reading.
    const few = build(40, 2500, 600, 128 * 150, 128 * 50);
    few.phase[19] = { n: summary.MIN_PAIRS - 1, med: 5 };
    expect([summary.gates(few, ab).bit, summary.gates(few, ab).item14]).toEqual([16, 'kill']);
    few.phase[16] = { n: 0, med: NaN };
    const g3 = summary.gates(few, ab);
    expect([g3.bit, g3.item14]).toEqual([0, 'no reading']);
    expect(summary.formatGates(g3)[1]).toMatch(/^item 14: under 30 pairs against D 198: no reading; against D with /);
    // Not a number anywhere (a build with no slabs in its rest): no reading, not a verdict.
    const nan = (s) => Object.assign({}, s, { rest: Object.assign({}, s.rest, { slab: { n: 0, med: NaN }, work: { n: 0, med: NaN } }) });
    const g4 = summary.gates(nan(st), ab);
    expect([g4.item14, g4.item14m, g4.item15]).toEqual(['no reading', 'no reading', 'no reading']);
    expect(summary.formatGates(g4)[2]).toBe('item 15: abl8 saves - of - cycles a slab, -: no reading (kill under 3%)');
    expect(summary.gates(st, nan(ab)).item15).toBe('no reading');
    // A build with no stamps, or with every block in the first round: no reading, and why.
    const none = summary.gates(st, summarize([wave({ block: 0, start: 0, ticks: 1, bits: 0 })]));
    expect(summary.formatGates(none)).toEqual(['gates: no reading: the abl8 build: no stamps: this fold has none on this '
      + 'card (only gfx11 and gfx12 do)']);
    const firstOnly = build(40, 2500, 600, 128 * 150, 128 * 50, true);
    expect([firstOnly.rest.n, firstOnly.phase[19].n]).toEqual([0, 40]);
    expect(summary.formatGates(summary.gates(firstOnly, ab))).toEqual(['gates: no reading: the stamp build has no '
      + 'blocks after the first round (each started before the first one ended): run more blocks']);
    expect(summary.gates(ab, firstOnly).why).toMatch(/^the abl8 build has no blocks after the first round/);
  });

  it('runs from the command line', () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'amd-stamps-'));
    try {
      const f = path.join(tmp, 'stamps.bin');
      fs.writeFileSync(f, file([wave({ block: 0, start: 0, ticks: 10000 }), wave({ block: 1, start: 20000, ticks: 10000 })]));
      const r = spawnSync(process.execPath, [SUMMARY, 'summary', f], { encoding: 'utf8' });
      expect(r.status).toBe(0);
      expect(r.stdout.split('\n')[0]).toMatch(/^run: k 4096, 131072 x 4096, the last of 300 launches; 2 blocks/);
      const g = spawnSync(process.execPath, [SUMMARY, 'gates', f, f], { encoding: 'utf8' });
      expect([g.status, g.stdout.split('\n')[2]]).toEqual([0, 'item 15: abl8 saves 0 of 1930 cycles a slab, 0.00%: kill (kill under 3%)']);
      const out = [], err = [];
      expect(summary.main(['summary'], (l) => out.push(l), (l) => err.push(l))).toBe(2);
      expect(err[0]).toMatch(/^usage: stamp-summary.js summary/);
      expect(summary.main(['summary', path.join(tmp, 'none.bin')], (l) => out.push(l), (l) => err.push(l))).toBe(1);
      expect(err[1]).toMatch(/^stamps: ENOENT/);
      expect(out).toEqual([]);
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  });
});

describe('sweep.sh SWEEP_STAMP', () => {
  // A stand-in build-amd.sh whose fold test logs its arguments, checks a stamp build right and an ablation
  // wrong, and in its stamps mode copies a made-up stamps file to the path it is given.
  const STUB = [
    '#!/bin/bash',
    'mkdir -p build',
    'printf \'#!/bin/bash\\nf="%s"\\n\' "${PEARL_AMD_CFLAGS:-}" > build/pearl_amd_foldtest',
    'cat >> build/pearl_amd_foldtest <<\'FT\'',
    'echo "$*" >> "$ARGS_LOG"',
    'if [ -n "${9:-}" ]; then',
    '  case "$f" in *ABLATE=8*) cp "$STAMP_ABL8" "$9" ;; *STAMP*) cp "$STAMP_PLAIN" "$9" ;; *) echo "stamps: no clock stamps" >&2; exit 2 ;; esac',
    '  echo "$6  timed only: $1 x $2, $4 launches, 1.0000 s, 100.000 TMAC/s"; echo "stamps: $6, 3 blocks, the last of $4 launches, REALTIME 100000 kHz, in $9"; exit 0',
    'fi',
    'if [ "$5" = 0 ]; then echo "$6  timed only: $1 x $2, $4 launches, 1.0000 s, 100.000 TMAC/s"; exit 0; fi',
    'case "$f" in *ABLATE*) echo "$6  FAIL: 4096 of 4096 words differ; first: region 0 word 0, got 1 want 2"; echo FAIL; exit 1 ;; esac',
    'echo "$6  ok: $1 x $2, all 16 words match; 1 launches, 0.0100 s, 1.000 TMAC/s"; echo PASS',
    'FT',
    'chmod +x build/pearl_amd_foldtest',
    'echo "${PEARL_AMD_CFLAGS:-}" > build/pearl_core_hip.node',
    '',
  ].join('\n');

  let tmp;
  beforeEach(() => { tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'amd-sweep-stamp-')); });
  afterEach(() => { fs.rmSync(tmp, { recursive: true, force: true }); });

  // A copy of sweep.sh and stamp-summary.js with the stand-ins, writing nothing outside tmp: the container-log
  // copy of each line goes to /dev/null.
  function sweep(env) {
    const amd = path.join(tmp, 'amd');
    fs.mkdirSync(path.join(amd, 'test'), { recursive: true });
    fs.mkdirSync(path.join(tmp, 'probes'), { recursive: true });
    fs.mkdirSync(path.join(tmp, 'root'), { recursive: true });
    fs.writeFileSync(path.join(amd, 'build-amd.sh'), STUB, { mode: 0o755 });
    fs.writeFileSync(path.join(tmp, 'probes', 'hashrate.js'), 'console.log(JSON.stringify({ th: 100 }));\n');
    fs.writeFileSync(path.join(amd, 'test', 'sweep.sh'), SWEEP.replace('/proc/1/fd/1', '/dev/null'));
    fs.copyFileSync(SUMMARY, path.join(amd, 'test', 'stamp-summary.js'));
    const stampFile = (name, mhz) => {
      const f = path.join(tmp, name);
      fs.writeFileSync(f, file([0, 1, 2].map((b) => wave({ block: b, start: b * 20000, ticks: 10000, mhz, bar: 128 * 150 }))));
      return f.replace(/\\/g, '/');
    };
    const log = path.join(tmp, 'args.log');
    const e = { ...process.env, ARCH: 'gfx1100', SWEEP_ROOT: path.join(tmp, 'root').replace(/\\/g, '/'),
      ARGS_LOG: log.replace(/\\/g, '/'), SWEEP_FOLD: '', SWEEP_ROUNDS: '0',
      STAMP_PLAIN: stampFile('plain.bin', 2500), STAMP_ABL8: stampFile('abl8.bin', 2400), ...env };
    delete e.PEARL_PROFILE_K;
    for (const k of ['SWEEP_OPERANDS', 'SWEEP_STAMP_RUN', 'SWEEP_TIMEONLY']) if (!(k in env)) delete e[k];
    const r = spawnSync('bash', ['test/sweep.sh'], { cwd: amd, encoding: 'utf8', env: e });
    return {
      status: r.status,
      lines: r.stdout.split('\n').filter(Boolean).map((l) => l.replace(/^\[amdtest\] \S+ sweep /, '')),
      args: fs.existsSync(log) ? fs.readFileSync(log, 'utf8').split('\n').filter(Boolean) : [],
      root: e.SWEEP_ROOT,
    };
  }

  bashIt('stamps each variant it names on the miner\'s operands, prints the summary, then the gates', () => {
    const { status, lines, args, root } = sweep({
      SWEEP: 'base=;stamp=-DPEARL_AMD_STAMP=1;abl8s=-DPEARL_AMD_ABLATE=8 -DPEARL_AMD_STAMP=1',
      SWEEP_TIMEONLY: 'abl8s', SWEEP_STAMP: 'stamp abl8s gone', SWEEP_STAMP_RUN: '131072:4096:600',
    });
    expect(status).toBe(0);
    expect(lines[0]).toMatch(/; stamps from stamp abl8s gone at 131072:4096:600$/);
    // The stamps mode: m, cols, seed 1, the launches at k 2048, no check, the card's fold and k, miner, the file.
    expect(args.filter((a) => / miner /.test(a))).toEqual([
      '131072 4096 1 600 0 wmma11 0 miner ' + root + '/var/stamp/stamps.bin',
      '131072 4096 1 600 0 wmma11 0 miner ' + root + '/var/abl8s/stamps.bin',
    ]);
    const stamp = lines.filter((l) => l.startsWith('stamp '));
    expect(stamp.filter((l) => l.startsWith('stamp stamp '))).toEqual([
      'stamp stamp wmma11 timed only: 131072 x 4096, 600 launches, 1.0000 s, 100.000 TMAC/s',
      expect.stringMatching(/^stamp stamp stamps: wmma11, 3 blocks, the last of 600 launches/),
      expect.stringMatching(/^stamp stamp run: k 4096, 131072 x 4096, the last of 300 launches; 3 blocks, 3 waves read/),
      expect.stringMatching(/^stamp stamp first: 1 blocks, 2500 MHz/),
      expect.stringMatching(/^stamp stamp rest: 2 blocks, 2500 MHz .* barrier .*, 150 a slab;/),
      expect.stringMatching(/^stamp stamp counter: /),
      expect.stringMatching(/^stamp stamp counter across blocks: 2 waves /),
      expect.stringMatching(/^stamp stamp pairs: 0 pairs/),
      expect.stringMatching(/^stamp stamp phase bit 16: 0 pairs/),
      expect.stringMatching(/^stamp stamp phase bit 19: 0 pairs/),
    ]);
    expect(stamp).toContainEqual(expect.stringMatching(/^stamp abl8s rest: 2 blocks, 2400 MHz/));
    expect(stamp).toContain('stamp gone: not stamped, it isn\'t a variant that built and checked');
    expect(stamp.filter((l) => l.startsWith('stamp gates '))).toEqual([
      expect.stringMatching(/^stamp gates stamp\/abl8s D: 228 cycles a slab = barrier wait 150 \+ LDS fragment loads 78 /),
      'stamp gates stamp/abl8s item 14: under 30 pairs against D 228: no reading; against D with memory wait 228: no '
        + 'reading (kill if the phase is larger); the counter across blocks is unchecked (2 waves)',
      'stamp gates stamp/abl8s item 15: abl8 saves 78 of 1930 cycles a slab, 4.05%: pass (kill under 3%)',
    ]);
  }, 30000);

  bashIt('says when a variant wrote no stamps, prints no gates without a pair, and checks SWEEP_STAMP_RUN', () => {
    const { status, lines } = sweep({ SWEEP: 'base=;stamp=-DPEARL_AMD_STAMP=1', SWEEP_STAMP: 'base stamp' });
    expect(status).toBe(0);
    expect(lines).toContain('stamp base: no stamps written');
    expect(lines.some((l) => l.startsWith('stamp gates'))).toBe(false);
    expect(lines.some((l) => l.startsWith('stamp stamp rest: '))).toBe(true);
    // Not asked for: no stamps run, and the start line says nothing of them.
    const none = sweep({ SWEEP: 'base=' });
    expect(none.lines.some((l) => l.startsWith('stamp'))).toBe(false);
    expect(none.lines[0]).not.toContain('stamps');
    for (const bad of ['131072:4096', '131072::600', 'a:b:c']) {
      const r = sweep({ SWEEP: 'base=', SWEEP_STAMP_RUN: bad });
      expect([bad, r.status, r.lines]).toEqual([bad, 2, ['SWEEP_STAMP_RUN is m:cols:launches, not ' + bad]]);
    }
  }, 60000);
});
