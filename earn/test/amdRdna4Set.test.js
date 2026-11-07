'use strict';

// PEARL_AMD_RDNA4_SET (earn/native/amd/src/pearl_amd_fold.h): the RDNA 4 cuts of 4 November, on by default in RDNA 4
// (gfx12) code of the Linux core: RDSEL=1, RDPERM=2, CRANK=1, SCHEDB=1, LDS32=1 and LDS32_PIN=1. The set was timed
// and verified on Salad boxes built with those six -D flags, so the default must give exactly those values in a
// gfx12 device pass and leave every other pass, and the Windows core, as they were. CRANK also makes the matrix fold
// assume rank 128, which the host has to enforce from the card's caps, since the host pass never sees the gfx12
// default. No HIP compiler runs here, so the C++ is read as text.

const fs = require('fs');
const path = require('path');
const { preprocess } = require('./helpers/amdSwitches');

const AMD = path.join(__dirname, '..', 'native', 'amd');
const read = (...p) => fs.readFileSync(path.join(AMD, ...p), 'utf8').replace(/\r\n/g, '\n');
const FOLD = read('src', 'pearl_amd_fold.h');
const HOST = read('src', 'pearl_host_hip.cpp');
const BUILD = read('build-amd.sh');

// The switches of the set and the value each takes with the set on.
const SET = { PEARL_AMD_RDSEL: 1, PEARL_AMD_RDPERM: 2, PEARL_AMD_CRANK: 1, PEARL_AMD_SCHEDB: 1, PEARL_AMD_LDS32: 1,
  PEARL_AMD_LDS32_PIN: 1 };

// The default an #ifndef block gives a macro, as written.
function defaultOf(name) {
  const m = new RegExp(`#ifndef ${name}\\n#define ${name} (.*)\\n#endif`).exec(FOLD);
  if (!m) throw new Error('no default for ' + name);
  return m[1];
}
const evalWith = (expr, vars) => Number(new Function(...Object.keys(vars), `return (${expr});`)(...Object.values(vars)));

describe('the RDNA 4 set (PEARL_AMD_RDNA4_SET)', () => {
  it('is on in a gfx12 device pass only: its default is PEARL_AMD_WMMA12', () => {
    expect(defaultOf('PEARL_AMD_RDNA4_SET')).toBe('PEARL_AMD_WMMA12');
    expect(FOLD).toContain('#if defined(__GFX12__)\n#define PEARL_AMD_WMMA12 1\n#endif');
    // That block is inside the device-pass gate, with every other instruction gate.
    const gate = FOLD.indexOf('#if !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__)\n#if defined(__gfx942__)');
    expect(gate).toBeGreaterThan(0);
    expect(FOLD.indexOf('#define PEARL_AMD_WMMA12 1')).toBeGreaterThan(gate);
    expect(FOLD).toContain('#if PEARL_AMD_RDNA4_SET != 0 && PEARL_AMD_RDNA4_SET != 1\n#error');
  });

  it('gives each switch the value the boxes timed and verified with it on, and 0 with it off', () => {
    for (const [name, on] of Object.entries(SET)) {
      const expr = defaultOf(name);
      expect([name, expr]).toEqual([name, expect.stringContaining('PEARL_AMD_RDNA4_SET')]);
      expect([name, evalWith(expr, { PEARL_AMD_RDNA4_SET: 1 })]).toEqual([name, on]);
      expect([name, evalWith(expr, { PEARL_AMD_RDNA4_SET: 0 })]).toEqual([name, 0]);
    }
    // Every default is defined after the set's, which comes after the instruction gates.
    const at = FOLD.indexOf('#define PEARL_AMD_RDNA4_SET PEARL_AMD_WMMA12');
    expect(at).toBeGreaterThan(FOLD.indexOf('#define PEARL_AMD_WMMA12 0'));
    for (const name of Object.keys(SET)) expect(FOLD.indexOf(`#define ${name} `)).toBeGreaterThan(at);
  });

  it('leaves out the parts that stopped paying in the set, and the timing-only switches', () => {
    for (const name of ['PEARL_AMD_STOFF', 'PEARL_AMD_UNROLL4', 'PEARL_AMD_CMASK', 'PEARL_AMD_LDS_PAD',
      'PEARL_AMD_LDS32_S48']) {
      expect([name, defaultOf(name)]).toEqual([name, '0']);
    }
    // LPIN is on only with PEARL_AMD_PF2PIN, and only in gfx11 code (amdPf2pin.test.js), so it is 0 in RDNA 4's.
    expect(defaultOf('PEARL_AMD_LPIN')).toBe('(2 * PEARL_AMD_PF2PIN_ON)');
    expect(FOLD).toContain('#define PEARL_AMD_PF2PIN_ON (PEARL_AMD_PF2PIN && PEARL_AMD_WMMA11)\n');
    // PEARL_AMD_WAVES has no default: the attribute is there only when it is set.
    expect(FOLD).toContain('#ifdef PEARL_AMD_WAVES\n#define PEARL_AMD_WMMA12_OCC __attribute__((amdgpu_waves_per_eu(PEARL_AMD_WAVES)))');
    expect(FOLD).not.toMatch(/#define PEARL_AMD_WAVES /);
    // The split slab barrier was taken out of the code.
    expect(FOLD).not.toContain('PEARL_AMD_SPLIT');
    // The switches the sched pin and the swizzle helper hang on are read after their defaults.
    expect(FOLD.indexOf('#if (PEARL_AMD_SCHEDB || PEARL_AMD_LPIN)')).toBeGreaterThan(FOLD.indexOf('#define PEARL_AMD_SCHEDB '));
    expect(FOLD.indexOf('#if PEARL_AMD_LDS32\n__device__')).toBeGreaterThan(FOLD.indexOf('#define PEARL_AMD_LDS32 '));
  });

  it('builds the set\'s slab loop in RDNA 4 code: a constant chunk, the pin before the stores, pinned offsets', () => {
    const on = preprocess(FOLD, { ...SET, PEARL_AMD_ABLATE: 0, PEARL_AMD_STAMP: 0, PEARL_AMD_CMASK: 0 });
    const off = preprocess(FOLD, { ...Object.fromEntries(Object.keys(SET).map((n) => [n, 0])), PEARL_AMD_ABLATE: 0,
      PEARL_AMD_STAMP: 0, PEARL_AMD_CMASK: 0 });
    expect(on).toContain('constexpr uint32_t perChunk = PEARL_FOLD_RANK / (uint32_t)KS;');
    expect(on).toContain('    if constexpr (B == 1)\n    if ((s + 1) % perChunk == 0) readout((s + 1) / perChunk - 1);\n'
      + '    PEARL_AMD_SCHED_PIN();   // PEARL_AMD_SCHEDB (above)');
    expect(on).toContain('const int off = l32rd + kk * P::kTK;');
    expect(on).toContain('asm("" : "=v"(old) : "v"(x));');
    expect(on).toContain('if (c & 8u) {\n          asm volatile("; rdsel hi");');
    expect(off).toContain('const uint32_t slabs = k / (uint32_t)KS, perChunk = rank / (uint32_t)KS;');
    expect(off).not.toContain('l32rd');
    // The RDSEL arms are behind the switch, and only in a GPU's device pass.
    expect(FOLD).toContain('#if PEARL_AMD_RDSEL && !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__)\n');
    expect(off).not.toContain('PEARL_AMD_SCHED_PIN();   // PEARL_AMD_SCHEDB (above)');
    expect(off).not.toContain('if constexpr (B == 1)\n    if ((s + 1) % perChunk == 0)');
  });

  it('stays out of the Windows core, before any tuning flags, in its build and its check', () => {
    expect(BUILD).toContain('SET_OS=\nif windows; then SET_OS="-DPEARL_AMD_RDNA4_SET=0 -DPEARL_AMD_RDNA3_BUFLD=0"; fi\n');
    expect(BUILD).toContain('-Wno-unused-value $KWRAP_OS $SET_OS ${PEARL_AMD_CFLAGS:-}"');
    expect(BUILD).toContain('-std=c++17 -O3 $KWRAP_OS $SET_OS ${PEARL_AMD_CFLAGS:-} $inc --cuda-device-only');
    // The gpu build (Linux, the release's) has no OS flag: the default holds there.
    const gpu = BUILD.slice(BUILD.indexOf('\n  gpu)\n'), BUILD.indexOf('\n  winlib)\n'));
    expect(gpu).not.toContain('SET_OS');
    expect(gpu).not.toContain('KWRAP_OS');
  });
});

describe('the rank CRANK assumes (PEARL_AMD_CAP_RANK)', () => {
  it('is a cap bit of its own, set by a code object whose PF2 loop has CRANK', () => {
    const bits = [...FOLD.matchAll(/#define PEARL_AMD_CAP_\w+ (0x[0-9a-f]+|\d+)u/g)].map((m) => Number(m[1]));
    expect(new Set(bits).size).toBe(bits.length);
    expect(FOLD).toContain('#define PEARL_AMD_CAP_RANK 0x2000u');
    expect(FOLD).toContain('  if (PEARL_AMD_CRANK && PEARL_AMD_PF2) out[0] |= PEARL_AMD_CAP_RANK;\n');
    // The fold's own assumption, which the cap stands for: a chunk is PEARL_FOLD_RANK / KS slabs.
    expect(FOLD).toContain('#if PEARL_AMD_CRANK && PEARL_AMD_PF2\n');
    expect(FOLD).toContain('__builtin_assume((slabs & (perChunk - 1u)) == 0u);');
  });

  it('makes the host refuse any other rank for the card\'s matrix fold, from the caps, not the macro', () => {
    expect(HOST).toContain('ctx->rank128 = (caps[0] & PEARL_AMD_CAP_RANK) != 0;');
    expect(HOST).toContain('|| (ctx->rank128 && f.cap != PEARL_AMD_CAP_REF && rank != PEARL_FOLD_RANK)');
    // The host pass never sees the gfx12 default, so its code must not read the macro.
    const code = HOST.split('\n').map((l) => l.replace(/\/\/.*$/, '')).join('\n');
    expect(code).not.toContain('PEARL_AMD_CRANK');
    expect(HOST).toContain('ctx->rank128 ? ", exactly 128 for this card\'s matrix fold" : ""');
  });
});
