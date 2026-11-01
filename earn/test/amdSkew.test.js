'use strict';

// PEARL_AMD_SKEW (earn/native/amd/src/pearl_amd_fold.h) delays each of a launch's first blocks on an RDNA card once,
// before its first slab, by its place in the order its WGP's three blocks arrived in, so the three run their slabs
// out of phase (research item 14). The host sets the waits at run time (PEARL_AMD_SKEW_WAIT), so one build races
// several. These pin that a release build gets the same code, that it can ship, which targets have it, the waits (the
// host's reading run here as JS, out of the C), the WGP counter's index and tickets, the fold test's probe (its sum
// run as JS, out of the C), the host's and the fold test's setup, and test/skew-race.sh on stand-ins that print the
// fold test's own lines. No HIP compiler runs here: the C is read as text (helpers/amdSwitches.js), and the shell
// parts run under bash where there is one.

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const { preprocess } = require('./helpers/amdSwitches');

const NATIVE = path.join(__dirname, '..', 'native');
const AMD = path.join(NATIVE, 'amd');
const read = (...f) => fs.readFileSync(path.join(NATIVE, ...f), 'utf8').replace(/\r\n/g, '\n');
const FOLD = read('amd', 'src', 'pearl_amd_fold.h');
const HOST = read('amd', 'src', 'pearl_host_hip.cpp');
const FOLDTEST = read('amd', 'test', 'pearl_amd_foldtest.cpp');
const BUILD = read('amd', 'build-amd.sh');
const README = read('amd', 'README.md');
const SWEEP = read('amd', 'test', 'sweep.sh');
const RACE = read('amd', 'test', 'skew-race.sh');

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
// The C as a pass with these switches leaves it: a release's timing switches off, the skew as given.
const at = (text, sw) => preprocess(text, Object.assign({ PEARL_AMD_ABLATE: 0, PEARL_AMD_STAMP: 0, PEARL_AMD_SKEW: 0 }, sw));
// A compile pass of the fold header: an RDNA device pass (with the skew's targets or not), or the host, CDNA or RDNA 2's
// (neither); not HIP-CPU. PEARL_AMD_SKEW_THREE is what the header's own target test gives (checked below).
const pass = (sw) => at(FOLD, Object.assign({ PEARL_AMD_WMMA11: 0, PEARL_AMD_WMMA12: 0, PEARL_AMD_EMU: 0,
  PEARL_AMD_SKEW_THREE: 0 }, sw));
const gfx1100 = { PEARL_AMD_WMMA11: 1, PEARL_AMD_SKEW_THREE: 1 };
const gfx1102 = { PEARL_AMD_WMMA11: 1, PEARL_AMD_SKEW_THREE: 0 };
const gfx1201 = { PEARL_AMD_WMMA12: 1, PEARL_AMD_SKEW_THREE: 1 };
const on = (s, sw) => pass(Object.assign({ PEARL_AMD_SKEW: s }, sw));
const code = (t) => t.split('\n').filter((l) => l.trim() && !/^\s*\/\//.test(l));
const body = (t) => slice(t, 'void pearl_amd_fold_body(', '\n}\n');
const slabStep = (t) => slice(t, 'auto slab_step = [&](uint32_t s, auto parity) {', '\n  };\n');
const caps = (t) => slice(t, 'extern "C" __global__ void pearl_amd_caps(uint32_t *out) {', '\n}\n');
const fn = (t, head) => slice(t, head, '\n}\n');
const define = (name) => Number(grab(FOLD, new RegExp('^#define ' + name + ' (\\d+)u$', 'm')));
// A C expression of unsigned constants and the names given, run as BigInt, whose / truncates as C's does.
const cExpr = (expr, names) => new Function(...Object.keys(names),
  'return ' + expr.replace(/\b(0x[0-9a-fA-F]+|\d+)u\b/g, '$1n') + ';')(...Object.values(names));
// The same with Numbers, for bit operations under 2^32: >> is >>>, as on a uint32_t.
const cBits = (expr, names) => new Function(...Object.keys(names),
  'return (' + expr.replace(/\b(0x[0-9a-fA-F]+|\d+)u\b/g, '$1').replace(/>>/g, '>>>') + ') >>> 0;')(...Object.values(names));
const NAMES = /PEARL_AMD_SKEW|pearl_amd_skew|kPearlAmdSkew|skew_probe|atomicInc|s_sleep|s_sendmsg_rtn\(|dSkew|readfirstlane|s_getreg|pearl_amd_sgpr|pearl_amd_hw_id1/;

const WAYS = define('PEARL_AMD_SKEW_WAYS');
const WGPS = define('PEARL_AMD_SKEW_WGPS');
const SLAB11 = define('PEARL_AMD_SKEW_SLAB11');
const SLAB12 = define('PEARL_AMD_SKEW_SLAB12');
const PCT_MAX = define('PEARL_AMD_SKEW_PCT_MAX');
// PEARL_AMD_SKEW_COUNT(pct, slab), out of its macro.
const COUNT = (() => {
  const m = /^#define PEARL_AMD_SKEW_COUNT\(pct, slab\) (.*)$/m.exec(FOLD);
  return (pct, slab) => Number(cExpr(m[1], { pct: BigInt(pct), slab: BigInt(slab) }));
})();
// kPearlAmdSkewPct, each value's waits for places 0, 1 and 2.
const PCT = JSON.parse('[' + grab(FOLD, /constexpr uint32_t kPearlAmdSkewPct\[4\]\[PEARL_AMD_SKEW_WAYS\] = \{\n\s*(.*)\};/)
  .replace(/u\b/g, '').replace(/\{/g, '[').replace(/\}/g, ']') + ']');

// pearl_amd_skew_waits as JS: the same statements, with a C string's NUL at the end. Its source is also handed to the
// stand-in fold test below, so the lines it prints come out of the C's reading of PEARL_AMD_SKEW_WAIT.
const WAITS_SRC = (() => {
  const js = fn(FOLD, 'inline bool pearl_amd_skew_waits(')
    .replace(/^inline bool pearl_amd_skew_waits\(.*\{\n/, '')
    .replace(/\n\}\n$/, '')
    .replace('uint32_t v[3] = {0u, 0u, 0u}, i = 0u, digits = 0u;', 'let v = [0, 0, 0], i = 0, digits = 0;')
    .replace('bool good =', 'let good =')
    .replace(/nullptr/g, 'null')
    .replace('for (uint32_t p = 0u;', 'for (let p = 0;')
    .replace(/\(uint32_t\)/g, '')
    .replace(/\b(\d+)u\b/g, '$1');
  if (/uint32_t|bool |->/.test(js)) throw new Error('not all C: ' + js);
  return js;
})();
const WAITS_ARGS = ['env', 'wmma11', 'pct', 'sleeps', 'kPearlAmdSkewPct', 'PEARL_AMD_SKEW', 'PEARL_AMD_SKEW_COUNT',
  'PEARL_AMD_SKEW_SLAB11', 'PEARL_AMD_SKEW_SLAB12', 'PEARL_AMD_SKEW_PCT_MAX'];
const waitsFn = new Function(...WAITS_ARGS, WAITS_SRC);
// What the host reads: { set, pct, sleeps } for an environment value (null: unset), a build's value and a fold.
function waits(env, skew, wmma11) {
  const pct = [], sleeps = [];
  const set = waitsFn(env === null ? null : env + '\0', wmma11, pct, sleeps, PCT, skew, COUNT, SLAB11, SLAB12, PCT_MAX);
  return { set, pct, sleeps };
}

// skew_probe_sum (the fold test) as JS, and the test its line makes of the sum.
const probeSum = (() => {
  const js = fn(FOLDTEST, 'void skew_probe_sum(')
    .replace(/^void skew_probe_sum\(.*\{\n/, '')
    .replace(/\n\}\n$/, '')
    .replace('uint32_t n[PEARL_AMD_SKEW_WGPS] = {0u}, seen[PEARL_AMD_SKEW_WGPS] = {0u};',
      'let n = Array(PEARL_AMD_SKEW_WGPS).fill(0), seen = Array(PEARL_AMD_SKEW_WGPS).fill(0);')
    .replace(/~0u/g, '0xFFFFFFFF')
    .replace(/for \(uint32_t (\w) = 0u;/g, 'for (let $1 = 0;')
    .replace(/const uint32_t /g, 'const ')
    .replace(/>> 8/g, '>>> 8')
    .replace(/\b(0x[0-9a-fA-F]+|\d+)u\b/g, '$1');
  if (/uint32_t/.test(js)) throw new Error('not all C: ' + js);
  const f = new Function('out', 'blocks', 's', 'PEARL_AMD_SKEW_WGPS', 'PEARL_AMD_SKEW_WAYS', js);
  return (out) => { const s = []; f(out, out.length, s, WGPS, WAYS); return s; };
})();
const designedC = grab(FOLDTEST, /const bool designed = (.*);/);
const designed = (s, first) => new Function('s', 'first', 'return ' + designedC + ';')(s, first);

// C's printf, as far as the fold test's lines use it: %d, %u, %s and %f with '-', a width and a precision, and %%.
const SPRINTF = `function sprintf(f, ...a) {
  let i = 0;
  return f.replace(/%(-?)(\\d*)(?:\\.(\\d+))?([dusf%])/g, (m, left, width, prec, c) => {
    if (c === '%') return '%';
    const v = a[i++];
    let s = c === 'f' ? Number(v).toFixed(prec === undefined ? 6 : Number(prec)) : c === 's' ? String(v) : String(Math.trunc(v));
    while (s.length < Number(width || 0)) s = left ? s + ' ' : ' ' + s;
    return s;
  });
}`;
const sprintf = new Function(SPRINTF + '\nreturn sprintf;')();
// The format of the fold test's printf that starts with `start`: its string literals, joined.
function fmt(start) {
  const i = FOLDTEST.indexOf('printf("' + start);
  if (i < 0) throw new Error('no printf: ' + start);
  const lits = /^printf\(((?:\s*"(?:[^"\\]|\\.)*")+)/.exec(FOLDTEST.slice(i))[1];
  return lits.match(/"(?:[^"\\]|\\.)*"/g).map((s) => JSON.parse(s)).join('');
}
const FMT = {
  device: fmt('device: %s, %s, %d CUs'),
  skew: fmt('skew: PEARL_AMD_SKEW=%d on'),
  probe: fmt('skew probe: %u blocks, %u took'),
  none: fmt('skew probe: %u blocks, none'),
  timed: fmt('%-7s timed only'),
  ok: fmt('%-7s ok'),
  designed: grab(FOLDTEST, /designed \? "([^"]+)"/),
  notDesigned: grab(FOLDTEST, /designed \? "[^"]+" : "([^"]+)"/),
  fromEnv: grab(FOLDTEST, /fromEnv \? "([^"]+)"/),
  build: grab(FOLDTEST, /fromEnv \? "[^"]+" : "([^"]+)"/),
};

describe('the start-skew switch (PEARL_AMD_SKEW)', () => {
  it('is 0 unless a build sets it, and takes 0 to 3', () => {
    expect(FOLD).toContain('#ifndef PEARL_AMD_SKEW\n#define PEARL_AMD_SKEW 0\n#endif\n'
      + '#if PEARL_AMD_SKEW < 0 || PEARL_AMD_SKEW > 3\n#error "PEARL_AMD_SKEW is 0, 1, 2 or 3"\n#endif\n');
    const cond = grab(FOLD, /^#if (PEARL_AMD_SKEW < 0 \|\| PEARL_AMD_SKEW > 3)\n/m);
    const refused = (v) => new Function('PEARL_AMD_SKEW', 'return ' + cond + ';')(v);
    expect([-1, 0, 1, 2, 3, 4, 8].map(refused)).toEqual([true, false, false, false, false, true, true]);
  });

  it('leaves a release build as it was: at 0 nothing it adds is left', () => {
    // The token check is cl /EP's on every pass; here, what the switch guards. Only its default is left.
    for (const sw of [{}, gfx1100, gfx1102, gfx1201, { PEARL_AMD_EMU: 1 }]) {
      expect(code(pass(sw)).filter((l) => NAMES.test(l))).toEqual(['#ifndef PEARL_AMD_SKEW', '#define PEARL_AMD_SKEW 0']);
    }
    // The fold starts as it did: the slab count, then slab 0's load.
    expect(body(pass(gfx1100))).toContain('  const uint32_t slabs = k / (uint32_t)KS, perChunk = rank / (uint32_t)KS;\n'
      + '  PEARL_AMD_LOAD(0)\n  PEARL_AMD_STORE(0)\n  __syncthreads();\n');
    for (const sw of [{}, { PEARL_AMD_EMU: 1 }]) {
      expect(code(at(HOST, sw)).filter((l) => NAMES.test(l))).toEqual([]);
      expect(code(at(FOLDTEST, sw)).filter((l) => NAMES.test(l))).toEqual([]);
    }
  });

  it('can ship: no caps bit, and neither the host nor build-amd.sh refuses it', () => {
    expect(FOLD).not.toMatch(/CAP_SKEW/);
    for (const sw of [{}, gfx1100, gfx1201]) expect(caps(on(1, sw))).toBe(caps(pass(sw)));
    expect(BUILD).not.toContain('SKEW');
    // The host refuses only the timing builds' targets; nothing it says names the skew.
    expect(fn(HOST, 'extern "C" bool pearl_host_submit(')).not.toMatch(/SKEW|skew/);
    expect(code(at(HOST, { PEARL_AMD_SKEW: 2 })).filter((l) => /snprintf\(err|fail\(err/.test(l) && /skew/i.test(l))).toEqual([]);
  });
});

describe('the targets', () => {
  const cond = grab(FOLD, /^#if (.*)\n#define PEARL_AMD_SKEW_THREE 1\n#else\n#define PEARL_AMD_SKEW_THREE 0\n#endif$/m);
  // The header's test, for a target: its device pass defines __gfxNNNN__, and WMMA11 or WMMA12 by family.
  const three = (t) => new Function('PEARL_AMD_WMMA11', 'PEARL_AMD_WMMA12',
    'return (' + cond.replace(/defined\((__gfx\w+__)\)/g, (m, d) => String(d === `__${t}__`)) + ') ? 1 : 0;')(
    t.startsWith('gfx11') ? 1 : 0, t.startsWith('gfx12') ? 1 : 0);

  it('are the RDNA targets whose WGP holds three blocks of the fold, as README.md\'s compile table has them', () => {
    // The table: target, fold, VGPRs, AGPRs, LDS, waves a SIMD. Three blocks of 8 waves on a WGP's 4 SIMDs is 6.
    const rows = README.split('\n').map((l) => /^\| (gfx[\w, ]+) \| (wmma1[12]) \| \d+ \| - \| 36 KB \| (\d+) \|$/.exec(l))
      .filter(Boolean);
    const waves = {};
    for (const [, ts, , w] of rows) for (const t of ts.split(', ')) waves[t] = Number(w);
    expect(waves).toEqual({ gfx1100: 6, gfx1101: 6, gfx1102: 4, gfx1150: 4, gfx1151: 6, gfx1200: 6, gfx1201: 6 });
    for (const t of Object.keys(waves)) expect([t, three(t)]).toEqual([t, waves[t] === 6 ? 1 : 0]);
    // Every RDNA 3 and RDNA 4 target a build makes is in the table.
    const archs = grab(BUILD, /^DEFAULT_ARCHS="(.*)"$/m).split(' ').filter((t) => /^gfx1[12]/.test(t));
    expect(archs.sort()).toEqual(Object.keys(waves).sort());
    // No other pass: CDNA, RDNA 2, the host.
    for (const t of ['gfx942', 'gfx1030', 'host']) expect(three(t)).toBe(0);
  });

  it('gives a two-block target no skew in its fold, but the symbols the host sets', () => {
    expect(body(on(1, gfx1102))).toBe(body(pass(gfx1102)));
    expect(code(on(1, gfx1102)).join('\n')).not.toMatch(/pearl_amd_skew_(place|wait|se|wgp)\(/);
    // The host sets the symbols on any RDNA card (it can't see the target), so every code object must have them.
    for (const sw of [gfx1102, {}, gfx1201]) {
      expect(on(1, sw)).toContain('__constant__ uint32_t *pearl_amd_skew_ctr = nullptr;\n'
        + '__constant__ uint32_t pearl_amd_skew_first = 0u;\n'
        + '__constant__ uint32_t pearl_amd_skew_sleeps[PEARL_AMD_SKEW_WAYS] = {0u, 0u, 0u};');
    }
  });
});

describe('the waits', () => {
  it('are each place\'s, in percent of a slab: PEARL_AMD_SKEW_WAIT\'s, or the value\'s thirds, two thirds or sixths', () => {
    expect(PCT).toEqual([[0, 0, 0], [0, 33, 67], [0, 67, 133], [0, 17, 33]]);
    // gfx11's slab is the 7,000 clocks measured on a 7900 XTX; gfx12's is scaled from the fold's rate.
    expect([SLAB11, SLAB12, PCT_MAX]).toEqual([7000, 2800, 400]);
    expect(waits(null, 1, true)).toEqual({ set: false, pct: [0, 33, 67], sleeps: [0, 36, 73] });
    expect(waits(null, 2, true)).toEqual({ set: false, pct: [0, 67, 133], sleeps: [0, 73, 145] });
    expect(waits(null, 3, false)).toEqual({ set: false, pct: [0, 17, 33], sleeps: [0, 7, 14] });
    expect(waits('0:8:17', 1, true)).toEqual({ set: true, pct: [0, 8, 17], sleeps: [0, 9, 19] });
    expect(waits('67:67:67', 3, false)).toEqual({ set: true, pct: [67, 67, 67], sleeps: [29, 29, 29] });
    expect(waits('400:0:400', 1, true)).toEqual({ set: true, pct: [400, 0, 400], sleeps: [438, 0, 438] });
    // A count is 64 clocks: the nearest to the part of a slab asked for.
    for (const slab of [SLAB11, SLAB12]) {
      for (let p = 0; p <= PCT_MAX; p++) expect(Math.abs(COUNT(p, slab) * 64 - p * slab / 100)).toBeLessThanOrEqual(32);
    }
  });

  it('reads PEARL_AMD_SKEW_WAIT only as three whole numbers of at most 400 joined by colons', () => {
    for (const v of ['0:0:0', '1:2:3', '400:400:400', '007:8:017', '0:19:44']) expect([v, waits(v, 1, true).set]).toEqual([v, true]);
    for (const v of ['', ':', '1:2', '1:2:3:4', '1::3', ':1:2', '1:2:', '401:0:0', '0:0:1000', '1 :2:3', '1:2:3 ',
      '-1:2:3', '1.5:2:3', 'a:b:c', '0x1:2:3', '1:2:3\n']) {
      expect([v, waits(v, 1, true)]).toEqual([v, { set: false, pct: [0, 33, 67], sleeps: [0, 36, 73] }]);
    }
  });

  it('cost a miner\'s launch under 0.02% at SKEW 1, and fit any count in the loop', () => {
    // A launch's first blocks wait once; a WGP is late by at most its longest wait. A 7900 XTX at batch 128: 8192
    // blocks on 48 WGPs at 64 slabs; a 9070 XT: 16384 on 32 at k 4096, 128 slabs.
    for (const [wmma11, blocks, wgps, slabs] of [[true, 8192, 48, 64], [false, 16384, 32, 128]]) {
      const launch = blocks / wgps / WAYS * slabs * (wmma11 ? SLAB11 : SLAB12);
      expect(Math.max(...waits(null, 1, wmma11).sleeps) * 64 / launch).toBeLessThan(0.0002);
    }
    // The count is a loop's, not s_sleep's 7 bits: s_sleep(1) runs as many times as it says.
    expect(fn(on(1, gfx1100), 'void pearl_amd_skew_wait(uint32_t wave) {'))
      .toContain('  for (uint32_t i = 0; i < n; i++) __builtin_amdgcn_s_sleep(1);\n');
  });
});

describe('the place on a WGP', () => {
  const wgpC = grab(FOLD, /uint32_t pearl_amd_skew_wgp\(uint32_t hw, uint32_t se\) \{\n {2}return (.*);\n\}/);
  const wgp = (hw, se) => cBits(wgpC, { hw: hw >>> 0, se: se >>> 0 });
  const seFn = (sw) => code(fn(on(1, sw), 'uint32_t pearl_amd_skew_se(uint32_t hw) {'));

  it('takes the engine from HW_ID1 on gfx11, and from RTN_GET_SE_HW_ID on gfx12, whose HW_ID1 has none', () => {
    expect(seFn(gfx1100)).toEqual(['uint32_t pearl_amd_skew_se(uint32_t hw) {', '  return hw >> 18;', '}']);
    expect(seFn(gfx1201)).toEqual(['uint32_t pearl_amd_skew_se(uint32_t hw) {', '  (void)hw;',
      '  return (uint32_t)__builtin_amdgcn_s_sendmsg_rtn(0x87);', '}']);
    // HW_ID1 is hwreg 23, all 32 bits, on both.
    for (const sw of [gfx1100, gfx1201]) {
      expect(fn(on(1, sw), 'uint32_t pearl_amd_hw_id1() {')).toContain('return (uint32_t)__builtin_amdgcn_s_getreg(0xF817);');
    }
    expect({ id: 0xF817 & 63, offset: (0xF817 >> 6) & 31, size: (0xF817 >> 11) + 1 }).toEqual({ id: 23, offset: 0, size: 32 });
  });

  it('is one counter for each engine, array and WGP, whatever the wave and SIMD', () => {
    expect([WGPS, WAYS]).toEqual([256, 3]);
    const seen = new Set();
    let rng = 12345;
    const rand = () => { rng = (rng * 1103515245 + 12345) >>> 0; return rng; };
    for (let se = 0; se < 8; se++) {
      for (let sa = 0; sa < 2; sa++) {
        for (let w = 0; w < 16; w++) {
          const hw = (se << 18) | (sa << 16) | (w << 10);
          const i = wgp(hw, hw >>> 18);   // gfx11
          expect(i).toBe(se * 32 + sa * 16 + w);
          seen.add(i);
          // WAVE_ID, SIMD_ID and the other bits don't move it; on gfx12 the engine is the message's, not HW_ID1's.
          for (let j = 0; j < 8; j++) {
            const other = rand() & ~((7 << 18) | (1 << 16) | (15 << 10));
            expect(wgp((hw | other) >>> 0, se)).toBe(i);
            expect(wgp(((sa << 16) | (w << 10) | rand() & ~((1 << 16) | (15 << 10))) >>> 0, se | (rand() & 0xF00))).toBe(i);
          }
        }
      }
    }
    expect(seen.size).toBe(WGPS);
    expect(Math.max(...seen)).toBe(WGPS - 1);
    // A 9070 XT: 4 engines of 2 arrays of 4 WGPs, 32 counters. Its HW_ID1 bits 20:18 aren't the engine; had the index
    // taken them (as gfx11's), and had they read 0, its 32 WGPs would share 8 counters.
    const navi48 = [];
    for (let se = 0; se < 4; se++) for (let sa = 0; sa < 2; sa++) for (let w = 0; w < 4; w++) navi48.push([se, (sa << 16) | (w << 10)]);
    expect(new Set(navi48.map(([se, hw]) => wgp(hw, se))).size).toBe(32);
    expect(new Set(navi48.map(([, hw]) => wgp(hw, hw >>> 18))).size).toBe(8);
  });

  it('gives a WGP\'s three first blocks three places every launch, with no reset', () => {
    // atomicInc(p, v): the old value back; v or more becomes 0, else old + 1. The fold's is v = 2, and % 3 after.
    const place = fn(on(1, gfx1201), 'uint32_t pearl_amd_skew_place(uint32_t wave) {');
    expect(place).toContain('  if (threadIdx.x == 0u) ticket = atomicInc(ctr + at, PEARL_AMD_SKEW_WAYS - 1u);\n'
      + '  return pearl_amd_sgpr(ticket) % PEARL_AMD_SKEW_WAYS;');
    const inc = (c, i, v) => { const old = c[i]; c[i] = old >= v ? 0 : old + 1; return old; };
    let rng = 7;
    const rand = (n) => { rng = (rng * 1103515245 + 12345) >>> 0; return (rng >>> 8) % n; };
    // One launch's first blocks: `per` a WGP on 48 WGPs (counter g * 4), arriving in any order, or only the first
    // `upto` of them. Each WGP's places, in the order its blocks took them.
    const launch = (c, per, upto) => {
      const arrivals = [];
      for (let g = 0; g < 48; g++) for (let b = 0; b < per; b++) arrivals.push(g * 4);
      arrivals.splice(upto);
      for (let i = arrivals.length - 1; i > 0; i--) {
        const j = rand(i + 1);
        [arrivals[i], arrivals[j]] = [arrivals[j], arrivals[i]];
      }
      const places = {};
      for (const g of arrivals) (places[g] = places[g] || []).push(inc(c, g, WAYS - 1) % WAYS);
      return Object.values(places);
    };
    // The host zeroes the counters once. A launch's first wave fills each WGP with three: places 0, 1 and 2, and
    // every counter is back at 0 after it, launch after launch, with nothing reset.
    const c = Array(WGPS).fill(0);
    for (let n = 0; n < 20; n++) {
      for (const ps of launch(c, 3, 144)) expect(ps.slice().sort()).toEqual([0, 1, 2]);
      expect(c.every((x) => x === 0)).toBe(true);
    }
    // A small launch (the fold test's 1024 x 1024 check: 32 blocks) leaves the counters anywhere in 0-2. Any three
    // tickets in a row on a WGP still differ.
    for (let n = 0; n < 40; n++) {
      const small = n % 5 === 2;
      for (const ps of launch(c, 3, small ? 32 : 144)) {
        for (const p of ps) expect(p).toBeLessThan(WAYS);
        if (!small) expect(ps.slice().sort()).toEqual([0, 1, 2]);
      }
      expect(c.every((x) => x >= 0 && x < WAYS)).toBe(true);
    }
    // A counter that held anything else (never zeroed) still gives a place in 0-2, and is in range after one ticket.
    const g = [1000];
    expect(inc(g, 0, WAYS - 1) % WAYS).toBe(1);
    expect(g[0]).toBe(0);
  });
});

describe('the skew in the fold', () => {
  it('waits once, before slab 0\'s load, and leaves the slab loop as the release has it', () => {
    for (const sw of [gfx1100, gfx1201]) {
      for (const s of [1, 2, 3]) {
        const b = body(on(s, sw));
        const call = '  pearl_amd_skew_wait((uint32_t)wave);\n  PEARL_AMD_LOAD(0)\n';
        expect(b.split('pearl_amd_skew_wait(').length - 1).toBe(1);
        expect(b).toContain(call);
        expect(b.indexOf(call)).toBeLessThan(b.indexOf('auto slab_step'));
        expect(slabStep(on(s, sw))).toBe(slabStep(pass(sw)));
        // Take the skew's lines out of the body and it is the release's.
        expect(code(b).filter((l) => !/skew|PEARL_AMD_SKEW/.test(l))).toEqual(code(body(pass(sw))));
      }
    }
  });

  it('is only in the device passes of the skew\'s targets: not the host, CDNA, RDNA 2, gfx1102 or HIP-CPU', () => {
    for (const sw of [{}, gfx1102]) {
      expect(code(body(on(1, sw))).join('\n')).not.toMatch(/skew|SKEW/);
      expect(code(on(1, sw)).join('\n')).not.toMatch(/pearl_amd_skew_wait\(/);
    }
    const emu = code(on(1, { PEARL_AMD_EMU: 1 })).join('\n');
    expect(emu).not.toMatch(/pearl_amd_skew_(ctr|first|sleeps|waits|wait|place)|PEARL_AMD_SKEW_(WAYS|COUNT)/);
    // The reference fold is the same with it on.
    const ref = (t) => fn(t, 'void pearl_amd_fold_ref(');
    expect(ref(on(1, gfx1100))).toBe(ref(pass(gfx1100)));
  });

  it('takes its place and its count in SGPRs, and the ticket\'s VGPRs only in the prologue', () => {
    expect(code(fn(on(1, gfx1201), 'uint32_t pearl_amd_skew_place(uint32_t wave) {'))).toEqual([
      'uint32_t pearl_amd_skew_place(uint32_t wave) {',
      '  uint32_t *const ctr = pearl_amd_skew_ctr;',
      '  const uint32_t block = blockIdx.y * gridDim.x + blockIdx.x;',
      '  if (pearl_amd_sgpr(wave) != 0u || !ctr || block >= pearl_amd_skew_first) return PEARL_AMD_SKEW_WAYS;',
      '  const uint32_t hw = pearl_amd_hw_id1();',
      '  const uint32_t at = pearl_amd_sgpr(pearl_amd_skew_wgp(hw, pearl_amd_skew_se(hw)));',
      '  uint32_t ticket = 0u;',
      '  if (threadIdx.x == 0u) ticket = atomicInc(ctr + at, PEARL_AMD_SKEW_WAYS - 1u);',
      '  return pearl_amd_sgpr(ticket) % PEARL_AMD_SKEW_WAYS;   // 0, 1 or 2, whatever the counter held',
      '}',
    ]);
    expect(code(fn(on(1, gfx1100), 'void pearl_amd_skew_wait(uint32_t wave) {'))).toEqual([
      'void pearl_amd_skew_wait(uint32_t wave) {',
      '  const uint32_t place = pearl_amd_skew_place(wave);',
      '  const uint32_t n = place < PEARL_AMD_SKEW_WAYS ? pearl_amd_skew_sleeps[place] : 0u;',
      '  for (uint32_t i = 0; i < n; i++) __builtin_amdgcn_s_sleep(1);',
      '}',
    ]);
    // pearl_amd_sgpr is readfirstlane on both RDNA passes.
    for (const sw of [gfx1100, gfx1201]) {
      expect(fn(on(1, sw), 'uint32_t pearl_amd_sgpr(uint32_t x) {')).toContain('  return (uint32_t)__builtin_amdgcn_readfirstlane((int)x);\n}');
    }
    // The block's linear number is the order the dispatcher hands blocks out in: x first, as the stamps number them.
    expect(FOLD).toContain('b = blockIdx.y * gridDim.x + blockIdx.x');
  });

  it('asserts three blocks a WGP, which the release\'s LDS gives and a 64-byte slab doesn\'t', () => {
    const b = body(on(1, gfx1100));
    expect(b).toContain('  static_assert(PEARL_AMD_SKEW_WAYS * (sizeof sA + sizeof sB) <= 131072u\n'
      + '                && (PEARL_AMD_SKEW_WAYS + 1u) * (sizeof sA + sizeof sB) > 131072u,');
    // sA and sB: two buffers each of 128 and 256 rows, a row the slab plus 16 bytes.
    const lds = (slabBytes) => 2 * (128 + 256) * (slabBytes + 16);
    const fits = (slabBytes) => 3 * lds(slabBytes) <= 131072 && 4 * lds(slabBytes) > 131072;
    expect(lds(32)).toBe(36864);
    expect([fits(32), fits(64)]).toEqual([true, false]);
    expect(FOLD).toContain('#define PEARL_AMD_SLAB 32u');
  });
});

describe('the host and the fold test', () => {
  const host = at(HOST, { PEARL_AMD_SKEW: 1, PEARL_AMD_EMU: 0 });
  const create = fn(host, 'extern "C" void *pearl_host_create(');
  const destroy = fn(host, 'extern "C" void pearl_host_destroy(void *handle) {');

  it('gives the RDNA folds zeroed counters, three blocks a WGP and the waits, once a context', () => {
    expect(host).toContain('  uint32_t *dSkew = nullptr;    // PEARL_AMD_SKEW: the RDNA folds\' counters, one a WGP (pearl_amd_fold.h)');
    const set = slice(create, '    const uint32_t cap = kFolds[ctx->fold].cap;', '    (void)hipGetLastError();\n');
    expect(code(set)).toEqual([
      '    const uint32_t cap = kFolds[ctx->fold].cap;',
      '    hipDeviceProp_t prop;',
      '    if ((cap == PEARL_AMD_CAP_WMMA11 || cap == PEARL_AMD_CAP_WMMA12)',
      '        && hipGetDeviceProperties(&prop, ctx->device) == hipSuccess && prop.multiProcessorCount > 0) {',
      '      const uint32_t first = PEARL_AMD_SKEW_WAYS * (uint32_t)prop.multiProcessorCount;',
      '      uint32_t pct[PEARL_AMD_SKEW_WAYS], sleeps[PEARL_AMD_SKEW_WAYS];',
      '      pearl_amd_skew_waits(getenv("PEARL_AMD_SKEW_WAIT"), cap == PEARL_AMD_CAP_WMMA11, pct, sleeps);',
      '      HIP_OK(hipMalloc(&ctx->dSkew, PEARL_AMD_SKEW_WGPS * sizeof(uint32_t)), "allocating the skew counters");',
      '      HIP_OK(hipMemset(ctx->dSkew, 0, PEARL_AMD_SKEW_WGPS * sizeof(uint32_t)), "zeroing the skew counters");',
      '      HIP_OK(hipMemcpyToSymbol(HIP_SYMBOL(pearl_amd_skew_sleeps), sleeps, sizeof sleeps), "setting the skew\'s waits");',
      '      HIP_OK(hipMemcpyToSymbol(HIP_SYMBOL(pearl_amd_skew_first), &first, sizeof first),',
      '             "setting the skew\'s first blocks");',
      '      HIP_OK(hipMemcpyToSymbol(HIP_SYMBOL(pearl_amd_skew_ctr), &ctx->dSkew, sizeof ctx->dSkew),',
      '             "pointing the folds at the skew counters");',
      '    }',
      '    (void)hipGetLastError();',
    ]);
    // After the fold is chosen; the counters zeroed and the waits set before the folds are pointed at the counters.
    expect(create.indexOf('resolve_fold(ctx);')).toBeLessThan(create.indexOf('kFolds[ctx->fold].cap'));
    expect(set.indexOf('hipMemset(ctx->dSkew')).toBeLessThan(set.indexOf('pearl_amd_skew_ctr'));
    expect(set.indexOf('pearl_amd_skew_sleeps')).toBeLessThan(set.indexOf('pearl_amd_skew_ctr'));
    // Nothing a launch: the submit path has none of it.
    expect(fn(host, 'extern "C" bool pearl_host_submit(')).not.toMatch(/skew|Skew/);
  });

  it('points the folds away from the counters before it frees them', () => {
    expect(code(slice(destroy, '  if (ctx->dSkew) {', '\n  }\n'))).toEqual([
      '  if (ctx->dSkew) {   // no fold may take a ticket from it once it is freed',
      '    uint32_t *none = nullptr;',
      '    hipMemcpyToSymbol(HIP_SYMBOL(pearl_amd_skew_ctr), &none, sizeof none);',
      '    hipFree(ctx->dSkew);',
      '  }',
    ]);
    expect(code(at(HOST, { PEARL_AMD_SKEW: 1, PEARL_AMD_EMU: 1 })).filter((l) => /skew|Skew/.test(l))).toEqual([]);
  });

  it('the fold test sets them as the host does, before any launch, says what, and runs the probe', () => {
    const t = at(FOLDTEST, { PEARL_AMD_SKEW: 3, PEARL_AMD_EMU: 0 });
    const set = slice(t, '  uint32_t *dSkew = nullptr;\n', '\n  } else {\n');
    expect(set).toContain('    const bool wmma11 = (caps[0] & PEARL_AMD_CAP_WMMA11) != 0u;\n'
      + '    const uint32_t first = PEARL_AMD_SKEW_WAYS * (uint32_t)prop.multiProcessorCount;\n');
    expect(set).toContain('const bool fromEnv = pearl_amd_skew_waits(getenv("PEARL_AMD_SKEW_WAIT"), wmma11, pct, sleeps);');
    expect(set.indexOf('hipMemset(dSkew, 0, PEARL_AMD_SKEW_WGPS * 4)')).toBeLessThan(set.indexOf('pearl_amd_skew_ctr'));
    expect(set).toContain('HIP_SYMBOL(pearl_amd_skew_sleeps), sleeps');
    expect(set).toContain('HIP_SYMBOL(pearl_amd_skew_first), &first');
    expect(set.indexOf('printf("skew: ')).toBeLessThan(set.indexOf('    if (!skew_probe(first)) return 2;'));
    // After the caps are read, before the first launch of a fold (the check's).
    expect(t.indexOf('hipLaunchKernelGGL(pearl_amd_caps')).toBeLessThan(t.indexOf('  uint32_t *dSkew = nullptr;'));
    expect(t.indexOf('  uint32_t *dSkew = nullptr;')).toBeLessThan(t.indexOf('    launch(f, grid, fa, fb'));
    expect(t).toContain('  if (dSkew) {\n    uint32_t *none = nullptr;\n'
      + '    hipMemcpyToSymbol(HIP_SYMBOL(pearl_amd_skew_ctr), &none, sizeof none);\n    hipFree(dSkew);\n  }\n');
    expect(code(at(FOLDTEST, { PEARL_AMD_SKEW: 3, PEARL_AMD_EMU: 1 })).filter((l) => /skew|Skew/.test(l))).toEqual([]);
    // Its lines, as the C prints them.
    expect(sprintf(FMT.skew, 1, 'wmma11', 144, 0, 8, 17, 7000, 0, 9, 19, 'from PEARL_AMD_SKEW_WAIT')).toBe(
      'skew: PEARL_AMD_SKEW=1 on wmma11; first 144 blocks; waits 0:8:17% of a 7000-clock slab (0:9:19 sleeps), '
      + 'from PEARL_AMD_SKEW_WAIT\n');
    expect(sprintf(FMT.probe, 144, 144, 48, 3, 3, 48, FMT.designed)).toBe(
      'skew probe: 144 blocks, 144 took a place, on 48 counters of 3 to 3 blocks, 48 with places 0, 1 and 2: as designed\n');
    // sweep.sh's check line takes the fold's line, PASS and FAIL, never these.
    const sweepGrep = new RegExp(grab(SWEEP, /chk=\$\(printf '%s\\n' "\$cout" \| grep -E "(\^\(\$FOLD\|PASS\|FAIL\|no fold\))"/)
      .replace('$FOLD', 'wmma11'));
    expect('PASS').toMatch(sweepGrep);
    for (const l of [FMT.skew, FMT.probe, FMT.none]) expect(sprintf(l, 1, 'wmma11', 1, 1, 1, 1, 1, 1, 1, 1, 'x')).not.toMatch(sweepGrep);
  });

  it('the probe launches a launch\'s first blocks with the fold\'s threads and LDS, each taking a place as the fold does', () => {
    const t = at(FOLDTEST, { PEARL_AMD_SKEW: 1, PEARL_AMD_EMU: 0, PEARL_AMD_SKEW_THREE: 1 });
    expect(code(fn(t, 'extern "C" __global__ __launch_bounds__(PEARL_AMD_THREADS) void pearl_amd_skew_probe('))).toEqual([
      'extern "C" __global__ __launch_bounds__(PEARL_AMD_THREADS) void pearl_amd_skew_probe(uint32_t *out, uint32_t hold) {',
      '  const uint32_t wave = threadIdx.x / 32u;',
      '  const uint32_t place = pearl_amd_skew_place(wave);',
      '  if (wave != 0u) return;',
      '  const uint32_t hw = pearl_amd_hw_id1();',
      '  const uint32_t at = pearl_amd_skew_wgp(hw, pearl_amd_skew_se(hw));',
      '  if (threadIdx.x == 0u) out[blockIdx.x] = place < PEARL_AMD_SKEW_WAYS ? (at << 8) | place : ~0u;',
      '  for (uint32_t i = 0; i < hold; i++) __builtin_amdgcn_s_sleep(127);',
      '}',
    ]);
    // A target without the skew: every block says no place.
    expect(fn(at(FOLDTEST, { PEARL_AMD_SKEW: 1, PEARL_AMD_EMU: 0, PEARL_AMD_SKEW_THREE: 0 }),
      'void pearl_amd_skew_probe(')).toContain('  (void)hold;\n  if (threadIdx.x == 0u) out[blockIdx.x] = ~0u;\n}');
    const probe = fn(t, 'bool skew_probe(uint32_t first) {');
    expect(probe).toContain('hipLaunchKernelGGL(pearl_amd_skew_probe, dim3(first), dim3(PEARL_AMD_THREADS), lds, 0, dOut, hold);');
    // Its LDS is the fold's sA and sB, so the same three blocks fit a WGP.
    const lds = grab(probe, /const uint32_t lds = (.*);/);
    const n = (name) => BigInt(grab(FOLD, new RegExp('^#define ' + name + ' (\\d+)u$', 'm')));
    expect(Number(cExpr(lds, { PEARL_AMD_BM: n('PEARL_AMD_BM'), PEARL_AMD_BN: n('PEARL_AMD_BN'),
      PEARL_AMD_SLAB: n('PEARL_AMD_SLAB') }))).toBe(36864);
    // Wave 0 holds its WGP `hold` sleeps of 127 counts, over 0.1 ms even at 3 GHz: far longer than the dispatcher
    // takes to hand out a launch's first blocks, so they are all there at once.
    const hold = Number(grab(probe, /const uint32_t hold = (\d+)u;/));
    const count = Number(grab(fn(t, 'void pearl_amd_skew_probe('), /__builtin_amdgcn_s_sleep\((\d+)\);/));
    expect((hold * (count - 1) * 64) / 3e9).toBeGreaterThan(1e-4);
  });

  it('sums the probe\'s words, and calls them as designed only with three blocks a counter, places 0, 1 and 2', () => {
    const word = (counter, place) => ((counter << 8) | place) >>> 0;
    const NONE = 0xFFFFFFFF;
    // A 7900 XTX as designed: 48 WGPs, three blocks each, places in any order.
    const xtx = [];
    for (let g = 0; g < 48; g++) for (const p of [2, 0, 1]) xtx.push(word(g * 3 + 1, p));
    expect(probeSum(xtx)).toEqual([144, 48, 3, 3, 48]);
    expect(designed(probeSum(xtx), 144)).toBe(true);
    // A 9070 XT whose index lost the engine: 32 WGPs on 8 counters, 12 tickets each, so places repeat.
    const lost = [];
    for (let i = 0; i < 96; i++) lost.push(word(i % 8, Math.floor(i / 8) % 3));
    expect(probeSum(lost)).toEqual([96, 8, 12, 12, 0]);
    expect(designed(probeSum(lost), 96)).toBe(false);
    // HIP's count as CUs: twice the blocks, six a counter.
    const cus = xtx.concat(xtx);
    expect(probeSum(cus)).toEqual([288, 48, 6, 6, 0]);
    expect(designed(probeSum(cus), 288)).toBe(false);
    // A WGP whose blocks took places 0, 0 and 1, and a block with no place.
    const bad = xtx.slice();
    bad[2] = word(1, 0);
    expect(probeSum(bad)).toEqual([144, 48, 3, 3, 47]);
    expect(designed(probeSum(bad), 144)).toBe(false);
    const missing = xtx.slice();
    missing[0] = NONE;
    expect(probeSum(missing)).toEqual([143, 48, 2, 3, 47]);
    expect(designed(probeSum(missing), 144)).toBe(false);
    // None placed: a target without the skew, which the line says apart.
    expect(probeSum(Array(48).fill(NONE)).slice(0, 2)).toEqual([0, 0]);
    expect(sprintf(FMT.none, 48)).toBe('skew probe: 48 blocks, none took a place: this target has no skew\n');
  });
});

describe('test/skew-race.sh', () => {
  let tmp;
  beforeEach(() => { tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'amd-skew-race-')); });
  afterEach(() => { fs.rmSync(tmp, { recursive: true, force: true }); });

  // Stand-ins. variant-check.sh keeps STUB_KEEP. sweep.sh prints STUB_SWEEP_OUT and leaves a core and a fold test for
  // each build. The fold test prints the C's own lines (FMT) and reads PEARL_AMD_SKEW_WAIT with the C's own reading
  // (WAITS_SRC); its fold rates come from the size, STUB_R1 and STUB_R2, and STUB_GAIN[size][waits] by round.
  // hashrate.js reads STUB_TH[base or waits] in turn; verify-hits.js answers STUB_VERIFY. Each logs what it was given.
  const VC = [
    '#!/bin/bash',
    'echo "variant-check VARIANTS=$VARIANTS TWINS=$TWINS ARCH=$ARCH CHECK_ROOT=$CHECK_ROOT" >> "$STUB_LOG"',
    ': > "$KEEP_TO"',
    'IFS=\';\' read -ra VS <<< "$VARIANTS"',
    'for v in "${VS[@]}"; do case " $STUB_KEEP " in *" ${v%%=*} "*) echo "$v" >> "$KEEP_TO" ;; esac; done',
    '',
  ].join('\n');
  const SW = [
    '#!/bin/bash',
    'echo "sweep SWEEP=$SWEEP FOLD=${SWEEP_FOLD-unset} ROUNDS=$SWEEP_ROUNDS ARCH=$ARCH" >> "$STUB_LOG"',
    'IFS=\';\' read -ra VS <<< "$SWEEP"',
    'for v in "${VS[@]}"; do',
    '  n=${v%%=*}; mkdir -p "$SWEEP_ROOT/var/$n"; : > "$SWEEP_ROOT/var/$n/pearl_core_hip.node"',
    '  cp "$STUB_FOLDTEST" "$SWEEP_ROOT/var/$n/pearl_amd_foldtest"',
    'done',
    'cat "$STUB_SWEEP_OUT"',
    '',
  ].join('\n');
  const FT_JS = [
    '\'use strict\';',
    'const fs = require(\'fs\');',
    'const path = require(\'path\');',
    SPRINTF,
    'const E = process.env;',
    'const [self, ...args] = process.argv.slice(2);',
    'const build = path.basename(path.dirname(self));',
    'const [m, cols, , reps, check, fold] = args;',
    'const M = Number(m), C = Number(cols), W = Number(E.STUB_WGPS), F = JSON.parse(E.STUB_FMT);',
    'const wait = E.PEARL_AMD_SKEW_WAIT;',
    'fs.appendFileSync(E.STUB_LOG, `foldtest ${build} ${args.join(\' \')} wait=${wait === undefined ? \'-\' : wait}\\n`);',
    'const wmma11 = fold === \'wmma11\';',
    'const out = [sprintf(F.device, \'AMD Radeon RX\', E.STUB_ARCH, W, wmma11 ? 2048 : 4096, wmma11 ? \'\' : \', wrap readout\')];',
    'let key = \'\';',
    'if (build !== \'base\') {',
    '  const pct = [], sleeps = [];',
    '  const env = E.STUB_IGNORE_ENV || wait === undefined ? null : wait + \'\\0\';',
    '  const [s11, s12, max] = JSON.parse(E.STUB_SLABS);',
    '  const set = new Function(...JSON.parse(E.STUB_WAITS_ARGS), E.STUB_WAITS_SRC)(env, wmma11, pct, sleeps,',
    '    JSON.parse(E.STUB_PCT), 1, (p, s) => Math.floor((p * s + 3200) / 6400), s11, s12, max);',
    '  key = pct.join(\':\');',
    '  out.push(sprintf(F.skew, 1, fold, 3 * W, ...pct, wmma11 ? s11 : s12, ...sleeps, set ? F.fromEnv : F.build));',
    '  const p = E.STUB_PROBE || \'designed\';',
    '  if (p === \'none\') out.push(sprintf(F.none, 3 * W));',
    '  else if (p === \'lost\') out.push(sprintf(F.probe, 3 * W, 3 * W, 8, 3 * W / 8, 3 * W / 8, 0, F.notDesigned));',
    '  else out.push(sprintf(F.probe, 3 * W, 3 * W, W, 3, 3, W, F.designed));',
    '}',
    'const size = M === 131072 ? \'mined\' : M === 128 * W ? \'one-gen\' : M === 1024 * W ? \'eight-gen\' : \'other\';',
    'const R1 = Number(E.STUB_R1 || 1.8), R2 = Number(E.STUB_R2 || 1.3);',
    'let rate = { mined: 58, \'eight-gen\': 55, other: 40 }[size]',
    '  || (C === 768 ? 50 : C === 512 ? 50 * 2 / 3 * R2 : 50 / 3 * R1);',
    'if (size === \'one-gen\' && C !== 768 && E.STUB_NORATE) rate = 0;',
    'if (key) {',
    '  const f = path.join(E.STUB_STATE, `fold-${size}-${key.replace(/:/g, \'-\')}`);   // no colons in a Windows name',
    '  const i = fs.existsSync(f) ? Number(fs.readFileSync(f, \'utf8\')) : 0;',
    '  fs.writeFileSync(f, String(i + 1));',
    '  const g = ((JSON.parse(E.STUB_GAIN || \'{}\')[size] || {})[key] || [])[i] || 0;',
    '  rate *= 1 + g / 100;',
    '}',
    'const regions = (M / 16) * (C / 16);',
    'if (check === \'0\') {',
    '  if (rate) out.push(sprintf(F.timed, fold, regions, Number(reps), 0.5, rate));',
    '} else if ((E.STUB_BIGFAIL || \'\').split(\' \').includes(build)) {',
    '  out.push(`${fold} FAIL: 3 of 1048576 words differ`, \'FAIL\');',
    '} else {',
    '  out.push(sprintf(F.ok, fold, regions, 1, 0.1, rate), \'PASS\');',
    '}',
    'process.stdout.write(out.map((l) => l.replace(/\\n$/, \'\')).join(\'\\n\') + \'\\n\');',
    '',
  ].join('\n');
  const HR = [
    '\'use strict\';',
    'const fs = require(\'fs\');',
    'const path = require(\'path\');',
    'const n = path.basename(path.dirname(process.argv[2]));',
    'const w = process.env.PEARL_AMD_SKEW_WAIT;',
    'const key = n === \'base\' ? \'base\' : w;',
    'const f = path.join(process.env.STUB_STATE, \'hr-\' + String(key).replace(/:/g, \'-\'));',
    'const i = fs.existsSync(f) ? Number(fs.readFileSync(f, \'utf8\')) : 0;',
    'fs.writeFileSync(f, String(i + 1));',
    'fs.appendFileSync(process.env.STUB_LOG, `hashrate ${n} ${process.argv.slice(3).join(\' \')} wait=${w}\\n`);',
    'const th = (JSON.parse(process.env.STUB_TH)[key] || [])[i];',
    'console.log(th == null ? JSON.stringify({ error: \'no device\' }) : JSON.stringify({ file: n + \'/pearl_core_hip.node\', k: 2048, th }));',
    '',
  ].join('\n');
  const VH = [
    '\'use strict\';',
    'const fs = require(\'fs\');',
    'const path = require(\'path\');',
    'const n = path.basename(path.dirname(process.argv[2]));',
    'const dump = process.env.PEARL_SHARE_DUMP;',
    'fs.appendFileSync(process.env.STUB_LOG, `verify ${n} ${process.argv[3]} ${dump} wait=${process.env.PEARL_AMD_SKEW_WAIT}'
      + ' existed=${fs.existsSync(dump)}\\n`);',
    'console.log(\'core opened\');',
    'console.log(process.env.STUB_VERIFY.replace(\'NAME\', n));',
    '',
  ].join('\n');
  // sweep.sh's check lines for base and skew.
  const sweepOut = (baseOk = true, skewOk = true) => [['base', baseOk, ''], ['skew', skewOk, '-DPEARL_AMD_SKEW=1']]
    .map(([n, ok, f]) => `[amdtest] 00:00:00 sweep ${n} built in 80s (${f}); `
      + `check: wmma11 ok: 4096 regions, all 16 words match ${ok ? 'PASS' : 'FAIL'} `).join('\n') + '\n';

  // Runs a copy of the script from a stand-in earn/native/amd in a directory of its own.
  function run(env, { sweep = sweepOut(), shares } = {}) {
    const dir = fs.mkdtempSync(path.join(tmp, 'r-'));
    const p = (...f) => path.join(dir, ...f);
    const fwd = (...f) => p(...f).replace(/\\/g, '/');
    for (const d of [['amd', 'test'], ['probes'], ['bin'], ['state'], ['root', 'shares']]) fs.mkdirSync(p(...d), { recursive: true });
    fs.writeFileSync(p('amd', 'test', 'skew-race.sh'), RACE.replace('/proc/1/fd/1', '/dev/null'));
    fs.writeFileSync(p('amd', 'test', 'variant-check.sh'), VC);
    fs.writeFileSync(p('amd', 'test', 'sweep.sh'), SW);
    fs.writeFileSync(p('foldtest.js'), FT_JS);
    fs.writeFileSync(p('foldtest'), '#!/bin/bash\nexec node "$STUB_FT_JS" "$0" "$@"\n', { mode: 0o755 });
    fs.writeFileSync(p('probes', 'hashrate.js'), HR);
    fs.writeFileSync(p('probes', 'verify-hits.js'), VH);
    // Git Bash on Windows can find Windows' own timeout.exe first: this one runs the command as it is.
    fs.writeFileSync(p('bin', 'timeout'), '#!/bin/bash\nshift\nexec "$@"\n', { mode: 0o755 });
    fs.writeFileSync(p('sweep.txt'), sweep);
    if (shares) for (const [f, text] of Object.entries(shares)) fs.writeFileSync(p('root', 'shares', f), text);
    const e = { ...process.env, RUN_ROOT: fwd('root'), STUB_LOG: fwd('log.txt'), STUB_STATE: p('state'),
      STUB_FOLDTEST: fwd('foldtest'), STUB_FT_JS: p('foldtest.js'), STUB_SWEEP_OUT: fwd('sweep.txt'),
      STUB_FMT: JSON.stringify(FMT), STUB_WAITS_SRC: WAITS_SRC, STUB_WAITS_ARGS: JSON.stringify(WAITS_ARGS),
      STUB_PCT: JSON.stringify(PCT), STUB_SLABS: JSON.stringify([SLAB11, SLAB12, PCT_MAX]), STUB_WGPS: '48', STUB_ARCH: 'gfx1100', SKEW_CLASS: '', STUB_KEEP: 'base skew',
      STUB_TH: '{}', STUB_VERIFY: '{"file":"NAME","k":2048,"hits":731,"verified":400,"failed":0,"salts":95,"PASS":true}',
      ...env };
    for (const k of ['SKEW_FOLD', 'SKEW_RACE', 'SKEW_VERIFY', 'SKEW_EXTRA', 'PEARL_AMD_SKEW_WAIT']) if (!(k in env)) delete e[k];
    // The stand-in timeout and this node first. Windows spells the variable Path.
    const pathKey = Object.keys(e).find((k) => k.toUpperCase() === 'PATH') || 'PATH';
    e[pathKey] = [p('bin'), path.dirname(process.execPath), e[pathKey] || ''].join(path.delimiter);
    const r = spawnSync('bash', ['test/skew-race.sh'], { cwd: p('amd'), encoding: 'utf8', env: e });
    const log = fs.existsSync(p('log.txt')) ? fs.readFileSync(p('log.txt'), 'utf8') : '';
    return {
      status: r.status,
      err: r.stderr,
      lines: r.stdout.split('\n').filter((l) => / skew /.test(l)).map((l) => l.replace(/^\[amdtest\] \S+ skew /, '')),
      log: log.split('\n').filter(Boolean),
      root: fwd('root'),
      shares: (f) => (fs.existsSync(p('root', 'shares', f)) ? fs.readFileSync(p('root', 'shares', f), 'utf8') : null),
    };
  }
  const folds = (r, size) => r.log.filter((l) => l.startsWith('foldtest') && l.includes(size));

  bashIt('parses', () => {
    expect(spawnSync('bash', ['-n', path.join(AMD, 'test', 'skew-race.sh')]).status).toBe(0);
  });

  // RDNA 3. r1 1.8 and r2 1.3 aim w17 at 0:9:22 and w33 at 0:19:44. At eight generations w17, w33 and t17 gain most;
  // at the mined launch w17 and t17. In hashrate.js w17 has the larger mean but one round under 0.3%, t17 passes.
  const GAIN = {
    'one-gen': { '0:8:17': [2, 0.2, 0.2, 0.2], '0:17:33': [0.4, 0.4, 0.4, 0.4], '0:33:67': [0, 0, 0, 0],
      '0:9:22': [0.8, 0.8, 0.8, 0.8], '0:19:44': [0.6, 0.6, 0.6, 0.6], '67:67:67': [-1, -1, -1, -1] },
    'eight-gen': { '0:8:17': [0.1, 0.1, 0.1, 0.1], '0:17:33': [0.3, 0.3, 0.3, 0.3], '0:33:67': [0.2, 0.2, 0.2, 0.2],
      '0:9:22': [0.5, 0.5, 0.5, 0.5], '0:19:44': [0.4, 0.4, 0.4, 0.4], '67:67:67': [3, 3, 3, 3] },
    mined: { '0:17:33': [0.35, 0.35, 0.35, 0.35], '0:9:22': [0.4, 0.4, 0.4, 0.4], '0:19:44': [0.2, 0.2, 0.2, 0.2] },
  };
  const TH = { base: [55, 55, 55, 55], '0:9:22': [55.5, 55.1, 55.3, 55.3], '0:17:33': [55, 55.2, 55.2, 55.2] };
  const RDNA3 = { ARCH: 'gfx1100', SKEW_CLASS: 'rdna3', STUB_GAIN: JSON.stringify(GAIN), STUB_TH: JSON.stringify(TH) };

  bashIt('gates, probes, measures r1 and r2, times the rungs at three sizes, races two and verifies one that passes', () => {
    const r = run(RDNA3, { shares: { 'verify-t17.jsonl': '{"old":true}\n' } });
    expect([r.status, r.err]).toEqual([0, '']);
    // The gate: one skew build against base's VGPRs. Then both built and checked, with no timing in sweep.sh.
    expect(r.log[0]).toBe(`variant-check VARIANTS=base=;skew=-DPEARL_AMD_SKEW=1 TWINS=skew<base ARCH=gfx1100 CHECK_ROOT=${r.root}/variant-check`);
    expect(r.log[1]).toBe('sweep SWEEP=base=;skew=-DPEARL_AMD_SKEW=1 FOLD= ROUNDS=0 ARCH=gfx1100');
    expect(r.lines[0]).toBe('start gfx1100 fold wmma11; the mined launch 131072:2048:1200; up to 3 rungs at the mined '
      + 'launch, 2 in hashrate.js; verify-hits 150s');
    // The probe, with a rung's waits; then the big check with the build's.
    expect(r.log[2]).toBe('foldtest skew 128 256 1 1 0 wmma11 wait=0:8:17');
    expect(r.lines).toContain('probe: skew: PEARL_AMD_SKEW=1 on wmma11; first 144 blocks; waits 0:8:17% of a 7000-clock '
      + 'slab (0:9:19 sleeps), from PEARL_AMD_SKEW_WAIT');
    expect(r.lines).toContain('probe: skew probe: 144 blocks, 144 took a place, on 48 counters of 3 to 3 blocks, 48 with '
      + 'places 0, 1 and 2: as designed');
    expect(r.log[3]).toBe('foldtest skew 2048 4096 7 1 1 wmma11 wait=');
    expect(r.lines).toContain('skew check at 2048 x 4096, the build\'s waits: skew: PEARL_AMD_SKEW=1 on wmma11; first 144 '
      + 'blocks; waits 0:33:67% of a 7000-clock slab (0:36:73 sleeps), the build\'s skew probe: 144 blocks, 144 took a '
      + 'place, on 48 counters of 3 to 3 blocks, 48 with places 0, 1 and 2: as designed wmma11  ok: 32768 regions, all '
      + '16 words match; 1 launches, 0.1000 s, 40.000 TMAC/s PASS ');
    // r1 and r2: base at one generation (m 128 x 48), cols 256, 512 and 768, three times, 3000 launches.
    expect(r.log.slice(4, 13)).toEqual([0, 1, 2].flatMap(() => [256, 512, 768]
      .map((c) => `foldtest base 6144 ${c} 1 3000 0 wmma11 0 miner wait=`)));
    expect(r.lines).toContain('base at one generation (m 6144), TMAC/s: one block a WGP 30.000, two 43.333, three 50.000: '
      + 'r1 1.80, r2 1.30 (each held to 1-3 and 1-1.5)');
    expect(r.lines).toContain('rungs, places 0:1:2 in percent of a slab: t8 0:8:17, t17 0:17:33, t33 0:33:67, w17 0:9:22, '
      + 'w33 0:19:44, ctl 67:67:67');
    // Each size: base then every rung, in turn, four rounds; at the mined launch only the three best at eight.
    const one = folds(r, ' 6144 768 ').slice(3);   // after r1 and r2's three runs at cols 768
    expect(one).toHaveLength(4 * 7);
    expect(one.slice(0, 7)).toEqual(['foldtest base 6144 768 1 3000 0 wmma11 0 miner wait=',
      ...['0:8:17', '0:17:33', '0:33:67', '0:9:22', '0:19:44', '67:67:67']
        .map((w) => `foldtest skew 6144 768 1 3000 0 wmma11 0 miner wait=${w}`)]);
    expect(r.lines).toContain('fold one-gen, round 1 of 4, TMAC/s: base 50.000, t8 51.000 (+2.00%), t17 50.200 (+0.40%), '
      + 't33 50.000 (+0.00%), w17 50.400 (+0.80%), w33 50.300 (+0.60%), ctl 49.500 (-1.00%)');
    // Round 1 isn't scored.
    expect(r.lines).toContain('fold one-gen, t8 (0:8:17): rounds 2-4 +0.20, +0.20, +0.20%; mean +0.20%');
    expect(r.lines).toContain('fold one-gen, ctl (67:67:67): rounds 2-4 -1.00, -1.00, -1.00%; mean -1.00%');
    // Each launch there ends on a block that waited: the rungs against the control.
    expect(r.lines).toContain('fold one-gen against ctl, the mean change less ctl\'s (-1.00%), in points: t8 +1.20, '
      + 't17 +1.40, t33 +1.00, w17 +1.80, w33 +1.60');
    expect(folds(r, ' 49152 768 ')).toHaveLength(28);
    expect(folds(r, ' 49152 768 ')[0]).toBe('foldtest base 49152 768 1 400 0 wmma11 0 miner wait=');
    // ctl leads at eight generations, but it isn't a rung to ship: w17, w33 and t17 go on.
    expect(r.lines).toContain('fold mined: m 131072, cols 2048, 1200 launches a run');
    expect(folds(r, ' 131072 2048 ').slice(0, 4)).toEqual(['foldtest base 131072 2048 1 1200 0 wmma11 0 miner wait=',
      ...['0:9:22', '0:19:44', '0:17:33'].map((w) => `foldtest skew 131072 2048 1 1200 0 wmma11 0 miner wait=${w}`)]);
    expect(folds(r, ' 131072 2048 ')).toHaveLength(16);
    expect(r.lines).toContain('fold mined, w17 (0:9:22): rounds 2-4 +0.40, +0.40, +0.40%; mean +0.40%');
    // hashrate.js: base, w17 and t17, four rounds, each on its waits.
    expect(r.log.filter((l) => l.startsWith('hashrate'))).toEqual(Array(4).fill(['hashrate base 60 20 wait=',
      'hashrate skew 60 20 wait=0:9:22', 'hashrate skew 60 20 wait=0:17:33']).flat());
    expect(r.lines).toContain('hashrate round 1 of 4, TH/s: base 55, w17 55.5 (+0.91%), t17 55 (+0.00%)');
    expect(r.lines).toContain('w17 verdict (0:9:22): hashrate rounds 2-4 +0.18, +0.55, +0.55%; mean +0.43%; +0.3% or more '
      + 'in every scored round: no (1 under)');
    expect(r.lines).toContain('t17 verdict (0:17:33): hashrate rounds 2-4 +0.36, +0.36, +0.36%; mean +0.36%; +0.3% or more '
      + 'in every scored round: yes');
    // verify-hits on t17, which passes, not w17's larger mean; on its waits, into a share file emptied first.
    expect(r.log.filter((l) => l.startsWith('verify'))).toEqual([
      `verify skew 150 ${r.root}/shares/verify-t17.jsonl wait=0:17:33 existed=false`]);
    expect(r.shares('verify-t17.jsonl')).toBeNull();
    expect(r.lines.slice(-3)).toEqual([
      't17 verify-hits (0:17:33): {"file":"skew","k":2048,"hits":731,"verified":400,"failed":0,"salts":95,"PASS":true}',
      't17: verify-hits 400 of 400, 0 failed: the rule\'s 400 of 400 holds',
      'done',
    ]);
  }, 120000);

  bashIt('verifies the best mean when no rung passes, and exits 1 when verify-hits isn\'t 400 of 400', () => {
    const th = { base: [55, 55, 55, 55], '0:9:22': [55, 55.1, 55.1, 55.1], '0:17:33': [55, 55.05, 55.05, 55.05] };
    const r = run({ ...RDNA3, STUB_TH: JSON.stringify(th),
      STUB_VERIFY: '{"file":"NAME","k":2048,"hits":731,"verified":399,"failed":1,"salts":95,"PASS":false}' });
    expect(r.status).toBe(1);
    expect(r.lines).toContain('w17 verdict (0:9:22): hashrate rounds 2-4 +0.18, +0.18, +0.18%; mean +0.18%; +0.3% or more '
      + 'in every scored round: no (3 under)');
    expect(r.log.filter((l) => l.startsWith('verify'))).toEqual([
      `verify skew 150 ${r.root}/shares/verify-w17.jsonl wait=0:9:22 existed=false`]);
    expect(r.lines.slice(-2)).toEqual(['w17: verify-hits 399 verified, 1 failed: not the rule\'s 400 of 400', 'done']);
  }, 120000);

  bashIt('times SKEW_EXTRA\'s rungs at every size and races them, on top of the best, and refuses a bad one', () => {
    const th = { ...TH, '0:10:23': [55, 55.3, 55.3, 55.3] };
    const r = run({ ...RDNA3, STUB_TH: JSON.stringify(th), SKEW_EXTRA: 'won=0:10:23 t17=1:2:3' });
    expect(r.status).toBe(0);
    expect(r.lines).toContain('SKEW_EXTRA: t17 is a rung already: not raced twice');
    expect(r.lines).toContain('rungs, places 0:1:2 in percent of a slab: t8 0:8:17, t17 0:17:33, t33 0:33:67, w17 0:9:22, '
      + 'w33 0:19:44, won 0:10:23, ctl 67:67:67');
    // It gains nothing at eight generations, and still goes on: the mined launch, then hashrate.js after the best two.
    expect(r.lines).toContain('fold eight-gen, won (0:10:23): rounds 2-4 +0.00, +0.00, +0.00%; mean +0.00%');
    expect(folds(r, ' 131072 2048 ').slice(0, 5).map((l) => l.replace(/.* wait=/, ''))).toEqual(['', '0:9:22', '0:19:44',
      '0:17:33', '0:10:23']);
    expect(r.log.filter((l) => l.startsWith('hashrate')).slice(0, 4)).toEqual(['hashrate base 60 20 wait=',
      'hashrate skew 60 20 wait=0:9:22', 'hashrate skew 60 20 wait=0:17:33', 'hashrate skew 60 20 wait=0:10:23']);
    expect(r.lines).toContain('won verdict (0:10:23): hashrate rounds 2-4 +0.55, +0.55, +0.55%; mean +0.55%; +0.3% or more '
      + 'in every scored round: yes');
    expect(r.log.filter((l) => l.startsWith('verify'))).toEqual([
      `verify skew 150 ${r.root}/shares/verify-won.jsonl wait=0:10:23 existed=false`]);
    for (const x of ['x=0:0:401', 'ctl=1:2:3', 'base=1:2:3', 'X=1:2:3', 'a=1:2', 'a=1:2:3:4', 'a=1:-2:3', '=1:2:3']) {
      const bad = run({ ARCH: 'gfx1100', SKEW_EXTRA: `ok=0:1:2 ${x}` });
      expect([x, bad.status, bad.lines]).toEqual([x, 2, [`SKEW_EXTRA takes name=a:b:c, each wait at most 400 and the name not ctl or base, not ${x}`]]);
    }
  }, 120000);

  bashIt('on RDNA 4: wmma12, k 4096\'s mined launch, SKEW_FOLD and SKEW_RACE, and no work-aimed rung without rates', () => {
    const r = run({ ARCH: 'gfx1201', SKEW_CLASS: 'rdna4', STUB_ARCH: 'gfx1201', STUB_WGPS: '32', SKEW_FOLD: '1',
      SKEW_RACE: '1', SKEW_VERIFY: '120', STUB_NORATE: '1',
      STUB_GAIN: JSON.stringify({ 'eight-gen': { '0:17:33': [0.2, 0.2, 0.2, 0.2] }, mined: { '0:17:33': [0.3, 0.3, 0.3, 0.3] } }),
      STUB_TH: JSON.stringify({ base: [110, 110, 110, 110], '0:17:33': [111, 111, 111, null] }) });
    expect(r.status).toBe(1);
    expect(r.lines).toContain('probe: skew: PEARL_AMD_SKEW=1 on wmma12; first 96 blocks; waits 0:8:17% of a 2800-clock '
      + 'slab (0:4:7 sleeps), from PEARL_AMD_SKEW_WAIT');
    expect(r.lines).toContain('base at one generation (m 4096): not every rate was read (?, ?, 50.000), so no work-aimed rungs');
    expect(r.lines).toContain('rungs, places 0:1:2 in percent of a slab: t8 0:8:17, t17 0:17:33, t33 0:33:67, ctl 67:67:67');
    expect(folds(r, ' 4096 768 ')[0]).toBe('foldtest base 4096 768 1 4000 0 wmma12 0 miner wait=');
    expect(folds(r, ' 32768 768 ')[0]).toBe('foldtest base 32768 768 1 500 0 wmma12 0 miner wait=');
    expect(folds(r, ' 131072 4096 ')).toEqual(Array(4).fill(['foldtest base 131072 4096 1 300 0 wmma12 0 miner wait=',
      'foldtest skew 131072 4096 1 300 0 wmma12 0 miner wait=0:17:33']).flat());
    // Round 4's run failed: three scored rounds aren't there, so there's no verdict and no verify.
    expect(r.lines).toContain('hashrate round 4 of 4, TH/s: base 110, t17 ?');
    expect(r.lines).toContain('t17 verdict: 2 of 3 scored rounds read; no verdict');
    expect(r.lines[r.lines.length - 1]).toBe('no raced rung has three scored rounds: no verify-hits');
    expect(r.log.filter((l) => l.startsWith('verify'))).toEqual([]);
  }, 120000);

  bashIt('stops when the probe finds the counters don\'t tell WGPs apart, or the fold test ignores the waits', () => {
    const lost = run({ ARCH: 'gfx1201', STUB_ARCH: 'gfx1201', STUB_WGPS: '32', STUB_PROBE: 'lost' });
    expect(lost.status).toBe(1);
    expect(lost.lines.slice(-2)).toEqual(['probe: skew probe: 96 blocks, 96 took a place, on 8 counters of 12 to 12 blocks, '
      + '0 with places 0, 1 and 2: not as designed', 'the probe\'s places aren\'t as designed: nothing to race']);
    expect(lost.log.filter((l) => l.startsWith('foldtest'))).toHaveLength(1);
    const none = run({ ARCH: 'gfx1100', STUB_PROBE: 'none' });
    expect([none.status, none.lines[none.lines.length - 1]]).toEqual([1, 'the probe\'s places aren\'t as designed: nothing to race']);
    const deaf = run({ ARCH: 'gfx1100', STUB_IGNORE_ENV: '1' });
    expect([deaf.status, deaf.lines[deaf.lines.length - 1]])
      .toEqual([1, 'the fold test didn\'t take PEARL_AMD_SKEW_WAIT=0:8:17: nothing to race']);
    const big = run({ ARCH: 'gfx1100', STUB_BIGFAIL: 'skew' });
    expect(big.status).toBe(1);
    expect(big.lines[big.lines.length - 1]).toBe('the skew build\'s check at 2048 x 4096 didn\'t pass: nothing to race');
    expect(big.log.filter((l) => l.startsWith('foldtest'))).toHaveLength(2);
  }, 120000);

  bashIt('stops before timing anything when a build is dropped or fails its check, or the card is the wrong one', () => {
    const dropped = run({ ARCH: 'gfx1100', STUB_KEEP: 'base' });
    expect([dropped.status, dropped.lines[dropped.lines.length - 1]]).toEqual([0, 'the skew build was dropped by the gate: nothing to race']);
    expect(dropped.log.filter((l) => !l.startsWith('variant-check'))).toEqual([]);
    const noBase = run({ ARCH: 'gfx1100', STUB_KEEP: 'skew' });
    expect([noBase.status, noBase.lines[noBase.lines.length - 1]]).toEqual([1, 'base was dropped or not built: nothing to race against']);
    const badCheck = run({ ARCH: 'gfx1100' }, { sweep: sweepOut(true, false) });
    expect([badCheck.status, badCheck.lines[badCheck.lines.length - 1]])
      .toEqual([1, 'skew\'s fold check at 1024 x 1024 didn\'t pass: nothing to race']);
    expect(badCheck.log.filter((l) => l.startsWith('foldtest'))).toEqual([]);
    const wrong = run({ ARCH: 'gfx1201', SKEW_CLASS: 'rdna3' });
    expect([wrong.status, wrong.lines, wrong.log]).toEqual([2, ['this card is gfx1201, not rdna3: run the other class\'s script'], []]);
    const two = run({ ARCH: 'gfx1102' });
    expect([two.status, two.lines]).toEqual([2, ['gfx1102 has no skew: its WGP holds two blocks of the fold, not three (pearl_amd_fold.h)']]);
    const cdna = run({ ARCH: 'gfx942' });
    expect([cdna.status, cdna.lines]).toEqual([2, ['no RDNA 3 or RDNA 4 card here (gfx942): the skew is in their folds only']]);
    const bad = run({ ARCH: 'gfx1100', SKEW_CLASS: 'rdna5' });
    expect([bad.status, bad.lines]).toEqual([2, ['SKEW_CLASS is rdna3 or rdna4, not rdna5']]);
    for (const [k, v] of [['SKEW_RACE', '0'], ['SKEW_FOLD', 'x'], ['SKEW_VERIFY', '010']]) {
      const n = run({ ARCH: 'gfx1100', [k]: v });
      expect([k, n.status, n.lines]).toEqual([k, 2, ['SKEW_FOLD, SKEW_RACE and SKEW_VERIFY are whole numbers above 0']]);
    }
  }, 120000);

  it('times the mined launch at the batch the miner folds on each card, with the same MACs a run', () => {
    // amd_col_batch: gfx1100 and gfx1101 cap the batch at a width of their own, every other card at
    // PEARL_AMD_COL_BATCH. A batch's columns are 16 a column offset (PEARL_COLS_COUNT).
    const batch = fn(HOST, 'uint32_t amd_col_batch(');
    expect(batch).toContain('strncmp(arch, "gfx1100", 7) == 0 || strncmp(arch, "gfx1101", 7) == 0');
    const narrow = Number(grab(batch, /arch\[7\] == ':'\)\)\n\s*limit = (\d+)u;/));
    const wide = Number(grab(HOST, /^#define PEARL_AMD_COL_BATCH (\d+)u$/m));
    const cols = Number(grab(read('src', 'pearl_config.h'), /^#define PEARL_COLS_COUNT (\d+)$/m));
    const mined = grab(RACE, /^(case "\$A" in\n {2}gfx1100\* \| gfx1101\*\) MINED=[\s\S]*?esac)$/m);
    expect(mined).toBe('case "$A" in\n'
      + `  gfx1100* | gfx1101*) MINED=131072:${narrow * cols}:1200 ;;\n`
      + `  gfx11*) MINED=131072:${wide * cols}:600 ;;\n`
      + `  *) MINED=131072:${wide * cols}:300 ;;\n`
      + 'esac');
    // 600 launches of 4096 columns at k 2048, the fold timing sweep.sh has always used, on every card.
    expect([narrow * cols * 1200 * 2048, wide * cols * 600 * 2048, wide * cols * 300 * 4096]).toEqual(Array(3).fill(4096 * 600 * 2048));
  });
});
