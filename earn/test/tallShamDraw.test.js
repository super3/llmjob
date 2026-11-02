'use strict';

const fs = require('fs');
const path = require('path');
const { preprocess: resolve } = require('./helpers/amdSwitches');
const noise = require('../src/shared/miner/noise');

// PEARL_TALL_SHAM_DRAW (pearl_config.h, 0 by default): item 16, step 2, before any work on #6.
// On top of PEARL_TALL_AB_SKIP=2 it prices what a B' gather draw adds back: each B fragment's
// bytes read a second time and combined with one IADD3 a register into 63 + x - y a byte, from a
// B' ring the prologue fills with bytes in [0, 63]. Timing only: the host refuses any target but
// zero.
//
// Nothing here runs device code. These check:
// - off by default, 0 or 1, and refused unless PEARL_TALL_AB_SKIP=2, the streamed body and the
//   ring's roles;
// - off, the fold and the host are the AB_SKIP code, and nothing of it reaches a shipped build;
// - on, the same mma and the same first reads, one second read after each B fragment load, one
//   IADD3 a register, the table, and no B' fill;
// - the second read is conflict-free, stays in the warp's own B' columns, and never reads the
//   pair's own (bytes the fold reads too: one half is another pair's matrices 2 and 3);
// - one IADD3 makes exactly 63 + U[p] - U[q] a byte, noise.js's B' with B filled with 63;
// - the host refuses to mine with it and names the arm.

const SRC = path.join(__dirname, '..', 'native', 'src');
const NL = String.fromCharCode(10);
const BS = String.fromCharCode(92);
// Line endings normalised: a Windows checkout has CRLF.
const rawRead = (file) => fs.readFileSync(file, 'utf8').split(String.fromCharCode(13) + NL).join(NL);
const read = (name) => rawRead(path.join(SRC, name));
const HEADER = read('pearl_config.h');
const KERNEL = read('pearl_kernel.cu');
const FOLD = read('pearl_fold_tall.cuh');
const HOST = read('pearl_host.cu');
const ROOT = path.join(__dirname, '..', '..');
const SHIPPED = [
  '.github/workflows/native-core.yml',
  '.github/workflows/miner-build.yml',
  'earn/native/build-local.ps1',
  'earn/native/binding.gyp',
];
const NAME = 'PEARL_TALL_SHAM_DRAW';
const SKIP = 'PEARL_TALL_AB_SKIP';
const STUBS_OFF = { PEARL_TALL_STUB_SPIN: 0, PEARL_TALL_STUB_ALU: 0 };
const ON = { [SKIP]: 2, [NAME]: 1 };
const ARCH = '__CUDA_ARCH__';

// Source without comments: comment lines dropped, trailing ones cut.
function code(text) {
  return text.split(NL).filter((l) => !l.trim().startsWith('//')).map((l) => {
    const at = l.indexOf('//');
    return at > 0 && l.slice(0, at).trim() ? l.slice(0, at).trimEnd() : l;
  }).join(NL);
}
const lines = (...l) => l.join(NL);

// The body of the function whose head is head, braces and all.
function body(src, head) {
  const at = src.indexOf(head);
  expect([head, at >= 0]).toEqual([head, true]);
  const open = src.indexOf('{', at + head.length - 1);
  let depth = 0;
  for (let i = open; i < src.length; i++) {
    if (src[i] === '{') depth++;
    if (src[i] === '}' && --depth === 0) return src.slice(open, i + 1);
  }
  throw new Error('unbalanced ' + head);
}

// A #define and its continuation lines, each without its trailing backslash.
function macro(text, head) {
  const all = text.split(NL);
  const at = all.findIndex((l) => l.startsWith(head));
  expect([head, at >= 0]).toEqual([head, true]);
  const out = [];
  for (let i = at; i < all.length; i++) {
    const l = all[i].trimEnd();
    out.push(l.endsWith(BS) ? l.slice(0, -1).trimEnd() : l);
    if (!l.endsWith(BS)) break;
  }
  return out;
}

// The lines of a that b does not have, as runs of consecutive lines, when b is a in order with
// those runs taken out. Fails if it is not.
function extraRuns(a, b) {
  const runs = [];
  let j = 0;
  let run = null;
  for (const l of a) {
    if (j < b.length && l === b[j]) {
      j++;
      run = null;
    } else {
      if (!run) runs.push((run = []));
      run.push(l);
    }
  }
  expect(j).toBe(b.length);
  return runs;
}

// ---------------------------------------------------------------------------
// A small C preprocessor for the guards, as tallHashStub.test.js and tallAbSkip.test.js have it: object-like #define,
// #undef, #if, #ifdef, #ifndef, #elif, #else, #endif and #error, with -D flags. Returns the
// #error lines it reaches and the definitions.
function tokens(expr) {
  const out = [];
  const word = (c) => (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c === '_';
  let i = 0;
  while (i < expr.length) {
    const c = expr[i];
    if (c === ' ' || c === String.fromCharCode(9)) {
      i++;
    } else if (word(c)) {
      let j = i;
      while (j < expr.length && word(expr[j])) j++;
      out.push(expr.slice(i, j));
      i = j;
    } else if (['&&', '||', '==', '!=', '<=', '>='].includes(expr.slice(i, i + 2))) {
      out.push(expr.slice(i, i + 2));
      i += 2;
    } else if ('()!<>+-*/%?:'.includes(c)) {
      out.push(c);
      i++;
    } else {
      throw new Error(`no token at ${expr.slice(i)}`);
    }
  }
  return out;
}

function evaluate(expr, defs, seen = new Set()) {
  const t = tokens(expr);
  let p = 0;
  const take = (s) => (t[p] === s ? (p++, true) : false);
  const unary = () => {
    const tk = t[p++];
    if (tk === undefined) throw new Error(`short expression: ${expr}`);
    if (tk === '(') {
      const v = ternary();
      if (!take(')')) throw new Error(`no ) in ${expr}`);
      return v;
    }
    if (tk === '!') return unary() === 0 ? 1 : 0;
    if (tk === '-') return -unary();
    if (tk === 'defined') {
      const paren = take('(');
      const name = t[p++];
      if (paren && !take(')')) throw new Error(`no ) in ${expr}`);
      return defs.has(name) ? 1 : 0;
    }
    if (tk[0] >= '0' && tk[0] <= '9') {
      let s = tk;
      while ('uUlL'.includes(s[s.length - 1])) s = s.slice(0, -1);
      const v = Number(s);
      if (!Number.isFinite(v)) throw new Error(`not a number: ${tk}`);
      return v;
    }
    if (')!<>+*/%?:&|='.includes(tk[0])) throw new Error(`unexpected ${tk} in ${expr}`);
    if (!defs.has(tk) || seen.has(tk)) return 0;
    if (defs.get(tk) === null) throw new Error(`function-like ${tk} in ${expr}`);
    return evaluate(defs.get(tk), defs, new Set([...seen, tk]));
  };
  const binary = (next, ops, f) => () => {
    let v = next();
    for (;;) {
      const op = ops.find((x) => t[p] === x);
      if (!op) return v;
      p++;
      v = f(op, v, next());
    }
  };
  const num = (b) => (b ? 1 : 0);
  const mul = binary(unary, ['*', '/', '%'],
    (op, a, b) => (op === '*' ? a * b : op === '/' ? Math.trunc(a / b) : a % b));
  const add = binary(mul, ['+', '-'], (op, a, b) => (op === '+' ? a + b : a - b));
  const rel = binary(add, ['<', '>', '<=', '>='],
    (op, a, b) => num(op === '<' ? a < b : op === '>' ? a > b : op === '<=' ? a <= b : a >= b));
  const eq = binary(rel, ['==', '!='], (op, a, b) => num(op === '==' ? a === b : a !== b));
  const and = binary(eq, ['&&'], (op, a, b) => num(a !== 0 && b !== 0));
  const or = binary(and, ['||'], (op, a, b) => num(a !== 0 || b !== 0));
  function ternary() {
    const c = or();
    if (!take('?')) return c;
    const x = ternary();
    if (!take(':')) throw new Error(`no : in ${expr}`);
    const y = ternary();
    return c !== 0 ? x : y;
  }
  const v = ternary();
  if (p !== t.length) throw new Error(`left over in ${expr}`);
  return v;
}

function preprocess(text, flags) {
  const defs = new Map(Object.entries(flags).map(([k, v]) => [k, String(v)]));
  const errors = [];
  const stack = [];
  for (const raw of text.split(BS + NL).join(' ').split(NL)) {
    const line = raw.trim();
    if (!line.startsWith('#')) continue;
    const sp = line.indexOf(' ');
    const dir = sp < 0 ? line.slice(1) : line.slice(1, sp);
    let rest = sp < 0 ? '' : line.slice(sp + 1).trim();
    if (dir !== 'error' && rest.includes('//')) rest = rest.slice(0, rest.indexOf('//')).trim();
    const live = stack.every((f) => f.on);
    if (dir === 'if' || dir === 'ifdef' || dir === 'ifndef') {
      const on = live && (dir === 'if' ? evaluate(rest, defs) !== 0 : defs.has(rest) === (dir === 'ifdef'));
      stack.push({ on, done: on, outer: live });
    } else if (dir === 'elif') {
      const f = stack[stack.length - 1];
      f.on = f.outer && !f.done && evaluate(rest, defs) !== 0;
      f.done = f.done || f.on;
    } else if (dir === 'else') {
      const f = stack[stack.length - 1];
      f.on = f.outer && !f.done;
      f.done = true;
    } else if (dir === 'endif') {
      expect(stack.length).toBeGreaterThan(0);
      stack.pop();
    } else if (!live) {
      continue;
    } else if (dir === 'define') {
      let end = 0;
      while (end < rest.length && rest[end] !== ' ' && rest[end] !== '(') end++;
      defs.set(rest.slice(0, end), rest[end] === '(' ? null : rest.slice(end).trim());
    } else if (dir === 'undef') {
      defs.delete(rest);
    } else if (dir === 'error') {
      errors.push(rest);
    }
  }
  expect(stack).toEqual([]);
  return { errors, defs };
}

// What a build goes through: the defaults the guards read (the hash stubs', PEARL_TALL_TMA and
// the device side's PEARL_TALL_TMA_BODY, the cluster switch, the ring's roles), the skip's block,
// the streamed body's default, then the sham's own block.
function guards() {
  const stubs = lines('#ifndef PEARL_TALL_STUB_SPIN', '#define PEARL_TALL_STUB_SPIN 0', '#endif',
    '#ifndef PEARL_TALL_STUB_ALU', '#define PEARL_TALL_STUB_ALU 0', '#endif');
  const tma = lines('#ifndef PEARL_TALL_TMA', '#define PEARL_TALL_TMA 1', '#endif');
  const tmaBody = lines('#if PEARL_TALL_TMA && defined(__CUDA_ARCH__) && PEARL_TMA_BODY_ARCH(__CUDA_ARCH__)',
    '#define PEARL_TALL_TMA_BODY 1', '#else', '#define PEARL_TALL_TMA_BODY 0', '#endif');
  const cluster = lines('#ifndef PEARL_TALL_CLUSTER', '#define PEARL_TALL_CLUSTER 0', '#endif');
  const roles = lines('#ifndef PEARL_TALL_TMA_ROLES', '#define PEARL_TALL_TMA_ROLES 1', '#endif');
  const pipe = lines('#ifndef PEARL_TALL_FRAG_PIPE', '#define PEARL_TALL_FRAG_PIPE 1', '#endif');
  for (const t of [stubs, tma, tmaBody, cluster, roles, pipe]) expect(HEADER).toContain(t);
  expect(HEADER).toContain('#define PEARL_TMA_BODY_ARCH(a) ((a) >= 1200 || PEARL_SM100_ARCH(a))');
  const tmaBodyOut = tmaBody.replace('PEARL_TMA_BODY_ARCH(__CUDA_ARCH__)',
    '((__CUDA_ARCH__) >= 1200 || ((__CUDA_ARCH__) == 1000))');
  const skipFrom = HEADER.indexOf(`#ifndef ${SKIP}${NL}`);
  const skipTo = HEADER.indexOf(NL + NL + '// The cp.async tall fold\'s tile walk', skipFrom);
  const from = HEADER.indexOf(`#ifndef ${NAME}${NL}`);
  const to = HEADER.indexOf(NL + NL + '// Blackwell\'s ring bookkeeping', from);
  expect([skipFrom > 0, skipTo > skipFrom, from > 0, to > from]).toEqual([true, true, true, true]);
  // The sham's block comes after everything it reads.
  for (const t of [stubs, tmaBody, cluster, roles, pipe]) expect(HEADER.indexOf(t)).toBeLessThan(from);
  expect(skipTo).toBeLessThan(from);
  return [stubs, tma, tmaBodyOut, cluster, roles, HEADER.slice(skipFrom, skipTo), pipe,
    HEADER.slice(from, to)].join(NL);
}

// The fold's lane geometry (pinned below against the source).
const SK = 64;
const BN = 256;
const BM = 192;
const STAGE = (BM + BN) * SK;
const STRIDE = STAGE + 512;
const f4 = (lane) => (lane >> 1) & 3;
const blcol = (lane) => (lane & 7) + ((lane >> 4) & 1) * 8;
const bLane0 = (wc, lane) => (wc * 64 + blcol(lane)) * SK + 16 * (((lane >> 3) & 1) ^ f4(lane));
// The second read's two halves (PEARL_TALL_SHAM_B): ldmatrix.x2 takes lanes 0-15's addresses.
const HALVES = [512, 128];

// Bytes of a word, and a word of bytes, low byte first.
const bytesOf = (w) => [0, 8, 16, 24].map((s) => (w >>> s) & 0xFF);
const wordOf = (b) => ((b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)) >>> 0);
// One IADD3 as the fold writes it: 0x3F3F3F3Fu + b - d, mod 2^32.
const iadd3 = (b, d) => ((0x3F3F3F3F + b - d) >>> 0);

// ---------------------------------------------------------------------------
describe('TMA tall fold sham draw (PEARL_TALL_SHAM_DRAW)', () => {
  test('is off by default, 0 or 1, and only on top of the B\' skip, the streamed body and the roles', () => {
    expect(HEADER).toContain(`#ifndef ${NAME}${NL}#define ${NAME} 0${NL}#endif`);
    const GUARDS = guards();
    const errorsFor = (flags) => preprocess(GUARDS, flags).errors;
    expect(evaluate(NAME, preprocess(GUARDS, {}).defs)).toBe(0);
    // It builds on the host pass and both TMA device passes, with the skip of B'.
    for (const arch of [null, 1000, 1200]) {
      const flags = arch ? { ...ON, [ARCH]: arch } : { ...ON };
      expect([flags, errorsFor(flags)]).toEqual([flags, []]);
    }
    // Off, every skip arm and every other card builds as before.
    for (const mask of [0, 1, 2, 3]) expect(errorsFor({ [SKIP]: mask, [ARCH]: 1200 })).toEqual([]);
    for (const arch of [610, 750, 800, 860, 890, 900]) expect(errorsFor({ [ARCH]: arch })).toEqual([]);
    // Refused, naming the switch, and each fine without it.
    const named = [
      [{ [NAME]: 2 }, {}],
      [{ [NAME]: -1 }, {}],
      ...[0, 1, 3].map((mask) => [{ [SKIP]: mask, [NAME]: 1 }, { [SKIP]: mask }]),
      [{ ...ON, PEARL_TALL_FRAG_PIPE: 0 }, { [SKIP]: 2, PEARL_TALL_FRAG_PIPE: 0 }],
      [{ ...ON, PEARL_TALL_TMA_ROLES: 0 }, { [SKIP]: 2, PEARL_TALL_TMA_ROLES: 0 }],
    ];
    for (const [f, without] of named) {
      const errors = errorsFor(f);
      expect([f, errors.length > 0 && errors.some((e) => e.includes(NAME))]).toEqual([f, true]);
      expect([without, errorsFor(without)]).toEqual([without, []]);
    }
    // And where the skip itself is refused (the cluster build, the stubs, another card), so is the sham.
    for (const f of [{ ...ON, PEARL_TALL_CLUSTER: 1, [ARCH]: 1200 }, { ...ON, PEARL_TALL_STUB_ALU: 1 },
      { ...ON, [ARCH]: 890 }, { ...ON, PEARL_TALL_TMA: 0, [ARCH]: 1200 }]) {
      expect([f, errorsFor(f).length > 0]).toEqual([f, true]);
    }
  });

  test('off, the fold and the host are the AB_SKIP code, and no shipped build sees any of it', () => {
    // Every block of it is its own #if, seven in the fold; the host's sit inside the skip's.
    expect(FOLD.match(/^#if PEARL_TALL_SHAM_DRAW$/gm)).toHaveLength(7);
    expect(FOLD.match(/PEARL_TALL_SHAM_DRAW/g).length).toBeGreaterThanOrEqual(7);
    for (const mask of [0, 1, 2, 3]) {
      const OFF = resolve(FOLD, { [SKIP]: mask, [NAME]: 0 });
      for (const t of ['SHAM', 'm8n8.x2', '0x3F3F3F3Fu + b', 'constexpr bool ab_load']) {
        expect([mask, t, OFF.includes(t)]).toEqual([mask, t, false]);
      }
    }
    // The skip's own flag and hook stay as they were raced.
    const SKIP2 = resolve(FOLD, { [SKIP]: 2, [NAME]: 0 });
    expect(SKIP2).toContain('  bool ab_load = true;');
    expect(SKIP2).toContain('#define PEARL_TALL_REFILL_LOAD ab_load = v < tile_stride;');
    // The host's blocks are inside the skip's, so the shipped host has none of them even unresolved.
    const HOST_OFF = resolve(HOST, { ...STUBS_OFF, [SKIP]: 0 });
    expect(HOST_OFF).not.toMatch(/SHAM|sham-draw/);
    expect(resolve(HOST, { ...STUBS_OFF, [SKIP]: 2, [NAME]: 0 })).not.toMatch(/SHAM|sham-draw/);
    expect(KERNEL).not.toContain(NAME);
    for (const file of SHIPPED) {
      const text = rawRead(path.join(ROOT, file));
      expect(text.length).toBeGreaterThan(0);
      expect([file, text.includes(NAME)]).toEqual([file, false]);
    }
  });

  test('on, the same mma and first reads, a second read after each B load, and no B\' fill', () => {
    const OFF = code(resolve(FOLD, { [SKIP]: 2, [NAME]: 0 }));
    const SHAM = code(resolve(FOLD, ON));
    // The skip's flag becomes a constant false, so no fill issues B' (the first tile's either), and
    // the refill's hook is empty: those two lines are swapped, everything else is added.
    expect(OFF).toContain('  bool ab_load = true;');
    expect(SHAM).toContain('  constexpr bool ab_load = false;');
    expect(SHAM.split(NL)).toContain('#define PEARL_TALL_REFILL_LOAD');
    expect(SHAM).not.toContain('ab_load = v < tile_stride');
    const swapped = OFF.split(NL).map((l) => (l === '  bool ab_load = true;' ? '  constexpr bool ab_load = false;'
      : l === '#define PEARL_TALL_REFILL_LOAD ab_load = v < tile_stride;' ? '#define PEARL_TALL_REFILL_LOAD' : l));
    const runs = extraRuns(SHAM.split(NL), swapped);
    // The table, the macro, its three uses (the first fragment, then the prefetch's two cases), its #undef.
    const first = (r) => (r[0].trim().endsWith(BS) ? r[0].trim().slice(0, -1).trimEnd() : r[0].trim());
    expect(runs.map(first)).toEqual([
      'for (uint32_t o = threadIdx.x * 16u; o < NST * BN * SK; o += PEARL_TALL_THREADS * 16u) {',
      '#define PEARL_TALL_SHAM_B(b, base, q)',
      'PEARL_TALL_SHAM_B(bp[0], bS, 0u)',
      'if (p + 1u < NB / 2u)',
      '#undef PEARL_TALL_SHAM_B']);
    // With the skip's fill: B' is never issued, A' always is.
    expect(SHAM).toContain(lines(
      '    if (!ab_load) {',
      '      constexpr bool skipA = (PEARL_TALL_AB_SKIP & 1) != 0, skipB = (PEARL_TALL_AB_SKIP & 2) != 0;'));
    expect(SHAM).toContain('        if (!skipB) pearl_tma_3d(dst, &tmB, 0u, boxB, kofs / SK, full);');
    // The same mma, the same first reads, the same readout.
    for (const op of ['pearl_mma_m16n8k32(', 'pearl_ldmatrix_x4(', 'readout(', 'fold_pair(', 'tma_fill(']) {
      expect([op, SHAM.split(op).length]).toEqual([op, OFF.split(op).length]);
    }
    // A second read right after each B fragment load, from the same k-step's base, for the same pair.
    expect(SHAM).toContain(lines(
      '          pearl_ldmatrix_x4(bp[0][0], bp[0][1], bp[0][2], bp[0][3], bS);',
      '          PEARL_TALL_SHAM_B(bp[0], bS, 0u)'));
    expect(SHAM).toContain(lines(
      '              if (p + 1u < NB / 2u)',
      '                pearl_ldmatrix_x4(bp[nxt][0], bp[nxt][1], bp[nxt][2], bp[nxt][3],',
      '                                  (bS ^ kt) + (p + 1u) * 16u * SK);',
      '              else if (step_next)',
      '                pearl_ldmatrix_x4(bp[nxt][0], bp[nxt][1], bp[nxt][2], bp[nxt][3],',
      '                                  bS ^ (kt + 32u));',
      '              if (p + 1u < NB / 2u)',
      '                PEARL_TALL_SHAM_B(bp[nxt], bS ^ kt, p + 1u)',
      '              else if (step_next)',
      '                PEARL_TALL_SHAM_B(bp[nxt], bS ^ (kt + 32u), 0u)'));
    expect(SHAM.match(/pearl_ldmatrix_x4\(bp\[/g)).toHaveLength(3);
    expect(SHAM.match(/PEARL_TALL_SHAM_B\(bp\[/g)).toHaveLength(3);
    // The macro: two x2 halves of pair q ^ 2, one IADD3 a register, nothing else.
    expect(macro(SHAM, '#define PEARL_TALL_SHAM_B(b, base, q)')).toEqual([
      '#define PEARL_TALL_SHAM_B(b, base, q)',
      '  {',
      '    uint32_t d0, d1;',
      '    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"',
      '                 : "=r"(d0), "=r"(d1) : "r"((base) + ((q) ^ 2u) * 16u * SK + 512u));',
      '    b[0] = 0x3F3F3F3Fu + b[0] - d0;',
      '    b[1] = 0x3F3F3F3Fu + b[1] - d1;',
      '    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"',
      '                 : "=r"(d0), "=r"(d1) : "r"((base) + ((q) ^ 2u) * 16u * SK + 128u));',
      '    b[2] = 0x3F3F3F3Fu + b[2] - d0;',
      '    b[3] = 0x3F3F3F3Fu + b[3] - d1;',
      '  }']);
  });

  test('the second read is conflict-free, in the warp\'s own B\' columns, and never the pair\'s own', () => {
    // The fold's lane bases and stage reads, as the model above has them.
    const F = code(FOLD);
    expect(F).toContain('  const uint32_t f = (lane >> 1) & 3u;');
    expect(F).toContain('  const uint32_t blcol = (lane & 7u) + ((lane >> 4) & 1u) * 8u;');
    expect(F).toContain('  uint32_t bLane0 = (wc * 64u + blcol) * SK + 16u * (((lane >> 3) & 1u) ^ f);');
    expect(F).toContain('          const uint32_t aS = aLane0 + so, bS = bLane0 + so;');
    expect(HEADER).toContain('#define PEARL_TALL_STAGE_STRIDE_TMA (PEARL_TALL_STAGE_BYTES + 512u)');
    const bank = (a) => (a >> 4) & 7;   // 16-byte bank group in a 128-byte row of banks
    for (const sbase of [0, 1024, 3072]) {
      for (let buf = 0; buf < 3; buf++) {
        const so = sbase + buf * STRIDE;
        for (let wc = 0; wc < 4; wc++) {
          // Every lane address of every fold x4 of B in the stage, both k-steps: pair q at k-step t.
          const base = (t, lane) => ((so + bLane0(wc, lane)) ^ (32 * t)) >>> 0;
          const fold = (t, q) => Array.from({ length: 32 }, (_, lane) => base(t, lane) + q * 1024);
          const stage = new Set();
          for (let t = 0; t < 2; t++) {
            for (let q = 0; q < 4; q++) {
              const r = fold(t, q);
              for (const a of r) stage.add(a);
              // The fold's own x4 is conflict-free, phase by phase.
              for (let ph = 0; ph < 4; ph++) expect(new Set(r.slice(8 * ph, 8 * ph + 8).map(bank)).size).toBe(8);
            }
          }
          // The fold reads every 16 bytes of the warp's 64 columns once in a stage.
          expect(stage.size).toBe(64 * SK / 16);
          for (let t = 0; t < 2; t++) {
            for (let q = 0; q < 4; q++) {
              const half = {};
              for (const off of HALVES) {
                const s = [];
                for (let lane = 0; lane < 16; lane++) s.push(base(t, lane) + (q ^ 2) * 1024 + off);
                half[off] = s;
                // Two phases of eight 16-byte rows, each in eight bank groups.
                for (let ph = 0; ph < 2; ph++) {
                  const at = [sbase, buf, wc, t, q, off, ph];
                  expect([...at, new Set(s.slice(8 * ph, 8 * ph + 8).map(bank)).size]).toEqual([...at, 8]);
                }
                for (const a of s) {
                  // Inside this warp's 64 columns of this buffer's B': table bytes, never A'.
                  expect(a % 16).toBe(0);
                  expect(a >= so + wc * 64 * SK && a + 16 <= so + (wc + 1) * 64 * SK).toBe(true);
                  // Bytes the fold reads in this stage too (shared memory has no cache, so the read
                  // pays its own wavefronts all the same), but never pair q's own columns, so the
                  // IADD3 is never b - b.
                  expect(stage.has(a)).toBe(true);
                  expect(Math.floor((Math.floor((a - so) / SK) - wc * 64) / 16)).toBe(q ^ 2);
                }
              }
              // The +512 half is exactly the second half of pair q ^ 2's own x4 at this k-step (its
              // matrices 2 and 3); the +128 half is no fold x4's half.
              expect(half[512]).toEqual(fold(t, q ^ 2).slice(16));
              for (let t2 = 0; t2 < 2; t2++) {
                for (let q2 = 0; q2 < 4; q2++) {
                  const r = fold(t2, q2).join(',');
                  expect(r.includes(half[128].join(','))).toBe(false);
                }
              }
            }
          }
        }
      }
    }
  });

  test('one IADD3 a register makes 63 + x - y a byte, the B\' a draw with B filled with 63 gives', () => {
    // Exhaustive per byte, in every byte position, with the other bytes at their extremes.
    for (const pos of [0, 1, 2, 3]) {
      for (const other of [0, 63]) {
        for (let x = 0; x < 64; x++) {
          for (let y = 0; y < 64; y++) {
            const xb = [other, other, other, other];
            const yb = [63 - other, 63 - other, 63 - other, 63 - other];
            xb[pos] = x;
            yb[pos] = y;
            const out = bytesOf(iadd3(wordOf(xb), wordOf(yb)));
            for (let i = 0; i < 4; i++) {
              if (out[i] !== 63 + xb[i] - yb[i]) throw new Error(`byte ${i}: ${xb} - ${yb} gave ${out}`);
            }
          }
        }
      }
    }
    // Out of [0, 63] a borrow crosses bytes: why the table is held to that range.
    expect(bytesOf(iadd3(wordOf([0, 0, 0, 0]), wordOf([64, 0, 0, 0])))).not.toEqual([63 - 64 + 256, 63, 63, 63]);
    expect(bytesOf(iadd3(wordOf([0, 0, 0, 0]), wordOf([64, 0, 0, 0])))[1]).toBe(62);
    // Against noise.js: B's column noise is dense[p0] - dense[p1], dense in [-32, 32). The table
    // T = dense + 32 is in [0, 63], and the IADD3 of four packed T[p0] less four T[p1] is
    // 63 + noise for each of the four k, which never needs the saturation.
    const key = Buffer.alloc(32, 7);
    const dense = noise.generateUniformRandomMatrix(noise.SEED_LABEL_B, key, [0, 1, 517], 128);
    const perm = noise.generatePermutationMatrix(noise.SEED_LABEL_B, key, 256, 128);
    for (const row of dense) {
      const T = Array.from(row, (v) => v + 32);
      expect(Math.min(...T)).toBeGreaterThanOrEqual(0);
      expect(Math.max(...T)).toBeLessThanOrEqual(63);
      const n = noise.matvecSparsePerm(perm, row, 256);
      for (let kk = 0; kk < 256; kk += 4) {
        const X = wordOf([0, 1, 2, 3].map((i) => T[perm[2 * (kk + i)]]));
        const Y = wordOf([0, 1, 2, 3].map((i) => T[perm[2 * (kk + i) + 1]]));
        const want = [0, 1, 2, 3].map((i) => noise.satInt8(63 + n[kk + i]));
        expect(bytesOf(iadd3(X, Y))).toEqual(want);
        for (const w of want) expect(w >= 0 && w <= 126).toBe(true);
      }
    }
  });

  test('the prologue\'s table: bytes uniform in [0, 63], written over the three B\' regions only', () => {
    const SHAM = code(resolve(FOLD, ON));
    expect(SHAM).toContain(lines(
      '  for (uint32_t o = threadIdx.x * 16u; o < NST * BN * SK; o += PEARL_TALL_THREADS * 16u) {',
      '    uint32_t w[4];',
      '#pragma unroll',
      '    for (uint32_t j = 0; j < 4u; j++) {',
      '      uint32_t h = (blockIdx.x * 0x9E3779B9u) ^ (o + 4u * j);',
      '      h = (h ^ (h >> 16)) * 0x85EBCA6Bu;',
      '      h = (h ^ (h >> 13)) * 0xC2B2AE35u;',
      '      w[j] = (h ^ (h >> 16)) & 0x3F3F3F3Fu;',
      '    }',
      '    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};"',
      '                 ::"r"(sbase + (o / (BN * SK)) * STRIDE + o % (BN * SK)), "r"(w[0]), "r"(w[1]),',
      '                 "r"(w[2]), "r"(w[3]) : "memory");',
      '  }',
      '  __syncthreads();'));
    // Before the barrier that publishes it, which every read of the ring comes after.
    expect(SHAM.indexOf('  __syncthreads();')).toBeGreaterThan(SHAM.indexOf('w[j] = (h ^ (h >> 16)) & 0x3F3F3F3Fu;'));
    const THREADS = 256;
    for (const block of [0, 1, 169]) {
      const counts = new Array(64).fill(0);
      const hit = new Set();
      for (let tid = 0; tid < THREADS; tid++) {
        for (let o = tid * 16; o < 3 * BN * SK; o += THREADS * 16) {
          const at = Math.floor(o / (BN * SK)) * STRIDE + (o % (BN * SK));
          // In buffer o / 16384's B' region, never its A' or its barriers.
          const buf = Math.floor(at / STRIDE);
          expect(at - buf * STRIDE + 16 <= BN * SK).toBe(true);
          hit.add(at);
          for (let j = 0; j < 4; j++) {
            let h = (Math.imul(block, 0x9E3779B9) ^ (o + 4 * j)) >>> 0;
            h = Math.imul(h ^ (h >>> 16), 0x85EBCA6B) >>> 0;
            h = Math.imul(h ^ (h >>> 13), 0xC2B2AE35) >>> 0;
            for (const b of bytesOf(((h ^ (h >>> 16)) & 0x3F3F3F3F) >>> 0)) counts[b]++;
          }
        }
      }
      // All of it: 3 x 16 KB in 16-byte rows.
      expect(hit.size).toBe(3 * BN * SK / 16);
      // Uniform: every value present, and chi-square on 63 degrees of freedom under 120 (a uniform
      // source passes that 99.998% of the time; these blocks read 57 to 94).
      const mean = 3 * BN * SK / 64;
      const chi = counts.reduce((s, c) => s + (c - mean) ** 2 / mean, 0);
      expect([block, counts.every((c) => c > 0), chi < 120]).toEqual([block, true, true]);
    }
  });

  test('the host refuses to mine with it, and names the arm', () => {
    const submit = code(body(HOST, 'extern "C" bool pearl_host_submit('));
    const on = resolve(submit, { ...STUBS_OFF, ...ON });
    expect(on).toContain(lines(
      '    if (!zero) {',
      '      if (err && err_len)',
      '        snprintf(err, err_len, "tall-fold sham-draw build is timing-only: it runs only at "',
      '                 "target 0 (hashrate.js, race.cu), never on a job; build without "',
      '                 "PEARL_TALL_SHAM_DRAW and PEARL_TALL_AB_SKIP to mine");',
      '      return false;'));
    expect(on).not.toContain('operand-skip build');
    // Nothing is queued or launched before the refusal.
    expect(on.slice(0, on.indexOf('bool zero = true;'))).not.toMatch(/<<<|cudaLaunch|pendCount\+\+/);
    const name = code(resolve(body(HOST, 'extern "C" const char *pearl_host_fold_name('), ON));
    expect(name).toContain(lines(
      '  if (ctx->foldTall && ctx->foldTma) {',
      '    snprintf(const_cast<Ctx *>(ctx)->foldName, sizeof ctx->foldName,',
      '             "tall 192x256, 8 warps of 96x64, TMA ring, k-blocked operands%s, AB_SKIP=2, "',
      '             "SHAM_DRAW=1: B\' never loaded, each B fragment read twice and IADD3\'d (timing only)",',
      '             PEARL_SM100_ARCH(ctx->tallArch * 10) ? ", sm_100 build of sm_120\'s fold" : "");',
      '    return ctx->foldName;'));
    expect(HOST).toContain('  char foldName[224] = {0};');
    for (const sm100 of [false, true]) {
      const text = 'tall 192x256, 8 warps of 96x64, TMA ring, k-blocked operands'
        + (sm100 ? ', sm_100 build of sm_120\'s fold' : '')
        + ', AB_SKIP=2, SHAM_DRAW=1: B\' never loaded, each B fragment read twice and IADD3\'d (timing only)';
      expect(text.length).toBeLessThan(224);
    }
  });
});
