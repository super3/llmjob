'use strict';

const fs = require('fs');
const path = require('path');
const { preprocess: resolve } = require('./helpers/amdSwitches');

// PEARL_TALL_AB_SKIP (pearl_config.h, 0 by default): research round 3's #11, step 1, on the TMA
// tall fold (sm_120, and sm_100 on the same code). After each block's first tile the producer's
// fills skip the operand boxes the mask names (1 A', 2 B', 3 both), so the warps fold stale
// operands from shared memory, and a race reads what those bytes cost in clock at the power cap.
// Timing only: the host refuses any target but zero.
//
// Nothing here runs device code. These check:
// - off by default, a mask of 1 to 3, and refused anywhere but the one-CTA TMA ring;
// - off, the fold and the host are the shipped code token for token;
// - on, only the producer's fill changes: the same ldmatrix and mma, from the same shared
//   addresses, into the same readout; FULL expects exactly the bytes still issued; and every
//   buffer holds real operands before the first skip;
// - the host refuses to mine with it, before anything is queued, and names the arm;
// - no build that ships names it.

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
const PROBES = path.join(SRC, '..', 'probes');
const RACE = rawRead(path.join(PROBES, 'race.cu'));
const COMPILERS = rawRead(path.join(PROBES, 'turing', 'compilers.sh'));
const SHIPPED = [
  '.github/workflows/native-core.yml',
  '.github/workflows/miner-build.yml',
  'earn/native/build-local.ps1',
  'earn/native/binding.gyp',
];
const NAME = 'PEARL_TALL_AB_SKIP';
const STUBS_OFF = { PEARL_TALL_STUB_SPIN: 0, PEARL_TALL_STUB_ALU: 0 };
const ARCH = '__CUDA_ARCH__';

// Source without comments: comment lines dropped, trailing ones cut. The pins below hold the
// code, so a reworded comment does not break them.
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

// The parts come in this order in src, each of them once.
function inOrder(src, ...parts) {
  let at = -1;
  for (const p of parts) {
    const next = src.indexOf(p);
    expect([p, next > at, src.indexOf(p, next + 1)]).toEqual([p, true, -1]);
    at = next;
  }
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
// A small C preprocessor for the guards, as tallHashStub.test.js has it: object-like #define,
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
// the device side's PEARL_TALL_TMA_BODY, the cluster switch), then the switch's own block.
function guards() {
  const stubs = lines('#ifndef PEARL_TALL_STUB_SPIN', '#define PEARL_TALL_STUB_SPIN 0', '#endif',
    '#ifndef PEARL_TALL_STUB_ALU', '#define PEARL_TALL_STUB_ALU 0', '#endif');
  const tma = lines('#ifndef PEARL_TALL_TMA', '#define PEARL_TALL_TMA 1', '#endif');
  const tmaBody = lines('#if PEARL_TALL_TMA && defined(__CUDA_ARCH__) && PEARL_TMA_BODY_ARCH(__CUDA_ARCH__)',
    '#define PEARL_TALL_TMA_BODY 1', '#else', '#define PEARL_TALL_TMA_BODY 0', '#endif');
  const cluster = lines('#ifndef PEARL_TALL_CLUSTER', '#define PEARL_TALL_CLUSTER 0', '#endif');
  for (const t of [stubs, tma, tmaBody, cluster]) expect(HEADER).toContain(t);
  // PEARL_TMA_BODY_ARCH written out: the small preprocessor has no function-like macros.
  expect(HEADER).toContain('#define PEARL_SM100_ARCH(a) ((a) == 1000)');
  expect(HEADER).toContain('#define PEARL_TMA_BODY_ARCH(a) ((a) >= 1200 || PEARL_SM100_ARCH(a))');
  const tmaBodyOut = tmaBody.replace('PEARL_TMA_BODY_ARCH(__CUDA_ARCH__)',
    '((__CUDA_ARCH__) >= 1200 || ((__CUDA_ARCH__) == 1000))');
  const from = HEADER.indexOf(`#ifndef ${NAME}${NL}`);
  const to = HEADER.indexOf(NL + NL + '// The cp.async tall fold\'s tile walk', from);
  expect(from).toBeGreaterThan(0);
  expect(to).toBeGreaterThan(from);
  // The switch's block comes after everything it reads.
  for (const t of [stubs, tmaBody, cluster]) expect(HEADER.indexOf(t)).toBeLessThan(from);
  return [stubs, tma, tmaBodyOut, cluster, HEADER.slice(from, to)].join(NL);
}

// The three runs of code the switch adds to the fold (comments dropped): the fill's flag, the
// fill's skip, and the refill's hook.
const FLAG = ['  bool ab_load = true;'];
const SKIP = [
  '    if (!ab_load) {',
  '      constexpr bool skipA = (PEARL_TALL_AB_SKIP & 1) != 0, skipB = (PEARL_TALL_AB_SKIP & 2) != 0;',
  '      if (skipA && skipB) {',
  '        pearl_mbar_arrive(full);',
  '      } else {',
  '        pearl_mbar_expect_tx(full, (skipB ? 0u : BN * SK) + (skipA ? 0u : BM * SK));',
  '        if (!skipB) pearl_tma_3d(dst, &tmB, 0u, boxB, kofs / SK, full);',
  '        if (!skipA) pearl_tma_3d(dst + BN * SK, &tmA, 0u, boxA, kofs / SK, full);',
  '      }',
  '      return;',
  '    }',
];
const HOOK = ['#define PEARL_TALL_REFILL_LOAD ab_load = v < tile_stride;'];
const EMPTY_HOOK = '#define PEARL_TALL_REFILL_LOAD';

// ---------------------------------------------------------------------------
describe('TMA tall fold operand skip (PEARL_TALL_AB_SKIP)', () => {
  test('is off by default, a mask of 1 to 3, and only on the one-CTA TMA ring', () => {
    expect(HEADER).toContain(`#ifndef ${NAME}${NL}#define ${NAME} 0${NL}#endif`);
    const GUARDS = guards();
    const errorsFor = (flags) => preprocess(GUARDS, flags).errors;
    expect(evaluate(NAME, preprocess(GUARDS, {}).defs)).toBe(0);
    // Every arm builds on the host pass and on both TMA device passes (sm_120, sm_100).
    for (const mask of [0, 1, 2, 3]) {
      for (const arch of [null, 1000, 1200]) {
        const flags = arch ? { [NAME]: mask, [ARCH]: arch } : { [NAME]: mask };
        expect([flags, errorsFor(flags)]).toEqual([flags, []]);
      }
    }
    // Off, every other card's device pass builds as before.
    const OTHERS = [610, 750, 800, 860, 890, 900];
    for (const arch of OTHERS) expect(errorsFor({ [ARCH]: arch })).toEqual([]);
    // Each refused with an error that names the switch, and each fine without it.
    const refused = [
      [{ [NAME]: -1 }, {}],
      [{ [NAME]: 4 }, {}],
      [{ [NAME]: 3, PEARL_TALL_CLUSTER: 1, [ARCH]: 1200 }, { PEARL_TALL_CLUSTER: 1, [ARCH]: 1200 }],
      [{ [NAME]: 3, PEARL_TALL_STUB_SPIN: 3000 }, { PEARL_TALL_STUB_SPIN: 3000 }],
      [{ [NAME]: 1, PEARL_TALL_STUB_ALU: 1 }, { PEARL_TALL_STUB_ALU: 1 }],
      [{ [NAME]: 3, PEARL_TALL_TMA: 0, [ARCH]: 1200 }, { PEARL_TALL_TMA: 0, [ARCH]: 1200 }],
      ...OTHERS.flatMap((arch) => [1, 2, 3].map((mask) => [{ [NAME]: mask, [ARCH]: arch }, { [ARCH]: arch }])),
    ];
    for (const [f, without] of refused) {
      const errors = errorsFor(f);
      expect([f, errors.length > 0 && errors.every((e) => e.includes(NAME))]).toEqual([f, true]);
      expect([without, errorsFor(without)]).toEqual([without, []]);
    }
  });

  test('off, the fold and the host are the shipped code token for token', () => {
    // Everything the switch adds to the fold sits in one of three #if blocks of it alone. Off,
    // what is left of it is the refill's hook, defined empty, used once and undefined.
    expect(FOLD.match(/^#if PEARL_TALL_AB_SKIP$/gm)).toHaveLength(3);
    const OFF = resolve(FOLD, { [NAME]: 0 });
    const traces = OFF.split(NL).filter((l) => /AB_SKIP|ab_load|REFILL_LOAD/.test(l))
      .map((l) => (l.trimEnd().endsWith(BS) ? l.trimEnd().slice(0, -1) : l).trim());
    expect(traces).toEqual([EMPTY_HOOK, 'PEARL_TALL_REFILL_LOAD', '#undef PEARL_TALL_REFILL_LOAD']);
    // With the empty hook expanded, the refill is the shipped one.
    expect(macro(OFF, '#define PEARL_TALL_REFILL(prod)').filter((l) => l.trim() !== 'PEARL_TALL_REFILL_LOAD'))
      .toEqual([
        '#define PEARL_TALL_REFILL(prod)',
        '  if (threadIdx.x == (prod) && stage_next) {',
        '    PEARL_TALL_REFILL_WAIT',
        '    tma_fill(bi, cn * rank + s * SK);',
        '  }']);
    // The host: the refusal and the fold's name are all of it, and neither is left off.
    const HOST_OFF = resolve(HOST, { ...STUBS_OFF, [NAME]: 0 });
    expect(HOST_OFF).not.toMatch(/AB_SKIP|operand-skip/);
    expect(HOST.match(/^#if .*PEARL_TALL_AB_SKIP/gm)).toEqual([
      '#if PEARL_TALL_STUB_SPIN || PEARL_TALL_STUB_ALU || PEARL_TALL_AB_SKIP',
      '#if PEARL_TALL_AB_SKIP',
      '#if PEARL_TALL_AB_SKIP']);
    // The kernel file has none of it: the skip is the fold's and the config's.
    expect(KERNEL).not.toContain(NAME);
    expect(KERNEL).not.toContain('ab_load');
  });

  test('on, only the producer\'s fill changes: the same reads and mma, in the same places', () => {
    const OFF = code(resolve(FOLD, { [NAME]: 0 })).split(NL);
    const offLessHook = OFF.filter((l) => l !== EMPTY_HOOK);
    expect(OFF.length - offLessHook.length).toBe(1);
    for (const mask of [1, 2, 3]) {
      const ON = code(resolve(FOLD, { [NAME]: mask }));
      // The fold off is the fold on less exactly these three runs.
      expect([mask, extraRuns(ON.split(NL), offLessHook)]).toEqual([mask, [FLAG, SKIP, HOOK]]);
      // In place: the flag before the fill, the skip at the top of the fill, before its whole
      // fill, and the hook where the empty one was, ahead of the refill that reads it.
      expect(ON).toContain(lines(...FLAG, '#if PEARL_TALL_TMA_ROLES', '  auto tma_fill = [&](uint32_t bi, uint32_t kofs) {'));
      expect(ON).toContain(lines(
        '    const uint32_t full = barFull + 8u * bi, dst = sbase + bi * STAGE;',
        '#endif',
        ...SKIP,
        '    pearl_mbar_expect_tx(full, STAGE);',
        '#if PEARL_TALL_CLUSTER_BODY'));
      expect(ON).toContain(lines(...HOOK, '#define PEARL_TALL_REFILL(prod)'));
      // The flag is read by the fill alone, and the fill runs on the producer alone.
      expect(ON.match(/\bab_load\b/g)).toHaveLength(3);
      expect(ON).toContain(lines(
        '  if (threadIdx.x == PRODUCER) {',
        '#if PEARL_TALL_TMA_ROLES',
        '#pragma unroll',
        '    for (uint32_t s = 0; s < SPC; s++) tma_fill(sbase + s * STRIDE, s * SK);'));
      expect(macro(ON, '#define PEARL_TALL_REFILL(prod)')).toEqual([
        '#define PEARL_TALL_REFILL(prod)',
        '  if (threadIdx.x == (prod) && stage_next) {',
        '    PEARL_TALL_REFILL_WAIT',
        '    PEARL_TALL_REFILL_LOAD',
        '    tma_fill(bi, cn * rank + s * SK);',
        '  }']);
      expect(ON.match(/tma_fill\(/g)).toHaveLength(3);
      // The same ldmatrix and mma, on and off.
      for (const op of ['pearl_ldmatrix_x4(', 'pearl_mma_m16n8k32(', 'readout(', 'fold_pair(']) {
        expect([mask, op, ON.split(op).length]).toEqual([mask, op, OFF.join(NL).split(op).length]);
      }
    }
  });

  test('the mma still read the shared operands and feed the transcripts', () => {
    // So ptxas keeps every one of them: each fragment is an ldmatrix of the stage's buffer,
    // each mma's accumulators go to the readout's shared XOR, and the hasher reads that back.
    const F = code(FOLD);
    expect(F).toContain(lines(
      '          const uint32_t aS = aLane0 + so, bS = bLane0 + so;',
      '          uint32_t ag[MT][4];',
      '          uint32_t bp[2][4];',
      '          pearl_ldmatrix_x4(bp[0][0], bp[0][1], bp[0][2], bp[0][3], bS);'));
    expect(F).toContain('            pearl_ldmatrix_x4(ag[mb][0], ag[mb][1], ag[mb][2], ag[mb][3], aS + mb * 16u * SK);');
    expect(F).toContain(lines(
      '                  pearl_mma_m16n8k32(acc[mb][nb][0], acc[mb][nb][1], acc[mb][nb][2],',
      '                                     acc[mb][nb][3], ag[mb][0], ag[mb][1], ag[mb][2], ag[mb][3],',
      '                                     bp[cur][2u * j], bp[cur][2u * j + 1u]);'));
    expect(F).toContain('    const int32_t *p0 = acc[2u * rl][2u * np], *p1 = acc[2u * rl + 1u][2u * np];');
    expect(F).toContain('    asm volatile("{' + BS + 'n .reg .pred p;' + BS + 'n setp.ne.u32 p, %2, 99;' + BS
      + 'n @p red.shared.xor.b32 [%0], %1;' + BS + 'n}"');
    // The TMA build streams its stage body (PEARL_TALL_FRAG_PIPE) and XORs its readout into shared.
    expect(HEADER).toContain(`#ifndef PEARL_TALL_FRAG_PIPE${NL}#define PEARL_TALL_FRAG_PIPE 1${NL}#endif`);
    // Both helpers are volatile asm, so the front end keeps them too.
    expect(KERNEL).toContain(lines(
      '  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"',
      '               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)',
      '               : "r"(a));'));
    expect(KERNEL).toContain(lines(
      '  asm volatile(',
      '      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "',
      '      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};' + BS + 'n"'));
  });

  test('FULL expects exactly the bytes of the boxes each arm still issues', () => {
    // The geometry the fold's constants come from: B' 256 columns and A' 192 rows of a 64-byte stage.
    expect(HEADER).toContain('#define PEARL_TALL_ROW_TILES 6u');
    expect(HEADER).toContain('#define PEARL_TALL_BM (2u * PEARL_TALL_ROW_TILES * 16u)');
    expect(HEADER).toContain('#define PEARL_TALL_BN (4u * 64u)');
    expect(HEADER).toContain('#define PEARL_TALL_STAGE_K 64u');
    expect(HEADER).toContain('#define PEARL_TALL_STAGE_BYTES ((PEARL_TALL_BM + PEARL_TALL_BN) * PEARL_TALL_STAGE_K)');
    expect(FOLD).toContain('  constexpr uint32_t BM = PEARL_TALL_BM, BN = PEARL_TALL_BN;');
    expect(FOLD).toContain('  constexpr uint32_t SK = PEARL_TALL_STAGE_K;');
    expect(FOLD).toContain('  constexpr uint32_t STAGE = PEARL_TALL_STAGE_BYTES;');
    const BM = 2 * 6 * 16, BN = 4 * 64, SK = 64, STAGE = (BM + BN) * SK;
    // The whole fill: both boxes, B' at the buffer's start and A' after it, and all their bytes.
    expect(code(FOLD)).toContain(lines(
      '    pearl_mbar_expect_tx(full, STAGE);',
      '#if PEARL_TALL_CLUSTER_BODY',
      '    if (pearl_cluster_ctarank() == 0u)',
      '      pearl_tma_3d_multicast(dst, &tmB, 0u, boxB, kofs / SK, full,',
      '                             (uint16_t)((1u << CLUSTER) - 1u));',
      '#else',
      '    pearl_tma_3d(dst, &tmB, 0u, boxB, kofs / SK, full);',
      '#endif',
      '    pearl_tma_3d(dst + BN * SK, &tmA, 0u, boxA, kofs / SK, full);'));
    expect(STAGE).toBe(BN * SK + BM * SK);
    // The skip (SKIP, pinned above), arm by arm: a mask bit drops its box and its bytes; with
    // both dropped the producer's plain arrival is FULL's one arrival and no byte is expected.
    for (const mask of [1, 2, 3]) {
      const skipA = (mask & 1) !== 0;
      const skipB = (mask & 2) !== 0;
      const boxes = [];
      if (!skipB) boxes.push({ at: 0, bytes: BN * SK });
      if (!skipA) boxes.push({ at: BN * SK, bytes: BM * SK });
      const landed = boxes.reduce((s, b) => s + b.bytes, 0);
      if (skipA && skipB) {
        expect([mask, boxes]).toEqual([mask, []]);
      } else {
        const expected = (skipB ? 0 : BN * SK) + (skipA ? 0 : BM * SK);
        expect([mask, expected]).toEqual([mask, landed]);
        expect(expected).toBeGreaterThan(0);
      }
      expect(landed).toBe(mask === 1 ? BN * SK : mask === 2 ? BM * SK : 0);
    }
    // FULL counts one arrival, the producer's, as on the whole fill.
    expect(code(FOLD)).toContain('    pearl_mbar_init(sbase + threadIdx.x * STRIDE + STAGE, 1u);');
    expect(KERNEL).toContain(lines(
      '__device__ __forceinline__ void pearl_mbar_arrive(uint32_t bar) {',
      '  asm volatile("{' + BS + 'n' + BS + 't.reg .b64 st;' + BS + 'n' + BS
        + 'tmbarrier.arrive.shared.b64 st, [%0];' + BS + 'n' + BS + 't}" ::"r"(bar)'));
  });

  test('every buffer holds real operands before the first skip, and only the first tile loads', () => {
    // The tile walk: v starts at blockIdx.x and steps by the grid, so the first tile is the only
    // one with v below tile_stride (the cluster build, which walks pairs, refuses the switch).
    const F = code(FOLD);
    expect(F).toContain('  const uint32_t tile_stride = gridDim.x;');
    expect(F).toContain('  const uint32_t tile_first = blockIdx.x;');
    expect(F).toContain('#define PEARL_TALL_NEXT (v + tile_stride)');
    expect(F).toContain('  for (uint32_t v = tile_first; v < tiles; v = PEARL_TALL_NEXT) {');
    // The ring as the source has it. Roles (PEARL_TALL_TMA_ROLES, the default): stage 0 reads A
    // and refills C, stage 1 reads B and refills A, and the roles rotate each chunk. Without
    // roles: stage g reads buffer g % 3 and refills the one before it. Both refill at the top
    // of the stage (PEARL_TALL_TMA_FILL_PT 0), before the stage's FULL wait.
    expect(F).toContain(lines(
      '        const uint32_t so = s == 0u ? soA : soB;',
      '        const uint32_t bi = s == 0u ? soC : soA;'));
    expect(F).toContain(lines(
      '        const uint32_t tso = soC, tpar = pC;',
      '        soC = soB;',
      '        pC = pB ^ 1u;',
      '        soB = soA;',
      '        pB = pA ^ 1u;',
      '        soA = tso;',
      '        pA = tpar;'));
    expect(F).toContain('  uint32_t soA = sbase, soB = sbase + STRIDE, soC = sbase + 2u * STRIDE;');
    expect(F).toContain('        const uint32_t bi = fb == 0u ? 2u : fb - 1u;');
    expect(F).toContain(lines(
      '#if PEARL_TALL_TMA_FILL_PT == 0',
      '        PEARL_TALL_REFILL(PEARL_TALL_PRODUCER_S)',
      '#endif',
      '#ifndef PEARL_ABLATE_RING',
      '        pearl_mbar_wait(fullSo, rpar);'));
    expect(HEADER).toContain(`#ifndef PEARL_TALL_TMA_FILL_PT${NL}#define PEARL_TALL_TMA_FILL_PT 0${NL}#endif`);
    // Mainnet on a 5090 (683 row groups by 128 column tiles at col_batch 2048, 170 blocks), on
    // a 5080 (by 64 at col_batch 1024, 84 blocks), and a small grid; 16 chunks of 2 stages.
    const CHUNKS = 16;
    const SPC = 2;
    for (const roles of [true, false]) {
      for (const [tiles, grid] of [[683 * 128, 170], [683 * 64, 84], [40, 7]]) {
        for (const block of [0, 1, grid - 1]) {
          const held = [true, true, false];   // the prologue fills buffers 0 and 1 whole
          const unheld = [];
          let A = 0, B = 1, C = 2, g = 0, loads = SPC, skips = 0, firstSkip = null;
          for (let v = block; v < tiles; v += grid) {
            const hasNext = v + grid < tiles;
            for (let chunk = 0; chunk < CHUNKS; chunk++) {
              for (let s = 0; s < SPC; s++, g++) {
                const so = roles ? (s === 0 ? A : B) : g % 3;
                const bi = roles ? (s === 0 ? C : A) : (g % 3 === 0 ? 2 : g % 3 - 1);
                if (chunk + 1 < CHUNKS || hasNext) {
                  if (v < grid) {   // the hook: ab_load = v < tile_stride
                    held[bi] = true;
                    loads++;
                  } else {
                    if (!firstSkip) firstSkip = held.slice();
                    skips++;
                  }
                }
                if (!held[so]) unheld.push([v, chunk, s]);   // a read of a buffer never filled
              }
              if (roles) [A, B, C] = [C, A, B];
            }
          }
          const walk = Math.ceil((tiles - block) / grid);
          const at = [roles, tiles, grid, block];
          expect([...at, unheld]).toEqual([...at, []]);
          expect([...at, walk > 1, firstSkip]).toEqual([...at, true, [true, true, true]]);
          // The first tile's 32 stages and the next tile's first chunk load; nothing after.
          expect([...at, loads, loads + skips]).toEqual([...at, CHUNKS * SPC + SPC, walk * CHUNKS * SPC]);
          // At mainnet that leaves about 0.2% of the operand fills, as pearl_config.h says.
          if (grid > 7) expect([...at, loads / (loads + skips) < 0.0025]).toEqual([...at, true]);
        }
      }
    }
  });

  test('the host refuses to mine with it, and names the arm in the fold\'s name', () => {
    const submit = code(body(HOST, 'extern "C" bool pearl_host_submit('));
    inOrder(submit,
      'if (!ctx || !ctx->haveJob) return false;',
      '#if PEARL_TALL_STUB_SPIN || PEARL_TALL_STUB_ALU || PEARL_TALL_AB_SKIP',
      'for (int i = 0; i < PEARL_HASH_BYTES; i++) zero = zero && ctx->target[i] == 0;',
      'if (!zero) {',
      '#if PEARL_TALL_AB_SKIP',
      '"tall-fold operand-skip build is timing-only: it runs only at "',
      'if (ctx->pendCount >= Ctx::kSlots) {',
      'DeviceScope scope(ctx->device);');
    // Nothing is queued or launched before it.
    const guard = submit.indexOf('#if PEARL_TALL_STUB_SPIN || PEARL_TALL_STUB_ALU || PEARL_TALL_AB_SKIP');
    expect(submit.slice(0, guard)).not.toMatch(/<<<|cudaLaunch|pendCount\+\+/);
    // Each arm's build: the refusal with the skip's message, and not the stubs'.
    for (const mask of [1, 2, 3]) {
      const on = resolve(submit, { ...STUBS_OFF, [NAME]: mask });
      expect([mask, on.includes(lines(
        '  {',
        '    bool zero = true;',
        '    for (int i = 0; i < PEARL_HASH_BYTES; i++) zero = zero && ctx->target[i] == 0;',
        '    if (!zero) {',
        '      if (err && err_len)',
        '        snprintf(err, err_len, "tall-fold operand-skip build is timing-only: it runs only at "',
        '                 "target 0 (hashrate.js, race.cu), never on a job; build without "',
        '                 "PEARL_TALL_AB_SKIP to mine");',
        '      return false;',
        '    }',
        '  }'))]).toEqual([mask, true]);
      expect(on).not.toContain('tall-fold stub build');
    }
    // Off, with neither stub, there is no refusal at all.
    expect(resolve(submit, { ...STUBS_OFF, [NAME]: 0 })).not.toContain('bool zero = true;');
    // The target it checks is the job's, and every way to search goes through it.
    expect(code(body(HOST, 'extern "C" void pearl_host_set_job_salted('))).toContain(
      'memcpy(ctx->target, target, PEARL_HASH_BYTES);');
    expect(code(body(HOST, 'extern "C" bool pearl_host_search('))).toContain('if (!pearl_host_submit(handle, ');
    // The race sets target zero, which a skip build runs at.
    expect(RACE).toContain('memset(target, 0x00, sizeof target);');
    // The fold's name: the TMA ring's, then the arm, after the cluster's and Ampere's names and
    // before sm_100's, which it carries itself.
    const name = code(body(HOST, 'extern "C" const char *pearl_host_fold_name('));
    inOrder(name,
      'if (ctx->foldTall && ctx->foldCluster)',
      'if (ctx->foldAmpere) {',
      '#if PEARL_TALL_AB_SKIP',
      'if (ctx->foldTall && ctx->foldTma) {',
      'if (ctx->foldTall && ctx->foldTma && PEARL_SM100_ARCH(ctx->tallArch * 10))');
    expect(name).toContain(lines(
      '    snprintf(const_cast<Ctx *>(ctx)->foldName, sizeof ctx->foldName,',
      '             "tall 192x256, 8 warps of 96x64, TMA ring, k-blocked operands%s, AB_SKIP=%d: %s "',
      '             "skipped after each block\'s first tile (timing only)",',
      '             PEARL_SM100_ARCH(ctx->tallArch * 10) ? ", sm_100 build of sm_120\'s fold" : "",',
      '             (int)PEARL_TALL_AB_SKIP,',
      '             PEARL_TALL_AB_SKIP == 3 ? "A\' and B\'" : (PEARL_TALL_AB_SKIP == 1 ? "A\'" : "B\'"));',
      '    return ctx->foldName;'));
    expect(name).toContain('return ctx->foldTma ? "tall 192x256, 8 warps of 96x64, TMA ring, k-blocked operands"');
    // Every arm's name fits the context's buffer, and names the operands its mask skips.
    expect(HOST).toContain('  char foldName[224] = {0};');
    for (const mask of [1, 2, 3]) {
      for (const sm100 of [false, true]) {
        const which = mask === 3 ? 'A\' and B\'' : mask === 1 ? 'A\'' : 'B\'';
        const text = 'tall 192x256, 8 warps of 96x64, TMA ring, k-blocked operands'
          + (sm100 ? ', sm_100 build of sm_120\'s fold' : '')
          + `, AB_SKIP=${mask}: ${which} skipped after each block's first tile (timing only)`;
        expect(text.length).toBeLessThan(224);
      }
    }
    // compilers.sh builds each arm's host with the arm's flags, so a skip build's own host
    // always has the refusal.
    expect(COMPILERS).toContain('nvcc $COMMON $flags "${keep[@]}" -Xptxas -v -c "$K/src/pearl_kernel.cu"');
    expect(COMPILERS).toContain('nvcc $COMMON $flags -c "$K/src/pearl_host.cu" -o "$K/cuda-build/pearl_host.o"');
  });

  test('no build that ships names it', () => {
    for (const file of SHIPPED) {
      const text = rawRead(path.join(ROOT, file));
      expect(text.length).toBeGreaterThan(0);
      expect([file, text.includes(NAME)]).toEqual([file, false]);
    }
  });
});
