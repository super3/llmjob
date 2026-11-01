'use strict';

const fs = require('fs');
const path = require('path');
const { keyedHash } = require('../src/shared/miner/blake3');

// Turing's instruction-cache switches (pearl_config.h, both off by default). They replace
// only the filter's compression in the B-direct fold's hash_tile:
// - PEARL_BD_STUB_HASH, timing only: a small stand-in whose top word is not the hash's.
//   The host refuses any target but zero, so it can never mine.
// - PEARL_BD_ROLLED_HASH: the same compression with rounds 0-5 in a loop.
//
// Nothing here runs device code. These pin both functions to the source, then run them
// on the CPU with the g calls and the permutation read from that source. They check:
// - both switches off by default, and the guards, through a small preprocessor;
// - the host's refusal, before anything is queued, and the fold-name tags;
// - that no build that ships names either switch;
// - the rolled word against BLAKE3's keyed hash for 2000 random keys and transcripts,
//   in both byte orders;
// - that the stub's word is not the hash's, depends on every key and transcript word,
//   and that a hit still needs the full hash.

const SRC = path.join(__dirname, '..', 'native', 'src');
const NL = String.fromCharCode(10);
const BS = String.fromCharCode(92);
// Line endings normalised: a Windows checkout has CRLF.
const rawRead = (file) => fs.readFileSync(file, 'utf8').split(String.fromCharCode(13) + NL).join(NL);
const read = (name) => rawRead(path.join(SRC, name));
const FOLD = read('pearl_fold_bd.cuh');
const HEADER = read('pearl_config.h');
const HOST = read('pearl_host.cu');
const KERNEL = read('pearl_kernel.cu');
const ROOT = path.join(__dirname, '..', '..');
const SHIPPED = [
  '.github/workflows/native-core.yml',
  '.github/workflows/miner-build.yml',
  'earn/native/build-local.ps1',
];
const COMPILERS = rawRead(path.join(SRC, '..', 'probes', 'turing', 'compilers.sh'));
const RACE = rawRead(path.join(SRC, '..', 'probes', 'race.cu'));
const HASHRATE = rawRead(path.join(SRC, '..', 'probes', 'hashrate.js'));

// Source without comments: comment lines dropped, trailing ones cut. The pins below
// hold the code, so a reworded comment does not break them.
function code(text) {
  return text.split(NL).filter((l) => !l.trim().startsWith('//')).map((l) => {
    const at = l.indexOf('//');
    return at > 0 && l.slice(0, at).trim() ? l.slice(0, at).trimEnd() : l;
  }).join(NL);
}
const FOLD_CODE = code(FOLD);
const lines = (...l) => l.join(NL);

// The body of the function or lambda whose head is head, braces and all.
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

function xorshift(seed) {
  let x = (seed >>> 0) || 1;
  return () => {
    x ^= x << 13; x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5; x >>>= 0;
    return x;
  };
}

// ---------------------------------------------------------------------------
// The two functions as the source has them (pinned in 'the source is as run here').

const STUB_HEAD = '__device__ __forceinline__ uint32_t pearl_bd_stub_msw(';
const ROLLED_HEAD = '__device__ __forceinline__ uint32_t pearl_bd_rolled_msw(';
const STUB = code(body(FOLD, STUB_HEAD));
const ROLLED = code(body(FOLD, ROLLED_HEAD));

const rotr = (x, n) => ((x >>> n) | (x << (32 - n))) >>> 0;
// pearl_kernel.cu's g().
function g(s, a, b, c, d, mx, my) {
  s[a] = (s[a] + s[b] + mx) >>> 0;
  s[d] = rotr((s[d] ^ s[a]) >>> 0, 16);
  s[c] = (s[c] + s[d]) >>> 0;
  s[b] = rotr((s[b] ^ s[c]) >>> 0, 12);
  s[a] = (s[a] + s[b] + my) >>> 0;
  s[d] = rotr((s[d] ^ s[a]) >>> 0, 8);
  s[c] = (s[c] + s[d]) >>> 0;
  s[b] = rotr((s[b] ^ s[c]) >>> 0, 7);
}
const bswap = (x) => (((x & 0xff) << 24) | ((x & 0xff00) << 8) | ((x >>> 8) & 0xff00) | (x >>> 24)) >>> 0;
// pearl_hash_word_msf(h, 0, hbe): the top word, word 0 byte-reversed or word 7.
const msf = (h, hbe) => (hbe ? bswap(h[0]) : h[7]);
const IV = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a];
const FLAGS = 1 | 2 | 8 | 16;  // PEARL_TRANSCRIPT_FLAGS: CHUNK_START | CHUNK_END | ROOT | KEYED_HASH

// The g calls in text, in order: [a, b, c, d, x, y] for g(s, a, b, c, d, m[x], m[y]).
function gCalls(text) {
  return [...text.matchAll(/g\(s, (\d+), (\d+), (\d+), (\d+), m\[(\d+)\], m\[(\d+)\]\);/g)]
    .map((x) => x.slice(1).map(Number));
}
// The loop count in text's one for loop over the named counter.
function passes(text, v) {
  const all = [...text.matchAll(new RegExp(`for \\(int ${v} = 0; ${v} < (\\d+); ${v}\\+\\+\\) \\{`, 'g'))];
  expect([v, all.length]).toEqual([v, 1]);
  return Number(all[0][1]);
}

// pearl_bd_rolled_msw, from the source: the lambda's g calls, the loop's passes, each a
// round and the permutation (pN = m[i], then m[N] = pN), the rounds after the loop, and
// the output words the return line xors.
function rolledFromSource(text) {
  const round = gCalls(body(text, 'auto one_round = [&]() {'));
  const loopHead = `for (int r = 0; r < ${passes(text, 'r')}; r++) {`;
  const loop = body(text, loopHead);
  const after = text.slice(text.indexOf(loopHead) + loopHead.length - 1 + loop.length);
  const count = (t, s) => t.split(s).length - 1;
  expect([count(loop, 'one_round();'), loop.indexOf('one_round();') < loop.indexOf('p0 = ')]).toEqual([1, true]);
  const p = [];
  for (const x of loop.matchAll(/p(\d+) = m\[(\d+)\]/g)) p[Number(x[1])] = Number(x[2]);
  for (let i = 0; i < 16; i++) expect(loop).toContain(`m[${i}] = p${i};`);
  // out[0] and out[7] as s[i] ^ s[j].
  const o = /const uint32_t out\[8\] = \{s\[(\d+)\] \^ s\[(\d+)\], 0u, 0u, 0u, 0u, 0u, 0u, s\[(\d+)\] \^ s\[(\d+)\]\};/
    .exec(after).slice(1).map(Number);
  expect(after).toContain('return pearl_hash_word_msf(out, 0, hash_big_endian);');
  const n = passes(text, 'r');
  const tail = count(after, 'one_round();');
  return (key, m0, hbe) => {
    const m = m0.slice();
    const s = [...key, ...IV, 0, 0, 64, FLAGS];
    const run = () => { for (const [a, b, c, d, x, y] of round) g(s, a, b, c, d, m[x], m[y]); };
    for (let r = 0; r < n; r++) {
      run();
      const q = p.map((i) => m[i]);
      for (let i = 0; i < 16; i++) m[i] = q[i];
    }
    for (let r = 0; r < tail; r++) run();
    return msf([(s[o[0]] ^ s[o[1]]) >>> 0, 0, 0, 0, 0, 0, 0, (s[o[2]] ^ s[o[3]]) >>> 0], hbe);
  };
}

// pearl_bd_stub_msw, from the source: words 8-15 xored into the key, the loop's g calls
// and passes, then every state word xored together, taken as the top word.
function stubFromSource(text) {
  const steps = gCalls(text);
  const n = passes(text, 'h');
  expect(text).toContain(lines('  const uint32_t out[8] = {x, 0u, 0u, 0u, 0u, 0u, 0u, x};',
    '  return pearl_hash_word_msf(out, 0, hash_big_endian);'));
  return (key, m, hbe) => {
    const s = [...key.map((k, i) => (k ^ m[8 + i]) >>> 0), ...IV, 0, 0, 64, FLAGS];
    for (let h = 0; h < n; h++) for (const [a, b, c, d, x, y] of steps) g(s, a, b, c, d, m[x], m[y]);
    const x = s.reduce((acc, v) => (acc ^ v) >>> 0, 0);
    return msf([x, 0, 0, 0, 0, 0, 0, x], hbe);
  };
}

// The hash's top word, as pearl_transcript_msw has it: output word 7, or word 0 byte-reversed.
function hashTopWord(kw, m, hbe) {
  const key = Buffer.alloc(32);
  const msg = Buffer.alloc(64);
  kw.forEach((w, i) => key.writeUInt32LE(w >>> 0, 4 * i));
  m.forEach((w, i) => msg.writeUInt32LE(w >>> 0, 4 * i));
  const out = keyedHash(key, msg);
  return hbe ? bswap(out.readUInt32LE(0)) : out.readUInt32LE(28);
}

function randomCases(seed, count) {
  const rand = xorshift(seed);
  const cases = [
    [new Array(8).fill(0), new Array(16).fill(0)],
    [new Array(8).fill(0xffffffff), new Array(16).fill(0xffffffff)],
  ];
  while (cases.length < count) {
    cases.push([Array.from({ length: 8 }, () => rand()), Array.from({ length: 16 }, () => rand())]);
  }
  return cases;
}

// ---------------------------------------------------------------------------
// A small C preprocessor for the guards, as turingHashSplit.test.js has it: object-like
// #define, #undef, #if, #ifdef, #ifndef, #elif, #else, #endif and #error, with -D
// flags. Returns the #error lines it reaches and the definitions.
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

// What a build of the fold goes through: the header's Turing block, from its first
// switch to the forced block size, and the fold's own #error lines.
function guards() {
  const slice = (src, from, to) => {
    const at = src.indexOf(from);
    const end = src.indexOf(to, at);
    expect(at).toBeGreaterThanOrEqual(0);
    expect(end).toBeGreaterThan(at);
    return src.slice(at, end);
  };
  return slice(HEADER, '#ifndef PEARL_TURING_WIDE' + NL, NL + '// A block size forced from the command line') + NL
    + slice(FOLD, '#if PEARL_BD_A_DIRECT && (defined(PEARL_ABLATE_BARRIER)', '#if PEARL_BD_GROUPS == 2');
}

// ---------------------------------------------------------------------------

describe('Turing instruction-cache switches (PEARL_BD_STUB_HASH, PEARL_BD_ROLLED_HASH)', () => {
  test('are off by default, one at a time, and only on the B-direct fold\'s own hash', () => {
    for (const name of ['PEARL_BD_STUB_HASH', 'PEARL_BD_ROLLED_HASH']) {
      expect(HEADER).toContain(`#ifndef ${name}${NL}#define ${name} 0${NL}#endif`);
    }
    const GUARDS = guards();
    const errorsFor = (flags) => preprocess(GUARDS, flags).errors;
    const value = (name, flags) => evaluate(name, preprocess(GUARDS, flags).defs);
    expect([value('PEARL_BD_STUB_HASH', {}), value('PEARL_BD_ROLLED_HASH', {})]).toEqual([0, 0]);
    const D = (o) => Object.fromEntries(Object.entries(o)
      .map(([k, v]) => [k.startsWith('PEARL_') ? k : 'PEARL_BD_' + k, v]));
    for (const f of [{}, { STUB_HASH: 1 }, { ROLLED_HASH: 1 }, { STUB_HASH: 0, ROLLED_HASH: 0 }]) {
      expect([f, errorsFor(D(f))]).toEqual([f, []]);
    }
    const LOADS = { RUNBASE: 1, TAIL_LDG: 1 };
    // Each refused with an error that names the switch, and each fine without it.
    const refused = [
      [{ STUB_HASH: 2 }, {}], [{ ROLLED_HASH: -1 }, {}],
      [{ STUB_HASH: 1, ROLLED_HASH: 1 }, { STUB_HASH: 1 }],
      [{ STUB_HASH: 1, PEARL_ABLATE_TRANSCRIPT_HASH: 1 }, { PEARL_ABLATE_TRANSCRIPT_HASH: 1 }],
      [{ ROLLED_HASH: 1, PEARL_ABLATE_TRANSCRIPT_HASH: 1 }, { PEARL_ABLATE_TRANSCRIPT_HASH: 1 }],
      [{ STUB_HASH: 1, HASH_SPLIT: 4, ...LOADS }, { HASH_SPLIT: 4, ...LOADS }],
      [{ ROLLED_HASH: 1, HASH_SPLIT: 4, ...LOADS }, { HASH_SPLIT: 4, ...LOADS }],
      [{ ROLLED_HASH: 1, PEARL_TURING_BDIRECT: 0 }, null],
    ];
    for (const [f, without] of refused) {
      const errors = errorsFor(D(f));
      const named = errors.some((e) => e.includes('PEARL_BD_STUB_HASH') || e.includes('PEARL_BD_ROLLED_HASH'));
      expect([f, named]).toEqual([f, true]);
      if (without) expect([without, errorsFor(D(without))]).toEqual([without, []]);
    }
  });

  test('the source is as run here, line for line', () => {
    // The stand-in: 12 passes of the column step on words 0-7, from the starting state
    // with words 8-15 xored into the key, then all 16 state words xored.
    expect(FOLD).toContain('#if PEARL_BD_STUB_HASH' + NL + '// ');
    expect(STUB).toBe(lines(
      '{',
      '  uint32_t s[16];',
      '#pragma unroll',
      '  for (int i = 0; i < 8; i++) s[i] = key[i] ^ m[8 + i];',
      '  s[8] = BLAKE3_IV[0]; s[9] = BLAKE3_IV[1]; s[10] = BLAKE3_IV[2]; s[11] = BLAKE3_IV[3];',
      '  s[12] = 0u; s[13] = 0u; s[14] = 64u; s[15] = PEARL_TRANSCRIPT_FLAGS;',
      '#pragma unroll 1',
      '  for (int h = 0; h < 12; h++) {',
      '    g(s, 0, 4, 8, 12, m[0], m[1]);',
      '    g(s, 1, 5, 9, 13, m[2], m[3]);',
      '    g(s, 2, 6, 10, 14, m[4], m[5]);',
      '    g(s, 3, 7, 11, 15, m[6], m[7]);',
      '  }',
      '  uint32_t x = 0u;',
      '#pragma unroll',
      '  for (int i = 0; i < 16; i++) x ^= s[i];',
      '  const uint32_t out[8] = {x, 0u, 0u, 0u, 0u, 0u, 0u, x};',
      '  return pearl_hash_word_msf(out, 0, hash_big_endian);',
      '}'));
    expect(FOLD).toContain(STUB_HEAD + 'const uint32_t key[8], const uint32_t m[16],' + NL
      + '                                                      int hash_big_endian) {');
    // The rolled compression: rounds 0-5 in a loop that is not unrolled, each with the
    // permutation as 16 moves, then round 6.
    expect(FOLD).toContain('#if PEARL_BD_ROLLED_HASH' + NL + '// ');
    expect(ROLLED).toBe(lines(
      '{',
      '  uint32_t s[16], m[16];',
      '#pragma unroll',
      '  for (int i = 0; i < 16; i++) m[i] = m0[i];',
      '#pragma unroll',
      '  for (int i = 0; i < 8; i++) s[i] = key[i];',
      '  s[8] = BLAKE3_IV[0]; s[9] = BLAKE3_IV[1]; s[10] = BLAKE3_IV[2]; s[11] = BLAKE3_IV[3];',
      '  s[12] = 0u; s[13] = 0u; s[14] = 64u; s[15] = PEARL_TRANSCRIPT_FLAGS;',
      '  auto one_round = [&]() {',
      '    g(s, 0, 4, 8, 12, m[0], m[1]);',
      '    g(s, 1, 5, 9, 13, m[2], m[3]);',
      '    g(s, 2, 6, 10, 14, m[4], m[5]);',
      '    g(s, 3, 7, 11, 15, m[6], m[7]);',
      '    g(s, 0, 5, 10, 15, m[8], m[9]);',
      '    g(s, 1, 6, 11, 12, m[10], m[11]);',
      '    g(s, 2, 7, 8, 13, m[12], m[13]);',
      '    g(s, 3, 4, 9, 14, m[14], m[15]);',
      '  };',
      '#pragma unroll 1',
      '  for (int r = 0; r < 6; r++) {',
      '    one_round();',
      '    const uint32_t p0 = m[2], p1 = m[6], p2 = m[3], p3 = m[10];',
      '    const uint32_t p4 = m[7], p5 = m[0], p6 = m[4], p7 = m[13];',
      '    const uint32_t p8 = m[1], p9 = m[11], p10 = m[12], p11 = m[5];',
      '    const uint32_t p12 = m[9], p13 = m[14], p14 = m[15], p15 = m[8];',
      '    m[0] = p0; m[1] = p1; m[2] = p2; m[3] = p3;',
      '    m[4] = p4; m[5] = p5; m[6] = p6; m[7] = p7;',
      '    m[8] = p8; m[9] = p9; m[10] = p10; m[11] = p11;',
      '    m[12] = p12; m[13] = p13; m[14] = p14; m[15] = p15;',
      '  }',
      '  one_round();',
      '  const uint32_t out[8] = {s[0] ^ s[8], 0u, 0u, 0u, 0u, 0u, 0u, s[7] ^ s[15]};',
      '  return pearl_hash_word_msf(out, 0, hash_big_endian);',
      '}'));
    expect(FOLD).toContain(ROLLED_HEAD + 'const uint32_t key[8], const uint32_t m0[16],' + NL
      + '                                                        int hash_big_endian) {');
    // Read back the way the CPU copies read them.
    expect(gCalls(STUB)).toHaveLength(4);
    expect(gCalls(body(ROLLED, 'auto one_round = [&]() {'))).toHaveLength(8);
    expect([passes(STUB, 'h'), passes(ROLLED, 'r')]).toEqual([12, 6]);

    // The call site: one filter per build, then the full hash for anything that passes,
    // so a hit the stub reports is still a real one.
    expect(FOLD_CODE).toContain(lines(
      '#if PEARL_BD_STUB_HASH',
      '        if (pearl_bd_stub_msw(test.key, tm, test.hash_big_endian) > test.target_w[0]) return;',
      '#elif PEARL_BD_ROLLED_HASH',
      '        if (pearl_bd_rolled_msw(test.key, tm, test.hash_big_endian) > test.target_w[0]) return;',
      '#else',
      '        if (pearl_transcript_msw(test.key, tm, test.hash_big_endian) > test.target_w[0]) return;',
      '#endif',
      '        uint32_t h[8];',
      '        pearl_transcript_hash_again(test.key, tm, h);',
      '        if (!pearl_hash_meets_words(h, test.target_w, test.hash_big_endian)) return;'));
    expect(FOLD_CODE.split('pearl_bd_stub_msw(').length - 1).toBe(2);
    expect(FOLD_CODE.split('pearl_bd_rolled_msw(').length - 1).toBe(2);
    // The functions are defined before hash_tile uses them.
    inOrder(FOLD_CODE, STUB_HEAD, ROLLED_HEAD, 'auto hash_tile = [&](uint32_t vtile) {');

    // pearl_kernel.cu's pieces they share with the shipped hash.
    expect(KERNEL).toContain(lines(
      '  s[a] = s[a] + s[b] + mx;', '  s[d] = rotr32(s[d] ^ s[a], 16);', '  s[c] = s[c] + s[d];',
      '  s[b] = rotr32(s[b] ^ s[c], 12);', '  s[a] = s[a] + s[b] + my;', '  s[d] = rotr32(s[d] ^ s[a], 8);',
      '  s[c] = s[c] + s[d];', '  s[b] = rotr32(s[b] ^ s[c], 7);'));
    expect(KERNEL).toContain('  return (x >> n) | (x << (32 - n));');
    expect(KERNEL).toContain('#define PEARL_TRANSCRIPT_FLAGS (CHUNK_START | CHUNK_END | ROOT | KEYED_HASH)');
    for (const [name, bit] of [['CHUNK_START', 0], ['CHUNK_END', 1], ['ROOT', 3], ['KEYED_HASH', 4]]) {
      expect(KERNEL).toContain(`#define ${name} (1u << ${bit})`);
    }
    expect(KERNEL).toContain('__constant__ uint32_t BLAKE3_IV[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u,'
      + NL + '                                      0xA54FF53Au,');
    expect(KERNEL).toContain('return __byte_perm(x, 0u, 0x0123u);');
    expect(KERNEL).toContain('return hash_big_endian ? pearl_bswap32(h[i]) : h[7 - i];');
    // The shipped filter's word: output word 7, or word 0 byte-reversed. The fold picks
    // its words only through pearl_hash_word_msf (nativeConfig.test.js).
    expect(KERNEL).toContain('return hash_big_endian ? pearl_bswap32(out16[0]) : out16[7];');
    expect(KERNEL).toContain('out[i] = s[i] ^ s[i + 8];');
    expect(FOLD).not.toContain('pearl_bswap32(');
  });

  test('the rolled word is the hash\'s top word, in both byte orders', () => {
    const rolled = rolledFromSource(ROLLED);
    const bad = [];
    const cases = randomCases(0x1a2b3c4d, 2000);
    for (let n = 0; n < cases.length; n++) {
      const [kw, m] = cases[n];
      for (const hbe of [0, 1]) {
        const want = hashTopWord(kw, m, hbe);
        const got = rolled(kw, m, hbe);
        if (got !== want && bad.length < 8) bad.push(`case ${n} hbe ${hbe}: ${got} not ${want}`);
      }
    }
    expect(bad).toEqual([]);
  });

  test('the stub\'s word is not the hash\'s, and every key and transcript word feeds it', () => {
    const stub = stubFromSource(STUB);
    const cases = randomCases(0x5eed1234, 2000);
    let same = 0;
    for (const [kw, m] of cases) {
      for (const hbe of [0, 1]) if (stub(kw, m, hbe) === hashTopWord(kw, m, hbe)) same++;
    }
    expect(same).toBe(0);
    // Flip one bit of one word: the word changes, so none of the 16 loads or the key's
    // words can be dropped by the compiler.
    const rand = xorshift(77);
    for (let t = 0; t < 50; t++) {
      const kw = Array.from({ length: 8 }, () => rand());
      const m = Array.from({ length: 16 }, () => rand());
      const base = stub(kw, m, 0);
      for (let i = 0; i < 24; i++) {
        const k2 = kw.slice();
        const m2 = m.slice();
        const bit = 1 << (rand() & 31);
        if (i < 8) k2[i] = (k2[i] ^ bit) >>> 0;
        else m2[i - 8] = (m2[i - 8] ^ bit) >>> 0;
        expect([t, i, stub(k2, m2, 0) !== base]).toEqual([t, i, true]);
      }
    }
  });

  test('the host refuses to mine with the stub, and names both in the fold\'s name', () => {
    const submit = code(body(HOST, 'extern "C" bool pearl_host_submit('));
    inOrder(submit,
      'if (!ctx || !ctx->haveJob) return false;',
      '#if PEARL_BD_STUB_HASH',
      'for (int i = 0; i < PEARL_HASH_BYTES; i++) zero = zero && ctx->target[i] == 0;',
      'if (!zero) {',
      '"stub-hash build is timing-only: it runs only at target 0 "',
      'if (ctx->pendCount >= Ctx::kSlots) {',
      'DeviceScope scope(ctx->device);',
      'pearl_submit_dp4a(');
    // Nothing is launched before it.
    const guard = submit.indexOf('#if PEARL_BD_STUB_HASH');
    expect(submit.indexOf('<<<')).toBeGreaterThan(guard);
    expect(submit.slice(0, guard)).not.toContain('cudaLaunch');
    expect(submit.slice(guard, submit.indexOf('#endif', guard))).toContain(lines(
      '    if (!zero) {',
      '      if (err && err_len)',
      '        snprintf(err, err_len, "stub-hash build is timing-only: it runs only at target 0 "',
      '                 "(hashrate.js, race.cu), never on a job; build without PEARL_BD_STUB_HASH to mine");',
      '      return false;',
      '    }'));
    // pearl_host_search goes through pearl_host_submit.
    expect(code(body(HOST, 'extern "C" bool pearl_host_search('))).toContain('if (!pearl_host_submit(handle, ');
    // The timing tools set target zero, which the stub runs at.
    expect(RACE).toContain('memset(target, 0x00, sizeof target);');
    expect(HASHRATE).toContain('target: 0n');
    // The fold's name: after the other switches' list, which ends at its #endif, joining
    // it or starting a list of its own, then returned.
    const name = code(body(HOST, 'extern "C" const char *pearl_host_fold_name('));
    expect(name).toContain(lines(
      '      if (strlen(c->foldName) == named + strlen(", with")) c->foldName[named] = 0;',
      '    }',
      '#endif',
      '#if PEARL_BD_STUB_HASH || PEARL_BD_ROLLED_HASH',
      '    {',
      '      Ctx *c = const_cast<Ctx *>(ctx);',
      '      const size_t n = strlen(c->foldName);',
      '      snprintf(c->foldName + n, sizeof c->foldName - n, "%s%s", strstr(c->foldName, ", with") ? "" : ", with",',
      '               PEARL_BD_STUB_HASH ? " STUB_HASH" : " ROLLED_HASH");',
      '    }',
      '#endif',
      '    return ctx->foldName;'));
    // Only once, and only inside the B-direct branch.
    expect(name.split('" STUB_HASH"').length - 1).toBe(1);
    inOrder(name, 'if (ctx->foldBDirect) {', '" STUB_HASH"', 'if (ctx->foldTuring)');
    // compilers.sh builds each variant's host with the fold's flags, so a stub's own
    // host always has the refusal.
    expect(COMPILERS).toContain('nvcc $COMMON $flags "${keep[@]}" -Xptxas -v -c "$K/src/pearl_kernel.cu"');
    expect(COMPILERS).toContain('nvcc $COMMON $flags -c "$K/src/pearl_host.cu" -o "$K/cuda-build/pearl_host.o"');
  });

  test('no build that ships names either switch', () => {
    for (const file of SHIPPED) {
      const text = rawRead(path.join(ROOT, file));
      expect(text.length).toBeGreaterThan(0);
      for (const name of ['PEARL_BD_STUB_HASH', 'PEARL_BD_ROLLED_HASH']) {
        expect([file, name, text.includes(name)]).toEqual([file, name, false]);
      }
    }
  });
});
