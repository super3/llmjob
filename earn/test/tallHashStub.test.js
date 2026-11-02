'use strict';

const fs = require('fs');
const path = require('path');
const { keyedHash } = require('../src/shared/miner/blake3');
const { preprocess: resolve } = require('./helpers/amdSwitches');

// The tall fold's tile-hash timing stubs (pearl_config.h, both off by default), round 4's
// RTX 40 rank 1. On the cp.async ring's paired hash (sm_86, sm_89) they put a stand-in
// where the hasher's two compressions were:
// - PEARL_TALL_STUB_SPIN=N: a clock64 spin until N cycles after the hand-off, and no other work;
// - PEARL_TALL_STUB_ALU=1: an ALU burst of about the hash's issue count in a small loop;
//   2 the same burst as one chain, 3 with its adds as IMAD, 4 with 32 IMMA, 5 all IMAD.
// Both are timing only: the host refuses any target but zero, so neither can mine.
// The hasher's stamps (PEARL_TALL_HASHCYC, bench only) are in the second block below.
//
// Nothing here runs device code. These check:
// - both switches off by default, and the guards, through a small preprocessor;
// - the fold's stand-in site, and that a hit still needs the full hash;
// - the stand-ins as the source has them, and the bursts' words on the CPU: every key
//   and transcript word feeds them, and none is the hash's top word;
// - the host's refusal, before anything is queued, and the fold-name tags;
// - that no build that ships names either switch.

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
];
const PROBES = path.join(SRC, '..', 'probes');
const COMPILERS = rawRead(path.join(PROBES, 'turing', 'compilers.sh'));
const RACE = rawRead(path.join(PROBES, 'race.cu'));
const BENCH = rawRead(path.join(PROBES, 'bench.cu'));
const HASHRATE = rawRead(path.join(PROBES, 'hashrate.js'));
const SWITCHES = ['PEARL_TALL_STUB_SPIN', 'PEARL_TALL_STUB_ALU'];
const STAMPS = 'PEARL_TALL_HASHCYC';

// Source without comments: comment lines dropped, trailing ones cut. The pins below hold
// the code, so a reworded comment does not break them.
function code(text) {
  return text.split(NL).filter((l) => !l.trim().startsWith('//')).map((l) => {
    const at = l.indexOf('//');
    return at > 0 && l.slice(0, at).trim() ? l.slice(0, at).trimEnd() : l;
  }).join(NL);
}
const lines = (...l) => l.join(NL);
const KERNEL_CODE = code(KERNEL);
const FOLD_CODE = code(FOLD);

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
// A small C preprocessor for the guards, as turingHashSplit.test.js has it: object-like
// #define, #undef, #if, #ifdef, #ifndef, #elif, #else, #endif and #error, with -D flags.
// Returns the #error lines it reaches and the definitions.
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

// What a build goes through: the paired hash's default, then the stubs' block, from its
// first switch to the comment after its last guard.
function guards() {
  const pairs = lines('#ifndef PEARL_TALL_HASH_PAIRS', '#define PEARL_TALL_HASH_PAIRS 1', '#endif');
  const from = HEADER.indexOf('#ifndef PEARL_TALL_STUB_SPIN' + NL);
  const to = HEADER.indexOf(NL + '// Blackwell\'s tall fold makes its readout', from);
  expect(HEADER.indexOf(pairs)).toBeGreaterThan(0);
  expect(HEADER.indexOf(pairs)).toBeLessThan(from);
  expect(to).toBeGreaterThan(from);
  return pairs + NL + HEADER.slice(from, to);
}

// ---------------------------------------------------------------------------
// The bursts as the source has them (pinned in 'the stand-ins are as run here'): each
// STUB_ALU value's function, under its own #if or #elif line.
const ALU_HEAD = '__device__ __forceinline__ uint32_t pearl_tall_stub_alu(';
const SPIN_HEAD = '__device__ __forceinline__ void pearl_tall_stub_spin(long long t0) {';
function armBody(marker) {
  const at = KERNEL_CODE.indexOf(marker + NL + ALU_HEAD);
  expect([marker, at >= 0]).toEqual([marker, true]);
  return body(KERNEL_CODE.slice(at), ALU_HEAD);
}
const ALU = armBody('#if PEARL_TALL_STUB_ALU == 1');
const ALU_CHAIN = armBody('#elif PEARL_TALL_STUB_ALU == 2');
const ALU_IMAD = armBody('#elif PEARL_TALL_STUB_ALU == 3');
const ALU_MMA = armBody('#elif PEARL_TALL_STUB_ALU == 4');
const ALU_FMA = armBody('#elif PEARL_TALL_STUB_ALU == 5');
// g's twelve steps on chain c, as =1 and =4 have them.
const G_STEPS = [
  'a[c] = a[c] + d[c] + mx;',
  'd[c] = rotr32(d[c] ^ a[c], 16);',
  'a[c] = a[c] + d[c];',
  'd[c] = rotr32(d[c] ^ a[c], 12);',
  'a[c] = a[c] + d[c] + my;',
  'd[c] = rotr32(d[c] ^ a[c], 8);',
  'a[c] = a[c] + d[c];',
  'd[c] = rotr32(d[c] ^ a[c], 7);',
];
const rotr = (x, n) => ((x >>> n) | (x << (32 - n))) >>> 0;
const bswap = (x) => (((x & 0xff) << 24) | ((x & 0xff00) << 8) | ((x >>> 8) & 0xff00) | (x >>> 24)) >>> 0;

// pearl_tall_stub_alu, with its pass count and message words read from the source.
function aluFromSource(text) {
  const passes = [...text.matchAll(/for \(int r = 0; r < (\d+); r\+\+\) \{/g)];
  expect(passes).toHaveLength(1);
  const n = Number(passes[0][1]);
  const mm = /const uint32_t mx = tm\[(\d+)\], my = tp\[(\d+)\];/.exec(text);
  const mxi = Number(mm[1]);
  const myi = Number(mm[2]);
  return (key, tm, tp) => {
    const a = [];
    const d = [];
    for (let c = 0; c < 4; c++) {
      a[c] = (key[c] ^ tm[c] ^ tp[c] ^ tm[4 + c] ^ tp[4 + c]) >>> 0;
      d[c] = (key[4 + c] ^ tm[8 + c] ^ tp[8 + c] ^ tm[12 + c] ^ tp[12 + c]) >>> 0;
    }
    const mx = tm[mxi];
    const my = tp[myi];
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < 4; c++) {
        a[c] = (a[c] + d[c] + mx) >>> 0;
        d[c] = rotr((d[c] ^ a[c]) >>> 0, 16);
        a[c] = (a[c] + d[c]) >>> 0;
        d[c] = rotr((d[c] ^ a[c]) >>> 0, 12);
        a[c] = (a[c] + d[c] + my) >>> 0;
        d[c] = rotr((d[c] ^ a[c]) >>> 0, 8);
        a[c] = (a[c] + d[c]) >>> 0;
        d[c] = rotr((d[c] ^ a[c]) >>> 0, 7);
      }
    }
    return (a[0] ^ a[1] ^ a[2] ^ a[3] ^ d[0] ^ d[1] ^ d[2] ^ d[3]) >>> 0;
  };
}

// The pass count and message words of a burst's source.
function burstShape(text) {
  const passes = [...text.matchAll(/for \(int r = 0; r < (\d+); r\+\+\) \{/g)];
  expect(passes).toHaveLength(1);
  const mm = /const uint32_t mx = tm\[(\d+)\], my = tp\[(\d+)\];/.exec(text);
  return { n: Number(passes[0][1]), mxi: Number(mm[1]), myi: Number(mm[2]) };
}

// STUB_ALU=2: one chain of the same steps.
function chainFromSource(text) {
  const { n, mxi, myi } = burstShape(text);
  return (key, tm, tp) => {
    let a = 0;
    let d = 0;
    for (let c = 0; c < 4; c++) {
      a = (a ^ key[c] ^ tm[c] ^ tp[c] ^ tm[4 + c] ^ tp[4 + c]) >>> 0;
      d = (d ^ key[4 + c] ^ tm[8 + c] ^ tp[8 + c] ^ tm[12 + c] ^ tp[12 + c]) >>> 0;
    }
    const mx = tm[mxi];
    const my = tp[myi];
    for (let r = 0; r < n * 4; r++) {
      a = (a + d + mx) >>> 0;
      d = rotr((d ^ a) >>> 0, 16);
      a = (a + d) >>> 0;
      d = rotr((d ^ a) >>> 0, 12);
      a = (a + d + my) >>> 0;
      d = rotr((d ^ a) >>> 0, 8);
      a = (a + d) >>> 0;
      d = rotr((d ^ a) >>> 0, 7);
    }
    return (a ^ d) >>> 0;
  };
}

// STUB_ALU=3: the adds as d * one + a, the message words in the XORs.
function imadFromSource(text) {
  const { n, mxi, myi } = burstShape(text);
  return (key, tm, tp, one) => {
    const a = [];
    const d = [];
    for (let c = 0; c < 4; c++) {
      a[c] = (key[c] ^ tm[c] ^ tp[c] ^ tm[4 + c] ^ tp[4 + c]) >>> 0;
      d[c] = (key[4 + c] ^ tm[8 + c] ^ tp[8 + c] ^ tm[12 + c] ^ tp[12 + c]) >>> 0;
    }
    const mx = tm[mxi];
    const my = tp[myi];
    const mad = (x, y) => (Math.imul(x, one) + y) >>> 0;
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < 4; c++) {
        a[c] = mad(d[c], a[c]);
        d[c] = rotr((d[c] ^ a[c] ^ mx) >>> 0, 16);
        a[c] = mad(d[c], a[c]);
        d[c] = rotr((d[c] ^ a[c]) >>> 0, 12);
        a[c] = mad(d[c], a[c]);
        d[c] = rotr((d[c] ^ a[c] ^ my) >>> 0, 8);
        a[c] = mad(d[c], a[c]);
        d[c] = rotr((d[c] ^ a[c]) >>> 0, 7);
      }
    }
    return (a[0] ^ a[1] ^ a[2] ^ a[3] ^ d[0] ^ d[1] ^ d[2] ^ d[3]) >>> 0;
  };
}

// STUB_ALU=5: four chains of IMAD steps, a = d * one + a then d = a * one + d, six pairs a pass.
function fmaFromSource(text) {
  const passes = /for \(int r = 0; r < (\d+); r\+\+\) \{/.exec(text);
  const pairs = /for \(int s = 0; s < (\d+); s\+\+\) \{/.exec(text);
  const n = Number(passes[1]) * Number(pairs[1]);
  return (key, tm, tp, one) => {
    const a = [];
    const d = [];
    for (let c = 0; c < 4; c++) {
      a[c] = (key[c] ^ tm[c] ^ tp[c] ^ tm[4 + c] ^ tp[4 + c]) >>> 0;
      d[c] = (key[4 + c] ^ tm[8 + c] ^ tp[8 + c] ^ tm[12 + c] ^ tp[12 + c]) >>> 0;
    }
    const mad = (x, y) => (Math.imul(x, one) + y) >>> 0;
    for (let k = 0; k < n; k++) {
      for (let c = 0; c < 4; c++) {
        a[c] = mad(d[c], a[c]);
        d[c] = mad(a[c], d[c]);
      }
    }
    return (a[0] ^ a[1] ^ a[2] ^ a[3] ^ d[0] ^ d[1] ^ d[2] ^ d[3]) >>> 0;
  };
}

// The hash's top word, as pearl_hp_msw has it: output word 7, or word 0 byte-reversed.
function hashTopWord(kw, m, hbe) {
  const key = Buffer.alloc(32);
  const msg = Buffer.alloc(64);
  kw.forEach((w, i) => key.writeUInt32LE(w >>> 0, 4 * i));
  m.forEach((w, i) => msg.writeUInt32LE(w >>> 0, 4 * i));
  const out = keyedHash(key, msg);
  return hbe ? bswap(out.readUInt32LE(0)) : out.readUInt32LE(28);
}

// Every key and transcript bit reaches a burst's word, and the word is never the hash's.
function checkBurst(alu) {
  const rand = xorshift(0x7a11ab5e);
  const words = (n) => Array.from({ length: n }, () => rand());
  // Flip one bit of any of the 8 key and 32 transcript words: the word changes, so ptxas
  // must keep all 8 loads and wait for them, as the hash does.
  for (let t = 0; t < 40; t++) {
    const key = words(8);
    const tm = words(16);
    const tp = words(16);
    const base = alu(key, tm, tp);
    for (let i = 0; i < 40; i++) {
      const k2 = key.slice();
      const m2 = tm.slice();
      const p2 = tp.slice();
      const bit = 1 << (rand() & 31);
      if (i < 8) k2[i] = (k2[i] ^ bit) >>> 0;
      else if (i < 24) m2[i - 8] = (m2[i - 8] ^ bit) >>> 0;
      else p2[i - 24] = (p2[i - 24] ^ bit) >>> 0;
      expect([t, i, alu(k2, m2, p2) !== base]).toEqual([t, i, true]);
    }
  }
  // Not the hash's top word of either transcript, in either byte order.
  let same = 0;
  for (let t = 0; t < 1000; t++) {
    const key = words(8);
    const tm = words(16);
    const tp = words(16);
    const w = alu(key, tm, tp);
    for (const hbe of [0, 1]) if (w === hashTopWord(key, tm, hbe) || w === hashTopWord(key, tp, hbe)) same++;
  }
  expect(same).toBe(0);
}

// ---------------------------------------------------------------------------
describe('tall-fold tile-hash timing stubs (PEARL_TALL_STUB_SPIN, PEARL_TALL_STUB_ALU)', () => {
  test('are off by default, one at a time, and only on the sm_86 and sm_89 paired hash', () => {
    for (const name of SWITCHES) expect(HEADER).toContain(`#ifndef ${name}${NL}#define ${name} 0${NL}#endif`);
    const GUARDS = guards();
    const errorsFor = (flags) => preprocess(GUARDS, flags).errors;
    const value = (name, flags) => evaluate(name, preprocess(GUARDS, flags).defs);
    expect([value('PEARL_TALL_STUB_SPIN', {}), value('PEARL_TALL_STUB_ALU', {})]).toEqual([0, 0]);
    const ARCH = '__CUDA_ARCH__';
    // The race's arms build, on the host pass and on both cp.async cards' device passes.
    const arms = [{}, { PEARL_TALL_STUB_SPIN: 3000 }, { PEARL_TALL_STUB_SPIN: 1700 },
      { PEARL_TALL_STUB_SPIN: 1 }, { PEARL_TALL_STUB_ALU: 1 }, { PEARL_TALL_STUB_SPIN: 0, PEARL_TALL_STUB_ALU: 0 },
      { PEARL_TALL_STUB_ALU: 2 }, { PEARL_TALL_STUB_ALU: 3 }, { PEARL_TALL_STUB_ALU: 4 },
      { PEARL_TALL_STUB_ALU: 5 }, { PEARL_TALL_STUB_SPIN: 4600 }];
    for (const f of arms) {
      for (const arch of [null, 860, 890]) {
        const flags = arch ? { ...f, [ARCH]: arch } : f;
        expect([flags, errorsFor(flags)]).toEqual([flags, []]);
      }
    }
    // With both off, every other card's device pass and the ablation build as before.
    for (const arch of [750, 800, 900, 1000, 1200]) expect(errorsFor({ [ARCH]: arch })).toEqual([]);
    expect(errorsFor({ PEARL_ABLATE_TRANSCRIPT_HASH: 1, PEARL_TALL_HASH_PAIRS: 0 })).toEqual([]);
    // Each refused with an error that names a switch, and each fine without the stub.
    const refused = [
      [{ PEARL_TALL_STUB_SPIN: -1 }, {}],
      [{ PEARL_TALL_STUB_SPIN: 100001 }, {}],
      [{ PEARL_TALL_STUB_ALU: -1 }, {}],
      [{ PEARL_TALL_STUB_ALU: 6 }, {}],
      [{ PEARL_TALL_STUB_SPIN: 3000, PEARL_TALL_STUB_ALU: 1 }, { PEARL_TALL_STUB_SPIN: 3000 }],
      ...[2, 3, 4, 5].flatMap((alu) => [
        [{ PEARL_TALL_STUB_SPIN: 1, PEARL_TALL_STUB_ALU: alu }, { PEARL_TALL_STUB_SPIN: 1 }],
        [{ PEARL_TALL_STUB_ALU: alu, PEARL_ABLATE_TRANSCRIPT_HASH: 1 }, { PEARL_ABLATE_TRANSCRIPT_HASH: 1 }],
        [{ PEARL_TALL_STUB_ALU: alu, PEARL_TALL_HASH_PAIRS: 0 }, { PEARL_TALL_HASH_PAIRS: 0 }],
      ]),
      [{ PEARL_TALL_STUB_SPIN: 1700, PEARL_ABLATE_TRANSCRIPT_HASH: 1 }, { PEARL_ABLATE_TRANSCRIPT_HASH: 1 }],
      [{ PEARL_TALL_STUB_ALU: 1, PEARL_ABLATE_TRANSCRIPT_HASH: 1 }, { PEARL_ABLATE_TRANSCRIPT_HASH: 1 }],
      [{ PEARL_TALL_STUB_SPIN: 3000, PEARL_TALL_HASH_PAIRS: 0 }, { PEARL_TALL_HASH_PAIRS: 0 }],
      [{ PEARL_TALL_STUB_ALU: 1, PEARL_TALL_HASH_PAIRS: 0 }, { PEARL_TALL_HASH_PAIRS: 0 }],
      ...[750, 800, 900, 1000, 1200].flatMap((arch) => [
        [{ PEARL_TALL_STUB_SPIN: 3000, [ARCH]: arch }, { [ARCH]: arch }],
        ...[1, 2, 3, 4, 5].map((alu) => [{ PEARL_TALL_STUB_ALU: alu, [ARCH]: arch }, { [ARCH]: arch }]),
      ]),
    ];
    for (const [f, without] of refused) {
      const errors = errorsFor(f);
      expect([f, errors.length > 0 && errors.every((e) => SWITCHES.some((s) => e.includes(s)))]).toEqual([f, true]);
      expect([without, errorsFor(without)]).toEqual([without, []]);
    }
  });

  test('the fold puts the stand-in where the hash was, and a hit still needs the full hash', () => {
    // On the paired hash: the two compressions go unread under a stub, so ptxas drops
    // them, and the stand-in's words go to the same hit test.
    // The spin's clock is read at the hand-off, before the tile's coordinates (about
    // 150-300 cycles), as the phase and the partner's window are counted: a spin counted from
    // after them would hold the hasher that much past the window at 1700.
    inOrder(FOLD_CODE,
      lines(
        '#if PEARL_TALL_HASH_PAIRS_ON',
        '    if (__any_sync(0xffffffffu, wr == 0u)) {',
        '#if PEARL_TALL_STUB_SPIN',
        '      const long long stub_t0 = clock64();',
        '#endif',
        '#if PEARL_TALL_HASHCYC',
        '      pearl_tall_hcyc_mark(sTr - 112u + 8u * wc);',
        '#endif',
        '#else',
        '    if (wr == 0u) {',
        '#endif'),
      '      tile_coords(hv, hrbg, hcbg);',
      '      const uint32_t h1 = pearl_hp_msw(test.key, tm, test.hash_big_endian);',
      lines(
        '      const uint32_t h2 = pearl_hp_msw_pair(test.key, tp, hh, test.hash_big_endian);',
        '#if PEARL_TALL_STUB_SPIN',
        '      pearl_tall_stub_spin(stub_t0);',
        '      const uint32_t w1 = ~0u, w2 = ~0u;',
        '#elif PEARL_TALL_STUB_ALU',
        '      const uint32_t w1 = pearl_tall_stub_alu(test.key, tm, tp, (uint32_t)test.hash_big_endian | 1u), w2 = ~0u;',
        '#else',
        '      const uint32_t w1 = h1, w2 = h2;',
        '#endif',
        '      auto record = [&](uint32_t Lc) {'),
      lines(
        '      if (wr == 0u) {',
        '#ifndef PEARL_ABLATE_TRANSCRIPT_HASH',
        '        if (w1 <= test.target_w[0]) record(lane);',
        '        if (hh == (test.hash_big_endian ? 0u : 1u) && w2 <= test.target_w[0]) record(Lp);',
        '#endif',
        '      }',
        '#if PEARL_TALL_HASHCYC',
        '      pearl_tall_hcyc_mark(sTr - 80u + 8u * wc);',
        '#endif',
        '      __syncwarp();'));
    // The hash's words feed nothing but the default's hit test, and the hit test reads
    // only w1 and w2.
    for (const h of ['h1', 'h2']) expect(FOLD_CODE.match(new RegExp(`\\b${h}\\b`, 'g'))).toHaveLength(2);
    for (const w of ['w1', 'w2']) expect(FOLD_CODE.match(new RegExp(`\\b${w}\\b`, 'g'))).toHaveLength(4);
    // Each stand-in once, and only in the paired hash.
    expect(FOLD_CODE.split('pearl_tall_stub_spin(').length - 1).toBe(1);
    expect(FOLD_CODE.match(/\bstub_t0\b/g)).toHaveLength(2);
    expect(FOLD_CODE.split('pearl_tall_stub_alu(').length - 1).toBe(1);
    // record() hashes the region again in full and tests every word before it reports, so a
    // stand-in's word that passes the top-word test (~0u at a target whose top word is all
    // ones, or a burst that comes out at or under the target's) still reports only a real hit.
    const record = body(FOLD_CODE, 'auto record = [&](uint32_t Lc) {');
    inOrder(record,
      'pearl_transcript_hash_again(test.key, t, h);',
      'if (!pearl_hash_meets_words(h, test.target_w, test.hash_big_endian)) return;',
      'const uint32_t slot = atomicAdd(hits.count, 1u);');
    // The transcripts are still zeroed for the next tile after a stub.
    expect(FOLD_CODE).toContain(lines(
      '      __syncwarp();',
      '#pragma unroll',
      '      for (uint32_t q = 0; q < 4; q++) {',
      '        asm volatile("st.shared.v4.u32 [%0], {%1,%1,%1,%1};" ::"r"(ra + 16u * q), "r"(0u) : "memory");'));
    // The default top words are the hash's: both passes, in their byte order.
    expect(KERNEL_CODE).toContain('  return hbe ? pearl_bswap32(s[0] ^ s[8]) : (s[7] ^ s[15]);');
  });

  test('the stand-ins are as run here, line for line', () => {
    // All in pearl_kernel.cu, under the paired hash's #if, before the fold's two includes:
    // one burst for each STUB_ALU value, each under its own line.
    inOrder(KERNEL_CODE,
      '#if PEARL_TALL_HASH_PAIRS_ON',
      'uint32_t pearl_hp_msw_pair(',
      '#if PEARL_TALL_STUB_SPIN' + NL + SPIN_HEAD,
      '#if PEARL_TALL_STUB_ALU == 1' + NL + ALU_HEAD,
      '#elif PEARL_TALL_STUB_ALU == 2' + NL + ALU_HEAD,
      '#elif PEARL_TALL_STUB_ALU == 3' + NL + ALU_HEAD,
      '#elif PEARL_TALL_STUB_ALU == 4' + NL + ALU_HEAD,
      '#elif PEARL_TALL_STUB_ALU == 5' + NL + ALU_HEAD,
      '#define PEARL_TALL_FN pearl_tile_fold_tall' + NL,
      '#define PEARL_TALL_FN pearl_tile_fold_tall_pinb' + NL);
    expect(KERNEL_CODE.split(ALU_HEAD).length - 1).toBe(5);
    expect(body(KERNEL_CODE, SPIN_HEAD)).toBe(lines(
      '{',
      '  while (clock64() - t0 < (long long)PEARL_TALL_STUB_SPIN) {',
      '  }',
      '}'));
    // Every arm takes the same arguments; only STUB_ALU=3 and 5 read one.
    expect(KERNEL_CODE.split(ALU_HEAD + 'const uint32_t key[8], const uint32_t tm[16],' + NL
      + '                                                        const uint32_t tp[16], uint32_t one) {').length - 1)
      .toBe(5);
    expect(ALU).toBe(lines(
      '{',
      '  (void)one;',
      '  uint32_t a[4], d[4];',
      '#pragma unroll',
      '  for (int c = 0; c < 4; c++) {',
      '    a[c] = key[c] ^ tm[c] ^ tp[c] ^ tm[4 + c] ^ tp[4 + c];',
      '    d[c] = key[4 + c] ^ tm[8 + c] ^ tp[8 + c] ^ tm[12 + c] ^ tp[12 + c];',
      '  }',
      '  const uint32_t mx = tm[0], my = tp[1];',
      '#pragma unroll 1',
      '  for (int r = 0; r < 21; r++) {',
      '#pragma unroll',
      '    for (int c = 0; c < 4; c++) {',
      '      a[c] = a[c] + d[c] + mx;',
      '      d[c] = rotr32(d[c] ^ a[c], 16);',
      '      a[c] = a[c] + d[c];',
      '      d[c] = rotr32(d[c] ^ a[c], 12);',
      '      a[c] = a[c] + d[c] + my;',
      '      d[c] = rotr32(d[c] ^ a[c], 8);',
      '      a[c] = a[c] + d[c];',
      '      d[c] = rotr32(d[c] ^ a[c], 7);',
      '    }',
      '  }',
      '  return a[0] ^ a[1] ^ a[2] ^ a[3] ^ d[0] ^ d[1] ^ d[2] ^ d[3];',
      '}'));
    // =2: the same steps on one chain, so each reads the one before.
    expect(ALU_CHAIN).toBe(lines(
      '{',
      '  (void)one;',
      '  uint32_t a = 0u, d = 0u;',
      '#pragma unroll',
      '  for (int c = 0; c < 4; c++) {',
      '    a ^= key[c] ^ tm[c] ^ tp[c] ^ tm[4 + c] ^ tp[4 + c];',
      '    d ^= key[4 + c] ^ tm[8 + c] ^ tp[8 + c] ^ tm[12 + c] ^ tp[12 + c];',
      '  }',
      '  const uint32_t mx = tm[0], my = tp[1];',
      '#pragma unroll 1',
      '  for (int r = 0; r < 21; r++) {',
      '#pragma unroll',
      '    for (int c = 0; c < 4; c++) {',
      ...G_STEPS.map((l) => '      ' + l.split('[c]').join('')),
      '    }',
      '  }',
      '  return a ^ d;',
      '}'));
    // =3: every add an IMAD by one, the message words in the XORs.
    expect(ALU_IMAD).toBe(lines(
      '{',
      '  uint32_t a[4], d[4];',
      '#pragma unroll',
      '  for (int c = 0; c < 4; c++) {',
      '    a[c] = key[c] ^ tm[c] ^ tp[c] ^ tm[4 + c] ^ tp[4 + c];',
      '    d[c] = key[4 + c] ^ tm[8 + c] ^ tp[8 + c] ^ tm[12 + c] ^ tp[12 + c];',
      '  }',
      '  const uint32_t mx = tm[0], my = tp[1];',
      '#pragma unroll 1',
      '  for (int r = 0; r < 21; r++) {',
      '#pragma unroll',
      '    for (int c = 0; c < 4; c++) {',
      '      a[c] = d[c] * one + a[c];',
      '      d[c] = rotr32(d[c] ^ a[c] ^ mx, 16);',
      '      a[c] = d[c] * one + a[c];',
      '      d[c] = rotr32(d[c] ^ a[c], 12);',
      '      a[c] = d[c] * one + a[c];',
      '      d[c] = rotr32(d[c] ^ a[c] ^ my, 8);',
      '      a[c] = d[c] * one + a[c];',
      '      d[c] = rotr32(d[c] ^ a[c], 7);',
      '    }',
      '  }',
      '  return a[0] ^ a[1] ^ a[2] ^ a[3] ^ d[0] ^ d[1] ^ d[2] ^ d[3];',
      '}'));
    // =4: =1's passes as quarters (each chain's add, then its XOR and rotate), with an IMMA
    // after quarters 3, 5 and 8 of each two-pass turn, and after quarters 2 and 4 of the last.
    const QUARTER = lines(
      '  auto quarter = [&](int q) {',
      '#pragma unroll',
      '    for (int c = 0; c < 4; c++) {',
      '      a[c] = a[c] + d[c] + (q == 0 ? mx : q == 2 ? my : 0u);',
      '      d[c] = rotr32(d[c] ^ a[c], q == 0 ? 16 : q == 1 ? 12 : q == 2 ? 8 : 7);',
      '    }',
      '  };');
    const call = (x) => '    ' + x + ';';
    const LOOP = lines(
      '#pragma unroll 1',
      '  for (int r = 0; r < 10; r++) {',
      ...['quarter(0)', 'quarter(1)', 'quarter(2)', 'mma(0)', 'quarter(3)', 'quarter(0)', 'mma(1)',
        'quarter(1)', 'quarter(2)', 'quarter(3)', 'mma(2)'].map(call),
      '  }',
      ...['quarter(0)', 'quarter(1)', 'mma(0)', 'quarter(2)', 'quarter(3)', 'mma(1)'].map((x) => '  ' + x + ';'));
    expect(ALU_MMA).toBe(lines(
      '{',
      '  (void)one;',
      '  uint32_t a[4], d[4];',
      '#pragma unroll',
      '  for (int c = 0; c < 4; c++) {',
      '    a[c] = key[c] ^ tm[c] ^ tp[c] ^ tm[4 + c] ^ tp[4 + c];',
      '    d[c] = key[4 + c] ^ tm[8 + c] ^ tp[8 + c] ^ tm[12 + c] ^ tp[12 + c];',
      '  }',
      '  const uint32_t mx = tm[0], my = tp[1];',
      '  int32_t e[3][4];',
      '#pragma unroll',
      '  for (int i = 0; i < 3; i++)',
      '#pragma unroll',
      '    for (int j = 0; j < 4; j++) e[i][j] = 0;',
      QUARTER,
      '  auto mma = [&](int i) {',
      '    pearl_mma_m16n8k32(e[i][0], e[i][1], e[i][2], e[i][3], a[0], a[1], a[2], a[3], d[0], d[1]);',
      '  };',
      LOOP,
      '  uint32_t x = a[0] ^ a[1] ^ a[2] ^ a[3] ^ d[0] ^ d[1] ^ d[2] ^ d[3];',
      '#pragma unroll',
      '  for (int i = 0; i < 3; i++)',
      '#pragma unroll',
      '    for (int j = 0; j < 4; j++) x ^= (uint32_t)e[i][j];',
      '  return x;',
      '}'));
    // The run of calls, the loop's turns written out: quarters 0 to 3 in order, so 21 passes as
    // the burst runs, and 32 IMMA, each two or three quarters after the one before, never back to
    // back (none waits for another's tensor pipe), and each accumulator every third.
    const turns = Number(/for \(int r = 0; r < (\d+); r\+\+\)/.exec(LOOP)[1]);
    const calls = (t) => [...t.matchAll(/(quarter|mma)\((\d)\);/g)].map((m) => [m[1], Number(m[2])]);
    const inTurn = calls(LOOP.slice(LOOP.indexOf('{'), LOOP.indexOf(NL + '  }')));
    const after = calls(LOOP.slice(LOOP.indexOf(NL + '  }') + 4));
    const run = [...Array.from({ length: turns }, () => inTurn).flat(), ...after];
    const qs = run.filter(([f]) => f === 'quarter').map(([, q]) => q);
    expect(qs).toEqual(Array.from({ length: 84 }, (_, i) => i % 4));
    const at = [];
    let quarters = 0;
    for (const [f, n] of run) {
      if (f === 'quarter') quarters++;
      else at.push([quarters, n]);
    }
    expect(at).toHaveLength(32);
    const gaps = at.slice(1).map(([q], i) => q - at[i][0]);
    expect([Math.min(...gaps), Math.max(...gaps)]).toEqual([2, 3]);
    expect(at.map(([, n]) => n)).toEqual(Array.from({ length: 32 }, (_, i) => i % 3));
    // A quarter is =1's two steps for q on every chain, in order: the four quarters are its pass.
    const step = (q) => [`a[c] = a[c] + d[c]${q === 0 ? ' + mx' : q === 2 ? ' + my' : ''};`,
      `d[c] = rotr32(d[c] ^ a[c], ${[16, 12, 8, 7][q]});`];
    expect([0, 1, 2, 3].flatMap(step)).toEqual(G_STEPS);
    // =5: four chains of six pairs of IMAD steps a pass, the same 21 passes, no message words.
    expect(ALU_FMA).toBe(lines(
      '{',
      '  uint32_t a[4], d[4];',
      '#pragma unroll',
      '  for (int c = 0; c < 4; c++) {',
      '    a[c] = key[c] ^ tm[c] ^ tp[c] ^ tm[4 + c] ^ tp[4 + c];',
      '    d[c] = key[4 + c] ^ tm[8 + c] ^ tp[8 + c] ^ tm[12 + c] ^ tp[12 + c];',
      '  }',
      '#pragma unroll 1',
      '  for (int r = 0; r < 21; r++) {',
      '#pragma unroll',
      '    for (int s = 0; s < 6; s++) {',
      '#pragma unroll',
      '      for (int c = 0; c < 4; c++) {',
      '        a[c] = d[c] * one + a[c];',
      '        d[c] = a[c] * one + d[c];',
      '      }',
      '    }',
      '  }',
      '  return a[0] ^ a[1] ^ a[2] ^ a[3] ^ d[0] ^ d[1] ^ d[2] ^ d[3];',
      '}'));
    // The burst's steps are =1's, so the passes are the same work.
    for (const l of G_STEPS) expect(ALU).toContain(NL + '      ' + l + NL);
    // pearl_kernel.cu's rotate, which the CPU copy's rotr is.
    expect(KERNEL).toContain(lines('__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) {',
      '  return (x >> n) | (x << (32 - n));', '}'));
  });

  test('each burst\'s word takes every key and transcript word, and is not the hash\'s', () => {
    // STUB_ALU=3's one is the call's (uint32_t)test.hash_big_endian | 1u: 1 in either byte order.
    expect(FOLD_CODE).toContain('pearl_tall_stub_alu(test.key, tm, tp, (uint32_t)test.hash_big_endian | 1u)');
    expect([0, 1].map((hbe) => (hbe | 1) >>> 0)).toEqual([1, 1]);
    const imad = imadFromSource(ALU_IMAD);
    const fma = fmaFromSource(ALU_FMA);
    for (const alu of [aluFromSource(ALU), chainFromSource(ALU_CHAIN), (k, m, p) => imad(k, m, p, 1),
      (k, m, p) => fma(k, m, p, 1)]) checkBurst(alu);
    // With one = 1 the IMAD arm is the burst with its message words moved into the XORs: a
    // word of its own, not =1's; and a one other than 1 changes it, so ptxas cannot drop it.
    const rand = xorshift(0x1a2b3c4d);
    const w = (n) => Array.from({ length: n }, () => rand());
    const [key, tm, tp] = [w(8), w(16), w(16)];
    expect(imad(key, tm, tp, 1)).not.toBe(aluFromSource(ALU)(key, tm, tp));
    expect(imad(key, tm, tp, 3)).not.toBe(imad(key, tm, tp, 1));
    // The IMAD-only arm: a word of its own, and one changes it too. Mod 2 its 126 pairs of steps
    // are the identity (each pair maps a, d to a + d, a: period 3), so any input bit's flip
    // flips that bit of the word, the top bit too, where carries cannot help.
    expect(fma(key, tm, tp, 1)).not.toBe(aluFromSource(ALU)(key, tm, tp));
    expect(fma(key, tm, tp, 3)).not.toBe(fma(key, tm, tp, 1));
    const pairs = Number(/r < (\d+);/.exec(ALU_FMA)[1]) * Number(/s < (\d+);/.exec(ALU_FMA)[1]);
    expect([pairs, pairs % 3]).toEqual([126, 0]);
    const top = fma(key, tm, tp, 1);
    for (let i = 0; i < 40; i++) {
      const [k2, m2, p2] = [key.slice(), tm.slice(), tp.slice()];
      if (i < 8) k2[i] = (k2[i] ^ 0x80000000) >>> 0;
      else if (i < 24) m2[i - 8] = (m2[i - 8] ^ 0x80000000) >>> 0;
      else p2[i - 24] = (p2[i - 24] ^ 0x80000000) >>> 0;
      expect([i, (fma(k2, m2, p2, 1) ^ top) >>> 0]).toEqual([i, 0x80000000]);
    }
  });

  test('the host refuses to mine with a stub, and names it in the fold\'s name', () => {
    const submit = code(body(HOST, 'extern "C" bool pearl_host_submit('));
    // (The same refusal serves the TMA fold's operand skip, PEARL_TALL_AB_SKIP, with its own
    // message: tallAbSkip.test.js.)
    const GUARD = '#if PEARL_TALL_STUB_SPIN || PEARL_TALL_STUB_ALU || PEARL_TALL_AB_SKIP || PEARL_TALL_HASHCYC';
    inOrder(submit,
      'if (!ctx || !ctx->haveJob) return false;',
      GUARD,
      'for (int i = 0; i < PEARL_HASH_BYTES; i++) zero = zero && ctx->target[i] == 0;',
      'if (!zero) {',
      '"tall-fold stub build is timing-only: it runs only at target 0 "',
      'if (ctx->pendCount >= Ctx::kSlots) {',
      'DeviceScope scope(ctx->device);');
    // Nothing is queued or launched before it.
    const guard = submit.indexOf(GUARD);
    expect(submit.slice(0, guard)).not.toMatch(/<<<|cudaLaunch|pendCount\+\+/);
    // Each stub's build (the skip off) refuses with the stub's message, with or without the
    // stamps.
    const STUBS = [{ PEARL_TALL_STUB_SPIN: 3000, PEARL_TALL_STUB_ALU: 0 },
      ...[1, 2, 3, 4, 5].map((alu) => ({ PEARL_TALL_STUB_SPIN: 0, PEARL_TALL_STUB_ALU: alu }))];
    for (const stub of [...STUBS, ...STUBS.map((x) => ({ ...x, [STAMPS]: 1 }))]) {
      const on = resolve(submit, { [STAMPS]: 0, ...stub, PEARL_TALL_AB_SKIP: 0 });
      expect([stub, on.includes('hasher-stamp')]).toEqual([stub, false]);
      expect([stub, on.includes(lines(
        '    if (!zero) {',
        '      if (err && err_len)',
        '        snprintf(err, err_len, "tall-fold stub build is timing-only: it runs only at target 0 "',
        '                 "(hashrate.js, race.cu), never on a job; build without PEARL_TALL_STUB_SPIN "',
        '                 "and PEARL_TALL_STUB_ALU to mine");',
        '      return false;',
        '    }'))]).toEqual([stub, true]);
    }
    // The target it checks is the job's, and every way to search goes through it.
    expect(code(body(HOST, 'extern "C" void pearl_host_set_job_salted('))).toContain(
      'memcpy(ctx->target, target, PEARL_HASH_BYTES);');
    expect(code(body(HOST, 'extern "C" bool pearl_host_search('))).toContain('if (!pearl_host_submit(handle, ');
    // The timing tools set target zero, which a stub runs at.
    expect(RACE).toContain('memset(target, 0x00, sizeof target);');
    expect(BENCH).toContain('memset(target, verify ? 0xFF : 0x00, sizeof target);');
    expect(HASHRATE).toContain('target: 0n');
    // The fold's name: the tag, the spin with its cycles; added to Ampere's name (sm_86)
    // and Ada's cp.async ring (sm_89), not to the TMA ring's or any other fold's.
    // STUB_ALU=1 keeps its tag; the others say which they are; the stamps add theirs.
    expect(HOST).toContain(lines(
      '#if PEARL_TALL_HASHCYC',
      '#define PEARL_TALL_HCYC_TAG ", hasher stamps"',
      '#else',
      '#define PEARL_TALL_HCYC_TAG ""',
      '#endif',
      '#if PEARL_TALL_STUB_SPIN',
      '#define PEARL_TALL_STUB_STR2(x) #x',
      '#define PEARL_TALL_STUB_STR(x) PEARL_TALL_STUB_STR2(x)',
      'static const char kTallStubTag[] = ", STUB_SPIN=" PEARL_TALL_STUB_STR(PEARL_TALL_STUB_SPIN) PEARL_TALL_HCYC_TAG;',
      '#elif PEARL_TALL_STUB_ALU == 1',
      'static const char kTallStubTag[] = ", STUB_ALU" PEARL_TALL_HCYC_TAG;',
      '#elif PEARL_TALL_STUB_ALU == 2',
      'static const char kTallStubTag[] = ", STUB_ALU=2 (one chain)" PEARL_TALL_HCYC_TAG;',
      '#elif PEARL_TALL_STUB_ALU == 3',
      'static const char kTallStubTag[] = ", STUB_ALU=3 (adds as IMAD)" PEARL_TALL_HCYC_TAG;',
      '#elif PEARL_TALL_STUB_ALU == 4',
      'static const char kTallStubTag[] = ", STUB_ALU=4 (with 32 IMMA)" PEARL_TALL_HCYC_TAG;',
      '#elif PEARL_TALL_STUB_ALU == 5',
      'static const char kTallStubTag[] = ", STUB_ALU=5 (IMAD only)" PEARL_TALL_HCYC_TAG;',
      '#elif PEARL_TALL_HASHCYC',
      'static const char kTallStubTag[] = PEARL_TALL_HCYC_TAG;',
      '#endif'));
    const name = code(body(HOST, 'extern "C" const char *pearl_host_fold_name('));
    expect(name.split('kTallStubTag').length - 1).toBe(2);
    inOrder(name,
      'if (ctx->foldAmpere) {',
      lines(
        '             PEARL_TALL_CTR ? ", tiles from a counter" : "", kAblateDramTag);',
        '#if PEARL_TALL_STUB_SPIN || PEARL_TALL_STUB_ALU || PEARL_TALL_HASHCYC',
        '    {',
        '      Ctx *c = const_cast<Ctx *>(ctx);',
        '      const size_t n = strlen(c->foldName);',
        '      snprintf(c->foldName + n, sizeof c->foldName - n, "%s", kTallStubTag);',
        '    }',
        '#endif',
        '    return ctx->foldName;'),
      lines(
        '#if PEARL_TALL_STUB_SPIN || PEARL_TALL_STUB_ALU || PEARL_TALL_HASHCYC',
        '  if (ctx->foldTall && !ctx->foldTma) {',
        '    snprintf(const_cast<Ctx *>(ctx)->foldName, sizeof ctx->foldName,',
        '             "tall 192x256, 8 warps of 96x64, cp.async ring%s%s",',
        '             ctx->foldTiled ? ", per-tile operands" : "", kTallStubTag);',
        '    return ctx->foldName;',
        '  }',
        '#endif',
        '  if (ctx->foldTall)'));
    // The name it replaces, so the two stay the same apart from the tag.
    expect(name).toContain('(ctx->foldTiled ? "tall 192x256, 8 warps of 96x64, cp.async ring, per-tile operands"');
    expect(name).toContain(': "tall 192x256, 8 warps of 96x64, cp.async ring");');
    // compilers.sh builds each arm's host with the arm's flags, so a stub's own host
    // always has the refusal.
    expect(COMPILERS).toContain('nvcc $COMMON $flags "${keep[@]}" -Xptxas -v -c "$K/src/pearl_kernel.cu"');
    expect(COMPILERS).toContain('nvcc $COMMON $flags -c "$K/src/pearl_host.cu" -o "$K/cuda-build/pearl_host.o"');
  });

  test('no build that ships names either switch, or the stamps', () => {
    for (const file of SHIPPED) {
      const text = rawRead(path.join(ROOT, file));
      expect(text.length).toBeGreaterThan(0);
      for (const name of [...SWITCHES, STAMPS]) expect([file, name, text.includes(name)]).toEqual([file, name, false]);
    }
  });
});

// ---------------------------------------------------------------------------
// The hasher's stamps, as bench.cu reads them: the cycles at a quantile of a histogram of
// bins bin cycles wide, the middle of the bin holding the ceil(q n)-th seam (from 1), and
// -1 for an empty one. A JS twin of hcyc_at; the test pins the C it copies.
function hcycAt(h, n, q, bin) {
  if (n === 0) return -1;
  let want = Math.floor(q * n);
  if (want < q * n) want++;
  if (want < 1) want = 1;
  let seen = 0;
  for (let i = 0; i < h.length; i++) {
    seen += h[i];
    if (seen >= want) return (i + 0.5) * bin;
  }
  return (h.length - 0.5) * bin;
}

// The bin a seam of c cycles lands in, as pearl_tall_hcyc_add counts it.
const hcycBin = (c, bin, bins) => Math.min(Math.floor(c / bin), bins - 1);

describe('tall-fold hasher stamps (PEARL_TALL_HASHCYC)', () => {
  const BIN = Number(/#define PEARL_TALL_HCYC_BIN (\d+)u/.exec(HEADER)[1]);
  const BINS = Number(/#define PEARL_TALL_HCYC_BINS (\d+)u/.exec(HEADER)[1]);

  test('is off by default, needs PEARL_TALL_CYC and the paired hash, and only on sm_86 and sm_89', () => {
    expect(HEADER).toContain(`#ifndef ${STAMPS}${NL}#define ${STAMPS} 0${NL}#endif`);
    const GUARDS = guards();
    const errorsFor = (flags) => preprocess(GUARDS, flags).errors;
    expect(evaluate(STAMPS, preprocess(GUARDS, {}).defs)).toBe(0);
    const ARCH = '__CUDA_ARCH__';
    // Every arm of the step-0 run builds with the stamps, on the host pass and both cards.
    const arms = [{}, ...[1, 1200, 1700, 2000, 2400, 3000, 4600].map((n) => ({ PEARL_TALL_STUB_SPIN: n })),
      ...[1, 2, 3, 4, 5].map((alu) => ({ PEARL_TALL_STUB_ALU: alu }))];
    for (const f of arms) {
      for (const arch of [null, 860, 890]) {
        const flags = { ...f, [STAMPS]: 1, PEARL_TALL_CYC: 1, ...(arch ? { [ARCH]: arch } : {}) };
        expect([flags, errorsFor(flags)]).toEqual([flags, []]);
      }
    }
    // The bins: 8 cycles, 2048 of them, so 16,376 cycles before the last bin, past any
    // seam measured (about 3,000); defined only with the stamps.
    expect([BIN, BINS]).toEqual([8, 2048]);
    expect(evaluate('PEARL_TALL_HCYC_BINS', preprocess(GUARDS, {}).defs)).toBe(0);
    // Off, every other card builds as before.
    for (const arch of [750, 800, 900, 1000, 1200]) expect(errorsFor({ [ARCH]: arch })).toEqual([]);
    const refused = [
      [{ [STAMPS]: 2, PEARL_TALL_CYC: 1 }, { PEARL_TALL_CYC: 1 }],
      [{ [STAMPS]: -1, PEARL_TALL_CYC: 1 }, { PEARL_TALL_CYC: 1 }],
      [{ [STAMPS]: 1 }, {}],
      [{ [STAMPS]: 1, PEARL_TALL_CYC: 1, PEARL_TALL_HASH_PAIRS: 0 }, { PEARL_TALL_CYC: 1, PEARL_TALL_HASH_PAIRS: 0 }],
      ...[750, 800, 900, 1000, 1200].map((arch) => [
        { [STAMPS]: 1, PEARL_TALL_CYC: 1, [ARCH]: arch }, { PEARL_TALL_CYC: 1, [ARCH]: arch }]),
    ];
    for (const [f, without] of refused) {
      const errors = errorsFor(f);
      expect([f, errors.length > 0 && errors.every((e) => e.includes(STAMPS))]).toEqual([f, true]);
      expect([without, errorsFor(without)]).toEqual([without, []]);
    }
  });

  test('off, the fold, the kernel file, the host and bench are the shipped code token for token', () => {
    const OFF = { [STAMPS]: 0, PEARL_TALL_STUB_SPIN: 0, PEARL_TALL_STUB_ALU: 0, PEARL_TALL_AB_SKIP: 0 };
    // The fold: three blocks of the stamps' own, and nothing else names them.
    expect(FOLD.match(/^#if PEARL_TALL_HASHCYC$/gm)).toHaveLength(3);
    expect(FOLD.match(/^#if .*PEARL_TALL_HASHCYC/gm)).toHaveLength(3);
    const fold = resolve(FOLD, OFF);
    expect(fold).not.toMatch(/hcyc|\bcv\b/);
    // The kernel file: the histograms and their take only under the switch.
    expect(KERNEL.match(/^#if PEARL_TALL_HASHCYC$/gm)).toHaveLength(2);
    expect(resolve(KERNEL, OFF)).not.toMatch(/hcyc|HCYC/);
    // The host: no refusal, no tag.
    const host = resolve(HOST, OFF);
    expect(host).not.toMatch(/bool zero = true;|hasher-stamp|kTallStubTag/);
    // bench.cu: nothing more to print.
    expect(resolve(BENCH, OFF)).not.toMatch(/hcyc|HCYC/);
    // pearl_config.h: the bins only with the switch.
    expect(resolve(HEADER, { [STAMPS]: 0 })).not.toContain('PEARL_TALL_HCYC_BIN');
  });

  test('every stamp is in the hasher branch, after the hand-off and outside the chunk loop', () => {
    // t0 at the hand-off, t1 after the hit test (it waits for the hash's words), and the
    // release after the zeroing; all inside the vote-guarded hasher branch.
    inOrder(FOLD_CODE,
      '    pearl_bar_sync(1u + wc, 64u);',
      lines(
        '    if (__any_sync(0xffffffffu, wr == 0u)) {',
        '#if PEARL_TALL_STUB_SPIN',
        '      const long long stub_t0 = clock64();',
        '#endif',
        '#if PEARL_TALL_HASHCYC',
        '      pearl_tall_hcyc_mark(sTr - 112u + 8u * wc);',
        '#endif'),
      lines(
        '        if (hh == (test.hash_big_endian ? 0u : 1u) && w2 <= test.target_w[0]) record(Lp);',
        '#endif',
        '      }',
        '#if PEARL_TALL_HASHCYC',
        '      pearl_tall_hcyc_mark(sTr - 80u + 8u * wc);',
        '#endif',
        '      __syncwarp();'),
      lines(
        '          asm volatile("st.shared.v4.u32 [%0], {%1,%1,%1,%1};" ::"r"(rp + 16u * q), "r"(0u) : "memory");',
        '      }',
        '#if PEARL_TALL_HASHCYC',
        '      {',
        '        uint32_t cv = v;',
        '        asm volatile("" : "+r"(cv));',
        '        pearl_tall_hcyc_add(sTr - 112u + 8u * wc, cv + tile_stride < tiles, lane);',
        '      }',
        '#endif',
        '#endif',
        '    }'));
    // Nothing of the stamps before the hand-off, where the chunk loop is.
    const handoff = FOLD_CODE.indexOf('    pearl_bar_sync(1u + wc, 64u);');
    expect(FOLD_CODE.slice(0, handoff)).not.toMatch(/HASHCYC|hcyc/);
    // The slots: each hasher warp's (wc 0 to 3) two words, t0 at sTr - 112 + 8 wc and t1 32
    // bytes on, are bytes 16 to 79 of the 128 behind ring buffer 2's rows (sTr is that
    // pad's end), past its FULL and EMPTY barriers (bytes 0 to 15), clear of PEARL_TALL_CYC's
    // word (buffer 0's bytes 16 to 23) and of the transcripts from sTr on.
    expect(FOLD_CODE).toContain('  const uint32_t sTr = sbase + NST * STRIDE;');
    expect(HEADER).toContain('#define PEARL_TALL_STAGE_STRIDE (PEARL_TALL_STAGE_BYTES + 128u)');
    const slots = [0, 1, 2, 3].flatMap((wc) => [128 - 112 + 8 * wc, 128 - 112 + 8 * wc + 32]);
    expect(Math.min(...slots)).toBe(16);
    expect(Math.max(...slots) + 8).toBe(80);
    expect(new Set(slots).size).toBe(8);
    // A stamp is one asm: the clock and its store, with no register for the compiler to hold.
    expect(body(KERNEL_CODE, '__device__ __forceinline__ void pearl_tall_hcyc_mark(')).toBe(lines(
      '{',
      '  asm volatile("{' + BS + 'n .reg .u64 t;' + BS + 'n mov.u64 t, %%clock64;' + BS + 'n st.shared.u64 [%0], t;'
        + BS + 'n}" ::"r"(slot) : "memory");',
      '}'));
    // The add: the release's clock, both stamps back, then lane 0 counts a seam with a next
    // tile, in both histograms, each clamped to its last bin, by name inside the asm.
    expect(body(KERNEL_CODE, '__device__ __forceinline__ void pearl_tall_hcyc_add(')).toBe(lines(
      '{',
      '  const long long t2 = clock64();',
      '  long long t0, t1;',
      '  asm volatile("ld.shared.u64 %0, [%1];" : "=l"(t0) : "r"(slot) : "memory");',
      '  asm volatile("ld.shared.u64 %0, [%1];" : "=l"(t1) : "r"(slot + 32u) : "memory");',
      '  if (counted && lane == 0u) {',
      '    const unsigned long long h = (unsigned long long)(t1 - t0) / PEARL_TALL_HCYC_BIN;',
      '    const unsigned long long r = (unsigned long long)(t2 - t0) / PEARL_TALL_HCYC_BIN;',
      '    const uint32_t hb = h < PEARL_TALL_HCYC_BINS ? (uint32_t)h : PEARL_TALL_HCYC_BINS - 1u;',
      '    const uint32_t rb = PEARL_TALL_HCYC_BINS + (r < PEARL_TALL_HCYC_BINS ? (uint32_t)r : PEARL_TALL_HCYC_BINS - 1u);',
      '    asm volatile("{' + BS + 'n .reg .u64 p, q;' + BS + 'n mov.u64 p, pearl_tall_hcyc;' + BS + 'n"',
      '                 " mad.wide.u32 q, %0, 4, p;' + BS + 'n red.global.add.u32 [q], 1;' + BS + 'n"',
      '                 " mad.wide.u32 q, %1, 4, p;' + BS + 'n red.global.add.u32 [q], 1;' + BS + 'n}" ::"r"(hb), "r"(rb) : "memory");',
      '  }',
      '}'));
    expect(KERNEL_CODE).toContain('__device__ unsigned int pearl_tall_hcyc[2u * PEARL_TALL_HCYC_BINS];');
    // Its bins, as the twin counts them.
    expect([0, 7, 8, 1235, 16375, 16376, 16383, 16384, 1e6].map((c) => hcycBin(c, BIN, BINS)))
      .toEqual([0, 0, 1, 154, 2046, 2047, 2047, 2047, 2047]);
    // The take reads both and zeroes them, after the device is idle.
    inOrder(body(KERNEL_CODE, 'extern "C" void pearl_tall_hcyc_take('),
      'cudaDeviceSynchronize();',
      'cudaMemcpyFromSymbol(out, pearl_tall_hcyc, sizeof pearl_tall_hcyc);',
      'cudaGetSymbolAddress(&p, pearl_tall_hcyc);',
      'cudaMemset(p, 0, sizeof pearl_tall_hcyc);');
  });

  test('bench prints each histogram\'s median and 90th percentile, and compilers.sh reads them', () => {
    // bench.cu's reader, as the twin above has it.
    expect(body(code(BENCH), 'static double hcyc_at(')).toBe(lines(
      '{',
      '  if (n == 0) return -1.0;',
      '  unsigned long long want = (unsigned long long)(q * (double)n);',
      '  if ((double)want < q * (double)n) want++;',
      '  if (want < 1) want = 1;',
      '  unsigned long long seen = 0;',
      '  for (unsigned i = 0; i < PEARL_TALL_HCYC_BINS; i++) {',
      '    seen += h[i];',
      '    if (seen >= want) return (i + 0.5) * PEARL_TALL_HCYC_BIN;',
      '  }',
      '  return (PEARL_TALL_HCYC_BINS - 0.5) * PEARL_TALL_HCYC_BIN;',
      '}'));
    const at = (h, q) => hcycAt(h, h.reduce((x, y) => x + y, 0), q, BIN);
    const empty = new Array(BINS).fill(0);
    expect(at(empty, 0.5)).toBe(-1);
    // Ten seams in bins 0 to 9: the median is the 5th (bin 4), the 90th percentile the 9th.
    const ten = empty.slice();
    for (let i = 0; i < 10; i++) ten[i] = 1;
    expect([at(ten, 0.5), at(ten, 0.9)]).toEqual([4.5 * BIN, 8.5 * BIN]);
    // One seam: both are its bin.
    const one = empty.slice();
    one[154] = 1;
    expect([at(one, 0.5), at(one, 0.9)]).toEqual([1236, 1236]);
    // A real shape: 900 seams near 1,800 cycles and 100 near 2,600: median 1,804, p90 1,804,
    // and the 901st seam is the first of the long ones.
    const two = empty.slice();
    two[hcycBin(1800, BIN, BINS)] = 900;
    two[hcycBin(2600, BIN, BINS)] = 100;
    expect([at(two, 0.5), at(two, 0.9), at(two, 0.901)]).toEqual([1804, 1804, 2604]);
    // The last bin answers for everything past it.
    const over = empty.slice();
    over[BINS - 1] = 3;
    expect(at(over, 0.9)).toBe((BINS - 0.5) * BIN);
    // Both histograms, after the cycles a tile; the warm-up's dropped first.
    inOrder(BENCH,
      'pearl_tall_cyc_take(cyc);   // drop the warm-up\'s',
      'pearl_tall_hcyc_take(hcyc);   // drop the warm-up\'s',
      'printf("  cycles a tile %.0f over %llu tiles',
      lines('  pearl_tall_hcyc_take(hcyc);', '  {'),
      'hcyc_at(hcyc, nh, 0.5), hcyc_at(hcyc, nh, 0.9), hcyc_at(hr, nr, 0.5), hcyc_at(hr, nr, 0.9)');
    // The printed line, with numbers in, is what compilers.sh's bench stage picks up; an empty
    // histogram's -1 is not.
    const fmt = /printf\(" {2}hasher cycles a seam \(median\/p90\): ([^"]*)\\n"/.exec(BENCH)[1];
    const fill = (vals) => {
      let i = 0;
      return fmt.replace(/%\.0f|%llu|%u/g, () => String(vals[i++]));
    };
    const line = '  hasher cycles a seam (median/p90): ' + fill([1804, 1900, 1844, 1940, 52000, 0, 16376]);
    const grep = /hash [0-9]+\/[0-9]+ release [0-9]+\/[0-9]+/;
    expect(COMPILERS).toContain(`hcyc=$(grep -oE '${grep.source.split('\\/').join('/')}' "$K/bench.r$r" | tail -1)`);
    expect(grep.exec(line)[0]).toBe('hash 1804/1900 release 1844/1940');
    expect(grep.test('  hasher cycles a seam (median/p90): ' + fill([-1, -1, -1, -1, 0, 0, 16376]))).toBe(false);
    expect(COMPILERS).toContain('${cyc:+ cyc $cyc}${hcyc:+ hcyc [$hcyc]}${k:+ k $k} (n=$n)"');
  });

  test('the host refuses to mine with the stamps alone, and names them', () => {
    const submit = code(body(HOST, 'extern "C" bool pearl_host_submit('));
    const on = resolve(submit, { [STAMPS]: 1, PEARL_TALL_STUB_SPIN: 0, PEARL_TALL_STUB_ALU: 0, PEARL_TALL_AB_SKIP: 0 });
    expect(on).toContain(lines(
      '    if (!zero) {',
      '      if (err && err_len)',
      '        snprintf(err, err_len, "tall-fold hasher-stamp build is timing-only: it runs only at "',
      '                 "target 0 (bench, race.cu), never on a job; build without PEARL_TALL_HASHCYC "',
      '                 "to mine");',
      '      return false;',
      '    }'));
    expect(on).not.toContain('tall-fold stub build');
    // bench, the one tool that reads the stamps, runs at target zero.
    expect(BENCH).toContain('memset(target, verify ? 0xFF : 0x00, sizeof target);');
    // The fold's name carries the tag on both cp.async cards.
    const name = resolve(code(body(HOST, 'extern "C" const char *pearl_host_fold_name(')),
      { [STAMPS]: 1, PEARL_TALL_STUB_SPIN: 0, PEARL_TALL_STUB_ALU: 0 });
    expect(name.split('kTallStubTag').length - 1).toBe(2);
  });
});
