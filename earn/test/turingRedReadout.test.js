'use strict';

const fs = require('fs');
const path = require('path');

// Turing's red readout and lean loads: PEARL_BD_RED, PEARL_BD_LEAN, their parts
// PEARL_BD_TADDR, _RUNBASE, _TAIL_LDG and _GATE, and PEARL_BD_HIT_COORDS. LEAN and
// HIT_COORDS are on by default on the default fold; RED and the split hash turn LEAN
// off, and the other folds build with both off. The rest are off by default.
//
// RED xors each lane's share of a region into the transcript word (red.shared.xor)
// instead of gathering the four shares with shuffles, so the words have to start at
// zero. TADDR holds the store's index in a register. RUNBASE reads the chunk loop's
// load offsets from a base held once a pass. TAIL_LDG loads the next tile's k-step 0
// with coherent loads. GATE 1 gates only the even n-block's B word. HIT_COORDS works
// out a region's number only for a hit.
//
// Nothing here can run the device code. A small C preprocessor below picks out the
// code a build with given switches compiles, and the checks run that code's
// expressions on the CPU: the held index names the default's word; xoring every share
// gives the shuffles' word, tile after tile with the zeroing, whatever order the
// threads run in; every k-step loads from the offset it did; a hit gets the same
// region. Each expression read out of the source is pinned to its text.

const SRC = path.join(__dirname, '..', 'native', 'src');
const NL = String.fromCharCode(10);
const BS = String.fromCharCode(92);
// Line endings normalised: a Windows checkout has CRLF.
const read = (name) => fs.readFileSync(path.join(SRC, name), 'utf8')
  .split(String.fromCharCode(13) + NL).join(NL);
const FOLD = read('pearl_fold_bd.cuh');
const HEADER = read('pearl_config.h');
const U32 = 0xffffffffn;

// ---- C expressions ----------------------------------------------------------------

const TOKEN = /\s*(?:(0[xX][0-9a-fA-F]+|[0-9]+)[uUlL]*|([A-Za-z_]\w*(?:\.\w+)*)|(&&|\|\||==|!=|<=|>=|<<|>>|[-+*/%<>!~&|^?:()]))/y;
const PREC = {
  '||': 1, '&&': 2, '|': 3, '^': 4, '&': 5, '==': 6, '!=': 6, '<': 7, '<=': 7, '>': 7, '>=': 7,
  '<<': 8, '>>': 8, '+': 9, '-': 9, '*': 10, '/': 10, '%': 10,
};
const APPLY = {
  '||': (a, b) => (a || b ? 1n : 0n),
  '&&': (a, b) => (a && b ? 1n : 0n),
  '|': (a, b) => a | b,
  '^': (a, b) => a ^ b,
  '&': (a, b) => a & b,
  '==': (a, b) => (a === b ? 1n : 0n),
  '!=': (a, b) => (a !== b ? 1n : 0n),
  '<': (a, b) => (a < b ? 1n : 0n),
  '<=': (a, b) => (a <= b ? 1n : 0n),
  '>': (a, b) => (a > b ? 1n : 0n),
  '>=': (a, b) => (a >= b ? 1n : 0n),
  '<<': (a, b) => a << b,
  '>>': (a, b) => a >> b,
  '+': (a, b) => a + b,
  '-': (a, b) => a - b,
  '*': (a, b) => a * b,
  '/': (a, b) => {
    if (b === 0n) throw new Error('division by zero');
    return a / b;
  },
  '%': (a, b) => {
    if (b === 0n) throw new Error('division by zero');
    return a % b;
  },
};

function tokens(text) {
  const out = [];
  const s = text.trim();
  TOKEN.lastIndex = 0;
  while (TOKEN.lastIndex < s.length) {
    const at = TOKEN.lastIndex;
    const m = TOKEN.exec(s);
    if (!m) throw new Error(`cannot read "${s.slice(at)}" in "${s}"`);
    if (m[1] !== undefined) out.push({ num: BigInt(m[1]) });
    else if (m[2] !== undefined) out.push({ id: m[2] });
    else out.push({ op: m[3] });
  }
  return out;
}

// A C expression as a function of a scope ({ value(name), defined(name) }), in
// BigInt. mask U32 makes every result a uint32_t; none, the preprocessor's integers.
const compiled = new Map();
function compile(text, mask) {
  const key = (mask === undefined ? 'i ' : 'u ') + text;
  if (compiled.has(key)) return compiled.get(key);
  const ts = tokens(text);
  let i = 0;
  const wrap = mask === undefined ? (v) => v : (v) => v & mask;
  const take = (op) => {
    if (!ts[i] || ts[i].op !== op) throw new Error(`expected ${op} in "${text}"`);
    i++;
  };
  function unary() {
    const t = ts[i++];
    if (!t) throw new Error(`"${text}" ends early`);
    if (t.num !== undefined) return () => wrap(t.num);
    if (t.op === '(') {
      const e = ternary();
      take(')');
      return e;
    }
    if (t.op === '!') {
      const e = unary();
      return (s) => (e(s) ? 0n : 1n);
    }
    if (t.op === '~' || t.op === '-') {
      const e = unary();
      return t.op === '~' ? (s) => wrap(~e(s)) : (s) => wrap(-e(s));
    }
    if (t.id === 'defined') {
      const paren = ts[i] && ts[i].op === '(';
      if (paren) i++;
      const name = ts[i++].id;
      if (paren) take(')');
      return (s) => (s.defined(name) ? 1n : 0n);
    }
    if (t.id !== undefined) {
      if (ts[i] && ts[i].op === '(') throw new Error(`call ${t.id}() in "${text}"`);
      return (s) => s.value(t.id);
    }
    throw new Error(`unexpected ${t.op} in "${text}"`);
  }
  function binary(min) {
    let left = unary();
    for (;;) {
      const op = ts[i] && ts[i].op;
      const p = PREC[op];
      if (!p || p < min) return left;
      i++;
      const l = left;
      const r = binary(p + 1);
      const f = APPLY[op];
      left = (s) => wrap(f(l(s), r(s)));
    }
  }
  function ternary() {
    const c = binary(1);
    if (!(ts[i] && ts[i].op === '?')) return c;
    i++;
    const a = ternary();
    take(':');
    const b = ternary();
    return (s) => (c(s) ? a(s) : b(s));
  }
  const e = ternary();
  if (i !== ts.length) throw new Error(`"${text}" goes on after the expression`);
  compiled.set(key, e);
  return e;
}

// #if's names: a macro's value, 0 for any other name.
function ifScope(defs, seen = []) {
  return {
    defined: (n) => defs.has(n),
    value: (n) => {
      const d = defs.get(n);
      if (d === undefined || seen.includes(n)) return 0n;
      if (d.params) throw new Error(`function-like ${n} in #if`);
      return compile(d.body)(ifScope(defs, [...seen, n]));
    },
  };
}

// Code's names: env's, then object-like macros. A name that is neither, or in env but
// not set yet, is an error.
function codeScope(env, defs) {
  return {
    defined: (n) => defs.has(n),
    value: (n) => {
      if (Object.prototype.hasOwnProperty.call(env, n)) {
        if (env[n] === undefined) throw new Error(`${n} read before it is set`);
        return BigInt(env[n]);
      }
      if (n === 'true') return 1n;
      if (n === 'false') return 0n;
      const d = defs.get(n);
      if (d && !d.params) return compile(d.body, U32)(codeScope({}, defs));
      throw new Error(`unknown name ${n}`);
    },
  };
}

// ---- A small C preprocessor ---------------------------------------------------------

// A line without its // comment (one outside a string literal).
function uncomment(line) {
  let inStr = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inStr) {
      if (ch === BS) i++;
      else if (ch === '"') inStr = false;
    } else if (ch === '"') {
      inStr = true;
    } else if (ch === '/' && line[i + 1] === '/') {
      return line.slice(0, i).trimEnd();
    }
  }
  return line.trimEnd();
}

// The code lines a compile with these macros keeps (without comments), its #error
// messages, and the macros it ends with. #pragma lines stay as code.
function preprocess(text, given) {
  const defs = new Map(given);
  const lines = [];
  const errors = [];
  const stack = [];
  const live = () => stack.length === 0 || stack[stack.length - 1].on;
  const raw = text.split(NL);
  for (let n = 0; n < raw.length; n++) {
    let line = raw[n];
    if (/^\s*#/.test(line)) {
      while (line.endsWith(BS) && n + 1 < raw.length) line = line.slice(0, -1) + ' ' + raw[++n];
    }
    const m = /^\s*#\s*(\w+)(.*)$/.exec(line);
    if (!m) {
      const code = uncomment(line);
      if (live() && code.trim() !== '') lines.push(code);
      continue;
    }
    const dir = m[1];
    const rest = uncomment(m[2]).trim();
    const top = stack[stack.length - 1];
    if (dir === 'if' || dir === 'ifdef' || dir === 'ifndef') {
      const outer = live();
      let on = false;
      if (outer && dir === 'if') on = compile(rest)(ifScope(defs)) !== 0n;
      else if (outer) on = defs.has(rest) === (dir === 'ifdef');
      stack.push({ on, taken: on, outer });
    } else if (dir === 'elif') {
      top.on = top.outer && !top.taken && compile(rest)(ifScope(defs)) !== 0n;
      top.taken = top.taken || top.on;
    } else if (dir === 'else') {
      top.on = top.outer && !top.taken;
      top.taken = true;
    } else if (dir === 'endif') {
      stack.pop();
    } else if (!live()) {
      continue;
    } else if (dir === 'define') {
      const name = /^\w+/.exec(rest)[0];
      const after = rest.slice(name.length);
      if (after.startsWith('(')) {
        const close = after.indexOf(')');
        const params = after.slice(1, close).split(',').map((p) => p.trim()).filter(Boolean);
        defs.set(name, { params, body: after.slice(close + 1).trim() });
      } else {
        defs.set(name, { params: null, body: after.trim() });
      }
    } else if (dir === 'undef') {
      defs.delete(rest);
    } else if (dir === 'error') {
      errors.push(rest);
    } else if (dir === 'pragma') {
      lines.push(line.trim());
    } else {
      throw new Error(`#${dir}`);
    }
  }
  if (stack.length) throw new Error('unclosed #if');
  return { lines, errors, defs };
}

// -D flags, PEARL_BD_ ones by their short names: D('RED=1 GATE=1'). A later flag
// wins over an earlier one for the same switch.
function D(spec) {
  const defs = new Map();
  for (const f of spec.split(' ').filter(Boolean)) {
    const [name, v = '1'] = f.split('=');
    defs.set(name.startsWith('PEARL_') ? name : 'PEARL_BD_' + name, { params: null, body: v });
  }
  return defs;
}

// Every switch off on the default fold: the default before LEAN and HIT_COORDS went
// on.
const OFF = 'LEAN=0 HIT_COORDS=0';

// pearl_config.h's Turing switches, PEARL_TURING_WIDE to PEARL_BD_SPLIT_SMEM, which
// read nothing defined above them.
const TURING = HEADER.slice(HEADER.indexOf('#ifndef PEARL_TURING_WIDE' + NL),
  HEADER.indexOf(NL, HEADER.indexOf('#define PEARL_BD_SPLIT_SMEM ')));

function config(spec) {
  const r = preprocess(TURING, D(spec));
  return { errors: r.errors, defs: r.defs, value: (name) => Number(compile(name)(ifScope(r.defs))) };
}

// The fold an sm_75 compile with these flags builds. extra: flags set after the
// header, past its guards.
const folds = new Map();
function fold(spec, extra = '') {
  const key = spec + '|' + extra;
  if (!folds.has(key)) {
    const c = config(spec);
    expect(c.errors).toEqual([]);
    const defs = new Map([...c.defs, ...D(extra)]);
    // Set further down pearl_config.h (and above the slice) for an sm_75 compile.
    defs.set('PEARL_FOLD_THREADS', { params: null, body: 'PEARL_FOLD_TURING_THREADS' });
    defs.set('PEARL_STAGE_REGS', { params: null, body: '1' });
    defs.set('PEARL_JACKPOT_BUCKETS', { params: null, body: /#define PEARL_JACKPOT_BUCKETS (\d+)/.exec(HEADER)[1] });
    const r = preprocess(FOLD, defs);
    const text = r.lines.join(NL);
    folds.set(key, { text, flat: flat(text), errors: r.errors, defs });
  }
  return folds.get(key);
}
function built(spec, extra = '') {
  const f = fold(spec, extra);
  expect(f.errors).toEqual([]);
  return f;
}

// ---- Reading code -----------------------------------------------------------------

const flat = (s) => s.replace(/\s+/g, ' ').trim();
const count = (text, needle) => text.split(needle).length - 1;

// The function or lambda whose head (ending in '{') is in text, to its closing brace.
function block(text, head) {
  const at = text.indexOf(head);
  if (at < 0) throw new Error(`no "${head}"`);
  let depth = 0;
  let inStr = false;
  for (let i = at + head.length - 1; i < text.length; i++) {
    const ch = text[i];
    if (inStr) {
      if (ch === BS) i++;
      else if (ch === '"') inStr = false;
    } else if (ch === '"') {
      inStr = true;
    } else if (ch === '{') {
      depth++;
    } else if (ch === '}' && --depth === 0) {
      return text.slice(at, i + 1);
    }
  }
  throw new Error(`"${head}" does not close`);
}

// A template whose @ each stand for an expression (no ; or brace in it). shape: text
// is all of it; find: it is somewhere in text. Both give the expressions.
function pattern(template, whole) {
  const parts = template.split('@').map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  return new RegExp((whole ? '^' : '') + parts.join('([^;{}]+?)') + (whole ? '$' : ''));
}
function shape(text, template) {
  const m = pattern(template, true).exec(text);
  return m ? m.slice(1) : null;
}
function find(text, template) {
  const m = pattern(template, false).exec(text);
  if (!m) throw new Error(`no "${template}"`);
  return m.slice(1);
}

function xorshift(seed) {
  let x = seed >>> 0;
  return () => {
    x ^= x << 13; x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5; x >>>= 0;
    return x;
  };
}

// The fold's geometry (pinned in the tests): 8 warps of 16 regions, 16 chunks, and a
// thread's names, from pearl_fold_bd's first lines.
const GEOM = { regions_per_warp: 4, warp_regions: 16, TWORDS: 2048 };
const WARPS = 8;
const CHUNKS = 16;
const THREADS = 256;
// Heads of lambdas whose bodies the tests read.
const KSTEP = 'auto kstep = [&](uint32_t ch, uint32_t t, bool ro, bool last, uint32_t dch = 0u) {';
const HASH_TILE_V = 'auto hash_tile = [&](uint32_t vtile) {';
const HASH_TILE_RC = 'auto hash_tile = [&](uint32_t rbg, uint32_t cbg) {';
function ids(t) {
  const warp = t >> 5;
  return { 'threadIdx.x': t, warp, lane: t & 31, wr: warp >> 2, sid: t & 127 };
}

// A uint32_t expression of code, as a function of env (numbers).
function cfun(f, expr) {
  const e = compile(expr, U32);
  return (env) => Number(e(codeScope({ ...GEOM, ...env }, f.defs)));
}

// fn, remembering its value for each env key(env) gives.
function memo(fn, key) {
  const seen = new Map();
  return (env) => {
    const k = key(env);
    if (!seen.has(k)) seen.set(k, fn(env));
    return seen.get(k);
  };
}

// What the readout and the hash do with the transcripts in a build with these flags,
// read out of its code: the readout's store and its index, the persistent loop's
// buffer and held index, the hashers, their reads and zeroes, and the prologue's
// zeroes.
function readoutCode(spec) {
  const f = built(spec);
  const out = { f, expr: {}, fn: {} };
  const set = (name, expr) => {
    out.expr[name] = expr;
    out.fn[name] = cfun(f, expr);
  };
  out.st = flat(block(f.text, 'auto readout_st = [&](uint32_t rl, uint32_t c, uint32_t x) {'));
  out.ht = flat(block(f.text, f.text.includes(HASH_TILE_V) ? HASH_TILE_V : HASH_TILE_RC));
  out.loop = flat(block(f.text, 'for (uint32_t v = blockIdx.x; v < tiles; it++) {'));
  const st = out.st.slice(out.st.indexOf('{'));
  let x;
  if ((x = shape(st, '{ asm volatile("red.shared.xor.b32 [%0], %1;" ::"r"(sT0 + 4u * (@)), "r"(x) : "memory"); }'))) {
    out.store = 'red';
    set('store', x[0]);
  } else if ((x = shape(st, '{ if (((lane >> 2) & 3u) == 0u) sT[@] = x; }'))) {
    out.store = 'taddr';
    set('store', x[0]);
  } else if ((x = shape(st, '{ if (((lane >> 2) & 3u) == 0u) { const uint32_t L = @; sT[@] = x; } }'))) {
    out.store = 'default';
    set('L', x[0]);
    set('store', x[1]);
  } else {
    throw new Error(`readout_st of "${spec}": ${out.st}`);
  }
  if (out.store !== 'default') {
    x = find(f.flat, 'uint32_t tK; { const uint32_t L0 = @; tK = @; }');
    set('L0', x[0]);
    set('tK0', x[1]);
    x = find(out.loop, '{ tbuf = @; tK = @; asm volatile("" : "+r"(tK));');
    set('tbuf', x[0]);
    set('tKtile', x[1]);
  } else {
    set('tbuf', find(out.loop, '{ tbuf = @; uint32_t vgrab = 0u;')[0]);
  }
  x = find(out.ht, 'const bool hasher = @; const uint32_t L = @; if (hasher) {');
  set('hasher', x[0]);
  set('hL', x[1]);
  set('read', find(out.ht, 'for (uint32_t c = 0; c < 16u; c++) tm[c] = sT[@];')[0]);
  const zero = pattern('for (uint32_t c = 0; c < 16u; c++) sT[@] = 0u;', false).exec(out.ht);
  if (zero) {
    set('zero', zero[1]);
    out.zeroAt = zero.index;
  }
  const pro = pattern('for (uint32_t i = 0; i < @; i++) sT[@] = 0u;', false).exec(f.flat);
  if (pro) {
    set('proBound', pro[1]);
    set('proIndex', pro[2]);
    out.proAt = pro.index;
  }
  return out;
}

// LOP3.LUT: bit i of the result is bit (a_i b_i c_i) of lut.
function lop3(a, b, c, lut) {
  let r = 0;
  for (let i = 0; i < 32; i++) {
    const k = (((a >>> i) & 1) << 2) | (((b >>> i) & 1) << 1) | ((c >>> i) & 1);
    r |= ((lut >>> k) & 1) << i;
  }
  return r >>> 0;
}
// PRMT, default mode: byte i of the result is byte n & 7 of {b:a} (selector nibble
// n), or that byte's sign with n & 8.
function prmt(a, b, sel) {
  const bytes = [a & 255, (a >>> 8) & 255, (a >>> 16) & 255, a >>> 24,
    b & 255, (b >>> 8) & 255, (b >>> 16) & 255, b >>> 24];
  let r = 0;
  for (let i = 0; i < 4; i++) {
    const n = (sel >>> (4 * i)) & 15;
    const v = n & 8 ? (bytes[n & 7] & 128 ? 255 : 0) : bytes[n & 7];
    r |= v << (8 * i);
  }
  return r >>> 0;
}

// One block's tiles under PEARL_BD_RED, thread by thread. Between two barriers the
// threads run in a random order, each in its own program order: after the hand-off of
// tile it - 1, a hasher reads its region of that tile's buffer and zeroes it, then
// every thread xors its shares of tile it into the other buffer. Before the loop
// every thread runs the prologue's zeroes, on whatever shared held at launch. Each
// word a hasher reads must be the one the shuffle readout would have stored there,
// from the same folds. mutate breaks one thing, to show the checks would see it.
function simulate(spec, { tiles, seed, mutate = '' }) {
  const red = readoutCode(spec);
  const base = readoutCode('');
  expect(red.store).toBe('red');
  const redStore = memo(red.fn.store, (e) => `${e.tK}.${e.rl}.${e.c}`);
  const baseStore = memo(base.fn.store, (e) => `${e.tbuf}.${e.L}.${e.c}`);
  const baseL = memo(base.fn.L, (e) => `${e.warp}.${e.lane}.${e.rl}`);
  const readAt = memo(red.fn.read, (e) => `${e.tbuf}.${e.L}.${e.c}`);
  const zeroAt = memo(red.fn.zero, (e) => `${e.tbuf}.${e.L}.${e.c}`);
  const rand = xorshift(seed);
  const W = GEOM.TWORDS;
  const sT = new Uint32Array(2 * W);
  for (let i = 0; i < sT.length; i++) sT[i] = rand();
  const res = { wrong: 0, read: 0, badXors: 0, badZeroes: 0, dirty: 0, outside: 0 };
  const poke = (a, fn) => {
    if (a < 0 || a >= sT.length) res.outside++;
    else fn(a);
  };
  // Random interleaving of the threads' queues.
  const run = (queues) => {
    const live = queues.map((q, i) => i).filter((i) => queues[i].length);
    const pos = new Array(queues.length).fill(0);
    while (live.length) {
      const k = rand() % live.length;
      const i = live[k];
      queues[i][pos[i]++]();
      if (pos[i] === queues[i].length) {
        live[k] = live[live.length - 1];
        live.pop();
      }
    }
  };
  const bound = red.fn.proBound({}) / (mutate === 'prologue-one-buffer' ? 2 : 1);
  run(Array.from({ length: THREADS }, (_, t) => (mutate === 'no-prologue' ? [] : Array.from(
    { length: bound }, (_, i) => () => poke(red.fn.proIndex({ ...ids(t), i }), (a) => { sT[a] = 0; })))));
  const tK = Array.from({ length: THREADS }, (_, t) => red.fn.tK0({ L0: red.fn.L0({ ...ids(t) }) }));
  const want = [];
  const tbufs = [];
  for (let it = 0; it <= tiles; it++) {
    const xors = new Uint8Array(sT.length);
    const zeroes = new Uint8Array(sT.length);
    const queues = Array.from({ length: THREADS }, () => []);
    if (it >= 1) {
      const pb = tbufs[it - 1];
      for (let t = 0; t < THREADS; t++) {
        if (!red.fn.hasher(ids(t))) continue;
        const L = red.fn.hL(ids(t));
        for (let c = 0; c < 16; c++) {
          const a = readAt({ tbuf: pb, L, c });
          queues[t].push(() => poke(a, () => {
            res.read++;
            if (sT[a] !== want[it - 1].get(a)) res.wrong++;
          }));
        }
        if (mutate === 'no-zero') continue;
        for (let c = 0; c < 16; c++) {
          const a = zeroAt({ tbuf: mutate === 'zero-other-buffer' ? pb ^ W : pb, L, c });
          queues[t].push(() => poke(a, () => {
            sT[a] = 0;
            zeroes[a]++;
          }));
        }
      }
    }
    if (it < tiles) {
      const tbuf = red.fn.tbuf({ it });
      tbufs.push(tbuf);
      // Each lane's fold of region rl at chunk c, and the words the shuffles would
      // give: x ^ s4 ^ s8 ^ s12 (pearl_xor3 is LOP3 0x96), stored by the lanes with
      // ((lane >> 2) & 3) == 0 at the default's index.
      const x = new Uint32Array(THREADS * 2 * CHUNKS).map(() => rand());
      const share = (warp, lane, rl, c) => x[((warp * 32 + lane) * 2 + rl) * CHUNKS + c];
      const words = new Map();
      for (let warp = 0; warp < WARPS; warp++) {
        for (let lane = 0; lane < 32; lane++) {
          if (((lane >> 2) & 3) !== 0) continue;
          for (let rl = 0; rl < 2; rl++) {
            const L = baseL({ warp, lane, rl });
            for (let c = 0; c < CHUNKS; c++) {
              const v = lop3(share(warp, lane, rl, c), share(warp, lane ^ 4, rl, c), share(warp, lane ^ 8, rl, c), 0x96)
                ^ share(warp, lane ^ 12, rl, c);
              words.set(baseStore({ tbuf, L, c }), v >>> 0);
            }
          }
        }
      }
      expect(words.size).toBe(W);
      want.push(words);
      for (let t = 0; t < THREADS; t++) {
        if (!(mutate === 'stale-index' && it > 0)) tK[t] = red.fn.tKtile({ tK: tK[t], tbuf });
        const { warp, lane } = ids(t);
        for (let c = 0; c < CHUNKS; c++) {
          for (let rl = 0; rl < 2; rl++) {
            const a = redStore({ tK: tK[t], rl, c });
            const v = share(warp, lane, rl, c);
            queues[t].push(() => poke(a, () => {
              sT[a] ^= v;
              xors[a]++;
            }));
          }
        }
      }
    }
    run(queues);
    // The barrier. Four lanes on every word of this tile's buffer, none on the other.
    if (it < tiles) {
      for (let i = 0; i < W; i++) {
        if (xors[tbufs[it] + i] !== 4 || xors[(tbufs[it] ^ W) + i] !== 0) res.badXors++;
      }
    }
    // Every word of the hashed buffer zeroed once, and zero, before the tile after
    // next xors into it.
    if (it >= 1) {
      for (let i = 0; i < W; i++) {
        const a = tbufs[it - 1] + i;
        if (zeroes[a] !== 1 || sT[a] !== 0) res.badZeroes++;
      }
    }
  }
  res.dirty = sT.filter((w) => w !== 0).length;
  return res;
}

// tile()'s code in a build with these flags, run on the CPU: every kstep call of one
// tile in order, with its k-step and the offsets its loads read (the build's own ao
// and bo), and the readout stores' chunks. offs: the tile's aoff and boff, and the
// next tile's, which next_srcs sets as aoffn and boffn.
function tileCalls(spec, extra, offs) {
  const f = built(spec, extra);
  const ks = flat(block(f.text, KSTEP));
  const expr = {
    s: find(ks, 'const uint32_t s = @;')[0],
    ao: find(ks, 'const uint32_t ao = @;')[0],
    bo: find(ks, 'const uint32_t bo = @;')[0],
    nxt: find(ks, 'const uint32_t cur = @, nxt = @;')[1],
  };
  const ev = (e, env) => cfun(f, e)(env);
  const RO = find(f.flat, 'constexpr bool RO = @;')[0];
  const env = {
    chunks: CHUNKS, KSTEPS: 4, RO: RO === 'true' ? 1 : 0, aoff: offs.aoff, boff: offs.boff,
    aoffn: undefined, boffn: undefined, arun: undefined, brun: undefined,
  };
  const calls = [];
  const stores = [];
  const lines = block(f.text, 'auto tile = [&](auto next_srcs) {').split(NL).slice(1, -1).map((l) => l.trim());
  function run(part) {
    for (let i = 0; i < part.length; i++) {
      const line = part[i];
      let m;
      if ((m = shape(line, 'for (uint32_t @ = @; @; @++) {'))) {
        expect(m[3]).toBe(m[0]);
        let depth = 1;
        let j = i + 1;
        for (; depth > 0; j++) depth += count(part[j], '{') - count(part[j], '}');
        const inner = part.slice(i + 1, j - 1);
        for (env[m[0]] = ev(m[1], env); ev(m[2], env); env[m[0]] += 1) run(inner);
        delete env[m[0]];
        i = j - 1;
      } else if ((m = shape(line, 'arun = @;') || shape(line, 'brun = @;'))) {
        env[line.slice(0, 4)] = ev(m[0], env);
      } else if (line === 'asm volatile("" : "+r"(arun), "+r"(brun));') {
        // Opaque: the values stay.
      } else if ((m = shape(line, 'constexpr uint32_t first_chunk = @;'))) {
        env.first_chunk = ev(m[0], env);
      } else if (line === 'next_srcs();') {
        env.aoffn = offs.aoffn;
        env.boffn = offs.boffn;
      } else if ((m = shape(line, 'kstep(@);'))) {
        const [ch, t, ro, last, dch = 0] = m[0].split(',').map((a) => ev(a, env));
        const e = { ...env, ch, t, ro, last, dch };
        e.s = ev(expr.s, e);
        calls.push({ ch, t, ro, last, dch, s: e.s, ao: ev(expr.ao, e), bo: ev(expr.bo, e), nxt: ev(expr.nxt, e) });
      } else if ((m = shape(line, 'for (uint32_t rl = 0; rl < RPL; rl++) readout_st(rl, @, rx[rl]);')
        || shape(line, 'for (uint32_t rl = 0; rl < RPL; rl++) readout(rl, @);'))) {
        for (let rl = 0; rl < 2; rl++) stores.push(`${rl}.${ev(m[0], env)}`);
      } else if (/\b(arun|brun|aoff|boff|kstep|readout_st|readout|next_srcs|chunk)\b/.test(line)
        && !line.startsWith('for (uint32_t rl = 0; rl < RPL; rl++) rx[rl] = readout_gather(rxs[rl]);')) {
        throw new Error(`tile() line not understood: ${line}`);
      }
    }
  }
  run(lines);
  return { calls, stores, expr, sites: count(lines.join(NL), 'kstep(') };
}

describe('Turing red readout and lean loads (PEARL_BD_RED, PEARL_BD_LEAN and their parts)', () => {
  test('LEAN and HIT_COORDS are on by default on the default fold; LEAN and RED turn on their parts, and the header refuses the rest', () => {
    // The evaluator, on C's precedence.
    const none = ifScope(new Map());
    expect(Number(compile('1 | 4 == 5')(none))).toBe(1);
    expect(Number(compile('2 + 3 * 4 << 1')(none))).toBe(28);
    expect(Number(compile('!0 ? 6 : 7 - 7')(none))).toBe(6);
    expect(Number(compile('-1 & 0xffu', U32)(none))).toBe(255);
    expect(Number(compile('defined(PEARL_X) && PEARL_X > 1')(ifScope(D('PEARL_X=2'))))).toBe(1);
    expect(Number(compile('defined PEARL_Y || PEARL_Y')(ifScope(D('PEARL_X=2'))))).toBe(0);

    expect(TURING.startsWith('#ifndef PEARL_TURING_WIDE' + NL)).toBe(true);
    expect(TURING.endsWith('#define PEARL_BD_SPLIT_SMEM (PEARL_BD_HASH_SPLIT ? 4u * 32u * 16u * 4u : 0u)')).toBe(true);
    const def = (name) => {
      const at = HEADER.indexOf('#ifndef ' + name + NL);
      expect(at).toBeGreaterThanOrEqual(0);
      return HEADER.slice(at).split(NL)[1].trim();
    };
    // LEAN and HIT_COORDS are on by default on the fold they are written for, and LEAN
    // only without RED or the split hash, its alternatives.
    expect(def('PEARL_BD_LEAN')).toBe(
      '#define PEARL_BD_LEAN (PEARL_BD_DEFAULT_FOLD && !PEARL_BD_RED && !PEARL_BD_HASH_SPLIT)');
    expect(def('PEARL_BD_RED')).toBe('#define PEARL_BD_RED 0');
    expect(def('PEARL_BD_HASH_SPLIT')).toBe('#define PEARL_BD_HASH_SPLIT 0');
    expect(def('PEARL_BD_HIT_COORDS')).toBe('#define PEARL_BD_HIT_COORDS PEARL_BD_DEFAULT_FOLD');
    expect(flat(config('').defs.get('PEARL_BD_DEFAULT_FOLD').body)).toBe('(PEARL_BD_A_DIRECT && PEARL_BD_KORDER '
      + '&& PEARL_BD_PERSIST && PEARL_BD_GROUPS == 2 && PEARL_BD_T_BUFS == 2)');
    expect(def('PEARL_BD_RUNBASE')).toBe('#define PEARL_BD_RUNBASE (PEARL_BD_LEAN || PEARL_BD_RED)');
    expect(def('PEARL_BD_TAIL_LDG')).toBe('#define PEARL_BD_TAIL_LDG (PEARL_BD_LEAN || PEARL_BD_RED)');
    expect(def('PEARL_BD_TADDR')).toBe('#define PEARL_BD_TADDR PEARL_BD_RED');
    expect(def('PEARL_BD_GATE')).toBe('#define PEARL_BD_GATE (PEARL_BD_LEAN ? 1 : 2)');

    const SW = ['LEAN', 'RED', 'RUNBASE', 'TAIL_LDG', 'TADDR', 'GATE', 'HIT_COORDS'];
    const of = (spec) => {
      const c = config(spec);
      expect(c.errors).toEqual([]);
      return Object.fromEntries(SW.map((s) => [s, c.value('PEARL_BD_' + s)]));
    };
    // With no flags: LEAN, so its parts, and HIT_COORDS.
    expect(of('')).toEqual({ LEAN: 1, RED: 0, RUNBASE: 1, TAIL_LDG: 1, TADDR: 0, GATE: 1, HIT_COORDS: 1 });
    expect(of('LEAN=1 HIT_COORDS=1')).toEqual(of(''));
    // The fold they are written for is the default one.
    const c0 = config('');
    expect(['A_DIRECT', 'KORDER', 'PERSIST', 'T_BUFS', 'GROUPS', 'A_REALIGN'].map((s) => c0.value('PEARL_BD_' + s)))
      .toEqual([1, 1, 1, 2, 2, 0]);
    expect(c0.value('PEARL_BD_DEFAULT_FOLD')).toBe(1);
    // OFF turns both off, and with them every part.
    expect(of(OFF)).toEqual({ LEAN: 0, RED: 0, RUNBASE: 0, TAIL_LDG: 0, TADDR: 0, GATE: 2, HIT_COORDS: 0 });
    expect(of('LEAN=1 HIT_COORDS=0')).toEqual({ LEAN: 1, RED: 0, RUNBASE: 1, TAIL_LDG: 1, TADDR: 0, GATE: 1, HIT_COORDS: 0 });
    // RED alone turns LEAN off and keeps HIT_COORDS. So does the split hash, which takes
    // the load parts as flags.
    expect(of('RED=1')).toEqual({ LEAN: 0, RED: 1, RUNBASE: 1, TAIL_LDG: 1, TADDR: 1, GATE: 2, HIT_COORDS: 1 });
    expect(of('RED=1 HIT_COORDS=0')).toEqual({ LEAN: 0, RED: 1, RUNBASE: 1, TAIL_LDG: 1, TADDR: 1, GATE: 2, HIT_COORDS: 0 });
    expect(of('HASH_SPLIT=4 RUNBASE=1 TAIL_LDG=1'))
      .toEqual({ LEAN: 0, RED: 0, RUNBASE: 1, TAIL_LDG: 1, TADDR: 0, GATE: 2, HIT_COORDS: 1 });
    // On any other fold, no flags is every switch off.
    for (const other of ['PERSIST=0', 'A_REALIGN=1', 'KORDER=0', 'A_DIRECT=0', 'GROUPS=1']) {
      expect([other, config(other).value('PEARL_BD_DEFAULT_FOLD'), of(other)]).toEqual([other, 0, of(OFF)]);
    }
    // A part given on the command line wins over the one LEAN or RED would set.
    expect(of('RED=1 GATE=1').GATE).toBe(1);
    expect(of('LEAN=1 GATE=2').GATE).toBe(2);
    expect(of('RED=1 RUNBASE=0 TAIL_LDG=0')).toMatchObject({ RUNBASE: 0, TAIL_LDG: 0, TADDR: 1 });
    // LEAN and RED with HIT_COORDS or without, and each part alone or on top of the
    // default, build.
    const PARTS = ['RUNBASE=1', 'TAIL_LDG=1', 'TADDR=1', 'GATE=1', 'HIT_COORDS=1', 'RUNBASE=1 TAIL_LDG=1',
      'TADDR=1 RUNBASE=1 TAIL_LDG=1'];
    for (const spec of ['LEAN=1 HIT_COORDS=1', 'RED=1 HIT_COORDS=1', 'LEAN=1 HIT_COORDS=0', 'RED=1 HIT_COORDS=0',
      ...PARTS.map((p) => `${OFF} ${p}`), ...PARTS]) {
      expect([spec, config(spec).errors]).toEqual([spec, []]);
    }

    const refused = (spec, msg) => {
      const errors = config(spec).errors;
      expect([spec, errors.length > 0]).toEqual([spec, true]);
      expect(errors.join(NL)).toContain(msg);
    };
    for (const s of ['LEAN', 'RED', 'HIT_COORDS']) {
      refused(`${s}=2`, 'PEARL_BD_LEAN, PEARL_BD_RED and PEARL_BD_HIT_COORDS are 0 or 1');
    }
    for (const s of ['RUNBASE', 'TAIL_LDG', 'TADDR']) {
      refused(`${s}=2`, 'PEARL_BD_RUNBASE, PEARL_BD_TAIL_LDG and PEARL_BD_TADDR are 0 or 1');
    }
    for (const g of [0, 3]) refused(`GATE=${g}`, 'PEARL_BD_GATE is 1 or 2');
    refused('LEAN=1 RED=1', 'PEARL_BD_LEAN and PEARL_BD_RED are alternatives');
    refused('RED=1 TADDR=0', 'it needs PEARL_BD_TADDR=1');
    // Each, given on any other fold. With no flags those build (above).
    for (const other of ['PERSIST=0', 'A_REALIGN=1', 'KORDER=0', 'A_DIRECT=0', 'GROUPS=1']) {
      expect(config(other).errors).toEqual([]);
      for (const part of ['LEAN=1', 'RED=1', 'RUNBASE=1', 'TAIL_LDG=1', 'TADDR=1', 'GATE=1', 'HIT_COORDS=1']) {
        refused(`${part} ${other}`,
          'written for the default fold: A-direct, k-lo first, persistent, two row groups, two transcript buffers');
      }
    }
    // The fold's own guard: the ablation skips shuffles RED does not have.
    expect(fold('RED=1 PEARL_ABLATE_TRANSCRIPT').errors.join(NL))
      .toContain('PEARL_ABLATE_TRANSCRIPT skips the readout\'s shuffles, which PEARL_BD_RED does not have');
    expect(fold('PEARL_ABLATE_TRANSCRIPT').errors).toEqual([]);
    expect(fold('RED=1').errors).toEqual([]);
  });

  test('with every switch off, no fold builds any of the new code', () => {
    // OFF on the default fold, and no flags on the others. dch, the ldg wrapper and
    // region_of are shared; the rest stays under #if.
    for (const spec of [OFF, 'PERSIST=0', 'A_REALIGN=1', 'KORDER=0', 'A_DIRECT=0', 'GROUPS=1']) {
      const f = built(spec);
      for (const re of [/\btK\b/, /\barun\b/, /\bbrun\b/, /pearl_bd_ldg_c/, /red\.shared/, /\bsT0\b/, /\bvtile\b/,
        /sT\[[^\]]*\] = 0u;/]) {
        expect([spec, re.test(f.text)]).toEqual([spec, false]);
      }
      expect([spec, count(f.text, '__shfl_xor_sync(')]).toEqual([spec, spec === 'A_DIRECT=0' ? 3 : 6]);
    }
    const f = built(OFF);
    for (const line of [
      'const uint32_t ao = last ? aoffn : aoff + (s + 1u) * 512u;',
      'const uint32_t bo = last ? boffn : boff + (s + 1u) * 512u;',
      'b0[rl] = pearl_bd_after(b0[rl], rxs[rl]);',
      'b2[rl] = pearl_bd_after(b2[rl], rxs[rl]);',
      'sT[tbuf + L * 16u + (c ^ ((L >> 1) & 15u))] = x;',
      'hash_tile(rbg, cbg);',
      'const uint32_t region = region_of(rbg, cbg);',
    ]) {
      expect(f.text).toContain(line);
    }
  });

  test('with no flags the default fold is the LEAN + HIT_COORDS build, line for line', () => {
    const lines = (spec) => built(spec).text.split(NL);
    expect(lines('')).toEqual(lines('LEAN=1 HIT_COORDS=1'));
    expect(lines('')).not.toEqual(lines(OFF));
    // RED alone builds RED with HIT_COORDS. The other folds build as with OFF.
    expect(lines('RED=1')).toEqual(lines('RED=1 HIT_COORDS=1'));
    for (const other of ['PERSIST=0', 'A_REALIGN=1', 'KORDER=0', 'A_DIRECT=0', 'GROUPS=1']) {
      expect([other, lines(other)]).toEqual([other, lines(`${other} ${OFF}`)]);
    }
  });

  test('TADDR\'s held index names the word the default readout writes, tile after tile', () => {
    expect(FOLD).toContain('constexpr uint32_t regions_per_warp = 4u, warp_regions = 16u;');
    expect(FOLD).toContain('const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;');
    expect(FOLD).toContain('const uint32_t wr = warp >> 2, wc = warp & 3u;');
    const twords = find(built('').flat, 'constexpr uint32_t TWORDS = @;')[0];
    expect(twords).toBe('(PEARL_FOLD_THREADS / 32u) * warp_regions * PEARL_JACKPOT_BUCKETS');
    expect(cfun(built(''), twords)({})).toBe(GEOM.TWORDS);

    const base = readoutCode('');
    expect(base.store).toBe('default');
    expect(base.expr.L).toBe('warp * warp_regions + (lane & 3u) * regions_per_warp + 2u * rl + ((lane >> 4) & 1u)');
    expect(base.expr.store).toBe('tbuf + L * 16u + (c ^ ((L >> 1) & 15u))');
    expect(base.expr.tbuf).toBe('(it & 1u) * TWORDS');
    for (const spec of [`${OFF} TADDR=1`, 'RED=1']) {
      const m = readoutCode(spec);
      expect(m.store).toBe(spec === 'RED=1' ? 'red' : 'taddr');
      expect(m.expr.store).toBe('tK ^ (33u * rl) ^ c');
      expect(m.expr.L0).toBe('warp * warp_regions + (lane & 3u) * regions_per_warp + ((lane >> 4) & 1u)');
      expect(m.expr.tK0).toBe('L0 * 16u | ((L0 >> 1) & 15u)');
      expect(m.expr.tbuf).toBe('(it & 1u) * TWORDS');
      expect(m.expr.tKtile).toBe('(tK & (TWORDS - 1u)) | tbuf');
      // The tile's buffer goes into the index before the tile's first readout.
      expect(m.loop.indexOf('tK = (tK & (TWORDS - 1u)) | tbuf;')).toBeLessThan(m.loop.indexOf('tile([&]() {'));
      let bad = 0;
      let checked = 0;
      for (let t = 0; t < THREADS; t++) {
        const { warp, lane } = ids(t);
        let tK = m.fn.tK0({ L0: m.fn.L0({ warp, lane }) });
        // TWORDS is above its bits, so the per-tile fold just sets the buffer.
        if (tK >= GEOM.TWORDS) bad++;
        for (let it = 0; it < 6; it++) {
          const tbuf = m.fn.tbuf({ it });
          tK = m.fn.tKtile({ tK, tbuf });
          for (let rl = 0; rl < 2; rl++) {
            const L = base.fn.L({ warp, lane, rl });
            for (let c = 0; c < CHUNKS; c++) {
              if (m.fn.store({ tK, rl, c }) !== base.fn.store({ tbuf, L, c })) bad++;
              checked++;
            }
          }
        }
      }
      expect(bad).toBe(0);
      expect(checked).toBe(THREADS * 6 * 2 * CHUNKS);
    }
    // The default's writers, lanes with ((lane >> 2) & 3) == 0, cover each buffer's
    // words once: so do the held index's, the same words.
    for (const tbuf of [0, GEOM.TWORDS]) {
      const hits = new Uint8Array(2 * GEOM.TWORDS);
      for (let t = 0; t < THREADS; t++) {
        const { warp, lane } = ids(t);
        if (((lane >> 2) & 3) !== 0) continue;
        for (let rl = 0; rl < 2; rl++) {
          for (let c = 0; c < CHUNKS; c++) hits[base.fn.store({ tbuf, L: base.fn.L({ warp, lane, rl }), c })]++;
        }
      }
      expect(hits.slice(tbuf, tbuf + GEOM.TWORDS).every((n) => n === 1)).toBe(true);
      expect(hits.reduce((a, n) => a + n, 0)).toBe(GEOM.TWORDS);
    }
  });

  test('with RED nothing shuffles: each lane xors its own share into the word', () => {
    const gather = (spec) => flat(block(built(spec).text, 'auto readout_gather = [&](uint32_t x) -> uint32_t {'));
    const rx = (spec) => flat(block(built(spec).text, 'auto readout_x = [&](uint32_t rl) -> uint32_t {'));
    const SHUFFLES = 'const uint32_t s4 = __shfl_xor_sync(0xffffffffu, x, 4); '
      + 'const uint32_t s8 = __shfl_xor_sync(0xffffffffu, x, 8); '
      + 'const uint32_t s12 = __shfl_xor_sync(0xffffffffu, x, 12); x = pearl_xor3(x, s4, s8) ^ s12;';
    const FOLDS = 'uint32_t x = 0u; #pragma unroll for (uint32_t np = 0; np < NB / 2u; np++) x = fold_pair(x, rl, np);';
    for (const spec of [OFF, 'LEAN=1', `${OFF} TADDR=1`, `${OFF} RUNBASE=1 TAIL_LDG=1`, 'LEAN=0 HIT_COORDS=1']) {
      expect(gather(spec)).toBe(`auto readout_gather = [&](uint32_t x) -> uint32_t { ${SHUFFLES} return x; }`);
      expect(rx(spec)).toBe(`auto readout_x = [&](uint32_t rl) -> uint32_t { ${FOLDS} ${SHUFFLES} return x; }`);
    }
    for (const spec of ['RED=1 HIT_COORDS=0', 'RED=1 GATE=1', 'RED=1 HIT_COORDS=1', 'RED=1 RUNBASE=0 TAIL_LDG=0']) {
      const f = built(spec);
      expect([spec, count(f.text, '__shfl')]).toEqual([spec, 0]);
      expect(gather(spec)).toBe('auto readout_gather = [&](uint32_t x) -> uint32_t { return x; }');
      expect(rx(spec)).toBe(`auto readout_x = [&](uint32_t rl) -> uint32_t { ${FOLDS} return x; }`);
      // Every lane stores (no lane test), into the word at sT0 + 4 index bytes.
      const m = readoutCode(spec);
      expect(m.store).toBe('red');
      expect(count(f.text, 'red.shared.xor.b32')).toBe(1);
      expect(find(f.flat, 'const uint32_t sT0 = @;')[0]).toBe('(uint32_t)__cvta_generic_to_shared(sT)');
    }
    expect(FOLD).toContain('#if PEARL_BD_RED' + NL
      + '    // No gather: each lane xors its own share into the word (readout_st).' + NL
      + '#elif !defined(PEARL_ABLATE_TRANSCRIPT)');
    expect(FOLD).toContain('auto readout_gather = [&](uint32_t x) -> uint32_t {' + NL
      + '#if !defined(PEARL_ABLATE_TRANSCRIPT) && !PEARL_BD_RED');
    // pearl_xor3 is LOP3 0x96, a ^ b ^ c.
    expect(read('pearl_kernel.cu')).toContain('asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(d) : "r"(a), "r"(b), "r"(c));');
    const rand = xorshift(5);
    for (let i = 0; i < 200; i++) {
      const [a, b, c] = [rand(), rand(), rand()];
      expect(lop3(a, b, c, 0x96)).toBe((a ^ b ^ c) >>> 0);
    }
    // Lanes L, L ^ 4, L ^ 8 and L ^ 12 hold a region's four shares, and RED's index
    // is the same for all four: the word gets the xor of all four.
    const m = readoutCode('RED=1');
    for (let t = 0; t < THREADS; t++) {
      const { warp, lane } = ids(t);
      const tK = m.fn.tK0({ L0: m.fn.L0({ warp, lane }) });
      for (const o of [4, 8, 12]) expect(m.fn.tK0({ L0: m.fn.L0({ warp, lane: lane ^ o }) })).toBe(tK);
    }
  });

  test('xoring every share gives the shuffle readout\'s words, tile after tile, in any thread order', () => {
    const m = readoutCode('RED=1');
    // The hashers (two groups, persistent), their reads, and RED's zeroes after them,
    // of the same words.
    expect(m.expr.hasher).toBe('wr ? sid >= 64u : sid < 64u');
    expect(m.expr.hL).toBe('wr * 64u + (sid & 63u)');
    expect(m.expr.read).toBe('tbuf + L * 16u + (c ^ ((L >> 1) & 15u))');
    expect(m.expr.zero).toBe('tbuf + L * 16u + (c ^ ((L >> 1) & 15u))');
    expect(m.zeroAt).toBeGreaterThan(m.ht.indexOf('tm[c] = sT['));
    expect(FOLD).toContain('const uint32_t sid = threadIdx.x & 127u;');
    // The hashers take each of the 128 regions once.
    const regions = [];
    for (let t = 0; t < THREADS; t++) if (m.fn.hasher(ids(t))) regions.push(m.fn.hL(ids(t)));
    expect(regions.sort((a, b) => a - b)).toEqual(Array.from({ length: 128 }, (_, i) => i));
    // Each tile's readout stores each region's chunks 0-15 once (tile()'s calls).
    const { stores } = tileCalls('RED=1', '', { aoff: 0, boff: 0, aoffn: 0, boffn: 0 });
    expect([...stores].sort()).toEqual(Array.from({ length: 32 }, (_, i) => `${i >> 4}.${i & 15}`).sort());

    for (const [spec, seed] of [['RED=1 HIT_COORDS=0', 3], ['RED=1 GATE=1 HIT_COORDS=1', 6]]) {
      const res = simulate(spec, { tiles: 7, seed });
      expect([spec, res]).toEqual([spec,
        { wrong: 0, read: 7 * 128 * 16, badXors: 0, badZeroes: 0, dirty: 0, outside: 0 }]);
    }
  });

  test('the simulation catches a missing or misplaced zeroing', () => {
    const run = (mutate) => simulate('RED=1', { tiles: 4, seed: 9, mutate });
    // No zeroing: tile 2 xors onto tile 0's words.
    expect(run('no-zero').wrong).toBeGreaterThan(0);
    // The other buffer zeroed: the hashed words stay, and the next tile's are lost.
    const other = run('zero-other-buffer');
    expect(other.wrong).toBeGreaterThan(0);
    expect(other.badZeroes).toBeGreaterThan(0);
    // No prologue, or one buffer of it: tile 0's or tile 1's words xor onto garbage.
    expect(run('no-prologue').wrong).toBeGreaterThan(0);
    expect(run('prologue-one-buffer').wrong).toBeGreaterThan(0);
    // The index never moved to the tile's buffer: tile 1 xors into words being hashed.
    const stale = run('stale-index');
    expect(stale.wrong).toBeGreaterThan(0);
    expect(stale.badXors).toBeGreaterThan(0);
  });

  test('the prologue zeroes both buffers, every word once, before the first tile', () => {
    for (const spec of [OFF, '']) expect([spec, readoutCode(spec).proAt]).toEqual([spec, undefined]);
    for (const spec of ['RED=1', 'RED=1 GATE=1']) {
      const m = readoutCode(spec);
      expect(m.expr.proBound).toBe('PEARL_BD_T_BUFS * TWORDS / PEARL_FOLD_THREADS');
      expect(m.expr.proIndex).toBe('i * PEARL_FOLD_THREADS + threadIdx.x');
      expect(m.fn.proBound({})).toBe(16);
      expect(m.f.text).toContain('static_assert(PEARL_BD_T_BUFS * TWORDS % PEARL_FOLD_THREADS == 0u, "16 words a thread");');
      const hits = new Uint8Array(2 * GEOM.TWORDS + 1);
      for (let t = 0; t < THREADS; t++) {
        for (let i = 0; i < m.fn.proBound({}); i++) hits[Math.min(m.fn.proIndex({ ...ids(t), i }), 2 * GEOM.TWORDS)]++;
      }
      expect(hits.slice(0, 2 * GEOM.TWORDS).every((n) => n === 1)).toBe(true);
      expect(hits[2 * GEOM.TWORDS]).toBe(0);
      // In the persistent prologue, after the early return and before the
      // __syncthreads the first tile's readout follows; nothing between them reads
      // or writes the transcripts or waits.
      const start = m.f.flat.indexOf('if (blockIdx.x >= tiles) return;');
      const sync = m.f.flat.indexOf('__syncthreads(); uint32_t it = 0u; for (uint32_t v = blockIdx.x; v < tiles; it++) {');
      expect(start).toBeGreaterThanOrEqual(0);
      expect(m.proAt).toBeGreaterThan(start);
      expect(sync).toBeGreaterThan(m.proAt);
      const between = m.f.flat.slice(m.proAt, sync);
      for (const word of ['__syncthreads', 'sT[', 'tile(', 'readout', 'return']) {
        expect(count(between, word)).toBe(word === 'sT[' ? 1 : 0);
      }
      expect(m.f.flat.indexOf('tile([&]() {')).toBeGreaterThan(sync);
    }
  });

  test('RUNBASE loads every k-step from the offset it did', () => {
    // Offsets as a_off and b_off give them, 16-byte steps, some crossing 2^31 or 2^32
    // within the tile.
    const M = 2 ** 32;
    const offs = [
      { aoff: 0x10, boff: 0x7ff0, aoffn: 0x40000, boffn: 0x80 },
      { aoff: 2 ** 31 - 3 * 2048 + 0x1f0, boff: 2 ** 31 + 0x1f0, aoffn: 2 ** 31 - 16, boffn: 2 ** 31 },
      { aoff: M - 5 * 2048 + 0x30, boff: M - 16, aoffn: M - 32768, boffn: 0x1230 },
    ];
    const strip = (c) => ({ ch: c.ch, t: c.t, ro: c.ro, last: c.last, s: c.s, ao: c.ao, bo: c.bo, nxt: c.nxt });
    for (const o of offs) {
      // The loads with every switch off.
      const want = tileCalls(OFF, '', o);
      expect(want.expr.ao).toBe('last ? aoffn : aoff + (s + 1u) * 512u');
      expect(want.expr.bo).toBe('last ? boffn : boff + (s + 1u) * 512u');
      expect(want.expr.s).toBe('ch * KSTEPS + t');
      // A tile's 64 k-steps in order; each loads the next one's quads, the last the
      // next tile's k-step 0; the 15 after a chunk carry its readout.
      expect(want.sites).toBe(8);
      expect(want.calls.map((c) => c.s)).toEqual(Array.from({ length: 64 }, (_, i) => i));
      expect(want.calls.filter((c) => c.last).map((c) => c.s)).toEqual([63]);
      expect(want.calls.filter((c) => c.ro).map((c) => c.s)).toEqual(Array.from({ length: 15 }, (_, i) => 4 * i + 4));
      for (const c of want.calls) {
        expect(c.ao).toBe(c.last ? o.aoffn : (o.aoff + (c.s + 1) * 512) % M);
        expect(c.bo).toBe(c.last ? o.boffn : (o.boff + (c.s + 1) * 512) % M);
      }
      for (const spec of [`${OFF} RUNBASE=1`, 'LEAN=1', 'RED=1', 'LEAN=0 RUNBASE=1 TAIL_LDG=1 HIT_COORDS=1',
        'RED=1 GATE=1']) {
        const got = tileCalls(spec, '', o);
        expect(got.expr.ao).toBe('last ? aoffn : arun + (dch * KSTEPS + t + 1u) * 512u');
        expect(got.expr.bo).toBe('last ? boffn : brun + (dch * KSTEPS + t + 1u) * 512u');
        expect(got.sites).toBe(8);
        expect([spec, got.calls.map(strip)]).toEqual([spec, want.calls.map(strip)]);
        expect(got.stores).toEqual(want.stores);
        // dch is 1 exactly for a pass's k-steps in the next chunk.
        expect(got.calls.filter((c) => c.dch).map((c) => c.s % 4)).toEqual(new Array(30).fill(0).map((_, i) => i & 1));
      }
      // A_REALIGN's peeled chunk 0, which the header refuses with these switches,
      // would load right too.
      const peeled = tileCalls('A_REALIGN=1', 'RUNBASE=1', o);
      expect(peeled.expr.ao).toBe('last ? aoffn : arun + (dch * KSTEPS + t + 1u) * 512u');
      expect(peeled.calls.map(strip)).toEqual(tileCalls('A_REALIGN=1', '', o).calls.map(strip));
      expect(peeled.calls.map(strip)).toEqual(want.calls.map(strip));
    }
    // The pass's base, behind an opaque asm, and the loop it is set in.
    expect(FOLD).toContain('      arun = aoff + chunk * (KSTEPS * 512u);' + NL
      + '      brun = boff + chunk * (KSTEPS * 512u);' + NL + '      asm volatile("" : "+r"(arun), "+r"(brun));');
    expect(FOLD).toContain('constexpr uint32_t MB = 4u, NB = 8u, KSTEPS = rank / 32u;');
    expect(FOLD).toContain('static_assert(rank == PEARL_SB_STRIDE && KSTEPS == 4u, "a staged A row is one 128-byte chunk");');
    expect(FOLD).toContain('static_assert(chunks == PEARL_JACKPOT_BUCKETS, "one transcript word a region a chunk");');
  });

  test('TAIL_LDG loads only the next tile\'s k-step 0 coherent, the same bytes', () => {
    const ldg = (spec) => flat(block(built(spec).text, 'auto ldg = [&](const int8_t *p) -> int4 {'));
    expect(ldg(OFF)).toBe('auto ldg = [&](const int8_t *p) -> int4 { return pearl_bd_ldg(p); }');
    for (const spec of [`${OFF} TAIL_LDG=1`, 'LEAN=1', 'RED=1']) {
      expect(ldg(spec)).toBe('auto ldg = [&](const int8_t *p) -> int4 { '
        + 'if (last) return pearl_bd_ldg_c(p); return pearl_bd_ldg(p); }');
      const f = built(spec);
      // load_half's loads, two quads of A and two of B a half, all go through it.
      const lh = flat(block(f.text, 'auto load_half = [&](uint32_t hi) {'));
      expect(lh).toContain('for (uint32_t q = hi; q < 4u; q += 2u) af[nxt][q] = ldg(Aprime + ao + q * FRAG_BLOCK);');
      expect(lh).toContain('for (uint32_t q = hi; q < 4u; q += 2u) bf[nxt][q] = ldg(Bprime + bo + q * FRAG_BLOCK);');
      expect(count(lh, '= ldg(')).toBe(2);
      expect(count(lh, 'pearl_bd_ldg')).toBe(0);
      // pearl_bd_ldg_c is pearl_bd_ldg without .nc.
      const c = flat(block(f.text, '__device__ __forceinline__ int4 pearl_bd_ldg_c(const int8_t *p) {'));
      const nc = flat(block(f.text, '__device__ __forceinline__ int4 pearl_bd_ldg(const int8_t *p) {'));
      expect(c.replace('pearl_bd_ldg_c(', 'pearl_bd_ldg(').replace('"ld.global.v4.s32 ', '"ld.global.nc.v4.s32 '))
        .toBe(nc);
      expect(c).toContain('asm volatile("ld.global.v4.s32 {%0,%1,%2,%3}, [%4];"');
      // Only k-step 63 is last; it loads next_srcs's offsets into buffer 0, the next
      // tile's k-step 0, which the next tile reads from af[0] and bf[0].
      const o = { aoff: 0x10, boff: 0x20, aoffn: 0x123450, boffn: 0x777770 };
      const lasts = tileCalls(spec, '', o).calls.filter((x) => x.last);
      expect(lasts.map(({ ch, t, ao, bo, nxt }) => ({ ch, t, ao, bo, nxt })))
        .toEqual([{ ch: 15, t: 3, ao: o.aoffn, bo: o.boffn, nxt: 0 }]);
      // A' and B' are read only.
      expect(f.text).not.toMatch(/\b[AB]prime\s*\[[^\]]*\]\s*=[^=]/);
    }
    expect(FOLD).toContain('pearl_fold_bd(const int8_t *__restrict__ Aprime,' + NL
      + '                                              const int8_t *__restrict__ Bprime,');
    expect(FOLD).toContain('      aoffn = a_off(rbgn);' + NL + '      boffn = b_off(cbgn);');
    expect(FOLD).toContain('    // The next tile\'s k-step 0 is already in af[0] and bf[0].' + NL
      + '    aoff = aoffn;' + NL + '    boff = boffn;');
    expect(FOLD).toContain('for (uint32_t mb = 0; mb < MB; mb++) af[0][mb] = pearl_bd_ldg(Aprime + aoff + mb * FRAG_BLOCK);');
    expect(FOLD).toContain('for (uint32_t j = 0; j < 4u; j++) bf[0][j] = pearl_bd_ldg(Bprime + boff + j * FRAG_BLOCK);');
  });

  test('GATE 1 drops only the odd n-block\'s gate, and no gate changes a word', () => {
    // pearl_bd_after is PRMT 0x3210 of b and x: b's four bytes, so b unchanged.
    expect(FOLD).toContain('asm("prmt.b32 %0, %1, %2, 0x3210;" : "=r"(r) : "r"(b), "r"(x));');
    const rand = xorshift(7);
    for (let i = 0; i < 1000; i++) {
      const [b, x] = [rand(), rand()];
      expect(prmt(b, x, 0x3210)).toBe(b);
      expect(prmt(b, x, 0x7654)).toBe(x);
    }
    const fp = flat(block(built('').text, 'auto fold_pair = [&](uint32_t x, uint32_t rl, uint32_t np) -> uint32_t {'));
    const reads = [...fp.matchAll(/\*?(p[0-3]) = acc\[([^\]]+)\]\[([^\]]+)\]/g)].map((x) => x.slice(1));
    expect(reads.map((x) => x.join(' '))).toEqual(['p0 2u * rl 2u * np', 'p1 2u * rl + 1u 2u * np',
      'p2 2u * rl 2u * np + 1u', 'p3 2u * rl + 1u 2u * np + 1u']);
    const f0 = built('');
    const readSet = (rl, np) => new Set(reads.map(([, mb, nb]) => `${cfun(f0, mb)({ rl })}.${cfun(f0, nb)({ np })}`));
    expect(FOLD).toContain('constexpr uint32_t RPL = MB / 2u;');
    for (const [spec, gate] of [[OFF, 2], [`${OFF} GATE=1`, 1], ['LEAN=1', 1], ['RED=1', 2], ['RED=1 GATE=1', 1],
      ['LEAN=1 SNAKE=1', 1]]) {
      const ks = flat(block(built(spec).text, KSTEP));
      const klo = ks.slice(ks.indexOf('load_half(0u);'), ks.indexOf('load_half(1u);'));
      const ro = flat(block(klo, 'if (ro) {'));
      const gates = gate === 2
        ? 'b0[rl] = pearl_bd_after(b0[rl], rxs[rl]); b2[rl] = pearl_bd_after(b2[rl], rxs[rl]); }'
        : 'b0[rl] = pearl_bd_after(b0[rl], rxs[rl]); }';
      expect(ro).toBe('if (ro) { #pragma unroll for (uint32_t rl = 0; rl < RPL; rl++) rxs[rl] = fold_pair(rxs[rl], rl, j); '
        + '#pragma unroll for (uint32_t rl = 0; rl < RPL; rl++) { ' + gates + ' }');
      // A PRMT a gate, for 4 pairs and 2 regions: 16 a pass, or 8.
      expect(count(ro, 'pearl_bd_after(') * 4 * 2).toBe(gate * 8);
      // Every k-lo IMMA of pair j, gated or not, overwrites an accumulator the fold
      // of its region and pair has just read (fold_pair(rxs[mb / 2], mb / 2, j)), so
      // it follows that read either way; the gate is only a hint to ptxas.
      const immas = [...klo.matchAll(
        /pearl_bd_imma\(acc\[mb\]\[([^\]]+)\]\[(\d)\], acc\[mb\]\[([^\]]+)\]\[(\d)\], a_word\(mb, 0u, ([01])u\), (b[02])\[mb \/ 2u\]\);/g)];
      expect(immas.length).toBe(4);
      expect(klo.indexOf('if (ro) {')).toBeLessThan(klo.indexOf('pearl_bd_imma('));
      for (const x of immas) {
        expect(x[1]).toBe(x[3]);
        expect(x[1]).toBe(x[6] === 'b0' ? '2u * j' : '2u * j + 1u');
        for (let j = 0; j < 4; j++) {
          for (let mb = 0; mb < 4; mb++) {
            expect(readSet(mb >> 1, j).has(`${mb}.${cfun(f0, x[1])({ j })}`)).toBe(true);
          }
        }
      }
    }
  });

  test('HIT_COORDS numbers a hit\'s region as before, from the same tile, only for a hit', () => {
    // The region's number as shipped before the switch (4303563), inline in hash_tile.
    const before = (L, rbg, cbg, rowsValid) => {
      const ow = Math.floor(L / 16);
      const ocb = Math.floor((L % 16) / 4);
      const oreg = L % 4;
      const orb = rbg * 2 + Math.floor(ow / 4);
      const ocgb = cbg * 4 + (ow % 4);
      return ((ocgb * 4 + ocb) * rowsValid + orb * 4 + oreg) >>> 0;
    };
    // region_of, read out of the source and run line by line.
    const f = built('');
    const ro = flat(block(f.text, 'auto region_of = [&](uint32_t rbg_, uint32_t cbg_) -> uint32_t {'));
    const steps = [...ro.matchAll(/const uint32_t (\w+) = ([^;]+);/g)].map((x) => [x[1], x[2]]);
    expect(steps).toEqual([['ow', 'L / warp_regions'], ['ocb', '(L % warp_regions) / regions_per_warp'],
      ['oreg', 'L % regions_per_warp'], ['orb', 'rbg_ * 2u + ow / 4u'], ['ocgb', 'cbg_ * 4u + ow % 4u']]);
    const ret = find(ro, 'return @; }')[0];
    expect(ret).toBe('(ocgb * 4u + ocb) * rows_valid + orb * regions_per_warp + oreg');
    const regionOf = (L, rbg, cbg, rowsValid) => {
      const env = { L, rbg_: rbg, cbg_: cbg, rows_valid: rowsValid };
      for (const [name, e] of steps) env[name] = cfun(f, e)(env);
      return cfun(f, ret)(env);
    };
    let checked = 0;
    for (const rowsValid of [16, 64, 1000, 4096]) {
      for (let L = 0; L < 128; L++) {
        for (const rbg of [0, 1, 2, 5, rowsValid / 8 - 1]) {
          for (const cbg of [0, 1, 2, 7, 31]) {
            expect(regionOf(L, rbg, cbg, rowsValid)).toBe(before(L, rbg, cbg, rowsValid));
            checked++;
          }
        }
      }
    }
    expect(checked).toBe(4 * 128 * 5 * 5);

    const HIT = 'if (slot >= PEARL_MAX_HITS) return;';
    // With HIT_COORDS (on by default on the default fold, and in the split hash) and
    // without.
    for (const spec of [OFF, 'LEAN=1 HIT_COORDS=0', 'RED=1 HIT_COORDS=0', 'PERSIST=0', 'A_DIRECT=0',
      'LEAN=0 HIT_COORDS=1', '', 'LEAN=1 HIT_COORDS=1', 'RED=1', 'HASH_SPLIT=4 RUNBASE=1 TAIL_LDG=1',
      'HASH_SPLIT=4 RUNBASE=1 TAIL_LDG=1 HIT_COORDS=0']) {
      const g = built(spec);
      const hc = config(spec).value('PEARL_BD_HIT_COORDS') === 1;
      const ht = flat(block(g.text, hc ? HASH_TILE_V : HASH_TILE_RC));
      if (hc) {
        // Worked out after the hit's slot, from the tile hash_tile was handed.
        expect(ht).toContain(`${HIT} uint32_t rbg, cbg; tile_coords(vtile, rbg, cbg); `
          + 'const uint32_t region = region_of(rbg, cbg); hits.index[slot] = region;');
        expect(count(ht, 'region_of(rbg, cbg)')).toBe(1);
        expect(count(ht, 'tile_coords(')).toBe(1);
      } else {
        expect(ht).toContain('const uint32_t region = region_of(rbg, cbg); [&]() {');
        expect(ht).toContain(`${HIT} hits.index[slot] = region;`);
      }
      // Every call passes what the signature takes: the tile itself, or its
      // coordinates from tile_coords of the same tile.
      const calls = [...g.flat.matchAll(/(?<![\w.])hash_tile\(([^;]*)\);/g)].map((x) => x[1]);
      expect(calls.length).toBeGreaterThan(0);
      for (const a of calls) expect([spec, a.split(',').length]).toEqual([spec, hc ? 1 : 2]);
      if (!spec.includes('PERSIST=0') && !spec.includes('A_DIRECT=0') && !spec.includes('HASH_SPLIT')) {
        const loop = flat(block(g.text, 'for (uint32_t v = blockIdx.x; v < tiles; it++) {'));
        expect(loop).toContain(hc ? '__syncthreads(); hash_tile(v); aoff = aoffn;'
          : '__syncthreads(); { uint32_t rbg, cbg; tile_coords(v, rbg, cbg); hash_tile(rbg, cbg); } aoff = aoffn;');
      }
    }
  });

  test('the fold name the bench prints names every switch that is on', () => {
    // pearl_host_fold_name: the default's name, then the switches, so a reading from
    // one of them is not credited to the default fold.
    const HOST = read('pearl_host.cu');
    const fn = HOST.indexOf('extern "C" const char *pearl_host_fold_name(void *handle) {');
    const from = HOST.indexOf('#if PEARL_BD_RUNBASE || PEARL_BD_TAIL_LDG', fn);
    const to = HOST.indexOf('#endif', from);
    expect(fn).toBeGreaterThan(0);
    expect(from).toBeGreaterThan(HOST.indexOf('PEARL_BD_PERSIST ? "persistent" : "a block a tile", ctx->bdBand);', fn));
    const body = HOST.slice(from, to);
    const cond = body.split(NL).slice(0, 2).join(' ').split(BS).join(' ').replace('#if', '').replace(/\s+/g, ' ').trim();
    expect(cond).toBe('PEARL_BD_RUNBASE || PEARL_BD_TAIL_LDG || PEARL_BD_TADDR || PEARL_BD_RED || PEARL_BD_GATE != 2 '
      + '|| PEARL_BD_HIT_COORDS || PEARL_BD_SNAKE || PEARL_BD_HASH_SPLIT');
    const names = [...body.matchAll(/(PEARL_BD_\w+(?: != 2)?) \? "( [\w=]+)" : ""/g)].map((x) => [x[1], x[2]]);
    expect(names).toEqual([['PEARL_BD_RUNBASE', ' RUNBASE'], ['PEARL_BD_TAIL_LDG', ' TAIL_LDG'],
      ['PEARL_BD_GATE != 2', ' GATE=1'], ['PEARL_BD_TADDR', ' TADDR'], ['PEARL_BD_RED', ' RED'],
      ['PEARL_BD_HIT_COORDS', ' HIT_COORDS'], ['PEARL_BD_SNAKE', ' SNAKE']]);
    expect(body).toContain('", with%s%s%s%s%s%s%s",');
    expect(body).toContain('if (PEARL_BD_HASH_SPLIT)' + NL
      + '        snprintf(c->foldName + n, sizeof c->foldName - n, " HASH_SPLIT=%d", PEARL_BD_HASH_SPLIT);');
    // What it appends for a build, from the header's macros.
    const suffix = (spec) => {
      const c = config(spec);
      expect([spec, c.errors]).toEqual([spec, []]);
      if (!c.value(cond)) return '';
      const split = c.value('PEARL_BD_HASH_SPLIT');
      return ', with' + names.map(([e, n]) => (c.value(e) ? n : '')).join('') + (split ? ' HASH_SPLIT=' + split : '');
    };
    // With no flags, LEAN's parts and HIT_COORDS. With every switch off, on the default
    // fold or another, nothing.
    expect(suffix('')).toBe(', with RUNBASE TAIL_LDG GATE=1 HIT_COORDS');
    expect(suffix('LEAN=1 HIT_COORDS=1')).toBe(', with RUNBASE TAIL_LDG GATE=1 HIT_COORDS');
    expect(suffix(OFF)).toBe('');
    for (const other of ['PERSIST=0', 'A_REALIGN=1', 'KORDER=0', 'A_DIRECT=0', 'GROUPS=1']) {
      expect([other, suffix(other)]).toEqual([other, '']);
    }
    expect(suffix('LEAN=1 HIT_COORDS=0')).toBe(', with RUNBASE TAIL_LDG GATE=1');
    expect(suffix('RED=1')).toBe(', with RUNBASE TAIL_LDG TADDR RED HIT_COORDS');
    expect(suffix('RED=1 HIT_COORDS=0')).toBe(', with RUNBASE TAIL_LDG TADDR RED');
    expect(suffix('SNAKE=1')).toBe(', with RUNBASE TAIL_LDG GATE=1 HIT_COORDS SNAKE');
    expect(suffix(`${OFF} SNAKE=1`)).toBe(', with SNAKE');
    expect(suffix('SNAKE=1 RED=1 GATE=1')).toBe(', with RUNBASE TAIL_LDG GATE=1 TADDR RED HIT_COORDS SNAKE');
    expect(suffix('SNAKE=1 RED=1 GATE=1 HIT_COORDS=0')).toBe(', with RUNBASE TAIL_LDG GATE=1 TADDR RED SNAKE');
    expect(suffix('HASH_SPLIT=4 RUNBASE=1 TAIL_LDG=1')).toBe(', with RUNBASE TAIL_LDG HIT_COORDS HASH_SPLIT=4');
    expect(suffix('HASH_SPLIT=4 RUNBASE=1 TAIL_LDG=1 HIT_COORDS=0')).toBe(', with RUNBASE TAIL_LDG HASH_SPLIT=4');
    // Each part alone names just itself.
    for (const s of ['RUNBASE=1', 'TAIL_LDG=1', 'TADDR=1', 'GATE=1', 'HIT_COORDS=1']) {
      expect([s, suffix(`${OFF} ${s}`)])
        .toEqual([s, ', with ' + names.find(([e]) => e.includes(s.split('=')[0]))[1].trim()]);
    }
    // The longest name fits the buffer.
    expect(HOST).toContain('  char foldName[224] = {0};');
    const longest = 'A-direct, no realign, k-lo first 128x256, 8 warps of 64x64, 2 row groups, persistent, '
      + 'band 64, m8n8k16 (Turing), with RUNBASE TAIL_LDG GATE=1 TADDR RED HIT_COORDS SNAKE HASH_SPLIT=13';
    expect(longest.length).toBeLessThan(224);
  });
});
