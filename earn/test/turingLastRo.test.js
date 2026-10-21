'use strict';

const fs = require('fs');
const path = require('path');

// Jest runs a test file in a vm context, where every read of a global goes through the
// context. Bound here once, the simulation below runs about three times faster.
const { Set, Map, WeakMap, Math, Array, Object, Number, String, Symbol } = globalThis;

// Turing's last chunk read out in the next tile (PEARL_BD_LAST_RO, off by default).
// Without it a tile ends with chunk 15's readout after its last IMMA. With it the next
// tile's first k-step folds each B pair's accumulators just before the pair's first
// IMMA, which take C = 0, k-step 1 stores the words into the finished tile's buffer,
// and only then come the hand-off barrier and the hash. The block's first tile runs
// those two k-steps with nothing to read out, before the prologue's barrier; its last
// runs them over its own operands and drops what they add.
//
// Nothing here can run the device code, so this runs the fold's own source on the CPU.
// A small C preprocessor (turingRedReadout.test.js's) picks out the code a build with
// given switches compiles, and a small interpreter runs pearl_fold_bd's body as written,
// statement by statement, for every warp of a block. A warp is one thread of the
// interpreter whose values may differ by lane: lane, sid and threadIdx.x do, and an if
// on such a value runs its body for the lanes it holds, as the readout's stores and
// thread 0's tile counter do. Values are symbolic: a loaded word knows its operand,
// tile, k-step and place; an accumulator knows which tile and k-step it has summed to;
// a fold is the set of accumulator values it xors (lane-relative, so the shuffles move
// them between lanes); a gate (pearl_bd_after) adds the fold it waits on to the word it
// passes on. The eight warps of a block run in a random order between barriers, a turn
// at a time, each in its own program order: a turn is one shared-memory access or an
// eighth of a k-step.
//
// The checks: each IMMA sums its accumulator's next k-step, from its own tile's A and B
// words; every word a store writes is one whole region of one tile's chunk, at that
// chunk's place; at each hand-off the tile's buffer holds all 2048 words of that tile,
// and they are the words a build without the switch has there at that hand-off; each
// hasher reads its own tile's words; each C = 0 IMMA of tile_head waits, through its B
// word, on the fold that read the accumulator it overwrites; and the last tile's
// tile_head reads its own operands, and nothing reads what it adds. Edits to the
// source that break each of these are run too, to show the checks see them.

const SRC = path.join(__dirname, '..', 'native', 'src');
const NL = String.fromCharCode(10);
const BS = String.fromCharCode(92);
// Line endings normalised: a Windows checkout has CRLF.
const read = (name) => fs.readFileSync(path.join(SRC, name), 'utf8')
  .split(String.fromCharCode(13) + NL).join(NL);
const FOLD = read('pearl_fold_bd.cuh');
const HEADER = read('pearl_config.h');
const KERNEL = read('pearl_kernel.cu');

// ---- #if expressions (turingRedReadout.test.js's) -----------------------------------

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

// A preprocessor expression as a function of a scope ({ value(name), defined(name) }).
const compiled = new Map();
function compile(text) {
  if (compiled.has(text)) return compiled.get(text);
  const ts = tokens(text);
  let i = 0;
  const take = (op) => {
    if (!ts[i] || ts[i].op !== op) throw new Error(`expected ${op} in "${text}"`);
    i++;
  };
  function unary() {
    const t = ts[i++];
    if (!t) throw new Error(`"${text}" ends early`);
    if (t.num !== undefined) return () => t.num;
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
      return t.op === '~' ? (s) => ~e(s) : (s) => -e(s);
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
      left = (s) => f(l(s), r(s));
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
  compiled.set(text, e);
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

// ---- A small C preprocessor (turingRedReadout.test.js's) ----------------------------

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

// -D flags, PEARL_BD_ ones by their short names: D('LAST_RO=1 SNAKE=1').
function D(spec) {
  const defs = new Map();
  for (const f of spec.split(' ').filter(Boolean)) {
    const [name, v = '1'] = f.split('=');
    defs.set(name.startsWith('PEARL_') ? name : 'PEARL_BD_' + name, { params: null, body: v });
  }
  return defs;
}

// Every switch off on the default fold: the default before LEAN and HIT_COORDS went on.
const OFF = 'LEAN=0 HIT_COORDS=0';

// pearl_config.h's Turing switches, PEARL_TURING_WIDE to PEARL_BD_SPLIT_SMEM, which read
// nothing defined above them.
const TURING = HEADER.slice(HEADER.indexOf('#ifndef PEARL_TURING_WIDE' + NL),
  HEADER.indexOf(NL, HEADER.indexOf('#define PEARL_BD_SPLIT_SMEM ')));

function config(spec) {
  const r = preprocess(TURING, D(spec));
  return { errors: r.errors, defs: r.defs, value: (name) => Number(compile(name)(ifScope(r.defs))) };
}

// The macros an sm_75 compile sets further down pearl_config.h, or above the slice, that
// the fold's code reads: the fold's k and rank (so 16 chunks), the jackpot buckets.
const headerDefine = (name) => {
  const m = new RegExp(`^#define ${name} (.+)$`, 'm').exec(HEADER);
  if (!m) throw new Error(`no #define ${name}`);
  return m[1].trim();
};
const CODE_MACROS = ['PEARL_FOLD_K', 'PEARL_FOLD_RANK', 'PEARL_FOLD_CHUNKS', 'PEARL_JACKPOT_BUCKETS'];

// The fold an sm_75 compile with these flags builds, from pearl_fold_bd.cuh's text (or an
// edit of it). extra: flags set after the header, past its guards.
const folds = new Map();
function fold(spec, extra = '', text = FOLD) {
  const key = spec + '|' + extra + '|' + (text === FOLD ? '' : text);
  if (!folds.has(key)) {
    const c = config(spec);
    expect(c.errors).toEqual([]);
    const defs = new Map([...c.defs, ...D(extra)]);
    defs.set('PEARL_FOLD_THREADS', { params: null, body: 'PEARL_FOLD_TURING_THREADS' });
    defs.set('PEARL_STAGE_REGS', { params: null, body: '1' });
    for (const name of CODE_MACROS) defs.set(name, { params: null, body: headerDefine(name) });
    const r = preprocess(text, defs);
    const t = r.lines.join(NL);
    folds.set(key, { text: t, flat: flat(t), errors: r.errors, defs });
  }
  return folds.get(key);
}
function built(spec, extra = '', text = FOLD) {
  const f = fold(spec, extra, text);
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
  return text.slice(at, closeOf(text, at + head.length - 1));
}

// From the opening bracket at s[i] to just past its closing one; strings skipped whole.
function closeOf(s, i) {
  let depth = 0;
  for (let k = i; k < s.length; k++) {
    const ch = s[k];
    if (ch === '"') {
      for (k++; k < s.length && s[k] !== '"'; k++) if (s[k] === BS) k++;
    } else if (ch === '(' || ch === '[' || ch === '{') {
      depth++;
    } else if ((ch === ')' || ch === ']' || ch === '}') && --depth === 0) {
      return k + 1;
    }
  }
  throw new Error(`"${s.slice(i, i + 40)}" does not close`);
}

// text split at each sep outside brackets and strings.
function splitTop(text, sep) {
  const out = [];
  let depth = 0;
  let from = 0;
  for (let k = 0; k < text.length; k++) {
    const ch = text[k];
    if (ch === '"') {
      for (k++; k < text.length && text[k] !== '"'; k++) if (text[k] === BS) k++;
    } else if (ch === '(' || ch === '[' || ch === '{') {
      depth++;
    } else if (ch === ')' || ch === ']' || ch === '}') {
      depth--;
    } else if (ch === sep && depth === 0) {
      out.push(text.slice(from, k).trim());
      from = k + 1;
    }
  }
  out.push(text.slice(from).trim());
  return out;
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

// PRMT, default mode: byte i of the result is byte n & 7 of {b:a} (selector nibble n),
// or that byte's sign with n & 8.
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

// ---- Statements -------------------------------------------------------------------

// The statements of a C++ body: { k: 'block', body }, { k: 'for', init, cond, step, body },
// { k: 'if', cond, then, els }, { k: 'return', expr } and { k: 'simple', text } (one
// statement to its ';', flattened). #pragma unroll is dropped.
function parseStatements(src) {
  const s = src.replace(/#pragma unroll/g, ' ');
  let i = 0;
  const ws = () => {
    while (i < s.length && /\s/.test(s[i])) i++;
  };
  const word = (w) => s.startsWith(w, i) && !/\w/.test(s[i + w.length] || '');
  const paren = () => {
    ws();
    if (s[i] !== '(') throw new Error(`expected ( at "${s.slice(i, i + 40)}"`);
    const end = closeOf(s, i);
    const inner = s.slice(i + 1, end - 1);
    i = end;
    return inner;
  };
  const toSemicolon = () => {
    let depth = 0;
    const from = i;
    for (; i < s.length; i++) {
      const ch = s[i];
      if (ch === '"') {
        for (i++; i < s.length && s[i] !== '"'; i++) if (s[i] === BS) i++;
      } else if (ch === '(' || ch === '[' || ch === '{') {
        depth++;
      } else if (ch === ')' || ch === ']' || ch === '}') {
        depth--;
      } else if (ch === ';' && depth === 0) {
        i++;
        return flat(s.slice(from, i - 1));
      }
    }
    throw new Error(`no ; after "${s.slice(from, from + 60)}"`);
  };
  function statement() {
    ws();
    if (s[i] === '{') {
      i++;
      const body = [];
      for (ws(); s[i] !== '}'; ws()) {
        if (i >= s.length) throw new Error('a block does not close');
        body.push(statement());
      }
      i++;
      return { k: 'block', body };
    }
    if (word('for')) {
      i += 3;
      const parts = splitTop(paren(), ';');
      if (parts.length !== 3) throw new Error(`for (${parts.join('; ')})`);
      return { k: 'for', init: flat(parts[0]), cond: flat(parts[1]), step: flat(parts[2]), body: statement() };
    }
    if (word('if')) {
      i += 2;
      const cond = flat(paren());
      const then = statement();
      ws();
      let els = null;
      if (word('else')) {
        i += 4;
        els = statement();
      }
      return { k: 'if', cond, then, els };
    }
    if (word('return')) {
      i += 6;
      return { k: 'return', expr: toSemicolon() };
    }
    return { k: 'simple', text: toSemicolon() };
  }
  const out = [];
  for (ws(); i < s.length; ws()) out.push(statement());
  return out;
}

// ---- Values -----------------------------------------------------------------------

const UNINIT = Object.freeze({ uninit: true });   // a variable before it is set
const GARBAGE = Object.freeze({ garbage: true }); // shared memory before it is written
const SHARED = Symbol('shared');                  // a warp's yield before a shared access or a k-step
const BAR = Symbol('barrier');                    // a warp's yield at __syncthreads

// A value that differs across a warp's 32 lanes; one that is the same everywhere is
// kept as that value.
class LaneVec {
  constructor(a) {
    this.a = a;
  }
}
function lanes(f) {
  const a = new Array(32);
  for (let l = 0; l < 32; l++) a[l] = f(l);
  for (let l = 1; l < 32; l++) if (a[l] !== a[0]) return new LaneVec(a);
  return a[0];
}
const at = (v, l) => (v instanceof LaneVec ? v.a[l] : v);
const describe_ = (v) => (v instanceof Object ? v.constructor.name + (v.key ? ' ' + v.key : '') : String(v));

// The state of one simulation (runFold): interned values, counts, and the warp running.
let RUN = null;
const CTX = { warp: null };

// Equal values are one object: warps that run the same code on the same data then hold
// the same objects, which kstep's memo (below) compares by identity.
function intern(key, make) {
  let v = RUN.vals.get(key);
  if (!v) {
    v = make();
    v.key = key;
    v.id = RUN.ids++;
    RUN.vals.set(key, v);
  }
  return v;
}
const EMPTY = Object.freeze(new Set());
const depsKey = (d) => (d.size ? [...d].sort((a, b) => a - b).join(',') : '');
function union(a, b) {
  if (a === b || !b.size) return a;
  if (!a.size) return b;
  const u = new Set(a);
  for (const x of b) u.add(x);
  return u;
}

// An operand word as loaded: A or B, the tile its offset belongs to, k-step s of that
// tile, quad q (the LDG.128 of block q of the warp's four) and component comp (x y z w);
// deps: the accumulator values whose folds it waits on (pearl_bd_after).
class Word {}
function word(kind, tile, s, q, comp, deps) {
  return intern(`w${kind}${tile}.${s}.${q}.${comp}|${depsKey(deps)}`, () => Object.assign(new Word(),
    { kind, tile, s, q, comp, deps }));
}
class Quad {}
function quad(x, y, z, w) {
  return intern(`q${x.id},${y.id},${z.id},${w.id}`, () => Object.assign(new Quad(), { x, y, z, w }));
}

// An accumulator's value. r = mb * 32 + nb * 4 + i (acc[mb][nb][i]). st: 1 + tile * 128 +
// s * 2 + half once it has summed k-step s's k-lo (half 0) or k-hi (half 1) IMMA of that
// tile, 0 for a wrong sum, -1 before any; zero: set to 0, prev the value it replaced.
class Reg {}
const stCode = (tile, s, half) => 1 + tile * 128 + s * 2 + half;
const stOf = (st) => ({ tile: Math.floor((st - 1) / 128), s: ((st - 1) % 128) >> 1, half: (st - 1) & 1 });
function reg(r, st) {
  const k = (st + 2) * 128 + r;
  let v = RUN.regs.get(k);
  if (!v) {
    v = Object.assign(new Reg(), { r, st, zero: false, id: RUN.ids++ });
    RUN.regs.set(k, v);
  }
  return v;
}
function zreg(r, prev) {
  const k = -(prev * 128 + r) - 1;
  let v = RUN.regs.get(k);
  if (!v) {
    v = Object.assign(new Reg(), { r, st: -1, zero: true, prev, id: RUN.ids++ });
    RUN.regs.set(k, v);
  }
  return v;
}

// A fold: the xor of accumulator values, as a set of terms (value, lane offset): term =
// (max(st, 0) * 128 + r) * 4 + (lane xor) / 4, the lane xor the shuffles moved it by (0,
// 4, 8 or 12). xor is the sets' symmetric difference. deps: the values it read.
class Sym {
  constructor(terms, deps) {
    this.terms = terms;
    this.deps = deps;
  }
}
const EMPTY_SYM = new Sym(EMPTY, EMPTY);
function toSym(v) {
  if (v instanceof Sym) return v;
  if (v === 0) return EMPTY_SYM;
  if (v instanceof Reg) {
    if (v.zero) return EMPTY_SYM;
    if (!RUN.foldedAt.has(v.id)) RUN.foldedAt.set(v.id, CTX.warp.epoch);
    CTX.warp.blk.lastFold = Math.max(CTX.warp.blk.lastFold, CTX.warp.epoch);
    return new Sym(new Set([((v.st > 0 ? v.st : 0) * 128 + v.r) * 4]), new Set([v.id]));
  }
  throw new Error(`xor of ${describe_(v)}`);
}
function xorV(a, b) {
  if (typeof a === 'number' && typeof b === 'number') return (a ^ b) >>> 0;
  if (a instanceof LaneVec || b instanceof LaneVec) return lanes((l) => xorV(at(a, l), at(b, l)));
  const A = toSym(a);
  const B = toSym(b);
  if (B === EMPTY_SYM) return A;
  if (A === EMPTY_SYM) return B;
  const t = new Set(A.terms);
  for (const x of B.terms) {
    if (t.has(x)) t.delete(x);
    else t.add(x);
  }
  return new Sym(t, union(A.deps, B.deps));
}
const depsOf = (x) => {
  if (x instanceof Sym || x instanceof Word) return x.deps;
  if (typeof x === 'number') return EMPTY;
  throw new Error(`a gate on ${describe_(x)}`);
};

// What a stored word is: one whole region of one chunk of one tile, or not. The default
// readout's word is the 64 accumulators of m-blocks 2r and 2r + 1 (region half r) of
// four lanes (lane xors 0, 4, 8, 12); PEARL_BD_ROWPAIR's all 128 of two lanes (0, 4).
// Either way every accumulator at the end of the same chunk's last k-step (k-hi).
const words = new WeakMap();
function wordOf(x) {
  if (!(x instanceof Sym)) return { bad: `a store of ${describe_(x)}` };
  if (words.has(x)) return words.get(x);
  let st = null;
  const regs = new Set();
  const offs = new Set();
  let w = null;
  for (const t of x.terms) {
    const s = Math.floor(t / 512);
    if (st === null) st = s;
    else if (st !== s) w = { bad: 'accumulators of two k-steps' };
    regs.add((t >> 2) & 127);
    offs.add(t & 3);
  }
  if (!w && !st) w = { bad: st === null ? 'nothing' : 'garbage' };
  if (!w && x.terms.size !== regs.size * offs.size) w = { bad: 'not every accumulator in every lane' };
  if (!w) {
    const { tile, s, half } = stOf(st);
    const mbs = new Set([...regs].map((r) => r >> 5));
    if (half !== 1 || s % 4 !== 3) w = { bad: `k-step ${s} half ${half}, not a chunk's end` };
    else if (offs.size === 4 && regs.size === 64 && mbs.size === 2 && [...mbs].every((mb) => mb >> 1 === [...mbs][0] >> 1)) {
      w = { tile, c: (s - 3) / 4, r: [...mbs][0] >> 1 };
    } else if (offs.size === 2 && offs.has(1) && regs.size === 128) {
      w = { tile, c: (s - 3) / 4, r: 'all' };
    } else {
      w = { bad: `${regs.size} accumulators in ${offs.size} lanes` };
    }
  }
  words.set(x, w);
  return w;
}

// ---- Expressions -------------------------------------------------------------------

const CTOK = /\s*(?:(0[xX][0-9a-fA-F]+|\d+)[uUlL]*|([A-Za-z_]\w*)|(&&|\|\||==|!=|<=|>=|<<|>>|[-+*/%<>!~&|^?:()[\],.]))/y;
const TYPES = new Set(['uint32_t', 'int32_t', 'int', 'bool', 'uint64_t', 'void', 'int8_t']);
function parseExpr(text) {
  const ts = [];
  CTOK.lastIndex = 0;
  while (CTOK.lastIndex < text.length) {
    const from = CTOK.lastIndex;
    const m = CTOK.exec(text);
    if (!m) {
      if (/^\s*$/.test(text.slice(from))) break;
      throw new Error(`cannot read "${text.slice(from)}" in "${text}"`);
    }
    if (m[1] !== undefined) ts.push({ num: Number(m[1]) });
    else if (m[2] !== undefined) ts.push({ id: m[2] });
    else ts.push({ op: m[3] });
  }
  let i = 0;
  const peek = (o) => ts[i] && ts[i].op === o;
  const take = (o) => {
    if (!peek(o)) throw new Error(`expected ${o} in "${text}"`);
    i++;
  };
  function primary() {
    const t = ts[i++];
    if (!t) throw new Error(`"${text}" ends early`);
    if (t.num !== undefined) return { t: 'num', v: t.num };
    if (t.id !== undefined) return { t: 'id', n: t.id };
    if (t.op === '(') {
      const n1 = ts[i];
      const n2 = ts[i + 1];
      if (n1 && TYPES.has(n1.id) && n2 && (n2.op === ')' || (n2.op === '*' && ts[i + 2] && ts[i + 2].op === ')'))) {
        i += n2.op === ')' ? 2 : 3;
        return { t: 'cast', a: unary() };
      }
      const e = ternary();
      take(')');
      return e;
    }
    throw new Error(`unexpected ${t.op} in "${text}"`);
  }
  function postfix() {
    let e = primary();
    for (;;) {
      if (peek('[')) {
        i++;
        const ix = ternary();
        take(']');
        e = { t: 'idx', a: e, i: ix };
      } else if (peek('.')) {
        i++;
        e = { t: 'mem', a: e, m: ts[i++].id };
      } else if (peek('(')) {
        i++;
        const args = [];
        while (!peek(')')) {
          args.push(ternary());
          if (peek(',')) i++;
        }
        take(')');
        e = { t: 'call', f: e, args };
      } else {
        return e;
      }
    }
  }
  function unary() {
    if (peek('!') || peek('~') || peek('-')) {
      const op = ts[i++].op;
      return { t: 'un', op, a: unary() };
    }
    return postfix();
  }
  function binary(min) {
    let l = unary();
    for (;;) {
      const op = ts[i] && ts[i].op;
      const p = PREC[op];
      if (!p || p < min) return l;
      i++;
      l = { t: 'bin', op, a: l, b: binary(p + 1) };
    }
  }
  function ternary() {
    const c = binary(1);
    if (!peek('?')) return c;
    i++;
    const a = ternary();
    take(':');
    return { t: 'tern', c, a, b: ternary() };
  }
  const e = ternary();
  if (i !== ts.length) throw new Error(`"${text}" goes on after the expression`);
  return e;
}

function num(v) {
  if (typeof v !== 'number') throw new Error(`arithmetic on ${describe_(v)}`);
  return v;
}
const NUM = {
  '+': (a, b) => (a + b) >>> 0,
  '-': (a, b) => (a - b) >>> 0,
  '*': (a, b) => Math.imul(a, b) >>> 0,
  '/': (a, b) => {
    if (!b) throw new Error('division by zero');
    return Math.floor(a / b);
  },
  '%': (a, b) => {
    if (!b) throw new Error('division by zero');
    return a % b;
  },
  '<<': (a, b) => (b >= 32 ? 0 : (a << b) >>> 0),
  '>>': (a, b) => (b >= 32 ? 0 : a >>> b),
  '&': (a, b) => (a & b) >>> 0,
  '|': (a, b) => (a | b) >>> 0,
  '<': (a, b) => (a < b ? 1 : 0),
  '<=': (a, b) => (a <= b ? 1 : 0),
  '>': (a, b) => (a > b ? 1 : 0),
  '>=': (a, b) => (a >= b ? 1 : 0),
  '==': (a, b) => (a === b ? 1 : 0),
  '!=': (a, b) => (a !== b ? 1 : 0),
};
const lift = (f) => (a, b) => ((a instanceof LaneVec || b instanceof LaneVec)
  ? lanes((l) => f(num(at(a, l)), num(at(b, l)))) : f(num(a), num(b)));
const BIN = Object.fromEntries(Object.entries(NUM).map(([op, f]) => [op, lift(f)]));
BIN['^'] = xorV;
const truthy = (v) => {
  if (v instanceof LaneVec) throw new Error('a branch that differs by lane');
  return num(v) !== 0;
};

// ---- The interpreter ----------------------------------------------------------------

class Lambda {
  constructor(def, closure) {
    this.def = def;
    this.closure = closure;
  }
}
class Builtin {
  constructor(name, fn, { refs = [], gen = false } = {}) {
    Object.assign(this, { name, fn, refs, gen });
  }
}
// A C++ reference (int &b0 = ..., a & parameter): the lvalue it names.
class Ref {
  constructor(lv) {
    this.lv = lv;
  }
}
// One block's shared array: sT (the transcripts) or sNext.
class Shared {
  constructor(name, size, blk) {
    Object.assign(this, { name, arr: new Array(size).fill(GARBAGE), blk });
  }
  read(i) {
    num(i);
    if (i >= this.arr.length) throw new Error(`${this.name}[${i}] is past its end`);
    return this.arr[i];
  }
}

// Names resolve when the code is compiled, as C++'s do: to a slot of the frame of the
// block, for header, lambda call or warp that declares them, some frames up, or else to
// a builtin or a macro. A block that declares nothing runs in its parent's frame.
class Scope {
  constructor(parent) {
    this.parent = parent;
    this.names = new Map();
  }
  add(name) {
    if (!this.names.has(name)) this.names.set(name, this.names.size);
    return this.names.get(name);
  }
  find(name) {
    let d = 0;
    for (let s = this; s; s = s.parent, d++) if (s.names.has(name)) return { d, k: s.names.get(name) };
    return null;
  }
}
class Frame {
  constructor(p, n) {
    this.p = p;
    this.v = new Array(n).fill(UNINIT);
  }
}
function up(fr, d) {
  let f = fr;
  for (let i = 0; i < d; i++) f = f.p;
  return f;
}
const getter = (d, k) => {
  if (d === 0) return (fr) => fr.v[k];
  if (d === 1) return (fr) => fr.p.v[k];
  if (d === 2) return (fr) => fr.p.p.v[k];
  if (d === 3) return (fr) => fr.p.p.p.v[k];
  return (fr) => up(fr, d).v[k];
};

// An lvalue: a variable, an array element (an accumulator keeps its identity), an int4's
// member (written as a new int4, which a reference to the element then sees), or a
// shared word.
class VarSlot {
  constructor(f, k) {
    this.f = f;
    this.k = k;
  }
  get v() {
    return this.f.v[this.k];
  }
  set v(x) {
    this.f.v[this.k] = x;
  }
}
class Slot {
  constructor(a, i) {
    this.a = a;
    this.i = i;
  }
  get v() {
    return this.a[this.i];
  }
  set v(x) {
    const old = this.a[this.i];
    if (old instanceof Reg) {
      if (x === 0) x = zreg(old.r, old.id);
      else if (!(x instanceof Reg) || x.r !== old.r) throw new Error(`accumulator ${old.r} set to ${describe_(x)}`);
    }
    this.a[this.i] = x;
  }
}
class MemberSlot {
  constructor(base, m) {
    this.base = base;
    this.m = m;
  }
  get v() {
    return member(this.base.v, this.m);
  }
  set v(x) {
    const q = this.base.v;
    if (!(q instanceof Quad)) throw new Error(`member ${this.m} of ${describe_(q)}`);
    const c = { x: q.x, y: q.y, z: q.z, w: q.w };
    c[this.m] = x;
    this.base.v = quad(c.x, c.y, c.z, c.w);
  }
}
class SharedSlot {
  constructor(sh, i) {
    this.sh = sh;
    this.i = i;
  }
  get v() {
    if (this.i instanceof LaneVec) return lanes((l) => this.sh.read(this.i.a[l]));
    return this.sh.read(this.i);
  }
  // The lanes in mask (all without one) store x's lane value at i's.
  store(x, mask) {
    const done = new Map();
    for (let l = 0; l < 32; l++) {
      if (mask !== undefined && !at(mask, l)) continue;
      const i = num(at(this.i, l));
      const v = at(x, l);
      if (done.has(i)) {
        if (done.get(i) !== v) throw new Error(`two lanes store different words to ${this.sh.name}[${i}]`);
        continue;
      }
      done.set(i, v);
      onStore(this.sh, i, v, l);
    }
  }
}
function member(o, m) {
  if (o && typeof o === 'object' && !(o instanceof LaneVec) && m in o) return o[m];
  throw new Error(`.${m} of ${describe_(o)}`);
}
function index(a, i) {
  if (a instanceof Shared) return new SharedSlot(a, i).v;
  if (!Array.isArray(a)) throw new Error(`[${describe_(i)}] of ${describe_(a)}`);
  num(i);
  if (i >= a.length) throw new Error(`index ${i} past ${a.length}`);
  return a[i];
}
function slotOf(a, i) {
  if (a instanceof Shared) return new SharedSlot(a, i);
  if (!Array.isArray(a)) throw new Error(`[${describe_(i)}] of ${describe_(a)}`);
  num(i);
  if (i >= a.length) throw new Error(`index ${i} past ${a.length}`);
  return new Slot(a, i);
}
// An assignment, for the lanes of the running warp's mask (an if on a value that differs
// by lane).
function assign(lv, x) {
  const mask = CTX.warp.mask;
  if (lv instanceof SharedSlot) lv.store(x, mask);
  else if (mask === undefined) lv.v = x;
  else {
    const old = lv.v;
    lv.v = lanes((l) => (at(mask, l) ? at(x, l) : at(old, l)));
  }
}

const SHARED_RE = /\bsT\b|\bsNext\b|st\.shared|\batomicAdd\b|\b__syncthreads\b/;
const DECL_RE = /^((?:(?:const|constexpr)\s+)*)(uint32_t|int32_t|int|bool|int4|uint64_t)\b\s*/;
const DECLARATOR_RE = /^([*&]?)\s*(?:const\s+)?(\w+)\s*((?:\[[^\]]*\]\s*)*)(?:=\s*([\s\S]+))?$/;
const PARAM_RE = /^(?:const\s+)?\w+\s*([*&]?)\s*(\w+)(?:\s*=\s*([\s\S]+))?$/;
const PRED_STORE_RE = /^asm volatile\("\{ \.reg \.pred p; setp\.eq\.u32 p, %2, 0; @p st\.shared\.u32 \[%0\], %1; \}" ::"r"\((.+)\), "r"\((.+)\), "r"\((.+)\) : "memory"\)$/;
// The names a warp starts with: threadIdx.x differs by lane; tiles and the grid are the
// run's; A' at 0, B' at 2^31 (see SPAN).
const ROOT = ['threadIdx', 'blockIdx', 'gridDim', 'Aprime', 'Bprime', 'col_off', 'rows_valid', 'tiles', 'test', 'hits'];

// A build's pearl_fold_bd, ready to run. overrides: lambdas of the source replaced by the
// simulation's own (the tile's coordinates and offsets, and the hash's reads).
class Program {
  constructor(f, overrides) {
    this.f = f;
    this.overrides = overrides;
    this.macros = new Map();
    this.skipped = [];
    const head = 'void pearl_fold_bd(';
    const at0 = f.text.indexOf(head);
    if (at0 < 0) throw new Error('no pearl_fold_bd');
    const open = f.text.indexOf('{', f.text.indexOf(') {', at0));
    const stmts = parseStatements(f.text.slice(open + 1, closeOf(f.text, open) - 1));
    // Named lambdas, and which of them wait (a shared access or a barrier, in them or in
    // what they call): those run as generators.
    this.named = new Map();
    const collect = (list) => {
      for (const n of list) {
        if (n.k === 'simple') {
          const lam = lambdaOf(n.text);
          if (lam && lam.name) {
            this.named.set(lam.name, lam);
            if (!overrides[lam.name]) collect(lam.stmts);
          }
        } else if (n.k === 'block') collect(n.body);
        else if (n.k === 'for') collect([n.body]);
        else if (n.k === 'if') collect([n.then, ...(n.els ? [n.els] : [])]);
      }
    };
    collect(stmts);
    this.gen = new Set();
    for (let grew = true; grew;) {
      grew = false;
      for (const [name, lam] of this.named) {
        if (!this.gen.has(name) && !overrides[name] && lam.stmts.some((n) => this.waits(n))) {
          this.gen.add(name);
          grew = true;
        }
      }
    }
    this.root = new Scope(null);
    for (const n of ROOT) this.root.add(n);
    this.bodyScope = new Scope(this.root);
    this.main = this.block(stmts, this.bodyScope, true);
  }

  // Whether a statement can wait: it touches shared memory or calls something that can
  // (a named lambda that does, a waiting builtin, or a lambda passed in).
  waits(n) {
    if (n.k === 'block') return n.body.some((x) => this.waits(x));
    if (n.k === 'for') return this.waits(n.body);
    if (n.k === 'if') return this.waits(n.then) || (n.els !== null && this.waits(n.els));
    if (n.k === 'return') return false;
    if (/^auto \w+ = \[&\]/.test(n.text)) return false;
    if (SHARED_RE.test(n.text)) return true;
    for (const [, name] of n.text.matchAll(/\b([A-Za-z_]\w*)\s*\(/g)) if (this.waitingCall(name)) return true;
    return false;
  }

  waitingCall(name) {
    if (this.gen.has(name) || name === 'kstep') return true;
    if (this.overrides[name]) return this.overrides[name].gen;
    if (BUILTINS[name]) return BUILTINS[name].gen;
    return !this.named.has(name) && !TYPES.has(name) && !['asm', 'volatile', 'static_assert'].includes(name);
  }

  // A name's value outside the frames: a builtin, or an object-like macro.
  global(name) {
    if (BUILTINS[name]) return BUILTINS[name];
    if (!this.macros.has(name)) {
      const d = this.f.defs.get(name);
      if (!d || d.params) throw new Error(`unknown name ${name}`);
      this.macros.set(name, this.expr(d.body, null)(null));
    }
    return this.macros.get(name);
  }

  expr(text, sc) {
    return this.val(parseExpr(text), sc);
  }

  val(n, sc) {
    switch (n.t) {
      case 'num': {
        const v = n.v;
        return () => v;
      }
      case 'id': {
        const name = n.n;
        if (name === 'true') return () => 1;
        if (name === 'false') return () => 0;
        if (sc && sc.lits && sc.lits.has(name)) {
          const def = sc.lits.get(name);
          return (fr) => new Lambda(def, fr);
        }
        const r = sc ? sc.find(name) : null;
        if (!r) {
          let g;
          return () => (g === undefined ? (g = this.global(name)) : g);
        }
        const get = getter(r.d, r.k);
        return (fr) => {
          const x = get(fr);
          if (x === UNINIT) throw new Error(`${name} read before it is set`);
          return x instanceof Ref ? x.lv.v : x;
        };
      }
      case 'cast':
        return this.val(n.a, sc);
      case 'idx': {
        const a = this.val(n.a, sc);
        const i = this.val(n.i, sc);
        return (fr) => index(a(fr), i(fr));
      }
      case 'mem': {
        const a = this.val(n.a, sc);
        const m = n.m;
        return (fr) => member(a(fr), m);
      }
      case 'un': {
        const a = this.val(n.a, sc);
        if (n.op === '!') return (fr) => lift((x) => (x ? 0 : 1))(a(fr), 0);
        if (n.op === '~') return (fr) => lift((x) => (~x) >>> 0)(a(fr), 0);
        return (fr) => lift((x) => (-x) >>> 0)(a(fr), 0);
      }
      case 'bin': {
        const a = this.val(n.a, sc);
        const b = this.val(n.b, sc);
        if (n.op === '&&') return (fr) => (truthy(a(fr)) && truthy(b(fr)) ? 1 : 0);
        if (n.op === '||') return (fr) => (truthy(a(fr)) || truthy(b(fr)) ? 1 : 0);
        const f = BIN[n.op];
        return (fr) => f(a(fr), b(fr));
      }
      case 'tern': {
        const c = this.val(n.c, sc);
        const a = this.val(n.a, sc);
        const b = this.val(n.b, sc);
        return (fr) => {
          const cv = c(fr);
          if (cv instanceof LaneVec) {
            const av = a(fr);
            const bv = b(fr);
            return lanes((l) => (num(cv.a[l]) ? at(av, l) : at(bv, l)));
          }
          return truthy(cv) ? a(fr) : b(fr);
        };
      }
      case 'call': {
        const call = this.call(n, sc);
        return (fr) => {
          const { f, args } = call(fr);
          return invoke(f, args);
        };
      }
      default:
        throw new Error(`expression ${n.t}`);
    }
  }

  lv(n, sc) {
    switch (n.t) {
      case 'id': {
        const r = sc && sc.find(n.n);
        if (!r) return null;
        const { d, k } = r;
        return (fr) => {
          const f = up(fr, d);
          const x = f.v[k];
          return x instanceof Ref ? x.lv : new VarSlot(f, k);
        };
      }
      case 'idx': {
        const a = this.val(n.a, sc);
        const i = this.val(n.i, sc);
        return (fr) => slotOf(a(fr), i(fr));
      }
      case 'mem': {
        const b = this.lv(n.a, sc);
        return b && ((fr) => new MemberSlot(b(fr), n.m));
      }
      case 'tern': {
        const c = this.val(n.c, sc);
        const a = this.lv(n.a, sc);
        const b = this.lv(n.b, sc);
        return a && b && ((fr) => (truthy(c(fr)) ? a(fr) : b(fr)));
      }
      case 'cast':
        return this.lv(n.a, sc);
      default:
        return null;
    }
  }

  // A call's callee and arguments: values, or lvalues for a builtin's reference
  // parameters and a lambda's & ones.
  call(n, sc) {
    if (n.f.t !== 'id') throw new Error('a call through an expression');
    const callee = this.val(n.f, sc);
    const args = n.args.map((a) => ({ v: this.val(a, sc), l: this.lv(a, sc) }));
    return (fr) => {
      const f = callee(fr);
      let refs;
      if (f instanceof Builtin) refs = f.refs;
      else if (f instanceof Lambda) refs = f.def.refs;
      else throw new Error(`call of ${describe_(f)}`);
      return {
        f,
        args: args.map((a, k) => {
          if (!refs.includes(k)) return a.v(fr);
          if (!a.l) throw new Error('a reference to a value');
          return a.l(fr);
        }),
      };
    };
  }

  // A lambda's compiled parts: its parameters in a frame of their own, then its body.
  lambda(lam, sc, name) {
    const ps = new Scope(sc);
    const params = lam.params.map((p) => ({
      ref: p.ref, k: ps.add(p.name), def: p.def ? this.expr(p.def, ps) : null,
    }));
    const def = { name, params, refs: params.map((p, k) => (p.ref ? k : -1)).filter((k) => k >= 0), scope: ps };
    // The body's outer declarations share the parameters' scope, as in C++.
    def.body = this.block(lam.stmts, ps, true);
    return def;
  }

  // own: the block has a frame of its own already (a call's or the warp's).
  block(list, sc, own) {
    const declares = list.some((n) => n.k === 'simple' && (DECL_RE.test(n.text) || /^auto \w+ = /.test(n.text)
      || /^(?:extern )?__shared__ /.test(n.text)));
    const bs = declares && !own ? new Scope(sc) : sc;
    const parts = list.map((n) => this.statement(n, bs));
    const fresh = bs !== sc;
    if (!parts.some((p) => p.gen)) {
      return {
        gen: false,
        run: (fr) => {
          const f = fresh ? new Frame(fr, bs.names.size) : fr;
          for (const p of parts) {
            const r = p.run(f);
            if (r) return r;
          }
          return undefined;
        },
      };
    }
    return {
      gen: true,
      run: function* run(fr) {
        const f = fresh ? new Frame(fr, bs.names.size) : fr;
        for (const p of parts) {
          const r = p.gen ? yield* p.run(f) : p.run(f);
          if (r) return r;
        }
        return undefined;
      },
    };
  }

  statement(n, sc) {
    if (n.k === 'block') return this.block(n.body, sc, false);
    if (n.k === 'return') {
      const e = n.expr ? this.expr(n.expr, sc) : () => undefined;
      return { gen: false, run: (fr) => ({ ret: e(fr) }) };
    }
    if (n.k === 'if') {
      const c = this.expr(n.cond, sc);
      const then = this.statement(n.then, sc);
      const els = n.els ? this.statement(n.els, sc) : null;
      const gen = then.gen || (els !== null && els.gen);
      // On a value that differs by lane, the body runs for those lanes: the warp's mask.
      const maskOf = (cv) => {
        if (els) throw new Error('an else on a value that differs by lane');
        const outer = CTX.warp.mask;
        return lanes((l) => (num(cv.a[l]) && (outer === undefined || at(outer, l)) ? 1 : 0));
      };
      if (!gen) {
        return {
          gen,
          run: (fr) => {
            const cv = c(fr);
            if (cv instanceof LaneVec) {
              const m = maskOf(cv);
              if (m === 0) return undefined;
              const W = CTX.warp;
              const outer = W.mask;
              W.mask = m;
              try {
                return then.run(fr);
              } finally {
                W.mask = outer;
              }
            }
            if (truthy(cv)) return then.run(fr);
            return els ? els.run(fr) : undefined;
          },
        };
      }
      return {
        gen,
        run: function* run(fr) {
          const cv = c(fr);
          let pick = els;
          let m;
          if (cv instanceof LaneVec) {
            m = maskOf(cv);
            pick = m === 0 ? null : then;
          } else if (truthy(cv)) {
            pick = then;
          }
          if (!pick) return undefined;
          const W = CTX.warp;
          const outer = W.mask;
          if (m !== undefined) W.mask = m;
          try {
            return pick.gen ? yield* pick.run(fr) : pick.run(fr);
          } finally {
            W.mask = outer;
          }
        },
      };
    }
    if (n.k === 'for') {
      const fs_ = new Scope(sc);
      const init = this.simple(n.init, fs_);
      const cond = this.expr(n.cond, fs_);
      const step = this.simple(n.step, fs_);
      const body = this.statement(n.body, fs_);
      if (init.gen || step.gen) throw new Error('a for header that waits');
      if (!body.gen) {
        return {
          gen: false,
          run: (fr) => {
            const f = new Frame(fr, fs_.names.size);
            init.run(f);
            while (truthy(cond(f))) {
              const r = body.run(f);
              if (r) return r;
              step.run(f);
            }
            return undefined;
          },
        };
      }
      return {
        gen: true,
        run: function* run(fr) {
          const f = new Frame(fr, fs_.names.size);
          init.run(f);
          while (truthy(cond(f))) {
            const r = yield* body.run(f);
            if (r) return r;
            step.run(f);
          }
          return undefined;
        },
      };
    }
    return this.simple(n.text, sc);
  }

  simple(text, sc) {
    const nop = { gen: false, run: () => undefined };
    if (/^static_assert\(/.test(text)) {
      this.skipped.push(text);
      return nop;
    }
    // (void)x, and an empty asm: opaque to the compiler, the values stay.
    if (/^\(void\)\w+$/.test(text) || /^asm(?: volatile)?\(""\s*:[^"]*(?:"[^"]*"[^"]*)*\)$/.test(text)) return nop;
    let m;
    if ((m = PRED_STORE_RE.exec(text))) {
      // A predicated st.shared at byte address a (sT0 + 4 i): the lanes whose third
      // operand is 0 store.
      const a = this.expr(m[1], sc);
      const x = this.expr(m[2], sc);
      const p = this.expr(m[3], sc);
      const sT = this.expr('sT', sc);
      return {
        gen: true,
        run: function* run(fr) {
          yield SHARED;
          const av = a(fr);
          const pv = p(fr);
          const idx = lanes((l) => {
            const b = num(at(av, l));
            if (b % 4) throw new Error(`st.shared at byte ${b}`);
            return b / 4;
          });
          const outer = CTX.warp.mask;
          const mask = lanes((l) => (num(at(pv, l)) === 0 && (outer === undefined || at(outer, l)) ? 1 : 0));
          new SharedSlot(sT(fr), idx).store(x(fr), mask);
          return undefined;
        },
      };
    }
    const lam = lambdaOf(text);
    if (lam && lam.name) {
      const name = lam.name;
      const k = sc.add(name);
      if (this.overrides[name]) {
        const b = this.overrides[name];
        if (b.compile) b.compile(this, sc);
        return { gen: false, run: (fr) => { fr.v[k] = b; if (b.frame) b.frame(fr); } };
      }
      const def = this.lambda(lam, sc, name);
      // readout_gather reads only its argument (pinned in the tests): its result is
      // remembered for each fold it is given.
      if (name === 'readout_gather') def.pure = true;
      if (name === 'kstep') def.memo = { data: KSTEP_DATA.map((x) => sc.find(x)), nums: KSTEP_NUMS.map((x) => sc.find(x)) };
      return { gen: false, run: (fr) => { fr.v[k] = new Lambda(def, fr); } };
    }
    if ((m = /^extern __shared__ __align__\(\d+\) uint32_t (\w+)\[\]$/.exec(text))) {
      const k = sc.add(m[1]);
      return { gen: false, run: (fr) => { fr.v[k] = CTX.warp.blk.sT; } };
    }
    if ((m = /^__shared__ uint32_t (\w+)\[(\d+)\]$/.exec(text))) {
      const name = m[1];
      const size = Number(m[2]);
      const k = sc.add(name);
      return { gen: false, run: (fr) => { fr.v[k] = CTX.warp.blk.shared(name, size); } };
    }
    // Lambda literals in the statement: names bound to them where it runs.
    let t = text;
    for (let k = t.indexOf('[&]('); k >= 0; k = t.indexOf('[&](')) {
      const l = lambdaOf(t.slice(k));
      if (!sc.lits) sc.lits = new Map();
      const name = `__lam${sc.lits.size}`;
      sc.lits.set(name, this.lambda(l, sc, null));
      t = t.slice(0, k) + name + t.slice(k + l.end);
    }
    const plain = this.plain(t, sc);
    if (plain.call && this.waitingCall(plain.callee)) {
      const call = plain.call;
      const steps = plain.callee === 'kstep' ? KSTEP_STEPS : 0;
      return {
        gen: true,
        run: function* run(fr) {
          for (let i = 0; i < steps; i++) yield SHARED;
          const { f, args } = call(fr);
          if (f instanceof Builtin && f.gen) yield* f.fn(fr, ...args);
          else if (f instanceof Lambda && f.def.body.gen) yield* f.def.body.run(bind(f, args));
          else invoke(f, args);
          return undefined;
        },
      };
    }
    if (!SHARED_RE.test(t)) return { gen: false, run: plain.run };
    return {
      gen: true,
      run: function* run(fr) {
        yield SHARED;
        plain.run(fr);
        return undefined;
      },
    };
  }

  // A statement that does not wait as a whole: a declaration, an assignment, a ++, or a
  // call.
  plain(text, sc) {
    let m;
    if ((m = DECL_RE.exec(text))) {
      const type = m[2];
      const ds = splitTop(text.slice(m[0].length), ',').map((d) => {
        const q = DECLARATOR_RE.exec(d);
        if (!q) throw new Error(`declarator "${d}" in "${text}"`);
        const dims = [...q[3].matchAll(/\[([^\]]*)\]/g)].map((x) => this.expr(x[1], sc));
        const init = q[4];
        const list = init && init.startsWith('{') ? splitTop(init.slice(1, -1), ',').map((e) => this.expr(e, sc)) : null;
        let ie = null;
        if (init && !list) ie = q[1] === '&' ? this.lv(parseExpr(init), sc) : this.expr(init, sc);
        if (q[1] === '&' && !ie) throw new Error(`reference ${q[2]} to a value`);
        // Declared after its initialiser is read, as in C++.
        return { kind: q[1], k: sc.add(q[2]), dims, list, init: ie };
      });
      return {
        run: (fr) => {
          for (const d of ds) {
            if (d.kind === '&') fr.v[d.k] = new Ref(d.init(fr));
            else if (d.dims.length) fr.v[d.k] = makeArray(d.dims.map((e) => num(e(fr))), type, d.list && d.list.map((e) => e(fr)));
            else fr.v[d.k] = d.init ? d.init(fr) : UNINIT;
          }
          return undefined;
        },
      };
    }
    if ((m = /^(\w+)\+\+$/.exec(text))) {
      const lv = this.lv({ t: 'id', n: m[1] }, sc);
      if (!lv) throw new Error(`${m[1]}++ of an undeclared name`);
      return { run: (fr) => { const l = lv(fr); assign(l, BIN['+'](l.v, 1)); } };
    }
    const as = assignSplit(text);
    if (as) {
      const lv = this.lv(parseExpr(as.lhs), sc);
      if (!lv) throw new Error(`assignment to "${as.lhs}"`);
      const rhs = this.expr(as.rhs, sc);
      const op = as.op ? BIN[as.op] : null;
      if (as.op && !op) throw new Error(`${as.op}= in "${text}"`);
      return {
        run: (fr) => {
          const l = lv(fr);
          const x = rhs(fr);
          assign(l, op ? op(l.v, x) : x);
          return undefined;
        },
      };
    }
    const e = parseExpr(text);
    if (e.t !== 'call') throw new Error(`statement not understood: ${text}`);
    const call = this.call(e, sc);
    return {
      call,
      callee: e.f.n,
      run: (fr) => {
        const { f, args } = call(fr);
        invoke(f, args);
        return undefined;
      },
    };
  }
}

// A lambda's parts from a statement or a literal: [&](params) [-> type] { body }.
function lambdaOf(text) {
  const m = /^(?:auto (\w+) = )?\[&\]\(/.exec(text);
  if (!m) return null;
  const p0 = m[0].length - 1;
  const p1 = closeOf(text, p0);
  const b0 = text.indexOf('{', p1);
  const b1 = closeOf(text, b0);
  if (b1 !== text.length && m[1]) throw new Error(`a lambda with more after it: ${text}`);
  const params = text.slice(p0 + 1, p1 - 1).trim();
  return {
    name: m[1],
    end: b1,
    params: params ? splitTop(params, ',').map((p) => {
      const q = PARAM_RE.exec(p);
      if (!q) throw new Error(`parameter "${p}"`);
      return { ref: q[1] === '&', name: q[2], def: q[3] || null };
    }) : [],
    stmts: parseStatements(text.slice(b0 + 1, b1 - 1)),
  };
}

// The assignment in a statement, if any: { lhs, op (for x op= y), rhs }.
function assignSplit(text) {
  let depth = 0;
  for (let k = 0; k < text.length; k++) {
    const ch = text[k];
    if (ch === '"') {
      for (k++; k < text.length && text[k] !== '"'; k++) if (text[k] === BS) k++;
    } else if ('([{'.includes(ch)) {
      depth++;
    } else if (')]}'.includes(ch)) {
      depth--;
    } else if (ch === '=' && depth === 0) {
      if (text[k + 1] === '=') {
        k++;
        continue;
      }
      const two = text.slice(k - 2, k);
      if (two === '<<' || two === '>>') return { lhs: text.slice(0, k - 2).trim(), op: two, rhs: text.slice(k + 1).trim() };
      const pv = text[k - 1];
      if ('=!<>'.includes(pv)) continue;
      if ('^|&+-*/%'.includes(pv)) return { lhs: text.slice(0, k - 1).trim(), op: pv, rhs: text.slice(k + 1).trim() };
      return { lhs: text.slice(0, k).trim(), op: null, rhs: text.slice(k + 1).trim() };
    }
  }
  return null;
}

// int32_t acc[MB][NB][4] is the accumulators, the rest plain arrays, unset or from a
// brace list.
function makeArray(dims, type, list) {
  if (type === 'int32_t' && dims.length === 3) {
    return Array.from({ length: dims[0] }, (_, a) => Array.from({ length: dims[1] }, (__, b) => Array.from(
      { length: dims[2] }, (___, c) => reg((a * dims[1] + b) * dims[2] + c, -1))));
  }
  if (list && dims.length !== 1) throw new Error('a brace list for a 2-D array');
  const make = (d) => (d === dims.length - 1
    ? Array.from({ length: dims[d] }, (_, k) => (list ? (k < list.length ? list[k] : 0) : UNINIT))
    : Array.from({ length: dims[d] }, () => make(d + 1)));
  return make(0);
}

// A lambda's frame for a call: the parameters (a & one a reference to the argument's
// lvalue), below its closure's frame.
function bind(f, args) {
  if (CTX.warp.mask instanceof LaneVec) throw new Error(`${f.def.name} called for some lanes only`);
  const ps = f.def.params;
  if (args.length > ps.length) throw new Error(`${args.length} arguments for ${ps.length}`);
  const nf = new Frame(f.closure, f.def.scope.names.size);
  for (let k = 0; k < ps.length; k++) {
    const p = ps[k];
    if (p.ref) nf.v[p.k] = new Ref(args[k]);
    else nf.v[p.k] = k < args.length ? args[k] : p.def(nf);
  }
  return nf;
}

function invoke(f, args) {
  if (f instanceof Builtin) {
    if (f.gen) throw new Error(`${f.name} waits, called inside an expression`);
    return f.fn(null, ...args);
  }
  if (f.def.body.gen) throw new Error(`${f.def.name || 'a lambda'} waits, called inside an expression`);
  if (f.def.memo && RUN.memo) return memoKstep(f, args);
  if (f.def.pure && args[0] instanceof Sym) {
    let c = RUN.pure.get(f.def);
    if (!c) RUN.pure.set(f.def, (c = new WeakMap()));
    if (!c.has(args[0])) {
      const r = f.def.body.run(bind(f, args));
      c.set(args[0], r && r.ret);
    }
    return c.get(args[0]);
  }
  const r = f.def.body.run(bind(f, args));
  return r && r.ret;
}

// kstep, the 128 IMMA of a k-step, is most of the work. Every warp of a block runs it on
// the same values (a warp's words, accumulators and folds do not depend on which warp it
// is), so the first warp to make its n-th call runs it, and the others, when they make
// theirs on the same arguments and the same state, take its result. kstep reads and
// writes only these of the fold's variables (pinned in the tests).
const KSTEP_DATA = ['af', 'bf', 'acc', 'rxs'];
const KSTEP_NUMS = ['arun', 'brun', 'aoff', 'boff', 'aoffn', 'boffn'];
// A k-step takes this many of a warp's turns (a turn for 16 of its IMMA), so warps that
// run k-steps fall behind warps that only read, as on the card.
const KSTEP_STEPS = 8;
function flatState(cl, slots) {
  const out = [];
  const walk = (v) => {
    if (Array.isArray(v)) v.forEach(walk);
    else out.push(v);
  };
  for (const s of slots) walk(up(cl, s.d).v[s.k]);
  return out;
}
function setState(cl, slots, vals) {
  let k = 0;
  const walk = (a) => {
    for (let i = 0; i < a.length; i++) {
      if (Array.isArray(a[i])) walk(a[i]);
      else a[i] = vals[k++];
    }
  };
  for (const s of slots) walk(up(cl, s.d).v[s.k]);
}
function memoKstep(f, args) {
  const W = CTX.warp;
  const n = W.kcalls++;
  const cl = f.closure;
  const { data, nums } = f.def.memo;
  const key = args.join(',') + '|' + nums.map((s) => (s ? String(up(cl, s.d).v[s.k] === UNINIT ? 'u' : up(cl, s.d).v[s.k]) : '-')).join(',');
  const pre = flatState(cl, data);
  const e = W.blk.memo.get(n);
  if (e && e.key === key && e.pre.length === pre.length && e.pre.every((v, i) => v === pre[i])) {
    setState(cl, data, e.post);
    RUN.memoHits++;
    return undefined;
  }
  f.def.body.run(bind(f, args));
  W.blk.memo.set(n, { key, pre, post: flatState(cl, data) });
  return undefined;
}

// ---- The simulated machine -----------------------------------------------------------

// The offsets a_off and b_off stand for: tile v's A' at v * SPAN, its B' at 2^31 + v *
// SPAN (Bprime), quad q at q * FRAG_BLOCK, k-step s at s * 512. The lane's 16 bytes and
// the warp's blocks are left out: they are the same for every tile.
const SPAN = 2 ** 20;
const BBASE = 2 ** 31;
let FRAG = 0;
function decodeLoad(p) {
  num(p);
  const kind = p >= BBASE ? 'B' : 'A';
  const o = p - (kind === 'B' ? BBASE : 0);
  const tile = Math.floor(o / SPAN);
  const q = Math.floor((o % SPAN) / FRAG);
  const r = o % FRAG;
  if (r % 512 || q > 3) throw new Error(`a load at ${kind} + ${o}, off the fragments`);
  return { kind, tile, s: r / 512, q };
}
function load(p) {
  const d = decodeLoad(p);
  if (d.tile >= RUN.tiles) bad('a load past the last tile');
  const w = (c) => word(d.kind, d.tile, d.s, d.q, c, EMPTY);
  return quad(w(0), w(1), w(2), w(3));
}

function bad(what) {
  RUN.bad[what] = (RUN.bad[what] || 0) + 1;
}

// pearl_bd_imma(c0, c1, a, b): c0, c1 += a x b, one m8n8k16 on Turing. With k-lo first,
// the pair acc[mb][nb][2u], [2u + 1] takes A word (mb & 1) * 2 + u of quad (mb / 2) * 2 +
// hi and B word (j & 1) * 2 + odd of quad (j / 2) * 2 + hi, with j = nb / 2, odd = nb & 1
// and hi the half (turingKorder.test.js checks these are pearl_mma_m16n8k32's words).
// The first IMMA of a tile takes C = 0 (the accumulator was set to 0); each later one
// must sum the next half of the same tile.
function imma(fr, c0, c1, a, b) {
  RUN.imma++;
  if (!(a instanceof Word) || a.kind !== 'A' || !(b instanceof Word) || b.kind !== 'B') {
    bad('operands');
    return;
  }
  const half = a.q & 1;
  if (a.tile !== b.tile || a.s !== b.s || (b.q & 1) !== half) bad('operands of two k-steps');
  const W = CTX.warp;
  for (let k = 0; k < 2; k++) {
    const lv = k ? c1 : c0;
    const old = lv.v;
    if (!(old instanceof Reg)) throw new Error(`an IMMA on ${describe_(old)}`);
    const r = old.r;
    const mb = r >> 5;
    const nb = (r >> 2) & 7;
    const i = r & 3;
    if ((i & 1) !== k) bad('accumulator pair');
    const u = i >> 1;
    if (a.q !== (mb >> 1) * 2 + half || a.comp !== (mb & 1) * 2 + u) bad('A word');
    if (b.q !== (nb >> 2) * 2 + half || b.comp !== ((nb >> 1) & 1) * 2 + (nb & 1)) bad('B word');
    let st = 0;
    if (old.zero) {
      // C = 0, the tile's first IMMA on it. If a fold read the value it replaces since
      // the last barrier, this IMMA must wait on that fold, through one of its words.
      const fe = RUN.foldedAt.get(old.prev);
      if (fe === W.epoch) {
        RUN.gateChecks++;
        if (!a.deps.has(old.prev) && !b.deps.has(old.prev)) RUN.ungated++;
      }
      const h = W.blk.heads.get(W.epoch) || new Set();
      h.add(a.tile);
      W.blk.heads.set(W.epoch, h);
      if (a.s === 0 && half === 0) st = stCode(a.tile, 0, 0);
      else bad('a first IMMA not at k-step 0');
    } else if (old.st > 0) {
      const p = stOf(old.st);
      const ns = p.half ? p.s + 1 : p.s;
      if (p.tile === a.tile && ns === a.s && (p.half ^ 1) === half) st = stCode(a.tile, a.s, half);
      else bad('an IMMA out of order');
    } else {
      bad('an IMMA on garbage');
    }
    lv.v = reg(r, st);
  }
}

function shfl(fr, mask, x, d) {
  if (mask !== 0xffffffff) throw new Error(`shuffle mask ${mask}`);
  if (![4, 8, 12].includes(d)) throw new Error(`shuffle by ${d}`);
  if (!(x instanceof Sym)) throw new Error(`shuffle of ${describe_(x)}`);
  const t = new Set();
  for (const v of x.terms) t.add(v ^ (d >> 2));
  return new Sym(t, x.deps);
}

const BUILTINS = {
  pearl_xor3: new Builtin('pearl_xor3', (fr, a, b, c) => xorV(xorV(a, b), c)),
  pearl_bd_after: new Builtin('pearl_bd_after', (fr, b, x) => {
    if (!(b instanceof Word)) throw new Error(`a gate on ${describe_(b)}`);
    return word(b.kind, b.tile, b.s, b.q, b.comp, union(b.deps, depsOf(x)));
  }),
  pearl_bd_imma: new Builtin('pearl_bd_imma', imma, { refs: [0, 1] }),
  pearl_bd_ldg: new Builtin('pearl_bd_ldg', (fr, p) => load(p)),
  pearl_bd_ldg_c: new Builtin('pearl_bd_ldg_c', (fr, p) => load(p)),
  __shfl_xor_sync: new Builtin('__shfl_xor_sync', shfl),
  __ldg: new Builtin('__ldg', () => 0x80000001),
  __trap: new Builtin('__trap', () => {
    throw new Error('__trap');
  }),
  __cvta_generic_to_shared: new Builtin('__cvta_generic_to_shared', (fr, p) => {
    if (!(p instanceof Shared) || p.name !== 'sT') throw new Error(`__cvta of ${describe_(p)}`);
    return 0;
  }),
  // The tile counter: one add a lane of the mask, in lane order.
  atomicAdd: new Builtin('atomicAdd', (fr, p, n) => {
    const mask = CTX.warp.mask;
    if (!(mask instanceof LaneVec) || n !== 1) throw new Error('atomicAdd not by one lane');
    return lanes((l) => (mask.a[l] ? RUN.grab(CTX.warp.blk) : 0));
  }),
  __syncthreads: new Builtin('__syncthreads', function* sync() {
    yield BAR;
  }, { gen: true }),
};

// The lambdas the simulation runs in place of the source's: tile v's coordinates are v
// itself, its offsets v's (above); hash_tile reads its region's 16 words where the
// source's hash_tile reads them, with the source's hashers and regions, and checks each
// is its tile's word for that chunk.
function overridesFor(f) {
  const hc = f.text.includes('auto hash_tile = [&](uint32_t vtile) {');
  const ht = flat(block(f.text, hc ? 'auto hash_tile = [&](uint32_t vtile) {' : 'auto hash_tile = [&](uint32_t rbg, uint32_t cbg) {'));
  const role = /const bool hasher = ([^;]+); const uint32_t L = ([^;]+); if \(hasher\) \{/.exec(ht);
  const rd = /for \(uint32_t c = 0; c < 16u; c\+\+\) tm\[c\] = sT\[([^;]+)\];/.exec(ht);
  if (!role || !rd) throw new Error('hash_tile: no hashers or reads');
  const osid = ht.includes('uint32_t osid = sid; asm("" : "+r"(osid));');
  const hash = new Builtin('hash_tile', function* hashTile(fr, a0, a1) {
    if (!hc && a0 !== a1) throw new Error('hash_tile of two tiles');
    const tile = a0;
    const W = CTX.warp;
    const { fns, k } = hash;
    const hf = new Frame(W.hashFrame, 3);
    if (osid) hf.v[k.osid] = fns.sid(hf);
    const hasher = fns.hasher(hf);
    hf.v[k.L] = fns.L(hf);
    W.blk.hashCalls.push({ warp: W.w, tile, epoch: W.epoch, lanes: lanes((l) => (num(at(hasher, l)) ? 1 : 0)) });
    for (let c = 0; c < 16; c++) {
      yield SHARED;
      hf.v[k.c] = c;
      const idx = fns.read(hf);
      for (let l = 0; l < 32; l++) {
        if (!num(at(hasher, l))) continue;
        const tag = W.blk.sT.read(num(at(idx, l)));
        RUN.reads++;
        RUN.readsOf.set(tile, (RUN.readsOf.get(tile) || 0) + 1);
        if (typeof tag !== 'string' || !tag.startsWith(`${tile}.${c}.`)) RUN.wrong++;
      }
    }
  }, { gen: true });
  // Its expressions, compiled where the source defines hash_tile, with its osid, L and c.
  hash.compile = (prog, sc) => {
    const hs = new Scope(sc);
    hash.k = { osid: hs.add('osid'), L: hs.add('L'), c: hs.add('c') };
    hash.fns = { sid: prog.expr('sid', hs), hasher: prog.expr(role[1], hs), L: prog.expr(role[2], hs), read: prog.expr(rd[1], hs) };
  };
  hash.frame = (fr) => {
    CTX.warp.hashFrame = fr;
  };
  return {
    exprs: { hasher: role[1], L: role[2], read: rd[1] },
    tile_coords: new Builtin('tile_coords', (fr, v, r, c) => {
      r.v = num(v);
      c.v = v;
    }, { refs: [1, 2] }),
    a_off: new Builtin('a_off', (fr, v) => num(v) * SPAN),
    b_off: new Builtin('b_off', (fr, v) => num(v) * SPAN),
    hash_tile: hash,
  };
}

// A store to shared. sT's words are kept as tags: tile, chunk, region half (or 'all'),
// and the warp and lane that stored it; a word that is not one whole region of one
// chunk is kept as BAD.
function onStore(sh, i, v, lane) {
  const blk = sh.blk;
  if (sh.name !== 'sT') {
    if (i >= sh.arr.length) throw new Error(`${sh.name}[${i}] is past its end`);
    sh.arr[i] = v;
    return;
  }
  if (i >= sh.arr.length) {
    RUN.outside++;
    return;
  }
  RUN.stores++;
  const w = wordOf(v);
  if (w.bad) {
    bad(`store: ${w.bad}`);
    sh.arr[i] = `BAD ${w.bad}`;
    return;
  }
  // Word c of region L sits at L * 16 + (c ^ ((L >> 1) & 15)) in its buffer.
  const L = (i & 2047) >> 4;
  if (((i & 15) ^ ((L >> 1) & 15)) !== w.c) RUN.misplaced++;
  const W = CTX.warp;
  sh.arr[i] = `${w.tile}.${w.c}.${w.r}.${W.w}.${lane}`;
  const k = `${w.tile}.${i}`;
  if (blk.stored.has(k)) RUN.twice++;
  blk.stored.add(k);
  if (!blk.storesOf.has(w.tile)) blk.storesOf.set(w.tile, []);
  blk.storesOf.get(w.tile).push({ i, c: w.c, epoch: W.epoch });
}

// A build ready to simulate.
const programs = new Map();
function programOf(spec, text = FOLD) {
  const key = spec + '|' + (text === FOLD ? '' : text);
  if (!programs.has(key)) {
    const f = built(spec, '', text);
    const ov = overridesFor(f);
    const prog = new Program(f, { tile_coords: ov.tile_coords, a_off: ov.a_off, b_off: ov.b_off, hash_tile: ov.hash_tile });
    prog.ov = ov;
    programs.set(key, prog);
  }
  return programs.get(key);
}

// Runs a grid of blocks of a build: G blocks (or the listed ones of a grid of G), tiles
// tiles, every block's eight warps a turn at a time in a random order, each block's
// meeting at its barriers. grab: the tile counter's next value for a block (by default
// one counter for the grid). Returns what it saw.
function runFold(spec, { G = 1, only = null, tiles, seed, grab = null, text = FOLD }) {
  const prog = programOf(spec, text);
  FRAG = prog.global('PEARL_FOLD_K') / 32 * 512; // FRAG_BLOCK, (k / 32) * 512
  RUN = {
    vals: new Map(), regs: new Map(), ids: 0, foldedAt: new Map(), memo: true, pure: new Map(), counter: 0,
    imma: 0, stores: 0, reads: 0, wrong: 0, twice: 0,
    outside: 0, misplaced: 0, gateChecks: 0, ungated: 0, memoHits: 0, bad: {}, readsOf: new Map(),
  };
  RUN.grab = grab || (() => RUN.counter++);
  RUN.tiles = tiles;
  const rand = xorshift(seed);
  const blocks = (only || Array.from({ length: G }, (_, b) => b)).map((b) => {
    const blk = {
      b, memo: new Map(), handoffs: [], hashCalls: [], heads: new Map(), stored: new Set(), storesOf: new Map(),
      atBar: [], finished: 0, barriers: 0, lastFold: -1, warps: [], sharedArrs: new Map(),
      shared(name, size) {
        if (!this.sharedArrs.has(name)) this.sharedArrs.set(name, new Shared(name, size, this));
        return this.sharedArrs.get(name);
      },
    };
    blk.sT = new Shared('sT', 2 * 2048, blk);
    for (let w = 0; w < 8; w++) {
      const root = new Frame(null, ROOT.length);
      const tid = new LaneVec(Array.from({ length: 32 }, (_, l) => w * 32 + l));
      const vals = {
        threadIdx: { x: tid }, blockIdx: { x: b }, gridDim: { x: G }, Aprime: 0, Bprime: BBASE, col_off: 0,
        rows_valid: 256, tiles, test: {}, hits: { count: 0 },
      };
      ROOT.forEach((n, k) => { root.v[k] = vals[n]; });
      const W = { w, blk, epoch: 0, kcalls: 0, done: false, mask: undefined, hashFrame: null };
      W.gen = prog.main.run(new Frame(root, prog.bodyScope.names.size));
      blk.warps.push(W);
    }
    return blk;
  });
  const live = blocks.flatMap((blk) => blk.warps);
  while (live.length) {
    const k = rand() % live.length;
    const W = live[k];
    CTX.warp = W;
    let r;
    try {
      r = W.gen.next();
    } catch (e) {
      e.message = `block ${W.blk.b} warp ${W.w}: ${e.message}`;
      throw e;
    }
    if (r.done || r.value === BAR) {
      live[k] = live[live.length - 1];
      live.pop();
    }
    if (r.done) {
      W.done = true;
      W.blk.finished++;
    } else if (r.value === BAR) {
      const blk = W.blk;
      blk.atBar.push(W);
      if (blk.atBar.length === 8) {
        const e = blk.atBar[0].epoch;
        if (blk.atBar.some((x) => x.epoch !== e)) throw new Error('warps of a block meet at different barriers');
        handoff(blk, e);
        for (const x of blk.atBar) {
          x.epoch++;
          live.push(x);
        }
        blk.atBar = [];
        blk.barriers++;
      }
    } else if (r.value !== SHARED) {
      throw new Error(`a warp yielded ${String(r.value)}`);
    }
  }
  const res = {
    imma: RUN.imma, stores: RUN.stores, reads: RUN.reads, wrong: RUN.wrong, twice: RUN.twice, outside: RUN.outside,
    misplaced: RUN.misplaced, bad: RUN.bad, gateChecks: RUN.gateChecks, ungated: RUN.ungated, memoHits: RUN.memoHits,
    readsOf: RUN.readsOf, blocks: [],
  };
  for (const blk of blocks) {
    if (blk.atBar.length) throw new Error(`block ${blk.b}: ${blk.atBar.length} warps left at a barrier`);
    const seq = [];
    for (const h of blk.hashCalls) {
      if (h.warp === 0) seq.push(h.tile);
    }
    res.blocks.push({
      b: blk.b, seq, barriers: blk.barriers, handoffs: blk.handoffs, hashCalls: blk.hashCalls, heads: blk.heads,
      storesOf: blk.storesOf, lastFold: blk.lastFold,
    });
  }
  RUN = null;
  CTX.warp = null;
  return res;
}

// At a block's barrier after epoch e (e barriers passed). From the second on, it is the
// hand-off of the loop's tile e - 1: its buffer, (e - 1) & 1, must hold one tile's 2048
// words, all of them stored. Kept: the tile and its words, buffer-relative.
function handoff(blk, e) {
  if (e === 0) return;
  const it = e - 1;
  const base = (it & 1) * 2048;
  const ws = blk.sT.arr.slice(base, base + 2048);
  const first = typeof ws[0] === 'string' ? ws[0].split('.')[0] : null;
  const tile = first !== null && /^\d+$/.test(first) ? Number(first) : null;
  const complete = tile !== null && ws.every((w) => typeof w === 'string' && w.startsWith(`${tile}.`));
  blk.handoffs.push({ it, tile, complete, words: ws });
}

// ---- Checking a run ----------------------------------------------------------------

// The faults a run counted: wrong reads, words stored twice or outside the buffers or
// at another chunk's place, wrong sums (bad), and C = 0 IMMA that do not wait on the
// fold of the value they overwrite.
const faults = (r) => ({
  wrong: r.wrong, twice: r.twice, outside: r.outside, misplaced: r.misplaced, bad: r.bad, ungated: r.ungated,
});
const CLEAN = { wrong: 0, twice: 0, outside: 0, misplaced: 0, bad: {}, ungated: 0 };

// A block's run, as checkable facts: the tiles every warp hashed (all the same?); the
// tile each hand-off's buffer held, and whether all 2048 of its words were that tile's;
// stores of a tile outside its buffer ((position & 1) * 2048, the loop's tbuf) and
// outside its epoch (after the barrier before it, before its hand-off); its chunk 15
// words; the tile each epoch's first IMMA (C = 0) summed; the last epoch with a fold.
function report(blk) {
  const perWarp = Array.from({ length: 8 }, (_, w) => blk.hashCalls.filter((h) => h.warp === w).map((h) => h.tile));
  let stray = 0;
  let late = 0;
  let chunk15 = 0;
  blk.seq.forEach((T, it) => {
    const base = (it & 1) * 2048;
    for (const st of blk.storesOf.get(T) || []) {
      if (st.i < base || st.i >= base + 2048) stray++;
      if (st.epoch !== it + 1) late++;
      if (st.c === 15) chunk15++;
    }
  });
  return {
    sameTiles: perWarp.every((x) => x.join() === blk.seq.join()),
    handoffs: blk.handoffs.map((h) => h.tile),
    complete: blk.handoffs.every((h) => h.complete),
    barriers: blk.barriers,
    storesPerTile: blk.seq.map((T) => (blk.storesOf.get(T) || []).length),
    stray,
    late,
    chunk15,
    heads: [...blk.heads].sort((x, y) => x[0] - y[0]).map(([e, ts]) => `${e}:${[...ts].join('/')}`),
    lastFold: blk.lastFold,
  };
}
// What report gives for a block that hashed seq, with LAST_RO or without. Epoch e is
// the code between barrier e and e + 1 (the prologue's is barrier 1): tile it runs in
// epoch it + 1 and is handed off at its end. With LAST_RO, epoch 0 starts the first
// tile, epoch it + 1 starts tile it + 1, and the last epoch's head sums the last tile
// again; without it, epoch it + 1 starts tile it.
function expected(seq, lastRo) {
  const n = seq.length;
  return {
    sameTiles: true,
    handoffs: seq,
    complete: true,
    barriers: n + 1,
    storesPerTile: seq.map(() => 2048),
    stray: 0,
    late: 0,
    chunk15: n * 128,
    heads: lastRo ? [`0:${seq[0]}`, ...seq.map((_, it) => `${it + 1}:${seq[Math.min(it + 1, n - 1)]}`)]
      : seq.map((T, it) => `${it + 1}:${T}`),
    lastFold: n,
  };
}
// Each hand-off's 2048 words (buffer-relative), by tile.
function snapshots(r) {
  const out = new Map();
  for (const blk of r.blocks) for (const h of blk.handoffs) out.set(h.tile, h.words);
  return out;
}
// Tiles whose words at the hand-off differ from ref's, and how many words differ.
function differ(r, ref) {
  const a = snapshots(r);
  const b = snapshots(ref);
  const out = [];
  for (const [T, ws] of a) {
    const want = b.get(T);
    if (!want) out.push(`${T}: no reference`);
    else {
      const n = ws.filter((w, i) => w !== want[i]).length;
      if (n) out.push(`${T}: ${n} words`);
    }
  }
  return out;
}
// The run's totals for these blocks, as the code should give them: 2048 reads and
// stores a tile, 128 IMMA a k-step (once a block: the other warps take kstep's result),
// 64 k-steps a tile and, with LAST_RO, two more a block (the last tile's head), and 128
// waits a head with a tile before it.
function totals(r) {
  return { reads: r.reads, stores: r.stores, imma: r.imma, gateChecks: r.gateChecks };
}
function expectedTotals(seqs, lastRo) {
  const n = seqs.reduce((x, s) => x + s.length, 0);
  return {
    reads: n * 2048, stores: n * 2048, imma: (n * 64 + (lastRo ? 2 * seqs.length : 0)) * 128,
    gateChecks: lastRo ? n * 128 : 0,
  };
}

// pearl_fold_bd.cuh with these edits, each of a text found once.
function edited(edits) {
  let t = FOLD;
  for (const [from, to] of edits) {
    expect([from, count(t, from)]).toEqual([from, 1]);
    t = t.split(from).join(to);
  }
  return t;
}

// ---- Tests --------------------------------------------------------------------------

describe('Turing last chunk read out in the next tile (PEARL_BD_LAST_RO)', () => {
  test('is off by default, builds on the default fold with LEAN\'s parts and HIT_COORDS, and the header refuses the rest', () => {
    const at0 = HEADER.indexOf('#ifndef PEARL_BD_LAST_RO' + NL);
    expect(at0).toBeGreaterThanOrEqual(0);
    expect(HEADER.slice(at0).split(NL).slice(0, 3)).toEqual(['#ifndef PEARL_BD_LAST_RO', '#define PEARL_BD_LAST_RO 0',
      '#endif']);
    expect(config('').value('PEARL_BD_LAST_RO')).toBe(0);
    expect(HEADER).toContain('#if PEARL_BD_LAST_RO != 0 && PEARL_BD_LAST_RO != 1' + NL
      + '#error "PEARL_BD_LAST_RO is 0 or 1"');
    // Off, no build has any of it.
    for (const spec of ['', OFF, 'SNAKE=1', 'HASH_RG=1', 'ROWPAIR=1', 'RED=1', 'PERSIST=0', 'A_DIRECT=0']) {
      expect([spec, /\btile_head\b/.test(built(spec).text)]).toEqual([spec, false]);
    }
    // On: the default fold with LEAN and HIT_COORDS (both on by default), and SNAKE,
    // HASH_RG and ROWPAIR on top; the diagnostic builds compile too.
    const c = config('LAST_RO=1');
    expect(['LEAN', 'HIT_COORDS', 'RUNBASE', 'TAIL_LDG', 'GATE', 'TADDR', 'RED', 'HASH_SPLIT', 'DEFAULT_FOLD']
      .map((x) => c.value('PEARL_BD_' + x))).toEqual([1, 1, 1, 1, 1, 0, 0, 0, 1]);
    // LEAN's parts given one by one build it too.
    for (const spec of ['LAST_RO=1', 'LAST_RO=1 LEAN=1 HIT_COORDS=1', 'LAST_RO=1 LEAN=0 RUNBASE=1 TAIL_LDG=1 GATE=1 HIT_COORDS=1',
      'LAST_RO=1 SNAKE=1', 'LAST_RO=1 HASH_RG=1', 'LAST_RO=1 ROWPAIR=1', 'LAST_RO=1 SNAKE=1 HASH_RG=1 ROWPAIR=1']) {
      expect([spec, config(spec).errors, fold(spec).errors]).toEqual([spec, [], []]);
      expect([spec, count(built(spec).text, 'tile_head('), count(built(spec).text, 'auto tile_head = [&](bool prev) {')])
        .toEqual([spec, 2, 1]);
    }
    for (const ablate of ['PEARL_ABLATE_READOUT', 'PEARL_ABLATE_TRANSCRIPT', 'PEARL_ABLATE_TRANSCRIPT_HASH',
      'PEARL_ABLATE_BLOAD', 'PEARL_ABLATE_BSHARE', 'PEARL_ABLATE_BHIT']) {
      expect([ablate, fold('LAST_RO=1', ablate).errors]).toEqual([ablate, []]);
    }
    const refused = (spec, msg) => {
      const errors = config(spec).errors;
      expect([spec, errors.length > 0]).toEqual([spec, true]);
      expect(errors.join(NL)).toContain(msg);
    };
    for (const v of ['2', '3']) refused(`LAST_RO=${v}`, 'PEARL_BD_LAST_RO is 0 or 1');
    // Any fold but the default, and the default without one of LEAN's parts or
    // HIT_COORDS.
    for (const s of [OFF, 'LEAN=0', 'HIT_COORDS=0', 'RUNBASE=0', 'TAIL_LDG=0', 'GATE=2', 'PERSIST=0',
      'A_REALIGN=1', 'KORDER=0', 'A_DIRECT=0', 'GROUPS=1', 'LEAN=0 RUNBASE=1 TAIL_LDG=1 GATE=1 HIT_COORDS=0']) {
      refused(`LAST_RO=1 ${s}`, 'PEARL_BD_LAST_RO is written for the default fold with LEAN and HIT_COORDS');
    }
    // RED, TADDR and the split hash store the readout their own way, or change the
    // hand-off.
    for (const s of ['RED=1 GATE=1', 'TADDR=1', 'HASH_SPLIT=4 RUNBASE=1 TAIL_LDG=1']) {
      refused(`LAST_RO=1 ${s}`, 'PEARL_BD_LAST_RO is not built with PEARL_BD_RED, _TADDR or the split hash');
    }
  });

  test('the moved code: tile() runs k-steps 2-63, tile_head k-steps 0-1 and the last chunk\'s readout, then the hand-off', () => {
    // tile(): with the switch, no zeroing, no k-step 0 or 1 and no readout of the last
    // chunk; the chunk loop gathers before k-step 1 and stores after it.
    expect(FOLD).toContain('  auto tile = [&](auto next_srcs) {' + NL + '#if PEARL_BD_LAST_RO' + NL);
    expect(FOLD).toContain('      kstep(chunk + 1u, 1u, false, false, 1u);' + NL
      + '#if PEARL_BD_LAST_RO && !PEARL_BD_ROWPAIR && !defined(PEARL_ABLATE_READOUT)' + NL);
    expect(FOLD).toContain('#pragma unroll' + NL + '      for (uint32_t rl = 0; rl < RPL; rl++) readout_st(rl, chunk, rx[rl]);'
      + NL + '#endif' + NL + '    }' + NL + '    next_srcs();');
    expect(FOLD).toContain('    kstep(chunks - 1u, 3u, false, true);' + NL + '#if !PEARL_BD_LAST_RO' + NL);
    const f = built('LAST_RO=1');
    const tile = flat(block(f.text, 'auto tile = [&](auto next_srcs) {'));
    expect(tile).toBe('auto tile = [&](auto next_srcs) { constexpr uint32_t first_chunk = 0u; '
      + 'for (uint32_t chunk = first_chunk; chunk + 1u < chunks; chunk++) { '
      + 'arun = aoff + chunk * (KSTEPS * 512u); brun = boff + chunk * (KSTEPS * 512u); '
      + 'asm volatile("" : "+r"(arun), "+r"(brun)); '
      + 'kstep(chunk, 2u, false, false); kstep(chunk, 3u, false, false); kstep(chunk + 1u, 0u, RO, false, 1u); '
      + 'uint32_t rx[RPL]; #pragma unroll for (uint32_t rl = 0; rl < RPL; rl++) rx[rl] = readout_gather(rxs[rl]); '
      + 'kstep(chunk + 1u, 1u, false, false, 1u); '
      + '#pragma unroll for (uint32_t rl = 0; rl < RPL; rl++) readout_st(rl, chunk, rx[rl]); } '
      + 'next_srcs(); arun = aoff + (chunks - 1u) * (KSTEPS * 512u); brun = boff + (chunks - 1u) * (KSTEPS * 512u); '
      + 'kstep(chunks - 1u, 2u, false, false); kstep(chunks - 1u, 3u, false, true); }');

    // tile_head, line for line: with a tile before it, each B pair's fold of both
    // regions, then the pair's two k-lo B words gated on it (the gating lines); the
    // zeroing; k-step 0 from the tile's own offsets; the gather and the stores of chunk
    // 15; k-step 1.
    const HEAD = [
      '  auto tile_head = [&](bool prev) {',
      '    uint32_t rx[RPL] = {0u, 0u};',
      '    if (prev) {',
      '#pragma unroll',
      '      for (uint32_t j = 0; j < 4u; j++) {',
      '#pragma unroll',
      '        for (uint32_t rl = 0; rl < RPL; rl++) rx[rl] = fold_pair(rx[rl], rl, j);',
      '        const uint32_t g = rx[0] ^ rx[1];',
      '        int4 &q = bf[0][(j / 2u) * 2u];',
      '        int &b0 = j & 1u ? q.z : q.x, &b2 = j & 1u ? q.w : q.y;',
      '        b0 = (int)pearl_bd_after((uint32_t)b0, g);',
      '        b2 = (int)pearl_bd_after((uint32_t)b2, g);',
      '      }',
      '    }',
      '#pragma unroll',
      '    for (uint32_t mb = 0; mb < MB; mb++)',
      '#pragma unroll',
      '      for (uint32_t nb = 0; nb < NB; nb++)',
      '#pragma unroll',
      '        for (uint32_t i = 0; i < 4; i++) acc[mb][nb][i] = 0;',
      '    arun = aoff;',
      '    brun = boff;',
      '    kstep(0u, 0u, false, false);',
      '    if (prev) {',
      '#pragma unroll',
      '      for (uint32_t rl = 0; rl < RPL; rl++) rx[rl] = readout_gather(rx[rl]);',
      '#pragma unroll',
      '      for (uint32_t rl = 0; rl < RPL; rl++) readout_st(rl, chunks - 1u, rx[rl]);',
      '    }',
      '    kstep(0u, 1u, false, false);',
      '  };',
    ];
    const lines = f.text.split(NL);
    const h0 = lines.indexOf(HEAD[0]);
    expect(h0).toBeGreaterThanOrEqual(0);
    expect(lines.slice(h0, h0 + HEAD.length)).toEqual(HEAD);
    for (const line of HEAD.slice(6, 12)) expect(FOLD).toContain(line + NL);
    // The gated words are the ones k-step 0's k-lo pass reads for pair j (b_word with
    // cur 0, hi 0: odd 0 is b0, odd 1 is b2).
    const ks = flat(block(f.text, 'auto kstep = [&](uint32_t ch, uint32_t t, bool ro, bool last, uint32_t dch = 0u) {'));
    expect(ks).toContain('const uint32_t cur = t & 1u, nxt = (t + 1u) & 1u;');
    expect(ks).toContain('auto b_word = [&](uint32_t j, uint32_t hi, uint32_t odd) -> uint32_t { '
      + 'const int4 v = bf[cur][(j / 2u) * 2u + hi]; return (uint32_t)(j & 1u ? (odd ? v.w : v.z) : (odd ? v.y : v.x)); };');
    expect(ks).toContain('b0[rl] = b_word(j, 0u, 0u); b2[rl] = b_word(j, 0u, 1u);');

    // The first tile's head before the prologue's barrier; each later one after the tile
    // before it, from next_srcs's offsets, then the hand-off and the hash. tbuf moves on
    // only at the loop's top.
    expect(FOLD).toContain('    tile_prologue();' + NL + '  }' + NL + '#if PEARL_BD_LAST_RO' + NL);
    // PEARL_BD_TALL's first write of sCur sits between them, in its own #if.
    expect(FOLD).toContain('  tile_head(false);' + NL + '#endif' + NL + '#if PEARL_BD_TALL' + NL);
    expect(FOLD).toContain('#endif' + NL + '  __syncthreads();' + NL + '  uint32_t it = 0u;');
    expect(FOLD).toContain('      boffn = b_off(cbgn);' + NL + '    });' + NL + '#if PEARL_BD_LAST_RO' + NL);
    expect(FOLD).toContain('    aoff = aoffn;' + NL + '    boff = boffn;' + NL + '    tile_head(true);' + NL + '#endif' + NL
      + '#if PEARL_BD_HASH_SPLIT' + NL);
    expect(f.flat).toContain('tile_prologue(); } tile_head(false); __syncthreads(); uint32_t it = 0u; '
      + 'for (uint32_t v = blockIdx.x; v < tiles; it++) { tbuf = (it & 1u) * TWORDS; uint32_t vgrab = 0u; '
      + 'if (threadIdx.x == 0u) vgrab = gridDim.x + atomicAdd(tile_ctr, 1u); uint32_t vnext = 0u; '
      + 'tile([&]() { vnext = sNext[it & 1u]; uint32_t rbgn, cbgn; tile_coords(vnext < tiles ? vnext : v, rbgn, cbgn); '
      + 'aoffn = a_off(rbgn); boffn = b_off(cbgn); }); aoff = aoffn; boff = boffn; tile_head(true); '
      + 'if (threadIdx.x == 0u) sNext[(it + 1u) & 1u] = vgrab; __syncthreads(); hash_tile(v); '
      + 'aoff = aoffn; boff = boffn; v = vnext; } }');
    // With ROWPAIR the head gathers the xor of the two folds once and stores one word.
    expect(flat(block(built('LAST_RO=1 ROWPAIR=1').text, 'auto tile_head = [&](bool prev) {')))
      .toContain('kstep(0u, 0u, false, false); if (prev) { readout_st(chunks - 1u, readout_gather(rx[0] ^ rx[1])); } '
        + 'kstep(0u, 1u, false, false); }');
  });

  test('the simulation runs the fold\'s own code, and stands in only for what it names', () => {
    // The builtins it runs in their place. pearl_bd_after is PRMT 0x3210: b's bytes, so
    // b unchanged, once x is there.
    expect(FOLD).toContain('asm("prmt.b32 %0, %1, %2, 0x3210;" : "=r"(r) : "r"(b), "r"(x));');
    const rand = xorshift(3);
    for (let i = 0; i < 200; i++) {
      const [b, x] = [rand(), rand()];
      expect(prmt(b, x, 0x3210)).toBe(b);
    }
    // pearl_xor3 is LOP3 0x96, a ^ b ^ c; pearl_bd_imma one m8n8k16 into its two
    // accumulators (read, then written); the loads one LDG.128 each.
    expect(KERNEL).toContain('asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(d) : "r"(a), "r"(b), "r"(c));');
    expect(FOLD).toContain('asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"'
      + NL + '               : "+r"(c0), "+r"(c1)' + NL + '               : "r"(a), "r"(b));');
    expect(FOLD).toContain('asm volatile("ld.global.nc.v4.s32 {%0,%1,%2,%3}, [%4];"');
    expect(FOLD).toContain('asm volatile("ld.global.v4.s32 {%0,%1,%2,%3}, [%4];"');
    // The lambdas it stands in for: a tile's coordinates and offsets are a function of
    // the tile alone, one tile's differ from another's, and the hash reads 16 words a
    // region where hash_tile does.
    const f = built('LAST_RO=1');
    expect(f.flat).toContain('auto a_off = [&](uint32_t rbg_) -> uint32_t { return (rbg_ * 8u + wr * 4u) * FRAG_BLOCK + lane * 16u; };');
    expect(f.flat).toContain('auto b_off = [&](uint32_t cbg_) -> uint32_t { return (col_off + cbg_ * 16u + wc * 4u) * FRAG_BLOCK + lane * 16u; };');
    expect(f.flat).toContain('cbg_ = band_first + in_band % band_cols; rbg_ = in_band / band_cols; };');
    const prog = programOf('LAST_RO=1');
    expect(prog.ov.exprs).toEqual({ hasher: 'wr ? sid >= 64u : sid < 64u', L: 'wr * 64u + (sid & 63u)',
      read: 'tbuf + L * 16u + (c ^ ((L >> 1) & 15u))' });
    // It skips only static_asserts.
    expect(prog.skipped.length).toBeGreaterThan(0);
    expect(prog.skipped.every((x) => x.startsWith('static_assert('))).toBe(true);
    // The geometry its tags assume: 16 chunks, two buffers of 2048 words.
    expect(f.flat).toContain('constexpr uint32_t TWORDS = (PEARL_FOLD_THREADS / 32u) * warp_regions * PEARL_JACKPOT_BUCKETS;');
    expect(['PEARL_FOLD_K', 'PEARL_FOLD_RANK', 'PEARL_FOLD_CHUNKS', 'PEARL_JACKPOT_BUCKETS'].map((x) => prog.global(x)))
      .toEqual([2048, 128, 16, 16]);
    // kstep's memo: of the fold's variables, kstep reads and writes only af, bf, acc and
    // rxs, and reads the offsets; the other names it reads are constants (its calls are
    // to its own lambdas, fold_pair and builtins).
    for (const spec of ['LAST_RO=1', 'LAST_RO=1 SNAKE=1', 'LAST_RO=1 ROWPAIR=1', '', 'ROWPAIR=1']) {
      const ks = flat(block(built(spec).text, 'auto kstep = [&](uint32_t ch, uint32_t t, bool ro, bool last, uint32_t dch = 0u) {'));
      const local = new Set(['ch', 't', 'ro', 'last', 'dch']);
      for (const m of ks.matchAll(/\b(?:uint32_t|int4|int|bool) (\w+)/g)) local.add(m[1]);
      for (const m of ks.matchAll(/, (\w+) = /g)) local.add(m[1]);
      for (const m of ks.matchAll(/auto (\w+) = \[&\]\(([^)]*)\)/g)) {
        local.add(m[1]);
        for (const p of m[2].split(',')) local.add(p.trim().split(/[\s*&]+/).pop());
      }
      for (const m of ks.matchAll(/uint32_t \w+\[[^\]]*\], (\w+)\[/g)) local.add(m[1]);
      const free = new Set();
      for (const m of ks.slice(ks.indexOf('{')).matchAll(/(?<![.\w])([A-Za-z_]\w*)\b(?!\s*\()/g)) {
        const id = m[1];
        if (local.has(id) || TYPES.has(id) || ['int4', 'const', 'auto', 'return', 'pragma', 'unroll'].includes(id)
          || f.defs.has(id)) continue;
        free.add(id);
      }
      expect([spec, [...free].sort()]).toEqual([spec, ['Aprime', 'Bprime', 'FRAG_BLOCK', 'KSTEPS', 'MB', 'RPL', 'acc',
        'af', 'aoffn', 'arun', 'bf', 'boffn', 'brun', 'rxs']]);
      for (const k of ['FRAG_BLOCK', 'KSTEPS', 'MB', 'RPL']) expect(built(spec).flat).toMatch(new RegExp(`constexpr uint32_t [^;]*\\b${k} = `));
    }
  });

  test('one block of 1, 2, 3 and 6 tiles, warps at random speeds: each tile\'s words are all stored before its hand-off, in its own buffer, and are the words the fold without the switch stores', () => {
    // The fold without the switch: the words every LAST_RO run is held to, tile by tile.
    const ref = runFold('', { tiles: 6, seed: 100 });
    expect(faults(ref)).toEqual(CLEAN);
    expect(ref.blocks.map(report)).toEqual([expected([0, 1, 2, 3, 4, 5], false)]);
    expect(totals(ref)).toEqual(expectedTotals([[0, 1, 2, 3, 4, 5]], false));
    for (const [n, seeds] of [[1, [1, 2]], [2, [3]], [3, [4]], [6, [5]]]) {
      for (const seed of seeds) {
        const r = runFold('LAST_RO=1', { tiles: n, seed });
        const seq = Array.from({ length: n }, (_, i) => i);
        expect([n, seed, faults(r)]).toEqual([n, seed, CLEAN]);
        expect([n, seed, r.blocks.map(report)]).toEqual([n, seed, [expected(seq, true)]]);
        expect([n, seed, totals(r)]).toEqual([n, seed, expectedTotals([seq], true)]);
        expect([n, seed, differ(r, ref)]).toEqual([n, seed, []]);
        expect([...r.readsOf.values()]).toEqual(seq.map(() => 2048));
      }
    }
  }, 60000);

  test('tile_head folds chunk 15 out of the tile\'s own accumulators, and each first IMMA on a pair waits on that fold', () => {
    // In the runs above every chunk 15 word is a whole region of its own tile's k-step
    // 63, and all 128 accumulators of each head wait, through their B word, on the fold
    // that read the value they overwrite. Edits that break one or the other, on one tile
    // (its head, the dropped one, has the tile to read out):
    const run = (edits, n = 1) => {
      const r = runFold('LAST_RO=1', { tiles: n, seed: 31, text: edited(edits) });
      return { ...faults(r), gateChecks: r.gateChecks };
    };
    // The odd n-block's word left ungated: its 64 accumulators a head do not wait.
    expect(run([['        b2 = (int)pearl_bd_after((uint32_t)b2, g);' + NL, '']]))
      .toEqual({ ...CLEAN, ungated: 64, gateChecks: 128 });
    // Gated on region 0's fold only: region 1's 64 do not.
    expect(run([['const uint32_t g = rx[0] ^ rx[1];', 'const uint32_t g = rx[0];']]))
      .toEqual({ ...CLEAN, ungated: 64, gateChecks: 128 });
    // The other buffer's words gated, which k-step 0 does not read: none waits.
    expect(run([['int4 &q = bf[0][(j / 2u) * 2u];', 'int4 &q = bf[1][(j / 2u) * 2u];']]))
      .toEqual({ ...CLEAN, ungated: 128, gateChecks: 128 });
    // The accumulators zeroed before the fold: chunk 15's 128 words a tile are nothing.
    const zeroFirst = run([['    uint32_t rx[RPL] = {0u, 0u};' + NL + '    if (prev) {',
      '    uint32_t rx[RPL] = {0u, 0u};' + NL + '    for (uint32_t mb = 0; mb < MB; mb++)' + NL
      + '      for (uint32_t nb = 0; nb < NB; nb++)' + NL + '        for (uint32_t i = 0; i < 4; i++) acc[mb][nb][i] = 0;'
      + NL + '    if (prev) {']]);
    expect(zeroFirst.bad).toEqual({ 'store: nothing': 128 });
    expect(zeroFirst.wrong).toBeGreaterThan(0);
    // Folded after k-step 0 has overwritten them: the words are k-step 0's, not the
    // last chunk's, and no IMMA had a fold to wait on.
    const late = run([['    uint32_t rx[RPL] = {0u, 0u};' + NL + '    if (prev) {', '    uint32_t rx[RPL] = {0u, 0u};' + NL
      + '    if (false) {'], ['    kstep(0u, 0u, false, false);' + NL + '    if (prev) {' + NL,
      '    kstep(0u, 0u, false, false);' + NL + '    if (prev) {' + NL + '      for (uint32_t j = 0; j < 4u; j++)' + NL
      + '        for (uint32_t rl = 0; rl < RPL; rl++) rx[rl] = fold_pair(rx[rl], rl, j);' + NL]]);
    expect(late.bad).toEqual({ 'store: k-step 0 half 1, not a chunk\'s end': 128 });
    expect([late.gateChecks, late.ungated]).toEqual([0, 0]);
    expect(late.wrong).toBeGreaterThan(0);
  }, 60000);

  test('the moved stores go to the finished tile\'s buffer before its hand-off, and the last tile runs its head over its own operands and hands off', () => {
    // From the runs above: every store of tile it is in buffer it & 1, chunk 15's in
    // epoch it + 1, before hand-off it; the last epoch's head sums the last tile's own
    // k-steps 0-1 and nothing folds them. Here the same for a block whose tiles are not
    // consecutive (the grid's other blocks take the counter's other values).
    // The prologue's grab and one a tile: tiles 2, 4 + 5, 4 + 17, 4 + 30, then 4 + 36 is
    // past the last.
    const grabs = [5, 17, 30, 36, 39];
    const grab = () => grabs.shift();
    const r = runFold('LAST_RO=1', { G: 4, only: [2], tiles: 40, seed: 12, grab });
    expect(faults(r)).toEqual(CLEAN);
    expect(r.blocks.map(report)).toEqual([expected([2, 9, 21, 34], true)]);
    expect(totals(r)).toEqual(expectedTotals([[2, 9, 21, 34]], true));
    const ref = runFold('', { G: 4, only: [2], tiles: 40, seed: 13, grab: ((g) => () => g.shift())([5, 17, 30, 36, 39]) });
    expect(faults(ref)).toEqual(CLEAN);
    expect(differ(r, ref)).toEqual([]);

    const run = (edits, n = 1) => {
      const x = runFold('LAST_RO=1', { tiles: n, seed: 41, text: edited(edits) });
      return { ...faults(x), report: report(x.blocks[0]) };
    };
    // The buffer moved on before the head: chunk 15 lands in the next tile's buffer, so
    // the hand-off is missing it and the hashers read old words.
    const flip = run([['    boff = boffn;' + NL + '    tile_head(true);',
      '    boff = boffn;' + NL + '    tbuf = ((it + 1u) & 1u) * TWORDS;' + NL + '    tile_head(true);']]);
    expect(flip.report).toMatchObject({ complete: false, stray: 128, late: 0, chunk15: 128 });
    expect(flip.wrong).toBeGreaterThan(0);
    // The head after the hand-off: chunk 15 is stored while the hashers may read. They
    // run their own head first, so they mostly read in time: the race shows at the
    // hand-off, whose buffer is missing the words.
    const after = run([['    boff = boffn;' + NL + '    tile_head(true);' + NL + '#endif', '    boff = boffn;' + NL + '#endif'],
      ['    __syncthreads();' + NL + '#if PEARL_BD_HIT_COORDS' + NL + '    hash_tile(v);',
        '    __syncthreads();' + NL + '#if PEARL_BD_LAST_RO' + NL + '    tile_head(true);' + NL + '#endif' + NL
        + '#if PEARL_BD_HIT_COORDS' + NL + '    hash_tile(v);']]);
    expect(after.report).toMatchObject({ complete: false, stray: 0, late: 128, chunk15: 128 });
    // The head stores chunk 15 at chunk 14's place.
    const c14 = run([['readout_st(rl, chunks - 1u, rx[rl]);', 'readout_st(rl, chunks - 2u, rx[rl]);']]);
    expect(c14.misplaced).toBe(128);
    expect(c14.report.complete).toBe(false);
    // The head's two words swapped between the region halves: each is still a whole
    // region of the tile's chunk 15, in place, but not the region stored there without
    // the switch.
    const swap = runFold('LAST_RO=1', { tiles: 1, seed: 42,
      text: edited([['readout_st(rl, chunks - 1u, rx[rl]);', 'readout_st(rl, chunks - 1u, rx[rl ^ 1u]);']]) });
    expect(faults(swap)).toEqual(CLEAN);
    expect(report(swap.blocks[0]).complete).toBe(true);
    expect(differ(swap, runFold('', { tiles: 1, seed: 43 }))).toEqual(['0: 128 words']);
    // The head on the finished tile's offsets: k-step 1 sums the old tile's words.
    const own = run([['    aoff = aoffn;' + NL + '    boff = boffn;' + NL + '    tile_head(true);', '    tile_head(true);']], 2);
    expect(own.bad['an IMMA out of order']).toBeGreaterThan(0);
    // No head for the first tile: its k-steps 0-1 never run.
    const first = run([['  tile_head(false);' + NL, '']]);
    expect(first.bad['an IMMA on garbage']).toBeGreaterThan(0);
    // next_srcs without its clamp: the last tile's tail and head load past the last tile.
    const clamp = run([['      tile_coords(vnext < tiles ? vnext : v, rbgn, cbgn);', '      tile_coords(vnext, rbgn, cbgn);']]);
    expect(clamp.bad['a load past the last tile']).toBeGreaterThan(0);
    expect(clamp.report.heads[clamp.report.heads.length - 1]).toBe('1:1');
  }, 60000);

  test('a grid of blocks sharing the tile counter: every tile is hashed once, by one block, with the words the fold without the switch stores', () => {
    for (const [G, tiles, seed] of [[3, 7, 51]]) {
      const r = runFold('LAST_RO=1', { G, tiles, seed });
      const ref = runFold('', { G, tiles, seed: seed + 1000 });
      for (const x of [r, ref]) {
        expect(faults(x)).toEqual(CLEAN);
        const seqs = x.blocks.map((b) => b.seq);
        expect(seqs.flat().sort((a, b) => a - b)).toEqual(Array.from({ length: tiles }, (_, i) => i));
        expect(x.blocks.map(report)).toEqual(seqs.map((s) => expected(s, x === r)));
        expect(totals(x)).toEqual(expectedTotals(seqs, x === r));
      }
      expect(differ(r, ref)).toEqual([]);
      // Each block starts on its own tile, b.
      expect(r.blocks.map((b) => b.seq[0])).toEqual(Array.from({ length: G }, (_, b) => b));
    }
  }, 60000);

  test('with SNAKE, HASH_RG and ROWPAIR on top, the words are those of the same build without LAST_RO', () => {
    for (const [spec, base, n, seed] of [['LAST_RO=1 SNAKE=1', '', 2, 61], ['LAST_RO=1 HASH_RG=1', 'HASH_RG=1', 2, 62],
      ['LAST_RO=1 ROWPAIR=1', 'ROWPAIR=1', 2, 63], ['LAST_RO=1 SNAKE=1 HASH_RG=1 ROWPAIR=1', 'ROWPAIR=1', 2, 64]]) {
      const seq = Array.from({ length: n }, (_, i) => i);
      const r = runFold(spec, { tiles: n, seed });
      const ref = runFold(base, { tiles: n, seed: seed + 1000 });
      for (const [x, lastRo] of [[r, true], [ref, false]]) {
        expect([spec, lastRo, faults(x)]).toEqual([spec, lastRo, CLEAN]);
        expect([spec, lastRo, x.blocks.map(report)]).toEqual([spec, lastRo, [expected(seq, lastRo)]]);
        expect([spec, lastRo, totals(x)]).toEqual([spec, lastRo, expectedTotals([seq], lastRo)]);
      }
      expect([spec, differ(r, ref)]).toEqual([spec, []]);
    }
    // HASH_RG's hashers are warps 4-7, the default's 0, 1, 6 and 7, one lane a region.
    const hashers = (spec) => {
      const r = runFold(spec, { tiles: 1, seed: 65 });
      return r.blocks[0].hashCalls.filter((h) => h.lanes !== 0).map((h) => h.warp).sort();
    };
    expect(hashers('LAST_RO=1')).toEqual([0, 1, 6, 7]);
    expect(hashers('LAST_RO=1 HASH_RG=1')).toEqual([4, 5, 6, 7]);
  }, 60000);
});
