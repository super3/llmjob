'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

// Jest runs a test file in a vm context, where every read of a global goes through the
// context. Bound here once, the launch walks below run several times faster.
const { Set, Map, Math, Array, Uint8Array, Object, Number, JSON } = globalThis;

// Turing's tall warp tiles (PEARL_BD_TALL, off by default). Each warp holds 96x64 instead
// of 64x64, so a tile is 192x256 and a row group 12 row offsets. A k-step's 192 IMMA run in
// three sub-passes, one A quad (m-blocks 2q and 2q + 1) each, and the next quads load into
// the registers a sub-pass has finished with. The tile is a loop over chunks. The last row
// group runs past m into A's zeroed padding, and hash_tile reports no region from it. 192
// regions a tile go to two 12 KB transcript buffers; the default's four hashers hash 48
// each, in two passes. The persistent loop keeps the tile in shared (sCur), not a register.
//
// Nothing here can run the device code. A small C preprocessor picks out the code a build
// with given switches compiles, and the code's expressions are read out of it and run on
// the CPU. Where the code is too long to run that way (the k-step, the tile and the
// hand-off), it is copied here and the copy is pinned to the build's lines. The checks:
// - the default, the guards, and every other build without it line for line as it was
//   before the switch (d2e61d4);
// - each of a tile's 192 regions hashed once, 48 by each hasher warp;
// - each region's 16 words stored where its hasher reads them, in 24 KB;
// - over whole launches, the shapes compilers.sh's complete stage runs among them, every
//   region numbered once, as the default numbers it, and none from the padding rows;
// - every load inside the padded A' and B' the host allocates;
// - every IMMA's A and B words, from A' and B' as the host draws them (12-block row groups
//   of turingAdirect and turingKorder);
// - the operands' rotation through k-steps, chunks and tiles, and each readout's sums;
// - the hand-off, in random and adversarial warp orders;
// - the host's tile count, shared size and refusals, and the bench's fold name;
// - the host's and the kernel's host code the same without the switch, and the pairing
//   that keeps a host and a fold built with and without it from linking together.

const SRC = path.join(__dirname, '..', 'native', 'src');
const NL = String.fromCharCode(10);
const BS = String.fromCharCode(92);
// Line endings normalised: a Windows checkout has CRLF.
const read = (name) => fs.readFileSync(path.join(SRC, name), 'utf8')
  .split(String.fromCharCode(13) + NL).join(NL);
const FOLD = read('pearl_fold_bd.cuh');
const HEADER = read('pearl_config.h');
const HOST = read('pearl_host.cu');
const KERNEL = read('pearl_kernel.cu');
const COMPILERS = fs.readFileSync(path.join(__dirname, '..', 'native', 'probes', 'turing', 'compilers.sh'), 'utf8');

// ---- C expressions (turingHashRg.test.js's) ----------------------------------------

const TOKEN = /\s*(?:(0[xX][0-9a-fA-F]+|[0-9]+)[uUlL]*|([A-Za-z_]\w*(?:\.\w+)*)|(&&|\|\||==|!=|<=|>=|<<|>>|[-+*/%<>!~&|^?:()]))/y;
const PREC = {
  '||': 1, '&&': 2, '|': 3, '^': 4, '&': 5, '==': 6, '!=': 6, '<': 7, '<=': 7, '>': 7, '>=': 7,
  '<<': 8, '>>': 8, '+': 9, '-': 9, '*': 10, '/': 10, '%': 10,
};
const b01 = (x) => (x ? 1 : 0);
const OPS = {
  '||': (a, b) => b01(a || b),
  '&&': (a, b) => b01(a && b),
  '==': (a, b) => b01(a === b),
  '!=': (a, b) => b01(a !== b),
  '<': (a, b) => b01(a < b),
  '<=': (a, b) => b01(a <= b),
  '>': (a, b) => b01(a > b),
  '>=': (a, b) => b01(a >= b),
};
const INT = {
  ...OPS,
  '|': (a, b) => a | b, '^': (a, b) => a ^ b, '&': (a, b) => a & b, '<<': (a, b) => a << b,
  '>>': (a, b) => a >> b, '+': (a, b) => a + b, '-': (a, b) => a - b, '*': (a, b) => a * b,
  '/': (a, b) => Math.trunc(a / b), '%': (a, b) => a % b,
};
const U32 = {
  ...OPS,
  '|': (a, b) => (a | b) >>> 0, '^': (a, b) => (a ^ b) >>> 0, '&': (a, b) => (a & b) >>> 0,
  '<<': (a, b) => (a << b) >>> 0, '>>': (a, b) => a >>> b, '+': (a, b) => (a + b) >>> 0,
  '-': (a, b) => (a - b) >>> 0, '*': (a, b) => Math.imul(a, b) >>> 0,
  '/': (a, b) => Math.floor(a / b), '%': (a, b) => a % b,
};

function tokens(text) {
  const out = [];
  const s = text.trim();
  TOKEN.lastIndex = 0;
  while (TOKEN.lastIndex < s.length) {
    const at = TOKEN.lastIndex;
    const m = TOKEN.exec(s);
    if (!m) throw new Error(`cannot read "${s.slice(at)}" in "${s}"`);
    if (m[1] !== undefined) out.push({ num: Number(m[1]) });
    else if (m[2] !== undefined) out.push({ id: m[2] });
    else out.push({ op: m[3] });
  }
  return out;
}

// A C expression as a function of a scope ({ value(name), defined(name) }). u32: every
// result a uint32_t, as code computes it; otherwise the preprocessor's integers.
const compiled = new Map();
function compile(text, u32) {
  const key = (u32 ? 'u ' : 'i ') + text;
  if (compiled.has(key)) return compiled.get(key);
  const ts = tokens(text);
  const ops = u32 ? U32 : INT;
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
      return (s) => b01(!e(s));
    }
    if (t.op === '-' || t.op === '~') {
      const e = unary();
      if (t.op === '-') return u32 ? (s) => (-e(s)) >>> 0 : (s) => -e(s);
      return u32 ? (s) => (~e(s)) >>> 0 : (s) => ~e(s);
    }
    if (t.id === 'defined') {
      const paren = ts[i] && ts[i].op === '(';
      if (paren) i++;
      const name = ts[i++].id;
      if (paren) take(')');
      return (s) => b01(s.defined(name));
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
      const f = ops[op];
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
  compiled.set(key, e);
  return e;
}

function macroScope(defs, seen = []) {
  return {
    defined: (n) => defs.has(n),
    value: (n) => {
      const body = defs.get(n);
      if (body === undefined || seen.includes(n)) return 0;
      if (body === null) throw new Error(`function-like ${n} in #if`);
      return compile(body, false)(macroScope(defs, [...seen, n]));
    },
  };
}

// Code's names: env's, then object-like macros. Any other name is an error.
function codeScope(env, defs = new Map()) {
  return {
    defined: (n) => defs.has(n),
    value: (n) => {
      if (Object.prototype.hasOwnProperty.call(env, n)) return env[n];
      if (n === 'true') return 1;
      if (n === 'false') return 0;
      const body = defs.get(n);
      if (typeof body === 'string') return compile(body, true)(codeScope({}, defs));
      throw new Error(`unknown name ${n}`);
    },
  };
}
const run = (expr, env, defs) => compile(expr, true)(codeScope(env, defs));

// ---- A small C preprocessor (turingHashRg.test.js's) --------------------------------

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

function preprocess(text, given) {
  const defs = new Map(given);
  const lines = [];
  const errors = [];
  const stack = [];
  const live = () => stack.every((f) => f.on);
  const raw = text.split(NL);
  for (let n = 0; n < raw.length; n++) {
    let line = raw[n];
    if (/^\s*#/.test(line)) {
      while (line.endsWith(BS) && n + 1 < raw.length) line = line.slice(0, -1) + ' ' + raw[++n];
    }
    const m = /^\s*#\s*(\w+)(.*)$/.exec(line);
    if (!m) {
      const code = uncomment(line).trim();
      if (live() && code !== '') lines.push(code);
      continue;
    }
    const dir = m[1];
    const rest = uncomment(m[2]).trim();
    const top = stack[stack.length - 1];
    if (dir === 'if' || dir === 'ifdef' || dir === 'ifndef') {
      const outer = live();
      let on = false;
      if (outer && dir === 'if') on = compile(rest, false)(macroScope(defs)) !== 0;
      else if (outer) on = defs.has(rest) === (dir === 'ifdef');
      stack.push({ on, taken: on, outer });
    } else if (dir === 'elif') {
      top.on = top.outer && !top.taken && compile(rest, false)(macroScope(defs)) !== 0;
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
      defs.set(name, after.startsWith('(') ? null : after.trim());
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

// The lines a file keeps with PEARL_BD_TALL set to tall, every other #if left as it is,
// both its sides kept. For the host and the kernel's host code, which preprocess() cannot
// read whole. A directive that names the switch must be '#if PEARL_BD_TALL' itself.
function pickTall(text, tall) {
  const stack = [];
  const out = [];
  for (const line of text.split(NL)) {
    const m = /^\s*#\s*(\w+)\s*(.*)$/.exec(line);
    const dir = m && m[1];
    if (m && /\bPEARL_BD_TALL\b/.test(m[2])) {
      expect([line, dir === 'if' && uncomment(m[2]).trim() === 'PEARL_BD_TALL']).toEqual([line, true]);
      stack.push({ ours: true, on: tall === 1 });
      continue;
    }
    if (dir === 'if' || dir === 'ifdef' || dir === 'ifndef') stack.push({ ours: false });
    const top = stack[stack.length - 1];
    if (top && top.ours && dir === 'elif') throw new Error('#elif after #if PEARL_BD_TALL');
    if (top && top.ours && dir === 'else') {
      top.on = !top.on;
      continue;
    }
    if (dir === 'endif') {
      stack.pop();
      if (top.ours) continue;
    }
    if (stack.every((f) => !f.ours || f.on)) out.push(line);
  }
  expect(stack).toEqual([]);
  return out.join(NL);
}

// -D flags, PEARL_BD_ ones by their short names: D('TALL=1'). A bare name is 1.
function D(spec) {
  const defs = new Map();
  for (const f of spec.split(' ').filter(Boolean)) {
    const [name, v = '1'] = f.split('=');
    defs.set(name.startsWith('PEARL_') ? name : 'PEARL_BD_' + name, v);
  }
  return defs;
}
const OFF = 'LEAN=0 HIT_COORDS=0';
const T = (base) => `${base} TALL=1`.trim();

function slice(src, from, to) {
  const at = src.indexOf(from);
  expect([from, at >= 0]).toEqual([from, true]);
  const end = src.indexOf(to, at);
  expect([to, end > at]).toEqual([to, true]);
  const eol = src.indexOf(NL, end);
  return src.slice(src.lastIndexOf(NL, at) + 1, eol < 0 ? src.length : eol);
}

// The guards a build of the fold goes through: pearl_config.h's Turing switches and the
// fold's own #error lines, TALL's last.
const GUARDS = slice(HEADER, '#ifndef PEARL_TURING_WIDE' + NL, '#define PEARL_BD_SPLIT_SMEM ') + NL
  + slice(FOLD, '#if PEARL_BD_A_DIRECT && (defined(PEARL_ABLATE_BARRIER)',
    '#error "PEARL_ABLATE_BLOAD and _BHIT change the default') + NL + '#endif';

const configs = new Map();
function config(spec) {
  if (!configs.has(spec)) {
    const r = preprocess(GUARDS, D(spec));
    configs.set(spec, {
      errors: r.errors, defs: r.defs, value: (name) => compile(name, false)(macroScope(r.defs)),
    });
  }
  return configs.get(spec);
}

const builds = new Map();
function build(spec) {
  if (!builds.has(spec)) {
    const c = config(spec);
    expect([spec, c.errors]).toEqual([spec, []]);
    const r = preprocess(FOLD, new Map([...c.defs, ...D('PEARL_STAGE_REGS=1')]));
    expect([spec, r.errors]).toEqual([spec, []]);
    builds.set(spec, { lines: r.lines, text: r.lines.join(NL), defs: r.defs });
  }
  return builds.get(spec);
}

function depth(line) {
  let d = 0;
  let inStr = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inStr) {
      if (ch === BS) i++;
      else if (ch === '"') inStr = false;
    } else if (ch === '"') {
      inStr = true;
    } else if (ch === '{') {
      d++;
    } else if (ch === '}') {
      d--;
    }
  }
  return d;
}

// The lines of the block that opens at or after the only line equal to head, to its
// closing brace.
function block(lines, head) {
  const at = lines.indexOf(head);
  expect([head, at >= 0, lines.lastIndexOf(head) === at]).toEqual([head, true, true]);
  let d = 0;
  let opened = false;
  for (let i = at; i < lines.length; i++) {
    d += depth(lines[i]);
    if (d > 0) opened = true;
    if (opened && d === 0) return lines.slice(at, i + 1);
  }
  throw new Error(`"${head}" does not close`);
}
const flatLines = (lines) => lines.join(' ').replace(/\s+/g, ' ').trim();

function inOrder(lines, parts, from = 0) {
  let at = from - 1;
  return parts.map((p) => {
    const i = lines.indexOf(p, at + 1);
    expect([p, i > at]).toEqual([p, true]);
    at = i;
    return i;
  });
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

// ---- The geometry --------------------------------------------------------------------

const THREADS = 256;
const CHUNKS = 16;
const KSTEPS = 4;
const STEPS = CHUNKS * KSTEPS;    // k-steps a tile, k 2048
const MB = 6;                     // 16-row blocks a warp
const RPL = MB / 2;               // a lane's regions: one per pair of m-blocks
const REGIONS = 8 * 4 * MB;       // 192 a tile
const TWORDS = REGIONS * CHUNKS;  // 3072 words a buffer
const FRAG = STEPS * 512;         // FRAG_BLOCK: a 16-row block of A' or B', k 2048
const TALL_ROW_OFFSETS = 12;
const ids = (t) => {
  const warp = t >> 5;
  return { 'threadIdx.x': t, warp, lane: t & 31, wr: warp >> 2, wc: warp & 3, sid: t & 127 };
};
// pearl_config.h's PEARL_TALL_A_ROWS: A's rows padded to whole 192-row tiles.
const TALL_A_ROWS = (m) => Math.ceil(m / 192) * 192;

// The tall hash_tile's head: the role, the loop over the two passes, and each pass's
// region.
const ROLE = ['const bool hasher = wr ? sid >= 64u : sid < 64u;', 'const uint32_t L = wr * 64u + (sid & 63u);',
  'const uint32_t hid = L;', '#pragma unroll 1', 'for (uint32_t pass = 0; hasher && pass < 2u; pass++) {',
  'if (pass && (hid & 31u) >= 16u) break;', 'const uint32_t L = pass ? 128u + (hid >> 5) * 16u + (hid & 15u) : hid;'];
const HASH_TILE = 'auto hash_tile = [&](uint32_t vtile) {';

// The regions thread t hashes, pass by pass, as the build's lines compute them.
function hashedBy(t) {
  const b = build('TALL=1');
  const ht = block(b.lines, HASH_TILE);
  const at = inOrder(ht, ROLE)[0];
  expect(ht.slice(at, at + ROLE.length)).toEqual(ROLE);
  const env = ids(t);
  const hasher = run('wr ? sid >= 64u : sid < 64u', env) === 1;
  const hid = run('wr * 64u + (sid & 63u)', env);
  const out = [];
  for (let pass = 0; hasher && pass < 2; pass++) {
    if (run('pass && (hid & 31u) >= 16u', { pass, hid })) break;
    out.push(run('pass ? 128u + (hid >> 5) * 16u + (hid & 15u) : hid', { pass, hid }));
  }
  return out;
}

// region_of and TALL's wrapper, copied: the number a hit reports for region L of the tile
// at row group rbg and column group cbg, or none in the padding rows.
// Integer division of the small non-negative values here.
const idiv = (a, b) => (a / b) | 0;
function regionOf(L, rbg, cbg, rowsValid, rpw = 6, wrg = 24) {
  const ow = idiv(L, wrg);
  const ocb = idiv(L % wrg, rpw);
  const oreg = L % rpw;
  const orb = rbg * 2 + (ow >> 2);
  const ocgb = cbg * 4 + (ow & 3);
  return (ocgb * 4 + ocb) * rowsValid + orb * rpw + oreg;
}
const NONE = 0xffffffff;
function tallRegionOf(L, rbg, cbg, rowsValid) {
  const ow = idiv(L, 24);
  const oreg = L % 6;
  const orb = rbg * 2 + (ow >> 2);
  if (orb * 6 + oreg >= rowsValid) return NONE;
  return regionOf(L, rbg, cbg, rowsValid);
}
// The fold's row groups and tile_coords (column bands, PEARL_BD_WALK 1), copied.
const rowGroupsOf = (rowsValid) => Math.floor((rowsValid + 2 * MB - 1) / (2 * MB));
function tileCoords(v, { rowsValid, tiles, bandDepth }, rowGroups = rowGroupsOf(rowsValid)) {
  const colGroups = idiv(tiles, rowGroups);
  const bandTiles = bandDepth * rowGroups;
  const band = idiv(v, bandTiles);
  const inBand = v % bandTiles;
  const bandFirst = band * bandDepth;
  const bandCols = colGroups - bandFirst < bandDepth ? colGroups - bandFirst : bandDepth;
  return { cbg: bandFirst + (inBand % bandCols), rbg: idiv(inBand, bandCols) };
}
// The host's tile count with the switch: ceil(rows_valid / 12) row groups by col_batch / 16.
const hostTiles = (rowsValid, colBatch) => Math.ceil(rowsValid / TALL_ROW_OFFSETS) * (colBatch / 4 / 4);

// The shapes compilers.sh's complete stage runs: m, col_batch, seconds.
const COMPLETE = /^COMPLETE_SHAPES=\$\{COMPLETE_SHAPES:-(.+)\}$/m.exec(COMPILERS)[1].split(';')
  .map((s) => s.split(' ').map(Number)).map(([m, colBatch]) => ({ m, colBatch }));

// ---- Tests -----------------------------------------------------------------------------

const MSG_VALUE = '"PEARL_BD_TALL is 0 or 1"';
const MSG_FOLD = '"PEARL_BD_TALL is written for the default fold with LEAN and HIT_COORDS: A-direct, k-lo first, '
  + 'persistent, two row groups, two transcript buffers, RUNBASE, TAIL_LDG and GATE 1"';
const MSG_WITH = '"PEARL_BD_TALL is not built with RED, TADDR, the split hash, SNAKE, LAST_RO, ROWPAIR or HASH_RG"';
const MSG_ABLATE = '"PEARL_ABLATE_BLOAD and _BHIT change the default\'s B loads, which PEARL_BD_TALL\'s k-step does not have"';
// Builds with the switch off: the default, the other folds, every other switch, the
// ablations.
const BASES = ['', OFF, 'HIT_COORDS=0', 'SNAKE=1', 'LAST_RO=1', 'ROWPAIR=1', 'HASH_RG=1', 'TADDR=1', 'RED=1',
  'RED=1 GATE=1', 'HASH_SPLIT=4 RUNBASE=1 TAIL_LDG=1', 'LAST_RO=1 ROWPAIR=1 HASH_RG=1', 'PERSIST=0', 'A_REALIGN=1',
  'KORDER=0', 'A_DIRECT=0', 'GROUPS=1', 'PEARL_ABLATE_TRANSCRIPT_HASH', 'PEARL_ABLATE_READOUT',
  'PEARL_ABLATE_TRANSCRIPT', 'PEARL_ABLATE_BLOAD', 'PEARL_ABLATE_BHIT', 'PEARL_ABLATE_BSHARE'];
// Each build's lines at d2e61d4, before PEARL_BD_TALL (see the line-for-line test).
const BEFORE_TALL = {
  '': '5691adfcb68a', 'LEAN=0 HIT_COORDS=0': '1f449377837a', 'HIT_COORDS=0': '9e2595e28901', 'SNAKE=1': 'df5d021a0378',
  'LAST_RO=1': '1a659a3d8136', 'ROWPAIR=1': 'a4ae013f1518', 'HASH_RG=1': '0e0ba8303f48', 'TADDR=1': '70112cf891ea',
  'RED=1': 'e82a491ed42e', 'RED=1 GATE=1': '59dc92ea0fa1', 'HASH_SPLIT=4 RUNBASE=1 TAIL_LDG=1': '39a43a2642ff',
  'LAST_RO=1 ROWPAIR=1 HASH_RG=1': '57046bd3e709', 'PERSIST=0': '8aa5df230dbd', 'A_REALIGN=1': 'e96750e61a53',
  'KORDER=0': '99f745c1ea05', 'A_DIRECT=0': 'e0b439949bc9', 'GROUPS=1': 'cb1f6b7a7874',
  'PEARL_ABLATE_TRANSCRIPT_HASH': '0750cee541a8', 'PEARL_ABLATE_READOUT': '102fb1af8f96',
  'PEARL_ABLATE_TRANSCRIPT': '82bcb884bba4', 'PEARL_ABLATE_BLOAD': '30f5b319653c', 'PEARL_ABLATE_BHIT': 'e57af394050e',
  'PEARL_ABLATE_BSHARE': '8ab9ae846ba9',
};

describe('Turing tall warp tiles (PEARL_BD_TALL)', () => {
  test('is off by default, and the header refuses it where it is not written for', () => {
    const at = HEADER.indexOf('#ifndef PEARL_BD_TALL' + NL);
    expect(at).toBeGreaterThanOrEqual(0);
    expect(HEADER.slice(at).split(NL).slice(0, 3)).toEqual(['#ifndef PEARL_BD_TALL', '#define PEARL_BD_TALL 0', '#endif']);
    expect(config('').value('PEARL_BD_TALL')).toBe(0);

    // It builds on the default fold, and with the ablations that apply to it.
    for (const s of ['', 'LEAN=1 HIT_COORDS=1', 'PEARL_ABLATE_TRANSCRIPT_HASH', 'PEARL_ABLATE_READOUT',
      'PEARL_ABLATE_TRANSCRIPT', 'PEARL_ABLATE_BSHARE']) {
      expect([T(s), config(T(s)).errors, config(T(s)).value('PEARL_BD_TALL')]).toEqual([T(s), [], 1]);
      expect(build(T(s)).lines.length).toBeGreaterThan(0);
    }
    // It changes no other switch.
    const a = config('');
    const b = config('TALL=1');
    expect([...b.defs.keys()].sort()).toEqual([...a.defs.keys()].sort());
    for (const name of a.defs.keys()) {
      if (name !== 'PEARL_BD_TALL') expect([name, b.value(name)]).toEqual([name, a.value(name)]);
    }

    // Each refusal: exactly TALL's #error lines, on a build that has none without it.
    const refused = (spec, msgs) => {
      expect([spec, config(spec).errors]).toEqual([spec, []]);
      expect([spec, config(T(spec)).errors]).toEqual([spec, msgs]);
    };
    for (const v of ['2', '-1']) expect(config(`TALL=${v}`).errors).toEqual([MSG_VALUE]);
    for (const s of ['PERSIST=0', 'A_REALIGN=1', 'KORDER=0', 'A_DIRECT=0', 'GROUPS=1', 'LEAN=0', 'HIT_COORDS=0', OFF,
      'GATE=2', 'RUNBASE=0', 'TAIL_LDG=0']) refused(s, [MSG_FOLD]);
    for (const s of ['TADDR=1', 'SNAKE=1', 'LAST_RO=1', 'ROWPAIR=1', 'HASH_RG=1', 'RED=1 GATE=1']) refused(s, [MSG_WITH]);
    // RED and the split hash also leave LEAN, so GATE 1, off.
    for (const s of ['RED=1', 'HASH_SPLIT=4 RUNBASE=1 TAIL_LDG=1']) refused(s, [MSG_FOLD, MSG_WITH]);
    // The fold refuses the two ablations of the default's B loads.
    for (const s of ['PEARL_ABLATE_BLOAD', 'PEARL_ABLATE_BHIT']) {
      const c = config(T(s));
      expect([s, c.errors]).toEqual([s, [MSG_ABLATE]]);
      expect(config(s).errors).toEqual([]);
    }
    // The guards read only names the header has set by then.
    const guards = HEADER.split(NL).map((l, i) => [l, i]).filter(([l]) => /^#if .*PEARL_BD_TALL\b/.test(l));
    expect(guards.length).toBe(3);
    const defLine = (name) => HEADER.split(NL).findIndex((l) => l.startsWith('#define ' + name + ' '));
    for (const [, i] of guards) {
      let full = HEADER.split(NL)[i];
      for (let k = i; full.endsWith(BS); k++) full = full.slice(0, -1) + ' ' + HEADER.split(NL)[k + 1];
      for (const name of full.match(/PEARL_\w+/g)) {
        expect([name, defLine(name) >= 0 && defLine(name) < i]).toEqual([name, true]);
      }
    }
  });

  test('without the switch every build is the build before it, line for line', () => {
    // Each build's lines without the switch, as preprocess() reads them, against d2e61d4,
    // the commit before PEARL_BD_TALL: the fold there gives these same lines for every
    // build (sha256 of the lines, first 12 hex digits). A later change to one of these
    // builds changes its digest; recompute it then.
    const digest = (lines) => crypto.createHash('sha256').update(lines.join(NL)).digest('hex').slice(0, 12);
    const got = {};
    for (const base of BASES) {
      const fold = preprocess(FOLD, new Map([...config(`${base} TALL=0`.trim()).defs, ...D('PEARL_STAGE_REGS=1')]));
      expect([base, config(base).errors, fold.errors]).toEqual([base, [], []]);
      got[base] = digest(fold.lines);
    }
    expect(got).toEqual(BEFORE_TALL);
    // And no name of the switch's own appears without it.
    const own = /\b(sCur|hid|pass|bfl|bfh|ro_step|tall_region_of|hi_ready|par|sub)\b/;
    for (const base of BASES) {
      expect([base, build(base).lines.filter((l) => own.test(l))]).toEqual([base, []]);
    }
    const t = build('TALL=1').text;
    for (const n of ['sCur', 'hid', 'pass', 'bfl', 'bfh', 'ro_step', 'tall_region_of', 'hi_ready', 'par']) {
      expect([n, new RegExp(`\\b${n}\\b`).test(t)]).toEqual([n, true]);
    }
  });

  test('the four hashers hash each of a tile\'s 192 regions once, 48 a warp', () => {
    const seen = new Map();
    const perWarp = new Map();
    for (let t = 0; t < THREADS; t++) {
      const Ls = hashedBy(t);
      const { warp, lane } = ids(t);
      Ls.forEach((L, pass) => {
        seen.set(L, (seen.get(L) || 0) + 1);
        perWarp.set(warp, (perWarp.get(warp) || 0) + 1);
        // Pass 1 only on lanes 0-15.
        if (pass === 1) expect([t, lane < 16]).toEqual([t, true]);
      });
      // Pass 0 is the default's region for the thread.
      const d = { hasher: run('wr ? sid >= 64u : sid < 64u', ids(t)) === 1, L: run('wr * 64u + (sid & 63u)', ids(t)) };
      if (d.hasher) expect([t, Ls[0]]).toEqual([t, d.L]);
      else expect([t, Ls]).toEqual([t, []]);
    }
    expect([...seen.keys()].sort((x, y) => x - y)).toEqual(Array.from({ length: REGIONS }, (_, i) => i));
    expect([...seen.values()].every((n) => n === 1)).toBe(true);
    // Warps 0, 1, 6 and 7, one on each scheduler, 48 regions each.
    expect([...perWarp.entries()].sort((x, y) => x[0] - y[0])).toEqual([[0, 48], [1, 48], [6, 48], [7, 48]]);
    // The default's role picks them (the lines before the loop are its own).
    const b = build('TALL=1');
    const ht = block(b.lines, HASH_TILE);
    const dflt = block(build('').lines, HASH_TILE);
    const i = ht.indexOf(ROLE[2]);
    expect(ht.slice(0, i)).toEqual(dflt.slice(0, dflt.indexOf('if (hasher) {')));
    // After the loop's head, a pass is the default's body for region L: its 16 words, its
    // number and its test.
    const j = dflt.indexOf('if (hasher) {');
    expect(ht.slice(i + 3, i + 8)).toEqual([ROLE[5], ROLE[6], ...dflt.slice(j + 1, j + 4)]);
  });

  test('each region\'s 16 words are stored where its hasher reads them, in two buffers of 12 KB', () => {
    const b = build('TALL=1');
    const st = flatLines(block(b.lines, 'auto readout_st = [&](uint32_t rl, uint32_t c, uint32_t x) {'));
    const m = new RegExp('^auto readout_st = \\[&\\]\\(uint32_t rl, uint32_t c, uint32_t x\\) \\{ const uint32_t L = (.+?); '
      + 'const uint32_t a = \\(uint32_t\\)__cvta_generic_to_shared\\(sT \\+ (.+?)\\); asm volatile\\("\\{ \\.reg \\.pred p; '
      + 'setp\\.eq\\.u32 p, %2, 0; @p st\\.shared\\.u32 \\[%0\\], %1; \\}" ::"r"\\(a\\), "r"\\(x\\), "r"\\((.+?)\\) : "memory"\\); \\};$')
      .exec(st);
    expect(m).not.toBeNull();
    const [, regionExpr, storeExpr, predExpr] = m;
    expect([regionExpr, storeExpr, predExpr]).toEqual([
      'warp * warp_regions + (lane & 3u) * regions_per_warp + 2u * rl + ((lane >> 4) & 1u)',
      'tbuf + L * 16u + (c ^ ((L >> 1) & 15u))', '(lane >> 2) & 3u']);
    const readLine = block(b.lines, HASH_TILE).find((l) => l.startsWith('for (uint32_t c = 0; c < 16u; c++) tm[c] = sT['));
    expect(readLine).toBe('for (uint32_t c = 0; c < 16u; c++) tm[c] = sT[tbuf + L * 16u + (c ^ ((L >> 1) & 15u))];');
    const readExpr = 'tbuf + L * 16u + (c ^ ((L >> 1) & 15u))';
    // The sizes the expressions run with: the build's.
    expect(b.lines).toContain('constexpr uint32_t MB = 6u, NB = 8u, KSTEPS = rank / 32u;');
    expect(b.lines).toContain('constexpr uint32_t regions_per_warp = MB, warp_regions = 4u * MB;');
    expect(b.lines).toContain('constexpr uint32_t RPL = MB / 2u;');
    expect(b.lines).toContain('constexpr uint32_t TWORDS = (PEARL_FOLD_THREADS / 32u) * warp_regions * PEARL_JACKPOT_BUCKETS;');
    expect(b.lines).toContain('static_assert(PEARL_BD_SMEM_A_STAGES == 0 && PEARL_BD_T_BUFS == 2 && TWORDS * 4u == 12288u,');
    const consts = { regions_per_warp: MB, warp_regions: 4 * MB };
    for (const tbuf of [0, TWORDS]) {
      const used = new Map();
      const writerOf = new Map();
      for (let warp = 0; warp < 8; warp++) {
        for (let lane = 0; lane < 32; lane++) {
          if (run(predExpr, { lane }) !== 0) continue;
          for (let rl = 0; rl < RPL; rl++) {
            const L = run(regionExpr, { ...consts, warp, lane, rl });
            expect([L, writerOf.has(L)]).toEqual([L, false]);
            writerOf.set(L, { warp, lane, rl });
            // The lane's region: its warp, its column offset lane & 3, and row offset 2 rl +
            // (lane >> 4 & 1), from m-blocks 2 rl and 2 rl + 1 (fold_pair, below).
            expect([Math.floor(L / 24), Math.floor((L % 24) / 6), L % 6]).toEqual([warp, lane & 3, 2 * rl + ((lane >> 4) & 1)]);
            for (let c = 0; c < CHUNKS; c++) {
              const at = run(storeExpr, { tbuf, L, c });
              expect(run(readExpr, { tbuf, L, c })).toBe(at);
              expect([at >= tbuf, at < tbuf + TWORDS, used.has(at)]).toEqual([true, true, false]);
              used.set(at, `${L}.${c}`);
            }
          }
        }
      }
      expect(writerOf.size).toBe(REGIONS);
      expect(used.size).toBe(TWORDS);
    }
    // The fold of region rl reads m-blocks 2 rl and 2 rl + 1.
    expect(b.lines).toContain('const int32_t *p0 = acc[2u * rl][2u * np], *p1 = acc[2u * rl + 1u][2u * np];');
    expect(b.lines).toContain('const int32_t *p2 = acc[2u * rl][2u * np + 1u], *p3 = acc[2u * rl + 1u][2u * np + 1u];');
  });

  test('a region\'s number names the rows its accumulators hold, as the default\'s does', () => {
    // region_of gives row offset o = orb * regions_per_warp + oreg; the region's fold reads
    // m-blocks 2 rl and 2 rl + 1, which a_off puts at A' blocks o & ~1 and o | 1 in both.
    const tall = build('TALL=1');
    const dflt = build('');
    expect(tall.lines).toContain('return (rbg_ * (2u * MB) + wr * MB) * FRAG_BLOCK + lane * 16u;');
    expect(dflt.lines).toContain('return (rbg_ * 8u + wr * 4u) * FRAG_BLOCK + lane * 16u;');
    const ro = ['auto region_of = [&](uint32_t rbg_, uint32_t cbg_) -> uint32_t {', 'const uint32_t ow = L / warp_regions;',
      'const uint32_t ocb = (L % warp_regions) / regions_per_warp;', 'const uint32_t oreg = L % regions_per_warp;',
      'const uint32_t orb = rbg_ * 2u + ow / 4u;', 'const uint32_t ocgb = cbg_ * 4u + ow % 4u;',
      'return (ocgb * 4u + ocb) * rows_valid + orb * regions_per_warp + oreg;', '};'];
    for (const b of [tall, dflt]) expect(block(block(b.lines, HASH_TILE), ro[0])).toEqual(ro);
    for (const [b, mb, rpw] of [[tall, 6, 6], [dflt, 4, 4]]) {
      const aoff = (rbg, wr) => run(b === tall ? '(rbg_ * (2u * MB) + wr * MB)' : '(rbg_ * 8u + wr * 4u)', { rbg_: rbg, wr, MB: mb });
      for (let rbg = 0; rbg < 5; rbg++) {
        for (let L = 0; L < 32 * mb; L++) {
          const env = { L, rbg_: rbg, cbg_: 0, rows_valid: 1 << 20, regions_per_warp: rpw, warp_regions: 4 * rpw };
          for (const s of ro.slice(1, 6)) {
            const [, n, e] = /^const uint32_t (\w+) = (.+);$/.exec(s);
            env[n] = run(e, env);
          }
          const o = env.orb * rpw + env.oreg;
          const warp = Math.floor(L / (4 * rpw));
          const rl = (L % rpw) >> 1;
          expect([o & ~1, o | 1]).toEqual([aoff(rbg, warp >> 2) + 2 * rl, aoff(rbg, warp >> 2) + 2 * rl + 1]);
        }
      }
    }
  });

  test('over whole launches every region is numbered once, as without the switch, and none from the padding', () => {
    const b = build('TALL=1');
    // The copies, pinned.
    expect(b.lines).toContain('const uint32_t row_block_groups = (rows_valid + 2u * MB - 1u) / (2u * MB);');
    expect(b.lines).toContain('const uint32_t col_block_groups = tiles / row_block_groups;');
    expect(b.lines).toContain('const uint32_t band_tiles = band_depth * row_block_groups;');
    expect(flatLines(block(b.lines, 'auto tile_coords = [&](uint32_t v, uint32_t &rbg_, uint32_t &cbg_) {'))).toBe(
      'auto tile_coords = [&](uint32_t v, uint32_t &rbg_, uint32_t &cbg_) { const uint32_t band = v / band_tiles; '
      + 'const uint32_t in_band = v % band_tiles; const uint32_t band_first = band * band_depth; '
      + 'const uint32_t band_cols = col_block_groups - band_first < band_depth ? col_block_groups - band_first : band_depth; '
      + 'cbg_ = band_first + in_band % band_cols; rbg_ = in_band / band_cols; };');
    const ht = block(b.lines, HASH_TILE);
    expect(block(ht, 'auto tall_region_of = [&](uint32_t rbg_, uint32_t cbg_) -> uint32_t {')).toEqual([
      'auto tall_region_of = [&](uint32_t rbg_, uint32_t cbg_) -> uint32_t {', 'const uint32_t ow = L / warp_regions;',
      'const uint32_t oreg = L % regions_per_warp;', 'const uint32_t orb = rbg_ * 2u + ow / 4u;',
      'if (orb * regions_per_warp + oreg >= rows_valid) return 0xFFFFFFFFu;', 'return region_of(rbg_, cbg_);', '};']);
    // A hit: its number, dropped in the padding before it takes a slot, then recorded.
    inOrder(ht, ['if (!pearl_hash_meets_words(h, test.target_w, test.hash_big_endian)) return;', 'uint32_t rbg, cbg;',
      'tile_coords(vtile, rbg, cbg);', 'const uint32_t region = tall_region_of(rbg, cbg);',
      'if (region == 0xFFFFFFFFu) return;', 'const uint32_t slot = atomicAdd(hits.count, 1u);',
      'if (slot >= PEARL_MAX_HITS) return;', 'hits.index[slot] = region;']);
    expect(ht.filter((l) => /tile_coords\(|atomicAdd\(/.test(l))).toEqual(['tile_coords(vtile, rbg, cbg);',
      'const uint32_t slot = atomicAdd(hits.count, 1u);']);

    // The copies run as the code does: row groups from the build's expression, the walk
    // from tile_coords's lines, the number from region_of's and tall_region_of's.
    const walk = flatLines(block(b.lines, 'auto tile_coords = [&](uint32_t v, uint32_t &rbg_, uint32_t &cbg_) {'))
      .replace(/^auto tile_coords = \[&\]\(uint32_t v, uint32_t &rbg_, uint32_t &cbg_\) \{ /, '').replace(/ \};$/, '')
      .split(';').map((x) => x.trim()).filter(Boolean);
    // Each statement as [name, compiled expression], run in order over one environment.
    const steps = (list) => list.map(([n, e]) => [n, compile(e, true)]);
    const runSteps = (list, env) => {
      const scope = codeScope(env);
      for (const [n, f] of list) env[n] = f(scope);
      return env;
    };
    const walkSteps = steps([['row_block_groups', '(rows_valid + 2u * MB - 1u) / (2u * MB)'],
      ['col_block_groups', 'tiles / row_block_groups'], ['band_tiles', 'band_depth * row_block_groups'],
      ...walk.map((st) => /^(?:const uint32_t )?(\w+) = (.+)$/.exec(st).slice(1, 3))]);
    const fromCode = (v, geo) => {
      const env = runSteps(walkSteps, { rows_valid: geo.rowsValid, tiles: geo.tiles, band_depth: geo.bandDepth, MB, v });
      return { rbg: env.rbg_, cbg: env.cbg_ };
    };
    // tall_region_of's lines, then region_of's (pinned above).
    const padSteps = steps([['ow', 'L / warp_regions'], ['oreg', 'L % regions_per_warp'], ['orb', 'rbg_ * 2u + ow / 4u'],
      ['pad', 'orb * regions_per_warp + oreg >= rows_valid']]);
    const numSteps = steps([['ocb', '(L % warp_regions) / regions_per_warp'], ['ocgb', 'cbg_ * 4u + ow % 4u'],
      ['n', '(ocgb * 4u + ocb) * rows_valid + orb * regions_per_warp + oreg']]);
    const numberFromCode = (L, rbg, cbg, rowsValid) => {
      const env = runSteps(padSteps, { L, rbg_: rbg, cbg_: cbg, rows_valid: rowsValid, regions_per_warp: 6, warp_regions: 24 });
      return env.pad ? NONE : runSteps(numSteps, env).n;
    };
    const hashers = [];
    for (let t = 0; t < THREADS; t++) for (const L of hashedBy(t)) hashers.push(L);

    // Every shape compilers.sh's complete stage runs, and a few more, at several bands.
    expect(COMPLETE).toEqual([{ m: 131072, colBatch: 2048 }, { m: 8192, colBatch: 32 }, { m: 8192, colBatch: 16 },
      { m: 4096, colBatch: 16 }]);
    const shapes = [...COMPLETE.map((s) => ({ ...s, bands: s.m === 131072 ? [5] : [1, 2, 4, 5, 9] })),
      { m: 32 * 16, colBatch: 48, bands: [1, 2, 3] }, { m: 192 * 16, colBatch: 32, bands: [1, 2] }];
    for (const { m, colBatch, bands } of shapes) {
      const rowsValid = m / 16;
      const tiles = hostTiles(rowsValid, colBatch);
      const rowGroups = rowGroupsOf(rowsValid);
      expect([m, tiles % rowGroups, tiles / rowGroups]).toEqual([m, 0, colBatch / 16]);
      const regions = rowsValid * colBatch;
      const pad = (rowGroups * TALL_ROW_OFFSETS - rowsValid) * colBatch;
      let checked = 0;
      for (const bandDepth of bands) {
        const geo = { rowsValid, tiles, bandDepth };
        const count = new Uint8Array(regions);
        let padding = 0;
        let strayPad = 0;
        for (let v = 0; v < tiles; v++) {
          const { rbg, cbg } = tileCoords(v, geo);
          // The copies against the code: on every tile of a small shape at its first band,
          // and on a sample of the large one.
          const sample = m < 131072 ? bandDepth === bands[0] : v % 977 === 0;
          if (sample) expect([v, fromCode(v, geo)]).toEqual([v, { rbg, cbg }]);
          for (const L of hashers) {
            const n = tallRegionOf(L, rbg, cbg, rowsValid);
            if (sample) {
              expect(numberFromCode(L, rbg, cbg, rowsValid)).toBe(n);
              checked++;
            }
            if (n === NONE) {
              // Only the last row group's rows past m.
              if (rbg !== rowGroups - 1 || (rbg * 2 + idiv(L, 96)) * 6 + (L % 6) < rowsValid) strayPad++;
              padding++;
            } else {
              count[n]++;
            }
          }
        }
        expect([m, colBatch, bandDepth, padding, strayPad]).toEqual([m, colBatch, bandDepth, pad, 0]);
        expect([m, colBatch, bandDepth, count.every((n) => n === 1)]).toEqual([m, colBatch, bandDepth, true]);
      }
      expect([m, checked]).toEqual([m, m < 131072 ? tiles * REGIONS : Math.ceil(tiles / 977) * REGIONS]);
      // The default numbers the same regions, every one once, from 8-offset row groups:
      // a region's number is its row and column offset either way.
      if (rowsValid % 8 === 0 && m <= 8192) {
        const count = new Uint8Array(regions);
        const dTiles = (rowsValid / 8) * (colBatch / 16);
        for (let v = 0; v < dTiles; v++) {
          const { rbg, cbg } = tileCoords(v, { rowsValid, tiles: dTiles, bandDepth: 2 }, rowsValid / 8);
          for (let L = 0; L < 128; L++) count[regionOf(L, rbg, cbg, rowsValid, 4, 16)]++;
        }
        expect([m, colBatch, count.every((n) => n === 1)]).toEqual([m, colBatch, true]);
      }
    }
  });

  test('every load stays inside the A\' and B\' the host allocates', () => {
    // The host pads A' to whole 192-row tiles and zeroes the padding once; TALL refuses
    // an m that is not a multiple of 32 or a padded A' of 4 GB or more.
    const def = /#define PEARL_TALL_A_ROWS\(m\) \\\n\s*(.+)\n/.exec(HEADER);
    expect(def[1]).toBe('((((m) + PEARL_TALL_BM - 1u) / PEARL_TALL_BM) * PEARL_TALL_BM)');
    expect(HEADER).toContain('#define PEARL_TALL_BM (2u * PEARL_TALL_ROW_TILES * 16u)');
    expect(HEADER).toContain('#define PEARL_TALL_ROW_TILES 6u');
    expect(HOST).toContain('    const size_t apBytes = (size_t)PEARL_TALL_A_ROWS(profile->m) * k;' + NL
      + '    CUDA_OK(cudaMalloc(&ctx->dAp, apBytes), "allocating the noised A");' + NL
      + '    if (apBytes > aBytes) cudaMemset(ctx->dAp + aBytes, 0, apBytes - aBytes);');
    expect(HOST).toContain('#if PEARL_BD_TALL' + NL + '  // PEARL_BD_TALL reads A\' past m');
    expect(HOST).toContain('      && (ctx->profile.m % 32u != 0u' + NL
      + '          || (uint64_t)PEARL_TALL_A_ROWS(ctx->profile.m) * k > 0xFFFFFFFFull)) {');
    // A warp's loads: A quad q of a k-step at a_off + (2 q or 2 q + 1) blocks + s * 512,
    // B's at b_off + (2 q or 2 q + 1) blocks + s * 512, the last k-step's from the next
    // tile's offsets (another tile of the launch, or this one).
    const b = build('TALL=1');
    for (const l of ['af[q] = pearl_bd_ldg(Aprime + ao_this + (2u * q + 1u) * FRAG_BLOCK);',
      'af[q] = ldg(Aprime + ao + (2u * q) * FRAG_BLOCK);', 'bfl[q] = ldg(Bprime + bo + (2u * q) * FRAG_BLOCK);',
      'bfh[q] = pearl_bd_ldg(Bprime + bo_this + (2u * q + 1u) * FRAG_BLOCK);',
      'bfh[q] = pearl_bd_ldg_c(Bprime + boffn + (2u * q + 1u) * FRAG_BLOCK);',
      'tile_coords(vnext < tiles ? vnext : blockIdx.x, rbgn, cbgn);']) {
      expect([l, b.lines.some((x) => x.endsWith(l))]).toEqual([l, true]);
    }
    expect(b.lines).toContain('return (col_off + cbg_ * 16u + wc * 4u) * FRAG_BLOCK + lane * 16u;');
    const check = (m, n, colBatch, colOff, bandDepth) => {
      const rowsValid = m / 16;
      const tiles = hostTiles(rowsValid, colBatch);
      let aMax = 0;
      let bMax = 0;
      let pastM = 0;
      for (let v = 0; v < tiles; v++) {
        const { rbg, cbg } = tileCoords(v, { rowsValid, tiles, bandDepth });
        // wr 1, quad 2's k-hi block (5), lane 31, k-step 63, sixteen bytes.
        const aTop = (rbg * 12 + 6 + 5) * FRAG + 31 * 16 + (STEPS - 1) * 512 + 16;
        const bTop = (colOff + cbg * 16 + 12 + 3) * FRAG + 31 * 16 + (STEPS - 1) * 512 + 16;
        if (aTop > m * 2048) {
          expect(rbg).toBe(rowGroupsOf(rowsValid) - 1);
          pastM++;
        }
        aMax = Math.max(aMax, aTop);
        bMax = Math.max(bMax, bTop);
      }
      expect([m, aMax <= TALL_A_ROWS(m) * 2048, aMax <= 2 ** 32]).toEqual([m, true, true]);
      expect([n, bMax <= n * 2048, bMax <= 2 ** 32]).toEqual([n, true, true]);
      return { aMax, pastM };
    };
    const colsValid = 262144 / 16;
    for (const band of [2, 4, 5, 8]) {
      const r = check(131072, 262144, 2048, colsValid - 2048, band);
      // The last row group reads 64 rows of padding, to the padded end exactly.
      expect(r.aMax).toBe(TALL_A_ROWS(131072) * 2048);
      expect(r.pastM).toBe(2048 / 16);
      check(131072, 262144, 2048, 0, band);
    }
    for (const { m, colBatch } of COMPLETE) {
      for (const band of [1, 4, 5]) expect(check(m, 262144, colBatch, 0, band).aMax).toBe(TALL_A_ROWS(m) * 2048);
    }
    expect([TALL_A_ROWS(131072) - 131072, TALL_A_ROWS(8192) - 8192, TALL_A_ROWS(4096) - 4096]).toEqual([64, 64, 128]);
  });

  test('every IMMA takes the A and B words its accumulator sums, from A\' and B\' as the host draws them', () => {
    // The draw's store (turingKorder.test.js's copy, pinned there): pairs of 16-row blocks,
    // block 2p's slot holding both blocks' k 0-15 words and block 2p + 1's their k 16-31.
    const mat = KERNEL.slice(KERNEL.indexOf('extern "C" __global__ void pearl_materialize16_afrag('));
    expect(mat).toContain('const uint32_t dbl = paired ? (bl & ~1u) | ((kk >> 4) & 1u) : bl;');
    expect(mat).toContain('const uint32_t reg = paired ? (bl & 1u) * 2u + ((r >> 3) & 1u)');
    const kLog2 = 8;
    const K = 2 ** kLog2;
    const bytes = (n, seed) => {
      const rand = xorshift(seed);
      return Uint8Array.from({ length: n }, () => rand() & 255);
    };
    const word = (bs, at) => (bs[at] | (bs[at + 1] << 8) | (bs[at + 2] << 16) | (bs[at + 3] << 24)) >>> 0;
    const draw = (src, rows) => {
      const out = new Uint8Array(rows * K);
      const written = new Uint8Array(rows * K);
      for (let v = 0; v < (rows * K) / 16; v++) {
        const idx = v * 16;
        const r = Math.floor(idx / K);
        const kk = idx & (K - 1);
        const bl = r >> 4;
        const paired = (bl | 1) < (rows >> 4);
        const dbl = paired ? (bl & ~1) | ((kk >> 4) & 1) : bl;
        const blk = dbl * 2 ** (kLog2 - 5) + (kk >> 5);
        const reg = paired ? (bl & 1) * 2 + ((r >> 3) & 1) : ((r >> 3) & 1) * 2 + ((kk >> 4) & 1);
        const o = blk * 512 + (r & 7) * 64 + reg * 4;
        for (let w = 0; w < 4; w++) {
          for (let j = 0; j < 4; j++) {
            out[o + w * 16 + j] = src[idx + 4 * w + j];
            written[o + w * 16 + j]++;
          }
        }
      }
      expect(written.every((x) => x === 1)).toBe(true);
      return out;
    };
    // Two row groups of A (24 blocks, 384 rows: warps at blocks 0, 6, 12 and 18) and 8
    // blocks of B (two warps' column slots, at blocks 0 and 4).
    const aRows = 24 * 16;
    const bRows = 8 * 16;
    const a = bytes(aRows * K, 21);
    const bb = bytes(bRows * K, 22);
    const A = draw(a, aRows);
    const B = draw(bb, bRows);
    const quad = (frag, blk, s, lane) => {
      const at = blk * (K / 32) * 512 + s * 512 + lane * 16;
      return [word(frag, at), word(frag, at + 4), word(frag, at + 8), word(frag, at + 12)];
    };
    // m8n8k16's operands from the row-major bytes: lane L holds A row L / 4 and B column
    // L / 4, k bytes 4 (L % 4)..+3 of the k-step's half.
    const specA = (blk, u, s, hi, lane) => word(a, (16 * blk + 8 * u + (lane >> 2)) * K + 32 * s + 16 * hi + 4 * (lane & 3));
    const specB = (blk, odd, s, hi, lane) => word(bb, (16 * blk + 8 * odd + (lane >> 2)) * K + 32 * s + 16 * hi + 4 * (lane & 3));

    // The sub-pass's IMMA, as the build has them: accumulator acc[mb][n][2 u], [2 u + 1]
    // takes A word a(u + 2 (mb - 2 q)) of quad q and B word be (even n-block) or bo_.
    const sub = block(block(build('TALL=1').lines, 'auto kstep = [&](uint32_t ch, uint32_t t, bool ro, bool last, uint32_t dch = 0u,'),
      'auto sub = [&](uint32_t q, uint32_t hi, bool ro_) {');
    expect(sub.slice(1, 3)).toEqual(['const uint32_t a0 = (uint32_t)af[q].x, a1 = (uint32_t)af[q].y;',
      'const uint32_t a2 = (uint32_t)af[q].z, a3 = (uint32_t)af[q].w;']);
    const kb = block(build('TALL=1').lines, 'auto kstep = [&](uint32_t ch, uint32_t t, bool ro, bool last, uint32_t dch = 0u,');
    expect(flatLines(block(kb, 'auto b_word = [&](uint32_t j, uint32_t hi, uint32_t odd) -> uint32_t {'))).toBe(
      'auto b_word = [&](uint32_t j, uint32_t hi, uint32_t odd) -> uint32_t { const int4 v = hi ? bfh[j / 2u] : bfl[j / 2u]; '
      + 'return (uint32_t)(j & 1u ? (odd ? v.w : v.z) : (odd ? v.y : v.x)); };');
    const immaRe = /^pearl_bd_imma\(acc\[(.+?)\]\[(.+?)\]\[(\d)\], acc\[\1\]\[\2\]\[(\d)\], (a\d), (be|bo_)\);$/;
    const immas = sub.filter((l) => l.startsWith('pearl_bd_imma(')).map((l) => {
      const m = immaRe.exec(l);
      expect([l, m !== null]).toEqual([l, true]);
      expect(Number(m[4])).toBe(Number(m[3]) + 1);
      return { mb: m[1], n: m[2], i: Number(m[3]), a: Number(m[5][1]), odd: m[6] === 'be' ? 0 : 1 };
    });
    expect(immas.length).toBe(8);
    expect(sub).toContain('uint32_t be = b_word(j, hi, 0u);');
    expect(sub).toContain('const uint32_t bo_ = b_word(j, hi, 1u);');
    // A k-step's 192 IMMA: sub-pass q, pair j, half hi, and the line's accumulator and
    // words. af[q] holds A quad 2 q + hi of the warp's blocks (the rotation test checks
    // it), and bfl / bfh B quad 2 (j / 2) + hi.
    const table = [];
    for (let hi = 0; hi < 2; hi++) {
      for (let q = 0; q < RPL; q++) {
        for (let j = 0; j < 4; j++) {
          for (const x of immas) table.push({ hi, q, j, mb: run(x.mb, { q, j }), n: run(x.n, { q, j }), u: x.i / 2, a: x.a, odd: x.odd });
        }
      }
    }
    // Every accumulator pair of the warp, both halves, once; the accumulator's rows are
    // m-block mb's rows 8 u..; its columns n-block n's, of pair j.
    expect(new Set(table.map((e) => `${e.mb}.${e.n}.${e.u}.${e.hi}`)).size).toBe(192);
    for (const e of table) expect([e.mb >> 1, e.n >> 1, e.n & 1]).toEqual([e.q, e.j, e.odd]);
    const bad = [];
    let checked = 0;
    for (const aBase of [0, 6, 12, 18]) {
      for (const bBase of [0, 4]) {
        for (let s = 0; s < K / 32; s++) {
          for (let lane = 0; lane < 32; lane++) {
            const af = [0, 1].map((hi) => [0, 1, 2].map((q) => quad(A, aBase + 2 * q + hi, s, lane)));
            const bq = [0, 1].map((hi) => [0, 1].map((p) => quad(B, bBase + 2 * p + hi, s, lane)));
            for (const e of table) {
              const aw = af[e.hi][e.q][e.a];
              const bw = bq[e.hi][e.j >> 1][(e.j & 1) * 2 + e.odd];
              if ((aw !== specA(aBase + e.mb, e.u, s, e.hi, lane) || bw !== specB(bBase + e.j, e.odd, s, e.hi, lane))
                && bad.length < 5) bad.push(`A ${aBase} B ${bBase} k-step ${s} lane ${lane} acc ${e.mb}.${e.n}.${e.u} hi ${e.hi}`);
              checked++;
            }
          }
        }
      }
    }
    expect(bad).toEqual([]);
    expect(checked).toBe(4 * 2 * (K / 32) * 32 * 192);
  });

  test('the operands rotate right through every k-step, chunk and tile, and each readout reads whole chunks', () => {
    // The tall k-step, tile and prologue, copied below; the copies are pinned here, line
    // for line.
    const b = build('TALL=1');
    expect(flatLines(block(b.lines, 'auto tile_prologue = [&]() {'))).toBe('auto tile_prologue = [&]() { #pragma unroll '
      + 'for (uint32_t q = 0; q < MB / 2u; q++) af[q] = pearl_bd_ldg(Aprime + aoff + (2u * q) * FRAG_BLOCK); #pragma unroll '
      + 'for (uint32_t q = 0; q < 2u; q++) { bfl[q] = pearl_bd_ldg(Bprime + boff + (2u * q) * FRAG_BLOCK); '
      + 'bfh[q] = pearl_bd_ldg(Bprime + boff + (2u * q + 1u) * FRAG_BLOCK); } };');
    const ks = flatLines(block(b.lines, 'auto kstep = [&](uint32_t ch, uint32_t t, bool ro, bool last, uint32_t dch = 0u,'));
    expect(ks).toBe('auto kstep = [&](uint32_t ch, uint32_t t, bool ro, bool last, uint32_t dch = 0u, bool hi_ready = false) { '
      + '(void)ch; const uint32_t ao_this = arun + (dch * KSTEPS + t) * 512u; '
      + 'const uint32_t bo_this = brun + (dch * KSTEPS + t) * 512u; '
      + 'const uint32_t ao = last ? aoffn : arun + (dch * KSTEPS + t + 1u) * 512u; '
      + 'const uint32_t bo = last ? boffn : brun + (dch * KSTEPS + t + 1u) * 512u; '
      + 'auto ldg = [&](const int8_t *p) -> int4 { if (last) return pearl_bd_ldg_c(p); return pearl_bd_ldg(p); }; '
      + 'if (!hi_ready) { #pragma unroll for (uint32_t q = 0; q < 2u; q++) '
      + 'bfh[q] = pearl_bd_ldg(Bprime + bo_this + (2u * q + 1u) * FRAG_BLOCK); } '
      + 'auto b_word = [&](uint32_t j, uint32_t hi, uint32_t odd) -> uint32_t { const int4 v = hi ? bfh[j / 2u] : bfl[j / 2u]; '
      + 'return (uint32_t)(j & 1u ? (odd ? v.w : v.z) : (odd ? v.y : v.x)); }; '
      + 'auto sub = [&](uint32_t q, uint32_t hi, bool ro_) { '
      + 'const uint32_t a0 = (uint32_t)af[q].x, a1 = (uint32_t)af[q].y; const uint32_t a2 = (uint32_t)af[q].z, a3 = (uint32_t)af[q].w; '
      + '#pragma unroll for (uint32_t j = 0; j < 4u; j++) { uint32_t be = b_word(j, hi, 0u); const uint32_t bo_ = b_word(j, hi, 1u); '
      + 'if (ro_) { rxs[q] = fold_pair(rxs[q], q, j); be = pearl_bd_after(be, rxs[q]); } '
      + 'pearl_bd_imma(acc[2u * q][2u * j][0], acc[2u * q][2u * j][1], a0, be); '
      + 'pearl_bd_imma(acc[2u * q][2u * j][2], acc[2u * q][2u * j][3], a1, be); '
      + 'pearl_bd_imma(acc[2u * q + 1u][2u * j][0], acc[2u * q + 1u][2u * j][1], a2, be); '
      + 'pearl_bd_imma(acc[2u * q + 1u][2u * j][2], acc[2u * q + 1u][2u * j][3], a3, be); '
      + 'pearl_bd_imma(acc[2u * q][2u * j + 1u][0], acc[2u * q][2u * j + 1u][1], a0, bo_); '
      + 'pearl_bd_imma(acc[2u * q][2u * j + 1u][2], acc[2u * q][2u * j + 1u][3], a1, bo_); '
      + 'pearl_bd_imma(acc[2u * q + 1u][2u * j + 1u][0], acc[2u * q + 1u][2u * j + 1u][1], a2, bo_); '
      + 'pearl_bd_imma(acc[2u * q + 1u][2u * j + 1u][2], acc[2u * q + 1u][2u * j + 1u][3], a3, bo_); } }; '
      + '#pragma unroll for (uint32_t q = 0; q < MB / 2u; q++) { sub(q, 0u, ro); '
      + 'af[q] = pearl_bd_ldg(Aprime + ao_this + (2u * q + 1u) * FRAG_BLOCK); } '
      + '#pragma unroll for (uint32_t q = 0; q < 2u; q++) bfl[q] = ldg(Bprime + bo + (2u * q) * FRAG_BLOCK); '
      + '#pragma unroll for (uint32_t q = 0; q < MB / 2u; q++) { sub(q, 1u, false); af[q] = ldg(Aprime + ao + (2u * q) * FRAG_BLOCK); } '
      + 'if (last) { #pragma unroll for (uint32_t q = 0; q < 2u; q++) '
      + 'bfh[q] = pearl_bd_ldg_c(Bprime + boffn + (2u * q + 1u) * FRAG_BLOCK); } };');
    expect(flatLines(block(b.lines, 'auto tile = [&](auto next_srcs) {'))).toBe('auto tile = [&](auto next_srcs) { '
      + '#pragma unroll for (uint32_t mb = 0; mb < MB; mb++) #pragma unroll for (uint32_t nb = 0; nb < NB; nb++) '
      + '#pragma unroll for (uint32_t i = 0; i < 4; i++) acc[mb][nb][i] = 0; arun = aoff; brun = boff; '
      + 'kstep(0u, 0u, false, false, 0u, true); #pragma unroll 1 for (uint32_t t = 1u; t < KSTEPS; t++) kstep(0u, t, false, false); '
      + 'auto ro_step = [&](uint32_t chunk) { arun = aoff + chunk * (KSTEPS * 512u); brun = boff + chunk * (KSTEPS * 512u); '
      + 'asm volatile("" : "+r"(arun), "+r"(brun)); kstep(chunk, 0u, RO, false); uint32_t rx[RPL]; #pragma unroll '
      + 'for (uint32_t rl = 0; rl < RPL; rl++) rx[rl] = readout_gather(rxs[rl]); #pragma unroll '
      + 'for (uint32_t rl = 0; rl < RPL; rl++) readout_st(rl, chunk - 1u, rx[rl]); }; '
      + '#pragma unroll 1 for (uint32_t chunk = 1u; chunk < chunks; chunk++) { ro_step(chunk); '
      + 'const uint32_t tend = chunk + 1u < chunks ? KSTEPS : KSTEPS - 1u; '
      + '#pragma unroll 1 for (uint32_t t = 1u; t < tend; t++) kstep(chunk, t, false, false); } '
      + 'next_srcs(); kstep(chunks - 1u, KSTEPS - 1u, false, true); #pragma unroll '
      + 'for (uint32_t rl = 0; rl < RPL; rl++) readout(rl, chunks - 1u); };');
    expect(b.lines).toContain('auto readout = [&](uint32_t rl, uint32_t c) { readout_st(rl, c, readout_x(rl)); };');

    // The copy, over a block's tiles. A load's offset names its tile (TILE apart), its
    // block of the warp's and its k-step; a register holds that tag. An IMMA must take A
    // and B words of the tile being summed, of the same k-step and half, from the blocks
    // its accumulator needs, and each accumulator must take k-step 0's halves, then 1's,
    // and so on. A readout must read sums of whole chunks. A loaded register overwritten
    // before an IMMA reads it is a wasted load: only the last tile's tail may have one.
    const TILE = 8 * FRAG;
    function runTiles(order, mutate = '') {
      const res = { imma: 0, wrong: [], wasted: 0, readouts: 0, badReadout: 0, tiles: 0 };
      const regs = { af: [null, null, null], bfl: [null, null], bfh: [null, null] };
      // acc[mb][n][2 u] and [2 u + 1], one IMMA's pair, as one entry.
      const acc = Array.from({ length: MB * 8 * 2 }, () => []);
      let cur = -1;
      let aoff = 0;
      let boff = 0;
      let aoffn = 0;
      let boffn = 0;
      let arun = 0;
      let brun = 0;
      const decode = (op, off) => ({ op, tile: Math.floor(off / TILE), blk: Math.floor((off % TILE) / FRAG), s: (off % FRAG) / 512 });
      const put = (name, i, tag) => {
        const old = regs[name][i];
        if (old && !old.used) res.wasted++;
        regs[name][i] = { ...tag, used: false };
      };
      const fail = (why) => {
        if (res.wrong.length < 5) res.wrong.push(why);
      };
      const imma = (mb, n, u, half, aw, bw, s) => {
        res.imma++;
        aw.used = true;
        bw.used = true;
        const want = { tile: cur, s };
        if (aw.op !== 'A' || bw.op !== 'B' || aw.tile !== cur || bw.tile !== cur || aw.s !== bw.s) fail(`IMMA on ${JSON.stringify([aw, bw])}`);
        if (aw.blk !== 2 * (mb >> 1) + half || bw.blk !== 2 * (n >> 2) + half) fail(`IMMA blocks ${mb}.${n} half ${half}`);
        const h = acc[(mb * 8 + n) * 2 + u];
        const step = aw.s * 2 + half;
        if (h.length !== step) fail(`accumulator ${mb}.${n}.${u} at ${h.length}, IMMA of ${step} (${JSON.stringify(want)})`);
        h.push(step);
      };
      const readout = (rl, c, nb) => {
        res.readouts++;
        for (const mb of [2 * rl, 2 * rl + 1]) {
          for (const n of nb) {
            for (const u of [0, 1]) if (acc[(mb * 8 + n) * 2 + u].length !== (c + 1) * KSTEPS * 2) res.badReadout++;
          }
        }
      };
      const kstep = (ch, t, ro, last, dch = 0, hiReady = false) => {
        const aoThis = arun + (dch * KSTEPS + t) * 512;
        const boThis = brun + (dch * KSTEPS + t) * 512;
        const ao = last ? aoffn : arun + (dch * KSTEPS + t + 1) * 512;
        const bo = last ? boffn : brun + (dch * KSTEPS + t + 1) * 512;
        if (!hiReady || mutate === 'no-hi-ready') for (let q = 0; q < 2; q++) put('bfh', q, decode('B', boThis + (2 * q + 1) * FRAG));
        const s = Math.floor((aoThis % TILE % FRAG) / 512);
        const sub = (q, hi, ro_) => {
          const aq = regs.af[q];
          for (let j = 0; j < 4; j++) {
            const bq = (hi ? regs.bfh : regs.bfl)[j >> 1];
            if (ro_) readout(q, ch - 1, [2 * j, 2 * j + 1]);
            for (const odd of [0, 1]) {
              for (const mbo of [0, 1]) for (const u of [0, 1]) imma(2 * q + mbo, 2 * j + odd, u, hi, aq, bq, s);
            }
          }
        };
        for (let q = 0; q < 3; q++) {
          sub(q, 0, ro);
          put('af', q, decode('A', (mutate === 'k-lo-loads-next' ? ao : aoThis) + (2 * q + 1) * FRAG));
        }
        for (let q = 0; q < 2; q++) put('bfl', q, decode('B', bo + 2 * q * FRAG));
        for (let q = 0; q < 3; q++) {
          sub(q, 1, false);
          put('af', q, decode('A', ao + 2 * q * FRAG));
        }
        if (last && mutate !== 'no-hi-tail') for (let q = 0; q < 2; q++) put('bfh', q, decode('B', boffn + (2 * q + 1) * FRAG));
      };
      const tile = (nextSrcs) => {
        res.tiles++;
        for (const h of acc) h.length = 0;
        arun = aoff;
        brun = boff;
        kstep(0, 0, false, false, 0, true);
        for (let t = 1; t < KSTEPS; t++) kstep(0, t, false, false);
        for (let chunk = 1; chunk < CHUNKS; chunk++) {
          arun = aoff + chunk * (KSTEPS * 512);
          brun = boff + chunk * (KSTEPS * 512);
          kstep(chunk, 0, true, false);
          const tend = chunk + 1 < CHUNKS ? KSTEPS : (mutate === 'tend' ? KSTEPS : KSTEPS - 1);
          for (let t = 1; t < tend; t++) kstep(chunk, t, false, false);
        }
        nextSrcs();
        kstep(CHUNKS - 1, KSTEPS - 1, false, true);
        for (let rl = 0; rl < RPL; rl++) readout(rl, CHUNKS - 1, [0, 1, 2, 3, 4, 5, 6, 7]);
      };
      // The persistent loop: the prologue loads the first tile's k-step 0, each tile's
      // tail the next one's (the block's own first tile after the last, as next_srcs's
      // clamp does).
      cur = order[0];
      aoff = cur * TILE;
      boff = cur * TILE;
      for (let q = 0; q < 3; q++) put('af', q, decode('A', aoff + 2 * q * FRAG));
      for (let q = 0; q < 2; q++) {
        put('bfl', q, decode('B', boff + 2 * q * FRAG));
        put('bfh', q, decode('B', boff + (2 * q + 1) * FRAG));
      }
      for (let i = 0; i < order.length; i++) {
        const next = i + 1 < order.length ? order[i + 1] : order[0];
        tile(() => {
          aoffn = next * TILE;
          boffn = next * TILE;
        });
        aoff = aoffn;
        boff = boffn;
        cur = i + 1 < order.length ? next : -1;
      }
      return res;
    }
    for (const order of [[3], [0, 1], [2, 7, 5, 9]]) {
      const r = runTiles(order);
      expect([order, r]).toEqual([order, {
        imma: order.length * STEPS * 2 * 96, wrong: [], wasted: 0, readouts: order.length * (15 * 3 * 4 + 3),
        badReadout: 0, tiles: order.length,
      }]);
    }
    // And it sees each of these: k-step 0 reloading its k-hi B (the tail's go unread), a
    // tail without the next tile's k-hi B, the k-lo pass loading the next k-step's k-hi A,
    // and a last chunk with four k-steps before the peeled one.
    expect(runTiles([1, 2], 'no-hi-ready').wasted).toBeGreaterThan(0);
    expect(runTiles([1, 2], 'no-hi-tail').wrong.length).toBeGreaterThan(0);
    expect(runTiles([1, 2], 'k-lo-loads-next').wrong.length).toBeGreaterThan(0);
    expect(runTiles([1, 2], 'tend').wrong.length).toBeGreaterThan(0);
  });

  test('the hand-off: each hasher hashes its own tile, from its own words, in any warp order', () => {
    // The persistent loop as the build has it, pinned here; the simulation below runs it.
    const b = build('TALL=1');
    const LOOP = 'for (uint32_t v = blockIdx.x; v < tiles; it++) {';
    inOrder(b.lines, ['if (threadIdx.x == 0u) sNext[0] = gridDim.x + atomicAdd(tile_ctr, 1u);', 'tile_prologue();',
      '__shared__ uint32_t sCur[2];', 'if (threadIdx.x == 0u) sCur[0] = blockIdx.x;', 'tbuf = TWORDS;',
      '__syncthreads();', 'uint32_t it = 0u;', LOOP]);
    expect(block(b.lines, LOOP)).toEqual([LOOP, 'tbuf ^= TWORDS;', 'asm volatile("" : "+r"(tbuf));',
      'const uint32_t par = tbuf ? 1u : 0u;', 'uint32_t vgrab = 0u;',
      'if (threadIdx.x == 0u) vgrab = gridDim.x + atomicAdd(tile_ctr, 1u);', 'uint32_t vnext = 0u;', 'tile([&]() {',
      'vnext = sNext[par];', 'uint32_t rbgn, cbgn;', 'tile_coords(vnext < tiles ? vnext : blockIdx.x, rbgn, cbgn);',
      'aoffn = a_off(rbgn);', 'boffn = b_off(cbgn);', '});', 'if (threadIdx.x == 0u) sNext[par ^ 1u] = vgrab;',
      '__syncthreads();', 'if (threadIdx.x == 0u) sCur[par ^ 1u] = vnext;',
      'hash_tile(*(volatile uint32_t *)&sCur[par]);', 'aoff = aoffn;', 'boff = boffn;', 'v = vnext;', '}']);
    expect(b.lines.filter((l) => /\btbuf =|\btbuf \^=/.test(l))).toEqual(['uint32_t tbuf = 0u;', 'tbuf = TWORDS;', 'tbuf ^= TWORDS;']);
    // The readout stores chunk c - 1 in chunk c's first k-step, next_srcs reads sNext
    // before the last k-step, and the last chunk's words come after it (pinned above).

    // A block's 256 threads, each a generator in its own program order, yielding after
    // each shared access and at each barrier. Between barriers the scheduler runs them in a
    // random order, or thread 0 first ('t0') or last ('last0'). Words carry the tile they
    // were summed from. old: thread 0 writes sCur before the barrier, as round 7's build
    // (988c3cdf7b04) did.
    const regionsOf = Array.from({ length: THREADS }, () => []);
    for (let t = 0; t < THREADS; t++) {
      const { warp, lane } = ids(t);
      if (((lane >> 2) & 3) !== 0) continue;
      for (let rl = 0; rl < RPL; rl++) regionsOf[t].push(warp * 24 + (lane & 3) * 6 + 2 * rl + ((lane >> 4) & 1));
    }
    const hashOf = Array.from({ length: THREADS }, (_, t) => hashedBy(t));
    const at = (tbuf, L, c) => tbuf + L * 16 + (c ^ ((L >> 1) & 15));
    function handoff({ G, blk, tiles, seed, policy = 'random', old = false }) {
      const rand = xorshift(seed);
      const sT = new Array(2 * TWORDS).fill('garbage');
      const sNext = ['garbage', 'garbage'];
      const sCur = ['garbage', 'garbage'];
      let counter = 0;
      const grab = () => G + counter++;
      const res = { hashed: new Map(), wrongTile: 0, wrongWord: 0, tilesRun: [] };
      function* thread(t) {
        let tbuf = 0;
        if (t === 0) sNext[0] = grab();
        yield 'step';
        if (t === 0) sCur[0] = blk;
        tbuf = TWORDS;
        yield 'bar';
        let cur = blk;  // the tile whose operands the thread holds
        for (let v = blk; v < tiles;) {
          tbuf ^= TWORDS;
          const par = tbuf ? 1 : 0;
          let vgrab = 0;
          if (t === 0) {
            vgrab = grab();
            res.tilesRun.push(cur);
          }
          // The tile: chunks 0-14's words, next_srcs, chunk 15's.
          for (let c = 0; c < CHUNKS - 1; c++) {
            for (const L of regionsOf[t]) {
              sT[at(tbuf, L, c)] = `${cur}.${L}.${c}`;
              yield 'step';
            }
          }
          const vnext = sNext[par];
          yield 'step';
          for (const L of regionsOf[t]) {
            sT[at(tbuf, L, CHUNKS - 1)] = `${cur}.${L}.${CHUNKS - 1}`;
            yield 'step';
          }
          if (t === 0) {
            sNext[par ^ 1] = vgrab;
            if (old) sCur[par ^ 1] = vnext;
          }
          yield 'bar';
          if (t === 0 && !old) {
            sCur[par ^ 1] = vnext;
            yield 'step';
          }
          if (hashOf[t].length) {
            const vt = sCur[par];
            yield 'step';
            if (vt !== cur) res.wrongTile++;
            for (const L of hashOf[t]) {
              res.hashed.set(`${vt}.${L}`, (res.hashed.get(`${vt}.${L}`) || 0) + 1);
              for (let c = 0; c < CHUNKS; c++) {
                if (sT[at(tbuf, L, c)] !== `${cur}.${L}.${c}`) res.wrongWord++;
                yield 'step';
              }
            }
          }
          v = vnext;
          cur = vnext < tiles ? vnext : blk;
        }
      }
      const gens = Array.from({ length: THREADS }, (_, t) => thread(t));
      let live = gens.map((g, i) => i);
      while (live.length) {
        const active = live.slice();
        const finished = new Set();
        const pick = () => {
          if (policy === 't0' && active.includes(0)) return active.indexOf(0);
          if (policy === 'last0' && active.length > 1) {
            const k = rand() % (active.length - 1);
            return active[k] === 0 ? active.length - 1 : k;
          }
          return rand() % active.length;
        };
        while (active.length) {
          const k = pick();
          const i = active[k];
          const r = gens[i].next();
          if (r.done) finished.add(i);
          if (r.done || r.value === 'bar') active.splice(k, 1);
        }
        live = live.filter((i) => !finished.has(i));
      }
      return res;
    }
    const expectedTiles = (G, blk, tiles) => [blk, ...Array.from({ length: Math.max(0, tiles - G) }, (_, i) => G + i)];
    const clean = (r, want) => {
      expect([r.wrongTile, r.wrongWord]).toEqual([0, 0]);
      expect(r.tilesRun).toEqual(want);
      for (const v of want) {
        for (let L = 0; L < REGIONS; L++) expect([v, L, r.hashed.get(`${v}.${L}`)]).toEqual([v, L, 1]);
      }
      expect(r.hashed.size).toBe(want.length * REGIONS);
    };
    // Block 1 of a grid of 4: tile 1 first, then the tiles the counter hands out (here all
    // of them: the other blocks are not run), for 1 to 6 tiles a block.
    for (const [tiles, seed] of [[2, 3], [5, 7], [7, 11], [9, 5]]) {
      for (const policy of ['random', 't0', 'last0']) {
        clean(handoff({ G: 4, blk: 1, tiles, seed, policy }), expectedTiles(4, 1, tiles));
      }
    }
    // Written before the barrier, the slot can be overwritten before a slow hasher reads
    // it: thread 0, run first after a hand-off, reaches the next one first.
    expect(handoff({ G: 1, blk: 0, tiles: 4, seed: 9, policy: 't0', old: true }).wrongTile).toBeGreaterThan(0);
    clean(handoff({ G: 1, blk: 0, tiles: 4, seed: 9, policy: 't0' }), expectedTiles(1, 0, 4));
  });

  test('PEARL_ABLATE_TRANSCRIPT_HASH returns in both passes', () => {
    const ht = block(build('TALL=1 PEARL_ABLATE_TRANSCRIPT_HASH').lines, HASH_TILE);
    const loop = block(ht, ROLE[4]);
    const [lam] = inOrder(loop, ['[&]() {', 'return;',
      'if (pearl_transcript_msw(test.key, tm, test.hash_big_endian) > test.target_w[0]) return;']);
    expect(loop[lam + 1]).toBe('return;');
    expect(block(build('TALL=1').lines, HASH_TILE).includes('return;')).toBe(false);
  });

  test('the host counts the tiles, gives the shared and names the fold from the same switch', () => {
    expect(HOST).toContain('#if PEARL_BD_TALL' + NL
      + '  // Turing\'s B-direct fold built with PEARL_BD_TALL takes a ragged last row group, as' + NL
      + '  // the tall fold does (below), so only its columns have to tile.' + NL
      + '  const bool rowsTile = ctx->foldBDirect || rowBlocks % warpRows == 0;' + NL + '#else' + NL
      + '  const bool rowsTile = rowBlocks % warpRows == 0;' + NL + '#endif' + NL
      + '  if (!ctx->foldTall && (warpsPerBlock % warpRows != 0 || !rowsTile');
    expect(HOST).toContain('  const unsigned tallRowGroups =' + NL
      + '      (unsigned)((ctx->rowsValid + PEARL_TALL_ROW_OFFSETS - 1u) / PEARL_TALL_ROW_OFFSETS);');
    expect(HOST).toContain('#if PEARL_BD_TALL' + NL
      + '  // PEARL_BD_TALL: 192x256 tiles, 12 row offsets by 16 column offsets, the last row group' + NL
      + '  // ragged, as the tall fold\'s. The fold works out the same row groups from rows_valid.' + NL
      + '  if (ctx->foldBDirect) tiles = tallRowGroups * (unsigned)(colBlocks / warpCols);' + NL + '#endif' + NL);
    expect(HOST).toContain('  ctx->rowsValid = profile->m / PEARL_ROWS_COUNT;');
    expect(HEADER).toContain('#define PEARL_TALL_ROW_OFFSETS (PEARL_TALL_BM / 16u)             // 12');
    expect(HOST).toContain('#if PEARL_BD_TALL' + NL
      + '  // PEARL_BD_TALL: a warp holds six row offsets\' regions, not four, so its two' + NL
      + '  // transcript buffers are 12 KB each: 24 KB. It has no A stages and no split hash.' + NL
      + '  if (ctx->foldBDirect)' + NL
      + '    smem = (size_t)PEARL_BD_T_BUFS * warpsPerBlock * PEARL_TALL_ROW_TILES * PEARL_WMMA_COL_BLK' + NL
      + '           * PEARL_JACKPOT_BUCKETS * 4u;' + NL + '#endif' + NL);
    // 24 KB: the fold's two buffers.
    const bufs = config('TALL=1').value('PEARL_BD_T_BUFS');
    expect(bufs * 8 * 6 * 4 * 16 * 4).toBe(2 * TWORDS * 4);
    expect(2 * TWORDS * 4).toBe(24576);
    // The host's tiles are the fold's row groups by its column groups, every shape.
    for (const rowsValid of [8192, 512, 256, 32, 40, 13]) {
      for (const colBatch of [16, 32, 2048]) {
        const tiles = hostTiles(rowsValid, colBatch);
        expect([rowsValid, colBatch, tiles / rowGroupsOf(rowsValid)]).toEqual([rowsValid, colBatch, colBatch / 16]);
      }
    }
    // The bench's name: the tile's shape.
    expect(HOST).toContain('             "%s%s%s %s, %d row group%s, %s, band %u, m8n8k16 (Turing)",');
    expect(HOST).toContain('             PEARL_BD_TALL ? "192x256, 8 warps of 96x64" : "128x256, 8 warps of 64x64",');
    // compilers.sh builds the host with each build's flags.
    expect(COMPILERS).toContain('nvcc $COMMON $flags -c "$K/src/pearl_host.cu" -o "$K/cuda-build/pearl_host.o"');
  });

  test('the host\'s and the kernel\'s changes are all inside #if PEARL_BD_TALL', () => {
    // What the host does only with the switch, and the default's line it replaces.
    const TALL_ONLY = ['ctx->foldBDirect || rowBlocks % warpRows', 'tallRowGroups * (unsigned)(colBlocks / warpCols)',
      'ctx->profile.m % 32u != 0u', 'PEARL_TALL_ROW_TILES', 'pearl_bd_tall_'];
    const DEFAULT_ONLY = ['  const bool rowsTile = rowBlocks % warpRows == 0;'];
    const off = pickTall(HOST, 0);
    const on = pickTall(HOST, 1);
    for (const t of TALL_ONLY) expect([t, off.includes(t), on.includes(t)]).toEqual([t, false, true]);
    for (const t of DEFAULT_ONLY) expect([t, off.includes(t), on.includes(t)]).toEqual([t, true, false]);
    expect(HOST.split(NL).filter((l) => l === '#if PEARL_BD_TALL').length).toBe(5);
    // pearl_kernel.cu's host code: only the pairing below.
    const KERNEL_OWN = ['extern "C" const unsigned pearl_bd_tall_host;',
      'extern "C" const unsigned pearl_bd_tall_fold = 1u;',
      'extern "C" const unsigned *const pearl_bd_tall_fold_needs = &pearl_bd_tall_host;'];
    expect(pickTall(KERNEL, 1).split(NL).filter((l) => !pickTall(KERNEL, 0).split(NL).includes(l))
      .filter((l) => !l.startsWith('//'))).toEqual(KERNEL_OWN);
    expect(KERNEL.split(NL).filter((l) => /PEARL_BD_TALL\b/.test(l) && l.startsWith('#'))).toEqual(['#if PEARL_BD_TALL']);
    // Built with the switch, each side defines one name and needs the other's, so a host
    // and a fold built with and without it do not link or load together.
    expect(HOST).toContain('#if PEARL_BD_TALL' + NL + '// The fold and this host must both be built');
    expect(on).toContain('extern "C" const unsigned pearl_bd_tall_fold;' + NL
      + 'extern "C" const unsigned pearl_bd_tall_host = 1u;' + NL
      + 'extern "C" const unsigned *const pearl_bd_tall_host_needs = &pearl_bd_tall_fold;');
  });
});
