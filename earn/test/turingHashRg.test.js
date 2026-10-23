'use strict';

const { spawnSync } = require('child_process');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

// Turing's hashers from the second row group (PEARL_BD_HASH_RG, off by default). At each
// tile's hand-off one warp on each scheduler (warp & 3) hashes 32 of the tile's 128
// regions while the scheduler's other warp starts the next tile. As shipped those are
// warps 0, 1, 6 and 7. With the switch they are warps 4-7, the second row group. hash_tile
// keeps the default's test and flips it with sid < 64 on a copy of sid behind an empty
// asm (osid), and takes region L = osid.
//
// Nothing here can run the device code. A small C preprocessor picks out the code a build
// with given switches compiles. The role's expressions are read out of that code and run
// on the CPU for all 256 threads, beside a JS copy pinned to the source lines. The checks:
// - the default and the guards;
// - each region hashed once a tile, by one whole warp on each scheduler, and by the same
//   lane of the same scheduler as without the switch;
// - each region read from the same words and given the same number, over every tile of
//   a launch, so the host reports the same hits once it sorts them;
// - outside the role, every build HASH_RG stacks with is line for line the build without;
// - with LAST_RO too, in a random thread order, each word a hasher reads is its own
//   tile's, for 1 to 7 tiles a block;
// - the bench's fold name.
// And the second sm_75 kernel (PEARL_BD_RG_KERNEL), pearl_tile_fold_wmma_rg, which runs
// HASH_RG's code on cards with 4 MB of L2 or more without the switch:
// - where it is built, and the policy;
// - its fold is HASH_RG's line for line, and the default kernel's is what it was;
// - the host's L2 rule, the sweep's override and the fold name;
// - every place the host takes the fold kernel's address uses the kernel it picked;
// - the probes and CI see it.

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
const TURING = path.join(SRC, '..', 'probes', 'turing');
const rawRead = (file) => fs.readFileSync(file, 'utf8').split(String.fromCharCode(13) + NL).join(NL);
const COMPILERS = rawRead(path.join(TURING, 'compilers.sh'));
const LAUNCH = rawRead(path.join(TURING, 'launch.sh'));
const CI = rawRead(path.join(SRC, '..', '..', '..', '.github', 'workflows', 'native-core.yml'));

// ---- C expressions ----------------------------------------------------------------

const TOKEN = /\s*(?:(0[xX][0-9a-fA-F]+|[0-9]+)[uUlL]*|([A-Za-z_]\w*(?:\.\w+)*)|(&&|\|\||==|!=|<=|>=|<<|>>|[-+*/%<>!~&|^?:()]))/y;
const PREC = {
  '||': 1, '&&': 2, '|': 3, '^': 4, '&': 5, '==': 6, '!=': 6, '<': 7, '<=': 7, '>': 7, '>=': 7,
  '<<': 8, '>>': 8, '+': 9, '-': 9, '*': 10, '/': 10, '%': 10,
};
const b01 = (x) => (x ? 1 : 0);
// The binary operators: as the preprocessor's integers, or as uint32_t code.
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

// #if's names: a macro's value, 0 for any other name.
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
function codeScope(env, defs) {
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

// The code lines a compile with these macros keeps (trimmed, without comments or blank
// lines), its #error messages, and the macros it ends with (name -> body, null for a
// function-like one). #pragma lines stay as code.
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

// -D flags, PEARL_BD_ ones by their short names: D('HASH_RG=1 LAST_RO=1'). A bare name
// is 1.
function D(spec) {
  const defs = new Map();
  for (const f of spec.split(' ').filter(Boolean)) {
    const [name, v = '1'] = f.split('=');
    defs.set(name.startsWith('PEARL_') ? name : 'PEARL_BD_' + name, v);
  }
  return defs;
}

// Every switch off on the default fold: the default before LEAN and HIT_COORDS went on.
const OFF = 'LEAN=0 HIT_COORDS=0';

// From the line holding from to the end of the line holding to.
function slice(src, from, to) {
  const at = src.indexOf(from);
  expect([from, at >= 0]).toEqual([from, true]);
  const end = src.indexOf(to, at);
  expect([to, end > at]).toEqual([to, true]);
  const eol = src.indexOf(NL, end);
  return src.slice(src.lastIndexOf(NL, at) + 1, eol < 0 ? src.length : eol);
}

// The guards a build of the fold goes through: pearl_config.h's Turing switches (their
// defaults and #error lines, HASH_RG's among them) and the fold's own #error lines.
const GUARDS = slice(HEADER, '#ifndef PEARL_TURING_WIDE' + NL, '#define PEARL_BD_SPLIT_SMEM ') + NL
  + slice(FOLD, '#if PEARL_BD_A_DIRECT && (defined(PEARL_ABLATE_BARRIER)',
    '#error "PEARL_ABLATE_TRANSCRIPT_HASH prices the hash after the hand-off') + NL + '#endif';

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

// The fold as the compiler instantiates pearl_fold_bd<kHashRg>: the template line and its
// static_assert dropped, and the hashers' if constexpr resolved to the branch kHashRg
// picks, as the const declarations it amounts to. pearl_tile_fold_wmma passes
// PEARL_BD_HASH_RG, pearl_tile_fold_wmma_rg (the second sm_75 kernel) true.
const RG_TEMPLATE = 'template <bool kHashRg>' + NL;
const RG_ASSERT = '  static_assert(!kHashRg || PEARL_BD_HASH_RG_OK,' + NL
  + '                "the second row group hashes only where PEARL_BD_HASH_RG builds");' + NL;
const ROLE_HEAD = '    bool hasher;' + NL + '    uint32_t L;' + NL + '    if constexpr (kHashRg) {' + NL;
const ROLE_ELSE = NL + '    } else {' + NL;
const ROLE_END = NL + '    }' + NL;
function instantiate(text, rg) {
  for (const part of [RG_TEMPLATE, RG_ASSERT, ROLE_HEAD]) expect([part, text.split(part).length - 1]).toEqual([part, 1]);
  const t = text.replace(RG_TEMPLATE, '').replace(RG_ASSERT, '');
  const at = t.indexOf(ROLE_HEAD);
  const mid = t.indexOf(ROLE_ELSE, at);
  const end = t.indexOf(ROLE_END, mid + ROLE_ELSE.length);
  const branch = rg ? t.slice(at + ROLE_HEAD.length, mid) : t.slice(mid + ROLE_ELSE.length, end);
  const lines = branch.split(NL).map((l) => l.slice(2).replace(/^( *)hasher = /, '$1const bool hasher = ')
    .replace(/^( *)L = /, '$1const uint32_t L = '));
  return t.slice(0, at) + lines.join(NL) + NL + t.slice(end + ROLE_END.length);
}

// The template argument a kernel's sm_75 body passes to pearl_fold_bd, read from
// pearl_kernel.cu: one call in the #if PEARL_TURING_BD_BODY part of that kernel.
const FOLD_CALL = /^pearl_fold_bd<([^>]+)>\(Aprime, Bprime, col_off, rows_valid, tiles, test, hits\);$/;
function kernelArg(name) {
  const lines = KERNEL.split(NL).map((l) => l.trim());
  const at = lines.indexOf(`extern "C" __global__ __launch_bounds__(PEARL_FOLD_THREADS) void ${name}(`);
  expect([name, at > 0]).toEqual([name, true]);
  const body = lines.indexOf('#if PEARL_TURING_BD_BODY', at);
  const calls = lines.slice(body, lines.indexOf('#else', body)).map((l) => FOLD_CALL.exec(l)).filter(Boolean);
  expect([name, calls.length]).toEqual([name, 1]);
  return calls[0][1];
}

// The fold an sm_75 compile with these flags builds: its code lines, trimmed. kernel
// 'default' is pearl_tile_fold_wmma's, 'rg' pearl_tile_fold_wmma_rg's. kHashRg is the
// argument that kernel passes, worked out under these flags.
const builds = new Map();
function build(spec, kernel = 'default') {
  const key = kernel + '|' + spec;
  if (!builds.has(key)) {
    const c = config(spec);
    expect([spec, c.errors]).toEqual([spec, []]);
    if (kernel === 'rg') expect([spec, c.value('PEARL_BD_RG_KERNEL_BUILT')]).toEqual([spec, 1]);
    const arg = kernelArg(kernel === 'rg' ? 'pearl_tile_fold_wmma_rg' : 'pearl_tile_fold_wmma');
    const rg = arg === 'true' || (arg !== 'false' && compile(arg, false)(macroScope(c.defs)) !== 0);
    const r = preprocess(instantiate(FOLD, rg), new Map([...c.defs, ...D('PEARL_STAGE_REGS=1')]));
    expect([spec, r.errors]).toEqual([spec, []]);
    builds.set(key, { lines: r.lines, text: r.lines.join(NL), defs: r.defs });
  }
  return builds.get(key);
}

// The braces a line opens less those it closes, outside string literals.
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

// The lines of the block that opens at the only line equal to head, to its closing brace.
function block(lines, head) {
  const at = lines.indexOf(head);
  expect([head, at >= 0, lines.lastIndexOf(head) === at]).toEqual([head, true, true]);
  let d = 0;
  for (let i = at; i < lines.length; i++) {
    d += depth(lines[i]);
    if (d === 0) return lines.slice(at, i + 1);
  }
  throw new Error(`"${head}" does not close`);
}

// A block's lines as one line, single spaces.
function flatLines(lines) {
  return lines.join(' ').replace(/\s+/g, ' ').trim();
}

// Where each of parts is in lines, the first at or after from, each after the one before.
function inOrder(lines, parts, from = 0) {
  let at = from - 1;
  return parts.map((p) => {
    const i = lines.indexOf(p, at + 1);
    expect([p, i > at]).toEqual([p, true]);
    at = i;
    return i;
  });
}

// ---- The fold, copied ---------------------------------------------------------------

// The block's shape (pinned below): 256 threads, 8 warps of 16 regions, 16 chunks, a
// transcript buffer of 2048 words.
const THREADS = 256;
const REGIONS = 128;
const CHUNKS = 16;
const TWORDS = 2048;

// A thread's names, from pearl_fold_bd's first lines: warp w is row group w >> 2 and runs
// on scheduler w & 3, and sid is the thread's place in its row group.
function ids(t) {
  const warp = t >> 5;
  return { 'threadIdx.x': t, warp, lane: t & 31, wr: warp >> 2, wc: warp & 3, sid: t & 127 };
}

// hash_tile's role, copied: does thread t hash, and which region. osid is sid through
// asm("" : "+r"(osid)), which emits nothing and so leaves it as it is.
const ROLE = {
  dflt: (t) => {
    const { wr, sid } = ids(t);
    return { hasher: wr ? sid >= 64 : sid < 64, L: wr * 64 + (sid & 63) };
  },
  rg: (t) => {
    const { wr, sid } = ids(t);
    const osid = sid;
    return { hasher: (wr ? sid >= 64 : sid < 64) !== (osid < 64), L: osid };
  },
};
// The lines those copy, as a build has them.
const ROLE_LINES = {
  rg: ['uint32_t osid = sid;', 'asm("" : "+r"(osid));',
    'const bool hasher = (wr ? sid >= 64u : sid < 64u) != (osid < 64u);', 'const uint32_t L = osid;'],
  dflt: ['const bool hasher = wr ? sid >= 64u : sid < 64u;', 'const uint32_t L = wr * 64u + (sid & 63u);'],
};

// Word c of region L in buffer tbuf: readout_st stores it there and hash_tile reads it.
const wordOf = (tbuf, L, c) => tbuf + L * 16 + (c ^ ((L >> 1) & 15));
// readout_st's region rl of a lane: lanes 0-3 and 16-19 store, two regions each.
const writes = (lane) => ((lane >> 2) & 3) === 0;
const writerRegion = (warp, lane, rl) => warp * 16 + (lane & 3) * 4 + 2 * rl + ((lane >> 4) & 1);

// region_of: the number a hit reports for region L of the tile at row group rbg and
// column group cbg.
function regionOf(L, rbg, cbg, rowsValid) {
  const ow = Math.floor(L / 16);
  const ocb = Math.floor((L % 16) / 4);
  const oreg = L % 4;
  const orb = rbg * 2 + Math.floor(ow / 4);
  const ocgb = cbg * 4 + (ow % 4);
  return (ocgb * 4 + ocb) * rowsValid + orb * 4 + oreg;
}

// tile_coords, column bands (PEARL_BD_WALK 1, the default): tile v's row group and column
// group, for a launch of rowsValid rows and tiles tiles, bandDepth column groups a band.
function tileCoords(v, { rowsValid, tiles, bandDepth }) {
  const rowBlockGroups = Math.floor(rowsValid / 4 / 2);
  const colBlockGroups = Math.floor(tiles / rowBlockGroups);
  const bandTiles = bandDepth * rowBlockGroups;
  const band = Math.floor(v / bandTiles);
  const inBand = v % bandTiles;
  const bandFirst = band * bandDepth;
  const bandCols = colBlockGroups - bandFirst < bandDepth ? colBlockGroups - bandFirst : bandDepth;
  return { cbg: bandFirst + (inBand % bandCols), rbg: Math.floor(inBand / bandCols) };
}

// The hash_tile lambda of a build, whichever signature HIT_COORDS gives it.
const HASH_TILE_V = 'auto hash_tile = [&](uint32_t vtile) {';
const HASH_TILE_RC = 'auto hash_tile = [&](uint32_t rbg, uint32_t cbg) {';
function hashTile(b) {
  return block(b.lines, b.lines.includes(HASH_TILE_V) ? HASH_TILE_V : HASH_TILE_RC);
}

// The role a build compiles, read out of its hash_tile and run for thread t.
function roleOf(spec) {
  const ht = hashTile(build(spec));
  const at = ht.findIndex((l) => l.startsWith('const bool hasher = '));
  expect(at).toBeGreaterThan(0);
  const hasher = /^const bool hasher = (.+);$/.exec(ht[at])[1];
  const L = /^const uint32_t L = (.+);$/.exec(ht[at + 1])[1];
  expect(ht[at + 2]).toBe('if (hasher) {');
  const osid = ht[at - 2] === ROLE_LINES.rg[0];
  if (osid) expect(ht[at - 1]).toBe(ROLE_LINES.rg[1]);
  const fh = compile(hasher, true);
  const fl = compile(L, true);
  const defs = build(spec).defs;
  return {
    osid,
    lines: ht.slice(osid ? at - 2 : at, at + 2),
    of: (t) => {
      const env = { ...ids(t) };
      if (osid) env.osid = env.sid;
      const scope = codeScope(env, defs);
      return { hasher: fh(scope) === 1, L: fl(scope) };
    },
  };
}

// A cheap stand-in for the hash's test: a 32-bit mix of the 16 words.
function mix(words) {
  let h = 0x811c9dc5;
  for (const w of words) {
    h = Math.imul(h ^ w, 0x01000193) >>> 0;
    h ^= h >>> 15;
  }
  return h >>> 0;
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

// Runs the threads' queues of steps, each queue in its order, the queues interleaved at
// random.
function interleave(queues, rand) {
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
}

// The readout's store and the hash's read as a build has them, run on the CPU: which
// lanes store, the region a lane stores for rl, the index it stores word c of region L
// at in buffer tbuf, and the index a hasher reads it from.
const STORE_HEAD = 'auto readout_st = [&](uint32_t rl, uint32_t c, uint32_t x) {';
function transcriptCode(spec) {
  const b = build(spec);
  const st = flatLines(block(b.lines, STORE_HEAD));
  const m = /^auto readout_st = \[&\]\(uint32_t rl, uint32_t c, uint32_t x\) \{ if \((.+?)\) \{ const uint32_t L = (.+?); sT\[(.+?)\] = x; \} \};$/
    .exec(st);
  expect([spec, m !== null]).toEqual([spec, true]);
  const read = hashTile(b).map((l) => /^for \(uint32_t c = 0; c < 16u; c\+\+\) tm\[c\] = sT\[(.+)\];$/.exec(l))
    .filter(Boolean);
  expect([spec, read.length]).toEqual([spec, 1]);
  const run = (expr) => {
    const f = compile(expr, true);
    return (env) => f(codeScope({ regions_per_warp: 4, warp_regions: 16, ...env }, b.defs));
  };
  const writer = run(m[1]);
  const region = run(m[2]);
  const store = run(m[3]);
  const readAt = run(read[0][1]);
  return {
    expr: { writer: m[1], region: m[2], store: m[3], read: read[0][1] },
    writer: (lane) => writer({ lane }) === 1,
    region: (warp, lane, rl) => region({ warp, lane, rl }),
    storeAt: (tbuf, L, c) => store({ tbuf, L, c }),
    readAt: (tbuf, L, c) => readAt({ tbuf, L, c }),
  };
}

// One block's tiles, thread by thread, as the persistent loop runs them (pinned in the
// tests). Between two barriers the threads run in a random order, each in its own program
// order. In iteration it a thread stores tile it's words into buffer tbuf(it): chunks
// 0-14 in tile() and chunk 15 at the tile's end, or with LAST_RO in tile_head(true), the
// next tile's k-steps 0-1 before the hand-off. Then the hand-off barrier, and a hasher
// reads its region of tile it, still from tbuf(it): tbuf moves on only at the loop's top.
// Each word is tagged with its tile, region and chunk. At each hand-off the tile's buffer
// must hold all of its words, every region of every tile must be hashed once, and every
// word a hasher reads must be its own tile's. mutate breaks one thing, to show the checks
// see it.
function simulate(spec, { tiles, seed, mutate = '' }) {
  const lastRo = config(spec).value('PEARL_BD_LAST_RO') === 1;
  const role = roleOf(spec);
  const tc = transcriptCode(spec);
  const roleAt = Array.from({ length: THREADS }, (_, t) => role.of(t));
  if (mutate === 'rg-with-default-L') {
    for (let t = 0; t < THREADS; t++) roleAt[t] = { hasher: roleAt[t].hasher, L: ROLE.dflt(t).L };
  }
  const tbufExpr = /^tbuf = (.+);$/.exec(build(spec).lines.find((l) => /^tbuf = /.test(l)))[1];
  const tbufOf = (it) => (mutate === 'one-buffer' ? 0 : compile(tbufExpr, true)(codeScope({ it, TWORDS }, new Map())));
  const rand = xorshift(seed);
  const sT = new Array(2 * TWORDS).fill('launch');
  const res = { read: 0, wrong: 0, outside: 0, badHandoff: 0, twice: 0, never: 0 };
  const hashed = new Map();
  const poke = (a, fn) => {
    if (a < 0 || a >= sT.length) res.outside++;
    else fn(a);
  };
  // Chunk 15's stores moved past the barrier ('head-after-barrier'): they open the
  // thread's next queue.
  const late = Array.from({ length: THREADS }, () => []);
  for (let it = 0; it <= tiles; it++) {
    const queues = late.map((q) => q.splice(0));
    for (let t = 0; t < THREADS; t++) {
      const { warp, lane } = ids(t);
      const { hasher, L } = roleAt[t];
      const reads = (tile, buf, into) => {
        hashed.set(`${tile}.${L}`, (hashed.get(`${tile}.${L}`) || 0) + 1);
        for (let c = 0; c < CHUNKS; c++) {
          const a = tc.readAt(buf, L, c);
          into.push(() => poke(a, () => {
            res.read++;
            if (sT[a] !== `${tile}.${L}.${c}`) res.wrong++;
          }));
        }
      };
      if (it >= 1 && hasher && mutate !== 'hash-before-barrier') {
        reads(it - 1, mutate === 'flip-before-hash' ? tbufOf(it) : tbufOf(it - 1), queues[t]);
      }
      if (it < tiles) {
        const buf = tbufOf(it);
        for (let c = 0; c < CHUNKS; c++) {
          if (!tc.writer(lane)) continue;
          for (let rl = 0; rl < 2; rl++) {
            const R = tc.region(warp, lane, rl);
            const a = tc.storeAt(buf, R, c);
            const into = c === CHUNKS - 1 && lastRo && mutate === 'head-after-barrier' ? late[t] : queues[t];
            into.push(() => poke(a, () => { sT[a] = `${it}.${R}.${c}`; }));
          }
        }
        if (hasher && mutate === 'hash-before-barrier') reads(it, buf, queues[t]);
      }
    }
    interleave(queues, rand);
    // Hand-off it: tile it's buffer holds its 2048 words.
    if (it < tiles) {
      const buf = tbufOf(it);
      let bad = 0;
      for (let L = 0; L < REGIONS; L++) {
        for (let c = 0; c < CHUNKS; c++) if (sT[tc.readAt(buf, L, c)] !== `${it}.${L}.${c}`) bad++;
      }
      if (bad) res.badHandoff++;
    }
  }
  for (let it = 0; it < tiles; it++) {
    for (let L = 0; L < REGIONS; L++) {
      const n = hashed.get(`${it}.${L}`) || 0;
      if (n === 0) res.never++;
      if (n > 1) res.twice++;
    }
  }
  return res;
}

// The builds HASH_RG is written for, as the build without the switch: the default fold,
// with or without LEAN's parts and HIT_COORDS, and with SNAKE, LAST_RO and ROWPAIR.
const BASES = ['', OFF, 'HIT_COORDS=0', 'SNAKE=1', 'LAST_RO=1', 'LAST_RO=1 SNAKE=1', 'ROWPAIR=1',
  'ROWPAIR=1 HIT_COORDS=0', 'ROWPAIR=1 SNAKE=1', 'ROWPAIR=1 LAST_RO=1', 'ROWPAIR=1 LAST_RO=1 SNAKE=1'];
const rg = (base) => `${base} HASH_RG=1`.trim();
// The other folds, and the messages of HASH_RG's three #error lines.
const OTHER_FOLDS = ['PERSIST=0', 'A_REALIGN=1', 'KORDER=0', 'A_DIRECT=0', 'GROUPS=1'];
const MSG_VALUE = '"PEARL_BD_HASH_RG is 0 or 1"';
const MSG_FOLD = '"PEARL_BD_HASH_RG is written for the default fold: A-direct, k-lo first, persistent, '
  + 'two row groups, two transcript buffers"';
const MSG_WITH = '"PEARL_BD_HASH_RG is not built with PEARL_BD_RED or the split hash: with RED ptxas '
  + 'reschedules the chunk loop, and the split hash pairs each scheduler\'s warps by the default\'s hashers"';

describe('Turing hashers from the second row group (PEARL_BD_HASH_RG)', () => {
  test('is off by default, and the header refuses it off the default fold, with RED and with the split hash', () => {
    // The evaluator, on C's precedence.
    const none = macroScope(new Map());
    expect(compile('1 | 4 == 5', false)(none)).toBe(1);
    expect(compile('2 + 3 * 4 << 1', false)(none)).toBe(28);
    expect(compile('!0 ? 6 : 7 - 7', false)(none)).toBe(6);
    expect(compile('(1 ? 70 >= 64u : 70 < 64u) != (70 < 64u)', true)(none)).toBe(1);
    expect(compile('0u - 1u', true)(none)).toBe(0xffffffff);
    expect(compile('defined(PEARL_X) && PEARL_X > 1', false)(macroScope(D('PEARL_X=2')))).toBe(1);

    expect(GUARDS.startsWith('#ifndef PEARL_TURING_WIDE' + NL)).toBe(true);
    const at = HEADER.indexOf('#ifndef PEARL_BD_HASH_RG' + NL);
    expect(at).toBeGreaterThanOrEqual(0);
    expect(HEADER.slice(at).split(NL).slice(0, 3)).toEqual(['#ifndef PEARL_BD_HASH_RG', '#define PEARL_BD_HASH_RG 0',
      '#endif']);
    expect(config('').value('PEARL_BD_HASH_RG')).toBe(0);
    expect(config('').value('PEARL_BD_DEFAULT_FOLD')).toBe(1);

    // It builds on the default fold, with each switch it stacks with, and with the
    // ablations that apply to it.
    for (const base of [...BASES, 'LEAN=1 HIT_COORDS=1', 'GATE=2', 'RUNBASE=0 TAIL_LDG=0',
      'PEARL_ABLATE_TRANSCRIPT_HASH', 'PEARL_ABLATE_READOUT', 'PEARL_ABLATE_TRANSCRIPT']) {
      const c = config(rg(base));
      expect([rg(base), c.errors, c.value('PEARL_BD_HASH_RG')]).toEqual([rg(base), [], 1]);
    }
    // It changes no other switch: every macro the header ends with has the same value,
    // but for the second kernel, which a build with the switch has no need of.
    for (const base of BASES) {
      const a = config(base);
      const b = config(rg(base));
      expect([...b.defs.keys()].sort()).toEqual([...a.defs.keys()].sort());
      for (const name of a.defs.keys()) {
        if (name !== 'PEARL_BD_HASH_RG' && name !== 'PEARL_BD_RG_KERNEL_BUILT') {
          expect([base, name, b.value(name)]).toEqual([base, name, a.value(name)]);
        }
      }
      expect([base, a.value('PEARL_BD_RG_KERNEL_BUILT'), b.value('PEARL_BD_RG_KERNEL_BUILT')]).toEqual([base, 1, 0]);
    }
    // Set to 0 it is the default, word for word.
    expect(build('HASH_RG=0').lines).toEqual(build('').lines);

    // Each refusal is exactly one #error, HASH_RG's, on a build that has none without it.
    const refused = (spec, msg) => {
      expect([spec, config(spec).errors]).toEqual([spec, [msg]]);
    };
    for (const v of ['2', '3', '-1']) refused(`HASH_RG=${v}`, MSG_VALUE);
    for (const other of OTHER_FOLDS) {
      expect([other, config(other).errors, config(other).value('PEARL_BD_DEFAULT_FOLD')]).toEqual([other, [], 0]);
      refused(`${other} HASH_RG=1`, MSG_FOLD);
      // The other folds build with it off.
      expect([other, config(`${other} HASH_RG=0`).errors]).toEqual([other, []]);
    }
    for (const s of ['RED=1', 'RED=1 GATE=1', 'RED=1 HIT_COORDS=0', 'HASH_SPLIT=1 RUNBASE=1 TAIL_LDG=1',
      'HASH_SPLIT=4 RUNBASE=1 TAIL_LDG=1', 'HASH_SPLIT=13 RUNBASE=1 TAIL_LDG=1']) {
      expect([s, config(s).errors, config(s).value('PEARL_BD_DEFAULT_FOLD')]).toEqual([s, [], 1]);
      refused(`${s} HASH_RG=1`, MSG_WITH);
      expect([s, config(`${s} HASH_RG=0`).errors]).toEqual([s, []]);
    }
    // The guards read only names the header has set by then: HASH_RG's own, and RED's,
    // the split hash's and the default fold's, defined above them.
    const guards = HEADER.split(NL).map((l, i) => [l, i]).filter(([l]) => /^#if .*PEARL_BD_HASH_RG/.test(l));
    expect(guards.map(([l]) => l)).toEqual(['#if PEARL_BD_HASH_RG != 0 && PEARL_BD_HASH_RG != 1',
      '#if PEARL_BD_HASH_RG && !PEARL_BD_DEFAULT_FOLD', '#if PEARL_BD_HASH_RG && (PEARL_BD_RED || PEARL_BD_HASH_SPLIT)']);
    const defLine = (name) => HEADER.split(NL).findIndex((l) => l.startsWith('#define ' + name + ' '));
    for (const [l, i] of guards) {
      for (const name of l.match(/PEARL_\w+/g)) {
        expect([name, defLine(name) >= 0 && defLine(name) < i]).toEqual([name, true]);
      }
    }
  });

  test('the fold builds the new role only with the switch, and the default\'s without it', () => {
    // The source, comments aside: the template's kHashRg picks HASH_RG's role first, the
    // default's after it. Nothing in the fold reads PEARL_BD_HASH_RG itself: the default
    // kernel passes it as kHashRg.
    const code = FOLD.split(NL).map((l) => uncomment(l)).filter((l) => !/^\s*$/.test(l));
    const at = code.indexOf('    if constexpr (kHashRg) {');
    expect(at).toBeGreaterThan(0);
    expect(code.lastIndexOf('    if constexpr (kHashRg) {')).toBe(at);
    const assign = (l) => l.replace(/^const bool hasher = /, 'hasher = ').replace(/^const uint32_t L = /, 'L = ');
    expect(code.slice(at - 8, at + 13)).toEqual([
      '#if PEARL_BD_HIT_COORDS', '  ' + HASH_TILE_V, '#else', '  ' + HASH_TILE_RC, '#endif',
      '#if PEARL_BD_GROUPS == 2 && PEARL_BD_PERSIST', '    bool hasher;', '    uint32_t L;',
      '    if constexpr (kHashRg) {',
      ...ROLE_LINES.rg.map((l) => '      ' + assign(l)),
      '    } else {',
      ...ROLE_LINES.dflt.map((l) => '      ' + assign(l)),
      '    }',
      '#elif PEARL_BD_GROUPS == 2',
      '    const bool hasher = sid < 64u;',
      '    const uint32_t L = wr * 64u + sid;',
      '#else',
    ]);
    expect(code.filter((l) => l.includes('PEARL_BD_HASH_RG'))).toEqual([
      '  static_assert(!kHashRg || PEARL_BD_HASH_RG_OK,',
      '                "the second row group hashes only where PEARL_BD_HASH_RG builds");']);
    expect(code.filter((l) => l.includes('kHashRg'))).toEqual(['template <bool kHashRg>',
      '  static_assert(!kHashRg || PEARL_BD_HASH_RG_OK,', '    if constexpr (kHashRg) {']);
    expect(FOLD).toContain(RG_TEMPLATE + '__device__ __forceinline__ void pearl_fold_bd(const int8_t *__restrict__ Aprime,');
    // The names the role reads: a thread's warp, row group and place in its row group.
    expect(FOLD).toContain('  const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;');
    expect(FOLD).toContain('  const uint32_t wr = warp >> 2, wc = warp & 3u;');
    expect(FOLD).toContain('  const uint32_t sid = threadIdx.x & 127u;  // a thread\'s place in its row group, for the hash');
    // The copy of sid: an asm with no instruction, sid's register in and out.
    expect(ROLE_LINES.rg[1]).toBe('asm("" : "+r"(osid));');

    for (const base of BASES) {
      const a = roleOf(base);
      const b = roleOf(rg(base));
      expect([base, a.osid, a.lines]).toEqual([base, false, ROLE_LINES.dflt]);
      expect([rg(base), b.osid, b.lines]).toEqual([rg(base), true, ROLE_LINES.rg]);
      // osid nowhere else, and nowhere in a build without the switch.
      expect([base, build(base).text.includes('osid')]).toEqual([base, false]);
      expect([rg(base), build(rg(base)).lines.filter((l) => l.includes('osid'))])
        .toEqual([rg(base), ROLE_LINES.rg.filter((l) => l.includes('osid'))]);
      // sid is set once, as the thread's place in its row group.
      expect([rg(base), build(rg(base)).lines.filter((l) => /\bo?sid =/.test(l))])
        .toEqual([rg(base), ['const uint32_t sid = threadIdx.x & 127u;', 'uint32_t osid = sid;']]);
    }
    // The copies run as the code does, for every thread.
    for (const [spec, copy] of [['', ROLE.dflt], ['HASH_RG=1', ROLE.rg]]) {
      const role = roleOf(spec);
      for (let t = 0; t < THREADS; t++) {
        const want = copy(t);
        expect([spec, t, role.of(t)]).toEqual([spec, t, { hasher: want.hasher, L: want.L }]);
      }
    }
  });

  test('all 256 threads together hash each region once a tile, one whole warp on each scheduler', () => {
    const hashers = (of) => Array.from({ length: THREADS }, (_, t) => t).filter((t) => of(t).hasher);
    const warpsOf = (ts) => [...new Set(ts.map((t) => t >> 5))];
    for (const spec of [...BASES, ...BASES.map(rg)]) {
      const role = roleOf(spec);
      const of = (t) => role.of(t);
      const isRg = role.osid;
      const hs = hashers(of);
      // 128 hashers, each region 0-127 once.
      expect([spec, hs.length]).toEqual([spec, REGIONS]);
      expect([spec, hs.map((t) => of(t).L).sort((x, y) => x - y)])
        .toEqual([spec, Array.from({ length: REGIONS }, (_, i) => i)]);
      // Whole warps: all 32 lanes of a warp hash, or none, so if (hasher) never splits a
      // warp.
      for (let w = 0; w < 8; w++) {
        const lanes = Array.from({ length: 32 }, (_, l) => of(w * 32 + l).hasher);
        expect([spec, w, lanes.every((h) => h === lanes[0])]).toEqual([spec, w, true]);
      }
      // One warp on each scheduler: warps 4-7 with the switch, 0, 1, 6 and 7 without.
      const ws = warpsOf(hs);
      expect([spec, ws]).toEqual([spec, isRg ? [4, 5, 6, 7] : [0, 1, 6, 7]]);
      expect([spec, ws.map((w) => w & 3).sort()]).toEqual([spec, [0, 1, 2, 3]]);
      // With the switch the second row group hashes and the first starts the next tile;
      // without it two warps of each.
      expect([spec, [0, 1].map((g) => ws.filter((w) => w >> 2 === g).length)]).toEqual([spec, isRg ? [0, 4] : [2, 2]]);
      // Each hasher takes region sid, the region of the same lane of the same scheduler
      // as without the switch: 32 s + lane on scheduler s.
      for (const t of hs) {
        const { warp, lane, sid } = ids(t);
        expect([spec, t, of(t).L]).toEqual([spec, t, sid]);
        expect([spec, t, of(t).L]).toEqual([spec, t, 32 * (warp & 3) + lane]);
      }
    }
    // HASH_RG's role is the row group: (wr ? sid >= 64 : sid < 64) != (sid < 64) is wr.
    // Threads t and t ^ 128 share a scheduler and a lane, one in each row group. Without
    // the switch one of the two hashes, region t & 127; with it, the one in row group 1.
    for (let t = 0; t < THREADS; t++) {
      const { wr } = ids(t);
      expect([t, ROLE.rg(t)]).toEqual([t, { hasher: wr === 1, L: t & 127 }]);
      expect([t, ROLE.dflt(t).hasher !== ROLE.dflt(t ^ 128).hasher]).toEqual([t, true]);
      const d = ROLE.dflt(t).hasher ? ROLE.dflt(t) : ROLE.dflt(t ^ 128);
      expect([t, d.L]).toEqual([t, t & 127]);
    }
  });

  test('each region is read from the same words and gets the same number, over every tile of a launch', () => {
    // hash_tile reads region L's words at the index readout_st stores them, and region_of
    // numbers a region from L and the tile alone, whichever thread hashes it.
    for (const spec of ['', 'HASH_RG=1', 'LAST_RO=1 HASH_RG=1', `${OFF} HASH_RG=1`]) {
      const b = build(spec);
      const ht = hashTile(b);
      expect(ht).toContain('for (uint32_t c = 0; c < 16u; c++) tm[c] = sT[tbuf + L * 16u + (c ^ ((L >> 1) & 15u))];');
      const ro = block(ht, 'auto region_of = [&](uint32_t rbg_, uint32_t cbg_) -> uint32_t {');
      expect([spec, ro]).toEqual([spec, ['auto region_of = [&](uint32_t rbg_, uint32_t cbg_) -> uint32_t {',
        'const uint32_t ow = L / warp_regions;', 'const uint32_t ocb = (L % warp_regions) / regions_per_warp;',
        'const uint32_t oreg = L % regions_per_warp;', 'const uint32_t orb = rbg_ * 2u + ow / 4u;',
        'const uint32_t ocgb = cbg_ * 4u + ow % 4u;',
        'return (ocgb * 4u + ocb) * rows_valid + orb * regions_per_warp + oreg;', '};']]);
      const names = new Set(ro.slice(1, -1).join(' ').match(/(?<!\w)[A-Za-z_]\w*/g));
      for (const n of ['const', 'uint32_t', 'return', 'ow', 'ocb', 'oreg', 'orb', 'ocgb']) names.delete(n);
      expect([spec, [...names].sort()]).toEqual([spec, ['L', 'cbg_', 'rbg_', 'regions_per_warp', 'rows_valid',
        'warp_regions']]);
      // A hit records that number, the hash and the 16 words.
      for (const line of ['const uint32_t slot = atomicAdd(hits.count, 1u);', 'if (slot >= PEARL_MAX_HITS) return;',
        'hits.index[slot] = region;', 'for (int i = 0; i < 8; i++) hits.hash[slot * 8u + i] = h[i];',
        'for (int i = 0; i < 16; i++) hits.transcript[slot * 16u + i] = tm[i];']) {
        expect([spec, line, ht.includes(line)]).toEqual([spec, line, true]);
      }
      if (config(spec).value('PEARL_BD_HIT_COORDS') === 1) {
        inOrder(ht, ['if (slot >= PEARL_MAX_HITS) return;', 'uint32_t rbg, cbg;', 'tile_coords(vtile, rbg, cbg);',
          'const uint32_t region = region_of(rbg, cbg);', 'hits.index[slot] = region;']);
      } else {
        inOrder(ht, ['const uint32_t region = region_of(rbg, cbg);', 'if (slot >= PEARL_MAX_HITS) return;',
          'hits.index[slot] = region;']);
      }
    }
    // The copies, pinned.
    expect(FOLD).toContain('constexpr uint32_t regions_per_warp = 4u, warp_regions = 16u;');
    expect(FOLD).toContain('const uint32_t row_blocks = rows_valid / regions_per_warp;');
    expect(FOLD).toContain('const uint32_t row_block_groups = row_blocks / 2u;');
    expect(FOLD).toContain('const uint32_t col_block_groups = tiles / row_block_groups;');
    expect(HEADER).toContain(NL + '#ifndef PEARL_BD_WALK' + NL + '#define PEARL_BD_WALK 1' + NL);
    const walk = block(build('HASH_RG=1').lines, 'auto tile_coords = [&](uint32_t v, uint32_t &rbg_, uint32_t &cbg_) {');
    expect(build('HASH_RG=1').lines).toContain('const uint32_t band_tiles = band_depth * row_block_groups;');
    expect(walk.join(' ')).toBe('auto tile_coords = [&](uint32_t v, uint32_t &rbg_, uint32_t &cbg_) { '
      + 'const uint32_t band = v / band_tiles; const uint32_t in_band = v % band_tiles; '
      + 'const uint32_t band_first = band * band_depth; '
      + 'const uint32_t band_cols = col_block_groups - band_first < band_depth ? col_block_groups - band_first : band_depth; '
      + 'cbg_ = band_first + in_band % band_cols; rbg_ = in_band / band_cols; };');
    const ro = flatLines(hashTile(build('HASH_RG=1')));
    // region_of as the build has it, run on the CPU, is the copy.
    const steps = [...ro.matchAll(/const uint32_t (ow|ocb|oreg|orb|ocgb) = ([^;]+);/g)].map((x) => [x[1], x[2]]);
    const ret = /return ([^;]+); \};/.exec(ro)[1];
    const fromCode = (L, rbg, cbg, rowsValid) => {
      const env = { L, rbg_: rbg, cbg_: cbg, rows_valid: rowsValid, regions_per_warp: 4, warp_regions: 16 };
      for (const [n, e] of steps) env[n] = compile(e, true)(codeScope(env, new Map()));
      return compile(ret, true)(codeScope(env, new Map()));
    };
    for (const L of [0, 1, 17, 63, 64, 100, 127]) {
      for (const [rbg, cbg, rowsValid] of [[0, 0, 8], [3, 5, 64], [16, 2, 136], [511, 31, 4096]]) {
        expect([L, rbg, cbg, fromCode(L, rbg, cbg, rowsValid)]).toEqual([L, rbg, cbg, regionOf(L, rbg, cbg, rowsValid)]);
      }
    }

    // A launch, tile by tile: the regions the hashers number are every region of the
    // launch once, with the switch and without, and each tile's region L the same.
    for (const geo of [{ rowsValid: 8, tiles: 1, bandDepth: 1 }, { rowsValid: 64, tiles: 8 * 3, bandDepth: 2 },
      { rowsValid: 136, tiles: 17 * 7, bandDepth: 5 }, { rowsValid: 64, tiles: 8 * 7, bandDepth: 9 }]) {
      const numbers = {};
      for (const [name, copy] of Object.entries(ROLE)) {
        const seen = [];
        for (let v = 0; v < geo.tiles; v++) {
          const { rbg, cbg } = tileCoords(v, geo);
          for (let t = 0; t < THREADS; t++) {
            const { hasher, L } = copy(t);
            if (hasher) seen.push([v, L, regionOf(L, rbg, cbg, geo.rowsValid)]);
          }
        }
        seen.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
        numbers[name] = seen;
        const all = seen.map((x) => x[2]).sort((x, y) => x - y);
        expect([name, geo, all.length]).toEqual([name, geo, geo.tiles * REGIONS]);
        expect([name, geo, all.every((n, i) => n === i)]).toEqual([name, geo, true]);
      }
      expect(numbers.rg).toEqual(numbers.dflt);
    }
  });

  test('the host reports the same hits, in the same order, whichever warps hash', () => {
    // The kernel appends hits with an atomic, in whatever order the hashers get there; the
    // host sorts the batch by region number before it reports it.
    expect(HOST).toContain('    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {' + NL
      + '      return ctx->hHitIndex[a] < ctx->hHitIndex[b];' + NL + '    });' + NL
      + '    fill_hit(ctx, p, slot, order[0], out);');
    expect(HOST).toContain('    for (uint32_t i = 1; i < n_hits; i++) fill_hit(ctx, p, slot, order[i], &ctx->extraHits[i - 1]);');
    const maxHits = Number(/#define PEARL_MAX_HITS (\d+)/.exec(HEADER)[1]);
    expect(maxHits).toBe(64);
    // A batch: random words for each tile's regions, and a region hits when the mix of its
    // 16 words is under a threshold (about one in 300). Each role's hashers run in a random
    // order, each taking a slot as the kernel does, and the host sorts.
    const geo = { rowsValid: 64, tiles: 8 * 6, bandDepth: 2 };
    const batch = (seed) => {
      const rand = xorshift(seed);
      const words = Array.from({ length: geo.tiles }, () => Uint32Array.from({ length: TWORDS }, () => rand()));
      const run = (copy, order) => {
        const slots = [];
        const events = [];
        for (let v = 0; v < geo.tiles; v++) for (let t = 0; t < THREADS; t++) if (copy(t).hasher) events.push([v, t]);
        for (let i = events.length - 1; i > 0; i--) {
          const j = order() % (i + 1);
          [events[i], events[j]] = [events[j], events[i]];
        }
        for (const [v, t] of events) {
          const { L } = copy(t);
          const tm = Array.from({ length: CHUNKS }, (_, c) => words[v][wordOf(0, L, c)]);
          if (mix(tm) >= 0xffffffff / 300) continue;
          const slot = slots.length;
          if (slot >= maxHits) continue;
          const { rbg, cbg } = tileCoords(v, geo);
          slots.push({ index: regionOf(L, rbg, cbg, geo.rowsValid), transcript: tm });
        }
        const sorted = slots.map((s, i) => i).sort((a, b) => slots[a].index - slots[b].index).map((i) => slots[i]);
        return { slots, sorted };
      };
      return { dflt: run(ROLE.dflt, xorshift(seed + 1)), rg: run(ROLE.rg, xorshift(seed + 2)) };
    };
    let reordered = 0;
    for (const seed of [3, 5, 8, 13]) {
      const { dflt, rg: r } = batch(seed);
      expect(dflt.slots.length).toBeGreaterThan(1);
      expect(dflt.slots.length).toBeLessThan(maxHits);
      expect(r.sorted).toEqual(dflt.sorted);
      if (JSON.stringify(r.slots) !== JSON.stringify(dflt.slots)) reordered++;
    }
    // Only the slots move, and the sort puts them back.
    expect(reordered).toBeGreaterThan(0);
  });

  test('outside the role, a build with the switch is the build without it, line for line', () => {
    for (const base of [...BASES, 'PEARL_ABLATE_TRANSCRIPT_HASH', 'PEARL_ABLATE_READOUT', 'PEARL_ABLATE_TRANSCRIPT']) {
      const a = build(base).lines;
      const b = build(rg(base)).lines;
      const i = b.indexOf(ROLE_LINES.rg[0]);
      expect([base, i > 0, b.lastIndexOf(ROLE_LINES.rg[0]) === i]).toEqual([base, true, true]);
      expect([base, b.slice(i, i + 4)]).toEqual([base, ROLE_LINES.rg]);
      expect([base, a.slice(i, i + 2)]).toEqual([base, ROLE_LINES.dflt]);
      expect([base, b.slice(0, i)]).toEqual([base, a.slice(0, i)]);
      expect([base, b.slice(i + 4)]).toEqual([base, a.slice(i + 2)]);
    }
  });

  test('with LAST_RO too, every region of every tile is hashed once from its own words, in any thread order', () => {
    // The persistent loop as each build has it: tbuf moves on at the loop's top, tile()
    // stores chunks 0-14, chunk 15 is stored at the tile's end or by tile_head(true)
    // before the hand-off, and the hash of the tile follows the barrier. The block's first
    // tile_head stores nothing, and nothing is hashed after the loop.
    const LOOP = 'for (uint32_t v = blockIdx.x; v < tiles; it++) {';
    const sims = ['', 'LAST_RO=1', 'HASH_RG=1', 'LAST_RO=1 HASH_RG=1', 'LAST_RO=1 SNAKE=1 HASH_RG=1', `${OFF} HASH_RG=1`,
      'HIT_COORDS=0 HASH_RG=1'];
    for (const spec of sims) {
      const b = build(spec);
      const lastRo = config(spec).value('PEARL_BD_LAST_RO') === 1;
      // The store and the read the simulation runs, as the build has them: the copies.
      const tc = transcriptCode(spec);
      expect([spec, tc.expr]).toEqual([spec, {
        writer: '((lane >> 2) & 3u) == 0u',
        region: 'warp * warp_regions + (lane & 3u) * regions_per_warp + 2u * rl + ((lane >> 4) & 1u)',
        store: 'tbuf + L * 16u + (c ^ ((L >> 1) & 15u))',
        read: 'tbuf + L * 16u + (c ^ ((L >> 1) & 15u))',
      }]);
      for (let t = 0; t < THREADS; t++) {
        const { warp, lane } = ids(t);
        expect(tc.writer(lane)).toBe(writes(lane));
        for (let rl = 0; rl < 2; rl++) expect(tc.region(warp, lane, rl)).toBe(writerRegion(warp, lane, rl));
      }
      for (const tbuf of [0, TWORDS]) {
        for (let L = 0; L < REGIONS; L++) {
          for (let c = 0; c < CHUNKS; c++) {
            expect(tc.storeAt(tbuf, L, c)).toBe(wordOf(tbuf, L, c));
            expect(tc.readAt(tbuf, L, c)).toBe(wordOf(tbuf, L, c));
          }
        }
      }
      expect(b.lines).toContain('constexpr uint32_t TWORDS = (PEARL_FOLD_THREADS / 32u) * warp_regions * PEARL_JACKPOT_BUCKETS;');
      expect([spec, b.lines.filter((l) => /\btbuf =/.test(l))]).toEqual([spec, ['uint32_t tbuf = 0u;',
        'tbuf = (it & 1u) * TWORDS;']]);
      const loop = block(b.lines, LOOP);
      const hand = lastRo ? ['tile_head(true);'] : [];
      inOrder(loop, [LOOP, 'tbuf = (it & 1u) * TWORDS;', 'tile([&]() {', ...hand, '__syncthreads();',
        config(spec).value('PEARL_BD_HIT_COORDS') ? 'hash_tile(v);' : 'hash_tile(rbg, cbg);', 'v = vnext;']);
      expect([spec, loop.filter((l) => l === '__syncthreads();').length]).toEqual([spec, 1]);
      // hash_tile is called once, in the loop.
      expect([spec, b.lines.filter((l) => /(^|[^.\w])hash_tile\(/.test(l)).length]).toEqual([spec, 1]);
      const tile = block(b.lines, 'auto tile = [&](auto next_srcs) {');
      expect(tile).toContain('for (uint32_t chunk = first_chunk; chunk + 1u < chunks; chunk++) {');
      expect(tile).toContain('constexpr uint32_t first_chunk = 0u;');
      expect(tile).toContain('for (uint32_t rl = 0; rl < RPL; rl++) readout_st(rl, chunk, rx[rl]);');
      if (lastRo) {
        expect([spec, tile.some((l) => /readout(_st)?\(.*chunks - 1u/.test(l))]).toEqual([spec, false]);
        const head = block(b.lines, 'auto tile_head = [&](bool prev) {');
        const [, store] = inOrder(head, ['kstep(0u, 0u, false, false);',
          'for (uint32_t rl = 0; rl < RPL; rl++) readout_st(rl, chunks - 1u, rx[rl]);', 'kstep(0u, 1u, false, false);']);
        expect(head.slice(0, store).filter((l) => l === 'if (prev) {').length).toBe(2);
        inOrder(b.lines, ['tile_head(false);', '__syncthreads();', 'uint32_t it = 0u;', LOOP]);
      } else {
        expect(tile).toContain('for (uint32_t rl = 0; rl < RPL; rl++) readout(rl, chunks - 1u);');
        expect([spec, b.text.includes('tile_head')]).toEqual([spec, false]);
      }
      for (const [tiles, seed] of [[1, 2], [2, 3], [7, 5]]) {
        expect([spec, tiles, simulate(spec, { tiles, seed })]).toEqual([spec, tiles,
          { read: tiles * REGIONS * CHUNKS, wrong: 0, outside: 0, badHandoff: 0, twice: 0, never: 0 }]);
      }
    }
    // And it sees each of these: one buffer for both tiles, tbuf moved on before the hash,
    // the hash before the barrier, chunk 15's stores after it, or HASH_RG's hashers with
    // the default's regions.
    const broken = (spec, mutate) => simulate(spec, { tiles: 4, seed: 11, mutate });
    expect(broken('LAST_RO=1 HASH_RG=1', 'one-buffer').wrong).toBeGreaterThan(0);
    expect(broken('LAST_RO=1 HASH_RG=1', 'flip-before-hash').wrong).toBeGreaterThan(0);
    expect(broken('LAST_RO=1 HASH_RG=1', 'hash-before-barrier').wrong).toBeGreaterThan(0);
    expect(broken('LAST_RO=1 HASH_RG=1', 'head-after-barrier').badHandoff).toBe(4);
    expect(broken('HASH_RG=1', 'rg-with-default-L')).toMatchObject({ twice: 4 * 64, never: 4 * 64 });
  });

  test('the bench names the switch in the fold name', () => {
    // The host's suffix after the fold's name: its #if, the ", with" list and the ifs
    // after it, run for a build's switches.
    const host = HOST.split(NL);
    const at = host.findIndex((l) => l.startsWith('#if PEARL_BD_RUNBASE || PEARL_BD_TAIL_LDG'));
    expect(at).toBeGreaterThan(0);
    const cond = host.slice(at, at + 3).join(' ').split(BS).join(' ').replace(/^#if\s+/, '').replace(/\s+/g, ' ').trim();
    expect(cond).toBe('PEARL_BD_RUNBASE || PEARL_BD_TAIL_LDG || PEARL_BD_TADDR || PEARL_BD_RED || PEARL_BD_GATE != 2 '
      + '|| PEARL_BD_HIT_COORDS || PEARL_BD_SNAKE || PEARL_BD_HASH_SPLIT || PEARL_BD_HASH_RG || PEARL_BD_RG_KERNEL_BUILT');
    const end = host.indexOf('#endif', at);
    const body = host.slice(at + 3, end).map((l) => l.trim()).join(' ');
    // HASH_RG built in, or the second kernel running on this card (Ctx::bdHashRg); and with
    // nothing on, no ", with".
    expect(body).toContain('if (PEARL_BD_HASH_RG || c->bdHashRg) snprintf(c->foldName + n, sizeof c->foldName - n, " HASH_RG");');
    expect(body).toContain('size_t n = strlen(c->foldName); const size_t named = n;');
    expect(body).toContain('if (strlen(c->foldName) == named + strlen(", with")) c->foldName[named] = 0; }');
    const list = [...body.matchAll(/(PEARL_BD_\w+(?: != 2)?) \? "( [A-Z_=0-9]+)" : ""/g)].map((m) => [m[1], m[2]]);
    expect(list.map((x) => x[1])).toEqual([' RUNBASE', ' TAIL_LDG', ' GATE=1', ' TADDR', ' RED', ' HIT_COORDS', ' SNAKE']);
    const ifs = [...body.matchAll(/if \((PEARL_BD_\w+)\) snprintf\(c->foldName \+ n, sizeof c->foldName - n, "([^"]*)"(?:, (PEARL_BD_\w+))?\);/g)]
      .map((m) => [m[1], m[2], m[3]]);
    expect(ifs.map((x) => x[0])).toEqual(['PEARL_BD_HASH_SPLIT', 'PEARL_BD_LAST_RO', 'PEARL_BD_ROWPAIR']);
    // HASH_RG's if, which also reads the context, goes between LAST_RO's and ROWPAIR's.
    ifs.splice(2, 0, ['PEARL_BD_HASH_RG', ' HASH_RG', undefined]);
    // rgRuns: this card runs the second kernel.
    const suffix = (spec, c = cond, rgRuns = false) => {
      const v = (e) => (e === 'PEARL_BD_HASH_RG' ? b01(config(spec).value(e) || rgRuns) : config(spec).value(e));
      if (rgRuns) expect([spec, v('PEARL_BD_RG_KERNEL_BUILT')]).toEqual([spec, 1]);
      if (!compile(c, false)(macroScope(config(spec).defs))) return '';
      const s = ', with' + list.filter(([e]) => v(e)).map(([, x]) => x).join('')
        + ifs.filter(([e]) => v(e)).map(([, x, arg]) => x.replace('%d', arg ? String(v(arg)) : '')).join('');
      return s === ', with' ? '' : s;
    };
    expect(suffix('')).toBe(', with RUNBASE TAIL_LDG GATE=1 HIT_COORDS');
    expect(suffix('HASH_RG=1')).toBe(', with RUNBASE TAIL_LDG GATE=1 HIT_COORDS HASH_RG');
    expect(suffix('LAST_RO=1 HASH_RG=1')).toBe(', with RUNBASE TAIL_LDG GATE=1 HIT_COORDS LAST_RO HASH_RG');
    expect(suffix('ROWPAIR=1 LAST_RO=1 SNAKE=1 HASH_RG=1'))
      .toBe(', with RUNBASE TAIL_LDG GATE=1 HIT_COORDS SNAKE LAST_RO HASH_RG ROWPAIR');
    expect(suffix(OFF)).toBe('');
    // On the fold before LEAN, HASH_RG is the only switch on, and its term in the #if is
    // what lets the name through.
    expect(suffix(`${OFF} HASH_RG=1`)).toBe(', with HASH_RG');
    expect(suffix(`${OFF} HASH_RG=1`, cond.replace(' || PEARL_BD_HASH_RG || PEARL_BD_RG_KERNEL_BUILT', ''))).toBe('');
    for (const base of BASES) {
      expect([base, suffix(base).split(' ').includes('HASH_RG'), suffix(rg(base)).split(' ').includes('HASH_RG')])
        .toEqual([base, false, true]);
      // The second kernel running names it as the switch does.
      expect([base, suffix(base, cond, true)]).toEqual([base, suffix(rg(base))]);
    }
    // On the fold before LEAN only the second kernel's term lets the name through, and with
    // the kernel built but not running there is no ", with".
    expect(suffix(OFF, cond, true)).toBe(', with HASH_RG');
    expect(suffix(OFF, cond.replace(' || PEARL_BD_RG_KERNEL_BUILT', ''), true)).toBe('');
    expect(config(OFF).value(cond)).toBe(1);
  });
});

// ---- The second sm_75 kernel (PEARL_BD_RG_KERNEL) --------------------------------------

// pearl_host.cu's code lines, trimmed, without comments or blank lines.
const HOST_LINES = HOST.split(NL).map((l) => uncomment(l).trim()).filter((l) => l !== '');
const KERNEL_LINES = KERNEL.split(NL).map((l) => uncomment(l).trim()).filter((l) => l !== '');

// pearl_bd_rg_for for a build, read out of the host and run: whether a card with l2 bytes
// of L2 runs the second kernel. Its body is ifs that return true or false, then a return.
function rgFor(spec) {
  const body = block(HOST_LINES, 'bool pearl_bd_rg_for(uint64_t l2) {').slice(1, -1);
  const steps = body.map((l) => {
    let m = /^if \((.+)\) return (true|false);$/.exec(l);
    if (m) return { cond: compile(m[1], true), ret: m[2] === 'true' };
    m = /^return (.+);$/.exec(l);
    if (m) return { expr: compile(m[1].split('(uint64_t)').join(''), true) };
    throw new Error(`pearl_bd_rg_for: cannot read "${l}"`);
  });
  expect(steps[steps.length - 1].expr).toBeDefined();
  const { defs } = config(spec);
  return (l2) => {
    const sc = codeScope({ l2 }, defs);
    for (const st of steps) {
      if (!st.expr && st.cond(sc)) return st.ret;
      if (st.expr) return st.expr(sc) === 1;
    }
    throw new Error('pearl_bd_rg_for runs off its end');
  };
}

// The L2 each sm_75 card reports (cudaDevAttrL2CacheSize).
const MB = 1048576;
const CARDS = [
  ['GTX 1650', 1 * MB], ['GTX 1660 Ti', 1.5 * MB], ['RTX 2060', 3 * MB], ['RTX 2060 12 GB', 3 * MB],
  ['RTX 2060 Super', 4 * MB], ['RTX 2070', 4 * MB], ['RTX 2070 Super', 4 * MB], ['RTX 2080', 4 * MB],
  ['RTX 2080 Super', 4 * MB], ['Tesla T4', 4 * MB], ['RTX 2080 Ti', 5.5 * MB], ['Titan RTX', 6 * MB],
];

// The sweep's override for the second kernel, read out of pearl_bd_sweep and run, with
// pearl_bd_sweep_env's reading of a value: unset or empty is not set, digits are a
// number, anything else is ignored.
function sweepRg(spec) {
  const flatSweep = block(HOST_LINES, 'bool pearl_bd_sweep(PearlBdLaunch *s, uint32_t colsValid) {').join(' ');
  const m = /if \(pearl_bd_sweep_env\("PEARL_BD_SWEEP_HASH_RG", &v\)\) \{ any = true; if \(([^)]+)\) s->hashRg = ([^;]+); \}/
    .exec(flatSweep);
  expect(m).not.toBeNull();
  const ok = compile(m[1], true);
  const val = compile(m[2], true);
  const { defs } = config(spec);
  return (env, rule) => {
    if (env === undefined || env === '' || !/^[0-9]+$/.test(env)) return { any: false, hashRg: rule };
    const sc = codeScope({ v: Number(env) }, defs);
    return { any: true, hashRg: ok(sc) ? val(sc) === 1 : rule };
  };
}

describe('the second sm_75 kernel, HASH_RG\'s code by L2 size (PEARL_BD_RG_KERNEL)', () => {
  test('the header: by L2 by default, 4 MB, and built wherever HASH_RG can be and is off', () => {
    const c = config('');
    expect(c.value('PEARL_BD_RG_KERNEL')).toBe(2);
    expect(c.value('PEARL_BD_RG_L2_MIN')).toBe(4 * MB);
    expect(c.value('PEARL_BD_HASH_RG_OK')).toBe(1);
    expect(c.value('PEARL_BD_RG_KERNEL_BUILT')).toBe(1);
    for (const v of [0, 1, 2]) {
      expect([v, config(`RG_KERNEL=${v}`).errors, config(`RG_KERNEL=${v}`).value('PEARL_BD_RG_KERNEL')]).toEqual([v, [], v]);
      // The policy is the host's: it changes no other macro.
      expect(config(`RG_KERNEL=${v}`).value('PEARL_BD_RG_KERNEL_BUILT')).toBe(1);
    }
    for (const v of ['3', '-1']) {
      expect(config(`RG_KERNEL=${v}`).errors).toEqual(['"PEARL_BD_RG_KERNEL is 0 (never), 1 (always) or 2 (by L2)"']);
    }
    // HASH_RG_OK says exactly where the switch's guards let it build, and the second kernel
    // is built there, unless the switch is on and the first kernel already is that code.
    const specs = [...BASES, 'TADDR=1', 'GATE=2', 'RUNBASE=0 TAIL_LDG=0', ...OTHER_FOLDS, 'RED=1', 'RED=1 GATE=1',
      'HASH_SPLIT=4 RUNBASE=1 TAIL_LDG=1', 'HASH_SPLIT=13 RUNBASE=1 TAIL_LDG=1'];
    for (const spec of specs) {
      const ok = config(spec).value('PEARL_BD_HASH_RG_OK');
      expect([spec, config(rg(spec)).errors.length === 0]).toEqual([spec, ok === 1]);
      expect([spec, config(spec).value('PEARL_BD_RG_KERNEL_BUILT')]).toEqual([spec, ok]);
      if (ok) expect([spec, config(rg(spec)).value('PEARL_BD_RG_KERNEL_BUILT')]).toEqual([spec, 0]);
    }
    for (const spec of OTHER_FOLDS) expect([spec, config(spec).value('PEARL_BD_RG_KERNEL_BUILT')]).toEqual([spec, 0]);
    // Not without the B-direct fold.
    expect(config('PEARL_TURING_BDIRECT=0').value('PEARL_BD_RG_KERNEL_BUILT')).toBe(0);
    // The guards and the macros it reads come before it.
    const lines = HEADER.split(NL);
    const at = lines.indexOf('#define PEARL_BD_RG_KERNEL_BUILT (PEARL_TURING_BDIRECT && PEARL_BD_HASH_RG_OK && !PEARL_BD_HASH_RG)');
    expect(at).toBeGreaterThan(0);
    for (const name of ['PEARL_TURING_BDIRECT', 'PEARL_BD_HASH_RG_OK', 'PEARL_BD_HASH_RG']) {
      const d = lines.findIndex((l) => l.startsWith('#define ' + name + ' '));
      expect([name, d >= 0 && d < at]).toEqual([name, true]);
    }
    const guard = lines.indexOf('#if PEARL_BD_HASH_RG && (PEARL_BD_RED || PEARL_BD_HASH_SPLIT)');
    expect(guard).toBeGreaterThan(0);
    expect(guard).toBeLessThan(lines.findIndex((l) => l.startsWith('#define PEARL_BD_HASH_RG_OK ')));
  });

  test('pearl_kernel.cu: the second kernel is the first with kHashRg true, and only where it is built', () => {
    const head = (name) => `extern "C" __global__ __launch_bounds__(PEARL_FOLD_THREADS) void ${name}(`;
    const params = (name) => {
      const at = KERNEL.indexOf(head(name));
      expect([name, at > 0, KERNEL.lastIndexOf(head(name)) === at]).toEqual([name, true, true]);
      return KERNEL.slice(at + head(name).length, KERNEL.indexOf(') {', at)).replace(/\s+/g, ' ');
    };
    expect(params('pearl_tile_fold_wmma_rg')).toBe(params('pearl_tile_fold_wmma'));
    // The second kernel, whole, as the code lines it is.
    const at = KERNEL_LINES.indexOf(head('pearl_tile_fold_wmma_rg'));
    const call = 'pearl_fold_bd<true>(Aprime, Bprime, col_off, rows_valid, tiles, test, hits);';
    expect(KERNEL_LINES.slice(at - 1, at + 17)).toEqual(['#if PEARL_BD_RG_KERNEL_BUILT',
      head('pearl_tile_fold_wmma_rg'),
      'const int8_t *__restrict__ Aprime, const int8_t *__restrict__ Bprime,',
      'uint32_t m, uint32_t n, uint32_t k_arg, uint32_t rank_arg, uint32_t chunks_arg,',
      'uint32_t col_off, uint32_t rows_valid, uint32_t col_groups, uint32_t tiles,',
      'const PearlTranscriptTest test, const PearlHitList hits) {',
      '#if PEARL_TURING_BD_BODY',
      'if (k_arg != PEARL_FOLD_K || rank_arg != PEARL_FOLD_RANK || chunks_arg != PEARL_FOLD_CHUNKS',
      '|| blockDim.x != PEARL_FOLD_THREADS)', 'return;', '(void)m; (void)n; (void)col_groups;', call,
      '#else',
      '(void)Aprime; (void)Bprime; (void)m; (void)n; (void)k_arg; (void)rank_arg; (void)chunks_arg;',
      '(void)col_off; (void)rows_valid; (void)col_groups; (void)tiles; (void)test; (void)hits;',
      '#endif', '}', '#endif']);
    // The first kernel's sm_75 body is the same lines, passing PEARL_BD_HASH_RG.
    const first = KERNEL_LINES.indexOf(head('pearl_tile_fold_wmma'));
    const sm75 = (from) => {
      const i = KERNEL_LINES.indexOf('#if PEARL_TURING_BD_BODY', from);
      return KERNEL_LINES.slice(i, KERNEL_LINES.indexOf('#else', i));
    };
    expect(sm75(first).map((l) => (l === call ? 'CALL' : l)).join(NL).replace(
      'pearl_fold_bd<PEARL_BD_HASH_RG != 0>(Aprime, Bprime, col_off, rows_valid, tiles, test, hits);', 'CALL'))
      .toBe(sm75(at).map((l) => (l === call ? 'CALL' : l)).join(NL));
    expect(KERNEL.split('pearl_fold_bd<').length - 1).toBe(2);
    // The argument each kernel passes, which build() compiles the fold with.
    expect(kernelArg('pearl_tile_fold_wmma')).toBe('PEARL_BD_HASH_RG != 0');
    expect(kernelArg('pearl_tile_fold_wmma_rg')).toBe('true');
    expect(sm75(first)).toContain('pearl_fold_bd<PEARL_BD_HASH_RG != 0>(Aprime, Bprime, col_off, rows_valid, tiles, test, hits);');
  });

  test('the second kernel\'s fold is HASH_RG\'s line for line, and the first kernel\'s is what it was', () => {
    // Everything the switch's build compiles, the second kernel compiles without it.
    for (const base of [...BASES, 'TADDR=1', 'GATE=2', 'RUNBASE=0 TAIL_LDG=0', 'PEARL_ABLATE_TRANSCRIPT_HASH',
      'PEARL_ABLATE_READOUT', 'PEARL_ABLATE_TRANSCRIPT']) {
      expect([base, build(base, 'rg').lines]).toEqual([base, build(rg(base)).lines]);
      expect([base, build(base).text.includes('osid'), build(base, 'rg').text.includes('osid')]).toEqual([base, false, true]);
    }
    // The code lines each build compiled at 2aa8201, before the template, hashed: the first
    // kernel's are the same, and the second kernel's are what HASH_RG=1 built then. ptxas
    // 12.8 builds 5c97fbadf64a and ac1a72f2434b from them (pearl_config.h).
    const sha = (b) => crypto.createHash('sha256').update(b.text).digest('hex').slice(0, 16);
    expect(sha(build(''))).toBe('5691adfcb68a0ab5');
    expect(sha(build('', 'rg'))).toBe('0e0ba8303f480c82');
    expect(sha(build('HASH_RG=1'))).toBe('0e0ba8303f480c82');
    expect(sha(build(OFF))).toBe('1f449377837a5ec0');
    expect(sha(build('LAST_RO=1'))).toBe('1a659a3d81361789');
    expect(sha(build('ROWPAIR=1'))).toBe('a4ae013f15182f68');
    expect(sha(build('SNAKE=1'))).toBe('df5d021a03785a1b');
    // No kHashRg is left in any build's code once instantiated.
    for (const spec of ['', 'HASH_RG=1', OFF]) expect([spec, build(spec).text.includes('kHashRg')]).toEqual([spec, false]);
  });

  test('the host runs it on cards with 4 MB of L2 or more (pearl_bd_rg_for)', () => {
    expect(block(HOST_LINES, 'bool pearl_bd_rg_for(uint64_t l2) {')).toEqual(['bool pearl_bd_rg_for(uint64_t l2) {',
      'if (!PEARL_BD_RG_KERNEL_BUILT) return false;', 'if (PEARL_BD_RG_KERNEL == 1) return true;',
      'if (PEARL_BD_RG_KERNEL == 0) return false;', 'return l2 >= (uint64_t)PEARL_BD_RG_L2_MIN;', '}']);
    // Beside pearl_bd_band_for, and read with it.
    const lines = HOST.split(NL);
    const band = lines.indexOf('uint32_t pearl_bd_band_for(uint32_t k, uint64_t l2) {');
    const rule = lines.indexOf('bool pearl_bd_rg_for(uint64_t l2) {');
    expect(rule).toBeGreaterThan(band);
    expect(lines.slice(band, rule).filter((l) => /^\S.*\) \{$/.test(l))).toEqual([lines[band]]);
    expect(block(HOST_LINES, 'PearlBdLaunch pearl_bd_launch_for(uint32_t k, uint64_t l2, uint32_t sms, uint32_t colBatch) {'))
      .toContain('s.hashRg = pearl_bd_rg_for(l2);');
    // By L2, the default: 4 MB and up, not 3 MB, 1.5 MB or an L2 the card did not report.
    const by = rgFor('');
    for (const [l2, want] of [[4 * MB, true], [4 * MB + 1, true], [5.5 * MB, true], [6 * MB, true], [4 * MB - 1, false],
      [3 * MB, false], [1.5 * MB, false], [1 * MB, false], [0, false]]) {
      expect([l2, by(l2)]).toEqual([l2, want]);
    }
    expect(CARDS.filter(([, l2]) => by(l2)).map(([n]) => n)).toEqual(['RTX 2060 Super', 'RTX 2070', 'RTX 2070 Super',
      'RTX 2080', 'RTX 2080 Super', 'Tesla T4', 'RTX 2080 Ti', 'Titan RTX']);
    expect(rgFor('RG_KERNEL=2')(3 * MB)).toBe(false);
    // 0 never, 1 always, the L2 unknown too.
    for (const [, l2] of [...CARDS, ['unknown', 0]]) {
      expect([l2, rgFor('RG_KERNEL=0')(l2), rgFor('RG_KERNEL=1')(l2)]).toEqual([l2, false, true]);
    }
    // Never where the build has no second kernel, whatever the policy.
    for (const spec of ['HASH_RG=1', 'RED=1', 'HASH_SPLIT=4 RUNBASE=1 TAIL_LDG=1', 'A_DIRECT=0', 'PERSIST=0',
      'PEARL_TURING_BDIRECT=0']) {
      for (const pol of ['', ' RG_KERNEL=1']) {
        expect([spec + pol, [0, 3 * MB, 4 * MB, 6 * MB].map(rgFor(spec + pol))]).toEqual([spec + pol, [false, false, false, false]]);
      }
    }
  });

  test('PEARL_BD_SWEEP_HASH_RG=0|1 overrides the rule, and the sweep line says which ran', () => {
    // pearl_bd_sweep_env: unset or empty is not set; a value that is not all digits, or
    // is over 32 bits, is ignored. Digits by hand, not strtoul, which under glibc 2.38's
    // headers would tie the addon to glibc 2.38.
    const env = block(HOST_LINES, 'bool pearl_bd_sweep_env(const char *name, uint32_t *out) {');
    expect(env).toContain('if (!v || !*v) return false;');
    expect(env).toContain("if (*p < '0' || *p > '9' || (x = x * 10u + (uint64_t)(*p - '0')) > 0xFFFFFFFFull) {");
    expect(env.join('\n')).not.toMatch(/strto|atoi|scanf/);
    expect(HOST).toContain('//   PEARL_BD_SWEEP_HASH_RG=0|1  the default kernel, or the second one (pearl_bd_rg_for);');
    const sw = sweepRg('');
    for (const rule of [false, true]) {
      expect(sw(undefined, rule)).toEqual({ any: false, hashRg: rule });
      expect(sw('', rule)).toEqual({ any: false, hashRg: rule });
      expect(sw('x', rule)).toEqual({ any: false, hashRg: rule });
      expect(sw('0', rule)).toEqual({ any: true, hashRg: false });
      expect(sw('1', rule)).toEqual({ any: true, hashRg: true });
      expect(sw('2', rule)).toEqual({ any: true, hashRg: rule });
      // Any policy: the override runs on the kernel the build has.
      expect(sweepRg('RG_KERNEL=0')('1', rule)).toEqual({ any: true, hashRg: true });
      // A build without the second kernel cannot run it.
      for (const spec of ['HASH_RG=1', 'RED=1', 'A_DIRECT=0']) {
        expect([spec, sweepRg(spec)('1', rule)]).toEqual([spec, { any: true, hashRg: false }]);
      }
    }
    // The line: every conversion has its argument, and hash_rg's are the shape's and the
    // rule's. By then s.hashRg is the kernel that runs (Ctx::bdHashRg, checked below).
    const at = HOST.indexOf('"[pearl] launch sweep: ');
    const stmt = HOST.slice(at, HOST.indexOf('l2 / 1024);', at) + 'l2 / 1024'.length);
    const fmt = [...stmt.matchAll(/"((?:[^"\\]|\\.)*)"/g)].map((x) => x[1]).join('');
    const args = stmt.slice(stmt.lastIndexOf('",') + 2).split(',').map((a) => a.trim());
    expect(fmt).toBe('[pearl] launch sweep: band %u (rule %u), col_batch %u (rule %u), streams %d (rule %d), '
      + 'hash_rg %d (rule %d); %d SMs, L2 %d KB' + BS + 'n');
    expect(args).toEqual(['s.band', 'rule.band', 's.colBatch', 'rule.colBatch', '(int)s.streams', '(int)rule.streams',
      '(int)s.hashRg', '(int)rule.hashRg', 'sms', 'l2 / 1024']);
    expect(fmt.match(/%[ud]/g).length).toBe(args.length);
  });

  test('every place the host takes the fold kernel\'s address uses the kernel it picked', () => {
    const cast = (k) => `reinterpret_cast<const void *>(${k})`;
    const count = (needle) => HOST.split(needle).length - 1;
    // The first kernel's address: resolve_fold, which reads which binary loaded, and
    // pearl_wmma_fold_fn. The second's: the create check and pearl_wmma_fold_fn.
    expect(count(cast('pearl_tile_fold_wmma'))).toBe(2);
    expect(count(cast('pearl_tile_fold_wmma_rg'))).toBe(2);
    expect(block(HOST_LINES, 'const void *pearl_wmma_fold_fn(const Ctx *ctx) {')).toEqual([
      'const void *pearl_wmma_fold_fn(const Ctx *ctx) {', '#if PEARL_BD_RG_KERNEL_BUILT',
      `if (ctx->bdHashRg) return ${cast('pearl_tile_fold_wmma_rg')};`, '#else', '(void)ctx;', '#endif',
      `return ${cast('pearl_tile_fold_wmma')};`, '}']);
    const resolve = block(HOST_LINES, 'void resolve_fold(Ctx *ctx) {');
    expect(resolve.filter((l) => l.includes('pearl_tile_fold_wmma'))).toEqual([`cudaFuncGetAttributes(&fa, ${cast('pearl_tile_fold_wmma')})`]);
    // The search: foldFn is the picked kernel, and every attribute and occupancy call
    // takes foldFn (or Hopper's own kernel, set up in resolve_fold).
    const search = HOST.slice(HOST.indexOf('const void *foldFn = '));
    expect(search.startsWith('const void *foldFn = ctx->foldTall ? reinterpret_cast<const void *>(pearl_tile_fold_tall)' + NL
      + '                                     : pearl_wmma_fold_fn(ctx);')).toBe(true);
    const sites = [...HOST.matchAll(/(cudaFuncSetAttribute|cudaFuncGetAttributes|cudaOccupancyMaxActiveBlocksPerMultiprocessor|cudaOccupancyMaxActiveClusters)\((?:&\w+,\s*)?(reinterpret_cast<const void \*>\(\w+\)|\w+)/g)]
      .map((x) => [x[1], x[2]]);
    const off = sites.filter(([, k]) => k !== 'foldFn');
    // Two more aren't fold kernels the search picks: Pascal's dp4a fold, which resolve_fold
    // checks first, and pearl_gen_perm, which create asks whether the core has code for
    // the card at all.
    expect(off).toEqual([
      ['cudaFuncGetAttributes', cast('pearl_tile_fold_dp4a')],
      ['cudaFuncGetAttributes', cast('pearl_tile_fold_wmma')],
      ['cudaFuncGetAttributes', cast('pearl_tile_fold_tall')],
      ['cudaFuncGetAttributes', cast('pearl_tile_fold_hopper')],
      ['cudaFuncSetAttribute', cast('pearl_tile_fold_hopper')],
      ['cudaFuncGetAttributes', cast('pearl_gen_perm')],
      ['cudaFuncGetAttributes', cast('pearl_tile_fold_wmma_rg')],
      ['cudaOccupancyMaxActiveClusters', 'pearl_tile_fold_hopper'],
    ]);
    expect(sites.filter(([, k]) => k === 'foldFn').map(([f]) => f)).toEqual(['cudaFuncGetAttributes', 'cudaFuncSetAttribute',
      'cudaFuncSetAttribute', 'cudaOccupancyMaxActiveBlocksPerMultiprocessor', 'cudaOccupancyMaxActiveClusters']);
    // The launch: the second kernel when the card picked it, with the same arguments.
    const launch = (k) => `${k}<<<blocks, threads, smem, st>>>(`;
    expect(count(launch('pearl_tile_fold_wmma'))).toBe(1);
    expect(count(launch('pearl_tile_fold_wmma_rg'))).toBe(1);
    inOrder(HOST_LINES, ['#if PEARL_BD_RG_KERNEL_BUILT', 'else if (ctx->bdHashRg)', launch('pearl_tile_fold_wmma_rg'),
      'ctx->dAp, ctx->dBp, ctx->profile.m, ctx->profile.n, k, rank, chunks,',
      'col_off, ctx->rowsValid, col_groups, tiles, test, hitList);', '#endif', 'else', launch('pearl_tile_fold_wmma'),
      'ctx->dAp, ctx->dBp, ctx->profile.m, ctx->profile.n, k, rank, chunks,',
      'col_off, ctx->rowsValid, col_groups, tiles, test, hitList);'], HOST_LINES.indexOf(launch('pearl_tile_fold_tall')));
    // bdHashRg: false until create sets it, once, on a B-direct card, when the rule (or
    // the sweep) picks it and its sm_75 body is what loaded. Before the first search,
    // which sets the shared opt-in and the occupancy on foldFn.
    expect(HOST).toContain('  bool bdHashRg = false;' + NL);
    expect(HOST.match(/bdHashRg = .*/g)).toEqual(['bdHashRg = false;', 'bdHashRg = true;']);
    const create = HOST_LINES.indexOf('if (ctx->foldBDirect) {', HOST_LINES.indexOf('const PearlBdLaunch rule = s;') - 20);
    inOrder(HOST_LINES, ['if (ctx->foldBDirect) {', 'if (cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, ctx->device) != cudaSuccess) {',
      'pearl_bd_launch_for(profile->k, (uint64_t)l2, (uint32_t)sms, ctx->colBatch);', 'const PearlBdLaunch rule = s;',
      'const bool swept = pearl_bd_sweep(&s, ctx->colsValid);', '#if PEARL_BD_RG_KERNEL_BUILT', 'if (s.hashRg) {',
      'cudaFuncAttributes fr;', `if (cudaFuncGetAttributes(&fr, ${cast('pearl_tile_fold_wmma_rg')})`, '== cudaSuccess',
      '&& fr.binaryVersion == 75)', 'ctx->bdHashRg = true;', 'else', '(void)cudaGetLastError();', '}', '#endif',
      // The sweep line comes after the check, and prints the kernel that will run.
      's.hashRg = ctx->bdHashRg;', 'if (swept)', 'fprintf(stderr,', '"[pearl] launch sweep: band %u (rule %u), col_batch %u (rule %u), streams %d "',
      'ctx->bdBand = s.band;'], create);
    expect(HOST.split('pearl_bd_sweep(&s, ').length - 1).toBe(1);
    expect(HOST.indexOf('ctx->bdHashRg = true;')).toBeLessThan(HOST.indexOf('extern "C" bool pearl_host_search'));
    // The declaration matches the kernel's, under the same #if.
    expect(HOST).toContain('#if PEARL_BD_RG_KERNEL_BUILT' + NL);
    expect(HOST).toContain('extern "C" __global__ void pearl_tile_fold_wmma_rg(' + NL
      + '    const int8_t *Aprime, const int8_t *Bprime, uint32_t m, uint32_t n,');
  });

  test('the probes and CI see the second kernel', () => {
    // compilers.sh: the second kernel's code hash, registers and spill on the build line,
    // or rg=none; the expect check, fold.hash and the patch stay on the first.
    expect(COMPILERS).toContain('rg_hash() {');
    expect(COMPILERS).toContain('awk \'$1 == "pearl_tile_fold_wmma_rg" {print $2; exit}\'');
    expect(COMPILERS).toContain('rg=$(rg_hash "$K/build/Release/pearl_core.node")');
    expect(COMPILERS).toContain(' $rg runs=$runs fold=$h $([ -z "$expect" ] && echo "expect=-" || { [ "$h" = "$expect" ]');
    expect(COMPILERS).toContain('    rg="rg=none"');
    expect(COMPILERS).toContain('  echo "$h" > "$K/fold.hash"');
    expect(COMPILERS).toContain('awk \'/Compiling entry function .pearl_tile_fold_wmma_rg. for .sm_75./{f=1;next} f&&/Compiling entry/{f=0} f\'');
    // Its pattern for the first kernel needs " for" after the name, so the second does
    // not match it.
    const ptxas = (k) => `ptxas info    : Compiling entry function '${k}' for 'sm_75'`;
    const first = /Compiling entry function .pearl_tile_fold_wmma. for .sm_75./;
    expect([first.test(ptxas('pearl_tile_fold_wmma')), first.test(ptxas('pearl_tile_fold_wmma_rg'))]).toEqual([true, false]);
    // launch.sh matched /Compiling entry function .pearl_tile_fold_wmma./, which the second
    // kernel matches too, and ptxas compiles the second first. kstat matches the quoted name.
    expect(/Compiling entry function .pearl_tile_fold_wmma./.test(ptxas('pearl_tile_fold_wmma_rg'))).toBe(true);
    expect(LAUNCH).toContain('awk -v k="\'$1\'" \'/Compiling entry function/ { on = index($0, "function " k " ") > 0; next }');
    expect(LAUNCH).toContain('fold=$(kstat pearl_tile_fold_wmma); rgfold=$(kstat pearl_tile_fold_wmma_rg)');
    const kstat = (k, line) => line.includes(`function '${k}' `);
    expect([kstat('pearl_tile_fold_wmma', ptxas('pearl_tile_fold_wmma')),
      kstat('pearl_tile_fold_wmma', ptxas('pearl_tile_fold_wmma_rg')),
      kstat('pearl_tile_fold_wmma_rg', ptxas('pearl_tile_fold_wmma_rg'))]).toEqual([true, false, true]);
    // On a 4 MB card a default build runs the second kernel, which fold=, the expect check
    // and the patch do not see. A build with a patch list gets policy 0, so the patched
    // first kernel is what runs; the build line's runs= says which kernel runs.
    const addPolicy = '  if [ -n "$patch" ] && [[ "$flags" != *PEARL_BD_RG_KERNEL=* ]]; then' + NL
      + '    flags="${flags:+$flags }-DPEARL_BD_RG_KERNEL=0"' + NL + '  fi' + NL;
    const runsCase = '  case " $flags " in' + NL + '    *PEARL_BD_RG_KERNEL=0*) runs=fold ;;' + NL
      + '    *PEARL_BD_RG_KERNEL=1*) runs=rg ;;' + NL + "    *) runs='rg_if_L2>=4MB' ;;" + NL + '  esac' + NL
      + '  [ "$rg" = rg=none ] && runs=fold' + NL;
    expect(COMPILERS.split(addPolicy).length - 1).toBe(1);
    expect(COMPILERS.split(runsCase).length - 1).toBe(1);
    // Before the first nvcc, which the patch replay repeats with the same flags.
    const bv = COMPILERS.slice(COMPILERS.indexOf('build_variant() {'));
    expect(bv.indexOf(addPolicy)).toBeLessThan(bv.indexOf('if ! nvcc $COMMON $flags'));
    expect(bv.indexOf(runsCase)).toBeLessThan(bv.indexOf('say "build $name regs='));
    // The default builds race the first kernel on every card.
    const list = COMPILERS.slice(COMPILERS.indexOf('VARIANTS=(' + NL), COMPILERS.indexOf(NL + ')' + NL));
    const entries = list.split(NL).slice(1).map((l) => l.trim().replace(/^"|"$/g, '').split('|'));
    expect(entries.map((e) => e[0])).toEqual(['base', 'rul8', 'cmr255', 'cmr255rul8']);
    for (const e of entries) expect([e[0], e[1].startsWith('-DPEARL_BD_RG_KERNEL=0')]).toEqual([e[0], true]);
    // The driver probe's hashrate.js runs the first kernel, the one its fold= hashes.
    expect(COMPILERS).toContain('PEARL_BD_SWEEP_HASH_RG=0 timeout -k 10 120 node native/probes/hashrate.js');
    // Both pieces run under bash, where there is one.
    if (spawnSync('bash', ['--version'], { stdio: 'ignore' }).status === 0) {
      const run = (patch, flags, rg) => {
        const r = spawnSync('bash', ['-c', addPolicy + runsCase + 'printf "%s|%s" "$flags" "$runs"'],
          { env: { PATH: process.env.PATH, patch, flags, rg }, encoding: 'utf8' });
        return r.stdout;
      };
      const RG = 'rg=ac1a72f2434b rg_regs=243 rg_spill=0/0';
      for (const [patch, flags, rg, want] of [
        ['', '', RG, '|rg_if_L2>=4MB'],
        ['', '-DPEARL_BD_RG_KERNEL=0', RG, '-DPEARL_BD_RG_KERNEL=0|fold'],
        ['', '-DPEARL_BD_RG_KERNEL=1', RG, '-DPEARL_BD_RG_KERNEL=1|rg'],
        ['', '-DPEARL_BD_LAST_RO=1', RG, '-DPEARL_BD_LAST_RO=1|rg_if_L2>=4MB'],
        ['', '-DPEARL_BD_RED=1', 'rg=none', '-DPEARL_BD_RED=1|fold'],
        ['sm75/sbfix1.json', '', RG, '-DPEARL_BD_RG_KERNEL=0|fold'],
        ['sm75/sbfix1.json', '-Xptxas -O3', RG, '-Xptxas -O3 -DPEARL_BD_RG_KERNEL=0|fold'],
        ['sm75/sbfix1.json', '-DPEARL_BD_RG_KERNEL=2', RG, '-DPEARL_BD_RG_KERNEL=2|rg_if_L2>=4MB'],
      ]) {
        expect([patch, flags, run(patch, flags, rg)]).toEqual([patch, flags, want]);
      }
    }
    // CI's spill check reads it and fails when ptxas printed nothing for it.
    expect(CI).toContain('              pearl_tile_fold_wmma_rg) seen_wmma_rg=1 ;;' + NL);
    expect(CI).toContain('[ "$seen_wmma_rg" != 1 ]');
  });
});
