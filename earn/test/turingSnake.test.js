'use strict';

const fs = require('fs');
const path = require('path');

// Turing's snake IMMA order (PEARL_BD_SNAKE, off by default). In k-lo first's
// k-step it issues each m-block's four IMMA of an n-block pair in a row (the even
// n-block's rows 0-7 and 8-15, then the odd n-block's rows 8-15 and 0-7) in both
// passes, and loads B's quads before A's. Nothing here can run the device code, so
// this copies the k-step's code in both orders, pins the copy to the source line
// for line, and runs it on the CPU over random A' and B'. It checks word for word
// that the snake issues the default's 128 IMMA a k-step: each accumulator half
// gets the A and B words pearl_mma_m16n8k32 gives it, k-lo first and k-hi 64 IMMA
// later, and each k-lo IMMA takes the B word gated on the fold that reads its
// accumulator. The guards go through a small preprocessor. The draw that lays out
// A' and B' is copied from turingKorder.test.js, which pins it to the source.

const SRC = path.join(__dirname, '..', 'native', 'src');
const NL = String.fromCharCode(10);
const BS = String.fromCharCode(92);
// Line endings normalised: a Windows checkout has CRLF.
const read = (name) => fs.readFileSync(path.join(SRC, name), 'utf8')
  .split(String.fromCharCode(13) + NL).join(NL);
const KERNEL = read('pearl_kernel.cu');
const FOLD = read('pearl_fold_bd.cuh');
const HEADER = read('pearl_config.h');

// The fold's constants (pinned below): four m-blocks, eight n-blocks, two regions a
// lane, four k-steps a chunk.
const MB = 4;
const NB = 8;
const RPL = 2;
const KSTEPS = 4;

// The fold's lines without comments: comment lines dropped, trailing ones cut.
function code(text) {
  return text.split(NL).filter((l) => !l.trim().startsWith('//')).map((l) => {
    const at = l.indexOf('//');
    return at > 0 && l.slice(0, at).trim() ? l.slice(0, at).trimEnd() : l;
  });
}
const CODE = code(FOLD);

// Pins want to the fold's code: it starts at the only line equal to want[0] and
// goes on line for line. Returns where it starts.
function pin(want) {
  const at = CODE.indexOf(want[0]);
  expect(at).toBeGreaterThanOrEqual(0);
  expect(CODE.lastIndexOf(want[0])).toBe(at);
  expect(CODE.slice(at, at + want.length)).toEqual(want);
  return at;
}

// The copy. Each part is its lines before the #if PEARL_BD_SNAKE, the snake's lines,
// the default's (under #else) and its lines after the #endif.
const switched = (part) => [...part.head, '#if PEARL_BD_SNAKE', ...part.snake, '#else',
  ...part.dflt, '#endif', ...part.tail];
const LOAD_B = '      for (uint32_t q = hi; q < 4u; q += 2u) bf[nxt][q] = ldg(Bprime + bo + q * FRAG_BLOCK);';
const LOAD_A = '      for (uint32_t q = hi; q < 4u; q += 2u) af[nxt][q] = ldg(Aprime + ao + q * FRAG_BLOCK);';
const LOAD_HALF = {
  head: ['    auto load_half = [&](uint32_t hi) {'],
  snake: ['#pragma unroll', LOAD_B, '#pragma unroll', LOAD_A],
  dflt: ['#pragma unroll', LOAD_A, '#ifndef PEARL_ABLATE_BLOAD', '#pragma unroll', LOAD_B, '#else',
    '      (void)bo;', '      asm volatile("" ::: "memory");', '#endif'],
  tail: ['    };'],
};
const WORDS = [
  '    auto a_word = [&](uint32_t mb, uint32_t hi, uint32_t u) -> uint32_t {',
  '      const int4 v = af[cur][(mb / 2u) * 2u + hi];',
  '      return (uint32_t)(mb & 1u ? (u ? v.w : v.z) : (u ? v.y : v.x));',
  '    };',
  '    auto b_word = [&](uint32_t j, uint32_t hi, uint32_t odd) -> uint32_t {',
  '      const int4 v = bf[cur][(j / 2u) * 2u + hi];',
  '      return (uint32_t)(j & 1u ? (odd ? v.w : v.z) : (odd ? v.y : v.x));',
  '    };',
];
const MB_LOOP = '      for (uint32_t mb = 0; mb < MB; mb++) {';
const KLO = {
  head: [
    '    load_half(0u);',
    '#pragma unroll',
    '    for (uint32_t j = 0; j < 4u; j++) {',
    '      uint32_t b0[RPL], b2[RPL];',
    '#pragma unroll',
    '      for (uint32_t rl = 0; rl < RPL; rl++) {',
    '        b0[rl] = b_word(j, 0u, 0u);',
    '        b2[rl] = b_word(j, 0u, 1u);',
    '      }',
    '      if (ro) {',
    '#pragma unroll',
    '        for (uint32_t rl = 0; rl < RPL; rl++) rxs[rl] = fold_pair(rxs[rl], rl, j);',
    '#pragma unroll',
    '        for (uint32_t rl = 0; rl < RPL; rl++) {',
    '          b0[rl] = pearl_bd_after(b0[rl], rxs[rl]);',
    '#if PEARL_BD_GATE == 2',
    '          b2[rl] = pearl_bd_after(b2[rl], rxs[rl]);',
    '#endif',
    '        }',
    '      }',
  ],
  // The 8 snake IMMA lines of the k-lo pass, then the default's.
  snake: [
    '#pragma unroll',
    MB_LOOP,
    '        pearl_bd_imma(acc[mb][2u * j][0], acc[mb][2u * j][1], a_word(mb, 0u, 0u), b0[mb / 2u]);',
    '        pearl_bd_imma(acc[mb][2u * j][2], acc[mb][2u * j][3], a_word(mb, 0u, 1u), b0[mb / 2u]);',
    '        pearl_bd_imma(acc[mb][2u * j + 1u][2], acc[mb][2u * j + 1u][3], a_word(mb, 0u, 1u),',
    '                      b2[mb / 2u]);',
    '        pearl_bd_imma(acc[mb][2u * j + 1u][0], acc[mb][2u * j + 1u][1], a_word(mb, 0u, 0u),',
    '                      b2[mb / 2u]);',
    '      }',
  ],
  dflt: [
    '#pragma unroll',
    MB_LOOP,
    '        pearl_bd_imma(acc[mb][2u * j][0], acc[mb][2u * j][1], a_word(mb, 0u, 0u), b0[mb / 2u]);',
    '        pearl_bd_imma(acc[mb][2u * j][2], acc[mb][2u * j][3], a_word(mb, 0u, 1u), b0[mb / 2u]);',
    '      }',
    '#pragma unroll',
    MB_LOOP,
    '        pearl_bd_imma(acc[mb][2u * j + 1u][0], acc[mb][2u * j + 1u][1], a_word(mb, 0u, 0u),',
    '                      b2[mb / 2u]);',
    '        pearl_bd_imma(acc[mb][2u * j + 1u][2], acc[mb][2u * j + 1u][3], a_word(mb, 0u, 1u),',
    '                      b2[mb / 2u]);',
    '      }',
  ],
  tail: ['    }'],
};
const KHI = {
  head: [
    '    load_half(1u);',
    '#pragma unroll',
    '    for (uint32_t j = 0; j < 4u; j++) {',
    '      const uint32_t b1 = b_word(j, 1u, 0u), b3 = b_word(j, 1u, 1u);',
  ],
  snake: [
    '#pragma unroll',
    MB_LOOP,
    '        pearl_bd_imma(acc[mb][2u * j][0], acc[mb][2u * j][1], a_word(mb, 1u, 0u), b1);',
    '        pearl_bd_imma(acc[mb][2u * j][2], acc[mb][2u * j][3], a_word(mb, 1u, 1u), b1);',
    '        pearl_bd_imma(acc[mb][2u * j + 1u][2], acc[mb][2u * j + 1u][3], a_word(mb, 1u, 1u), b3);',
    '        pearl_bd_imma(acc[mb][2u * j + 1u][0], acc[mb][2u * j + 1u][1], a_word(mb, 1u, 0u), b3);',
    '      }',
  ],
  dflt: [
    '#pragma unroll',
    MB_LOOP,
    '        pearl_bd_imma(acc[mb][2u * j][0], acc[mb][2u * j][1], a_word(mb, 1u, 0u), b1);',
    '        pearl_bd_imma(acc[mb][2u * j][2], acc[mb][2u * j][3], a_word(mb, 1u, 1u), b1);',
    '      }',
    '#pragma unroll',
    MB_LOOP,
    '        pearl_bd_imma(acc[mb][2u * j + 1u][0], acc[mb][2u * j + 1u][1], a_word(mb, 1u, 0u), b3);',
    '        pearl_bd_imma(acc[mb][2u * j + 1u][2], acc[mb][2u * j + 1u][3], a_word(mb, 1u, 1u), b3);',
    '      }',
  ],
  // The k-hi pass ends k-lo first's k-step: PEARL_BD_KORDER's #else follows.
  tail: ['    }', '#else'],
};
// The region of the fold the copy is, from load_half to PEARL_BD_KORDER's #else.
const KSTEP = [...switched(LOAD_HALF), ...WORDS, ...switched(KLO), ...switched(KHI)];
// fold_pair, which the k-lo pass's gate waits for: it reads all four words of the
// region's two m-blocks on the pair's two n-blocks.
const FOLD_PAIR = [
  '  auto fold_pair = [&](uint32_t x, uint32_t rl, uint32_t np) -> uint32_t {',
  '    const int32_t *p0 = acc[2u * rl][2u * np], *p1 = acc[2u * rl + 1u][2u * np];',
  '    const int32_t *p2 = acc[2u * rl][2u * np + 1u], *p3 = acc[2u * rl + 1u][2u * np + 1u];',
  '    const uint32_t ta = pearl_xor3((uint32_t)p0[0], (uint32_t)p0[1], (uint32_t)p0[2]);',
  '    const uint32_t tb = pearl_xor3((uint32_t)p0[3], (uint32_t)p1[0], (uint32_t)p1[1]);',
  '    const uint32_t tc = pearl_xor3((uint32_t)p1[2], (uint32_t)p1[3], (uint32_t)p2[0]);',
  '    const uint32_t td = pearl_xor3((uint32_t)p2[1], (uint32_t)p2[2], (uint32_t)p2[3]);',
  '    const uint32_t te = pearl_xor3(ta, tb, tc);',
  '    const uint32_t rest = np == 0u ? (te ^ td) : pearl_xor3(te, td, x);',
  '    const uint32_t t3 = pearl_xor3((uint32_t)p3[0], (uint32_t)p3[1], (uint32_t)p3[2]);',
  '    return pearl_xor3(t3, (uint32_t)p3[3], rest);',
  '  };',
];
function foldReads(rl, np) {
  const out = [];
  for (const mb of [2 * rl, 2 * rl + 1]) {
    for (const n of [2 * np, 2 * np + 1]) {
      for (let w = 0; w < 4; w++) out.push(`${mb}.${n}.${w}`);
    }
  }
  return out;
}

// A call's arguments, split at its top-level commas.
function args(call) {
  const out = [];
  let depth = 0;
  let cur = '';
  for (const ch of call) {
    if (ch === '(' || ch === '[') depth++;
    if (ch === ')' || ch === ']') depth--;
    if (ch === ',' && depth === 0) {
      out.push(cur.trim());
      cur = '';
    } else {
      cur += ch;
    }
  }
  out.push(cur.trim());
  return out;
}

// One pearl_bd_imma call: n-block 2j + odd, accumulator words i and i + 1, A word
// a_word(mb, hi, u), and B register b (b0, b2: one copy a region, [mb / 2u]).
function decode(call) {
  const a = args(call);
  expect(a.length).toBe(4);
  const acc = (s) => {
    const parts = s.split('][');
    expect(parts.length).toBe(3);
    expect(parts[0]).toBe('acc[mb');
    expect(['2u * j', '2u * j + 1u']).toContain(parts[1]);
    return { odd: parts[1] === '2u * j' ? 0 : 1, i: parseInt(parts[2], 10) };
  };
  const c0 = acc(a[0]);
  expect([0, 2]).toContain(c0.i);
  expect(acc(a[1])).toEqual({ odd: c0.odd, i: c0.i + 1 });
  expect(a[2].startsWith('a_word(mb, ')).toBe(true);
  const [hi, u] = a[2].slice('a_word(mb, '.length, -1).split(', ').map((s) => parseInt(s, 10));
  const b = a[3].split('[')[0];
  const copy = a[3].slice(b.length);
  expect(['b0', 'b1', 'b2', 'b3']).toContain(b);
  expect(['', '[mb / 2u]']).toContain(copy);
  return { odd: c0.odd, i: c0.i, hi, u, b, perRegion: copy !== '' };
}

// A pass body's loops over the m-blocks, each the list of its calls in order.
function loops(body) {
  const text = body.filter((l) => l !== '#pragma unroll').map((l) => l.trim()).join(' ');
  const head = MB_LOOP.trim();
  expect(text.startsWith(head)).toBe(true);
  return text.split(head).slice(1).map((s) => {
    const inner = s.trim();
    expect(inner.endsWith('}')).toBe(true);
    const calls = inner.slice(0, -1).split('pearl_bd_imma(');
    expect(calls[0].trim()).toBe('');
    return calls.slice(1).map((c) => {
      expect(c.trim().endsWith(');')).toBe(true);
      return decode(c.trim().slice(0, -2));
    });
  });
}

// A k-step's 128 IMMA in issue order, for the snake or the default: the k-lo pass,
// then the k-hi pass, each running its loops over the m-blocks for each pair j.
// acc names the accumulator half: m-block, n-block, half (words 2 * half and up).
function order(branch) {
  const out = [];
  for (const [pass, part] of [[0, KLO], [1, KHI]]) {
    const ls = loops(part[branch]);
    for (let j = 0; j < 4; j++) {
      for (const calls of ls) {
        for (let mb = 0; mb < MB; mb++) {
          for (const c of calls) {
            const n = 2 * j + c.odd;
            out.push({ ...c, pass, j, mb, n, acc: `${mb}.${n}.${c.i / 2}` });
          }
        }
      }
    }
  }
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

function bytes(n, seed) {
  const next = xorshift(seed);
  const a = new Uint8Array(n);
  for (let i = 0; i < n; i++) a[i] = next() & 0xff;
  return a;
}

function word(b, at) {
  return (b[at] | (b[at + 1] << 8) | (b[at + 2] << 16) | (b[at + 3] << 24)) >>> 0;
}

// The draw's store, as turingKorder.test.js copies it (pearl_materialize16_frag and
// _afrag): thread v holds the sixteen row-major bytes at v * 16 and stores word w at
// o[w * 4], o = out + blk * 512 + (r & 7) * 64 + reg * 4 bytes. paired is
// PEARL_BD_KORDER's layout; unpaired, isA picks A's word order or B's.
function draw(src, rows, kLog2, isA, paired) {
  const k = 2 ** kLog2;
  const out = new Uint8Array(rows * k);
  for (let v = 0; v < (rows * k) / 16; v++) {
    const idx = v * 16;
    const r = Math.floor(idx / k);
    const kk = idx & (k - 1);
    let blk;
    let reg;
    const bl = r >> 4;
    if (paired && (bl | 1) < (rows >> 4)) {
      const dbl = (bl & ~1) | ((kk >> 4) & 1);
      blk = dbl * 2 ** (kLog2 - 5) + (kk >> 5);
      reg = (bl & 1) * 2 + ((r >> 3) & 1);
    } else {
      blk = bl * 2 ** (kLog2 - 5) + (kk >> 5);
      reg = isA ? ((kk >> 4) & 1) * 2 + ((r >> 3) & 1) : ((r >> 3) & 1) * 2 + ((kk >> 4) & 1);
    }
    const o = blk * 512 + (r & 7) * 64 + reg * 4;
    for (let w = 0; w < 4; w++) {
      for (let j = 0; j < 4; j++) out[o + w * 16 + j] = src[idx + 4 * w + j];
    }
  }
  return out;
}

// A lane's four words of 16-row block blk at k-step s (the fold's LDG.128).
function quad(frag, kLog2, blk, s, lane) {
  const at = blk * (2 ** (kLog2 - 5)) * 512 + s * 512 + lane * 16;
  return [word(frag, at), word(frag, at + 4), word(frag, at + 8), word(frag, at + 12)];
}

// prmt.b32 d, a, b, sel with no sign bits set (each nibble of sel under 8): byte i
// of d is byte (sel >> 4i) & 7 of b:a, a's bytes 0-3 and b's 4-7. pearl_bd_after
// is prmt(b, x, 0x3210).
function prmt(a, b, sel) {
  let d = 0;
  for (let i = 0; i < 4; i++) {
    const n = (sel >>> (4 * i)) & 7;
    d |= (((n < 4 ? a : b) >>> (8 * (n & 3))) & 0xff) << (8 * i);
  }
  return d >>> 0;
}

// The copy's k-step for the snake or the default, decoded: load_half's loads in
// order, and each pass's loops.
function plan(branch) {
  return {
    loads: LOAD_HALF[branch].filter((l) => l === LOAD_A || l === LOAD_B),
    klo: loops(KLO[branch]),
    khi: loops(KHI[branch]),
  };
}

// One lane through k-steps 0 to 2^(kLog2 - 5) - 1 of a tile, as the copy runs:
// tile_prologue's loads into buffer 0, then each k-step's load_half(0u), k-lo pass,
// load_half(1u) and k-hi pass, with af and bf on t & 1 and the loads a k-step ahead
// (the last k-step's from k-step 0 again, the non-persistent wrap). With ro the
// k-lo pass's B words go through pearl_bd_after with a random fold value, b2 only at
// gate 2. Returns each k-step's IMMA as [accumulator, A word, B word] in issue order.
function run(p, A, B, kLog2, aBase, bBase, lane, ro, gate, rand) {
  const steps = 2 ** (kLog2 - 5);
  const loadA = (s, q) => quad(A, kLog2, aBase + q, s, lane);
  const loadB = (s, q) => quad(B, kLog2, bBase + q, s, lane);
  const { loads, klo: kloLoops, khi: khiLoops } = p;
  const af = [[], []];
  const bf = [[], []];
  for (let q = 0; q < 4; q++) {
    af[0][q] = loadA(0, q);
    bf[0][q] = loadB(0, q);
  }
  const out = [];
  for (let s = 0; s < steps; s++) {
    const t = s % KSTEPS;
    const cur = t & 1;
    const nxt = (t + 1) & 1;
    const from = (s + 1) % steps;
    const loadHalf = (hi) => {
      for (const l of loads) {
        for (let q = hi; q < 4; q += 2) {
          if (l === LOAD_A) af[nxt][q] = loadA(from, q);
          else bf[nxt][q] = loadB(from, q);
        }
      }
    };
    const aWord = (mb, hi, u) => {
      const v = af[cur][Math.floor(mb / 2) * 2 + hi];
      return mb & 1 ? (u ? v[3] : v[2]) : (u ? v[1] : v[0]);
    };
    const bWord = (j, hi, odd) => {
      const v = bf[cur][Math.floor(j / 2) * 2 + hi];
      return j & 1 ? (odd ? v[3] : v[2]) : (odd ? v[1] : v[0]);
    };
    const imma = [];
    const issue = (ls, j, regs, pass) => {
      for (const calls of ls) {
        for (let mb = 0; mb < MB; mb++) {
          for (const c of calls) {
            const b = c.perRegion ? (regs[c.b] || [])[Math.floor(mb / 2)] : regs[c.b];
            imma.push([`${mb}.${2 * j + c.odd}.${c.i / 2}.${pass}`, aWord(mb, c.hi, c.u), b]);
          }
        }
      }
    };
    loadHalf(0);
    for (let j = 0; j < 4; j++) {
      const b0 = [];
      const b2 = [];
      for (let rl = 0; rl < RPL; rl++) {
        b0[rl] = bWord(j, 0, 0);
        b2[rl] = bWord(j, 0, 1);
      }
      if (ro) {
        for (let rl = 0; rl < RPL; rl++) {
          const x = rand();
          b0[rl] = prmt(b0[rl], x, 0x3210);
          if (gate === 2) b2[rl] = prmt(b2[rl], x, 0x3210);
        }
      }
      issue(kloLoops, j, { b0, b2 }, 0);
    }
    loadHalf(1);
    for (let j = 0; j < 4; j++) issue(khiLoops, j, { b1: bWord(j, 1, 0), b3: bWord(j, 1, 1) }, 1);
    out.push(imma);
  }
  return out;
}

// The default fold's words (PEARL_BD_KORDER=0, unpaired draw): pearl_mma_m16n8k32
// takes a0-a3 from m-block mb's block and b0, b1 (b2, b3 for the odd n-block) from
// pair j's, and adds c01 += a0 b0, a2 b1; c23 += a1 b0, a3 b1. By accumulator half
// and pass (k-lo, k-hi): [A word, B word].
function reference(A0, B0, kLog2, aBase, bBase, s, lane) {
  const out = new Map();
  for (let mb = 0; mb < MB; mb++) {
    const a = quad(A0, kLog2, aBase + mb, s, lane);
    for (let j = 0; j < 4; j++) {
      const b = quad(B0, kLog2, bBase + j, s, lane);
      for (let odd = 0; odd < 2; odd++) {
        for (let half = 0; half < 2; half++) {
          for (let hi = 0; hi < 2; hi++) {
            out.set(`${mb}.${2 * j + odd}.${half}.${hi}`, [a[hi * 2 + half], b[odd * 2 + hi]]);
          }
        }
      }
    }
  }
  return out;
}

// A small C preprocessor for the guards: object-like #define, #undef, #if, #ifdef,
// #ifndef, #elif, #else, #endif and #error, with -D flags (a bare name is 1).
// Returns the #error lines it reaches.
function tokens(expr) {
  const out = [];
  const word = (c) => (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')
    || c === '_';
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
  const isDigit = (c) => c >= '0' && c <= '9';
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
    if (isDigit(tk[0])) {
      let s = tk;
      while ('uUlL'.includes(s[s.length - 1])) s = s.slice(0, -1);
      const v = Number(s);
      if (!Number.isFinite(v)) throw new Error(`not a number: ${tk}`);
      return v;
    }
    if (')!<>+*/%?:&|='.includes(tk[0])) throw new Error(`unexpected ${tk} in ${expr}`);
    // An identifier: its definition, or 0 if it has none.
    if (!defs.has(tk) || seen.has(tk)) return 0;
    if (defs.get(tk) === null) throw new Error(`function-like ${tk} in ${expr}`);
    return evaluate(defs.get(tk), defs, new Set([...seen, tk]));
  };
  const binary = (next, ops, f) => () => {
    let v = next();
    for (;;) {
      const op = ops.find((o) => t[p] === o);
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
      const name = rest.slice(0, end);
      defs.set(name, rest[end] === '(' ? null : rest.slice(end).trim());
    } else if (dir === 'undef') {
      defs.delete(rest);
    } else if (dir === 'error') {
      errors.push(rest);
    }
  }
  expect(stack).toEqual([]);
  return errors;
}

// The guards a build of the fold goes through: the header's Turing block (its
// switches, their defaults and #error lines, PEARL_BD_SNAKE's among them) and the
// fold's own #error lines.
function guards() {
  const slice = (src, from, to) => {
    const at = src.indexOf(from);
    const end = src.indexOf(to, at);
    expect(at).toBeGreaterThanOrEqual(0);
    expect(end).toBeGreaterThan(at);
    return src.slice(at, end);
  };
  return slice(HEADER, '#ifndef PEARL_TURING_WIDE' + NL, '#define PEARL_BD_SPLIT_SMEM') + NL
    + slice(FOLD, '#if PEARL_BD_A_DIRECT && (defined(PEARL_ABLATE_BARRIER)', '#if PEARL_BD_GROUPS == 2');
}

describe('Turing snake IMMA order (PEARL_BD_SNAKE)', () => {
  test('is off by default and alone builds on LEAN, needs k-lo first, and stacks only where the gates pass', () => {
    const at = HEADER.indexOf('#ifndef PEARL_BD_SNAKE' + NL);
    expect(at).toBeGreaterThanOrEqual(0);
    expect(HEADER.slice(at).split(NL)[1].trim()).toBe('#define PEARL_BD_SNAKE 0');
    expect(HEADER).toContain('#if PEARL_BD_SNAKE != 0 && PEARL_BD_SNAKE != 1' + NL
      + '#error "PEARL_BD_SNAKE is 0 or 1"');
    expect(HEADER).toContain('#if PEARL_BD_SNAKE && !PEARL_BD_KORDER' + NL
      + '#error "PEARL_BD_SNAKE orders k-lo first');
    expect(HEADER).toContain('#if PEARL_BD_SNAKE && PEARL_BD_RED ' + BS + NL
      + '    && !(PEARL_BD_RUNBASE && PEARL_BD_TAIL_LDG && PEARL_BD_GATE == 1)'
      + NL + '#error "PEARL_BD_SNAKE stacks with PEARL_BD_RED only with');
    expect(FOLD).toContain('#if PEARL_BD_SNAKE && defined(PEARL_ABLATE_BLOAD)' + NL + '#error ');
    expect(HEADER).toContain('#if PEARL_BD_HASH_SPLIT && (PEARL_BD_RED || PEARL_BD_TADDR || PEARL_BD_SNAKE');
    // The preprocessor reads the defaults as the compiler does.
    const GUARDS = guards();
    const errorsFor = (flags) => preprocess(GUARDS, flags);
    const value = (cond, flags) => (preprocess(GUARDS + NL + '#if ' + cond + NL + '#error yes' + NL
      + '#endif', flags).includes('yes') ? 1 : 0);
    // With no flags: k-lo first with LEAN (its parts) and HIT_COORDS, as the default
    // fold builds. OFF turns both off, and with them every part.
    const OFF = { PEARL_BD_LEAN: 0, PEARL_BD_HIT_COORDS: 0 };
    expect(value('PEARL_BD_SNAKE == 0', {})).toBe(1);
    expect(value('PEARL_BD_KORDER == 1 && PEARL_BD_LEAN && PEARL_BD_HIT_COORDS && PEARL_BD_GATE == 1 '
      + '&& PEARL_BD_RUNBASE && PEARL_BD_TAIL_LDG', {})).toBe(1);
    expect(value('PEARL_BD_KORDER == 1 && PEARL_BD_GATE == 2 && !PEARL_BD_RUNBASE && !PEARL_BD_TAIL_LDG', OFF))
      .toBe(1);
    expect(value('PEARL_BD_GATE == 1 && PEARL_BD_RUNBASE && PEARL_BD_TAIL_LDG',
      { PEARL_BD_LEAN: 1, PEARL_BD_HIT_COORDS: 0 })).toBe(1);
    // SNAKE alone stacks on LEAN and HIT_COORDS.
    expect(value('PEARL_BD_SNAKE && PEARL_BD_LEAN && PEARL_BD_HIT_COORDS', { PEARL_BD_SNAKE: 1 })).toBe(1);
    // A_DIRECT=0 alone builds the older fold, with LEAN and HIT_COORDS off.
    expect(value('PEARL_BD_KORDER == 0 && !PEARL_BD_LEAN && !PEARL_BD_HIT_COORDS', { PEARL_BD_A_DIRECT: 0 }))
      .toBe(1);
    // The builds pearl_config.h names as passing the gates, and the plain ones.
    const S = { PEARL_BD_SNAKE: 1 };
    const allowed = [
      {},
      OFF,
      S,
      { ...S, ...OFF },
      { ...S, ...OFF, PEARL_BD_RUNBASE: 1 },
      { ...S, ...OFF, PEARL_BD_RUNBASE: 1, PEARL_BD_TAIL_LDG: 1 },
      // TAIL_LDG alone keeps SNAKE's own loop.
      { ...S, ...OFF, PEARL_BD_TAIL_LDG: 1 },
      { ...S, PEARL_BD_LEAN: 1, PEARL_BD_HIT_COORDS: 0 },
      { ...S, PEARL_BD_LEAN: 1, PEARL_BD_HIT_COORDS: 1 },
      { ...S, PEARL_BD_RED: 1, PEARL_BD_GATE: 1, PEARL_BD_HIT_COORDS: 0 },
      { ...S, PEARL_BD_RED: 1, PEARL_BD_GATE: 1, PEARL_BD_HIT_COORDS: 1 },
      { ...S, PEARL_BD_LEAN: 0, PEARL_BD_HIT_COORDS: 1 },
      { ...S, ...OFF, PEARL_BD_GATE: 1 },
      { ...S, ...OFF, PEARL_BD_TADDR: 1 },
      { ...S, PEARL_BD_PERSIST: 0 },
      { ...S, PEARL_BD_A_REALIGN: 1 },
      { ...S, PEARL_ABLATE_BHIT: 1 },
    ];
    for (const flags of allowed) expect([flags, errorsFor(flags)]).toEqual([flags, []]);
    // Refused by a guard that names the switch, and fine without it.
    const refused = [
      [{ PEARL_BD_SNAKE: 2 }, null],
      [{ ...S, PEARL_BD_KORDER: 0 }, { PEARL_BD_KORDER: 0 }],
      [{ ...S, PEARL_BD_A_DIRECT: 0 }, { PEARL_BD_A_DIRECT: 0 }],
      [{ ...S, PEARL_BD_RED: 1, PEARL_BD_HIT_COORDS: 0 }, { PEARL_BD_RED: 1, PEARL_BD_HIT_COORDS: 0 }],
      [{ ...S, PEARL_BD_RED: 1, PEARL_BD_HIT_COORDS: 1 }, { PEARL_BD_RED: 1, PEARL_BD_HIT_COORDS: 1 }],
      // RED at gate 1 without one load part or both: the next tile's loads after the barrier.
      [{ ...S, PEARL_BD_RED: 1, PEARL_BD_GATE: 1, PEARL_BD_RUNBASE: 0 },
        { PEARL_BD_RED: 1, PEARL_BD_GATE: 1, PEARL_BD_RUNBASE: 0 }],
      [{ ...S, PEARL_BD_RED: 1, PEARL_BD_GATE: 1, PEARL_BD_TAIL_LDG: 0 },
        { PEARL_BD_RED: 1, PEARL_BD_GATE: 1, PEARL_BD_TAIL_LDG: 0 }],
      [{ ...S, PEARL_BD_RED: 1, PEARL_BD_GATE: 1, PEARL_BD_RUNBASE: 0, PEARL_BD_TAIL_LDG: 0 },
        { PEARL_BD_RED: 1, PEARL_BD_GATE: 1, PEARL_BD_RUNBASE: 0, PEARL_BD_TAIL_LDG: 0 }],
      [{ ...S, PEARL_ABLATE_BLOAD: 1 }, { PEARL_ABLATE_BLOAD: 1 }],
      [{ ...S, PEARL_BD_HASH_SPLIT: 4, PEARL_BD_RUNBASE: 1, PEARL_BD_TAIL_LDG: 1 },
        { PEARL_BD_HASH_SPLIT: 4, PEARL_BD_RUNBASE: 1, PEARL_BD_TAIL_LDG: 1 }],
    ];
    for (const [flags, without] of refused) {
      const errors = errorsFor(flags);
      expect([flags, errors.length > 0 && errors.every((e) => e.includes('SNAKE'))]).toEqual([flags, true]);
      if (without) expect([without, errorsFor(without)]).toEqual([without, []]);
    }
  });

  test('the copy matches the source, line for line', () => {
    // load_half to the end of k-lo first's k-step, comments aside.
    const at = pin(KSTEP);
    // Inside k-lo first's k-step, after its buffers and the next k-step's offsets.
    const kstep = CODE.indexOf(
      '  auto kstep = [&](uint32_t ch, uint32_t t, bool ro, bool last, uint32_t dch = 0u) {');
    expect(kstep).toBeGreaterThan(0);
    const korder = CODE.indexOf('#if PEARL_BD_KORDER', kstep);
    expect(korder).toBeGreaterThan(kstep);
    const cur = CODE.indexOf('    const uint32_t cur = t & 1u, nxt = (t + 1u) & 1u;', korder);
    expect(cur).toBeGreaterThan(korder);
    expect(cur).toBeLessThan(at);
    for (const off of ['    const uint32_t ao = last ? aoffn : aoff + (s + 1u) * 512u;',
      '    const uint32_t bo = last ? boffn : boff + (s + 1u) * 512u;']) {
      expect(CODE.indexOf(off, cur)).toBeGreaterThan(cur);
      expect(CODE.indexOf(off, cur)).toBeLessThan(at);
    }
    // The copy has the fold's only #if PEARL_BD_SNAKE blocks but the ABLATE_BLOAD
    // guard, and all its IMMA: pearl_bd_imma's definition and these 16 calls.
    expect(CODE.filter((l) => l.trim().startsWith('#if PEARL_BD_SNAKE')).length).toBe(4);
    expect(KSTEP.filter((l) => l === '#if PEARL_BD_SNAKE').length).toBe(3);
    expect(FOLD.split('pearl_bd_imma(').length - 1).toBe(1 + 16);
    expect(KSTEP.join(NL).split('pearl_bd_imma(').length - 1).toBe(16);
    // The pieces the run copies: the gate, the IMMA, the default's mma and words, the
    // prologue's loads, the loads' wrapper and the fold's sizes.
    pin(FOLD_PAIR);
    expect(FOLD).toContain('asm("prmt.b32 %0, %1, %2, 0x3210;" : "=r"(r) : "r"(b), "r"(x));');
    expect(FOLD).toContain('__device__ __forceinline__ uint32_t pearl_bd_after(uint32_t b, uint32_t x) {');
    expect(FOLD).toContain('asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32'
      + ' {%0,%1}, {%2}, {%3}, {%0,%1};"'
      + NL + '               : "+r"(c0), "+r"(c1)' + NL + '               : "r"(a), "r"(b));');
    expect(FOLD).toContain(
      '__device__ __forceinline__ void pearl_bd_imma(int32_t &c0, int32_t &c1, uint32_t a, uint32_t b) {');
    for (const m of ['{%0,%1}, {%4}, {%8}', '{%0,%1}, {%6}, {%9}', '{%2,%3}, {%5}, {%8}',
      '{%2,%3}, {%7}, {%9}']) {
      expect(KERNEL).toContain('"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 ' + m + ', '
        + m.slice(0, 7) + ';' + BS + 'n"');
    }
    expect(KERNEL).toContain('    int32_t &c0, int32_t &c1, int32_t &c2, int32_t &c3,' + NL
      + '    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,' + NL + '    uint32_t b0, uint32_t b1) {');
    expect(FOLD).toContain('uint32_t b0 = (uint32_t)bf[t & 1u][j].x, b1 = (uint32_t)bf[t & 1u][j].y;');
    expect(FOLD).toContain('uint32_t b2 = (uint32_t)bf[t & 1u][j].z, b3 = (uint32_t)bf[t & 1u][j].w;');
    expect(FOLD).toContain('(uint32_t)a.w, b0, b1);');
    expect(FOLD).toContain('(uint32_t)a.y, (uint32_t)a.z, (uint32_t)a.w, b2, b3);');
    expect(FOLD).toContain(
      'for (uint32_t mb = 0; mb < MB; mb++) af[0][mb] = pearl_bd_ldg(Aprime + aoff + mb * FRAG_BLOCK);');
    expect(FOLD).toContain(
      'for (uint32_t j = 0; j < 4u; j++) bf[0][j] = pearl_bd_ldg(Bprime + boff + j * FRAG_BLOCK);');
    expect(FOLD).toContain('    auto ldg = [&](const int8_t *p) -> int4 {' + NL + '#if PEARL_BD_TAIL_LDG' + NL
      + '      if (last) return pearl_bd_ldg_c(p);' + NL + '#endif' + NL + '      return pearl_bd_ldg(p);');
    expect(FOLD).toContain('constexpr uint32_t MB = 4u, NB = 8u, KSTEPS = rank / 32u;');
    expect(FOLD).toContain('static_assert(rank == PEARL_SB_STRIDE && KSTEPS == 4u,');
    expect(FOLD).toContain('constexpr uint32_t RPL = MB / 2u;');
    expect(FOLD).toContain('constexpr uint32_t FRAG_BLOCK = (k / 32u) * 512u;');
  });

  test('load_half loads the same quads, B\'s first, both through ldg', () => {
    const loads = (branch) => LOAD_HALF[branch].filter((l) => l.trim().startsWith('for ('));
    expect(loads('dflt')).toEqual([LOAD_A, LOAD_B]);
    expect(loads('snake')).toEqual([LOAD_B, LOAD_A]);
    // Each half loads quads hi and hi + 2 of both: all four of the next k-step's over
    // the two calls, into nxt, which this k-step's passes do not read.
    for (const l of [LOAD_A, LOAD_B]) {
      expect(l).toContain('for (uint32_t q = hi; q < 4u; q += 2u) ');
      expect(l).toContain('f[nxt][q] = ldg(');
    }
    expect(WORDS.filter((l) => l.includes('[cur]')).length).toBe(2);
    expect(KSTEP.filter((l) => l.includes('[nxt]')).length).toBe(4);
  });

  test('for random A\' and B\', all lanes and k-steps, the snake issues the default\'s IMMA and words', () => {
    // 6 blocks of A (one warp's four, at 0 or at 2), 10 of B; k 256, eight k-steps.
    const kLog2 = 8;
    const k = 2 ** kLog2;
    const steps = k / 32;
    const mRows = 6 * 16;
    const nRows = 10 * 16;
    const a = bytes(mRows * k, 21);
    const b = bytes(nRows * k, 23);
    const A0 = draw(a, mRows, kLog2, true, false);
    const B0 = draw(b, nRows, kLog2, false, false);
    const A1 = draw(a, mRows, kLog2, true, true);
    const B1 = draw(b, nRows, kLog2, false, true);
    const rand = xorshift(29);
    const SNAKE = plan('snake');
    const DFLT = plan('dflt');
    const bad = [];
    let checked = 0;
    const key = (x) => x.join(' ');
    for (const aBase of [0, 2]) {
      for (const bBase of [0, 4, 6]) {
        for (let lane = 0; lane < 32; lane++) {
          for (const [ro, gate] of [[false, 2], [true, 2], [true, 1]]) {
            const snake = run(SNAKE, A1, B1, kLog2, aBase, bBase, lane, ro, gate, rand);
            const dflt = run(DFLT, A1, B1, kLog2, aBase, bBase, lane, ro, gate, rand);
            for (let s = 0; s < steps; s++) {
              const ref = reference(A0, B0, kLog2, aBase, bBase, s, lane);
              const where = `A ${aBase} B ${bBase} lane ${lane} ro ${ro} gate ${gate} k-step ${s}`;
              // The same 128 (accumulator, A word, B word), each accumulator half
              // once a pass.
              expect(snake[s].length).toBe(128);
              if (key(snake[s].map(key).sort()) !== key(dflt[s].map(key).sort()) && bad.length < 5) {
                bad.push(`${where}: not the default's IMMA`);
              }
              if (new Set(snake[s].map((x) => x[0])).size !== 128 && bad.length < 5) {
                bad.push(`${where}: an accumulator half twice in a pass`);
              }
              // Each with the words the default fold gives that accumulator half.
              for (const [acc, aw, bw] of snake[s]) {
                const want = ref.get(acc);
                if ((!want || want[0] !== aw || want[1] !== bw) && bad.length < 5) {
                  bad.push(`${where} IMMA ${acc}`);
                }
                checked++;
              }
            }
          }
        }
      }
    }
    expect(bad).toEqual([]);
    expect(checked).toBe(2 * 3 * 32 * 3 * steps * 128);
  });

  test('each accumulator half once a pass, k-hi 64 IMMA after k-lo, words from the pass\'s quads', () => {
    for (const branch of ['snake', 'dflt']) {
      const o = order(branch);
      expect(o.length).toBe(128);
      for (let p = 0; p < 128; p++) {
        // A and B from this pass's half: k-lo quads 0 and 2, k-hi quads 1 and 3.
        expect(o[p].pass).toBe(p < 64 ? 0 : 1);
        expect(o[p].hi).toBe(o[p].pass);
        expect(o[p].b).toBe([['b0', 'b2'], ['b1', 'b3']][o[p].pass][o[p].odd]);
        // Rows 0-7 into words 0-1, rows 8-15 into words 2-3, as pearl_mma_m16n8k32.
        expect(o[p].u).toBe(o[p].i / 2);
      }
      for (let p = 0; p < 64; p++) {
        expect(o[p + 64].acc).toBe(o[p].acc);
        expect(o[p + 64].u).toBe(o[p].u);
        expect(o.filter((x) => x.acc === o[p].acc).length).toBe(2);
      }
      expect(new Set(o.slice(0, 64).map((x) => x.acc)).size).toBe(MB * NB * 2);
    }
    // The orders, accumulator half by half, at the start of pair 0.
    expect(order('dflt').slice(0, 10).map((x) => x.acc)).toEqual(
      ['0.0.0', '0.0.1', '1.0.0', '1.0.1', '2.0.0', '2.0.1', '3.0.0', '3.0.1', '0.1.0', '0.1.1']);
    expect(order('snake').slice(0, 10).map((x) => x.acc)).toEqual(
      ['0.0.0', '0.0.1', '0.1.1', '0.1.0', '1.0.0', '1.0.1', '1.1.1', '1.1.0', '2.0.0', '2.0.1']);
  });

  test('every k-lo IMMA of pair j takes the B word gated on the fold that reads its accumulator', () => {
    for (const branch of ['snake', 'dflt']) {
      const o = order(branch);
      for (let p = 0; p < 64; p++) {
        const x = o[p];
        // b0[mb / 2u] or b2[mb / 2u]: copy rl = mb / 2, gated on fold_pair(rxs[rl], rl, j),
        // which reads this IMMA's accumulator.
        expect(x.perRegion).toBe(true);
        expect(foldReads(Math.floor(x.mb / 2), x.j)).toContain(`${x.mb}.${x.n}.${x.i}`);
        // At gate 1 b2 is not gated: an earlier IMMA on the same pair and region is,
        // and the IMMA keep their order (asm volatile).
        if (x.b === 'b2') {
          const before = o.slice(0, p).filter((y) => y.j === x.j && y.b === 'b0'
            && Math.floor(y.mb / 2) === Math.floor(x.mb / 2));
          expect([branch, p, before.length > 0]).toEqual([branch, p, true]);
        }
      }
      for (let p = 64; p < 128; p++) expect(o[p].perRegion).toBe(false);
      // The k-step after a chunk's last (ro): each pair's two folds, then its k-lo
      // IMMA, then the k-hi pass. Each accumulator word is read once, before the
      // k-step's first IMMA on it writes it.
      const read = new Map();
      const written = new Set();
      let late = 0;
      let time = 0;
      for (let j = 0; j < 4; j++) {
        for (let rl = 0; rl < RPL; rl++) {
          for (const w of foldReads(rl, j)) {
            expect(read.has(w)).toBe(false);
            if (written.has(w)) late++;
            read.set(w, time++);
          }
        }
        for (const x of o.slice(0, 64).filter((y) => y.j === j)) {
          written.add(`${x.mb}.${x.n}.${x.i}`);
          written.add(`${x.mb}.${x.n}.${x.i + 1}`);
        }
      }
      expect(late).toBe(0);
      expect(read.size).toBe(MB * NB * 4);
      expect(written.size).toBe(MB * NB * 4);
    }
  });

  test('the snake: in each m-block\'s four IMMA, each shares its A or B word with the one before', () => {
    const o = order('snake');
    let shared = 0;
    for (let p = 0; p < 128; p++) {
      if (p % 4 === 0) {
        // A new m-block (or pair) starts every four.
        if (p > 0) expect(o[p].mb !== o[p - 1].mb || o[p].j !== o[p - 1].j).toBe(true);
        continue;
      }
      const x = o[p];
      const y = o[p - 1];
      expect([x.mb, x.j, x.pass]).toEqual([y.mb, y.j, y.pass]);
      const sameA = x.hi === y.hi && x.u === y.u;
      const sameB = x.b === y.b;
      expect([p, sameA || sameB]).toEqual([p, true]);
      shared++;
    }
    expect(shared).toBe(96);
  });
});
