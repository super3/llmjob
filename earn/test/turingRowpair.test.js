'use strict';

const fs = require('fs');
const path = require('path');
const {
  regionToTile, expandOffset, ROWS_MASK, COLS_MASK, ROWS_PATTERN, COLS_PATTERN,
} = require('../src/shared/miner/pearlhash');
const { foldTranscript } = require('../src/shared/miner/reference');

// Turing's row-pair readout (PEARL_BD_ROWPAIR, off by default). The A' draw
// (pearl_materialize16_afrag) permutes the rows of each whole 64-row warp tile, so lanes
// L and L ^ 4 hold all 256 sums of one region, and the readout gathers a region's word
// with one shuffle instead of three. Nothing here can run the device code, so this copies
// the draw, the fold's loads and IMMA operands, the readout and the hashers' reads into
// JS, pins each copy to the source line for line, and runs them on the CPU. It draws
// random A and B, works out every lane's accumulators from the words its IMMA read (the
// m8n8k16 layout), reads them out the switch's way and the default's, and checks that the
// hashers find the same word for every region and chunk, and that it is the word the JS
// reference gives (regionToTile and foldTranscript). It also ports round 2's
// rowpair_check.py, checks that the draw with k-lo first's block pairing writes every
// byte once, and runs the guards through a small preprocessor. turingRedReadout.test.js
// reads the same code out of the source with its own preprocessor; this is a plain copy.

const SRC = path.join(__dirname, '..', 'native', 'src');
const NL = String.fromCharCode(10);
const BS = String.fromCharCode(92);
// Line endings normalised: a Windows checkout has CRLF.
const read = (name) => fs.readFileSync(path.join(SRC, name), 'utf8')
  .split(String.fromCharCode(13) + NL).join(NL);
const KERNEL = read('pearl_kernel.cu');
const FOLD = read('pearl_fold_bd.cuh');
const HEADER = read('pearl_config.h');

// A file's lines without comments: comment lines dropped, trailing ones cut.
function code(text) {
  return text.split(NL).filter((l) => !l.trim().startsWith('//')).map((l) => {
    const at = l.indexOf('//');
    return at > 0 && l.slice(0, at).trim() ? l.slice(0, at).trimEnd() : l;
  });
}

// From signature to the first closing brace at the start of a line ('' if missing).
function body(src, signature) {
  const at = src.indexOf(signature);
  if (at < 0) return '';
  return src.slice(at, src.indexOf(NL + '}', at) + 2);
}

const FOLD_CODE = code(FOLD);
const AFRAG_SIG = 'extern "C" __global__ void pearl_materialize16_afrag(';
const FRAG_SIG = 'extern "C" __global__ void pearl_materialize16_frag(';
const AFRAG = code(body(KERNEL, AFRAG_SIG));
const FRAG = code(body(KERNEL, FRAG_SIG));

// Pins want to lines: it occurs there exactly once, line for line. On a miss, shows the
// lines from where want[0] first occurs.
function pin(lines, want) {
  let n = 0;
  for (let i = 0; i + want.length <= lines.length; i++) {
    if (want.every((l, j) => lines[i + j] === l)) n++;
  }
  if (n === 0) {
    const first = lines.indexOf(want[0]);
    expect(first >= 0 ? lines.slice(first, first + want.length) : [`no line "${want[0]}"`]).toEqual(want);
  }
  expect([want[0], n]).toEqual([want[0], 1]);
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

// n random bytes in [-64, 63], the operands' int7 range.
function int7(n, seed) {
  const rand = xorshift(seed);
  const a = new Int8Array(n);
  for (let i = 0; i < n; i++) a[i] = (rand() & 127) - 64;
  return a;
}

// The fold's geometry (pinned below): eight 64x64 warp tiles over a 128x256 tile, warp w
// at row slot w >> 2 and column slot w & 3; 4 m-blocks of 16 rows and 8 n-blocks of 8
// columns a warp; 16 regions a warp; 16 chunks of 4 k-steps of 32, k 2048; a transcript
// buffer of 2048 words, two of them.
const MB = 4;
const NB = 8;
const WARPS = 8;
const CHUNKS = 16;
const K_LOG2 = 11;
const TWORDS = 2048;

// ---- The draw (pearl_materialize16_afrag and _frag) ----------------------------------

// PEARL_BD_ROWPAIR's row map, rq: row r of a whole 64-row warp tile, r & 63 = 32 s + 8 j +
// 4 b + 2 u1 + u0, goes to the fold's row 16 j + 8 u0 + 4 s + 2 b + u1 of the same tile. A
// row past the last whole 64 stays where it is.
function rq(r, rows) {
  const o6 = r & 63;
  return (r | 63) < rows
    ? (r & ~63) | (((o6 >> 3) & 3) << 4) | ((o6 & 1) << 3)
      | (((o6 >> 5) & 1) << 2) | (((o6 >> 2) & 1) << 1) | ((o6 >> 1) & 1)
    : r;
}

// The draw's stores, thread by thread: thread v holds the sixteen row-major bytes at v * 16
// (row r, k kk on) and stores its word w at o[w * 4], o = out + blk * 512 + (row & 7) * 64
// bytes + reg words, with row r, or rq with the switch. k-lo first pairs 16-row blocks 2p
// and 2p + 1: block 2p's slot holds both blocks' k 0-15 words and 2p + 1's their k 16-31
// words. A last block without a partner keeps A's word order (k before rows, isA) or B's
// (rows before k). map: a row map in place of rq, for the breakages below. out[o] is the
// byte at o, from[o] its row-major index, written[o] how many threads stored it.
function draw(src, rows, kLog2, { isA = true, rowpair = false, map = null } = {}) {
  const k = 2 ** kLog2;
  const out = new Int8Array(rows * k);
  const from = new Int32Array(rows * k).fill(-1);
  const written = new Uint8Array(rows * k);
  let outside = 0;
  for (let v = 0; v < (rows * k) / 16; v++) {
    const idx = v * 16;
    const r = Math.floor(idx / k);
    const kk = idx & (k - 1);
    const row = map ? map(r, rows) : rowpair ? rq(r, rows) : r;
    const bl = row >> 4;
    const paired = (bl | 1) < (rows >> 4);
    const dbl = paired ? (bl & ~1) | ((kk >> 4) & 1) : bl;
    const blk = dbl * 2 ** (kLog2 - 5) + (kk >> 5);
    const reg = paired ? (bl & 1) * 2 + ((row >> 3) & 1)
      : isA ? ((kk >> 4) & 1) * 2 + ((row >> 3) & 1) : ((row >> 3) & 1) * 2 + ((kk >> 4) & 1);
    const o = blk * 512 + (row & 7) * 64 + reg * 4;
    for (let w = 0; w < 4; w++) {
      for (let j = 0; j < 4; j++) {
        const at = o + w * 16 + j;
        if (at < 0 || at >= out.length) {
          outside++;
          continue;
        }
        out[at] = src[idx + 4 * w + j];
        from[at] = idx + 4 * w + j;
        written[at]++;
      }
    }
  }
  return { out, from, written, outside };
}

// ---- The fold's view --------------------------------------------------------------------

// The rows the fold's IMMA read, from a drawn A' or B'. Slot x (wr for A, wc for B) loads
// lane's sixteen bytes of 16-row blocks base + 4 x + q at k-step s from (base + 4 x + q) *
// FRAG_BLOCK + 512 s + 16 lane (a_off or b_off, then kstep's loads), base rbg * 8 for A and
// col_off + cbg * 16 for B. k-lo first takes m-block mb's word u of half hi from quad (mb /
// 2) * 2 + hi, word (mb & 1) * 2 + u (a_word), and n-block pair j's word odd the same way
// (b_word). The m8n8k16 .row.col operands: lane 4 g + t's A word is bytes 4 t..+3 (in k) of
// row g of the IMMA's eight rows, its B word bytes 4 t..+3 of column g of its eight. So
// operand row g of m-block mb's word u is the fold's row 16 mb + 8 u + g, column g of
// n-block 2 j + odd is the fold's column 16 j + 8 odd + g, and byte i is at k 32 s + 16 hi +
// 4 t + i. view[row * k + kk] is that byte, from[row * k + kk] its place in the draw.
function foldView(frag, kLog2, base, slots) {
  const k = 2 ** kLog2;
  const FRAG_BLOCK = (k / 32) * 512;
  const view = new Int8Array(slots * 64 * k);
  const from = new Int32Array(slots * 64 * k).fill(-1);
  for (let x = 0; x < slots; x++) {
    for (let lane = 0; lane < 32; lane++) {
      const off = (base + 4 * x) * FRAG_BLOCK + lane * 16;
      for (let s = 0; s < k / 32; s++) {
        for (let hi = 0; hi < 2; hi++) {
          for (let a = 0; a < 4; a++) {
            const q = Math.floor(a / 2) * 2 + hi;
            for (let b = 0; b < 2; b++) {
              const w = (a & 1) * 2 + b;
              const row = 64 * x + 16 * a + 8 * b + (lane >> 2);
              for (let i = 0; i < 4; i++) {
                const at = off + q * FRAG_BLOCK + s * 512 + 4 * w + i;
                const kk = 32 * s + 16 * hi + 4 * (lane & 3) + i;
                view[row * k + kk] = frag[at];
                from[row * k + kk] = at;
              }
            }
          }
        }
      }
    }
  }
  return { view, from };
}

// The tile's sums after each chunk: P[c][m * n + col] = the sum over k < 128 (c + 1) of
// rowsA[m] * rowsB[col], wrapping as int32. The accumulators are cumulative across chunks.
function products(rowsA, rowsB, m, n, k) {
  const P = [];
  const run = new Int32Array(m * n);
  for (let c = 0; c < CHUNKS; c++) {
    const k0 = c * 128;
    for (let r = 0; r < m; r++) {
      const a = r * k + k0;
      for (let col = 0; col < n; col++) {
        const b = col * k + k0;
        let s = 0;
        for (let kk = 0; kk < 128; kk++) s += rowsA[a + kk] * rowsB[b + kk];
        run[r * n + col] = (run[r * n + col] + s) | 0;
      }
    }
    P.push(run.slice());
  }
  return P;
}

// Lane 4 g + t's accumulators acc[mb][nb][i] in warp (wr, wc), from the tile's sums Pc:
// pearl_bd_imma(acc[mb][nb][2 u], acc[mb][nb][2 u + 1], a_word(mb, hi, u), b) is one
// m8n8k16, whose lane 4 g + t holds row g, columns 2 t and 2 t + 1 of its 8x8 result. So
// acc[mb][nb][i] is the fold's row 16 mb + 8 (i >> 1) + g, column 8 nb + 2 t + (i & 1).
function accOf(Pc, wr, wc, lane) {
  const g = lane >> 2;
  const t = lane & 3;
  const acc = [];
  for (let mb = 0; mb < MB; mb++) {
    const blk = [];
    for (let nb = 0; nb < NB; nb++) {
      const q = [];
      for (let i = 0; i < 4; i++) {
        q.push(Pc[(64 * wr + 16 * mb + 8 * (i >> 1) + g) * 256 + 64 * wc + 8 * nb + 2 * t + (i & 1)]);
      }
      blk.push(q);
    }
    acc.push(blk);
  }
  return acc;
}

// ---- The readout and the hashers --------------------------------------------------------

// pearl_xor3: lop3 0x96, a ^ b ^ c.
const xor3 = (a, b, c) => (a ^ b ^ c) >>> 0;

// fold_pair, line for line: B pair np of region rl into x.
function foldPair(acc, x, rl, np) {
  const p0 = acc[2 * rl][2 * np];
  const p1 = acc[2 * rl + 1][2 * np];
  const p2 = acc[2 * rl][2 * np + 1];
  const p3 = acc[2 * rl + 1][2 * np + 1];
  const ta = xor3(p0[0], p0[1], p0[2]);
  const tb = xor3(p0[3], p1[0], p1[1]);
  const tc = xor3(p1[2], p1[3], p2[0]);
  const td = xor3(p2[1], p2[2], p2[3]);
  const te = xor3(ta, tb, tc);
  const rest = np === 0 ? (te ^ td) >>> 0 : xor3(te, td, x);
  const t3 = xor3(p3[0], p3[1], p3[2]);
  return xor3(t3, p3[3], rest);
}

// A lane's chain for region rl: fold_pair over the four B pairs, as readout_x folds it, and
// as kstep (rxs[rl] = fold_pair(rxs[rl], rl, j), j 0-3) and tile_head (rx[rl]) do.
function chain(acc, rl) {
  let x = 0;
  for (let np = 0; np < NB / 2; np++) x = foldPair(acc, x, rl, np);
  return x;
}

// The switch's stores: the writers, their region L0 and the held index tK, moved to the
// tile's buffer once a tile.
const rpWriter = (lane) => (lane & 4) === 0;
const rpL0 = (warp, lane) => warp * 16 + (lane & 3) * 4 + (lane >> 3);
const rpTK0 = (L0) => (L0 * 16) | ((L0 >> 1) & 15);
const rpTile = (tK, tbuf) => (tK & (TWORDS - 1)) | tbuf;
// The default's: lanes 0-3 and 16-19, two regions each; and the word c of region L, which
// hash_tile reads.
const dfWriter = (lane) => ((lane >> 2) & 3) === 0;
const dfL = (warp, lane, rl) => warp * 16 + (lane & 3) * 4 + 2 * rl + ((lane >> 4) & 1);
const wordAt = (tbuf, L, c) => tbuf + L * 16 + (c ^ ((L >> 1) & 15));

// The held index of thread (warp, lane) in tile it of the block's walk: set from the
// thread index, then moved to buffer (i & 1) * TWORDS at the top of each tile i.
function heldTK(warp, lane, it, L0 = rpL0) {
  let tK = rpTK0(L0(warp, lane));
  for (let i = 0; i <= it; i++) tK = rpTile(tK, (i & 1) * TWORDS);
  return tK;
}

// One tile's readout into the two buffers, from the tile's sums P (P[c][m * 256 + n], m =
// 64 wr + the fold's row, n = 64 wc + the fold's column). how: 'rowpair' (the switch's:
// both chains xored, one shuffle, a word a writer at tK ^ c) or 'default' (a chain a
// region, three shuffles, xor3 then xor, two words a writer at word c of region L). it:
// the tile's place in the block's walk, which picks its buffer. brk breaks one thing, to
// show the checks would see it: shfl (the shuffle's offset), writer, L0, oneChain.
function readout(P, how, it, brk = {}) {
  const sT = new Uint32Array(2 * TWORDS);
  const writes = new Uint8Array(2 * TWORDS);
  const tbuf = (it & 1) * TWORDS;
  const store = (a, x) => {
    sT[a] = x;
    writes[a]++;
  };
  for (let c = 0; c < CHUNKS; c++) {
    for (let warp = 0; warp < WARPS; warp++) {
      const chains = [];
      for (let lane = 0; lane < 32; lane++) {
        const acc = accOf(P[c], warp >> 2, warp & 3, lane);
        chains.push([chain(acc, 0), chain(acc, 1)]);
      }
      if (how === 'rowpair') {
        const xs = chains.map(([x, y]) => (brk.oneChain ? x : (x ^ y) >>> 0));
        const shfl = brk.shfl === undefined ? 4 : brk.shfl;
        for (let lane = 0; lane < 32; lane++) {
          const x = (xs[lane] ^ xs[lane ^ shfl]) >>> 0;
          if ((brk.writer || rpWriter)(lane)) store(heldTK(warp, lane, it, brk.L0) ^ c, x);
        }
      } else {
        for (let rl = 0; rl < 2; rl++) {
          const xs = chains.map((ch) => ch[rl]);
          for (let lane = 0; lane < 32; lane++) {
            const x = (xor3(xs[lane], xs[lane ^ 4], xs[lane ^ 8]) ^ xs[lane ^ 12]) >>> 0;
            if (dfWriter(lane)) store(wordAt(tbuf, dfL(warp, lane, rl), c), x);
          }
        }
      }
    }
  }
  return { sT, writes, tbuf };
}

// What the hashers read of a tile: on the default fold the first row group hashes on
// threads with sid < 64 and the second on sid >= 64, thread (wr, sid) region L = wr * 64 +
// (sid & 63), word c at tbuf + L * 16 + (c ^ ((L >> 1) & 15)). words[L] are its 16 words.
function hashed(sT, tbuf) {
  const words = new Array(128).fill(null);
  for (let t = 0; t < 256; t++) {
    const wr = t >> 7;
    const sid = t & 127;
    if (!(wr ? sid >= 64 : sid < 64)) continue;
    const L = wr * 64 + (sid & 63);
    expect(words[L]).toBe(null);
    words[L] = Array.from({ length: CHUNKS }, (_, c) => sT[wordAt(tbuf, L, c)]);
  }
  return words;
}

// hash_tile's region_of: the number a hit reports for region L of the tile at row group
// rbg and column group cbg.
function regionOf(L, rbg, cbg, rowsValid) {
  const ow = Math.floor(L / 16);
  const ocb = Math.floor((L % 16) / 4);
  const oreg = L % 4;
  const orb = rbg * 2 + Math.floor(ow / 4);
  const ocgb = cbg * 4 + (ow % 4);
  return (ocgb * 4 + ocb) * rowsValid + orb * 4 + oreg;
}

// ---- The simulation ---------------------------------------------------------------------

// A' of 272 rows (four whole warp tiles, and a 16-row block past them that the switch
// leaves in place and pairs with no other), B' of 512 rows, k 2048. The tile is at row
// group 1 and column group 1 with col_off 0, so its region numbers are not the first
// tile's. Sums: Prp from the switch's draw, Pdf from the default's, Pref from A's and B's
// own rows.
const SIM = { M: 272, N: 512, rbg: 1, cbg: 1 };
// Building it takes about a second; the tests that draw at this size get time to spare on
// a loaded machine.
const SLOW = 60000;
let simCache = null;
function simulation() {
  if (simCache) return simCache;
  const { M, N, rbg, cbg } = SIM;
  const k = 2 ** K_LOG2;
  const A = int7(M * k, 101);
  const B = int7(N * k, 202);
  const Arp = draw(A, M, K_LOG2, { rowpair: true });
  const Adf = draw(A, M, K_LOG2);
  const Bd = draw(B, N, K_LOG2, { isA: false });
  const FArp = foldView(Arp.out, K_LOG2, rbg * 8, 2);
  const FAdf = foldView(Adf.out, K_LOG2, rbg * 8, 2);
  const FB = foldView(Bd.out, K_LOG2, cbg * 16, 4);
  const ownA = A.subarray(128 * rbg * k, (128 * rbg + 128) * k);
  const ownB = B.subarray(256 * cbg * k, (256 * cbg + 256) * k);
  simCache = {
    A, B, Arp, Adf, Bd, FArp, FAdf, FB, ownA, ownB,
    Prp: products(FArp.view, FB.view, 128, 256, k),
    Pdf: products(FAdf.view, FB.view, 128, 256, k),
    Pref: products(ownA, ownB, 128, 256, k),
  };
  return simCache;
}

// The reference's word c of every region L of the simulation's tile: the xor of the sums
// over the rows and columns regionToTile gives for the number region_of reports.
function referenceWords(Pref) {
  const { M, N, rbg, cbg } = SIM;
  const words = [];
  for (let L = 0; L < 128; L++) {
    const tile = regionToTile(regionOf(L, rbg, cbg, M / 16), { m: M, n: N });
    for (const r of tile.rows) expect(r >> 7).toBe(rbg);
    for (const col of tile.cols) expect(col >> 8).toBe(cbg);
    words.push(Array.from({ length: CHUNKS }, (_, c) => {
      let x = 0;
      for (const r of tile.rows) {
        for (const col of tile.cols) x ^= Pref[c][(r - 128 * rbg) * 256 + col - 256 * cbg];
      }
      return x >>> 0;
    }));
  }
  return words;
}

// How many of the hashed words differ from want.
function wrong(words, want) {
  let n = 0;
  for (let L = 0; L < 128; L++) {
    for (let c = 0; c < CHUNKS; c++) if (words[L][c] !== want[L][c]) n++;
  }
  return n;
}

// ---- A small C preprocessor for the guards ----------------------------------------------

// Object-like #define, #undef, #if, #ifdef, #ifndef, #elif, #else, #endif and #error, with
// -D flags (as turingSnake.test.js reads them). Returns the #error lines it reaches.
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
      defs.set(rest.slice(0, end), rest[end] === '(' ? null : rest.slice(end).trim());
    } else if (dir === 'undef') {
      defs.delete(rest);
    } else if (dir === 'error') {
      errors.push(rest);
    }
  }
  expect(stack).toEqual([]);
  return errors;
}

// The guards a build of the fold goes through: pearl_config.h's Turing block (its switches,
// their defaults and #error lines) and the fold's own #error lines.
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

describe('Turing row-pair readout (PEARL_BD_ROWPAIR)', () => {
  test('is off by default, needs the default fold with LEAN\'s parts, and refuses RED, TADDR and the split hash', () => {
    const at = HEADER.indexOf('#ifndef PEARL_BD_ROWPAIR' + NL);
    expect(at).toBeGreaterThanOrEqual(0);
    expect(HEADER.slice(at).split(NL).slice(0, 4)).toEqual(['#ifndef PEARL_BD_ROWPAIR',
      '#define PEARL_BD_ROWPAIR 0', '#endif', '#if PEARL_BD_ROWPAIR != 0 && PEARL_BD_ROWPAIR != 1']);
    expect(HEADER).toContain('#if PEARL_BD_ROWPAIR ' + BS + NL
      + '    && !(PEARL_BD_DEFAULT_FOLD && PEARL_BD_RUNBASE && PEARL_BD_TAIL_LDG && PEARL_BD_GATE == 1)' + NL
      + '#error "PEARL_BD_ROWPAIR is written for the default fold with LEAN\'s parts');
    expect(HEADER).toContain('#if PEARL_BD_ROWPAIR && (PEARL_BD_RED || PEARL_BD_TADDR || PEARL_BD_HASH_SPLIT)' + NL
      + '#error "PEARL_BD_ROWPAIR is not built with PEARL_BD_RED, _TADDR or the split hash');
    expect(HEADER).toContain('#define PEARL_BD_DEFAULT_FOLD ' + BS + NL
      + '  (PEARL_BD_A_DIRECT && PEARL_BD_KORDER && PEARL_BD_PERSIST && PEARL_BD_GROUPS == 2 ' + BS + NL
      + '   && PEARL_BD_T_BUFS == 2)');
    // The draw is A-direct's, built in the pass that builds the fold, from the same flags.
    expect(KERNEL).toContain('#if PEARL_TURING_BDIRECT && PEARL_BD_A_DIRECT' + NL + AFRAG_SIG);

    const GUARDS = guards();
    const errorsFor = (flags) => preprocess(GUARDS, flags);
    const value = (cond, flags) => (preprocess(GUARDS + NL + '#if ' + cond + NL + '#error yes' + NL
      + '#endif', flags).includes('yes') ? 1 : 0);
    // With no flags the switch is off, on the default fold with LEAN's parts.
    expect(value('PEARL_BD_ROWPAIR == 0', {})).toBe(1);
    expect(value('PEARL_BD_DEFAULT_FOLD && PEARL_BD_KORDER && PEARL_BD_RUNBASE && PEARL_BD_TAIL_LDG '
      + '&& PEARL_BD_GATE == 1 && !PEARL_BD_RED && !PEARL_BD_TADDR && !PEARL_BD_HASH_SPLIT', {})).toBe(1);
    const R = { PEARL_BD_ROWPAIR: 1 };
    const OFF = { PEARL_BD_LEAN: 0, PEARL_BD_HIT_COORDS: 0 };
    expect(value('PEARL_BD_ROWPAIR == 1 && PEARL_BD_LEAN && PEARL_BD_HIT_COORDS', R)).toBe(1);
    // The builds the switch is written for: LEAN's parts, with or without HIT_COORDS,
    // HASH_RG, SNAKE and LAST_RO, or the parts asked for one by one; the ablations.
    const allowed = [
      R,
      { ...R, PEARL_BD_HIT_COORDS: 0 },
      { ...R, PEARL_BD_HASH_RG: 1 },
      { ...R, PEARL_BD_SNAKE: 1 },
      { ...R, PEARL_BD_SNAKE: 1, PEARL_BD_HASH_RG: 1 },
      { ...R, PEARL_BD_LAST_RO: 1 },
      { ...R, PEARL_BD_LAST_RO: 1, PEARL_BD_HASH_RG: 1 },
      { ...R, PEARL_BD_LAST_RO: 1, PEARL_BD_SNAKE: 1 },
      { ...R, PEARL_BD_LAST_RO: 1, PEARL_BD_SNAKE: 1, PEARL_BD_HASH_RG: 1 },
      { ...R, PEARL_BD_LEAN: 1, PEARL_BD_HIT_COORDS: 1 },
      { ...R, ...OFF, PEARL_BD_RUNBASE: 1, PEARL_BD_TAIL_LDG: 1, PEARL_BD_GATE: 1 },
      { ...R, PEARL_ABLATE_READOUT: 1 },
      { ...R, PEARL_ABLATE_TRANSCRIPT: 1 },
    ];
    for (const flags of allowed) expect([flags, errorsFor(flags)]).toEqual([flags, []]);
    // Refused by the guard named, each a build that is fine without the switch.
    const LEAN_PARTS = 'PEARL_BD_ROWPAIR is written for the default fold with LEAN\'s parts';
    const OWN_STORES = 'PEARL_BD_ROWPAIR is not built with PEARL_BD_RED, _TADDR or the split hash';
    // The split hash as the races build it (split4).
    const SPLIT = { PEARL_BD_HASH_SPLIT: 4, PEARL_BD_RUNBASE: 1, PEARL_BD_TAIL_LDG: 1 };
    const refused = [
      [{ PEARL_BD_ROWPAIR: 2 }, 'PEARL_BD_ROWPAIR is 0 or 1', {}],
      [{ ...R, ...OFF }, LEAN_PARTS, OFF],
      [{ ...R, PEARL_BD_LEAN: 0 }, LEAN_PARTS, { PEARL_BD_LEAN: 0 }],
      [{ ...R, PEARL_BD_RUNBASE: 0 }, LEAN_PARTS, { PEARL_BD_RUNBASE: 0 }],
      [{ ...R, PEARL_BD_TAIL_LDG: 0 }, LEAN_PARTS, { PEARL_BD_TAIL_LDG: 0 }],
      [{ ...R, PEARL_BD_GATE: 2 }, LEAN_PARTS, { PEARL_BD_GATE: 2 }],
      [{ ...R, PEARL_BD_PERSIST: 0 }, LEAN_PARTS, { PEARL_BD_PERSIST: 0 }],
      [{ ...R, PEARL_BD_A_REALIGN: 1 }, LEAN_PARTS, { PEARL_BD_A_REALIGN: 1 }],
      [{ ...R, PEARL_BD_KORDER: 0 }, LEAN_PARTS, { PEARL_BD_KORDER: 0 }],
      [{ ...R, PEARL_BD_A_DIRECT: 0 }, LEAN_PARTS, { PEARL_BD_A_DIRECT: 0 }],
      [{ ...R, PEARL_BD_GROUPS: 1 }, LEAN_PARTS, { PEARL_BD_GROUPS: 1 }],
      [{ ...R, PEARL_BD_RED: 1 }, OWN_STORES, { PEARL_BD_RED: 1 }],
      [{ ...R, PEARL_BD_RED: 1, PEARL_BD_GATE: 1 }, OWN_STORES, { PEARL_BD_RED: 1, PEARL_BD_GATE: 1 }],
      [{ ...R, PEARL_BD_TADDR: 1 }, OWN_STORES, { PEARL_BD_TADDR: 1 }],
      [{ ...R, ...SPLIT }, OWN_STORES, SPLIT],
    ];
    for (const [flags, msg, without] of refused) {
      const errors = errorsFor(flags);
      expect([flags, errors.some((e) => e.includes(msg))]).toEqual([flags, true]);
      if (without) {
        expect([without, errorsFor(without)]).toEqual([without, []]);
        expect([flags, errors.every((e) => e.includes('PEARL_BD_ROWPAIR'))]).toEqual([flags, true]);
      }
    }
  });

  test('the copies match the source, line for line', () => {
    expect(AFRAG.length).toBeGreaterThan(10);
    expect(FRAG.length).toBeGreaterThan(10);
    // The fold's constants, and the chunks' k from the header.
    for (const line of [
      '  constexpr uint32_t k = PEARL_FOLD_K, rank = PEARL_FOLD_RANK, chunks = PEARL_FOLD_CHUNKS;',
      '  constexpr uint32_t MB = 4u, NB = 8u, KSTEPS = rank / 32u;',
      '  constexpr uint32_t FRAG_BLOCK = (k / 32u) * 512u;',
      '  static_assert(chunks == PEARL_JACKPOT_BUCKETS, "one transcript word a region a chunk");',
      '  static_assert(PEARL_ROWS_MASK == 0x1Bu && PEARL_COLS_MASK == 0x39u,',
      '  const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;',
      '  const uint32_t wr = warp >> 2, wc = warp & 3u;',
      '  constexpr uint32_t regions_per_warp = 4u, warp_regions = 16u;',
      '  constexpr uint32_t TWORDS = (PEARL_FOLD_THREADS / 32u) * warp_regions * PEARL_JACKPOT_BUCKETS;',
      '  int32_t acc[MB][NB][4];',
      '  constexpr uint32_t RPL = MB / 2u;',
    ]) pin(FOLD_CODE, [line]);
    pin(FOLD_CODE, ['#if PEARL_BD_GROUPS == 2', '  const uint32_t sid = threadIdx.x & 127u;', '#endif', '#else']);
    for (const def of ['#define PEARL_FOLD_RANK 128u', '#define PEARL_FOLD_K 2048u',
      '#define PEARL_FOLD_CHUNKS (PEARL_FOLD_K / PEARL_FOLD_RANK)', '#define PEARL_JACKPOT_BUCKETS 16',
      '#define PEARL_FOLD_TURING_THREADS 256u']) {
      expect(HEADER.split(NL)).toContain(def);
    }

    // The draw: a thread's row and k, the row its values come from (r, with or without the
    // switch: only their place moves), and its four words at o[w * 4].
    const THREAD = [
      '  const uint64_t v = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;',
      '  if (v >= (((uint64_t)rows << k_log2) >> 4)) return;',
      '  const uint64_t idx = v << 4;',
      '  const uint32_t r = (uint32_t)(idx >> k_log2);',
      '  const uint32_t kk = (uint32_t)(idx & ((1ull << k_log2) - 1u));',
      '  const int8_t *__restrict__ row = dense + (size_t)r * rank;',
      '  const int4 b4 = v < read_vecs ? reinterpret_cast<const int4 *>(base)[v]',
    ];
    const STORE = [
      '#pragma unroll',
      '  for (int w = 0; w < 4; w++) {',
      '    const uint4 q0 = pp[w * 2];',
      '    const uint4 q1 = pp[w * 2 + 1];',
      '    const uint32_t p0[4] = {q0.x, q0.z, q1.x, q1.z};',
      '    const uint32_t p1[4] = {q0.y, q0.w, q1.y, q1.w};',
      '    uint32_t packed = 0;',
      '#pragma unroll',
      '    for (int j = 0; j < 4; j++) {',
      '      const int32_t a = (int32_t)(int8_t)(bw[w] >> (j * 8));',
      '      int32_t sv = a + (int32_t)row[p0[j]] - (int32_t)row[p1[j]];',
      '      sv = sv < -128 ? -128 : (sv > 127 ? 127 : sv);',
      '      packed |= ((uint32_t)sv & 0xffu) << (j * 8);',
      '    }',
      '    o[w * 4] = packed;',
      '  }',
    ];
    const PAIRED = [
      '  const bool paired = (bl | 1u) < (rows >> 4);',
      '  const uint32_t dbl = paired ? (bl & ~1u) | ((kk >> 4) & 1u) : bl;',
      '  const uint64_t blk = ((uint64_t)dbl << (k_log2 - 5)) + (kk >> 5);',
    ];
    for (const lines of [AFRAG, FRAG]) {
      pin(lines, THREAD);
      pin(lines, STORE);
    }
    pin(AFRAG, [
      '#if PEARL_BD_ROWPAIR',
      '  const uint32_t o6 = r & 63u;',
      '  const uint32_t rq = (r | 63u) < rows',
      '                          ? (r & ~63u) | (((o6 >> 3) & 3u) << 4) | ((o6 & 1u) << 3)',
      '                                | (((o6 >> 5) & 1u) << 2) | (((o6 >> 2) & 1u) << 1) | ((o6 >> 1) & 1u)',
      '                          : r;',
      '  const uint32_t bl = rq >> 4;',
      ...PAIRED,
      '  const uint32_t reg = paired ? (bl & 1u) * 2u + ((rq >> 3) & 1u)',
      '                              : ((kk >> 4) & 1u) * 2u + ((rq >> 3) & 1u);',
      '  uint32_t *o = reinterpret_cast<uint32_t *>(out + blk * 512u + (rq & 7u) * 64u) + reg;',
      '#else',
      '#if PEARL_BD_KORDER',
      '  const uint32_t bl = r >> 4;',
      ...PAIRED,
      '  const uint32_t reg = paired ? (bl & 1u) * 2u + ((r >> 3) & 1u)',
      '                              : ((kk >> 4) & 1u) * 2u + ((r >> 3) & 1u);',
      '#else',
      '  const uint64_t blk = ((uint64_t)(r >> 4) << (k_log2 - 5)) + (kk >> 5);',
      '  const uint32_t reg = ((kk >> 4) & 1u) * 2u + ((r >> 3) & 1u);',
      '#endif',
      '  uint32_t *o = reinterpret_cast<uint32_t *>(out + blk * 512u + (r & 7u) * 64u) + reg;',
      '#endif',
      ...STORE,
    ]);
    // B's draw, which the switch leaves alone.
    expect(FRAG.join(NL)).not.toContain('ROWPAIR');
    pin(FRAG, [
      '#if PEARL_BD_KORDER',
      '  const uint32_t bl = r >> 4;',
      ...PAIRED,
      '  const uint32_t reg = paired ? (bl & 1u) * 2u + ((r >> 3) & 1u)',
      '                              : ((r >> 3) & 1u) * 2u + ((kk >> 4) & 1u);',
      '#else',
      '  const uint64_t blk = ((uint64_t)(r >> 4) << (k_log2 - 5)) + (kk >> 5);',
      '  const uint32_t reg = ((r >> 3) & 1u) * 2u + ((kk >> 4) & 1u);',
      '#endif',
      '  uint32_t *o = reinterpret_cast<uint32_t *>(out + blk * 512u + (r & 7u) * 64u) + reg;',
      ...STORE,
    ]);

    // The fold's loads: a warp's four blocks of a k-step, quad q at + q * FRAG_BLOCK, and
    // the words k-lo first takes from them.
    pin(FOLD_CODE, [
      '  auto a_off = [&](uint32_t rbg_) -> uint32_t {',
      '    return (rbg_ * 8u + wr * 4u) * FRAG_BLOCK + lane * 16u;',
      '  };',
      '  auto b_off = [&](uint32_t cbg_) -> uint32_t {',
      '#ifdef PEARL_ABLATE_BSHARE',
      '    cbg_ = wr ? cbg_ ^ 1u : cbg_;',
      '#endif',
      '    return (col_off + cbg_ * 16u + wc * 4u) * FRAG_BLOCK + lane * 16u;',
      '  };',
    ]);
    pin(FOLD_CODE, [
      '#else',
      '#pragma unroll',
      '      for (uint32_t q = hi; q < 4u; q += 2u) af[nxt][q] = ldg(Aprime + ao + q * FRAG_BLOCK);',
      '#ifndef PEARL_ABLATE_BLOAD',
      '#pragma unroll',
      '      for (uint32_t q = hi; q < 4u; q += 2u) bf[nxt][q] = ldg(Bprime + bo + q * FRAG_BLOCK);',
    ]);
    pin(FOLD_CODE, [
      '    auto a_word = [&](uint32_t mb, uint32_t hi, uint32_t u) -> uint32_t {',
      '      const int4 v = af[cur][(mb / 2u) * 2u + hi];',
      '      return (uint32_t)(mb & 1u ? (u ? v.w : v.z) : (u ? v.y : v.x));',
      '    };',
      '    auto b_word = [&](uint32_t j, uint32_t hi, uint32_t odd) -> uint32_t {',
      '      const int4 v = bf[cur][(j / 2u) * 2u + hi];',
      '      return (uint32_t)(j & 1u ? (odd ? v.w : v.z) : (odd ? v.y : v.x));',
      '    };',
    ]);
    // Each IMMA's accumulators and words: acc[mb][2 j + odd][2 u] and [2 u + 1] take
    // a_word(mb, hi, u) times b_word(j, hi, odd), k-lo then k-hi. (SNAKE issues the same
    // IMMA in another order, which the sums do not see.) The k-lo B words go through
    // pearl_bd_after, which returns its first operand unchanged (prmt 0x3210).
    const MB_LOOP = '      for (uint32_t mb = 0; mb < MB; mb++) {';
    pin(FOLD_CODE, [
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
    ]);
    pin(FOLD_CODE, [
      '#else',
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
      '#endif',
    ]);
    pin(FOLD_CODE, ['      const uint32_t b1 = b_word(j, 1u, 0u), b3 = b_word(j, 1u, 1u);']);
    pin(FOLD_CODE, [
      '#else',
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
      '#endif',
    ]);
    pin(FOLD_CODE, [
      '__device__ __forceinline__ void pearl_bd_imma(int32_t &c0, int32_t &c1, uint32_t a, uint32_t b) {',
      '  asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"',
      '               : "+r"(c0), "+r"(c1)',
      '               : "r"(a), "r"(b));',
      '}',
    ]);
    pin(FOLD_CODE, [
      '__device__ __forceinline__ uint32_t pearl_bd_after(uint32_t b, uint32_t x) {',
      '  uint32_t r;',
      '  asm("prmt.b32 %0, %1, %2, 0x3210;" : "=r"(r) : "r"(b), "r"(x));',
      '  return r;',
      '}',
    ]);
    pin(code(KERNEL), [
      '__device__ __forceinline__ uint32_t pearl_xor3(uint32_t a, uint32_t b, uint32_t c) {',
      '  uint32_t d;',
      '  asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(d) : "r"(a), "r"(b), "r"(c));',
      '  return d;',
      '}',
    ]);

    // The readout: fold_pair, both readout_x, both readout_gather, the held index, the
    // stores and readout.
    pin(FOLD_CODE, [
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
      '#if !PEARL_BD_LAST_RO',
      '#if PEARL_BD_ROWPAIR',
      '  auto readout_x = [&]() -> uint32_t {',
      '    uint32_t x = 0u, y = 0u;',
      '#pragma unroll',
      '    for (uint32_t np = 0; np < NB / 2u; np++) {',
      '      x = fold_pair(x, 0u, np);',
      '      y = fold_pair(y, 1u, np);',
      '    }',
      '    x ^= y;',
      '#ifndef PEARL_ABLATE_TRANSCRIPT',
      '    x ^= __shfl_xor_sync(0xffffffffu, x, 4);',
      '#endif',
      '    return x;',
      '  };',
      '#else',
      '  auto readout_x = [&](uint32_t rl) -> uint32_t {',
      '    uint32_t x = 0u;',
      '#pragma unroll',
      '    for (uint32_t np = 0; np < NB / 2u; np++) x = fold_pair(x, rl, np);',
      '#if PEARL_BD_RED',
      '#elif !defined(PEARL_ABLATE_TRANSCRIPT)',
      '    const uint32_t s4 = __shfl_xor_sync(0xffffffffu, x, 4);',
      '    const uint32_t s8 = __shfl_xor_sync(0xffffffffu, x, 8);',
      '    const uint32_t s12 = __shfl_xor_sync(0xffffffffu, x, 12);',
      '    x = pearl_xor3(x, s4, s8) ^ s12;',
      '#endif',
      '    return x;',
      '  };',
      '#endif',
      '#endif',
      '#if PEARL_BD_T_BUFS == 2',
      '  uint32_t tbuf = 0u;',
    ]);
    pin(FOLD_CODE, [
      '#elif PEARL_BD_ROWPAIR',
      '  uint32_t tK;',
      '  {',
      '    const uint32_t L0 = warp * warp_regions + (lane & 3u) * regions_per_warp + (lane >> 3);',
      '    tK = L0 * 16u | ((L0 >> 1) & 15u);',
      '  }',
      '#endif',
    ]);
    pin(FOLD_CODE, [
      '#elif PEARL_BD_ROWPAIR && PEARL_BD_LAST_RO',
      '  auto readout_st = [&](uint32_t c, uint32_t x) {',
      '    if ((lane & 4u) == 0u) sT[tK ^ c] = x;',
      '  };',
      '#elif PEARL_BD_ROWPAIR',
      '  const uint32_t sT0 = (uint32_t)__cvta_generic_to_shared(sT);',
      '  auto readout_st = [&](uint32_t c, uint32_t x) {',
      '    asm volatile("{ .reg .pred p; setp.eq.u32 p, %2, 0; @p st.shared.u32 [%0], %1; }"',
      '                 ::"r"(sT0 + 4u * (tK ^ c)), "r"(x), "r"(lane & 4u)',
      '                 : "memory");',
      '  };',
    ]);
    pin(FOLD_CODE, [
      '#else',
      '  auto readout_st = [&](uint32_t rl, uint32_t c, uint32_t x) {',
      '    if (((lane >> 2) & 3u) == 0u) {',
      '      const uint32_t L = warp * warp_regions + (lane & 3u) * regions_per_warp + 2u * rl',
      '                         + ((lane >> 4) & 1u);',
      '      sT[tbuf + L * 16u + (c ^ ((L >> 1) & 15u))] = x;',
      '    }',
      '  };',
      '#endif',
      '#if PEARL_BD_LAST_RO',
      '#elif PEARL_BD_ROWPAIR',
      '  auto readout = [&](uint32_t c) { readout_st(c, readout_x()); };',
      '#else',
      '  auto readout = [&](uint32_t rl, uint32_t c) { readout_st(rl, c, readout_x(rl)); };',
      '#endif',
    ]);
    pin(FOLD_CODE, [
      '#if PEARL_BD_ROWPAIR',
      '  auto readout_gather = [&](uint32_t x) -> uint32_t {',
      '#ifndef PEARL_ABLATE_TRANSCRIPT',
      '    x ^= __shfl_xor_sync(0xffffffffu, x, 4);',
      '#endif',
      '    return x;',
      '  };',
      '#else',
      '  auto readout_gather = [&](uint32_t x) -> uint32_t {',
      '#if !defined(PEARL_ABLATE_TRANSCRIPT) && !PEARL_BD_RED',
      '    const uint32_t s4 = __shfl_xor_sync(0xffffffffu, x, 4);',
      '    const uint32_t s8 = __shfl_xor_sync(0xffffffffu, x, 8);',
      '    const uint32_t s12 = __shfl_xor_sync(0xffffffffu, x, 12);',
      '    x = pearl_xor3(x, s4, s8) ^ s12;',
      '#endif',
      '    return x;',
      '  };',
      '#endif',
    ]);
    // Its three sites, each the two chains' xor gathered once: the chunk loop (chunks
    // 0-14, the chains folded in the next chunk's k-step 0), the tile's tail (chunk 15),
    // and with LAST_RO tile_head (chunk 15, in the next tile's k-step 0).
    pin(FOLD_CODE, [
      '      kstep(chunk + 1u, 0u, RO, false, 1u);',
      '#ifndef PEARL_ABLATE_READOUT',
      '#if PEARL_BD_ROWPAIR',
      '      readout_st(chunk, readout_gather(rxs[0] ^ rxs[1]));',
      '#else',
    ]);
    pin(FOLD_CODE, [
      '    kstep(chunks - 1u, 3u, false, true);',
      '#if !PEARL_BD_LAST_RO',
      '#if PEARL_BD_ROWPAIR',
      '    readout(chunks - 1u);',
      '#else',
    ]);
    pin(FOLD_CODE, [
      '  auto tile_head = [&](bool prev) {',
      '    uint32_t rx[RPL] = {0u, 0u};',
      '    if (prev) {',
      '#pragma unroll',
      '      for (uint32_t j = 0; j < 4u; j++) {',
      '#pragma unroll',
      '        for (uint32_t rl = 0; rl < RPL; rl++) rx[rl] = fold_pair(rx[rl], rl, j);',
    ]);
    pin(FOLD_CODE, [
      '    kstep(0u, 0u, false, false);',
      '    if (prev) {',
      '#if PEARL_BD_ROWPAIR',
      '      readout_st(chunks - 1u, readout_gather(rx[0] ^ rx[1]));',
      '#else',
    ]);
    // The held index moves to the tile's buffer at the top of each tile.
    pin(FOLD_CODE, [
      '  for (uint32_t v = blockIdx.x; v < tiles; it++) {',
      '#if PEARL_BD_T_BUFS == 2',
      '    tbuf = (it & 1u) * TWORDS;',
      '#if PEARL_BD_TADDR || PEARL_BD_ROWPAIR',
      '    tK = (tK & (TWORDS - 1u)) | tbuf;',
      '    asm volatile("" : "+r"(tK));',
      '#endif',
      '#endif',
    ]);
    // The hashers, their reads and a hit's region number.
    // The default's, the else branch of pearl_fold_bd<kHashRg>'s role (see
    // turingHashRg.test.js for the other).
    pin(FOLD_CODE, [
      '    } else {',
      '      hasher = wr ? sid >= 64u : sid < 64u;',
      '      L = wr * 64u + (sid & 63u);',
      '    }',
      '#elif PEARL_BD_GROUPS == 2',
    ]);
    pin(FOLD_CODE, [
      '    if (hasher) {',
      '      uint32_t tm[16];',
      '#pragma unroll',
      '      for (uint32_t c = 0; c < 16u; c++) tm[c] = sT[tbuf + L * 16u + (c ^ ((L >> 1) & 15u))];',
    ]);
    pin(FOLD_CODE, [
      '      auto region_of = [&](uint32_t rbg_, uint32_t cbg_) -> uint32_t {',
      '        const uint32_t ow = L / warp_regions;',
      '        const uint32_t ocb = (L % warp_regions) / regions_per_warp;',
      '        const uint32_t oreg = L % regions_per_warp;',
      '        const uint32_t orb = rbg_ * 2u + ow / 4u;',
      '        const uint32_t ocgb = cbg_ * 4u + ow % 4u;',
      '        return (ocgb * 4u + ocb) * rows_valid + orb * regions_per_warp + oreg;',
      '      };',
    ]);
  });

  test('the row map is rowpair_check.py\'s: a bijection on each whole 64-row warp tile', () => {
    // rowpair_check.py's mrow, as written there.
    const mrow = (o) => (((o >> 3) & 3) << 4) | ((o & 1) << 3) | (((o >> 5) & 1) << 2)
      | (((o >> 2) & 1) << 1) | ((o >> 1) & 1);
    for (let o = 0; o < 64; o++) {
      // pearl_config.h's words: row 32 s + 8 j + 4 b + 2 u1 + u0 at 16 j + 8 u0 + 4 s + 2 b + u1.
      const s = o >> 5;
      const j = (o >> 3) & 3;
      const b = (o >> 2) & 1;
      const u1 = (o >> 1) & 1;
      const u0 = o & 1;
      expect([o, mrow(o)]).toEqual([o, 16 * j + 8 * u0 + 4 * s + 2 * b + u1]);
      for (const G of [0, 1, 5]) expect(rq(64 * G + o, 64 * G + 64)).toBe(64 * G + mrow(o));
    }
    expect(new Set(Array.from({ length: 64 }, (_, o) => mrow(o))).size).toBe(64);
    // Rows 0 and 63 stay; past the last whole 64 every row stays.
    expect([rq(0, 64), rq(63, 64)]).toEqual([0, 63]);
    for (let r = 256; r < 272; r++) expect(rq(r, 272)).toBe(r);
    for (let r = 64; r < 112; r++) expect(rq(r, 112)).toBe(r);
  });

  test('the draw writes every byte of A\' once, k-lo first\'s pairs of the permuted rows', () => {
    for (const rows of [16, 32, 48, 64, 80, 112, 128, 144, 192, 208, 256, 272, 320, 336, 512]) {
      for (const kLog2 of [5, 6, 8]) {
        const k = 2 ** kLog2;
        const src = int7(rows * k, rows * 31 + kLog2);
        const on = draw(src, rows, kLog2, { rowpair: true });
        const off = draw(src, rows, kLog2);
        for (const d of [on, off]) {
          expect([rows, kLog2, d.outside, d.written.every((n) => n === 1)]).toEqual([rows, kLog2, 0, true]);
        }
        // Only the rows' places move: the switch's draw is the default draw of A with row r
        // at rq(r), byte for byte. So it is k-lo first's pairing of the permuted rows.
        const moved = new Int8Array(rows * k);
        for (let r = 0; r < rows; r++) moved.set(src.subarray(r * k, (r + 1) * k), rq(r, rows) * k);
        const want = draw(moved, rows, kLog2).out;
        expect([rows, kLog2, Buffer.compare(Buffer.from(on.out.buffer), Buffer.from(want.buffer))])
          .toEqual([rows, kLog2, 0]);
        // A whole warp tile's bytes stay in its 64 k bytes of A'; past the last whole 64
        // they are where the default puts them.
        const whole = Math.floor(rows / 64) * 64;
        let strays = 0;
        let tailMoved = 0;
        for (let at = 0; at < rows * k; at++) {
          const r = Math.floor(on.from[at] / k);
          if (r < whole && Math.floor(at / (64 * k)) !== Math.floor(r / 64)) strays++;
          if (at >= whole * k && (on.from[at] !== off.from[at] || on.out[at] !== off.out[at])) tailMoved++;
        }
        expect([rows, kLog2, strays, tailMoved]).toEqual([rows, kLog2, 0, 0]);
      }
    }
  }, SLOW);

  test('each row the fold\'s IMMA read is one permuted row of A, at its own k; B\'s columns stay', () => {
    const { A, B, Arp, Adf, Bd, FArp, FAdf, FB } = simulation();
    const { M, N, rbg, cbg } = SIM;
    const k = 2 ** K_LOG2;
    expect([Arp.outside, Adf.outside, Bd.outside]).toEqual([0, 0, 0]);
    for (const d of [Arp, Adf]) expect(d.written.every((n) => n === 1)).toBe(true);
    expect(Bd.written.every((n) => n === 1)).toBe(true);
    // Fold row 64 x + m of the tile reads original row 128 rbg + 64 x + mrow^-1(m) with the
    // switch, 128 rbg + 64 x + m without it; column n reads B's row 256 cbg + n. By index,
    // through the draws' from maps, and by value.
    const inv = new Array(64);
    for (let o = 0; o < 64; o++) inv[rq(o, 64)] = o;
    const check = (view, drawn, rowOf, rows, srcRows) => {
      let bad = 0;
      for (let m = 0; m < rows; m++) {
        const want = rowOf(m);
        for (let kk = 0; kk < k; kk++) {
          const idx = drawn.from[view.from[m * k + kk]];
          if (idx !== want * k + kk || view.view[m * k + kk] !== srcRows[want * k + kk]) bad++;
        }
      }
      return bad;
    };
    expect(check(FArp, Arp, (m) => 128 * rbg + (m & ~63) + inv[m & 63], 128, A)).toBe(0);
    expect(check(FAdf, Adf, (m) => 128 * rbg + m, 128, A)).toBe(0);
    expect(check(FB, Bd, (n) => 256 * cbg + n, 256, B)).toBe(0);
    // With the switch only rows 0 and 63 of a warp tile stay where they were: the map moves
    // every bit of the row, so a row stays only if all six are equal.
    let same = 0;
    for (let m = 0; m < 128; m++) if (inv[m & 63] === (m & 63)) same++;
    expect(same).toBe(2 * 2);
    // The rows past the tile's row groups (256-271) are drawn but not read.
    expect([M, N, 128 * rbg + 128 <= M - 16]).toEqual([272, 512, true]);
  }, SLOW);

  test('lanes L and L ^ 4 hold all of region warp * 16 + (L & 3) * 4 + L / 8 (rowpair_check.py)', () => {
    // Round 2's check, on the rows the fold reads from the switch's draw of random A: the
    // 16 rows and 16 columns of the sums lanes L and L ^ 4 hold (fold rows 16 mb + g + h,
    // h 0 or 8, of all four m-blocks; columns 8 nb + 2 t + cc) are the region's: its rows
    // roff | ROWS[i] and columns coff | COLS[i], with roff and coff the offsets hash_tile's
    // region L % 4 and (L % 16) / 4 name in the warp tile.
    expect([ROWS_MASK, COLS_MASK]).toEqual([0x1B, 0x39]);
    const ROWS = [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27];
    const COLS = [0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57];
    expect([ROWS, COLS]).toEqual([ROWS_PATTERN, COLS_PATTERN]);
    // rowpair_check.py's expand: deposit i's bits into the bits mask leaves free.
    const expand = (i, mask) => {
      let off = 0;
      let b = 0;
      let rest = i;
      while (rest) {
        while ((mask >> b) & 1) b++;
        if (rest & 1) off |= 1 << b;
        rest >>= 1;
        b++;
      }
      return off;
    };
    for (let i = 0; i < 64; i++) {
      expect([i, expand(i, ROWS_MASK), expand(i, COLS_MASK)])
        .toEqual([i, expandOffset(i, ROWS_MASK), expandOffset(i, COLS_MASK)]);
    }
    // Which original row of its warp tile each fold row holds, found by value.
    const { A, FArp, FAdf } = simulation();
    const k = 2 ** K_LOG2;
    const key = (a, r) => Buffer.from(a.buffer, a.byteOffset + r * k, k).toString('base64');
    const original = new Map();
    for (let r = 0; r < SIM.M; r++) original.set(key(A, r), r);
    const heldRows = (view) => Array.from({ length: 128 }, (_, m) => {
      const r = original.get(key(view, m));
      expect(r).toBeDefined();
      expect(r >> 6).toBe(2 * SIM.rbg + (m >> 6));
      return r & 63;
    });
    const sorted = (s) => [...s].sort((a, b) => a - b);
    const cell = (r, c) => r * 64 + c;
    // What a set of lanes holds of m-blocks mbs, in warp tile x.
    const held = (rowAt, x, lanes, mbs) => {
      const s = new Set();
      for (const L of lanes) {
        for (const mb of mbs) {
          for (const h of [0, 8]) {
            for (let nb = 0; nb < NB; nb++) {
              for (const cc of [0, 1]) s.add(cell(rowAt[64 * x + 16 * mb + (L >> 2) + h], 8 * nb + 2 * (L & 3) + cc));
            }
          }
        }
      }
      return sorted(s);
    };
    const region = (oreg, ocb) => {
      const roff = expand(oreg, ROWS_MASK);
      const coff = expand(ocb, COLS_MASK);
      const s = [];
      for (const r of ROWS) for (const c of COLS) s.push(cell(roff | r, coff | c));
      return sorted(s);
    };
    const onRows = heldRows(FArp.view);
    const offRows = heldRows(FAdf.view);
    for (let x = 0; x < 2; x++) {
      const covered = new Set();
      for (let lane = 0; lane < 32; lane++) {
        // The switch: L and L ^ 4, all four m-blocks, region q = L / 8 in rows and t in columns.
        const mine = held(onRows, x, [lane, lane ^ 4], [0, 1, 2, 3]);
        expect([x, lane, mine]).toEqual([x, lane, region(lane >> 3, lane & 3)]);
        // Its writer's L0 numbers that region: oreg L0 % 4, ocb (L0 % 16) / 4.
        const L0 = rpL0(4 * x, lane);
        expect([L0 % 4, (L0 % 16) >> 2]).toEqual([lane >> 3, lane & 3]);
        if (rpWriter(lane)) for (const c of mine) covered.add(c);
        // The default, on its own draw (rowpair_check.py's last comment): L, L ^ 4, L ^ 8 and
        // L ^ 12 hold region (2 rl + bit 4 of L, t) in m-blocks 2 rl and 2 rl + 1.
        for (let rl = 0; rl < 2; rl++) {
          const lanes = [lane, lane ^ 4, lane ^ 8, lane ^ 12];
          expect(held(offRows, x, lanes, [2 * rl, 2 * rl + 1])).toEqual(region(2 * rl + ((lane >> 4) & 1), lane & 3));
          // On the switch's draw they hold no region, so neither readout works on the other's draw.
          const mixed = held(onRows, x, lanes, [2 * rl, 2 * rl + 1]);
          expect(mixed.length).toBe(256);
          for (let q = 0; q < 4; q++) expect(mixed).not.toEqual(region(q, lane & 3));
        }
      }
      // The 16 writers' regions cover the warp tile once.
      expect(covered.size).toBe(64 * 64);
    }
  }, SLOW);

  test('the 16 storing lanes write the default\'s word indices, one word each, on 16 banks', () => {
    for (let it = 0; it < 4; it++) {
      const tbuf = (it & 1) * TWORDS;
      const hits = new Uint8Array(2 * TWORDS);
      const bad = [];
      for (let warp = 0; warp < WARPS; warp++) {
        const writers = Array.from({ length: 32 }, (_, l) => l).filter(rpWriter);
        expect(writers).toEqual([0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]);
        // Their regions are the warp's 16, as the default's (lane, rl) pairs are.
        const mine = writers.map((l) => rpL0(warp, l)).sort((a, b) => a - b);
        const dflt = [];
        for (let l = 0; l < 32; l++) if (dfWriter(l)) dflt.push(dfL(warp, l, 0), dfL(warp, l, 1));
        expect(mine).toEqual(Array.from({ length: 16 }, (_, i) => warp * 16 + i));
        expect(dflt.sort((a, b) => a - b)).toEqual(mine);
        for (let c = 0; c < CHUNKS; c++) {
          const banks = new Set();
          for (const lane of writers) {
            const L = rpL0(warp, lane);
            // TWORDS is above the index's bits, so moving it to a buffer sets the buffer.
            if (rpTK0(L) >= TWORDS) bad.push(`tK of region ${L} is ${rpTK0(L)}`);
            const a = heldTK(warp, lane, it) ^ c;
            if (a !== wordAt(tbuf, L, c)) bad.push(`tile ${it} warp ${warp} lane ${lane} chunk ${c}: ${a}`);
            hits[a]++;
            banks.add(a % 32);
          }
          if (banks.size !== 16) bad.push(`tile ${it} warp ${warp} chunk ${c}: ${banks.size} banks`);
        }
      }
      expect(bad).toEqual([]);
      // Every word of the tile's buffer once, none of the other's.
      expect(Array.from(hits.slice(tbuf, tbuf + TWORDS)).every((n) => n === 1)).toBe(true);
      expect(hits.reduce((a, n) => a + n, 0)).toBe(TWORDS);
    }
  });

  test('for random A and B the hashers read the default\'s word for every region and chunk, and it is the reference\'s', () => {
    const sim = simulation();
    const want = referenceWords(sim.Pref);
    for (const it of [0, 1, 2]) {
      const rp = readout(sim.Prp, 'rowpair', it);
      const df = readout(sim.Pdf, 'default', it);
      for (const r of [rp, df]) {
        expect(Array.from(r.writes.slice(r.tbuf, r.tbuf + TWORDS)).every((n) => n === 1)).toBe(true);
        expect(Array.from(r.writes.slice(TWORDS - r.tbuf, 2 * TWORDS - r.tbuf)).every((n) => n === 0)).toBe(true);
      }
      // Word for word, the same buffers; and the words are the reference's.
      expect(Buffer.compare(Buffer.from(rp.sT.buffer), Buffer.from(df.sT.buffer))).toBe(0);
      expect(wrong(hashed(rp.sT, rp.tbuf), want)).toBe(0);
      expect(hashed(rp.sT, rp.tbuf)).toEqual(want);
      expect(hashed(df.sT, df.tbuf)).toEqual(want);
    }
    // The words are not trivially equal: every region's chunks differ.
    expect(want.every((w) => new Set(w).size === CHUNKS)).toBe(true);
    // Against the JS reference miner's fold (foldTranscript, no noise: the draw here
    // stands for the noised operands), for regions in each warp and both row groups.
    const k = 2 ** K_LOG2;
    for (const L of [0, 7, 21, 46, 63, 64, 90, 127]) {
      const tile = regionToTile(regionOf(L, SIM.rbg, SIM.cbg, SIM.M / 16), { m: SIM.M, n: SIM.N });
      const jackpot = foldTranscript({
        A: sim.A, Bt: sim.B, rows: tile.rows, cols: tile.cols, k, rank: 128,
        noiseA: tile.rows.map(() => new Int8Array(k)), noiseB: tile.cols.map(() => new Int8Array(k)),
      });
      expect([L, Array.from(jackpot)]).toEqual([L, want[L]]);
    }
  }, SLOW);

  test('the simulation sees a broken draw or readout', () => {
    const sim = simulation();
    const want = referenceWords(sim.Pref);
    const words = (P, how, brk) => {
      const r = readout(P, how, 1, brk);
      return { wrong: wrong(hashed(r.sT, r.tbuf), want), notOnce: r.writes.filter((n, a) => a >= TWORDS && n !== 1).length };
    };
    // Unbroken, both readouts on their own draws.
    expect(words(sim.Prp, 'rowpair', {})).toEqual({ wrong: 0, notOnce: 0 });
    expect(words(sim.Pdf, 'default', {})).toEqual({ wrong: 0, notOnce: 0 });
    // The switch's readout on the default draw, and the default's on the switch's.
    expect(words(sim.Pdf, 'rowpair', {}).wrong).toBeGreaterThan(1900);
    expect(words(sim.Prp, 'default', {}).wrong).toBeGreaterThan(1900);
    // The shuffle by 8, one chain only, the default's region for a writer, other writers.
    expect(words(sim.Prp, 'rowpair', { shfl: 8 }).wrong).toBeGreaterThan(1900);
    expect(words(sim.Prp, 'rowpair', { oneChain: true }).wrong).toBeGreaterThan(1900);
    const dflt = words(sim.Prp, 'rowpair', { L0: (warp, lane) => dfL(warp, lane, 0) });
    expect(dflt.wrong).toBeGreaterThan(1000);
    expect(dflt.notOnce).toBeGreaterThan(0);
    const eights = words(sim.Prp, 'rowpair', { writer: (lane) => (lane & 8) === 0 });
    expect(eights.wrong).toBeGreaterThan(900);
    expect(eights.notOnce).toBeGreaterThan(0);
    // A row map with s and j's low bit traded (bits 2 and 4 of the fold's row): lanes L and
    // L ^ 4 then hold rows with bits 2 and 3 fixed, no region's.
    const traded = (r, rows) => {
      const q = rq(r, rows);
      return (r | 63) < rows ? (q & ~20) | (((q >> 2) & 1) << 4) | (((q >> 4) & 1) << 2) : q;
    };
    const bad = draw(sim.A, SIM.M, K_LOG2, { map: traded });
    expect(bad.written.every((n) => n === 1)).toBe(true);
    const P = products(foldView(bad.out, K_LOG2, SIM.rbg * 8, 2).view, sim.FB.view, 128, 256, 2 ** K_LOG2);
    expect(words(P, 'rowpair', {}).wrong).toBeGreaterThan(1900);
    // Trading u0 and u1 (bits 3 and 0) instead moves rows only inside each region: the same
    // 256 sums, the same words. So the checks see where a region's rows go, not their order.
    const inRegion = (r, rows) => {
      const q = rq(r, rows);
      return (r | 63) < rows ? (q & ~9) | ((q & 1) << 3) | ((q >> 3) & 1) : q;
    };
    const same = draw(sim.A, SIM.M, K_LOG2, { map: inRegion });
    const P2 = products(foldView(same.out, K_LOG2, SIM.rbg * 8, 2).view, sim.FB.view, 128, 256, 2 ** K_LOG2);
    expect(words(P2, 'rowpair', {})).toEqual({ wrong: 0, notOnce: 0 });
  }, SLOW);
});
