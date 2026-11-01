'use strict';

// Readout v2 in the RDNA folds (PEARL_AMD_READOUT_V2; earn/native/amd/src/pearl_amd_fold.h,
// round 4 item 4). No HIP compiler runs here, so these read v2's address arithmetic out of the
// source and run it. Over every lane of a block it must map one-to-one onto sB's row padding, put
// a region's eight lanes on one word, keep one instruction's four words in different banks, and
// stay clear of every staging store and fragment load. Then a batch goes through it, on LDS an
// earlier block left dirty, and must give the reference's transcripts. Last: the guards, the CPU
// build, the sweep's check size and fold rounds, CI's step, and the instruction census on
// stand-ins.

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const {
  ROWS_PATTERN, COLS_PATTERN, ROWS_MASK, COLS_MASK, expandOffset,
} = require('../src/shared/miner/pearlhash');
const census = require('../native/amd/test/readout-census');

const ROOT = path.join(__dirname, '..', '..');
const AMD = path.join(__dirname, '..', 'native', 'amd');
const read = (f) => fs.readFileSync(path.join(AMD, f), 'utf8').replace(/\r\n/g, '\n');
const FOLD = read('src/pearl_amd_fold.h');
const BUILD = read('build-amd.sh');
const CPUCHECKS = read('test/run-cpu-checks.sh');
const SWEEP = read('test/sweep.sh');
const CENSUS_SH = read('test/readout-census.sh');
const CI = fs.readFileSync(path.join(ROOT, '.github', 'workflows', 'native-core.yml'), 'utf8')
  .replace(/\r\n/g, '\n');

const hasBash = spawnSync('bash', ['--version'], { stdio: 'ignore' }).status === 0;
const bashIt = hasBash ? it : it.skip;

// The text from `first` to the end of `last`, each found once.
function between(text, first, last) {
  const a = text.indexOf(first);
  if (a < 0 || text.indexOf(first, a + 1) >= 0) throw new Error('not found once: ' + first);
  const b = text.indexOf(last, a);
  if (b < 0) throw new Error('not found: ' + last);
  return text.slice(a, b + last.length);
}
function grab(text, re) {
  const m = re.exec(text);
  if (!m) throw new Error('not found: ' + re);
  return m[1];
}
// A C expression of this fold as JS: no casts, no u suffixes, sB[b] as a call, and integer
// division where a name is divided by a number (every other division here is exact).
const toJs = (c) => c
  .replace(/\((?:const )?(?:u?int(?:8|16|32|64)_t|size_t|int|unsigned)\)/g, '')
  .replace(/\b(\d+)u\b/g, '$1')
  .replace(/sB\[(\w+)\]/g, 'sB($1)')
  .replace(/\b([a-z]\w*) \/ (\d+)\b/g, 'Math.floor($1 / $2)');

// The constants, from the source.
const num = (re) => Number(grab(FOLD, re));
const KS = num(/#define PEARL_AMD_SLAB (\d+)u/), BM = num(/#define PEARL_AMD_BM (\d+)u/);
const BN = num(/#define PEARL_AMD_BN (\d+)u/), T = num(/#define PEARL_AMD_THREADS (\d+)u/);
const S = KS + 16, QR = KS / 16, UA = BM * QR, UB = BN * QR, U = (UA + UB) / T;
const W = 32, WM = 64, WN = 64, TM = 16, TN = 16, TK = 16, WAVES = T / W, BUCKETS = 16;
// LDS as the fold declares it: sA[2][BM * S], then sB[2][BN * S] (the order doesn't change any
// check here: every address is relative to its own array, and the arrays are 16-byte aligned).
const sA = (b) => b * BM * S, sB = (b) => 2 * BM * S + b * BN * S, LDS = 2 * (BM + BN) * S;

function policy(name) {
  const a = FOLD.indexOf('struct ' + name + ' {');
  return FOLD.slice(a, FOLD.indexOf('\n};', a));
}

describe('the source', () => {
  it('has the constants and shapes these tests assume', () => {
    expect([KS, S, BM, BN, T, LDS]).toEqual([32, 48, 128, 256, 256, 36864]);
    expect(FOLD).toContain('constexpr int KS = (int)PEARL_AMD_SLAB, S = KS + 16;');
    for (const p of ['PearlAmdWmma11', 'PearlAmdWmma12']) {
      expect(policy(p)).toContain('static constexpr int kWave = 32, kTM = 16, kTN = 16, kTK = 16;');
      expect(policy(p)).toContain('static constexpr int kWR = 2, kWC = 4, kMT = 4, kNT = 4;');
      expect(policy(p)).toContain('static constexpr bool kRoByLane = false;');
    }
    expect(FOLD).toContain('__shared__ __attribute__((aligned(16))) int8_t sA[2][BM * S];\n'
      + '  __shared__ __attribute__((aligned(16))) int8_t sB[2][BN * S];');
    expect(FOLD).toContain('const int wr = wave / P::kWC, wc = wave % P::kWC;');
  });

  it('is off by default, refuses a bad value and the drop probe, and is RDNA only', () => {
    expect(FOLD).toContain('#ifndef PEARL_AMD_READOUT_V2\n#define PEARL_AMD_READOUT_V2 0\n#endif');
    expect(FOLD).toContain('#if PEARL_AMD_READOUT_V2 != 0 && PEARL_AMD_READOUT_V2 != 1 && PEARL_AMD_READOUT_V2 != 2\n#error');
    expect(FOLD).toContain('#if PEARL_AMD_READOUT_V2 && PEARL_AMD_READOUT_DROP\n#error');
    expect(FOLD).toContain('constexpr bool kV2 = PEARL_AMD_READOUT_V2 != 0 &&\n'
      + '                       (std::is_same<P, PearlAmdWmma12>::value || std::is_same<P, PearlAmdWmma11>::value);');
    expect(FOLD).toContain('constexpr bool kV2Wide = kV2 && PEARL_AMD_READOUT_V2 == 1;');
    // The layout's assumptions, checked by the compiler for every fold it builds v2 into.
    expect(FOLD).toContain('static_assert(!kV2 || (W == 32 && T / W == 8 && P::kWC == 4 && WN == 64 && RB == 2 && CB == 1 &&\n'
      + '                         NRO == 2 && S - KS == 16 && QR * 16 == KS && PEARL_JACKPOT_BUCKETS == 16 &&\n'
      + '                         PEARL_FOLD_CHUNKS <= PEARL_JACKPOT_BUCKETS),');
    // A v2 build is a correct one: it sets no cap bit, so the host mines with it.
    expect(FOLD).not.toMatch(/PEARL_AMD_CAP_\w*V2/);
  });

  it('combines with a relaxed, workgroup-scope XOR whose result goes unused', () => {
    const helper = between(FOLD, 'template <class U>\n__device__ __forceinline__ void pearl_amd_lds_xor(U *p, U v) {', '\n}\n');
    expect(helper).toContain('#if !PEARL_AMD_EMU && defined(__HIP_DEVICE_COMPILE__)\n'
      + '  __hip_atomic_fetch_xor(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);\n#else\n'
      + '  __atomic_fetch_xor(p, v, __ATOMIC_RELAXED);\n#endif');
  });

  it('has no cross-lane step, compare or select on the v2 path', () => {
    const v2 = between(FOLD, '        if constexpr (kV2) {\n', '          continue;\n        }\n');
    for (const re of [/pearl_amd_group_xor/, /\bmine\b/, /\bhi\b/, /keepLo|keepHi/, /__shfl|dpp|permlane/, /\?/])
      expect(v2).not.toMatch(re);
    expect(v2).toContain('          if constexpr (kV2Wide) {\n'
      + '            pearl_amd_lds_xor(reinterpret_cast<uint64_t *>(p), (uint64_t)x[0] | (uint64_t)x[1] << 32);\n'
      + '          } else {\n'
      + '            pearl_amd_lds_xor(reinterpret_cast<uint32_t *>(p), x[0]);\n'
      + '            pearl_amd_lds_xor(reinterpret_cast<uint32_t *>(p + 4), x[1]);\n'
      + '          }\n');
    // It comes after the unchanged tree and before today's keep step, which it skips.
    const readout = between(FOLD, '  auto readout = [&](uint32_t c) {', '\n  };\n');
    const tree = readout.indexOf('x[P::ro_of(g)] ^= (uint32_t)acc[mt][nt][g];');
    expect(tree).toBeGreaterThan(0);
    expect(readout.indexOf('if constexpr (kV2) {')).toBeGreaterThan(tree);
    expect(readout.indexOf('if constexpr (kV2) {')).toBeLessThan(readout.indexOf('pearl_amd_group_xor(x[q])'));
  });

  it('keeps today\'s readout and end of tile, word for word, for every other build', () => {
    expect(FOLD).toContain('    const bool mine = (c & 7u) == r, hi = c >= 8u;');
    expect(FOLD).toContain('          if constexpr (kDrop) {\n'
      + '            keepLo[rb][cb][q] ^= x[q];   // timing only: the transcript is wrong\n'
      + '          } else {\n'
      + '            const uint32_t v = pearl_amd_group_xor(x[q]);\n'
      + '            keepLo[rb][cb][q] = (mine && !hi) ? v : keepLo[rb][cb][q];\n'
      + '            keepHi[rb][cb][q] = (mine && hi) ? v : keepHi[rb][cb][q];\n'
      + '          }');
    expect(FOLD).toContain('    return;\n  }\n  // Each lane writes words r and r + 8 of its regions.\n'
      + '  const uint32_t coIdx = (uint32_t)((lane >> 1) & 3);');
    expect(FOLD).toContain('        w[r] = keepLo[rb][cb][q];\n        w[r + 8] = keepHi[rb][cb][q];');
    // Each v2 step is behind `if constexpr (kV2)`; nothing else in the fold body names v2's words.
    const body = between(FOLD, 'template <class P>\n__device__ __forceinline__ void pearl_amd_fold_body(', '\n}\n');
    const uses = body.split('\n').filter((l) => /v2Wave|v2Lane/.test(l));
    expect(uses.filter((l) => /\[\[maybe_unused\]\] int8_t \*const v2(Wave|Lane) = /.test(l))).toHaveLength(2);
    expect(body.match(/if constexpr \(kV2\) \{/g)).toHaveLength(3);
  });
});

// v2's arithmetic, out of the source.
const V2_ZERO = between(FOLD, '  if constexpr (kV2) {\n    // LDS holds', '  }\n  __syncthreads();');
const V2_READOUT = between(FOLD, '        if constexpr (kV2) {\n', '          continue;\n        }\n');
const V2_END = between(FOLD, '  if constexpr (kV2) {\n    // v2: every XOR came', '    return;\n  }\n');
const fn = (args, body) => new Function('S', 'KS', 'WN', 'WM', 'sB', 'PEARL_JACKPOT_BUCKETS', ...args, body)
  .bind(null, S, KS, WN, WM, sB, BUCKETS);
const L = {
  base: fn(['wr', 'wc', 'lane'], `const v2Wave = ${toJs(grab(FOLD, /int8_t \*const v2Wave = ([^;]+);/))};\n`
    + `const v2Lane = ${toJs(grab(FOLD, /int8_t \*const v2Lane = ([^;]+);/))};\nreturn { v2Wave, v2Lane };`),
  zero: fn(['v2Wave', 'lane'], 'return [' + [...V2_ZERO.matchAll(/reinterpret_cast<pearl_amd_v4i \*>\((.+)\) = zero;/g)]
    .map((m) => toJs(m[1])).join(', ') + '];'),
  xor: fn(['v2Lane', 'c', 'rb'], `return ${toJs(grab(V2_READOUT, /int8_t \*const p = ([^;]+);/))};`),
  back: fn(['v2Wave', 'lane', 'tileRow', 'tileCol', 'wr', 'wc', 'rows_valid'],
    `const ${toJs(grab(V2_END, /const int (rb = [^;]+);/))};\n`
    + `const src = ${toJs(grab(V2_END, /const int8_t \*const src = ([^;]+);/))};\n`
    + `const loads = [0, 1, 2, 3].map((i) => src + ${toJs(grab(V2_END, /reinterpret_cast<const uint64_t \*>\(src \+ ([^)]+)\);/))});\n`
    + `const rowIdx = ${toJs(grab(V2_END, /const uint32_t rowIdx = ([^;]+);/))};\n`
    + `const colIdx = ${toJs(grab(V2_END, /const uint32_t colIdx = ([^;]+);/))};\n`
    + `const w = ${toJs(grab(V2_END, /uint32_t \*const w = tr \+ ([^;]+);/))};\n`
    + 'return { loads, w };'),
};
const waveOf = (w) => ({ wr: Math.floor(w / 4), wc: w % 4 });

describe('the v2 layout', () => {
  it('reads back as the low word for row offset 0 and the high word for 1, stored region by region', () => {
    expect(V2_END).toContain('      w0[i] = (int32_t)(uint32_t)v;\n      w1[i] = (int32_t)(uint32_t)(v >> 32);');
    expect(V2_END).toContain('    *reinterpret_cast<pearl_amd_v4i *>(w) = w0;');
    expect(V2_END).toContain('    *reinterpret_cast<pearl_amd_v4i *>(w + PEARL_JACKPOT_BUCKETS) = w1;');
    expect(V2_ZERO).toContain('    const pearl_amd_v4i zero = {0, 0, 0, 0};');
    expect(L.zero(0, 0)).toHaveLength(2);
  });

  // Every word: wave, block, column offset, chunk -> one 8-byte pair (row offsets 0 and 1).
  const words = new Map();   // byte address -> "wave rb co c"
  const lanesAt = new Map(); // "wave rb c addr" -> lanes
  const clashes = [];        // two different words at one address
  for (let wave = 0; wave < WAVES; wave++) {
    const { wr, wc } = waveOf(wave);
    for (let lane = 0; lane < W; lane++) {
      const { v2Lane } = L.base(wr, wc, lane);
      for (let rb = 0; rb < 2; rb++)
        for (let c = 0; c < 16; c++) {
          const a = L.xor(v2Lane, c, rb), co = (lane >> 1) & 3;
          const key = `${wave} ${rb} ${co} ${c}`;
          if (words.has(a) && words.get(a) !== key) clashes.push(`word ${a}: ${words.get(a)} and ${key}`);
          words.set(a, key);
          const k = `${wave} ${rb} ${c} ${a}`;
          if (!lanesAt.has(k)) lanesAt.set(k, []);
          lanesAt.get(k).push(lane);
        }
    }
  }
  const pairBytes = (a) => [0, 1, 2, 3, 4, 5, 6, 7].map((i) => a + i);
  const wordSet = new Set([].concat(...[...words.keys()].map(pairBytes)));

  it('maps every word of every region of a block to its own 8 bytes, aligned, in sB\'s padding', () => {
    // 8 waves x 2 blocks x 4 column offsets x 16 chunks, each one pair: all 8 KB of padding.
    expect(clashes.slice(0, 3)).toEqual([]);
    expect(words.size).toBe(1024);
    expect(wordSet.size).toBe(8192);
    for (const [a, key] of words) {
      const wave = Number(key.split(' ')[0]), { wr, wc } = waveOf(wave);
      expect(a % 8).toBe(0);
      const buf = a >= sB(1) ? 1 : 0, off = a - sB(buf), row = Math.floor(off / S), byte = off % S;
      expect(a).toBeGreaterThanOrEqual(sB(0));
      expect(buf).toBe(wr);                                      // sB[wr]
      expect(Math.floor(row / 64)).toBe(wc);                     // the wave's own 64 rows
      expect(byte === KS || byte === KS + 8).toBe(true);         // inside bytes KS..KS+15
    }
  });

  it('sends a region\'s eight lanes (bits 0, 3, 4) to one word, and one instruction to four banks', () => {
    for (let wave = 0; wave < WAVES; wave++)
      for (let rb = 0; rb < 2; rb++)
        for (let c = 0; c < 16; c++) {
          const groups = [...lanesAt].filter(([k]) => k.startsWith(`${wave} ${rb} ${c} `));
          expect(groups).toHaveLength(4);
          const addrs = groups.map(([k]) => Number(k.split(' ')[3]));
          for (const [, lanes] of groups) {
            expect(lanes).toHaveLength(8);
            // The lanes pearl_amd_group_xor combined: the same lane bits 1-2, any bits 0, 3, 4.
            expect(new Set(lanes.map((l) => (l >> 1) & 3)).size).toBe(1);
            expect(new Set(lanes.map((l) => l & 0x19)).size).toBe(8);
          }
          for (const banks of [32, 64]) {
            // ds_xor_b64: two banks an address; ds_xor_b32: row offset 0, then row offset 1.
            const wide = [].concat(...addrs.map((a) => [a / 4, a / 4 + 1].map((x) => x % banks)));
            expect(new Set(wide).size).toBe(8);
            for (const q of [0, 1]) expect(new Set(addrs.map((a) => (a / 4 + q) % banks)).size).toBe(4);
          }
        }
  });

  it('zeroes and reads back exactly those bytes, and nothing else', () => {
    const zero = new Set(), back = new Set();
    for (let wave = 0; wave < WAVES; wave++) {
      const { wr, wc } = waveOf(wave);
      for (let lane = 0; lane < W; lane++) {
        const { v2Wave } = L.base(wr, wc, lane);
        for (const a of L.zero(v2Wave, lane)) {
          expect(a % 16).toBe(0);   // ds_store_b128
          for (let i = 0; i < 16; i++) { expect(zero.has(a + i)).toBe(false); zero.add(a + i); }
        }
        for (const a of L.back(v2Wave, lane, 0, 0, wr, wc, 8).loads) {
          expect(a % 8).toBe(0);    // ds_load_b64
          for (const b of pairBytes(a)) { expect(back.has(b)).toBe(false); back.add(b); }
        }
      }
    }
    expect([...zero].sort((a, b) => a - b)).toEqual([...wordSet].sort((a, b) => a - b));
    expect([...back].sort((a, b) => a - b)).toEqual([...wordSet].sort((a, b) => a - b));
  });

  it('never shares a byte with a staging store or a fragment load, on either buffer', () => {
    // PEARL_AMD_STORER and the fragment loads, as the source writes them.
    expect(FOLD).toContain('int8_t *dst = idx < UA ? sA[buf] + (idx / QR) * S : sB[buf] + ((idx - UA) / QR) * S;');
    expect(FOLD).toContain('*reinterpret_cast<pearl_amd_v4i *>(dst + 16 * (idx % QR)) = R[u];');
    expect(FOLD).toContain('const int8_t *aBase = sA[B] + wr * WM * S;\n    const int8_t *bBase = sB[B] + wc * WN * S;');
    expect(FOLD).toContain('a[mt] = P::load_a(aBase + mt * P::kTM * S + kk * P::kTK, S, lane);');
    expect(FOLD).toContain('b[nt] = P::load_b(bBase + nt * P::kTN * S + kk * P::kTK, S, lane);');
    expect(policy('PearlAmdWmma12')).toContain('s + (lane & 15) * stride + 8 * (lane >> 4)');
    expect(policy('PearlAmdWmma11')).toContain('*reinterpret_cast<const pearl_amd_v4i *>(s + (lane & 15) * stride);');
    let staged = 0, loaded = 0, shared = 0;
    const clear = (a, n) => { for (let i = 0; i < n; i++) if (wordSet.has(a + i)) shared++; };
    for (let buf = 0; buf < 2; buf++) {
      for (let t = 0; t < T; t++)
        for (let u = 0; u < U; u++) {
          const idx = t + u * T;
          const dst = idx < UA ? sA(buf) + Math.floor(idx / QR) * S : sB(buf) + Math.floor((idx - UA) / QR) * S;
          clear(dst + 16 * (idx % QR), 16);
          staged++;
        }
      for (let wave = 0; wave < WAVES; wave++) {
        const { wr, wc } = waveOf(wave);
        for (let lane = 0; lane < W; lane++)
          for (let kk = 0; kk < KS / TK; kk++)
            for (let i = 0; i < 4; i++) {
              const a = sA(buf) + wr * WM * S + i * TM * S + kk * TK, b = sB(buf) + wc * WN * S + i * TN * S + kk * TK;
              for (const s of [a, b]) {
                clear(s + (lane & 15) * S + 8 * (lane >> 4), 8);   // gfx12
                clear(s + (lane & 15) * S, 16);                    // gfx11
                loaded++;
              }
            }
      }
    }
    expect(shared).toBe(0);
    expect(staged).toBe(2 * (UA + UB));
    expect(loaded).toBe(2 * WAVES * W * 2 * 4 * 2);
  });
});

// The whole fold's readout, as v2 does it, against the fold test's reference.
describe('a batch through v2', () => {
  const tree = between(FOLD, '        for (int q = 0; q < NRO; q++) x[q] = 0u;', 'x[P::ro_of(g)] ^= (uint32_t)acc[mt][nt][g];');
  it('runs the tree it was written against', () => {
    expect(tree).toContain('for (int mt = rb * (32 / P::kTM); mt < (rb + 1) * (32 / P::kTM); mt++)');
    expect(tree).toContain('for (int nt = cb * (64 / P::kTN); nt < (cb + 1) * (64 / P::kTN); nt++)');
    expect(tree).toContain('for (int g = 0; g < P::kRegs; g++)');
  });

  // Each policy's C layout and row-offset split, from the source.
  const layout = (name) => {
    const p = policy(name);
    const ret = (re) => toJs(grab(p, re));
    return {
      row: new Function('lane', 'g', `return ${ret(/int c_row\(int lane, int g\) \{ return ([^;]+); \}/)};`),
      col: new Function('lane', 'g', `return ${ret(/int c_col\(int lane, int\) \{ return ([^;]+); \}/)};`),
      ro: new Function('g', `return ${ret(/int ro_of\(int g\) \{ return ([^;]+); \}/)};`),
    };
  };

  // The running sum of C[row][col] after chunk c: any 32-bit value will do.
  const m = 256, cols = 512, rowsValid = m / ROWS_PATTERN.length, colsValid = cols / COLS_PATTERN.length;
  const sums = Array.from({ length: 16 }, (_, c) => {
    const out = new Uint32Array(m * cols);
    for (let i = 0; i < out.length; i++) {
      let h = Math.imul(c + 1, 0x9E3779B1) ^ Math.imul(i + 7, 0x85EBCA77);
      h ^= h >>> 15; h = Math.imul(h, 0x2C1B3C6D); h ^= h >>> 12;
      out[i] = h >>> 0;
    }
    return out;
  });
  const cell = (c, row, col) => sums[c][row * cols + col];
  const want = new Uint32Array(rowsValid * colsValid * 16);
  for (let ci = 0; ci < colsValid; ci++)
    for (let ri = 0; ri < rowsValid; ri++) {
      const ro = expandOffset(ri, ROWS_MASK), co = expandOffset(ci, COLS_MASK);
      for (let c = 0; c < 16; c++) {
        let x = 0;
        for (const a of ROWS_PATTERN) for (const b of COLS_PATTERN) x ^= cell(c, ro | a, co | b);
        want[(ci * rowsValid + ri) * 16 + c] = x >>> 0;
      }
    }

  // One CU running the batch's blocks one after another on the same LDS, which starts dirty.
  function run(name, { wide = true, zero = true } = {}) {
    const lay = layout(name);
    const lds = new Uint32Array(LDS / 4);
    let seed = 12345;
    for (let i = 0; i < lds.length; i++) { seed = (Math.imul(seed, 1103515245) + 12345) >>> 0; lds[i] = seed; }
    const tr = new Uint32Array(want.length).fill(0xA5A5A5A5), writes = new Uint8Array(want.length);
    let misaligned = 0;
    const xor32 = (a, v) => { if (a % 4) misaligned++; else lds[a / 4] = (lds[a / 4] ^ v) >>> 0; };
    for (let by = 0; by < m / BM; by++)
      for (let bx = 0; bx < cols / BN; bx++) {
        const tileRow = by * BM, tileCol = bx * BN;
        if (zero)
          for (let wave = 0; wave < WAVES; wave++)
            for (let lane = 0; lane < W; lane++) {
              const { wr, wc } = waveOf(wave);
              for (const a of L.zero(L.base(wr, wc, lane).v2Wave, lane)) lds.fill(0, a / 4, a / 4 + 4);
            }
        for (let c = 0; c < 16; c++)
          for (let wave = 0; wave < WAVES; wave++) {
            const { wr, wc } = waveOf(wave);
            for (let lane = 0; lane < W; lane++) {
              const { v2Lane } = L.base(wr, wc, lane);
              for (let rb = 0; rb < 2; rb++) {
                const x = [0, 0];
                for (let mt = rb * (32 / TM); mt < (rb + 1) * (32 / TM); mt++)
                  for (let nt = 0; nt < 64 / TN; nt++)
                    for (let g = 0; g < 8; g++)
                      x[lay.ro(g)] ^= cell(c, tileRow + wr * WM + mt * TM + lay.row(lane, g), tileCol + wc * WN + nt * TN + lay.col(lane, g));
                const p = L.xor(v2Lane, c, rb);
                if (wide && p % 8) misaligned++;
                xor32(p, x[0] >>> 0);       // the low word of the pair, or the first ds_xor_b32
                xor32(p + 4, x[1] >>> 0);   // the high word, or the second
              }
            }
          }
        for (let wave = 0; wave < WAVES; wave++) {
          const { wr, wc } = waveOf(wave);
          for (let lane = 0; lane < W; lane++) {
            const { loads, w } = L.back(L.base(wr, wc, lane).v2Wave, lane, tileRow, tileCol, wr, wc, rowsValid);
            if (!Number.isInteger(w)) misaligned++;
            loads.forEach((a, i) => {
              for (const [at, v] of [[w + i, lds[a / 4]], [w + BUCKETS + i, lds[a / 4 + 1]]]) {
                tr[at] = v;
                writes[at]++;
              }
            });
          }
        }
      }
    return { tr, writes, misaligned };
  }

  for (const name of ['PearlAmdWmma12', 'PearlAmdWmma11'])
    for (const wide of [true, false])
      it(`gives every region's 16 words, each written once, in ${name} with ds_xor_b${wide ? 64 : 32}`, () => {
        const { tr, writes, misaligned } = run(name, { wide });
        expect(misaligned).toBe(0);
        expect(writes.every((n) => n === 1)).toBe(true);
        let bad = 0;
        for (let i = 0; i < want.length; i++) if (tr[i] !== want[i]) bad++;
        expect(bad).toBe(0);
      });

  it('needs the zeroing: without it the dirty LDS leaks into the words', () => {
    const { tr } = run('PearlAmdWmma12', { zero: false });
    let bad = 0;
    for (let i = 0; i < want.length; i++) if (tr[i] !== want[i]) bad++;
    expect(bad).toBeGreaterThan(want.length / 2);
  });
});

describe('the CPU checks', () => {
  it('build v2\'s fold tests and run every one they build', () => {
    expect(BUILD).toContain('flags="-std=c++17 -O2 -fPIC -DNDEBUG ${PEARL_AMD_CFLAGS:-}"');
    const built = [...BUILD.matchAll(/-o "\$OUT\/(pearl_amd_foldtest_cpu_v2\w*)"/g)].map((x) => x[1]);
    expect(built).toEqual(['pearl_amd_foldtest_cpu_v2', 'pearl_amd_foldtest_cpu_v2_pf2',
      'pearl_amd_foldtest_cpu_v2_pfx', 'pearl_amd_foldtest_cpu_v2w_pfx']);
    for (const b of built) expect(CPUCHECKS).toContain('"$B/' + b + '"');
    expect(CPUCHECKS).toContain('"$B/pearl_amd_foldtest_cpu_v2_pf2" 256 512 1 1 1 wmma11');
  });

  // build-amd.sh cpu with a stand-in compiler that logs its arguments.
  bashIt('builds them as v2 whatever PEARL_AMD_CFLAGS says about the readout, and keeps its other flags', () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'amd-cpu-'));
    try {
      const amd = path.join(tmp, 'amd'), src = path.join(tmp, 'src'), log = path.join(tmp, 'cc.log');
      for (const d of [amd, src, path.join(tmp, 'napi'), path.join(tmp, 'node'), path.join(tmp, 'hipcpu')])
        fs.mkdirSync(d, { recursive: true });
      fs.writeFileSync(path.join(amd, 'build-amd.sh'), BUILD);
      fs.writeFileSync(path.join(src, 'pearl_kernel.cu'),
        '// portable\n// A 16-byte global->shared copy that does not pass through registers\n');
      fs.writeFileSync(path.join(tmp, 'napi', 'napi.h'), '');
      fs.writeFileSync(path.join(tmp, 'node', 'node_api.h'), '');
      fs.writeFileSync(path.join(tmp, 'cc.sh'), 'printf "%s\\n" "$*" >> "$CC_LOG"\n');
      const fwd = (p) => p.replace(/\\/g, '/');
      const r = spawnSync('bash', [fwd(path.join(amd, 'build-amd.sh')), 'cpu'], {
        encoding: 'utf8',
        env: Object.assign({}, process.env, {
          CPU_CXX: 'bash ' + fwd(path.join(tmp, 'cc.sh')), CC_LOG: fwd(log), HIP_CPU_PATH: fwd(path.join(tmp, 'hipcpu')),
          NODE_ADDON_API: fwd(path.join(tmp, 'napi')), NODE_INCLUDE: fwd(path.join(tmp, 'node')),
          PEARL_AMD_CFLAGS: '-DPEARL_AMD_READOUT_DROP=1 -DPEARL_AMD_TILED=0',
        }),
      });
      expect(r.status).toBe(0);
      // The macros each compile ends with, -D and -U taken in order as the compiler takes them.
      const macros = (line) => {
        const out = {};
        for (const a of line.split(/\s+/)) {
          let x = /^-D(\w+)(?:=(.*))?$/.exec(a);
          if (x) out[x[1]] = x[2] === undefined ? '1' : x[2];
          x = /^-U(\w+)$/.exec(a);
          if (x) delete out[x[1]];
        }
        return out;
      };
      const lines = fs.readFileSync(log, 'utf8').split('\n').filter(Boolean);
      const byOut = (name) => lines.find((l) => l.includes('/' + name + ' ') || l.endsWith('/' + name));
      for (const [name, v2] of [['pearl_amd_foldtest_cpu_v2', '1'], ['pearl_amd_foldtest_cpu_v2_pf2', '1'],
        ['pearl_amd_foldtest_cpu_v2_pfx', '1'], ['pearl_amd_foldtest_cpu_v2w_pfx', '2']]) {
        const d = macros(byOut(name));
        expect([name, d.PEARL_AMD_READOUT_V2, d.PEARL_AMD_READOUT_DROP, d.PEARL_AMD_TILED]).toEqual([name, v2, undefined, '0']);
      }
      expect(macros(byOut('pearl_amd_foldtest_cpu_v2_pfx'))).toMatchObject({ PEARL_AMD_PF2: '1', PEARL_AMD_PFX: '1', PEARL_AMD_TID: '1' });
      // The core and the other fold tests get PEARL_AMD_CFLAGS as given.
      for (const name of ['pearl_amd_foldtest_cpu', 'pearl_host_hip.o'])
        expect(macros(byOut(name))).toMatchObject({ PEARL_AMD_READOUT_DROP: '1', PEARL_AMD_TILED: '0' });
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  }, 30000);
});

describe('sweep.sh', () => {
  // A stand-in build-amd.sh whose fold test answers by its flags: base 50.000 TMAC/s, v2 50.500,
  // v2w 50.250, drop 50.750 and wrong words; and a stand-in hashrate.js that reads the flags out of
  // the core.
  const STUB = [
    '#!/bin/bash',
    'mkdir -p build',
    'printf \'#!/bin/bash\\nf="%s"\\n\' "${PEARL_AMD_CFLAGS:-}" > build/pearl_amd_foldtest',
    'cat >> build/pearl_amd_foldtest <<\'FT\'',
    'case "$f" in *READOUT_DROP=1*) t=50.750; ok=0 ;; *READOUT_V2=1*) t=50.500; ok=1 ;; *READOUT_V2=2*) t=50.250; ok=1 ;;',
    '  *) t=50.000; ok=1 ;; esac',
    'if [ "$5" = 0 ]; then echo "$6  timed only: $1 x $2, $4 launches, 20.0 s, $t TMAC/s"; exit 0; fi',
    'if [ $ok = 1 ]; then echo "$6  ok: $1 x $2, all 16 words match; 1 launches, 0.0100 s, 1.000 TMAC/s"; echo PASS; exit 0; fi',
    'echo "$6  FAIL: 99 of 99 words differ; first: region 0 word 0, got 1 want 2"; echo FAIL; exit 1',
    'FT',
    'chmod +x build/pearl_amd_foldtest',
    'echo "${PEARL_AMD_CFLAGS:-}" > build/pearl_core_hip.node',
    'echo "build ${PEARL_AMD_CFLAGS:-}" >> builds.log',
    '',
  ].join('\n');
  const HASHRATE = [
    "const f = require('fs').readFileSync(process.argv[2], 'utf8');",
    "console.log(JSON.stringify({ th: /DROP/.test(f) ? 101.5 : /V2=1/.test(f) ? 101 : /V2=2/.test(f) ? 100.5 : 100 }));",
    '',
  ].join('\n');
  // The Salad race: both forms against today's readout and the probe.
  const RACE = 'base=;v2=-DPEARL_AMD_READOUT_V2=1;v2w=-DPEARL_AMD_READOUT_V2=2;drop=-DPEARL_AMD_READOUT_DROP=1';

  let tmp;
  beforeEach(() => { tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'amd-sweep-')); });
  afterEach(() => { fs.rmSync(tmp, { recursive: true, force: true }); });

  // Runs a copy of sweep.sh with the stand-ins, writing nothing outside tmp: the container-log
  // copy of each line goes to /dev/null.
  function sweep(env) {
    const amd = path.join(tmp, 'amd');
    fs.mkdirSync(path.join(amd, 'test'), { recursive: true });
    fs.mkdirSync(path.join(tmp, 'probes'));
    fs.mkdirSync(path.join(tmp, 'root'));
    fs.writeFileSync(path.join(amd, 'build-amd.sh'), STUB, { mode: 0o755 });
    fs.writeFileSync(path.join(tmp, 'probes', 'hashrate.js'), HASHRATE);
    fs.writeFileSync(path.join(amd, 'test', 'sweep.sh'), SWEEP.replace('/proc/1/fd/1', '/dev/null'));
    const r = spawnSync('bash', ['test/sweep.sh'], {
      cwd: amd, encoding: 'utf8',
      env: Object.assign({}, process.env, { ARCH: 'gfx1201', SWEEP_ROOT: path.join(tmp, 'root').replace(/\\/g, '/') }, env),
    });
    const builds = fs.existsSync(path.join(amd, 'builds.log'))
      ? fs.readFileSync(path.join(amd, 'builds.log'), 'utf8').split('\n').filter(Boolean).map((l) => l.trimEnd()) : [];
    return {
      status: r.status,
      lines: r.stdout.split('\n').filter(Boolean).map((l) => l.replace(/^\[amdtest\] \S+ sweep /, '')),
      builds,
    };
  }

  bashIt('runs the race: base, both forms and the probe checked at 2048 x 4096, then fold and hashrate rounds in turn', () => {
    const { status, lines, builds } = sweep({
      SWEEP: RACE, SWEEP_TIMEONLY: 'drop', SWEEP_CHECK: '2048:4096', SWEEP_FOLD: '131072:4096:100',
      SWEEP_FOLD_ROUNDS: '4', SWEEP_ROUNDS: '4', SWEEP_HASH: '60:20',
    });
    expect(status).toBe(0);
    expect(builds).toEqual(['build', 'build -DPEARL_AMD_READOUT_V2=1', 'build -DPEARL_AMD_READOUT_V2=2',
      'build -DPEARL_AMD_READOUT_DROP=1', 'build']);
    expect(lines[0]).toContain('(timing only: drop); check 2048 x 4096; fold timing 131072:4096:100 x 4 rounds in turn; '
      + 'hashrate 60s+20s x 4');
    for (const n of ['base', 'v2', 'v2w'])
      expect(lines.find((l) => l.startsWith(`${n} built`))).toContain('check: wmma12  ok: 2048 x 4096, all 16 words match');
    expect(lines).toContain('drop check FAIL (expected: timing only)');
    expect(lines.filter((l) => l.startsWith('fold 131072 x 4096, round'))).toEqual([1, 2, 3, 4].map((i) =>
      `fold 131072 x 4096, round ${i} of 4, TMAC/s: base 50.000, v2 50.500 (+1.00%), v2w 50.250 (+0.50%), drop 50.750 (+1.50%)`));
    expect(lines).toContain('v2w fold 131072 x 4096, round 3 of 4: wmma12 timed only: 131072 x 4096, 100 launches, 20.0 s, 50.250 TMAC/s');
    const hash = lines.filter((l) => / hashrate: /.test(l));
    expect(hash).toHaveLength(16);
    expect(hash.slice(0, 4)).toEqual(['base hashrate: {"th":100}', 'v2 hashrate: {"th":101}', 'v2w hashrate: {"th":100.5}',
      'drop hashrate: {"th":101.5}']);
    expect(lines[lines.length - 1]).toBe('done');
  }, 30000);

  bashIt('checks at 1024 x 1024 when SWEEP_CHECK isn\'t set, as before', () => {
    const { status, lines } = sweep({ SWEEP: 'base=', SWEEP_FOLD: '', SWEEP_ROUNDS: '0' });
    expect(status).toBe(0);
    expect(lines[0]).toContain('; check 1024 x 1024; fold timing none; hashrate 40s+10s x 0');
    expect(lines.find((l) => l.startsWith('base built'))).toContain('ok: 1024 x 1024');
  }, 30000);

  bashIt('refuses a SWEEP_CHECK or SWEEP_FOLD_ROUNDS it can\'t read', () => {
    const lines = SWEEP.split('\n');
    const at = lines.findIndex((l) => l.startsWith('FOLD_ROUNDS='));
    const parse = lines.slice(at, lines.findIndex((l) => l.startsWith('IFS=: read -r CM CC')) + 1).join('\n');
    const tryIt = (env) => spawnSync('bash', ['-c', 'out() { echo "$*"; }\n' + parse + '\necho "went-on $CM $CC"'],
      { env: Object.assign({ PATH: process.env.PATH }, env), encoding: 'utf8' });
    for (const [v, ok] of [['1024:4096', true], ['', true], ['1024', false], ['1024:', false], [':4096', false],
      ['1024:4096:1', false], ['a:b', false], ['1024 4096', false]]) {
      const r = tryIt(v === '' ? {} : { SWEEP_CHECK: v });
      expect([v, r.status, r.stdout.includes('went-on')]).toEqual([v, ok ? 0 : 2, ok]);
    }
    expect(tryIt({}).stdout).toContain('went-on 1024 1024');
    for (const [v, ok] of [['x', false], ['-1', false], ['1.5', false], ['0', true], ['4', true]]) {
      const r = tryIt({ SWEEP_FOLD_ROUNDS: v });
      expect([v, r.status]).toEqual([v, ok ? 0 : 2]);
    }
  }, 30000);
});

describe('the readout census', () => {
  // A stand-in for hip-clang's assembly of an RDNA fold: `copies` readouts, all but the last in the
  // slab loop (which holds the v_wmma, the fence's DS wait and the barrier).
  const acc = (i) => `v${i % 128}`;
  const chain = (dst, from) => {
    const out = [`v_xor3_b32 ${dst}, ${acc(from)}, ${acc(from + 1)}, ${acc(from + 2)}`];
    for (let i = 1; i < 16; i++) out.push(`v_xor3_b32 ${dst}, ${dst}, ${acc(from + 2 * i + 1)}, ${acc(from + 2 * i + 2)}`);
    return out;
  };
  const trees = () => [].concat(...[0, 1, 2, 3].map((w) => chain(`v${140 + w}`, 32 * w)));
  const READOUT = {
    base: () => [].concat(...[0, 1, 2, 3].map((w) => [
      ...chain(`v${140 + w}`, 32 * w),
      `v_xor_b32_dpp v${140 + w}, v${140 + w}, v${140 + w} quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf`,
      `v_xor_b32_dpp v${140 + w}, v${140 + w}, v${140 + w} row_ror:8 row_mask:0xf bank_mask:0xf`,
      `v_mov_b32_e32 v${150 + w}, v${140 + w}`,
      `v_permlanex16_b32 v${150 + w}, v${140 + w}, s4, s5`,
      `v_xor_b32_e32 v${140 + w}, v${140 + w}, v${150 + w}`,
    ]), 'v_cmp_eq_u32_e32 vcc_lo, s6, v160',
    ...[0, 1, 2, 3, 4, 5, 6, 7].map((i) => `v_cndmask_b32_e32 v${170 + i}, v${170 + i}, v${140 + (i % 4)}, vcc_lo`)),
    drop: () => [...trees(), ...[0, 1, 2, 3].map((w) => `v_xor_b32_e32 v${170 + w}, v${170 + w}, v${140 + w}`)],
    v2: ({ op = 'ds_xor_b64', extra = [] } = {}) => [...trees(), 'v_add_nc_u32_e32 v160, s8, v161',
      `${op} v160, v[140:141]`, `${op} v160, v[142:143] offset:1536`, ...extra],
    v2w: () => [...trees(), 'v_add_nc_u32_e32 v160, s8, v161', 'ds_xor_b32 v160, v140', 'ds_xor_b32 v160, v141 offset:4',
      'ds_xor_b32 v160, v142 offset:1536', 'ds_xor_b32 v160, v143 offset:1540'],
  };
  function asm(fold, readout, { copies = 3, slab = [] } = {}) {
    const gfx12 = fold.endsWith('12');
    const wait = gfx12 ? 's_wait_dscnt 0x0' : 's_waitcnt lgkmcnt(0)';
    const barrier = gfx12 ? ['s_barrier_signal -1', 's_barrier_wait -1'] : ['s_barrier'];
    return [
      `${fold}:                  ; @${fold}`,
      '; %bb.0:',
      '\ts_load_b128 s[0:3], s[0:1], 0x0',
      '\tds_store_b128 v200, v[0:3] offset:32',
      '.LBB4_1:                                ; =>This Inner Loop Header: Depth=1',
      '\tv_wmma_i32_16x16x16_iu8 v[0:7], v[200:201], v[202:203], v[0:7]',
      ...[].concat(...Array.from({ length: copies - 1 }, () => readout.map((l) => '\t' + l))),
      ...slab.map((l) => '\t' + l),
      '\t' + wait,
      ...barrier.map((l) => '\t' + l),
      '\ts_cbranch_scc1 .LBB4_1',
      '; %bb.2:',
      ...readout.map((l) => '\t' + l),
      '\tglobal_store_b128 v[220:221], v[0:3], off',
      '\ts_endpgm',
      '.Lfunc_end4:',
      'pearl_amd_fold_other:',
      '\tv_permlanex16_b32 v1, v1, s0, s1',
      '\tds_xor_rtn_b64 v[0:1], v2, v[4:5]',
      '.Lfunc_end5:',
    ].join('\n');
  }
  const counted = (fold, text) => census.count(census.parse(census.cutFunction(text, fold)));
  const RES12 = { vgprs: 198, scratch: 0, vgprSpill: 0, sgprSpill: 0, lds: 36864, occupancy: 6 };
  const RES11 = { ...RES12, vgprs: 240 };
  const F12 = 'pearl_amd_fold_wmma12', F11 = 'pearl_amd_fold_wmma11';
  const good12 = () => ({
    base: counted(F12, asm(F12, READOUT.base())), v2: counted(F12, asm(F12, READOUT.v2())),
    v2w: counted(F12, asm(F12, READOUT.v2w())), drop: counted(F12, asm(F12, READOUT.drop())),
  });
  const res12 = { base: RES12, v2: { ...RES12, vgprs: 190 }, v2w: { ...RES12, vgprs: 190 }, drop: RES12 };
  const misses = (c, r = res12, target = 'gfx1201') =>
    census.gates(c, r, target).filter((x) => x[2] === false).map((x) => x[0]);
  const ATOMICS = 'ds_xor_rtn / ds_cmp* / flat or global atomics (0)';
  const XOR64 = 'ds_xor_b64 (v2: 2N; others 0)';
  const XOR32 = 'ds_xor_b32 (v2w: 4N; others 0)';
  const SLABXOR = '  of them in the slab loop (v2: 2NL, v2w: 4NL)';

  it('picks the fold and the builds by target', () => {
    expect([census.foldOf('gfx1200'), census.foldOf('gfx1201'), census.foldOf('gfx1100')]).toEqual([F12, F12, F11]);
    expect(census.buildsOf('gfx1201')).toEqual(['base', 'v2', 'v2w', 'drop']);
    expect(census.buildsOf('gfx1100')).toEqual(['base', 'v2', 'v2w']);
  });

  it('counts today\'s readout, and v2\'s XORs, waits and saved VALU', () => {
    const c = good12();
    expect([c.base.permlane, c.base.slabPermlane, c.base.dpp]).toEqual([12, 8, 24]);
    expect(c.base.slabSelects).toBe(2 * 9);
    expect(c.base.readout).toBe(3 * 93);
    expect([c.v2.xor64, c.v2.slabXor, c.v2.xor32, c.v2w.xor32, c.v2w.slabXor]).toEqual([6, 4, 0, 12, 8]);
    expect([c.v2.slabDsWait, c.base.slabDsWait]).toEqual([1, 1]);
    // Base's slab loop: the v_wmma and two readouts of 93; v2's: two of 64 + 1.
    expect([c.base.slabValu, c.v2.slabValu]).toEqual([1 + 2 * 93, 1 + 2 * 65]);
    // The pair is formed where the tree leaves it.
    expect(c.v2.dataMov).toBe(0);
    expect(misses(c)).toEqual([]);
    const rows = census.gates(c, res12, 'gfx1201');
    // v2 saves 28 a copy here; the probe 25, since it still XORs each word into keepLo.
    expect(rows.find((x) => x[0] === '  saved a copy')[1]).toEqual(['', '28.0', '28.0', '25.0']);
  });

  it('passes gfx11 with its own fold, waits and register limit', () => {
    const c = {
      base: counted(F11, asm(F11, READOUT.base())), v2: counted(F11, asm(F11, READOUT.v2())),
      v2w: counted(F11, asm(F11, READOUT.v2w())),
    };
    expect(c.v2.slabDsWait).toBe(1);
    const r = { base: RES11, v2: { ...RES11, vgprs: 232 }, v2w: { ...RES11, vgprs: 232 } };
    expect(misses(c, r, 'gfx1100')).toEqual([]);
    expect(misses(c, { ...r, v2: { ...RES11, vgprs: 241 } }, 'gfx1100'))
      .toEqual(['VGPRs (<= 240)', 'VGPRs, v2 against base (no more)']);
    const waits = counted(F11, asm(F11, READOUT.v2({ extra: ['s_waitcnt lgkmcnt(0)'] })));
    expect(misses({ ...c, v2: waits }, r, 'gfx1100')).toEqual(['slab-loop DS waits (v2: no more than base)']);
  });

  it('misses a returning XOR, a flat atomic and a compare-and-swap loop', () => {
    const c = good12();
    expect(misses({ ...c, v2: counted(F12, asm(F12, READOUT.v2({ op: 'ds_xor_rtn_b64' }))) }))
      .toEqual([XOR64, SLABXOR, ATOMICS]);
    expect(misses({ ...c, v2: counted(F12, asm(F12, READOUT.v2({ op: 'flat_atomic_xor_b64' }))) }))
      .toEqual([XOR64, SLABXOR, ATOMICS]);
    const cas = ['ds_cmpstore_rtn_b64 v[0:1], v160, v[2:3], v[4:5]'];
    expect(misses({ ...c, v2w: counted(F12, asm(F12, [...READOUT.v2w(), ...cas])) })).toEqual([ATOMICS]);
  });

  it('misses a pair split in two, a lane move or select left in, a new DS wait, and too many registers', () => {
    const c = good12();
    expect(misses({ ...c, v2: counted(F12, asm(F12, READOUT.v2w())) })).toEqual([XOR64, XOR32, SLABXOR]);
    const dpp = 'v_mov_b32_dpp v150, v140 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf';
    expect(misses({ ...c, v2: counted(F12, asm(F12, READOUT.v2({ extra: [dpp] }))) })).toEqual(['DPP ops (v2: 0)']);
    const selects = Array.from({ length: 8 }, (_, i) => `v_cndmask_b32_e32 v${170 + i}, v${170 + i}, v140, vcc_lo`);
    expect(misses({ ...c, v2: counted(F12, asm(F12, READOUT.v2({ extra: selects }))) }))
      .toEqual(['slab-loop selects: v_cndmask, v_cmp, v_movrel (v2: <= base - 8 a copy)']);
    const valu = Array.from({ length: 10 }, (_, i) => `v_add_nc_u32_e32 v${200 + i}, s9, v${200 + i}`);
    expect(misses({ ...c, v2: counted(F12, asm(F12, READOUT.v2({ extra: valu }))) }))
      .toEqual(['slab-loop VALU (v2: <= base - 20 a copy)']);
    expect(misses({ ...c, v2: counted(F12, asm(F12, READOUT.v2({ extra: ['s_wait_dscnt 0x0'] }))) }))
      .toEqual(['slab-loop DS waits (v2: no more than base)']);
    expect(misses(c, { ...res12, v2: { ...RES12, vgprs: 200 } })).toEqual(['VGPRs, v2 against base (no more)']);
    expect(misses(c, { ...res12, v2w: { ...RES12, vgprs: 224, occupancy: 5 } }))
      .toEqual(['VGPRs (<= 216; target 192)', 'VGPRs, v2 against base (no more)', 'occupancy, waves a SIMD (v2: no fewer)']);
    expect(misses(c, { ...res12, v2: { ...RES12, lds: 45056 } })).toEqual(['LDS bytes (as base)']);
    expect(misses(c, { ...res12, drop: { ...RES12, scratch: 8 } })).toEqual(['scratch bytes / spills (0)']);
  });

  it('records copies made to form the pair', () => {
    const c = good12();
    const moved = counted(F12, asm(F12, [...trees(), 'v_mov_b32_e32 v181, v141', 'v_mov_b32_e32 v180, v140',
      'ds_xor_b64 v160, v[180:181]', 'ds_xor_b64 v160, v[142:143] offset:1536']));
    expect(moved.dataMov).toBe(6);
    const rows = census.gates({ ...c, v2: moved }, res12, 'gfx1201');
    expect(rows.find((x) => x[0].includes('feeding their data'))[1]).toEqual(['', '2.0', '0.0', '']);
  });

  it('misses a build whose slab loop or copies it can\'t find', () => {
    const c = good12();
    const flat = counted(F12, asm(F12, READOUT.v2()).replace('v_wmma_i32_16x16x16_iu8', 's_nop'));
    expect(misses({ ...c, v2: flat })).toEqual(expect.arrayContaining(['slab loop (the loop with the v_wmma) found']));
    const lost = counted(F12, asm(F12, READOUT.drop()));
    expect(misses({ ...c, base: lost })).toEqual(expect.arrayContaining(['copies, from base: function (N) / slab loop']));
  });

  it('reads the resource remarks build-amd.sh check reads', () => {
    const t = '-Rpass-analysis=kernel-resource-usage';
    const remarks = [F11, F12].map((f, i) => [
      `x.cpp:1:1: remark: Function Name: ${f} [${t}]`,
      `x.cpp:1:1: remark:     VGPRs: ${190 + i} [${t}]`,
      `x.cpp:1:1: remark:     ScratchSize [bytes/lane]: ${i ? 0 : 8} [${t}]`,
      `x.cpp:1:1: remark:     Occupancy [waves/SIMD]: 6 [${t}]`,
      `x.cpp:1:1: remark:     SGPRs Spill: 0 [${t}]`,
      `x.cpp:1:1: remark:     VGPRs Spill: 0 [${t}]`,
      `x.cpp:1:1: remark:     LDS Size [bytes/block]: 36864 [${t}]`,
    ].join('\n')).join('\n');
    expect(census.resources(remarks, F12))
      .toEqual({ vgprs: 191, scratch: 0, occupancy: 6, sgprSpill: 0, vgprSpill: 0, lds: 36864 });
  });

  it('prints one column a build', () => {
    const c = good12();
    const text = census.table('gfx1201', census.buildsOf('gfx1201'), census.gates(c, res12, 'gfx1201'));
    expect(text.split('\n')[0]).toBe('gfx1201: pearl_amd_fold_wmma12');
    expect(text.split('\n')[1].trim().split(/\s+/)).toEqual(['base', 'v2', 'v2w', 'drop']);
    expect(text).not.toContain('MISS');
  });

  it('compiles the four builds with the readout switches its own, and CI runs it after the upload', () => {
    expect(CENSUS_SH).toContain('targets="${*:-gfx1100 gfx1200 gfx1201}"');
    expect(CENSUS_SH).toContain('case $a in gfx12*) builds="$builds drop" ;; esac');
    for (const s of ['v2) flags="-DPEARL_AMD_READOUT_V2=1" ;;', 'v2w) flags="-DPEARL_AMD_READOUT_V2=2" ;;',
      'drop) flags="-DPEARL_AMD_READOUT_DROP=1" ;;'])
      expect(CENSUS_SH).toContain(s);
    expect(CENSUS_SH).toContain('${PEARL_AMD_CFLAGS:-} \\\n        -UPEARL_AMD_READOUT_DROP -UPEARL_AMD_READOUT_V2 $flags $inc');
    expect(CENSUS_SH).toContain('--cuda-device-only -S');
    const job = between(CI, '\n  amd:\n', '\n  amd-windows:\n');
    const step = job.slice(job.indexOf('- name: Readout v2 on RDNA 3 and 4'));
    expect(job.indexOf('- name: Readout v2 on RDNA 3 and 4')).toBeGreaterThan(job.indexOf('uses: actions/upload-artifact@v4'));
    expect(step).toContain('continue-on-error: true');
    expect(step).toMatch(/timeout-minutes: \d+/);
    expect(step).toContain('PEARL_AMD_CFLAGS=-DPEARL_AMD_READOUT_V2=1 ./build-amd.sh check gfx1100 gfx1200 gfx1201 || rc=1');
    expect(step).toContain('bash test/readout-census.sh gfx1100 gfx1200 gfx1201 || rc=1');
    // It is the one step of its kind: the job builds nothing else with v2.
    expect(job.match(/READOUT_V2/g)).toHaveLength(2);
  });
});
