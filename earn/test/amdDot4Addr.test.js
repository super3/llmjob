'use strict';

// PEARL_AMD_DOT4_ADDR and PEARL_AMD_DOT4_SWZ (earn/native/amd/src/pearl_amd_fold.h): RDNA 2's slab loop stages each
// slab with buffer loads at offsets fixed per thread, into LDS laid out [A' 0][B' 0][A' 1][B' 1], with A' rows
// swizzled and each staging store's lanes on rows 4 apart. No HIP compiler runs here, so the C++ is read as text and
// its arithmetic run as JS: every 16-byte unit of a slab is loaded once and stored where the fold's reads look for it,
// and the bank-conflict claims in the comments hold.

const fs = require('fs');
const path = require('path');

const AMD = path.join(__dirname, '..', 'native', 'amd');
const FOLD = fs.readFileSync(path.join(AMD, 'src', 'pearl_amd_fold.h'), 'utf8').replace(/\r\n/g, '\n');
const BUILD = fs.readFileSync(path.join(AMD, 'build-amd.sh'), 'utf8').replace(/\r\n/g, '\n');

function slice(text, from, to) {
  const at = text.indexOf(from);
  if (at < 0) throw new Error('not found: ' + from);
  return text.slice(at, text.indexOf(to, at + from.length) + to.length);
}
function defaultOf(name) {
  const m = new RegExp(`#ifndef ${name}\\n#define ${name} (.*)\\n#endif`).exec(FOLD);
  if (!m) throw new Error('no default for ' + name);
  return m[1];
}
const num = (re) => Number(re.exec(FOLD)[1]);

// The staging map and lane bases (before the first slab's staging), and the slab body (in the one-slab loop).
const setup = slice(FOLD, '#if PEARL_AMD_DOT4_ADDR_ON\n  // PEARL_AMD_DOT4_ADDR (above), RDNA 2\'s slab loop',
  '#define PEARL_AMD_STORE(buf) PEARL_AMD_STORER(buf, pre)\n#endif\n');
const body = slice(FOLD, '#if PEARL_AMD_DOT4_ADDR_ON\n    {   // PEARL_AMD_DOT4_ADDR (above): RDNA 2\'s slab',
  '#endif  // PEARL_AMD_DOT4_ADDR_ON\n');

// The block's shape and the dot4 fold's slab, as the source has them.
const BM = num(/#define PEARL_AMD_BM (\d+)u/), BN = num(/#define PEARL_AMD_BN (\d+)u/);
const T = num(/#define PEARL_AMD_THREADS (\d+)u/), KS = num(/#define PEARL_AMD_DOT4_SLAB (\d+)u/);
const S = KS + 16, QR = KS / 16, UA = BM * QR, UB = BN * QR, U = (UA + UB) / T;
const WM = 64, WN = 64, KTM = 32;   // PearlAmdDot4: a wave holds 2 tiles of 32x64 (kWR 2, kWC 4)

// The loop's arithmetic, written as the C++ writes it (each line is checked against the source below).
const swzOf = (on) => (row) => (on ? row ^ (((row >> 3) & 3) << 1) : row);
function layout(on) {
  const swz = swzOf(on);
  const units = [];
  for (let t = 0; t < T; t++) {
    const tl = t & 31, c16 = t & 3;
    const prow0 = 8 * (t >> 5) + (on ? ((tl >> 2) & 1) * 4 + (tl >> 3) : (tl >> 2));
    const lst = prow0 * S + 16 * c16;
    for (let u = 0; u < U; u++) {
      const isA = u * T < UA;
      const prow = prow0 + (isA ? u * T / QR : (u * T - UA) / QR);
      const vo = 16 * (QR * (isA ? swz(prow) : prow) + c16);
      const lds = lst + (isA ? (u * T / QR) * S : BM * S + ((u * T - UA) / QR) * S);
      units.push({ t, u, isA, vo, lds });
    }
  }
  // Lane bases: A' row of each tile, B' rows j = 0-3 and 4-7.
  const bases = (wave, lane) => {
    const wr = Math.floor(wave / 4), wc = wave % 4;
    const la0 = (wr * WM + swz(4 * (lane & 1) + ((lane >> 3) & 3) + 8 * ((lane & 7) >> 1))) * S;
    const lb0 = BM * S + wc * WN * S + (lane & 7) * S;
    return { wr, wc, la0, la1: la0 + KTM * S, lb0, lb4: lb0 + 32 * S };
  };
  return { swz, units, bases };
}
const bRow = (b, j) => (j < 4 ? b.lb0 + 8 * j * S : b.lb4 + 8 * (j - 4) * S);

// LDS banks a read or write of `bytes` at byte address addr covers (32 banks of 4 bytes).
const banks = (addr, bytes) => Array.from({ length: bytes / 4 }, (_, i) => ((addr >> 2) + i) % 32);
// The most distinct addresses that share one bank among a group of accesses (same addresses broadcast).
function worstBank(addrs, bytes) {
  const per = new Map();
  for (const a of new Set(addrs)) for (const b of banks(a, bytes)) per.set(b, (per.get(b) || 0) + 1);
  return Math.max(...per.values());
}

describe('RDNA 2\'s slab loop (PEARL_AMD_DOT4_ADDR, PEARL_AMD_DOT4_SWZ)', () => {
  it('is on by default for the DPPA k-step\'s 8-byte reads, the swizzle with it, and out of the Windows core', () => {
    expect(defaultOf('PEARL_AMD_DOT4_ADDR')).toBe('1');
    expect(defaultOf('PEARL_AMD_DOT4_SWZ')).toBe('1');
    expect(FOLD).toContain('#if PEARL_AMD_DOT4_ADDR != 0 && PEARL_AMD_DOT4_ADDR != 1\n#error');
    expect(FOLD).toContain('#if PEARL_AMD_DOT4_SWZ != 0 && PEARL_AMD_DOT4_SWZ != 1\n#error');
    expect(FOLD).toContain('#if PEARL_AMD_DOT4_ADDR && PEARL_AMD_DOT4_DPPA_ON && PEARL_AMD_DOT4_DPPA_KB == 8 && !PEARL_AMD_PF2');
    expect(BUILD).toContain('if windows; then SET_OS="$SET_OS -DPEARL_AMD_DOT4_ADDR=0"; fi\n');
  });

  it('stages every slab with its map, and runs its slab body in the one-slab loop', () => {
    // Its map is the staging macros' in that pass, so the first slab's staging and the one-slab loop's load use it.
    expect(setup).toContain('#define PEARL_AMD_STORE(buf) PEARL_AMD_STORE_AT((uint32_t)(buf) * kFlip)\n#else\n'
      + '#define PEARL_AMD_LOAD(slab) PEARL_AMD_LOADR(slab, pre)\n');
    expect(FOLD).toContain('  PEARL_AMD_LOAD(0)\n  PEARL_AMD_STORE(0)\n  __syncthreads();\n');
    expect(FOLD).toContain('    if (s + 1 < slabs) { PEARL_AMD_LOAD(s + 1) }\n#if PEARL_AMD_DOT4_ADDR_ON\n'
      + '    {   // PEARL_AMD_DOT4_ADDR');
    // The body ends by storing the next slab, reading out a finished chunk, the barrier and the flip, and skips the
    // rest of the loop's body.
    expect(body).toContain([
      '        if (s + 1 < slabs) { PEARL_AMD_STORE_AT(kFlip - par) }',
      '      }',
      '      if ((s + 1) % perChunk == 0) readout((s + 1) / perChunk - 1);',
      '      __syncthreads();',
      '      par = kFlip - par;',
      '      continue;',
      '    }',
      '#endif  // PEARL_AMD_DOT4_ADDR_ON'].join('\n'));
    expect(setup).toContain('uint32_t par = 0u;');
  });

  it('computes its places as this test does', () => {
    const code = setup + body;
    for (const line of [
      'auto swz = [](int row) { return kSwz ? row ^ (((row >> 3) & 3) << 1) : row; };',
      'const int tl = t & 31, c16 = t & 3;',
      'const int prow0 = 8 * (t >> 5) + (kSwz ? ((tl >> 2) & 1) * 4 + (tl >> 3) : (tl >> 2));',
      'const int prow = prow0 + (isA ? u * T / QR : (u * T - UA) / QR);',
      'vo[u] = (uint32_t)(16 * (QR * (isA ? swz(prow) : prow) + c16));',
      'uint32_t la0 = (uint32_t)((wr * WM + swz(4 * (lane & 1) + ((lane >> 3) & 3) + 8 * ((lane & 7) >> 1))) * S);',
      'uint32_t la1 = la0 + (uint32_t)(P::kTM * S);',
      'uint32_t lb0 = (uint32_t)(BM * S + wc * WN * S + (lane & 7) * S);',
      'uint32_t lb4 = lb0 + (uint32_t)(32 * S);',
      'uint32_t lst = (uint32_t)(prow0 * S + 16 * c16);',
      'int8_t *const dst = sL + (lst + (po));',
      '*reinterpret_cast<pearl_amd_v4i *>(dst + (u * T < UA ? (u * T / QR) * S : BM * S + ((u * T - UA) / QR) * S)) =',
      'u * T < UA ? __builtin_amdgcn_raw_buffer_load_b128(rsA, (int)vo[u], (int)((slab) * (uint32_t)(BM * KS)), 0)',
      ': __builtin_amdgcn_raw_buffer_load_b128(rsB, (int)vo[u], (int)((slab) * (uint32_t)(BN * KS)), 0));',
      'const int8_t *const aR0 = sL + (la0 + par);',
      'const int8_t *const aR1 = sL + (la1 + par);',
      'const int8_t *const bR0 = sL + (lb0 + par);',
      'const int8_t *const bR4 = sL + (lb4 + par);',
      'b2[j] = *reinterpret_cast<const pearl_amd_v2i *>((j < 4 ? bR0 + 8 * j * S : bR4 + 8 * (j - 4) * S) + 8 * kp);',
      'constexpr uint32_t kFlip = (uint32_t)((BM + BN) * S);',
    ]) expect([line, code.includes(line)]).toEqual([line, true]);
    // Both buffers fit a DS instruction's 16-bit offset, and two blocks fit a WGP's 128 KB.
    expect(2 * (BM + BN) * S).toBeLessThanOrEqual(65536);
    expect(2 * 2 * (BM + BN) * S).toBeLessThanOrEqual(131072);
  });

  for (const on of [true, false]) {
    describe(on ? 'with the swizzle' : 'without it (PEARL_AMD_DOT4_SWZ=0)', () => {
      const { swz, units, bases } = layout(on);

      it('loads every 16-byte unit of a slab once, inside its tile, and stores it where the reads look for it', () => {
        expect(units.filter((x) => x.isA)).toHaveLength(UA);
        const seen = { A: new Set(), B: new Set() };
        for (const x of units) {
          const tile = x.isA ? 'A' : 'B';
          expect(x.vo % 16).toBe(0);
          expect(x.vo >= 0 && x.vo + 16 <= (x.isA ? BM : BN) * KS).toBe(true);
          expect(seen[tile].has(x.vo)).toBe(false);
          seen[tile].add(x.vo);
          // The unit holds tile row vo / KS, 16-byte column (vo % KS) / 16, which belongs at that row's LDS place.
          const row = Math.floor(x.vo / KS), col = (x.vo % KS) / 16;
          const want = x.isA ? swz(row) * S + 16 * col : BM * S + row * S + 16 * col;
          if (x.lds !== want) throw new Error(`thread ${x.t} unit ${x.u}: LDS ${x.lds}, want ${want}`);
        }
        expect([seen.A.size, seen.B.size]).toEqual([UA, UB]);
      });

      it('reads each lane\'s A\' row of each tile and its eight B\' rows from those places', () => {
        for (let wave = 0; wave < T / 32; wave++) {
          for (let lane = 0; lane < 32; lane++) {
            const b = bases(wave, lane);
            // PearlAmdDot4's rows: lane rho = lane & 7 of a dpp8 group reads row 4 (rho & 1) + i + 8 (rho >> 1).
            const rho = lane & 7, i = (lane >> 3) & 3;
            for (let mt = 0; mt < 2; mt++) {
              const row = b.wr * WM + mt * KTM + 4 * (rho & 1) + i + 8 * (rho >> 1);
              expect([wave, lane, mt, (mt ? b.la1 : b.la0)]).toEqual([wave, lane, mt, swz(row) * S]);
            }
            for (let j = 0; j < 8; j++) expect(bRow(b, j)).toBe(BM * S + (b.wc * WN + (lane & 7) + 8 * j) * S);
          }
        }
      });

      it(on ? 'stores with no bank conflict and reads A\' 2 rows a bank, B\' none'
        : 'stores 2 rows a bank and reads A\' 4 rows a bank (what the swizzle fixed)', () => {
        // A 16-byte store goes 8 lanes at a time.
        let store = 0;
        for (let u = 0; u < U; u++) {
          for (let t0 = 0; t0 < T; t0 += 8) {
            const g = units.filter((x) => x.u === u && x.t >= t0 && x.t < t0 + 8).map((x) => x.lds);
            store = Math.max(store, worstBank(g, 16));
          }
        }
        // An 8-byte read goes 16 lanes at a time.
        let readA = 0, readB = 0;
        for (let wave = 0; wave < T / 32; wave++) {
          for (let half = 0; half < 2; half++) {
            const lanes = Array.from({ length: 16 }, (_, n) => 16 * half + n);
            for (let kp = 0; kp < KS / 8; kp++) {
              for (const mt of [0, 1]) {
                const at = lanes.map((l) => (mt ? bases(wave, l).la1 : bases(wave, l).la0) + 8 * kp);
                readA = Math.max(readA, worstBank(at, 8));
              }
              for (let j = 0; j < 8; j++) readB = Math.max(readB, worstBank(lanes.map((l) => bRow(bases(wave, l), j) + 8 * kp), 8));
            }
          }
        }
        expect({ store, readA, readB }).toEqual(on ? { store: 1, readA: 2, readB: 1 } : { store: 2, readA: 4, readB: 1 });
      });
    });
  }
});
