'use strict';

// PEARL_AMD_LDS32 (earn/native/amd/src/pearl_amd_fold.h): RDNA 4's fold with 32-byte LDS rows and no padding, the
// row's two 16-byte halves swapped on row bits PEARL_AMD_LDS32_SWZ. That is right only if every fragment read finds
// the half the staging stored for its row, and fast only if the reads and stores keep off each other's banks. No
// HIP compiler runs here, so the C++ is read as text and its layout run as JS.

const fs = require('fs');
const path = require('path');
const { preprocess } = require('./helpers/amdSwitches');

const FOLD = fs.readFileSync(path.join(__dirname, '..', 'native', 'amd', 'src', 'pearl_amd_fold.h'), 'utf8')
  .replace(/\r\n/g, '\n');

function grab(text, re) {
  const m = re.exec(text);
  if (!m) throw new Error('not found: ' + re);
  return m[1];
}

const SWZ = Number(grab(FOLD, /#define PEARL_AMD_LDS32_SWZ (\d+)\n/));
const swz = (row, mask = SWZ) => {
  let x = row & mask;
  let p = 0;
  while (x) { p ^= x & 1; x >>= 1; }
  return p;
};
// The fold's tile: 128 rows of A' and 256 of B', 32-byte slabs (16-byte units: QR 2), 256 threads, 3 units a thread.
const S = 32, QR = 2, T = 256, UA = 128 * QR, U = (128 * QR + 256 * QR) / T;
const A0 = 0, B0 = 2 * 128 * S;   // sA[0] and sB[0]; buffer 1 of each is one tile further

describe('32-byte LDS rows (PEARL_AMD_LDS32)', () => {
  it('is on by default in RDNA 4 code with its pin, swaps on row bits 2 and 3, and is in the paired-read RDNA 4 fold only', () => {
    // On in the RDNA 4 set (PEARL_AMD_RDNA4_SET, on in gfx12 device passes only: amdRdna4Set.test.js), pinned.
    expect(FOLD).toContain('#ifndef PEARL_AMD_LDS32\n#define PEARL_AMD_LDS32 PEARL_AMD_RDNA4_SET\n#endif');
    expect(FOLD).toContain('#ifndef PEARL_AMD_LDS32_PIN\n#define PEARL_AMD_LDS32_PIN PEARL_AMD_RDNA4_SET\n#endif');
    expect(SWZ).toBe(12);
    // The RDNA 4 fold in either wave size: wave32 (PearlAmdWmma12) or wave64 (PearlAmdWmma12W64, PEARL_AMD_W64). Its
    // slab is 32 bytes, or 64 in the wave64 fold with PEARL_AMD_W64K (amdW64k.test.js).
    expect(FOLD).toContain('  constexpr int KS = std::is_same<P, PearlAmdDot4>::value ? (int)PEARL_AMD_DOT4_SLAB\n'
      + '                     : std::is_same<P, PearlAmdWmma12W64>::value && PEARL_AMD_W64K ? 64 : (int)PEARL_AMD_SLAB;\n'
      + '  constexpr bool kL32 = PEARL_AMD_KPAIR && (std::is_same<P, PearlAmdWmma12>::value || std::is_same<P, PearlAmdWmma12W64>::value);\n'
      + '  static_assert(!kL32 || KS == 32 || (KS == 64 && std::is_same<P, PearlAmdWmma12W64>::value),\n'
      + '                "PEARL_AMD_LDS32 is for 32-byte slabs (64 in the wave64 fold: W64K)");\n'
      + '  constexpr int S = kL32 ? KS : KS + 16;\n');
    // Every other target's text, and RDNA 4's with the set off, is the 48-byte rows'.
    const rows48 = preprocess(FOLD, { PEARL_AMD_LDS32: 0, PEARL_AMD_LDS32_PIN: 0 });
    expect(rows48).toContain(
      'constexpr int KS = std::is_same<P, PearlAmdDot4>::value ? (int)PEARL_AMD_DOT4_SLAB : (int)PEARL_AMD_SLAB, S = KS + 16;');
    expect(rows48).toContain('const int off = (lane & 15) * S + 16 * (lane >> 4) + kk * P::kTK;');
    expect(rows48).not.toContain('l32');
  });

  it('reads every half of every row where the staging stored it, pinned or not', () => {
    for (const pin of [0, 1]) {
      const src = preprocess(FOLD, { PEARL_AMD_LDS32: 1, PEARL_AMD_LDS32_PIN: pin });
      if (pin) {
        expect(src).toContain('int l32st = (t / QR) * S + 16 * ((t % QR) ^ l32sw);');
        // The wave32 fold's offset is the last arm; the wave64 arms are amdW64k.test.js's.
        expect(src).toContain(
          '  int l32rd = kW64 && KS == 64 ? (lane & 15) * S + 16 * ((lane >> 4) ^ (((lane & 15) >> 2) & 3))   // W64K (above)\n'
          + '              : kW64 ? (lane & 15) * S + 8 * ((lane >> 4) ^ (kL32 ? 2 * pearl_amd_l32_swz(lane & 15) : 0))\n'
          + '                   : (lane & 15) * S + 16 * ((lane >> 4) ^ (kL32 ? pearl_amd_l32_swz(lane & 15) : 0));\n');
        expect(src).toContain('const int off = l32rd + kk * P::kTK;');
      } else {
        expect(src).toContain('*reinterpret_cast<pearl_amd_v4i *>(dst + 16 * ((idx % QR) ^ l32sw)) = R[u];');
        expect(src).toContain(
          'const int off = (lane & 15) * S + 16 * ((lane >> 4) ^ (kL32 ? pearl_amd_l32_swz(lane & 15) : 0)) + kk * P::kTK;');
      }
      // At the 32-byte slab simulated here the swap is pearl_amd_l32_swz's; a 64-byte slab turns quarters.
      expect(src).toContain(
        'const int l32sw = kL32 ? (KS == 64 ? ((t / QR) >> 2) & 3 : pearl_amd_l32_swz(t / QR)) : 0;   // W64K: quarters\n');
    }
    // The staging: thread t's unit u is row idx / QR (of A' below UA, else of B'), half idx % QR, at the half
    // swapped by t / QR's bits, which are the row's: rows past t / QR are whole 128 further.
    const stored = new Map();
    for (let t = 0; t < T; t++) {
      for (let u = 0; u < U; u++) {
        const idx = t + u * T;
        const isA = idx < UA;
        const row = isA ? idx / QR | 0 : (idx - UA) / QR | 0;
        expect(swz(row)).toBe(swz(t / QR | 0));
        const at = (isA ? A0 : B0) + row * S + 16 * ((idx % QR) ^ swz(t / QR | 0));
        expect(stored.has(at)).toBe(false);
        stored.set(at, `${isA ? 'A' : 'B'} ${row} ${idx % QR}`);
      }
    }
    expect(stored.size).toBe((128 + 256) * QR);
    // The reads: wave (wr, wc), tile mt (nt), lane l: row 64 wr + 16 mt + l % 16 of A' (64 wc + 16 nt + ... of B'),
    // half l / 16, which PEARL_AMD_KPAIR gives lanes 16-31.
    for (let wave = 0; wave < 8; wave++) {
      const wr = wave >> 2, wc = wave & 3;
      for (let tile = 0; tile < 4; tile++) {
        for (let l = 0; l < 32; l++) {
          const off = (l & 15) * S + 16 * ((l >> 4) ^ swz(l & 15));
          const a = A0 + wr * 64 * S + tile * 16 * S + off;
          const b = B0 + wc * 64 * S + tile * 16 * S + off;
          expect(stored.get(a)).toBe(`A ${wr * 64 + tile * 16 + (l & 15)} ${l >> 4}`);
          expect(stored.get(b)).toBe(`B ${wc * 64 + tile * 16 + (l & 15)} ${l >> 4}`);
        }
      }
    }
  });

  it('keeps a fragment read off its own banks: 8 lanes over 32 banks and 16 over 64', () => {
    // A 16-byte read is 4 banks a lane. Bits 2 and 3 together are needed: either alone conflicts in one of the two.
    const banks = (lanes, nb, mask) => lanes.flatMap((l) => {
      const dw = ((l & 15) * S + 16 * ((l >> 4) ^ swz(l & 15, mask))) / 4;
      return [0, 1, 2, 3].map((i) => (dw + i) % nb);
    });
    const free = (mask) => [0, 8, 16, 24].every((g) => {
      const b8 = banks([...Array(8).keys()].map((i) => g + i), 32, mask);
      return new Set(b8).size === b8.length;
    }) && [0, 16].every((g) => {
      const b16 = banks([...Array(16).keys()].map((i) => g + i), 64, mask);
      return new Set(b16).size === b16.length;
    });
    expect(free(SWZ)).toBe(true);
    expect(free(0)).toBe(false);
    expect(free(4)).toBe(false);
    expect(free(8)).toBe(false);
  });
});
