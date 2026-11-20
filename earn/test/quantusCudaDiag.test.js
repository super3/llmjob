'use strict';

// qc_dmul (native/quantus/cuda/qc_field.cuh), the partial rounds' product a * D + S mod p under QC_DIAG 1, modelled
// word for word. It writes a D as U + V 2^32 with U = a0 d0 + a1 u1 and V = a0 d1 + a1 v1, where u1 and v1 are
// 32-bit constants derived from D, adds S's halves in the products' addends, and folds the carries with
// 2^64 = 2^32 - 1 and 2^96 = -1 instead of a reduction multiply. The model follows its PTX: mul.wide.u32 and
// 64-bit adds whose carries out are cu and cv, then the combination in C's u32 and u64 arithmetic.

const fs = require('fs');
const path = require('path');
const p2 = require('../src/shared/quantus/poseidon2');

const P = p2.P;
const M32 = (1n << 32n) - 1n;
const M64 = (1n << 64n) - 1n;
const src = fs.readFileSync(path.join(__dirname, '../native/quantus/cuda/qc_kernel.cuh'), 'utf8');

// The diagonal as qc_kernel.cuh defines it (QC_D0 .. QC_D11).
function kernelDiag() {
  const d = [];
  for (let i = 0; i < 12; i++) {
    const m = new RegExp(`#define QC_D${i} (0x[0-9a-f]+)ull`).exec(src);
    if (!m) throw new Error(`no QC_D${i} in qc_kernel.cuh`);
    d.push(BigInt(m[1]));
  }
  return d;
}

// The constexpr part of qc_dmul.
function consts(D) {
  const d0 = D & M32;
  const d1 = D >> 32n;
  const g = d0 + d1 - 1n;
  const caseA = g < 1n << 32n;
  const u1 = caseA ? (1n << 32n) - d1 : M32 - d1;
  const v1 = caseA ? g : g + 1n - (1n << 32n);
  return { d0, d1, u1, v1, ucf: d0 + u1 <= M32, vcf: d1 + v1 <= M32 - 1n };
}

// qc_dmul's arithmetic. A chain marked carry-free (ucf, vcf) has no carry out in the kernel, so the model
// fails if one would be lost there.
function dmul(a, S, D) {
  const { d0, d1, u1, v1, ucf, vcf } = consts(D);
  const a0 = a & M32;
  const a1 = a >> 32n;
  const Ufull = a0 * d0 + (S & M32) + a1 * u1;
  if (ucf && Ufull > M64) throw new Error('U carried in a chain written without a carry');
  const U = Ufull & M64;
  const cu = Ufull >> 64n;
  const Vfull = a0 * d1 + (S >> 32n) + cu + a1 * v1;
  if (vcf && Vfull > M64) throw new Error('V carried in a chain written without a carry');
  const V = Vfull & M64;
  const cv = Vfull >> 64n;
  const hs = (U >> 32n) + (V & M32) + (V >> 32n);
  const hq = hs & M32;
  const t = hs >> 32n;
  const m = (t + cu + cv) & M32;
  return (((((hq + t) & M32) << 32n) | (U & M32)) - ((V >> 32n) + m)) & M64;
}

describe('qc_dmul (QC_DIAG 1)', () => {
  const D = kernelDiag();

  test('the kernel diagonal is the reference one', () => {
    expect(D).toEqual(p2.MATRIX_DIAG);
  });

  test('u1 and v1 are 32-bit and a D = U + V 2^32 (mod p) for every D', () => {
    for (const d of D) {
      const { d0, d1, u1, v1 } = consts(d);
      for (const c of [u1, v1]) expect(c >= 0n && c <= M32).toBe(true);
      // a = 1 and a = 2^32 pin both multiplier pairs: d0 + d1 2^32 = D and u1 + v1 2^32 = 2^32 D (mod p).
      expect((d0 + d1 * (1n << 32n)) % P).toBe(d % P);
      expect((u1 + v1 * (1n << 32n)) % P).toBe((d << 32n) % P);
    }
  });

  test('the carry-free chains cannot carry: their largest sums stay below 2^64', () => {
    for (const d of D) {
      const { d0, d1, u1, v1, ucf, vcf } = consts(d);
      if (ucf) expect(M32 * (d0 + u1) + M32 <= M64).toBe(true);
      if (vcf) expect(M32 * (d1 + v1) + M32 + 1n <= M64).toBe(true);
    }
  });

  test('which chains carry, as the round was tuned for (9 carries a round, not 24)', () => {
    const carries = D.map((d) => {
      const { ucf, vcf } = consts(d);
      return (ucf ? 0 : 1) + (vcf ? 0 : 1);
    });
    expect(carries).toEqual([1, 0, 1, 1, 1, 1, 0, 0, 2, 1, 1, 0]);
  });

  test('matches a D + S mod p for random a and S below 2^64', () => {
    let x = 0x2545f4914f6cdd1dn;
    const next = () => {
      x = (x * 6364136223846793005n + 1442695040888963407n) & M64;
      return x;
    };
    for (const d of D) {
      for (let i = 0; i < 400; i++) {
        const a = next();
        const S = next();
        expect(dmul(a, S, d) % P).toBe((a * d + S) % P);
      }
    }
  });

  test('matches at the edges of a and S, except a = p with s0 = 2^32 - 1, the rare high-word wrap', () => {
    const edges = [0n, 1n, M32, 1n << 32n, P - 1n, M64 - 1n, M64];
    for (const d of D) {
      for (const a of edges) {
        for (const S of [0n, 1n, M32, P - 1n, M64]) expect(dmul(a, S, d) % P).toBe((a * d + S) % P);
      }
    }
    const misses = D.filter((d) => dmul(P, M32, d) % P !== (P * d + M32) % P);
    expect(misses.length).toBeGreaterThan(0);
    expect(misses.length).toBeLessThan(12);
  });
});
