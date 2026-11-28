'use strict';

// The CUDA core's constant tables and the one add whose bound depends on them.
//
// qc_kernel.cuh holds the round constants twice: in the order build h's loops
// read them (QC_RC_F1, QC_RC_P, QC_RC_T, QC_RC_I0) and in the order the compact
// loop reads them (QC_RC_SEQ, QC_RC_P22). Both must match the CPU reference
// (src/shared/quantus/poseidon2.js), which matches Quantus's published vectors.
//
// The compact loop adds the last partial round's constants for elements 1..11
// with qc_add_small (qc_field.cuh). That add folds one carry out of 2^64 back
// in as EPS = 2^32 - 1 and drops any second carry, so it is exact (a value below
// 2^64, congruent to a + b mod p) only while a + b - 2^64 + EPS stays below
// 2^64: for every a below 2^64 that means b <= p. The tests below model the add
// word for word and pin that bound, and check every constant it is given.

const fs = require('fs');
const path = require('path');
const p2 = require('../src/shared/quantus/poseidon2');

const P = p2.P;
const EPS = (1n << 32n) - 1n;
const M64 = (1n << 64n) - 1n;
const src = fs.readFileSync(path.join(__dirname, '../native/quantus/cuda/qc_kernel.cuh'), 'utf8');

// The values of `__constant__ u64 NAME[...] = { ... };`, flattened, as BigInts.
function table(name) {
  const m = new RegExp(`__constant__ u64 ${name}(\\[\\d+\\])+ = \\{([\\s\\S]*?)\\};`).exec(src);
  if (!m) throw new Error(`no table ${name} in qc_kernel.cuh`);
  return m[2].split(/[\s,{}]+/).filter(Boolean).map((v) => BigInt(v.replace(/ull$/, '')));
}
const rows = (flat, n) => Array.from({ length: flat.length / n }, (_, i) => flat.slice(i * n, i * n + n));

// qc_add_small as its PTX does it: a 64-bit add, then c * EPS added with
// mad.lo.cc / madc.hi, whose carry out of bit 63 is lost.
function addSmall(a, b) {
  const s = a + b;
  const c = s >> 64n;
  return ((s & M64) + c * EPS) & M64;
}

describe('quantus_cuda constant tables', () => {
  const I = p2.INITIAL_EXTERNAL_CONSTANTS;
  const T = p2.TERMINAL_EXTERNAL_CONSTANTS;
  const N = p2.INTERNAL_CONSTANTS;

  test('build h tables match the reference', () => {
    expect(table('QC_RC_I0')).toEqual(I[0]);
    expect(rows(table('QC_RC_F1'), 12)).toEqual([I[1], I[2], I[3], [N[0], ...Array(11).fill(0n)]]);
    expect(table('QC_RC_P')).toEqual(N.slice(1));
    expect(rows(table('QC_RC_T'), 12)).toEqual([...T, [1n, 1n, ...Array(10).fill(0n)]]);
  });

  test('the compact tables are the same constants in the compact loop order', () => {
    expect(rows(table('QC_RC_SEQ'), 12)).toEqual([
      I[0], I[1], I[2], I[3], [N[0], ...Array(11).fill(0n)], T[1], T[2], T[3], [1n, 1n, ...Array(10).fill(0n)],
    ]);
    expect(table('QC_RC_P22')).toEqual([...N.slice(1), T[0][0]]);
  });

  test('every constant qc_add_small is given is at most p', () => {
    for (const b of T[0].slice(1)) expect(b <= P).toBe(true);
  });
});

describe('qc_add_small bound', () => {
  const exact = (a, b) => {
    const r = addSmall(a, b);
    return r <= M64 && r % P === (a + b) % P;
  };

  test('exact for every a below 2^64 when b <= p, at the edges', () => {
    const edges = [0n, 1n, EPS, P - 1n, P, M64 - 1n, M64];
    for (const a of edges) {
      for (const b of [0n, 1n, EPS, 1n << 40n, P - 1n, P]) expect(exact(a, b)).toBe(true);
    }
  });

  test('exact for random a below 2^64 and b below p', () => {
    let x = 0x9e3779b97f4a7c15n;
    const next = () => {
      x = (x * 6364136223846793005n + 1442695040888963407n) & M64;
      return x;
    };
    for (let i = 0; i < 2000; i++) expect(exact(next(), next() % P)).toBe(true);
  });

  test('the bound is tight: b = p + 1 with a = 2^64 - 1 wraps', () => {
    expect(exact(M64, P + 1n)).toBe(false);
  });

  test('every round constant used with it in the kernel keeps the add exact for a = 2^64 - 1', () => {
    for (const b of p2.TERMINAL_EXTERNAL_CONSTANTS[0]) expect(exact(M64, b)).toBe(true);
  });
});
