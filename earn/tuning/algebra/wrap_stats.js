'use strict';

// Where form 3's int8 wraps fall (algebra.js form 3: dC = La H, H = M_c Lb^T, fills 0), from the real noise
// generators: the rate by A-side degree of the rank index, distinct wrapped rank indices per 64-column block a
// chunk, and how many correction rounds a warp needs when its 4 column classes (t = (j % 8) >> 1 of the m8n8
// accumulator) work in parallel.
//
//   node earn/tuning/algebra/wrap_stats.js [seeds] [k]

const { generateUniformRandomMatrix, generatePermutationMatrix, SEED_LABEL_A, SEED_LABEL_B } =
  require('../../src/shared/miner/noise');
const { hash } = require('../../src/shared/miner/blake3');

const R = 128, BC = 64;
const seeds = Number(process.argv[2] || 4), k = Number(process.argv[3] || 8192);
const byDeg = new Map(); // deg -> [entries, wraps]
const distinct = [], rounds = [], perBlock = [];
for (let s = 0; s < seeds; s++) {
  const aSeed = hash(Buffer.from(`wrap-a-${s}`)), bSeed = hash(Buffer.from(`wrap-b-${s}`));
  const cols = Array.from({ length: BC }, (_, i) => 4096 * s + i);
  const Lb = generateUniformRandomMatrix(SEED_LABEL_B, bSeed, cols, R);
  const pA = generatePermutationMatrix(SEED_LABEL_A, aSeed, k, R);
  const pB = generatePermutationMatrix(SEED_LABEL_B, bSeed, k, R);
  for (let c = 0; c < k / R; c++) {
    const H = Array.from({ length: R }, () => new Int32Array(BC));
    const deg = new Int32Array(R);
    for (let kk = c * R; kk < (c + 1) * R; kk++) {
      const a0 = pA[2 * kk], a1 = pA[2 * kk + 1], b0 = pB[2 * kk], b1 = pB[2 * kk + 1];
      deg[a0]++; deg[a1]++;
      for (let j = 0; j < BC; j++) { const v = Lb[j][b0] - Lb[j][b1]; H[a0][j] += v; H[a1][j] -= v; }
    }
    const ps = new Set(), cls = [0, 0, 0, 0];
    let n = 0;
    for (let p = 0; p < R; p++) {
      const e = byDeg.get(deg[p]) || [0, 0];
      for (let j = 0; j < BC; j++) {
        e[0]++;
        if (H[p][j] < -128 || H[p][j] > 127) { e[1]++; n++; ps.add(p); cls[(j % 8) >> 1]++; }
      }
      byDeg.set(deg[p], e);
    }
    perBlock.push(n); distinct.push(ps.size); rounds.push(Math.max(...cls));
  }
}
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
let te = 0, tw = 0;
for (const d of [...byDeg.keys()].sort((a, b) => a - b)) {
  const [e, w] = byDeg.get(d); te += e; tw += w;
  console.log(`A degree ${d}: ${(100 * e / (R * BC * perBlock.length)).toFixed(1)}% of rank rows, wrap rate ${(100 * w / e).toFixed(3)}%`);
}
console.log(`all: wrap rate ${(100 * tw / te).toFixed(3)}%; per 64-column block a chunk: ${mean(perBlock).toFixed(1)} wraps, ` +
  `${mean(distinct).toFixed(1)} distinct rank indices, ${mean(rounds).toFixed(1)} rounds (max ${Math.max(...rounds)}) over ${perBlock.length} chunks`);
