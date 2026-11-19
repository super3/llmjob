'use strict';

// CPU check of two exact rewrites of Pearl's noised product, against the shipped reference fold
// (src/shared/miner/reference.js foldTranscript), bit for bit. Analysis only: nothing here ships.
//
//   node earn/tuning/algebra/algebra.js [seed] [tiles]
//
// The noise, per noise.js: A'[r][kk] = A[r][kk] + La[r][pA0(kk)] - La[r][pA1(kk)], and the same for B'
// with Lb and pB. So inside chunk c (kk in [128c, 128c + 128)):
//
//   dC_c = A'_c B'_c = La M_c Lb^T  (+ fill terms),   M_c = S_A,c^T S_B,c   (128 x 128),
//
// where row kk of S is e_p0 - e_p1. Form 1 computes the cumulative tile through M_c. Form 2 is the int8
// GEMM with columns dropped: a cycle in a chunk's A graph makes one k column the signed sum of the others,
// so that column goes and its B row is added into theirs (and the same with the sides swapped). A row that
// takes one addition is 4 table terms, still inside int8 (|sum| <= 126); a second would not be, so the
// cycles used must not share a k. Form 2 prints how many of the 128 columns that leaves. Form 3 keeps La
// fixed as one operand and builds H_c = M_c Lb^T as the other, int8 with a correction for each entry that
// wraps; it prints how often that happens.

const { generateUniformRandomMatrix, generatePermutationMatrix, SEED_LABEL_A, SEED_LABEL_B } =
  require('../../src/shared/miner/noise');
const { computeNoiseForIndices } = require('../../src/shared/miner/noise');
const { foldTranscript, rotl13 } = require('../../src/shared/miner/reference');
const { hash } = require('../../src/shared/miner/blake3');

const RANK = 128;
const LANES = 16;

function fold(tileChunks, h, w) {
  // tileChunks(c) -> Int32Array(h*w) of dC for chunk c; returns the 16-lane transcript.
  const jackpot = new Uint32Array(LANES);
  const acc = new Int32Array(h * w);
  for (let c = 0; ; c++) {
    const d = tileChunks(c);
    if (!d) break;
    let x = 0;
    for (let i = 0; i < h * w; i++) { acc[i] = (acc[i] + d[i]) | 0; x = (x ^ acc[i]) >>> 0; }
    jackpot[c % LANES] = (rotl13(jackpot[c % LANES]) ^ x) >>> 0;
  }
  return jackpot;
}

// Form 1: dC_c = fa*fb*128 + fa * colsum_c(E_B) + fb * rowsum_c(E_A) + La M_c Lb^T.
function form1(ctx) {
  const { La, Lb, pA, pB, h, w, k, fa, fb } = ctx;
  return (c) => {
    if (c * RANK >= k) return null;
    const M = new Int32Array(RANK * RANK);
    const rowE = new Int32Array(h), colE = new Int32Array(w);
    for (let kk = c * RANK; kk < (c + 1) * RANK; kk++) {
      const a0 = pA[2 * kk], a1 = pA[2 * kk + 1], b0 = pB[2 * kk], b1 = pB[2 * kk + 1];
      M[a0 * RANK + b0] += 1; M[a0 * RANK + b1] -= 1; M[a1 * RANK + b0] -= 1; M[a1 * RANK + b1] += 1;
      for (let r = 0; r < h; r++) rowE[r] += La[r][a0] - La[r][a1];
      for (let j = 0; j < w; j++) colE[j] += Lb[j][b0] - Lb[j][b1];
    }
    const out = new Int32Array(h * w);
    for (let r = 0; r < h; r++) {
      const t = new Int32Array(RANK); // (La M)[r][q]
      for (let p = 0; p < RANK; p++) {
        const l = La[r][p];
        if (!l) continue;
        for (let q = 0; q < RANK; q++) t[q] += l * M[p * RANK + q];
      }
      for (let j = 0; j < w; j++) {
        let s = fa * fb * RANK + fa * colE[j] + fb * rowE[r];
        for (let q = 0; q < RANK; q++) s += t[q] * Lb[j][q];
        out[r * w + j] = s | 0;
      }
    }
    return out;
  };
}

// Shortest cycle through the alive k's of one side's graph, as [e, ...path] with signs:
// a_e + sum s_i a_i = 0 over the path, a_i = e_p0(i) - e_p1(i).
function shortestCycle(perm, alive) {
  const adj = new Map();
  const add = (x, y, i) => { if (!adj.has(x)) adj.set(x, []); adj.get(x).push([y, i]); };
  for (const i of alive) { add(perm[2 * i], perm[2 * i + 1], i); add(perm[2 * i + 1], perm[2 * i], i); }
  let best = null;
  for (const e of alive) {
    const x = perm[2 * e], y = perm[2 * e + 1];
    // BFS from y to x without e; a path edge walked from u to v contributes e_u - e_v.
    const prev = new Map([[y, null]]);
    const q = [y];
    for (let qi = 0; qi < q.length && !prev.has(x); qi++) {
      const u = q[qi];
      for (const [v, i] of adj.get(u) || []) {
        if (i === e || prev.has(v)) continue;
        prev.set(v, [u, i]); q.push(v);
      }
    }
    if (!prev.has(x)) continue;
    // Walked y -> x: sum (e_u - e_v) = e_y - e_x = -a_e, so a_e + sum s_i a_i = 0 with
    // s_i = +1 when a_i = e_u - e_v (p0 = u), else -1.
    const path = [];
    for (let v = x; prev.get(v); ) {
      const [u, i] = prev.get(v);
      path.push([i, perm[2 * i] === u ? 1 : -1]);
      v = u;
    }
    if (!best || path.length < best.path.length) best = { e, path };
  }
  return best;
}

// Form 2: per chunk, the int8 GEMM over the k columns that remain after edge-disjoint cycle drops.
function form2(ctx, stats) {
  const { La, Lb, pA, pB, h, w, k } = ctx;
  return (c) => {
    if (c * RANK >= k) return null;
    const ks = []; for (let kk = c * RANK; kk < (c + 1) * RANK; kk++) ks.push(kk);
    // Each term: an A vector and a B vector over rank indices (sparse maps), int8 by construction.
    const ua = new Map(), vb = new Map();
    const vec = (perm, kk) => new Map([[perm[2 * kk], 1], [perm[2 * kk + 1], -1]]);
    for (const kk of ks) { ua.set(kk, vec(pA, kk)); vb.set(kk, vec(pB, kk)); }
    const used = new Set();
    for (;;) {
      const alive = ks.filter((kk) => !used.has(kk));
      const cA = shortestCycle(pA, alive), cB = shortestCycle(pB, alive);
      let pick = null, side = null;
      if (cA && (!cB || cA.path.length <= cB.path.length)) { pick = cA; side = 'A'; } else if (cB) { pick = cB; side = 'B'; }
      if (!pick) break;
      // A cycle: column e of A' is -sum s_i A'[:, i], so B'[e] moves into each B'[i] with sign -s_i.
      const recv = side === 'A' ? vb : ua;
      const give = recv.get(pick.e);
      for (const [i, s] of pick.path) {
        const t = recv.get(i);
        for (const [q, x] of give) t.set(q, (t.get(q) || 0) - s * x);
      }
      ua.delete(pick.e); vb.delete(pick.e);
      used.add(pick.e); for (const [i] of pick.path) used.add(i);
    }
    stats.push(ua.size);
    const out = new Int32Array(h * w);
    for (const [kk, a] of ua) {
      const b = vb.get(kk);
      const colA = new Int32Array(h), rowB = new Int32Array(w);
      for (let r = 0; r < h; r++) { let s = 0; for (const [p, x] of a) s += x * La[r][p]; colA[r] = s; }
      for (let j = 0; j < w; j++) { let s = 0; for (const [q, x] of b) s += x * Lb[j][q]; rowB[j] = s; }
      for (const v of colA) if (v < -128 || v > 127) throw new Error('A operand left int8');
      for (const v of rowB) if (v < -128 || v > 127) throw new Error('B operand left int8');
      for (let r = 0; r < h; r++) for (let j = 0; j < w; j++) out[r * w + j] = (out[r * w + j] + colA[r] * rowB[j]) | 0;
    }
    return out;
  };
}


// Form 3, the one operand fixed: with A's fill 0,
//   dC_c[r][j] = sum_p La[r][p] H_c[p][j],   H_c = M_c Lb^T + fb * dA_c 1^T,   dA_c = S_A,c^T 1.
// La is the same for every chunk of a tile (it can sit in registers); H_c is one 128 x n operand a chunk, shared
// by every row. H is fed as int8 wrapped mod 256 and every entry that wrapped is an event (p, j, delta) whose
// 256 * delta * La[r][p] is added back on the CUDA cores. Counts the events.
function form3(ctx, ev) {
  const { La, Lb, pA, pB, h, w, k, fb } = ctx;
  return (c) => {
    if (c * RANK >= k) return null;
    const H = Array.from({ length: RANK }, () => new Int32Array(w));
    for (let kk = c * RANK; kk < (c + 1) * RANK; kk++) {
      const a0 = pA[2 * kk], a1 = pA[2 * kk + 1], b0 = pB[2 * kk], b1 = pB[2 * kk + 1];
      for (let j = 0; j < w; j++) {
        const bp = fb + Lb[j][b0] - Lb[j][b1]; // B'[kk][j]
        H[a0][j] += bp; H[a1][j] -= bp;
      }
    }
    const out = new Int32Array(h * w);
    for (let p = 0; p < RANK; p++) for (let j = 0; j < w; j++) {
      const v = H[p][j];
      const w8 = ((v + 128) & 255) - 128; // what an int8 operand holds
      const delta = (v - w8) / 256;
      ev.entries++;
      if (delta) ev.events++;
      for (let r = 0; r < h; r++) out[r * w + j] = (out[r * w + j] + La[r][p] * w8 + 256 * delta * La[r][p]) | 0;
    }
    return out;
  };
}

function main() {
  const seed = Number(process.argv[2] || 1);
  const tiles = Number(process.argv[3] || 3);
  const h = 16, w = 16;
  let ok = 0, checks = 0;
  const kept = [];
  const ev = { 0: { entries: 0, events: 0 }, 48: { entries: 0, events: 0 } };
  for (let t = 0; t < tiles; t++) {
    const aSeed = hash(Buffer.from(`algebra-a-${seed}-${t}`));
    const bSeed = hash(Buffer.from(`algebra-b-${seed}-${t}`));
    const k = t % 2 ? 4096 : 2048; // 4096 wraps the 16 lanes once
    const rows = Array.from({ length: h }, (_, i) => 1000 * t + 8 * i);
    const cols = Array.from({ length: w }, (_, i) => 777 * t + 3 * i);
    const { noiseA, noiseB } = computeNoiseForIndices({ k, rank: RANK, aSeed, bSeed, rowIndices: rows, colIndices: cols });
    const La = generateUniformRandomMatrix(SEED_LABEL_A, aSeed, rows, RANK);
    const Lb = generateUniformRandomMatrix(SEED_LABEL_B, bSeed, cols, RANK);
    const pA = generatePermutationMatrix(SEED_LABEL_A, aSeed, k, RANK);
    const pB = generatePermutationMatrix(SEED_LABEL_B, bSeed, k, RANK);
    for (const [fa, fb] of [[0, 0], [48, 48], [-17, 63], [0, 48]]) {
      const A = new Int8Array(h * k).fill(fa), Bt = new Int8Array(w * k).fill(fb);
      const ref = foldTranscript({ A, Bt, noiseA, noiseB, rows: rows.map((_, i) => i), cols: cols.map((_, i) => i), k, rank: RANK });
      const ctx = { La, Lb, pA, pB, h, w, k, fa, fb };
      const f1 = fold(form1(ctx), h, w);
      const same1 = f1.every((x, i) => x === ref[i]);
      let line = `tile ${t} k ${k} fill ${fa}/${fb}: form 1 ${same1 ? 'matches' : 'DIFFERS'}`;
      ok += same1; checks++;
      if (fa === 0) {
        const f3 = fold(form3(ctx, ev[fb] || (ev[fb] = { entries: 0, events: 0 })), h, w);
        const same3 = f3.every((x, i) => x === ref[i]);
        ok += same3; checks++;
        line += `, form 3 ${same3 ? 'matches' : 'DIFFERS'}`;
      }
      if (fa === 0 && fb === 0) {
        const f2 = fold(form2(ctx, kept), h, w);
        const same2 = f2.every((x, i) => x === ref[i]);
        ok += same2; checks++;
        line += `, form 2 ${same2 ? 'matches' : 'DIFFERS'}`;
      }
      console.log(line);
    }
  }
  const mean = kept.reduce((a, b) => a + b, 0) / kept.length;
  console.log(`form 2 k columns kept a chunk: mean ${mean.toFixed(1)} of 128, min ${Math.min(...kept)}, max ${Math.max(...kept)} (${kept.length} chunks)`);
  for (const [fb, e] of Object.entries(ev)) if (e.entries) {
    console.log(`form 3, B fill ${fb}: ${e.events} of ${e.entries} H entries wrapped (${(100 * e.events / e.entries).toFixed(3)}%, ${(RANK * e.events / e.entries).toFixed(2)} a column a chunk)`);
  }
  console.log(`checks passed: ${ok} of ${checks}`);
  if (ok !== checks) process.exit(1);
}

main();
