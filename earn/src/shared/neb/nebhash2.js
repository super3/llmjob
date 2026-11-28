'use strict';

// nebhash v2, the proof of work of NEB (Neural Engine Byte), in JavaScript.
//
// One attempt is one integer training step of a small network the chain owns:
// the attempt's key picks 16 samples of a dataset, the network runs forward
// and backward on them in int32, and the clipped gradient is hashed into the
// ticket. No floating point anywhere, so this gives the chain's bytes.
//
// Ported from the chain's C reference, pow/ref/nebhash2.c and shares.c of
// github.com/SilasEigen/neb (Apache-2.0), following spec/nebhash-v2.md and
// spec/shares.md. test/nebHash.test.js replays the chain's own vectors
// (vectors/nebhash2.txt) through it: digests, tickets, sample indices and the
// model states after each step.
//
// This is the oracle. The GPU cores are checked against it, and the miner
// evaluates every nonce a card reports here before it sends it.

const blake3 = require('../miner/blake3');
const chacha8 = require('./chacha8');

const NB = 16;          // samples an attempt
const CLIP = 256;       // bound of a gradient entry of one attempt
const TPOS = 7;
const TNEG = -8;
const NOUT = 2;
const KEP1 = 1;
const RET2 = 2;
const CONV1D = 1;
const CONV2D = 2;
const LINEAR = 3;

function layer(kind, cin, cout, k, stride, pad, pool, s, sb, su, inLen, inOff, src, convLen, outLen, nw, wOff) {
  return { kind, cin, cout, k, stride, pad, pool, s, sb, su, inLen, inOff, src, convLen, outLen, nw, wOff };
}

// The tables of spec/nebhash-v2.md section 3, as pow/ref/nebhash2.c writes them.
const ARCHS = {
  [KEP1]: {
    id: KEP1, nin: 784, nweights: 26000,
    layers: [
      layer(CONV1D, 1, 8, 5, 1, 2, 4, 2, 0, 15, 583, 0, [], 583, 145, 40, 0),
      layer(CONV1D, 8, 16, 5, 1, 2, 4, 4, 5, 15, 145, 0, [0], 145, 36, 640, 40),
      layer(CONV1D, 1, 8, 5, 1, 2, 4, 3, 0, 15, 201, 583, [], 201, 50, 40, 680),
      layer(CONV1D, 8, 16, 5, 1, 2, 4, 4, 4, 15, 50, 0, [2], 50, 12, 640, 720),
      layer(LINEAR, 768, 32, 1, 1, 0, 1, 5, 5, 15, 1, 0, [1, 3], 1, 1, 24576, 1360),
      layer(LINEAR, 32, 2, 1, 1, 0, 1, 4, 3, 12, 1, 0, [4], 1, 1, 64, 25936),
    ],
  },
  [RET2]: {
    id: RET2, nin: 4096, nweights: 22280,
    layers: [
      layer(CONV2D, 1, 8, 3, 2, 1, 2, 3, 0, 16, 64, 0, [], 32, 16, 72, 0),
      layer(CONV2D, 8, 16, 3, 1, 1, 2, 4, 6, 16, 16, 0, [0], 16, 8, 1152, 72),
      layer(CONV2D, 16, 32, 3, 1, 1, 2, 5, 6, 16, 8, 0, [1], 8, 4, 4608, 1224),
      layer(LINEAR, 512, 32, 1, 1, 0, 1, 5, 5, 16, 1, 0, [2], 1, 1, 16384, 5832),
      layer(LINEAR, 32, 2, 1, 1, 0, 1, 4, 3, 13, 1, 0, [3], 1, 1, 64, 22216),
    ],
  },
};

function archOf(id) {
  return ARCHS[id] || null;
}

const sq = (L, n) => (L.kind === CONV2D ? n * n : n);
const inSize = (L) => L.cin * sq(L, L.inLen);
const convSize = (L) => L.cout * sq(L, L.convLen);
const outSize = (L) => L.cout * sq(L, L.outLen);

function clipi(v, lo, hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// rshift of spec/shares.md section 2.1: rounds half up; rshift(x, 0) is x.
function rshift(x, n) {
  return n === 0 ? x : (x + (1 << (n - 1))) >> n;
}

// ── weights ────────────────────────────────────────────────────────────────

// Signed nibbles, low nibble first (spec/shares.md section 3.4).
function unpackWeights(packed, n) {
  const w = new Int8Array(n);
  for (let e = 0; e < n; e++) {
    const v = (packed[e >> 1] >> ((e & 1) * 4)) & 15;
    w[e] = v < 8 ? v : v - 16;
  }
  return w;
}

function packWeights(w) {
  const out = Buffer.alloc((w.length + 1) >> 1);
  for (let e = 0; e < w.length; e++) out[e >> 1] |= (w[e] & 15) << ((e & 1) * 4);
  return out;
}

// weights_hash of section 6: what a job names its model by.
function weightsHash(packed) {
  return blake3.hash(Buffer.concat([Buffer.from('neb-model-weights'), packed]));
}

function residualHash(r) {
  const buf = Buffer.alloc(18 + 4 * r.length);
  buf.write('neb-model-residual');
  for (let f = 0; f < r.length; f++) buf.writeInt32LE(r[f], 18 + 4 * f);
  return blake3.hash(buf);
}

// A model: { arch, w (Int8Array), r (Int32Array residuals) }.
function modelFromPacked(arch, packed) {
  const A = archOf(arch);
  if (!A) throw new Error('unknown NEB architecture ' + arch);
  if (packed.length !== A.nweights / 2) throw new Error('a model of ' + packed.length + ' bytes; this network takes ' + A.nweights / 2);
  return { arch, w: unpackWeights(packed, A.nweights), r: new Int32Array(A.nweights) };
}

// The initial state of section 6 for a 32-byte seed.
function modelInit(arch, seed) {
  const A = archOf(arch);
  const key = blake3.hash(Buffer.concat([Buffer.from('neb-model-init'), seed]));
  const s = chacha8.stream(key, A.nweights);
  const w = new Int8Array(A.nweights);
  for (let k = 0; k < A.nweights; k++) {
    const b = s[k];
    w[k] = (b & 3) + ((b >> 2) & 3) + ((b >> 4) & 3) + ((b >> 6) & 3) - 6;
  }
  return { arch, w, r: new Int32Array(A.nweights) };
}

// The model step of section 5: gs is the sum of the clipped gradients.
function modelApply(m, gs) {
  const A = archOf(m.arch);
  for (const L of A.layers) {
    const unit = 2 ** L.su;
    for (let f = L.wOff; f < L.wOff + L.nw; f++) {
      let v = m.r[f] + gs[f];
      const d = clipi(Math.trunc(v / unit), -1, 1);
      v -= d * unit;
      m.r[f] = clipi(v, -unit, unit);
      m.w[f] = clipi(m.w[f] - d, -8, 7);
    }
  }
}

// ── one sample ─────────────────────────────────────────────────────────────

function makeWork(A) {
  return A.layers.map((L) => ({
    in: new Int8Array(inSize(L)),
    mm: new Int32Array(inSize(L)),
    a: new Int8Array(convSize(L)),
    y: new Int32Array(convSize(L)),
    d: new Int32Array(convSize(L)),
    out: new Int8Array(outSize(L)),
    arg: new Int32Array(outSize(L)),
    e: new Int32Array(outSize(L)),
  }));
}

function gatherInput(A, W, i, x, xOff) {
  const L = A.layers[i];
  const inp = W[i].in;
  if (!L.src.length) {
    for (let j = 0; j < inp.length; j++) inp[j] = x[xOff + L.inOff + j];
    return;
  }
  let at = 0;
  for (const s of L.src) {
    inp.set(W[s].out, at);
    at += W[s].out.length;
  }
}

function forwardLayer(A, w, W, i) {
  const L = A.layers[i];
  const B = W[i];
  const inp = B.in;
  const wo = L.wOff;
  if (L.kind === LINEAR) {
    for (let o = 0; o < L.cout; o++) {
      let z = 0;
      const base = wo + o * L.cin;
      for (let j = 0; j < L.cin; j++) z += w[base + j] * inp[j];
      B.y[o] = rshift(z, L.s);
      B.a[o] = clipi(B.y[o], -8, 7);
      B.out[o] = B.a[o];
      B.arg[o] = o;
    }
    return;
  }
  const n = L.inLen, cl = L.convLen, k = L.k, st = L.stride, pad = L.pad, ol = L.outLen, pl = L.pool;
  if (L.kind === CONV1D) {
    for (let o = 0; o < L.cout; o++) {
      for (let p = 0; p < cl; p++) {
        let z = 0;
        for (let c = 0; c < L.cin; c++) {
          const base = wo + (o * L.cin + c) * k;
          const row = c * n;
          for (let t = 0; t < k; t++) {
            const q = p * st + t - pad;
            if (q >= 0 && q < n) z += w[base + t] * inp[row + q];
          }
        }
        const g = o * cl + p;
        B.y[g] = rshift(z, L.s);
        B.a[g] = clipi(B.y[g], -8, 7);
      }
    }
    for (let o = 0; o < L.cout; o++) {
      for (let j = 0; j < ol; j++) {
        let best = o * cl + j * pl;
        for (let t = 1; t < pl; t++) if (B.a[o * cl + j * pl + t] > B.a[best]) best = o * cl + j * pl + t;
        B.out[o * ol + j] = B.a[best];
        B.arg[o * ol + j] = best;
      }
    }
    return;
  }
  for (let o = 0; o < L.cout; o++) {
    for (let py = 0; py < cl; py++) {
      for (let px = 0; px < cl; px++) {
        let z = 0;
        for (let c = 0; c < L.cin; c++) {
          const base = wo + (o * L.cin + c) * k * k;
          const plane = c * n * n;
          for (let ky = 0; ky < k; ky++) {
            const qy = py * st + ky - pad;
            if (qy < 0 || qy >= n) continue;
            for (let kx = 0; kx < k; kx++) {
              const qx = px * st + kx - pad;
              if (qx >= 0 && qx < n) z += w[base + ky * k + kx] * inp[plane + qy * n + qx];
            }
          }
        }
        const g = (o * cl + py) * cl + px;
        B.y[g] = rshift(z, L.s);
        B.a[g] = clipi(B.y[g], -8, 7);
      }
    }
  }
  // p x p windows from (0, 0); the first maximum in row-major order keeps the position.
  for (let o = 0; o < L.cout; o++) {
    for (let jy = 0; jy < ol; jy++) {
      for (let jx = 0; jx < ol; jx++) {
        let best = (o * cl + jy * pl) * cl + jx * pl;
        for (let ty = 0; ty < pl; ty++) {
          for (let tx = 0; tx < pl; tx++) {
            const g = (o * cl + jy * pl + ty) * cl + jx * pl + tx;
            if (B.a[g] > B.a[best]) best = g;
          }
        }
        const jo = (o * ol + jy) * ol + jx;
        B.out[jo] = B.a[best];
        B.arg[jo] = best;
      }
    }
  }
}

// Section 4.2: route e back through the pooling, mask, add d (x) in to the
// layer's gradient g, and send the error on to the layers this one reads.
function backwardLayer(A, w, g, W, i) {
  const L = A.layers[i];
  const B = W[i];
  const wo = L.wOff;
  B.d.fill(0);
  for (let j = 0; j < B.e.length; j++) B.d[B.arg[j]] = B.e[j];
  // No gradient through a clipped activation, except on the output layer.
  if (i !== A.layers.length - 1) {
    for (let c = 0; c < B.d.length; c++) if (B.y[c] < -8 || B.y[c] > 7) B.d[c] = 0;
  }
  const inp = B.in;
  const send = L.src.length > 0;
  if (send) B.mm.fill(0);
  if (L.kind === LINEAR) {
    for (let o = 0; o < L.cout; o++) {
      const d = B.d[o];
      if (d === 0) continue;
      const base = wo + o * L.cin;
      for (let j = 0; j < L.cin; j++) g[base + j] += d * inp[j];
      if (send) for (let j = 0; j < L.cin; j++) B.mm[j] += d * w[base + j];
    }
  } else if (L.kind === CONV1D) {
    const n = L.inLen, cl = L.convLen, k = L.k, st = L.stride, pad = L.pad;
    for (let o = 0; o < L.cout; o++) {
      for (let p = 0; p < cl; p++) {
        const d = B.d[o * cl + p];
        if (d === 0) continue;
        for (let c = 0; c < L.cin; c++) {
          const base = wo + (o * L.cin + c) * k;
          for (let t = 0; t < k; t++) {
            const q = p * st + t - pad;
            if (q < 0 || q >= n) continue;
            g[base + t] += d * inp[c * n + q];
            if (send) B.mm[c * n + q] += d * w[base + t];
          }
        }
      }
    }
  } else {
    const n = L.inLen, cl = L.convLen, k = L.k, st = L.stride, pad = L.pad;
    for (let o = 0; o < L.cout; o++) {
      for (let py = 0; py < cl; py++) {
        for (let px = 0; px < cl; px++) {
          const d = B.d[(o * cl + py) * cl + px];
          if (d === 0) continue;
          for (let c = 0; c < L.cin; c++) {
            const base = wo + (o * L.cin + c) * k * k;
            for (let ky = 0; ky < k; ky++) {
              const qy = py * st + ky - pad;
              if (qy < 0 || qy >= n) continue;
              for (let kx = 0; kx < k; kx++) {
                const qx = px * st + kx - pad;
                if (qx < 0 || qx >= n) continue;
                const q = (c * n + qy) * n + qx;
                g[base + ky * k + kx] += d * inp[q];
                if (send) B.mm[q] += d * w[base + ky * k + kx];
              }
            }
          }
        }
      }
    }
  }
  if (!send) return;
  let at = 0;
  for (const s of L.src) {
    const e = W[s].e;
    for (let j = 0; j < e.length; j++) e[j] = clipi(rshift(B.mm[at + j], L.sb), -8, 7);
    at += e.length;
  }
}

// Adds one sample's gradient (before the clip) to g. x holds the sample's
// inputs from xOff. Returns true when the sample is live: its output differs
// from its target. A sample that is not live adds nothing: with a zero output
// error every error behind it is clip(rshift(0, SB)) = 0.
function sampleGrad(A, m, W, x, xOff, label, g) {
  const last = A.layers.length - 1;
  for (let i = 0; i <= last; i++) {
    gatherInput(A, W, i, x, xOff);
    forwardLayer(A, m.w, W, i);
  }
  let live = false;
  for (let o = 0; o < NOUT; o++) {
    const e = clipi(W[last].out[o] - (o === label ? TPOS : TNEG), -8, 7);
    W[last].e[o] = e;
    if (e !== 0) live = true;
  }
  if (!live) return false;
  for (let i = last; i >= 0; i--) backwardLayer(A, m.w, g, W, i);
  return true;
}

// ── one attempt ────────────────────────────────────────────────────────────

// The 16 sample indices of a key (spec/shares.md section 5.1).
function shareIndices(key, n) {
  const u = chacha8.blockWords(chacha8.keyWords(key), 0);
  const idx = new Array(NB);
  // The product can pass 2^53.
  for (let j = 0; j < NB; j++) idx[j] = Number((BigInt(u[j]) * BigInt(n)) >> 32n);
  return idx;
}

// Clips g in place and returns the 32-byte digest (section 4.3).
function digestOf(A, g) {
  const D = new Uint32Array(8);
  for (let i = 0; i < A.layers.length; i++) {
    const L = A.layers[i];
    const tag = 7919 * (i + 1);
    for (let f = 0; f < L.nw; f++) {
      const c = clipi(g[L.wOff + f], -CLIP, CLIP);
      g[L.wOff + f] = c;
      D[f & 7] += c ^ (f + tag);
    }
  }
  const out = Buffer.alloc(32);
  for (let d = 0; d < 8; d++) out.writeUInt32LE(D[d], 4 * d);
  return out;
}

function ticketOf(digest, key) {
  return blake3.hash(Buffer.concat([digest, key]));
}

// One attempt on a model and a dataset { n, arch, values (n * nin bytes),
// labels (n bytes) }: the clipped gradient, the digest, the ticket, the sample
// indices and how many of the 16 samples were live.
function shareEval(m, ds, key) {
  const A = archOf(m.arch);
  const W = makeWork(A);
  const idx = shareIndices(key, ds.n);
  const g = new Int32Array(A.nweights);
  let live = 0;
  for (let b = 0; b < NB; b++) {
    if (sampleGrad(A, m, W, ds.values, idx[b] * A.nin, ds.labels[idx[b]], g)) live++;
  }
  const digest = digestOf(A, g);
  return { g, digest, ticket: ticketOf(digest, key), idx, live };
}

// Which samples of the dataset are live on this model, one byte each.
function liveSamples(m, ds) {
  const A = archOf(m.arch);
  const W = makeWork(A);
  const g = new Int32Array(A.nweights);
  const live = new Uint8Array(ds.n);
  for (let i = 0; i < ds.n; i++) live[i] = sampleGrad(A, m, W, ds.values, i * A.nin, ds.labels[i], g) ? 1 : 0;
  return live;
}

// How many of an attempt's 16 samples are live, for a key and a liveSamples
// list. The GPU cores only evaluate attempts where this is small.
function liveCount(key, live) {
  let c = 0;
  for (const i of shareIndices(key, live.length)) c += live[i];
  return c;
}

// The share of attempts with at most maxLive live samples, when a fraction p
// of the dataset is live: the binomial's low tail over 16 draws.
function candidateShare(p, maxLive) {
  let sum = 0;
  let choose = 1;
  for (let k = 0; k <= maxLive; k++) {
    sum += choose * p ** k * (1 - p) ** (NB - k);
    choose = choose * (NB - k) / (k + 1);
  }
  return sum;
}

// A ticket and a target are 256-bit little-endian integers.
function meetsTarget(ticket, target) {
  for (let i = 31; i >= 0; i--) {
    if (ticket[i] !== target[i]) return ticket[i] < target[i];
  }
  return true;
}

// ── datasets ───────────────────────────────────────────────────────────────

// The network's dataset file (spec/shares.md section 11.6): "NMDS", a count,
// then the datasets. Only NMD2 datasets (the convolutional networks) are read;
// the main network has no other kind.
function parseDatasets(buf) {
  if (buf.length < 8 || buf.toString('latin1', 0, 4) !== 'NMDS') throw new Error('not a NEB dataset file');
  const count = buf.readUInt32LE(4);
  const sets = [];
  let at = 8;
  for (let s = 0; s < count; s++) {
    if (at + 12 > buf.length) throw new Error('the NEB dataset file is cut short');
    const magic = buf.toString('latin1', at, at + 4);
    if (magic !== 'NMD2') throw new Error('dataset ' + s + ' is ' + magic + '; this miner reads NMD2 only');
    const arch = buf.readUInt32LE(at + 4);
    const n = buf.readUInt32LE(at + 8);
    const A = archOf(arch);
    if (!A) throw new Error('dataset ' + s + ' names an unknown network (' + arch + ')');
    const rec = A.nin + 1;
    at += 12;
    if (n < 1 || at + n * rec > buf.length) throw new Error('the NEB dataset file is cut short');
    const values = Buffer.alloc(n * A.nin);
    const labels = Buffer.alloc(n);
    for (let i = 0; i < n; i++) {
      buf.copy(values, i * A.nin, at + i * rec, at + i * rec + A.nin);
      labels[i] = buf[at + i * rec + A.nin];
    }
    at += n * rec;
    sets.push({ arch, n, values, labels });
  }
  if (at !== buf.length) throw new Error('the NEB dataset file has bytes after its last dataset');
  return sets;
}

// The test dataset of spec/nebhash-v2.md section 8 (2,048 samples).
const SYN2_N = 2048;
function syn2Dataset(arch) {
  const A = archOf(arch);
  const values = Buffer.alloc(SYN2_N * A.nin);
  const labels = Buffer.alloc(SYN2_N);
  const protos = [0, 1].map((c) => chacha8.stream(
    blake3.hash(Buffer.concat([Buffer.from('neb-syn2-proto'), Buffer.from([arch, c])])), A.nin));
  const le = Buffer.alloc(4);
  for (let i = 0; i < SYN2_N; i++) {
    const c = i % 2;
    le.writeUInt32LE(i);
    const noise = chacha8.stream(blake3.hash(Buffer.concat([Buffer.from('neb-syn2-sample'), Buffer.from([arch]), le])), A.nin);
    for (let p = 0; p < A.nin; p++) values[i * A.nin + p] = clipi((protos[c][p] & 7) + (noise[p] & 3) - 1, 0, 7);
    labels[i] = c;
  }
  return { arch, n: SYN2_N, values, labels };
}

module.exports = {
  NB, CLIP, KEP1, RET2, CONV1D, CONV2D, LINEAR, SYN2_N,
  archOf, rshift, unpackWeights, packWeights, weightsHash, residualHash, modelFromPacked, modelInit, modelApply,
  makeWork, sampleGrad, backwardLayer, shareIndices, digestOf, ticketOf, shareEval, liveSamples, liveCount, candidateShare, meetsTarget, parseDatasets, syn2Dataset,
};
