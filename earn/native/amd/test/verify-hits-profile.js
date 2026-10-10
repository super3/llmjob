// node verify-hits-profile.js <core.node> [seconds=40] [targetBits=252] [m=256] [n=512] [fill=1]
//
// probes/verify-hits.js at a profile of your choosing. That script is fixed to the mainnet
// profile, which is right on a GPU and far too big for the HIP-CPU core, where the kernels
// run on the CPU. This one takes m, n and the operand fill (1 = constant, 0 = hashed) and
// is otherwise the same check: the first 400 hits the core reports are recomputed from
// scratch in JS the way the pool's verifier does -- Merkle proofs, seed chain, noise, the
// cumulative fold, the transcript hash -- and must match the core's jackpot hash exactly.
// It passes with no mismatch, at least 20 hits checked, and at least 3 salts seen, so the
// run crosses restamps.
//
// m must be a power of two and a multiple of 128 (the AMD matrix folds' tile), n a power
// of two and a multiple of 256. PEARL_AMD_FOLD picks the fold, as it does in the app.
'use strict';
const path = require('path');
const R = path.join(__dirname, '..', '..', '..', 'src', 'shared', 'miner') + path.sep;
const { PROFILE, buildConfig52, regionToTile, meetsTarget, SEED_SALT_A, SEED_SALT_B, bindMessage } = require(R + 'pearlhash');
const { hash, keyedHash } = require(R + 'blake3');
const { buildShareProof } = require(R + 'shareProof');
const { foldTranscript, transcriptBytes } = require(R + 'reference');
const { computeNoiseForIndices } = require(R + 'noise');

const file = path.resolve(process.argv[2]);
const secs = Number(process.argv[3] || 40);
const bits = BigInt(process.argv[4] || 252);
const fill = Number(process.argv[7] === undefined ? 1 : process.argv[7]);
const profile = {
  ...PROFILE,
  m: Number(process.argv[5] || 256),
  n: Number(process.argv[6] || 512),
  operandFill: fill ? 'constant' : 'hashed',
  operandFillCode: fill ? 1 : 0,
};
const { k, rank, m, n } = profile;
const addon = require(file);
const core = addon.createCore(profile, {});
const header = Buffer.alloc(76, 0x5A);
const jobKey = hash(Buffer.concat([header, buildConfig52(profile)]));
const target = 2n ** bits;
let hits = 0, ok = 0, bad = 0;
const salts = new Set();
const fails = [];

function rowsFromLeaves(side, want) {
  // Leaves are 1024-byte chunks in leafIndices order; row r spans bytes [r*k, r*k+k).
  const map = new Map();
  Array.from(side.leafIndices).forEach((li, i) => map.set(li, side.leafData.subarray(i * 1024, i * 1024 + 1024)));
  return want.map((r) => {
    const out = new Int8Array(k);
    for (let b = 0; b < k; b += 1024) {
      const off = r * k + b;
      const leaf = map.get(Math.floor(off / 1024));
      if (!leaf) throw new Error('missing leaf for row ' + r);
      out.set(new Int8Array(leaf.buffer, leaf.byteOffset + (off % 1024), 1024), b);
    }
    return out;
  });
}

function check(hit) {
  if (!buildShareProof(hit, jobKey, profile)) return 'buildShareProof rejected (Merkle / leaf indices)';
  const rootA = Buffer.from(hit.proofA.root), rootB = Buffer.from(hit.proofBt.root);
  const boundA = keyedHash(SEED_SALT_A, bindMessage(rootA, m)), boundB = keyedHash(SEED_SALT_B, bindMessage(rootB, n));
  const bSeed = hash(Buffer.concat([jobKey, boundB])), aSeed = hash(Buffer.concat([bSeed, boundA]));
  if (aSeed.toString('hex') !== hit.aSeed || bSeed.toString('hex') !== hit.bSeed) return 'seed chain mismatch';
  const { rows, cols } = regionToTile(hit.nonce, profile);
  const Ar = rowsFromLeaves(hit.proofA, rows);
  const Bc = rowsFromLeaves(hit.proofBt, cols);
  for (const r of Ar.concat(Bc)) for (const v of r) if (v < -63 || v > 63) return 'operand outside int7';
  const { noiseA, noiseB } = computeNoiseForIndices({ k, rank, aSeed, bSeed, rowIndices: rows, colIndices: cols });
  const A = new Int8Array(16 * k), Bt = new Int8Array(16 * k);
  Ar.forEach((r, i) => A.set(r, i * k));
  Bc.forEach((c, i) => Bt.set(c, i * k));
  const idx = Array.from({ length: 16 }, (_, i) => i);
  const jp = foldTranscript({ A, Bt, noiseA, noiseB, rows: idx, cols: idx, k, rank });
  const h = keyedHash(aSeed, transcriptBytes(jp));
  if (!h.equals(Buffer.from(hit.jackpotHash))) return 'jackpot hash mismatch (fold/transcript)';
  if (!meetsTarget(h, target)) return 'reported hit does not meet target';
  return null;
}

core.on('hit', (hit) => {
  hits++;
  salts.add(hit.salt);
  if (ok + bad >= 400) return;
  let why;
  try { why = check(hit); } catch (e) { why = 'exception: ' + e.message; }
  if (why) {
    bad++;
    if (fails.length < 5) fails.push({ nonce: hit.nonce, salt: hit.salt, why });
  } else {
    ok++;
  }
});
core.on('error', (e) => { console.log(JSON.stringify({ error: String(e) })); process.exit(2); });
let rate = 0;
core.on('hashrate', (th) => { rate = th; });
core.setJob({ header, target, jobId: 'verify' });
setTimeout(() => {
  core.stop();
  const res = {
    file: path.basename(file), fold: process.env.PEARL_AMD_FOLD || 'default', m, n, fill: profile.operandFill,
    hits, verified: ok, failed: bad, salts: salts.size, thPerSec: rate, fails,
    PASS: bad === 0 && ok >= 20 && salts.size >= 3,
  };
  console.log(JSON.stringify(res));
  process.exit(res.PASS ? 0 : 1);
}, secs * 1000);
