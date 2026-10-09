// node verify-hits.js <core.node> [--hits 400] [--seconds 600] [--bits 232]
//                     [--m M] [--n N] [--col-batch C] [--hashed]
//
// probes/verify-hits.js for any core and any m and n. The first --hits hits the
// core reports are recomputed from scratch in JS the way the pool's verifier
// would: Merkle proofs, seed chain, noise, the cumulative fold, the transcript
// hash. Each must match the core's jackpot hash exactly and meet the target. It
// stops once --hits hits are checked, or after --seconds.
//
// m and n are the miner's own choice (they are bound into the seeds by cert-v3,
// not into config52), so a hit at a small m and n is a real Pearl hit, checked
// by the same code. That is what lets the SYCL CPU device run this here: at
// m = n = 4096 a salt is 65,536 regions instead of 2^31. k and rank are the
// mainnet protocol values; the profile is PROFILE with only m, n, colBatch and
// (with --hashed) the operand fill changed.
//
// Prints one JSON line. PASS: every checked hit matched, --hits of them were
// checked, and they came from at least 3 salts (operand redraws).
'use strict';
const path = require('path');
const R = path.join(__dirname, '..', '..', 'src', 'shared', 'miner') + path.sep;
const { PROFILE, buildConfig52, regionToTile, meetsTarget, SEED_SALT_A, SEED_SALT_B, bindMessage } = require(R + 'pearlhash');
const { hash, keyedHash } = require(R + 'blake3');
const { buildShareProof } = require(R + 'shareProof');
const { foldTranscript, transcriptBytes } = require(R + 'reference');
const { computeNoiseForIndices } = require(R + 'noise');

function parseArgs(argv) {
  const o = { file: null, hits: 400, seconds: 600, bits: 232, m: PROFILE.m, n: PROFILE.n, colBatch: PROFILE.colBatch, hashed: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const num = () => Number(argv[++i]);
    if (a === '--hits') o.hits = num();
    else if (a === '--seconds') o.seconds = num();
    else if (a === '--bits') o.bits = num();
    else if (a === '--m') o.m = num();
    else if (a === '--n') o.n = num();
    else if (a === '--col-batch') o.colBatch = num();
    else if (a === '--hashed') o.hashed = true;
    else if (!o.file) o.file = a;
    else throw new Error('unknown argument ' + a);
  }
  if (!o.file) throw new Error('usage: node verify-hits.js <core.node> [--hits N] [--seconds S] [--bits B] [--m M] [--n N] [--col-batch C] [--hashed]');
  return o;
}

const opt = parseArgs(process.argv.slice(2));
const profile = Object.assign({}, PROFILE, { m: opt.m, n: opt.n, colBatch: opt.colBatch });
if (opt.hashed) Object.assign(profile, { operandFill: 'hashed', operandFillCode: 0 });
const addon = require(path.resolve(opt.file));
const core = addon.createCore(profile, {});
const header = Buffer.alloc(76, 0x5A);
const jobKey = hash(Buffer.concat([header, buildConfig52(profile)]));
const target = 2n ** BigInt(opt.bits);
const { k, rank, m, n } = profile;
let hits = 0, ok = 0, bad = 0, rate = 0, rates = 0;
const salts = new Set();
const fails = [];
const t0 = Date.now();

function rowsFromLeaves(side, want) {
  // leaves are 1024-byte chunks in leafIndices order; row r spans bytes [r*k, r*k+k)
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
  if (!Buffer.from(transcriptBytes(jp)).equals(Buffer.from(hit.proof))) return 'transcript mismatch (fold)';
  const h = keyedHash(aSeed, transcriptBytes(jp));
  if (!h.equals(Buffer.from(hit.jackpotHash))) return 'jackpot hash mismatch (transcript hash)';
  if (!meetsTarget(h, target)) return 'reported hit does not meet target';
  return null;
}

let done = false;
function finish() {
  if (done) return;
  done = true;
  core.stop();
  const res = {
    file: path.basename(opt.file), device: core.device && core.device.name, fold: process.env.PEARL_SYCL_FOLD || 'auto',
    m, n, colBatch: opt.colBatch, bits: opt.bits, fill: opt.hashed ? 'hashed' : 'constant',
    hits, verified: ok, failed: bad, salts: salts.size,
    seconds: Math.round((Date.now() - t0) / 100) / 10, meanThs: rates ? Math.round(rate / rates * 1000) / 1000 : 0,
    fails, PASS: bad === 0 && ok >= opt.hits && salts.size >= 3,
  };
  console.log(JSON.stringify(res));
  process.exit(res.PASS ? 0 : 1);
}

core.on('hit', (hit) => {
  hits++;
  if (ok + bad >= opt.hits) return finish();
  salts.add(hit.salt);
  let why;
  try { why = check(hit); } catch (e) { why = 'exception: ' + e.message; }
  if (why) { bad++; if (fails.length < 5) fails.push({ nonce: hit.nonce, salt: hit.salt, why }); } else ok++;
  if (ok + bad >= opt.hits) finish();
});
core.on('error', (e) => { console.log(JSON.stringify({ error: String(e) })); process.exit(2); });
core.on('hashrate', (r) => { rate += r; rates++; });
core.setJob({ header, target, jobId: 'verify' });
setTimeout(finish, opt.seconds * 1000);
