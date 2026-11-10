// node verify-hits.js <core.node> [--hits 400] [--seconds 600] [--bits 232]
//                     [--m M] [--n N] [--col-batch C] [--hashed]
//
// probes/verify-hits.js for any core and any m and n. The first --hits hits the
// core reports are recomputed from scratch in JS the way the pool's verifier
// would: Merkle proofs, seed chain, noise, the cumulative fold, the transcript
// hash. Each must match the core's jackpot hash exactly and meet the target. It
// stops once --hits hits are checked, or after --seconds.
//
// The recompute is shared/miner/checkHit.js, the check the app runs before it
// sends a hit of a card whose fold has not run on its kind of card. Every Intel
// GPU is such a card today. Since every hit is checked here, the core is opened
// with checkHits, as the app opens it, and the JSON says what the core
// answered (untested).
//
// m and n are the miner's own choice (they are bound into the seeds by cert-v3,
// not into config52), so a hit at a small m and n is a real Pearl hit, checked
// by the same code. That is what lets the SYCL CPU device run this here: at
// m = n = 4096 a salt is 65,536 regions instead of 2^31. k and rank are the
// mainnet protocol values; the profile is PROFILE with only m, n, colBatch and
// (with --hashed) the operand fill changed.
//
// PEARL_GPU_INDEX as one number pins the card, so each card of a rig can be
// checked in turn. A list (the CLI takes one) or nothing leaves the choice to
// the core.
//
// Prints one JSON line. PASS: every checked hit matched, --hits of them were
// checked, and they came from at least 3 salts (operand redraws).
//
// This shows that the hits the core reports are real. It cannot show that none
// went missing; verify.sh's fold check does that, region by region. As a rough
// cross-check the line also has expectedHits, the work the core reports doing
// (each 'hashrate' event times the time since the one before) over the work a
// hit takes on average, and hitRatio, hits / expectedHits. With 400 hits the
// ratio should be within about 0.85-1.15; well under that means the fold drops
// hits that it counts as searched.
'use strict';
const path = require('path');
const R = path.join(__dirname, '..', '..', 'src', 'shared', 'miner') + path.sep;
const { PROFILE, buildConfig52, meetsTarget } = require(R + 'pearlhash');
const { hash } = require(R + 'blake3');
const { buildShareProof } = require(R + 'shareProof');
const { checkHit } = require(R + 'checkHit');

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
const coreOpts = { checkHits: true };
if (/^\d+$/.test(process.env.PEARL_GPU_INDEX || '')) coreOpts.deviceIndex = Number(process.env.PEARL_GPU_INDEX);
let core;
try {
  core = addon.createCore(profile, coreOpts);
} catch (e) {
  // The core's own words: no such card, no kernels for it in this build, too little memory.
  console.log(JSON.stringify({ error: String((e && e.message) || e) }));
  process.exit(2);
}
const header = Buffer.alloc(76, 0x5A);
const jobKey = hash(Buffer.concat([header, buildConfig52(profile)]));
const target = 2n ** BigInt(opt.bits);
const { k, m, n } = profile;
let hits = 0, ok = 0, bad = 0, rate = 0, rates = 0;
// MACs the core reports, from its rate events: each is the rate over the time
// since the previous one (the first window starts when the search does).
let work = 0, lastRate = 0, lastAt = 0;
const salts = new Set();
const fails = [];
const t0 = Date.now();

function check(hit) {
  if (!buildShareProof(hit, jobKey, profile)) return 'buildShareProof rejected (Merkle / leaf indices)';
  const why = checkHit(hit, jobKey, profile);
  if (why) return why;
  if (!meetsTarget(Buffer.from(hit.jackpotHash), target)) return 'reported hit does not meet target';
  return null;
}

let done = false;
function finish() {
  if (done) return;
  done = true;
  const device = core.device;
  const untested = core.untested;
  core.stop();
  const now = Date.now();
  const seconds = (now - t0) / 1000;
  const meanThs = rates ? rate / rates : 0;
  work += lastRate * 1e12 * (now - (lastAt || now)) / 1000;  // since the last event
  // regions searched = MACs / (16 x 16 x k); a region hits with chance target / 2^256
  const expected = work / (256 * k) * Math.pow(2, opt.bits - 256);
  const res = {
    file: path.basename(opt.file), device, untested, fold: process.env.PEARL_SYCL_FOLD || 'auto',
    m, n, colBatch: opt.colBatch, bits: opt.bits, fill: opt.hashed ? 'hashed' : 'constant',
    hits, verified: ok, failed: bad, salts: salts.size,
    seconds: Math.round(seconds * 10) / 10, meanThs: Math.round(meanThs * 1000) / 1000,
    expectedHits: Math.round(expected * 10) / 10, hitRatio: expected ? Math.round(hits / expected * 100) / 100 : null,
    fails, PASS: bad === 0 && ok >= opt.hits && salts.size >= 3,
  };
  console.log(JSON.stringify(res));
  process.exit(res.PASS ? 0 : 1);
}

// A hit is checked in a turn of the event loop of its own, not inside its
// 'hit' event. A check takes milliseconds. Done inside the event, with hits
// arriving as fast as they were checked, it kept Node in the hit events and the
// 'hashrate' events waited behind them: a run on the CPU device, where the
// kernels and this script share the cores, ended with no rate, so no hitRatio.
const queue = [];
let draining = false;
function drain() {
  draining = false;
  if (done) return;
  const hit = queue.shift();
  if (hit) {
    salts.add(hit.salt);
    let why;
    try { why = check(hit); } catch (e) { why = 'exception: ' + e.message; }
    if (why) { bad++; if (fails.length < 5) fails.push({ nonce: hit.nonce, salt: hit.salt, why }); } else ok++;
    if (ok + bad >= opt.hits) return finish();
  }
  if (queue.length) { draining = true; setImmediate(drain); }
}
core.on('hit', (hit) => {
  hits++;
  if (done || ok + bad + queue.length >= opt.hits) return;
  queue.push(hit);
  if (!draining) { draining = true; setImmediate(drain); }
});
core.on('error', (e) => { console.log(JSON.stringify({ error: String(e) })); process.exit(2); });
core.on('hashrate', (r) => {
  const now = Date.now();
  work += r * 1e12 * (now - (lastAt || t0)) / 1000;
  lastRate = r;
  lastAt = now;
  rate += r;
  rates++;
});
core.setJob({ header, target, jobId: 'verify' });
setTimeout(finish, opt.seconds * 1000);
