// node verify-hits-profile.js <core.node> [seconds=40] [targetBits=252] [m=256] [n=512] [fill=1]
//
// probes/verify-hits.js at a profile of your choosing. That script is fixed to the mainnet
// profile, which is right on a GPU and far too big for the HIP-CPU core, where the kernels
// run on the CPU. This one takes m, n and the operand fill (1 = constant, 0 = hashed) and
// is otherwise the same check: the first 400 hits the core reports are recomputed from
// scratch in JS the way the pool's verifier does -- Merkle proofs, seed chain, noise, the
// cumulative fold, the transcript hash -- and must match the core's jackpot hash exactly.
// It passes with no mismatch, at least 20 hits checked, and at least 3 salts seen, so the
// run crosses restamps. The recompute is shared/miner/checkHit.js, as in verify-hits.js, and
// the core is opened with checkHits for the same reason.
//
// m must be a power of two and a multiple of 128 (the AMD matrix folds' tile), n a power
// of two and a multiple of 256. PEARL_AMD_FOLD picks the fold, as it does in the app.
//
// PEARL_CARD_K=1 opens the core as the app does (cardK): the core may pick a larger k for the
// card (k 4096 where its code object has the wrap readout), and everything here then uses the k
// it reports (core.k). The JSON line names the k checked. PEARL_SHARE_DUMP=file writes each hit
// that passes as a share, one JSON object a line ({ header, target, plain_proof, k }), for the
// reference verifier (probes/refverify.rs); the target is the pool's, before the k scaling.
'use strict';
const fs = require('fs');
const path = require('path');
const R = path.join(__dirname, '..', '..', '..', 'src', 'shared', 'miner') + path.sep;
const {
  PROFILE, buildConfig52, meetsTarget, profileForK, penalizedAdjustmentFactor,
} = require(R + 'pearlhash');
const { hash } = require(R + 'blake3');
const { buildShareProof } = require(R + 'shareProof');
const { checkHit } = require(R + 'checkHit');

const file = path.resolve(process.argv[2]);
const secs = Number(process.argv[3] || 40);
const bits = BigInt(process.argv[4] || 252);
const fill = Number(process.argv[7] === undefined ? 1 : process.argv[7]);
const asked = {
  ...PROFILE,
  m: Number(process.argv[5] || 256),
  n: Number(process.argv[6] || 512),
  operandFill: fill ? 'constant' : 'hashed',
  operandFillCode: fill ? 1 : 0,
};
const cardK = process.env.PEARL_CARD_K === '1';
const dumpTo = process.env.PEARL_SHARE_DUMP || '';
const addon = require(file);
const core = addon.createCore(asked, cardK ? { cardK: true, checkHits: true } : { checkHits: true });
// The k the core mines: the one asked, unless cardK let it pick another.
const profile = cardK && core.k !== undefined ? profileForK(asked, core.k) : asked;
if (!profile) {
  core.stop();
  console.log(JSON.stringify({ error: 'the core chose k ' + core.k + ', which a share can\'t carry' }));
  process.exit(2);
}
const { k, m, n } = profile;
const header = Buffer.alloc(76, 0x5A);
const jobKey = hash(Buffer.concat([header, buildConfig52(profile)]));
const target = 2n ** bits;
// The pool's target for this bound: a pool scales its target by tile x (k / rank) x 128 (shareBound).
const poolTarget = target / BigInt(penalizedAdjustmentFactor(profile));
let hits = 0, ok = 0, bad = 0;
const salts = new Set();
const fails = [];

let share = null;
function check(hit) {
  share = buildShareProof(hit, jobKey, profile);
  if (!share) return 'buildShareProof rejected (Merkle / leaf indices)';
  const why = checkHit(hit, jobKey, profile);
  if (why) return why;
  if (!meetsTarget(Buffer.from(hit.jackpotHash), target)) return 'reported hit does not meet target';
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
    if (dumpTo) {
      fs.appendFileSync(dumpTo, JSON.stringify({
        header: header.toString('hex'), target: poolTarget.toString(16).padStart(64, '0'), plain_proof: share, k,
      }) + '\n');
    }
  }
});
core.on('error', (e) => { console.log(JSON.stringify({ error: String(e) })); process.exit(2); });
let rate = 0;
core.on('hashrate', (th) => { rate = th; });
core.setJob({ header, target, jobId: 'verify' });
setTimeout(() => {
  core.stop();
  const res = {
    file: path.basename(file), fold: process.env.PEARL_AMD_FOLD || 'default', k, m, n, fill: profile.operandFill,
    hits, verified: ok, failed: bad, salts: salts.size, thPerSec: rate, fails,
    PASS: bad === 0 && ok >= 20 && salts.size >= 3,
  };
  console.log(JSON.stringify(res));
  process.exit(res.PASS ? 0 : 1);
}, secs * 1000);
