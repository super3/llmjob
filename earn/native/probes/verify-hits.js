// node verify-hits.js <pearl_core.node> [seconds=40] [targetBits=232]
// End-to-end correctness gate, independent of the pool: every hit the core
// reports is recomputed from scratch in JS the way the pool's verifier would --
// Merkle proofs, seed chain, noise, the cumulative fold, the transcript hash --
// and must match the device's jackpot hash exactly. The target is set easy
// enough to hit about once a batch so the run crosses many operand redraws.
'use strict';
const path = require('path');
const R = path.join(__dirname, '..', '..', 'src', 'shared', 'miner') + path.sep;
const { PROFILE, buildConfig52, regionToTile, meetsTarget, SEED_SALT_A, SEED_SALT_B, bindMessage } = require(R + 'pearlhash');
const { hash, keyedHash } = require(R + 'blake3');
const { buildShareProof } = require(R + 'shareProof');
const { foldTranscript, transcriptBytes } = require(R + 'reference');
const { computeNoiseForIndices } = require(R + 'noise');
const { leafIndicesFromRows } = require(R + 'merkle');
const file = path.resolve(process.argv[2]);
const secs = Number(process.argv[3] || 40);
const bits = BigInt(process.argv[4] || 232);
const addon = require(file);
const core = addon.createCore(PROFILE, {});
const header = Buffer.alloc(76, 0x5A);
const jobKey = hash(Buffer.concat([header, buildConfig52(PROFILE)]));
const target = 2n ** bits;
const tb = Buffer.alloc(32); { let t = target; for (let i = 31; i >= 0; i--) { tb[i] = Number(t & 0xffn); t >>= 8n; } }
const { k, rank, m, n } = PROFILE;
let hits = 0, ok = 0, bad = 0; const salts = new Set(); const fails = [];
function rowsFromLeaves(side, idxs, want) {
  // leaves are 1024-byte chunks in leafIndices order; row r spans bytes [r*k, r*k+k)
  const map = new Map(); const L = Array.from(side.leafIndices);
  L.forEach((li, i) => map.set(li, side.leafData.subarray(i * 1024, i * 1024 + 1024)));
  return want.map((r) => { const out = new Int8Array(k); for (let b = 0; b < k; b += 1024) { const off = r * k + b; const leaf = map.get(Math.floor(off / 1024)); if (!leaf) throw new Error('missing leaf for row ' + r); out.set(new Int8Array(leaf.buffer, leaf.byteOffset + (off % 1024), 1024), b); } return out; });
}
function check(hit) {
  if (!buildShareProof(hit, jobKey, PROFILE)) return 'buildShareProof rejected (Merkle / leaf indices)';
  const rootA = Buffer.from(hit.proofA.root), rootB = Buffer.from(hit.proofBt.root);
  const boundA = keyedHash(SEED_SALT_A, bindMessage(rootA, m)), boundB = keyedHash(SEED_SALT_B, bindMessage(rootB, n));
  const bSeed = hash(Buffer.concat([jobKey, boundB])), aSeed = hash(Buffer.concat([bSeed, boundA]));
  if (aSeed.toString('hex') !== hit.aSeed || bSeed.toString('hex') !== hit.bSeed) return 'seed chain mismatch';
  const { rows, cols } = regionToTile(hit.nonce, PROFILE);
  const Ar = rowsFromLeaves(hit.proofA, leafIndicesFromRows(rows, k), rows);
  const Bc = rowsFromLeaves(hit.proofBt, leafIndicesFromRows(cols, k), cols);
  for (const r of Ar.concat(Bc)) for (const v of r) if (v < -63 || v > 63) return 'operand outside int7';
  const { noiseA, noiseB } = computeNoiseForIndices({ k, rank, aSeed, bSeed, rowIndices: rows, colIndices: cols });
  const A = new Int8Array(16 * k), Bt = new Int8Array(16 * k);
  Ar.forEach((r, i) => A.set(r, i * k)); Bc.forEach((c, i) => Bt.set(c, i * k));
  const idx = Array.from({ length: 16 }, (_, i) => i);
  const jp = foldTranscript({ A, Bt, noiseA, noiseB, rows: idx, cols: idx, k, rank });
  const h = keyedHash(aSeed, transcriptBytes(jp));
  if (!h.equals(Buffer.from(hit.jackpotHash))) return 'jackpot hash mismatch (fold/transcript)';
  if (!meetsTarget(h, target)) return 'reported hit does not meet target';
  return null;
}
core.on('hit', (hit) => {
  hits++; salts.add(hit.salt);
  if (ok + bad >= 400) return;
  let why; try { why = check(hit); } catch (e) { why = 'exception: ' + e.message; }
  if (why) { bad++; if (fails.length < 5) fails.push({ nonce: hit.nonce, salt: hit.salt, why }); } else ok++;
});
core.on('error', (e) => { console.log(JSON.stringify({ error: String(e) })); process.exit(2); });
core.on('hashrate', () => {});
core.setJob({ header, target, jobId: 'verify' });
setTimeout(() => { core.stop(); const res = { file: path.basename(path.dirname(file)), hits, verified: ok, failed: bad, salts: salts.size, fails, PASS: bad === 0 && ok >= 20 && salts.size >= 3 }; console.log(JSON.stringify(res)); process.exit(res.PASS ? 0 : 1); }, secs * 1000);
