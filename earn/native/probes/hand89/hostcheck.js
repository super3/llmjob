'use strict';
// node hostcheck.js <gprobe.exe> <hand cubin or -> <k> <tiles a block> <tiles to check> [order=1] [rot=1] [seed=1] [hash=0]
//
// A job's own noise through the card's fold, and every hit of it through the reference check.
//   1. A made-up header, operands filled with 63 (a salt stamp in a row no tile uses), the profile of the fold's
//      proof tile (one thread's sums: 12 rows by 16 columns). The seeds, the dense factors and the rank indices
//      come from the repo's own port of the protocol (earn/src/shared/miner), which pools accept.
//   2. gprobe folds them on the card (PD_INPUT) and writes every thread's transcript of the checked tiles
//      (PD_DUMPFILE). Its own CPU check runs too.
//   3. Here every transcript is hashed with the job's key; a hash under the bound is a hit; every hit goes
//      through a port of the pool's check (pattern from the hit's own indices, job key from that pattern, the
//      sanity checks, operand range, noise, compute_jackpot, hash, bound: HC_LIB, from the Pearl project's
//      public verifier). A sample of the other lanes' transcripts is compared with compute_jackpot too.
//   4. With hash=1 (a gprobe built with -DGP_HASH=1) the card hashes every thread's transcript of every tile
//      itself and lists its hits. In the checked tiles its list must be the list made here; of its hits in the
//      other tiles up to 400 go through the same check, their transcripts refolded here from the job's inputs.
// PROBE ONLY: no share is built or sent and no Merkle proof is made.
const fs = require('fs');
const path = require('path');
const cp = require('child_process');

// HC_LIB: the prototype library with the ports of the Pearl project's public verifier (PeriodicPattern, the
// sanity checks, compute_jackpot) and the fold's tiling. It was written for the maths review of 11 November and
// is not in this repo; it loads the repo's own port of the protocol from ALGO_REPO.
process.env.ALGO_REPO = process.env.ALGO_REPO || path.join(__dirname, '..', '..', '..', 'src', 'shared', 'miner');
if (!process.env.HC_LIB) {
  console.log('hostcheck: set HC_LIB to the prototype library (lib.js of the maths review)');
  process.exit(2);
}
const L = require(path.resolve(process.env.HC_LIB));
const { noise, reference, pearlhash } = L;

const [exe, cubin, kArg, tpbArg, checkArg, orderArg, rotArg, seedArg, hashArg] = process.argv.slice(2);
const HASH = Number(hashArg || 0);       // 1: the build hashes on the card (-DGP_HASH=1) and lists its hits
const K = Number(kArg), TPB = Number(tpbArg), CHECK = Number(checkArg);
const ORDER = orderArg === undefined ? 1 : Number(orderArg), ROT = rotArg === undefined ? 1 : Number(rotArg);
const SEED = seedArg === undefined ? 1 : Number(seedArg);
const RANK = 128, FILL = 63, RG = 8, BLOCKS = Number(process.env.GP_BLOCKS || 128);
const tiling = L.makeTiling({ m16Side: 'A', w16: 2, w8: 4, t16: 6, t8: 8, interleaved: false });
const BM = tiling.rows, BN = tiling.cols;                 // 192 x 256
const tiles = BLOCKS * TPB;
const CB = Math.min(Math.ceil(tiles / RG), 1024);
const ROWS = RG * BM, COLS = CB * BN, EXTRA = 16, M = ROWS + EXTRA, N = COLS;

let rngS = (SEED * 2654435761) >>> 0;
const rnd = () => { rngS ^= rngS << 13; rngS >>>= 0; rngS ^= rngS >>> 17; rngS ^= rngS << 5; rngS >>>= 0; return rngS; };
const out = { k: K, m: M, n: N, tilesChecked: CHECK, order: ORDER, rot: ROT, seed: SEED };
const t0 = Date.now();

// The proof tile's pattern, from lane 0 of warp 0.
const tile0 = L.laneTile(tiling, tiling.warps[0], 0);
const rowsShape = L.patFromList(tile0.rows.map((x) => x - tile0.rows[0]));
const colsShape = L.patFromList(tile0.cols.map((x) => x - tile0.cols[0]));
const rowsPat = L.patToList(rowsShape), colsPat = L.patToList(colsShape);
out.tile = `${rowsPat.length}x${colsPat.length}`;

// The job.
const header = Buffer.alloc(76);
for (let i = 0; i < 76; i++) header[i] = rnd() & 255;
const profile = { k: K, rank: RANK, mmaType: 0, rows: rowsPat, cols: colsPat, m: M, n: N, seedDerivation: 'salted' };
const A = new Int8Array(M * K).fill(FILL);
const Bt = new Int8Array(N * K).fill(FILL);
for (let i = 0; i < 11; i++) A[ROWS * K + i] = (rnd() % 127) - 63;      // the stamp, in a row no tile uses
const key = reference.jobKey(header, profile);
const { aSeed, bSeed } = reference.deriveSeeds(key, A, Bt, profile);
const allRows = Array.from({ length: ROWS }, (_, i) => i), allCols = Array.from({ length: COLS }, (_, i) => i);
const eAL = noise.generateUniformRandomMatrix(noise.SEED_LABEL_A, aSeed, allRows, RANK);    // [row][p], -32..31
const permA = noise.generatePermutationMatrix(noise.SEED_LABEL_A, aSeed, K, RANK);          // [2l] plus, [2l+1] minus
const permB = noise.generatePermutationMatrix(noise.SEED_LABEL_B, bSeed, K, RANK);
const eBR = noise.generateUniformRandomMatrix(noise.SEED_LABEL_B, bSeed, allCols, RANK);    // [col][p]
out.msJob = Date.now() - t0;

// gprobe's input.
const dir = fs.mkdtempSync(path.join(require('os').tmpdir(), 'hostcheck-'));
const inFile = path.join(dir, 'in.bin'), dumpFile = path.join(dir, 'dump.bin');
{
  const buf = Buffer.alloc(16 + (ROWS + COLS) * RANK + 4 * K);
  buf.writeUInt32LE(0x4e495047, 0); buf.writeUInt32LE(ROWS, 4); buf.writeUInt32LE(COLS, 8); buf.writeUInt32LE(K, 12);
  let o = 16;
  for (let r = 0; r < ROWS; r++) for (let p = 0; p < RANK; p++) buf[o++] = eAL[r][p] + 32;
  for (let c = 0; c < COLS; c++) for (let p = 0; p < RANK; p++) buf[o++] = eBR[c][p] + 32;
  for (const perm of [permA, permB]) for (let sg = 0; sg < 2; sg++) for (let l = 0; l < K; l++) buf[o++] = perm[2 * l + sg];
  fs.writeFileSync(inFile, buf);
}
// The bound, and the card's key and bound when it hashes.
const sane = L.sanityCheck({ k: K, r: RANK, h: rowsPat.length, w: colsPat.length, m: M, n: N, tRows: tile0.rows[0],
  tCols: tile0.cols[0], rowsShape, colsShape });
const target = (1n << 252n) / BigInt(sane.factor);          // about one lane in sixteen
const bound = target * BigInt(sane.factor);
const hashFile = path.join(dir, 'hash.bin'), hitFile = path.join(dir, 'hits.bin');
{
  const kb = Buffer.alloc(64);
  Buffer.from(aSeed).copy(kb, 0);
  let x = bound;
  for (let i = 0; i < 32; i++) { kb[32 + i] = Number(x & 255n); x >>= 8n; }
  fs.writeFileSync(hashFile, kb);
}
const env = { ...process.env, PD_INPUT: inFile, PD_DUMPFILE: dumpFile };
delete env.PD_HASH;
if (HASH) { env.PD_HASH = hashFile; env.PD_HITFILE = hitFile; }
delete env.PD_CUBIN;
if (cubin && cubin !== '-') env.PD_CUBIN = path.resolve(cubin);
const t1 = Date.now();
const run = cp.spawnSync(path.resolve(exe), ['hostcheck', String(K), String(TPB), '0', String(CHECK), String(ORDER), '0', String(ROT)],
  { env, encoding: 'utf8' });
out.msCard = Date.now() - t1;
const line = (run.stdout || '').trim().split('\n').pop() || '';
out.gprobe = (line.match(/pipe\s+[\d.]+%/) || [''])[0] + ' ' + (line.match(/check \d+\/\d+/) || ['no check line'])[0];
const fail = (m) => { console.log('FAIL: ' + m); process.exitCode = 1; };
if (run.status !== 0 || !fs.existsSync(dumpFile)) {
  fail('gprobe: ' + (run.stderr || '').trim().split('\n').slice(0, 3).join(' | '));
  console.log(JSON.stringify(out));
  try { fs.rmSync(dir, { recursive: true, force: true }); } catch (e) { /* the files are small */ }
  process.exit(1);
}
{
  const m = line.match(/check (\d+)\/(\d+)/);
  if (!m || m[1] !== m[2]) fail("gprobe's own CPU check: " + (m ? m[0] : 'none'));
}
const dumpBuf = fs.readFileSync(dumpFile);
const dump = new Uint32Array(dumpBuf.buffer, dumpBuf.byteOffset, dumpBuf.length >> 2);
if (dump.length !== CHECK * 256 * 16) fail(`the dump holds ${dump.length} words, not ${CHECK * 256 * 16}`);

// Every thread of every checked tile: its tile's rows and columns in the whole matrix, its hash.
const lanes = [];
for (let v = 0; v < CHECK; v++) {
  const rbg = v % RG, cbg = Math.floor(v / RG) % CB;
  tiling.warps.forEach((w, wi) => {
    for (let l = 0; l < 32; l++) {
      const t = L.laneTile(tiling, w, l);
      const at = (v * 256 + wi * 32 + l) * 16;
      const msg = dump.slice(at, at + 16);
      lanes.push({ v, wi, l, rows: t.rows.map((r) => r + rbg * BM), cols: t.cols.map((c) => c + cbg * BN), msg,
        hash: L.jackpotHash(msg, aSeed) });
    }
  });
}
// The gather needs warp w's rows in row band (w >> 2) and its columns in slot (w & 3): the same as the tiling's
// warp order (i along rows, j along columns).
const fold = (ln) => {
  const secretA = ln.rows.map((r) => A.subarray(r * K, r * K + K)), secretB = ln.cols.map((c) => Bt.subarray(c * K, c * K + K));
  for (const sx of secretA.concat(secretB)) for (const x of sx) if (x < -64 || x > 64) throw new Error('Matrix value out of range');
  const nz = noise.computeNoiseForIndices({ k: K, rank: RANK, aSeed, bSeed, rowIndices: ln.rows, colIndices: ln.cols });
  return L.computeJackpot({ secretA, secretB, noiseA: nz.noiseA, noiseB: nz.noiseB, k: K, r: RANK });
};
let hits = 0, ok = 0, sampled = 0, sampledOk = 0;
const why = [];
const t2 = Date.now();
// A hit through the port of the pool's check. ln.hash is the card's transcript's hash when the tile was
// dumped; a hit the card listed in another tile has none, and the refolded transcript must meet the bound.
const poolCheck = (ln) => {
  const pr = L.listToPattern(ln.rows), pc = L.listToPattern(ln.cols);
  // the share's config is rebuilt from its own indices; it must give the job key the miner used
  const cfgProfile = { ...profile, rows: L.patToList(pr.shape), cols: L.patToList(pc.shape) };
  if (!reference.jobKey(header, cfgProfile).equals(key)) throw new Error('job key differs');
  L.sanityCheck({ k: K, r: RANK, h: ln.rows.length, w: ln.cols.length, m: M, n: N, tRows: pr.offset, tCols: pc.offset,
    rowsShape: pr.shape, colsShape: pc.shape });
  const j = fold(ln);
  const hsh = L.jackpotHash(j.msg, aSeed);
  if (ln.hash && !hsh.equals(ln.hash)) throw new Error('jackpot hash differs');
  if (pearlhash.leBytesToBigInt(hsh) > bound) throw new Error('hash does not meet the bound');
  // our own port of the fold, as checkHit uses it
  const nz = noise.computeNoiseForIndices({ k: K, rank: RANK, aSeed, bSeed, rowIndices: ln.rows, colIndices: ln.cols });
  const ours = reference.foldTranscript({ A, Bt, noiseA: nz.noiseA, noiseB: nz.noiseB, rows: ln.rows, cols: ln.cols, k: K, rank: RANK });
  for (let q = 0; q < 16; q++) if ((ours[q] >>> 0) !== j.msg[q]) throw new Error('foldTranscript differs');
};
const mine = new Set();
lanes.forEach((ln, i) => {
  const isHit = pearlhash.leBytesToBigInt(ln.hash) <= bound;
  if (isHit) {
    hits++;
    mine.add(ln.v * 256 + ln.wi * 32 + ln.l);
    try {
      poolCheck(ln);
      ok++;
    } catch (e) { if (why.length < 4) why.push(`tile ${ln.v} warp ${ln.wi} lane ${ln.l}: ${e.message}`); }
  } else if (i % 61 === 0) {
    sampled++;
    try {
      const j = fold(ln);
      let same = true;
      for (let q = 0; q < 16; q++) if (j.msg[q] !== ln.msg[q]) same = false;
      if (same) sampledOk++;
      else if (why.length < 4) why.push(`tile ${ln.v} warp ${ln.wi} lane ${ln.l}: transcript differs`);
    } catch (e) { if (why.length < 4) why.push(e.message); }
  }
});
out.msCheck = Date.now() - t2;
out.lanes = lanes.length;
out.hitsThroughPoolCheckPort = `${ok} of ${hits}`;
out.otherLanesSampled = `${sampledOk} of ${sampled}`;
if (ok !== hits || hits === 0 || sampledOk !== sampled) fail(why.join('; ') || 'no hit to check');
if (HASH) {
  // The card's own list: the same hits as here in the checked tiles, and its hits elsewhere through the check.
  const hb = fs.existsSync(hitFile) ? fs.readFileSync(hitFile) : Buffer.alloc(8);
  const list = new Uint32Array(hb.buffer, hb.byteOffset, hb.length >> 2);
  const n = (list.length - 2) >> 1;
  let inChecked = 0, same = 0, elsewhere = 0, elsewhereOk = 0;
  const seen = new Set();
  for (let i = 0; i < n; i++) {
    const v = list[2 + 2 * i], tid = list[3 + 2 * i];
    if (v < CHECK) {
      inChecked++;
      if (mine.has(v * 256 + tid) && !seen.has(v * 256 + tid)) same++;
      seen.add(v * 256 + tid);
    } else if (elsewhere < 400) {
      elsewhere++;
      const rbg = v % RG, cbg = Math.floor(v / RG) % CB, wi = tid >> 5, l = tid & 31;
      const t = L.laneTile(tiling, tiling.warps[wi], l);
      try {
        poolCheck({ rows: t.rows.map((r) => r + rbg * BM), cols: t.cols.map((c) => c + cbg * BN) });
        elsewhereOk++;
      } catch (e) { if (why.length < 4) why.push(`the card's hit in tile ${v} thread ${tid}: ${e.message}`); }
    }
  }
  out.cardHits = list[0];
  out.cardHitsInCheckedTiles = `${same} of ${inChecked} are the ${mine.size} found here`;
  out.cardHitsElsewhereThroughPoolCheckPort = `${elsewhereOk} of ${elsewhere}`;
  if (list[0] !== n || same !== inChecked || same !== mine.size || elsewhereOk !== elsewhere) {
    fail("the card's hit list: " + (why.join('; ') || 'differs from the list made here'));
  }
}
out.PASS = !process.exitCode;
try { fs.rmSync(dir, { recursive: true, force: true }); } catch (e) { /* the files are small */ }
console.log(JSON.stringify(out));
