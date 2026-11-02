// A stand-in pool that checks every share the way a real one does, and a run that mines two k on one card
// through one pool connection. For RDNA 4 mining k 4096 beside cards at k 2048 (round 4 item 5, step 2).
//
//   node verify-pool.js serve PORT SECS [BITS=210] [plain]
//     A stratum pool on 127.0.0.1:PORT for SECS seconds, for the app's CLI (--pool 127.0.0.1:PORT). Every
//     connection gets one job, and a new one every PEARL_JOB_EVERY seconds (default 30), at the target 2^BITS.
//     It agrees to compressed proofs, as both pools do, unless `plain` is given: then the CLI sends raw ones,
//     about 180 KB a line at k 4096. Each share is checked as zk-pow's verify_plain_proof does (checkShare
//     below) and answered accepted, or rejected with the reason. Ends with one JSON line: shares by k, accepted,
//     rejected and the reasons, and PASS (at least one share, none rejected).
//
//   node verify-pool.js mixed CORE.node SECS [BITS=210 | pool]
//     PearlMiner itself, as the app runs it, with two cores on one card (PEARL_GPU_INDEX, default 0): the first
//     opened with cardK and the second without, so an RDNA 4 card mines k 4096 and k 2048 side by side on one
//     pool connection. Against this stand-in pool in the same process at 2^BITS, or with `pool` against the live
//     pool in POOL (host:port) as WALLET / WORKER, which are never printed. Ends with one JSON line: each core's
//     k, and the shares sent, accepted and rejected at each k as the pool answered them. PASS: both k sent shares,
//     every share at both k accepted, and (stand-in) every share checked. The two cores share a device index, so
//     the miner's per-card hashrate shows one of them: don't read a TH/s total from this.
//
// What the stand-in checks, in verify_plain_proof's order: the proof parses, the sanity checks (rank, k
// against rank, the tile, m and n, the worker input), the tile's rows and columns form a pattern at a valid
// offset, config52 rebuilt from k, rank and the patterns, job_key = blake3(header || config52), both trees have
// the leaves m*k and n*k imply, both Merkle proofs open to their roots under that job_key, the strips are in
// the proof and in [-64, 64], the salted seed chain, the noise, the cumulative fold, the transcript hash, and
// the hash against the target scaled both ways (tile * k, and tile * (k / rank) * 128). It does not run the
// circuit's degree check. It is our JS: refverify.rs is the spec's own code, for what that leaves.
'use strict';
const net = require('net');
const path = require('path');
const S = path.join(__dirname, '..', '..', 'src') + path.sep;
const M = S + 'shared' + path.sep + 'miner' + path.sep;
const { decodePlainProof } = require(M + 'plainProof');
const {
  buildConfig52, patternFromList, leBytesToBigInt, SEED_SALT_A, SEED_SALT_B, bindMessage, PROFILE,
} = require(M + 'pearlhash');
const { hash, keyedHash } = require(M + 'blake3');
const { verifyProof } = require(M + 'merkle');
const { computeNoiseForIndices } = require(M + 'noise');
const { foldTranscript, transcriptBytes } = require(M + 'reference');

// The verifier's PeriodicPattern::offset_is_valid, over patternFromList's dimensions (smallest stride first).
function offsetIsValidFor(shape, offset) {
  let o = offset;
  for (const [stride, length] of shape.slice().reverse()) {
    o %= stride * length;
    if (o >= stride) return false;
  }
  return true;
}

// list_to_pattern: a strictly increasing list as (pattern, offset), the offset valid for the pattern.
function listToPattern(list) {
  if (!list.length) throw new Error('pattern parsing error: Empty indices');
  for (let i = 1; i < list.length; i++) {
    if (!(list[i - 1] < list[i])) throw new Error('pattern parsing error: Indices not strictly increasing');
  }
  const offset = list[0];
  const pattern = list.map((x) => x - offset);
  const shape = patternFromList(pattern);
  if (!offsetIsValidFor(shape, offset)) throw new Error('pattern parsing error: offset ' + offset + ' is not valid for pattern');
  return { pattern, offset };
}

// Rows of k bytes each, read out of a Merkle proof's leaves, as extract_strips does: row r is [r*k, r*k + k).
function strips(proof, rows, k) {
  const leaf = new Map(proof.leafIndices.map((li, i) => [li, proof.leafData[i]]));
  return rows.map((r) => {
    const out = new Int8Array(k);
    for (let b = 0; b < k; b++) {
      const off = r * k + b;
      const l = leaf.get(Math.floor(off / 1024));
      if (!l) throw new Error('Failed to extract strip');
      out[b] = (l[off % 1024] << 24) >> 24;
    }
    return out;
  });
}

// Why a share fails the check, or null when it passes. `header` is the job's 76 bytes, `target` the pool's
// share target as a BigInt, `pp` the decoded proof (decodePlainProof).
function reason(header, target, pp) {
  const { m, n, k, noiseRank: r } = pp;
  const rows = pp.a.rowIndices, cols = pp.bt.rowIndices;
  const h = rows.length, w = cols.length;
  if (!(r >= 32 && r <= 1024 && (r & (r - 1)) === 0)) return 'Rank must be 2^5..2^10 || r=' + r;
  if (k > 1 << 16) return 'k must be <= 2^16 || k=' + k;
  if (k % 64) return 'k must be divisible by 64 || k=' + k;
  if (k > 4 * r * r) return 'k must be no more than 4r^2 || k=' + k + ' r=' + r;
  if (k < 16 * r) return 'k must be >= 16r || k=' + k + ' r=' + r;
  if (h % 2 || w % 2) return 'Inner hash tile dimensions must be divisible by 2 || h=' + h + ' w=' + w;
  if (h * w < 32 || h * w > 256) return 'Inner hash size must be 32..256 || h=' + h + ' w=' + w;
  if (m > 1 << 24 || n > 1 << 24) return 'm and n must be <= 2^24 || m=' + m + ' n=' + n;
  if ((h + w) * k > 1 << 22) return 'Worker input supported up to 4 MiB';
  let A, B;
  try {
    A = listToPattern(rows);
    B = listToPattern(cols);
  } catch (e) {
    return e.message;
  }
  if (A.offset + A.pattern[h - 1] >= m) return 't_rows + pattern max must be < m';
  if (B.offset + B.pattern[w - 1] >= n) return 't_cols + pattern max must be < n';
  const config = { k, rank: r, mmaType: 0, rows: A.pattern, cols: B.pattern };
  const jobKey = hash(Buffer.concat([header, buildConfig52(config)]));
  const leaves = (dim) => Math.ceil((dim * k) / 1024);
  if (pp.a.proof.totalLeaves !== leaves(m)) return 'A Merkle tree declares ' + pp.a.proof.totalLeaves + ' leaves but m=' + m + ' k=' + k + ' imply ' + leaves(m);
  if (pp.bt.proof.totalLeaves !== leaves(n)) return 'B^T Merkle tree declares ' + pp.bt.proof.totalLeaves + ' leaves but n=' + n + ' k=' + k + ' imply ' + leaves(n);
  if (!verifyProof(jobKey, pp.a.proof)) return 'Hash A mismatch (the proof does not open to its root under this job_key)';
  if (!verifyProof(jobKey, pp.bt.proof)) return 'Hash B mismatch (the proof does not open to its root under this job_key)';
  let sa, sb;
  try {
    sa = strips(pp.a.proof, rows, k);
    sb = strips(pp.bt.proof, cols, k);
  } catch (e) {
    return e.message;
  }
  for (const s of sa.concat(sb)) for (const v of s) if (v < -64 || v > 64) return 'Matrix value ' + v + ' out of range [-64, 64]';
  const boundA = keyedHash(SEED_SALT_A, bindMessage(pp.a.proof.root, m));
  const boundB = keyedHash(SEED_SALT_B, bindMessage(pp.bt.proof.root, n));
  const bSeed = hash(Buffer.concat([jobKey, boundB]));
  const aSeed = hash(Buffer.concat([bSeed, boundA]));
  const { noiseA, noiseB } = computeNoiseForIndices({ k, rank: r, aSeed, bSeed, rowIndices: rows, colIndices: cols });
  const At = new Int8Array(h * k), Bt = new Int8Array(w * k);
  sa.forEach((s, i) => At.set(s, i * k));
  sb.forEach((s, i) => Bt.set(s, i * k));
  const jp = foldTranscript({
    A: At, Bt, noiseA, noiseB, rows: rows.map((_, i) => i), cols: cols.map((_, i) => i), k, rank: r,
  });
  const v = leBytesToBigInt(keyedHash(aSeed, transcriptBytes(jp)));
  const tile = BigInt(h * w);
  if (v > target * tile * BigInt(k)) return 'Jackpot condition not satisfied: hash does not meet difficulty target';
  if (v > target * tile * BigInt(Math.floor(k / r) * 128)) {
    return 'Jackpot condition not satisfied: hash does not meet the rank-penalized difficulty target';
  }
  return null;
}

// One share's `plain_proof` field, checked: { k, why }, why null when it passes. gzip: whether the
// connection agreed to compressed proofs.
function checkShare(header, target, field, { gzip = false } = {}) {
  let pp;
  try {
    pp = decodePlainProof(field, { gzip });
  } catch (e) {
    return { k: null, why: e.message };
  }
  return { k: pp.k, why: reason(header, target, pp) };
}

// The stand-in pool. Returns { server, stats, close() }; stats.byK[k] = { shares, accepted, rejected },
// stats.reasons[reason] = count. `target`, when given, is the share target itself instead of 2^bits.
function servePool({ port, bits = 210, target: exact, plain = false, jobEvery = 30, log = () => {} }) {
  const target = exact === undefined ? 2n ** BigInt(bits) : exact;
  const stats = { byK: {}, reasons: {}, connections: 0, jobs: 0 };
  const conns = new Set();
  let seq = 0;
  const job = () => {
    const header = Buffer.alloc(76);
    header.writeUInt32LE(++seq, 0);
    header.fill(0x5A, 4);
    stats.jobs++;
    return { id: 'stand-in-' + seq, header };
  };
  const server = net.createServer((sock) => {
    stats.connections++;
    conns.add(sock);
    const jobs = new Map();
    let gzip = false;
    let buf = '';
    const send = (msg) => sock.write(JSON.stringify(msg) + '\n');
    const notify = () => {
      const j = job();
      jobs.set(j.id, j);
      send({ id: null, method: 'mining.notify', params: {
        job_id: j.id, header: j.header.toString('hex'), target: target.toString(16).padStart(64, '0'), height: seq, cert_version: 3,
      } });
    };
    const timer = jobEvery > 0 ? setInterval(notify, jobEvery * 1000) : null;
    sock.on('close', () => { if (timer) clearInterval(timer); conns.delete(sock); });
    sock.on('error', () => {});
    sock.on('data', (chunk) => {
      buf += chunk;
      let i;
      while ((i = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, i);
        buf = buf.slice(i + 1);
        let m;
        try { m = JSON.parse(line); } catch (e) { continue; }
        if (m.method === 'mining.authorize') {
          gzip = !plain && m.params && m.params.type === 'v2';
          send({ id: m.id, result: true, error: null, ...(gzip ? { type: 'v2' } : {}) });
          notify();
        } else if (m.method === 'mining.submit') {
          const p = m.params || {};
          const j = jobs.get(p.job_id);
          const { k, why } = j ? checkShare(j.header, target, p.plain_proof, { gzip }) : { k: null, why: 'job not found' };
          const s = (stats.byK[k] = stats.byK[k] || { shares: 0, accepted: 0, rejected: 0 });
          s.shares++;
          if (why) {
            s.rejected++;
            stats.reasons[why] = (stats.reasons[why] || 0) + 1;
            send({ id: m.id, result: null, error: [23, why] });
          } else {
            s.accepted++;
            send({ id: m.id, result: true, error: null });
          }
          log('share k=' + k + ' ' + (why ? 'rejected: ' + why : 'accepted'));
        }
      }
    });
  });
  server.listen(port, '127.0.0.1');
  return {
    server,
    stats,
    close() {
      for (const c of conns) c.destroy();
      server.close();
    },
  };
}

function shareCounts(byK) {
  return Object.values(byK).reduce((a, s) => ({ shares: a.shares + s.shares, rejected: a.rejected + s.rejected }),
    { shares: 0, rejected: 0 });
}

function serveMain(argv) {
  const port = Number(argv[0]), secs = Number(argv[1]);
  const bits = Number(argv[2] || 210), plain = argv[3] === 'plain';
  const jobEvery = Number(process.env.PEARL_JOB_EVERY || 30);
  const pool = servePool({ port, bits, plain, jobEvery, log: (l) => console.error('[verify-pool] ' + l) });
  pool.server.on('listening', () => console.error('[verify-pool] listening on 127.0.0.1:' + port
    + (plain ? ', raw proofs only' : ', compressed proofs offered') + ', target 2^' + bits));
  const end = () => {
    pool.close();
    const c = shareCounts(pool.stats.byK);
    const res = { mode: 'serve', secs, bits, plain, ...pool.stats, PASS: c.shares > 0 && c.rejected === 0 };
    console.log(JSON.stringify(res));
    process.exit(res.PASS ? 0 : 1);
  };
  setTimeout(end, secs * 1000);
  process.on('SIGTERM', end);
}

function mixedMain(argv) {
  const { PearlMiner } = require(S + 'main' + path.sep + 'pearlMiner');
  const file = path.resolve(argv[0]);
  const secs = Number(argv[1] || 300);
  const live = argv[2] === 'pool';
  const gpu = Number(process.env.PEARL_GPU_INDEX || 0);
  // PEARL_PROFILE_MN=MxN mines a smaller m and n than the mainnet profile's, for stand-in cores whose shares are
  // built in JS (the tests). A real card mines the mainnet profile.
  const mn = /^(\d+)x(\d+)$/.exec(process.env.PEARL_PROFILE_MN || '');
  const profile = mn ? { ...PROFILE, m: Number(mn[1]), n: Number(mn[2]) } : PROFILE;
  const addon = require(file);
  const cores = [];
  // The first core (salt lane 0) is opened as the app opens it, with cardK; the second without.
  const createCore = (p, opts) => {
    const c = addon.createCore(p, opts.saltBase % opts.saltStride === 1 ? { ...opts, cardK: false } : opts);
    cores.push({ lane: opts.saltBase % opts.saltStride, k: c.k, kNote: c.kNote, device: c.device });
    return c;
  };
  let pool = null, endpoint;
  if (live) {
    endpoint = process.env.POOL || 'prl-us.kryptex.network:7048';
  } else {
    pool = servePool({ port: 0, bits: Number(argv[2] || 210), jobEvery: Number(process.env.PEARL_JOB_EVERY || 30) });
  }
  // What went out, by submit id, and what came back: the k of each share as its proof says.
  const sent = new Map();
  const byK = {};
  const tally = (k) => (byK[k] = byK[k] || { sent: 0, accepted: 0, rejected: 0, reasons: {} });
  const connect = (host, port) => {
    const sock = net.connect({ host, port });
    const write = sock.write.bind(sock);
    sock.write = (data, ...rest) => {
      try {
        const m = JSON.parse(String(data));
        if (m.method === 'mining.submit') {
          const raw = Buffer.from(m.params.plain_proof, 'base64');
          const k = decodePlainProof(m.params.plain_proof, { gzip: raw[0] === 0x1f && raw[1] === 0x8b }).k;
          sent.set(m.id, k);
          tally(k).sent++;
        }
      } catch (e) { /* not a submit */ }
      return write(data, ...rest);
    };
    let buf = '';
    sock.on('data', (chunk) => {
      buf += chunk;
      let i;
      while ((i = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, i);
        buf = buf.slice(i + 1);
        let m;
        try { m = JSON.parse(line); } catch (e) { continue; }
        if (!sent.has(m.id) || m.method) continue;
        const t = tally(sent.get(m.id));
        sent.delete(m.id);
        if (m.result === true && !m.error) t.accepted++;
        else {
          t.rejected++;
          const why = JSON.stringify(m.error || m.result).slice(0, 120);
          t.reasons[why] = (t.reasons[why] || 0) + 1;
        }
      }
    });
    return sock;
  };
  const start = () => {
    const miner = new PearlMiner({ connect, createCore });
    miner.on('log', (l) => console.error('[miner] ' + l.line));
    miner.on('error', (e) => console.error('[miner] error: ' + ((e && e.message) || e)));
    const card = { index: gpu, name: 'GPU ' + gpu };
    miner.start({
      endpoint, address: process.env.WALLET || 'prl1pstandin', worker: process.env.WORKER || 'verify-pool',
      gpus: [card, card], profile,
    });
    setTimeout(() => {
      miner.stop();
      if (pool) pool.close();
      const ks = Object.keys(byK);
      const both = ks.length >= 2 && ks.every((k) => byK[k].sent > 0);
      const clean = ks.every((k) => byK[k].rejected === 0 && byK[k].accepted === byK[k].sent);
      const checked = !pool || shareCounts(pool.stats.byK).rejected === 0;
      const res = {
        mode: 'mixed', pool: live ? 'live' : 'stand-in', secs, cores, byK, unanswered: sent.size,
        ...(pool ? { standIn: pool.stats } : {}), PASS: both && clean && checked,
      };
      console.log(JSON.stringify(res));
      process.exit(res.PASS ? 0 : 1);
    }, secs * 1000);
  };
  if (pool) {
    pool.server.on('listening', () => {
      endpoint = '127.0.0.1:' + pool.server.address().port;
      start();
    });
  } else {
    start();
  }
}

if (require.main === module) {
  const [mode, ...rest] = process.argv.slice(2);
  if (mode === 'serve') serveMain(rest);
  else if (mode === 'mixed') mixedMain(rest);
  else {
    console.error('usage: node verify-pool.js serve PORT SECS [BITS] [plain] | mixed CORE.node SECS [BITS | pool]');
    process.exit(2);
  }
}

module.exports = { checkShare, servePool, listToPattern, offsetIsValidFor };
