'use strict';

// PearlHash parameters and the pure-arithmetic parts of the proof-of-work — the
// bits that don't need a GPU. The heavy compute (keyed BLAKE3, low-rank noise
// generation, the int8 GEMM, and the per-tile jackpot fold) lives in the CUDA
// core under earn/native and is reached through the PearlCore interface; this
// module is what the JS host uses to set that core up and to check what it
// returns.
//
// Algorithm reference (PROTOCOL.md, cross-checked against the ISC-licensed
// pearl-research-labs/pearl zk-pow crate — NOT against any dev-fee-licensed
// miner):
//
//   job_key = blake3(header76 ‖ config52)
//   hash_a  = blake3(pad1024(A_rowmajor), key=job_key)
//   hash_b  = blake3(pad1024(Bᵀ),        key=job_key)
//   b_seed  = blake3(job_key ‖ hash_b)
//   a_seed  = blake3(b_seed  ‖ hash_a)
//   noise:  E_A = E_AL·E_AR, E_B = E_BL·E_BR   (uniform + permutation draws from
//           keyed BLAKE3; rank r, default 128)
//   tile:   rows_pattern × cols_pattern; C accumulated in rank chunks; per chunk
//           jackpot[tid] = rotl13(jackpot[tid]) ^ xor(tile),  tid = chunk % 16
//   share iff  int_le(blake3(transcript64, key=a_seed)) <= target
//
// The default mainnet profile is the rank-128 one the softfork mandates (the
// same rank the pool asks for in its job params). Rank ≠ 128 work is uncredited
// post-fork, which is the whole reason the old alpha-miner fallbacks were pulled.

// The mandated mainnet profile. m/n/k and rank are the matmul geometry; rows/cols
// are the sub-tile the transcript folds over (2×64 for the default). These are
// the numbers baked into config52 and handed to the core; a job whose params
// disagree is mined at the wrong geometry and earns nothing, so the host asserts
// the pool's rank against PROFILE.rank before it starts a search.
const PROFILE = {
  m: 131072,
  n: 131072,
  k: 4096,
  rank: 128,
  hashTile: 16,
  rows: [0, 8],
  cols: (() => {
    // The default column pattern: pairs (i, i+1) at every 8th position across 64
    // columns — [0,1,8,9,16,17,…,248,249]. Written out rather than hard-coded so
    // the shape is legible and a profile change is a one-line edit.
    const out = [];
    for (let base = 0; base < 256; base += 8) { out.push(base, base + 1); }
    return out;
  })(),
};

const CONFIG_BYTES = 52;
const JACKPOT_BUCKETS = 16;
const ROTL_BITS = 13;

// Lay out the 52-byte config block that is hashed together with the 76-byte
// header to derive job_key. Field order and widths follow the reference: little-
// endian u32/u16 for the geometry, then the rank. Any change here changes every
// downstream hash, so it is the single source of truth the core is generated
// from (native/src/pearl_config.h mirrors these offsets — kept in sync by the
// round-trip test, not by hand).
function buildConfig52(profile) {
  const p = profile || PROFILE;
  const b = Buffer.alloc(CONFIG_BYTES);
  b.writeUInt32LE(p.m >>> 0, 0);
  b.writeUInt32LE(p.n >>> 0, 4);
  b.writeUInt32LE(p.k >>> 0, 8);
  b.writeUInt16LE(p.rank & 0xffff, 12);
  b.writeUInt16LE(p.hashTile & 0xffff, 14);
  // rows/cols counts (the patterns themselves are derived by the core from these
  // counts plus the fixed stride, exactly as the reference does).
  b.writeUInt16LE(p.rows.length & 0xffff, 16);
  b.writeUInt16LE(p.cols.length & 0xffff, 18);
  // bytes 20..51 are reserved zero in the current profile; the core reads the
  // same width so the job_key matches.
  return b;
}

// Read a 32-byte hash as a little-endian unsigned integer, which is how the
// jackpot hash is compared to the target (the target itself is big-endian — see
// stratum.decodeTarget). Isolated and tested here so the endianness, which is
// the single easiest thing to get backwards and the hardest to notice, is
// pinned: get it wrong and every "share" the core reports is spurious.
function leBytesToBigInt(bytes) {
  const b = Buffer.isBuffer(bytes) ? bytes : Buffer.from(bytes || []);
  let v = 0n;
  for (let i = b.length - 1; i >= 0; i--) v = (v << 8n) | BigInt(b[i]);
  return v;
}

// Does a jackpot hash (32 LE bytes) satisfy the share target (a BigInt from
// stratum.decodeTarget)? The core does this check on-GPU for throughput; the
// host re-checks every hit the core returns before submitting, so a core bug —
// or a stale job whose target moved under vardiff — can never push a bad share
// to the pool and earn a ban.
function meetsTarget(jackpotHashLE, target) {
  if (target == null) return false;
  return leBytesToBigInt(jackpotHashLE) <= target;
}

// rotl13 on a byte lane, the transcript fold's mixing step. Exposed for the
// core's known-answer test (native and JS must agree bit-for-bit or the whole
// pipeline is silently wrong), not used in the hot path here.
function rotl13(x) {
  const v = x >>> 0;
  return ((v << ROTL_BITS) | (v >>> (32 - ROTL_BITS))) >>> 0;
}

// The pool tells us the rank it wants in the job (…_<suffix> and its own params).
// Mining any other rank is uncredited post-softfork, so the host calls this to
// refuse a job whose rank does not match the profile the core was built for,
// rather than burn power on work that will never be paid.
function rankMatches(jobRank, profile) {
  const want = (profile || PROFILE).rank;
  return jobRank == null || Number(jobRank) === want;
}

module.exports = {
  PROFILE,
  CONFIG_BYTES,
  JACKPOT_BUCKETS,
  ROTL_BITS,
  buildConfig52,
  leBytesToBigInt,
  meetsTarget,
  rotl13,
  rankMatches,
};
