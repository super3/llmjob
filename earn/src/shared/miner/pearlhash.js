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
// A periodic index pattern, the shape the protocol uses to describe which rows
// of A and columns of B form the jackpot tile. Three (stride, length) dimensions,
// serialised to SIX bytes as (factor-1, length-1) per dimension, where factor is
// the stride divided by the running product of the previous dimensions.
//
// This replaced a guess. The tile was previously derived as "row i = i*8, columns
// in pairs at stride 8", which produced a 2x64 tile — the real default is 4x8,
// and the index sets are not a simple stride at all.
const ROWS_PATTERN = [0, 8, 64, 72];
const COLS_PATTERN = [0, 1, 8, 9, 32, 33, 40, 41];

// Expand a pattern's (stride, length) dimensions into its index list, exactly as
// PeriodicPattern::to_list does: start from [0] and, for each dimension, replace
// the set with {r + i*stride} for i in 0..length.
function patternToList(shape) {
  let res = [0];
  for (const [stride, length] of shape) {
    const next = [];
    for (let i = 0; i < length; i++) for (const r of res) next.push(r + i * stride);
    res = next;
  }
  return res;
}

// Recover the (stride, length) dimensions from an index list — the inverse of
// patternToList, matching PeriodicPattern::from_list. Dimensions come out
// smallest-stride first, which is the order to_bytes expects.
function patternFromList(list) {
  let p = list.slice();
  const shape = [];
  while (p.length > 1) {
    let found = false;
    for (let period = 1; period < p.length; period++) {
      if (p.length % period) continue;
      const stride = p[period];
      let periodic = true;
      for (let i = 0; i + period < p.length; i++) {
        if (p[i] + stride !== p[i + period]) { periodic = false; break; }
      }
      if (!periodic) continue;
      shape.unshift([stride, p.length / period]);
      p = p.slice(0, period);
      found = true;
      break;
    }
    if (!found) throw new Error('index pattern is not periodic: ' + list.join(','));
  }
  return shape;
}

// Six bytes: (factor-1, length-1) per dimension, factor = stride / running
// product. The unused third dimension is padded as (min_stride, 1) so its factor
// is 1 — the protocol rejects any other encoding as non-canonical.
function patternToBytes(shape) {
  const out = Buffer.alloc(6);
  let minStride = 1;
  const dims = shape.slice();
  while (dims.length < 3) dims.push([minStridePad(shape), 1]);
  for (let i = 0; i < 3; i++) {
    const [stride, length] = dims[i];
    const factor = Math.floor(stride / minStride);
    out[2 * i] = factor - 1;
    out[2 * i + 1] = length - 1;
    minStride = stride * length;
  }
  return out;
}

// The stride the padding dimension must carry so its factor comes out as 1.
function minStridePad(shape) {
  let s = 1;
  for (const [stride, length] of shape) s = stride * length;
  return s;
}

// The mandated mainnet profile, taken from the reference implementation's own
// defaults rather than inferred:
//
//   rank = 128        the rank-penalty floor; below it blocks are penalised,
//                     above it costs more work for no extra credit
//   k    = 16 * rank  the smallest common dimension the sanity checks allow
//   tile = 4 x 8      rows_pattern x cols_pattern
//
// Note what is NOT here: m and n. The matmul's outer dimensions are not part of
// the mining configuration at all — the miner chooses them, because the work is
// meant to be a real workload. They therefore never enter job_key, and the
// search is over the tile OFFSET (t_rows, t_cols) within whatever output the
// miner computed.
//
// k / rank = 16 chunks, which is exactly the number of transcript lanes — so
// every chunk lands in its own lane and the rotation never wraps. The earlier
// k = 4096 gave 32 chunks and wrapped each lane twice.
const PROFILE = {
  // Hashed into config52 — protocol-mandated, must match the network exactly.
  k: 2048,
  rank: 128,
  mmaType: 0, // Int7xInt7ToInt32
  rows: ROWS_PATTERN,
  cols: COLS_PATTERN,
  // NOT hashed. The outer dimensions are the miner's own choice of workload, so
  // they never enter job_key — they only size the operands we allocate and bound
  // the tile offset. Bigger m*n means more candidate offsets per commitment,
  // which is the whole amortisation story.
  m: 4096,
  n: 4096,
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
// The 52-byte mining configuration, hashed with the 76-byte header to derive
// job_key. Layout is the reference's MiningConfiguration::to_bytes, byte for
// byte:
//
//   common_dim u32 (4) | rank u16 (2) | mma_type u16 (2)
//   rows_pattern   (6) | cols_pattern (6) | MoE trailer (32)
//
// Getting any of this wrong changes job_key and therefore every hash downstream,
// with no error anywhere — which is exactly what the first version did: it
// packed m, n, k, rank, hashTile and two pattern COUNTS, none of which the
// protocol carries.
function buildConfig52(profile) {
  const p = profile || PROFILE;
  const b = Buffer.alloc(CONFIG_BYTES);
  b.writeUInt32LE(p.k >>> 0, 0);
  b.writeUInt16LE(p.rank & 0xffff, 4);
  b.writeUInt16LE((p.mmaType || 0) & 0xffff, 6);
  // The tile patterns are protocol constants, not per-profile knobs — the device
  // hardcodes them too, so a profile that omits them still gets the right bytes.
  patternToBytes(patternFromList(p.rows || ROWS_PATTERN)).copy(b, 8);
  patternToBytes(patternFromList(p.cols || COLS_PATTERN)).copy(b, 14);
  // Bytes 20..51 are the MoE trailer, zero for a standard (non-GROUPED_GEMM) job.
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
  PROFILE, ROWS_PATTERN, COLS_PATTERN,
  patternToList, patternFromList, patternToBytes,
  CONFIG_BYTES,
  JACKPOT_BUCKETS,
  ROTL_BITS,
  buildConfig52,
  leBytesToBigInt,
  meetsTarget,
  rotl13,
  rankMatches,
};
