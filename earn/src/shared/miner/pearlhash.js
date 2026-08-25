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
// The tile is CONTIGUOUS: four consecutive rows by sixteen consecutive columns.
//
// This is what the reference miner actually mines. Its own test fixtures open a
// block with
//
//   A_row_indices    = [192, 193, 194, 195]
//   B_column_indices = [96, 97, 98, ...]
//
// not the strided {0,8,64,72} x {0,1,8,9,32,33,40,41} that MiningConfiguration
// carries as a DEFAULT. Both are legal -- the pattern is self-describing in
// config52, and the miner chooses it -- but contiguous is better in every way
// that matters here: the sixteen B columns are one contiguous run of the
// operand rather than sixteen scattered rows, the Merkle proof covers
// consecutive chunks and so needs fewer siblings, and a 4x16 tile is the shape
// an int8 tensor-core mma maps onto directly.
//
// The offset rule is unchanged in form: rows {0,1,2,3} are the subsets of bits
// {0,1} and columns {0..15} the subsets of bits {0,1,2,3}, so a valid offset is
// still one with the pattern's own bits clear -- a multiple of 4 down and of 16
// across. Re-checked against a transcription of offset_is_valid.
const ROWS_PATTERN = [0, 1, 2, 3];
const COLS_PATTERN = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

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
  //
  // Rank sits on the PENALTY_BASE_RANK floor of 128, and k is the smallest the
  // sanity checks allow for it (k >= 16r). That is not a tuning preference --
  // it is worth an exact factor of two in accepted shares.
  //
  // The share bound is scaled by penalized_adjustment_factor,
  //   tile_size * (k/rank) * PENALTY_BASE_RANK,
  // while one attempt costs tile_size * k multiply-accumulates. So
  //   shares/second  ~  (MAC rate) * PENALTY_BASE_RANK / (rank * difficulty),
  // which falls as 1/rank for the same GPU work. At rank 256 every attempt cost
  // twice the arithmetic and earned the same bound: the unpenalized factor was
  // tile*k = 2^18, but the penalty rule only ever granted tile*(k/rank)*128 =
  // 2^17. At rank 128 with k = 16r the two coincide and nothing is forfeited.
  // The reference miner sits on the floor for exactly this reason.
  //
  // k/rank is 16 either way, which is what fills the 16-lane transcript once.
  k: 2048,
  rank: 128,
  mmaType: 0, // Int7xInt7ToInt32
  rows: ROWS_PATTERN,
  cols: COLS_PATTERN,
  // NOT hashed. The outer dimensions are the miner's own choice of workload, so
  // they never enter job_key — they only size the operands we allocate and bound
  // the tile offset. Bigger m*n means more candidate offsets per commitment,
  // which is the whole amortisation story.
  // Measured on an RTX 4090 with a working instrument. The old numbers here
  // came from a hashrate that was not a rate and a miner that stalled a few
  // seconds into every measurement, so they described nothing.
  //
  // Measured on a 4090 with the tensor-core partials kernel, at col_batch 512:
  // 16384 -> 62.2, 32768 -> 65.7 TH/s, and 65536 is within noise of that. The
  // curve plateaus rather than peaking, so this takes the knee.
  //
  // MUST be a power of two, and not merely as a tuning preference. The operand
  // commitment is a BLAKE3 Merkle tree over 1024-byte chunks and the device
  // folds it as a balanced pairwise reduction, which is only the right tree at
  // a power-of-two chunk count -- BLAKE3's real layout is left-heavy. A value
  // like 12288 (3 * 4096) gives a wrong root, wrong seeds, and shares no pool
  // will accept, and the small parity profile cannot catch it.
  m: 32768,
  n: 32768,

  // Column offsets per launch. Not protocol: it trades VRAM for amortised
  // launch overhead. 32 -> 55.4, 64 -> 58.3, 256 -> 60.8, 512 -> 63.1,
  // 1024 -> 64.4 TH/s at m=6144, so this is well past the knee.
  colBatch: 512,
  // Which seed derivation binds the operand roots. 'salted' is cert-v3:
  //   hash_a' = blake3(hash_a ‖ pad32(m), key=SEED_SALT_A)
  //   hash_b' = blake3(hash_b ‖ pad32(n), key=SEED_SALT_B)
  // 'legacy' uses the raw roots.
  //
  // Salted is the default because it is the only thing that commits m and n.
  // They are the miner's own choice and deliberately absent from config52, so
  // under 'legacy' nothing in the chain pins them at all — which is precisely
  // the hole cert-v3 was introduced to close.
  //
  // NOT yet confirmed against the live network: both derivations produce a
  // perfectly self-consistent miner and they differ only in whether a pool
  // accepts the share. If shares are rejected with everything else verified,
  // this flag is the first thing to flip.
  seedDerivation: 'salted',

  // The same choice as a number, because that is what the addon reads. Keeping
  // the string as the source of truth and mapping it here is deliberate: the
  // addon used to read the STRING through Uint32Value(), which yields 0 for any
  // non-numeric text. That happened to be the value for 'salted', so setting
  // 'legacy' would have been silently ignored and mined the wrong derivation.
  seedDerivationCode: 0,
};

// 0 = cert-v3 salted, 1 = legacy. The addon takes the number.
function seedDerivationCode(profile) {
  const p = profile || PROFILE;
  return p.seedDerivation === 'legacy' ? 1 : 0;
}

// The difficulty adjustment factor: tile size x dot product length.
//
// The protocol scales the jackpot bound in proportion to the work one attempt
// costs, so a hashrate is multiply-accumulates per second, not attempts per
// second. That is why competing miners quote hundreds of TH/s on a card that
// could not perform 1e14 BLAKE3 hashes a second: 296 TH/s of MACs is about 45%
// of an RTX 4090's int8 tensor-core peak, which is exactly what a well-tuned
// GEMM achieves.
function difficultyAdjustmentFactor(profile) {
  const p = profile || PROFILE;
  return (p.rows || ROWS_PATTERN).length * (p.cols || COLS_PATTERN).length * p.k;
}

// Domain-separation salts for the cert-v3 seed derivation. Hardcoded in the
// reference so consensus does not depend on runtime string hashing; both are
// re-derived from their strings in the tests, which doubles as an outside check
// on our BLAKE3.
//   SEED_SALT_A = blake3("pearl/cert-v3/noise-seed/A")
//   SEED_SALT_B = blake3("pearl/cert-v3/noise-seed/B")
const SEED_SALT_A = Buffer.from(
  '8249406ca0ed15169616f692fcf076f892dbdb2a7023b852f0d47719c390017b', 'hex');
const SEED_SALT_B = Buffer.from(
  '11300632ec6301ca2be2af718b3f4d4f1ae9c63988e8cc044844301d71b89aa9', 'hex');

// The 64-byte message the salt is applied to: root ‖ dim(u32 LE) ‖ 28 zeros.
function bindMessage(root, dim) {
  const msg = Buffer.alloc(64);
  Buffer.from(root).copy(msg, 0);
  msg.writeUInt32LE(dim >>> 0, 32);
  return msg;
}

// The rank the penalty is measured against. Mining below it is refused
// outright; mining above it costs more work for no extra credit.
const PENALTY_BASE_RANK = 128;

// The factor a MINER must scale a share target by.
//
//   penalized_adjustment_factor = tile_size * (k / rank) * PENALTY_BASE_RANK
//
// This is not the same quantity as difficultyAdjustmentFactor, even though both
// come to 65536 at the mandated rank-128 profile and so are easy to conflate.
// The consensus path scales by tile*k; a miner scaling a SHARE target scales by
// this, which divides out the rank and re-multiplies by the base. They diverge
// the moment rank != 128, and the reference is explicit that this is the one a
// miner wants (see penalized_target_bound in sanity_checks.rs).
function penalizedAdjustmentFactor(profile) {
  const p = profile || PROFILE;
  const tile = (p.rows || ROWS_PATTERN).length * (p.cols || COLS_PATTERN).length;
  return tile * Math.floor(p.k / p.rank) * PENALTY_BASE_RANK;
}

// The bound a jackpot hash is actually compared against:
//
//   int_le(jackpot_hash) <= target * penalized_adjustment_factor
//
// The bound is made EASIER in proportion to the work one attempt costs, which
// is the whole reason a hashrate here counts multiply-accumulates rather than
// attempts. Comparing against the raw target instead — which this miner did —
// makes every share 65536x rarer than the pool intends, so a correct miner at a
// realistic pool difficulty simply never finds one. It looks exactly like being
// slow, and at a target of 2^203 it works out to one share per 3.5 years.
//
// Returns null rather than saturating when the product will not fit: the
// reference deliberately refuses here, because handing back U256::MAX would give
// a bound that EVERY hash satisfies and flood the pool with junk.
function shareBound(target, profile) {
  if (target == null) return null;
  const factor = BigInt(penalizedAdjustmentFactor(profile));
  if (factor <= 0n) return null;
  const t = BigInt(target);
  const max = (1n << 256n) - 1n;
  if (t > max / factor) return null;
  return t * factor;
}

// A tile offset is VALID only if it has the pattern's own bits clear.
//
// The verifier rebuilds the pattern from a submitted proof's row indices and
// checks the offset with PeriodicPattern::offset_is_valid, which reduces the
// offset modulo each (stride, length) dimension and requires it to stay below
// the stride. For these patterns that reduces exactly to (offset & mask) == 0,
// where mask is the OR of the pattern's own values.
//
// Searching invalid offsets is not merely wasteful, it is unusable: the share
// comes back as "offset N is not valid for pattern". Only 1 in 32 regions
// qualifies, and the valid tiles PARTITION the grid rather than overlapping.
const ROWS_MASK = ROWS_PATTERN.reduce((a, b) => a | b, 0);
const COLS_MASK = COLS_PATTERN.reduce((a, b) => a | b, 0);

function offsetIsValid(offset, mask) {
  return (offset & mask) === 0;
}

// The i-th offset with (offset & mask) == 0: deposit i's bits into the
// positions the mask leaves free.
function expandOffset(i, mask) {
  let out = 0;
  let bit = 1;
  let rest = i;
  while (rest) {
    if (!(mask & bit)) {
      if (rest & 1) out |= bit;
      rest >>>= 1;
    }
    bit <<= 1;
  }
  return out >>> 0;
}

// A region index -> the tile it names. Row and column offsets are enumerated
// over VALID offsets only, so this is dense in the submittable space.
function regionToTile(region, profile) {
  const p = profile || PROFILE;
  const rowsValid = p.m / (p.rows || ROWS_PATTERN).length;
  const colsValid = p.n / (p.cols || COLS_PATTERN).length;
  const rowOff = expandOffset(region % rowsValid, ROWS_MASK);
  const colOff = expandOffset(Math.floor(region / rowsValid) % colsValid, COLS_MASK);
  return {
    rowOff,
    colOff,
    // The pattern's bits are clear in a valid offset, so this is an OR.
    rows: (p.rows || ROWS_PATTERN).map((r) => rowOff | r),
    cols: (p.cols || COLS_PATTERN).map((c) => colOff | c),
  };
}

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
  SEED_SALT_A, SEED_SALT_B, bindMessage,
  PENALTY_BASE_RANK, penalizedAdjustmentFactor, shareBound, seedDerivationCode,
  ROWS_MASK, COLS_MASK, offsetIsValid, expandOffset, regionToTile,
  patternToList, patternFromList, patternToBytes, difficultyAdjustmentFactor,
  CONFIG_BYTES,
  JACKPOT_BUCKETS,
  ROTL_BITS,
  buildConfig52,
  leBytesToBigInt,
  meetsTarget,
  rotl13,
  rankMatches,
};
