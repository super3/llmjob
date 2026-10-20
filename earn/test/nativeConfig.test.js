'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const {
  PROFILE, CONFIG_BYTES, JACKPOT_BUCKETS, ROTL_BITS, buildConfig52,
  patternFromList, patternToBytes,
  SEED_SALT_A, SEED_SALT_B, meetsTarget,
} = require('../src/shared/miner/pearlhash');

// The CUDA core and the JS reference must agree byte for byte. config52 is
// hashed with the header to derive job_key, so ONE mismatched offset silently
// changes every downstream hash and the miner produces work no pool will accept
// — with no error anywhere to point at it. That is not hypothetical: the first
// version of this file packed m, n, k, rank, hash_tile and two pattern counts,
// three of which the protocol does not carry at all.
//
// The native side cannot be compiled on this box, so agreement is asserted by
// reading the C header's own constants. It is a real guard: it fails the moment
// either side's profile, widths, tile patterns or fold parameters move.

const HEADER = fs.readFileSync(path.join(__dirname, '..', 'native', 'src', 'pearl_config.h'), 'utf8');

// The offset masks are DERIVED on both sides now rather than written down.
// They were hand-maintained constants, and when the tile grew from 4 rows to 16
// the native rows mask stayed at 3 while the JS one followed the pattern. Every
// hash still agreed; only the row indices in the proof were wrong, which no
// hash check can see. The native side computes its mask with a constexpr fold
// over its own pattern, so the two cannot drift again -- this asserts they
// start from the same pattern.
//
// Read a #define's integer value. Deliberately no regex: this file is written
// through shell heredocs that mangle backslash escapes, and a silently broken
// pattern here would make every assertion below vacuously pass on null.
function defineOf(name) {
  const at = HEADER.indexOf('#define ' + name + ' ');
  if (at < 0) return null;
  const eol = HEADER.indexOf(String.fromCharCode(10), at);
  const tail = HEADER.slice(at + 8 + name.length + 1, eol < 0 ? undefined : eol).trim();
  const n = parseInt(tail, 10);
  return Number.isFinite(n) ? n : null;
}

// Read the numbers out of a braced C initialiser, located by name. Plain string
// scanning rather than a regex over the whole file — the initialisers contain
// braces and commas and the intent is easier to read this way.
function initialiserNumbers(name) {
  const at = HEADER.indexOf(name);
  if (at < 0) return null;
  const open = HEADER.indexOf('{', at);
  const close = HEADER.indexOf('}', open);
  if (open < 0 || close < 0) return null;
  const nums = HEADER.slice(open + 1, close).split(/[^0-9]+/).filter(Boolean);
  return nums ? nums.map(Number) : null;
}


// The salts are written as 0xNN, which the decimal reader above would shred
// into pairs of digits. Parsed separately rather than by loosening that one.
function initialiserBytes(name) {
  const at = HEADER.indexOf(name);
  if (at < 0) return null;
  const open = HEADER.indexOf('{', at);
  const close = HEADER.indexOf('}', open);
  if (open < 0 || close < 0) return null;
  return (HEADER.slice(open + 1, close).match(/0x[0-9a-fA-F]{2}/g) || [])
    .map((h) => parseInt(h, 16));
}

describe('native/JS config agreement', () => {
  test('the fixed widths match', () => {
    expect(defineOf('PEARL_CONFIG_BYTES')).toBe(CONFIG_BYTES);
    expect(defineOf('PEARL_HEADER_BYTES')).toBe(76);
    expect(defineOf('PEARL_HASH_BYTES')).toBe(32);
  });

  test('the transcript fold parameters match', () => {
    expect(defineOf('PEARL_JACKPOT_BUCKETS')).toBe(JACKPOT_BUCKETS);
    expect(defineOf('PEARL_ROTL_BITS')).toBe(ROTL_BITS);
  });

  test('the tile dimensions match', () => {
    expect(defineOf('PEARL_ROWS_COUNT')).toBe(PROFILE.rows.length);
    expect(defineOf('PEARL_COLS_COUNT')).toBe(PROFILE.cols.length);
  });

  // { k, rank, mma_type, m, n } — m and n come last because they are the
  // miner's own dimensions rather than protocol, and never enter job_key.
  test('the mainnet profile matches the C initialiser', () => {
    const nums = initialiserNumbers('PEARL_MAINNET_PROFILE');
    expect(nums).toBeTruthy();
    expect(nums.slice(0, 3)).toEqual([PROFILE.k, PROFILE.rank, PROFILE.mmaType]);
  });

  // The tile index sets are duplicated as C arrays. A drift means the device
  // folds a different tile than the oracle and every hash diverges silently.
  test('the tile patterns match the C arrays', () => {
    expect(initialiserNumbers('PEARL_ROWS_PATTERN[')).toEqual(PROFILE.rows);
    expect(initialiserNumbers('PEARL_COLS_PATTERN[')).toEqual(PROFILE.cols);
  });

  // The C side hardcodes the six-byte periodic encodings rather than deriving
  // them, so they must equal what the JS derivation produces.
  test('the precomputed pattern bytes match the JS derivation', () => {
    expect(Buffer.from(initialiserNumbers('PEARL_ROWS_PATTERN_BYTES')))
      .toEqual(patternToBytes(patternFromList(PROFILE.rows)));
    expect(Buffer.from(initialiserNumbers('PEARL_COLS_PATTERN_BYTES')))
      .toEqual(patternToBytes(patternFromList(PROFILE.cols)));
  });

  test('the config block carries k, rank and mma_type at the reference offsets', () => {
    const block = buildConfig52();
    expect(block).toHaveLength(CONFIG_BYTES);
    expect(block.readUInt32LE(0)).toBe(PROFILE.k);
    expect(block.readUInt16LE(4)).toBe(PROFILE.rank);
    expect(block.readUInt16LE(6)).toBe(PROFILE.mmaType);
    // Bytes 8..19 are the two patterns; 20..51 the MoE trailer, zero here.
    expect(block.slice(20).every((b) => b === 0)).toBe(true);
  });

  // The cert-v3 salts are duplicated as C byte arrays. They are consensus
  // constants, so a drift means the device derives different seeds than the
  // oracle and every share is rejected with nothing to point at.
  test('the cert-v3 salts match the JS constants', () => {
    expect(Buffer.from(initialiserBytes('PEARL_SEED_SALT_A'))).toEqual(SEED_SALT_A);
    expect(Buffer.from(initialiserBytes('PEARL_SEED_SALT_B'))).toEqual(SEED_SALT_B);
  });

  // Both sides must default to the same derivation, or they disagree on every
  // seed while each stays internally consistent.
  test('both sides default to the cert-v3 derivation', () => {
    expect(defineOf('PEARL_SEED_SALTED')).toBe(0);
    expect(defineOf('PEARL_SEED_LEGACY')).toBe(1);
    expect(PROFILE.seedDerivation).toBe('salted');
    // The field is written as the macro, not as a literal, so the decimal
    // reader cannot see it — assert on the initialiser text instead.
    const at = HEADER.indexOf('PEARL_MAINNET_PROFILE');
    const init = HEADER.slice(at, HEADER.indexOf('}', at));
    expect(init).toContain('PEARL_SEED_SALTED');
  });

  // The operand fill is not protocol, but the two sides must still name the same
  // one: the JS profile's code is what the addon is handed, and a profile with no
  // code falls back to the C default.
  test('both sides default to the constant operand fill, and it is int7', () => {
    expect(defineOf('PEARL_OPERAND_HASHED')).toBe(0);
    expect(defineOf('PEARL_OPERAND_CONST')).toBe(1);
    expect(PROFILE.operandFillCode).toBe(defineOf('PEARL_OPERAND_CONST'));
    const at = HEADER.indexOf('PEARL_MAINNET_PROFILE =');
    const init = HEADER.slice(at, HEADER.indexOf('}', at));
    expect(init).toContain('PEARL_OPERAND_CONST');
    // The noise adds another int7, and the sum must stay inside int8.
    const fill = defineOf('PEARL_OPERAND_FILL');
    expect(fill).not.toBeNull();
    expect(Math.abs(fill)).toBeLessThanOrEqual(63);
  });

  // job_key is UNKEYED. Hashing it keyed with a zero key is a different
  // function, and was what made the device and the oracle disagree silently.
  test('the device derives job_key with the unkeyed kernel', () => {
    const host = fs.readFileSync(
      path.join(__dirname, '..', 'native', 'src', 'pearl_host.cu'), 'utf8');
    expect(host).toContain('pearl_blake3_unkeyed<<<1, 1>>>(dSeedInput');
    expect(host).not.toContain('zeroKey');
  });
});

// The share test the folds run on the device, in pearl_kernel.cu. The host
// re-checks every hit with meetsTarget before submitting, so a byte-order slip
// here cannot send a bad share. It makes the device test the wrong number: it
// skips real shares, and the hits it does report fail the re-check.
//
// Nothing here can run the device code, so the source is checked for the order
// it uses, and a JS transcription of that order is checked against meetsTarget.
describe('native target test', () => {
  const SRC = path.join(__dirname, '..', 'native', 'src');
  const KERNEL = fs.readFileSync(path.join(SRC, 'pearl_kernel.cu'), 'utf8');
  const HOST = fs.readFileSync(path.join(SRC, 'pearl_host.cu'), 'utf8');
  const FOLD_BD = fs.readFileSync(path.join(SRC, 'pearl_fold_bd.cuh'), 'utf8');
  const NL = String.fromCharCode(10);

  // A function's source, from its signature to the brace that closes it at the
  // start of a line.
  function body(src, signature) {
    const at = src.indexOf(signature);
    expect(at).toBeGreaterThanOrEqual(0);
    const end = src.indexOf(NL + '}', at);
    expect(end).toBeGreaterThan(at);
    return src.slice(at, end);
  }

  function count(src, needle) {
    let n = 0;
    for (let at = src.indexOf(needle); at >= 0; at = src.indexOf(needle, at + 1)) n++;
    return n;
  }

  // The hash comes out of BLAKE3 as eight little-endian words. Read as a
  // little-endian number its top word is word 7; the target is packed as
  // big-endian words, most significant first. The full compare walks both from
  // the top, and a tie is a share.
  test('the full compare reads the hash little-endian, from word 7, against big-endian target words', () => {
    expect(body(KERNEL, 'uint32_t pearl_bswap32(')).toContain('__byte_perm(x, 0u, 0x0123u)');
    expect(body(KERNEL, 'uint32_t pearl_hash_word_msf('))
      .toContain('return hash_big_endian ? pearl_bswap32(h[i]) : h[7 - i];');
    const meets = body(KERNEL, 'bool pearl_hash_meets_words(');
    expect(meets).toContain('for (int i = 0; i < 8; i++) {');
    expect(meets).toContain('const uint32_t hw = pearl_hash_word_msf(h, i, hash_big_endian);');
    expect(meets).toContain('if (hw != target_w[i]) return hw < target_w[i];');
    expect(meets).toContain('return true;  // exactly equal counts as a share');
    expect(HOST).toContain('const uint8_t *t = ctx->target + i * 4;');
    expect(HOST).toContain('test.target_w[i] = ((uint32_t)t[0] << 24) | ((uint32_t)t[1] << 16) |');
    expect(HOST).toContain('((uint32_t)t[2] << 8) | (uint32_t)t[3];');
  });

  // The folds test only the top word for every region, and run the full
  // compare on the few that pass. Each top-word test computes that word itself,
  // so each has to pick the same word: output word 7 (s7 ^ s15 inside the
  // compression), or word 0 byte-reversed under hash_big_endian.
  test('every top-word test takes the word the full compare starts with', () => {
    const sites = [
      ['uint32_t pearl_transcript_msw(', 'hash_big_endian ? pearl_bswap32(out16[0]) : out16[7]'],
      ['uint32_t pearl_hp_msw(', 'hbe ? pearl_bswap32(s[0] ^ s[8]) : (s[7] ^ s[15])'],
      // A lane pair splits the state by columns: half 0 holds s0 and s8, half 1
      // holds s7 and s15, and the caller reads the half that has the word.
      ['uint32_t pearl_hp_msw_pair(', 'hbe ? pearl_bswap32(A[0] ^ C[0]) : (B[1] ^ Dd[1])'],
      ['void pearl_tall_hash80(', 'test.hash_big_endian ? pearl_bswap32(v[0] ^ v[8]) : (v[7] ^ v[15])'],
    ];
    for (const [signature, choice] of sites) expect(body(KERNEL, signature)).toContain(choice);
    expect(KERNEL).toContain('hh == (test.hash_big_endian ? 0u : 1u) && w2 <= test.target_w[0]');
    // These and pearl_hash_word_msf's are the only places that pick the word.
    // A new one has to be added above.
    expect(count(KERNEL, '? pearl_bswap32(')).toBe(sites.length + 1);
    expect(count(FOLD_BD, 'pearl_bswap32(')).toBe(0);
    // And each test passes a tie on to the full compare: it rejects only a top
    // word greater than the target's.
    for (const src of [KERNEL, FOLD_BD]) {
      const uses = count(src, 'test.target_w[0]');
      expect(uses).toBeGreaterThan(0);
      expect(count(src, '> test.target_w[0]) return;') + count(src, '<= test.target_w[0]) record('))
        .toBe(uses);
    }
  });

  // The order pinned above, transcribed: hash bytes as little-endian words,
  // compared from word 7 against the target's big-endian words.
  function deviceMeets(hash, target) {
    const t = Buffer.from(target.toString(16).padStart(64, '0'), 'hex');
    for (let i = 0; i < 8; i++) {
      const hw = hash.readUInt32LE(4 * (7 - i));
      const tw = t.readUInt32BE(4 * i);
      if (hw !== tw) return hw < tw;
    }
    return true;
  }

  // Deterministic, so a failure reproduces.
  function bytes(seed) {
    return crypto.createHash('sha256').update(String(seed)).digest();
  }

  function asNumber(hash) {
    let v = 0n;
    for (let i = 31; i >= 0; i--) v = (v << 8n) | BigInt(hash[i]);
    return v;
  }

  test('that order is the one meetsTarget uses', () => {
    const top = (1n << 256n) - 1n;
    let checked = 0;
    for (let s = 0; s < 200; s++) {
      const hash = bytes('hash ' + s);
      const h = asNumber(hash);
      // Targets around the hash, so every word position decides some case.
      const targets = [h, h - 1n, h + 1n, asNumber(bytes('target ' + s))];
      for (let w = 0; w < 8; w++) targets.push(h ^ (1n << BigInt(32 * w + (s % 32))));
      for (const target of targets) {
        if (target < 0n || target > top) continue;
        expect(deviceMeets(hash, target)).toBe(meetsTarget(hash, target));
        // The top-word test may only reject a hash the full compare rejects.
        const msw = hash.readUInt32LE(28);
        const tw0 = Number(target >> 224n);
        if (msw > tw0) expect(meetsTarget(hash, target)).toBe(false);
        checked++;
      }
    }
    expect(checked).toBeGreaterThan(2000);
  });
});

// The value of a #define that has a numeric one. Some names are first defined as
// an alias (PEARL_WARP_ROWS is PEARL_FOLD_WIDE_WARP_ROWS in the Ada build) and
// only later as a number, so this skips past aliases instead of stopping at the
// first match as defineOf does.
function numericDefineOf(name) {
  const key = '#define ' + name + ' ';
  for (let at = HEADER.indexOf(key); at >= 0; at = HEADER.indexOf(key, at + 1)) {
    const eol = HEADER.indexOf(String.fromCharCode(10), at);
    const n = parseInt(HEADER.slice(at + key.length, eol < 0 ? undefined : eol).trim(), 10);
    if (Number.isFinite(n)) return n;
  }
  return null;
}

describe('Turing fold geometry', () => {
  // Compared line by line, trimmed: a Windows checkout has CRLF endings.
  const LINES = HEADER.split(String.fromCharCode(10)).map((l) => l.trim());
  const SM75_BLOCK =
    '#if !defined(PEARL_FOLD_THREADS) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800';

  function defines(names) {
    const out = {};
    for (const name of names) {
      out[name] = numericDefineOf(name);
      expect(out[name]).not.toBeNull();
    }
    return out;
  }

  // Turing grants a block at most 64 KB of shared memory. If a fold outgrows
  // it, every RTX 20 card stops at its first search with "fold needs ... card
  // allows 65536". These are the host's formulas (pearl_host.cu, the smem it
  // opts in to) with Turing's thread count.
  //
  // The fold that ships is the B-direct one: two A stages of the tile's rows,
  // then the block's transcripts, 64 bytes a region (pearl_fold_bd.cuh).
  function bdirectSmem() {
    const d = defines(['PEARL_FOLD_TURING_THREADS', 'PEARL_FOLD_WIDE_WARP_ROWS',
      'PEARL_FOLD_WIDE_ROW_TILES', 'PEARL_WMMA_COL_BLK', 'PEARL_WMMA_ROWS', 'PEARL_ROWS_COUNT',
      'PEARL_SB_STRIDE', 'PEARL_BD_A_STAGES', 'PEARL_JACKPOT_BUCKETS']);
    const warps = d.PEARL_FOLD_TURING_THREADS / 32;
    const regionsPerWarp = d.PEARL_FOLD_WIDE_ROW_TILES * (d.PEARL_WMMA_ROWS / d.PEARL_ROWS_COUNT);
    return d.PEARL_BD_A_STAGES * d.PEARL_FOLD_WIDE_WARP_ROWS * regionsPerWarp * d.PEARL_ROWS_COUNT
        * d.PEARL_SB_STRIDE
      + warps * regionsPerWarp * d.PEARL_WMMA_COL_BLK * d.PEARL_JACKPOT_BUCKETS * 4;
  }

  // The fallbacks stage both operands: one stage of the 128x256 tile
  // (PEARL_TURING_BDIRECT=0), or two of the 128x128 one (PEARL_TURING_WIDE=0).
  function stagedSmem(wide) {
    const d = defines(['PEARL_FOLD_TURING_THREADS', 'PEARL_WMMA_COL_BLK', 'PEARL_WMMA_ROWS',
      'PEARL_ROWS_COUNT', 'PEARL_SB_STRIDE']);
    const warpRows = numericDefineOf(wide ? 'PEARL_FOLD_WIDE_WARP_ROWS' : 'PEARL_WARP_ROWS');
    const rowTiles = numericDefineOf(wide ? 'PEARL_FOLD_WIDE_ROW_TILES' : 'PEARL_WMMA_ROW_TILES');
    const stages = numericDefineOf(wide ? 'PEARL_TURING_WIDE_STAGE_BUFS' : 'PEARL_STAGE_BUFS');
    for (const v of [warpRows, rowTiles, stages]) expect(v).not.toBeNull();
    const warpCols = d.PEARL_FOLD_TURING_THREADS / 32 / warpRows;
    expect(Number.isInteger(warpCols) && warpCols > 0).toBe(true);
    const regionsPerWarp = rowTiles * (d.PEARL_WMMA_ROWS / d.PEARL_ROWS_COUNT);
    return stages * (warpCols * d.PEARL_WMMA_COL_BLK * 16 + warpRows * regionsPerWarp
      * d.PEARL_ROWS_COUNT) * d.PEARL_SB_STRIDE;
  }

  // An unflagged build ships the B-direct fold with two row groups: what the
  // 2080 Ti measured fastest.
  test('the default is the B-direct fold, two row groups, on the 128x256 tile', () => {
    expect(numericDefineOf('PEARL_TURING_WIDE')).toBe(1);
    expect(numericDefineOf('PEARL_BD_GROUPS')).toBe(2);
    // B-direct follows PEARL_TURING_WIDE, so -DPEARL_TURING_WIDE=0 alone still
    // builds the 128x128 fold.
    const at = LINES.indexOf('#ifndef PEARL_TURING_BDIRECT');
    expect(at).toBeGreaterThanOrEqual(0);
    expect(LINES.slice(at + 1, at + 7)).toEqual([
      '#if PEARL_TURING_WIDE',
      '#define PEARL_TURING_BDIRECT 1',
      '#else',
      '#define PEARL_TURING_BDIRECT 0',
      '#endif',
      '#endif',
    ]);
  });

  // 40 KB is also what the kernel lays out (a static_assert in the fold). It
  // takes Turing's 64 KB shared carveout and leaves 32 KB of L1 for B.
  test('the B-direct fold takes 40 KB of shared, as the kernel lays it out', () => {
    expect(bdirectSmem()).toBe(40960);
    expect(bdirectSmem()).toBeLessThanOrEqual(65536);
    const fold = fs.readFileSync(
      path.join(__dirname, '..', 'native', 'src', 'pearl_fold_bd.cuh'), 'utf8');
    expect(fold).toContain('== 40960u');
  });

  test('both fallbacks fit the 64 KB of shared Turing grants a block', () => {
    expect(stagedSmem(true)).toBeLessThanOrEqual(65536);
    expect(stagedSmem(false)).toBeLessThanOrEqual(65536);
  });

  // The kernel picks Turing's thread count, and the B-direct body, only for
  // sm_75 builds. Everything from sm_80 up keeps its own fold.
  test('only builds below sm_80 take it', () => {
    const at = LINES.indexOf(SM75_BLOCK);
    expect(at).toBeGreaterThanOrEqual(0);
    expect(LINES[at + 1]).toBe('#define PEARL_FOLD_THREADS PEARL_FOLD_TURING_THREADS');
  });

  // The tests above size each shape from the names the host reads. The sm_75
  // kernel must map its own warp grid and stage count onto those same names,
  // and build the B-direct body exactly when the host's switch says so, or the
  // host would write B' in a layout the fold does not read.
  test('the sm_75 kernel builds its shape from the switches the host reads', () => {
    const at = LINES.indexOf(SM75_BLOCK);
    expect(LINES.slice(at + 2, at + 9)).toEqual([
      '#if PEARL_TURING_WIDE',
      '#define PEARL_WARP_ROWS PEARL_FOLD_WIDE_WARP_ROWS',
      '#define PEARL_WMMA_ROW_TILES PEARL_FOLD_WIDE_ROW_TILES',
      '#define PEARL_STAGE_BUFS PEARL_TURING_WIDE_STAGE_BUFS',
      '#endif',
      '#define PEARL_TURING_BD_BODY PEARL_TURING_BDIRECT',
      '#endif',
    ]);
    // Defined nowhere else but as the 0 every other pass gets.
    const bodies = LINES.filter((l) => l.startsWith('#define PEARL_TURING_BD_BODY '));
    expect(bodies).toEqual([
      '#define PEARL_TURING_BD_BODY PEARL_TURING_BDIRECT',
      '#define PEARL_TURING_BD_BODY 0',
    ]);
  });
});
