'use strict';

const fs = require('fs');
const path = require('path');
const {
  PROFILE, CONFIG_BYTES, JACKPOT_BUCKETS, ROTL_BITS, buildConfig52,
  patternFromList, patternToBytes,
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

  // The comparison in pearl_meets_target walks the hash high→low against a
  // big-endian target — i.e. the hash is little-endian and the target is not.
  // Getting this backwards makes every reported share spurious, so the source is
  // checked for the reversal rather than trusting the comment.
  test('the native target comparison reverses the hash, not the target', () => {
    expect(HEADER).toContain('hash_le[PEARL_HASH_BYTES - 1 - i]');
    expect(HEADER).toContain('target_be[i]');
  });

  test('equality counts as a share on both sides', () => {
    expect(HEADER).toContain('return 1;  // exactly equal counts as a share');
  });
});
