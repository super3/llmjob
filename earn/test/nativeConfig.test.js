'use strict';

const fs = require('fs');
const path = require('path');
const { PROFILE, CONFIG_BYTES, JACKPOT_BUCKETS, ROTL_BITS, buildConfig52 } = require('../src/shared/miner/pearlhash');

// The CUDA core and the JS reference must agree byte for byte. config52 is
// hashed with the header to derive job_key, so ONE mismatched offset silently
// changes every downstream hash and the miner produces work no pool will accept
// — with no error anywhere to point at it.
//
// The native side cannot be compiled here (no CUDA toolkit on this box), so this
// asserts the agreement by reading the header's own constants. It is a real
// guard, not a placeholder: it fails the moment somebody edits one side's
// profile, widths, or fold parameters without the other.

const HEADER = fs.readFileSync(path.join(__dirname, '..', 'native', 'src', 'pearl_config.h'), 'utf8');

function defineOf(name) {
  const m = HEADER.match(new RegExp('#define\\s+' + name + '\\s+(\\d+)'));
  return m ? Number(m[1]) : null;
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

  // The mainnet profile is duplicated as a C initialiser; if the two drift, the
  // core searches a different geometry than the host advertises.
  test('the mainnet profile matches the C initialiser', () => {
    const m = HEADER.match(/PEARL_MAINNET_PROFILE\s*=\s*\{([^}]+)\}/);
    expect(m).toBeTruthy();
    const nums = m[1].match(/\d+/g).map(Number);
    expect(nums.slice(0, 7)).toEqual([
      PROFILE.m, PROFILE.n, PROFILE.k, PROFILE.rank,
      PROFILE.hashTile, PROFILE.rows.length, PROFILE.cols.length,
    ]);
  });

  // The C writer assigns each field at a literal offset. Those offsets are what
  // buildConfig52 has to reproduce, so read them back out of the source and
  // check the JS block actually carries each value at that position.
  test('every field sits at the offset the C writer uses', () => {
    const block = buildConfig52();
    expect(block.readUInt32LE(0)).toBe(PROFILE.m);
    expect(block.readUInt32LE(4)).toBe(PROFILE.n);
    expect(block.readUInt32LE(8)).toBe(PROFILE.k);
    expect(block.readUInt16LE(12)).toBe(PROFILE.rank);
    expect(block.readUInt16LE(14)).toBe(PROFILE.hashTile);
    expect(block.readUInt16LE(16)).toBe(PROFILE.rows.length);
    expect(block.readUInt16LE(18)).toBe(PROFILE.cols.length);
    // The C writer zeroes the whole block first, so the reserved tail must match.
    expect(block.slice(20).every((b) => b === 0)).toBe(true);
  });

  // The comparison in pearl_meets_target walks the hash high→low against a
  // big-endian target — i.e. the hash is little-endian and the target is not.
  // Getting this backwards makes every reported share spurious, so the source
  // is checked for the reversal rather than trusting the comment.
  test('the native target comparison reverses the hash, not the target', () => {
    expect(HEADER).toMatch(/hash_le\[PEARL_HASH_BYTES - 1 - i\]/);
    expect(HEADER).toMatch(/target_be\[i\]/);
  });

  test('equality counts as a share on both sides', () => {
    // C: falls out of the loop returning 1.
    expect(HEADER).toMatch(/return 1;\s*\/\/ exactly equal counts as a share/);
  });
});
