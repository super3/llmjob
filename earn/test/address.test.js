'use strict';

const {
  normalizeAddress, isValidAddress, isValidMdlAddress,
  combinePayoutAddress, shortenAddress,
} = require('../src/shared/address');

const VALID = 'prl1pql8r6m4z9x7v2k0t3whu8e2snd4p6c';
const VALID_MDL = 'mdl1pql8r6m4z9x7v2k0t3whu8e2snd4p6c';

describe('address', () => {
  test('normalizeAddress trims, lowercases, and handles null', () => {
    expect(normalizeAddress(null)).toBe('');
    expect(normalizeAddress(undefined)).toBe('');
    expect(normalizeAddress('  PRL1Pabc  ')).toBe('prl1pabc');
  });

  test('isValidAddress accepts a well-formed Pearl address', () => {
    expect(isValidAddress(VALID)).toBe(true);
    expect(isValidAddress('  ' + VALID.toUpperCase() + '  ')).toBe(true);
  });

  test('isValidAddress rejects malformed addresses', () => {
    expect(isValidAddress('')).toBe(false);
    expect(isValidAddress('btc1pql8r6m4z9x7v2k0t3whu8e2snd4p6c')).toBe(false); // wrong prefix
    expect(isValidAddress('prl1pshort')).toBe(false); // too short
    expect(isValidAddress(VALID + '!')).toBe(false); // illegal char
    expect(isValidAddress('prl1p' + 'a'.repeat(90))).toBe(false); // too long
  });

  test('shortenAddress collapses long addresses and leaves short ones', () => {
    expect(shortenAddress(VALID)).toBe('prl1pql8…d4p6c');
    expect(shortenAddress(VALID)).toContain('…');
    expect(shortenAddress('prl1p')).toBe('prl1p');
    expect(shortenAddress(null)).toBe('');
  });

  test('isValidMdlAddress accepts mdl1p… and rejects everything else', () => {
    expect(isValidMdlAddress(VALID_MDL)).toBe(true);
    expect(isValidMdlAddress('  ' + VALID_MDL.toUpperCase() + '  ')).toBe(true);
    expect(isValidMdlAddress(VALID)).toBe(false);   // a Pearl address is not an MDL one
    expect(isValidMdlAddress('mdl1pshort')).toBe(false);
    expect(isValidMdlAddress('')).toBe(false);
    expect(isValidMdlAddress(null)).toBe(false);
  });

  test('combinePayoutAddress appends a valid MDL address for merge mining', () => {
    expect(combinePayoutAddress(VALID, VALID_MDL)).toBe(VALID + '+' + VALID_MDL);
    expect(combinePayoutAddress('  ' + VALID + '  ', VALID_MDL)).toBe(VALID + '+' + VALID_MDL); // PRL trimmed
  });

  test('combinePayoutAddress falls back to Pearl-only when the MDL side is absent or bad', () => {
    expect(combinePayoutAddress(VALID, '')).toBe(VALID);
    expect(combinePayoutAddress(VALID, null)).toBe(VALID);
    expect(combinePayoutAddress(VALID, 'mdl1pshort')).toBe(VALID); // typo → never corrupts --address
    expect(combinePayoutAddress('', VALID_MDL)).toBe('');          // no Pearl address → nothing to mine
    expect(combinePayoutAddress(null, null)).toBe('');
  });
});
