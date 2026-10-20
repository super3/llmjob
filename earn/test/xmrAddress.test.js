'use strict';

const {
  ALPHABET, BLOCK_CHARS, decodeBase58, encodeBase58, checkXmrAddress, shortenXmrAddress,
} = require('../src/shared/xmr/address');
const { keccak256 } = require('../src/shared/xmr/keccak');

// The founder's payout address (a subaddress) and the Monero General Fund's
// published donation address (a standard one).
const SUBADDRESS = '87wTMj88rEGV3BFiYXJc2UHeTGmHikMRWAvAskmPgaF3PZxuxonkKaJZsprNmuhM1Aca7P6UbyatGFHkQfUTYgoNN1JxND9';
const STANDARD = '44AFFq5kSiGBoZ4NMDwYtN18obc8AemS33DBLWs3H7otXft3XjrpDtQGv7SqSsaBYBb98uNbr2VBBEt7f2wfn3RVGQBEP3A';

// An address of any network byte and body length, with a correct checksum.
function makeAddress(network, bodyBytes) {
  const body = Buffer.concat([Buffer.from([network]), Buffer.alloc(bodyBytes, 7)]);
  return encodeBase58(Buffer.concat([body, keccak256(body).subarray(0, 4)]));
}

describe('Monero base58', () => {
  test('round-trips real addresses', () => {
    for (const a of [SUBADDRESS, STANDARD]) {
      const bytes = decodeBase58(a);
      expect(bytes).toHaveLength(69);
      expect(encodeBase58(bytes)).toBe(a);
    }
  });

  // 8 bytes make 11 characters and a short last block takes fewer; that is
  // what makes a 69-byte address 95 characters.
  test('uses Monero\'s block sizes', () => {
    expect(BLOCK_CHARS).toEqual([0, 2, 3, 5, 6, 7, 9, 10, 11]);
    expect(encodeBase58(Buffer.alloc(69))).toHaveLength(95);
    expect(encodeBase58(Buffer.from([0]))).toBe('11');
    expect(decodeBase58('11')).toEqual(Buffer.from([0]));
  });

  test('refuses what is not Monero base58', () => {
    expect(decodeBase58('')).toBeNull();
    expect(decodeBase58(null)).toBeNull();
    expect(decodeBase58('1')).toBeNull();            // no block is 1 character
    expect(decodeBase58('1111')).toBeNull();         // nor 4
    expect(decodeBase58('0O')).toBeNull();           // not in the alphabet
    expect(decodeBase58('zz')).toBeNull();           // 3363 does not fit one byte
    expect(ALPHABET).not.toMatch(/[0OIl]/);
  });
});

describe('checkXmrAddress', () => {
  test('takes a standard address and a subaddress', () => {
    expect(checkXmrAddress(STANDARD)).toEqual({ ok: true, kind: 'standard' });
    expect(checkXmrAddress(SUBADDRESS)).toEqual({ ok: true, kind: 'subaddress' });
    expect(checkXmrAddress('  ' + SUBADDRESS + '\n')).toEqual({ ok: true, kind: 'subaddress' });
  });

  test('catches a typo with the checksum', () => {
    const typo = SUBADDRESS.slice(0, -1) + '8';
    expect(checkXmrAddress(typo)).toEqual({ ok: false, error: 'the checksum does not match: check for a typo' });
  });

  test('says why other input is not an address', () => {
    expect(checkXmrAddress('')).toEqual({ ok: false, error: 'enter a Monero address' });
    expect(checkXmrAddress(undefined).error).toBe('enter a Monero address');
    expect(checkXmrAddress('prl1pql8r6m4z9x7v2k0t3whu8e2snd4p6c').error).toBe('not a Monero address');
    expect(checkXmrAddress('11').error).toBe('not a Monero address');   // too short to carry a checksum
    expect(checkXmrAddress(SUBADDRESS + 'x').error).toBe('not a Monero address');
  });

  // Integrated addresses carry a payment ID; pools read "." after an address
  // as one, and an exchange deposit needing one is not what this field is for.
  test('refuses integrated, testnet and stagenet addresses with a reason', () => {
    expect(checkXmrAddress(makeAddress(19, 72)).error).toMatch(/^integrated addresses/);
    expect(checkXmrAddress(makeAddress(53, 64)).error).toBe('this is a testnet address, not a mainnet one');
    expect(checkXmrAddress(makeAddress(24, 64)).error).toBe('this is a stagenet address, not a mainnet one');
  });

  test('refuses an unknown network byte, or a known one at the wrong length', () => {
    expect(checkXmrAddress(makeAddress(7, 64)).error).toBe('not a Monero address');
    expect(checkXmrAddress(makeAddress(18, 63)).error).toBe('not a Monero address');
    expect(makeAddress(18, 64)).toMatch(/^4/);
    expect(checkXmrAddress(makeAddress(18, 64))).toEqual({ ok: true, kind: 'standard' });
  });
});

describe('shortenXmrAddress', () => {
  test('keeps the ends, and the case', () => {
    expect(shortenXmrAddress(SUBADDRESS)).toBe('87wTMj88…1JxND9');
    expect(shortenXmrAddress('  4abc  ')).toBe('4abc');
    expect(shortenXmrAddress(null)).toBe('');
  });
});
