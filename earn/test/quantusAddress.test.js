'use strict';

const a = require('../src/shared/quantus/address');

// Addresses made here from fixed accounts. The checksum scheme was checked
// against Quantus's own: every one of the 66 addresses in the chain's mainnet
// genesis (runtime/src/genesis_config_presets/mainnet_vesting.rs) passes
// checkQuantusAddress, and so does the example on Kryptex's "How to mine
// Quantus" page. Neither is copied here: they are other people's addresses.
const ZERO = a.encodeQuantusAddress(Buffer.alloc(32));
const SEQ = a.encodeQuantusAddress(Buffer.from(Array.from({ length: 32 }, (_, i) => i)));

test('an address starts qz and is 49 characters', () => {
  expect(ZERO).toBe('qzjTEdq3Wd1d98kaeE1RTDPBBYBfY6XpfCtaLq97wmfcACqCd');
  expect(SEQ.startsWith('qz')).toBe(true);
  expect(SEQ).toHaveLength(49);
  expect(a.checkQuantusAddress(' ' + SEQ + ' ')).toEqual({ ok: true });
  expect(a.isValidQuantusAddress(ZERO)).toBe(true);
});

test('what is wrong with an address, said', () => {
  expect(a.checkQuantusAddress('')).toEqual({ ok: false, error: 'empty' });
  expect(a.checkQuantusAddress(null)).toEqual({ ok: false, error: 'empty' });
  expect(a.checkQuantusAddress('prl1pabc')).toEqual({ ok: false, error: 'a Quantus address starts with qz' });
  expect(a.checkQuantusAddress('qz0OIl')).toEqual({ ok: false, error: 'not base58' });
  expect(a.checkQuantusAddress('qzabc')).toEqual({ ok: false, error: 'wrong length' });
  // One character changed.
  const typo = SEQ.slice(0, 20) + (SEQ[20] === 'a' ? 'b' : 'a') + SEQ.slice(21);
  expect(a.checkQuantusAddress(typo)).toEqual({ ok: false, error: 'bad checksum (a typo?)' });
  expect(a.isValidQuantusAddress(typo)).toBe(false);
});

test('another SS58 network\'s address is refused', () => {
  // A prefix-445 address has the length and starts qz too, and is still not
  // Quantus's.
  const crypto = require('crypto');
  const body = Buffer.concat([Buffer.from([((445 & 0xfc) >> 2) | 0x40, (445 >> 8) | ((445 & 3) << 6)]), Buffer.alloc(32)]);
  const sum = crypto.createHash('blake2b512').update(Buffer.concat([Buffer.from('SS58PRE'), body])).digest().subarray(0, 2);
  const other = a.base58Encode(Buffer.concat([body, sum]));
  expect(other.startsWith('qz')).toBe(true);
  expect(a.checkQuantusAddress(other)).toEqual({ ok: false, error: 'not a Quantus network address' });
});

test('base58 keeps leading zero bytes', () => {
  const buf = Buffer.from([0, 0, 1, 2]);
  expect(a.base58Encode(buf)).toBe('115T');
  expect(a.base58Decode('115T')).toEqual(buf);
  expect(a.base58Encode(Buffer.alloc(0))).toBe('');
  expect(a.base58Decode('11')).toEqual(Buffer.alloc(2));
});

test('encoding wants a 32-byte account', () => {
  expect(() => a.encodeQuantusAddress(Buffer.alloc(31))).toThrow('an account is 32 bytes');
  expect(() => a.encodeQuantusAddress('x'.repeat(32))).toThrow('an account is 32 bytes');
});

test('shortened for logs', () => {
  expect(a.shortenQuantusAddress(ZERO)).toBe('qzjTEdq3…cACqCd');
  expect(a.shortenQuantusAddress('qzabc')).toBe('qzabc');
  expect(a.shortenQuantusAddress(null)).toBe('');
});
