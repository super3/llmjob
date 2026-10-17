'use strict';

const crypto = require('crypto');
const { keccak256, sponge } = require('../src/shared/xmr/keccak');

describe('keccak256', () => {
  // The published Keccak-256 values (the pre-SHA3 padding Monero and Ethereum
  // use), for the empty string and "abc".
  test('matches the known Keccak-256 vectors', () => {
    expect(keccak256(Buffer.alloc(0)).toString('hex'))
      .toBe('c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470');
    expect(keccak256(Buffer.from('abc')).toString('hex'))
      .toBe('4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45');
  });

  test('takes an array of bytes as well as a Buffer', () => {
    expect(keccak256([0x61, 0x62, 0x63]).equals(keccak256(Buffer.from('abc')))).toBe(true);
  });

  // SHA3-256 is the same permutation with a different padding byte, and Node
  // has it. Agreeing with it over lengths around the 136-byte block (empty,
  // one short of a block, exactly one, several) checks the permutation and the
  // absorbing; the two vectors above check the Keccak padding.
  test('with SHA3 padding, agrees with Node\'s sha3-256 at every block boundary', () => {
    for (const n of [0, 1, 55, 135, 136, 137, 271, 272, 273, 500]) {
      const data = crypto.createHash('sha512').update(String(n)).digest();
      const input = Buffer.alloc(n);
      for (let i = 0; i < n; i++) input[i] = data[i % data.length] ^ i;
      expect(sponge(input, 0x06).toString('hex'))
        .toBe(crypto.createHash('sha3-256').update(input).digest('hex'));
    }
  });
});
