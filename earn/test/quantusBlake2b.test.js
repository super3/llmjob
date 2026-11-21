'use strict';

const crypto = require('crypto');
const { blake2b512 } = require('../src/shared/quantus/blake2b');

describe('BLAKE2b-512 in JavaScript', () => {
  test('the published digests of "" and "abc" (RFC 7693, appendix A)', () => {
    expect(blake2b512(Buffer.alloc(0)).toString('hex')).toBe(
      '786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419'
      + 'd25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce');
    expect(blake2b512(Buffer.from('abc')).toString('hex')).toBe(
      'ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d1'
      + '7d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923');
  });

  // Node's own BLAKE2b (OpenSSL) is the reference; Electron's crypto has none.
  test('matches Node\'s on every length around the 128-byte block', () => {
    for (const n of [1, 36, 41, 127, 128, 129, 255, 256, 257, 1000]) {
      const data = crypto.randomBytes(n);
      expect(blake2b512(data).equals(crypto.createHash('blake2b512').update(data).digest())).toBe(true);
    }
  });
});

// The desktop app's crypto (Electron's BoringSSL) throws for BLAKE2b, which
// is how a Quantus START ran nothing there. The address check must not need it.
describe('a Quantus address without Node\'s BLAKE2b', () => {
  test('still checks out, and a typo still fails', () => {
    const real = crypto.createHash;
    const spy = jest.spyOn(crypto, 'createHash').mockImplementation((alg, ...rest) => {
      if (/blake2b/i.test(alg)) throw new Error('Digest method not supported');
      return real.call(crypto, alg, ...rest);
    });
    try {
      jest.isolateModules(() => {
        const a = require('../src/shared/quantus/address');
        const good = a.encodeQuantusAddress(Buffer.alloc(32, 7));
        expect(a.checkQuantusAddress(good)).toEqual({ ok: true });
        const typo = good.slice(0, -1) + (good.endsWith('g') ? 'h' : 'g');
        expect(a.checkQuantusAddress(typo).ok).toBe(false);
      });
      expect(spy).not.toHaveBeenCalledWith(expect.stringMatching(/blake2b/i));
    } finally {
      spy.mockRestore();
    }
  });
});
