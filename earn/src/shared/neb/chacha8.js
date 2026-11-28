'use strict';

// ChaCha8 as NEB uses it (spec/shares.md section 3.2 of github.com/SilasEigen/neb):
// Bernstein's original layout, 8 rounds, a 64-bit block counter in words 12
// and 13, a zero IV, the key as eight little-endian words. Ported from the
// chain's C reference (pow/ref/chacha8.c, Apache-2.0).

function rotl(v, n) {
  return (v << n) | (v >>> (32 - n));
}

// One 64-byte block as 16 words, for a key given as 8 words.
function blockWords(key, counter) {
  const s = new Uint32Array(16);
  s[0] = 0x61707865; s[1] = 0x3320646e; s[2] = 0x79622d32; s[3] = 0x6b206574;
  for (let i = 0; i < 8; i++) s[4 + i] = key[i];
  s[12] = counter >>> 0;
  s[13] = Math.floor(counter / 0x100000000) >>> 0;
  const x = Uint32Array.from(s);
  const qr = (a, b, c, d) => {
    x[a] += x[b]; x[d] = rotl(x[d] ^ x[a], 16);
    x[c] += x[d]; x[b] = rotl(x[b] ^ x[c], 12);
    x[a] += x[b]; x[d] = rotl(x[d] ^ x[a], 8);
    x[c] += x[d]; x[b] = rotl(x[b] ^ x[c], 7);
  };
  for (let r = 0; r < 4; r++) {
    qr(0, 4, 8, 12); qr(1, 5, 9, 13); qr(2, 6, 10, 14); qr(3, 7, 11, 15);
    qr(0, 5, 10, 15); qr(1, 6, 11, 12); qr(2, 7, 8, 13); qr(3, 4, 9, 14);
  }
  for (let i = 0; i < 16; i++) x[i] += s[i];
  return x;
}

function keyWords(key) {
  const w = new Uint32Array(8);
  for (let i = 0; i < 8; i++) w[i] = key.readUInt32LE(4 * i);
  return w;
}

// The first n bytes of the keystream of a 32-byte key.
function stream(key, n) {
  const k = keyWords(key);
  const out = Buffer.alloc(n);
  for (let counter = 0, at = 0; at < n; counter++) {
    const x = blockWords(k, counter);
    for (let i = 0; i < 16 && at < n; i++) {
      for (let b = 0; b < 4 && at < n; b++) out[at++] = (x[i] >>> (8 * b)) & 0xff;
    }
  }
  return out;
}

module.exports = { blockWords, keyWords, stream };
