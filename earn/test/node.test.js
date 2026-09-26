'use strict';

const nacl = require('tweetnacl');
const naclUtil = require('tweetnacl-util');
const crypto = require('crypto');
const {
  generateKeypair, fingerprint, pingMessage, signMessage, signRig, NODE_ID_HEX,
} = require('../src/shared/node');

describe('generateKeypair / fingerprint', () => {
  test('makes a base64 Ed25519 keypair and a 16-hex fingerprint', () => {
    const kp = generateKeypair();
    expect(naclUtil.decodeBase64(kp.publicKey).length).toBe(32);
    expect(naclUtil.decodeBase64(kp.secretKey).length).toBe(64);
    expect(fingerprint(kp.publicKey)).toMatch(/^[0-9a-f]{16}$/);
  });

  test('fingerprint is stable per key and tolerates a nullish input', () => {
    expect(fingerprint('abc')).toBe(fingerprint('abc'));
    expect(fingerprint(null)).toMatch(/^[0-9a-f]{16}$/);
  });

  // 6 hex characters (24 bits) put two honest nodes on the same id with ~3%
  // probability at 1,000 nodes and 52% at 5,000 — and the loser was silently
  // unusable, its signed pings refused for a key mismatch. 16 characters
  // (64 bits) pushes the same 50% point past a billion nodes.
  test('is wide enough that the fleet will not collide', () => {
    expect(NODE_ID_HEX).toBe(16);
    expect(fingerprint('a')).not.toBe(fingerprint('b'));
  });

  // The server recomputes the id from the public key to check a report's claim.
  // If the two ever disagree no rig could be identified, so pin the exact digest
  // this end produces.
  test('is sha256(publicKey) truncated, which is what the server computes', () => {
    const expected = crypto.createHash('sha256').update('somekey').digest('hex').slice(0, 16);
    expect(fingerprint('somekey')).toBe(expected);
    expect(fingerprint('')).toBe(fingerprint(null));
  });
});

describe('pingMessage / signMessage', () => {
  test('signs the challenge so the server (nacl.verify) accepts it', () => {
    const kp = generateKeypair();
    const msg = pingMessage('abc123', 1700000000000);
    expect(msg).toBe('abc123:1700000000000');
    const sig = signMessage(msg, kp.secretKey);
    const ok = nacl.sign.detached.verify(
      naclUtil.decodeUTF8(msg), naclUtil.decodeBase64(sig), naclUtil.decodeBase64(kp.publicKey),
    );
    expect(ok).toBe(true);
  });
});

// What a board report carries to identify its rig. It has to verify exactly the
// way the server checks it, and it must never be the reason a report goes
// missing: no identity, or one that cannot sign, is an unsigned report.
describe('signRig', () => {
  test('signs "<rigId>:<timestamp>" so the server (nacl.verify) accepts it', () => {
    const kp = generateKeypair();
    const identity = { nodeId: fingerprint(kp.publicKey), publicKey: kp.publicKey, secretKey: kp.secretKey };
    const signed = signRig(identity, 1700000000000);
    expect(signed).toEqual({
      rigId: identity.nodeId, publicKey: kp.publicKey, timestamp: 1700000000000, signature: expect.any(String),
    });
    const ok = nacl.sign.detached.verify(
      naclUtil.decodeUTF8(identity.nodeId + ':1700000000000'),
      naclUtil.decodeBase64(signed.signature),
      naclUtil.decodeBase64(kp.publicKey),
    );
    expect(ok).toBe(true);
    // The secret key never rides along.
    expect(JSON.stringify(signed)).not.toContain(kp.secretKey);
  });

  test('is null with no identity, or one missing a field', () => {
    const kp = generateKeypair();
    expect(signRig(null, 1)).toBeNull();
    expect(signRig({ publicKey: kp.publicKey, secretKey: kp.secretKey }, 1)).toBeNull();
    expect(signRig({ nodeId: 'abc', secretKey: kp.secretKey }, 1)).toBeNull();
    expect(signRig({ nodeId: 'abc', publicKey: kp.publicKey }, 1)).toBeNull();
  });

  test('is null, not a throw, for a node.json whose key cannot sign', () => {
    expect(signRig({ nodeId: 'abc', publicKey: 'pk', secretKey: 'not-a-key' }, 1)).toBeNull();
  });
});
