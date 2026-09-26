const crypto = require('crypto');
const nacl = require('tweetnacl');
const naclUtil = require('tweetnacl-util');
const { verifyRig, MAX_SKEW_MS } = require('../src/services/rigIdentity');
// The client half. Signing with the real client code is what keeps the two
// ends from drifting apart on the message format or the id width.
const client = require('../../earn/src/shared/node');

const NOW = 1790000000000;
const sha = (s) => crypto.createHash('sha256').update(s).digest('hex');

function rig(idHex) {
  const kp = nacl.sign.keyPair();
  const publicKey = naclUtil.encodeBase64(kp.publicKey);
  return {
    publicKey,
    secretKey: naclUtil.encodeBase64(kp.secretKey),
    nodeId: sha(publicKey).slice(0, idHex),
  };
}

const signed = (identity, timestamp = NOW) => client.signRig(identity, timestamp);

describe('verifyRig', () => {
  test('accepts a report signed by the real client', () => {
    const r = rig(client.NODE_ID_HEX);
    expect(verifyRig(signed(r), NOW)).toBe(r.nodeId);
  });

  test('accepts the 6-hex ids that older rigs minted', () => {
    const r = rig(6);
    expect(verifyRig(signed(r), NOW)).toBe(r.nodeId);
  });

  test('accepts a timestamp up to the skew window either side', () => {
    const r = rig(16);
    expect(verifyRig(signed(r, NOW - MAX_SKEW_MS), NOW)).toBe(r.nodeId);
    expect(verifyRig(signed(r, NOW + MAX_SKEW_MS), NOW)).toBe(r.nodeId);
  });

  test('an unsigned report (every older client) is simply not identified', () => {
    expect(verifyRig(undefined, NOW)).toBeNull();
    expect(verifyRig({}, NOW)).toBeNull();
    expect(verifyRig({ address: 'prl1p…', hashrate: 5 }, NOW)).toBeNull();
  });

  test('rejects a rig id of the wrong shape', () => {
    const good = signed(rig(16));
    for (const rigId of [123, 'ABCDEF0123456789', 'a1b2c3d4', 'zzzzzz', '']) {
      expect(verifyRig({ ...good, rigId }, NOW)).toBeNull();
    }
  });

  test('rejects a key or signature that is missing or oversized', () => {
    const good = signed(rig(16));
    expect(verifyRig({ ...good, publicKey: null }, NOW)).toBeNull();
    expect(verifyRig({ ...good, publicKey: 'A'.repeat(129) }, NOW)).toBeNull();
    expect(verifyRig({ ...good, signature: 42 }, NOW)).toBeNull();
    expect(verifyRig({ ...good, signature: 'A'.repeat(129) }, NOW)).toBeNull();
  });

  test('rejects a timestamp that is not an integer or is outside the window', () => {
    const r = rig(16);
    expect(verifyRig({ ...signed(r), timestamp: String(NOW) }, NOW)).toBeNull();
    expect(verifyRig({ ...signed(r), timestamp: NOW + 0.5 }, NOW)).toBeNull();
    expect(verifyRig(signed(r, NOW - MAX_SKEW_MS - 1), NOW)).toBeNull();
    expect(verifyRig(signed(r, NOW + MAX_SKEW_MS + 1), NOW)).toBeNull();
  });

  // Otherwise any keyholder could sign as any rig id it liked.
  test('rejects an id that is not the fingerprint of the key presented', () => {
    const mine = rig(16);
    const theirs = rig(16);
    const claim = { ...signed(mine), rigId: theirs.nodeId };
    // Re-sign so the signature itself is valid for the claimed id.
    claim.signature = client.signMessage(client.pingMessage(theirs.nodeId, NOW), mine.secretKey);
    expect(verifyRig(claim, NOW)).toBeNull();
  });

  test('rejects a signature made by a different key, or over a different time', () => {
    const r = rig(16);
    const other = rig(16);
    const forged = { ...signed(r), signature: client.signMessage(client.pingMessage(r.nodeId, NOW), other.secretKey) };
    expect(verifyRig(forged, NOW)).toBeNull();
    // A valid signature from one report does not carry over to another timestamp.
    expect(verifyRig({ ...signed(r, NOW - 1000), timestamp: NOW }, NOW)).toBeNull();
  });

  test('malformed base64, or a key of the wrong length, is not identified rather than thrown', () => {
    const r = rig(16);
    const badSig = { ...signed(r), signature: 'not base64!!' };
    expect(verifyRig(badSig, NOW)).toBeNull();

    // Valid base64, wrong length: nacl throws on a 16-byte key.
    const shortKey = naclUtil.encodeBase64(new Uint8Array(16));
    const shortClaim = { ...signed(r), publicKey: shortKey, rigId: sha(shortKey).slice(0, 16) };
    expect(verifyRig(shortClaim, NOW)).toBeNull();
  });
});
