const crypto = require('crypto');
const nacl = require('tweetnacl');
const naclUtil = require('tweetnacl-util');

// Rig identity on the miner check-in. Each Earn install holds an Ed25519
// keypair (earn/src/shared/node.js) and signs "<rigId>:<timestamp>" with it on
// every report, where rigId is the leading hex of sha256(publicKey). Checking
// that here lets the server tell rigs apart — and know that a report claiming
// to be rig X came from the machine holding X's key — with no accounts.
//
// This is deliberately NOT a gate. Clients older than the identity send no
// signature, and a rig whose clock has drifted past the window fails the check;
// both still report and still appear on the board. They just carry no rig id.
//
// What the signature proves is "this key, at about this time". It does not
// cover the rest of the report, so a captured report could be replayed with
// other numbers inside the window. That is fine for telling rigs apart; anything
// that later ACTS on a rig id (sending one rig a build) needs the payload signed.

const MAX_SKEW_MS = 5 * 60 * 1000;
// 16 hex today; 6 hex for rigs that minted their key before the id was widened.
const RIG_ID_RE = /^(?:[0-9a-f]{6}|[0-9a-f]{16})$/;
const MAX_KEY_LEN = 128; // base64 of a 32-byte key is 44; a signature is 88

// The verified rig id, or null when the report is unsigned or does not check out.
function verifyRig(input, now) {
  const { rigId, publicKey, signature, timestamp } = input || {};
  if (typeof rigId !== 'string' || !RIG_ID_RE.test(rigId)) return null;
  if (typeof publicKey !== 'string' || publicKey.length > MAX_KEY_LEN) return null;
  if (typeof signature !== 'string' || signature.length > MAX_KEY_LEN) return null;
  if (!Number.isSafeInteger(timestamp) || Math.abs(now - timestamp) > MAX_SKEW_MS) return null;

  // The id has to be the key's own fingerprint, or any key could claim any id.
  const expected = crypto.createHash('sha256').update(publicKey).digest('hex').slice(0, rigId.length);
  if (expected !== rigId) return null;

  try {
    const ok = nacl.sign.detached.verify(
      naclUtil.decodeUTF8(rigId + ':' + timestamp),
      naclUtil.decodeBase64(signature),
      naclUtil.decodeBase64(publicKey)
    );
    return ok ? rigId : null;
  } catch (e) {
    // Malformed base64, or a key/signature of the wrong length.
    return null;
  }
}

module.exports = { verifyRig, MAX_SKEW_MS };
