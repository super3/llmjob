'use strict';

// Rig identity. Each machine holds an Ed25519 signing keypair (only the public
// key ever leaves it), and its id is a short fingerprint of that key. The miner
// signs its once-a-minute board report with it, so the server can tell one rig
// from another — and trust that a report claiming to be rig X came from the
// machine holding X's key — with no accounts involved. These are the pure,
// deterministic halves; the keypair itself is persisted by main/nodeStore.js,
// which both shells share, so one machine keeps one id whether it runs the GUI
// or the CLI.
//
// "node" in the names is historical: this keypair was first minted to link a
// machine to an account as an LLM node, and existing rigs already hold one in
// node.json. Reusing it means every rig that ever ran LLMJob keeps its id.

const nacl = require('tweetnacl');
const naclUtil = require('tweetnacl-util');
const crypto = require('crypto');

// A fresh Ed25519 keypair, base64-encoded (matches the server's tweetnacl verify).
function generateKeypair() {
  const kp = nacl.sign.keyPair();
  return {
    publicKey: naclUtil.encodeBase64(kp.publicKey),
    secretKey: naclUtil.encodeBase64(kp.secretKey),
  };
}

// Short, stable id = first NODE_ID_HEX hex of sha256(publicKey). The server
// recomputes it from the public key to check a report's claim, so the two must
// agree exactly.
//
// 16 characters (64 bits), not the 6 (24 bits) this used to be: 24 bits put two
// honest rigs on the same id with ~3% probability at 1,000 rigs and 52% at
// 5,000. This is only ever called to MINT an id; a machine that minted one at
// the old width keeps the 6-character id already stored in its node.json, and
// the server accepts either width.
const NODE_ID_HEX = 16;

function fingerprint(publicKey) {
  return crypto.createHash('sha256')
    .update(String(publicKey == null ? '' : publicKey))
    .digest('hex')
    .slice(0, NODE_ID_HEX);
}

// The challenge a rig signs to prove it holds the secret key: its id and the
// time, so a signature cannot be replayed beyond the server's window.
function pingMessage(nodeId, timestamp) {
  return String(nodeId) + ':' + String(timestamp);
}

// Detached base64 signature of `message` under the base64 secret key.
function signMessage(message, secretKeyB64) {
  const sig = nacl.sign.detached(naclUtil.decodeUTF8(String(message)), naclUtil.decodeBase64(secretKeyB64));
  return naclUtil.encodeBase64(sig);
}

// The identity fields a board report carries: { rigId, publicKey, timestamp,
// signature }. Null when there is no usable identity — none stored, or a
// node.json whose key is corrupt or truncated and cannot sign — because a report
// without an identity is still a report: the rig just shows up unsigned, the way
// every older client does, rather than going missing from the board.
function signRig(identity, timestamp) {
  if (!identity || !identity.nodeId || !identity.publicKey || !identity.secretKey) return null;
  try {
    return {
      rigId: identity.nodeId,
      publicKey: identity.publicKey,
      timestamp,
      signature: signMessage(pingMessage(identity.nodeId, timestamp), identity.secretKey),
    };
  } catch (e) {
    return null;
  }
}

module.exports = {
  generateKeypair, fingerprint, pingMessage, signMessage, signRig, NODE_ID_HEX,
};
