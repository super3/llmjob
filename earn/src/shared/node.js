'use strict';

// Pure node-identity + protocol helpers for "Connect with LLMJob". The machine
// holds an Ed25519 signing keypair (only the public key ever leaves it); the
// nodeId is a short fingerprint of the public key. These build the exact request
// bodies the server expects (/api/nodes/join, /api/nodes/ping) and sign the ping
// challenge — all deterministic and unit-testable. The IO (persist the key, POST
// to the server, sample telemetry) lives in main.js.

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

// Short, stable node id = first 6 hex of sha256(publicKey) (same as the server).
function fingerprint(publicKey) {
  return crypto.createHash('sha256').update(String(publicKey == null ? '' : publicKey)).digest('hex').slice(0, 6);
}

// The ping challenge the node signs to prove it holds the secret key.
function pingMessage(nodeId, timestamp) {
  return String(nodeId) + ':' + String(timestamp);
}

// Detached base64 signature of `message` under the base64 secret key.
function signMessage(message, secretKeyB64) {
  const sig = nacl.sign.detached(naclUtil.decodeUTF8(String(message)), naclUtil.decodeBase64(secretKeyB64));
  return naclUtil.encodeBase64(sig);
}

// Body for POST /api/nodes/join — attach this machine to an account with a
// pairing/join token. Falls back to a Node-<id> name when none is given.
function buildJoinBody({ token, nodeId, publicKey, name } = {}) {
  return {
    token: token || '',
    nodeId,
    publicKey,
    name: (name && String(name).trim()) || ('Node-' + nodeId),
  };
}

// Map the app's live state into the server's ping telemetry shape. Anything the
// app can't read right now is sent as null / 0 rather than omitted.
function buildTelemetry({ model, quant, device, vram, tokensPerSec, ready } = {}) {
  return {
    capabilities: ready ? ['chat'] : [],
    activeJobs: 0,
    maxConcurrentJobs: 1,
    device: device || null,
    vramTotal: vram && Number.isFinite(vram.totalMb) ? vram.totalMb : null,
    vramUsed: vram && Number.isFinite(vram.usedMb) ? vram.usedMb : null,
    model: model || null,
    quant: quant || null,
    tps: Number(tokensPerSec) || 0,
  };
}

// Body for POST /api/nodes/ping — signed challenge + telemetry.
function buildPingBody({ nodeId, publicKey, secretKey, timestamp, telemetry } = {}) {
  return Object.assign({
    nodeId,
    publicKey,
    signature: signMessage(pingMessage(nodeId, timestamp), secretKey),
    timestamp,
  }, telemetry || {});
}

module.exports = {
  generateKeypair, fingerprint, pingMessage, signMessage,
  buildJoinBody, buildTelemetry, buildPingBody,
};
