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

// Short, stable node id = first NODE_ID_HEX hex of sha256(publicKey). Must match
// the server's nodeService.generateNodeFingerprint exactly, or a client cannot
// address itself.
//
// 16 characters (64 bits), not the 6 (24 bits) this used to be: 24 bits put two
// honest nodes on the same id with ~3% probability at 1,000 nodes and 52% at
// 5,000, and the loser was silently unusable — its signed pings refused for a
// key mismatch, its polls 401ing, and no way to claim it to an account.
//
// This is only ever called to MINT an id. A machine that enrolled under the old
// width keeps the 6-character id already stored in its node.json (see
// main/nodeStore.js) and the server still recognises it, so nothing has to be
// re-paired.
const NODE_ID_HEX = 16;

function fingerprint(publicKey) {
  return crypto.createHash('sha256')
    .update(String(publicKey == null ? '' : publicKey))
    .digest('hex')
    .slice(0, NODE_ID_HEX);
}

// The node id to persist after the server answers an enrolment call
// (/api/nodes/register or /api/nodes/join).
//
// The server is authoritative about which id a machine is enrolled under. It
// chooses between this key's current-width fingerprint and the narrower one a
// machine minted before ids were widened, and it can legitimately hand back an
// id we did not compute: a machine whose old narrow id turns out to be occupied
// by somebody ELSE's key is enrolled on the wide id instead.
//
// Adopting the answer is what stops such a machine going on to sign every later
// call as an id the server has no row for — which reads as "my rig serves
// nothing and there is no way to tell why". It also retires the standing
// requirement that two independent implementations agree on the id forever.
//
// Anything that is not a plausible id is ignored, so a garbled or truncated
// response can never rewrite this machine's identity.
function adoptedNodeId(localId, serverId) {
  if (typeof serverId !== 'string') return localId;
  const id = serverId.trim();
  return /^[0-9a-f]{6,64}$/.test(id) ? id : localId;
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
// app can't read right now is sent as null / 0 rather than omitted. `name` rides
// along so renaming the worker propagates on the next ping (the server ignores
// a null name rather than clobbering the stored one).
function buildTelemetry({ model, quant, device, vram, tokensPerSec, ready, activeJobs, name } = {}) {
  return {
    capabilities: ready ? ['chat'] : [],
    activeJobs: Number(activeJobs) || 0,
    maxConcurrentJobs: 1,
    device: device || null,
    vramTotal: vram && Number.isFinite(vram.totalMb) ? vram.totalMb : null,
    vramUsed: vram && Number.isFinite(vram.usedMb) ? vram.usedMb : null,
    model: model || null,
    quant: quant || null,
    tps: Number(tokensPerSec) || 0,
    name: name || null,
  };
}

// A signed request body for any node→server call the `verifySignature` middleware
// guards (ping, job poll/chunks/complete/…): the identity + a detached signature
// over "<nodeId>:<timestamp>", merged with the call-specific `extra` fields.
function signedBody({ nodeId, publicKey, secretKey, timestamp } = {}, extra) {
  return Object.assign({
    nodeId,
    publicKey,
    signature: signMessage(pingMessage(nodeId, timestamp), secretKey),
    timestamp,
  }, extra || {});
}

// Body for POST /api/nodes/ping — a signed challenge carrying telemetry.
function buildPingBody({ nodeId, publicKey, secretKey, timestamp, telemetry } = {}) {
  return signedBody({ nodeId, publicKey, secretKey, timestamp }, telemetry);
}

module.exports = {
  generateKeypair, fingerprint, adoptedNodeId, pingMessage, signMessage,
  buildJoinBody, buildTelemetry, signedBody, buildPingBody, NODE_ID_HEX,
};
