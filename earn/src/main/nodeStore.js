'use strict';

// One node identity per machine, shared by BOTH shells. The GUI and CLI used to
// keep separate node.json files (Electron userData vs ~/.local/share), which
// gave one machine two nodeIds: linking in the GUI left the CLI unlinked (it
// silently refused to serve jobs) and pairing both registered one GPU as two
// cluster nodes. Everything now reads/writes ~/.local/share/llmjob-earn/node.json
// — a path both shells can compute without Electron. The secret key never leaves
// this file.

const fs = require('fs');
const path = require('path');
const os = require('os');
const nodeProto = require('../shared/node');

function storeDir() { return path.join(os.homedir(), '.local', 'share', 'llmjob-earn'); }
function nodePath() { return path.join(storeDir(), 'node.json'); }

function loadNode() {
  try { return JSON.parse(fs.readFileSync(nodePath(), 'utf8')); } catch (e) { return null; }
}

function saveNode(node) {
  fs.mkdirSync(storeDir(), { recursive: true });
  fs.writeFileSync(nodePath(), JSON.stringify(node, null, 2), { mode: 0o600 });
}

// One-time migration from a shell's old private location (the GUI's Electron
// userData dir). Only runs when the shared store is empty and the legacy file
// holds a full identity, so an existing pairing survives the path change.
function migrateFrom(legacyPath) {
  if (loadNode() || !legacyPath) return false;
  try {
    const old = JSON.parse(fs.readFileSync(legacyPath, 'utf8'));
    if (old && old.publicKey && old.secretKey) { saveNode(old); return true; }
  } catch (e) { /* nothing to migrate */ }
  return false;
}

// The persisted identity (keypair + id), created on first use.
function getOrCreateNode() {
  let node = loadNode();
  if (!node || !node.publicKey || !node.secretKey) {
    const kp = nodeProto.generateKeypair();
    node = {
      nodeId: nodeProto.fingerprint(kp.publicKey),
      publicKey: kp.publicKey,
      secretKey: kp.secretKey,
      name: null,
      connected: false,
      createdAt: new Date().toISOString(),
    };
    saveNode(node);
  }
  return node;
}

module.exports = { nodePath, loadNode, saveNode, migrateFrom, getOrCreateNode };
