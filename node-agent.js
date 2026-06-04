#!/usr/bin/env node
'use strict';

// LLMJob node agent — a single, dependency-free script.
//
// It is served by the app and meant to be run straight from a pipe:
//   curl -fsSL <server>/node-agent.js | node - join --server <server> --token <token> [--name <name>]
//
// It generates an Ed25519 keypair locally (only the public key ever leaves the
// machine), joins this machine to your account with the join token, then pings
// so the node shows as online. Requires Node 18+ (built-in crypto + fetch).

const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');

const CONFIG_DIR = process.env.LLMJOB_CONFIG_DIR || path.join(os.homedir(), '.llmjob');
const CONFIG_FILE = path.join(CONFIG_DIR, 'config.json');
const DEFAULT_SERVER = 'https://llmjob-production.up.railway.app';
const PING_INTERVAL_MS = 5 * 60 * 1000;

function parseArgs(argv) {
  const opts = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const eq = a.indexOf('=');
      if (eq !== -1) {
        opts[a.slice(2, eq)] = a.slice(eq + 1);
      } else {
        opts[a.slice(2)] = argv[++i];
      }
    } else {
      opts._.push(a);
    }
  }
  return opts;
}

// The raw 32-byte Ed25519 public key is the tail of the SPKI DER encoding.
function rawPublicKey(publicKey) {
  const der = publicKey.export({ type: 'spki', format: 'der' });
  return der.subarray(der.length - 32);
}

function fingerprint(publicKeyB64) {
  return crypto.createHash('sha256').update(publicKeyB64).digest('hex').slice(0, 6);
}

function loadOrCreateConfig(serverUrl) {
  if (!fs.existsSync(CONFIG_DIR)) {
    fs.mkdirSync(CONFIG_DIR, { recursive: true, mode: 0o700 });
  }

  if (fs.existsSync(CONFIG_FILE)) {
    const config = JSON.parse(fs.readFileSync(CONFIG_FILE, 'utf8'));
    if (serverUrl && config.serverUrl !== serverUrl) {
      config.serverUrl = serverUrl;
      saveConfig(config);
    }
    return config;
  }

  const { publicKey, privateKey } = crypto.generateKeyPairSync('ed25519');
  const publicKeyB64 = Buffer.from(rawPublicKey(publicKey)).toString('base64');
  const config = {
    nodeId: fingerprint(publicKeyB64),
    publicKey: publicKeyB64,
    privateKeyPem: privateKey.export({ type: 'pkcs8', format: 'pem' }),
    serverUrl: serverUrl || DEFAULT_SERVER,
    createdAt: new Date().toISOString()
  };
  saveConfig(config);
  return config;
}

function saveConfig(config) {
  fs.writeFileSync(CONFIG_FILE, JSON.stringify(config, null, 2), { mode: 0o600 });
}

function sign(message, privateKeyPem) {
  const key = crypto.createPrivateKey(privateKeyPem);
  return Buffer.from(crypto.sign(null, Buffer.from(message, 'utf8'), key)).toString('base64');
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  let data = {};
  try { data = await res.json(); } catch (_) { /* ignore non-JSON */ }
  return { ok: res.ok, status: res.status, data };
}

async function join(config, token, name) {
  return postJson(`${config.serverUrl}/api/nodes/join`, {
    token,
    nodeId: config.nodeId,
    publicKey: config.publicKey,
    name: name || `node-${config.nodeId}`
  });
}

async function ping(config) {
  const timestamp = Date.now();
  const message = `${config.nodeId}:${timestamp}`;
  return postJson(`${config.serverUrl}/api/nodes/ping`, {
    nodeId: config.nodeId,
    publicKey: config.publicKey,
    signature: sign(message, config.privateKeyPem),
    timestamp
  });
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const command = opts._[0] || 'join';

  if (command !== 'join') {
    console.error(`Unknown command "${command}". Usage: node-agent.js join --server <url> --token <token>`);
    process.exit(1);
  }
  if (!opts.token) {
    console.error('Error: --token is required (copy the full command from your dashboard).');
    process.exit(1);
  }
  if (typeof fetch !== 'function') {
    console.error('Error: Node 18+ is required (global fetch was not found).');
    process.exit(1);
  }

  const config = loadOrCreateConfig(opts.server);

  console.log(`LLMJob node ${config.nodeId} → ${config.serverUrl}`);
  const result = await join(config, opts.token, opts.name);
  if (!result.ok) {
    console.error(`✗ Failed to join: ${(result.data && result.data.error) || `HTTP ${result.status}`}`);
    process.exit(1);
  }
  console.log('✓ Joined and claimed to your account');

  const beat = async () => {
    const r = await ping(config);
    const t = new Date().toLocaleTimeString();
    if (r.ok) console.log(`[${t}] ✓ ping`);
    else console.error(`[${t}] ✗ ping failed: ${(r.data && r.data.error) || `HTTP ${r.status}`}`);
  };

  await beat();
  const timer = setInterval(beat, PING_INTERVAL_MS);
  const shutdown = () => { clearInterval(timer); console.log('\nStopped.'); process.exit(0); };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

main().catch((err) => {
  console.error('Fatal:', err.message);
  process.exit(1);
});
