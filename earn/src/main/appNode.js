'use strict';

// The desktop app's link to LLMJob, and the cluster job worker. Moved out of
// main.js unchanged. The parts of main.js it reads and calls come through
// appContext.

const { ctx, share, late } = require('./appContext');
const nodeStore = require('./nodeStore');
const { createNodeClient } = require('./nodeClient');
const { NODE } = require('../shared/config');

const send = late('send');

// ── Connect with LLMJob (node identity + pairing/ping) ───────────────────────
// The machine's Ed25519 keypair lives in the shared nodeStore (one identity for
// the GUI AND the CLI); only the public key leaves it. "Connect" self-registers
// with a pairing token (/api/nodes/join); once linked it pings (/api/nodes/ping)
// with signed telemetry so the node shows online in the user's cluster. The
// requests themselves are nodeClient's, shared with the CLI. All best-effort —
// failures never touch mining.

let nodePinger = null;
const { loadNode, saveNode, getOrCreateNode } = nodeStore;
const nodeClient = createNodeClient({
  log: (level, line) => send('miner:log', { level, line }),
  linkHint: 'connect it to your account for private jobs',
});

// Renderer-safe view (no secret key).
function nodeStatus() {
  const node = loadNode();
  return {
    connected: !!(node && node.connected),
    nodeId: node ? node.nodeId : null,
    name: node ? node.name : null,
    user: node ? (node.user || null) : null,
  };
}

function sendNodeStatus() { send('node:status', nodeStatus()); }

// One signed ping with fresh telemetry. Silent on failure. Carries the node's
// current name so a Settings rename propagates to the server (see syncNodeName).
async function pingNode() {
  const node = loadNode();
  const { fleet, llmStatus } = ctx;
  // Linked machines always ping. An unlinked one pings only while it is actually
  // serving (the fleet is up), so a machine that merely once had an identity
  // doesn't advertise itself as an online node while doing nothing.
  if (!node || !(node.connected || fleet)) return;
  const telemetry = await nodeClient.telemetry({
    model: ctx.servingModel, ready: llmStatus.ready, tokensPerSec: llmStatus.tokensPerSec,
    activeJobs: fleet ? fleet.activeJobs() : 0, name: node.name,
  });
  await nodeClient.ping(node, telemetry);
}

// Keep the linked node's name in step with the Settings worker name: the
// connected card says "rename in Settings", so a changed worker name updates
// node.json and is pushed on the next ping (the server picks up non-null names).
function syncNodeName(settings) {
  const node = loadNode();
  const worker = settings && settings.worker && String(settings.worker).trim();
  if (node && node.connected && worker && worker !== node.name) {
    saveNode(Object.assign({}, node, { name: worker }));
    sendNodeStatus();
    pingNode();
  }
}

function startNodePinger() {
  stopNodePinger();
  pingNode();
  nodePinger = setInterval(pingNode, NODE.pingIntervalMs);
  nodePinger.unref();
}

function stopNodePinger() {
  if (nodePinger) { clearInterval(nodePinger); nodePinger = null; }
}

// Link this machine to an account with a pairing/join token. Returns a
// renderer-safe result; on success the node is saved connected and starts pinging.
async function connectNode({ token, name }) {
  const t = String(token || '').trim();
  if (!t) return { error: 'Enter your pairing token first.' };
  const node = getOrCreateNode();
  const nm = (name && String(name).trim()) || node.name || null;

  const res = await nodeClient.join(node, t, nm);
  if (res.unreachable) return { error: 'Could not reach LLMJob — check your connection.' };
  if (!res.ok) return { error: res.message || ('Link failed (HTTP ' + res.status + ').') };

  saveNode(Object.assign({}, node, {
    nodeId: res.nodeId, connected: true, name: nm, user: res.user, linkedAt: new Date().toISOString(),
  }));
  startNodePinger();
  syncWorker();
  sendNodeStatus();
  return { success: true, nodeId: node.nodeId, name: nm, user: res.user };
}

function disconnectNode() {
  const node = loadNode();
  if (node) saveNode(Object.assign({}, node, { connected: false }));
  stopNodePinger();
  stopWorker();
  sendNodeStatus();
  return { ok: true };
}

// ── Cluster job worker: serve inference relayed through LLMJob ────────────────
// When this node is linked AND the local LLM is up, poll for jobs and run them
// against the local model, streaming chunks back — all outbound, so callers can
// use this GPU through the shared API without any inbound networking here.

// Build a cluster job-worker for the ready LLM instance at `baseUrl` — one per
// serving GPU, each running jobs against its own llama-server. A rig that can run
// the model is useful to the network whether or not anyone linked it, so an
// unlinked box serves public work too (the server hands an unclaimed node
// non-private jobs only, so "unlinked" costs it access to private queues and
// nothing else) — matching the headless CLI's default.
// `track` is the fleet's, for the card this worker serves (LlmFleet.trackRequest).
function makeJobWorker(baseUrl, track) {
  // Reached only via the fleet after syncWorker() armed it, which already bailed
  // if no identity could be minted — so the node is always present here.
  const w = nodeClient.makeJobWorker(getOrCreateNode(), { baseUrl, servingModel: () => ctx.servingModel, track });
  if (!ctx.serveLogged) {
    ctx.serveLogged = true;
    send('miner:log', { level: 'info', line: 'serving cluster jobs for the LLMJob network' });
  }
  return w;
}

// Start cluster serving across every ready instance once a model is up — account
// or not. Idempotent, called from connect/disconnect and LLM ready. An unlinked
// machine mints an identity and self-registers so the server will answer its
// polls (nodeClient.enrol, retried with every ping until it lands); a linked one
// is already registered and just keeps serving. Pinging starts either way, so
// the node shows online and can be targeted for testing.
function syncWorker() {
  const fleet = ctx.fleet;
  if (!fleet) return;
  const node = getOrCreateNode();
  if (!node) return; // no identity and none could be minted — serve nothing
  nodeClient.enrol(node);
  fleet.syncWorkers(true);
  startNodePinger();
}

function stopWorker() {
  if (ctx.fleet) ctx.fleet.syncWorkers(false);
  ctx.serveLogged = false;
}

share({}, { syncWorker, makeJobWorker, syncNodeName });

module.exports = { nodeStatus, connectNode, disconnectNode, startNodePinger };
