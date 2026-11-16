const NodeService = require('../services/nodeService');
const NodeTokenService = require('../services/nodeTokenService');
const { clerkClient } = require('@clerk/clerk-sdk-node');
const { handle, reply, replyError } = require('./http');

// Best-effort display handle (Clerk username) for a user id. Null if it can't be
// resolved — the node still joins; the client falls back to the worker name.
async function resolveUsername(userId) {
  try {
    const user = await clerkClient.users.getUser(userId);
    return user.username || null;
  } catch (e) {
    return null;
  }
}

// POST /api/nodes/register - Self-register an UNCLAIMED node (no account, no
// join token). Signature-verified, so the caller proves it holds the key; the
// nodeId is derived from that public key server-side rather than trusted from
// the body. An unclaimed node is only ever assigned non-private jobs.
const registerNode = handle(async (req, res) => {
  // `nodeId` is the id the caller signs as, passed through so a machine whose
  // node.json predates the node-id widening keeps its existing 6-character
  // identity. It is NOT trusted as an identity: nodeService honours it only
  // when it matches this key's own legacy fingerprint. See _enrolledNodeId.
  const { publicKey, nodeId } = req.verifiedNode;
  const { name } = req.body;

  const nodeService = new NodeService(req.app.locals.db);
  const result = await nodeService.registerNode(publicKey, name, nodeId);

  reply(res, result);
}, 'Failed to register node');

const pingNode = handle(async (req, res) => {
  const { publicKey, nodeId } = req.verifiedNode;
  const { capabilities, activeJobs, maxConcurrentJobs,
    device, vramTotal, vramUsed, model, quant, tps, name } = req.body;

  const nodeService = new NodeService(req.app.locals.db);
  const result = await nodeService.updateNodeStatus(nodeId, publicKey, {
    capabilities,
    activeJobs,
    maxConcurrentJobs,
    device,
    vramTotal,
    vramUsed,
    model,
    quant,
    tps,
    name
  });

  reply(res, result);
}, 'Failed to update node status');

const getUserNodes = handle(async (req, res) => {
  const userId = req.user.id;

  const nodeService = new NodeService(req.app.locals.db);
  const nodes = await nodeService.getUserNodes(userId);
  res.json({ nodes });
}, 'Failed to get nodes');

const getPublicNodes = handle(async (req, res) => {
  const nodeService = new NodeService(req.app.locals.db);
  const result = await nodeService.getPublicNodes();
  res.json(result);
}, 'Failed to get public nodes');

// The serving fleet and how fast each node generates. Public and unauthenticated
// because it's what the network page paints: until now nothing enumerated the
// nodes actually serving jobs — getPublicNodes only lists ones a user flagged
// public, and the miner board only shows machines that are mining.
const getServingNodes = handle(async (req, res) => {
  const nodeService = new NodeService(req.app.locals.db);
  const nodes = await nodeService.listServingNodes();
  res.json({ nodes, total: nodes.length });
}, 'Failed to get serving nodes');

// GET /api/nodes/join-token - Return the user's join token (creating on first use)
const getJoinToken = handle(async (req, res) => {
  const userId = req.user.id;
  const tokenService = new NodeTokenService(req.app.locals.db);
  const record = await tokenService.getOrCreateToken(userId);
  res.json(record);
}, 'Failed to get join token');

// POST /api/nodes/join-token/rotate - Issue a fresh join token, invalidating the old one
const rotateJoinToken = handle(async (req, res) => {
  const userId = req.user.id;
  const tokenService = new NodeTokenService(req.app.locals.db);
  const record = await tokenService.rotateToken(userId);
  res.json(record);
}, 'Failed to rotate join token');

// POST /api/nodes/join - Attach a node using a join token (no Clerk session).
// Used by the install script so a machine can self-register non-interactively.
const joinNode = handle(async (req, res) => {
  // `nodeId` is the id the installer's node.json already holds, passed through
  // for the same reason as on the signed routes: a machine that minted a
  // 6-character id before the widening keeps it. nodeService constrains it to a
  // derivation of the publicKey in this same body, so it grants no reach the
  // join token did not already give.
  const { token, publicKey, name, nodeId } = req.body;

  if (!token || !publicKey) {
    return res.status(400).json({ error: 'token and publicKey are required' });
  }

  const tokenService = new NodeTokenService(req.app.locals.db);
  const userId = await tokenService.verifyToken(token);
  if (!userId) {
    return res.status(401).json({ error: 'Invalid join token' });
  }

  const nodeService = new NodeService(req.app.locals.db);
  const result = await nodeService.claimNode(publicKey, name || 'node', userId, nodeId);

  if (result.error) return replyError(res, result);

  const user = await resolveUsername(userId);
  res.status(201).json({ success: true, nodeId: result.nodeId, user });
}, 'Failed to join node');

module.exports = {
  registerNode,
  pingNode,
  getUserNodes,
  getPublicNodes,
  getServingNodes,
  getJoinToken,
  rotateJoinToken,
  joinNode
};
