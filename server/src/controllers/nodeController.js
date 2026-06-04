const NodeService = require('../services/nodeService');
const NodeTokenService = require('../services/nodeTokenService');

async function claimNode(req, res) {
  try {
    const { publicKey, name } = req.body;
    const userId = req.user.id;

    if (!publicKey || !name) {
      return res.status(400).json({ error: 'Public key and name are required' });
    }

    const nodeService = new NodeService(req.app.locals.redis);
    const result = await nodeService.claimNode(publicKey, name, userId);

    if (result.error) {
      return res.status(400).json({ error: result.error });
    }

    res.json(result);
  } catch (error) {
    console.error('Claim node error:', error);
    res.status(500).json({ error: 'Failed to claim node' });
  }
}

async function pingNode(req, res) {
  try {
    const { publicKey, nodeId } = req.verifiedNode;
    const { capabilities, activeJobs, maxConcurrentJobs,
      device, vramTotal, vramUsed, model, quant, tps } = req.body;

    const nodeService = new NodeService(req.app.locals.redis);
    const result = await nodeService.updateNodeStatus(nodeId, publicKey, {
      capabilities,
      activeJobs,
      maxConcurrentJobs,
      device,
      vramTotal,
      vramUsed,
      model,
      quant,
      tps
    });

    if (result.error) {
      return res.status(400).json({ error: result.error });
    }

    res.json(result);
  } catch (error) {
    console.error('Ping node error:', error);
    res.status(500).json({ error: 'Failed to update node status' });
  }
}

async function getUserNodes(req, res) {
  try {
    const userId = req.user.id;

    const nodeService = new NodeService(req.app.locals.redis);
    const nodes = await nodeService.getUserNodes(userId);
    res.json({ nodes });
  } catch (error) {
    console.error('Get user nodes error:', error);
    res.status(500).json({ error: 'Failed to get nodes' });
  }
}

async function getPublicNodes(req, res) {
  try {
    const nodeService = new NodeService(req.app.locals.redis);
    const result = await nodeService.getPublicNodes();
    res.json(result);
  } catch (error) {
    console.error('Get public nodes error:', error);
    res.status(500).json({ error: 'Failed to get public nodes' });
  }
}

async function updateNodeVisibility(req, res) {
  try {
    const { id: nodeId } = req.params;
    const { isPublic } = req.body;
    const userId = req.user.id;

    if (typeof isPublic !== 'boolean') {
      return res.status(400).json({ error: 'isPublic must be a boolean' });
    }

    const nodeService = new NodeService(req.app.locals.redis);
    const result = await nodeService.updateNodeVisibility(nodeId, userId, isPublic);

    if (result.error) {
      return res.status(result.status || 400).json({ error: result.error });
    }

    res.json(result);
  } catch (error) {
    console.error('Update visibility error:', error);
    res.status(500).json({ error: 'Failed to update node visibility' });
  }
}

// GET /api/nodes/join-token - Return the user's join token (creating on first use)
async function getJoinToken(req, res) {
  try {
    const userId = req.user.id;
    const tokenService = new NodeTokenService(req.app.locals.redis);
    const record = await tokenService.getOrCreateToken(userId);
    res.json(record);
  } catch (error) {
    console.error('Get join token error:', error);
    res.status(500).json({ error: 'Failed to get join token' });
  }
}

// POST /api/nodes/join-token/rotate - Issue a fresh join token, invalidating the old one
async function rotateJoinToken(req, res) {
  try {
    const userId = req.user.id;
    const tokenService = new NodeTokenService(req.app.locals.redis);
    const record = await tokenService.rotateToken(userId);
    res.json(record);
  } catch (error) {
    console.error('Rotate join token error:', error);
    res.status(500).json({ error: 'Failed to rotate join token' });
  }
}

// POST /api/nodes/join - Attach a node using a join token (no Clerk session).
// Used by the install script so a machine can self-register non-interactively.
async function joinNode(req, res) {
  try {
    const { token, publicKey, name } = req.body;

    if (!token || !publicKey) {
      return res.status(400).json({ error: 'token and publicKey are required' });
    }

    const tokenService = new NodeTokenService(req.app.locals.redis);
    const userId = await tokenService.verifyToken(token);
    if (!userId) {
      return res.status(401).json({ error: 'Invalid join token' });
    }

    const nodeService = new NodeService(req.app.locals.redis);
    const result = await nodeService.claimNode(publicKey, name || 'node', userId);

    if (result.error) {
      return res.status(400).json({ error: result.error });
    }

    res.status(201).json({ success: true, nodeId: result.nodeId });
  } catch (error) {
    console.error('Join node error:', error);
    res.status(500).json({ error: 'Failed to join node' });
  }
}

module.exports = {
  claimNode,
  pingNode,
  getUserNodes,
  getPublicNodes,
  updateNodeVisibility,
  getJoinToken,
  rotateJoinToken,
  joinNode
};
