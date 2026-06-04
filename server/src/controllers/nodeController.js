const NodeService = require('../services/nodeService');

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

module.exports = {
  claimNode,
  pingNode,
  getUserNodes,
  getPublicNodes,
  updateNodeVisibility
};
