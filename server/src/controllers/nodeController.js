const nodeService = require('../services/nodeService');

async function claimNode(req, res) {
  try {
    const { publicKey, name } = req.body;
    const userId = req.user.id;
    const redis = req.app.locals.redis;
    
    if (!publicKey || !name) {
      return res.status(400).json({ error: 'Public key and name are required' });
    }
    
    const result = await nodeService.claimNode(redis, publicKey, name, userId);
    
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
    const redis = req.app.locals.redis;
    
    const result = await nodeService.updateNodeStatus(redis, nodeId, publicKey);
    
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
    const redis = req.app.locals.redis;
    
    const nodes = await nodeService.getUserNodes(redis, userId);
    res.json({ nodes });
  } catch (error) {
    console.error('Get user nodes error:', error);
    res.status(500).json({ error: 'Failed to get nodes' });
  }
}

async function getPublicNodes(req, res) {
  try {
    const redis = req.app.locals.redis;
    
    const result = await nodeService.getPublicNodes(redis);
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
    const redis = req.app.locals.redis;
    
    if (typeof isPublic !== 'boolean') {
      return res.status(400).json({ error: 'isPublic must be a boolean' });
    }
    
    const result = await nodeService.updateNodeVisibility(redis, nodeId, userId, isPublic);
    
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