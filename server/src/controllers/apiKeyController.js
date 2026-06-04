const ApiKeyService = require('../services/apiKeyService');

async function createKey(req, res) {
  try {
    const { name } = req.body;
    const userId = req.user.id;

    if (!name || !name.trim()) {
      return res.status(400).json({ error: 'Key name is required' });
    }

    const apiKeyService = new ApiKeyService(req.app.locals.db);
    const result = await apiKeyService.createKey(userId, name.trim());

    // `result.key` is the raw secret — returned exactly once.
    res.status(201).json(result);
  } catch (error) {
    console.error('Create API key error:', error);
    res.status(500).json({ error: 'Failed to create API key' });
  }
}

async function listKeys(req, res) {
  try {
    const userId = req.user.id;

    const apiKeyService = new ApiKeyService(req.app.locals.db);
    const keys = await apiKeyService.listKeys(userId);
    res.json({ keys });
  } catch (error) {
    console.error('List API keys error:', error);
    res.status(500).json({ error: 'Failed to list API keys' });
  }
}

async function revokeKey(req, res) {
  try {
    const { id } = req.params;
    const userId = req.user.id;

    const apiKeyService = new ApiKeyService(req.app.locals.db);
    const result = await apiKeyService.revokeKey(userId, id);

    if (result.error) {
      return res.status(result.status || 400).json({ error: result.error });
    }

    res.json(result);
  } catch (error) {
    console.error('Revoke API key error:', error);
    res.status(500).json({ error: 'Failed to revoke API key' });
  }
}

module.exports = {
  createKey,
  listKeys,
  revokeKey
};
