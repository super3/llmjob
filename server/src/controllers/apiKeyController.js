const ApiKeyService = require('../services/apiKeyService');
const { handle, reply } = require('./http');

const createKey = handle(async (req, res) => {
  const { name, visibility } = req.body;
  const userId = req.user.id;

  if (!name || !name.trim()) {
    return res.status(400).json({ error: 'Key name is required' });
  }
  if (visibility !== undefined && visibility !== 'public' && visibility !== 'private') {
    return res.status(400).json({ error: "visibility must be 'public' or 'private'" });
  }

  const apiKeyService = new ApiKeyService(req.app.locals.db);
  const result = await apiKeyService.createKey(userId, name.trim(), visibility);

  // `result.key` is the raw secret — returned exactly once.
  res.status(201).json(result);
}, 'Failed to create API key');

// PUT /api/keys/:id/visibility - flip a key between 'public' and 'private' routing.
const updateKeyVisibility = handle(async (req, res) => {
  const { id } = req.params;
  const { visibility } = req.body;
  const userId = req.user.id;

  if (visibility !== 'public' && visibility !== 'private') {
    return res.status(400).json({ error: "visibility must be 'public' or 'private'" });
  }

  const apiKeyService = new ApiKeyService(req.app.locals.db);
  reply(res, await apiKeyService.updateKeyVisibility(userId, id, visibility));
}, 'Failed to update API key');

const listKeys = handle(async (req, res) => {
  const apiKeyService = new ApiKeyService(req.app.locals.db);
  const keys = await apiKeyService.listKeys(req.user.id);
  res.json({ keys });
}, 'Failed to list API keys');

const revokeKey = handle(async (req, res) => {
  const apiKeyService = new ApiKeyService(req.app.locals.db);
  reply(res, await apiKeyService.revokeKey(req.user.id, req.params.id));
}, 'Failed to revoke API key');

module.exports = {
  createKey,
  listKeys,
  revokeKey,
  updateKeyVisibility
};
