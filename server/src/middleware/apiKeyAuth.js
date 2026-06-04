const ApiKeyService = require('../services/apiKeyService');

// Authenticate a request using a raw LLMJob API key passed as a bearer token.
// On success attaches `req.apiKey` ({ userId, id, name, hash }) and a minimal
// `req.user` so downstream handlers can treat it like an authenticated user.
async function apiKeyAuth(req, res, next) {
  try {
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return res.status(401).json({ error: 'No API key provided' });
    }

    const rawKey = authHeader.split(' ')[1];
    if (!rawKey || !rawKey.startsWith('lj-')) {
      return res.status(401).json({ error: 'Invalid API key' });
    }

    const apiKeyService = new ApiKeyService(req.app.locals.redis);
    const resolved = await apiKeyService.verifyKey(rawKey);

    if (!resolved) {
      return res.status(401).json({ error: 'Invalid API key' });
    }

    req.apiKey = resolved;
    req.user = { id: resolved.userId };
    next();
  } catch (error) {
    console.error('API key auth error:', error);
    return res.status(500).json({ error: 'Authentication error' });
  }
}

module.exports = { apiKeyAuth };
