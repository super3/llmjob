const LogService = require('../services/logService');
const ApiKeyService = require('../services/apiKeyService');

// GET /api/logs — recent request logs plus the activity histogram for the
// dashboard chart. Authenticated via Clerk (dashboard user).
async function getLogs(req, res) {
  try {
    const userId = req.user.id;
    const limit = req.query.limit ? parseInt(req.query.limit, 10) : 50;

    const logService = new LogService(req.app.locals.db);
    const logs = await logService.getLogs(userId, limit);
    const activity = await logService.getActivity(userId);

    res.json({ logs, activity });
  } catch (error) {
    console.error('Get logs error:', error);
    res.status(500).json({ error: 'Failed to get logs' });
  }
}

// POST /api/usage — record one completed generation. Authenticated via an API
// key (apiKeyAuth), so it both writes a log entry and bills the key's usage.
async function recordUsage(req, res) {
  try {
    const { userId, name: keyName, hash } = req.apiKey;
    const { model, node, finish } = req.body;
    const inTokens = req.body.in || 0;
    const outTokens = req.body.out || 0;
    const speed = req.body.speed || 0;

    if (!model || !node) {
      return res.status(400).json({ error: 'model and node are required' });
    }

    const logService = new LogService(req.app.locals.db);
    const apiKeyService = new ApiKeyService(req.app.locals.db);

    const entry = await logService.recordLog(userId, {
      model,
      node,
      app: req.body.app || 'api',
      in: inTokens,
      out: outTokens,
      speed,
      finish: finish || 'stop',
      key: keyName
    });

    await apiKeyService.recordUsage(hash, inTokens + outTokens);

    res.status(201).json({ success: true, log: entry });
  } catch (error) {
    console.error('Record usage error:', error);
    res.status(500).json({ error: 'Failed to record usage' });
  }
}

module.exports = {
  getLogs,
  recordUsage
};
