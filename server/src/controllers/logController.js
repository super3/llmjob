const LogService = require('../services/logService');
const { handle } = require('./http');
const { clampInt } = require('../clampInt');

// The most rows GET /api/logs may return, and the default when none is asked
// for. Unbounded, `?limit=` reached SQL directly — a non-numeric value made it
// `LIMIT NaN` (a 500), and a huge one scanned the table.
const MAX_LOG_LIMIT = 500;
const DEFAULT_LOG_LIMIT = 50;

function logLimit(v) {
  return clampInt(v, { min: 1, max: MAX_LOG_LIMIT, fallback: DEFAULT_LOG_LIMIT });
}

// GET /api/logs — recent request logs plus the activity histogram for the
// dashboard chart. Authenticated via Clerk (dashboard user).
const getLogs = handle(async (req, res) => {
  const userId = req.user.id;
  const limit = logLimit(req.query.limit);

  const logService = new LogService(req.app.locals.db);
  const logs = await logService.getLogs(userId, limit);
  const activity = await logService.getActivity(userId);

  res.json({ logs, activity });
}, 'Failed to get logs');

module.exports = {
  getLogs,
  logLimit,
  MAX_LOG_LIMIT
};
