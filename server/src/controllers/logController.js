const LogService = require('../services/logService');
const ApiKeyService = require('../services/apiKeyService');

// The largest token count one request may report. Far above any real
// generation, low enough that a bad or malicious value can't meaningfully move
// the lifetime public total.
const MAX_REPORTED_TOKENS = 10000000;

// A non-negative integer token count from an untrusted body field.
function countField(v) {
  const n = Math.floor(Number(v));
  if (!Number.isFinite(n) || n < 0) return 0;
  return Math.min(n, MAX_REPORTED_TOKENS);
}

// A non-negative finite number (tokens/sec) from an untrusted body field.
function numField(v) {
  const n = Number(v);
  if (!Number.isFinite(n) || n < 0) return 0;
  return n;
}

// The most rows GET /api/logs may return, and the default when none is asked
// for. Unbounded, `?limit=` reached SQL directly — a non-numeric value made it
// `LIMIT NaN` (a 500), and a huge one scanned the table.
const MAX_LOG_LIMIT = 500;
const DEFAULT_LOG_LIMIT = 50;

function logLimit(v) {
  const n = Math.floor(Number(v));
  if (!Number.isFinite(n) || n < 1) return DEFAULT_LOG_LIMIT;
  return Math.min(n, MAX_LOG_LIMIT);
}

// GET /api/logs — recent request logs plus the activity histogram for the
// dashboard chart. Authenticated via Clerk (dashboard user).
async function getLogs(req, res) {
  try {
    const userId = req.user.id;
    const limit = logLimit(req.query.limit);

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
    // Coerce, don't trust. These are self-reported by the key holder and land in
    // `usage = usage + $2`, whose SUM is served publicly by /api/chat/usage. Left
    // raw, `{"in":"5","out":"5"}` billed "55" (string concatenation, not
    // addition), a negative erased usage, and 1e18 poisoned the public counter.
    const inTokens = countField(req.body.in);
    const outTokens = countField(req.body.out);
    const speed = numField(req.body.speed);

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
  recordUsage,
  countField,
  numField,
  logLimit,
  MAX_REPORTED_TOKENS,
  MAX_LOG_LIMIT
};
