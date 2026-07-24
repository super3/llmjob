const { requireAuth } = require('./auth');
const { apiKeyAuth } = require('./apiKeyAuth');
const { getBearerToken } = require('./bearer');

// Authenticate with EITHER a Clerk session JWT (the dashboard) or an LLMJob API
// key (SDK callers) — for routes both kinds of caller legitimately reach, such
// as reading back a job you submitted. The bearer's `lj-` prefix picks the
// scheme; both middlewares set `req.user.id` to the same user id, so handlers
// downstream don't care which one ran.
function anyAuth(req, res, next) {
  const token = getBearerToken(req);
  if (!token) {
    return res.status(401).json({ error: 'No authorization token provided' });
  }
  return token.startsWith('lj-') ? apiKeyAuth(req, res, next) : requireAuth(req, res, next);
}

module.exports = { anyAuth };
