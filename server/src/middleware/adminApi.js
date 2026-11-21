const crypto = require('crypto');
const { getBearerToken } = require('./bearer');
const { requireAuth } = require('./auth');
const { requireAdmin } = require('./admin');

// Admin access for the container-job calls, which scripts use as well as people.
//
// A person signs in as for any admin route: a Clerk session, then
// ADMIN_USER_IDS (requireAuth + requireAdmin). A script such as the Salad
// watcher can't hold a Clerk session (they last a minute), so it may instead
// send the CONTAINERS_ADMIN_TOKEN secret as its bearer token. With that env var
// unset only the Clerk path exists, so the routes stay closed by default.
function tokenMatches(token) {
  const want = process.env.CONTAINERS_ADMIN_TOKEN || '';
  if (!token || !want) return false;
  const a = Buffer.from(token);
  const b = Buffer.from(want);
  return a.length === b.length && crypto.timingSafeEqual(a, b);
}

function requireAdminApi(req, res, next) {
  if (tokenMatches(getBearerToken(req))) {
    req.user = { id: 'containers-admin-token' };
    return next();
  }
  return requireAuth(req, res, () => requireAdmin(req, res, next));
}

module.exports = { requireAdminApi, tokenMatches };
