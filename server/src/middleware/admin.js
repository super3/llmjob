// Restrict a route to admin users. Runs after `requireAuth`, so `req.user` is
// already populated. The allow-list is the comma-separated ADMIN_USER_IDS env
// var (Clerk user ids); when it is empty no one is an admin, so the route is
// closed by default rather than open.
function adminIds() {
  return (process.env.ADMIN_USER_IDS || '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
}

function requireAdmin(req, res, next) {
  const userId = req.user && req.user.id;
  if (!userId || !adminIds().includes(userId)) {
    return res.status(403).json({ error: 'Admin access required' });
  }
  next();
}

module.exports = { requireAdmin };
