const { verifyToken } = require('@clerk/clerk-sdk-node');
const { getBearerToken } = require('./bearer');

// Authenticate a dashboard request using a Clerk session JWT.
//
// The token's signature is verified against Clerk's JWKS via `verifyToken`
// (using CLERK_SECRET_KEY) before any claim is trusted. Earlier versions
// base64-decoded the payload and trusted its `sub`/`sid` without checking the
// signature — that let a forged token impersonate any user. Only the verified
// payload's `sub` is used.
//
// `sub` is the whole identity. This used to fetch the Clerk user on every
// request to fill in email and username, which nothing reads.
async function requireAuth(req, res, next) {
  try {
    const token = getBearerToken(req);
    if (!token) {
      return res.status(401).json({ error: 'No authorization token provided' });
    }

    try {
      const payload = await verifyToken(token, {
        secretKey: process.env.CLERK_SECRET_KEY,
      });

      req.user = { id: payload.sub };

      next();
    } catch (error) {
      console.error('Token verification failed:', error);
      return res.status(401).json({ error: 'Invalid or expired token' });
    }
  } catch (error) {
    console.error('Auth middleware error:', error);
    return res.status(500).json({ error: 'Authentication error' });
  }
}

module.exports = { requireAuth };
