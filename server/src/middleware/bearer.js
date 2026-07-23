// Pull a bearer token out of the Authorization header.
// Returns the raw token string, or null when the header is missing or not a
// well-formed `Bearer <token>`. Shared by the Clerk-session and API-key auth
// middleware so the parsing lives in one place.
function getBearerToken(req) {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return null;
  }
  const token = authHeader.slice('Bearer '.length).trim();
  return token || null;
}

module.exports = { getBearerToken };
