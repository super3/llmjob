const crypto = require('crypto');

// Join tokens authorize a machine to attach itself to a user's account from a
// non-interactive context (the `curl | sh` installer), where an interactive
// Clerk login isn't possible. Unlike API keys, a join token is reusable and
// shown repeatedly in the dashboard, so the raw value is stored (not just a
// hash) and can be rotated.
const USER_TOKEN_PREFIX = 'nodejoin:user:';
const TOKEN_PREFIX = 'nodejoin:token:';

function generateToken() {
  return 'ljn_' + crypto.randomBytes(18).toString('hex');
}

class NodeTokenService {
  constructor(redis) {
    // Production passes the real redis v5 client; tests pass an equivalent
    // adapter. Either way it is used directly.
    this.redis = redis;
  }

  // Return the user's current join token, creating one on first use.
  async getOrCreateToken(userId) {
    const userKey = `${USER_TOKEN_PREFIX}${userId}`;
    const existing = await this.redis.get(userKey);
    if (existing) {
      return JSON.parse(existing);
    }
    return this._issue(userId);
  }

  // Replace the user's join token, invalidating the previous one.
  async rotateToken(userId) {
    const userKey = `${USER_TOKEN_PREFIX}${userId}`;
    const existing = await this.redis.get(userKey);
    if (existing) {
      const old = JSON.parse(existing);
      await this.redis.del(`${TOKEN_PREFIX}${old.token}`);
    }
    return this._issue(userId);
  }

  // Resolve a raw token to the owning user id, or null if unknown.
  async verifyToken(rawToken) {
    if (!rawToken) {
      return null;
    }
    const userId = await this.redis.get(`${TOKEN_PREFIX}${rawToken}`);
    return userId || null;
  }

  async _issue(userId) {
    const record = { token: generateToken(), createdAt: Date.now() };
    await this.redis.set(`${USER_TOKEN_PREFIX}${userId}`, JSON.stringify(record));
    await this.redis.set(`${TOKEN_PREFIX}${record.token}`, userId);
    return record;
  }
}

NodeTokenService.generateToken = generateToken;

module.exports = NodeTokenService;
