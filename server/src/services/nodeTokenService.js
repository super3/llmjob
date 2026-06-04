const crypto = require('crypto');

// Join tokens authorize a machine to attach itself to a user's account from a
// non-interactive context (the installer), where an interactive Clerk login
// isn't possible. The token is reusable and shown repeatedly in the dashboard,
// so the raw value is stored (one row per user) and can be rotated.
function generateToken() {
  return 'ljn_' + crypto.randomBytes(18).toString('hex');
}

class NodeTokenService {
  constructor(db) {
    this.db = db;
  }

  // Return the user's current join token, creating one on first use.
  async getOrCreateToken(userId) {
    const r = await this.db.query(
      'SELECT token, created_at FROM node_join_tokens WHERE user_id = $1',
      [userId]
    );
    if (r.rows.length > 0) {
      return { token: r.rows[0].token, createdAt: Number(r.rows[0].created_at) };
    }
    return this._issue(userId);
  }

  // Replace the user's join token, invalidating the previous one.
  async rotateToken(userId) {
    return this._issue(userId);
  }

  // Resolve a raw token to the owning user id, or null if unknown.
  async verifyToken(rawToken) {
    if (!rawToken) {
      return null;
    }
    const r = await this.db.query(
      'SELECT user_id FROM node_join_tokens WHERE token = $1',
      [rawToken]
    );
    return r.rows.length > 0 ? r.rows[0].user_id : null;
  }

  async _issue(userId) {
    const token = generateToken();
    const createdAt = Date.now();
    await this.db.query(
      `INSERT INTO node_join_tokens (user_id, token, created_at)
       VALUES ($1, $2, $3)
       ON CONFLICT (user_id) DO UPDATE SET token = EXCLUDED.token, created_at = EXCLUDED.created_at`,
      [userId, token, createdAt]
    );
    return { token, createdAt };
  }
}

NodeTokenService.generateToken = generateToken;

module.exports = NodeTokenService;
