const crypto = require('crypto');

// API keys let a user authenticate OpenAI-compatible API requests without
// going through Clerk. Only a SHA-256 hash of each key is ever stored, so the
// raw secret is shown exactly once at creation time and can never be recovered.
function sha256(value) {
  return crypto.createHash('sha256').update(value).digest('hex');
}

// Build the public-facing, redacted form of a key, e.g. "lj-live-7c2f…3f9a".
function maskKey(rawKey) {
  return `${rawKey.slice(0, 8)}${rawKey.slice(8, 12)}…${rawKey.slice(-4)}`;
}

// A key's request routing: 'private' sends its requests only to the owner's own
// nodes; anything else is 'public' (any node on the network may serve them).
function normalizeVisibility(v) {
  return v === 'private' ? 'private' : 'public';
}

class ApiKeyService {
  constructor(db) {
    this.db = db;
  }

  // Generate a fresh secret. `lj-live-` prefix mirrors the dashboard mock.
  static generateKey() {
    const raw = 'lj-live-' + crypto.randomBytes(16).toString('hex');
    return { raw, hash: sha256(raw), masked: maskKey(raw) };
  }

  // Create and persist a new key. Returns the raw secret (shown once) plus
  // the stored metadata. `visibility` defaults to 'public'.
  async createKey(userId, name, visibility) {
    const { raw, hash, masked } = ApiKeyService.generateKey();
    const id = 'key_' + crypto.randomBytes(6).toString('hex');
    const createdAt = Date.now();
    const vis = normalizeVisibility(visibility);

    await this.db.query(
      `INSERT INTO api_keys (hash, id, user_id, name, masked, created_at, last_used, usage, visibility)
       VALUES ($1, $2, $3, $4, $5, $6, NULL, 0, $7)`,
      [hash, id, userId, name, masked, createdAt, vis]
    );

    return { id, userId, name, masked, createdAt, lastUsed: null, usage: 0, visibility: vis, key: raw };
  }

  // List a user's keys (redacted), newest first.
  async listKeys(userId) {
    const r = await this.db.query(
      `SELECT id, name, masked, created_at, last_used, usage, visibility
       FROM api_keys WHERE user_id = $1 ORDER BY created_at DESC`,
      [userId]
    );
    return r.rows.map((row) => ({
      id: row.id,
      name: row.name,
      masked: row.masked,
      createdAt: Number(row.created_at),
      lastUsed: row.last_used == null ? null : Number(row.last_used),
      usage: Number(row.usage),
      visibility: normalizeVisibility(row.visibility)
    }));
  }

  // Resolve a raw key to its owner (and its routing visibility). Updates lastUsed
  // on success.
  async verifyKey(rawKey) {
    const hash = sha256(rawKey);
    const r = await this.db.query(
      'UPDATE api_keys SET last_used = $2 WHERE hash = $1 RETURNING user_id, id, name, visibility',
      [hash, Date.now()]
    );
    if (r.rows.length === 0) {
      return null;
    }
    const row = r.rows[0];
    return { userId: row.user_id, id: row.id, name: row.name, visibility: normalizeVisibility(row.visibility), hash };
  }

  // Flip a key's request routing between 'public' and 'private'. Verifies
  // ownership via the user id; the change applies to the key's future requests.
  async updateKeyVisibility(userId, keyId, visibility) {
    const vis = normalizeVisibility(visibility);
    const r = await this.db.query(
      'UPDATE api_keys SET visibility = $3 WHERE user_id = $1 AND id = $2 RETURNING id',
      [userId, keyId, vis]
    );
    if (r.rows.length === 0) {
      return { error: 'Key not found', status: 404 };
    }
    return { success: true, id: keyId, visibility: vis };
  }

  // Add token usage to a key identified by its hash.
  async recordUsage(hash, tokens) {
    const r = await this.db.query(
      'UPDATE api_keys SET usage = usage + $2 WHERE hash = $1 RETURNING usage',
      [hash, tokens]
    );
    if (r.rows.length === 0) {
      return { error: 'Key not found' };
    }
    return { success: true, usage: Number(r.rows[0].usage) };
  }

  // Revoke a key by its public id. Verifies ownership via the user id.
  async revokeKey(userId, keyId) {
    const r = await this.db.query(
      'DELETE FROM api_keys WHERE user_id = $1 AND id = $2 RETURNING id',
      [userId, keyId]
    );
    if (r.rows.length === 0) {
      return { error: 'Key not found', status: 404 };
    }
    return { success: true, id: keyId, message: 'Key revoked' };
  }
}

ApiKeyService.maskKey = maskKey;
ApiKeyService.sha256 = sha256;
ApiKeyService.normalizeVisibility = normalizeVisibility;

module.exports = ApiKeyService;
