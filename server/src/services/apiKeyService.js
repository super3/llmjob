const crypto = require('crypto');

// API keys let a user authenticate OpenAI-compatible API requests without
// going through Clerk. Only a SHA-256 hash of each key is ever stored, so the
// raw secret is shown exactly once at creation time and can never be recovered.
const KEY_PREFIX = 'apikey:';
const USER_KEYS_PREFIX = 'user_apikeys:';

function sha256(value) {
  return crypto.createHash('sha256').update(value).digest('hex');
}

// Build the public-facing, redacted form of a key, e.g. "lj-live-7c2f…3f9a".
function maskKey(rawKey) {
  return `${rawKey.slice(0, 8)}${rawKey.slice(8, 12)}…${rawKey.slice(-4)}`;
}

class ApiKeyService {
  constructor(redis) {
    // Production passes the real redis v5 client (camelCase, promise-based);
    // tests pass an equivalent adapter. Either way we use it directly.
    this.redis = redis;
  }

  // Generate a fresh secret. `lj-live-` prefix mirrors the dashboard mock.
  static generateKey() {
    const raw = 'lj-live-' + crypto.randomBytes(16).toString('hex');
    return { raw, hash: sha256(raw), masked: maskKey(raw) };
  }

  // Create and persist a new key. Returns the raw secret (shown once) plus
  // the stored metadata.
  async createKey(userId, name) {
    const { raw, hash, masked } = ApiKeyService.generateKey();
    const id = 'key_' + crypto.randomBytes(6).toString('hex');

    const meta = {
      id,
      userId,
      name,
      masked,
      createdAt: Date.now(),
      lastUsed: null,
      usage: 0
    };

    await this.redis.set(`${KEY_PREFIX}${hash}`, JSON.stringify(meta));
    await this.redis.sAdd(`${USER_KEYS_PREFIX}${userId}`, hash);

    return { ...meta, key: raw };
  }

  // List a user's keys (redacted), newest first.
  async listKeys(userId) {
    const hashes = await this.redis.sMembers(`${USER_KEYS_PREFIX}${userId}`);
    if (!hashes || hashes.length === 0) {
      return [];
    }

    const keys = [];
    for (const hash of hashes) {
      const data = await this.redis.get(`${KEY_PREFIX}${hash}`);
      if (data) {
        const meta = JSON.parse(data);
        keys.push({
          id: meta.id,
          name: meta.name,
          masked: meta.masked,
          createdAt: meta.createdAt,
          lastUsed: meta.lastUsed,
          usage: meta.usage
        });
      }
    }

    keys.sort((a, b) => b.createdAt - a.createdAt);
    return keys;
  }

  // Resolve a raw key to its owner. Updates lastUsed on success.
  async verifyKey(rawKey) {
    const hash = sha256(rawKey);
    const data = await this.redis.get(`${KEY_PREFIX}${hash}`);
    if (!data) {
      return null;
    }

    const meta = JSON.parse(data);
    meta.lastUsed = Date.now();
    await this.redis.set(`${KEY_PREFIX}${hash}`, JSON.stringify(meta));

    return { userId: meta.userId, id: meta.id, name: meta.name, hash };
  }

  // Add token usage to a key identified by its hash.
  async recordUsage(hash, tokens) {
    const data = await this.redis.get(`${KEY_PREFIX}${hash}`);
    if (!data) {
      return { error: 'Key not found' };
    }

    const meta = JSON.parse(data);
    meta.usage = (meta.usage || 0) + tokens;
    await this.redis.set(`${KEY_PREFIX}${hash}`, JSON.stringify(meta));

    return { success: true, usage: meta.usage };
  }

  // Revoke a key by its public id. Verifies ownership via the user's set.
  async revokeKey(userId, keyId) {
    const hashes = await this.redis.sMembers(`${USER_KEYS_PREFIX}${userId}`);

    for (const hash of hashes) {
      const data = await this.redis.get(`${KEY_PREFIX}${hash}`);
      if (data) {
        const meta = JSON.parse(data);
        if (meta.id === keyId) {
          await this.redis.del(`${KEY_PREFIX}${hash}`);
          await this.redis.sRem(`${USER_KEYS_PREFIX}${userId}`, hash);
          return { success: true, id: keyId, message: 'Key revoked' };
        }
      }
    }

    return { error: 'Key not found', status: 404 };
  }
}

ApiKeyService.maskKey = maskKey;
ApiKeyService.sha256 = sha256;

module.exports = ApiKeyService;
