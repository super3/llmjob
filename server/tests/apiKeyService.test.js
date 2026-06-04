const ApiKeyService = require('../src/services/apiKeyService');
const { createCamelClient } = require('./helpers/camelRedis');

describe('ApiKeyService', () => {
  let redisClient;
  let service;

  beforeEach(async () => {
    redisClient = createCamelClient();
    service = new ApiKeyService(redisClient);
    await redisClient.flushall();
  });

  afterEach(async () => {
    await redisClient.quit();
  });

  describe('generateKey', () => {
    it('produces a prefixed secret, hash, and mask', () => {
      const { raw, hash, masked } = ApiKeyService.generateKey();
      expect(raw).toMatch(/^lj-live-[0-9a-f]{32}$/);
      expect(hash).toHaveLength(64);
      expect(masked).toContain('…');
      expect(masked.startsWith('lj-live-')).toBe(true);
      expect(masked.endsWith(raw.slice(-4))).toBe(true);
    });
  });

  describe('maskKey helper', () => {
    it('redacts the middle of the key', () => {
      const masked = ApiKeyService.maskKey('lj-live-abcd1234deadbeef');
      expect(masked).toBe('lj-live-abcd…beef');
    });
  });

  describe('createKey', () => {
    it('stores a key and returns the raw secret once', async () => {
      const result = await service.createKey('user1', 'laptop');

      expect(result.key).toMatch(/^lj-live-/);
      expect(result.id).toMatch(/^key_/);
      expect(result.name).toBe('laptop');
      expect(result.usage).toBe(0);
      expect(result.lastUsed).toBeNull();

      const hashes = await redisClient.sMembers('user_apikeys:user1');
      expect(hashes).toHaveLength(1);
    });
  });

  describe('listKeys', () => {
    it('returns an empty array when the user has no keys', async () => {
      expect(await service.listKeys('nobody')).toEqual([]);
    });

    it('returns redacted keys newest first', async () => {
      const a = await service.createKey('user1', 'first');
      // Force a later createdAt for the second key.
      const b = await service.createKey('user1', 'second');
      await bumpCreatedAt(redisClient, 'user1', b.id, a.createdAt + 1000);

      const keys = await service.listKeys('user1');
      expect(keys).toHaveLength(2);
      expect(keys[0].name).toBe('second');
      expect(keys[1].name).toBe('first');
      // No secret material is ever returned.
      expect(keys[0].key).toBeUndefined();
      expect(keys[0].masked).toContain('…');
    });

    it('skips hashes whose metadata is missing', async () => {
      await service.createKey('user1', 'real');
      await redisClient.sAdd('user_apikeys:user1', 'dangling-hash');

      const keys = await service.listKeys('user1');
      expect(keys).toHaveLength(1);
    });
  });

  describe('verifyKey', () => {
    it('resolves a valid key and stamps lastUsed', async () => {
      const created = await service.createKey('user1', 'laptop');

      const resolved = await service.verifyKey(created.key);
      expect(resolved).toMatchObject({ userId: 'user1', id: created.id, name: 'laptop' });
      expect(resolved.hash).toHaveLength(64);

      const keys = await service.listKeys('user1');
      expect(keys[0].lastUsed).toEqual(expect.any(Number));
    });

    it('returns null for an unknown key', async () => {
      expect(await service.verifyKey('lj-live-doesnotexist')).toBeNull();
    });
  });

  describe('recordUsage', () => {
    it('adds tokens to the running total', async () => {
      const created = await service.createKey('user1', 'laptop');
      const resolved = await service.verifyKey(created.key);

      const r1 = await service.recordUsage(resolved.hash, 100);
      expect(r1).toEqual({ success: true, usage: 100 });
      const r2 = await service.recordUsage(resolved.hash, 50);
      expect(r2.usage).toBe(150);
    });

    it('errors when the key hash is unknown', async () => {
      expect(await service.recordUsage('nope', 10)).toEqual({ error: 'Key not found' });
    });
  });

  describe('revokeKey', () => {
    it('removes a key the user owns', async () => {
      const created = await service.createKey('user1', 'laptop');

      const result = await service.revokeKey('user1', created.id);
      expect(result).toEqual({ success: true, id: created.id, message: 'Key revoked' });
      expect(await service.listKeys('user1')).toEqual([]);
    });

    it('returns 404 when the id is not found, skipping dangling hashes', async () => {
      await service.createKey('user1', 'laptop');
      await redisClient.sAdd('user_apikeys:user1', 'dangling-hash');
      const result = await service.revokeKey('user1', 'key_missing');
      expect(result).toEqual({ error: 'Key not found', status: 404 });
    });

    it('ignores dangling hashes while searching', async () => {
      const created = await service.createKey('user1', 'laptop');
      await redisClient.sAdd('user_apikeys:user1', 'dangling-hash');

      const result = await service.revokeKey('user1', created.id);
      expect(result.success).toBe(true);
    });
  });
});

// Helper: rewrite a key's createdAt so list ordering is deterministic.
async function bumpCreatedAt(redis, userId, keyId, createdAt) {
  const hashes = await redis.sMembers(`user_apikeys:${userId}`);
  for (const hash of hashes) {
    const data = await redis.get(`apikey:${hash}`);
    const meta = JSON.parse(data);
    if (meta.id === keyId) {
      meta.createdAt = createdAt;
      await redis.set(`apikey:${hash}`, JSON.stringify(meta));
    }
  }
}
