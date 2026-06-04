const request = require('supertest');
const express = require('express');
const routes = require('../src/routes');
const ApiKeyService = require('../src/services/apiKeyService');
const { createCamelClient } = require('./helpers/camelRedis');

// Mock Clerk so requireAuth resolves to a fixed user.
jest.mock('@clerk/clerk-sdk-node', () => ({
  clerkClient: {
    sessions: {
      getSession: jest.fn().mockResolvedValue({ userId: 'test_user_123' })
    },
    users: {
      getUser: jest.fn().mockResolvedValue({
        id: 'test_user_123',
        emailAddresses: [{ emailAddress: 'test@example.com' }],
        username: 'testuser'
      })
    }
  }
}));

const mockToken = () => {
  const payload = { sid: 'sess_123', sub: 'test_user_123' };
  return `header.${Buffer.from(JSON.stringify(payload)).toString('base64')}.signature`;
};

describe('Dashboard API (keys, logs, usage)', () => {
  let app;
  let redisClient;

  beforeEach(async () => {
    app = express();
    app.use(express.json());
    redisClient = createCamelClient();
    app.locals.redis = redisClient;
    app.use('/api', routes);
    await redisClient.flushall();
  });

  afterEach(async () => {
    await redisClient.quit();
  });

  // ---- API keys -----------------------------------------------------------
  describe('POST /api/keys', () => {
    it('creates a key and returns the raw secret once', async () => {
      const res = await request(app)
        .post('/api/keys')
        .set('Authorization', `Bearer ${mockToken()}`)
        .send({ name: 'laptop' });

      expect(res.status).toBe(201);
      expect(res.body.key).toMatch(/^lj-live-/);
      expect(res.body.name).toBe('laptop');
    });

    it('trims whitespace from the name', async () => {
      const res = await request(app)
        .post('/api/keys')
        .set('Authorization', `Bearer ${mockToken()}`)
        .send({ name: '  spaced  ' });
      expect(res.body.name).toBe('spaced');
    });

    it('rejects a missing name', async () => {
      const res = await request(app)
        .post('/api/keys')
        .set('Authorization', `Bearer ${mockToken()}`)
        .send({ name: '   ' });
      expect(res.status).toBe(400);
    });

    it('returns 500 when storage fails', async () => {
      redisClient.set = jest.fn().mockRejectedValue(new Error('boom'));
      const res = await request(app)
        .post('/api/keys')
        .set('Authorization', `Bearer ${mockToken()}`)
        .send({ name: 'laptop' });
      expect(res.status).toBe(500);
    });
  });

  describe('GET /api/keys', () => {
    it('lists the user keys', async () => {
      await new ApiKeyService(redisClient).createKey('test_user_123', 'home');
      const res = await request(app)
        .get('/api/keys')
        .set('Authorization', `Bearer ${mockToken()}`);
      expect(res.status).toBe(200);
      expect(res.body.keys).toHaveLength(1);
      expect(res.body.keys[0].name).toBe('home');
    });

    it('returns 500 when storage fails', async () => {
      redisClient.sMembers = jest.fn().mockRejectedValue(new Error('boom'));
      const res = await request(app)
        .get('/api/keys')
        .set('Authorization', `Bearer ${mockToken()}`);
      expect(res.status).toBe(500);
    });
  });

  describe('DELETE /api/keys/:id', () => {
    it('revokes an owned key', async () => {
      const created = await new ApiKeyService(redisClient).createKey('test_user_123', 'home');
      const res = await request(app)
        .delete(`/api/keys/${created.id}`)
        .set('Authorization', `Bearer ${mockToken()}`);
      expect(res.status).toBe(200);
      expect(res.body.success).toBe(true);
    });

    it('returns 404 for an unknown key', async () => {
      const res = await request(app)
        .delete('/api/keys/key_missing')
        .set('Authorization', `Bearer ${mockToken()}`);
      expect(res.status).toBe(404);
    });

    it('falls back to 400 when the service errors without a status', async () => {
      jest.spyOn(ApiKeyService.prototype, 'revokeKey')
        .mockResolvedValueOnce({ error: 'nope' });
      const res = await request(app)
        .delete('/api/keys/key_x')
        .set('Authorization', `Bearer ${mockToken()}`);
      expect(res.status).toBe(400);
      ApiKeyService.prototype.revokeKey.mockRestore();
    });

    it('returns 500 when storage fails', async () => {
      redisClient.sMembers = jest.fn().mockRejectedValue(new Error('boom'));
      const res = await request(app)
        .delete('/api/keys/key_x')
        .set('Authorization', `Bearer ${mockToken()}`);
      expect(res.status).toBe(500);
    });
  });

  // ---- Logs ---------------------------------------------------------------
  describe('GET /api/logs', () => {
    it('returns logs and an activity histogram', async () => {
      const res = await request(app)
        .get('/api/logs')
        .set('Authorization', `Bearer ${mockToken()}`);
      expect(res.status).toBe(200);
      expect(res.body.logs).toEqual([]);
      expect(res.body.activity).toHaveLength(24);
    });

    it('respects an explicit limit', async () => {
      const res = await request(app)
        .get('/api/logs?limit=10')
        .set('Authorization', `Bearer ${mockToken()}`);
      expect(res.status).toBe(200);
    });

    it('returns 500 when storage fails', async () => {
      redisClient.zRange = jest.fn().mockRejectedValue(new Error('boom'));
      const res = await request(app)
        .get('/api/logs')
        .set('Authorization', `Bearer ${mockToken()}`);
      expect(res.status).toBe(500);
    });
  });

  // ---- Usage (API-key auth) ----------------------------------------------
  describe('POST /api/usage', () => {
    let rawKey;

    beforeEach(async () => {
      const created = await new ApiKeyService(redisClient).createKey('test_user_123', 'home');
      rawKey = created.key;
    });

    it('records a generation, writing a log and billing usage', async () => {
      const res = await request(app)
        .post('/api/usage')
        .set('Authorization', `Bearer ${rawKey}`)
        .send({ model: 'gemma', node: 'rig1', in: 100, out: 50, speed: 90, finish: 'stop' });

      expect(res.status).toBe(201);
      expect(res.body.log.key).toBe('home');

      // The log shows up for the dashboard user, and usage was billed.
      const logsRes = await request(app)
        .get('/api/logs')
        .set('Authorization', `Bearer ${mockToken()}`);
      expect(logsRes.body.logs).toHaveLength(1);

      const keysRes = await request(app)
        .get('/api/keys')
        .set('Authorization', `Bearer ${mockToken()}`);
      expect(keysRes.body.keys[0].usage).toBe(150);
    });

    it('defaults the app field when omitted', async () => {
      const res = await request(app)
        .post('/api/usage')
        .set('Authorization', `Bearer ${rawKey}`)
        .send({ model: 'gemma', node: 'rig1' });
      expect(res.body.log.app).toBe('api');
    });

    it('rejects requests missing model or node', async () => {
      const res = await request(app)
        .post('/api/usage')
        .set('Authorization', `Bearer ${rawKey}`)
        .send({ model: 'gemma' });
      expect(res.status).toBe(400);
    });

    it('rejects a missing bearer token', async () => {
      const res = await request(app)
        .post('/api/usage')
        .send({ model: 'gemma', node: 'rig1' });
      expect(res.status).toBe(401);
    });

    it('rejects a non-LLMJob token', async () => {
      const res = await request(app)
        .post('/api/usage')
        .set('Authorization', 'Bearer sk-someother')
        .send({ model: 'gemma', node: 'rig1' });
      expect(res.status).toBe(401);
    });

    it('rejects an unknown API key', async () => {
      const res = await request(app)
        .post('/api/usage')
        .set('Authorization', 'Bearer lj-live-unknown')
        .send({ model: 'gemma', node: 'rig1' });
      expect(res.status).toBe(401);
    });

    it('returns 500 when key verification fails', async () => {
      redisClient.get = jest.fn().mockRejectedValue(new Error('boom'));
      const res = await request(app)
        .post('/api/usage')
        .set('Authorization', `Bearer ${rawKey}`)
        .send({ model: 'gemma', node: 'rig1' });
      expect(res.status).toBe(500);
    });

    it('returns 500 when recording fails', async () => {
      redisClient.zAdd = jest.fn().mockRejectedValue(new Error('boom'));
      const res = await request(app)
        .post('/api/usage')
        .set('Authorization', `Bearer ${rawKey}`)
        .send({ model: 'gemma', node: 'rig1' });
      expect(res.status).toBe(500);
    });
  });
});
