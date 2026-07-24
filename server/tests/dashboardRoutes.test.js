const request = require('supertest');
const express = require('express');
const routes = require('../src/routes');
const ApiKeyService = require('../src/services/apiKeyService');
const NodeTokenService = require('../src/services/nodeTokenService');
const NodeService = require('../src/services/nodeService');
const { createTestDb } = require('./helpers/pgmem');

// Mock Clerk so requireAuth resolves to a fixed user. `verifyToken` checks the
// JWT signature in production; here it decodes the test token to recover `sub`.
jest.mock('@clerk/clerk-sdk-node', () => ({
  clerkClient: {
    users: {
      getUser: jest.fn().mockResolvedValue({
        id: 'test_user_123',
        emailAddresses: [{ emailAddress: 'test@example.com' }],
        username: 'testuser'
      })
    }
  },
  verifyToken: jest.fn(async (token) => {
    const payload = JSON.parse(Buffer.from(token.split('.')[1], 'base64').toString());
    return { sub: payload.sub };
  })
}));

const mockToken = () => {
  const payload = { sid: 'sess_123', sub: 'test_user_123' };
  return `header.${Buffer.from(JSON.stringify(payload)).toString('base64')}.signature`;
};

describe('Dashboard API (keys, logs, usage)', () => {
  let app;
  let db;

  beforeEach(async () => {
    app = express();
    app.use(express.json());
    db = await createTestDb();
    app.locals.db = db;
    app.use('/api', routes);
  });

  afterEach(async () => {
    if (db.end) await db.end();
  });

  // Make every service query reject, to exercise the controllers' 500 paths.
  const breakDb = () => { db.query = jest.fn().mockRejectedValue(new Error('boom')); };

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

    it('defaults new keys to public routing', async () => {
      const res = await request(app)
        .post('/api/keys')
        .set('Authorization', `Bearer ${mockToken()}`)
        .send({ name: 'laptop' });
      expect(res.body.visibility).toBe('public');
    });

    it('accepts an explicit private routing at creation', async () => {
      const res = await request(app)
        .post('/api/keys')
        .set('Authorization', `Bearer ${mockToken()}`)
        .send({ name: 'laptop', visibility: 'private' });
      expect(res.status).toBe(201);
      expect(res.body.visibility).toBe('private');
    });

    it('rejects an invalid routing value', async () => {
      const res = await request(app)
        .post('/api/keys')
        .set('Authorization', `Bearer ${mockToken()}`)
        .send({ name: 'laptop', visibility: 'sideways' });
      expect(res.status).toBe(400);
    });

    it('returns 500 when storage fails', async () => {
      breakDb();
      const res = await request(app)
        .post('/api/keys')
        .set('Authorization', `Bearer ${mockToken()}`)
        .send({ name: 'laptop' });
      expect(res.status).toBe(500);
    });
  });

  describe('PUT /api/keys/:id/visibility', () => {
    it('toggles an owned key between public and private', async () => {
      const created = await new ApiKeyService(db).createKey('test_user_123', 'home'); // public by default
      const res = await request(app)
        .put(`/api/keys/${created.id}/visibility`)
        .set('Authorization', `Bearer ${mockToken()}`)
        .send({ visibility: 'private' });
      expect(res.status).toBe(200);
      expect(res.body).toMatchObject({ success: true, id: created.id, visibility: 'private' });
      // reflected in the listing
      const list = await request(app).get('/api/keys').set('Authorization', `Bearer ${mockToken()}`);
      expect(list.body.keys[0].visibility).toBe('private');
    });

    it('rejects an invalid routing value', async () => {
      const created = await new ApiKeyService(db).createKey('test_user_123', 'home');
      const res = await request(app)
        .put(`/api/keys/${created.id}/visibility`)
        .set('Authorization', `Bearer ${mockToken()}`)
        .send({ visibility: 'maybe' });
      expect(res.status).toBe(400);
    });

    it('returns 404 for an unknown or unowned key', async () => {
      const res = await request(app)
        .put('/api/keys/key_missing/visibility')
        .set('Authorization', `Bearer ${mockToken()}`)
        .send({ visibility: 'private' });
      expect(res.status).toBe(404);
    });

    it('falls back to 400 when the service errors without a status', async () => {
      jest.spyOn(ApiKeyService.prototype, 'updateKeyVisibility')
        .mockResolvedValueOnce({ error: 'nope' });
      const res = await request(app)
        .put('/api/keys/key_x/visibility')
        .set('Authorization', `Bearer ${mockToken()}`)
        .send({ visibility: 'private' });
      expect(res.status).toBe(400);
      ApiKeyService.prototype.updateKeyVisibility.mockRestore();
    });

    it('returns 500 when storage fails', async () => {
      breakDb();
      const res = await request(app)
        .put('/api/keys/key_x/visibility')
        .set('Authorization', `Bearer ${mockToken()}`)
        .send({ visibility: 'private' });
      expect(res.status).toBe(500);
    });
  });

  describe('GET /api/keys', () => {
    it('lists the user keys', async () => {
      await new ApiKeyService(db).createKey('test_user_123', 'home');
      const res = await request(app)
        .get('/api/keys')
        .set('Authorization', `Bearer ${mockToken()}`);
      expect(res.status).toBe(200);
      expect(res.body.keys).toHaveLength(1);
      expect(res.body.keys[0].name).toBe('home');
    });

    it('returns 500 when storage fails', async () => {
      breakDb();
      const res = await request(app)
        .get('/api/keys')
        .set('Authorization', `Bearer ${mockToken()}`);
      expect(res.status).toBe(500);
    });
  });

  describe('DELETE /api/keys/:id', () => {
    it('revokes an owned key', async () => {
      const created = await new ApiKeyService(db).createKey('test_user_123', 'home');
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
      breakDb();
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
      breakDb();
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
      const created = await new ApiKeyService(db).createKey('test_user_123', 'home');
      rawKey = created.key;
    });

    it('records a generation, writing a log and billing usage', async () => {
      const res = await request(app)
        .post('/api/usage')
        .set('Authorization', `Bearer ${rawKey}`)
        .send({ model: 'gemma', node: 'rig1', in: 100, out: 50, speed: 90, finish: 'stop' });

      expect(res.status).toBe(201);
      expect(res.body.log.key).toBe('home');

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
      breakDb();
      const res = await request(app)
        .post('/api/usage')
        .set('Authorization', `Bearer ${rawKey}`)
        .send({ model: 'gemma', node: 'rig1' });
      expect(res.status).toBe(500);
    });

    it('returns 500 when recording fails', async () => {
      // Let auth succeed, then make the next query (the log insert) fail.
      const real = db.query.bind(db);
      let calls = 0;
      db.query = jest.fn((...args) => {
        calls += 1;
        return calls === 1 ? real(...args) : Promise.reject(new Error('boom'));
      });
      const res = await request(app)
        .post('/api/usage')
        .set('Authorization', `Bearer ${rawKey}`)
        .send({ model: 'gemma', node: 'rig1' });
      expect(res.status).toBe(500);
    });
  });

  // ---- Node join tokens ---------------------------------------------------
  describe('GET /api/nodes/join-token', () => {
    it('returns a join token for the user', async () => {
      const res = await request(app)
        .get('/api/nodes/join-token')
        .set('Authorization', `Bearer ${mockToken()}`);
      expect(res.status).toBe(200);
      expect(res.body.token).toMatch(/^ljn_/);
    });

    it('returns 500 when storage fails', async () => {
      breakDb();
      const res = await request(app)
        .get('/api/nodes/join-token')
        .set('Authorization', `Bearer ${mockToken()}`);
      expect(res.status).toBe(500);
    });
  });

  describe('POST /api/nodes/join-token/rotate', () => {
    it('issues a different token', async () => {
      const first = await new NodeTokenService(db).getOrCreateToken('test_user_123');
      const res = await request(app)
        .post('/api/nodes/join-token/rotate')
        .set('Authorization', `Bearer ${mockToken()}`);
      expect(res.status).toBe(200);
      expect(res.body.token).toMatch(/^ljn_/);
      expect(res.body.token).not.toBe(first.token);
    });

    it('returns 500 when storage fails', async () => {
      breakDb();
      const res = await request(app)
        .post('/api/nodes/join-token/rotate')
        .set('Authorization', `Bearer ${mockToken()}`);
      expect(res.status).toBe(500);
    });
  });

  describe('POST /api/nodes/join', () => {
    it('registers a node with a valid token and surfaces it in the list', async () => {
      const { token } = await new NodeTokenService(db).getOrCreateToken('test_user_123');

      const res = await request(app)
        .post('/api/nodes/join')
        .send({ token, publicKey: 'pk-join-1', name: 'rig4090' });
      expect(res.status).toBe(201);
      expect(res.body.success).toBe(true);
      expect(res.body.user).toBe('testuser'); // account handle resolved from Clerk

      const list = await request(app)
        .get('/api/nodes')
        .set('Authorization', `Bearer ${mockToken()}`);
      expect(list.body.nodes.some((n) => n.name === 'rig4090')).toBe(true);
    });

    it('returns a null handle when the user has no username', async () => {
      const { clerkClient } = require('@clerk/clerk-sdk-node');
      clerkClient.users.getUser.mockResolvedValueOnce({ id: 'test_user_123', username: null });
      const { token } = await new NodeTokenService(db).getOrCreateToken('test_user_123');
      const res = await request(app)
        .post('/api/nodes/join')
        .send({ token, publicKey: 'pk-join-nouser', name: 'rig-nouser' });
      expect(res.status).toBe(201);
      expect(res.body.user).toBeNull();
    });

    it('still joins (null handle) when the handle lookup fails', async () => {
      const { clerkClient } = require('@clerk/clerk-sdk-node');
      clerkClient.users.getUser.mockRejectedValueOnce(new Error('clerk down'));
      const { token } = await new NodeTokenService(db).getOrCreateToken('test_user_123');
      const res = await request(app)
        .post('/api/nodes/join')
        .send({ token, publicKey: 'pk-join-clerkerr', name: 'rig-err' });
      expect(res.status).toBe(201);
      expect(res.body.user).toBeNull();
    });

    it('defaults the node name when omitted', async () => {
      const { token } = await new NodeTokenService(db).getOrCreateToken('test_user_123');
      const res = await request(app)
        .post('/api/nodes/join')
        .send({ token, publicKey: 'pk-join-2' });
      expect(res.status).toBe(201);
    });

    it('rejects a request missing token or publicKey', async () => {
      const res = await request(app)
        .post('/api/nodes/join')
        .send({ publicKey: 'pk-only' });
      expect(res.status).toBe(400);
    });

    it('rejects an invalid token', async () => {
      const res = await request(app)
        .post('/api/nodes/join')
        .send({ token: 'ljn_bogus', publicKey: 'pk-x', name: 'n' });
      expect(res.status).toBe(401);
    });

    it('surfaces a claim conflict as 400', async () => {
      const { token } = await new NodeTokenService(db).getOrCreateToken('test_user_123');
      await new NodeService(db).claimNode('pk-taken', 'taken', 'other_user');

      const res = await request(app)
        .post('/api/nodes/join')
        .send({ token, publicKey: 'pk-taken', name: 'n' });
      expect(res.status).toBe(400);
    });

    it('returns 500 when verification fails', async () => {
      breakDb();
      const res = await request(app)
        .post('/api/nodes/join')
        .send({ token: 'ljn_whatever', publicKey: 'pk-x', name: 'n' });
      expect(res.status).toBe(500);
    });
  });
});
