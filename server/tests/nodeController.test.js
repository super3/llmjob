const request = require('supertest');
const express = require('express');
const nacl = require('tweetnacl');
const naclUtil = require('tweetnacl-util');
const routes = require('../src/routes');
const NodeService = require('../src/services/nodeService');
const { createTestDb } = require('./helpers/pgmem');

// Mock Clerk. `verifyToken` checks the JWT signature in production; here it just
// decodes the (unsigned) test token to recover its `sub`, so authHeader(sub)
// resolves to that user.
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

const authHeader = (sub = 'test_user_123') => {
  const payload = { sid: 'sess_123', sub };
  return `Bearer header.${Buffer.from(JSON.stringify(payload)).toString('base64')}.signature`;
};

describe('Node API Endpoints', () => {
  let app;
  let db;
  let testKeypair;
  let testPublicKey;
  let testNodeId;

  beforeAll(() => {
    testKeypair = nacl.sign.keyPair();
    testPublicKey = naclUtil.encodeBase64(testKeypair.publicKey);
    // Derived, not hardcoded to a width: the service owns how wide a node id is
    // (it grew from 6 hex characters to 16), and a fixture that pins the old
    // width tests the fixture rather than the code.
    testNodeId = NodeService.generateNodeFingerprint(testPublicKey);
  });

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

  const seedNode = (over = {}) => {
    const n = {
      node_id: 'n', public_key: 'k', name: 'N', user_id: 'u', status: 'online',
      is_public: false, last_seen: Date.now(), claimed_at: Date.now(), ...over
    };
    return db.query(
      `INSERT INTO nodes (node_id, public_key, name, user_id, status, is_public, last_seen, claimed_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8)`,
      [n.node_id, n.public_key, n.name, n.user_id, n.status, n.is_public, n.last_seen, n.claimed_at]
    );
  };

  const signedPing = (over = {}) => {
    const timestamp = over.timestamp ?? Date.now();
    const message = `${testNodeId}:${timestamp}`;
    const signature = naclUtil.encodeBase64(nacl.sign.detached(naclUtil.decodeUTF8(message), testKeypair.secretKey));
    return { publicKey: testPublicKey, signature, timestamp, nodeId: testNodeId, ...over };
  };

  // Both routes are gone. Claiming happens through /api/nodes/join with a join
  // token, and nothing sets a node public any more.
  describe('removed node routes', () => {
    it('no longer serves POST /api/nodes/claim', async () => {
      const response = await request(app)
        .post('/api/nodes/claim')
        .set('Authorization', authHeader())
        .send(signedPing({ name: 'Test Node' }));
      expect(response.status).toBe(404);
    });

    it('no longer serves PUT /api/nodes/:id/visibility', async () => {
      const response = await request(app)
        .put(`/api/nodes/${testNodeId}/visibility`)
        .set('Authorization', authHeader())
        .send({ isPublic: true });
      expect(response.status).toBe(404);
    });
  });

  describe('POST /api/nodes/register', () => {
    it('registers an unclaimed node from a signature alone — no token, no session', async () => {
      const response = await request(app).post('/api/nodes/register').send(signedPing({ name: 'Anon rig' }));
      expect(response.status).toBe(200);
      expect(response.body).toMatchObject({ success: true, nodeId: testNodeId, claimed: false });

      // The row exists with no owner, so /jobs/poll will answer it and the queue
      // will hand it public (non-private) work only.
      const row = await db.query('SELECT user_id, status FROM nodes WHERE node_id = $1', [testNodeId]);
      expect(row.rows[0]).toMatchObject({ user_id: null, status: 'online' });
    });

    it('lets a registered node ping, which it could not do before registering', async () => {
      expect((await request(app).post('/api/nodes/ping').send(signedPing())).status).toBe(400);
      await request(app).post('/api/nodes/register').send(signedPing());
      expect((await request(app).post('/api/nodes/ping').send(signedPing())).status).toBe(200);
    });

    it('reports an already-claimed node as claimed without reassigning it', async () => {
      await seedNode({ node_id: testNodeId, public_key: testPublicKey, name: 'Mine', user_id: 'test_user_123', status: 'offline' });
      const response = await request(app).post('/api/nodes/register').send(signedPing());
      expect(response.body).toMatchObject({ success: true, claimed: true });
      const row = await db.query('SELECT user_id FROM nodes WHERE node_id = $1', [testNodeId]);
      expect(row.rows[0].user_id).toBe('test_user_123');
    });

    it('still requires a valid signature', async () => {
      const response = await request(app)
        .post('/api/nodes/register')
        .send({ publicKey: testPublicKey, signature: 'invalid_signature', timestamp: Date.now(), nodeId: testNodeId });
      expect(response.status).toBe(401);
    });
  });

  describe('POST /api/nodes/ping', () => {
    beforeEach(async () => {
      await seedNode({ node_id: testNodeId, public_key: testPublicKey, name: 'Test Node', user_id: 'test_user_123', status: 'offline' });
    });

    it('should update node status with valid signature', async () => {
      const response = await request(app).post('/api/nodes/ping').send(signedPing());
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.status).toBe('online');
    });

    it('should reject ping with invalid signature', async () => {
      const response = await request(app)
        .post('/api/nodes/ping')
        .send({ publicKey: testPublicKey, signature: 'invalid_signature', timestamp: Date.now(), nodeId: testNodeId });
      expect(response.status).toBe(401);
      expect(response.body.error).toBe('Invalid signature format');
    });

    it('should reject ping with old timestamp', async () => {
      const response = await request(app).post('/api/nodes/ping').send(signedPing({ timestamp: Date.now() - 10 * 60 * 1000 }));
      expect(response.status).toBe(401);
      expect(response.body.error).toBe('Timestamp too old or too far in future');
    });

    it('should reject ping with missing fields', async () => {
      const response = await request(app)
        .post('/api/nodes/ping')
        .send({ publicKey: testPublicKey, nodeId: testNodeId });
      expect(response.status).toBe(400);
      expect(response.body.error).toBe('Missing required fields');
    });
  });

  describe('GET /api/nodes', () => {
    beforeEach(async () => {
      await seedNode({ node_id: 'node1', public_key: 'key1', name: 'Node 1', user_id: 'test_user_123', status: 'online' });
      await seedNode({ node_id: 'node2', public_key: 'key2', name: 'Node 2', user_id: 'test_user_123', is_public: true, last_seen: Date.now() - 20 * 60 * 1000 });
    });

    it('should return user nodes with auth', async () => {
      const response = await request(app).get('/api/nodes').set('Authorization', authHeader());
      expect(response.status).toBe(200);
      expect(response.body.nodes).toHaveLength(2);
      expect(response.body.nodes[0].nodeId).toBe('node1');
      expect(response.body.nodes[0].status).toBe('online');
      expect(response.body.nodes[1].nodeId).toBe('node2');
      expect(response.body.nodes[1].status).toBe('offline');
    });

    it('should reject without auth token', async () => {
      const response = await request(app).get('/api/nodes');
      expect(response.status).toBe(401);
      expect(response.body.error).toBe('No authorization token provided');
    });
  });

  describe('Error handling', () => {
    it('should handle database errors in pingNode', async () => {
      await seedNode({ node_id: testNodeId, public_key: testPublicKey, name: 'Test Node', user_id: 'test_user_123', status: 'offline' });
      db.query = jest.fn().mockRejectedValue(new Error('Database error'));
      const response = await request(app).post('/api/nodes/ping').send(signedPing());
      expect(response.status).toBe(500);
      expect(response.body.error).toBe('Failed to update node status');
    });

    it('should handle database errors in registerNode', async () => {
      db.query = jest.fn().mockRejectedValue(new Error('Database error'));
      const response = await request(app).post('/api/nodes/register').send(signedPing());
      expect(response.status).toBe(500);
      expect(response.body.error).toBe('Failed to register node');
    });

    it('should handle service errors in registerNode', async () => {
      const spy = jest.spyOn(NodeService.prototype, 'registerNode')
        .mockResolvedValue({ error: 'nope' });
      const response = await request(app).post('/api/nodes/register').send(signedPing());
      expect(response.status).toBe(400);
      expect(response.body.error).toBe('nope');
      spy.mockRestore();
    });

    it('should handle database errors in getUserNodes', async () => {
      db.query = jest.fn().mockRejectedValue(new Error('Database error'));
      const response = await request(app).get('/api/nodes').set('Authorization', authHeader());
      expect(response.status).toBe(500);
      expect(response.body.error).toBe('Failed to get nodes');
    });

    // GET /api/nodes/serving — the only endpoint that enumerates the fleet that
    // actually serves jobs, and the source for the network page's speed dots.
    it('lists the serving fleet with measured speeds', async () => {
      const svc = new NodeService(db);
      const { nodeId } = await svc.claimNode('pk-serving', 'rig-serving', 'user-serving');
      await svc.recordSpeedSample(nodeId, 400, 10000); // 40 tok/s

      const response = await request(app).get('/api/nodes/serving');
      expect(response.status).toBe(200);
      expect(response.body.total).toBe(response.body.nodes.length);
      const node = response.body.nodes.find((n) => n.nodeId === nodeId);
      expect(node).toMatchObject({ tps: 40, stale: false, samples: 1 });
      expect(node.name).toBeUndefined();   // unauthenticated: no owner-set strings
      expect(node.device).toBeUndefined();
    });

    it('should handle database errors in getServingNodes', async () => {
      const spy = jest.spyOn(NodeService.prototype, 'listServingNodes')
        .mockRejectedValue(new Error('Database error'));
      const response = await request(app).get('/api/nodes/serving');
      expect(response.status).toBe(500);
      expect(response.body.error).toBe('Failed to get serving nodes');
      spy.mockRestore();
    });

    it('should handle service errors in pingNode', async () => {
      // Node exists but with no stored public key -> "Public key mismatch" (400).
      await seedNode({ node_id: testNodeId, public_key: null, name: 'Test Node', user_id: 'test_user_123', status: 'offline' });
      const response = await request(app).post('/api/nodes/ping').send(signedPing());
      expect(response.status).toBe(400);
      expect(response.body.error).toBeDefined();
    });
  });
});
