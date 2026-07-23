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
    const crypto = require('crypto');
    testNodeId = crypto.createHash('sha256').update(testPublicKey).digest('hex').substring(0, 6);
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

  describe('POST /api/nodes/claim', () => {
    it('should claim a node successfully', async () => {
      const response = await request(app)
        .post('/api/nodes/claim')
        .set('Authorization', authHeader())
        .send({ publicKey: testPublicKey, name: 'Test Node' });

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.nodeId).toBe(testNodeId);
      expect(response.body.message).toBe('Node claimed successfully');
    });

    it('should reject claim without auth token', async () => {
      const response = await request(app)
        .post('/api/nodes/claim')
        .send({ publicKey: testPublicKey, name: 'Test Node' });

      expect(response.status).toBe(401);
      expect(response.body.error).toBe('No authorization token provided');
    });

    it('should reject claim with missing fields', async () => {
      const response = await request(app)
        .post('/api/nodes/claim')
        .set('Authorization', authHeader())
        .send({ publicKey: testPublicKey });

      expect(response.status).toBe(400);
      expect(response.body.error).toBe('Public key and name are required');
    });

    it('should prevent claiming already claimed node', async () => {
      await request(app)
        .post('/api/nodes/claim')
        .set('Authorization', authHeader())
        .send({ publicKey: testPublicKey, name: 'Test Node' });

      const { clerkClient } = require('@clerk/clerk-sdk-node');
      clerkClient.users.getUser.mockResolvedValueOnce({
        id: 'different_user',
        emailAddresses: [{ emailAddress: 'other@example.com' }],
        username: 'otheruser'
      });

      const response = await request(app)
        .post('/api/nodes/claim')
        .set('Authorization', authHeader('different_user'))
        .send({ publicKey: testPublicKey, name: 'Test Node' });

      expect(response.status).toBe(400);
      expect(response.body.error).toBe('Node already claimed by another user');
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

  describe('GET /api/nodes/public', () => {
    beforeEach(async () => {
      await seedNode({ node_id: 'public1', name: 'Public Node 1', user_id: 'user1', is_public: true });
      await seedNode({ node_id: 'private1', name: 'Private Node', user_id: 'user2', is_public: false });
      await seedNode({ node_id: 'public2', name: 'Public Node 2', user_id: 'user3', is_public: true });
    });

    it('should return only public nodes without auth', async () => {
      const response = await request(app).get('/api/nodes/public');
      expect(response.status).toBe(200);
      expect(response.body.nodes).toHaveLength(2);
      expect(response.body.nodes.every((n) => n.nodeId.startsWith('public'))).toBe(true);
    });
  });

  describe('PUT /api/nodes/:id/visibility', () => {
    beforeEach(async () => {
      await seedNode({ node_id: testNodeId, public_key: testPublicKey, name: 'Test Node', user_id: 'test_user_123' });
    });

    it('should update node visibility', async () => {
      const response = await request(app)
        .put(`/api/nodes/${testNodeId}/visibility`)
        .set('Authorization', authHeader())
        .send({ isPublic: true });
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.isPublic).toBe(true);
      expect(response.body.message).toBe('Node visibility updated to public');
    });

    it('should reject update for non-owned node', async () => {
      await seedNode({ node_id: 'other', public_key: 'otherkey', name: 'Other Node', user_id: 'different_user' });
      const response = await request(app)
        .put('/api/nodes/other/visibility')
        .set('Authorization', authHeader())
        .send({ isPublic: true });
      expect(response.status).toBe(403);
      expect(response.body.error).toBe('Unauthorized: You do not own this node');
    });

    it('should reject with invalid isPublic value', async () => {
      const response = await request(app)
        .put(`/api/nodes/${testNodeId}/visibility`)
        .set('Authorization', authHeader())
        .send({ isPublic: 'not_boolean' });
      expect(response.status).toBe(400);
      expect(response.body.error).toBe('isPublic must be a boolean');
    });

    it('should return 404 for non-existent node', async () => {
      const response = await request(app)
        .put('/api/nodes/nonexistent/visibility')
        .set('Authorization', authHeader())
        .send({ isPublic: true });
      expect(response.status).toBe(404);
      expect(response.body.error).toBe('Node not found');
    });
  });

  describe('Error handling', () => {
    it('should handle database errors in claimNode', async () => {
      db.query = jest.fn().mockRejectedValue(new Error('Database error'));
      const response = await request(app)
        .post('/api/nodes/claim')
        .set('Authorization', authHeader())
        .send({ publicKey: testPublicKey, name: 'Test Node' });
      expect(response.status).toBe(500);
      expect(response.body.error).toBe('Failed to claim node');
    });

    it('should handle database errors in pingNode', async () => {
      await seedNode({ node_id: testNodeId, public_key: testPublicKey, name: 'Test Node', user_id: 'test_user_123', status: 'offline' });
      db.query = jest.fn().mockRejectedValue(new Error('Database error'));
      const response = await request(app).post('/api/nodes/ping').send(signedPing());
      expect(response.status).toBe(500);
      expect(response.body.error).toBe('Failed to update node status');
    });

    it('should handle database errors in getUserNodes', async () => {
      db.query = jest.fn().mockRejectedValue(new Error('Database error'));
      const response = await request(app).get('/api/nodes').set('Authorization', authHeader());
      expect(response.status).toBe(500);
      expect(response.body.error).toBe('Failed to get nodes');
    });

    it('should handle database errors in getPublicNodes', async () => {
      db.query = jest.fn().mockRejectedValue(new Error('Database error'));
      const response = await request(app).get('/api/nodes/public');
      expect(response.status).toBe(500);
      expect(response.body.error).toBe('Failed to get public nodes');
    });

    it('should handle database errors in updateNodeVisibility', async () => {
      const spy = jest.spyOn(NodeService.prototype, 'updateNodeVisibility')
        .mockRejectedValue(new Error('Database error'));
      const response = await request(app)
        .put(`/api/nodes/${testNodeId}/visibility`)
        .set('Authorization', authHeader())
        .send({ isPublic: true });
      expect(response.status).toBe(500);
      expect(response.body.error).toBe('Failed to update node visibility');
      spy.mockRestore();
    });

    it('should handle service errors in pingNode', async () => {
      // Node exists but with no stored public key -> "Public key mismatch" (400).
      await seedNode({ node_id: testNodeId, public_key: null, name: 'Test Node', user_id: 'test_user_123', status: 'offline' });
      const response = await request(app).post('/api/nodes/ping').send(signedPing());
      expect(response.status).toBe(400);
      expect(response.body.error).toBeDefined();
    });

    it('should handle service error without status in updateNodeVisibility', async () => {
      const spy = jest.spyOn(NodeService.prototype, 'updateNodeVisibility')
        .mockResolvedValue({ error: 'Some error without status field' });
      const response = await request(app)
        .put(`/api/nodes/${testNodeId}/visibility`)
        .set('Authorization', authHeader())
        .send({ isPublic: true });
      expect(response.status).toBe(400);
      expect(response.body.error).toBe('Some error without status field');
      spy.mockRestore();
    });
  });
});
