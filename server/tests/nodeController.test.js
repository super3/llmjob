const request = require('supertest');
const express = require('express');
const nacl = require('tweetnacl');
const naclUtil = require('tweetnacl-util');
const routes = require('../src/routes');
const redisMock = require('redis-mock');

// Mock Redis
jest.mock('redis', () => require('redis-mock'));

// Mock Clerk
jest.mock('@clerk/clerk-sdk-node', () => ({
  clerkClient: {
    sessions: {
      verifySession: jest.fn().mockResolvedValue({ userId: 'test_user_123' })
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

describe('Node API Endpoints', () => {
  let app;
  let redisClient;
  let testKeypair;
  let testPublicKey;
  let testNodeId;

  beforeAll(() => {
    // Generate test keypair
    testKeypair = nacl.sign.keyPair();
    testPublicKey = naclUtil.encodeBase64(testKeypair.publicKey);
    
    // Calculate node ID (first 6 chars of SHA256 hash)
    const crypto = require('crypto');
    const hash = crypto.createHash('sha256').update(testPublicKey).digest('hex');
    testNodeId = hash.substring(0, 6);
  });

  beforeEach(async () => {
    // Create Express app
    app = express();
    app.use(express.json());
    
    // Create Redis client
    redisClient = redisMock.createClient();
    
    // Wrap Redis methods to be async
    const originalGet = redisClient.get.bind(redisClient);
    redisClient.get = (key) => new Promise((resolve) => {
      originalGet(key, (err, result) => resolve(result));
    });
    
    const originalSet = redisClient.set.bind(redisClient);
    redisClient.set = (key, value) => new Promise((resolve) => {
      originalSet(key, value, (err, result) => resolve(result));
    });
    
    const originalSetex = redisClient.setex.bind(redisClient);
    redisClient.setex = (key, ttl, value) => new Promise((resolve) => {
      originalSetex(key, ttl, value, (err, result) => resolve(result));
    });
    
    const originalSadd = redisClient.sadd.bind(redisClient);
    redisClient.sadd = (key, ...members) => new Promise((resolve) => {
      originalSadd(key, ...members, (err, result) => resolve(result));
    });
    
    const originalSmembers = redisClient.smembers.bind(redisClient);
    redisClient.smembers = (key) => new Promise((resolve) => {
      originalSmembers(key, (err, result) => resolve(result || []));
    });
    
    const originalKeys = redisClient.keys.bind(redisClient);
    redisClient.keys = (pattern) => new Promise((resolve) => {
      originalKeys(pattern, (err, result) => resolve(result || []));
    });
    
    const originalTtl = redisClient.ttl.bind(redisClient);
    redisClient.ttl = (key) => new Promise((resolve) => {
      originalTtl(key, (err, result) => resolve(result || -1));
    });
    
    app.locals.redis = redisClient;
    
    // Mount routes
    app.use('/api', routes);
    
    // Clear Redis
    await redisClient.flushall();
  });

  afterEach(async () => {
    await redisClient.quit();
  });

  describe('POST /api/nodes/claim', () => {
    it('should claim a node successfully', async () => {
      const response = await request(app)
        .post('/api/nodes/claim')
        .set('Authorization', 'Bearer test_token')
        .send({
          publicKey: testPublicKey,
          name: 'Test Node'
        });

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.nodeId).toBe(testNodeId);
      expect(response.body.message).toBe('Node claimed successfully');
    });

    it('should reject claim without auth token', async () => {
      const response = await request(app)
        .post('/api/nodes/claim')
        .send({
          publicKey: testPublicKey,
          name: 'Test Node'
        });

      expect(response.status).toBe(401);
      expect(response.body.error).toBe('No authorization token provided');
    });

    it('should reject claim with missing fields', async () => {
      const response = await request(app)
        .post('/api/nodes/claim')
        .set('Authorization', 'Bearer test_token')
        .send({
          publicKey: testPublicKey
        });

      expect(response.status).toBe(400);
      expect(response.body.error).toBe('Public key and name are required');
    });

    it('should prevent claiming already claimed node', async () => {
      // First claim
      await request(app)
        .post('/api/nodes/claim')
        .set('Authorization', 'Bearer test_token')
        .send({
          publicKey: testPublicKey,
          name: 'Test Node'
        });

      // Mock different user
      const { clerkClient } = require('@clerk/clerk-sdk-node');
      clerkClient.sessions.verifySession.mockResolvedValueOnce({ userId: 'different_user' });
      clerkClient.users.getUser.mockResolvedValueOnce({
        id: 'different_user',
        emailAddresses: [{ emailAddress: 'other@example.com' }],
        username: 'otheruser'
      });

      // Try to claim same node
      const response = await request(app)
        .post('/api/nodes/claim')
        .set('Authorization', 'Bearer test_token')
        .send({
          publicKey: testPublicKey,
          name: 'Test Node'
        });

      expect(response.status).toBe(400);
      expect(response.body.error).toBe('Node already claimed by another user');
    });
  });

  describe('POST /api/nodes/ping', () => {
    beforeEach(async () => {
      // Claim a node first
      await redisClient.set(`node:${testNodeId}`, JSON.stringify({
        nodeId: testNodeId,
        publicKey: testPublicKey,
        name: 'Test Node',
        userId: 'test_user_123',
        status: 'offline',
        isPublic: false,
        lastSeen: Date.now(),
        claimedAt: Date.now()
      }));
    });

    it('should update node status with valid signature', async () => {
      const timestamp = Date.now();
      const message = `${testNodeId}:${timestamp}`;
      const messageBytes = naclUtil.decodeUTF8(message);
      const signature = nacl.sign.detached(messageBytes, testKeypair.secretKey);
      const signatureBase64 = naclUtil.encodeBase64(signature);

      const response = await request(app)
        .post('/api/nodes/ping')
        .send({
          publicKey: testPublicKey,
          signature: signatureBase64,
          timestamp,
          nodeId: testNodeId
        });

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.status).toBe('online');
    });

    it('should reject ping with invalid signature', async () => {
      const timestamp = Date.now();
      
      const response = await request(app)
        .post('/api/nodes/ping')
        .send({
          publicKey: testPublicKey,
          signature: 'invalid_signature',
          timestamp,
          nodeId: testNodeId
        });

      expect(response.status).toBe(401);
      expect(response.body.error).toBe('Invalid signature format');
    });

    it('should reject ping with old timestamp', async () => {
      const timestamp = Date.now() - (10 * 60 * 1000); // 10 minutes ago
      const message = `${testNodeId}:${timestamp}`;
      const messageBytes = naclUtil.decodeUTF8(message);
      const signature = nacl.sign.detached(messageBytes, testKeypair.secretKey);
      const signatureBase64 = naclUtil.encodeBase64(signature);

      const response = await request(app)
        .post('/api/nodes/ping')
        .send({
          publicKey: testPublicKey,
          signature: signatureBase64,
          timestamp,
          nodeId: testNodeId
        });

      expect(response.status).toBe(401);
      expect(response.body.error).toBe('Timestamp too old or too far in future');
    });

    it('should reject ping with missing fields', async () => {
      const response = await request(app)
        .post('/api/nodes/ping')
        .send({
          publicKey: testPublicKey,
          nodeId: testNodeId
        });

      expect(response.status).toBe(400);
      expect(response.body.error).toBe('Missing required fields');
    });
  });

  describe('GET /api/nodes', () => {
    beforeEach(async () => {
      // Add some test nodes
      await redisClient.set(`node:node1`, JSON.stringify({
        nodeId: 'node1',
        publicKey: 'key1',
        name: 'Node 1',
        userId: 'test_user_123',
        status: 'online',
        isPublic: false,
        lastSeen: Date.now()
      }));
      
      await redisClient.set(`node:node2`, JSON.stringify({
        nodeId: 'node2',
        publicKey: 'key2',
        name: 'Node 2',
        userId: 'test_user_123',
        status: 'offline',
        isPublic: true,
        lastSeen: Date.now() - (20 * 60 * 1000) // 20 minutes ago
      }));
      
      await redisClient.sadd('user_nodes:test_user_123', 'node1', 'node2');
    });

    it('should return user nodes with auth', async () => {
      const response = await request(app)
        .get('/api/nodes')
        .set('Authorization', 'Bearer test_token');

      expect(response.status).toBe(200);
      expect(response.body.nodes).toHaveLength(2);
      expect(response.body.nodes[0].nodeId).toBe('node1');
      expect(response.body.nodes[0].status).toBe('online');
      expect(response.body.nodes[1].nodeId).toBe('node2');
      expect(response.body.nodes[1].status).toBe('offline');
    });

    it('should reject without auth token', async () => {
      const response = await request(app)
        .get('/api/nodes');

      expect(response.status).toBe(401);
      expect(response.body.error).toBe('No authorization token provided');
    });
  });

  describe('GET /api/nodes/public', () => {
    beforeEach(async () => {
      // Add test nodes
      await redisClient.set(`node:public1`, JSON.stringify({
        nodeId: 'public1',
        publicKey: 'key1',
        name: 'Public Node 1',
        userId: 'user1',
        status: 'online',
        isPublic: true,
        lastSeen: Date.now()
      }));
      
      await redisClient.set(`node:private1`, JSON.stringify({
        nodeId: 'private1',
        publicKey: 'key2',
        name: 'Private Node',
        userId: 'user2',
        status: 'online',
        isPublic: false,
        lastSeen: Date.now()
      }));
      
      await redisClient.set(`node:public2`, JSON.stringify({
        nodeId: 'public2',
        publicKey: 'key3',
        name: 'Public Node 2',
        userId: 'user3',
        status: 'online',
        isPublic: true,
        lastSeen: Date.now()
      }));
    });

    it('should return only public nodes without auth', async () => {
      const response = await request(app)
        .get('/api/nodes/public');

      expect(response.status).toBe(200);
      expect(response.body.nodes).toHaveLength(2);
      expect(response.body.nodes.every(n => n.nodeId.startsWith('public'))).toBe(true);
    });
  });

  describe('PUT /api/nodes/:id/visibility', () => {
    beforeEach(async () => {
      // Add a test node owned by test user
      await redisClient.set(`node:${testNodeId}`, JSON.stringify({
        nodeId: testNodeId,
        publicKey: testPublicKey,
        name: 'Test Node',
        userId: 'test_user_123',
        status: 'online',
        isPublic: false,
        lastSeen: Date.now()
      }));
    });

    it('should update node visibility', async () => {
      const response = await request(app)
        .put(`/api/nodes/${testNodeId}/visibility`)
        .set('Authorization', 'Bearer test_token')
        .send({ isPublic: true });

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.isPublic).toBe(true);
      expect(response.body.message).toBe('Node visibility updated to public');
    });

    it('should reject update for non-owned node', async () => {
      // Add node owned by different user
      await redisClient.set(`node:other`, JSON.stringify({
        nodeId: 'other',
        publicKey: 'otherkey',
        name: 'Other Node',
        userId: 'different_user',
        status: 'online',
        isPublic: false,
        lastSeen: Date.now()
      }));

      const response = await request(app)
        .put('/api/nodes/other/visibility')
        .set('Authorization', 'Bearer test_token')
        .send({ isPublic: true });

      expect(response.status).toBe(403);
      expect(response.body.error).toBe('Unauthorized: You do not own this node');
    });

    it('should reject with invalid isPublic value', async () => {
      const response = await request(app)
        .put(`/api/nodes/${testNodeId}/visibility`)
        .set('Authorization', 'Bearer test_token')
        .send({ isPublic: 'not_boolean' });

      expect(response.status).toBe(400);
      expect(response.body.error).toBe('isPublic must be a boolean');
    });

    it('should return 404 for non-existent node', async () => {
      const response = await request(app)
        .put('/api/nodes/nonexistent/visibility')
        .set('Authorization', 'Bearer test_token')
        .send({ isPublic: true });

      expect(response.status).toBe(404);
      expect(response.body.error).toBe('Node not found');
    });
  });
});