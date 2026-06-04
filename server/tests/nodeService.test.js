const NodeService = require('../src/services/nodeService');
const { createCamelClient } = require('./helpers/camelRedis');

describe('Node Service', () => {
  let redisClient;
  let nodeService;

  beforeEach(async () => {
    redisClient = createCamelClient();
    nodeService = new NodeService(redisClient);
    await redisClient.flushall();
  });

  afterEach(async () => {
    await redisClient.quit();
  });

  describe('generateNodeFingerprint', () => {
    it('should generate consistent 6-character fingerprint', () => {
      const publicKey = 'test_public_key_123';
      const fingerprint1 = NodeService.generateNodeFingerprint(publicKey);
      const fingerprint2 = NodeService.generateNodeFingerprint(publicKey);

      expect(fingerprint1).toBe(fingerprint2);
      expect(fingerprint1).toHaveLength(6);
      expect(fingerprint1).toMatch(/^[a-f0-9]{6}$/);
    });

    it('should generate different fingerprints for different keys', () => {
      const fingerprint1 = NodeService.generateNodeFingerprint('key1');
      const fingerprint2 = NodeService.generateNodeFingerprint('key2');

      expect(fingerprint1).not.toBe(fingerprint2);
    });
  });

  describe('claimNode', () => {
    it('should successfully claim a new node', async () => {
      const result = await nodeService.claimNode('test_public_key', 'Test Node', 'user123');

      expect(result.success).toBe(true);
      expect(result.nodeId).toHaveLength(6);
      expect(result.message).toBe('Node claimed successfully');

      // Verify node was stored
      const nodeData = await redisClient.get(`node:${result.nodeId}`);
      const node = JSON.parse(nodeData);
      expect(node.name).toBe('Test Node');
      expect(node.userId).toBe('user123');
      expect(node.isPublic).toBe(false);
      expect(node.status).toBe('online'); // Node should be online when claimed
    });

    it('should prevent claiming node already owned by another user', async () => {
      const publicKey = 'test_public_key';

      // First claim
      await nodeService.claimNode(publicKey, 'Node 1', 'user1');

      // Try to claim with different user
      const result = await nodeService.claimNode(publicKey, 'Node 2', 'user2');

      expect(result.error).toBe('Node already claimed by another user');
    });

    it('should allow reclaiming by same user', async () => {
      const publicKey = 'test_public_key';
      const userId = 'user123';

      // First claim
      await nodeService.claimNode(publicKey, 'Node 1', userId);

      // Reclaim with same user
      const result = await nodeService.claimNode(publicKey, 'Updated Node', userId);

      expect(result.success).toBe(true);
    });
  });

  describe('updateNodeStatus', () => {
    let nodeId;
    const publicKey = 'test_public_key';

    beforeEach(async () => {
      const result = await nodeService.claimNode(publicKey, 'Test Node', 'user123');
      nodeId = result.nodeId;
    });

    it('should update node status to online', async () => {
      const result = await nodeService.updateNodeStatus(nodeId, publicKey);

      expect(result.success).toBe(true);
      expect(result.status).toBe('online');

      // Verify node was updated
      const nodeData = await redisClient.get(`node:${nodeId}`);
      const node = JSON.parse(nodeData);
      expect(node.status).toBe('online');
    });

    it('should update status with capabilities, activeJobs and maxConcurrentJobs', async () => {
      const result = await nodeService.updateNodeStatus(nodeId, publicKey, {
        capabilities: { gpu: true },
        activeJobs: 2,
        maxConcurrentJobs: 4
      });

      expect(result.success).toBe(true);

      const node = JSON.parse(await redisClient.get(`node:${nodeId}`));
      expect(node.capabilities).toEqual({ gpu: true });
      expect(node.activeJobs).toBe(2);
      expect(node.maxConcurrentJobs).toBe(4);
    });

    it('should reject update with wrong public key', async () => {
      const result = await nodeService.updateNodeStatus(nodeId, 'wrong_key');

      expect(result.error).toBe('Public key mismatch');
    });

    it('should reject update for non-existent node', async () => {
      const result = await nodeService.updateNodeStatus('nonexistent', publicKey);

      expect(result.error).toBe('Node not found. Please claim the node first.');
    });
  });

  describe('getUserNodes', () => {
    beforeEach(async () => {
      // Create nodes for user
      await nodeService.claimNode('key1', 'Node 1', 'user123');
      await nodeService.claimNode('key2', 'Node 2', 'user123');
      await nodeService.claimNode('key3', 'Node 3', 'otheruser');
    });

    it('should return only nodes belonging to user', async () => {
      const nodes = await nodeService.getUserNodes('user123');

      expect(nodes).toHaveLength(2);
      expect(nodes[0].name).toBe('Node 1');
      expect(nodes[1].name).toBe('Node 2');
    });

    it('should return empty array for user with no nodes', async () => {
      const nodes = await nodeService.getUserNodes('nouser');

      expect(nodes).toEqual([]);
    });

    it('should mark old nodes as offline', async () => {
      // Manually set an old node
      const nodeId = NodeService.generateNodeFingerprint('oldkey');
      await redisClient.set(`node:${nodeId}`, JSON.stringify({
        nodeId,
        name: 'Old Node',
        userId: 'user456',
        status: 'online',
        isPublic: false,
        lastSeen: Date.now() - (20 * 60 * 1000) // 20 minutes ago - should be offline
      }));
      await redisClient.sAdd('user_nodes:user456', nodeId);

      const nodes = await nodeService.getUserNodes('user456');

      expect(nodes[0].status).toBe('offline');
    });
  });

  describe('getPublicNodes', () => {
    beforeEach(async () => {
      // Create mix of public and private nodes
      const node1 = await nodeService.claimNode('key1', 'Public 1', 'user1');
      const node2 = await nodeService.claimNode('key2', 'Private', 'user2');
      const node3 = await nodeService.claimNode('key3', 'Public 2', 'user3');

      // Make some public
      await nodeService.updateNodeVisibility(node1.nodeId, 'user1', true);
      await nodeService.updateNodeVisibility(node3.nodeId, 'user3', true);
    });

    it('should return only public nodes', async () => {
      const result = await nodeService.getPublicNodes();

      expect(result.nodes).toHaveLength(2);
      expect(result.nodes.every(n => n.name.includes('Public'))).toBe(true);
      expect(result.totalOnline).toBe(3); // All nodes are online when claimed
    });
  });

  describe('updateNodeVisibility', () => {
    let nodeId;
    const userId = 'user123';

    beforeEach(async () => {
      const result = await nodeService.claimNode('test_key', 'Test Node', userId);
      nodeId = result.nodeId;
    });

    it('should update node visibility to public', async () => {
      const result = await nodeService.updateNodeVisibility(nodeId, userId, true);

      expect(result.success).toBe(true);
      expect(result.isPublic).toBe(true);
      expect(result.message).toMatch(/public/);

      // Verify it appears in public nodes
      const publicResult = await nodeService.getPublicNodes();
      expect(publicResult.nodes.some(n => n.nodeId === nodeId)).toBe(true);
    });

    it('should update node visibility to private', async () => {
      // First make it public
      await nodeService.updateNodeVisibility(nodeId, userId, true);

      // Then make it private
      const result = await nodeService.updateNodeVisibility(nodeId, userId, false);

      expect(result.success).toBe(true);
      expect(result.isPublic).toBe(false);
      expect(result.message).toMatch(/private/);

      // Verify it doesn't appear in public nodes
      const publicResult = await nodeService.getPublicNodes();
      expect(publicResult.nodes.some(n => n.nodeId === nodeId)).toBe(false);
    });

    it('should reject update from non-owner', async () => {
      const result = await nodeService.updateNodeVisibility(nodeId, 'otheruser', true);

      expect(result.error).toBe('Unauthorized: You do not own this node');
      expect(result.status).toBe(403);
    });

    it('should return 404 for non-existent node', async () => {
      const result = await nodeService.updateNodeVisibility('nonexistent', userId, true);

      expect(result.error).toBe('Node not found');
      expect(result.status).toBe(404);
    });
  });

  describe('checkNodeStatuses', () => {
    it('should log node status counts', async () => {
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();

      // Add some nodes
      await nodeService.claimNode('key1', 'Node 1', 'user1');
      await nodeService.claimNode('key2', 'Node 2', 'user2');

      await nodeService.checkNodeStatuses();

      expect(consoleSpy).toHaveBeenCalled();
      expect(consoleSpy.mock.calls[0][0]).toContain('Node status check:');

      consoleSpy.mockRestore();
    });

    it('should handle empty node list', async () => {
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();

      await nodeService.checkNodeStatuses();

      expect(consoleSpy).toHaveBeenCalledWith('Node status check: 0 online, 0 offline');

      consoleSpy.mockRestore();
    });

    it('should count online nodes with positive TTL', async () => {
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();

      // Add a node and update its status to online (with TTL)
      const result = await nodeService.claimNode('key1', 'Node 1', 'user1');
      await nodeService.updateNodeStatus(result.nodeId, 'key1');

      await nodeService.checkNodeStatuses();

      expect(consoleSpy).toHaveBeenCalled();
      const logMessage = consoleSpy.mock.calls[0][0];
      expect(logMessage).toContain('1 online');

      consoleSpy.mockRestore();
    });

    it('should count offline nodes whose key has no TTL', async () => {
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();

      // A node key stored without a TTL counts as offline during the sweep.
      await redisClient.set('node:no-ttl', JSON.stringify({ nodeId: 'no-ttl' }));

      await nodeService.checkNodeStatuses();

      expect(consoleSpy.mock.calls[0][0]).toContain('0 online, 1 offline');

      consoleSpy.mockRestore();
    });
  });

  describe('getPublicNodes edge cases', () => {
    it('should handle empty nodes', async () => {
      const result = await nodeService.getPublicNodes();
      expect(result).toEqual({ nodes: [], totalOnline: 0 });
    });

    it('should handle null data from redis', async () => {
      // Mock Redis to return keys but null data
      redisClient.keys = jest.fn().mockResolvedValue(['node:test']);
      redisClient.get = jest.fn().mockResolvedValue(null);

      const result = await nodeService.getPublicNodes();
      expect(result).toEqual({ nodes: [], totalOnline: 0 });
    });

    it('should mark old public nodes as offline', async () => {
      // Add a public node with old lastSeen
      const oldTime = Date.now() - (20 * 60 * 1000); // 20 minutes ago - beyond offline threshold
      await redisClient.set('node:old', JSON.stringify({
        nodeId: 'old',
        publicKey: 'key_old',
        name: 'Old Node',
        userId: 'user1',
        status: 'online',
        isPublic: true,
        lastSeen: oldTime
      }));

      const result = await nodeService.getPublicNodes();
      expect(result.nodes).toHaveLength(1);
      expect(result.nodes[0].status).toBe('offline');
      expect(result.totalOnline).toBe(0); // Old node is offline
    });
  });

  describe('getUserNodes edge cases', () => {
    it('should handle null data from redis', async () => {
      // Add node ID to user's list but no actual node data
      await redisClient.sAdd('user_nodes:user1', 'nonexistent');

      const nodes = await nodeService.getUserNodes('user1');
      expect(nodes).toEqual([]);
    });
  });

  describe('updateNodeVisibility edge cases', () => {
    it('should handle node with expired TTL', async () => {
      // Setup a node
      await nodeService.claimNode('key1', 'Node 1', 'user1');
      const nodeId = NodeService.generateNodeFingerprint('key1');

      // Mock TTL to return -1 (expired)
      redisClient.ttl = jest.fn().mockResolvedValue(-1);

      const result = await nodeService.updateNodeVisibility(nodeId, 'user1', true);

      expect(result.success).toBe(true);
      expect(result.isPublic).toBe(true);
    });

    it('should handle node with positive TTL', async () => {
      // Setup a node with online status (has TTL)
      const result = await nodeService.claimNode('key2', 'Node 2', 'user2');
      await nodeService.updateNodeStatus(result.nodeId, 'key2');

      // Mock TTL to return positive value
      redisClient.ttl = jest.fn().mockResolvedValue(300); // 5 minutes

      const updateResult = await nodeService.updateNodeVisibility(result.nodeId, 'user2', true);

      expect(updateResult.success).toBe(true);
      expect(updateResult.isPublic).toBe(true);
    });
  });

  describe('claimNode edge cases', () => {
    it('should allow claiming unclaimed node', async () => {
      // Create a node without userId (unclaimed)
      const nodeId = NodeService.generateNodeFingerprint('key1');
      await redisClient.set(`node:${nodeId}`, JSON.stringify({
        nodeId,
        publicKey: 'key1',
        name: 'Unclaimed Node',
        // No userId
        status: 'offline',
        isPublic: false,
        lastSeen: Date.now()
      }));

      const result = await nodeService.claimNode('key1', 'Claimed Node', 'user1');

      expect(result.success).toBe(true);
      expect(result.nodeId).toBe(nodeId);
    });
  });

  describe('getNode', () => {
    it('should retrieve node using getNode method', async () => {
      // Create a node
      const result = await nodeService.claimNode('key1', 'Node 1', 'user1');

      // Get node using the getNode method
      const node = await nodeService.getNode(result.nodeId);

      expect(node).toBeDefined();
      expect(node.nodeId).toBe(result.nodeId);
      expect(node.publicKey).toBe('key1');
      expect(node.name).toBe('Node 1');
    });

    it('should return null for non-existent node', async () => {
      const node = await nodeService.getNode('non-existent');
      expect(node).toBeNull();
    });

    it('should handle getNode with malformed data', async () => {
      const nodeId = 'test-node';
      // Set invalid JSON data
      await redisClient.set(`node:${nodeId}`, 'invalid-json');

      // Should throw when trying to parse invalid JSON
      await expect(nodeService.getNode(nodeId)).rejects.toThrow();
    });
  });

  describe('Additional edge cases', () => {
    it('should handle updateNodeStatus with missing node gracefully', async () => {
      const result = await nodeService.updateNodeStatus('non-existent', 'key1');
      expect(result.error).toBe('Node not found. Please claim the node first.');
    });

    it('should handle getPublicNodes with malformed node data', async () => {
      await redisClient.set('node:broken-node', 'invalid-json');

      // Should throw an error when trying to parse invalid JSON
      await expect(nodeService.getPublicNodes()).rejects.toThrow();
    });

    it('should handle getUserNodes when user has no nodes', async () => {
      const nodes = await nodeService.getUserNodes('user-with-no-nodes');
      expect(nodes).toEqual([]);
    });

    it('should handle getUserNodes with deleted node references', async () => {
      // Add node reference for user but don't create actual node data
      await redisClient.sAdd('user_nodes:user1', 'deleted-node');

      const nodes = await nodeService.getUserNodes('user1');
      // Should return empty array as the node data doesn't exist
      expect(nodes).toEqual([]);
    });

    it('should handle concurrent node updates', async () => {
      const result = await nodeService.claimNode('key1', 'Node', 'user1');

      // Simulate concurrent updates
      await Promise.all([
        nodeService.updateNodeStatus(result.nodeId, 'key1'),
        nodeService.updateNodeVisibility(result.nodeId, 'user1', true)
      ]);

      const node = await nodeService.getNode(result.nodeId);
      expect(node).toBeDefined();
      expect(node.status).toBe('online');
    });

    it('should handle node visibility toggle for offline node', async () => {
      const result = await nodeService.claimNode('key1', 'Node', 'user1');

      // Try to make offline node public
      const visibilityResult = await nodeService.updateNodeVisibility(result.nodeId, 'user1', true);

      expect(visibilityResult.success).toBe(true);
      expect(visibilityResult.isPublic).toBe(true);
    });

    it('should reject visibility update from wrong user', async () => {
      const result = await nodeService.claimNode('key1', 'Node', 'user1');

      // Try to update visibility from different user
      const visibilityResult = await nodeService.updateNodeVisibility(result.nodeId, 'wrong-user', true);

      expect(visibilityResult.error).toBe('Unauthorized: You do not own this node');
    });
  });
});
