const NodeService = require('../src/services/nodeService');
const { createTestDb } = require('./helpers/pgmem');

describe('NodeService', () => {
  let db;
  let service;

  beforeEach(async () => {
    db = await createTestDb();
    service = new NodeService(db);
  });

  afterEach(async () => {
    if (db.end) await db.end();
  });

  const setLastSeen = (nodeId, ms) =>
    db.query('UPDATE nodes SET last_seen = $1 WHERE node_id = $2', [ms, nodeId]);

  describe('generateNodeFingerprint', () => {
    it('is a 6-char hex prefix of the key hash', () => {
      expect(NodeService.generateNodeFingerprint('somekey')).toMatch(/^[0-9a-f]{6}$/);
    });
  });

  describe('claimNode', () => {
    it('claims a new node online', async () => {
      const res = await service.claimNode('key1', 'Node 1', 'user1');
      expect(res.success).toBe(true);
      const node = await service.getNode(res.nodeId);
      expect(node).toMatchObject({ name: 'Node 1', userId: 'user1', status: 'online', isPublic: false });
    });

    it('lets the same user re-claim (overwriting name/visibility)', async () => {
      const { nodeId } = await service.claimNode('key1', 'Old', 'user1');
      await service.updateNodeVisibility(nodeId, 'user1', true);
      await service.claimNode('key1', 'New', 'user1');
      const node = await service.getNode(nodeId);
      expect(node.name).toBe('New');
      expect(node.isPublic).toBe(false);
    });

    it('rejects a claim by a different user', async () => {
      await service.claimNode('key1', 'Node 1', 'user1');
      const res = await service.claimNode('key1', 'Mine', 'user2');
      expect(res).toEqual({ error: 'Node already claimed by another user' });
    });
  });

  describe('updateNodeStatus', () => {
    it('errors when the node was never claimed', async () => {
      const res = await service.updateNodeStatus('nope', 'key', {});
      expect(res).toEqual({ error: 'Node not found. Please claim the node first.' });
    });

    it('errors on a public key mismatch', async () => {
      const { nodeId } = await service.claimNode('key1', 'n', 'user1');
      const res = await service.updateNodeStatus(nodeId, 'wrong', {});
      expect(res.error).toBe('Public key mismatch');
    });

    it('updates capabilities, activeJobs and maxConcurrentJobs', async () => {
      const { nodeId } = await service.claimNode('key1', 'n', 'user1');
      const res = await service.updateNodeStatus(nodeId, 'key1', {
        capabilities: { gpu: true }, activeJobs: 2, maxConcurrentJobs: 4
      });
      expect(res.success).toBe(true);
      const node = await service.getNode(nodeId);
      expect(node.capabilities).toEqual({ gpu: true });
      expect(node.activeJobs).toBe(2);
      expect(node.maxConcurrentJobs).toBe(4);
    });

    it('preserves existing values when fields are omitted', async () => {
      const { nodeId } = await service.claimNode('key1', 'n', 'user1');
      await service.updateNodeStatus(nodeId, 'key1', { capabilities: { gpu: true } });
      await service.updateNodeStatus(nodeId, 'key1', {});
      const node = await service.getNode(nodeId);
      expect(node.capabilities).toEqual({ gpu: true });
    });

    it('works when called with no additional data', async () => {
      const { nodeId } = await service.claimNode('key1', 'n', 'user1');
      const res = await service.updateNodeStatus(nodeId, 'key1'); // default {} param
      expect(res).toEqual({ success: true, status: 'online', message: 'Node status updated' });
    });
  });

  describe('getUserNodes', () => {
    beforeEach(async () => {
      await service.claimNode('key1', 'Node 1', 'user123');
      await service.claimNode('key2', 'Node 2', 'user123');
      await service.claimNode('key3', 'Node 3', 'otheruser');
    });

    it('returns only nodes belonging to the user, in claim order', async () => {
      const nodes = await service.getUserNodes('user123');
      expect(nodes).toHaveLength(2);
      expect(nodes[0].name).toBe('Node 1');
      expect(nodes[1].name).toBe('Node 2');
    });

    it('returns an empty array for a user with no nodes', async () => {
      expect(await service.getUserNodes('nobody')).toEqual([]);
    });

    it('marks stale nodes offline', async () => {
      const id = NodeService.generateNodeFingerprint('key1');
      await setLastSeen(id, Date.now() - 20 * 60 * 1000);
      const nodes = await service.getUserNodes('user123');
      expect(nodes.find((x) => x.name === 'Node 1').status).toBe('offline');
    });
  });

  describe('getPublicNodes', () => {
    it('returns empty results when there are no nodes', async () => {
      expect(await service.getPublicNodes()).toEqual({ nodes: [], totalOnline: 0 });
    });

    it('lists public nodes and counts all online nodes', async () => {
      const a = await service.claimNode('pk-a', 'Public', 'user1');
      await service.updateNodeVisibility(a.nodeId, 'user1', true);
      await service.claimNode('pk-b', 'Private', 'user1');

      const res = await service.getPublicNodes();
      expect(res.totalOnline).toBe(2);
      expect(res.nodes).toHaveLength(1);
      expect(res.nodes[0].name).toBe('Public');
    });

    it('shows a stale public node as offline and excludes it from the online count', async () => {
      const a = await service.claimNode('pk-a', 'Public', 'user1');
      await service.updateNodeVisibility(a.nodeId, 'user1', true);
      await setLastSeen(a.nodeId, Date.now() - 20 * 60 * 1000);

      const res = await service.getPublicNodes();
      expect(res.totalOnline).toBe(0);
      expect(res.nodes[0].status).toBe('offline');
    });
  });

  describe('updateNodeVisibility', () => {
    it('returns 404 for an unknown node', async () => {
      expect(await service.updateNodeVisibility('nope', 'user1', true))
        .toEqual({ error: 'Node not found', status: 404 });
    });

    it('returns 403 when the user does not own the node', async () => {
      const { nodeId } = await service.claimNode('key1', 'n', 'user1');
      const res = await service.updateNodeVisibility(nodeId, 'user2', true);
      expect(res.status).toBe(403);
    });

    it('toggles visibility for the owner', async () => {
      const { nodeId } = await service.claimNode('key1', 'n', 'user1');
      const on = await service.updateNodeVisibility(nodeId, 'user1', true);
      expect(on).toMatchObject({ success: true, isPublic: true });
      const off = await service.updateNodeVisibility(nodeId, 'user1', false);
      expect(off.isPublic).toBe(false);
    });
  });

  describe('checkNodeStatuses', () => {
    it('prunes week-old nodes and logs an online/offline summary', async () => {
      const spy = jest.spyOn(console, 'log').mockImplementation(() => {});
      const fresh = await service.claimNode('fresh', 'fresh', 'u');
      const stale = await service.claimNode('stale', 'stale', 'u');
      const old = await service.claimNode('old', 'old', 'u');
      await setLastSeen(stale.nodeId, Date.now() - 20 * 60 * 1000);
      await setLastSeen(old.nodeId, Date.now() - 8 * 24 * 60 * 60 * 1000);

      await service.checkNodeStatuses();

      expect(await service.getNode(old.nodeId)).toBeNull();
      expect(await service.getNode(fresh.nodeId)).not.toBeNull();
      expect(spy).toHaveBeenCalledWith('Node status check: 1 online, 1 offline');
      spy.mockRestore();
    });

    it('handles an empty table', async () => {
      const spy = jest.spyOn(console, 'log').mockImplementation(() => {});
      await service.checkNodeStatuses();
      expect(spy).toHaveBeenCalledWith('Node status check: 0 online, 0 offline');
      spy.mockRestore();
    });
  });

  describe('getNode', () => {
    it('returns null for an unknown node', async () => {
      expect(await service.getNode('nope')).toBeNull();
    });
  });
});
