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

  describe('getNodeStatus', () => {
    it('reports a freshly claimed node as existing and online', async () => {
      const { nodeId } = await service.claimNode('key1', 'Rig', 'user1');
      expect(await service.getNodeStatus(nodeId)).toEqual({ exists: true, online: true });
    });

    it('reports a node past the offline threshold as existing but offline', async () => {
      const { nodeId } = await service.claimNode('key1', 'Rig', 'user1');
      await setLastSeen(nodeId, Date.now() - 20 * 60 * 1000); // 20 min ago (> 15 min)
      expect(await service.getNodeStatus(nodeId)).toEqual({ exists: true, online: false });
    });

    it('reports an unknown node id as neither existing nor online', async () => {
      expect(await service.getNodeStatus('nope')).toEqual({ exists: false, online: false });
    });
  });

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

    it('refuses to overwrite a node registered under a different key (collision takeover)', async () => {
      // Simulate the attack: a victim node already occupies this fingerprint with
      // its own key; the attacker presents a different key that grinds to the same
      // id. The claim must not overwrite the victim's registered key.
      const attackerKey = 'attacker-key';
      const nodeId = NodeService.generateNodeFingerprint(attackerKey);
      await db.query(
        `INSERT INTO nodes (node_id, public_key, name, user_id, status, is_public, last_seen, claimed_at)
         VALUES ($1, $2, 'victim', NULL, 'online', false, $3, $3)`,
        [nodeId, 'victim-key', Date.now()]
      );

      const res = await service.claimNode(attackerKey, 'pwned', 'attacker');
      expect(res).toEqual({ error: 'Node key mismatch' });
      const node = await service.getNode(nodeId);
      expect(node.publicKey).toBe('victim-key'); // untouched
      expect(node.userId).toBeNull();            // not hijacked
    });
  });

  describe('registerNode', () => {
    it('creates an unclaimed, online node with no owner', async () => {
      const res = await service.registerNode('key1', 'Rig');
      expect(res).toMatchObject({ success: true, claimed: false });
      const node = await service.getNode(res.nodeId);
      expect(node).toMatchObject({ name: 'Rig', userId: null, status: 'online', isPublic: false });
    });

    it('derives the same nodeId as a claim of the same key, and names it when none is given', async () => {
      const res = await service.registerNode('key1');
      expect(res.nodeId).toBe(NodeService.generateNodeFingerprint('key1'));
      expect((await service.getNode(res.nodeId)).name).toBe(`Node-${res.nodeId}`);
    });

    it('is idempotent — re-registering just marks it online again', async () => {
      const { nodeId } = await service.registerNode('key1', 'Rig');
      await setLastSeen(nodeId, 1);
      const again = await service.registerNode('key1', 'Rig');
      expect(again).toMatchObject({ success: true, nodeId, claimed: false });
      expect((await service.getNode(nodeId)).lastSeen).toBeGreaterThan(1);
    });

    it('never steals an already-claimed node from its owner', async () => {
      const { nodeId } = await service.claimNode('key1', 'Mine', 'user1');
      const res = await service.registerNode('key1', 'Not yours');
      expect(res).toMatchObject({ success: true, nodeId, claimed: true });
      const node = await service.getNode(nodeId);
      expect(node.userId).toBe('user1');
      expect(node.name).toBe('Mine'); // name is not clobbered either
    });

    it('an unclaimed node can be adopted later by a join/claim, keeping its id', async () => {
      const { nodeId } = await service.registerNode('key1', 'Anon rig');
      const claimed = await service.claimNode('key1', 'My rig', 'user1');
      expect(claimed.nodeId).toBe(nodeId); // same identity, now owned
      expect((await service.getNode(nodeId)).userId).toBe('user1');
    });

    it('lets an unclaimed node ping (it exists now), unlike an unregistered one', async () => {
      const { nodeId } = await service.registerNode('key1', 'Rig');
      const ping = await service.updateNodeStatus(nodeId, 'key1', { tps: 12 });
      expect(ping).toMatchObject({ success: true, status: 'online' });
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
