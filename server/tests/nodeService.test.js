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
    // A CLAIMED node carries state its owner set and cannot recreate by waiting —
    // the name, is_public, and the user_id that makes it eligible for that user's
    // `private` jobs. Pruning it because the rig was off for a week silently
    // downgraded the owner to public-only routing. Only unclaimed rows are pruned;
    // those re-register themselves on the next ping.
    it('prunes week-old UNCLAIMED nodes and keeps claimed ones', async () => {
      const spy = jest.spyOn(console, 'log').mockImplementation(() => {});
      const fresh = await service.claimNode('fresh', 'fresh', 'u');
      const stale = await service.claimNode('stale', 'stale', 'u');
      const oldClaimed = await service.claimNode('oldclaimed', 'oldclaimed', 'u');
      const oldAnon = await service.registerNode('oldanon', 'oldanon');
      await setLastSeen(stale.nodeId, Date.now() - 20 * 60 * 1000);
      await setLastSeen(oldClaimed.nodeId, Date.now() - 8 * 24 * 60 * 60 * 1000);
      await setLastSeen(oldAnon.nodeId, Date.now() - 8 * 24 * 60 * 60 * 1000);

      await service.checkNodeStatuses();

      // The unclaimed one is gone; the claimed one survives with its owner intact.
      expect(await service.getNode(oldAnon.nodeId)).toBeNull();
      const kept = await service.getNode(oldClaimed.nodeId);
      expect(kept).not.toBeNull();
      expect(kept.userId).toBe('u');
      expect(await service.getNode(fresh.nodeId)).not.toBeNull();
      expect(spy).toHaveBeenCalledWith('Node status check: 1 online, 2 offline');
      spy.mockRestore();
    });

    it('handles an empty table', async () => {
      const spy = jest.spyOn(console, 'log').mockImplementation(() => {});
      await service.checkNodeStatuses();
      expect(spy).toHaveBeenCalledWith('Node status check: 0 online, 0 offline');
      spy.mockRestore();
    });
  });

  // Server-measured generation speed: the number assignment gates on, and what
  // the network page paints. Deliberately not the `tps` a node reports on its
  // ping — see nodeService for why a self-reported figure can't decide who gets
  // work.
  describe('speed measurement', () => {
    const seed = async (id) => (await service.claimNode('pk-' + id, id, 'u1')).nodeId;

    it('sets the first sample outright, then blends later ones', async () => {
      const id = await seed('a');

      // 400 tokens in 10s = 40 tok/s. Nothing to blend with, so it lands as-is.
      expect((await service.recordSpeedSample(id, 400, 10000)).tps).toBeCloseTo(40, 5);

      // A slower run moves the number toward it, not onto it: one unlucky job (a
      // node that picked mining back up mid-generation) must not swing what work
      // the node is offered.
      const second = await service.recordSpeedSample(id, 200, 10000); // 20 tok/s
      expect(second.tps).toBeCloseTo(40 * 0.7 + 20 * 0.3, 5);
      expect(second.samples).toBe(2);
    });

    it('replaces rather than blends when told to (a cold node\'s warm-up run)', async () => {
      const id = await seed('b');
      await service.recordSpeedSample(id, 100, 10000); // 10 tok/s — model still loading
      // The real measurement must not be dragged down by the warm-up figure.
      const after = await service.recordSpeedSample(id, 400, 10000, { replace: true });
      expect(after.tps).toBeCloseTo(40, 5);
    });

    it('ignores nonsense samples and unknown nodes', async () => {
      const id = await seed('c');
      expect(await service.recordSpeedSample(id, 0, 10000)).toBeNull();
      expect(await service.recordSpeedSample(id, 400, 0)).toBeNull();
      expect(await service.recordSpeedSample(id, 'x', 10000)).toBeNull();
      expect(await service.recordSpeedSample(id, 400, 'y')).toBeNull();
      expect(await service.recordSpeedSample('', 400, 10000)).toBeNull();
      expect(await service.recordSpeedSample('ghost', 400, 10000)).toBeNull();
      expect((await service.getSpeed(id)).tps).toBeNull(); // nothing was written
    });

    it('discards a reply too short to measure', async () => {
      const id = await seed('g');
      // 40 tokens in 5s reads as 8 tok/s on a card that really runs at 40: the
      // wall time is mostly prompt prefill and two HTTP round trips. That single
      // reading used to set the node's speed outright and gate it out of all
      // traffic, so a reply this small is not a measurement.
      expect(await service.recordSpeedSample(id, 40, 5000)).toBeNull();
      expect((await service.getSpeed(id)).tps).toBeNull();

      expect(await service.recordSpeedSample(id, NodeService.MIN_SAMPLE_TOKENS, 5000)).not.toBeNull();
    });

    it('discards a sample claiming an impossible rate', async () => {
      const id = await seed('f');
      await service.recordSpeedSample(id, 400, 10000); // 40 tok/s, real
      // totalTokens comes from the node's own metrics, and the rate now decides
      // which jobs it is offered — so over-reporting is a way to attract longer
      // work than the hardware can finish. Dropped, not clamped: a clamped value
      // would still drag the average up to the ceiling.
      expect(await service.recordSpeedSample(id, 10_000_000, 1000)).toBeNull();
      const after = await service.getSpeed(id);
      expect(after.tps).toBeCloseTo(40, 5);
      expect(after.samples).toBe(1); // the lie never counted

      // Right at the ceiling still counts — the guard is for the absurd, not the fast.
      expect(await service.recordSpeedSample(id, NodeService.MAX_SAMPLE_TPS, 1000)).not.toBeNull();
    });

    it('reports an unmeasured node as unknown, not as slow', async () => {
      const id = await seed('d');
      const speed = await service.getSpeed(id);
      expect(speed).toEqual({ tps: null, samples: 0, at: null, stale: true, known: false });
      expect(await service.getSpeed('ghost')).toBeNull();
    });

    it('treats an old measurement as stale', async () => {
      const id = await seed('e');
      await service.recordSpeedSample(id, 400, 10000);
      expect((await service.getSpeed(id)).known).toBe(true);

      // The node may have swapped cards or started co-running the miner since.
      await db.query('UPDATE nodes SET speed_at = $1 WHERE node_id = $2',
        [Date.now() - NodeService.SPEED_STALE_MS - 1000, id]);
      const speed = await service.getSpeed(id);
      expect(speed.stale).toBe(true);
      expect(speed.known).toBe(false); // so it neither gates the node nor counts as measured
    });
  });

  describe('listServingNodes', () => {
    it('lists online nodes with their measured speed, fastest first', async () => {
      const slow = (await service.claimNode('pk-slow', 'slow', 'u1')).nodeId;
      const fast = (await service.claimNode('pk-fast', 'fast', 'u1')).nodeId;
      const cold = (await service.claimNode('pk-cold', 'cold', 'u1')).nodeId;
      await service.recordSpeedSample(slow, 200, 10000); // 20 tok/s
      await service.recordSpeedSample(fast, 450, 10000); // 45 tok/s

      const nodes = await service.listServingNodes();
      expect(nodes.map((n) => n.nodeId)).toEqual([fast, slow, cold]); // NULLS LAST
      expect(nodes[0].tps).toBe(45);
      expect(nodes[0].stale).toBe(false);
      expect(nodes[2].tps).toBeNull();   // never measured
      expect(nodes[2].stale).toBe(true);
      expect(nodes[0].name).toBe('fast');
    });

    it('leaves out nodes that have gone offline', async () => {
      const id = (await service.claimNode('pk-gone', 'gone', 'u1')).nodeId;
      await setLastSeen(id, Date.now() - 60 * 60 * 1000);
      expect(await service.listServingNodes()).toEqual([]);
    });

    it('reports a node with no name as null rather than empty', async () => {
      await db.query(
        "INSERT INTO nodes (node_id, public_key, status, last_seen) VALUES ('bare', 'k', 'online', $1)",
        [Date.now()]
      );
      const [node] = await service.listServingNodes();
      expect(node).toMatchObject({ nodeId: 'bare', name: null, device: null, model: null });
    });
  });

  describe('getNode', () => {
    it('returns null for an unknown node', async () => {
      expect(await service.getNode('nope')).toBeNull();
    });
  });
});
