const redisMock = require('redis-mock');
const { createCamelClient } = require('./helpers/camelRedis');
const BaseRepository = require('../src/repositories/BaseRepository');
const { JobRepository, NodeRepository } = require('../src/repositories');
const JobServiceV2 = require('../src/services/jobServiceV2');
const NodeServiceV2 = require('../src/services/nodeServiceV2');
const JobService = require('../src/services/jobService');
const JobController = require('../src/controllers/jobController');
const nodeServiceV1 = require('../src/services/nodeService');
const { createRedisCompat } = require('../src/utils/redisCompat');

const flush = (client) => new Promise((resolve) => client.flushall(resolve));

// The full semantic suites run against a camelCase (Redis v5 style,
// promise-based) client, which the compatibility layers fully support. The
// lowercase (redis-mock, callback-based) branches are covered separately.
describe('Repositories + V2 services [camelCase]', () => {
  let client;
  let jobRepo;
  let nodeRepo;
  let jobService;
  let nodeService;

  beforeEach(async () => {
    client = createCamelClient();
    await flush(client);
    jobRepo = new JobRepository(client);
    nodeRepo = new NodeRepository(client);
    jobService = new JobServiceV2(client);
    nodeService = new NodeServiceV2(client);
  });

  afterEach(() => {
    client.quit();
  });

  describe('JobRepository', () => {
    it('creates, reads, updates and deletes a job', async () => {
      await jobRepo.createJob({ id: 'j1', prompt: 'hi', priority: 5, options: { a: 1 } });
      const job = await jobRepo.getJob('j1');
      expect(job.id).toBe('j1');
      expect(job.options).toEqual({ a: 1 });

      // updateJob with both object and primitive values
      await jobRepo.updateJob('j1', { status: 'running', result: { ok: true } });
      const updated = await jobRepo.getJob('j1');
      expect(updated.status).toBe('running');
      expect(updated.result).toEqual({ ok: true });

      await jobRepo.deleteJob('j1');
      expect(await jobRepo.getJob('j1')).toBeNull();
    });

    it('returns raw value when a JSON field cannot be parsed', async () => {
      await jobRepo.createJob({ id: 'j-bad', prompt: 'x' });
      await jobRepo.updateJob('j-bad', { result: 'not-json' });
      const job = await jobRepo.getJob('j-bad');
      expect(job.result).toBe('not-json');
    });

    it('manages all queues and stats', async () => {
      await jobRepo.addToPendingQueue('p1');
      await jobRepo.addToAssignedQueue('a1');
      await jobRepo.addToCompletedQueue('c1');
      await jobRepo.addToFailedQueue('f1');

      expect(await jobRepo.getPendingJobs(10)).toContain('p1');
      expect(await jobRepo.getAssignedJobs()).toContain('a1');
      expect(await jobRepo.getCompletedJobs()).toContain('c1');
      expect(await jobRepo.getFailedJobs()).toContain('f1');

      await jobRepo.removeFromPendingQueue('p1');
      await jobRepo.removeFromAssignedQueue('a1');
      expect(await jobRepo.getPendingJobs(10)).not.toContain('p1');

      const stats = await jobRepo.getQueueStats();
      expect(stats.completed).toBe(1);
      expect(stats.running).toBe(stats.assigned);
    });

    it('handles locks, lock holders and lock extension', async () => {
      await jobRepo.acquireLock('jl', 'nodeA', 300);
      expect(await jobRepo.checkLock('jl', 'nodeA')).toBe(true);
      expect(await jobRepo.getLockHolder('jl')).toBe('nodeA');

      // extendLock when holding the lock
      expect(await jobRepo.extendLock('jl', 'nodeA', 100)).toBeTruthy();
      // extendLock when NOT holding the lock
      expect(await jobRepo.extendLock('jl', 'other')).toBe(false);

      // releaseLock with wrong node does nothing
      expect(await jobRepo.releaseLock('jl', 'other')).toBe(false);
      // releaseLock with correct node
      await jobRepo.releaseLock('jl', 'nodeA');
      expect(await jobRepo.checkLock('jl', 'nodeA')).toBe(false);
    });

    it('stores, retrieves and deletes chunks', async () => {
      await jobRepo.storeChunk('jc', 0, 'a');
      await jobRepo.storeChunk('jc', 1, 'b');
      const chunks = await jobRepo.getChunks('jc');
      expect(chunks.map((c) => c.content)).toEqual(['a', 'b']);
      await jobRepo.deleteChunks('jc');
      expect(await jobRepo.getChunks('jc')).toEqual([]);
    });

    it('cleans up old completed/failed jobs', async () => {
      await jobRepo.createJob({ id: 'old1', prompt: 'x' });
      await jobRepo.createJob({ id: 'old2', prompt: 'y' });
      await jobRepo.addToCompletedQueue('old1', 1000);
      await jobRepo.addToFailedQueue('old2', 1000);

      const removed = await jobRepo.cleanupOldJobs();
      expect(removed).toBe(2);
    });

    it('returns timed out jobs back to pending', async () => {
      await jobRepo.createJob({ id: 'to1', prompt: 'x' });
      await jobRepo.addToAssignedQueue('to1', 1000);
      await jobRepo.acquireLock('to1', 'nodeA', 300);

      const timedOut = await jobRepo.checkTimeouts();
      expect(timedOut).toContain('to1');
      expect(await jobRepo.getPendingJobs(10)).toContain('to1');
    });
  });

  describe('NodeRepository', () => {
    it('creates, reads, updates and deletes nodes', async () => {
      await nodeRepo.createNode({
        nodeId: 'n1', userId: 'u1', name: 'N1', status: 'online',
        isPublic: true, lastSeen: Date.now(),
      });
      expect((await nodeRepo.getNode('n1')).name).toBe('N1');

      // updateNode toggling public -> private
      await nodeRepo.updateNode('n1', { isPublic: false });
      expect(await nodeRepo.isPublicNode('n1')).toBeFalsy();
      // updateNode toggling private -> public
      await nodeRepo.updateNode('n1', { isPublic: true });
      expect(await nodeRepo.isPublicNode('n1')).toBe(true);

      // updateNode on missing node returns null
      expect(await nodeRepo.updateNode('missing', { x: 1 })).toBeNull();

      // deleteNode missing returns false
      expect(await nodeRepo.deleteNode('missing')).toBe(false);
      // deleteNode existing
      await nodeRepo.deleteNode('n1');
      expect(await nodeRepo.getNode('n1')).toBeNull();
    });

    it('manages node status transitions', async () => {
      await nodeRepo.createNode({ nodeId: 'ns', userId: 'u', status: 'online', lastSeen: Date.now() });
      await nodeRepo.markNodeOnline('ns', { capabilities: { gpu: true } });
      await nodeRepo.markNodeOffline('ns');
      expect((await nodeRepo.getNode('ns')).status).toBe('offline');

      // checkNodeStatus on a stale online node flips it offline
      await nodeRepo.updateNode('ns', { status: 'online', lastSeen: Date.now() - 20 * 60 * 1000 });
      expect(await nodeRepo.checkNodeStatus('ns')).toBe('offline');
      // checkNodeStatus on a fresh node returns its status
      await nodeRepo.updateNode('ns', { status: 'online', lastSeen: Date.now() });
      expect(await nodeRepo.checkNodeStatus('ns')).toBe('online');
      // checkNodeStatus on missing node returns null
      expect(await nodeRepo.checkNodeStatus('nope')).toBeNull();
    });

    it('lists user nodes (marking stale ones offline) and counts them', async () => {
      await nodeRepo.createNode({ nodeId: 'un1', userId: 'uu', status: 'online', lastSeen: Date.now() });
      await nodeRepo.createNode({ nodeId: 'un2', userId: 'uu', status: 'online', lastSeen: Date.now() - 20 * 60 * 1000 });

      const nodes = await nodeRepo.getUserNodes('uu');
      expect(nodes).toHaveLength(2);
      const stale = nodes.find((n) => n.nodeId === 'un2');
      expect(stale.status).toBe('offline');

      expect(await nodeRepo.countUserNodes('uu')).toBe(2);
      // empty user
      expect(await nodeRepo.getUserNodes('none')).toEqual([]);
      expect(await nodeRepo.countUserNodes('none')).toBe(0);
    });

    it('lists public nodes with online/offline/error handling', async () => {
      // public + recently active -> online
      await nodeRepo.createNode({ nodeId: 'pub1', userId: 'p', status: 'online', isPublic: true, lastSeen: Date.now() });
      // public + stale but key has TTL -> offline
      await nodeRepo.createNode({ nodeId: 'pub2', userId: 'p', status: 'online', isPublic: true, lastSeen: Date.now() - 20 * 60 * 1000 });

      // public node whose data is corrupt -> error path is swallowed
      await nodeRepo.redis.set('node:pubbad', 'not-json');
      await nodeRepo.addToPublicNodes('pubbad');

      const publicNodes = await nodeRepo.getPublicNodes();
      const ids = publicNodes.map((n) => n.nodeId);
      expect(ids).toContain('pub1');
      expect(ids).toContain('pub2');
      expect(publicNodes.find((n) => n.nodeId === 'pub1').status).toBe('online');
      expect(publicNodes.find((n) => n.nodeId === 'pub2').status).toBe('offline');

      await nodeRepo.removeFromPublicNodes('pub1');
      expect(await nodeRepo.isPublicNode('pub1')).toBeFalsy();
      // empty public set
      await nodeRepo.removeFromPublicNodes('pub2');
      await nodeRepo.removeFromPublicNodes('pubbad');
      expect(await nodeRepo.getPublicNodes()).toEqual([]);
    });

    it('treats public nodes without a TTL as offline', async () => {
      // Store a node directly (no TTL) so the ttl<=0 branch is exercised
      await nodeRepo.redis.set('node:pub-nottl', JSON.stringify({ nodeId: 'pub-nottl', lastSeen: Date.now() }));
      await nodeRepo.addToPublicNodes('pub-nottl');
      const publicNodes = await nodeRepo.getPublicNodes();
      const node = publicNodes.find((n) => n.nodeId === 'pub-nottl');
      expect(node.status).toBe('offline');
    });

    it('claims nodes and enforces ownership', async () => {
      const node = await nodeRepo.claimNode('pk-claim', 'My Node', 'owner1');
      expect(node.nodeId).toBe(nodeRepo.generateNodeFingerprint('pk-claim'));

      // re-claim by same owner is fine
      await expect(nodeRepo.claimNode('pk-claim', 'My Node', 'owner1')).resolves.toBeDefined();
      // claim by a different owner throws
      await expect(nodeRepo.claimNode('pk-claim', 'My Node', 'owner2')).rejects.toThrow('already claimed');
    });

    it('updates visibility with auth checks', async () => {
      await nodeRepo.claimNode('pk-vis', 'Vis', 'owner-vis');
      const id = nodeRepo.generateNodeFingerprint('pk-vis');

      await nodeRepo.updateNodeVisibility(id, 'owner-vis', true);
      expect(await nodeRepo.isPublicNode(id)).toBe(true);

      await expect(nodeRepo.updateNodeVisibility('missing', 'owner-vis', true)).rejects.toThrow('Node not found');
      await expect(nodeRepo.updateNodeVisibility(id, 'someone-else', true)).rejects.toThrow('Unauthorized');
    });

    it('cleans up inactive nodes and reports stats', async () => {
      await nodeRepo.createNode({ nodeId: 'fresh', userId: 'u', status: 'online', lastSeen: Date.now() });
      await nodeRepo.createNode({ nodeId: 'inactive', userId: 'u', status: 'online', isPublic: true, lastSeen: 1000 });

      const stats = await nodeRepo.getNodeStats();
      expect(stats.total).toBeGreaterThanOrEqual(2);
      expect(stats.online).toBeGreaterThanOrEqual(1);
      expect(stats.offline).toBeGreaterThanOrEqual(1);

      const deleted = await nodeRepo.cleanupInactiveNodes(60 * 1000);
      expect(deleted).toBeGreaterThanOrEqual(1);
    });
  });

  describe('JobServiceV2', () => {
    it('creates jobs and throws on missing job', async () => {
      const job = await jobService.createJob({ prompt: 'p', userId: 'u' });
      expect(job.status).toBe('pending');
      expect((await jobService.getJob(job.id)).prompt).toBe('p');
      await expect(jobService.getJob('nope')).rejects.toThrow('not found');
    });

    it('transitions job status through every queue', async () => {
      const job = await jobService.createJob({ prompt: 'p', userId: 'u' });
      await jobService.updateJobStatus(job.id, 'assigned');
      await jobService.updateJobStatus(job.id, 'completed');
      const job2 = await jobService.createJob({ prompt: 'p2', userId: 'u' });
      await jobService.updateJobStatus(job2.id, 'assigned');
      await jobService.updateJobStatus(job2.id, 'failed');
      const job3 = await jobService.createJob({ prompt: 'p3', userId: 'u' });
      await jobService.updateJobStatus(job3.id, 'assigned');
      await jobService.updateJobStatus(job3.id, 'pending');
      // status without a queue transition
      await jobService.updateJobStatus(job3.id, 'running');
      expect((await jobService.getJob(job3.id)).status).toBe('running');
    });

    it('assigns jobs, heartbeats, stores chunks and completes', async () => {
      const job = await jobService.createJob({ prompt: 'p', userId: 'u' });
      const assigned = await jobService.assignJobsToNode('nodeA', 1);
      expect(assigned[0].assignedTo).toBe('nodeA');

      const hb = await jobService.handleHeartbeat(job.id, 'nodeA');
      expect(hb.success).toBe(true);

      const stored = await jobService.storeChunk(job.id, 'nodeA', { chunkIndex: 0, content: 'hello', metrics: { t: 1 } });
      expect(stored.stored).toBe(true);

      const completed = await jobService.completeJob(job.id, 'nodeA');
      expect(completed.status).toBe('completed');
      expect(completed.result).toBe('hello');
    });

    it('completes a job with an explicit final output', async () => {
      const job = await jobService.createJob({ prompt: 'p', userId: 'u' });
      await jobService.assignJobsToNode('nodeA', 1);
      const completed = await jobService.completeJob(job.id, 'nodeA', 'FINAL');
      expect(completed.result).toBe('FINAL');
    });

    it('fails a job and enforces lock ownership across operations', async () => {
      const job = await jobService.createJob({ prompt: 'p', userId: 'u' });
      await jobService.assignJobsToNode('nodeA', 1);

      await expect(jobService.handleHeartbeat(job.id, 'wrong')).rejects.toThrow('does not hold lock');
      await expect(jobService.storeChunk(job.id, 'wrong', { chunkIndex: 0, content: 'x' })).rejects.toThrow('does not hold lock');
      await expect(jobService.completeJob(job.id, 'wrong')).rejects.toThrow('does not hold lock');
      await expect(jobService.failJob(job.id, 'wrong', 'r')).rejects.toThrow('does not hold lock');

      const failed = await jobService.failJob(job.id, 'nodeA', 'boom');
      expect(failed.status).toBe('failed');
      expect(failed.failureReason).toBe('boom');
    });

    it('gets job results including partial chunks while running', async () => {
      const job = await jobService.createJob({ prompt: 'p', userId: 'u' });
      await jobService.assignJobsToNode('nodeA', 1);
      await jobService.storeChunk(job.id, 'nodeA', { chunkIndex: 0, content: 'partial' });

      const result = await jobService.getJobResult(job.id);
      expect(result.chunks.length).toBeGreaterThanOrEqual(1);

      await expect(jobService.getJobResult('missing')).rejects.toThrow('not found');
    });

    it('reports stats, checks timeouts and cleans up', async () => {
      await jobService.createJob({ prompt: 'p', userId: 'u' });
      const stats = await jobService.getQueueStats();
      expect(stats.pending).toBeGreaterThanOrEqual(1);
      expect(await jobService.checkTimeouts()).toEqual(expect.any(Array));
      expect(await jobService.cleanupOldJobs()).toBe(0);
    });

    it('cancels jobs with the right guards', async () => {
      // cancel a pending job (no lock holder)
      const pending = await jobService.createJob({ prompt: 'p', userId: 'u' });
      expect((await jobService.cancelJob(pending.id)).success).toBe(true);

      // cancel an assigned job (releases the lock holder)
      const assigned = await jobService.createJob({ prompt: 'p', userId: 'u' });
      await jobService.assignJobsToNode('nodeA', 1);
      expect((await jobService.cancelJob(assigned.id, 'stop')).success).toBe(true);

      // cannot cancel completed jobs
      const done = await jobService.createJob({ prompt: 'p', userId: 'u' });
      await jobService.updateJobStatus(done.id, 'completed');
      await expect(jobService.cancelJob(done.id)).rejects.toThrow('Cannot cancel');

      await expect(jobService.cancelJob('missing')).rejects.toThrow('not found');
    });

    it('lists jobs by status and validates the status', async () => {
      const job = await jobService.createJob({ prompt: 'p', userId: 'u' });
      expect((await jobService.getJobsByStatus('pending')).length).toBeGreaterThanOrEqual(1);

      await jobService.assignJobsToNode('nodeA', 1);
      expect(await jobService.getJobsByStatus('assigned')).toEqual(expect.any(Array));
      expect(await jobService.getJobsByStatus('running')).toEqual(expect.any(Array));
      await jobService.updateJobStatus(job.id, 'completed');
      expect(await jobService.getJobsByStatus('completed')).toEqual(expect.any(Array));
      const j2 = await jobService.createJob({ prompt: 'p2', userId: 'u' });
      await jobService.updateJobStatus(j2.id, 'assigned');
      await jobService.updateJobStatus(j2.id, 'failed');
      expect(await jobService.getJobsByStatus('failed')).toEqual(expect.any(Array));

      await expect(jobService.getJobsByStatus('bogus')).rejects.toThrow('Invalid status');
    });

    it('retries only failed jobs', async () => {
      const job = await jobService.createJob({ prompt: 'p', userId: 'u' });
      await jobService.assignJobsToNode('nodeA', 1);
      await jobService.failJob(job.id, 'nodeA', 'err');
      expect((await jobService.retryJob(job.id)).success).toBe(true);

      // now pending, cannot retry
      await expect(jobService.retryJob(job.id)).rejects.toThrow('Can only retry');
      await expect(jobService.retryJob('missing')).rejects.toThrow('not found');
    });
  });

  describe('NodeServiceV2', () => {
    it('claims nodes and surfaces claim errors', async () => {
      const res = await nodeService.claimNode('svc-key', 'Node', 'svc-user');
      expect(res.success).toBe(true);

      const conflict = await nodeService.claimNode('svc-key', 'Node', 'other-user');
      expect(conflict.success).toBe(false);
      expect(conflict.error).toMatch(/already claimed/);
    });

    it('updates node status with validation', async () => {
      const res = await nodeService.claimNode('status-key', 'Node', 'u');
      expect((await nodeService.updateNodeStatus(res.nodeId, 'status-key', { capabilities: { gpu: true } })).success).toBe(true);
      expect((await nodeService.updateNodeStatus('missing', 'k')).error).toMatch(/not found/);
      expect((await nodeService.updateNodeStatus(res.nodeId, 'wrong-key')).error).toMatch(/mismatch/);
    });

    it('reads nodes, stats and visibility', async () => {
      const res = await nodeService.claimNode('vis-key', 'Node', 'vis-u');
      expect(await nodeService.getNode(res.nodeId)).toBeDefined();
      expect((await nodeService.getUserNodes('vis-u')).length).toBe(1);

      const visible = await nodeService.updateNodeVisibility(res.nodeId, 'vis-u', true);
      expect(visible.success).toBe(true);
      const pub = await nodeService.getPublicNodes();
      expect(pub.success).toBe(true);
      expect(pub.count).toBeGreaterThanOrEqual(1);

      const failVisible = await nodeService.updateNodeVisibility('missing', 'vis-u', true);
      expect(failVisible.success).toBe(false);

      expect(await nodeService.checkNodeStatus(res.nodeId)).toBeDefined();
      expect((await nodeService.getNodeStats()).total).toBeGreaterThanOrEqual(1);
      expect(await nodeService.cleanupInactiveNodes(1)).toBeGreaterThanOrEqual(0);
    });

    it('updates capabilities and job info with validation', async () => {
      const res = await nodeService.claimNode('cap-key', 'Node', 'cap-u');

      expect((await nodeService.updateNodeCapabilities(res.nodeId, 'cap-key', { gpu: true })).success).toBe(true);
      expect((await nodeService.updateNodeCapabilities('missing', 'k', {})).error).toMatch(/not found/);
      expect((await nodeService.updateNodeCapabilities(res.nodeId, 'wrong', {})).error).toMatch(/mismatch/);

      expect((await nodeService.updateNodeJobInfo(res.nodeId, 2, 4)).success).toBe(true);
      expect((await nodeService.updateNodeJobInfo('missing', 1, 1)).error).toMatch(/not found/);
    });

    it('queries nodes by status, counts and ownership', async () => {
      const res = await nodeService.claimNode('own-key', 'Node', 'own-u');

      expect(await nodeService.getNodesByStatus('online')).toEqual(expect.any(Array));
      expect(await nodeService.getOnlineNodes()).toEqual(expect.any(Array));
      expect(await nodeService.getOfflineNodes()).toEqual(expect.any(Array));
      expect(await nodeService.countUserNodes('own-u')).toBe(1);
      expect(await nodeService.isPublicNode(res.nodeId)).toBeFalsy();

      expect((await nodeService.validateNodeOwnership(res.nodeId, 'own-u')).valid).toBe(true);
      expect((await nodeService.validateNodeOwnership(res.nodeId, 'intruder')).valid).toBe(false);
      expect((await nodeService.validateNodeOwnership('missing', 'own-u')).valid).toBe(false);

      const bulk = await nodeService.bulkUpdateNodeStatuses();
      expect(bulk.checked).toBeGreaterThanOrEqual(1);

      expect(nodeService.generateNodeFingerprint('own-key')).toHaveLength(6);
    });
  });
});

describe('NodeServiceV2 module exports', () => {
  let client;
  beforeEach(async () => {
    client = createCamelClient();
    await flush(client);
  });
  afterEach(() => client.quit());

  it('exposes a factory and a backward-compatible singleton API', async () => {
    expect(NodeServiceV2.createNodeService(client)).toBeInstanceOf(NodeServiceV2);

    const api = NodeServiceV2.nodeService;
    expect(api.generateNodeFingerprint('singleton-key')).toHaveLength(6);

    const claim = await api.claimNode(client, 'singleton-key', 'Node', 'singleton-user');
    expect(claim.success).toBe(true);

    expect((await api.updateNodeStatus(client, claim.nodeId, 'singleton-key', {})).success).toBe(true);
    expect(await api.getNode(claim.nodeId, client)).toBeDefined();
    expect((await api.getUserNodes(client, 'singleton-user')).length).toBe(1);
    expect((await api.updateNodeVisibility(client, claim.nodeId, 'singleton-user', true)).success).toBe(true);
    expect((await api.getPublicNodes(client)).success).toBe(true);
  });
});

describe('BaseRepository compatibility helpers [camelCase]', () => {
  let client;
  let repo;

  beforeEach(async () => {
    client = createCamelClient();
    await flush(client);
    repo = new BaseRepository(client, 'base:');
  });
  afterEach(() => client.quit());

  it('runs string, hash, set, sorted-set, list and ttl helpers', async () => {
    await repo.set('s', { v: 1 });
    expect(await repo.get('s')).toEqual({ v: 1 });
    expect(await repo.get('missing')).toBeNull();
    await repo.set('s-ttl', { v: 2 }, 100);
    expect(await repo.exists('s-ttl')).toBe(true);
    await repo.expire('s-ttl', 50);
    expect(await repo.ttl('s-ttl')).toBeGreaterThan(0);
    await repo.delete('s');
    expect(await repo.exists('s')).toBe(false);

    await repo.hSet('h', 'f', 'v');
    expect(await repo.hGet('h', 'f')).toBe('v');
    expect(await repo.hGetAll('h')).toEqual({ f: 'v' });
    await repo.hDel('h', 'f');

    await repo.sAdd('set', 'm1');
    expect(await repo.sMembers('set')).toContain('m1');
    await repo.sRem('set', 'm1');
    await repo.sAddDirect('directset', 'd1');
    expect(await repo.sMembersDirect('directset')).toContain('d1');
    await repo.sRemDirect('directset', 'd1');

    await repo.zAdd('z', 1, 'zm');
    expect(await repo.zRange('z', 0, -1)).toContain('zm');
    expect(await repo.zCard('z')).toBe(1);
    await repo.zRem('z', 'zm');

    await repo.lPush('l', 'a', 'b');
    expect(await repo.lRange('l', 0, -1)).toEqual(expect.arrayContaining(['a', 'b']));
    expect(await repo.lLen('l')).toBe(2);

    expect(await repo.keys('*')).toEqual(expect.any(Array));
  });

  it('supports the raw array form of zAdd', async () => {
    await repo.redis.zAdd('base:raw', [{ score: 2, value: 'rawmember' }]);
    expect(await repo.zRange('raw', 0, -1)).toContain('rawmember');
  });
});

describe('BaseRepository lowercase (callback) compat branches', () => {
  // redis-mock exposes only lowercase callback methods, so the compat creators
  // take their lowercase branches here. We only assert the calls resolve.
  let client;
  let repo;

  beforeEach(async () => {
    client = redisMock.createClient();
    await flush(client);
    repo = new BaseRepository(client, 'lc:');
  });
  afterEach(() => client.quit());

  it('exercises hash, sorted-set and list lowercase compat operations', async () => {
    await repo.redis.hSet('lc:h', 'f', 'v');
    expect(await repo.redis.hGet('lc:h', 'f')).toBe('v');
    expect(await repo.redis.hGetAll('lc:h')).toEqual({ f: 'v' });
    await repo.redis.hDel('lc:h', 'f');

    await repo.redis.zAdd('lc:z', { score: 1, member: 'm' });
    expect(await repo.redis.zRange('lc:z', 0, -1)).toContain('m');
    expect(await repo.redis.zCard('lc:z')).toBe(1);
    expect(await repo.redis.zRangeByScore('lc:z', 0, Date.now() + 1000)).toEqual(expect.any(Array));
    await repo.redis.zRem('lc:z', 'm');

    await repo.redis.lPush('lc:l', 'a');
    expect(await repo.redis.lRange('lc:l', 0, -1)).toContain('a');
    expect(await repo.redis.lLen('lc:l')).toBe(1);

    await repo.redis.sRem('lc:s', 'x');

    // raw lowercase zAdd argument forms (object and positional)
    await repo.redis.zAdd('lc:z2', { score: 3, member: 'mm' });
    await repo.redis.zAdd('lc:z3', 4, 'mmm');
  });
});

describe('redisCompat direct branches', () => {
  it('falls back to callbacks for sRem on a lowercase client', async () => {
    const client = redisMock.createClient();
    await flush(client);
    await new Promise((r) => client.sadd('s', 'a', 'b', r));
    const compat = createRedisCompat(client);
    expect(await compat.sRem('s', 'a')).toBeGreaterThanOrEqual(0);
    client.quit();
  });

  it('supports promise-returning lowercase set operations', async () => {
    const promiseClient = {
      sadd: (key, ...rest) => { rest.pop(); return Promise.resolve(2); },
      smembers: (key, cb) => { if (cb) cb(null, ['x']); return Promise.resolve(['x']); },
      srem: (key, ...rest) => { rest.pop(); return Promise.resolve(1); },
    };
    const compat = createRedisCompat(promiseClient);
    expect(await compat.sAdd('k', 'a')).toBe(2);
    expect(await compat.sMembers('k')).toEqual(['x']);
    expect(await compat.sRem('k', 'a')).toBe(1);
  });

  it('returns defaults when basic operations are unavailable', async () => {
    const compat = createRedisCompat({});
    expect(await compat.get('x')).toBeNull();
    expect(await compat.set('x', 'y')).toBe('OK');
  });

  it('uses lowercase fallbacks for setEx and reads keys', async () => {
    const client = redisMock.createClient();
    await flush(client);
    const compat = createRedisCompat(client);
    await compat.setEx('e', 100, 'v');
    expect(await compat.get('e')).toBe('v');
    // keys() exercises the camelCase-named branch; redis-mock needs a callback
    // so we only assert it resolves without throwing.
    await compat.keys('*');
    client.quit();
  });
});

describe('JobService (V1) compatibility branches', () => {
  // Construct the V1 JobService with a camelCase client to exercise the
  // camelCase branches of its inline Redis compatibility layer.
  let client;
  let service;

  beforeEach(async () => {
    client = createCamelClient();
    await flush(client);
    service = new JobService(client);
  });
  afterEach(() => client.quit());

  it('runs a full job lifecycle over the camelCase compat layer', async () => {
    const job = await service.createJob({ prompt: 'hi', userId: 'u', priority: 1 });
    const assigned = await service.assignJobsToNode('nodeA', 1);
    expect(assigned[0].assignedTo).toBe('nodeA');

    await service.handleHeartbeat(job.id, 'nodeA');
    await service.storeChunk(job.id, 'nodeA', { chunkIndex: 0, content: 'x', metrics: { t: 1 } });

    const stats = await service.getQueueStats();
    expect(stats.running).toBeGreaterThanOrEqual(1);

    const completed = await service.completeJob(job.id, 'nodeA');
    expect(completed.status).toBe('completed');

    const result = await service.getJobResult(job.id);
    expect(result.status).toBe('completed');
  });

  it('fails jobs and checks timeouts over the camelCase compat layer', async () => {
    const job = await service.createJob({ prompt: 'hi', userId: 'u' });
    await service.assignJobsToNode('nodeA', 1);
    await service.failJob(job.id, 'nodeA', 'boom');
    expect(await service.checkTimeouts()).toEqual(expect.any(Array));
  });

  it('walks assigned jobs (with live locks) when checking timeouts', async () => {
    await service.createJob({ prompt: 'hi', userId: 'u' });
    await service.assignJobsToNode('nodeA', 1);
    // The job is assigned with a live lock, so checkTimeouts reads its ttl
    // (exercising the camelCase ttl promise branch) without timing it out.
    expect(await service.checkTimeouts()).toEqual([]);
  });

  it('cleans up old completed and failed jobs', async () => {
    const completed = await service.createJob({ prompt: 'c', userId: 'u' });
    const failed = await service.createJob({ prompt: 'f', userId: 'u' });
    // Place both jobs in their status sets with an old timestamp score
    await service.redis.zAdd('jobs:completed', { score: 1000, member: completed.id });
    await service.redis.zAdd('jobs:failed', { score: 1000, member: failed.id });

    const removed = await service.cleanupOldJobs();
    expect(removed).toBe(2);
    expect(await service.getJob(completed.id)).toBeNull();
  });

  it('supports raw zAdd argument forms', async () => {
    // camelCase non-{member} object branch
    await service.redis.zAdd('jobs:raw', [{ score: 1, value: 'cm' }]);
    expect(await service.redis.zRange('jobs:raw', 0, -1)).toContain('cm');

    // lowercase non-object branch on a redis-mock client
    const rmClient = redisMock.createClient();
    await flush(rmClient);
    const rmService = new JobService(rmClient);
    await rmService.redis.zAdd('jobs:raw2', 5, 'lm');
    expect(await rmService.redis.zRange('jobs:raw2', 0, -1)).toContain('lm');
    rmClient.quit();
  });
});

describe('nodeService V1 remaining branches', () => {
  let redisClient;
  beforeEach(async () => {
    redisClient = redisMock.createClient();
    await flush(redisClient);
  });
  afterEach(() => redisClient.quit());

  it('updates status with capabilities, activeJobs and maxConcurrentJobs', async () => {
    await nodeServiceV1.claimNode(redisClient, 'v1-key', 'Node', 'v1-user');
    const nodeId = nodeServiceV1.generateNodeFingerprint('v1-key');

    const result = await nodeServiceV1.updateNodeStatus(redisClient, nodeId, 'v1-key', {
      capabilities: { gpu: true },
      activeJobs: 2,
      maxConcurrentJobs: 4,
    });
    expect(result.success).toBe(true);
  });

  it('counts online and offline nodes during a status sweep', async () => {
    const client = {
      keys: async () => ['node:online-a', 'node:offline-b'],
      ttl: async (key) => (key === 'node:online-a' ? 100 : -2),
    };
    // Should not throw and should walk both the online and offline branches.
    await expect(nodeServiceV1.checkNodeStatuses(client)).resolves.toBeUndefined();
  });
});

describe('JobController remaining branches', () => {
  let controller;
  let jobService;
  let nodeService;
  let redisClient;
  let res;

  beforeEach(async () => {
    jest.spyOn(console, 'error').mockImplementation(() => {});
    redisClient = redisMock.createClient();
    await flush(redisClient);
    jobService = new JobService(redisClient);
    nodeService = { getNode: jest.fn().mockResolvedValue({ nodeId: 'n', status: 'online' }) };
    controller = new JobController(jobService, nodeService, redisClient);
    res = { status: jest.fn().mockReturnThis(), json: jest.fn() };
  });
  afterEach(() => {
    redisClient.quit();
    jest.restoreAllMocks();
  });

  it('falls back to an anonymous user id when no auth user is present', async () => {
    const req = { body: { prompt: 'p' } }; // no req.user
    await controller.submitJob(req, res);
    expect(res.status).toHaveBeenCalledWith(201);
  });

  it('defaults to one job when maxJobs is omitted while polling', async () => {
    const req = { body: { nodeId: 'n' } }; // no maxJobs
    await controller.pollJobs(req, res);
    expect(res.json).toHaveBeenCalled();
  });
});

describe('Defensive / fallback branch coverage', () => {
  let client;
  let jobRepo;
  let nodeRepo;

  beforeEach(async () => {
    client = createCamelClient();
    await flush(client);
    jobRepo = new JobRepository(client);
    nodeRepo = new NodeRepository(client);
  });
  afterEach(() => client.quit());

  it('uses default arguments for paginated/queue helpers', async () => {
    await jobRepo.addToPendingQueue('dp1');
    expect(await jobRepo.getPendingJobs()).toContain('dp1'); // default limit
    // acquireLock with default ttl
    expect(await jobRepo.acquireLock('dlock', 'nodeZ')).toBeTruthy();
  });

  it('skips lock expiry when the lock cannot be set', async () => {
    jobRepo.redis.set = async () => null; // simulate set failing
    expect(await jobRepo.acquireLock('nolock', 'nodeZ')).toBeFalsy();
  });

  it('handles nodes without a userId on delete', async () => {
    await nodeRepo.createNode({ nodeId: 'no-user', status: 'online', isPublic: true, lastSeen: Date.now() });
    expect(await nodeRepo.deleteNode('no-user')).toBeTruthy();
  });

  it('marks a node online using default additional data', async () => {
    await nodeRepo.createNode({ nodeId: 'def-online', userId: 'u', status: 'offline', lastSeen: Date.now() });
    await nodeRepo.markNodeOnline('def-online');
    expect((await nodeRepo.getNode('def-online')).status).toBe('online');
  });

  it('skips missing nodes when listing user and public nodes', async () => {
    // user set references a node that does not exist
    await nodeRepo.sAddDirect('user_nodes:ghost-user', 'ghost-node');
    expect(await nodeRepo.getUserNodes('ghost-user')).toEqual([]);

    // public set references a node that does not exist
    await nodeRepo.addToPublicNodes('ghost-public');
    expect(await nodeRepo.getPublicNodes()).toEqual([]);
  });

  it('treats a null member set as empty when counting and aggregating', async () => {
    nodeRepo.sMembersDirect = async () => null;
    expect(await nodeRepo.countUserNodes('whoever')).toBe(0);

    nodeRepo.keys = async () => null;
    const stats = await nodeRepo.getNodeStats();
    expect(stats.total).toBe(0);
    expect(await nodeRepo.cleanupInactiveNodes()).toBe(0); // default maxInactiveTime
  });

  it('skips phantom node keys when aggregating stats', async () => {
    nodeRepo.keys = async () => ['node:phantom'];
    nodeRepo.getNode = async () => null;
    const stats = await nodeRepo.getNodeStats();
    expect(stats.total).toBe(1);
    expect(stats.online).toBe(0);
    expect(stats.offline).toBe(0);
  });
});

describe('JobServiceV2 lock-holder release on cancel', () => {
  let client;
  let jobService;
  beforeEach(async () => {
    client = createCamelClient();
    await flush(client);
    jobService = new JobServiceV2(client);
  });
  afterEach(() => client.quit());

  it('releases the held lock when cancelling an assigned job', async () => {
    const job = await jobService.createJob({ prompt: 'p', userId: 'u' });
    await jobService.assignJobsToNode('nodeA', 1);
    const result = await jobService.cancelJob(job.id, 'stop');
    expect(result.success).toBe(true);
  });
});

describe('NodeServiceV2 remaining branches', () => {
  let client;
  let nodeService;
  beforeEach(async () => {
    client = createCamelClient();
    await flush(client);
    nodeService = new NodeServiceV2(client);
  });
  afterEach(() => client.quit());

  it('reports the private message when making a node private', async () => {
    const res = await nodeService.claimNode('priv-key', 'Node', 'priv-u');
    const out = await nodeService.updateNodeVisibility(res.nodeId, 'priv-u', false);
    expect(out.message).toMatch(/private/);
  });

  it('does not count nodes whose status check is falsy during a bulk update', async () => {
    // A node with an empty status produces a falsy checkNodeStatus result.
    await nodeService.nodeRepo.createNode({ nodeId: 'blank', userId: 'u', status: '', lastSeen: Date.now() });
    const bulk = await nodeService.bulkUpdateNodeStatuses();
    expect(bulk.checked).toBeGreaterThanOrEqual(1);
    expect(bulk.updated).toBeLessThan(bulk.checked);
  });
});

describe('redisCompat camelCase sRem branch', () => {
  it('delegates to a camelCase sRem when available', async () => {
    const compat = createRedisCompat({ sRem: async (key, ...members) => members.length });
    expect(await compat.sRem('k', 'a', 'b')).toBe(2);
  });
});

describe('Compat fallback defaults and error paths', () => {
  it('BaseRepository lowercase callbacks fall back to empty defaults', async () => {
    // A lowercase client whose callbacks yield undefined exercises the
    // `result || []` / `|| 0` / `|| {}` fallbacks in the compat creators.
    const nullClient = {
      zrange: (k, s, e, cb) => cb(null, undefined),
      zcard: (k, cb) => cb(null, undefined),
      zrangebyscore: (k, mn, mx, cb) => cb(null, undefined),
      hgetall: (k, cb) => cb(null, undefined),
      lrange: (k, s, e, cb) => cb(null, undefined),
      llen: (k, cb) => cb(null, undefined),
    };
    const repo = new BaseRepository(nullClient); // default keyPrefix ''
    expect(repo.getKey('x')).toBe('x');
    expect(await repo.redis.zRange('k', 0, -1)).toEqual([]);
    expect(await repo.redis.zCard('k')).toBe(0);
    expect(await repo.redis.zRangeByScore('k', 0, 1)).toEqual([]);
    expect(await repo.redis.hGetAll('k')).toEqual({});
    expect(await repo.redis.lRange('k', 0, -1)).toEqual([]);
    expect(await repo.redis.lLen('k')).toBe(0);
  });

  it('redisCompat set operations surface errors', async () => {
    const errClient = {
      sadd: (k, ...rest) => rest.pop()(new Error('sadd failed')),
      smembers: (k, cb) => cb(new Error('smembers failed')),
      srem: (k, ...rest) => rest.pop()(new Error('srem failed')),
    };
    const compat = createRedisCompat(errClient);
    await expect(compat.sAdd('k', 'm')).rejects.toThrow('sadd failed');
    await expect(compat.sMembers('k')).rejects.toThrow('smembers failed');
    await expect(compat.sRem('k', 'm')).rejects.toThrow('srem failed');
  });

  it('redisCompat set operations fall back to empty defaults', async () => {
    const undefClient = {
      sadd: (k, ...rest) => rest.pop()(null, undefined),
      smembers: (k, cb) => cb(null, undefined),
      srem: (k, ...rest) => rest.pop()(null, undefined),
    };
    const compat = createRedisCompat(undefClient);
    expect(await compat.sAdd('k', 'm')).toBe(0);
    expect(await compat.sMembers('k')).toEqual([]);
    expect(await compat.sRem('k', 'm')).toBe(0);
  });

  it('redisCompat set operations handle promise-returning clients (falsy results)', async () => {
    const promiseClient = {
      sadd: () => Promise.resolve(undefined),
      smembers: () => Promise.resolve(undefined),
      srem: () => Promise.resolve(undefined),
    };
    const compat = createRedisCompat(promiseClient);
    expect(await compat.sAdd('k', 'm')).toBe(0);
    expect(await compat.sMembers('k')).toEqual([]);
    expect(await compat.sRem('k', 'm')).toBe(0);
  });
});

describe('JobService (V1) reachable edge branches', () => {
  let client;
  let service;
  beforeEach(async () => {
    client = createCamelClient();
    await flush(client);
    service = new JobService(client);
  });
  afterEach(() => client.quit());

  it('applies default fields when creating a job and honours provided fields', async () => {
    const defaulted = await service.createJob({ prompt: 'only-prompt' });
    expect(defaulted.model).toBe('llama3.2:3b');
    expect(defaulted.maxTokens).toBe(1000);
    expect(defaulted.temperature).toBe(0.7);
    expect(defaulted.priority).toBe(0);

    const custom = await service.createJob({
      prompt: 'p', model: 'm', options: { a: 1 }, priority: 9, maxTokens: 50, temperature: 0.1, userId: 'u',
    });
    expect(custom.model).toBe('m');
    expect(custom.priority).toBe(9);
  });

  it('honours NX semantics in the set override', async () => {
    expect(await service.redis.set('nxkey', 'first', { NX: true, EX: 10 })).toBe('OK');
    // Second NX set on an existing key is rejected
    expect(await service.redis.set('nxkey', 'second', { NX: true, EX: 10 })).toBeNull();
    // EX-only path
    expect(await service.redis.set('exkey', 'v', { EX: 10 })).toBeTruthy();
    // options present but neither the NX+EX nor EX-only branch applies
    expect(await service.redis.set('nooptkey', 'v', { NX: true })).toBeTruthy();
  });

  it('skips timed out jobs whose data has disappeared', async () => {
    // A job id sits in the assigned queue with no data and no lock (ttl -2)
    await service.redis.zAdd('jobs:assigned', { score: Date.now(), member: 'ghost-timeout' });
    const timedOut = await service.checkTimeouts();
    expect(timedOut).not.toContain('ghost-timeout');
  });

  it('returns lock-expired jobs to the queue', async () => {
    const job = await service.createJob({ prompt: 'p', userId: 'u' });
    await service.assignJobsToNode('nodeA', 1);
    // Force the lock to look expired
    await service.redis.del(`job:${job.id}:lock`);
    const timedOut = await service.checkTimeouts();
    expect(timedOut).toContain(job.id);
  });

  it('returns jobs whose heartbeat has gone stale', async () => {
    const job = await service.createJob({ prompt: 'p', userId: 'u' });
    await service.assignJobsToNode('nodeA', 1);
    // Lock still alive, but heartbeat is old
    await service.redis.set(`job:${job.id}:heartbeat`, String(Date.now() - 120000));
    const timedOut = await service.checkTimeouts();
    expect(timedOut).toContain(job.id);
  });

  it('formats job results for completed, failed, running and other statuses', async () => {
    // completed
    const c = await service.createJob({ prompt: 'p', userId: 'u' });
    await service.assignJobsToNode('nodeA', 1);
    await service.storeChunk(c.id, 'nodeA', { chunkIndex: 0, content: 'done' });
    await service.completeJob(c.id, 'nodeA');
    expect((await service.getJobResult(c.id)).status).toBe('completed');

    // failed
    const f = await service.createJob({ prompt: 'p', userId: 'u' });
    await service.assignJobsToNode('nodeA', 1);
    await service.failJob(f.id, 'nodeA', 'bad');
    expect((await service.getJobResult(f.id)).status).toBe('failed');

    // running (partial)
    const r = await service.createJob({ prompt: 'p', userId: 'u' });
    await service.assignJobsToNode('nodeA', 1);
    await service.updateJobStatus(r.id, 'running');
    await service.storeChunk(r.id, 'nodeA', { chunkIndex: 0, content: 'partial' });
    const running = await service.getJobResult(r.id);
    expect(running.status).toBe('running');
    expect(running.partial).toBe('partial');

    // other (pending)
    const p = await service.createJob({ prompt: 'p', userId: 'u' });
    expect((await service.getJobResult(p.id)).status).toBe('pending');

    // missing job throws
    await expect(service.getJobResult('nope')).rejects.toThrow('not found');
  });
});

describe('JobServiceV2 reachable edge branches', () => {
  let client;
  let jobService;
  beforeEach(async () => {
    client = createCamelClient();
    await flush(client);
    jobService = new JobServiceV2(client);
  });
  afterEach(() => client.quit());

  it('skips jobs whose lock cannot be acquired', async () => {
    await jobService.createJob({ prompt: 'p', userId: 'u' });
    jobService.jobRepo.acquireLock = async () => false;
    const assigned = await jobService.assignJobsToNode('nodeA', 1);
    expect(assigned).toEqual([]);
  });

  it('includes chunks for a running job result', async () => {
    const job = await jobService.createJob({ prompt: 'p', userId: 'u' });
    await jobService.assignJobsToNode('nodeA', 1);
    await jobService.updateJobStatus(job.id, 'running');
    await jobService.storeChunk(job.id, 'nodeA', { chunkIndex: 0, content: 'c' });
    const result = await jobService.getJobResult(job.id);
    expect(result.status).toBe('running');
    expect(result.chunks.length).toBeGreaterThanOrEqual(1);
  });

  it('skips missing jobs when listing by status', async () => {
    await jobService.jobRepo.addToCompletedQueue('ghost-job');
    expect(await jobService.getJobsByStatus('completed')).toEqual([]);
  });

  it('returns a completed job result without attaching chunks', async () => {
    const job = await jobService.createJob({ prompt: 'p', userId: 'u' });
    await jobService.assignJobsToNode('nodeA', 1);
    await jobService.completeJob(job.id, 'nodeA', 'done');
    const result = await jobService.getJobResult(job.id);
    expect(result.status).toBe('completed');
    expect(result.chunks).toBeUndefined();
  });

  it('defaults the attempt counter when a failed job has none', async () => {
    // A failed job stored without an `attempts` field exercises the
    // `(job.attempts || 0)` fallback in retryJob.
    await jobService.jobRepo.createJob({ id: 'no-attempts', prompt: 'p', status: 'failed' });
    await jobService.jobRepo.addToFailedQueue('no-attempts');
    const result = await jobService.retryJob('no-attempts');
    expect(result.success).toBe(true);
  });

  it('uses a default maxJobs when assigning', async () => {
    await jobService.createJob({ prompt: 'p', userId: 'u' });
    const assigned = await jobService.assignJobsToNode('nodeA'); // default maxJobs
    expect(assigned.length).toBe(1);
  });
});

describe('JobService (V1) inline compat defaults and heartbeat/timeout branches', () => {
  it('falls back to empty defaults in the inline sorted-set/ttl compat', async () => {
    const nullClient = {
      zrange: (k, s, e, cb) => cb(null, undefined),
      zcard: (k, cb) => cb(null, undefined),
      zrangebyscore: (k, mn, mx, cb) => cb(null, undefined),
      ttl: (k, cb) => { if (cb) cb(null, undefined); },
    };
    const service = new JobService(nullClient);
    expect(await service.redis.zRange('k', 0, -1)).toEqual([]);
    expect(await service.redis.zCard('k')).toBe(0);
    expect(await service.redis.zRangeByScore('k', 0, 1)).toEqual([]);
    expect(await service.redis.ttl('k')).toBe(-2);
  });

  it('leaves the job status alone when a heartbeat arrives for a non-assigned job', async () => {
    const client = createCamelClient();
    await flush(client);
    const service = new JobService(client);
    const job = await service.createJob({ prompt: 'p', userId: 'u' });
    const assigned = await service.assignJobsToNode('nodeA'); // default maxJobs
    expect(assigned.length).toBe(1);
    await service.handleHeartbeat(job.id, 'nodeA'); // assigned -> running
    // Second heartbeat: status is already running, so the assigned branch is skipped
    const result = await service.handleHeartbeat(job.id, 'nodeA');
    expect(result.success).toBe(true);
    client.quit();
  });

  it('returns timed out jobs that have no assigned node', async () => {
    const client = createCamelClient();
    await flush(client);
    const service = new JobService(client);
    const job = await service.createJob({ prompt: 'p', userId: 'u' });
    // Mark assigned without an assignedTo field and with no lock -> ttl is -2
    await service.updateJobStatus(job.id, 'assigned');
    const timedOut = await service.checkTimeouts();
    expect(timedOut).toContain(job.id);
    client.quit();
  });
});
