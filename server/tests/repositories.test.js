const redis = require('redis-mock');
const { JobRepository, NodeRepository } = require('../src/repositories');
const JobServiceV2 = require('../src/services/jobServiceV2');
const NodeServiceV2 = require('../src/services/nodeServiceV2');

describe('Repository Pattern Implementation', () => {
  let redisClient;
  let jobRepo;
  let nodeRepo;
  let jobService;
  let nodeService;

  beforeEach(async () => {
    redisClient = redis.createClient();
    jobRepo = new JobRepository(redisClient);
    nodeRepo = new NodeRepository(redisClient);
    jobService = new JobServiceV2(redisClient);
    nodeService = new NodeServiceV2(redisClient);
    
    // Clear Redis
    await new Promise(resolve => redisClient.flushall(resolve));
  });

  afterEach(() => {
    redisClient.quit();
  });

  describe('JobRepository', () => {
    it('should create and retrieve a job', async () => {
      const jobData = {
        id: 'test-job-1',
        prompt: 'Test prompt',
        userId: 'user123',
        status: 'pending'
      };

      await jobRepo.createJob(jobData);
      const retrieved = await jobRepo.getJob('test-job-1');

      expect(retrieved.id).toBe('test-job-1');
      expect(retrieved.prompt).toBe('Test prompt');
      expect(retrieved.status).toBe('pending');
    });

    it('should manage job queues', async () => {
      await jobRepo.addToPendingQueue('job1');
      await jobRepo.addToPendingQueue('job2');
      
      const pending = await jobRepo.getPendingJobs(10);
      expect(pending).toContain('job1');
      expect(pending).toContain('job2');

      await jobRepo.removeFromPendingQueue('job1');
      await jobRepo.addToAssignedQueue('job1');
      
      const assigned = await jobRepo.getAssignedJobs();
      expect(assigned).toContain('job1');
    });

    it('should handle job locks', async () => {
      const jobId = 'job-lock-test';
      const nodeId = 'node123';
      
      // Acquire lock
      await jobRepo.acquireLock(jobId, nodeId, 300);
      
      // Check lock
      const hasLock = await jobRepo.checkLock(jobId, nodeId);
      expect(hasLock).toBe(true);
      
      // Wrong node shouldn't have lock
      const wrongNodeHasLock = await jobRepo.checkLock(jobId, 'wrong-node');
      expect(wrongNodeHasLock).toBe(false);
      
      // Release lock
      await jobRepo.releaseLock(jobId, nodeId);
      const hasLockAfterRelease = await jobRepo.checkLock(jobId, nodeId);
      expect(hasLockAfterRelease).toBe(false);
    });

    it('should store and retrieve chunks', async () => {
      const jobId = 'chunk-test';
      
      await jobRepo.storeChunk(jobId, 0, 'Chunk 1');
      await jobRepo.storeChunk(jobId, 1, 'Chunk 2');
      
      const chunks = await jobRepo.getChunks(jobId);
      
      expect(chunks).toHaveLength(2);
      expect(chunks[0].content).toBe('Chunk 1');
      expect(chunks[1].content).toBe('Chunk 2');
    });

    it('should get queue statistics', async () => {
      await jobRepo.addToPendingQueue('job1');
      await jobRepo.addToAssignedQueue('job2');
      await jobRepo.addToCompletedQueue('job3');
      await jobRepo.addToFailedQueue('job4');
      
      const stats = await jobRepo.getQueueStats();
      
      expect(stats.pending).toBe(1);
      expect(stats.assigned).toBe(1);
      expect(stats.completed).toBe(1);
      expect(stats.failed).toBe(1);
    });
  });

  describe('NodeRepository', () => {
    it('should create and retrieve a node', async () => {
      const nodeData = {
        nodeId: 'test-node-1',
        publicKey: 'public-key-123',
        name: 'Test Node',
        userId: 'user123',
        status: 'online',
        isPublic: false
      };

      await nodeRepo.createNode(nodeData);
      const retrieved = await nodeRepo.getNode('test-node-1');

      expect(retrieved.nodeId).toBe('test-node-1');
      expect(retrieved.name).toBe('Test Node');
      expect(retrieved.status).toBe('online');
    });

    it('should manage user nodes', async () => {
      const node1 = {
        nodeId: 'node1',
        userId: 'user123',
        name: 'Node 1',
        status: 'online',
        lastSeen: Date.now()
      };
      
      const node2 = {
        nodeId: 'node2',
        userId: 'user123',
        name: 'Node 2',
        status: 'offline',
        lastSeen: Date.now()
      };

      await nodeRepo.createNode(node1);
      await nodeRepo.createNode(node2);
      
      const userNodes = await nodeRepo.getUserNodes('user123');
      expect(userNodes).toHaveLength(2);
      expect(userNodes.map(n => n.nodeId)).toContain('node1');
      expect(userNodes.map(n => n.nodeId)).toContain('node2');
    });

    it('should manage public nodes', async () => {
      const publicNode = {
        nodeId: 'public-node',
        name: 'Public Node',
        isPublic: true,
        status: 'online',
        lastSeen: Date.now()
      };
      
      const privateNode = {
        nodeId: 'private-node',
        name: 'Private Node',
        isPublic: false,
        status: 'online',
        lastSeen: Date.now()
      };

      await nodeRepo.createNode(publicNode);
      await nodeRepo.createNode(privateNode);
      
      const publicNodes = await nodeRepo.getPublicNodes();
      expect(publicNodes).toHaveLength(1);
      expect(publicNodes[0].nodeId).toBe('public-node');
    });

    it('should update node visibility', async () => {
      const node = {
        nodeId: 'visibility-test',
        userId: 'user123',
        isPublic: false,
        status: 'online',
        lastSeen: Date.now()
      };
      
      await nodeRepo.createNode(node);
      
      // Update to public
      await nodeRepo.updateNodeVisibility('visibility-test', 'user123', true);
      
      const isPublic = await nodeRepo.isPublicNode('visibility-test');
      expect(isPublic).toBe(true);
    });

    it('should generate consistent node fingerprints', () => {
      const publicKey = 'test-public-key';
      const fingerprint1 = nodeRepo.generateNodeFingerprint(publicKey);
      const fingerprint2 = nodeRepo.generateNodeFingerprint(publicKey);
      
      expect(fingerprint1).toBe(fingerprint2);
      expect(fingerprint1).toHaveLength(6);
    });
  });

  describe('JobServiceV2', () => {
    it('should create and manage jobs using repository', async () => {
      const job = await jobService.createJob({
        prompt: 'Test prompt',
        userId: 'user123',
        model: 'llama3.2:3b'
      });

      expect(job.id).toBeDefined();
      expect(job.status).toBe('pending');
      
      const retrieved = await jobService.getJob(job.id);
      expect(retrieved.prompt).toBe('Test prompt');
    });

    it('should assign jobs to nodes', async () => {
      // Create jobs
      await jobService.createJob({ prompt: 'Job 1', userId: 'user123' });
      await jobService.createJob({ prompt: 'Job 2', userId: 'user123' });
      
      // Assign to node
      const assigned = await jobService.assignJobsToNode('node123', 2);
      
      expect(assigned).toHaveLength(2);
      expect(assigned[0].status).toBe('assigned');
      expect(assigned[0].assignedTo).toBe('node123');
    });

    it('should handle job completion', async () => {
      const job = await jobService.createJob({ prompt: 'Test', userId: 'user123' });
      
      // Assign job
      await jobService.assignJobsToNode('node123', 1);
      
      // Complete job
      const completed = await jobService.completeJob(job.id, 'node123', 'Result');
      
      expect(completed.status).toBe('completed');
      expect(completed.result).toBe('Result');
    });

    it('should handle job failure', async () => {
      const job = await jobService.createJob({ prompt: 'Test', userId: 'user123' });
      
      // Assign job
      await jobService.assignJobsToNode('node123', 1);
      
      // Fail job
      const failed = await jobService.failJob(job.id, 'node123', 'Error occurred');
      
      expect(failed.status).toBe('failed');
      expect(failed.failureReason).toBe('Error occurred');
    });

    it('should get queue statistics', async () => {
      await jobService.createJob({ prompt: 'Job 1', userId: 'user123' });
      await jobService.createJob({ prompt: 'Job 2', userId: 'user123' });
      
      const stats = await jobService.getQueueStats();
      
      expect(stats.pending).toBe(2);
      expect(stats.assigned).toBe(0);
      expect(stats.completed).toBe(0);
      expect(stats.failed).toBe(0);
    });
  });

  describe('NodeServiceV2', () => {
    it('should claim nodes using repository', async () => {
      const result = await nodeService.claimNode('public-key-123', 'My Node', 'user123');
      
      expect(result.success).toBe(true);
      expect(result.nodeId).toBeDefined();
      expect(result.status).toBe('online');
    });

    it('should update node status', async () => {
      const claimResult = await nodeService.claimNode('public-key-456', 'Node 2', 'user456');
      
      const updateResult = await nodeService.updateNodeStatus(
        claimResult.nodeId,
        'public-key-456',
        { capabilities: { gpu: true } }
      );
      
      expect(updateResult.success).toBe(true);
      expect(updateResult.status).toBe('online');
    });

    it('should manage node visibility', async () => {
      const claimResult = await nodeService.claimNode('public-key-789', 'Node 3', 'user789');
      
      const visibilityResult = await nodeService.updateNodeVisibility(
        claimResult.nodeId,
        'user789',
        true
      );
      
      expect(visibilityResult.success).toBe(true);
      expect(visibilityResult.isPublic).toBe(true);
    });

    it('should get user nodes', async () => {
      await nodeService.claimNode('key1', 'Node 1', 'user999');
      await nodeService.claimNode('key2', 'Node 2', 'user999');
      
      const userNodes = await nodeService.getUserNodes('user999');
      
      expect(userNodes).toHaveLength(2);
      expect(userNodes.map(n => n.name)).toContain('Node 1');
      expect(userNodes.map(n => n.name)).toContain('Node 2');
    });

    it('should get public nodes', async () => {
      const claim1 = await nodeService.claimNode('pub-key1', 'Public Node', 'user111');
      await nodeService.updateNodeVisibility(claim1.nodeId, 'user111', true);
      
      const claim2 = await nodeService.claimNode('priv-key1', 'Private Node', 'user222');
      
      const publicNodes = await nodeService.getPublicNodes();
      
      expect(publicNodes.nodes).toHaveLength(1);
      expect(publicNodes.nodes[0].name).toBe('Public Node');
    });
  });

  describe('Repository Pattern Benefits', () => {
    it('should have consistent Redis operations across services', () => {
      // Both repositories use the same base methods
      expect(jobRepo.get).toBeDefined();
      expect(nodeRepo.get).toBeDefined();
      expect(jobRepo.set).toBeDefined();
      expect(nodeRepo.set).toBeDefined();
      
      // Consistent key prefixing
      expect(jobRepo.getKey('123')).toBe('job:123');
      expect(nodeRepo.getKey('abc')).toBe('node:abc');
    });

    it('should handle Redis compatibility automatically', () => {
      // All Redis operations are abstracted
      expect(jobRepo.redis.zAdd).toBeDefined();
      expect(jobRepo.redis.hSet).toBeDefined();
      expect(nodeRepo.redis.sAdd).toBeDefined();
      expect(nodeRepo.redis.setEx).toBeDefined();
    });

    it('should reduce code duplication', () => {
      // Services no longer need Redis compatibility code
      expect(jobService.jobRepo).toBeDefined();
      expect(nodeService.nodeRepo).toBeDefined();
      
      // Services focus on business logic
      expect(jobService.setupRedisCompat).toBeUndefined();
      expect(nodeService.setupRedisCompat).toBeUndefined();
    });
  });
});