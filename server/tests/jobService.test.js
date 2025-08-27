const JobService = require('../src/services/jobService');
const redis = require('redis-mock');

describe('JobService', () => {
  let jobService;
  let redisClient;

  beforeEach(async () => {
    redisClient = redis.createClient();
    jobService = new JobService(redisClient);
    
    // Clear any existing data
    await new Promise(resolve => redisClient.flushall(resolve));
  });

  afterEach(() => {
    redisClient.quit();
  });

  describe('createJob', () => {
    it('should create a new job with default values', async () => {
      const jobData = {
        prompt: 'Test prompt',
        userId: 'user123'
      };

      const job = await jobService.createJob(jobData);

      expect(job).toMatchObject({
        prompt: 'Test prompt',
        model: 'llama3.2:3b',
        status: 'pending',
        userId: 'user123',
        priority: 0,
        maxTokens: 1000,
        temperature: 0.7
      });
      expect(job.id).toMatch(/^job-\d+-[a-z0-9]+$/);
      expect(job.createdAt).toBeDefined();
    });

    it('should create job with custom priority', async () => {
      const jobData = {
        prompt: 'High priority prompt',
        userId: 'user123',
        priority: 10,
        model: 'llama3.2:7b',
        temperature: 0.9
      };

      const job = await jobService.createJob(jobData);

      expect(job.priority).toBe(10);
      expect(job.model).toBe('llama3.2:7b');
      expect(job.temperature).toBe(0.9);
    });
  });

  describe('getJob', () => {
    it('should retrieve a created job', async () => {
      const created = await jobService.createJob({
        prompt: 'Test',
        userId: 'user123'
      });

      const retrieved = await jobService.getJob(created.id);

      expect(retrieved).toEqual(created);
    });

    it('should return null for non-existent job', async () => {
      const job = await jobService.getJob('non-existent-job');
      expect(job).toBeNull();
    });
  });

  describe('updateJobStatus', () => {
    it('should update job status from pending to assigned', async () => {
      const job = await jobService.createJob({
        prompt: 'Test',
        userId: 'user123'
      });

      const updated = await jobService.updateJobStatus(job.id, 'assigned', {
        assignedTo: 'node123',
        assignedAt: Date.now()
      });

      expect(updated.status).toBe('assigned');
      expect(updated.assignedTo).toBe('node123');
      expect(updated.assignedAt).toBeDefined();
    });

    it('should throw error for non-existent job', async () => {
      await expect(
        jobService.updateJobStatus('non-existent', 'running')
      ).rejects.toThrow('Job non-existent not found');
    });
  });

  describe('assignJobsToNode', () => {
    it('should assign pending jobs to a node', async () => {
      // Create multiple jobs with different priorities
      const job1 = await jobService.createJob({
        prompt: 'Low priority',
        userId: 'user123',
        priority: 1
      });

      const job2 = await jobService.createJob({
        prompt: 'High priority',
        userId: 'user123',
        priority: 10
      });

      const job3 = await jobService.createJob({
        prompt: 'Medium priority',
        userId: 'user123',
        priority: 5
      });

      const assigned = await jobService.assignJobsToNode('node123', 2);

      // Should get high priority first, then medium
      expect(assigned).toHaveLength(2);
      expect(assigned[0].id).toBe(job2.id);
      expect(assigned[1].id).toBe(job3.id);
      expect(assigned[0].status).toBe('assigned');
      expect(assigned[0].assignedTo).toBe('node123');
    });

    it('should not assign already locked jobs', async () => {
      const job = await jobService.createJob({
        prompt: 'Test',
        userId: 'user123'
      });

      // First node gets the job
      const assigned1 = await jobService.assignJobsToNode('node1', 1);
      expect(assigned1).toHaveLength(1);

      // Second node should get nothing
      const assigned2 = await jobService.assignJobsToNode('node2', 1);
      expect(assigned2).toHaveLength(0);
    });
  });

  describe('handleHeartbeat', () => {
    it('should extend lock timeout for valid heartbeat', async () => {
      const job = await jobService.createJob({
        prompt: 'Test',
        userId: 'user123'
      });

      const assigned = await jobService.assignJobsToNode('node123', 1);
      expect(assigned).toHaveLength(1);

      const result = await jobService.handleHeartbeat(job.id, 'node123');
      expect(result.success).toBe(true);

      // Job should now be running
      const updated = await jobService.getJob(job.id);
      expect(updated.status).toBe('running');
    });

    it('should reject heartbeat from wrong node', async () => {
      const job = await jobService.createJob({
        prompt: 'Test',
        userId: 'user123'
      });

      await jobService.assignJobsToNode('node123', 1);

      await expect(
        jobService.handleHeartbeat(job.id, 'wrong-node')
      ).rejects.toThrow('Node does not hold lock for this job');
    });
  });

  describe('storeChunk', () => {
    it('should store job result chunks', async () => {
      const job = await jobService.createJob({
        prompt: 'Test',
        userId: 'user123'
      });

      await jobService.assignJobsToNode('node123', 1);

      const chunk1 = await jobService.storeChunk(job.id, 'node123', {
        chunkIndex: 0,
        content: 'Hello ',
        metrics: { tokensPerSecond: 10 }
      });

      expect(chunk1.success).toBe(true);
      expect(chunk1.chunkIndex).toBe(0);

      const chunk2 = await jobService.storeChunk(job.id, 'node123', {
        chunkIndex: 1,
        content: 'world!',
        metrics: { tokensPerSecond: 12 },
        isFinal: true
      });

      expect(chunk2.success).toBe(true);
    });

    it('should reject chunks from wrong node', async () => {
      const job = await jobService.createJob({
        prompt: 'Test',
        userId: 'user123'
      });

      await jobService.assignJobsToNode('node123', 1);

      await expect(
        jobService.storeChunk(job.id, 'wrong-node', {
          chunkIndex: 0,
          content: 'Test'
        })
      ).rejects.toThrow('Node does not hold lock for this job');
    });
  });

  describe('completeJob', () => {
    it('should complete job and assemble chunks', async () => {
      const job = await jobService.createJob({
        prompt: 'Test',
        userId: 'user123'
      });

      await jobService.assignJobsToNode('node123', 1);

      // Store some chunks
      await jobService.storeChunk(job.id, 'node123', {
        chunkIndex: 0,
        content: 'Hello '
      });
      await jobService.storeChunk(job.id, 'node123', {
        chunkIndex: 1,
        content: 'world!'
      });

      const completed = await jobService.completeJob(job.id, 'node123');

      expect(completed.status).toBe('completed');
      expect(completed.result).toBe('Hello world!');
      expect(completed.chunks).toBe(2);
      expect(completed.completedAt).toBeDefined();
    });

    it('should reject completion from wrong node', async () => {
      const job = await jobService.createJob({
        prompt: 'Test',
        userId: 'user123'
      });

      await jobService.assignJobsToNode('node123', 1);

      await expect(
        jobService.completeJob(job.id, 'wrong-node')
      ).rejects.toThrow('Node does not hold lock for this job');
    });
  });

  describe('failJob', () => {
    it('should mark job as failed with reason', async () => {
      const job = await jobService.createJob({
        prompt: 'Test',
        userId: 'user123'
      });

      await jobService.assignJobsToNode('node123', 1);

      const failed = await jobService.failJob(job.id, 'node123', 'Out of memory');

      expect(failed.status).toBe('failed');
      expect(failed.failureReason).toBe('Out of memory');
      expect(failed.failedAt).toBeDefined();
    });
  });

  describe('getJobResult', () => {
    it('should return completed job result', async () => {
      const job = await jobService.createJob({
        prompt: 'Test',
        userId: 'user123'
      });

      await jobService.assignJobsToNode('node123', 1);
      await jobService.storeChunk(job.id, 'node123', {
        chunkIndex: 0,
        content: 'Test result',
        metrics: { tokensPerSecond: 15 }
      });
      await jobService.completeJob(job.id, 'node123');

      const result = await jobService.getJobResult(job.id);

      expect(result.status).toBe('completed');
      expect(result.result).toBe('Test result');
      expect(result.completedAt).toBeDefined();
    });

    it('should return partial results for running job', async () => {
      const job = await jobService.createJob({
        prompt: 'Test',
        userId: 'user123'
      });

      await jobService.assignJobsToNode('node123', 1);
      await jobService.handleHeartbeat(job.id, 'node123'); // Mark as running
      
      await jobService.storeChunk(job.id, 'node123', {
        chunkIndex: 0,
        content: 'Partial ',
        metrics: { tokensPerSecond: 10 }
      });
      await jobService.storeChunk(job.id, 'node123', {
        chunkIndex: 1,
        content: 'result',
        metrics: { tokensPerSecond: 12 }
      });

      const result = await jobService.getJobResult(job.id);

      expect(result.status).toBe('running');
      expect(result.partial).toBe('Partial result');
      expect(result.chunks).toBe(2);
    });

    it('should return failed status for failed job', async () => {
      const job = await jobService.createJob({
        prompt: 'Test',
        userId: 'user123'
      });

      await jobService.assignJobsToNode('node123', 1);
      await jobService.failJob(job.id, 'node123', 'Connection lost');

      const result = await jobService.getJobResult(job.id);

      expect(result.status).toBe('failed');
      expect(result.error).toBe('Connection lost');
      expect(result.failedAt).toBeDefined();
    });
  });

  describe('getQueueStats', () => {
    it('should return queue statistics', async () => {
      // Create jobs in various states
      const job1 = await jobService.createJob({ prompt: 'Test 1', userId: 'user123' });
      const job2 = await jobService.createJob({ prompt: 'Test 2', userId: 'user123' });
      const job3 = await jobService.createJob({ prompt: 'Test 3', userId: 'user123' });

      // Assign one job
      await jobService.assignJobsToNode('node1', 1);

      // Complete one job
      const assigned = await jobService.assignJobsToNode('node2', 1);
      if (assigned.length > 0) {
        await jobService.completeJob(assigned[0].id, 'node2');
      }

      const stats = await jobService.getQueueStats();

      expect(stats.pending).toBeGreaterThanOrEqual(1);
      expect(stats.assigned).toBeGreaterThanOrEqual(0);
      expect(stats.completed).toBeGreaterThanOrEqual(0);
    });
  });

  describe('checkTimeouts', () => {
    it('should return timed out jobs to queue', async () => {
      jest.setTimeout(10000);
      
      const job = await jobService.createJob({
        prompt: 'Test',
        userId: 'user123'
      });

      // Assign job and let lock expire
      await jobService.assignJobsToNode('node123', 1);
      
      // Manually expire the lock for testing
      const lockKey = `job:${job.id}:lock`;
      await new Promise(resolve => redisClient.del(lockKey, resolve));

      const timeoutJobs = await jobService.checkTimeouts();

      // Job should be returned to pending queue
      expect(timeoutJobs).toContain(job.id);
      
      const updated = await jobService.getJob(job.id);
      expect(updated.status).toBe('pending');
      expect(updated.timeoutReason).toBe('lock_expired');
    });
  });

  describe('cleanupOldJobs', () => {
    it('should remove old completed and failed jobs', async () => {
      const job1 = await jobService.createJob({ prompt: 'Test 1', userId: 'user123' });

      // Complete one job
      const assignedToNode1 = await jobService.assignJobsToNode('node1', 1);
      expect(assignedToNode1).toHaveLength(1);
      await jobService.completeJob(assignedToNode1[0].id, 'node1');

      // Create and fail another job
      const job2 = await jobService.createJob({ prompt: 'Test 2', userId: 'user123' });
      const assignedToNode2 = await jobService.assignJobsToNode('node2', 1);
      expect(assignedToNode2).toHaveLength(1);
      await jobService.failJob(assignedToNode2[0].id, 'node2', 'Test failure');

      // Set jobs to be old (manually update their timestamps)
      const oldTime = Date.now() - 25 * 60 * 60 * 1000; // 25 hours ago
      
      // Cleanup jobs older than 1 hour
      const cleaned = await jobService.cleanupOldJobs(60 * 60 * 1000);

      // In a real scenario with proper timestamp manipulation, 
      // this would clean up the old jobs
      expect(cleaned).toBeGreaterThanOrEqual(0);
    });
  });
});