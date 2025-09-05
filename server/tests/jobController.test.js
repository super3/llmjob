const JobController = require('../src/controllers/jobController');
const JobService = require('../src/services/jobService');
const redis = require('redis-mock');

describe('JobController', () => {
  let jobController;
  let jobService;
  let nodeService;
  let redisClient;
  let req, res;
  let consoleErrorSpy;

  beforeEach(async () => {
    // Mock console.error for error tests
    consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation();
    redisClient = redis.createClient();
    jobService = new JobService(redisClient);
    
    // Mock nodeService
    nodeService = {
      getNode: jest.fn()
    };
    
    jobController = new JobController(jobService, nodeService, redisClient);
    
    // Clear any existing data
    await new Promise(resolve => redisClient.flushall(resolve));
    
    // Setup request and response mocks
    req = {
      body: {},
      params: {},
      user: { id: 'user123' },
      verifiedNode: { nodeId: 'node123', publicKey: 'test-public-key' }
    };
    
    res = {
      status: jest.fn().mockReturnThis(),
      json: jest.fn(),
      send: jest.fn()
    };
  });

  afterEach(() => {
    redisClient.quit();
    jest.clearAllMocks();
    consoleErrorSpy.mockRestore();
  });

  describe('submitJob', () => {
    it('should submit a new job successfully', async () => {
      req.body = {
        prompt: 'Test prompt',
        model: 'llama3.2:3b',
        priority: 5
      };

      await jobController.submitJob(req, res);

      expect(res.status).toHaveBeenCalledWith(201);
      expect(res.json).toHaveBeenCalledWith({
        success: true,
        job: expect.objectContaining({
          prompt: 'Test prompt',
          model: 'llama3.2:3b',
          priority: 5,
          status: 'pending',
          userId: 'user123'
        })
      });
    });

    it('should reject job without prompt', async () => {
      req.body = { model: 'llama3.2:3b' };

      await jobController.submitJob(req, res);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({ error: 'Prompt is required' });
    });
  });

  describe('pollJobs', () => {
    beforeEach(async () => {
      // Create some test jobs
      await jobService.createJob({ prompt: 'Test 1', userId: 'user123' });
      await jobService.createJob({ prompt: 'Test 2', userId: 'user123' });
    });

    it('should assign jobs to valid node', async () => {
      nodeService.getNode.mockResolvedValue({
        nodeId: 'node123',
        publicKey: 'test-public-key',
        status: 'online'
      });

      req.body = {
        nodeId: 'node123',
        signature: 'test-signature',
        timestamp: Date.now(),
        maxJobs: 2
      };

      await jobController.pollJobs(req, res);

      expect(nodeService.getNode).toHaveBeenCalledWith('node123', redisClient);
      expect(res.json).toHaveBeenCalledWith({
        success: true,
        jobs: expect.arrayContaining([
          expect.objectContaining({
            prompt: expect.any(String),
            model: 'llama3.2:3b'
          })
        ])
      });
    });

    it('should reject polling from unknown node', async () => {
      nodeService.getNode.mockResolvedValue(null);

      req.body = {
        nodeId: 'unknown-node',
        signature: 'test-signature',
        timestamp: Date.now(),
        maxJobs: 1
      };

      await jobController.pollJobs(req, res);

      expect(res.status).toHaveBeenCalledWith(404);
      expect(res.json).toHaveBeenCalledWith({ error: 'Node not found' });
    });
  });

  describe('heartbeat', () => {
    let job;

    beforeEach(async () => {
      job = await jobService.createJob({ prompt: 'Test', userId: 'user123' });
      await jobService.assignJobsToNode('node123', 1);
    });

    it('should handle heartbeat from valid node', async () => {
      nodeService.getNode.mockResolvedValue({
        nodeId: 'node123',
        publicKey: 'test-public-key'
      });

      req.params = { jobId: job.id };
      req.body = {
        nodeId: 'node123',
        signature: 'test-signature',
        timestamp: Date.now(),
        status: 'running'
      };

      await jobController.heartbeat(req, res);

      expect(res.json).toHaveBeenCalledWith({
        success: true,
        timestamp: expect.any(Number)
      });
    });

    it('should reject heartbeat from unknown node', async () => {
      nodeService.getNode.mockResolvedValue(null);

      req.params = { jobId: job.id };
      req.body = {
        nodeId: 'unknown-node',
        signature: 'test-signature',
        timestamp: Date.now()
      };

      await jobController.heartbeat(req, res);

      expect(res.status).toHaveBeenCalledWith(404);
      expect(res.json).toHaveBeenCalledWith({ error: 'Node not found' });
    });
  });

  describe('receiveChunk', () => {
    let job;

    beforeEach(async () => {
      job = await jobService.createJob({ prompt: 'Test', userId: 'user123' });
      await jobService.assignJobsToNode('node123', 1);
    });

    it('should receive and store chunk', async () => {
      nodeService.getNode.mockResolvedValue({
        nodeId: 'node123',
        publicKey: 'test-public-key'
      });

      req.params = { jobId: job.id };
      req.body = {
        nodeId: 'node123',
        signature: 'test-signature',
        timestamp: Date.now(),
        chunkIndex: 0,
        content: 'Test chunk',
        metrics: { tokensPerSecond: 10 }
      };

      await jobController.receiveChunk(req, res);

      expect(res.json).toHaveBeenCalledWith({
        success: true,
        chunkIndex: 0
      });
    });

    it('should reject chunk from wrong node', async () => {
      nodeService.getNode.mockResolvedValue({
        nodeId: 'wrong-node',
        publicKey: 'wrong-key'
      });

      req.params = { jobId: job.id };
      req.body = {
        nodeId: 'wrong-node',
        chunkIndex: 0,
        content: 'Test chunk'
      };

      await jobController.receiveChunk(req, res);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({
        error: expect.stringContaining('Node does not hold lock')
      });
    });

    it('should handle chunk with non-existent node', async () => {
      nodeService.getNode.mockResolvedValue(null);

      req.params = { jobId: job.id };
      req.body = {
        nodeId: 'non-existent-node',
        chunkIndex: 0,
        content: 'Test chunk'
      };

      await jobController.receiveChunk(req, res);

      expect(res.status).toHaveBeenCalledWith(404);
      expect(res.json).toHaveBeenCalledWith({ error: 'Node not found' });
    });
  });

  describe('completeJob', () => {
    let job;

    beforeEach(async () => {
      job = await jobService.createJob({ prompt: 'Test', userId: 'user123' });
      await jobService.assignJobsToNode('node123', 1);
      await jobService.storeChunk(job.id, 'node123', {
        chunkIndex: 0,
        content: 'Complete result'
      });
    });

    it('should complete job successfully', async () => {
      nodeService.getNode.mockResolvedValue({
        nodeId: 'node123',
        publicKey: 'test-public-key'
      });

      req.params = { jobId: job.id };
      req.body = {
        nodeId: 'node123',
        signature: 'test-signature',
        timestamp: Date.now()
      };

      await jobController.completeJob(req, res);

      expect(res.json).toHaveBeenCalledWith({
        success: true,
        job: expect.objectContaining({
          status: 'completed',
          result: 'Complete result'
        })
      });
    });

    it('should handle complete with non-existent node', async () => {
      nodeService.getNode.mockResolvedValue(null);

      req.params = { jobId: job.id };
      req.body = {
        nodeId: 'non-existent-node',
        signature: 'test-signature',
        timestamp: Date.now()
      };

      await jobController.completeJob(req, res);

      expect(res.status).toHaveBeenCalledWith(404);
      expect(res.json).toHaveBeenCalledWith({ error: 'Node not found' });
    });
  });

  describe('failJob', () => {
    let job;

    beforeEach(async () => {
      job = await jobService.createJob({ prompt: 'Test', userId: 'user123' });
      await jobService.assignJobsToNode('node123', 1);
    });

    it('should mark job as failed', async () => {
      nodeService.getNode.mockResolvedValue({
        nodeId: 'node123',
        publicKey: 'test-public-key'
      });

      req.params = { jobId: job.id };
      req.body = {
        nodeId: 'node123',
        signature: 'test-signature',
        timestamp: Date.now(),
        error: 'Out of memory'
      };

      await jobController.failJob(req, res);

      expect(res.json).toHaveBeenCalledWith({
        success: true,
        job: expect.objectContaining({
          status: 'failed',
          failureReason: 'Out of memory'
        })
      });
    });

    it('should handle fail with non-existent node', async () => {
      nodeService.getNode.mockResolvedValue(null);

      req.params = { jobId: job.id };
      req.body = {
        nodeId: 'non-existent-node',
        error: 'Test error'
      };

      await jobController.failJob(req, res);

      expect(res.status).toHaveBeenCalledWith(404);
      expect(res.json).toHaveBeenCalledWith({ error: 'Node not found' });
    });
  });

  describe('getJob', () => {
    it('should return job result', async () => {
      const job = await jobService.createJob({ prompt: 'Test', userId: 'user123' });
      
      req.params = { jobId: job.id };

      await jobController.getJob(req, res);

      expect(res.json).toHaveBeenCalledWith({
        success: true,
        jobId: job.id,
        status: 'pending',
        createdAt: expect.any(Number)
      });
    });

    it('should handle non-existent job', async () => {
      req.params = { jobId: 'non-existent' };

      await jobController.getJob(req, res);

      expect(res.status).toHaveBeenCalledWith(404);
      expect(res.json).toHaveBeenCalledWith({
        error: expect.stringContaining('not found')
      });
    });
  });

  describe('getStats', () => {
    it('should return queue statistics', async () => {
      // Create jobs in various states
      await jobService.createJob({ prompt: 'Test 1', userId: 'user123' });
      await jobService.createJob({ prompt: 'Test 2', userId: 'user123' });

      await jobController.getStats(req, res);

      expect(res.json).toHaveBeenCalledWith({
        success: true,
        stats: expect.objectContaining({
          pending: expect.any(Number),
          assigned: expect.any(Number),
          running: expect.any(Number),
          completed: expect.any(Number),
          failed: expect.any(Number)
        })
      });
    });
  });

  describe('checkTimeouts', () => {
    it('should check and return timed out jobs', async () => {
      await jobController.checkTimeouts(req, res);

      expect(res.json).toHaveBeenCalledWith({
        success: true,
        timeoutJobs: expect.any(Array)
      });
    });
  });

  describe('cleanupJobs', () => {
    it('should cleanup old jobs', async () => {
      req.body = { maxAge: 86400000 }; // 24 hours

      await jobController.cleanupJobs(req, res);

      expect(res.json).toHaveBeenCalledWith({
        success: true,
        cleaned: expect.any(Number)
      });
    });
  });

  // Additional error case tests for complete coverage
  describe('Error handling', () => {
    it('should handle submitJob error', async () => {
      req.body = { prompt: 'Test' };
      jobService.createJob = jest.fn().mockRejectedValue(new Error('Database error'));

      await jobController.submitJob(req, res);

      expect(console.error).toHaveBeenCalledWith('Error submitting job:', expect.any(Error));
      expect(res.status).toHaveBeenCalledWith(500);
      expect(res.json).toHaveBeenCalledWith({ error: 'Failed to submit job' });
    });

    it('should handle pollJobs error when getNode throws', async () => {
      req.body = { nodeId: 'node123', maxJobs: 1 };
      nodeService.getNode = jest.fn().mockRejectedValue(new Error('Database error'));

      await jobController.pollJobs(req, res);

      expect(console.error).toHaveBeenCalledWith('Error polling jobs:', expect.any(Error));
      expect(res.status).toHaveBeenCalledWith(500);
      expect(res.json).toHaveBeenCalledWith({ error: 'Failed to poll jobs' });
    });

    it('should handle heartbeat error when getNode throws', async () => {
      req.params = { jobId: 'job123' };
      req.body = { nodeId: 'node123', status: 'running', activeJobs: 1 };
      nodeService.getNode = jest.fn().mockRejectedValue(new Error('Database error'));

      await jobController.heartbeat(req, res);

      expect(console.error).toHaveBeenCalledWith('Error handling heartbeat:', expect.any(Error));
      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({ error: 'Database error' });
    });

    it('should handle completeJob error when jobService throws', async () => {
      req.params = { jobId: 'job123' };
      req.body = { nodeId: 'node123', finalOutput: 'result' };
      nodeService.getNode = jest.fn().mockResolvedValue({ id: 'node123', publicKey: 'test-key' });
      jobService.completeJob = jest.fn().mockRejectedValue(new Error('Database error'));

      await jobController.completeJob(req, res);

      expect(console.error).toHaveBeenCalledWith('Error completing job:', expect.any(Error));
      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({ error: 'Database error' });
    });

    it('should handle failJob error when jobService throws', async () => {
      req.params = { jobId: 'job123' };
      req.body = { nodeId: 'node123', error: 'Job failed' };
      nodeService.getNode = jest.fn().mockResolvedValue({ id: 'node123', publicKey: 'test-key' });
      jobService.failJob = jest.fn().mockRejectedValue(new Error('Database error'));

      await jobController.failJob(req, res);

      expect(console.error).toHaveBeenCalledWith('Error failing job:', expect.any(Error));
      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({ error: 'Database error' });
    });

    it('should handle getStats error', async () => {
      jobService.getQueueStats = jest.fn().mockRejectedValue(new Error('Database error'));

      await jobController.getStats(req, res);

      expect(console.error).toHaveBeenCalledWith('Error getting stats:', expect.any(Error));
      expect(res.status).toHaveBeenCalledWith(500);
      expect(res.json).toHaveBeenCalledWith({ error: 'Failed to get queue statistics' });
    });

    it('should handle cleanupJobs error', async () => {
      req.body = { maxAge: 86400000 };
      jobService.cleanupOldJobs = jest.fn().mockRejectedValue(new Error('Database error'));

      await jobController.cleanupJobs(req, res);

      expect(console.error).toHaveBeenCalledWith('Error cleaning up jobs:', expect.any(Error));
      expect(res.status).toHaveBeenCalledWith(500);
      expect(res.json).toHaveBeenCalledWith({ error: 'Failed to cleanup jobs' });
    });

    it('should handle checkTimeouts error', async () => {
      jobService.checkTimeouts = jest.fn().mockRejectedValue(new Error('Database error'));

      await jobController.checkTimeouts(req, res);

      expect(console.error).toHaveBeenCalledWith('Error checking timeouts:', expect.any(Error));
      expect(res.status).toHaveBeenCalledWith(500);
      expect(res.json).toHaveBeenCalledWith({ error: 'Failed to check timeouts' });
    });
  });
});