class JobController {
  constructor(jobService, nodeService, redis) {
    this.jobService = jobService;
    this.nodeService = nodeService;
    this.redis = redis;
  }

  // POST /api/jobs - Submit a new job
  async submitJob(req, res) {
    try {
      const { prompt, model, options, priority, maxTokens, temperature } = req.body;
      
      // Validate required fields
      if (!prompt) {
        return res.status(400).json({ error: 'Prompt is required' });
      }

      // Get user ID from auth middleware
      const userId = req.user?.id || 'anonymous';

      const job = await this.jobService.createJob({
        prompt,
        model,
        options,
        priority,
        maxTokens,
        temperature,
        userId
      });

      res.status(201).json({
        success: true,
        job
      });
    } catch (error) {
      console.error('Error submitting job:', error);
      res.status(500).json({ error: 'Failed to submit job' });
    }
  }

  // POST /api/jobs/poll - Poll for available jobs (called by nodes)
  async pollJobs(req, res) {
    try {
      const { nodeId, signature, timestamp, maxJobs } = req.body;

      // Verify node exists and is active
      const node = await this.nodeService.getNode(nodeId, this.redis);
      if (!node) {
        return res.status(404).json({ error: 'Node not found' });
      }

      // Assign jobs to node
      const jobs = await this.jobService.assignJobsToNode(nodeId, maxJobs || 1);

      res.json({
        success: true,
        jobs: jobs.map(job => ({
          id: job.id,
          prompt: job.prompt,
          model: job.model,
          options: job.options,
          maxTokens: job.maxTokens,
          temperature: job.temperature
        }))
      });
    } catch (error) {
      console.error('Error polling jobs:', error);
      res.status(500).json({ error: 'Failed to poll jobs' });
    }
  }

  // POST /api/jobs/:jobId/heartbeat - Receive heartbeat from node
  async heartbeat(req, res) {
    try {
      const { jobId } = req.params;
      const { nodeId, signature, timestamp, status, activeJobs } = req.body;

      // Verify node
      const node = await this.nodeService.getNode(nodeId, this.redis);
      if (!node) {
        return res.status(404).json({ error: 'Node not found' });
      }

      // Handle heartbeat
      await this.jobService.handleHeartbeat(jobId, nodeId);

      res.json({
        success: true,
        timestamp: Date.now()
      });
    } catch (error) {
      console.error('Error handling heartbeat:', error);
      res.status(400).json({ error: error.message });
    }
  }

  // POST /api/jobs/:jobId/chunks - Receive job result chunks
  async receiveChunk(req, res) {
    try {
      const { jobId } = req.params;
      const { nodeId, signature, timestamp, chunkIndex, content, metrics, isFinal } = req.body;

      // Verify node
      const node = await this.nodeService.getNode(nodeId, this.redis);
      if (!node) {
        return res.status(404).json({ error: 'Node not found' });
      }

      // Store chunk
      const result = await this.jobService.storeChunk(jobId, nodeId, {
        chunkIndex,
        content,
        metrics,
        isFinal,
        timestamp
      });

      res.json({
        success: true,
        ...result
      });
    } catch (error) {
      console.error('Error receiving chunk:', error);
      res.status(400).json({ error: error.message });
    }
  }

  // POST /api/jobs/:jobId/complete - Mark job as completed
  async completeJob(req, res) {
    try {
      const { jobId } = req.params;
      const { nodeId, signature, timestamp } = req.body;

      // Verify node
      const node = await this.nodeService.getNode(nodeId, this.redis);
      if (!node) {
        return res.status(404).json({ error: 'Node not found' });
      }

      // Complete job
      const job = await this.jobService.completeJob(jobId, nodeId);

      res.json({
        success: true,
        job
      });
    } catch (error) {
      console.error('Error completing job:', error);
      res.status(400).json({ error: error.message });
    }
  }

  // POST /api/jobs/:jobId/fail - Mark job as failed
  async failJob(req, res) {
    try {
      const { jobId } = req.params;
      const { nodeId, signature, timestamp, error: failureReason } = req.body;

      // Verify node
      const node = await this.nodeService.getNode(nodeId, this.redis);
      if (!node) {
        return res.status(404).json({ error: 'Node not found' });
      }

      // Fail job
      const job = await this.jobService.failJob(jobId, nodeId, failureReason);

      res.json({
        success: true,
        job
      });
    } catch (error) {
      console.error('Error failing job:', error);
      res.status(400).json({ error: error.message });
    }
  }

  // GET /api/jobs/:jobId - Get job status and results
  async getJob(req, res) {
    try {
      const { jobId } = req.params;

      const result = await this.jobService.getJobResult(jobId);

      res.json({
        success: true,
        ...result
      });
    } catch (error) {
      console.error('Error getting job:', error);
      res.status(404).json({ error: error.message });
    }
  }

  // GET /api/jobs/stats - Get queue statistics
  async getStats(req, res) {
    try {
      const stats = await this.jobService.getQueueStats();

      res.json({
        success: true,
        stats
      });
    } catch (error) {
      console.error('Error getting stats:', error);
      res.status(500).json({ error: 'Failed to get queue statistics' });
    }
  }

  // POST /api/jobs/cleanup - Clean up old jobs (admin only)
  async cleanupJobs(req, res) {
    try {
      const { maxAge } = req.body;
      
      const cleaned = await this.jobService.cleanupOldJobs(maxAge);

      res.json({
        success: true,
        cleaned
      });
    } catch (error) {
      console.error('Error cleaning up jobs:', error);
      res.status(500).json({ error: 'Failed to cleanup jobs' });
    }
  }

  // POST /api/jobs/check-timeouts - Check for timed out jobs (called periodically)
  async checkTimeouts(req, res) {
    try {
      const timeoutJobs = await this.jobService.checkTimeouts();

      res.json({
        success: true,
        timeoutJobs
      });
    } catch (error) {
      console.error('Error checking timeouts:', error);
      res.status(500).json({ error: 'Failed to check timeouts' });
    }
  }
}

module.exports = JobController;