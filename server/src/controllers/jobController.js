class JobController {
  constructor(jobService, nodeService) {
    this.jobService = jobService;
    this.nodeService = nodeService;
  }

  // Look up the node for a request and prove the caller really is that node.
  //
  // verifySignature only proves the sender owns the keypair it presented — it
  // does NOT tie the claimed nodeId to that key. So the presented key must be
  // checked against the one the node registered, or anyone could sign
  // "<someone-else's-nodeId>:<ts>" with a freshly generated key and act as that
  // node — polling its jobs (which, for a private API key, are another user's
  // prompts), or completing/failing them. nodeIds are not secret: GET
  // /api/nodes/public lists them unauthenticated. This mirrors the same check
  // nodeService.updateNodeStatus already makes on the ping path.
  //
  // Returns the node on success, or null after sending the response.
  async _requireNode(req, res) {
    const { nodeId } = req.body;
    const node = await this.nodeService.getNode(nodeId);
    if (!node) {
      res.status(404).json({ error: 'Node not found' });
      return null;
    }
    const presented = req.verifiedNode && req.verifiedNode.publicKey;
    if (!node.publicKey || node.publicKey !== presented) {
      res.status(401).json({ error: 'Public key mismatch' });
      return null;
    }
    return node;
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
      const { nodeId, maxJobs } = req.body;

      // Verify node exists and is active
      if (!(await this._requireNode(req, res))) return;

      // Assign jobs to node
      const jobs = await this.jobService.assignJobsToNode(nodeId, maxJobs || 1);

      res.json({
        success: true,
        jobs: jobs.map(job => ({
          id: job.id,
          prompt: job.prompt,
          messages: job.messages, // present for OpenAI-gateway jobs; undefined otherwise
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
      const { nodeId } = req.body;

      // Verify node
      if (!(await this._requireNode(req, res))) return;

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
      const { nodeId, chunkIndex, content, metrics, isFinal, timestamp } = req.body;

      // Verify node
      if (!(await this._requireNode(req, res))) return;

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
      const { nodeId } = req.body;

      // Verify node
      if (!(await this._requireNode(req, res))) return;

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
      const { nodeId, error: failureReason } = req.body;

      // Verify node
      if (!(await this._requireNode(req, res))) return;

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

  // GET /api/jobs/:jobId - Get status and results for a job you submitted.
  // Authenticated (Clerk session or API key) and scoped to the submitter: a job
  // carries the prompt and the model's reply, so anyone who could read an
  // arbitrary jobId could read other people's conversations — including those a
  // private API key routed to the owner's own hardware.
  async getJob(req, res) {
    try {
      const { jobId } = req.params;

      const job = await this.jobService.getJob(jobId);
      // 404 rather than 403 when it isn't yours: a 403 confirms the id exists,
      // which is exactly the oracle an id-guessing attacker wants. Anonymous
      // public-chat jobs (userId null) match no caller, so they stay unreadable
      // over HTTP — the chat gateway reads them in-process instead.
      if (!job || job.userId !== req.user.id) {
        return res.status(404).json({ error: `Job ${jobId} not found` });
      }

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
