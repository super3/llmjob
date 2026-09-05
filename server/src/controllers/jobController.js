const JobService = require('../services/jobService');

// The most jobs one poll may claim. A node runs them one at a time
// (jobWorker.processJob awaits each), so a large batch buys nothing but holds
// locks the rest of the fleet could be working.
const MAX_JOBS_PER_POLL = 10;

// Coerce a node-supplied maxJobs into [1, MAX_JOBS_PER_POLL]. Anything absent,
// non-numeric, negative or fractional falls back to 1.
function clampMaxJobs(v) {
  const n = Math.floor(Number(v));
  if (!Number.isFinite(n) || n < 1) return 1;
  return Math.min(n, MAX_JOBS_PER_POLL);
}

class JobController {
  // `opts` is injected in tests so the long-poll below needs no real timers:
  // same shape openaiController already uses for the caller-side hold.
  constructor(jobService, nodeService, opts = {}) {
    this.jobService = jobService;
    this.nodeService = nodeService;
    // How long to hold an empty poll open before answering with nothing.
    //
    // 25s, and the margin is the point: the node posts with a 30s client timeout
    // (makeCliJobWorker), so a 30s hold would have the client giving up at the
    // same instant the server answers -- every idle poll a coin-flip between a
    // clean empty response and a transport error that backs the node off. Well
    // under the 5-minute no-bytes cutoff a proxy enforces, too.
    this.holdMs = opts.holdMs || 25000;
    // How long to wait between claim attempts when NOTHING has told us the queue
    // changed. This is a safety net, not the dispatch mechanism: a job created in
    // this process wakes the hold immediately (JobService's queue signal), so
    // this only covers what that signal cannot see — a job created by another
    // replica.
    //
    // It was 250ms, which meant every idle node re-ran the full claim
    // transaction four times a second forever: 16 SQL statements per second per
    // idle node, each taking a pooled connection, purely to learn there was
    // still nothing to do. At 2s the idle cost drops 8x and dispatch latency
    // actually improves, because the common case no longer waits for a tick at
    // all.
    this.idleRecheckMs = opts.idleRecheckMs || 2000;
    this.now = opts.now || Date.now;
    // The one wait between claim attempts: returns as soon as the queue changes,
    // or after idleRecheckMs. Injectable so a test can stand in for the wait
    // without real timers.
    this.awaitQueue = opts.awaitQueue
      || ((since, ms) => JobService.awaitQueueChange(since, ms));
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

      // Assign jobs to node. maxJobs is clamped, not trusted: it lands in a SQL
      // LIMIT, and node enrollment is open to the internet (/api/nodes/register
      // accepts any self-minted keypair). Unclamped, one anonymous node polling
      // with maxJobs: 100000 locks every pending public job for the full LOCK_MS
      // — which is both a fleet-wide denial of service and a way to be handed
      // every other caller's prompts.
      // Long-poll rather than answer empty immediately.
      //
      // The node used to poll on an exponential backoff -- 5s doubling to 60s --
      // so an idle rig sat on a 20-40s rung and a job waited that long to be
      // ASKED for, before any model work began. That was invisible while nodes
      // kept a model resident and polled steadily; with demand mode it became
      // the largest term in time-to-first-token, larger than the model load.
      //
      // Holding the request open costs one connection per idle node and needs no
      // protocol change: same route, same signature check, same atomic claim, so
      // it does not matter which server instance holds which connection.
      const limit = clampMaxJobs(maxJobs);
      const deadline = this.now() + this.holdMs;
      // Captured BEFORE each claim, so a job created in the window between a
      // claim coming back empty and the wait starting has already moved the
      // version and the wait returns at once rather than sleeping through work
      // that is sitting in the queue.
      let version = JobService.queueVersion();
      let jobs = await this.jobService.assignJobsToNode(nodeId, limit);
      while (!jobs.length && this.now() < deadline) {
        // Stop holding if the node hung up: continuing would keep claiming jobs
        // for a connection nobody will read, and each claim locks a job.
        if (res.writableEnded || res.destroyed) return;
        await this.awaitQueue(version, this.idleRecheckMs);
        version = JobService.queueVersion();
        jobs = await this.jobService.assignJobsToNode(nodeId, limit);
      }

      res.json({
        success: true,
        jobs: jobs.map(job => ({
          id: job.id,
          prompt: job.prompt,
          messages: job.messages, // present for OpenAI-gateway jobs; undefined otherwise
          model: job.model,
          options: job.options,
          maxTokens: job.maxTokens,
          temperature: job.temperature,
          // Fences this attempt against a sibling worker on the same node id —
          // the node echoes it on every write for this job. See _assertLock.
          lockToken: job.lockToken
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
      const { nodeId, lockToken } = req.body;

      // Verify node
      if (!(await this._requireNode(req, res))) return;

      // Handle heartbeat
      await this.jobService.handleHeartbeat(jobId, nodeId, lockToken);

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
      const { nodeId, lockToken, chunkIndex, content, reasoning, metrics, isFinal, timestamp } = req.body;

      // Verify node
      if (!(await this._requireNode(req, res))) return;

      // Store chunk
      const result = await this.jobService.storeChunk(jobId, nodeId, {
        chunkIndex,
        content,
        reasoning,
        metrics,
        isFinal,
        timestamp
      }, lockToken);

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
      const { nodeId, lockToken } = req.body;

      // Verify node
      if (!(await this._requireNode(req, res))) return;

      // Complete job
      const job = await this.jobService.completeJob(jobId, nodeId, lockToken);

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
      const { nodeId, lockToken, error: failureReason } = req.body;

      // Verify node
      if (!(await this._requireNode(req, res))) return;

      // Fail job
      const job = await this.jobService.failJob(jobId, nodeId, failureReason, lockToken);

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
module.exports.MAX_JOBS_PER_POLL = MAX_JOBS_PER_POLL;
module.exports.clampMaxJobs = clampMaxJobs;
