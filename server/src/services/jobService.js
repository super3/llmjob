class JobService {
  constructor(redis) {
    // Production passes the real redis v5 client (camelCase, promise-based);
    // tests pass an equivalent adapter. Either way we use it directly.
    this.redis = redis;
  }

  // Generate a unique job ID
  generateJobId() {
    return `job-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  // Create a new job
  async createJob(jobData) {
    const jobId = this.generateJobId();
    const timestamp = Date.now();

    const job = {
      id: jobId,
      prompt: jobData.prompt,
      model: jobData.model || 'llama3.2:3b',
      options: jobData.options || {},
      priority: jobData.priority || 0,
      status: 'pending',
      createdAt: timestamp,
      updatedAt: timestamp,
      userId: jobData.userId,
      maxTokens: jobData.maxTokens || 1000,
      temperature: jobData.temperature || 0.7
    };

    // Store job data
    await this.redis.set(`job:${jobId}`, JSON.stringify(job));

    // Add to pending queue with priority score (higher priority = lower score for earlier processing)
    await this.redis.zAdd('jobs:pending', {
      score: -job.priority * 1000000 + timestamp, // Negative priority for reverse order, then timestamp
      value: jobId
    });

    return job;
  }

  // Get job by ID
  async getJob(jobId) {
    const data = await this.redis.get(`job:${jobId}`);
    return data ? JSON.parse(data) : null;
  }

  // Update job status
  async updateJobStatus(jobId, status, additionalData = {}) {
    const job = await this.getJob(jobId);
    if (!job) {
      throw new Error(`Job ${jobId} not found`);
    }

    const updatedJob = {
      ...job,
      status,
      ...additionalData,
      updatedAt: Date.now()
    };

    await this.redis.set(`job:${jobId}`, JSON.stringify(updatedJob));

    // Remove from old status set and add to new one
    const statusSets = ['pending', 'assigned', 'running', 'completed', 'failed'];
    for (const s of statusSets) {
      if (s !== status) {
        await this.redis.zRem(`jobs:${s}`, jobId);
      }
    }

    // Add to new status set
    if (status !== 'pending') { // pending jobs stay in the priority queue
      await this.redis.zAdd(`jobs:${status}`, {
        score: Date.now(),
        value: jobId
      });
    }

    return updatedJob;
  }

  // Assign jobs to a node
  async assignJobsToNode(nodeId, maxJobs = 1) {
    const assignedJobs = [];

    // Get pending jobs sorted by priority/timestamp
    const pendingJobIds = await this.redis.zRange('jobs:pending', 0, maxJobs - 1);

    for (const jobId of pendingJobIds) {
      // Try to acquire lock on this job
      const lockKey = `job:${jobId}:lock`;
      const lockAcquired = await this.redis.set(lockKey, nodeId, {
        NX: true, // Only set if doesn't exist
        EX: 600   // 10 minute expiry
      });

      if (lockAcquired) {
        // Remove from pending queue
        await this.redis.zRem('jobs:pending', jobId);

        // Update job status
        const job = await this.updateJobStatus(jobId, 'assigned', {
          assignedTo: nodeId,
          assignedAt: Date.now()
        });

        // Track assignment for this node
        await this.redis.sAdd(`node:${nodeId}:jobs`, jobId);

        assignedJobs.push(job);
      }
    }

    return assignedJobs;
  }

  // Handle job heartbeat
  async handleHeartbeat(jobId, nodeId) {
    const lockKey = `job:${jobId}:lock`;
    const currentLock = await this.redis.get(lockKey);

    if (currentLock !== nodeId) {
      throw new Error('Node does not hold lock for this job');
    }

    // Extend lock timeout
    await this.redis.expire(lockKey, 600); // Reset to 10 minutes

    // Update last heartbeat time
    await this.redis.set(`job:${jobId}:heartbeat`, Date.now(), { EX: 60 }); // 60 second expiry

    // Update job status if needed
    const job = await this.getJob(jobId);
    if (job && job.status === 'assigned') {
      await this.updateJobStatus(jobId, 'running', {
        startedAt: job.startedAt || Date.now()
      });
    }

    return { success: true };
  }

  // Store job chunk
  async storeChunk(jobId, nodeId, chunkData) {
    const lockKey = `job:${jobId}:lock`;
    const currentLock = await this.redis.get(lockKey);

    if (currentLock !== nodeId) {
      throw new Error('Node does not hold lock for this job');
    }

    const chunkKey = `job:${jobId}:chunks`;
    const chunk = {
      index: chunkData.chunkIndex,
      content: chunkData.content,
      metrics: chunkData.metrics,
      isFinal: chunkData.isFinal || false,
      timestamp: chunkData.timestamp || Date.now()
    };

    // Store chunk in sorted set by index
    await this.redis.zAdd(chunkKey, {
      score: chunk.index,
      value: JSON.stringify(chunk)
    });

    // Update job with latest metrics
    if (chunkData.metrics) {
      const job = await this.getJob(jobId);
      await this.redis.set(`job:${jobId}`, JSON.stringify({
        ...job,
        lastMetrics: chunkData.metrics,
        updatedAt: Date.now()
      }));
    }

    return { success: true, chunkIndex: chunk.index };
  }

  // Complete a job
  async completeJob(jobId, nodeId) {
    const lockKey = `job:${jobId}:lock`;
    const currentLock = await this.redis.get(lockKey);

    if (currentLock !== nodeId) {
      throw new Error('Node does not hold lock for this job');
    }

    // Get all chunks and assemble result
    const chunks = await this.redis.zRange(`job:${jobId}:chunks`, 0, -1);
    const assembledContent = chunks
      .map(chunk => JSON.parse(chunk).content)
      .join('');

    // Update job status
    const job = await this.updateJobStatus(jobId, 'completed', {
      completedAt: Date.now(),
      result: assembledContent,
      chunks: chunks.length
    });

    // Release lock
    await this.redis.del(lockKey);

    // Remove from node's active jobs
    await this.redis.sRem(`node:${nodeId}:jobs`, jobId);

    // Clean up heartbeat
    await this.redis.del(`job:${jobId}:heartbeat`);

    return job;
  }

  // Fail a job
  async failJob(jobId, nodeId, reason) {
    const lockKey = `job:${jobId}:lock`;
    const currentLock = await this.redis.get(lockKey);

    if (currentLock !== nodeId) {
      throw new Error('Node does not hold lock for this job');
    }

    // Update job status
    const job = await this.updateJobStatus(jobId, 'failed', {
      failedAt: Date.now(),
      failureReason: reason
    });

    // Release lock
    await this.redis.del(lockKey);

    // Remove from node's active jobs
    await this.redis.sRem(`node:${nodeId}:jobs`, jobId);

    // Clean up heartbeat
    await this.redis.del(`job:${jobId}:heartbeat`);

    return job;
  }

  // Check for timed out jobs and return them to queue
  async checkTimeouts() {
    const now = Date.now();
    const timeoutJobs = [];

    // Get all assigned and running jobs
    const assignedJobs = await this.redis.zRange('jobs:assigned', 0, -1);
    const runningJobs = await this.redis.zRange('jobs:running', 0, -1);
    const allJobs = [...assignedJobs, ...runningJobs];

    for (const jobId of allJobs) {
      const lockKey = `job:${jobId}:lock`;
      const heartbeatKey = `job:${jobId}:heartbeat`;

      // Check if lock exists
      const lockTTL = await this.redis.ttl(lockKey);
      const lastHeartbeat = await this.redis.get(heartbeatKey);

      // If lock expired or no recent heartbeat
      if (lockTTL === -2 || (lastHeartbeat && now - parseInt(lastHeartbeat) > 60000)) {
        const job = await this.getJob(jobId);
        if (job) {
          // Return to pending queue
          await this.updateJobStatus(jobId, 'pending', {
            previousStatus: job.status,
            returnedToQueue: now,
            timeoutReason: lockTTL === -2 ? 'lock_expired' : 'heartbeat_timeout'
          });

          // Re-add to pending queue
          await this.redis.zAdd('jobs:pending', {
            score: -job.priority * 1000000 + now,
            value: jobId
          });

          // Clean up
          await this.redis.del(lockKey);
          await this.redis.del(heartbeatKey);
          if (job.assignedTo) {
            await this.redis.sRem(`node:${job.assignedTo}:jobs`, jobId);
          }

          timeoutJobs.push(jobId);
        }
      }
    }

    return timeoutJobs;
  }

  // Get job results
  async getJobResult(jobId) {
    const job = await this.getJob(jobId);
    if (!job) {
      throw new Error(`Job ${jobId} not found`);
    }

    // Always get chunks for any status where they might exist
    let chunks = [];
    if (job.status === 'running' || job.status === 'completed') {
      const rawChunks = await this.redis.zRange(`job:${jobId}:chunks`, 0, -1);
      chunks = rawChunks.map(chunk => JSON.parse(chunk));
    }

    if (job.status === 'completed') {
      return {
        jobId,
        status: 'completed',
        result: job.result,
        chunks: chunks, // Include chunks array for UI compatibility
        metrics: job.lastMetrics,
        completedAt: job.completedAt,
        assignedTo: job.assignedTo
      };
    }

    if (job.status === 'failed') {
      return {
        jobId,
        status: 'failed',
        error: job.failureReason,
        failedAt: job.failedAt,
        assignedTo: job.assignedTo
      };
    }

    // For running jobs, return partial results from chunks
    if (job.status === 'running') {
      const partialContent = chunks
        .map(chunk => chunk.content)
        .join('');

      return {
        jobId,
        status: 'running',
        partial: partialContent,
        chunks: chunks, // Return chunks array instead of count
        metrics: job.lastMetrics,
        assignedTo: job.assignedTo
      };
    }

    return {
      jobId,
      status: job.status,
      createdAt: job.createdAt,
      assignedTo: job.assignedTo
    };
  }

  // Get queue statistics
  async getQueueStats() {
    const stats = {
      pending: await this.redis.zCard('jobs:pending'),
      assigned: await this.redis.zCard('jobs:assigned'),
      running: await this.redis.zCard('jobs:running'),
      completed: await this.redis.zCard('jobs:completed'),
      failed: await this.redis.zCard('jobs:failed')
    };

    return stats;
  }

  // Clean up old completed/failed jobs
  async cleanupOldJobs(maxAge = 86400000) { // Default 24 hours
    const cutoff = Date.now() - maxAge;

    // Get old completed jobs
    const completedJobs = await this.redis.zRangeByScore('jobs:completed', 0, cutoff);
    const failedJobs = await this.redis.zRangeByScore('jobs:failed', 0, cutoff);

    const allOldJobs = [...completedJobs, ...failedJobs];

    for (const jobId of allOldJobs) {
      // Delete job data
      await this.redis.del(`job:${jobId}`);
      await this.redis.del(`job:${jobId}:chunks`);
      await this.redis.del(`job:${jobId}:lock`);
      await this.redis.del(`job:${jobId}:heartbeat`);

      // Remove from status sets
      await this.redis.zRem('jobs:completed', jobId);
      await this.redis.zRem('jobs:failed', jobId);
    }

    return allOldJobs.length;
  }
}

module.exports = JobService;
