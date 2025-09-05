const JobRepository = require('../repositories/JobRepository');

/**
 * Refactored JobService using Repository pattern
 */
class JobServiceV2 {
  constructor(redis) {
    this.jobRepo = new JobRepository(redis);
  }

  /**
   * Generate a unique job ID
   */
  generateJobId() {
    return `job-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Create a new job
   */
  async createJob(jobData) {
    const job = {
      id: this.generateJobId(),
      ...jobData,
      status: 'pending',
      createdAt: Date.now(),
      updatedAt: Date.now(),
      attempts: 0,
      priority: jobData.priority || Date.now()
    };

    await this.jobRepo.createJob(job);
    return job;
  }

  /**
   * Get job by ID
   */
  async getJob(jobId) {
    const job = await this.jobRepo.getJob(jobId);
    if (!job) {
      throw new Error(`Job ${jobId} not found`);
    }
    return job;
  }

  /**
   * Update job status and additional data
   */
  async updateJobStatus(jobId, status, additionalData = {}) {
    const updates = {
      status,
      updatedAt: Date.now(),
      ...additionalData
    };

    await this.jobRepo.updateJob(jobId, updates);
    
    // Handle queue transitions
    if (status === 'assigned') {
      await this.jobRepo.removeFromPendingQueue(jobId);
      await this.jobRepo.addToAssignedQueue(jobId);
    } else if (status === 'completed') {
      await this.jobRepo.removeFromAssignedQueue(jobId);
      await this.jobRepo.addToCompletedQueue(jobId);
    } else if (status === 'failed') {
      await this.jobRepo.removeFromAssignedQueue(jobId);
      await this.jobRepo.addToFailedQueue(jobId);
    } else if (status === 'pending') {
      await this.jobRepo.removeFromAssignedQueue(jobId);
      await this.jobRepo.addToPendingQueue(jobId);
    }

    return await this.jobRepo.getJob(jobId);
  }

  /**
   * Assign jobs to a node
   */
  async assignJobsToNode(nodeId, maxJobs = 1) {
    const pendingJobs = await this.jobRepo.getPendingJobs(maxJobs);
    const assignedJobs = [];

    for (const jobId of pendingJobs) {
      // Try to acquire lock
      const lockAcquired = await this.jobRepo.acquireLock(jobId, nodeId, 300);
      
      if (lockAcquired) {
        // Update job status
        await this.updateJobStatus(jobId, 'assigned', {
          assignedTo: nodeId,
          assignedAt: Date.now()
        });

        const job = await this.jobRepo.getJob(jobId);
        assignedJobs.push(job);

        if (assignedJobs.length >= maxJobs) {
          break;
        }
      }
    }

    return assignedJobs;
  }

  /**
   * Handle heartbeat from node
   */
  async handleHeartbeat(jobId, nodeId) {
    // Check if node holds the lock
    const hasLock = await this.jobRepo.checkLock(jobId, nodeId);
    
    if (!hasLock) {
      throw new Error('Node does not hold lock for this job');
    }

    // Extend lock
    await this.jobRepo.extendLock(jobId, nodeId, 300);

    // Update job heartbeat
    await this.jobRepo.updateJob(jobId, {
      lastHeartbeat: Date.now(),
      status: 'running'
    });

    // Update queue if needed
    const job = await this.jobRepo.getJob(jobId);
    if (job.status === 'assigned') {
      await this.updateJobStatus(jobId, 'running');
    }

    return { success: true, extended: true };
  }

  /**
   * Store chunk data for streaming results
   */
  async storeChunk(jobId, nodeId, chunkData) {
    // Verify node holds lock
    const hasLock = await this.jobRepo.checkLock(jobId, nodeId);
    
    if (!hasLock) {
      throw new Error('Node does not hold lock for this job');
    }

    // Store chunk
    await this.jobRepo.storeChunk(jobId, chunkData.chunkIndex, chunkData.content);

    // Update job progress
    await this.jobRepo.updateJob(jobId, {
      lastChunkAt: Date.now(),
      chunkCount: (chunkData.chunkIndex || 0) + 1,
      metrics: chunkData.metrics
    });

    return { stored: true, chunkIndex: chunkData.chunkIndex };
  }

  /**
   * Complete a job
   */
  async completeJob(jobId, nodeId, finalOutput = null) {
    // Verify node holds lock
    const hasLock = await this.jobRepo.checkLock(jobId, nodeId);
    
    if (!hasLock) {
      throw new Error('Node does not hold lock for this job');
    }

    // Get chunks if available
    const chunks = await this.jobRepo.getChunks(jobId);
    const result = finalOutput || chunks.map(c => c.content).join('');

    // Update job status
    const completedJob = await this.updateJobStatus(jobId, 'completed', {
      result,
      completedAt: Date.now(),
      completedBy: nodeId
    });

    // Release lock
    await this.jobRepo.releaseLock(jobId, nodeId);

    // Clean up chunks
    await this.jobRepo.deleteChunks(jobId);

    return completedJob;
  }

  /**
   * Fail a job
   */
  async failJob(jobId, nodeId, reason) {
    // Verify node holds lock
    const hasLock = await this.jobRepo.checkLock(jobId, nodeId);
    
    if (!hasLock) {
      throw new Error('Node does not hold lock for this job');
    }

    // Update job status
    const failedJob = await this.updateJobStatus(jobId, 'failed', {
      failureReason: reason,
      failedAt: Date.now(),
      failedBy: nodeId
    });

    // Release lock
    await this.jobRepo.releaseLock(jobId, nodeId);

    // Clean up chunks
    await this.jobRepo.deleteChunks(jobId);

    return failedJob;
  }

  /**
   * Check for timed out jobs
   */
  async checkTimeouts(timeout = 600000) {
    return await this.jobRepo.checkTimeouts(timeout);
  }

  /**
   * Get job result
   */
  async getJobResult(jobId) {
    const job = await this.jobRepo.getJob(jobId);
    
    if (!job) {
      throw new Error(`Job ${jobId} not found`);
    }

    // Include chunks if job is still running
    if (job.status === 'running' || job.status === 'assigned') {
      const chunks = await this.jobRepo.getChunks(jobId);
      job.chunks = chunks;
    }

    return job;
  }

  /**
   * Get queue statistics
   */
  async getQueueStats() {
    return await this.jobRepo.getQueueStats();
  }

  /**
   * Clean up old jobs
   */
  async cleanupOldJobs(maxAge = 86400000) {
    return await this.jobRepo.cleanupOldJobs(maxAge);
  }

  /**
   * Cancel a job
   */
  async cancelJob(jobId, reason = 'Cancelled by user') {
    const job = await this.jobRepo.getJob(jobId);
    
    if (!job) {
      throw new Error(`Job ${jobId} not found`);
    }

    // Check if job can be cancelled
    if (job.status === 'completed' || job.status === 'failed') {
      throw new Error(`Cannot cancel job in ${job.status} state`);
    }

    // Get lock holder if any
    const lockHolder = await this.jobRepo.getLockHolder(jobId);

    // Update job status
    await this.updateJobStatus(jobId, 'failed', {
      failureReason: reason,
      cancelledAt: Date.now()
    });

    // Release lock if exists
    if (lockHolder) {
      await this.jobRepo.releaseLock(jobId, lockHolder);
    }

    // Clean up chunks
    await this.jobRepo.deleteChunks(jobId);

    return { success: true, jobId };
  }

  /**
   * Get jobs by status
   */
  async getJobsByStatus(status, limit = 100) {
    let jobIds = [];

    switch (status) {
      case 'pending':
        jobIds = await this.jobRepo.getPendingJobs(limit);
        break;
      case 'assigned':
      case 'running':
        jobIds = await this.jobRepo.getAssignedJobs(limit);
        break;
      case 'completed':
        jobIds = await this.jobRepo.getCompletedJobs(limit);
        break;
      case 'failed':
        jobIds = await this.jobRepo.getFailedJobs(limit);
        break;
      default:
        throw new Error(`Invalid status: ${status}`);
    }

    const jobs = [];
    for (const jobId of jobIds) {
      const job = await this.jobRepo.getJob(jobId);
      if (job) {
        jobs.push(job);
      }
    }

    return jobs;
  }

  /**
   * Retry a failed job
   */
  async retryJob(jobId) {
    const job = await this.jobRepo.getJob(jobId);
    
    if (!job) {
      throw new Error(`Job ${jobId} not found`);
    }

    if (job.status !== 'failed') {
      throw new Error(`Can only retry failed jobs. Current status: ${job.status}`);
    }

    // Reset job to pending
    await this.updateJobStatus(jobId, 'pending', {
      attempts: (job.attempts || 0) + 1,
      retriedAt: Date.now(),
      failureReason: null,
      failedAt: null,
      failedBy: null
    });

    return { success: true, jobId };
  }
}

module.exports = JobServiceV2;