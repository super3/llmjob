const BaseRepository = require('./BaseRepository');

/**
 * Repository for Job-related Redis operations
 */
class JobRepository extends BaseRepository {
  constructor(redis) {
    super(redis, 'job:');
    this.pendingQueueKey = 'queue:pending';
    this.assignedQueueKey = 'queue:assigned';
    this.completedQueueKey = 'queue:completed';
    this.failedQueueKey = 'queue:failed';
    this.lockPrefix = 'job:lock:';
    this.chunkPrefix = 'job:chunks:';
  }

  // Job CRUD operations
  async createJob(jobData) {
    const jobId = jobData.id;
    
    // Store job data as hash
    const hashData = {};
    Object.keys(jobData).forEach(key => {
      hashData[key] = typeof jobData[key] === 'object' 
        ? JSON.stringify(jobData[key]) 
        : String(jobData[key]);
    });
    
    // Store job in hash
    for (const [field, value] of Object.entries(hashData)) {
      await this.hSet(jobId, field, value);
    }
    
    // Add to pending queue
    await this.redis.zAdd(this.pendingQueueKey, {
      score: jobData.priority || Date.now(),
      member: jobId
    });
    
    return jobId;
  }

  async getJob(jobId) {
    const jobData = await this.hGetAll(jobId);
    
    if (!jobData || Object.keys(jobData).length === 0) {
      return null;
    }
    
    // Parse JSON fields
    const parsedData = {};
    Object.keys(jobData).forEach(key => {
      try {
        if (key === 'options' || key === 'result' || key === 'chunks') {
          parsedData[key] = JSON.parse(jobData[key]);
        } else {
          parsedData[key] = jobData[key];
        }
      } catch {
        parsedData[key] = jobData[key];
      }
    });
    
    return parsedData;
  }

  async updateJob(jobId, updates) {
    for (const [field, value] of Object.entries(updates)) {
      const serializedValue = typeof value === 'object' 
        ? JSON.stringify(value) 
        : String(value);
      await this.hSet(jobId, field, serializedValue);
    }
  }

  async deleteJob(jobId) {
    // Remove from all queues
    await this.redis.zRem(this.pendingQueueKey, jobId);
    await this.redis.zRem(this.assignedQueueKey, jobId);
    await this.redis.zRem(this.completedQueueKey, jobId);
    await this.redis.zRem(this.failedQueueKey, jobId);
    
    // Delete job data
    await this.delete(jobId);
    
    // Delete associated data
    await this.redis.del(`${this.lockPrefix}${jobId}`);
    await this.redis.del(`${this.chunkPrefix}${jobId}`);
  }

  // Queue operations
  async addToPendingQueue(jobId, priority = Date.now()) {
    return await this.redis.zAdd(this.pendingQueueKey, { score: priority, member: jobId });
  }

  async removeFromPendingQueue(jobId) {
    return await this.redis.zRem(this.pendingQueueKey, jobId);
  }

  async addToAssignedQueue(jobId, timestamp = Date.now()) {
    return await this.redis.zAdd(this.assignedQueueKey, { score: timestamp, member: jobId });
  }

  async removeFromAssignedQueue(jobId) {
    return await this.redis.zRem(this.assignedQueueKey, jobId);
  }

  async addToCompletedQueue(jobId, timestamp = Date.now()) {
    return await this.redis.zAdd(this.completedQueueKey, { score: timestamp, member: jobId });
  }

  async addToFailedQueue(jobId, timestamp = Date.now()) {
    return await this.redis.zAdd(this.failedQueueKey, { score: timestamp, member: jobId });
  }

  async getPendingJobs(limit = 10) {
    return await this.redis.zRange(this.pendingQueueKey, 0, limit - 1);
  }

  async getAssignedJobs(limit = -1) {
    return await this.redis.zRange(this.assignedQueueKey, 0, limit);
  }

  async getCompletedJobs(limit = -1) {
    return await this.redis.zRange(this.completedQueueKey, 0, limit);
  }

  async getFailedJobs(limit = -1) {
    return await this.redis.zRange(this.failedQueueKey, 0, limit);
  }

  // Lock operations
  async acquireLock(jobId, nodeId, ttl = 300) {
    const lockKey = `${this.lockPrefix}${jobId}`;
    const result = await this.redis.set(lockKey, nodeId);
    if (result) {
      await this.redis.expire(lockKey, ttl);
    }
    return result;
  }

  async checkLock(jobId, nodeId) {
    const lockKey = `${this.lockPrefix}${jobId}`;
    const lockHolder = await this.redis.get(lockKey);
    return lockHolder === nodeId;
  }

  async releaseLock(jobId, nodeId) {
    const lockKey = `${this.lockPrefix}${jobId}`;
    const lockHolder = await this.redis.get(lockKey);
    
    if (lockHolder === nodeId) {
      return await this.redis.del(lockKey);
    }
    return false;
  }

  async getLockHolder(jobId) {
    const lockKey = `${this.lockPrefix}${jobId}`;
    return await this.redis.get(lockKey);
  }

  async extendLock(jobId, nodeId, ttl = 300) {
    if (await this.checkLock(jobId, nodeId)) {
      const lockKey = `${this.lockPrefix}${jobId}`;
      return await this.redis.expire(lockKey, ttl);
    }
    return false;
  }

  // Chunk operations
  async storeChunk(jobId, chunkIndex, content) {
    const chunkKey = `${this.chunkPrefix}${jobId}`;
    return await this.redis.lPush(chunkKey, JSON.stringify({ chunkIndex, content, timestamp: Date.now() }));
  }

  async getChunks(jobId) {
    const chunkKey = `${this.chunkPrefix}${jobId}`;
    const chunks = await this.redis.lRange(chunkKey, 0, -1);
    return chunks.map(chunk => JSON.parse(chunk)).reverse();
  }

  async deleteChunks(jobId) {
    const chunkKey = `${this.chunkPrefix}${jobId}`;
    return await this.redis.del(chunkKey);
  }

  // Queue statistics
  async getQueueStats() {
    const [pending, assigned, completed, failed] = await Promise.all([
      this.redis.zCard(this.pendingQueueKey),
      this.redis.zCard(this.assignedQueueKey),
      this.redis.zCard(this.completedQueueKey),
      this.redis.zCard(this.failedQueueKey)
    ]);
    
    return {
      pending,
      assigned,
      running: assigned, // alias for compatibility
      completed,
      failed
    };
  }

  // Cleanup operations
  async cleanupOldJobs(maxAge = 86400000) { // 24 hours default
    const cutoffTime = Date.now() - maxAge;
    
    // Get old completed jobs
    const oldCompleted = await this.redis.zRangeByScore(
      this.completedQueueKey,
      0,
      cutoffTime
    );
    
    // Get old failed jobs
    const oldFailed = await this.redis.zRangeByScore(
      this.failedQueueKey,
      0,
      cutoffTime
    );
    
    const jobsToDelete = [...oldCompleted, ...oldFailed];
    
    // Delete each job
    for (const jobId of jobsToDelete) {
      await this.deleteJob(jobId);
    }
    
    return jobsToDelete.length;
  }

  // Timeout checking
  async checkTimeouts(timeout = 600000) { // 10 minutes default
    const cutoffTime = Date.now() - timeout;
    
    // Get jobs that have been assigned too long
    const timedOutJobs = await this.redis.zRangeByScore(
      this.assignedQueueKey,
      0,
      cutoffTime
    );
    
    for (const jobId of timedOutJobs) {
      // Move back to pending queue
      await this.removeFromAssignedQueue(jobId);
      await this.addToPendingQueue(jobId);
      
      // Update job status
      await this.updateJob(jobId, {
        status: 'pending',
        timeoutReason: 'lock_expired',
        lastTimeout: Date.now()
      });
      
      // Release lock
      const lockKey = `${this.lockPrefix}${jobId}`;
      await this.redis.del(lockKey);
    }
    
    return timedOutJobs;
  }
}

module.exports = JobRepository;