const crypto = require('crypto');

const LOCK_MS = 10 * 60 * 1000;       // assignment lock lifetime (10 min)
const HEARTBEAT_STALE_MS = 60 * 1000; // consider a job stalled after 60s silence
// How long a job may sit pending before it is abandoned. Both gateways give up
// waiting after 120s, so anything older than this has no caller left listening —
// running it later would burn a node's GPU on a reply nobody receives, and the
// rows would otherwise accumulate forever (nothing else clears `pending`). The
// margin over 120s means this can never expire a job someone is still waiting on.
const PENDING_TTL_MS = 5 * 60 * 1000;
// The model the earn-client fleet actually serves (earn/src/shared/config.js
// LLM.model.name) — the default a job records must match what runs it.
const DEFAULT_MODEL = 'Gemma-4-E4B-it-Q4_K_M';

// A job's routing (inherited from the API key that created it): 'private' may
// only run on the owner's own nodes; anything else is 'public' (any node).
function normalizeVisibility(v) {
  return v === 'private' ? 'private' : 'public';
}

class JobService {
  constructor(db) {
    this.db = db;
  }

  // Crypto-random suffix, not Math.random(): a job id is a capability-ish handle
  // (it addresses a conversation), and V8's PRNG state is recoverable from a few
  // observed outputs, which would make ids predictable rather than merely hard to
  // guess. The timestamp prefix is kept for sortability/debuggability.
  generateJobId() {
    return `job-${Date.now()}-${crypto.randomBytes(8).toString('hex')}`;
  }

  async createJob(jobData) {
    const jobId = this.generateJobId();
    const timestamp = Date.now();

    const job = {
      id: jobId,
      prompt: jobData.prompt,
      model: jobData.model || DEFAULT_MODEL,
      options: jobData.options || {},
      priority: jobData.priority || 0,
      status: 'pending',
      createdAt: timestamp,
      updatedAt: timestamp,
      userId: jobData.userId,
      visibility: normalizeVisibility(jobData.visibility),
      maxTokens: jobData.maxTokens || 1000,
      temperature: jobData.temperature || 0.7
    };

    // A job from the OpenAI gateway carries a full chat `messages` array so the
    // node can serve multi-turn conversations (the single `prompt` is kept as a
    // display/fallback). Only stored when provided, to leave simple jobs as-is.
    if (Array.isArray(jobData.messages) && jobData.messages.length) {
      job.messages = jobData.messages;
    }

    await this.db.query(
      `INSERT INTO jobs (id, data, status, priority, created_at, updated_at, user_id, visibility)
       VALUES ($1, $2, 'pending', $3, $4, $4, $5, $6)`,
      [jobId, JSON.stringify(job), job.priority, timestamp, job.userId, job.visibility]
    );

    return job;
  }

  async getJob(jobId) {
    const r = await this.db.query('SELECT data FROM jobs WHERE id = $1', [jobId]);
    return r.rows.length > 0 ? r.rows[0].data : null;
  }

  async updateJobStatus(jobId, status, additionalData = {}) {
    const job = await this.getJob(jobId);
    if (!job) {
      throw new Error(`Job ${jobId} not found`);
    }

    const updatedJob = { ...job, status, ...additionalData, updatedAt: Date.now() };

    await this.db.query(
      'UPDATE jobs SET data = $2, status = $3, assigned_to = $4, updated_at = $5 WHERE id = $1',
      [jobId, JSON.stringify(updatedJob), status, updatedJob.assignedTo || null, updatedJob.updatedAt]
    );

    return updatedJob;
  }

  // Claim up to maxJobs pending jobs for a node, locking them in one transaction.
  // A node may only be handed PUBLIC jobs, or PRIVATE jobs owned by the same user
  // that owns the node — so a private key's requests never reach another user's
  // hardware. The node's owner is read from the DB (the single source of truth),
  // not trusted from the caller. A NULL job visibility (pre-feature rows) counts
  // as public. A node with no owner (unclaimed) can serve only public jobs.
  async assignJobsToNode(nodeId, maxJobs = 1) {
    const assignedJobs = [];
    const client = await this.db.connect();
    try {
      await client.query('BEGIN');
      const owner = await client.query('SELECT user_id FROM nodes WHERE node_id = $1', [nodeId]);
      const ownerUserId = owner.rows.length ? owner.rows[0].user_id : null;
      const pending = await client.query(
        `SELECT id, data FROM jobs
         WHERE status = 'pending'
           AND (visibility IS NULL OR visibility <> 'private' OR user_id = $2)
         ORDER BY priority DESC, created_at ASC LIMIT $1 FOR UPDATE`,
        [maxJobs, ownerUserId]
      );

      for (const row of pending.rows) {
        const now = Date.now();
        const job = { ...row.data, status: 'assigned', assignedTo: nodeId, assignedAt: now, updatedAt: now };
        await client.query(
          `UPDATE jobs SET data = $2, status = 'assigned', assigned_to = $3, updated_at = $4,
             lock_node = $3, lock_expires_at = $5 WHERE id = $1`,
          [job.id, JSON.stringify(job), nodeId, now, now + LOCK_MS]
        );
        assignedJobs.push(job);
      }

      await client.query('COMMIT');
    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }

    return assignedJobs;
  }

  // Verify the node holds a live lock on the job; returns the row or throws.
  async _assertLock(jobId, nodeId) {
    const r = await this.db.query(
      'SELECT data, status, lock_node, lock_expires_at FROM jobs WHERE id = $1',
      [jobId]
    );
    const row = r.rows[0];
    const now = Date.now();
    const held = row && row.lock_expires_at != null && Number(row.lock_expires_at) > now;
    if (!held || row.lock_node !== nodeId) {
      throw new Error('Node does not hold lock for this job');
    }
    return row;
  }

  async handleHeartbeat(jobId, nodeId) {
    const row = await this._assertLock(jobId, nodeId);
    const now = Date.now();

    if (row.status === 'assigned') {
      const job = { ...row.data, status: 'running', startedAt: row.data.startedAt || now, updatedAt: now };
      await this.db.query(
        `UPDATE jobs SET data = $2, status = 'running', updated_at = $3,
           lock_expires_at = $4, heartbeat_at = $3 WHERE id = $1`,
        [jobId, JSON.stringify(job), now, now + LOCK_MS]
      );
    } else {
      await this.db.query(
        'UPDATE jobs SET lock_expires_at = $2, heartbeat_at = $3 WHERE id = $1',
        [jobId, now + LOCK_MS, now]
      );
    }

    return { success: true };
  }

  async storeChunk(jobId, nodeId, chunkData) {
    await this._assertLock(jobId, nodeId);

    const chunk = {
      index: chunkData.chunkIndex,
      content: chunkData.content,
      metrics: chunkData.metrics,
      isFinal: chunkData.isFinal || false,
      timestamp: chunkData.timestamp || Date.now()
    };

    await this.db.query(
      `INSERT INTO job_chunks (job_id, idx, chunk) VALUES ($1, $2, $3)
       ON CONFLICT (job_id, idx) DO UPDATE SET chunk = EXCLUDED.chunk`,
      [jobId, chunk.index, JSON.stringify(chunk)]
    );

    if (chunkData.metrics) {
      const job = await this.getJob(jobId);
      const updated = { ...job, lastMetrics: chunkData.metrics, updatedAt: Date.now() };
      await this.db.query('UPDATE jobs SET data = $2, updated_at = $3 WHERE id = $1',
        [jobId, JSON.stringify(updated), updated.updatedAt]);
    }

    return { success: true, chunkIndex: chunk.index };
  }

  async _getChunks(jobId) {
    const r = await this.db.query('SELECT chunk FROM job_chunks WHERE job_id = $1 ORDER BY idx', [jobId]);
    return r.rows.map((row) => row.chunk);
  }

  async completeJob(jobId, nodeId) {
    await this._assertLock(jobId, nodeId);

    const chunks = await this._getChunks(jobId);
    const assembledContent = chunks.map((c) => c.content).join('');

    const job = await this.updateJobStatus(jobId, 'completed', {
      completedAt: Date.now(),
      result: assembledContent,
      chunks: chunks.length
    });

    await this._releaseLock(jobId);
    return job;
  }

  async failJob(jobId, nodeId, reason) {
    await this._assertLock(jobId, nodeId);

    const job = await this.updateJobStatus(jobId, 'failed', {
      failedAt: Date.now(),
      failureReason: reason
    });

    await this._releaseLock(jobId);
    return job;
  }

  async _releaseLock(jobId) {
    await this.db.query(
      'UPDATE jobs SET lock_node = NULL, lock_expires_at = NULL, heartbeat_at = NULL WHERE id = $1',
      [jobId]
    );
  }

  // Fail jobs that have sat pending past PENDING_TTL_MS. Nothing else ever
  // clears `pending`: checkTimeouts only rescues assigned/running jobs, and
  // cleanupOldJobs only deletes completed/failed ones — so a job queued while no
  // node was serving would otherwise stay forever, and be run hours later for a
  // caller that is long gone. Marked 'failed' rather than given a new status so
  // the existing cleanup sweep collects them and any late reader gets a clear
  // reason instead of an indefinite wait. Returns the expired job ids.
  async expireStalePending() {
    const now = Date.now();
    const r = await this.db.query(
      "SELECT id, data FROM jobs WHERE status = 'pending' AND created_at < $1",
      [now - PENDING_TTL_MS]
    );

    const expired = [];
    for (const row of r.rows) {
      const updated = {
        ...row.data,
        status: 'failed',
        failedAt: now,
        failureReason: 'expired: no node picked this job up',
        updatedAt: now
      };
      await this.db.query(
        "UPDATE jobs SET data = $2, status = 'failed', updated_at = $3 WHERE id = $1",
        [row.id, JSON.stringify(updated), now]
      );
      expired.push(row.id);
    }

    return expired;
  }

  // Return assigned/running jobs whose lock expired or heartbeat went stale.
  async checkTimeouts() {
    const now = Date.now();
    const r = await this.db.query(
      `SELECT id, data, status, lock_expires_at, heartbeat_at FROM jobs
       WHERE status IN ('assigned', 'running')`,
      []
    );

    const timeoutJobs = [];
    for (const row of r.rows) {
      const lockExpired = row.lock_expires_at == null || Number(row.lock_expires_at) <= now;
      const heartbeatStale = row.heartbeat_at != null && now - Number(row.heartbeat_at) > HEARTBEAT_STALE_MS;

      if (lockExpired || heartbeatStale) {
        const job = { ...row.data };
        const updated = {
          ...job,
          status: 'pending',
          previousStatus: row.status,
          returnedToQueue: now,
          timeoutReason: lockExpired ? 'lock_expired' : 'heartbeat_timeout',
          updatedAt: now
        };
        await this.db.query(
          `UPDATE jobs SET data = $2, status = 'pending', assigned_to = NULL, updated_at = $3,
             lock_node = NULL, lock_expires_at = NULL, heartbeat_at = NULL WHERE id = $1`,
          [job.id, JSON.stringify(updated), now]
        );
        timeoutJobs.push(job.id);
      }
    }

    return timeoutJobs;
  }

  async getJobResult(jobId) {
    const job = await this.getJob(jobId);
    if (!job) {
      throw new Error(`Job ${jobId} not found`);
    }

    let chunks = [];
    if (job.status === 'running' || job.status === 'completed') {
      chunks = await this._getChunks(jobId);
    }

    if (job.status === 'completed') {
      return {
        jobId,
        status: 'completed',
        result: job.result,
        chunks,
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

    if (job.status === 'running') {
      return {
        jobId,
        status: 'running',
        partial: chunks.map((c) => c.content).join(''),
        chunks,
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

  async getQueueStats() {
    const r = await this.db.query('SELECT status, count(*)::int AS c FROM jobs GROUP BY status', []);
    const stats = { pending: 0, assigned: 0, running: 0, completed: 0, failed: 0 };
    for (const row of r.rows) {
      if (row.status in stats) {
        stats[row.status] = row.c;
      }
    }
    return stats;
  }

  async cleanupOldJobs(maxAge = 86400000) {
    const cutoff = Date.now() - maxAge;
    // Two bulk deletes instead of a SELECT plus a delete-pair per row: first the
    // chunks belonging to the expiring jobs, then the jobs themselves. The
    // job-count comes from the second statement's RETURNING.
    await this.db.query(
      `DELETE FROM job_chunks WHERE job_id IN (
         SELECT id FROM jobs WHERE status IN ('completed', 'failed') AND updated_at < $1
       )`,
      [cutoff]
    );
    const r = await this.db.query(
      `DELETE FROM jobs WHERE status IN ('completed', 'failed') AND updated_at < $1 RETURNING id`,
      [cutoff]
    );

    return r.rowCount;
  }
}

module.exports = JobService;
