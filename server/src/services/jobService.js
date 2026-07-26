const crypto = require('crypto');

const LOCK_MS = 10 * 60 * 1000;       // assignment lock lifetime (10 min)
const HEARTBEAT_STALE_MS = 60 * 1000; // consider a job stalled after 60s silence
// How long a job may sit pending before it is abandoned. Both gateways give up
// waiting after 120s, so anything older than this has no caller left listening —
// running it later would burn a node's GPU on a reply nobody receives, and the
// rows would otherwise accumulate forever (nothing else clears `pending`). The
// margin over 120s means this can never expire a job someone is still waiting on.
const PENDING_TTL_MS = 5 * 60 * 1000;
// The model a job records when the caller names none. It must match what a node
// reports serving, since assignJobsToNode routes on that name (see modelsMatch):
// this is the network's headline model, and only nodes serving it pick up default
// traffic. Nodes too small for it serve the Gemma tier
// (earn/src/shared/config.js LLM.models) and take jobs that ask for Gemma by name.
const DEFAULT_MODEL = 'Qwen3.6-27B-Q4_K_M';
// Generation budget for a job that doesn't specify one, matched to the node's
// 6400-token context window (earn/src/shared/config.js LLM.ctxSize). The prompt
// shares that window, so a reply tops out a little under this — the node's context
// is the real cap, and this default exists so a caller who sends no max_tokens
// isn't held below it.
const DEFAULT_MAX_TOKENS = 6400;
// How many pending jobs to scan for a model match before giving up on this poll,
// when routing by model. Keeps a node from locking the whole queue while it looks
// for work it can serve. Trade-off: a matching job that sorts behind ASSIGN_SCAN
// non-matching jobs (higher priority, or older) is invisible to a capable node
// until those drain — and if they don't within PENDING_TTL_MS it is expired
// unserved. Bounded (FIFO + the pending TTL), and rare on a lightly-loaded queue;
// a fuller fix would filter by model in SQL (a stored normalized model key).
const ASSIGN_SCAN = 25;

// Canonicalize a model name so the chat-gateway id (e.g. "qwen/qwen3.6-27b") and
// the node's served GGUF name (e.g. "Qwen3.6-27B-Q4_K_M") compare equal: drop a
// "vendor/" prefix, drop a trailing GGUF quant (…-Q4_K_M), and strip separators.
function normalizeModel(name) {
  let s = String(name == null ? '' : name).toLowerCase().trim();
  const slash = s.lastIndexOf('/');
  if (slash >= 0) s = s.slice(slash + 1);       // drop a "vendor/" prefix
  s = s.replace(/[-_.\s]q\d[a-z0-9_]*$/i, '');   // drop a trailing GGUF quant (…-q4_k_m)
  return s.replace(/[^a-z0-9]/g, '');            // canonicalize separators
}

// Can a node serving `nodeModel` run a job that requested `jobModel`? A missing
// model on either side doesn't block routing (model-agnostic job / a node that
// hasn't reported its model yet); otherwise the normalized names must match.
function modelsMatch(jobModel, nodeModel) {
  const a = normalizeModel(jobModel);
  const b = normalizeModel(nodeModel);
  if (!a || !b) return true;
  return a === b;
}

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
      // Optional: pin this job to one specific node (health/perf testing). null
      // means any eligible node may serve it. Only narrows the candidate set — the
      // visibility filter still applies, so it can never route to a node the caller
      // wouldn't otherwise be allowed to use.
      targetNode: jobData.targetNode || null,
      // A caller who sends no max_tokens gets this. 1000 was sized when the whole
      // budget went to the answer; a reasoning model can spend that much thinking
      // and return an empty completion, so the default has to cover the thoughts
      // plus the answer.
      maxTokens: jobData.maxTokens || DEFAULT_MAX_TOKENS,
      temperature: jobData.temperature || 0.7
    };

    // A job from the OpenAI gateway carries a full chat `messages` array so the
    // node can serve multi-turn conversations (the single `prompt` is kept as a
    // display/fallback). Only stored when provided, to leave simple jobs as-is.
    if (Array.isArray(jobData.messages) && jobData.messages.length) {
      job.messages = jobData.messages;
    }

    await this.db.query(
      `INSERT INTO jobs (id, data, status, priority, created_at, updated_at, user_id, visibility, target_node)
       VALUES ($1, $2, 'pending', $3, $4, $4, $5, $6, $7)`,
      [jobId, JSON.stringify(job), job.priority, timestamp, job.userId, job.visibility, job.targetNode]
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
  //
  // When `nodeModel` is given, only jobs that node can serve are also claimed
  // (see modelsMatch), so a rig serving Qwen3.6 27B isn't handed a job that asked
  // for a model it doesn't have — it scans a window of the (visibility-filtered)
  // queue for matching work. Omitting nodeModel keeps the model-agnostic behavior.
  async assignJobsToNode(nodeId, maxJobs = 1, nodeModel) {
    const assignedJobs = [];
    const client = await this.db.connect();
    try {
      await client.query('BEGIN');
      const owner = await client.query('SELECT user_id FROM nodes WHERE node_id = $1', [nodeId]);
      const ownerUserId = owner.rows.length ? owner.rows[0].user_id : null;
      // SKIP LOCKED, not a plain FOR UPDATE: without it, two nodes polling at once
      // both try to lock the same top-priority rows, so the second BLOCKS until the
      // first commits — serializing the whole fleet through one assignment at a
      // time. SKIP LOCKED lets each poller step over rows another is already taking
      // and grab the next ones, so N nodes fan out to N jobs. The target_node filter
      // pins a job to one node when set (null = any eligible node).
      //
      // A model-filtering node selects a WINDOW (ASSIGN_SCAN) rather than just
      // maxJobs, because the match happens per-row below and the jobs it can serve
      // may sit behind ones it can't. That interacts with SKIP LOCKED: the window is
      // locked for the transaction, so a concurrent poller skips those rows instead
      // of blocking on them and may come back empty even though one matched it. The
      // locks last only to COMMIT and the poller retries, so this costs a poll
      // interval, never a job.
      const limit = nodeModel ? Math.max(maxJobs, ASSIGN_SCAN) : maxJobs;
      const pending = await client.query(
        `SELECT id, data FROM jobs
         WHERE status = 'pending'
           AND (visibility IS NULL OR visibility <> 'private' OR user_id = $2)
           AND (target_node IS NULL OR target_node = $3)
         ORDER BY priority DESC, created_at ASC LIMIT $1 FOR UPDATE SKIP LOCKED`,
        [limit, ownerUserId, nodeId]
      );

      for (const row of pending.rows) {
        if (assignedJobs.length >= maxJobs) break;
        if (nodeModel && !modelsMatch(row.data.model, nodeModel)) continue;
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
    // Only thinking models send this, and only on the final chunk — keep it off
    // ordinary chunks rather than storing an empty field on every one.
    if (chunkData.reasoning) chunk.reasoning = chunkData.reasoning;

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

JobService.normalizeModel = normalizeModel;
JobService.modelsMatch = modelsMatch;

module.exports = JobService;
module.exports.DEFAULT_MODEL = DEFAULT_MODEL;
