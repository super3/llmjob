const crypto = require('crypto');
const NodeService = require('./nodeService');
const { SPEED_STALE_MS } = NodeService;

const LOCK_MS = 10 * 60 * 1000;       // assignment lock lifetime (10 min)
// Consider a job stalled after this much silence. The node beats every 30s
// (earn/src/main/jobWorker.js heartbeatMs), so this must leave room for more than
// one lost POST. At 60s a SINGLE dropped beat requeued a job that was still
// generating — the chunk purge below keeps that from corrupting the answer, but
// it still throws away a generation that was going to finish. Four missed beats
// is a real outage, not a blip.
const HEARTBEAT_STALE_MS = 120 * 1000;
// How long a job may sit pending before it is abandoned. Both gateways give up
// waiting after 120s, so anything older than this has no caller left listening —
// running it later would burn a node's GPU on a reply nobody receives, and the
// rows would otherwise accumulate forever (nothing else clears `pending`). The
// margin over 120s means this can never expire a job someone is still waiting on.
const PENDING_TTL_MS = 5 * 60 * 1000;
// The model the earn-client fleet actually serves (earn/src/shared/config.js
// LLM.model.name) — the default a job records must match what runs it.
const DEFAULT_MODEL = 'Gemma-4-E4B-it-Q4_K_M';
// Generation budget for a job that doesn't specify one. Deliberately NOT the
// node's context window any more: the two used to be the same 6400 and one
// constant played both roles, so raising the window would have raised what every
// caller silently asks for. Ordinary traffic keeps the budget it already had.
const DEFAULT_MAX_TOKENS = 6400;
// The hard ceiling a caller may raise max_tokens to, matched to the node's
// context window (earn/src/shared/config.js LLM.ctxSize). The prompt shares that
// window, so a reply tops out a little under this. Callers that need room —
// reasoning benchmarks, mainly — opt in by sending max_tokens explicitly; a
// 6400-token cap made AIME unmeasurable, cutting 26% of samples off mid-working.
const MAX_TOKENS_CEILING = 32768;
// How long a node actually has to finish a job, for admission control. The
// gateway gives up at 280s; a node is only offered work it can finish in 80% of
// that at its measured speed, so a sample that's a little optimistic (or a
// prompt that's longer than usual) doesn't turn into a timeout anyway.
// Must match nodeService's OFFLINE_THRESHOLD: a model is only "live" if the node
// reporting it would still be listed as serving.
const NODE_OFFLINE_MS = 15 * 60 * 1000;

const SERVE_BUDGET_MS = 280 * 1000;
const SERVE_MARGIN = 0.8;

// What a request actually generates, as opposed to what it is allowed to.
// `max_tokens` is a ceiling, and almost nobody sets it: the gateway fills in its
// 6400 ceiling for every caller who omits the field, and jobs created directly
// default to the same. Gating on that number therefore demanded 6400/224 = 28.6
// tok/s of EVERY default request and silently cut the fleet's slower half out of
// all traffic — reproducing the blanket floor this design exists to avoid, since
// a node measured at 18 tok/s serves the observed mean completion (2563 tokens
// across a 198-question run) in 142s, well inside the budget.
//
// So a node that can produce a typical reply in time is never withheld work. Only
// one that cannot even manage that is held to what it can finish.
const TYPICAL_TOKENS = 2048;

// The largest job a node running at `tps` should be offered. Null (unknown speed)
// means no limit — see assignJobsToNode for why unmeasured nodes stay permissive.
function tokenCapacity(tps) {
  if (tps == null || !(tps > 0)) return null;
  return Math.floor(tps * (SERVE_BUDGET_MS * SERVE_MARGIN) / 1000);
}

// The ceiling to filter the queue by, or null to offer the node everything.
// Above TYPICAL_TOKENS of capacity there is nothing to protect against: the node
// keeps up with real traffic, and the long tail that might still overrun is
// better handled by the timeout than by removing the node from the pool.
function admissionLimit(tps) {
  const capacity = tokenCapacity(tps);
  return capacity != null && capacity < TYPICAL_TOKENS ? capacity : null;
}

// One sample is never enough to withhold work on. Two means a single unlucky
// generation can't gate a node by itself, and a node is always served while it
// earns its second.
const MIN_SAMPLES_TO_GATE = 2;

// Caller-supplied priority, bounded. assignJobsToNode orders the GLOBAL pending
// queue by priority DESC, and every in-repo producer writes 0 — so an unbounded
// value from POST /api/jobs let one account jump ahead of all paid API and public
// chat traffic indefinitely. A small range keeps the knob useful for genuine
// ordering without letting it become a starvation lever.
//
// The floor sits below zero rather than at it, because this lever only cuts one
// way: a value under 0 can only ever YIELD to other traffic, so it starves
// nobody. The benchmark sweeper needs exactly that — measuring the fleet must
// never delay serving it — and with a floor of 0 its -1 was silently promoted to
// ordinary priority, putting benchmarks level with paying requests.
const MAX_PRIORITY = 10;
const MIN_PRIORITY = -1;
function clampPriority(v) {
  const n = Math.floor(Number(v));
  if (!Number.isFinite(n)) return 0;
  return Math.min(Math.max(n, MIN_PRIORITY), MAX_PRIORITY);
}

// A caller-supplied generation budget, bounded to what a node can actually run
// (its context window) and to what the promoted int4 column can hold.
// 0 is preserved rather than treated as absent: it is a meaningful OpenAI value
// and main's own test pins it. Only a missing, unparseable or negative budget
// falls back to the default.
function clampMaxTokens(v) {
  if (v == null) return DEFAULT_MAX_TOKENS;
  const n = Math.floor(Number(v));
  if (!Number.isFinite(n) || n < 0) return DEFAULT_MAX_TOKENS;
  return Math.min(n, MAX_TOKENS_CEILING);
}

// A stored epoch-ms timestamp as a number, or 0 for anything unusable — so a
// caller can Math.max() over several without NaN swallowing the result.
function toMs(v) {
  const n = Number(v);
  return Number.isFinite(n) && n > 0 ? n : 0;
}

// A job's routing (inherited from the API key that created it): 'private' may
// only run on the owner's own nodes; anything else is 'public' (any node).
function normalizeVisibility(v) {
  return v === 'private' ? 'private' : 'public';
}

class JobService {
  constructor(db, nodes) {
    this.db = db;
    // Used only to fold a completed job into the node's measured speed. Injectable
    // so a test can assert the sample without a second service.
    this.nodes = nodes || new NodeService(db);
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
      priority: clampPriority(jobData.priority),
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
      // `!= null`, not `||`: `max_tokens: 0` and especially `temperature: 0` are
      // meaningful OpenAI values — 0 is THE setting for deterministic output, and
      // coalescing it to 0.7 silently served a random sample to every caller that
      // asked for a repeatable one.
      // Clamped, not merely defaulted: POST /api/jobs hands this straight from the
      // request body, and it is now written to an integer column — an oversized
      // value used to surface as a 500 from Postgres rather than a 400.
      maxTokens: clampMaxTokens(jobData.maxTokens),
      temperature: jobData.temperature != null ? jobData.temperature : 0.7
    };

    // A job from the OpenAI gateway carries a full chat `messages` array so the
    // node can serve multi-turn conversations (the single `prompt` is kept as a
    // display/fallback). Only stored when provided, to leave simple jobs as-is.
    if (Array.isArray(jobData.messages) && jobData.messages.length) {
      job.messages = jobData.messages;
    }

    // Marks a job the server queued to measure a node rather than to serve a
    // caller. `benchmarkWarmup` says this is the node's first ever measurement,
    // so its result replaces the stored speed instead of blending into it.
    if (jobData.benchmark) {
      job.benchmark = true;
      job.benchmarkWarmup = !!jobData.benchmarkWarmup;
    }

    // An explicit node target outranks the model name, exactly as it already
    // outranks the capacity filter above: the caller named a machine, and a
    // machine serves what it serves. Without that, a request carrying both --
    // "this node" and a model that node is not running -- was excluded by the
    // model filter from its target and by the target filter from everyone else,
    // so it was unassignable by anyone and the caller waited out the whole
    // gateway budget for a 504.
    //
    // Pin the job to nodes running the requested model -- but ONLY when the fleet
    // actually advertises it. A name no live node reports resolves to null, which
    // is today's behaviour: served by whoever polls next, running whatever they
    // have. That keeps the documented contract ("any unrecognised model id is
    // served by it too, so this is a guide rather than a whitelist") while making
    // a name that IS live actually mean something.
    //
    // Resolved here rather than in the poll query so an unmatchable name cannot
    // leave a job pending forever waiting for a node that does not exist.
    // requestedModel is a ROUTING-only channel, deliberately separate from
    // jobData.model. The gateway must not let a caller's string into data.model,
    // because that is echoed back as "the model that ran" -- which is how a
    // request for "gpt-4" once came back looking like the fleet had served it.
    // Pinning is safe: it only narrows which node may take the job.
    const pinnedModel = await this._resolveModelPin(
      jobData.requestedModel != null ? jobData.requestedModel : jobData.model,
      job.visibility === 'private' ? job.userId : null
    );

    await this.db.query(
      `INSERT INTO jobs (id, data, status, priority, created_at, updated_at, user_id, visibility, target_node, max_tokens, model)
       VALUES ($1, $2, 'pending', $3, $4, $4, $5, $6, $7, $8, $9)`,
      [jobId, JSON.stringify(job), job.priority, timestamp, job.userId, job.visibility, job.targetNode, job.maxTokens,
        pinnedModel]
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
      const owner = await client.query(
        'SELECT user_id, measured_tps, speed_at, speed_samples, model FROM nodes WHERE node_id = $1', [nodeId]
      );
      const ownerUserId = owner.rows.length ? owner.rows[0].user_id : null;
      // What this node currently has loaded. A job pinned to a model is only
      // offered to a node running it; a node that reports nothing (an old client,
      // or one whose model is not up yet) is offered unpinned work only, rather
      // than being handed a job it cannot serve as asked.
      const nodeModel = owner.rows.length ? owner.rows[0].model : null;

      // Admission control. A node is only offered a job it can finish inside the
      // gateway's budget at the speed we measured it at — which is what stops a
      // slow card accepting an oversized job it has no chance of completing, then
      // burning ten minutes of GPU on a reply the caller already gave up on.
      //
      // Deliberately per-job rather than a per-node on/off switch, and keyed on
      // TYPICAL_TOKENS rather than on the request's ceiling — see admissionLimit.
      // A node that keeps up with real traffic is never withheld work; only one
      // that cannot manage even a typical reply is held to what it can finish.
      //
      // Three ways out of the gate, all deliberate. An unmeasured node has no
      // limit rather than no work: the whole fleet is unmeasured the moment this
      // ships, and a stricter rule would stall it until the sweeper caught up. A
      // node with a single sample is likewise ungated — one unlucky generation
      // must not be able to gate a node by itself, and it needs to keep serving
      // to earn its second. And targeted jobs bypass the check entirely, which is
      // how a benchmark reaches a node at all.
      const speedAt = owner.rows.length ? Number(owner.rows[0].speed_at) : NaN;
      const fresh = Number.isFinite(speedAt) && (Date.now() - speedAt) <= SPEED_STALE_MS;
      const samples = owner.rows.length ? Number(owner.rows[0].speed_samples) || 0 : 0;
      const capacity = fresh && samples >= MIN_SAMPLES_TO_GATE
        ? admissionLimit(Number(owner.rows[0].measured_tps))
        : null;
      // SKIP LOCKED, not a plain FOR UPDATE: without it, two nodes polling at once
      // both try to lock the same top-priority rows, so the second BLOCKS until the
      // first commits — serializing the whole fleet through one assignment at a
      // time. SKIP LOCKED lets each poller step over rows another is already taking
      // and grab the next ones, so N nodes fan out to N jobs. The target_node filter
      // pins a job to one node when set (null = any eligible node).
      const pending = await client.query(
        `SELECT id, data FROM jobs
         WHERE status = 'pending'
           AND (visibility IS NULL OR visibility <> 'private' OR user_id = $2)
           AND (target_node IS NULL OR target_node = $3)
           AND ($4::int IS NULL OR target_node IS NOT NULL OR max_tokens IS NULL OR max_tokens <= $4)
           AND (model IS NULL OR target_node IS NOT NULL OR model = $5)
         ORDER BY priority DESC, created_at ASC LIMIT $1 FOR UPDATE SKIP LOCKED`,
        [maxJobs, ownerUserId, nodeId, capacity, nodeModel]
      );

      for (const row of pending.rows) {
        const now = Date.now();
        // Fence this attempt. `lock_node` identifies the machine, and one machine
        // deliberately runs several workers under a single node id (one per GPU
        // that fits the model, plus a shared GUI/CLI identity), so the node id
        // alone cannot tell one attempt from another — see _assertLock.
        const lockToken = crypto.randomBytes(16).toString('hex');
        const job = { ...row.data, status: 'assigned', assignedTo: nodeId, assignedAt: now, updatedAt: now };
        await client.query(
          `UPDATE jobs SET data = $2, status = 'assigned', assigned_to = $3, updated_at = $4,
             lock_node = $3, lock_token = $6, lock_expires_at = $5 WHERE id = $1`,
          [job.id, JSON.stringify(job), nodeId, now, now + LOCK_MS, lockToken]
        );
        // Handed to the node (and only to the node) so it can present it back on
        // every write for this job. Kept off the stored `data` blob, which is
        // readable by the job's submitter via GET /api/jobs/:id.
        assignedJobs.push({ ...job, lockToken });
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

  // Verify the caller holds a live lock on the job; returns the row or throws.
  //
  // The node id proves which MACHINE is calling, never which attempt. One machine
  // runs a job worker per GPU that can hold the model, and the GUI and CLI share a
  // single node.json — so every worker on a rig signs as the same node. Without a
  // second factor, a worker whose job was requeued for a stale heartbeat could
  // still post chunks, /complete or /fail against the sibling that had since
  // picked the job up, killing a job that was running correctly.
  //
  // `lockToken` is that second factor: minted per assignment, cleared on release
  // and on requeue, so a stale attempt presents a token that no longer matches.
  //
  // A caller that presents no token is accepted when the row has one, which is
  // what lets jobs assigned before this deploy — and clients not yet updated —
  // finish normally. Those callers are no worse off than before; an updated
  // client on both sides of a requeue is fully fenced. The grandfather clause can
  // be dropped once the fleet has rolled over.
  async _assertLock(jobId, nodeId, lockToken) {
    const r = await this.db.query(
      'SELECT data, status, lock_node, lock_token, lock_expires_at FROM jobs WHERE id = $1',
      [jobId]
    );
    const row = r.rows[0];
    const now = Date.now();
    const held = row && row.lock_expires_at != null && Number(row.lock_expires_at) > now;
    if (!held || row.lock_node !== nodeId) {
      throw new Error('Node does not hold lock for this job');
    }
    if (lockToken != null && row.lock_token != null && row.lock_token !== lockToken) {
      throw new Error('Stale job lock: this attempt was superseded');
    }
    return row;
  }

  async handleHeartbeat(jobId, nodeId, lockToken) {
    const row = await this._assertLock(jobId, nodeId, lockToken);
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

  async storeChunk(jobId, nodeId, chunkData, lockToken) {
    await this._assertLock(jobId, nodeId, lockToken);

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

  async completeJob(jobId, nodeId, lockToken) {
    const row = await this._assertLock(jobId, nodeId, lockToken);

    const chunks = await this._getChunks(jobId);
    const assembledContent = chunks.map((c) => c.content).join('');
    const completedAt = Date.now();

    const job = await this.updateJobStatus(jobId, 'completed', {
      completedAt,
      result: assembledContent,
      chunks: chunks.length
    });

    await this._releaseLock(jobId);
    // Every real job is a speed measurement, so most nodes never need a synthetic
    // benchmark — this keeps the number current, for free, under exactly the
    // conditions the node actually serves under (miner co-running and all).
    await this._recordSpeed(nodeId, row, chunks, completedAt);
    return job;
  }

  // The model id to pin a job to, or null for "any eligible node".
  //
  // Case-insensitive against what nodes report, because the id a caller copies
  // out of /v1/models is the node's own filename stem and nobody types that
  // exactly. Returns the node's spelling, not the caller's, so the poll query
  // compares like with like.
  // `ownerUserId` is set for a private-visibility request, whose job may only be
  // served by its owner's nodes: resolving against the whole fleet would pin it
  // to a model no eligible node runs, and the caller would wait out the gateway
  // budget for nothing.
  async _resolveModelPin(requested, ownerUserId) {
    const want = typeof requested === 'string' ? requested.trim() : '';
    if (!want) return null;
    // Matching is case-insensitive but the poll filter compares exactly, so WHICH
    // spelling we store decides which nodes can take the job. Rank by how many
    // nodes use each spelling and break ties deterministically -- an unordered
    // LIMIT 1 could pin to a lone typo and exclude the rest of the fleet, and the
    // same request could resolve differently on two calls.
    const r = await this.db.query(
      `SELECT model, COUNT(*)::int AS nodes FROM nodes
        WHERE last_seen >= $1 AND model IS NOT NULL AND lower(model) = lower($2)
          AND ($3::text IS NULL OR user_id = $3)
        GROUP BY model ORDER BY nodes DESC, model ASC LIMIT 1`,
      [Date.now() - NODE_OFFLINE_MS, want, ownerUserId == null ? null : ownerUserId]
    );
    return r.rows.length ? r.rows[0].model : null;
  }

  // Fold this job's generation into the node's measured speed. Token count comes
  // from the node's final-chunk metrics; the clock is the server's, taken from
  // when the node picked the job up. Best-effort: a speed sample is never worth
  // failing a completed job over.
  async _recordSpeed(nodeId, row, chunks, completedAt) {
    try {
      const data = row.data;
      const final = chunks.find((c) => c && c.isFinal && c.metrics);
      const tokens = final ? Number(final.metrics.totalTokens) : NaN;
      // assignedAt only, and both halves of that matter.
      //
      // Not startedAt: that is stamped by the node's first heartbeat, so the node
      // chooses it. Since a later start means a shorter measured interval, a node
      // that simply delays its first beat by a minute reports a materially faster
      // rate — on a 138s job, 18 tok/s becomes 31. This number decides which jobs
      // a node is offered, so it cannot have an input the node controls.
      //
      // And assignedAt specifically because assignJobsToNode rewrites it on every
      // assignment. checkTimeouts requeues by spreading the old data and
      // handleHeartbeat keeps `startedAt || now`, so after a requeue startedAt
      // still points at the FIRST attempt — charging a dead node's silence to
      // whoever finishes the re-run measured a node that generated 1000 tokens
      // instantly at 6.7 tok/s.
      //
      // The cost is that the poll-to-first-token gap counts as generation time.
      // That is a second or two, it errs toward reporting a node as slower than it
      // is, and it is the server's own clock end to end.
      const begun = toMs(data.assignedAt);
      if (!Number.isFinite(tokens) || !(begun > 0)) return;
      await this.nodes.recordSpeedSample(nodeId, tokens, completedAt - begun, {
        // A benchmark of a node we've never measured replaces rather than blends:
        // its first run carries model load and KV warm-up, so it reads far slower
        // than the node really is.
        replace: !!(row.data && row.data.benchmark && row.data.benchmarkWarmup),
      });
    } catch (e) {
      console.error('Failed to record node speed:', e.message);
    }
  }

  async failJob(jobId, nodeId, reason, lockToken) {
    await this._assertLock(jobId, nodeId, lockToken);

    const job = await this.updateJobStatus(jobId, 'failed', {
      failedAt: Date.now(),
      failureReason: reason
    });

    await this._releaseLock(jobId);
    return job;
  }

  async _releaseLock(jobId) {
    await this.db.query(
      'UPDATE jobs SET lock_node = NULL, lock_token = NULL, lock_expires_at = NULL, heartbeat_at = NULL WHERE id = $1',
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
      // Guard on the current status: if a node claimed this job between the SELECT
      // above and this UPDATE, it is no longer pending, and stomping it to 'failed'
      // would orphan a job the node is actively running (status 'failed' while
      // lock_node is still set). Only count the ones we actually expired.
      const res = await this.db.query(
        "UPDATE jobs SET data = $2, status = 'failed', updated_at = $3 WHERE id = $1 AND status = 'pending'",
        [row.id, JSON.stringify(updated), now]
      );
      if (res.rowCount) expired.push(row.id);
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
        // Guard on the current status: if the node completed or failed the job
        // between the SELECT above and this UPDATE, don't clobber that terminal
        // state back to 'pending' — that would drop the result and re-run work the
        // caller already got (or is about to).
        const res = await this.db.query(
          `UPDATE jobs SET data = $2, status = 'pending', assigned_to = NULL, updated_at = $3,
             lock_node = NULL, lock_token = NULL, lock_expires_at = NULL, heartbeat_at = NULL
           WHERE id = $1 AND status IN ('assigned', 'running')`,
          [job.id, JSON.stringify(updated), now]
        );
        if (res.rowCount) {
          // Clear the previous attempt's streamed chunks. A re-run starts again at
          // chunk index 0 and storeChunk upserts by (job_id, idx), while
          // completeJob assembles EVERY chunk row by index — so leaving the old
          // rows would splice a dead attempt's trailing output onto the new
          // result. Nothing else clears them until the job is deleted.
          await this.db.query('DELETE FROM job_chunks WHERE job_id = $1', [job.id]);
          timeoutJobs.push(job.id);
        }
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
module.exports.DEFAULT_MODEL = DEFAULT_MODEL;
module.exports.MAX_PRIORITY = MAX_PRIORITY;
module.exports.MIN_PRIORITY = MIN_PRIORITY;
module.exports.TYPICAL_TOKENS = TYPICAL_TOKENS;
module.exports.MIN_SAMPLES_TO_GATE = MIN_SAMPLES_TO_GATE;
module.exports.admissionLimit = admissionLimit;
module.exports.tokenCapacity = tokenCapacity;
module.exports.DEFAULT_MAX_TOKENS = DEFAULT_MAX_TOKENS;
module.exports.MAX_TOKENS_CEILING = MAX_TOKENS_CEILING;
module.exports.clampMaxTokens = clampMaxTokens;
module.exports.clampPriority = clampPriority;
