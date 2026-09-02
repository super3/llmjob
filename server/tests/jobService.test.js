const JobService = require('../src/services/jobService');
const { createTestDb } = require('./helpers/pgmem');

describe('JobService', () => {
  let jobService;
  let db;

  beforeEach(async () => {
    db = await createTestDb();
    jobService = new JobService(db);
  });

  afterEach(async () => {
    if (db.end) await db.end();
  });

  const expireLock = (id) => db.query('UPDATE jobs SET lock_expires_at = $1 WHERE id = $2', [Date.now() - 1, id]);
  const staleHeartbeat = (id) => db.query('UPDATE jobs SET heartbeat_at = $1 WHERE id = $2', [Date.now() - 300000, id]);
  const ageJob = (id) => db.query('UPDATE jobs SET updated_at = 1000 WHERE id = $1', [id]);

  describe('createJob', () => {
    it('creates a job with default values', async () => {
      const job = await jobService.createJob({ prompt: 'Test prompt', userId: 'user123' });
      expect(job).toMatchObject({
        // default model = what the earn-client fleet actually serves
        prompt: 'Test prompt', model: 'Gemma-4-E4B-it-Q4_K_M', status: 'pending',
        // maxTokens default covers a reasoning model's thoughts plus its answer
        userId: 'user123', priority: 0, maxTokens: 6400, temperature: 0.7
      });
      expect(job.id).toMatch(/^job-\d+-[a-z0-9]+$/);
      expect(job.createdAt).toBeDefined();
    });

    // `||` swallowed these: temperature 0 is THE deterministic setting and was
    // silently served as 0.7, and max_tokens 0 became the full default budget.
    it('preserves an explicit temperature of 0 instead of coalescing it', async () => {
      const job = await jobService.createJob({ prompt: 'p', userId: 'u', temperature: 0 });
      expect(job.temperature).toBe(0);
    });

    it('preserves an explicit maxTokens of 0', async () => {
      const job = await jobService.createJob({ prompt: 'p', userId: 'u', maxTokens: 0 });
      expect(job.maxTokens).toBe(0);
    });

    // POST /api/jobs hands maxTokens straight from the request body, and it is now
    // written to an int4 column — an oversized value surfaced as a 500 from
    // Postgres rather than a 400. (pg-mem doesn't enforce int4 range, so only the
    // clamp itself can be asserted here.)
    it('clamps a caller-supplied maxTokens to what a node can run', async () => {
      const { clampMaxTokens, DEFAULT_MAX_TOKENS, MAX_TOKENS_CEILING } = JobService;
      // An oversized ask is cut to the node's context window, NOT to the default:
      // the two are different numbers now, so that a caller who wants room can
      // have it while one who says nothing keeps the budget traffic always had.
      expect(clampMaxTokens(2147483647)).toBe(MAX_TOKENS_CEILING);
      expect(clampMaxTokens(Number.MAX_SAFE_INTEGER)).toBe(MAX_TOKENS_CEILING);
      expect(MAX_TOKENS_CEILING).toBeGreaterThan(DEFAULT_MAX_TOKENS);
      // A budget between the two is honoured as asked — this is the whole point
      // of the split, and would still pass if the ceiling were wired to DEFAULT.
      expect(clampMaxTokens(DEFAULT_MAX_TOKENS + 1)).toBe(DEFAULT_MAX_TOKENS + 1);
      expect(clampMaxTokens(16384)).toBe(16384);
      // Absent or unusable still falls back to the default, not the ceiling.
      expect(clampMaxTokens(-5)).toBe(DEFAULT_MAX_TOKENS);
      expect(clampMaxTokens('abc')).toBe(DEFAULT_MAX_TOKENS);
      expect(clampMaxTokens(Infinity)).toBe(DEFAULT_MAX_TOKENS);
      expect(clampMaxTokens(undefined)).toBe(DEFAULT_MAX_TOKENS);
      expect(clampMaxTokens(0)).toBe(0);      // meaningful, kept
      expect(clampMaxTokens(512.9)).toBe(512);
      const job = await jobService.createJob({ prompt: 'p', userId: 'u', maxTokens: 2147483647 });
      expect(job.maxTokens).toBe(MAX_TOKENS_CEILING);
    });

    // Unknown speed means no limit, not zero capacity — otherwise a node with a
    // rejected sample would be gated out of everything it could actually serve.
    it('tokenCapacity and admissionLimit report unknown rather than zero', async () => {
      const { tokenCapacity, admissionLimit, TYPICAL_TOKENS } = JobService;
      expect(tokenCapacity(null)).toBeNull();
      expect(tokenCapacity(0)).toBeNull();
      expect(admissionLimit(null)).toBeNull();
      // Fast enough for a typical reply → ungated entirely.
      expect(admissionLimit(TYPICAL_TOKENS / 224 + 1)).toBeNull();
      // Too slow for one → held to what it can finish.
      expect(admissionLimit(5)).toBe(tokenCapacity(5));
    });

    // assignJobsToNode orders the GLOBAL queue by priority DESC and every in-repo
    // producer writes 0, so an unbounded caller value was a starvation lever.
    it('clamps a caller-supplied priority into range', async () => {
      const { MAX_PRIORITY, MIN_PRIORITY, clampPriority } = JobService;
      const huge = await jobService.createJob({ prompt: 'p', userId: 'u', priority: 2147483647 });
      expect(huge.priority).toBe(MAX_PRIORITY);

      expect(clampPriority(undefined)).toBe(0);
      expect(clampPriority('abc')).toBe(0);
      // The floor is below zero on purpose: a negative priority can only yield to
      // other traffic, never jump it, and the benchmark sweeper relies on it to
      // stay behind real requests.
      expect(clampPriority(-5)).toBe(MIN_PRIORITY);
      expect(clampPriority(MIN_PRIORITY)).toBe(MIN_PRIORITY);
      expect(clampPriority(0)).toBe(0);
      expect(clampPriority(Infinity)).toBe(0);
      expect(clampPriority(3)).toBe(3);
      expect(clampPriority(3.9)).toBe(3);
      expect(clampPriority(MAX_PRIORITY + 1)).toBe(MAX_PRIORITY);
    });

    it('creates a job with custom values', async () => {
      const job = await jobService.createJob({
        prompt: 'p', userId: 'user123', priority: 10, model: 'llama3.2:7b',
        temperature: 0.9, options: { invalid: 'option' }, maxTokens: 4096
      });
      expect(job.priority).toBe(10);
      expect(job.model).toBe('llama3.2:7b');
      expect(job.temperature).toBe(0.9);
      expect(job.options).toEqual({ invalid: 'option' });
      expect(job.maxTokens).toBe(4096);
    });

    it('keeps an empty prompt', async () => {
      const job = await jobService.createJob({ prompt: '', userId: 'u' });
      expect(job.prompt).toBe('');
      expect(job.id).toBeDefined();
    });

    it('records the routing visibility (public by default, private when set)', async () => {
      const pub = await jobService.createJob({ prompt: 'p', userId: 'u' });
      expect(pub.visibility).toBe('public');
      const priv = await jobService.createJob({ prompt: 'p', userId: 'u', visibility: 'private' });
      expect(priv.visibility).toBe('private');
      // stored on the promoted column so the poller can filter on it
      const row = (await db.query('SELECT visibility FROM jobs WHERE id = $1', [priv.id])).rows[0];
      expect(row.visibility).toBe('private');
    });
  });

  describe('getJob', () => {
    it('retrieves a created job', async () => {
      const created = await jobService.createJob({ prompt: 'Test', userId: 'user123' });
      expect(await jobService.getJob(created.id)).toEqual(created);
    });

    it('returns null for a non-existent job', async () => {
      expect(await jobService.getJob('non-existent-job')).toBeNull();
    });
  });

  describe('updateJobStatus', () => {
    it('updates status with extra fields', async () => {
      const job = await jobService.createJob({ prompt: 'Test', userId: 'u' });
      const updated = await jobService.updateJobStatus(job.id, 'assigned', { assignedTo: 'node123', assignedAt: Date.now() });
      expect(updated.status).toBe('assigned');
      expect(updated.assignedTo).toBe('node123');
      expect(updated.assignedAt).toBeDefined();
    });

    it('accepts an arbitrary status value', async () => {
      const job = await jobService.createJob({ prompt: 'Test', userId: 'u' });
      await jobService.updateJobStatus(job.id, 'custom_status');
      expect((await jobService.getJob(job.id)).status).toBe('custom_status');
    });

    it('throws for a non-existent job', async () => {
      await expect(jobService.updateJobStatus('non-existent', 'running'))
        .rejects.toThrow('Job non-existent not found');
    });
  });

  describe('assignJobsToNode', () => {
    it('assigns pending jobs by priority then age', async () => {
      await jobService.createJob({ prompt: 'low', userId: 'u', priority: 1 });
      const job2 = await jobService.createJob({ prompt: 'high', userId: 'u', priority: 10 });
      const job3 = await jobService.createJob({ prompt: 'med', userId: 'u', priority: 5 });

      const assigned = await jobService.assignJobsToNode('node123', 2);
      expect(assigned).toHaveLength(2);
      expect(assigned[0].id).toBe(job2.id);
      expect(assigned[1].id).toBe(job3.id);
      expect(assigned[0].status).toBe('assigned');
      expect(assigned[0].assignedTo).toBe('node123');
    });

    it('uses a default maxJobs when omitted', async () => {
      await jobService.createJob({ prompt: 'p', userId: 'u' });
      expect(await jobService.assignJobsToNode('nodeA')).toHaveLength(1);
    });

    it('does not re-assign a job already taken', async () => {
      await jobService.createJob({ prompt: 'Test', userId: 'u' });
      expect(await jobService.assignJobsToNode('node1', 1)).toHaveLength(1);
      expect(await jobService.assignJobsToNode('node2', 1)).toHaveLength(0);
    });

    it('pins a targeted job to its node and hides it from every other node', async () => {
      const job = await jobService.createJob({ prompt: 'test node A', userId: 'u', targetNode: 'nodeA' });
      expect(job.targetNode).toBe('nodeA');
      // The wrong node never sees it...
      expect(await jobService.assignJobsToNode('nodeB', 1)).toHaveLength(0);
      // ...and it's still pending, waiting for its node.
      expect((await jobService.getJobResult(job.id)).status).toBe('pending');
      // The targeted node gets it.
      const got = await jobService.assignJobsToNode('nodeA', 1);
      expect(got.map((j) => j.id)).toEqual([job.id]);
    });

    it('leaves an untargeted job assignable by any node (target_node null)', async () => {
      const job = await jobService.createJob({ prompt: 'anyone', userId: 'u' });
      expect(job.targetNode).toBeNull();
      expect((await jobService.assignJobsToNode('whoever', 1)).map((j) => j.id)).toEqual([job.id]);
    });

    it('rolls back and rethrows on a query error mid-transaction', async () => {
      await jobService.createJob({ prompt: 'p', userId: 'u' });
      const real = db.connect.bind(db);
      jest.spyOn(db, 'connect').mockImplementationOnce(async () => {
        const client = await real();
        const realQuery = client.query.bind(client);
        let n = 0;
        client.query = (...args) => {
          n += 1;
          return n === 2 ? Promise.reject(new Error('boom')) : realQuery(...args);
        };
        return client;
      });
      await expect(jobService.assignJobsToNode('nodeX', 1)).rejects.toThrow('boom');
    });
  });

  describe('expireStalePending', () => {
    const agePending = (id, ms) => db.query('UPDATE jobs SET created_at = $1 WHERE id = $2', [Date.now() - ms, id]);

    it('fails jobs left pending past the TTL and leaves fresh ones alone', async () => {
      const stale = await jobService.createJob({ prompt: 'nobody picked me up', userId: 'u' });
      const fresh = await jobService.createJob({ prompt: 'just queued', userId: 'u' });
      await agePending(stale.id, 6 * 60 * 1000); // 6 min — past the 5 min TTL

      expect(await jobService.expireStalePending()).toEqual([stale.id]);

      const failed = await jobService.getJobResult(stale.id);
      expect(failed.status).toBe('failed');
      expect(failed.error).toMatch(/expired/);
      expect((await jobService.getJobResult(fresh.id)).status).toBe('pending');
    });

    it('leaves a job still inside the caller-wait window queued', async () => {
      const job = await jobService.createJob({ prompt: 'p', userId: 'u' });
      await agePending(job.id, 90 * 1000); // 90s — a caller may still be waiting
      expect(await jobService.expireStalePending()).toEqual([]);
      expect((await jobService.getJobResult(job.id)).status).toBe('pending');
    });

    it('ignores jobs that are not pending', async () => {
      const job = await jobService.createJob({ prompt: 'p', userId: 'u' });
      await jobService.assignJobsToNode('node1', 1); // now 'assigned'
      await agePending(job.id, 60 * 60 * 1000);
      expect(await jobService.expireStalePending()).toEqual([]);
    });

    it('does not fail a job that got claimed between the SELECT and the UPDATE', async () => {
      // Simulate the race: the SELECT sees a stale pending job, but a node claims
      // it before the guarded UPDATE runs, so 0 rows match and nothing is expired.
      const fakeDb = {
        query: async (sql) => {
          if (/SELECT id, data FROM jobs WHERE status = 'pending'/.test(sql)) {
            return { rows: [{ id: 'j1', data: { id: 'j1' } }] };
          }
          return { rows: [], rowCount: 0 }; // the guarded UPDATE matches nothing
        }
      };
      expect(await new JobService(fakeDb).expireStalePending()).toEqual([]);
    });

    it('lets the normal cleanup sweep collect what it expired', async () => {
      const job = await jobService.createJob({ prompt: 'p', userId: 'u' });
      await agePending(job.id, 6 * 60 * 1000);
      await jobService.expireStalePending();
      await ageJob(job.id); // push updated_at past the cleanup cutoff
      expect(await jobService.cleanupOldJobs()).toBe(1);
    });
  });

  describe('assignJobsToNode routing (public/private)', () => {
    const addNode = (nodeId, userId) => db.query('INSERT INTO nodes (node_id, user_id) VALUES ($1, $2)', [nodeId, userId]);

    it("hands a private job only to a node owned by the job's owner", async () => {
      await addNode('alice-node', 'alice');
      await addNode('bob-node', 'bob');
      const job = await jobService.createJob({ prompt: 'secret', userId: 'alice', visibility: 'private' });

      // Bob's node must never be handed Alice's private job.
      expect(await jobService.assignJobsToNode('bob-node', 5)).toHaveLength(0);
      // Alice's own node gets it.
      const mine = await jobService.assignJobsToNode('alice-node', 5);
      expect(mine.map((j) => j.id)).toEqual([job.id]);
    });

    it('hands public jobs to any node, regardless of owner', async () => {
      await addNode('bob-node', 'bob');
      await jobService.createJob({ prompt: 'open', userId: 'alice', visibility: 'public' });
      const got = await jobService.assignJobsToNode('bob-node', 5);
      expect(got).toHaveLength(1);
      expect(got[0].visibility).toBe('public');
    });

    it('treats a legacy job with NULL visibility as public', async () => {
      await addNode('bob-node', 'bob');
      const job = await jobService.createJob({ prompt: 'p', userId: 'alice' });
      await db.query('UPDATE jobs SET visibility = NULL WHERE id = $1', [job.id]); // pre-feature row
      expect(await jobService.assignJobsToNode('bob-node', 5)).toHaveLength(1);
    });

    it('gives an unclaimed (owner-less) node public jobs but never private ones', async () => {
      await addNode('orphan', null);
      await jobService.createJob({ prompt: 'pub', userId: 'alice', visibility: 'public' });
      await jobService.createJob({ prompt: 'priv', userId: 'alice', visibility: 'private' });
      const got = await jobService.assignJobsToNode('orphan', 5);
      expect(got).toHaveLength(1);
      expect(got[0].visibility).toBe('public');
    });
  });

  describe('handleHeartbeat', () => {
    it('moves an assigned job to running', async () => {
      const job = await jobService.createJob({ prompt: 'Test', userId: 'u' });
      await jobService.assignJobsToNode('node123', 1);
      const result = await jobService.handleHeartbeat(job.id, 'node123');
      expect(result.success).toBe(true);
      expect((await jobService.getJob(job.id)).status).toBe('running');
    });

    it('leaves an already-running job running', async () => {
      const job = await jobService.createJob({ prompt: 'p', userId: 'u' });
      await jobService.assignJobsToNode('nodeA', 1);
      await jobService.handleHeartbeat(job.id, 'nodeA');
      const result = await jobService.handleHeartbeat(job.id, 'nodeA');
      expect(result.success).toBe(true);
      expect((await jobService.getJob(job.id)).status).toBe('running');
    });

    it('rejects a heartbeat from the wrong node', async () => {
      const job = await jobService.createJob({ prompt: 'Test', userId: 'u' });
      await jobService.assignJobsToNode('node123', 1);
      await expect(jobService.handleHeartbeat(job.id, 'wrong-node'))
        .rejects.toThrow('Node does not hold lock for this job');
    });

    it('rejects a heartbeat when no lock is held', async () => {
      const job = await jobService.createJob({ prompt: 'Test', userId: 'u' });
      await expect(jobService.handleHeartbeat(job.id, 'any-node'))
        .rejects.toThrow('Node does not hold lock for this job');
    });
  });

  describe('storeChunk', () => {
    it('stores chunks for the locking node', async () => {
      const job = await jobService.createJob({ prompt: 'Test', userId: 'u' });
      await jobService.assignJobsToNode('node123', 1);
      const c1 = await jobService.storeChunk(job.id, 'node123', { chunkIndex: 0, content: 'Hello ', metrics: { tokensPerSecond: 10 } });
      expect(c1).toEqual({ success: true, chunkIndex: 0 });
      const c2 = await jobService.storeChunk(job.id, 'node123', { chunkIndex: 1, content: 'world!', isFinal: true });
      expect(c2.success).toBe(true);
    });

    it('rejects chunks from the wrong node', async () => {
      const job = await jobService.createJob({ prompt: 'Test', userId: 'u' });
      await jobService.assignJobsToNode('node123', 1);
      await expect(jobService.storeChunk(job.id, 'wrong-node', { chunkIndex: 0, content: 'x' }))
        .rejects.toThrow('Node does not hold lock for this job');
    });
  });

  describe('completeJob / failJob', () => {
    it('completes a job and assembles chunks', async () => {
      const job = await jobService.createJob({ prompt: 'Test', userId: 'u' });
      await jobService.assignJobsToNode('node123', 1);
      await jobService.storeChunk(job.id, 'node123', { chunkIndex: 0, content: 'Hello ' });
      await jobService.storeChunk(job.id, 'node123', { chunkIndex: 1, content: 'world!' });

      const completed = await jobService.completeJob(job.id, 'node123');
      expect(completed.status).toBe('completed');
      expect(completed.result).toBe('Hello world!');
      expect(completed.chunks).toBe(2);
      expect(completed.completedAt).toBeDefined();
    });

    it('rejects completion from the wrong node', async () => {
      const job = await jobService.createJob({ prompt: 'Test', userId: 'u' });
      await jobService.assignJobsToNode('node123', 1);
      await expect(jobService.completeJob(job.id, 'wrong-node'))
        .rejects.toThrow('Node does not hold lock for this job');
    });

    it('fails a job with a reason', async () => {
      const job = await jobService.createJob({ prompt: 'Test', userId: 'u' });
      await jobService.assignJobsToNode('node123', 1);
      const failed = await jobService.failJob(job.id, 'node123', 'Out of memory');
      expect(failed.status).toBe('failed');
      expect(failed.failureReason).toBe('Out of memory');
      expect(failed.failedAt).toBeDefined();
    });

    it('rejects failure from the wrong node', async () => {
      const job = await jobService.createJob({ prompt: 'Test', userId: 'u' });
      await jobService.assignJobsToNode('node123', 1);
      await expect(jobService.failJob(job.id, 'wrong-node', 'err'))
        .rejects.toThrow('Node does not hold lock for this job');
    });
  });

  // Admission control: a node is only offered work it can finish inside the
  // gateway's budget at the speed we measured it at. This is what stops a slow
  // card taking a job it has no chance of completing — the failure mode that put
  // 6 of 6 timeouts on the two slowest nodes of a 198-question benchmark run.
  describe('assignJobsToNode capacity gating', () => {
    // capacity = tps × 280s × 0.8. The gate only engages below TYPICAL_TOKENS of
    // capacity, i.e. under 2048/224 = 9.14 tok/s.
    const addNode = (nodeId, tps, samples = 3) => db.query(
      `INSERT INTO nodes (node_id, measured_tps, speed_samples, speed_at) VALUES ($1, $2, $3, $4)`,
      [nodeId, tps, samples, Date.now()]
    );

    it('skips a job the node is too slow to finish, and takes one it can', async () => {
      await addNode('slow', 5); // 1120 tokens of capacity
      const big = await jobService.createJob({ prompt: 'long', maxTokens: 6400, userId: 'u' });
      const small = await jobService.createJob({ prompt: 'short', maxTokens: 1000, userId: 'u' });

      const got = await jobService.assignJobsToNode('slow', 5);
      expect(got.map((j) => j.id)).toEqual([small.id]);
      // The oversized job is left pending for a node that can actually serve it.
      expect(got.map((j) => j.id)).not.toContain(big.id);
    });

    // The bug this rule replaced: max_tokens is a ceiling and almost every request
    // carries the 6400 default, so gating on it demanded 28.6 tok/s of everything
    // and cut the fleet's slower half out of ALL traffic — the blanket floor the
    // per-job design exists to avoid.
    it('serves default-sized jobs to a node that keeps up with typical traffic', async () => {
      await addNode('modest', 18); // 4032 capacity: under the 6400 ceiling, over typical
      await jobService.createJob({ prompt: 'default', userId: 'u' }); // no maxTokens → 6400
      expect(await jobService.assignJobsToNode('modest', 5)).toHaveLength(1);
    });

    it('gives a fast node everything', async () => {
      await addNode('fast', 45); // 10080 tokens of capacity
      await jobService.createJob({ prompt: 'long', maxTokens: 6400, userId: 'u' });
      await jobService.createJob({ prompt: 'short', maxTokens: 1000, userId: 'u' });
      expect(await jobService.assignJobsToNode('fast', 5)).toHaveLength(2);
    });

    it('does not gate a node on a single sample', async () => {
      // One unlucky generation must not be able to withhold work — the node needs
      // to keep serving to earn a second reading and argue with the first.
      await addNode('onesample', 5, 1);
      await jobService.createJob({ prompt: 'long', maxTokens: 6400, userId: 'u' });
      expect(await jobService.assignJobsToNode('onesample', 5)).toHaveLength(1);
    });

    it('serves an unmeasured node permissively', async () => {
      await db.query('INSERT INTO nodes (node_id) VALUES ($1)', ['cold']);
      await jobService.createJob({ prompt: 'long', maxTokens: 6400, userId: 'u' });
      // The whole fleet is unmeasured the moment this ships; gating on absent data
      // would stall it until the benchmark sweeper caught up.
      expect(await jobService.assignJobsToNode('cold', 5)).toHaveLength(1);
    });

    it('ignores a stale measurement rather than gating on it', async () => {
      await addNode('rusty', 1); // slow enough to gate everything…
      await db.query('UPDATE nodes SET speed_at = $1 WHERE node_id = $2', [Date.now() - 7 * 60 * 60 * 1000, 'rusty']);
      await jobService.createJob({ prompt: 'long', maxTokens: 6400, userId: 'u' });
      // …but the figure describes hardware that may not be there any more.
      expect(await jobService.assignJobsToNode('rusty', 5)).toHaveLength(1);
    });

    it('serves permissively when the stored speed is zero or missing', async () => {
      // A fresh timestamp with no usable rate (a node whose only sample was
      // rejected) must read as "unknown", not as "capacity zero" — otherwise it
      // would be gated out of every job and could never earn a better number.
      await db.query('INSERT INTO nodes (node_id, measured_tps, speed_at) VALUES ($1, NULL, $2)', ['blank', Date.now()]);
      await db.query('INSERT INTO nodes (node_id, measured_tps, speed_at) VALUES ($1, 0, $2)', ['zero', Date.now()]);
      await jobService.createJob({ prompt: 'long', maxTokens: 6400, userId: 'u' });
      expect(await jobService.assignJobsToNode('blank', 5)).toHaveLength(1);
      await jobService.createJob({ prompt: 'long', maxTokens: 6400, userId: 'u' });
      expect(await jobService.assignJobsToNode('zero', 5)).toHaveLength(1);
    });

    it('lets a targeted job through regardless of capacity', async () => {
      await addNode('slow', 5);
      // This is how a benchmark reaches a node in the first place — and a caller
      // who names a node has said what they want.
      await jobService.createJob({ prompt: 'bench', maxTokens: 6400, targetNode: 'slow', userId: 'u' });
      expect(await jobService.assignJobsToNode('slow', 5)).toHaveLength(1);
    });

    it('serves a job with no recorded budget', async () => {
      await addNode('slow', 5);
      await jobService.createJob({ prompt: 'legacy', userId: 'u' });
      await db.query('UPDATE jobs SET max_tokens = NULL', []); // a row from before the column existed
      expect(await jobService.assignJobsToNode('slow', 5)).toHaveLength(1);
    });
  });

  // Every completed job doubles as a speed measurement, so most nodes never need
  // a synthetic benchmark.
  describe('speed sampling on completion', () => {
    const finish = async (nodeId, metrics, backdateTo) => {
      const job = await jobService.createJob({ prompt: 'x', userId: 'u' });
      await jobService.assignJobsToNode(nodeId, 1);
      await jobService.handleHeartbeat(job.id, nodeId);
      if (backdateTo) {
        const d = await jobService.getJob(job.id);
        await db.query('UPDATE jobs SET data = $2 WHERE id = $1',
          [job.id, JSON.stringify({ ...d, assignedAt: backdateTo, startedAt: backdateTo })]);
      }
      await jobService.storeChunk(job.id, nodeId, { chunkIndex: 0, content: 'hi', isFinal: true, metrics });
      await jobService.completeJob(job.id, nodeId);
      return job;
    };

    it('records the node speed from the server clock, not the node\'s claim', async () => {
      const nodes = { recordSpeedSample: jest.fn() };
      jobService = new JobService(db, nodes);
      // Backdate the assignment so the interval is a known, non-trivial value
      // instead of the handful of milliseconds the test itself takes.
      const startedAt = Date.now() - 20000;
      await finish('n1', { totalTokens: 300, tokensPerSecond: 999 }, startedAt);

      expect(nodes.recordSpeedSample).toHaveBeenCalledTimes(1);
      const [nodeId, tokens, elapsedMs] = nodes.recordSpeedSample.mock.calls[0];
      expect(nodeId).toBe('n1');
      expect(tokens).toBe(300);           // the node's token count is fine to trust
      // Pinned to the interval the server actually observed. `toBeGreaterThan(0)`
      // was satisfied by any constant, so replacing the subtraction outright —
      // recording every node at exactly `tokens` tok/s — passed the whole suite.
      expect(elapsedMs).toBeGreaterThanOrEqual(20000);
      expect(elapsedMs).toBeLessThan(21000);
    });

    it('flags a cold node\'s benchmark so its result replaces rather than blends', async () => {
      const nodes = { recordSpeedSample: jest.fn() };
      jobService = new JobService(db, nodes);
      const job = await jobService.createJob({
        prompt: 'bench', userId: 'u', targetNode: 'n2', benchmark: true, benchmarkWarmup: true,
      });
      await jobService.assignJobsToNode('n2', 1);
      await jobService.handleHeartbeat(job.id, 'n2');
      await jobService.storeChunk(job.id, 'n2', { chunkIndex: 0, content: 'x', isFinal: true, metrics: { totalTokens: 50 } });
      await jobService.completeJob(job.id, 'n2');

      expect(nodes.recordSpeedSample).toHaveBeenCalledWith('n2', 50, expect.any(Number), { replace: true });
    });

    it('ignores the node-supplied start time, which the node could delay to look fast', async () => {
      // startedAt is stamped by the node's first heartbeat, so the node picks it.
      // A later start means a shorter measured interval, so simply withholding the
      // first beat reports a faster rate — here 20s of work would read as 5s, and
      // this number decides which jobs the node is offered.
      const nodes = { recordSpeedSample: jest.fn() };
      jobService = new JobService(db, nodes);
      const job = await jobService.createJob({ prompt: 'x', userId: 'u' });
      const [a] = await jobService.assignJobsToNode('n9', 1);
      await jobService.handleHeartbeat(job.id, 'n9', a.lockToken);

      const now = Date.now();
      const data = await jobService.getJob(job.id);
      await db.query('UPDATE jobs SET data = $2 WHERE id = $1',
        [job.id, JSON.stringify({ ...data, assignedAt: now - 20000, startedAt: now - 5000 })]);
      await jobService.storeChunk(job.id, 'n9',
        { chunkIndex: 0, content: 'x', isFinal: true, metrics: { totalTokens: 300 } }, a.lockToken);
      await jobService.completeJob(job.id, 'n9', a.lockToken);

      const [, , elapsedMs] = nodes.recordSpeedSample.mock.calls[0];
      expect(elapsedMs).toBeGreaterThanOrEqual(20000); // the server's assignment stamp
      expect(elapsedMs).toBeLessThan(21000);
    });

    it('measures only the attempt that finished, not a requeued job\'s first try', async () => {
      // checkTimeouts requeues by spreading the old data, and handleHeartbeat keeps
      // `startedAt || now` — so after a requeue startedAt still points at the FIRST
      // attempt. Charging that dead time to whoever finishes the re-run measured a
      // node that generated instantly at 6.7 tok/s, which the EWMA then drags
      // toward the amber line and cuts off from full-length jobs.
      const nodes = { recordSpeedSample: jest.fn() };
      jobService = new JobService(db, nodes);
      const job = await jobService.createJob({ prompt: 'x', userId: 'u' });

      const [a] = await jobService.assignJobsToNode('nodeA', 1);
      await jobService.handleHeartbeat(job.id, 'nodeA', a.lockToken);
      // Node A started 150s ago and went silent.
      const long = Date.now() - 150000;
      const data = await jobService.getJob(job.id);
      await db.query('UPDATE jobs SET data = $2, heartbeat_at = $3 WHERE id = $1',
        [job.id, JSON.stringify({ ...data, startedAt: long, assignedAt: long }), long]);
      expect(await jobService.checkTimeouts()).toEqual([job.id]);

      const [b] = await jobService.assignJobsToNode('nodeB', 1);
      await jobService.handleHeartbeat(job.id, 'nodeB', b.lockToken);
      await jobService.storeChunk(job.id, 'nodeB',
        { chunkIndex: 0, content: 'hi', isFinal: true, metrics: { totalTokens: 1000 } }, b.lockToken);
      await jobService.completeJob(job.id, 'nodeB', b.lockToken);

      const [nodeId, tokens, elapsedMs] = nodes.recordSpeedSample.mock.calls[0];
      expect(nodeId).toBe('nodeB');
      expect(tokens).toBe(1000);
      // Node B's own possession only — seconds, not the 150s+ since node A began.
      expect(elapsedMs).toBeLessThan(10000);
    });

    it('skips the sample when the node reported no usable metrics', async () => {
      const nodes = { recordSpeedSample: jest.fn() };
      jobService = new JobService(db, nodes);
      await finish('n3', undefined);              // no final metrics at all
      await finish('n3', { tokensPerSecond: 20 }); // metrics without a token count
      expect(nodes.recordSpeedSample).not.toHaveBeenCalled();
    });

    it('skips the sample when the job carries no usable start time', async () => {
      // Defensive: assignJobsToNode always stamps assignedAt, but a row that lost
      // it (a hand-edited job, a future code path) would otherwise be measured
      // from epoch 0 — an elapsed of 56 years, recorded as a near-zero rate that
      // gates the node out of everything.
      const nodes = { recordSpeedSample: jest.fn() };
      jobService = new JobService(db, nodes);
      const job = await jobService.createJob({ prompt: 'x', userId: 'u' });
      const [a] = await jobService.assignJobsToNode('n5', 1);
      await jobService.handleHeartbeat(job.id, 'n5', a.lockToken);
      const data = await jobService.getJob(job.id);
      delete data.assignedAt;
      delete data.startedAt;
      await db.query('UPDATE jobs SET data = $2 WHERE id = $1', [job.id, JSON.stringify(data)]);
      await jobService.storeChunk(job.id, 'n5',
        { chunkIndex: 0, content: 'x', isFinal: true, metrics: { totalTokens: 10 } }, a.lockToken);
      await jobService.completeJob(job.id, 'n5', a.lockToken);

      expect(nodes.recordSpeedSample).not.toHaveBeenCalled();
    });

    it('never fails a completed job over a speed sample', async () => {
      const spy = jest.spyOn(console, 'error').mockImplementation(() => {});
      const nodes = { recordSpeedSample: jest.fn().mockRejectedValue(new Error('nodes table down')) };
      jobService = new JobService(db, nodes);

      const job = await finish('n4', { totalTokens: 100 });
      // The caller's completion is what matters; the measurement is a bonus.
      expect((await jobService.getJob(job.id)).status).toBe('completed');
      expect(spy).toHaveBeenCalledWith('Failed to record node speed:', 'nodes table down');
      spy.mockRestore();
    });
  });

  describe('getJobResult', () => {
    it('returns a completed result', async () => {
      const job = await jobService.createJob({ prompt: 'Test', userId: 'u' });
      await jobService.assignJobsToNode('node123', 1);
      await jobService.storeChunk(job.id, 'node123', { chunkIndex: 0, content: 'Test result', metrics: { tokensPerSecond: 15 } });
      await jobService.completeJob(job.id, 'node123');

      const result = await jobService.getJobResult(job.id);
      expect(result.status).toBe('completed');
      expect(result.result).toBe('Test result');
      expect(result.completedAt).toBeDefined();
    });

    it('returns partial results for a running job', async () => {
      const job = await jobService.createJob({ prompt: 'Test', userId: 'u' });
      await jobService.assignJobsToNode('node123', 1);
      await jobService.handleHeartbeat(job.id, 'node123');
      await jobService.storeChunk(job.id, 'node123', { chunkIndex: 0, content: 'Partial ' });
      await jobService.storeChunk(job.id, 'node123', { chunkIndex: 1, content: 'result' });

      const result = await jobService.getJobResult(job.id);
      expect(result.status).toBe('running');
      expect(result.partial).toBe('Partial result');
      expect(result.chunks).toHaveLength(2);
    });

    it('returns failed status for a failed job', async () => {
      const job = await jobService.createJob({ prompt: 'Test', userId: 'u' });
      await jobService.assignJobsToNode('node123', 1);
      await jobService.failJob(job.id, 'node123', 'Connection lost');

      const result = await jobService.getJobResult(job.id);
      expect(result.status).toBe('failed');
      expect(result.error).toBe('Connection lost');
      expect(result.failedAt).toBeDefined();
    });

    it('returns basic info for a pending job', async () => {
      const job = await jobService.createJob({ prompt: 'Test', userId: 'u' });
      const result = await jobService.getJobResult(job.id);
      expect(result.status).toBe('pending');
      expect(result.createdAt).toBeDefined();
    });

    it('throws for a non-existent job', async () => {
      await expect(jobService.getJobResult('non-existent'))
        .rejects.toThrow('Job non-existent not found');
    });
  });

  describe('getQueueStats', () => {
    it('counts known statuses and ignores others', async () => {
      await jobService.createJob({ prompt: 'a', userId: 'u' });
      await jobService.createJob({ prompt: 'b', userId: 'u' });
      const c = await jobService.createJob({ prompt: 'c', userId: 'u' });
      await jobService.assignJobsToNode('node1', 1);
      await jobService.updateJobStatus(c.id, 'custom_status'); // not counted

      const stats = await jobService.getQueueStats();
      expect(stats).toMatchObject({ pending: expect.any(Number), assigned: 1, running: 0, completed: 0, failed: 0 });
      expect(stats.pending).toBeGreaterThanOrEqual(1);
    });
  });

  describe('checkTimeouts', () => {
    // main's purge keeps a requeue from CORRUPTING the answer; this keeps the
    // requeue from firing in the first place. The node beats every 30s, so at the
    // old 60s threshold a single dropped POST threw away a generation that was
    // still running and going to finish.
    it('leaves a job alone when only one heartbeat has been missed', async () => {
      const job = await jobService.createJob({ prompt: 'Test', userId: 'u' });
      await jobService.assignJobsToNode('nodeA', 1);
      await jobService.handleHeartbeat(job.id, 'nodeA');
      await db.query('UPDATE jobs SET heartbeat_at = $1 WHERE id = $2', [Date.now() - 65000, job.id]);

      expect(await jobService.checkTimeouts()).not.toContain(job.id);
    });

    it('returns a job whose lock expired', async () => {
      const job = await jobService.createJob({ prompt: 'Test', userId: 'u' });
      await jobService.assignJobsToNode('node123', 1);
      await expireLock(job.id);

      const timedOut = await jobService.checkTimeouts();
      expect(timedOut).toContain(job.id);
      const updated = await jobService.getJob(job.id);
      expect(updated.status).toBe('pending');
      expect(updated.timeoutReason).toBe('lock_expired');
    });

    it('returns a job whose heartbeat went stale (lock still alive)', async () => {
      const job = await jobService.createJob({ prompt: 'p', userId: 'u' });
      await jobService.assignJobsToNode('nodeA', 1);
      await staleHeartbeat(job.id);

      const timedOut = await jobService.checkTimeouts();
      expect(timedOut).toContain(job.id);
      expect((await jobService.getJob(job.id)).timeoutReason).toBe('heartbeat_timeout');
    });

    it('leaves a freshly assigned, still-locked job alone', async () => {
      await jobService.createJob({ prompt: 'p', userId: 'u' });
      await jobService.assignJobsToNode('nodeA', 1);
      expect(await jobService.checkTimeouts()).toEqual([]);
    });

    it('returns an assigned job that never got a lock', async () => {
      const job = await jobService.createJob({ prompt: 'p', userId: 'u' });
      await jobService.updateJobStatus(job.id, 'assigned'); // no lock set
      expect(await jobService.checkTimeouts()).toContain(job.id);
    });

    it('returns an empty array when there are no in-flight jobs', async () => {
      expect(await jobService.checkTimeouts()).toEqual([]);
    });

    it('clears the timed-out attempt\'s chunks so a shorter re-run is not corrupted', async () => {
      // The corruption vector: a timed-out attempt streamed two chunks; the re-run
      // streams only one. If the old chunks survive, completeJob assembles both
      // attempts and returns spliced garbage.
      const job = await jobService.createJob({ prompt: 'p', userId: 'u' });
      await jobService.assignJobsToNode('nodeA', 1);
      await jobService.storeChunk(job.id, 'nodeA', { chunkIndex: 0, content: 'Hello ' });
      await jobService.storeChunk(job.id, 'nodeA', { chunkIndex: 1, content: 'world!' });
      await expireLock(job.id);

      expect(await jobService.checkTimeouts()).toContain(job.id); // back to pending

      // A second node picks it up and produces a shorter answer.
      await jobService.assignJobsToNode('nodeB', 1);
      await jobService.storeChunk(job.id, 'nodeB', { chunkIndex: 0, content: 'Hi' });
      const completed = await jobService.completeJob(job.id, 'nodeB');
      expect(completed.result).toBe('Hi'); // not 'Hiworld!' — the stale chunk is gone
    });

    it('does not requeue or clear chunks when the job left assigned/running before the UPDATE', async () => {
      // Simulate the race: the SELECT sees a running job, but by the time the
      // guarded UPDATE runs the node has completed it, so 0 rows match.
      const calls = [];
      const fakeDb = {
        query: async (sql) => {
          calls.push(sql);
          if (/SELECT id, data, status/.test(sql)) {
            return { rows: [{ id: 'j1', data: { id: 'j1' }, status: 'running', lock_expires_at: 1, heartbeat_at: null }] };
          }
          return { rows: [], rowCount: 0 }; // the guarded UPDATE matches nothing
        }
      };
      const svc = new JobService(fakeDb);
      expect(await svc.checkTimeouts()).toEqual([]);
      expect(calls.some((s) => /DELETE FROM job_chunks/.test(s))).toBe(false);
    });

    it('clears the lock token when it requeues a job', async () => {
      const job = await jobService.createJob({ prompt: 'p', userId: 'u' });
      await jobService.assignJobsToNode('nodeA', 1);
      await staleHeartbeat(job.id);
      await jobService.checkTimeouts();
      const r = await db.query('SELECT lock_token FROM jobs WHERE id = $1', [job.id]);
      expect(r.rows[0].lock_token).toBeNull();
    });
  });

  // One machine runs a job worker per GPU that can hold the model, and the GUI
  // and CLI share a single node.json — so every worker on a rig signs as the
  // same node id. The lock therefore has to fence the ATTEMPT, not the machine.
  describe('lock token fencing', () => {
    it('issues a distinct lock token with each assignment', async () => {
      await jobService.createJob({ prompt: 'a', userId: 'u' });
      await jobService.createJob({ prompt: 'b', userId: 'u' });
      const assigned = await jobService.assignJobsToNode('nodeA', 2);
      expect(assigned).toHaveLength(2);
      for (const j of assigned) expect(j.lockToken).toMatch(/^[0-9a-f]{32}$/);
      expect(assigned[0].lockToken).not.toBe(assigned[1].lockToken);
    });

    it('rejects a superseded attempt from the SAME node id', async () => {
      const job = await jobService.createJob({ prompt: 'p', userId: 'u' });
      const [first] = await jobService.assignJobsToNode('rig', 1);

      // Worker 1 stalls, the job is requeued, and a sibling worker on the SAME
      // rig picks it up — the node id is identical, only the token differs.
      await staleHeartbeat(job.id);
      await jobService.checkTimeouts();
      const [second] = await jobService.assignJobsToNode('rig', 1);
      expect(second.lockToken).not.toBe(first.lockToken);

      const stale = /Stale job lock/;
      await expect(jobService.storeChunk(job.id, 'rig', { chunkIndex: 0, content: 'x' }, first.lockToken)).rejects.toThrow(stale);
      await expect(jobService.handleHeartbeat(job.id, 'rig', first.lockToken)).rejects.toThrow(stale);
      await expect(jobService.completeJob(job.id, 'rig', first.lockToken)).rejects.toThrow(stale);
      await expect(jobService.failJob(job.id, 'rig', 'boom', first.lockToken)).rejects.toThrow(stale);

      // The live attempt is untouched by its sibling's late writes.
      expect((await jobService.getJob(job.id)).status).toBe('assigned');
      await jobService.storeChunk(job.id, 'rig', { chunkIndex: 0, content: 'good' }, second.lockToken);
      expect((await jobService.completeJob(job.id, 'rig', second.lockToken)).result).toBe('good');
    });

    it('accepts the holder\'s own token', async () => {
      const job = await jobService.createJob({ prompt: 'p', userId: 'u' });
      const [a] = await jobService.assignJobsToNode('rig', 1);
      await expect(jobService.handleHeartbeat(job.id, 'rig', a.lockToken)).resolves.toEqual({ success: true });
      await expect(jobService.completeJob(job.id, 'rig', a.lockToken)).resolves.toMatchObject({ status: 'completed' });
    });

    // Grandfather clause: a client too old to echo the token still works, so a
    // deploy doesn't strand jobs already in flight or un-updated rigs.
    it('accepts a caller that presents no token', async () => {
      const job = await jobService.createJob({ prompt: 'p', userId: 'u' });
      await jobService.assignJobsToNode('rig', 1);
      await jobService.storeChunk(job.id, 'rig', { chunkIndex: 0, content: 'old client' });
      expect((await jobService.completeJob(job.id, 'rig')).result).toBe('old client');
    });

    it('accepts a token when the row has none (job assigned before the upgrade)', async () => {
      const job = await jobService.createJob({ prompt: 'p', userId: 'u' });
      await jobService.assignJobsToNode('rig', 1);
      await db.query('UPDATE jobs SET lock_token = NULL WHERE id = $1', [job.id]);
      await expect(jobService.completeJob(job.id, 'rig', 'some-token')).resolves.toMatchObject({ status: 'completed' });
    });

    it('clears the lock token when a job completes', async () => {
      const job = await jobService.createJob({ prompt: 'p', userId: 'u' });
      const [a] = await jobService.assignJobsToNode('rig', 1);
      await jobService.completeJob(job.id, 'rig', a.lockToken);
      const r = await db.query('SELECT lock_token FROM jobs WHERE id = $1', [job.id]);
      expect(r.rows[0].lock_token).toBeNull();
    });
  });

  describe('cleanupOldJobs', () => {
    it('removes old completed and failed jobs', async () => {
      const completed = await jobService.createJob({ prompt: 'a', userId: 'u' });
      await jobService.assignJobsToNode('n', 1);
      await jobService.completeJob(completed.id, 'n');

      const failed = await jobService.createJob({ prompt: 'b', userId: 'u' });
      await jobService.assignJobsToNode('n', 1);
      await jobService.failJob(failed.id, 'n', 'err');

      await ageJob(completed.id);
      await ageJob(failed.id);

      expect(await jobService.cleanupOldJobs()).toBe(2);
      expect(await jobService.getJob(completed.id)).toBeNull();
      expect(await jobService.getJob(failed.id)).toBeNull();
    });

    it('returns zero when there are no old jobs', async () => {
      expect(await jobService.cleanupOldJobs()).toBe(0);
    });
  });
  describe('model routing', () => {
    // The `model` field was stored and never read: a caller naming a model got
    // whichever node polled next, running whatever it had loaded.
    const liveNode = (nodeId, model) => db.query(
      'INSERT INTO nodes (node_id, public_key, last_seen, model) VALUES ($1, $2, $3, $4)',
      [nodeId, 'k-' + nodeId, Date.now(), model]
    );

    it('pins a job to the model the caller asked for', async () => {
      await liveNode('n-qwen', 'Qwen3.8-27B-UD-Q4_K_XL');
      const job = await jobService.createJob({
        prompt: 'p', userId: 'u', requestedModel: 'Qwen3.8-27B-UD-Q4_K_XL',
      });
      const r = await db.query('SELECT model FROM jobs WHERE id = $1', [job.id]);
      expect(r.rows[0].model).toBe('Qwen3.8-27B-UD-Q4_K_XL');
    });

    it('matches case-insensitively but stores the node\'s spelling', async () => {
      // The id a caller copies out of /v1/models is a filename stem; nobody types
      // it exactly. The poll query compares against what the node reports, so the
      // node's spelling is what has to be stored.
      await liveNode('n-qwen', 'Qwen3.8-27B-UD-Q4_K_XL');
      const job = await jobService.createJob({
        prompt: 'p', userId: 'u', requestedModel: '  qwen3.8-27b-ud-q4_k_xl  ',
      });
      const r = await db.query('SELECT model FROM jobs WHERE id = $1', [job.id]);
      expect(r.rows[0].model).toBe('Qwen3.8-27B-UD-Q4_K_XL');
    });

    it('does not pin a model no live node reports', async () => {
      // The documented contract: an unrecognised id is still served, by whoever
      // takes it. Pinning it would leave the job pending forever.
      await liveNode('n-gemma', 'gemma-4-E4B-it-Q4_K_M');
      const job = await jobService.createJob({ prompt: 'p', userId: 'u', requestedModel: 'gpt-4' });
      const r = await db.query('SELECT model FROM jobs WHERE id = $1', [job.id]);
      expect(r.rows[0].model).toBeNull();
    });

    it('does not pin when the only node running it has gone offline', async () => {
      await db.query('INSERT INTO nodes (node_id, public_key, last_seen, model) VALUES ($1,$2,$3,$4)',
        ['n-old', 'k', Date.now() - (16 * 60 * 1000), 'Qwen3.8-27B-UD-Q4_K_XL']);
      const job = await jobService.createJob({
        prompt: 'p', userId: 'u', requestedModel: 'Qwen3.8-27B-UD-Q4_K_XL',
      });
      const r = await db.query('SELECT model FROM jobs WHERE id = $1', [job.id]);
      expect(r.rows[0].model).toBeNull();
    });

    it('does not pin when no model was asked for', async () => {
      await liveNode('n-qwen', 'Qwen3.8-27B-UD-Q4_K_XL');
      for (const req of [undefined, null, '', '   ', 42]) {
        const job = await jobService.createJob({ prompt: 'p', userId: 'u', requestedModel: req });
        const r = await db.query('SELECT model FROM jobs WHERE id = $1', [job.id]);
        expect(r.rows[0].model).toBeNull();
      }
    });

    it('falls back to jobData.model for direct /api/jobs callers', async () => {
      await liveNode('n-qwen', 'Qwen3.8-27B-UD-Q4_K_XL');
      const job = await jobService.createJob({ prompt: 'p', userId: 'u', model: 'Qwen3.8-27B-UD-Q4_K_XL' });
      const r = await db.query('SELECT model FROM jobs WHERE id = $1', [job.id]);
      expect(r.rows[0].model).toBe('Qwen3.8-27B-UD-Q4_K_XL');
    });

    it('offers a pinned job only to a node running that model', async () => {
      await liveNode('n-qwen', 'Qwen3.8-27B-UD-Q4_K_XL');
      await liveNode('n-gemma', 'gemma-4-E4B-it-Q4_K_M');
      const job = await jobService.createJob({
        prompt: 'p', userId: 'u', requestedModel: 'Qwen3.8-27B-UD-Q4_K_XL',
      });

      expect(await jobService.assignJobsToNode('n-gemma', 5)).toEqual([]);
      const mine = await jobService.assignJobsToNode('n-qwen', 5);
      expect(mine.map((j) => j.id)).toEqual([job.id]);
    });

    it('offers an unpinned job to any node, including one reporting no model', async () => {
      await db.query('INSERT INTO nodes (node_id, public_key, last_seen) VALUES ($1,$2,$3)',
        ['n-bare', 'k', Date.now()]);
      const job = await jobService.createJob({ prompt: 'p', userId: 'u' });
      const got = await jobService.assignJobsToNode('n-bare', 5);
      expect(got.map((j) => j.id)).toEqual([job.id]);
    });

    it('never offers a pinned job to a node reporting no model', async () => {
      // An older client, or one whose model is not up yet: it cannot serve the
      // request as asked, so handing it the job would silently answer with
      // something else.
      await liveNode('n-qwen', 'Qwen3.8-27B-UD-Q4_K_XL');
      await db.query('INSERT INTO nodes (node_id, public_key, last_seen) VALUES ($1,$2,$3)',
        ['n-bare', 'k', Date.now()]);
      await jobService.createJob({ prompt: 'p', userId: 'u', requestedModel: 'Qwen3.8-27B-UD-Q4_K_XL' });
      expect(await jobService.assignJobsToNode('n-bare', 5)).toEqual([]);
    });

    it('is unknown to a node the server has never seen', async () => {
      await jobService.createJob({ prompt: 'p', userId: 'u' });
      expect(await jobService.assignJobsToNode('n-ghost', 5)).toHaveLength(1);
    });
  });
});
