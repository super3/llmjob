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
  const staleHeartbeat = (id) => db.query('UPDATE jobs SET heartbeat_at = $1 WHERE id = $2', [Date.now() - 120000, id]);
  const ageJob = (id) => db.query('UPDATE jobs SET updated_at = 1000 WHERE id = $1', [id]);

  describe('createJob', () => {
    it('creates a job with default values', async () => {
      const job = await jobService.createJob({ prompt: 'Test prompt', userId: 'user123' });
      expect(job).toMatchObject({
        // default model = what the earn-client fleet actually serves
        prompt: 'Test prompt', model: 'Gemma-4-E4B-it-Q4_K_M', status: 'pending',
        userId: 'user123', priority: 0, maxTokens: 1000, temperature: 0.7
      });
      expect(job.id).toMatch(/^job-\d+-[a-z0-9]+$/);
      expect(job.createdAt).toBeDefined();
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

    it('routes only matching-model jobs to a node that reports a served model', async () => {
      // gateway ids on the jobs; the node reports its GGUF name
      await jobService.createJob({ prompt: 'a', userId: 'u', model: 'qwen/qwen3.6-35b-a3b' });
      const want = await jobService.createJob({ prompt: 'b', userId: 'u', model: 'qwen/qwen3.6-27b' });
      await jobService.createJob({ prompt: 'c', userId: 'u', model: 'qwen/qwen3.6-35b-a3b' });

      const assigned = await jobService.assignJobsToNode('node27b', 2, 'Qwen3.6-27B-Q4_K_M');
      expect(assigned).toHaveLength(1);
      expect(assigned[0].id).toBe(want.id);

      // the 35B jobs are still pending for a node that serves that model
      const other = await jobService.assignJobsToNode('node35b', 5, 'Qwen3.6-35B-A3B-Q4_K_M');
      expect(other).toHaveLength(2);
    });

    it('stops at maxJobs even when more matching jobs are in the scan window', async () => {
      for (let i = 0; i < 4; i++) {
        await jobService.createJob({ prompt: 'p' + i, userId: 'u', model: 'qwen/qwen3.6-27b' });
      }
      const assigned = await jobService.assignJobsToNode('n', 2, 'Qwen3.6-27B-Q4_K_M');
      expect(assigned).toHaveLength(2);
    });

    it('a node with no reported model stays model-agnostic (undefined nodeModel)', async () => {
      await jobService.createJob({ prompt: 'p', userId: 'u', model: 'qwen/qwen3.6-27b' });
      expect(await jobService.assignJobsToNode('n', 1, undefined)).toHaveLength(1);
    });
  });

  describe('model matching', () => {
    it('normalizes gateway ids and GGUF names to a comparable key', () => {
      expect(JobService.normalizeModel('qwen/qwen3.6-27b')).toBe('qwen3627b');
      expect(JobService.normalizeModel('Qwen3.6-27B-Q4_K_M')).toBe('qwen3627b');
      expect(JobService.normalizeModel('Qwen3.6-35B-A3B-Q4_K_M')).toBe('qwen3635ba3b');
      expect(JobService.normalizeModel('Gemma-4-E4B-it-Q4_K_M')).toBe('gemma4e4bit');
      expect(JobService.normalizeModel(null)).toBe('');
    });

    it('matches across naming schemes and rejects different models', () => {
      expect(JobService.modelsMatch('qwen/qwen3.6-27b', 'Qwen3.6-27B-Q4_K_M')).toBe(true);
      expect(JobService.modelsMatch('qwen/qwen3.6-35b-a3b', 'Qwen3.6-27B-Q4_K_M')).toBe(false);
      // a missing model on either side never blocks routing
      expect(JobService.modelsMatch(null, 'Qwen3.6-27B-Q4_K_M')).toBe(true);
      expect(JobService.modelsMatch('qwen/qwen3.6-27b', '')).toBe(true);
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
});
