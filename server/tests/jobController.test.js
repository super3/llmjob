const JobController = require('../src/controllers/jobController');
const JobService = require('../src/services/jobService');
const { createTestDb } = require('./helpers/pgmem');

describe('JobController', () => {
  let jobController;
  let jobService;
  let nodeService;
  let db;
  let req, res;
  let consoleErrorSpy;

  beforeEach(async () => {
    // Mock console.error for error tests
    consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation();
    db = await createTestDb();
    jobService = new JobService(db);

    // Mock nodeService
    nodeService = {
      getNode: jest.fn()
    };

    jobController = new JobController(jobService, nodeService);

    // Setup request and response mocks
    req = {
      body: {},
      params: {},
      user: { id: 'user123' },
      verifiedNode: { nodeId: 'node123', publicKey: 'test-public-key' }
    };
    
    res = {
      status: jest.fn().mockReturnThis(),
      json: jest.fn(),
      send: jest.fn()
    };
  });

  afterEach(async () => {
    if (db.end) await db.end();
    jest.clearAllMocks();
    consoleErrorSpy.mockRestore();
  });

  describe('submitJob', () => {
    it('should submit a new job successfully', async () => {
      req.body = {
        prompt: 'Test prompt',
        model: 'llama3.2:3b',
        priority: 5
      };

      await jobController.submitJob(req, res);

      expect(res.status).toHaveBeenCalledWith(201);
      expect(res.json).toHaveBeenCalledWith({
        success: true,
        job: expect.objectContaining({
          prompt: 'Test prompt',
          model: 'llama3.2:3b',
          priority: 5,
          status: 'pending',
          userId: 'user123'
        })
      });
    });

    it('should reject job without prompt', async () => {
      req.body = { model: 'llama3.2:3b' };

      await jobController.submitJob(req, res);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({ error: 'Prompt is required' });
    });

    it('should fall back to an anonymous user id when no auth user is present', async () => {
      req.user = undefined;
      req.body = { prompt: 'Test prompt' };

      await jobController.submitJob(req, res);

      expect(res.status).toHaveBeenCalledWith(201);
      expect(res.json).toHaveBeenCalledWith({
        success: true,
        job: expect.objectContaining({ userId: 'anonymous' })
      });
    });
  });

  describe('pollJobs', () => {
    beforeEach(async () => {
      // Create some test jobs
      await jobService.createJob({ prompt: 'Test 1', userId: 'user123' });
      await jobService.createJob({ prompt: 'Test 2', userId: 'user123' });
    });

    describe('long-poll', () => {
      // The node used to back off exponentially on empty polls -- 5s doubling to
      // 60s -- so an idle rig sat on a 20-40s rung and a job waited that long
      // just to be ASKED for. Holding the request open moves that wait inside
      // the request, where it ends the moment work exists.
      const held = (opts) => {
        const clock = { t: 0 };
        const ctl = new JobController(jobService, nodeService, Object.assign({
          holdMs: 1000,
          idleRecheckMs: 100,
          now: () => clock.t,
          // Injected: advancing the clock IS the wait, so no real time passes.
          awaitQueue: async (since, ms) => { clock.t += ms; },
        }, opts));
        return { ctl, clock };
      };

      beforeEach(() => {
        nodeService.getNode.mockResolvedValue({ nodeId: 'node123', publicKey: 'test-public-key', status: 'online' });
        req.body = { nodeId: 'node123' };
      });

      it('answers at once when a job is already waiting, without holding', async () => {
        const { ctl } = held();
        const spy = jest.spyOn(jobService, 'assignJobsToNode');
        await ctl.pollJobs(req, res);
        expect(spy).toHaveBeenCalledTimes(1);
        expect(res.json.mock.calls[0][0].jobs.length).toBeGreaterThan(0);
      });

      it('returns the instant a job appears mid-hold', async () => {
        const { ctl } = held();
        let n = 0;
        jest.spyOn(jobService, 'assignJobsToNode').mockImplementation(async () => {
          n += 1;
          return n < 3 ? [] : [{ id: 'j1', prompt: 'p', lockToken: 'tok' }];
        });
        await ctl.pollJobs(req, res);
        expect(n).toBe(3);                                   // two empties, then work
        expect(res.json.mock.calls[0][0].jobs[0].id).toBe('j1');
      });

      it('gives up at the deadline and answers empty', async () => {
        const { ctl } = held();
        jest.spyOn(jobService, 'assignJobsToNode').mockResolvedValue([]);
        await ctl.pollJobs(req, res);
        expect(res.json).toHaveBeenCalledWith({ success: true, jobs: [] });
      });

      it('stops claiming when the node hangs up mid-hold', async () => {
        // Continuing would keep claiming jobs for a connection nobody will read,
        // and every claim locks a job for the full lock window.
        const { ctl } = held();
        const spy = jest.spyOn(jobService, 'assignJobsToNode').mockResolvedValue([]);
        res.destroyed = true;
        await ctl.pollJobs(req, res);
        expect(spy).toHaveBeenCalledTimes(1);                // the first, then it stops
        expect(res.json).not.toHaveBeenCalled();
      });

      it('defaults hold and cadence when not configured', async () => {
        const ctl = new JobController(jobService, nodeService);
        expect(ctl.holdMs).toBe(25000);   // under the node's 30s client timeout
        // The fallback recheck, not the dispatch mechanism: an in-process job
        // wakes the hold immediately via the queue signal. It was 250ms, which
        // cost every idle node 16 SQL statements a second to learn nothing.
        expect(ctl.idleRecheckMs).toBe(2000);
        expect(typeof ctl.now).toBe('function');
        // The default wait is the real queue signal. With the version already
        // moved it short-circuits, which is what makes this safe to await here.
        JobService.signalQueue();
        await expect(ctl.awaitQueue(-1, 0)).resolves.toBeUndefined();
      });

      // The point of the whole change: an idle hold must not re-run the claim on
      // a timer. It claims once, then waits to be told.
      it('claims once and then waits, instead of polling on a timer', async () => {
        const { ctl, clock } = held({ holdMs: 10000, idleRecheckMs: 2000 });
        const spy = jest.spyOn(jobService, 'assignJobsToNode').mockResolvedValue([]);
        await ctl.pollJobs(req, res);
        // 10s hold / 2s fallback = 5 waits, so 6 claims. The old 250ms cadence
        // would have run 41.
        expect(spy).toHaveBeenCalledTimes(6);
        expect(clock.t).toBe(10000);
      });

      it('wakes on the queue signal without waiting for the fallback tick', async () => {
        // A real awaitQueue (not the clock-advancing fake), so this exercises the
        // signal path end to end: the wait resolves because a job was created,
        // not because time passed.
        const clock = { t: 0 };
        const ctl = new JobController(jobService, nodeService, {
          holdMs: 60000,
          idleRecheckMs: 60000,           // a tick this long would never fire in time
          now: () => clock.t,
        });
        let n = 0;
        jest.spyOn(jobService, 'assignJobsToNode').mockImplementation(async () => {
          n += 1;
          return n === 1 ? [] : [{ id: 'j-signal', prompt: 'p', lockToken: 'tok' }];
        });

        const done = ctl.pollJobs(req, res);
        await new Promise((r) => setImmediate(r));  // let the hold reach its wait
        JobService.signalQueue();                   // …and a job shows up
        await done;

        expect(res.json.mock.calls[0][0].jobs[0].id).toBe('j-signal');
        expect(clock.t).toBe(0);                    // no fallback tick was needed
      });

      it('does not sleep through a job created while the claim was in flight', async () => {
        // The race the version counter exists for: the signal fires AFTER the
        // empty claim returned but BEFORE the wait starts. Without capturing the
        // version first, the wait would block for the full fallback tick with
        // work already sitting in the queue.
        const clock = { t: 0 };
        const ctl = new JobController(jobService, nodeService, {
          holdMs: 60000,
          idleRecheckMs: 60000,
          now: () => clock.t,
        });
        let n = 0;
        jest.spyOn(jobService, 'assignJobsToNode').mockImplementation(async () => {
          n += 1;
          if (n === 1) {
            JobService.signalQueue();   // arrives before anyone is waiting
            return [];
          }
          return [{ id: 'j-race', prompt: 'p', lockToken: 'tok' }];
        });

        await ctl.pollJobs(req, res);
        expect(res.json.mock.calls[0][0].jobs[0].id).toBe('j-race');
        expect(clock.t).toBe(0);
      });
    });

    describe('queue signal', () => {
      // Records the timers the wait sets and clears, so "cleans up after itself"
      // is actually asserted rather than assumed.
      const spyTimers = () => {
        const t = { set: 0, cleared: 0, pending: new Map() };
        let id = 0;
        return {
          stats: t,
          timers: {
            setTimeout: (fn, ms) => { t.set++; const k = ++id; t.pending.set(k, { fn, ms }); return k; },
            clearTimeout: (k) => { t.cleared++; t.pending.delete(k); },
          },
          fire: (k) => { const e = t.pending.get(k); t.pending.delete(k); e.fn(); },
        };
      };

      it('resolves a waiter as soon as the queue changes, and clears its timer', async () => {
        const { stats, timers } = spyTimers();
        const since = JobService.queueVersion();
        let resolved = false;
        const wait = JobService.awaitQueueChange(since, 60000, timers).then(() => { resolved = true; });
        await new Promise((r) => setImmediate(r));
        expect(resolved).toBe(false);
        expect(stats.set).toBe(1);

        JobService.signalQueue();
        await wait;
        expect(resolved).toBe(true);
        // The fallback timer must not be left pending — otherwise every held poll
        // leaks one for its full duration.
        expect(stats.cleared).toBe(1);
        expect(stats.pending.size).toBe(0);
      });

      it('returns immediately when the queue already moved, setting no timer', async () => {
        const { stats, timers } = spyTimers();
        const since = JobService.queueVersion();
        JobService.signalQueue();
        await JobService.awaitQueueChange(since, 60000, timers);
        expect(stats.set).toBe(0);
      });

      it('falls back to the timeout when nothing signals, and unregisters', async () => {
        const { timers, fire, stats } = spyTimers();
        const since = JobService.queueVersion();
        const wait = JobService.awaitQueueChange(since, 5, timers);
        fire([...stats.pending.keys()][0]);
        await wait;
        // Having lost the race, the waiter must not be left registered: the next
        // signal has no one to wake and must not throw.
        expect(() => JobService.signalQueue()).not.toThrow();
      });

      it('resolves once even when a scheduler fires synchronously', async () => {
        // The waiter is registered before the timer is set, so a synchronous
        // scheduler reaches the cleanup with no timer handle yet.
        const cleared = [];
        await JobService.awaitQueueChange(JobService.queueVersion(), 0, {
          setTimeout: (fn) => { fn(); return 7; },
          clearTimeout: (k) => cleared.push(k),
        });
        expect(cleared).toEqual([null]);          // nothing pending to cancel
        expect(() => JobService.signalQueue()).not.toThrow();  // and not still registered
      });

      it('uses real timers when none are injected', async () => {
        const since = JobService.queueVersion();
        await JobService.awaitQueueChange(since, 1);
      });

      it('signalling with no waiters is a no-op', () => {
        expect(() => JobService.signalQueue()).not.toThrow();
        expect(JobService.queueVersion()).toBeGreaterThan(0);
      });

      it('createJob signals the queue so a held poll wakes', async () => {
        const before = JobService.queueVersion();
        await jobService.createJob({ prompt: 'wake up', userId: 'user123' });
        expect(JobService.queueVersion()).toBeGreaterThan(before);
      });

      it('checkTimeouts signals only when it actually requeued something', async () => {
        const quiet = JobService.queueVersion();
        expect(await jobService.checkTimeouts()).toEqual([]);
        expect(JobService.queueVersion()).toBe(quiet);   // nothing requeued, nothing to say

        // Claim whatever is at the head of the queue, then strand it.
        const [claimed] = await jobService.assignJobsToNode('node123', 1);
        await db.query('UPDATE jobs SET lock_expires_at = $1 WHERE id = $2', [Date.now() - 1, claimed.id]);
        const before = JobService.queueVersion();
        expect(await jobService.checkTimeouts()).toEqual([claimed.id]);
        expect(JobService.queueVersion()).toBeGreaterThan(before);
      });
    });

    // maxJobs lands in a SQL LIMIT and node enrollment is open to the internet,
    // so an unclamped value let one anonymous node lock the entire public queue
    // — a fleet-wide DoS and a way to be handed every caller's prompts.
    it('clamps an absurd maxJobs to the per-poll ceiling', async () => {
      nodeService.getNode.mockResolvedValue({ nodeId: 'node123', publicKey: 'test-public-key', status: 'online' });
      const spy = jest.spyOn(jobService, 'assignJobsToNode');

      req.body = { nodeId: 'node123', maxJobs: 100000 };
      await jobController.pollJobs(req, res);

      expect(spy).toHaveBeenCalledWith('node123', JobController.MAX_JOBS_PER_POLL);
    });

    // The fence is only as good as its plumbing: the token has to reach the node
    // on /poll and be forwarded back on every write. All five of those hops are
    // one-liners that would revert silently — the service-level tests still pass
    // because they call the service directly, bypassing the controller.
    it('hands the assignment\'s lockToken to the node', async () => {
      nodeService.getNode.mockResolvedValue({ nodeId: 'node123', publicKey: 'test-public-key', status: 'online' });
      req.body = { nodeId: 'node123', maxJobs: 1 };
      await jobController.pollJobs(req, res);

      const [payload] = res.json.mock.calls[0];
      expect(payload.jobs).toHaveLength(1);
      expect(payload.jobs[0].lockToken).toMatch(/^[0-9a-f]{32}$/);
    });

    it('forwards the lockToken to the service on every write', async () => {
      nodeService.getNode.mockResolvedValue({ nodeId: 'node123', publicKey: 'test-public-key', status: 'online' });
      req.body = { nodeId: 'node123', maxJobs: 1 };
      await jobController.pollJobs(req, res);
      const { id, lockToken } = res.json.mock.calls[0][0].jobs[0];

      const beat = jest.spyOn(jobService, 'handleHeartbeat');
      const chunk = jest.spyOn(jobService, 'storeChunk');
      const done = jest.spyOn(jobService, 'completeJob');
      const fail = jest.spyOn(jobService, 'failJob');

      req.params = { jobId: id };
      req.body = { nodeId: 'node123', lockToken };
      await jobController.heartbeat(req, res);
      expect(beat).toHaveBeenCalledWith(id, 'node123', lockToken);

      req.body = { nodeId: 'node123', lockToken, chunkIndex: 0, content: 'x' };
      await jobController.receiveChunk(req, res);
      expect(chunk).toHaveBeenCalledWith(id, 'node123', expect.any(Object), lockToken);

      req.body = { nodeId: 'node123', lockToken };
      await jobController.completeJob(req, res);
      expect(done).toHaveBeenCalledWith(id, 'node123', lockToken);

      // failJob takes the reason before the token — an easy argument to shift.
      const other = await jobService.createJob({ prompt: 'p2', userId: 'user123' });
      const [assigned] = await jobService.assignJobsToNode('node123', 1);
      req.params = { jobId: assigned.id };
      req.body = { nodeId: 'node123', lockToken: assigned.lockToken, error: 'boom' };
      await jobController.failJob(req, res);
      expect(fail).toHaveBeenCalledWith(assigned.id, 'node123', 'boom', assigned.lockToken);
      expect(other.id).toBeDefined();
    });

    it('coerces a missing, junk or non-positive maxJobs to 1', () => {
      const { clampMaxJobs, MAX_JOBS_PER_POLL } = JobController;
      expect(clampMaxJobs(undefined)).toBe(1);
      expect(clampMaxJobs(null)).toBe(1);
      expect(clampMaxJobs('abc')).toBe(1);
      expect(clampMaxJobs(0)).toBe(1);
      expect(clampMaxJobs(-9)).toBe(1);
      expect(clampMaxJobs(Infinity)).toBe(1);
      expect(clampMaxJobs(3)).toBe(3);
      expect(clampMaxJobs(2.9)).toBe(2);
      expect(clampMaxJobs(MAX_JOBS_PER_POLL + 1)).toBe(MAX_JOBS_PER_POLL);
    });

    it('should assign jobs to valid node', async () => {
      nodeService.getNode.mockResolvedValue({
        nodeId: 'node123',
        publicKey: 'test-public-key',
        status: 'online'
      });

      req.body = {
        nodeId: 'node123',
        signature: 'test-signature',
        timestamp: Date.now(),
        maxJobs: 2
      };

      await jobController.pollJobs(req, res);

      expect(nodeService.getNode).toHaveBeenCalledWith('node123');
      expect(res.json).toHaveBeenCalledWith({
        success: true,
        jobs: expect.arrayContaining([
          expect.objectContaining({
            prompt: expect.any(String),
            model: 'Gemma-4-E4B-it-Q4_K_M'
          })
        ])
      });
    });

    it('should default to one job when maxJobs is omitted', async () => {
      nodeService.getNode.mockResolvedValue({
        nodeId: 'node123',
        publicKey: 'test-public-key',
        status: 'online'
      });

      req.body = { nodeId: 'node123' }; // no maxJobs

      await jobController.pollJobs(req, res);

      expect(res.json).toHaveBeenCalledWith({
        success: true,
        jobs: expect.any(Array)
      });
    });

    it('should reject polling from unknown node', async () => {
      nodeService.getNode.mockResolvedValue(null);

      req.body = {
        nodeId: 'unknown-node',
        signature: 'test-signature',
        timestamp: Date.now(),
        maxJobs: 1
      };

      await jobController.pollJobs(req, res);

      expect(res.status).toHaveBeenCalledWith(404);
      expect(res.json).toHaveBeenCalledWith({ error: 'Node not found' });
    });
  });

  describe('heartbeat', () => {
    let job;

    beforeEach(async () => {
      job = await jobService.createJob({ prompt: 'Test', userId: 'user123' });
      await jobService.assignJobsToNode('node123', 1);
    });

    it('should handle heartbeat from valid node', async () => {
      nodeService.getNode.mockResolvedValue({
        nodeId: 'node123',
        publicKey: 'test-public-key'
      });

      req.params = { jobId: job.id };
      req.body = {
        nodeId: 'node123',
        signature: 'test-signature',
        timestamp: Date.now(),
        status: 'running'
      };

      await jobController.heartbeat(req, res);

      expect(res.json).toHaveBeenCalledWith({
        success: true,
        timestamp: expect.any(Number)
      });
    });

    it('should reject heartbeat from unknown node', async () => {
      nodeService.getNode.mockResolvedValue(null);

      req.params = { jobId: job.id };
      req.body = {
        nodeId: 'unknown-node',
        signature: 'test-signature',
        timestamp: Date.now()
      };

      await jobController.heartbeat(req, res);

      expect(res.status).toHaveBeenCalledWith(404);
      expect(res.json).toHaveBeenCalledWith({ error: 'Node not found' });
    });
  });

  describe('receiveChunk', () => {
    let job;

    beforeEach(async () => {
      job = await jobService.createJob({ prompt: 'Test', userId: 'user123' });
      await jobService.assignJobsToNode('node123', 1);
    });

    it('should receive and store chunk', async () => {
      nodeService.getNode.mockResolvedValue({
        nodeId: 'node123',
        publicKey: 'test-public-key'
      });

      req.params = { jobId: job.id };
      req.body = {
        nodeId: 'node123',
        signature: 'test-signature',
        timestamp: Date.now(),
        chunkIndex: 0,
        content: 'Test chunk',
        metrics: { tokensPerSecond: 10 }
      };

      await jobController.receiveChunk(req, res);

      expect(res.json).toHaveBeenCalledWith({
        success: true,
        chunkIndex: 0
      });
    });

    it('should reject a chunk signed with a key the node did not register', async () => {
      // The signature is valid for the presented key, but that key is not the
      // one this nodeId registered — i.e. someone signing as another node.
      nodeService.getNode.mockResolvedValue({
        nodeId: 'wrong-node',
        publicKey: 'some-other-nodes-key'
      });

      req.params = { jobId: job.id };
      req.body = {
        nodeId: 'wrong-node',
        chunkIndex: 0,
        content: 'Test chunk'
      };

      await jobController.receiveChunk(req, res);

      expect(res.status).toHaveBeenCalledWith(401);
      expect(res.json).toHaveBeenCalledWith({ error: 'Public key mismatch' });
    });

    it('should reject chunk from wrong node', async () => {
      // A properly authenticated node that simply does not hold the job's lock.
      nodeService.getNode.mockResolvedValue({
        nodeId: 'wrong-node',
        publicKey: 'test-public-key'
      });

      req.params = { jobId: job.id };
      req.body = {
        nodeId: 'wrong-node',
        chunkIndex: 0,
        content: 'Test chunk'
      };

      await jobController.receiveChunk(req, res);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({
        error: expect.stringContaining('Node does not hold lock')
      });
    });

    it('should handle chunk with non-existent node', async () => {
      nodeService.getNode.mockResolvedValue(null);

      req.params = { jobId: job.id };
      req.body = {
        nodeId: 'non-existent-node',
        chunkIndex: 0,
        content: 'Test chunk'
      };

      await jobController.receiveChunk(req, res);

      expect(res.status).toHaveBeenCalledWith(404);
      expect(res.json).toHaveBeenCalledWith({ error: 'Node not found' });
    });
  });

  describe('completeJob', () => {
    let job;

    beforeEach(async () => {
      job = await jobService.createJob({ prompt: 'Test', userId: 'user123' });
      await jobService.assignJobsToNode('node123', 1);
      await jobService.storeChunk(job.id, 'node123', {
        chunkIndex: 0,
        content: 'Complete result'
      });
    });

    it('should complete job successfully', async () => {
      nodeService.getNode.mockResolvedValue({
        nodeId: 'node123',
        publicKey: 'test-public-key'
      });

      req.params = { jobId: job.id };
      req.body = {
        nodeId: 'node123',
        signature: 'test-signature',
        timestamp: Date.now()
      };

      await jobController.completeJob(req, res);

      expect(res.json).toHaveBeenCalledWith({
        success: true,
        job: expect.objectContaining({
          status: 'completed',
          result: 'Complete result'
        })
      });
    });

    it('should handle complete with non-existent node', async () => {
      nodeService.getNode.mockResolvedValue(null);

      req.params = { jobId: job.id };
      req.body = {
        nodeId: 'non-existent-node',
        signature: 'test-signature',
        timestamp: Date.now()
      };

      await jobController.completeJob(req, res);

      expect(res.status).toHaveBeenCalledWith(404);
      expect(res.json).toHaveBeenCalledWith({ error: 'Node not found' });
    });
  });

  describe('failJob', () => {
    let job;

    beforeEach(async () => {
      job = await jobService.createJob({ prompt: 'Test', userId: 'user123' });
      await jobService.assignJobsToNode('node123', 1);
    });

    it('should mark job as failed', async () => {
      nodeService.getNode.mockResolvedValue({
        nodeId: 'node123',
        publicKey: 'test-public-key'
      });

      req.params = { jobId: job.id };
      req.body = {
        nodeId: 'node123',
        signature: 'test-signature',
        timestamp: Date.now(),
        error: 'Out of memory'
      };

      await jobController.failJob(req, res);

      expect(res.json).toHaveBeenCalledWith({
        success: true,
        job: expect.objectContaining({
          status: 'failed',
          failureReason: 'Out of memory'
        })
      });
    });

    it('should handle fail with non-existent node', async () => {
      nodeService.getNode.mockResolvedValue(null);

      req.params = { jobId: job.id };
      req.body = {
        nodeId: 'non-existent-node',
        error: 'Test error'
      };

      await jobController.failJob(req, res);

      expect(res.status).toHaveBeenCalledWith(404);
      expect(res.json).toHaveBeenCalledWith({ error: 'Node not found' });
    });
  });

  describe('getJob', () => {
    it('should return job result', async () => {
      const job = await jobService.createJob({ prompt: 'Test', userId: 'user123' });
      
      req.params = { jobId: job.id };

      await jobController.getJob(req, res);

      expect(res.json).toHaveBeenCalledWith({
        success: true,
        jobId: job.id,
        status: 'pending',
        createdAt: expect.any(Number)
      });
    });

    it('should handle non-existent job', async () => {
      req.params = { jobId: 'non-existent' };

      await jobController.getJob(req, res);

      expect(res.status).toHaveBeenCalledWith(404);
      expect(res.json).toHaveBeenCalledWith({
        error: expect.stringContaining('not found')
      });
    });

    it("should 404 (not 403) on another user's job, revealing nothing", async () => {
      const theirs = await jobService.createJob({ prompt: 'their secret', userId: 'another-user' });
      req.params = { jobId: theirs.id };

      await jobController.getJob(req, res);

      expect(res.status).toHaveBeenCalledWith(404);
      expect(JSON.stringify(res.json.mock.calls)).not.toContain('their secret');
    });

    it('should 404 when reading the result throws', async () => {
      const job = await jobService.createJob({ prompt: 'Test', userId: 'user123' });
      jobService.getJobResult = jest.fn().mockRejectedValue(new Error('db down'));
      req.params = { jobId: job.id };

      await jobController.getJob(req, res);

      expect(console.error).toHaveBeenCalledWith('Error getting job:', expect.any(Error));
      expect(res.status).toHaveBeenCalledWith(404);
      expect(res.json).toHaveBeenCalledWith({ error: 'db down' });
    });
  });

  describe('getStats', () => {
    it('should return queue statistics', async () => {
      // Create jobs in various states
      await jobService.createJob({ prompt: 'Test 1', userId: 'user123' });
      await jobService.createJob({ prompt: 'Test 2', userId: 'user123' });

      await jobController.getStats(req, res);

      expect(res.json).toHaveBeenCalledWith({
        success: true,
        stats: expect.objectContaining({
          pending: expect.any(Number),
          assigned: expect.any(Number),
          running: expect.any(Number),
          completed: expect.any(Number),
          failed: expect.any(Number)
        })
      });
    });
  });

  describe('checkTimeouts', () => {
    it('should check and return timed out jobs', async () => {
      await jobController.checkTimeouts(req, res);

      expect(res.json).toHaveBeenCalledWith({
        success: true,
        timeoutJobs: expect.any(Array)
      });
    });
  });

  describe('cleanupJobs', () => {
    it('should cleanup old jobs', async () => {
      req.body = { maxAge: 86400000 }; // 24 hours

      await jobController.cleanupJobs(req, res);

      expect(res.json).toHaveBeenCalledWith({
        success: true,
        cleaned: expect.any(Number)
      });
    });
  });

  // Additional error case tests for complete coverage
  describe('Error handling', () => {
    it('should handle submitJob error', async () => {
      req.body = { prompt: 'Test' };
      jobService.createJob = jest.fn().mockRejectedValue(new Error('Database error'));

      await jobController.submitJob(req, res);

      expect(console.error).toHaveBeenCalledWith('Error submitting job:', expect.any(Error));
      expect(res.status).toHaveBeenCalledWith(500);
      expect(res.json).toHaveBeenCalledWith({ error: 'Failed to submit job' });
    });

    it('should handle pollJobs error when getNode throws', async () => {
      req.body = { nodeId: 'node123', maxJobs: 1 };
      nodeService.getNode = jest.fn().mockRejectedValue(new Error('Database error'));

      await jobController.pollJobs(req, res);

      expect(console.error).toHaveBeenCalledWith('Error polling jobs:', expect.any(Error));
      expect(res.status).toHaveBeenCalledWith(500);
      expect(res.json).toHaveBeenCalledWith({ error: 'Failed to poll jobs' });
    });

    it('should handle heartbeat error when getNode throws', async () => {
      req.params = { jobId: 'job123' };
      req.body = { nodeId: 'node123', status: 'running', activeJobs: 1 };
      nodeService.getNode = jest.fn().mockRejectedValue(new Error('Database error'));

      await jobController.heartbeat(req, res);

      expect(console.error).toHaveBeenCalledWith('Error handling heartbeat:', expect.any(Error));
      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({ error: 'Database error' });
    });

    it('should handle completeJob error when jobService throws', async () => {
      req.params = { jobId: 'job123' };
      req.body = { nodeId: 'node123', finalOutput: 'result' };
      nodeService.getNode = jest.fn().mockResolvedValue({ id: 'node123', publicKey: 'test-public-key' });
      jobService.completeJob = jest.fn().mockRejectedValue(new Error('Database error'));

      await jobController.completeJob(req, res);

      expect(console.error).toHaveBeenCalledWith('Error completing job:', expect.any(Error));
      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({ error: 'Database error' });
    });

    it('should handle failJob error when jobService throws', async () => {
      req.params = { jobId: 'job123' };
      req.body = { nodeId: 'node123', error: 'Job failed' };
      nodeService.getNode = jest.fn().mockResolvedValue({ id: 'node123', publicKey: 'test-public-key' });
      jobService.failJob = jest.fn().mockRejectedValue(new Error('Database error'));

      await jobController.failJob(req, res);

      expect(console.error).toHaveBeenCalledWith('Error failing job:', expect.any(Error));
      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({ error: 'Database error' });
    });

    it('should handle getStats error', async () => {
      jobService.getQueueStats = jest.fn().mockRejectedValue(new Error('Database error'));

      await jobController.getStats(req, res);

      expect(console.error).toHaveBeenCalledWith('Error getting stats:', expect.any(Error));
      expect(res.status).toHaveBeenCalledWith(500);
      expect(res.json).toHaveBeenCalledWith({ error: 'Failed to get queue statistics' });
    });

    it('should handle cleanupJobs error', async () => {
      req.body = { maxAge: 86400000 };
      jobService.cleanupOldJobs = jest.fn().mockRejectedValue(new Error('Database error'));

      await jobController.cleanupJobs(req, res);

      expect(console.error).toHaveBeenCalledWith('Error cleaning up jobs:', expect.any(Error));
      expect(res.status).toHaveBeenCalledWith(500);
      expect(res.json).toHaveBeenCalledWith({ error: 'Failed to cleanup jobs' });
    });

    it('should handle checkTimeouts error', async () => {
      jobService.checkTimeouts = jest.fn().mockRejectedValue(new Error('Database error'));

      await jobController.checkTimeouts(req, res);

      expect(console.error).toHaveBeenCalledWith('Error checking timeouts:', expect.any(Error));
      expect(res.status).toHaveBeenCalledWith(500);
      expect(res.json).toHaveBeenCalledWith({ error: 'Failed to check timeouts' });
    });
  });
});