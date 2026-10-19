// The speed-benchmark sweeper: queues a targeted job at any online node whose
// generation speed we can't currently vouch for, so assignment has a number to
// gate on. Runs against pg-mem with the real SQL.
const { createTestDb } = require('./helpers/pgmem');
const BenchmarkService = require('../src/services/benchmarkService');
const JobService = require('../src/services/jobService');
const NodeService = require('../src/services/nodeService');

const HOUR = 60 * 60 * 1000;

async function seedNode(db, nodeId, { lastSeen, tps, samples, speedAt } = {}) {
  await db.query(
    `INSERT INTO nodes (node_id, public_key, name, status, last_seen, measured_tps, speed_samples, speed_at)
     VALUES ($1, $1, $1, 'online', $2, $3, $4, $5)`,
    [nodeId, lastSeen == null ? Date.now() : lastSeen, tps == null ? null : tps, samples || 0,
      speedAt == null ? null : speedAt]
  );
}

describe('BenchmarkService', () => {
  let db, svc, jobs;

  beforeEach(async () => {
    db = await createTestDb();
    jobs = new JobService(db);
    svc = new BenchmarkService(db, { jobs });
  });

  it('benchmarks a node we have never measured, and marks it as the warm-up run', async () => {
    await seedNode(db, 'cold01');

    expect(await svc.sweep()).toEqual(['cold01']);

    const r = await db.query('SELECT target_node, max_tokens, priority, data FROM jobs', []);
    expect(r.rows).toHaveLength(1);
    const job = r.rows[0];
    expect(job.target_node).toBe('cold01');          // pinned, so it reaches that node
    expect(job.max_tokens).toBe(BenchmarkService.BENCH_MAX_TOKENS);
    expect(job.priority).toBeLessThan(0);            // never ahead of real traffic
    expect(job.data.benchmark).toBe(true);
    // First measurement of this node: its result replaces the stored speed rather
    // than blending, because a cold llama-server's first run includes model load.
    expect(job.data.benchmarkWarmup).toBe(true);
    expect(job.data.messages[0].content).toBe(BenchmarkService.BENCH_PROMPT);
  });

  it('re-benchmarks a node whose measurement went stale, without the warm-up flag', async () => {
    await seedNode(db, 'stale1', { tps: 30, samples: 5, speedAt: Date.now() - 25 * HOUR });

    expect(await svc.sweep()).toEqual(['stale1']);

    const r = await db.query('SELECT data FROM jobs', []);
    // It has produced tokens before, so this run blends in normally.
    expect(r.rows[0].data.benchmarkWarmup).toBe(false);
  });

  it('leaves a freshly measured node alone', async () => {
    await seedNode(db, 'fresh1', { tps: 40, samples: 9, speedAt: Date.now() });
    expect(await svc.sweep()).toEqual([]);
    expect((await db.query('SELECT id FROM jobs', [])).rows).toHaveLength(0);
  });

  it('re-measures a node once a day, not sooner', async () => {
    expect(BenchmarkService.BENCH_INTERVAL_MS).toBe(24 * HOUR);
    // The measurement lasts exactly as long, so a node is re-measured as its
    // speed goes stale rather than sitting unmeasured in between.
    expect(BenchmarkService.BENCH_INTERVAL_MS).toBe(NodeService.SPEED_STALE_MS);
    await seedNode(db, 'day1', { tps: 30, samples: 5, speedAt: Date.now() - 23 * HOUR });
    expect(await svc.sweep()).toEqual([]);
  });

  it('ignores a node that has gone offline', async () => {
    await seedNode(db, 'gone01', { lastSeen: Date.now() - 60 * 60 * 1000 });
    expect(await svc.sweep()).toEqual([]);
  });

  it('does not queue a second benchmark while one is still in flight', async () => {
    await seedNode(db, 'cold01');

    expect(await svc.sweep()).toEqual(['cold01']);
    // The sweeper runs far more often than a benchmark takes to come back; without
    // this guard every tick would pile another one onto the same node.
    expect(await svc.sweep()).toEqual([]);
    expect((await db.query('SELECT id FROM jobs', [])).rows).toHaveLength(1);
  });

  it('waits a day after a benchmark that failed', async () => {
    // The reported case: the node's GPU had dropped out from under llama-server,
    // so every benchmark failed, no speed was recorded, and each 5-minute sweep
    // sent another one, pausing the node's miner every time.
    await seedNode(db, 'lost01');
    expect(await svc.sweep()).toEqual(['lost01']);
    const [job] = await jobs.assignJobsToNode('lost01', 1);
    await jobs.failJob(job.id, 'lost01', 'llama-server HTTP 500', job.lockToken);
    expect((await db.query('SELECT status FROM jobs', [])).rows[0].status).toBe('failed');

    expect(await svc.sweep()).toEqual([]);
    expect((await db.query('SELECT id FROM jobs', [])).rows).toHaveLength(1);

    // A day on, it gets one more try.
    await db.query('UPDATE jobs SET created_at = $1', [Date.now() - BenchmarkService.BENCH_INTERVAL_MS - 1000]);
    expect(await svc.sweep()).toEqual(['lost01']);
  });

  it('tries again soon after a benchmark nobody picked up', async () => {
    // A linked app pings before its model is up, so the first benchmark often
    // expires unclaimed. It never ran, so it doesn't use up the day.
    await seedNode(db, 'cold01');
    await svc.sweep();
    await db.query('UPDATE jobs SET created_at = $1', [Date.now() - 10 * 60 * 1000]);
    await jobs.expireStalePending();
    expect((await db.query('SELECT status FROM jobs', [])).rows[0].status).toBe('failed');
    expect(await svc.sweep()).toEqual(['cold01']);
  });

  it('still waits a day after one that was picked up, timed out and expired', async () => {
    // It ran (and paused the miner) before the node went quiet, so it counts.
    await seedNode(db, 'hung01');
    await svc.sweep();
    await jobs.assignJobsToNode('hung01', 1);
    await db.query('UPDATE jobs SET lock_expires_at = $1', [Date.now() - 1000]);
    await jobs.checkTimeouts();
    await db.query('UPDATE jobs SET created_at = $1', [Date.now() - 10 * 60 * 1000]);
    await jobs.expireStalePending();
    expect((await db.query('SELECT status FROM jobs', [])).rows[0].status).toBe('failed');
    expect(await svc.sweep()).toEqual([]);
  });

  it('does not count a request the caller pinned to the node as a benchmark', async () => {
    await seedNode(db, 'pin01');
    // X-LLMJob-Node pins a real request to one node; that is not a measurement.
    await jobs.createJob({ prompt: 'hi', targetNode: 'pin01', userId: 'u' });
    expect(await svc.sweep()).toEqual(['pin01']);
  });

  it('leaves a gated node for the daily re-measure like any other', async () => {
    // It used to be re-measured every 30 minutes. Each one paused its miner, and
    // the network carries too little traffic to be worth that.
    await seedNode(db, 'gated1', { tps: 4, samples: 3, speedAt: Date.now() - 31 * 60 * 1000 });
    expect(await svc.sweep()).toEqual([]);
  });

  it('does not re-measure a healthy node just because it is idle', async () => {
    // Above the gate it still serves real traffic, so it measures itself.
    await seedNode(db, 'ok1', { tps: 40, samples: 9, speedAt: Date.now() - 31 * 60 * 1000 });
    expect(await svc.sweep()).toEqual([]);
  });

  it('caps how many it queues per sweep', async () => {
    for (const id of ['n1', 'n2', 'n3', 'n4']) await seedNode(db, id);
    const capped = new BenchmarkService(db, { jobs, maxPerSweep: 2 });
    // A fleet that all goes stale at once (a redeploy) must not bury real traffic
    // under a wave of benchmarks.
    expect(await capped.sweep()).toHaveLength(2);
  });

  it('builds its own services when none are injected', () => {
    const bare = new BenchmarkService(db);
    expect(bare.jobs).toBeInstanceOf(JobService);
    expect(bare.nodes).toBeInstanceOf(NodeService);
    expect(bare.now()).toBeGreaterThan(0);
    expect(bare.maxPerSweep).toBeGreaterThan(0);
  });
});
