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
    await seedNode(db, 'stale1', { tps: 30, samples: 5, speedAt: Date.now() - 7 * HOUR });

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

  it('re-benchmarks once an in-flight one has aged out', async () => {
    await seedNode(db, 'cold01');
    await svc.sweep();
    // Push the existing benchmark past its TTL: it is never coming back, so the
    // node is unmeasured again and deserves another go.
    await db.query('UPDATE jobs SET created_at = $1', [Date.now() - 2 * BenchmarkService.BENCH_TTL_MS]);
    expect(await svc.sweep()).toEqual(['cold01']);
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
