const crypto = require('crypto');
const ContainerService = require('../src/services/containerService');
const { createTestDb } = require('./helpers/pgmem');

const DIGEST = 'sha256:' + 'a'.repeat(64);
const IMAGE = 'ghcr.io/super3/llmjob-bench@' + DIGEST;
const NODE = 'abcdef0123456789';
const NODE2 = '0123456789abcdef';

// Two cards: a 4090 and a 3060, both free.
const CARDS = [
  { index: 0, gpu: 'NVIDIA GeForce RTX 4090', vramMb: 24564, vendor: 'nvidia', free: true },
  { index: 1, gpu: 'NVIDIA GeForce RTX 3060', vramMb: 12288, vendor: 'nvidia', free: true },
];
const RIG = { worker: 'rig01', platform: 'linux', docker: true, driver: '580.65', cards: CARDS };

describe('ContainerService helpers', () => {
  const prev = process.env.CONTAINER_REGISTRY;
  afterEach(() => {
    if (prev === undefined) delete process.env.CONTAINER_REGISTRY;
    else process.env.CONTAINER_REGISTRY = prev;
  });

  test('imageProblem takes only our registry, pinned by digest', () => {
    expect(ContainerService.imageProblem(IMAGE)).toBeNull();
    expect(ContainerService.imageProblem('')).toMatch(/required/);
    expect(ContainerService.imageProblem(undefined)).toMatch(/required/);
    expect(ContainerService.imageProblem('docker.io/library/ubuntu@' + DIGEST)).toMatch(/must come from/);
    expect(ContainerService.imageProblem('ghcr.io/super3/llmjob-bench:latest')).toMatch(/digest/);
    expect(ContainerService.imageProblem('ghcr.io/super3/x@sha256:abc')).toMatch(/digest/);
    process.env.CONTAINER_REGISTRY = 'registry.example/';
    expect(ContainerService.imageProblem('registry.example/x@' + DIGEST)).toBeNull();
  });

  test('cleanEnv checks keys and values', () => {
    expect(ContainerService.cleanEnv(null)).toEqual({});
    expect(ContainerService.cleanEnv({ A: 1, B_2: null })).toEqual({ A: '1', B_2: '' });
    expect(ContainerService.cleanEnv([])).toMatch(/object/);
    expect(ContainerService.cleanEnv('x')).toMatch(/object/);
    expect(ContainerService.cleanEnv({ 'bad key': 'x' })).toMatch(/bad env key/);
    expect(ContainerService.cleanEnv({ A: 'x'.repeat(8193) })).toMatch(/too long/);
    const many = {};
    for (let i = 0; i < 65; i++) many['K' + i] = 'v';
    expect(ContainerService.cleanEnv(many)).toMatch(/more than/);
  });

  test('cleanTarget takes an exact rig or a filter', () => {
    expect(ContainerService.cleanTarget(undefined)).toEqual({});
    expect(ContainerService.cleanTarget([])).toMatch(/object/);
    expect(ContainerService.cleanTarget({ nodeId: NODE, card: '1' })).toEqual({ nodeId: NODE, card: 1 });
    expect(ContainerService.cleanTarget({ nodeId: 'zz', card: 0 })).toMatch(/node id/);
    expect(ContainerService.cleanTarget({ nodeId: NODE, card: -1 })).toMatch(/card index/);
    expect(ContainerService.cleanTarget({ nodeId: NODE })).toMatch(/card index/);
    expect(ContainerService.cleanTarget({ gpu: ' 4090 ', minVramMb: 20000, minDriver: '570', vendor: 'NVIDIA' }))
      .toEqual({ gpu: '4090', minVramMb: 20000, minDriver: 570, vendor: 'nvidia' });
    expect(ContainerService.cleanTarget({ gpu: '  ' })).toEqual({});
    expect(ContainerService.cleanTarget({ minVramMb: 'x' })).toMatch(/minVramMb/);
    expect(ContainerService.cleanTarget({ minVramMb: -1 })).toMatch(/minVramMb/);
    expect(ContainerService.cleanTarget({ minDriver: 'x' })).toMatch(/minDriver/);
    expect(ContainerService.cleanTarget({ vendor: 'intel' })).toMatch(/vendor/);
  });

  test('cleanCards drops junk', () => {
    expect(ContainerService.cleanCards('x')).toEqual([]);
    expect(ContainerService.cleanCards([null, 5, { index: 'x' }, { index: 40 }, { index: 2 }])).toEqual([
      { index: 2, gpu: null, vramMb: null, vendor: null, free: false },
    ]);
  });

  test('cardFits applies each filter', () => {
    const card = CARDS[0];
    const rig = { nodeId: NODE, driver: '580.65' };
    expect(ContainerService.cardFits(undefined, card, rig)).toBe(true);
    expect(ContainerService.cardFits({ nodeId: NODE, card: 0 }, card, rig)).toBe(true);
    expect(ContainerService.cardFits({ nodeId: NODE, card: 1 }, card, rig)).toBe(false);
    expect(ContainerService.cardFits({ nodeId: NODE2, card: 0 }, card, rig)).toBe(false);
    expect(ContainerService.cardFits({ gpu: 'rtx 4090' }, card, rig)).toBe(true);
    expect(ContainerService.cardFits({ gpu: '5090' }, card, rig)).toBe(false);
    expect(ContainerService.cardFits({ gpu: '5090' }, { index: 0 }, rig)).toBe(false);
    expect(ContainerService.cardFits({ minVramMb: 30000 }, card, rig)).toBe(false);
    expect(ContainerService.cardFits({ minDriver: 590 }, card, rig)).toBe(false);
    expect(ContainerService.cardFits({ minDriver: 570 }, card, rig)).toBe(true);
    expect(ContainerService.cardFits({ vendor: 'amd' }, card, rig)).toBe(false);
    expect(ContainerService.cardFits({ vendor: 'amd' }, { index: 0 }, rig)).toBe(false);
  });

  test('paidMinutes counts from the start, capped at the duration', () => {
    expect(ContainerService.paidMinutes(null, 1000, 60)).toBe(0);
    expect(ContainerService.paidMinutes(0, 90000, 3600)).toBe(1.5);
    expect(ContainerService.paidMinutes(0, 10 * 3600 * 1000, 3600)).toBe(60);
    expect(ContainerService.paidMinutes(5000, 1000, 3600)).toBe(0);
  });

  test('nodeIdMatchesKey ties the id to the key', () => {
    const key = 'PUBKEY';
    const full = crypto.createHash('sha256').update(key).digest('hex');
    expect(ContainerService.nodeIdMatchesKey(full.slice(0, 16), key)).toBe(true);
    expect(ContainerService.nodeIdMatchesKey(full.slice(0, 6), key)).toBe(true);
    expect(ContainerService.nodeIdMatchesKey(NODE, key)).toBe(false);
    expect(ContainerService.nodeIdMatchesKey(full.slice(0, 10), key)).toBe(false);
    expect(ContainerService.nodeIdMatchesKey(5, key)).toBe(false);
    expect(ContainerService.nodeIdMatchesKey(full.slice(0, 16), '')).toBe(false);
  });
});

describe('ContainerService with a database', () => {
  let db;
  let clock;
  let svc;
  beforeEach(async () => {
    db = await createTestDb();
    clock = 1_000_000;
    svc = new ContainerService(db, { now: () => clock });
  });

  const create = (over = {}) => svc.createJob(Object.assign({ image: IMAGE, durationSec: 3600 }, over), 'admin');

  test('uses Date.now when no clock is given', () => {
    expect(new ContainerService(db).now).toBe(Date.now);
  });

  test('createJob validates every field', async () => {
    expect((await create({ image: 'nope' })).status).toBe(400);
    expect((await create({ env: [] })).error).toMatch(/env/);
    expect((await create({ target: 'x' })).error).toMatch(/target/);
    expect((await create({ durationSec: 10 })).error).toMatch(/durationSec/);
    expect((await create({ durationSec: 1.5 })).error).toMatch(/durationSec/);
    expect((await create({ replicas: 0 })).error).toMatch(/replicas/);
    expect((await create({ replicas: 2, target: { nodeId: NODE, card: 0 } })).error).toMatch(/replicas: 1/);
    expect((await create({ payRateUsdHour: -1 })).error).toMatch(/payRate/);
    expect((await create({ payRateUsdHour: 'x' })).error).toMatch(/payRate/);
  });

  test('createJob stores the job and getJob/listJobs read it back', async () => {
    const { job } = await create({ name: 'bench', env: { A: 'b' }, replicas: 2, payRateUsdHour: 0.3 });
    expect(job).toMatchObject({
      name: 'bench', image: IMAGE, env: { A: 'b' }, target: {}, durationSec: 3600, replicas: 2,
      payRateUsdHour: 0.3, status: 'pending', createdBy: 'admin', runs: [], minutes: 0,
    });
    expect(job.id).toMatch(/^cj_[0-9a-f]{16}$/);
    // The name defaults to the id.
    const { job: unnamed } = await svc.createJob({ image: IMAGE, durationSec: 60 });
    expect(unnamed.name).toBe(unnamed.id);
    expect(unnamed.createdBy).toBeNull();
    expect((await svc.getJob(job.id)).job.id).toBe(job.id);
    expect((await svc.getJob('cj_none')).status).toBe(404);
    expect((await svc.listJobs()).jobs).toHaveLength(2);
    expect((await svc.listJobs('x')).jobs).toHaveLength(2);
    expect((await svc.listJobs(1)).jobs).toHaveLength(1);
  });

  test('a rig with no docker, on Windows, or with no free card gets nothing', async () => {
    await create();
    expect((await svc.poll(NODE, Object.assign({}, RIG, { docker: false }))).job).toBeNull();
    expect((await svc.poll(NODE, Object.assign({}, RIG, { platform: 'win32' }))).job).toBeNull();
    expect((await svc.poll(NODE, Object.assign({}, RIG, { cards: [{ index: 0 }] }))).job).toBeNull();
    // A bare poll is still recorded.
    const out = await svc.poll(NODE);
    expect(out).toEqual({ job: null, pollMs: ContainerService.POLL_MS });
    const rigs = (await svc.listRigs()).rigs;
    expect(rigs).toEqual([{
      nodeId: NODE, worker: null, platform: null, docker: false, driver: null, cards: [], lastPoll: clock,
    }]);
    // Rigs not heard from in ten minutes drop off the list.
    clock += 11 * 60 * 1000;
    expect((await svc.listRigs()).rigs).toEqual([]);
  });

  test('the full life of a run: claim, pull, run, logs, end, minutes', async () => {
    const { job } = await create({ env: { MODE: 'bench' }, payRateUsdHour: 0.6, target: { gpu: '4090' } });
    const { job: claimed } = await svc.poll(NODE, RIG);
    expect(claimed).toMatchObject({
      jobId: job.id, image: IMAGE, env: { MODE: 'bench' }, card: 0, durationSec: 3600, payRateUsdHour: 0.6,
    });
    expect((await svc.getJob(job.id)).job.status).toBe('running');
    // The job is full: the same rig gets nothing more.
    expect((await svc.poll(NODE, RIG)).job).toBeNull();

    clock += 5000;
    expect(await svc.heartbeat(NODE, claimed.runId, { state: 'pulling', logs: ['pulling'] }))
      .toEqual({ action: 'continue', endsAt: null });
    clock += 5000;
    const startedAt = clock;
    expect(await svc.heartbeat(NODE, claimed.runId, { state: 'running', logs: ['a', 'b'] }))
      .toEqual({ action: 'continue', endsAt: startedAt + 3600 * 1000 });
    clock += 30 * 60 * 1000;
    await svc.heartbeat(NODE, claimed.runId, { state: 'running' });
    let status = (await svc.getJob(job.id)).job;
    expect(status.runs[0]).toMatchObject({ status: 'running', startedAt, logLines: 3, minutes: 30 });

    clock += 60 * 1000;
    expect(await svc.endRun(NODE, claimed.runId, { reason: 'exited', exitCode: 0, logs: ['done'] }))
      .toEqual({ success: true, status: 'exited' });
    status = (await svc.getJob(job.id)).job;
    expect(status.status).toBe('done');
    expect(status.runs[0]).toMatchObject({ status: 'exited', exitCode: 0, reason: 'exited', minutes: 31 });
    expect(status.minutes).toBe(31);

    const logs = await svc.getLogs(job.id);
    expect(logs.lines.map((l) => l.line)).toEqual(['pulling', 'a', 'b', 'done']);
    expect(logs.lines[0]).toMatchObject({ runId: claimed.runId, nodeId: NODE, card: 0, seq: 0 });
    const page = await svc.getLogs(job.id, { since: logs.lines[1].id, limit: 1, runId: claimed.runId });
    expect(page.lines.map((l) => l.line)).toEqual(['b']);
    expect(page.next).toBe(logs.lines[2].id);
    const empty = await svc.getLogs(job.id, { since: 1e9 });
    expect(empty).toEqual({ lines: [], next: 1e9 });
    expect((await svc.getLogs('cj_none')).status).toBe(404);

    // The ledger: 31 minutes at $0.60 an hour.
    expect((await svc.ledger()).rows).toEqual([{ nodeId: NODE, card: 0, runs: 1, minutes: 31, hours: 0.5167, amountUsd: 0.31 }]);
    expect((await svc.ledger({ nodeId: NODE2 })).rows).toEqual([]);

    // An ended run stays ended.
    expect(await svc.endRun(NODE, claimed.runId, { reason: 'failed' })).toEqual({ success: true, status: 'exited' });
    expect(await svc.heartbeat(NODE, claimed.runId, {})).toEqual({ action: 'stop', reason: 'run is exited' });
  });

  test('a job of two runs keeps running while one is still up, and a card may have no name', async () => {
    const { job } = await create({ replicas: 2 });
    const bare = Object.assign({}, RIG, { worker: undefined, cards: [{ index: 3, free: true }] });
    const { job: a } = await svc.poll(NODE, bare);
    await svc.poll(NODE2, RIG);
    await svc.endRun(NODE, a.runId, { reason: 'exited', exitCode: 0 });
    const status = (await svc.getJob(job.id)).job;
    expect(status.status).toBe('running');
    expect(status.runs.find((r) => r.id === a.runId)).toMatchObject({ card: 3, gpu: null, worker: null });
  });

  test('a run belongs to its rig only', async () => {
    await create();
    const { job: run } = await svc.poll(NODE, RIG);
    expect((await svc.heartbeat(NODE2, run.runId, {})).status).toBe(404);
    expect((await svc.endRun(NODE2, run.runId, {})).status).toBe(404);
    expect((await svc.heartbeat(NODE, 'cr_none', {})).status).toBe(404);
  });

  test('an unknown end reason reads as failed, with the detail kept', async () => {
    const { job } = await create();
    const { job: run } = await svc.poll(NODE, RIG);
    await svc.endRun(NODE, run.runId, { reason: 'weird', detail: 'docker pull failed', exitCode: 1.5 });
    const r = (await svc.getJob(job.id)).job.runs[0];
    expect(r).toMatchObject({ status: 'failed', reason: 'docker pull failed', exitCode: null, minutes: 0 });
    // The slot opened again, but not for the rig that failed it.
    expect((await svc.getJob(job.id)).job.status).toBe('pending');
    expect((await svc.poll(NODE, RIG)).job).toBeNull();
    expect((await svc.poll(NODE2, RIG)).job).not.toBeNull();
  });

  test('an exact target goes to its one rig and card, and may try again', async () => {
    const { job } = await create({ target: { nodeId: NODE, card: 1 } });
    expect((await svc.poll(NODE2, RIG)).job).toBeNull();
    const { job: run } = await svc.poll(NODE, RIG);
    expect(run.card).toBe(1);
    await svc.endRun(NODE, run.runId, { reason: 'taken_back' });
    expect((await svc.getJob(job.id)).job.status).toBe('pending');
    // The owner took the card back; the same rig may take it again later.
    expect((await svc.poll(NODE, RIG)).job.card).toBe(1);
  });

  test('a card already running a job is not handed another', async () => {
    await create({ target: { nodeId: NODE, card: 0 } });
    await svc.poll(NODE, RIG);
    await create({ target: { nodeId: NODE, card: 0 } });
    expect((await svc.poll(NODE, RIG)).job).toBeNull();
    // The other card is still free for a filter job.
    await create();
    expect((await svc.poll(NODE, RIG)).job.card).toBe(1);
  });

  test('deleting a job: stopped at once with nothing running, else stopping', async () => {
    const { job: idle } = await create();
    expect((await svc.deleteJob(idle.id)).job.status).toBe('stopped');
    // Deleting twice is fine.
    expect((await svc.deleteJob(idle.id)).job.status).toBe('stopped');
    expect((await svc.deleteJob('cj_none')).status).toBe(404);

    const { job } = await create();
    const { job: run } = await svc.poll(NODE, RIG);
    expect((await svc.deleteJob(job.id)).job.status).toBe('stopping');
    expect(await svc.heartbeat(NODE, run.runId, { state: 'running' })).toEqual({ action: 'stop', reason: 'job deleted' });
    await svc.endRun(NODE, run.runId, { reason: 'stopped' });
    expect((await svc.getJob(job.id)).job.status).toBe('stopped');
    // A stopped job is offered to no one.
    expect((await svc.poll(NODE2, RIG)).job).toBeNull();
  });

  test('a run past its time is told to stop', async () => {
    await create({ durationSec: 60 });
    const { job: run } = await svc.poll(NODE, RIG);
    await svc.heartbeat(NODE, run.runId, { state: 'running' });
    clock += 60 * 1000 + ContainerService.OVERRUN_GRACE_MS + 1;
    expect(await svc.heartbeat(NODE, run.runId, { state: 'running' })).toEqual({ action: 'stop', reason: 'job time is up' });
  });

  test('a quiet rig is released, paid to its last heartbeat, and the slot reopens', async () => {
    const { job } = await create({ payRateUsdHour: 1 });
    const { job: run } = await svc.poll(NODE, RIG);
    await svc.heartbeat(NODE, run.runId, { state: 'running' });
    clock += 2 * 60 * 1000;
    await svc.heartbeat(NODE, run.runId, { state: 'running' });
    clock += ContainerService.RUN_TIMEOUT_MS + 1;
    expect(await svc.releaseStale()).toEqual([run.runId]);
    const status = (await svc.getJob(job.id)).job;
    expect(status.status).toBe('pending');
    expect(status.runs[0]).toMatchObject({ status: 'lost', reason: 'no heartbeat', minutes: 2 });
    expect(await svc.releaseStale()).toEqual([]);
    // A heartbeat after the release is told to stop.
    expect((await svc.heartbeat(NODE, run.runId, {})).action).toBe('stop');
    // Another rig takes the slot (the poll also sweeps).
    expect((await svc.poll(NODE2, RIG)).job.jobId).toBe(job.id);
  });

  test('a run that stops a deleted job while others still run keeps it stopping', async () => {
    const { job } = await create({ replicas: 2 });
    const { job: a } = await svc.poll(NODE, RIG);
    await svc.poll(NODE2, RIG);
    await svc.deleteJob(job.id);
    await svc.endRun(NODE, a.runId, { reason: 'stopped' });
    expect((await svc.getJob(job.id)).job.status).toBe('stopping');
  });

  test('a claim rolls back on a database error', async () => {
    await create();
    const realConnect = db.connect.bind(db);
    db.connect = async () => {
      const client = await realConnect();
      const q = client.query.bind(client);
      client.query = (sql, ...rest) => (/INSERT INTO container_minutes/.test(sql) ? Promise.reject(new Error('boom')) : q(sql, ...rest));
      return client;
    };
    await expect(svc.poll(NODE, RIG)).rejects.toThrow('boom');
    db.connect = realConnect;
  });

  test('logs stop at the per-run cap', async () => {
    await create();
    const { job: run } = await svc.poll(NODE, RIG);
    await db.query('UPDATE container_runs SET log_lines = $2 WHERE id = $1', [run.runId, ContainerService.MAX_LOG_LINES - 1]);
    await svc.heartbeat(NODE, run.runId, { state: 'running', logs: ['x', 'y', null] });
    const r = await db.query('SELECT log_lines FROM container_runs WHERE id = $1', [run.runId]);
    expect(Number(r.rows[0].log_lines)).toBe(ContainerService.MAX_LOG_LINES);
    await svc.heartbeat(NODE, run.runId, { state: 'running', logs: 'not a list' });
  });
});
