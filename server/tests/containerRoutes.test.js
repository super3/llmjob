// The container-job routes end to end over HTTP: an admin creates a job with
// the admin token, a rig signed with its node key claims it, heartbeats, ends
// it, and the admin reads status, logs and the minutes ledger.
jest.mock('@clerk/clerk-sdk-node', () => ({ verifyToken: jest.fn() }));
const crypto = require('crypto');
const nacl = require('tweetnacl');
const naclUtil = require('tweetnacl-util');
const request = require('supertest');
const express = require('express');
const routes = require('../src/routes');
const { createTestDb } = require('./helpers/pgmem');

const IMAGE = 'ghcr.io/super3/llmjob-bench@sha256:' + 'b'.repeat(64);
const TOKEN = 'test-admin-token';

function makeApp(db) {
  const app = express();
  app.use(express.json());
  app.locals.db = db;
  app.use('/api', routes);
  return app;
}

function rigKey() {
  const kp = nacl.sign.keyPair();
  const publicKey = naclUtil.encodeBase64(kp.publicKey);
  const nodeId = crypto.createHash('sha256').update(publicKey).digest('hex').slice(0, 16);
  return { kp, publicKey, nodeId };
}

function signed(rig, extra, nodeId = rig.nodeId) {
  const timestamp = Date.now();
  const signature = naclUtil.encodeBase64(
    nacl.sign.detached(naclUtil.decodeUTF8(nodeId + ':' + timestamp), rig.kp.secretKey)
  );
  return Object.assign({ nodeId, publicKey: rig.publicKey, signature, timestamp }, extra);
}

const RIG = {
  worker: 'rig01', platform: 'linux', docker: true, driver: '580.65',
  cards: [{ index: 0, gpu: 'NVIDIA GeForce RTX 4090', vramMb: 24564, vendor: 'nvidia', free: true }],
};

describe('container routes', () => {
  let db;
  let app;
  let prevToken;
  const admin = (r) => r.set('Authorization', 'Bearer ' + TOKEN);

  beforeEach(async () => {
    prevToken = process.env.CONTAINERS_ADMIN_TOKEN;
    process.env.CONTAINERS_ADMIN_TOKEN = TOKEN;
    db = await createTestDb();
    app = makeApp(db);
    jest.spyOn(console, 'error').mockImplementation(() => {});
  });
  afterEach(() => {
    if (prevToken === undefined) delete process.env.CONTAINERS_ADMIN_TOKEN;
    else process.env.CONTAINERS_ADMIN_TOKEN = prevToken;
    jest.restoreAllMocks();
  });

  test('admin calls need the token', async () => {
    expect((await request(app).get('/api/containers')).status).toBe(401);
    expect((await request(app).post('/api/containers').send({})).status).toBe(401);
  });

  test('create, claim, heartbeat, end, then status, logs, ledger and rigs', async () => {
    const created = await admin(request(app).post('/api/containers'))
      .send({ name: 'bench', image: IMAGE, env: { MODE: 'x' }, durationSec: 600, payRateUsdHour: 0.5 });
    expect(created.status).toBe(201);
    const jobId = created.body.job.id;
    expect((await admin(request(app).post('/api/containers')).send({ image: 'nope' })).status).toBe(400);
    expect((await admin(request(app).post('/api/containers'))).status).toBe(400);

    const rig = rigKey();
    const poll = await request(app).post('/api/rigs/poll').send(signed(rig, RIG));
    expect(poll.status).toBe(200);
    expect(poll.body.job).toMatchObject({ jobId, image: IMAGE, env: { MODE: 'x' }, card: 0, durationSec: 600 });
    const runId = poll.body.job.runId;

    const hb = await request(app).post('/api/rigs/runs/' + runId + '/heartbeat')
      .send(signed(rig, { state: 'running', logs: ['hello'] }));
    expect(hb.status).toBe(200);
    expect(hb.body.action).toBe('continue');

    const end = await request(app).post('/api/rigs/runs/' + runId + '/end')
      .send(signed(rig, { reason: 'exited', exitCode: 0, logs: ['bye'] }));
    expect(end.body).toEqual({ success: true, status: 'exited' });

    const status = await admin(request(app).get('/api/containers/' + jobId));
    expect(status.body.job).toMatchObject({ status: 'done', runs: [{ nodeId: rig.nodeId, status: 'exited' }] });
    expect((await admin(request(app).get('/api/containers/cj_none'))).status).toBe(404);

    const logs = await admin(request(app).get('/api/containers/' + jobId + '/logs?since=0&limit=10'));
    expect(logs.body.lines.map((l) => l.line)).toEqual(['hello', 'bye']);

    const list = await admin(request(app).get('/api/containers?limit=5'));
    expect(list.body.jobs.map((j) => j.id)).toEqual([jobId]);

    const ledger = await admin(request(app).get('/api/containers/ledger?nodeId=' + rig.nodeId));
    expect(ledger.body.rows).toHaveLength(1);
    expect(ledger.body.rows[0]).toMatchObject({ nodeId: rig.nodeId, card: 0, runs: 1 });

    const rigs = await admin(request(app).get('/api/containers/rigs'));
    expect(rigs.body.rigs[0]).toMatchObject({ nodeId: rig.nodeId, worker: 'rig01', docker: true });

    const del = await admin(request(app).delete('/api/containers/' + jobId));
    expect(del.body.job.status).toBe('stopped');
  });

  test('a rig must sign with the key its node id comes from', async () => {
    const rig = rigKey();
    const other = rigKey();
    const res = await request(app).post('/api/rigs/poll').send(signed(rig, RIG, other.nodeId));
    expect(res.status).toBe(401);
    expect(res.body.error).toMatch(/does not match/);
    expect((await request(app).post('/api/rigs/runs/x/heartbeat').send(signed(rig, {}, other.nodeId))).status).toBe(401);
    expect((await request(app).post('/api/rigs/runs/x/end').send(signed(rig, {}, other.nodeId))).status).toBe(401);
    // Unsigned is refused before that.
    expect((await request(app).post('/api/rigs/poll').send(RIG)).status).toBe(400);
  });

  test('database errors are a plain 500', async () => {
    const broken = makeApp({ query: () => Promise.reject(new Error('db down')) });
    const rig = rigKey();
    const calls = [
      admin(request(broken).post('/api/containers')).send({ image: IMAGE, durationSec: 600 }),
      admin(request(broken).get('/api/containers')),
      admin(request(broken).get('/api/containers/x')),
      admin(request(broken).get('/api/containers/x/logs')),
      admin(request(broken).delete('/api/containers/x')),
      admin(request(broken).get('/api/containers/ledger')),
      admin(request(broken).get('/api/containers/rigs')),
      request(broken).post('/api/rigs/poll').send(signed(rig, RIG)),
      request(broken).post('/api/rigs/runs/x/heartbeat').send(signed(rig, {})),
      request(broken).post('/api/rigs/runs/x/end').send(signed(rig, {})),
    ];
    for (const res of await Promise.all(calls)) expect(res.status).toBe(500);
  });
});
