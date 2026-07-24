// Exercises the job route wrapper functions registered by initJobRoutes.
// The auth and signature middleware are mocked to pass through so each
// route handler (and its controller call) actually executes.
jest.mock('../src/middleware/auth', () => ({
  requireAuth: (req, res, next) => {
    req.user = { id: 'user-routes' };
    next();
  },
}));
// Stands in for the real Ed25519 check: the real middleware proves only that the
// caller owns the key it presented, so the mock echoes the presented key back —
// letting a test present a key the node never registered (the spoofing case).
jest.mock('../src/middleware/signature', () => ({
  verifySignature: (req, res, next) => {
    req.verifiedNode = { nodeId: req.body && req.body.nodeId, publicKey: req.body && req.body.publicKey };
    next();
  },
}));

const request = require('supertest');
const express = require('express');
const { createTestDb } = require('./helpers/pgmem');
const NodeService = require('../src/services/nodeService');

describe('Job routes (handlers executed end-to-end)', () => {
  let app;
  let db;

  // The routes module exports a single shared router and initJobRoutes APPENDS
  // handlers to it, so re-initialising across tests would leave the first
  // registration (bound to the first test's db) serving every request — making
  // later tests silently 404 against an empty database. Load a fresh module
  // registry per test so each one gets its own router bound to its own db.
  beforeEach(async () => {
    db = await createTestDb();
    jest.resetModules();
    const router = require('../src/routes');
    router.initJobRoutes(db);
    app = express();
    app.use(express.json());
    app.use('/api', router);
  });

  afterEach(async () => {
    if (db.end) await db.end();
  });

  it('submits a job and reads it back', async () => {
    const submit = await request(app).post('/api/jobs').send({ prompt: 'hello', model: 'm' });
    expect(submit.status).toBe(201);
    const jobId = submit.body.job.id;

    const get = await request(app).get(`/api/jobs/${jobId}`);
    expect(get.status).toBe(200);
  });

  it('handles node-facing job endpoints', async () => {
    // Register a node so poll can find it
    await new NodeService(db).claimNode('route-key', 'Route Node', 'user-routes');
    const nodeId = NodeService.generateNodeFingerprint('route-key');
    const publicKey = 'route-key'; // the key this node actually registered

    const submit = await request(app).post('/api/jobs').send({ prompt: 'hi' });
    const jobId = submit.body.job.id;

    const poll = await request(app).post('/api/jobs/poll').send({ nodeId, publicKey, maxJobs: 1 });
    expect(poll.status).toBe(200);
    expect(poll.body.jobs.map((j) => j.id)).toEqual([jobId]);
    expect((await request(app).post(`/api/jobs/${jobId}/heartbeat`).send({ nodeId, publicKey })).status).toBe(200);
    expect((await request(app).post(`/api/jobs/${jobId}/chunks`).send({ nodeId, publicKey, chunkIndex: 0, content: 'x' })).status).toBe(200);
    expect((await request(app).post(`/api/jobs/${jobId}/complete`).send({ nodeId, publicKey })).status).toBe(200);
    // the job is finished, so its lock is released and /fail no longer applies
    expect((await request(app).post(`/api/jobs/${jobId}/fail`).send({ nodeId, publicKey, reason: 'r' })).status).toBe(400);
  });

  // The guarantee behind a "private" API key: its jobs run only on the owner's
  // own nodes. nodeIds are public (GET /api/nodes/public lists them), and a valid
  // signature only proves the caller owns the key it presented — so every
  // node-facing route must reject a caller presenting a key that nodeId never
  // registered. Otherwise anyone could poll as someone else's node and read
  // their private prompts.
  it('rejects a caller impersonating another node with its own keypair', async () => {
    await new NodeService(db).claimNode('victim-key', 'Victim Node', 'victim-user');
    const nodeId = NodeService.generateNodeFingerprint('victim-key');

    // A private job belonging to the victim, waiting for one of their nodes.
    const JobService = require('../src/services/jobService');
    await new JobService(db).createJob({ prompt: 'my secrets', userId: 'victim-user', visibility: 'private' });

    // The attacker knows the nodeId and signs with a keypair they generated.
    const spoof = { nodeId, publicKey: 'attacker-key' };
    const poll = await request(app).post('/api/jobs/poll').send({ ...spoof, maxJobs: 5 });
    expect(poll.status).toBe(401);
    expect(poll.body).toEqual({ error: 'Public key mismatch' });
    expect(JSON.stringify(poll.body)).not.toContain('my secrets');

    // …and the same for every other node-facing route.
    for (const path of ['heartbeat', 'chunks', 'complete', 'fail']) {
      const res = await request(app).post(`/api/jobs/job-x/${path}`).send({ ...spoof, chunkIndex: 0, content: 'x' });
      expect(res.status).toBe(401);
    }

    // The victim's own node still gets its job.
    const mine = await request(app).post('/api/jobs/poll').send({ nodeId, publicKey: 'victim-key', maxJobs: 5 });
    expect(mine.status).toBe(200);
    expect(mine.body.jobs.map((j) => j.prompt)).toEqual(['my secrets']);
  });

  it('handles admin endpoints', async () => {
    // cleanup is admin-gated; the mocked auth user is 'user-routes'.
    const prev = process.env.ADMIN_USER_IDS;
    process.env.ADMIN_USER_IDS = 'user-routes';
    try {
      const cleanup = await request(app).post('/api/jobs/cleanup').send({});
      expect(cleanup.status).toBe(200);

      const timeouts = await request(app).post('/api/jobs/check-timeouts').send({});
      expect(timeouts.status).toBe(200);

      const stats = await request(app).get('/api/jobs/stats');
      expect(stats.status).toBe(200);
    } finally {
      process.env.ADMIN_USER_IDS = prev;
    }
  });

  it('rejects cleanup for a non-admin user', async () => {
    const prev = process.env.ADMIN_USER_IDS;
    process.env.ADMIN_USER_IDS = 'someone-else';
    try {
      const cleanup = await request(app).post('/api/jobs/cleanup').send({});
      expect(cleanup.status).toBe(403);
    } finally {
      process.env.ADMIN_USER_IDS = prev;
    }
  });
});
