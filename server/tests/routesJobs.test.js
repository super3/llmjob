// Exercises the job route wrapper functions registered by initJobRoutes.
// The auth and signature middleware are mocked to pass through so each
// route handler (and its controller call) actually executes.
jest.mock('../src/middleware/auth', () => ({
  requireAuth: (req, res, next) => {
    req.user = { id: 'user-routes' };
    next();
  },
}));
jest.mock('../src/middleware/signature', () => ({
  verifySignature: (req, res, next) => {
    req.verifiedNode = { nodeId: req.body && req.body.nodeId, publicKey: 'pk' };
    next();
  },
}));

const request = require('supertest');
const express = require('express');
const redis = require('redis-mock');
const router = require('../src/routes');
const { initJobRoutes } = require('../src/routes');
const nodeService = require('../src/services/nodeService');

describe('Job routes (handlers executed end-to-end)', () => {
  let app;
  let redisClient;

  beforeEach(async () => {
    redisClient = redis.createClient();
    await new Promise((r) => redisClient.flushall(r));
    initJobRoutes(redisClient);
    app = express();
    app.use(express.json());
    app.use('/api', router);
  });

  afterEach(() => redisClient.quit());

  it('submits a job and reads it back', async () => {
    const submit = await request(app).post('/api/jobs').send({ prompt: 'hello', model: 'm' });
    expect(submit.status).toBe(201);
    const jobId = submit.body.job.id;

    const get = await request(app).get(`/api/jobs/${jobId}`);
    expect(get.status).toBe(200);
  });

  it('handles node-facing job endpoints', async () => {
    // Register a node so poll can find it
    await nodeService.claimNode(redisClient, 'route-key', 'Route Node', 'user-routes');
    const nodeId = nodeService.generateNodeFingerprint('route-key');

    const submit = await request(app).post('/api/jobs').send({ prompt: 'hi' });
    const jobId = submit.body.job.id;

    await request(app).post('/api/jobs/poll').send({ nodeId, maxJobs: 1 });
    await request(app).post(`/api/jobs/${jobId}/heartbeat`).send({ nodeId });
    await request(app).post(`/api/jobs/${jobId}/chunks`).send({ nodeId, chunkIndex: 0, content: 'x' });
    await request(app).post(`/api/jobs/${jobId}/complete`).send({ nodeId });
    await request(app).post(`/api/jobs/${jobId}/fail`).send({ nodeId, reason: 'r' });
  });

  it('handles admin endpoints', async () => {
    const cleanup = await request(app).post('/api/jobs/cleanup').send({});
    expect(cleanup.status).toBe(200);

    const timeouts = await request(app).post('/api/jobs/check-timeouts').send({});
    expect(timeouts.status).toBe(200);

    const stats = await request(app).get('/api/jobs/stats');
    expect(stats.status).toBe(200);
  });
});
