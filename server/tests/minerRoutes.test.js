const request = require('supertest');
const express = require('express');
const routes = require('../src/routes');
const { createTestDb } = require('./helpers/pgmem');

const ADDR = 'prl1p' + 'a'.repeat(30);

function makeApp(db) {
  const app = express();
  app.use(express.json());
  app.locals.db = db;
  app.use('/api', routes);
  return app;
}

// A db stub whose every query rejects, to exercise the controllers' 500 paths.
const brokenDb = { query: () => Promise.reject(new Error('db down')) };

describe('Miner API', () => {
  let db;
  let app;
  beforeEach(async () => {
    db = await createTestDb();
    app = makeApp(db);
  });
  afterEach(async () => {
    if (db.end) await db.end();
  });

  test('POST /api/miners/ping records a miner', async () => {
    const res = await request(app).post('/api/miners/ping')
      .send({ address: ADDR, worker: 'rig01', gpu: 'RTX 4090', region: 'us1', hashrate: 100, accepted: 5 });
    expect(res.status).toBe(200);
    expect(res.body.success).toBe(true);
    expect(res.body.id).toMatch(/^[0-9a-f]{12}$/);
  });

  test('POST /api/miners/ping rejects a bad address with 400', async () => {
    const res = await request(app).post('/api/miners/ping').send({ address: 'nope' });
    expect(res.status).toBe(400);
    expect(res.body.error).toMatch(/address/i);
  });

  test('GET /api/miners returns online miners, one row per worker/GPU', async () => {
    await request(app).post('/api/miners/ping').send({ address: ADDR, worker: 'rig01', gpu: 'RTX 4090', hashrate: 100 });
    const res = await request(app).get('/api/miners');
    expect(res.status).toBe(200);
    expect(res.body.totalOnline).toBe(1);
    expect(res.body.totalWorkers).toBe(1);
    expect(res.body.miners[0]).toMatchObject({ addr: ADDR, worker: 'rig01', gpu: 'RTX 4090', hash: 100 });
  });

  test('POST returns 500 when the db fails', async () => {
    const res = await request(makeApp(brokenDb)).post('/api/miners/ping').send({ address: ADDR });
    expect(res.status).toBe(500);
  });

  test('GET returns 500 when the db fails', async () => {
    const res = await request(makeApp(brokenDb)).get('/api/miners');
    expect(res.status).toBe(500);
  });
});
