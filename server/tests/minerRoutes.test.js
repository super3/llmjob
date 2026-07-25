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

  // Regression: the controller whitelists the ping body field by field, and
  // llmModel was missing from that list — so the served-model column the client
  // reports was dropped before it ever reached the service, and the board read
  // null for every host. Assert the field survives the whole round trip
  // (request body → upsert → board payload), on the host row and its card.
  test('POST /api/miners/ping stores the served LLM and GET returns it', async () => {
    await request(app).post('/api/miners/ping')
      .send({ address: ADDR, worker: 'rig01', gpu: 'RTX 4090', hashrate: 100, llmModel: 'Gemma-4-E4B-it-Q4_K_M' });
    const res = await request(app).get('/api/miners');
    expect(res.body.miners[0].llmModel).toBe('Gemma-4-E4B-it-Q4_K_M');
    expect(res.body.miners[0].cards[0].llmModel).toBe('Gemma-4-E4B-it-Q4_K_M');
  });

  // A client too old to report the field (or a card with too little VRAM to
  // serve) keeps reading null rather than failing the ping.
  test('POST /api/miners/ping without an llmModel leaves it null', async () => {
    await request(app).post('/api/miners/ping')
      .send({ address: ADDR, worker: 'rig01', gpu: 'RTX 4090', hashrate: 100 });
    const res = await request(app).get('/api/miners');
    expect(res.body.miners[0].llmModel).toBeNull();
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
