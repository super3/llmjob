const request = require('supertest');
const express = require('express');
const routes = require('../src/routes');
const { createTestDb } = require('./helpers/pgmem');
const { generateKeypair, fingerprint, signRig } = require('../../earn/src/shared/node');

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

  // The retired LLM fields older clients still send are ignored, not stored or
  // echoed back onto the public board.
  test('POST /api/miners/ping ignores the retired llmModel/nodeId fields', async () => {
    await request(app).post('/api/miners/ping')
      .send({ address: ADDR, worker: 'rig01', gpu: 'RTX 4090', hashrate: 100, llmModel: 'Gemma-4-E4B-it-Q4_K_M', nodeId: '5840fc' });
    const res = await request(app).get('/api/miners');
    expect(res.body.miners[0]).not.toHaveProperty('llmModel');
    expect(res.body.miners[0]).not.toHaveProperty('nodeId');
    const stored = (await db.query('SELECT llm_model, node_id FROM miners', [])).rows[0];
    expect(stored).toEqual({ llm_model: null, node_id: null });
  });

  test('POST /api/miners/ping stores the health fields and a verified rig id', async () => {
    const kp = generateKeypair();
    const identity = { ...kp, nodeId: fingerprint(kp.publicKey) };
    const res = await request(app).post('/api/miners/ping').send({
      address: ADDR, worker: 'rig01', hashrate: 100, rejected: 2, tempC: 61, powerW: 290, powerLimitW: 450,
      coreClockMhz: 2520, memClockMhz: 10501, fanPct: 55, driver: '580.82', os: 'linux', client: 'gui',
      uptimeSec: 120, lastShareSec: 4, ...signRig(identity, Date.now()),
    });
    expect(res.status).toBe(200);
    const row = (await db.query('SELECT * FROM miners', [])).rows[0];
    expect(row).toMatchObject({ temp_c: 61, power_w: 290, core_clock_mhz: 2520, mem_clock_mhz: 10501, fan_pct: 55,
      driver: '580.82', os: 'linux', client: 'gui', rig_id: identity.nodeId });
    expect(Number(row.rejected)).toBe(2);
  });

  test('POST /api/miners/ping with no JSON body is a bad address, not a crash', async () => {
    const res = await request(app).post('/api/miners/ping').set('Content-Type', 'text/plain').send('hello');
    expect(res.status).toBe(400);
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
