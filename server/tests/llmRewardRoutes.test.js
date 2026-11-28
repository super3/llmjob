// The LLM rewards routes (#331): the public owner view, and the admin list and
// mark-paid endpoints behind requireAuth + requireAdmin.
jest.mock('@clerk/clerk-sdk-node', () => ({ verifyToken: jest.fn() }));

const request = require('supertest');
const express = require('express');
const { verifyToken } = require('@clerk/clerk-sdk-node');
const routes = require('../src/routes');
const { dayOf, addDays } = require('../src/services/llmRewardService');
const { createTestDb } = require('./helpers/pgmem');

const ADDR = 'prl1p' + 'a'.repeat(30);

function makeApp(db) {
  const app = express();
  app.use(express.json());
  app.locals.db = db;
  app.use('/api', routes);
  return app;
}

describe('LLM rewards API', () => {
  let db;
  let app;
  const yesterday = addDays(dayOf(Date.now()), -1);

  beforeEach(async () => {
    db = await createTestDb();
    app = makeApp(db);
    process.env.ADMIN_USER_IDS = 'admin1';
    verifyToken.mockResolvedValue({ sub: 'admin1' });
    jest.spyOn(console, 'error').mockImplementation(() => {});
    await db.query("INSERT INTO prl_prices (day, usd) VALUES ('2020-01-01', 2)");
    await db.query(
      `INSERT INTO llm_usage (id, ts, day, node_id, address, model, prompt_tokens, output_tokens, source, usd)
       VALUES ('u1', 1, $1, 'n', $2, 'm', 100, 900, 'api', 4)`,
      [yesterday, ADDR]
    );
  });

  afterEach(async () => {
    delete process.env.ADMIN_USER_IDS;
    jest.restoreAllMocks();
    if (db.end) await db.end();
  });

  test('GET /api/llm-rewards/:address returns the owner view', async () => {
    const res = await request(app).get('/api/llm-rewards/' + ADDR);
    expect(res.status).toBe(200);
    expect(res.body).toMatchObject({ address: ADDR, unpaidPrl: 2, paidPrl: 0, minPayoutPrl: 1 });
  });

  test('GET /api/llm-rewards/:address rejects a bad address with 400', async () => {
    const res = await request(app).get('/api/llm-rewards/nope');
    expect(res.status).toBe(400);
    expect(res.body.error).toBe('Not a PRL address');
  });

  test('admin routes need an admin', async () => {
    verifyToken.mockResolvedValue({ sub: 'someone' });
    const res = await request(app).get('/api/admin/llm-rewards').set('Authorization', 'Bearer t');
    expect(res.status).toBe(403);
    const none = await request(app).post('/api/admin/llm-rewards/paid');
    expect(none.status).toBe(401);
  });

  test('GET /api/admin/llm-rewards lists what is owed, as JSON or CSV', async () => {
    const res = await request(app).get('/api/admin/llm-rewards').set('Authorization', 'Bearer t');
    expect(res.status).toBe(200);
    expect(res.body.rows).toEqual([expect.objectContaining({ address: ADDR, prl: 2, payable: true })]);

    const csv = await request(app)
      .get(`/api/admin/llm-rewards?from=2020-01-01&to=${yesterday}&format=csv`)
      .set('Authorization', 'Bearer t');
    expect(csv.status).toBe(200);
    expect(csv.headers['content-type']).toMatch(/^text\/csv/);
    expect(csv.headers['content-disposition']).toBe(`attachment; filename="llm-rewards-2020-01-01-to-${yesterday}.csv"`);
    expect(csv.text).toBe(`address,days,requests,prompt_tokens,output_tokens,usd,prl,payable\n${ADDR},1,1,100,900,4.000000,2.000000,yes\n`);
  });

  test('GET /api/admin/llm-rewards rejects a bad range with 400', async () => {
    const res = await request(app).get('/api/admin/llm-rewards?to=soon').set('Authorization', 'Bearer t');
    expect(res.status).toBe(400);
  });

  test('POST /api/admin/llm-rewards/paid marks the range paid', async () => {
    const res = await request(app).post('/api/admin/llm-rewards/paid')
      .set('Authorization', 'Bearer t').send({ tx: 'abc' });
    expect(res.status).toBe(200);
    expect(res.body.paid).toEqual([expect.objectContaining({ address: ADDR, prl: 2, tx: 'abc' })]);
    const after = await request(app).get('/api/llm-rewards/' + ADDR);
    expect(after.body).toMatchObject({ unpaidPrl: 0, paidPrl: 2, payouts: 1 });
  });

  test('POST /api/admin/llm-rewards/paid works with no body', async () => {
    const appNoJson = express();
    appNoJson.locals.db = db;
    appNoJson.use('/api', routes);
    const res = await request(appNoJson).post('/api/admin/llm-rewards/paid').set('Authorization', 'Bearer t');
    expect(res.status).toBe(200);
  });

  test('a database failure is a 500', async () => {
    app.locals.db = { query: () => Promise.reject(new Error('db down')) };
    const res = await request(app).get('/api/llm-rewards/' + ADDR);
    expect(res.status).toBe(500);
    expect(res.body.error).toBe('Failed to get LLM rewards');
    const owed = await request(app).get('/api/admin/llm-rewards').set('Authorization', 'Bearer t');
    expect(owed.status).toBe(500);
    const paid = await request(app).post('/api/admin/llm-rewards/paid').set('Authorization', 'Bearer t');
    expect(paid.status).toBe(500);
  });
});
