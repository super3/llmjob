const request = require('supertest');
const express = require('express');
const routes = require('../src/routes');
const WaitlistService = require('../src/services/waitlistService');
const { createTestDb } = require('./helpers/pgmem');

function makeApp(db) {
  const app = express();
  app.use(express.json());
  app.locals.db = db;
  app.use('/api', routes);
  return app;
}

const rows = async (db) =>
  (await db.query('SELECT email, gpus, note, source FROM managed_waitlist ORDER BY email', [])).rows;

describe('WaitlistService helpers', () => {
  test('normalizeEmail trims and lowercases, tolerating null', () => {
    expect(WaitlistService.normalizeEmail('  Me@Example.COM ')).toBe('me@example.com');
    expect(WaitlistService.normalizeEmail(null)).toBe('');
  });

  test('isValidEmail accepts ordinary addresses and rejects junk', () => {
    expect(WaitlistService.isValidEmail('a@b.co')).toBe(true);
    expect(WaitlistService.isValidEmail('')).toBe(false);
    expect(WaitlistService.isValidEmail('no-at-sign.com')).toBe(false);
    expect(WaitlistService.isValidEmail('a@nodot')).toBe(false);
    expect(WaitlistService.isValidEmail('a b@c.co')).toBe(false);
    expect(WaitlistService.isValidEmail('a@' + 'b'.repeat(260) + '.co')).toBe(false);
  });

  test('parseGpus floors, clamps, and treats blanks and junk as unknown', () => {
    expect(WaitlistService.parseGpus('12.9')).toBe(12);
    expect(WaitlistService.parseGpus(0)).toBe(0);
    expect(WaitlistService.parseGpus(1e9)).toBe(100000);
    expect(WaitlistService.parseGpus(undefined)).toBeNull();
    expect(WaitlistService.parseGpus('')).toBeNull();
    expect(WaitlistService.parseGpus('lots')).toBeNull();
    expect(WaitlistService.parseGpus(-3)).toBeNull();
  });

  test('join with no input is an invalid-email error, not a crash', async () => {
    const service = new WaitlistService({ query: jest.fn() });
    await expect(service.join()).resolves.toEqual({ error: expect.stringMatching(/email/i) });
    expect(service.db.query).not.toHaveBeenCalled();
  });

  test('clampText trims, caps the length, and maps empty to null', () => {
    expect(WaitlistService.clampText('  hi  ', 10)).toBe('hi');
    expect(WaitlistService.clampText('abcdef', 3)).toBe('abc');
    expect(WaitlistService.clampText('   ', 10)).toBeNull();
    expect(WaitlistService.clampText(null, 10)).toBeNull();
  });
});

describe('POST /api/waitlist', () => {
  let db;
  let app;
  beforeEach(async () => {
    db = await createTestDb();
    app = makeApp(db);
  });
  afterEach(async () => {
    if (db.end) await db.end();
  });

  test('stores a signup, normalizing the email', async () => {
    const res = await request(app).post('/api/waitlist')
      .send({ email: ' Miner@Example.com ', gpus: '8', note: 'two 4090 rigs', source: 'managed' });
    expect(res.status).toBe(200);
    expect(res.body).toEqual({ success: true });
    expect(await rows(db)).toEqual([
      { email: 'miner@example.com', gpus: 8, note: 'two 4090 rigs', source: 'managed' },
    ]);
  });

  test('a repeat signup updates the row instead of duplicating it, keeping answers left blank', async () => {
    await request(app).post('/api/waitlist').send({ email: 'a@b.co', gpus: 4, note: 'first' });
    await request(app).post('/api/waitlist').send({ email: 'A@B.CO', gpus: 12 });
    expect(await rows(db)).toEqual([{ email: 'a@b.co', gpus: 12, note: 'first', source: null }]);
  });

  test('only an email is required', async () => {
    const res = await request(app).post('/api/waitlist').send({ email: 'solo@b.co' });
    expect(res.status).toBe(200);
    expect(await rows(db)).toEqual([{ email: 'solo@b.co', gpus: null, note: null, source: null }]);
  });

  test('rejects a bad or missing email with 400', async () => {
    const bad = await request(app).post('/api/waitlist').send({ email: 'nope' });
    expect(bad.status).toBe(400);
    expect(bad.body.error).toMatch(/email/i);
    const missing = await request(app).post('/api/waitlist').send({});
    expect(missing.status).toBe(400);
    expect(await rows(db)).toEqual([]);
  });

  test('a filled honeypot looks like success but stores nothing', async () => {
    const res = await request(app).post('/api/waitlist')
      .send({ email: 'bot@spam.co', website: 'http://spam.example' });
    expect(res.status).toBe(200);
    expect(res.body).toEqual({ success: true });
    expect(await rows(db)).toEqual([]);
  });

  test('returns 500 when the db fails', async () => {
    jest.spyOn(console, 'error').mockImplementation(() => {});
    const broken = makeApp({ query: () => Promise.reject(new Error('db down')) });
    const res = await request(broken).post('/api/waitlist').send({ email: 'a@b.co' });
    expect(res.status).toBe(500);
    expect(res.body.error).toMatch(/waitlist/i);
    console.error.mockRestore();
  });

  test('tolerates a request with no parsed body', async () => {
    const bare = express();
    bare.locals.db = db;
    bare.use('/api', routes); // no express.json(): req.body is undefined
    const res = await request(bare).post('/api/waitlist');
    expect(res.status).toBe(400);
  });
});
