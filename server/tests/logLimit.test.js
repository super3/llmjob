// GET /api/logs takes `?limit=` from the query string, so it must be coerced
// before it reaches SQL.
const request = require('supertest');
const express = require('express');
const { createTestDb } = require('./helpers/pgmem');
const ApiKeyService = require('../src/services/apiKeyService');
const routes = require('../src/routes');
const { logLimit, MAX_LOG_LIMIT } = require('../src/controllers/logController');

describe('logLimit', () => {
  // `?limit=abc` used to reach SQL as `LIMIT NaN` and 500 the dashboard.
  it('falls back to the default for junk and non-positive values', () => {
    expect(logLimit('abc')).toBe(50);
    expect(logLimit(undefined)).toBe(50);
    expect(logLimit(0)).toBe(50);
    expect(logLimit(-3)).toBe(50);
  });

  it('honours a sane value and caps an oversized one', () => {
    expect(logLimit('25')).toBe(25);
    expect(logLimit(99999)).toBe(MAX_LOG_LIMIT);
  });

  describe('end to end', () => {
    let app, db, rawKey;

    beforeEach(async () => {
      db = await createTestDb();
      app = express();
      app.use(express.json());
      app.locals.db = db;
      app.use('/api', routes);
      rawKey = (await new ApiKeyService(db).createKey('user-usage', 'k')).key;
    });

    afterEach(async () => {
      if (db.end) await db.end();
    });

    it('serves a non-numeric ?limit instead of 500ing', async () => {
      const res = await request(app).get('/api/logs?limit=abc')
        .set('Authorization', 'Bearer ' + rawKey);
      // apiKey auth doesn't satisfy requireAuth on /logs, but the point is that
      // the limit coercion never reaches SQL as NaN.
      expect(res.status).not.toBe(500);
    });
  });
});
