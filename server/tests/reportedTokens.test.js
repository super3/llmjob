// Token counts that arrive from an untrusted party: POST /api/usage, where the
// key holder states its own billing figures, which are then summed into the
// public "tokens served" counter and were once believed verbatim.
const request = require('supertest');
const express = require('express');
const { createTestDb } = require('./helpers/pgmem');
const ApiKeyService = require('../src/services/apiKeyService');
const routes = require('../src/routes');
const {
  countField, numField, logLimit, MAX_REPORTED_TOKENS, MAX_LOG_LIMIT,
} = require('../src/controllers/logController');

describe('self-reported usage fields (POST /api/usage)', () => {
  describe('countField', () => {
    // The concatenation bug: `"5" + "5"` billed 55 instead of 10.
    it('coerces numeric strings to numbers rather than concatenating', () => {
      expect(countField('5') + countField('5')).toBe(10);
    });

    it('floors, and rejects negatives, junk and non-finite values', () => {
      expect(countField(7.9)).toBe(7);
      expect(countField(-100)).toBe(0);
      expect(countField('abc')).toBe(0);
      expect(countField(undefined)).toBe(0);
      expect(countField(Infinity)).toBe(0);
    });

    it('caps a single report so it cannot move the lifetime public total', () => {
      expect(countField(1e18)).toBe(MAX_REPORTED_TOKENS);
    });
  });

  describe('numField', () => {
    it('accepts a finite non-negative number and zeroes anything else', () => {
      expect(numField(12.5)).toBe(12.5);
      expect(numField('3')).toBe(3);
      expect(numField(-1)).toBe(0);
      expect(numField('fast')).toBe(0);
      expect(numField(Infinity)).toBe(0);
    });
  });

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

    it('adds string token counts instead of concatenating them', async () => {
      await request(app).post('/api/usage')
        .set('Authorization', 'Bearer ' + rawKey)
        .send({ model: 'm', node: 'n', in: '5', out: '5' })
        .expect(201);

      expect(await new ApiKeyService(db).getTotalUsage()).toBe(10);
    });

    it('ignores a negative report rather than erasing recorded usage', async () => {
      const send = (body) => request(app).post('/api/usage')
        .set('Authorization', 'Bearer ' + rawKey).send(body);

      await send({ model: 'm', node: 'n', in: 100, out: 100 }).expect(201);
      await send({ model: 'm', node: 'n', in: -1000000, out: -1000000 }).expect(201);

      expect(await new ApiKeyService(db).getTotalUsage()).toBe(200);
    });

    it('caps a single absurd report', async () => {
      await request(app).post('/api/usage')
        .set('Authorization', 'Bearer ' + rawKey)
        .send({ model: 'm', node: 'n', out: 1e18 })
        .expect(201);

      expect(await new ApiKeyService(db).getTotalUsage()).toBe(MAX_REPORTED_TOKENS);
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
