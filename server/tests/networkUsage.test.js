// GET /api/chat/usage — the network page's "tokens served" figure and the free
// hosted-model budget. The path is kept from when the free web chat lived beside
// it (see networkUsageController); the chat itself is gone.
const request = require('supertest');
const express = require('express');
const { createTestDb } = require('./helpers/pgmem');
const { initUsageRoutes } = require('../src/routes');
const ChatUsageService = require('../src/services/chatUsageService');
const ApiKeyService = require('../src/services/apiKeyService');
const NetworkUsageController = require('../src/controllers/networkUsageController');

function makeApp(db, opts) {
  const app = express();
  app.locals.db = db;
  initUsageRoutes(app, opts);
  return app;
}

describe('GET /api/chat/usage', () => {
  let db;
  beforeEach(async () => { db = await createTestDb(); });
  afterEach(async () => { if (db.end) await db.end(); });

  it('reports running totals and remaining free budget', async () => {
    await new ChatUsageService(db).recordUsage({ model: 'm', inTokens: 10, outTokens: 20 });
    const res = await request(makeApp(db, { freeBudget: 100 })).get('/api/chat/usage');
    expect(res.status).toBe(200);
    expect(res.body.totals.totalTokens).toBe(30);
    expect(res.body.freeBudget).toBe(100);
    expect(res.body.remaining).toBe(70);
    expect(res.body.exhausted).toBe(false);
  });

  it('marks the budget exhausted once totals cross it', async () => {
    await new ChatUsageService(db).recordUsage({ model: 'm', inTokens: 60, outTokens: 60 });
    const res = await request(makeApp(db, { freeBudget: 100 })).get('/api/chat/usage');
    expect(res.body.exhausted).toBe(true);
    expect(res.body.remaining).toBe(0);
  });

  it('combines API-gateway tokens into the network total, leaving the free cap alone', async () => {
    await new ChatUsageService(db).recordUsage({ model: 'm', inTokens: 10, outTokens: 20 });
    const keys = new ApiKeyService(db);
    const k = await keys.createKey('u1', 'gateway');
    await keys.recordUsage(ApiKeyService.sha256(k.key), 500);

    const res = await request(makeApp(db, { freeBudget: 100 })).get('/api/chat/usage');
    expect(res.body.network.apiTokens).toBe(500);
    expect(res.body.network.totalTokens).toBe(530); // 30 hosted + 500 node-served API
    // The free-usage cap must not see node-served API traffic, or it would burn
    // the free budget on somebody else's hardware.
    expect(res.body.totals.totalTokens).toBe(30);
    expect(res.body.remaining).toBe(70);
    expect(res.body.exhausted).toBe(false);
  });

  it('counts hosted-model API traffic once, and against the free cap', async () => {
    // A hosted generation served for an API key is recorded twice on purpose:
    // in the OpenRouter totals (it is our credit being spent, so the cap must
    // see it) and against the key (so the owner's dashboard is complete). The
    // public "tokens served" figure must not double it.
    const usage = new ChatUsageService(db);
    await usage.recordUsage({ model: 'm', inTokens: 10, outTokens: 20 });                 // lifetime chat tokens
    await usage.recordUsage({ model: 'm', inTokens: 4, outTokens: 6, viaApiKey: true });  // hosted via /v1
    const keys = new ApiKeyService(db);
    const k = await keys.createKey('u1', 'gateway');
    await keys.recordUsage(ApiKeyService.sha256(k.key), 10 + 500); // the same 10, plus node-served work

    const res = await request(makeApp(db, { freeBudget: 100 })).get('/api/chat/usage');
    expect(res.body.network.apiTokens).toBe(510);
    expect(res.body.network.totalTokens).toBe(540); // 30 chat + 10 hosted + 500 node, each once
    expect(res.body.totals.totalTokens).toBe(40);
    expect(res.body.remaining).toBe(60);
  });

  it('reports zero API tokens when no keys have been used', async () => {
    const res = await request(makeApp(db, { freeBudget: 0 })).get('/api/chat/usage');
    expect(res.body.network).toEqual({ apiTokens: 0, totalTokens: 0 });
  });

  it('reports no cap when the free budget is disabled', async () => {
    const res = await request(makeApp(db, { freeBudget: 0 })).get('/api/chat/usage');
    expect(res.body.freeBudget).toBeNull();
    expect(res.body.remaining).toBeNull();
    expect(res.body.exhausted).toBe(false);
  });

  it('works with no options, reading the budget from the environment defaults', async () => {
    const res = await request(makeApp(db)).get('/api/chat/usage');
    expect(res.status).toBe(200);
    expect(res.body.network).toEqual({ apiTokens: 0, totalTokens: 0 });
  });

  // The web chat's routes are gone; only the usage figure is left under /api/chat.
  it('no longer serves the web chat endpoints', async () => {
    const app = makeApp(db);
    expect((await request(app).post('/api/chat/completions').send({ messages: [] })).status).toBe(404);
    expect((await request(app).get('/api/chat/models')).status).toBe(404);
  });
});

describe('NetworkUsageController', () => {
  it('falls back to OpenRouter-only totals when no key service is wired up', async () => {
    // Injected service bundles need not carry every service; a missing one must
    // not 500.
    const ctrl = new NetworkUsageController({
      services: { chatUsage: { getTotals: async () => ({ totalTokens: 42 }) } }
    });
    let body = null;
    await ctrl.usage({}, { json: (b) => { body = b; } });
    expect(body.network).toEqual({ apiTokens: 0, totalTokens: 42 });
  });
});
