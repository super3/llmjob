const ChatUsageService = require('../src/services/chatUsageService');
const { CHAT_LOG_CAP } = ChatUsageService;
const { createTestDb } = require('./helpers/pgmem');

describe('ChatUsageService', () => {
  let db;
  let service;

  beforeEach(async () => {
    db = await createTestDb();
    service = new ChatUsageService(db);
  });

  afterEach(async () => {
    if (db.end) await db.end();
  });

  const count = async () => (await db.query('SELECT count(*)::int AS c FROM chat_requests')).rows[0].c;

  describe('recordUsage', () => {
    it('stores a perf row with defaults filled in and no prompt column', async () => {
      const entry = await service.recordUsage({ model: 'qwen' });
      expect(entry.id).toMatch(/^creq_/);
      expect(entry.in).toBe(0);
      expect(entry.out).toBe(0);
      expect(entry.total).toBe(0);
      expect(entry.speed).toBe(0);
      expect(entry.latencyMs).toBe(0);
      expect(entry.ttftMs).toBe(0);
      expect(entry.finish).toBe('stop');
      expect(entry.timestamp).toEqual(expect.any(Number));
      expect(entry).not.toHaveProperty('prompt');
    });

    it('honors provided fields, rounds/clamps tokens, and computes total', async () => {
      const entry = await service.recordUsage({
        model: 'llama', inTokens: 12.6, outTokens: 30.2, speed: 44.5,
        latencyMs: 812.7, ttftMs: 120.4, finish: 'length', timestamp: 1700000000000
      });
      expect(entry).toMatchObject({
        model: 'llama', in: 13, out: 30, total: 43, speed: 44.5,
        latencyMs: 813, ttftMs: 120, finish: 'length', timestamp: 1700000000000
      });
    });

    it('labels the model "unknown" when none is provided', async () => {
      const entry = await service.recordUsage({ inTokens: 1, outTokens: 1 });
      expect(entry.model).toBe('unknown');
    });

    it('never stores negative token counts', async () => {
      const entry = await service.recordUsage({ model: 'm', inTokens: -5, outTokens: -1 });
      expect(entry.in).toBe(0);
      expect(entry.out).toBe(0);
    });

    it('accumulates the running totals across requests', async () => {
      await service.recordUsage({ model: 'a', inTokens: 10, outTokens: 5 });
      await service.recordUsage({ model: 'b', inTokens: 3, outTokens: 7 });
      const totals = await service.getTotals();
      expect(totals).toEqual({ requests: 2, inTokens: 13, outTokens: 12, totalTokens: 25 });
    });

    it('trims perf rows beyond the cap but keeps lifetime totals intact', async () => {
      for (let i = 0; i < CHAT_LOG_CAP + 5; i++) {
        await service.recordUsage({ model: 'm', inTokens: 1, outTokens: 1, timestamp: 1000 + i });
      }
      expect(await count()).toBe(CHAT_LOG_CAP);
      const totals = await service.getTotals();
      expect(totals.requests).toBe(CHAT_LOG_CAP + 5);
      expect(totals.totalTokens).toBe((CHAT_LOG_CAP + 5) * 2);
      // The oldest rows are the ones trimmed.
      const recent = await service.getRecent(CHAT_LOG_CAP + 100);
      const oldest = Math.min(...recent.map((r) => r.timestamp));
      expect(oldest).toBe(1005);
    });
  });

  describe('getTotals', () => {
    it('returns zeroes before any usage is recorded', async () => {
      expect(await service.getTotals()).toEqual({ requests: 0, inTokens: 0, outTokens: 0, totalTokens: 0 });
    });
  });

  describe('getRecent', () => {
    it('returns an empty array when there are none', async () => {
      expect(await service.getRecent()).toEqual([]);
    });

    it('returns rows newest first, capped at the limit', async () => {
      await service.recordUsage({ model: 'a', timestamp: 100 });
      await service.recordUsage({ model: 'b', timestamp: 200 });
      await service.recordUsage({ model: 'c', timestamp: 300 });
      const recent = await service.getRecent(2);
      expect(recent.map((r) => r.model)).toEqual(['c', 'b']);
      expect(recent[0]).toMatchObject({ model: 'c', in: 0, out: 0, total: 0, finish: 'stop' });
    });
  });
});
