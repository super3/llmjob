const LogService = require('../src/services/logService');
const { createCamelClient } = require('./helpers/camelRedis');

describe('LogService', () => {
  let redisClient;
  let service;

  beforeEach(async () => {
    redisClient = createCamelClient();
    service = new LogService(redisClient);
    await redisClient.flushall();
  });

  afterEach(async () => {
    await redisClient.quit();
  });

  describe('recordLog', () => {
    it('stores an entry with defaults filled in', async () => {
      const entry = await service.recordLog('user1', { model: 'gemma', node: 'rig1' });
      expect(entry.id).toMatch(/^log_/);
      expect(entry.app).toBe('api');
      expect(entry.in).toBe(0);
      expect(entry.out).toBe(0);
      expect(entry.speed).toBe(0);
      expect(entry.finish).toBe('stop');
      expect(entry.key).toBeNull();
      expect(entry.timestamp).toEqual(expect.any(Number));
    });

    it('honors explicitly provided fields', async () => {
      const entry = await service.recordLog('user1', {
        model: 'gemma', node: 'rig1', app: 'bench',
        in: 100, out: 50, speed: 91.4, finish: 'length', key: 'home',
        timestamp: 1700000000000
      });
      expect(entry).toMatchObject({
        model: 'gemma', node: 'rig1', app: 'bench',
        in: 100, out: 50, speed: 91.4, finish: 'length', key: 'home',
        timestamp: 1700000000000
      });
    });

    it('trims old entries beyond the cap', async () => {
      // Cap is 200; insert 205 and confirm only 200 remain.
      for (let i = 0; i < 205; i++) {
        await service.recordLog('user1', { model: 'm', node: 'n', timestamp: 1000 + i });
      }
      const count = await redisClient.zCard('user_logs:user1');
      expect(count).toBe(200);

      // The five oldest timestamps should have been evicted.
      const logs = await service.getLogs('user1', 1000);
      const timestamps = logs.map((l) => l.timestamp);
      expect(Math.min(...timestamps)).toBe(1005);
    });
  });

  describe('getLogs', () => {
    it('returns an empty array when there are none', async () => {
      expect(await service.getLogs('nobody')).toEqual([]);
    });

    it('returns newest first and respects the limit', async () => {
      await service.recordLog('user1', { model: 'a', node: 'n', timestamp: 100 });
      await service.recordLog('user1', { model: 'b', node: 'n', timestamp: 300 });
      await service.recordLog('user1', { model: 'c', node: 'n', timestamp: 200 });

      const logs = await service.getLogs('user1', 2);
      expect(logs).toHaveLength(2);
      expect(logs[0].model).toBe('b');
      expect(logs[1].model).toBe('c');
    });
  });

  describe('getActivity', () => {
    it('returns 24 zeroed buckets when there is no traffic', async () => {
      const buckets = await service.getActivity('nobody', 0);
      expect(buckets).toHaveLength(24);
      expect(buckets.every((b) => b === 0)).toBe(true);
    });

    it('buckets entries by hour within the window', async () => {
      const now = 24 * 60 * 60 * 1000; // window start is 0
      const HOUR = 60 * 60 * 1000;
      // Two in the first hour, one in the last hour.
      await service.recordLog('user1', { model: 'm', node: 'n', timestamp: 10 });
      await service.recordLog('user1', { model: 'm', node: 'n', timestamp: 20 });
      await service.recordLog('user1', { model: 'm', node: 'n', timestamp: now });

      const buckets = await service.getActivity('user1', now);
      expect(buckets[0]).toBe(2);
      expect(buckets[23]).toBe(1);
    });

    it('clamps a timestamp exactly at now into the last bucket', async () => {
      const HOUR = 60 * 60 * 1000;
      const now = 100 * HOUR;
      const windowStart = now - 24 * HOUR;
      // Exactly at the window start (idx 0) and exactly at now (would be idx 24).
      await service.recordLog('user1', { model: 'm', node: 'n', timestamp: windowStart });
      await service.recordLog('user1', { model: 'm', node: 'n', timestamp: now });

      const buckets = await service.getActivity('user1', now);
      expect(buckets[0]).toBe(1);
      expect(buckets[23]).toBe(1);
      expect(buckets.reduce((a, b) => a + b, 0)).toBe(2);
    });
  });
});
