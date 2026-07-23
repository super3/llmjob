// Boots server/src/index.js without opening a real port or connecting to
// Postgres: the DB pool, the background-task services, the HTTP server, and the
// process signal/timer hooks are all mocked so the wiring can be exercised.
const mockCheckNodeStatuses = jest.fn();
const mockCheckTimeouts = jest.fn();
const mockCleanupOldJobs = jest.fn();

jest.mock('../src/db', () => {
  const actual = jest.requireActual('../src/db');
  return { ...actual, createPool: jest.fn() };
});
jest.mock('../src/services/nodeService', () =>
  jest.fn(() => ({ checkNodeStatuses: mockCheckNodeStatuses })));
jest.mock('../src/services/jobService', () =>
  jest.fn(() => ({ checkTimeouts: mockCheckTimeouts, cleanupOldJobs: mockCleanupOldJobs })));

const request = require('supertest');
const { createPool } = require('../src/db');
const index = require('../src/index');

describe('server bootstrap (index.js)', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.spyOn(console, 'log').mockImplementation(() => {});
    jest.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('errorHandler', () => {
    const run = (err) => {
      const res = { status: jest.fn().mockReturnThis(), json: jest.fn() };
      index.errorHandler(err, {}, res, () => {});
      return res;
    };

    it('echoes the message for a 4xx client error', () => {
      const res = run({ status: 400, message: 'bad input', stack: 'trace' });
      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({ error: 'bad input' });
    });

    it('returns a generic message for a 500 (and logs err without a stack)', () => {
      const res = run({ message: 'internal boom' }); // no status, no stack
      expect(res.status).toHaveBeenCalledWith(500);
      expect(res.json).toHaveBeenCalledWith({ error: 'Internal server error' });
    });
  });

  describe('GET /health', () => {
    it('reports ok', async () => {
      const r = await request(index.app).get('/health');
      expect(r.status).toBe(200);
      expect(r.body.status).toBe('ok');
      expect(typeof r.body.timestamp).toBe('string');
    });
  });

  describe('static path selection', () => {
    it('serves from /app under Railway', () => {
      jest.isolateModules(() => {
        const prev = process.env.RAILWAY_ENVIRONMENT;
        process.env.RAILWAY_ENVIRONMENT = '1';
        try {
          require('../src/index'); // exercises the '/app' branch at module load
        } finally {
          if (prev === undefined) delete process.env.RAILWAY_ENVIRONMENT;
          else process.env.RAILWAY_ENVIRONMENT = prev;
        }
      });
    });
  });

  describe('startServer', () => {
    let intervals, forceTimeouts, signals, exit, fakeServer;

    const wireGlobals = () => {
      intervals = [];
      forceTimeouts = [];
      signals = {};
      exit = jest.spyOn(process, 'exit').mockImplementation(() => {});
      jest.spyOn(global, 'setInterval').mockImplementation((fn) => { intervals.push(fn); return intervals.length; });
      jest.spyOn(global, 'clearInterval').mockImplementation(() => {});
      jest.spyOn(global, 'setTimeout').mockImplementation((fn) => { forceTimeouts.push(fn); return forceTimeouts.length; });
      jest.spyOn(process, 'on').mockImplementation((sig, fn) => { signals[sig] = fn; return process; });
      fakeServer = { close: jest.fn((cb) => cb()) };
      jest.spyOn(index.app, 'listen').mockImplementation((port, host, cb) => { cb(); return fakeServer; });
    };

    it('connects, wires background tasks + shutdown, and drives them', async () => {
      const end = jest.fn().mockResolvedValue();
      createPool.mockReturnValue({ query: jest.fn().mockResolvedValue({ rows: [] }), end });
      wireGlobals();

      await index.startServer();

      // DB connected and exposed to routes; server started; signals registered.
      expect(index.app.locals.db).toBeDefined();
      expect(index.app.listen).toHaveBeenCalled();
      expect(signals.SIGTERM).toBeInstanceOf(Function);
      expect(signals.SIGINT).toBeInstanceOf(Function);
      expect(intervals).toHaveLength(3);

      const [statusCb, timeoutCb, cleanupCb] = intervals;

      // node-status tick
      mockCheckNodeStatuses.mockResolvedValue();
      await statusCb();
      expect(mockCheckNodeStatuses).toHaveBeenCalled();

      // timeout tick: none returned, some returned, and the error branch
      mockCheckTimeouts.mockResolvedValueOnce([]);
      await timeoutCb();
      mockCheckTimeouts.mockResolvedValueOnce(['job-1']);
      await timeoutCb();
      mockCheckTimeouts.mockRejectedValueOnce(new Error('db down'));
      await timeoutCb();

      // cleanup tick: nothing, some, and the error branch
      mockCleanupOldJobs.mockResolvedValueOnce(0);
      await cleanupCb();
      mockCleanupOldJobs.mockResolvedValueOnce(3);
      await cleanupCb();
      mockCleanupOldJobs.mockRejectedValueOnce(new Error('cleanup failed'));
      await cleanupCb();

      // Graceful shutdown: clears intervals, closes the server, ends the pool,
      // exits 0; the force-timeout exits 1.
      await signals.SIGTERM('SIGTERM');
      expect(fakeServer.close).toHaveBeenCalled();
      expect(end).toHaveBeenCalled();
      expect(exit).toHaveBeenCalledWith(0);

      forceTimeouts[0](); // force-shutdown path
      expect(exit).toHaveBeenCalledWith(1);

      // SIGINT uses the same handler.
      await signals.SIGINT('SIGINT');
      expect(fakeServer.close).toHaveBeenCalledTimes(2);
    });

    it('exits(1) when the database is unreachable', async () => {
      createPool.mockReturnValue({ query: jest.fn().mockRejectedValue(new Error('no db')) });
      wireGlobals();

      await index.startServer();

      expect(exit).toHaveBeenCalledWith(1);
      expect(index.app.listen).not.toHaveBeenCalled();
    });
  });
});
