const { createPool, SCHEMA } = require('../src/db');
const { createTestDb } = require('./helpers/pgmem');

describe('db module', () => {
  it('createPool returns a usable pg pool', async () => {
    const pool = createPool();
    expect(typeof pool.query).toBe('function');
    expect(typeof pool.connect).toBe('function');
    await pool.end();
  });

  // pg's defaults are max: 10 and NO connection timeout. Both are stated
  // explicitly now: 10 is small for a process where dispatch, both gateways and
  // the dashboard all check out connections at once, and an unset timeout means
  // a request arriving at an exhausted pool waits forever instead of failing.
  it('createPool sizes the pool and bounds the wait for a connection', async () => {
    const pool = createPool();
    expect(pool.options.max).toBe(20);
    expect(pool.options.connectionTimeoutMillis).toBe(10000);
    await pool.end();
  });

  it('createPool takes its sizing from the environment when set', async () => {
    const prev = [process.env.PGPOOL_MAX, process.env.PGPOOL_CONNECT_TIMEOUT_MS];
    process.env.PGPOOL_MAX = '42';
    process.env.PGPOOL_CONNECT_TIMEOUT_MS = '2500';
    const pool = createPool();
    expect(pool.options.max).toBe(42);
    expect(pool.options.connectionTimeoutMillis).toBe(2500);
    await pool.end();

    // A junk value falls back to the default rather than handing pg a NaN.
    process.env.PGPOOL_MAX = 'lots';
    process.env.PGPOOL_CONNECT_TIMEOUT_MS = '0';
    const fallback = createPool();
    expect(fallback.options.max).toBe(20);
    expect(fallback.options.connectionTimeoutMillis).toBe(10000);
    await fallback.end();

    [process.env.PGPOOL_MAX, process.env.PGPOOL_CONNECT_TIMEOUT_MS] = prev;
    if (prev[0] === undefined) delete process.env.PGPOOL_MAX;
    if (prev[1] === undefined) delete process.env.PGPOOL_CONNECT_TIMEOUT_MS;
  });

  // pg emits 'error' on the Pool when a client sitting IDLE in it drops — a
  // Postgres restart or a provider idle-kill. An 'error' event with no listener
  // is rethrown by EventEmitter as an uncaught exception, so without this the
  // whole API process died on a routine database blip.
  it('createPool attaches an error listener so an idle-client drop is logged, not fatal', async () => {
    const spy = jest.spyOn(console, 'error').mockImplementation(() => {});
    const pool = createPool();

    expect(pool.listenerCount('error')).toBe(1);
    expect(() => pool.emit('error', new Error('idle client terminated'))).not.toThrow();
    expect(spy).toHaveBeenCalledWith(
      'Postgres pool error (idle client):', 'idle client terminated'
    );

    spy.mockRestore();
    await pool.end();
  });

  it('initSchema creates the tables (via the shared SCHEMA)', async () => {
    const db = await createTestDb(); // runs initSchema(SCHEMA)
    expect(SCHEMA).toMatch(/CREATE TABLE IF NOT EXISTS nodes/);
    const r = await db.query('SELECT count(*)::int AS c FROM nodes');
    expect(r.rows[0].c).toBe(0);
    if (db.end) await db.end();
  });
});
