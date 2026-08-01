const { createPool, SCHEMA } = require('../src/db');
const { createTestDb } = require('./helpers/pgmem');

describe('db module', () => {
  it('createPool returns a usable pg pool', async () => {
    const pool = createPool();
    expect(typeof pool.query).toBe('function');
    expect(typeof pool.connect).toBe('function');
    await pool.end();
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
