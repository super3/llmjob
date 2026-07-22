const { createPool, SCHEMA } = require('../src/db');
const { createTestDb } = require('./helpers/pgmem');

describe('db module', () => {
  it('createPool returns a usable pg pool', async () => {
    const pool = createPool();
    expect(typeof pool.query).toBe('function');
    expect(typeof pool.connect).toBe('function');
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
