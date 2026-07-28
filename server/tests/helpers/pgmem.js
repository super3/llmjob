// An in-memory Postgres for tests, backed by pg-mem. Returns a pool with the
// same `query`/`connect` surface the services use against the real `pg` Pool in
// production, so tests exercise real SQL without needing a live database.
const { newDb } = require('pg-mem');
const { initSchema } = require('../../src/db');

// pg-mem can't parse "FOR UPDATE SKIP LOCKED" (it has no real row locking), and it
// has no concurrency for the clause to matter anyway. Strip it so tests exercise
// the same query the production code runs, minus the lock modifier pg-mem rejects.
const stripSkipLocked = (sql) =>
  typeof sql === 'string' ? sql.replace(/\s+SKIP\s+LOCKED/gi, '') : sql;

async function createTestDb() {
  const mem = newDb();
  const pg = mem.adapters.createPg();
  const pool = new pg.Pool();
  const poolQuery = pool.query.bind(pool);
  pool.query = (sql, ...rest) => poolQuery(stripSkipLocked(sql), ...rest);
  const poolConnect = pool.connect.bind(pool);
  pool.connect = async (...args) => {
    const client = await poolConnect(...args);
    const clientQuery = client.query.bind(client);
    client.query = (sql, ...rest) => clientQuery(stripSkipLocked(sql), ...rest);
    return client;
  };
  await initSchema(pool);
  return pool;
}

module.exports = { createTestDb };
