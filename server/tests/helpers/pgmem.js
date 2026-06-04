// An in-memory Postgres for tests, backed by pg-mem. Returns a pool with the
// same `query`/`connect` surface the services use against the real `pg` Pool in
// production, so tests exercise real SQL without needing a live database.
const { newDb } = require('pg-mem');
const { initSchema } = require('../../src/db');

async function createTestDb() {
  const mem = newDb();
  const pg = mem.adapters.createPg();
  const pool = new pg.Pool();
  await initSchema(pool);
  return pool;
}

module.exports = { createTestDb };
