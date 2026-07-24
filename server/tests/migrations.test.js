// Exercises the actual node-pg-migrate migration files against an in-memory
// Postgres, so the migrations that run in production are covered — not just the
// db.js SCHEMA constant the other tests build from. Each migration here uses
// only `pgm.sql(...)`; the runner collects those statements and executes them.
const fs = require('fs');
const path = require('path');
const { newDb } = require('pg-mem');

const MIGRATIONS_DIR = path.join(__dirname, '../migrations');

function migrationFiles() {
  return fs.readdirSync(MIGRATIONS_DIR).filter((f) => f.endsWith('.cjs')).sort();
}

function load(file) {
  return require(path.join(MIGRATIONS_DIR, file));
}

function freshPool() {
  return new (newDb().adapters.createPg().Pool)();
}

// Run a migration's up/down by capturing its pgm.sql() calls and executing them.
async function apply(pool, mod, direction) {
  const statements = [];
  const pgm = { sql: (s) => statements.push(s) };
  await mod[direction](pgm);
  for (const s of statements) {
    await pool.query(s);
  }
}

const byName = (needle) => migrationFiles().find((f) => f.includes(needle));

describe('migrations', () => {
  it('every migration exports up/down functions', () => {
    const files = migrationFiles();
    expect(files.length).toBeGreaterThan(0);
    for (const file of files) {
      const mod = load(file);
      expect(typeof mod.up).toBe('function');
      expect(typeof mod.down).toBe('function');
    }
  });

  it('init migration creates a usable schema and rolls back', async () => {
    const pool = freshPool();
    const init = load(byName('init-schema'));

    await apply(pool, init, 'up');

    // Each core table exists and accepts a row.
    await pool.query("INSERT INTO jobs (id, status) VALUES ('j1', 'pending')");
    await pool.query("INSERT INTO nodes (node_id) VALUES ('n1')");
    await pool.query("INSERT INTO api_keys (hash) VALUES ('h1')");
    await pool.query("INSERT INTO request_logs (id) VALUES ('l1')");
    await pool.query("INSERT INTO node_join_tokens (user_id, token) VALUES ('u1', 't1')");
    await pool.query("INSERT INTO miners (id) VALUES ('m1')");
    const jobs = await pool.query('SELECT id FROM jobs');
    expect(jobs.rows).toHaveLength(1);

    await apply(pool, init, 'down');
    await expect(pool.query('SELECT 1 FROM jobs')).rejects.toBeDefined();
  });

  it('add-miners creates and drops the miners table', async () => {
    // Applied against a bare database (as it would be on a deployment created
    // before the miners table existed) the migration must create it standalone.
    const pool = freshPool();
    const mod = load(byName('add-miners'));

    await apply(pool, mod, 'up');
    await pool.query("INSERT INTO miners (id, address) VALUES ('m1', 'prl1abc')");
    expect((await pool.query('SELECT id FROM miners')).rows).toHaveLength(1);

    await apply(pool, mod, 'down');
    await expect(pool.query('SELECT 1 FROM miners')).rejects.toBeDefined();
  });

  it('add-miner-vram adds and drops the vram columns', async () => {
    const pool = freshPool();
    await apply(pool, load(byName('init-schema')), 'up');
    // Simulate an older miners table without the vram columns.
    await pool.query('ALTER TABLE miners DROP COLUMN vram_used');
    await pool.query('ALTER TABLE miners DROP COLUMN vram_total');

    const mod = load(byName('add-miner-vram'));
    await apply(pool, mod, 'up');
    await pool.query("INSERT INTO miners (id, vram_used, vram_total) VALUES ('m1', 1.5, 24)");
    expect((await pool.query('SELECT vram_used FROM miners')).rows[0].vram_used).toBe(1.5);

    await apply(pool, mod, 'down');
    await expect(pool.query('SELECT vram_used FROM miners')).rejects.toBeDefined();
  });

  it('add-miner-version adds and drops the version column', async () => {
    const pool = freshPool();
    await apply(pool, load(byName('init-schema')), 'up');
    await pool.query('ALTER TABLE miners DROP COLUMN version');

    const mod = load(byName('add-miner-version'));
    await apply(pool, mod, 'up');
    await pool.query("INSERT INTO miners (id, version) VALUES ('m1', '0.2.7')");
    expect((await pool.query('SELECT version FROM miners')).rows[0].version).toBe('0.2.7');

    await apply(pool, mod, 'down');
    await expect(pool.query('SELECT version FROM miners')).rejects.toBeDefined();
  });

  it('add-miner-llm-model adds and drops the llm_model column', async () => {
    const pool = freshPool();
    await apply(pool, load(byName('init-schema')), 'up');
    await pool.query('ALTER TABLE miners DROP COLUMN llm_model');

    const mod = load(byName('add-miner-llm-model'));
    await apply(pool, mod, 'up');
    await pool.query("INSERT INTO miners (id, llm_model) VALUES ('m1', 'Gemma-4-E4B-it-Q4_K_M')");
    expect((await pool.query('SELECT llm_model FROM miners')).rows[0].llm_model).toBe('Gemma-4-E4B-it-Q4_K_M');

    await apply(pool, mod, 'down');
    await expect(pool.query('SELECT llm_model FROM miners')).rejects.toBeDefined();
  });
});
