/* eslint-disable camelcase */
// Index the three hottest query shapes, none of which had one.
//
//   1. nodes (last_seen) — "which nodes are live right now". Every job creation
//      resolves its model pin against it, GET /v1/models answers from it, and
//      the public network page reads it on a 15s timer. Each of those was a
//      full scan of the nodes table.
//
//   2. jobs (priority DESC, created_at ASC) WHERE status = 'pending' — the
//      assignment query, which every serving node runs on every poll and is by
//      far the hottest statement in the system. idx_jobs_status could find the
//      pending rows but not their order, so Postgres still sorted the whole
//      pending set to answer "ORDER BY priority DESC, created_at ASC LIMIT n".
//      Partial on `pending` so the index holds only the live queue rather than
//      the day of completed jobs the table keeps for the cleanup sweep.
//
//   3. jobs (status, updated_at) — the timeout and cleanup sweeps, which run
//      every 30 seconds and every hour respectively.
//
// CREATE INDEX IF NOT EXISTS is idempotent, so this is a no-op on a fresh
// database whose schema (server/src/db.js) already declares all three.
//
// Deliberately NOT CONCURRENTLY: node-pg-migrate wraps each migration in a
// transaction and CREATE INDEX CONCURRENTLY cannot run inside one. These tables
// are small enough that the brief write lock is not worth the added complexity
// of a non-transactional migration.

exports.shorthands = undefined;

exports.up = (pgm) => {
  pgm.sql(`
    CREATE INDEX IF NOT EXISTS idx_nodes_last_seen ON nodes (last_seen);
    CREATE INDEX IF NOT EXISTS idx_jobs_pending_queue
      ON jobs (priority DESC, created_at ASC) WHERE status = 'pending';
    CREATE INDEX IF NOT EXISTS idx_jobs_status_updated ON jobs (status, updated_at);
  `);
};

exports.down = (pgm) => {
  pgm.sql(`
    DROP INDEX IF EXISTS idx_nodes_last_seen;
    DROP INDEX IF EXISTS idx_jobs_pending_queue;
    DROP INDEX IF EXISTS idx_jobs_status_updated;
  `);
};
