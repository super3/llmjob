/* eslint-disable camelcase */
// Per-attempt fencing token for a job's assignment lock.
//
// `lock_node` alone identifies the MACHINE, not the attempt, and one machine
// deliberately runs several job workers under a single node id (a multi-GPU rig
// serves the model from every card that fits it, and the GUI and CLI share one
// node.json). So after a heartbeat-stale requeue hands a job to a sibling
// worker, the original worker's late /chunks, /complete or /fail still matched
// `lock_node` and were accepted — letting a stale attempt kill or corrupt the
// job its sibling was running correctly.
//
// The token is minted fresh on every assignment and cleared whenever the lock is
// released or the job is requeued, so a stale worker presents a token that no
// longer matches and is rejected. ADD COLUMN IF NOT EXISTS is idempotent — a
// no-op on a fresh database whose SCHEMA already includes it. Existing rows keep
// NULL, which the service treats as "no token to check" (today's behaviour), so
// jobs already in flight during the deploy finish normally.

exports.shorthands = undefined;

exports.up = (pgm) => {
  pgm.sql(`
    ALTER TABLE jobs ADD COLUMN IF NOT EXISTS lock_token text;
  `);
};

exports.down = (pgm) => {
  pgm.sql(`
    ALTER TABLE jobs DROP COLUMN IF EXISTS lock_token;
  `);
};
