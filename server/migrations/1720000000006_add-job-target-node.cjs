/* eslint-disable camelcase */
// Per-request node targeting. A caller can pin a /v1/chat/completions request to
// one specific node (to test whether that node serves and how fast), via the
// `jobs.target_node` column the poll query filters on — the same "promote a
// queryable column so the node poller can filter" pattern used for visibility.
// ADD COLUMN IF NOT EXISTS is idempotent — a no-op on a fresh database whose
// SCHEMA already includes it. Existing jobs keep NULL, which the poller treats as
// "any eligible node" (today's behaviour).

exports.shorthands = undefined;

exports.up = (pgm) => {
  pgm.sql(`
    ALTER TABLE jobs ADD COLUMN IF NOT EXISTS target_node text;
  `);
};

exports.down = (pgm) => {
  pgm.sql(`
    ALTER TABLE jobs DROP COLUMN IF EXISTS target_node;
  `);
};
