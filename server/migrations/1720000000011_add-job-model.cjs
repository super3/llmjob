/* eslint-disable camelcase */
// Route a request to a node running the model it asked for.
//
// `model` has always been stored INSIDE jobs.data and never read on the way out:
// the poll query had no column to match on, so a caller naming a model got
// whichever node polled next running whatever it had loaded. Promoting it to a
// column is the same "make it queryable so the poller can filter" pattern used
// for visibility, target_node and max_tokens.
//
// NULL means "any eligible node", which is what every existing row and every
// request that does not name a live network model gets — so this is inert until
// a caller opts in by naming one.
//
// ADD COLUMN IF NOT EXISTS is idempotent: a no-op on a fresh database whose
// schema already includes it.

exports.shorthands = undefined;

exports.up = (pgm) => {
  pgm.sql(`
    ALTER TABLE jobs ADD COLUMN IF NOT EXISTS model text;
  `);
};

exports.down = (pgm) => {
  pgm.sql(`
    ALTER TABLE jobs DROP COLUMN IF EXISTS model;
  `);
};
