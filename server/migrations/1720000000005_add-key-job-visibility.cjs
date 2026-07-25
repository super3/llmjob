/* eslint-disable camelcase */
// Per-API-key request routing. `api_keys.visibility` is 'public' (a request may
// be served by any node on the network) or 'private' (only the key owner's own
// nodes). Each job records the visibility it was created with in
// `jobs.visibility`, so the node poller can filter. ADD COLUMN IF NOT EXISTS is
// idempotent — a no-op on a fresh database where the SCHEMA already includes them.
// Existing keys default to 'public' (today's behaviour); existing jobs keep NULL,
// which the poller treats as public.

exports.shorthands = undefined;

exports.up = (pgm) => {
  pgm.sql(`
    ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS visibility text DEFAULT 'public';
    ALTER TABLE jobs ADD COLUMN IF NOT EXISTS visibility text;
  `);
};

exports.down = (pgm) => {
  pgm.sql(`
    ALTER TABLE api_keys DROP COLUMN IF EXISTS visibility;
    ALTER TABLE jobs DROP COLUMN IF EXISTS visibility;
  `);
};
