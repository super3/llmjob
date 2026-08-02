/* eslint-disable camelcase */
// Server-measured node speed, and the job column the assignment filter reads.
//
// `nodes.tps` already exists but is whatever the node reports on its ping. A
// number that decides which jobs a node is offered can't be self-reported, so
// these are computed at the server: tokens produced over the wall time the
// server itself observed. Kept in separate columns rather than overwriting
// `tps` so the dashboard can still show what the node claims alongside what we
// measured.
//
// `jobs.max_tokens` is promoted out of `data` for the same reason `target_node`
// and `visibility` were: the poll query has to filter on it.
//
// ADD COLUMN IF NOT EXISTS is idempotent — a no-op on a fresh database whose
// SCHEMA already includes these. Existing rows keep NULL, which every reader
// treats as "not measured yet" (served permissively until a benchmark lands).

exports.shorthands = undefined;

exports.up = (pgm) => {
  pgm.sql(`
    ALTER TABLE nodes ADD COLUMN IF NOT EXISTS measured_tps double precision;
    ALTER TABLE nodes ADD COLUMN IF NOT EXISTS speed_samples integer DEFAULT 0;
    ALTER TABLE nodes ADD COLUMN IF NOT EXISTS speed_at bigint;
    ALTER TABLE jobs  ADD COLUMN IF NOT EXISTS max_tokens integer;
  `);
};

exports.down = (pgm) => {
  pgm.sql(`
    ALTER TABLE nodes DROP COLUMN IF EXISTS measured_tps;
    ALTER TABLE nodes DROP COLUMN IF EXISTS speed_samples;
    ALTER TABLE nodes DROP COLUMN IF EXISTS speed_at;
    ALTER TABLE jobs  DROP COLUMN IF EXISTS max_tokens;
  `);
};
