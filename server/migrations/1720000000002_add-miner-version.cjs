/* eslint-disable camelcase */
// Adds the reporting client's `version` to the `miners` table for databases
// created before this column existed. ADD COLUMN IF NOT EXISTS is idempotent — a
// no-op on a fresh database where MINERS_SCHEMA already includes it.

exports.shorthands = undefined;

exports.up = (pgm) => {
  pgm.sql('ALTER TABLE miners ADD COLUMN IF NOT EXISTS version text;');
};

exports.down = (pgm) => {
  pgm.sql('ALTER TABLE miners DROP COLUMN IF EXISTS version;');
};
