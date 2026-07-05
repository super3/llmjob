/* eslint-disable camelcase */
// Adds vram_used / vram_total to the `miners` table for databases created before
// these columns existed. ADD COLUMN IF NOT EXISTS is idempotent — a no-op on a
// fresh database where MINERS_SCHEMA already includes them.

exports.shorthands = undefined;

exports.up = (pgm) => {
  pgm.sql(`
    ALTER TABLE miners ADD COLUMN IF NOT EXISTS vram_used double precision;
    ALTER TABLE miners ADD COLUMN IF NOT EXISTS vram_total double precision;
  `);
};

exports.down = (pgm) => {
  pgm.sql(`
    ALTER TABLE miners DROP COLUMN IF EXISTS vram_used;
    ALTER TABLE miners DROP COLUMN IF EXISTS vram_total;
  `);
};
