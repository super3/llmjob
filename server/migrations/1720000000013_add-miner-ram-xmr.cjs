/* eslint-disable camelcase */
// The machine-level numbers a miner ping now carries: the host's system RAM
// (ram_used / ram_total, MB) and the hashrate of its Monero CPU miner
// (xmr_hashrate, H/s). The network board shows the RAM; the Monero figure is
// stored and served now so the board can add it later without another client
// release. Every card row of a host carries the same values (they belong to
// the machine, not a card), so the board takes them once per host.
// ADD COLUMN IF NOT EXISTS is idempotent: a no-op on a fresh database whose
// MINERS_SCHEMA already includes them. Existing rows keep NULL, which reads as
// "unknown", the same as a client too old to send them.

exports.shorthands = undefined;

exports.up = (pgm) => {
  pgm.sql(`
    ALTER TABLE miners ADD COLUMN IF NOT EXISTS ram_used double precision;
    ALTER TABLE miners ADD COLUMN IF NOT EXISTS ram_total double precision;
    ALTER TABLE miners ADD COLUMN IF NOT EXISTS xmr_hashrate double precision;
  `);
};

exports.down = (pgm) => {
  pgm.sql(`
    ALTER TABLE miners DROP COLUMN IF EXISTS ram_used;
    ALTER TABLE miners DROP COLUMN IF EXISTS ram_total;
    ALTER TABLE miners DROP COLUMN IF EXISTS xmr_hashrate;
  `);
};
