// What a miner row's GPUs mine. NULL is Pearl, which is every row from before
// this column and every row a client too old to send it reports. 'quantus' is
// a rig mining Quantus (the CLI's --coin quantus, or Quantus picked in the
// app): its address is a Quantus one and its hashrate is in H/s, so the board
// keeps it out of Pearl's total. ADD COLUMN IF NOT EXISTS is idempotent: a
// no-op on a fresh database whose MINERS_SCHEMA already includes it.

exports.shorthands = undefined;

exports.up = (pgm) => {
  pgm.sql('ALTER TABLE miners ADD COLUMN IF NOT EXISTS coin text;');
};

exports.down = (pgm) => {
  pgm.sql('ALTER TABLE miners DROP COLUMN IF EXISTS coin;');
};
