// The rig's compute mode and what its local LLM is doing, two short codes a
// miner ping now carries. `mode` is the owner's setting: auto, llm or mine.
// `llm` is the LLM's state: on (running), off (not wanted), vram (wanted, but
// the card has too little VRAM), cpu (wanted, but held back by the weak-CPU,
// low-RAM or low-disk check) or error (wanted, but it failed to start,
// crashed, was too slow or slowed mining down). They belong to the machine, so
// every card row of a host carries the same values. ADD COLUMN IF NOT EXISTS
// is idempotent: a no-op on a fresh database whose MINERS_SCHEMA already
// includes them. Existing rows keep NULL, "unknown", the same as a client too
// old to send them.

exports.shorthands = undefined;

exports.up = (pgm) => {
  pgm.sql(`
    ALTER TABLE miners ADD COLUMN IF NOT EXISTS mode text;
    ALTER TABLE miners ADD COLUMN IF NOT EXISTS llm text;
  `);
};

exports.down = (pgm) => {
  pgm.sql(`
    ALTER TABLE miners DROP COLUMN IF EXISTS mode;
    ALTER TABLE miners DROP COLUMN IF EXISTS llm;
  `);
};
