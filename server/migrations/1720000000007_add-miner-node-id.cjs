/* eslint-disable camelcase */
// Let the network board tell "this rig is running an LLM" from "this rig actually
// serves the cluster". The miner ping is anonymous telemetry keyed on
// address+worker; the node system is a separate signed identity keyed on node_id,
// and nothing linked the two — so a machine advertising a model looked identical
// to one polling for jobs. `miners.node_id` carries the machine's node id on the
// ping, but only while it is armed to serve, so the board can mark those rows.
// ADD COLUMN IF NOT EXISTS is idempotent — a no-op on a fresh database whose
// SCHEMA already includes it. Existing rows keep NULL (advertising only), which is
// today's behaviour for every client too old to send it.

exports.shorthands = undefined;

exports.up = (pgm) => {
  pgm.sql(`
    ALTER TABLE miners ADD COLUMN IF NOT EXISTS node_id text;
  `);
};

exports.down = (pgm) => {
  pgm.sql(`
    ALTER TABLE miners DROP COLUMN IF EXISTS node_id;
  `);
};
