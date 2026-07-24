/* eslint-disable camelcase */
// Adds `llm_model` to the `miners` table — the local LLM a serving GPU runs, so
// the network board can show which model (if any) each card is serving. ADD
// COLUMN IF NOT EXISTS is idempotent — a no-op on a fresh database where
// MINERS_SCHEMA already includes it.

exports.shorthands = undefined;

exports.up = (pgm) => {
  pgm.sql('ALTER TABLE miners ADD COLUMN IF NOT EXISTS llm_model text;');
};

exports.down = (pgm) => {
  pgm.sql('ALTER TABLE miners DROP COLUMN IF EXISTS llm_model;');
};
