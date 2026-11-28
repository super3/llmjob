/* eslint-disable camelcase */
// Adds the LLM rewards tables (#331): llm_usage (one row per request a node
// served for the server), prl_prices (one PRL price per UTC day) and
// llm_payouts (what was paid by hand). The DDL is LLM_REWARDS_SCHEMA from
// server/src/db.js, all IF NOT EXISTS, so it is a no-op on a fresh database
// whose init schema already has them.
const { LLM_REWARDS_SCHEMA } = require('../src/db');

exports.shorthands = undefined;

exports.up = (pgm) => {
  pgm.sql(LLM_REWARDS_SCHEMA);
};

exports.down = (pgm) => {
  pgm.sql('DROP TABLE IF EXISTS llm_usage, prl_prices, llm_payouts CASCADE;');
};
