/* eslint-disable camelcase */
// Initial schema for the LLMJob backend. The canonical DDL lives in
// server/src/db.js (SCHEMA) so the same definition is applied both by
// node-pg-migrate in production and by the in-memory Postgres used in tests.
const { SCHEMA } = require('../server/src/db');

exports.shorthands = undefined;

exports.up = (pgm) => {
  pgm.sql(SCHEMA);
};

exports.down = (pgm) => {
  pgm.sql(
    'DROP TABLE IF EXISTS miners, job_chunks, jobs, node_join_tokens, request_logs, api_keys, nodes CASCADE;'
  );
};
