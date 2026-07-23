/* eslint-disable camelcase */
// Adds the free web-chat usage tables (`chat_requests`, `chat_usage_totals`) to
// databases created before they existed. The DDL is the canonical CHAT_SCHEMA
// from server/src/db.js (CREATE ... IF NOT EXISTS, so it is a no-op on a fresh
// database where the init migration already applied it). No prompts are stored —
// only performance/token metrics and the running usage totals.
const { CHAT_SCHEMA } = require('../src/db');

exports.shorthands = undefined;

exports.up = (pgm) => {
  pgm.sql(CHAT_SCHEMA);
};

exports.down = (pgm) => {
  pgm.sql('DROP TABLE IF EXISTS chat_requests, chat_usage_totals CASCADE;');
};
