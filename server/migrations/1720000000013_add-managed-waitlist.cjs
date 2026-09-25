/* eslint-disable camelcase */
// Adds the managed-mining waitlist (`managed_waitlist`) to databases created
// before it existed. The DDL is the canonical WAITLIST_SCHEMA from
// server/src/db.js (CREATE ... IF NOT EXISTS, so it is a no-op on a fresh
// database where the init migration already applied it).
const { WAITLIST_SCHEMA } = require('../src/db');

exports.shorthands = undefined;

exports.up = (pgm) => {
  pgm.sql(WAITLIST_SCHEMA);
};

exports.down = (pgm) => {
  pgm.sql('DROP TABLE IF EXISTS managed_waitlist CASCADE;');
};
