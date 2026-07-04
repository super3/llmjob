/* eslint-disable camelcase */
// Adds the `miners` table to databases created before it existed. The DDL is the
// canonical MINERS_SCHEMA from server/src/db.js (CREATE ... IF NOT EXISTS, so it
// is a no-op on a fresh database where the init migration already applied it).
const { MINERS_SCHEMA } = require('../server/src/db');

exports.shorthands = undefined;

exports.up = (pgm) => {
  pgm.sql(MINERS_SCHEMA);
};

exports.down = (pgm) => {
  pgm.sql('DROP TABLE IF EXISTS miners CASCADE;');
};
