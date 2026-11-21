/* eslint-disable camelcase */
// Adds the rented-container tables (issue #318) to databases created before
// they existed: container_jobs, container_runs, container_minutes (the minutes
// ledger owners are paid on), container_logs and container_rigs. The DDL is
// CONTAINERS_SCHEMA from server/src/db.js, all CREATE ... IF NOT EXISTS, so it
// is a no-op on a fresh database where the init migration already applied it.
//
// Numbered 015, not 014: 1720000000014_add-miner-coin is already taken by the
// open release branch, and two files must never share a number.
const { CONTAINERS_SCHEMA } = require('../src/db');

exports.shorthands = undefined;

exports.up = (pgm) => {
  pgm.sql(CONTAINERS_SCHEMA);
};

exports.down = (pgm) => {
  pgm.sql('DROP TABLE IF EXISTS container_jobs, container_runs, container_minutes, container_logs, container_rigs CASCADE;');
};
