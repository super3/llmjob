/* eslint-disable camelcase */
// Per-card health and rig identity on the miner check-in: rejected shares,
// temperature, power draw and limit, clocks, fan, driver, OS, which client
// (GUI or CLI), how long the miner has run and since its last share, and the
// verified rig id. Stored for fleet diagnostics; the public board does not show
// them. ADD COLUMN IF NOT EXISTS is idempotent — a no-op on a fresh database
// whose SCHEMA already includes them. Existing rows keep NULL, which is what an
// older client that sends none of these reads as.

exports.shorthands = undefined;

const COLUMNS = [
  ['rejected', 'bigint'],
  ['temp_c', 'double precision'],
  ['power_w', 'double precision'],
  ['power_limit_w', 'double precision'],
  ['core_clock_mhz', 'integer'],
  ['mem_clock_mhz', 'integer'],
  ['fan_pct', 'integer'],
  ['driver', 'text'],
  ['os', 'text'],
  ['client', 'text'],
  ['uptime_sec', 'bigint'],
  ['last_share_sec', 'bigint'],
  ['rig_id', 'text'],
];

exports.COLUMNS = COLUMNS;

exports.up = (pgm) => {
  pgm.sql(COLUMNS.map(([name, type]) => `ALTER TABLE miners ADD COLUMN IF NOT EXISTS ${name} ${type};`).join('\n'));
};

exports.down = (pgm) => {
  pgm.sql(COLUMNS.map(([name]) => `ALTER TABLE miners DROP COLUMN IF EXISTS ${name};`).join('\n'));
};
