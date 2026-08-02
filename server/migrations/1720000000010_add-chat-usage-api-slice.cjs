/* eslint-disable camelcase */
// Splits out the slice of OpenRouter usage that the /v1 API gateway served.
//
// `chat_usage_totals` is every token we have bought from OpenRouter, and it is
// what the free-usage cap is measured against. Now that a `public` API key can
// ask the gateway for a hosted model, that spend has to land here too — or a key
// could drain the credit the cap exists to protect.
//
// But those same tokens are also billed to the key (`api_keys.usage`), and the
// network/chat pages add the two sums together for their headline "tokens
// served" figure. `api_total_tokens` records the overlap so the pages can
// subtract it and count each token exactly once. Existing rows default to 0:
// before this column, no API traffic had ever reached OpenRouter.
//
// ADD COLUMN IF NOT EXISTS is idempotent — a no-op on a fresh database where
// db.js's CHAT_SCHEMA already includes it.

exports.shorthands = undefined;

exports.up = (pgm) => {
  pgm.sql(`
    ALTER TABLE chat_usage_totals ADD COLUMN IF NOT EXISTS api_total_tokens bigint DEFAULT 0;
  `);
};

exports.down = (pgm) => {
  pgm.sql(`
    ALTER TABLE chat_usage_totals DROP COLUMN IF EXISTS api_total_tokens;
  `);
};
