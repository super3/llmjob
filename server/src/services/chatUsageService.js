const crypto = require('crypto');

// Usage accounting for the free public web chat (the OpenRouter proxy). We keep
// two things and, by design, NEVER the prompt or reply text:
//
//   1. chat_requests   — one row per request with performance + token counts
//                        (latency, time-to-first-token, tok/s, in/out tokens).
//                        Capped to the most recent CHAT_LOG_CAP rows so it can't
//                        grow without bound.
//   2. chat_usage_totals — a single 'global' row of running sums, so lifetime
//                          free usage can be summed and capped even after the
//                          per-request rows are trimmed.
const CHAT_LOG_CAP = 200;
const TOTALS_ID = 'global';

function n(v) {
  const x = Number(v);
  return Number.isFinite(x) ? x : 0;
}

function rowToEntry(row) {
  return {
    id: row.id,
    timestamp: Number(row.ts),
    model: row.model,
    in: row.in_tokens,
    out: row.out_tokens,
    total: row.total_tokens,
    speed: row.speed,
    latencyMs: row.latency_ms,
    ttftMs: row.ttft_ms,
    finish: row.finish
  };
}

class ChatUsageService {
  constructor(db) {
    this.db = db;
  }

  // Record a completed generation. Writes a performance row (no prompt) and
  // folds the token counts into the running totals. Returns the stored entry.
  async recordUsage(entry) {
    const stored = {
      id: 'creq_' + crypto.randomBytes(6).toString('hex'),
      timestamp: entry.timestamp !== undefined ? entry.timestamp : Date.now(),
      model: entry.model || 'unknown',
      in: Math.max(0, Math.round(n(entry.inTokens))),
      out: Math.max(0, Math.round(n(entry.outTokens))),
      speed: n(entry.speed),
      latencyMs: Math.max(0, Math.round(n(entry.latencyMs))),
      ttftMs: Math.max(0, Math.round(n(entry.ttftMs))),
      finish: entry.finish || 'stop'
    };
    stored.total = stored.in + stored.out;

    await this.db.query(
      `INSERT INTO chat_requests (id, ts, model, in_tokens, out_tokens, total_tokens, speed, latency_ms, ttft_ms, finish)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)`,
      [stored.id, stored.timestamp, stored.model, stored.in, stored.out, stored.total,
        stored.speed, stored.latencyMs, stored.ttftMs, stored.finish]
    );

    // Trim performance rows beyond the cap (oldest first).
    await this.db.query(
      `DELETE FROM chat_requests WHERE id IN (
         SELECT id FROM chat_requests ORDER BY ts DESC OFFSET $1)`,
      [CHAT_LOG_CAP]
    );

    // Fold into the running totals (never trimmed — this is the lifetime sum).
    await this.db.query(
      `INSERT INTO chat_usage_totals (id, requests, in_tokens, out_tokens, total_tokens)
       VALUES ($1, 1, $2, $3, $4)
       ON CONFLICT (id) DO UPDATE SET
         requests = chat_usage_totals.requests + 1,
         in_tokens = chat_usage_totals.in_tokens + $2,
         out_tokens = chat_usage_totals.out_tokens + $3,
         total_tokens = chat_usage_totals.total_tokens + $4`,
      [TOTALS_ID, stored.in, stored.out, stored.total]
    );

    return stored;
  }

  // Lifetime running totals used for the free-usage cap and public display.
  async getTotals() {
    const r = await this.db.query(
      'SELECT requests, in_tokens, out_tokens, total_tokens FROM chat_usage_totals WHERE id = $1',
      [TOTALS_ID]
    );
    const row = r.rows[0];
    if (!row) return { requests: 0, inTokens: 0, outTokens: 0, totalTokens: 0 };
    return {
      requests: Number(row.requests),
      inTokens: Number(row.in_tokens),
      outTokens: Number(row.out_tokens),
      totalTokens: Number(row.total_tokens)
    };
  }

  // Recent performance rows, newest first, capped at `limit`.
  async getRecent(limit = 50) {
    const r = await this.db.query(
      `SELECT id, ts, model, in_tokens, out_tokens, total_tokens, speed, latency_ms, ttft_ms, finish
       FROM chat_requests ORDER BY ts DESC LIMIT $1`,
      [limit]
    );
    return r.rows.map(rowToEntry);
  }
}

module.exports = ChatUsageService;
module.exports.CHAT_LOG_CAP = CHAT_LOG_CAP;
