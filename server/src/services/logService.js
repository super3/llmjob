const crypto = require('crypto');

// Recent inference request logs, per user. Capped to the most recent entries so
// the table can't grow without bound.
const LOG_CAP = 200; // keep at most this many recent entries per user
const HOUR_MS = 60 * 60 * 1000;
const ACTIVITY_BUCKETS = 24; // hourly buckets for the dashboard chart

function rowToEntry(row) {
  return {
    id: row.id,
    timestamp: Number(row.ts),
    model: row.model,
    node: row.node,
    app: row.app,
    in: row.in_tokens,
    out: row.out_tokens,
    speed: row.speed,
    finish: row.finish,
    key: row.key
  };
}

class LogService {
  constructor(db) {
    this.db = db;
  }

  // Append a request log entry. Returns the stored entry.
  async recordLog(userId, entry) {
    const stored = {
      id: 'log_' + crypto.randomBytes(6).toString('hex'),
      timestamp: entry.timestamp !== undefined ? entry.timestamp : Date.now(),
      model: entry.model,
      node: entry.node,
      app: entry.app || 'api',
      in: entry.in || 0,
      out: entry.out || 0,
      speed: entry.speed || 0,
      finish: entry.finish || 'stop',
      key: entry.key || null
    };

    await this.db.query(
      `INSERT INTO request_logs (id, user_id, ts, model, node, app, in_tokens, out_tokens, speed, finish, key)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)`,
      [stored.id, userId, stored.timestamp, stored.model, stored.node, stored.app,
        stored.in, stored.out, stored.speed, stored.finish, stored.key]
    );

    // Trim oldest entries beyond the cap.
    await this.db.query(
      `DELETE FROM request_logs WHERE id IN (
         SELECT id FROM request_logs WHERE user_id = $1 ORDER BY ts DESC OFFSET $2)`,
      [userId, LOG_CAP]
    );

    return stored;
  }

  // Return recent entries, newest first, capped at `limit`.
  async getLogs(userId, limit = 50) {
    const r = await this.db.query(
      `SELECT id, ts, model, node, app, in_tokens, out_tokens, speed, finish, key
       FROM request_logs WHERE user_id = $1 ORDER BY ts DESC LIMIT $2`,
      [userId, limit]
    );
    return r.rows.map(rowToEntry);
  }

  // Bucket the last 24 hours of activity into hourly request counts. Index 0 is
  // the oldest hour, the last index is the current hour.
  async getActivity(userId, now = Date.now()) {
    const windowStart = now - ACTIVITY_BUCKETS * HOUR_MS;
    const r = await this.db.query(
      'SELECT ts FROM request_logs WHERE user_id = $1 AND ts >= $2 AND ts <= $3',
      [userId, windowStart, now]
    );

    const buckets = new Array(ACTIVITY_BUCKETS).fill(0);
    for (const row of r.rows) {
      let idx = Math.floor((Number(row.ts) - windowStart) / HOUR_MS);
      if (idx >= ACTIVITY_BUCKETS) idx = ACTIVITY_BUCKETS - 1;
      buckets[idx]++;
    }
    return buckets;
  }
}

module.exports = LogService;
