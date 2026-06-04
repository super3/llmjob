const crypto = require('crypto');

// Recent inference request logs, per user. Backed by a sorted set scored by
// timestamp so we can read newest-first and cheaply trim old entries.
const USER_LOGS_PREFIX = 'user_logs:';
const LOG_CAP = 200; // keep at most this many recent entries per user
const HOUR_MS = 60 * 60 * 1000;
const ACTIVITY_BUCKETS = 24; // hourly buckets for the dashboard chart

class LogService {
  constructor(redis) {
    // Production passes the real redis v5 client; tests pass an equivalent
    // adapter. Either way it is used directly.
    this.redis = redis;
  }

  // Append a request log entry. Returns the stored entry.
  async recordLog(userId, entry) {
    const timestamp = entry.timestamp !== undefined ? entry.timestamp : Date.now();
    const stored = {
      id: 'log_' + crypto.randomBytes(6).toString('hex'),
      timestamp,
      model: entry.model,
      node: entry.node,
      app: entry.app || 'api',
      in: entry.in || 0,
      out: entry.out || 0,
      speed: entry.speed || 0,
      finish: entry.finish || 'stop',
      key: entry.key || null
    };

    const setKey = `${USER_LOGS_PREFIX}${userId}`;
    await this.redis.zAdd(setKey, { score: timestamp, value: JSON.stringify(stored) });

    // Trim oldest entries beyond the cap.
    const count = await this.redis.zCard(setKey);
    if (count > LOG_CAP) {
      const oldest = await this.redis.zRange(setKey, 0, count - LOG_CAP - 1);
      for (const member of oldest) {
        await this.redis.zRem(setKey, member);
      }
    }

    return stored;
  }

  // Return recent entries, newest first, capped at `limit`.
  async getLogs(userId, limit = 50) {
    const setKey = `${USER_LOGS_PREFIX}${userId}`;
    const raw = await this.redis.zRange(setKey, 0, -1);
    if (!raw || raw.length === 0) {
      return [];
    }

    const entries = raw.map((m) => JSON.parse(m));
    entries.sort((a, b) => b.timestamp - a.timestamp);
    return entries.slice(0, limit);
  }

  // Bucket the last 24 hours of activity into hourly request counts. Index 0 is
  // the oldest hour, the last index is the current hour.
  async getActivity(userId, now = Date.now()) {
    const setKey = `${USER_LOGS_PREFIX}${userId}`;
    const windowStart = now - ACTIVITY_BUCKETS * HOUR_MS;
    const raw = await this.redis.zRangeByScore(setKey, windowStart, now);

    const buckets = new Array(ACTIVITY_BUCKETS).fill(0);
    if (raw && raw.length > 0) {
      for (const member of raw) {
        const { timestamp } = JSON.parse(member);
        // zRangeByScore lower bound is inclusive, so timestamp >= windowStart
        // and idx is always >= 0. A timestamp exactly at `now` lands one past
        // the last bucket, so clamp it back in.
        let idx = Math.floor((timestamp - windowStart) / HOUR_MS);
        if (idx >= ACTIVITY_BUCKETS) idx = ACTIVITY_BUCKETS - 1;
        buckets[idx]++;
      }
    }

    return buckets;
  }
}

module.exports = LogService;
