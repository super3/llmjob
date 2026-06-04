// A camelCase (Redis v5 style) client that returns Promises, implemented by
// delegating to an underlying redis-mock (lowercase, callback-based) client.
// This lets the same repository/service tests exercise the camelCase code
// paths in the Redis compatibility layers, which redis-mock alone cannot.
const redisMock = require('redis-mock');

function promisify(fn, thisArg, ...args) {
  return new Promise((resolve, reject) => {
    fn.call(thisArg, ...args, (err, res) => {
      if (err) reject(err);
      else resolve(res);
    });
  });
}

function createCamelClient() {
  const rm = redisMock.createClient();

  const client = {
    // Basic string operations
    get: (key) => promisify(rm.get, rm, key),
    set: (key, value) => promisify(rm.set, rm, key, value),
    setEx: (key, seconds, value) => promisify(rm.setex, rm, key, seconds, value),
    del: (key) => promisify(rm.del, rm, key),
    exists: (key) => promisify(rm.exists, rm, key),
    expire: (key, seconds) => promisify(rm.expire, rm, key, seconds),
    ttl: (key) => promisify(rm.ttl, rm, key),
    keys: (pattern) => promisify(rm.keys, rm, pattern),

    // Set operations
    sAdd: (key, ...members) => promisify(rm.sadd, rm, key, ...members),
    sMembers: (key) => promisify(rm.smembers, rm, key),
    sRem: (key, ...members) => promisify(rm.srem, rm, key, ...members),

    // Sorted set operations
    zAdd: (key, arg) => {
      const entries = Array.isArray(arg) ? arg : [arg];
      return entries.reduce(
        (p, { score, value, member }) =>
          p.then(() => promisify(rm.zadd, rm, key, score, value !== undefined ? value : member)),
        Promise.resolve()
      );
    },
    zRem: (key, member) => promisify(rm.zrem, rm, key, member),
    zRange: (key, start, stop) => promisify(rm.zrange, rm, key, start, stop),
    zCard: (key) => promisify(rm.zcard, rm, key),
    zRangeByScore: (key, min, max) => promisify(rm.zrangebyscore, rm, key, min, max),

    // Hash operations
    hSet: (key, field, value) => promisify(rm.hset, rm, key, field, value),
    hGetAll: (key) => promisify(rm.hgetall, rm, key),
    hGet: (key, field) => promisify(rm.hget, rm, key, field),
    hDel: (key, field) => promisify(rm.hdel, rm, key, field),

    // List operations
    lPush: (key, ...values) => promisify(rm.lpush, rm, key, ...values),
    lRange: (key, start, stop) => promisify(rm.lrange, rm, key, start, stop),
    lLen: (key) => promisify(rm.llen, rm, key),

    // Lifecycle
    quit: () => rm.quit(),
    flushall: (cb) => rm.flushall(cb),
  };

  return client;
}

module.exports = { createCamelClient };
