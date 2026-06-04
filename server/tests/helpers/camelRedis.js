// A faithful camelCase (Redis v5 style) client that returns Promises, backed by
// redis-mock (which only speaks the lowercase, callback-based API).
//
// Production uses the real `redis` v5 client, whose API is camelCase and
// promise-based, so the services call the client directly with no compatibility
// shim. Tests run against this adapter so they exercise the same API shape the
// services use in production, including the `set` options (`NX`/`EX`) and the
// v5 `zAdd` argument format.
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
    // Strings / keys
    get: (key) => promisify(rm.get, rm, key),
    set: async (key, value, options) => {
      // Emulate the subset of v5 `set` options the services rely on. The real
      // redis v5 client supports these natively.
      if (options && options.NX) {
        const existing = await promisify(rm.get, rm, key);
        if (existing) return null;
      }
      await promisify(rm.set, rm, key, value);
      if (options && options.EX) {
        await promisify(rm.expire, rm, key, options.EX);
      }
      return 'OK';
    },
    setEx: (key, seconds, value) => promisify(rm.setex, rm, key, seconds, value),
    del: (key) => promisify(rm.del, rm, key),
    expire: (key, seconds) => promisify(rm.expire, rm, key, seconds),
    ttl: (key) => promisify(rm.ttl, rm, key),
    keys: (pattern) => promisify(rm.keys, rm, pattern),

    // Sets
    sAdd: (key, ...members) => promisify(rm.sadd, rm, key, ...members),
    sMembers: (key) => promisify(rm.smembers, rm, key),
    sRem: (key, ...members) => promisify(rm.srem, rm, key, ...members),

    // Sorted sets
    zAdd: (key, arg) => {
      const entries = Array.isArray(arg) ? arg : [arg];
      return entries.reduce(
        (p, { score, value }) => p.then(() => promisify(rm.zadd, rm, key, score, value)),
        Promise.resolve()
      );
    },
    zRem: (key, member) => promisify(rm.zrem, rm, key, member),
    zRange: (key, start, stop) => promisify(rm.zrange, rm, key, start, stop),
    zCard: (key) => promisify(rm.zcard, rm, key),
    zRangeByScore: (key, min, max) => promisify(rm.zrangebyscore, rm, key, min, max),

    // Lifecycle
    quit: () => rm.quit(),
    flushall: (cb) => (cb ? rm.flushall(cb) : promisify(rm.flushall, rm)),
  };

  return client;
}

module.exports = { createCamelClient };
