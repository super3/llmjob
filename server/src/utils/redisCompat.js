// Redis compatibility wrapper to handle both redis-mock (lowercase) and redis v5 (camelCase)
function createRedisCompat(redis) {
  return {
    // Basic operations
    get: async (key) => {
      if (typeof redis.get === 'function') {
        const result = redis.get(key);
        // Handle both Promise and callback-based APIs
        if (result && typeof result.then === 'function') {
          return result;
        }
        // For redis-mock callback-based API
        return new Promise((resolve) => {
          redis.get(key, (err, result) => resolve(result));
        });
      }
      return null;
    },
    
    set: async (key, value) => {
      if (typeof redis.set === 'function') {
        const result = redis.set(key, value);
        // Handle both Promise and callback-based APIs
        if (result && typeof result.then === 'function') {
          return result;
        }
        // For redis-mock callback-based API
        return new Promise((resolve) => {
          redis.set(key, value, (err, result) => resolve(result || 'OK'));
        });
      }
      return 'OK';
    },
    
    // Set operations - try camelCase first, fallback to lowercase.
    // The lowercase fallback supports both callback-based (real redis-mock)
    // and promise-based (patched/async) APIs, mirroring get/set above.
    sAdd: async (key, ...members) => {
      if (typeof redis.sAdd === 'function') {
        return redis.sAdd(key, ...members);
      }
      return new Promise((resolve, reject) => {
        const result = redis.sadd(key, ...members, (err, res) => {
          if (err) reject(err);
          else resolve(res || 0);
        });
        if (result && typeof result.then === 'function') {
          result.then((res) => resolve(res || 0), reject);
        }
      });
    },

    sMembers: async (key) => {
      if (typeof redis.sMembers === 'function') {
        return redis.sMembers(key);
      }
      return new Promise((resolve, reject) => {
        const result = redis.smembers(key, (err, res) => {
          if (err) reject(err);
          else resolve(res || []);
        });
        if (result && typeof result.then === 'function') {
          result.then((res) => resolve(res || []), reject);
        }
      });
    },

    sRem: async (key, ...members) => {
      if (typeof redis.sRem === 'function') {
        return redis.sRem(key, ...members);
      }
      return new Promise((resolve, reject) => {
        const result = redis.srem(key, ...members, (err, res) => {
          if (err) reject(err);
          else resolve(res || 0);
        });
        if (result && typeof result.then === 'function') {
          result.then((res) => resolve(res || 0), reject);
        }
      });
    },
    
    // Expiration operations
    setEx: async (key, seconds, value) => {
      if (typeof redis.setEx === 'function') {
        return redis.setEx(key, seconds, value);
      }
      return redis.setex(key, seconds, value);
    },
    
    ttl: async (key) => {
      /* istanbul ignore else: ttl has the same name in both Redis APIs, so the
         callback fallback is never reached. */
      if (typeof redis.ttl === 'function') {
        return redis.ttl(key);
      } else {
        return new Promise((resolve) => {
          redis.ttl(key, (err, result) => resolve(result || -1));
        });
      }
    },

    // Key operations
    keys: async (pattern) => {
      /* istanbul ignore else: keys has the same name in both Redis APIs, so the
         callback fallback is never reached. */
      if (typeof redis.keys === 'function') {
        return redis.keys(pattern);
      } else {
        return new Promise((resolve) => {
          redis.keys(pattern, (err, result) => resolve(result || []));
        });
      }
    }
  };
}

module.exports = { createRedisCompat };