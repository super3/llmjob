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
    
    // Set operations - try camelCase first, fallback to lowercase with callback
    sAdd: async (key, ...members) => {
      if (typeof redis.sAdd === 'function') {
        return redis.sAdd(key, ...members);
      }
      // For redis-mock callback-based API
      return new Promise((resolve) => {
        redis.sadd(key, ...members, (err, result) => resolve(result || 0));
      });
    },
    
    sMembers: async (key) => {
      if (typeof redis.sMembers === 'function') {
        return redis.sMembers(key);
      }
      // For redis-mock callback-based API
      return new Promise((resolve) => {
        redis.smembers(key, (err, result) => resolve(result || []));
      });
    },
    
    sRem: async (key, ...members) => {
      if (typeof redis.sRem === 'function') {
        return redis.sRem(key, ...members);
      }
      // For redis-mock callback-based API
      return new Promise((resolve) => {
        redis.srem(key, ...members, (err, result) => resolve(result || 0));
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
      if (typeof redis.ttl === 'function') {
        return redis.ttl(key);
      }
      // For redis-mock compatibility
      return new Promise((resolve) => {
        redis.ttl(key, (err, result) => resolve(result || -1));
      });
    },
    
    // Key operations
    keys: async (pattern) => {
      if (typeof redis.keys === 'function') {
        return redis.keys(pattern);
      }
      // For redis-mock compatibility
      return new Promise((resolve) => {
        redis.keys(pattern, (err, result) => resolve(result || []));
      });
    }
  };
}

module.exports = { createRedisCompat };