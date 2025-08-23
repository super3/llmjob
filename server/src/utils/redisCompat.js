// Redis compatibility wrapper to handle both redis-mock (lowercase) and redis v5 (camelCase)
function createRedisCompat(redis) {
  return {
    // Basic operations
    get: (key) => redis.get(key),
    set: (key, value) => redis.set(key, value),
    
    // Set operations - try camelCase first, fallback to lowercase
    sAdd: async (key, ...members) => {
      if (typeof redis.sAdd === 'function') {
        return redis.sAdd(key, ...members);
      }
      return redis.sadd(key, ...members);
    },
    
    sMembers: async (key) => {
      if (typeof redis.sMembers === 'function') {
        return redis.sMembers(key);
      }
      return redis.smembers(key);
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