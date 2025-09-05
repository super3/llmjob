const { createRedisCompat } = require('../utils/redisCompat');

/**
 * Base Repository class for Redis operations
 * Provides common CRUD operations and Redis utilities
 */
class BaseRepository {
  constructor(redis, keyPrefix = '') {
    this.redis = this.setupRedisCompat(redis);
    this.keyPrefix = keyPrefix;
  }

  /**
   * Setup Redis compatibility layer with all needed operations
   */
  setupRedisCompat(redis) {
    const compat = createRedisCompat(redis);
    
    // Add sorted set operations
    compat.zAdd = this.createZAddCompat(redis);
    compat.zRem = this.createZRemCompat(redis);
    compat.zRange = this.createZRangeCompat(redis);
    compat.zCard = this.createZCardCompat(redis);
    compat.zRangeByScore = this.createZRangeByScoreCompat(redis);
    
    // Add hash operations
    compat.hSet = this.createHSetCompat(redis);
    compat.hGetAll = this.createHGetAllCompat(redis);
    compat.hGet = this.createHGetCompat(redis);
    compat.hDel = this.createHDelCompat(redis);
    
    // Add list operations
    compat.lPush = this.createLPushCompat(redis);
    compat.lRange = this.createLRangeCompat(redis);
    compat.lLen = this.createLLenCompat(redis);
    
    // Add other operations
    compat.del = this.createDelCompat(redis);
    compat.exists = this.createExistsCompat(redis);
    compat.expire = this.createExpireCompat(redis);
    compat.sRem = this.createSRemCompat(redis);
    
    return compat;
  }

  // Sorted Set Operations
  createZAddCompat(redis) {
    return async (key, ...args) => {
      if (typeof redis.zAdd === 'function') {
        if (args.length === 1 && typeof args[0] === 'object' && 'member' in args[0]) {
          const { score, member } = args[0];
          return redis.zAdd(key, [{ score, value: member }]);
        }
        return redis.zAdd(key, ...args);
      }
      if (args.length === 1 && typeof args[0] === 'object') {
        const { score, member } = args[0];
        return redis.zadd(key, score, member);
      }
      return redis.zadd(key, ...args);
    };
  }

  createZRemCompat(redis) {
    return async (key, member) => {
      if (typeof redis.zRem === 'function') {
        return redis.zRem(key, member);
      }
      return redis.zrem(key, member);
    };
  }

  createZRangeCompat(redis) {
    return async (key, start, stop) => {
      if (typeof redis.zRange === 'function') {
        return redis.zRange(key, start, stop);
      }
      return new Promise((resolve) => {
        redis.zrange(key, start, stop, (err, result) => resolve(result || []));
      });
    };
  }

  createZCardCompat(redis) {
    return async (key) => {
      if (typeof redis.zCard === 'function') {
        return redis.zCard(key);
      }
      return new Promise((resolve) => {
        redis.zcard(key, (err, result) => resolve(result || 0));
      });
    };
  }

  createZRangeByScoreCompat(redis) {
    return async (key, min, max) => {
      if (typeof redis.zRangeByScore === 'function') {
        return redis.zRangeByScore(key, min, max);
      }
      return new Promise((resolve) => {
        redis.zrangebyscore(key, min, max, (err, result) => resolve(result || []));
      });
    };
  }

  // Hash Operations
  createHSetCompat(redis) {
    return async (key, field, value) => {
      if (typeof redis.hSet === 'function') {
        return redis.hSet(key, field, value);
      }
      return new Promise((resolve) => {
        redis.hset(key, field, value, (err, result) => resolve(result));
      });
    };
  }

  createHGetAllCompat(redis) {
    return async (key) => {
      if (typeof redis.hGetAll === 'function') {
        return redis.hGetAll(key);
      }
      return new Promise((resolve) => {
        redis.hgetall(key, (err, result) => resolve(result || {}));
      });
    };
  }

  createHGetCompat(redis) {
    return async (key, field) => {
      if (typeof redis.hGet === 'function') {
        return redis.hGet(key, field);
      }
      return new Promise((resolve) => {
        redis.hget(key, field, (err, result) => resolve(result));
      });
    };
  }

  createHDelCompat(redis) {
    return async (key, field) => {
      if (typeof redis.hDel === 'function') {
        return redis.hDel(key, field);
      }
      return new Promise((resolve) => {
        redis.hdel(key, field, (err, result) => resolve(result));
      });
    };
  }

  // List Operations
  createLPushCompat(redis) {
    return async (key, ...values) => {
      if (typeof redis.lPush === 'function') {
        return redis.lPush(key, ...values);
      }
      return new Promise((resolve) => {
        redis.lpush(key, ...values, (err, result) => resolve(result));
      });
    };
  }

  createLRangeCompat(redis) {
    return async (key, start, stop) => {
      if (typeof redis.lRange === 'function') {
        return redis.lRange(key, start, stop);
      }
      return new Promise((resolve) => {
        redis.lrange(key, start, stop, (err, result) => resolve(result || []));
      });
    };
  }

  createLLenCompat(redis) {
    return async (key) => {
      if (typeof redis.lLen === 'function') {
        return redis.lLen(key);
      }
      return new Promise((resolve) => {
        redis.llen(key, (err, result) => resolve(result || 0));
      });
    };
  }

  // Other Operations
  createDelCompat(redis) {
    return async (key) => {
      if (typeof redis.del === 'function') {
        return redis.del(key);
      }
      return new Promise((resolve) => {
        redis.del(key, (err, result) => resolve(result));
      });
    };
  }

  createExistsCompat(redis) {
    return async (key) => {
      if (typeof redis.exists === 'function') {
        return redis.exists(key);
      }
      return new Promise((resolve) => {
        redis.exists(key, (err, result) => resolve(result === 1));
      });
    };
  }

  createExpireCompat(redis) {
    return async (key, seconds) => {
      if (typeof redis.expire === 'function') {
        return redis.expire(key, seconds);
      }
      return new Promise((resolve) => {
        redis.expire(key, seconds, (err, result) => resolve(result));
      });
    };
  }

  createSRemCompat(redis) {
    return async (key, member) => {
      if (typeof redis.sRem === 'function') {
        return redis.sRem(key, member);
      }
      return redis.srem(key, member);
    };
  }

  // Helper methods
  getKey(id) {
    return `${this.keyPrefix}${id}`;
  }

  // Common CRUD operations
  async get(id) {
    const key = this.getKey(id);
    const data = await this.redis.get(key);
    return data ? JSON.parse(data) : null;
  }

  async set(id, data, ttl = null) {
    const key = this.getKey(id);
    const serialized = JSON.stringify(data);
    
    if (ttl) {
      return await this.redis.setEx(key, ttl, serialized);
    }
    return await this.redis.set(key, serialized);
  }

  async delete(id) {
    const key = this.getKey(id);
    return await this.redis.del(key);
  }

  async exists(id) {
    const key = this.getKey(id);
    return await this.redis.exists(key);
  }

  // Hash operations
  async hGet(id, field) {
    const key = this.getKey(id);
    return await this.redis.hGet(key, field);
  }

  async hSet(id, field, value) {
    const key = this.getKey(id);
    return await this.redis.hSet(key, field, value);
  }

  async hGetAll(id) {
    const key = this.getKey(id);
    return await this.redis.hGetAll(key);
  }

  async hDel(id, field) {
    const key = this.getKey(id);
    return await this.redis.hDel(key, field);
  }

  // Set operations
  async sAdd(id, member) {
    const key = this.getKey(id);
    return await this.redis.sAdd(key, member);
  }

  async sMembers(id) {
    const key = this.getKey(id);
    return await this.redis.sMembers(key);
  }

  async sRem(id, member) {
    const key = this.getKey(id);
    return await this.redis.sRem(key, member);
  }

  // Direct set operations (for custom keys)
  async sAddDirect(key, member) {
    if (typeof this.redis.sAdd === 'function') {
      return await this.redis.sAdd(key, member);
    }
    return new Promise((resolve) => {
      this.redis.sadd(key, member, (err, result) => resolve(result));
    });
  }

  async sMembersDirect(key) {
    if (typeof this.redis.sMembers === 'function') {
      return await this.redis.sMembers(key);
    }
    return new Promise((resolve) => {
      this.redis.smembers(key, (err, result) => resolve(result || []));
    });
  }

  async sRemDirect(key, member) {
    if (typeof this.redis.sRem === 'function') {
      return await this.redis.sRem(key, member);
    }
    return new Promise((resolve) => {
      this.redis.srem(key, member, (err, result) => resolve(result));
    });
  }

  // Sorted set operations
  async zAdd(id, score, member) {
    const key = this.getKey(id);
    return await this.redis.zAdd(key, { score, member });
  }

  async zRange(id, start, stop) {
    const key = this.getKey(id);
    return await this.redis.zRange(key, start, stop);
  }

  async zRem(id, member) {
    const key = this.getKey(id);
    return await this.redis.zRem(key, member);
  }

  async zCard(id) {
    const key = this.getKey(id);
    return await this.redis.zCard(key);
  }

  // List operations
  async lPush(id, ...values) {
    const key = this.getKey(id);
    return await this.redis.lPush(key, ...values);
  }

  async lRange(id, start, stop) {
    const key = this.getKey(id);
    return await this.redis.lRange(key, start, stop);
  }

  async lLen(id) {
    const key = this.getKey(id);
    return await this.redis.lLen(key);
  }

  // TTL operations
  async expire(id, seconds) {
    const key = this.getKey(id);
    return await this.redis.expire(key, seconds);
  }

  async ttl(id) {
    const key = this.getKey(id);
    return await this.redis.ttl(key);
  }

  // Pattern matching
  async keys(pattern) {
    const fullPattern = `${this.keyPrefix}${pattern}`;
    return await this.redis.keys(fullPattern);
  }
}

module.exports = BaseRepository;