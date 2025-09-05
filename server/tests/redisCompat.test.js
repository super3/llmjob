const { createRedisCompat } = require('../src/utils/redisCompat');

describe('redisCompat', () => {
  describe('get operation', () => {
    it('should handle promise-based get', async () => {
      const mockRedis = {
        get: jest.fn().mockResolvedValue('value')
      };
      
      const compat = createRedisCompat(mockRedis);
      const result = await compat.get('key');
      
      expect(result).toBe('value');
      expect(mockRedis.get).toHaveBeenCalledWith('key');
    });

    it('should handle callback-based get', async () => {
      const mockRedis = {
        get: jest.fn((key, callback) => {
          // Simulate non-promise return value
          if (callback) {
            callback(null, 'value');
          }
          return undefined; // Not a promise
        })
      };
      
      const compat = createRedisCompat(mockRedis);
      const result = await compat.get('key');
      
      expect(result).toBe('value');
    });

    it('should handle missing get method', async () => {
      const mockRedis = {};
      
      const compat = createRedisCompat(mockRedis);
      const result = await compat.get('key');
      
      expect(result).toBeNull();
    });
  });

  describe('set operation', () => {
    it('should handle promise-based set', async () => {
      const mockRedis = {
        set: jest.fn().mockResolvedValue('OK')
      };
      
      const compat = createRedisCompat(mockRedis);
      const result = await compat.set('key', 'value');
      
      expect(result).toBe('OK');
      expect(mockRedis.set).toHaveBeenCalledWith('key', 'value');
    });

    it('should handle callback-based set', async () => {
      const mockRedis = {
        set: jest.fn((key, value, callback) => {
          if (callback) {
            callback(null, 'OK');
          }
          return undefined; // Not a promise
        })
      };
      
      const compat = createRedisCompat(mockRedis);
      const result = await compat.set('key', 'value');
      
      expect(result).toBe('OK');
    });

    it('should handle callback-based set with null result', async () => {
      const mockRedis = {
        set: jest.fn((key, value, callback) => {
          if (callback) {
            callback(null, null);
          }
          return undefined; // Not a promise
        })
      };
      
      const compat = createRedisCompat(mockRedis);
      const result = await compat.set('key', 'value');
      
      expect(result).toBe('OK');
    });

    it('should handle missing set method', async () => {
      const mockRedis = {};
      
      const compat = createRedisCompat(mockRedis);
      const result = await compat.set('key', 'value');
      
      expect(result).toBe('OK');
    });
  });

  describe('sAdd operation', () => {
    it('should use sAdd when available', async () => {
      const mockRedis = {
        sAdd: jest.fn().mockResolvedValue(1)
      };
      
      const compat = createRedisCompat(mockRedis);
      const result = await compat.sAdd('key', 'member1', 'member2');
      
      expect(result).toBe(1);
      expect(mockRedis.sAdd).toHaveBeenCalledWith('key', 'member1', 'member2');
    });

    it('should fallback to sadd when sAdd not available', async () => {
      const mockRedis = {
        sadd: jest.fn().mockResolvedValue(1)
      };
      
      const compat = createRedisCompat(mockRedis);
      const result = await compat.sAdd('key', 'member');
      
      expect(result).toBe(1);
      expect(mockRedis.sadd).toHaveBeenCalledWith('key', 'member');
    });
  });

  describe('sMembers operation', () => {
    it('should use sMembers when available', async () => {
      const mockRedis = {
        sMembers: jest.fn().mockResolvedValue(['member1', 'member2'])
      };
      
      const compat = createRedisCompat(mockRedis);
      const result = await compat.sMembers('key');
      
      expect(result).toEqual(['member1', 'member2']);
      expect(mockRedis.sMembers).toHaveBeenCalledWith('key');
    });

    it('should fallback to smembers when sMembers not available', async () => {
      const mockRedis = {
        smembers: jest.fn().mockResolvedValue(['member1'])
      };
      
      const compat = createRedisCompat(mockRedis);
      const result = await compat.sMembers('key');
      
      expect(result).toEqual(['member1']);
      expect(mockRedis.smembers).toHaveBeenCalledWith('key');
    });
  });

  describe('setEx operation', () => {
    it('should use setEx when available', async () => {
      const mockRedis = {
        setEx: jest.fn().mockResolvedValue('OK')
      };
      
      const compat = createRedisCompat(mockRedis);
      const result = await compat.setEx('key', 60, 'value');
      
      expect(result).toBe('OK');
      expect(mockRedis.setEx).toHaveBeenCalledWith('key', 60, 'value');
    });

    it('should fallback to setex when setEx not available', async () => {
      const mockRedis = {
        setex: jest.fn().mockResolvedValue('OK')
      };
      
      const compat = createRedisCompat(mockRedis);
      const result = await compat.setEx('key', 60, 'value');
      
      expect(result).toBe('OK');
      expect(mockRedis.setex).toHaveBeenCalledWith('key', 60, 'value');
    });
  });

  describe('ttl operation', () => {
    it('should handle promise-based ttl', async () => {
      const mockRedis = {
        ttl: jest.fn().mockResolvedValue(300)
      };
      
      const compat = createRedisCompat(mockRedis);
      const result = await compat.ttl('key');
      
      expect(result).toBe(300);
      expect(mockRedis.ttl).toHaveBeenCalledWith('key');
    });

    it('should handle non-promise ttl', async () => {
      const mockRedis = {
        ttl: jest.fn(() => {
          return 100; // Return non-promise value
        })
      };
      
      const compat = createRedisCompat(mockRedis);
      const result = await compat.ttl('key');
      
      expect(result).toBe(100);
    });
  });

  describe('keys operation', () => {
    it('should handle promise-based keys', async () => {
      const mockRedis = {
        keys: jest.fn().mockResolvedValue(['key1', 'key2'])
      };
      
      const compat = createRedisCompat(mockRedis);
      const result = await compat.keys('pattern*');
      
      expect(result).toEqual(['key1', 'key2']);
      expect(mockRedis.keys).toHaveBeenCalledWith('pattern*');
    });

    it('should handle non-promise keys', async () => {
      const mockRedis = {
        keys: jest.fn(() => {
          return ['key1']; // Return non-promise value
        })
      };
      
      const compat = createRedisCompat(mockRedis);
      const result = await compat.keys('pattern*');
      
      expect(result).toEqual(['key1']);
    });
  });

});