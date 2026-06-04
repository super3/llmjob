const NodeTokenService = require('../src/services/nodeTokenService');
const { createTestDb } = require('./helpers/pgmem');

describe('NodeTokenService', () => {
  let db;
  let service;

  beforeEach(async () => {
    db = await createTestDb();
    service = new NodeTokenService(db);
  });

  afterEach(async () => {
    if (db.end) await db.end();
  });

  describe('generateToken', () => {
    it('produces a prefixed random token', () => {
      const t = NodeTokenService.generateToken();
      expect(t).toMatch(/^ljn_[0-9a-f]{36}$/);
    });
  });

  describe('getOrCreateToken', () => {
    it('creates a token on first use', async () => {
      const rec = await service.getOrCreateToken('user1');
      expect(rec.token).toMatch(/^ljn_/);
      expect(rec.createdAt).toEqual(expect.any(Number));
    });

    it('returns the same token on subsequent calls', async () => {
      const a = await service.getOrCreateToken('user1');
      const b = await service.getOrCreateToken('user1');
      expect(b.token).toBe(a.token);
    });
  });

  describe('verifyToken', () => {
    it('resolves a valid token to its owner', async () => {
      const rec = await service.getOrCreateToken('user1');
      expect(await service.verifyToken(rec.token)).toBe('user1');
    });

    it('returns null for an unknown token', async () => {
      expect(await service.verifyToken('ljn_nope')).toBeNull();
    });

    it('returns null when no token is supplied', async () => {
      expect(await service.verifyToken()).toBeNull();
    });
  });

  describe('rotateToken', () => {
    it('issues a new token and invalidates the previous one', async () => {
      const first = await service.getOrCreateToken('user1');
      const second = await service.rotateToken('user1');

      expect(second.token).not.toBe(first.token);
      expect(await service.verifyToken(first.token)).toBeNull();
      expect(await service.verifyToken(second.token)).toBe('user1');
    });

    it('works even if the user has no existing token', async () => {
      const rec = await service.rotateToken('fresh-user');
      expect(await service.verifyToken(rec.token)).toBe('fresh-user');
    });
  });
});
