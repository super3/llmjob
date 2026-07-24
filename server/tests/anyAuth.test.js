const { anyAuth } = require('../src/middleware/anyAuth');
const ApiKeyService = require('../src/services/apiKeyService');
const { clerkClient, verifyToken } = require('@clerk/clerk-sdk-node');
const { createTestDb } = require('./helpers/pgmem');

// Clerk is mocked so the session branch runs without a network round-trip; the
// API-key branch runs against a real (in-memory) database.
jest.mock('@clerk/clerk-sdk-node', () => ({
  clerkClient: { users: { getUser: jest.fn() } },
  verifyToken: jest.fn()
}));

describe('anyAuth (Clerk session OR API key)', () => {
  let req, res, next, db;

  beforeEach(async () => {
    db = await createTestDb();
    req = { headers: {}, app: { locals: { db } } };
    res = { status: jest.fn().mockReturnThis(), json: jest.fn() };
    next = jest.fn();
    jest.clearAllMocks();
  });

  afterEach(async () => {
    if (db.end) await db.end();
  });

  it('authenticates an LLMJob API key and resolves its owner', async () => {
    const created = await new ApiKeyService(db).createKey('user_key_owner', 'sdk');
    req.headers.authorization = `Bearer ${created.key}`;

    await anyAuth(req, res, next);

    expect(next).toHaveBeenCalled();
    expect(req.user.id).toBe('user_key_owner');
    expect(verifyToken).not.toHaveBeenCalled(); // never treated as a session JWT
  });

  it('rejects an unknown API key', async () => {
    req.headers.authorization = 'Bearer lj-live-not-a-real-key';
    await anyAuth(req, res, next);
    expect(res.status).toHaveBeenCalledWith(401);
    expect(next).not.toHaveBeenCalled();
  });

  it('falls through to the Clerk session for a non-lj token', async () => {
    req.headers.authorization = 'Bearer some.jwt.token';
    verifyToken.mockResolvedValue({ sub: 'user_session' });
    clerkClient.users.getUser.mockResolvedValue({
      id: 'user_session', emailAddresses: [{ emailAddress: 'a@b.c' }], username: 'u'
    });

    await anyAuth(req, res, next);

    expect(next).toHaveBeenCalled();
    expect(req.user.id).toBe('user_session');
  });

  it('rejects a request with no bearer token at all', async () => {
    await anyAuth(req, res, next);
    expect(res.status).toHaveBeenCalledWith(401);
    expect(res.json).toHaveBeenCalledWith({ error: 'No authorization token provided' });
    expect(next).not.toHaveBeenCalled();
  });
});
