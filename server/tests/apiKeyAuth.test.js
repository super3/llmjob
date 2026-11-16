// The API-key middleware in front of /v1/*. These cases were once covered
// through POST /api/usage, which has been removed.
const { createTestDb } = require('./helpers/pgmem');
const ApiKeyService = require('../src/services/apiKeyService');
const { apiKeyAuth } = require('../src/middleware/apiKeyAuth');

describe('apiKeyAuth', () => {
  let db, res, next;

  const run = (authorization) => {
    const req = { headers: authorization ? { authorization } : {}, app: { locals: { db } } };
    return apiKeyAuth(req, res, next).then(() => req);
  };

  beforeEach(async () => {
    db = await createTestDb();
    res = { status: jest.fn().mockReturnThis(), json: jest.fn() };
    next = jest.fn();
  });

  afterEach(async () => {
    if (db.end) await db.end();
  });

  it('attaches the key and user for a valid key', async () => {
    const { key } = await new ApiKeyService(db).createKey('user-1', 'home');
    const req = await run('Bearer ' + key);
    expect(next).toHaveBeenCalled();
    expect(req.apiKey.name).toBe('home');
    expect(req.user).toEqual({ id: 'user-1' });
  });

  it('rejects a missing bearer token', async () => {
    await run();
    expect(res.status).toHaveBeenCalledWith(401);
    expect(next).not.toHaveBeenCalled();
  });

  it('rejects a non-LLMJob token', async () => {
    await run('Bearer sk-someother');
    expect(res.status).toHaveBeenCalledWith(401);
  });

  it('rejects an unknown key', async () => {
    await run('Bearer lj-live-unknown');
    expect(res.status).toHaveBeenCalledWith(401);
  });

  it('returns 500 when key lookup fails', async () => {
    db.query = jest.fn().mockRejectedValue(new Error('boom'));
    const spy = jest.spyOn(console, 'error').mockImplementation(() => {});
    await run('Bearer lj-live-x');
    spy.mockRestore();
    expect(res.status).toHaveBeenCalledWith(500);
  });
});
