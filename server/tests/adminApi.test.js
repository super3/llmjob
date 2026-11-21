jest.mock('@clerk/clerk-sdk-node', () => ({ verifyToken: jest.fn() }));
const { verifyToken } = require('@clerk/clerk-sdk-node');
const { requireAdminApi, tokenMatches } = require('../src/middleware/adminApi');

describe('requireAdminApi', () => {
  let res, next, prevToken, prevIds;

  beforeEach(() => {
    res = { status: jest.fn().mockReturnThis(), json: jest.fn() };
    next = jest.fn();
    prevToken = process.env.CONTAINERS_ADMIN_TOKEN;
    prevIds = process.env.ADMIN_USER_IDS;
    jest.clearAllMocks();
    jest.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    if (prevToken === undefined) delete process.env.CONTAINERS_ADMIN_TOKEN;
    else process.env.CONTAINERS_ADMIN_TOKEN = prevToken;
    if (prevIds === undefined) delete process.env.ADMIN_USER_IDS;
    else process.env.ADMIN_USER_IDS = prevIds;
    jest.restoreAllMocks();
  });

  const req = (token) => ({ headers: token ? { authorization: 'Bearer ' + token } : {} });

  it('lets a script in with the admin token', async () => {
    process.env.CONTAINERS_ADMIN_TOKEN = 'secret-token';
    const r = req('secret-token');
    await requireAdminApi(r, res, next);
    expect(next).toHaveBeenCalled();
    expect(r.user).toEqual({ id: 'containers-admin-token' });
    expect(verifyToken).not.toHaveBeenCalled();
  });

  it('falls back to a Clerk admin when the token does not match', async () => {
    process.env.CONTAINERS_ADMIN_TOKEN = 'secret-token';
    process.env.ADMIN_USER_IDS = 'user_admin';
    verifyToken.mockResolvedValue({ sub: 'user_admin' });
    const r = req('a-clerk-jwt');
    await requireAdminApi(r, res, next);
    expect(next).toHaveBeenCalled();
    expect(r.user).toEqual({ id: 'user_admin' });
  });

  it('refuses a signed-in user who is not an admin', async () => {
    process.env.ADMIN_USER_IDS = 'user_admin';
    verifyToken.mockResolvedValue({ sub: 'someone' });
    await requireAdminApi(req('jwt'), res, next);
    expect(res.status).toHaveBeenCalledWith(403);
    expect(next).not.toHaveBeenCalled();
  });

  it('refuses a request with no token at all', async () => {
    await requireAdminApi(req(null), res, next);
    expect(res.status).toHaveBeenCalledWith(401);
  });

  it('never matches when no admin token is set, or on a length mismatch', () => {
    delete process.env.CONTAINERS_ADMIN_TOKEN;
    expect(tokenMatches('anything')).toBe(false);
    process.env.CONTAINERS_ADMIN_TOKEN = 'abc';
    expect(tokenMatches('abcd')).toBe(false);
    expect(tokenMatches('')).toBe(false);
    expect(tokenMatches('abc')).toBe(true);
  });
});
