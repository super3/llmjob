const { requireAdmin } = require('../src/middleware/admin');

describe('requireAdmin', () => {
  let res, next, prev;

  beforeEach(() => {
    res = { status: jest.fn().mockReturnThis(), json: jest.fn() };
    next = jest.fn();
    prev = process.env.ADMIN_USER_IDS;
  });

  afterEach(() => {
    process.env.ADMIN_USER_IDS = prev;
  });

  it('calls next for a user in the allow-list', () => {
    process.env.ADMIN_USER_IDS = 'a, admin_1 ,b';
    const req = { user: { id: 'admin_1' } };

    requireAdmin(req, res, next);

    expect(next).toHaveBeenCalled();
    expect(res.status).not.toHaveBeenCalled();
  });

  it('rejects a user not in the allow-list', () => {
    process.env.ADMIN_USER_IDS = 'admin_1';
    const req = { user: { id: 'someone_else' } };

    requireAdmin(req, res, next);

    expect(res.status).toHaveBeenCalledWith(403);
    expect(res.json).toHaveBeenCalledWith({ error: 'Admin access required' });
    expect(next).not.toHaveBeenCalled();
  });

  it('rejects when there is no authenticated user', () => {
    process.env.ADMIN_USER_IDS = 'admin_1';

    requireAdmin({}, res, next);

    expect(res.status).toHaveBeenCalledWith(403);
    expect(next).not.toHaveBeenCalled();
  });

  it('is closed by default when ADMIN_USER_IDS is unset', () => {
    delete process.env.ADMIN_USER_IDS;
    const req = { user: { id: 'anyone' } };

    requireAdmin(req, res, next);

    expect(res.status).toHaveBeenCalledWith(403);
    expect(next).not.toHaveBeenCalled();
  });
});
