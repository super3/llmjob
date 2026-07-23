const { requireAuth } = require('../src/middleware/auth');
const { clerkClient, verifyToken } = require('@clerk/clerk-sdk-node');

// Mock Clerk. `verifyToken` checks the JWT signature against Clerk's JWKS in
// production; here it is mocked so tests exercise the middleware without a
// network round-trip.
jest.mock('@clerk/clerk-sdk-node', () => ({
  clerkClient: {
    users: {
      getUser: jest.fn()
    }
  },
  verifyToken: jest.fn()
}));

describe('Auth Middleware', () => {
  let req, res, next;

  beforeEach(() => {
    req = {
      headers: {},
      user: null
    };
    res = {
      status: jest.fn().mockReturnThis(),
      json: jest.fn()
    };
    next = jest.fn();
    jest.clearAllMocks();
  });

  describe('requireAuth', () => {
    it('should authenticate a token with a valid signature', async () => {
      req.headers.authorization = 'Bearer valid.jwt.token';

      verifyToken.mockResolvedValue({ sub: 'user_123' });
      clerkClient.users.getUser.mockResolvedValue({
        id: 'user_123',
        emailAddresses: [{ emailAddress: 'test@example.com' }],
        username: 'testuser'
      });

      await requireAuth(req, res, next);

      expect(verifyToken).toHaveBeenCalledWith('valid.jwt.token', expect.any(Object));
      expect(clerkClient.users.getUser).toHaveBeenCalledWith('user_123');
      expect(next).toHaveBeenCalled();
      expect(req.user).toEqual({
        id: 'user_123',
        email: 'test@example.com',
        username: 'testuser'
      });
    });

    it('should reject request without authorization header', async () => {
      await requireAuth(req, res, next);

      expect(res.status).toHaveBeenCalledWith(401);
      expect(res.json).toHaveBeenCalledWith({
        error: 'No authorization token provided'
      });
      expect(next).not.toHaveBeenCalled();
    });

    it('should reject request with invalid authorization format', async () => {
      req.headers.authorization = 'InvalidFormat token';

      await requireAuth(req, res, next);

      expect(res.status).toHaveBeenCalledWith(401);
      expect(res.json).toHaveBeenCalledWith({
        error: 'No authorization token provided'
      });
      expect(next).not.toHaveBeenCalled();
    });

    it('should reject a token that fails signature verification', async () => {
      req.headers.authorization = 'Bearer forged.jwt.token';
      verifyToken.mockRejectedValue(new Error('Invalid signature'));

      await requireAuth(req, res, next);

      expect(clerkClient.users.getUser).not.toHaveBeenCalled();
      expect(res.status).toHaveBeenCalledWith(401);
      expect(res.json).toHaveBeenCalledWith({
        error: 'Invalid or expired token'
      });
      expect(next).not.toHaveBeenCalled();
    });

    it('should handle user fetch failure', async () => {
      req.headers.authorization = 'Bearer valid.jwt.token';
      verifyToken.mockResolvedValue({ sub: 'user_123' });
      clerkClient.users.getUser.mockRejectedValue(new Error('User not found'));

      await requireAuth(req, res, next);

      expect(res.status).toHaveBeenCalledWith(401);
      expect(res.json).toHaveBeenCalledWith({
        error: 'Invalid or expired token'
      });
      expect(next).not.toHaveBeenCalled();
    });

    it('should handle unexpected errors', async () => {
      // Make reading the authorization header throw, tripping the outer catch.
      Object.defineProperty(req.headers, 'authorization', {
        get: () => { throw new Error('Unexpected error'); }
      });

      await requireAuth(req, res, next);

      expect(res.status).toHaveBeenCalledWith(500);
      expect(res.json).toHaveBeenCalledWith({
        error: 'Authentication error'
      });
      expect(next).not.toHaveBeenCalled();
    });

    it('should handle a user without email addresses', async () => {
      req.headers.authorization = 'Bearer valid.jwt.token';
      verifyToken.mockResolvedValue({ sub: 'user_123' });
      clerkClient.users.getUser.mockResolvedValue({
        id: 'user_123',
        emailAddresses: [],
        username: 'testuser'
      });

      await requireAuth(req, res, next);

      expect(next).toHaveBeenCalled();
      expect(req.user).toEqual({
        id: 'user_123',
        email: undefined,
        username: 'testuser'
      });
    });
  });
});
