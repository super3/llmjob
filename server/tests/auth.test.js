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

      await requireAuth(req, res, next);

      expect(verifyToken).toHaveBeenCalledWith('valid.jwt.token', expect.any(Object));
      expect(next).toHaveBeenCalled();
      expect(req.user).toEqual({ id: 'user_123' });
    });

    // The verified token's `sub` is the identity. A Clerk API call per request
    // only fetched email and username, which nothing reads.
    it('does not call the Clerk API', async () => {
      req.headers.authorization = 'Bearer valid.jwt.token';
      verifyToken.mockResolvedValue({ sub: 'user_123' });

      await requireAuth(req, res, next);

      expect(clerkClient.users.getUser).not.toHaveBeenCalled();
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
  });
});
