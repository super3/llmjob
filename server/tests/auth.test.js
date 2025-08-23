const { requireAuth } = require('../src/middleware/auth');
const { clerkClient } = require('@clerk/clerk-sdk-node');

// Mock Clerk
jest.mock('@clerk/clerk-sdk-node', () => ({
  clerkClient: {
    sessions: {
      getSession: jest.fn()
    },
    users: {
      getUser: jest.fn()
    }
  }
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
    it('should authenticate valid token with session ID', async () => {
      // Create a mock JWT token with session ID
      const payload = { sid: 'sess_123', sub: 'user_123' };
      const mockToken = `header.${Buffer.from(JSON.stringify(payload)).toString('base64')}.signature`;
      req.headers.authorization = `Bearer ${mockToken}`;
      
      clerkClient.sessions.getSession.mockResolvedValue({
        userId: 'user_123'
      });
      
      clerkClient.users.getUser.mockResolvedValue({
        id: 'user_123',
        emailAddresses: [{ emailAddress: 'test@example.com' }],
        username: 'testuser'
      });

      await requireAuth(req, res, next);

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

    it('should handle token verification failure', async () => {
      const payload = { sid: 'sess_invalid' };
      const mockToken = `header.${Buffer.from(JSON.stringify(payload)).toString('base64')}.signature`;
      req.headers.authorization = `Bearer ${mockToken}`;
      
      clerkClient.sessions.getSession.mockRejectedValue(
        new Error('Invalid token')
      );

      await requireAuth(req, res, next);

      expect(res.status).toHaveBeenCalledWith(401);
      expect(res.json).toHaveBeenCalledWith({
        error: 'Invalid or expired token'
      });
      expect(next).not.toHaveBeenCalled();
    });

    it('should handle user fetch failure', async () => {
      const payload = { sid: 'sess_123', sub: 'user_123' };
      const mockToken = `header.${Buffer.from(JSON.stringify(payload)).toString('base64')}.signature`;
      req.headers.authorization = `Bearer ${mockToken}`;
      
      clerkClient.sessions.getSession.mockResolvedValue({
        userId: 'user_123'
      });
      
      clerkClient.users.getUser.mockRejectedValue(
        new Error('User not found')
      );

      await requireAuth(req, res, next);

      expect(res.status).toHaveBeenCalledWith(401);
      expect(res.json).toHaveBeenCalledWith({
        error: 'Invalid or expired token'
      });
      expect(next).not.toHaveBeenCalled();
    });

    it('should handle unexpected errors', async () => {
      req.headers.authorization = 'Bearer token';
      
      // Mock an unexpected error by making req.headers.authorization throw
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

    it('should handle user without email addresses', async () => {
      const payload = { sid: 'sess_123', sub: 'user_123' };
      const mockToken = `header.${Buffer.from(JSON.stringify(payload)).toString('base64')}.signature`;
      req.headers.authorization = `Bearer ${mockToken}`;
      
      clerkClient.sessions.getSession.mockResolvedValue({
        userId: 'user_123'
      });
      
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

    it('should authenticate token with only user ID (no session)', async () => {
      const payload = { sub: 'user_123' }; // No sid
      const mockToken = `header.${Buffer.from(JSON.stringify(payload)).toString('base64')}.signature`;
      req.headers.authorization = `Bearer ${mockToken}`;
      
      clerkClient.users.getUser.mockResolvedValue({
        id: 'user_123',
        emailAddresses: [{ emailAddress: 'test@example.com' }],
        username: 'testuser'
      });

      await requireAuth(req, res, next);

      expect(next).toHaveBeenCalled();
      expect(req.user).toEqual({
        id: 'user_123',
        email: 'test@example.com',
        username: 'testuser'
      });
      expect(clerkClient.sessions.getSession).not.toHaveBeenCalled();
    });

    it('should reject token with invalid format', async () => {
      req.headers.authorization = 'Bearer invalid.token'; // Only 2 parts
      
      await requireAuth(req, res, next);
      
      expect(res.status).toHaveBeenCalledWith(401);
      expect(res.json).toHaveBeenCalledWith({
        error: 'Invalid or expired token'
      });
      expect(next).not.toHaveBeenCalled();
    });
  });
});