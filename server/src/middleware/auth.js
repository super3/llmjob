const { clerkClient } = require('@clerk/clerk-sdk-node');

async function requireAuth(req, res, next) {
  try {
    // Get the token from the Authorization header
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return res.status(401).json({ error: 'No authorization token provided' });
    }

    const token = authHeader.split(' ')[1];
    
    // Verify the session token with Clerk
    try {
      const session = await clerkClient.sessions.verifySession('', token);
      
      // Get user details
      const user = await clerkClient.users.getUser(session.userId);
      
      // Attach user to request
      req.user = {
        id: user.id,
        email: user.emailAddresses[0]?.emailAddress,
        username: user.username
      };
      
      next();
    } catch (error) {
      console.error('Token verification failed:', error);
      return res.status(401).json({ error: 'Invalid or expired token' });
    }
  } catch (error) {
    console.error('Auth middleware error:', error);
    return res.status(500).json({ error: 'Authentication error' });
  }
}

module.exports = { requireAuth };