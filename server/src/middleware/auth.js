const { clerkClient } = require('@clerk/clerk-sdk-node');

async function requireAuth(req, res, next) {
  try {
    // Get the token from the Authorization header
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return res.status(401).json({ error: 'No authorization token provided' });
    }

    const sessionToken = authHeader.split(' ')[1];
    
    // Verify the session token with Clerk
    try {
      // For Clerk v4, we need to decode the JWT to get the session ID
      // The token from getToken() is a JWT that contains the session info
      const tokenParts = sessionToken.split('.');
      if (tokenParts.length !== 3) {
        throw new Error('Invalid token format');
      }
      
      // Decode the JWT payload
      const payload = JSON.parse(Buffer.from(tokenParts[1], 'base64').toString());
      
      // Get the session using the session ID from the token
      const sessionId = payload.sid || payload.sess;
      if (sessionId) {
        const session = await clerkClient.sessions.getSession(sessionId);
        
        // Get user details
        const user = await clerkClient.users.getUser(session.userId);
        
        // Attach user to request
        req.user = {
          id: user.id,
          email: user.emailAddresses[0]?.emailAddress,
          username: user.username
        };
        
        next();
      } else {
        // If no session ID, try to get user directly from sub claim
        const userId = payload.sub;
        const user = await clerkClient.users.getUser(userId);
        
        // Attach user to request
        req.user = {
          id: user.id,
          email: user.emailAddresses[0]?.emailAddress,
          username: user.username
        };
        
        next();
      }
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