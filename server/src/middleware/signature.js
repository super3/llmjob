const nacl = require('tweetnacl');
const naclUtil = require('tweetnacl-util');

function verifySignature(req, res, next) {
  try {
    const { publicKey, signature, timestamp, nodeId } = req.body;
    
    if (!publicKey || !signature || !timestamp || !nodeId) {
      return res.status(400).json({ error: 'Missing required fields' });
    }
    
    // Check timestamp is within 5 minutes
    const now = Date.now();
    const timeDiff = Math.abs(now - timestamp);
    if (timeDiff > 5 * 60 * 1000) { // 5 minutes
      return res.status(401).json({ error: 'Timestamp too old or too far in future' });
    }
    
    // Create message to verify
    const message = `${nodeId}:${timestamp}`;
    
    // Verify signature
    try {
      const publicKeyBytes = naclUtil.decodeBase64(publicKey);
      const signatureBytes = naclUtil.decodeBase64(signature);
      const messageBytes = naclUtil.decodeUTF8(message);
      
      const isValid = nacl.sign.detached.verify(
        messageBytes,
        signatureBytes,
        publicKeyBytes
      );
      
      if (!isValid) {
        return res.status(401).json({ error: 'Invalid signature' });
      }
      
      // Attach verified data to request
      req.verifiedNode = {
        publicKey,
        nodeId,
        timestamp
      };
      
      next();
    } catch (error) {
      console.error('Signature verification error:', error);
      return res.status(401).json({ error: 'Invalid signature format' });
    }
  } catch (error) {
    console.error('Signature middleware error:', error);
    return res.status(500).json({ error: 'Signature verification error' });
  }
}

module.exports = { verifySignature };