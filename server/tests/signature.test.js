const { verifySignature } = require('../src/middleware/signature');
const nacl = require('tweetnacl');
const naclUtil = require('tweetnacl-util');

describe('Signature Middleware', () => {
  let req, res, next;
  let testKeypair;
  let testPublicKey;
  let testNodeId;

  beforeEach(() => {
    // Generate test keypair
    testKeypair = nacl.sign.keyPair();
    testPublicKey = naclUtil.encodeBase64(testKeypair.publicKey);
    testNodeId = 'test123';

    req = {
      body: {},
      verifiedNode: null
    };
    res = {
      status: jest.fn().mockReturnThis(),
      json: jest.fn()
    };
    next = jest.fn();
    jest.clearAllMocks();
  });

  describe('verifySignature', () => {
    it('should verify valid signature', () => {
      const timestamp = Date.now();
      const message = `${testNodeId}:${timestamp}`;
      const messageBytes = naclUtil.decodeUTF8(message);
      const signature = nacl.sign.detached(messageBytes, testKeypair.secretKey);
      const signatureBase64 = naclUtil.encodeBase64(signature);

      req.body = {
        publicKey: testPublicKey,
        signature: signatureBase64,
        timestamp,
        nodeId: testNodeId
      };

      verifySignature(req, res, next);

      expect(next).toHaveBeenCalled();
      expect(req.verifiedNode).toEqual({
        publicKey: testPublicKey,
        nodeId: testNodeId,
        timestamp
      });
    });

    it('should reject request with missing fields', () => {
      req.body = {
        publicKey: testPublicKey,
        // Missing signature, timestamp, nodeId
      };

      verifySignature(req, res, next);

      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({
        error: 'Missing required fields'
      });
      expect(next).not.toHaveBeenCalled();
    });

    it('should reject old timestamp', () => {
      const timestamp = Date.now() - (10 * 60 * 1000); // 10 minutes ago
      const message = `${testNodeId}:${timestamp}`;
      const messageBytes = naclUtil.decodeUTF8(message);
      const signature = nacl.sign.detached(messageBytes, testKeypair.secretKey);
      const signatureBase64 = naclUtil.encodeBase64(signature);

      req.body = {
        publicKey: testPublicKey,
        signature: signatureBase64,
        timestamp,
        nodeId: testNodeId
      };

      verifySignature(req, res, next);

      expect(res.status).toHaveBeenCalledWith(401);
      expect(res.json).toHaveBeenCalledWith({
        error: 'Timestamp too old or too far in future'
      });
      expect(next).not.toHaveBeenCalled();
    });

    it('should reject future timestamp', () => {
      const timestamp = Date.now() + (10 * 60 * 1000); // 10 minutes in future
      const message = `${testNodeId}:${timestamp}`;
      const messageBytes = naclUtil.decodeUTF8(message);
      const signature = nacl.sign.detached(messageBytes, testKeypair.secretKey);
      const signatureBase64 = naclUtil.encodeBase64(signature);

      req.body = {
        publicKey: testPublicKey,
        signature: signatureBase64,
        timestamp,
        nodeId: testNodeId
      };

      verifySignature(req, res, next);

      expect(res.status).toHaveBeenCalledWith(401);
      expect(res.json).toHaveBeenCalledWith({
        error: 'Timestamp too old or too far in future'
      });
      expect(next).not.toHaveBeenCalled();
    });

    it('should reject invalid signature', () => {
      const timestamp = Date.now();
      // Create signature for wrong message
      const wrongMessage = `wrong:${timestamp}`;
      const messageBytes = naclUtil.decodeUTF8(wrongMessage);
      const signature = nacl.sign.detached(messageBytes, testKeypair.secretKey);
      const signatureBase64 = naclUtil.encodeBase64(signature);

      req.body = {
        publicKey: testPublicKey,
        signature: signatureBase64,
        timestamp,
        nodeId: testNodeId
      };

      verifySignature(req, res, next);

      expect(res.status).toHaveBeenCalledWith(401);
      expect(res.json).toHaveBeenCalledWith({
        error: 'Invalid signature'
      });
      expect(next).not.toHaveBeenCalled();
    });

    it('should handle invalid base64 in signature', () => {
      const timestamp = Date.now();

      req.body = {
        publicKey: testPublicKey,
        signature: 'invalid_base64!!!',
        timestamp,
        nodeId: testNodeId
      };

      verifySignature(req, res, next);

      expect(res.status).toHaveBeenCalledWith(401);
      expect(res.json).toHaveBeenCalledWith({
        error: 'Invalid signature format'
      });
      expect(next).not.toHaveBeenCalled();
    });

    it('should handle invalid base64 in public key', () => {
      const timestamp = Date.now();
      const message = `${testNodeId}:${timestamp}`;
      const messageBytes = naclUtil.decodeUTF8(message);
      const signature = nacl.sign.detached(messageBytes, testKeypair.secretKey);
      const signatureBase64 = naclUtil.encodeBase64(signature);

      req.body = {
        publicKey: 'invalid_base64!!!',
        signature: signatureBase64,
        timestamp,
        nodeId: testNodeId
      };

      verifySignature(req, res, next);

      expect(res.status).toHaveBeenCalledWith(401);
      expect(res.json).toHaveBeenCalledWith({
        error: 'Invalid signature format'
      });
      expect(next).not.toHaveBeenCalled();
    });

    it('should handle unexpected errors', () => {
      // Mock an unexpected error by making req.body throw
      Object.defineProperty(req, 'body', {
        get: () => { throw new Error('Unexpected error'); }
      });

      verifySignature(req, res, next);

      expect(res.status).toHaveBeenCalledWith(500);
      expect(res.json).toHaveBeenCalledWith({
        error: 'Signature verification error'
      });
      expect(next).not.toHaveBeenCalled();
    });
  });
  // Math.abs(now - "abc") is NaN, and `NaN > window` is false — so before this a
  // non-numeric timestamp skipped the freshness check entirely, letting a node
  // mint a signature that never expired.
  describe('timestamp coercion', () => {
    const sign = (timestamp) => {
      const message = `${testNodeId}:${timestamp}`;
      return naclUtil.encodeBase64(
        nacl.sign.detached(naclUtil.decodeUTF8(message), testKeypair.secretKey)
      );
    };

    it('rejects a non-numeric timestamp instead of skipping the freshness check', () => {
      req.body = {
        publicKey: testPublicKey, signature: sign('not-a-number'),
        timestamp: 'not-a-number', nodeId: testNodeId,
      };
      verifySignature(req, res, next);
      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({ error: 'Invalid timestamp' });
      expect(next).not.toHaveBeenCalled();
    });

    it('rejects non-finite timestamps', () => {
      for (const bad of [Infinity, -Infinity]) {
        jest.clearAllMocks();
        req.body = {
          publicKey: testPublicKey, signature: sign(bad),
          timestamp: bad, nodeId: testNodeId,
        };
        verifySignature(req, res, next);
        expect(res.status).toHaveBeenCalledWith(400);
        expect(next).not.toHaveBeenCalled();
      }
    });

    // JSON has no integer type distinction, so a client sending the timestamp as
    // a string is plausible and must still work — coercion, not rejection.
    it('accepts a fresh numeric-string timestamp', () => {
      const ts = Date.now();
      req.body = {
        publicKey: testPublicKey, signature: sign(String(ts)),
        timestamp: String(ts), nodeId: testNodeId,
      };
      verifySignature(req, res, next);
      expect(next).toHaveBeenCalled();
    });
  });
});
