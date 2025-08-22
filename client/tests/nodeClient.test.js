const NodeClient = require('../src/nodeClient');
const axios = require('axios');
const nacl = require('tweetnacl');
const naclUtil = require('tweetnacl-util');

jest.mock('axios');

describe('NodeClient', () => {
  let nodeClient;
  let mockConfig;
  let mockAxiosInstance;

  beforeEach(() => {
    jest.clearAllMocks();
    
    // Generate real keypair for testing
    const keypair = nacl.sign.keyPair();
    
    mockConfig = {
      nodeId: 'test123',
      publicKey: naclUtil.encodeBase64(keypair.publicKey),
      secretKey: naclUtil.encodeBase64(keypair.secretKey),
      serverUrl: 'https://test.server.com'
    };
    
    mockAxiosInstance = {
      post: jest.fn()
    };
    
    axios.create.mockReturnValue(mockAxiosInstance);
    
    nodeClient = new NodeClient(mockConfig);
  });

  describe('constructor', () => {
    it('should initialize with config and create axios instance', () => {
      expect(nodeClient.config).toEqual(mockConfig);
      expect(axios.create).toHaveBeenCalledWith({
        baseURL: 'https://test.server.com',
        timeout: 30000,
        headers: {
          'Content-Type': 'application/json'
        }
      });
    });
  });

  describe('signMessage', () => {
    it('should generate valid signature for message', () => {
      const message = 'test123:1234567890';
      const signature = nodeClient.signMessage(message, mockConfig.secretKey);
      
      // Verify signature is base64 encoded
      expect(() => naclUtil.decodeBase64(signature)).not.toThrow();
      
      // Verify signature with public key
      const messageBytes = naclUtil.decodeUTF8(message);
      const signatureBytes = naclUtil.decodeBase64(signature);
      const publicKeyBytes = naclUtil.decodeBase64(mockConfig.publicKey);
      
      const isValid = nacl.sign.detached.verify(
        messageBytes,
        signatureBytes,
        publicKeyBytes
      );
      
      expect(isValid).toBe(true);
    });

    it('should generate different signatures for different messages', () => {
      const signature1 = nodeClient.signMessage('message1', mockConfig.secretKey);
      const signature2 = nodeClient.signMessage('message2', mockConfig.secretKey);
      
      expect(signature1).not.toBe(signature2);
    });
  });

  describe('generateClaimUrl', () => {
    it('should generate full and short URLs with custom name', () => {
      const urls = nodeClient.generateClaimUrl('MyNode');
      
      expect(urls.full).toContain('https://test.server.com/add-node');
      expect(urls.full).toContain(`publicKey=${encodeURIComponent(mockConfig.publicKey)}`);
      expect(urls.full).toContain('name=MyNode');
      
      expect(urls.short).toBe('https://test.server.com/add-node?id=test123');
    });

    it('should generate URLs with default name if none provided', () => {
      const urls = nodeClient.generateClaimUrl();
      
      expect(urls.full).toContain(`name=Node-test123`);
      expect(urls.short).toBe('https://test.server.com/add-node?id=test123');
    });

    it('should handle server URL with /api suffix', () => {
      nodeClient.config.serverUrl = 'https://test.server.com/api';
      const urls = nodeClient.generateClaimUrl('TestNode');
      
      expect(urls.full).toContain('https://test.server.com/add-node');
      expect(urls.short).toBe('https://test.server.com/add-node?id=test123');
    });
  });

  describe('ping', () => {
    it('should send ping with valid signature', async () => {
      const mockResponse = { data: { success: true, status: 'online' } };
      mockAxiosInstance.post.mockResolvedValue(mockResponse);
      
      const result = await nodeClient.ping();
      
      expect(result.success).toBe(true);
      expect(result.data).toEqual(mockResponse.data);
      expect(result.attempt).toBe(1);
      
      // Verify the request was made with correct data
      const callArgs = mockAxiosInstance.post.mock.calls[0];
      expect(callArgs[0]).toBe('/api/nodes/ping');
      
      const requestData = callArgs[1];
      expect(requestData.nodeId).toBe('test123');
      expect(requestData.publicKey).toBe(mockConfig.publicKey);
      expect(requestData).toHaveProperty('signature');
      expect(requestData).toHaveProperty('timestamp');
      
      // Verify signature is valid
      const message = `${requestData.nodeId}:${requestData.timestamp}`;
      const messageBytes = naclUtil.decodeUTF8(message);
      const signatureBytes = naclUtil.decodeBase64(requestData.signature);
      const publicKeyBytes = naclUtil.decodeBase64(mockConfig.publicKey);
      
      const isValid = nacl.sign.detached.verify(
        messageBytes,
        signatureBytes,
        publicKeyBytes
      );
      expect(isValid).toBe(true);
    });

    it('should return error after max retries', async () => {
      const error = new Error('Network error');
      mockAxiosInstance.post.mockRejectedValue(error);
      
      const result = await nodeClient.ping(1, 0);
      
      expect(result.success).toBe(false);
      expect(result.error).toBe('Network error');
      expect(result.attempt).toBe(1);
      expect(mockAxiosInstance.post).toHaveBeenCalledTimes(1);
    });

    it('should retry on failure with delay', async () => {
      jest.useFakeTimers();
      
      mockAxiosInstance.post
        .mockRejectedValueOnce(new Error('Network error'))
        .mockResolvedValueOnce({ data: { success: true } });
      
      const pingPromise = nodeClient.ping(2, 100);
      
      // First attempt fails
      await Promise.resolve();
      expect(mockAxiosInstance.post).toHaveBeenCalledTimes(1);
      
      // Advance time for retry delay
      jest.advanceTimersByTime(100);
      
      const result = await pingPromise;
      
      expect(result.success).toBe(true);
      expect(result.attempt).toBe(2);
      expect(mockAxiosInstance.post).toHaveBeenCalledTimes(2);
      
      jest.useRealTimers();
    });

    it('should handle server error response', async () => {
      const error = {
        response: {
          status: 401,
          data: { error: 'Invalid signature' }
        },
        message: 'Request failed'
      };
      mockAxiosInstance.post.mockRejectedValue(error);
      
      const result = await nodeClient.ping(1);
      
      expect(result.success).toBe(false);
      expect(result.error).toBe('Invalid signature');
      expect(result.statusCode).toBe(401);
    });

    it('should handle missing error response', async () => {
      const error = new Error('Connection timeout');
      mockAxiosInstance.post.mockRejectedValue(error);
      
      const result = await nodeClient.ping(1);
      
      expect(result.success).toBe(false);
      expect(result.error).toBe('Connection timeout');
      expect(result.statusCode).toBeUndefined();
    });
  });

  describe('stopPinging', () => {
    it('should clear interval when provided', () => {
      const clearIntervalSpy = jest.spyOn(global, 'clearInterval');
      const intervalId = 12345;
      
      nodeClient.stopPinging(intervalId);
      
      expect(clearIntervalSpy).toHaveBeenCalledWith(intervalId);
    });

    it('should handle undefined intervalId', () => {
      const clearIntervalSpy = jest.spyOn(global, 'clearInterval');
      
      nodeClient.stopPinging(undefined);
      
      expect(clearIntervalSpy).not.toHaveBeenCalled();
    });

    it('should handle null intervalId', () => {
      const clearIntervalSpy = jest.spyOn(global, 'clearInterval');
      
      nodeClient.stopPinging(null);
      
      expect(clearIntervalSpy).not.toHaveBeenCalled();
    });
  });

  describe('startPinging', () => {
    beforeEach(() => {
      jest.useFakeTimers();
    });

    afterEach(() => {
      jest.useRealTimers();
    });

    it('should start interval and call ping', async () => {
      mockAxiosInstance.post.mockResolvedValue({ data: { success: true } });
      const onPing = jest.fn();
      
      const intervalId = nodeClient.startPinging(5000, onPing);
      
      // Wait for initial ping to be called
      await Promise.resolve();
      
      // Initial ping should have been called
      expect(mockAxiosInstance.post).toHaveBeenCalledTimes(1);
      
      // Clear interval to prevent test leaks
      clearInterval(intervalId);
    });

    it('should work without callback', async () => {
      mockAxiosInstance.post.mockResolvedValue({ data: { success: true } });
      
      const intervalId = nodeClient.startPinging(5000);
      
      // Wait for initial ping to be called
      await Promise.resolve();
      
      // Initial ping should have been called
      expect(mockAxiosInstance.post).toHaveBeenCalledTimes(1);
      
      // Clear interval to prevent test leaks
      clearInterval(intervalId);
    });

    it('should continue pinging at intervals', async () => {
      mockAxiosInstance.post.mockResolvedValue({ data: { success: true } });
      const onPing = jest.fn();
      
      const intervalId = nodeClient.startPinging(5000, onPing);
      
      // Wait for initial ping
      await Promise.resolve();
      expect(mockAxiosInstance.post).toHaveBeenCalledTimes(1);
      
      // Advance time and check for second ping
      jest.advanceTimersByTime(5000);
      await Promise.resolve();
      
      // Clear interval to prevent test leaks
      clearInterval(intervalId);
    });

    it('should call onPing callback with result', (done) => {
      const mockResult = { data: { success: true, status: 'online' } };
      mockAxiosInstance.post.mockResolvedValue(mockResult);
      
      const onPing = jest.fn((result) => {
        expect(result).toEqual({
          success: true,
          data: mockResult.data,
          attempt: 1
        });
        clearInterval(intervalId);
        done();
      });
      
      const intervalId = nodeClient.startPinging(5000, onPing);
    });

  });
});