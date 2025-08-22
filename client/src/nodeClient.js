const axios = require('axios');
const nacl = require('tweetnacl');
const naclUtil = require('tweetnacl-util');

class NodeClient {
  constructor(config) {
    this.config = config;
    this.axiosInstance = axios.create({
      baseURL: config.serverUrl,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }

  signMessage(message, secretKey) {
    const messageBytes = naclUtil.decodeUTF8(message);
    const secretKeyBytes = naclUtil.decodeBase64(secretKey);
    const signature = nacl.sign.detached(messageBytes, secretKeyBytes);
    return naclUtil.encodeBase64(signature);
  }

  generateClaimUrl(nodeName) {
    const baseUrl = this.config.serverUrl.replace('/api', '');
    const params = new URLSearchParams({
      publicKey: this.config.publicKey,
      name: nodeName || `Node-${this.config.nodeId}`
    });
    
    const fullUrl = `${baseUrl}/add-node?${params.toString()}`;
    const shortUrl = `${baseUrl}/add-node?id=${this.config.nodeId}`;
    
    return {
      full: fullUrl,
      short: shortUrl
    };
  }

  async ping(retries = 3, retryDelay = 5000) {
    const timestamp = Date.now();
    const message = `${this.config.nodeId}:${timestamp}`;
    const signature = this.signMessage(message, this.config.secretKey);

    const data = {
      nodeId: this.config.nodeId,
      publicKey: this.config.publicKey,
      signature,
      timestamp
    };

    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        const response = await this.axiosInstance.post('/api/nodes/ping', data);
        return {
          success: true,
          data: response.data,
          attempt
        };
      } catch (error) {
        if (attempt === retries) {
          const errorMessage = error.response?.data?.error || error.message;
          return {
            success: false,
            error: errorMessage,
            statusCode: error.response?.status,
            attempt
          };
        }
        
        // Wait before retry
        if (attempt < retries) {
          await new Promise(resolve => setTimeout(resolve, retryDelay));
        }
      }
    }
  }

  startPinging(interval = 10 * 60 * 1000, onPing = null) {
    // Initial ping
    this.ping().then(result => {
      if (onPing) onPing(result);
    });

    // Set up interval
    const intervalId = setInterval(async () => {
      const result = await this.ping();
      if (onPing) onPing(result);
    }, interval);

    return intervalId;
  }

  stopPinging(intervalId) {
    if (intervalId) {
      clearInterval(intervalId);
    }
  }
}

module.exports = NodeClient;