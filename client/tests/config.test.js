const fs = require('fs');
const path = require('path');
const os = require('os');
const ConfigManager = require('../src/config');
const nacl = require('tweetnacl');
const naclUtil = require('tweetnacl-util');

// Mock fs module
jest.mock('fs');

describe('ConfigManager', () => {
  let configManager;
  const testConfigDir = '/test/.llmjob';
  const testConfigFile = '/test/.llmjob/config.json';

  beforeEach(() => {
    jest.clearAllMocks();
    configManager = new ConfigManager('/test/.llmjob');
    
    // Default mocks
    fs.existsSync.mockReturnValue(false);
    fs.mkdirSync.mockImplementation(() => {});
    fs.writeFileSync.mockImplementation(() => {});
    fs.readFileSync.mockImplementation(() => {});
    fs.unlinkSync.mockImplementation(() => {});
  });

  describe('constructor', () => {
    it('should use custom config path if provided', () => {
      const customPath = '/custom/path';
      const cm = new ConfigManager(customPath);
      expect(cm.configDir).toBe(customPath);
      expect(cm.configFile).toBe(path.join(customPath, 'config.json'));
    });

    it('should use default home directory if no path provided', () => {
      const originalHomedir = os.homedir;
      os.homedir = jest.fn().mockReturnValue('/home/user');
      
      const cm = new ConfigManager();
      expect(cm.configDir).toBe('/home/user/.llmjob');
      
      os.homedir = originalHomedir;
    });
  });

  describe('ensureConfigDir', () => {
    it('should create directory if it does not exist', () => {
      fs.existsSync.mockReturnValue(false);
      
      configManager.ensureConfigDir();
      
      expect(fs.mkdirSync).toHaveBeenCalledWith(
        testConfigDir,
        { recursive: true, mode: 0o700 }
      );
    });

    it('should not create directory if it already exists', () => {
      fs.existsSync.mockReturnValue(true);
      
      configManager.ensureConfigDir();
      
      expect(fs.mkdirSync).not.toHaveBeenCalled();
    });
  });

  describe('generateNodeFingerprint', () => {
    it('should generate consistent 6-character fingerprint', () => {
      const publicKey = 'test_public_key_123';
      const fingerprint1 = configManager.generateNodeFingerprint(publicKey);
      const fingerprint2 = configManager.generateNodeFingerprint(publicKey);
      
      expect(fingerprint1).toBe(fingerprint2);
      expect(fingerprint1).toHaveLength(6);
      expect(fingerprint1).toMatch(/^[a-f0-9]{6}$/);
    });

    it('should generate different fingerprints for different keys', () => {
      const fingerprint1 = configManager.generateNodeFingerprint('key1');
      const fingerprint2 = configManager.generateNodeFingerprint('key2');
      
      expect(fingerprint1).not.toBe(fingerprint2);
    });
  });

  describe('generateKeypair', () => {
    it('should generate valid ED25519 keypair', () => {
      const keypair = configManager.generateKeypair();
      
      expect(keypair).toHaveProperty('publicKey');
      expect(keypair).toHaveProperty('secretKey');
      
      // Verify keys are base64 encoded
      expect(() => naclUtil.decodeBase64(keypair.publicKey)).not.toThrow();
      expect(() => naclUtil.decodeBase64(keypair.secretKey)).not.toThrow();
      
      // Verify key lengths
      const publicKeyBytes = naclUtil.decodeBase64(keypair.publicKey);
      const secretKeyBytes = naclUtil.decodeBase64(keypair.secretKey);
      expect(publicKeyBytes).toHaveLength(32);
      expect(secretKeyBytes).toHaveLength(64);
    });

    it('should generate unique keypairs', () => {
      const keypair1 = configManager.generateKeypair();
      const keypair2 = configManager.generateKeypair();
      
      expect(keypair1.publicKey).not.toBe(keypair2.publicKey);
      expect(keypair1.secretKey).not.toBe(keypair2.secretKey);
    });
  });

  describe('loadConfig', () => {
    it('should return null if config file does not exist', () => {
      fs.existsSync.mockReturnValue(false);
      
      const config = configManager.loadConfig();
      
      expect(config).toBeNull();
      expect(fs.mkdirSync).toHaveBeenCalled();
    });

    it('should load and parse existing config file', () => {
      const mockConfig = {
        nodeId: 'abc123',
        publicKey: 'public_key',
        secretKey: 'secret_key'
      };
      
      fs.existsSync.mockImplementation((path) => {
        if (path === testConfigDir) return true;
        if (path === testConfigFile) return true;
        return false;
      });
      fs.readFileSync.mockReturnValue(JSON.stringify(mockConfig));
      
      const config = configManager.loadConfig();
      
      expect(config).toEqual(mockConfig);
      expect(fs.readFileSync).toHaveBeenCalledWith(testConfigFile, 'utf8');
    });

    it('should throw error if config file is invalid JSON', () => {
      fs.existsSync.mockImplementation((path) => {
        if (path === testConfigDir) return true;
        if (path === testConfigFile) return true;
        return false;
      });
      fs.readFileSync.mockReturnValue('invalid json');
      
      expect(() => configManager.loadConfig()).toThrow('Failed to read config file');
    });

    it('should throw error if reading file fails', () => {
      fs.existsSync.mockImplementation((path) => {
        if (path === testConfigDir) return true;
        if (path === testConfigFile) return true;
        return false;
      });
      fs.readFileSync.mockImplementation(() => {
        throw new Error('Permission denied');
      });
      
      expect(() => configManager.loadConfig()).toThrow('Failed to read config file: Permission denied');
    });
  });

  describe('saveConfig', () => {
    it('should save config with proper formatting and permissions', () => {
      const config = { nodeId: 'test123', publicKey: 'key' };
      
      configManager.saveConfig(config);
      
      expect(fs.writeFileSync).toHaveBeenCalledWith(
        testConfigFile,
        JSON.stringify(config, null, 2),
        { mode: 0o600 }
      );
    });

    it('should create directory if it does not exist', () => {
      fs.existsSync.mockReturnValue(false);
      const config = { nodeId: 'test123' };
      
      configManager.saveConfig(config);
      
      expect(fs.mkdirSync).toHaveBeenCalled();
    });

    it('should throw error if writing fails', () => {
      fs.writeFileSync.mockImplementation(() => {
        throw new Error('Disk full');
      });
      
      expect(() => configManager.saveConfig({})).toThrow('Failed to save config file: Disk full');
    });
  });

  describe('getOrCreateConfig', () => {
    it('should return existing config if it exists', () => {
      const existingConfig = {
        nodeId: 'existing',
        publicKey: 'existing_key',
        secretKey: 'existing_secret'
      };
      
      fs.existsSync.mockImplementation((path) => {
        if (path === testConfigDir) return true;
        if (path === testConfigFile) return true;
        return false;
      });
      fs.readFileSync.mockReturnValue(JSON.stringify(existingConfig));
      
      const config = configManager.getOrCreateConfig();
      
      expect(config).toEqual(existingConfig);
      expect(fs.writeFileSync).not.toHaveBeenCalled();
    });

    it('should create new config if none exists', () => {
      fs.existsSync.mockReturnValue(false);
      
      const config = configManager.getOrCreateConfig();
      
      expect(config).toHaveProperty('nodeId');
      expect(config).toHaveProperty('publicKey');
      expect(config).toHaveProperty('secretKey');
      expect(config).toHaveProperty('serverUrl');
      expect(config).toHaveProperty('createdAt');
      expect(fs.writeFileSync).toHaveBeenCalled();
    });

    it('should use environment variable for server URL if set', () => {
      const originalEnv = process.env.LLMJOB_SERVER_URL;
      process.env.LLMJOB_SERVER_URL = 'https://custom.server.com';
      fs.existsSync.mockReturnValue(false);
      
      const config = configManager.getOrCreateConfig();
      
      expect(config.serverUrl).toBe('https://custom.server.com');
      
      process.env.LLMJOB_SERVER_URL = originalEnv;
    });

    it('should use default server URL if environment variable not set', () => {
      delete process.env.LLMJOB_SERVER_URL;
      fs.existsSync.mockReturnValue(false);
      
      const config = configManager.getOrCreateConfig();
      
      expect(config.serverUrl).toBe('https://llmjob-production.up.railway.app');
    });

    it('should generate valid nodeId from public key', () => {
      fs.existsSync.mockReturnValue(false);
      
      const config = configManager.getOrCreateConfig();
      const expectedNodeId = configManager.generateNodeFingerprint(config.publicKey);
      
      expect(config.nodeId).toBe(expectedNodeId);
    });
  });

  describe('updateConfig', () => {
    it('should update existing config with new values', () => {
      const existingConfig = {
        nodeId: 'test123',
        publicKey: 'key1',
        serverUrl: 'https://old.server.com'
      };
      
      fs.existsSync.mockImplementation((path) => {
        if (path === testConfigDir) return true;
        if (path === testConfigFile) return true;
        return false;
      });
      fs.readFileSync.mockReturnValue(JSON.stringify(existingConfig));
      
      const updates = { serverUrl: 'https://new.server.com' };
      const updatedConfig = configManager.updateConfig(updates);
      
      expect(updatedConfig.serverUrl).toBe('https://new.server.com');
      expect(updatedConfig.nodeId).toBe('test123');
      expect(fs.writeFileSync).toHaveBeenCalled();
    });

    it('should create config if none exists', () => {
      fs.existsSync.mockReturnValue(false);
      
      const updates = { customField: 'value' };
      const updatedConfig = configManager.updateConfig(updates);
      
      expect(updatedConfig.customField).toBe('value');
      expect(updatedConfig).toHaveProperty('nodeId');
      expect(updatedConfig).toHaveProperty('publicKey');
    });
  });

  describe('deleteConfig', () => {
    it('should delete config file if it exists', () => {
      fs.existsSync.mockReturnValue(true);
      
      configManager.deleteConfig();
      
      expect(fs.unlinkSync).toHaveBeenCalledWith(testConfigFile);
    });

    it('should not throw if config file does not exist', () => {
      fs.existsSync.mockReturnValue(false);
      
      expect(() => configManager.deleteConfig()).not.toThrow();
      expect(fs.unlinkSync).not.toHaveBeenCalled();
    });
  });
});