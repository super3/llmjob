const fs = require('fs');
const path = require('path');
const os = require('os');
const nacl = require('tweetnacl');
const naclUtil = require('tweetnacl-util');
const crypto = require('crypto');

class ConfigManager {
  constructor(configPath = null) {
    this.configDir = configPath || path.join(os.homedir(), '.llmjob');
    this.configFile = path.join(this.configDir, 'config.json');
  }

  ensureConfigDir() {
    if (!fs.existsSync(this.configDir)) {
      fs.mkdirSync(this.configDir, { recursive: true, mode: 0o700 });
    }
  }

  generateNodeFingerprint(publicKey) {
    const hash = crypto.createHash('sha256').update(publicKey).digest('hex');
    return hash.substring(0, 6);
  }

  generateKeypair() {
    const keypair = nacl.sign.keyPair();
    return {
      publicKey: naclUtil.encodeBase64(keypair.publicKey),
      secretKey: naclUtil.encodeBase64(keypair.secretKey)
    };
  }

  loadConfig() {
    this.ensureConfigDir();
    
    if (fs.existsSync(this.configFile)) {
      try {
        const data = fs.readFileSync(this.configFile, 'utf8');
        return JSON.parse(data);
      } catch (error) {
        throw new Error(`Failed to read config file: ${error.message}`);
      }
    }
    
    return null;
  }

  saveConfig(config) {
    this.ensureConfigDir();
    
    try {
      fs.writeFileSync(
        this.configFile,
        JSON.stringify(config, null, 2),
        { mode: 0o600 }
      );
    } catch (error) {
      throw new Error(`Failed to save config file: ${error.message}`);
    }
  }

  getOrCreateConfig() {
    let config = this.loadConfig();
    
    if (!config) {
      const keypair = this.generateKeypair();
      const nodeId = this.generateNodeFingerprint(keypair.publicKey);
      
      config = {
        nodeId,
        publicKey: keypair.publicKey,
        secretKey: keypair.secretKey,
        serverUrl: process.env.LLMJOB_SERVER_URL || 'https://llmjob-production.up.railway.app',
        createdAt: new Date().toISOString()
      };
      
      this.saveConfig(config);
    }
    
    return config;
  }

  updateConfig(updates) {
    const config = this.getOrCreateConfig();
    const updatedConfig = { ...config, ...updates };
    this.saveConfig(updatedConfig);
    return updatedConfig;
  }

  deleteConfig() {
    if (fs.existsSync(this.configFile)) {
      fs.unlinkSync(this.configFile);
    }
  }
}

module.exports = ConfigManager;