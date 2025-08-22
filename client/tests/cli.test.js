const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

describe('CLI Integration Tests', () => {
  const cliPath = path.join(__dirname, '..', 'src', 'cli.js');
  const testConfigDir = path.join(os.tmpdir(), 'llmjob-test-' + Date.now());
  
  beforeAll(() => {
    // Create test config directory
    fs.mkdirSync(testConfigDir, { recursive: true });
  });
  
  afterAll(() => {
    // Clean up test config directory
    if (fs.existsSync(testConfigDir)) {
      fs.rmSync(testConfigDir, { recursive: true, force: true });
    }
  });
  
  function runCLI(args = [], env = {}) {
    return new Promise((resolve, reject) => {
      const child = spawn('node', [cliPath, ...args], {
        env: {
          ...process.env,
          LLMJOB_CONFIG_DIR: testConfigDir,
          ...env
        }
      });
      
      let stdout = '';
      let stderr = '';
      
      child.stdout.on('data', (data) => {
        stdout += data.toString();
      });
      
      child.stderr.on('data', (data) => {
        stderr += data.toString();
      });
      
      child.on('close', (code) => {
        resolve({ code, stdout, stderr });
      });
      
      child.on('error', (error) => {
        reject(error);
      });
      
      // Kill long-running processes after timeout
      setTimeout(() => {
        child.kill('SIGTERM');
      }, 2000);
    });
  }
  
  describe('help command', () => {
    it('should display help when no arguments provided', async () => {
      const result = await runCLI([]);
      
      // When no command is provided, help is shown but exit code is 1
      expect(result.stdout).toContain('LLMJob Node Client');
      expect(result.stdout).toContain('Commands:');
      expect(result.stdout).toContain('start');
      expect(result.stdout).toContain('info');
      expect(result.stdout).toContain('reset');
      expect(result.stdout).toContain('config');
    });
    
    it('should display help with --help flag', async () => {
      const result = await runCLI(['--help']);
      
      expect(result.code).toBe(0);
      expect(result.stdout).toContain('LLMJob Node Client');
    });
    
    it('should display version with --version flag', async () => {
      const result = await runCLI(['--version']);
      
      expect(result.code).toBe(0);
      expect(result.stdout).toMatch(/\d+\.\d+\.\d+/);
    });
  });
  
  describe('info command', () => {
    it('should display node information', async () => {
      const result = await runCLI(['info']);
      
      expect(result.code).toBe(0);
      expect(result.stdout).toContain('LLMJob Node Information');
      expect(result.stdout).toContain('Node ID:');
      expect(result.stdout).toContain('Public Key:');
      expect(result.stdout).toContain('Server:');
      expect(result.stdout).toContain('Claim URLs:');
    });
  });
  
  describe('config command', () => {
    it('should display config file location', async () => {
      const result = await runCLI(['config']);
      
      expect(result.code).toBe(0);
      expect(result.stdout).toContain('Config file:');
      expect(result.stdout).toContain('config.json');
    });
  });
  
  describe('reset command', () => {
    it('should require force flag', async () => {
      const result = await runCLI(['reset']);
      
      expect(result.code).toBe(1);
      expect(result.stdout).toContain('Warning');
      expect(result.stdout).toContain('--force');
    });
    
    it('should reset config with force flag', async () => {
      const result = await runCLI(['reset', '--force']);
      
      expect(result.code).toBe(0);
      expect(result.stdout).toContain('Node configuration reset successfully');
      expect(result.stdout).toContain('New Node ID:');
    });
  });
  
  describe('start command', () => {
    it('should start node client', async () => {
      const result = await runCLI(['start', '--interval', '1']);
      
      // Process should be killed by timeout
      expect(result.stdout).toContain('LLMJob Node Client Started');
      expect(result.stdout).toContain('Node ID:');
      expect(result.stdout).toContain('Claim your node:');
      expect(result.stdout).toContain('Pinging server every 1 minutes');
    });
    
    it('should accept custom node name', async () => {
      const result = await runCLI(['start', '--name', 'TestNode', '--interval', '1']);
      
      expect(result.stdout).toContain('LLMJob Node Client Started');
      expect(result.stdout).toContain('name=TestNode');
    });
  });
});