// Create mock execAsync before any imports
const mockExecAsync = jest.fn();

// Mock modules before imports
jest.mock('ollama');
jest.mock('fs', () => ({
  promises: {
    mkdir: jest.fn(),
    writeFile: jest.fn(),
    readFile: jest.fn(),
    rm: jest.fn()
  }
}));
jest.mock('child_process');
jest.mock('util', () => ({
  ...jest.requireActual('util'),
  promisify: jest.fn(() => mockExecAsync)
}));

// Now import modules
const OllamaClient = require('../src/ollama');
const { Ollama } = require('ollama');
const fs = require('fs').promises;
const path = require('path');
const os = require('os');
const { exec } = require('child_process');
const { promisify } = require('util');

describe('OllamaClient', () => {
  let client;
  let tempDir;
  let mockOllama;
  
  beforeEach(() => {
    tempDir = path.join(os.tmpdir(), 'llmjob-test-' + Date.now());
    
    // Create mock Ollama instance
    mockOllama = {
      list: jest.fn(),
      pull: jest.fn(),
      generate: jest.fn(),
      chat: jest.fn(),
      embed: jest.fn()
    };
    
    // Mock Ollama constructor
    Ollama.mockImplementation(() => mockOllama);
    
    // Reset all mocks including execAsync and fs
    jest.clearAllMocks();
    mockExecAsync.mockReset();
    fs.mkdir.mockReset();
    fs.writeFile.mockReset();
    fs.readFile.mockReset();
    fs.rm.mockReset();
    
    client = new OllamaClient(tempDir);
  });
  
  afterEach(async () => {
    // Clean up temp directory
    try {
      await fs.rm(tempDir, { recursive: true });
    } catch (e) {
      // Ignore cleanup errors
    }
  });
  
  describe('detectHardwareCapabilities', () => {
    it('should detect basic hardware capabilities', async () => {
      // Mock GPU detection to avoid hanging
      client.detectGPU = jest.fn().mockResolvedValue({
        type: 'none',
        model: 'CPU only',
        available: false
      });
      
      const capabilities = await client.detectHardwareCapabilities();
      
      expect(capabilities).toHaveProperty('cpu');
      expect(capabilities.cpu).toHaveProperty('cores');
      expect(capabilities.cpu.cores).toBeGreaterThan(0);
      
      expect(capabilities).toHaveProperty('memory');
      expect(capabilities.memory).toHaveProperty('total');
      expect(capabilities.memory.total).toBeGreaterThan(0);
      
      expect(capabilities).toHaveProperty('platform');
      expect(capabilities).toHaveProperty('arch');
      expect(capabilities).toHaveProperty('gpu');
      expect(capabilities).toHaveProperty('detectedAt');
    });
  });
  
  describe('storeCapabilities', () => {
    it('should store capabilities to file', async () => {
      const mockCapabilities = {
        cpu: { cores: 8, model: 'Test CPU', speed: 2400 },
        memory: { total: 16, free: 8 },
        platform: 'darwin',
        arch: 'x64',
        gpu: { type: 'none', model: 'CPU only', available: false },
        detectedAt: new Date().toISOString()
      };
      
      // Mock detectHardwareCapabilities
      client.detectHardwareCapabilities = jest.fn().mockResolvedValue(mockCapabilities);
      
      // Mock fs operations
      fs.mkdir.mockResolvedValue();
      fs.writeFile.mockResolvedValue();
      
      const result = await client.storeCapabilities();
      
      expect(fs.mkdir).toHaveBeenCalledWith(tempDir, { recursive: true });
      expect(fs.writeFile).toHaveBeenCalledWith(
        path.join(tempDir, 'capabilities.json'),
        JSON.stringify(mockCapabilities, null, 2),
        'utf8'
      );
      expect(result).toEqual(mockCapabilities);
      expect(client.capabilities).toEqual(mockCapabilities);
    });
  });
  
  describe('loadCapabilities', () => {
    it('should load capabilities from file if exists', async () => {
      const mockCapabilities = {
        cpu: { cores: 8, model: 'Test CPU', speed: 2400 },
        memory: { total: 16, free: 8 },
        platform: 'darwin',
        arch: 'x64',
        gpu: { type: 'none', model: 'CPU only', available: false },
        detectedAt: '2024-01-01T00:00:00.000Z'
      };
      
      fs.readFile.mockResolvedValue(JSON.stringify(mockCapabilities));
      
      const result = await client.loadCapabilities();
      
      expect(fs.readFile).toHaveBeenCalledWith(
        path.join(tempDir, 'capabilities.json'),
        'utf8'
      );
      expect(result).toEqual(mockCapabilities);
      expect(client.capabilities).toEqual(mockCapabilities);
    });
    
    it('should detect and store capabilities if file does not exist', async () => {
      fs.readFile.mockRejectedValue(new Error('File not found'));
      
      const mockCapabilities = {
        cpu: { cores: 8, model: 'Test CPU', speed: 2400 },
        memory: { total: 16, free: 8 },
        platform: 'darwin',
        arch: 'x64',
        gpu: { type: 'none', model: 'CPU only', available: false },
        detectedAt: new Date().toISOString()
      };
      
      client.storeCapabilities = jest.fn().mockResolvedValue(mockCapabilities);
      
      const result = await client.loadCapabilities();
      
      expect(client.storeCapabilities).toHaveBeenCalled();
      expect(result).toEqual(mockCapabilities);
    });
  });
  
  describe('isOllamaInstalled', () => {
    it('should return true if ollama is installed', async () => {
      mockExecAsync.mockResolvedValue({ stdout: '/usr/local/bin/ollama' });
      
      const result = await client.isOllamaInstalled();
      
      expect(mockExecAsync).toHaveBeenCalledWith('which ollama');
      expect(result).toBe(true);
    });
    
    it('should return false if ollama is not installed', async () => {
      mockExecAsync.mockRejectedValue(new Error('Command not found'));
      
      const result = await client.isOllamaInstalled();
      
      expect(mockExecAsync).toHaveBeenCalledWith('which ollama');
      expect(result).toBe(false);
    });
  });
  
  describe('checkServiceStatus', () => {
    it('should return true if service is running', async () => {
      mockOllama.list.mockResolvedValue({ models: [] });
      
      const result = await client.checkServiceStatus();
      
      expect(mockOllama.list).toHaveBeenCalled();
      expect(result).toBe(true);
    });
    
    it('should return false if service is not running', async () => {
      mockOllama.list.mockRejectedValue(new Error('Connection refused'));
      
      const result = await client.checkServiceStatus();
      
      expect(result).toBe(false);
    });
  });
  
  describe('getVersion', () => {
    it('should return version info', async () => {
      mockExecAsync.mockResolvedValue({ stdout: 'ollama version 0.1.0\n' });
      
      const result = await client.getVersion();
      
      expect(mockExecAsync).toHaveBeenCalledWith('ollama --version');
      expect(result).toEqual({ version: 'ollama version 0.1.0' });
    });
    
    it('should throw error if command fails', async () => {
      mockExecAsync.mockRejectedValue(new Error('Command not found'));
      
      await expect(client.getVersion()).rejects.toThrow('Failed to get Ollama version');
      expect(mockExecAsync).toHaveBeenCalledWith('ollama --version');
    });
  });
  
  describe('listModels', () => {
    it('should return list of models', async () => {
      const mockModels = [
        { name: 'llama3.2:3b', size: 2000000000 },
        { name: 'mistral:7b', size: 4000000000 }
      ];
      mockOllama.list.mockResolvedValue({ models: mockModels });
      
      const result = await client.listModels();
      
      expect(mockOllama.list).toHaveBeenCalled();
      expect(result).toEqual(mockModels);
    });
    
    it('should return empty array if no models', async () => {
      mockOllama.list.mockResolvedValue({});
      
      const result = await client.listModels();
      
      expect(result).toEqual([]);
    });
  });
  
  describe('hasModel', () => {
    it('should return true if model exists', async () => {
      client.listModels = jest.fn().mockResolvedValue([
        { name: 'llama3.2:3b' },
        { name: 'mistral:7b' }
      ]);
      
      const result = await client.hasModel('llama3.2:3b');
      
      expect(result).toBe(true);
    });
    
    it('should return false if model does not exist', async () => {
      client.listModels = jest.fn().mockResolvedValue([
        { name: 'mistral:7b' }
      ]);
      
      const result = await client.hasModel('llama3.2:3b');
      
      expect(result).toBe(false);
    });
  });
  
  describe('pullModel', () => {
    it('should pull model with progress updates', async () => {
      const mockStream = [
        { status: 'downloading', completed: 500000000, total: 1000000000 },
        { status: 'downloading', completed: 1000000000, total: 1000000000 },
        { status: 'success' }
      ];
      
      mockOllama.pull.mockResolvedValue(mockStream);
      
      const onProgress = jest.fn();
      const result = await client.pullModel('llama3.2:3b', onProgress);
      
      expect(mockOllama.pull).toHaveBeenCalledWith({
        model: 'llama3.2:3b',
        stream: true
      });
      expect(onProgress).toHaveBeenCalledWith(50, 'downloading');
      expect(onProgress).toHaveBeenCalledWith(100, 'downloading');
      expect(result).toBe(true);
    });
  });
  
  describe('testInference', () => {
    it('should perform inference with streaming', async () => {
      const mockStream = [
        { response: 'Hello' },
        { response: ' world' },
        { response: '!', done: true }
      ];
      
      mockOllama.generate.mockResolvedValue(mockStream);
      
      const result = await client.testInference('Test prompt', true);
      
      expect(mockOllama.generate).toHaveBeenCalledWith({
        model: 'llama3.2:3b',
        prompt: 'Test prompt',
        stream: true
      });
      expect(result.response).toBe('Hello world!');
      expect(result.tokenCount).toBe(3);
      expect(result).toHaveProperty('tokensPerSecond');
      expect(result).toHaveProperty('duration');
    });
    
    it('should perform inference without streaming', async () => {
      const mockResponse = {
        response: 'Hello world!',
        model: 'llama3.2:3b'
      };
      
      mockOllama.generate.mockResolvedValue(mockResponse);
      
      const result = await client.testInference('Test prompt', false);
      
      expect(mockOllama.generate).toHaveBeenCalledWith({
        model: 'llama3.2:3b',
        prompt: 'Test prompt',
        stream: false
      });
      expect(result.response).toBe('Hello world!');
      expect(result).toHaveProperty('duration');
    });
  });
  
  describe('benchmarkInference', () => {
    it('should run benchmark with multiple prompts', async () => {
      // Mock testInference
      client.testInference = jest.fn()
        .mockResolvedValueOnce({
          response: '4',
          tokensPerSecond: 10.5,
          duration: 0.5,
          tokenCount: 5
        })
        .mockResolvedValueOnce({
          response: 'Haiku text',
          tokensPerSecond: 8.2,
          duration: 1.2,
          tokenCount: 10
        })
        .mockResolvedValueOnce({
          response: 'Quantum explanation',
          tokensPerSecond: 9.3,
          duration: 2.0,
          tokenCount: 18
        });
      
      const result = await client.benchmarkInference();
      
      expect(client.testInference).toHaveBeenCalledTimes(3);
      expect(result.results).toHaveLength(3);
      expect(result.averageTokensPerSecond).toBeCloseTo(9.3, 1);
    });
  });
  
  describe('initialize', () => {
    it('should initialize Ollama with all steps', async () => {
      // Mock all methods
      client.storeCapabilities = jest.fn().mockResolvedValue({
        cpu: { cores: 8 },
        memory: { total: 16 },
        gpu: { model: 'Test GPU' }
      });
      client.isOllamaInstalled = jest.fn().mockResolvedValue(true);
      client.checkServiceStatus = jest.fn().mockResolvedValue(true);
      client.getVersion = jest.fn().mockResolvedValue({ version: '0.1.0' });
      client.hasModel = jest.fn().mockResolvedValue(true);
      client.testInference = jest.fn().mockResolvedValue({
        response: 'OK',
        tokensPerSecond: 10
      });
      
      const result = await client.initialize();
      
      expect(result.success).toBe(true);
      expect(result.capabilities).toBeDefined();
      expect(result.steps).toContainEqual({ step: 'capabilities', success: true, data: expect.any(Object) });
      expect(result.steps).toContainEqual({ step: 'install', success: true, skipped: true });
      expect(result.steps).toContainEqual({ step: 'service', success: true });
      expect(result.steps).toContainEqual({ step: 'version', success: true, data: { version: '0.1.0' } });
      expect(result.steps).toContainEqual({ step: 'model', success: true, skipped: true });
      expect(result.steps).toContainEqual({ step: 'test', success: true, data: expect.any(Object) });
    });
    
    it('should install Ollama if not installed', async () => {
      client.storeCapabilities = jest.fn().mockResolvedValue({});
      client.isOllamaInstalled = jest.fn().mockResolvedValue(false);
      client.installOllama = jest.fn().mockResolvedValue(true);
      client.checkServiceStatus = jest.fn().mockResolvedValue(true);
      client.getVersion = jest.fn().mockResolvedValue({ version: '0.1.0' });
      client.hasModel = jest.fn().mockResolvedValue(true);
      client.testInference = jest.fn().mockResolvedValue({});
      
      const result = await client.initialize();
      
      expect(client.installOllama).toHaveBeenCalled();
      expect(result.steps).toContainEqual({ step: 'install', success: true });
    });
    
    it('should pull model if not available', async () => {
      client.storeCapabilities = jest.fn().mockResolvedValue({});
      client.isOllamaInstalled = jest.fn().mockResolvedValue(true);
      client.checkServiceStatus = jest.fn().mockResolvedValue(true);
      client.getVersion = jest.fn().mockResolvedValue({ version: '0.1.0' });
      client.hasModel = jest.fn().mockResolvedValue(false);
      client.pullModel = jest.fn().mockResolvedValue(true);
      client.testInference = jest.fn().mockResolvedValue({});
      
      const result = await client.initialize();
      
      expect(client.pullModel).toHaveBeenCalledWith('llama3.2:3b', expect.any(Function));
      expect(result.steps).toContainEqual({ step: 'model', success: true });
    });
    
    it('should skip steps based on options', async () => {
      client.storeCapabilities = jest.fn().mockResolvedValue({});
      client.isOllamaInstalled = jest.fn().mockResolvedValue(true);
      client.checkServiceStatus = jest.fn().mockResolvedValue(true);
      client.getVersion = jest.fn().mockResolvedValue({ version: '0.1.0' });
      client.hasModel = jest.fn().mockResolvedValue(true);
      
      const installOllamaSpy = jest.spyOn(client, 'installOllama');
      const pullModelSpy = jest.spyOn(client, 'pullModel');
      const testInferenceSpy = jest.spyOn(client, 'testInference');
      
      const result = await client.initialize({
        skipInstall: true,
        skipModel: true,
        skipTest: true
      });
      
      expect(installOllamaSpy).not.toHaveBeenCalled();
      expect(pullModelSpy).not.toHaveBeenCalled();
      expect(testInferenceSpy).not.toHaveBeenCalled();
      expect(result.success).toBe(true);
    });
  });
  
  describe('generate', () => {
    it('should call ollama generate with correct params', async () => {
      const mockResponse = { response: 'Generated text' };
      mockOllama.generate.mockResolvedValue(mockResponse);
      
      const result = await client.generate('Test prompt', { temperature: 0.7 });
      
      expect(mockOllama.generate).toHaveBeenCalledWith({
        model: 'llama3.2:3b',
        prompt: 'Test prompt',
        stream: false,
        temperature: 0.7
      });
      expect(result).toEqual(mockResponse);
    });
  });
  
  describe('chat', () => {
    it('should call ollama chat with correct params', async () => {
      const messages = [{ role: 'user', content: 'Hello' }];
      const mockResponse = { message: { content: 'Hi there!' } };
      mockOllama.chat.mockResolvedValue(mockResponse);
      
      const result = await client.chat(messages, { temperature: 0.5 });
      
      expect(mockOllama.chat).toHaveBeenCalledWith({
        model: 'llama3.2:3b',
        messages,
        stream: false,
        temperature: 0.5
      });
      expect(result).toEqual(mockResponse);
    });
  });
  
  describe('embed', () => {
    it('should call ollama embed with correct params', async () => {
      const mockResponse = { embeddings: [[0.1, 0.2, 0.3]] };
      mockOllama.embed.mockResolvedValue(mockResponse);
      
      const result = await client.embed('Test text');
      
      expect(mockOllama.embed).toHaveBeenCalledWith({
        model: 'llama3.2:3b',
        input: 'Test text'
      });
      expect(result).toEqual(mockResponse);
    });
  });
});