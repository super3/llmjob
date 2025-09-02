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

describe('OllamaClient Error Handling', () => {
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
    
    // Reset all mocks
    jest.clearAllMocks();
    mockExecAsync.mockReset();
    fs.mkdir.mockReset();
    fs.writeFile.mockReset();
    fs.readFile.mockReset();
    fs.rm.mockReset();
    
    client = new OllamaClient(tempDir);
  });
  
  describe('Ollama Service Unavailability', () => {
    it('should handle Ollama service not running', async () => {
      mockOllama.list.mockRejectedValue(new Error('ECONNREFUSED'));
      
      const result = await client.checkServiceStatus();
      
      expect(result).toBe(false);
      expect(mockOllama.list).toHaveBeenCalled();
    });
    
    it('should throw error when initializing with Ollama not running', async () => {
      // Mock all required methods
      client.storeCapabilities = jest.fn().mockResolvedValue({
        cpu: { cores: 8 },
        memory: { total: 16 }
      });
      client.isOllamaInstalled = jest.fn().mockResolvedValue(true);
      client.checkServiceStatus = jest.fn().mockResolvedValue(false);
      client.startOllamaService = jest.fn().mockResolvedValue(false);
      
      await expect(client.initialize()).rejects.toThrow('Failed to start Ollama service');
      
      expect(client.checkServiceStatus).toHaveBeenCalled();
      expect(client.startOllamaService).toHaveBeenCalled();
    });
    
    it('should handle network errors during model pulling', async () => {
      const onProgress = jest.fn();
      
      // Mock the pull to return an async generator that yields then throws
      mockOllama.pull.mockImplementation(async function* () {
        yield { status: 'downloading', completed: 500000000, total: 1000000000 };
        throw new Error('Network timeout');
      });
      
      await expect(client.pullModel('llama3.2:3b', onProgress)).rejects.toThrow('Failed to pull model: Network timeout');
      
      expect(onProgress).toHaveBeenCalledWith(50, 'downloading');
    });
  });
  
  describe('Inference Error Handling', () => {
    it('should handle generation errors gracefully', async () => {
      mockOllama.generate.mockRejectedValue(new Error('Model not found'));
      
      await expect(client.generate('test prompt')).rejects.toThrow('Model not found');
    });
    
    it('should handle streaming errors during generation', async () => {
      const mockStream = async function*() {
        yield { response: 'Start ' };
        throw new Error('Stream interrupted');
      };
      
      mockOllama.generate.mockReturnValue(mockStream());
      
      await expect(client.testInference('Test prompt', true)).rejects.toThrow('Inference failed: Stream interrupted');
    });
    
    it('should handle chat errors', async () => {
      mockOllama.chat.mockRejectedValue(new Error('Invalid message format'));
      
      await expect(client.chat([{ role: 'user', content: 'test' }]))
        .rejects.toThrow('Invalid message format');
    });
    
    it('should handle embedding errors', async () => {
      mockOllama.embed.mockRejectedValue(new Error('Model does not support embeddings'));
      
      await expect(client.embed('test text'))
        .rejects.toThrow('Model does not support embeddings');
    });
  });
  
  describe('Hardware Detection Errors', () => {
    it('should handle GPU detection failures gracefully', async () => {
      // Mock GPU detection to return error state
      client.detectGPU = jest.fn().mockResolvedValue({
        type: 'none',
        model: 'Detection failed',
        available: false
      });
      
      const capabilities = await client.detectHardwareCapabilities();
      
      expect(capabilities).toHaveProperty('cpu');
      expect(capabilities).toHaveProperty('memory');
      expect(capabilities.gpu).toEqual({
        type: 'none',
        model: 'Detection failed',
        available: false
      });
    });
    
    it('should handle filesystem errors when storing capabilities', async () => {
      fs.mkdir.mockRejectedValue(new Error('Permission denied'));
      
      // Mock detectHardwareCapabilities
      client.detectHardwareCapabilities = jest.fn().mockResolvedValue({
        cpu: { cores: 8 },
        memory: { total: 16 }
      });
      
      await expect(client.storeCapabilities()).rejects.toThrow('Permission denied');
    });
  });
  
  describe('Installation Error Handling', () => {
    it('should handle installation failures on macOS', async () => {
      mockExecAsync.mockRejectedValue(new Error('Installation failed'));
      client.isOllamaInstalled = jest.fn().mockResolvedValue(false);
      
      // Mock platform
      const originalPlatform = process.platform;
      Object.defineProperty(process, 'platform', {
        value: 'darwin'
      });
      
      await expect(client.installOllama()).rejects.toThrow('Installation failed');
      
      // Restore platform
      Object.defineProperty(process, 'platform', {
        value: originalPlatform
      });
    });
    
    it('should throw error for unsupported platforms', async () => {
      // Mock os.platform to return unsupported platform
      const os = require('os');
      const originalPlatform = os.platform;
      os.platform = jest.fn().mockReturnValue('freebsd');
      
      // Create new client after mocking
      const testClient = new OllamaClient(tempDir);
      
      await expect(testClient.installOllama()).rejects.toThrow('Unsupported platform');
      
      // Restore platform
      os.platform = originalPlatform;
    });
  });
  
  describe('Model Management Errors', () => {
    it('should handle model listing errors', async () => {
      mockOllama.list.mockRejectedValue(new Error('Service unavailable'));
      
      await expect(client.listModels()).rejects.toThrow('Service unavailable');
    });
    
    it('should handle corrupted model list response', async () => {
      mockOllama.list.mockResolvedValue({ corrupt: 'data' });
      
      const result = await client.listModels();
      
      expect(result).toEqual([]);
    });
    
    it('should handle model check with service errors', async () => {
      client.listModels = jest.fn().mockRejectedValue(new Error('Connection refused'));
      
      await expect(client.hasModel('llama3.2:3b')).rejects.toThrow('Connection refused');
    });
  });
  
  describe('Benchmark Error Handling', () => {
    it('should fail benchmark if inference fails', async () => {
      client.testInference = jest.fn()
        .mockResolvedValueOnce({ response: 'OK', tokensPerSecond: 10, duration: 1, tokenCount: 10 })
        .mockRejectedValueOnce(new Error('Inference failed'));
      
      await expect(client.benchmarkInference()).rejects.toThrow('Inference failed');
      expect(client.testInference).toHaveBeenCalledTimes(2);
    });
  });
  
  describe('Recovery and Retry Logic', () => {
    it('should handle failed list operations', async () => {
      // Mock a function that fails
      mockOllama.list.mockRejectedValue(new Error('Temporary failure'));
      
      await expect(client.listModels()).rejects.toThrow('Temporary failure');
      expect(mockOllama.list).toHaveBeenCalledTimes(1);
    });
    
    it('should handle concurrent operation failures', async () => {
      mockOllama.generate.mockRejectedValue(new Error('Resource exhausted'));
      
      const promises = [
        client.generate('prompt1'),
        client.generate('prompt2'),
        client.generate('prompt3')
      ];
      
      await expect(Promise.all(promises)).rejects.toThrow('Resource exhausted');
    });
  });
});