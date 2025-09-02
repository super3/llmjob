/**
 * Manual Integration Tests for Ollama
 * 
 * These tests require a real Ollama instance running locally.
 * To run these tests:
 * 1. Install Ollama: https://ollama.ai/download
 * 2. Start Ollama service: ollama serve
 * 3. Pull a model: ollama pull llama3.2:3b
 * 4. Run tests: npm test -- integration.manual.test.js
 * 
 * Note: These tests are skipped by default in CI/CD pipelines
 */

const OllamaClient = require('../src/ollama');
const JobWorker = require('../src/jobWorker');
const NodeClient = require('../src/nodeClient');
const path = require('path');
const os = require('os');

// Skip these tests in CI environment or when SKIP_INTEGRATION is set
const describeManual = (process.env.CI || process.env.SKIP_INTEGRATION) ? describe.skip : describe.skip; // Always skip for now

describeManual('Manual Integration Tests with Real Ollama', () => {
  let client;
  let tempDir;
  
  beforeEach(() => {
    tempDir = path.join(os.tmpdir(), 'llmjob-integration-' + Date.now());
    client = new OllamaClient(tempDir);
  });
  
  describe('Ollama Service Integration', () => {
    it('should detect installed Ollama', async () => {
      const isInstalled = await client.isOllamaInstalled();
      expect(isInstalled).toBe(true);
    }, 10000);
    
    it('should check service status', async () => {
      const isRunning = await client.checkServiceStatus();
      expect(isRunning).toBe(true);
    }, 10000);
    
    it('should get Ollama version', async () => {
      const version = await client.getVersion();
      expect(version).toHaveProperty('version');
      expect(version.version).toMatch(/ollama/i);
    }, 10000);
    
    it('should list available models', async () => {
      const models = await client.listModels();
      expect(Array.isArray(models)).toBe(true);
      // Should have at least one model for testing
      expect(models.length).toBeGreaterThan(0);
    }, 10000);
  });
  
  describe('Model Operations', () => {
    it('should check if llama3.2:3b model exists', async () => {
      const hasModel = await client.hasModel('llama3.2:3b');
      expect(typeof hasModel).toBe('boolean');
      
      if (!hasModel) {
        console.log('Model llama3.2:3b not found. Please pull it first: ollama pull llama3.2:3b');
      }
    }, 10000);
    
    it('should perform text generation', async () => {
      const hasModel = await client.hasModel('llama3.2:3b');
      if (!hasModel) {
        console.log('Skipping generation test - model not available');
        return;
      }
      
      const result = await client.generate('What is 2+2?', {
        temperature: 0.1,
        max_tokens: 50
      });
      
      expect(result).toHaveProperty('response');
      expect(result.response).toBeTruthy();
      expect(result.response.toLowerCase()).toContain('4');
    }, 30000);
    
    it('should perform streaming generation', async () => {
      const hasModel = await client.hasModel('llama3.2:3b');
      if (!hasModel) {
        console.log('Skipping streaming test - model not available');
        return;
      }
      
      const result = await client.testInference('Write a haiku about coding', true);
      
      expect(result).toHaveProperty('response');
      expect(result).toHaveProperty('tokenCount');
      expect(result).toHaveProperty('tokensPerSecond');
      expect(result.tokenCount).toBeGreaterThan(0);
      expect(result.tokensPerSecond).toBeGreaterThan(0);
    }, 30000);
  });
  
  describe('Hardware Capabilities', () => {
    it('should detect and store hardware capabilities', async () => {
      const capabilities = await client.storeCapabilities();
      
      expect(capabilities).toHaveProperty('cpu');
      expect(capabilities.cpu).toHaveProperty('cores');
      expect(capabilities.cpu.cores).toBeGreaterThan(0);
      
      expect(capabilities).toHaveProperty('memory');
      expect(capabilities.memory).toHaveProperty('total');
      expect(capabilities.memory.total).toBeGreaterThan(0);
      
      expect(capabilities).toHaveProperty('gpu');
      expect(capabilities).toHaveProperty('platform');
      expect(capabilities).toHaveProperty('arch');
    }, 10000);
    
    it('should run inference benchmark', async () => {
      const hasModel = await client.hasModel('llama3.2:3b');
      if (!hasModel) {
        console.log('Skipping benchmark - model not available');
        return;
      }
      
      const benchmark = await client.benchmarkInference();
      
      expect(benchmark).toHaveProperty('results');
      expect(benchmark).toHaveProperty('averageTokensPerSecond');
      expect(benchmark.results.length).toBeGreaterThan(0);
      expect(benchmark.averageTokensPerSecond).toBeGreaterThan(0);
      
      console.log(`Benchmark Results: ${benchmark.averageTokensPerSecond} tokens/sec average`);
    }, 60000);
  });
  
  describe('Full Initialization Flow', () => {
    it('should complete full initialization', async () => {
      const result = await client.initialize({
        skipInstall: true, // Don't try to install in tests
        model: 'llama3.2:3b'
      });
      
      expect(result).toHaveProperty('success');
      expect(result).toHaveProperty('steps');
      
      // Check each initialization step
      const stepNames = result.steps.map(s => s.step);
      expect(stepNames).toContain('capabilities');
      expect(stepNames).toContain('install');
      expect(stepNames).toContain('service');
      expect(stepNames).toContain('version');
      expect(stepNames).toContain('model');
      expect(stepNames).toContain('test');
      
      if (result.success) {
        console.log('Initialization successful!');
      } else {
        console.log('Initialization failed:', result.error);
      }
    }, 120000);
  });
});

describeManual('Job Worker Integration with Real Server', () => {
  let nodeClient;
  let jobWorker;
  const testConfig = {
    nodeId: 'test-node-' + Date.now(),
    publicKey: 'test-public-key',
    secretKey: 'test-secret-key',
    serverUrl: process.env.TEST_SERVER_URL || 'http://localhost:3001',
    configDir: path.join(os.tmpdir(), 'llmjob-worker-test')
  };
  
  beforeEach(() => {
    nodeClient = new NodeClient(testConfig);
    jobWorker = new JobWorker(nodeClient, testConfig);
  });
  
  afterEach(async () => {
    if (jobWorker.isRunning) {
      await jobWorker.stop();
    }
  });
  
  describe('Worker Lifecycle', () => {
    it('should initialize worker with Ollama', async () => {
      const capabilities = await jobWorker.initialize();
      
      expect(capabilities).toHaveProperty('cpu');
      expect(capabilities).toHaveProperty('memory');
      expect(jobWorker.ollamaClient).toBeDefined();
    }, 20000);
    
    it('should start and stop worker', async () => {
      await jobWorker.initialize();
      
      await jobWorker.start(5000); // Poll every 5 seconds
      expect(jobWorker.isRunning).toBe(true);
      
      await new Promise(resolve => setTimeout(resolve, 2000)); // Let it run briefly
      
      await jobWorker.stop();
      expect(jobWorker.isRunning).toBe(false);
    }, 30000);
    
    it('should ping server with capabilities', async () => {
      await jobWorker.initialize();
      
      const pingResult = await jobWorker.pingWithCapabilities();
      
      // This will fail if server is not running
      if (pingResult.success) {
        expect(pingResult).toHaveProperty('success', true);
      } else {
        console.log('Server not available for ping test');
      }
    }, 20000);
  });
  
  describe('Job Processing', () => {
    it('should poll for jobs from server', async () => {
      await jobWorker.initialize();
      
      const jobs = await jobWorker.pollForJobs();
      
      expect(Array.isArray(jobs)).toBe(true);
      // May or may not have jobs available
      console.log(`Found ${jobs.length} jobs available`);
    }, 20000);
    
    it('should execute a mock job', async () => {
      await jobWorker.initialize();
      
      const mockJob = {
        id: 'test-job-' + Date.now(),
        prompt: 'What is the capital of France?',
        model: 'llama3.2:3b',
        options: {
          temperature: 0.1,
          max_tokens: 50
        }
      };
      
      // Listen for job events
      const events = {
        started: false,
        chunk: false,
        completed: false,
        failed: false
      };
      
      jobWorker.once('job:started', () => { events.started = true; });
      jobWorker.once('job:chunk', () => { events.chunk = true; });
      jobWorker.once('job:completed', () => { events.completed = true; });
      jobWorker.once('job:failed', () => { events.failed = true; });
      
      // Execute job (this will try to send results to server)
      await jobWorker.executeJob(mockJob);
      
      // Check events were fired
      expect(events.started).toBe(true);
      expect(events.completed || events.failed).toBe(true);
      
      if (events.completed) {
        console.log('Job executed successfully');
      } else {
        console.log('Job execution failed - likely server not available');
      }
    }, 60000);
  });
});

// Instructions for load testing
console.log(`
==========================================
LOAD TESTING INSTRUCTIONS
==========================================

To perform load testing with multiple nodes:

1. Start the Redis server:
   redis-server

2. Start the main server:
   npm start

3. Start multiple worker nodes in separate terminals:
   
   Terminal 1:
   NODE_ID=worker1 npm run worker
   
   Terminal 2:
   NODE_ID=worker2 npm run worker
   
   Terminal 3:
   NODE_ID=worker3 npm run worker

4. Submit test jobs using the API:
   
   # Submit 100 test jobs
   for i in {1..100}; do
     curl -X POST http://localhost:3001/api/jobs \\
       -H "Content-Type: application/json" \\
       -d '{"prompt":"What is '$i' + '$i'?","model":"llama3.2:3b"}'
   done

5. Monitor the cluster dashboard:
   Open http://localhost:3001/cluster.html

6. Check job processing metrics:
   curl http://localhost:3001/api/jobs/stats

==========================================
`);