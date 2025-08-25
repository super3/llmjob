const os = require('os');
const fs = require('fs').promises;
const path = require('path');
const { exec } = require('child_process');
const { promisify } = require('util');
const execAsync = promisify(exec);
const { Ollama } = require('ollama');

const DEFAULT_MODEL = 'llama3.2:3b';

class OllamaClient {
  constructor(configDir = null) {
    this.configDir = configDir || path.join(os.homedir(), '.llmjob');
    this.capabilitiesFile = path.join(this.configDir, 'capabilities.json');
    this.capabilities = null;
    this.ollama = new Ollama({ host: 'http://localhost:11434' });
  }

  // Detect hardware capabilities
  async detectHardwareCapabilities() {
    const capabilities = {
      cpu: {
        cores: os.cpus().length,
        model: os.cpus()[0]?.model || 'Unknown',
        speed: os.cpus()[0]?.speed || 0
      },
      memory: {
        total: Math.round(os.totalmem() / (1024 * 1024 * 1024) * 10) / 10, // GB with 1 decimal
        free: Math.round(os.freemem() / (1024 * 1024 * 1024) * 10) / 10
      },
      platform: os.platform(),
      arch: os.arch(),
      gpu: await this.detectGPU(),
      detectedAt: new Date().toISOString()
    };

    return capabilities;
  }

  // Detect GPU (basic detection, can be enhanced)
  async detectGPU() {
    const platform = os.platform();
    
    try {
      if (platform === 'darwin') {
        // macOS - check for Apple Silicon
        const { stdout } = await execAsync('sysctl -n machdep.cpu.brand_string');
        if (stdout.includes('Apple')) {
          const { stdout: gpuInfo } = await execAsync('system_profiler SPDisplaysDataType | grep "Chipset Model" | head -1').catch(() => ({ stdout: '' }));
          return {
            type: 'apple_silicon',
            model: gpuInfo.trim().replace('Chipset Model:', '').trim() || 'Apple Silicon',
            available: true
          };
        }
      } else if (platform === 'linux') {
        // Linux - check for NVIDIA GPU
        try {
          const { stdout } = await execAsync('nvidia-smi --query-gpu=name --format=csv,noheader');
          return {
            type: 'nvidia',
            model: stdout.trim(),
            available: true
          };
        } catch {
          // No NVIDIA GPU
        }
      } else if (platform === 'win32') {
        // Windows - basic GPU detection
        try {
          const { stdout } = await execAsync('wmic path win32_VideoController get name');
          const lines = stdout.split('\n').filter(line => line.trim() && !line.includes('Name'));
          return {
            type: 'windows',
            model: lines[0]?.trim() || 'Unknown',
            available: true
          };
        } catch {
          // Could not detect GPU
        }
      }
    } catch (error) {
      // GPU detection failed
    }

    return {
      type: 'none',
      model: 'CPU only',
      available: false
    };
  }

  // Store capabilities to file
  async storeCapabilities() {
    const capabilities = await this.detectHardwareCapabilities();
    
    // Ensure directory exists
    await fs.mkdir(this.configDir, { recursive: true });
    
    // Write capabilities to file
    await fs.writeFile(
      this.capabilitiesFile,
      JSON.stringify(capabilities, null, 2),
      'utf8'
    );

    this.capabilities = capabilities;
    return capabilities;
  }

  // Load capabilities from file
  async loadCapabilities() {
    try {
      const data = await fs.readFile(this.capabilitiesFile, 'utf8');
      this.capabilities = JSON.parse(data);
      return this.capabilities;
    } catch (error) {
      // File doesn't exist or is invalid, detect and store
      return await this.storeCapabilities();
    }
  }

  // Check if Ollama is installed
  async isOllamaInstalled() {
    try {
      const { stdout } = await execAsync('which ollama');
      return stdout.trim().length > 0;
    } catch {
      return false;
    }
  }

  // Install Ollama (platform-specific)
  async installOllama() {
    const platform = os.platform();
    
    console.log('Installing Ollama...');
    
    try {
      if (platform === 'darwin') {
        // macOS - try brew first
        try {
          await execAsync('which brew');
          console.log('Installing via Homebrew...');
          await execAsync('brew install ollama');
        } catch {
          // Fallback to curl script
          console.log('Installing via curl script...');
          await execAsync('curl -fsSL https://ollama.ai/install.sh | sh');
        }
      } else if (platform === 'linux') {
        // Linux - use curl script
        console.log('Installing via curl script...');
        await execAsync('curl -fsSL https://ollama.ai/install.sh | sh');
      } else if (platform === 'win32') {
        // Windows - provide instructions
        throw new Error('Please install Ollama manually from https://ollama.ai/download');
      } else {
        throw new Error(`Unsupported platform: ${platform}`);
      }
      
      console.log('Ollama installed successfully');
      return true;
    } catch (error) {
      console.error('Failed to install Ollama:', error.message);
      throw error;
    }
  }

  // Start Ollama service
  async startOllamaService() {
    const platform = os.platform();
    
    try {
      if (platform === 'darwin') {
        // macOS - Ollama runs as a service
        await execAsync('ollama serve', { detached: true });
      } else if (platform === 'linux') {
        // Linux - start service
        await execAsync('systemctl start ollama || ollama serve', { detached: true });
      }
      
      // Wait for service to be ready
      await this.waitForService();
      return true;
    } catch (error) {
      console.error('Failed to start Ollama service:', error.message);
      return false;
    }
  }

  // Wait for Ollama service to be ready
  async waitForService(maxAttempts = 30) {
    for (let i = 0; i < maxAttempts; i++) {
      if (await this.checkServiceStatus()) {
        return true;
      }
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    return false;
  }

  // Check if Ollama service is running
  async checkServiceStatus() {
    try {
      await this.ollama.list();
      return true;
    } catch {
      return false;
    }
  }

  // Get Ollama version
  async getVersion() {
    try {
      const { stdout } = await execAsync('ollama --version');
      return { version: stdout.trim() };
    } catch (error) {
      throw new Error('Failed to get Ollama version: ' + error.message);
    }
  }

  // List available models
  async listModels() {
    try {
      const response = await this.ollama.list();
      return response.models || [];
    } catch (error) {
      throw new Error('Failed to list models: ' + error.message);
    }
  }

  // Check if model exists
  async hasModel(modelName = DEFAULT_MODEL) {
    const models = await this.listModels();
    return models.some(model => model.name === modelName);
  }

  // Pull model with progress reporting
  async pullModel(modelName = DEFAULT_MODEL, onProgress = null) {
    console.log(`Pulling model ${modelName}...`);
    
    try {
      const stream = await this.ollama.pull({ 
        model: modelName,
        stream: true 
      });
      
      let lastProgress = 0;
      
      for await (const part of stream) {
        if (part.status) {
          // Calculate progress if available
          if (part.completed && part.total) {
            const progress = Math.round((part.completed / part.total) * 100);
            if (progress !== lastProgress) {
              lastProgress = progress;
              if (onProgress) {
                onProgress(progress, part.status);
              } else {
                process.stdout.write(`\rProgress: ${progress}% - ${part.status}`);
              }
            }
          } else if (part.status !== 'success') {
            if (onProgress) {
              onProgress(null, part.status);
            } else {
              process.stdout.write(`\r${part.status}`);
            }
          }
        }
      }
      
      console.log('\nModel pulled successfully');
      return true;
    } catch (error) {
      throw new Error('Failed to pull model: ' + error.message);
    }
  }

  // Test inference with streaming
  async testInference(prompt = 'Hello, how are you?', stream = true) {
    try {
      const startTime = Date.now();
      let fullResponse = '';
      let tokenCount = 0;

      if (stream) {
        const response = await this.ollama.generate({
          model: DEFAULT_MODEL,
          prompt: prompt,
          stream: true
        });

        for await (const part of response) {
          if (part.response) {
            fullResponse += part.response;
            tokenCount++;
            process.stdout.write(part.response);
          }
        }

        const duration = (Date.now() - startTime) / 1000;
        const tokensPerSecond = tokenCount / duration;

        return {
          response: fullResponse,
          tokenCount,
          duration,
          tokensPerSecond: Math.round(tokensPerSecond * 10) / 10,
          model: DEFAULT_MODEL
        };
      } else {
        const response = await this.ollama.generate({
          model: DEFAULT_MODEL,
          prompt: prompt,
          stream: false
        });

        const duration = (Date.now() - startTime) / 1000;
        
        return {
          response: response.response,
          duration,
          model: DEFAULT_MODEL
        };
      }
    } catch (error) {
      throw new Error('Inference failed: ' + error.message);
    }
  }

  // Benchmark inference speed
  async benchmarkInference(prompts = null) {
    if (!prompts) {
      prompts = [
        'What is 2+2?',
        'Write a haiku about programming.',
        'Explain quantum computing in simple terms.'
      ];
    }

    console.log('Running inference benchmark...\n');
    const results = [];

    for (const prompt of prompts) {
      console.log(`\nPrompt: "${prompt}"\n`);
      const result = await this.testInference(prompt, true);
      console.log(`\n\nTokens/sec: ${result.tokensPerSecond}`);
      console.log(`Duration: ${result.duration}s`);
      console.log(`Tokens: ${result.tokenCount}`);
      
      results.push({
        prompt,
        tokensPerSecond: result.tokensPerSecond,
        duration: result.duration,
        tokenCount: result.tokenCount
      });
    }

    const avgTokensPerSecond = results.reduce((sum, r) => sum + r.tokensPerSecond, 0) / results.length;
    
    console.log('\n' + '='.repeat(50));
    console.log(`Average tokens/sec: ${Math.round(avgTokensPerSecond * 10) / 10}`);
    
    return {
      results,
      averageTokensPerSecond: Math.round(avgTokensPerSecond * 10) / 10
    };
  }

  // Initialize Ollama (full setup)
  async initialize(options = {}) {
    const steps = [];
    
    // 1. Detect and store hardware capabilities
    console.log('Detecting hardware capabilities...');
    const capabilities = await this.storeCapabilities();
    steps.push({ step: 'capabilities', success: true, data: capabilities });
    
    // 2. Check/Install Ollama
    const isInstalled = await this.isOllamaInstalled();
    if (!isInstalled && !options.skipInstall) {
      console.log('Ollama not found. Installing...');
      try {
        await this.installOllama();
        steps.push({ step: 'install', success: true });
      } catch (error) {
        steps.push({ step: 'install', success: false, error: error.message });
        if (!options.continueOnError) throw error;
      }
    } else {
      steps.push({ step: 'install', success: true, skipped: isInstalled });
    }
    
    // 3. Check/Start service
    console.log('Checking Ollama service...');
    let serviceRunning = await this.checkServiceStatus();
    if (!serviceRunning) {
      console.log('Starting Ollama service...');
      serviceRunning = await this.startOllamaService();
    }
    steps.push({ step: 'service', success: serviceRunning });
    
    if (!serviceRunning) {
      throw new Error('Failed to start Ollama service');
    }
    
    // 4. Get version
    const version = await this.getVersion();
    console.log(`Ollama version: ${version.version}`);
    steps.push({ step: 'version', success: true, data: version });
    
    // 5. Check/Pull model
    const hasModel = await this.hasModel(DEFAULT_MODEL);
    if (!hasModel && !options.skipModel) {
      await this.pullModel(DEFAULT_MODEL, (progress, status) => {
        if (progress !== null) {
          process.stdout.write(`\rPulling model: ${progress}% - ${status}`);
        }
      });
      console.log('');
      steps.push({ step: 'model', success: true });
    } else {
      steps.push({ step: 'model', success: true, skipped: hasModel });
    }
    
    // 6. Test inference
    if (!options.skipTest) {
      console.log('\nTesting inference...\n');
      const test = await this.testInference('Hello! Please respond with "OK" if you are working.');
      console.log('\n');
      steps.push({ step: 'test', success: true, data: test });
    }
    
    return {
      success: true,
      capabilities,
      steps
    };
  }

  // Generate completion (for job processing)
  async generate(prompt, options = {}) {
    return await this.ollama.generate({
      model: options.model || DEFAULT_MODEL,
      prompt,
      stream: options.stream || false,
      ...options
    });
  }

  // Chat completion (for conversational tasks)
  async chat(messages, options = {}) {
    return await this.ollama.chat({
      model: options.model || DEFAULT_MODEL,
      messages,
      stream: options.stream || false,
      ...options
    });
  }

  // Generate embeddings
  async embed(input, options = {}) {
    return await this.ollama.embed({
      model: options.model || DEFAULT_MODEL,
      input,
      ...options
    });
  }
}

module.exports = OllamaClient;