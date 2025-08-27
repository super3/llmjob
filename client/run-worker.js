#!/usr/bin/env node

const ConfigManager = require('./src/config');
const NodeClient = require('./src/nodeClient');
const JobWorker = require('./src/jobWorker');

async function runWorker() {
  console.log('🚀 Starting LLM Job Worker Node...\n');
  
  try {
    // Initialize configuration
    const configManager = new ConfigManager();
    
    // Get or create node configuration
    const config = configManager.getOrCreateConfig();
    if (!config.nodeId) {
      console.error('❌ Node not configured. Please run setup first.');
      console.error('   Run: node setup.js');
      process.exit(1);
    }
    
    // Add configDir to config object (needed by JobWorker)
    config.configDir = configManager.configDir;
    
    console.log(`📋 Node ID: ${config.nodeId}`);
    console.log(`🔑 Public Key: ${config.publicKey.substring(0, 20)}...`);
    console.log(`🌐 Server URL: ${config.serverUrl}\n`);
    
    // Create node client (NodeClient expects a config object, not just serverUrl)
    const nodeClient = new NodeClient(config);
    
    // Create and initialize job worker
    const jobWorker = new JobWorker(nodeClient, config);
    
    console.log('🔧 Initializing worker...');
    const capabilities = await jobWorker.initialize();
    console.log('✅ Worker initialized');
    console.log(`💪 Capabilities: ${capabilities.gpuAvailable ? 'GPU' : 'CPU'}, ${capabilities.totalMemory} memory\n`);
    
    // Register event handlers
    jobWorker.on('job:assigned', (job) => {
      console.log(`📥 Job assigned: ${job.id}`);
      console.log(`   Model: ${job.model}`);
      console.log(`   Prompt: ${job.prompt.substring(0, 50)}...`);
    });
    
    jobWorker.on('job:started', (jobId) => {
      console.log(`▶️  Job started: ${jobId}`);
    });
    
    jobWorker.on('job:chunk', (jobId, chunk) => {
      process.stdout.write('.');
    });
    
    jobWorker.on('job:completed', (jobId, result) => {
      console.log(`\n✅ Job completed: ${jobId}`);
      console.log(`   Result length: ${result.length} characters`);
    });
    
    jobWorker.on('job:failed', (jobId, error) => {
      console.log(`\n❌ Job failed: ${jobId}`);
      console.log(`   Error: ${error}`);
    });
    
    jobWorker.on('polling', () => {
      console.log('🔍 Polling for jobs...');
    });
    
    jobWorker.on('error', (error) => {
      console.error('❌ Worker error:', error);
      if (error.details) {
        console.error('   Details:', error.details);
      }
    });
    
    // Start the worker
    console.log('🏃 Starting worker...\n');
    await jobWorker.start();
    console.log('✅ Worker is running and polling for jobs\n');
    console.log('Press Ctrl+C to stop\n');
    
    // Handle graceful shutdown
    process.on('SIGINT', async () => {
      console.log('\n\n🛑 Shutting down worker...');
      await jobWorker.stop();
      console.log('👋 Worker stopped. Goodbye!');
      process.exit(0);
    });
    
    process.on('SIGTERM', async () => {
      console.log('\n\n🛑 Shutting down worker...');
      await jobWorker.stop();
      process.exit(0);
    });
    
  } catch (error) {
    console.error('❌ Failed to start worker:', error);
    process.exit(1);
  }
}

// Run the worker
runWorker().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});