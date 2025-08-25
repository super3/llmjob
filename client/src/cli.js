#!/usr/bin/env node

const { program } = require('commander');
const chalk = require('chalk');
const ConfigManager = require('./config');
const NodeClient = require('./nodeClient');
const OllamaClient = require('./ollama');
const packageJson = require('../package.json');

// Support test environment config directory
const configDir = process.env.LLMJOB_CONFIG_DIR || null;
const configManager = new ConfigManager(configDir);

program
  .name('llmjob-node')
  .description('LLMJob Node Client - Connect your compute resources to the LLMJob network')
  .version(packageJson.version);

program
  .command('start')
  .description('Start the node client and begin pinging the server')
  .option('-i, --interval <minutes>', 'Ping interval in minutes', '5')
  .option('-n, --name <name>', 'Node name for claiming')
  .action((options) => {
    const config = configManager.getOrCreateConfig();
    const client = new NodeClient(config);
    
    console.log(chalk.cyan('═══════════════════════════════════════════════════════════'));
    console.log(chalk.cyan.bold('           LLMJob Node Client Started'));
    console.log(chalk.cyan('═══════════════════════════════════════════════════════════'));
    console.log();
    console.log(chalk.white('Node ID:'), chalk.yellow(config.nodeId));
    console.log(chalk.white('Server:'), chalk.gray(config.serverUrl));
    console.log();
    
    // Generate claim URL
    const url = client.generateClaimUrl(options.name);
    console.log(chalk.green.bold('✨ Claim your node:'));
    console.log(chalk.white('   '), chalk.blue.underline(url.full));
    console.log();
    console.log(chalk.gray('Visit the URL above to associate this node with your account'));
    console.log(chalk.cyan('───────────────────────────────────────────────────────────'));
    console.log();
    
    const intervalMs = parseInt(options.interval) * 60 * 1000;
    console.log(chalk.white(`Pinging server every ${options.interval} minutes...`));
    console.log();
    
    const intervalId = client.startPinging(intervalMs, (result) => {
      const timestamp = new Date().toLocaleTimeString();
      
      if (result.success) {
        if (result.data.message && result.data.message.includes('not found')) {
          console.log(chalk.yellow(`[${timestamp}] ⚠ Node not claimed yet (attempt ${result.attempt})`));
        } else {
          console.log(chalk.green(`[${timestamp}] ✓ Ping successful (attempt ${result.attempt})`));
        }
      } else {
        console.log(chalk.red(`[${timestamp}] ✗ Ping failed: ${result.error} (attempt ${result.attempt})`));
      }
    });
    
    // Handle graceful shutdown
    process.on('SIGINT', () => {
      console.log();
      console.log(chalk.yellow('Shutting down...'));
      client.stopPinging(intervalId);
      process.exit(0);
    });
    
    process.on('SIGTERM', () => {
      client.stopPinging(intervalId);
      process.exit(0);
    });
  });

program
  .command('info')
  .description('Display node information and claim URLs')
  .action(() => {
    const config = configManager.getOrCreateConfig();
    const client = new NodeClient(config);
    
    console.log(chalk.cyan('═══════════════════════════════════════════════════════════'));
    console.log(chalk.cyan.bold('           LLMJob Node Information'));
    console.log(chalk.cyan('═══════════════════════════════════════════════════════════'));
    console.log();
    console.log(chalk.white('Node ID:'), chalk.yellow(config.nodeId));
    console.log(chalk.white('Public Key:'), chalk.gray(config.publicKey.substring(0, 20) + '...'));
    console.log(chalk.white('Server:'), chalk.gray(config.serverUrl));
    console.log(chalk.white('Created:'), chalk.gray(new Date(config.createdAt).toLocaleString()));
    console.log();
    
    const url = client.generateClaimUrl();
    console.log(chalk.green.bold('Claim URL:'));
    console.log(chalk.white('  '), chalk.blue.underline(url.full));
    console.log();
  });

program
  .command('reset')
  .description('Reset node configuration (generates new keypair)')
  .option('-f, --force', 'Force reset without confirmation')
  .action(async (options) => {
    if (!options.force) {
      console.log(chalk.yellow('⚠ Warning: This will generate a new keypair and node ID.'));
      console.log(chalk.yellow('  You will need to reclaim the node if it was previously claimed.'));
      console.log();
      console.log(chalk.gray('Use --force to skip this confirmation.'));
      process.exit(1);
    }
    
    configManager.deleteConfig();
    const config = configManager.getOrCreateConfig();
    
    console.log(chalk.green('✓ Node configuration reset successfully'));
    console.log(chalk.white('New Node ID:'), chalk.yellow(config.nodeId));
  });

program
  .command('config')
  .description('Display configuration file location')
  .action(() => {
    console.log(chalk.white('Config file:'), chalk.gray(configManager.configFile));
  });

program
  .command('ollama')
  .description('Manage Ollama integration for LLM inference')
  .option('--init', 'Initialize Ollama (detect hardware, install, pull model)')
  .option('--capabilities', 'Show hardware capabilities')
  .option('--status', 'Check Ollama service status')
  .option('--test', 'Test inference with a simple prompt')
  .option('--benchmark', 'Run inference benchmark')
  .option('--pull [model]', 'Pull a specific model (default: llama3.2:3b)')
  .action(async (options) => {
    const ollama = new OllamaClient(configManager.configDir);
    
    try {
      if (options.init) {
        console.log(chalk.cyan('═══════════════════════════════════════════════════════════'));
        console.log(chalk.cyan.bold('           Initializing Ollama Integration'));
        console.log(chalk.cyan('═══════════════════════════════════════════════════════════'));
        console.log();
        
        const result = await ollama.initialize();
        
        console.log();
        console.log(chalk.green('✓ Ollama initialized successfully'));
        console.log();
        console.log(chalk.white('Hardware:'));
        console.log(chalk.gray(`  CPU: ${result.capabilities.cpu.cores} cores - ${result.capabilities.cpu.model}`));
        console.log(chalk.gray(`  RAM: ${result.capabilities.memory.total} GB`));
        console.log(chalk.gray(`  GPU: ${result.capabilities.gpu.model}`));
      } else if (options.capabilities) {
        const capabilities = await ollama.loadCapabilities();
        
        console.log(chalk.cyan('═══════════════════════════════════════════════════════════'));
        console.log(chalk.cyan.bold('           Hardware Capabilities'));
        console.log(chalk.cyan('═══════════════════════════════════════════════════════════'));
        console.log();
        console.log(chalk.white('CPU:'));
        console.log(chalk.gray(`  Cores: ${capabilities.cpu.cores}`));
        console.log(chalk.gray(`  Model: ${capabilities.cpu.model}`));
        console.log(chalk.gray(`  Speed: ${capabilities.cpu.speed} MHz`));
        console.log();
        console.log(chalk.white('Memory:'));
        console.log(chalk.gray(`  Total: ${capabilities.memory.total} GB`));
        console.log(chalk.gray(`  Free: ${capabilities.memory.free} GB`));
        console.log();
        console.log(chalk.white('GPU:'));
        console.log(chalk.gray(`  Type: ${capabilities.gpu.type}`));
        console.log(chalk.gray(`  Model: ${capabilities.gpu.model}`));
        console.log(chalk.gray(`  Available: ${capabilities.gpu.available ? 'Yes' : 'No'}`));
        console.log();
        console.log(chalk.white('System:'));
        console.log(chalk.gray(`  Platform: ${capabilities.platform}`));
        console.log(chalk.gray(`  Architecture: ${capabilities.arch}`));
        console.log(chalk.gray(`  Detected: ${capabilities.detectedAt}`));
      } else if (options.status) {
        console.log(chalk.white('Checking Ollama status...'));
        
        const isInstalled = await ollama.isOllamaInstalled();
        const isRunning = await ollama.checkServiceStatus();
        
        console.log();
        console.log(chalk.white('Ollama installed:'), isInstalled ? chalk.green('Yes') : chalk.red('No'));
        console.log(chalk.white('Service running:'), isRunning ? chalk.green('Yes') : chalk.red('No'));
        
        if (isRunning) {
          const version = await ollama.getVersion();
          console.log(chalk.white('Version:'), chalk.gray(version.version));
          
          const models = await ollama.listModels();
          console.log(chalk.white('Models:'), models.length > 0 ? chalk.gray(models.map(m => m.name).join(', ')) : chalk.yellow('None'));
        }
      } else if (options.test) {
        console.log(chalk.white('Testing inference...'));
        console.log();
        
        const result = await ollama.testInference('What is 2+2? Please answer with just the number.');
        
        console.log();
        console.log();
        console.log(chalk.green('✓ Inference test completed'));
        console.log(chalk.gray(`  Tokens/sec: ${result.tokensPerSecond}`));
        console.log(chalk.gray(`  Duration: ${result.duration}s`));
        console.log(chalk.gray(`  Token count: ${result.tokenCount}`));
      } else if (options.benchmark) {
        const result = await ollama.benchmarkInference();
        
        console.log();
        console.log(chalk.green('✓ Benchmark completed'));
        console.log(chalk.white('Average performance:'), chalk.yellow(`${result.averageTokensPerSecond} tokens/sec`));
      } else if (options.pull) {
        const model = typeof options.pull === 'string' ? options.pull : 'llama3.2:3b';
        await ollama.pullModel(model);
      } else {
        // Show help for ollama command
        console.log(chalk.cyan('Ollama Integration Commands:'));
        console.log();
        console.log(chalk.white('  --init         '), chalk.gray('Initialize Ollama (detect hardware, install, pull model)'));
        console.log(chalk.white('  --capabilities '), chalk.gray('Show hardware capabilities'));
        console.log(chalk.white('  --status       '), chalk.gray('Check Ollama service status'));
        console.log(chalk.white('  --test         '), chalk.gray('Test inference with a simple prompt'));
        console.log(chalk.white('  --benchmark    '), chalk.gray('Run inference benchmark'));
        console.log(chalk.white('  --pull [model] '), chalk.gray('Pull a specific model (default: llama3.2:3b)'));
      }
    } catch (error) {
      console.error(chalk.red('Error:'), error.message);
      process.exit(1);
    }
  });

program.parse(process.argv);

// Show help if no command provided
if (!process.argv.slice(2).length) {
  program.outputHelp();
}