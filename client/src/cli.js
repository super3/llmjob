#!/usr/bin/env node

const { program } = require('commander');
const chalk = require('chalk');
const ConfigManager = require('./config');
const NodeClient = require('./nodeClient');
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

program.parse(process.argv);

// Show help if no command provided
if (!process.argv.slice(2).length) {
  program.outputHelp();
}