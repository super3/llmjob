# LLMJob Node Client

A Node.js CLI client for connecting compute resources to the LLMJob network.

## Installation

```bash
npm install -g llmjob-node
```

Or run directly with npx:
```bash
npx llmjob-node
```

## Usage

### Start the node client
```bash
llmjob-node start
```

This will:
- Generate a unique ED25519 keypair on first run
- Display claim URLs to associate the node with your account
- Start pinging the server every 10 minutes
- Automatically retry failed connections

### Options

```bash
llmjob-node start --interval 5 --name "My GPU Node"
```

- `--interval <minutes>`: Set ping interval (default: 10 minutes)
- `--name <name>`: Set node name for claiming

### Other Commands

```bash
# Display node information and claim URLs
llmjob-node info

# Show configuration file location
llmjob-node config

# Reset node (generates new keypair)
llmjob-node reset --force

# Display help
llmjob-node --help
```

## Configuration

The client stores its configuration in `~/.llmjob/config.json` with:
- Node ID (6-character fingerprint)
- ED25519 keypair
- Server URL
- Creation timestamp

## Environment Variables

- `LLMJOB_SERVER_URL`: Override the default server URL
- `LLMJOB_CONFIG_DIR`: Override the config directory location

## How It Works

1. **First Run**: Generates an ED25519 keypair and derives a 6-character node ID
2. **Claiming**: Visit the provided URL to associate the node with your account
3. **Pinging**: Signs each ping with the private key and sends to the server
4. **Retry Logic**: Automatically retries failed pings up to 3 times with delays

## Development

```bash
# Install dependencies
npm install

# Run tests
npm test

# Run with coverage
npm run test:coverage
```

## License

MIT