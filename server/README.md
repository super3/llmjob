# LLMJob Node Tracking Server

Express API server for tracking distributed LLM nodes with Redis storage.

## Features

- Node registration with public key authentication
- Signature-based ping verification  
- Public/private node visibility
- Automatic offline detection (15 minutes)
- Node persistence for 7 days (allowing recovery from extended downtime)
- Clerk JWT authentication for users
- OpenAPI-compatible endpoints

## Setup

### Local Development

1. Install dependencies:
```bash
npm install
```

2. Set up Redis locally:
```bash
docker run -p 6379:6379 redis:alpine
```

3. Create `.env` file with the following variables:
```bash
# Server Configuration
PORT=3001

# Redis Configuration
REDIS_URL=redis://localhost:6379

# Clerk Configuration (get from Clerk dashboard)
CLERK_PUBLISHABLE_KEY=your_clerk_publishable_key
CLERK_SECRET_KEY=your_clerk_secret_key
```

4. Start development server:
```bash
npm run dev
```

### Testing

Run tests with coverage:
```bash
npm test
```

Watch mode for development:
```bash
npm run test:watch
```

## API Endpoints

### Authentication Required (Clerk)

- `POST /api/nodes/claim` - Associate node with user account
- `GET /api/nodes` - Get user's nodes (incl. dashboard telemetry: device, VRAM, served model/quant, throughput, uptime)
- `PUT /api/nodes/:id/visibility` - Toggle public/private
- `POST /api/keys` - Create an API key (raw secret returned **once**)
- `GET /api/keys` - List the user's API keys (redacted, with usage + last-used)
- `DELETE /api/keys/:id` - Revoke an API key
- `GET /api/logs` - Recent request logs plus a 24-bucket activity histogram
- `GET /api/nodes/join-token` - Get the user's reusable node join token (created on first use)
- `POST /api/nodes/join-token/rotate` - Rotate the join token, invalidating the old one

### No Clerk Authentication

- `GET /api/nodes/public` - Get all public nodes
- `POST /api/nodes/ping` - Node status update (node signature required)
- `POST /api/nodes/join` - Self-register a node with a **join token** (used by the `install.sh` installer); attaches the node to the token owner's account
- `POST /api/usage` - Record a completed generation (LLMJob **API key** required); writes a request log entry and bills the key's token usage

### Node join flow

The dashboard's "Add node" dialog shows a one-line installer:

```bash
curl -fsSL <base>/install.sh | sh -s -- --server <base> --token <join-token>
```

`install.sh` installs the `llmjob-node` client and runs `llmjob-node join --token …`,
which generates the node's keypair **locally** (only the public key is sent) and calls
`POST /api/nodes/join`. The join token authorizes the claim without an interactive
login; rotate it from the dashboard to revoke outstanding installers.

### API key authentication

API keys (`lj-live-…`) authenticate OpenAI-compatible / usage requests via the
`Authorization: Bearer <key>` header. Only a SHA-256 hash of each key is stored,
so the raw secret is shown exactly once at creation and is unrecoverable.

## Deployment

### Railway

1. Install Railway CLI:
```bash
npm install -g @railway/cli
```

2. Login and link project:
```bash
railway login
railway link
```

3. Deploy:
```bash
railway up
```

### Environment Variables

Required for production:

- `CLERK_PUBLISHABLE_KEY` - From Clerk dashboard
- `CLERK_SECRET_KEY` - From Clerk dashboard  
- `REDIS_URL` - Automatically provided by Railway
- `PORT` - Automatically provided by Railway

## Architecture

- Express.js server with CORS
- Redis for data storage with intelligent TTL management
- ED25519 signatures for node authentication
- Clerk for user authentication
- Jest + Supertest for testing