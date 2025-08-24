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

### Authentication Required

- `POST /api/nodes/claim` - Associate node with user account
- `GET /api/nodes` - Get user's nodes
- `PUT /api/nodes/:id/visibility` - Toggle public/private

### No Authentication

- `GET /api/nodes/public` - Get all public nodes
- `POST /api/nodes/ping` - Node status update (signature required)

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