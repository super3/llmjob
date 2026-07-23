# LLMJob Node Tracking Server

Express API server for tracking distributed LLM nodes with Postgres storage.

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

2. Set up Postgres locally:
```bash
docker run -p 5432:5432 -e POSTGRES_DB=llmjob -e POSTGRES_HOST_AUTH_METHOD=trust postgres:16
```

3. Create `.env` file with the following variables:
```bash
# Server Configuration
PORT=3001

# Postgres Configuration
DATABASE_URL=postgres://localhost:5432/llmjob

# Clerk Configuration (get from Clerk dashboard)
CLERK_PUBLISHABLE_KEY=your_clerk_publishable_key
CLERK_SECRET_KEY=your_clerk_secret_key
```

4. Run database migrations:
```bash
npm run migrate:up
```

5. Start development server (also runs migrations):
```bash
npm run dev
```

### Database migrations

Schema is managed with [node-pg-migrate](https://github.com/salsita/node-pg-migrate).
Migrations live in `server/migrations/`; `npm start` / `npm run dev` apply them automatically.

```bash
npm run migrate:up                 # apply pending migrations
npm run migrate:down               # roll back the last migration
npm run migrate:create my_change   # scaffold a new migration (CJS)
```

Tests run against an in-memory Postgres (pg-mem) using the same schema, so no
database is required for `npm test`.

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
- `POST /api/nodes/join` - Self-register a node with a **join token** (used by the node agent); attaches the node to the token owner's account
- `POST /v1/chat/completions` - OpenAI-compatible chat completions (LLMJob **API key** required); turns the request into an inference job served by an online node, streams if `stream: true`, and bills the key's token usage
- `POST /api/usage` - Record a completed generation (LLMJob **API key** required); writes a request log entry and bills the key's token usage. Used by clients that run inference elsewhere and only report usage — the dashboard advertises `/v1/chat/completions` as the endpoint to call

#### Free web chat (OpenRouter proxy)

Powers the public **Chat** page (`chat.html`). No auth — this is the "open
usage" front door, so the OpenRouter API key stays server-side and every request
is gated by a global free-token budget instead of per-user auth. Prompts are
**never stored**; only performance (latency, time-to-first-token, tok/s) and
token counts are recorded, plus a running lifetime total used for the cap and the
"tokens served" display. Once free chat is proven out this can be repointed at
the LLMJob node network.

- `POST /api/chat/completions` - Proxy a chat to OpenRouter. Streams a small SSE
  protocol by default (`data: {"delta":…}`, then `data: {"done":true,"meta":…}`,
  then `data: [DONE]`); pass `{"stream": false}` for a single JSON body. Returns
  `402` once the free-token budget is spent and `503` when no OpenRouter key is
  configured. Only allow-listed models are reachable, and `max_tokens` / prompt
  length are clamped server-side.
- `GET /api/chat/models` - The allow-listed models (`{ id, label }`) the Chat UI
  may offer.
- `GET /api/chat/usage` - Running token totals plus remaining free budget.

### Node join flow

A machine links to an account with a **join token** (created per user, rotatable
from the dashboard). The dashboard's "Add node" dialog shows the token and the
command to run it with the headless CLI:

```bash
llmjob-earn-cli connect --token <join-token>
```

The client — the **LLMJob Earn** desktop app (`earn/`, API → Connect tab) or its
headless CLI (`llmjob-earn-cli connect`) — creates an Ed25519 key **locally**
(only the public key is sent), calls `POST /api/nodes/join` with the token to
claim the node, then pings `POST /api/nodes/ping` on an interval so it shows as
online. The join token authorizes the claim without an interactive login; rotate
it from the dashboard to revoke outstanding agents.

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
- `DATABASE_URL` - Automatically provided by Railway (Postgres plugin)
- `PORT` - Automatically provided by Railway

Free web chat (OpenRouter proxy) — all optional, with sensible defaults:

- `OPENROUTER_API_KEY` - OpenRouter key for the free Chat page. Without it,
  `POST /api/chat/completions` returns `503` (the rest of the server is
  unaffected).
- `OPENROUTER_MODELS` - JSON array of allow-listed models,
  e.g. `[{"id":"qwen/qwen3-32b","label":"Qwen3 32B"}]`. Defaults to a small
  built-in Qwen list.
- `OPENROUTER_FREE_TOKEN_BUDGET` - Total tokens of free usage before the endpoint
  starts returning `402` (default `1000000`; set `0` to disable the cap).
- `OPENROUTER_MAX_TOKENS` - Per-request completion ceiling (default `1024`).
- `OPENROUTER_BASE_URL` - Override the OpenRouter base URL (default
  `https://openrouter.ai/api/v1`).
- `OPENROUTER_REFERER` - Sent as the `HTTP-Referer` attribution header (default
  `https://llmjob.com`).

## Architecture

- Express.js server with CORS
- Postgres for data storage, schema managed by node-pg-migrate
- ED25519 signatures for node authentication
- Clerk for user authentication
- Jest + Supertest for testing