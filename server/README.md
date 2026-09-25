# LLMJob Server

Express API behind LLMJob Earn, with Postgres storage. It serves three things:

- **The miners board** — Earn clients check in once a minute
  (`POST /api/miners/ping`) and the public network page reads them back
  (`GET /api/miners`), grouped into one row per rig.
- **The managed-mining waitlist** — the signup form on `/managed` posts to
  `POST /api/waitlist`.
- **The static site** — the pages built from `site/` into `dist/`.

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
```

4. Start the server (builds the site and applies migrations first):
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

The tables from the LLM era (`nodes`, `jobs`, `api_keys`, `request_logs`,
`chat_*`, …) are still created by the migration history but nothing reads or
writes them any more. Dropping them is a deliberate, separate migration.

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

All public, no authentication.

- `POST /api/miners/ping` — a mining client reports its live status: payout
  `address` (required, `prl1p…`), `worker`, `gpu`, `region`, `hashrate`,
  `accepted`, `vramUsedMb`, `vramTotalMb`, `version`.
- `GET /api/miners` — online rigs for the network page, one row per host with
  its cards nested, plus `totalOnline`, `totalWorkers` and `totalHashrate`.
- `POST /api/waitlist` — join the managed-mining waitlist: `email` (required),
  `gpus`, `note`, `source`. Signing up again with the same email updates the
  existing row.
- `GET /health` — liveness check.

Signups land in the `managed_waitlist` table; read them straight from Postgres
(for example from the Railway database console).

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

- `DATABASE_URL` - Automatically provided by Railway (Postgres plugin)
- `PORT` - Automatically provided by Railway
- `PGPOOL_MAX`, `PGPOOL_CONNECT_TIMEOUT_MS` - Optional pool sizing overrides

The Clerk and OpenRouter variables the LLM features used are no longer read and
can be removed from the Railway service.

## Architecture

- Express.js server with CORS restricted to our own origins
- Postgres for data storage, schema managed by node-pg-migrate
