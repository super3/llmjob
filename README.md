# LLMJob

Build your own AI infrastructure with spare GPUs and devices. Get OpenAI-compatible API access and monetize excess capacity when idle.

[![Test Status](https://img.shields.io/github/actions/workflow/status/super3/llmjob/test.yml?branch=main&label=tests)](https://github.com/super3/llmjob/actions/workflows/test.yml)
[![Deploy Status](https://img.shields.io/website?url=https%3A%2F%2Fllmjob-production.up.railway.app%2Fhealth&label=deploy&up_message=live&down_message=down)](https://llmjob-production.up.railway.app)
[![Coverage Status](https://coveralls.io/repos/github/super3/llmjob/badge.svg?branch=main)](https://coveralls.io/github/super3/llmjob?branch=main)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?label=license)](https://github.com/super3/llmjob/blob/main/LICENSE)

## Features

- ⚡ Pool office workstations and spare hardware into a private AI cluster
- 🤖 Fully OpenAI-compatible API endpoints  
- 📊 Real-time cluster monitoring dashboard
- 🔒 Keep sensitive data on your own infrastructure

This repository contains two packages:

- **Server** (repo root) — the Express API plus the static dashboard pages,
  backed by Postgres and deployed to Railway / GitHub Pages.
- **Node client** ([`client/`](client)) — the `llmjob-node` worker that runs on
  a machine with a GPU and processes jobs via Ollama.

## Add a node

The quickest way to connect a machine is the one-line installer (a pure-shell
agent — no Node or npm required). Grab your personalized command, which bakes in
the server URL and join token, from the **Add Node** page in the dashboard:

```bash
curl -fsSL https://llmjob-production.up.railway.app/install.sh/<token> | bash
```

Prefer the Node.js client? See [`client/README.md`](client/README.md):

```bash
npm install -g llmjob-node
llmjob-node start
```

## Running the server

```bash
git clone https://github.com/super3/llmjob.git && cd llmjob
npm install                        # Install dependencies

npm start                          # Apply migrations, then start the server (default port 3001)
npm run dev                        # Same, with auto-reload (development)
npm test                           # Run test suite with coverage
npm run test:watch                 # Run tests in watch mode
```

The server requires a `DATABASE_URL` pointing at Postgres; migrations in
[`migrations/`](migrations) are applied automatically by `npm start` / `npm run dev`.