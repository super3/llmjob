# LLMJob

A free, one-click Pearl (PRL) miner for NVIDIA GPUs. Download it, paste a payout
address, hit **Start**. No command line, no account, no fee.

[![Test Status](https://img.shields.io/github/actions/workflow/status/super3/llmjob/test.yml?branch=main&label=tests)](https://github.com/super3/llmjob/actions/workflows/test.yml)
[![Deploy Status](https://img.shields.io/website?url=https%3A%2F%2Fllmjob-production.up.railway.app%2Fhealth&label=deploy&up_message=live&down_message=down)](https://llmjob-production.up.railway.app)
[![Coverage Status](https://coveralls.io/repos/github/super3/llmjob/badge.svg?branch=main)](https://coveralls.io/github/super3/llmjob?branch=main)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?label=license)](https://github.com/super3/llmjob/blob/main/LICENSE)

## Features

- ⛏️ Our own CUDA miner for Pearl, built into the app. Nothing third-party to download
- 🖱️ One click: paste a `prl1p…` address and hit Start, every GPU is detected automatically
- 💸 No dev fee, no pool fee, payouts straight to your wallet
- 📡 A public board of every rig mining with LLMJob, live
- 🛠️ Coming next: managed mining, where we tune and maintain your rigs around the clock

This repository contains two packages:

- **Server** (repo root) — the Express API behind the miners board and the
  managed-mining waitlist, plus the static site, backed by Postgres and deployed
  to Railway / GitHub Pages. See [`server/README.md`](server/README.md).
- **LLMJob Earn** ([`earn/`](earn)) — the desktop app (Windows / Linux) and the
  headless CLI that run our Pearl miner. See [`earn/README.md`](earn/README.md).

## Run LLMJob Earn

Download the latest installer from the
[releases page](https://github.com/super3/llmjob/releases/latest), install, paste
a Pearl (`prl1p…`) payout address, and hit **Start**. To run it from source:

```bash
cd earn
npm install
npm start                          # launch the Electron app
```

See [`earn/README.md`](earn/README.md) for the miner, merge mining, the live
balance, and building the installer.

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
[`server/migrations/`](server/migrations) are applied automatically by `npm start` / `npm run dev`.
