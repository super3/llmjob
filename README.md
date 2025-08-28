# LLMJob

Build your own AI infrastructure with spare GPUs and devices. Get OpenAI-compatible API access and monetize excess capacity when idle.

[![Test Status](https://img.shields.io/github/actions/workflow/status/super3/llmjob/test.yml?branch=main&label=tests)](https://github.com/super3/llmjob/actions/workflows/test.yml)
[![Deploy Status](https://img.shields.io/github/actions/workflow/status/super3/llmjob/railway-deploy.yml?branch=main&label=deploy)](https://github.com/super3/llmjob/actions/workflows/railway-deploy.yml)
[![Coverage Status](https://coveralls.io/repos/github/super3/llmjob/badge.svg?branch=main)](https://coveralls.io/github/super3/llmjob?branch=main)
[![npm version](https://badge.fury.io/js/llmjob.svg)](https://www.npmjs.com/package/llmjob)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?label=license)](https://github.com/super3/llmjob/blob/main/LICENSE)

## Features

- ⚡ Pool office workstations and spare hardware into a private AI cluster
- 🤖 Fully OpenAI-compatible API endpoints  
- 📊 Real-time cluster monitoring dashboard
- 🔒 Keep sensitive data on your own infrastructure

## Installation

### npm (Recommended)
```bash
npm install -g llmjob
```

### From Source
```bash
git clone https://github.com/super3/llmjob.git && cd llmjob
npm install && npm link            # Install dependencies and llmjob command globally
```

## Usage

```bash
npm start                          # Start the server (default port 3001)
npm run dev                        # Start with auto-reload (development)
npm test                           # Run test suite
npm run test:watch                 # Run tests in watch mode
```