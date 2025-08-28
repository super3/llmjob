# LLMJob

A distributed job processing system for AI inference workloads, enabling decentralized compute networks.

[![npm version](https://badge.fury.io/js/llmjob.svg)](https://www.npmjs.com/package/llmjob)
[![Test Status](https://img.shields.io/github/actions/workflow/status/super3/llmjob/test.yml?branch=main&label=tests)](https://github.com/super3/llmjob/actions/workflows/test.yml)
[![Coverage Status](https://coveralls.io/repos/github/super3/llmjob/badge.svg?branch=main)](https://coveralls.io/github/super3/llmjob?branch=main)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?label=license)](https://github.com/super3/llmjob/blob/main/LICENSE)

## Features

- ⚡ High-performance distributed job processing
- 🤖 OpenAI-compatible API endpoints
- 📊 Real-time cluster monitoring dashboard
- 🔄 Automatic job timeout and retry handling
- 🔐 Secure authentication with Clerk
- 💾 Redis-backed persistent storage
- 🌐 RESTful API for job submission and management
- 📈 Node status tracking and health checks

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