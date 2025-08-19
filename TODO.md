# TODO

## Backend Development

### 1. Node Tracking Server (Vercel Deployment)
Create serverless API endpoints that manage and track connected nodes in the network.

**Requirements:**
- REST API endpoints for node registration and status updates
- Store node information (ID, name, status, last ping time) in Vercel KV
- Track node online/offline status based on ping interval
- Mark nodes as offline if no ping received within 15 minutes
- Provide endpoint to list all nodes and their current status
- Handle stateless serverless function constraints

**Tech Stack:**
- Node.js (Vercel Serverless Functions)
- Vercel KV (Redis) for storage - required for serverless environment
- API Routes in Next.js or standalone functions

### 2. Node Client
Create a lightweight client that nodes run to connect and report to the server.

**Requirements:**
- Connect to the tracking server on startup
- Send ping to server every 10 minutes with:
  - Node ID (generated on first run)
  - Node name (user configurable)
  - Capabilities (GPU type, VRAM, RAM, etc.)
  - Current status
- Handle connection failures gracefully
- Auto-reconnect if connection is lost
- Simple CLI interface for configuration

**Tech Stack:**
- Node.js
- Axios or fetch for API communication
- JSON configuration file for server URL and node settings