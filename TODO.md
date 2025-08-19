### Phase 1: Node Tracking Server

Deploy an Express API to Railway with Redis for node tracking.

**API Endpoints:**
- [ ] `POST /api/nodes/claim` - Associate node with logged-in user (via Clerk JWT)
- [ ] `POST /api/nodes/ping` - Update node status (verify signature)
- [ ] `GET /api/nodes` - Return all nodes for authenticated user

**Setup Tasks:**
- [ ] Create Express server with CORS enabled
- [ ] Set up Railway project with Redis service
- [ ] Implement 15-minute TTL for auto-marking nodes offline
- [ ] Add Clerk JWT verification for user authentication
- [ ] Add signature verification for pings
- [ ] Store public keys as node IDs with user association
- [ ] Deploy to Railway with GitHub integration

### Phase 2: Node Client

Build a Node.js CLI client that connects to the tracking server.

**Core Features:**
- [ ] Generate ED25519 keypair on first run
- [ ] Use public key fingerprint as short node ID
- [ ] Store keypair securely in local config (~/.llmjob/config.json)
- [ ] Generate claim URL with public key and name
- [ ] Display both full URL and short link for claiming
- [ ] Sign each ping with private key
- [ ] Send ping to server every 10 minutes
- [ ] Include node capabilities in ping (GPU, VRAM, RAM)
- [ ] Start pinging immediately (no user association required)
- [ ] Implement retry logic for failed connections
- [ ] Package as installable npm module

### Phase 3: Frontend Integration

Connect the cluster dashboard to real backend data.

**Updates Needed:**
- [ ] Create /add-node page for claiming nodes via URL
- [ ] Handle not-logged-in users:
  - [ ] Save claim intent to sessionStorage before redirect
  - [ ] Redirect to Clerk sign-in with return URL preserved
  - [ ] Auto-complete claim after successful sign in/up
  - [ ] Clear pending claim from sessionStorage after success
- [ ] Show confirmation dialog when claiming a node
- [ ] Replace mock data with API calls in cluster page
- [ ] Fetch user's nodes list on page load
- [ ] Show actual online/offline status
- [ ] Display short fingerprints as node IDs (e.g., "4f2a8b")
- [ ] Show node names and capabilities
- [ ] Add "How to add a node" instructions with npm install command
- [ ] Auto-refresh node status every 30 seconds