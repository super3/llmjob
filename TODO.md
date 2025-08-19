### Phase 1: Node Tracking Server

Deploy an Express API to Railway with Redis for node tracking.

**API Endpoints:**
- [ ] `POST /api/nodes/claim` - Associate node with logged-in user (via Clerk JWT)
- [ ] `POST /api/nodes/ping` - Update node status (verify signature)
- [ ] `GET /api/nodes` - Return all nodes for authenticated user
- [ ] `GET /api/nodes/public` - Return all public nodes (no auth required)
- [ ] `PUT /api/nodes/:id/visibility` - Toggle node between public/private

**Setup Tasks:**
- [ ] Create Express server with CORS enabled
- [ ] Set up Railway project with Redis service
- [ ] Implement 15-minute TTL for auto-marking nodes offline
- [ ] Add Clerk JWT verification for user authentication
- [ ] Add signature verification for pings
- [ ] Store public keys as node IDs with user association
- [ ] Add `isPublic` boolean field (default: false) to node data
- [ ] Deploy to Railway with GitHub integration

**Testing Requirements:**
- [ ] Set up Jest with supertest for API testing
- [ ] Test all API endpoints with valid/invalid inputs
- [ ] Test Clerk JWT verification with mocked tokens
- [ ] Test signature verification for pings
- [ ] Test Redis operations (use redis-mock)
- [ ] Test TTL expiration for offline nodes
- [ ] Test error handling and edge cases
- [ ] Achieve >95% code coverage

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
- [ ] Start pinging immediately (no user association required)
- [ ] Implement retry logic for failed connections
- [ ] Package as installable npm module

**Testing Requirements:**
- [ ] Set up Jest for CLI testing
- [ ] Test keypair generation and storage
- [ ] Test config file creation and reading
- [ ] Test URL generation with correct parameters
- [ ] Test ping signature generation
- [ ] Test retry logic with mocked network failures
- [ ] Test CLI argument parsing
- [ ] Mock filesystem operations for testing
- [ ] Test timer/interval behavior
- [ ] Achieve >95% code coverage

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
- [ ] Add toggle switch for public/private mode per node
- [ ] Update nodes.html to show public nodes from API
- [ ] Show actual online/offline status
- [ ] Display short fingerprints as node IDs (e.g., "4f2a8b")
- [ ] Show node names
- [ ] Add "How to add a node" instructions with npm install command
- [ ] Auto-refresh node status every 30 seconds

**Testing Requirements:**
- [ ] Set up Jest with Testing Library for frontend tests
- [ ] Test add-node page flow with/without auth
- [ ] Test sessionStorage handling for pending claims
- [ ] Test API call error handling
- [ ] Test auto-refresh functionality with timers
- [ ] Test public/private toggle functionality
- [ ] Mock Clerk authentication states
- [ ] Mock fetch calls to backend API
- [ ] Test UI state changes (loading, error, success)
- [ ] Achieve >90% code coverage (excluding Clerk SDK)