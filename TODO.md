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