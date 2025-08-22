### Phase 2: Node Client ✅

Build a Node.js CLI client that connects to the tracking server.

**Core Features:**
- [x] Generate ED25519 keypair on first run
- [x] Use public key fingerprint as short node ID
- [x] Store keypair securely in local config (~/.llmjob/config.json)
- [x] Generate claim URL with public key and name
- [x] Display both full URL and short link for claiming
- [x] Sign each ping with private key
- [x] Send ping to server every 10 minutes
- [x] Start pinging immediately (no user association required)
- [x] Implement retry logic for failed connections
- [x] Package as installable npm module

**Testing Requirements:**
- [x] Set up Jest for CLI testing
- [x] Test keypair generation and storage
- [x] Test config file creation and reading
- [x] Test URL generation with correct parameters
- [x] Test ping signature generation
- [x] Test retry logic with mocked network failures
- [x] Test CLI argument parsing
- [x] Mock filesystem operations for testing
- [x] Test timer/interval behavior
- [x] Achieve >95% code coverage (100% line coverage achieved)

### Phase 3: Frontend Integration ✅

Connect the cluster dashboard to real backend data.

**Updates Completed:**
- [x] Create /add-node page for claiming nodes via URL
- [x] Handle not-logged-in users:
  - [x] Save claim intent to sessionStorage before redirect
  - [x] Redirect to Clerk sign-in with return URL preserved
  - [x] Auto-complete claim after successful sign in/up
  - [x] Clear pending claim from sessionStorage after success
- [x] Show confirmation dialog when claiming a node
- [x] Replace mock data with API calls in cluster page
- [x] Fetch user's nodes list on page load
- [x] Add toggle switch for public/private mode per node
- [x] Show actual online/offline status
- [x] Display short fingerprints as node IDs (e.g., "2891f7")
- [x] Show node names
- [x] Add "How to add a node" instructions with npm install command
- [x] Auto-refresh node status every 30 seconds

**Testing Requirements:** (Frontend tests pending - functional testing completed)
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