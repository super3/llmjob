const express = require('express');
const router = express.Router();
const { requireAuth } = require('./middleware/auth');
const { requireAdmin } = require('./middleware/admin');
const { verifySignature } = require('./middleware/signature');
const { apiKeyAuth } = require('./middleware/apiKeyAuth');
const nodeController = require('./controllers/nodeController');
const minerController = require('./controllers/minerController');
const apiKeyController = require('./controllers/apiKeyController');
const logController = require('./controllers/logController');
const JobController = require('./controllers/jobController');
const OpenAiController = require('./controllers/openaiController');
const ChatController = require('./controllers/chatController');
const JobService = require('./services/jobService');
const NodeService = require('./services/nodeService');

// POST /api/nodes/claim - Associate node with logged-in user (requires auth)
router.post('/nodes/claim', requireAuth, nodeController.claimNode);

// POST /api/nodes/ping - Update node status (requires signature verification)
router.post('/nodes/ping', verifySignature, nodeController.pingNode);

// GET /api/nodes - Return all nodes for authenticated user
router.get('/nodes', requireAuth, nodeController.getUserNodes);

// GET /api/nodes/public - Return all public nodes (no auth required)
router.get('/nodes/public', nodeController.getPublicNodes);

// PUT /api/nodes/:id/visibility - Toggle node between public/private
router.put('/nodes/:id/visibility', requireAuth, nodeController.updateNodeVisibility);

// GET /api/nodes/join-token - Get the user's reusable join token (dashboard)
router.get('/nodes/join-token', requireAuth, nodeController.getJoinToken);

// POST /api/nodes/join-token/rotate - Rotate the user's join token
router.post('/nodes/join-token/rotate', requireAuth, nodeController.rotateJoinToken);

// POST /api/nodes/join - Self-register a node with a join token (install script)
router.post('/nodes/join', nodeController.joinNode);

// Miners (LLMJob Earn crypto clients) — public, no auth.
// POST /api/miners/ping - A mining client reports its live status
router.post('/miners/ping', minerController.pingMiner);
// GET /api/miners - Online miners grouped by address (network page)
router.get('/miners', minerController.getPublicMiners);

// API keys (dashboard, Clerk auth)
// POST /api/keys - Create a new API key (raw secret returned once)
router.post('/keys', requireAuth, apiKeyController.createKey);
// GET /api/keys - List the user's API keys (redacted)
router.get('/keys', requireAuth, apiKeyController.listKeys);
// PUT /api/keys/:id/visibility - toggle a key's request routing (public/private)
router.put('/keys/:id/visibility', requireAuth, apiKeyController.updateKeyVisibility);
// DELETE /api/keys/:id - Revoke an API key
router.delete('/keys/:id', requireAuth, apiKeyController.revokeKey);

// Logs (dashboard, Clerk auth)
// GET /api/logs - Recent request logs plus the activity histogram
router.get('/logs', requireAuth, logController.getLogs);

// Usage (API-key auth) - record a completed generation: writes a log entry and
// bills the key's token usage.
router.post('/usage', apiKeyAuth, logController.recordUsage);

// Initialize job controller with dependencies
const initJobRoutes = (db) => {
  const jobService = new JobService(db);
  const nodeService = new NodeService(db);
  const jobController = new JobController(jobService, nodeService);

  // Job submission and management
  router.post('/jobs', requireAuth, (req, res) => jobController.submitJob(req, res));
  router.get('/jobs/stats', requireAuth, (req, res) => jobController.getStats(req, res));
  router.get('/jobs/:jobId', (req, res) => jobController.getJob(req, res));
  
  // Node job operations (require signature verification)
  router.post('/jobs/poll', verifySignature, (req, res) => jobController.pollJobs(req, res));
  router.post('/jobs/:jobId/heartbeat', verifySignature, (req, res) => jobController.heartbeat(req, res));
  router.post('/jobs/:jobId/chunks', verifySignature, (req, res) => jobController.receiveChunk(req, res));
  router.post('/jobs/:jobId/complete', verifySignature, (req, res) => jobController.completeJob(req, res));
  router.post('/jobs/:jobId/fail', verifySignature, (req, res) => jobController.failJob(req, res));
  
  // Admin operations. Cleanup deletes data, so it is gated to admins
  // (ADMIN_USER_IDS). check-timeouts mutates queue state; the server also runs
  // it on an internal interval, so the HTTP route just needs to be authenticated
  // rather than public.
  router.post('/jobs/cleanup', requireAuth, requireAdmin, (req, res) => jobController.cleanupJobs(req, res));
  router.post('/jobs/check-timeouts', requireAuth, (req, res) => jobController.checkTimeouts(req, res));
};

// OpenAI-compatible gateway. Mounted at the app root (not under /api) so callers
// point any OpenAI SDK at `https://<host>/v1`. API-key auth; each request becomes
// an inference job served by an online node. `opts` (poll cadence / timeout) is
// injectable for tests.
const initOpenAiRoutes = (app, opts) => {
  const ctrl = new OpenAiController(opts || {});
  app.post('/v1/chat/completions', apiKeyAuth, (req, res) => ctrl.chatCompletions(req, res));
  return ctrl;
};

// Free public web-chat gateway (chat.html), proxied to OpenRouter. No API key —
// this is the "open usage" front door, gated by a global free-token budget in
// the controller rather than per-user auth. Mounted at the app root. `opts`
// (OpenRouter key/models/budget, fetch) is injectable for tests.
const initChatRoutes = (app, opts) => {
  const ctrl = new ChatController(opts || {});
  app.post('/api/chat/completions', (req, res) => ctrl.chatCompletions(req, res));
  app.get('/api/chat/models', (req, res) => ctrl.listModels(req, res));
  app.get('/api/chat/usage', (req, res) => ctrl.usage(req, res));
  return ctrl;
};

// Export router as default for backward compatibility with tests
module.exports = router;
// Also export the initializers as named exports
module.exports.initJobRoutes = initJobRoutes;
module.exports.initOpenAiRoutes = initOpenAiRoutes;
module.exports.initChatRoutes = initChatRoutes;