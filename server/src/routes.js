const express = require('express');
const router = express.Router();
const { requireAuth } = require('./middleware/auth');
const { verifySignature } = require('./middleware/signature');
const nodeController = require('./controllers/nodeController');
const JobController = require('./controllers/jobController');
const JobService = require('./services/jobService');
const nodeService = require('./services/nodeService');

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

// Initialize job controller with dependencies
const initJobRoutes = (redis) => {
  const jobService = new JobService(redis);
  const jobController = new JobController(jobService, nodeService, redis);

  // Job submission and management
  router.post('/jobs', requireAuth, (req, res) => jobController.submitJob(req, res));
  router.get('/jobs/stats', (req, res) => jobController.getStats(req, res));
  router.get('/jobs/:jobId', (req, res) => jobController.getJob(req, res));
  
  // Node job operations (require signature verification)
  router.post('/jobs/poll', verifySignature, (req, res) => jobController.pollJobs(req, res));
  router.post('/jobs/:jobId/heartbeat', verifySignature, (req, res) => jobController.heartbeat(req, res));
  router.post('/jobs/:jobId/chunks', verifySignature, (req, res) => jobController.receiveChunk(req, res));
  router.post('/jobs/:jobId/complete', verifySignature, (req, res) => jobController.completeJob(req, res));
  router.post('/jobs/:jobId/fail', verifySignature, (req, res) => jobController.failJob(req, res));
  
  // Admin operations
  router.post('/jobs/cleanup', requireAuth, (req, res) => jobController.cleanupJobs(req, res));
  router.post('/jobs/check-timeouts', (req, res) => jobController.checkTimeouts(req, res));
};

// Export router as default for backward compatibility with tests
module.exports = router;
// Also export initJobRoutes as a named export
module.exports.initJobRoutes = initJobRoutes;