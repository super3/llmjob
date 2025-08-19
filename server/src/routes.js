const express = require('express');
const router = express.Router();
const { requireAuth } = require('./middleware/auth');
const { verifySignature } = require('./middleware/signature');
const nodeController = require('./controllers/nodeController');

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

module.exports = router;