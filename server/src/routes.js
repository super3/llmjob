const express = require('express');
const router = express.Router();
const minerController = require('./controllers/minerController');
const waitlistController = require('./controllers/waitlistController');

// Miners (Earn clients) — public, no auth.
// POST /api/miners/ping - A mining client reports its live status
router.post('/miners/ping', minerController.pingMiner);
// GET /api/miners - Online miners grouped by host (network page)
router.get('/miners', minerController.getPublicMiners);

// Managed-mining waitlist — public, no auth. The signup form on /managed.
router.post('/waitlist', waitlistController.joinWaitlist);

module.exports = router;
