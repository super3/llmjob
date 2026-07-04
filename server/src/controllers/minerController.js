const MinerService = require('../services/minerService');

// POST /api/miners/ping — a mining client reports its live status (no auth;
// this is public leaderboard data validated by payout-address format).
async function pingMiner(req, res) {
  try {
    const { address, worker, gpu, region, hashrate, accepted } = req.body;
    const service = new MinerService(req.app.locals.db);
    const result = await service.reportMiner({ address, worker, gpu, region, hashrate, accepted });
    if (result.error) {
      return res.status(400).json({ error: result.error });
    }
    res.json(result);
  } catch (error) {
    console.error('Miner ping error:', error);
    res.status(500).json({ error: 'Failed to report miner' });
  }
}

// GET /api/miners — online miners grouped by address for the network page.
async function getPublicMiners(req, res) {
  try {
    const service = new MinerService(req.app.locals.db);
    const result = await service.getPublicMiners();
    res.json(result);
  } catch (error) {
    console.error('Get miners error:', error);
    res.status(500).json({ error: 'Failed to get miners' });
  }
}

module.exports = { pingMiner, getPublicMiners };
