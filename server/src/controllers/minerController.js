const MinerService = require('../services/minerService');

// POST /api/miners/ping — a mining client reports its live status (no auth;
// this is public leaderboard data validated by payout-address format).
//
// The body is whitelisted field by field rather than forwarded wholesale, so a
// client can't smuggle extra columns into the upsert. That means every new
// reported field must be added HERE as well as in the service — `llmModel`
// (which card serves which local LLM) was stored, queried and rendered by the
// board while this whitelist silently dropped it, so the column read null for
// every host on the network. See the route test that pings with an llmModel and
// reads it back off GET /api/miners.
async function pingMiner(req, res) {
  try {
    const { address, worker, gpu, region, hashrate, accepted, vramUsedMb, vramTotalMb, version, llmModel, nodeId } = req.body;
    const service = new MinerService(req.app.locals.db);
    const result = await service.reportMiner({ address, worker, gpu, region, hashrate, accepted, vramUsedMb, vramTotalMb, version, llmModel, nodeId });
    if (result.error) {
      return res.status(400).json({ error: result.error });
    }
    res.json(result);
  } catch (error) {
    console.error('Miner ping error:', error);
    res.status(500).json({ error: 'Failed to report miner' });
  }
}

// GET /api/miners — online miners for the network page (one row per worker/GPU).
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
