const MinerService = require('../services/minerService');

// POST /api/miners/ping — a mining client reports its live status (no auth;
// this is public leaderboard data validated by payout-address format).
//
// The body is whitelisted field by field rather than forwarded wholesale, so a
// client can't smuggle extra columns into the upsert. That means every new
// reported field must be added HERE as well as in the service — a field the
// service stores but this list drops silently reads null for every host on the
// board. Older clients still send the retired LLM fields (llmModel, nodeId);
// they are simply not picked up.
const PING_FIELDS = [
  'address', 'worker', 'gpu', 'region', 'hashrate', 'accepted', 'vramUsedMb', 'vramTotalMb', 'version',
  // Per-card health and rig details, stored for diagnostics and not shown.
  'rejected', 'tempC', 'powerW', 'powerLimitW', 'coreClockMhz', 'memClockMhz', 'fanPct',
  'driver', 'os', 'client', 'uptimeSec', 'lastShareSec',
  // The signed rig identity (see services/rigIdentity).
  'rigId', 'publicKey', 'timestamp', 'signature',
];

async function pingMiner(req, res) {
  try {
    const body = req.body || {};
    const input = {};
    for (const k of PING_FIELDS) input[k] = body[k];
    const service = new MinerService(req.app.locals.db);
    const result = await service.reportMiner(input);
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
