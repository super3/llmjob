const MinerService = require('../services/minerService');
const { handle, reply } = require('./http');

// POST /api/miners/ping — a mining client reports its live status (no auth;
// this is public leaderboard data validated by payout-address format).
//
// The body is whitelisted field by field rather than forwarded wholesale, so a
// client can't smuggle extra columns into the upsert. That means every new
// reported field must be added HERE as well as in the service — `llmModel`
// (which card serves which local LLM) was stored, queried and rendered by the
// board while this whitelist silently dropped it, so the column read null for
// every host on the network. See the route test that pings with an llmModel and
// reads it back off GET /api/miners. ramUsedMb, ramTotalMb and xmrHashrate (the
// machine's RAM and Monero hashrate) have a round-trip test of their own, as
// does `coin` (a rig mining Quantus).
const pingMiner = handle(async (req, res) => {
  const {
    address, worker, gpu, region, hashrate, accepted, vramUsedMb, vramTotalMb, version, llmModel, nodeId,
    ramUsedMb, ramTotalMb, xmrHashrate, coin,
  } = req.body;
  const service = new MinerService(req.app.locals.db);
  reply(res, await service.reportMiner({
    address, worker, gpu, region, hashrate, accepted, vramUsedMb, vramTotalMb, version, llmModel, nodeId,
    ramUsedMb, ramTotalMb, xmrHashrate, coin,
  }));
}, 'Failed to report miner');

// GET /api/miners — online miners for the network page (one row per worker/GPU).
const getPublicMiners = handle(async (req, res) => {
  const service = new MinerService(req.app.locals.db);
  res.json(await service.getPublicMiners());
}, 'Failed to get miners');

module.exports = { pingMiner, getPublicMiners };
