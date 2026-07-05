'use strict';

// Build the /api/miners/ping payload from the current settings and a live stats
// snapshot (shared/miningStats). Pure so it's unit-tested; main.js POSTs it to
// the network page's API while mining.
function buildMinerReport(settings, snap) {
  const s = settings || {};
  const n = snap || {};
  return {
    address: String(s.address || '').trim(),
    worker: String(s.worker || 'rig01').trim() || 'rig01',
    region: s.region || 'us2',
    gpu: n.gpu || null,
    hashrate: Number(n.total) || 0,
    accepted: Number(n.accepted) || 0,
    vramUsedMb: Number(n.vramUsedMb) || 0,
    vramTotalMb: Number(n.vramTotalMb) || 0,
  };
}

module.exports = { buildMinerReport };
