'use strict';

// Build the /api/miners report payloads from the current settings, a live stats
// snapshot (shared/miningStats) and the per-card VRAM probe (shared/gpu's
// parseGpuStats). Pure so it's unit-tested; the CLI/GUI POST each payload to the
// network board while mining.
//
// One payload per GPU: a rig running several cards posts one row per card, each
// with its own GPU name, hashrate and VRAM — because per-card VRAM (not the
// rig's summed total) is what tells you whether a card can hold a given model.
// Each card gets a distinct `worker` (so the server, which keys rows on
// address+worker, stores them as separate rows); single-GPU rigs keep the bare
// worker name, matching what older clients sent.
function buildMinerReports(settings, snap, gpuVram) {
  const s = settings || {};
  const n = snap || {};
  const base = {
    address: String(s.address || '').trim(),
    worker: String(s.worker || 'rig01').trim() || 'rig01',
    region: s.region || 'us2',
  };

  const vram = Array.isArray(gpuVram) ? gpuVram : [];
  const vramFor = (index) => vram.find((v) => v && Number(v.index) === index) || null;
  const cards = Array.isArray(n.gpus) ? n.gpus : [];

  // No per-card engine data yet (nothing mined, or an event with no card index):
  // fall back to a single rig-level row, summing VRAM across whatever cards the
  // probe saw — the same one-row shape the board showed before per-GPU rows.
  if (!cards.length) {
    return [{
      ...base,
      gpu: n.gpu || (vram[0] && vram[0].name) || null,
      hashrate: Number(n.total) || 0,
      accepted: Number(n.accepted) || 0,
      vramUsedMb: vram.reduce((a, v) => a + (Number(v.usedMb) || 0), 0),
      vramTotalMb: vram.reduce((a, v) => a + (Number(v.totalMb) || 0), 0),
    }];
  }

  const multi = cards.length > 1;
  return cards.map((c) => {
    const v = vramFor(c.index);
    return {
      ...base,
      worker: multi ? base.worker + '/gpu' + c.index : base.worker,
      gpu: c.gpu || (v && v.name) || null,
      hashrate: Number(c.hashrate) || 0,
      accepted: Number(c.accepted) || 0,
      vramUsedMb: v ? Number(v.usedMb) || 0 : 0,
      vramTotalMb: v ? Number(v.totalMb) || 0 : 0,
    };
  });
}

module.exports = { buildMinerReports };
