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
//
// Rows are normally driven by the engine's per-card stats (alpha-miner tags each
// card with `gpu=<index>` and reports that card's hashrate — see shared/parser).
// But nvidia-smi is the authority on how many cards physically exist, so if the
// engine ever reports FEWER cards than nvidia-smi sees (an engine build that logs
// one aggregate line instead of per-card lines), we fall back to enumerating the
// physical cards and splitting the engine's reported total evenly across them —
// that keeps every card's real VRAM visible instead of collapsing the rig into
// one row, at the cost of an even (rather than measured) hashrate split, which is
// the best that's possible when the engine doesn't break the hashrate out.
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
  const workerFor = (index, count) => (count > 1 ? base.worker + '/gpu' + index : base.worker);

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

  // Safety net: the engine under-enumerated GPUs (e.g. one aggregate line) while
  // nvidia-smi sees more cards. Enumerate the physical cards so none are hidden
  // and each shows its own VRAM; split the engine's reported total hashrate/
  // shares evenly, since per-card values aren't available in this shape.
  if (vram.length > cards.length) {
    const total = cards.reduce((a, c) => a + (Number(c.hashrate) || 0), 0);
    const accepted = cards.reduce((a, c) => a + (Number(c.accepted) || 0), 0);
    const engineName = cards[0] && cards[0].gpu;
    const n2 = vram.length;
    return vram.map((v) => ({
      ...base,
      worker: workerFor(v.index, n2),
      gpu: v.name || engineName || null,
      hashrate: total / n2,
      accepted: Math.round(accepted / n2),
      vramUsedMb: Number(v.usedMb) || 0,
      vramTotalMb: Number(v.totalMb) || 0,
    }));
  }

  // Normal path: the engine reports each card, so use its per-card hashrate and
  // match each card's VRAM by index.
  return cards.map((c) => {
    const v = vramFor(c.index);
    return {
      ...base,
      worker: workerFor(c.index, cards.length),
      gpu: c.gpu || (v && v.name) || null,
      hashrate: Number(c.hashrate) || 0,
      accepted: Number(c.accepted) || 0,
      vramUsedMb: v ? Number(v.usedMb) || 0 : 0,
      vramTotalMb: v ? Number(v.totalMb) || 0 : 0,
    };
  });
}

module.exports = { buildMinerReports };
