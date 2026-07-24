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
// `serving` (optional) tags the cards currently running the local LLM so the
// network board can show which GPU serves which model: { model, indices } where
// `indices` are the GPU indices from the fleet's servingIndices(). A card whose
// index isn't listed (insufficient VRAM), or any row from a client that doesn't
// pass `serving` at all (older version), reports llmModel null → blank on the board.
function buildMinerReports(settings, snap, gpuVram, version, serving) {
  const s = settings || {};
  const n = snap || {};
  const base = {
    address: String(s.address || '').trim(),
    worker: String(s.worker || 'rig01').trim() || 'rig01',
    region: s.region || 'us2',
    version: version != null ? String(version) : null, // earn client version, so the board can see fleet versions
  };

  const serveModel = serving && serving.model ? String(serving.model) : null;
  const serveSet = new Set(
    serving && Array.isArray(serving.indices) ? serving.indices.map((i) => Number(i)) : []
  );
  // The model this card serves, or null when it isn't serving (or nothing is).
  const llmFor = (index) => (serveModel && serveSet.has(Number(index)) ? serveModel : null);

  const vram = Array.isArray(gpuVram) ? gpuVram : [];
  const vramFor = (index) => vram.find((v) => v && Number(v.index) === index) || null;
  const cards = Array.isArray(n.gpus) ? n.gpus : [];
  const workerFor = (index, count) => (count > 1 ? base.worker + '/gpu' + index : base.worker);

  // One row per physical card (from nvidia-smi), splitting a rig-level total
  // evenly — used whenever we know the card count but not each card's hashrate.
  // The distinct "/gpuN" worker is what keeps the cards as separate board rows.
  const splitRows = (total, accepted, fallbackName) => vram.map((v) => ({
    ...base,
    worker: workerFor(v.index, vram.length),
    gpu: v.name || fallbackName || null,
    hashrate: total / vram.length,
    accepted: Math.round(accepted / vram.length),
    vramUsedMb: Number(v.usedMb) || 0,
    vramTotalMb: Number(v.totalMb) || 0,
    llmModel: llmFor(v.index),
  }));

  // No per-card engine data yet (startup before the first per-card event, or an
  // engine build that only logs one aggregate line). For a MULTI-GPU rig, still
  // post one row per physical card (splitting the rig total evenly) rather than a
  // single bare-worker aggregate row: the board keys on address+worker, so a
  // lingering bare-worker row groups as a phantom extra card and double-counts
  // the host's VRAM. A genuine single-GPU rig (or one with no nvidia-smi) keeps
  // the single bare row, matching what older clients sent.
  if (!cards.length) {
    if (vram.length > 1) return splitRows(Number(n.total) || 0, Number(n.accepted) || 0, n.gpu);
    return [{
      ...base,
      gpu: n.gpu || (vram[0] && vram[0].name) || null,
      hashrate: Number(n.total) || 0,
      accepted: Number(n.accepted) || 0,
      vramUsedMb: vram.reduce((a, v) => a + (Number(v.usedMb) || 0), 0),
      vramTotalMb: vram.reduce((a, v) => a + (Number(v.totalMb) || 0), 0),
      llmModel: llmFor(vram[0] ? vram[0].index : 0),
    }];
  }

  // Safety net: the engine under-enumerated GPUs (e.g. one aggregate line) while
  // nvidia-smi sees more cards. Enumerate the physical cards so none are hidden
  // and each shows its own VRAM; split the engine's reported total hashrate/
  // shares evenly, since per-card values aren't available in this shape.
  if (vram.length > cards.length) {
    const total = cards.reduce((a, c) => a + (Number(c.hashrate) || 0), 0);
    const accepted = cards.reduce((a, c) => a + (Number(c.accepted) || 0), 0);
    return splitRows(total, accepted, cards[0] && cards[0].gpu);
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
      llmModel: llmFor(c.index),
    };
  });
}

module.exports = { buildMinerReports };
