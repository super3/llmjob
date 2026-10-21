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
// The rows are the cards nvidia-smi lists (the caller leaves out cards the user
// took off mining), plus any card the engine reports that nvidia-smi does not.
// Each carries its own numbers from the engine's per-card stats (PearlEngine
// tags each status event with the card's `gpuIndex` — see shared/miningStats).
// A card the engine does not report reads 0: its core refused (usually the
// local LLM holds its VRAM), or the first status has not come yet.
//
// This used to split the engine's total evenly over every card nvidia-smi
// listed whenever the engine reported fewer, for an alpha-miner build that
// logged one aggregate line. Our engine reports every card it mines, so the
// split only ever put an idle card on the board at half the real card's rate.
// `serving` (optional) tags the cards currently running the local LLM so the
// network board can show which GPU serves which model: { model, indices, nodeId }
// where `indices` are the GPU indices from the fleet's servingIndices(). A card
// whose index isn't listed (insufficient VRAM), or any row from a client that
// doesn't pass `serving` at all (older version), reports llmModel null → blank on
// the board.
//
// `serving.nodeId` is the machine's node id, and is only set when this machine is
// a linked node whose job worker is armed — i.e. when it actually polls the
// cluster for work. That's a different thing from running the model: a rig can
// have the LLM loaded (llmModel set) and still serve nobody, which is exactly the
// case the board couldn't previously show. It's rig-level, not per-card, so it
// rides on every row of a multi-GPU host.
//
// `host` (optional) is the rest of what belongs to the machine rather than a
// card: { ramUsedMb, ramTotalMb, xmrHashrate }, from hostStats() below. Like
// nodeId it rides on every row, so the server must take it once per host and
// never sum it across cards. Anything missing reports null, which the server
// stores as "unknown", the same as a client too old to send it.
function buildMinerReports(settings, snap, gpuVram, version, serving, host) {
  const s = settings || {};
  const n = snap || {};
  const h = host || {};
  const base = {
    address: String(s.address || '').trim(),
    worker: String(s.worker || 'rig01').trim() || 'rig01',
    region: s.region || 'us2',
    version: version != null ? String(version) : null, // earn client version, so the board can see fleet versions
    nodeId: serving && serving.nodeId ? String(serving.nodeId) : null,
    ramUsedMb: numOrNull(h.ramUsedMb),
    ramTotalMb: numOrNull(h.ramTotalMb),
    xmrHashrate: numOrNull(h.xmrHashrate),
  };

  const serveModel = serving && serving.model ? String(serving.model) : null;
  const serveSet = new Set(
    serving && Array.isArray(serving.indices) ? serving.indices.map((i) => Number(i)) : []
  );
  // The model this card serves, or null when it isn't serving (or nothing is).
  const llmFor = (index) => (serveModel && serveSet.has(Number(index)) ? serveModel : null);

  // ONE rule for the card name, everywhere below: nvidia-smi first, the engine's
  // own label only as a fallback.
  //
  // The two sources used to be indistinguishable — alpha-miner 1.8.x logged
  // `gpu=0:NVIDIA GeForce RTX 4090`, the exact nvidia-smi string — so the
  // row-building paths here disagreed about precedence for a long time without
  // anyone noticing. 1.9.4 renders a stats table whose name column is an
  // abbreviated label ("RTX 5090"), which made that latent disagreement visible:
  // the same rig reported "RTX 5090" as a single card and "NVIDIA GeForce RTX
  // 5090" once it had two, purely because different branches picked differently.
  //
  // nvidia-smi is the canonical device name, so it wins; the engine's label is
  // kept as the fallback because a rig without nvidia-smi (a slim container) has
  // nothing else to offer.
  const vram = Array.isArray(gpuVram) ? gpuVram : [];
  const vramFor = (index) => vram.find((v) => v && Number(v.index) === index) || null;
  const cards = Array.isArray(n.gpus) ? n.gpus : [];
  const cardFor = (index) => cards.find((c) => c.index === index) || null;

  // No per-card engine data yet and at most one card: a genuine single-GPU rig
  // (or one with no nvidia-smi) posts the single bare row, matching what older
  // clients sent.
  if (!cards.length && vram.length <= 1) {
    return [{
      ...base,
      gpu: (vram[0] && vram[0].name) || n.gpu || null,
      hashrate: Number(n.total) || 0,
      accepted: Number(n.accepted) || 0,
      vramUsedMb: vram.reduce((a, v) => a + (Number(v.usedMb) || 0), 0),
      vramTotalMb: vram.reduce((a, v) => a + (Number(v.totalMb) || 0), 0),
      llmModel: llmFor(vram[0] ? vram[0].index : 0),
    }];
  }

  // One row per card, each matched to its VRAM and its engine numbers by index.
  // On a multi-GPU rig every card gets a distinct "/gpuN" worker, which keeps
  // them as separate board rows, from the first report on: a bare-worker row
  // left over would group as a phantom extra card and double-count the host's
  // VRAM.
  const indices = vram.map((v) => Number(v.index));
  for (const c of cards) if (!indices.includes(c.index)) indices.push(c.index);
  indices.sort((a, b) => a - b);
  return indices.map((index) => {
    const v = vramFor(index);
    const c = cardFor(index);
    return {
      ...base,
      worker: indices.length > 1 ? base.worker + '/gpu' + index : base.worker,
      gpu: (v && v.name) || (c && c.gpu) || null,
      hashrate: c ? Number(c.hashrate) || 0 : 0,
      accepted: c ? Number(c.accepted) || 0 : 0,
      vramUsedMb: v ? Number(v.usedMb) || 0 : 0,
      vramTotalMb: v ? Number(v.totalMb) || 0 : 0,
      llmModel: llmFor(index),
    };
  });
}

// A finite number, or null. Number(null) is 0, so null is checked first: a
// field nobody measured must not read as a measured zero.
function numOrNull(v) {
  if (v == null) return null;
  const x = Number(v);
  return Number.isFinite(x) ? x : null;
}

const MB = 1024 * 1024;

// This machine's system RAM in whole MB: used is total minus free, as the OS
// reports them. In a container with a memory limit (docker --memory, a rented
// Vast box) the OS figures are the whole host's, other tenants included, so
// the limit is the total and used is what the container has in use. That is
// the same cap readCpuFacts puts on memory for the CPU miner's plan. `os` is
// Node's os module and `proc` the process, passed in so the tests can hand them
// any numbers. A reading that makes no sense comes back null, so the board
// shows it as unknown rather than wrong.
function readRamMb(os, proc) {
  const p = proc || process;
  let total = Number(os.totalmem());
  if (!(total > 0)) return { usedMb: null, totalMb: null };
  let free = Number(os.freemem());
  // constrainedMemory() is 0, or a huge number, when nothing limits us.
  const limit = typeof p.constrainedMemory === 'function' ? Number(p.constrainedMemory()) : 0;
  if (limit > 0 && limit < total) {
    total = limit;
    // The limit minus what the container uses, by its cgroup's count; 0 while
    // use runs past the limit, which it can for a moment. Without it, used is
    // unknown: the OS's free figure is the host's, not ours.
    free = typeof p.availableMemory === 'function' ? Math.min(Number(p.availableMemory()), total) : NaN;
  }
  const used = free >= 0 && free <= total ? total - free : null;
  return { usedMb: used == null ? null : Math.round(used / MB), totalMb: Math.round(total / MB) };
}

// The `host` fields for buildMinerReports, read now: system RAM, and the Monero
// miner's hashrate in H/s. `cpu` is the XmrEngine, or null when CPU mining is
// off. One that is not running reports null as well, so the board can tell "not
// CPU mining" from "CPU mining at 0 H/s", which is what a running engine reads
// while its pool is out (the GPU figure does the same). `proc` is for the
// tests; the shells leave it out and get the real process.
function hostStats(os, cpu, proc) {
  const ram = readRamMb(os, proc);
  return {
    ramUsedMb: ram.usedMb,
    ramTotalMb: ram.totalMb,
    xmrHashrate: cpu && cpu.isRunning() ? Number(cpu.snapshot().hashrate) || 0 : null,
  };
}

module.exports = { buildMinerReports, hostStats, readRamMb };
