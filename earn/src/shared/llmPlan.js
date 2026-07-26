'use strict';

const { ALL_LAYERS, computeGpuLayers, hasEnoughVram } = require('./vram');

// Plan which GPUs each run their OWN local llama-server instance.
//
// The model is small (~a few GB), so on a multi-GPU rig we can hold an
// independent copy on every card that has room — running one instance per
// eligible GPU multiplies the rig's serving capacity instead of using only the
// single best card (the old behaviour). Given per-card VRAM stats
// ([{ index, usedMb, totalMb }, …] — e.g. shared/gpu.parseGpuStats), the model,
// and the mining VRAM reserve, return one plan entry per card that can hold the
// model after the reserve:
//   { index, freeMb, nGpuLayers }
// sorted by GPU index. Each entry becomes a separate llama-server pinned to that
// card (--main-gpu <index>) on its own port.
//
// VRAM handling mirrors the single-card path:
//   • cards measured, some fit   → one entry per fitting card
//   • cards measured, none fit    → [] (serve nothing; every card is too full)
//   • no card measurable at all   → a single unknown-placement entry
//                                   ({ index: null, full offload }) so llama.cpp
//                                   decides, exactly as when nvidia-smi is absent
// opts: { maxInstances } — an explicit operator ceiling (--llm-max-instances),
// optional. The list is truncated keeping the lowest GPU indices (already
// sorted, so the choice is stable and predictable).
//
// Note on host RAM: loading is what costs memory, not running. llama-server
// streams the GGUF through the page cache into VRAM and the weights live on the
// card, so a loaded instance's host footprint is small — many instances can
// coexist on a modest box. What breaks is loading them ALL AT ONCE: N processes
// each streaming the same multi-GB file evict each other's pages and can crawl
// indefinitely (seen on a 13-GPU rig with 8 GB of RAM: every server alive,
// answering 503, not one byte in VRAM). The fix is serialised startup in
// LlmFleet, not a smaller fleet — so this planner deliberately does NOT cap by
// memory.
function planLlmInstances(cards, model, reserveMb, opts) {
  const list = Array.isArray(cards) ? cards : [];
  const eligible = [];
  let anyParsed = false;

  for (const c of list) {
    if (!c) continue;
    const index = Math.floor(Number(c.index));
    const total = Number(c.totalMb);
    const used = Number(c.usedMb);
    if (!Number.isFinite(index) || index < 0) continue;
    if (!Number.isFinite(total) || !Number.isFinite(used)) continue;
    anyParsed = true;
    const freeMb = Math.max(0, total - used);
    if (hasEnoughVram(freeMb, model) !== true) continue; // card can't hold the model
    const nGpuLayers = computeGpuLayers(freeMb, model, reserveMb || 0);
    if (nGpuLayers <= 0) continue; // the whole model doesn't fit after the reserve
    eligible.push({ index, freeMb, nGpuLayers });
  }

  if (eligible.length) {
    eligible.sort((a, b) => a.index - b.index);
    return capInstances(eligible, model, opts || {});
  }

  // Some cards were measured but none had room → serve nothing.
  if (anyParsed) return [];

  // No card could be measured (non-NVIDIA / no driver) → one instance, unknown
  // placement, full offload; let llama.cpp decide, as the single-card path did.
  return [{ index: null, freeMb: null, nGpuLayers: ALL_LAYERS }];
}

// Truncate a VRAM-eligible list to the operator's cap, if they set one. At least
// one instance always survives: if a card can hold the model, serving from one
// card beats serving from none.
function capInstances(eligible, model, opts) {
  // No cap set at all — note null/undefined must short-circuit BEFORE Number(),
  // since Number(null) is 0 and would silently shrink the fleet to one.
  if (opts.maxInstances == null) return eligible;
  const cap = Math.floor(Number(opts.maxInstances));
  if (!Number.isFinite(cap) || cap >= eligible.length) return eligible;
  return eligible.slice(0, Math.max(1, cap));
}

module.exports = { planLlmInstances };
