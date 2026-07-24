'use strict';

const { computeGpuLayers, hasEnoughVram, pickLlmGpu, pickServableModel } = require('./vram');
const { pickShardPlan } = require('./shard');

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
function planLlmInstances(cards, model, reserveMb) {
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
    if (nGpuLayers <= 0) continue; // no layers fit after the mining reserve
    eligible.push({ index, freeMb, nGpuLayers });
  }

  if (eligible.length) {
    eligible.sort((a, b) => a.index - b.index);
    return eligible;
  }

  // Some cards were measured but none had room → serve nothing.
  if (anyParsed) return [];

  // No card could be measured (non-NVIDIA / no driver) → one instance, unknown
  // placement, full offload; let llama.cpp decide, as the single-card path did.
  const layers = Math.floor(Number(model && model.layers)) || 0;
  return [{ index: null, freeMb: null, nGpuLayers: layers }];
}

// Decide what the whole rig should serve, tiering the model to the hardware and
// sharding when a bigger model needs more than one card. Given the per-card VRAM
// stats, the model catalog (config.LLM.models) and the mining reserve, returns:
//   { model, sharded, instances }
// where `model` is the single model the rig serves (so telemetry + job routing
// stay per-node), `sharded` says whether it spans cards, and `instances` is the
// fleet plan:
//   • sharded  → one entry across the shard's cards
//                ({ index: mainGpu, nGpuLayers, splitMode:'layer', tensorSplit, devices })
//   • per-card → one entry per card that holds the model (planLlmInstances)
//
// Policy: serve the LARGEST model the rig can. Take the biggest model the best
// single card fits; if sharding the aggregate unlocks a bigger one (no single
// card fits it), serve that one sharded instead. `model` is null (empty plan)
// when even the smallest model won't fit any card — the caller then skips the LLM.
function planLlmServing(cards, models, reserveMb) {
  const list = Array.isArray(cards) ? cards : [];
  const reserve = reserveMb || 0;

  const best = pickLlmGpu(list);
  const singleModel = pickServableModel(best ? best.freeMb : null, models);
  const singleFloor = singleModel ? Number(singleModel.minVramMb) : 0;

  // Sharding wins only when it hosts a bigger model than any single card can.
  const shard = pickShardPlan(list, models, reserve);
  if (shard && Number(shard.model.minVramMb) > singleFloor) {
    return {
      model: shard.model,
      sharded: true,
      instances: [{
        index: shard.mainGpu,
        freeMb: shard.freeMb,
        nGpuLayers: shard.model.layers,
        splitMode: 'layer',
        tensorSplit: shard.tensorSplit,
        devices: shard.devices,
      }],
    };
  }

  if (!singleModel) return { model: null, sharded: false, instances: [] };
  return { model: singleModel, sharded: false, instances: planLlmInstances(list, singleModel, reserve) };
}

module.exports = { planLlmInstances, planLlmServing };
