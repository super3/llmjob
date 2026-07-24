'use strict';

// Plan a multi-GPU shard for a model too big to fit on any single card. A rig's
// aggregate VRAM (e.g. 2× 16 GB A4000 = 32 GB) can host a ~20 GB model that no
// one card holds, by splitting it across cards with llama.cpp's
// --split-mode layer/row + --tensor-split. This is pure planning: it decides
// WHICH physical cards to use and the per-card proportion; main.js / earn-cli.js
// pass the result to buildServerArgs and pause mining on exactly those cards.
//
// The current client pins the LLM to one GPU (--split-mode none), so this is the
// net-new path that unlocks the big models on the A4000 / 3070 / 4070 rigs. For
// high-throughput tensor parallelism a dedicated backend (vLLM / SGLang) is the
// better engine; this llama.cpp layer-split path is the minimal shipped one and
// is most effective on rigs with real interconnect (the A4000 datacenter boxes).

const { pickServableModel } = require('./vram');

// Free VRAM (MB) for a parsed card ({ index, usedMb, totalMb }), clamped ≥ 0.
function cardFreeMb(c) {
  const total = Number(c && c.totalMb);
  const used = Number(c && c.usedMb);
  if (!Number.isFinite(total) || !Number.isFinite(used)) return 0;
  return Math.max(0, total - used);
}

// Build a sharded serving plan across a rig's cards, or null when sharding isn't
// possible/needed. `cards` is gpu.parseGpuStats output ([{ index, usedMb,
// totalMb }, …]); `models` is the config catalog; `reserveMb` is kept free per
// serving card for mining headroom (0 to pack tight). Returns:
//   { model, devices:[idx…], tensorSplit:[perPhysicalCard…], mainGpu, freeMb }
// where `devices` are the physical card indices hosting the model, `tensorSplit`
// is a proportion per card ordered by index (0 for non-serving cards), and
// `freeMb` is the usable aggregate. Strategy: greedily add the cards with the
// most free VRAM until the aggregate clears the largest model's floor, then size
// the split by each card's free VRAM. Returns null when even all cards together
// can't host the smallest catalog model, or when a single card already fits the
// biggest model (no need to shard — the single-card path handles it).
function pickShardPlan(cards, models, reserveMb) {
  if (!Array.isArray(cards) || cards.length < 2) return null;
  if (!Array.isArray(models) || !models.length) return null;
  const reserve = Number(reserveMb) || 0;

  const usable = cards
    .map((c) => ({ index: Math.floor(Number(c && c.index)), freeMb: Math.max(0, cardFreeMb(c) - reserve) }))
    .filter((c) => Number.isFinite(c.index) && c.index >= 0 && c.freeMb > 0)
    .sort((a, b) => b.freeMb - a.freeMb || a.index - b.index);
  if (usable.length < 2) return null;

  const aggregate = usable.reduce((a, c) => a + c.freeMb, 0);
  const model = pickServableModel(aggregate, models);
  if (!model) return null; // not even the smallest model fits across the whole rig
  const floor = Number(model.minVramMb);

  // If the single best card already fits this model, don't shard — the caller's
  // single-card path is faster and simpler (no cross-card traffic).
  if (usable[0].freeMb >= floor) return null;

  // Greedily take the biggest cards until the model's floor is covered. The whole
  // rig's aggregate already cleared it (pickServableModel), so this always does.
  const chosen = [];
  let sum = 0;
  for (const c of usable) {
    chosen.push(c);
    sum += c.freeMb;
    if (sum >= floor) break;
  }

  // tensor-split proportion per physical card, ordered by index up to the highest
  // card in the shard, 0 for cards inside that range that aren't serving (and
  // trailing cards past it default to 0 in llama.cpp). Raw free-MB weights are
  // fine — llama.cpp normalizes the vector.
  const maxIndex = Math.max(...chosen.map((c) => c.index));
  const tensorSplit = new Array(maxIndex + 1).fill(0);
  for (const c of chosen) tensorSplit[c.index] = c.freeMb;

  const devices = chosen.map((c) => c.index).sort((a, b) => a - b);
  return {
    model,
    devices,
    tensorSplit,
    mainGpu: devices[0], // the card with the lowest index among the shard
    freeMb: sum,
  };
}

module.exports = { pickShardPlan, cardFreeMb };
