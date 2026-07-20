'use strict';

// Decide how much of the model to put on the GPU so it fits alongside the miner.
// llama.cpp's --n-gpu-layers offloads N transformer layers to the GPU (the rest
// run on the CPU); this picks N from the free VRAM, keeping `reserveMb` free for
// mining. Pure math so it's unit-tested without a GPU.

// Returns 0..model.layers. 0 = CPU-only (nothing fit); model.layers = full offload.
function computeGpuLayers(freeMb, model, reserveMb) {
  const layers = Math.floor(Number(model && model.layers)) || 0;
  const full = Number(model && model.vramFullMb) || 0;
  const free = Number(freeMb);
  if (layers <= 0 || full <= 0 || !Number.isFinite(free)) return 0;

  const budget = free - (Number(reserveMb) || 0);
  if (budget <= 0) return 0;          // no room after the mining reserve
  if (budget >= full) return layers;  // everything fits on the GPU
  return Math.floor((budget / full) * layers); // the fraction of layers that fit
}

// The minimum free VRAM (MB) we require before putting the model on the GPU: an
// explicit per-model floor (`minVramMb`), else its full-offload estimate. 0 when
// neither is configured (no floor → always allowed).
function requiredVramMb(model) {
  return Number(model && model.minVramMb) || Number(model && model.vramFullMb) || 0;
}

// Is `freeMb` enough to start the model on the GPU without risking an OOM?
//   true  — free VRAM is known and covers the requirement
//   false — free VRAM is known and falls short (caller should not start)
//   null  — free VRAM can't be measured (no NVIDIA / no driver); caller decides
// A model with no configured floor always returns true.
function hasEnoughVram(freeMb, model) {
  const need = requiredVramMb(model);
  if (need <= 0) return true;
  if (freeMb == null) return null;         // not measured (detectVram returned null)
  const free = Number(freeMb);
  if (!Number.isFinite(free)) return null; // unparseable → treat as unknown
  return free >= need;
}

module.exports = { computeGpuLayers, requiredVramMb, hasEnoughVram };
