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

module.exports = { computeGpuLayers };
