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

// From per-card VRAM stats ([{ index, usedMb, totalMb }, ...] — e.g. the output
// of gpu.parseGpuStats), pick the GPU best suited to host the local LLM: the
// card with the most free VRAM (total − used), ties broken by the lower index.
// llama-server runs with --split-mode none, so the model lives entirely on ONE
// card — the preflight and layer sizing must use THAT card's free VRAM, not the
// rig's summed total. On a multi-GPU miner the sum is huge but unusable (the
// model can't span cards), and sizing against it would try to cram the model
// onto device 0 and risk an OOM there. Returns { index, freeMb } for the chosen
// card, or null when no card parses (caller then treats VRAM as unknown and
// lets llama.cpp decide, exactly as when nvidia-smi isn't present at all).
function pickLlmGpu(cards) {
  if (!Array.isArray(cards)) return null;
  let best = null;
  for (const c of cards) {
    if (!c) continue;
    const index = Math.floor(Number(c.index));
    const total = Number(c.totalMb);
    const used = Number(c.usedMb);
    if (!Number.isFinite(index) || index < 0) continue;
    if (!Number.isFinite(total) || !Number.isFinite(used)) continue;
    const freeMb = Math.max(0, total - used);
    if (!best || freeMb > best.freeMb || (freeMb === best.freeMb && index < best.index)) {
      best = { index, freeMb };
    }
  }
  return best;
}

// Choose which catalog model a node should serve from the free VRAM on the card
// that will host it. Given `freeMb` (one card's free VRAM, from pickLlmGpu) and
// the ordered `models` catalog (config.LLM.models), return the LARGEST model
// whose `minVramMb` floor fits — so a 24 GB card serves the 27B while an 8 GB
// card falls back to the small default. Returns:
//   - the biggest fitting model when free VRAM is known and something fits
//   - null when free VRAM is known and not even the smallest model fits
//   - the `default: true` model (else the smallest) when free VRAM is unknown
//     (freeMb null / non-finite), matching the "let llama.cpp decide" fallback
//     used elsewhere when nvidia-smi can't be read
// Pure so it's unit-tested without a GPU.
function pickServableModel(freeMb, models) {
  if (!Array.isArray(models) || !models.length) return null;
  const floor = (m) => Number(m && m.minVramMb) || 0;
  const sorted = models.slice().sort((a, b) => floor(a) - floor(b));

  const free = Number(freeMb);
  if (freeMb == null || !Number.isFinite(free)) {
    return sorted.find((m) => m && m.default) || sorted[0];
  }

  let best = null;
  for (const m of sorted) {
    if (floor(m) <= free) best = m; // sorted ascending → last fit is the largest
  }
  return best;
}

module.exports = { computeGpuLayers, requiredVramMb, hasEnoughVram, pickLlmGpu, pickServableModel };
