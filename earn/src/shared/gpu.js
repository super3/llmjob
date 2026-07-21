'use strict';

// Pure helper for GPU auto-detection. main.js runs the system query
// (Win32_VideoController) and feeds the raw adapter-name list here; keeping the
// selection logic here makes it unit-testable without touching the OS.

// Virtual / basic display adapters that aren't real mining GPUs.
const IGNORE = /basic display|microsoft basic|remote|rdp|virtual|meta|parsec|citrix|vmware|oray/i;

// Integrated GPUs (iGPU / APU). They enumerate alongside a discrete card — often
// first — but mine at a fraction of its rate, so they're a last resort. Matches
// Intel iGPUs (UHD/HD/Iris) and AMD APUs, whose adapter name is a bare "Radeon
// Graphics" / "…Vega… Graphics" with no discrete model (RX/Pro/Instinct).
const INTEGRATED = /\bintel\b|\buhd\b|\bhd graphics\b|\biris\b|radeon(\(tm\))? graphics|vega.*graphics|integrated/i;

// Pick the best real GPU from a list of adapter names, or null if none. Prefers
// a discrete card over an integrated one (Win32_VideoController may list the
// iGPU before the discrete GPU that actually mines), keeping list order within a
// tier and falling back to an integrated GPU only when it's all that's present.
function pickGpu(names) {
  if (!Array.isArray(names)) return null;
  const real = [];
  for (const raw of names) {
    const name = String(raw == null ? '' : raw).trim();
    if (name && !IGNORE.test(name)) real.push(name);
  }
  if (!real.length) return null;
  return real.find((n) => !INTEGRATED.test(n)) || real[0];
}

// Count the GPUs that actually mine: discrete cards when any are present (an
// iGPU alongside them contributes nothing worth counting), else 1 if only an
// integrated GPU exists, else 0. Multi-GPU rigs use this to scale the static
// share difficulty — the pool's table is per card class, so a rig's aggregate
// hashrate wants roughly per-card × count.
function countGpus(names) {
  if (!Array.isArray(names)) return 0;
  let real = 0;
  let discrete = 0;
  for (const raw of names) {
    const name = String(raw == null ? '' : raw).trim();
    if (!name || IGNORE.test(name)) continue;
    real++;
    if (!INTEGRATED.test(name)) discrete++;
  }
  return discrete > 0 ? discrete : (real > 0 ? 1 : 0);
}

// Parse `nvidia-smi --query-gpu=index,name,memory.used,memory.total
// --format=csv,noheader,nounits` into one entry per card:
//   [{ index, name, usedMb, totalMb }, ...]
// The network board uses this to report each GPU's own VRAM (the limiting
// factor for co-running an LLM) instead of the rig's summed total. Rows that
// don't parse cleanly are skipped; index/used/total are read positionally (the
// name is the middle field and never contains a comma) so a stray column can't
// misalign the numbers.
function parseGpuStats(out) {
  const list = [];
  for (const row of String(out == null ? '' : out).split(/\r?\n/)) {
    const line = row.trim();
    if (!line) continue;
    const parts = line.split(',').map((x) => x.trim());
    if (parts.length < 4) continue;
    const index = parseInt(parts[0], 10);
    const usedMb = parseInt(parts[parts.length - 2], 10);
    const totalMb = parseInt(parts[parts.length - 1], 10);
    if (!Number.isFinite(index) || !Number.isFinite(usedMb) || !Number.isFinite(totalMb)) continue;
    const name = parts.slice(1, parts.length - 2).join(',').trim() || null;
    list.push({ index, name, usedMb, totalMb });
  }
  return list;
}

module.exports = { IGNORE, INTEGRATED, pickGpu, countGpus, parseGpuStats };
