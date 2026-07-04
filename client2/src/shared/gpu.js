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

module.exports = { IGNORE, INTEGRATED, pickGpu };
