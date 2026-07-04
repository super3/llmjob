'use strict';

// Pure helper for GPU auto-detection. main.js runs the system query
// (Win32_VideoController) and feeds the raw adapter-name list here; keeping the
// selection logic here makes it unit-testable without touching the OS.

// Virtual / basic display adapters that aren't real mining GPUs.
const IGNORE = /basic display|microsoft basic|remote|rdp|virtual|meta|parsec|citrix|vmware|oray/i;

// Pick the first real GPU from a list of adapter names, or null if none.
function pickGpu(names) {
  if (!Array.isArray(names)) return null;
  for (const raw of names) {
    const name = String(raw == null ? '' : raw).trim();
    if (name && !IGNORE.test(name)) return name;
  }
  return null;
}

module.exports = { IGNORE, pickGpu };
