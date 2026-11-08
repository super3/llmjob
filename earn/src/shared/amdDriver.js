'use strict';

// AMD's Windows driver, as Win32_VideoController's DriverVersion gives it
// ("32.0.31007.5012"), and the builds reported to move an idle card's VRAM into
// system RAM.
//
// From llama.cpp discussion #23443: from 32.0.31007.5012 (12 May 2026) through
// Adrenalin 26.8.1 and PRO 26.9.1, Windows moved an idle Radeon AI PRO R9700's
// VRAM into system RAM, under Vulkan and HIP alike. PCs short of RAM crashed,
// and on two-R9700 rigs the idle card dropped out of Vulkan within minutes.
// Adrenalin 26.9.2 (32.0.32015.2008, 28 September 2026) fixed it. It is a
// community report, so the app only warns. It never blocks mining.
//
// Pure: probe.detectGpuInfo reads the versions, in the query that names the cards.

const BAD_FROM = '32.0.31007.5012';
const FIXED_IN = '32.0.32015.2008';

// [32, 0, 31007, 5012], or null for anything that isn't numbers between dots.
function parseDriverVersion(v) {
  const text = String(v == null ? '' : v).trim();
  if (!/^\d+(\.\d+)*$/.test(text)) return null;
  return text.split('.').map(Number);
}

// -1, 0 or 1, part by part, a missing part counting as 0. Null when either
// isn't a version.
function compareDriverVersions(a, b) {
  const x = parseDriverVersion(a);
  const y = parseDriverVersion(b);
  if (!x || !y) return null;
  for (let i = 0; i < Math.max(x.length, y.length); i++) {
    const d = (x[i] || 0) - (y[i] || 0);
    if (d) return d < 0 ? -1 : 1;
  }
  return 0;
}

// Whether `v` is one of the builds reported to move an idle card's VRAM.
function movesIdleVram(v) {
  const from = compareDriverVersions(v, BAD_FROM);
  return from !== null && from >= 0 && compareDriverVersions(v, FIXED_IN) < 0;
}

// The log lines for an AMD rig's driver, as [{ level, line }]: its version,
// and a warning when it is one of those builds. `det` is probe.detectGpuInfo's
// answer; only Windows gives its `amd` cards a driverVersion. [] when no
// version is known.
function describeAmdDrivers(det) {
  const cards = det && Array.isArray(det.amd) ? det.amd : [];
  const versions = [...new Set(cards.map((c) => c.driverVersion).filter(Boolean))];
  if (!versions.length) return [];
  const lines = [{ level: 'info', line: 'AMD graphics driver ' + versions.join(', ') }];
  const bad = versions.filter(movesIdleVram);
  if (bad.length) {
    lines.push({
      level: 'warn',
      line: 'AMD graphics driver ' + bad.join(', ') + ' is reported to move an idle card\'s VRAM to system RAM, '
        + 'which can slow or crash the PC (llama.cpp discussion #23443). Adrenalin 26.9.2 or newer fixes it.',
    });
  }
  return lines;
}

module.exports = {
  BAD_FROM, FIXED_IN, parseDriverVersion, compareDriverVersions, movesIdleVram, describeAmdDrivers,
};
