'use strict';

// Pure helper for pool-region auto-detection. main.js measures TCP connect
// latency to each region's Stratum endpoint and feeds the timings here; keeping
// the selection logic here makes it unit-testable without touching the network.

// Pick the region with the lowest latency from a list of { region, ms } results.
// Entries with a non-numeric ms (unreachable / timed out) are ignored. Returns
// the chosen region key, or `fallback` when nothing was reachable.
function pickFastestRegion(results, fallback) {
  let best = null;
  for (const r of Array.isArray(results) ? results : []) {
    if (r && typeof r.ms === 'number' && (best === null || r.ms < best.ms)) best = r;
  }
  return best ? best.region : (fallback == null ? null : fallback);
}

module.exports = { pickFastestRegion };
