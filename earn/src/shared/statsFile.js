'use strict';

// Payload written to the CLI's --stats-file path so external consumers — the
// HiveOS custom-miner h-stats.sh hook in particular — can read live telemetry
// without parsing miner logs. Pure so it's unit-testable; the CLI wires the
// timer and the atomic write around it.
//
// Units: `ths` is the engine's total hashrate in TH/s (the snapshot's native
// unit). Consumers convert — HiveOS wants kH/s for `khs` and MH/s for `hs`.
function statsFilePayload(snap, meta) {
  const s = snap || {};
  const m = meta || {};
  const ths = Number(s.total);
  return {
    ver: String(m.version || ''),
    algo: 'pearlhash',
    ths: Number.isFinite(ths) && ths >= 0 ? ths : 0,
    accepted: Number(s.accepted) || 0,
    rejected: Number(s.rejected) || 0,
    uptimeSec: Number(s.uptimeSec) || 0,
    gpu: s.gpu || null,
    updatedMs: Number(m.nowMs) || 0,
  };
}

module.exports = { statsFilePayload };
