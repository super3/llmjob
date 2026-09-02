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
    // ── additive ────────────────────────────────────────────────────────────
    // Everything above keeps its exact name and unit: HiveOS's h-stats.sh reads
    // those keys by name with jq, so adding beside them cannot break it.
    // `schema` exists so a consumer can tell an old writer from a new one
    // without inferring it from which keys happen to be present.
    schema: 1,
    // What the node is DOING, which the counters above cannot express: a rig in
    // demand mode with 0 TH/s is not a broken miner, it is a busy one.
    mode: m.mode || null,
    strategy: m.strategy || null,   // 'demand' | 'corun' | null (not auto)
    gate: m.gate || null,           // MINING | SERVING | SWITCHING->* | null
    mining: !!m.mining,
    // null rather than 0 -- "never" and "at the epoch" are different, and a
    // consumer showing "56 years ago" is worse than one showing nothing.
    lastShareMs: s.lastShareMs == null ? null : Number(s.lastShareMs),
    model: modelOf(m.llm),
    tps: {
      // Two different measurements, not two samples of one. They routinely
      // differ by an order of magnitude.
      gen: Number(m.llm && m.llm.tps) || 0,
      prefill: Number(m.llm && m.llm.promptTps) || 0,
    },
    gpus: Array.isArray(s.gpus) ? s.gpus : [],
  };
}

// The served model, flattened to the fields a consumer can act on. Returns null
// when nothing is loaded rather than an object of nulls, so "no model" is one
// check instead of five.
function modelOf(llm) {
  const model = llm && llm.model;
  if (!model) return null;
  return {
    name: model.name || null,
    quant: model.quant || null,
    ctxSize: Number(model.ctxSize) || null,
    vision: !!model.vision,
    ready: !!(llm && llm.ready),
  };
}

module.exports = { statsFilePayload };
