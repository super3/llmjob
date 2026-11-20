'use strict';

// Payload written to the CLI's --stats-file path so external consumers — the
// HiveOS custom-miner h-stats.sh hook in particular — can read live telemetry
// without parsing miner logs. Pure so it's unit-testable; the CLI wires the
// timer and the atomic write around it.
//
// Units: `ths` and each card's `hashrate` in `gpus` are in TH/s (the snapshot's
// native unit). Consumers convert. HiveOS's h-stats.sh multiplies by 1e9 and
// sends kH/s for both `khs` and each card's `hs`, with hs_units "khs". Keep hs
// in kH/s: HiveOS range-checks it as kH/s whatever hs_units says.
//
// A Quantus run (meta.coin 'quantus', the CLI's --coin quantus) has its
// snapshot in H/s. Its file says algo "quantus" and carries the rate in MH/s,
// Quantus's own unit: `mhs`, and `mhs` on each card, which h-stats.sh reads
// first. `ths` and each card's `hashrate` stay in TH/s, so the file means the
// same to a reader that knows only those.
const QUANTUS = 'quantus';

function statsFilePayload(snap, meta) {
  const s = snap || {};
  const m = meta || {};
  const quantus = m.coin === QUANTUS;
  // The snapshot's rate in TH/s: a Quantus snapshot is in H/s.
  const toThs = quantus ? (v) => v / 1e12 : (v) => v;
  const total = Number(s.total);
  const rate = Number.isFinite(total) && total >= 0 ? total : 0;
  return {
    ver: String(m.version || ''),
    algo: quantus ? QUANTUS : 'pearlhash',
    ths: toThs(rate),
    ...(quantus ? { mhs: rate / 1e6 } : null),
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
    // One row per mining card. Each also gets its PCI bus id (`pciBusId`,
    // nvidia-smi's "00000000:01:00.0", or null when unknown) from
    // meta.pciBusIds, keyed by the card's index. HiveOS's h-stats.sh sends it as
    // bus_numbers, which is how the dashboard puts each card's hashrate on that
    // card's row instead of matching by position.
    gpus: Array.isArray(s.gpus) ? s.gpus.map((g) => withBusId(quantus ? quantusCard(g, toThs) : g, m.pciBusIds)) : [],
    // The CPU miner (Monero), or null when it isn't on. `hs` is in H/s, not
    // TH/s like everything above. h-stats.sh sends it to HiveOS as the rig's
    // second algorithm.
    xmr: xmrOf(m.xmr),
  };
}

function xmrOf(x) {
  if (!x) return null;
  return {
    algo: 'randomx',
    hs: Math.max(0, Number(x.hashrate) || 0),
    accepted: Number(x.accepted) || 0,
    rejected: Number(x.rejected) || 0,
    threads: Number(x.threads) || 0,
    state: x.state || null,
  };
}

// A Quantus card's row: its H/s as TH/s in `hashrate`, as every row has it,
// and as MH/s in `mhs`.
function quantusCard(card, toThs) {
  const hs = Number(card && card.hashrate) || 0;
  return Object.assign({}, card, { hashrate: toThs(hs), mhs: hs / 1e6 });
}

function withBusId(card, busIds) {
  const id = busIds && card && busIds[card.index];
  return Object.assign({}, card, { pciBusId: id || null });
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
