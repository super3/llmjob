'use strict';

// Parse a line of `alpha-miner` stdout into a structured event.
//
// alpha-miner (github.com/AlphaMine-Tech/alpha-miner) emits structured
// `key=value` logs. The two we care about:
//
//   ...gpu=0:NVIDIA GeForce RTX 4090 component=miner status attempts=100 hits=3
//      accepted=3 rejected=0 dropped=0 hashrate_th_s=286.86 ... power=449W
//   ...gpu=0:NVIDIA GeForce RTX 4090 component=pool connected host=us2.alphapool.tech port=5566
//
// The periodic `miner status` line carries the authoritative cumulative share
// counters plus the live hashrate (already in TH/s) and the GPU name. Anything
// unrecognized returns null so callers pass it through as raw log text.

// A numeric `key=value` field (value may be a float; trailing units like the W
// in `power=449W` or the c in `ctemp=71c` are ignored). Returns null if absent.
function numField(s, key) {
  const m = String(s).match(new RegExp('\\b' + key + '=([\\d.]+)'));
  return m ? Number(m[1]) : null;
}

// The GPU name from a `gpu=<index>:<name> component=...` field, or null when the
// engine reports no real device (early lines say `gpu=system`).
function gpuName(s) {
  const m = String(s).match(/\bgpu=(?:\d+:)?(.+?)\s+component=/);
  if (!m) return null;
  const name = m[1].trim();
  return name.toLowerCase() === 'system' ? null : name;
}

// The 0-based card index from a `gpu=<index>:<name>` field, or null when the
// line carries no index (`gpu=system`, or a name-only `gpu=<name>`). Multi-GPU
// rigs tag every status/pool line with the card it refers to, so this is how a
// per-card accumulator knows which GPU a hashrate belongs to.
function gpuIndex(s) {
  const m = String(s).match(/\bgpu=(\d+):/);
  return m ? Number(m[1]) : null;
}

// alpha-miner 1.9.4 stopped emitting key=value logs and renders a live stats
// table instead, one row per card:
//
//    #0  RTX 5090   308.17 TH/s   70C   46%   281W   1.098   2902   13801   3   0   0
//
// (hashrate, temp, fan, power, efficiency, core clock, mem clock, then the
// cumulative accepted / rejected / ignored counters). The `Total` row carries no
// `#N` and is skipped, so a multi-GPU rig still accumulates per card.
//
// Parsed by locating the hashrate token and reading outwards rather than by
// matching the full column layout: a rig that reports no fan or clock would
// otherwise fail the whole row and silently show zero — the exact failure this
// replaces.
const ANSI = /\[[0-9;]*m/g; // eslint-disable-line no-control-regex
const HASHRATE = /([\d.]+)\s*([KMGTPE]?)H\/s/i;
const UNIT_TH = { '': 1e-12, K: 1e-9, M: 1e-6, G: 1e-3, T: 1, P: 1e3, E: 1e6 };

function parseStatsRow(s) {
  const head = s.match(/^#(\d+)\s+(.+)$/);
  if (!head) return null;
  const rest = head[2];
  const hr = rest.match(HASHRATE);
  if (!hr) return null;

  const tail = rest.slice(hr.index + hr[0].length);
  const counters = tail.trim().split(/\s+/).filter((t) => /^\d+$/.test(t)).slice(-3);
  const temp = tail.match(/(\d+)\s*C\b/);
  const power = tail.match(/(\d+)\s*W\b/);
  const name = rest.slice(0, hr.index).trim();

  return {
    type: 'status',
    gpuIndex: Number(head[1]),
    hashrate: Number(hr[1]) * UNIT_TH[hr[2].toUpperCase()],
    accepted: counters.length === 3 ? Number(counters[0]) : null,
    rejected: counters.length === 3 ? Number(counters[1]) : null,
    power: power ? Number(power[1]) : null,
    temp: temp ? Number(temp[1]) : null,
    gpu: name || null,
  };
}

function parseLine(line) {
  const s = String(line == null ? '' : line).replace(ANSI, '').trim();
  if (!s) return null;

  // Periodic miner status: hashrate + cumulative accepted/rejected + GPU.
  if (/\bhashrate_th_s=/.test(s)) {
    return {
      type: 'status',
      gpuIndex: gpuIndex(s),
      hashrate: numField(s, 'hashrate_th_s'),
      accepted: numField(s, 'accepted'),
      rejected: numField(s, 'rejected'),
      power: numField(s, 'power'),
      // Core temperature (`ctemp=86c` — numField drops the trailing c). Surfaced
      // beside the GPU name so a rig that keeps crashing can be checked for heat
      // without leaving the app for nvidia-smi.
      temp: numField(s, 'ctemp'),
      gpu: gpuName(s),
    };
  }

  // Pool connection.
  const conn = s.match(/component=pool\s+connected\s+host=(\S+)\s+port=(\d+)/i);
  if (conn) {
    return { type: 'connected', gpuIndex: gpuIndex(s), endpoint: conn[1] + ':' + conn[2], gpu: gpuName(s) };
  }

  // 1.9.4's stats table and its plainer connection line.
  const row = parseStatsRow(s);
  if (row) return row;

  const conn194 = s.match(/\[stratum\]\s+connected to\s+(\S+?):(\d+)/i);
  if (conn194) {
    return { type: 'connected', gpuIndex: null, endpoint: conn194[1] + ':' + conn194[2], gpu: null };
  }

  return null;
}

module.exports = { numField, gpuName, gpuIndex, parseLine };
