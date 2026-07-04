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

function parseLine(line) {
  const s = String(line == null ? '' : line).trim();
  if (!s) return null;

  // Periodic miner status: hashrate + cumulative accepted/rejected + GPU.
  if (/\bhashrate_th_s=/.test(s)) {
    return {
      type: 'status',
      hashrate: numField(s, 'hashrate_th_s'),
      accepted: numField(s, 'accepted'),
      rejected: numField(s, 'rejected'),
      power: numField(s, 'power'),
      gpu: gpuName(s),
    };
  }

  // Pool connection.
  const conn = s.match(/component=pool\s+connected\s+host=(\S+)\s+port=(\d+)/i);
  if (conn) {
    return { type: 'connected', endpoint: conn[1] + ':' + conn[2], gpu: gpuName(s) };
  }

  return null;
}

module.exports = { numField, gpuName, parseLine };
