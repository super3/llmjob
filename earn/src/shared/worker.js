'use strict';

// The default worker name for this machine.
//
// A worker name is half of the board's identity for a rig: the server keys miner
// rows on (address, worker), and groups a multi-GPU host's "<worker>/gpuN" rows
// back together by the shared base. So two machines mining the same payout
// address under the SAME worker name are indistinguishable to the server, and
// collide:
//   • both single-GPU  → identical row id; each ping overwrites the other's row,
//                        so the board shows one flip-flopping entry
//   • one multi-GPU    → the multi-GPU host's per-card rows make the other
//                        machine's bare row look like a stale leftover, and it is
//                        dropped from the board entirely along with its hashrate
//
// Deriving from the hostname makes them distinct by default. A shared constant
// like "rig01" guarantees the collision for anyone who runs a second machine
// without renaming it — which is the default path, since nothing in either shell
// prompts for a worker name.
//
// Kept here rather than in config.js so both shells share ONE implementation:
// the CLI had this and the GUI didn't, which is exactly how the two drifted.

const os = require('os');
const { DEFAULTS } = require('./config');

// This machine's hostname reduced to a safe stratum token: first DNS label,
// lowercased, non [a-z0-9-] stripped, trimmed of leading/trailing dashes and
// capped at 32 chars. Falls back to the shared constant when the hostname is
// missing or reduces to nothing (a hostname of "!!!" and an unset one are both
// unusable), so the result is always a valid worker name.
function defaultWorker(hostname) {
  const raw = hostname != null ? hostname : os.hostname();
  const host = String(raw || '').trim().toLowerCase().split('.')[0];
  const name = host.replace(/[^a-z0-9-]/g, '').replace(/^-+|-+$/g, '').slice(0, 32);
  return name || DEFAULTS.worker;
}

module.exports = { defaultWorker };
