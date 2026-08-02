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
const crypto = require('crypto');
const { DEFAULTS } = require('./config');

// This machine's hostname reduced to a safe stratum token: first DNS label,
// lowercased, non [a-z0-9-] stripped, trimmed of leading/trailing dashes and
// capped at 32 chars.
//
// When that leaves nothing usable, derive a stable token from the hostname
// instead of falling back to the shared constant. Stripping to ASCII erases a
// non-Latin hostname entirely — "Домашний-ПК" and "Рабочий-ПК" both reduce to
// "" — so the constant would put every such machine back on one name and
// recreate the collision this function exists to prevent, for exactly the users
// least likely to notice. The hash is over the original hostname, so it is
// stable across runs and distinct per machine.
//
// Only the unusable branch changed: a hostname that already sanitises to a
// real token keeps producing the same worker name it always has, so nobody's
// established board row is renamed. A genuinely empty hostname still yields the
// shared default — there is nothing there to tell two machines apart with.
function defaultWorker(hostname) {
  const raw = hostname != null ? hostname : os.hostname();
  const host = String(raw || '').trim().toLowerCase().split('.')[0];
  const name = host.replace(/[^a-z0-9-]/g, '').replace(/^-+|-+$/g, '').slice(0, 32);
  if (name.length >= 2) return name;
  if (!host) return DEFAULTS.worker;
  return 'rig-' + crypto.createHash('sha256').update(host).digest('hex').slice(0, 6);
}

module.exports = { defaultWorker };
