'use strict';

// How the AMD core's search threads wait for their cards (PEARL_WAIT in
// native/amd/src/pearl_host_hip.cpp). They busy-wait by default, a whole CPU
// core a card, which is the fastest wait when there are cores to spare. On a
// rig with fewer cores than cards that costs the rig its CPU: issue #308's six
// RX 6000 cards on a 2-core G4400 had six search threads at 25-30% CPU each and
// a load average near 7. There the threads sleep instead (block).
//
// Spin stays the default where cores are plenty, because block has not been
// raced against spin on an AMD card yet and the hot path should not change
// unmeasured. One core more than the cards is the line: each spinning thread
// takes a core, and the miner's own JavaScript thread needs one too.

// The wait to set for `cards` AMD cards on `cpus` usable CPUs, or null to leave
// the core's default. Nothing changes when PEARL_WAIT is already set, by the
// user or by --gpu-wait.
function amdWait({ cards, cpus, env }) {
  if (!(cards > 0) || (env && env.PEARL_WAIT)) return null;
  if (!(cpus > 0) || cpus >= cards + 1) return null;
  return {
    mode: 'block',
    line: 'gpu wait:   block: ' + cpus + (cpus === 1 ? ' CPU' : ' CPUs') + ' for ' + cards
      + (cards === 1 ? ' card' : ' cards') + ', so each card\'s thread sleeps between batches instead of '
      + 'taking a CPU core (--gpu-wait spin to change it)',
  };
}

module.exports = { amdWait };
