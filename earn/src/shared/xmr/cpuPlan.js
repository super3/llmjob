'use strict';

// How many threads the CPU miner runs.
//
// The GPU miner needs the CPU too: its share proofs are built in JS on the
// main thread, and each card has CUDA host threads. Measured on a rented host
// (SRBMiner on rx/0 beside our Pearl miner), a CPU miner that left enough CPU
// cost nothing, but one that started 256 threads in an 18-CPU container took
// Pearl from 64 to 44 TH/s. That container's CPU limit (cgroup cpu.max) was
// the real count, not the host's cores, and the whole container is throttled
// once it is used up, the GPU miner's threads included. So threads are counted
// from what this process may actually use, and at least one CPU always stays
// free.
//
// The default leaves two free on 6 CPUs or fewer. It is 0, so CPU mining stays
// off unless a thread count is set by hand, with 4 CPUs or fewer or under 8 GB
// of memory: mining-rig Celerons and Pentiums earn cents a day there and need
// their cores for the GPU. Each thread wants 2 MiB of L3 cache; threads past
// that add power, not hashrate, so the default stops there when L3 is known.
//
// Pure: main/cpuFacts.js reads the facts, this decides.

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;

// RandomX's fast mode: a 2080 MiB dataset, a 256 MiB cache and 2 MiB of
// scratchpad per thread. Below 4 GB of RAM the dataset doesn't fit beside the
// app and the OS, and below 8 GB it is off by default.
//
// The thresholds sit under 4 and 8 GiB because memoryBytes is what the OS
// reports (os.totalmem), which is less than the RAM fitted: the firmware, the
// kernel and an integrated GPU keep some. 8 GB of RAM reads as 7.6-7.8 GiB,
// so 8 GiB here would leave every 8 GB desktop off by default.
const DATASET_BYTES = 2080 * MIB;
const MIN_MEMORY = 3.5 * GIB;
const RECOMMENDED_MEMORY = 7 * GIB;
const L3_PER_THREAD = 2 * MIB;

function gb(bytes) {
  return (bytes / GIB).toFixed(1) + ' GB';
}

// facts: { cpus, quotaCpus, memoryBytes, l3Bytes }
//   cpus        CPUs in this process's affinity mask
//   quotaCpus   the cgroup CPU limit in CPUs (cpu.max), or null for none
//   memoryBytes total memory, or the cgroup limit when lower
//   l3Bytes     total L3 cache, or null when unknown
// requested: a thread count, or 0/null for the default.
//
// Returns { available, reserve, auto, threads, reason }:
//   available  whole CPUs this process may use
//   reserve    how many the default leaves free
//   auto       the default thread count (0: off by default here)
//   threads    what to run: the request (capped), or auto
//   reason     one line on why threads is 0 or was capped (by the request or
//              by L3 cache), or null
function planCpuThreads(facts, requested) {
  const f = facts || {};
  const cpus = Math.max(1, Math.floor(Number(f.cpus) || 1));
  const quota = Number(f.quotaCpus);
  const available = Number.isFinite(quota) && quota > 0 ? Math.max(1, Math.min(cpus, Math.floor(quota))) : cpus;
  const reserve = available <= 6 ? 2 : 1;
  const max = available - 1;
  const memory = Number(f.memoryBytes) || 0;
  const l3 = Number(f.l3Bytes);
  const l3Threads = Number.isFinite(l3) && l3 > 0 ? Math.max(1, Math.floor(l3 / L3_PER_THREAD)) : Infinity;
  const ask = Math.floor(Number(requested) || 0);
  const out = (auto, threads, reason) => ({ available, reserve, auto, threads, reason });

  if (max < 1) {
    return out(0, 0, 'this app may use only 1 CPU here, and one always stays free for the GPU miner');
  }
  if (memory > 0 && memory < MIN_MEMORY) {
    return out(0, 0, 'it needs at least 4 GB of memory (2.4 GB for RandomX, plus the app and the OS), and this'
      + ' computer has ' + gb(memory));
  }

  const recommended = available > 4 && (memory === 0 || memory >= RECOMMENDED_MEMORY);
  const auto = recommended ? Math.max(1, Math.min(available - reserve, l3Threads)) : 0;

  if (ask > 0) {
    if (ask > max) {
      // Both counts are 2 or more here: one CPU returned early above.
      return out(auto, max, ask + ' threads asked for; running ' + max
        + ' so one of the ' + available + ' CPUs stays free for the GPU miner');
    }
    return out(auto, ask, null);
  }
  if (!auto) {
    return out(0, 0, 'off by default with ' + available + ' CPUs'
      + (memory > 0 && memory < RECOMMENDED_MEMORY ? ' and ' + gb(memory) + ' of memory' : '')
      + ' (it turns on by itself above 4 CPUs and 8 GB); set a thread count to mine anyway');
  }
  // Said, or "N of M CPUs" reads as if the rest were kept for the GPU miner.
  if (auto < available - reserve) {
    return out(auto, auto, 'one thread per 2 MB of L3 cache, and this CPU has ' + Math.round(l3 / MIB)
      + ' MB; more threads add power, not hashrate');
  }
  return out(auto, auto, null);
}

module.exports = {
  DATASET_BYTES, MIN_MEMORY, RECOMMENDED_MEMORY, L3_PER_THREAD, planCpuThreads,
};
