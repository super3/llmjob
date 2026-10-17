'use strict';

// What the CPU miner's thread plan (shared/xmr/cpuPlan) needs to know about
// this machine, read from the OS. Everything is injected so the tests can hand
// it any /proc and /sys they like.
//
//   cpus        os.availableParallelism(): the CPUs in this process's affinity
//               mask, which is what taskset or a container's cpuset leaves it
//   quotaCpus   the cgroup CPU limit (cgroup v2 cpu.max, v1 cfs_quota_us),
//               in CPUs. A container limited to 18 CPUs on a 128-core host
//               still sees 128 in its affinity mask, and only this says 18.
//   memoryBytes os.totalmem(), or the cgroup memory limit when lower
//   l3Bytes     the L3 cache across all sockets, from sysfs (Linux only)

const pathPosix = require('path').posix;

function readText(fs, file) {
  try {
    return String(fs.readFileSync(file, 'utf8')).trim();
  } catch (e) {
    return null;
  }
}

// cgroup v2 cpu.max: "max 100000" for no limit, or "<quota> <period>".
function parseCpuMax(text) {
  const m = /^(\d+|max)\s+(\d+)$/.exec(String(text || '').trim());
  if (!m || m[1] === 'max' || Number(m[2]) <= 0) return null;
  return Number(m[1]) / Number(m[2]);
}

// cgroup v1: cfs_quota_us is -1 for no limit.
function parseCfs(quotaText, periodText) {
  const q = Number(quotaText);
  const p = Number(periodText);
  if (!Number.isFinite(q) || q <= 0 || !Number.isFinite(p) || p <= 0) return null;
  return q / p;
}

// This cgroup and every parent up to the root, as directories under `root`. A
// limit set on any of them applies, so the smallest wins.
function cgroupDirs(root, cgPath) {
  const dirs = [];
  let p = pathPosix.normalize('/' + (cgPath || ''));
  for (;;) {
    dirs.push(p === '/' ? root : pathPosix.join(root, p));
    if (p === '/') break;
    p = pathPosix.dirname(p);
  }
  return dirs;
}

function readQuotaCpus(fs) {
  const lines = (readText(fs, '/proc/self/cgroup') || '').split('\n');
  const quotas = [];
  for (const line of lines) {
    const m = /^\d+:([^:]*):(.*)$/.exec(line.trim());
    if (!m) continue;
    const controllers = m[1].split(',');
    if (m[1] === '') {
      // cgroup v2: one unified hierarchy.
      for (const dir of cgroupDirs('/sys/fs/cgroup', m[2])) {
        quotas.push(parseCpuMax(readText(fs, dir + '/cpu.max')));
      }
    } else if (controllers.includes('cpu')) {
      // cgroup v1: the cpu controller, mounted alone or with cpuacct.
      for (const base of ['/sys/fs/cgroup/cpu', '/sys/fs/cgroup/cpu,cpuacct']) {
        for (const dir of cgroupDirs(base, m[2])) {
          quotas.push(parseCfs(readText(fs, dir + '/cpu.cfs_quota_us'), readText(fs, dir + '/cpu.cfs_period_us')));
        }
      }
    }
  }
  const found = quotas.filter((q) => q != null);
  return found.length ? Math.min(...found) : null;
}

// sysfs cache sizes: "32768K", "8M", or bytes.
function parseCacheSize(text) {
  const m = /^(\d+)\s*([KMG]?)$/i.exec(String(text || '').trim());
  if (!m) return null;
  const mult = { '': 1, K: 1024, M: 1024 * 1024, G: 1024 * 1024 * 1024 }[m[2].toUpperCase()];
  return Number(m[1]) * mult;
}

// Every distinct L3 cache, added up. Each CPU lists the caches it uses, and
// the CPUs sharing one L3 all list it, so caches are told apart by their
// shared_cpu_list.
function readL3Bytes(fs) {
  let entries;
  try {
    entries = fs.readdirSync('/sys/devices/system/cpu');
  } catch (e) {
    return null;
  }
  const seen = new Map();
  for (const name of entries) {
    if (!/^cpu\d+$/.test(name)) continue;
    const base = '/sys/devices/system/cpu/' + name + '/cache';
    let indexes;
    try {
      indexes = fs.readdirSync(base);
    } catch (e) {
      continue;
    }
    for (const idx of indexes) {
      if (!/^index\d+$/.test(idx)) continue;
      const dir = base + '/' + idx;
      if (readText(fs, dir + '/level') !== '3') continue;
      const size = parseCacheSize(readText(fs, dir + '/size'));
      const shared = readText(fs, dir + '/shared_cpu_list') || name;
      if (size) seen.set(shared, size);
    }
  }
  if (!seen.size) return null;
  let total = 0;
  for (const size of seen.values()) total += size;
  return total;
}

function readCpuFacts({ fs, os, proc, platform }) {
  const p = proc || process;
  const cpus = typeof os.availableParallelism === 'function' ? os.availableParallelism() : os.cpus().length;
  let memoryBytes = os.totalmem();
  const limit = typeof p.constrainedMemory === 'function' ? Number(p.constrainedMemory()) : 0;
  if (limit > 0 && limit < memoryBytes) memoryBytes = limit;
  const linux = (platform || p.platform) === 'linux';
  return {
    cpus,
    quotaCpus: linux ? readQuotaCpus(fs) : null,
    memoryBytes,
    l3Bytes: linux ? readL3Bytes(fs) : null,
  };
}

module.exports = { parseCpuMax, parseCfs, parseCacheSize, cgroupDirs, readQuotaCpus, readL3Bytes, readCpuFacts };
