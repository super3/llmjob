'use strict';

const {
  parseCpuMax, parseCfs, parseCacheSize, cgroupDirs, readQuotaCpus, readL3Bytes, readCpuFacts,
} = require('../src/main/cpuFacts');

// A filesystem made of a { path: contents } map, with directories listed in
// `dirs`. Anything else does not exist.
function fakeFs(files, dirs = {}) {
  return {
    readFileSync(p) {
      if (!(p in files)) throw Object.assign(new Error('ENOENT ' + p), { code: 'ENOENT' });
      return files[p];
    },
    readdirSync(p) {
      if (!(p in dirs)) throw Object.assign(new Error('ENOENT ' + p), { code: 'ENOENT' });
      return dirs[p];
    },
  };
}

describe('parsers', () => {
  test('cgroup v2 cpu.max', () => {
    expect(parseCpuMax('max 100000')).toBeNull();
    expect(parseCpuMax('1800000 100000')).toBe(18);
    expect(parseCpuMax('1344000 100000\n')).toBeCloseTo(13.44);
    expect(parseCpuMax('100000 0')).toBeNull();
    expect(parseCpuMax('')).toBeNull();
    expect(parseCpuMax(null)).toBeNull();
  });

  test('cgroup v1 cfs quota and period', () => {
    expect(parseCfs('-1', '100000')).toBeNull();
    expect(parseCfs('400000', '100000')).toBe(4);
    expect(parseCfs('50000', '0')).toBeNull();
    expect(parseCfs(null, null)).toBeNull();
  });

  test('sysfs cache sizes', () => {
    expect(parseCacheSize('33792K')).toBe(33792 * 1024);
    expect(parseCacheSize('8M')).toBe(8 * 1024 * 1024);
    expect(parseCacheSize('1g')).toBe(1024 ** 3);
    expect(parseCacheSize('4096')).toBe(4096);
    expect(parseCacheSize('big')).toBeNull();
    expect(parseCacheSize(undefined)).toBeNull();
  });

  test('a cgroup and every parent up to the root', () => {
    expect(cgroupDirs('/sys/fs/cgroup', '/docker/abc')).toEqual([
      '/sys/fs/cgroup/docker/abc', '/sys/fs/cgroup/docker', '/sys/fs/cgroup',
    ]);
    expect(cgroupDirs('/sys/fs/cgroup', '/')).toEqual(['/sys/fs/cgroup']);
    expect(cgroupDirs('/sys/fs/cgroup', '')).toEqual(['/sys/fs/cgroup']);
  });
});

describe('readQuotaCpus', () => {
  // The SRBMiner host: an 18-CPU limit on a 256-core EPYC, cgroup v2.
  test('cgroup v2: the smallest limit on the way to the root', () => {
    const fs = fakeFs({
      '/proc/self/cgroup': '0::/docker/abc\n',
      '/sys/fs/cgroup/docker/abc/cpu.max': '1828571 100000',
      '/sys/fs/cgroup/docker/cpu.max': '6400000 100000',
      '/sys/fs/cgroup/cpu.max': 'max 100000',
    });
    expect(readQuotaCpus(fs)).toBeCloseTo(18.28571);
  });

  test('cgroup v1: the cpu controller, alone or with cpuacct', () => {
    const fs = fakeFs({
      '/proc/self/cgroup': '9:memory:/x\n4:cpu,cpuacct:/docker/abc\ngarbage\n',
      '/sys/fs/cgroup/cpu,cpuacct/docker/abc/cpu.cfs_quota_us': '400000',
      '/sys/fs/cgroup/cpu,cpuacct/docker/abc/cpu.cfs_period_us': '100000',
      '/sys/fs/cgroup/cpu/cpu.cfs_quota_us': '-1',
      '/sys/fs/cgroup/cpu/cpu.cfs_period_us': '100000',
    });
    expect(readQuotaCpus(fs)).toBe(4);
  });

  test('no limit anywhere, or no /proc at all', () => {
    expect(readQuotaCpus(fakeFs({ '/proc/self/cgroup': '0::/\n', '/sys/fs/cgroup/cpu.max': 'max 100000' }))).toBeNull();
    expect(readQuotaCpus(fakeFs({}))).toBeNull();
  });
});

describe('readL3Bytes', () => {
  const cpuDirs = {
    '/sys/devices/system/cpu': ['cpu0', 'cpu1', 'cpu2', 'cpu3', 'cpufreq', 'online'],
    '/sys/devices/system/cpu/cpu0/cache': ['index0', 'index3', 'uevent'],
    '/sys/devices/system/cpu/cpu1/cache': ['index3'],
    '/sys/devices/system/cpu/cpu2/cache': ['index2', 'index3'],
  };
  const l3 = (cpu, size, shared) => ({
    ['/sys/devices/system/cpu/' + cpu + '/cache/index3/level']: '3',
    ['/sys/devices/system/cpu/' + cpu + '/cache/index3/size']: size,
    ['/sys/devices/system/cpu/' + cpu + '/cache/index3/shared_cpu_list']: shared,
  });

  // Two sockets: cpu0 and cpu1 share one 16 MiB L3, cpu2 has its own 32 MiB
  // one, and cpu3 has no cache directory. Each cache counts once.
  test('adds up each L3 once, across sockets', () => {
    const fs = fakeFs(Object.assign(
      { '/sys/devices/system/cpu/cpu0/cache/index0/level': '1', '/sys/devices/system/cpu/cpu2/cache/index2/level': '2' },
      l3('cpu0', '16384K', '0-1'), l3('cpu1', '16384K', '0-1'), l3('cpu2', '32M', '2'),
    ), cpuDirs);
    expect(readL3Bytes(fs)).toBe(48 * 1024 * 1024);
  });

  test('a cache with no shared list is its CPU\'s own; an unreadable size is skipped', () => {
    const files = l3('cpu0', '8M', '0');
    delete files['/sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list'];
    Object.assign(files, l3('cpu1', 'lots', '1'));
    expect(readL3Bytes(fakeFs(files, cpuDirs))).toBe(8 * 1024 * 1024);
  });

  test('no sysfs, or no L3 in it', () => {
    expect(readL3Bytes(fakeFs({}))).toBeNull();
    expect(readL3Bytes(fakeFs({}, cpuDirs))).toBeNull();
  });
});

describe('readCpuFacts', () => {
  const os = { availableParallelism: () => 8, totalmem: () => 32e9, cpus: () => new Array(16) };

  test('on Linux: affinity, the cgroup limit, memory under its limit, and L3', () => {
    const fs = fakeFs({ '/proc/self/cgroup': '0::/', '/sys/fs/cgroup/cpu.max': '600000 100000' });
    expect(readCpuFacts({ fs, os, proc: { platform: 'linux', constrainedMemory: () => 16e9 } })).toEqual({
      cpus: 8, quotaCpus: 6, memoryBytes: 16e9, l3Bytes: null,
    });
  });

  test('a memory "limit" above the real memory is not one', () => {
    const fs = fakeFs({});
    expect(readCpuFacts({ fs, os, proc: { platform: 'linux', constrainedMemory: () => 2 ** 62 } }).memoryBytes).toBe(32e9);
    expect(readCpuFacts({ fs, os, proc: { platform: 'linux', constrainedMemory: () => 0 } }).memoryBytes).toBe(32e9);
    expect(readCpuFacts({ fs, os, proc: { platform: 'linux' } }).memoryBytes).toBe(32e9);
  });

  test('on Windows nothing is read from the filesystem', () => {
    const fs = { readFileSync: () => { throw new Error('no'); }, readdirSync: () => { throw new Error('no'); } };
    const oldOs = { totalmem: () => 8e9, cpus: () => new Array(4) };
    expect(readCpuFacts({ fs, os: oldOs, proc: {}, platform: 'win32' })).toEqual({
      cpus: 4, quotaCpus: null, memoryBytes: 8e9, l3Bytes: null,
    });
  });

  test('defaults to this process', () => {
    const facts = readCpuFacts({ fs: require('fs'), os: require('os') });
    expect(facts.cpus).toBeGreaterThan(0);
    expect(facts.memoryBytes).toBeGreaterThan(0);
  });
});
