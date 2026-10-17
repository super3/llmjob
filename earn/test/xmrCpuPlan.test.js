'use strict';

const {
  planCpuThreads, MIN_MEMORY, RECOMMENDED_MEMORY, L3_PER_THREAD, DATASET_BYTES,
} = require('../src/shared/xmr/cpuPlan');

const GB = 1024 ** 3;
const MB = 1024 ** 2;

describe('planCpuThreads', () => {
  test('a desktop: one CPU kept free above 6, two at 6 or fewer', () => {
    expect(planCpuThreads({ cpus: 16, memoryBytes: 32 * GB })).toEqual({
      available: 16, reserve: 1, auto: 15, threads: 15, reason: null,
    });
    expect(planCpuThreads({ cpus: 6, memoryBytes: 16 * GB })).toMatchObject({ reserve: 2, auto: 4, threads: 4 });
    expect(planCpuThreads({ cpus: 5, memoryBytes: 16 * GB })).toMatchObject({ reserve: 2, auto: 3, threads: 3 });
  });

  // The measured case: SRBMiner started 256 threads in an 18-CPU container and
  // Pearl fell from 64 to 44 TH/s. The container's limit is the count.
  test('a container\'s CPU limit counts, not the host\'s cores', () => {
    expect(planCpuThreads({ cpus: 256, quotaCpus: 18.29, memoryBytes: 18 * GB })).toMatchObject({
      available: 18, auto: 17, threads: 17,
    });
    expect(planCpuThreads({ cpus: 8, quotaCpus: 64, memoryBytes: 16 * GB }).available).toBe(8);
    expect(planCpuThreads({ cpus: 8, quotaCpus: 0, memoryBytes: 16 * GB }).available).toBe(8);
    expect(planCpuThreads({ cpus: 8, quotaCpus: null, memoryBytes: 16 * GB }).available).toBe(8);
  });

  test('the default stops at 2 MiB of L3 per thread', () => {
    // A 12-thread CPU with 12 MiB of L3: six threads use all of it.
    expect(planCpuThreads({ cpus: 12, memoryBytes: 16 * GB, l3Bytes: 12 * MB })).toEqual({
      available: 12, reserve: 1, auto: 6, threads: 6,
      reason: 'one thread per 2 MB of L3 cache, and this CPU has 12 MB; more threads add power, not hashrate',
    });
    // A count set by hand is the user's call.
    expect(planCpuThreads({ cpus: 12, memoryBytes: 16 * GB, l3Bytes: 12 * MB }, 8)).toMatchObject({ threads: 8, reason: null });
    expect(planCpuThreads({ cpus: 12, memoryBytes: 16 * GB, l3Bytes: 1 * MB }).auto).toBe(1);
    expect(planCpuThreads({ cpus: 12, memoryBytes: 16 * GB, l3Bytes: 64 * MB })).toMatchObject({ auto: 11, reason: null });
    expect(planCpuThreads({ cpus: 12, memoryBytes: 16 * GB, l3Bytes: 0 }).auto).toBe(11);
  });

  test('off by default with 4 CPUs or fewer, or under 8 GB', () => {
    const small = planCpuThreads({ cpus: 4, memoryBytes: 16 * GB });
    expect(small).toMatchObject({ auto: 0, threads: 0 });
    expect(small.reason).toBe('off by default with 4 CPUs (it turns on by itself above 4 CPUs and 8 GB); set a thread count to mine anyway');
    const low = planCpuThreads({ cpus: 8, memoryBytes: 6 * GB });
    expect(low).toMatchObject({ auto: 0, threads: 0 });
    expect(low.reason).toBe('off by default with 8 CPUs and 6.0 GB of memory (it turns on by itself above 4 CPUs and 8 GB); set a thread count to mine anyway');
  });

  test('a thread count set by hand mines there anyway, up to all CPUs but one', () => {
    expect(planCpuThreads({ cpus: 4, memoryBytes: 16 * GB }, 2)).toMatchObject({ auto: 0, threads: 2, reason: null });
    expect(planCpuThreads({ cpus: 4, memoryBytes: 16 * GB }, 4)).toEqual({
      available: 4, reserve: 2, auto: 0, threads: 3,
      reason: '4 threads asked for; running 3 so one of the 4 CPUs stays free for the GPU miner',
    });
    expect(planCpuThreads({ cpus: 2, memoryBytes: 16 * GB }, 9).reason)
      .toBe('9 threads asked for; running 1 so one of the 2 CPUs stays free for the GPU miner');
    expect(planCpuThreads({ cpus: 16, memoryBytes: 32 * GB }, 3)).toMatchObject({ auto: 15, threads: 3 });
    expect(planCpuThreads({ cpus: 16, memoryBytes: 32 * GB }, '2').threads).toBe(2);
  });

  test('never on a single CPU, or without room for the dataset', () => {
    expect(planCpuThreads({ cpus: 1, memoryBytes: 16 * GB }, 1)).toEqual({
      available: 1, reserve: 2, auto: 0, threads: 0,
      reason: 'this app may use only 1 CPU here, and one always stays free for the GPU miner',
    });
    expect(planCpuThreads({ cpus: 64, quotaCpus: 0.5, memoryBytes: 16 * GB }).threads).toBe(0);
    expect(planCpuThreads({ cpus: 8, memoryBytes: 3.4 * GB }, 4)).toMatchObject({
      threads: 0,
      reason: 'it needs at least 4 GB of memory (2.4 GB for RandomX, plus the app and the OS), and this computer has 3.4 GB',
    });
  });

  // The OS reports less than the RAM fitted: 8 GB reads as about 7.7 GiB.
  test('memory as the OS reports it: 4 GB and 8 GB computers count as such', () => {
    expect(planCpuThreads({ cpus: 8, memoryBytes: 7.7 * GB })).toMatchObject({ auto: 7, threads: 7 });
    expect(planCpuThreads({ cpus: 8, memoryBytes: 3.8 * GB }, 2)).toMatchObject({ threads: 2, reason: null });
    expect(planCpuThreads({ cpus: 8, memoryBytes: 3.8 * GB }).auto).toBe(0);
  });

  test('unknown memory does not hold it back, and missing facts are one CPU', () => {
    expect(planCpuThreads({ cpus: 8 }).threads).toBe(7);
    expect(planCpuThreads(null).threads).toBe(0);
    expect(planCpuThreads({ cpus: 'many' }).available).toBe(1);
  });

  test('constants', () => {
    expect(DATASET_BYTES).toBe(2080 * MB);
    expect(MIN_MEMORY).toBe(3.5 * GB);
    expect(RECOMMENDED_MEMORY).toBe(7 * GB);
    expect(L3_PER_THREAD).toBe(2 * MB);
  });
});
