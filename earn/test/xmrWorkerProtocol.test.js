'use strict';

const w = require('../src/shared/xmr/worker');
const { parseJob } = require('../src/shared/xmr/stratum');

const SEED = 'ab'.repeat(32);
const BLOB = 'cd'.repeat(76);

describe('commands to the worker', () => {
  test('the command line: threads, and priority when asked', () => {
    expect(w.workerArgs({ threads: 6 })).toEqual(['--threads', '6']);
    expect(w.workerArgs({ threads: 3.7, priority: 'low' }))
      .toEqual(['--threads', '3', '--priority', 'low']);
    expect(w.workerArgs({ threads: 0, priority: 'realtime' })).toEqual(['--threads', '1']);
    expect(w.workerArgs()).toEqual(['--threads', '1']);
  });

  test('a pool job becomes one job line with the 64-bit target', () => {
    const job = parseJob({ job_id: 'j1', blob: BLOB, target: 'b88d0600', seed_hash: SEED, algo: 'rx/0' });
    const line = w.jobCommand(job);
    expect(line.endsWith('\n')).toBe(true);
    expect(JSON.parse(line)).toEqual({
      cmd: 'job', id: 'j1', blob: BLOB, target: '00068db8bac710cb', seed: SEED, variant: 'v1', commit: false,
      nonce: expect.any(Number),
    });
    expect(JSON.parse(w.jobCommand(job, 7)).nonce).toBe(7);
  });

  // A job sent twice from the same start finds the same shares twice.
  test('each job line starts the search at a random 32-bit nonce', () => {
    const job = parseJob({ job_id: 'j1', blob: BLOB, target: 'b88d0600', seed_hash: SEED, algo: 'rx/0' });
    const starts = new Set();
    for (let i = 0; i < 20; i++) {
      const n = JSON.parse(w.jobCommand(job)).nonce;
      expect(Number.isInteger(n) && n >= 0 && n < 2 ** 32).toBe(true);
      starts.add(n);
    }
    expect(starts.size).toBeGreaterThan(1);
    const n = w.randomNonce();
    expect(Number.isInteger(n) && n >= 0 && n < 2 ** 32).toBe(true);
  });

  test('pause, resume and quit', () => {
    expect(w.pauseCommand()).toBe('{"cmd":"pause"}\n');
    expect(w.resumeCommand()).toBe('{"cmd":"resume"}\n');
    expect(w.quitCommand()).toBe('{"cmd":"quit"}\n');
  });
});

describe('parseWorkerLine', () => {
  const p = (o) => w.parseWorkerLine(JSON.stringify(o));

  // These are the worker's own lines, as earn/xmr-worker prints them.
  test('hello', () => {
    expect(w.parseWorkerLine('{"type":"hello","protocol":1,"version":"0.5.14","randomx":"7607fb2faed2",'
      + '"cpus":4,"threads":3,"jit":true,"hardAes":true,"argon2":"avx2"}')).toEqual({
      kind: 'hello', protocol: 1, version: '0.5.14', randomx: '7607fb2faed2', cpus: 4, threads: 3, jit: true, hardAes: true,
    });
    expect(p({ type: 'hello' })).toEqual({
      kind: 'hello', protocol: null, version: '', randomx: '', cpus: null, threads: null, jit: false, hardAes: false,
    });
  });

  test('selftest', () => {
    expect(p({ type: 'selftest', ok: true, ms: 1871 })).toEqual({ kind: 'selftest', ok: true, ms: 1871, message: null });
    expect(p({ type: 'selftest', ok: false, msg: 'wrong hash' }))
      .toEqual({ kind: 'selftest', ok: false, ms: null, message: 'wrong hash' });
  });

  test('dataset building and ready', () => {
    expect(p({ type: 'dataset', state: 'building', seed: SEED, threads: 2, pages: 'thp' }))
      .toEqual({ kind: 'dataset', state: 'building', ms: null, threads: 2, pages: 'thp' });
    expect(p({ type: 'dataset', state: 'ready', ms: 42268 }))
      .toEqual({ kind: 'dataset', state: 'ready', ms: 42268, threads: null, pages: null });
    expect(p({ type: 'dataset', state: 'sideways' }).kind).toBe('unknown');
  });

  test('share, with or without a commitment, and a malformed one', () => {
    const share = { type: 'share', job: 'j1', nonce: '3a000000', result: 'ab'.repeat(32) };
    expect(p(share)).toEqual({ kind: 'share', jobId: 'j1', nonce: '3a000000', result: 'ab'.repeat(32), commitment: null });
    expect(p(Object.assign({ commitment: 'cd'.repeat(32) }, share)).commitment).toBe('cd'.repeat(32));
    expect(p(Object.assign({}, share, { nonce: '3a' })).kind).toBe('unknown');
    expect(p(Object.assign({}, share, { result: 'AB'.repeat(32) })).kind).toBe('unknown');
    expect(p(Object.assign({}, share, { job: 5 })).kind).toBe('unknown');
  });

  test('hashrate, log, error, bench', () => {
    expect(p({ type: 'hashrate', hs: 496.4, total: 9, threads: 2 })).toEqual({ kind: 'hashrate', hs: 496.4, total: 9, threads: 2 });
    expect(p({ type: 'hashrate', hs: -1 }).hs).toBe(0);
    expect(p({ type: 'hashrate' }).hs).toBe(0);
    expect(p({ type: 'log', level: 'warn', msg: 'hi' })).toEqual({ kind: 'log', level: 'warn', message: 'hi' });
    expect(p({ type: 'log', level: 'error' })).toEqual({ kind: 'log', level: 'error', message: '' });
    expect(p({ type: 'log', level: 'shout', msg: 'x' }).level).toBe('info');
    expect(p({ type: 'error', fatal: true, msg: 'no memory' })).toEqual({ kind: 'error', fatal: true, message: 'no memory' });
    expect(p({ type: 'error' })).toEqual({ kind: 'error', fatal: false, message: 'unknown error' });
    expect(p({ type: 'bench', hs: 496.4, seconds: 30 })).toEqual({ kind: 'bench', hs: 496.4, seconds: 30 });
  });

  test('never throws on junk', () => {
    expect(w.parseWorkerLine('')).toEqual({ kind: 'empty' });
    expect(w.parseWorkerLine(undefined)).toEqual({ kind: 'empty' });
    expect(w.parseWorkerLine('Segmentation fault')).toEqual({ kind: 'unparseable', raw: 'Segmentation fault' });
    expect(w.parseWorkerLine('[1]').kind).toBe('unparseable');
    expect(w.parseWorkerLine('null').kind).toBe('unparseable');
    expect(p({ type: 'what' })).toEqual({ kind: 'unknown', raw: '{"type":"what"}' });
  });

  // Measured on a rented Ryzen 9 5950X with every CPU busy: 14 threads made
  // 1.74 kH/s at idle, 3.97 at low and 6.47 at normal, with Pearl the same.
  test('the default priority: low on Linux, idle on Windows', () => {
    expect(w.defaultPriority('linux')).toBe('low');
    expect(w.defaultPriority('win32')).toBe('idle');
    expect(w.defaultPriority('darwin')).toBe('low');
  });

  test('constants', () => {
    expect(w.PROTOCOL).toBe(1);
    expect(w.PRIORITIES).toEqual(['idle', 'low', 'normal']);
  });
});
