'use strict';

const R = require('../src/shared/rental');

const IMAGE = 'ghcr.io/super3/llmjob-bench@sha256:' + 'c'.repeat(64);

describe('rental rules', () => {
  test('imageProblem: only our registry, only by digest', () => {
    expect(R.imageProblem(IMAGE)).toBeNull();
    expect(R.imageProblem(undefined)).toMatch(/not from/);
    expect(R.imageProblem('docker.io/x@sha256:' + 'c'.repeat(64))).toMatch(/not from/);
    expect(R.imageProblem('ghcr.io/super3/llmjob-bench:latest')).toMatch(/digest/);
    expect(R.imageProblem('reg.example/x@sha256:' + 'c'.repeat(64), 'reg.example/')).toBeNull();
  });

  test('envProblem', () => {
    expect(R.envProblem({ A: '1' })).toBeNull();
    expect(R.envProblem(null)).toMatch(/object/);
    expect(R.envProblem([])).toMatch(/object/);
    expect(R.envProblem({ 'A B': '1' })).toMatch(/bad env key/);
  });

  test('containerName keeps only safe characters', () => {
    expect(R.containerName('cr_ab12')).toBe('llmjob-run-cr_ab12');
    expect(R.containerName('a/b;c')).toBe('llmjob-run-abc');
  });

  test('capsFor takes half the threads and half the RAM', () => {
    expect(R.capsFor(16, 32 * 2 ** 30)).toEqual({ cpus: 8, memoryMb: 16384 });
    expect(R.capsFor(1, 0)).toEqual({ cpus: 1, memoryMb: 1024 });
    expect(R.capsFor(undefined, undefined)).toEqual({ cpus: 1, memoryMb: 1024 });
  });

  test('dockerRunArgs: one card, no mounts, not privileged, capped', () => {
    const caps = { cpus: 4, memoryMb: 8192 };
    const nv = R.dockerRunArgs({ name: 'n', image: IMAGE, env: { A: 'secret' }, card: 2, vendor: 'nvidia', caps });
    expect(nv).toEqual([
      'run', '-d', '--name', 'n', '--security-opt', 'no-new-privileges', '--cpus', '4', '--memory', '8192m',
      '--pids-limit', '1024', '--label', 'llmjob.rental=1', '--gpus', 'device=2', '-e', 'A', IMAGE,
    ]);
    // The value never appears in the arguments.
    expect(nv.join(' ')).not.toMatch(/secret/);
    expect(nv.join(' ')).not.toMatch(/privileged|-v |--volume|--mount/);
    const amd = R.dockerRunArgs({ name: 'n', image: IMAGE, env: {}, card: 1, vendor: 'amd', caps });
    expect(amd).toEqual(expect.arrayContaining(['--device', '/dev/kfd', '/dev/dri', '--group-add', 'video', 'HIP_VISIBLE_DEVICES=1']));
    expect(amd).not.toContain('--gpus');
  });

  test('tempProblem: 85 core, 95 hotspot, fan stopped under load', () => {
    expect(R.tempProblem(undefined)).toBeNull();
    expect(R.tempProblem({ core: 70, hotspot: 80, fan: 40, util: 99 })).toBeNull();
    expect(R.tempProblem({ core: 85 })).toMatch(/core at 85/);
    expect(R.tempProblem({ core: null, hotspot: 95 })).toMatch(/hotspot at 95/);
    expect(R.tempProblem({ fan: 0, util: 60 })).toMatch(/fan stopped/);
    expect(R.tempProblem({ fan: 0, util: 10 })).toBeNull();
    expect(R.tempProblem({ fan: 0, util: null })).toBeNull();
  });

  test('formatLeft and rentedLabel', () => {
    expect(R.formatLeft(-5)).toBe('<1m');
    expect(R.formatLeft(42 * 60000 + 5)).toBe('42m');
    expect(R.formatLeft(65 * 60000)).toBe('1h 05m');
    expect(R.rentedLabel(null, 0)).toBeNull();
    expect(R.rentedLabel({ state: 'pulling' }, 0)).toBe('Rented: starting a container');
    expect(R.rentedLabel({ state: 'running', endsAt: 30 * 60000 }, 0)).toBe('Rented: running a container · 30m left');
  });

  test('rentalUnavailable says so off Linux', () => {
    expect(R.rentalUnavailable('linux')).toBeNull();
    expect(R.rentalUnavailable('win32')).toBe('Rented containers: not available on Windows yet');
    expect(R.rentalUnavailable('darwin')).toMatch(/not available on this system/);
  });

  test('splitLines carries a partial line', () => {
    expect(R.splitLines('', 'a\r\nb\npart')).toEqual({ lines: ['a', 'b'], rest: 'part' });
    expect(R.splitLines('par', 't\n')).toEqual({ lines: ['part'], rest: '' });
    expect(R.splitLines('', 'x'.repeat(3000) + '\n').lines[0]).toHaveLength(R.MAX_LINE_CHARS);
  });
});
