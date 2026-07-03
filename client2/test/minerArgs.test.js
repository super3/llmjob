'use strict';

const { resolveBinary, buildArgs, buildEnv } = require('../src/shared/minerArgs');

describe('resolveBinary', () => {
  test('prefers a configured binary path', () => {
    expect(resolveBinary('/opt/alpha-miner', 'win32')).toBe('/opt/alpha-miner');
  });

  test('uses the right Windows engine per GPU vendor', () => {
    expect(resolveBinary(null, 'win32')).toBe('alpha-miner-windows.exe');
    expect(resolveBinary(null, 'win32', 'amd')).toBe('alpha-miner-amd-windows-fixed.exe');
  });

  test('uses the bare name off Windows', () => {
    expect(resolveBinary('', 'linux')).toBe('alpha-miner');
    expect(resolveBinary(undefined, 'darwin')).toBe('alpha-miner');
  });
});

describe('buildArgs', () => {
  test('uses defaults when called with no settings', () => {
    expect(buildArgs()).toEqual([
      '--pool', 'stratum+tcp://us2.alphapool.tech:5566',
      '--address', '',
      '--worker', 'rig01',
      '--password', 'x;d=524288',
    ]);
  });

  test('honors region, worker, difficulty and backend overrides', () => {
    expect(buildArgs({ address: 'prl1pabc', region: 'eu1', worker: 'rig9', difficulty: 1000, backend: 'ampere' })).toEqual([
      '--pool', 'stratum+tcp://eu1.alphapool.tech:5566',
      '--address', 'prl1pabc',
      '--worker', 'rig9',
      '--password', 'x;d=1000',
      '--force-backend', 'ampere',
    ]);
  });

  test('an explicit endpoint wins and an empty worker drops the --worker flag', () => {
    expect(buildArgs({ address: 'prl1pabc', endpoint: 'custom:1', worker: '' })).toEqual([
      '--pool', 'stratum+tcp://custom:1',
      '--address', 'prl1pabc',
      '--password', 'x;d=524288',
    ]);
  });

  test('defaults the worker when omitted', () => {
    const args = buildArgs({ address: 'prl1pabc' });
    expect(args).toEqual(expect.arrayContaining(['--address', 'prl1pabc', '--worker', 'rig01']));
    expect(args).not.toContain('--algo');
  });
});

describe('buildEnv', () => {
  test('maps settings to the launcher environment variables', () => {
    expect(buildEnv({ address: 'prl1pabc', worker: 'rig9', difficulty: 1000 })).toEqual({
      PRL_ADDRESS: 'prl1pabc',
      WORKER: 'rig9',
      PEARL_DIFFICULTY: '1000',
    });
  });

  test('applies defaults and keeps an explicit empty worker', () => {
    expect(buildEnv()).toEqual({ PRL_ADDRESS: '', WORKER: 'rig01', PEARL_DIFFICULTY: '524288' });
    expect(buildEnv({ worker: '' }).WORKER).toBe('');
  });
});
