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
      '--algo', 'pearlhash',
      '--url', 'stratum+tcp://us2.alphapool.tech:5566',
      '--user', '.rig01',
      '--password', 'x;d=524288',
    ]);
  });

  test('honors region, worker, difficulty, algo and backend overrides', () => {
    expect(buildArgs({ address: 'prl1pabc', region: 'eu1', worker: 'rig9', difficulty: 1000, algo: 'x', backend: 'ampere' })).toEqual([
      '--algo', 'x',
      '--url', 'stratum+tcp://eu1.alphapool.tech:5566',
      '--user', 'prl1pabc.rig9',
      '--password', 'x;d=1000',
      '--backend', 'ampere',
    ]);
  });

  test('an explicit endpoint wins and an empty worker drops the suffix', () => {
    expect(buildArgs({ address: 'prl1pabc', endpoint: 'custom:1', worker: '' })).toEqual([
      '--algo', 'pearlhash',
      '--url', 'stratum+tcp://custom:1',
      '--user', 'prl1pabc',
      '--password', 'x;d=524288',
    ]);
  });

  test('defaults worker when omitted', () => {
    expect(buildArgs({ address: 'prl1pabc' })).toContain('prl1pabc.rig01');
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
