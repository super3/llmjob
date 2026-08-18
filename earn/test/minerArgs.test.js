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

// The rank-128 CLI (alpha-miner 1.9.4 on Windows). Selected by the engine
// descriptor's `cli` field, not a version comparison, so these also pin the
// dispatch: an engine that does not declare it keeps the 1.8.x vector.
describe('buildArgs on the worker-address CLI', () => {
  const PRL = 'prl1pql8r6m4z9x7v2k0t3whu8e2snd4p6c';
  const win = { platform: 'win32', engineVersion: '1.9.4' };

  test('carries the payout address inside --worker and pins the card', () => {
    expect(buildArgs(Object.assign({ address: PRL, worker: 'rig9', difficulty: 1000, gpuIndex: 2 }, win)))
      .toEqual(['--host', 'us2.alphapool.tech', '--port', '5566', '--worker', PRL + '.rig9', '--password', 'x;d=1000', '--gpu', '2']);
  });

  test('sends a bare address when the worker is blank', () => {
    expect(buildArgs(Object.assign({ address: PRL, worker: '' }, win))).toContain(PRL);
  });

  // The guard that keeps every pre-fork engine on its own vector.
  test('leaves a legacy engine on the --pool/--address vector', () => {
    expect(buildArgs({ address: PRL, platform: 'win32', engineVersion: '1.8.6' })[0]).toBe('--pool');
  });
});

// The field report this came from: an endpoint override pasted in the old
// `stratum+tcp://` form reached --host verbatim, so the engine tried to resolve
// the scheme as part of the hostname and looped on 'No such host is known'.
test('an endpoint override keeps its scheme out of --host', () => {
  const eng = require('../src/shared/engine');
  const args = buildArgs({
    address: 'prl1pql8r6m4z9x7v2k0t3whu8e2snd4p6c', worker: 'rig01',
    platform: 'win32', engineVersion: eng.ENGINE.windows,
    endpoint: 'stratum+tcp://us1.alphapool.tech:5566',
  });
  expect(args[args.indexOf('--host') + 1]).toBe('us1.alphapool.tech');
  expect(args[args.indexOf('--port') + 1]).toBe('5566');
});

// The legacy vector builds its own stratum+tcp:// prefix, so a cleaned endpoint
// must not leave it with a doubled scheme.
test('the legacy --pool vector still gets exactly one scheme', () => {
  const args = buildArgs({
    address: 'prl1pql8r6m4z9x7v2k0t3whu8e2snd4p6c', platform: 'linux',
    engineVersion: '1.8.8', endpoint: 'stratum+tcp://us1.alphapool.tech:5566',
  });
  expect(args[args.indexOf('--pool') + 1]).toBe('stratum+tcp://us1.alphapool.tech:5566');
});

// A portless endpoint must not gain an invented port: the engine has its own
// default, and guessing one is how the doubled-port bug happened in reverse.
test('omits --port entirely when the endpoint carries none', () => {
  const eng = require('../src/shared/engine');
  const args = buildArgs({
    address: 'prl1pql8r6m4z9x7v2k0t3whu8e2snd4p6c', worker: 'rig01',
    platform: 'win32', engineVersion: eng.ENGINE.windows,
    endpoint: 'pool.example.internal',
  });
  expect(args).not.toContain('--port');
  expect(args[args.indexOf('--host') + 1]).toBe('pool.example.internal');
});

// The GUI blocks START without a valid address and the CLI errors out, so this
// is a degenerate shape rather than a real one — but the vector still has to be
// well-formed rather than containing "undefined", which the pool would happily
// accept as a login and credit to nobody.
test('a missing address yields an empty login rather than "undefined"', () => {
  const eng = require('../src/shared/engine');
  const args = buildArgs({ worker: 'rig01', platform: 'win32', engineVersion: eng.ENGINE.windows });
  expect(args[args.indexOf('--worker') + 1]).toBe('.rig01');
  expect(args.join(' ')).not.toMatch(/undefined|null/);
});
