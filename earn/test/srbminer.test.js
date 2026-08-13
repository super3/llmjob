'use strict';

const path = require('path');
const {
  SRB, PLAIN_STRATUM_PORT,
  srbArchiveName, srbDirName, srbBinaryName, srbDownloadUrl, srbBinaryPath, srbFiles,
  srbPassword, plainStratumEndpoint, srbArgs, srbApiUrl, srbStatusEvents,
} = require('../src/shared/srbminer');

// The archive layout is not guessed — 3.5.4 was downloaded and listed:
//   SRBMiner-Multi-3-5-4/SRBMiner-MULTI
describe('SRBMiner artifact layout', () => {
  test('archive, directory and binary names per platform', () => {
    expect(srbArchiveName('win32')).toBe('SRBMiner-Multi-3-5-4-win64.zip');
    expect(srbArchiveName('linux')).toBe('SRBMiner-Multi-3-5-4-Linux.tar.gz');
    expect(srbDirName()).toBe('SRBMiner-Multi-3-5-4');
    expect(srbBinaryName('win32')).toBe('SRBMiner-MULTI.exe');
    expect(srbBinaryName('linux')).toBe('SRBMiner-MULTI');
  });

  test('download URL points at the pinned GitHub release', () => {
    expect(srbDownloadUrl('linux')).toBe(
      'https://github.com/doktor83/SRBMiner-Multi/releases/download/3.5.4/SRBMiner-Multi-3-5-4-Linux.tar.gz');
    expect(srbDownloadUrl('win32')).toContain('-win64.zip');
  });

  test('the binary lives inside the archive\'s own folder', () => {
    // Same shape as the alpha-miner package: the top-level folder IS the
    // install dir, so nothing may strip components on extract.
    expect(srbBinaryPath('/cache', 'linux')).toBe(path.join('/cache', 'SRBMiner-Multi-3-5-4', 'SRBMiner-MULTI'));
    expect(srbFiles('/cache', 'win32')).toEqual([
      path.join('/cache', 'SRBMiner-Multi-3-5-4', 'SRBMiner-MULTI.exe'),
    ]);
  });
});

describe('srbPassword', () => {
  // The subtle one. SRBMiner treats ; and ! as separators BETWEEN per-pool
  // passwords, so our `x;d=524288` would be read as two passwords, the
  // difficulty pin silently dropped, and the rig left on vardiff.
  test('escapes the separators SRBMiner would otherwise split on', () => {
    expect(srbPassword('x;d=524288')).toBe('x#;d=524288');
    expect(srbPassword('x;d=524288;mdl=mdl1abc')).toBe('x#;d=524288#;mdl=mdl1abc');
    expect(srbPassword('a!b')).toBe('a#!b');
  });

  test('leaves an ordinary password and empties alone', () => {
    expect(srbPassword('x')).toBe('x');
    expect(srbPassword('')).toBe('');
    expect(srbPassword(null)).toBe('');
    expect(srbPassword(undefined)).toBe('');
  });
});

describe('plainStratumEndpoint', () => {
  test('keeps the host and swaps to the plain-stratum port', () => {
    expect(plainStratumEndpoint('us1.alphapool.tech:5566')).toBe('us1.alphapool.tech:' + PLAIN_STRATUM_PORT);
    expect(plainStratumEndpoint('eu2.alphapool.tech:5573')).toBe('eu2.alphapool.tech:5571');
    expect(plainStratumEndpoint('us1.alphapool.tech')).toBe('us1.alphapool.tech:5571');
  });

  test('nothing in, nothing out', () => {
    expect(plainStratumEndpoint('')).toBe('');
    expect(plainStratumEndpoint(null)).toBe('');
  });
});

describe('srbArgs', () => {
  // Flag spellings come from the 3.5.4 binary's own --help, not from docs.
  test('builds the documented argument vector', () => {
    expect(srbArgs({
      endpoint: 'us2.alphapool.tech:5566',
      address: 'prl1abc',
      worker: 'rig01',
      password: 'x;d=524288',
    })).toEqual([
      '--algorithm', 'pearlhash',
      '--pool', 'us2.alphapool.tech:5571',
      '--wallet', 'prl1abc',
      '--worker', 'rig01',
      '--password', 'x#;d=524288',
      '--disable-cpu',
      '--api-enable', '--api-port', '21550',
    ]);
  });

  test('worker and password are optional, and the API port is overridable', () => {
    const args = srbArgs({ endpoint: 'eu1.alphapool.tech:5566', address: ' prl1xyz ', apiPort: 21999 });
    expect(args).not.toContain('--worker');
    expect(args).not.toContain('--password');
    expect(args[args.indexOf('--wallet') + 1]).toBe('prl1xyz'); // trimmed
    expect(args[args.indexOf('--api-port') + 1]).toBe('21999');
  });

  test('always disables CPU mining', () => {
    // The app is about spare GPU capacity; quietly pinning every core is not
    // what anyone signed up for.
    expect(srbArgs({})).toContain('--disable-cpu');
    expect(srbArgs({}).indexOf('--wallet')).toBeGreaterThan(-1);
  });

  test('called with nothing at all, still produces a well-formed vector', () => {
    const args = srbArgs();
    expect(args.slice(0, 2)).toEqual(['--algorithm', 'pearlhash']);
    expect(args[args.indexOf('--pool') + 1]).toBe('');
  });

  test('api url follows the port', () => {
    expect(srbApiUrl()).toBe('http://127.0.0.1:21550/');
    expect(srbApiUrl(21999)).toBe('http://127.0.0.1:21999/');
  });
});

describe('srbStatusEvents', () => {
  // Field names could not be verified from the dev container (the binary is
  // packed and exits silently with no supported GPU), so the mapper is
  // deliberately tolerant of several shapes and returns nothing rather than
  // confident zeros when it recognises none.
  test('maps a device list onto alpha-miner-shaped status events', () => {
    const events = srbStatusEvents({
      gpu_devices: [
        {
          device_id: 0,
          device_name: 'NVIDIA GeForce RTX 5080',
          hashrate: 195_000_000_000_000,
          accepted_shares: 58,
          rejected_shares: 1,
          temperature: 67,
          power: 360,
        },
      ],
    });
    expect(events).toEqual([{
      type: 'status',
      gpuIndex: 0,
      hashrate: 195,
      accepted: 58,
      rejected: 1,
      power: 360,
      temp: 67,
      gpu: 'NVIDIA GeForce RTX 5080',
    }]);
  });

  test('accepts the alternative key spellings and a nested algorithms array', () => {
    const [e] = srbStatusEvents({
      algorithms: [{ gpu_devices: [{ id: 3, name: 'RTX 4090', speed: '1000000000000', shares_accepted: '7', temp: 55 }] }],
    });
    expect(e.gpuIndex).toBe(3);
    expect(e.gpu).toBe('RTX 4090');
    expect(e.hashrate).toBe(1);
    expect(e.accepted).toBe(7);
    expect(e.temp).toBe(55);
    expect(e.rejected).toBeNull();
  });

  test('indexes by position when the device carries no id', () => {
    const events = srbStatusEvents({ devices: [{ hashrate: 0 }, { hashrate: 0 }] });
    expect(events.map((e) => e.gpuIndex)).toEqual([0, 1]);
    expect(events[0].hashrate).toBe(0);
  });

  test('a custom divisor overrides the unit assumption', () => {
    // If a real run shows the rig reading 1e12x low, this is the one knob.
    const [e] = srbStatusEvents({ gpus: [{ hashrate: 195 }] }, 1);
    expect(e.hashrate).toBe(195);
  });

  test('returns nothing rather than fake zeros when it recognises nothing', () => {
    // A confident 0 H/s beside a rig that is really mining is precisely the
    // failure a user just spent a day chasing on the pool dashboard.
    expect(srbStatusEvents(null)).toEqual([]);
    expect(srbStatusEvents('nope')).toEqual([]);
    expect(srbStatusEvents({})).toEqual([]);
    expect(srbStatusEvents({ gpu_devices: [] })).toEqual([]);
    expect(srbStatusEvents({ gpu_devices: 'not-an-array' })).toEqual([]);
  });

  test('a null entry in the device list does not take the whole poll down', () => {
    const events = srbStatusEvents({ gpu_devices: [null, { hashrate: 0 }] });
    expect(events).toHaveLength(2);
    expect(events[0].hashrate).toBeNull();
    expect(events[0].gpuIndex).toBe(0);
  });

  test('an unreadable field is null, not zero', () => {
    const [e] = srbStatusEvents({ gpu_devices: [{ hashrate: 'abc', temperature: null }] });
    expect(e.hashrate).toBeNull();
    expect(e.temp).toBeNull();
    expect(e.gpu).toBeNull();
  });
});

describe('SRB constants', () => {
  test('the dev fee is carried explicitly so the UI can show it', () => {
    // alpha-miner is 0%; this is the cost of the alternative and the user has
    // to see it before choosing.
    expect(SRB.devFeePercent).toBe(2);
    expect(SRB.algorithm).toBe('pearlhash');
    expect(PLAIN_STRATUM_PORT).toBe(5571);
  });
});
