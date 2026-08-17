'use strict';

jest.mock('net');
jest.mock('http');
jest.mock('https');
jest.mock('child_process');

const net = require('net');
const https = require('https');
const { execFile } = require('child_process');
const { EventEmitter } = require('events');
const probe = require('../src/main/probe');

// A fake net.Socket that fires one lifecycle event after connect().
function fakeSocket(fire) {
  const h = {};
  return {
    setTimeout: jest.fn(),
    once(ev, cb) { h[ev] = cb; return this; },
    destroy: jest.fn(),
    connect: jest.fn(() => { process.nextTick(() => h[fire] && h[fire]()); }),
  };
}

// A fake net.Server for findFreePort: either binds ("listening") or fails.
function fakeServer(fail) {
  const h = {};
  return {
    once(ev, cb) { h[ev] = cb; return this; },
    close(cb) { if (cb) cb(); },
    listen() { process.nextTick(() => (fail ? h.error && h.error() : h.listening && h.listening())); },
  };
}

function fakeRes() {
  const res = new EventEmitter();
  res.resume = () => {};
  return res;
}
function fakeReq() {
  const req = new EventEmitter();
  req.write = jest.fn();
  req.end = jest.fn();
  req.destroy = jest.fn();
  return req;
}

// execFile stub.
function execCb(err, stdout) {
  execFile.mockImplementation((_bin, _args, _opts, cb) => cb(err, stdout));
}

beforeEach(() => {
  jest.clearAllMocks();
});

describe('pingEndpoint', () => {
  it('resolves a numeric latency on connect', async () => {
    net.Socket.mockImplementation(() => fakeSocket('connect'));
    const ms = await probe.pingEndpoint('host:5566', 1000);
    expect(typeof ms).toBe('number');
    expect(ms).toBeGreaterThanOrEqual(0);
  });

  it('resolves null on timeout (default timeout)', async () => {
    net.Socket.mockImplementation(() => fakeSocket('timeout'));
    expect(await probe.pingEndpoint('host:5566')).toBeNull();
  });

  it('resolves null on error', async () => {
    net.Socket.mockImplementation(() => fakeSocket('error'));
    expect(await probe.pingEndpoint('host:5566', 500)).toBeNull();
  });

  it('settles once even if multiple socket events fire', async () => {
    // connect then error: the second done() must be a no-op (settled guard).
    const h = {};
    const sock = {
      setTimeout: jest.fn(),
      once(ev, cb) { h[ev] = cb; return this; },
      destroy: jest.fn(),
      connect: jest.fn(() => process.nextTick(() => { h.connect(); h.error(); })),
    };
    net.Socket.mockImplementation(() => sock);
    const ms = await probe.pingEndpoint('host:5566', 100);
    expect(typeof ms).toBe('number');       // resolved from the first (connect)
    expect(sock.destroy).toHaveBeenCalledTimes(1);
  });
});

describe('detectRegion', () => {
  it('returns a region when endpoints respond', async () => {
    net.Socket.mockImplementation(() => fakeSocket('connect'));
    const region = await probe.detectRegion();
    expect(typeof region).toBe('string');
    expect(region.length).toBeGreaterThan(0);
  });

  it('falls back to the default when nothing is reachable', async () => {
    net.Socket.mockImplementation(() => fakeSocket('error'));
    expect(await probe.detectRegion()).toBe('us2'); // DEFAULTS.region
  });
});

describe('detectVram', () => {
  it('sums used/total across GPU lines', async () => {
    execCb(null, '1024, 8192\n2048, 8192\n');
    expect(await probe.detectVram()).toEqual({ usedMb: 3072, totalMb: 16384 });
  });

  it('returns null on error', async () => {
    execCb(new Error('no smi'));
    expect(await probe.detectVram()).toBeNull();
  });

  it('returns null when nothing parses', async () => {
    execCb(null, 'garbage\n');
    expect(await probe.detectVram()).toBeNull();
  });
});

describe('detectGpusVram', () => {
  it('returns [] on error', async () => {
    execCb(new Error('x'));
    expect(await probe.detectGpusVram()).toEqual([]);
  });

  it('parses per-GPU rows', async () => {
    execCb(null, '0, RTX 4090, 1024, 24576\n');
    const rows = await probe.detectGpusVram();
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({ index: 0, usedMb: 1024, totalMb: 24576 });
  });
});

describe('detectDriverMajor', () => {
  it('returns the parsed major version', async () => {
    execCb(null, '580.42\n');
    expect(await probe.detectDriverMajor()).toBe(580);
  });

  it('returns null on error', async () => {
    execCb(new Error('x'));
    expect(await probe.detectDriverMajor()).toBeNull();
  });
});

describe('postMinerReport', () => {
  it('POSTs over https (the configured report url) and resolves on end', async () => {
    const req = fakeReq();
    const res = fakeRes();
    https.request.mockImplementation((_u, _opts, cb) => { cb(res); return req; });
    const done = probe.postMinerReport({ hello: 'world' });
    res.emit('end');
    await expect(done).resolves.toBeUndefined();
    expect(req.write).toHaveBeenCalledWith(JSON.stringify({ hello: 'world' }));
  });

  it('resolves (never rejects) on a request error', async () => {
    const req = fakeReq();
    https.request.mockImplementation((_u, _opts) => req);
    const done = probe.postMinerReport({ a: 1 });
    req.emit('error', new Error('offline'));
    await expect(done).resolves.toBeUndefined();
  });

  it('resolves on a request timeout', async () => {
    const req = fakeReq();
    https.request.mockImplementation((_u, _opts) => req);
    const done = probe.postMinerReport({ a: 1 });
    req.emit('timeout');
    await expect(done).resolves.toBeUndefined();
    expect(req.destroy).toHaveBeenCalled();
  });

  it('swallows a synchronous failure (e.g. an unserializable payload)', async () => {
    const circular = {};
    circular.self = circular; // JSON.stringify throws
    await expect(probe.postMinerReport(circular)).resolves.toBeUndefined();
  });

  it('uses http when the report url is http', async () => {
    await jest.isolateModulesAsync(async () => {
      jest.doMock('../src/shared/config', () => ({
        NETWORK: { reportUrl: 'http://board.local/report' },
        REGIONS: {},
        DEFAULTS: { region: 'us2' },
      }));
      const p = require('../src/main/probe');
      const httpMock = require('http');
      const req = fakeReq();
      const res = fakeRes();
      httpMock.request.mockImplementation((_u, _opts, cb) => { cb(res); return req; });
      const done = p.postMinerReport({ ok: 1 });
      res.emit('end');
      await expect(done).resolves.toBeUndefined();
      expect(httpMock.request).toHaveBeenCalled();
    });
  });
});

describe('findFreePort', () => {
  it('returns the first port that binds (default tries)', async () => {
    net.createServer.mockImplementation(() => fakeServer(false));
    expect(await probe.findFreePort('127.0.0.1', 8080)).toBe(8080);
  });

  it('walks forward and falls back to the start port when none bind', async () => {
    net.createServer.mockImplementation(() => fakeServer(true));
    expect(await probe.findFreePort('127.0.0.1', 8080, 3)).toBe(8080);
  });
});

describe('detectGpuInfo', () => {
  // Both shells used to carry their own copy of this and had drifted into
  // different detection METHODS — the GUI asked Windows' WMI and returned null
  // everywhere else, so the shipped Linux AppImage had no device name and never
  // applied the per-card difficulty. One implementation, nvidia-smi first.
  const withPlatform = (value, fn) => {
    const original = Object.getOwnPropertyDescriptor(process, 'platform');
    Object.defineProperty(process, 'platform', { value, configurable: true });
    return Promise.resolve(fn()).finally(() => Object.defineProperty(process, 'platform', original));
  };

  it('reads the card name and count from nvidia-smi', async () => {
    execCb(null, 'NVIDIA GeForce RTX 3070\nNVIDIA GeForce RTX 3070\n');
    expect(await probe.detectGpuInfo()).toEqual({ name: 'NVIDIA GeForce RTX 3070', count: 2 });
    expect(execFile).toHaveBeenCalledWith('nvidia-smi', expect.any(Array), { timeout: 5000 }, expect.any(Function));
  });

  // The regression that mattered: on Linux this used to be an unconditional null.
  it('works on Linux, where the old GUI copy always returned null', async () => {
    await withPlatform('linux', async () => {
      execCb(null, 'NVIDIA GeForce RTX 4090\n');
      expect(await probe.detectGpuInfo()).toEqual({ name: 'NVIDIA GeForce RTX 4090', count: 1 });
    });
  });

  it('gives up on a non-Windows box with no nvidia-smi', async () => {
    await withPlatform('linux', async () => {
      execCb(new Error('ENOENT'));
      expect(await probe.detectGpuInfo()).toBeNull();
    });
  });

  it('falls back to WMI on Windows so a non-NVIDIA card still gets a name', async () => {
    await withPlatform('win32', async () => {
      execFile.mockImplementation((bin, _args, _opts, cb) => (bin === 'nvidia-smi'
        ? cb(new Error('not installed'))
        : cb(null, 'Intel(R) UHD Graphics\r\nAMD Radeon RX 7900 XTX\r\n')));
      expect(await probe.detectGpuInfo()).toEqual({ name: 'AMD Radeon RX 7900 XTX', count: 1 });
      expect(execFile).toHaveBeenCalledWith('powershell.exe', expect.any(Array), { timeout: 5000 }, expect.any(Function));
    });
  });

  it('returns null when the Windows fallback also fails', async () => {
    await withPlatform('win32', async () => {
      execFile.mockImplementation((_bin, _args, _opts, cb) => cb(new Error('nope')));
      expect(await probe.detectGpuInfo()).toBeNull();
    });
  });

  // macOS has neither nvidia-smi nor WMI, so before this it fell straight to
  // null and the device label read "GPU · auto-detect" — on the one platform
  // where that GPU is the only thing the app uses.
  const MAC_JSON = JSON.stringify({
    SPDisplaysDataType: [{ _name: 'Apple M3 Max', sppci_model: 'Apple M3 Max', sppci_cores: '40' }],
  });

  it('names the GPU on macOS via system_profiler', async () => {
    await withPlatform('darwin', async () => {
      execFile.mockImplementation((bin, _args, _opts, cb) => (bin === 'nvidia-smi'
        ? cb(new Error('ENOENT'))
        : cb(null, MAC_JSON)));
      expect(await probe.detectGpuInfo()).toEqual({ name: 'Apple M3 Max', count: 1 });
      expect(execFile).toHaveBeenCalledWith('system_profiler',
        ['SPDisplaysDataType', '-json'], { timeout: 10000 }, expect.any(Function));
    });
  });

  // If Apple reshapes that JSON, the CPU brand string still names the same SoC
  // on Apple silicon — the same answer by another route, not a guess.
  it('falls back to the CPU brand string when system_profiler cannot be parsed', async () => {
    await withPlatform('darwin', async () => {
      execFile.mockImplementation((bin, _args, _opts, cb) => {
        if (bin === 'nvidia-smi') return cb(new Error('ENOENT'));
        if (bin === 'system_profiler') return cb(null, 'not json at all');
        return cb(null, 'Apple M4 Pro\n');
      });
      expect(await probe.detectGpuInfo()).toEqual({ name: 'Apple M4 Pro', count: 1 });
      expect(execFile).toHaveBeenCalledWith('sysctl',
        ['-n', 'machdep.cpu.brand_string'], { timeout: 5000 }, expect.any(Function));
    });
  });

  it('falls back the same way when system_profiler itself fails', async () => {
    await withPlatform('darwin', async () => {
      execFile.mockImplementation((bin, _args, _opts, cb) => (bin === 'sysctl'
        ? cb(null, 'Apple M1\n')
        : cb(new Error('nope'))));
      expect(await probe.detectGpuInfo()).toEqual({ name: 'Apple M1', count: 1 });
    });
  });

  it('returns null on macOS when both probes fail', async () => {
    await withPlatform('darwin', async () => {
      execFile.mockImplementation((_bin, _args, _opts, cb) => cb(new Error('nope')));
      expect(await probe.detectGpuInfo()).toBeNull();
    });
  });

  it('returns null on macOS when sysctl answers with nothing usable', async () => {
    await withPlatform('darwin', async () => {
      execFile.mockImplementation((bin, _args, _opts, cb) => (bin === 'sysctl'
        ? cb(null, '   \n')
        : cb(new Error('nope'))));
      expect(await probe.detectGpuInfo()).toBeNull();
    });
  });

  it('returns null when WMI answers but names nothing usable', async () => {
    await withPlatform('win32', async () => {
      execFile.mockImplementation((bin, _args, _opts, cb) => (bin === 'nvidia-smi'
        ? cb(new Error('not installed'))
        : cb(null, '\r\n\r\n')));
      expect(await probe.detectGpuInfo()).toBeNull();
    });
  });

  // nvidia-smi can exit 0 yet print nothing useful; that must fall through to the
  // platform fallback rather than reporting a nameless GPU.
  it('falls through when nvidia-smi succeeds but names nothing', async () => {
    await withPlatform('linux', async () => {
      execCb(null, '\n\n');
      expect(await probe.detectGpuInfo()).toBeNull();
    });
  });
});
