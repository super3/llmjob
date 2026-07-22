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
