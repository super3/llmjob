'use strict';

const {
  pingEndpoint, detectRegion, detectVram, detectGpusVram, detectDriverMajor,
  postMinerReport, findFreePort,
} = require('../src/main/probe');

// A fake net.Socket that fires whichever lifecycle event the test asks for.
function fakeSocket(event, latencyMs) {
  const handlers = {};
  return {
    handlers,
    setTimeout() {},
    once(name, fn) { handlers[name] = fn; return this; },
    destroy() { this.destroyed = true; },
    connect() {
      // Fire asynchronously so the caller has attached all listeners.
      process.nextTick(() => { if (handlers[event]) handlers[event](); });
    },
  };
}

// execFile stub: invokes the callback with the given err/stdout.
function fakeExecFile(err, stdout) {
  return (_bin, _args, _opts, cb) => cb(err, stdout);
}

describe('pingEndpoint', () => {
  it('resolves the latency on connect', async () => {
    let t = 100;
    const now = () => (t += 5); // start(105) then connect(110) => 5ms
    const net = { Socket: function () { return fakeSocket('connect'); } };
    const ms = await pingEndpoint('host:5566', 1000, { net, now });
    expect(ms).toBe(5);
  });

  it('resolves null on timeout', async () => {
    const net = { Socket: function () { return fakeSocket('timeout'); } };
    expect(await pingEndpoint('host:5566', 1000, { net })).toBeNull();
  });

  it('resolves null on error', async () => {
    const net = { Socket: function () { return fakeSocket('error'); } };
    expect(await pingEndpoint('host:5566', undefined, { net })).toBeNull();
  });
});

describe('detectRegion', () => {
  it('picks the lowest-latency reachable region', async () => {
    // Every endpoint "connects" instantly, so pickFastestRegion gets equal
    // latencies and returns a valid region key (not the bare default fallback).
    const net = { Socket: function () { return fakeSocket('connect'); } };
    const region = await detectRegion({ net, now: () => 0 });
    expect(typeof region).toBe('string');
    expect(region.length).toBeGreaterThan(0);
  });

  it('falls back to the default when nothing is reachable', async () => {
    const net = { Socket: function () { return fakeSocket('error'); } };
    const region = await detectRegion({ net });
    expect(region).toBe('us2'); // DEFAULTS.region
  });
});

describe('detectVram', () => {
  it('sums used/total across GPU lines', async () => {
    const execFile = fakeExecFile(null, '1024, 8192\n2048, 8192\n');
    expect(await detectVram({ execFile })).toEqual({ usedMb: 3072, totalMb: 16384 });
  });

  it('returns null on nvidia-smi error', async () => {
    expect(await detectVram({ execFile: fakeExecFile(new Error('no smi')) })).toBeNull();
  });

  it('returns null when no line parses', async () => {
    expect(await detectVram({ execFile: fakeExecFile(null, 'garbage\n') })).toBeNull();
  });
});

describe('detectGpusVram', () => {
  it('returns [] on error', async () => {
    expect(await detectGpusVram({ execFile: fakeExecFile(new Error('x')) })).toEqual([]);
  });

  it('parses per-GPU rows', async () => {
    const rows = await detectGpusVram({ execFile: fakeExecFile(null, '0, RTX 4090, 1024, 24576\n') });
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({ index: 0, usedMb: 1024, totalMb: 24576 });
  });
});

describe('detectDriverMajor', () => {
  it('returns the parsed major version', async () => {
    expect(await detectDriverMajor({ execFile: fakeExecFile(null, '580.42\n') })).toBe(580);
  });

  it('returns null on error', async () => {
    expect(await detectDriverMajor({ execFile: fakeExecFile(new Error('x')) })).toBeNull();
  });
});

describe('postMinerReport', () => {
  function fakeHttps(capture) {
    return {
      request(_url, _opts, cb) {
        const req = {
          on() { return this; },
          write(body) { capture.body = body; },
          end() { process.nextTick(() => cb({ resume() {}, on: (e, fn) => e === 'end' && fn() })); },
        };
        return req;
      },
    };
  }

  it('POSTs the payload and resolves on end', async () => {
    const capture = {};
    await postMinerReport({ hello: 'world' }, { https: fakeHttps(capture) });
    expect(JSON.parse(capture.body)).toEqual({ hello: 'world' });
  });

  it('never rejects when the request errors', async () => {
    const https = {
      request(_url, _opts) {
        return {
          on(evt, fn) { if (evt === 'error') process.nextTick(fn); return this; },
          write() {},
          end() {},
        };
      },
    };
    await expect(postMinerReport({ a: 1 }, { https })).resolves.toBeUndefined();
  });
});

describe('findFreePort', () => {
  function fakeServer(fail) {
    const handlers = {};
    return {
      once(name, fn) { handlers[name] = fn; return this; },
      listen() {
        process.nextTick(() => {
          if (fail) handlers.error && handlers.error();
          else handlers.listening && handlers.listening();
        });
      },
      close(cb) { if (cb) cb(); },
    };
  }

  it('returns the first port that binds', async () => {
    const net = { createServer: () => fakeServer(false) };
    expect(await findFreePort('127.0.0.1', 8080, 10, { net })).toBe(8080);
  });

  it('falls back to the start port when none bind', async () => {
    const net = { createServer: () => fakeServer(true) };
    expect(await findFreePort('127.0.0.1', 8080, 3, { net })).toBe(8080);
  });
});
