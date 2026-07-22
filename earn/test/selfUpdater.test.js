'use strict';

jest.mock('https');
jest.mock('fs');
jest.mock('child_process');
jest.mock('node:sea', () => ({ isSea: jest.fn() }));
jest.mock('../src/shared/selfUpdate', () => ({
  LATEST_RELEASE_API: 'https://api.github.com/repos/x/y/releases/latest',
  parseRelease: jest.fn(),
  planUpdate: jest.fn(),
}));

const https = require('https');
const fs = require('fs');
const { spawnSync } = require('child_process');
const sea = require('node:sea');
const { parseRelease } = require('../src/shared/selfUpdate');
const { EventEmitter } = require('events');
const updater = require('../src/cli/selfUpdater');

function fakeRes({ statusCode = 200, headers = {} } = {}) {
  const res = new EventEmitter();
  res.statusCode = statusCode;
  res.headers = headers;
  res.resume = () => {};
  res.pipe = (dest) => { res._dest = dest; return dest; };
  return res;
}
function fakeReq() {
  const req = new EventEmitter();
  req.destroy = jest.fn();
  return req;
}
function fakeWrite() {
  const w = new EventEmitter();
  w.close = (cb) => { if (cb) cb(); };
  return w;
}

// Point https.get at a queue of responses (one per call), returning the reqs.
function wire(responses) {
  let i = 0;
  const reqs = [];
  https.get.mockImplementation((_url, _opts, cb) => {
    const realCb = typeof _opts === 'function' ? _opts : cb;
    const req = fakeReq();
    reqs.push(req);
    const res = responses[Math.min(i, responses.length - 1)];
    i++;
    realCb(res);
    return req;
  });
  return reqs;
}

beforeEach(() => {
  jest.clearAllMocks();
});

describe('fetchLatestRelease', () => {
  it('parses the release when reachable', async () => {
    const res = fakeRes({ statusCode: 200 });
    wire([res]);
    parseRelease.mockReturnValue({ version: '9.9.9' });
    const p = updater.fetchLatestRelease();
    res.emit('data', '{"tag_name":"v9.9.9"}');
    res.emit('end');
    await expect(p).resolves.toEqual({ version: '9.9.9' });
    expect(parseRelease).toHaveBeenCalledWith({ tag_name: 'v9.9.9' });
  });

  it('returns null when unreachable', async () => {
    const res = fakeRes({ statusCode: 500 });
    wire([res]);
    await expect(updater.fetchLatestRelease()).resolves.toBeNull();
    expect(parseRelease).not.toHaveBeenCalled();
  });
});

describe('isPackaged', () => {
  afterEach(() => { delete process.pkg; });

  it('is true inside a Node SEA', () => {
    sea.isSea.mockReturnValue(true);
    expect(updater.isPackaged()).toBe(true);
  });

  it('is false when not a SEA', () => {
    sea.isSea.mockReturnValue(false);
    expect(updater.isPackaged()).toBe(false);
  });

  it('falls back to process.pkg when node:sea throws', () => {
    sea.isSea.mockImplementation(() => { throw new Error('no sea api'); });
    process.pkg = {};
    expect(updater.isPackaged()).toBe(true);
    delete process.pkg;
    expect(updater.isPackaged()).toBe(false);
  });
});

describe('download', () => {
  it('streams a 200 body to the destination', async () => {
    const res = fakeRes({ statusCode: 200 });
    wire([res]);
    const out = fakeWrite();
    fs.createWriteStream.mockReturnValue(out);
    const p = updater.download('https://host/bin', '/tmp/bin');
    res.emit('data', Buffer.from('x'));
    out.emit('finish');
    await expect(p).resolves.toBe('/tmp/bin');
    expect(fs.createWriteStream).toHaveBeenCalledWith('/tmp/bin');
  });

  it('follows a redirect', async () => {
    const redirect = fakeRes({ statusCode: 302, headers: { location: 'https://host/final' } });
    const ok = fakeRes({ statusCode: 200 });
    wire([redirect, ok]);
    const out = fakeWrite();
    fs.createWriteStream.mockReturnValue(out);
    const p = updater.download('https://host/bin', '/tmp/bin');
    out.emit('finish');
    await expect(p).resolves.toBe('/tmp/bin');
  });

  it('rejects after too many redirects', async () => {
    await expect(updater.download('https://host/bin', '/tmp/bin', 6)).rejects.toThrow('too many redirects');
  });

  it('rejects on a non-200 (missing status treated as 0)', async () => {
    const res = fakeRes({ statusCode: 0, headers: {} });
    wire([res]);
    await expect(updater.download('https://host/bin', '/tmp/bin')).rejects.toThrow('HTTP 0');
  });

  it('rejects on a write error', async () => {
    const res = fakeRes({ statusCode: 200 });
    wire([res]);
    const out = fakeWrite();
    fs.createWriteStream.mockReturnValue(out);
    const p = updater.download('https://host/bin', '/tmp/bin');
    out.emit('error', new Error('disk full'));
    await expect(p).rejects.toThrow('disk full');
  });

  it('rejects on a request error', async () => {
    const res = fakeRes({ statusCode: 200 });
    const reqs = wire([res]);
    fs.createWriteStream.mockReturnValue(fakeWrite());
    const p = updater.download('https://host/bin', '/tmp/bin');
    reqs[0].emit('error', new Error('ECONNRESET'));
    await expect(p).rejects.toThrow('ECONNRESET');
  });
});

describe('applyUpdate', () => {
  it('downloads to a temp file, chmods, and renames over the exe', async () => {
    const res = fakeRes({ statusCode: 200 });
    wire([res]);
    const out = fakeWrite();
    fs.createWriteStream.mockReturnValue(out);
    fs.chmodSync.mockImplementation(() => {});
    fs.renameSync.mockImplementation(() => {});

    const p = updater.applyUpdate({ downloadUrl: 'https://host/bin' }, '/opt/earn');
    out.emit('finish');
    await expect(p).resolves.toBe('/opt/earn');

    const tmp = fs.chmodSync.mock.calls[0][0];
    expect(tmp).toBe('/opt/earn.new-' + process.pid);
    expect(fs.chmodSync).toHaveBeenCalledWith(tmp, 0o755);
    expect(fs.renameSync).toHaveBeenCalledWith(tmp, '/opt/earn');
  });

  it('defaults the exe path to process.execPath', async () => {
    const res = fakeRes({ statusCode: 200 });
    wire([res]);
    fs.createWriteStream.mockReturnValue(fakeWrite());
    fs.chmodSync.mockImplementation(() => {});
    fs.renameSync.mockImplementation(() => {});

    const p = updater.applyUpdate({ downloadUrl: 'https://host/bin' });
    fs.createWriteStream.mock.results; // no-op; finish below
    // The write stream created above emits finish:
    const out = fs.createWriteStream.mock.results[0].value;
    out.emit('finish');
    await expect(p).resolves.toBe(process.execPath);
  });
});

describe('reexec', () => {
  it('returns the child exit status', () => {
    spawnSync.mockReturnValue({ status: 3 });
    expect(updater.reexec(['a', 'b'])).toBe(3);
    const [bin, argv, opts] = spawnSync.mock.calls[0];
    expect(bin).toBe(process.execPath);
    expect(argv).toEqual(['a', 'b']);
    expect(opts.env[updater.UPDATED_ENV]).toBe('1');
  });

  it('maps a null status to 1', () => {
    spawnSync.mockReturnValue({ status: null });
    expect(updater.reexec([])).toBe(1);
  });
});
