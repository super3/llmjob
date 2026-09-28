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
  req.setTimeout = jest.fn(); // io.downloadFile arms a stall timeout on the request
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

describe('applyUpdate', () => {
  // applyUpdate now delegates the transfer to io.downloadFile, which streams to a
  // unique <tmp>.<pid>.<seq>.part scratch path and renames it onto <tmp>, then
  // applyUpdate chmods <tmp> and renames it over the exe.
  it('downloads to a temp file, chmods, and renames over the exe', async () => {
    const res = fakeRes({ statusCode: 200 });
    wire([res]);
    const out = fakeWrite();
    fs.createWriteStream.mockReturnValue(out);
    fs.renameSync.mockImplementation(() => {});
    fs.chmodSync.mockImplementation(() => {});

    const p = updater.applyUpdate({ downloadUrl: 'https://host/bin' }, '/opt/earn');
    res.emit('data', Buffer.from('x'));
    out.emit('finish'); // io.downloadFile: finish → close → rename(part → tmp) → resolve(tmp)
    await expect(p).resolves.toBe('/opt/earn');

    const tmp = '/opt/earn.new-' + process.pid;
    // The scratch file was written under a unique .part name derived from tmp…
    const part = fs.createWriteStream.mock.calls[0][0];
    expect(part).toMatch(new RegExp('^' + tmp.replace(/[.]/g, '\\.') + '\\.\\d+\\.\\d+\\.part$'));
    // …renamed onto tmp by io.downloadFile, then tmp chmod'd and renamed over the exe.
    expect(fs.renameSync).toHaveBeenCalledWith(part, tmp);
    expect(fs.chmodSync).toHaveBeenCalledWith(tmp, 0o755);
    expect(fs.renameSync).toHaveBeenCalledWith(tmp, '/opt/earn');
  });

  it('downloads the paired pearl_core.node beside the exe when the plan has one', async () => {
    // Two sequential downloads: the binary, then the core. Each gets its own
    // response + write stream in order.
    const res1 = fakeRes({ statusCode: 200 });
    const res2 = fakeRes({ statusCode: 200 });
    wire([res1, res2]);
    const out1 = fakeWrite();
    const out2 = fakeWrite();
    fs.createWriteStream.mockReturnValueOnce(out1).mockReturnValueOnce(out2);
    fs.renameSync.mockImplementation(() => {});
    fs.chmodSync.mockImplementation(() => {});

    const p = updater.applyUpdate(
      { downloadUrl: 'https://host/bin', coreUrl: 'https://host/pearl_core.node' },
      '/opt/rig/earn');
    res1.emit('data', Buffer.from('x'));
    out1.emit('finish');
    // let the first rename settle before feeding the second transfer
    await new Promise((r) => setImmediate(r));
    res2.emit('data', Buffer.from('y'));
    out2.emit('finish');
    await expect(p).resolves.toBe('/opt/rig/earn');

    // The core landed beside the exe via its own temp-then-rename, so a failed
    // download can never leave a half-written addon where the loader probes.
    // path.join, not a literal: the core path is built with the platform's
    // separators, so the expectation must be too or it only passes on Linux.
    const core = require('path').join('/opt/rig', 'pearl_core.node');
    expect(fs.renameSync).toHaveBeenCalledWith(core + '.new-' + process.pid, core);
  });

  it('downloads the CUDA 13 core beside the exe when the plan has one', async () => {
    const res1 = fakeRes({ statusCode: 200 });
    const res2 = fakeRes({ statusCode: 200 });
    wire([res1, res2]);
    const out1 = fakeWrite();
    const out2 = fakeWrite();
    fs.createWriteStream.mockReturnValueOnce(out1).mockReturnValueOnce(out2);
    fs.renameSync.mockImplementation(() => {});
    fs.chmodSync.mockImplementation(() => {});

    const p = updater.applyUpdate(
      { downloadUrl: 'https://host/bin', coreCu13Url: 'https://host/pearl_core_cu13.node' },
      '/opt/rig/earn');
    res1.emit('data', Buffer.from('x'));
    out1.emit('finish');
    await new Promise((r) => setImmediate(r));
    res2.emit('data', Buffer.from('y'));
    out2.emit('finish');
    await expect(p).resolves.toBe('/opt/rig/earn');

    const cu13 = require('path').join('/opt/rig', 'pearl_core_cu13.node');
    expect(fs.renameSync).toHaveBeenCalledWith(cu13 + '.new-' + process.pid, cu13);
    // The old copy is removed BEFORE the new one is fetched, so no failure on
    // the way can leave last release's core where the loader prefers it.
    expect(fs.rmSync).toHaveBeenCalledWith(cu13, { force: true });
    const renamed = fs.renameSync.mock.calls.findIndex((c) => c[1] === cu13);
    expect(fs.rmSync.mock.invocationCallOrder[0])
      .toBeLessThan(fs.renameSync.mock.invocationCallOrder[renamed]);
  });

  // The CUDA 13 core is optional. By the time it is fetched the binary and the
  // 12.8 core are already replaced; failing the update there would have the
  // caller "continue on" the old version beside the new 12.8 core.
  it('does not fail the update when the CUDA 13 core cannot be installed', async () => {
    const res1 = fakeRes({ statusCode: 200 });
    const res2 = fakeRes({ statusCode: 200 });
    wire([res1, res2]);
    const out1 = fakeWrite();
    const out2 = fakeWrite();
    fs.createWriteStream.mockReturnValueOnce(out1).mockReturnValueOnce(out2);
    const cu13 = require('path').join('/opt/rig', 'pearl_core_cu13.node');
    fs.renameSync.mockImplementation((from, to) => {
      if (to === cu13) throw new Error('EACCES: permission denied');
    });
    fs.chmodSync.mockImplementation(() => {});

    const p = updater.applyUpdate(
      { downloadUrl: 'https://host/bin', coreCu13Url: 'https://host/pearl_core_cu13.node' },
      '/opt/rig/earn');
    res1.emit('data', Buffer.from('x'));
    out1.emit('finish');
    await new Promise((r) => setImmediate(r));
    res2.emit('data', Buffer.from('y'));
    out2.emit('finish');
    await expect(p).resolves.toBe('/opt/rig/earn');
    // ...and last release's copy is gone all the same: the rig mines on the
    // 12.8 core rather than an old CUDA 13 one.
    expect(fs.rmSync).toHaveBeenCalledWith(cu13, { force: true });
  });

  // On a Blackwell rig the loader prefers the CUDA 13 file, so last release's
  // copy beside this release's binary would be the skew the pairing prevents.
  it('deletes an old CUDA 13 core when the new release has none', async () => {
    const res = fakeRes({ statusCode: 200 });
    wire([res]);
    fs.createWriteStream.mockReturnValue(fakeWrite());
    fs.renameSync.mockImplementation(() => {});
    fs.chmodSync.mockImplementation(() => {});

    const p = updater.applyUpdate({ downloadUrl: 'https://host/bin', coreCu13Url: null }, '/opt/rig/earn');
    fs.createWriteStream.mock.results[0].value.emit('finish');
    await expect(p).resolves.toBe('/opt/rig/earn');
    expect(fs.rmSync).toHaveBeenCalledWith(require('path').join('/opt/rig', 'pearl_core_cu13.node'), { force: true });
  });

  it('defaults the exe path to process.execPath', async () => {
    const res = fakeRes({ statusCode: 200 });
    wire([res]);
    fs.createWriteStream.mockReturnValue(fakeWrite());
    fs.renameSync.mockImplementation(() => {});
    fs.chmodSync.mockImplementation(() => {});

    const p = updater.applyUpdate({ downloadUrl: 'https://host/bin' });
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
