'use strict';

jest.mock('http');
jest.mock('https');
jest.mock('fs');
jest.mock('child_process');
// Hand-written: the automock would drop rootCertificates, which the recovery
// pass concatenates onto.
jest.mock('tls', () => ({ rootCertificates: ['ROOT'], connect: jest.fn() }));

const http = require('http');
const https = require('https');
const fs = require('fs');
const tls = require('tls');
const { execFile } = require('child_process');
const { EventEmitter } = require('events');
const io = require('../src/main/io');

// A fake IncomingMessage.
function fakeRes({ statusCode = 200, headers = {} } = {}) {
  const res = new EventEmitter();
  res.statusCode = statusCode;
  res.headers = headers;
  res.setEncoding = () => {};
  res.resume = () => {};
  res.destroy = jest.fn();
  res.pipe = (dest) => { res._dest = dest; return dest; };
  return res;
}

// A fake ClientRequest. destroy(err) mirrors Node: it surfaces the error.
function fakeReq() {
  const req = new EventEmitter();
  req.write = jest.fn();
  req.end = jest.fn();
  req.setTimeout = jest.fn((ms, cb) => { req._timeoutCb = cb; });
  req.destroy = jest.fn((err) => { if (err) req.emit('error', err); });
  return req;
}

// Point a lib's request/get at a queue of responses (one per call), returning
// the created requests so the test can drive them.
function wire(lib, responses) {
  let i = 0;
  const reqs = [];
  const impl = (_url, a, b) => {
    const cb = typeof a === 'function' ? a : b;
    const req = fakeReq();
    reqs.push(req);
    // Advance the response cursor BEFORE invoking cb: a redirect recurses
    // synchronously inside cb, so the next response must already be selected.
    const res = responses[Math.min(i, responses.length - 1)];
    i++;
    cb(res);
    return req;
  };
  lib.request = jest.fn(impl);
  lib.get = jest.fn(impl);
  return reqs;
}

beforeEach(() => {
  jest.clearAllMocks();
});

describe('postJson', () => {
  it('resolves parsed JSON for a 2xx response', async () => {
    const res = fakeRes({ statusCode: 200 });
    const reqs = wire(https, [res]);
    const p = io.postJson('https://host/api', { a: 1 });
    res.emit('data', '{"ok":true}');
    res.emit('end');
    await expect(p).resolves.toEqual({ status: 200, data: { ok: true }, raw: '{"ok":true}' });
    expect(reqs[0].write).toHaveBeenCalledWith(JSON.stringify({ a: 1 }));
  });

  it('resolves any status; non-JSON body leaves data null', async () => {
    const res = fakeRes({ statusCode: 500 });
    wire(http, [res]);
    const p = io.postJson('http://host/api', {}, 1000);
    res.emit('data', 'oops');
    res.emit('end');
    await expect(p).resolves.toEqual({ status: 500, data: null, raw: 'oops' });
  });

  it('an empty body resolves data null', async () => {
    const res = fakeRes({ statusCode: 204 });
    wire(https, [res]);
    const p = io.postJson('https://host/api', {});
    res.emit('end');
    await expect(p).resolves.toEqual({ status: 204, data: null, raw: '' });
  });

  it('reports status 0 when the response has no status code', async () => {
    const res = fakeRes({ statusCode: 0 });
    wire(https, [res]);
    const p = io.postJson('https://host/api', {});
    res.emit('end');
    await expect(p).resolves.toEqual({ status: 0, data: null, raw: '' });
  });

  it('rejects on an invalid URL', async () => {
    await expect(io.postJson('::nope::', {})).rejects.toBeDefined();
  });

  it('rejects on a transport error', async () => {
    const res = fakeRes();
    const reqs = wire(https, [res]);
    const p = io.postJson('https://host/api', {});
    reqs[0].emit('error', new Error('boom'));
    await expect(p).rejects.toThrow('boom');
  });

  it('rejects when the request times out', async () => {
    const res = fakeRes();
    const reqs = wire(https, [res]);
    const p = io.postJson('https://host/api', {});
    reqs[0].emit('timeout');
    await expect(p).rejects.toThrow('request timed out');
  });
});

describe('getJson', () => {
  it('resolves parsed JSON on 200 (https, with headers)', async () => {
    const res = fakeRes({ statusCode: 200 });
    wire(https, [res]);
    const p = io.getJson('https://host/api', { headers: { 'User-Agent': 'x' }, timeout: 1000 });
    res.emit('data', '{"ok":true}');
    res.emit('end');
    await expect(p).resolves.toEqual({ ok: true });
    expect(https.get).toHaveBeenCalledWith(
      expect.any(URL),
      expect.objectContaining({ timeout: 1000, headers: { 'User-Agent': 'x' } }),
      expect.any(Function),
    );
  });

  it('works over http with the default timeout', async () => {
    const res = fakeRes({ statusCode: 200 });
    wire(http, [res]);
    const p = io.getJson('http://host/api');
    res.emit('data', '{"n":1}');
    res.emit('end');
    await expect(p).resolves.toEqual({ n: 1 });
    expect(http.get).toHaveBeenCalledWith(
      expect.any(URL), expect.objectContaining({ timeout: 8000 }), expect.any(Function));
  });

  it('resolves null on a non-200', async () => {
    const res = fakeRes({ statusCode: 500 });
    wire(https, [res]);
    await expect(io.getJson('https://host/api')).resolves.toBeNull();
  });

  it('resolves null on unparseable JSON', async () => {
    const res = fakeRes({ statusCode: 200 });
    wire(https, [res]);
    const p = io.getJson('https://host/api');
    res.emit('data', 'not json');
    res.emit('end');
    await expect(p).resolves.toBeNull();
  });

  it('aborts an oversized body and resolves null', async () => {
    const res = fakeRes({ statusCode: 200 });
    const reqs = wire(https, [res]);
    const p = io.getJson('https://host/api');
    res.emit('data', 'x'.repeat(4_000_001));
    res.emit('end');
    await expect(p).resolves.toBeNull();
    expect(reqs[0].destroy).toHaveBeenCalled();
  });

  it('resolves null on an invalid URL', async () => {
    await expect(io.getJson('::bad::')).resolves.toBeNull();
  });

  it('resolves null on a request error', async () => {
    const res = fakeRes({ statusCode: 200 });
    const reqs = wire(https, [res]);
    const p = io.getJson('https://host/api');
    reqs[0].emit('error', new Error('offline'));
    await expect(p).resolves.toBeNull();
  });

  it('resolves null on a request timeout', async () => {
    const res = fakeRes({ statusCode: 200 });
    const reqs = wire(https, [res]);
    const p = io.getJson('https://host/api');
    reqs[0].emit('timeout');
    await expect(p).resolves.toBeNull();
    expect(reqs[0].destroy).toHaveBeenCalled();
  });
});

describe('downloadFile', () => {
  function fakeWrite() {
    const w = new EventEmitter();
    w.close = (cb) => { if (cb) cb(); };
    w.destroy = jest.fn();
    return w;
  }

  it('downloads a 200 body, reports progress and renames into place', async () => {
    const res = fakeRes({ statusCode: 200, headers: { 'content-length': '4' } });
    wire(https, [res]);
    const out = fakeWrite();
    fs.createWriteStream.mockReturnValue(out);
    fs.renameSync.mockImplementation(() => {});
    const progress = [];

    const p = io.downloadFile('https://host/f.bin', '/tmp/f.bin', (pct) => progress.push(pct));
    res.emit('data', Buffer.from('ab'));
    res.emit('data', Buffer.from('cd'));
    out.emit('finish');

    await expect(p).resolves.toBe('/tmp/f.bin');
    // The scratch path is unique per call (see the .part race test below), so
    // assert its shape rather than a fixed name — and that the rename lands on
    // the same file that was written.
    const part = fs.createWriteStream.mock.calls.pop()[0];
    expect(part).toMatch(/^\/tmp\/f\.bin\.\d+\.\d+\.part$/);
    expect(fs.renameSync).toHaveBeenCalledWith(part, '/tmp/f.bin');
    expect(progress[progress.length - 1]).toBe(100);
  });

  // Two downloads of the same dest used to share one `<dest>.part`: whichever
  // finished first renamed it away and the rest died on `ENOENT … rename`. That
  // is exactly what the Windows client hit when a user clicked START LLM through
  // a failing setup. Concurrent calls must not collide on the scratch path.
  it('gives every in-flight download its own .part scratch file', async () => {
    const [r1, r2] = [fakeRes({ statusCode: 200, headers: {} }), fakeRes({ statusCode: 200, headers: {} })];
    wire(https, [r1, r2]);
    const [o1, o2] = [fakeWrite(), fakeWrite()];
    fs.createWriteStream.mockReturnValueOnce(o1).mockReturnValueOnce(o2);
    fs.renameSync.mockImplementation(() => {});

    const p1 = io.downloadFile('https://host/a', '/tmp/llama-download.archive');
    const p2 = io.downloadFile('https://host/a', '/tmp/llama-download.archive');
    o1.emit('finish');
    o2.emit('finish');
    await Promise.all([p1, p2]);

    const [first, second] = fs.createWriteStream.mock.calls.map((c) => c[0]);
    expect(first).not.toBe(second);
    // Both still land at the real destination — last writer wins, nobody errors.
    expect(fs.renameSync).toHaveBeenCalledWith(first, '/tmp/llama-download.archive');
    expect(fs.renameSync).toHaveBeenCalledWith(second, '/tmp/llama-download.archive');
  });

  it('works over http and without an onProgress callback', async () => {
    const res = fakeRes({ statusCode: 200, headers: {} });
    wire(http, [res]);
    const out = fakeWrite();
    fs.createWriteStream.mockReturnValue(out);
    fs.renameSync.mockImplementation(() => {});

    const p = io.downloadFile('http://host/f.bin', '/tmp/f.bin');
    res.emit('data', Buffer.from('x'));
    out.emit('finish');
    await expect(p).resolves.toBe('/tmp/f.bin');
  });

  it('follows a redirect', async () => {
    const redirect = fakeRes({ statusCode: 302, headers: { location: 'https://host/final.bin' } });
    const ok = fakeRes({ statusCode: 200, headers: {} });
    wire(https, [redirect, ok]);
    const out = fakeWrite();
    fs.createWriteStream.mockReturnValue(out);
    fs.renameSync.mockImplementation(() => {});

    const p = io.downloadFile('https://host/f.bin', '/tmp/f.bin');
    ok.emit('data', Buffer.from('x'));
    out.emit('finish');
    await expect(p).resolves.toBe('/tmp/f.bin');
  });

  it('rejects after too many redirects', async () => {
    await expect(io.downloadFile('https://host/f', '/tmp/f', null, 6)).rejects.toThrow('too many redirects');
  });

  it('rejects on a non-200, non-redirect status', async () => {
    const res = fakeRes({ statusCode: 404, headers: {} });
    wire(https, [res]);
    await expect(io.downloadFile('https://host/f', '/tmp/f')).rejects.toThrow('HTTP 404');
  });

  it('treats a missing status code as HTTP 0', async () => {
    const res = fakeRes({ statusCode: 0, headers: {} });
    wire(https, [res]);
    await expect(io.downloadFile('https://host/f', '/tmp/f')).rejects.toThrow('HTTP 0');
  });

  // A dropped connection is retried, resuming from what is on disk. These tests
  // drive every attempt to exhaustion; `advance` walks the backoff (2s, 4s, 6s)
  // and each attempt's response is served synchronously by `wire`.
  const ATTEMPTS = 4;
  const advance = async (attempt) => {
    await Promise.resolve();
    jest.advanceTimersByTime(2000 * attempt);
    await Promise.resolve();
  };

  it('retries a dropped response and gives up only after the last attempt', async () => {
    jest.useFakeTimers();
    const resList = Array.from({ length: ATTEMPTS }, () => fakeRes({ statusCode: 200, headers: {} }));
    wire(https, resList);
    const outs = resList.map(() => fakeWrite());
    let i = 0;
    fs.createWriteStream.mockImplementation(() => outs[i++]);
    fs.statSync.mockReturnValue({ size: 0 });
    fs.unlink.mockImplementation((p, cb) => cb && cb());

    const p = io.downloadFile('https://host/f', '/tmp/f');
    const settled = p.then(() => null, (e) => e);
    for (let a = 1; a <= ATTEMPTS; a++) {
      resList[a - 1].emit('error', new Error('aborted'));
      await advance(a);
    }

    expect((await settled).message).toMatch('aborted');
    expect(https.get).toHaveBeenCalledTimes(ATTEMPTS);
    expect(outs[0].destroy).toHaveBeenCalled();
    // The scratch file survives the retries and is only removed at the end.
    expect(fs.unlink).toHaveBeenCalledTimes(1);
    jest.useRealTimers();
  });

  it('retries once when a single stall errors both the response and the request', async () => {
    // req.destroy(err) from the stall timeout surfaces 'error' on both the request
    // and the response; that must not fork two retry chains onto the same .part.
    jest.useFakeTimers();
    const r1 = fakeRes({ statusCode: 200, headers: {} });
    const r2 = fakeRes({ statusCode: 200, headers: {} });
    const reqs = wire(https, [r1, r2]);
    const [o1, o2] = [fakeWrite(), fakeWrite()];
    fs.createWriteStream.mockReturnValueOnce(o1).mockReturnValueOnce(o2);
    fs.renameSync.mockImplementation(() => {});

    const p = io.downloadFile('https://host/f', '/tmp/f');
    // The stall fires both error events for the SAME attempt.
    r1.emit('error', new Error('stalled'));    // response path → out.destroy + retryOrFail
    reqs[0].emit('error', new Error('stalled')); // request path → retryOrFail again (must be ignored)
    await advance(1);

    // Exactly one retry: a second GET, not two.
    expect(https.get).toHaveBeenCalledTimes(2);
    r2.emit('data', Buffer.from('x'));
    o2.emit('finish');
    await expect(p).resolves.toBe('/tmp/f');
    jest.useRealTimers();
  });

  it('retries a write-stream error too', async () => {
    jest.useFakeTimers();
    const resList = Array.from({ length: ATTEMPTS }, () => fakeRes({ statusCode: 200, headers: {} }));
    wire(https, resList);
    const outs = resList.map(() => fakeWrite());
    let i = 0;
    fs.createWriteStream.mockImplementation(() => outs[i++]);
    fs.statSync.mockReturnValue({ size: 0 });
    fs.unlink.mockImplementation((p, cb) => cb && cb());

    const p = io.downloadFile('https://host/f', '/tmp/f');
    const settled = p.then(() => null, (e) => e);
    for (let a = 1; a <= ATTEMPTS; a++) {
      outs[a - 1].emit('error', new Error('disk full'));
      await advance(a);
    }

    expect((await settled).message).toMatch('disk full');
    jest.useRealTimers();
  });

  // A retry succeeds and, crucially, starts the scratch file CLEAN. Resuming
  // from a partial was tried and removed: the model is only ever validated with
  // existsSync, so a resume that got the offset wrong would leave a corrupt
  // multi-GB GGUF at the final path that every later start accepts.
  it('retries from a clean scratch file rather than resuming a partial', async () => {
    jest.useFakeTimers();
    const realSetTimeout = global.setTimeout;
    global.setTimeout = (fn, ms) => { realSetTimeout(fn, ms); return {}; }; // handle without unref()
    const r1 = fakeRes({ statusCode: 200, headers: { 'content-length': '1000' } });
    const r2 = fakeRes({ statusCode: 200, headers: { 'content-length': '1000' } });
    wire(https, [r1, r2]);
    const [o1, o2] = [fakeWrite(), fakeWrite()];
    fs.createWriteStream.mockReturnValueOnce(o1).mockReturnValueOnce(o2);
    fs.renameSync.mockImplementation(() => {});
    const progress = [];

    const p = io.downloadFile('https://host/f', '/tmp/f', (pct) => progress.push(pct));
    r1.emit('data', Buffer.alloc(600)); // 600 of 1000 before the drop…
    r1.emit('error', new Error('aborted'));
    await advance(1);

    // …and the second attempt asks for the whole file again, into a truncating
    // open of the same scratch path.
    expect(typeof https.get.mock.calls[1][1]).toBe('function'); // callback, not an options object → no Range header
    expect(fs.createWriteStream.mock.calls[1][1]).toBeUndefined(); // truncating, not { flags: 'a' }
    expect(fs.createWriteStream.mock.calls[1][0]).toBe(fs.createWriteStream.mock.calls[0][0]);
    expect(fs.statSync).not.toHaveBeenCalled(); // nothing measures the partial any more

    r2.emit('data', Buffer.alloc(1000));
    o2.emit('finish');
    await expect(p).resolves.toBe('/tmp/f');
    expect(progress[progress.length - 1]).toBe(100);
    global.setTimeout = realSetTimeout;
    jest.useRealTimers();
  });

  it('rejects when the final rename fails', async () => {
    const res = fakeRes({ statusCode: 200, headers: {} });
    wire(https, [res]);
    const out = fakeWrite();
    fs.createWriteStream.mockReturnValue(out);
    fs.renameSync.mockImplementation(() => { throw new Error('rename EXDEV'); });

    const p = io.downloadFile('https://host/f', '/tmp/f');
    out.emit('finish');
    await expect(p).rejects.toThrow('rename EXDEV');
  });

  it('retries a stalled socket and surfaces it once attempts run out', async () => {
    jest.useFakeTimers();
    const resList = Array.from({ length: ATTEMPTS }, () => fakeRes({ statusCode: 200, headers: {} }));
    const reqs = wire(https, resList);
    fs.createWriteStream.mockImplementation(() => fakeWrite());
    fs.statSync.mockReturnValue({ size: 0 });
    fs.unlink.mockImplementation((p, cb) => cb && cb());

    const p = io.downloadFile('https://host/f', '/tmp/f');
    const settled = p.then(() => null, (e) => e);
    for (let a = 1; a <= ATTEMPTS; a++) {
      reqs[a - 1]._timeoutCb(); // fire the idle-socket handler
      await advance(a);
    }
    expect((await settled).message).toMatch('download stalled');
    jest.useRealTimers();
  });

  it('retries a request error and surfaces it once attempts run out', async () => {
    jest.useFakeTimers();
    const resList = Array.from({ length: ATTEMPTS }, () => fakeRes({ statusCode: 200, headers: {} }));
    const reqs = wire(https, resList);
    fs.createWriteStream.mockImplementation(() => fakeWrite());
    fs.statSync.mockReturnValue({ size: 0 });
    fs.unlink.mockImplementation((p, cb) => cb && cb());

    const p = io.downloadFile('https://host/f', '/tmp/f');
    const settled = p.then(() => null, (e) => e);
    for (let a = 1; a <= ATTEMPTS; a++) {
      reqs[a - 1].emit('error', new Error('ECONNRESET'));
      await advance(a);
    }
    expect((await settled).message).toMatch('ECONNRESET');
    jest.useRealTimers();
  });
});

describe('streamChatCompletion', () => {
  it('emits batched deltas and resolves when the stream signals done', async () => {
    const res = fakeRes({ statusCode: 200 });
    wire(http, [res]);
    const deltas = [];
    const { done } = io.streamChatCompletion('http://127.0.0.1:8080', { messages: [] },
      (text) => deltas.push(text));

    res.emit('data', 'data: {"choices":[{"delta":{"content":"He"}}]}\n\n');
    res.emit('data', 'data: {"choices":[{"delta":{"content":"llo"}}]}\n\ndata: [DONE]\n\n');
    await expect(done).resolves.toEqual({ finishReason: null });
    expect(deltas.join('')).toBe('Hello');
    expect(res.destroy).toHaveBeenCalled();
  });

  it('resolves on a normal stream end, and a second end is a no-op', async () => {
    const res = fakeRes({ statusCode: 200 });
    wire(https, [res]);
    const { done } = io.streamChatCompletion('https://host', { messages: [] }, () => {});
    res.emit('end');
    res.emit('end'); // idempotent — finish() must guard on `settled`
    await expect(done).resolves.toEqual({ finishReason: null });
  });

  it('ignores chunks that carry no delta', async () => {
    const res = fakeRes({ statusCode: 200 });
    wire(http, [res]);
    const onDelta = jest.fn();
    const { done } = io.streamChatCompletion('http://127.0.0.1:8080', {}, onDelta);
    // A role-only opening frame has no content delta.
    res.emit('data', 'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n');
    res.emit('data', 'data: [DONE]\n\n');
    await expect(done).resolves.toEqual({ finishReason: null });
    expect(onDelta).not.toHaveBeenCalled();
  });

  it('reports reasoning_content and resolves with the finish_reason', async () => {
    const res = fakeRes({ statusCode: 200 });
    wire(http, [res]);
    const onDelta = jest.fn();
    const onReasoning = jest.fn();
    const { done } = io.streamChatCompletion('http://127.0.0.1:8080', {}, onDelta, onReasoning);

    res.emit('data', 'data: {"choices":[{"delta":{"reasoning_content":"hm"}}]}\n\n');
    res.emit('data', 'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\ndata: [DONE]\n\n');
    await expect(done).resolves.toEqual({ finishReason: 'length' });
    expect(onReasoning).toHaveBeenCalledWith('hm', 1);
    expect(onDelta).not.toHaveBeenCalled();
  });

  it('drops reasoning when the caller supplies no onReasoning', async () => {
    const res = fakeRes({ statusCode: 200 });
    wire(http, [res]);
    const { done } = io.streamChatCompletion('http://127.0.0.1:8080', {}, () => {});
    res.emit('data', 'data: {"choices":[{"delta":{"reasoning_content":"hm"}}]}\n\ndata: [DONE]\n\n');
    await expect(done).resolves.toEqual({ finishReason: null });
  });

  it('rejects on a non-200 from llama-server', async () => {
    const res = fakeRes({ statusCode: 503 });
    wire(http, [res]);
    const { done } = io.streamChatCompletion('http://127.0.0.1:8080', {}, () => {});
    await expect(done).rejects.toThrow('llama-server HTTP 503');
  });

  it('rejects on a response error, and a later error is a no-op', async () => {
    const res = fakeRes({ statusCode: 200 });
    const reqs = wire(http, [res]);
    const { done } = io.streamChatCompletion('http://127.0.0.1:8080', {}, () => {});
    res.emit('error', new Error('stream reset'));
    reqs[0].emit('error', new Error('later')); // idempotent — fail() guards on `settled`
    await expect(done).rejects.toThrow('stream reset');
  });

  it('rejects on a request error', async () => {
    const res = fakeRes({ statusCode: 200 });
    const reqs = wire(http, [res]);
    const { done } = io.streamChatCompletion('http://127.0.0.1:8080', {}, () => {});
    reqs[0].emit('error', new Error('connect refused'));
    await expect(done).rejects.toThrow('connect refused');
  });

  it('cancel settles done before destroying the request', async () => {
    const res = fakeRes({ statusCode: 200 });
    const reqs = wire(http, [res]);
    const { done, cancel } = io.streamChatCompletion('http://127.0.0.1:8080', {}, () => {});
    cancel('user stopped');
    await expect(done).rejects.toThrow('user stopped');
    expect(reqs[0].destroy).toHaveBeenCalled();
  });

  it('cancel uses a default reason', async () => {
    const res = fakeRes({ statusCode: 200 });
    wire(http, [res]);
    const { done, cancel } = io.streamChatCompletion('http://127.0.0.1:8080', {}, () => {});
    cancel();
    await expect(done).rejects.toThrow('cancelled');
  });

  it('returns a no-op cancel and rejects on an invalid base URL', async () => {
    const { done, cancel } = io.streamChatCompletion('::bad::', {}, () => {});
    expect(() => cancel()).not.toThrow();
    await expect(done).rejects.toBeDefined();
  });
});

describe('extractLlamaZip', () => {
  function magic(bytes) {
    fs.openSync.mockReturnValue(7);
    fs.readSync.mockImplementation((fd, buf) => { buf[0] = bytes[0]; buf[1] = bytes[1]; return 2; });
    fs.closeSync.mockImplementation(() => {});
  }

  it('extracts a gzip tarball with tar and strip-components', async () => {
    magic([0x1f, 0x8b]);
    execFile.mockImplementation((tool, args, opts, cb) => cb(null));
    fs.existsSync.mockReturnValue(true);

    await expect(io.extractLlamaZip('/tmp/a.zip', '/opt/llama/llama-server')).resolves.toBe('/opt/llama/llama-server');
    const [tool, args] = execFile.mock.calls[0];
    expect(tool).toBe('tar');
    expect(args).toContain('--strip-components=1');
  });

  it('extracts a plain zip with unzip -j', async () => {
    magic([0x50, 0x4b]); // "PK"
    execFile.mockImplementation((tool, args, opts, cb) => cb(null));
    fs.existsSync.mockReturnValue(true);

    await expect(io.extractLlamaZip('/tmp/a.zip', '/opt/llama/llama-server')).resolves.toBeTruthy();
    expect(execFile.mock.calls[0][0]).toBe('unzip');
  });

  it('rejects when the archive cannot be read', async () => {
    fs.openSync.mockImplementation(() => { throw new Error('EACCES'); });
    await expect(io.extractLlamaZip('/tmp/a.zip', '/opt/x')).rejects.toThrow('could not read the llama-server archive');
  });

  it('rejects with the hint when extraction fails', async () => {
    magic([0x50, 0x4b]);
    execFile.mockImplementation((tool, args, opts, cb) => cb(new Error('unzip missing')));
    await expect(io.extractLlamaZip('/tmp/a.zip', '/opt/x', 'install unzip'))
      .rejects.toThrow(/could not extract.*install unzip/s);
  });

  it('rejects without a hint when none is given', async () => {
    magic([0x1f, 0x8b]);
    execFile.mockImplementation((tool, args, opts, cb) => cb(new Error('tar missing')));
    const err = await io.extractLlamaZip('/tmp/a.zip', '/opt/x').catch((e) => e);
    expect(err.message).toContain('could not extract');
    expect(err.message).not.toContain('—');
  });

  it('rejects when the binary is absent from the archive', async () => {
    magic([0x50, 0x4b]);
    execFile.mockImplementation((tool, args, opts, cb) => cb(null));
    fs.existsSync.mockReturnValue(false);
    await expect(io.extractLlamaZip('/tmp/a.zip', '/opt/x')).rejects.toThrow('was not found in the downloaded archive');
  });
});

// ── HTTPS trust recovery ────────────────────────────────────────────────────
// The rig that prompted this printed `engine setup failed: unable to verify the
// first certificate` and never got an engine. Node trusts only its compiled-in
// roots and never chases the AIA pointer, so a server that omits its
// intermediate — or a proxy/antivirus re-signing TLS with a root the OS trusts
// but Node can't see — kills a download the same machine makes fine in a
// browser. Recovery re-tries with more anchors; it never stops verifying.
describe('trustRecoveryCa', () => {
  const AIA = 'http://ca.example/issuer.cer';

  function fakeSocket() {
    const s = new EventEmitter();
    s.getPeerCertificate = jest.fn(() => ({}));
    s.destroy = jest.fn();
    return s;
  }

  // tls.connect calls back only once the handshake lands, i.e. after the caller
  // has the socket in hand — mirror that or the callback sees no socket.
  function wireTls(setup) {
    tls.connect.mockImplementation((opts, cb) => {
      const s = fakeSocket();
      setImmediate(() => setup(s, cb, opts));
      return s;
    });
  }

  // The certificate chain the server presented, leaf first.
  function wireChain(leaf) {
    wireTls((s, cb) => { s.getPeerCertificate.mockReturnValue(leaf); cb(); });
  }

  function certWithAia(uri) {
    return { infoAccess: { 'CA Issuers - URI': [uri] } };
  }

  // One response for the AIA fetch.
  function wireCertFetch(lib, drive) {
    lib.get = jest.fn((url, opts, cb) => {
      const req = fakeReq();
      setImmediate(() => drive(req, cb, url));
      return req;
    });
  }

  beforeEach(() => {
    // No OS trust store unless a test puts one there.
    fs.readFileSync.mockImplementation(() => { throw new Error('ENOENT'); });
    tls.connect.mockReset();
    wireTls((s) => s.emit('error', new Error('handshake failed')));
    delete process.env.NODE_EXTRA_CA_CERTS;
  });

  it('adds the OS trust store, which Node otherwise ignores', async () => {
    // Exactly the corporate-proxy / antivirus-TLS-scanner case: the root is
    // installed system-wide, so the user's browser is happy and Node is not.
    fs.readFileSync.mockImplementation((p) => {
      if (p !== '/etc/ssl/certs/ca-certificates.crt') throw new Error('ENOENT');
      return 'UBUNTU-BUNDLE';
    });
    await expect(io.trustRecoveryCa('https://host/f.bin')).resolves.toEqual(['ROOT', 'UBUNTU-BUNDLE']);
  });

  it('honours NODE_EXTRA_CA_CERTS', async () => {
    process.env.NODE_EXTRA_CA_CERTS = '/opt/corp.pem';
    fs.readFileSync.mockImplementation((p) => {
      if (p !== '/opt/corp.pem') throw new Error('ENOENT');
      return 'CORP';
    });
    await expect(io.trustRecoveryCa('https://host/f.bin')).resolves.toEqual(['ROOT', 'CORP']);
  });

  it('is null when there is nothing new to trust — retrying would just fail again', async () => {
    await expect(io.trustRecoveryCa('https://host/f.bin')).resolves.toBeNull();
  });

  it('fetches the intermediate the server points at and PEM-wraps the DER', async () => {
    wireChain(certWithAia(AIA));
    wireCertFetch(http, (req, cb) => {
      const res = fakeRes({ statusCode: 200 });
      cb(res);
      res.emit('data', Buffer.from('DERBYTES'));
      res.emit('end');
    });

    const ca = await io.trustRecoveryCa('https://host/f.bin');
    expect(ca).toHaveLength(2);
    expect(ca[1]).toBe('-----BEGIN CERTIFICATE-----\n'
      + Buffer.from('DERBYTES').toString('base64') + '\n-----END CERTIFICATE-----\n');
    // The probe must not have been pointed anywhere but the failing host.
    expect(tls.connect.mock.calls[0][0]).toMatchObject({ host: 'host', port: 443, servername: 'host' });
  });

  it('walks to the top of the presented chain and takes its issuer pointer', async () => {
    // The leaf's own AIA is not the missing link — the certificate at the end of
    // what the server actually sent is.
    const root = certWithAia(AIA);
    root.issuerCertificate = root; // self-signed: the walk stops here
    wireChain({ infoAccess: { 'CA Issuers - URI': ['http://ca.example/leaf.cer'] }, issuerCertificate: root });
    wireCertFetch(http, (req, cb) => {
      const res = fakeRes({ statusCode: 200 });
      cb(res);
      res.emit('data', Buffer.from('X'));
      res.emit('end');
    });

    await io.trustRecoveryCa('https://host/f.bin');
    expect(http.get.mock.calls[0][0]).toBe(AIA);
  });

  it('does not spin on a chain that loops back on itself', async () => {
    const a = {};
    const b = { issuerCertificate: a };
    a.issuerCertificate = b;
    wireChain(a);
    await expect(io.trustRecoveryCa('https://host/f.bin')).resolves.toBeNull();
  });

  it('passes a PEM body straight through and uses https for an https AIA URL', async () => {
    wireChain(certWithAia('https://ca.example/issuer.pem'));
    wireCertFetch(https, (req, cb) => {
      const res = fakeRes({ statusCode: 200 });
      cb(res);
      res.emit('data', Buffer.from('-----BEGIN CERTIFICATE-----\nabc\n-----END CERTIFICATE-----\n'));
      res.emit('end');
    });
    const ca = await io.trustRecoveryCa('https://host/f.bin');
    expect(ca[1]).toContain('abc');
  });

  it.each([
    ['a certificate with no AIA extension at all', {}],
    ['an AIA extension with no CA-Issuers entry', { infoAccess: {} }],
    ['an empty CA-Issuers list', { infoAccess: { 'CA Issuers - URI': [] } }],
    ['no peer certificate', null],
  ])('gives up quietly on %s', async (_label, leaf) => {
    wireChain(leaf);
    await expect(io.trustRecoveryCa('https://host/f.bin')).resolves.toBeNull();
  });

  it('gives up quietly when the probe handshake times out', async () => {
    wireTls((s) => s.emit('timeout'));
    await expect(io.trustRecoveryCa('https://host/f.bin')).resolves.toBeNull();
  });

  it('gives up quietly on an unparseable URL', async () => {
    await expect(io.trustRecoveryCa('not a url')).resolves.toBeNull();
    expect(tls.connect).not.toHaveBeenCalled();
  });

  it.each([
    ['a non-200 from the CA', (req, cb) => { cb(fakeRes({ statusCode: 404 })); }],
    ['an empty body', (req, cb) => { const r = fakeRes({ statusCode: 200 }); cb(r); r.emit('end'); }],
    ['a transport error', (req) => { req.emit('error', new Error('reset')); }],
    ['a timeout', (req) => { req.emit('timeout'); }],
    ['a stream error', (req, cb) => { const r = fakeRes({ statusCode: 200 }); cb(r); r.emit('error', new Error('boom')); }],
    ['an absurdly large body', (req, cb) => {
      const r = fakeRes({ statusCode: 200 });
      cb(r);
      r.emit('data', Buffer.alloc(70000));
    }],
    ['an unparseable AIA URL', null],
  ])('gives up quietly on %s', async (label, drive) => {
    wireChain(certWithAia(drive ? AIA : '::not a url::'));
    if (drive) wireCertFetch(http, drive);
    await expect(io.trustRecoveryCa('https://host/f.bin')).resolves.toBeNull();
  });
});

describe('downloadFile trust recovery', () => {
  const TLS_ERR = () => Object.assign(new Error('unable to verify the first certificate'),
    { code: 'UNABLE_TO_VERIFY_LEAF_SIGNATURE' });

  // Drive each https.get call with its own handler.
  function wireGets(handlers) {
    let i = 0;
    https.get = jest.fn((url, a, b) => {
      const cb = typeof a === 'function' ? a : b;
      const req = fakeReq();
      const h = handlers[Math.min(i, handlers.length - 1)];
      i++;
      h(req, cb);
      return req;
    });
  }

  beforeEach(() => {
    fs.readFileSync.mockImplementation(() => { throw new Error('ENOENT'); });
    tls.connect.mockReset();
    tls.connect.mockImplementation(() => {
      const s = new EventEmitter();
      s.destroy = jest.fn();
      setImmediate(() => s.emit('error', new Error('no probe')));
      return s;
    });
  });

  it('retries once with the recovered anchors and still verifies', async () => {
    fs.readFileSync.mockImplementation((p) => {
      if (p !== '/etc/ssl/certs/ca-certificates.crt') throw new Error('ENOENT');
      return 'UBUNTU-BUNDLE';
    });
    const out = new EventEmitter();
    out.close = (cb) => cb();
    out.destroy = jest.fn();
    fs.createWriteStream.mockReturnValue(out);
    fs.renameSync.mockImplementation(() => {});
    wireGets([
      (req) => setImmediate(() => req.emit('error', TLS_ERR())),
      (req, cb) => { cb(fakeRes({ statusCode: 200, headers: {} })); setImmediate(() => out.emit('finish')); },
    ]);

    await expect(io.downloadFile('https://host/f.bin', '/tmp/f.bin')).resolves.toBe('/tmp/f.bin');
    // Two calls, not five: a trust failure is deterministic, so the four-attempt
    // backoff is skipped in favour of going straight to recovery.
    expect(https.get).toHaveBeenCalledTimes(2);
    expect(https.get.mock.calls[1][1]).toEqual({ ca: ['ROOT', 'UBUNTU-BUNDLE'] });
  });

  it('reports the certificate failure when there is nothing left to try', async () => {
    wireGets([(req) => setImmediate(() => req.emit('error', TLS_ERR()))]);
    await expect(io.downloadFile('https://host/f.bin', '/tmp/f.bin'))
      .rejects.toThrow('unable to verify the first certificate');
    expect(https.get).toHaveBeenCalledTimes(1);
  });

  it('leaves other failures to the existing retry path', async () => {
    wireGets([(req, cb) => { cb(fakeRes({ statusCode: 404, headers: {} })); }]);
    await expect(io.downloadFile('https://host/f.bin', '/tmp/f.bin')).rejects.toThrow('HTTP 404');
    expect(tls.connect).not.toHaveBeenCalled();
  });
});
