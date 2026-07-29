'use strict';

// Shared plain-Node IO for both shells (Electron main and the headless CLI):
// JSON POSTs to the LLMJob server, resumable-safe file downloads, the SSE
// chat-completions stream against the local llama-server, and llama.cpp zip
// extraction. One implementation instead of per-shell copies, so protocol and
// bug fixes land everywhere at once. No Electron dependencies.

const fs = require('fs');
const path = require('path');
const http = require('http');
const https = require('https');
const { execFile } = require('child_process');
const { progressPercent } = require('../shared/engine');
const { parseChatStream } = require('../shared/llmChat');

// Minimal JSON POST → { status, data, raw }. Resolves for ANY HTTP status
// (callers must check `status`); rejects only on transport errors/timeouts.
function postJson(url, body, timeoutMs) {
  return new Promise((resolve, reject) => {
    let u;
    try { u = new URL(url); } catch (e) { return reject(e); }
    const lib = u.protocol === 'http:' ? http : https;
    const payload = JSON.stringify(body);
    const req = lib.request(u, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) },
      timeout: timeoutMs || 30000,
    }, (res) => {
      let raw = '';
      res.setEncoding('utf8');
      res.on('data', (c) => { raw += c; });
      res.on('end', () => {
        let data = null;
        try { data = raw ? JSON.parse(raw) : null; } catch (e) { /* non-JSON */ }
        resolve({ status: res.statusCode || 0, data, raw });
      });
    });
    req.on('error', reject);
    req.on('timeout', () => { req.destroy(new Error('request timed out')); });
    req.write(payload);
    req.end();
  });
}

// Best-effort JSON GET: resolves the parsed object, or null on any error,
// non-200, timeout, or unparseable body. Never rejects. `opts.headers` and
// `opts.timeout` (ms, default 8000) tune the request; a 4 MB cap guards against
// a runaway body. Shared by the app's economics/balance fetches and the CLI
// self-updater's release check, which each used to keep their own copy.
function getJson(url, opts = {}) {
  return new Promise((resolve) => {
    try {
      const u = new URL(url);
      const lib = u.protocol === 'http:' ? http : https;
      const req = lib.get(u, { timeout: opts.timeout || 8000, headers: opts.headers }, (res) => {
        if (res.statusCode !== 200) { res.resume(); return resolve(null); }
        let data = '';
        res.on('data', (c) => { data += c; if (data.length > 4e6) req.destroy(); });
        res.on('end', () => { try { resolve(JSON.parse(data)); } catch (e) { resolve(null); } });
      });
      req.on('error', () => resolve(null));
      req.on('timeout', () => { req.destroy(); resolve(null); });
    } catch (e) {
      resolve(null);
    }
  });
}

// Serial number for scratch download paths — see `part` in downloadFile.
let partSeq = 0;

// Stream a URL to a file, following redirects and reporting download progress.
// Writes to a scratch `<dest>.<pid>.<n>.part` and renames on completion, so an
// interrupted download (multi-GB GGUFs especially) never leaves a truncated file
// at the final path that an existsSync "is it installed?" check would wrongly
// accept. A 60s idle socket timeout stops a stalled connection from hanging
// setup forever.
//
// The scratch name is unique per call, not a fixed `<dest>.part`: two downloads
// of the same dest raced on that one path, and whichever finished first renamed
// it out from under the others, which then died on `ENOENT … rename
// llama-download.archive.part`. Uniqueness makes concurrent attempts merely
// redundant (last writer wins the rename) instead of failing.
// A dropped connection part-way through a multi-GB download used to throw the
// whole transfer away. Retry a few times, resuming from what already landed on
// disk via a Range request, so a 4.6 GB model that dies at 57% continues from
// 57% instead of restarting at zero — on a link that drops every few minutes,
// restarting never converges. Attempts are per call, and the scratch file is
// removed once they are exhausted: a partial left between runs could belong to a
// different URL, and validating that is not worth the complexity.
const DOWNLOAD_ATTEMPTS = 4;
const DOWNLOAD_RETRY_MS = 2000;

function downloadFile(url, dest, onProgress, redirects) {
  const part = dest + '.' + process.pid + '.' + (partSeq = (partSeq + 1) % 1e6) + '.part';
  return downloadAttempt(url, dest, part, onProgress, redirects || 0, 0, 1);
}

// One HTTP attempt for `url` into `part`, resuming at byte `resumeFrom`. On a
// transport failure it waits and re-enters itself with whatever is on disk,
// until DOWNLOAD_ATTEMPTS is reached.
function downloadAttempt(url, dest, part, onProgress, redirects, resumeFrom, attempt) {
  return new Promise((resolve, reject) => {
    if (redirects > 5) return reject(new Error('too many redirects'));

    // Give up for good, or wait and resume. The byte count comes from the file
    // on disk rather than from what we counted: the write stream may be holding
    // buffered chunks that never landed, and resuming past them would corrupt
    // the file with a gap.
    const retryOrFail = (err) => {
      if (attempt >= DOWNLOAD_ATTEMPTS) { fs.unlink(part, () => {}); return reject(err); }
      let have = 0;
      try { have = fs.statSync(part).size; } catch (e) { have = 0; }
      const timer = setTimeout(() => {
        downloadAttempt(url, dest, part, onProgress, redirects, have, attempt + 1).then(resolve, reject);
      }, DOWNLOAD_RETRY_MS * attempt);
      if (timer.unref) timer.unref();
    };

    const lib = url.startsWith('https') ? https : http;
    const opts = resumeFrom > 0 ? { headers: { Range: 'bytes=' + resumeFrom + '-' } } : {};
    const req = lib.get(url, opts, (res) => {
      const code = res.statusCode || 0;
      if (code >= 300 && code < 400 && res.headers.location) {
        res.resume();
        const next = new URL(res.headers.location, url).toString();
        return resolve(downloadAttempt(next, dest, part, onProgress, redirects + 1, resumeFrom, attempt));
      }
      // A server that ignores Range answers 200 with the whole body again, so
      // start the scratch file over rather than appending a second copy onto the
      // bytes we already have.
      const restart = resumeFrom > 0 && code === 200;
      if (restart) resumeFrom = 0;
      if (code !== 200 && code !== 206) {
        res.resume();
        return reject(new Error('HTTP ' + code + ' for ' + url));
      }
      // On a 206 content-length is the size of the REMAINDER, so the real total
      // is what we already hold plus what is coming.
      const len = parseInt(res.headers['content-length'] || '0', 10);
      const total = resumeFrom > 0 && len ? resumeFrom + len : len;
      let received = resumeFrom;
      const out = fs.createWriteStream(part, resumeFrom > 0 ? { flags: 'a' } : undefined);
      const fail = (err) => { out.destroy(); retryOrFail(err); };
      res.on('data', (c) => { received += c.length; if (onProgress) onProgress(progressPercent(received, total)); });
      res.on('error', fail);
      res.pipe(out);
      out.on('finish', () => out.close(() => {
        try { fs.renameSync(part, dest); } catch (e) { return reject(e); }
        resolve(dest);
      }));
      out.on('error', fail);
    });
    req.setTimeout(60000, () => req.destroy(new Error('download stalled (no data for 60s)')));
    req.on('error', retryOrFail);
  });
}

// Stream a chat request to the local llama-server's OpenAI endpoint. Deltas are
// batched per network chunk — onDelta(text, tokenCount) — instead of one call
// per token. A thinking model's `reasoning_content` is reported the same way via
// the optional onReasoning(text, tokenCount), so those tokens are accounted for
// rather than silently dropped. Returns { done, cancel }: `done` resolves with
// { finishReason } when the stream finishes and rejects on transport/HTTP errors
// or cancel(reason); cancel is safe to call at any point and settles `done`
// before destroying the request, so callers always observe an outcome (no
// orphaned in-flight state).
function streamChatCompletion(baseUrl, chatBody, onDelta, onReasoning) {
  let settled = false;
  let finishReason = null;
  let resolveDone, rejectDone;
  const done = new Promise((res, rej) => { resolveDone = res; rejectDone = rej; });
  const finish = () => { if (!settled) { settled = true; resolveDone({ finishReason }); } };
  const fail = (err) => { if (!settled) { settled = true; rejectDone(err); } };

  let url;
  try { url = new URL(baseUrl + '/v1/chat/completions'); } catch (e) {
    fail(e);
    return { done, cancel: () => {} };
  }
  const lib = url.protocol === 'http:' ? http : https;
  const payload = JSON.stringify(chatBody);
  const req = lib.request(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) },
  }, (res) => {
    if (res.statusCode !== 200) { res.resume(); return fail(new Error('llama-server HTTP ' + res.statusCode)); }
    let buf = '';
    res.setEncoding('utf8');
    res.on('data', (c) => {
      buf += c;
      const parsed = parseChatStream(buf);
      buf = parsed.rest;
      if (parsed.deltas.length) onDelta(parsed.deltas.join(''), parsed.deltas.length);
      if (parsed.reasoning.length && onReasoning) {
        onReasoning(parsed.reasoning.join(''), parsed.reasoning.length);
      }
      if (parsed.finishReason) finishReason = parsed.finishReason;
      if (parsed.done) { res.destroy(); finish(); }
    });
    res.on('end', finish);
    res.on('error', fail);
  });
  req.on('error', fail);
  req.write(payload);
  req.end();

  return {
    done,
    cancel: (reason) => { fail(new Error(reason || 'cancelled')); req.destroy(); },
  };
}

// Extract the llama.cpp release archive on Linux/macOS, flattening it into the
// install dir so `llama-server` lands next to its shared libraries (.so/.dylib)
// — llama.cpp resolves libs from the binary's own directory ($ORIGIN rpath), so
// co-locating them is what makes the downloaded server run. llama.cpp ships
// Linux/macOS as .tar.gz (a build-named top folder) and Windows as .zip; the
// download always lands at one format-neutral name (llmEngineManager's
// ARCHIVE_TMP), so sniff the magic bytes rather than trust the name: gzip
// (1f 8b) → `tar --strip-components=1`, otherwise `unzip -j`.
// `hint` is appended to the extraction error (e.g. the CLI's --llm-binary
// escape hatch).
function extractLlamaZip(zipPath, dest, hint) {
  return new Promise((resolve, reject) => {
    const dir = path.dirname(dest);
    let gzip = false;
    try {
      const fd = fs.openSync(zipPath, 'r');
      const head = Buffer.alloc(2);
      fs.readSync(fd, head, 0, 2, 0);
      fs.closeSync(fd);
      gzip = head[0] === 0x1f && head[1] === 0x8b;
    } catch (e) {
      return reject(new Error('could not read the llama-server archive (' + e.message + ')'));
    }
    const tool = gzip ? 'tar' : 'unzip';
    const args = gzip
      ? ['-xzf', zipPath, '-C', dir, '--strip-components=1'] // strip the build-named top folder
      : ['-o', '-j', zipPath, '-d', dir];                    // junk paths → flatten
    execFile(tool, args, { timeout: 120000 }, (err) => {
      if (err) {
        return reject(new Error('could not extract the llama-server archive with `' + tool + '` ('
          + err.message + ')' + (hint ? ' — ' + hint : '')));
      }
      if (!fs.existsSync(dest)) {
        return reject(new Error('llama-server was not found in the downloaded archive'));
      }
      resolve(dest);
    });
  });
}

module.exports = { postJson, getJson, downloadFile, streamChatCompletion, extractLlamaZip };
