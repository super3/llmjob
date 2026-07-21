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

// Stream a URL to a file, following redirects and reporting download progress.
// Writes to `<dest>.part` and renames on completion, so an interrupted download
// (multi-GB GGUFs especially) never leaves a truncated file at the final path
// that an existsSync "is it installed?" check would wrongly accept. A 60s idle
// socket timeout stops a stalled connection from hanging setup forever.
function downloadFile(url, dest, onProgress, redirects) {
  redirects = redirects || 0;
  return new Promise((resolve, reject) => {
    if (redirects > 5) return reject(new Error('too many redirects'));
    const lib = url.startsWith('https') ? https : http;
    const req = lib.get(url, (res) => {
      const code = res.statusCode || 0;
      if (code >= 300 && code < 400 && res.headers.location) {
        res.resume();
        return resolve(downloadFile(new URL(res.headers.location, url).toString(), dest, onProgress, redirects + 1));
      }
      if (code !== 200) {
        res.resume();
        return reject(new Error('HTTP ' + code + ' for ' + url));
      }
      const total = parseInt(res.headers['content-length'] || '0', 10);
      let received = 0;
      const part = dest + '.part';
      const out = fs.createWriteStream(part);
      const fail = (err) => { out.destroy(); fs.unlink(part, () => {}); reject(err); };
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
    req.on('error', reject);
  });
}

// Stream a chat request to the local llama-server's OpenAI endpoint. Deltas are
// batched per network chunk — onDelta(text, tokenCount) — instead of one call
// per token. Returns { done, cancel }: `done` resolves when the stream finishes
// and rejects on transport/HTTP errors or cancel(reason); cancel is safe to call
// at any point and settles `done` before destroying the request, so callers
// always observe an outcome (no orphaned in-flight state).
function streamChatCompletion(baseUrl, chatBody, onDelta) {
  let settled = false;
  let resolveDone, rejectDone;
  const done = new Promise((res, rej) => { resolveDone = res; rejectDone = rej; });
  const finish = () => { if (!settled) { settled = true; resolveDone(); } };
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
// download is always named `.zip`, so sniff the magic bytes rather than trust
// the name: gzip (1f 8b) → `tar --strip-components=1`, otherwise `unzip -j`.
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

module.exports = { postJson, downloadFile, streamChatCompletion, extractLlamaZip };
