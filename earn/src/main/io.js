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
const tls = require('tls');
const { execFile } = require('child_process');
const { progressPercent } = require('../shared/engine');
const { isTlsTrustError } = require('../shared/engineError');
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

// ── HTTPS trust recovery ────────────────────────────────────────────────────
//
// Node verifies against its own compiled-in Mozilla root list and, unlike every
// browser, never chases the AIA pointer to fetch a missing intermediate. So two
// ordinary rig setups fail a download that the same machine fetches fine in
// Chrome, both surfacing as `unable to verify the first certificate`:
//
//   • the server presents its leaf without the intermediate, leaving the caller
//     to find it (a server misconfiguration browsers paper over), and
//   • a proxy, VPN or antivirus TLS-scanner re-signs the connection with a
//     private root that is installed in the OS trust store — which Node ignores.
//
// Both are fixable without weakening anything: retry with the OS trust store and
// with the certificate the server itself points at added to the anchors. The
// retry still verifies in full, so a fetched intermediate only helps if it
// chains to a root the machine already trusts; nothing here can make an
// untrusted chain pass. Disabling rejectUnauthorized would "fix" the same
// symptom by accepting any MITM handing us a binary we then execute — never do
// that here.

// Where distributions keep the OS trust store. Read as text and appended to
// Node's own roots; a path that doesn't exist on this distro is simply skipped.
const SYSTEM_CA_FILES = [
  '/etc/ssl/certs/ca-certificates.crt', // Debian / Ubuntu
  '/etc/pki/tls/certs/ca-bundle.crt',   // RHEL / Fedora
  '/etc/ssl/ca-bundle.pem',             // SUSE
  '/etc/ssl/cert.pem',                  // Alpine, macOS OpenSSL builds
];

const CERT_FETCH_TIMEOUT_MS = 10000;
const CERT_MAX_BYTES = 65536; // a certificate is a couple of KB; more is junk

function systemCaCerts() {
  const files = SYSTEM_CA_FILES.slice();
  if (process.env.NODE_EXTRA_CA_CERTS) files.unshift(process.env.NODE_EXTRA_CA_CERTS);
  const out = [];
  for (const f of files) {
    try { out.push(fs.readFileSync(f, 'utf8')); } catch (e) { /* not on this system */ }
  }
  return out;
}

// The CA-Issuers URL advertised by the top-most certificate the server actually
// presented, or null. The probe handshake runs unverified on purpose: it sends
// no request and trusts no response, it only READS the offered chain so we know
// which issuer to go and fetch. Whatever it yields is a candidate that the real,
// fully verified download must still validate against a trusted root.
function peerIssuerUrl(url) {
  return new Promise((resolve) => {
    let u;
    try { u = new URL(url); } catch (e) { return resolve(null); }
    const done = (v) => { socket.destroy(); resolve(v); };
    const socket = tls.connect({
      host: u.hostname,
      port: u.port || 443,
      servername: u.hostname,
      rejectUnauthorized: false,
      timeout: CERT_FETCH_TIMEOUT_MS,
    }, () => {
      let cert = socket.getPeerCertificate(true);
      // Walk to the end of the presented chain — the certificate whose issuer is
      // the one the rig is missing. Bounded so a server looping its own chain
      // back on itself can't spin here.
      for (let i = 0; i < 8 && cert && cert.issuerCertificate && cert.issuerCertificate !== cert; i++) {
        cert = cert.issuerCertificate;
      }
      const uris = (cert && cert.infoAccess && cert.infoAccess['CA Issuers - URI']) || [];
      done(uris[0] || null);
    });
    socket.on('error', () => done(null));
    socket.on('timeout', () => done(null));
  });
}

// Fetch a certificate from an AIA CA-Issuers URL (published as raw DER, though
// PEM is accepted). Never rejects — a null just means no recovery from here.
function fetchCert(url) {
  return new Promise((resolve) => {
    let lib;
    try { lib = new URL(url).protocol === 'https:' ? https : http; } catch (e) { return resolve(null); }
    const req = lib.get(url, { timeout: CERT_FETCH_TIMEOUT_MS }, (res) => {
      if (res.statusCode !== 200) { res.resume(); return resolve(null); }
      const chunks = [];
      let len = 0;
      res.on('data', (c) => {
        len += c.length;
        if (len > CERT_MAX_BYTES) { req.destroy(); return resolve(null); }
        chunks.push(c);
      });
      res.on('end', () => resolve(certToPem(Buffer.concat(chunks))));
      res.on('error', () => resolve(null));
    });
    req.on('error', () => resolve(null));
    req.on('timeout', () => { req.destroy(); resolve(null); });
  });
}

function certToPem(buf) {
  if (!buf.length) return null;
  const text = buf.toString('utf8');
  if (text.includes('-----BEGIN CERTIFICATE-----')) return text;
  return '-----BEGIN CERTIFICATE-----\n'
    + buf.toString('base64').match(/.{1,64}/g).join('\n')
    + '\n-----END CERTIFICATE-----\n';
}

// Node's roots plus everything we could scrape together for `url`, or null when
// there is nothing new to add (retrying with the same anchors would just fail
// again).
async function trustRecoveryCa(url) {
  const extra = systemCaCerts();
  const issuerUrl = await peerIssuerUrl(url);
  if (issuerUrl) {
    const pem = await fetchCert(issuerUrl);
    if (pem) extra.push(pem);
  }
  return extra.length ? tls.rootCertificates.concat(extra) : null;
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
// A dropped connection is retried a few times with a backoff, and every attempt
// starts the scratch file from scratch.
//
// Resuming from the bytes already on disk (a Range request, appending to the
// partial) is the obvious optimisation and was deliberately removed: the model
// is only ever validated with existsSync, so a resume that got the offset wrong
// would leave a corrupt multi-GB GGUF at the final path that every later start
// accepts, failing to load with an opaque error until someone manually deletes
// a file they don't know about. Getting it right needs Content-Range validation
// (a server may answer 206 from an offset other than the one asked for), care
// around bytes still buffered in the write stream when it is destroyed, and a
// size or checksum check before the rename. None of that is worth carrying to
// avoid re-downloading on the minority of transfers that drop — especially now
// that a stalled setup is visible on the hero rather than silent, so a user can
// see what happened and retry deliberately.
const DOWNLOAD_ATTEMPTS = 4;
const DOWNLOAD_RETRY_MS = 2000;

function downloadFile(url, dest, onProgress, redirects) {
  const part = dest + '.' + process.pid + '.' + (partSeq = (partSeq + 1) % 1e6) + '.part';
  return downloadAttempt(url, dest, part, onProgress, redirects || 0, 1, null)
    .catch(async (err) => {
      // Certificate chain we couldn't verify: gather more trust anchors (OS
      // store + the issuer the server points at) and try once more, still fully
      // verified. Anything else — and a second failure — is the caller's.
      if (!isTlsTrustError(err)) throw err;
      // err.url is where the handshake actually failed, which after a redirect
      // is not the URL we asked for.
      const ca = await trustRecoveryCa(err.url);
      if (!ca) throw err;
      return downloadAttempt(url, dest, part, onProgress, redirects || 0, 1, ca);
    });
}

// One HTTP attempt for `url` into `part`. On a transport failure it waits and
// re-enters itself, until DOWNLOAD_ATTEMPTS is reached. `ca` overrides the trust
// anchors (see downloadFile's recovery pass); null means Node's defaults.
function downloadAttempt(url, dest, part, onProgress, redirects, attempt, ca) {
  return new Promise((resolve, reject) => {
    if (redirects > 5) return reject(new Error('too many redirects'));

    const retryOrFail = (err) => {
      // Remember where it failed so the recovery pass probes the host that
      // actually rejected us, which after a redirect isn't the one we asked for.
      err.url = url;
      // A trust failure is deterministic — three more identical handshakes only
      // delay the recovery pass by the backoff.
      if (attempt >= DOWNLOAD_ATTEMPTS || isTlsTrustError(err)) {
        fs.unlink(part, () => {});
        return reject(err);
      }
      const timer = setTimeout(() => {
        downloadAttempt(url, dest, part, onProgress, redirects, attempt + 1, ca).then(resolve, reject);
      }, DOWNLOAD_RETRY_MS * attempt);
      if (timer.unref) timer.unref();
    };

    const lib = url.startsWith('https') ? https : http;
    const onResponse = (res) => {
      const code = res.statusCode || 0;
      if (code >= 300 && code < 400 && res.headers.location) {
        res.resume();
        const next = new URL(res.headers.location, url).toString();
        return resolve(downloadAttempt(next, dest, part, onProgress, redirects + 1, attempt, ca));
      }
      if (code !== 200) {
        res.resume();
        return reject(new Error('HTTP ' + code + ' for ' + url));
      }
      const total = parseInt(res.headers['content-length'] || '0', 10);
      let received = 0;
      // Truncating open: a retry overwrites whatever the failed attempt left,
      // so no attempt ever reasons about partial state.
      const out = fs.createWriteStream(part);
      const fail = (err) => { out.destroy(); retryOrFail(err); };
      res.on('data', (c) => { received += c.length; if (onProgress) onProgress(progressPercent(received, total)); });
      res.on('error', fail);
      res.pipe(out);
      out.on('finish', () => out.close(() => {
        try { fs.renameSync(part, dest); } catch (e) { return reject(e); }
        resolve(dest);
      }));
      out.on('error', fail);
    };
    const req = ca ? lib.get(url, { ca }, onResponse) : lib.get(url, onResponse);
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

module.exports = {
  postJson,
  getJson,
  downloadFile,
  trustRecoveryCa,
  streamChatCompletion,
  extractLlamaZip,
};
