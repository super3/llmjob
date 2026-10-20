'use strict';

// Monero Stratum, in the dialect XMRig made standard and every RandomX pool
// speaks (https://github.com/xmrig/xmrig-proxy/blob/master/doc/STRATUM.md).
// It is not the Pearl one in ../miner/stratum.js: there is a login with a
// session id instead of mining.authorize, jobs come as "job" with a hashing
// blob, and submits carry a nonce and the hash.
//
//   >> {"id":1,"jsonrpc":"2.0","method":"login","params":{"login":"4…","pass":"rig01",
//        "agent":"llmjob-earn/0.5.14","rigid":"rig01","algo":["rx/0"]}}
//   << {"id":1,"jsonrpc":"2.0","error":null,"result":{"id":"<session>","job":{…},
//        "extensions":["algo","keepalive",…],"status":"OK"}}
//   << {"jsonrpc":"2.0","method":"job","params":{"blob":"<hex>","job_id":"…",
//        "target":"b88d0600","algo":"rx/0","height":3781596,"seed_hash":"<64 hex>"}}
//   >> {"id":2,"jsonrpc":"2.0","method":"submit","params":{"id":"<session>",
//        "job_id":"…","nonce":"<8 hex>","result":"<64 hex>"}}
//   << {"id":2,"jsonrpc":"2.0","error":null,"result":{"status":"OK"}}
//   >> {"id":3,"jsonrpc":"2.0","method":"keepalived","params":{"id":"<session>"}}
//   << {"id":3,"jsonrpc":"2.0","error":null,"result":{"status":"KEEPALIVED"}}
//
// The worker name goes in both `pass` and `rigid`. HeroMiners reads it from
// `pass`; XMRig-era pools read `rigid`. It must not go after the address with a
// ".": HeroMiners' config has "paymentIdSeparator": ".", so that would be read
// as a payment ID.
//
// Kryptex (`--xmr-pool xmr-us.kryptex.network:7029`) names the worker from
// "address/worker" in `login`, as its Pearl pool does (see isKryptexHost).
// Only a Kryptex host gets the slash; HeroMiners' login stays exactly as it was.
//
// Pure: it builds request lines and classifies response lines. The socket is
// main/xmrEngine.js's.

const { isKryptexHost } = require('../miner/stratum');

const LOGIN_ID = 1;
const MAX_U64 = (1n << 64n) - 1n;

// The pool's algorithm names this miner can mine, and how the worker mines
// each: `variant` is the RandomX version (v1 or v2) and `commit` whether the
// target is checked against the RandomX commitment instead of the hash.
// Monero's v17 fork changes the proof of work to RandomX v2 with commitments
// (monero-project/monero#10038, no date yet). When pools name the new
// algorithm, it gets an entry here; the worker already does both.
const ALGOS = {
  'rx/0': { variant: 'v1', commit: false },
};
const DEFAULT_ALGO = 'rx/0';

// The nonce sits at byte 39 of a Monero hashing blob, so a blob shorter than
// 43 bytes has nowhere to put it.
const NONCE_OFFSET = 39;
const MIN_BLOB_BYTES = NONCE_OFFSET + 4;
const MAX_BLOB_BYTES = 512;

function buildLogin({ address, worker, agent, host }) {
  const name = String(worker == null ? '' : worker).trim() || 'x';
  const wallet = String(address == null ? '' : address).trim();
  return {
    id: LOGIN_ID,
    jsonrpc: '2.0',
    method: 'login',
    params: {
      login: isKryptexHost(host) ? wallet + '/' + name : wallet,
      pass: name,
      agent: agent || 'llmjob-earn',
      rigid: name,
      algo: Object.keys(ALGOS),
    },
  };
}

function buildSubmit(id, sessionId, share) {
  return {
    id,
    jsonrpc: '2.0',
    method: 'submit',
    params: { id: sessionId, job_id: share.jobId, nonce: share.nonce, result: share.result },
  };
}

function buildKeepalive(id, sessionId) {
  return { id, jsonrpc: '2.0', method: 'keepalived', params: { id: sessionId } };
}

function encode(msg) {
  return JSON.stringify(msg) + '\n';
}

// The pool's target as the 64-bit number a hash is compared with: a share is a
// hash whose last 8 bytes, read little-endian, are below it.
//
// It comes in two forms, both little-endian hex. Eight digits is the compact
// 32-bit form, which stands for the 64-bit target 2^64 / (2^32 / t), computed
// the way XMRig does so both agree to the unit. Sixteen digits is the 64-bit
// target itself. Returns null for anything else, or a zero target.
function parseTarget(hex) {
  const s = String(hex == null ? '' : hex).trim().toLowerCase();
  if (!/^[0-9a-f]+$/.test(s) || (s.length !== 8 && s.length !== 16)) return null;
  const n = BigInt('0x' + Buffer.from(s, 'hex').reverse().toString('hex'));
  if (n === 0n) return null;
  if (s.length === 16) return n;
  return MAX_U64 / (0xffffffffn / n);
}

// The share difficulty a target stands for: how many hashes one share is worth
// on average. This is what pools count, and what a hashrate is estimated from.
function targetDifficulty(target) {
  return Number(MAX_U64 / target);
}

// The target the worker takes: the 64-bit number as 16 big-endian hex digits.
function workerTarget(target) {
  return target.toString(16).padStart(16, '0');
}

function isHex(s, bytes) {
  return typeof s === 'string' && /^[0-9a-fA-F]*$/.test(s) && s.length % 2 === 0
    && (bytes == null || s.length === bytes * 2);
}

// A job's params, from a login result or a "job" notification, made ready for
// the worker. Anything the worker can't mine is { kind: 'bad-job', reason }.
function parseJob(params) {
  const p = params && typeof params === 'object' && !Array.isArray(params) ? params : {};
  const jobId = p.job_id == null ? '' : String(p.job_id);
  const bad = (reason) => ({ kind: 'bad-job', jobId: jobId || null, reason });
  if (!jobId) return bad('no job_id');
  const blob = typeof p.blob === 'string' ? p.blob.trim().toLowerCase() : '';
  if (!isHex(blob) || blob.length < MIN_BLOB_BYTES * 2 || blob.length > MAX_BLOB_BYTES * 2) {
    return bad('the blob is not ' + MIN_BLOB_BYTES + '-' + MAX_BLOB_BYTES + ' bytes of hex');
  }
  const target = parseTarget(p.target);
  if (target == null) return bad('unreadable target ' + JSON.stringify(p.target));
  const seedHash = typeof p.seed_hash === 'string' ? p.seed_hash.trim().toLowerCase() : '';
  if (!isHex(seedHash, 32)) return bad('no RandomX seed_hash');
  const algo = p.algo == null || p.algo === '' ? DEFAULT_ALGO : String(p.algo);
  const mode = Object.prototype.hasOwnProperty.call(ALGOS, algo) ? ALGOS[algo] : null;
  if (!mode) return bad('algorithm ' + algo + ' is not one this miner mines');
  return {
    kind: 'job',
    jobId,
    blob,
    target,
    difficulty: targetDifficulty(target),
    seedHash,
    height: Number.isFinite(p.height) ? p.height : null,
    algo,
    variant: mode.variant,
    commit: mode.commit,
  };
}

// The pool's errors arrive as {code, message}; reduce anything else to the
// same shape so the host has one thing to log.
function normalizeError(err) {
  if (err == null) return null;
  if (typeof err === 'object' && !Array.isArray(err)) {
    return { code: err.code == null ? null : err.code, message: String(err.message == null ? '' : err.message) };
  }
  return { code: null, message: String(err) };
}

// Classify one line from the pool. Never throws: a junk line becomes
// { kind: 'unparseable' } for the log, not a crash.
//
// Replies carry the id of the request they answer, so the host passes the ids
// of its keepalives (`keepaliveIds`, a Set) to tell those from submits.
function parseMessage(line, keepaliveIds) {
  const s = String(line == null ? '' : line).trim();
  if (!s) return { kind: 'empty' };
  let msg;
  try {
    msg = JSON.parse(s);
  } catch (e) {
    return { kind: 'unparseable', raw: s };
  }
  if (!msg || typeof msg !== 'object' || Array.isArray(msg)) return { kind: 'unknown', raw: s };

  if (msg.method === 'job') return parseJob(msg.params);
  if (msg.method != null) return { kind: 'unknown', raw: s };
  if (!('result' in msg) && !('error' in msg)) return { kind: 'unknown', raw: s };

  const error = normalizeError(msg.error);
  if (msg.id === LOGIN_ID) {
    const r = msg.result;
    if (error || !r || typeof r !== 'object' || !r.id) {
      return { kind: 'login-fail', error: error || { code: null, message: 'the pool sent no session id' } };
    }
    return {
      kind: 'login-ok',
      sessionId: String(r.id),
      job: r.job ? parseJob(r.job) : null,
      extensions: Array.isArray(r.extensions) ? r.extensions.map(String) : [],
    };
  }
  if (msg.id == null) {
    return error ? { kind: 'pool-error', error } : { kind: 'unknown', raw: s };
  }
  if (keepaliveIds && keepaliveIds.has(msg.id)) return { kind: 'keepalive', id: msg.id, error };
  if (error) return { kind: 'submit-rejected', id: msg.id, error };
  return { kind: 'submit-accepted', id: msg.id };
}

function errorText(err) {
  if (!err) return 'unknown error';
  return (err.code != null ? '[' + err.code + '] ' : '') + (err.message || 'unknown error');
}

module.exports = {
  LOGIN_ID, ALGOS, DEFAULT_ALGO, NONCE_OFFSET, MIN_BLOB_BYTES, MAX_BLOB_BYTES,
  buildLogin, buildSubmit, buildKeepalive, encode,
  parseTarget, targetDifficulty, workerTarget, parseJob, parseMessage, normalizeError, errorText,
};
