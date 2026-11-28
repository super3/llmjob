'use strict';

// The NEB pool protocol, version 1 (spec/pool.md of github.com/SilasEigen/neb):
// JSON lines in the style of stratum.
//
//   >> {"id":1,"method":"mining.subscribe","params":["llmjob-earn/0.8.0","neb-pool/1"]}
//   << {"id":1,"result":{"session":"f2f3b4ac","extranonce1":"ff681386","extranonce2_size":12,
//        "chain_id":"<64 hex>","dataset_hash":"<64 hex>"},"error":null}
//   >> {"id":2,"method":"mining.authorize","params":["neb1….rig01","x"]}
//   << {"id":2,"result":true,"error":null}
//   << {"id":null,"method":"mining.set_target","params":["<64 hex>"]}
//   << {"id":null,"method":"mining.set_model","params":[0,"<weights hash>","<13,000 bytes of hex>"]}
//   << {"id":null,"method":"mining.notify","params":[{"job_id":"a737f3","clean":true,"version":1,"parent":"…",
//        "time_start":1795840703,"time_window":3600,"bits":3096293279,"target":"…","share_target":"…",
//        "coinbase_prefix":"…","coinbase_suffix":"…","branch":["…"],"tx_count":2,"shares_root":"…",
//        "height":13990,"slot":0,"weights_hash":"…","expiry":1795841003}]}
//   >> {"id":10,"method":"mining.submit","params":["neb1….rig01","a737f3","<extranonce2 hex>",1795840703,30817]}
//   << {"id":10,"result":true,"error":null}
//
// That exchange is rabidmining's (neb.rabidmining.com:3470), recorded on
// 28 Nov 2026. Hex is in the chain's byte order, nothing reversed. Targets are
// 32 bytes read as little-endian integers. Errors are [code, "message"]: 21
// job not found or expired, 22 duplicate, 23 ticket above the pool target, 24
// login not authorized, 25 not subscribed, 26 time outside the job's window,
// 27 extranonce2 of the wrong length.
//
// Pure: it builds request lines and classifies response lines. The socket is
// main/nebEngine.js's.

const { encode, normalizeError } = require('../miner/stratum');
const { errorText } = require('../xmr/stratum');

const SUBSCRIBE_ID = 1;
const AUTHORIZE_ID = 2;
const PROTOCOL = 'neb-pool/1';
// The longest line is mining.set_model. The spec allows 262,144 bytes.
const MAX_LINE = 262144;

function isHex(s, bytes) {
  return typeof s === 'string' && /^[0-9a-f]*$/.test(s) && s.length % 2 === 0 && (bytes == null || s.length === bytes * 2);
}

const hex = (v) => (typeof v === 'string' ? v.trim().toLowerCase() : '');
const isU32 = (v) => Number.isInteger(v) && v >= 0 && v <= 0xffffffff;

function buildSubscribe(agent) {
  return { id: SUBSCRIBE_ID, method: 'mining.subscribe', params: [agent || 'llmjob-earn', PROTOCOL] };
}

// "address.worker": a worker name is at most 32 characters of [A-Za-z0-9_-].
function loginOf(address, worker) {
  const name = String(worker == null ? '' : worker).trim().replace(/[^A-Za-z0-9_-]/g, '-').slice(0, 32);
  const wallet = String(address == null ? '' : address).trim();
  return name ? wallet + '.' + name : wallet;
}

function buildAuthorize(login) {
  return { id: AUTHORIZE_ID, method: 'mining.authorize', params: [login, 'x'] };
}

function buildSubmit(id, login, share) {
  return { id, method: 'mining.submit', params: [login, share.jobId, share.extranonce2, share.time, share.nonce] };
}

function parseSubscribe(r) {
  if (!r || typeof r !== 'object') return null;
  const extranonce1 = hex(r.extranonce1);
  const size = r.extranonce2_size;
  const datasetHash = hex(r.dataset_hash);
  if (!isHex(extranonce1) || !extranonce1) return null;
  if (!Number.isInteger(size) || size < 1 || size > 64) return null;
  if (!isHex(datasetHash, 32)) return null;
  return {
    kind: 'subscribed', session: String(r.session == null ? '' : r.session),
    extranonce1: Buffer.from(extranonce1, 'hex'), extranonce2Size: size,
    chainId: hex(r.chain_id), datasetHash,
  };
}

// A job this miner can build a header from, or { kind: 'bad-job', reason }.
function parseJob(p) {
  const jobId = p && p.job_id != null ? String(p.job_id) : '';
  const bad = (reason) => ({ kind: 'bad-job', jobId: jobId || null, reason });
  if (!p || typeof p !== 'object' || Array.isArray(p)) return bad('the job is not an object');
  if (!jobId) return bad('no job_id');
  for (const k of ['version', 'time_start', 'time_window', 'bits', 'tx_count']) {
    if (!isU32(p[k])) return bad(k + ' is not a 32-bit number');
  }
  for (const k of ['parent', 'target', 'shares_root', 'weights_hash']) {
    if (!isHex(hex(p[k]), 32)) return bad(k + ' is not 32 bytes of hex');
  }
  for (const k of ['coinbase_prefix', 'coinbase_suffix']) {
    if (!isHex(hex(p[k]))) return bad(k + ' is not hex');
  }
  if (!Array.isArray(p.branch) || !p.branch.every((b) => isHex(hex(b), 32))) return bad('branch is not a list of 32-byte values');
  if (!Number.isInteger(p.slot) || p.slot < 0 || p.slot > 255) return bad('slot is not a dataset index');
  const buf = (k) => Buffer.from(hex(p[k]), 'hex');
  return {
    kind: 'job',
    jobId,
    clean: p.clean === true,
    version: p.version,
    parent: buf('parent'),
    timeStart: p.time_start,
    timeWindow: p.time_window,
    bits: p.bits,
    blockTarget: buf('target'),
    shareTarget: isHex(hex(p.share_target), 32) ? buf('share_target') : null,
    coinbasePrefix: buf('coinbase_prefix'),
    coinbaseSuffix: buf('coinbase_suffix'),
    branch: p.branch.map((b) => Buffer.from(hex(b), 'hex')),
    txCount: p.tx_count,
    sharesRoot: buf('shares_root'),
    height: Number.isFinite(p.height) ? p.height : null,
    slot: p.slot,
    weightsHash: hex(p.weights_hash),
    expiry: Number.isFinite(p.expiry) ? p.expiry : null,
  };
}

// Classify one line from the pool. Never throws.
function parseMessage(line) {
  const s = String(line == null ? '' : line).trim();
  if (!s) return { kind: 'empty' };
  if (s.length > MAX_LINE) return { kind: 'unparseable', raw: s.slice(0, 200) };
  let msg;
  try {
    msg = JSON.parse(s);
  } catch (e) {
    return { kind: 'unparseable', raw: s.slice(0, 200) };
  }
  if (!msg || typeof msg !== 'object' || Array.isArray(msg)) return { kind: 'unknown', raw: s.slice(0, 200) };
  const params = Array.isArray(msg.params) ? msg.params : [];
  switch (msg.method) {
    case 'mining.set_target': {
      const t = hex(params[0]);
      return isHex(t, 32) ? { kind: 'target', target: Buffer.from(t, 'hex') } : { kind: 'unknown', raw: s.slice(0, 200) };
    }
    case 'mining.set_model': {
      const hash = hex(params[1]);
      const weights = hex(params[2]);
      if (!Number.isInteger(params[0]) || !isHex(hash, 32) || !isHex(weights) || !weights) {
        return { kind: 'bad-model', reason: 'a model that is not [slot, hash, weights]' };
      }
      return { kind: 'model', slot: params[0], weightsHash: hash, weights: Buffer.from(weights, 'hex') };
    }
    case 'mining.notify':
      return parseJob(params[0]);
    default:
      if (msg.method != null) return { kind: 'unknown', raw: s.slice(0, 200) };
  }
  if (!('result' in msg) && !('error' in msg)) return { kind: 'unknown', raw: s.slice(0, 200) };
  const error = normalizeError(msg.error);
  if (msg.id === SUBSCRIBE_ID) {
    const sub = error ? null : parseSubscribe(msg.result);
    return sub || { kind: 'subscribe-fail', error: error || { code: null, message: 'the pool\'s answer names no extranonce or dataset' } };
  }
  if (msg.id === AUTHORIZE_ID) {
    return !error && msg.result === true ? { kind: 'authorized' }
      : { kind: 'login-fail', error: error || { code: null, message: 'the pool did not authorize the login' } };
  }
  if (msg.id == null) return error ? { kind: 'pool-error', error } : { kind: 'unknown', raw: s.slice(0, 200) };
  if (error) return { kind: 'submit-rejected', id: msg.id, error };
  if (msg.result === true) return { kind: 'submit-accepted', id: msg.id };
  return { kind: 'submit-rejected', id: msg.id, error: { code: null, message: 'no reason given (result ' + String(JSON.stringify(msg.result)).slice(0, 80) + ')' } };
}

module.exports = {
  SUBSCRIBE_ID, AUTHORIZE_ID, PROTOCOL, MAX_LINE,
  buildSubscribe, buildAuthorize, buildSubmit, loginOf, parseJob, parseMessage, encode, errorText,
};
