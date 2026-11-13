'use strict';

// Quantus (QTC) pool stratum: JSON-RPC lines with login / job / submit /
// keepalived objects, the Monero-style shape (../xmr/stratum.js) with Quantus's
// job fields. SRBMiner calls it "LuckyPool stratum"; since SRBMiner 3.7.3
// dropped the QUIC protocol it is the only one the GPU miners speak.
//
// Where each shape comes from:
//   - LuckyPool's login reply and job notification, recorded and published as
//     test data by not-only-mining-pool (github.com/mining-pool/
//     not-only-mining-pool, engine/quantus/testdata, Apache-2.0), copied into
//     test/quantusStratum.test.js;
//   - the submit and keepalive request and reply, from that pool's own
//     stratum test (engine/quantus/stratum_test.go);
//   - Kryptex (qtc-us.kryptex.network:7049) and HeroMiners
//     (us2.quantus.herominers.com:1155) both answer a "login" with a JSON-RPC
//     reply. Asked with a login that is not an address, 13 Nov 2026:
//       << {"id":1,"result":null,"error":{"code":-1,"message":"Invalid login"}}            Kryptex
//       << {"id":1,"error":{"code":-20,"message":"Invalid Quantus address"},"result":null}  HeroMiners
//     and closed the connection. No job from either has been seen yet: that
//     needs a real address.
//
//   >> {"id":1,"jsonrpc":"2.0","method":"login","params":{"login":"qz…","pass":"rig01","agent":"llmjob-earn/…","rigid":"rig01"}}
//   << {"id":1,"result":{"id":"<session>","status":"OK","extensions":["keepalive"],"job":{
//        "algo":"qpow-poseidon2","difficulty":10000000000,"extranonce":"0004ee45","job_id":"22729-10000000000",
//        "mining_hash":"<64 hex>","seq":22729,"target":"<128 hex>"}}}
//   << {"jsonrpc":"2.0","method":"job","params":{"clean_jobs":true,"job":{…the same fields…}}}
//   >> {"id":2,"jsonrpc":"2.0","method":"submit","params":{"id":"<session>","job_id":"…","nonce":"<128 hex>"}}
//   << {"id":2,"result":{"status":"OK"},"error":null}
//   >> {"id":3,"jsonrpc":"2.0","method":"keepalived","params":{"id":"<session>"}}
//   << {"id":3,"result":{"status":"KEEPALIVED"}}
//
// The nonce must start with the job's extranonce. `target` is the full 512-bit
// target, big-endian; `difficulty` is the same thing as a number (target =
// (2^512 - 1) / difficulty). A job with only one of them is taken too.
//
// Pure: it builds request lines and classifies response lines. The socket is
// main/quantusEngine.js's.

const { isKryptexHost } = require('../miner/stratum');
const { normalizeError, errorText, encode } = require('../xmr/stratum');
const { targetFromDifficulty, difficultyFromTarget, MAX_TARGET } = require('./qpow');

const LOGIN_ID = 1;
// The names pools give the algorithm. LuckyPool's is the first; a job naming
// another algorithm is not mined.
const ALGOS = ['qpow-poseidon2', 'qpow', 'quantus', 'poseidon2'];
const MAX_EXTRANONCE_BYTES = 32;

// The worker name goes in `pass` and `rigid`, as for Monero. Kryptex names the
// worker from "address/worker", as it does for Pearl and Monero; that it does
// so for Quantus too has not been seen yet (no login has been accepted).
function buildLogin({ address, worker, agent, host }) {
  const name = String(worker == null ? '' : worker).trim() || 'x';
  const wallet = String(address == null ? '' : address).trim();
  return {
    id: LOGIN_ID,
    jsonrpc: '2.0',
    method: 'login',
    params: { login: isKryptexHost(host) ? wallet + '/' + name : wallet, pass: name, agent: agent || 'llmjob-earn', rigid: name },
  };
}

function buildSubmit(id, sessionId, share) {
  return { id, jsonrpc: '2.0', method: 'submit', params: { id: sessionId, job_id: share.jobId, nonce: share.nonce } };
}

function buildKeepalive(id, sessionId) {
  return { id, jsonrpc: '2.0', method: 'keepalived', params: { id: sessionId } };
}

function isHex(s, bytes) {
  return typeof s === 'string' && /^[0-9a-fA-F]*$/.test(s) && s.length % 2 === 0 && (bytes == null || s.length === bytes * 2);
}

// A difficulty as a BigInt: a whole number, or a string of digits. Null for
// anything else, and for 0.
function readDifficulty(d) {
  let n = null;
  if (typeof d === 'number' && Number.isSafeInteger(d)) n = BigInt(d);
  else if (typeof d === 'string' && /^[0-9]+$/.test(d.trim())) n = BigInt(d.trim());
  return n != null && n > 0n ? n : null;
}

// A job, from a login reply or a "job" notification: { kind: 'job', … }, or
// { kind: 'bad-job', jobId, reason } for one this miner cannot mine.
// LuckyPool wraps it as params.job; a job's own fields at the top level work too.
function parseJob(params) {
  const outer = params && typeof params === 'object' && !Array.isArray(params) ? params : {};
  const p = outer.job && typeof outer.job === 'object' && !Array.isArray(outer.job) ? outer.job : outer;
  const jobId = p.job_id == null ? '' : String(p.job_id);
  const bad = (reason) => ({ kind: 'bad-job', jobId: jobId || null, reason });
  if (!jobId) return bad('no job_id');
  const hash = typeof p.mining_hash === 'string' ? p.mining_hash.trim().toLowerCase() : '';
  if (!isHex(hash, 32)) return bad('mining_hash is not 32 bytes of hex');
  const algo = p.algo == null || p.algo === '' ? null : String(p.algo).toLowerCase();
  if (algo && !ALGOS.includes(algo)) return bad('algorithm ' + algo + ' is not one this miner mines');
  const extranonce = p.extranonce == null ? '' : String(p.extranonce).trim().toLowerCase();
  if (!isHex(extranonce) || extranonce.length > MAX_EXTRANONCE_BYTES * 2) {
    return bad('extranonce is not hex of ' + MAX_EXTRANONCE_BYTES + ' bytes or fewer');
  }
  let target = null;
  if (p.target != null) {
    const t = String(p.target).trim().toLowerCase();
    if (!isHex(t, 64)) return bad('target is not 64 bytes of hex');
    target = BigInt('0x' + t);
    if (target === 0n) return bad('the target is 0');
  }
  const difficulty = readDifficulty(p.difficulty);
  if (p.difficulty != null && difficulty == null) return bad('unreadable difficulty ' + JSON.stringify(p.difficulty));
  if (target == null && difficulty == null) return bad('no target and no difficulty');
  if (target == null) target = targetFromDifficulty(difficulty);
  return {
    kind: 'job',
    jobId,
    miningHash: Buffer.from(hash, 'hex'),
    target,
    difficulty: difficulty != null ? difficulty : difficultyFromTarget(target),
    extranonce: Buffer.from(extranonce, 'hex'),
    seq: Number.isFinite(p.seq) ? p.seq : null,
    height: Number.isFinite(p.height) ? p.height : null,
    algo,
  };
}

// Classify one line from the pool. Never throws.
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
  const r = msg.result;
  if (msg.id === LOGIN_ID) {
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
  if (msg.id == null) return error ? { kind: 'pool-error', error } : { kind: 'unknown', raw: s };
  if (keepaliveIds && keepaliveIds.has(msg.id)) return { kind: 'keepalive', id: msg.id, error };
  if (error) return { kind: 'submit-rejected', id: msg.id, error };
  // Accepted is {"status":"OK"}, or a bare true. Anything else did not count.
  if (r === true || (r && typeof r === 'object' && String(r.status).toUpperCase() === 'OK')) {
    return { kind: 'submit-accepted', id: msg.id };
  }
  return {
    kind: 'submit-rejected', id: msg.id,
    error: { code: null, message: 'no reason given (result ' + String(JSON.stringify(r)).slice(0, 80) + ')' },
  };
}

module.exports = {
  LOGIN_ID, ALGOS, MAX_EXTRANONCE_BYTES, MAX_TARGET,
  buildLogin, buildSubmit, buildKeepalive, encode, parseJob, parseMessage, readDifficulty, errorText,
};
