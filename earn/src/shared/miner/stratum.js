'use strict';

// The Pearl Stratum protocol as HeroMiners actually speaks it — every shape here
// was captured off the live wire (us.pearl.herominers.com:1200, 2026-08-23), not
// read off a doc. Two findings that a from-the-spec implementation would get
// wrong, and that cost the reference miners a round of trial and error:
//
//   • authorize takes OBJECT params {wallet, worker}. The array form
//     ["wallet.worker","x"] is rejected with {code:20,"params must be an
//     object"}, and mining.configure / mining.subscribe are rejected outright
//     with "Undefined stratum" — this pool has no handshake before authorize.
//     So the whole handshake is a single authorize message.
//
//   • mining.notify params are an OBJECT, not the positional array of classic
//     Stratum: {job_id, header(76B hex), target(32B hex), height, cert_version}.
//
// This module is pure: it builds request lines and classifies response lines,
// and never touches a socket. The socket lifecycle lives in main/pearlMiner.js
// so this stays unit-testable against the captured fixtures.

// Live capture, verbatim, used as the golden fixture in the tests:
//   >> {"id":1,"method":"mining.authorize","params":{"wallet":"prl1p…","worker":"sniff"}}
//   << {"id":1,"error":null,"result":true}
//   << {"id":null,"method":"mining.notify","params":{
//        "job_id":"00000000_2097152",
//        "header":"000000203a49…b7b5d155918b6a0dea0018",   (76 bytes = 152 hex)
//        "target":"00000000000007fff800…0000",             (32 bytes = 64 hex)
//        "height":103353,"cert_version":3}}

const HEADER_BYTES = 76;
const TARGET_BYTES = 32;

// The id we send authorize under. Fixed, because there is exactly one request in
// this protocol whose result we wait for; jobs and difficulty arrive unsolicited
// with id null, and submits carry their own ids allocated by the caller.
const AUTHORIZE_ID = 1;

function buildAuthorize(wallet, worker) {
  return { id: AUTHORIZE_ID, method: 'mining.authorize', params: { wallet, worker } };
}

// A share submission. The proof encoding is pool-specific and is the one part of
// this protocol NOT yet confirmed against a HeroMiners-accepted share — see
// pearlhash.js for what the core produces. The fields below are the documented
// v2 shape (job_id, the two seeds, the nonce, and the base64 proof the core
// emits); `serializeProof` is isolated as its own function so the exact wire
// encoding can be corrected in one place once a real submit is observed.
function buildSubmit(id, share) {
  return {
    id,
    method: 'mining.submit',
    params: {
      wallet: share.wallet,
      worker: share.worker,
      job_id: share.jobId,
      nonce: share.nonce,
      type: 'v2',
      sigma: share.aSeed,
      b_seed: share.bSeed,
      plain_proof: serializeProof(share.proof),
    },
  };
}

// The proof bytes the core returns, encoded for the wire. Base64 of the raw
// bytes is the documented v2 encoding; kept separate so a pool that wants a
// different container (Kryptex uses protobuf) is a one-function change.
function serializeProof(proof) {
  const buf = Buffer.isBuffer(proof) ? proof : Buffer.from(proof || []);
  return buf.toString('base64');
}

// Serialize a request object to a newline-terminated JSON line, as the pool
// expects (one JSON value per line, '\n'-delimited).
function encode(msg) {
  return JSON.stringify(msg) + '\n';
}

// A big-endian uint256 target (the pool's 64-hex string) as a BigInt. This is
// the numeric threshold a candidate must come in at or under. Rejects anything
// that is not exactly 32 bytes of hex, so a truncated or malformed job fails
// loudly here rather than silently comparing against a wrong bound.
function decodeTarget(hex) {
  const s = String(hex == null ? '' : hex).trim().toLowerCase();
  if (!/^[0-9a-f]{64}$/.test(s)) return null;
  return BigInt('0x' + s);
}

// Classify one response line. Returns a tagged object the host can switch on, or
// { kind: 'unknown' } for anything unrecognised (kept, not dropped, so the host
// can still log it). Never throws on malformed JSON — a pool that sends a junk
// line must not crash the miner.
function parseMessage(line) {
  const s = String(line == null ? '' : line).trim();
  if (!s) return { kind: 'empty' };
  let msg;
  try {
    msg = JSON.parse(s);
  } catch (e) {
    return { kind: 'unparseable', raw: s };
  }
  if (!msg || typeof msg !== 'object') return { kind: 'unknown', raw: s };

  if (msg.method === 'mining.notify') return parseJob(msg.params);
  if (msg.method === 'mining.set_difficulty') return parseDifficulty(msg.params);

  // A result carries no method — it answers a request id. authorize (id 1) and
  // submits (ids the caller allocated) both come back this way.
  if (msg.method == null && ('result' in msg || 'error' in msg)) {
    const failed = msg.error != null;
    if (msg.id === AUTHORIZE_ID) {
      return { kind: failed ? 'auth-fail' : 'auth-ok', id: msg.id, error: normalizeError(msg.error) };
    }
    return {
      kind: failed ? 'submit-rejected' : 'submit-accepted',
      id: msg.id,
      error: normalizeError(msg.error),
    };
  }
  return { kind: 'unknown', raw: s };
}

// The pool reports errors in two shapes seen live: a bare object
// {code, msg} on a bad authorize, and (per the reference notes) a [code, "text"]
// array on a bad submit. Reduce both to { code, message } so the host has one
// thing to log.
function normalizeError(err) {
  if (err == null) return null;
  if (Array.isArray(err)) return { code: err[0] == null ? null : err[0], message: String(err[1] == null ? '' : err[1]) };
  if (typeof err === 'object') {
    const message = err.msg != null ? err.msg : (err.message != null ? err.message : '');
    return { code: err.code == null ? null : err.code, message: String(message) };
  }
  return { code: null, message: String(err) };
}

// Turn a mining.notify params object into a job the core can search. Validates
// the two fixed-width hex fields — a job with a short header or a malformed
// target is unusable and must be rejected here, not fed to the GPU.
function parseJob(params) {
  const p = params && typeof params === 'object' && !Array.isArray(params) ? params : {};
  const headerHex = String(p.header == null ? '' : p.header).trim().toLowerCase();
  const target = decodeTarget(p.target);
  const validHeader = /^[0-9a-f]+$/.test(headerHex) && headerHex.length === HEADER_BYTES * 2;
  if (!validHeader || target == null || !p.job_id) {
    return { kind: 'bad-job', jobId: p.job_id || null };
  }
  return {
    kind: 'job',
    jobId: String(p.job_id),
    header: Buffer.from(headerHex, 'hex'),
    headerHex,
    target,
    targetHex: String(p.target).trim().toLowerCase(),
    height: Number.isFinite(p.height) ? p.height : null,
    certVersion: Number.isFinite(p.cert_version) ? p.cert_version : null,
    // The rank the pool is paying for, when it states one. HeroMiners does not
    // send this today (the job id's trailing number is the difficulty factor,
    // not the rank), so it is usually null and the host trusts the pool. It is
    // read here so that a pool which DOES state a rank can be refused before we
    // spend a GPU on work the fork will not credit — the exact failure that made
    // alpha-miner's Ada builds mine rank 512 for nothing.
    rank: Number.isFinite(p.rank) ? p.rank : null,
  };
}

// mining.set_difficulty. HeroMiners vardiffs, so this arrives to widen or narrow
// the share target independently of the job. Params observed as a bare number in
// classic Stratum; accept a number or a {difficulty} object defensively.
function parseDifficulty(params) {
  let d = null;
  if (typeof params === 'number') d = params;
  else if (Array.isArray(params) && typeof params[0] === 'number') d = params[0];
  else if (params && typeof params === 'object' && typeof params.difficulty === 'number') d = params.difficulty;
  return { kind: 'difficulty', difficulty: d };
}

module.exports = {
  HEADER_BYTES,
  TARGET_BYTES,
  AUTHORIZE_ID,
  buildAuthorize,
  buildSubmit,
  serializeProof,
  encode,
  decodeTarget,
  parseMessage,
  parseJob,
  parseDifficulty,
  normalizeError,
};
