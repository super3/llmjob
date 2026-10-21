'use strict';

// The Pearl Stratum protocol as HeroMiners actually speaks it — every shape here
// was captured off the live wire (us.pearl.herominers.com:1200, 2026-08-23), not
// read off a doc. The one exception is the compressed proofs at the end, which
// follow Kryptex's spec and say so. Two findings that a from-the-spec
// implementation would get wrong, and that cost the reference miners a round of
// trial and error:
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
//   >> {"id":100,"method":"mining.submit","params":{"job_id":…,"plain_proof":…,"hs":…}}
//   << {"id":100,"error":null,"result":true}
// Kryptex (prl-us.kryptex.network:7048) answers the same, with the keys in the
// other order: {"id":100,"result":true,"error":null}. Both captured 2026-10-17.
//
// Compressed proofs ("type":"v2"), from Kryptex's "Pearl stratum gzip
// protocol" (gist.github.com/maxmalysh/eaaf4332dbc5ca99d0a78f24a733fffe) and
// not from a capture: a miner that adds "type":"v2" to authorize, and gets
// "type":"v2" back on the reply, sends plain_proof as base64(gzip(bincode))
// instead of base64(bincode). Kryptex shows "Implement proper gzip support" on
// the address page of a miner that does not. Only the login was tried live, on
// both pools on 2026-10-20, with no shares sent. The request is the shape we
// send, not a copy of the probe; the replies are verbatim:
//   >> {"id":1,"method":"mining.authorize","params":{"wallet":…,"worker":…,
//        "agent":"llmjob-earn/…","type":"v2"}}
//   << {"id":1,"result":true,"error":null,"type":"v2"}     Kryptex
//   << {"id":1,"error":null,"result":true,"type":"v2"}     HeroMiners
// and Kryptex asked without the type: {"id":1,"result":true,"error":null}.
// The rest is the spec's word, not yet seen on the wire: the gzipped submit,
// and that the reply decides, not the request. Once a pool has said v2 it
// unpacks every submit on that connection, so a plain proof there is an
// invalid share, and a reply without it means plain proofs. So the choice
// belongs to one connection and is made again on the next one (see
// main/pearlMiner.js).

const HEADER_BYTES = 76;

// The id we send authorize under. Fixed, because there is exactly one request in
// this protocol whose result we wait for; jobs and difficulty arrive unsolicited
// with id null, and submits carry their own ids allocated by the caller.
const AUTHORIZE_ID = 1;

// The authorize type that asks for compressed proofs, and the one a pool's
// reply carries when it agrees (see the captures above).
const GZIP_TYPE = 'v2';

// Kryptex (`--pool prl-us.kryptex.network:7048`) takes the same message, but
// names the worker from "wallet/worker" and ignores the worker field: a login
// that sent only the field showed up there as "worker". Only Kryptex gets the
// slash. HeroMiners' wallet field stays exactly as captured, and another pool
// gets the standard one rather than a guess.
function isKryptexHost(host) {
  return /(^|\.)kryptex\.network$/i.test(String(host || ''));
}

// Every pool is asked for compressed proofs: a pool that does not offer them
// answers without the type, and the proofs stay plain. `gzip: false` leaves the
// type out, for a pool that agreed and then rejected what we sent (see
// PearlMiner's GZIP_REJECT_LIMIT). `agent` names the miner to the pool, as
// Kryptex's documented message does; it is left out when not given.
function buildAuthorize(wallet, worker, host, { agent, gzip = true } = {}) {
  const login = worker && isKryptexHost(host) ? wallet + '/' + worker : wallet;
  const params = { wallet: login, worker };
  if (agent) params.agent = agent;
  if (gzip) params.type = GZIP_TYPE;
  return { id: AUTHORIZE_ID, method: 'mining.authorize', params };
}

// A share submission. The proof is the whole message: it carries m, n, k, the
// noise rank, and both Merkle proofs with their row indices, and the verifier
// reconstructs the mining configuration from those indices rather than being
// told it. So there is no nonce to send and no seeds to send -- the earlier
// shape here carried wallet, worker, nonce, sigma and b_seed, none of which the
// pool reads, and it never produced an accepted share.
//
// hs is the miner's own hashrate in multiply-accumulates per second, which is
// what the protocol's difficulty adjustment counts. It is advisory: the pool
// scores on shares.
function buildSubmit(id, share) {
  return {
    id,
    method: 'mining.submit',
    params: {
      job_id: share.jobId,
      plain_proof: share.plainProof,
      hs: Math.max(0, Math.round(share.hashrate || 0)),
    },
  };
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
      if (failed) return { kind: 'auth-fail', id: msg.id, error: normalizeError(msg.error) };
      // `gzip`: whether this connection's proofs must go compressed. Only the
      // exact type we ask for counts; any other type, or none, means plain.
      return { kind: 'auth-ok', id: msg.id, error: null, gzip: msg.type === GZIP_TYPE };
    }
    // A share counts only on result true, the answer both pools give. One with
    // no error and any other result, such as false, was not accepted: counting
    // it would put our figure above the pool's.
    if (!failed && msg.result === true) return { kind: 'submit-accepted', id: msg.id, error: null };
    return {
      kind: 'submit-rejected',
      id: msg.id,
      error: failed ? normalizeError(msg.error)
        : { code: null, message: 'no reason given (' + describeResult(msg) + ')' },
    };
  }
  return { kind: 'unknown', raw: s };
}

// What a reply with no error said instead of true, for the log.
function describeResult(msg) {
  if (!('result' in msg)) return 'no result';
  return 'result ' + String(JSON.stringify(msg.result)).slice(0, 80);
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
  AUTHORIZE_ID,
  GZIP_TYPE,
  buildAuthorize,
  isKryptexHost,
  buildSubmit,
  encode,
  decodeTarget,
  parseMessage,
  parseJob,
  parseDifficulty,
  normalizeError,
};
