'use strict';

// The 80-byte NEB header of an attempt, from a pool job (spec/pool.md section 5
// of github.com/SilasEigen/neb):
//
//   coinbase  = coinbase_prefix || extranonce1 || extranonce2 || coinbase_suffix
//   h         = H("leaf", BLAKE3(coinbase)); for each s in branch: h = H("node", h, s)
//   body_root = H("body", LE32(tx_count), h, shares_root)
//   header    = LE32(version) || parent || body_root || LE32(time) || LE32(bits) || LE32(nonce)
//   key       = BLAKE3(header)
//
// H is BLAKE3 over the tag's ASCII bytes and the parts, with no separator.
// The nonce is bytes 76 to 79, so a card rolls the last word of the header's
// second BLAKE3 block.

const blake3 = require('../miner/blake3');

const HEADER_BYTES = 80;
const NONCE_OFFSET = 76;

function tagged(tag, ...parts) {
  return blake3.hash(Buffer.concat([Buffer.from(tag, 'ascii'), ...parts]));
}

function le32(v) {
  const b = Buffer.alloc(4);
  b.writeUInt32LE(v >>> 0);
  return b;
}

function bodyRoot(job, extranonce1, extranonce2) {
  const coinbase = Buffer.concat([job.coinbasePrefix, extranonce1, extranonce2, job.coinbaseSuffix]);
  let h = tagged('leaf', blake3.hash(coinbase));
  for (const s of job.branch) h = tagged('node', h, s);
  return tagged('body', le32(job.txCount), h, job.sharesRoot);
}

// The header with a zero nonce.
function buildHeader(job, extranonce1, extranonce2, time) {
  return Buffer.concat([
    le32(job.version), job.parent, bodyRoot(job, extranonce1, extranonce2), le32(time), le32(job.bits), le32(0),
  ]);
}

function withNonce(header, nonce) {
  const h = Buffer.from(header);
  h.writeUInt32LE(nonce >>> 0, NONCE_OFFSET);
  return h;
}

// The key of the attempt a header and a nonce make.
function attemptKey(header, nonce) {
  return blake3.hash(withNonce(header, nonce));
}

module.exports = { HEADER_BYTES, NONCE_OFFSET, tagged, bodyRoot, buildHeader, withNonce, attemptKey };
