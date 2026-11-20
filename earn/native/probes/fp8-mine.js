// node fp8-mine.js <pearl_core_fp8.node> [--socket PATH] [--m 4096 --n 32768 --k 2048 --h 4]
//                  [--device 0] [--secs 0] [--bench [--nbits HEX]]
//
// A solo FP8 miner for testing the core (earn/native/fp8) end to end. It is a tool, not the
// app's miner: it needs PEARL_FP8=1, and it talks to Pearl's own gateway (pearl-gateway on
// the fp8-scheme branch), which turns a plain proof into a block and hands it to pearld.
//
//   gateway   line-delimited JSON-RPC on a Unix socket: getMiningInfo, submitPlainProof
//   job       the 76-byte proposed header keys A; the parent's 108-byte header keys B
//   attempt   the header has no nonce, so each attempt swaps two values of A's first chunk.
//             That changes A's commitment, so seedA, so A's E noise lines and the ticket key.
//             B, its noise and both F bases stay as they are until the header changes.
//             The card samples the lines, noises A and searches; the host hashes one
//             Merkle path and the seeds.
//   hit       the card names the tile; fp8Host builds the PlainProofV4 and replays it on the
//             CPU reference (policy included) before anything is sent
//
// --bench runs without a gateway on a made-up header and a bound no tile meets, to time it.
// Prints a line a job and a rate line every few seconds; the rate is m.n.k a second, the
// unit Pearl's miner credits.
'use strict';
const net = require('net');
const path = require('path');
const R = path.join(__dirname, '..', '..', 'src', 'shared', 'miner') + path.sep;
const fp8 = require(R + 'fp8');
const host = require(R + 'fp8Host');

if (!fp8.enabled(process.env)) {
  console.error('fp8-mine: set PEARL_FP8=1 to run the FP8 tools');
  process.exit(2);
}

const args = process.argv.slice(3);
const opt = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i < 0 ? fallback : args[i + 1];
};
const M = Number(opt('m', 4096));
const N = Number(opt('n', 32768));
const K = Number(opt('k', 2048));
const H = Number(opt('h', 4));
const W = 256 / H;
const SECS = Number(opt('secs', 0));
const BENCH = args.includes('--bench');
const SOCKET = opt('socket', '/tmp/pearlgw.sock');
const B200 = fp8.DEVICE_B200;

const core = require(path.resolve(process.argv[2]));
const card = core.open(Number(opt('device', 0)));
core.configure(M, N, K, H, 64);
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
log(`card ${card.name} (compute ${card.major}.${card.minor}); A ${M} x ${K}, B ${N} x ${K}, tile ${H} x ${W}, ${(M * N * K / 1e12).toFixed(1)} T-MAC an attempt`);

// --- the gateway ---

function connect() {
  return new Promise((resolve, reject) => {
    const sock = net.connect(SOCKET);
    let buf = '';
    const waiting = [];
    sock.on('data', (d) => {
      buf += d;
      for (let at = buf.indexOf('\n'); at >= 0; at = buf.indexOf('\n')) {
        waiting.shift()(buf.slice(0, at));
        buf = buf.slice(at + 1);
      }
    });
    sock.on('error', reject);
    sock.on('connect', () => resolve({
      call: (line) => new Promise((done) => { waiting.push(done); sock.write(line + '\n'); }),
      close: () => sock.destroy(),
    }));
  });
}

// The job as the gateway wrote it. Its target is an integer too large for a JS number, so
// the text is kept and sent back untouched with a proof.
async function getJob(gw) {
  const line = await gw.call('{"jsonrpc":"2.0","method":"getMiningInfo","params":{},"id":1}');
  const from = line.indexOf('"result":');
  const to = line.lastIndexOf(', "id"');
  if (from < 0 || to < 0) return null;
  const raw = line.slice(from + 9, to).trim();
  const header = /"incomplete_header_bytes":\s*"([^"]*)"/.exec(raw);
  const parent = /"ancestor_headers":\s*\["([^"]*)"/.exec(raw);
  if (!header || !parent) return null;
  return { raw, proposedHeader: Buffer.from(header[1], 'base64'), ancestorHeader: Buffer.from(parent[1], 'base64') };
}

function benchJob() {
  const proposedHeader = Buffer.alloc(host.HEADER_LEN, 7);
  proposedHeader.writeUInt32LE(parseInt(opt('nbits', '03000001'), 16), 72);  // by default a target of 1: nothing wins
  return { raw: null, proposedHeader, ancestorHeader: Buffer.alloc(host.ANCESTOR_LEN, 9) };
}

// --- the operands: made once, policy-passing, unit scales ---

const job = host.makeJob({ m: M, n: N, k: K, h: H, w: W });
const aInts = host.fillValues(Date.now() & 0x7FFFFFFF, M, K);
const bInts = host.fillValues((Date.now() >>> 3) ^ 0x5bd1e995, N, K);
const aScales = host.unitScales(M, K);
const bScales = host.unitScales(N, K);
const aParams = host.rowParams(B200, aInts, aScales, M, K);
const bParams = host.rowParams(B200, bInts, bScales, N, K);
core.load(0, aInts, aParams.signal, aParams.beta, aParams.rewrite);
core.load(1, bInts, bParams.signal, bParams.beta, bParams.rewrite);
const tree = host.coreTree(core);
const firstChunk = aInts.subarray(0, 1024);
const next = host.rng(Date.now() ^ 0x1234567);

// Swap two unequal values in the first 1024 values of A's row 0: one leaf of the values
// tree. A row's norms do not change under a swap while its scales are all one, so its
// noise numbers stand.
function tweakA() {
  for (;;) {
    const i = next() % 1024;
    const j = next() % 1024;
    if (aInts[i] === aInts[j]) continue;
    const v = aInts[i];
    aInts[i] = aInts[j];
    aInts[j] = v;
    return;
  }
}

const stats = { attempts: 0, hits: 0, void: 0, sent: 0, t0: Date.now(), host: 0, noise: 0, search: 0 };

function report() {
  const secs = (Date.now() - stats.t0) / 1000;
  const rate = stats.attempts * M * N * K / secs / 1e12;
  log(`${stats.attempts} attempts, ${rate.toFixed(1)} T-MAC/s; an attempt: host ${(stats.host / stats.attempts).toFixed(2)} ms, noise ${(stats.noise / stats.attempts).toFixed(2)} ms, search ${(stats.search / stats.attempts).toFixed(2)} ms; hits ${stats.hits}, void ${stats.void}, sent ${stats.sent}`);
}

async function main() {
  const gw = BENCH ? null : await connect();
  let current = null;
  let a = null;
  let b = null;
  let lastReport = Date.now();
  let id = 2;
  const end = SECS ? Date.now() + SECS * 1000 : Infinity;
  while (Date.now() < end) {
    const got = BENCH ? (current || benchJob()) : await getJob(gw);
    if (!got) {
      log('no job from the gateway; waiting');
      await new Promise((r) => setTimeout(r, 2000));
      continue;
    }
    if (!current || !got.proposedHeader.equals(current.proposedHeader)) {
      const t = Date.now();
      current = got;
      a = host.commitOperand(fp8.keyA(got.proposedHeader), aInts, aScales, tree);
      b = host.commitOperand(fp8.keyB(got.ancestorHeader), bInts, bScales, tree);
      const seeds = host.attemptSeeds({ proposedHeader: got.proposedHeader, ancestorHeader: got.ancestorHeader, job, hashA: a.digest, hashB: b.digest });
      // B's noise and A's F base hang off seedB alone, so they last the whole job.
      const lineKeyB = fp8.noiseLineKey(seeds.seedB);
      core.seedLines(1, fp8.FACTOR_E, lineKeyB);
      core.seedLines(1, fp8.FACTOR_F, lineKeyB);
      core.renoise(1);
      core.seedLines(0, fp8.FACTOR_F, lineKeyB);
      log(`new job: prev ${Buffer.from(got.proposedHeader.subarray(4, 36)).reverse().toString('hex').slice(0, 16)}, nbits ${got.proposedHeader.readUInt32LE(72).toString(16)}, set up in ${Date.now() - t} ms`);
    }

    let t = performance.now();
    tweakA();
    host.updateLeaf(a.key, a.valuesLayers, a.values, 0);
    a.digest = host.digestOf(a.key, a.valuesLayers, a.scalesLayers);
    const seeds = host.attemptSeeds({ proposedHeader: current.proposedHeader, ancestorHeader: current.ancestorHeader, job, hashA: a.digest, hashB: b.digest });
    stats.host += performance.now() - t;
    t = performance.now();
    core.patch(0, 0, firstChunk);
    core.seedLines(0, fp8.FACTOR_E, fp8.noiseLineKey(seeds.seedA));
    core.renoise(0);
    stats.noise += performance.now() - t;
    t = performance.now();
    const found = core.search(seeds.jackpotKey, seeds.bound, false);
    stats.search += performance.now() - t;
    stats.attempts += 1;

    for (let i = 0; i < found.hits.length; i += 2) {
      stats.hits += 1;
      const proof = host.buildProof({ ancestorHeader: current.ancestorHeader, job, a, b, tileA: found.hits[i], tileB: found.hits[i + 1] });
      let checked;
      try {
        checked = host.verifyProof(current.proposedHeader, proof);
      } catch (e) {
        stats.void += 1;
        log(`tile (${found.hits[i]}, ${found.hits[i + 1]}) is void: ${e.message}`);
        continue;
      }
      log(`tile (${found.hits[i]}, ${found.hits[i + 1]}) ticket ${checked.ticket.toString('hex')} wins on the CPU reference: ${checked.wins}`);
      if (!checked.wins) {
        stats.void += 1;
        continue;
      }
      const plain = host.serializeProof(proof).toString('base64');
      if (!gw) continue;
      const reply = await gw.call(`{"jsonrpc":"2.0","method":"submitPlainProof","params":{"plain_proof":"${plain}","mining_job":${current.raw}},"id":${id++}}`);
      stats.sent += 1;
      log(`sent a ${plain.length}-character proof; the gateway said ${reply.slice(0, 200)}`);
    }
    if (Date.now() - lastReport > 10000) {
      report();
      lastReport = Date.now();
    }
    await new Promise((r) => setImmediate(r));
  }
  report();
  core.close();
  if (gw) gw.close();
}

main().catch((e) => {
  console.error('fp8-mine failed:', e);
  process.exit(1);
});
