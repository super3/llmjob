'use strict';

// The real side of rented containers (issue #318): docker, cosign, the GPU
// readings, the take-back file and the signed calls to our server, wired into
// a RentalRunner. Both shells build theirs here.
//
// LLMJOB_DOCKER and LLMJOB_COSIGN name other binaries (a test or a fake docker
// for an end-to-end check). LLMJOB_RENT_REQUIRE_SIGNATURE=1 refuses every
// image when cosign is not installed; without it an unsigned check is logged
// and the image runs, since we don't sign images in CI yet.

const { execFile, spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const nodeProto = require('../shared/node');
const R = require('../shared/rental');
const { postJson } = require('./io');
const probe = require('./probe');
const nodeStore = require('./nodeStore');
const { serverOf } = require('./nodeClient');
const { RentalRunner } = require('./rentalRunner');

// Pulling a large image can take a while on a home connection.
const DOCKER_TIMEOUT_MS = 30 * 60 * 1000;
// Who may sign our images: GitHub Actions in our repo, keyless.
const COSIGN_IDENTITY = '^https://github.com/super3/llmjob/';
const COSIGN_ISSUER = 'https://token.actions.githubusercontent.com';

const dockerBin = () => process.env.LLMJOB_DOCKER || 'docker';
const cosignBin = () => process.env.LLMJOB_COSIGN || 'cosign';

// Run a command and resolve { code, stdout, stderr }. Never rejects: a missing
// binary is code 127.
function exec(bin, args, env, timeoutMs = DOCKER_TIMEOUT_MS) {
  return new Promise((resolve) => {
    execFile(bin, args, { env: Object.assign({}, process.env, env), timeout: timeoutMs, maxBuffer: 16 * 1048576 },
      (err, stdout, stderr) => {
        const code = !err ? 0 : (Number.isInteger(err.code) ? err.code : 127);
        resolve({ code, stdout: String(stdout || ''), stderr: String(stderr || (err && err.message) || '') });
      });
  });
}

const docker = {
  available: () => exec(dockerBin(), ['version', '--format', '{{.Server.Version}}'], {}, 15000).then((r) => r.code === 0),
  run: (args, env) => exec(dockerBin(), args, env),
  // Remove containers a crash left behind, by our label.
  cleanup: async () => {
    const r = await exec(dockerBin(), ['ps', '-aq', '--filter', 'label=llmjob.rental=1'], {}, 15000);
    const ids = r.code === 0 ? r.stdout.split(/\s+/).filter(Boolean) : [];
    if (ids.length) await exec(dockerBin(), ['rm', '-f'].concat(ids), {}, 60000);
    return ids.length;
  },
  // Follow a container's output, stdout and stderr, line by line.
  logs: (name, onLine) => {
    const child = spawn(dockerBin(), ['logs', '-f', name], { stdio: ['ignore', 'pipe', 'pipe'] });
    const rest = { out: '', err: '' };
    const feed = (key) => (chunk) => {
      const s = R.splitLines(rest[key], chunk);
      rest[key] = s.rest;
      s.lines.forEach(onLine);
    };
    child.stdout.on('data', feed('out'));
    child.stderr.on('data', feed('err'));
    child.on('error', () => {});
    return { stop: () => child.kill() };
  },
  // Resolves the container's exit code once it stops.
  wait: (name) => exec(dockerBin(), ['wait', name], {}, 0).then((r) => {
    const code = parseInt(r.stdout, 10);
    if (r.code !== 0 || !Number.isFinite(code)) throw new Error(r.stderr.trim().split(/\r?\n/)[0] || 'no exit code');
    return code;
  }),
};

const cosign = {
  available: () => exec(cosignBin(), ['version'], {}, 15000).then((r) => r.code === 0),
  verify: (image) => exec(cosignBin(), [
    'verify', '--certificate-identity-regexp', COSIGN_IDENTITY, '--certificate-oidc-issuer', COSIGN_ISSUER, image,
  ], {}, 120000),
};

// Each card as the server's filters see it: [{ index, gpu, vramMb, vendor }].
// NVIDIA from nvidia-smi; with none, AMD from sysfs. Intel is phase 4.
async function rentalCards() {
  const nv = await probe.detectGpusVram();
  if (nv.length) return nv.map((c) => ({ index: c.index, gpu: c.name, vramMb: c.totalMb, vendor: 'nvidia' }));
  return probe.detectAmdGpus().map((c) => ({ index: c.index, gpu: c.name, vramMb: c.totalMb, vendor: 'amd' }));
}

// nvidia-smi's driver version, or null.
function driverVersion() {
  return exec('nvidia-smi', ['--query-gpu=driver_version', '--format=csv,noheader'], {}, 5000)
    .then((r) => (r.code === 0 ? r.stdout.split(/\r?\n/)[0].trim() || null : null));
}

const toNum = (s) => {
  const n = parseFloat(s);
  return Number.isFinite(n) ? n : null;
};

// nvidia-smi rows of "index, temp, fan %, util %" as readings. NVIDIA reports
// no hotspot here.
function parseNvidiaTemps(stdout) {
  const out = {};
  for (const row of String(stdout).split(/\r?\n/)) {
    const p = row.split(',').map((x) => x.trim());
    const index = parseInt(p[0], 10);
    if (p.length < 4 || !Number.isFinite(index)) continue;
    out[index] = { core: toNum(p[1]), hotspot: null, fan: toNum(p[2]), util: toNum(p[3]) };
  }
  return out;
}

// An AMD card's reading from sysfs: edge (temp1) as core, junction (temp2) as
// hotspot, fan rpm, and busy percent.
function amdReading(busId, fsx = fs) {
  const dev = '/sys/bus/pci/devices/' + busId;
  const read = (p) => { try { return String(fsx.readFileSync(p, 'utf8')).trim(); } catch (_) { return ''; } };
  let hw = '';
  try { hw = fsx.readdirSync(dev + '/hwmon').find((d) => /^hwmon\d+$/.test(d)) || ''; } catch (_) { /* none */ }
  const base = dev + '/hwmon/' + hw + '/';
  const milli = (f) => { const n = toNum(read(base + f)); return n == null ? null : Math.round(n / 1000); };
  return {
    core: hw ? milli('temp1_input') : null,
    hotspot: hw ? milli('temp2_input') : null,
    fan: hw ? toNum(read(base + 'fan1_input')) : null,
    util: toNum(read(dev + '/gpu_busy_percent')),
  };
}

// Every card's reading, { index: { core, hotspot, fan, util } }. Never rejects.
async function rentalTemps(fsx = fs) {
  const r = await exec('nvidia-smi', [
    '--query-gpu=index,temperature.gpu,fan.speed,utilization.gpu', '--format=csv,noheader,nounits',
  ], {}, 5000);
  if (r.code === 0) return parseNvidiaTemps(r.stdout);
  const out = {};
  const ids = probe.detectAmdBusIds();
  for (const index of Object.keys(ids)) out[index] = amdReading(ids[index], fsx);
  return out;
}

const takeBackPath = () => path.join(path.dirname(nodeStore.nodePath()), R.TAKEBACK_FILE);

// Ask the running miner to end its job (`llmjob-earn-cli take-back`).
function requestTakeBack(fsx = fs) {
  fsx.writeFileSync(takeBackPath(), String(Date.now()));
  return takeBackPath();
}

// Signed calls to our server, as this machine's node identity.
function rentalApi() {
  const call = async (route, extra) => {
    const node = nodeStore.getOrCreateNode();
    const body = nodeProto.signedBody({
      nodeId: node.nodeId, publicKey: node.publicKey, secretKey: node.secretKey, timestamp: Date.now(),
    }, extra);
    const res = await postJson(serverOf(node) + route, body, 30000);
    if (res.status !== 200) throw new Error('HTTP ' + res.status + ((res.data && res.data.error) ? ': ' + res.data.error : ''));
    return res.data;
  };
  return {
    poll: (body) => call('/api/rigs/poll', body),
    heartbeat: (runId, body) => call('/api/rigs/runs/' + encodeURIComponent(runId) + '/heartbeat', body),
    end: (runId, body) => call('/api/rigs/runs/' + encodeURIComponent(runId) + '/end', body),
  };
}

// A runner wired to the real world. `opts`: { log(level, line), pauseCard(i),
// resumeCard(i), llmBusy(), worker, platform? }.
function createRental(opts) {
  return new RentalRunner({
    api: rentalApi(),
    docker,
    cosign,
    cards: rentalCards,
    driver: driverVersion,
    temps: () => rentalTemps(),
    takeBackRequested: () => fs.existsSync(takeBackPath()),
    clearTakeBack: () => { try { fs.unlinkSync(takeBackPath()); } catch (_) { /* already gone */ } },
    pauseCard: opts.pauseCard,
    resumeCard: opts.resumeCard,
    llmBusy: opts.llmBusy,
    log: opts.log,
    worker: opts.worker,
    platform: opts.platform || process.platform,
    caps: R.capsFor(os.cpus().length, os.totalmem()),
    requireSignature: process.env.LLMJOB_RENT_REQUIRE_SIGNATURE === '1',
  });
}

module.exports = {
  createRental, requestTakeBack, takeBackPath, rentalApi, rentalCards, rentalTemps, driverVersion,
  parseNvidiaTemps, amdReading, exec, docker, cosign, COSIGN_IDENTITY, COSIGN_ISSUER,
};
