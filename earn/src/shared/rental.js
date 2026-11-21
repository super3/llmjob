'use strict';

// Rented containers (issue #318): the rules a rig applies to a job our server
// hands it, kept pure so they can be tested without docker. The runner that
// acts on them is src/main/rentalRunner.js; the real docker, cosign and server
// calls are src/main/rentalHost.js.
//
// A job runs one container on one card. Mining stops on that card for the
// length of the job and starts again after. The owner can take the card back
// at any time, and a job never starts while the owner's LLM is answering.

// Only images from our own registry, pinned by digest. The server checks the
// same rule when a job is created; the rig checks it again before it pulls, so
// a wrong or compromised server still can't make a rig run someone else's image.
const REGISTRY = 'ghcr.io/super3/';
const DIGEST_RE = /^[a-z0-9]+(?:[._/-][a-z0-9]+)*@sha256:[0-9a-f]{64}$/;
const ENV_KEY_RE = /^[A-Za-z_][A-Za-z0-9_]{0,127}$/;

// The temperature guard from the 12 Nov lent-rig incident: stop the container
// at 85 °C core, 95 °C hotspot, or a fan stopped while the card is under load.
const MAX_CORE_C = 85;
const MAX_HOTSPOT_C = 95;
const LOAD_UTIL_PCT = 50;

// Timings.
const HEARTBEAT_MS = 15 * 1000;        // the server releases a run after 3 min of silence
const TEMP_CHECK_MS = 5 * 1000;
const POLL_MS = 60 * 1000;             // used until the server says otherwise
const TAKEBACK_CHECK_MS = 2 * 1000;
// After the owner takes the PC back, no new job for this long.
const HOLD_AFTER_TAKEBACK_MS = 60 * 60 * 1000;
const MAX_LOG_BATCH = 200;             // lines per heartbeat
const MAX_LOG_BUFFER = 2000;           // lines held between heartbeats; older ones drop
const MAX_LINE_CHARS = 2000;

// The file `llmjob-earn-cli take-back` writes beside node.json. The running
// miner sees it within TAKEBACK_CHECK_MS, ends the job and deletes it.
const TAKEBACK_FILE = 'rental-takeback';

// Why `image` may not run, or null when it may.
function imageProblem(image, registry = REGISTRY) {
  const s = typeof image === 'string' ? image : '';
  if (!s.startsWith(registry)) return 'image is not from ' + registry;
  if (!DIGEST_RE.test(s.slice(registry.length))) return 'image is not pinned by digest';
  return null;
}

// Why `env` may not be passed in, or null when it may.
function envProblem(env) {
  if (!env || typeof env !== 'object' || Array.isArray(env)) return 'env is not an object';
  for (const k of Object.keys(env)) if (!ENV_KEY_RE.test(k)) return 'bad env key';
  return null;
}

// The container's name, from the run id the server gave.
function containerName(runId) {
  return 'llmjob-run-' + String(runId).replace(/[^a-zA-Z0-9_.-]/g, '').slice(0, 40);
}

// CPU and RAM caps for a container: half the machine's threads and half its
// RAM, so the owner's PC and the CPU miner keep the rest.
function capsFor(cpuCount, totalMemBytes) {
  return {
    cpus: Math.max(1, Math.floor((Number(cpuCount) || 2) / 2)),
    memoryMb: Math.max(1024, Math.floor((Number(totalMemBytes) || 0) / 2 / 1048576)),
  };
}

// The `docker run` arguments for a job. No host mounts, not privileged, no new
// privileges, capped CPU, RAM and process count, and only the one card:
// NVIDIA through --gpus device=N; AMD through /dev/kfd and /dev/dri (every
// render node for now; per-card AMD passthrough is phase 4 of the issue).
// The job's env is passed by name only (-e KEY). Docker reads each value from
// its own environment, which the caller sets, so values never sit in the
// process list.
function dockerRunArgs({ name, image, env, card, vendor, caps }) {
  const args = [
    'run', '-d', '--name', name,
    '--security-opt', 'no-new-privileges',
    '--cpus', String(caps.cpus),
    '--memory', caps.memoryMb + 'm',
    '--pids-limit', '1024',
    '--label', 'llmjob.rental=1',
  ];
  if (vendor === 'amd') {
    args.push('--device', '/dev/kfd', '--device', '/dev/dri', '--group-add', 'video',
      '-e', 'HIP_VISIBLE_DEVICES=' + card);
  } else {
    args.push('--gpus', 'device=' + card);
  }
  for (const k of Object.keys(env)) args.push('-e', k);
  args.push(image);
  return args;
}

// Why the container must stop now, from one card's reading, or null. A reading
// is { core, hotspot, fan, util } with any field null when unknown.
function tempProblem(r) {
  if (!r) return null;
  if (r.core != null && r.core >= MAX_CORE_C) return 'core at ' + r.core + ' °C (limit ' + MAX_CORE_C + ')';
  if (r.hotspot != null && r.hotspot >= MAX_HOTSPOT_C) return 'hotspot at ' + r.hotspot + ' °C (limit ' + MAX_HOTSPOT_C + ')';
  if (r.fan === 0 && r.util != null && r.util >= LOAD_UTIL_PCT) return 'fan stopped at ' + r.util + '% load';
  return null;
}

// "1h 05m", "42m", "<1m".
function formatLeft(ms) {
  const m = Math.floor(Math.max(0, ms) / 60000);
  if (m < 1) return '<1m';
  if (m < 60) return m + 'm';
  return Math.floor(m / 60) + 'h ' + String(m % 60).padStart(2, '0') + 'm';
}

// What the app, the log and HiveOS say while a job runs on a card, or null
// when none does. `status` is RentalRunner.status().
function rentedLabel(status, now) {
  if (!status) return null;
  if (status.state !== 'running') return 'Rented: starting a container';
  return 'Rented: running a container · ' + formatLeft(status.endsAt - now) + ' left';
}

// What to tell an owner whose platform has no runner yet, or null.
function rentalUnavailable(platform) {
  if (platform === 'linux') return null;
  if (platform === 'win32') return 'Rented containers: not available on Windows yet';
  return 'Rented containers: not available on this system yet';
}

// Log lines from a chunk of docker output, carrying a partial last line over.
// Returns { lines, rest }.
function splitLines(rest, chunk) {
  const parts = (rest + String(chunk)).split(/\r?\n/);
  const tail = parts.pop();
  return { lines: parts.map((l) => l.slice(0, MAX_LINE_CHARS)), rest: tail };
}

module.exports = {
  REGISTRY, MAX_CORE_C, MAX_HOTSPOT_C, LOAD_UTIL_PCT,
  HEARTBEAT_MS, TEMP_CHECK_MS, POLL_MS, TAKEBACK_CHECK_MS, HOLD_AFTER_TAKEBACK_MS,
  MAX_LOG_BATCH, MAX_LOG_BUFFER, MAX_LINE_CHARS, TAKEBACK_FILE,
  imageProblem, envProblem, containerName, capsFor, dockerRunArgs, tempProblem,
  formatLeft, rentedLabel, rentalUnavailable, splitLines,
};
