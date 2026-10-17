'use strict';

// Which build of the Pearl core a rig should load. Pure: the shells run
// nvidia-smi (probe.detectCudaCards) and pass what it said in here.
//
// A release ships two cores:
//
//   pearl_core.node       CUDA 12.8, sm_75/80/86/89/90a/100/120. Runs on every
//                         card and every driver the app supports. What every rig
//                         loaded before.
//   pearl_core_cu13.node  CUDA 13.x, sm_89 and sm_120 (Ada, compute 8.9, and
//                         Blackwell RTX cards, compute 12.x). No sm_80, sm_86,
//                         sm_90a or sm_100.
//
// The second exists because ptxas 13 compiles the sm_120 fold much better than
// ptxas 12.8: 2.67-3.0 instructions per IMMA with 160/192 B operands reused,
// against 3.891 with 0/192. On an RTX 5090 at 600 W the same v0.5.7 source ran
// 103.97 TH/s built with CUDA 12.8.1 and 107.27 built with 13.3 (+3.2%).
//
// Its sm_89 half is no faster. On an RTX 4090 at 450 W, three interleaved
// rounds against the 12.8 build of the same source averaged 313.23 TH/s to
// 313.08 (+0.05%, inside the run-to-run spread), and verify-hits passed 400/400
// (2026-10-06, native/probes/README.md). ptxas 12.8 already compiles the Ada
// fold well; it is the sm_120 fold it does badly. So this file never picks the
// CUDA 13 build for an Ada card on its own: a new build for no gain. To bench
// it again, point native/probes/hashrate.js and verify-hits.js at
// pearl_core_cu13.node: they take the .node path as their first argument and
// never come through this file. PEARL_CORE_VARIANT=cu13 is the override for
// the app and earn-cli, which do come through here. Either way the card needs
// driver 580 or newer.
//
// It cannot simply replace the first. The runtime is linked statically, and a
// CUDA 13 runtime needs driver 580 or newer; on an older driver it does not fail
// at require() but at the first CUDA call, as "no CUDA device found". Much of
// the 3090/4090 fleet and many 5090 rigs run older drivers. And a 2080 Ti, an
// A100, a 3090, an H100 or a B200 gains nothing from it: it carries no sm_75,
// sm_80, sm_86, sm_90a or sm_100 code, so those cards stay on the 12.8 build.
//
// So the CUDA 13 build is used only when BOTH hold:
//   - the driver is 580 or newer, and
//   - every card that will mine is one it is selected for automatically, which
//     today means compute 12.x (cu13AutoSelectsFor).
// Anything else, including anything we could not read, gets the 12.8 build.
// A mixed rig (a 4090 beside a 5090) loads one addon for all its cards, so it
// gets the 12.8 build too.
//
// A third core, for AMD cards, is opt-in only: pearl_core_hip.node, built with
// ROCm from earn/native/amd (see its README). Nothing here picks it on its
// own; PEARL_CORE_VARIANT=amd asks for it, and PEARL_CORE_PATH can name it.
// It is not in a release yet.

const CU12 = 'cu12';
const CU13 = 'cu13';
const AMD = 'amd';
const FILES = { [CU12]: 'pearl_core.node', [CU13]: 'pearl_core_cu13.node', [AMD]: 'pearl_core_hip.node' };
const LABELS = { [CU12]: 'CUDA 12.8 build', [CU13]: 'CUDA 13 build', [AMD]: 'AMD build (ROCm)' };
// NVIDIA's minor-version compatibility table: every 13.x runtime runs on any
// driver >= 580, on Linux and Windows alike. Below that it cannot start at all.
const MIN_DRIVER_CU13 = 580;
// Blackwell consumer and workstation cards (RTX 50, RTX PRO) are compute 12.x.
// Ada (RTX 40) is compute 8.9 exactly; 8.6 is Ampere (RTX 30) and 8.0 is
// GA100 Ampere (A100, A800, A30, CMP 170HX). 9.0 is Hopper (H100, H200), whose
// sm_90a code is in the 12.8 build only. 10.0 is data-center Blackwell (B200):
// also Blackwell, but a different major, and its sm_100 code is in the 12.8
// build only, so it must never be matched as compute 12.x is.
const BLACKWELL_COMPUTE_MAJOR = 12;
const ADA_COMPUTE_MAJOR = 8;
const ADA_COMPUTE_MINOR = 9;

// Two facts about a card, kept apart on purpose. What the CUDA 13 build is
// compiled for is set by the workflow's gencode list (sm_89 and sm_120); which
// of those cards it is picked for on its own is a decision this file makes,
// and it is Blackwell only: the 4090 bench found Ada's half no faster (see the
// header). Letting Ada in would be making cu13AutoSelectsFor return
// cu13HasCodeFor.
function cu13HasCodeFor(card) {
  return card.major === BLACKWELL_COMPUTE_MAJOR
    || (card.major === ADA_COMPUTE_MAJOR && card.minor === ADA_COMPUTE_MINOR);
}
function cu13AutoSelectsFor(card) {
  return card.major === BLACKWELL_COMPUTE_MAJOR;
}

// Parse `nvidia-smi --query-gpu=index,compute_cap,driver_version
// --format=csv,noheader` ("0, 12.0, 610.57.04") into
//   [{ index, major, minor, driverMajor }]
// Rows that do not parse are dropped. A driver too old to know the compute_cap
// field makes nvidia-smi fail outright, which the caller turns into [] -- and
// [] means the 12.8 build, which is right for a driver that old anyway.
function parseCudaCards(out) {
  const list = [];
  for (const row of String(out == null ? '' : out).split(/\r?\n/)) {
    const parts = row.split(',').map((x) => x.trim());
    if (parts.length < 3) continue;
    const index = parseInt(parts[0], 10);
    const cap = /^(\d+)\.(\d+)$/.exec(parts[1]);
    const drv = /^(\d+)\.\d+/.exec(parts[2]);
    if (!Number.isInteger(index) || !cap) continue;
    list.push({
      index,
      major: parseInt(cap[1], 10),
      minor: parseInt(cap[2], 10),
      driverMajor: drv ? parseInt(drv[1], 10) : null,
    });
  }
  return list;
}

// PEARL_CORE_VARIANT=cu12|cu13 forces one build, for debugging, and
// PEARL_CORE_VARIANT=amd asks for the AMD one. Anything else is ignored (and
// said so in the reason), never guessed at.
function readForced(env) {
  const raw = env && env.PEARL_CORE_VARIANT;
  if (raw == null || String(raw).trim() === '') return { forced: null, note: '' };
  const v = String(raw).trim().toLowerCase();
  if (v === CU12 || v === CU13 || v === AMD) return { forced: v, note: '' };
  return { forced: null, note: 'PEARL_CORE_VARIANT=' + raw + ' ignored (use cu12, cu13 or amd); ' };
}

// Decide the build. `cards` is parseCudaCards' list; `gpus` is the mining list
// (shared/gpu.planMinerGpus) -- empty when the shell starts one core and lets it
// choose, in which case every card has to qualify, since any of them may be
// the one it picks. Returns { variant, reason }, the reason written for the log.
function pickCoreVariant({ env, cards, gpus } = {}) {
  const { forced, note } = readForced(env || {});
  if (forced) return { variant: forced, reason: 'PEARL_CORE_VARIANT=' + forced };
  const cu12 = (why) => ({ variant: CU12, reason: note + why });

  const known = Array.isArray(cards) ? cards : [];
  if (!known.length) return cu12('GPU compute capability unknown');

  const driverMajor = known.map((c) => c.driverMajor).find((d) => Number.isInteger(d));
  if (driverMajor == null) return cu12('driver version unknown');
  if (driverMajor < MIN_DRIVER_CU13) {
    return cu12('driver ' + driverMajor + ' (the CUDA 13 build needs ' + MIN_DRIVER_CU13 + '+)');
  }

  const wanted = Array.isArray(gpus) && gpus.length
    ? gpus.map((g) => g.index)
    : known.map((c) => c.index);
  const caps = [];
  for (const index of wanted) {
    const card = known.find((c) => c.index === index);
    if (!card) return cu12('GPU ' + index + ' compute capability unknown');
    if (!cu13AutoSelectsFor(card)) {
      return cu12('GPU ' + index + ' is compute ' + card.major + '.' + card.minor
        + (cu13HasCodeFor(card)
          ? ' (the CUDA 13 build has code for it but measured no faster on a 4090)'
          : ' (the CUDA 13 build has no code for it)'));
    }
    caps.push(card.major + '.' + card.minor);
  }
  const capText = Array.from(new Set(caps)).join('/');
  return {
    variant: CU13,
    reason: note + 'driver ' + driverMajor + ', '
      + (wanted.length > 1 ? 'every mining card is' : 'mining card is') + ' compute ' + capText,
  };
}

// True when a createCore error from the CUDA 13 core looks like "this runtime
// cannot run here" rather than a problem with the card or the job, which is
// the case worth retrying on the 12.8 build.
//
// The first pattern is the one that matters. pearl_host.cu reports ANY failure
// of cudaGetDeviceCount as "no CUDA device found", and that is where a static
// CUDA 13 runtime on a pre-580 driver stops (cudaErrorInsufficientDriver). The
// rest are cudaGetErrorString's own words for the same family, in case a path
// that passes the error text through reaches them first. A genuinely missing
// GPU also matches the first; retrying it on the 12.8 build costs one extra
// failed call and ends with that build's own error, which is the same message.
const RUNTIME_ERROR = new RegExp([
  'no CUDA device found',
  'driver version is insufficient',
  'unsupported display driver',
  'forward compatibility',
  'no kernel image',
  'unsupported toolchain',
  'device kernel image is invalid',
  'installed CUDA driver',
].join('|'), 'i');

function isRuntimeError(err) {
  const msg = err && err.message != null ? err.message : err;
  return RUNTIME_ERROR.test(String(msg == null ? '' : msg));
}

module.exports = {
  CU12, CU13, AMD, FILES, LABELS, MIN_DRIVER_CU13,
  cu13HasCodeFor, cu13AutoSelectsFor,
  parseCudaCards, pickCoreVariant, isRuntimeError,
};
