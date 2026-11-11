'use strict';

const path = require('path');
const { createRequire } = require('module');
const {
  CU12, CU13, AMD, INTEL, INTEL_DIR, FILES, LABELS, pickCoreVariant, isRuntimeError,
} = require('../shared/coreVariant');
const { clockOffsets } = require('./gpuClockOffset');

// Loads the native PearlHash core (earn/native → a compiled `pearl_core.node`
// N-API addon) and adapts it to the small event interface the host drives:
//
//   core.setJob({ header, target, jobId })   // switch what the GPU searches
//   core.stop()                              // release the GPU, end the search
//   core.on('hit', hit)                      // a candidate at/under target
//   core.on('hashrate', thPerSec)            // periodic throughput
//   core.on('error', err)
//   core.device, core.k, core.kNote          // the card it opened, the k it mines
//   core.untested                            // check its hits before sending them
//
// With `cardK` in its options a core may mine a larger k than the profile's on
// its card (an RDNA 4 card mines k 4096 or 8192), and says so in core.k; PearlMiner
// builds that card's job_key, bound and proofs from it. With `checkHits` the
// AMD core also opens a card whose fold has not run on its kind of card (an RX
// 6000 or Instinct card), and says so in core.untested; PearlMiner checks each
// of its hits before sending it. Every wrapper below passes the options through
// as they are, so both reach the addon.
//
// The CUDA builds of the addon also export the two calls that read and set a
// card's core clock offset (main/gpuClockOffset). They go to the miner with the
// factory: every createCore this returns carries `clockOffsets`, bound to the
// addon its cores come from.
//
// Returns null when the addon is not present. That is the EXPECTED state on any
// machine without a CUDA build of the core (including this dev box and, until the
// build lands, CI): the JS host is complete and tested, but there is nothing to
// mine with until native/ is compiled. The host treats a null core as a first-
// class "engine not built" condition and says so, rather than crashing on a
// missing require.
//
// The require is injected so the whole thing is unit-testable without a real
// .node file, and so the several candidate paths (a dev build tree vs. the
// packaged app's resources) can be probed in one place.
//
// There are two builds of the addon -- see shared/coreVariant for which one a
// rig gets and why. findCore finds `pearl_core.node` (CUDA 12.8); coreFactory
// decides whether to try `pearl_core_cu13.node` first. A third, for AMD cards
// (`pearl_core_hip.node`), is tried first when nvidia-smi lists no NVIDIA card,
// and loads with no fallback when PEARL_CORE_VARIANT=amd asks for it (the CLI's
// --core amd sets that). A fourth, for Intel Arc cards (`pearl_core_sycl.node`,
// in a folder of its own with the libraries it loads), is tried first only
// when nvidia-smi lists no NVIDIA card and the operating system reports Arc
// cards and no AMD card (`opts.intel`), and with no fallback when
// PEARL_CORE_VARIANT=intel asks for it.

// The HIP runtime the Windows AMD core links (native-core.yml checks it), which
// AMD's graphics driver puts in System32.
const WIN_HIP_DLL = 'amdhip64_7.dll';

// What every probe shares: the require to load with, and the directories a
// packaged install keeps its addons in.
function loaderContext(opts) {
  // Not this module's own require: inside a packaged (SEA) binary the
  // bundler's require can only see the snapshot, and a native addon cannot be
  // dlopen'd out of a snapshot at all. createRequire anchored to the real
  // executable can load from the real filesystem in every packaging.
  const req = opts.require || createRequire(opts.execPath || process.execPath);
  const env = opts.env || process.env;
  const execDir = path.dirname(opts.execPath || process.execPath);
  const dirs = [];
  if (opts.resourcesPath) dirs.push(path.join(opts.resourcesPath, 'native'));
  // Packaged CLI: the release ships the cores BESIDE the executable (and the
  // HiveOS tarball unpacks them there), because process.resourcesPath is
  // Electron-only and a snapshot path cannot host a .node file.
  dirs.push(execDir);
  dirs.push(path.join(execDir, 'native'));
  return { req, env, dirs };
}

// The first candidate that loads and looks like our addon, as { addon, file },
// or null. `failed`, when given, collects { file, error } for each candidate
// that is there but would not load, so the caller can say why.
function firstLoadable(req, candidates, failed) {
  for (const c of candidates) {
    try {
      const addon = req(c);
      if (addon && typeof addon.createCore === 'function') return { addon, file: c };
    } catch (e) {
      // Not at this path — try the next. A genuinely broken addon (present but
      // throwing on load) also lands here and falls through to null, which is
      // the right outcome: nobody mines on a core that would not load. But its
      // error is kept: a missing library or a too-old glibc otherwise reads as
      // a missing file, which is how v0.6.7's core failed on Ubuntu 22.04.
      if (failed && !(e && e.code === 'MODULE_NOT_FOUND')) failed.push({ file: c, error: e });
    }
  }
  return null;
}

// What the first file that is there but would not load said, for the log.
function wouldNotLoad(failed) {
  const { file, error } = failed[0];
  const msg = errText(error);
  // dlopen names the library it could not find. For the AMD core that is
  // ROCm's HIP runtime, which the release does not carry.
  let hint = '';
  if (/lib(amdhip64|hsa-runtime64|amd_comgr|rocprofiler-register)/.test(msg)) {
    hint = '. It needs the ROCm 7 HIP runtime (libamdhip64.so.7): install rocm-hip-runtime 7.x, or put ROCm\'s lib folder on LD_LIBRARY_PATH';
  } else if (path.basename(file) === FILES[AMD] && /specified module could not be found/i.test(msg)) {
    // Windows names no library. The one the AMD core needs comes with AMD's driver.
    hint = '. It needs ' + WIN_HIP_DLL + ', which AMD\'s graphics driver installs: install or update the driver';
  }
  if (path.basename(file) === FILES[INTEL]) hint = intelLoadHint(msg);
  // Windows puts the file's path on a second line; one log line reads better.
  return file + ' is there but would not load: ' + msg.replace(/\s*\r?\n\s*/g, ' ').trim() + hint;
}

// Why the Intel core would not load, when the loader's own words do not say.
// It links only the libraries in its folder and the system's C and C++
// runtime, so a missing library is a damaged install, and an old glibc is an
// old system. Windows names no library at all.
function intelLoadHint(msg) {
  if (/GLIBC(XX)?_[0-9.]+. not found/.test(msg)) {
    return '. It needs glibc 2.27 and GCC 7\'s libstdc++ or newer (Ubuntu 18.04 and later have both)';
  }
  if (/lib(pearl_sycl|sycl|umf|ur_adapter)[^ :]*\.so|specified module could not be found/i.test(msg)) {
    return '. Its folder must hold every file the release put there (the SYCL runtime sits beside the core)'
      + (/specified module/i.test(msg) ? ', and Windows needs Microsoft\'s Visual C++ runtime installed' : '')
      + ': install the release again';
  }
  return '';
}

// The CUDA 12.8 core: PEARL_CORE_PATH, then the packaged locations, then the
// dev build tree.
function findCore(opts, failed) {
  const { req, env, dirs } = loaderContext(opts);
  const candidates = [];
  // An operator override first: it makes field diagnosis a one-liner, and it
  // lets a rig run a locally built core without touching the install.
  if (env.PEARL_CORE_PATH) candidates.push(env.PEARL_CORE_PATH);
  for (const d of dirs) candidates.push(path.join(d, FILES[CU12]));
  const dev = [
    path.join(__dirname, '..', '..', 'native', 'build', 'Release', FILES[CU12]),
    path.join(__dirname, '..', '..', 'native', 'build', 'Debug', FILES[CU12]),
  ];
  const found = firstLoadable(req, candidates.concat(dev), failed);
  // A dev-tree build is whatever toolkit that box has -- the 5090 rig builds
  // its own with CUDA 13.3 -- so the log must not call it the 12.8 release.
  if (found && dev.includes(found.file)) found.dev = true;
  return found;
}

// The CUDA 13 core, from the packaged locations only. Nothing builds it into the
// dev tree (a local build is one pearl_core.node, for whatever toolkit that box
// has), and a rig running its own build points PEARL_CORE_PATH at it.
function findCu13Core(opts, failed) {
  const { req, dirs } = loaderContext(opts);
  return firstLoadable(req, dirs.map((d) => path.join(d, FILES[CU13])), failed);
}

// The AMD core (PEARL_CORE_VARIANT=amd only): the packaged locations, then the
// dev tree's build folder, earn/native/amd/build, where build-amd.sh writes it.
function findAmdCore(opts, failed) {
  const { req, dirs } = loaderContext(opts);
  const dev = path.join(__dirname, '..', '..', 'native', 'amd', 'build', FILES[AMD]);
  const found = firstLoadable(req, dirs.map((d) => path.join(d, FILES[AMD])).concat([dev]), failed);
  if (found && found.file === dev) found.dev = true;
  return found;
}

// The Intel core: in its folder (shared/coreVariant.INTEL_DIR) under each
// packaged location, then the dev tree's build folder, earn/native/sycl/build,
// where build.sh writes it.
function findIntelCore(opts, failed) {
  const { req, dirs } = loaderContext(opts);
  const dev = path.join(__dirname, '..', '..', 'native', 'sycl', 'build', FILES[INTEL]);
  const found = firstLoadable(req, dirs.map((d) => path.join(d, INTEL_DIR, FILES[INTEL])).concat([dev]), failed);
  if (found && found.file === dev) found.dev = true;
  return found;
}

// The two cores that are tried before a CUDA build: how each is found, what a
// log line calls it, and how its "there is no such card here" reads.
const FIRST = {
  [AMD]: { find: findAmdCore, name: 'AMD', noCard: /no AMD GPU found/i },
  [INTEL]: { find: findIntelCore, name: 'Intel', noCard: /no Intel GPU found/i },
};

function errText(e) {
  return (e && e.message) || String(e);
}

// Hand a createCore the core clock offset calls of the addon its cores come
// from (main/gpuClockOffset). `current` returns that addon, and is asked on
// every call: the factories with a fallback change addon after their first
// card, and the offset calls must follow.
function withOffsets(createCore, current) {
  createCore.clockOffsets = clockOffsets(current);
  return createCore;
}

// A factory the host calls to get a running core, or null when the addon is
// unavailable. The host depends only on this small surface
// (`createCore(profile, opts) -> core | null`), which a test can fake with a
// bare EventEmitter.
//
// `opts` is per-core: which card it opens, and which slice of the search space
// it owns. The host builds one core per card and fills those in — see
// shared/gpu.planMinerGpus. An addon built before those options existed ignores
// the second argument and behaves exactly as it did, which is what a rig running
// an older pearl_core.node needs.
//
// Which build: `opts.cards` is what nvidia-smi said about the rig
// (probe.detectCudaCards) and `opts.gpus` the cards that will mine; from those
// shared/coreVariant.pickCoreVariant chooses. A caller that passes neither gets
// the 12.8 build, as every rig did before there were two. PEARL_CORE_PATH still
// beats everything: an operator who names a file gets that file.
// `opts.log(level, line)` hears the choice, once, and any fallback.
function coreFactory(opts = {}) {
  const env = opts.env || process.env;
  const log = typeof opts.log === 'function' ? opts.log : () => {};
  const say = (level, variant, why) => log(level, 'Pearl core: ' + LABELS[variant] + ' · ' + why);
  const wrap = (addon) => withOffsets((profile, coreOpts) => addon.createCore(profile, coreOpts || {}), () => addon);

  if (env.PEARL_CORE_PATH) {
    // With PEARL_CORE_VARIANT=amd the named file is the only one tried: every
    // fallback findCore has is a CUDA build, which on an AMD rig would load and
    // then fail on every start. The path counts from the executable's folder
    // when it is relative, not from where the command ran.
    const asked = pickCoreVariant({ env }).variant;
    if (FIRST[asked]) {
      const failed = [];
      const named = firstLoadable(loaderContext(opts).req, [env.PEARL_CORE_PATH], failed);
      if (!named) {
        log('error', 'Pearl core: ' + LABELS[asked] + ' · ' + (failed.length ? wouldNotLoad(failed)
          + ' (PEARL_CORE_PATH; PEARL_CORE_VARIANT=' + asked + ', so no CUDA build is tried)'
          : 'PEARL_CORE_PATH=' + env.PEARL_CORE_PATH + ' not found ('
            + (path.isAbsolute(env.PEARL_CORE_PATH) ? '' : 'a relative path counts from the executable\'s folder; ')
            + 'PEARL_CORE_VARIANT=' + asked + ', so no CUDA build is tried)'));
        return null;
      }
      log('info', 'Pearl core: ' + named.file + ' (PEARL_CORE_PATH)');
      return wrap(named.addon);
    }
    const failed = [];
    const found = findCore(opts, failed);
    if (!found) {
      if (failed.length) log('error', 'Pearl core: ' + wouldNotLoad(failed));
      return null;
    }
    // The named file was there but would not load, and a CUDA build did: say
    // why, since the AMD core named without PEARL_CORE_VARIANT=amd lands here.
    if (found.file !== env.PEARL_CORE_PATH && failed.length) log('warn', 'Pearl core: ' + wouldNotLoad(failed));
    log('info', 'Pearl core: ' + found.file
      + (found.file === env.PEARL_CORE_PATH ? ' (PEARL_CORE_PATH)' : ' (PEARL_CORE_PATH did not load)'));
    return wrap(found.addon);
  }

  const pick = pickCoreVariant({ env, cards: opts.cards, gpus: opts.gpus, intel: opts.intel });
  // The CLI's --core sets PEARL_CORE_VARIANT; the log names what was typed.
  if (opts.coreFlag && pick.reason === 'PEARL_CORE_VARIANT=' + pick.variant) pick.reason = '--core ' + pick.variant;
  const win = (opts.platform || process.platform) === 'win32';

  // The AMD or the Intel core. Asked for by name, there is no fallback to a
  // CUDA build: on such a rig there is nothing for one to run on, and on an
  // NVIDIA rig the operator said otherwise. Picked because nvidia-smi listed no
  // NVIDIA card (and, for the Intel core, the system has Arc cards), it is
  // only a first try.
  const first = FIRST[pick.variant];
  const firstFailed = [];
  if (first) {
    const found = first.find(opts, firstFailed);
    if (found) {
      say('info', pick.variant, (found.dev ? 'local build ' + found.file + ', ' : '') + pick.reason
        // No Windows PC with an Intel GPU has run the Intel core.
        + (pick.variant === INTEL && win ? '; experimental on Windows, where no Intel card has run it yet' : ''));
      return pick.auto ? otherFirst(pick.variant, found.addon, opts, say) : wrap(found.addon);
    }
    if (!pick.auto) {
      log('error', 'Pearl core: ' + LABELS[pick.variant] + ' · ' + (firstFailed.length ? wouldNotLoad(firstFailed)
        : FILES[pick.variant] + ' not found (' + pick.reason + '; ' + whereItIs(pick.variant, win) + ')'));
      return null;
    }
  }

  const cu13Failed = [];
  const cu13 = pick.variant === CU13 ? findCu13Core(opts, cu13Failed) : null;

  if (!cu13) {
    const failed = [];
    const found = findCore(opts, failed);
    if (!found) {
      if (failed.length) log('error', 'Pearl core: ' + wouldNotLoad(failed));
      return null;
    }
    if (found.dev) log('info', 'Pearl core: local build ' + found.file);
    else if (first) {
      say(firstFailed.length ? 'warn' : 'info', CU12, pick.reason + ', and '
        + (firstFailed.length ? wouldNotLoad(firstFailed) : FILES[pick.variant] + ' is not installed'));
    } else if (pick.variant !== CU13) say('info', CU12, pick.reason);
    else if (cu13Failed.length) say('warn', CU12, wouldNotLoad(cu13Failed));
    else say('info', CU12, FILES[CU13] + ' is not installed');
    if (first && firstFailed.length) return cudaAfter(first.name, found.addon, wouldNotLoad(firstFailed));
    return wrap(found.addon);
  }

  say('info', CU13, pick.reason);
  // The fallback. A static CUDA 13 runtime loads on any driver and fails at its
  // first CUDA call, which is inside createCore -- so that is where an old
  // driver the gate above did not catch shows up, and where the 12.8 build
  // takes over. Only until a CUDA 13 core has started: once one has, the
  // runtime demonstrably works here, and a later card's failure is that card's
  // own (usually no VRAM), which the 12.8 build would hit too.
  //
  // The 12.8 addon is loaded only then, so a rig that passed the gate never
  // has two CUDA runtimes in one process.
  let addon = cu13.addon;
  let settled = false;
  return withOffsets((profile, coreOpts) => {
    const o = coreOpts || {};
    if (settled) return addon.createCore(profile, o);
    try {
      const core = addon.createCore(profile, o);
      settled = true;
      return core;
    } catch (e) {
      const cu12 = isRuntimeError(e) ? findCore(opts) : null;
      if (!cu12) throw e;
      say('warn', CU12, 'the CUDA 13 build could not start (' + errText(e) + ')');
      addon = cu12.addon;
      settled = true;
      return addon.createCore(profile, o);
    }
  }, () => addon);
}

// The AMD core picked because nvidia-smi listed no NVIDIA card. Its first card
// decides, as the CUDA 13 build's does: when the AMD core can't start there (no
// AMD card, a card it has no code for, too little VRAM), the 12.8 build
// takes over for every card. That covers an NVIDIA rig whose nvidia-smi failed
// beside AMD's driver for a Ryzen's built-in graphics. When the 12.8 build can't
// start either, the AMD core's reason is the one thrown, since on a rig with no
// NVIDIA card it is the one that says what to fix; unless that reason is that
// there is no AMD card, which makes the CUDA one just as likely.
//
// The Intel core is the same, with one difference in whose reason is thrown:
// it is tried only where the system reports Arc cards, so "no Intel GPU found"
// from it means the cards are there and Intel's compute runtime is not. That
// is the reason an Intel rig needs, with what to install (see intelNoDevice),
// and it is the one thrown.
//
// It is thrown again on every later start that fails the same way, without
// the advice. That takes remembering it: pearl_core.cc keeps its class in one
// global (__pearl_exports), which each addon sets as it loads, so once the
// 12.8 build has been loaded here, createCore of EITHER addon builds the 12.8
// build's core. From the second failed start on, the reason caught from "the
// Intel core" is therefore the CUDA build's own, which would send an Intel rig
// looking for an NVIDIA driver.
function otherFirst(variant, firstAddon, opts, say) {
  const { name, noCard } = FIRST[variant];
  const win = (opts.platform || process.platform) === 'win32';
  let noDevice = null;  // the Intel core's "no Intel GPU found", once it has said it
  let addon = firstAddon;
  let settled = false;
  return withOffsets((profile, coreOpts) => {
    const o = coreOpts || {};
    if (settled) return addon.createCore(profile, o);
    try {
      const core = addon.createCore(profile, o);
      settled = true;
      return core;
    } catch (e) {
      const cu12 = findCore(opts);
      if (!cu12) throw e;
      let core;
      try {
        core = cu12.addon.createCore(profile, o);
      } catch (e2) {
        if (!isRuntimeError(e2)) throw e2;
        const first = variant === INTEL && noCard.test(errText(e));
        if (first || (noDevice && isRuntimeError(e))) {
          const said = noDevice != null;
          if (first) noDevice = errText(e);
          throw new Error(noDevice + (said ? '' : intelNoDevice(win)));
        }
        throw noCard.test(errText(e)) ? e2 : e;
      }
      say('warn', CU12, 'the ' + name + ' build could not start (' + errText(e) + ')');
      addon = cu12.addon;
      settled = true;
      return core;
    }
  }, () => addon);
}

// What to install when the system has Arc cards and the Intel core finds none:
// it reaches a card through Intel's Level Zero or OpenCL driver, which is not
// part of the kernel's graphics driver. Said once a run, with the first error.
function intelNoDevice(win) {
  return win
    ? ': the system lists an Arc card, so Intel\'s graphics driver is missing or too old. Install the current Arc driver from Intel'
    : ': the system lists an Arc card, so Intel\'s compute runtime is missing or this user may not open the card.'
      + ' Install intel-opencl-icd (OpenCL) or libze-intel-gpu1 and libze1 (Level Zero), and add the user to the render group';
}

// Where a release keeps the AMD or the Intel core, for a rig that asked for one
// by name and has none.
function whereItIs(variant, win) {
  if (variant === INTEL) {
    return (win ? 'the Windows installer has it in resources\\native\\' + INTEL_DIR
      : 'the Linux release has it in the ' + INTEL_DIR + ' folder beside the CLI binary and in the AppImage, or build it with'
        + ' earn/native/sycl/build.sh --jit') + ', or set PEARL_CORE_PATH';
  }
  return win ? 'the Windows installer has it in resources\\native, or set PEARL_CORE_PATH'
    : 'the Linux release has it beside the CLI binary and in the AppImage, or build it with'
      + ' earn/native/amd/build-amd.sh gpu, or set PEARL_CORE_PATH';
}

// The 12.8 build standing in for an AMD core that would not load. nvidia-smi
// listed no NVIDIA card, so it is there only in case nvidia-smi was wrong. When
// it finds no card either, its "is an NVIDIA driver installed?" sends an AMD rig
// the wrong way (a 9070 XT rig without ROCm restart-looped on it, 6 Nov), so the
// error thrown is the AMD core's, which says what to install. Only the first
// start can tell; once a core has started the CUDA build plainly works here.
function cudaAfter(name, addon, why) {
  let settled = false;
  return withOffsets((profile, coreOpts) => {
    const o = coreOpts || {};
    if (settled) return addon.createCore(profile, o);
    try {
      const core = addon.createCore(profile, o);
      settled = true;
      return core;
    } catch (e) {
      if (!isRuntimeError(e)) throw e;
      // "no CUDA device found" stays: the HiveOS package's CI smoke test, on a
      // runner with no GPU, waits for those words as proof a core loaded.
      throw new Error('no CUDA device found, and the ' + name + ' core would not load: ' + why);
    }
  }, () => addon);
}

module.exports = { coreFactory };
