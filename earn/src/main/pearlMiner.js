'use strict';

const { EventEmitter } = require('events');
const {
  buildAuthorize, buildSubmit, encode, parseMessage, isKryptexHost,
} = require('../shared/miner/stratum');
const {
  PROFILE, meetsTarget, rankMatches, shareBound, buildConfig52, profileForCore, difficultyAdjustmentFactor,
} = require('../shared/miner/pearlhash');
const { hash } = require('../shared/miner/blake3');
const { buildShareProof } = require('../shared/miner/shareProof');
const { checkHit } = require('../shared/miner/checkHit');
const { gzipSizes } = require('../shared/miner/plainProof');
const { combinePayoutAddress, isValidMdlAddress, normalizeAddress } = require('../shared/address');
const gpuClocks = require('./gpuClocks');
const gpuClockOffset = require('./gpuClockOffset');
const memClockTrial = require('../shared/memClockTrial');
const coreOffsetTrial = require('../shared/coreOffsetTrial');
const foldCheck = require('../shared/foldCheck');
const { TRIAL_SECS: SHAPE_TRIAL_SECS } = require('../shared/shapePick');

// The host for our own Pearl miner: it owns the pool socket and the job/lifecycle
// state machine, and drives a PearlCore (the CUDA addon) for the actual search.
// It is the JS counterpart to a competitive miner's thin host layer — everything
// here is protocol, bookkeeping and safety; none of it touches the GPU directly.
//
// Everything IO is injected (connect → a duplex-ish socket, createCore → the
// native core factory, clocks → gpuClocks' nvidia-smi calls, offsets → the
// addon's core clock offset calls) so the entire state machine is unit-testable
// against a fake socket and a fake core, with no network and no GPU.
//
// Emits, for PearlEngine to translate into the events the app shows:
//   log            { level, line }
//   authorized     { host, address, family }  (the pool's address that took the login)
//   job            { jobId, height }
//   share          { jobId, accepted:true }        (pool confirmed)
//   rejected       { jobId, reason }
//   hashrate       thPerSec
//   stopped        {}
//   error          Error
//
// The dev fee is ZERO and there is no dev-address code path, by design — this is
// our own implementation written against the ISC-licensed reference, not a
// derivative of any fee-bearing miner, so there is nothing to disclose or route.

const RECONNECT_MS = 5000;

// Every login asks for compressed proofs (see shared/miner/stratum.js), and a
// pool that agrees is sent nothing else on that connection. A pool that agreed
// and then cannot read them would reject every share for as long as we kept
// asking. So when this many compressed shares are rejected before any is
// accepted, the miner stops asking for the rest of the run and logs in again
// without. One accepted compressed share shows the pool reads them, and any
// rejection after that is about the share, not the compression.
const GZIP_REJECT_LIMIT = 3;

// A core may mine a larger k than the profile's on its card (cardK; on AMD,
// RDNA 4 mines k 4096, or k 8192 on the big gfx1201 cards). Every pool that
// runs the reference verifier takes it (HeroMiners and Kryptex took k 8192 on
// 4 November), but a pool with a check of its own could refuse it: a size cap,
// say, since a k 4096 proof is about 135 KB against 68 KB. So when this many shares at a
// k other than the profile's go unaccepted before any is accepted, every card
// mines the profile's k for the rest of the run. A share counts when the pool
// rejects it, when it goes unanswered for K_ANSWER_MS, or when the pool drops
// the connection while it waits (one a connection): a pool that refuses a long
// line may just hang up. One accepted share shows the pool takes that k.
const K_REJECT_LIMIT = 3;
const K_ANSWER_MS = 30 * 1000;
// A rejection that says nothing about the share's k or its compression: the
// pool read the share and it lost a race. Counted toward neither fallback.
const STALE_RE = /stale|duplicate|job not found|unknown job|expired|outdated/i;

// A card whose search fails is opened again after this long. The faults seen so
// far ("misaligned address", "illegal instruction") came after minutes or hours
// of clean mining, from an overclock at its edge or a driver hiccup, and the
// card mined again after a restart. Left off, it cost its whole hashrate until
// somebody noticed and restarted mining by hand.
const RESTART_MS = 30 * 1000;
// A card that fails again within STABLE_MS of coming back counts toward this
// limit; after RESTART_LIMIT of those in a row it is left off, because a card
// that cannot mine for ten minutes is not having a hiccup. A card that mines
// that long first starts the count over.
const RESTART_LIMIT = 3;
const STABLE_MS = 10 * 60 * 1000;
// Each opening of a card searches salts of its own: the run's first salt is moved
// on by this many of the card's salts. Otherwise a card restarted under the same
// job would search the salts it already searched and send the pool the same
// shares again, which it rejects as duplicates. The card's salts only grow by
// several a second, so 2^32 of them is years. Moving in steps of the stride keeps
// the card in its own lane, clear of every other card.
const SALT_RUN = 2 ** 32;
// The core's refusal when the card has too little free memory for its profile
// (native/src/pearl_host.cu, pearl_host_create's pre-flight).
const NO_VRAM_RE = /not enough free VRAM/i;

// The job a card searches while a step of its core clock offset trial is
// checked (see _startOffsetCheck). Its ids start with OFFSET_CHECK_JOB, and its
// header is no block's, so nothing found for it could be a share.
const OFFSET_CHECK_JOB = 'offset-check-';
const OFFSET_CHECK_HEADER = Buffer.alloc(76, 0x5a);
// Each card's offset trial starts this long after the one planned before it. A
// hit check recomputes about 30 hits a second on the miner's event loop for a
// few seconds, 10 at most, and every card of a rig would otherwise check at the
// same moment.
const OFFSET_STAGGER_MS = 15 * 1000;
// An RTX 30 card may spend its first minutes timing its launch shapes against
// each other (shared/shapePick), a new shape every few seconds. Its offset
// trial waits until that is over, or the two would read each other's changes.
// This fixed wait is the fallback, for a createCore that cannot say when the
// shape trial ends (see _planCoreOffset). It is a guess: the shape trial has
// no fixed length.
const OFFSET_SHAPE_WAIT_MS = (SHAPE_TRIAL_SECS + 30) * 1000;

// The core's refusal when a card was to mine on the rotation fold and the fold
// would not start: the driver refused its file, or its tables would not fit
// (native/src/pearl_host_rot.inc, rot_create). The card is then opened on the
// normal kernel (see _openCore).
const ROT_FAIL_RE = /rotation fold: /;
// Where a card whose rotation fold failed its check is reported.
const REPORT_ROT = 'earn/native/README.md, "A second RTX 40 fold"';

// An offset as the log says it: +50 MHz.
function plusMhz(mhz) {
  return (mhz > 0 ? '+' : '') + mhz + ' MHz';
}

// What to add to a line about an offset call NVML refused for want of rights
// (`r` is main/gpuClockOffset's answer). No sudo is named: the call is made by
// this process, so it is the process that needs them.
function rightsNote(r) {
  return r.noRights ? ': setting a clock offset needs root or administrator rights' : '';
}

// The card a core opened, as { index, name }, or null when it won't say.
//
// A core built before the device choice existed has no `device` at all, and the
// host must stay usable against it — an older pearl_core.node beside a newer
// app is the normal state of a rig mid-upgrade. Null then means "unknown", and
// every consumer falls back to what it did before.
//
// A core that reports its card's memory adds { memTotalMb, memUsedMb } (see
// cardMemory). A core that does not leaves the device as { index, name }.
function readDevice(core) {
  const d = core && core.device;
  if (!d || !Number.isInteger(d.index) || d.index < 0) return null;
  return { index: d.index, name: d.name ? String(d.name) : 'GPU ' + d.index, ...cardMemory(d) };
}

// A card's memory as its core reports it, in whole MB: { memTotalMb, memUsedMb },
// the card's size and what this core has allocated on it. {} when the core
// gives no size. Only the Intel core gives one: nothing else can tell an Intel
// card's memory, where nvidia-smi and amdgpu report the other cards' (see
// shared/minerReport, which is where these end up). Read once for each opening:
// the Intel core allocates everything in its create call (pearl_host_create in
// native/sycl/pearl_sycl_host.cpp). A size with no usable `used` beside it
// reads 0 used.
function cardMemory(d) {
  if (!Number.isFinite(d.memTotalMb) || d.memTotalMb <= 0) return {};
  const used = Number.isFinite(d.memUsedMb) && d.memUsedMb > 0 ? d.memUsedMb : 0;
  return { memTotalMb: Math.round(d.memTotalMb), memUsedMb: Math.round(used) };
}

// A card whose core has just closed, for the 'hashrate' of 0 that says so. The
// core has given the card's memory back, so a device with memory figures reads
// 0 used from here; the board would otherwise show a card that is paused or
// has failed as still holding what it mined with. A device without figures is
// passed on as it is.
function closedDevice(device) {
  return device.memTotalMb == null ? device : { ...device, memUsedMb: 0 };
}

// What a memory clock trial measured, as the log says it: "103.4 TH/s locked
// against 95.7 (+8.0%)". '' when there is nothing to quote: a trial with too
// few samples has no rates, and a remembered answer is read from a file, so
// its rates may be anything.
function lockRates(onTh, offTh) {
  if (!Number.isFinite(onTh) || !Number.isFinite(offTh) || offTh <= 0) return '';
  const gain = onTh / offTh - 1;
  return onTh.toFixed(1) + ' TH/s locked against ' + offTh.toFixed(1) + ' ('
    + (gain >= 0 ? '+' : '') + (gain * 100).toFixed(1) + '%)';
}

// What kind of card a checked core says it runs on, for the hit check's lines
// (see _checkedHit), or null. The AMD core ends its fold's description
// (core.device.fold) with the card's gcnArchName, e.g. gfx1030 or
// gfx90a:sramecc+:xnack-, and names none when it could not read it. The Intel
// core begins its description with the fold: xmx16 or xmx8, the card's XMX
// matrix engines at that sub-group size, or dot, the plain fold.
function archOf(core) {
  const fold = String((core.device && core.device.fold) || '');
  const m = /\bgfx[0-9a-f]+/.exec(fold);
  if (m) return m[0];
  const intel = /^(xmx16|xmx8|dot)\b/.exec(fold);
  return intel ? 'Intel ' + intel[1] + ' fold' : null;
}

// Where a wrong hit from a checked card is reported: the README of the core
// that card runs, at the part that says what to send.
const REPORT_AMD = 'earn/native/amd/README.md, "Testing on real hardware"';
const REPORT_INTEL = 'earn/native/sycl/README.md, "Untested cards"';
function reportTo(arch) {
  return /^Intel /.test(arch || '') ? REPORT_INTEL : REPORT_AMD;
}

class PearlMiner extends EventEmitter {
  constructor({
    connect, createCore, reconnectMs, restartMs, clocks, offsets, agent, now, memClockStore, env,
  } = {}) {
    super();
    this.connect = connect;                 // (host, port) -> socket
    this.createCore = createCore || null;   // (profile) -> core, or null when unbuilt
    this.reconnectMs = reconnectMs == null ? RECONNECT_MS : reconnectMs;
    this.restartMs = restartMs == null ? RESTART_MS : restartMs;
    this.clocks = clocks || gpuClocks;      // { lockMemoryClock, resetMemoryClock }
    // { readCoreOffset, setCoreOffset }: the core clock offset calls of the addon
    // the cores come from, which main/pearlCore hangs on the createCore it
    // returns. A createCore without them is one whose core has no such calls.
    this.offsets = offsets || (createCore && createCore.clockOffsets) || gpuClockOffset.NO_OFFSETS;
    this.now = now || Date.now;             // the clock trials' clock
    // Where a memory clock trial's answer is kept from one start to the next,
    // as { get(key), set(key, value) }: both shells pass a file (see
    // _recallMemClock). Without one nothing is remembered, and every start
    // tries its cards. `env` is read for the switch that forgets the answers.
    this.memClockStore = memClockStore || null;
    this.env = env || process.env;
    this.agent = agent || null;             // 'llmjob-earn/<version>', sent with the login
    // Pending restarts of cards that failed (see _scheduleRestart), cleared by stop().
    this.restartTimers = new Set();
    // slot -> timer for a card waiting for its restart. Still one of this run's
    // cards, so the last other card going down does not end the run (see
    // _outOfCards). Cleared by stop().
    this.restarts = new Map();

    this.sock = null;
    // One core per card. `cores` is [{ core, device, slot, profile }] in the
    // order they were started; `device` is what that core reported, so it is the
    // card really mining rather than the one we asked for, `slot` is what it
    // takes to open that card again (see _openCore), and `profile` is what that
    // card mines: the run's profile at the k its core chose (core.k).
    this.cores = [];
    this.hashrates = new Map();   // card index -> its latest TH/s
    this.running = false;
    this.authorized = false;
    // When a connection the pool had accepted closed, until the next login. The
    // engine lets a routine reconnect pass without telling the board (see
    // PearlEngine.poolConnected).
    this.lostAt = null;
    // Connections in a row that closed before the pool took the login, and the
    // last socket error said. A pool out of reach is retried every few seconds,
    // and saying so each time filled the log (see _openSocket).
    this.poolFails = 0;
    this.lastSockError = null;
    this.job = null;          // the current parsed job the core is searching
    this.prevJob = null;      // the job it replaced: { jobId, height, until } (see _onJob)
    // The current job's job_key for each mining configuration a card mines,
    // keyed by its config52: a card at k 4096 has its own. Filled when a hit
    // first needs one, and emptied by every new job. `tooEasy` is the k whose
    // bound the current job's target was too easy to scale to (see _giveJob).
    this.jobKeys = new Map();
    this.tooEasy = new Set();
    this.buf = '';            // partial-line accumulator for the socket
    this.submitId = 100;      // submit request ids start clear of the authorize id (1)
    // submit id -> { jobId, index, gzip, k }. The card index rides along so an
    // accept or a reject lands on the card that found the share, not on card 0,
    // gzip and k so the verdict can count toward GZIP_REJECT_LIMIT and
    // K_REJECT_LIMIT.
    this.pending = new Map();
    // Per run (see K_REJECT_LIMIT): whether cores may choose their card's k,
    // shares at another k than the profile's turned down so far, and whether
    // one has been accepted. `kTimers` holds a timer for each such share still
    // waiting for its answer, by submit id.
    this.cardK = true;
    this.kRejects = 0;
    this.kAccepted = false;
    this.kTimers = new Map();
    // Set while this miner closes the socket itself (see _gzipRejected), so
    // that close is not taken for a pool hanging up on a share.
    this.selfClosed = false;
    // Compressed proofs. `gzip` is whether THIS connection's pool agreed to
    // them, set by its login's reply and false on every other connection, and
    // `gzipSaid` whether its first compressed share has been logged. The rest
    // is per run (see GZIP_REJECT_LIMIT): whether logins still ask, compressed
    // shares rejected so far, and whether one has been accepted.
    this.gzip = false;
    this.gzipSaid = false;
    this.askGzip = true;
    this.gzipRejects = 0;
    this.gzipAccepted = false;
    this.settings = null;
    // Card indices whose memory clock this run locked, so stop() releases
    // exactly those: never a card whose lock failed, and never one it did not
    // touch. start() does not clear it. A run's locks are released by its own
    // stop(), and a start that fails locks nothing, so start() never finds any
    // -- and if it somehow did, forgetting them would leave them locked.
    this.memClockLocked = new Set();
    // Memory clock trials under way, as card index -> { mhz, byDefault, trial }, and the
    // ones this run has finished, as card index -> kept (see _startMemClockTrial).
    this.memClockTrials = new Map();
    this.memClockVerdicts = new Map();
    // Core clock offsets (see _planCoreOffset). `coreOffsetSet` is card index ->
    // the MHz this run has that card at, for the cards it has above 0, so stop()
    // puts back exactly those. Like memClockLocked, start() does not clear it.
    this.coreOffsetSet = new Map();
    // Offset trials waiting or under way, as card index -> { steps, wait, from,
    // trial, mhz, check }; the cards whose trial is over or will not run, as
    // card index -> the MHz kept (0 for none); and the cards this run has had
    // above 0, whose hits are recomputed before they are sent (see _onHit).
    this.coreOffsetTrials = new Map();
    this.coreOffsetVerdicts = new Map();
    this.coreOffsetChecked = new Set();
    // Hit checks started, which numbers their jobs.
    this.offsetChecks = 0;
    // Cards set aside while the local LLM answers a prompt on them, as
    // slot -> device (see pauseCards). Not in `cores`, so jobs, hashrate ticks
    // and stop() pass them by, but still this run's cards.
    this.paused = new Map();
    // slot -> timer for a paused card whose resume found its VRAM taken (see
    // resumeCards). Cleared by stop(), and by a new pause of that card.
    this.resumeRetries = new Map();
  }

  isRunning() { return this.running; }

  // The cards this rig is mining on, as [{ index, name }]. Empty before a start,
  // and [null] entries are filtered out — a core that won't name its card leaves
  // the caller on its old single-card behaviour.
  devices() {
    return this.cores.map((c) => c.device).filter(Boolean);
  }

  // How many cards this run has: mining, paused for a prompt, or waiting for a
  // restart, the cards _outOfCards counts. A card that would not start is not
  // one, so a planned card HIP doesn't have is left out.
  cardCount() {
    return this.cores.length + this.paused.size + this.restarts.size;
  }

  start(settings = {}) {
    if (this.running) return false;
    this.settings = settings;
    this.running = true;
    this.authorized = false;
    this.poolFails = 0;
    this.lastSockError = null;
    this.job = null;
    this.buf = '';
    // Cleared here, not left from the last run: if no core starts this time, the
    // honest answer is "no cards", not the ones a previous run got.
    this.cores = [];
    this.hashrates = new Map();   // card index -> its latest TH/s
    // Set by releaseMemClocks(): this run's cards stay at the driver's clock.
    this.memClocksReleased = false;
    // Each run settles its cards' locks afresh: from the answer remembered for
    // a card, or from a new trial when there is none that stands.
    this.memClockTrials = new Map();
    this.memClockVerdicts = new Map();
    // And their core clock offsets: what a card can do changes with heat.
    this.coreOffsetTrials = new Map();
    this.coreOffsetVerdicts = new Map();
    this.coreOffsetChecked = new Set();
    // A new run asks for compressed proofs again, whatever the last one found.
    this.askGzip = true;
    this.gzipRejects = 0;
    this.gzipAccepted = false;
    // And lets cores choose their card's k again.
    this.cardK = true;
    this.kRejects = 0;
    this.kAccepted = false;
    this.selfClosed = false;

    // No core means the native addon is not built for this machine. That is a
    // clean, explicable stop — not a crash — so the host says exactly that and
    // does not open a pool socket it could never feed a share to.
    if (!this.createCore) {
      this.emit('log', {
        level: 'error',
        // The loader has logged which file it looked for, or why one that is
        // there would not load, so this only says what follows from it.
        line: 'No Pearl core loaded — nothing to mine with. A release carries pearl_core.node, and '
          + 'pearl_core_hip.node for AMD cards; the line before this one says what went wrong.',
      });
      this.running = false;
      this.emit('stopped', {});
      return false;
    }

    const [host, port] = String(settings.endpoint || '').split(':');
    // Kryptex has no MDL merge mining, and answered a login carrying an MDL
    // address with "Invalid login" (2026-10-20), so there the rig mines
    // Pearl alone. HeroMiners takes the combined address.
    const mdl = isKryptexHost(host) ? null : settings.mdlAddress;
    if (mdl !== settings.mdlAddress && isValidMdlAddress(normalizeAddress(settings.mdlAddress))) {
      this.emit('log', { level: 'warn', line: 'Kryptex has no MDL merge mining: mining Pearl only' });
    }
    const wallet = combinePayoutAddress(settings.address, mdl);
    const worker = settings.worker || 'rig01';
    // Kept for a card restarted mid-run, which is wired like the ones started here.
    this.wallet = wallet;
    this.worker = worker;

    if (!this._startCores(settings, wallet, worker)) {
      this.running = false;
      return false;
    }

    this._openSocket(host, Number(port), wallet, worker);
    return true;
  }

  // Start one core per card, and keep the ones that start.
  //
  // `settings.gpus` is the card list (shared/gpu.planMinerGpus, via
  // probe.detectMinerGpus). An empty list means nvidia-smi told us nothing, so
  // we start a single core with no index and let it choose — the single-card
  // behaviour that shipped before.
  //
  // A card that refuses is skipped, not fatal. The usual reason is no room: the
  // local LLM holds most of the VRAM on that card. One full card used to take
  // the whole rig's mining down with it; now the rest keep mining.
  //
  // Each core gets its own slice of the search space (saltBase/saltStride), or
  // every card would search the same operands and find the same shares.
  _startCores(settings, wallet, worker) {
    const cards = Array.isArray(settings.gpus) && settings.gpus.length
      ? settings.gpus
      : [null];                       // no list: one core, its own choice of card
    const failures = [];

    for (let i = 0; i < cards.length; i++) {
      const card = cards[i];
      const opts = { saltBase: i, saltStride: cards.length };
      if (card && Number.isInteger(card.index)) opts.deviceIndex = card.index;
      // Everything a restart needs to open this card again (see _scheduleRestart).
      const slot = { opts, runs: 0, fails: 0, startedAt: 0 };
      let device;
      try {
        ({ device } = this._openCore(slot, wallet, worker));
      } catch (e) {
        failures.push({ card, message: (e && e.message) || String(e) });
        continue;
      }

      // A core that won't say which card it opened is one built before any of
      // this existed, and it ignores the card we asked for — it mines on CUDA's
      // device 0 whatever we pass. Starting a second such core would stack two
      // searches on that one card and label them as two. Mine on the one, which
      // is exactly what that build did on its own.
      if (!device && cards.length > 1) {
        this.emit('log', {
          level: 'info',
          line: 'this pearl_core.node predates per-card mining — mining on one card. '
            + 'Update to mine on all ' + cards.length + '.',
        });
        break;
      }
    }

    for (const f of failures) {
      const where = f.card
        ? 'GPU ' + f.card.index + (f.card.name ? ' (' + f.card.name + ')' : '')
        : 'the GPU';
      this.emit('log', {
        level: this.cores.length ? 'info' : 'error',
        line: 'skipping ' + where + ': ' + f.message,
      });
    }

    // Nothing started at all. Report the first reason as the error: with one
    // card that IS the reason, and the log above has already listed the rest.
    // There is always a reason — every card either started or failed.
    if (!this.cores.length) {
      this.emit('error', new Error(failures[0].message));
      return false;
    }
    return true;
  }

  // Open one card's core and put it to work: wired, in `cores`, named in the log
  // and, when the memory-clock plan names its card, locked. Returns its `cores`
  // entry, { core, device, slot, profile }, or throws when the card will not
  // open. Used by start() and by a restart alike.
  _openCore(slot, wallet, worker, { quiet = false } = {}) {
    const { opts } = slot;
    // A fresh run of salts each time this card opens (see SALT_RUN). The first
    // run starts where it always has, at the card's own base.
    const saltBase = opts.saltBase + opts.saltStride * SALT_RUN * slot.runs;
    slot.runs++;
    slot.startedAt = Date.now();
    const want = this.profile();
    // cardK lets the core mine a larger k on its card than the profile's, where
    // its fold is faster (see K_REJECT_LIMIT for when it stops). checkHits
    // says this miner checks every hit of a card the core reports untested
    // (see _checkedHit), so the core opens such a card rather than refusing it.
    //
    // A card whose rotation fold failed its check, or would not start, is
    // opened with no choice of k: on an NVIDIA card that is the normal kernel
    // (slot.noRot; see _foldFailed). The rotation fold is the one thing cardK
    // turns on there, and only when it was asked for (--rotation-fold).
    const open = (cardK) => this.createCore(want, { ...opts, saltBase, cardK, checkHits: true });
    let core;
    try {
      core = open(this.cardK && !slot.noRot);
    } catch (e) {
      const message = messageOf(e);
      if (slot.noRot || !ROT_FAIL_RE.test(message)) throw e;
      // The fold would not start on this card. It falls back; it does not
      // take the card down.
      slot.noRot = true;
      this.emit('log', {
        level: 'warn',
        line: (Number.isInteger(opts.deviceIndex) ? 'GPU ' + opts.deviceIndex : 'the GPU')
          + ': rotation fold asked for but not used: ' + message.replace(ROT_FAIL_RE, '')
          + '. Mining on the normal kernel',
      });
      core = open(false);
    }
    // A factory that returns nothing rather than throwing: no core, so the same
    // answer as one that refused. Wiring is inside the caller's try for the same
    // reason — a core we cannot listen to is a core we cannot mine with.
    if (!core) throw new Error('the Pearl core did not initialise');
    // What this card mines: the profile at the k the core reports. A core built
    // before cores could choose has no `k` and mines the profile as it is. This
    // card's config52, job_key, share bound and proofs all come from it, so a
    // rig whose cards mine different k never mixes them.
    // A core with a tile of its own (core.tile: the CUDA core's rotation fold, off by default) mines that.
    const profile = profileForCore(want, core);
    if (!profile) {
      try { core.stop(); } catch (e) { /* already gone */ }
      throw new Error('the core chose k ' + core.k + ', which a share can\'t carry');
    }
    const device = readDevice(core);
    // A card whose fold has not run on its kind of card in a release
    // (core.untested; on AMD, RX 6000 and Instinct cards) has every hit it
    // would send recomputed first (see _checkedHit). A core built before this
    // has no `untested`, and opens no such card.
    const check = core.untested === true ? { slot, arch: archOf(core) } : null;
    // A card on the rotation fold (core.tile) has the fold checked before it
    // is trusted, and every share it finds after that (see _planFoldCheck).
    const fold = core.tile ? { slot } : null;
    this._wireCore(core, device, profile, wallet, worker, check, fold);
    const entry = { core, device, slot, profile };
    this.cores.push(entry);
    // Name the card in the log, every run and every card. The core chooses it
    // — the host cannot see CUDA's device list — so this is the only place the
    // two halves of "which GPU is mining" are written down together. A rig
    // whose UI names one card and whose fan spins up on another (issue #226)
    // is diagnosable from a log file because of it. A card mining another k
    // than the profile's says which, and a core that could have and didn't
    // says why (kNote: too little VRAM, or PEARL_AMD_K).
    //
    // Not on a resume after a prompt: that happens every time the LLM answers
    // one, and the pause and resume are logged already. Unless the card comes
    // back at another k, which the LLM's memory can cause.
    if (device && (!quiet || slot.k !== profile.k)) {
      this.emit('log', {
        level: 'info',
        line: 'mining on GPU ' + device.index + ' · ' + device.name
          + (profile.k !== want.k ? ' · k ' + profile.k : '') + (core.kNote ? ' · ' + core.kNote : ''),
      });
    }
    slot.k = profile.k;
    // Only a card whose core started gets locked: a card that refused is not
    // mining, and the lock would only slow whatever else is using it. Which
    // cards, and at what clock, is { index: mhz } from shared/memClock's
    // planMemClocks, which the shells run: the miner cannot see the compute
    // capabilities, the request, or whether an LLM co-runs, so it only takes the
    // lock a card was planned for. A card the plan has no entry for -- a 4090
    // beside a 5090 on the default -- is left alone. A core that will not name
    // its card is told about in _lockMemClock.
    //
    // Once releaseMemClocks() has run, no card takes the lock again this run. A
    // card that fails and is opened again (see _restartCore) would otherwise
    // lock its clock under the LLM that releaseMemClocks() made way for.
    const memClocks = (!this.memClocksReleased && this.settings.mineMemClockByIndex) || {};
    if (Object.keys(memClocks).length > 0) {
      const index = device ? device.index : null;
      if (index == null || memClocks[index]) {
        // A card on trial (an RTX 5090 by default, or --mine-mem-clock auto)
        // keeps the driver's clock for now and earns the lock. Opened again in
        // the same run, it takes the trial's answer: the lock at once if it was
        // kept, nothing if it was not. A card whose answer is remembered from
        // an earlier start takes that the same way, with no trial.
        const byDefault = !!this.settings.mineMemClockDefault;
        const onTrial = index != null && (this.settings.mineMemClockTrial || []).includes(index);
        const verdict = this.memClockVerdicts.get(index);
        if (onTrial && verdict === undefined) {
          if (!this._recallMemClock(index, memClocks[index], byDefault)) {
            this._startMemClockTrial(index, memClocks[index], byDefault);
          }
        } else if (!onTrial || verdict) {
          this._lockMemClock(index, index == null ? null : memClocks[index], byDefault);
        }
      }
    }
    this._planCoreOffset(core, device);
    if (fold) this._planFoldCheck(slot, device);
    return entry;
  }

  // A card has opened on the rotation fold (shared/foldCheck has the method
  // and the reasons). Its check starts at its first hashrate sample (see
  // _feedFoldCheck); until then it mines the pool's job, and any share it
  // finds is recomputed before it is sent, as after the check. A card that
  // passed earlier in this run and is opened again (after a pause for a
  // prompt, or a restart) is not checked twice.
  _planFoldCheck(slot, device) {
    if (slot.fold && slot.fold.state === 'passed') return;
    slot.fold = foldCheck.createCheck();
    this.emit('log', {
      level: 'info',
      line: cardName(device) + ': rotation fold on (experimental). Checking ' + foldCheck.CHECK_HITS
        + ' of its hits on the CPU before trusting it',
    });
  }

  // A hashrate sample from a card on the rotation fold: start its check at the
  // first one, and end a check that has run out of time.
  _feedFoldCheck(entry, th) {
    const { slot, device, profile } = entry;
    const check = slot.fold;
    const step = foldCheck.tick(check, th, this.now());
    if (step === 'fail') {
      this._foldFailed(slot, device, check.why);
    } else if (step === 'begin') {
      // One card at a time. A hit of the fold takes about a quarter of a second
      // of this event loop to recompute, and a check asks for three a second:
      // a rig of eight cards checking at once would not keep up, and every
      // check would run out of time. A card that waits its turn mines the
      // pool's job meanwhile, its shares recomputed as ever.
      if (this.cores.some((c) => c.slot !== slot && c.slot.fold && c.slot.fold.state === 'checking')) return;
      // A job of the miner's own, as an offset step's check has (see
      // _startOffsetCheck): the pool's job comes back when the check ends.
      this.offsetChecks += 1;
      const jobId = foldCheck.FOLD_CHECK_JOB + this.offsetChecks;
      check.bound = coreOffsetTrial.checkBound(th, difficultyAdjustmentFactor(profile), foldCheck.CHECK_HITS_PER_S);
      check.jobKey = hash(Buffer.concat([OFFSET_CHECK_HEADER, buildConfig52(profile)]));
      foldCheck.begin(check, jobId, this.now());
      entry.core.setJob({ header: OFFSET_CHECK_HEADER, target: check.bound, jobId });
    }
  }

  // A hit of a fold check's job, recomputed as a pool's verifier would. One for
  // no check under way is dropped, as an offset check's is.
  _onFoldCheckHit(hit, fold, device, profile) {
    const check = fold && fold.slot.fold;
    if (!check || check.state !== 'checking' || check.jobId !== hit.jobId) return;
    const why = checkHit(hit, check.jobKey, profile)
      || (meetsTarget(hit.jackpotHash, check.bound) ? null : 'the hash does not meet the check\'s target');
    const step = foldCheck.hit(check, hit.jobId, why);
    if (step === 'fail') {
      this._foldFailed(fold.slot, device, why);
    } else if (step === 'pass') {
      this.emit('log', {
        level: 'info',
        line: cardName(device) + ': rotation fold passed its check (' + foldCheck.CHECK_HITS + ' of '
          + foldCheck.CHECK_HITS + ' hits right). Mining with it',
      });
      // Back to the pool's job, whichever is current now.
      const entry = this.running && this.cores.find((c) => c.slot === fold.slot);
      if (entry) this._giveJob(entry);
    }
  }

  // A share about to be sent from a card on the rotation fold, recomputed
  // first (shared/miner/checkHit; about a quarter of a second at k 65536). True
  // when it is right. A wrong one is not sent, and the card goes back to the
  // normal kernel. Nor is one a core still reports after its fold has failed.
  _foldHitRight(fold, hit, jobKey, profile, device) {
    if (fold.slot.noRot) return false;
    const why = checkHit(hit, jobKey, profile);
    if (!why) return true;
    this._foldFailed(fold.slot, device, why);
    return false;
  }

  // The rotation fold computed something wrong on this card, or its check ran
  // out of time. Nothing it found is sent, and the card is opened again on the
  // normal kernel, on fresh salts, for the rest of the run. A card that is
  // paused or waiting for a restart opens that way when it comes back.
  _foldFailed(slot, device, why) {
    slot.noRot = true;
    slot.fold = null;
    this.emit('log', {
      level: 'error',
      line: cardName(device) + ': the rotation fold failed its check (' + why + '). No share from it was sent, '
        + 'and this card mines on the normal kernel for the rest of this run. Please report this line ('
        + REPORT_ROT + ').',
    });
    const i = this.cores.findIndex((c) => c.slot === slot);
    if (i < 0) return;
    const [c] = this.cores.splice(i, 1);
    try { c.core.stop(); } catch (e) { /* already gone */ }
    // As for any card closed mid-run (see _kStrike): its last figure leaves
    // the total, and its clock lock and offset go with its core.
    if (c.device) {
      this.hashrates.delete(c.device.index);
      this._cardClosed(c.device.index);
    }
    this.hashrate = this.totalHashrate();
    let entry;
    try {
      entry = this._openCore(slot, this.wallet, this.worker);
    } catch (e) {
      this.emit('log', {
        level: 'error',
        line: 'could not open ' + cardName(c.device) + ' again on the normal kernel: ' + messageOf(e),
      });
      if (c.device) this.emit('hashrate', 0, closedDevice(c.device));
      this._cardDown(slot, c.device, e);
      return;
    }
    this._giveJob(entry);
  }

  // Start one card's memory clock trial (shared/memClockTrial has the method
  // and the reasons). The card mines throughout; only its memory clock changes.
  //
  // The trial's first turn is the driver's clock, so the card is handed back
  // to the driver first. A run that was killed never ran its stop(), and its
  // lock stays on the card until a reset or a reboot. Every RTX 5090 used to be
  // locked by default at the very clock its trial now tries, so a 5090 can come
  // to its first trial still locked at it. That trial would time the lock
  // against itself in its first two turns, and give up a lock that pays. On a
  // card with no lock the reset changes nothing.
  //
  // A reset that fails means this process cannot hand the card's clock back
  // (no rights, as a rule), so it could not undo a lock either. The trial is
  // not run, and the rig hears why now, not a turn in.
  //
  // `byDefault` is true when nobody asked for the trial (the RTX 5090's
  // default). Its lines are then information, not warnings, as for a lock
  // nobody asked for (see _lockMemClock). The lines themselves are the same:
  // the shell's plan line has already said the trial is the default's.
  _startMemClockTrial(index, mhz, byDefault) {
    const reset = this.clocks.resetMemoryClock(index);
    if (!reset.ok) {
      this._endMemClockTrial(index);
      this.emit('log', {
        level: byDefault ? 'info' : 'warn',
        line: 'memory clock trial not run on GPU ' + index + ' (' + reset.error + '): setting the memory clock '
          + 'needs root or administrator rights for nvidia-smi. Mining continues at the default memory clock.',
      });
      return;
    }
    this.memClockTrials.set(index, { mhz, byDefault, trial: memClockTrial.createTrial(this.now()) });
    const secs = memClockTrial.TURN_MS / 1000;
    this.emit('log', {
      level: 'info',
      line: 'memory clock trial on GPU ' + index + ': mining ' + secs + " s at the driver's clock and "
        + secs + ' s locked at ' + mhz + ' MHz, twice; the lock is kept only if it is faster',
    });
  }

  // One hashrate sample from a card. Moves its trial on, if it has one: takes
  // or gives back the lock between turns, and after the last turn keeps the
  // lock or releases it for the rest of the run.
  _feedMemClockTrial(index, th) {
    const t = this.memClockTrials.get(index);
    if (!t) return;
    const step = memClockTrial.feed(t.trial, th, this.now());
    if (!step) return;
    if (step.done) {
      this.memClockTrials.delete(index);
      this.memClockVerdicts.set(index, step.keep);
      const measured = lockRates(step.onTh, step.offTh);
      const rates = measured ? ': ' + measured : '';
      // A lock that is not kept comes off before the answer is written down:
      // a miner killed between the two must not leave "no lock" remembered
      // for a card that is still locked.
      //
      // Nor is "no lock" written down when the lock would not come off. For a
      // week every start would take the answer, run no trial and leave the
      // card locked at the clock its trial found slower. With nothing
      // remembered the next start tries the card again, and a trial hands the
      // card back to the driver first. _resetMemClock has said how to release
      // it by hand.
      if (!step.keep && !this._resetMemClock(index, true)) {
        this.emit('log', {
          level: 'info',
          line: 'memory clock trial on GPU ' + index + ' did not keep the lock (' + step.reason + ')' + rates
            + ', but the lock is still on the card; nothing is remembered, so the next start tries again',
        });
        return;
      }
      const until = this._rememberMemClock(index, t.mhz, step);
      this.emit('log', {
        level: 'info',
        line: (step.keep
          ? 'memory clock kept at ' + t.mhz + ' MHz on GPU ' + index + ' while mining'
          : 'memory clock left to the driver on GPU ' + index + ' (' + step.reason + ')')
          + rates + (until ? '; remembered until ' + until : ''),
      });
      return;
    }
    if (!step.lock) {
      // A lock that will not come off ends the trial; _resetMemClock has said so.
      if (!this._resetMemClock(index, true)) this._endMemClockTrial(index);
      return;
    }
    const r = this.clocks.lockMemoryClock(index, t.mhz);
    if (r.ok) {
      this.memClockLocked.add(index);
      return;
    }
    // The card is not tried again this run. A trial that was asked for says so
    // in a warning; the default's is information (see _startMemClockTrial).
    // The reset at the trial's start worked, so this is a driver that hands a
    // clock back but will not lock it, for want of rights or for its own
    // reason. Its own words are in the line.
    this._endMemClockTrial(index);
    this.emit('log', {
      level: t.byDefault ? 'info' : 'warn',
      line: 'memory clock trial ended on GPU ' + index + ' (' + r.error + '): setting the memory clock needs '
        + 'root or administrator rights for nvidia-smi. Mining continues at the default memory clock.',
    });
  }

  // Give up a card's trial without an answer in its favour. Nothing is
  // remembered: the trial compared nothing, and the next start tries again.
  _endMemClockTrial(index) {
    this.memClockTrials.delete(index);
    this.memClockVerdicts.set(index, false);
  }

  // Keep a finished trial's answer for this card's later starts
  // (shared/memClockTrial has what is kept, for how long, and under what key).
  // Returns the day it stands until, for the trial's last line, or null when
  // nothing was kept: no store, or a trial that compared nothing.
  _rememberMemClock(index, mhz, verdict) {
    const answer = this.memClockStore && memClockTrial.remember(verdict, mhz, this.now());
    if (!answer) return null;
    this.memClockStore.set(memClockTrial.cardKey(index, this.settings.gpus), answer);
    return memClockTrial.day(answer.at + memClockTrial.REMEMBER_MS);
  }

  // Use the answer this card's last trial gave, if one is remembered and still
  // stands: under a week old, for the same lock, with the same card in the
  // slot (shared/memClockTrial). True when it was used. The card then has its
  // answer for this run with no trial: the lock at once if it was kept, and
  // the driver's clock if it was not. The line says the answer is a remembered
  // one, what was measured, and when the card is tried again.
  //
  // "No lock" hands the card back to the driver, as the start of a trial does
  // (see _startMemClockTrial). The answer says what the trial found, not what
  // the card is at now: a run that was killed while it held a lock (a trial
  // asked for again, a clock asked for by number) leaves it on the card, and
  // nothing the miner reads says so. Without the reset that card would mine
  // locked for the rest of the answer's week. On a card with no lock the reset
  // changes nothing. One that fails is said: the answer still stands.
  //
  // So a card pays for its trial once a week, not on every start. Tried on
  // every start, a card where the lock pays mined half of each trial without
  // it, and a rig whose mining runs are shorter than a trial never kept the
  // lock at all.
  _recallMemClock(index, mhz, byDefault) {
    if (!this.memClockStore) return false;
    const saved = memClockTrial.recall(
      this.memClockStore.get(memClockTrial.cardKey(index, this.settings.gpus)),
      mhz, this.now(), memClockTrial.retryAsked(this.env));
    if (!saved) return false;
    this.memClockVerdicts.set(index, saved.keep);
    const measured = lockRates(saved.onTh, saved.offTh);
    this.emit('log', {
      level: 'info',
      line: (saved.keep ? 'memory clock on GPU ' + index + ': its trial of ' + memClockTrial.day(saved.at) + ' kept'
        : 'memory clock left to the driver on GPU ' + index + ': its trial of ' + memClockTrial.day(saved.at)
          + ' did not keep')
        + ' the ' + mhz + ' MHz lock' + (measured ? ', ' + measured : '') + '. Remembered: no trial until '
        + memClockTrial.day(saved.at + memClockTrial.REMEMBER_MS) + '.',
    });
    if (saved.keep) {
      this._lockMemClock(index, mhz, byDefault);
      return true;
    }
    const reset = this.clocks.resetMemoryClock(index);
    if (!reset.ok) {
      this.emit('log', {
        level: byDefault ? 'info' : 'warn',
        line: 'could not hand the memory clock on GPU ' + index + ' back to the driver (' + reset.error + '): if an '
          + 'earlier run left it locked, it still is. Setting the memory clock needs root or administrator rights '
          + 'for nvidia-smi.',
      });
    }
    return true;
  }

  // Lock one mining card's memory clock outright: a --mine-mem-clock request,
  // or a lock the card's trial kept, taken again when the card is opened again
  // (see gpuClocks for the measurements). The index is the card the core says
  // it opened. A core that does not say is one that ignores the card it was
  // asked for (see below), so the index it was given is a guess, and a wrong
  // guess would lock a card something else is using.
  //
  // `byDefault` is true when nobody asked for this lock: it comes from the RTX
  // 5090's default, whose trial kept it. Not getting it is then information,
  // not a warning. A rig with no rights to set clocks is the common case, and
  // it should not read a warning on every start for something it never asked
  // for. A requested lock that fails keeps its warning.
  //
  // These lines reach both shells, so they name neither a flag nor sudo: the
  // GUI has no --mine-mem-clock, and on Windows there is no sudoers rule, only
  // running the app as administrator. The shell's own plan line
  // (shared/memClock) says how to turn the default off, and the README says
  // what rights each platform needs.
  _lockMemClock(index, mhz, byDefault) {
    if (index == null) {
      this.emit('log', {
        level: byDefault ? 'info' : 'warn',
        line: 'memory clock left at default: this pearl_core.node does not say which GPU it opened',
      });
      return;
    }
    const r = this.clocks.lockMemoryClock(index, mhz);
    if (r.ok) {
      this.memClockLocked.add(index);
      this.emit('log', {
        level: 'info',
        line: 'memory clock locked at ' + mhz + ' MHz on GPU ' + index + ' while mining',
      });
      return;
    }
    if (byDefault) {
      this.emit('log', {
        level: 'info',
        line: 'memory clock left at the driver\'s default on GPU ' + index + ' (' + r.error
          + '): the RTX 5090 lock needs root or administrator rights for nvidia-smi',
      });
      return;
    }
    this.emit('log', {
      level: 'warn',
      line: 'could not lock the memory clock on GPU ' + index + ' (' + r.error + '): it needs root, '
        + 'or a sudoers NOPASSWD rule for nvidia-smi. Mining continues at the default memory clock.',
    });
  }

  // Release every lock this run took. Synchronous, and done before 'stopped' is
  // emitted: the demand gate starts llama-server as soon as it hears the miner
  // stopped, and shutdown exits soon after, and the lock must be gone before
  // either. Said out loud, because "was the card still locked when the model
  // came up?" is the question a slow LLM on this rig raises, and the log is
  // where it gets answered.
  _releaseMemClocks() {
    // Trials end with the locks: a card in an unlocked turn holds none, and
    // would otherwise take one on its next sample.
    this.memClockTrials.clear();
    for (const index of this.memClockLocked) this._releaseMemClock(index);
  }

  // Release every lock while the miner keeps running. For the GUI's START LLM
  // while mining-only is up: the miner is kept (restarting it would drop the
  // pool connection for nothing) but a model is about to be served from its
  // cards, and a served model must never sit on a locked card. Mining then
  // carries on at the driver's clock -- slower on a card whose lock paid, never
  // wrong. Idempotent, and a later stop() finds nothing left to release. A card
  // restarted after this does not lock again (see _openCore); the next start
  // plans the locks afresh.
  //
  // The same goes for core clock offsets, and more so: every trial ends, and
  // every card goes back to offset 0 and stays there for the run. An offset is
  // only kept while this miner checks what the card computes, and nothing
  // checks what a model computes.
  releaseMemClocks() {
    this.memClocksReleased = true;
    this._releaseMemClocks();
    this._releaseCoreOffsets();
  }

  // Release one card's lock, if this run holds it. Also used when a single core
  // fails mid-run: that card has stopped mining, and whatever runs on it next
  // should not be stuck at the mining clock until the whole rig stops.
  _releaseMemClock(index) {
    // The card is no longer mining, so its trial is over too. Not an answer:
    // if the card is opened again this run, it is tried again.
    this.memClockTrials.delete(index);
    this._resetMemClock(index, false);
  }

  // Hand one card's memory clock back to the driver, if this run holds its
  // lock. True unless the reset failed. `quiet` leaves out the line for a
  // release that worked: a trial releases the lock between turns, and says what
  // it found at the end.
  _resetMemClock(index, quiet) {
    if (!this.memClockLocked.delete(index)) return true;
    const r = this.clocks.resetMemoryClock(index);
    if (r.ok) {
      if (!quiet) this.emit('log', { level: 'info', line: 'memory clock released on GPU ' + index });
      return true;
    }
    this.emit('log', {
      level: 'error',
      line: 'could not release the memory clock on GPU ' + index + ' (' + r.error + '). '
        + 'An LLM on it will decode slowly until you run: sudo nvidia-smi -i ' + index + ' -rmc',
    });
    return false;
  }

  // A card has opened. If the plan names it, it gets a core clock offset trial:
  // it mines turns at offset 0 and at each step up, every step first having its
  // hits recomputed, and keeps the fastest step that is right
  // (shared/coreOffsetTrial has the method and the reasons). The plan is
  // settings.mineCoreOffsetTrial, the card indices from shared/coreOffset's
  // planCoreOffsets, which the shells run: like the memory clock's plan, it
  // needs the compute capabilities, the request and whether an LLM co-runs,
  // none of which the miner can see.
  //
  // Once a run for each card, and only on the card's first opening. A card
  // opened again, after a fault or a pause, comes back at offset 0 and stays
  // there: whatever closed its core ended its trial and took its offset off
  // (see _closedCoreOffset), and a card that has just faulted is no card to
  // run above its own clock. Nor does any card start a trial once
  // releaseMemClocks() has run: an LLM is on the cards.
  //
  // A rig that did not ask is not touched: with no card in the plan, NVML is
  // never called.
  //
  // These lines reach both shells, like the memory clock's, so they name no
  // flag. The shell's plan line says what was asked for.
  _planCoreOffset(core, device) {
    const planned = (!this.memClocksReleased && this.settings.mineCoreOffsetTrial) || [];
    if (!planned.length) return;
    // A core that does not say which card it opened ignores the card it was
    // asked for, so its index is a guess (see _lockMemClock).
    if (!device) {
      this.emit('log', {
        level: 'warn',
        line: 'core clock offset not tried: this pearl_core.node does not say which GPU it opened',
      });
      return;
    }
    const { index } = device;
    if (!planned.includes(index) || this.coreOffsetVerdicts.has(index)) return;
    // The trial was asked for, so not running it is a warning, once a card.
    const notRun = (why) => {
      this.coreOffsetVerdicts.set(index, 0);
      this.emit('log', {
        level: 'warn',
        line: 'core clock offset trial not run on GPU ' + index + why + '. Mining continues at the card\'s own clock.',
      });
    };
    const at = this.offsets.readCoreOffset(index);
    if (!at.ok) { notRun(' (' + at.error + ')' + rightsNote(at)); return; }
    const steps = coreOffsetTrial.stepsFor(at.max);
    if (!steps.length) {
      notRun(': its driver allows at most ' + plusMhz(at.max) + ', less than the first step of '
        + plusMhz(coreOffsetTrial.STEPS_MHZ[0]));
      return;
    }
    // The trial starts from offset 0, whatever the card was left at. Setting it
    // here is also how a rig without the rights for it hears so at the start,
    // not a turn in.
    const zero = this.offsets.setCoreOffset(index, 0);
    if (!zero.ok) {
      notRun(' (' + zero.error + ')' + rightsNote(zero));
      return;
    }
    // An offset the card already had is replaced, and not put back when mining
    // stops or the trial keeps nothing. Nothing the miner can read says whose
    // it was. One a killed run left (it cannot undo its own) would be put back
    // on a card with nothing checking what it computes, and this start is what
    // clears that. So a rig that set its own with another tool loses it, and
    // is told so plainly, in a warning.
    if (at.mhz !== 0) {
      this.emit('log', {
        level: 'warn',
        line: 'GPU ' + index + ' had a core clock offset of ' + plusMhz(at.mhz) + ', left by an earlier run or '
          + 'set by another tool. The trial replaces it: the card starts from 0, mines at what the trial keeps, '
          + 'and is at 0 when mining stops. The ' + plusMhz(at.mhz) + ' is not put back; set it again afterwards '
          + 'if it was yours.',
      });
    }
    // An RTX 30 card may be timing its launch shapes (shared/shapePick), and
    // the offset trial waits for that to end. The shells' createCore says
    // whether this core still is (timingShapes), from the core's own 'shape'
    // event. So a card with a remembered shape does not wait at all, and a
    // trial that runs long, as it does when new jobs restart its turns, is
    // waited out in full. A trial that never reports leaves the offset trial
    // unrun: no offset is better than one measured against changing shapes.
    //
    // A createCore that cannot say gets the fixed wait instead, when its core
    // has shapes to time.
    const asks = this.createCore && typeof this.createCore.timingShapes === 'function';
    const timing = asks ? () => this.createCore.timingShapes(core) : null;
    const shapes = !asks && Array.isArray(core.shapes) && core.shapes.length > 1;
    // How long the trial waits once nothing else is in its way (see
    // _feedCoreOffsetTrial): its turn in the stagger, and the fixed wait above.
    const wait = OFFSET_STAGGER_MS * this.coreOffsetTrials.size + (shapes ? OFFSET_SHAPE_WAIT_MS : 0);
    this.coreOffsetTrials.set(index, { steps, wait, timing, from: null, trial: null, mhz: 0, check: null });
    const first = [
      this.memClockTrials.has(index) ? 'the memory clock trial is over' : '',
      timing && timing() ? 'its launch shapes are timed' : '',
    ].filter(Boolean);
    this.emit('log', {
      level: 'info',
      line: 'core clock offset trial on GPU ' + index + ': mining ' + coreOffsetTrial.TURN_MS / 1000
        + ' s at each of 0, ' + steps.map((mhz) => '+' + mhz).join(', ') + ' MHz and 0 again'
        + (first.length ? ', once ' + first.join(' and ') : '') + '; each step first has '
        + coreOffsetTrial.CHECK_HITS + ' of its hits recomputed, and the fastest step that is right is kept',
    });
  }

  // One hashrate sample from a card. Moves its offset trial on, if it has one.
  _feedCoreOffsetTrial(index, th) {
    const t = this.coreOffsetTrials.get(index);
    if (!t) return;
    const now = this.now();
    if (!t.trial) {
      // The memory clock trial goes first, so the two never overlap: each would
      // read the other's changes as its own. The lock it kept stays in place.
      if (this.memClockTrials.has(index)) return;
      // So does the card's launch shape trial, for the same reason (see
      // _planCoreOffset).
      if (t.timing && t.timing()) return;
      // Then the card's own wait, counted from here and not from when it
      // opened: cards whose memory clock trials end on the same sample still
      // start 15 seconds apart. The first turn starts on a sample, with the
      // card mining.
      if (t.from == null) t.from = now + t.wait;
      if (now < t.from) return;
      t.trial = coreOffsetTrial.createTrial(now, t.steps);
      return;
    }
    this._stepCoreOffset(index, t, coreOffsetTrial.feed(t.trial, th, now));
  }

  // Do what the trial says next (see coreOffsetTrial.feed): move the offset,
  // start a step's hit check or end one, and at the end keep the step it chose.
  // The order matters. The offset goes up before the check's job goes out, and
  // comes down before the pool's job comes back.
  _stepCoreOffset(index, t, step) {
    if (!step) return;
    const entry = this.cores.find((c) => c.device && c.device.index === index);
    // Whatever closes a card's core ends its trial first (see
    // _closedCoreOffset), so a card on trial is mining. Should one ever not
    // be, its trial ends here, at offset 0, and not with a throw in an event
    // handler.
    if (!entry) {
      this._dropCoreOffset(index, false);
      return;
    }
    const moved = step.offset === t.mhz || this._moveCoreOffset(index, t, step.offset);
    if (moved && step.check) {
      this._startOffsetCheck(index, t, entry);
      return;
    }
    // The check is over, passed or failed: back to the pool's job, whichever is
    // current by now.
    if (t.check) {
      t.check = null;
      this._giveJob(entry);
    }
    if (moved && step.done) this._endCoreOffsetTrial(index, step);
  }

  // Set a card on trial to `mhz`. True when it is there. False ends the trial:
  // NVML refused, which it does not do halfway through a trial without a
  // reason, so the card goes back to 0 and stays there.
  _moveCoreOffset(index, t, mhz) {
    let said = false;
    if (mhz !== 0) {
      const r = this.offsets.setCoreOffset(index, mhz);
      if (r.ok) {
        this.coreOffsetSet.set(index, mhz);
        // From here every hit this card sends is recomputed first (see _onHit).
        this.coreOffsetChecked.add(index);
        t.mhz = mhz;
        return true;
      }
      said = true;
      this.emit('log', {
        level: 'warn',
        line: 'core clock offset trial ended on GPU ' + index + ' (' + plusMhz(mhz) + ': ' + r.error + ')'
          + rightsNote(r) + '. Mining continues at the card\'s own clock.',
      });
    }
    // Back to 0: asked for by the trial, or because a step could not be set. A
    // reset that fails has said so itself (see _resetCoreOffset).
    if (this._resetCoreOffset(index, true) && !said) {
      t.mhz = 0;
      return true;
    }
    this.coreOffsetTrials.delete(index);
    this.coreOffsetVerdicts.set(index, 0);
    return false;
  }

  // Check a step: hand the card a job of the miner's own with a target easy
  // enough to hit about 30 times a second (coreOffsetTrial.checkBound, from the
  // card's rate), as native/probes/verify-hits.js does. Its hits come back
  // through _onHit, which sends them to _onOffsetCheckHit and never to the
  // pool. A job the pool sends meanwhile is kept as the current job, as ever,
  // and this card is handed it when the check ends (see _giveJob).
  _startOffsetCheck(index, t, entry) {
    const { profile } = entry;
    this.offsetChecks += 1;
    const jobId = OFFSET_CHECK_JOB + this.offsetChecks;
    const bound = coreOffsetTrial.checkBound(this.hashrates.get(index), difficultyAdjustmentFactor(profile));
    t.check = {
      jobId, bound, profile, jobKey: hash(Buffer.concat([OFFSET_CHECK_HEADER, buildConfig52(profile)])),
    };
    entry.core.setJob({ header: OFFSET_CHECK_HEADER, target: bound, jobId });
  }

  // A hit of a check's job, recomputed from scratch as a pool's verifier would
  // (shared/miner/checkHit). One that is for no check under way is dropped: the
  // core finishes the batches it had queued after its job changes, so a few
  // arrive once their check is over.
  _onOffsetCheckHit(hit, device) {
    const t = device && this.coreOffsetTrials.get(device.index);
    if (!t || !t.check || t.check.jobId !== hit.jobId) return;
    const { jobKey, profile, bound } = t.check;
    const why = checkHit(hit, jobKey, profile)
      || (meetsTarget(hit.jackpotHash, bound) ? null : 'the hash does not meet the check\'s target');
    this._stepCoreOffset(device.index, t, coreOffsetTrial.hit(t.trial, why, this.now()));
  }

  // The trial is over, and the card is at the offset it kept. Say what was
  // measured: the rate at 0 and at each step that had a turn, and the step that
  // failed its check, if one did.
  _endCoreOffsetTrial(index, v) {
    this.coreOffsetTrials.delete(index);
    this.coreOffsetVerdicts.set(index, v.offset);
    const rates = v.baseTh == null ? ''
      : ': ' + v.baseTh.toFixed(1) + ' TH/s at 0' + v.rates.map((r) => ', '
        + (r.th == null ? 'no reading' : r.th.toFixed(1)) + ' at ' + plusMhz(r.mhz)).join('');
    const failed = v.failed ? '; ' + plusMhz(v.failed.mhz) + ' failed its hit check (' + v.failed.why + ')' : '';
    this.emit('log', {
      level: 'info',
      line: (v.offset
        ? 'core clock offset kept at ' + plusMhz(v.offset) + ' on GPU ' + index + ' while mining (+'
          + (v.gain * 100).toFixed(1) + '%)'
        : 'core clock offset left at 0 on GPU ' + index + ' (' + v.reason + ')') + rates + failed,
    });
  }

  // Put one card back at offset 0, if this run has it above. True unless the
  // reset failed. `quiet` leaves out the line for a reset that worked: the
  // trial goes back to 0 for its last turn, and says what it found at the end.
  _resetCoreOffset(index, quiet) {
    const mhz = this.coreOffsetSet.get(index);
    if (!this.coreOffsetSet.delete(index)) return true;
    const r = this.offsets.setCoreOffset(index, 0);
    if (r.ok) {
      if (!quiet) this.emit('log', { level: 'info', line: 'core clock offset back to 0 on GPU ' + index });
      return true;
    }
    // nvidia-smi has no switch for an offset, so there is no command to give.
    this.emit('log', {
      level: 'error',
      line: 'could not set the core clock offset back to 0 on GPU ' + index + ' (' + r.error + '). The card '
        + 'stays at ' + plusMhz(mhz) + ' until the PC restarts or the NVIDIA driver reloads, or until mining '
        + 'next starts with the offset trial on, with the rights to set it.',
    });
    return false;
  }

  // End one card's offset for the rest of the run: its trial, if one is waiting
  // or under way, and the offset it has, back to 0. True when there was either.
  // A card whose hit check this cuts short goes back to the pool's job, if it
  // is still mining.
  _dropCoreOffset(index, quiet) {
    const t = this.coreOffsetTrials.get(index);
    if (!t && !this.coreOffsetSet.has(index)) return false;
    this.coreOffsetTrials.delete(index);
    this.coreOffsetVerdicts.set(index, 0);
    this._resetCoreOffset(index, quiet);
    const entry = t && t.check && this.running && this.cores.find((c) => c.device && c.device.index === index);
    if (entry) this._giveJob(entry);
    return true;
  }

  // A card's core has closed mid-run: it failed, or was set aside for a
  // prompt. Its offset goes with it, at once and not when the card next opens:
  // whatever uses the card meanwhile must find it at its own clock. And for
  // the rest of the run (see _planCoreOffset).
  _closedCoreOffset(index) {
    if (!this._dropCoreOffset(index, true)) return;
    this.emit('log', {
      level: 'info',
      line: 'core clock offset ended on GPU ' + index + ': the card stopped mining, and it gets none for the '
        + 'rest of this run',
    });
  }

  // One card's core has closed mid-run, whatever closed it. Its memory clock
  // lock and its core clock offset go with it, and their trials end. Every
  // place that closes one card's core calls this, so none can leave one of the
  // two behind: the k fallback once released the offset and not the lock.
  _cardClosed(index) {
    this._releaseMemClock(index);
    this._closedCoreOffset(index);
  }

  // Put every card back at offset 0 and end every trial, before 'stopped' is
  // emitted and before an LLM joins the cards (see releaseMemClocks).
  _releaseCoreOffsets() {
    for (const index of new Set([...this.coreOffsetTrials.keys(), ...this.coreOffsetSet.keys()])) {
      this._dropCoreOffset(index, false);
    }
  }

  // A hit about to be sent from a card this run has had above its own clock,
  // recomputed first (shared/miner/checkHit: 4.6 ms). True when it is right. A
  // wrong one is not sent, and the card goes back to offset 0 for the rest of
  // the run: a step that passed its check has since stopped computing right,
  // as a card can when it warms up.
  _offsetHitRight(hit, jobKey, profile, device) {
    const why = checkHit(hit, jobKey, profile);
    if (!why) return true;
    const ended = this._dropCoreOffset(device.index, true);
    this.emit('log', {
      level: 'error',
      line: cardName(device) + ': a hit came out wrong (' + why + '). No share was sent'
        + (ended ? ', and this card gets no core clock offset for the rest of this run.' : '.'),
    });
    return false;
  }

  _openSocket(host, port, wallet, worker) {
    let sock;
    try {
      sock = this.connect(host, port);
    } catch (e) {
      this.emit('error', e);
      return;
    }
    this.sock = sock;
    this.poolHost = host;
    // A new connection sends plain proofs until its own login's reply says
    // otherwise. What the last connection agreed does not carry over.
    this.gzip = false;
    this.gzipSaid = false;

    sock.on('connect', () => {
      this.emit('log', { level: 'info', line: 'connecting to ' + host + ':' + port + ' · worker ' + worker });
      sock.write(encode(buildAuthorize(wallet, worker, host, { agent: this.agent, gzip: this.askGzip })));
    });
    // Only the current socket is heard (see 'close' below).
    sock.on('data', (chunk) => { if (this.sock === sock) this._onData(chunk, wallet, worker); });
    sock.on('error', (err) => {
      // Said once, not on every retry while the pool stays out of reach. A
      // different error is said again, and so is the first one after a login.
      if (err.message !== this.lastSockError) {
        this.emit('log', { level: 'error', line: 'pool socket error: ' + err.message });
      }
      this.lastSockError = err.message;
      // Structured, because a name that never resolves and a pool that is down
      // look identical in the log line above. The host says so once, naming the
      // host it tried — a rig that cannot do DNS otherwise just looks broken.
      this.emit('connect-failed', {
        reason: err.message,
        dns: /ENOTFOUND|EAI_AGAIN|getaddrinfo/i.test(String(err.message)),
      });
    });
    sock.on('close', () => {
      // A socket this miner already let go of: stop() destroyed it, and a start
      // since may have opened another. Its close must not touch that one.
      if (this.sock !== sock) return;
      this._onClose(host, port, wallet, worker);
    });
  }

  _onData(chunk, wallet, worker) {
    this.buf += String(chunk);
    let i;
    while ((i = this.buf.indexOf('\n')) >= 0) {
      const line = this.buf.slice(0, i);
      this.buf = this.buf.slice(i + 1);
      if (line.trim()) this._onMessage(line, wallet, worker);
    }
  }

  _onMessage(line, wallet, worker) {
    const m = parseMessage(line);
    switch (m.kind) {
      case 'auth-ok':
        this.authorized = true;
        // The reply decides how this connection's proofs go: compressed only
        // when the pool said "type":"v2" (see shared/miner/stratum.js).
        this.gzip = m.gzip;
        this.lostAt = null;
        this.poolFails = 0;
        this.lastSockError = null;
        // The address this socket reached, which the pool lookup saves as one
        // that worked (see shared/poolLookup). The reply came in on this.sock.
        this.emit('authorized', {
          host: this.poolHost, address: this.sock.remoteAddress, family: this.sock.remoteFamily,
        });
        // "authorized" stays a line of its own: rigs' health checks match it
        // exactly, and one held every release from v0.6.7 to v0.6.12 back when
        // the compression note joined it (6 Nov).
        this.emit('log', { level: 'info', line: 'authorized' });
        if (m.gzip) this.emit('log', { level: 'info', line: 'the pool takes compressed share proofs' });
        break;
      case 'auth-fail':
        this.emit('log', { level: 'error', line: 'pool rejected the wallet: ' + errText(m.error) });
        break;
      case 'job':
        this._onJob(m);
        break;
      case 'bad-job':
        this.emit('log', { level: 'error', line: 'ignoring an unusable job' + (m.jobId ? ' ' + m.jobId : '') });
        break;
      case 'difficulty':
        // vardiff arrives independently of the job; the next mining.notify carries
        // the widened/narrowed target, so there is nothing to do but note it.
        this.emit('log', { level: 'info', line: 'pool difficulty → ' + m.difficulty });
        break;
      case 'submit-accepted': {
        const p = this._answered(m);
        if (!p) break;
        if (p.gzip) this.gzipAccepted = true;
        if (p.k !== this.profile().k) this.kAccepted = true;
        // `index` is the card that found it, so a multi-card rig credits the
        // right one.
        this.emit('share', { jobId: p.jobId, accepted: true, index: p.index });
        this.emit('log', { level: 'info', line: 'share accepted' });
        break;
      }
      case 'submit-rejected': {
        const p = this._answered(m);
        if (!p) break;
        const reason = errText(m.error);
        const otherK = p.k !== this.profile().k;
        this.emit('rejected', { jobId: p.jobId, reason, index: p.index });
        this.emit('log', { level: 'error', line: 'share rejected' + (otherK ? ' (k ' + p.k + ')' : '') + ': ' + reason });
        // A share at another k counts toward that fallback only, so it goes
        // first: k 2048 with compression is what both pools have taken since
        // v0.6.7. A stale or duplicate share counts toward neither, and one
        // already counted as unanswered is not counted again.
        if (STALE_RE.test(reason)) break;
        if (otherK) {
          if (!p.late) this._kStrike(p.k);
        } else if (p.gzip) {
          this._gzipRejected();
        }
        break;
      }
      case 'unparseable':
      case 'unknown':
        this.emit('log', { level: 'info', line: 'pool: ' + m.raw.slice(0, 200) });
        break;
      // No default: blank lines are filtered before dispatch, so 'empty' never
      // reaches this switch and a default would be a branch that cannot run.
    }
  }

  // The submit a verdict answers, taken off the waiting list. A verdict for an id
  // with nothing waiting (one this miner never sent, or a second answer to one)
  // counts for nothing: counting it would put our figures out of step with the
  // pool's. Its answer is still logged, at error level for an error: an error
  // with no id at all is how a pool says why it is dropping the worker.
  _answered(m) {
    const p = this.pending.get(m.id);
    if (!p) {
      const failed = m.kind === 'submit-rejected';
      this.emit('log', {
        level: failed ? 'error' : 'warn',
        line: 'pool answered ' + (m.id == null ? 'with no share id' : 'share id ' + m.id + ', which this miner is not waiting on')
          + ': ' + (failed ? errText(m.error) : 'accepted') + '; not counted',
      });
      return null;
    }
    this.pending.delete(m.id);
    this._clearKTimer(m.id);
    return p;
  }

  // A share at another k than the profile's was turned down (see
  // K_REJECT_LIMIT). At K_REJECT_LIMIT with none accepted this run, every card
  // at another k is opened again at the profile's, on fresh salts, and cores
  // choose no k of their own for the rest of the run. Once a run.
  _kStrike(k) {
    if (this.kAccepted || !this.cardK) return;
    if (++this.kRejects < K_REJECT_LIMIT) return;
    this.cardK = false;
    const base = this.profile().k;
    this.emit('log', {
      level: 'warn',
      line: 'the pool turned down ' + K_REJECT_LIMIT + ' shares at k ' + k + ' (rejected, unanswered, or dropped '
        + 'with the connection) and accepted none; mining k ' + base + ' on every card for the rest of this run',
    });
    for (const c of this.cores.filter((e) => e.profile.k !== base)) {
      this.cores.splice(this.cores.indexOf(c), 1);
      try { c.core.stop(); } catch (e) { /* already gone */ }
      // Its last figure, at k 4096, leaves the rig total until the new core
      // reports, and the board too if it does not open (as in _onCoreError).
      // Its memory clock lock and its core clock offset go with its core, as
      // for any card closed mid-run. The card opens again just below and takes
      // the lock again there, if it has one to take (see _openCore).
      if (c.device) {
        this.hashrates.delete(c.device.index);
        this._cardClosed(c.device.index);
      }
      this.hashrate = this.totalHashrate();
      let entry;
      try {
        entry = this._openCore(c.slot, this.wallet, this.worker);
      } catch (e) {
        this.emit('log', {
          level: 'error',
          line: 'could not open ' + cardName(c.device) + ' again at k ' + base + ': ' + ((e && e.message) || String(e)),
        });
        if (c.device) this.emit('hashrate', 0, closedDevice(c.device));
        this._cardDown(c.slot, c.device, e);
        continue;
      }
      this._giveJob(entry);
    }
  }

  // Wait K_ANSWER_MS for the answer to a share at another k than the profile's;
  // none by then counts as turned down (see K_REJECT_LIMIT). Not holding the
  // event loop open. Whatever takes the share off `pending` clears its timer
  // (_answered, _onClose, stop), so the share is still waiting when it fires.
  _watchK(id) {
    const timer = setTimeout(() => {
      this.kTimers.delete(id);
      const p = this.pending.get(id);
      p.late = true;
      this._kStrike(p.k);
    }, K_ANSWER_MS);
    timer.unref();
    this.kTimers.set(id, timer);
  }

  _clearKTimer(id) {
    clearTimeout(this.kTimers.get(id));
    this.kTimers.delete(id);
  }

  _clearKTimers() {
    for (const t of this.kTimers.values()) clearTimeout(t);
    this.kTimers.clear();
  }

  // A compressed share was rejected. At GZIP_REJECT_LIMIT of them with none
  // accepted this run, stop asking for compression and log in again: closing
  // the socket runs the usual reconnect (see _onClose), and the next login
  // leaves the type out. Once a run: more rejections arriving before the close
  // do not close it again.
  _gzipRejected() {
    if (this.gzipAccepted || !this.askGzip) return;
    if (++this.gzipRejects < GZIP_REJECT_LIMIT) return;
    this.askGzip = false;
    this.emit('log', {
      level: 'warn',
      line: 'the pool rejected ' + GZIP_REJECT_LIMIT + ' compressed shares and accepted none; '
        + 'reconnecting to send shares uncompressed for the rest of this run',
    });
    // Our own close: not the pool hanging up on a share (see _onClose).
    this.selfClosed = true;
    this.sock.destroy();
  }

  _onJob(job) {
    // Refuse a job whose stated rank is not the one the core was built for.
    // Post-softfork, mining any other rank produces work the network does not
    // credit — which is precisely how alpha-miner's Ada builds ended up running
    // rank 512 and earning nothing while looking perfectly healthy. A job that
    // states no rank (HeroMiners does not) is trusted: the pool decides.
    if (!rankMatches(job.rank, this.settings && this.settings.profile)) {
      this.emit('log', {
        level: 'error',
        line: 'refusing job ' + job.jobId + ': rank ' + job.rank + ' is not the credited '
          + (((this.settings && this.settings.profile) || PROFILE).rank) + ' — not mining uncredited work',
      });
      return;
    }
    // The job this one replaces, so a hit the cores still find for it can say
    // how it was lost (see _onHit).
    this.prevJob = this.job ? { jobId: this.job.jobId, height: this.job.height, until: Date.now() } : null;
    this.job = job;
    // Each configuration's job_key is the new header's from here (see _jobKey).
    this.jobKeys = new Map();
    this.tooEasy = new Set();
    this.emit('job', { jobId: job.jobId, height: job.height });
    // Every card gets the same job. They search different salts (see
    // _startCores), so the same job is a different search on each of them.
    for (const c of this.cores) this._giveJob(c);
  }

  // job_key binds the header to the mining configuration, and everything
  // downstream -- both commitment roots, the noise seeds, and the operands too
  // under the hashed fill (the default constant fill does not read it) -- hangs
  // off it. The pool never sends the configuration, so both sides derive this
  // independently and a single wrong byte is silent. A card mining another k
  // has another config52 and so another job_key: each card's hits are
  // certified with its own, and a hit certified with the other fails the
  // Merkle check and is never sent.
  _jobKey(job, profile) {
    const config = buildConfig52(profile);
    const id = config.toString('hex');
    if (!this.jobKeys.has(id)) this.jobKeys.set(id, hash(Buffer.concat([job.header, config])));
    return this.jobKeys.get(id);
  }

  // The profile this miner mines, which a caller may override wholesale. A
  // card's core may mine it at another k (see _openCore).
  profile() {
    return (this.settings && this.settings.profile) || PROFILE;
  }

  _wireCore(core, device, profile, wallet, worker, check, fold) {
    core.on('hashrate', (th) => {
      // A sample the core queued before it failed can arrive after _onCoreError
      // dropped it; counting it would put the dead card back into the total we
      // report to the pool and the board.
      const entry = this.cores.find((c) => c.core === core);
      if (!entry) return;
      // Per card for the UI and the board, and summed for what we tell the pool.
      if (device) this.hashrates.set(device.index, th);
      this.hashrate = this.totalHashrate();
      this.emit('hashrate', th, device);
      if (device) {
        this._feedMemClockTrial(device.index, th);
        this._feedCoreOffsetTrial(device.index, th);
      }
      if (fold) this._feedFoldCheck(entry, th);
    });
    core.on('error', (err) => this._onCoreError(core, device, err));
    // The profile is this core's, bound here: a hit still on its way from a
    // core that has since been closed is certified at the k it was found at.
    // So is `check`, for a card whose hits are checked (see _openCore).
    // And `fold`, for a card on the rotation fold.
    core.on('hit', (hit) => this._onHit(hit, wallet, worker, device, profile, check, fold));
  }

  // A core's search failed, and a failed core does not search again (its CUDA
  // context is usually gone). Drop that one card, keep mining on the rest, and
  // open the card again in a while (see _scheduleRestart).
  //
  // This used to be relayed as a plain 'error' with the core left in `cores`,
  // so the next pool job called setJob on it. The native core then started a
  // new search thread over the old one, which calls std::terminate, and on
  // Windows the whole app closed a few seconds after the error with nothing on
  // screen. That is how the v0.5.7/v0.5.8 "misaligned address" fault on
  // Blackwell looked to a user.
  _onCoreError(core, device, err) {
    const i = this.cores.findIndex((c) => c.core === core);
    // Already dropped, or the miner has stopped: nothing left to do.
    if (i < 0) return;
    const { slot } = this.cores[i];
    this.cores.splice(i, 1);
    // Frees its VRAM and joins its (finished) search thread.
    try { core.stop(); } catch (e) { /* already gone */ }

    const message = (err && err.message) || String(err);
    if (device) {
      // The core already names its card ("GPU 1: ..."); say it once.
      const prefix = 'GPU ' + device.index + ': ';
      const reason = message.startsWith(prefix) ? message.slice(prefix.length) : message;
      this.emit('log', {
        level: 'error',
        line: 'GPU ' + device.index + ' (' + device.name + ') stopped: ' + reason,
      });
      // Its last tick would otherwise stand in the rig total and on the board
      // for as long as the others keep mining.
      this.hashrates.delete(device.index);
      this.hashrate = this.totalHashrate();
      this.emit('hashrate', 0, closedDevice(device));
      this._cardClosed(device.index);
    } else {
      this.emit('log', { level: 'error', line: 'the GPU stopped: ' + message });
    }

    this._cardDown(slot, device, err);
  }

  // A card is down. Open it again in a while, unless it keeps failing.
  //
  // A card left off with others still mining just stays off. The last card left
  // off is the end of mining: the same end as a start with no core, an error
  // that says why and then 'stopped', so the app shows mining has stopped
  // instead of a pool connection that mines nothing. While the last card waits
  // for its restart, the pool connection stays open, so it comes back to work.
  _cardDown(slot, device, err) {
    if (this._scheduleRestart(slot, device)) return;
    if (this._outOfCards()) {
      this.emit('error', err);
      this.stop(true);
    }
  }

  // Whether no card of this run is left. A paused card comes back when the
  // prompt ends, and one waiting for its restart when its timer fires, so
  // both still count: ending the run would clear that restart and leave a
  // healthy card off.
  _outOfCards() {
    return !this.cores.length && !this.paused.size && !this.restarts.size;
  }

  // Returns true when a restart is scheduled, false when the card is left off.
  //
  // The timer is not unref'd, unlike the reconnect timer. With the last card
  // down, it can be the one thing this miner has left to do, and a process that
  // drains its loop and exits 0 there is one systemd's Restart=on-failure never
  // brings back. stop() clears it.
  _scheduleRestart(slot, device) {
    const where = cardName(device);
    // A card that mined a good while since it last came up was healthy, so this
    // is a new fault, not the same one again.
    if (Date.now() - slot.startedAt >= STABLE_MS) slot.fails = 0;
    if (slot.fails >= RESTART_LIMIT) {
      this.emit('log', {
        level: 'error',
        line: 'leaving ' + where + ' off: it stopped again within ' + STABLE_MS / 60000
          + ' minutes of each of ' + RESTART_LIMIT + ' restarts. Stop and start mining to try it again.',
      });
      return false;
    }
    slot.fails++;
    this.emit('log', {
      level: 'info',
      line: 'restarting ' + where + ' in ' + Math.round(this.restartMs / 1000) + ' s',
    });
    const timer = setTimeout(() => {
      this.restartTimers.delete(timer);
      this.restarts.delete(slot);
      this._restartCore(slot, device);
    }, this.restartMs);
    this.restartTimers.add(timer);
    this.restarts.set(slot, timer);
    return true;
  }

  // Open a failed card again and give it the job the rest of the rig is on. The
  // pool will not send that job again, and a core does not search until it has
  // one. Only reached while running: stop() clears the timers that call it.
  _restartCore(slot, device) {
    let entry;
    try {
      entry = this._openCore(slot, this.wallet, this.worker);
    } catch (e) {
      this.emit('log', {
        level: 'error',
        line: 'could not restart ' + cardName(device) + ': ' + ((e && e.message) || String(e)),
      });
      this._cardDown(slot, device, e);
      return;
    }
    this._giveJob(entry);
  }

  // Hand a card the current job: every card when a job comes, and a card that
  // just opened the job the rest of the rig is on. Until it has one it does not
  // search, and the pool will not send this job again.
  //
  // The core compares against the SCALED bound, not the pool's raw target. The
  // protocol makes the bound easier in proportion to the work one attempt costs
  // — that is the same factor that makes a hashrate here count
  // multiply-accumulates rather than attempts. Comparing against the raw target
  // makes shares 65536x rarer than the pool intends, which is indistinguishable
  // from simply being slow. Each card's bound is at its own k: a k 4096 card's
  // is twice a k 2048 card's, as each attempt costs twice the work.
  //
  // A target too easy to scale is refused rather than saturated, which would
  // let every hash through. Said once a job for each k; those cards wait for
  // the next job.
  _giveJob(entry) {
    const job = this.job;
    if (!job) return;
    // A card in a hit check keeps the check's job (see _startOffsetCheck). It
    // is handed the pool's job when the check ends, whichever is current then,
    // so a job that arrives during a check is not lost.
    const onTrial = entry.device && this.coreOffsetTrials.get(entry.device.index);
    if (onTrial && onTrial.check) return;
    // So does a card whose rotation fold is in its check (see _feedFoldCheck).
    if (entry.slot.fold && entry.slot.fold.state === 'checking') return;
    const bound = shareBound(job.target, entry.profile);
    if (bound == null) {
      if (!this.tooEasy.has(entry.profile.k)) {
        this.tooEasy.add(entry.profile.k);
        this.emit('log', {
          level: 'error',
          line: 'pool target is too easy to scale for k ' + entry.profile.k + '; not mining this job at that k',
        });
      }
      return;
    }
    entry.core.setJob({ header: job.header, target: bound, jobId: job.jobId });
  }

  // Set cards aside while the local LLM answers a prompt on them, and bring them
  // back after (see shared/promptPause). `indices` is the cards' numbers, or
  // null for every card. Returns the cards it acted on, as devices.
  //
  // A pause closes the card's core, which stops its search and frees its VRAM,
  // and keeps everything else: the pool connection, the job and the other
  // cards. It is the core's only way to stop searching (native/src/pearl_core.cc
  // has no lighter one). Its hashrate goes to 0, so the app and the board show
  // the card is not mining rather than its last tick.
  pauseCards(indices) {
    if (!this.running) return [];
    // A card still waiting to come back from the last pause waits for this
    // prompt instead: its retry must not start it under the model.
    for (const [slot, device] of this.paused) {
      const index = device ? device.index : null;
      if (indices == null || indices.includes(index)) this._cancelResumeRetry(slot);
    }
    const done = [];
    for (const c of [...this.cores]) {
      const index = c.device ? c.device.index : null;
      if (indices != null && !indices.includes(index)) continue;
      this.cores.splice(this.cores.indexOf(c), 1);
      try { c.core.stop(); } catch (e) { /* already gone */ }
      this.paused.set(c.slot, c.device);
      if (c.device) {
        this.hashrates.delete(index);
        this.hashrate = this.totalHashrate();
        this.emit('hashrate', 0, closedDevice(c.device));
        // A model must never run on a locked card (see releaseMemClocks), or
        // on one above its own clock.
        this._cardClosed(index);
      }
      done.push(c.device);
    }
    return done;
  }

  // Open paused cards again the way a restart after a fault does: on a fresh run
  // of salts, so the pool is never sent a share twice, and on the current job.
  // A card that will not open is a card down like any other (see _cardDown).
  resumeCards(indices) {
    if (!this.running) return [];
    const done = [];
    for (const [slot, device] of [...this.paused]) {
      const index = device ? device.index : null;
      if (indices != null && !indices.includes(index)) continue;
      this.paused.delete(slot);
      this._cancelResumeRetry(slot);
      // A pause is not a fault. Keep when the card last came up, so a fault
      // soon after a resume is not counted as one soon after a restart.
      const startedAt = slot.startedAt;
      let entry;
      try {
        entry = this._openCore(slot, this.wallet, this.worker, { quiet: true });
      } catch (e) {
        slot.startedAt = startedAt;
        const message = (e && e.message) || String(e);
        // The VRAM the pause gave up was taken while the LLM answered: a game,
        // a browser, another GPU app. That is not the card failing, and
        // counting it as one would leave the card off for the session after
        // four tries (or stop a one-card rig). Keep it paused and try again
        // until the memory is back, saying so once.
        if (NO_VRAM_RE.test(message)) {
          this.paused.set(slot, device);
          if (!slot.waitingForVram) {
            slot.waitingForVram = true;
            this.emit('log', {
              level: 'warn',
              line: 'could not resume ' + cardName(device) + ': ' + message + ' Something else took the '
                + 'memory it gave up while the local LLM answered. Trying again every '
                + Math.round(this.restartMs / 1000) + ' s.',
            });
          }
          const timer = setTimeout(() => {
            this.restartTimers.delete(timer);
            this.resumeRetries.delete(slot);
            this.resumeCards(device ? [device.index] : null);
          }, this.restartMs);
          this.restartTimers.add(timer);
          this.resumeRetries.set(slot, timer);
          continue;
        }
        this.emit('log', {
          level: 'error',
          line: 'could not resume ' + cardName(device) + ': ' + message,
        });
        this._cardDown(slot, device, e);
        continue;
      }
      slot.startedAt = startedAt;
      if (slot.waitingForVram) {
        slot.waitingForVram = false;
        this.emit('log', { level: 'info', line: cardName(device) + ' has its memory back and is mining again' });
      }
      this._giveJob(entry);
      done.push(device);
    }
    return done;
  }

  _cancelResumeRetry(slot) {
    const timer = this.resumeRetries.get(slot);
    if (!timer) return;
    clearTimeout(timer);
    this.restartTimers.delete(timer);
    this.resumeRetries.delete(slot);
  }

  // The rig's throughput: every card's latest tick, added up. A card that has
  // not ticked yet contributes nothing rather than a guess.
  totalHashrate() {
    let total = 0;
    for (const th of this.hashrates.values()) total += Number(th) || 0;
    return total;
  }

  // A candidate the core found. Re-verify it against the CURRENT job's target in
  // JS before submitting: the core may have been searching a job that vardiff has
  // since moved, and a bad submit earns a ban. Stale hits are dropped silently —
  // they are not errors, just races. `profile` is what the core that found it
  // mines: the bound, the job_key and the proof are all at its k. `check` is
  // set for a card whose hits are checked (see _openCore).
  _onHit(hit, wallet, worker, device, profile, check, fold) {
    // A hit of a hit check's job is the check's, and never goes to the pool
    // (see _startOffsetCheck).
    if (String(hit.jobId).startsWith(OFFSET_CHECK_JOB)) {
      this._onOffsetCheckHit(hit, device);
      return;
    }
    // Nor does one of a rotation fold's check (see _feedFoldCheck).
    if (String(hit.jobId).startsWith(foldCheck.FOLD_CHECK_JOB)) {
      this._onFoldCheckHit(hit, fold, device, profile);
      return;
    }
    // A card one of whose hits failed the check sends nothing more, not even a
    // hit already on its way from its stopped core (see _wrongHit).
    if (check && check.slot.wrongHit) return;
    // The pool connection is down (see _onClose), so there is nowhere to send
    // it, and the next connection brings jobs of its own. Said, because this is
    // a share lost to the reconnect, not a race. It used to be written to the
    // closed socket, which drops it without a word. A socket the miner has
    // destroyed itself (see _gzipRejected) counts as down: it is still
    // this.sock until its close event comes, and a write to it is lost.
    if (this.running && (!this.sock || this.sock.destroyed)) {
      this.emit('log', { level: 'info', line: 'dropping a hit found while the pool connection is down' });
      return;
    }
    const job = this.job;
    if (!job) return;
    if (hit.jobId !== job.jobId) {
      // A hit from the batches the core finishes after a new job arrives (see
      // SearchLoop in native/src/pearl_core.cc). Said, with whether the block
      // changed, because these are shares the pool never sees: the hashrate
      // counts their work, the pool does not. One for a job at the same height
      // is one a pool that keeps recent jobs might still take.
      const prev = this.prevJob;
      if (prev && hit.jobId === prev.jobId) {
        const same = prev.height != null && prev.height === job.height;
        this.emit('log', {
          level: 'info',
          line: 'dropping a hit for the job replaced ' + (Date.now() - prev.until) + ' ms ago ('
            + (same ? 'same block' : 'new block') + ')',
        });
      }
      return;
    }
    if (!meetsTarget(hit.jackpotHash, shareBound(job.target, profile))) {
      this.emit('log', { level: 'info', line: 'dropping a hit that no longer meets target (vardiff moved)' });
      return;
    }
    // Certify the hit before it goes anywhere. The proof was captured on the
    // device at the moment of the hit, because the search re-draws its operands
    // every few tens of milliseconds and a proof read back afterwards belongs to
    // a different matrix than the hash it certifies.
    //
    // Compressed exactly when the connection it goes out on agreed to it: a
    // pool that agreed rejects a plain proof, and one that did not cannot read
    // a compressed one.
    const gzip = this.gzip;
    const jobKey = this._jobKey(job, profile);
    // A checked card's hit is checked first. Its check covers what
    // buildShareProof checks, so a proof that does not verify stops the card
    // too, as a wrong tree or leaf readout on the card would give every hit.
    if (check && !this._checkedHit(check, hit, jobKey, profile, device)) return;
    // So is a hit of a card this run has had above its own clock, at any time
    // since: a batch computed at the offset can still report after it is gone.
    if (!check && device && this.coreOffsetChecked.has(device.index)
      && !this._offsetHitRight(hit, jobKey, profile, device)) return;
    // And every share of a card on the rotation fold, whose loop is written
    // by hand: before its check, during it and for as long as it mines.
    if (fold && !this._foldHitRight(fold, hit, jobKey, profile, device)) return;
    const plainProof = buildShareProof(hit, jobKey, profile, { gzip });
    if (!plainProof) {
      this.emit('log', { level: 'error', line: 'dropping a hit whose proof does not verify locally' });
      return;
    }
    // Once a connection, so the log shows what the compression is worth.
    if (gzip && !this.gzipSaid) {
      this.gzipSaid = true;
      const { plain, sent } = gzipSizes(plainProof);
      this.emit('log', { level: 'info', line: 'sending compressed share proofs: ' + kb(plain) + ' -> ' + kb(sent) });
    }
    const id = this.submitId++;
    this.pending.set(id, { jobId: job.jobId, index: device ? device.index : 0, gzip, k: profile.k });
    if (profile.k !== this.profile().k) this._watchK(id);
    this.sock.write(encode(buildSubmit(id, {
      jobId: job.jobId, plainProof, hashrate: (this.hashrate || 0) * 1e12,
    })));
  }

  // A hit from a card whose fold has not run on its kind of card, recomputed as
  // the pool's verifier would before it is sent (shared/miner/checkHit: 4.6 ms
  // at k 2048, 8.6 at k 4096). True when it is right; the first right one is
  // said once a run. A fold that is wrong on the card makes every hash wrong,
  // and every one still meets the target, so every share would be rejected and
  // a pool can ban the address for that. A wrong hit is not sent, and the card
  // goes off (see _wrongHit).
  _checkedHit(check, hit, jobKey, profile, device) {
    const why = checkHit(hit, jobKey, profile);
    const where = cardName(device) + ' · ' + (check.arch || 'unknown architecture');
    if (why) {
      this._wrongHit(check, device, where, why);
      return false;
    }
    if (!check.slot.hitsChecked) {
      check.slot.hitsChecked = true;
      this.emit('log', {
        level: 'info',
        line: where + ': this kind of card has not mined in a release yet, so each of its hits is checked '
          + 'here before it is sent; the first was right',
      });
    }
    return true;
  }

  // A hit of a checked card came out wrong, so its fold is likely wrong on it
  // and every hit it finds would be too. The card is off for the rest of the
  // run: its core is stopped and not opened again, by a restart or a resume,
  // and the other cards keep mining. The hit can come from a core already
  // stopped (paused for a prompt, or failed and waiting for its restart), so
  // the card is found by its slot, taken out of `paused`, and its restart
  // cancelled.
  _wrongHit(check, device, where, why) {
    const { slot } = check;
    slot.wrongHit = true;
    const i = this.cores.findIndex((c) => c.slot === slot);
    if (i >= 0) {
      const [{ core }] = this.cores.splice(i, 1);
      try { core.stop(); } catch (e) { /* already gone */ }
    }
    this._cancelResumeRetry(slot);
    this.paused.delete(slot);
    const restart = this.restarts.get(slot);
    if (restart) {
      clearTimeout(restart);
      this.restartTimers.delete(restart);
      this.restarts.delete(slot);
    }
    const line = where + ': a hit came out wrong (' + why + '). No share was sent, and this card is off for '
      + 'the rest of this run. This kind of card has not mined in a release yet: please report this line '
      + '(' + reportTo(check.arch) + ').';
    this.emit('log', { level: 'error', line });
    if (device) {
      this.hashrates.delete(device.index);
      this.hashrate = this.totalHashrate();
      this.emit('hashrate', 0, closedDevice(device));
      this._cardClosed(device.index);
    }
    // The last card going is the end of mining, as in _cardDown.
    if (this._outOfCards()) {
      this.emit('error', new Error(line));
      this.stop(true);
    }
  }

  _onClose(host, port, wallet, worker) {
    if (this.authorized) this.lostAt = Date.now();
    else this.poolFails++;
    this.authorized = false;
    // A pool that will not take a share at this k may just hang up on it. So a
    // close the miner did not make, with such a share still unanswered (and not
    // already counted as late), counts once toward K_REJECT_LIMIT.
    const waiting = [...this.pending.values()].find((p) => p.k !== this.profile().k && !p.late);
    const hungUp = waiting && !this.selfClosed;
    this.selfClosed = false;
    // The socket goes, and what belonged to it: the pool will not answer the
    // submits sent on it, and the next connection sends jobs of its own, so a
    // share of this one must not reach it. The cores keep searching the job
    // they have, so a brief pool blip does not idle the GPU; a hit found before
    // the next job is dropped (see _onHit).
    this.sock = null;
    this.gzip = false;
    this.job = null;
    this.pending.clear();
    this._clearKTimers();
    this.buf = '';
    if (!this.running) return;
    if (hungUp) this._kStrike(waiting.k);
    // Reopen after a backoff. Said for the first close and the first retry
    // that fails, not for every retry while the pool stays out of reach.
    if (this.poolFails <= 1) {
      this.emit('log', {
        level: 'info',
        line: 'pool connection closed; reconnecting'
          + (this.poolFails ? ' (until it connects, the retries are not logged)' : ''),
      });
    }
    this._scheduleReconnect(host, port, wallet, worker);
  }

  _scheduleReconnect(host, port, wallet, worker) {
    this._reconnectTimer = setTimeout(() => {
      this._openSocket(host, port, wallet, worker);
    }, this.reconnectMs);
    // Do not keep the event loop alive on a timer alone.
    this._reconnectTimer.unref();
  }

  // `failed` marks the one stop the miner makes on its own, when its last card
  // has died, and 'stopped' carries it ({ failed: true }) so a supervisor can
  // tell that from a stop somebody asked for (see PearlEngine).
  stop(failed = false) {
    if (!this.running) return false;
    this.running = false;
    if (this._reconnectTimer) { clearTimeout(this._reconnectTimer); this._reconnectTimer = null; }
    for (const t of this.restartTimers) clearTimeout(t);
    this.restartTimers.clear();
    this.restarts.clear();
    this.resumeRetries.clear();
    for (const c of this.cores) {
      try { c.core.stop(); } catch (e) { /* that core is already gone */ }
    }
    this._releaseMemClocks();
    this._releaseCoreOffsets();
    this.cores = [];
    this.paused.clear();
    this.hashrates.clear();
    if (this.sock) { try { this.sock.destroy(); } catch (e) { /* already closed */ } this.sock = null; }
    this.gzip = false;
    this.job = null;
    this.pending.clear();
    this._clearKTimers();
    this.emit('stopped', failed ? { failed: true } : {});
    return true;
  }
}

// How the log names a card: by index and name, or not at all for a core that
// will not say which card it opened.
// What was thrown, as text: an Error's message, or anything else as a string.
function messageOf(e) {
  return e instanceof Error ? e.message : String(e);
}

function cardName(device) {
  return device ? 'GPU ' + device.index + ' (' + device.name + ')' : 'the GPU';
}

// A byte count for the log, in KB: to a tenth under 10 KB, so a compressed
// proof of a few KB does not round to a misleading whole number.
function kb(bytes) {
  const k = bytes / 1024;
  return (k < 10 ? k.toFixed(1) : String(Math.round(k))) + ' KB';
}

// Only ever called for an auth-fail or submit-rejected, both of which carry a
// non-null error by construction — hence no null guard.
function errText(err) {
  return (err.code != null ? '[' + err.code + '] ' : '') + (err.message || '');
}

module.exports = {
  PearlMiner, RECONNECT_MS, RESTART_MS, RESTART_LIMIT, STABLE_MS, SALT_RUN, GZIP_REJECT_LIMIT,
  K_REJECT_LIMIT, K_ANSWER_MS,
};
