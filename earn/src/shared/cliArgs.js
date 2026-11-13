'use strict';

// Pure argument parsing for the headless Linux CLI miner (src/cli/earn-cli.js).
// Turns a bare argv array into a validated settings object, the same shape the
// GUI's main process builds (address, worker, region, GPU). Kept pure and
// dependency-free so it's fully unit-tested; the CLI shell wires the real IO
// (download, spawn, network reporting) around it.

const { REGIONS, LEGACY_REGIONS, DEFAULTS, splitEndpoint } = require('./config');
const { isValidAddress, isValidMdlAddress, normalizeAddress } = require('./address');
const { MODES, DEFAULT_MODE, isValidMode } = require('./llmMode');
const { parseMemClockMhz } = require('./memClock');
const { parseCoreOffset } = require('./coreOffset');
const { parseGpuIndexList } = require('./gpu');
const { checkXmrAddress } = require('./xmr/address');
const { PRIORITIES } = require('./xmr/worker');
const { CU12, CU13, AMD, INTEL } = require('./coreVariant');
const { flightSheetPool } = require('./hiveos');
const { CHECK_HITS: FOLD_CHECK_HITS } = require('./foldCheck');

// --core's choices, the values PEARL_CORE_VARIANT takes (shared/coreVariant).
const CORES = [CU12, CU13, AMD, INTEL];

// --gpu-wait's choices, the values PEARL_WAIT takes (native/src/pearl_host.cu): how a
// card's search thread waits for the GPU between launches.
const GPU_WAITS = ['block', 'poll', 'spin'];

// Short flags → their canonical long form.
// --mdl / -m are deliberately absent from USAGE: merge mining is retired from
// the UI and undocumented, but still parsed so an existing HiveOS flight sheet
// carrying it keeps mining instead of dying on 'unknown option'.
const ALIASES = {
  '-a': '--address',
  '-m': '--mdl',
  '-r': '--region',
  '-w': '--worker',
  '-g': '--gpu',

  '-h': '--help',
  '-v': '--version',
};

// Options that consume a following value.
const VALUE_FLAGS = new Set([
  '--address', '--mdl', '--region', '--pool', '--hive-pool-url', '--worker',
  '--gpu', '--gpu-index', '--core', '--gpu-wait',
  '--stats-file',
  '--mode', '--llm-binary', '--llm-model', '--llm-max-instances', '--gate-port', '--gate-host',
  '--gate-quiet', '--mine-mem-clock', '--mine-core-offset',
  '--xmr-address', '--xmr-pool', '--cpu-threads', '--cpu-priority',
]);

// Own keys only, so `--region constructor` is not a region.
function has(table, key) {
  return Object.prototype.hasOwnProperty.call(table, key);
}

function regionChoices() {
  return Object.keys(REGIONS).join(', ');
}

const USAGE = [
  'LLMJob Earn — headless Pearl (PRL) miner for Linux',
  '',
  'Usage: llmjob-earn-cli --address <prl1p…> [options]',
  '       llmjob-earn-cli connect --token <pairing-token>   Link this box to your LLMJob account',
  '       llmjob-earn-cli update                            Update the CLI to the latest release',
  '',
  'Required:',
  '  -a, --address <prl1p…>   Your Pearl payout address',
  '',
  'Options:',
  '      --mode <mode>        Compute mode: ' + MODES.join('/') + ' (default: ' + DEFAULT_MODE + ').',
  '                           "auto" mines and serves a local LLM, "mining" only',
  '                           mines, "llm" only serves (no payout address needed).',
  '                           "both" is the old name for "auto" and still works.',
  '      --llm-binary <path>  Path to a prebuilt llama-server binary. Optional:',
  '                           the CLI downloads one and unpacks it with `tar`',
  '                           — use this to skip that or point at your own build.',
  '      --llm-model <path>   Path to a GGUF model file (default: download the',
  '                           bundled small model on first run)',
  '      --gate-port <port>   Port the LLM gate serves on (default: 8000). The gate',
  '                           runs in llm mode, and in auto mode unless the LLM',
  '                           co-runs with the miner.',
  '      --gate-quiet <secs>  Seconds of no requests before the GPU goes back to',
  '                           mining (default: 60). Raise it for agent workloads:',
  '                           a release drops the prompt cache, so the next turn',
  '                           re-prefills the whole context.',
  '      --gate-host <addr>   Address the gate binds (default: 0.0.0.0, every',
  '                           interface). Set 127.0.0.1 to keep it to this box.',
  '      --llm-max-instances <n>  Cap how many llama-servers run (default: one per',
  '                           eligible GPU, itself capped by free system RAM)',
  '      --mine-mem-clock <MHz>  Lock each mining GPU\'s memory clock to <MHz>',
  '                           while it mines, with no trial. Default: an RTX',
  '                           5090 tries 7001 for five minutes and keeps it',
  '                           only if it mines faster, and every other card',
  '                           is left alone. 0 leaves the driver\'s clock',
  '                           everywhere; auto runs that trial on RTX 40 and',
  '                           RTX 50 cards. A trial\'s answer is remembered',
  '                           for a week. The fold barely touches DRAM,',
  '                           so a power-capped card spends the watts on its SM',
  '                           clock instead: 7001 took a 600 W RTX 5090 from',
  '                           95.7 to 103.4 TH/s (+8%). Only one',
  '                           5090 has been measured; lower raises the clock',
  '                           but cuts the work done per clock. Needs root, or',
  '                           a sudoers NOPASSWD rule for nvidia-smi. Released',
  '                           whenever mining stops; skipped while an LLM',
  '                           co-runs with it.',
  '      --mine-core-offset <auto|0>  auto tries a core clock offset on each',
  '                           RTX 20, 30, 40 or 50 card while it mines: +50,',
  '                           +100, then +150 MHz. A step is tried only after',
  '                           100 of its hits are recomputed right, and the',
  '                           fastest step that is right is kept.',
  '                           Default: 0, off. It runs the card above the',
  '                           clock its maker set, and only one RTX 4090 has',
  '                           been measured (+4.3% at +150 MHz, at 450 W): an',
  '                           unstable step can crash the driver or the PC',
  '                           before any check sees it. Needs root, or',
  '                           administrator rights on Windows. Set back to 0',
  '                           whenever mining stops; skipped while an LLM',
  '                           co-runs with it.',
  '  -r, --region <id>        Pool region: ' + Object.keys(REGIONS).join('/') + ' (default: auto-detect fastest)',
  '                           An old AlphaPool id (eu1, sg1, …) maps to the nearest.',
  '      --pool <host:port>   Mine Pearl on this stratum endpoint instead of a',
  '                           HeroMiners region, such as',
  '                           prl-us.kryptex.network:7048. It must use the same',
  '                           protocol as HeroMiners. --xmr-pool moves Monero.',
  '      --hive-pool-url <url>  The flight sheet\'s Pool URL, which the HiveOS package',
  '                           passes on. A HeroMiners address leaves the region',
  '                           to the miner. Another host:port is mined on, as',
  '                           with --pool, which beats it.',
  '  -w, --worker <name>      Worker/rig name (default: this machine\'s hostname)',
  '  -g, --gpu <card>         GPU name to report on the board (default: auto-detect via nvidia-smi)',
  '      --gpu-index <list>   Mine only on these GPUs: nvidia-smi indices such as',
  '                           0,2, or "none" for no GPU (default: every GPU). The',
  '                           local LLM picks its own cards. PEARL_GPU_INDEX takes',
  '                           the same list; the flag wins.',
  '      --core <build>       Which Pearl core to load: amd for AMD Radeon and',
  '                           Instinct cards (Linux, needs the ROCm 7 HIP',
  '                           runtime), intel for Intel Arc cards (needs',
  '                           Intel\'s Level Zero or OpenCL driver), or',
  '                           cu12/cu13 to force a CUDA build. Default: the',
  '                           CUDA build that fits the NVIDIA cards; when',
  '                           nvidia-smi lists none, the Intel core on a rig',
  '                           of Arc cards, else the AMD core first.',
  '                           PEARL_CORE_VARIANT takes the same values; the',
  '                           flag wins.',
  '      --gpu-wait <how>     How the miner waits for an NVIDIA card between',
  '                           launches: block sleeps until the driver wakes it',
  '                           (default, almost no CPU), poll sleeps and looks',
  '                           every half millisecond, spin busy-waits and takes',
  '                           a whole CPU core a card. PEARL_WAIT takes the same',
  '                           values; the flag wins.',
  '      --no-rotation-fold   Turn off the rotation fold, the second kernel RTX',
  '                           30, 40 and 50 cards mine on by default. It was',
  '                           2.7% to 9.7% faster than the normal kernel on',
  '                           every RTX 30, 40 and 50 PC we measured (Salad,',
  '                           12-13 November), and it uses about 1.3 GB more',
  '                           card memory. Each card checks ' + FOLD_CHECK_HITS + ' of the',
  '                           fold\'s hits on the CPU before it trusts it, and',
  '                           every share after that. If the card is not an RTX',
  '                           30, 40 or 50, a file is missing, memory is short',
  '                           or a check fails, the card mines on the normal',
  '                           kernel. In the desktop app: LLMJOB_ROTATION_FOLD=0',
  '                           in its environment; PEARL_FOLD_ROT=0 works for',
  '                           both. --rotation-fold turns it on over that.',

  '      --xmr-address <4…|8…>  Also mine Monero (XMR) on the CPU, paid to this',
  '                           address (a standard 4… address or an 8… subaddress).',
  '                           Off without it. Mines on HeroMiners, 0.9% pool fee,',
  '                           unless --xmr-pool says otherwise.',
  '      --xmr-pool <host:port>  Mine Monero on this stratum endpoint instead of',
  '                           HeroMiners, such as xmr-us.kryptex.network:7029.',
  '      --cpu-threads <n>    CPU threads for Monero (default: every CPU this',
  '                           process may use, less 1, or 2 on 6 CPUs or fewer,',
  '                           and no more than 1 per 2 MB of L3 cache; off by',
  '                           default on 4 CPUs or fewer or under 8 GB).',
  '                           At least one CPU always stays free for the GPU',
  '                           miner. 0 turns CPU mining off.',
  '      --cpu-priority <p>   Priority of the Monero threads: ' + PRIORITIES.join(', '),
  '                           (default: low on Linux, idle on Windows). One CPU',
  '                           stays free for the GPU miner at any priority;',
  '                           normal gets the most from a server whose CPUs',
  '                           are busy with other work.',
  '      --stats-file <path>  Write live stats JSON here every 10s (for HiveOS h-stats etc.)',
  '      --no-report          Do not publish live status to the public network board',
  '      --no-serve           Do not serve inference jobs for the LLMJob network',
  '                           (by default a rig serves public jobs even when it is',
  '                           not linked to an account; linking adds private queues)',
  '      --no-update          Do not auto-update the CLI to a newer release on start',
  '  -h, --help               Show this help and exit',
  '  -v, --version            Print the version and exit',
].join('\n');

// Fold the collected option map into a validated settings object, appending any
// validation problems to `errors`.
function buildSettings(opts, errors, report, update, serve) {
  // Compute mode (which engines run). Parsed before the address check because
  // LLM-only doesn't mine, so it needs no payout address.
  let mode = DEFAULT_MODE;
  if (opts['--mode'] != null) {
    mode = String(opts['--mode']).trim();
    if (!isValidMode(mode)) {
      errors.push('unknown mode: ' + mode + ' (choices: ' + MODES.join(', ') + ')');
    }
  }

  const address = opts['--address'] != null ? String(opts['--address']).trim() : '';
  if (!address) {
    // The address is only required when the mode actually mines. "llm" is
    // LLM-only, so it can run with no payout address.
    if (mode !== 'llm') errors.push('--address is required (your prl1p… payout address)');
  } else if (!isValidAddress(address)) {
    errors.push('invalid Pearl address: ' + address);
  }

  let mdlAddress = null;
  if (opts['--mdl'] != null) {
    const m = normalizeAddress(opts['--mdl']);
    if (isValidMdlAddress(m)) mdlAddress = m;
    else errors.push('invalid MDL address: ' + opts['--mdl']);
  }

  // An AlphaPool region id (eu1, sg1, ...) from an old flight sheet or service
  // file maps to the nearest HeroMiners region, and `legacyRegion` keeps what
  // was typed so the CLI can say so. Anything else is an error rather than the
  // default: a typo should be loud, not mine on another continent.
  // config.migrateRegion falls back to the default, so it is not used here.
  let region = DEFAULTS.region;
  let legacyRegion = null;
  if (opts['--region'] != null) {
    region = String(opts['--region']).trim();
    if (!has(REGIONS, region)) {
      if (has(LEGACY_REGIONS, region)) {
        legacyRegion = region;
        region = LEGACY_REGIONS[region];
      } else {
        errors.push('unknown region: ' + region + ' (choices: ' + regionChoices() + ')');
      }
    }
  }

  // Another pool for Pearl. Both parts are required: a pool has no default
  // port, and a missing one should be loud, not a connect to the wrong port.
  let endpoint = null;
  if (opts['--pool'] != null) {
    const { host, port } = splitEndpoint(opts['--pool']);
    if (host && Number.isInteger(port) && port >= 1 && port <= 65535) endpoint = host + ':' + port;
    else errors.push('invalid --pool: ' + opts['--pool'] + ' (expected host:port, such as prl-us.kryptex.network:7048)');
  }
  // Where the endpoint came from, for the log: '--pool', 'pool-url' or null.
  let endpointFrom = endpoint ? '--pool' : null;

  // The flight sheet's Pool URL under HiveOS. --pool beats it. A HeroMiners
  // address or one that can't be used changes nothing, and is never an error:
  // the field is filled in on every rig (see shared/hiveos flightSheetPool).
  let poolUrl = null;
  if (opts['--hive-pool-url'] != null) {
    poolUrl = flightSheetPool(opts['--hive-pool-url']);
    if (opts['--pool'] == null && poolUrl.kind === 'pool') {
      endpoint = poolUrl.endpoint;
      endpointFrom = 'pool-url';
    }
  }

  const worker = opts['--worker'] != null ? String(opts['--worker']).trim() : DEFAULTS.worker;
  const gpu = opts['--gpu'] != null ? String(opts['--gpu']).trim() : null;

  // Which cards mine: null for every card, [] for none, else nvidia-smi
  // indices. Blank is an error rather than "every card": HiveOS's h-config.sh
  // writes this flag, and a value lost on the way should be loud.
  let gpuIndices = null;
  if (opts['--gpu-index'] != null) {
    const parsed = parseGpuIndexList(opts['--gpu-index']);
    if (!parsed || parsed.error) {
      errors.push('invalid --gpu-index: ' + opts['--gpu-index'] + ' ('
        + (parsed ? parsed.error : 'must not be empty') + ')');
    } else {
      gpuIndices = parsed.indices;
    }
  }

  // Which core loads. Null leaves it to PEARL_CORE_VARIANT and then to
  // shared/coreVariant's pick. A flag because HiveOS's Extra config takes
  // flags and no environment variables, and an AMD rig has to ask for its core.
  let core = null;
  if (opts['--core'] != null) {
    const c = String(opts['--core']).trim().toLowerCase();
    if (CORES.includes(c)) core = c;
    else errors.push('invalid --core: ' + opts['--core'] + ' (choices: ' + CORES.join(', ') + ')');
  }

  // How a card's search thread waits between launches. Null leaves it to PEARL_WAIT
  // and then to the core's default. A flag for the same reason --core is one.
  let gpuWait = null;
  if (opts['--gpu-wait'] != null) {
    const w = String(opts['--gpu-wait']).trim().toLowerCase();
    if (GPU_WAITS.includes(w)) gpuWait = w;
    else errors.push('invalid --gpu-wait: ' + opts['--gpu-wait'] + ' (choices: ' + GPU_WAITS.join(', ') + ')');
  }

  // Monero on the CPU (shared/xmr). Off unless an address is given; the thread
  // count is null for the default, which shared/xmr/cpuPlan works out from
  // the CPUs this process may use.
  // A CPU flag with no address is a warning, not an error. A HiveOS Extra
  // config whose address was removed still carries them, and an error there
  // stopped Pearl too, the CLI exiting and restarting every few seconds.
  const warnings = [];
  const noCpuMining = (flag) => flag + ' does nothing without --xmr-address: no CPU mining';
  let xmrAddress = null;
  if (opts['--xmr-address'] != null) {
    const a = String(opts['--xmr-address']).trim();
    const check = checkXmrAddress(a);
    if (check.ok) xmrAddress = a;
    else errors.push('invalid Monero address: ' + a + ' (' + check.error + ')');
  }
  let cpuThreads = null;
  if (opts['--cpu-threads'] != null) {
    const raw = String(opts['--cpu-threads']).trim();
    cpuThreads = raw === '' ? NaN : Number(raw);
    if (!Number.isInteger(cpuThreads) || cpuThreads < 0) {
      errors.push('invalid --cpu-threads: ' + opts['--cpu-threads'] + ' (must be 0 or more)');
      cpuThreads = null;
    } else if (cpuThreads > 0 && opts['--xmr-address'] == null) {
      warnings.push(noCpuMining('--cpu-threads'));
    }
  }

  // Another pool for Monero, built like --pool. Without an address there is no
  // Monero to move, which is a warning like the other CPU flags.
  let xmrEndpoint = null;
  if (opts['--xmr-pool'] != null) {
    const { host, port } = splitEndpoint(opts['--xmr-pool']);
    if (!(host && Number.isInteger(port) && port >= 1 && port <= 65535)) {
      errors.push('invalid --xmr-pool: ' + opts['--xmr-pool'] + ' (expected host:port, such as xmr-us.kryptex.network:7029)');
    } else if (opts['--xmr-address'] == null) {
      warnings.push(noCpuMining('--xmr-pool'));
    } else {
      xmrEndpoint = host + ':' + port;
    }
  }

  // null: the platform's default (shared/xmr/worker.defaultPriority).
  let cpuPriority = null;
  if (opts['--cpu-priority'] != null) {
    const p = String(opts['--cpu-priority']).trim();
    if (!PRIORITIES.includes(p)) errors.push('invalid --cpu-priority: ' + p + ' (choices: ' + PRIORITIES.join(', ') + ')');
    else if (opts['--xmr-address'] == null) warnings.push(noCpuMining('--cpu-priority'));
    else cpuPriority = p;
  }

  // The rotation fold (shared/foldCheck): a second kernel for RTX 30, 40 and 50
  // cards, on by default. A flag for the reason --core is one; the CUDA core reads
  // PEARL_FOLD_ROT, which the CLI sets from this (shared/rotationFold). true asked
  // for, false turned off, null nothing said (on, unless the rig set
  // PEARL_FOLD_ROT=0 itself).
  let rotationFold = null;
  if (opts['--rotation-fold'] === true && opts['--no-rotation-fold'] === true) {
    errors.push('--rotation-fold and --no-rotation-fold together: pick one');
  } else if (opts['--rotation-fold'] === true) rotationFold = true;
  else if (opts['--no-rotation-fold'] === true) rotationFold = false;

  const statsFile = opts['--stats-file'] != null ? String(opts['--stats-file']) : null;
  const llmBinary = opts['--llm-binary'] != null ? String(opts['--llm-binary']) : null;
  const llmModel = opts['--llm-model'] != null ? String(opts['--llm-model']) : null;

  let gatePort = null;
  if (opts['--gate-port'] != null) {
    gatePort = Number(opts['--gate-port']);
    if (!Number.isInteger(gatePort) || gatePort < 0 || gatePort > 65535) {
      errors.push('invalid --gate-port: ' + opts['--gate-port'] + ' (must be 0-65535)');
    }
  }
  // Seconds in, milliseconds out: the flag is in the unit an operator thinks in,
  // the gate is in the unit it compares against. 0 is allowed and means "never
  // release", which is what a rig dedicated to inference wants.
  let gateQuietMs = null;
  if (opts['--gate-quiet'] != null) {
    // Empty is rejected rather than read as 0: Number('') is 0, so a cleared
    // setting would silently mean "never release the card", which is the
    // opposite of harmless.
    const raw = String(opts['--gate-quiet']).trim();
    const secs = raw === '' ? NaN : Number(raw);
    if (!Number.isFinite(secs) || secs < 0) {
      errors.push('invalid --gate-quiet: ' + opts['--gate-quiet'] + ' (must be 0 or more seconds)');
    } else {
      gateQuietMs = secs === 0 ? Infinity : Math.round(secs * 1000);
    }
  }
  // Empty string is rejected rather than silently meaning "all interfaces":
  // `--gate-host ""` reading as 0.0.0.0 would be the opposite of what someone
  // clearing the setting expects.
  let gateHost = null;
  if (opts['--gate-host'] != null) {
    gateHost = String(opts['--gate-host']).trim();
    if (!gateHost) errors.push('invalid --gate-host: must not be empty');
  }
  // A memory clock to lock while mining, in MHz, with no trial. Null (not
  // given) means the RTX 5090 default applies (shared/memClock): each 5090
  // tries a lock and keeps it only if it is faster. 0 turns that off and leaves
  // every card at the driver's clock; 'auto' tries a lock on the cards that
  // have one to try and keeps it where it is faster. The value's rules, and the GUI's
  // environment variable that takes the same values, are shared/memClock's.
  let mineMemClockMhz = null;
  if (opts['--mine-mem-clock'] != null) {
    const parsed = parseMemClockMhz(opts['--mine-mem-clock']);
    if (parsed.error) {
      errors.push('invalid --mine-mem-clock: ' + opts['--mine-mem-clock'] + ' (' + parsed.error + ')');
    } else {
      mineMemClockMhz = parsed.mhz;
    }
  }
  // Whether to try a core clock offset on the mining cards: 'auto' runs the
  // trial, 0 and null (not given) leave every card at its own clock. There is
  // no MHz to ask for: an offset is only set once the trial has checked it on
  // that card (shared/coreOffset, which also has the GUI's variable).
  let mineCoreOffset = null;
  if (opts['--mine-core-offset'] != null) {
    const parsed = parseCoreOffset(opts['--mine-core-offset']);
    if (parsed.error) {
      errors.push('invalid --mine-core-offset: ' + opts['--mine-core-offset'] + ' (' + parsed.error + ')');
    } else {
      mineCoreOffset = parsed.offset;
    }
  }
  // An explicit ceiling on concurrent llama-servers. Null means "no operator
  // opinion" — the planner then caps by eligible GPUs and free RAM on its own.
  let llmMaxInstances = null;
  if (opts['--llm-max-instances'] != null) {
    llmMaxInstances = Number(opts['--llm-max-instances']);
    if (!Number.isInteger(llmMaxInstances) || llmMaxInstances < 1) {
      errors.push('invalid --llm-max-instances: ' + opts['--llm-max-instances'] + ' (must be a positive integer)');
    }
  }

  // Which knobs the user set explicitly. The CLI auto-detects the ones left
  // unset (fastest region; a per-host worker name), so
  // it needs to tell an explicit `--region us2` / `--worker rig01` from the
  // default.
  const regionProvided = opts['--region'] != null;
  const gpuProvided = opts['--gpu'] != null;
  const workerProvided = opts['--worker'] != null;
  const modeProvided = opts['--mode'] != null;

  return {
    address, mdlAddress, region, legacyRegion, endpoint, endpointFrom, poolUrl, worker, gpu, gpuIndices, core, gpuWait, statsFile, xmrAddress, xmrEndpoint, cpuThreads, cpuPriority,
    mode, llmBinary, llmModel, llmMaxInstances, gatePort, gateHost, gateQuietMs, mineMemClockMhz, mineCoreOffset,
    rotationFold,
    report, update, serve: serve !== false, regionProvided, gpuProvided, workerProvided, modeProvided,
    warnings,
  };
}

// Parse a bare argv (typically process.argv.slice(2)) into:
//   { help, version, errors, settings }
// `settings` is null when --help/--version short-circuits. Never throws — bad
// input is reported via the `errors` array so the caller controls exit codes.
// --no-report, --no-update and --no-serve land in settings.
function parseCliArgs(argv) {
  const args = Array.isArray(argv) ? argv : [];
  const opts = {};
  const errors = [];
  let help = false;
  let version = false;
  let report = true;
  let update = true;
  let serve = true;

  for (let i = 0; i < args.length; i++) {
    let token = String(args[i]);
    let value = null;

    // Support --flag=value.
    const eq = token.indexOf('=');
    if (token.startsWith('--') && eq !== -1) {
      value = token.slice(eq + 1);
      token = token.slice(0, eq);
    }

    const flag = ALIASES[token] || token;

    if (flag === '--help') { help = true; continue; }
    if (flag === '--version') { version = true; continue; }
    if (flag === '--no-report') { report = false; continue; }
    if (flag === '--no-serve') { serve = false; continue; }
    if (flag === '--no-update') { update = false; continue; }
    if (flag === '--rotation-fold' || flag === '--no-rotation-fold') {
      // A switch: `--rotation-fold=0` must not read as "on".
      if (value != null) errors.push(flag + ' takes no value');
      else opts[flag] = true;
      continue;
    }

    if (VALUE_FLAGS.has(flag)) {
      if (value == null) {
        const next = i + 1 < args.length ? String(args[i + 1]) : null;
        if (next == null || next.startsWith('-')) {
          errors.push('missing value for ' + flag);
          continue;
        }
        value = next;
        i++;
      }
      opts[flag] = value;
      continue;
    }

    errors.push('unknown option: ' + token);
  }

  if (help || version) return { help, version, errors, settings: null };

  const settings = buildSettings(opts, errors, report, update, serve);
  return { help, version, errors, settings };
}

module.exports = { ALIASES, VALUE_FLAGS, USAGE, regionChoices, buildSettings, parseCliArgs };
