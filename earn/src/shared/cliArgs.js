'use strict';

// Pure argument parsing for the headless Linux CLI miner (src/cli/earn-cli.js).
// Turns a bare argv array into a validated settings object — the same shape the
// GUI's main process hands to MinerManager (address / worker / region /
// difficulty / backend), so the two share buildArgs, EngineManager, etc. Kept
// pure and dependency-free so it's fully unit-tested; the CLI shell wires the
// real IO (download, spawn, network reporting) around it.

const { REGIONS, DEFAULTS, difficultyForCard } = require('./config');
const { isValidAddress, isValidMdlAddress, normalizeAddress } = require('./address');
const { MODES, DEFAULT_MODE, isValidMode } = require('./llmMode');

// Short flags → their canonical long form.
const ALIASES = {
  '-a': '--address',
  '-m': '--mdl',
  '-r': '--region',
  '-w': '--worker',
  '-d': '--difficulty',
  '-g': '--gpu',
  '-b': '--binary',
  '-h': '--help',
  '-v': '--version',
};

// Options that consume a following value.
const VALUE_FLAGS = new Set([
  '--address', '--mdl', '--region', '--worker',
  '--difficulty', '--gpu', '--backend', '--binary', '--engine-dir',
  '--stats-file',
  '--mode', '--llm-binary', '--llm-model',
]);

function regionChoices() {
  return Object.keys(REGIONS).join(', ');
}

const USAGE = [
  'LLMJob Earn — headless Pearl (PRL) miner for Linux',
  '',
  'Usage: llmjob-earn-cli --address <prl1p…> [options]',
  '',
  'Required:',
  '  -a, --address <prl1p…>   Your Pearl payout address',
  '',
  'Options:',
  '  -m, --mdl <mdl1p…>       Also merge-mine ModelOS (MDL) on the same shares',
  '      --mode <mode>        Compute mode: ' + MODES.join('/') + ' (default: ' + DEFAULT_MODE + ').',
  '                           "both"/"auto" co-run a local LLM alongside mining;',
  '                           "llm" runs the LLM only (no payout address needed).',
  '      --llm-binary <path>  Path to a prebuilt llama-server binary. Optional:',
  '                           the CLI auto-downloads + extracts one (needs `unzip`)',
  '                           — use this to skip that or point at your own build.',
  '      --llm-model <path>   Path to a GGUF model file (default: download the',
  '                           bundled small model on first run)',
  '  -r, --region <id>        Pool region: ' + Object.keys(REGIONS).join('/') + ' (default: auto-detect fastest)',
  '  -w, --worker <name>      Worker/rig name (default: this machine\'s hostname)',
  '  -d, --difficulty <n>     Static share difficulty (default: from detected/--gpu card, else ' + DEFAULTS.difficulty + ')',
  '  -g, --gpu <card>         GPU name for the difficulty table (default: auto-detect via nvidia-smi)',
  '      --backend <name>     Force an engine backend (e.g. ampere)',
  '  -b, --binary <path>      Use this alpha-miner binary instead of downloading one',
  '      --engine-dir <path>  Where to cache the downloaded engine',
  '      --stats-file <path>  Write live stats JSON here every 10s (for HiveOS h-stats etc.)',
  '      --no-report          Do not publish live status to the public network board',
  '      --no-update          Do not auto-update the CLI to a newer release on start',
  '  -h, --help               Show this help and exit',
  '  -v, --version            Print the version and exit',
].join('\n');

// Fold the collected option map into a validated settings object, appending any
// validation problems to `errors`.
function buildSettings(opts, errors, report, update) {
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

  let region = DEFAULTS.region;
  if (opts['--region'] != null) {
    region = String(opts['--region']).trim();
    if (!REGIONS[region]) {
      errors.push('unknown region: ' + region + ' (choices: ' + regionChoices() + ')');
    }
  }

  const worker = opts['--worker'] != null ? String(opts['--worker']).trim() : DEFAULTS.worker;
  const gpu = opts['--gpu'] != null ? String(opts['--gpu']).trim() : null;

  let difficulty;
  if (opts['--difficulty'] != null) {
    difficulty = Number(opts['--difficulty']);
    if (!Number.isInteger(difficulty) || difficulty <= 0) {
      errors.push('invalid difficulty: ' + opts['--difficulty'] + ' (must be a positive integer)');
    }
  } else {
    difficulty = gpu ? difficultyForCard(gpu) : DEFAULTS.difficulty;
  }

  const backend = opts['--backend'] != null ? String(opts['--backend']).trim() : null;
  const binaryPath = opts['--binary'] != null ? String(opts['--binary']) : null;
  const engineDir = opts['--engine-dir'] != null ? String(opts['--engine-dir']) : null;
  const statsFile = opts['--stats-file'] != null ? String(opts['--stats-file']) : null;
  const llmBinary = opts['--llm-binary'] != null ? String(opts['--llm-binary']) : null;
  const llmModel = opts['--llm-model'] != null ? String(opts['--llm-model']) : null;

  // Which knobs the user set explicitly. The CLI auto-detects the ones left
  // unset (fastest region; GPU → static difficulty; a per-host worker name), so
  // it needs to tell an explicit `--region us2` / `--worker rig01` from the
  // default.
  const regionProvided = opts['--region'] != null;
  const gpuProvided = opts['--gpu'] != null;
  const difficultyProvided = opts['--difficulty'] != null;
  const workerProvided = opts['--worker'] != null;
  const modeProvided = opts['--mode'] != null;

  return {
    address, mdlAddress, region, worker, gpu, difficulty, backend, binaryPath, engineDir, statsFile,
    mode, llmBinary, llmModel,
    report, update, regionProvided, gpuProvided, difficultyProvided, workerProvided, modeProvided,
  };
}

// Parse a bare argv (typically process.argv.slice(2)) into:
//   { help, version, report, errors, settings }
// `settings` is null when --help/--version short-circuits. Never throws — bad
// input is reported via the `errors` array so the caller controls exit codes.
function parseCliArgs(argv) {
  const args = Array.isArray(argv) ? argv : [];
  const opts = {};
  const errors = [];
  let help = false;
  let version = false;
  let report = true;
  let update = true;

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
    if (flag === '--no-update') { update = false; continue; }

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

  if (help || version) {
    return { help, version, report, update, errors, settings: null };
  }

  const settings = buildSettings(opts, errors, report, update);
  return { help, version, report, update, errors, settings };
}

module.exports = { ALIASES, VALUE_FLAGS, USAGE, regionChoices, buildSettings, parseCliArgs };
