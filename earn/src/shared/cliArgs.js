'use strict';

// Pure argument parsing for the headless Linux CLI miner (src/cli/earn-cli.js).
// Turns a bare argv array into a validated settings object — the same shape the
// GUI's main process hands to the miner (address, worker, region, GPU), so the
// two share one settings shape. Kept pure and dependency-free so it's fully
// unit-tested; the CLI shell wires the real IO (network reporting, the engine)
// around it.

const { REGIONS, DEFAULTS } = require('./config');
const { isValidAddress, isValidMdlAddress, normalizeAddress } = require('./address');

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
  '--address', '--mdl', '--region', '--worker',
  '--gpu',
  '--stats-file',
]);

// The local-LLM options, retired with the LLM itself. Still ACCEPTED — value
// and all — for the same reason as --mdl: the CLI auto-updates on start, so a
// systemd unit or flight sheet written for an older build (`--mode auto`,
// `--no-serve`, `--gate-port 8000`) would otherwise update itself into a hard
// 'unknown option' exit and stop mining. They are ignored, and listed in
// settings.retired so the CLI can say so once at startup.
const RETIRED_VALUE_FLAGS = new Set([
  '--mode', '--llm-binary', '--llm-model', '--llm-max-instances',
  '--gate-port', '--gate-host', '--gate-quiet',
]);
const RETIRED_SWITCHES = new Set(['--no-serve']);

function regionChoices() {
  return Object.keys(REGIONS).join(', ');
}

const USAGE = [
  'LLMJob Earn — headless Pearl (PRL) miner for Linux',
  '',
  'Usage: llmjob-earn-cli --address <prl1p…> [options]',
  '       llmjob-earn-cli update                            Update the CLI to the latest release',
  '',
  'Required:',
  '  -a, --address <prl1p…>   Your Pearl payout address',
  '',
  'Options:',
  '  -r, --region <id>        Pool region: ' + Object.keys(REGIONS).join('/') + ' (default: auto-detect fastest)',
  '  -w, --worker <name>      Worker/rig name (default: this machine\'s hostname)',
  '  -g, --gpu <card>         GPU name to report on the board (default: auto-detect via nvidia-smi)',
  '      --stats-file <path>  Write live stats JSON here every 10s (for HiveOS h-stats etc.)',
  '      --no-report          Do not publish live status to the public network board',
  '      --no-update          Do not auto-update the CLI to a newer release on start',
  '  -h, --help               Show this help and exit',
  '  -v, --version            Print the version and exit',
].join('\n');

// Fold the collected option map into a validated settings object, appending any
// validation problems to `errors`.
function buildSettings(opts, errors, report, update, retired) {
  const address = opts['--address'] != null ? String(opts['--address']).trim() : '';
  if (!address) {
    // An old LLM-only unit (`--mode llm`) had no address because it never mined.
    // Say why it now needs one rather than just that it does.
    const wasLlmOnly = opts['--mode'] != null && String(opts['--mode']).trim() === 'llm';
    errors.push('--address is required (your prl1p… payout address)'
      + (wasLlmOnly ? '; --mode llm was retired and this build only mines' : ''));
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
  const statsFile = opts['--stats-file'] != null ? String(opts['--stats-file']) : null;

  // Which knobs the user set explicitly. The CLI auto-detects the ones left
  // unset (fastest region; a per-host worker name), so it needs to tell an
  // explicit `--region us2` / `--worker rig01` from the default.
  const regionProvided = opts['--region'] != null;
  const gpuProvided = opts['--gpu'] != null;
  const workerProvided = opts['--worker'] != null;

  return {
    address, mdlAddress, region, worker, gpu, statsFile,
    report, update, regionProvided, gpuProvided, workerProvided,
    retired: retired || [],
  };
}

// Parse a bare argv (typically process.argv.slice(2)) into:
//   { help, version, report, update, errors, settings }
// `settings` is null when --help/--version short-circuits. Never throws — bad
// input is reported via the `errors` array so the caller controls exit codes.
function parseCliArgs(argv) {
  const args = Array.isArray(argv) ? argv : [];
  const opts = {};
  const errors = [];
  const retired = [];
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
    if (RETIRED_SWITCHES.has(flag)) { retired.push(flag); continue; }

    const retiredValue = RETIRED_VALUE_FLAGS.has(flag);
    if (VALUE_FLAGS.has(flag) || retiredValue) {
      if (value == null) {
        const next = i + 1 < args.length ? String(args[i + 1]) : null;
        if (next == null || next.startsWith('-')) {
          // A retired flag with nothing after it is still harmless: there is no
          // value to lose, so ignore it rather than fail a unit over it.
          if (retiredValue) { retired.push(flag); continue; }
          errors.push('missing value for ' + flag);
          continue;
        }
        value = next;
        i++;
      }
      opts[flag] = value;
      if (retiredValue) retired.push(flag);
      continue;
    }

    errors.push('unknown option: ' + token);
  }

  if (help || version) {
    return { help, version, report, update, errors, settings: null };
  }

  const settings = buildSettings(opts, errors, report, update, retired);
  return { help, version, report, update, errors, settings };
}

module.exports = {
  ALIASES, VALUE_FLAGS, RETIRED_VALUE_FLAGS, RETIRED_SWITCHES, USAGE,
  regionChoices, buildSettings, parseCliArgs,
};
