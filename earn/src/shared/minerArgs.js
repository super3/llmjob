'use strict';

const { resolveEndpoint, DEFAULTS } = require('./config');
const { combinePayoutAddress } = require('./address');

// How often PeakMiner prints its status table. Its own default is 60s, which is
// far too slow for a UI that shows a live hashrate — the dashboard would sit
// blank for a minute after every start. Ten seconds matches roughly what
// alpha-miner emitted and is what the parser's cadence assumptions are built on.
const STATUS_INTERVAL_SECS = 10;

// Resolve the miner binary. A configured absolute path wins; otherwise the
// engine's own name for the platform. EngineManager installs under a versioned
// filename, so this is only the fallback used when nothing resolved a full path.
function resolveBinary(binaryPath, platform) {
  if (binaryPath) return binaryPath;
  return platform === 'win32' ? 'peakminer.exe' : 'peakminer';
}

// Build the PeakMiner argument vector.
//
//   -c pearl                 the coin, and the only one we ever ask for.
//   -o stratum+tcp://host:port
//                            PeakMiner parses a scheme-qualified URL and detects
//                            TLS on its own. The endpoint is passed WHOLE rather
//                            than split into host and port flags — the opposite
//                            of what alpha-miner needed, where a combined
//                            `--host host:port` was re-appended to a default
//                            port by some builds and produced
//                            `pool us2.alphapool.tech:5566:5566`, then looped on
//                            "DNS lookup failed". PeakMiner has one URL argument
//                            and no port flag, so that failure mode cannot recur
//                            here. resolveEndpoint still normalises away any
//                            scheme a user pasted into an override, so we never
//                            emit `stratum+tcp://stratum+tcp://…`.
//   -u <address>             the payout address, sent VERBATIM — upstream's help
//                            is emphatic that it is never split. Merge mining
//                            rides here unchanged: HeroMiners documents its MDL
//                            login as `PRL+MDL.WORKER`, which is exactly what
//                            combinePayoutAddress already produces.
//   -w <worker>              a SEPARATE field, not an edit of -u. PeakMiner
//                            appends `.<worker>` for single-string logins and
//                            ignores it when -u already carries a `.` or `/`,
//                            so passing both is safe in either dialect.
//   -p x;d=N                 static difficulty, passed through verbatim.
//                            HeroMiners vardiffs and ignores it (see
//                            DEFAULTS.difficulty) but a pool reached through an
//                            endpoint override may honour it.
//   -d <id>                  one card by index; PeakMiner's default is `all`,
//                            which on a multi-GPU rig would quietly enlist every
//                            card the user did not select.
//   -i N                     status-table interval — see STATUS_INTERVAL_SECS.
//   --no-tips                suppress the periodic usage tips printed under the
//                            table. Upstream documents this flag for exactly our
//                            case: "a log scraper that wants nothing but the
//                            table can turn them off here".
//   -a 0                     disable the HTTP stats API. It defaults to ON,
//                            listening on 127.0.0.1:4068 — a listening socket
//                            the user never asked for, and a port collision
//                            between two rigs on one box. We read the log, so we
//                            do not need it.
//
// No backend override: PeakMiner selects a kernel profile from the card's
// compute capability and takes no such option — that is what backendForEngine
// strips before it can reach here.
function buildArgs(settings = {}) {
  // resolveEndpoint, not a raw read: an override pasted in `stratum+tcp://…`
  // form must not end up double-schemed once we prepend our own.
  const endpoint = resolveEndpoint(settings);
  const worker = settings.worker != null ? settings.worker : DEFAULTS.worker;
  const difficulty = settings.difficulty || DEFAULTS.difficulty;
  const address = combinePayoutAddress(settings.address, settings.mdlAddress);
  const gpu = settings.gpuIndex == null ? 0 : settings.gpuIndex;

  const args = ['-c', 'pearl', '-o', 'stratum+tcp://' + endpoint, '-u', address];
  if (worker) args.push('-w', worker);
  return args.concat([
    '-p', 'x;d=' + difficulty,
    '-d', String(gpu),
    '-i', String(STATUS_INTERVAL_SECS),
    '--no-tips',
    '-a', '0',
  ]);
}

module.exports = { STATUS_INTERVAL_SECS, resolveBinary, buildArgs };
