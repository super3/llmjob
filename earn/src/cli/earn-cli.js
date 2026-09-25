#!/usr/bin/env node
'use strict';

// Headless LLMJob Earn miner for Linux — the command-line counterpart to the
// Electron GUI. It shares all the real logic with the desktop app (config,
// address handling, the Pearl engine and stats accumulator); this file is only
// the thin IO shell that wires the real filesystem / network around them,
// exactly like main.js does for the GUI. No window, no DOM — just stdout.

const fs = require('fs');

const { parseCliArgs, USAGE } = require('../shared/cliArgs');
const selfUpdater = require('./selfUpdater');
const { planUpdate } = require('../shared/selfUpdate');
const net = require('net');
const { PearlEngine } = require('../main/pearlEngine');
const { coreFactory } = require('../main/pearlCore');
const { detectRegion, detectGpusVram, detectMinerGpus, postMinerReport } = require('../main/probe');
const probe = require('../main/probe');
const { initStats, applyEvent, snapshot } = require('../shared/miningStats');
const { NETWORK, resolveEndpoint, regionLabel } = require('../shared/config');
const { defaultWorker } = require('../shared/worker');
const { buildMinerReports } = require('../shared/minerReport');
const { statsFilePayload } = require('../shared/statsFile');
const { shortenAddress } = require('../shared/address');
const { minerUnsupportedNote } = require('../shared/platform');
const { alignCudaDeviceOrder, clearCudaVisibleDevices, describeClearedCuda } = require('../shared/gpu');
const format = require('../shared/format');
const pkg = require('../../package.json');

// Number the GPUs the way nvidia-smi does, before anything opens a CUDA device.
// Everything here — the device label, per-card VRAM, temperatures, the board's
// rows — speaks nvidia-smi's indices, and the CUDA runtime does not unless told
// to. Set at load, because the mining core initialises CUDA inside THIS process
// and reads it then. For the same reason CUDA must see every card nvidia-smi
// lists; the removed value is logged when a mining run starts.
alignCudaDeviceOrder(process.env);
const clearedCudaLine = describeClearedCuda(clearCudaVisibleDevices(process.env));

// The shortest gap between two mining status lines. See the miner event handler.
const MINE_LOG_MS = 1000;

// Write a log line. When attached to a TTY we prefix a wall-clock time; when
// piped (systemd/journald, `docker logs`, a file) we drop it, since the log
// collector adds its own timestamp and two would just be noise.
function log(line, stream) {
  const out = stream || process.stdout;
  const prefix = out.isTTY ? '[' + format.formatLogTime(new Date()) + '] ' : '';
  out.write(prefix + line + '\n');
}

// Resolve { name, count } — the representative card plus how many discrete
// GPUs the rig actually mines with. Never rejects — the engine still finds the
// real devices to mine; this is only for the status label.
// Delegates to the shared probe so the GUI and the CLI detect the same way —
// they had drifted into two different methods, and the GUI's was Windows-only.
function detectGpu() {
  return probe.detectGpuInfo();
}

// Explicit `llmjob-earn-cli update` — check the latest release and, if this is
// the packaged binary, replace it in place.
async function runExplicitUpdate() {
  log('checking for updates (current v' + pkg.version + ')…');
  const release = await selfUpdater.fetchLatestRelease();
  if (!release) { log('could not reach the update server', process.stderr); return 1; }

  const plan = planUpdate({ currentVersion: pkg.version, release, platform: process.platform });
  if (!plan.updateAvailable) {
    if (plan.reason === 'up-to-date') log('already up to date (v' + pkg.version + ')');
    else if (plan.reason === 'asset-missing') log('v' + plan.latestVersion + ' is out but has no Linux CLI binary yet', process.stderr);
    else if (plan.reason === 'unsupported-platform') log('self-update is only available for the Linux binary', process.stderr);
    else log('no newer release found', process.stderr);
    return 0;
  }

  if (!selfUpdater.isPackaged()) {
    log('v' + plan.latestVersion + ' is available (you have v' + plan.currentVersion + ').');
    log('this is running from source — update via git/npm, or download: ' + plan.downloadUrl);
    return 0;
  }

  log('updating ' + plan.currentVersion + ' → ' + plan.latestVersion + ' …');
  try {
    const exe = await selfUpdater.applyUpdate(plan);
    log('updated to v' + plan.latestVersion + ' (' + exe + '). Re-run to use it.');
    return 0;
  } catch (e) {
    log('update failed: ' + e.message, process.stderr);
    return 1;
  }
}

// Best-effort auto-update on start. Returns an exit code when it replaced and
// re-ran the binary (caller should return it), or null to keep mining.
async function maybeAutoUpdate(argv) {
  if (process.env[selfUpdater.UPDATED_ENV]) return null; // already the updated child
  const release = await selfUpdater.fetchLatestRelease();
  if (!release) return null; // offline — never block mining

  const plan = planUpdate({ currentVersion: pkg.version, release, platform: process.platform });
  if (!plan.updateAvailable) return null;

  if (!selfUpdater.isPackaged()) {
    log('a newer release is available: v' + plan.latestVersion + ' (run "llmjob-earn-cli update")');
    return null;
  }

  log('updating ' + plan.currentVersion + ' → ' + plan.latestVersion + ' before starting…');
  try {
    await selfUpdater.applyUpdate(plan);
    log('updated to v' + plan.latestVersion + '; restarting');
    return selfUpdater.reexec(argv);
  } catch (e) {
    log('auto-update failed (' + e.message + '); continuing on v' + pkg.version, process.stderr);
    return null;
  }
}

async function run(argv) {
  if (argv[0] === 'update') return runExplicitUpdate();
  // `connect` linked a box to an account so it could serve LLM jobs. Both went
  // with the LLM; say so rather than report it as an unknown option.
  if (argv[0] === 'connect') {
    log('"connect" was retired with the local LLM — there is no account to link a rig to.'
      + ' To mine, run: llmjob-earn-cli --address <prl1p…>', process.stderr);
    return 1;
  }

  const parsed = parseCliArgs(argv);

  if (parsed.help) { process.stdout.write(USAGE + '\n'); return 0; }
  if (parsed.version) { process.stdout.write(pkg.version + '\n'); return 0; }

  if (parsed.errors.length) {
    for (const e of parsed.errors) log('error: ' + e, process.stderr);
    log('run with --help for usage', process.stderr);
    return 1;
  }

  const settings = parsed.settings;

  if (settings.update) {
    const code = await maybeAutoUpdate(argv);
    if (code != null) return code;
  }

  log('LLMJob Earn CLI v' + pkg.version);
  // Said once, so a unit written for an older build explains itself in the
  // journal instead of silently ignoring half its command line.
  if (settings.retired.length) {
    log('ignoring retired option' + (settings.retired.length > 1 ? 's' : '') + ': '
      + settings.retired.join(', ') + ' (the local LLM was removed; this build only mines)', process.stderr);
  }
  // macOS has no mining engine at all (see shared/platform), and with the LLM
  // gone there is nothing else to run.
  const platformNote = minerUnsupportedNote(process.platform);
  if (platformNote) {
    log(platformNote, process.stderr);
    return 1;
  }

  // Auto-detect the knobs the user didn't pin. Best-effort: any failure falls
  // back to the defaults already in `settings` and never blocks mining. Explicit
  // --region / --gpu always win.
  if (!settings.workerProvided) settings.worker = defaultWorker();
  if (!settings.regionProvided) settings.region = await detectRegion();
  // Probe regardless of --gpu. Naming the card and counting the cards are two
  // different questions, and folding them into one `if` meant that passing
  // --gpu (to name it) also skipped the COUNT — so an N-card rig fell back to
  // gpuCount 1 and reported one card on a board row for N.
  const det = await detectGpu();
  if (!settings.gpuProvided && det && det.name) settings.gpu = det.name;
  // Every card mines, one core each.
  settings.gpus = await detectMinerGpus();
  // Always set, so downstream reads don't need a fallback: 1 when detection
  // found nothing or found a single card.
  settings.gpuCount = det && det.count > 1 ? det.count : 1;
  const endpoint = resolveEndpoint(settings);
  log('address:    ' + shortenAddress(settings.address) + (settings.mdlAddress ? '  (+MDL ' + shortenAddress(settings.mdlAddress) + ')' : ''));
  log('pool:       ' + endpoint + '  ' + regionLabel(settings.region) + (settings.regionProvided ? '' : '  (auto)'));
  log('worker:     ' + settings.worker + (settings.workerProvided ? '' : '  (auto)'));
  if (settings.gpu) {
    log('gpu:        ' + (settings.gpuCount > 1 ? settings.gpuCount + '× ' : '') + settings.gpu
      + (settings.gpuProvided ? '' : '  (auto)'));
  }
  // What will actually mine, which is not always what the line above names: a
  // mixed rig has one name there and several cards here, and PEARL_GPU_INDEX
  // narrows it to one. Each card names itself again as its core starts.
  if (clearedCudaLine) log(clearedCudaLine);
  if (settings.gpus.length > 1) {
    log('mining on:  ' + settings.gpus.length + ' GPUs ['
      + settings.gpus.map((g) => g.index).join(', ') + ']');
  }

  // No engine to resolve: the GPU work is a linked N-API addon, so there is
  // nothing to download, no version to pick and no driver gate to clear. A
  // build with no core has nothing to do, and exiting non-zero says so to
  // systemd, where an exit 0 would read as success and produce a silent
  // ten-second restart loop that mined nothing.
  const createCore = coreFactory({ resourcesPath: process.resourcesPath });
  if (!createCore) {
    const where = 'searched: PEARL_CORE_PATH, beside the executable, and the dev tree';
    log('pearl_core.node not found -- this build cannot mine (' + where + ').', process.stderr);
    log('Fix: keep pearl_core.node from the release next to the executable, or set PEARL_CORE_PATH=/path/to/pearl_core.node.', process.stderr);
    return 1;
  }

  const stats = initStats(Date.now());
  let reporter = null;
  let statsWriter = null;
  let stopping = false;

  const miner = new PearlEngine({
    connect: (host, port) => net.connect(port, host),
    createCore,
    // Without this the engine never polls for a card temperature, so every
    // headless rig reported temp 0 -- to the stats file, to the miner report,
    // and to the network board. The GUI has always passed it (main.js).
    readTemps: () => probe.detectGpuTemps(),
  });
  miner.on('log', (l) => log(l.line, l.level === 'error' ? process.stderr : process.stdout));
  // The line reports the RIG, and every card reports itself: each core ticks
  // its hashrate about twice a second, so a 13-card rig would write 26 copies
  // of the same totals every second into the journal. One a second is plenty
  // — no card's numbers are lost, they are all in the total.
  let lastMineLog = 0;
  miner.on('event', (evt) => {
    applyEvent(stats, evt, Date.now());
    if (evt.type === 'status') {
      const now = Date.now();
      if (now - lastMineLog < MINE_LOG_MS) return;
      lastMineLog = now;
      const snap = snapshot(stats, now);
      log('⛏  ' + format.formatHashrate(snap.total) + ' TH/s · '
        + format.formatInt(snap.accepted) + ' accepted · ' + snap.rejected + ' rejected · up '
        + format.formatUptime(snap.uptimeSec));
    }
  });
  miner.on('error', (err) => log('engine error: ' + err.message, process.stderr));

  if (settings.report) {
    // Sample per-card live VRAM (nvidia-smi) and post one board row per GPU,
    // just like the GUI — otherwise the board shows 0 GB for a CLI-driven rig.
    const report = async () => {
      const snap = snapshot(stats, Date.now());
      const gpuVram = await detectGpusVram();
      return Promise.all(buildMinerReports(settings, snap, gpuVram, pkg.version).map(postMinerReport));
    };
    report();
    reporter = setInterval(report, NETWORK.reportIntervalMs);
    if (reporter.unref) reporter.unref();
  }

  return new Promise((resolve) => {
    let settled = false;
    const finish = (code) => {
      if (settled) return;
      settled = true;
      resolve(code);
    };

    const shutdown = () => {
      if (stopping) return;
      stopping = true;
      log('shutting down…');
      if (reporter) clearInterval(reporter);
      if (statsWriter) clearInterval(statsWriter);
      // isRunning(), not truthiness: PearlEngine.stop() on a miner that is
      // already stopped emits nothing, so waiting on its 'stopped' would never
      // resolve the run.
      if (miner.isRunning()) miner.stop();
      else finish(0);
    };
    process.on('SIGINT', shutdown);
    process.on('SIGTERM', shutdown);

    // Write live stats JSON for external consumers (HiveOS h-stats.sh reads this
    // to feed the dashboard). Atomic write (tmp + rename) so readers never see a
    // torn file; best-effort — a failed write must never affect mining. The
    // model/gate fields the LLM used to fill stay in the payload as nulls, so a
    // consumer reading them by name keeps working.
    if (settings.statsFile) {
      const writeStats = () => {
        try {
          const payload = statsFilePayload(snapshot(stats, Date.now()), {
            version: pkg.version,
            nowMs: Date.now(),
            mode: 'mining',
            mining: miner.isRunning(),
          });
          const tmp = settings.statsFile + '.tmp';
          fs.writeFileSync(tmp, JSON.stringify(payload));
          fs.renameSync(tmp, settings.statsFile);
        } catch (e) { /* best effort */ }
      };
      writeStats();
      statsWriter = setInterval(writeStats, 10000);
      if (statsWriter.unref) statsWriter.unref();
    }

    miner.on('stopped', (code) => {
      if (reporter) clearInterval(reporter);
      if (statsWriter) clearInterval(statsWriter);
      log('engine exited (code ' + code + ')');
      finish(stopping ? 0 : (code || 0));
    });
    try {
      // A false return is a fatal start failure, not a hiccup: the core did not
      // construct, so there is no socket, no job, and no 'stopped' event coming.
      // Left unchecked the process simply ran out of work and exited 0 -- which
      // under Restart=always is a ten-second restart loop that mines nothing and
      // looks healthy to systemd. Exit non-zero so a supervisor can see it.
      if (miner.start(Object.assign({}, settings, { endpoint })) === false) {
        log('engine failed to start — see the error above', process.stderr);
        finish(1);
      }
    } catch (e) {
      log('failed to launch engine: ' + e.message, process.stderr);
      finish(1);
    }
  });
}

// Force the exit if the loop will not drain on its own.
//
// Setting exitCode asks node to leave once nothing is left to do, which assumes
// everything we started has been torn down. A handle that outlives its stop
// keeps the loop alive, and the process sits there -- so systemd sees `active`,
// Restart=on-failure never fires, and the rig is not mining until someone
// notices. A non-zero exit that does not exit is not an exit; give the teardown
// a few seconds and then insist.
/* istanbul ignore next */
function exitWith(code) {
  process.exitCode = code;
  const t = setTimeout(() => process.exit(code), 5000);
  if (t.unref) t.unref();
}

/* istanbul ignore next */
if (require.main === module) {
  run(process.argv.slice(2)).then(exitWith).catch((e) => {
    log('fatal: ' + (e && e.message ? e.message : e), process.stderr);
    exitWith(1);
  });
}

module.exports = { run };
