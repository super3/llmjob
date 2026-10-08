'use strict';

// Runs the real HiveOS hook scripts in hiveos/ under bash, the way HiveOS runs
// them, against made-up flight sheets, stats files and GPU lists in a temp dir:
//   h-config.sh  sourced, with the flight sheet's variables set
//   h-stats.sh   sourced, with the agent's $gpu_stats and $gpu_detect_json set
//   h-run.sh     executed, then stopped with Ctrl+C the way `miner stop` does
//
// They are HiveOS (Linux) scripts and need bash and jq. Without them, or on
// Windows, the suite is skipped and its name says why.

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn, spawnSync } = require('child_process');
const { parseCliArgs } = require('../src/shared/cliArgs');

const PKG = path.join(__dirname, '..', 'hiveos');
const STATS_PATH = '/run/hive/llmjob-earn-stats.json';
const ADDR = 'prl1pql8r6m4z9x7v2k0t3whu8e2snd4p6c';
const MDL = 'mdl1pql8r6m4z9x7v2k0t3whu8e2snd4p6c';

function has(cmd) {
  return spawnSync(cmd, ['--version'], { stdio: 'ignore' }).status === 0;
}
const skipReason = process.platform === 'win32' ? 'HiveOS scripts do not run on Windows'
  : !has('bash') ? 'bash not found'
    : !has('jq') ? 'jq not found' : null;
const suite = (name, fn) => (skipReason
  ? describe.skip(name + ' (skipped: ' + skipReason + ')', fn)
  : describe(name, fn));

// A copy of the package in a temp dir. The manifest's absolute paths point into
// the temp dir, and bin/ holds stand-ins for the GPU tools that record any call:
// h-stats.sh must never run them.
let tmp;
let pkg;
let bin;
beforeEach(() => {
  tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'hiveos-'));
  pkg = path.join(tmp, 'llmjob-earn');
  bin = path.join(tmp, 'bin');
  fs.mkdirSync(pkg);
  fs.mkdirSync(bin);
  for (const f of ['h-config.sh', 'h-run.sh', 'h-stats.sh']) {
    fs.copyFileSync(path.join(PKG, f), path.join(pkg, f));
    fs.chmodSync(path.join(pkg, f), 0o755);
  }
  const manifest = fs.readFileSync(path.join(PKG, 'h-manifest.conf'), 'utf8')
    .replace(/^CUSTOM_CONFIG_FILENAME=.*$/m, 'CUSTOM_CONFIG_FILENAME=' + path.join(tmp, 'llmjob-earn.conf'))
    .replace(/^CUSTOM_LOG_BASENAME=.*$/m, 'CUSTOM_LOG_BASENAME=' + path.join(tmp, 'log', 'llmjob-earn'));
  fs.writeFileSync(path.join(pkg, 'h-manifest.conf'), manifest);
  for (const tool of ['gpu-stats', 'nvidia-smi', 'nvtool', 'gpu-detect']) {
    fs.writeFileSync(path.join(bin, tool),
      '#!/bin/sh\ntouch "' + path.join(tmp, 'ran-' + tool) + '"\nexit 1\n', { mode: 0o755 });
  }
});
afterEach(() => {
  fs.rmSync(tmp, { recursive: true, force: true });
});

// A script that never returns fails its test instead of hanging the run.
const SYNC = { encoding: 'utf8', timeout: 10000, killSignal: 'SIGKILL' };

function env(vars) {
  return Object.assign({
    PATH: bin + path.delimiter + process.env.PATH,
    HOME: tmp,
    // Where h-stats.sh looks when the agent's variables are empty. Never the
    // real /run/hive.
    GPU_STATS_JSON: path.join(tmp, 'no-gpu-stats.json'),
    GPU_DETECT_JSON: path.join(tmp, 'no-gpu-detect.json'),
  }, vars);
}

// ── h-config.sh ─────────────────────────────────────────────────────────────

// Source h-config.sh like HiveOS's miner_config_gen, and return the argument
// line it wrote (null if none) and what the CLI makes of it.
function config(vars) {
  const r = spawnSync('bash', ['-c', 'source "$1/h-config.sh"', '_', pkg], Object.assign({ env: env(vars) }, SYNC));
  const conf = path.join(tmp, 'llmjob-earn.conf');
  const line = fs.existsSync(conf) ? fs.readFileSync(conf, 'utf8').trim() : null;
  const parsed = line == null ? null : parseCliArgs(line.split(/\s+/));
  return { status: r.status, out: r.stdout + r.stderr, line, parsed };
}

suite('h-config.sh: wallet template', () => {
  test('%WAL% alone: the address, with WORKER_NAME as the worker', () => {
    const r = config({ CUSTOM_TEMPLATE: ADDR, WORKER_NAME: 'rig7' });
    expect(r.status).toBe(0);
    expect(r.line).toBe('--address ' + ADDR + ' --worker=rig7 --mode mining --no-update --stats-file ' + STATS_PATH);
    expect(r.parsed.errors).toEqual([]);
    expect(r.parsed.settings).toMatchObject({
      address: ADDR, mdlAddress: null, worker: 'rig7', mode: 'mining', update: false, statsFile: STATS_PATH,
    });
  });

  // HiveOS's usual template. It used to reach --address whole, and the CLI
  // exited with "invalid Pearl address".
  test('%WAL%.%WORKER_NAME%: split at the first dot, and the template worker wins', () => {
    const r = config({ CUSTOM_TEMPLATE: ADDR + '.farm-b', WORKER_NAME: 'rig7' });
    expect(r.parsed.errors).toEqual([]);
    expect(r.parsed.settings).toMatchObject({ address: ADDR, worker: 'farm-b' });
  });

  test('a worker name with dots keeps them', () => {
    const r = config({ CUSTOM_TEMPLATE: ADDR + '.rig.01' });
    expect(r.parsed.settings).toMatchObject({ address: ADDR, worker: 'rig.01' });
  });

  test('the old prl1…+mdl1… wallet still splits into --address and --mdl', () => {
    const r = config({ CUSTOM_TEMPLATE: ADDR + '+' + MDL, WORKER_NAME: 'rig7' });
    expect(r.line).toContain('--address ' + ADDR + ' --mdl ' + MDL + ' --worker=rig7');
    expect(r.parsed.errors).toEqual([]);
    expect(r.parsed.settings).toMatchObject({ address: ADDR, mdlAddress: MDL, worker: 'rig7' });
  });

  test('prl1…+mdl1….worker: both splits at once', () => {
    const r = config({ CUSTOM_TEMPLATE: ADDR + '+' + MDL + '.farm-b', WORKER_NAME: 'rig7' });
    expect(r.parsed.errors).toEqual([]);
    expect(r.parsed.settings).toMatchObject({ address: ADDR, mdlAddress: MDL, worker: 'farm-b' });
  });

  test('stray whitespace from the form is dropped', () => {
    const r = config({ CUSTOM_TEMPLATE: '  ' + ADDR + '.farm-b \n', WORKER_NAME: 'rig7' });
    expect(r.parsed.errors).toEqual([]);
    expect(r.parsed.settings).toMatchObject({ address: ADDR, worker: 'farm-b' });
  });

  test('no worker anywhere: no --worker, so the CLI uses the hostname', () => {
    const r = config({ CUSTOM_TEMPLATE: ADDR });
    expect(r.line).not.toContain('--worker');
    expect(r.parsed.errors).toEqual([]);
    expect(r.parsed.settings.workerProvided).toBe(false);
  });

  // h-run.sh splits the line on whitespace and expands globs, so neither may
  // reach it inside a worker name. A leading '-' must not read as a flag.
  test('a worker name cannot split the line, glob, or pass for a flag', () => {
    let r = config({ CUSTOM_TEMPLATE: ADDR, WORKER_NAME: 'my rig*[1]?' });
    expect(r.parsed.errors).toEqual([]);
    expect(r.parsed.settings.worker).toBe('my_rig__1__');
    r = config({ CUSTOM_TEMPLATE: ADDR, WORKER_NAME: '-rig' });
    expect(r.parsed.errors).toEqual([]);
    expect(r.parsed.settings.worker).toBe('-rig');
  });

  test('no wallet: says so, exits 1, writes nothing', () => {
    for (const t of ['', '   ', '.rig7', '+' + MDL]) {
      const r = config({ CUSTOM_TEMPLATE: t, WORKER_NAME: 'rig7' });
      expect(r.status).toBe(1);
      expect(r.out).toContain('No wallet set');
      expect(r.line).toBeNull();
    }
  });
});

suite('h-config.sh: mode and Extra config', () => {
  test('mines only by default', () => {
    expect(config({ CUSTOM_TEMPLATE: ADDR }).parsed.settings.mode).toBe('mining');
  });

  test('a --mode in Extra config replaces the default', () => {
    for (const [extra, mode] of [['--mode auto', 'auto'], ['--region de --mode=llm', 'llm'],
      ['--mode\tauto', 'auto']]) {
      const r = config({ CUSTOM_TEMPLATE: ADDR, CUSTOM_USER_CONFIG: extra });
      expect(r.line).not.toContain('--mode mining');
      expect(r.parsed.errors).toEqual([]);
      expect(r.parsed.settings.mode).toBe(mode);
    }
  });

  test('other flags keep the default and go last', () => {
    const r = config({ CUSTOM_TEMPLATE: ADDR, CUSTOM_USER_CONFIG: '--region de --llm-model /m.gguf' });
    expect(r.line.endsWith('--region de --llm-model /m.gguf')).toBe(true);
    expect(r.parsed.errors).toEqual([]);
    expect(r.parsed.settings).toMatchObject({ mode: 'mining', region: 'de' });
  });

  test('the Pool URL and Pass fields are ignored', () => {
    const r = config({ CUSTOM_TEMPLATE: ADDR, CUSTOM_URL: 'alphapool.tech:5566', CUSTOM_PASS: 'x' });
    expect(r.line).not.toContain('alphapool');
    expect(r.parsed.errors).toEqual([]);
  });

  // Not defended against in h-config.sh: the CLI names the bad flag in the
  // miner log, and HiveOS marks "Miner starting error" after a few restarts.
  test('a flag the CLI does not have is passed through for the CLI to report', () => {
    const r = config({ CUSTOM_TEMPLATE: ADDR, CUSTOM_USER_CONFIG: '--difficulty 131072' });
    expect(r.parsed.errors).toContain('unknown option: --difficulty');
  });
});

// ── h-stats.sh ──────────────────────────────────────────────────────────────

// TH/s to kH/s, rounded down the way h-stats.sh does (both are IEEE doubles).
const kh = (ths) => Math.floor(ths * 1e9);

function card(index, ths, busId, extra) {
  return Object.assign({ index, gpu: 'NVIDIA GeForce RTX 4090', hashrate: ths, accepted: 10, rejected: 0,
    power: 0, temp: 50 + index, pciBusId: busId }, extra);
}

function statsFile(cards, extra) {
  return Object.assign({
    ver: '0.5.12', algo: 'pearlhash', schema: 1, mode: 'mining', mining: true,
    ths: cards.reduce((a, c) => a + c.hashrate, 0),
    accepted: cards.reduce((a, c) => a + c.accepted, 0),
    rejected: cards.reduce((a, c) => a + c.rejected, 0),
    uptimeSec: 1200, gpus: cards,
  }, extra);
}

// gpu-stats and gpu-detect output for these cards, in PCI order. `extra` are
// non-mining display devices (an iGPU, a BMC), listed first like gpu-detect
// does with IGPU_FIRST=1.
function gpuLists(nvidia, extra) {
  const all = (extra || []).map((busid) => ({ busid, brand: 'cpu', temp: '0', fan: '0' }))
    .concat(nvidia.map(([busid, temp, fan]) => ({ busid, brand: 'nvidia', temp: String(temp), fan: String(fan) })));
  return {
    gpu_stats: JSON.stringify({
      temp: all.map((g) => g.temp), fan: all.map((g) => g.fan), power: all.map(() => '0'),
      busids: all.map((g) => g.busid), brand: all.map((g) => g.brand),
    }),
    gpu_detect_json: JSON.stringify(all.map((g) => ({ busid: g.busid, brand: g.brand, name: 'x' }))),
  };
}

// Write the stats file (`ageSec` old) and source h-stats.sh like the agent.
function runStats(content, vars, ageSec) {
  const file = path.join(tmp, 'stats.json');
  if (content !== undefined) {
    fs.writeFileSync(file, typeof content === 'string' ? content : JSON.stringify(content));
    const t = Date.now() / 1000 - (ageSec || 0);
    fs.utimesSync(file, t, t);
  }
  const r = spawnSync('bash', ['-c', 'source "$1/h-stats.sh"; printf "%s\\n" "$khs" "$stats"', '_', pkg],
    Object.assign({ env: env(Object.assign({ LLMJOB_EARN_STATS_FILE: file }, vars)) }, SYNC));
  expect(r.status).toBe(0);
  const [khs, stats] = r.stdout.trim().split('\n');
  return { khs: Number(khs), stats: JSON.parse(stats) };
}

// The checks HiveOS's agent (hive 0.6-231, bin/sanitize) applies before it
// sends miner stats, restated: per-card hashrates are range-checked as kH/s
// whatever hs_units says, with 9999999999999 the most it passes; bus numbers
// are PCI bus numbers (0-255); ver is cut at 50 characters.
function expectHiveOsShape(khs, stats) {
  expect(Number.isInteger(khs) && khs >= 0 && khs <= 9999999999999).toBe(true);
  expect(stats.hs_units).toBe('khs');
  for (const v of stats.hs) expect(Number.isInteger(v) && v >= 0 && v <= 9999999999999).toBe(true);
  expect(stats.temp).toHaveLength(stats.hs.length);
  expect(stats.fan).toHaveLength(stats.hs.length);
  if (stats.bus_numbers) {
    expect(stats.bus_numbers).toHaveLength(stats.hs.length);
    for (const b of stats.bus_numbers) expect(Number.isInteger(b) && b >= 0 && b <= 255).toBe(true);
  }
  expect(stats.ver.length).toBeLessThanOrEqual(50);
  expect(stats.algo).toBe('pearlhash');
}

suite('h-stats.sh', () => {
  afterEach(() => {
    for (const tool of ['gpu-stats', 'nvidia-smi', 'nvtool', 'gpu-detect']) {
      expect(fs.existsSync(path.join(tmp, 'ran-' + tool))).toBe(false);
    }
  });

  test('1 GPU: its hashrate, bus, temp and fan', () => {
    const { khs, stats } = runStats(statsFile([card(0, 315.8, '00000000:01:00.0')]),
      gpuLists([['01:00.0', 64, 72]]));
    expectHiveOsShape(khs, stats);
    expect(khs).toBe(kh(315.8));
    expect(stats).toEqual({
      hs: [kh(315.8)], hs_units: 'khs', temp: [64], fan: [72], uptime: 1200, ver: '0.5.12',
      ar: [10, 0], algo: 'pearlhash', bus_numbers: [1],
    });
  });

  // It used to split the total evenly over the cards.
  test('2 GPUs: each card its own hashrate, bound to its own bus', () => {
    const { khs, stats } = runStats(
      statsFile([card(0, 310, '00000000:01:00.0'), card(1, 290, '00000000:0A:00.0')]),
      gpuLists([['01:00.0', 62, 70], ['0a:00.0', 59, 66]]));
    expectHiveOsShape(khs, stats);
    expect(khs).toBe(kh(600));
    expect(stats.hs).toEqual([kh(310), kh(290)]);
    expect(stats.bus_numbers).toEqual([1, 10]);
    expect(stats.temp).toEqual([62, 59]);
    expect(stats.fan).toEqual([70, 66]);
    expect(stats.ar).toEqual([20, 0]);
  });

  // It used to count the iGPU as a third card and give it a third of the total.
  test('2 GPUs and an iGPU: two entries, and the iGPU gets nothing', () => {
    const { khs, stats } = runStats(
      statsFile([card(0, 310, '00000000:01:00.0'), card(1, 290, '00000000:03:00.0')]),
      gpuLists([['01:00.0', 62, 70], ['03:00.0', 59, 66]], ['00:02.0']));
    expectHiveOsShape(khs, stats);
    expect(stats.hs).toEqual([kh(310), kh(290)]);
    expect(stats.bus_numbers).toEqual([1, 3]);
    expect(stats.temp).toEqual([62, 59]);
    expect(stats.fan).toEqual([70, 66]);
  });

  test('cards the CLI has no bus for: the NVIDIA cards gpu-detect lists stand in, past a BMC', () => {
    const { khs, stats } = runStats(statsFile([card(0, 130, null), card(1, 130, null)]),
      gpuLists([['41:00.0', 55, 60], ['c1:00.0', 57, 61]], ['03:00.0']));
    expectHiveOsShape(khs, stats);
    expect(stats.bus_numbers).toEqual([65, 193]);
    expect(stats.temp).toEqual([55, 57]);
  });

  test('a stats file with no bus at all (an older CLI) is read the same way', () => {
    const cards = [card(0, 130), card(1, 130)];
    for (const c of cards) delete c.pciBusId;
    const { stats } = runStats(statsFile(cards), gpuLists([['01:00.0', 55, 60], ['02:00.0', 57, 61]]));
    expect(stats.bus_numbers).toEqual([1, 2]);
  });

  test('a disabled card in gpu-detect is not counted', () => {
    const lists = gpuLists([['01:00.0', 55, 60], ['02:00.0', 57, 61]]);
    const detect = JSON.parse(lists.gpu_detect_json);
    detect.unshift({ busid: '00:10.0', brand: 'nvidia', vbios: 'DISABLED' });
    const { stats } = runStats(statsFile([card(0, 130, null), card(1, 130, null)]),
      Object.assign(lists, { gpu_detect_json: JSON.stringify(detect) }));
    expect(stats.bus_numbers).toEqual([1, 2]);
  });

  test('no bus from anywhere: bus_numbers left out, the CLI\'s own temps used', () => {
    const { khs, stats } = runStats(statsFile([card(0, 130, null), card(1, 120, null)]), {});
    expectHiveOsShape(khs, stats);
    expect(stats.bus_numbers).toBeUndefined();
    expect(stats.hs).toEqual([kh(130), kh(120)]);
    expect(stats.temp).toEqual([50, 51]);
    expect(stats.fan).toEqual([0, 0]);
  });

  test('gpu-detect covering fewer cards than mine: bus_numbers left out', () => {
    const { stats } = runStats(statsFile([card(0, 130, null), card(1, 120, null)]),
      gpuLists([['01:00.0', 55, 60]]));
    expect(stats.bus_numbers).toBeUndefined();
  });

  test('a card gpu-stats does not list keeps the CLI\'s temp and fan 0', () => {
    const { stats } = runStats(statsFile([card(0, 130, '00000000:05:00.0')]), gpuLists([['01:00.0', 55, 60]]));
    expect(stats.bus_numbers).toEqual([5]);
    expect(stats.temp).toEqual([50]);
    expect(stats.fan).toEqual([0]);
  });

  test('unreadable agent variables fall back to the files, then to nothing', () => {
    const lists = gpuLists([['01:00.0', 64, 72]]);
    fs.writeFileSync(path.join(tmp, 'gs.json'), lists.gpu_stats);
    const { stats } = runStats(statsFile([card(0, 100, '00000000:01:00.0')]),
      { gpu_stats: '', gpu_detect_json: 'not json', GPU_STATS_JSON: path.join(tmp, 'gs.json') });
    expect(stats.temp).toEqual([64]);
    expect(runStats(statsFile([card(0, 100, null)]), { gpu_stats: '[1]' }).stats.temp).toEqual([50]);
  });

  test('nothing mined yet: no cards, khs 0', () => {
    const { khs, stats } = runStats(statsFile([]), gpuLists([['01:00.0', 40, 30]]));
    expect(khs).toBe(0);
    expect(stats.hs).toEqual([]);
    expect(stats.bus_numbers).toBeUndefined();
  });

  test('a file 2 minutes old still counts', () => {
    expect(runStats(statsFile([card(0, 100, '00000000:01:00.0')]), {}, 110).khs).toBe(kh(100));
  });

  test('a stale file (miner hung or gone) reports 0 and null', () => {
    expect(runStats(statsFile([card(0, 100, '00000000:01:00.0')]), {}, 180)).toEqual({ khs: 0, stats: null });
  });

  test('no file, or one that is not JSON, reports 0 and null', () => {
    expect(runStats(undefined, {})).toEqual({ khs: 0, stats: null });
    expect(runStats('{"ths": 1', {})).toEqual({ khs: 0, stats: null });
  });
});

// ── h-run.sh ────────────────────────────────────────────────────────────────

// A stand-in CLI that logs its arguments and, on Ctrl+C, logs a few shutdown
// lines over 300 ms before exiting, as the real one does. It gives up after
// 10 s so nothing outlives a failed test.
function fakeCli() {
  fs.writeFileSync(path.join(pkg, 'llmjob-earn-cli-linux'), [
    '#!/usr/bin/env bash',
    'trap \'echo "stopping"; sleep 0.3; echo "released clocks"; echo "stopped"; exit 0\' INT',
    'echo "started $*"',
    'for i in $(seq 200); do sleep 0.05; done',
  ].join('\n') + '\n', { mode: 0o755 });
}

function logText() {
  const f = path.join(tmp, 'log', 'llmjob-earn.log');
  return fs.existsSync(f) ? fs.readFileSync(f, 'utf8') : '';
}

async function waitFor(fn, ms) {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    if (fn()) return true;
    await new Promise((r) => setTimeout(r, 25));
  }
  return fn();
}

// Start h-run.sh in its own process group, then send Ctrl+C to the whole group
// the way `miner stop` does through the screen's terminal.
const starts = () => (logText().match(/^started/gm) || []).length;
const groups = [];
afterEach(() => {
  for (const pid of groups.splice(0)) {
    try { process.kill(-pid, 'SIGKILL'); } catch (e) { /* already gone */ }
  }
});
async function runAndStop() {
  const before = starts();
  const child = spawn('bash', [path.join(pkg, 'h-run.sh')], { env: env({}), detached: true, stdio: 'ignore' });
  groups.push(child.pid);
  const exited = new Promise((resolve) => child.on('exit', (code, signal) => resolve({ code, signal })));
  expect(await waitFor(() => starts() > before, 5000)).toBe(true);
  process.kill(-child.pid, 'SIGINT');
  return exited;
}

suite('h-run.sh', () => {
  test('passes the config line to the CLI, and keeps its shutdown lines on Ctrl+C', async () => {
    fakeCli();
    fs.writeFileSync(path.join(tmp, 'llmjob-earn.conf'), '--address ' + ADDR + ' --worker=rig7\n');
    const end = await runAndStop();
    expect(end).toEqual({ code: 0, signal: null });
    expect(logText()).toBe('started --address ' + ADDR + ' --worker=rig7\nstopping\nreleased clocks\nstopped\n');
  }, 15000);

  // A restart within 30 s skips HiveOS's log rotation; the run that crashed is
  // the one worth keeping.
  test('appends to the log rather than replacing it', async () => {
    fakeCli();
    fs.writeFileSync(path.join(tmp, 'llmjob-earn.conf'), '--address ' + ADDR + '\n');
    await runAndStop();
    await runAndStop();
    expect(starts()).toBe(2);
    expect(logText().match(/^stopped$/gm)).toHaveLength(2);
  }, 15000);

  test('no config file: says so and exits 1', () => {
    fakeCli();
    const r = spawnSync('bash', [path.join(pkg, 'h-run.sh')], Object.assign({ env: env({}) }, SYNC));
    expect(r.status).toBe(1);
    expect(r.stdout).toContain('Run miner restart');
    expect(logText()).toBe('');
  });
});
