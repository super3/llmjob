'use strict';

// NEB (Neural Engine Byte) on the CLI: the cards --card-coin gives to NEB mine
// on an engine of their own (main/nebEngine.js), beside the Pearl run or alone.
// The CLI passes in everything this touches (the log, the file system, the
// engine and core factories), so it runs the same under test.
//
// What this first cut leaves out, on purpose: NEB cards send no row to the
// board and none to the stats file (the server knows two coins), and the
// network's dataset file must be on disk already (--neb-dataset, or
// neb-datasets.bin in the CLI's data folder).

const path = require('path');
const blake3 = require('../miner/blake3');
const nh = require('./nebhash2');
const { shortenNebAddress } = require('./address');
const { formatHs } = require('../format');

const DEFAULT_ENDPOINT = 'neb.rabidmining.com:3470';
const DATASET_FILE = 'neb-datasets.bin';
// Where the file comes from, for the message when it is missing.
const DATASET_SOURCE = 'https://github.com/SilasEigen/neb/releases';
// The gap between two NEB status lines.
const LOG_MS = 10000;

// --card-coin's value with its NEB pairs taken out: { neb: [card, ...], rest: 'the other pairs' }.
// shared/forkPlan plans Pearl and Quantus, in the desktop app too, and knows nothing of NEB.
function splitNebPairs(text) {
  const neb = [];
  const rest = [];
  for (const part of String(text == null ? '' : text).split(',').map((p) => p.trim()).filter(Boolean)) {
    const m = /^(\d+)\s*[:=]\s*neb$/i.exec(part);
    if (m) neb.push(Number(m[1]));
    else rest.push(part);
  }
  return { neb: [...new Set(neb)].sort((a, b) => a - b), rest: rest.join(',') };
}

// Which of the rig's cards mine NEB. `settings.gpus` is the rig's listed cards
// ([] when it could not list them) and `settings.nebCards` the indices
// --card-coin gave to NEB.
//   cards     the NEB cards
//   rest      the listed cards left for the Pearl run
//   unlisted  the rig's cards are not listed: NEB takes the indices as given,
//             and no other coin can share the rig
//   skipped   NEB indices that are not among the listed cards
function takeNebCards(settings) {
  const want = settings.nebCards || [];
  const listed = settings.gpus || [];
  if (!want.length) return { cards: [], rest: listed, unlisted: false, skipped: [] };
  if (!listed.length) return { cards: want.slice(), rest: [], unlisted: true, skipped: [] };
  const have = new Set(listed.map((g) => g.index));
  return {
    cards: want.filter((i) => have.has(i)),
    rest: listed.filter((g) => !want.includes(g.index)),
    unlisted: false,
    skipped: want.filter((i) => !have.has(i)),
  };
}

// What the split did that the user should know, as log lines.
function splitLines(split) {
  const lines = [];
  if (split.skipped.length) lines.push('warning: --card-coin gives GPU ' + split.skipped.join(', ') + ' to NEB, but this run has no such card');
  if (split.unlisted) lines.push('NEB on GPU ' + split.cards.join(', ') + ': this rig\'s cards are not listed, so no other coin shares it');
  return lines;
}

// The network's dataset file: { datasets, hash }. Throws with what to do.
function loadDatasets(file, fs) {
  let buf;
  try {
    buf = fs.readFileSync(file);
  } catch (e) {
    throw new Error('the NEB dataset file is not at ' + file + ' (' + e.code + '). Download neb-datasets.bin from '
      + DATASET_SOURCE + ' and put it there, or name it with --neb-dataset');
  }
  try {
    return { datasets: nh.parseDatasets(buf), hash: blake3.hash(buf).toString('hex') };
  } catch (e) {
    throw new Error(file + ' is not a NEB dataset file this miner reads: ' + e.message);
  }
}

// Starts NEB on `cards` (null: every card the core lists). Returns the engine,
// or null when it could not start; the log says why.
//   d = { settings, cards, dataDir, engineFor, coreFactory, fs, log, err, logAt, now, resourcesPath }
function start(d) {
  const { settings, cards } = d;
  let data;
  try {
    data = loadDatasets(settings.nebDataset || path.join(d.dataDir, DATASET_FILE), d.fs);
  } catch (e) {
    d.err('NEB: ' + e.message);
    return null;
  }
  const factory = d.coreFactory({ resourcesPath: d.resourcesPath, log: d.logAt });
  if (!factory) {
    d.err('no NEB core loaded; the lines above say why. NEB mines nothing.');
    return null;
  }
  const engine = d.engineFor({ factory });
  engine.on('log', (l) => d.logAt(l.level, l.line));
  // Already in the log.
  engine.on('error', () => {});
  const last = new Map();
  let lastLog = 0;
  engine.on('event', (evt) => {
    if (evt.type !== 'status') return;
    last.set(evt.gpuIndex, evt);
    const t = d.now();
    if (t - lastLog < LOG_MS) return;
    lastLog = t;
    const sum = (k) => [...last.values()].reduce((a, e) => a + e[k], 0);
    d.log('⛏  neb ' + formatHs(sum('hashrate')) + ' · ' + sum('accepted') + ' accepted · ' + sum('rejected') + ' rejected');
  });
  engine.on('stopped', (code) => d.log('neb engine exited (code ' + code + ')'));
  const endpoint = settings.nebEndpoint || DEFAULT_ENDPOINT;
  d.log('neb:        ' + (cards ? 'GPU ' + cards.join(', ') : 'every card') + ' → ' + shortenNebAddress(settings.nebAddress) + ' on ' + endpoint);
  const ok = engine.start({
    address: settings.nebAddress, worker: settings.worker, endpoint, cards, datasets: data.datasets, datasetHash: data.hash,
  });
  if (!ok) {
    d.err('neb engine failed to start — see the error above');
    return null;
  }
  return engine;
}

// A run that mines NEB and nothing else: resolves with the exit code when the
// engine stops, by a signal (0) or on its own (its code).
//   d: as for start, and { version, proc } (the process, for its signals)
function runAlone(d) {
  d.log('LLMJob Earn CLI v' + d.version);
  d.log('coin:       NEB (Neural Engine Byte), experimental');
  const engine = start(d);
  if (!engine) return Promise.resolve(1);
  return new Promise((resolve) => {
    let stopping = false;
    const shutdown = () => {
      if (stopping) return;
      stopping = true;
      d.log('shutting down…');
      engine.stop();
    };
    d.proc.on('SIGINT', shutdown);
    d.proc.on('SIGTERM', shutdown);
    engine.on('stopped', (code) => resolve(stopping ? 0 : code));
  });
}

module.exports = { DEFAULT_ENDPOINT, DATASET_FILE, DATASET_SOURCE, LOG_MS, splitNebPairs, takeNebCards, splitLines, loadDatasets, start, runAlone };
