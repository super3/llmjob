'use strict';

// Which launch shape each RTX 30 card mines with, picked on that PC.
//
// The core picks an RTX 30 card's launch shape (the flip or the band walk, the band depth,
// the batch width) from the card's L2 size alone. The best one moves with the PC: the
// 3080 Ti's flip read -0.14% on a 400 W PC and +1.63% on a 350 W one (benchmark.md). So
// the first time a card opens, the core times its shapes against each other while it mines
// (shapeTrialSecs, in 4 s turns, as probes/race.cu races builds), keeps the fastest when it
// clearly wins, and says which in a 'shape' event. This wraps a createCore to cache that
// pick per card and pass it back (`shape`) on every later start, so the trial runs once a
// week a card, not on every start.
//
// A core with nothing to choose (any other card, or a core built before this) reports no
// shapes, and nothing here changes how it opens. PEARL_SHAPE_TRIAL=0 turns trials off and
// PEARL_SHAPE=<name> opens every card in that shape, for testing.

const path = require('path');

const TRIAL_SECS = 150;
// A pick is timed again after a week: drivers, power limits and cooling change.
const MAX_AGE_MS = 7 * 24 * 3600 * 1000;

// The cache key for a card: its index and, when the card list names it, its name, so a
// card swapped into the same slot is timed again.
function shapeKey(coreOpts, gpus) {
  const index = coreOpts && Number.isInteger(coreOpts.deviceIndex) ? coreOpts.deviceIndex : null;
  if (index == null) return 'auto';
  const card = (Array.isArray(gpus) ? gpus : []).find((g) => g && g.index === index);
  return 'gpu' + index + (card && card.name ? ' ' + card.name : '');
}

// The trial's report, as the core sends it ({ shape, rule, trial: [{ shape, pct, ahead,
// rounds }] }), in one log line. null for one that doesn't parse.
function readTrial(json) {
  try {
    const t = JSON.parse(String(json));
    return t && typeof t.shape === 'string' && t.shape ? t : null;
  } catch (e) {
    return null;
  }
}

function trialLine(t) {
  const rows = (Array.isArray(t.trial) ? t.trial : [])
    .map((r) => r.shape + ' ' + (r.pct >= 0 ? '+' : '') + Number(r.pct).toFixed(2) + '% (ahead '
      + r.ahead + ' of ' + r.rounds + ')');
  return (t.shape === t.rule ? 'kept the default ' + t.shape : t.shape + ' over the default ' + t.rule)
    + (rows.length ? ' · against the default: ' + rows.join(', ') : '');
}

// Wrap createCore(profile, opts). `store` is { get(key), set(key, value), remove(key) },
// `log(level, line)` hears what happens, `gpus` is the mining card list (for names), and
// `now` and `env` are injectable. Returns null for a null createCore, so a shell can wrap
// coreFactory's answer as it is.
function withShapePick(createCore, { store, log = () => {}, gpus = [], now = Date.now, env = process.env } = {}) {
  if (typeof createCore !== 'function') return null;
  const wrapped = (profile, coreOpts) => {
    const opts = Object.assign({}, coreOpts);
    const key = shapeKey(opts, gpus);
    const forced = env.PEARL_SHAPE ? String(env.PEARL_SHAPE) : '';
    const trialsOff = env.PEARL_SHAPE_TRIAL === '0';
    const saved = forced ? null : store.get(key);
    const fresh = saved && typeof saved.shape === 'string' && Number(saved.at) > now() - MAX_AGE_MS;
    if (forced) opts.shape = forced;
    else if (fresh) opts.shape = saved.shape;
    else if (!trialsOff) opts.shapeTrialSecs = TRIAL_SECS;

    const core = createCore(profile, opts);
    if (!core || !Array.isArray(core.shapes) || !core.shapes.length) return core;
    const card = core.device && Number.isInteger(core.device.index) ? 'GPU ' + core.device.index : 'the GPU';
    if (opts.shape) {
      if (core.shape === opts.shape) {
        log('info', card + ': launch shape ' + opts.shape + (forced ? ' (PEARL_SHAPE)' : ' (timed on this PC)'));
      } else {
        log('info', card + ': launch shape ' + opts.shape + ' is not one this core has; mining with '
          + core.shape + (forced ? '' : ', and timing the shapes at the next start'));
        if (!forced) store.remove(key);
      }
      return core;
    }
    if (!opts.shapeTrialSecs) return core;
    log('info', card + ': timing its launch shapes for about ' + TRIAL_SECS + ' s while it mines ('
      + core.shapes.join(', ') + ')');
    core.on('shape', (json) => {
      const t = readTrial(json);
      if (!t) {
        log('warn', card + ': launch shape trial sent a report that does not parse');
        return;
      }
      store.set(key, { shape: t.shape, at: now() });
      log('info', card + ': launch shape trial ' + trialLine(t));
    });
    return core;
  };
  // The factory's core clock offset calls (main/pearlCore) go to the miner with it.
  wrapped.clockOffsets = createCore.clockOffsets;
  return wrapped;
}

// The cache as a JSON file, { key: { shape, at } }, read once and written whole on each
// change. `readSettings` and `writeSettings` are shared/settingsStore's, which log a bad
// file or a failed write rather than throw.
//
// The folder is made first, as shared/poolLookup's fileCache does. The CLI
// keeps these files beside node.json, and a mining-only or --no-serve run
// never writes node.json, so the folder may not be there. Without it every
// save failed, nothing was remembered, and the trials ran again on each start.
// A folder that cannot be made is left to the write to report.
function shapeFileStore(file, { readSettings, writeSettings, fs, log }) {
  let data = null;
  const load = () => {
    if (!data) data = readSettings(file, { fs, log });
    return data;
  };
  const save = () => {
    try {
      fs.mkdirSync(path.dirname(file), { recursive: true });
    } catch (e) {
      // The write below says what is wrong.
    }
    writeSettings(file, data, { fs, log });
  };
  return {
    get: (key) => load()[key] || null,
    set: (key, value) => {
      load()[key] = value;
      save();
    },
    remove: (key) => {
      if (!(key in load())) return;
      delete data[key];
      save();
    },
  };
}

module.exports = { TRIAL_SECS, MAX_AGE_MS, shapeKey, readTrial, trialLine, withShapePick, shapeFileStore };
