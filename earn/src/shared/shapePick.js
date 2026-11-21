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
//
// An RTX 50 card times its two rotation fold files the same way, but its turns end on a
// whole salt, and a salt takes 4 to 12 s there, so a round of two turns takes 25 to 55 s.
// 150 s gave a capped RTX 5080 three rounds, which read its +0.5% as +0.18% and kept the
// slower file (benchmark.md, 15 November). Such a card gets ROT_TRIAL_SECS instead.
//
// That trial also waits: it starts after ROT_TRIAL_DELAY_SECS of mining, not with the first job.
// Half of its 10 minutes are in the other file, which is the slower one on most PCs (0.2 to
// 1.4%), so a card that started with it mined its first 10 minutes on a fresh PC 0.1 to 0.7%
// down, and a box that restarts often paid that each time (benchmark.md, 19 November). The
// card is also still warming up then, and a trial reads a drifting clock badly. Until the
// trial has run the card mines on the rule's file. PEARL_SHAPE_TRIAL_DELAY=<secs> sets the
// wait (0 starts the trial at once), for testing.

const path = require('path');

const TRIAL_SECS = 150;
const ROT_TRIAL_SECS = 600;
const ROT_TRIAL_DELAY_SECS = 1800;
// The cards that get ROT_TRIAL_SECS, by the name the card list gives: RTX 50 and the RTX
// PRO Blackwell cards (sm_120, the rotation fold's two files).
const ROT_CARD_RE = /\bRTX 50[0-9]0\b|\bBlackwell\b/i;
// A pick is timed again after a week: drivers, power limits and cooling change.
const MAX_AGE_MS = 7 * 24 * 3600 * 1000;

// Whether something saved at `at` still stands at `now`: younger than `maxAge`
// and not dated after `now`. A future date means the PC's clock was set back,
// so its age is not known. Shared with memClockTrial.
function stillFresh(at, now, maxAge) {
  const t = Number(at);
  return t > now - maxAge && t <= now;
}

// The cache key for a card: its index and, when the card list names it, its name, so a
// card swapped into the same slot is timed again.
function cardName(coreOpts, gpus) {
  const index = coreOpts && Number.isInteger(coreOpts.deviceIndex) ? coreOpts.deviceIndex : null;
  if (index == null) return null;
  const card = (Array.isArray(gpus) ? gpus : []).find((g) => g && g.index === index);
  return card && card.name ? String(card.name) : '';
}

function shapeKey(coreOpts, gpus) {
  const name = cardName(coreOpts, gpus);
  if (name == null) return 'auto';
  return 'gpu' + coreOpts.deviceIndex + (name ? ' ' + name : '');
}

// How long a card's trial runs: ROT_TRIAL_SECS for an RTX 50 card, TRIAL_SECS for the rest
// (and for a card the list does not name).
function trialSecs(coreOpts, gpus) {
  return ROT_CARD_RE.test(cardName(coreOpts, gpus) || '') ? ROT_TRIAL_SECS : TRIAL_SECS;
}

// How long a card mines before its trial starts: ROT_TRIAL_DELAY_SECS for an RTX 50 card, 0
// for the rest, whose trial is short and often finds a faster shape. `env`'s
// PEARL_SHAPE_TRIAL_DELAY, a whole number of seconds, replaces either.
function trialDelaySecs(coreOpts, gpus, env = {}) {
  const asked = env.PEARL_SHAPE_TRIAL_DELAY;
  if (asked != null && /^\d+$/.test(String(asked))) return Number(asked);
  return ROT_CARD_RE.test(cardName(coreOpts, gpus) || '') ? ROT_TRIAL_DELAY_SECS : 0;
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
  // The cores whose trial is under way: from the open that asked for it until
  // the core's 'shape' event (see timingShapes below).
  const timing = new WeakSet();
  const wrapped = (profile, coreOpts) => {
    const opts = Object.assign({}, coreOpts);
    const key = shapeKey(opts, gpus);
    const forced = env.PEARL_SHAPE ? String(env.PEARL_SHAPE) : '';
    const trialsOff = env.PEARL_SHAPE_TRIAL === '0';
    const saved = forced ? null : store.get(key);
    const fresh = saved && typeof saved.shape === 'string' && stillFresh(saved.at, now(), MAX_AGE_MS);
    if (forced) opts.shape = forced;
    else if (fresh) opts.shape = saved.shape;
    else if (!trialsOff) {
      opts.shapeTrialSecs = trialSecs(opts, gpus);
      const delay = trialDelaySecs(opts, gpus, env);
      if (delay > 0) opts.shapeTrialDelaySecs = delay;
    }

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
    log('info', card + ': timing its launch shapes for about ' + opts.shapeTrialSecs + ' s while it mines ('
      + core.shapes.join(', ') + ')'
      + (opts.shapeTrialDelaySecs ? ', after ' + opts.shapeTrialDelaySecs + ' s of mining' : ''));
    // The core only runs a trial when it has two shapes or more, and one that
    // runs none sends no 'shape' event to end the wait.
    if (core.shapes.length > 1) timing.add(core);
    core.on('shape', (json) => {
      timing.delete(core);
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
  // Whether `core` is still timing its launch shapes. The miner asks before it
  // starts a card's core clock offset trial (main/pearlMiner, _planCoreOffset):
  // the two would read each other's changes. It asks here and does not listen
  // for 'shape' itself, because the core keeps one listener an event and a
  // second would take this one's place. A core opened with a cached or forced
  // shape, or with trials off, times nothing, and the answer is false at once.
  wrapped.timingShapes = (core) => timing.has(core);
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

module.exports = { TRIAL_SECS, ROT_TRIAL_SECS, ROT_TRIAL_DELAY_SECS, MAX_AGE_MS, stillFresh, shapeKey, trialSecs, trialDelaySecs, readTrial, trialLine, withShapePick, shapeFileStore };
