// The options hashrate.js and verify-hits.js open a core with, read from the environment.
//
// PEARL_PROFILE_K=4096 pins the core to that k, without cardK, for a bench build made for it (the AMD
// core with -DPEARL_FOLD_K=4096). Unset, the core is opened as the app opens it (cardK) and may mine a
// larger k on its card than the mainnet profile's 2048.
//
// PEARL_SHAPE_TRIAL=secs runs the core's startup trial of RTX 30's launch shapes, and PEARL_SHAPE=<name>
// ("flip 128", "band 8 x 512") opens the core in that shape. PEARL_SHAPE_TRIAL_DELAY=secs starts that trial
// after so many seconds of mining.
'use strict';

// The profile asked for: { kEnv, profile }, or { error } for a k the core can't take.
function askedProfile(mainnet, env = process.env) {
  const kEnv = env.PEARL_PROFILE_K;
  const profile = kEnv ? { ...mainnet, k: Number(kEnv) } : mainnet;
  // The sanity checks' least k is 16 x rank, and the operand trees need a power of two.
  if (kEnv && !(Number.isInteger(profile.k) && profile.k >= 16 * profile.rank && (profile.k & (profile.k - 1)) === 0)) {
    return { error: `PEARL_PROFILE_K must be a power of two of at least ${16 * profile.rank}, not ${kEnv}` };
  }
  // PEARL_PROFILE_MN=MxN (timing only, goal/r1-rtx30): another m and n, powers of two, to price a matrix whose
  // noise factors fit the card's L2, as the competitors' shares show they mine (m 7682-15744, n 16384-31488).
  const mn = /^(\d+)x(\d+)$/.exec(env.PEARL_PROFILE_MN || '');
  if (env.PEARL_PROFILE_MN) {
    const pow2 = (v) => Number.isInteger(v) && v >= 256 && v % 256 === 0;   // r1-rtx20: any multiple of 256
    if (!mn || !pow2(Number(mn[1])) || !pow2(Number(mn[2]))) {
      return { error: `PEARL_PROFILE_MN must be MxN, both multiples of 256, not ${env.PEARL_PROFILE_MN}` };
    }
    return { kEnv, profile: { ...profile, m: Number(mn[1]), n: Number(mn[2]) } };
  }
  return { kEnv, profile };
}

// createCore's options: cardK unless the k is pinned, then `extra`, then the launch shape.
function coreOptions(kEnv, extra = {}, env = process.env) {
  const opts = kEnv ? { ...extra } : { cardK: true, ...extra };
  if (env.PEARL_SHAPE_TRIAL) opts.shapeTrialSecs = Number(env.PEARL_SHAPE_TRIAL);
  if (env.PEARL_SHAPE_TRIAL_DELAY) opts.shapeTrialDelaySecs = Number(env.PEARL_SHAPE_TRIAL_DELAY);
  if (env.PEARL_SHAPE) opts.shape = env.PEARL_SHAPE;
  return opts;
}

// The card to open: `--device N` in argv, else PEARL_DEVICE. Returns { device, args }, where args is
// argv without the flag and device is null when neither names one (the core then picks the card), or
// { error } when the value is not a card index.
function deviceOption(argv, env = process.env) {
  const args = argv.slice();
  let value = env.PEARL_DEVICE;
  let from = 'PEARL_DEVICE';
  const i = args.indexOf('--device');
  if (i >= 0) { value = args[i + 1] === undefined ? '' : args[i + 1]; from = '--device'; args.splice(i, 2); }
  if (from === 'PEARL_DEVICE' && (value === undefined || value === '')) return { device: null, args };
  if (!/^\d+$/.test(value)) return { error: `${from} must be a card index (0, 1, ...), not ${value}` };
  return { device: Number(value), args };
}

module.exports = { askedProfile, coreOptions, deviceOption };
