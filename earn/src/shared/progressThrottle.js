'use strict';

// Which download percentages are worth a line. downloadFile reports every
// chunk, a few hundred a second, which would flood the desktop app's IPC and
// bury both shells' logs. Returns a function that takes each percent and
// answers the one to show, or null for none:
//   step     how many points apart two lines are
//   round    show the figure rounded down to a multiple of `step` (the CLI's
//            "40%, 50%, ..."), rather than as it came
//   everyMs  a line at least this often on a slow link, however few points it
//            moved (the desktop app's 2 s). Infinity: never.
//   now      the clock, for tests
// 100% always shows. A percent that is null (no content-length: the size is
// unknown, and Number(null) would read as a permanent, wrong 0%), not a
// number, or negative shows nothing.
function progressThrottle({ step, round = false, everyMs = Infinity, now = Date.now }) {
  let last = -1;
  let lastAt = 0;
  return (pct) => {
    const raw = Math.floor(Number(pct));
    if (pct == null || !Number.isFinite(raw) || raw < 0) return null;
    const p = round ? Math.floor(raw / step) * step : raw;
    // Rounded, any figure past the last is a step on (the first is 0%).
    const ahead = round ? p > last : p - last >= step;
    const at = now();
    if (p !== 100 && !ahead && at - lastAt < everyMs) return null;
    if (p === last) return null;
    last = p;
    lastAt = at;
    return p;
  };
}

module.exports = { progressThrottle };
