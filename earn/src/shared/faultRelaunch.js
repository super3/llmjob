'use strict';

// How often the app may start itself again for a card whose CUDA state is lost (PearlMiner._stuckFault).
// A card that faults again soon after every new start is not having a one-off fault, and a restart a
// minute would only hide that, so after this many in an hour the card is left to its restarts and the
// log says why.
const FAULT_RELAUNCH_LIMIT = 3;
const FAULT_RELAUNCH_WINDOW_MS = 60 * 60 * 1000;

// `times` is the saved list of earlier starts (ms). Returns the list to save with `now` added, or null
// when there have been FAULT_RELAUNCH_LIMIT already within the window. Anything that isn't a time is
// dropped, so a damaged file allows a start rather than blocking every one.
function nextRelaunches(times, now) {
  const recent = (Array.isArray(times) ? times : [])
    .filter((t) => typeof t === 'number' && t <= now && now - t < FAULT_RELAUNCH_WINDOW_MS);
  return recent.length >= FAULT_RELAUNCH_LIMIT ? null : recent.concat(now);
}

module.exports = { FAULT_RELAUNCH_LIMIT, FAULT_RELAUNCH_WINDOW_MS, nextRelaunches };
