'use strict';

// Both clock plans for one mining start, as both shells make them: the memory
// clock lock (shared/memClock) and the core clock offset trial
// (shared/coreOffset). Only how a shell reads its requests differs: the CLI has
// --mine-mem-clock and --mine-core-offset, the desktop app two environment
// variables, which can also be set to something it cannot use (a warning).
//
//   memMhz, offset           the two requests, as planMemClocks and
//                            planCoreOffsets take them
//   memName, offsetName      the switch each log line names (the CLI's flags
//                            when left out)
//   memWarning, offsetWarning a line for a request the shell could not use, or
//                            null; it goes before that plan's own line
//   cards, gpus, llmCoRuns   as both planners take them: nvidia-smi's cards
//                            only, so an AMD or Intel rig passes gpus []
//   rotationFold             the fold is on (an RTX 40 card then tries a lock)
// Returns { start, lines }: `start` is what the miner's start takes
// (mineMemClockByIndex, mineMemClockDefault, mineMemClockTrial,
// mineCoreOffsetTrial), `lines` the log lines as [{ level, line }], in order.
// A request refused because the LLM co-runs is a warning; any other plan line
// is info.

const { planMemClocks } = require('./memClock');
const { planCoreOffsets } = require('./coreOffset');

function planClocks({
  memMhz, offset, memName, offsetName, memWarning = null, offsetWarning = null,
  cards, gpus, llmCoRuns, rotationFold,
}) {
  const lines = [];
  const say = (plan) => { if (plan.reason) lines.push({ level: plan.dropped ? 'warn' : 'info', line: plan.reason }); };
  const memPlan = planMemClocks({ requestedMhz: memMhz, requestName: memName, cards, gpus, llmCoRuns, rotationFold });
  if (memWarning) lines.push({ level: 'warn', line: memWarning });
  say(memPlan);
  const offsetPlan = planCoreOffsets({ requested: offset, requestName: offsetName, cards, gpus, llmCoRuns });
  if (offsetWarning) lines.push({ level: 'warn', line: offsetWarning });
  say(offsetPlan);
  return {
    start: {
      mineMemClockByIndex: memPlan.byIndex,
      mineMemClockDefault: memPlan.isDefault,
      mineMemClockTrial: memPlan.trial,
      mineCoreOffsetTrial: offsetPlan.trial,
    },
    lines,
  };
}

// The start fields for a core with no NVIDIA card to plan for (the AMD and
// Intel cores): nothing locked, nothing tried. New each time, as a plan's are.
function noClocks() {
  return { mineMemClockByIndex: {}, mineMemClockDefault: false, mineMemClockTrial: [], mineCoreOffsetTrial: [] };
}

module.exports = { planClocks, noClocks };
