'use strict';

// Pause mining on a card while the local LLM answers a prompt on it.
//
// When the LLM co-runs with the miner, both share the card, and a prompt is
// answered at whatever speed is left over. This hands the card to the LLM for
// the length of each request: mining stops on that card the moment its
// llama-server starts one, and starts again once it has had nothing to do for
// `resumeAfterMs`. Other cards keep mining.
//
// The busy signal comes from llama-server's own log (LlmManager 'busy', passed
// on per card by LlmFleet), because most prompts never pass through our code.
// This module only keeps the bookkeeping and the timers. What a pause IS, it is
// told: `pause(indices)` and `resume(indices)` act on whatever miner is running
// at the time, with `indices` an array of card numbers or null for every card.
// An instance with no known card (llama.cpp chose it) pauses every card.
// `pause` returns whether it stopped any card. Only then is anything logged: in
// LLM-only mode, or on a card that is not mining, a prompt pauses nothing, and
// saying "mining paused" there would be wrong.
//
// Kept apart from the miner object on purpose. The shells stop and start the
// miner for their own reasons (idle-only, STOP, a mode change), and a miner that
// starts while a prompt is running must start paused: reapply() does that.

const ALL = 'all';

function createPromptPause({
  pause, resume, log = () => {}, resumeAfterMs, maxPauseMs,
  setTimer = setTimeout, clearTimer = clearTimeout,
}) {
  const busy = new Set();           // cards whose llama-server is working on a request
  const paused = new Set();         // cards this module has paused
  const resumeTimers = new Map();   // card -> timer that resumes it
  const capTimers = new Map();      // card -> safety-net timer (maxPauseMs)
  const shown = new Set();          // paused cards whose pause actually stopped mining

  const keyOf = (index) => (Number.isInteger(index) ? index : ALL);
  const indicesOf = (key) => (key === ALL ? null : [key]);
  const name = (key) => (key === ALL ? '' : ' on GPU ' + key);

  function clearTimers(key) {
    if (resumeTimers.has(key)) { clearTimer(resumeTimers.get(key)); resumeTimers.delete(key); }
    if (capTimers.has(key)) { clearTimer(capTimers.get(key)); capTimers.delete(key); }
  }

  // `why` replaces the usual line, when there is more to say than that.
  function doResume(key, why) {
    clearTimers(key);
    if (!paused.delete(key)) return;
    resume(indicesOf(key));
    if (!shown.delete(key)) return;
    if (why) log('warn', 'mining resumed' + name(key) + ': ' + why);
    else log('info', 'mining resumed' + name(key));
  }

  function doPause(key) {
    if (pause(indicesOf(key)) && !shown.has(key)) {
      shown.add(key);
      log('info', 'mining paused' + name(key) + ' while the local LLM answers a prompt');
    }
  }

  // A card's llama-server started (busy) or finished (not busy) a request.
  //
  // The safety net counts from the start of the CURRENT stretch of work, not
  // from the pause: a steady run of prompts with short gaps between them is not
  // a stuck one, and should keep the card paused.
  function setBusy(index, isBusy) {
    const key = keyOf(index);
    if (!isBusy) {
      busy.delete(key);
      if (capTimers.has(key)) { clearTimer(capTimers.get(key)); capTimers.delete(key); }
      if (!paused.has(key) || resumeTimers.has(key)) return;
      const t = setTimer(() => { resumeTimers.delete(key); doResume(key); }, resumeAfterMs);
      if (t && t.unref) t.unref();
      resumeTimers.set(key, t);
      return;
    }
    busy.add(key);
    clearTimers(key);
    const cap = setTimer(() => {
      capTimers.delete(key);
      busy.delete(key);
      doResume(key, 'the local LLM has been busy for '
        + Math.round(maxPauseMs / 60000) + ' minutes without a break');
    }, maxPauseMs);
    if (cap && cap.unref) cap.unref();
    capTimers.set(key, cap);
    if (paused.has(key)) return;
    paused.add(key);
    doPause(key);
  }

  // A miner just started. Pause on it whatever is paused now.
  function reapply() {
    for (const key of paused) doPause(key);
  }

  // The LLM is gone: nothing can be busy, so mine everywhere now.
  function reset() {
    busy.clear();
    for (const key of [...paused]) doResume(key);
  }

  // Shutting down: forget everything and touch nothing.
  function stop() {
    for (const key of [...resumeTimers.keys(), ...capTimers.keys()]) clearTimers(key);
    busy.clear();
    paused.clear();
    shown.clear();
  }

  return {
    setBusy, reapply, reset, stop,
    isPaused: (index) => paused.has(keyOf(index)),
    isBusy: (index) => busy.has(keyOf(index)),
  };
}

module.exports = { createPromptPause };
