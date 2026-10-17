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
// `pause` and `resume` return whether they stopped or started any card, and
// only then is anything logged. In LLM-only mode, on a card that is not mining,
// or after STOP or an idle-only pause took the miner away, a prompt pauses or
// resumes nothing, and saying it did would be wrong.
//
// Kept apart from the miner object on purpose. The shells stop and start the
// miner for their own reasons (idle-only, STOP, a mode change), and a miner that
// starts while a prompt is running must start paused: reapply() does that.

const { normalizePciBusId } = require('./gpu');

const ALL = 'all';

// Which card a prompt runs on, as setBusy takes it: a card number, null for
// every card, or undefined to leave mining alone. `ev` is LlmFleet's busy event,
// `busIds` nvidia-smi's { index: pciBusId } (probe.detectPciBusIds), and
// `miningCards` how many cards the miner runs on.
//
// The card an instance was planned for is the number it was started with
// (--main-gpu), but llama-server counts cards in Vulkan's order, which need not
// be nvidia-smi's (the miner's): on Windows it follows the adapter driving the
// desktop. So a two-card rig with its monitor on the second card can load the
// model on card 1 when asked for card 0, and pausing card 0 would stop the idle
// card while the busy one kept mining. The bus id the server printed at load
// says which card it really is.
//
// Without that id, a one-card rig has only one card it can be, and several
// cards are left mining: guessing wrong costs a card's mining on every prompt
// for no gain. The cards are counted both ways, because nvidia-smi is not the
// only way a rig mines: an AMD rig (PEARL_CORE_VARIANT=amd) has no nvidia-smi
// and so no bus ids, and only the miner knows it runs several cards. A bus id
// nvidia-smi does not list is a card it does not know (an AMD card beside
// NVIDIA ones, say), so nothing pauses for it.
function cardForPrompt(ev, busIds, miningCards = 0) {
  const ids = Object.entries(busIds || {});
  const want = normalizePciBusId(ev && ev.busId);
  if (want && ids.length) {
    const hit = ids.find(([, id]) => normalizePciBusId(id) === want);
    return hit ? Number(hit[0]) : undefined;
  }
  if (Math.max(ids.length, miningCards) > 1) return undefined;
  return ev && Number.isInteger(ev.index) ? ev.index : null;
}

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
    const acted = resume(indicesOf(key));
    if (!shown.delete(key) || !acted) return;
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

  // A miner just started. Pause on it whatever is paused now, and say so: what
  // was said about the last miner is no longer true.
  function reapply() {
    shown.clear();
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

module.exports = { createPromptPause, cardForPrompt };
