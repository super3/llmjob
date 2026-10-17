'use strict';

// Which model a node should serve, given the card it will serve from.
//
// Until now the fleet ran exactly one model and every node ran it. That works
// while the model is small enough for every card, and stops working the moment
// there is a model worth running that only a large card can hold: either every
// node is held to the small model, or small nodes are handed one they cannot
// load and crash-loop. Neither is acceptable, so the choice is made per node
// from the VRAM it actually has.
//
// The rule is deliberately the same all-or-nothing rule vram.js already applies
// to layers: a model is offered to a card only if the whole thing fits with the
// mining reserve set aside. There is no partial offload and no "mostly fits".

const { LLM } = require('./config');
const { requiredFreeMb } = require('./vram');

// Every model a node might serve, biggest first, with the small default last.
// LLM.tiers is optional so a config without it behaves exactly as before.
function allModels() {
  const tiers = Array.isArray(LLM.tiers) ? LLM.tiers : [];
  return [...tiers, LLM.model].filter(Boolean);
}

// The model sized to one context: its VRAM figures and context ladder rewritten
// for that rung, so everything downstream (vram.js, llmPlan, the shells' error
// messages, llama-server's --ctx-size) works on the context the node will run
// rather than the model's largest. The model's own top rung returns the model
// itself, unchanged, when its own figure is that rung's; without the second
// check a model carrying only per-context figures had no floor at all there.
function atContext(model, ctx) {
  const vram = Number(model.vramByCtxMb[ctx]);
  if (ctx === ctxLadder(model)[0] && vram === Number(model.vramFullMb)) return model;
  const margin = Math.max(0, (Number(model.minVramMb) || 0) - (Number(model.vramFullMb) || 0));
  return Object.assign({}, model, {
    ctxSize: ctx,
    ctxLadder: ctxLadder(model).filter((n) => n <= ctx),
    vramFullMb: vram,
    minVramMb: vram + margin,
  });
}

// The contexts a model may be chosen at, largest first. A model with no
// per-context VRAM figures has one choice: itself, at its own context. One with
// them may also be chosen at any rung of its ladder that has a measured figure
// and is no smaller than its minCtxSize.
function contextChoices(model) {
  if (!model.vramByCtxMb) return [model];
  const floor = Number(model.minCtxSize) || 0;
  return ctxLadder(model)
    .filter((ctx) => ctx >= floor && Number(model.vramByCtxMb[ctx]) > 0)
    .map((ctx) => atContext(model, ctx));
}

// The best model `freeMb` can host, at the largest context it can hold there, or
// the fallback default when none of the larger tiers fit. Returns the same object
// shape everything downstream already expects from LLM.model, so callers need no
// special-casing.
//
// Model before context: a card that can hold the large model at a smaller
// context gets it, rather than the small model. That is what lets a 32 GB card
// serve Qwen3.8 at 131072 while it mines, where it used to fall back to Gemma
// because 262144 plus the mining reserve does not fit.
//
// `freeMb == null` means VRAM could not be measured (no NVIDIA, no driver). That
// is NOT treated as "unlimited": an unmeasured card gets the default model,
// because the alternative is downloading 17 GB onto a machine that may not be
// able to run it. vram.js makes the same call for the same reason.
function pickModel(freeMb, reserveMb = 0, models = null) {
  const list = models || allModels();
  const fallback = list[list.length - 1] || null;
  const free = Number(freeMb);
  if (freeMb == null || !Number.isFinite(free)) return fallback;
  for (const m of list) {
    if (!m) continue;
    for (const choice of contextChoices(m)) {
      if (free >= requiredFreeMb(choice, reserveMb)) return choice;
    }
  }
  return fallback;
}

// The context window to ask llama-server for, and the fallbacks to try if it
// refuses to start at that size. A model with no ladder gets a single-entry one
// so the caller never has to branch.
//
// The ladder exists because the VRAM cost of a very large context is the part we
// cannot predict from the weights: most of Qwen3.8's layers use linear attention
// whose state does not grow with context, so the true cost of 262144 is neither
// the linear extrapolation nor something we can read off the file size. Walking
// down beats guessing, and beats a node that restarts forever.
function ctxLadder(model) {
  const m = model || {};
  const ladder = Array.isArray(m.ctxLadder) ? m.ctxLadder.filter((n) => Number(n) > 0) : [];
  if (ladder.length) return [...ladder];
  const one = Number(m.ctxSize) || Number(LLM.ctxSize) || 0;
  return one > 0 ? [one] : [];
}

// Does this model need the vision projector? Used to decide whether there is a
// second file to download before the server can start.
function needsMmproj(model) {
  return !!(model && model.mmproj && model.mmproj.file && model.mmproj.url);
}

// Which shape auto mode should take on this card.
//
// Auto has always meant "co-run the LLM and the miner", and pickModel is given the
// mining reserve so it only ever picks a model that fits ALONGSIDE mining. On a
// card big enough for a large tier but not for that tier PLUS the miner, that
// silently downgrades: a 5090 with 32 GB free serves the small default forever,
// because Qwen3.8 needs 30,720 MiB and the reserve leaves 30,085.
//
// So compare the two choices. If the card could serve a bigger model, or the same
// model at a bigger context, with the GPU to itself than it can while mining, auto
// becomes demand-driven -- mine until something asks for tokens, then switch. If
// they are the same model at the same context, nothing changes and the node
// co-runs exactly as before, which is what keeps small cards undisturbed.
//
// The context half keeps a 32 GB card where it was. pickModel now gives it
// Qwen3.8 at 131072 beside the miner, but alone the card holds 262144, so it
// stays demand-driven at the full context. Co-running would also cost it the
// RTX 5090's memory-clock lock, which shared/memClock drops whenever an LLM
// shares the card.
function planAutoMode(freeMb, reserveMb = 0, models = null) {
  const shared = pickModel(freeMb, reserveMb, models);
  const exclusive = pickModel(freeMb, 0, models);
  const bigger = !!(exclusive && shared && (exclusive.key !== shared.key
    || ctxLadder(exclusive)[0] > ctxLadder(shared)[0]));
  // Only `model` is returned. Both candidates are computed to make the choice
  // above, but a node runs ONE model: in 'corun' it is loaded alongside the
  // miner, in 'demand' the miner is stopped first and nothing is co-resident.
  // An earlier version also returned the losing candidate as `coRunModel`,
  // which nothing ever loaded and which read as though two models ran at once.
  return bigger
    ? { strategy: 'demand', model: exclusive }
    : { strategy: 'corun', model: shared };
}

module.exports = {
  planAutoMode, allModels, pickModel, ctxLadder, needsMmproj };
