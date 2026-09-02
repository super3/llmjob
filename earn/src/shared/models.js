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

// What a model costs at a given context. vramFullMb is measured at the model's
// top rung; only the KV cache moves with context, and mbPerCtxToken prices it.
// A model without that figure cannot be priced below its top rung, so it keeps
// its measured cost and is only ever offered at full context.
function vramAtCtx(model, ctx) {
  const full = Number(model && model.vramFullMb) || 0;
  const per = Number(model && model.mbPerCtxToken) || 0;
  const top = Number(model && model.ctxSize) || 0;
  // `!full` matters as well as the rest: a model with no measured cost cannot be
  // priced at a smaller window either, and subtracting a rung's worth of cache
  // from zero would hand back a NEGATIVE requirement that fits any card.
  if (!full || !per || !top || !(ctx > 0) || ctx >= top) return full;
  return Math.ceil(full - (top - ctx) * per);
}

// The same model pinned to the highest rung of its ladder that `freeMb` can
// actually host, or null if even the lowest rung will not fit.
//
// The pinned copy carries CORRECTED vramFullMb/minVramMb, not just a smaller
// ctxSize, because llmPlan, computeGpuLayers, the refusal message and the ladder
// handed to llama-server all re-derive the requirement from those two fields.
// Pinning them makes every one of those agree by construction instead of each
// re-deriving the top rung's cost.
function pinToFittingRung(model, freeMb, reserveMb) {
  const rungs = ctxLadder(model);
  const headroom = (Number(model.minVramMb) || 0) - (Number(model.vramFullMb) || 0);
  const floor = Number(model.minOfferCtx) || 0;
  for (let i = 1; i < rungs.length; i++) {   // 0 is the top rung, already tested
    if (rungs[i] < floor) break;             // below this the tier is not worth offering
    const full = vramAtCtx(model, rungs[i]);
    const pinned = Object.assign({}, model, {
      ctxSize: rungs[i],
      ctxLadder: rungs.slice(i),
      vramFullMb: full,
      minVramMb: full + headroom,
    });
    if (freeMb >= requiredFreeMb(pinned, reserveMb)) return pinned;
  }
  return null;
}

// The best model `freeMb` can host, or the fallback default when none of the
// larger tiers fit. Returns the same object shape everything downstream already
// expects from LLM.model, so callers need no special-casing.
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
    // Full context fits: hand back the config object itself, unchanged.
    if (free >= requiredFreeMb(m, reserveMb)) return m;
    // It does not, but a smaller window might. Admission asks "does ANY rung
    // fit"; configuration takes the HIGHEST that does. Gating on the top rung
    // alone refused a 24 GB card the tier outright even though it can serve it
    // at 65536, which made ctxLadder dead code on exactly those cards.
    const pinned = pinToFittingRung(m, free, reserveMb);
    if (pinned) return pinned;
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
// So compare the two choices. If the card could serve a bigger model with the GPU
// to itself than it can while mining, auto becomes demand-driven -- mine until
// something asks for tokens, then switch. If they are the same model, nothing
// changes and the node co-runs exactly as before, which is what keeps small cards
// undisturbed.
function planAutoMode(freeMb, reserveMb = 0, models = null) {
  const shared = pickModel(freeMb, reserveMb, models);
  const exclusive = pickModel(freeMb, 0, models);
  // "Better with the card to itself" is not only a bigger MODEL: since the gate
  // now resolves a context rung too, it can be the same model at a bigger window.
  // A mining 5090 resolves Qwen at 131072 while co-running and 262144 alone, and
  // treating those as equivalent would quietly give up half the context the tier
  // exists to offer.
  const bigger = !!(exclusive && shared
    && (exclusive.key !== shared.key
        || (Number(exclusive.ctxSize) || 0) > (Number(shared.ctxSize) || 0)));
  return bigger
    ? { strategy: 'demand', model: exclusive, coRunModel: shared }
    : { strategy: 'corun', model: shared, coRunModel: shared };
}

module.exports = {
  planAutoMode, allModels, pickModel, ctxLadder, needsMmproj };
