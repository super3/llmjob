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

// What a model costs at one of its ladder rungs, from the per-rung figures the
// config MEASURED. A lookup, deliberately not an extrapolation: the numbers
// exist, so there is no arithmetic here to get wrong and no way for a derived
// price to drift from the sweep it came from.
//
// A rung with no measured figure is priced at the model's FULL cost, which
// refuses it. That is the safe direction and it is the whole reason this returns
// `full` rather than something cheaper: an unmeasured rung that guesses low
// OOMs a node, while one that refuses merely leaves it on the default. Same
// reasoning for a model with no table at all -- the small default has one rung
// and never needs to be priced below it.
function vramAtCtx(model, ctx) {
  const full = Number(model && model.vramFullMb) || 0;
  const table = (model && model.ctxVramMb) || null;
  if (!full || !table || !(ctx > 0)) return full;
  const at = Number(table[ctx]);
  return Number.isFinite(at) && at > 0 ? at : full;
}

// The same model pinned to the highest rung of its ladder that `freeMb` can
// actually host, or null when no offered rung fits.
//
// The pinned copy carries CORRECTED vramFullMb and minVramMb, not merely a
// smaller ctxSize, because llmPlan, vram.computeGpuLayers, the "needs ~N GB"
// refusal in main.js and the ladder handed to llama-server ALL re-derive the
// requirement from those two fields. Pinning them makes every one of those agree
// by construction rather than each independently re-deriving the top rung's cost
// and contradicting the window the node is actually being started at.
function pinToFittingRung(model, freeMb, reserveMb) {
  const rungs = ctxLadder(model);
  // Clamped at zero. Neither field is optional in the shipped config, but a tier
  // that lost one would otherwise yield a NEGATIVE headroom and a floor below
  // the model's own measured cost -- a requirement that fits any card at all,
  // which is the one failure shape this whole module exists to prevent.
  const headroom = Math.max(0, (Number(model.minVramMb) || 0) - (Number(model.vramFullMb) || 0));
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
//
// A tier is admitted at its top rung on any card, and at a SMALLER rung only on
// a card serving alone. Gating that on the mining reserve is the point, and it
// is what the first version of this got wrong: admitting reduced rungs
// everywhere cannot be scoped by a floor on the rung, because any floor low
// enough to offer a 24 GB card 65536 also offers a MINING 32 GB card 131072 --
// which quietly moved every 32 GB GUI node in the fleet off the small default
// and onto a 27B at half its context. The reserve separates the two exactly: a
// co-running card sees precisely the behaviour it saw before, and a card with
// the GPU to itself gets the largest window it can genuinely host.
function pickModel(freeMb, reserveMb = 0, models = null) {
  const list = models || allModels();
  const fallback = list[list.length - 1] || null;
  const free = Number(freeMb);
  if (freeMb == null || !Number.isFinite(free)) return fallback;
  const exclusive = !(Number(reserveMb) > 0);
  for (const m of list) {
    if (!m) continue;
    // Full context fits: hand back the config object itself, unchanged, so a
    // card that was always eligible keeps object identity as well as behaviour.
    if (free >= requiredFreeMb(m, reserveMb)) return m;
    if (!exclusive) continue;
    // It does not, but a smaller window might. Admission asks "does ANY offered
    // rung fit"; configuration then takes the HIGHEST that does. Testing only
    // the top rung refused a 24 GB card this tier outright even though it can
    // serve it at 65536, which also made ctxLadder unreachable on exactly the
    // cards it was written for -- it could only be walked by a card already
    // admitted at full context that then failed to start.
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
//
// Comparing the model KEY is still the whole comparison, and gating reduced
// windows on the mining reserve is what keeps that true. `shared` is always a
// top-rung model, so the two candidates can never be the same tier at different
// windows: if the top rung fits WITH the reserve it also fits without it. The
// only cross-rung case is fallback-versus-pinned-tier, which the keys already
// separate. Admitting reduced rungs while co-running would break that -- a card
// could then resolve the same tier at two different windows and be called a
// co-run, giving up half the context the tier exists to offer.
function planAutoMode(freeMb, reserveMb = 0, models = null) {
  const shared = pickModel(freeMb, reserveMb, models);
  const exclusive = pickModel(freeMb, 0, models);
  const bigger = !!(exclusive && shared && exclusive.key !== shared.key);
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
  planAutoMode, allModels, pickModel, ctxLadder, needsMmproj, vramAtCtx, pinToFittingRung };
