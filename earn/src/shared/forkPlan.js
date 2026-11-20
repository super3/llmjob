'use strict';

// Which coin each card mines, before and after Pearl's FP8 fork (PIP-3).
//
// At the fork Pearl's proof of work changes to one only NVIDIA Hopper and
// Blackwell cards can produce. Every other card stops earning Pearl at the fork
// block. Pearl has set no mainnet height, so nothing here knows one: the fork
// is "active" once a pool sends a job with certificate version 4, which the
// miner reports (PearlMiner 'fork').
//
// The rules, for both shells:
//   - Each card mines the coin picked for it (cardCoins), else the rig's coin.
//   - Before the fork nothing else happens.
//   - Once the fork is active, a card on Pearl that has no FP8 core moves to
//     Quantus when the rig has a Quantus address, and stops when it has none.
//   - Only RTX 50 and RTX PRO Blackwell cards (compute 12.x) can have an FP8
//     core, and only when PEARL_FP8=1 says this build carries one.

const { isBlackwell, fp8Family, listGpus } = require('./cardFamily');

// The certificate version of an FP8 job. Versions up to 3 are today's scheme.
const FP8_CERT_VERSION = 4;

const COINS = ['pearl', 'quantus'];

// What a user whose card will stop is told before the fork.
const FORK_NOTICE = 'Pearl\'s planned FP8 upgrade drops support for this card; it can mine Quantus instead.';

// The FP8 core is built apart from this and is off unless PEARL_FP8=1.
function fp8On(env) {
  return !!env && env.PEARL_FP8 === '1';
}

// A job of the FP8 scheme, by the version the pool states.
function isFp8Job(certVersion) {
  return Number.isFinite(certVersion) && certVersion >= FP8_CERT_VERSION;
}

// The per-card choices as saved or typed: { "<index>": coin }. Anything that is
// not a card index with a coin we know is dropped.
function cleanCardCoins(raw) {
  const out = {};
  if (!raw || typeof raw !== 'object') return out;
  for (const [k, v] of Object.entries(raw)) {
    if (/^\d+$/.test(k) && COINS.includes(v)) out[Number(k)] = v;
  }
  return out;
}

// --card-coin's value, "0:pearl,1:quantus": { map } or { error }.
function parseCardCoins(text) {
  const map = {};
  for (const part of String(text == null ? '' : text).split(',').map((p) => p.trim()).filter(Boolean)) {
    const m = /^(\d+)\s*[:=]\s*([a-z]+)$/i.exec(part);
    const coin = m ? m[2].toLowerCase() : null;
    if (!COINS.includes(coin)) return { error: 'expected <card>:<coin> pairs such as 0:pearl,1:quantus, got "' + part + '"' };
    map[Number(m[1])] = coin;
  }
  if (!Object.keys(map).length) return { error: 'expected <card>:<coin> pairs such as 0:pearl,1:quantus' };
  return { map };
}

// Whether this card can mine Pearl once the fork is active: an RTX 50 or RTX
// PRO Blackwell card, on a build with the FP8 core.
function minesFp8(card, fp8) {
  return !!fp8 && isBlackwell(card);
}

// Whether we know Pearl's FP8 scheme has no place for this card. An AMD or
// Intel rig never has; an NVIDIA card has not when nvidia-smi gave a compute
// capability outside the FP8 families. A card nothing is known about is not
// called dropped: the notice must not go to an RTX 50 nvidia-smi failed to list.
function forkDrops(card, vendor) {
  if (vendor === 'amd' || vendor === 'intel') return true;
  return !!card && !fp8Family(card);
}

// The coin plan for one start.
//   indices     the cards that mine ([] when the rig's cards are not known,
//               which is planned as one card, index null)
//   cards       nvidia-smi's cards with their compute capability
//   coin        the rig's coin, 'pearl' unless Quantus was picked
//   cardCoins   the per-card choices (cleanCardCoins)
//   pearlOk, quantusOk   whether there is a valid address for that coin
//   forkActive  a pool has sent an FP8 job in this process
//   fp8         fp8On(env)
// Returns { whole, pearl, quantus, moved, stopped }: card indices mining each
// coin, the ones the fork moved from Pearl to Quantus (a subset of `quantus`),
// and the ones that mine nothing, as { index, why }: 'fork' (the fork ended
// Pearl for it and there is no Quantus address), 'no-quantus-address' or
// 'no-pearl-address' (its own coin has no address). `whole` marks a rig planned
// as one unknown card.
function planCoins({ indices, cards, coin, cardCoins, pearlOk, quantusOk, forkActive, fp8 }) {
  const known = Array.isArray(indices) && indices.length > 0;
  const choices = cleanCardCoins(cardCoins);
  const byIndex = new Map((Array.isArray(cards) ? cards : []).map((c) => [c.index, c]));
  const plan = { whole: !known, pearl: [], quantus: [], moved: [], stopped: [] };
  for (const index of known ? indices : [null]) {
    const want = choices[index] || (coin === 'quantus' ? 'quantus' : 'pearl');
    if (want === 'quantus') {
      if (quantusOk) plan.quantus.push(index);
      else plan.stopped.push({ index, why: 'no-quantus-address' });
    } else if (!pearlOk) {
      plan.stopped.push({ index, why: 'no-pearl-address' });
    } else if (!forkActive || minesFp8(byIndex.get(index), fp8)) {
      plan.pearl.push(index);
    } else if (quantusOk) {
      plan.quantus.push(index);
      plan.moved.push(index);
    } else {
      plan.stopped.push({ index, why: 'fork' });
    }
  }
  return plan;
}

// Whether a plan is today's: every card on the rig's own coin, none moved and
// none stopped. The shells then start exactly as they did before any of this.
function planIsPlain(plan, coin) {
  if (plan.stopped.length || plan.moved.length) return false;
  return coin === 'quantus' ? !plan.pearl.length : !plan.quantus.length;
}

// What a plan mines, for the board, the stats file and the app's reading:
//   coin          'quantus' when no card mines Pearl and one mines Quantus,
//                 else 'pearl'
//   quantusCards  on a rig that mines both, the cards on Quantus; else null
function planCoin(plan) {
  if (!plan.pearl.length) return { coin: plan.quantus.length ? 'quantus' : 'pearl', quantusCards: null };
  return { coin: 'pearl', quantusCards: plan.quantus.length ? plan.quantus.slice() : null };
}

// Whether a START can mine at all, asked before the cards are known: some card
// may want a coin there is an address for. After the fork a rig with no FP8
// build mines Quantus or nothing.
function canMine({ coin, cardCoins, pearlOk, quantusOk, forkActive, fp8 }) {
  const picked = Object.values(cleanCardCoins(cardCoins));
  const rig = coin === 'quantus' ? 'quantus' : 'pearl';
  const wants = (c) => rig === c || picked.includes(c);
  if (quantusOk && (wants('quantus') || forkActive)) return true;
  return !!pearlOk && wants('pearl') && (!forkActive || !!fp8);
}

// Whether the GPUs mine Quantus alone as far as can be told before the cards
// are known: Quantus is the rig's coin, or the fork has ended Pearl for this
// build. Such a START runs no local LLM beside the miner, as --coin quantus
// never has.
function quantusOnly({ coin, forkActive, fp8 }) {
  return coin === 'quantus' || (!!forkActive && !fp8);
}

// "GPU 0, 1" for a list of cards, or `whole` when the cards are not known.
function cardsText(list, whole) {
  return list.length && list[0] != null ? listGpus(list) : whole;
}

const STOP_TEXT = {
  fork: ' cannot mine Pearl since its FP8 upgrade, and there is no Quantus address to mine to instead.'
    + ' Add one and start again.',
  'no-quantus-address': ' is set to mine Quantus, but there is no Quantus address: not mining.',
  'no-pearl-address': ' is set to mine Pearl, but there is no Pearl address: not mining.',
};

// What a plan did that the user did not ask for, as log lines.
function planLines(plan) {
  const lines = [];
  if (plan.moved.length) {
    lines.push({
      level: 'warn',
      line: 'Pearl has switched to FP8, which ' + cardsText(plan.moved, 'this rig') + ' cannot mine. Mining Quantus instead.',
    });
  }
  for (const why of Object.keys(STOP_TEXT)) {
    const list = plan.stopped.filter((s) => s.why === why).map((s) => s.index);
    if (list.length) lines.push({ level: 'error', line: cardsText(list, 'This rig') + STOP_TEXT[why] });
  }
  return lines;
}

// The cards to tell, before the fork, that it ends Pearl for them: the ones
// mining Pearl now that Pearl's FP8 scheme has no place for. None once the fork
// is active (planLines speaks then), and none with LLMJOB_FORK_NOTICE=0.
function noticeCards({ plan, cards, vendor, forkActive, env }) {
  if (forkActive || (env && env.LLMJOB_FORK_NOTICE === '0')) return [];
  const byIndex = new Map((Array.isArray(cards) ? cards : []).map((c) => [c.index, c]));
  return plan.pearl.filter((index) => forkDrops(byIndex.get(index), vendor));
}

// The notice as one line, or null when it goes to no card. `quantusOk` false
// adds what to do about it.
function noticeLine(list, quantusOk) {
  if (!list.length) return null;
  const who = list[0] != null ? listGpus(list) + ': ' : '';
  return who + FORK_NOTICE + (quantusOk ? '' : ' Add a Quantus address and it switches by itself.');
}

module.exports = {
  FP8_CERT_VERSION, COINS, FORK_NOTICE, fp8On, isFp8Job, cleanCardCoins, parseCardCoins, minesFp8, forkDrops,
  planCoins, planIsPlain, planCoin, canMine, quantusOnly, planLines, noticeCards, noticeLine,
};
